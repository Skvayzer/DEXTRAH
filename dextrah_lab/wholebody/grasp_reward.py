"""Contact-prioritized grasp reward with thumb-finger opposition (opt-in:
--grasp-reward-scale), after InterMimicGen (arXiv:2610.06850, App. C.2, Eqs. 10-11),
and its object-energy regularizer (App. C.1; opt-in: --object-energy-scale).

Per distal pad f (thumb h, index, middle, ring, pinky) with pad centre p_f and the
nearest object-surface point u_f:
    b_f = p_f - u_f,  gap D_f = max(|b_f| - r_f, 0)
    P_f = exp(-(D_f / sigma)^2)                 proximity, sigma = 5 cm
    C_f = S_f * clip(1 - D_f / d_c, 0, 1)       contact proxy, d_c = 2.5 cm
    O_f = clip((1 - b_h . b_f) / 2, 0, 1)       opposition (1: finger opposite the thumb)
    A = max_f sqrt(P_h P_f), E = max_f sqrt(P_h P_f) O_f, C_opp = C_h max_f C_f O_f,
    C_any = max_pads C_f, C_palm = S_palm
    G = 0.05 A + 0.10 E + 0.65 C_opp + 0.15 C_any + 0.05 C_palm   in [0, 1]
reward += scale * G every policy step (the paper multiplies a bounded factor into its
tracking reward; this task's reward is additive).

Our choices where the paper is generic or does not apply:
* p_f is the Revo2 touch-pad centre: the pad's surface point on the G1 distal collision
  mesh moved inward by half the finger thickness along the pad normal; r_f is that half
  thickness (paper: "collision-mesh-derived effective radius").
* S_f / S_palm: summed |force components| > 0.1 N of contacts between the distal link /
  the hand base link and the OBJECT only (the paper's threshold counts any partner and
  relies on proximity to filter; the per-partner contact pool lets us filter exactly).
* Active at every step (the paper gates by reference-active hand phases; this task always
  manipulates the object, and the contact terms vanish without contact).
* Object surface from the per-world primitives (boxes/capsules), as the hand-surface
  observation; true object pose (reward, not observation).

Object energy (paper: "penalizes the simulated linear and angular acceleration of the
object, obtained by differencing velocities at the control interval"; form not given):
    reward -= scale * (1 - exp(-(|a| / a0)^2 - (|alpha| / alpha0)^2)),  a0 = 20 m/s^2,
    alpha0 = 200 rad/s^2: bounded in [-scale, 0]; zero on the first step after a reset.
"""
import torch

from dextrah_lab.g1_adept.contact import FINGERS
from dextrah_lab.wholebody.hand_surface import nearest_surface_vectors, quat_to_mat

SIGMA, D_CONTACT, FORCE_THRESHOLD = .05, .025, .1
WEIGHTS = dict(A=.05, E=.10, C_opp=.65, C_any=.15, C_palm=.05)
THUMB = FINGERS.index('thumb')


def pad_centres(pad_geometry, device):
    """(5, 3) pad centres in distal-link coordinates and (5,) effective radii r_f."""
    centres, radii = [], []
    for p in pad_geometry:
        half = .5*float((p.surface_point-p.back_surface_point) @ p.outward)
        centres.append(torch.as_tensor(p.surface_point-half*p.outward, dtype=torch.float32))
        radii.append(half)
    return torch.stack(centres).to(device), torch.tensor(radii, dtype=torch.float32, device=device)


def grasp_score(b, radius, finger_force, palm_force):
    """b (n, 5, 3): pad centre minus nearest object-surface point (FINGERS order);
    radius (5,); finger_force (n, 5), palm_force (n,): object-contact |force| sums.
    Returns G (n,) and the components."""
    dist = b.norm(dim=-1)
    gap = (dist-radius).clamp_min(0.)
    P = torch.exp(-(gap/SIGMA)**2)
    C = (finger_force > FORCE_THRESHOLD).float()*(1-gap/D_CONTACT).clamp(0., 1.)
    unit = b/dist.clamp_min(1e-9)[..., None]
    others = [i for i in range(5) if i != THUMB]
    O = ((1-(unit[:, THUMB:THUMB+1]*unit[:, others]).sum(-1))/2).clamp(0., 1.)
    pair = torch.sqrt(P[:, THUMB:THUMB+1]*P[:, others])
    parts = dict(A=pair.amax(-1), E=(pair*O).amax(-1), C_opp=C[:, THUMB]*(C[:, others]*O).amax(-1),
                 C_any=C.amax(-1), C_palm=(palm_force > FORCE_THRESHOLD).float())
    G = sum(WEIGHTS[k]*v for k, v in parts.items())
    return G, parts


class GraspReward:
    """Grasp score (and optional object energy) of the mjlab G1 + Revo2 env."""

    def __init__(self, env, scale, energy_scale=0., energy_a0=20., energy_alpha0=200.):
        from dextrah_lab.mj.scene import object_indices
        from dextrah_lab.wholebody.hand_surface import MJ_GEOM_BOX, MJ_GEOM_CAPSULE
        self.env, self.scale, self.energy_scale = env, float(scale), float(energy_scale)
        self.a0, self.alpha0 = float(energy_a0), float(energy_alpha0)
        device = env.device
        self.centre, self.radius = pad_centres(env.pad_geometry, device)
        bank = env.object_bank
        t = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
        self.pos, self.size, self.rot = t(bank.pos), t(bank.size), quat_to_mat(t(bank.quat))
        self.used = torch.as_tensor(bank.used, device=device, dtype=torch.bool)
        _, slots = object_indices(env.sim.mj_model)
        types = [int(env.sim.mj_model.geom_type[g]) for g in slots]
        if any(x not in (MJ_GEOM_BOX, MJ_GEOM_CAPSULE) for x in types):
            raise ValueError(f'Object slots must be boxes or capsules, got types {types}')
        self.is_box = torch.tensor([x == MJ_GEOM_BOX for x in types], device=device)
        self.prev_vel = None

    def pad_points(self):
        """(n, 5, 3) env-local pad centres (FINGERS order) from the distal link poses."""
        env = self.env
        bodies = env.contacts.finger_bodies
        pos, quat = env.sim.data.xpos[:, bodies], env.sim.data.xquat[:, bodies]
        return pos+torch.einsum('nfij,fj->nfi', quat_to_mat(quat), self.centre)-env.scene.env_origins[:, None]

    def surface_offsets(self):
        """b_f = p_f - u_f (n, 5, 3) for the true object pose."""
        env = self.env
        a = env._object_asset_index_per_env
        obj_pos = env.object.data.root_pos_w-env.scene.env_origins
        R = quat_to_mat(env.object.data.root_quat_w)
        centre = obj_pos[:, None]+torch.einsum('nij,nsj->nsi', R, self.pos[a])
        rot = R[:, None] @ self.rot[a]
        return -nearest_surface_vectors(self.pad_points(), centre, rot, self.size[a], self.is_box, self.used[a])

    @torch.no_grad()
    def __call__(self):
        """(reward (n,), {name: (n,) per-env values for logging})."""
        env = self.env
        c = env._last_contacts
        b = self.surface_offsets()
        force = c['finger_object_force_l1']
        G, parts = grasp_score(b, self.radius, force, c['palm_object_force_l1'])
        reward = self.scale*G
        terms = dict(grasp_score_rew=self.scale*G)
        logs = dict(score=G, **parts)
        # Diagnostics (while lifted): object contact of the thumb, pad gaps at contact, and the
        # touch-pad pinch (thumb + another pad > 0.25 N) for comparison with C_opp.
        lifted = env._lifted_object.float()
        mean = lambda v, m: (v*m).sum()/m.sum().clamp_min(1.)
        touching = (force > 0.1).float()
        gap = ((b.norm(dim=-1)-self.radius).clamp_min(0.))*100.
        pad = env.touch_force_w.norm(dim=-1) > .25
        others = [i for i in range(5) if i != THUMB]
        logs['thumb_object_contact_lifted'] = mean(touching[:, THUMB], lifted)
        logs['any_finger_object_contact_lifted'] = mean(touching[:, others].amax(-1), lifted)
        logs['thumb_gap_cm_at_contact'] = mean(gap[:, THUMB], touching[:, THUMB])
        logs['finger_gap_cm_at_contact'] = (gap[:, others]*touching[:, others]).sum()/touching[:, others].sum().clamp_min(1.)
        logs['pad_pinch_lifted'] = mean((pad[:, THUMB] & pad[:, others].any(-1)).float(), lifted)
        if self.energy_scale > 0:
            vel = env.object.data.root_state_w[:, 7:13].clone()
            if self.prev_vel is None:
                self.prev_vel = vel.clone()
            acc = (vel-self.prev_vel)/env.step_dt
            fresh = env.episode_length_buf <= 1
            energy = 1-torch.exp(-(acc[:, :3].norm(dim=-1)/self.a0)**2-(acc[:, 3:].norm(dim=-1)/self.alpha0)**2)
            energy = torch.where(fresh, torch.zeros_like(energy), energy)
            self.prev_vel = vel
            reward = reward-self.energy_scale*energy
            terms['object_energy_penalty'] = -self.energy_scale*energy
            logs['object_energy'] = energy
        return reward, terms, logs
