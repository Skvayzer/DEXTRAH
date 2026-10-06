"""Carry-expert pieces with no simulator imports: walking references, frames, reward.

The carry expert (Phase 1 of the MoE plan) holds an already-grasped object while
frozen SONIC follows walking references. There is no object-pose goal. Rewards
favour retention, low hand-object slip, walking-reference tracking and moderate
grip force; robot falls and drops terminate the episode.
"""
from dataclasses import dataclass
import numpy as np
import torch

CLIP_HZ = 50
FUTURE_FRAMES = 10
FUTURE_SPACING_S = .1


# Quaternions are wxyz unless a name says xyzw.
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_from_yaw(yaw):
    half = yaw/2
    zero = torch.zeros_like(yaw)
    return torch.stack((half.cos(), zero, zero, half.sin()), -1)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rotate_z(v, angle):
    """Rotate (..., 3) vectors about +Z by angle broadcast over leading dims."""
    c, s = angle.cos(), angle.sin()
    while c.dim() < v.dim()-1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    x, y, z = v.unbind(-1)
    return torch.stack((c*x-s*y, s*x+c*y, z), -1)


def nlerp(a, b, w):
    b = torch.where(((a*b).sum(-1, keepdim=True) < 0), -b, b)
    q = a+(b-a)*w
    return q/q.norm(dim=-1, keepdim=True)


@dataclass
class MovingFrame:
    """Express env-frame quantities as if the robot were still at its reset pose.

    At the default standing pose this is the identity, so a policy initialised
    from the standing reposing expert initially sees exactly what it saw before.
    While walking, palm/object terms stay body-relative instead of drifting with
    the robot's displacement from the environment origin. Rotation is yaw only.
    """
    root_pos: torch.Tensor      # (N, 3) current pelvis, env-local
    root_yaw: torch.Tensor      # (N,)
    home_pos: torch.Tensor      # (N, 3) default pelvis, env-local
    home_yaw: torch.Tensor      # (N,)

    @property
    def angle(self):
        return self.home_yaw-self.root_yaw

    def position(self, x):
        extra = x.dim()-2
        root, home = self.root_pos, self.home_pos
        for _ in range(extra):
            root, home = root.unsqueeze(1), home.unsqueeze(1)
        return home+rotate_z(x-root, self.angle)

    def vector(self, v):
        return rotate_z(v, self.angle)

    def quat_wxyz(self, q):
        return quat_mul(quat_from_yaw(self.angle), q)

    def quat_xyzw(self, q):
        wxyz = torch.cat((q[..., 3:], q[..., :3]), -1)
        out = self.quat_wxyz(wxyz)
        return torch.cat((out[..., 1:], out[..., :1]), -1)


def transform_clean_observation(obs, context, frame):
    """Apply MovingFrame to the source's clean observation dict and context.

    Keys follow isaacsimenvs ``_build_clean_observation_dict``. Relative vectors
    are rotated; absolute env-frame positions are re-anchored; policy quaternions
    are xyzw and context quaternions are wxyz, as in the source.
    """
    obs, context = dict(obs), dict(context)
    obs['palm_pos'] = frame.position(obs['palm_pos'])
    obs['palm_rot'] = frame.quat_xyzw(obs['palm_rot'])
    obs['object_rot'] = frame.quat_xyzw(obs['object_rot'])
    for key in ('palm_vel', 'object_vel'):
        v = obs[key]
        obs[key] = torch.cat((frame.vector(v[:, :3]), frame.vector(v[:, 3:])), -1)
    for key in ('fingertip_pos_rel_palm', 'keypoints_rel_palm', 'keypoints_rel_goal'):
        shape = obs[key].shape
        obs[key] = frame.vector(obs[key].reshape(shape[0], -1, 3)).reshape(shape)
    for key in ('palm_pos', 'obj_pos', 'goal_pos'):
        context[key] = frame.position(context[key])
    for key in ('obj_rot', 'goal_rot'):
        context[key] = frame.quat_wxyz(context[key])
    for key in ('obj_linvel', 'obj_angvel'):
        context[key] = frame.vector(context[key])
    context['obj_vel'] = torch.cat((context['obj_linvel'], context['obj_angvel']), -1)
    return obs, context


def field_slice(field_list, sizes, name):
    start = 0
    for field in field_list:
        if field == name:
            return slice(start, start+sizes[field])
        start += sizes[field]
    raise KeyError(name)


class WalkingClips:
    """GPU sampler for precomputed released-planner walking references.

    Clips are 50 Hz body-joint references (Isaac order) plus planned root
    orientation and planned body-frame velocity. Each environment plays one clip
    anchored to the robot's measured yaw at the clip start; past the end it
    holds the last (standing) frame. Arm joints are replaced by the trained
    standing arm reference by the caller, not here.
    """
    def __init__(self, q, root_quat, velocity, command, lengths, names, device='cpu'):
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)            # (C, T, 29)
        self.root_quat = torch.as_tensor(root_quat, dtype=torch.float32, device=device)  # (C, T, 4)
        self.velocity = torch.as_tensor(velocity, dtype=torch.float32, device=device)    # (C, T, 3)
        self.command = torch.as_tensor(command, dtype=torch.float32, device=device)      # (C, T, 3)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        self.names = list(names)
        if self.q.shape[:2] != self.root_quat.shape[:2] or self.q.shape[-1] != 29:
            raise ValueError('Inconsistent walking clip arrays')
        if (self.lengths < 2).any() or (self.lengths > self.q.shape[1]).any():
            raise ValueError('Invalid clip lengths')
        # Anchor each clip at zero initial yaw so a measured robot yaw is the offset.
        start = yaw_of(self.root_quat[:, 0])
        self.root_quat = quat_mul(quat_from_yaw(-start)[:, None].expand_as(self.root_quat), self.root_quat)

    @classmethod
    def load(cls, path, device='cpu'):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['q'], data['root_quat'], data['velocity'], data['command'],
                       data['lengths'], [str(x) for x in data['names']], device)

    def index(self, name):
        return self.names.index(name)

    def duration(self, clip):
        return (self.lengths[clip]-1).float()/CLIP_HZ

    def _interp(self, values, clip, tau):
        """Linear interpolation on (N, K) clip times, holding the end frames."""
        last = (self.lengths[clip]-1)[:, None]
        f = torch.minimum((tau*CLIP_HZ).clamp(min=0), last.float())
        lo = f.floor().long()
        hi = torch.minimum(lo+1, last)
        c = clip[:, None].expand_as(lo)
        return values[c, lo], values[c, hi], (f-lo.float()).unsqueeze(-1)

    def reference(self, clip, elapsed, heading):
        """Return q, qd (N,10,29), world root quats (N,10,4), planned v and command (N,3)."""
        tau = elapsed[:, None]+torch.arange(FUTURE_FRAMES, device=elapsed.device)*FUTURE_SPACING_S
        a, b, w = self._interp(self.q, clip, tau)
        q = a+(b-a)*w
        a2, b2, w2 = self._interp(self.q, clip, tau+1/CLIP_HZ)
        qd = (a2+(b2-a2)*w2-q)*CLIP_HZ
        qa, qb, qw = self._interp(self.root_quat, clip, tau)
        root = nlerp(qa, qb, qw)
        root = quat_mul(quat_from_yaw(heading)[:, None].expand_as(root), root)
        va, vb, vw = self._interp(self.velocity, clip, elapsed[:, None])
        ca, cb, cw = self._interp(self.command, clip, elapsed[:, None])
        return q, qd, root, (va+(vb-va)*vw)[:, 0], torch.where(cw < .5, ca, cb)[:, 0]


def grasp_gate(object_lift_m, normal_n, rel_speed, *, lift=.05, force=.2, speed=.2):
    """Instantaneous 'secure grasp' test; callers require it for 0.5 s.

    object_lift_m: object height above its resting reset height.
    normal_n: (N, 5) fingertip normal forces, thumb first.
    rel_speed: object linear speed relative to the palm point it would follow.

    The reposing expert mostly holds objects against the (unsensed) palm with
    index/middle fingers; its thumb touches in only ~25-30% of lifted frames
    (final-best recording, 2026-10-06). So any fingertip contact counts.
    """
    return (object_lift_m > lift) & ((normal_n > force).any(-1)) & (rel_speed < speed)


@dataclass
class CarryRewardCfg:
    hold: float = 1.
    hold_margin_m: float = .05
    contact_force_n: float = .2
    slip_linear: float = .5
    slip_angular: float = .05
    track_linear: float = .5
    track_yaw: float = .25
    track_sigma: float = .05
    force_limit_n: float = 15.
    force_penalty: float = .01
    latent_rate: float = .01
    finger_rate: float = .05
    fall_penalty: float = 10.
    drop_penalty: float = 10.


def carry_reward(cfg, *, distance, home_distance, normal_n, rel_linear, rel_angular,
                 body_velocity, planned_velocity, latent_delta, finger_delta, fell, dropped):
    """Return total reward and named terms. Shapes are (N,) unless noted.

    distance/home_distance: object-palm distance now / at the restored grasp.
    normal_n (N, 5); rel_linear/rel_angular (N, 3); body_velocity and
    planned_velocity are body-frame (vx, vy, wz).
    """
    near = (distance < home_distance+cfg.hold_margin_m).float()
    contact_ok = (normal_n > cfg.contact_force_n).any(-1).float()
    terms = dict(
        hold=cfg.hold*near*(.5+.5*contact_ok),
        slip=-cfg.slip_linear*rel_linear.norm(dim=-1).clamp(max=1.)
             - cfg.slip_angular*rel_angular.norm(dim=-1).clamp(max=10.),
        track=cfg.track_linear*torch.exp(-((body_velocity[:, :2]-planned_velocity[:, :2])**2).sum(-1)/cfg.track_sigma)
              + cfg.track_yaw*torch.exp(-(body_velocity[:, 2]-planned_velocity[:, 2])**2/cfg.track_sigma),
        force=-cfg.force_penalty*(normal_n-cfg.force_limit_n).clamp(min=0).sum(-1),
        smooth=-cfg.latent_rate*(latent_delta**2).mean(-1)-cfg.finger_rate*(finger_delta**2).mean(-1),
        fall=-cfg.fall_penalty*fell.float(),
        drop=-cfg.drop_penalty*dropped.float(),
    )
    return sum(terms.values()), terms
