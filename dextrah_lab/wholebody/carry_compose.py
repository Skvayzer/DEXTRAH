"""Carry expert C on the mjlab whole body, after DexCompose (arXiv:2606.28323): simulator-free parts.

DexCompose composes a frozen object-retention skill (Task A) with a downstream skill (Task B)
by (i) a bounded Task-A residual stabilizer around the stored hold reference,
q = q_ref + beta_A tanh(pi_A(o)) (Eqs. 6-7), (ii) a bounded Task-B residual on the frozen
downstream policy, a = a_B + M_B beta_B tanh(pi_B(o, a_B)) (Eqs. 10-11), (iii) hard action
ownership a = M_A a_A + M_B a_B (Eq. 12), both residual heads starting at zero (mean head 0,
log-std -1.5, App. B.1). Here Task A = keep the grasped object, Task B = walk with frozen SONIC
following released-planner references:

* fingers   = clamp(q_ref + beta_A tanh(d_A), -1, 1)    (q_ref: the holding state's hand targets in
              the task's absolute finger-command units)
* latent    = beta_B tanh(d_B), zero in stage 1          (SONIC pre-FSQ residual, x0.1 as before)
* body      = legs/waist from SONIC with zero residual (bare SONIC, which walks), arms from SONIC
              with the residual: the residual cannot move the legs (ownership mask).
* SONIC reference: planner legs/waist + the holding state's arm joint angles (the arm analogue of
  q_ref), so "residual = 0" means "walk while keeping the grasp posture".

Reward (per policy step; zero during the start warm-up):
  hold  = w_hold exp(-(|p_obj^palm - p_ref^palm| / sigma_hold)^2)   (DexCompose: -|p_obj - p_palm|;
          a bounded positive kernel here because drops end the episode, and an all-negative
          reward would make ending it attractive)
  track = w_lin exp(-|v_xy - v_plan_xy|^2 / s) + w_yaw exp(-(w_z - w_plan)^2 / s)   (planned velocity
          of the clip, not its command: the planner deviates from commands, e.g. 0.30 m/s for 0.4)
  slip  = -k_lin |v_obj - v_palm point| - k_ang |w_obj - w_palm|
  resid = -alpha (|d_A|^2 + |d_B|^2) on the bounded residuals (DexCompose alpha = 0.001)
  force = -k_force sum_f max(F_f - F_max, 0)
  drop / fall = -penalty on the terminating step.
"""
from dataclasses import asdict, dataclass, field
import torch

from .carry import CLIP_HZ, WalkingClips  # noqa: F401  (re-exported for the env)

LEVELS = (0., .1, .2, .3, .4)       # speed curriculum: standing only, then clips up to this speed


@dataclass
class CarrySettings:
    stage: int = 1                      # 1: finger stabilizer only (latent residual 0); 2: + arm residual
    episode_s: float = 20.
    warmup_s: float = .5                # no reward: the recurrent state warms up from zero
    confirm_lift_m: float = .03         # grasp confirmed at the end of the warm-up: object above the table
    confirm_distance_m: float = .05     # ... and within this of its palm-frame hold position
    table_keep_probability: float = .2  # else the table is parked below the floor after the warm-up
    settle_s: float = 1.                # standing after the table is removed, before walking
    final_stand_s: float = 2.           # standing at the end (hand-back states)
    cut_probability: float = .5         # a clip is cut at a random time (sudden command change)
    blend_s: float = .2
    beta_fingers: float = .3            # finger residual bound, command units ([-1, 1] = joint range)
    beta_latent: float = 2.             # latent residual bound (x0.1 before FSQ)
    arm_reference: str = 'grasp'        # 'grasp': holding-state arm angles; 'standing': trained standing arms
    residual_joints: str = 'arms'       # body joints driven by the residual decode (rest: bare SONIC)
    drop_excess_m: float = .12          # drop: palm-frame object offset beyond the hold by this ...
    drop_steps: int = 6                 # ... for this many policy steps, or object below drop_floor_m
    drop_floor_m: float = .25
    retain_m: float = .05               # success: retained within this at the end (DexCompose d_ret)
    mass_scale: tuple = (.7, 2.)        # object mass / inertia scale per episode (log-uniform) ...
    mass_max_kg: float = .6             # ... capped at this mass (job 1551: a 0.89 kg hammer (2.4 x 0.37 kg) made the
                                        # MuJoCo Warp float32 solver non-finite in 256/256 repeats, 0.71 kg and less in 0/256)
    com_offset_m: float = .02           # object COM shift per axis (uniform)
    push_probability: float = 1/120.    # per policy step: a push to the right wrist starts
    push_force_n: float = 15.
    push_s: tuple = (.1, .2)
    promote_success: float = .75        # curriculum: next level at this rolling training success
    promote_window: int = 4096
    promote_min_episodes: int = 0       # episodes started at the level before promotion (0: 2 x num_envs)
    level: int = 0
    w_hold: float = 1.
    sigma_hold_m: float = .03
    w_track_lin: float = .5
    w_track_yaw: float = .25
    track_sigma: float = .05
    k_slip_lin: float = .5
    k_slip_ang: float = .05
    alpha_residual: float = .001
    k_force: float = .01
    force_max_n: float = 15.
    drop_penalty: float = 10.
    fall_penalty: float = 10.
    table_contact_penalty: float = .5

    def __post_init__(self):
        if self.stage not in (1, 2):
            raise ValueError('Carry stage must be 1 or 2')
        if self.arm_reference not in ('grasp', 'standing') or self.residual_joints not in ('arms', 'arms_waist'):
            raise ValueError('Unknown carry arm reference / residual joints')
        if not 0 <= self.level < len(LEVELS):
            raise ValueError('Curriculum level out of range')

    def as_dict(self):
        return asdict(self)


def hand_command(targets, lower, upper):
    """Absolute finger command ([-1, 1] over the joint range) whose raw target is ``targets``
    (inverse of the source pipeline's hand mapping)."""
    return (2*(targets-lower)/(upper-lower)-1).clamp(-1, 1)


def finger_command(q_ref, raw, beta):
    """DexCompose Task-A stabilizer: the stored hold plus a bounded correction."""
    return (q_ref+beta*torch.tanh(raw)).clamp(-1, 1)


def body_joint_masks(names, residual_joints='arms'):
    """Bool mask over BODY_JOINTS: True where the residual decode drives the joint."""
    arms = ('shoulder', 'elbow', 'wrist')
    parts = arms+(('waist',) if residual_joints == 'arms_waist' else ())
    return torch.tensor([any(p in n for p in parts) for n in names])


def compose_body(bare, residual, residual_mask):
    """Ownership (DexCompose Eq. 12): bare SONIC on the walking joints, the residual decode elsewhere."""
    return torch.where(residual_mask.to(bare.device)[None], residual, bare)


def clips_up_to(clips, level):
    """Clip ids allowed at a curriculum level (the idle clip at level 0)."""
    top = LEVELS[level]
    out = []
    for i, name in enumerate(clips.names):
        speed = float(name.split('@')[1]) if '@' in name else 0.
        if name == 'idle' or (0 < speed <= top+1e-6):
            out.append(i)
    if level > 0:
        out = [i for i in out if clips.names[i] != 'idle']
    return out


def carry_reward_v2(s, *, offset_error, rel_linear, rel_angular, body_velocity, planned_velocity,
                    finger_residual, latent_residual, normal_n, dropped, fell, table_contact, active):
    """Per-step reward and named terms (shapes (N,)); ``active`` False -> all terms 0 (warm-up)."""
    a = active.float()
    terms = dict(
        hold=s.w_hold*torch.exp(-(offset_error/s.sigma_hold_m)**2),
        track=s.w_track_lin*torch.exp(-((body_velocity[:, :2]-planned_velocity[:, :2])**2).sum(-1)/s.track_sigma)
              + s.w_track_yaw*torch.exp(-(body_velocity[:, 2]-planned_velocity[:, 2])**2/s.track_sigma),
        slip=-s.k_slip_lin*rel_linear.norm(dim=-1).clamp(max=1.)-s.k_slip_ang*rel_angular.norm(dim=-1).clamp(max=10.),
        residual=-s.alpha_residual*((finger_residual**2).sum(-1)+(latent_residual**2).sum(-1)),
        force=-s.k_force*(normal_n-s.force_max_n).clamp(min=0).sum(-1),
        table=-s.table_contact_penalty*table_contact.float(),
        drop=-s.drop_penalty*dropped.float(),
        fall=-s.fall_penalty*fell.float())
    terms = {k: v*a for k, v in terms.items()}
    return sum(terms.values()), terms


class ClipSequencer:
    """Per-env sequence of walking clips: settle, random clips at the curriculum level (each cut at a
    random time with ``cut_probability``), standing for the last ``final_stand_s``.

    Times are per-env episode seconds. Phase codes: 0 warm-up, 1 settle, 2 walking, 3 final stand.
    """
    WARMUP, SETTLE, WALK, FINAL = range(4)

    def __init__(self, clips, settings, num_envs, device, generator=None):
        self.clips, self.s, self.g = clips, settings, generator
        self.idle = clips.index('idle')
        n = num_envs
        self.clip = torch.full((n,), self.idle, dtype=torch.long, device=device)
        self.t0 = torch.zeros(n, device=device)
        self.cut = torch.full((n,), float('inf'), device=device)
        self.heading = torch.zeros(n, device=device)
        self.backoff_first = torch.zeros(n, dtype=torch.bool, device=device)
        self.set_level(settings.level)

    def set_level(self, level):
        self.level = int(level)
        self.allowed = torch.tensor(clips_up_to(self.clips, self.level), device=self.clip.device)
        back = [i for i in self.allowed.tolist() if self.clips.names[i].startswith('backward@')]
        self.backward = torch.tensor(back if back else [self.idle], device=self.clip.device)

    def walk_start(self):
        return self.s.warmup_s+self.s.settle_s

    def walk_end(self):
        return self.s.episode_s-self.s.final_stand_s

    def phase(self, t):
        p = torch.full_like(t, self.WALK, dtype=torch.long)
        p = torch.where(t < self.walk_start(), torch.full_like(p, self.SETTLE), p)
        p = torch.where(t < self.s.warmup_s, torch.full_like(p, self.WARMUP), p)
        return torch.where(t >= self.walk_end(), torch.full_like(p, self.FINAL), p)

    def reset(self, ids, heading, backoff_first):
        self.clip[ids] = self.idle
        self.t0[ids] = 0.
        self.cut[ids] = float('inf')
        self.heading[ids] = heading
        self.backoff_first[ids] = backoff_first

    def _rand(self, n, device):
        return torch.rand(n, device=device, generator=self.g)

    def update(self, t, yaw):
        """Advance clips at episode times t (N,); yaw: measured robot yaw (anchors new clips).
        Returns the ids that started a new clip (their reference blends from the previous one)."""
        dev = t.device
        walking = (t >= self.walk_start()) & (t < self.walk_end())
        idle_now = self.clip == self.idle
        over = (t-self.t0 >= torch.minimum(self.clips.duration(self.clip), self.cut))
        start = walking & (idle_now | over)
        stop = (t >= self.walk_end()) & ~idle_now
        ids = torch.nonzero(start).flatten()
        if ids.numel():
            pick = self.allowed[(self._rand(ids.numel(), dev)*len(self.allowed)).long().clamp(max=len(self.allowed)-1)]
            first = self.backoff_first[ids] & idle_now[ids]
            back = self.backward[(self._rand(ids.numel(), dev)*len(self.backward)).long().clamp(max=len(self.backward)-1)]
            self.clip[ids] = torch.where(first, back, pick)
            self.t0[ids] = t[ids]
            self.heading[ids] = yaw[ids]
            dur = self.clips.duration(self.clip[ids])
            cut = self._rand(ids.numel(), dev) < self.s.cut_probability
            self.cut[ids] = torch.where(cut & ~first, 1.5+self._rand(ids.numel(), dev)*(dur-1.5).clamp(min=0),
                                        torch.full_like(dur, float('inf')))
        sid = torch.nonzero(stop).flatten()
        if sid.numel():
            self.clip[sid] = self.idle
            self.t0[sid] = t[sid]
            self.heading[sid] = yaw[sid]
            self.cut[sid] = float('inf')
        return torch.cat((ids, sid))

    def reference(self, t):
        """(q, qd, root quats, planned velocity, command) of every env's current clip at time t."""
        return self.clips.reference(self.clip, (t-self.t0).clamp(min=0), self.heading)


class Curriculum:
    """Speed levels (LEVELS); promotion at a rolling training success over finished episodes.

    All envs start together (run start, every resume), so the first finished episodes are early
    failures followed by a burst of full-length successes: a recency window then overstates the
    success rate (jobs 1549, 1555 promoted at a true ~0.4). So only episodes that STARTED at the
    current level and at least one episode length after the run start or the last promotion count
    (``burn_in_until``, in env steps; the caller filters), promotion needs ``min_episodes`` of them
    (default 2 x num_envs), and a resume restarts the window and the burn-in (level kept)."""

    def __init__(self, settings, device, num_envs=0):
        self.s = settings
        self.level = settings.level
        self.window = torch.zeros(settings.promote_window, device=device)
        self.count = 0
        self.min_episodes = settings.promote_min_episodes or 2*num_envs
        self.burn_in_until = 0          # env step; set by the env (episode length after start / promotion)

    def record(self, success):
        n = success.numel()
        if n == 0:
            return False
        w = self.window
        idx = (torch.arange(n, device=w.device)+self.count) % len(w)
        w[idx] = success.float()
        self.count += n
        if (self.count >= max(len(w), self.min_episodes) and float(w.mean()) >= self.s.promote_success
                and self.level < len(LEVELS)-1):
            self.level += 1
            self.window.zero_()
            self.count = 0
            return True
        return False

    def rate(self):
        n = min(self.count, len(self.window))
        return float(self.window[:n].mean()) if n else 0.

    def state_dict(self):
        return dict(level=self.level, window=self.window.cpu(), count=self.count)

    def load_state_dict(self, state):
        """Level only: the resumed envs start together again, so the window and burn-in restart."""
        self.level = int(state['level'])
        self.window.zero_()
        self.count = 0
