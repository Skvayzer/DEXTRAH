"""Logged carry transitions -> tensors for the hand-object world model (scripts/neurosim/collect_carry_transitions.py).

Row t of a field (T, N, d) is the state after control step t; 'done'[t] = 1 means step t ended an episode
and row t already belongs to the next one (transition t-1 -> t has no next state, only its end label).

Model quantities per row:
  state   x = object COM position (palm frame, m), object orientation (6D, palm frame), object velocity
          relative to rigid palm motion (palm frame), hand joint positions / velocities;
  forces  log(1 + f) of the 5 pad and palm contact forces and of the object-table / object-floor forces
          (part of the recurrent input: the next step sees the predicted forces);
  action  hand joint targets applied during the step into the row;
  exo     palm linear / angular velocity before and after the step (each in its own palm frame), gravity
          in the palm frame before the step, table position relative to the palm and its parked flag, the
          external force on the object during the step (pushes);
  context BPS shape features (per env, 'bps' (N, 128)), log object mass and COM offset (per episode,
          'masscom' (T, N, 4)); gather() joins them;
  labels  contact flags (force > 0.5 N), supported by the hand, lost (episode ended by a drop or a failed
          grasp check).
"""
from dataclasses import dataclass
import math
from pathlib import Path

import torch

from .geometry import matrix_to_rot6, quat_to_matrix

END_REASONS = ('none', 'timeout', 'drop', 'fall', 'start_fail', 'other')
LOST = (END_REASONS.index('drop'), END_REASONS.index('start_fail'))
DROP, START_FAIL = END_REASONS.index('drop'), END_REASONS.index('start_fail')
# fields build_tensors reads (load_chunks(fields=FIELDS) skips the rest of the log)
FIELDS = ('obj_pos', 'obj_quat', 'obj_linvel', 'obj_angvel', 'hand_q', 'hand_qd', 'hand_target', 'palm_linvel',
          'palm_angvel', 'gravity', 'table_pos', 'table_parked', 'object_force', 'finger_force', 'palm_force',
          'table_force', 'ground_force', 'supported', 'mass', 'com_offset', 'done', 'end', 'phase', 'carry_t',
          'arm_now', 'command')
PHASE_BUCKETS = ('secure', 'walk<5s', 'walk5-10s', 'walk>10s', 'final')
CONTACT_N = .5


@dataclass
class Layout:
    hand: int                      # hand joints (actuated + passive)
    actuated: int = 6
    bps: int = 128

    @property
    def state(self):               # continuous state x: pos 3, rot6 6, linvel 3, angvel 3, hand q, hand qd
        return 15+2*self.hand

    @property
    def delta(self):               # predicted change: pos 3, rotvec 3, linvel 3, angvel 3, hand q, hand qd
        return 12+2*self.hand

    forces: int = 8                # 5 pads, palm, table, floor (log1p N)
    exo: int = 3+3+3+3+3+3+1+3     # palm lin/ang vel before, after; gravity; table pos, parked; object force

    @property
    def context(self):             # BPS shape features (the env's width), log mass, COM offset
        return self.bps+1+3


def load_chunks(directory, fields=None, max_chunks=None):
    """Concatenate chunk_*.pt along time -> dict of float32 tensors (T, N, d) and the context file."""
    directory = Path(directory)
    files = sorted(directory.glob('chunk_*.pt'))[:max_chunks]
    if not files:
        raise FileNotFoundError(f'no chunks in {directory}')
    parts = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    keys = [k for k in fields if k in parts[0]] if fields else parts[0].keys()      # optional fields may be absent
    data = {k: torch.cat([p[k] for p in parts]).float() for k in keys}
    return data, torch.load(directory/'context.pt', map_location='cpu', weights_only=False)


def build_tensors(data, context):
    """Model tensors (T, N, d) from the logged fields."""
    T, N = data['done'].shape[:2]
    R = quat_to_matrix(torch.nn.functional.normalize(data['obj_quat'], dim=-1))
    state = torch.cat((data['obj_pos'], matrix_to_rot6(R), data['obj_linvel'], data['obj_angvel'],
                       data['hand_q'], data['hand_qd']), -1)
    forces = torch.log1p(torch.cat((data['finger_force'], data['palm_force'], data['table_force'],
                                    data['ground_force']), -1).clamp_min(0))
    palm_prev = torch.cat((data['palm_linvel'][:1], data['palm_linvel'][:-1]))
    ang_prev = torch.cat((data['palm_angvel'][:1], data['palm_angvel'][:-1]))
    grav_prev = torch.cat((data['gravity'][:1], data['gravity'][:-1]))
    exo = torch.cat((palm_prev, ang_prev, data['palm_linvel'], data['palm_angvel'], grav_prev,
                     data['table_pos'], data['table_parked'], data['object_force']), -1)
    masscom = torch.cat((torch.log(data['mass'].clamp_min(1e-4)), data['com_offset']), -1)
    end = data['end'][..., 0].long()
    done = data['done'][..., 0] > .5
    lost = done & ((end == LOST[0]) | (end == LOST[1]))
    extra = {}
    if 'phase' in data and 'carry_t' in data:
        extra['bucket'] = phase_bucket(data['phase'][..., 0], data['carry_t'][..., 0].float()*context['dt'])
    extra['drop'] = (done & (end == DROP)).float()          # physical drops only
    extra['censor'] = done & (end == START_FAIL)             # failed grasp check: a verdict on the start state
    return dict(state=state, rot=R, forces=forces, action=data['hand_target'], exo=exo, masscom=masscom,
                bps=context['bps'].float(),
                contact=(forces[..., :6] > math.log1p(CONTACT_N)).float(), supported=data['supported'][..., 0],
                done=done, lost=lost.float(), asset=context['asset'], **extra)


def phase_bucket(phase, episode_s, walk_start_s=2.5):
    """Bucket index (PHASE_BUCKETS) of each row: warm-up and secure phase, walking < 5 s / 5-10 s / > 10 s
    after the walk start (C v2: 0.5 s warm-up + 2 s secure), final stand."""
    p = phase.round().long()
    w = episode_s-walk_start_s
    b = torch.where(w < 5, 1, torch.where(w < 10, 2, 3))
    b = torch.where(p <= 1, 0, torch.where(p >= 3, 4, b))
    return b.to(torch.uint8)


class Normalizer(torch.nn.Module):
    """Per-feature mean / std (buffers, saved with the model)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.eps = eps

    def fit(self, x):
        x = x.reshape(-1, x.shape[-1]).double()
        self.mean.copy_(x.mean(0).float())
        self.std.copy_(x.std(0).float().clamp_min(self.eps))
        return self

    def forward(self, x):
        return (x-self.mean)/self.std


def split_envs(asset, test_fraction=.1, seed=0):
    """Train / test env masks by object asset (held-out objects are never seen in training)."""
    g = torch.Generator().manual_seed(seed)
    assets = torch.unique(asset)
    test_assets = assets[torch.randperm(len(assets), generator=g)[:max(1, round(test_fraction*len(assets)))]]
    test = torch.isin(asset, test_assets)
    return ~test, test


def sample_windows(t, envs, k, batch, generator=None, device=None):
    """Random windows: start rows s (uniform over [0, T-k-1]) and env indices from ``envs``."""
    T = t['done'].shape[0]
    s = torch.randint(0, T-k, (batch,), generator=generator)
    e = envs[torch.randint(0, len(envs), (batch,), generator=generator).to(envs.device)]
    return (s.to(device), e.to(device)) if device else (s.to(envs.device), e)


def gather(t, s, e, k):
    """Window tensors (B, k+1, ...) for start rows s and envs e. Masks for steps j = 1..k:
    alive[j]: no episode end at rows s+1 .. s+j-1 (the transition into s+j is from the same episode);
    has_next[j]: alive and row s+j continues the episode (not done) -> state / force / contact losses;
    the lost label at j counts wherever alive (an ending transition has a label but no next state)."""
    idx = s[:, None]+torch.arange(k+1, device=s.device)[None]
    pick = lambda x: x[idx, e[:, None]]
    out = {name: pick(t[name]) for name in ('state', 'rot', 'forces', 'action', 'exo', 'contact', 'supported', 'lost')}
    out['context'] = torch.cat((t['bps'][e][:, None].expand(-1, k+1, -1), pick(t['masscom'])), -1)
    done = pick(t['done'])
    ended_before = torch.cumsum(done[:, 1:].float(), 1)-done[:, 1:].float()     # ends strictly before step j
    alive = ended_before == 0
    out['alive'] = alive
    out['has_next'] = alive & ~done[:, 1:]
    return out
