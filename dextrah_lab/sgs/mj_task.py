"""SGS on the mjlab fixed-body G1 + Revo2 reposing task (opt-in: train --sgs).

Task configuration tau = (s0, g, e) (paper Sec. 3.1): the object asset (e), the
robot joint state, table pose and object state (s0) and the first goal (g).
Episode outcome = the Boolean task success of the episode: at least one goal
reached (``any_goal_success``; the reward is unchanged).

Reset families, balanced in the configuration set as in the paper's
manipulation buffer (Appendix A.2, after OmniReset):

* ``reaching``: this task's own reset distribution rho (object dropped above
  the table at a random pose, robot near its default pose, absolute goal),
  drawn by calling the task's reset functions with their writes captured, so
  rho itself is unchanged.
* ``stable_grasp``: the object held in the hand above the table, fresh goal
  from rho's goal distribution.
* ``near_goal``: the object held in the hand inside the goal volume, goal =
  the object pose moved by a fraction f ~ U[0, f_max] (paper: [0, 1.1] of the
  assembly path) of one of this task's own goal steps (delta_goal_distance,
  delta_rotation_degrees, clamped to the goal volume).

The paper creates Stable Grasp / Near-Goal states with IK around task grasp
points of a parallel-jaw gripper. The dexterous hand has no such generator, so
those states are collected from the policy's own rollouts at the start of the
run (complete MuJoCo qpos/qvel of the world, so each grasp is physically exact)
and then frozen; until a family is complete it has zero probability.

The object is part of the configuration: mjlab keeps each world's object as
per-world model fields (geometry, mass, inertia), rewritten here when an
episode starts with another asset. Derived solver constants (invweight0,
subtree mass) are shared and were computed once from world 0 at build; they
are not recomputed, so every world simulates exactly as in the fixed
assignment.

Evaluation (Sec. 4.3: "Evaluation resets are freshly sampled from the training
distribution"): every ``eval_every``-th env always resets from a fresh sample
of rho with a uniformly random asset and is logged separately (sgs/eval_*).
Unlike the paper's separate evaluation runs, these episodes also feed PPO.

Opt-in, after InterMimicGen (arXiv:2610.06850):
* self-evolving held starts (``reharvest_every``; Sec. 3.4 data flywheel): every round
  replaces part of each held family (unfilled slots, goal edits that never succeeded,
  over-represented objects first) by (a) goal edits of verified held states (same
  physical state, a fresh / farther goal; the edit generation grows) and (b) new held
  states harvested from the current policy's rollouts, objects with the fewest held
  states first. Edits that fail are replaced in a later round ("keep only what
  executes"); replaced slots restart their outcome windows, and episodes started from a
  replaced configuration are not credited to its successor.
* hardness-weighted object sampling (``hard_object_share``; Eq. 9): that share of the
  training resets first draws an object with probability
  rho/|O| + (1-rho) clip(c_o, c_min, 1)^-beta / sum (c_o: the object's from-scratch
  success over its reaching configurations' recent tries), then one of its available
  configurations by its SGS probability.

Whole-body (frozen SONIC) task: the same families on the floating G1. Reaching
starts are that task's rho (standing reset: base at its default root, plus the
opt-in table spawn when enabled, applied to the drawn asset). Held states also
restore the floating base (part of qpos) and SONIC's proprioceptive history up
to the stored step, so the frozen controller sees the motion that led there.
As in the fixed-body port, the action delay queue refills from the episode's
first action (source behaviour) and the touch filter restarts. The spawn check
holds the base at its standing reset pose (as the fixed-body teacher's welded
base) while the joints hold their reset targets.
"""
from dataclasses import asdict, dataclass, field
import inspect
import json
import math

import numpy as np
import torch

from dextrah_lab.mj.compat import p2p
from dextrah_lab.mj.scene import TABLE_SIZE, object_indices
from dextrah_lab.object_shape.bank import map_features
from .sampler import MANIPULATION, SuccessGuidedSampler

FAMILIES = ('reaching', 'stable_grasp', 'near_goal')
RHO_FIELDS = ('joint_pos', 'joint_vel', 'targets', 'setpoint', 'table_z', 'object_pose', 'object_init_z', 'goal_pose')
HARVESTED = ('stable_grasp', 'near_goal')
THUMB = 0   # dextrah_lab.g1_adept.contact.FINGERS order


@dataclass
class SgsSettings:
    num_configs: int = MANIPULATION['num_configs']
    history: int = MANIPULATION['history']
    temperature: float = MANIPULATION['temperature']
    target: float = MANIPULATION['target']
    kappa: float = MANIPULATION['kappa']
    eps: float = MANIPULATION['eps']
    families: tuple = FAMILIES
    eval_every: int = 16                 # env i is an evaluation env when i % eval_every == eval_every-1 (0: none)
    harvest_per_step: int = 32           # new held states per family per env step while collecting
    harvest_max_steps: int = 6000        # then freeze whatever was collected
    harvest_asset_cap: float = 2.        # per asset at most cap * (family size / assets) states
    near_goal_max_fraction: float = 1.1  # f ~ U[0, f_max] of one goal step
    grasp_force_n: float = .25           # thumb + another pad above this force
    table_clearance_m: float = .05       # object origin above the table top
    max_object_speed: float = .5         # m/s, centre of mass
    max_object_spin: float = 5.          # rad/s
    stats_window: int = 3000             # episodes per rolling statistic (rl_games games_to_track)
    # Spawn validation (opt-in): reaching configurations and evaluation starts must pass a
    # passive settle check (robot holding its reset targets, no policy).
    validate_spawns: bool = False
    eval_pool_size: int = 16384          # validated fresh rho starts the evaluation envs draw from
    settle_s: float = 1.5                # simulated settling time per candidate
    spawn_overlap_robot_m: float = .002  # first-step penetration into the robot that counts as an overlap spawn
    spawn_overlap_table_m: float = .005  # ... into the table
    spawn_max_speed: float = 4.          # m/s, COM, during settling (safety net for violent ejections)
    spawn_rest_speed: float = .05        # m/s, COM, at the end of settling
    spawn_edge_margin: float = .03       # settled COM at least this far inside the table top edges
    spawn_max_rounds: int = 20           # redraws per slot before giving up
    # Self-evolving held starts (opt-in; see the module docstring).
    reharvest_every: int = 0             # env steps between rounds (0: off); starts after the first harvest
    reharvest_steps: int = 300           # env steps of collection per round
    reharvest_fraction: float = .2       # at most this share of each held family replaced per round
    reharvest_min_tries: int = 10        # outcomes before a configuration is judged
    edit_fraction: float = .5            # share of a round's replacements made by goal edits
    edit_parent_success: float = .5      # edits copy held states with at least this success over their tries
    edit_goal_growth: float = .5         # near-goal edits: goal fraction bound grows by this per generation
    # Whole body (opt-in): held states only while frozen SONIC's executed body action and its
    # stored history stay within this bound (normalized SONIC action units; 0: off). Policies
    # pin the right wrist at its joint limit (~21) in ~20-27% of lifted steps; restoring such
    # states with a reset policy memory destabilized PPO (job 1447). Saved held states outside
    # the bound are replaced by goal edits of in-bound ones when the state is loaded.
    held_max_body_action: float = 0.
    # Hardness-weighted object sampling (opt-in).
    hard_object_share: float = 0.        # share of training resets drawn object-first
    hard_object_beta: float = .5
    hard_object_min: float = .05
    hard_object_uniform: float = .25

    def __post_init__(self):
        self.families = tuple(self.families)
        unknown = set(self.families)-set(FAMILIES)
        if unknown or not self.families or len(set(self.families)) != len(self.families):
            raise ValueError(f'SGS families must be distinct members of {FAMILIES}, got {self.families}')
        if self.num_configs < len(self.families):
            raise ValueError('Fewer configurations than families')
        if set(self.families) & set(HARVESTED) and 'reaching' not in self.families:
            raise ValueError('Held states are collected from reaching episodes: keep the reaching family')
        if self.reharvest_every < 0 or not 0. <= self.reharvest_fraction <= 1. or not 0. <= self.edit_fraction <= 1.:
            raise ValueError('Invalid re-harvest settings')
        if not 0. <= self.hard_object_share <= 1. or not 0. <= self.hard_object_uniform <= 1. or self.hard_object_min <= 0.:
            raise ValueError('Invalid hardness-weighted sampling settings')
        if self.hard_object_share and 'reaching' not in self.families:
            raise ValueError('Object hardness is measured on the reaching family')

    def layout(self):
        """Fields that fix the saved configuration set and outcome windows."""
        return dict(num_configs=self.num_configs, history=self.history, families=tuple(self.families))


HEIGHT_BINS = 4              # table-height bins for the per-height success metrics
HELD_TABLE_CLEARANCE = .03   # held starts fit a table whose top is at least this far below the lowest object keypoint
                             # and below every robot geom above the table (exact: table_fit_mask)
HELD_FIT_TRIES = 8           # re-draws of a held start that does not fit its sampled table height
TABLE_FIT_STEP = .01         # table-top grid of the robot clearance masks (m)
RESTORE_CONTACT_N = .5       # robot-table normal force on the first step after a restore that counts as contact


def table_mocap(m, table_body):
    """(mocap id, z of the table body above its mocap ancestor): the table box is welded under a mocap body."""
    b, dz = int(table_body), 0.
    while int(m.body_mocapid[b]) < 0:
        if b == 0:
            raise RuntimeError('The table is expected to hang under a mocap body')
        dz += float(m.body_pos[b][2])
        b = int(m.body_parentid[b])
    return int(m.body_mocapid[b]), dz


PENETRATION_TOL = .001       # a table height may deepen a robot geom's table penetration by at most this
ARM_FOLLOW_JOINTS = ('right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                     'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
ARM_FOLLOW_BODY = 'right_base_link'    # the hand base: moved with the table, orientation kept, fingers ride along


def arm_follow_ik(m, qpos, shifts, qadr, dadr, hand_body, iters=150, tol_pos=2e-4, tol_rot=2e-3, damping=1e-2,
                  rot_weight=.2, max_step=.1, null_gain=.2):
    """Arm joint offsets that move the hand body straight up by shifts[i, j] with its orientation kept, everything
    else fixed: damped least squares on the joints qadr / dadr (rotation error weighted rot_weight m/rad, step norm
    <= max_step rad, null-space pull toward the original arm pose, joint limits respected; per state the shifts
    are solved outward from the smallest, each warm-started from its neighbour). qpos (n, nq), shifts (n, S).
    Returns (offsets (n, S, len(qadr)), position residual (n, S) m, rotation residual (n, S) rad)."""
    import mujoco
    d = mujoco.MjData(m)
    qadr, dadr = np.asarray(qadr), np.asarray(dadr)
    jnt = np.array([int(np.nonzero(m.jnt_qposadr == a)[0][0]) for a in qadr])
    lo = np.where(m.jnt_limited[jnt] > 0, m.jnt_range[jnt, 0], -np.inf)
    hi = np.where(m.jnt_limited[jnt] > 0, m.jnt_range[jnt, 1], np.inf)
    n, S = shifts.shape
    out = np.zeros((n, S, len(qadr)))
    res_p, res_r = np.zeros((n, S)), np.zeros((n, S))
    jacp, jacr, err_r = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros(3)
    eye = damping**2*np.eye(6)
    for i in range(n):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(m, d)
        p0, q0 = d.xpos[hand_body].copy(), d.xquat[hand_body].copy()
        base = qpos[i, qadr].copy()
        order = np.argsort(shifts[i])
        start = int(np.argmin(np.abs(shifts[i][order])))
        for sweep in (order[start:], order[:start+1][::-1]):
            q = base.copy()
            for j in sweep:
                target = p0+np.array([0., 0., shifts[i, j]])
                for _ in range(iters):
                    d.qpos[qadr] = q
                    mujoco.mj_kinematics(m, d)
                    e_p = target-d.xpos[hand_body]
                    mujoco.mju_subQuat(err_r, q0, d.xquat[hand_body])
                    e_r = d.xmat[hand_body].reshape(3, 3) @ err_r
                    if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
                        break
                    mujoco.mj_comPos(m, d)
                    mujoco.mj_jacBody(m, d, jacp, jacr, hand_body)
                    J = np.vstack((jacp[:, dadr], rot_weight*jacr[:, dadr]))
                    pinv = J.T @ np.linalg.inv(J @ J.T+eye)
                    dq = pinv @ np.concatenate((e_p, rot_weight*e_r))
                    dq += (np.eye(len(qadr))-pinv @ J) @ (null_gain*(base-q))     # stay near the original arm pose
                    norm = np.linalg.norm(dq)
                    if norm > max_step:
                        dq *= max_step/norm
                    q = np.clip(q+dq, lo, hi)
                d.qpos[qadr] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mju_subQuat(err_r, q0, d.xquat[hand_body])
                out[i, j] = q-base
                res_p[i, j] = np.linalg.norm(target-d.xpos[hand_body])
                res_r[i, j] = np.linalg.norm(err_r)
            d.qpos[:] = qpos[i]
    return out, res_p, res_r


class TableContacts:
    """Robot-table contacts of robot states (CPU MuJoCo) with the table top at a given height; `margin` raises the
    table geom's contact margin (contacts up to that distance are reported)."""

    def __init__(self, m, robot_bodies, table_body, margin=0., table_xy=(0., 0.)):
        import copy
        import mujoco
        self.mujoco = mujoco
        self.m = copy.deepcopy(m)
        table_geoms = np.nonzero(self.m.geom_bodyid == table_body)[0]
        self.m.geom_margin[table_geoms] = margin
        self.d = mujoco.MjData(self.m)
        self.mocap, self.dz = table_mocap(self.m, table_body)
        self.half = float(self.m.geom_size[table_geoms[0], 2])
        self.is_table = np.zeros(self.m.ngeom, dtype=bool)
        self.is_table[table_geoms] = True
        self.is_robot = np.isin(self.m.geom_bodyid, np.asarray(robot_bodies))
        self.xy = table_xy

    def at(self, qpos, top):
        """(robot geom, distance, above-the-top flag) of each robot-table contact; above = contact point at the top
        face with a mostly vertical normal."""
        d = self.d
        d.qpos[:] = qpos
        d.mocap_pos[self.mocap] = (self.xy[0], self.xy[1], float(top)-self.half-self.dz)
        self.mujoco.mj_kinematics(self.m, d)
        self.mujoco.mj_collision(self.m, d)
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        pair = (self.is_table[g1] & self.is_robot[g2]) | (self.is_table[g2] & self.is_robot[g1])
        robot = np.where(self.is_table[g1], g2, g1)[pair]
        above = (d.contact.pos[:n, 2] > float(top)-.005) & (np.abs(d.contact.frame[:n, 2]) > .5)
        return robot, d.contact.dist[:n][pair], above[pair]

    def penetration(self, qpos, top):
        """(ngeom,) table penetration depth per robot geom (<= 0, 0 = none)."""
        robot, dist, _ = self.at(qpos, top)
        base = np.zeros(self.m.ngeom)
        np.minimum.at(base, robot, np.minimum(dist, 0.))
        return base


def table_fit_mask(m, qpos, tops, stored_tops, robot_bodies, table_body, clearance, table_xy=(0., 0.)):
    """Robot clearance of held states per table-top height, NO WORSE THAN THE STORED STATE (whose table top is
    stored_tops[i]; M's held states lean on the table edge in about a fifth of the cases, < 2.2 mm deep):
      * no robot geom penetrates the table more than PENETRATION_TOL deeper than at the stored table, and
      * when the table is raised above its stored height, no robot geom is within `clearance` above its top
        (side gaps to the box faces and gaps under the box bottom do not count).
    Returns ((n, len(tops)) bool, (n,) bool: robot-table penetration at the stored table). A table contact
    interval along the vertical is at least the box height (0.3 m) long, so two clear neighbouring grid heights
    mean the heights between them are clear too."""
    near = TableContacts(m, robot_bodies, table_body, margin=clearance, table_xy=table_xy)
    out = np.zeros((len(qpos), len(tops)), dtype=bool)
    stored_contact = np.zeros(len(qpos), dtype=bool)
    for i in range(len(qpos)):
        base = near.penetration(qpos[i], stored_tops[i])
        stored_contact[i] = bool((base < 0).any())
        for j, top in enumerate(tops):
            robot, dist, above = near.at(qpos[i], top)
            deeper = dist < base[robot]-PENETRATION_TOL
            close = (float(top) > float(stored_tops[i])+1e-6) & (dist < clearance) & above
            out[i, j] = not bool((deeper | close).any())
    return out, stored_contact


def lowest_keypoint_z(pose, offsets):
    """Lowest z of the object keypoints (n, K, 3 offsets in the object frame) at env-local poses (n, 7: pos, wxyz)."""
    w, x, y, z = pose[:, 3:7].unbind(-1)
    row = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)          # third row of R
    return pose[:, 2]+(offsets*row[:, None]).sum(-1).amin(-1)


class _Rolling:
    """Mean of the last ``capacity`` values (device ring buffer)."""

    def __init__(self, capacity, device):
        self.values = torch.zeros(capacity, device=device)
        self.head, self.size = 0, 0

    def add(self, values):
        values = values.reshape(-1).float()[-len(self.values):]
        n, cap = values.numel(), len(self.values)
        if n == 0:
            return
        idx = (self.head+torch.arange(n, device=values.device)) % cap
        self.values[idx] = values
        self.head, self.size = (self.head+n) % cap, min(self.size+n, cap)

    def mean(self):
        return self.values[:self.size].mean() if self.size else torch.zeros((), device=self.values.device)


class _WorstBin:
    """Rolling-window view for SgsEvalBest: the lowest evaluation success over the table-height bins (+1e-3 x their
    mean as the tiebreak); 'full' once every bin's window is."""

    def __init__(self, bins):
        self.bins = bins
        self.values = bins[0].values

    @property
    def size(self):
        return min(b.size for b in self.bins)

    def mean(self):
        m = torch.stack([b.mean() for b in self.bins])
        return m.min()+1e-3*m.mean()

    def add(self, values):
        raise RuntimeError('table_worst_bin is derived from the per-bin windows')


class MjSgsTask:
    def __init__(self, env, settings=SgsSettings(), generator=None, state=None):
        self.wholebody = not getattr(env, '_fixed_body', False)
        if self.wholebody and not getattr(env, '_wholebody_ready', False):
            raise NotImplementedError('SGS supports the fixed-body task and the frozen-SONIC whole-body task')
        rc = env.cfg.reset
        if (rc.fixed_start_pose is not None or rc.fixed_goal_pose is not None or rc.fixed_trajectory_file
                or tuple(rc.table_reset_xy_range_m) != (0., 0.) or float(rc.table_reset_yaw_range_deg) != 0.):
            raise NotImplementedError('SGS expects the run-383 reset setup (random start and goal, table moves in z only)')
        self.env, self.settings, self.device = env, settings, env.device
        self.generator = generator
        # Swapping objects rewrites only primary per-world fields. Derived
        # constants must be shared (computed once from world 0), otherwise a
        # swapped world would keep its previous object's values.
        per_world = [f for f in ('body_subtreemass', 'body_invweight0', 'dof_invweight0')
                     if getattr(env.sim.wp_model, f).shape[0] != 1]
        if per_world:
            raise NotImplementedError(f'SGS object swaps would need recomputed per-world {per_world}')
        self.reset_utils, self.goal_sampling = p2p('reset_utils'), p2p('goal_sampling')
        self._check_reset_order()
        n_env = env.num_envs
        self.sampler = SuccessGuidedSampler(settings.num_configs, settings.history, settings.temperature,
                                            settings.target, settings.kappa, settings.eps, self.device)
        # Balanced family blocks: [start, stop) per family.
        sizes = [settings.num_configs//len(settings.families)+(i < settings.num_configs % len(settings.families))
                 for i in range(len(settings.families))]
        bounds = np.cumsum([0]+sizes)
        self.blocks = {f: (int(bounds[i]), int(bounds[i+1])) for i, f in enumerate(settings.families)}
        self.family = torch.empty(settings.num_configs, dtype=torch.long, device=self.device)
        for i, f in enumerate(settings.families):
            self.family[slice(*self.blocks[f])] = FAMILIES.index(f)
        self._build_asset_tables()
        # Configuration storage.
        N, nj = settings.num_configs, env.robot.data.joint_pos.shape[1]
        nq, nv = env.sim.data.qpos.shape[1], env.sim.data.qvel.shape[1]
        self.na = int(env.sim.mj_model.na)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=self.device)
        self.cfg = dict(asset=z(N, dtype=torch.long), table_z=z(N), joint_pos=z(N, nj), joint_vel=z(N, nj),
                        targets=z(N, nj), setpoint=z(N, nj), object_pose=z(N, 7), object_init_z=z(N),
                        goal_pose=z(N, 7), qpos=z(N, nq), qvel=z(N, nv), act=z(N, max(self.na, 1)))
        if self.wholebody:
            # SONIC proprioceptive history frames (oldest..newest, newest = the stored step).
            hist = env._body_history.buffers
            self.hist_dims = [b.shape[2] for b in hist]
            self.cfg['history'] = z(N, hist[0].shape[1], sum(self.hist_dims))
        self.filled = torch.zeros(N, dtype=torch.bool, device=self.device)
        # Self-evolving held starts: edit lineage, slot versions (outcome attribution), rounds.
        self.generation = torch.zeros(N, dtype=torch.long, device=self.device)
        self.parent = torch.full((N,), -1, dtype=torch.long, device=self.device)
        self.edited = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.slot_version = torch.zeros(N, dtype=torch.long, device=self.device)
        self.round, self.round_start, self.round_queue, self.round_log = 0, None, {}, {}
        self._asset_index, self._completion, self._completion_step = None, None, -1
        self.fill_count = {f: 0 for f in settings.families}
        self.harvest_steps, self.harvest_done = 0, not any(f in HARVESTED for f in settings.families)
        self.asset_harvest = {f: torch.zeros(self.n_assets, dtype=torch.long, device=self.device) for f in HARVESTED}
        # Per-env bookkeeping.
        self.current = torch.full((n_env,), -1, dtype=torch.long, device=self.device)
        self.current_version = torch.zeros(n_env, dtype=torch.long, device=self.device)
        ids = torch.arange(n_env, device=self.device)
        e = settings.eval_every
        self.eval_mask = (ids % e == e-1) if e > 0 else torch.zeros(n_env, dtype=torch.bool, device=self.device)
        if bool(self.eval_mask.all()):
            raise ValueError('SGS needs training envs besides the evaluation envs')
        self.eval_started = torch.zeros(n_env, dtype=torch.bool, device=self.device)
        w = settings.stats_window
        self.eval_stats = {k: _Rolling(w, self.device) for k in ('any_goal_success_rate', 'goals_reached_count', 'ever_lifted_rate')
                           +(('robot_fall_rate',) if self.wholebody else ())}
        self.train_stats = {f: _Rolling(w, self.device) for f in settings.families}
        self.fall_stats = {f: _Rolling(w, self.device) for f in settings.families}
        self.robot_fall_stats = {f: _Rolling(w, self.device) for f in settings.families} if self.wholebody else {}
        # Table heights (opt-in, cfg.sonic_body.sgs_table_heights; 2026-10-28 placement review): the table height of
        # every training start is drawn first (uniform over the reset range), then the start: a held / near-goal
        # start that does not fit that height (table top within HELD_TABLE_CLEARANCE of the object's lowest
        # keypoint, its stored goal outside the goal band above that table, or a robot geom deeper in the table /
        # within HELD_TABLE_CLEARANCE above its top than in the stored state: table_fit_mask) is re-drawn from the
        # sampler, so the height stays independent of the start type.  Startup check: SGS_TABLE_FIT_CHECK. Resting starts move object, lift reference and goal
        # with the table (robot pose kept); held starts keep object and goal, only the table moves. Success and
        # knock-off rate are logged per start family x table-top bin (train) and per bin (evaluation).
        rc = env.cfg.reset
        self.table_heights = bool(getattr(getattr(env.cfg, 'sonic_body', None), 'sgs_table_heights', False))
        self.table_lo = float(rc.table_reset_z)-float(rc.table_reset_z_range)
        self.table_hi = float(rc.table_reset_z)+float(rc.table_reset_z_range)
        nominal_top = float(rc.table_reset_z)+TABLE_SIZE[2]/2
        self.goal_band = (float(rc.target_volume_mins[2])-nominal_top, float(rc.target_volume_maxs[2])-nominal_top)
        self.height_edges = torch.linspace(self.table_lo, self.table_hi, HEIGHT_BINS+1, device=self.device)
        self.height_train = {(f, b): _Rolling(w, self.device) for f in settings.families for b in range(HEIGHT_BINS)}
        self.height_knock = {(f, b): _Rolling(w, self.device) for f in settings.families for b in range(HEIGHT_BINS)}
        self.height_eval = [_Rolling(w, self.device) for _ in range(HEIGHT_BINS)]
        self.height_eval_knock = [_Rolling(w, self.device) for _ in range(HEIGHT_BINS)]
        self.height_starts = torch.zeros(len(FAMILIES), HEIGHT_BINS, device=self.device)   # starts per family x bin
        self.pending_table_z = torch.zeros(n_env, device=self.device)
        self.fit_stats = dict(held_drawn=0., held_rejected=0., fallback=0.)
        if self.table_heights:
            # best checkpoint by the worst table-height bin (BEST_METRIC=sgs_eval_table_worst)
            self.eval_stats['table_worst_bin'] = _WorstBin(self.height_eval)
        # Robot clearance of held starts per table-top grid height (table_fit_mask, CPU MuJoCo; not saved:
        # recomputed at startup and when a slot gets a new held state).
        self.fit_tops = torch.arange(round((self.table_hi-self.table_lo)/TABLE_FIT_STEP)+1, device=self.device)*TABLE_FIT_STEP \
            + self.table_lo+TABLE_SIZE[2]/2
        self.table_fit = torch.zeros(N, len(self.fit_tops), dtype=torch.bool, device=self.device)
        self.stored_contact = torch.zeros(N, dtype=torch.bool, device=self.device)   # robot on the table when stored
        # Resting starts and fresh rho draws (evaluation pool): the right arm follows the table (2026-10-29: M's
        # reset arm pose puts the hand at the 0.80 m table top, so on raised tables it started inside the table):
        # arm offsets per resting configuration and table-top grid height (arm_follow_ik, CPU, cached in the
        # state); rho draws are shifted by their table's offset from the nominal height before validation.
        self.arm_follow = self.table_heights or bool(getattr(getattr(env.cfg, 'sonic_body', None), 'arm_follows_table', False))
        self.arm_shift = torch.zeros(N, len(self.fit_tops), len(ARM_FOLLOW_JOINTS), device=self.device)
        self.arm_known = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.arm_stats = {}
        if self.arm_follow:
            m, names = env.sim.mj_model, list(env.robot.joint_names)
            self.arm_cols = torch.tensor([names.index(j) for j in ARM_FOLLOW_JOINTS], device=self.device)
            self.arm_qadr = [int(m.jnt_qposadr[m.joint(f'robot/{j}').id]) for j in ARM_FOLLOW_JOINTS]
            self.arm_dadr = [int(m.jnt_dofadr[m.joint(f'robot/{j}').id]) for j in ARM_FOLLOW_JOINTS]
            self.arm_hand = int(m.body(f'robot/{ARM_FOLLOW_BODY}').id)
            self.joint_qadr = np.array([int(m.jnt_qposadr[m.joint(f'robot/{j}').id]) for j in names])
            root = env.robot.data.default_root_state[0, :7].clone()
            root[:3] -= env.scene.env_origins[0]
            self.rest_root = root.cpu().numpy().astype(np.float64)
        self.fit_known = torch.zeros(N, dtype=torch.bool, device=self.device)
        # First step after a restore: robot-table contact rate of training starts (0 resting, 1 held).
        self.contact_check = torch.full((n_env,), -1, dtype=torch.long, device=self.device)
        self.restore_contact = [_Rolling(w, self.device) for _ in range(3)]   # resting, held new, held any
        self.restore_contact_warned = 0
        self.fit_check = {}
        self.steps = 0
        self.validated = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.spawn_stats = dict(checked=0, valid=0, spawn_overlap=0, ejected=0, off_table=0, not_at_rest=0, max_nefc=0, max_nefc_other=0)
        if state is not None:
            self.load_state_dict(state)
        elif 'reaching' in settings.families:
            self._fill_reaching()
        if self.table_heights:
            rest = torch.nonzero(self.filled & (self.family == FAMILIES.index('reaching')) & ~self.arm_known).flatten()
            self._arm_tables(rest)
            held = torch.nonzero(self.filled & (self.family != FAMILIES.index('reaching'))).flatten()
            self._table_fit_cpu(held[~self.fit_known[held]])
            self._fit_check(held)
        # Evaluation starts: fresh rho (default) or a validated pool of fresh rho draws.
        self.eval_pool = None
        self.eval_current = torch.zeros(n_env, dtype=torch.long, device=self.device)
        if settings.validate_spawns and settings.eval_pool_size > 0 and bool(self.eval_mask.any()):
            P = settings.eval_pool_size
            self.eval_pool = {k: torch.zeros(P, *v.shape[1:], dtype=v.dtype, device=self.device) for k, v in self.cfg.items()
                              if k in RHO_FIELDS+('asset',)}
            self.eval_pool['asset'] = torch.randint(0, self.n_assets, (P,), device=self.device, generator=self.generator)
            self._draw_valid(torch.arange(P, device=self.device), self.eval_pool)
        if settings.validate_spawns:
            print('SGS_SPAWN_CHECK '+json.dumps(self.spawn_stats), flush=True)
        if self.arm_follow:
            print('SGS_ARM_FOLLOW '+json.dumps(self.arm_stats), flush=True)
        self._sync_available()

    # ------------------------------------------------------------------ setup
    def _check_reset_order(self):
        """The reaching capture replays reset_env_state's sampling calls."""
        src = inspect.getsource(self.reset_utils.reset_env_state)
        calls = ['_randomize_robot_dof_state(env, env_ids)', '_reset_table_pose(env, env_ids)',
                 '_reset_object_pose(env, env_ids)', '_reset_goal_pose(env, env_ids, mode="absolute")']
        where = [src.find(c) for c in calls]
        if min(where) < 0 or where != sorted(where):
            raise RuntimeError('reset_env_state no longer samples robot, table, object, goal in this order')

    def _build_asset_tables(self):
        env, t = self.env, (lambda x: torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float32))
        bank = env.object_bank
        self.n_assets = len(env._object_urdf_paths)
        self.body, geoms = object_indices(env.sim.mj_model)
        self.geoms = torch.as_tensor(geoms, device=self.device, dtype=torch.long)
        self.tab = dict(size=t(bank.size), quat=t(bank.quat), rbound=t(bank.rbound),
                        aabb=t(bank.aabb).view(self.n_assets, len(geoms), 2, 3), mass=t(bank.mass), com=t(bank.com),
                        inertia=t(bank.inertia))
        self.tab['pos'] = t(bank.pos)-self.tab['com'][:, None]
        self.tab['bps'] = torch.as_tensor(map_features(env._bps_bank, np.arange(self.n_assets)), device=self.device,
                                          dtype=env._bps_features.dtype)
        # Per-asset scale: recovered from the initial assignment (every asset is assigned).
        asset = env._object_asset_index_per_env
        if int(torch.unique(asset).numel()) != self.n_assets:
            raise RuntimeError('SGS needs every asset in the initial assignment (num_envs >= assets)')
        scale = torch.zeros(self.n_assets, env._object_scale_per_env.shape[1], device=self.device)
        scale[asset] = env._object_scale_per_env
        self.tab['scale'] = scale
        rew = env.cfg.reward
        corners = torch.tensor(p2p('obs_utils').KEYPOINT_CORNERS, device=self.device, dtype=torch.float32)
        self.tab['keypoints'] = corners[None]*(scale*rew.object_base_size*rew.keypoint_scale*.5)[:, None]
        self._verify_tables()

    def _verify_tables(self):
        """The tables must reproduce the current per-world state exactly (at
        startup, and periodically after object swaps)."""
        env, a, m = self.env, self.env._object_asset_index_per_env, self.env.sim.model
        g = self.geoms
        checks = dict(geom_size=(m.geom_size[:, g], self.tab['size'][a]), geom_pos=(m.geom_pos[:, g], self.tab['pos'][a]),
                      geom_quat=(m.geom_quat[:, g], self.tab['quat'][a]), geom_rbound=(m.geom_rbound[:, g], self.tab['rbound'][a]),
                      geom_aabb=(m.geom_aabb[:, g], self.tab['aabb'][a]), body_mass=(m.body_mass[:, self.body], self.tab['mass'][a]),
                      body_inertia=(m.body_inertia[:, self.body], self.tab['inertia'][a]),
                      com_offset=(env.object.com_offset, self.tab['com'][a]), default_mass=(env.object.default_mass[:, 0], self.tab['mass'][a]),
                      scale=(env._object_scale_per_env, self.tab['scale'][a]), keypoints=(env._keypoint_offsets, self.tab['keypoints'][a]),
                      bps=(env._bps_features, self.tab['bps'][a]))
        bad = [k for k, (x, y) in checks.items() if x.shape != y.shape or not torch.equal(x.float(), y.float())]
        if bad:
            raise RuntimeError(f'SGS asset tables do not reproduce the scene: {bad}')

    def _capture_reaching(self, n, assets=None):
        """n draws of the task's reset distribution rho (robot, table, object, absolute
        goal) from the task's own reset functions, writes captured, not applied.
        ``assets``: the draws' objects (used only by the whole-body table spawn)."""
        env, ru = self.env, self.reset_utils
        saved = {k: getattr(env, k).clone() for k in ('_prev_targets', '_cur_targets', '_table_z_per_env', '_object_init_z')}
        rec = {}
        patches = {(env.robot, 'write_joint_state_to_sim'): lambda pos, vel, joint_ids=None, env_ids=None: rec.update(joint_pos=pos.clone(), joint_vel=vel.clone()),
                   (env.table, 'write_root_pose_to_sim'): lambda pose, env_ids=None: rec.update(table=pose.clone()),
                   (env.object, 'write_root_pose_to_sim'): lambda pose, env_ids=None: rec.update(object=pose.clone()),
                   (env.object, 'write_root_velocity_to_sim'): lambda vel, env_ids=None: rec.update(object_vel=vel.clone()),
                   (env.goal_viz, 'write_root_pose_to_sim'): lambda pose, env_ids=None: rec.update(goal=pose.clone())}
        for (obj, name), fn in patches.items():
            setattr(obj, name, fn)
        out = {k: [] for k in RHO_FIELDS}
        try:
            k = 0
            while k < n:
                ids = torch.arange(min(n-k, env.num_envs), device=self.device)
                rec.clear()
                ru._randomize_robot_dof_state(env, ids)
                ru._reset_table_pose(env, ids)
                ru._reset_object_pose(env, ids)
                ru._reset_goal_pose(env, ids, mode='absolute')
                if self.wholebody and getattr(env, '_table_spawn', None) is not None:
                    # Opt-in whole-body table spawn (wholebody/spawn.py, applied after the
                    # source reset in env._reset_idx): part of rho; depends on the asset.
                    if assets is None:
                        raise ValueError('The table spawn needs the assets of the draws')
                    rc = env.cfg.reset
                    pos = env._table_spawn.place(rec['object'][:, 3:7], env._table_z_per_env[ids], rc.object_spawn_xy_mins,
                                                 rc.object_spawn_xy_maxs, ids, assets=assets[k:k+len(ids)])
                    rec['object'] = torch.cat((pos+env.scene.env_origins[ids], rec['object'][:, 3:7]), -1)
                    env._object_init_z[ids] = pos[:, 2]
                if bool(rec['object_vel'].abs().max() > 0):
                    raise RuntimeError('Reaching resets are expected to start the object at rest')
                origin = env.scene.env_origins[ids]
                out['joint_pos'].append(rec['joint_pos'])
                out['joint_vel'].append(rec['joint_vel'])
                out['targets'].append(env._cur_targets[ids].clone())
                out['setpoint'].append(env._cur_targets[ids].clone())
                out['table_z'].append(rec['table'][:, 2]-origin[:, 2])
                out['object_pose'].append(torch.cat((rec['object'][:, :3]-origin, rec['object'][:, 3:]), -1))
                out['object_init_z'].append(env._object_init_z[ids].clone())
                out['goal_pose'].append(torch.cat((rec['goal'][:, :3]-origin, rec['goal'][:, 3:]), -1))
                k += len(ids)
        finally:
            for obj, name in patches:
                delattr(obj, name)
            for key, value in saved.items():
                getattr(env, key).copy_(value)
        return {k: torch.cat(v) for k, v in out.items()}

    def _draw_valid(self, slots, store):
        """Fill store[...][slots] (assets already set) with rho draws; with
        validate_spawns, redraw each slot until its draw passes the settle check.
        Returns the slots that never passed in spawn_max_rounds: they keep their
        last (unvalidated) rho draw and are logged with the last failure reason."""
        s = self.settings
        todo = slots
        for _ in range(s.spawn_max_rounds if s.validate_spawns else 1):
            cand = self._capture_reaching(len(todo), store['asset'][todo])
            if self.arm_follow:
                self._arm_follow_capture(cand)
            ok = self._spawn_check(cand, store['asset'][todo]) if s.validate_spawns else \
                torch.ones(len(todo), dtype=torch.bool, device=self.device)
            keep = todo[ok]
            for key, value in cand.items():
                store[key][keep] = value[ok]
            if store is self.cfg:
                self.arm_known[keep] = False
            if bool(ok.all()):
                return todo[:0]
            last = {key: value[~ok] for key, value in cand.items()}
            reasons = self._last_reasons[~ok] if hasattr(self, '_last_reasons') else None
            todo = todo[~ok]
        for key, value in last.items():
            store[key][todo] = value
        names = ('valid', 'spawn_overlap', 'ejected', 'off_table', 'not_at_rest')
        unresolved = [dict(asset=int(a), reason=names[int(r)] if reasons is not None else 'unknown')
                      for a, r in zip(store['asset'][todo].tolist(), (reasons if reasons is not None else todo).tolist())]
        self.spawn_stats.setdefault('unresolved', []).extend(unresolved)
        print(f'SGS_SPAWN_UNRESOLVED {len(unresolved)} after {s.spawn_max_rounds} rounds: {unresolved}', flush=True)
        return todo

    def _rest_qpos(self, joint_pos):
        """CPU qpos of standing robots (default root) with joint positions joint_pos (n, nj); object at qpos0."""
        m = self.env.sim.mj_model
        q = np.tile(m.qpos0.astype(np.float64), (len(joint_pos), 1))
        q[:, :7] = self.rest_root
        q[:, self.joint_qadr] = joint_pos.cpu().numpy().astype(np.float64)
        return q

    def _arm_ik(self, joint_pos, shifts):
        return arm_follow_ik(self.env.sim.mj_model, self._rest_qpos(joint_pos), shifts, self.arm_qadr, self.arm_dadr,
                             self.arm_hand)

    def _arm_follow_capture(self, cand):
        """Fresh rho draws: shift the arm so the hand keeps its height above the drawn table (rho's arm pose is
        set for the nominal table)."""
        dz = (cand['table_z']-float(self.env.cfg.reset.table_reset_z)).cpu().numpy().astype(np.float64)[:, None]
        dq, res, _ = self._arm_ik(cand['joint_pos'], dz)
        dq = torch.as_tensor(dq[:, 0], dtype=torch.float32, device=self.device)
        for key in ('joint_pos', 'targets', 'setpoint'):
            cand[key][:, self.arm_cols] += dq
        st = self.arm_stats.setdefault('rho_draws', dict(n=0, over_5mm=0, max_mm=0.))
        st['n'] += len(dz)
        st['over_5mm'] += int((res > .005).sum())
        st['max_mm'] = max(st['max_mm'], round(1000*float(res.max()), 1) if res.size else 0.)

    def _arm_tables(self, slots):
        """Arm offsets of resting configurations per table-top grid height, relative to their stored table."""
        if slots.numel() == 0:
            return
        c = self.cfg
        shifts = (self.fit_tops[None, :]-(c['table_z'][slots]+TABLE_SIZE[2]/2)[:, None]).cpu().numpy().astype(np.float64)
        dq, res, rot = self._arm_ik(c['joint_pos'][slots], shifts)
        self.arm_shift[slots] = torch.as_tensor(dq, dtype=torch.float32, device=self.device)
        self.arm_known[slots] = True
        over = res > .005
        self.arm_stats['resting'] = dict(configs=int(slots.numel()), over_5mm_by_top={f'{t:.2f}': round(float(v), 3) for t, v in
                                                                                      zip(self.fit_tops[::4].tolist(), over.mean(0)[::4].tolist())},
                                         max_mm=round(1000*float(res.max()), 1), rot_p99_deg=round(float(np.degrees(np.quantile(rot, .99))), 2))
        print('SGS_ARM_FOLLOW '+json.dumps(self.arm_stats), flush=True)

    def _arm_offsets(self, k, top):
        """Arm offsets of resting configurations k at table tops top (linear between grid heights)."""
        x = ((top-self.fit_tops[0])/TABLE_FIT_STEP).clamp(0, len(self.fit_tops)-1)
        i0 = x.floor().long()
        i1 = x.ceil().long()
        w = (x-i0.float())[:, None]
        return (1-w)*self.arm_shift[k, i0]+w*self.arm_shift[k, i1]

    def _fill_reaching(self):
        """Reaching configurations: rho draws (validated with validate_spawns), assets
        balanced and independent of the poses (rho does not couple them)."""
        start, stop = self.blocks['reaching']
        n = stop-start
        perm = torch.randperm(n, device=self.device, generator=self.generator)
        self.cfg['asset'][start:stop] = perm % self.n_assets
        failed = self._draw_valid(torch.arange(start, stop, device=self.device), self.cfg)
        self.filled[start:stop] = True
        self.validated[start:stop] = self.settings.validate_spawns
        self.validated[failed] = False
        self.fill_count['reaching'] = n

    # ------------------------------------------------------- spawn check
    def _write_rho(self, ids, store, k):
        """Write rho-type starts store[k] (robot, table, object at rest, goal) into envs ids."""
        env = self.env
        origin = env.scene.env_origins[ids]
        table = torch.zeros(len(ids), 7, device=self.device)
        table[:, :3] = origin
        table[:, 2] += store['table_z'][k]
        table[:, 3] = 1.
        env.table.write_root_pose_to_sim(table, env_ids=ids)
        env._table_z_per_env[ids] = store['table_z'][k]
        env.robot.write_joint_state_to_sim(store['joint_pos'][k], store['joint_vel'][k], env_ids=ids)
        pose = store['object_pose'][k].clone()
        pose[:, :3] += origin
        env.object.write_root_pose_to_sim(pose, env_ids=ids)
        env.object.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        env._prev_targets[ids] = store['targets'][k]
        env._cur_targets[ids] = store['targets'][k]
        env.robot.reset_setpoint(ids, store['setpoint'][k])
        env._object_init_z[ids] = store['object_init_z'][k]
        goal = store['goal_pose'][k].clone()
        goal[:, :3] += origin
        env.goal_viz.write_root_pose_to_sim(goal, env_ids=ids)
        if self.wholebody:
            self._standing_body(ids, store['targets'][k])

    def _standing_root(self, ids):
        env = self.env
        root = env.robot.data.default_root_state[ids].clone()
        root[:, :3] += env.scene.env_origins[ids]
        root[:, 7:] = 0
        return root

    def _standing_body(self, ids, targets):
        """Whole-body reset of envs ids as env._reset_idx does it: floating base at its
        default root at rest, fresh SONIC history, executed body action from the targets."""
        env = self.env
        env.robot.write_root_state_to_sim(self._standing_root(ids), env_ids=ids)
        env._body_history.reset(ids)
        env._last_body_action[ids] = (targets[:, env._body_ids]-env._body_nominal)/env._body_scales

    def _history_now(self, ids):
        """SONIC history of envs ids including the current step's frame (the frame this
        step's observation pushes, unless a terminal observation already pushed it)."""
        env = self.env
        bufs = [b[ids] for b in env._body_history.buffers]
        if env._body_extra_step != env._sim_step_counter:
            terms = env._body_terms()
            bufs = [torch.cat((b[:, 1:], t[ids, None]), 1) for b, t in zip(bufs, terms)]
        return torch.cat(bufs, -1)

    def _restore_history(self, ids, history):
        """Restore stored histories (newest frame = the restored state). The next
        observation pushes the restored state's frame unless this step already pushed."""
        env = self.env
        if env._body_extra_step != env._sim_step_counter:
            history = torch.cat((history[:, :1], history[:, :-1]), 1)      # the oldest copy is dropped by that push
        for buf, part in zip(env._body_history.buffers, torch.split(history, self.hist_dims, -1)):
            buf[ids] = part
        env._body_history.fresh[ids] = False

    def _object_contact_depth(self):
        """Per world, the deepest object-robot and object-table contact distance of
        the last step (+inf without contact)."""
        import warp as wp
        from dextrah_lab.mj.tactile import OBJECT, ROBOT, TABLE
        env = self.env
        d = env.sim.wp_data
        c = d.contact
        geom = wp.to_torch(c.geom).long()
        world = wp.to_torch(c.worldid).long()
        dist = wp.to_torch(c.dist)
        partner = env.contacts.partner
        active = torch.arange(geom.shape[0], device=self.device) < wp.to_torch(d.nacon)
        active &= (wp.to_torch(c.type) & 1) > 0
        active &= (geom >= 0).all(-1) & (geom < partner.numel()).all(-1) & (world >= 0) & (world < env.num_envs)
        g0, g1 = torch.where(active, geom[:, 0], 0), torch.where(active, geom[:, 1], 0)
        p0, p1 = partner[g0], partner[g1]
        out = []
        for other in (ROBOT, TABLE):
            hit = active & (((p0 == OBJECT) & (p1 == other)) | ((p1 == OBJECT) & (p0 == other)))
            depth = torch.full((env.num_envs,), float('inf'), device=self.device)
            depth.scatter_reduce_(0, world[hit], dist[hit], reduce='amin')
            out.append(depth)
        return out

    @torch.no_grad()
    def _spawn_check(self, cand, assets):
        """Simulate each candidate start for settle_s with the robot holding its
        reset targets (no policy). Invalid: the object starts inside the robot or
        the table (first-step penetration beyond spawn_overlap_*_m; MuJoCo ejects
        it), exceeds spawn_max_speed, or does not end at rest on the table top
        spawn_edge_margin inside its edges (fell or rolled off). Touching the
        resting hand while landing is allowed. Whole body: the floating base is held
        at its standing reset pose (welded, as in the fixed-body task). Uses the env's worlds; the whole
        simulation state is reset afterwards (a world that diverged here must not
        leave e.g. a non-finite solver warm start behind)."""
        import warp as wp
        env, s = self.env, self.settings
        if not hasattr(env, 'contacts'):
            env._init_contacts()
        n_env = env.num_envs
        free, qadr = env.object._free_dof, env.object._free_qadr
        half = torch.tensor((TABLE_SIZE[0]/2-s.spawn_edge_margin, TABLE_SIZE[1]/2-s.spawn_edge_margin), device=self.device)
        result, codes = [], []
        for start in range(0, len(assets), n_env):
            m = min(n_env, len(assets)-start)
            ids = torch.arange(m, device=self.device)
            sub = {k: v[start:start+m] for k, v in cand.items()}
            self.assign_objects(ids, assets[start:start+m])
            self._write_rho(ids, sub, torch.arange(m, device=self.device))
            env.robot.set_joint_position_target(sub['targets'][:, env._position_target_joint_ids],
                                                joint_ids=env._position_target_joint_ids_list, env_ids=ids)
            env.object._wrench = None
            env.sim.data.xfrc_applied.zero_()
            root = None
            if self.wholebody:
                # Every world's base is held (worlds without a candidate stand at the default
                # pose): a free-falling humanoid would lie on the floor with hundreds of
                # constraint rows (nefc overflow, job 1374).
                every = torch.arange(n_env, device=self.device)
                park = every[m:]
                if park.numel():
                    pose = env.robot.data.default_joint_pos[park]
                    env.robot.write_joint_state_to_sim(pose, torch.zeros_like(pose), env_ids=park)
                    env.robot.set_joint_position_target(pose[:, env._position_target_joint_ids],
                                                        joint_ids=env._position_target_joint_ids_list, env_ids=park)
                    env.robot.reset_setpoint(park, pose)
                root = (every, self._standing_root(every))
                env.robot.write_root_state_to_sim(root[1], env_ids=every)
            env.sim.forward()
            vmax = torch.zeros(n_env, device=self.device)
            nefc = torch.zeros(n_env, dtype=torch.long, device=self.device)
            for step in range(max(1, round(s.settle_s/env.physics_dt))):
                if root is not None:
                    env.robot.write_root_state_to_sim(root[1], env_ids=root[0])
                env._write_data_to_sim()
                env.sim.step()
                nefc = torch.maximum(nefc, wp.to_torch(env.sim.wp_data.nefc).long())
                env.object.limit_angular_velocity(env.PHYSX_MAX_ANGULAR_VELOCITY)
                if step == 0:
                    robot_depth, table_depth = self._object_contact_depth()
                    touched = (robot_depth < -s.spawn_overlap_robot_m) | (table_depth < -s.spawn_overlap_table_m)
                vmax = torch.maximum(vmax, torch.nan_to_num(env.sim.data.qvel[:, free:free+3].norm(dim=-1), nan=float('inf')))
            pos = (env.sim.data.qpos[:, qadr:qadr+3]-env.scene.env_origins)[:m]      # COM (body frame at the COM)
            speed = torch.nan_to_num(env.sim.data.qvel[:m, free:free+3].norm(dim=-1), nan=float('inf'))
            top = sub['table_z']+TABLE_SIZE[2]/2
            on_table = (pos[:, 2] > top-.005) & (pos[:, 0].abs() < half[0]) & (pos[:, 1].abs() < half[1])
            touched, vmax = touched[:m], vmax[:m]
            ejected = ~touched & (vmax >= s.spawn_max_speed)
            off = ~touched & ~ejected & ~on_table
            rest = ~touched & ~ejected & on_table & (speed >= s.spawn_rest_speed)
            ok = ~touched & ~ejected & on_table & (speed < s.spawn_rest_speed)
            code = torch.zeros(m, dtype=torch.long, device=self.device)
            for c, mask in ((1, touched), (2, ejected), (3, off), (4, rest)):
                code[mask] = c
            codes.append(code)
            st = self.spawn_stats
            st['max_nefc'] = max(st['max_nefc'], int(nefc[:m].max()))                  # candidate worlds
            st['max_nefc_other'] = max(st['max_nefc_other'], int(nefc[m:].max()) if m < n_env else 0)
            st['checked'] += m
            for key, mask in (('valid', ok), ('spawn_overlap', touched), ('ejected', ejected), ('off_table', off), ('not_at_rest', rest)):
                st[key] += int(mask.sum())
            result.append(ok)
        env.sim.reset()             # clear all simulation state (warm starts, contacts) before training
        self._last_reasons = torch.cat(codes)
        return torch.cat(result)

    def _sync_available(self):
        """A family takes part once complete (or once collection stops)."""
        available = torch.zeros_like(self.filled)
        for f in self.settings.families:
            a, b = self.blocks[f]
            if f not in HARVESTED or self.harvest_done or self.fill_count[f] == b-a:
                available[a:b] = self.filled[a:b]
        if not bool(available.any()):
            raise RuntimeError('SGS: no configuration available (harvested families only, collection not finished?)')
        self.sampler.available.copy_(available)

    # -------------------------------------------------------------- objects
    def assign_objects(self, env_ids, assets):
        """Rewrite the per-world object (model fields + per-env task buffers)."""
        env, tab, m = self.env, self.tab, self.env.sim.model
        if env_ids.numel() == 0:
            return
        rows, cols = env_ids[:, None], self.geoms[None, :]
        m.geom_size[rows, cols] = tab['size'][assets]
        m.geom_pos[rows, cols] = tab['pos'][assets]
        m.geom_quat[rows, cols] = tab['quat'][assets]
        m.geom_rbound[rows, cols] = tab['rbound'][assets]
        m.geom_aabb[rows, cols] = tab['aabb'][assets]
        m.body_mass[env_ids, self.body] = tab['mass'][assets]
        m.body_inertia[env_ids, self.body] = tab['inertia'][assets]
        env.object.com_offset[env_ids] = tab['com'][assets]
        env.object.default_mass[env_ids, 0] = tab['mass'][assets]
        env._object_mass[env_ids, 0] = tab['mass'][assets]
        env._object_asset_index_per_env[env_ids] = assets
        env._object_scale_per_env[env_ids] = tab['scale'][assets]
        env._keypoint_offsets[env_ids] = tab['keypoints'][assets]
        env._bps_features[env_ids] = tab['bps'][assets]

    # ---------------------------------------------------------- env hooks
    def before_reset(self, env_ids):
        """Record finished episodes, choose the next configurations and objects
        (before the task's reset writes poses with the new object's COM)."""
        env, s = self.env, self.settings
        env_ids = env_ids.reshape(-1)
        cfg_ids = self.current[env_ids]
        # Episodes started from a configuration replaced since then are not credited to it.
        active = (cfg_ids >= 0) & (self.slot_version[cfg_ids.clamp_min(0)] == self.current_version[env_ids])
        if bool(active.any()):
            success = env._successes[env_ids[active]] > 0
            self.sampler.record(cfg_ids[active], success)
            fam = self.family[cfg_ids[active]]
            fall = env._termination_reasons['fall'][env_ids[active]]
            robot_fall = env._body_fallen[env_ids[active]] if self.wholebody else None
            if self.table_heights:
                b = torch.bucketize(env._table_z_per_env[env_ids[active]], self.height_edges[1:-1])
                for i, f in enumerate(FAMILIES):
                    if f in self.train_stats:
                        for j in range(HEIGHT_BINS):
                            sel = (fam == i) & (b == j)
                            self.height_train[(f, j)].add(success[sel])
                            self.height_knock[(f, j)].add(fall[sel])
            for i, f in enumerate(FAMILIES):
                if f in self.train_stats:
                    self.train_stats[f].add(success[fam == i])
                    self.fall_stats[f].add(fall[fam == i])
                    if robot_fall is not None:
                        self.robot_fall_stats[f].add(robot_fall[fam == i])
        is_eval = self.eval_mask[env_ids]
        finished_eval = env_ids[is_eval & self.eval_started[env_ids]]
        if finished_eval.numel():
            self.eval_stats['any_goal_success_rate'].add(env._successes[finished_eval] > 0)
            self.eval_stats['goals_reached_count'].add(env._successes[finished_eval])
            self.eval_stats['ever_lifted_rate'].add(env._episode_ever_lifted[finished_eval])
            if self.table_heights:
                z = env._table_z_per_env[finished_eval]
                self._add_by_height(self.height_eval, z, env._successes[finished_eval] > 0)
                self._add_by_height(self.height_eval_knock, z, env._termination_reasons['fall'][finished_eval])
            if self.wholebody:
                self.eval_stats['robot_fall_rate'].add(env._body_fallen[finished_eval])
        # Evaluation: fresh rho sample with a uniformly random asset, or a uniform
        # draw from the pool of validated fresh rho starts.
        eval_ids = env_ids[is_eval]
        if self.eval_pool is None:
            self.assign_objects(eval_ids, torch.randint(0, self.n_assets, (eval_ids.numel(),), device=self.device,
                                                        generator=self.generator))
        else:
            pick = torch.randint(0, len(self.eval_pool['asset']), (eval_ids.numel(),), device=self.device,
                                 generator=self.generator)
            self.eval_current[eval_ids] = pick
            self.assign_objects(eval_ids, self.eval_pool['asset'][pick])
        self.eval_started[eval_ids] = True
        self.current[eval_ids] = -1
        train_ids = env_ids[~is_eval]
        chosen = self.sampler.sample(train_ids.numel(), generator=self.generator)
        if s.hard_object_share > 0 and train_ids.numel():
            hard = torch.rand(train_ids.numel(), device=self.device, generator=self.generator) < s.hard_object_share
            if bool(hard.any()):
                chosen[hard] = self._sample_hard(int(hard.sum()))
        if self.table_heights and train_ids.numel():
            chosen, heights = self._draw_heights(chosen)
            self.pending_table_z[train_ids] = heights
        self.current[train_ids] = chosen
        self.current_version[train_ids] = self.slot_version[chosen]
        self.assign_objects(train_ids, self.cfg['asset'][chosen])

    def after_reset(self, env_ids):
        """Overwrite the task's rho reset with the chosen configurations (whole body:
        called after the env's standing reset)."""
        env, c = self.env, self.cfg
        env_ids = env_ids.reshape(-1)
        if self.eval_pool is not None:
            e = env_ids[self.eval_mask[env_ids]]
            if e.numel():
                self._write_rho(e, self.eval_pool, self.eval_current[e])
        ids = env_ids[~self.eval_mask[env_ids]]
        if ids.numel() == 0:
            return
        k = self.current[ids]
        origin = env.scene.env_origins[ids]
        reach = self.family[k] == FAMILIES.index('reaching')
        dz = self.pending_table_z[ids]-c['table_z'][k] if self.table_heights else torch.zeros(len(ids), device=self.device)
        table = torch.zeros(len(ids), 7, device=self.device)
        table[:, 2], table[:, 3] = c['table_z'][k]+dz, 1.
        env.table.write_root_pose_to_sim(table+torch.cat((origin, torch.zeros_like(table[:, 3:])), -1), env_ids=ids)
        env._table_z_per_env[ids] = c['table_z'][k]+dz
        if self.table_heights:
            b = torch.bucketize(env._table_z_per_env[ids], self.height_edges[1:-1])
            self.height_starts.index_put_((self.family[k], b), torch.ones(len(ids), device=self.device), accumulate=True)
            self.contact_check[ids] = (~reach).long()
        targets, setpoint = c['targets'][k].clone(), c['setpoint'][k].clone()
        if bool(reach.any()):
            r, kr = ids[reach], k[reach]
            joint_pos = c['joint_pos'][kr].clone()
            if self.table_heights:                              # the arm follows the table (hand height above it kept)
                dq = self._arm_offsets(kr, env._table_z_per_env[r]+TABLE_SIZE[2]/2)
                joint_pos[:, self.arm_cols] += dq
                rows = torch.nonzero(reach).flatten()
                targets[rows[:, None], self.arm_cols] += dq
                setpoint[rows[:, None], self.arm_cols] += dq
            env.robot.write_joint_state_to_sim(joint_pos, c['joint_vel'][kr], env_ids=r)
            pose = c['object_pose'][kr].clone()
            pose[:, :3] += env.scene.env_origins[r]
            pose[:, 2] += dz[reach]                             # the resting object moves with the table
            env.object.write_root_pose_to_sim(pose, env_ids=r)
            env.object.write_root_velocity_to_sim(torch.zeros(len(r), 6, device=self.device), env_ids=r)
            if self.wholebody:
                self._standing_body(r, targets[reach])
        held = ~reach
        if bool(held.any()):
            h, kh = ids[held], k[held]
            env.sim.data.qpos[h] = c['qpos'][kh]
            env.sim.data.qvel[h] = c['qvel'][kh]
            if self.na:
                env.sim.data.act[h] = c['act'][kh]
            env._lifted_object[h] = True
            env._episode_ever_lifted[h] = True
            if self.wholebody:
                # Floating base: restored with qpos/qvel. SONIC inputs: history and executed body action.
                self._restore_history(h, c['history'][kh])
                env._last_body_action[h] = (c['targets'][kh][:, env._body_ids]-env._body_nominal)/env._body_scales
        env._prev_targets[ids] = targets
        env._cur_targets[ids] = targets
        env.robot.reset_setpoint(ids, setpoint)
        env._object_init_z[ids] = c['object_init_z'][k]+dz          # lift reference: rest height on this table
        goal = c['goal_pose'][k].clone()
        goal[:, :3] += origin
        goal[:, 2] += torch.where(reach, dz, torch.zeros_like(dz))  # resting starts: goal keeps its height above the table
        env.goal_viz.write_root_pose_to_sim(goal, env_ids=ids)

    def _fits(self, k, table_z):
        """Configurations k fit tables with centre heights table_z: resting starts always (they move with the table);
        held starts when the table top is HELD_TABLE_CLEARANCE below the object's lowest keypoint and their stored
        goal stays inside the goal band above that table."""
        c = self.cfg
        ok = torch.ones(len(k), dtype=torch.bool, device=self.device)
        held = self.family[k] != FAMILIES.index('reaching')
        if bool(held.any()):
            kh = k[held]
            top = table_z[held]+TABLE_SIZE[2]/2
            low = lowest_keypoint_z(c['object_pose'][kh], self.tab['keypoints'][c['asset'][kh]])
            goal = c['goal_pose'][kh, 2]-top
            x = ((top-self.fit_tops[0])/TABLE_FIT_STEP).clamp(0, len(self.fit_tops)-1)
            robot = self.table_fit[kh, x.floor().long()] & self.table_fit[kh, x.ceil().long()]     # both grid neighbours clear
            ok[held] = robot & (top <= low-HELD_TABLE_CLEARANCE) & (goal >= self.goal_band[0]) & (goal <= self.goal_band[1])
        return ok

    def _draw_heights(self, chosen):
        """Table heights first (uniform), then starts that fit them: held starts that do not fit are re-drawn from
        the sampler (HELD_FIT_TRIES times), then replaced by a uniformly drawn available resting start."""
        n = len(chosen)
        heights = self.table_lo+(self.table_hi-self.table_lo)*torch.rand(n, device=self.device, generator=self.generator)
        held = self.family[chosen] != FAMILIES.index('reaching')
        fits = self._fits(chosen, heights)
        self.fit_stats['held_drawn'] += float(held.sum())
        self.fit_stats['held_rejected'] += float((held & ~fits).sum())
        for _ in range(HELD_FIT_TRIES):
            bad = torch.nonzero(~fits).flatten()
            if bad.numel() == 0:
                break
            chosen[bad] = self.sampler.sample(bad.numel(), generator=self.generator)
            fits[bad] = self._fits(chosen[bad], heights[bad])
        bad = torch.nonzero(~fits).flatten()
        if bad.numel():
            a, b = self.blocks.get('reaching', (0, 0))
            pool = a+torch.nonzero(self.sampler.available[a:b]).flatten()
            if pool.numel():
                chosen[bad] = pool[torch.randint(0, len(pool), (bad.numel(),), device=self.device, generator=self.generator)]
            else:                                   # no resting start available: lowest table for these starts
                heights[bad] = self.table_lo
            self.fit_stats['fallback'] += float(bad.numel())
        return chosen, heights

    def _table_xy(self):
        env = self.env
        return (env.table.data.root_pos_w[0, :2]-env.scene.env_origins[0, :2]).tolist()

    def _table_fit_cpu(self, slots):
        """table_fit of the held states in slots (table_fit_mask on their stored qpos)."""
        if slots.numel() == 0:
            return
        m = self.env.sim.mj_model
        mask, stored = table_fit_mask(m, self.cfg['qpos'][slots].cpu().numpy().astype(np.float64), self.fit_tops.cpu().numpy(),
                                      (self.cfg['table_z'][slots]+TABLE_SIZE[2]/2).cpu().numpy(),
                                      self.env._contacts_cfg['robot_bodies'], int(m.body('table/box').id),
                                      HELD_TABLE_CLEARANCE, self._table_xy())
        self.table_fit[slots] = torch.as_tensor(mask, device=self.device)
        self.stored_contact[slots] = torch.as_tensor(stored, device=self.device)
        self.fit_known[slots] = True

    def _fit_check(self, held, limit=4000):
        """Startup check of the fit test: held starts at a random table height they fit (drawn uniformly over the
        range, kept where _fits accepts it; between grid heights) must have no new or deeper robot-table
        penetration than at their stored table (CPU MuJoCo, margin 0). Raises above 1%."""
        if held.numel() == 0:
            return
        c, m = self.cfg, self.env.sim.mj_model
        g = torch.Generator(device=self.device).manual_seed(0)
        z = self.table_lo+(self.table_hi-self.table_lo)*torch.rand(len(held), device=self.device, generator=g)
        fits = self._fits(held, z)
        top = z+TABLE_SIZE[2]/2
        goal = c['goal_pose'][held, 2]-top
        in_band = (goal >= self.goal_band[0]) & (goal <= self.goal_band[1])
        object_ok = top <= lowest_keypoint_z(c['object_pose'][held], self.tab['keypoints'][c['asset'][held]])-HELD_TABLE_CLEARANCE
        sel = torch.nonzero(fits).flatten()
        sel = sel[torch.randperm(len(sel), device=self.device, generator=g)[:limit]]
        contacts = TableContacts(m, self.env._contacts_cfg['robot_bodies'], int(m.body('table/box').id), 0., self._table_xy())
        q = c['qpos'][held[sel]].cpu().numpy().astype(np.float64)
        tops, stored = top[sel].cpu().numpy(), (c['table_z'][held[sel]]+TABLE_SIZE[2]/2).cpu().numpy()
        touching, depth = 0, 0.
        for i in range(len(sel)):
            base = contacts.penetration(q[i], stored[i])
            robot, dist, _ = contacts.at(q[i], tops[i])
            extra = dist-base[robot]
            worse = extra < -PENETRATION_TOL
            touching += int(worse.any())
            depth = min(depth, float(extra[worse].min()) if worse.any() else 0.)
        clear = self.table_fit[held].float().mean(0)
        self.fit_check = dict(held=int(held.numel()), random_height=dict(fits=int(fits.sum()), goal_band_ok=int(in_band.sum()),
                                                                         goal_band_and_object_ok=int((in_band & object_ok).sum())),
                              checked=len(sel), new_or_deeper_penetration=int(touching), deepest_extra_mm=round(-1000*depth, 2),
                              robot_on_table_when_stored=round(float(self.stored_contact[held].float().mean()), 3),
                              robot_clear_share_by_top={f'{t:.2f}': round(v, 3) for t, v in
                                                        zip(self.fit_tops[::4].tolist(), clear[::4].tolist())})
        print('SGS_TABLE_FIT_CHECK '+json.dumps(self.fit_check), flush=True)
        if touching > .01*max(len(sel), 1):
            raise RuntimeError(f'Held starts penetrate the table deeper than when stored at a height they fit: {touching}/{len(sel)}')

    def _check_restore_contact(self):
        """Robot-table contact on the first step after a restore (normal force > RESTORE_CONTACT_N, any robot geom):
        rolling rates of resting starts, of held starts with contact their stored state did not have, and of held
        starts with any contact; warning when either of the first two exceeds 1%."""
        env = self.env
        ids = torch.nonzero(self.contact_check >= 0).flatten()
        kind = self.contact_check[ids]
        self.contact_check[ids] = -1
        force = getattr(env, '_robot_table_force_step', None)
        if force is None:
            return
        touch = force[ids] > RESTORE_CONTACT_N
        k = self.current[ids].clamp_min(0)
        new = touch & ~((kind == 1) & self.stored_contact[k])           # held: contact the stored state did not have
        self.restore_contact[0].add(touch[kind == 0])
        self.restore_contact[1].add(new[kind == 1])
        self.restore_contact[2].add(touch[kind == 1])
        rates = [float(r.mean()) if r.size else 0. for r in self.restore_contact]
        if self.restore_contact[1].size >= 1000 and max(rates[:2]) > .01 and self.restore_contact_warned < 5:
            self.restore_contact_warned += 1
            print(f'SGS_TABLE_WARNING robot-table contact on the first step after a restore: resting {100*rates[0]:.1f}%, '
                  f'held (new vs the stored state) {100*rates[1]:.1f}%, held (any) {100*rates[2]:.1f}%', flush=True)

    def _add_by_height(self, stats, table_z, values):
        b = torch.bucketize(table_z, self.height_edges[1:-1])
        for i, r in enumerate(stats):
            r.add(values[b == i])

    @torch.no_grad()
    def after_step(self):
        """Collect held states (until the families are full) and publish metrics.
        Called after rewards/dones, before done envs are reset."""
        if self.table_heights and bool((self.contact_check >= 0).any()):
            self._check_restore_contact()
        if not self.harvest_done:
            self._harvest()
        elif self.settings.reharvest_every and any(f in self.blocks for f in HARVESTED):
            if self.round_start is None and self.steps > 0 and self.steps % self.settings.reharvest_every == 0:
                self._start_round()
            if self.round_start is not None:
                self._reharvest_step()
        self.steps += 1
        if self.steps % 2000 == 0:
            self._verify_tables()
        self._publish()

    # ------------------------------------------------------------ harvest
    def held_mask(self):
        env, s = self.env, self.settings
        origin = env.scene.env_origins
        obj = env.object.data
        z = obj.root_pos_w[:, 2]-origin[:, 2]
        table_top = env._table_z_per_env+TABLE_SIZE[2]/2
        force = env.touch_force_w.norm(dim=-1)                      # (n, 5) pad forces this step
        grasp = (force[:, THUMB] > s.grasp_force_n) & (torch.cat((force[:, :THUMB], force[:, THUMB+1:]), 1) > s.grasp_force_n).any(1)
        slow = (obj.root_lin_vel_w.norm(dim=-1) < s.max_object_speed) & (obj.root_ang_vel_w.norm(dim=-1) < s.max_object_spin)
        finite = torch.isfinite(env.sim.data.qpos).all(1) & torch.isfinite(env.sim.data.qvel).all(1)
        held = env._lifted_object & grasp & slow & finite & (z > table_top+s.table_clearance_m) & ~env.reset_buf & ~self.eval_mask
        if self.wholebody:
            held &= ~env._body_fallen & ~env._body_numerical_failure & ~env._body_history.fresh
            bound = s.held_max_body_action
            if bound > 0:
                action_history = env._body_history.buffers[3]          # (n, frames, 29) executed body actions
                held &= (action_history.abs().amax((1, 2)) <= bound) & (env._last_body_action.abs().amax(1) <= bound)
        return held

    def in_goal_volume(self):
        rc = self.env.cfg.reset
        lo = torch.tensor(rc.target_volume_mins, device=self.device)
        hi = torch.tensor(rc.target_volume_maxs, device=self.device)
        center, half = (lo+hi)/2, (hi-lo)/2*rc.target_volume_region_scale
        pos = self.env.object.data.root_pos_w-self.env.scene.env_origins
        return ((pos >= center-half) & (pos <= center+half)).all(1)

    def _harvest(self):
        s, env = self.settings, self.env
        self.harvest_steps += 1
        held = self.held_mask()
        for f in HARVESTED:
            if f not in self.blocks:
                continue
            a, b = self.blocks[f]
            need = b-a-self.fill_count[f]
            if need <= 0:
                continue
            cand = held & self.in_goal_volume() if f == 'near_goal' else held
            ids = cand.nonzero().squeeze(-1)
            if ids.numel() == 0:
                continue
            ids = ids[torch.randperm(ids.numel(), device=self.device, generator=self.generator)[:4*s.harvest_per_step]]
            assets = env._object_asset_index_per_env[ids]
            cap = max(1, math.ceil(s.harvest_asset_cap*(b-a)/self.n_assets))
            order = torch.argsort(assets, stable=True)
            sorted_assets = assets[order]
            _, counts = torch.unique_consecutive(sorted_assets, return_counts=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=self.device)-torch.repeat_interleave(torch.cumsum(counts, 0)-counts, counts)
            ok = self.asset_harvest[f][assets]+rank < cap
            ids, assets = ids[ok][:min(s.harvest_per_step, need)], assets[ok][:min(s.harvest_per_step, need)]
            if ids.numel():
                self._store_held(f, ids, assets)
        if self.harvest_steps >= s.harvest_max_steps or all(
                self.fill_count[f] == self.blocks[f][1]-self.blocks[f][0] for f in HARVESTED if f in self.blocks):
            self.harvest_done = True
            print('SGS_HARVEST_DONE '+' '.join(f'{f}={self.fill_count[f]}/{self.blocks[f][1]-self.blocks[f][0]}'
                                                for f in HARVESTED if f in self.blocks)+f' steps={self.harvest_steps}', flush=True)
        self._sync_available()

    def _store_held(self, family, ids, assets, slots=None):
        """Store the held states of envs ids: in the next free slots (initial harvest) or,
        when re-harvesting, in the given slots (fresh lineage, outcome window restarted)."""
        env, c, rc = self.env, self.cfg, self.env.cfg.reset
        a, b = self.blocks[family]
        sequential = slots is None
        if sequential:
            slots = torch.arange(a+self.fill_count[family], a+self.fill_count[family]+len(ids), device=self.device)
        origin = env.scene.env_origins[ids]
        obj_pos = env.object.data.root_pos_w[ids]-origin
        obj_quat = env.object.data.root_quat_w[ids]
        c['asset'][slots] = assets
        c['qpos'][slots], c['qvel'][slots] = env.sim.data.qpos[ids], env.sim.data.qvel[ids]
        if self.na:
            c['act'][slots] = env.sim.data.act[ids]
        c['targets'][slots] = env._cur_targets[ids]
        setpoint = getattr(env.robot, '_setpoint', None)
        c['setpoint'][slots] = (setpoint if setpoint is not None else env._cur_targets)[ids]
        c['table_z'][slots] = env._table_z_per_env[ids]
        c['object_init_z'][slots] = env._object_init_z[ids]
        c['object_pose'][slots] = torch.cat((obj_pos, obj_quat), -1)
        c['joint_pos'][slots], c['joint_vel'][slots] = env.robot.data.joint_pos[ids], env.robot.data.joint_vel[ids]
        if self.wholebody:
            c['history'][slots] = self._history_now(ids)
        n = len(ids)
        if family == 'stable_grasp':
            pos, quat = self.goal_sampling.sample_absolute_goal_pose(
                mins=rc.target_volume_mins, maxs=rc.target_volume_maxs, scale=rc.target_volume_region_scale,
                n_envs=n, device=self.device)
        else:
            frac = torch.rand(n, device=self.device, generator=self.generator)*self.settings.near_goal_max_fraction
            pos, quat = self.goal_sampling.sample_delta_goal_pose(
                prev_pos=obj_pos, prev_quat_wxyz=obj_quat, delta_distance=frac[:, None]*rc.delta_goal_distance,
                delta_rotation_degrees=frac*rc.delta_rotation_degrees, mins=rc.target_volume_mins,
                maxs=rc.target_volume_maxs, scale=rc.target_volume_region_scale)
        c['goal_pose'][slots] = torch.cat((pos, quat), -1)
        self.filled[slots] = True
        if self.table_heights:
            self._table_fit_cpu(slots)                  # new held states: their robot clearance per table height
        if sequential:
            self.fill_count[family] += n
        else:
            self.fill_count[family] = int(self.filled[a:b].sum())
            self.generation[slots], self.parent[slots], self.edited[slots] = 0, -1, False
            self._renew(slots)
        self.asset_harvest[family].index_add_(0, assets, torch.ones_like(assets))
        self._asset_index = None

    # ------------------------------------------- self-evolving held starts
    def _tried(self):
        """Per configuration: outcomes in its window (<= history) and success over them."""
        tried = self.sampler.count.clamp(max=self.sampler.history)
        return tried, self.sampler.outcomes.sum(-1).float()/tried.clamp_min(1).float()

    def _renew(self, slots):
        """New configurations in these slots: empty outcome windows, new version."""
        self.sampler.outcomes[slots] = False
        self.sampler.head[slots] = 0
        self.sampler.count[slots] = 0
        self.slot_version[slots] += 1

    def _start_round(self):
        s = self.settings
        self.round += 1
        self.round_start = self.steps
        self.round_log = dict(round=self.round, step=self.steps)
        tried, rate = self._tried()
        for f in HARVESTED:
            if f not in self.blocks:
                continue
            a, b = self.blocks[f]
            slots = torch.arange(a, b, device=self.device)
            filled, assets, r = self.filled[a:b], self.cfg['asset'][a:b], rate[a:b]
            counts = torch.bincount(assets[filled], minlength=self.n_assets)
            quota = max(1, math.ceil((b-a)/self.n_assets))
            judged = tried[a:b] >= s.reharvest_min_tries
            failed_edit = self.edited[a:b] & judged & (r == 0)
            over = filled & (counts[assets] > quota)
            eligible = ~filled | failed_edit | (over & judged)
            score = 4.*~filled+3.*failed_edit+1.*over+r+1e-3*torch.rand(b-a, device=self.device, generator=self.generator)
            k = min(int(s.reharvest_fraction*(b-a)), int(eligible.sum()))
            chosen = slots[torch.topk(score.masked_fill(~eligible, float('-inf')), k).indices] if k else slots[:0]
            parents_ok = filled & judged & (r >= s.edit_parent_success)
            parents_ok[chosen-a] = False
            n_edit = min(int(round(s.edit_fraction*k)), k) if bool(parents_ok.any()) else 0
            if n_edit:
                # Under-represented objects first (weight 1 / held states of the object).
                weight = parents_ok.float()/counts[assets].clamp_min(1).float()
                parents = slots[torch.multinomial(weight, n_edit, replacement=True, generator=self.generator)]
                self._make_edits(f, chosen[:n_edit], parents)
            self.round_queue[f] = chosen[n_edit:]
            self.round_log[f] = dict(replaced=k, edits=n_edit, harvested=0, failed_edits=int(failed_edit.sum()),
                                     unfilled=int((~filled).sum()), assets_covered_before=int((counts > 0).sum()))
        self._sync_available()

    def _make_edits(self, family, slots, parents):
        """Goal edits: copy the verified held state, new goal (stable grasp: fresh rho goal;
        near goal: a delta goal whose fraction bound grows with the edit generation)."""
        c, rc = self.cfg, self.env.cfg.reset
        for key in c:
            c[key][slots] = c[key][parents]
        gen = self.generation[parents]+1
        n = len(slots)
        if family == 'stable_grasp':
            pos, quat = self.goal_sampling.sample_absolute_goal_pose(
                mins=rc.target_volume_mins, maxs=rc.target_volume_maxs, scale=rc.target_volume_region_scale,
                n_envs=n, device=self.device)
        else:
            bound = self.settings.near_goal_max_fraction*(1.+self.settings.edit_goal_growth*gen.float())
            frac = torch.rand(n, device=self.device, generator=self.generator)*bound
            obj = c['object_pose'][slots]
            pos, quat = self.goal_sampling.sample_delta_goal_pose(
                prev_pos=obj[:, :3], prev_quat_wxyz=obj[:, 3:], delta_distance=frac[:, None]*rc.delta_goal_distance,
                delta_rotation_degrees=frac*rc.delta_rotation_degrees, mins=rc.target_volume_mins,
                maxs=rc.target_volume_maxs, scale=rc.target_volume_region_scale)
        c['goal_pose'][slots] = torch.cat((pos, quat), -1)
        self.generation[slots], self.parent[slots], self.edited[slots] = gen, parents, True
        self.table_fit[slots], self.fit_known[slots] = self.table_fit[parents], self.fit_known[parents]   # same state
        self.stored_contact[slots] = self.stored_contact[parents]
        self.filled[slots] = True
        self._renew(slots)
        self._asset_index = None

    def _reharvest_step(self):
        s, env = self.settings, self.env
        held = self.held_mask()
        for f, queue in self.round_queue.items():
            if queue.numel() == 0:
                continue
            a, b = self.blocks[f]
            cand = held & self.in_goal_volume() if f == 'near_goal' else held
            ids = cand.nonzero().squeeze(-1)
            if ids.numel() == 0:
                continue
            ids = ids[torch.randperm(ids.numel(), device=self.device, generator=self.generator)[:4*s.harvest_per_step]]
            assets = env._object_asset_index_per_env[ids]
            counts = torch.bincount(self.cfg['asset'][a:b][self.filled[a:b]], minlength=self.n_assets)
            quota = max(1, math.ceil((b-a)/self.n_assets))
            cap = max(quota, math.ceil(s.harvest_asset_cap*(b-a)/self.n_assets))
            order = torch.argsort(assets, stable=True)
            _, groups = torch.unique_consecutive(assets[order], return_counts=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=self.device)-torch.repeat_interleave(torch.cumsum(groups, 0)-groups, groups)
            ok = (counts[assets]+rank < cap).nonzero().squeeze(-1)
            deficit = (quota-counts[assets]-rank).float()
            ok = ok[torch.argsort(-deficit[ok], stable=True)]              # objects with the fewest held states first
            m = min(s.harvest_per_step, queue.numel(), ok.numel())
            if m:
                pick = ok[:m]
                self._store_held(f, ids[pick], assets[pick], slots=queue[:m])
                self.round_queue[f] = queue[m:]
                self.round_log[f]['harvested'] += m
        if self.steps-self.round_start+1 >= s.reharvest_steps or all(q.numel() == 0 for q in self.round_queue.values()):
            for f in self.round_queue:
                a, b = self.blocks[f]
                counts = torch.bincount(self.cfg['asset'][a:b][self.filled[a:b]], minlength=self.n_assets)
                self.round_log[f].update(assets_covered_after=int((counts > 0).sum()), unused=int(self.round_queue[f].numel()),
                                         edited_configs=int(self.edited[a:b].sum()), max_generation=int(self.generation[a:b].max()))
            print('SGS_REHARVEST '+json.dumps(self.round_log), flush=True)
            self.round_queue, self.round_start = {}, None
        self._sync_available()

    # ------------------------------------- hardness-weighted object sampling
    def _asset_completion(self):
        """Per object: from-scratch success over its reaching configurations' recent tries
        (objects without tries: the overall mean); refreshed every 200 steps."""
        if self._completion is None or self.steps-self._completion_step >= 200:
            a, b = self.blocks['reaching']
            tried, _ = self._tried()
            assets = self.cfg['asset'][a:b]
            t = torch.zeros(self.n_assets, device=self.device).index_add_(0, assets, tried[a:b].float())
            won = torch.zeros(self.n_assets, device=self.device).index_add_(0, assets, self.sampler.outcomes[a:b].sum(-1).float())
            self._completion = torch.where(t > 0, won/t.clamp_min(1.), won.sum()/t.sum().clamp_min(1.))
            self._completion_step = self.steps
        return self._completion

    def _sample_hard(self, k):
        """k configuration ids: object by hardness (InterMimicGen Eq. 9), then one of its
        available configurations by its SGS probability."""
        s = self.settings
        prob = self.sampler.probabilities().double()
        if self._asset_index is None:
            assets = self.cfg['asset']
            counts = torch.bincount(assets, minlength=self.n_assets)
            self._asset_index = (torch.argsort(assets, stable=True), torch.cumsum(counts, 0)-counts, counts)
        order, starts, counts = self._asset_index
        cum = torch.cumsum(prob[order], 0)
        mass = torch.zeros(self.n_assets, dtype=torch.float64, device=self.device).index_add_(0, self.cfg['asset'], prob)
        has = mass > 0
        w = self._asset_completion().double().clamp(s.hard_object_min, 1.).pow(-s.hard_object_beta)*has
        p = s.hard_object_uniform*has/has.sum()+(1.-s.hard_object_uniform)*w/w.sum()
        assets = torch.multinomial(p.float(), k, replacement=True, generator=self.generator)
        first = starts[assets]
        lo = torch.where(first > 0, cum[(first-1).clamp_min(0)], torch.zeros_like(cum[first]))
        u = lo+torch.rand(k, dtype=torch.float64, device=self.device, generator=self.generator)*mass[assets]
        idx = torch.searchsorted(cum, u, right=True)
        idx = torch.minimum(torch.maximum(idx, first), first+counts[assets]-1)
        self._hard_draws = getattr(self, '_hard_draws', 0)+k
        return order[idx]

    # ------------------------------------------------------------ metrics
    def _publish(self):
        """Scalars for the rl_games observer (0-d tensors, no host sync)."""
        x, p = self.env.extras, self.sampler.success_rate()
        avail = self.sampler.available.float()
        masked_mean = lambda v, m: (v*m).sum()/m.sum().clamp_min(1.)
        for k, r in self.eval_stats.items():
            x[f'sgs/eval_{k}'] = r.mean()
        x['sgs/eval_episodes_in_window'] = float(self.eval_stats['any_goal_success_rate'].size)
        prob = self.sampler.probabilities()
        x['sgs/effective_configs_fraction'] = 1./(prob.square().sum()*avail.sum().clamp_min(1.))
        x['sgs/available_configs'] = avail.sum()
        x['sgs/success_rate_mean'] = masked_mean(p, avail)
        x['sgs/configs_p0_fraction'] = masked_mean((p == 0).float(), avail)
        x['sgs/configs_p1_fraction'] = masked_mean((p == 1).float(), avail)
        for f in self.settings.families:
            a, b = self.blocks[f]
            x[f'sgs/{f}/success_rate_mean'] = masked_mean(p[a:b], avail[a:b])
            x[f'sgs/{f}/sampling_share'] = prob[a:b].sum()
            x[f'sgs/{f}/train_any_goal_success_rate'] = self.train_stats[f].mean()
            x[f'sgs/{f}/train_fall_rate'] = self.fall_stats[f].mean()
            if self.wholebody:
                x[f'sgs/{f}/train_robot_fall_rate'] = self.robot_fall_stats[f].mean()
            x[f'sgs/{f}/configs'] = float(self.fill_count[f])
        if self.table_heights:
            tops = (self.height_edges+TABLE_SIZE[2]/2).tolist()
            share = self.height_starts/self.height_starts.sum(0, keepdim=True).clamp_min(1.)   # start mix within each bin
            for j in range(HEIGHT_BINS):
                name = f'sgs/table_top_{tops[j]:.2f}_{tops[j+1]:.2f}'
                x[f'{name}/eval_success'] = self.height_eval[j].mean()
                x[f'{name}/eval_knockoff'] = self.height_eval_knock[j].mean()
                x[f'{name}/eval_episodes'] = torch.tensor(float(self.height_eval[j].size))   # in the rolling window
                for i, f in enumerate(FAMILIES):
                    if f in self.train_stats:
                        x[f'{name}/{f}/train_success'] = self.height_train[(f, j)].mean()
                        x[f'{name}/{f}/train_knockoff'] = self.height_knock[(f, j)].mean()
                        x[f'{name}/{f}/start_share'] = share[i, j]
            fs = self.fit_stats
            x['sgs/table_heights/held_rejected_first_draw'] = fs['held_rejected']/max(fs['held_drawn'], 1.)
            x['sgs/table_heights/fallback_to_resting'] = fs['fallback']
            x['sgs/table_heights/restore_robot_table_contact_resting'] = self.restore_contact[0].mean()
            x['sgs/table_heights/restore_robot_table_contact_held_new'] = self.restore_contact[1].mean()
            x['sgs/table_heights/restore_robot_table_contact_held_any'] = self.restore_contact[2].mean()
        x['sgs/harvest_done'] = float(self.harvest_done)
        if self.settings.reharvest_every:
            x['sgs/reharvest/rounds'] = float(self.round)
            for f in HARVESTED:
                if f in self.blocks:
                    a, b = self.blocks[f]
                    x[f'sgs/{f}/edited_fraction'] = self.edited[a:b].float().mean()
                    x[f'sgs/{f}/mean_generation'] = self.generation[a:b].float().mean()
        if self.settings.hard_object_share > 0 and self._completion is not None:
            c = self._completion
            x['sgs/hard_object/completion_median'] = c.median()
            x['sgs/hard_object/objects_below_5pct'] = (c < .05).float().mean()

    # ------------------------------------------------------------ state
    def state_dict(self):
        return dict(settings=asdict(self.settings), sampler=self.sampler.state_dict(),
                    cfg={k: v.cpu() for k, v in self.cfg.items()}, filled=self.filled.cpu(),
                    fill_count=dict(self.fill_count), harvest_steps=self.harvest_steps, harvest_done=self.harvest_done,
                    asset_harvest={k: v.cpu() for k, v in self.asset_harvest.items()},
                    n_assets=self.n_assets, nq=self.cfg['qpos'].shape[1], nv=self.cfg['qvel'].shape[1],
                    validated=self.validated.cpu(), wholebody=self.wholebody,
                    lineage=dict(generation=self.generation.cpu(), parent=self.parent.cpu(), edited=self.edited.cpu(),
                                 slot_version=self.slot_version.cpu(), round=self.round),
                    arm_follow=dict(tops=self.fit_tops.cpu(), shift=self.arm_shift.cpu(), known=self.arm_known.cpu()))

    def load_state_dict(self, state):
        saved = SgsSettings(**state['settings']).layout()
        if saved != self.settings.layout():
            raise ValueError(f'SGS configuration layout changed since the checkpoint: {saved} vs {self.settings.layout()}')
        if ((state['n_assets'], state['nq'], state['nv'], state.get('wholebody', False))
                != (self.n_assets, self.cfg['qpos'].shape[1], self.cfg['qvel'].shape[1], self.wholebody)):
            raise ValueError('SGS checkpoint is for another scene')
        self.sampler.load_state_dict(state['sampler'])
        for k, v in state['cfg'].items():
            self.cfg[k].copy_(v.to(self.device))
        self.filled.copy_(state['filled'].to(self.device))
        self.fill_count = dict(state['fill_count'])
        self.harvest_steps, self.harvest_done = state['harvest_steps'], state['harvest_done']
        for k, v in state['asset_harvest'].items():
            self.asset_harvest[k].copy_(v.to(self.device))
        if 'validated' in state:
            self.validated.copy_(state['validated'].to(self.device))
        arm = state.get('arm_follow')                    # cached arm offsets of resting configurations (same grid only)
        if arm is not None and arm['shift'].shape == self.arm_shift.shape and torch.allclose(arm['tops'].to(self.device), self.fit_tops):
            self.arm_shift.copy_(arm['shift'].to(self.device))
            self.arm_known.copy_(arm['known'].to(self.device))
        if 'lineage' in state:                           # older states: everything harvested, generation 0
            for key in ('generation', 'parent', 'edited', 'slot_version'):
                getattr(self, key).copy_(state['lineage'][key].to(self.device))
            self.round = int(state['lineage']['round'])
        if self.settings.validate_spawns and 'reaching' in self.blocks:
            self._revalidate_reaching()
        if self.wholebody and self.settings.held_max_body_action > 0:
            self._sanitize_held()
        self._sync_available()

    def _sanitize_held(self):
        """Replace saved held states whose stored SONIC action history exceeds
        held_max_body_action by goal edits of in-bound held states of the same family."""
        bound, report = self.settings.held_max_body_action, {}
        start = sum(self.hist_dims[:3])
        for f in HARVESTED:
            if f not in self.blocks:
                continue
            a, b = self.blocks[f]
            slots = torch.arange(a, b, device=self.device)
            peak = self.cfg['history'][a:b, :, start:start+self.hist_dims[3]].abs().amax((1, 2))
            bad = self.filled[a:b] & (peak > bound)
            good = self.filled[a:b] & ~bad
            if bool(bad.any()) and bool(good.any()):
                pick = torch.multinomial(good.float(), int(bad.sum()), replacement=True, generator=self.generator)
                self._make_edits(f, slots[bad], slots[pick])
            report[f] = dict(replaced=int(bad.sum()), in_bound=int(good.sum()))
        print('SGS_SANITIZED '+json.dumps(dict(held_max_body_action=bound, **report)), flush=True)

    def _revalidate_reaching(self):
        """Check saved reaching configurations not validated yet; redraw (same asset)
        the ones that fail and clear their outcome windows."""
        a, b = self.blocks['reaching']
        slots = torch.arange(a, b, device=self.device)[~self.validated[a:b]]
        if slots.numel() == 0:
            return
        ok = self._spawn_check({k: self.cfg[k][slots] for k in RHO_FIELDS}, self.cfg['asset'][slots])
        bad = slots[~ok]
        failed = bad[:0]
        if bad.numel():
            failed = self._draw_valid(bad, self.cfg)
            self.sampler.outcomes[bad] = False
            self.sampler.head[bad] = 0
            self.sampler.count[bad] = 0
        self.validated[slots] = True
        self.validated[failed] = False
        print(f'SGS_REVALIDATED reaching={slots.numel()} replaced={bad.numel()}', flush=True)
