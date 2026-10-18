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

    def __post_init__(self):
        self.families = tuple(self.families)
        unknown = set(self.families)-set(FAMILIES)
        if unknown or not self.families or len(set(self.families)) != len(self.families):
            raise ValueError(f'SGS families must be distinct members of {FAMILIES}, got {self.families}')
        if self.num_configs < len(self.families):
            raise ValueError('Fewer configurations than families')
        if set(self.families) & set(HARVESTED) and 'reaching' not in self.families:
            raise ValueError('Held states are collected from reaching episodes: keep the reaching family')

    def layout(self):
        """Fields that fix the saved configuration set and outcome windows."""
        return dict(num_configs=self.num_configs, history=self.history, families=tuple(self.families))


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
        self.fill_count = {f: 0 for f in settings.families}
        self.harvest_steps, self.harvest_done = 0, not any(f in HARVESTED for f in settings.families)
        self.asset_harvest = {f: torch.zeros(self.n_assets, dtype=torch.long, device=self.device) for f in HARVESTED}
        # Per-env bookkeeping.
        self.current = torch.full((n_env,), -1, dtype=torch.long, device=self.device)
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
        self.steps = 0
        self.validated = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.spawn_stats = dict(checked=0, valid=0, spawn_overlap=0, ejected=0, off_table=0, not_at_rest=0)
        if state is not None:
            self.load_state_dict(state)
        elif 'reaching' in settings.families:
            self._fill_reaching()
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
                        aabb=t(bank.aabb).view(self.n_assets, 4, 2, 3), mass=t(bank.mass), com=t(bank.com),
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
            ok = self._spawn_check(cand, store['asset'][todo]) if s.validate_spawns else \
                torch.ones(len(todo), dtype=torch.bool, device=self.device)
            keep = todo[ok]
            for key, value in cand.items():
                store[key][keep] = value[ok]
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
            env.sim.forward()
            vmax = torch.zeros(n_env, device=self.device)
            root = self._standing_root(ids) if self.wholebody else None
            for step in range(max(1, round(s.settle_s/env.physics_dt))):
                if root is not None:
                    env.robot.write_root_state_to_sim(root, env_ids=ids)
                env._write_data_to_sim()
                env.sim.step()
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
        active = cfg_ids >= 0
        if bool(active.any()):
            success = env._successes[env_ids[active]] > 0
            self.sampler.record(cfg_ids[active], success)
            fam = self.family[cfg_ids[active]]
            fall = env._termination_reasons['fall'][env_ids[active]]
            robot_fall = env._body_fallen[env_ids[active]] if self.wholebody else None
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
        self.current[train_ids] = chosen
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
        table = torch.zeros(len(ids), 7, device=self.device)
        table[:, 2], table[:, 3] = c['table_z'][k], 1.
        env.table.write_root_pose_to_sim(table+torch.cat((origin, torch.zeros_like(table[:, 3:])), -1), env_ids=ids)
        env._table_z_per_env[ids] = c['table_z'][k]
        reach = self.family[k] == FAMILIES.index('reaching')
        if bool(reach.any()):
            r, kr = ids[reach], k[reach]
            env.robot.write_joint_state_to_sim(c['joint_pos'][kr], c['joint_vel'][kr], env_ids=r)
            pose = c['object_pose'][kr].clone()
            pose[:, :3] += env.scene.env_origins[r]
            env.object.write_root_pose_to_sim(pose, env_ids=r)
            env.object.write_root_velocity_to_sim(torch.zeros(len(r), 6, device=self.device), env_ids=r)
            if self.wholebody:
                self._standing_body(r, c['targets'][kr])
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
        env._prev_targets[ids] = c['targets'][k]
        env._cur_targets[ids] = c['targets'][k]
        env.robot.reset_setpoint(ids, c['setpoint'][k])
        env._object_init_z[ids] = c['object_init_z'][k]
        goal = c['goal_pose'][k].clone()
        goal[:, :3] += origin
        env.goal_viz.write_root_pose_to_sim(goal, env_ids=ids)

    @torch.no_grad()
    def after_step(self):
        """Collect held states (until the families are full) and publish metrics.
        Called after rewards/dones, before done envs are reset."""
        if not self.harvest_done:
            self._harvest()
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

    def _store_held(self, family, ids, assets):
        env, c, rc = self.env, self.cfg, self.env.cfg.reset
        a, _ = self.blocks[family]
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
        self.fill_count[family] += n
        self.asset_harvest[family].index_add_(0, assets, torch.ones_like(assets))

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
        x['sgs/harvest_done'] = float(self.harvest_done)

    # ------------------------------------------------------------ state
    def state_dict(self):
        return dict(settings=asdict(self.settings), sampler=self.sampler.state_dict(),
                    cfg={k: v.cpu() for k, v in self.cfg.items()}, filled=self.filled.cpu(),
                    fill_count=dict(self.fill_count), harvest_steps=self.harvest_steps, harvest_done=self.harvest_done,
                    asset_harvest={k: v.cpu() for k, v in self.asset_harvest.items()},
                    n_assets=self.n_assets, nq=self.cfg['qpos'].shape[1], nv=self.cfg['qvel'].shape[1],
                    validated=self.validated.cpu(), wholebody=self.wholebody)

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
        if self.settings.validate_spawns and 'reaching' in self.blocks:
            self._revalidate_reaching()
        self._sync_available()

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
