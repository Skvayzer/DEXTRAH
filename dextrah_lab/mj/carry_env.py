"""Carry expert C on the mjlab whole body (opt-in: train --mj-carry): hold a grasped object while
frozen SONIC walks. Composition after DexCompose (arXiv:2606.28323), see
dextrah_lab/wholebody/carry_compose.py; port of the Isaac carry env (wholebody/carry_env.py).

Episode (20 s, CarrySettings):
  * start: a holding state of the manipulation expert M (its SGS stable_grasp / near_goal states:
    full MuJoCo state of the world + SONIC history, so the grasp is the one M produced: HANDFUL's
    forward initialisation), object assigned per world, success-guided choice of the state;
    object mass / inertia / COM randomised, the goal parked far away (no object-pose goal);
  * warm-up 0.5 s, no reward, table present: the recurrent state starts from zero, as on M's own SGS
    held starts (rl_games zeroes it on every reset; LSTM state is not part of the stored state);
  * grasp check at the end of the warm-up (object above the table and within 5 cm of its hold
    position in the palm frame), else the episode ends uncredited (start failure);
  * then the table is parked 2 m below the floor (80%; 20% keep it and back off first), settle 1 s,
    random walking clips of the curriculum level with random cuts, standing for the last 2 s.
Observations: the task's own, with the goal-relative keypoints and the goal-surface vectors zeroed
(masked, not faked), palm/object terms in a yaw-only frame that moves with the robot (identity at
the episode start), plus a context block appended after the hand block: goal_active (0), the clip's
command, phase one-hot, the object's velocity relative to the palm in the palm frame (critic only;
zero for the actor) and the Revo2 finger joint torques.
Drops: palm-frame offset beyond the hold by 12 cm for 6 steps, or the object below 25 cm. Robot
falls, numerical failures as in the task. Episodes logged by outcome and phase.
"""
import json
import math

import torch

from .compat import install, p2p
install()                                   # isaaclab.utils.math etc. without Isaac Sim (as mj/env.py)
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat  # noqa: E402
from dextrah_lab.wholebody.carry import MovingFrame, field_slice, quat_from_yaw, transform_clean_observation, yaw_of
from dextrah_lab.wholebody.carry_compose import (LEVELS, RIGHT_ARM_JOINTS, CarrySettings, ClipSequencer, Curriculum,
                                                 WalkingClips, arm_delta_step, body_joint_masks, carry_arm_reference,
                                                 carry_context_v2, carry_reward_v2, carry_reward_v3, compose_body,
                                                 finger_command, hand_command, hand_support, region_distance,
                                                 NetRotation)
from dextrah_lab.wholebody.contract import BODY_JOINTS  # noqa: E402
from .env import G1SonicTouchMjEnv  # noqa: E402

obs_utils = p2p('obs_utils')
TABLE_PARKING_Z = -2.
GOAL_PARKING_Z = 50.
CONTEXT = (('goal_active', 1), ('command', 3), ('phase', 3), ('object_velocity_palm', 6), ('finger_torque', 6))
CONTEXT_DIM = sum(n for _, n in CONTEXT)
# carry v2 (docs/carry_v2_design.md): appended after the v1 context, so a v1 network expands with zero columns.
CONTEXT_V2 = (('object_minus_carry_target', 3), ('right_arm_reference', 7), ('right_arm_delta', 7),
              ('hand_object_contacts', 6), ('supported', 1), ('opposition', 1))   # last two: critic only


def grasp_in_torso(torso, palm, obj_pos):
    """Carry posture lookup inputs: object position in the palm-link frame, palm orientation in the torso
    frame (torso / palm: body states (k, >= 7))."""
    t_po = quat_apply_inverse(palm[:, 3:7], obj_pos-palm[:, :3])
    return t_po, quat_mul(quat_conjugate(torso[:, 3:7]), palm[:, 3:7])


def object_in_torso(torso, obj_pos):
    return quat_apply_inverse(torso[:, 3:7], obj_pos-torso[:, :3])


def context_dim(version=1):
    return CONTEXT_DIM+(sum(n for _, n in CONTEXT_V2) if int(version) == 2 else 0)
OUTCOMES = ('success', 'retain_fail', 'start_fail', 'drop_warmup', 'drop_settle', 'drop_walk', 'drop_final',
            'robot_fall', 'other')


def install_moving_frame_observations():
    """Route the task's clean observations through an env-provided moving frame
    (as wholebody/carry_env.install_moving_frame_observations, for the mjlab obs module)."""
    original = obs_utils._build_clean_observation_dict
    if getattr(original, '_carry_aware', False):
        return

    def build(env, **kwargs):
        obs, context = original(env, **kwargs)
        hook = getattr(env, 'carry_moving_frame', None)
        frame = hook() if hook is not None else None
        return (obs, context) if frame is None else transform_clean_observation(obs, context, frame)

    build._carry_aware = True
    obs_utils._build_clean_observation_dict = build


class _Rolling:
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


class _ExactScore:
    """Best-checkpoint statistic (sgs/best.py interface): curriculum level + exact evaluation success
    rate (StartWindowRate); 'full' once enough evaluation episodes are in its start window."""

    def __init__(self, env, minimum=200):
        self.env, self.minimum = env, minimum
        self.values = [0.]

    @property
    def size(self):
        _, n = self.env._sgs.eval_exact.rate(self.env._env_steps)
        return 1 if n >= self.minimum else 0

    def mean(self):
        r, _ = self.env._sgs.eval_exact.rate(self.env._env_steps)
        return torch.tensor(float(self.env._curriculum.level)+(0. if r != r else r))


class CarryStarts:
    """env._sgs hook: holding-state starts, success-guided over the holding states, outcomes and the
    speed curriculum. Evaluation envs (every ``eval_every``-th) draw holding states uniformly."""

    def __init__(self, env, held, eval_every=16, window=2048, generator=None):
        from dextrah_lab.sgs.sampler import MANIPULATION, SuccessGuidedSampler
        self.env, self.held, self.g = env, held, generator
        c = held.cfg
        fam = held.family
        from dextrah_lab.sgs.mj_task import FAMILIES
        keep = held.filled & ((fam == FAMILIES.index('stable_grasp')) | (fam == FAMILIES.index('near_goal')))
        self.configs = torch.nonzero(keep).flatten()
        if self.configs.numel() == 0:
            raise ValueError('The SGS state has no holding states')
        self.sampler = SuccessGuidedSampler(self.configs.numel(), history=MANIPULATION['history'],
                                            temperature=MANIPULATION['temperature'], target=MANIPULATION['target'],
                                            kappa=MANIPULATION['kappa'], eps=MANIPULATION['eps'], device=env.device)
        n, d = env.num_envs, env.device
        self.current = torch.full((n,), -1, dtype=torch.long, device=d)
        self.eval_mask = torch.zeros(n, dtype=torch.bool, device=d)
        self.eval_mask[::eval_every] = True
        self.window = window
        self.train_stats = {k: _Rolling(window, d) for k in OUTCOMES}
        self.eval_stats = {k: _Rolling(window, d) for k in OUTCOMES}
        self.eval_stats['carry_score'] = _ExactScore(env)
        self.carry_seconds = _Rolling(window, d)
        from dextrah_lab.wholebody.carry_compose import StartWindowRate
        steps = int(round(env.s.max_episode_s()/env.step_dt))
        self.train_exact, self.eval_exact = StartWindowRate(steps, d), StartWindowRate(steps, d)
        self.assets = c['asset'][self.configs]
        self.objects_covered = int(torch.unique(self.assets).numel())

    def before_reset(self, env_ids):
        env = self.env
        ids = env_ids.reshape(-1)
        done = ids[self.current[ids] >= 0]
        if done.numel():
            env._carry_record(done)
        ev = self.eval_mask[ids]
        pick = torch.empty(ids.numel(), dtype=torch.long, device=env.device)
        n_eval = int(ev.sum())
        if n_eval:
            pick[ev] = torch.randint(0, self.configs.numel(), (n_eval,), device=env.device, generator=self.g)
        if ids.numel()-n_eval:
            pick[~ev] = self.sampler.sample(ids.numel()-n_eval, generator=self.g)
        self.current[ids] = pick
        self.held.assign_objects(ids, self.assets[pick])

    def after_reset(self, env_ids):
        self.env._carry_restore(env_ids.reshape(-1), self.configs[self.current[env_ids.reshape(-1)]])

    def after_step(self):
        self.env._carry_publish()

    def state_dict(self):
        return dict(sampler=self.sampler.state_dict(), configs=self.configs.cpu(), curriculum=self.env._curriculum.state_dict())

    def load_state_dict(self, state):
        if not torch.equal(state['configs'], self.configs.cpu()):
            raise ValueError('Carry state was built on other holding states')
        self.sampler.load_state_dict(state['sampler'])
        self.env._curriculum.load_state_dict(state['curriculum'])
        self.env._seq.set_level(self.env._curriculum.level)


class G1CarryMjEnv(G1SonicTouchMjEnv):
    def __init__(self, cfg, *, sonic, carry, holding_state, held_settings, clips_path, seed=None, **kwargs):
        if not getattr(cfg.sonic_body, 'hand_surface_obs', False) or not getattr(cfg.sonic_body, 'hand_surface_goal', False):
            raise ValueError('The carry env masks the goal-surface part of the hand-surface block: enable it')
        install_moving_frame_observations()
        self._carry_ready = False
        # The task config is the manipulation task's and is saved with the run: leave it unchanged.
        # Success counting cannot trigger (the goal is parked 50 m up) and the carry timer ends
        # episodes; the task timeout is moved beyond it on the env, not in the config.
        super().__init__(cfg, sonic=sonic, seed=seed, **kwargs)
        self.max_episode_length = math.ceil((carry.max_episode_s()+5.)/self.step_dt)
        from dextrah_lab.sgs.mj_task import MjSgsTask
        self.s = carry
        n, d = self.num_envs, self.device
        self._carry_generator = torch.Generator(device=d).manual_seed(int(cfg.seed if seed is None else seed)+7)
        state = torch.load(holding_state, map_location='cpu', weights_only=False)
        self._held = MjSgsTask(self, held_settings, generator=self._carry_generator, state=state)
        self._sgs = CarryStarts(self, self._held, generator=self._carry_generator)
        self.clips = WalkingClips.load(clips_path, d)
        self._seq = ClipSequencer(self.clips, carry, n, d, generator=self._carry_generator)
        if carry.promote_on not in ('uniform', 'train'):
            raise ValueError(f'promote_on must be uniform or train, not {carry.promote_on!r}')
        gate_envs = int(self._sgs.eval_mask.sum()) if carry.promote_on == 'uniform' else n
        self._curriculum = Curriculum(carry, d, num_envs=n, episode_steps=int(round(carry.max_episode_s()/self.step_dt)),
                                      min_starts=gate_envs)
        self._standing_q = self._reference_q.clone()
        self._arm_ids = torch.tensor([i for i, name in enumerate(BODY_JOINTS)
                                      if any(p in name for p in ('shoulder', 'elbow', 'wrist'))], device=d)
        self._residual_mask = body_joint_masks(BODY_JOINTS, carry.residual_joints).to(d)
        self._wrist_body = int(self.sim.mj_model.body('robot/right_wrist_yaw_link').id)
        # Observation slices: goal-relative keypoints (task part), goal-surface vectors (hand block).
        sizes = obs_utils._obs_field_sizes(13, 5)
        if sum(sizes[f] for f in cfg.obs.obs_list) != 92 or sum(sizes[f] for f in cfg.obs.state_list) != 114:
            raise ValueError('Unexpected source task observation layout')
        from dextrah_lab.wholebody.hand_surface import HAND_SURFACE_DIM
        from dextrah_lab.wholebody.task_contract import BODY_EXTRA_DIM, TASK_ACTOR_DIM, TASK_CRITIC_DIM
        self._goal_policy = field_slice(cfg.obs.obs_list, sizes, 'keypoints_rel_goal')
        self._goal_critic = field_slice(cfg.obs.state_list, sizes, 'keypoints_rel_goal')
        h = self.hand_surface_dim
        self._goal_surface_policy = slice(TASK_ACTOR_DIM+BODY_EXTRA_DIM+h-HAND_SURFACE_DIM, TASK_ACTOR_DIM+BODY_EXTRA_DIM+h)
        self._goal_surface_critic = slice(TASK_CRITIC_DIM+BODY_EXTRA_DIM+h-HAND_SURFACE_DIM, TASK_CRITIC_DIM+BODY_EXTRA_DIM+h)
        # Context block after the hand block (the network treats it as more hand inputs).
        self._v2 = carry.version == 2
        self._context_dim = context_dim(carry.version)
        self.hand_surface_dim += self._context_dim
        cfg.observation_space += self._context_dim
        cfg.state_space += self._context_dim
        import gymnasium as gym
        import numpy as np
        for key, size in (('policy', cfg.observation_space), ('critic', cfg.state_space)):
            self.single_observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        z = lambda *shape, **kw: torch.zeros(n, *shape, device=d, **kw)
        self._carry_t = z(dtype=torch.long)
        self._q_ref = z(6)
        self._arm_ref = z(len(self._arm_ids))
        self._ref_offset = z(3)                                   # object position in the palm frame at the start
        self._home_pos, self._home_yaw = z(3), z()
        self._capture = torch.ones(n, dtype=torch.bool, device=d)
        self._keep_table = torch.zeros(n, dtype=torch.bool, device=d)
        self._table_parked = torch.zeros(n, dtype=torch.bool, device=d)
        self._drop_count = z(dtype=torch.long)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=d)
        self._start_fail = torch.zeros(n, dtype=torch.bool, device=d)
        self._start_level = z(dtype=torch.long)                  # curriculum level the episode started at
        self._start_step = z(dtype=torch.long)                   # env step the episode started at
        self._env_steps = 0
        self._curriculum.burn_in_until = int(round(carry.max_episode_s()/self.step_dt))
        self._offset_error = z()
        self._finger_residual, self._latent_residual = z(6), z(64)
        self._ref_quat = z(10, 4); self._ref_quat[..., 0] = 1.
        self._planned_velocity, self._clip_command = z(3), z(3)
        self._blend_q, self._blend_t0 = self._standing_q.clone(), z()
        self._push_steps, self._push_force = z(dtype=torch.long), z(3)
        self._rel_vel_palm = z(6)
        self._body_compose = self._compose_body if carry.stage == 2 else None
        if self._v2:
            self._init_v2(z)
        self._carry_ready = True
        print('CARRY_ENV_READY '+json.dumps(dict(envs=n, holding_states=int(self._sgs.configs.numel()),
              objects=self._sgs.objects_covered, clips=len(self.clips.names), level=self._curriculum.level,
              stage=carry.stage, version=carry.version, context_dim=self._context_dim, actor=cfg.observation_space,
              critic=cfg.state_space)), flush=True)

    def _init_v2(self, z):
        """Carry v2 buffers (docs/carry_v2_design.md)."""
        s, d = self.s, self.device
        self.enable_grasp_reward(0.)          # object-only finger / palm contact forces + opposition (logs only)
        self._postures = None
        if s.arm_channel:
            from dextrah_lab.wholebody.carry_posture import CarryPostures
            self._postures = CarryPostures(s.postures, d, target=s.carry_target, target_jitter=s.carry_jitter)
        self._right_arm = torch.tensor([BODY_JOINTS.index(j) for j in RIGHT_ARM_JOINTS], device=d)
        names = self.robot.body_names
        self._torso_body, self._palm_link = names.index('torso_link'), names.index('right_base_link')
        self._pelvis_mj = int(self.sim.mj_model.body('robot/pelvis').id)
        self._object_mj = int(self._held.body)
        self._q_grasp, self._q_carry, self._arm_D, self._arm_now = z(7), z(7), z(7), z(7)
        self._carry_target = z(3)
        self._unsupported = z(dtype=torch.long)
        self._supported = torch.ones(self.num_envs, dtype=torch.bool, device=d)
        self._rel_smooth = z(6)
        self._prev_raw, self._action_delta = z(13), z(13)
        self._n_contacts, self._opposition, self._contact_flags = z(dtype=torch.long), z(), z(6)
        self._region_d, self._obj_torso = z(), z(3)
        self._pelvis_next, self._pelvis_steps, self._pelvis_force = z(), z(dtype=torch.long), z(3)
        self._object_steps, self._object_force = z(dtype=torch.long), z(3)
        self._hazard = torch.zeros(4, device=d)      # decaying sums: early drops, early held s, late drops, late held s
        # net palm-relative rotation rate over slip_window_s (the 'net' angular slip signal; logged in both modes)
        self._net_rot = NetRotation(self.num_envs, max(1, round(s.slip_window_s/self.step_dt)), self.step_dt, d)
        self._net_rate = z()

    def _init_contacts(self):
        super()._init_contacts()
        if getattr(self, '_v2', False):
            self.contacts.object_table = True
            self.contacts.object_ground = True

    # ------------------------------------------------------------------ frames
    def _palm(self):
        palm = self.robot.data.body_state_w[:, self._palm_body_id, :]
        centre = obs_utils._apply_local_offset(palm[:, :3], palm[:, 3:7], self._palm_center_offset, (self.num_envs,))
        return centre, palm

    def _palm_frame(self, v, palm_quat):
        """World vector -> palm frame."""
        from isaaclab.utils.math import quat_apply_inverse
        return quat_apply_inverse(palm_quat, v)

    def carry_moving_frame(self):
        if not getattr(self, '_carry_ready', False):
            return None
        data = self.robot.data
        return MovingFrame(root_pos=data.root_pos_w-self.scene.env_origins, root_yaw=yaw_of(data.root_quat_w),
                           home_pos=self._home_pos, home_yaw=self._home_yaw)

    def _hold_kinematics(self):
        centre, palm = self._palm()
        obj = self.object.data
        offset_w = obj.root_pos_w-centre                  # hold position: object link vs palm centre
        # Velocities are at the body COMs (mjlab body_com_vel_w / root_com_vel_w), so the rigid-attachment
        # prediction uses the COM positions (the link/centre offset added w x (COM - link) spuriously:
        # ~0.1-0.3 m/s for a hammer at walking palm rotation rates, job 1556).
        palm_com = self.robot.data._d.body_com_pos_w[:, self.robot.body_perm][:, self._palm_body_id]
        obj_com = self.object.free_qpos[:, :3]            # object body frame is at its COM
        rel_linear = obj.root_lin_vel_w-(palm[:, 7:10]+torch.cross(palm[:, 10:13], obj_com-palm_com, dim=-1))
        rel_angular = obj.root_ang_vel_w-palm[:, 10:13]
        q = palm[:, 3:7]
        return self._palm_frame(offset_w, q), self._palm_frame(rel_linear, q), self._palm_frame(rel_angular, q)

    def _capture_episode_refs(self):
        ids = torch.nonzero(self._capture).flatten()
        if ids.numel() == 0:
            return
        offset, _, _ = self._hold_kinematics()
        self._ref_offset[ids] = offset[ids]
        data = self.robot.data
        self._home_pos[ids] = (data.root_pos_w-self.scene.env_origins)[ids]
        self._home_yaw[ids] = yaw_of(data.root_quat_w[ids])
        if getattr(self, '_v2', False) and self._postures is not None:
            # Carry posture for THIS grasp: object (own palm offset) nearest the carry target, palm kept.
            torso = data.body_state_w[ids, self._torso_body]
            palm = data.body_state_w[ids, self._palm_link]
            t_po, palm_torso = grasp_in_torso(torso, palm, self.object.data.root_pos_w[ids])
            q_now = data.joint_pos[ids][:, self._body_ids][:, self._right_arm]
            target = self._postures.sample_targets(len(ids), self._carry_generator)
            _, q_c, _ = self._postures.select(t_po, palm_torso, q_now, target)
            self._q_grasp[ids], self._q_carry[ids], self._carry_target[ids] = q_now, q_c, target
        self._capture[ids] = False

    # ------------------------------------------------------------------ resets
    def _carry_restore(self, ids, k):
        """Holding state k into envs ids (after the standing reset); per-episode randomisation."""
        if ids.numel() == 0:
            return
        held, c = self._held, self._held.cfg
        origin = self.scene.env_origins[ids]
        table = torch.zeros(len(ids), 7, device=self.device)
        table[:, :3] = origin
        table[:, 2] += c['table_z'][k]
        table[:, 3] = 1.
        self.table.write_root_pose_to_sim(table, env_ids=ids)
        self._table_z_per_env[ids] = c['table_z'][k]
        self.sim.data.qpos[ids] = c['qpos'][k]
        self.sim.data.qvel[ids] = c['qvel'][k]
        if held.na:
            self.sim.data.act[ids] = c['act'][k]
        self._lifted_object[ids] = True
        self._episode_ever_lifted[ids] = True
        held._restore_history(ids, c['history'][k])
        targets = c['targets'][k]
        self._last_body_action[ids] = (targets[:, self._body_ids]-self._body_nominal)/self._body_scales
        self._prev_targets[ids] = targets
        self._cur_targets[ids] = targets
        self.robot.reset_setpoint(ids, c['setpoint'][k])
        self._object_init_z[ids] = c['object_init_z'][k]
        goal = torch.zeros(len(ids), 7, device=self.device)
        goal[:, :3] = origin
        goal[:, 2] += GOAL_PARKING_Z
        goal[:, 3] = 1.
        self.goal_viz.write_root_pose_to_sim(goal, env_ids=ids)
        # Hold references (DexCompose q_ref; arm angles for SONIC's reference).
        self._q_ref[ids] = hand_command(targets[:, self._hand_joint_ids], self._hand_lower[ids], self._hand_upper[ids])
        self._arm_ref[ids] = c['joint_pos'][k][:, self._body_ids][:, self._arm_ids]
        # Object mass / inertia (log-uniform scale) and COM randomisation.
        s, m, g = self.s, self.sim.model, self._carry_generator
        body = held.body
        lo, hi = math.log(s.mass_scale[0]), math.log(s.mass_scale[1])
        scale = torch.exp(lo+(hi-lo)*torch.rand(len(ids), device=self.device, generator=g))
        asset = self._object_asset_index_per_env[ids]
        base = held.tab['mass'][asset]
        mass = torch.minimum(base*scale, torch.clamp(base, min=s.mass_max_kg))
        scale = mass/base
        m.body_mass[ids, body] = mass
        m.body_inertia[ids, body] = held.tab['inertia'][asset]*scale[:, None]
        if s.com_offset_m > 0:
            # COM shift delta (object link frame) without touching body_ipos (MuJoCo flags bodies with
            # their inertial frame at the body origin as "simple" at compile time; a runtime body_ipos
            # gave non-finite states on the first step, job 1545). The body frame stays at the COM:
            # the geometry moves by -delta in the body frame and the body origin by +R delta, so the
            # geometry stays where the holding state had it; com_offset keeps the reported link pose.
            from isaaclab.utils.math import quat_apply
            delta = (torch.rand(len(ids), 3, device=self.device, generator=g)*2-1)*s.com_offset_m
            rows, cols = ids[:, None], held.geoms[None, :]
            m.geom_pos[rows, cols] = held.tab['pos'][asset]-delta[:, None]
            self.object.com_offset[ids] = held.tab['com'][asset]+delta
            qadr = self.object._free_qadr
            quat = self.sim.data.qpos[ids, qadr+3:qadr+7]
            self.sim.data.qpos[ids, qadr:qadr+3] += quat_apply(quat, delta)
        self.object.default_mass[ids, 0] = mass
        self._object_mass[ids, 0] = mass
        # Episode state.
        self._carry_t[ids] = 0
        self._capture[ids] = True
        self._keep_table[ids] = torch.rand(len(ids), device=self.device, generator=g) < s.table_keep_probability
        self._table_parked[ids] = False
        self._drop_count[ids] = 0
        self._dropped[ids] = False
        self._start_fail[ids] = False
        self._start_level[ids] = self._curriculum.level
        self._start_step[ids] = self._env_steps
        self._finger_residual[ids] = 0.
        self._latent_residual[ids] = 0.
        self._push_steps[ids] = 0
        self._push_force[ids] = 0.
        root_yaw = yaw_of(c['qpos'][k][:, 3:7])
        end = None
        if s.walk_s:
            lo, hi = s.walk_s
            end = s.warmup_s+s.settle_s+lo+(hi-lo)*torch.rand(len(ids), device=self.device, generator=g)+s.final_stand_s
        self._seq.reset(ids, root_yaw, self._keep_table[ids], end=end)
        if getattr(self, '_v2', False):
            self._arm_D[ids] = 0.
            self._q_grasp[ids] = c['joint_pos'][k][:, self._body_ids][:, self._right_arm]
            self._q_carry[ids] = self._q_grasp[ids]              # until the posture is chosen at capture
            self._unsupported[ids] = 0
            self._supported[ids] = True
            self._rel_smooth[ids] = 0.
            self._net_rot.reset(ids)
            self._net_rate[ids] = 0.
            self._prev_raw[ids] = 0.
            lo, hi = s.pelvis_push_interval_s
            self._pelvis_next[ids] = s.warmup_s+s.settle_s+lo+(hi-lo)*torch.rand(len(ids), device=self.device, generator=g)
            self._pelvis_steps[ids] = 0
            self._object_steps[ids] = 0
        reference = self._standing_q[ids].clone()
        if self.s.arm_reference == 'grasp':
            reference[:, :, self._arm_ids] = self._arm_ref[ids][:, None]
        self._reference_q[ids] = reference
        self._reference_qd[ids] = 0.
        self._blend_q[ids] = reference
        self._blend_t0[ids] = 0.

    def _park_tables(self, ids):
        if ids.numel() == 0:
            return
        pose = torch.zeros(len(ids), 7, device=self.device)
        pose[:, :3] = self.scene.env_origins[ids]
        pose[:, 2] += TABLE_PARKING_Z
        pose[:, 3] = 1.
        self.table.write_root_pose_to_sim(pose, env_ids=ids)
        self._table_parked[ids] = True

    # ------------------------------------------------------- walking reference
    def _episode_time(self):
        return self._carry_t.float()*self.step_dt

    def _update_reference(self):
        t = self._episode_time()
        yaw = yaw_of(self.robot.data.root_quat_w)
        changed = self._seq.update(t, yaw)
        if changed.numel():
            self._blend_q[changed] = self._reference_q[changed]
            self._blend_t0[changed] = t[changed]
        q, qd, root, velocity, command = self._seq.reference(t)
        clip = (self._seq.clip != self._seq.idle)
        q = torch.where(clip[:, None, None], q, self._standing_q)
        qd = torch.where(clip[:, None, None], qd, torch.zeros_like(qd))
        heading = quat_from_yaw(self._seq.heading)[:, None].expand(-1, 10, -1)
        root = torch.where(clip[:, None, None], root, heading)
        w = ((t-self._blend_t0)/self.s.blend_s).clamp(0, 1)[:, None, None]
        q = self._blend_q+(q-self._blend_q)*w
        qd = qd*w
        if self.s.arm_reference == 'grasp':
            q[:, :, self._arm_ids] = self._arm_ref[:, None]
        else:
            q[:, :, self._arm_ids] = self._standing_q[:, :, self._arm_ids]
        if getattr(self, '_v2', False) and self.s.arm_channel:
            # Carry v2: right arm from the grasp-time angles to the carry posture (minimum jerk over the
            # secure phase), plus the policy's low-pass-filtered bounded change.
            self._arm_now = carry_arm_reference(self._q_grasp, self._q_carry, self._arm_D, t, self.s.warmup_s, self.s.settle_s)
            q[:, :, self._right_arm] = self._arm_now[:, None]
        qd[:, :, self._arm_ids] = 0.
        self._reference_q, self._reference_qd, self._ref_quat = q, qd, root
        self._planned_velocity = torch.where(clip[:, None], velocity, torch.zeros_like(velocity))
        self._clip_command = torch.where(clip[:, None], command, torch.zeros_like(command))

    @torch.no_grad()
    def _wholebody_observation(self):
        if not getattr(self, '_carry_ready', False):
            return super()._wholebody_observation()
        self._update_reference()
        if self._body_extra_step != self._sim_step_counter:
            proprio = self._body_history.push(*self._body_terms())
        else:
            self._body_history.initialize_fresh(*self._body_terms())
            proprio = self._body_history.value()
        robot_yaw = yaw_quat(self.robot.data.root_quat_w)
        delta = quat_mul(quat_conjugate(robot_yaw)[:, None].expand(-1, 10, -1).reshape(-1, 4), self._ref_quat.reshape(-1, 4))
        ori6 = matrix_from_quat(delta.reshape(self.num_envs, 10, 4))[..., :2].reshape(self.num_envs, 10, 6)
        self._reference_ori6 = ori6
        if self._fast_step:
            tokens, self._sonic_latent = self._sonic_reference.reference_tokens_and_latent(
                self._reference_q, self._reference_qd, ori6)
            self._sonic_latent_ori6 = ori6
        else:
            tokens = self._sonic_reference.reference_tokens(self._reference_q, self._reference_qd, ori6)
        self._body_extra = torch.cat((proprio, tokens), -1)
        self._body_extra_step = self._sim_step_counter
        return self._body_extra

    # ---------------------------------------------------------------- actions
    def _compose_body(self, body, residual):
        """Stage 2 ownership: bare SONIC (zero residual) on the legs/waist, the residual decode on the arms."""
        zero = torch.zeros_like(residual)
        if self._fast_step and getattr(self, '_sonic_latent_ori6', None) is self._reference_ori6:
            bare = self._sonic_reference.decode_latent(self._body_extra[:, :930], self._sonic_latent, zero)
        else:
            bare = self._sonic_reference(self._body_extra[:, :930], self._reference_q, self._reference_qd,
                                         self._reference_ori6, zero)
        self._bare_body = bare
        return compose_body(bare, body, self._residual_mask)

    def _pre_physics_step(self, actions):
        if not getattr(self, '_carry_ready', False):
            return super()._pre_physics_step(actions)
        s = self.s
        if getattr(self, '_v2', False):
            fingers = finger_command(self._q_ref, actions[:, 64:70], s.beta_fingers, s.finger_close, s.finger_open)
            self._finger_residual = fingers-self._q_ref
            raw = torch.cat((torch.tanh(actions[:, :7]), torch.tanh(actions[:, 64:70])), -1)
            self._action_delta = raw-self._prev_raw
            self._prev_raw = raw
            if s.arm_channel:
                self._arm_D = arm_delta_step(self._arm_D, raw[:, :7], s)
            self._latent_residual = torch.zeros_like(actions[:, :64])
            super()._pre_physics_step(torch.cat((self._latent_residual, fingers), -1))
            self._v2_disturbances()
            return
        self._finger_residual = s.beta_fingers*torch.tanh(actions[:, 64:70])
        fingers = (self._q_ref+self._finger_residual).clamp(-1, 1)
        if s.stage == 2:
            self._latent_residual = s.beta_latent*torch.tanh(actions[:, :64])
        else:
            self._latent_residual = torch.zeros_like(actions[:, :64])
        super()._pre_physics_step(torch.cat((self._latent_residual, fingers), -1))
        # Random pushes to the right wrist (world-frame force at the body COM, held for 0.1-0.2 s).
        g = self._carry_generator
        start = (self._push_steps == 0) & (torch.rand(self.num_envs, device=self.device, generator=g) < s.push_probability)
        start &= self._episode_time() >= s.warmup_s
        ids = torch.nonzero(start).flatten()
        if ids.numel():
            lo, hi = s.push_s
            self._push_steps[ids] = torch.round((lo+(hi-lo)*torch.rand(len(ids), device=self.device, generator=g))/self.step_dt).long().clamp(min=1)
            self._push_force[ids] = torch.randn(len(ids), 3, device=self.device, generator=g)*s.push_force_n
        active = (self._push_steps > 0).float()[:, None]
        self.sim.data.xfrc_applied[:, self._wrist_body, :3] = self._push_force*active
        self._push_steps = (self._push_steps-1).clamp(min=0)

    def _v2_disturbances(self):
        """Carry v2 pushes, scaled by the disturbance level k: right wrist (as v1), pelvis (Workhorse
        protocol as the ceiling) and object (RobustDexGrasp's evaluation force), after the secure phase."""
        s, g, d, dt = self.s, self._carry_generator, self.device, self.step_dt
        k = s.disturbance
        t = self._episode_time()
        walking_on = t >= s.warmup_s+s.settle_s
        n = self.num_envs
        rand = lambda *shape: torch.rand(*shape, device=d, generator=g)

        def horizontal(m):
            a = rand(m)*2*math.pi
            return torch.stack((a.cos(), a.sin(), torch.zeros_like(a)), -1)
        # wrist (v1 mechanism), magnitude x k
        start = (self._push_steps == 0) & (rand(n) < s.push_probability) & walking_on
        ids = torch.nonzero(start).flatten()
        if ids.numel() and k > 0:
            lo, hi = s.push_s
            self._push_steps[ids] = torch.round((lo+(hi-lo)*rand(len(ids)))/dt).long().clamp(min=1)
            self._push_force[ids] = torch.randn(len(ids), 3, device=d, generator=g)*s.push_force_n*k
        # pelvis: every push_interval, horizontal, U(0, F k) for U(push_s)
        start = (self._pelvis_steps == 0) & (t >= self._pelvis_next) & walking_on
        ids = torch.nonzero(start).flatten()
        if ids.numel():
            lo, hi = s.pelvis_push_s
            self._pelvis_steps[ids] = torch.round((lo+(hi-lo)*rand(len(ids)))/dt).long().clamp(min=1)
            self._pelvis_force[ids] = horizontal(len(ids))*(rand(len(ids))*s.pelvis_push_n*k)[:, None]
            lo, hi = s.pelvis_push_interval_s
            self._pelvis_next[ids] = t[ids]+lo+(hi-lo)*rand(len(ids))
        # object: random direction, U(0, F k) for 0.1-0.3 s
        start = (self._object_steps == 0) & (rand(n) < s.object_push_probability) & walking_on
        ids = torch.nonzero(start).flatten()
        if ids.numel() and k > 0:
            v = torch.randn(len(ids), 3, device=d, generator=g)
            self._object_steps[ids] = torch.round((.1+.2*rand(len(ids)))/dt).long().clamp(min=1)
            self._object_force[ids] = v/v.norm(dim=-1, keepdim=True).clamp_min(1e-6)*(rand(len(ids))*s.object_push_n*k)[:, None]
        x = self.sim.data.xfrc_applied
        x[:, self._wrist_body, :3] = self._push_force*(self._push_steps > 0).float()[:, None]
        x[:, self._pelvis_mj, :3] = self._pelvis_force*(self._pelvis_steps > 0).float()[:, None]
        x[:, self._object_mj, :3] = self._object_force*(self._object_steps > 0).float()[:, None]
        self._push_steps = (self._push_steps-1).clamp(min=0)
        self._pelvis_steps = (self._pelvis_steps-1).clamp(min=0)
        self._object_steps = (self._object_steps-1).clamp(min=0)

    # ------------------------------------------------------------ dones
    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        if not getattr(self, '_carry_ready', False):
            return terminated, truncated
        s = self.s
        self._carry_t += 1
        t = self._episode_time()
        offset, rel_lin, rel_ang = self._hold_kinematics()
        self._rel_vel_palm = torch.cat((rel_lin, rel_ang), -1)
        self._offset_error = (offset-self._ref_offset).norm(dim=-1)
        far = self._offset_error > s.drop_excess_m
        self._drop_count = torch.where(far, self._drop_count+1, torch.zeros_like(self._drop_count))
        height = self.object.data.root_pos_w[:, 2]-self.scene.env_origins[:, 2]
        self._dropped = (self._drop_count >= s.drop_steps) | (height < s.drop_floor_m)
        if self._v2:
            # Supported by the hand: >= 1 hand-object contact (pad or palm > 0.5 N), within 15 cm of the
            # palm centre, no object-table / object-floor contact. Drop = unsupported for unsupported_s.
            c = self._last_contacts
            zero = torch.zeros_like(height)
            centre, _ = self._palm()
            self._contact_flags, self._n_contacts, self._supported = hand_support(
                c['finger_object_force_l1'], c['palm_object_force_l1'], c.get('object_table_force', zero),
                c.get('object_ground_force', zero), (self.object.data.root_pos_w-centre).norm(dim=-1))
            counting = t > s.warmup_s
            self._unsupported = torch.where(counting & ~self._supported, self._unsupported+1, torch.zeros_like(self._unsupported))
            if s.hold_mode == 'support':
                self._dropped = (self._unsupported >= int(round(s.unsupported_s/self.step_dt))) | (height < s.drop_floor_m)
            a = min(1., self.step_dt/s.slip_smooth_s)
            self._rel_smooth = self._rel_smooth+a*(self._rel_vel_palm-self._rel_smooth)
            _, palm = self._palm()
            self._net_rate = self._net_rot.step(quat_mul(quat_conjugate(palm[:, 3:7]), self.object.data.root_quat_w))
        # Grasp check at the end of the warm-up; then park the table (unless kept): v1 right after the check,
        # v2 after the secure phase (the arm moves to the carry posture over the table).
        warm = int(round(s.warmup_s/self.step_dt))
        at_check = self._carry_t == warm
        lifted = height > self._table_z_per_env+.15+s.confirm_lift_m      # table top = centre + 0.15
        confirmed = lifted & (self._offset_error < s.confirm_distance_m)
        self._start_fail = at_check & ~confirmed
        if self._v2:
            park = (self._carry_t == int(round((s.warmup_s+s.settle_s)/self.step_dt))) & ~self._keep_table & ~self._dropped
        else:
            park = at_check & confirmed & ~self._keep_table
        self._park_tables(torch.nonzero(park).flatten())
        terminated = terminated | self._dropped | self._start_fail
        end = torch.round(self._seq.end/self.step_dt).long() if self._seq.variable else int(round(s.episode_s/self.step_dt))
        truncated = (self._carry_t >= end) & ~terminated
        return terminated, truncated

    # ---------------------------------------------------------- rewards
    def _get_rewards(self):
        super()._get_rewards()          # task bookkeeping and terminal observations; the task reward is not used
        if not getattr(self, '_carry_ready', False):
            return torch.zeros(self.num_envs, device=self.device)
        s = self.s
        data = self.robot.data
        yaw = yaw_of(data.root_quat_w)
        c, sn = yaw.cos(), yaw.sin()
        v = data.root_lin_vel_w
        body_velocity = torch.stack((c*v[:, 0]+sn*v[:, 1], -sn*v[:, 0]+c*v[:, 1], data.root_ang_vel_w[:, 2]), -1)
        table_contact = getattr(self, '_body_table_contact', torch.zeros_like(self._dropped)) & ~self._table_parked
        active = self._episode_time() > s.warmup_s
        if self._v2:
            torso = data.body_state_w[:, self._torso_body]
            self._obj_torso = object_in_torso(torso, self.object.data.root_pos_w)
            self._region_d = region_distance(self._obj_torso, s.region_lo, s.region_hi)
            logs = getattr(self, '_grasp_logs', None) or {}
            self._opposition = logs.get('C_opp', torch.zeros_like(self._region_d))
            secure = self._seq.phase(self._episode_time()) == ClipSequencer.SETTLE
            # angular slip: smoothed |w_rel| ('ema') or the net rotation rate over the window ('net', a 1-vector)
            angular = self._rel_smooth[:, 3:] if s.slip_angular == 'ema' else self._net_rate[:, None]
            reward, terms = carry_reward_v3(s, supported=self._supported, rel_linear=self._rel_smooth[:, :3],
                rel_angular=angular, grasp_opposition=self._opposition, contacts=self._n_contacts,
                region_d=self._region_d, secure=secure, body_velocity=body_velocity,
                planned_velocity=self._planned_velocity, normal_n=self.touch_raw[..., 0],
                action_delta=self._action_delta, arm_delta=self._arm_D, table_contact=table_contact,
                dropped=self._dropped, fell=self._body_fallen, active=active)
            self._v2_metrics(active)
        else:
            reward, terms = carry_reward_v2(s, offset_error=self._offset_error, rel_linear=self._rel_vel_palm[:, :3],
                rel_angular=self._rel_vel_palm[:, 3:], body_velocity=body_velocity, planned_velocity=self._planned_velocity,
                finger_residual=self._finger_residual, latent_residual=self._latent_residual,
                normal_n=self.touch_raw[..., 0], dropped=self._dropped, fell=self._body_fallen,
                table_contact=table_contact, active=active)
        if self._body_penalty is not None:
            # Opt-in SONIC action / fall penalties (enable_body_penalty), on the composed body action.
            extra, more, _ = self._body_penalty()
            reward = reward+extra*active.float()
            terms.update({k: v*active.float() for k, v in more.items()})
        self._reward_terms = dict(terms, total_reward=reward)
        cumulative = self.extras.setdefault('episode_cumulative', {})
        for name, value in terms.items():
            cumulative['carry_'+name] = value
        final = self.extras.get('final_observation')
        if isinstance(final, dict) and 'critic' in final:
            final['critic'] = self._carry_critic(final['critic'])
        walking = self._seq.phase(self._episode_time()) == ClipSequencer.WALK
        if bool(walking.any()):
            err = (body_velocity[:, :2]-self._planned_velocity[:, :2]).norm(dim=-1)
            self.extras['carry/walk_velocity_error'] = err[walking].mean()
            self.extras['carry/walk_planned_speed'] = self._planned_velocity[walking, :2].norm(dim=-1).mean()
            self.extras['carry/walk_body_speed'] = body_velocity[walking, :2].norm(dim=-1).mean()
        self.extras['carry/offset_error_cm'] = 100*self._offset_error[active].mean() if bool(active.any()) else self._offset_error.mean()*0
        if bool(active.any()):
            lin, ang = self._rel_vel_palm[:, :3].norm(dim=-1), self._rel_vel_palm[:, 3:].norm(dim=-1)
            self.extras['carry/slip_linear_ms'] = lin[active].mean()
            self.extras['carry/slip_angular_rads'] = ang[active].mean()
            held = active & (self._offset_error < .03)      # holding (within 3 cm of the hold position)
            if bool(held.any()):
                self.extras['carry/slip_linear_held_median'] = lin[held].median()
                self.extras['carry/slip_angular_held_median'] = ang[held].median()
                self.extras['carry/slip_angular_held_p90'] = ang[held].quantile(.9)
        self.extras['carry/finger_residual_abs'] = self._finger_residual.abs().mean()
        self.extras['carry/latent_residual_abs'] = self._latent_residual.abs().mean()
        self.extras['carry/table_parked'] = self._table_parked.float().mean()
        return reward

    def _v2_metrics(self, active):
        """Carry v2 logs: drop hazard per held second in early (< 5 s) and late (> 10 s) walking (decaying
        sums, time constant ~17 s), support, contacts, opposition, carry-region distance, arm change."""
        s, dt = self.s, self.step_dt
        t = self._episode_time()
        walking = self._seq.phase(t) == ClipSequencer.WALK
        walk_t = t-self._seq.walk_start()
        early, late = walking & (walk_t < 5.), walking & (walk_t > 10.)
        drop = self._dropped.float()
        held = self._supported.float()
        now = torch.stack(((drop*early).sum(), (held*early).sum()*dt, (drop*late).sum(), (held*late).sum()*dt))
        self._hazard = .999*self._hazard+now
        h = self._hazard
        self.extras['carry/hazard_early_per_s'] = h[0]/h[1].clamp_min(1e-6)
        self.extras['carry/hazard_late_per_s'] = h[2]/h[3].clamp_min(1e-6)
        if bool(active.any()):
            self.extras['carry/supported'] = self._supported[active].float().mean()
            self.extras['carry/hand_object_contacts'] = self._n_contacts[active].float().mean()
            self.extras['carry/opposition'] = self._opposition[active].mean()
            self.extras['carry/region_distance_cm'] = 100*self._region_d[active].mean()
            self.extras['carry/slip_smoothed_angular_rads'] = self._rel_smooth[active, 3:].norm(dim=-1).mean()
            self.extras['carry/slip_net_rotation_rads'] = self._net_rate[active].mean()
        self.extras['carry/arm_delta_abs'] = self._arm_D.abs().mean()

    # ------------------------------------------------------ observations
    def _context(self, critic):
        phase = self._seq.phase(self._episode_time())
        one_hot = torch.stack(((phase <= ClipSequencer.SETTLE), phase == ClipSequencer.WALK, phase == ClipSequencer.FINAL), -1).float()
        velocity = self._rel_vel_palm if critic else torch.zeros_like(self._rel_vel_palm)
        torque = self.robot.data.applied_torque[:, self._hand_joint_ids].clamp(-5, 5)
        goal_active = torch.zeros(self.num_envs, 1, device=self.device)
        ctx = torch.cat((goal_active, self._clip_command, one_hot, velocity, torque), -1)
        if not getattr(self, '_v2', False):
            return ctx
        # v2 (appended): object - carry target (torso frame), right-arm reference, filtered arm change,
        # hand-object contacts (5 pads + palm); critic only: supported, opposition.
        priv = torch.stack((self._supported.float(), self._opposition), -1) if critic else torch.zeros(self.num_envs, 2, device=self.device)
        return torch.cat((ctx, carry_context_v2(self._obj_torso, self._carry_target, self._arm_now, self._arm_D,
                                                self._contact_flags, priv)), -1)

    def _carry_critic(self, critic):
        critic = critic.clone()
        critic[:, self._goal_critic] = 0
        critic[:, self._goal_surface_critic] = 0
        return torch.cat((critic, self._context(True)), -1)

    def _get_observations(self):
        if not getattr(self, '_carry_ready', False):
            return super()._get_observations()
        self._capture_episode_refs()
        obs = super()._get_observations()
        policy = obs['policy'].clone()
        policy[:, self._goal_policy] = 0
        policy[:, self._goal_surface_policy] = 0
        return dict(policy=torch.cat((policy, self._context(False)), -1), critic=self._carry_critic(obs['critic']))

    # ------------------------------------------------------ outcomes / metrics
    def _carry_record(self, ids):
        """Outcome of the finished episodes ids (before their reset)."""
        s, starts = self.s, self._sgs
        t = self._carry_t[ids].float()*self.step_dt
        dropped, fell, start_fail = self._dropped[ids], self._body_fallen[ids], self._start_fail[ids]
        if self._seq.variable:
            timeout = self._carry_t[ids] >= torch.round(self._seq.end[ids]/self.step_dt).long()
        else:
            timeout = self._carry_t[ids] >= int(round(s.episode_s/self.step_dt))
        if getattr(self, '_v2', False) and s.hold_mode == 'support':
            retained = self._supported[ids]          # v2: still supported by the hand (any re-grip / posture)
        else:
            retained = self._offset_error[ids] < s.retain_m
        success = timeout & retained & ~fell & ~dropped
        phase = self._seq.phase(t, ids)
        out = {
            'success': success, 'retain_fail': timeout & ~retained & ~fell & ~dropped,
            'start_fail': start_fail & ~dropped,
            'drop_warmup': dropped & (phase == ClipSequencer.WARMUP),
            'drop_settle': dropped & (phase == ClipSequencer.SETTLE),
            'drop_walk': dropped & (phase == ClipSequencer.WALK),
            'drop_final': dropped & (phase == ClipSequencer.FINAL),
            'robot_fall': fell & ~dropped}
        out['other'] = ~torch.stack(list(out.values())).any(0)
        ev = starts.eval_mask[ids]
        credited = ~start_fail          # start failures do not count against the holding state
        k = starts.current[ids]
        tr = ~ev & credited
        if bool(tr.any()):
            starts.sampler.record(k[tr], success[tr])
        gate = (ev if s.promote_on == 'uniform' else ~ev) & credited & (self._start_level[ids] == self._curriculum.level)
        if bool(gate.any()):
            if self._curriculum.record_exact(self._start_step[ids][gate], success[gate], self._env_steps):
                self._seq.set_level(self._curriculum.level)
                print(f'CARRY_LEVEL {self._curriculum.level} max_speed={LEVELS[self._curriculum.level]}', flush=True)
        for name, value in out.items():
            starts.train_stats[name].add(value[~ev])
            starts.eval_stats[name].add(value[ev])
        starts.train_exact.add(self._start_step[ids][tr], success[tr])
        evc = ev & credited
        starts.eval_exact.add(self._start_step[ids][evc], success[evc])
        starts.carry_seconds.add(t)

    def _carry_publish(self):
        self._env_steps += 1                # after_step: once per env step
        starts = self._sgs
        for name in OUTCOMES:
            self.extras[f'carry/train_{name}'] = starts.train_stats[name].mean()
            self.extras[f'carry/eval_{name}'] = starts.eval_stats[name].mean()
        self.extras['carry/eval_score'] = starts.eval_stats['carry_score'].mean().to(self.device)
        self.extras['carry/level'] = torch.tensor(float(self._curriculum.level), device=self.device)
        self.extras['carry/level_success'] = torch.tensor(self._curriculum.rate(self._env_steps), device=self.device)
        for name, rate in (('train', starts.train_exact), ('eval', starts.eval_exact)):
            r, n = rate.rate(self._env_steps)
            if n:
                self.extras[f'carry/{name}_success_exact'] = torch.tensor(r, device=self.device)
        self.extras['carry/episode_seconds'] = starts.carry_seconds.mean()

    def carry_contract(self):
        return dict(type='g1_frozen_sonic_carry_expert_mjlab_v1', settings=self.s.as_dict(), levels=list(LEVELS),
                    holding_states=int(self._sgs.configs.numel()), objects=self._sgs.objects_covered,
                    clips=self.clips.names, context=[dict(name=k, dim=v) for k, v in CONTEXT],
                    composition='DexCompose (arXiv:2606.28323): fingers = clamp(q_ref + beta_A tanh(a)), latent = '
                                'beta_B tanh(a) (stage 2; 0 in stage 1), legs/waist from bare SONIC, arms from the residual decode',
                    reference='planner legs/waist (walking clips), arms at the holding state angles' if self.s.arm_reference == 'grasp'
                              else 'planner legs/waist, trained standing arms',
                    goal='none: goal parked 50 m up; goal-relative keypoints and goal-surface vectors zeroed',
                    observation_frame='palm/object terms in a yaw-only frame moving with the robot (identity at the episode start)',
                    table='parked 2 m below the floor after the grasp check (keep probability %.2f)' % self.s.table_keep_probability,
                    lstm='not stored with holding states; zero at every start as on M\'s own SGS held starts; 0.5 s reward-free warm-up')
