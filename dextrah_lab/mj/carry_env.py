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
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat  # noqa: E402
from dextrah_lab.wholebody.carry import MovingFrame, field_slice, quat_from_yaw, transform_clean_observation, yaw_of
from dextrah_lab.wholebody.carry_compose import (LEVELS, CarrySettings, ClipSequencer, Curriculum, WalkingClips,
                                                 body_joint_masks, carry_reward_v2, compose_body, finger_command,
                                                 hand_command)
from dextrah_lab.wholebody.contract import BODY_JOINTS  # noqa: E402
from .env import G1SonicTouchMjEnv  # noqa: E402

obs_utils = p2p('obs_utils')
TABLE_PARKING_Z = -2.
GOAL_PARKING_Z = 50.
CONTEXT = (('goal_active', 1), ('command', 3), ('phase', 3), ('object_velocity_palm', 6), ('finger_torque', 6))
CONTEXT_DIM = sum(n for _, n in CONTEXT)
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
        self.eval_stats = {k: _Rolling(window, d) for k in OUTCOMES+('carry_score',)}
        self.carry_seconds = _Rolling(window, d)
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
        self.max_episode_length = math.ceil((carry.episode_s+5.)/self.step_dt)
        from dextrah_lab.sgs.mj_task import MjSgsTask
        self.s = carry
        n, d = self.num_envs, self.device
        self._carry_generator = torch.Generator(device=d).manual_seed(int(cfg.seed if seed is None else seed)+7)
        state = torch.load(holding_state, map_location='cpu', weights_only=False)
        self._held = MjSgsTask(self, held_settings, generator=self._carry_generator, state=state)
        self._sgs = CarryStarts(self, self._held, generator=self._carry_generator)
        self.clips = WalkingClips.load(clips_path, d)
        self._seq = ClipSequencer(self.clips, carry, n, d, generator=self._carry_generator)
        self._curriculum = Curriculum(carry, d, num_envs=n)
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
        self.hand_surface_dim += CONTEXT_DIM
        cfg.observation_space += CONTEXT_DIM
        cfg.state_space += CONTEXT_DIM
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
        self._curriculum.burn_in_until = int(round(carry.episode_s/self.step_dt))
        self._offset_error = z()
        self._finger_residual, self._latent_residual = z(6), z(64)
        self._ref_quat = z(10, 4); self._ref_quat[..., 0] = 1.
        self._planned_velocity, self._clip_command = z(3), z(3)
        self._blend_q, self._blend_t0 = self._standing_q.clone(), z()
        self._push_steps, self._push_force = z(dtype=torch.long), z(3)
        self._rel_vel_palm = z(6)
        self._body_compose = self._compose_body if carry.stage == 2 else None
        self._carry_ready = True
        print('CARRY_ENV_READY '+json.dumps(dict(envs=n, holding_states=int(self._sgs.configs.numel()),
              objects=self._sgs.objects_covered, clips=len(self.clips.names), level=self._curriculum.level,
              stage=carry.stage, context_dim=CONTEXT_DIM, actor=cfg.observation_space, critic=cfg.state_space)), flush=True)

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
        self._seq.reset(ids, root_yaw, self._keep_table[ids])
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
        # Grasp check at the end of the warm-up; then park the table (unless kept).
        warm = int(round(s.warmup_s/self.step_dt))
        at_check = self._carry_t == warm
        lifted = height > self._table_z_per_env+.15+s.confirm_lift_m      # table top = centre + 0.15
        confirmed = lifted & (self._offset_error < s.confirm_distance_m)
        self._start_fail = at_check & ~confirmed
        park = at_check & confirmed & ~self._keep_table
        self._park_tables(torch.nonzero(park).flatten())
        terminated = terminated | self._dropped | self._start_fail
        truncated = (self._carry_t >= int(round(s.episode_s/self.step_dt))) & ~terminated
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
            self.extras['carry/slip_linear_ms'] = self._rel_vel_palm[active, :3].norm(dim=-1).mean()
            self.extras['carry/slip_angular_rads'] = self._rel_vel_palm[active, 3:].norm(dim=-1).mean()
        self.extras['carry/finger_residual_abs'] = self._finger_residual.abs().mean()
        self.extras['carry/latent_residual_abs'] = self._latent_residual.abs().mean()
        self.extras['carry/table_parked'] = self._table_parked.float().mean()
        return reward

    # ------------------------------------------------------ observations
    def _context(self, critic):
        phase = self._seq.phase(self._episode_time())
        one_hot = torch.stack(((phase <= ClipSequencer.SETTLE), phase == ClipSequencer.WALK, phase == ClipSequencer.FINAL), -1).float()
        velocity = self._rel_vel_palm if critic else torch.zeros_like(self._rel_vel_palm)
        torque = self.robot.data.applied_torque[:, self._hand_joint_ids].clamp(-5, 5)
        goal_active = torch.zeros(self.num_envs, 1, device=self.device)
        return torch.cat((goal_active, self._clip_command, one_hot, velocity, torque), -1)

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
        timeout = self._carry_t[ids] >= int(round(s.episode_s/self.step_dt))
        retained = self._offset_error[ids] < s.retain_m
        success = timeout & retained & ~fell & ~dropped
        phase = self._seq.phase(t)
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
            same = tr & (self._start_level[ids] == self._curriculum.level) & (self._start_step[ids] >= self._curriculum.burn_in_until)
            if self._curriculum.record(success[same]):
                self._seq.set_level(self._curriculum.level)
                self._curriculum.burn_in_until = self._env_steps+int(round(s.episode_s/self.step_dt))
                print(f'CARRY_LEVEL {self._curriculum.level} max_speed={LEVELS[self._curriculum.level]}', flush=True)
        for name, value in out.items():
            starts.train_stats[name].add(value[~ev])
            starts.eval_stats[name].add(value[ev])
        starts.eval_stats['carry_score'].add((success[ev].float()+self._curriculum.level))
        starts.carry_seconds.add(t)

    def _carry_publish(self):
        self._env_steps += 1                # after_step: once per env step
        starts = self._sgs
        for name in OUTCOMES:
            self.extras[f'carry/train_{name}'] = starts.train_stats[name].mean()
            self.extras[f'carry/eval_{name}'] = starts.eval_stats[name].mean()
        self.extras['carry/eval_score'] = starts.eval_stats['carry_score'].mean()
        self.extras['carry/level'] = torch.tensor(float(self._curriculum.level), device=self.device)
        self.extras['carry/level_success'] = torch.tensor(self._curriculum.rate(), device=self.device)
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
