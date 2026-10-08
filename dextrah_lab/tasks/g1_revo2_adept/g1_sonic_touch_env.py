"""The existing BPS+touch SAPG manipulation task with a floating SONIC body.

Rewards, goals, success counters, object bank, noise, touch and finger physics
are inherited. This is not the older one-object LiveClipTask diagnostic.
"""
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat
from isaacsimenvs.tasks.play.utils.action_utils import apply_action_pipeline, apply_wrench_dr
from dextrah_lab.wholebody.actuators import body_motors
from dextrah_lab.wholebody.contract import BODY_JOINTS, nominal_body_pose, joint_indices
from dextrah_lab.wholebody.source_actions import apply_wholebody_action, apply_frozen_latent_action
from dextrah_lab.wholebody.body_termination import classify_body_state, combine_terminations
from dextrah_lab.wholebody.source_scene import floating_sonic_robot_scene, ground_contact_partners
from dextrah_lab.wholebody.timed_history import TimedSonicHistory
from dextrah_lab.wholebody.task_contract import TASK_ACTOR_DIM, TASK_CRITIC_DIM, BODY_EXTRA_DIM, BANK_SHA256
from .g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg


@configclass
class SonicBodyCfg:
    controller_mode: str = 'trainable_decoder'  # or frozen_pretrained_latent
    minimum_pelvis_height: float = .4
    minimum_upright_cosine: float = .5
    maximum_joint_speed: float = 1000.
    numerical_failure_mode: str = 'abort'  # diagnostic default; trainer selects reset explicitly
    # False: the table no longer collides with legs/pelvis/torso/left arm, so it
    # cannot support the body; right forearm/hand/fingers still touch it.
    table_supports_body: bool = True
    # Terminate (like a robot fall) when any body link other than the right
    # forearm/hand/fingers presses on the table: no leaning, no penetration.
    body_table_contact_termination: bool = False
    body_table_contact_force_n: float = 2.
    body_table_contact_steps: int = 3
    # True: PhysX parses every environment's own object. The source factorized
    # scene replicates env_0's physics, so all environments simulated env_0's
    # object while BPS/rendering used each env's assigned URDF (job 1096).
    per_env_object_physics: bool = False


@configclass
class G1SonicTouchEnvCfg(G1Revo2TouchEnvCfg):
    sonic_body: SonicBodyCfg = SonicBodyCfg()


class G1SonicTouchEnv(G1Revo2TouchEnv):
    def _extra_touch_partners(self):
        import isaaclab.sim as sim_utils
        paths = ground_contact_partners(sim_utils.get_current_stage())
        print('SONIC_TOUCH_GLOBAL_PARTNERS '+str(paths), flush=True)
        return paths

    def _setup_scene(self):
        clone = self.scene.clone_environments
        def clone_with_table_filter(*args, **kwargs):
            if self.cfg.sonic_body.per_env_object_physics:
                # The source scene requires replicate_physics=True for its own
                # checks; disable replication only for the actual clone so PhysX
                # parses each environment's own object.
                self.scene.cfg.replicate_physics = False
                print('PER_ENV_OBJECT_PHYSICS replicate_physics=False at clone', flush=True)
            if self.cfg.sonic_body.body_table_contact_termination:
                import isaaclab.sim as sim_utils
                from dextrah_lab.wholebody.source_scene import prepare_table_contact_reporting
                self._table_contact_source = prepare_table_contact_reporting(sim_utils.get_current_stage())
                print(f'BODY_TABLE_CONTACT before_clone links={len(self._table_contact_source[1])}', flush=True)
            if not self.cfg.sonic_body.table_supports_body:
                import isaaclab.sim as sim_utils
                from dextrah_lab.wholebody.source_scene import filter_table_body_contacts
                table, filtered, kept = filter_table_body_contacts(sim_utils.get_current_stage())
                self.table_filter_report = dict(table=table, filtered_links=filtered, table_contact_links=kept)
                print(f'TABLE_BODY_FILTER before_clone filtered={len(filtered)} kept={kept}', flush=True)
            return clone(*args, **kwargs)
        self.scene.clone_environments = clone_with_table_filter
        try:
            with floating_sonic_robot_scene():
                super()._setup_scene()
        finally:
            self.scene.clone_environments = clone
        self.body_table_sensors = []
        if self.cfg.sonic_body.body_table_contact_termination:
            if not self.cfg.sonic_body.table_supports_body:
                raise ValueError('Contact termination needs real table collisions')
            from isaaclab.sensors import ContactSensor, ContactSensorCfg
            table, links = self._table_contact_source
            partner = table.replace('env_0', 'env_.*', 1)
            for name, path in links.items():
                sensor = ContactSensor(ContactSensorCfg(prim_path=path.replace('env_0', 'env_.*', 1),
                    update_period=0., filter_prim_paths_expr=[partner]))
                self.scene.sensors[f'body_table_{name}'] = sensor
                self.body_table_sensors.append(sensor)
            self.body_table_names = list(links)
            print(f'BODY_TABLE_CONTACT sensors={len(self.body_table_sensors)} partner={partner} '
                  f'links={self.body_table_names}', flush=True)
        if not self.cfg.sonic_body.table_supports_body and self.num_envs > 1:
            import isaaclab.sim as sim_utils
            from pxr import Usd, UsdPhysics
            stage = sim_utils.get_current_stage()
            last = f'/World/envs/env_{self.num_envs-1}'
            tables = [p for p in Usd.PrimRange(stage.GetPrimAtPath(last+'/Table')) if p.HasAPI(UsdPhysics.FilteredPairsAPI)]
            targets = UsdPhysics.FilteredPairsAPI(tables[0]).GetFilteredPairsRel().GetTargets() if tables else []
            remapped = bool(targets) and all(str(x).startswith(last+'/') for x in targets)
            self.table_filter_report.update(cloned_env=last, cloned_targets=len(targets), cloned_targets_remapped=remapped)
            print(f'TABLE_BODY_FILTER after_clone {last} targets={len(targets)} remapped={remapped}', flush=True)

    def __init__(self, cfg, *, sonic, render_mode=None, **kwargs):
        self._wholebody_ready = False
        self._sonic_reference = sonic
        if cfg.sonic_body.controller_mode not in ('trainable_decoder', 'frozen_pretrained_latent'):
            raise ValueError('Unknown SONIC controller mode')
        self._frozen_latent = cfg.sonic_body.controller_mode == 'frozen_pretrained_latent'
        # Parent deliberately allocates the unchanged 13-action task queues
        # and 249/271 task observations before appending the body interface.
        cfg.action_space = 13
        super().__init__(cfg, render_mode, **kwargs)
        if (cfg.observation_space, cfg.state_space) != (TASK_ACTOR_DIM, TASK_CRITIC_DIM):
            raise ValueError('Expected the trained BPS128 + touch interface')
        if self._bps_manifest['features_sha256'] != BANK_SHA256:
            raise ValueError('Whole-body transfer generated a different object bank')
        names = self.robot.joint_names
        if len(names) != 51:
            raise RuntimeError(f'Full G1+Revo2 must retain 51 joints, found {len(names)}')
        self._body_ids = torch.tensor(joint_indices(names, BODY_JOINTS), device=self.device)
        self._body_nominal = torch.as_tensor(nominal_body_pose(), device=self.device)
        self._body_scales = torch.tensor([m.action_scale for m in body_motors().values()], device=self.device)
        self._body_lower = self.robot.data.joint_pos_limits[:, self._body_ids, 0].clone()
        self._body_upper = self.robot.data.joint_pos_limits[:, self._body_ids, 1].clone()
        # Right arm keeps its source margin; other joints use imported limits.
        for k, index in enumerate(self._arm_joint_ids.tolist()):
            body_index = self._body_ids.tolist().index(index)
            self._body_lower[:, body_index] = self._arm_lower[:, k]
            self._body_upper[:, body_index] = self._arm_upper[:, k]
        self._body_center = (self._body_lower+self._body_upper)/2
        self._body_half_range = (self._body_upper-self._body_lower)/2
        self._last_body_action = torch.zeros(self.num_envs, 29, device=self.device)
        action_dim = 70 if self._frozen_latent else 35
        self._wholebody_action_queue = torch.zeros(self.num_envs, self._action_queue.shape[1], action_dim, device=self.device)
        self._last_latent_action = torch.zeros(self.num_envs, 64, device=self.device)
        self._last_decoded_sonic_action = torch.zeros(self.num_envs, 29, device=self.device)
        self._body_history = TimedSonicHistory(self.num_envs, self.step_dt, self.device)
        self._body_extra_step = None
        self._body_extra = None
        self._body_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._body_table_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_table_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._body_numerical_failure = torch.zeros_like(self._body_fallen)
        self._body_falls_total = torch.zeros((), dtype=torch.long, device=self.device)
        self._body_numerical_failures_total = torch.zeros_like(self._body_falls_total)
        self._body_max_joint_speed = torch.zeros((), device=self.device)
        self._body_checked_transitions = 0
        self._reference_q = self.robot.data.default_joint_pos[:, self._body_ids, None].transpose(1, 2).expand(-1, 10, -1).clone()
        self._reference_qd = torch.zeros_like(self._reference_q)
        self._reference_heading = torch.tensor(cfg.assets.robot_init_rot, device=self.device).expand(self.num_envs, -1)
        cfg.action_space = action_dim
        if self._frozen_latent:
            low = np.r_[np.full(64, -np.inf), np.full(6, -1.)].astype(np.float32)
            high = np.r_[np.full(64, np.inf), np.full(6, 1.)].astype(np.float32)
            self.single_action_space = gym.spaces.Box(low, high)
        else:
            self.single_action_space = gym.spaces.Box(-1., 1., shape=(35,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        cfg.observation_space = TASK_ACTOR_DIM + BODY_EXTRA_DIM
        cfg.state_space = TASK_CRITIC_DIM + BODY_EXTRA_DIM
        for key, size in (('policy', cfg.observation_space), ('critic', cfg.state_space)):
            self.single_observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space['policy'], self.num_envs)
        self.state_space = gym.vector.utils.batch_space(self.single_observation_space['critic'], self.num_envs)
        self._wholebody_ready = True
        print(f'SONIC_SOURCE_TASK_READY actor={cfg.observation_space} critic={cfg.state_space} actions={action_dim} '
              f'controller={cfg.sonic_body.controller_mode} '
              'body_joints=29 total_joints=51 task_hz=60 touch_hz=70 reward=unchanged', flush=True)

    def _pre_physics_step(self, actions):
        if self._frozen_latent:
            apply_frozen_latent_action(self, actions, apply_action_pipeline)
        else:
            apply_wholebody_action(self, actions, apply_action_pipeline)
        apply_wrench_dr(self)

    def sonic_to_policy_body(self, sonic_action):
        targets = self._body_nominal+self._body_scales*sonic_action
        return (targets-self._body_center)/self._body_half_range

    def _body_terms(self):
        data = self.robot.data
        return (data.root_ang_vel_b, data.joint_pos[:, self._body_ids]-self._body_nominal,
                data.joint_vel[:, self._body_ids], self._last_body_action, data.projected_gravity_b)

    @torch.no_grad()
    def _wholebody_observation(self):
        # Called by terminal-critic and ordinary-observation hooks. Advance
        # once per control transition; resets fill ONLY the affected histories.
        if self._body_extra_step != self._sim_step_counter:
            proprio = self._body_history.push(*self._body_terms())
        else:
            self._body_history.initialize_fresh(*self._body_terms())
            proprio = self._body_history.value()
        heading_delta = quat_mul(quat_conjugate(yaw_quat(self.robot.data.root_quat_w)), self._reference_heading)
        ori6 = matrix_from_quat(heading_delta)[:, :, :2].reshape(self.num_envs, 1, 6).expand(-1, 10, -1)
        self._reference_ori6 = ori6
        tokens = self._sonic_reference.reference_tokens(self._reference_q, self._reference_qd, ori6)
        self._body_extra = torch.cat((proprio, tokens), -1)
        self._body_extra_step = self._sim_step_counter
        return self._body_extra

    def _get_observations(self):
        source = super()._get_observations()
        if not self._wholebody_ready:
            return source
        extra = self._wholebody_observation()
        return {k: torch.cat((v, extra), -1) for k, v in source.items()}

    def _get_dones(self):
        data, cfg = self.robot.data, self.cfg.sonic_body
        self._body_fallen, self._body_numerical_failure, speed = classify_body_state(
            data.joint_pos, data.joint_vel, data.root_state_w, data.projected_gravity_b,
            self.scene.env_origins, self.object.data.root_state_w,
            minimum_height=cfg.minimum_pelvis_height,
            minimum_upright=cfg.minimum_upright_cosine,
            maximum_joint_speed=cfg.maximum_joint_speed,
            numerical_failure_mode=cfg.numerical_failure_mode)
        self._body_falls_total += self._body_fallen.sum()
        self._body_numerical_failures_total += self._body_numerical_failure.sum()
        self._body_max_joint_speed = torch.maximum(self._body_max_joint_speed, speed.max())
        self._body_checked_transitions += self.num_envs
        task_terminated, truncated = super()._get_dones()
        if self.body_table_sensors and hasattr(self, '_body_table_count'):
            force = torch.stack([s.data.force_matrix_w[:, 0].norm(dim=-1).sum(-1) for s in self.body_table_sensors], -1)
            pressing = force.amax(-1) > cfg.body_table_contact_force_n
            self._body_table_count = torch.where(pressing, self._body_table_count+1, torch.zeros_like(self._body_table_count))
            self._body_table_contact = self._body_table_count >= cfg.body_table_contact_steps
            self._body_table_force = force
            # Same terminal treatment as a robot fall: a failure, never a time-out bootstrap.
            task_terminated = task_terminated | self._body_table_contact
            truncated = truncated & ~self._body_table_contact
        # A robot fall is NOT the source task's object-fall metric.
        return combine_terminations(task_terminated, truncated, self._body_fallen, self._body_numerical_failure)

    def _get_rewards(self):
        reward = super()._get_rewards()  # exact original manipulation reward
        self.extras['episode_final']['robot_fall'] = self._body_fallen.float()
        self.extras['episode_final']['numerical_failure'] = self._body_numerical_failure.float()
        if self.body_table_sensors and hasattr(self, '_body_table_force'):
            self.extras['episode_final']['done_body_table_contact'] = self._body_table_contact.float()
            touching = self._body_table_force > self.cfg.sonic_body.body_table_contact_force_n
            self.extras['body_table/any_link_touching'] = touching.any(-1).float().mean()
            self.extras['body_table/max_force_n'] = self._body_table_force.amax(-1).mean()
            for i, name in enumerate(self.body_table_names):
                if any(k in name for k in ('knee', 'pelvis', 'hip', 'torso')):
                    self.extras[f'body_table/touch_{name}'] = touching[:, i].float().mean()
        if 'final_observation' in self.extras:
            final = self.extras['final_observation']
            final['critic'] = torch.cat((final['critic'], self._wholebody_observation()), -1)
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not self._wholebody_ready:
            return
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self.scene.env_origins[ids]
        root[:, 7:] = 0
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        # Parent already randomizes ONLY right arm/hand position and velocity;
        # its hold-name cache zeroes velocities for the newly present body.
        self._body_history.reset(ids)
        self._last_body_action[ids] = (self._cur_targets[ids][:, self._body_ids]-self._body_nominal)/self._body_scales
        self._wholebody_action_queue[ids] = 0
        self._last_latent_action[ids] = 0
        self._last_decoded_sonic_action[ids] = 0
        self._body_fallen[ids] = False
        self._body_table_count[ids] = 0
        self._body_table_contact[ids] = False
        self._body_numerical_failure[ids] = False
