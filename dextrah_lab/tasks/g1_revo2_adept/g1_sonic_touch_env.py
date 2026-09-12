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
from dextrah_lab.wholebody.source_actions import apply_wholebody_action
from dextrah_lab.wholebody.source_scene import floating_sonic_robot_scene
from dextrah_lab.wholebody.timed_history import TimedSonicHistory
from dextrah_lab.wholebody.task_contract import TASK_ACTOR_DIM, TASK_CRITIC_DIM, BODY_EXTRA_DIM, BANK_SHA256
from .g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg


@configclass
class SonicBodyCfg:
    minimum_pelvis_height: float = .4
    minimum_upright_cosine: float = .5


@configclass
class G1SonicTouchEnvCfg(G1Revo2TouchEnvCfg):
    sonic_body: SonicBodyCfg = SonicBodyCfg()


class G1SonicTouchEnv(G1Revo2TouchEnv):
    def _setup_scene(self):
        with floating_sonic_robot_scene():
            super()._setup_scene()

    def __init__(self, cfg, *, sonic, render_mode=None, **kwargs):
        self._wholebody_ready = False
        self._sonic_reference = sonic
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
        self._wholebody_action_queue = torch.zeros(self.num_envs, self._action_queue.shape[1], 35, device=self.device)
        self._body_history = TimedSonicHistory(self.num_envs, self.step_dt, self.device)
        self._body_extra_step = None
        self._body_extra = None
        self._body_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reference_q = self.robot.data.default_joint_pos[:, self._body_ids, None].transpose(1, 2).expand(-1, 10, -1).clone()
        self._reference_qd = torch.zeros_like(self._reference_q)
        self._reference_heading = torch.tensor(cfg.assets.robot_init_rot, device=self.device).expand(self.num_envs, -1)
        cfg.action_space = 35
        self.single_action_space = gym.spaces.Box(-1., 1., shape=(35,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        cfg.observation_space = TASK_ACTOR_DIM + BODY_EXTRA_DIM
        cfg.state_space = TASK_CRITIC_DIM + BODY_EXTRA_DIM
        for key, size in (('policy', cfg.observation_space), ('critic', cfg.state_space)):
            self.single_observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space['policy'], self.num_envs)
        self.state_space = gym.vector.utils.batch_space(self.single_observation_space['critic'], self.num_envs)
        self._wholebody_ready = True
        print(f'SONIC_SOURCE_TASK_READY actor={cfg.observation_space} critic={cfg.state_space} actions=35 '
              'body_joints=29 total_joints=51 task_hz=60 touch_hz=70 reward=unchanged', flush=True)

    def _pre_physics_step(self, actions):
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
        task_terminated, truncated = super()._get_dones()
        height = self.robot.data.root_pos_w[:, 2]-self.scene.env_origins[:, 2]
        upright = -self.robot.data.projected_gravity_b[:, 2]
        self._body_fallen = ((height < self.cfg.sonic_body.minimum_pelvis_height) |
                            (upright < self.cfg.sonic_body.minimum_upright_cosine))
        # A robot fall is NOT the source task's object-fall metric.
        return task_terminated | self._body_fallen, truncated

    def _get_rewards(self):
        reward = super()._get_rewards()  # exact original manipulation reward
        self.extras['episode_final']['robot_fall'] = self._body_fallen.float()
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
        self._body_fallen[ids] = False
