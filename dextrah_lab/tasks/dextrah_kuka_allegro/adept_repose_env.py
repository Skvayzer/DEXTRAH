"""ADEPT Stage-1 reposing environment for the KUKA-Allegro embodiment."""

from __future__ import annotations

from collections.abc import Sequence

import carb
import torch

import isaaclab.sim as sim_utils

from dextrah_lab.adept.curriculum import RollingSuccessRate

from .adept_mdp import (
    ADEPT_PRIMITIVES,
    aggregate_body_contact_forces,
    contact_gate,
    keypoint_pose_error,
    primitive_surface_points,
    reposing_reward,
    quaternion_multiply_wxyz,
    rotation_vector_to_quaternion_wxyz,
    sample_uniform_quaternion,
    transform_pointcloud,
)
from .dextrah_kuka_allegro_env import DextrahKukaAllegroEnv
from .dextrah_kuka_allegro_env_cfg import AdeptKukaAllegroEnvCfg


class AdeptKukaAllegroReposeEnv(DextrahKukaAllegroEnv):
    """Paper-aligned reposing MDP using the public DextrAH simulator base."""

    cfg: AdeptKukaAllegroEnvCfg

    def __init__(self, cfg: AdeptKukaAllegroEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.object_goal = torch.tensor(
            self.cfg.inferred_goal_center, device=self.device
        ).repeat(self.num_envs, 1)
        self.object_goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_goal_quat[:, 0] = 1.0

        self.local_object_pointcloud, self.object_id_scalar = (
            self._build_local_object_pointcloud()
        )

        self._contact_body_indices = None
        self.fingertip_contact_forces = torch.zeros(
            self.num_envs, len(self.cfg.hand_body_names), 3, device=self.device
        )
        self._last_gravity_adr = None
        self._adept_curriculum_started = True
        self._ensure_adept_curriculum_buffers()

    def _build_local_object_pointcloud(self):
        point_bank = primitive_surface_points(self.cfg.num_pointcloud_points, device=self.device)
        spec_index = {spec.name: index for index, spec in enumerate(ADEPT_PRIMITIVES)}
        unknown_objects = sorted(set(self.object_names) - set(spec_index))
        if unknown_objects or len(self.object_names) != len(ADEPT_PRIMITIVES):
            raise ValueError(
                "adept_primitives must contain exactly the 16 appendix objects; "
                f"found {self.object_names}"
            )
        env_spec_indices = torch.tensor(
            [spec_index[self.object_names[index]] for index in self.multi_object_idx.tolist()],
            device=self.device,
            dtype=torch.long,
        )
        local_pointcloud = point_bank[env_spec_indices]
        object_id_scalar = env_spec_indices.float().unsqueeze(-1) / (
            len(ADEPT_PRIMITIVES) - 1
        )
        return local_pointcloud, object_id_scalar

    def _setup_policy_params(self):
        self.cfg.num_student_observations = 206
        self.cfg.num_teacher_observations = 391
        self.cfg.num_observations = (
            self.cfg.num_student_observations
            if self.cfg.distillation
            else self.cfg.num_teacher_observations
        )
        self.cfg.num_states = 438
        self.cfg.state_space = self.cfg.num_states
        self.cfg.observation_space = self.cfg.num_observations
        self.cfg.action_space = self.cfg.num_actions

    def _ordered_contact_body_indices(self):
        if getattr(self, "_contact_body_indices", None) is None:
            body_groups = (
                ("palm_link",),
                ("index_link_3", "index_biotac_tip"),
                ("middle_link_3", "middle_biotac_tip"),
                ("ring_link_3", "ring_biotac_tip"),
                ("thumb_link_3", "thumb_biotac_tip"),
            )
            indices = []
            for body_names in body_groups:
                group = []
                for body_name in body_names:
                    body_indices, _ = self.contact_sensor.find_bodies(body_name)
                    if len(body_indices) != 1:
                        raise RuntimeError(
                            f"Expected one contact-sensor body for {body_name}, "
                            f"got {body_indices}"
                        )
                    group.append(body_indices[0])
                if not group:
                    raise RuntimeError(
                        f"No contact-sensor bodies found for {body_names}"
                    )
                indices.append(group)
            self._contact_body_indices = indices
        return self._contact_body_indices

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        if hasattr(self, "robot_joint_pos_bias"):
            self._apply_adept_observation_noise()
        if hasattr(self, "contact_sensor"):
            raw_forces = self.contact_sensor.data.net_forces_w
            self.fingertip_contact_forces = aggregate_body_contact_forces(
                raw_forces, self._ordered_contact_body_indices()
            )

    def _adr_fraction(self) -> float:
        return min(
            1.0,
            self.dextrah_adr.num_increments() / float(self.cfg.num_adr_increments),
        )

    def _ensure_adept_curriculum_buffers(self) -> None:
        if not hasattr(self, "_episode_success"):
            self._episode_success = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if not hasattr(self, "_adr_success_window"):
            capacity = self.cfg.adr_success_window_episodes_per_env * self.num_envs
            self._adr_success_window = RollingSuccessRate(capacity, self.device)
            self._adr_running_success_rate = 0.0
            self._adr_level_control_steps = 0

    def _record_completed_episodes(self, env_ids: torch.Tensor) -> None:
        self._adr_success_window.update(self._episode_success[env_ids])
        self._adr_running_success_rate = self._adr_success_window.rate
        self._episode_success[env_ids] = False

    def _maybe_advance_adr(self) -> bool:
        if not self.cfg.enable_adr:
            return False
        if self.dextrah_adr.num_increments() >= self.cfg.num_adr_increments:
            return False
        if self._adr_level_control_steps < self.cfg.min_steps_for_dr_change:
            return False
        if not self._adr_success_window.ready:
            return False
        if self._adr_running_success_rate <= self.cfg.success_for_adr:
            return False
        if bool(self.local_adr_increment != self.global_min_adr_increment):
            return False

        self.dextrah_adr.increase_ranges(increase_counter=True)
        self.local_adr_increment.fill_(self.dextrah_adr.num_increments())
        self._adr_level_control_steps = 0
        self._adr_success_window.clear()
        self._adr_running_success_rate = 0.0
        self._last_gravity_adr = None
        return True

    def _update_global_fabric_curriculum(self):
        adr_level = self.dextrah_adr.num_increments()
        self.kuka_allegro_fabric.fabric_params["speed_control"]["energy_target"] = (
            self.dextrah_adr.get_custom_param_value(
                "fabric_speed_control", "energy_target"
            )
        )
        if self._last_gravity_adr == adr_level:
            return
        gravity_z = self.dextrah_adr.get_custom_param_value("gravity", "z")
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, gravity_z))
        self._last_gravity_adr = adr_level

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._ensure_adept_curriculum_buffers()
        if getattr(self, "_adept_curriculum_started", False):
            self._adr_level_control_steps += 1
        self._update_global_fabric_curriculum()
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._ensure_adept_noise_buffers()
        self._ensure_adept_curriculum_buffers()

        # Consume terminal episode outcomes before the inherited reset clears
        # success signals. Initial construction resets have no prior episode.
        if getattr(self, "_adept_curriculum_started", False):
            self._record_completed_episodes(env_ids)
            self._maybe_advance_adr()

        # DirectRLEnv may dispatch reset before this subclass constructor has
        # allocated goal buffers.
        if not hasattr(self, "object_goal_quat"):
            self.object_goal = torch.tensor(
                self.cfg.inferred_goal_center, device=self.device
            ).repeat(self.num_envs, 1)
            self.object_goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
            self.object_goal_quat[:, 0] = 1.0

        # DextrAH's inherited ADR condition samples instantaneous success after
        # clearing it and increments a reset counter rather than control steps.
        # This task owns the paper-aligned rolling episodic implementation.
        enable_adr = self.cfg.enable_adr
        self.cfg.enable_adr = False
        try:
            super()._reset_idx(env_ids)
        finally:
            self.cfg.enable_adr = enable_adr

        # Appendix-I biases are independently sampled per observed joint and
        # held for the rollout.  The inherited DextrAH implementation samples
        # one shared scalar and halves the published interval.
        fraction = self._adr_fraction()
        joint_position_bias = self.dextrah_adr.get_custom_param_value(
            "robot_state_noise", "robot_joint_pos_bias"
        )
        joint_velocity_bias = self.dextrah_adr.get_custom_param_value(
            "robot_state_noise", "robot_joint_vel_bias"
        )
        count = len(env_ids)
        self.robot_joint_pos_bias[env_ids] = joint_position_bias * (
            2.0 * torch.rand(count, self.cfg.num_actions, device=self.device) - 1.0
        )
        self.robot_joint_vel_bias[env_ids] = joint_velocity_bias * (
            2.0 * torch.rand(count, self.cfg.num_actions, device=self.device) - 1.0
        )
        # No sustained object-pose bias is listed in ADEPT Appendix I.
        self.object_pos_bias[env_ids] = 0.0
        self.object_rot_bias[env_ids] = 0.0

        center = torch.tensor(
            self.cfg.inferred_goal_center, device=self.device
        ).unsqueeze(0)
        half_width = torch.tensor(
            self.cfg.inferred_goal_position_half_width, device=self.device
        ).unsqueeze(0)
        self.object_goal[env_ids] = center + fraction * half_width * (
            2.0 * torch.rand(count, 3, device=self.device) - 1.0
        )

        sampled_goal = sample_uniform_quaternion(count, self.device)
        sampled_goal = torch.where(
            sampled_goal[:, :1] < 0.0, -sampled_goal, sampled_goal
        )
        identity = torch.zeros_like(sampled_goal)
        identity[:, 0] = 1.0
        interpolated = (1.0 - fraction) * identity + fraction * sampled_goal
        self.object_goal_quat[env_ids] = torch.nn.functional.normalize(
            interpolated, dim=-1
        )

        # DextrAH samples two-axis rotations. ADEPT specifies full SO(3) at
        # final ADR, so overwrite the reset orientation with an SO(3) sample.
        sampled_object = sample_uniform_quaternion(count, self.device)
        sampled_object = torch.where(
            sampled_object[:, :1] < 0.0, -sampled_object, sampled_object
        )
        object_quat = torch.nn.functional.normalize(
            (1.0 - fraction) * identity + fraction * sampled_object, dim=-1
        )
        object_pose = torch.cat(
            (self.object.data.root_pos_w[env_ids], object_quat), dim=-1
        )
        self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)

    def _ensure_adept_noise_buffers(self):
        """Allocate per-coordinate bias tensors before the first virtual reset."""

        expected_joint_shape = (self.num_envs, self.cfg.num_actions)
        if getattr(self, "robot_joint_pos_bias", None) is not None:
            if self.robot_joint_pos_bias.shape != expected_joint_shape:
                self.robot_joint_pos_bias = torch.zeros(
                    expected_joint_shape, device=self.device
                )
                self.robot_joint_vel_bias = torch.zeros_like(
                    self.robot_joint_pos_bias
                )

    def _apply_adept_observation_noise(self):
        """Apply ADEPT's Gaussian sensor noise with valid SO(3) rotations."""

        position_std = self.dextrah_adr.get_custom_param_value(
            "robot_state_noise", "robot_joint_pos_noise"
        )
        velocity_std = self.dextrah_adr.get_custom_param_value(
            "robot_state_noise", "robot_joint_vel_noise"
        )
        self.robot_dof_pos_noisy = (
            self.robot_dof_pos
            + position_std * torch.randn_like(self.robot_dof_pos)
            + self.robot_joint_pos_bias
        )
        self.robot_dof_vel_noisy = (
            self.robot_dof_vel
            + velocity_std * torch.randn_like(self.robot_dof_vel)
            + self.robot_joint_vel_bias
        )
        self.hand_pos_noisy, hand_points_jacobian = self.hand_points_taskmap(
            self.robot_dof_pos_noisy, None
        )
        self.hand_vel_noisy = torch.bmm(
            hand_points_jacobian, self.robot_dof_vel_noisy.unsqueeze(2)
        ).squeeze(2)

        object_position_std = self.dextrah_adr.get_custom_param_value(
            "object_state_noise", "object_pos_noise"
        )
        object_rotation_std = self.dextrah_adr.get_custom_param_value(
            "object_state_noise", "object_rot_noise"
        )
        self.object_pos_noisy = (
            self.object_pos
            + object_position_std * torch.randn_like(self.object_pos)
        )
        rotation_noise = rotation_vector_to_quaternion_wxyz(
            object_rotation_std
            * torch.randn(self.num_envs, 3, device=self.device)
        )
        self.object_rot_noisy = torch.nn.functional.normalize(
            quaternion_multiply_wxyz(self.object_rot, rotation_noise), dim=-1
        )

    def _pointcloud(self, noisy: bool) -> torch.Tensor:
        position = self.object_pos_noisy if noisy else self.object_pos
        quaternion = self.object_rot_noisy if noisy else self.object_rot
        return transform_pointcloud(
            self.local_object_pointcloud,
            position,
            quaternion,
            self.object_scale,
        )

    def compute_intermediate_reward_values(self):
        self.object_keypoint_error = keypoint_pose_error(
            self.object_pos,
            self.object_rot,
            self.object_goal,
            self.object_goal_quat,
            self.cfg.keypoint_half_extent,
        )
        self.hand_to_object_pos_error = torch.linalg.vector_norm(
            self.hand_pos - self.object_pos[:, None, :], dim=-1
        ).max(dim=-1).values
        self.contact_gate = contact_gate(
            self.fingertip_contact_forces[:, 1:, :],
            threshold=self.cfg.contact_force_threshold,
            thumb_index=-1,
        )
        self.in_success_region = self.object_keypoint_error < self.cfg.object_goal_tol
        if hasattr(self, "_episode_success"):
            self._episode_success.logical_or_(self.in_success_region)
        self.time_in_success_region = torch.where(
            self.in_success_region,
            self.time_in_success_region + self.cfg.sim.dt * self.cfg.decimation,
            0.0,
        )

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate_reward_values()
        total, reach, goal, contact_bonus = reposing_reward(
            self.hand_to_object_pos_error,
            self.object_keypoint_error,
            self.contact_gate,
            self.dextrah_adr.get_custom_param_value(
                "reward_weights", "object_to_goal_sharpness"
            ),
        )
        self.extras["hand_to_object_reward"] = reach.mean()
        self.extras["object_to_goal_reward"] = goal.mean()
        self.extras["contact_bonus"] = contact_bonus.mean()
        self.extras["contact_gate"] = self.contact_gate.float().mean()
        fingertip_force = torch.linalg.vector_norm(
            self.fingertip_contact_forces[:, 1:, :], dim=-1
        )
        above_threshold = fingertip_force > self.cfg.contact_force_threshold
        self.extras["thumb_contact"] = above_threshold[:, -1].float().mean()
        self.extras["other_finger_contact"] = (
            above_threshold[:, :-1].any(dim=-1).float().mean()
        )
        self.extras["max_fingertip_force"] = fingertip_force.max(dim=-1).values.mean()
        self.extras["keypoint_pose_error"] = self.object_keypoint_error.mean()
        self.extras["num_adr_increases"] = self.dextrah_adr.num_increments()
        self.extras["in_success_region"] = self.in_success_region.float().mean()
        self.extras["episode_success_rate"] = self._adr_running_success_rate
        self.extras["adr_window_episodes"] = self._adr_success_window.count
        self.extras["adr_level_control_steps"] = self._adr_level_control_steps
        # PBT must rank on rolling, unshaped success, never reward magnitude.
        self.extras["true_objective"] = self._adr_running_success_rate
        self.extras["adr_level"] = self.dextrah_adr.num_increments()
        return total

    def get_env_state(self):
        self._ensure_adept_curriculum_buffers()
        return {
            "version": 1,
            "adr_level": self.dextrah_adr.num_increments(),
            "adr_level_control_steps": self._adr_level_control_steps,
            "adr_success_window": self._adr_success_window.state_dict(),
        }

    def set_env_state(self, state):
        if state is None:
            return
        self._ensure_adept_curriculum_buffers()
        adr_level = int(state["adr_level"])
        if not 0 <= adr_level <= self.cfg.num_adr_increments:
            raise ValueError(f"invalid checkpoint ADR level: {adr_level}")
        self.dextrah_adr.set_num_increments(adr_level)
        self.local_adr_increment.fill_(adr_level)
        self.global_min_adr_increment.fill_(adr_level)
        self._adr_level_control_steps = int(
            state.get("adr_level_control_steps", 0)
        )
        success_window = state.get("adr_success_window")
        if success_window is not None:
            self._adr_success_window.load_state_dict(success_window)
        else:
            self._adr_success_window.clear()
        self._adr_running_success_rate = self._adr_success_window.rate
        # Physics state is recreated on process resume, so only completed
        # episode history—not the interrupted episode outcome—is restored.
        self._episode_success.zero_()
        self._last_gravity_adr = None
        self._update_global_fabric_curriculum()

    def compute_policy_observations(self):
        return torch.cat(
            (
                self.robot_dof_pos_noisy,
                self.robot_dof_vel_noisy,
                self.hand_pos_noisy,
                self.hand_vel_noisy,
                self.actions,
                self.fabric_q_for_obs,
                self.fabric_qd_for_obs,
                self.fabric_qdd_for_obs,
                self.fingertip_contact_forces.flatten(1),
                self.object_pos_noisy,
                self.object_rot_noisy,
                self.object_goal,
                self.object_goal_quat,
                self.object_id_scalar,
                self.object_scale,
                self._pointcloud(noisy=True).flatten(1),
            ),
            dim=-1,
        )

    def compute_student_policy_observations(self):
        """Exact 206-D KUKA vision-student low-dimensional observation."""

        nominal_hand_position, _ = self.hand_points_taskmap(
            self.robot_start_joint_pos, None
        )
        nominal_delta = torch.cat(
            (
                self.robot_dof_pos_noisy - self.robot_start_joint_pos,
                self.hand_pos_noisy - nominal_hand_position,
            ),
            dim=-1,
        )
        return torch.cat(
            (
                self.robot_dof_pos_noisy,
                self.robot_dof_vel_noisy,
                self.hand_pos_noisy,
                self.hand_vel_noisy,
                self.actions,
                self.fabric_q_for_obs,
                self.fabric_qd_for_obs,
                self.fabric_qdd_for_obs,
                nominal_delta,
            ),
            dim=-1,
        )

    def compute_critic_observations(self):
        return torch.cat(
            (
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.hand_pos.flatten(1),
                self.hand_vel[..., :3].flatten(1),
                self.actions,
                self.fabric_q,
                self.fabric_qd,
                self.fabric_qdd,
                self.fingertip_contact_forces.flatten(1),
                self.object_pos,
                self.object_rot,
                self.object_vel,
                self.object_goal,
                self.object_goal_quat,
                self.object_id_scalar,
                self.object_scale,
                self._pointcloud(noisy=False).flatten(1),
                self.hand_vel[..., 3:].flatten(1),
                self.fingertip_contact_forces[:, 0, :],
                self.measured_joint_torque[:, self.actuated_dof_indices],
            ),
            dim=-1,
        )
