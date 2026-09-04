"""ADEPT Stage-1 reposing environment for the KUKA-Allegro embodiment."""

from __future__ import annotations

from collections.abc import Sequence

import carb
import torch

import isaaclab.sim as sim_utils

from .adept_mdp import (
    ADEPT_PRIMITIVES,
    contact_gate,
    keypoint_pose_error,
    primitive_surface_points,
    reposing_reward,
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

        point_bank = primitive_surface_points(
            self.cfg.num_pointcloud_points, device=self.device
        )
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
        self.local_object_pointcloud = point_bank[env_spec_indices]
        self.object_id_scalar = env_spec_indices.float().unsqueeze(-1) / (
            len(ADEPT_PRIMITIVES) - 1
        )

        self._contact_body_indices = None
        self.fingertip_contact_forces = torch.zeros(
            self.num_envs, len(self.cfg.hand_body_names), 3, device=self.device
        )
        self._last_gravity_adr = None

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
        if self._contact_body_indices is None:
            indices = []
            for body_name in self.cfg.hand_body_names:
                body_indices, _ = self.contact_sensor.find_bodies(body_name)
                if len(body_indices) != 1:
                    raise RuntimeError(
                        f"Expected one contact-sensor body for {body_name}, got {body_indices}"
                    )
                indices.append(body_indices[0])
            self._contact_body_indices = indices
        return self._contact_body_indices

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        if hasattr(self, "contact_sensor"):
            indices = self._ordered_contact_body_indices()
            self.fingertip_contact_forces = self.contact_sensor.data.net_forces_w[
                :, indices, :
            ]

    def _adr_fraction(self) -> float:
        return self.dextrah_adr.num_increments() / float(self.cfg.num_adr_increments)

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
        self._update_global_fabric_curriculum()
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # DirectRLEnv may dispatch reset before this subclass constructor has
        # allocated goal buffers.
        if not hasattr(self, "object_goal_quat"):
            self.object_goal = torch.tensor(
                self.cfg.inferred_goal_center, device=self.device
            ).repeat(self.num_envs, 1)
            self.object_goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
            self.object_goal_quat[:, 0] = 1.0

        super()._reset_idx(env_ids)

        count = len(env_ids)
        fraction = self._adr_fraction()
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
        self.extras["keypoint_pose_error"] = self.object_keypoint_error.mean()
        self.extras["num_adr_increases"] = self.dextrah_adr.num_increments()
        self.extras["in_success_region"] = self.in_success_region.float().mean()
        return total

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
