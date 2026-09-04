"""ADEPT Stage-2 FMB post-training environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .adept_fmb_mdp import (
    downstream_reward,
    extruded_polygon_surface_points,
    goal_tolerance,
    l_shaped_goal_path,
    rounded_square_polygon,
    star_polygon,
)
from .adept_mdp import keypoint_pose_error, sample_uniform_quaternion
from .adept_repose_env import AdeptKukaAllegroReposeEnv
from .dextrah_kuka_allegro_env import DextrahKukaAllegroEnv
from .dextrah_kuka_allegro_env_cfg import AdeptKukaAllegroFmbEnvCfg


class AdeptKukaAllegroFmbEnv(AdeptKukaAllegroReposeEnv):
    cfg: AdeptKukaAllegroFmbEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.goal_path_enabled = True
        self.force_final_goal = False
        self.student_distillation_phase = "downstream"
        super().__init__(cfg, render_mode, **kwargs)
        if not hasattr(self, "object_receptacle_contact_force"):
            self.object_receptacle_contact_force = torch.zeros(
                self.num_envs, 3, device=self.device
            )

    def _setup_policy_params(self):
        self.cfg.num_student_observations = 206
        self.cfg.num_teacher_observations = 392
        self.cfg.num_observations = 206 if self.cfg.distillation else 392
        self.cfg.num_states = 280
        self.cfg.state_space = 280
        self.cfg.observation_space = self.cfg.num_observations
        self.cfg.action_space = 23

    def _setup_objects(self):
        super()._setup_objects()
        self.receptacle = RigidObject(self.cfg.fmb_board_cfg)
        self.scene.rigid_objects["receptacle"] = self.receptacle
        sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/object/.*/baseLink",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/receptacle/baseLink"],
        )
        self.object_receptacle_sensor = ContactSensor(sensor_cfg)
        self.scene.sensors["object_receptacle_contact"] = self.object_receptacle_sensor

    def _build_local_object_pointcloud(self):
        if self.cfg.fmb_variant == "star":
            polygon = star_polygon(device=self.device)
        elif self.cfg.fmb_variant == "square_round":
            polygon = rounded_square_polygon(device=self.device)
        else:
            raise ValueError(f"unsupported FMB variant: {self.cfg.fmb_variant}")
        points = extruded_polygon_surface_points(
            polygon, 0.15, self.cfg.num_pointcloud_points
        )
        return points.unsqueeze(0).expand(self.num_envs, -1, -1).contiguous(), torch.zeros(
            self.num_envs, 1, device=self.device
        )

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        if not hasattr(self, "receptacle"):
            return
        self.receptacle_pos = self.receptacle.data.root_pos_w - self.scene.env_origins
        self.receptacle_rot = self.receptacle.data.root_quat_w
        if hasattr(self, "object_receptacle_sensor"):
            force_matrix = self.object_receptacle_sensor.data.force_matrix_w
            if force_matrix is not None:
                self.object_receptacle_contact_force = force_matrix.sum(dim=(1, 2))
            elif not hasattr(self, "object_receptacle_contact_force"):
                self.object_receptacle_contact_force = torch.zeros(
                    self.num_envs, 3, device=self.device
                )
        if hasattr(self, "goal_path_start"):
            self._update_fmb_goal()

    def _update_fmb_goal(self):
        preinsert = self.receptacle_pos.clone()
        preinsert[:, 2] += self.cfg.inferred_preinsert_height
        insertion = self.receptacle_pos.clone()
        insertion[:, 2] += self.cfg.inferred_insertion_height
        if self.student_distillation_phase == "vision_pretrain":
            self.object_goal = preinsert
            self.object_goal_quat.zero_()
            self.object_goal_quat[:, 0] = 1.0
            return
        level = 50 if self.force_final_goal else self.dextrah_adr.num_increments()
        self.object_goal = l_shaped_goal_path(
            level, self.goal_path_start, preinsert, insertion
        ) if self.goal_path_enabled else insertion
        self.object_goal_quat.copy_(self.receptacle_rot)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Skip the reposing-specific goal and orientation reset.
        DextrahKukaAllegroEnv._reset_idx(self, env_ids)
        count = len(env_ids)
        object_quaternion = sample_uniform_quaternion(count, self.device)
        object_pose = torch.cat(
            (self.object.data.root_pos_w[env_ids], object_quaternion), dim=-1
        )
        self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)
        fraction = self.dextrah_adr.num_increments() / self.cfg.num_adr_increments
        if self.student_distillation_phase == "vision_pretrain":
            fraction = 0.0  # Appendix I: one fixed receptacle in student Stage 1.
        nominal = torch.tensor(self.cfg.fmb_board_position, device=self.device)
        x_range = torch.tensor((-0.07, 0.12), device=self.device) * fraction
        y_range = torch.tensor((-0.30, 0.10), device=self.device) * fraction
        board_position = nominal.repeat(count, 1)
        board_position[:, 0] += x_range[0] + torch.rand(count, device=self.device) * (x_range[1] - x_range[0])
        board_position[:, 1] += y_range[0] + torch.rand(count, device=self.device) * (y_range[1] - y_range[0])
        board_pose = torch.cat(
            (
                board_position + self.scene.env_origins[env_ids],
                torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(count, 1),
            ),
            dim=-1,
        )
        self.receptacle.write_root_pose_to_sim(board_pose, env_ids=env_ids)
        if not hasattr(self, "goal_path_start"):
            self.goal_path_start = torch.zeros(self.num_envs, 3, device=self.device)
        # Use the pose submitted above instead of relying on the asset data
        # view being refreshed before the next physics step.
        self.goal_path_start[env_ids] = object_pose[:, :3] - self.scene.env_origins[env_ids]
        self.goal_path_start[env_ids, 2] = board_position[:, 2] + self.cfg.inferred_preinsert_height

    def set_post_training_phase(self, phase: str):
        if phase == "actor_bc":
            self.goal_path_enabled = True
            self.force_final_goal = False
            self.cfg.enable_adr = False
            self.dextrah_adr.set_num_increments(20)
        elif phase in {"critic_warmup", "ppo"}:
            self.goal_path_enabled = False
            self.force_final_goal = True
            self.cfg.enable_adr = True
        else:
            raise ValueError(f"unknown post-training phase: {phase}")

    def set_student_distillation_phase(self, phase: str):
        if phase == "vision_pretrain":
            self.student_distillation_phase = phase
            self.goal_path_enabled = False
            self.force_final_goal = False
            self.cfg.enable_adr = False
            self.dextrah_adr.set_num_increments(self.cfg.num_adr_increments)
        elif phase == "downstream":
            self.student_distillation_phase = phase
            self.set_post_training_phase("ppo")
        else:
            raise ValueError(f"unknown student distillation phase: {phase}")

    def compute_intermediate_reward_values(self):
        self.object_keypoint_error = keypoint_pose_error(
            self.object_pos,
            self.object_rot,
            self.object_goal,
            self.object_goal_quat,
            self.cfg.keypoint_half_extent,
        )
        self.object_position_error = torch.linalg.vector_norm(
            self.object_pos - self.object_goal, dim=-1
        )
        self.hand_to_object_pos_error = torch.linalg.vector_norm(
            self.hand_pos - self.object_pos[:, None, :], dim=-1
        ).max(dim=-1).values
        tolerance = goal_tolerance(self.dextrah_adr.num_increments())
        self.in_success_region = self.object_position_error < tolerance
        self.time_in_success_region = torch.where(
            self.in_success_region,
            self.time_in_success_region + self.cfg.sim.dt * self.cfg.decimation,
            0.0,
        )

    def _get_rewards(self):
        self.compute_intermediate_reward_values()
        total, reach, goal = downstream_reward(
            self.hand_to_object_pos_error, self.object_keypoint_error
        )
        self.extras["hand_to_object_reward"] = reach.mean()
        self.extras["object_to_goal_reward"] = goal.mean()
        self.extras["true_objective"] = self.in_success_region.float().mean()
        self.extras["adr_level"] = self.dextrah_adr.num_increments()
        return total

    def _nominal_delta(self):
        nominal_hand_position, _ = self.hand_points_taskmap(self.robot_start_joint_pos, None)
        return torch.cat(
            (
                self.robot_dof_pos - self.robot_start_joint_pos,
                self.hand_pos.flatten(1) - nominal_hand_position,
            ),
            dim=-1,
        )

    def compute_policy_observations(self):
        contact_force = getattr(
            self,
            "object_receptacle_contact_force",
            torch.zeros(self.num_envs, 3, device=self.device),
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
                self.fingertip_contact_forces.flatten(1),
                self.object_goal,
                self.object_goal_quat,
                self._pointcloud(noisy=True).flatten(1),
                self.receptacle_pos,
                self.receptacle_rot,
                contact_force,
            ),
            dim=-1,
        )

    def compute_pretraining_teacher_observations(self):
        """Build the 391-D Stage-1 view needed to query the frozen teacher."""

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

    def _get_observations(self):
        if self.use_camera:
            self._randomize_fmb_render_materials()
        observations = super()._get_observations()
        # The raw Gym environment exposes both non-nested observation spaces
        # during Algorithm-1 BC. RL-Games continues to consume policy/critic.
        observations["pretraining_policy"] = (
            self.compute_pretraining_teacher_observations()
        )
        return observations

    def _randomize_fmb_render_materials(self):
        """Apply Appendix-I peg/board RGB ranges at each camera frame."""

        from pxr import Gf

        peg_low, peg_high = self.cfg.peg_diffuse_rgb_range
        board_low, board_high = self.cfg.board_diffuse_rgb_range
        for env_id in range(self.num_envs):
            peg_rgb = torch.empty(3).uniform_(peg_low, peg_high).tolist()
            board_rgb = torch.empty(3).uniform_(board_low, board_high).tolist()
            object_name = self.object_names[0]
            peg_shader = self.stage.GetPrimAtPath(
                f"/World/envs/env_{env_id}/object/object_{env_id}_{object_name}"
                "/baseLink/Looks/material/Shader"
            )
            peg_color = peg_shader.GetAttribute("inputs:diffuseColor")
            if peg_color.IsValid():
                peg_color.Set(Gf.Vec3f(*peg_rgb))
            board_shader = self.stage.GetPrimAtPath(
                f"/World/envs/env_{env_id}/receptacle/baseLink/Looks/material/Shader"
            )
            board_color = board_shader.GetAttribute("inputs:diffuseColor")
            if board_color.IsValid():
                board_color.Set(Gf.Vec3f(*board_rgb))

    def compute_critic_observations(self):
        contact_force = getattr(
            self,
            "object_receptacle_contact_force",
            torch.zeros(self.num_envs, 3, device=self.device),
        )
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
                self._nominal_delta(),
                self.receptacle_pos,
                self.receptacle_rot,
                contact_force,
                self.hand_forces[:, :6],
                self.measured_joint_torque[:, self.actuated_dof_indices],
            ),
            dim=-1,
        )
