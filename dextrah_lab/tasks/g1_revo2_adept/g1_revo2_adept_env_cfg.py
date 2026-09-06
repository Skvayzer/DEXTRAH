"""Typed configuration additions for G1 ADEPT on the Play reposing task."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaacsimenvs.tasks.play.play_env_cfg import PlayEnvCfg

from dextrah_lab.g1_adept import ReducedAdeptFabricConfig


@configclass
class G1AdeptFabricCfg:
    enabled: bool = True
    timestep: float = 1.0 / 60.0
    arm_dof: int = 7
    steps_per_policy_step: int = 2
    max_joint_delta: float = 0.10
    velocity_target_factor: float = 1.0
    tracking_error_resync_threshold: float = 0.35
    table_surface_offset: float = 0.15
    include_self_collision: bool = True

    append_state_to_observations: bool = True
    pca_enabled: bool = True
    pca_artifact_path: str = (
        "/data2/users/konstantin.smirnov/dex-ycb/revo2-g1-full-v1/"
        "revo2_human_motion_pca.npz"
    )
    pca_artifact_sha256: str = (
        "8cea2fe7602958bbefec826fd331a100c15145c7dd837c68406a07895a2237b3"
    )
    pca_prior_initial_weight: float = 0.35
    pca_prior_final_weight: float = 0.05
    pca_prior_anneal_frames: int = 1_000_000_000

    target_gain_arm: float = 80.0
    target_gain_hand: float = 120.0
    target_damping_arm: float = 18.0
    target_damping_hand: float = 12.0
    target_metric_arm: float = 1.0
    target_metric_hand: float = 0.35
    cspace_damping: float = 1.0

    joint_limit_metric_scalar: float = 0.02
    joint_limit_metric_exploder_offset: float = 0.02
    joint_limit_max_metric: float = 20.0
    joint_limit_gate_sharpness: float = 10.0
    joint_limit_gate_offset: float = 0.02
    joint_limit_acceleration: float = 20.0
    joint_limit_damping: float = 4.0

    collision_influence_distance: float = 0.08
    collision_minimum_distance: float = 0.002
    collision_metric_scalar: float = 0.02
    collision_metric_budget: float = 8.0
    collision_acceleration: float = 35.0
    collision_damping: float = 8.0

    max_arm_velocity: float = 2.0
    max_hand_velocity: float = 4.0
    max_arm_acceleration: float = 10.0
    max_hand_acceleration: float = 25.0
    max_arm_jerk: float = 400.0
    max_hand_jerk: float = 1000.0
    speed_energy_arm_weight: float = 0.75
    speed_energy_hand_weight: float = 0.25
    speed_energy_target: float = 1.0
    solve_regularization: float = 1.0e-4

    def reduced_config(self) -> ReducedAdeptFabricConfig:
        keys = ReducedAdeptFabricConfig.__dataclass_fields__
        return ReducedAdeptFabricConfig(
            **{name: getattr(self, name) for name in keys}
        )


@configclass
class G1Revo2AdeptEnvCfg(PlayEnvCfg):
    fabric: G1AdeptFabricCfg = G1AdeptFabricCfg()

    def __post_init__(self) -> None:
        """Apply every safety- and scale-critical G1 setting in typed config.

        Isaac Lab's standard Hydra loader consumes ``env_cfg_entry_point`` but
        not Play2Perfect's optional ``env_cfg_yaml_entry_point``. Keeping these
        values here prevents a valid-looking launch from silently falling back
        to the KUKA/SHARPA defaults.
        """

        self.decimation = 2
        self.episode_length_s = 10.0
        self.action_space = 13

        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.04
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_found_lost_pairs_capacity = 16_777_216
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 33_554_432
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16_777_216
        self.sim.physx.gpu_max_rigid_contact_count = 16_777_216
        self.sim.physx.gpu_max_rigid_patch_count = 4_194_304
        self.sim.physx.gpu_collision_stack_size = 536_870_912

        self.scene.num_envs = 24_576
        self.scene.env_spacing = 2.4
        self.scene.replicate_physics = True
        self.scene.clone_in_fabric = False
        self.scene.filter_collisions = True

        assets = self.assets
        assets.robot_profile = "g1_brainco_right"
        assets.robot_urdf = (
            "unitree_ros/robots/g1_with_brainco_hand/"
            "g1_29dof_mode_15_brainco_hand.urdf"
        )
        assets.robot_usd = None
        assets.robot_init_pos = (0.0, 0.42, 0.75)
        assets.robot_init_rot = (0.7071068, 0.0, 0.0, -0.7071068)
        assets.robot_fix_base = True
        assets.robot_self_collision = False
        assets.robot_merge_fixed_joints = False
        assets.robot_body_collision_enabled = True
        assets.robot_body_collision_links = ("torso_link",)
        assets.robot_body_collision_base_link = "pelvis"
        assets.robot_body_collision_include_visuals = False
        assets.robot_actuate_left_hand = False
        assets.strip_visuals = True
        assets.spawn_goal_viz = False
        assets.replicate_single_physics_template = False
        assets.replicate_grouped_physics_templates = False
        assets.replicate_factorized_object_templates = True
        assets.replicate_template_object_index = 0
        assets.replicate_template_table_index = 0
        assets.table_urdf = "assets/urdf/table_narrow.urdf"
        assets.object_name = "handle_head_primitives"
        assets.handle_head_types = (
            "hammer",
            "screwdriver",
            "marker",
            "spatula",
            "eraser",
            "brush",
        )
        # SimToolReal has two distributions per semantic family: 12 x 100.
        assets.num_assets_per_type = 100
        assets.shuffle_assets = True
        assets.modify_asset_frictions = True
        assets.robot_friction = 0.5
        assets.finger_tip_friction = 1.5
        assets.object_friction = 0.5
        assets.table_friction = 0.5
        assets.palm_center_offset = (0.0, 0.0, 0.0)
        assets.fingertip_offset = (0.0, 0.0, 0.0)

        self.action.joint_limit_margin = 0.02
        self.reset.table_reset_z = 0.65
        self.reset.table_reset_z_range = 0.01
        self.reset.table_object_z_offset = 0.25
        self.reset.object_spawn_xy_mins = (-0.12, 0.06)
        self.reset.object_spawn_xy_maxs = (0.12, 0.16)
        self.reset.target_volume_mins = (-0.16, 0.02, 0.90)
        self.reset.target_volume_maxs = (0.16, 0.18, 1.08)


__all__ = ["G1AdeptFabricCfg", "G1Revo2AdeptEnvCfg"]
