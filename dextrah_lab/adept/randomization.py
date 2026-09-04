"""Complete distillation domain-randomization specification (Appendix I)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicsRandomization:
    object_mass_scale: tuple[float, float] = (0.5, 3.0)
    object_static_friction: tuple[float, float] = (0.5, 1.2)
    object_dynamic_friction: tuple[float, float] = (0.3, 1.0)
    object_restitution: tuple[float, float] = (0.8, 1.0)
    robot_joint_stiffness_scale: tuple[float, float] = (0.5, 2.0)
    robot_joint_damping_scale: tuple[float, float] = (0.5, 2.0)
    robot_joint_friction_nm: tuple[float, float] = (0.0, 5.0)


@dataclass(frozen=True)
class SceneRandomization:
    peg_spawn_xy_jitter_m: float = 0.15
    peg_spawn_full_so3_at_final_adr: bool = True
    peg_wrench_linear_accel_mps2: float = 10.0
    board_x_jitter_m: tuple[float, float] = (-0.07, 0.12)
    board_y_jitter_m: tuple[float, float] = (-0.30, 0.10)


@dataclass(frozen=True)
class ObservationRandomization:
    joint_position_noise_std_rad: float = 0.08
    joint_position_bias_rad: tuple[float, float] = (-0.08, 0.08)
    joint_velocity_noise_std_rad_s: float = 0.18
    joint_velocity_bias_rad_s: tuple[float, float] = (-0.08, 0.08)
    object_position_noise_std_m: float = 0.03
    object_orientation_noise_std_rad: float = 0.1


@dataclass(frozen=True)
class VisualRandomization:
    lighting_intensity_color_direction: bool = True
    dome_background_texture: bool = True
    table_color_and_roughness: bool = True
    robot_link_color_and_roughness: bool = True
    peg_diffuse_rgb: tuple[float, float] = (0.90, 1.0)
    board_diffuse_rgb: tuple[float, float] = (0.04, 0.12)
    camera_position_jitter_m: float = 0.03
    camera_rotation_jitter_deg: float = 3.0


@dataclass(frozen=True)
class DistillationRandomization:
    physics: PhysicsRandomization = field(default_factory=PhysicsRandomization)
    scene: SceneRandomization = field(default_factory=SceneRandomization)
    observation: ObservationRandomization = field(default_factory=ObservationRandomization)
    visual: VisualRandomization = field(default_factory=VisualRandomization)
    adr_increments: int = 50
    success_threshold: float = 0.4


ADEPT_DISTILLATION_RANDOMIZATION = DistillationRandomization()
