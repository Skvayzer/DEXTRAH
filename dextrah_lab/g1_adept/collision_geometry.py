"""Collision geometry shared by the G1 ADEPT runtime and Viser audit.

The geometry is intentionally a small sphere model rather than the visual or
contact meshes.  PhysX still owns physical contacts.  These spheres provide a
smooth, cheap signed-clearance field for the 13-D fabric controller.

All fixed coordinates below are expressed in the G1 pelvis frame at the
Play2Perfect right-arm training posture (zero waist, left arm in its rest
pose).  Dynamic offsets are expressed in their named link frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .reduced_fabric import CollisionBatch


G1_REVO2_CANONICAL_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)

# (target joint, canonical source index, multiplier, offset)
G1_REVO2_MIMIC_RULES: tuple[tuple[str, int, float, float], ...] = (
    ("right_thumb_distal_joint", 8, 1.0, 0.0),
    ("right_index_distal_joint", 9, 1.155, 0.0),
    ("right_middle_distal_joint", 10, 1.155, 0.0),
    ("right_ring_distal_joint", 11, 1.155, 0.0),
    ("right_pinky_distal_joint", 12, 1.155, 0.0),
)


@dataclass(frozen=True)
class DynamicSphereSpec:
    name: str
    link_name: str
    offset: tuple[float, float, float]
    radius: float
    avoid_body: bool = True
    avoid_table: bool = True


@dataclass(frozen=True)
class FixedSphereSpec:
    name: str
    center: tuple[float, float, float]
    radius: float


# The shoulder-pitch and shoulder-roll links are omitted: they are adjacent to
# the torso by construction.  Two samples cover each long link; smaller links
# and fingertips use one sphere each.  Offsets were measured against the exact
# G1+BrainCo URDF used by Play2Perfect.
G1_REVO2_DYNAMIC_SPHERES: tuple[DynamicSphereSpec, ...] = (
    DynamicSphereSpec("upper_arm_0", "right_shoulder_yaw_link", (0.010, 0.0, -0.025), 0.042, False, False),
    DynamicSphereSpec("upper_arm_1", "right_shoulder_yaw_link", (0.012, 0.0, -0.080), 0.042, False, True),
    DynamicSphereSpec("forearm_0", "right_elbow_link", (0.020, 0.0, -0.010), 0.040),
    DynamicSphereSpec("forearm_1", "right_elbow_link", (0.075, 0.0, -0.010), 0.040),
    DynamicSphereSpec("wrist_roll", "right_wrist_roll_link", (0.018, 0.0, 0.0), 0.033),
    DynamicSphereSpec("wrist_pitch", "right_wrist_pitch_link", (0.025, 0.0, 0.0), 0.034),
    DynamicSphereSpec("wrist_yaw", "right_wrist_yaw_link", (0.020, 0.0, 0.0), 0.032),
    DynamicSphereSpec("palm_0", "right_base_link", (0.0, 0.0, 0.025), 0.032),
    DynamicSphereSpec("palm_1", "right_base_link", (0.0, 0.0, 0.065), 0.032),
    DynamicSphereSpec("thumb_tip", "right_thumb_tip", (0.0, 0.0, 0.008), 0.014),
    DynamicSphereSpec("index_tip", "right_index_tip", (0.0, 0.0, 0.008), 0.014),
    DynamicSphereSpec("middle_tip", "right_middle_tip", (0.0, 0.0, 0.008), 0.014),
    DynamicSphereSpec("ring_tip", "right_ring_tip", (0.0, 0.0, 0.008), 0.014),
    DynamicSphereSpec("pinky_tip", "right_pinky_tip", (0.0, 0.0, 0.008), 0.014),
)

# Three torso samples approximate the 0.150 x 0.215 x 0.321 m torso mesh.
# The left-arm samples preserve the frozen arm as a configuration-space
# obstacle even in the scalable articulation where that branch is pruned.
G1_REVO2_FIXED_SPHERES: tuple[FixedSphereSpec, ...] = (
    FixedSphereSpec("torso_lower", (0.000, 0.000, 0.095), 0.105),
    FixedSphereSpec("torso_middle", (0.004, 0.000, 0.215), 0.110),
    FixedSphereSpec("torso_upper", (0.008, 0.000, 0.325), 0.085),
    FixedSphereSpec("left_upper_arm_0", (-0.043, 0.168, 0.170), 0.050),
    FixedSphereSpec("left_upper_arm_1", (-0.049, 0.177, 0.120), 0.047),
    FixedSphereSpec("left_forearm_0", (-0.035, 0.190, 0.065), 0.043),
    FixedSphereSpec("left_forearm_1", (-0.010, 0.208, -0.020), 0.040),
)

# Non-adjacent self-collision pairs.  Adjacent arm samples and finger-to-palm
# pairs are intentionally excluded so the field does not oppose a grasp.
G1_REVO2_SELF_COLLISION_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (arm_index, hand_index)
    for arm_index in range(4)
    for hand_index in range(7, len(G1_REVO2_DYNAMIC_SPHERES))
)


def point_jacobian(
    frame_jacobian: torch.Tensor, rotated_offset: torch.Tensor
) -> torch.Tensor:
    """Shift spatial Jacobians from frame origins to attached sphere centers.

    ``frame_jacobian`` has shape ``(..., 6, dof)`` with linear velocity first;
    ``rotated_offset`` has shape ``(..., 3)`` in the same world-aligned frame.
    """

    if frame_jacobian.shape[-2] != 6:
        raise ValueError("frame_jacobian must have six spatial rows")
    if rotated_offset.shape != frame_jacobian.shape[:-2] + (3,):
        raise ValueError(
            "rotated_offset leading dimensions must match frame_jacobian"
        )
    linear = frame_jacobian[..., :3, :]
    angular = frame_jacobian[..., 3:, :]
    return linear + torch.linalg.cross(
        angular.movedim(-2, -1),
        rotated_offset.unsqueeze(-2),
        dim=-1,
    ).movedim(-2, -1)


def reduce_jacobian_to_canonical(
    full_jacobian: torch.Tensor,
    canonical_column_ids: torch.Tensor,
    mimic_column_ids: torch.Tensor,
    mimic_source_ids: torch.Tensor,
    mimic_multipliers: torch.Tensor,
) -> torch.Tensor:
    """Fold five physical distal joints into the 13-D independent space."""

    reduced = full_jacobian.index_select(-1, canonical_column_ids).clone()
    if mimic_column_ids.numel() == 0:
        return reduced
    if not (
        mimic_column_ids.numel()
        == mimic_source_ids.numel()
        == mimic_multipliers.numel()
    ):
        raise ValueError("mimic mapping arrays must have equal lengths")
    additions = full_jacobian.index_select(-1, mimic_column_ids)
    additions = additions * mimic_multipliers.to(additions).view(
        *((1,) * (additions.ndim - 1)), -1
    )
    reduced.index_add_(-1, mimic_source_ids, additions)
    return reduced


def build_g1_collision_batch(
    dynamic_positions: torch.Tensor,
    dynamic_jacobians: torch.Tensor,
    fixed_positions: torch.Tensor,
    *,
    table_height: torch.Tensor | float,
    include_self_collision: bool = True,
) -> CollisionBatch:
    """Build body, table, and non-adjacent self-clearance constraints.

    Shapes are ``dynamic_positions=(B,S,3)``,
    ``dynamic_jacobians=(B,S,3,D)``, and ``fixed_positions=(B,F,3)``.
    """

    if dynamic_positions.ndim != 3 or dynamic_positions.shape[-1] != 3:
        raise ValueError("dynamic_positions must have shape (batch, spheres, 3)")
    if dynamic_jacobians.shape[:-2] != dynamic_positions.shape[:-1] or dynamic_jacobians.shape[-2] != 3:
        raise ValueError("dynamic_jacobians must have shape (batch, spheres, 3, dof)")
    if fixed_positions.ndim != 3 or fixed_positions.shape[0] != dynamic_positions.shape[0] or fixed_positions.shape[-1] != 3:
        raise ValueError("fixed_positions must have shape (batch, fixed_spheres, 3)")

    device = dynamic_positions.device
    dtype = dynamic_positions.dtype
    batch_size, _, _ = dynamic_positions.shape
    num_dof = dynamic_jacobians.shape[-1]
    dynamic_radii = torch.tensor(
        [sphere.radius for sphere in G1_REVO2_DYNAMIC_SPHERES],
        device=device,
        dtype=dtype,
    )
    fixed_radii = torch.tensor(
        [sphere.radius for sphere in G1_REVO2_FIXED_SPHERES],
        device=device,
        dtype=dtype,
    )
    if dynamic_positions.shape[1] != dynamic_radii.numel():
        raise ValueError("dynamic sphere count does not match the G1 specification")
    if fixed_positions.shape[1] != fixed_radii.numel():
        raise ValueError("fixed sphere count does not match the G1 specification")

    clearances: list[torch.Tensor] = []
    jacobians: list[torch.Tensor] = []

    body_dynamic_ids = torch.tensor(
        [i for i, sphere in enumerate(G1_REVO2_DYNAMIC_SPHERES) if sphere.avoid_body],
        device=device,
        dtype=torch.long,
    )
    body_position = dynamic_positions.index_select(1, body_dynamic_ids)
    body_jacobian = dynamic_jacobians.index_select(1, body_dynamic_ids)
    delta = body_position[:, :, None, :] - fixed_positions[:, None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-9)
    normal = delta / distance.unsqueeze(-1)
    body_clearance = (
        distance
        - dynamic_radii.index_select(0, body_dynamic_ids)[None, :, None]
        - fixed_radii[None, None, :]
    )
    body_gradient = torch.einsum("bsfk,bskd->bsfd", normal, body_jacobian)
    clearances.append(body_clearance.flatten(1))
    jacobians.append(body_gradient.flatten(1, 2))

    table_dynamic_ids = torch.tensor(
        [i for i, sphere in enumerate(G1_REVO2_DYNAMIC_SPHERES) if sphere.avoid_table],
        device=device,
        dtype=torch.long,
    )
    table_position = dynamic_positions.index_select(1, table_dynamic_ids)
    table_jacobian = dynamic_jacobians.index_select(1, table_dynamic_ids)
    height = torch.as_tensor(table_height, device=device, dtype=dtype)
    if height.ndim == 0:
        height = height.expand(batch_size)
    if height.shape != (batch_size,):
        raise ValueError("table_height must be scalar or have shape (batch,)")
    table_clearance = (
        table_position[..., 2]
        - dynamic_radii.index_select(0, table_dynamic_ids)[None, :]
        - height[:, None]
    )
    clearances.append(table_clearance)
    jacobians.append(table_jacobian[..., 2, :])

    if include_self_collision and G1_REVO2_SELF_COLLISION_PAIRS:
        first_ids = torch.tensor(
            [pair[0] for pair in G1_REVO2_SELF_COLLISION_PAIRS],
            device=device,
            dtype=torch.long,
        )
        second_ids = torch.tensor(
            [pair[1] for pair in G1_REVO2_SELF_COLLISION_PAIRS],
            device=device,
            dtype=torch.long,
        )
        delta = dynamic_positions.index_select(1, first_ids) - dynamic_positions.index_select(1, second_ids)
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-9)
        normal = delta / distance.unsqueeze(-1)
        relative_jacobian = dynamic_jacobians.index_select(1, first_ids) - dynamic_jacobians.index_select(1, second_ids)
        self_clearance = (
            distance
            - dynamic_radii.index_select(0, first_ids)[None, :]
            - dynamic_radii.index_select(0, second_ids)[None, :]
        )
        self_gradient = torch.einsum("bpk,bpkd->bpd", normal, relative_jacobian)
        clearances.append(self_clearance)
        jacobians.append(self_gradient)

    return CollisionBatch(
        clearance=torch.cat(clearances, dim=1),
        jacobian=torch.cat(jacobians, dim=1).reshape(batch_size, -1, num_dof),
    )


__all__ = [
    "DynamicSphereSpec",
    "FixedSphereSpec",
    "G1_REVO2_CANONICAL_JOINT_NAMES",
    "G1_REVO2_DYNAMIC_SPHERES",
    "G1_REVO2_FIXED_SPHERES",
    "G1_REVO2_MIMIC_RULES",
    "G1_REVO2_SELF_COLLISION_PAIRS",
    "build_g1_collision_batch",
    "point_jacobian",
    "reduce_jacobian_to_canonical",
]
