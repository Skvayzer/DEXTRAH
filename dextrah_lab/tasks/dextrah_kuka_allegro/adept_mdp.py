"""Pure-Torch components of the ADEPT reposing MDP."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    shape: str
    dimensions: tuple[float, ...]


# Dimensions are transcribed from ADEPT v1, Appendix Fig. 8, in metres.
ADEPT_PRIMITIVES = (
    PrimitiveSpec("cuboid_050_100_100", "cuboid", (0.05, 0.10, 0.10)),
    PrimitiveSpec("cuboid_050_050_100", "cuboid", (0.05, 0.05, 0.10)),
    PrimitiveSpec("cuboid_025_100_100", "cuboid", (0.025, 0.10, 0.10)),
    PrimitiveSpec("cuboid_025_050_100", "cuboid", (0.025, 0.05, 0.10)),
    PrimitiveSpec("cuboid_025_025_100", "cuboid", (0.025, 0.025, 0.10)),
    PrimitiveSpec("cuboid_010_100_100", "cuboid", (0.01, 0.10, 0.10)),
    PrimitiveSpec("sphere_r050", "sphere", (0.05,)),
    PrimitiveSpec("sphere_r025", "sphere", (0.025,)),
    PrimitiveSpec("capsule_r040_h025", "capsule", (0.04, 0.025)),
    PrimitiveSpec("capsule_r040_h010", "capsule", (0.04, 0.01)),
    PrimitiveSpec("capsule_r040_h100", "capsule", (0.04, 0.10)),
    PrimitiveSpec("capsule_r025_h100", "capsule", (0.025, 0.10)),
    PrimitiveSpec("capsule_r025_h200", "capsule", (0.025, 0.20)),
    PrimitiveSpec("capsule_r010_h200", "capsule", (0.01, 0.20)),
    PrimitiveSpec("cone_r050_h100", "cone", (0.05, 0.10)),
    PrimitiveSpec("cone_r025_h100", "cone", (0.025, 0.10)),
)


def quaternion_apply_wxyz(quaternion: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D points by scalar-first quaternions with broadcasting."""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    vector = quaternion[..., 1:]
    scalar = quaternion[..., :1]
    while vector.ndim < points.ndim:
        vector = vector.unsqueeze(-2)
        scalar = scalar.unsqueeze(-2)
    vector = torch.broadcast_to(vector, points.shape)
    scalar = torch.broadcast_to(scalar, points.shape[:-1] + (1,))
    uv = torch.linalg.cross(vector, points, dim=-1)
    uuv = torch.linalg.cross(vector, uv, dim=-1)
    return points + 2.0 * (scalar * uv + uuv)


def pose_keypoints(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    half_extent: float = 0.15,
) -> torch.Tensor:
    """Return ADEPT's eight fixed cube keypoints in world coordinates."""
    offsets = torch.tensor(
        list(product((-half_extent, half_extent), repeat=3)),
        device=position.device,
        dtype=position.dtype,
    )
    offsets = offsets.expand(position.shape[:-1] + offsets.shape)
    return quaternion_apply_wxyz(quaternion, offsets) + position.unsqueeze(-2)


def keypoint_pose_error(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    goal_position: torch.Tensor,
    goal_quaternion: torch.Tensor,
    half_extent: float = 0.15,
) -> torch.Tensor:
    current = pose_keypoints(position, quaternion, half_extent)
    goal = pose_keypoints(goal_position, goal_quaternion, half_extent)
    return torch.linalg.vector_norm(current - goal, dim=-1).mean(dim=-1)


def contact_gate(
    fingertip_contact_forces: torch.Tensor,
    threshold: float = 1.0,
    thumb_index: int = -1,
) -> torch.Tensor:
    """Gate on thumb contact plus contact at any other fingertip."""
    if fingertip_contact_forces.ndim != 3 or fingertip_contact_forces.shape[-1] != 3:
        raise ValueError("fingertip_contact_forces must have shape (batch, fingers, 3)")
    if fingertip_contact_forces.shape[1] < 2:
        raise ValueError("at least two fingertips, including the thumb, are required")

    magnitudes = torch.linalg.vector_norm(fingertip_contact_forces, dim=-1)
    thumb_index %= magnitudes.shape[1]
    thumb_contact = magnitudes[:, thumb_index] > threshold
    non_thumb_mask = torch.ones(
        magnitudes.shape[1], dtype=torch.bool, device=magnitudes.device
    )
    non_thumb_mask[thumb_index] = False
    other_contact = torch.any(magnitudes[:, non_thumb_mask] > threshold, dim=1)
    return thumb_contact & other_contact


def reposing_reward(
    hand_to_object_distance: torch.Tensor,
    pose_error: torch.Tensor,
    gate: torch.Tensor,
    goal_sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ADEPT v1 Stage-1 reward and its three logged components."""
    reach = torch.exp(-10.0 * hand_to_object_distance)
    gate_float = gate.to(dtype=pose_error.dtype)
    goal = 5.0 * torch.exp(-goal_sharpness * pose_error) * gate_float
    contact_bonus = 0.01 * gate_float
    return reach + goal + contact_bonus, reach, goal, contact_bonus


def _sequence(count: int, device: torch.device | str, dtype: torch.dtype):
    index = torch.arange(count, device=device, dtype=dtype) + 0.5
    return index / count


def _cuboid_points(dimensions, count, device, dtype):
    dims = torch.tensor(dimensions, device=device, dtype=dtype)
    points = torch.empty(count, 3, device=device, dtype=dtype)
    for index in range(count):
        face = index % 6
        u = ((index * 0.61803398875) % 1.0) - 0.5
        v = ((index * 0.41421356237) % 1.0) - 0.5
        axis = face // 2
        other = [value for value in range(3) if value != axis]
        points[index, axis] = (0.5 if face % 2 == 0 else -0.5) * dims[axis]
        points[index, other[0]] = u * dims[other[0]]
        points[index, other[1]] = v * dims[other[1]]
    return points


def _sphere_points(radius, count, device, dtype):
    unit = _sequence(count, device, dtype)
    z = 1.0 - 2.0 * unit
    phi = torch.pi * (3.0 - 5.0**0.5) * torch.arange(
        count, device=device, dtype=dtype
    )
    radial = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    return radius * torch.stack((radial * torch.cos(phi), radial * torch.sin(phi), z), dim=-1)


def _capsule_points(radius, height, count, device, dtype):
    unit = _sequence(count, device, dtype)
    z_unit = 1.0 - 2.0 * unit
    phi = torch.pi * (3.0 - 5.0**0.5) * torch.arange(
        count, device=device, dtype=dtype
    )
    radial = radius * torch.sqrt((1.0 - z_unit * z_unit).clamp_min(0.0))
    cap_offset = torch.sign(z_unit) * height * 0.5
    return torch.stack(
        (radial * torch.cos(phi), radial * torch.sin(phi), radius * z_unit + cap_offset),
        dim=-1,
    )


def _cone_points(radius, height, count, device, dtype):
    side_count = (3 * count) // 4
    side_u = _sequence(side_count, device, dtype)
    phi = 2.0 * torch.pi * side_u * 7.0
    side_radius = radius * (1.0 - side_u)
    side_z = -height * 0.5 + height * side_u
    side = torch.stack(
        (side_radius * torch.cos(phi), side_radius * torch.sin(phi), side_z), dim=-1
    )
    base_count = count - side_count
    base_u = _sequence(base_count, device, dtype)
    base_phi = 2.0 * torch.pi * base_u * 5.0
    base_radius = radius * torch.sqrt(base_u)
    base = torch.stack(
        (
            base_radius * torch.cos(base_phi),
            base_radius * torch.sin(base_phi),
            torch.full_like(base_radius, -height * 0.5),
        ),
        dim=-1,
    )
    return torch.cat((side, base), dim=0)


def primitive_surface_points(
    num_points: int = 64,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic local-frame surface points for all 16 primitives."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    result = []
    for spec in ADEPT_PRIMITIVES:
        if spec.shape == "cuboid":
            points = _cuboid_points(spec.dimensions, num_points, device, dtype)
        elif spec.shape == "sphere":
            points = _sphere_points(spec.dimensions[0], num_points, device, dtype)
        elif spec.shape == "capsule":
            points = _capsule_points(*spec.dimensions, num_points, device, dtype)
        elif spec.shape == "cone":
            points = _cone_points(*spec.dimensions, num_points, device, dtype)
        else:
            raise ValueError(f"Unsupported primitive shape: {spec.shape}")
        result.append(points)
    return torch.stack(result)


def transform_pointcloud(
    local_points: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scaled = local_points * scale.reshape(scale.shape[0], 1, 1)
    return quaternion_apply_wxyz(quaternion, scaled) + position.unsqueeze(1)
