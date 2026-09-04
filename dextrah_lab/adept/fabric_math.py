"""Tensor-only response laws from ADEPT Appendix B.

The functions in this module are deliberately independent of Isaac Sim and
NVIDIA FABRICS. Besides making the equations auditable, this keeps their
numerical contracts covered by the normal CPU unit-test suite.
"""

from __future__ import annotations

import torch


def smooth_gate(
    velocity: torch.Tensor,
    *,
    sharpness: float,
    offset: float,
) -> torch.Tensor:
    """ADEPT's smooth tanh gate ``0.5 * (tanh(-a(v-b)) + 1)``."""

    return 0.5 * (torch.tanh(-sharpness * (velocity - offset)) + 1.0)


def normalized_joint_clearance(
    position: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upper/lower clearance as a fraction of each joint's range."""

    joint_range = upper - lower
    if torch.any(joint_range <= 0):
        raise ValueError("every upper joint limit must exceed its lower limit")
    return (upper - position) / joint_range, (position - lower) / joint_range


def joint_limit_metric_diagonal(
    clearance: torch.Tensor,
    clearance_velocity: torch.Tensor,
    *,
    metric_scalar: float,
    metric_exploder_offset: float,
    max_metric: float,
    gate_sharpness: float,
    gate_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Appendix-B joint-limit metric diagonal and velocity gate."""

    minimum_delta = (metric_scalar / max_metric) ** 0.5
    delta = torch.clamp(
        clearance - metric_exploder_offset,
        min=minimum_delta,
    )
    gate = smooth_gate(
        clearance_velocity,
        sharpness=gate_sharpness,
        offset=gate_offset,
    )
    return gate * metric_scalar / delta.square(), gate


def collision_metric_weights(
    signed_distance: torch.Tensor,
    sphere_radius: torch.Tensor,
    active: torch.Tensor,
    *,
    metric_scalar: float,
    metric_budget: float,
    minimum_distance: float,
) -> torch.Tensor:
    """Allocate ADEPT's bounded collision-metric budget per body sphere."""

    distance = torch.clamp(signed_distance, min=minimum_distance)
    radius_norm = torch.linalg.vector_norm(sphere_radius, dim=-1, keepdim=True)
    raw = metric_scalar * (sphere_radius / radius_norm) / distance.square()
    raw = raw * active.to(raw.dtype)
    raw_sum = raw.sum(dim=-1, keepdim=True)
    allocated_total = torch.clamp(raw_sum, max=metric_budget)
    return allocated_total * raw / torch.clamp(raw_sum, min=1.0e-12)


def normalize_collision_metric_per_sphere(
    metric: torch.Tensor,
    num_spheres: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each 3x3 sphere block independently in a stacked metric."""

    expected = 3 * num_spheres
    if metric.shape[-2:] != (expected, expected):
        raise ValueError(
            f"metric must end in {(expected, expected)}, got {metric.shape[-2:]}"
        )
    row_energy = metric.square().sum(dim=-1)
    block_norm = torch.sqrt(row_energy.reshape(-1, num_spheres, 3).sum(dim=-1))
    expanded_norm = block_norm.repeat_interleave(3, dim=-1).unsqueeze(-1)
    normalized = metric / torch.clamp(expanded_norm, min=1.0e-12)
    return normalized, block_norm > 1.0e-12


def collision_acceleration(
    direction: torch.Tensor,
    velocity: torch.Tensor,
    *,
    gain: float,
    damping: float = 0.0,
    geometric: bool,
) -> torch.Tensor:
    """Return per-sphere forcing or HD2 geometric collision acceleration."""

    if direction.shape != velocity.shape or direction.shape[-1] != 3:
        raise ValueError("direction and velocity must both have shape [..., 3]")
    if geometric:
        speed_squared = velocity.square().sum(dim=-1, keepdim=True)
        return -gain * speed_squared * direction
    return -gain * direction - damping * velocity
