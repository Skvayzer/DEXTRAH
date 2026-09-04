"""Pure-Torch ADEPT downstream FMB task primitives (Appendix A.2)."""

from __future__ import annotations

import torch


def downstream_reward(
    hand_to_object_distance: torch.Tensor,
    keypoint_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared from-scratch/bootstrap reward in ADEPT Eq. (3)."""

    reach = torch.exp(-10.0 * hand_to_object_distance)
    goal = 5.0 * torch.exp(-15.0 * keypoint_error)
    return reach + goal, reach, goal


def goal_tolerance(adr_level: int | torch.Tensor, num_increments: int = 50):
    """Anneal positional success tolerance from 5 cm to 2 cm."""

    if torch.is_tensor(adr_level):
        fraction = (adr_level / float(num_increments)).clamp(0.0, 1.0)
    else:
        fraction = min(1.0, max(0.0, adr_level / float(num_increments)))
    return 0.05 + fraction * (0.02 - 0.05)


def l_shaped_goal_path(
    adr_level: int | torch.Tensor,
    lift_start: torch.Tensor,
    preinsert: torch.Tensor,
    insertion: torch.Tensor,
    num_increments: int = 50,
) -> torch.Tensor:
    """Horizontal transport (ADR 0–25), then vertical insertion (25–50)."""

    level = torch.as_tensor(adr_level, device=lift_start.device, dtype=lift_start.dtype)
    midpoint = num_increments / 2.0
    transport_fraction = (level / midpoint).clamp(0.0, 1.0)
    insertion_fraction = ((level - midpoint) / midpoint).clamp(0.0, 1.0)
    while transport_fraction.ndim < lift_start.ndim:
        transport_fraction = transport_fraction.unsqueeze(-1)
        insertion_fraction = insertion_fraction.unsqueeze(-1)
    transport_goal = torch.lerp(lift_start, preinsert, transport_fraction)
    insertion_goal = torch.lerp(preinsert, insertion, insertion_fraction)
    return torch.where(level.reshape(level.shape + (1,) * (lift_start.ndim - level.ndim)) <= midpoint, transport_goal, insertion_goal)


def extruded_polygon_surface_points(
    polygon_xy: torch.Tensor,
    height: float,
    num_points: int = 64,
) -> torch.Tensor:
    """Deterministic surface samples for procedural FMB proxy pegs."""

    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2 or polygon_xy.shape[0] < 3:
        raise ValueError("polygon_xy must have shape (vertices>=3, 2)")
    if height <= 0.0 or num_points <= 0:
        raise ValueError("height and num_points must be positive")
    vertices = polygon_xy.shape[0]
    indices = torch.arange(num_points, device=polygon_xy.device)
    edge = indices % vertices
    next_edge = (edge + 1) % vertices
    interpolation = ((indices * 0.61803398875) % 1.0).to(polygon_xy.dtype).unsqueeze(-1)
    xy = torch.lerp(polygon_xy[edge], polygon_xy[next_edge], interpolation)
    z = (((indices * 0.41421356237) % 1.0) - 0.5).to(polygon_xy.dtype) * height
    points = torch.cat((xy, z.unsqueeze(-1)), dim=-1)
    # Include both cap interiors as well as side walls.
    cap_mask = indices % 4 < 2
    points[cap_mask, :2] *= ((indices[cap_mask] + 1) / num_points).to(polygon_xy.dtype).unsqueeze(-1)
    points[cap_mask, 2] = torch.where(
        indices[cap_mask] % 2 == 0,
        torch.as_tensor(height / 2, device=points.device, dtype=points.dtype),
        torch.as_tensor(-height / 2, device=points.device, dtype=points.dtype),
    )
    return points


def star_polygon(
    outer_radius: float = 0.025,
    inner_radius: float = 0.0125,
    points: int = 5,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    angle = torch.arange(2 * points, device=device) * torch.pi / points + torch.pi / 2
    radius = torch.where(
        torch.arange(2 * points, device=device) % 2 == 0,
        outer_radius,
        inner_radius,
    )
    return torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)


def rounded_square_polygon(
    radius: float = 0.025,
    vertices: int = 16,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Superellipse approximation of the FMB square/round cross-section."""

    angle = torch.arange(vertices, device=device) * 2 * torch.pi / vertices
    cosine, sine = angle.cos(), angle.sin()
    exponent = 0.5
    x = radius * cosine.sign() * cosine.abs().pow(exponent)
    y = radius * sine.sign() * sine.abs().pow(exponent)
    return torch.stack((x, y), dim=-1)

