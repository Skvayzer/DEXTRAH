import pytest
import torch

from dextrah_lab.adept.fabric_math import (
    collision_acceleration,
    collision_metric_weights,
    joint_limit_metric_diagonal,
    normalize_collision_metric_per_sphere,
    normalized_joint_clearance,
    smooth_gate,
)


def test_smooth_gate_prefers_motion_toward_limit():
    velocity = torch.tensor([-1.0, 0.0, 1.0])
    gate = smooth_gate(velocity, sharpness=10.0, offset=0.0)
    assert gate[0] > 0.999
    assert gate[1] == pytest.approx(0.5)
    assert gate[2] < 0.001


def test_joint_clearances_are_normalized_by_range():
    position = torch.tensor([[1.0, 2.0]])
    lower = torch.tensor([[0.0, -2.0]])
    upper = torch.tensor([[2.0, 6.0]])
    upper_clearance, lower_clearance = normalized_joint_clearance(
        position, lower, upper
    )
    assert torch.allclose(upper_clearance, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(lower_clearance, torch.tensor([[0.5, 0.5]]))


def test_joint_metric_is_capped_and_smoothly_velocity_gated():
    metric, gate = joint_limit_metric_diagonal(
        torch.tensor([[0.0, 0.2]]),
        torch.tensor([[-1.0, 1.0]]),
        metric_scalar=0.02,
        metric_exploder_offset=0.1,
        max_metric=100.0,
        gate_sharpness=10.0,
        gate_offset=0.0,
    )
    assert metric[0, 0] == pytest.approx(100.0, rel=1e-5)
    assert gate[0, 0] > gate[0, 1]


def test_collision_budget_is_split_by_radius_and_clearance():
    weights = collision_metric_weights(
        signed_distance=torch.tensor([[0.1, 0.2]]),
        sphere_radius=torch.tensor([[0.1, 0.1]]),
        active=torch.tensor([[True, True]]),
        metric_scalar=1.0,
        metric_budget=2.0,
        minimum_distance=0.01,
    )
    assert weights.sum() == pytest.approx(2.0)
    assert weights[0, 0] / weights[0, 1] == pytest.approx(4.0)


def test_collision_metric_normalizes_each_sphere_not_whole_stack():
    metric = torch.zeros(1, 6, 6)
    metric[0, 0, 0] = 3.0
    metric[0, 1, 1] = 4.0
    metric[0, 3, 3] = 12.0
    normalized, active = normalize_collision_metric_per_sphere(metric, 2)
    first = normalized[:, :3, :3]
    second = normalized[:, 3:, 3:]
    assert torch.linalg.matrix_norm(first) == pytest.approx(1.0)
    assert torch.linalg.matrix_norm(second) == pytest.approx(1.0)
    assert active.tolist() == [[True, True]]


def test_geometric_collision_acceleration_uses_per_sphere_speed():
    direction = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    velocity = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]]])
    acceleration = collision_acceleration(
        direction, velocity, gain=2.0, geometric=True
    )
    assert torch.allclose(
        acceleration,
        torch.tensor([[[-8.0, 0.0, 0.0], [0.0, -50.0, 0.0]]]),
    )
