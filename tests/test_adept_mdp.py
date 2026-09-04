import math

import torch

from dextrah_lab.tasks.dextrah_kuka_allegro.adept_mdp import (
    ADEPT_PRIMITIVES,
    contact_gate,
    keypoint_pose_error,
    pose_keypoints,
    primitive_surface_points,
    reposing_reward,
    transform_pointcloud,
)


def test_appendix_primitive_set_and_pointcloud_shape():
    points = primitive_surface_points()
    assert len(ADEPT_PRIMITIVES) == 16
    assert points.shape == (16, 64, 3)
    assert torch.isfinite(points).all()


def test_pose_keypoints_and_orientation_error():
    position = torch.zeros(1, 3)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    half_turn_z = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    quarter_turn_z = torch.tensor(
        [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]
    )

    corners = pose_keypoints(position, identity)
    assert corners.shape == (1, 8, 3)
    torch.testing.assert_close(
        keypoint_pose_error(position, identity, position, identity), torch.zeros(1)
    )
    # A cube is symmetric under 180-degree rotations, but not 90-degree rotations
    # under fixed corner correspondence.
    torch.testing.assert_close(
        keypoint_pose_error(position, identity, position, half_turn_z),
        torch.full((1,), math.sqrt(0.18)),
    )
    assert keypoint_pose_error(position, identity, position, quarter_turn_z).item() > 0


def test_contact_gate_requires_thumb_and_another_finger():
    forces = torch.zeros(4, 4, 3)
    forces[0, 3, 0] = 2.0
    forces[1, 0, 0] = 2.0
    forces[2, 3, 0] = 2.0
    forces[2, 1, 0] = 1.01
    forces[3, 3, 0] = 1.0
    forces[3, 2, 0] = 3.0
    assert contact_gate(forces).tolist() == [False, False, True, False]


def test_reposing_reward_gates_goal_but_not_reach():
    distance = torch.tensor([0.0, 0.0])
    error = torch.tensor([0.0, 0.0])
    gate = torch.tensor([False, True])
    total, reach, goal, contact = reposing_reward(distance, error, gate, 15.0)
    torch.testing.assert_close(reach, torch.ones(2))
    torch.testing.assert_close(goal, torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(contact, torch.tensor([0.0, 0.01]))
    torch.testing.assert_close(total, torch.tensor([1.0, 6.01]))


def test_pointcloud_transform_applies_scale_rotation_and_translation():
    local = torch.tensor([[[1.0, 0.0, 0.0]]])
    position = torch.tensor([[1.0, 2.0, 3.0]])
    quarter_turn_z = torch.tensor(
        [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]
    )
    transformed = transform_pointcloud(
        local, position, quarter_turn_z, torch.tensor([[2.0]])
    )
    torch.testing.assert_close(transformed, torch.tensor([[[1.0, 4.0, 3.0]]]))
