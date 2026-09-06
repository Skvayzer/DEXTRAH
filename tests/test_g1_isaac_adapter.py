from __future__ import annotations

from types import SimpleNamespace

import torch

from dextrah_lab.g1_adept import (
    G1_REVO2_CANONICAL_JOINT_NAMES,
    G1_REVO2_DYNAMIC_SPHERES,
    G1_REVO2_FIXED_SPHERES,
    IsaacG1CollisionAdapter,
    quaternion_apply_wxyz,
)


def test_quaternion_apply_wxyz_rotates_about_z() -> None:
    half = torch.tensor(torch.pi / 4)
    quaternion = torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]])
    vector = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = quaternion_apply_wxyz(quaternion, vector)
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6)


class _FakePhysxView:
    def __init__(self, jacobians: torch.Tensor) -> None:
        self.jacobians = jacobians

    def get_jacobians(self) -> torch.Tensor:
        return self.jacobians


def _fake_env(batch_size: int = 2):
    unique_links = tuple(dict.fromkeys(s.link_name for s in G1_REVO2_DYNAMIC_SPHERES))
    body_names = ("pelvis",) + unique_links
    joint_names = G1_REVO2_CANONICAL_JOINT_NAMES + tuple(
        f"mimic_{index}" for index in range(5)
    )
    body_state = torch.zeros(batch_size, len(body_names), 13)
    body_state[..., 3] = 1.0
    for index in range(1, len(body_names)):
        body_state[:, index, 0] = 0.20 + 0.03 * index
        body_state[:, index, 2] = 0.50
    jacobians = torch.zeros(batch_size, len(body_names) - 1, 6, len(joint_names))
    jacobians[..., 0, 0] = 1.0
    jacobians[..., 2, 1] = 1.0
    joint_pos = torch.zeros(batch_size, len(joint_names))
    robot_data = SimpleNamespace(
        body_names=body_names,
        joint_names=joint_names,
        body_state_w=body_state,
        joint_pos=joint_pos,
        root_pos_w=torch.zeros(batch_size, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(batch_size, -1),
    )
    robot = SimpleNamespace(
        data=robot_data,
        root_physx_view=_FakePhysxView(jacobians),
    )
    return SimpleNamespace(
        device="cpu",
        num_envs=batch_size,
        robot=robot,
        scene=SimpleNamespace(env_origins=torch.zeros(batch_size, 3)),
        _joint_names_canonical=G1_REVO2_CANONICAL_JOINT_NAMES,
        _canonical_joint_ids_lab=torch.arange(13),
        _mimic_target_joint_ids=torch.arange(13, 18),
        _mimic_source_hand_action_ids=torch.tensor([1, 2, 3, 4, 5]),
        _mimic_multipliers=torch.tensor([[1.0, 1.155, 1.155, 1.155, 1.155]]),
        _table_z_per_env=torch.full((batch_size,), 0.35),
    )


def test_adapter_builds_world_space_collision_batch() -> None:
    env = _fake_env()
    adapter = IsaacG1CollisionAdapter(env)
    collision = adapter.build()
    collision.validate(env.num_envs, 13)
    assert adapter.dynamic_positions.shape == (
        env.num_envs,
        len(G1_REVO2_DYNAMIC_SPHERES),
        3,
    )
    assert adapter.fixed_positions.shape == (
        env.num_envs,
        len(G1_REVO2_FIXED_SPHERES),
        3,
    )
    assert torch.isfinite(collision.clearance).all()


def test_adapter_folds_mimic_jacobian_columns() -> None:
    env = _fake_env(batch_size=1)
    env.robot.root_physx_view.jacobians[..., 0, 13] = 2.0
    adapter = IsaacG1CollisionAdapter(env)
    _, jacobians = adapter._dynamic_sphere_state()
    # Mimic joint 13 follows canonical hand source 8 with multiplier 1.
    assert torch.allclose(jacobians[..., 0, 8], torch.full_like(jacobians[..., 0, 8], 2.0))
