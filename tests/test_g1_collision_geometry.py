from __future__ import annotations

import torch

from dextrah_lab.g1_adept.collision_geometry import (
    G1_REVO2_DYNAMIC_SPHERES,
    G1_REVO2_FIXED_SPHERES,
    build_g1_collision_batch,
    point_jacobian,
    reduce_jacobian_to_canonical,
)


def test_point_jacobian_accounts_for_angular_velocity() -> None:
    jacobian = torch.zeros(1, 6, 2)
    jacobian[0, 3:, 0] = torch.tensor([0.0, 0.0, 1.0])
    jacobian[0, :3, 1] = torch.tensor([1.0, 2.0, 3.0])
    shifted = point_jacobian(jacobian, torch.tensor([[2.0, 0.0, 0.0]]))
    assert torch.allclose(shifted[0, :, 0], torch.tensor([0.0, 2.0, 0.0]))
    assert torch.allclose(shifted[0, :, 1], torch.tensor([1.0, 2.0, 3.0]))


def test_mimic_jacobians_are_folded_into_their_sources() -> None:
    full = torch.arange(2 * 3 * 9, dtype=torch.float32).reshape(2, 3, 9)
    canonical = torch.tensor([0, 2, 4, 6])
    mimic = torch.tensor([7, 8])
    source = torch.tensor([1, 3])
    multipliers = torch.tensor([1.0, 1.5])
    reduced = reduce_jacobian_to_canonical(
        full, canonical, mimic, source, multipliers
    )
    expected = full.index_select(-1, canonical).clone()
    expected[..., 1] += full[..., 7]
    expected[..., 3] += 1.5 * full[..., 8]
    assert torch.equal(reduced, expected)


def test_collision_batch_has_body_table_and_self_constraints() -> None:
    batch, dof = 2, 13
    num_dynamic = len(G1_REVO2_DYNAMIC_SPHERES)
    num_fixed = len(G1_REVO2_FIXED_SPHERES)
    positions = torch.zeros(batch, num_dynamic, 3)
    positions[..., 2] = 0.5
    jacobians = torch.randn(batch, num_dynamic, 3, dof)
    fixed = torch.full((batch, num_fixed, 3), 2.0)

    collision = build_g1_collision_batch(
        positions, jacobians, fixed, table_height=0.0
    )
    collision.validate(batch, dof)
    assert collision.clearance.shape[1] > num_dynamic + num_fixed
    assert torch.isfinite(collision.clearance).all()
    assert torch.isfinite(collision.jacobian).all()


def test_table_clearance_gradient_is_point_jacobian_z_row() -> None:
    batch, dof = 1, 13
    num_dynamic = len(G1_REVO2_DYNAMIC_SPHERES)
    num_fixed = len(G1_REVO2_FIXED_SPHERES)
    positions = torch.zeros(batch, num_dynamic, 3)
    positions[..., 2] = 0.25
    jacobians = torch.zeros(batch, num_dynamic, 3, dof)
    jacobians[..., 2, 0] = 3.0
    fixed = torch.full((batch, num_fixed, 3), 10.0)
    collision = build_g1_collision_batch(
        positions, jacobians, fixed, table_height=0.05,
        include_self_collision=False,
    )

    body_count = sum(s.avoid_body for s in G1_REVO2_DYNAMIC_SPHERES) * num_fixed
    table_count = sum(s.avoid_table for s in G1_REVO2_DYNAMIC_SPHERES)
    table_jacobian = collision.jacobian[:, body_count : body_count + table_count]
    assert torch.all(table_jacobian[..., 0] == 3.0)
    assert torch.all(table_jacobian[..., 1:] == 0.0)
