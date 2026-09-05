import numpy as np
import pytest
import torch

from dextrah_lab.retargeting.pca import fit_pca_action_space
from dextrah_lab.retargeting.pca_control import FrozenPCAHandActionMap


def _action_map() -> tuple[FrozenPCAHandActionMap, np.ndarray]:
    rng = np.random.default_rng(8)
    positions = rng.uniform(0.0, 1.5, size=(200, 6))
    names = tuple(f"joint_{index}" for index in range(6))
    artifact = fit_pca_action_space(
        positions,
        joint_names=names,
        joint_lower=np.zeros(6),
        joint_upper=np.full(6, 1.5),
        components=5,
    )
    return FrozenPCAHandActionMap(artifact, expected_joint_names=names), positions


def test_runtime_map_matches_numpy_artifact() -> None:
    action_map, positions = _action_map()
    q = torch.tensor(positions[:4], dtype=torch.float32)

    coordinates = action_map(q)

    torch.testing.assert_close(coordinates, q @ action_map.components.T)
    assert action_map.components.requires_grad is False
    assert tuple(action_map.parameters()) == ()


def test_normalized_policy_actions_map_to_fitted_bounds() -> None:
    action_map, _ = _action_map()
    raw = torch.stack(
        (
            torch.full((5,), -1.0),
            torch.zeros(5),
            torch.full((5,), 1.0),
            torch.full((5,), 2.0),
        )
    )

    targets = action_map.normalized_to_target(raw)

    torch.testing.assert_close(targets[0], action_map.coordinate_min)
    torch.testing.assert_close(
        targets[1], 0.5 * (action_map.coordinate_min + action_map.coordinate_max)
    )
    torch.testing.assert_close(targets[2], action_map.coordinate_max)
    torch.testing.assert_close(targets[3], action_map.coordinate_max)


def test_g1_revo2_fabric_matrix_ignores_arm_and_maps_hand() -> None:
    action_map, positions = _action_map()
    matrix = action_map.fabric_taskmap_matrix(arm_dof=7)
    arm = torch.randn(3, 7)
    hand = torch.tensor(positions[:3], dtype=torch.float32)
    combined = torch.cat((arm, hand), dim=-1)

    torch.testing.assert_close(combined @ matrix.T, action_map(hand))
    torch.testing.assert_close(matrix[:, :7], torch.zeros(5, 7))


def test_joint_order_mismatch_is_rejected() -> None:
    action_map, _ = _action_map()
    artifact_names = action_map.joint_names
    rng = np.random.default_rng(2)
    artifact = fit_pca_action_space(
        rng.normal(size=(20, 6)),
        joint_names=artifact_names,
        joint_lower=np.full(6, -3.0),
        joint_upper=np.full(6, 3.0),
        components=5,
    )

    with pytest.raises(ValueError, match="joint order"):
        FrozenPCAHandActionMap(
            artifact, expected_joint_names=tuple(reversed(artifact_names))
        )
