import numpy as np
import pytest

from dextrah_lab.retargeting.pca import PCAArtifact, fit_pca_action_space


def _dataset(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(500, 3))
    basis, _ = np.linalg.qr(rng.normal(size=(6, 3)))
    mean = np.linspace(0.1, 0.6, 6)
    positions = mean + latent @ basis.T
    return positions, mean


def test_fit_selects_low_rank_space_and_reconstructs() -> None:
    positions, _ = _dataset()
    artifact = fit_pca_action_space(
        positions,
        joint_names=tuple(f"joint_{i}" for i in range(6)),
        joint_lower=np.full(6, -3.0),
        joint_upper=np.full(6, 3.0),
        variance_threshold=0.98,
    )

    assert artifact.latent_dim == 3
    assert artifact.retained_variance == pytest.approx(1.0)
    coordinates = artifact.task_coordinates(positions)
    np.testing.assert_allclose(artifact.reconstruct(coordinates), positions, atol=1e-12)
    np.testing.assert_allclose(artifact.extended_taskmap(7)[:, 7:], artifact.components)


def test_fabric_coordinates_are_deliberately_not_centered() -> None:
    positions, _ = _dataset()
    artifact = fit_pca_action_space(
        positions,
        joint_names=tuple(f"joint_{i}" for i in range(6)),
        joint_lower=np.full(6, -3.0),
        joint_upper=np.full(6, 3.0),
        components=3,
    )

    expected = positions @ artifact.components.T
    centered = (positions - artifact.mean) @ artifact.components.T
    np.testing.assert_allclose(artifact.task_coordinates(positions), expected)
    assert not np.allclose(expected, centered)


def test_artifact_round_trip_without_pickle(tmp_path) -> None:
    positions, _ = _dataset()
    artifact = fit_pca_action_space(
        positions,
        joint_names=tuple(f"joint_{i}" for i in range(6)),
        joint_lower=np.full(6, -3.0),
        joint_upper=np.full(6, 3.0),
        components=3,
        metadata={"source": "synthetic"},
    )

    path = artifact.save(tmp_path / "revo2_pca.npz")
    loaded = PCAArtifact.load(path)

    assert loaded.joint_names == artifact.joint_names
    assert loaded.metadata["source"] == "synthetic"
    assert loaded.metadata["artifact_version"] == 1
    np.testing.assert_array_equal(loaded.components, artifact.components)
    np.testing.assert_array_equal(loaded.coordinate_min, artifact.coordinate_min)


def test_pca_rejects_constant_data() -> None:
    with pytest.raises(ValueError, match="constant"):
        fit_pca_action_space(
            np.ones((4, 6)),
            joint_names=tuple(f"joint_{i}" for i in range(6)),
            joint_lower=np.zeros(6),
            joint_upper=np.ones(6),
        )
