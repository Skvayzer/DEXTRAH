"""Fit and serialize the frozen hand task map used by DextrAH-G."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ARTIFACT_VERSION = 1


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class PCAArtifact:
    """A self-contained, immutable linear hand action representation.

    DextrAH-G applies PCA as a fabric task map ``x = A q``.  Coordinates are
    therefore intentionally *not* mean-centered at runtime.  The mean remains
    in the artifact so configurations can be reconstructed for diagnostics.
    """

    joint_names: tuple[str, ...]
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray
    coordinate_min: np.ndarray
    coordinate_max: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        components = np.asarray(self.components, dtype=np.float64)
        if components.ndim != 2 or components.shape[1] != len(self.joint_names):
            raise ValueError("components must have shape (latent_dim, joint_dim)")
        if not 1 <= components.shape[0] <= components.shape[1]:
            raise ValueError("latent dimension must lie in [1, joint dimension]")
        if not np.isfinite(components).all():
            raise ValueError("components contains non-finite values")
        identity = components @ components.T
        if not np.allclose(identity, np.eye(len(components)), atol=1.0e-7):
            raise ValueError("PCA component rows must be orthonormal")

        joint_dim = components.shape[1]
        latent_dim = components.shape[0]
        mean = _as_vector(self.mean, joint_dim, "mean")
        variance = np.asarray(self.explained_variance_ratio, dtype=np.float64)
        if variance.ndim != 1 or len(variance) < latent_dim:
            raise ValueError("explained_variance_ratio must cover every component")
        if not np.isfinite(variance).all() or np.any(variance < 0.0):
            raise ValueError("explained_variance_ratio must be finite and non-negative")
        coordinate_min = _as_vector(self.coordinate_min, latent_dim, "coordinate_min")
        coordinate_max = _as_vector(self.coordinate_max, latent_dim, "coordinate_max")
        if np.any(coordinate_min > coordinate_max):
            raise ValueError("coordinate_min cannot exceed coordinate_max")
        joint_lower = _as_vector(self.joint_lower, joint_dim, "joint_lower")
        joint_upper = _as_vector(self.joint_upper, joint_dim, "joint_upper")
        if np.any(joint_lower >= joint_upper):
            raise ValueError("joint limits must have positive width")

        # Normalize all public arrays even when constructed from lists.
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "explained_variance_ratio", variance)
        object.__setattr__(self, "coordinate_min", coordinate_min)
        object.__setattr__(self, "coordinate_max", coordinate_max)
        object.__setattr__(self, "joint_lower", joint_lower)
        object.__setattr__(self, "joint_upper", joint_upper)
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def latent_dim(self) -> int:
        return self.components.shape[0]

    @property
    def joint_dim(self) -> int:
        return self.components.shape[1]

    @property
    def retained_variance(self) -> float:
        return float(self.explained_variance_ratio[: self.latent_dim].sum())

    def task_coordinates(self, joint_positions: np.ndarray) -> np.ndarray:
        """Evaluate the exact DextrAH-G fabric task map ``x = A q``."""

        positions = np.asarray(joint_positions, dtype=np.float64)
        if positions.shape[-1] != self.joint_dim:
            raise ValueError(f"joint_positions must end in {self.joint_dim} values")
        return positions @ self.components.T

    def reconstruct(self, task_coordinates: np.ndarray) -> np.ndarray:
        """Least-squares reconstruction in the retained PCA affine subspace."""

        coordinates = np.asarray(task_coordinates, dtype=np.float64)
        if coordinates.shape[-1] != self.latent_dim:
            raise ValueError(f"task_coordinates must end in {self.latent_dim} values")
        mean_coordinates = self.mean @ self.components.T
        return self.mean + (coordinates - mean_coordinates) @ self.components

    def extended_taskmap(self, arm_dof: int = 7) -> np.ndarray:
        """Return ``[0, A]`` for a combined arm-plus-hand configuration."""

        if arm_dof < 0:
            raise ValueError("arm_dof cannot be negative")
        return np.concatenate(
            (np.zeros((self.latent_dim, arm_dof)), self.components), axis=1
        )

    def clip_coordinates(self, task_coordinates: np.ndarray) -> np.ndarray:
        return np.clip(task_coordinates, self.coordinate_min, self.coordinate_max)

    def save(self, path: str | Path) -> Path:
        """Save without object arrays so loading never needs pickle."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata.setdefault("artifact_version", ARTIFACT_VERSION)
        np.savez_compressed(
            path,
            joint_names=np.asarray(self.joint_names, dtype=np.str_),
            components=self.components,
            mean=self.mean,
            explained_variance_ratio=self.explained_variance_ratio,
            coordinate_min=self.coordinate_min,
            coordinate_max=self.coordinate_max,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PCAArtifact":
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "joint_names",
                "components",
                "mean",
                "explained_variance_ratio",
                "coordinate_min",
                "coordinate_max",
                "joint_lower",
                "joint_upper",
                "metadata_json",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"PCA artifact is missing keys: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            return cls(
                joint_names=tuple(str(name) for name in archive["joint_names"]),
                components=archive["components"],
                mean=archive["mean"],
                explained_variance_ratio=archive["explained_variance_ratio"],
                coordinate_min=archive["coordinate_min"],
                coordinate_max=archive["coordinate_max"],
                joint_lower=archive["joint_lower"],
                joint_upper=archive["joint_upper"],
                metadata=metadata,
            )


def fit_pca_action_space(
    joint_positions: np.ndarray,
    *,
    joint_names: Sequence[str],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    components: int | None = None,
    variance_threshold: float = 0.98,
    max_components: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> PCAArtifact:
    """Fit a deterministic PCA representation to retargeted configurations."""

    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != len(joint_names):
        raise ValueError("joint_positions must have shape (samples, len(joint_names))")
    if len(positions) < 2:
        raise ValueError("at least two joint configurations are required")
    if not np.isfinite(positions).all():
        raise ValueError("joint_positions contains non-finite values")
    joint_dim = positions.shape[1]
    if components is not None and not 1 <= components <= joint_dim:
        raise ValueError("components must lie in [1, joint dimension]")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must lie in (0, 1]")
    if max_components is None:
        max_components = joint_dim
    if not 1 <= max_components <= joint_dim:
        raise ValueError("max_components must lie in [1, joint dimension]")

    mean = positions.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(
        positions - mean, full_matrices=False
    )
    variance = np.square(singular_values)
    total_variance = float(variance.sum())
    if total_variance <= np.finfo(np.float64).eps:
        raise ValueError("cannot fit PCA to constant joint configurations")
    variance_ratio = variance / total_variance

    # Resolve the arbitrary sign of each singular vector for byte-stable output.
    for row in right_vectors:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0

    if components is None:
        selected = int(np.searchsorted(np.cumsum(variance_ratio), variance_threshold) + 1)
        selected = min(selected, max_components)
    else:
        selected = components
    matrix = right_vectors[:selected]
    coordinates = positions @ matrix.T

    details = dict(metadata or {})
    details.setdefault("fit_samples", int(len(positions)))
    details.setdefault("selection", "fixed" if components is not None else "variance")
    details.setdefault("variance_threshold", float(variance_threshold))
    return PCAArtifact(
        joint_names=tuple(joint_names),
        components=matrix,
        mean=mean,
        explained_variance_ratio=variance_ratio,
        coordinate_min=coordinates.min(axis=0),
        coordinate_max=coordinates.max(axis=0),
        joint_lower=np.asarray(joint_lower, dtype=np.float64),
        joint_upper=np.asarray(joint_upper, dtype=np.float64),
        metadata=details,
    )
