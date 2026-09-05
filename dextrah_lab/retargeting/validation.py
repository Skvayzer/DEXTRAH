"""Independent consistency checks for Revo2 retargeting artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .pca import PCAArtifact
from .revo2_kinematics import Revo2Kinematics


@dataclass(frozen=True)
class TrajectoryValidation:
    failures: tuple[str, ...]
    metrics: dict[str, float]

    @property
    def passed(self) -> bool:
        return not self.failures


def validate_trajectory_arrays(
    arrays: Mapping[str, np.ndarray],
    hand: Revo2Kinematics,
    artifact: PCAArtifact,
    *,
    tolerance: float = 1.0e-7,
) -> TrajectoryValidation:
    """Validate bounds, FK, losses, gamma schedule, and PCA projection."""

    required = {
        "joint_positions",
        "robot_fingertips",
        "scaled_human_fingertips",
        "gamma",
        "fingertip_error",
    }
    missing = required.difference(arrays)
    if missing:
        return TrajectoryValidation(
            failures=(f"missing arrays: {sorted(missing)}",), metrics={}
        )

    q = np.asarray(arrays["joint_positions"], dtype=np.float64)
    robot_tips = np.asarray(arrays["robot_fingertips"], dtype=np.float64)
    human_tips = np.asarray(arrays["scaled_human_fingertips"], dtype=np.float64)
    gamma = np.asarray(arrays["gamma"], dtype=np.float64)
    stored_error = np.asarray(arrays["fingertip_error"], dtype=np.float64)
    failures: list[str] = []
    frame_count = len(q) if q.ndim >= 1 else 0

    if q.ndim != 2 or q.shape[1:] != (artifact.joint_dim,):
        failures.append(f"joint_positions has invalid shape {q.shape}")
    if robot_tips.shape != (frame_count, 5, 3):
        failures.append(f"robot_fingertips has invalid shape {robot_tips.shape}")
    if human_tips.shape != (frame_count, 5, 3):
        failures.append(f"scaled_human_fingertips has invalid shape {human_tips.shape}")
    if gamma.shape != (frame_count,):
        failures.append(f"gamma has invalid shape {gamma.shape}")
    if stored_error.shape != (frame_count, 5):
        failures.append(f"fingertip_error has invalid shape {stored_error.shape}")
    if failures:
        return TrajectoryValidation(tuple(failures), {})

    for name, value in (
        ("joint_positions", q),
        ("robot_fingertips", robot_tips),
        ("scaled_human_fingertips", human_tips),
        ("gamma", gamma),
        ("fingertip_error", stored_error),
    ):
        if not np.isfinite(value).all():
            failures.append(f"{name} contains non-finite values")

    lower_violation = np.maximum(artifact.joint_lower - q, 0.0)
    upper_violation = np.maximum(q - artifact.joint_upper, 0.0)
    max_limit_violation = float(max(lower_violation.max(), upper_violation.max()))
    if max_limit_violation > tolerance:
        failures.append(f"joint-limit violation {max_limit_violation:.3e} rad")

    recomputed_tips = hand.fingertip_positions(
        torch.as_tensor(q, dtype=hand.lower.dtype, device=hand.lower.device)
    ).detach().cpu().numpy()
    max_fk_difference = float(np.max(np.abs(recomputed_tips - robot_tips)))
    if max_fk_difference > tolerance:
        failures.append(f"stored/recomputed FK difference {max_fk_difference:.3e} m")

    recomputed_error = np.linalg.norm(robot_tips - human_tips, axis=-1)
    max_error_difference = float(np.max(np.abs(recomputed_error - stored_error)))
    if max_error_difference > tolerance:
        failures.append(
            f"stored/recomputed fingertip error difference {max_error_difference:.3e} m"
        )

    if np.any((gamma < -tolerance) | (gamma > 1.0 + tolerance)):
        failures.append("gamma leaves [0, 1]")
    if len(gamma) > 1:
        if np.any(np.diff(gamma) > tolerance):
            failures.append("gamma is not monotonically non-increasing")
        if abs(gamma[-1]) > tolerance:
            failures.append("final gamma is not zero")

    coordinates = artifact.task_coordinates(q)
    coordinate_violation = np.maximum(
        np.maximum(artifact.coordinate_min - coordinates, 0.0),
        np.maximum(coordinates - artifact.coordinate_max, 0.0),
    )
    max_coordinate_violation = float(coordinate_violation.max())
    if max_coordinate_violation > tolerance:
        failures.append(
            f"PCA coordinate-bound violation {max_coordinate_violation:.3e}"
        )
    projection = artifact.reconstruct(coordinates)
    projection_rmse = float(np.sqrt(np.mean(np.square(q - projection))))
    imitation_frames = gamma >= 0.999
    imitation_endpoint_error = (
        float(recomputed_error[imitation_frames].mean())
        if np.any(imitation_frames)
        else float("nan")
    )

    return TrajectoryValidation(
        failures=tuple(failures),
        metrics={
            "frames": float(frame_count),
            "max_joint_limit_violation_rad": max_limit_violation,
            "max_fk_difference_m": max_fk_difference,
            "max_stored_error_difference_m": max_error_difference,
            "max_pca_coordinate_violation": max_coordinate_violation,
            "pca_projection_rmse_rad": projection_rmse,
            "mean_fingertip_error_m": float(recomputed_error.mean()),
            "p95_fingertip_error_m": float(
                np.percentile(recomputed_error, 95.0)
            ),
            "imitation_endpoint_error_m": imitation_endpoint_error,
        },
    )
