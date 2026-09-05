#!/usr/bin/env python3
"""Validate every trajectory and PCA artifact in a Revo2 offline run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dextrah_lab.retargeting import (  # noqa: E402
    PCAArtifact,
    REVO2_RIGHT_ACTUATED_JOINTS,
    Revo2Kinematics,
)
from dextrah_lab.retargeting.validation import validate_trajectory_arrays  # noqa: E402


DEFAULT_URDF = (
    REPOSITORY_ROOT
    / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--minimum-retained-variance", type=float, default=0.98)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    summary_path = args.run_dir / "summary.json"
    artifact_path = args.run_dir / "revo2_human_motion_pca.npz"
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    artifact = PCAArtifact.load(artifact_path)
    hand = Revo2Kinematics(args.urdf)
    failures: list[str] = []
    if artifact.joint_names != REVO2_RIGHT_ACTUATED_JOINTS:
        failures.append("artifact joint order does not match Revo2 hardware order")
    expected_urdf_sha256 = artifact.metadata.get("urdf_sha256")
    selected_urdf_sha256 = _sha256(args.urdf)
    if (
        expected_urdf_sha256 is not None
        and expected_urdf_sha256 != selected_urdf_sha256
    ):
        failures.append(
            "selected URDF checksum differs from the retargeting artifact: "
            f"expected {expected_urdf_sha256}, got {selected_urdf_sha256}"
        )
    if artifact.retained_variance < args.minimum_retained_variance:
        failures.append(
            f"retained variance {artifact.retained_variance:.6f} is below "
            f"{args.minimum_retained_variance:.6f}"
        )
    if len(summary.get("trajectories", [])) == 0:
        failures.append("summary contains no trajectories")

    validations = []
    metric_rows: list[dict[str, float]] = []
    for item in summary.get("trajectories", []):
        path = args.run_dir / item["trajectory"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        validation = validate_trajectory_arrays(
            arrays, hand, artifact, tolerance=args.tolerance
        )
        identifier = f"{item['subject']}/{item['capture']}/{item['grasp_mode']}"
        validations.append(
            {
                "trajectory": identifier,
                "passed": validation.passed,
                "failures": list(validation.failures),
                "metrics": validation.metrics,
            }
        )
        metric_rows.append(validation.metrics)
        failures.extend(f"{identifier}: {failure}" for failure in validation.failures)

    total_frames = int(sum(row.get("frames", 0.0) for row in metric_rows))
    report = {
        "passed": not failures,
        "failures": failures,
        "artifact": {
            "latent_dim": artifact.latent_dim,
            "retained_variance": artifact.retained_variance,
            "orthonormal_error": float(
                np.max(
                    np.abs(
                        artifact.components @ artifact.components.T
                        - np.eye(artifact.latent_dim)
                    )
                )
            ),
            "urdf_sha256": selected_urdf_sha256,
        },
        "dataset": {
            "trajectories": len(validations),
            "frames": total_frames,
            "max_fk_difference_m": max(
                (row.get("max_fk_difference_m", 0.0) for row in metric_rows),
                default=float("nan"),
            ),
            "max_joint_limit_violation_rad": max(
                (
                    row.get("max_joint_limit_violation_rad", 0.0)
                    for row in metric_rows
                ),
                default=float("nan"),
            ),
            "mean_pca_projection_rmse_rad": float(
                np.mean(
                    [
                        row["pca_projection_rmse_rad"]
                        for row in metric_rows
                        if "pca_projection_rmse_rad" in row
                    ]
                )
            ),
            "mean_imitation_endpoint_error_m": float(
                np.nanmean(
                    [
                        row.get("imitation_endpoint_error_m", float("nan"))
                        for row in metric_rows
                    ]
                )
            ),
        },
        "trajectories": validations,
    }
    output = args.output or args.run_dir / "validation.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("passed", "failures", "artifact", "dataset")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
