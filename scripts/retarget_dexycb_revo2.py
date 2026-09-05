#!/usr/bin/env python3
"""Build the offline DexYCB -> Revo2 -> PCA action representation.

This is deliberately separate from PPO.  It reproduces the ordering in
DextrAH-G Appendix D: human trajectories are retargeted first, then PCA is fit
once and saved as a frozen task map for policy training.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dextrah_lab.retargeting import (  # noqa: E402
    REVO2_RIGHT_ACTUATED_JOINTS,
    RetargetingConfig,
    Revo2Kinematics,
    Revo2Retargeter,
    fit_pca_action_space,
    iter_sequences,
)


DEFAULT_URDF = (
    REPOSITORY_ROOT
    / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--modes",
        default="power,precision_tripod",
        help="comma-separated power, precision, and/or precision_tripod",
    )
    parser.add_argument("--sequence-limit", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-sequence", type=int)
    parser.add_argument(
        "--no-motion-trim", action="store_true", help="retain each entire capture"
    )
    parser.add_argument(
        "--scale",
        default="auto",
        help="human-to-robot spatial scale, or 'auto' for pre-grasp IK search",
    )
    parser.add_argument("--scale-search-min", type=float, default=0.65)
    parser.add_argument("--scale-search-max", type=float, default=1.25)
    parser.add_argument("--scale-search-steps", type=int, default=13)
    parser.add_argument("--scale-calibration-frames", type=int, default=64)
    parser.add_argument("--scale-search-iterations", type=int, default=150)
    parser.add_argument("--regularization-weight", type=float, default=1.0e-3)
    parser.add_argument("--learning-rate", type=float, default=3.0e-2)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--minimum-iterations", type=int, default=40)
    parser.add_argument("--convergence-tolerance", type=float, default=1.0e-9)
    parser.add_argument(
        "--optimizer-execution",
        choices=("batched", "sequential"),
        default="batched",
        help="vectorized independent frames (fast) or trajectory warm starts",
    )
    parser.add_argument(
        "--gamma-schedule", choices=("endpoint", "paper_literal"), default="endpoint"
    )
    parser.add_argument(
        "--pca-components",
        default="5",
        help="fixed component count (paper uses 5), or 'auto'",
    )
    parser.add_argument("--variance-threshold", type=float, default=0.98)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"power", "precision", "precision_tripod"}
    unknown = set(modes).difference(allowed)
    if not modes or unknown:
        raise ValueError(f"invalid grasp modes: {sorted(unknown) if unknown else value!r}")
    return modes


def _parse_components(value: str) -> int | None:
    if value == "auto":
        return None
    components = int(value)
    if not 1 <= components <= len(REVO2_RIGHT_ACTUATED_JOINTS):
        raise ValueError("PCA component count must lie in [1, 6]")
    return components


def _parse_scale(value: str) -> float | None:
    if value == "auto":
        return None
    scale = float(value)
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output(output_dir: Path, *, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new directory or pass --overwrite"
        )
    trajectories = output_dir / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    return trajectories


def _calibrate_scale(
    hand: Revo2Kinematics,
    fingertips: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[float, dict[str, object]]:
    """Choose alpha by minimizing pure-imitation IK error on pre-grasp frames."""

    if not 0.0 < args.scale_search_min < args.scale_search_max:
        raise ValueError("scale search requires 0 < min < max")
    if args.scale_search_steps < 2:
        raise ValueError("scale search steps must be at least two")
    if args.scale_calibration_frames <= 0 or args.scale_search_iterations <= 0:
        raise ValueError("scale calibration frames/iterations must be positive")
    pregrasp = np.stack([tips[0] for tips in fingertips])
    if len(pregrasp) > args.scale_calibration_frames:
        selected = np.linspace(
            0, len(pregrasp) - 1, args.scale_calibration_frames, dtype=np.int64
        )
        pregrasp = pregrasp[selected]
    candidates = np.linspace(
        args.scale_search_min, args.scale_search_max, args.scale_search_steps
    )
    scores: list[float] = []
    for candidate in candidates:
        config = RetargetingConfig(
            mode="power",
            scale=float(candidate),
            regularization_weight=0.0,
            learning_rate=args.learning_rate,
            iterations=args.scale_search_iterations,
            minimum_iterations=min(40, args.scale_search_iterations),
            convergence_tolerance=args.convergence_tolerance,
            gamma_schedule=args.gamma_schedule,
            optimizer_execution="batched",
        )
        result = Revo2Retargeter(hand, config).retarget(
            pregrasp, gamma_override=np.ones(len(pregrasp))
        )
        score = float(result.fingertip_error.mean())
        scores.append(score)
        print(
            f"Scale calibration alpha={candidate:.5f}: "
            f"mean fingertip error={1000.0 * score:.2f} mm",
            flush=True,
        )
    best_index = int(np.argmin(scores))
    if best_index in {0, len(candidates) - 1}:
        print(
            "WARNING: best scale is at the search boundary; expand the range",
            flush=True,
        )
    return float(candidates[best_index]), {
        "method": "pregrasp_pure_imitation_ik_grid",
        "frames": int(len(pregrasp)),
        "iterations": int(args.scale_search_iterations),
        "candidates": candidates.tolist(),
        "mean_fingertip_error_m": scores,
        "best_index": best_index,
    }


def _write_result(
    path: Path,
    *,
    sequence,
    mode: str,
    result,
) -> None:
    """Write numeric/string arrays only; downstream loading never needs pickle."""

    np.savez_compressed(
        path,
        subject=np.asarray(sequence.subject, dtype=np.str_),
        capture=np.asarray(sequence.capture, dtype=np.str_),
        camera_serial=np.asarray(sequence.camera_serial, dtype=np.str_),
        grasp_mode=np.asarray(mode, dtype=np.str_),
        ycb_grasp_id=np.asarray(
            -1 if sequence.ycb_grasp_id is None else sequence.ycb_grasp_id,
            dtype=np.int64,
        ),
        frame_indices=sequence.frame_indices,
        joint_positions=result.joint_positions,
        robot_fingertips=result.robot_fingertips,
        scaled_human_fingertips=result.scaled_human_fingertips,
        gamma=result.gamma,
        total_loss=result.total_loss,
        imitation_loss=result.imitation_loss,
        closure_loss=result.closure_loss,
        regularization_loss=result.regularization_loss,
        optimizer_iterations=result.optimizer_iterations,
        converged=result.converged,
        fingertip_error=result.fingertip_error,
    )


def main() -> None:
    args = _arguments()
    modes = _parse_modes(args.modes)
    pca_components = _parse_components(args.pca_components)
    requested_scale = _parse_scale(args.scale)
    if args.frame_stride <= 0:
        raise ValueError("frame stride must be positive")
    if args.sequence_limit is not None and args.sequence_limit <= 0:
        raise ValueError("sequence limit must be positive")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence <= 0:
        raise ValueError("max frames per sequence must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    trajectory_dir = _prepare_output(args.output_dir, overwrite=args.overwrite)

    print(f"Discovering right-hand DexYCB captures under {args.dataset_root}", flush=True)
    sequences = list(
        iter_sequences(
            args.dataset_root,
            limit=args.sequence_limit,
            trim_to_object_motion=not args.no_motion_trim,
        )
    )
    if not sequences:
        raise RuntimeError("no valid right-hand DexYCB sequences were found")
    hand = Revo2Kinematics(args.urdf).to(device)

    fingertips: list[np.ndarray] = []
    prepared: list[tuple[object, np.ndarray]] = []
    for sequence in sequences:
        indices = np.arange(len(sequence.frame_indices))[:: args.frame_stride]
        if args.max_frames_per_sequence is not None:
            indices = indices[: args.max_frames_per_sequence]
        if len(indices) == 0:
            continue
        # Keep saved DexYCB indices aligned with the optimizer result.
        sequence = replace(
            sequence,
            frame_indices=sequence.frame_indices[indices],
            joints_camera=sequence.joints_camera[indices],
            object_translation_camera=None
            if sequence.object_translation_camera is None
            else sequence.object_translation_camera[indices],
        )
        tips = sequence.palm_relative_fingertips()
        fingertips.append(tips)
        prepared.append((sequence, tips))
    if not prepared:
        raise RuntimeError("frame filtering removed every DexYCB sequence")

    scale_calibration = None
    if requested_scale is None:
        scale, scale_calibration = _calibrate_scale(hand, fingertips, args)
        scale_source = "pregrasp_ik_grid_search"
    else:
        scale = requested_scale
        scale_source = "command_line"
    print(
        f"Loaded {len(prepared)} captures / {sum(len(t) for _, t in prepared)} frames; "
        f"human scale={scale:.6f} ({scale_source})",
        flush=True,
    )

    all_joint_positions: list[np.ndarray] = []
    all_fingertip_errors: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    started = time.monotonic()
    for sequence_index, (sequence, tips) in enumerate(prepared, start=1):
        for mode in modes:
            config = RetargetingConfig(
                mode=mode,
                scale=scale,
                regularization_weight=args.regularization_weight,
                learning_rate=args.learning_rate,
                iterations=args.iterations,
                minimum_iterations=args.minimum_iterations,
                convergence_tolerance=args.convergence_tolerance,
                gamma_schedule=args.gamma_schedule,
                optimizer_execution=args.optimizer_execution,
            )
            result = Revo2Retargeter(hand, config).retarget(tips)
            filename = f"{sequence.subject}_{sequence.capture}_{mode}.npz"
            output_path = trajectory_dir / filename
            _write_result(output_path, sequence=sequence, mode=mode, result=result)
            all_joint_positions.append(result.joint_positions)
            all_fingertip_errors.append(result.fingertip_error)
            manifest.append(
                {
                    "subject": sequence.subject,
                    "capture": sequence.capture,
                    "camera_serial": sequence.camera_serial,
                    "grasp_mode": mode,
                    "frames": int(len(result.joint_positions)),
                    "ycb_grasp_id": sequence.ycb_grasp_id,
                    "mean_fingertip_error_m": float(result.fingertip_error.mean()),
                    "p95_fingertip_error_m": float(
                        np.percentile(result.fingertip_error, 95.0)
                    ),
                    "trajectory": str(output_path.relative_to(args.output_dir)),
                }
            )
        elapsed = time.monotonic() - started
        print(
            f"[{sequence_index}/{len(prepared)}] {sequence.subject}/{sequence.capture} "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )

    positions = np.concatenate(all_joint_positions)
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "DexYCB",
        "paper": "DextrAH-G arXiv:2407.02274 Appendix D",
        "robot": "BrainCo Revo2 right hand",
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": _sha256(args.urdf),
        "grasp_modes": list(modes),
        "scale": scale,
        "scale_source": scale_source,
        "scale_calibration": scale_calibration,
        "gamma_schedule": args.gamma_schedule,
        "optimizer_execution": args.optimizer_execution,
        "sequence_count": len(prepared),
        "retargeted_trajectory_count": len(manifest),
        "frame_stride": args.frame_stride,
        "random_seed": args.seed,
    }
    artifact = fit_pca_action_space(
        positions,
        joint_names=REVO2_RIGHT_ACTUATED_JOINTS,
        joint_lower=hand.lower.detach().cpu().numpy(),
        joint_upper=hand.upper.detach().cpu().numpy(),
        components=pca_components,
        variance_threshold=args.variance_threshold,
        metadata=run_metadata,
    )
    artifact_path = artifact.save(args.output_dir / "revo2_human_motion_pca.npz")

    all_errors = np.concatenate(all_fingertip_errors)
    summary = {
        **run_metadata,
        "optimizer": {
            "regularization_weight": args.regularization_weight,
            "learning_rate": args.learning_rate,
            "iterations": args.iterations,
            "minimum_iterations": args.minimum_iterations,
            "convergence_tolerance": args.convergence_tolerance,
            "execution": args.optimizer_execution,
        },
        "joint_names": list(artifact.joint_names),
        "pca_components": artifact.latent_dim,
        "explained_variance_ratio": artifact.explained_variance_ratio.tolist(),
        "retained_variance": artifact.retained_variance,
        "coordinate_min": artifact.coordinate_min.tolist(),
        "coordinate_max": artifact.coordinate_max.tolist(),
        "mean_fingertip_error_m": float(all_errors.mean()),
        "p95_fingertip_error_m": float(np.percentile(all_errors, 95.0)),
        "max_fingertip_error_m": float(all_errors.max()),
        "elapsed_seconds": time.monotonic() - started,
        "artifact": artifact_path.name,
        "artifact_sha256": _sha256(artifact_path),
        "trajectories": manifest,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"Wrote {artifact_path}; {artifact.latent_dim} components retain "
        f"{100.0 * artifact.retained_variance:.3f}% variance",
        flush=True,
    )


if __name__ == "__main__":
    main()
