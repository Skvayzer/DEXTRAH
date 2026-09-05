"""Load palm-relative human hand trajectories from the DexYCB release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np
import yaml


DEXYCB_JOINT_NAMES = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)
DEXYCB_FINGERTIP_INDICES = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)
_FRAME_NUMBER = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class DexYCBSequence:
    """One camera's ground-truth trajectory for a unique DexYCB capture."""

    subject: str
    capture: str
    camera_serial: str
    mano_side: str
    frame_indices: np.ndarray
    joints_camera: np.ndarray
    object_translation_camera: np.ndarray | None
    ycb_grasp_id: int | None

    def palm_relative_fingertips(self) -> np.ndarray:
        return palm_relative_fingertips(self.joints_camera, mano_side=self.mano_side)


def _frame_number(path: Path) -> int:
    match = _FRAME_NUMBER.search(path.stem)
    if match is None:
        raise ValueError(f"cannot parse frame number from {path.name!r}")
    return int(match.group(1))


def discover_capture_dirs(root: str | Path) -> list[Path]:
    """Discover unique capture directories without duplicating camera views."""

    root = Path(root)
    return sorted(path.parent for path in root.glob("*-subject-*/*/meta.yml"))


def _camera_dirs(capture_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in capture_dir.iterdir()
        if path.is_dir() and any(path.glob("labels_*.npz"))
    )


def load_sequence(
    capture_dir: str | Path,
    *,
    camera_serial: str | None = None,
    trim_to_object_motion: bool = True,
    motion_margin: int = 40,
) -> DexYCBSequence:
    """Load a sequence from the official per-camera ``joint_3d`` labels.

    A single camera is selected deliberately: all eight cameras observe the
    same physical trial, so treating every view as a separate demonstration
    would bias PCA toward captures with more valid annotations.
    """

    capture_dir = Path(capture_dir)
    with (capture_dir / "meta.yml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    candidates = _camera_dirs(capture_dir)
    if camera_serial is not None:
        candidates = [path for path in candidates if path.name == camera_serial]
    if not candidates:
        selected = camera_serial if camera_serial is not None else "any camera"
        raise FileNotFoundError(f"no DexYCB labels for {selected} under {capture_dir}")
    camera_dir = candidates[0]

    frames: list[int] = []
    joints: list[np.ndarray] = []
    object_translations: list[np.ndarray] = []
    have_all_object_poses = True
    grasp_index = metadata.get("ycb_grasp_ind")
    ycb_ids = metadata.get("ycb_ids", [])
    grasp_id = None
    if grasp_index is not None and 0 <= int(grasp_index) < len(ycb_ids):
        grasp_id = int(ycb_ids[int(grasp_index)])

    for label_path in sorted(camera_dir.glob("labels_*.npz"), key=_frame_number):
        with np.load(label_path) as label:
            if "joint_3d" not in label:
                raise KeyError(f"{label_path} does not contain joint_3d")
            frame_joints = np.asarray(label["joint_3d"], dtype=np.float64).squeeze()
            if frame_joints.shape != (21, 3):
                raise ValueError(
                    f"joint_3d in {label_path} must have shape (1, 21, 3), "
                    f"got {label['joint_3d'].shape}"
                )
            if not np.isfinite(frame_joints).all() or np.all(frame_joints == -1.0):
                continue

            frames.append(_frame_number(label_path))
            joints.append(frame_joints)
            if grasp_index is None or "pose_y" not in label:
                have_all_object_poses = False
            else:
                pose_y = np.asarray(label["pose_y"], dtype=np.float64)
                index = int(grasp_index)
                if pose_y.ndim != 3 or pose_y.shape[1:] != (3, 4) or index >= len(pose_y):
                    have_all_object_poses = False
                else:
                    object_translations.append(pose_y[index, :, 3])

    if not joints:
        raise ValueError(f"no valid hand annotations under {camera_dir}")

    frame_array = np.asarray(frames, dtype=np.int64)
    joint_array = np.stack(joints)
    object_array = None
    if have_all_object_poses and len(object_translations) == len(joints):
        object_array = np.stack(object_translations)

    if trim_to_object_motion and object_array is not None:
        keep = object_motion_window(object_array, margin=motion_margin)
        frame_array = frame_array[keep]
        joint_array = joint_array[keep]
        object_array = object_array[keep]

    mano_sides = metadata.get("mano_sides", [metadata.get("mano_side", "right")])
    mano_side = str(mano_sides[0] if isinstance(mano_sides, list) else mano_sides)
    return DexYCBSequence(
        subject=capture_dir.parent.name,
        capture=capture_dir.name,
        camera_serial=camera_dir.name,
        mano_side=mano_side,
        frame_indices=frame_array,
        joints_camera=joint_array,
        object_translation_camera=object_array,
        ycb_grasp_id=grasp_id,
    )


def iter_sequences(
    root: str | Path,
    *,
    mano_side: str = "right",
    limit: int | None = None,
    trim_to_object_motion: bool = True,
) -> Iterator[DexYCBSequence]:
    """Yield unique captures, skipping the other hand and invalid sequences."""

    yielded = 0
    for capture_dir in discover_capture_dirs(root):
        try:
            sequence = load_sequence(
                capture_dir, trim_to_object_motion=trim_to_object_motion
            )
        except (FileNotFoundError, ValueError):
            continue
        if sequence.mano_side != mano_side:
            continue
        yield sequence
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def palm_frame(joints: np.ndarray, *, mano_side: str = "right") -> tuple[np.ndarray, np.ndarray]:
    """Return wrist origins and palm-frame rotations for 21-joint hands.

    Local ``+z`` points from the wrist toward the middle MCP, local ``+y``
    points toward the index/thumb side for a right hand, and local ``+x`` is
    the palm normal.  Rotation matrices contain these axes as columns.
    """

    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape[-2:] != (21, 3):
        raise ValueError("joints must end in shape (21, 3)")
    if mano_side not in {"right", "left"}:
        raise ValueError("mano_side must be 'right' or 'left'")

    origin = joints[..., 0, :]
    forward = joints[..., 9, :] - origin
    index_to_little = joints[..., 5, :] - joints[..., 17, :]
    if mano_side == "left":
        index_to_little = -index_to_little

    z_axis = _normalize(forward, "wrist-to-middle-MCP")
    lateral = index_to_little - np.sum(index_to_little * z_axis, axis=-1, keepdims=True) * z_axis
    y_axis = _normalize(lateral, "index-to-little-MCP")
    x_axis = _normalize(np.cross(y_axis, z_axis), "palm normal")
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack((x_axis, y_axis, z_axis), axis=-1)
    return origin, rotation


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError(f"degenerate {label} vector in hand annotation")
    return vector / norm


def palm_relative_points(
    points: np.ndarray,
    joints: np.ndarray,
    *,
    mano_side: str = "right",
) -> np.ndarray:
    """Express camera-frame points in the corresponding wrist/palm frame."""

    origin, rotation = palm_frame(joints, mano_side=mano_side)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[:-2] != joints.shape[:-2] or points.shape[-1] != 3:
        raise ValueError("points and joints must have matching batch dimensions")
    centered = points - origin[..., None, :]
    return centered @ rotation


def palm_relative_fingertips(
    joints: np.ndarray, *, mano_side: str = "right"
) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    return palm_relative_points(
        joints[..., DEXYCB_FINGERTIP_INDICES, :],
        joints,
        mano_side=mano_side,
    )


def object_motion_window(
    translation: np.ndarray,
    *,
    margin: int = 40,
    lookahead: int = 5,
    immediate_threshold: float = 0.02,
    future_threshold: float = 0.05,
) -> slice:
    """Return the grasp-centered window used by the public retargeting example."""

    translation = np.asarray(translation, dtype=np.float64)
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError("translation must have shape (frames, 3)")
    if len(translation) < 2:
        return slice(0, len(translation))

    moving = np.zeros(len(translation), dtype=bool)
    for index in range(len(translation) - 1):
        future = min(index + lookahead, len(translation) - 1)
        immediate_delta = np.linalg.norm(translation[index + 1] - translation[index])
        future_delta = np.linalg.norm(translation[future] - translation[index])
        moving[index] = (
            immediate_delta > immediate_threshold or future_delta > future_threshold
        )
    indices = np.flatnonzero(moving)
    if len(indices) == 0:
        return slice(0, len(translation))
    start = max(0, int(indices[0]) - margin)
    stop = min(len(translation), int(indices[-1]) + margin + 1)
    return slice(start, stop)
