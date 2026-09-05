#!/usr/bin/env python3
"""Inspect retargeted DexYCB/Revo2 trajectories and their PCA projections."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import viser
from viser.extras import ViserUrdf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dextrah_lab.retargeting import (  # noqa: E402
    PCAArtifact,
    REVO2_RIGHT_ACTUATED_JOINTS,
    Revo2Kinematics,
)


DEFAULT_URDF = (
    REPOSITORY_ROOT
    / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="trajectory NPZ; repeat to add selectable motion presets",
    )
    parser.add_argument(
        "--trajectory-label",
        action="append",
        help="display label corresponding to each repeated --trajectory",
    )
    parser.add_argument("--pca-artifact", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8089)
    parser.add_argument("--viewer-fps", type=float, default=20.0)
    return parser.parse_args()


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    required = {
        "joint_positions",
        "robot_fingertips",
        "scaled_human_fingertips",
        "gamma",
        "fingertip_error",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"trajectory is missing keys: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def main() -> None:
    args = _arguments()
    if args.viewer_fps <= 0.0:
        raise ValueError("viewer FPS must be positive")
    if args.trajectory_label is not None and len(args.trajectory_label) != len(
        args.trajectory
    ):
        raise ValueError("provide one --trajectory-label for every --trajectory")

    artifact = None
    if args.pca_artifact is not None:
        artifact = PCAArtifact.load(args.pca_artifact)
        if artifact.joint_names != REVO2_RIGHT_ACTUATED_JOINTS:
            raise ValueError("PCA artifact joint order does not match Revo2")

    hand = Revo2Kinematics(args.urdf)
    trajectories: dict[str, tuple[dict[str, np.ndarray], np.ndarray | None]] = {}
    maximum_frames = 0
    for index, path in enumerate(args.trajectory):
        trajectory = _load_trajectory(path)
        q_retargeted = trajectory["joint_positions"]
        if q_retargeted.ndim != 2 or q_retargeted.shape[1] != 6:
            raise ValueError(f"joint_positions in {path} must have shape (frames, 6)")
        if len(q_retargeted) == 0:
            raise ValueError(f"trajectory {path} has no frames")
        recomputed_tips = hand.fingertip_positions(
            torch.as_tensor(q_retargeted, dtype=torch.float64)
        ).numpy()
        if not np.allclose(
            recomputed_tips, trajectory["robot_fingertips"], atol=1.0e-7
        ):
            raise ValueError(
                f"stored robot fingertips in {path} do not match the selected URDF"
            )

        q_pca = None
        if artifact is not None:
            coordinates = artifact.task_coordinates(q_retargeted)
            q_pca = artifact.reconstruct(coordinates)
            q_pca = np.clip(q_pca, artifact.joint_lower, artifact.joint_upper)
        if args.trajectory_label is None:
            capture = str(
                trajectory.get("capture", np.asarray(path.stem)).item()
            )
            mode = str(
                trajectory.get("grasp_mode", np.asarray("unknown")).item()
            )
            label = f"{mode}: {capture}"
        else:
            label = str(args.trajectory_label[index])
        if label in trajectories:
            raise ValueError(f"duplicate trajectory label {label!r}")
        trajectories[label] = (trajectory, q_pca)
        maximum_frames = max(maximum_frames, len(q_retargeted))

    server = viser.ViserServer(host=args.viser_host, port=args.viser_port)
    server.configure_theme(
        control_layout="collapsible",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=(242, 151, 39),
    )
    # Revo2 +x is the palm normal; viewing it as vertical makes finger curl clear.
    server.set_up_direction("+x")
    server.add_grid(
        "/palm_plane",
        width=0.30,
        height=0.30,
        width_segments=15,
        height_segments=15,
        cell_size=0.02,
        section_size=0.10,
        plane="yz",
    )

    @server.on_client_connect
    def _initialize_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.34, 0.28, 0.30)
        client.camera.look_at = (0.035, 0.0, 0.085)
        client.camera.up_direction = (1.0, 0.0, 0.0)

    server.add_gui_markdown(
        "## DexYCB → BrainCo Revo2\n"
        "Choose a motion preset below. Human targets are orange; optimized "
        "robot fingertips are cyan. Switch the robot configuration between "
        "the Adam result and its frozen five-component PCA projection."
    )
    preset = server.add_gui_dropdown(
        "Motion preset",
        tuple(trajectories),
        initial_value=next(iter(trajectories)),
    )
    frame_slider = server.add_gui_slider(
        "Frame", min=0, max=maximum_frames - 1, step=1, initial_value=0
    )
    playback = server.add_gui_checkbox("Play", initial_value=False)
    playback_fps = server.add_gui_slider(
        "Playback FPS", min=1.0, max=60.0, step=1.0, initial_value=15.0
    )
    display_options = ["retargeted Adam result"]
    if artifact is not None:
        display_options.append("five-component PCA projection")
    display = server.add_gui_dropdown(
        "Robot configuration", tuple(display_options), initial_value=display_options[0]
    )
    status = server.add_gui_text("Fit", "initializing", disabled=True)
    joint_status = server.add_gui_text("Six actuator targets [rad]", "", disabled=True)

    robot_urdf = ViserUrdf(
        server,
        args.urdf,
        root_node_name="/revo2",
        mesh_color_override=(0.68, 0.72, 0.80),
    )
    urdf_joint_names = tuple(robot_urdf.get_actuated_joint_names())
    missing_visual_joints = set(REVO2_RIGHT_ACTUATED_JOINTS).difference(
        urdf_joint_names
    )
    if missing_visual_joints:
        raise RuntimeError(
            "Viser URDF is missing Revo2 command joints: "
            f"{sorted(missing_visual_joints)}"
        )
    visual_joint_indices = tuple(
        urdf_joint_names.index(name) for name in REVO2_RIGHT_ACTUATED_JOINTS
    )
    visual_configuration = np.zeros(len(urdf_joint_names), dtype=np.float64)

    human_markers = [
        server.add_icosphere(
            f"/targets/human/{index}", radius=0.0045, color=(255, 145, 25)
        )
        for index in range(5)
    ]
    robot_markers = [
        server.add_icosphere(
            f"/targets/robot/{index}", radius=0.0040, color=(35, 220, 245)
        )
        for index in range(5)
    ]
    error_segments = [
        server.add_spline_catmull_rom(
            f"/targets/errors/{index}",
            np.zeros((2, 3), dtype=np.float32),
            curve_type="catmullrom",
            line_width=2.0,
            color=(255, 80, 65),
            segments=2,
        )
        for index in range(5)
    ]

    last_frame = -1
    last_display = ""
    last_preset = ""
    next_playback = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            selected_preset = str(preset.value)
            trajectory, q_pca = trajectories[selected_preset]
            num_frames = len(trajectory["joint_positions"])
            if playback.value and now >= next_playback:
                frame_slider.value = (int(frame_slider.value) + 1) % num_frames
                next_playback = now + 1.0 / float(playback_fps.value)

            frame = min(int(frame_slider.value), num_frames - 1)
            if frame != int(frame_slider.value):
                frame_slider.value = frame
            selected_display = str(display.value)
            if (
                frame != last_frame
                or selected_display != last_display
                or selected_preset != last_preset
            ):
                q = trajectory["joint_positions"][frame]
                if selected_display.startswith("five-component"):
                    assert q_pca is not None
                    q = q_pca[frame]
                visual_configuration[list(visual_joint_indices)] = q
                robot_urdf.update_cfg(visual_configuration)
                tips = hand.fingertip_positions(
                    torch.as_tensor(q, dtype=torch.float64)
                ).numpy()
                human = trajectory["scaled_human_fingertips"][frame]
                for marker, position in zip(human_markers, human):
                    marker.position = position
                for marker, position in zip(robot_markers, tips):
                    marker.position = position
                for segment, human_tip, robot_tip in zip(
                    error_segments, human, tips
                ):
                    segment.positions = np.stack((human_tip, robot_tip)).astype(
                        np.float32
                )
                errors = np.linalg.norm(tips - human, axis=-1)
                capture = str(
                    trajectory.get("capture", np.asarray("unknown")).item()
                )
                mode = str(
                    trajectory.get("grasp_mode", np.asarray("unknown")).item()
                )
                status.value = (
                    f"{capture} · {mode} · frame {frame + 1}/{num_frames} · "
                    f"gamma={trajectory['gamma'][frame]:.3f} · "
                    f"mean tip error={1000.0 * errors.mean():.1f} mm · "
                    f"max={1000.0 * errors.max():.1f} mm"
                )
                joint_status.value = np.array2string(
                    q, precision=3, suppress_small=True
                )
                last_frame = frame
                last_display = selected_display
                last_preset = selected_preset
            time.sleep(1.0 / args.viewer_fps)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
