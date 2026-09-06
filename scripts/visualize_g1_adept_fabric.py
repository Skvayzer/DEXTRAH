#!/usr/bin/env python3
"""Audit the reduced G1/Revo2 ADEPT fabric in a Viser browser.

This viewer is deliberately independent of Isaac Sim.  Pinocchio evaluates the
exact Play2Perfect G1+BrainCo URDF and its analytic Jacobians; the same reduced
fabric and collision specification used by training advance the displayed
13-D state.  Run ``--validate-only`` first to compare every sphere Jacobian
against finite differences without starting a web server.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pinocchio as pin
import torch
import trimesh
import viser
from viser.extras import ViserUrdf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dextrah_lab.g1_adept import (  # noqa: E402
    G1_REVO2_CANONICAL_JOINT_NAMES,
    G1_REVO2_DYNAMIC_SPHERES,
    G1_REVO2_FIXED_SPHERES,
    G1_REVO2_MIMIC_RULES,
    G1_REVO2_SELF_COLLISION_PAIRS,
    ReducedAdeptFabric,
    ReducedAdeptFabricConfig,
    build_g1_collision_batch,
    point_jacobian,
    reduce_jacobian_to_canonical,
)
from dextrah_lab.retargeting import PCAArtifact, REVO2_RIGHT_ACTUATED_JOINTS  # noqa: E402


G1_URDF_RELATIVE_PATH = Path(
    "unitree_ros/robots/g1_with_brainco_hand/"
    "g1_29dof_mode_15_brainco_hand.urdf"
)
TABLE_CENTER = np.asarray((0.42, 0.0, -0.10), dtype=np.float64)
TABLE_DIMENSIONS = np.asarray((0.40, 0.475, 0.30), dtype=np.float64)
TABLE_HEIGHT = 0.05
HAND_SLICE = slice(7, 13)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=Path,
        help="full G1+BrainCo URDF (defaults to a sibling Play2Perfect checkout)",
    )
    parser.add_argument(
        "--play2perfect-root",
        type=Path,
        help="Play2Perfect checkout used to resolve the default URDF",
    )
    parser.add_argument("--pca-artifact", type=Path)
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8089)
    parser.add_argument("--viewer-fps", type=float, default=20.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate sphere Jacobians and a fabric rollout, then exit",
    )
    return parser.parse_args()


def _resolve_urdf(args: argparse.Namespace) -> Path:
    if args.urdf is not None:
        candidates = (args.urdf,)
    elif args.play2perfect_root is not None:
        candidates = (args.play2perfect_root / G1_URDF_RELATIVE_PATH,)
    else:
        candidates = (
            REPOSITORY_ROOT.parent / "play2perfect" / G1_URDF_RELATIVE_PATH,
            Path("/home/konstantin.smirnov/data1/play2perfect")
            / G1_URDF_RELATIVE_PATH,
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "G1+BrainCo URDF was not found; pass --urdf or --play2perfect-root. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def _body_frame_id(model: pin.Model, name: str) -> int:
    matches = [
        index
        for index, frame in enumerate(model.frames)
        if frame.name == name and frame.type == pin.FrameType.BODY
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one BODY frame named {name!r}, got {matches}")
    return matches[0]


class G1PinocchioKinematics:
    """Exact full-URDF FK with a 13-D independent-coordinate interface."""

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.canonical_q_ids = np.asarray(
            [self._joint_q_id(name) for name in G1_REVO2_CANONICAL_JOINT_NAMES],
            dtype=np.int64,
        )
        self.canonical_v_ids = torch.tensor(
            [self._joint_v_id(name) for name in G1_REVO2_CANONICAL_JOINT_NAMES],
            dtype=torch.long,
        )
        self.mimic_q_ids = np.asarray(
            [self._joint_q_id(name) for name, _, _, _ in G1_REVO2_MIMIC_RULES],
            dtype=np.int64,
        )
        self.mimic_v_ids = torch.tensor(
            [self._joint_v_id(name) for name, _, _, _ in G1_REVO2_MIMIC_RULES],
            dtype=torch.long,
        )
        self.mimic_source_ids = torch.tensor(
            [source for _, source, _, _ in G1_REVO2_MIMIC_RULES],
            dtype=torch.long,
        )
        self.mimic_multipliers = torch.tensor(
            [multiplier for _, _, multiplier, _ in G1_REVO2_MIMIC_RULES],
            dtype=torch.float64,
        )
        self.mimic_offsets = np.asarray(
            [offset for _, _, _, offset in G1_REVO2_MIMIC_RULES],
            dtype=np.float64,
        )
        self.dynamic_frame_ids = tuple(
            _body_frame_id(self.model, sphere.link_name)
            for sphere in G1_REVO2_DYNAMIC_SPHERES
        )
        self.base_configuration = pin.neutral(self.model)
        self._apply_fixed_posture(self.base_configuration)
        self.lower = self.model.lowerPositionLimit[self.canonical_q_ids].copy()
        self.upper = self.model.upperPositionLimit[self.canonical_q_ids].copy()
        self.lower += 0.02
        self.upper -= 0.02

    def _joint_q_id(self, name: str) -> int:
        joint_id = self.model.getJointId(name)
        if joint_id == 0 or self.model.joints[joint_id].nq != 1:
            raise RuntimeError(f"expected scalar joint {name!r}")
        return int(self.model.joints[joint_id].idx_q)

    def _joint_v_id(self, name: str) -> int:
        joint_id = self.model.getJointId(name)
        if joint_id == 0 or self.model.joints[joint_id].nv != 1:
            raise RuntimeError(f"expected scalar joint {name!r}")
        return int(self.model.joints[joint_id].idx_v)

    def _set_named(self, configuration: np.ndarray, name: str, value: float) -> None:
        joint_id = self.model.getJointId(name)
        if joint_id != 0 and self.model.joints[joint_id].nq == 1:
            configuration[self.model.joints[joint_id].idx_q] = value

    def _apply_fixed_posture(self, configuration: np.ndarray) -> None:
        for side in ("left", "right"):
            self._set_named(configuration, f"{side}_hip_pitch_joint", -0.05)
            self._set_named(configuration, f"{side}_knee_joint", 0.20)
            self._set_named(configuration, f"{side}_ankle_pitch_joint", -0.15)
        for name, value in {
            "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.18,
            "left_elbow_joint": 0.87,
        }.items():
            self._set_named(configuration, name, value)
        # The unusual tip joints in this URDF are fixed by equal [1, 1] limits.
        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                continue
            index = joint.idx_q
            lower = self.model.lowerPositionLimit[index]
            upper = self.model.upperPositionLimit[index]
            if np.isfinite(lower) and lower == upper:
                configuration[index] = lower

    def full_configuration(self, canonical: np.ndarray) -> np.ndarray:
        if canonical.shape != (13,):
            raise ValueError(f"canonical configuration must have shape (13,), got {canonical.shape}")
        configuration = self.base_configuration.copy()
        configuration[self.canonical_q_ids] = canonical
        for target_q_id, (_, source, multiplier, offset) in zip(
            self.mimic_q_ids, G1_REVO2_MIMIC_RULES
        ):
            configuration[target_q_id] = multiplier * canonical[source] + offset
        return configuration

    def visual_configuration(
        self, canonical: np.ndarray, visual_joint_names: tuple[str, ...]
    ) -> np.ndarray:
        full = self.full_configuration(canonical)
        result = np.zeros(len(visual_joint_names), dtype=np.float64)
        for index, name in enumerate(visual_joint_names):
            joint_id = self.model.getJointId(name)
            if joint_id != 0 and self.model.joints[joint_id].nq == 1:
                result[index] = full[self.model.joints[joint_id].idx_q]
        return result

    def dynamic_positions(self, canonical: np.ndarray) -> np.ndarray:
        full = self.full_configuration(canonical)
        pin.forwardKinematics(self.model, self.data, full)
        pin.updateFramePlacements(self.model, self.data)
        return np.stack(
            [
                self.data.oMf[frame_id].translation
                + self.data.oMf[frame_id].rotation @ np.asarray(sphere.offset)
                for sphere, frame_id in zip(
                    G1_REVO2_DYNAMIC_SPHERES, self.dynamic_frame_ids
                )
            ]
        )

    def dynamic_state(self, canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        full = self.full_configuration(canonical)
        pin.computeJointJacobians(self.model, self.data, full)
        pin.updateFramePlacements(self.model, self.data)
        positions: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        for sphere, frame_id in zip(
            G1_REVO2_DYNAMIC_SPHERES, self.dynamic_frame_ids
        ):
            placement = self.data.oMf[frame_id]
            rotated_offset = placement.rotation @ np.asarray(sphere.offset)
            positions.append(placement.translation + rotated_offset)
            frame_jacobian = pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            shifted = point_jacobian(
                torch.from_numpy(frame_jacobian).unsqueeze(0),
                torch.from_numpy(rotated_offset).unsqueeze(0),
            ).squeeze(0)
            reduced = reduce_jacobian_to_canonical(
                shifted,
                self.canonical_v_ids,
                self.mimic_v_ids,
                self.mimic_source_ids,
                self.mimic_multipliers,
            )
            jacobians.append(reduced.numpy())
        return np.stack(positions), np.stack(jacobians)


def _target_presets() -> dict[str, np.ndarray]:
    # The first four arm poses were solved against the exact URDF.  The unsafe
    # targets are intentional and should visibly differ from the fabric path.
    approach = np.asarray((-0.4640, 0.2626, 0.1621, 0.2849, 0.0049, 0.1120, 0.0304))
    low = np.asarray((-0.6139, 0.2460, 0.0539, 0.7807, 0.0028, 0.2003, 0.0068))
    torso = np.asarray((0.1313, -0.0410, 1.0558, -0.5039, -0.1117, -0.3386, 0.2903))
    limit = np.asarray((-3.04, 1.53, 2.55, -0.99, 1.90, -1.55, 1.55))
    return {
        "object approach · open hand": np.concatenate((approach, np.zeros(6))),
        "object approach · power grasp": np.concatenate(
            (approach, np.asarray((0.65, 0.88, 1.20, 1.18, 1.15, 1.12)))
        ),
        "object approach · thumb-index pinch": np.concatenate(
            (approach, np.asarray((0.75, 0.92, 1.20, 0.18, 0.10, 0.05)))
        ),
        "unsafe · below table": np.concatenate((low, np.full(6, 0.25))),
        "unsafe · through torso": np.concatenate((torso, np.full(6, 0.35))),
        "unsafe · joint limits": np.concatenate((limit, np.full(6, 1.40))),
    }


def _apply_pca_prior(
    target: np.ndarray, artifact: PCAArtifact | None, weight: float
) -> np.ndarray:
    if artifact is None or weight <= 0.0:
        return target
    hand = target[HAND_SLICE]
    coordinates = artifact.task_coordinates(hand[None, :])
    projected = artifact.reconstruct(coordinates)[0]
    result = target.copy()
    result[HAND_SLICE] = (1.0 - weight) * hand + weight * projected
    return result


def _fixed_positions() -> np.ndarray:
    return np.asarray([sphere.center for sphere in G1_REVO2_FIXED_SPHERES])


def _collision_state(
    kinematics: G1PinocchioKinematics, canonical: np.ndarray
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    positions, jacobians = kinematics.dynamic_state(canonical)
    collision = build_g1_collision_batch(
        torch.from_numpy(positions).unsqueeze(0),
        torch.from_numpy(jacobians).unsqueeze(0),
        torch.from_numpy(_fixed_positions()).unsqueeze(0),
        table_height=TABLE_HEIGHT,
    )
    return positions, jacobians, collision.clearance[0]


def _minimum_clearance_per_dynamic_sphere(positions: np.ndarray) -> np.ndarray:
    radii = np.asarray([sphere.radius for sphere in G1_REVO2_DYNAMIC_SPHERES])
    fixed = _fixed_positions()
    fixed_radii = np.asarray([sphere.radius for sphere in G1_REVO2_FIXED_SPHERES])
    minimum = np.full(len(positions), np.inf)
    for index, sphere in enumerate(G1_REVO2_DYNAMIC_SPHERES):
        if sphere.avoid_body:
            clearance = np.linalg.norm(positions[index] - fixed, axis=-1)
            clearance -= radii[index] + fixed_radii
            minimum[index] = min(minimum[index], float(clearance.min()))
        if sphere.avoid_table:
            minimum[index] = min(
                minimum[index], positions[index, 2] - radii[index] - TABLE_HEIGHT
            )
    for first, second in G1_REVO2_SELF_COLLISION_PAIRS:
        clearance = (
            np.linalg.norm(positions[first] - positions[second])
            - radii[first]
            - radii[second]
        )
        minimum[first] = min(minimum[first], clearance)
        minimum[second] = min(minimum[second], clearance)
    return minimum


def _validate(
    kinematics: G1PinocchioKinematics,
) -> None:
    canonical = _target_presets()["object approach · thumb-index pinch"].copy()
    positions, analytic = kinematics.dynamic_state(canonical)
    epsilon = 1.0e-6
    numerical = np.empty_like(analytic)
    for joint_index in range(13):
        delta = np.zeros(13)
        delta[joint_index] = epsilon
        forward = kinematics.dynamic_positions(canonical + delta)
        backward = kinematics.dynamic_positions(canonical - delta)
        numerical[..., joint_index] = (forward - backward) / (2.0 * epsilon)
    error = np.abs(analytic - numerical)
    if error.max() > 2.0e-5:
        sphere, axis, joint = np.unravel_index(np.argmax(error), error.shape)
        raise RuntimeError(
            "sphere Jacobian finite-difference check failed: "
            f"max={error.max():.3e} at sphere={G1_REVO2_DYNAMIC_SPHERES[sphere].name}, "
            f"axis={axis}, joint={G1_REVO2_CANONICAL_JOINT_NAMES[joint]}"
        )

    lower = torch.from_numpy(kinematics.lower)
    upper = torch.from_numpy(kinematics.upper)
    fabric = ReducedAdeptFabric(
        lower,
        upper,
        config=ReducedAdeptFabricConfig(timestep=1.0 / 60.0),
    )
    fabric.reset(torch.zeros(1, 13, dtype=torch.float64))
    target = torch.from_numpy(canonical).unsqueeze(0)
    for _ in range(120):
        state = fabric.state
        assert state is not None
        _, jacobians, collision_clearance = _collision_state(
            kinematics, state.position[0].numpy()
        )
        positions = kinematics.dynamic_positions(state.position[0].numpy())
        collision = build_g1_collision_batch(
            torch.from_numpy(positions).unsqueeze(0),
            torch.from_numpy(jacobians).unsqueeze(0),
            torch.from_numpy(_fixed_positions()).unsqueeze(0),
            table_height=TABLE_HEIGHT,
        )
        fabric.step(target, collision)
        if not torch.isfinite(collision_clearance).all():
            raise RuntimeError("non-finite collision clearance during rollout")
    assert fabric.state is not None
    if not all(
        torch.isfinite(value).all()
        for value in (
            fabric.state.position,
            fabric.state.velocity,
            fabric.state.acceleration,
        )
    ):
        raise RuntimeError("non-finite fabric state during rollout")
    print(
        "validated exact G1+BrainCo kinematics: "
        f"13 joints, {len(positions)} moving spheres, "
        f"Jacobian max error {error.max():.3e}, 120 finite fabric steps"
    )


def main() -> None:
    args = _arguments()
    if args.viewer_fps <= 0.0:
        raise ValueError("--viewer-fps must be positive")
    urdf_path = _resolve_urdf(args)
    kinematics = G1PinocchioKinematics(urdf_path)
    _validate(kinematics)
    if args.validate_only:
        return

    artifact = None
    if args.pca_artifact is not None:
        artifact = PCAArtifact.load(args.pca_artifact)
        if artifact.joint_names != REVO2_RIGHT_ACTUATED_JOINTS:
            raise ValueError("PCA artifact joint order does not match the Revo2 action order")

    presets = _target_presets()
    server = viser.ViserServer(host=args.viser_host, port=args.viser_port)
    server.configure_theme(
        control_layout="collapsible",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=(242, 151, 39),
    )
    server.set_up_direction("+z")
    server.add_grid(
        "/ground",
        width=1.2,
        height=1.2,
        width_segments=24,
        height_segments=24,
        cell_size=0.05,
        section_size=0.25,
        position=(0.15, 0.0, -0.25),
    )

    @server.on_client_connect
    def _initialize_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.95, -0.95, 0.72)
        client.camera.look_at = (0.18, -0.02, 0.14)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    server.add_gui_markdown(
        "## G1 + BrainCo ADEPT fabric audit\n"
        "Coordinates are pelvis-relative but reproduce the Play2Perfect table "
        "placement. Green spheres are safe, orange are inside the 8 cm fabric "
        "influence band, and red overlap. Small cyan spheres show the unfiltered "
        "policy target. The manipulated tool is intentionally absent from the "
        "avoidance field so contact-rich grasping remains possible."
    )
    preset = server.add_gui_dropdown(
        "Target preset", tuple(presets), initial_value=next(iter(presets))
    )
    run_controller = server.add_gui_checkbox("Run controller", initial_value=False)
    enable_fabric = server.add_gui_checkbox("Enable fabric filtering", initial_value=True)
    include_self = server.add_gui_checkbox("Non-adjacent self collision", initial_value=True)
    show_collision = server.add_gui_checkbox("Show collision spheres", initial_value=True)
    show_raw_target = server.add_gui_checkbox("Show raw target centers", initial_value=True)
    steps_per_frame = server.add_gui_slider(
        "60 Hz steps per frame", min=1, max=10, step=1, initial_value=3
    )
    pca_weight = server.add_gui_slider(
        "Frozen PCA hand-prior weight",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=0.25 if artifact is not None else 0.0,
        disabled=artifact is None,
    )
    step_button = server.add_gui_button("Single controller step", color="blue")
    reset_button = server.add_gui_button("Reset to zero pose", color="orange")
    clearance_status = server.add_gui_text("Clearance", "initializing", disabled=True)
    motion_status = server.add_gui_text("Fabric state", "initializing", disabled=True)
    target_status = server.add_gui_text("13-D target [rad]", "initializing", disabled=True)

    single_step_requested = threading.Event()
    reset_requested = threading.Event()

    @step_button.on_click
    def _single_step(_event) -> None:
        single_step_requested.set()

    @reset_button.on_click
    def _reset(_event) -> None:
        reset_requested.set()

    robot = ViserUrdf(
        server,
        urdf_path,
        root_node_name="/g1",
        mesh_color_override=(0.70, 0.73, 0.80),
    )
    visual_joint_names = tuple(robot.get_actuated_joint_names())
    table_mesh = trimesh.creation.box(extents=TABLE_DIMENSIONS)
    server.add_mesh_simple(
        "/table",
        np.asarray(table_mesh.vertices, dtype=np.float32),
        np.asarray(table_mesh.faces, dtype=np.uint32),
        color=(116, 78, 48),
        opacity=0.72,
        position=TABLE_CENTER,
    )

    fixed_handles = [
        server.add_icosphere(
            f"/collision/fixed/{sphere.name}",
            radius=sphere.radius,
            color=(145, 95, 220),
            position=sphere.center,
        )
        for sphere in G1_REVO2_FIXED_SPHERES
    ]
    dynamic_handles: list[tuple[viser.MeshHandle, viser.MeshHandle, viser.MeshHandle]] = []
    raw_target_handles: list[viser.MeshHandle] = []
    for sphere in G1_REVO2_DYNAMIC_SPHERES:
        dynamic_handles.append(
            (
                server.add_icosphere(
                    f"/collision/dynamic/{sphere.name}/safe",
                    radius=sphere.radius,
                    color=(50, 205, 105),
                ),
                server.add_icosphere(
                    f"/collision/dynamic/{sphere.name}/active",
                    radius=sphere.radius,
                    color=(255, 157, 40),
                    visible=False,
                ),
                server.add_icosphere(
                    f"/collision/dynamic/{sphere.name}/penetrating",
                    radius=sphere.radius,
                    color=(235, 55, 60),
                    visible=False,
                ),
            )
        )
        raw_target_handles.append(
            server.add_icosphere(
                f"/target/{sphere.name}",
                radius=max(0.006, 0.32 * sphere.radius),
                color=(40, 205, 245),
            )
        )

    lower = torch.from_numpy(kinematics.lower)
    upper = torch.from_numpy(kinematics.upper)
    fabric = ReducedAdeptFabric(
        lower,
        upper,
        config=ReducedAdeptFabricConfig(timestep=1.0 / 60.0),
    )
    fabric.reset(torch.zeros(1, 13, dtype=torch.float64))
    influence_distance = fabric.config.collision_influence_distance

    try:
        while True:
            if reset_requested.is_set():
                fabric.reset(torch.zeros(1, 13, dtype=torch.float64))
                reset_requested.clear()

            selected_target = _apply_pca_prior(
                presets[str(preset.value)], artifact, float(pca_weight.value)
            )
            selected_target = np.clip(selected_target, kinematics.lower, kinematics.upper)
            requested_steps = 0
            if bool(run_controller.value):
                requested_steps = int(steps_per_frame.value)
            if single_step_requested.is_set():
                requested_steps = max(requested_steps, 1)
                single_step_requested.clear()

            if not bool(enable_fabric.value):
                fabric.reset(torch.from_numpy(selected_target).unsqueeze(0))
            else:
                for _ in range(requested_steps):
                    assert fabric.state is not None
                    positions, jacobians = kinematics.dynamic_state(
                        fabric.state.position[0].numpy()
                    )
                    collision = build_g1_collision_batch(
                        torch.from_numpy(positions).unsqueeze(0),
                        torch.from_numpy(jacobians).unsqueeze(0),
                        torch.from_numpy(_fixed_positions()).unsqueeze(0),
                        table_height=TABLE_HEIGHT,
                        include_self_collision=bool(include_self.value),
                    )
                    fabric.step(torch.from_numpy(selected_target).unsqueeze(0), collision)

            assert fabric.state is not None
            canonical = fabric.state.position[0].numpy()
            robot.update_cfg(kinematics.visual_configuration(canonical, visual_joint_names))
            positions = kinematics.dynamic_positions(canonical)
            raw_positions = kinematics.dynamic_positions(selected_target)
            minimum = _minimum_clearance_per_dynamic_sphere(positions)
            for index, (handles, position, raw_position) in enumerate(
                zip(dynamic_handles, positions, raw_positions)
            ):
                state_index = 0 if minimum[index] >= influence_distance else 1
                if minimum[index] < 0.0:
                    state_index = 2
                for handle_index, handle in enumerate(handles):
                    handle.position = position
                    handle.visible = bool(show_collision.value) and handle_index == state_index
                raw_target_handles[index].position = raw_position
                raw_target_handles[index].visible = bool(show_raw_target.value)
            for handle in fixed_handles:
                handle.visible = bool(show_collision.value)

            active = int(np.count_nonzero(minimum < influence_distance))
            penetrating = int(np.count_nonzero(minimum < 0.0))
            closest_index = int(np.argmin(minimum))
            clearance_status.value = (
                f"minimum={1000.0 * minimum[closest_index]:+.1f} mm at "
                f"{G1_REVO2_DYNAMIC_SPHERES[closest_index].name} · "
                f"active spheres={active}/{len(minimum)} · overlaps={penetrating}"
            )
            motion_status.value = (
                f"|qd|max={fabric.state.velocity.abs().max().item():.3f} rad/s · "
                f"|qdd|max={fabric.state.acceleration.abs().max().item():.2f} rad/s²"
            )
            target_status.value = np.array2string(
                selected_target, precision=2, suppress_small=True
            )
            time.sleep(1.0 / args.viewer_fps)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
