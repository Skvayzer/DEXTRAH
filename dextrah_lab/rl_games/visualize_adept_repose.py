"""Inspect the live ADEPT Stage-1 environment in a Viser web client.

The robot joint state, object pose, target pose, contact measurements, and
FABRICS collision spheres all come from the running Isaac Lab environment.
Viser reconstructs only the presentation geometry needed by a web browser.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Adept-Kuka-Allegro-Repose")
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Use at least 16 to expose every Appendix-Fig.-8 primitive.",
)
parser.add_argument("--adr_level", type=int, default=50)
parser.add_argument("--viser_host", default="127.0.0.1")
parser.add_argument("--viser_port", type=int, default=8088)
parser.add_argument("--viewer_fps", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import trimesh
import viser
from isaaclab_tasks.utils import parse_env_cfg
from viser.extras import ViserUrdf

import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup  # noqa: F401, E402
from dextrah_lab.tasks.dextrah_kuka_allegro.adept_mdp import (
    ADEPT_PRIMITIVES,
    contact_gate,
    keypoint_pose_error,
    pose_keypoints,
    quaternion_apply_wxyz,
)


TABLE_DIMENSIONS = (0.725, 1.16, 0.03)
OBJECT_COLORS = (
    (26, 107, 199),
    (5, 140, 107),
    (59, 89, 179),
    (13, 140, 51),
    (122, 41, 161),
    (179, 31, 97),
    (217, 115, 8),
    (235, 184, 5),
)


def _primitive_mesh(shape: str, dimensions: tuple[float, ...]) -> trimesh.Trimesh:
    """Create centered presentation geometry with the same USD dimensions."""

    if shape == "cuboid":
        return trimesh.creation.box(extents=dimensions)
    if shape == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=dimensions[0])
    if shape == "capsule":
        radius, height = dimensions
        mesh = trimesh.creation.capsule(height=height, radius=radius)
        # trimesh places the cylindrical segment between z=0 and z=height;
        # USD Capsule is centered on the origin.
        mesh.apply_translation((0.0, 0.0, -height / 2.0))
        return mesh
    if shape == "cone":
        radius, height = dimensions
        mesh = trimesh.creation.cone(radius=radius, height=height, sections=32)
        # trimesh Cone spans z=[0, height], while the generated USD is centered.
        mesh.apply_translation((0.0, 0.0, -height / 2.0))
        return mesh
    raise ValueError(f"Unsupported ADEPT primitive shape: {shape}")


def _scaled_vertices(mesh: trimesh.Trimesh, scale: float) -> np.ndarray:
    return np.asarray(mesh.vertices, dtype=np.float32) * scale


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _local_root_pose(asset, env_origins: torch.Tensor, env_id: int):
    position = asset.data.root_pos_w[env_id] - env_origins[env_id]
    return _to_numpy(position), _to_numpy(asset.data.root_quat_w[env_id])


def main() -> None:
    if args.num_envs < len(ADEPT_PRIMITIVES):
        raise ValueError(
            f"--num_envs must be at least {len(ADEPT_PRIMITIVES)} to show all objects"
        )
    if not 0 <= args.adr_level <= 50:
        raise ValueError("--adr_level must be in [0, 50]")
    if args.viewer_fps <= 0:
        raise ValueError("--viewer_fps must be positive")

    cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
    )
    cfg.use_cuda_graph = False
    cfg.starting_adr_increments = args.adr_level
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    observations, _extras = env.reset()

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
        width=2.0,
        height=2.0,
        width_segments=20,
        height_segments=20,
        cell_size=0.1,
        section_size=0.5,
        position=(0.0, 0.0, 0.0),
    )

    @server.on_client_connect
    def _initialize_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (1.25, 1.10, 1.20)
        client.camera.look_at = (-0.35, 0.0, 0.48)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    server.add_gui_markdown(
        "## ADEPT Stage-1 live inspection\n"
        "The state below comes from Isaac Lab. The table and primitive meshes "
        "are web renderings of their configured dimensions. **Goal sampling "
        "bounds are inferred**, because ADEPT does not publish them."
    )

    spec_by_name = {spec.name: spec for spec in ADEPT_PRIMITIVES}
    env_object_names = [
        base.object_names[int(base.multi_object_idx[env_id].item())]
        for env_id in range(base.num_envs)
    ]
    visible_object_set = set(env_object_names[: len(ADEPT_PRIMITIVES)])
    expected_object_set = set(spec_by_name)
    if visible_object_set != expected_object_set:
        missing = sorted(expected_object_set - visible_object_set)
        raise RuntimeError(f"first 16 environments do not cover all objects: {missing}")

    env_options = tuple(
        f"{env_id:02d} | {env_object_names[env_id]} | scale "
        f"{float(base.object_scale[env_id, 0].item()):.3f}"
        for env_id in range(base.num_envs)
    )
    env_selector = server.add_gui_dropdown(
        "Environment / object", env_options, initial_value=env_options[0]
    )
    run_physics = server.add_gui_checkbox("Run physics", initial_value=False)
    action_mode = server.add_gui_dropdown(
        "Action input",
        ("zero (hold)", "small random"),
        initial_value="zero (hold)",
    )
    random_amplitude = server.add_gui_slider(
        "Random action amplitude",
        min=0.0,
        max=0.20,
        step=0.01,
        initial_value=0.05,
    )
    show_collision = server.add_gui_checkbox(
        "Show FABRICS collision spheres", initial_value=True
    )
    show_keypoints = server.add_gui_checkbox(
        "Show ADEPT pose keypoints", initial_value=True
    )
    show_pointcloud = server.add_gui_checkbox(
        "Show policy point cloud", initial_value=False
    )
    reset_button = server.add_gui_button("Resample/reset all environments", color="orange")

    object_status = server.add_gui_text("Object", "initializing", disabled=True)
    state_status = server.add_gui_text("State", "initializing", disabled=True)
    force_status = server.add_gui_text("Fingertip forces [N]", "initializing", disabled=True)
    fabric_status = server.add_gui_text("FABRICS", "initializing", disabled=True)

    reset_requested = threading.Event()

    @reset_button.on_click
    def _request_reset(_event) -> None:
        reset_requested.set()

    robot_root = server.add_frame("/robot", show_axes=False)
    robot_urdf = ViserUrdf(
        server,
        Path(base.urdf_path),
        root_node_name="/robot",
        mesh_color_override=(0.72, 0.74, 0.80),
    )
    urdf_joint_names = tuple(robot_urdf.get_actuated_joint_names())
    policy_joint_names = tuple(cfg.actuated_joint_names)
    if urdf_joint_names != policy_joint_names:
        raise RuntimeError(
            "Viser URDF and policy joint order differ: "
            f"URDF={urdf_joint_names}, policy={policy_joint_names}"
        )

    table_mesh = trimesh.creation.box(extents=TABLE_DIMENSIONS)
    table = server.add_mesh_simple(
        "/table",
        np.asarray(table_mesh.vertices, dtype=np.float32),
        np.asarray(table_mesh.faces, dtype=np.uint32),
        color=(116, 78, 48),
        opacity=0.72,
    )
    goal_volume_mesh = trimesh.creation.box(
        extents=2.0 * np.asarray(cfg.inferred_goal_position_half_width)
    )
    server.add_mesh_simple(
        "/goal_sampling_volume_inferred",
        np.asarray(goal_volume_mesh.vertices, dtype=np.float32),
        np.asarray(goal_volume_mesh.faces, dtype=np.uint32),
        color=(255, 165, 35),
        wireframe=True,
        opacity=0.30,
        position=cfg.inferred_goal_center,
    )

    object_handles = []
    goal_handles = []
    pointcloud_handles = []
    object_labels = []
    for env_id, object_name in enumerate(env_object_names):
        spec = spec_by_name[object_name]
        mesh = _primitive_mesh(spec.shape, spec.dimensions)
        scale = float(base.object_scale[env_id, 0].item())
        vertices = _scaled_vertices(mesh, scale)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        color = OBJECT_COLORS[list(spec_by_name).index(object_name) % len(OBJECT_COLORS)]
        object_handles.append(
            server.add_mesh_simple(
                f"/objects/env_{env_id}/current",
                vertices,
                faces,
                color=color,
                flat_shading=False,
                visible=env_id == 0,
            )
        )
        goal_handles.append(
            server.add_mesh_simple(
                f"/objects/env_{env_id}/goal",
                vertices,
                faces,
                color=(50, 220, 90),
                opacity=0.28,
                visible=env_id == 0,
            )
        )
        pointcloud_handles.append(
            server.add_point_cloud(
                f"/objects/env_{env_id}/policy_pointcloud",
                _to_numpy(base.local_object_pointcloud[env_id]) * scale,
                colors=(255, 238, 75),
                point_size=0.004,
                point_shape="circle",
                visible=False,
            )
        )
        object_labels.append(
            server.add_label(
                f"/objects/env_{env_id}/label",
                text=object_name,
                visible=env_id == 0,
            )
        )

    current_frame = server.add_frame(
        "/current_pose", axes_length=0.09, axes_radius=0.004
    )
    goal_frame = server.add_frame(
        "/goal_pose", axes_length=0.09, axes_radius=0.004
    )
    current_keypoints = [
        server.add_icosphere(
            f"/keypoints/current/{index}", radius=0.008, color=(50, 150, 255)
        )
        for index in range(8)
    ]
    goal_keypoints = [
        server.add_icosphere(
            f"/keypoints/goal/{index}", radius=0.008, color=(50, 235, 90)
        )
        for index in range(8)
    ]

    fingertip_markers = [
        server.add_icosphere(
            f"/contacts/{name}/position", radius=0.007, color=(45, 210, 240)
        )
        for name in cfg.hand_body_names
    ]
    contact_markers = [
        server.add_icosphere(
            f"/contacts/{name}/above_threshold",
            radius=0.011,
            color=(255, 55, 40),
            visible=False,
        )
        for name in cfg.hand_body_names
    ]

    fabric = base.kuka_allegro_fabric
    collision_radii = tuple(float(value) for value in fabric.collision_sphere_radii)
    collision_handles = []
    unit_sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    for index, radius in enumerate(collision_radii):
        collision_handles.append(
            server.add_mesh_simple(
                f"/fabric_collision_spheres/{index:02d}",
                np.asarray(unit_sphere.vertices * radius, dtype=np.float32),
                np.asarray(unit_sphere.faces, dtype=np.uint32),
                color=(255, 80, 45) if index < 10 else (245, 195, 35),
                opacity=0.24,
            )
        )

    selected_env_id = -1
    last_reward = torch.zeros(base.num_envs, device=base.device)
    frame_period = 1.0 / args.viewer_fps

    print(
        f"ADEPT_VISER_READY=http://localhost:{server.get_port()} "
        f"objects={len(expected_object_set)} adr_level={args.adr_level}",
        flush=True,
    )
    print(
        "ADEPT_VISER_CONTRACT="
        f"joints={len(urdf_joint_names)} collision_spheres={len(collision_radii)} "
        f"policy_obs={observations['policy'].shape[-1]} "
        f"critic_obs={observations['critic'].shape[-1]}",
        flush=True,
    )

    try:
        while simulation_app.is_running():
            started = time.monotonic()
            if reset_requested.is_set():
                observations, _extras = env.reset()
                last_reward.zero_()
                reset_requested.clear()

            if run_physics.value:
                if action_mode.value == "small random":
                    amplitude = float(random_amplitude.value)
                    actions = amplitude * (
                        2.0
                        * torch.rand(
                            base.num_envs, cfg.num_actions, device=base.device
                        )
                        - 1.0
                    )
                else:
                    actions = torch.zeros(
                        base.num_envs, cfg.num_actions, device=base.device
                    )
                observations, last_reward, _terminated, _truncated, _extras = env.step(
                    actions
                )

            next_env_id = env_options.index(env_selector.value)
            if next_env_id != selected_env_id:
                if selected_env_id >= 0:
                    object_handles[selected_env_id].visible = False
                    goal_handles[selected_env_id].visible = False
                    pointcloud_handles[selected_env_id].visible = False
                    object_labels[selected_env_id].visible = False
                selected_env_id = next_env_id
                object_handles[selected_env_id].visible = True
                goal_handles[selected_env_id].visible = True
                object_labels[selected_env_id].visible = True

            env_id = selected_env_id
            pointcloud_handles[env_id].visible = bool(show_pointcloud.value)
            for handle in collision_handles:
                handle.visible = bool(show_collision.value)
            for handle in current_keypoints + goal_keypoints:
                handle.visible = bool(show_keypoints.value)

            with server.atomic():
                robot_position, robot_quaternion = _local_root_pose(
                    base.robot, base.scene.env_origins, env_id
                )
                robot_root.position = robot_position
                robot_root.wxyz = robot_quaternion
                robot_urdf.update_cfg(_to_numpy(base.robot_dof_pos[env_id]))

                table.position, table.wxyz = _local_root_pose(
                    base.table, base.scene.env_origins, env_id
                )
                object_position = _to_numpy(base.object_pos[env_id])
                object_quaternion = _to_numpy(base.object_rot[env_id])
                goal_position = _to_numpy(base.object_goal[env_id])
                goal_quaternion = _to_numpy(base.object_goal_quat[env_id])

                current_object = object_handles[env_id]
                current_object.position = object_position
                current_object.wxyz = object_quaternion
                goal_object = goal_handles[env_id]
                goal_object.position = goal_position
                goal_object.wxyz = goal_quaternion
                pointcloud_handles[env_id].position = object_position
                pointcloud_handles[env_id].wxyz = object_quaternion
                object_labels[env_id].position = object_position + np.asarray(
                    (0.0, 0.0, 0.12)
                )
                current_frame.position = object_position
                current_frame.wxyz = object_quaternion
                goal_frame.position = goal_position
                goal_frame.wxyz = goal_quaternion

                current_points = pose_keypoints(
                    base.object_pos[env_id : env_id + 1],
                    base.object_rot[env_id : env_id + 1],
                    cfg.keypoint_half_extent,
                )[0]
                goal_points = pose_keypoints(
                    base.object_goal[env_id : env_id + 1],
                    base.object_goal_quat[env_id : env_id + 1],
                    cfg.keypoint_half_extent,
                )[0]
                for index in range(8):
                    current_keypoints[index].position = _to_numpy(current_points[index])
                    goal_keypoints[index].position = _to_numpy(goal_points[index])

                contact_forces = torch.linalg.vector_norm(
                    base.fingertip_contact_forces[env_id], dim=-1
                )
                current_contact_gate = contact_gate(
                    base.fingertip_contact_forces[env_id : env_id + 1, 1:, :],
                    threshold=cfg.contact_force_threshold,
                )[0]
                for index, position in enumerate(base.hand_pos[env_id]):
                    local_position = _to_numpy(position)
                    fingertip_markers[index].position = local_position
                    contact_markers[index].position = local_position
                    contact_markers[index].visible = bool(
                        contact_forces[index] > cfg.contact_force_threshold
                    )

                body_positions, _jacobian = fabric.get_taskmap("body_points")(
                    base.fabric_q, None
                )
                body_positions = body_positions.reshape(base.num_envs, -1, 3)
                if body_positions.shape[1] != len(collision_handles):
                    raise RuntimeError(
                        "FABRICS collision point/radius count mismatch: "
                        f"{body_positions.shape[1]} != {len(collision_handles)}"
                    )
                local_collision = quaternion_apply_wxyz(
                    base.robot.data.root_quat_w[env_id], body_positions[env_id]
                ) + torch.as_tensor(
                    robot_position,
                    device=body_positions.device,
                    dtype=body_positions.dtype,
                )
                for index, position in enumerate(local_collision):
                    collision_handles[index].position = _to_numpy(position)

                pose_error = keypoint_pose_error(
                    base.object_pos[env_id : env_id + 1],
                    base.object_rot[env_id : env_id + 1],
                    base.object_goal[env_id : env_id + 1],
                    base.object_goal_quat[env_id : env_id + 1],
                    cfg.keypoint_half_extent,
                )[0]
                joint_margin = torch.minimum(
                    base.fabric_q[env_id] - base.robot_dof_lower_limits[0],
                    base.robot_dof_upper_limits[0] - base.fabric_q[env_id],
                ).min()
                force_values = ", ".join(
                    f"{name.split('_')[0]}={force:.2f}"
                    for name, force in zip(
                        cfg.hand_body_names, contact_forces.detach().cpu().tolist()
                    )
                )
                object_status.value = (
                    f"env={env_id}, {env_object_names[env_id]}, "
                    f"scale={float(base.object_scale[env_id, 0]):.3f}"
                )
                state_status.value = (
                    f"pose_error={float(pose_error):.4f} m, "
                    f"reward={float(last_reward[env_id]):.4f}, "
                    f"contact_gate={bool(current_contact_gate)}"
                )
                force_status.value = force_values
                fabric_status.value = (
                    f"ADR={base.dextrah_adr.num_increments()}/50, "
                    f"min_joint_margin={float(joint_margin):.4f} rad, "
                    f"max|qd|={float(base.fabric_qd[env_id].abs().max()):.3f} rad/s"
                )

            server.flush()
            remaining = frame_period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        server.stop()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    finally:
        simulation_app.close()
