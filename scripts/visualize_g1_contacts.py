#!/usr/bin/env python3
"""Actual PhysX contacts in Viser; diagnostic fixtures, NOT learned grasps.

Run --validate-only before serving. This never loads/changes the SAPG policy.
Physics is rate-limited to real time and publication to --viewer-fps.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import signal
import threading
import time
import traceback
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--viser-host", default="127.0.0.1")
parser.add_argument("--viser-port", type=int, default=8089)
parser.add_argument("--viewer-fps", type=float, default=12.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
stop_requested = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
signal.signal(signal.SIGINT, lambda *_: stop_requested.set())

import numpy as np
import torch
import trimesh
import viser
from viser.extras import ViserUrdf
from isaaclab.utils.math import quat_apply, quat_from_matrix

torch.set_num_threads(1)

from dextrah_lab.g1_adept.contact import FINGERS, TIP_BODIES
from dextrah_lab.tasks.g1_revo2_adept.contact_env import G1ContactDiagnosticEnv, diagnostic_cfg


def numpy(value):
    return value.detach().cpu().numpy()


def urdf_mesh(path):
    parts = []
    for visual in ET.parse(path).getroot().findall("link/visual"):
        geometry = visual.find("geometry")
        if geometry.find("box") is not None:
            mesh = trimesh.creation.box(np.fromstring(geometry.find("box").get("size"), sep=" "))
        elif geometry.find("cylinder") is not None:
            cylinder = geometry.find("cylinder")
            mesh = trimesh.creation.cylinder(radius=float(cylinder.get("radius")), height=float(cylinder.get("length")))
        else:
            raise ValueError(f"Unsupported procedural geometry: {path}")
        origin = visual.find("origin")
        if origin is not None:
            matrix = trimesh.transformations.euler_matrix(*np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
            matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            mesh.apply_transform(matrix)
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


class Diagnostic:
    def __init__(self):
        if args.num_envs < 2 or not 0 < args.viewer_fps <= 30:
            raise ValueError("need >=2 environments and viewer fps in (0,30]")
        self.env = G1ContactDiagnosticEnv(diagnostic_cfg(args.num_envs, args.device))
        self.env.reset()
        self.actions = torch.zeros(self.env.num_envs, 13, device=self.env.device)
        self.actions[:, 7:] = -.6
        self.contacts = None
        self.object_meshes = [urdf_mesh(path) for path in self.env._object_urdf_paths]
        self.object_ids = self.env._object_asset_index_per_env.cpu().tolist()
        self.handle_radii = []
        for path in self.env._object_urdf_paths:
            geometry = ET.parse(path).getroot().find("link/collision/geometry")
            box, cylinder = geometry.find("box"), geometry.find("cylinder")
            self.handle_radii.append(np.fromstring(box.get("size"), sep=" ")[2] / 2
                                     if box is not None else float(cylinder.get("radius")))
        self.sim_time = 0.
        self.park_probe()

    def place(self, asset, env_id, position, quaternion=None):
        ids = torch.tensor([env_id], device=self.env.device, dtype=torch.long)
        pose = asset.data.root_state_w[ids, :7].clone()
        pose[0, :3] = torch.as_tensor(position, device=self.env.device)
        if quaternion is not None:
            pose[0, 3:7] = torch.as_tensor(quaternion, device=self.env.device)
        asset.write_root_pose_to_sim(pose, ids)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device), ids)

    def park_probe(self):
        env = self.env
        pose = env.contact_probe.data.default_root_state[:, :7].clone()
        pose[:, :3] = env.scene.env_origins + torch.tensor([0., 0., -5.], device=env.device)
        env.contact_probe.write_root_pose_to_sim(pose)

    def park_assets(self):
        for asset in (self.env.object, self.env.table):
            pose = asset.data.root_state_w[:, :7].clone()
            pose[:, :3] = self.env.scene.env_origins + torch.tensor([0., 0., -3.], device=self.env.device)
            asset.write_root_pose_to_sim(pose)
            asset.write_root_velocity_to_sim(torch.zeros(self.env.num_envs, 6, device=self.env.device))

    def surface(self, env_id, finger, back=False):
        env = self.env
        pad = env.pad_geometry[finger]
        body = env.tip_body_ids[finger]
        position = env.robot.data.body_pos_w[env_id, body]
        quaternion = env.robot.data.body_quat_w[env_id, body]
        point, normal = pad.surface_point.copy(), pad.outward.copy()
        if back:
            # Opposite ACTUAL collision surface, outside the Touch CAD window.
            point = pad.back_surface_point
            normal = -normal
        vector = lambda x: torch.as_tensor(x, device=env.device, dtype=torch.float32)
        return position + quat_apply(quaternion, vector(point)), quat_apply(quaternion, vector(normal))

    def fixture(self, mode, env_id, finger, gap=-.001):
        env = self.env
        if mode in ("Pad probe", "Back-of-finger probe"):
            point, normal = self.surface(env_id, finger, mode.startswith("Back"))
            self.place(env.contact_probe, env_id, point + normal * (env.probe_radius + gap))
        elif mode == "Tool contact fixture":
            point, normal = self.surface(env_id, finger)
            # Touch the positive-z side of the handle (not an arbitrary COM).
            radius = self.handle_radii[self.object_ids[env_id]]
            rotation = trimesh.geometry.align_vectors([0, 0, 1], -numpy(normal))[:3, :3]
            rotation_t = torch.as_tensor(rotation, device=env.device, dtype=torch.float32)
            root_pos = point + normal * gap - rotation_t[:, 2] * radius
            self.place(env.object, env_id, root_pos, quat_from_matrix(rotation_t))
        elif mode == "Table contact fixture":
            body = env.tip_body_ids[finger]
            point = env.robot.data.body_pos_w[env_id, body]
            self.place(env.table, env_id, point + torch.tensor([0., 0., -.14], device=env.device), [1., 0., 0., 0.])

    def step(self, fixture=None):
        env = self.env
        env._pre_physics_step(self.actions)
        for _ in range(env.cfg.decimation):
            if fixture is not None:
                fixture()
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.cfg.sim.dt)
            self.contacts = env.read_contacts()
            self.sim_time += env.cfg.sim.dt
            if not torch.isfinite(env.robot.data.joint_pos).all():
                raise RuntimeError("Non-finite physical joint state")
        env.common_step_counter += 1
        return self.contacts

    def validate(self):
        env = self.env
        self.park_assets()
        for _ in range(30):
            self.step()
        assert self.contacts["net_w"].norm(dim=-1).max() < .02, "unexpected free-space contact"
        results = {}
        for finger, name in enumerate(FINGERS):
            maximum = 0.
            for _ in range(15):
                data = self.step(lambda: self.fixture("Pad probe", 0, finger))
                maximum = max(maximum, float(data["pad_load_n"][0, finger, 2]))
                assert data["pairs_w"][1:, :, 2].abs().max() < .01, "cross-environment contact leakage"
                assert data["pairs_w"][0, :, :2].abs().max() < .02, "probe attributed to object/table"
            results[f"{name}_pad_probe_peak_n"] = maximum
            print("CONTACT_TEST " + json.dumps(results), flush=True)
            assert maximum > .05, f"no pad response on {name}"
            self.park_probe()
            for _ in range(10):
                self.step()
            assert self.contacts["pad_load_n"][:, :, 2].max() < .01, "probe force persists after release"
        # Deliberately outside-pad contact must remain visible as link contact.
        raw, pad = 0., 0.
        for _ in range(15):
            data = self.step(lambda: self.fixture("Back-of-finger probe", 0, 1))
            raw = max(raw, float(data["pairs_w"][0, 1, 2].norm()))
            pad = max(pad, float(data["pad_load_n"][0, 1, 2]))
        assert raw > .05 and pad < .01, f"backside rejection failed: raw={raw}, pad={pad}"
        results.update(backside_raw_peak_n=raw, backside_pad_peak_n=pad)
        self.park_probe()
        for _ in range(10):
            self.step()
        raw = 0.
        for _ in range(20):
            data = self.step(lambda: self.fixture("Tool contact fixture", 0, 1))
            raw = max(raw, float(data["pad_load_n"][0, 1, 0]))
        assert raw > .05, f"no actual procedural-tool pad contact: {raw}"
        results["tool_pad_peak_n"] = raw
        self.park_assets()
        raw = 0.
        for _ in range(15):
            data = self.step(lambda: self.fixture("Table contact fixture", 0, 1))
            raw = max(raw, float(data["pairs_w"][0, :, 1].norm(dim=-1).max()))
            assert not data["gate_pad_object"].any(), "table incorrectly triggers object grasp gate"
        assert raw > .05, "no table contact response"
        results["table_link_peak_n"] = raw
        before = env.touch_filter.filtered_n.clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        assert not env.touch_filter.contact[0].any()
        assert not env.touch_filter.filtered_n[0].any()
        torch.testing.assert_close(env.touch_filter.filtered_n[1:], before[1:])
        obs = env._get_observations()
        assert obs["policy"].shape == (env.num_envs, 131)
        assert obs["critic"].shape == (env.num_envs, 153)
        assert all(torch.isfinite(x).all() for x in obs.values())
        print("CONTACT_VALIDATION_PASS " + json.dumps(results), flush=True)

    def viewer(self):
        env = self.env
        server = viser.ViserServer(host=args.viser_host, port=args.viser_port)
        if server.get_port() != args.viser_port:
            server.stop()
            raise RuntimeError(f"Requested Viser port {args.viser_port} is occupied")
        server.set_up_direction("+z")
        # Viser 0.1.34's theme/logo toggle triggers React hook error #300 even
        # in a minimal empty scene. Keep its default theme for compatibility.
        server.add_gui_markdown(
            "## G1 + Revo2 contact inspection\n"
            "**Live PhysX; scripted fixtures, not a trained policy.** "
            "Cyan/yellow: inactive/active pad. Orange: tool; blue: table; purple: probe; red: outside pad."
        )
        status = server.add_gui_text("State", "initializing", disabled=True)
        gates = server.add_gui_text("Grasp gates", "initializing", disabled=True)
        readouts = [server.add_gui_text(name, "", disabled=True) for name in FINGERS]
        sources = server.add_gui_text("Tool / table / probe [N]", "", disabled=True)
        server.add_gui_markdown("R/F: raw/filtered pad resultant, L: whole distal-link resultant. Source loads above are for the selected finger. Arrows show direction only.")
        modes = ("Pad probe", "Back-of-finger probe", "Tool contact fixture", "Table contact fixture", "Free physics")
        mode = server.add_gui_dropdown("Inspection mode", modes, initial_value=modes[0])
        finger = server.add_gui_dropdown("Finger", FINGERS, initial_value="index")
        options = tuple(f"{i}: {Path(env._object_urdf_paths[self.object_ids[i]]).stem}" for i in range(env.num_envs))
        select = server.add_gui_dropdown("Environment / tool", options, initial_value=options[0])
        running = server.add_gui_checkbox("Run physics", initial_value=True)
        show_outside = server.add_gui_checkbox("Show outside-pad contacts (red)", initial_value=True)
        gap = server.add_gui_slider("Fixture gap [mm]", min=-3., max=30., step=.25, initial_value=-1.)
        server.add_gui_markdown("Fixture modes reposition a test sphere/tool/table each step. Negative gap means a controlled contact test, not a safe robot command. Select Free physics for unforced motion.")
        hand = server.add_gui_dropdown("Hand target", ("Open", "Power grasp", "Thumb-index pinch", "Manual"), initial_value="Open")
        with server.add_gui_folder("Manual finger targets"):
            hand_sliders = [server.add_gui_slider(name, min=-1., max=1., step=.02, initial_value=-.6)
                            for name in ("Thumb spread", "Thumb flex", "Index flex", "Middle flex", "Ring flex", "Pinky flex")]
        with server.add_gui_folder("Arm target [rad]"):
            arm_sliders = [server.add_gui_slider(str(name), min=float(env._arm_lower[0, i]),
                           max=float(env._arm_upper[0, i]), step=.01,
                           initial_value=float(env._prev_targets[0, env._arm_joint_ids[i]]))
                           for i, name in enumerate(env.robot.joint_names[j] for j in env._arm_joint_ids.tolist())]
        reset = server.add_gui_button("Reset scene", color="orange")
        reset_event = threading.Event()
        reset.on_click(lambda _: reset_event.set())
        focus = server.add_gui_button("Focus hand")
        focus_event = threading.Event()
        focus.on_click(lambda _: focus_event.set())
        server.add_gui_markdown("ADEPT grasp gates: thumb + another finger >1 N. Hysteresis 0.15/0.08 N, 30 ms filter and CAD +2 mm pad windows are provisional, not hardware calibration. Fabric and 5% soft PCA stay active. No pressure image or shear is synthesized. Object identity and exact contact locations are simulator-only privileges.")

        camera_focus = np.asarray([0., .06, .95])

        @server.on_client_connect
        def camera(client):
            client.camera.position = camera_focus + np.asarray([.23, -.26, .18])
            client.camera.look_at = camera_focus
            client.camera.up_direction = (0., 0., 1.)

        root = server.add_frame("/robot", show_axes=False)
        robot = ViserUrdf(server, env.g1_urdf, root_node_name="/robot", mesh_color_override=(.62, .65, .70))
        # Only the PRESENTATION copy drops mimic constraints. The physics and
        # training URDF are untouched; contact-loaded distal angles are measured.
        for joint in robot._urdf.joint_map.values():
            joint.mimic = None
        visual_names = tuple(robot._urdf.joint_map)
        from isaacsimenvs.tasks.play.utils.scene_utils import G1_BODY_DEFAULT_JOINT_POS
        base_q = np.asarray([G1_BODY_DEFAULT_JOINT_POS.get(name, 0.) for name in visual_names])
        joint_map = {name: i for i, name in enumerate(env.robot.joint_names)}
        table = server.add_mesh_trimesh("/table", trimesh.creation.box((.4, .475, .3)))
        objects = [server.add_mesh_simple(f"/tools/{i}", np.asarray(mesh.vertices, dtype=np.float32),
                   np.asarray(mesh.faces, dtype=np.uint32), color=(223, 158, 55), visible=False)
                   for i, mesh in enumerate(self.object_meshes)]
        pads, active_pads, links = [], [], []
        for i, (name, pad) in enumerate(zip(FINGERS, env.pad_geometry)):
            pads.append(server.add_mesh_simple(f"/pads/{name}", np.asarray(pad.mesh.vertices, dtype=np.float32),
                         np.asarray(pad.mesh.faces, dtype=np.uint32), color=(40, 215, 205), opacity=.65))
            active_pads.append(server.add_mesh_simple(f"/active_pads/{name}", np.asarray(pad.mesh.vertices, dtype=np.float32),
                         np.asarray(pad.mesh.faces, dtype=np.uint32), color=(255, 195, 35), opacity=.8, visible=False))
            mesh = trimesh.load(env.g1_urdf.parent / f"meshes/{TIP_BODIES[i]}.STL", force="mesh")
            links.append(server.add_mesh_simple(f"/distal/{name}", np.asarray(mesh.vertices, dtype=np.float32),
                          np.asarray(mesh.faces, dtype=np.uint32), color=(180, 70, 70), opacity=.4, visible=False))
        probe = server.add_icosphere("/probe", radius=env.probe_radius, color=(190, 70, 245))
        point_markers = [[server.add_icosphere(f"/points/{name}/{c}", radius=.002, color=color, visible=False)
                          for c, color in enumerate(((245, 170, 30), (60, 150, 250), (200, 80, 250)))] for name in FINGERS]
        outside_markers = [[server.add_icosphere(f"/outside/{name}/{c}", radius=.002, color=(255, 75, 75), visible=False)
                            for c in range(3)] for name in FINGERS]
        # Persistent glyphs: legacy Viser/React can fail when a node is removed
        # and re-created during one render. Length is deliberately fixed; exact
        # force magnitudes are in the readouts, not encoded as displacement.
        arrow_mesh = trimesh.creation.cylinder(radius=.0007, height=.04, sections=8)
        arrow_mesh.apply_translation([0, 0, .02])
        head = trimesh.creation.cone(radius=.0025, height=.006, sections=8)
        head.apply_translation([0, 0, .04])
        arrow_mesh = trimesh.util.concatenate((arrow_mesh, head))
        arrow_handles = {}
        for i in range(5):
            for channel, color in enumerate(((245, 170, 30), (60, 150, 250), (200, 80, 250))):
                for outside in (False, True):
                    arrow_handles[i, channel, outside] = server.add_mesh_simple(
                        f"/forces/{i}/{channel}/{outside}", np.asarray(arrow_mesh.vertices, dtype=np.float32),
                        np.asarray(arrow_mesh.faces, dtype=np.uint32),
                        color=(255, 75, 75) if outside else color, visible=False)
        last_publish = 0.
        previous_mode = None
        try:
            while app.is_running() and not stop_requested.is_set():
                tick = time.monotonic()
                # No client means no need to compete with training for GPU time.
                if self.contacts is not None and not server.get_clients():
                    time.sleep(.1)
                    continue
                index, finger_id = options.index(select.value), FINGERS.index(finger.value)
                setting = (mode.value, index, finger_id)
                if reset_event.is_set() or setting != previous_mode:
                    env.reset()
                    self.park_probe()
                    if mode.value != "Free physics":
                        self.park_assets()
                    self.contacts = None
                    reset_event.clear()
                    previous_mode = setting
                self.actions.zero_()
                arm_target = torch.tensor([s.value for s in arm_sliders], device=env.device)
                nominal = env._prev_targets[:, env._arm_joint_ids]
                self.actions[:, :7] = ((arm_target - nominal) / (env.step_dt * 1.5 * .1)).clamp(-1., 1.)
                values = dict(Open=[-.6] * 6, **{"Power grasp": [.6] * 6,
                             "Thumb-index pinch": [.5, .6, .6, -.6, -.6, -.6]})
                self.actions[:, 7:] = torch.tensor(values.get(hand.value, [s.value for s in hand_sliders]), device=env.device)
                if running.value or self.contacts is None:
                    self.step(lambda: self.fixture(mode.value, index, finger_id, gap.value / 1000.))
                if tick - last_publish >= 1 / args.viewer_fps:
                    last_publish = tick
                    data = {key: numpy(value[index]) for key, value in self.contacts.items()}
                    origin = numpy(env.scene.env_origins[index])
                    q = base_q.copy()
                    measured = numpy(env.robot.data.joint_pos[index])
                    for j, name in enumerate(visual_names):
                        if name in joint_map:
                            q[j] = measured[joint_map[name]]
                    robot.update_cfg(dict(zip(visual_names, q)))
                    root.position = numpy(env.robot.data.root_pos_w[index]) - origin
                    root.wxyz = numpy(env.robot.data.root_quat_w[index])
                    base_transform = trimesh.transformations.quaternion_matrix(root.wxyz)
                    base_transform[:3, 3] = root.position
                    errors = []
                    for i, body in enumerate(TIP_BODIES):
                        visual_pose = base_transform @ robot._urdf.get_transform(body)
                        errors.append(np.linalg.norm(visual_pose[:3, 3] - (data["tip_pos_w"][i] - origin)))
                    if max(errors) > .002:
                        raise RuntimeError(f"Viser/PhysX distal-frame misalignment: {errors} meters")
                    camera_focus = data["tip_pos_w"].mean(axis=0) - origin
                    if focus_event.is_set():
                        for client in server.get_clients().values():
                            camera(client)
                        focus_event.clear()
                    table.position = numpy(env.table.data.root_pos_w[index]) - origin
                    table.wxyz = numpy(env.table.data.root_quat_w[index])
                    for i, obj in enumerate(objects):
                        obj.visible = i == self.object_ids[index]
                        if obj.visible:
                            obj.position = numpy(env.object.data.root_pos_w[index]) - origin
                            obj.wxyz = numpy(env.object.data.root_quat_w[index])
                    probe.position = numpy(env.contact_probe.data.root_pos_w[index]) - origin
                    probe.visible = "probe" in mode.value.lower()
                    state = env.touch_filter
                    for i, name in enumerate(FINGERS):
                        for handle in (pads[i], active_pads[i], links[i]):
                            handle.position = data["tip_pos_w"][i] - origin
                            handle.wxyz = data["tip_quat_w"][i]
                        # Viser 0.1 MeshHandle has no dynamic color property.
                        active_pads[i].visible = bool(state.contact[index, i])
                        pads[i].visible = not active_pads[i].visible
                        links[i].visible = bool(np.linalg.norm(data["net_w"][i]) > .1 and not bool(state.contact[index, i]))
                        loads = data["pad_load_n"][i]
                        readouts[i].value = (f"R/F {float(state.raw_n[index,i]):.2f}/{float(state.filtered_n[index,i]):.2f} "
                                            f"L {np.linalg.norm(data['net_w'][i]):.2f} N; "
                                            f"{float(state.duration_s[index,i]):.1f}s")
                        for channel, color in enumerate(((245, 170, 30), (60, 150, 250), (200, 80, 250))):
                            point = data["pad_points_w"][i, channel] - origin
                            force = data["pad_pairs_w"][i, channel]
                            outside = (show_outside.value and loads[channel] < .01
                                       and np.linalg.norm(data["pairs_w"][i, channel]) > .01
                                       and data["point_valid"][i, channel])
                            if outside:
                                point = data["points_w"][i, channel] - origin
                                force = data["pairs_w"][i, channel]
                                color = (255, 75, 75)
                            outside_markers[i][channel].visible = bool(outside)
                            if outside:
                                outside_markers[i][channel].position = point
                            strength = float(np.linalg.norm(force))
                            marker = point_markers[i][channel]
                            marker.visible = bool(loads[channel] > .01)
                            if marker.visible:
                                marker.position = point
                            for is_outside in (False, True):
                                arrow = arrow_handles[i, channel, is_outside]
                                arrow.visible = bool(strength > .01 and bool(outside) == is_outside)
                                if arrow.visible:
                                    arrow.position = point
                                    arrow.wxyz = trimesh.transformations.quaternion_from_matrix(
                                        trimesh.geometry.align_vectors([0, 0, 1], force / strength))
                    gates.value = (f"all={int(data['gate_all'])} object={int(data['gate_object'])} "
                                   f"pad/object={int(data['gate_pad_object'])}")
                    sources.value = " / ".join(f"{x:.2f}" for x in data["pad_load_n"][finger_id])
                    status.value = f"sim {self.sim_time:.1f}s | {env.num_envs} envs | {mode.value}"
                time.sleep(max(0., env.step_dt - (time.monotonic() - tick)))
        finally:
            server.stop()


try:
    with torch.inference_mode():
        diagnostic = Diagnostic()
        try:
            if args.validate_only:
                diagnostic.validate()
            else:
                diagnostic.viewer()
        finally:
            diagnostic.env.close()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
app.close()
