#!/usr/bin/env python3
"""Render recorded PhysX poses to MP4 on the one GPU exposed by Slurm.

No physics, policy, or browser runs here. EGL selects the sole accessible
NVIDIA device; never default to a workstation-wide graphics device.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--video-name", default="play2perfect-sapg.mp4")
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--fabrics", action="store_true", help="Show recorded training collision spheres")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run rendering inside the user's Slurm GPU allocation")
    video = args.recording / args.video_name
    if video.exists():
        raise FileExistsError(video)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import numpy as np
    import pyrender
    import trimesh
    import yourdfpy
    from PIL import Image, ImageDraw, ImageFont
    from pyrender.platforms import egl

    candidates = [i for i, device in enumerate(egl.query_devices())
                  if b"EGL_NV_device_cuda" in
                  (egl._eglQueryDeviceStringEXT(device._display, 0x3055) or b"")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one accessible NVIDIA EGL device; found {candidates}")
    os.environ["EGL_DEVICE_ID"] = str(candidates[0])
    print("RENDER_EGL_DEVICE", candidates[0], "CUDA_VISIBLE_DEVICES",
          os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    meta = json.loads((args.recording / "metadata.json").read_text())
    if args.fabrics and not meta.get("fabrics_enabled"):
        raise ValueError("Sphere visualization requires a fabric-enabled policy recording")
    # NpzFile is lazy: indexing a member decompresses its entire trajectory.
    # Cache once, rather than decompressing thousands of frames for every
    # link on every frame of a long recording.
    with np.load(args.recording / "trajectory.npz", allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    frames = min(meta["frames"], args.limit_frames or meta["frames"])
    preview_frames = {0, 60, 150, 300, 450}
    if "lift_height" in trajectory:
        peak = int(np.argmax(trajectory["lift_height"][:frames]))
        preview_frames.update(peak + round(offset * meta["fps"]) for offset in (-2., -1., 0., 1.))
    for when in (meta.get("first_goal_seconds"), *(meta.get("longest_lift_interval_s") or [])):
        if when is not None:
            preview_frames.update(round((when + offset) * meta["fps"]) for offset in (-1., -.5, 0., .5))
    goal_counts = trajectory["goal_hits"][:frames]
    goal_event_frames = np.flatnonzero(np.diff(goal_counts, prepend=0) > 0)
    urdf = yourdfpy.URDF.load(meta["robot_urdf"])
    for joint in urdf.joint_map.values():
        joint.mimic = None
    if args.fabrics:
        urdf.update_cfg(meta["visual_static_joint_pos"])
    urdf.update_cfg(dict(zip(meta["joint_names"], trajectory["joint_pos"][0])))
    scene = pyrender.Scene(bg_color=[.94, .96, .98, 1.],
                           ambient_light=[.18, .18, .18] if args.fabrics else [.4, .4, .4])
    validation = {"frames": 0, "max_fk_error_m": 0., "max_fk_angle_rad": 0.}

    def pose(position, quaternion):
        value = trimesh.transformations.quaternion_matrix(quaternion)
        value[:3, 3] = position
        return value

    def add_mesh(mesh, color):
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,
            metallicFactor=.05, roughnessFactor=.65, doubleSided=True,
            alphaMode="BLEND" if color[3] < 1 else "OPAQUE")
        return scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))

    handles = []
    for node in urdf.scene.graph.nodes_geometry:
        transform, geometry_name = urdf.scene.graph.get(node)
        link = urdf.scene.graph.transforms.parents[node]
        while link not in urdf.link_map:
            link = urdf.scene.graph.transforms.parents[link]
        local = np.linalg.inv(urdf.get_transform(link)) @ transform
        mesh = urdf.scene.geometry[geometry_name].copy()
        mesh.apply_transform(local)
        color = [.70, .73, .77, 1.] if "right_" in link else [.33, .38, .44, 1.]
        if args.fabrics:
            color[3] = .45
        handles.append((link, add_mesh(mesh, color)))
    assets = {}
    for name in ("object", "table", "goal"):
        source = "object" if name == "goal" else name
        mesh = trimesh.load(args.recording / f"{source}.glb", force="mesh")
        assets[name] = add_mesh(mesh, {
            "object": [1., .48, .08, 1.], "table": [.46, .55, .61, 1.],
            "goal": [.10, .90, .35, .25],
        }[name])
    sphere_nodes = []
    if args.fabrics:
        from fabric_recording_geometry import verify_sphere_centers, sphere_clearances
        geometry = meta["fabric_geometry"]
        validation.update(max_sphere_center_error_m=0., max_clearance_error_m=0.,
                          min_proxy_clearance_m=float("inf"),
                          dynamic_spheres=len(geometry["dynamic"]), fixed_spheres=len(geometry["fixed"]))
        for group in ("dynamic", "fixed"):
            for spec in geometry[group]:
                radius = spec["radius"]
                base_color = [.02, .72, .70] if group == "dynamic" else [.15, .30, .95]
                fill_node = add_mesh(trimesh.creation.icosphere(subdivisions=2, radius=radius), [*base_color, .18])
                rings = []
                for axis in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]):
                    ring = trimesh.creation.annulus(r_min=radius-.0006, r_max=radius+.0006,
                                                    height=.0012, sections=48)
                    ring.apply_transform(trimesh.geometry.align_vectors([0., 0., 1.], axis))
                    rings.append(ring)
                outline = add_mesh(trimesh.util.concatenate(rings), [*base_color, 1.])
                sphere_nodes.append((group, fill_node, outline, base_color))
    ground = trimesh.creation.box([3., 3., .01])
    ground.apply_translation([0., 0., -.006])
    add_mesh(ground, [.83, .86, .89, 1.])
    eye, target = np.array([.72, -.75, 1.26]), np.array([.0, .16, .95])
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    camera_pose[:3, 3] = eye
    scene.add(pyrender.PerspectiveCamera(yfov=.80, znear=.02, zfar=10.), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.4 if args.fabrics else 2.5), pose=camera_pose)
    fill = camera_pose.copy()
    fill[:3, 3] = [-1., -.5, 2.]
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=2. if args.fabrics else 6.), pose=fill)
    body_ids = {name: i for i, name in enumerate(meta["body_names"])}
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 25)
    text_font = ImageFont.truetype(font_path, 19)
    small_font = ImageFont.truetype(font_path, 15)
    renderer = pyrender.OffscreenRenderer(1280, 720)
    encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720",
        "-framerate", str(meta["fps"]), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "19", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    try:
        for index in range(frames):
            urdf.update_cfg(dict(zip(meta["joint_names"], trajectory["joint_pos"][index])))
            base = trajectory["robot"][index]
            root = pose(base[:3], base[3:])
            for link, node in handles:
                fk = root @ urdf.get_transform(link)
                actual = fk
                if link in body_ids:
                    body = body_ids[link]
                    actual = pose(trajectory["body_pos"][index, body], trajectory["body_quat"][index, body])
                    error = float(np.linalg.norm(fk[:3, 3] - actual[:3, 3]))
                    angle = float(np.arccos(np.clip(
                        (np.trace(fk[:3, :3].T @ actual[:3, :3]) - 1.) / 2., -1., 1.)))
                    validation["max_fk_error_m"] = max(validation["max_fk_error_m"], error)
                    validation["max_fk_angle_rad"] = max(validation["max_fk_angle_rad"], angle)
                    if error > .002 or angle > .01:
                        raise RuntimeError(f"URDF/PhysX mismatch {link}: {error} m, {angle} rad")
                scene.set_pose(node, actual)
            for name, node in assets.items():
                value = trajectory[name][index]
                scene.set_pose(node, pose(value[:3], value[3:]))
            if args.fabrics:
                moving, fixed = trajectory["fabric_dynamic"][index], trajectory["fabric_fixed"][index]
                center_error = verify_sphere_centers(geometry, meta["body_names"],
                    trajectory["body_pos"][index], trajectory["body_quat"][index], base, moving, fixed)
                gaps, moving_min, fixed_min = sphere_clearances(
                    geometry, moving, fixed, trajectory["fabric_table_height"][index])
                recorded = trajectory["fabric_clearance"][index]
                np.testing.assert_allclose(gaps, recorded, atol=2.e-6, rtol=1.e-5)
                validation["max_sphere_center_error_m"] = max(validation["max_sphere_center_error_m"], center_error)
                validation["max_clearance_error_m"] = max(validation["max_clearance_error_m"], float(np.abs(gaps-recorded).max()))
                validation["min_proxy_clearance_m"] = min(validation["min_proxy_clearance_m"], float(gaps.min()))
                counters = {"dynamic": 0, "fixed": 0}
                for group, fill_node, outline, base_color in sphere_nodes:
                    s = counters[group]
                    center = moving[s] if group == "dynamic" else fixed[s]
                    minimum = moving_min[s] if group == "dynamic" else fixed_min[s]
                    # Fixed obstacles retain their blue identity. Moving proxies
                    # highlight geometric proximity, not measured contact forces.
                    color = base_color
                    if group == "dynamic":
                        color = ([.95, .10, .13] if minimum < 0 else
                                 [1., .55, .02] if minimum < geometry["influence_distance"] else base_color)
                    transform = pose(center, [1., 0., 0., 0.])
                    for node, alpha in ((fill_node, .18), (outline, 1.)):
                        scene.set_pose(node, transform)
                        node.mesh.primitives[0].material.baseColorFactor = [*color, alpha]
                    counters[group] += 1
            rgb, _ = renderer.render(scene)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, 1280, 75], fill=(242, 245, 248))
            title = "Fabrics applied to G1 RL training" if args.fabrics else "Play2Perfect SAPG | G1 + BrainCo Revo2"
            subtitle = ("14 moving spheres + 7 fixed body spheres | recorded fabric-enabled policy rollout"
                        if args.fabrics else f"Checkpoint epoch {meta['checkpoint_epoch']:,} | deterministic leader | seed {meta['seed']} | uncut rollout")
            if args.fabrics and meta.get("selection", "").startswith("Selected"):
                subtitle = "14 moving spheres + 7 fixed body spheres | selected continuous rollout; resets retained"
            draw.text((22, 10), title, font=title_font, fill=(25, 38, 52))
            draw.text((22, 43), subtitle,
                      font=text_font, fill=(60, 75, 90))
            draw.rectangle([0, 655, 1280, 720], fill=(242, 245, 248))
            seconds = trajectory["step"][index] * meta["policy_dt"]
            error_text = "--" if index == 0 else f"{trajectory['goal_error'][index]*100:.1f} cm"
            line = (f"{seconds:05.2f} / {meta['seconds']:.0f} s   |   Goals: {int(trajectory['goal_hits'][index])}"
                    f"   |   Lifted: {'yes' if trajectory['lifted'][index] else 'no'}"
                    f"   |   Goal error: {error_text}   |   Resets: {int(trajectory['resets'][index])}")
            if args.fabrics:
                line = (f"{seconds:05.2f} / {meta['seconds']:.0f} s  |  Goals: {int(goal_counts[index])}"
                        f"  |  Min proxy gap: {gaps.min()*100:.1f} cm"
                        f"  |  Resets: {int(trajectory['resets'][index])}  |  Body fixed")
                if "lift_height" in trajectory:
                    rise = (trajectory["lift_height"][index] - .05) * 100
                    line += f"  |  Object rise: {rise:.1f} cm"
                recent_goal = ((index - goal_event_frames >= 0)
                               & (index - goal_event_frames < 2 * meta["fps"])).any()
                if recent_goal:
                    draw.rectangle([20, 85, 400, 124], fill=(220, 246, 227))
                    draw.text((30, 93), "Manipulation goal reached", font=text_font, fill=(20, 105, 45))
            draw.text((22, 663), line, font=text_font, fill=(25, 38, 52))
            footer = ("Blue: fixed body  |  Teal: moving arm/hand  |  Amber: within avoidance range  |  Red: proxy overlap (not a contact sensor)"
                      if args.fabrics else "Measured simulation; whole body is static context. Original training disturbances enabled. Orange: tool. Green: target.")
            draw.text((22, 694), footer,
                      font=small_font, fill=(60, 75, 90))
            encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if index in preview_frames:
                image.save(args.recording / f"preview-{index:04d}.png")
            validation["frames"] += 1
            if index % 60 == 0:
                print(f"VIDEO_FRAME {index}/{frames}", flush=True)
    finally:
        encoder.stdin.close()
        code = encoder.wait(timeout=60)
        renderer.delete()
    if code:
        raise RuntimeError(f"Video encoding failed: {code}")
    (args.recording / "render_validation.json").write_text(json.dumps(validation, indent=2))
    print("VIDEO_RENDER_PASS " + str(video) + " " + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
