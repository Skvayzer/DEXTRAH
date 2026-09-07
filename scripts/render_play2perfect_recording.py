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
    trajectory = np.load(args.recording / "trajectory.npz", allow_pickle=False)
    frames = min(meta["frames"], args.limit_frames or meta["frames"])
    urdf = yourdfpy.URDF.load(meta["robot_urdf"])
    for joint in urdf.joint_map.values():
        joint.mimic = None
    urdf.update_cfg(dict(zip(meta["joint_names"], trajectory["joint_pos"][0])))
    scene = pyrender.Scene(bg_color=[.94, .96, .98, 1.], ambient_light=[.4, .4, .4])
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
        handles.append((link, add_mesh(mesh, color)))
    assets = {}
    for name in ("object", "table", "goal"):
        source = "object" if name == "goal" else name
        mesh = trimesh.load(args.recording / f"{source}.glb", force="mesh")
        assets[name] = add_mesh(mesh, {
            "object": [1., .48, .08, 1.], "table": [.46, .55, .61, 1.],
            "goal": [.10, .90, .35, .25],
        }[name])
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
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=camera_pose)
    fill = camera_pose.copy()
    fill[:3, 3] = [-1., -.5, 2.]
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=6.), pose=fill)
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
            rgb, _ = renderer.render(scene)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, 1280, 75], fill=(242, 245, 248))
            draw.text((22, 10), "Play2Perfect SAPG | G1 + BrainCo Revo2", font=title_font, fill=(25, 38, 52))
            draw.text((22, 43), f"Checkpoint epoch {meta['checkpoint_epoch']:,} | deterministic leader | seed {meta['seed']} | uncut rollout",
                      font=text_font, fill=(60, 75, 90))
            draw.rectangle([0, 655, 1280, 720], fill=(242, 245, 248))
            seconds = trajectory["step"][index] * meta["policy_dt"]
            error_text = "--" if index == 0 else f"{trajectory['goal_error'][index]*100:.1f} cm"
            line = (f"{seconds:05.2f} / {meta['seconds']:.0f} s   |   Goals: {int(trajectory['goal_hits'][index])}"
                    f"   |   Lifted: {'yes' if trajectory['lifted'][index] else 'no'}"
                    f"   |   Goal error: {error_text}   |   Resets: {int(trajectory['resets'][index])}")
            draw.text((22, 663), line, font=text_font, fill=(25, 38, 52))
            draw.text((22, 694), "Measured simulation; whole body is static context. Original training disturbances enabled. Orange: tool. Green: target.",
                      font=small_font, fill=(60, 75, 90))
            encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if index in (0, 60, 150, 300, 450):
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
