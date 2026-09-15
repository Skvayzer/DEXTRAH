#!/usr/bin/env python3
"""Render measured empty-floor robot poses beside kinematic planner markers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--cases', nargs='+', default=['forward_020', 'right_012_yaw90', 'right_040_yaw90'])
    p.add_argument('--plots-only', action='store_true')
    args = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta = json.loads((args.recording/'metadata.json').read_text())
    if not meta['completed']:
        raise ValueError('Require completed diagnostic')
    robot_names = list(meta['robots'])
    all_cases = [c['name'] for c in meta['cases']]
    if not set(args.cases) <= set(all_cases):
        raise ValueError('Selected case was not captured')
    traces = {}
    for kind in robot_names:
        for name in all_cases:
            with np.load(args.recording/kind/name/'trajectory.npz', allow_pickle=False) as archive:
                traces[kind, name] = dict(archive)
    fig, axes = plt.subplots(len(all_cases), 3, figsize=(13, 2.2*len(all_cases)), squeeze=False)
    for row, name in enumerate(all_cases):
        for kind in robot_names:
            data = traces[kind, name]
            for axis in range(2):
                axes[row, axis].plot(data['time_s'], data['velocity_body'][:, axis], label=kind, lw=1.)
            axes[row, 2].plot(data['root'][:, 0], data['root'][:, 1], label=kind)
            axes[row, 2].plot(data['reference_root'][:, 0, 0], data['reference_root'][:, 0, 1], '--', alpha=.6,
                              label=f'{kind} reference')
        for axis in range(2):
            axes[row, axis].plot(data['time_s'], data['command'][:, axis], 'k--', label='command')
            axes[row, axis].set_ylabel(f'{name}\n{"vx" if axis == 0 else "vy"} (m/s)')
            axes[row, axis].set_xlabel('Simulation time (s)')
        axes[row, 2].set(xlabel='World X (m)', ylabel='World Y (m)')
        axes[row, 2].axis('equal')
        for axis in axes[row]: axis.grid(alpha=.25)
    for axis in axes[0]: axis.legend(fontsize=7)
    fig.suptitle('Frozen SONIC empty-floor diagnostics | command, measured velocity and planned root path\n'
                 'No SAPG residual, objects, tables or within-trial resets', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.recording/'tracking_comparison.png', dpi=150)
    fig.savefig(args.recording/'tracking_comparison.pdf')
    plt.close(fig)
    if args.plots_only:
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('EGL rendering requires an allocated GPU')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import pyrender
    import trimesh
    import yourdfpy
    from PIL import Image, ImageDraw, ImageFont
    from pyrender.platforms import egl
    devices = [i for i, device in enumerate(egl.query_devices()) if b'EGL_NV_device_cuda' in
               (egl._eglQueryDeviceStringEXT(device._display, 0x3055) or b'')]
    if len(devices) != 1:
        raise RuntimeError('Require exactly one allocated NVIDIA EGL device')
    os.environ['EGL_DEVICE_ID'] = str(devices[0])

    def pose(position, quaternion):
        result = trimesh.transformations.quaternion_matrix(quaternion)
        result[:3, 3] = position
        return result

    def look_at(eye, target):
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        z = eye-target; z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
        out = np.eye(4); out[:3, :3] = np.column_stack((x, np.cross(z, x), z)); out[:3, 3] = eye
        return out

    def add(scene, mesh, color):
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=0., roughnessFactor=.8)
        return scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False))

    width, height, panel_width = 1600, 900, 1600//len(robot_names)
    renderer = pyrender.OffscreenRenderer(panel_width, 660)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    views = []
    for kind in robot_names:
        data = meta['robots'][kind]
        urdf = yourdfpy.URDF.load(data['robot_urdf'])
        bodies = {n: i for i, n in enumerate(data['body_names'])}
        scene = pyrender.Scene(bg_color=[.95, .97, .99, 1.], ambient_light=[.5, .5, .5])
        nodes, omitted = [], []
        for node in urdf.scene.graph.nodes_geometry:
            transform, geometry = urdf.scene.graph.get(node)
            link = urdf.scene.graph.transforms.parents[node]
            while link not in urdf.link_map: link = urdf.scene.graph.transforms.parents[link]
            # Fixed visual links may have been merged by PhysX. Recover their
            # static transform to the closest measured ancestor, not an FK body pose.
            relative = np.linalg.inv(urdf.get_transform(link))@transform
            ancestor = link
            while ancestor not in bodies:
                joints = [j for j in urdf.joint_map.values() if j.child == ancestor]
                if not joints or joints[0].type != 'fixed': break
                parent = joints[0].parent
                relative = np.linalg.inv(urdf.get_transform(parent))@urdf.get_transform(ancestor)@relative
                ancestor = parent
            if ancestor not in bodies:
                omitted.append(link); continue
            mesh = urdf.scene.geometry[geometry].copy(); mesh.apply_transform(relative)
            nodes.append((bodies[ancestor], add(scene, mesh, [.34, .43, .52, 1.])))
        ground = trimesh.creation.box([20, 20, .01]); ground.apply_translation([0, 0, -.008])
        add(scene, ground, [.83, .87, .90, 1.])
        for x in range(-8, 9):
            for size, center in (([.008, 16., .002], [x, 0, 0]), ([16., .008, .002], [0, x, 0])):
                grid = trimesh.creation.box(size); grid.apply_translation(center)
                add(scene, grid, [.68, .74, .80, 1.])
        marks = []
        for link in ('pelvis', 'left_knee_link', 'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link'):
            if link in urdf.link_map:
                marks.append((link, add(scene, trimesh.creation.icosphere(subdivisions=1, radius=.035), [1., .61, .08, 1.])))
        camera = scene.add(pyrender.PerspectiveCamera(yfov=.77), pose=look_at([3., -4., 2.2], [0, 0, .7]))
        light = scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5), pose=look_at([3., -4., 4.], [0, 0, .7]))
        views.append((kind, scene, urdf, nodes, marks, camera, light, omitted))
    validation = []
    try:
        for name in args.cases:
            path = args.recording/f'sonic-empty-floor-{name}.mp4'
            frames = len(traces[robot_names[0], name]['time_s'])
            encoder = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo',
                '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '25', '-i', '-',
                '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '2',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
            try:
                for i in range(frames):
                    canvas = Image.new('RGB', (width, height), (242, 246, 250))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((20, 12), f'Frozen SONIC | Empty-floor walking test | {name} | t={i/25:.2f}s', font=font, fill=(20, 35, 50))
                    draw.text((20, 49), 'Solid robot: measured physics | Orange markers: kinematic reference (pelvis, knees, feet)', font=small, fill=(40, 55, 70))
                    for k, (kind, scene, urdf, nodes, marks, camera, light, omitted) in enumerate(views):
                        trace = traces[kind, name]
                        for body, node in nodes:
                            scene.set_pose(node, pose(trace['body_pos'][i, body], trace['body_quat'][i, body]))
                        urdf.update_cfg(dict(zip(meta['body_joint_names'], trace['reference_q'][i, 0])))
                        root_ref = trace['reference_root'][i, 0]
                        base = pose(root_ref[:3], root_ref[3:])
                        for link, node in marks: scene.set_pose(node, base@urdf.get_transform(link))
                        focus = trace['root'][i, :3].copy(); focus[2] = .7
                        cp = look_at(focus+[2.7, -3.4, 1.5], focus)
                        scene.set_pose(camera, cp); scene.set_pose(light, cp)
                        rgb, _ = renderer.render(scene)
                        canvas.paste(Image.fromarray(rgb), (k*panel_width, 85))
                        label = 'Original NVIDIA G1 asset' if kind == 'upstream' else 'G1 + Revo2 source asset'
                        x = k*panel_width+16
                        draw.text((x, 95), label, font=font, fill=(20, 35, 50))
                        cmd, actual = trace['command'][i], trace['velocity_body'][i]
                        draw.text((x, 757), f'Command vx/vy: {cmd[0]:+.2f} / {cmd[1]:+.2f} m/s', font=small, fill=(20, 35, 50))
                        draw.text((x, 787), f'Actual vx/vy: {actual[0]:+.2f} / {actual[1]:+.2f} m/s', font=small, fill=(20, 35, 50))
                        draw.text((x, 817), f'Pelvis height: {trace["root"][i, 2]:.2f} m | Fall seen: {bool(trace["fell"][:i+1].any())}', font=small, fill=(20, 35, 50))
                    draw.text((20, 867), '50 Hz control / 200 Hz physics | No SAPG residual, tables, objects, optimizer updates or within-trial resets', font=small, fill=(40, 55, 70))
                    if i in (0, 150, frames-1): canvas.save(args.recording/f'{name}-{i:04d}.png')
                    encoder.stdin.write(np.asarray(canvas).tobytes())
            finally:
                encoder.stdin.close()
                code = encoder.wait(timeout=60)
            if code: raise RuntimeError(f'ffmpeg failed: {code}')
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], text=True))
            video = next(s for s in info['streams'] if s['codec_type'] == 'video')
            if int(video['nb_frames']) != frames: raise RuntimeError('Encoded frame-count mismatch')
            with path.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            validation.append(dict(video=str(path), frames=frames, duration_s=float(info['format']['duration']), sha256=checksum))
            print('EMPTY_FLOOR_VIDEO '+json.dumps(validation[-1]), flush=True)
    finally:
        renderer.delete()
    (args.recording/'video_validation.json').write_text(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
