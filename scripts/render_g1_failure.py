#!/usr/bin/env python3
"""Two-camera diagnostic videos from measured physics states, including stop."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--limit-frames', type=int, default=0)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation for EGL rendering')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import numpy as np
    import pyrender
    import trimesh
    import yourdfpy
    from PIL import Image, ImageDraw, ImageFont
    from pyrender.platforms import egl
    devices = [i for i, device in enumerate(egl.query_devices()) if b'EGL_NV_device_cuda' in
               (egl._eglQueryDeviceStringEXT(device._display, 0x3055) or b'')]
    if len(devices) != 1:
        raise RuntimeError(f'Expected one allocated NVIDIA EGL device: {devices}')
    os.environ['EGL_DEVICE_ID'] = str(devices[0])
    meta = json.loads((args.recording/'metadata.json').read_text())
    with np.load(args.recording/'trajectory.npz', allow_pickle=False) as data:
        trace = dict(data)
    count = min(meta['frames'], args.limit_frames or meta['frames'])
    if count < 2:
        raise ValueError('Need at least two measured physics samples')
    t = trace['time_s'][:count]
    if not np.all(np.diff(t) > 0):
        raise ValueError('Physics capture timestamps must increase')
    joint = meta['failed_joint']
    j = meta['failed_joint_index']
    names = {name: i for i, name in enumerate(meta['body_names'])}
    urdf = yourdfpy.URDF.load(meta['robot_urdf'])
    child = urdf.joint_map[joint].child
    if child not in names:
        raise ValueError(f'Failing joint child lacks measured body poses: {child}')
    child_id = names[child]
    monitored_position = trace['body_pos'][:count, child_id]
    speed = np.abs(trace['joint_vel'][:count, j])
    angle = trace['joint_pos'][:count, j]
    command = trace['commanded_joint_targets'][:count, j]
    finite = np.isfinite(trace['body_pos'][:count]).all(axis=(1, 2)) & np.isfinite(trace['body_quat'][:count]).all(axis=(1, 2))
    if not finite.all():
        raise RuntimeError('Nonfinite body poses; explicit last-valid-frame handling is required')

    def pose(position, quaternion):
        result = trimesh.transformations.quaternion_matrix(quaternion)
        result[:3, 3] = position
        return result

    def look_at(eye, target):
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        z = eye-target; z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
        result = np.eye(4)
        result[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        result[:3, 3] = eye
        return result

    def add(scene, mesh, color):
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=.04,
            roughnessFactor=.68, doubleSided=True, alphaMode='BLEND' if color[3] < 1 else 'OPAQUE')
        return scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))

    geometry = []
    omitted = []
    for node in urdf.scene.graph.nodes_geometry:
        transform, geometry_name = urdf.scene.graph.get(node)
        link = urdf.scene.graph.transforms.parents[node]
        while link not in urdf.link_map:
            link = urdf.scene.graph.transforms.parents[link]
        if link not in names:
            omitted.append(link)
            continue  # no synthetic fixed-body/FK animation
        mesh = urdf.scene.geometry[geometry_name].copy()
        mesh.apply_transform(np.linalg.inv(urdf.get_transform(link))@transform)
        color = [.90, .13, .13, 1.] if link == child else ([.70, .74, .79, 1.] if 'right_' in link else [.30, .36, .43, 1.])
        geometry.append((link, mesh, color))

    def make_scene(close):
        scene = pyrender.Scene(bg_color=[.94, .96, .98, 1.], ambient_light=[.43, .43, .43])
        robots = [(names[name], add(scene, mesh, color)) for name, mesh, color in geometry]
        assets = {}
        for name in ('object', 'table', 'goal'):
            path = args.recording/('object.glb' if name == 'goal' else f'{name}.glb')
            mesh = trimesh.load(path, force='mesh')
            color = dict(object=[1., .47, .06, 1.], table=[.50, .58, .64, .22 if close else 1.],
                         goal=[.10, .82, .30, .18])[name]
            assets[name] = add(scene, mesh, color)
        ground = trimesh.creation.box([4., 4., .01]); ground.apply_translation([0., 0., -.006])
        add(scene, ground, [.82, .86, .89, 1.])
        camera = pyrender.PerspectiveCamera(yfov=.70 if not close else .62, znear=.005, zfar=30.)
        transform = look_at([1.7, -2.2, 1.6], [0., .25, .80])
        camera_node = scene.add(camera, pose=transform)
        light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=transform)
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=5.), pose=look_at([-1., -.8, 2.], [0., 0., 1.]))
        return scene, robots, assets, camera, camera_node, light_node

    views = [make_scene(False), make_scene(True)]
    width, height, fps = 1920, 1080, 30
    viewport_widths, viewport_height = [1060, 860], 704
    renderer = pyrender.OffscreenRenderer(viewport_widths[0], viewport_height)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title, text, small, tiny = [ImageFont.truetype(font_path, size) for size in (30, 23, 19, 16)]
    encoders = []
    outputs = ['g1-sonic-failure-normal-speed.mp4', 'g1-sonic-failure-slow-8x.mp4']
    for name in outputs:
        encoders.append(subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.recording/name)], stdin=subprocess.PIPE))
    # Replay nearest PAST sample. No new intermediate body poses are generated.
    schedules = []
    for playback in (1., .125):
        times = np.arange(int(np.ceil((t[-1]-t[0])*fps/playback))+1)*playback/fps + t[0]
        indices = np.searchsorted(t, times+1e-10, side='right')-1
        indices = np.clip(indices, 0, count-1)
        indices = np.r_[np.zeros(fps//2, int), indices, np.full(3*fps, count-1, int)]
        schedules.append(np.bincount(indices, minlength=count))
    encoded = [0, 0]
    plotted_angle = np.r_[angle, command]
    ymin, ymax = float(plotted_angle.min()), float(plotted_angle.max())
    pad = max(.05, (ymax-ymin)*.1); ymin -= pad; ymax += pad
    maxlog = max(3.2, float(np.log10(1+np.nanmax(speed)))+.1)
    plots = [(32, 870, 918, 1020), (1010, 870, 1880, 1020)]

    def chart(draw, box, series, colors, index, low, high):
        x0, y0, x1, y1 = box
        draw.rectangle(box, fill=(255, 255, 255), outline=(199, 208, 217))
        xx = x0+(t-t[0])/max(t[-1]-t[0], 1e-9)*(x1-x0)
        for values, color in zip(series, colors):
            yy = y1-(np.asarray(values)-low)/(high-low)*(y1-y0)
            points = list(zip(xx, np.clip(yy, y0, y1)))
            draw.line(points, fill=(215, 222, 230), width=2)
            if index > 0:
                draw.line(points[:index+1], fill=color, width=3)
        cursor = xx[index]
        draw.line([(cursor, y0), (cursor, y1)], fill=(220, 52, 45), width=2)
        draw.text((x0+5, y0+4), f'{high:.2f}', font=tiny, fill=(80, 90, 103))
        draw.text((x0+5, y1-22), f'{low:.2f}', font=tiny, fill=(80, 90, 103))

    first_violation = np.flatnonzero(speed > 1000)
    key_frames = {0, count-1, max(0, count-13)}
    report = dict(measured_frames=count, actual_physics_poses=True, interpolation=False,
                  omitted_nonphysical_visual_links=sorted(set(omitted)), self_collision=meta['self_collision'],
                  normal_speed_fps=fps, slow_motion_factor=8, selection=meta['selection'])
    try:
        for i in range(count):
            if not any(schedule[i] for schedule in schedules) and i not in key_frames:
                continue
            image = Image.new('RGB', (width, height), (242, 245, 248))
            for view_id, (scene, robots, assets, camera, camera_node, light_node) in enumerate(views):
                for body, node in robots:
                    scene.set_pose(node, pose(trace['body_pos'][i, body], trace['body_quat'][i, body]))
                for name, node in assets.items():
                    scene.set_pose(node, pose(trace[name][i, :3], trace[name][i, 3:]))
                if view_id:
                    # Fixed viewing direction while tracking the affected joint.
                    focus = monitored_position[i]
                    offset = np.array([.44, -.55, .25])
                    transform = look_at(focus+offset, focus)
                    scene.set_pose(camera_node, transform); scene.set_pose(light_node, transform)
                renderer.viewport_width = viewport_widths[view_id]
                rgb, _ = renderer.render(scene)
                image.paste(Image.fromarray(rgb), (0 if view_id == 0 else 1060, 100))
            draw = ImageDraw.Draw(image)
            draw.text((24, 14), 'G1 + Revo2 | SAPG + SONIC | Numerical-failure reproduction', font=title, fill=(22, 35, 48))
            draw.text((24, 58), f"Environment {meta['env_id']} / {meta['num_envs']} | {meta['family']} | "
                      f"epoch {int(trace['epoch'][i])} | sim t = {t[i]:.4f} s", font=text, fill=(62, 77, 92))
            draw.text((26, 112), 'Full-body physics', font=text, fill=(24, 39, 54))
            draw.text((1080, 112), 'Affected joint close-up', font=text, fill=(24, 39, 54))
            draw.text((1080, 145), 'Table translucent here for visibility only', font=small, fill=(70, 85, 101))
            draw.text((1080, 174), 'Red link: '+child, font=small, fill=(165, 28, 28))
            # Mark joint origin in both measured-pose cameras, not a contact sphere.
            for view_id, (scene, _, _, camera, camera_node, _) in enumerate(views):
                camera_pose = scene.get_pose(camera_node)
                clip = camera.get_projection_matrix(viewport_widths[view_id], viewport_height) @ np.linalg.inv(camera_pose) @ np.r_[monitored_position[i], 1.]
                if clip[3] > 0:
                    xy = clip[:2]/clip[3]
                    u = (xy[0]+1)*viewport_widths[view_id]/2+(1060 if view_id else 0)
                    v = (1-xy[1])*viewport_height/2+100
                    if (1060 if view_id else 0) < u < (1920 if view_id else 1060) and 100 < v < 804:
                        draw.ellipse((u-10, v-10, u+10, v+10), outline=(230, 25, 25), width=3)
            draw.text((24, 816), f"{joint}: q = {angle[i]:+.3f} rad | target = {command[i]:+.3f} rad | "
                      f"speed = {speed[i]:.1f} rad/s", font=text, fill=(160, 25, 25) if speed[i] > 1000 else (24, 39, 54))
            chart(draw, plots[0], [angle, command], [(45, 118, 183), (228, 145, 30)], i, ymin, ymax)
            chart(draw, plots[1], [np.log10(1+speed)], [(206, 40, 40)], i, 0, maxlog)
            draw.text((32, 847), 'Joint angle (rad): measured blue / commanded orange', font=small, fill=(35, 48, 63))
            draw.text((1010, 847), 'Joint speed: log10(1 + |rad/s|)', font=small, fill=(35, 48, 63))
            threshold_y = plots[1][3]-np.log10(1001)/maxlog*(plots[1][3]-plots[1][1])
            draw.line([(plots[1][0], threshold_y), (plots[1][2], threshold_y)], fill=(226, 86, 64), width=1)
            draw.text((1450, threshold_y+2), 'Safety threshold: 1000 rad/s', font=tiny, fill=(170, 50, 35))
            draw.text((24, 1037), 'Measured PhysX motion at 120 Hz; no pose interpolation. Task control 60 Hz, touch 70 Hz. Red circle marks joint origin.',
                      font=small, fill=(65, 80, 96))
            if i and trace['resets'][i, 0] != trace['resets'][i-1, 0]:
                draw.rounded_rectangle((24, 739, 390, 786), radius=8, fill=(250, 225, 183))
                draw.text((35, 748), 'TASK RESET (not edited out)', font=text, fill=(110, 64, 10))
            if speed[i] > 1000:
                draw.rounded_rectangle((24, 187, 684, 242), radius=8, fill=(245, 207, 203))
                draw.text((36, 198), 'SAFETY STOP - final measured state', font=title, fill=(152, 22, 22))
            if i in key_frames:
                image.save(args.recording/f'failure-preview-{i:04d}.png')
            for which, encoder in enumerate(encoders):
                if schedules[which][i]:
                    frame = image.copy()
                    labels = ImageDraw.Draw(frame)
                    labels.rounded_rectangle((1550, 12, 1895, 51), radius=7, fill=(215, 229, 242))
                    labels.text((1563, 17), 'NORMAL SPEED' if which == 0 else '8x SLOW MOTION', font=text, fill=(28, 70, 108))
                    pixels = np.asarray(frame).tobytes()
                    for _ in range(int(schedules[which][i])):
                        encoder.stdin.write(pixels)
                        encoded[which] += 1
            if i % 120 == 0:
                print(f'FAILURE_RENDER_FRAME {i}/{count}', flush=True)
    finally:
        for encoder in encoders:
            encoder.stdin.close()
        codes = [encoder.wait(timeout=60) for encoder in encoders]
        renderer.delete()
    if any(codes):
        raise RuntimeError(f'FFmpeg failed: {codes}')
    report['videos'] = []
    for name, frames in zip(outputs, encoded):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(args.recording/name)], text=True))
        stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        if (stream['width'], stream['height'], int(stream['nb_frames'])) != (width, height, frames):
            raise RuntimeError('Encoded video dimensions/frame count mismatch')
        report['videos'].append(dict(name=name, frames=frames, duration_s=float(probe['format']['duration'])))
    (args.recording/'video_validation.json').write_text(json.dumps(report, indent=2))
    print('FAILURE_VIDEO_VALIDATED '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
