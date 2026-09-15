#!/usr/bin/env python3
"""Render measured full-body checkpoint rollouts, with every reset retained."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    args = parser.parse_args()
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
    navigation = meta.get('experiment', {}).get('type') == 'brush_navigation_transfer'
    transfer = navigation or meta.get('experiment', {}).get('type') == 'brush_table_transfer'
    from dextrah_lab.wholebody.recording_contract import goal_rule_caption
    termination = meta.get('goal_termination_config')
    criterion_source = 'capture metadata'
    if termination is None:
        # Legacy captures used this saved source config verbatim. Never use a
        # hard-coded numeric caption, or the current training progress file.
        from dextrah_lab.g1_adept.touch_continuation import load_yaml
        source_cfg = Path(meta['task_contract']['source_run'])/'params/env_resolved.yaml'
        termination = load_yaml(source_cfg)['termination']
        criterion_source = str(source_cfg)
    goal_caption = goal_rule_caption(termination)
    with np.load(args.recording/'trajectory.npz', allow_pickle=False) as data:
        trace = dict(data)
    if not meta['completed'] or meta['frames'] != len(trace['time_s']):
        raise ValueError('Incomplete capture')
    for key in ('body_pos', 'body_quat', 'object', 'goal', 'table'):
        if not np.isfinite(trace[key]).all():
            raise ValueError(f'Nonfinite captured geometry: {key}')
    if transfer and not np.isfinite(trace['receiving_table']).all():
        raise ValueError('Nonfinite receiving-table geometry')
    if not np.allclose(np.diff(trace['time_s']), 1/meta['fps'], atol=1e-6):
        raise ValueError('Nonuniform capture timestamps; do not invent frames')

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

    bodies = {name: i for i, name in enumerate(meta['body_names'])}
    wrist = next((bodies[name] for name in ('right_wrist_yaw_link', 'right_hand_base_link') if name in bodies), None)
    if wrist is None:
        raise ValueError('Measured wrist link missing')
    urdf = yourdfpy.URDF.load(meta['robot_urdf'])
    geometry, omitted = [], []
    for node in urdf.scene.graph.nodes_geometry:
        transform, name = urdf.scene.graph.get(node)
        link = urdf.scene.graph.transforms.parents[node]
        while link not in urdf.link_map:
            link = urdf.scene.graph.transforms.parents[link]
        if link not in bodies:
            omitted.append(link)
            continue
        mesh = urdf.scene.geometry[name].copy()
        mesh.apply_transform(np.linalg.inv(urdf.get_transform(link))@transform)
        geometry.append((bodies[link], mesh, [.70, .74, .79, 1.] if 'right_' in link else [.30, .36, .43, 1.]))

    def make_scene(close):
        scene = pyrender.Scene(bg_color=[.94, .96, .98, 1.], ambient_light=[.43, .43, .43])
        robots = [(index, add(scene, mesh, color)) for index, mesh, color in geometry]
        assets = {}
        for name in (('object', 'table', 'goal', 'receiving_table') if transfer else ('object', 'table', 'goal')):
            path = args.recording/('object.glb' if name == 'goal' else f'{name}.glb')
            color = dict(object=[1., .47, .06, 1.], table=[.50, .58, .64, .22 if close else 1.],
                         goal=[.10, .82, .30, .20], receiving_table=[.20, .48, .72, .25 if close else 1.])[name]
            assets[name] = add(scene, trimesh.load(path, force='mesh'), color)
        ground = trimesh.creation.box([4., 4., .01]); ground.apply_translation([0., 0., -.006])
        add(scene, ground, [.82, .86, .89, 1.])
        camera_pose = (look_at([1.2, -2.1, 1.65], [-.25, .18, .76]) if transfer
                       else look_at([1.65, -2., 1.6], [0., .25, .76]))
        camera = scene.add(pyrender.PerspectiveCamera(yfov=.66 if close else .70, znear=.005, zfar=30.), pose=camera_pose)
        light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=camera_pose)
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=5.), pose=look_at([-1., -.8, 2.], [0., 0., 1.]))
        return scene, robots, assets, camera, light

    views = [make_scene(False), make_scene(True)]
    width, height, view_height = 1600, 900, 690
    widths = [860, 740]
    renderer = pyrender.OffscreenRenderer(widths[0], view_height)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title, text, small = [ImageFont.truetype(font_path, size) for size in (29, 22, 18)]
    suffix = '-table-transfer' if transfer else ''
    if navigation:
        suffix = '-navigation-transfer'
    video = args.recording/f"g1-sonic-sapg-{meta['object_family']}{suffix}.mp4"
    encoder = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(meta['fps']),
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    reset_events = np.flatnonzero(np.diff(trace['resets'], prepend=0) > 0)
    fall_events = np.flatnonzero(np.diff(trace['robot_falls'], prepend=0) > 0)
    previews = {0, 30*meta['fps'], meta['frames']-1}
    if len(fall_events):
        previews.add(max(0, int(fall_events[0])-2))
    try:
        for i in range(meta['frames']):
            image = Image.new('RGB', (width, height), (242, 245, 248))
            for k, (scene, robots, assets, camera, light) in enumerate(views):
                for body, node in robots:
                    scene.set_pose(node, pose(trace['body_pos'][i, body], trace['body_quat'][i, body]))
                for name, node in assets.items():
                    scene.set_pose(node, pose(trace[name][i, :3], trace[name][i, 3:]))
                if k:
                    focus = trace['body_pos'][i, wrist]
                    camera_pose = look_at(focus+[.43, -.54, .28], focus)
                    scene.set_pose(camera, camera_pose); scene.set_pose(light, camera_pose)
                renderer.viewport_width = widths[k]
                rgb, _ = renderer.render(scene)
                image.paste(Image.fromarray(rgb), (0 if k == 0 else widths[0], 96))
            draw = ImageDraw.Draw(image)
            controller = 'Frozen SONIC' if meta.get('controller_mode') == 'frozen_pretrained_latent' else 'SONIC'
            label = ('Brush transfer with navigation' if navigation else 'Brush table transfer') if transfer else meta['object_family'].capitalize()
            draw.text((22, 12), f"G1 + Revo2 | {controller} + SAPG | {label}", font=title, fill=(22, 35, 48))
            draw.text((22, 54), f"Checkpoint epoch {meta['checkpoint_epoch']} | deterministic leader | uncut {meta['seconds']:g}-second rollout", font=text, fill=(62, 77, 92))
            draw.text((22, 108), 'Full-body physics', font=text, fill=(24, 39, 54))
            draw.text((882, 108), 'Hand close-up (table translucent for visibility)', font=small, fill=(24, 39, 54))
            line = (f"t={trace['time_s'][i]:05.2f}s | Goals: {int(trace['goals'][i])} | Resets: {int(trace['resets'][i])}"
                    f" | Robot falls: {int(trace['robot_falls'][i])} | Pose error: {100*trace['goal_error'][i]:.1f} cm")
            if transfer:
                phase = meta['experiment']['phase_names'][int(trace['transfer_phase'][i])]
                line = (f"t={trace['time_s'][i]:05.2f}s | Stage: {phase} | Carried over receiver: {int(trace['transfer_carried'][i])}"
                        f" | Placed/released: {int(trace['transfer_placed'][i])} | Resets: {int(trace['resets'][i])}"
                        f" | Robot falls: {int(trace['robot_falls'][i])}")
            draw.text((22, 798), line, font=text, fill=(24, 39, 54))
            if transfer:
                line = (f"Receiver contact: {trace['transfer_receiver_force_n'][i]:.2f} N | Robot-object contact: {trace['transfer_robot_object_force_n'][i]:.2f} N"
                        f" | Pelvis XY displacement: {100*trace['transfer_root_displacement_m'][i]:.1f} cm")
                footer = 'Orange: brush | Blue: receiver | Green: scripted OBJECT goal. No forced release, walking commands, or new training.'
                if navigation:
                    line = (f"cmd_vel body: {trace['transfer_cmd_vx'][i]:+.2f}, {trace['transfer_cmd_vy'][i]:+.2f} m/s, {trace['transfer_cmd_wz'][i]:+.2f} rad/s"
                            f" | Measured: {trace['transfer_measured_vx'][i]:+.2f}, {trace['transfer_measured_vy'][i]:+.2f} m/s"
                            f" | Waypoint: {int(trace['transfer_waypoint'][i])+1} | Arrived: {int(trace['transfer_navigation_arrived'][i])}")
                    footer = 'Velocity commands -> NVIDIA motion planner -> frozen SONIC + SAPG. Body-relative carry target; no new training or forced release.'
            else:
                line = 'Orange: object | Green: target pose | '+goal_caption
                footer = 'Measured PhysX body poses; no interpolation. BPS-128 + touch 70 Hz. No training or outcome-based selection.'
            draw.text((22, 835), line, font=small, fill=(62, 77, 92))
            draw.text((22, 866), footer, font=small, fill=(62, 77, 92))
            recent_fall = np.any((i-fall_events >= 0) & (i-fall_events < meta['fps']))
            recent_reset = np.any((i-reset_events >= 0) & (i-reset_events < meta['fps']))
            if recent_reset:
                draw.rounded_rectangle((22, 150, 430, 194), radius=7, fill=(251, 218, 199))
                draw.text((33, 158), 'ROBOT FALL / RESET' if recent_fall else 'TASK RESET', font=text, fill=(140, 48, 22))
            if i in previews:
                image.save(args.recording/f'preview-{i:04d}.png')
            encoder.stdin.write(np.asarray(image).tobytes())
            if i % 300 == 0:
                print(f'WHOLEBODY_VIDEO_FRAME {i}/{meta["frames"]}', flush=True)
    finally:
        encoder.stdin.close()
        code = encoder.wait(timeout=60)
        renderer.delete()
    if code:
        raise RuntimeError(f'ffmpeg failed: {code}')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], text=True))
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    if (stream['width'], stream['height'], int(stream['nb_frames'])) != (width, height, meta['frames']):
        raise RuntimeError('Encoded dimensions/frame count mismatch')
    validation = dict(video=str(video), frames=meta['frames'], duration_s=float(probe['format']['duration']),
        measured_body_poses=True, frame_interpolation=False, all_resets_retained=True,
        goal_caption=goal_caption, goal_termination_config=termination,
        goal_criterion_source=criterion_source, renderer_source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'),
        experiment=meta.get('experiment'),
        omitted_nonphysical_visual_links=sorted(set(omitted)))
    (args.recording/'video_validation.json').write_text(json.dumps(validation, indent=2))
    print('WHOLEBODY_VIDEO_VALIDATED '+json.dumps(validation), flush=True)


if __name__ == '__main__':
    main()
