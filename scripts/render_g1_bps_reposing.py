#!/usr/bin/env python3
"""Render measured reposing and an exact canonical BPS descriptor side by side."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording',type=Path)
    p.add_argument('--video-name',default='sapg-bps128-reposing.mp4')
    p.add_argument('--limit-frames',type=int,default=0)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation for EGL rendering')
    video=args.recording/args.video_name
    if video.exists():
        raise FileExistsError(video)
    os.environ['PYOPENGL_PLATFORM']='egl'
    import numpy as np
    import pyrender
    import trimesh
    import yourdfpy
    from PIL import Image, ImageDraw, ImageFont
    from pyrender.platforms import egl
    devices=[i for i,d in enumerate(egl.query_devices()) if b'EGL_NV_device_cuda' in
             (egl._eglQueryDeviceStringEXT(d._display,0x3055) or b'')]
    if len(devices)!=1:
        raise RuntimeError(f'Expected one Slurm-visible NVIDIA EGL device: {devices}')
    os.environ['EGL_DEVICE_ID']=str(devices[0])
    meta=json.loads((args.recording/'metadata.json').read_text())
    with np.load(args.recording/'trajectory.npz',allow_pickle=False) as data:
        trace={k:data[k] for k in data.files}
    with np.load(args.recording/'bps_geometry.npz',allow_pickle=False) as data:
        bps={k:data[k] for k in data.files}
    np.testing.assert_allclose(np.linalg.norm(bps['basis']-bps['nearest'],axis=1),
                               bps['distances'],rtol=1e-6,atol=1e-7)
    frames=min(meta['frames'],args.limit_frames or meta['frames'])
    if frames<1 or not meta['completed']:
        raise ValueError('Recording is not complete')

    def pose(position,quaternion):
        mat=trimesh.transformations.quaternion_matrix(quaternion)
        mat[:3,3]=position
        return mat

    def look_at(eye,target):
        eye,target=np.asarray(eye,dtype=float),np.asarray(target,dtype=float)
        z=eye-target; z/=np.linalg.norm(z)
        x=np.cross([0.,0.,1.],z); x/=np.linalg.norm(x)
        m=np.eye(4);m[:3,:3]=np.column_stack((x,np.cross(z,x),z));m[:3,3]=eye
        return m

    def mesh_node(scene,mesh,color):
        mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=.04,
            roughnessFactor=.65,doubleSided=True,alphaMode='BLEND' if color[3]<1 else 'OPAQUE')
        return scene.add(pyrender.Mesh.from_trimesh(mesh,material=mat,smooth=False))

    def scene_camera(scene,eye,target,fov):
        transform=look_at(eye,target)
        scene.add(pyrender.PerspectiveCamera(yfov=fov,znear=.01,zfar=20.),pose=transform)
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5),pose=transform)
        return transform

    scene=pyrender.Scene(bg_color=[.94,.96,.98,1.],ambient_light=[.4,.4,.4])
    urdf=yourdfpy.URDF.load(meta['robot_urdf'])
    for joint in urdf.joint_map.values():
        joint.mimic=None
    urdf.update_cfg(meta['visual_static_joint_pos'])
    urdf.update_cfg(dict(zip(meta['joint_names'],trace['joint_pos'][0])))
    handles=[]
    for name in urdf.scene.graph.nodes_geometry:
        transform,geometry=urdf.scene.graph.get(name)
        link=urdf.scene.graph.transforms.parents[name]
        while link not in urdf.link_map:
            link=urdf.scene.graph.transforms.parents[link]
        mesh=urdf.scene.geometry[geometry].copy()
        mesh.apply_transform(np.linalg.inv(urdf.get_transform(link))@transform)
        color=[.70,.73,.77,1.] if 'right_' in link else [.33,.38,.44,1.]
        handles.append((link,mesh_node(scene,mesh,color)))
    assets={}
    for name in ('object','table','goal'):
        mesh=trimesh.load(args.recording/('object.glb' if name=='goal' else f'{name}.glb'),force='mesh')
        assets[name]=mesh_node(scene,mesh,dict(object=[1.,.48,.08,1.],
            table=[.46,.55,.61,1.],goal=[.10,.9,.35,.28])[name])
    ground=trimesh.creation.box([3.,3.,.01]);ground.apply_translation([0.,0.,-.006])
    mesh_node(scene,ground,[.83,.86,.89,1.])
    main_camera=scene_camera(scene,[.72,-.75,1.26],[0.,.16,.95],.85)
    scene.add(pyrender.PointLight(color=np.ones(3),intensity=6.),pose=look_at([-1.,-.5,2.],[0.,0.,1.]))

    # Every node in the inset rotates together: presentation-camera-equivalent
    # motion only. This never rotates/recomputes the policy's BPS vector.
    inset=pyrender.Scene(bg_color=[.075,.10,.145,1.],ambient_light=[.5,.5,.5])
    nodes=[]
    nodes.append(mesh_node(inset,trimesh.Trimesh(bps['vertices'],bps['faces'],process=False),[.57,.66,.75,1.]))
    selected=16
    for points,radius,color in [(bps['basis'],.012,[.98,.70,.20,1.]),
                                 (bps['nearest'],.01,[.12,.86,.72,1.])]:
        spheres=[]
        for point in points:
            sphere=trimesh.creation.icosphere(subdivisions=1,radius=radius)
            sphere.apply_translation(point);spheres.append(sphere)
        nodes.append(mesh_node(inset,trimesh.util.concatenate(spheres),color))
    lines=[]
    for a,b in zip(bps['basis'],bps['nearest']):
        length=np.linalg.norm(b-a)
        if length<1e-7:
            continue
        cylinder=trimesh.creation.cylinder(radius=.0016,height=length,sections=6)
        cylinder.apply_transform(trimesh.geometry.align_vectors([0.,0.,1.],b-a))
        cylinder.apply_translation((a+b)/2)
        lines.append(cylinder)
    nodes.append(mesh_node(inset,trimesh.util.concatenate(lines),[.35,.47,.61,1.]))
    a,b=bps['basis'][selected],bps['nearest'][selected]
    link=trimesh.creation.cylinder(radius=.007,height=np.linalg.norm(b-a),sections=8)
    link.apply_transform(trimesh.geometry.align_vectors([0.,0.,1.],b-a));link.apply_translation((a+b)/2)
    nodes.append(mesh_node(inset,link,[.96,.25,.58,1.]))
    scene_camera(inset,[2.,-3.,1.8],[0.,0.,0.],.70)

    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title=ImageFont.truetype(font,31)
    text=ImageFont.truetype(font,24)
    small=ImageFont.truetype(font,19)
    tiny=ImageFont.truetype(font,16)
    # Cache one period of the static shape presentation, then reuse a SINGLE
    # EGL context for the physical rollout. Separate pyrender contexts share
    # an EGL display; deleting one terminates the display beneath the other.
    renderer=pyrender.OffscreenRenderer(640,590)
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','1920x1080','-framerate',str(meta['fps']),
        '-i','-','-an','-c:v','libx264','-crf','19','-preset','fast','-threads','2',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    bodies={name:i for i,name in enumerate(meta['body_names'])}
    goal_events=np.flatnonzero(np.diff(trace['goal_hits'][:frames],prepend=0)>0)
    reset_events=np.flatnonzero(np.diff(trace['resets'][:frames],prepend=0)>0)
    previews={0,60,min(frames-1,300)}
    if len(goal_events):
        previews.add(min(frames-1,int(goal_events[0])+10))
    validation=dict(frames=0,max_fk_error_m=0.,max_fk_angle_rad=0.,
        bps_queries=128,descriptor_error=float(np.max(np.abs(np.linalg.norm(bps['basis']-bps['nearest'],axis=1)-bps['distances']))),
        policy_frames_replayed=True,frame_interpolation=False,whole_body='Static visual context, not controlled')
    try:
        shape_frames=[]
        for j in range(min(frames,24*meta['fps'])):
            rotation=trimesh.transformations.rotation_matrix(
                j/meta['fps']*2*np.pi/24,[0.,0.,1.])
            for node in nodes:
                inset.set_pose(node,rotation)
            shape_frames.append(renderer.render(inset)[0])
        renderer.viewport_width=1280
        renderer.viewport_height=824
        for i in range(frames):
            urdf.update_cfg(dict(zip(meta['joint_names'],trace['joint_pos'][i])))
            root=pose(trace['robot'][i,:3],trace['robot'][i,3:])
            for link,node in handles:
                fk=root@urdf.get_transform(link)
                actual=fk
                if link in bodies:
                    index=bodies[link]
                    actual=pose(trace['body_pos'][i,index],trace['body_quat'][i,index])
                    error=float(np.linalg.norm(fk[:3,3]-actual[:3,3]))
                    angle=float(np.arccos(np.clip((np.trace(fk[:3,:3].T@actual[:3,:3])-1)/2,-1,1)))
                    validation['max_fk_error_m']=max(validation['max_fk_error_m'],error)
                    validation['max_fk_angle_rad']=max(validation['max_fk_angle_rad'],angle)
                    if error>.002 or angle>.01:
                        raise RuntimeError(f'URDF/PhysX mismatch: {link} {error}m {angle}rad')
                scene.set_pose(node,actual)
            for name,node in assets.items():
                scene.set_pose(node,pose(trace[name][i,:3],trace[name][i,3:]))
            seconds=float(trace['step'][i]*meta['policy_dt'])
            rgb,_=renderer.render(scene)
            shape_rgb=shape_frames[i%len(shape_frames)]
            image=Image.new('RGB',(1920,1080),(242,245,248))
            image.paste(Image.fromarray(rgb),(0,96));image.paste(Image.fromarray(shape_rgb),(1280,96))
            draw=ImageDraw.Draw(image)
            draw.text((24,15),f"SAPG + BPS-128 | G1 + BrainCo Revo2 | {meta['object_family'].capitalize()}",font=title,fill=(25,38,52))
            draw.text((24,58),'Final 8B checkpoint | deterministic leader | training disturbances retained | continuous rollout',font=text,fill=(60,75,90))
            draw.text((1300,106),'BPS-128: canonical object shape',font=text,fill=(234,240,248))
            draw.text((1300,651),'Amber: queries   Mint: nearest surface samples',font=small,fill=(225,235,245))
            draw.text((1300,697),'128 distance values supplied to the policy',font=small,fill=(25,38,52))
            left,right,top,bottom=1300,1890,745,840
            ymax=max(1.,float(np.ceil(bps['distances'].max()*5)/5))
            for j,distance in enumerate(bps['distances']):
                x=left+j*(right-left)/128
                color=(242,64,147) if j==selected else (85,122,156)
                draw.rectangle([x,bottom-distance/ymax*(bottom-top),x+3,bottom],fill=color)
            draw.text((left,845),f"Index 0 ... 127 | highlighted d[{selected}] = {bps['distances'][selected]:.3f}",font=tiny,fill=(60,75,90))
            draw.text((1300,876),'Display rotates; BPS values remain unchanged.',font=small,fill=(60,75,90))
            draw.line([(1280,96),(1280,920)],fill=(200,210,220),width=2)
            line=(f"{seconds:05.2f} / {meta['seconds']:.0f} s    Goals reached: {int(trace['goal_hits'][i])}"
                  f"    Resets: {int(trace['resets'][i])}    Pose error: {trace['goal_error'][i]*100:.1f} cm")
            draw.text((24,938),line,font=text,fill=(25,38,52))
            draw.text((24,978),'Orange: object | Green: target pose | Success: max 4-keypoint error <= 1.5 cm for 10 accumulated control steps',font=small,fill=(60,75,90))
            draw.text((24,1015),'BPS distances describe shape, not collision avoidance. Four original pose keypoints and rewards are unchanged.',font=small,fill=(60,75,90))
            draw.text((24,1048),'Measured simulated arm/hand motion; whole body is fixed visual context. No tactile sensing, fabrics, PCA, or training during recording.',font=tiny,fill=(60,75,90))
            if np.any((i-goal_events>=0)&(i-goal_events<2*meta['fps'])):
                draw.rounded_rectangle([24,115,440,158],radius=8,fill=(212,245,222))
                draw.text((38,122),'REPOSING GOAL REACHED',font=text,fill=(20,108,48))
            if np.any((i-reset_events>=0)&(i-reset_events<meta['fps'])):
                draw.rounded_rectangle([24,171,310,211],radius=8,fill=(251,226,201))
                draw.text((38,177),'Environment reset',font=text,fill=(135,70,15))
            encoder.stdin.write(np.asarray(image).tobytes())
            if i in previews:
                image.save(args.recording/f'bps-preview-{i:04d}.png')
            validation['frames']+=1
            if i%180==0:
                print(f'BPS_VIDEO_FRAME {i}/{frames}',flush=True)
    finally:
        encoder.stdin.close()
        code=encoder.wait(timeout=60)
        renderer.delete()
    if code:
        raise RuntimeError(f'FFmpeg failed: {code}')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
        '-show_format','-of','json',str(video)],text=True))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'])==(1920,1080)
    assert int(stream['nb_frames'])==frames
    validation.update(video=str(video),duration_seconds=float(probe['format']['duration']),
                      encoded_frames=int(stream['nb_frames']))
    (args.recording/(Path(args.video_name).stem+'-validation.json')).write_text(json.dumps(validation,indent=2))
    print('BPS_VIDEO_VALIDATED '+json.dumps(validation),flush=True)


if __name__=='__main__':
    main()
