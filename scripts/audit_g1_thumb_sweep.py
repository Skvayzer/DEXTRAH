#!/usr/bin/env python3
"""Check the entire sampled thumb range against every fixed-merged wrist mesh.

This is a CAD check, not a continuous-collision proof or hardware calibration.
All pair exclusions remain diagnostic unless their physical implications are
reviewed. The source robot assets are read-only.
"""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from audit_g1_thumb_housing import collision_mesh
from dextrah_lab.wholebody.kinematics import UrdfKinematics


def fixed_cluster(root, body_name):
    members={body_name}
    while True:
        before=len(members)
        for joint in root.findall('joint'):
            if joint.get('type')=='fixed' and joint.find('parent').get('link') in members:
                members.add(joint.find('child').get('link'))
        if len(members)==before:
            return sorted(members)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=41)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in a Slurm CPU allocation')
    if args.grid<3 or args.output.exists():
        raise ValueError('Require grid>=3 and a new output path')
    root=ET.parse(args.urdf).getroot()
    links={link.get('name'):link for link in root.findall('link')}
    names={name.lower():name for name in links}
    tree=UrdfKinematics(args.urdf)
    neutral={j.get('name'):0. for j in root.findall('joint')
             if j.get('type')=='revolute' and j.find('mimic') is None and
             j.find('limit').get('lower')!=j.find('limit').get('upper')}
    reports=[]
    hashes={}
    for side in ('left','right'):
        thumb_name=names[f'{side}_thumb_proximal_link']
        thumb,source=collision_mesh(links[thumb_name],args.urdf.parent)
        hashes.update(source)
        wrist_name=f'{side}_wrist_yaw_link'
        pieces=[]
        cluster=fixed_cluster(root,wrist_name)
        wrist_tf=tree.transform(wrist_name,neutral,1)[0]
        for name in cluster:
            mesh,source=collision_mesh(links[name],args.urdf.parent)
            hashes.update(source)
            if mesh is not None:
                mesh.apply_transform(np.linalg.inv(wrist_tf) @ tree.transform(name,neutral,1)[0])
                pieces.append(mesh)
        wrist=trimesh.util.concatenate(pieces)
        manager=trimesh.collision.CollisionManager()
        manager.add_object(wrist_name,wrist)
        joints=[f'{side}_thumb_{part}_joint' for part in ('metacarpal','proximal')]
        axes=[]
        for name in joints:
            limit=tree.joints[name].find('limit')
            axes.append(np.linspace(float(limit.get('lower')),float(limit.get('upper')),args.grid))
        grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
        positions={**neutral,**dict(zip(joints,grid.T))}
        transforms=np.linalg.inv(wrist_tf)[None] @ tree.transform(thumb_name,positions,len(grid))
        hits=[]
        minimum=float('inf')
        for q,tf in zip(grid,transforms):
            collision=manager.in_collision_single(thumb,transform=tf)
            clearance=manager.min_distance_single(thumb,transform=tf)
            minimum=min(minimum,clearance)
            if collision:
                hits.append(q.tolist())
        reports.append(dict(side=side,wrist_merged_links=cluster,joints=joints,
            joint_ranges_rad=[[float(a[0]),float(a[-1])] for a in axes],
            grid_shape=[args.grid,args.grid],sample_count=len(grid),
            cad_intersection_count=len(hits),intersecting_joint_positions_rad=hits,
            minimum_sampled_clearance_m=minimum))
        print(json.dumps(reports[-1]),flush=True)
    report=dict(source_urdf=str(args.urdf.resolve()),meshes_sha256=hashes,results=reports,
        all_sampled_poses_clear=all(r['cad_intersection_count']==0 for r in reports),
        continuous_collision_proof=False,hardware_calibrated=False,robot_assets_modified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
