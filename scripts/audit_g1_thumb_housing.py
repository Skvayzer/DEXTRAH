#!/usr/bin/env python3
"""CPU exact-CAD versus convex-hull overlap audit at measured hand poses.

Read-only with respect to robot assets. Does not authorize a permanent pair
filter or claim calibrated hardware geometry. Outputs contact locations in the
proximal-thumb frame so bearing-region artifacts can be distinguished from pad
or phalanx contacts. Requires python-fcl in the isolated SONIC overlay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from dextrah_lab.wholebody.kinematics import UrdfKinematics


def collision_mesh(link, directory):
    pieces=[]
    hashes={}
    for collision in link.findall('collision'):
        element=collision.find('geometry/mesh')
        if element is None:
            raise ValueError('This CAD diagnostic requires mesh collision geometry')
        path=(directory/element.get('filename')).resolve()
        mesh=trimesh.load(path,force='mesh',process=False)
        mesh.apply_scale(np.fromstring(element.get('scale','1 1 1'),sep=' '))
        origin=collision.find('origin')
        transform=np.eye(4)
        if origin is not None:
            transform[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        mesh.apply_transform(transform)
        pieces.append(mesh)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not pieces:
        return None,hashes
    return trimesh.util.concatenate(pieces),hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run the mesh collision audit in a Slurm CPU allocation')
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.probe/'trace.npz',allow_pickle=False) as archive:
        trace=dict(archive)
    metadata=json.loads((args.probe/'validation.json').read_text())
    names=metadata['joint_names']
    root=ET.parse(args.urdf).getroot()
    links={link.get('name'):link for link in root.findall('link')}
    lower={name.lower():name for name in links}
    tree=UrdfKinematics(args.urdf)
    reports=[]
    meshes={}
    for side in ('left','right'):
        proximal=names.index(f'{side}_thumb_proximal_joint')
        distal=names.index(f'{side}_thumb_distal_joint')
        errors=np.abs(trace['joint_pos'][:,0,distal]-trace['joint_pos'][:,0,proximal])
        samples=sorted(set([0,int(errors.argmax()),len(errors)-1]))
        thumb_name=lower[f'{side}_thumb_proximal_link']
        thumb,hashes=collision_mesh(links[thumb_name],args.urdf.parent)
        meshes.update(hashes)
        for sample in samples:
            positions=dict(zip(names,trace['joint_pos'][sample,0]))
            thumb_tf=tree.transform(thumb_name,positions,1)[0]
            for suffix in ('wrist_yaw_link','base2_link','base_link'):
                partner_name=lower[f'{side}_{suffix}']
                partner,hashes=collision_mesh(links[partner_name],args.urdf.parent)
                meshes.update(hashes)
                if partner is None:
                    continue
                relative=np.linalg.inv(thumb_tf) @ tree.transform(partner_name,positions,1)[0]
                for approximation in ('cad_triangles','convex_hulls','convex_thumb_only','convex_palm_only'):
                    a=thumb.convex_hull if approximation in ('convex_hulls','convex_thumb_only') else thumb
                    b=partner.convex_hull if approximation in ('convex_hulls','convex_palm_only') else partner
                    manager=trimesh.collision.CollisionManager()
                    manager.add_object(partner_name,b,transform=relative)
                    hit,contacts=manager.in_collision_single(a,return_data=True)
                    points=np.array([contact.point for contact in contacts])
                    reports.append(dict(side=side,sample=sample,time_s=sample/50,
                        proximal_angle_rad=float(positions[f'{side}_thumb_proximal_joint']),
                        opposition_angle_rad=float(positions[f'{side}_thumb_metacarpal_joint']),
                        coupling_error_rad=float(errors[sample]),partner=partner_name,
                        approximation=approximation,collision=bool(hit),contact_count=len(contacts),
                        proximal_frame_contact_bounds_m=([points.min(0).tolist(),points.max(0).tolist()] if len(points) else None)))
    report=dict(source_probe=str(args.probe.resolve()),source_urdf=str(args.urdf.resolve()),
        meshes_sha256=meshes,results=reports,robot_assets_modified=False,
        permanent_collision_filter_validated=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
