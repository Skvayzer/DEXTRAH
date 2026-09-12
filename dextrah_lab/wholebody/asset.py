"""Audit and stage the COMPLETE G1/Revo2 URDF without reduced-body conversion."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import BODY_JOINTS, hand_joints


def audit_urdf(path):
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    links = {link.get('name'): link for link in root.findall('link')}
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    if len(links) != len(root.findall('link')) or len(joints) != len(root.findall('joint')):
        raise ValueError('Duplicate link or joint')
    required = (*BODY_JOINTS, *hand_joints('left'), *hand_joints('right'))
    if set(required)-set(joints):
        raise ValueError(f'Missing body/hand joints: {set(required)-set(joints)}')
    children = {joint.find('child').get('link') for joint in joints.values()}
    if len(children) != len(joints):
        raise ValueError('Multiple joints parent the same link')
    roots = set(links)-children
    if roots != {'pelvis'}:
        raise ValueError(f'Expected floating pelvis root, got {roots}')
    reached = {'pelvis'}
    for _ in range(len(joints)):
        before = len(reached)
        for joint in joints.values():
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in links or child not in links:
                raise ValueError('Joint references a nonexistent link')
            if parent in reached:
                reached.add(child)
        if len(reached) == before:
            break
    if reached != set(links):
        raise ValueError('Disconnected or cyclic robot tree')
    for name in required:
        joint = joints[name]
        limit = joint.find('limit')
        if joint.get('type') != 'revolute' or limit is None:
            raise ValueError(f'Required joint fixed/missing limits: {name}')
        if not (float(limit.get('upper')) > float(limit.get('lower')) and
                float(limit.get('effort')) > 0 and float(limit.get('velocity')) > 0):
            raise ValueError(f'Invalid actuator limits: {name}')
    mass, masses, warnings = 0., {}, []
    for name, link in links.items():
        inertial = link.find('inertial')
        if inertial is None:
            warnings.append(f'No explicit inertia on link {name}; inspect fixed-link merging')
            continue
        m = float(inertial.find('mass').get('value'))
        v = inertial.find('inertia').attrib
        inertia = np.array([[float(v['ixx']),float(v['ixy']),float(v['ixz'])],
                            [float(v['ixy']),float(v['iyy']),float(v['iyz'])],
                            [float(v['ixz']),float(v['iyz']),float(v['izz'])]])
        moments = np.linalg.eigvalsh(inertia)
        if not np.isfinite(inertia).all() or not np.isfinite(m) or m <= 0 or moments.min() <= 0:
            raise ValueError(f'Invalid mass/inertia: {name}')
        if moments[-1] > moments[0]+moments[1]+1e-9:
            warnings.append(f'Principal inertia triangle inequality violated: {name}')
        mass += m
        masses[name] = m
    mimic = {}
    locked = {}
    for name, joint in joints.items():
        relation = joint.find('mimic')
        if relation is not None:
            mimic[name] = dict(relation.attrib)
        limit = joint.find('limit')
        if joint.get('type') == 'revolute' and limit is not None and float(limit.get('lower')) == float(limit.get('upper')):
            angle = float(limit.get('lower'))
            if not np.isfinite(angle):
                raise ValueError(f'Nonfinite locked joint angle: {name}')
            locked[name] = angle
    for side in ('left','right'):
        for finger in ('thumb','index','middle','ring','pinky'):
            name = f'{side}_{finger}_distal_joint'
            expected = dict(joint=f'{side}_{finger}_proximal_joint',
                            multiplier=1. if finger=='thumb' else 1.155, offset=0.)
            if name not in mimic or any((mimic[name].get(k) != value if isinstance(value,str)
                                        else float(mimic[name].get(k,'0')) != value)
                                       for k,value in expected.items()):
                raise ValueError(f'Unexpected Revo2 coupling: {name}')
    return dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        links=len(links),joint_count=len(joints),body_commands=29,hand_commands=12,
        total_mass_kg=mass,link_masses_kg=masses,mimic_relations=mimic,
        zero_range_joints=list(locked),locked_joint_angles_rad=locked,
        collision_shapes=len(root.findall('./link/collision')),
        warnings=warnings,physics_validated=False)


def prepare_urdf(source, output):
    """Only fix zero-range sensor-tip joints and resolve mesh filenames.

    Preserve all branches, inertial/collision data and native mimic relations.
    No gravity-off flags or world attachment are introduced. The simulator
    configuration must separately require fix_base=False and gravity=True.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_urdf(source)
    root = ET.parse(source).getroot()
    for joint in root.findall('joint'):
        if joint.get('name') in report['zero_range_joints']:
            # URDF FK is T_origin @ R_axis(q), NOT R_axis(q) @ T_origin.
            # Folding a nonzero locked angle into the fixed origin preserves
            # the child frame, including its visual/collision/inertial frames.
            angle = report['locked_joint_angles_rad'][joint.get('name')]
            if angle != 0:
                origin = joint.find('origin')
                if origin is None:
                    origin = ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
                axis_element = joint.find('axis')
                axis = np.fromstring(axis_element.get('xyz') if axis_element is not None else '1 0 0', sep=' ')
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                    raise ValueError(f'Invalid locked joint axis: {joint.get("name")}')
                original = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
                rotated = original * Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle)
                origin.set('rpy', ' '.join(format(v, '.17g') for v in rotated.as_euler('xyz')))
            joint.set('type','fixed')
            for tag in ('limit','axis','dynamics'):
                element = joint.find(tag)
                if element is not None:
                    joint.remove(element)
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename')
        if '://' in filename:
            raise ValueError(f'Unresolved mesh URI: {filename}')
        asset = (source.parent/filename).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        mesh.set('filename',str(asset))
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    after = audit_urdf(output)
    if report['link_masses_kg'] != after['link_masses_kg'] or report['collision_shapes'] != after['collision_shapes']:
        raise ValueError('Asset preparation changed mass or collision shape count')
    report.update(prepared=str(output),prepared_sha256=after['sha256'],
        removed_branches=[],locked_sensor_joints_fixed=report['zero_range_joints'],
        native_mimic_preserved=True)
    output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
