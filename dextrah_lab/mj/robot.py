"""G1 + BrainCo Revo2 as an MjSpec, built from the SAME URDF the Isaac task imports.

Mirrors the Play2Perfect/Isaac import (see docs/mjlab_port_plan_20261013.md):

* fixed sensor links ``imu_in_torso, imu_in_pelvis, d435_link, mid360_link``
  removed; revolute joints with lower == upper (the 10 ``*_tip`` joints)
  become fixed, without baking any angle (as `fix_locked_joints`);
* fixed-joint children stay separate bodies (``merge_fixed_joints=False``),
  so the 68 Isaac link names exist;
* every collision mesh is one convex hull (PhysX ``convex_hull``; vertex cap
  64 like PhysX's default hull limit), URDF masses/inertias are used as is;
* floating base, robot self-collision off (contype/conaffinity bits), visuals
  discarded;
* one position actuator per joint with the run's PD gains, armature and
  effort limits (`wholebody.actuators` for the 29 body joints, kp 1200 /
  kd 25 for both hands, efforts from the URDF).
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

ROBOT_URDF = Path('/data1/users/konstantin.smirnov/play2perfect/unitree_ros/robots/'
                  'g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf')
REMOVED_LINKS = ('imu_in_torso', 'imu_in_pelvis', 'd435_link', 'mid360_link')
# Contact bits: robot geoms never collide with robot geoms, but collide with
# everything in the world group (object, table, ground).
ROBOT_CONTYPE, ROBOT_CONAFFINITY = 2, 1
WORLD_CONTYPE, WORLD_CONAFFINITY = 1, 3
HAND_KP, HAND_KD = 1200., 25.


@dataclass
class ContactParams:
    """MuJoCo contact model knobs swept in the parity tests (no PhysX analogue)."""
    condim: int = 3
    solref: tuple = (0.02, 1.)
    solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.)
    friction: float = 0.5
    # PhysX averages the 1.5 fingertip-marker material with partners' 0.5; MuJoCo
    # takes the max, so 1.0 reproduces the effective coefficient.
    marker_friction: float = 1.0
    maxhullvert: int = 64


def prepared_urdf_text(path=ROBOT_URDF):
    tree = ET.parse(path)
    root = tree.getroot()
    for block in root.findall('mujoco'):
        root.remove(block)
    links = {l.get('name'): l for l in root.findall('link')}
    for name in REMOVED_LINKS:
        if name in links:
            root.remove(links[name])
    for joint in list(root.findall('joint')):
        child = joint.find('child').get('link')
        if child in REMOVED_LINKS:
            root.remove(joint)
            continue
        limit = joint.find('limit')
        if joint.get('type') == 'revolute' and limit is not None and float(limit.get('lower')) == float(limit.get('upper')):
            joint.set('type', 'fixed')
            joint.remove(limit)
            axis = joint.find('axis')
            if axis is not None:
                joint.remove(axis)
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename')
        if not filename.startswith('/'):
            mesh.set('filename', str((path.parent/filename).resolve()))
    compiler = ET.SubElement(ET.SubElement(root, 'mujoco'), 'compiler')
    compiler.set('discardvisual', 'true')
    compiler.set('fusestatic', 'false')
    compiler.set('strippath', 'false')
    compiler.set('balanceinertia', 'false')
    compiler.set('inertiafromgeom', 'false')
    return ET.tostring(root, encoding='unicode')


def urdf_joint_efforts(path=ROBOT_URDF):
    efforts = {}
    for joint in ET.parse(path).getroot().findall('joint'):
        limit = joint.find('limit')
        if limit is not None and limit.get('effort') is not None:
            efforts[joint.get('name')] = float(limit.get('effort'))
    return efforts


# PhysX joint drives are implicit in position AND velocity:
#   F = kp (q* - q_{n+1}) - kd v_{n+1},  q_{n+1} = q_n + dt v_{n+1}
#     = kp (q* - q_n) - (kd + kp dt) v_{n+1}.
# MuJoCo's implicitfast integrator treats actuator velocity terms implicitly
# (v_{n+1}) but position stiffness explicitly, which is unstable for the
# 1200 N m/rad finger drives on gram-scale links at dt = 1/120 s. Using
# kv = kd + kp dt reproduces the PhysX backward-Euler drive exactly.
IMPLICIT_DRIVE_DT = 1/120


def build_robot_spec(contact=ContactParams(), path=ROBOT_URDF, implicit_drive_dt=IMPLICIT_DRIVE_DT):
    from dextrah_lab.wholebody.actuators import body_motors
    spec = mujoco.MjSpec.from_string(prepared_urdf_text(path))
    spec.compiler.degree = False
    pelvis = spec.body('pelvis')
    pelvis.add_freejoint(name='floating_base_joint')
    for mesh in spec.meshes:
        mesh.maxhullvert = contact.maxhullvert
    for geom in spec.geoms:
        geom.contype, geom.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFFINITY
        geom.condim = contact.condim
        geom.solref = contact.solref
        geom.solimp = contact.solimp
        geom.margin = geom.gap = 0.
        marker = geom.parent.name.endswith('_tip') and geom.parent.name.startswith('right_')
        geom.friction = (contact.marker_friction if marker else contact.friction, 0.005, 0.0001)
    motors = body_motors()
    efforts = urdf_joint_efforts(path)
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint.damping = (0., 0., 0.)
        joint.frictionloss = 0.
        if joint.name in motors:
            m = motors[joint.name]
            kp, kd, armature, effort = m.stiffness, m.damping, m.armature, m.effort
        else:   # right and left Revo2 hands (actuated and distal follower joints)
            kp, kd, armature, effort = HAND_KP, HAND_KD, 0., efforts[joint.name]
        joint.armature = armature
        act = spec.add_actuator(name=joint.name, target=joint.name,
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kd+kp*implicit_drive_dt)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = (-effort, effort)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec
