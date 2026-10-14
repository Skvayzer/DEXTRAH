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
    # Revo2 finger joints (PhysX armature 0). MuJoCo solves contacts with the
    # explicit actuator force on the 9 g finger links; with zero armature an
    # open-commanded finger overlapping an object at reset pushed it with
    # 60-160 N for a whole 1/120 s step (launches at 10-19 m/s; replayed in
    # single-world mujoco_warp, gone with the servo off). PhysX solves the
    # implicit drive together with the contacts. Armature gives the contact
    # solve a finger inertia; the servo time constant armature/kv (<= 1 ms)
    # stays far below the 8.3 ms step, so free finger motion is unchanged.
    # Sweep (job 1143, spawn fix, 1200 envs x 40 s passive): spawn-window peak
    # object speed > 5 m/s: armature 0 8.7%, 1e-4 0.17%, 1e-3 0% (Isaac 0.77%);
    # zero-shot 1105 policy hand_far 61% / 15% / 0.5% (Isaac training 3%).
    # Free finger motion changes by <= 0.07 rad (transient).
    hand_armature: float = 1e-3
    # Diagnostics only: robot geoms collide with nothing but the ground.
    robot_ground_only: bool = False


def prepared_urdf_text(path=ROBOT_URDF, remove_links=REMOVED_LINKS, remove_subtrees=(), fix_joints=()):
    """remove_subtrees: links removed with all descendants (Play2Perfect
    `remove_link_subtrees`); fix_joints: joints made fixed at angle 0
    (`fix_joint_names` with no positions), as the right-arm training profile."""
    tree = ET.parse(path)
    root = tree.getroot()
    for block in root.findall('mujoco'):
        root.remove(block)
    removed = set(remove_links)
    children = {}
    for joint in root.findall('joint'):
        children.setdefault(joint.find('parent').get('link'), []).append(joint.find('child').get('link'))
    stack = list(remove_subtrees)
    while stack:
        link = stack.pop()
        removed.add(link)
        stack.extend(children.get(link, ()))
    links = {l.get('name'): l for l in root.findall('link')}
    for name in removed:
        if name in links:
            root.remove(links[name])
    for joint in list(root.findall('joint')):
        child = joint.find('child').get('link')
        if child in removed:
            root.remove(joint)
            continue
        limit = joint.find('limit')
        if joint.get('name') in fix_joints:
            joint.set('type', 'fixed')
            for tag in ('limit', 'axis', 'dynamics'):
                el = joint.find(tag)
                if el is not None:
                    joint.remove(el)
            continue
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


def hand_servo(name, path=ROBOT_URDF):
    """PhysX Revo2 finger drive (kp 1200, kd 25, maxForce = URDF effort,
    maxJointVelocity = URDF velocity) as an equivalent MuJoCo velocity servo.

    effort/kp is 0.4-1.7 mrad, so in PhysX these drives are saturated almost
    always: fingers move at the velocity cap and push with at most `effort`.
    A clamped MuJoCo position actuator loses its implicit damping when it
    saturates and gram-scale finger links then diverge (measured: >1e6 rad/s).
    The servo  tau = kv (v_cmd - qd),  kv = effort / vmax,
               v_cmd = clamp(kp (q* - q) / kv, -vmax, vmax)
    gives a blocked joint tau = min(kp |q* - q|, effort) (the PhysX static law)
    and free motion at |qd| <= vmax, with no force clamp (implicit, stable).
    """
    effort = urdf_joint_efforts(path)[name]
    vmax = {j.get('name'): float(j.find('limit').get('velocity')) for j in ET.parse(path).getroot().findall('joint')
            if j.find('limit') is not None}[name]
    return dict(kp=HAND_KP, effort=effort, vmax=vmax, kv=effort/vmax)


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
        if contact.robot_ground_only:
            geom.contype, geom.conaffinity = 4, 0
        geom.condim = contact.condim
        geom.solref = contact.solref
        geom.solimp = contact.solimp
        geom.margin = geom.gap = 0.
        marker = geom.parent.name.endswith('_tip') and geom.parent.name.startswith('right_')
        geom.friction = (contact.marker_friction if marker else contact.friction, 0.005, 0.0001)
    motors = body_motors()
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint.damping = (0., 0., 0.)
        joint.frictionloss = 0.
        act = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        if joint.name in motors:
            m = motors[joint.name]
            joint.armature = m.armature
            act.set_to_position(kp=m.stiffness, kv=m.damping+m.stiffness*implicit_drive_dt)
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = (-m.effort, m.effort)
        else:   # Revo2 hands: velocity servo, see hand_servo()
            joint.armature = contact.hand_armature
            act.set_to_velocity(kv=hand_servo(joint.name, path)['kv'])
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


def build_fixed_robot_spec(reference, contact=ContactParams(), path=ROBOT_URDF, implicit_drive_dt=IMPLICIT_DRIVE_DT):
    """The fixed-body (right-arm training) G1 + Revo2 of the Play2Perfect touch task.

    As Isaac's `g1_brainco_right` profile: sensor/logo/head links removed, the
    leg and left-arm subtrees removed, waist joints fixed, base welded at the
    robot init pose (entity init_state), robot gravity disabled (PhysX
    disable_gravity -> MuJoCo gravcomp 1). Every joint's drive comes from the
    Isaac dump `reference` (stiffness, damping, effort, armature): arm joints
    are implicit PD position actuators (kv = kd + kp dt, as the whole-body
    port), Revo2 joints the same velocity servos as the whole-body port.
    """
    from .compat import p2p
    su = p2p('scene_utils')
    spec = mujoco.MjSpec.from_string(prepared_urdf_text(
        path, remove_links=su.G1_RIGHT_TRAINING_FIXED_LINKS_TO_REMOVE,
        remove_subtrees=su.G1_RIGHT_TRAINING_BRANCH_ROOT_LINKS_TO_REMOVE,
        fix_joints=su.G1_RIGHT_TRAINING_JOINTS_TO_FIX))
    spec.compiler.degree = False
    for body in spec.bodies:
        body.gravcomp = 1.
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
    index = {n: i for i, n in enumerate(reference['joint_names'])}
    hand = set(su.G1_BRAINCO_HAND_TARGET_JOINT_NAMES)
    for joint in spec.joints:
        if joint.name not in index:
            raise RuntimeError(f'Joint {joint.name} not in the Isaac fixed-body reference')
        i = index[joint.name]
        joint.damping = (0., 0., 0.)
        joint.frictionloss = 0.
        act = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        if joint.name in hand:
            joint.armature = contact.hand_armature
            act.set_to_velocity(kv=hand_servo(joint.name, path)['kv'])
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
        else:
            kp, kd = float(reference['joint_stiffness'][i]), float(reference['joint_damping'][i])
            joint.armature = float(reference['joint_armature'][i])
            act.set_to_position(kp=kp, kv=kd+kp*implicit_drive_dt)
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            effort = float(reference['joint_effort_limits'][i])
            act.forcerange = (-effort, effort)
    missing = set(reference['joint_names']) - {j.name for j in spec.joints}
    if missing:
        raise RuntimeError(f'Isaac fixed-body joints missing in MuJoCo: {sorted(missing)}')
    return spec
