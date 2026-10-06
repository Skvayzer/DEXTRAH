"""Scoped robot-only extension of the pinned Play2Perfect scene builder.

Keep its URDF importer, hand target coupling, procedural objects, table,
materials, instancing and reset coordinates. Do not edit the external repo.
Isaac creates one scene per process; hooks are restored even on failure.
"""
from contextlib import contextmanager
from unittest.mock import patch
from .actuators import body_motors
from .contract import BODY_JOINTS, nominal_body_pose
from .teacher_bridge import RIGHT_ARM


def ground_contact_partners(stage):
    """Global static colliders aren't found by the source env-local body scan."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath('/World/ground')
    if not root.IsValid():
        raise RuntimeError('Source ground is missing from the whole-body scene')
    paths = [str(p.GetPath()) for p in Usd.PrimRange(root)
             if p.HasAPI(UsdPhysics.CollisionAPI)
             and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if not paths:
        raise RuntimeError('No enabled ground collision shapes for fingertip contact filtering')
    return paths


TABLE_CONTACT_RIGHT_ARM = ('right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link',
                           'right_wrist_yaw_link', 'right_base_link', 'right_base2_link')


def table_filtered_links(link_names):
    """Robot links that must not collide with the table (no body support).

    The right forearm, wrist, hand base and fingers keep table contact so
    grasping from and placing on the table are physically unchanged.
    """
    keep = set(TABLE_CONTACT_RIGHT_ARM)
    fingers = ('right_index', 'right_middle', 'right_ring', 'right_pinky', 'right_thumb')
    return [n for n in link_names if n not in keep and not n.startswith(fingers)]


def table_contact_monitored_links(link_names):
    """Body links whose table contact ends an episode (left-hand fingers skipped: they
    sit far from the table and each sensor costs a PhysX view)."""
    fingers = ('index', 'middle', 'ring', 'pinky', 'thumb')
    return [n for n in table_filtered_links(link_names) if not any(f in n for f in fingers)]


def prepare_table_contact_reporting(stage, env_path='/World/envs/env_0'):
    """Enable contact reporting on monitored links of the source env before cloning."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    tables = [p for p in Usd.PrimRange(stage.GetPrimAtPath(env_path+'/Table')) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    links = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(env_path+'/Robot'))
             if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if len(tables) != 1:
        raise RuntimeError('Expected exactly one table rigid body')
    names = table_contact_monitored_links(sorted(links))
    for name in names:
        PhysxSchema.PhysxContactReportAPI.Apply(links[name]).CreateThresholdAttr().Set(0.)
    return str(tables[0].GetPath()), {n: str(links[n].GetPath()) for n in names}


def filter_table_body_contacts(stage, env_path='/World/envs/env_0'):
    """Author PhysX filtered pairs on the source env's table before cloning."""
    from pxr import Usd, UsdPhysics
    tables = [p for p in Usd.PrimRange(stage.GetPrimAtPath(env_path+'/Table')) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    links = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(env_path+'/Robot'))
             if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if len(tables) != 1 or not links:
        raise RuntimeError(f'Expected one table rigid body and robot links, found {len(tables)} / {len(links)}')
    names = table_filtered_links(sorted(links))
    rel = UsdPhysics.FilteredPairsAPI.Apply(tables[0]).CreateFilteredPairsRel()
    for name in names:
        rel.AddTarget(links[name].GetPath())
    return str(tables[0].GetPath()), names, sorted(set(links)-set(names))


def standing_reset_pose():
    pose = dict(zip(BODY_JOINTS, map(float, nominal_body_pose())))
    pose.update(dict.fromkeys(RIGHT_ARM, 0.))  # original manipulation reset
    pose['left_shoulder_roll_joint'] = .6  # new left arm must clear the table
    return pose


@contextmanager
def floating_sonic_robot_scene():
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaacsimenvs.tasks.play.utils import scene_utils as source
    original_bake = source._bake_usd
    original_cfg = source.build_robot_articulation_usd_cfg

    def bake(raw_usd_path, bake_root, baked_subdir, **kwargs):
        if baked_subdir == 'robot':
            kwargs['props'] = dict(kwargs.get('props', {}), disable_gravity=False)
        return original_bake(raw_usd_path, bake_root, baked_subdir, **kwargs)

    def robot_cfg(usd_path, assets_cfg=None):
        cfg = original_cfg(usd_path, assets_cfg)
        motors = body_motors()
        # Right hand retains exactly the original implicit motor drives,
        # including PD-driven distal followers (not native mimic constraints).
        right_hand = cfg.actuators['hand']
        left_names = list(source.G1_BRAINCO_LEFT_HAND_HOLD_JOINT_NAMES)
        cfg.actuators = dict(
            body=ImplicitActuatorCfg(joint_names_expr=list(motors),
                effort_limit_sim={n: m.effort for n, m in motors.items()},
                velocity_limit_sim={n: m.velocity for n, m in motors.items()},
                stiffness={n: m.stiffness for n, m in motors.items()},
                damping={n: m.damping for n, m in motors.items()},
                armature={n: m.armature for n, m in motors.items()}),
            hand=right_hand,
            left_hand=ImplicitActuatorCfg(joint_names_expr=left_names, stiffness=1200., damping=25.))
        cfg.init_state.joint_pos.update(standing_reset_pose())
        return cfg

    with patch.object(source, '_bake_usd', bake), patch.object(source, 'build_robot_articulation_usd_cfg', robot_cfg):
        yield
