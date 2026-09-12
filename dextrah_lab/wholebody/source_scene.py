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
