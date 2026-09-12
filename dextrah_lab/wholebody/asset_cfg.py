"""Floating-base G1 + BOTH Revo2 hands; import only after AppLauncher."""
from .actuators import body_motors
from pathlib import Path
from .contract import BODY_JOINTS, hand_joints, nominal_body_pose
from .importer import native_mimic_cfg_flag


def fullbody_robot_cfg(urdf, usd_dir, *, hand_stiffness=1200., hand_damping=25.,
                       self_collision=True, position_iterations=8):
    import isaaclab.sim as sim
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    motors = body_motors()
    hands = [*hand_joints('left'), *hand_joints('right')]
    distal = [f'{side}_{finger}_distal_joint' for side in ('left','right')
              for finger in ('thumb','index','middle','ring','pinky')]
    return ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim.UrdfFileCfg(
            asset_path=str(Path(urdf).resolve()), usd_dir=str(Path(usd_dir).resolve()), usd_file_name='g1_revo2.usd',
            fix_base=False, merge_fixed_joints=True,
            convert_mimic_joints_to_normal_joints=native_mimic_cfg_flag(),
            self_collision=self_collision, replace_cylinders_with_capsules=True,
            activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,
                retain_accelerations=False, linear_damping=0., angular_damping=0.,
                max_linear_velocity=1000., max_angular_velocity=1000.,
                max_depenetration_velocity=1.),
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collision, solver_position_iteration_count=position_iterations,
                solver_velocity_iteration_count=4),
            joint_drive=sim.UrdfConverterCfg.JointDriveCfg(
                # Do NOT overwrite imported mimic followers with position drives.
                target_type={f'^{name}$':'position' for name in (*BODY_JOINTS,*hands)},
                gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.76),
            joint_pos={**dict(zip(BODY_JOINTS,map(float,nominal_body_pose()))),
                       **dict.fromkeys(hands+distal,0.)}, joint_vel={'.*':0.}),
        soft_joint_pos_limit_factor=.9,
        actuators={
            'body': ImplicitActuatorCfg(joint_names_expr=list(motors),
                effort_limit_sim={n:m.effort for n,m in motors.items()},
                velocity_limit_sim={n:m.velocity for n,m in motors.items()},
                stiffness={n:m.stiffness for n,m in motors.items()},
                damping={n:m.damping for n,m in motors.items()},
                armature={n:m.armature for n,m in motors.items()}),
            # Keep the teacher's hand PD gains and URDF motor limits for this
            # first probe. Unlike that teacher, distal joints are PASSIVE and
            # follow native mimic constraints. Validate this transfer gap.
            'hands': ImplicitActuatorCfg(joint_names_expr=hands, stiffness=hand_stiffness, damping=hand_damping),
            'coupled_distal': ImplicitActuatorCfg(joint_names_expr=distal, stiffness=0., damping=0.),
        })
