"""SONIC model-12 body actuator contract, independent of Isaac imports.

Parameter source: pinned NVlabs/GRAIL, imports/SONIC/gear_sonic/envs/
manager_env/robots/g1.py, G1_CYLINDER_MODEL_12_DEX_CFG and
G1_MODEL_12_ACTION_SCALE. Do not substitute the fixed-base teacher's arm PD.
"""
from dataclasses import dataclass, asdict
from .contract import BODY_JOINTS


@dataclass(frozen=True)
class Motor:
    armature: float
    effort: float
    velocity: float

    @property
    def stiffness(self):
        return self.armature * (10 * 2 * 3.1415926535)**2

    @property
    def damping(self):
        return 4 * self.armature * (10 * 2 * 3.1415926535)

    @property
    def action_scale(self):
        return .25 * self.effort / self.stiffness


def body_motors():
    result = {}
    for name in BODY_JOINTS:
        if any(part in name for part in ('hip_pitch', 'hip_roll', 'knee')):
            motor = Motor(.025101925, 139., 20.)
        elif 'hip_yaw' in name or name == 'waist_yaw_joint':
            motor = Motor(.010177520, 88., 32.)
        elif 'ankle_' in name or name in ('waist_pitch_joint', 'waist_roll_joint'):
            motor = Motor(2 * .003609725, 50., 37.)
        elif 'wrist_pitch' in name or 'wrist_yaw' in name:
            motor = Motor(.00425, 5., 22.)
        else:
            motor = Motor(.003609725, 25., 37.)
        result[name] = motor
    return result


def actuator_manifest():
    return {name: dict(**asdict(motor), stiffness=motor.stiffness,
            damping=motor.damping, action_scale=motor.action_scale)
            for name, motor in body_motors().items()}
