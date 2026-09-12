import numpy as np
from dextrah_lab.wholebody.actuators import body_motors
from dextrah_lab.wholebody.contract import BODY_JOINTS


def test_checkpoint_body_gains_scales_and_ownership():
    motors = body_motors()
    assert tuple(motors) == BODY_JOINTS
    assert len(motors) == 29
    # Numeric reference values independently checked against pinned g1.py.
    np.testing.assert_allclose(motors['left_hip_pitch_joint'].stiffness, 99.0984277767)
    np.testing.assert_allclose(motors['right_wrist_yaw_joint'].stiffness, 16.7783274809)
    assert motors['right_wrist_yaw_joint'].effort == 5
    assert motors['right_shoulder_pitch_joint'].effort == 25
    assert motors['waist_pitch_joint'].armature == 2 * .003609725
    for name, motor in motors.items():
        assert motor.stiffness > 0 and motor.damping > 0
        np.testing.assert_allclose(motor.stiffness * motor.action_scale, .25 * motor.effort)
        if name.startswith('left_'):
            assert motor == motors[name.replace('left_', 'right_', 1)]
