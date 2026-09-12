import numpy as np
import pytest
from dextrah_lab.wholebody.transfer_labels import sample_teacher_commands
from dextrah_lab.wholebody.teacher_bridge import RIGHT_ARM
from dextrah_lab.wholebody.contract import hand_joints


def fixture():
    names=list((*RIGHT_ARM,*hand_joints('right')))
    meta=dict(joint_names=names,action_joint_names=names,reference_dt=1/60)
    trace=dict(step=np.arange(10),resets=np.zeros(10,dtype=int),
        transition_valid=np.ones(10,dtype=bool),
        applied_joint_targets_after_step=np.repeat(np.arange(10)[:,None],13,axis=1),
        teacher_clipped_action=np.repeat(np.arange(10)[:,None]*.1,13,axis=1))
    reference=dict(time_s=np.arange(5)/50,source_sample=np.array([2,3,4,5,6]))
    return reference,trace,meta


def test_clock_conversion_never_uses_future_actions():
    reference,trace,meta=fixture()
    labels=sample_teacher_commands(reference,trace,meta,np.array([0.,.02,.033,.034,.08]))
    np.testing.assert_array_equal(labels['source_sample'],[2,3,3,4,6])
    np.testing.assert_array_equal(labels['right_arm_targets'][:,0],[2,3,3,4,6])
    np.testing.assert_allclose(labels['right_finger_actions'][:,0],[.2,.3,.3,.4,.6])
    assert labels['valid'].all()


def test_resets_and_endpoint_holds_are_not_training_labels():
    reference,trace,meta=fixture()
    trace['transition_valid'][3]=False
    labels=sample_teacher_commands(reference,trace,meta,np.array([-.01,.02,.081]))
    assert not labels['valid'].any()
    trace['resets'][5:]=1
    with pytest.raises(ValueError,match='crosses teacher reset'):
        sample_teacher_commands(reference,trace,meta,np.array([0.]))
