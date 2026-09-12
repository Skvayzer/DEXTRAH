import numpy as np
import pytest
from dextrah_lab.wholebody.teacher_bridge import RIGHT_ARM
from dextrah_lab.wholebody.contract import hand_joints
from dextrah_lab.wholebody.teacher_data import validate_policy_capture


def fixture():
    commands=np.arange(3*18).reshape(3,18).astype(float)
    actions=np.zeros((3,13))
    actions[:,0]=2.
    data=dict(step=np.arange(3),resets=np.array([0,0,1]),
        teacher_observation=np.zeros((3,225)),teacher_action=actions,
        teacher_clipped_action=np.clip(actions,-1,1),transition_valid=np.array([True,False,True]),
        applied_joint_targets_after_step=np.roll(commands,-1,axis=0),commanded_joint_targets=commands,
        teacher_rnn_state_0=np.zeros((3,1,16)))
    meta=dict(completed=True,capture_distillation=True,optimizer_updates=0,actor_dim=224,
        action_joint_names=list((*RIGHT_ARM,*hand_joints('right'))),joint_names=list(range(18)),
        action_clip=1.,checkpoint_sha256='test-only')
    return data,meta,dict(mean=np.zeros(224),variance=np.ones(224))


def test_capture_requires_causal_action_state_and_normalizer_alignment():
    data,meta,normalizer=fixture()
    report=validate_policy_capture(data,meta,normalizer)
    assert report['valid_transitions']==2 and report['causal_target_alignment']
    assert not report['adapter_trained']
    data['applied_joint_targets_after_step'][0,0]+=1
    with pytest.raises(ValueError,match='not aligned'):
        validate_policy_capture(data,meta,normalizer)


def test_reset_and_action_order_mismatches_fail():
    data,meta,normalizer=fixture()
    data['transition_valid'][1]=True
    with pytest.raises(ValueError,match='Reset/action timing'):
        validate_policy_capture(data,meta,normalizer)
    data,meta,normalizer=fixture()
    meta['action_joint_names'].reverse()
    with pytest.raises(ValueError,match='action convention'):
        validate_policy_capture(data,meta,normalizer)
