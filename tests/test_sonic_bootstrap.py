import numpy as np
import pytest
from dextrah_lab.wholebody.bootstrap import causal_episode,split_episode,term_history,standing_rehearsal
from dextrah_lab.wholebody.contract import BODY_JOINTS,nominal_body_pose
from dextrah_lab.wholebody.teacher_bridge import RIGHT_ARM


def test_causal_hold_and_applied_target_not_previous_or_future():
    n=120
    names=list(reversed(RIGHT_ARM))
    q=np.broadcast_to(np.arange(n)[:,None],(n,7)).copy().astype(float)
    trace=dict(step=np.arange(n)+123,resets=np.zeros(n),joint_pos=q,joint_vel=q*0,
        commanded_joint_targets=q+.1,applied_joint_targets_after_step=q+.2,
        teacher_observation=np.tile(np.arange(n)[:,None],(1,225)),
        teacher_clipped_action=np.zeros((n,13)),transition_valid=np.ones(n,dtype=bool))
    meta=dict(reference_dt=1/60,actor_dim=224,joint_names=names)
    result=causal_episode(trace,meta,0,n)
    ids=result['source_sample']
    assert (ids/60<=result['time_s']+1e-12).all()
    assert np.array_equal(result['task'][:,0],ids)
    assert np.allclose(result['arm_targets'][:,0],ids+.2)
    assert result['proprio'].shape==(len(ids),930)
    rest=nominal_body_pose();left=BODY_JOINTS.index('left_shoulder_roll_joint');rest[left]=.6
    clearance=causal_episode(trace,meta,0,n,rest_pose=rest)
    np.testing.assert_allclose(clearance['kinematic_body_q'][:,left],.6)
    np.testing.assert_array_equal(clearance['arm_targets'],result['arm_targets'])
    assert (clearance['previous_body_action'][:,left]>0).all()
    trace['resets'][60:]=1
    with pytest.raises(ValueError,match='reset'):
        causal_episode(trace,meta,0,n)


def test_temporal_split_and_per_term_history():
    train,val=split_episode(500)
    assert train.stop+50==val.start
    assert split_episode(50) is None
    terms=[np.tile(np.arange(20)[:,None],(1,d)) for d in (3,29,29,29,3)]
    history=term_history(*terms)
    assert np.array_equal(history[10,:30].reshape(10,3)[:,0],np.arange(1,11))
    assert not history[0].any()


def test_standing_labels_are_next_actions_and_reject_tracking_sources():
    n=25
    root=np.zeros((n,1,13)); root[:,:,3]=1
    trace=dict(root_state=root,joint_pos=np.tile(nominal_body_pose(),(n,1,1)),
        joint_vel=np.zeros((n,1,29)),normalized_action=np.broadcast_to(np.arange(n)[:,None,None],(n,1,29)))
    report=dict(joint_names=list(BODY_JOINTS),standing_probe_passed=True,native_coupling_passed=True,
        controller='sonic',controller_hz=50)
    data=standing_rehearsal(trace,report)
    assert np.array_equal(data['sonic_action'][:,0],data['post_step_index']+1)
    assert np.array_equal(data['proprio'][:,-30:].reshape(-1,10,3)[0],np.tile([0,0,-1],(10,1)))
    report['reference_type']='teacher motion'
    with pytest.raises(ValueError,match='nominal'):
        standing_rehearsal(trace,report)
