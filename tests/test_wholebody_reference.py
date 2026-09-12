import numpy as np
import pytest
from dextrah_lab.wholebody.reference import Segment, successful_segments, retime_segment


def trace():
    t=np.arange(181)/60
    return dict(step=np.arange(len(t)),resets=np.zeros(len(t)),
        goal_hits=(t>=1).astype(int),joint_pos=np.c_[t,t**2],
        joint_vel=np.c_[np.ones(len(t)),2*t],commanded_joint_targets=np.c_[10+t,20+t],
        robot=np.c_[t,np.zeros((len(t),2)),np.ones(len(t)),np.zeros((len(t),3))])


def test_retime_achieved_motion_independently_of_held_targets():
    data=trace()
    segment=successful_segments(data)[0]
    result=retime_segment(data,segment)
    t=result['time_s']
    np.testing.assert_allclose(np.diff(t),.02)
    np.testing.assert_allclose(result['joint_pos'],np.c_[t,t**2],atol=1e-12)
    np.testing.assert_allclose(result['joint_vel'],np.c_[np.ones(len(t)),2*t],atol=1e-12)
    assert result['commanded_joint_targets'][0,0]==10
    assert result['commanded_joint_targets'][1,0]==data['commanded_joint_targets'][1,0]
    np.testing.assert_allclose(result['robot'][:,0],t)
    np.testing.assert_allclose(result['robot'][:,3:],np.tile([1,0,0,0],(len(t),1)))


def test_reset_boundary_not_interpolated_or_counted_as_success():
    data=trace()
    data['resets'][60:]=1
    # The only success is recorded together with reset at frame 60.
    assert successful_segments(data)==[]
    with pytest.raises(ValueError,match='cross a reset'):
        retime_segment(data,Segment(0,181,1))
    data['goal_hits'][130:]+=1
    assert successful_segments(data)==[Segment(60,181,1)]


def test_missing_samples_and_invalid_quaternions_rejected():
    data=trace()
    data['step'][60:]+=1
    with pytest.raises(ValueError,match='EVERY policy step'):
        successful_segments(data)
    data=trace()
    data['robot'][0,3]=2
    with pytest.raises(ValueError,match='Non-unit'):
        retime_segment(data,Segment(0,181,1))
