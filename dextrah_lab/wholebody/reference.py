"""Offline teacher-motion preparation, NOT a dynamically feasible body reference.

Preserve reset boundaries and distinguish achieved motion from motor targets.
Exported motions still need world-frame alignment, whole-body IK and physical
validation before they may condition SONIC in a manipulation task.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int  # exclusive; never includes the first post-reset sample
    goal_hits: int


def successful_segments(trace, min_seconds=1., dt=1/60):
    steps = np.asarray(trace['step'])
    resets = np.asarray(trace['resets'])
    hits = np.asarray(trace['goal_hits'])
    if dt <= 0 or min_seconds <= 0 or steps.ndim != 1 or len(steps)<2:
        raise ValueError('Invalid trace timing')
    for name,array in (('steps',steps),('resets',resets),('hits',hits)):
        if array.shape != steps.shape or not np.isfinite(array).all() or (np.diff(array)<0).any():
            raise ValueError(f'Invalid cumulative trace field: {name}')
    if not (np.diff(steps)==1).all():
        raise ValueError('Reference capture must contain EVERY policy step')
    boundaries=np.r_[0,np.flatnonzero(np.diff(resets)!=0)+1,len(steps)]
    segments=[]
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        # A hit coincident with a reset cannot be attributed to an achieved
        # terminal pose: the recorded sample is already post-reset. Exclude it.
        count=int(hits[stop-1]-hits[start])
        if count>0 and (stop-start-1)*dt>=min_seconds:
            segments.append(Segment(int(start),int(stop),count))
    return segments


def _retime_pose(times, poses, query):
    poses=np.asarray(poses)
    if poses.shape != (len(times),7) or not np.isfinite(poses).all():
        raise ValueError('Expected finite xyz + wxyz poses')
    quat=poses[:,3:]
    if not np.allclose(np.linalg.norm(quat,axis=-1),1.,atol=1e-3):
        raise ValueError('Non-unit reference quaternion')
    rotation=Rotation.from_quat(quat[:,[1,2,3,0]])
    xyzw=Slerp(times,rotation)(query).as_quat()
    position=np.stack([np.interp(query,times,poses[:,axis]) for axis in range(3)],-1)
    return np.c_[position,xyzw[:,[3,0,1,2]]]


def retime_segment(trace, segment, source_dt=1/60, target_dt=1/50):
    """Smooth achieved q/qd; zero-order-hold motor targets; SLERP poses.

    Cubic interpolation can overshoot joint limits; intentionally do not hide
    that by clipping. The subsequent feasibility stage must check limits.
    No extrapolation, stitching across resets or treating targets as achieved q.
    """
    if source_dt<=0 or target_dt<=0:
        raise ValueError('Invalid sample period')
    a,b=segment.start,segment.stop
    length=len(trace['step'])
    if not 0<=a<b<=length or b-a<2:
        raise ValueError('Invalid segment bounds')
    if np.ptp(np.asarray(trace['resets'])[a:b])!=0:
        raise ValueError('A reference must not cross a reset')
    steps=np.asarray(trace['step'])[a:b]
    if not (np.diff(steps)==1).all():
        raise ValueError('Missing source samples')
    source=(steps-steps[0])*source_dt
    time=np.arange(int(np.floor((source[-1]+1e-10)/target_dt))+1)*target_dt
    time=np.minimum(time,source[-1])
    q,qd,targets=(np.asarray(trace[key])[a:b] for key in
        ('joint_pos','joint_vel','commanded_joint_targets'))
    if q.ndim!=2 or q.shape!=qd.shape or q.shape!=targets.shape:
        raise ValueError('Joint trace shape mismatch')
    if not all(np.isfinite(value).all() for value in (q,qd,targets)):
        raise ValueError('Nonfinite teacher joint data')
    spline=CubicHermiteSpline(source,q,qd,axis=0,extrapolate=False)
    held=np.searchsorted(source,time+1e-12,side='right')-1
    result=dict(time_s=time,joint_pos=spline(time),joint_vel=spline(time,1),
        commanded_joint_targets=targets[held],source_sample=held+a)
    # Goal can jump at a hit; hold its label instead of interpolating between
    # unrelated goals. Robot/object/table are actual physical poses.
    for key in ('robot','object','table'):
        if key in trace:
            result[key]=_retime_pose(source,np.asarray(trace[key])[a:b],time)
    if 'goal' in trace:
        result['goal']=np.asarray(trace['goal'])[a:b][held]
    return result
