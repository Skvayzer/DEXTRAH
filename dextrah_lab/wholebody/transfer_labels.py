"""Causal 60 Hz teacher commands sampled onto a planned 50 Hz reference.

These are action-fitting targets, NOT full-body student observations. A frozen
teacher observation from a different simulation must never masquerade as the
current student's object/tactile state.
"""
import numpy as np
from .contract import hand_joints, joint_indices
from .teacher_bridge import RIGHT_ARM


def sample_teacher_commands(reference, trace, metadata, query_s):
    query=np.asarray(query_s,dtype=float)
    times=np.asarray(reference['time_s'])
    source=np.asarray(reference['source_sample'])
    if query.ndim!=1 or not np.isfinite(query).all() or len(times)<2 or not (np.diff(times)>0).all():
        raise ValueError('Invalid reference times')
    if source.shape!=times.shape or not np.issubdtype(source.dtype,np.integer) or (np.diff(source)<0).any():
        raise ValueError('Invalid source-frame mapping')
    # Use original policy clock rather than nearest 50 Hz frame: choosing the
    # next resampled frame would leak a future teacher action at rate crossings.
    origin=int(source[0])
    steps=np.asarray(trace['step'])
    source_times=(steps-steps[origin])*metadata['reference_dt']
    indices=np.searchsorted(source_times,query+1e-12,side='right')-1
    in_range=(query>=0.)&(query<=times[-1])
    indices=np.clip(indices,origin,int(source[-1]))
    resets=np.asarray(trace['resets'])
    if np.ptp(resets[origin:int(source[-1])+1])!=0:
        raise ValueError('Reference crosses teacher reset')
    valid=in_range & np.asarray(trace['transition_valid'])[indices]
    ids=joint_indices(metadata['joint_names'],RIGHT_ARM)
    applied=np.asarray(trace['applied_joint_targets_after_step'])
    actions=np.asarray(trace['teacher_clipped_action'])
    if metadata['action_joint_names']!=list((*RIGHT_ARM,*hand_joints('right'))):
        raise ValueError('Unexpected teacher action order')
    return dict(source_sample=indices,valid=valid,
        right_arm_targets=applied[indices][:,ids],right_finger_actions=actions[indices,7:13])
