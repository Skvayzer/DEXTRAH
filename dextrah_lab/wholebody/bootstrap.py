"""Causal kinematic SAPG bootstrap and SEPARATE real standing rehearsal.

The lifted demonstrations are not full-body simulator observations. They put
recorded right-arm motion on nominal legs; object/task data remain in the
source teacher frame. This permits supervised initialization, never a claim
of dynamically valid manipulation or successful on-policy distillation.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .actuators import body_motors
from .contract import BODY_JOINTS, CONTROL_DT, joint_indices, nominal_body_pose
from .teacher_bridge import RIGHT_ARM


def term_history(*terms):
    """Per-term ten-sample oldest-to-newest history, endpoint held at start."""
    if len(terms)!=5 or any(x.ndim!=2 or len(x)!=len(terms[0]) for x in terms):
        raise ValueError('Expected five matching time-major proprioception terms')
    if [x.shape[1] for x in terms]!=[3,29,29,29,3]:
        raise ValueError('Wrong SONIC history term dimensions')
    ids=np.maximum(np.arange(len(terms[0]))[:,None]-np.arange(9,-1,-1),0)
    return np.concatenate([x[ids].reshape(len(ids),-1) for x in terms],-1).astype(np.float32)


def causal_episode(trace, metadata, start, stop, rest_pose=None):
    """Hold the latest complete 60 Hz sample onto a 50 Hz bootstrap clock.

    Do not smooth object observations, interpolate through goal changes or
    consult a future teacher frame. The student supplies its own recurrence.
    The old delayed/filtered applied target is the imitation label, not the
    raw delta action and not the previous target.
    """
    dt=metadata['reference_dt']
    if not np.isclose(dt,1/60) or not 0<=start<stop<=len(trace['step']) or stop-start<2:
        raise ValueError('Invalid source clock/episode bounds')
    if np.ptp(trace['resets'][start:stop])!=0:
        raise ValueError('Do not cross a source reset')
    source_times=(trace['step'][start:stop]-trace['step'][start])*dt
    if not (np.diff(source_times)>0).all():
        raise ValueError('Source time must increase')
    query=np.arange(int(np.floor((source_times[-1]+1e-10)/CONTROL_DT))+1)*CONTROL_DT
    ids=np.searchsorted(source_times,query+1e-12,side='right')-1+start
    arm=joint_indices(metadata['joint_names'],RIGHT_ARM)
    body_arm=joint_indices(BODY_JOINTS,RIGHT_ARM)
    nominal=nominal_body_pose()
    rest=nominal if rest_pose is None else np.asarray(rest_pose,dtype=np.float32)
    if rest.shape!=(29,) or not np.isfinite(rest).all():
        raise ValueError('Invalid standing rest pose')
    scales=np.asarray([m.action_scale for m in body_motors().values()])
    q=np.tile(rest,(len(ids),1))
    qd=np.zeros_like(q)
    q[:,body_arm]=trace['joint_pos'][ids][:,arm]
    qd[:,body_arm]=trace['joint_vel'][ids][:,arm]
    previous=np.tile(rest,(len(ids),1))
    previous[:,body_arm]=trace['commanded_joint_targets'][ids][:,arm]
    last=(previous-nominal)/scales
    gravity=np.tile([0.,0.,-1.],(len(ids),1))
    proprio=term_history(np.zeros((len(ids),3)),q-nominal,qd,last,gravity)
    result=dict(proprio=proprio,task=trace['teacher_observation'][ids,:metadata['actor_dim']].astype(np.float32),
        arm_targets=trace['applied_joint_targets_after_step'][ids][:,arm].astype(np.float32),
        finger_actions=trace['teacher_clipped_action'][ids,7:13].astype(np.float32),
        valid=trace['transition_valid'][ids].copy(),source_sample=ids,time_s=query,
        kinematic_body_q=q,previous_body_action=last.astype(np.float32))
    # These are usable only by an EXACT copy of the source LSTM, never by
    # the new GRU architecture. They initialize windows, not each future step.
    for k in ('teacher_rnn_state_0','teacher_rnn_state_1'):
        if k in trace:
            result[k]=trace[k][ids].astype(np.float32)
    return result


def split_episode(length, train_fraction=.7, gap_steps=50, min_steps=64):
    """Contiguous temporal holdout, with no overlapping history across split."""
    if not 0<train_fraction<1 or gap_steps<10 or min_steps<10:
        raise ValueError('Invalid temporal split')
    cut=int(length*train_fraction)
    if cut<min_steps or length-cut-gap_steps<min_steps:
        return None
    return slice(0,cut),slice(cut+gap_steps,length)


def standing_rehearsal(trace, report):
    """Reconstruct next pre-action histories from measured POST-step traces.

    Only nominal-reference standing probes are supported. Env zero is used
    because identical unperturbed copies must not multiply evidence. Returned
    labels are the NEXT recorded SONIC action, never the action that produced
    the current state. Check them against a fresh frozen SONIC forward.
    """
    if (not report.get('standing_probe_passed') or not report.get('native_coupling_passed')
            or report.get('reference_type') or report.get('controller')!='sonic'
            or report.get('controller_hz')!=50):
        raise ValueError('Require a passing nominal SONIC standing probe')
    ids=joint_indices(report['joint_names'],BODY_JOINTS)
    root=np.asarray(trace['root_state'])[:,0]
    q=np.asarray(trace['joint_pos'])[:,0][:,ids]
    qd=np.asarray(trace['joint_vel'])[:,0][:,ids]
    last=np.asarray(trace['normalized_action'])[:,0]
    rotation=Rotation.from_quat(root[:,[4,5,6,3]])
    omega=rotation.inv().apply(root[:,10:13])
    gravity=rotation.inv().apply(np.tile([0.,0.,-1.],(len(root),1)))
    proprio=term_history(omega,q-nominal_body_pose(),qd,last,gravity)
    # Current yaw-relative reference heading, exactly as the live SONIC probe.
    yaw=rotation.as_euler('xyz')[:,2]
    ori=Rotation.from_euler('z',-yaw[:,None]).as_matrix()[:,:,:2].reshape(-1,6)
    # The first nine rows would depend on artificial endpoint history filling.
    keep=slice(9,len(root)-1)
    return dict(proprio=proprio[keep],reference_ori6=ori[keep].astype(np.float32),
        sonic_action=last[10:].astype(np.float32),post_step_index=np.arange(len(root))[keep])
