"""Fail-closed checks for causal, checkpoint-provenanced imitation captures."""
import numpy as np
from .contract import hand_joints
from .teacher_bridge import RIGHT_ARM


def validate_policy_capture(trace, metadata, normalizer):
    if not metadata.get('completed') or not metadata.get('capture_distillation') or metadata.get('optimizer_updates') != 0:
        raise ValueError('Requires completed frozen teacher-policy capture')
    if metadata.get('action_joint_names') != list((*RIGHT_ARM, *hand_joints('right'))):
        raise ValueError('Teacher action convention mismatch')
    n = len(trace['step'])
    dim = metadata['actor_dim']
    required = {'teacher_observation': (n, dim+1), 'teacher_action': (n, 13),
        'teacher_clipped_action': (n, 13), 'transition_valid': (n,),
        'applied_joint_targets_after_step': (n, len(metadata['joint_names']))}
    for name, shape in required.items():
        value = np.asarray(trace[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid teacher field: {name}')
    if n < 2 or not (np.diff(trace['step'])==1).all():
        raise ValueError('Missing policy-rate samples')
    if not np.all(trace['teacher_observation'][:, -1] == 0):
        raise ValueError('Expected the SAPG zero-entropy leader coefficient')
    for key in ('mean', 'variance'):
        if normalizer[key].shape != (dim,) or not np.isfinite(normalizer[key]).all():
            raise ValueError('Invalid frozen teacher observation statistics')
    if (normalizer['variance'] < 0).any():
        raise ValueError('Negative teacher observation variance')
    clipped = np.clip(trace['teacher_action'], -metadata['action_clip'], metadata['action_clip'])
    if not np.array_equal(clipped, trace['teacher_clipped_action']):
        raise ValueError('Wrong teacher action clipping')
    valid = trace['transition_valid']
    if valid.dtype != np.bool_:
        raise ValueError('Transition validity must be boolean')
    reset_changed = np.diff(trace['resets']) != 0
    if not np.array_equal(~valid[:-1], reset_changed):
        raise ValueError('Reset/action timing mismatch')
    # Exact causal alignment: post-action applied commands become the next
    # recorded previous commands. Never use a post-reset target as supervision.
    error = np.abs(trace['applied_joint_targets_after_step'][:-1] - trace['commanded_joint_targets'][1:])
    if valid[:-1].any() and error[valid[:-1]].max() > 1e-6:
        raise ValueError('Applied targets are not aligned with the next pre-action state')
    recurrent = [key for key in trace if key.startswith('teacher_rnn_state_')]
    if not recurrent or any(trace[key].shape[0] != n or not np.isfinite(trace[key]).all() for key in recurrent):
        raise ValueError('Missing or invalid teacher recurrent-state capture')
    return dict(samples=n, valid_transitions=int(valid.sum()), actor_dim=dim,
        recurrent_fields=sorted(recurrent), causal_target_alignment=True,
        source_checkpoint_sha256=metadata['checkpoint_sha256'],
        fullbody_student_states_collected=False, adapter_trained=False)
