"""The completed 70 Hz SAPG run is the manipulation task specification.

No Isaac/torch imports: check this BEFORE constructing a simulator. Whole-body
control changes physics and the action interface, not the manipulation reward.
"""
from collections.abc import Mapping
import math

TOUCH_RUN = '/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH/outputs/0_bps128_touch70_continuous_383'
TOUCH_CHECKPOINT = TOUCH_RUN + '/nn/complete_17364025344.pth'
TOUCH_SHA256 = 'ed9b26a7ba335c25f15df6623c8e0a97f6d5b3285721ae7f8263f82d6227d2cd'
BANK_SHA256 = '69a73d99a05749459fd4339f9341d3cce142ece0f7965283c04cb6410a2f3328'
P2P_REVISION = '70e79b5e53f912ef04af294ff8f61ac1c7f42160'
TASK_ACTOR_DIM, TASK_CRITIC_DIM = 249, 271
BODY_HISTORY_DIM, REFERENCE_DIM = 930, 64
BODY_EXTRA_DIM = BODY_HISTORY_DIM + REFERENCE_DIM
POLICY_DT = 1 / 60


def canonical(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def verify_task_contract(saved, cfg):
    """Fail closed on accidental task changes; enumerate intentional deltas."""
    actual = canonical(cfg)
    saved = canonical(saved)
    for section in ('action', 'reward', 'termination', 'reset', 'domain_randomization', 'obs', 'touch'):
        if actual[section] != saved[section]:
            raise ValueError(f'SAPG manipulation contract changed: {section}')
    for key in ('handle_head_types', 'num_assets_per_type', 'shuffle_assets',
                'robot_init_pos', 'robot_init_rot', 'palm_center_offset', 'fingertip_offset',
                'robot_merge_fixed_joints', 'robot_hand_stiffness', 'robot_hand_damping',
                'table_urdf', 'table_scale_range_x', 'table_scale_range_y', 'table_scale_num_variants'):
        if actual['assets'].get(key) != saved['assets'].get(key):
            raise ValueError(f'SAPG scene contract changed: assets.{key}')
    if actual['seed'] != saved['seed'] or actual['fabric'] != saved['fabric']:
        raise ValueError('Source random seed/fabrics/PCA settings changed')
    if actual['fabric']['enabled'] or actual['fabric']['pca_enabled']:
        raise ValueError('This transfer does not use fabrics or PCA')
    dt = actual['sim']['dt'] * actual['decimation']
    if not math.isclose(dt, POLICY_DT, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Keep the manipulation task at 60 Hz')
    if not math.isclose(actual['sim']['dt'], saved['sim']['dt'], abs_tol=1e-12):
        raise ValueError('Physics cadence changed; requires an explicit new validated contract')
    if actual['episode_length_s'] != saved['episode_length_s']:
        raise ValueError('Source episode/goal timeout changed')
    a = actual['assets']
    if (a['robot_profile'] != 'g1_brainco' or a['robot_fix_base'] or
            a['robot_self_collision'] or not a['robot_actuate_left_hand'] or
            a['robot_body_collision_enabled']):
        raise ValueError('Expected floating complete G1, source self-collision OFF, no static torso proxy')
    return dict(source_run=TOUCH_RUN, source_checkpoint=TOUCH_CHECKPOINT,
        source_sha256=TOUCH_SHA256, bank_sha256=BANK_SHA256, p2p_revision=P2P_REVISION,
        policy_hz=60, physics_hz=1/actual['sim']['dt'], tactile_hz=actual['touch']['sensor_hz'],
        unchanged=['reward', 'goals', 'keypoints', 'success_counters', 'task_terminations',
                   'task_observations', 'BPS128', 'touch', 'domain_randomization', 'object_bank',
                   'right_finger_PD_and_target_coupling'],
        intentional_changes=['floating_complete_body_with_gravity', 'SONIC_body_PD_and_absolute_targets',
            '29_body_plus_6_right_finger_actions', 'append_SONIC_body_history_and_reference',
            'separate_robot_fall_termination', 'standing_reset_for_new_body_joints',
            'left_shoulder_clearance_pose', 'dynamic_torso_replaces_static_proxy'],
        self_collision=False, native_finger_coupling=False, fabrics=False, pca=False)
