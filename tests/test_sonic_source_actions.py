"""Exact finger delay/EMA/follower parity against the pinned source function."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from dextrah_lab.wholebody.source_actions import apply_wholebody_action


@pytest.fixture
def source_pipeline():
    path = Path('/data1/users/konstantin.smirnov/play2perfect/isaacsimenvs/tasks/play/utils/action_utils.py')
    if not path.exists():
        pytest.skip('Pinned original task utilities are on the workstation')
    spec = importlib.util.spec_from_file_location('source_action_parity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_action_pipeline


def environment(delay):
    n = 3
    limits = torch.tensor([-2., 2.]).expand(n, 51, 2).clone()
    return NS(num_envs=n, device='cpu', step_dt=1/60,
        cfg=NS(domain_randomization=NS(use_action_delay=delay, action_delay_max=3),
               action=NS(dof_speed_scale=1.5, arm_moving_average=.1, hand_moving_average=.1)),
        episode_length_buf=torch.zeros(n, dtype=torch.long), _successes=torch.zeros(n, dtype=torch.long),
        _action_queue=torch.zeros(n, 4, 13), _wholebody_action_queue=torch.zeros(n, 4, 35),
        _num_arm_action_joints=7, _arm_joint_ids=torch.arange(7), _hand_joint_ids=torch.arange(29, 35),
        _arm_lower=torch.full((n, 7), -2.), _arm_upper=torch.full((n, 7), 2.),
        _hand_lower=torch.full((n, 6), .02), _hand_upper=torch.full((n, 6), 1.48),
        _prev_targets=torch.zeros(n, 51), _cur_targets=torch.zeros(n, 51),
        _mimic_target_joint_ids=torch.arange(35, 40), _mimic_source_hand_action_ids=torch.arange(1, 6),
        _mimic_multipliers=torch.tensor([[.6, .7, .8, .9, 1.]]), _mimic_offsets=torch.zeros(1, 5),
        robot=NS(data=NS(joint_pos_limits=limits)), _body_ids=torch.arange(29),
        _body_nominal=torch.zeros(29), _body_scales=torch.full((29,), .2),
        _body_lower=torch.full((n, 29), -2.), _body_upper=torch.full((n, 29), 2.),
        _body_center=torch.zeros(n, 29), _body_half_range=torch.full((n, 29), 2.),
        _last_body_action=torch.zeros(n, 29))


@pytest.mark.parametrize('delay', [False, True])
def test_fingers_identical_across_delays_and_goal_counter_reset(source_pipeline, delay):
    original, wholebody = environment(delay), environment(delay)
    for step in range(15):
        torch.manual_seed(123+step)
        original_action = torch.randn(3, 13).clamp(-1, 1)
        body_action = torch.rand(3, 29)*2-1
        whole_action = torch.cat((body_action, original_action[:, 7:]), -1)
        if step == 6:
            # Advancing the object goal resets its step counter but must NOT
            # be treated as a new robot episode for the source delay queue.
            for env in (original, wholebody):
                env.episode_length_buf[1] = 0
                env._successes[1] = 2
        torch.manual_seed(step)
        source_pipeline(original, original_action)
        torch.manual_seed(step)
        apply_wholebody_action(wholebody, whole_action, source_pipeline)
        torch.testing.assert_close(wholebody._cur_targets[:, 29:40], original._cur_targets[:, 29:40], rtol=0, atol=0)
        torch.testing.assert_close(wholebody._prev_targets[:, 29:40], original._prev_targets[:, 29:40], rtol=0, atol=0)
        torch.testing.assert_close(wholebody._action_queue[:, :, 7:], original._action_queue[:, :, 7:], rtol=0, atol=0)
        # No legacy arm command may overwrite a SONIC body target. Executed
        # action history must invert the actual clamped body target exactly.
        torch.testing.assert_close(wholebody._last_body_action*.2, wholebody._cur_targets[:, :29])
        assert not wholebody._cur_targets[:, 40:].any()  # untouched left fingers
        for env in (original, wholebody):
            env.episode_length_buf += 1


def test_clipped_body_commands_and_invalid_input(source_pipeline):
    env = environment(False)
    apply_wholebody_action(env, torch.full((3, 35), 3.), source_pipeline)
    assert torch.all(env._cur_targets[:, :29] == 2.)
    assert torch.all(env._last_body_action == 10.)
    with pytest.raises(ValueError, match='finite'):
        apply_wholebody_action(env, torch.full((3, 35), float('nan')), source_pipeline)
