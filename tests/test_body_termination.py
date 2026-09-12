import pytest
import torch
from dextrah_lab.wholebody.body_termination import classify_body_state, combine_terminations


def state():
    root = torch.zeros(4, 13)
    root[:, 2] = .75
    gravity = torch.tensor([[0., 0., -1.]]).expand(4, -1).clone()
    return [torch.zeros(4, 51), torch.zeros(4, 51), root, gravity,
            torch.zeros(4, 3), torch.zeros(4, 13)]


def test_only_affected_episodes_terminate_without_mutating_physics():
    inputs = state()
    inputs[1][1, 34] = -1254.48877
    inputs[2][2, 2] = .35
    before = [x.clone() for x in inputs]
    fallen, numerical, speed = classify_body_state(*inputs, numerical_failure_mode='reset')
    assert fallen.tolist() == [False, False, True, False]
    assert numerical.tolist() == [False, True, False, False]
    terminated, truncated = combine_terminations(
        torch.tensor([False, False, False, True]), torch.ones(4, dtype=torch.bool), fallen, numerical)
    assert terminated.tolist() == [False, True, True, True]
    assert truncated.tolist() == [True, False, False, True]
    for actual, original in zip(inputs, before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize('component', [0, 1, 2, 3, 5])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_always_aborts(component, value):
    inputs = state()
    inputs[component][1, 0] = value
    with pytest.raises(RuntimeError, match='Non-finite'):
        classify_body_state(*inputs, numerical_failure_mode='reset')


def test_diagnostic_abort_and_ordinary_rough_motion():
    inputs = state()
    inputs[1][1, 34] = 1200.
    with pytest.raises(RuntimeError, match='maximum joint speed'):
        classify_body_state(*inputs)
    inputs[1][1, 34] = 80.
    fallen, numerical, _ = classify_body_state(*inputs, numerical_failure_mode='reset')
    assert not fallen.any() and not numerical.any()


def test_upright_threshold_and_environment_origins():
    inputs = state()
    inputs[2][0, 2] += 3.
    inputs[4][0, 2] = 3.
    inputs[3][2, 2] = -.4
    fallen, _, _ = classify_body_state(*inputs, numerical_failure_mode='reset')
    assert fallen.tolist() == [False, False, True, False]
