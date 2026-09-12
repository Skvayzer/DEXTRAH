import pytest
import torch
from dextrah_lab.wholebody.timed_history import TimedSonicHistory
from dextrah_lab.wholebody.sonic import SonicHistory


def terms(n, value):
    return [torch.full((n, d), float(value)) for d in TimedSonicHistory.DIMENSIONS]


def test_original_50hz_equivalence():
    timed, original = TimedSonicHistory(2, .02), SonicHistory(2)
    for step in range(24):
        if step == 14:
            ids = torch.tensor([0])
            timed.reset(ids)
            original.reset(ids)
        data = [torch.randn(2, d) for d in timed.DIMENSIONS]
        torch.testing.assert_close(timed.push(*data), original.push(*data), rtol=0, atol=0)


def test_60hz_ramp_and_causal_executed_action():
    history = TimedSonicHistory(1)
    for step in range(24):
        value = history.push(*terms(1, step/60))
    physical_times = torch.arange(10)*.02 + 23/60-.18
    torch.testing.assert_close(value[:, :30].reshape(10, 3)[:, 0], physical_times)
    offset = (3+29+29)*10
    executed = value[:, offset:offset+290].reshape(10, 29)[:, 0]
    expected = ((physical_times.double()*60 + 1e-6).floor()/60).float()
    torch.testing.assert_close(executed, expected)
    assert torch.all(executed <= physical_times + 1e-7)


def test_partial_reset_does_not_advance_other_environment():
    history = TimedSonicHistory(2)
    for step in range(20):
        history.push(*terms(2, step))
    other = history.value()[1].clone()
    history.reset(torch.tensor([0]))
    history.initialize_fresh(*terms(2, 100))
    torch.testing.assert_close(history.value()[1], other, rtol=0, atol=0)
    assert torch.all(history.value()[0] == 100)


@pytest.mark.parametrize('dt', [0, -.01, .03])
def test_invalid_rate_rejected(dt):
    with pytest.raises(ValueError):
        TimedSonicHistory(1, dt)
