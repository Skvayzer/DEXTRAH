import pytest
import torch

from dextrah_lab.adept.curriculum import RollingSuccessRate


def test_success_window_reports_only_after_data_arrives():
    window = RollingSuccessRate(4, "cpu")
    assert window.rate == 0.0
    assert not window.ready
    window.update(torch.tensor([True, False]))
    assert window.rate == pytest.approx(0.5)
    assert not window.ready


def test_success_window_wraps_and_retains_most_recent_outcomes():
    window = RollingSuccessRate(4, "cpu")
    window.update(torch.tensor([False, False, False]))
    window.update(torch.tensor([False, True, True]))
    assert window.ready
    assert window.rate == pytest.approx(0.5)


def test_success_window_round_trips_state():
    source = RollingSuccessRate(5, "cpu")
    source.update(torch.tensor([True, False, True]))
    restored = RollingSuccessRate(5, "cpu")
    restored.load_state_dict(source.state_dict())
    restored.update(torch.tensor([True, True, False]))
    assert restored.ready
    assert restored.rate == pytest.approx(0.6)


def test_success_window_clear_starts_a_new_adr_level():
    window = RollingSuccessRate(2, "cpu")
    window.update(torch.tensor([True, True]))
    window.clear()
    assert window.count == 0
    assert window.rate == 0.0
