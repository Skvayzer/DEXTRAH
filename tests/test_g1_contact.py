import math

import pytest
import torch

from dextrah_lab.g1_adept.contact import (
    ContactFilterConfig, FingertipContactFilter, adept_grasp_gate, rotate_to_local,
)


def test_gate_requires_thumb_and_other_strictly_above_one_newton():
    forces = torch.tensor([[2., 0, 0, 0, 0], [0., 2, 2, 2, 2],
                           [1., 2, 0, 0, 0], [2., 1.01, 0, 0, 0],
                           [float("nan"), 2, 0, 0, 0]])
    assert adept_grasp_gate(forces).tolist() == [False, False, False, True, False]


def test_table_contact_is_not_object_grasp():
    object_force = torch.zeros(1, 5, 3)
    table_force = torch.ones_like(object_force)
    assert adept_grasp_gate((object_force + table_force).norm(dim=-1)).item()
    assert not adept_grasp_gate(object_force.norm(dim=-1)).item()


def test_filter_hysteresis_duration_and_subset_reset():
    state = FingertipContactFilter(2, config=ContactFilterConfig(time_constant_s=0.))
    force = torch.zeros(2, 5, 3)
    force[..., 0] = .2
    state.update(force, .01)
    assert state.contact.all()
    force[..., 0] = .1
    state.update(force, .01)
    assert state.contact.all()
    torch.testing.assert_close(state.duration_s, torch.full((2, 5), .02))
    state.reset(torch.tensor([0]))
    assert not state.contact[0].any() and state.contact[1].all()
    state.update(force, .01)
    assert not state.contact[0].any() and state.contact[1].all()
    force.zero_()
    state.update(force, .01)
    assert not state.contact.any() and not state.duration_s.any()


def test_invalid_samples_do_not_create_or_retain_contact():
    state = FingertipContactFilter(1, config=ContactFilterConfig(time_constant_s=0.))
    force = torch.ones(1, 5, 3)
    state.update(force, .01)
    force[0, 0, 1] = float("inf")
    force[0, 2, 1] = float("nan")
    state.update(force, .01)
    assert state.valid.tolist() == [[False, True, False, True, True]]
    assert state.contact.tolist() == [[False, True, False, True, True]]
    assert torch.isfinite(state.filtered_n).all()
    assert torch.isfinite(state.derivative_n_s).all()


def test_exponential_filter_and_derivative():
    state = FingertipContactFilter(1)
    force = torch.zeros(1, 5, 3)
    force[..., 2] = 3.
    state.update(force, .03)
    expected = torch.full((1, 5), 3 * (1 - math.exp(-1)))
    torch.testing.assert_close(state.filtered_n, expected)
    torch.testing.assert_close(state.derivative_n_s, expected / .03)


def test_world_to_fingertip_rotation_and_norm():
    quat = torch.tensor([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])
    force = torch.tensor([[0., 2., 0.]])
    torch.testing.assert_close(rotate_to_local(quat, force), torch.tensor([[2., 0., 0.]]), atol=1.e-6, rtol=0)
    torch.testing.assert_close(rotate_to_local(-quat, force).norm(dim=-1), force.norm(dim=-1))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        ContactFilterConfig(on_n=.1, off_n=.2)
    with pytest.raises(ValueError):
        ContactFilterConfig(time_constant_s=float("nan"))
    with pytest.raises(ValueError):
        FingertipContactFilter(1).update(torch.zeros(1, 5, 3), 0.)
    with pytest.raises(ValueError):
        rotate_to_local(torch.zeros(1, 4), torch.ones(1, 3))
