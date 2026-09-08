import math

import pytest
import torch

from dextrah_lab.g1_adept.tactile_forces import (
    aggregate_pad_friction, decode_revo2_force_packet,
    project_pad_force, read_pad_contact_forces,
)


def geometry(n=1):
    return (torch.zeros(n, 3), torch.tensor([[1., 0, 0, 0]]).expand(n, 4),
            torch.zeros(3), torch.eye(3), torch.tensor([[-.2] * 3, [.2] * 3]))


def test_friction_masks_individual_anchors_and_preserves_pair_identity():
    # Independent/noncontiguous friction indices; one active point is off-pad.
    vectors = torch.full((8, 3), float("nan"))
    points = torch.full_like(vectors, float("nan"))
    vectors[1], vectors[2], vectors[5] = torch.tensor([2., 0, 0]), torch.tensor([9., 0, 0]), torch.tensor([0., -3, 0])
    points[1], points[2], points[5] = torch.tensor([.1, 0, 0]), torch.tensor([9., 0, 0]), torch.tensor([1.1, 0, 0])
    raw = (vectors, points, torch.tensor([[2, 0], [0, 1]]), torch.tensor([[1, 0], [0, 5]]))
    args = list(geometry(2))
    args[0][1, 0] = 1.
    masked, all_pairs = aggregate_pad_friction(raw, *args)
    torch.testing.assert_close(masked, torch.tensor([[[2., 0, 0], [0., 0, 0]], [[0., 0, 0], [0., -3, 0]]]))
    assert all_pairs[0, 0, 0] == 11.


def test_empty_friction_ignores_padding_and_empty_starts():
    raw = (torch.full((8, 3), float("nan")), torch.full((8, 3), float("nan")),
           torch.zeros(1, 2, dtype=torch.long), torch.full((1, 2), -99))
    for value in aggregate_pad_friction(raw, *geometry()):
        assert value.shape == (1, 2, 3)
        assert torch.isfinite(value).all() and not value.any()


@pytest.mark.parametrize("counts,starts", [([[1]], [[8]]), ([[1]], [[-1]]), ([[8]], [[0]]), ([[-1]], [[0]])])
def test_invalid_or_full_friction_buffer_fails_visibly(counts, starts):
    raw = (torch.zeros(8, 3), torch.zeros(8, 3), torch.tensor(counts), torch.tensor(starts))
    with pytest.raises(RuntimeError):
        aggregate_pad_friction(raw, *geometry())


def test_nonfinite_active_anchor_is_not_silently_zeroed():
    raw = (torch.full((8, 3), float("nan")), torch.zeros(8, 3),
           torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(RuntimeError, match="non-finite"):
        aggregate_pad_friction(raw, *geometry())


class SharedBufferView:
    """Same aliasing hazard as installed PhysX normal/friction getters."""

    def __init__(self):
        self.counts = torch.zeros(1, 2, dtype=torch.long)
        self.starts = torch.zeros(1, 2, dtype=torch.long)
        self.calls = []

    def get_contact_data(self, dt):
        self.calls.append(("normal", dt))
        self.counts[:] = torch.tensor([[2, 1]])
        self.starts[:] = torch.tensor([[1, 4]])
        force = torch.zeros(8, 1)
        force[1], force[2], force[4] = 2., 3., 4.
        return (force, torch.zeros(8, 3), torch.tensor([[0., 0, -1.]]).expand(8, 3),
                torch.zeros(8), self.counts, self.starts)

    def get_friction_data(self, dt):
        self.calls.append(("friction", dt))
        self.counts[:] = torch.tensor([[1, 1]])
        self.starts[:] = torch.tensor([[2, 5]])
        force = torch.zeros(8, 3)
        force[2], force[5] = torch.tensor([1., 0, 0]), torch.tensor([0., -2., 0])
        return force, torch.zeros(8, 3), self.counts, self.starts


def test_normal_aggregation_precedes_shared_buffer_overwrite_and_dt_is_not_reapplied():
    view = SharedBufferView()
    result = read_pad_contact_forces(view, 1 / 120, *geometry())
    assert view.calls == [("normal", 1 / 120), ("friction", 1 / 120)]
    torch.testing.assert_close(result.normal_load_n, torch.tensor([[5., 4.]]))
    torch.testing.assert_close(result.total_pairs_w, torch.tensor([[[1., 0, -5.], [0, -2., -4.]]]))
    torch.testing.assert_close(result.normal_pairs_w, result.reconstructed_normal_pairs_w)
    torch.testing.assert_close(result.friction_pairs_w, result.reconstructed_friction_pairs_w)


def test_no_friction_api_is_not_treated_as_zero_shear():
    class NormalOnly:
        get_contact_data = SharedBufferView().get_contact_data
    with pytest.raises(AttributeError):
        read_pad_contact_forces(NormalOnly(), 1 / 120, *geometry())
    with pytest.raises(ValueError, match="dt"):
        read_pad_contact_forces(SharedBufferView(), 0, *geometry())


def test_normal_and_two_signed_tangential_components():
    q = geometry(3)[1]
    force = torch.tensor([[3., 4, -10], [-3., -4., -10], [0., 0, -10.]])
    out = project_pad_force(force, q, torch.eye(3), normal_axis=2)
    torch.testing.assert_close(out["normal_n"], torch.full((3,), 10.))
    torch.testing.assert_close(out["tangential_n"], force[:, :2])
    torch.testing.assert_close(out["tangential_magnitude_n"], torch.tensor([5., 5., 0.]))


def test_projection_rotates_world_and_pad_frames_without_translations():
    q = torch.tensor([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])
    # World +Y maps to body +X. CAD +X maps to body -Y, CAD +Y to body +X.
    pad_r = torch.tensor([[0., 1., 0], [-1., 0, 0], [0., 0, 1.]])
    result = project_pad_force(torch.tensor([[0., 3., -2.]]), q, pad_r, 2)
    torch.testing.assert_close(result["normal_n"], torch.tensor([2.]))
    torch.testing.assert_close(result["tangential_n"], torch.tensor([[0., 3.]]), atol=1.e-6, rtol=0)


def test_bad_pad_frame_is_rejected():
    with pytest.raises(ValueError, match="rotation"):
        project_pad_force(torch.ones(1, 3), geometry()[1], torch.zeros(3, 3), 2)


def test_revo2_units_angle_wrap_and_clockwise_components():
    out = decode_revo2_force_packet(torch.tensor([100] * 6), torch.tensor([200] * 6),
                                   torch.tensor([0, 90, 180, 270, 359, 1]), torch.tensor([256] * 6))
    torch.testing.assert_close(out["normal_n"], torch.ones(6))
    torch.testing.assert_close(out["tangential_n"][:4],
        torch.tensor([[2., 0], [0, 2.], [-2., 0], [0, -2.]]), atol=1.e-6, rtol=0)
    assert (out["tangential_n"][4] - out["tangential_n"][5]).norm() < .08
    assert out["force_valid"].all()  # A nonzero SEQUENCE byte is healthy.


def test_invalid_directions_zero_load_and_status_remain_distinct():
    out = decode_revo2_force_packet(torch.tensor([100, 100, 100, 231, 100]),
        torch.tensor([100, 0, 100, 384, 100]), torch.tensor([65535, 65535, 360, 39, 0]),
        torch.tensor([0, 0, 0, (12 << 8) | 4, 1]))
    assert out["force_valid"].tolist() == [True, True, True, False, False]
    assert not out["direction_valid"].any()
    assert not out["tangential_n"].any()
    assert out["sensor_sequence"][3] == 12
    assert out["normal_n"][3] == torch.tensor(2.31)  # Preserve reading; mask separately.


def test_above_linear_range_is_flagged_not_silently_clipped():
    out = decode_revo2_force_packet(torch.tensor([100]), torch.tensor([3085]),
                                   torch.tensor([0]), torch.tensor([0]))
    assert out["above_linear_range"].item()
    torch.testing.assert_close(out["tangential_magnitude_n"], torch.tensor([30.85]))


def test_already_converted_forces_are_rejected_to_prevent_double_scaling():
    with pytest.raises(ValueError, match="integer"):
        decode_revo2_force_packet(torch.tensor([1.]), torch.tensor([100]),
                                  torch.tensor([0]), torch.tensor([0]))
