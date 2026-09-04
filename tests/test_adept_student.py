import torch
from torch import nn

from dextrah_lab.adept.student import (
    AdeptStudentPolicy,
    SpatiallyAnchoredTactile,
    TactileEncoder,
    fourier_position,
)


class FakeVision(nn.Module):
    def forward(self, left, center):
        return torch.zeros(left.shape[0], 256, device=left.device)


def test_fourier_position_has_27_dimensions_for_four_bands():
    encoded = fourier_position(torch.zeros(2, 5, 3), bands=4)
    assert encoded.shape == (2, 5, 27)


def test_shared_tactile_encoder_and_film_produce_160_dimensions():
    module = SpatiallyAnchoredTactile()
    depth = torch.rand(2, 5, 1, 16, 16)
    position = torch.rand(2, 5, 3)
    assert module(depth, position).shape == (2, 160)
    assert isinstance(module.encoder, TactileEncoder)


def test_kuka_vision_only_student_outputs_appendix_shapes():
    policy = AdeptStudentPolicy(
        proprio_dim=206,
        action_dim=23,
        use_tactile=False,
        vision_encoder=FakeVision(),
    )
    output = policy(torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8), torch.rand(2, 206))
    assert output["mean"].shape == (2, 23)
    assert output["log_std"].shape == (2, 23)
    assert output["keypoints"].shape == (2, 8, 3)


def test_flexiv_student_requires_and_consumes_five_tactile_maps():
    policy = AdeptStudentPolicy(
        proprio_dim=196,
        action_dim=29,
        use_tactile=True,
        vision_encoder=FakeVision(),
    )
    output = policy(
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 196),
        torch.rand(2, 5, 1, 16, 16),
        torch.rand(2, 5, 3),
    )
    assert output["mean"].shape == (2, 29)

