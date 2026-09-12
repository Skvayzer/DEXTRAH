import pytest
import torch
from torch import nn
from dextrah_lab.wholebody.transfer import TeacherInitializedAdapter, fit_teacher_arm_targets, adapter_imitation_loss
from dextrah_lab.wholebody.contract import BODY_JOINTS, nominal_body_pose, joint_indices
from dextrah_lab.wholebody.actuators import body_motors
from dextrah_lab.wholebody.teacher_bridge import RIGHT_ARM


class LinearFrozenController(nn.Module):
    """Analytic test double, never substituted for SONIC in a simulator."""
    def forward(self, obs, q, qd, ori, residual=None):
        return obs.new_zeros(len(obs), 29) if residual is None else residual[:, :29]

    def decode_for_imitation(self, obs, q, qd, ori, residual):
        return self(obs, q, qd, ori, residual)


def test_fitter_matches_arm_targets_and_retains_other_body_outputs():
    obs = torch.zeros(2, 930)
    q = torch.zeros(2, 10, 29)
    ori = torch.zeros(2, 10, 6)
    ids = joint_indices(BODY_JOINTS, RIGHT_ARM)
    nominal = torch.as_tensor(nominal_body_pose())
    scales = torch.tensor([m.action_scale for m in body_motors().values()])
    target = nominal[ids][None, :].repeat(2, 1) + .2*scales[ids]
    fitted = fit_teacher_arm_targets(LinearFrozenController(), obs, q, q, ori, target,
        steps=100, residual_penalty=0, tolerance_rad=.005)
    assert fitted.accepted.all()
    assert (fitted.arm_error_rad < fitted.baseline_arm_error_rad).all()
    other = [i for i in range(29) if i not in ids]
    assert not fitted.normalized_body_action[:, other].any()
    with pytest.raises(ValueError, match='seven-joint'):
        fit_teacher_arm_targets(LinearFrozenController(), obs, q, q, ori, torch.zeros(2, 13))


def test_adapter_episode_reset_and_zero_initial_body_residual():
    torch.manual_seed(42)
    model = TeacherInitializedAdapter(20, hidden_dim=16)
    obs = torch.randn(2, 3, 20)
    starts = torch.zeros(2, 3, dtype=torch.bool)
    starts[1, 2] = True
    latent, hand, state = model(obs, episode_starts=starts)
    assert latent.shape == (2, 3, 64) and hand.shape == (2, 3, 6)
    assert not latent.any()
    _, fresh_hand, _ = model(obs[1:, 2:])
    torch.testing.assert_close(hand[1:, 2:], fresh_hand)
    assert state.shape == (1, 2, 16)
    assert (hand.abs()<=1).all()


def test_invalid_latent_fit_cannot_supervise_body_but_keeps_valid_fingers():
    model = TeacherInitializedAdapter(20, hidden_dim=16)
    obs = torch.randn(2, 3, 20)
    latent, hand, _ = model(obs)
    valid = torch.ones(2, 3, dtype=torch.bool)
    accepted = torch.zeros_like(valid)
    loss, metrics = adapter_imitation_loss(latent, hand, torch.ones_like(latent),
        torch.ones_like(hand)*.3, valid, accepted)
    loss.backward()
    assert metrics['accepted_body_samples'] == 0 and metrics['hand_samples'] == 6
    assert not model.latent.weight.grad.any()
    assert model.fingers.weight.grad.abs().sum()>0
    with pytest.raises(ValueError, match='at least one'):
        adapter_imitation_loss(latent, hand, latent, hand, ~valid, accepted)
