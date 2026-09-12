"""Teacher-guided 13-action -> latent-and-fingers transfer building blocks.

No SAPG optimizer or simulator is started here. The body target fitter needs
REAL full-body proprioception and compatible planned references. Do not train
on invented standing states and call the result a manipulation controller.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .actuators import body_motors
from .contract import BODY_JOINTS, LATENT_DIM, nominal_body_pose, joint_indices
from .teacher_bridge import RIGHT_ARM


@dataclass
class LatentFit:
    residual: torch.Tensor
    normalized_body_action: torch.Tensor
    arm_error_rad: torch.Tensor
    baseline_arm_error_rad: torch.Tensor
    other_body_error_rad: torch.Tensor
    accepted: torch.Tensor


def fit_teacher_arm_targets(sonic, proprio, reference_q, reference_qd, reference_ori6,
                           teacher_arm_targets, *, steps=80, learning_rate=.05,
                           tolerance_rad=.05, residual_penalty=1e-4,
                           body_retention_weight=.1, max_residual=5., body_retention_tolerance_rad=.05):
    """Fit residual labels by matching decoded SEVEN arm targets in radians.

    This does NOT compare the teacher's 13-vector with a 70-vector. Fingers
    have the same hardware interface and are supervised separately. The other
    22 body outputs are regularized towards unmodified SONIC (never zeros or
    the fixed-body teacher's absent legs). Keep the best exact forward iterate
    independently per sample because straight-through FSQ loss is nonsmooth.
    """
    if steps < 1 or not all(math.isfinite(v) and v > 0 for v in
        (learning_rate, tolerance_rad, max_residual, body_retention_tolerance_rad)):
        raise ValueError('Invalid fitting schedule')
    if not all(math.isfinite(v) and v >= 0 for v in (residual_penalty, body_retention_weight)):
        raise ValueError('Invalid fitting regularization')
    count = len(proprio)
    if count<1:
        raise ValueError('Cannot fit an empty state batch')
    if teacher_arm_targets.shape != (count, 7) or not torch.isfinite(teacher_arm_targets).all():
        raise ValueError('Expected finite seven-joint targets in radians')
    target = teacher_arm_targets.detach()
    arm_ids = joint_indices(BODY_JOINTS, RIGHT_ARM)
    other_ids = [i for i in range(29) if i not in arm_ids]
    nominal = torch.as_tensor(nominal_body_pose(), device=proprio.device, dtype=proprio.dtype)
    scales = proprio.new_tensor([m.action_scale for m in body_motors().values()])
    with torch.no_grad():
        baseline = sonic(proprio, reference_q, reference_qd, reference_ori6)
    baseline_targets = nominal + baseline*scales
    residual = nn.Parameter(proprio.new_zeros(count, LATENT_DIM))
    optimizer = torch.optim.Adam([residual], lr=learning_rate)
    best_score = proprio.new_full((count,), float('inf'))
    best = residual.detach().clone()
    best_action = baseline.clone()
    for iteration in range(steps+1):
        action = sonic.decode_for_imitation(proprio, reference_q, reference_qd, reference_ori6, residual)
        physical_targets = nominal + action*scales
        arm_loss = (physical_targets[:, arm_ids]-target).square().mean(-1)
        retain = (physical_targets[:, other_ids]-baseline_targets[:, other_ids]).square().mean(-1)
        score = arm_loss + body_retention_weight*retain + residual_penalty*residual.square().mean(-1)
        if not torch.isfinite(score).all():
            raise ValueError('Nonfinite latent fitting loss')
        with torch.no_grad():
            improved = score < best_score
            best_score[improved] = score[improved]
            best[improved] = residual[improved]
            best_action[improved] = action[improved]
        if iteration == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        score.mean().backward()
        if residual.grad is None or not torch.isfinite(residual.grad).all():
            raise ValueError('No finite residual gradient through frozen controller')
        optimizer.step()
        with torch.no_grad():
            residual.clamp_(-max_residual, max_residual)
    error = ((nominal+best_action*scales)[:, arm_ids]-target).abs().amax(-1)
    baseline_error = (baseline_targets[:, arm_ids]-target).abs().amax(-1)
    other_error = ((nominal+best_action*scales)[:, other_ids]-baseline_targets[:, other_ids]).abs().amax(-1)
    return LatentFit(best.detach(), best_action.detach(), error.detach(),
                     baseline_error.detach(), other_error.detach(),
                     ((error<=tolerance_rad)&(other_error<=body_retention_tolerance_rad)).detach())


class TeacherInitializedAdapter(nn.Module):
    """Recurrent student trained from teacher data BEFORE any SAPG updates.

    Inputs must be normalized by the dataset's frozen, checkpoint-provenanced
    statistics. BPS/tactile packing is owned by the dataset/task contract, not
    guessed here. New output heads do NOT imply grasping RL from scratch.
    """
    def __init__(self, observation_dim, hidden_dim=256, hands=1):
        super().__init__()
        if observation_dim < 1 or hidden_dim < 1 or hands not in (1, 2):
            raise ValueError('Invalid adapter dimensions')
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.hand_dim = hands*6
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, LATENT_DIM)
        self.fingers = nn.Linear(hidden_dim, self.hand_dim)
        nn.init.zeros_(self.latent.weight)
        nn.init.zeros_(self.latent.bias)

    def forward(self, observations, state=None, episode_starts=None):
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError('Expected [batch, time, observation]')
        if not torch.isfinite(observations).all():
            raise ValueError('Nonfinite adapter observation')
        n, length, _ = observations.shape
        if length < 1:
            raise ValueError('Empty adapter sequence')
        if state is None:
            state = observations.new_zeros(1, n, self.hidden_dim)
        if state.shape != (1, n, self.hidden_dim):
            raise ValueError('Invalid adapter recurrent state')
        if episode_starts is None:
            episode_starts = torch.zeros(n, length, dtype=torch.bool, device=observations.device)
        if episode_starts.shape != (n, length) or episode_starts.dtype != torch.bool:
            raise ValueError('Expected boolean episode-start mask')
        encoded = self.encoder(observations)
        outputs = []
        for t in range(length):
            state = state * (~episode_starts[:, t])[None, :, None]
            out, state = self.recurrent(encoded[:, t:t+1], state)
            outputs.append(out)
        features = torch.cat(outputs, 1)
        return self.latent(features), torch.tanh(self.fingers(features)), state


def adapter_imitation_loss(latent, hand, latent_labels, hand_labels, valid, latent_valid):
    """Only accepted latent fits supervise the body; valid teacher fingers remain useful.

    All hand tensors use the teacher's normalized six-actuator action convention.
    Never treat recorded physical joint positions as normalized action labels.
    """
    if latent.shape != latent_labels.shape or hand.shape != hand_labels.shape:
        raise ValueError('Imitation target shape mismatch')
    if latent.shape[:-1] != hand.shape[:-1] or valid.shape != hand.shape[:-1] or latent_valid.shape != valid.shape:
        raise ValueError('Imitation mask shape mismatch')
    if valid.dtype != torch.bool or latent_valid.dtype != torch.bool or not valid.any():
        raise ValueError('Requires boolean masks and at least one valid teacher transition')
    body_mask = valid & latent_valid
    fingers = (hand[valid]-hand_labels.detach()[valid]).square().mean()
    body = ((latent[body_mask]-latent_labels.detach()[body_mask]).square().mean()
            if body_mask.any() else latent.sum()*0.)
    return body+fingers, dict(latent=body.detach(), fingers=fingers.detach(),
                             accepted_body_samples=int(body_mask.sum()), hand_samples=int(valid.sum()))
