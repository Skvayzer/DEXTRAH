"""Route 2: task-conditioned, TRAINABLE SONIC dynamic body decoder.

Not a latent adapter and not two independent controllers writing arm targets.
The reference encoder/quantizer can remain frozen while all body decoder
weights learn. Teacher networks and source checkpoints are never mutated.
"""
from copy import deepcopy

import torch
from torch import nn

from .actuators import body_motors
from .contract import BODY_JOINTS, hand_joints, joint_indices, nominal_body_pose
from .teacher_bridge import RIGHT_ARM

BODY_ACTION_DIM = 29
ACTION_JOINTS = (*BODY_JOINTS, *hand_joints('right'))
ACTION_DIM = len(ACTION_JOINTS)
ARCHITECTURE = 'sonic_trainable_decoder_recurrent_task_v1'


class FixedTaskNormalizer(nn.Module):
    """Captured teacher statistics, excluding its SAPG leader coefficient.

    The student recurrent state is its OWN state. Captured teacher RNN states
    are never substituted for a new architecture's hidden state. Touch masks
    retain the existing normal/shear/validity/age convention when enabled.
    """
    def __init__(self, mean, variance):
        super().__init__()
        mean, variance = torch.as_tensor(mean).float(), torch.as_tensor(variance).float()
        if mean.ndim != 1 or mean.shape != variance.shape or not len(mean):
            raise ValueError('Invalid task normalizer shape')
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError('Invalid task statistics')
        self.register_buffer('mean', mean.clone())
        self.register_buffer('variance', variance.clone())

    def forward(self, obs):
        if obs.shape[-1] != len(self.mean):
            raise ValueError('Task observation dimension differs from saved normalizer')
        out = ((obs-self.mean)/torch.sqrt(self.variance+1e-5)).clamp(-5.,5.)
        if len(self.mean) == 249:
            raw, normalized = obs[...,224:].reshape(*obs.shape[:-1],5,5), out[...,224:].reshape(*obs.shape[:-1],5,5)
            valid = raw[...,3].clamp(0,1)
            # No contact is valid zero force; sensor-invalid is a separate flag.
            force = normalized[...,:3] * (valid > .5)[...,None]
            age = (raw[...,4]/.1).clamp(0,10)
            out = torch.cat((out[...,:224],torch.cat((force,valid[...,None],age[...,None]),-1).flatten(-2)),-1)
        return out


class SonicManipulationStudent(nn.Module):
    """Batch-first recurrent task encoder and pretrained 29-body-action MLP.

    Inputs: proprio [B,T,930], current task [B,T,D], reference tokens [B,T,64].
    Outputs: mean actions [B,T,35], hidden [1,B,H]. Body action units are SONIC
    normalized targets; finger actions are absolute [-1,1] motor commands.
    """
    def __init__(self, decoder, task_mean, task_variance, hidden_dim=256):
        super().__init__()
        if not isinstance(decoder, nn.Sequential):
            raise ValueError('Requires the audited SONIC dynamic MLP')
        if not isinstance(decoder[0], nn.Linear) or decoder[0].in_features != 994:
            raise ValueError('Expected SONIC input: 64 reference + 930 proprioception')
        if not isinstance(decoder[-1], nn.Linear) or decoder[-1].out_features != 29:
            raise ValueError('Expected 29-joint SONIC body decoder')
        self.decoder = deepcopy(decoder).requires_grad_(True)
        self.normalizer = FixedTaskNormalizer(task_mean,task_variance)
        self.task_dim = len(self.normalizer.mean)
        self.hidden_dim = hidden_dim
        self.task_projection = nn.Sequential(nn.Linear(self.task_dim,hidden_dim),nn.SiLU())
        self.task_rnn = nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        self.task_to_body = nn.Linear(hidden_dim,self.decoder[0].out_features,bias=False)
        nn.init.zeros_(self.task_to_body.weight)
        self.fingers = nn.Linear(self.decoder[-1].in_features+hidden_dim,6)
        nn.init.zeros_(self.fingers.weight)
        nn.init.zeros_(self.fingers.bias)
        self.register_buffer('body_nominal',torch.as_tensor(nominal_body_pose()).clone())
        self.register_buffer('body_scales',torch.tensor([m.action_scale for m in body_motors().values()]))
        self.register_buffer('arm_indices',torch.tensor(joint_indices(BODY_JOINTS,RIGHT_ARM)))
        self.register_buffer('other_indices',torch.tensor([i for i,n in enumerate(BODY_JOINTS) if n not in RIGHT_ARM]))

    @classmethod
    def from_sonic(cls, sonic, task_mean, task_variance, hidden_dim=256):
        module = sonic.actor.actor_module
        if (list(module.decoder_input_features['g1_dyn']) != ['token_flattened','proprioception']
                or dict(module.decoder_output_feature_dims['g1_dyn']) != {'action':29}):
            raise ValueError('Pinned dynamic decoder contract changed')
        return cls(module.decoders['g1_dyn'].module,task_mean,task_variance,hidden_dim)

    def forward(self, proprio, task, tokens, hidden=None, episode_starts=None):
        if task.ndim != 3 or proprio.shape != (*task.shape[:2],930) or tokens.shape != (*task.shape[:2],64):
            raise ValueError('Expected matching batch/time body, task and token observations')
        batch, steps = task.shape[:2]
        if hidden is None:
            hidden = task.new_zeros(1,batch,self.hidden_dim)
        if hidden.shape != (1,batch,self.hidden_dim):
            raise ValueError('Wrong student recurrent-state shape')
        projected = self.task_projection(self.normalizer(task))
        if episode_starts is None:
            context, hidden = self.task_rnn(projected,hidden)
        else:
            if episode_starts.shape != (batch,steps) or episode_starts.dtype != torch.bool:
                raise ValueError('Episode-start mask must be boolean [B,T]')
            contexts=[]
            for t in range(steps):
                hidden = hidden * (~episode_starts[:,t])[None,:,None]
                current, hidden = self.task_rnn(projected[:,t:t+1],hidden)
                contexts.append(current)
            context = torch.cat(contexts,1)
        x = torch.cat((tokens,proprio),-1)
        x = self.decoder[0](x) + self.task_to_body(context)
        for layer in list(self.decoder.children())[1:-1]:
            x = layer(x)
        body = self.decoder[-1](x)
        fingers = torch.tanh(self.fingers(torch.cat((x,context),-1)))
        return torch.cat((body,fingers),-1),hidden

    def body_targets(self, action):
        if action.shape[-1] != ACTION_DIM:
            raise ValueError('Expected 35 actions, not the old SAPG 13 or latent 70')
        return self.body_nominal + action[...,:29]*self.body_scales


def direct_imitation_loss(model, action, arm_targets, finger_actions, sonic_action,
                          valid, standing_mask=None, retention_weight=1., rehearsal_weight=1.):
    """Masked physical-arm imitation and SOFT action-sliced SONIC retention.

    Manipulation supervision never labels absent teacher legs. Standing-only
    rehearsal supervises all body outputs and no manipulation fingers. Masks
    describe sample provenance; they must not be inferred from loss magnitude.
    """
    shape = action.shape[:-1]
    if (action.shape[-1] != 35 or arm_targets.shape != (*shape,7)
            or finger_actions.shape != (*shape,6) or sonic_action.shape != (*shape,29)
            or valid.shape != shape or valid.dtype != torch.bool):
        raise ValueError('Invalid direct distillation labels/mask')
    if min(retention_weight,rehearsal_weight) < 0:
        raise ValueError('Loss weights cannot be negative')
    if standing_mask is None:
        standing_mask = torch.zeros_like(valid)
    if standing_mask.shape != shape or standing_mask.dtype != torch.bool or (standing_mask & valid).any():
        raise ValueError('Standing and manipulation provenance masks must be disjoint')
    if not (valid.any() or standing_mask.any()):
        raise ValueError('At least one valid training sample is required')
    # Select before computing differences; NaN padding must not leak via NaN*0.
    zero = action[...,:0].sum()
    arm_loss = finger_loss = retention_loss = rehearsal_loss = zero
    metrics = dict(manipulation_samples=int(valid.sum()),standing_samples=int(standing_mask.sum()))
    if valid.any():
        targets = model.body_targets(action[valid])
        arm_error = targets[:,model.arm_indices] - arm_targets[valid].detach()
        finger_error = action[valid][...,29:] - finger_actions[valid].detach()
        body_error = (action[valid][...,:29]-sonic_action[valid].detach()) * model.body_scales
        arm_loss = arm_error.square().mean()
        finger_loss = finger_error.square().mean()
        retention_loss = body_error[:,model.other_indices].square().mean()
        metrics.update(arm_rmse_rad=float(arm_loss.detach().sqrt()),
            arm_max_error_rad=float(arm_error.detach().abs().max()),
            finger_rmse=float(finger_loss.detach().sqrt()),
            other_body_rmse_rad=float(retention_loss.detach().sqrt()))
    if standing_mask.any():
        error = (action[standing_mask][...,:29]-sonic_action[standing_mask].detach()) * model.body_scales
        rehearsal_loss = error.square().mean()
        metrics['standing_rmse_rad'] = float(rehearsal_loss.detach().sqrt())
    total = arm_loss+finger_loss+retention_weight*retention_loss+rehearsal_weight*rehearsal_loss
    if not torch.isfinite(total):
        raise ValueError('Nonfinite loss on valid supervision')
    metrics['loss'] = float(total.detach())
    return total,metrics
