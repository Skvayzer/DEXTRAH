"""SAPG meta-actions for the ORIGINAL frozen SONIC, not joint-space noise.

The environment owns SONIC; it is absent from the policy optimizer/checkpoint.
Every checkpoint contract pins its external pretrained weights by SHA-256.
Only the new latent adaptor and copied SAPG task/finger network learn.
"""
from copy import deepcopy
from types import MethodType

import torch
from torch import nn

from .sapg_network import recurrent_task_features
from .task_contract import TASK_ACTOR_DIM, BODY_EXTRA_DIM

FROZEN_ARCHITECTURE = 'frozen_pretrained_sonic_sapg_latent64_fingers6_v1'
LATENT_ACTION_DIM = 64
FROZEN_ACTION_DIM = 70


class FrozenSonicSapgNetwork(nn.Module):
    def __init__(self, source, num_seqs):
        super().__init__()
        self.source = deepcopy(source.a2c_network).requires_grad_(True)
        if (self.source.separate or not self.source.has_rnn or not self.source.is_rnn_before_mlp
                or self.source.rnn_name != 'lstm' or not self.source.rnn_ln
                or self.source.pid_idx != TASK_ACTOR_DIM or self.source.fixed_sigma != 'coef_cond'):
            raise ValueError('Unsupported source recurrent SAPG interface')
        self.source.num_seqs = num_seqs
        self.source.sigma.requires_grad_(False)
        # Zero final layer means the initial body command is exactly pristine
        # SONIC, while the six finger means and recurrent task features are the
        # completed 17.36B-frame touch teacher. Do not transplant the route-2
        # hidden-layer projection: it was co-trained with a different decoder.
        self.latent_adapter = nn.Sequential(
            nn.Linear(512+BODY_EXTRA_DIM, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, LATENT_ACTION_DIM)).to(self.source.mu.weight)
        nn.init.zeros_(self.latent_adapter[-1].weight)
        nn.init.zeros_(self.latent_adapter[-1].bias)
        old = self.source.sigma.detach()
        leader = int(torch.nonzero(self.source.param_ids == 0, as_tuple=True)[0].item())
        offset = (old[:, :7].mean(1)-old[leader, :7].mean()).clamp(-.693147, .693147)
        # This is latent exploration, NOT independent noise on body joints.
        # The environment applies the upstream 0.1 pre-FSQ scaling afterwards.
        latent = old.new_full((len(old), LATENT_ACTION_DIM), 1.).log()+offset[:, None]
        self.logstd = nn.Parameter(torch.cat((latent, old[:, 7:13]), -1))

    def is_rnn(self):
        return True

    def get_default_rnn_state(self):
        return self.source.get_default_rnn_state()

    def get_value_layer(self):
        return self.source.value

    def forward(self, inputs):
        obs = inputs['obs']
        task = torch.cat((obs[:, :TASK_ACTOR_DIM], obs[:, TASK_ACTOR_DIM+BODY_EXTRA_DIM:]), -1)
        features, states, ids = recurrent_task_features(self.source, task, inputs.get('rnn_states'),
            inputs.get('seq_length', 1), inputs.get('dones'), inputs.get('bptt_len', 0))
        extra = obs[:, TASK_ACTOR_DIM:TASK_ACTOR_DIM+BODY_EXTRA_DIM]
        latent = self.latent_adapter(torch.cat((features, extra), -1))
        fingers = self.source.mu_act(self.source.mu(features))[:, 7:13]
        value = self.source.value_act(self.source.value(features))
        return torch.cat((latent, fingers), -1), self.logstd[ids], value, states


def install_latent_action_bounds(algo):
    """Keep original finger bounds loss; latent coordinates are not joint limits.

    SAPG likelihoods/ratios are calculated in the actual 70-D meta-action space.
    No clipped latent distribution or extra body-joint Gaussian is introduced.
    """
    if algo.actions_num != FROZEN_ACTION_DIM or algo.clip_actions:
        raise ValueError('Frozen SONIC requires 70 raw meta-actions without global clipping')
    original = algo.bound_loss

    def finger_bounds(self, means):
        if means.shape[-1] != FROZEN_ACTION_DIM:
            raise ValueError('Expected 64 latent + 6 finger means')
        return original(means[..., LATENT_ACTION_DIM:])

    algo.bound_loss = MethodType(finger_bounds, algo)


def install_latent_optimizer(algo, adapter_lr=3e-4):
    """Fit the new adaptor faster while retaining the teacher's small task LR."""
    if algo.optimizer.state or adapter_lr <= 0:
        raise ValueError('Install latent optimizer before loading any optimizer state')
    network = algo.model.a2c_network
    adapter = list(network.latent_adapter.parameters())
    adapter_ids = {id(p) for p in adapter}
    source = [p for p in algo.model.parameters() if p.requires_grad and id(p) not in adapter_ids]
    scale = adapter_lr / float(algo.last_lr)
    algo.optimizer = torch.optim.Adam([
        dict(params=source, lr=float(algo.last_lr), lr_scale=1.),
        dict(params=adapter, lr=adapter_lr, lr_scale=scale)],
        lr=float(algo.last_lr), eps=1e-8, weight_decay=algo.weight_decay)
    original_update = algo.update_lr

    def update_lr(self, lr):
        original_update(lr)
        for group in self.optimizer.param_groups:
            group['lr'] *= group['lr_scale']

    algo.update_lr = MethodType(update_lr, algo)


class FrozenControllerAudit:
    """Bitwise all-tensor audit, including buffers; never just check one layer."""
    def __init__(self, sonic, optimizer=None):
        self.sonic = sonic
        self.original = {k: v.detach().clone() for k, v in sonic.state_dict().items()}
        self.check(optimizer)

    def check(self, optimizer=None):
        if self.sonic.training or self.sonic.actor.training:
            raise RuntimeError('Pretrained SONIC left evaluation mode')
        parameters = list(self.sonic.parameters())
        if any(p.requires_grad or p.grad is not None for p in parameters):
            raise RuntimeError('Pretrained SONIC must have no trainable parameters or gradients')
        if optimizer is not None:
            optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
            if any(id(p) in optimized for p in parameters):
                raise RuntimeError('Pretrained SONIC was included in the SAPG optimizer')
        state = self.sonic.state_dict()
        if state.keys() != self.original.keys() or any(
                not torch.equal(v, self.original[k]) for k, v in state.items()):
            raise RuntimeError('A pretrained SONIC parameter/buffer changed')
        return dict(pretrained_sonic_unchanged=True, pretrained_sonic_trainable_parameters=0,
                    pretrained_sonic_tensors_checked=len(state))
