"""Warm-start helpers for training a new expert from an existing checkpoint."""
import hashlib
from pathlib import Path
from types import MethodType
import torch


def reset_value_normalizers(module):
    """Reset count-based value normalizers; returns the names that were reset.

    rl_games' RunningMeanStd averages by sample count, so after billions of
    reposing transitions it would effectively never adapt to a new reward scale.
    Observation normalizers are kept: the carry observation layout is unchanged.
    """
    names = []
    for name, child in module.named_modules():
        if name.endswith('value_mean_std'):
            child.running_mean.zero_()
            child.running_var.fill_(1.)
            child.count.fill_(1.)
            names.append(name)
    return names


def load_initial_weights(algo, checkpoint, *, reset_values=False):
    """Load actor, normalizers and asymmetric critic; keep fresh optimizers and counters."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload[0] if 0 in payload else payload
    algo.set_weights(state)
    if algo.has_central_value:
        algo.central_value_net.load_state_dict(state['assymetric_vf_nets'])
    reset = reset_all_value_normalizers(algo) if reset_values else []
    with Path(checkpoint).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest,
                source_epoch=int(state.get('epoch', -1)), source_frame=int(state.get('frame', -1)),
                optimizers='fresh', epoch_and_frame='start at zero', value_normalizers_reset=reset)


def reset_all_value_normalizers(algo):
    names = ['actor.'+n for n in reset_value_normalizers(algo.model)]
    if algo.has_central_value:
        names += ['critic.'+n for n in reset_value_normalizers(algo.central_value_net)]
    return names


def install_critic_warmup(algo, epochs):
    """Skip actor optimizer steps for the first ``epochs`` epochs.

    The asymmetric critic has its own optimizer and keeps learning, so the new
    reward's value function settles before the copied policy is changed.
    """
    optimizer = algo.optimizer
    original = optimizer.step

    def step(self, *args, **kwargs):
        if algo.epoch_num <= epochs:
            return None
        return original(*args, **kwargs)

    optimizer.step = MethodType(step, optimizer)
    return dict(critic_warmup_epochs=epochs)
