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


def install_critic_warmup(algo, epochs, start=0):
    """Freeze the policy for the first ``epochs`` epochs after epoch ``start`` (a resumed run: its resume epoch).

    The asymmetric critic has its own optimizer and keeps learning, so the new
    reward's value function settles before the copied policy is changed. The
    policy network's own value head (``*.value`` parameters, e.g. a2c_network.source.value)
    also keeps learning, alone: it is trained by the actor optimizer through the shared
    trunk, and when frozen its stale loss hits the trunk on the first policy update
    (job 1587: actor value loss 1.5 through the warm-up, KL 0.48 on the first actor epoch).
    """
    optimizer = algo.optimizer
    original = optimizer.step
    model = getattr(algo, 'model', None)
    head = {id(p) for n, p in (model.named_parameters() if model is not None else ()) if n.split('.')[-2:-1] == ['value']}

    def step(self, *args, **kwargs):
        if algo.epoch_num <= start+epochs:
            if not head:
                return None
            for group in self.param_groups:
                for p in group['params']:
                    if id(p) not in head:
                        p.grad = None          # skipped by the optimizer (no moment updates)
        return original(*args, **kwargs)

    optimizer.step = MethodType(step, optimizer)
    return dict(critic_warmup_epochs=epochs, warmup_value_head_tensors=len(head))


def install_delayed_best(algo, epochs):
    """Start best-checkpoint tracking ``epochs`` updates after the current epoch.

    Until then the best record is +inf, so nothing is saved; afterwards it is
    reset so the first fully post-change return becomes the best.
    """
    start = int(algo.epoch_num)
    algo.last_mean_rewards = float('inf')
    original = algo.update_epoch

    def update_epoch(self):
        epoch = original()
        if epoch == start+epochs:
            self.last_mean_rewards = -1000000000
            print(f'BEST_TRACKING_STARTED epoch={epoch}', flush=True)
        return epoch

    algo.update_epoch = MethodType(update_epoch, algo)
    return dict(blocked_until_epoch=start+epochs, delay_epochs=epochs)


@torch.no_grad()
def init_carry_heads(algo, *, fingers=True, latent=True, logstd=-1.5):
    """DexCompose (arXiv:2606.28323, App. B.1) residual heads: mean head initialised to zero,
    log-std to -1.5. Frozen-SONIC SAPG network: the latent adapter's last layer (64 latent
    residuals) and the source mu rows 7:13 (6 fingers, here the finger stabilizer's correction)."""
    net = algo.model.a2c_network
    report = {}
    if latent:
        net.latent_adapter[-1].weight.zero_()
        net.latent_adapter[-1].bias.zero_()
        net.logstd[:, :64] = logstd
        report['latent'] = 'mean head zero, log-std %.2f' % logstd
    if fingers:
        net.source.mu.weight[7:13].zero_()
        net.source.mu.bias[7:13].zero_()
        net.logstd[:, 64:70] = logstd
        report['fingers'] = 'mean rows zero, log-std %.2f' % logstd
    return report


def freeze_latent_dims(algo, keep=7):
    """Carry v2 arm channel: only the first ``keep`` latent outputs act (right-arm reference changes);
    the adapter head rows and log-std columns keep..63 get no gradient, so those action dimensions
    keep a fixed mean (zero) and std and add nothing to the PPO ratio or entropy. The adapter's hidden
    layers stay trainable (they feed the acting rows)."""
    net = algo.model.a2c_network
    head = net.latent_adapter[-1]
    row = torch.zeros(head.weight.shape[0], 1, device=head.weight.device)
    row[:keep] = 1.
    hooks = [head.weight.register_hook(lambda g: g*row), head.bias.register_hook(lambda g: g*row[:, 0])]
    mask = torch.ones_like(net.logstd)
    mask[:, keep:64] = 0.
    hooks.append(net.logstd.register_hook(lambda g: g*mask))
    algo._carry_latent_freeze = hooks
    return dict(latent_head_rows_trainable=f'0..{keep-1}', latent_logstd_trainable=f'0..{keep-1}', fingers='trainable')


def freeze_latent_residual(algo):
    """Carry stage 1: the latent residual is not executed (the env zeroes it); keep its head and
    log-std fixed so the unused dimensions add nothing to the PPO ratio or entropy."""
    net = algo.model.a2c_network
    net.latent_adapter.requires_grad_(False)
    mask = torch.ones_like(net.logstd)
    mask[:, :64] = 0.
    handle = net.logstd.register_hook(lambda grad: grad*mask)
    algo._carry_latent_freeze = handle
    return dict(latent_adapter='requires_grad False', latent_logstd='gradient masked')
