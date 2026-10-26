"""Opt-in per-epoch breakdown of rl_games' policy KL (--kl-diagnostics; logging only).

Which action dimensions (64 latent vs 6 fingers) and how many samples make the reported KL, and
the frozen-SONIC latent adapter's input and last-hidden peaks (its 994 body inputs are raw SONIC
proprioception, not normalised). Wraps torch_ext.policy_kl with the identical formula; the
returned value and every gradient are unchanged.
"""
import torch
from rl_games.algos_torch import torch_ext

LATENT_DIM = 64


class KlDiagnostics:
    def __init__(self, algo, latent_dim=LATENT_DIM, feature_dim=512):
        self.latent_dim, self.feature_dim = latent_dim, feature_dim
        self.reset()
        original = torch_ext.policy_kl

        def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
            out = original(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce)
            with torch.no_grad():
                self.record(per_dim_kl(p0_mu, p0_sigma, p1_mu, p1_sigma))
            return out

        torch_ext.policy_kl = policy_kl
        self._original = original
        adapter = getattr(algo.model.a2c_network, 'latent_adapter', None)
        self.hooks = []
        if adapter is not None:
            self.hooks.append(adapter.register_forward_pre_hook(self._adapter_input))
            self.hooks.append(adapter[-2].register_forward_hook(self._adapter_hidden))
        algo._kl_diagnostics = self

    def reset(self):
        self.samples, self.latent_sum, self.finger_sum, self.over_one, self.over_ten = 0, 0., 0., 0, 0
        self.max_sample = 0.
        self.input_max, self.hidden_max = 0., 0.

    def record(self, kl):
        latent, fingers = kl[..., :self.latent_dim].sum(-1), kl[..., self.latent_dim:].sum(-1)
        total = latent+fingers
        stats = torch.stack((latent.sum(), fingers.sum(), (total > 1).sum(), (total > 10).sum(), total.max())).tolist()
        self.samples += total.numel()
        self.latent_sum += stats[0]
        self.finger_sum += stats[1]
        self.over_one += int(stats[2])
        self.over_ten += int(stats[3])
        self.max_sample = max(self.max_sample, stats[4])

    def _adapter_input(self, module, args):
        x = args[0]
        self.input_max = max(self.input_max, float(x[..., self.feature_dim:].detach().abs().max()))

    def _adapter_hidden(self, module, args, output):
        self.hidden_max = max(self.hidden_max, float(output.detach().norm(dim=-1).max()))

    def scalars(self):
        n = max(self.samples, 1)
        return {'kl_latent_mean': self.latent_sum/n, 'kl_finger_mean': self.finger_sum/n,
                'kl_sample_max': self.max_sample, 'kl_samples_over_1_fraction': self.over_one/n,
                'kl_samples_over_10_fraction': self.over_ten/n,
                'adapter_body_input_abs_max': self.input_max, 'adapter_last_hidden_norm_max': self.hidden_max}

    def close(self):
        torch_ext.policy_kl = self._original
        for hook in self.hooks:
            hook.remove()


def per_dim_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    """torch_ext.policy_kl before its sum over action dimensions."""
    c1 = torch.log(p1_sigma/p0_sigma+1e-5)
    c2 = (p0_sigma**2+(p1_mu-p0_mu)**2)/(2.0*(p1_sigma**2+1e-5))
    return c1+c2-.5
