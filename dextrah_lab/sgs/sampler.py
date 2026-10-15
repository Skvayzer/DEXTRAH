"""Success Guided Sampling (SGS).

Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots: "A Balanced Data
Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot
Control", CoRL 2026, arXiv:2610.12465 (https://sgs-rl.github.io).

Paper -> code (Sec. 3.2, Appendix B.2):

* a fixed set of N task configurations chosen before training;
* per configuration the last H Boolean outcomes in a circular buffer, success
  rate p_i = their mean. The window starts as H failures: the project page
  shows a configuration at "0 of 64 successes" at iteration 0, and its
  displayed weights (0.14 / 0.99 / 0.73 at 0/64, 27/64, 59/64) are exactly
  softmax(l/T) of this kernel with t=.5, kappa=1, eps=1e-4, T=2 relative to
  the mode, with p = successes / window size;
* Beta(1+kappa t, 1+kappa(1-t)) kernel in mode-concentration form
      w_i = (p_i+eps)^(kappa t) (1-p_i+eps)^(kappa(1-t))
      w~_i = max(w_i, eps),   l_i = log(w~_i + eps)         (Appendix B.2 floor)
* the next configuration is drawn i.i.d. from softmax(l_i / T_eff),
  T_eff = max(T, 1), every time an episode ends.
"""
import torch

# Sec. 3.2 "In practice ...": manipulation and locomotion settings.
MANIPULATION = dict(num_configs=32768, history=100, temperature=2., target=.5, kappa=1., eps=1e-4)
LOCOMOTION = dict(num_configs=104000, history=100, temperature=2., target=.66, kappa=5., eps=1e-8)


def sgs_logits(p, target, kappa, eps):
    """l_i = log(max(w_i, eps) + eps) for success rates p (any shape)."""
    w = (p+eps).pow(kappa*target)*(1.-p+eps).pow(kappa*(1.-target))
    return torch.log(torch.clamp_min(w, eps)+eps)


class SuccessGuidedSampler:
    """Outcome windows, kernel and i.i.d. sampling over N fixed configurations.

    ``available`` masks configurations that do not exist yet (e.g. a reset
    family still being collected); they get zero probability.
    """

    def __init__(self, num_configs, history=100, temperature=2., target=.5, kappa=1., eps=1e-4, device='cpu'):
        if num_configs < 1 or history < 1:
            raise ValueError('SGS needs at least one configuration and a non-empty history')
        if not 0. <= target <= 1. or kappa < 0. or eps <= 0. or temperature <= 0.:
            raise ValueError(f'Invalid SGS kernel t={target} kappa={kappa} eps={eps} T={temperature}')
        self.num_configs, self.history = int(num_configs), int(history)
        self.temperature, self.target, self.kappa, self.eps = float(temperature), float(target), float(kappa), float(eps)
        self.device = torch.device(device)
        self.outcomes = torch.zeros(self.num_configs, self.history, dtype=torch.bool, device=self.device)
        self.head = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)
        self.count = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)   # all outcomes ever recorded
        self.available = torch.ones(self.num_configs, dtype=torch.bool, device=self.device)

    @property
    def effective_temperature(self):
        return max(self.temperature, 1.)

    def success_rate(self):
        return self.outcomes.float().mean(-1)

    def logits(self):
        return sgs_logits(self.success_rate(), self.target, self.kappa, self.eps)

    def probabilities(self):
        logits = (self.logits()/self.effective_temperature).masked_fill(~self.available, float('-inf'))
        if not bool(self.available.any()):
            raise RuntimeError('SGS has no available configuration')
        return torch.softmax(logits, 0)

    def sample(self, n, generator=None):
        """n i.i.d. configuration ids."""
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.multinomial(self.probabilities(), n, replacement=True, generator=generator)

    def record(self, ids, success):
        """Append one outcome per entry; a configuration may appear several times
        (several envs finishing it in the same step): all are appended."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(-1)
        success = torch.as_tensor(success, dtype=torch.bool, device=self.device).reshape(-1)
        if ids.numel() != success.numel():
            raise ValueError('One outcome per configuration id')
        if ids.numel() == 0:
            return
        if bool(((ids < 0) | (ids >= self.num_configs)).any()):
            raise IndexError('SGS configuration id out of range')
        order = torch.argsort(ids, stable=True)
        ids, success = ids[order], success[order]
        unique, counts = torch.unique_consecutive(ids, return_counts=True)
        starts = torch.cumsum(counts, 0)-counts
        rank = torch.arange(ids.numel(), device=self.device)-torch.repeat_interleave(starts, counts)
        total = torch.repeat_interleave(counts, counts)
        keep = rank >= total-self.history                     # only the newest H of a group survive
        position = (self.head[ids]+rank) % self.history
        self.outcomes[ids[keep], position[keep]] = success[keep]
        self.head[unique] = (self.head[unique]+counts) % self.history
        self.count[unique] += counts

    def settings(self):
        return dict(num_configs=self.num_configs, history=self.history, temperature=self.temperature,
                    target=self.target, kappa=self.kappa, eps=self.eps)

    def state_dict(self):
        return dict(settings=self.settings(), outcomes=self.outcomes.cpu(), head=self.head.cpu(),
                    count=self.count.cpu(), available=self.available.cpu())

    def load_state_dict(self, state):
        """Outcome windows must match in shape; the kernel may be retuned on resume."""
        saved = state['settings']
        if (saved['num_configs'], saved['history']) != (self.num_configs, self.history):
            raise ValueError(f'SGS window shape changed: saved {saved}, now {self.settings()}')
        for key in ('outcomes', 'head', 'count', 'available'):
            getattr(self, key).copy_(state[key].to(self.device))
