"""Causal SONIC 20 ms history on the unchanged 60 Hz manipulation clock."""
import math
import torch


class TimedSonicHistory:
    DIMENSIONS = (3, 29, 29, 29, 3)

    def __init__(self, n, policy_dt=1/60, device='cpu'):
        if n < 1 or not 0 < policy_dt <= .02:
            raise ValueError('History requires a positive control interval <= 20 ms')
        self.policy_dt = policy_dt
        size = math.ceil(.18/policy_dt - 1e-9) + 1
        self.buffers = [torch.zeros(n, size, d, device=device) for d in self.DIMENSIONS]
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)
        # Oldest-to-newest, same ten physical timestamps as the checkpoint.
        indices = size-1-torch.arange(9, -1, -1, device=device, dtype=torch.float64)*.02/policy_dt
        indices = torch.where((indices-indices.round()).abs() < 1e-9, indices.round(), indices)
        self.lo, self.hi = indices.floor().long(), indices.ceil().long()
        self.weight = (indices-self.lo).float()[None, :, None]

    def reset(self, ids=None):
        self.fresh[slice(None) if ids is None else ids] = True

    def initialize_fresh(self, *terms):
        self._check(terms)
        for buffer, term in zip(self.buffers, terms):
            buffer[self.fresh] = term[self.fresh, None, :]
        self.fresh[:] = False

    def _check(self, terms):
        if len(terms) != len(self.DIMENSIONS) or any(
                x.shape != (len(self.fresh), d) for x, d in zip(terms, self.DIMENSIONS)):
            raise ValueError('Expected base omega, q-q0, qd, executed action, gravity')

    def push(self, *terms):
        self._check(terms)
        for buffer, term in zip(self.buffers, terms):
            buffer[:, :-1] = buffer[:, 1:].clone()
            buffer[:, -1] = term
        self.initialize_fresh(*terms)
        return self.value()

    def value(self):
        parts = []
        for i, buffer in enumerate(self.buffers):
            past = buffer[:, self.lo]
            # Action is a piecewise-constant executed command, never a blend
            # of the previous and a command not yet issued at the sample time.
            sampled = past if i == 3 else torch.lerp(past, buffer[:, self.hi], self.weight)
            parts.append(sampled.flatten(1))
        return torch.cat(parts, -1)
