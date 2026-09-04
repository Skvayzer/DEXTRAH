import torch

from dextrah_lab.adept.actor_bc import (
    RecurrentPolicyState,
    checkpoint_weights,
    policy_distribution,
)


class _FeedForwardPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def is_rnn(self):
        return False

    def forward(self, batch):
        count = batch["obs"].shape[0]
        return {
            "mus": self.mean.expand(count, -1),
            "sigmas": torch.ones(count, 2),
        }


def test_policy_distribution_preserves_actor_gradients():
    model = _FeedForwardPolicy()
    mean, std = policy_distribution(
        model,
        torch.zeros(3, 4),
        torch.zeros(3, 2),
        RecurrentPolicyState(),
        train=True,
    )
    (mean + std).sum().backward()
    torch.testing.assert_close(model.mean.grad, torch.full((2,), 3.0))


def test_checkpoint_weights_accepts_rank_wrapped_save():
    weights = {"layer": torch.ones(1)}
    assert checkpoint_weights({"model": weights}) is weights
    assert checkpoint_weights({0: {"model": weights}}) is weights
