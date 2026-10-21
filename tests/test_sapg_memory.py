"""Lean SAPG batches (dextrah_lab/wholebody/sapg_memory.py) == rl_games' original."""
import copy
from types import SimpleNamespace
import unittest

import torch

from dextrah_lab.mj.compat import install

install()
from rl_games.common.a2c_common import A2CBase  # noqa: E402
from dextrah_lab.wholebody import sapg_memory  # noqa: E402


def fake_algo(N=24, blocks=6, horizon=4, E=3, layers=1):
    g = torch.Generator().manual_seed(0)
    W = torch.randn(9+E, 1, generator=g)
    algo = SimpleNamespace(num_actors=N, intr_coef_block_size=N//blocks, config=dict(off_policy_ratio=1.0),
                           multi_gpu=False, use_others_experience='lf', horizon_length=horizon, gamma=.99,
                           intr_reward_coef_embd=torch.randn(N, E, generator=g),
                           intr_reward_coef=torch.randn(N, generator=g))
    algo.get_values = lambda obs, rnn_states=None: obs['states'] @ W   # deterministic critic stand-in
    B = N*horizon
    batch = dict(obses=torch.randn(B, 7+E, generator=g), states=torch.randn(B, 9+E, generator=g),
                 actions=torch.randn(B, 2, generator=g), neglogpacs=torch.randn(B, generator=g),
                 dones=torch.rand(B, generator=g) < .1, mus=torch.randn(B, 2, generator=g), sigmas=torch.rand(B, 2, generator=g),
                 returns=torch.randn(B, 1, generator=g), values=torch.randn(B, 1, generator=g),
                 rnn_states=[torch.randn(layers, N, 5, generator=g), torch.randn(layers, N, 5, generator=g)],
                 played_frames=B, step_time=0.1)
    for k in ('obses', 'states'):
        batch[k][:, -E:] = algo.intr_reward_coef_embd.repeat_interleave(horizon, 0)
    extras = dict(rewards=torch.randn(horizon, N, 1, generator=g), obs=torch.randn(horizon, N, 7+E, generator=g),
                  last_obs=dict(obs=torch.randn(N, 7+E, generator=g), states=torch.randn(N, 9+E, generator=g)),
                  states=torch.randn(horizon, N, 9+E, generator=g), dones=(torch.rand(horizon, N, generator=g) < .1).float(),
                  last_dones=(torch.rand(N, generator=g) < .1).float(), rnn_states=None, last_rnn_states=None,
                  mb_intr_rewards=None, mb_extr_rewards=None)
    return algo, batch, extras


class LeanAugmentTests(unittest.TestCase):
    def check(self, repeat_idxs, layers=1):
        algo, batch, extras = fake_algo(layers=layers)
        ref = A2CBase.augment_batch_for_mixed_expl(algo, copy.deepcopy(batch), extras, list(repeat_idxs))
        algo2, batch2, extras2 = fake_algo(layers=layers)
        out = sapg_memory.lean_augment(algo2, batch2, extras2, list(repeat_idxs))
        self.assertEqual(set(out), set(ref))
        for k, v in ref.items():
            if k == 'rnn_states':
                for a, b in zip(out[k], v):
                    self.assertTrue(torch.equal(a, b), k)
            elif torch.is_tensor(v):
                self.assertTrue(torch.equal(out[k], v), k)
            else:
                self.assertEqual(out[k], v)
        self.assertEqual(batch2, {})                       # consumed rollout dict emptied
        for k in ('obs', 'states'):                        # extras restored as in the original
            self.assertTrue(torch.equal(extras2[k], extras[k]))
        self.assertTrue(torch.equal(extras2['last_obs']['obs'], extras['last_obs']['obs']))

    def test_equal_to_original(self):
        for idxs in ([0, 1], [0, 3], [0, 5], [0, 2, 4]):
            self.check(idxs)

    def test_multilayer_rnn_states_follow_original(self):
        self.check([0, 2], layers=2)


if __name__ == '__main__':
    unittest.main()
