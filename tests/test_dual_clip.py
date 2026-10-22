"""Dual-clip PPO actor loss equals rl_games' loss except where it caps A < 0, large-ratio terms."""
import unittest

import torch

from dextrah_lab.mj.compat import install

install()
from rl_games.common.common_losses import actor_loss  # noqa: E402
from dextrah_lab.wholebody.dual_clip import dual_clip_actor_loss  # noqa: E402


class DualClipTests(unittest.TestCase):
    def test_matches_ppo_except_capped_negative_advantages(self):
        g = torch.Generator().manual_seed(0)
        old, new = torch.randn(1000, generator=g)*3, torch.randn(1000, generator=g)*3
        adv = torch.randn(1000, generator=g)
        ref = actor_loss(old, new, adv, True, .1)
        out = dual_clip_actor_loss(3.)(old, new, adv, True, .1)
        capped = (adv < 0) & (ref > -3*adv)
        self.assertTrue(bool(capped.any()))
        self.assertTrue(torch.equal(out[~capped], ref[~capped]))
        self.assertTrue(torch.allclose(out[capped], -3*adv[capped]))
        self.assertTrue(bool(torch.isfinite(dual_clip_actor_loss(3.)(torch.tensor([60.]), torch.tensor([-60.]),
                                                                     torch.tensor([-1.]), True, .1)).all()))


if __name__ == '__main__':
    unittest.main()
