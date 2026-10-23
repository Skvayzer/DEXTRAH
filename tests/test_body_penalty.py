"""Opt-in SONIC action / fall penalties (dextrah_lab/wholebody/body_penalty.py)."""
from types import SimpleNamespace
import unittest

import torch

from dextrah_lab.wholebody.body_penalty import BodyPenalty, sonic_action_excess


class BodyPenaltyTests(unittest.TestCase):
    def test_excess_and_penalty(self):
        a = torch.zeros(3, 29)
        a[1, 28] = 21.4          # wrist pinned at its limit
        a[2, 27] = -9.
        self.assertTrue(torch.allclose(sonic_action_excess(a, 8.), torch.tensor([0., 13.4, 1.])))
        env = SimpleNamespace(_last_decoded_sonic_action=a, _body_fallen=torch.tensor([False, False, True]))
        reward, terms, logs = BodyPenalty(env, action_scale=.05, action_bound=8., fall_penalty=100.)()
        self.assertTrue(torch.allclose(reward, torch.tensor([0., -.67, -100.05])))
        self.assertTrue(torch.equal(logs['sonic_action_over_bound'], torch.tensor([0., 1., 1.])))
        with self.assertRaises(ValueError):
            BodyPenalty(env, action_scale=-1.)


if __name__ == '__main__':
    unittest.main()
