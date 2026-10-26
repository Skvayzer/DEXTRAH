"""Opt-in KL breakdown (dextrah_lab/wholebody/kl_diagnostics.py): logging only, CPU."""
import types
import unittest

import torch
from torch import nn
from rl_games.algos_torch import torch_ext

from dextrah_lab.wholebody.kl_diagnostics import KlDiagnostics


class KlDiagnosticsTests(unittest.TestCase):
    def test_breakdown_and_unchanged_kl(self):
        original = torch_ext.policy_kl
        adapter = nn.Sequential(nn.Linear(4+3, 5), nn.SiLU(), nn.Linear(5, 2))
        algo = types.SimpleNamespace(model=types.SimpleNamespace(a2c_network=types.SimpleNamespace(latent_adapter=adapter)))
        d = KlDiagnostics(algo, latent_dim=2, feature_dim=4)
        try:
            mu0, mu1, s = torch.zeros(3, 4), torch.zeros(3, 4), torch.full((3, 4), .2)
            mu1[0, 0] = 1.                                         # one sample, one latent dim
            mu1[1, 3] = .1                                         # one sample, one finger dim
            out = torch_ext.policy_kl(mu1, s, mu0, s)
            self.assertTrue(torch.allclose(out, original(mu1, s, mu0, s)))
            sc = d.scalars()
            self.assertAlmostEqual(sc['kl_latent_mean']+sc['kl_finger_mean'], float(out), places=4)
            self.assertGreater(sc['kl_latent_mean'], sc['kl_finger_mean'])
            self.assertAlmostEqual(sc['kl_samples_over_1_fraction'], 1/3, places=5)
            adapter(torch.cat((torch.zeros(2, 4), torch.tensor([[0., -7., 1.], [2., 0., 0.]])), -1))
            self.assertEqual(d.scalars()['adapter_body_input_abs_max'], 7.)
            self.assertGreater(d.scalars()['adapter_last_hidden_norm_max'], 0.)
            d.reset()
            self.assertEqual(d.scalars()['kl_sample_max'], 0.)
        finally:
            d.close()
        self.assertIs(torch_ext.policy_kl, original)


if __name__ == '__main__':
    unittest.main()
