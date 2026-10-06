import unittest
from types import SimpleNamespace
import torch
from torch import nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from dextrah_lab.wholebody.carry_training import install_critic_warmup, reset_value_normalizers


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_mean_std = RunningMeanStd((1,))
        self.running_mean_std = RunningMeanStd((3,))
        self.layer = nn.Linear(3, 1)


class CarryTrainingTests(unittest.TestCase):
    def test_only_value_normalizers_reset(self):
        m = Model()
        for rms in (m.value_mean_std, m.running_mean_std):
            rms.running_mean.fill_(5.); rms.running_var.fill_(9.); rms.count.fill_(1e9)
        self.assertEqual(reset_value_normalizers(m), ['value_mean_std'])
        self.assertEqual(float(m.value_mean_std.count), 1.)
        self.assertEqual(float(m.value_mean_std.running_mean), 0.)
        self.assertEqual(float(m.running_mean_std.count), 1e9)

    def test_critic_warmup_skips_actor_steps(self):
        p = nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([p], lr=1.)
        algo = SimpleNamespace(optimizer=opt, epoch_num=1)
        install_critic_warmup(algo, 2)
        for epoch, expected in ((1, 1.), (2, 1.), (3, 0.)):
            algo.epoch_num = epoch
            p.grad = torch.ones(1)
            opt.step()
            self.assertEqual(float(p), expected)


if __name__ == '__main__':
    unittest.main()
