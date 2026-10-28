import unittest
from types import SimpleNamespace
import torch
from torch import nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from dextrah_lab.wholebody.carry_training import install_critic_warmup, install_delayed_best, reset_value_normalizers


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

    def test_critic_warmup_counts_from_the_resume_epoch(self):
        p = nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([p], lr=1.)
        algo = SimpleNamespace(optimizer=opt, epoch_num=2487)
        install_critic_warmup(algo, 2, start=2487)
        for epoch, expected in ((2488, 1.), (2489, 1.), (2490, 0.)):
            algo.epoch_num = epoch
            p.grad = torch.ones(1)
            opt.step()
            self.assertEqual(float(p), expected)

    def test_critic_warmup_trains_policy_value_head_only(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk, self.mu, self.value = nn.Linear(2, 2), nn.Linear(2, 1), nn.Linear(2, 1)
        model = nn.Module(); model.a2c_network = Net()
        opt = torch.optim.SGD(model.parameters(), lr=1.)
        algo = SimpleNamespace(optimizer=opt, epoch_num=1, model=model)
        self.assertEqual(install_critic_warmup(algo, 2)['warmup_value_head_tensors'], 2)
        net = model.a2c_network
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        for n, p in model.named_parameters():
            changed = not torch.equal(p, before[n])
            self.assertEqual(changed, '.value.' in n, n)
        algo.epoch_num = 3
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        self.assertFalse(torch.equal(net.trunk.weight, before['a2c_network.trunk.weight']))
        self.assertFalse(torch.equal(net.mu.bias, before['a2c_network.mu.bias']))

    def test_delayed_best_blocks_then_resets(self):
        class Algo:
            epoch_num = 100
            last_mean_rewards = 5e4
            def update_epoch(self):
                self.epoch_num += 1
                return self.epoch_num
        algo = Algo()
        install_delayed_best(algo, 3)
        self.assertEqual(algo.last_mean_rewards, float('inf'))
        algo.update_epoch(); algo.update_epoch()
        self.assertEqual(algo.last_mean_rewards, float('inf'))
        algo.update_epoch()
        self.assertEqual(algo.last_mean_rewards, -1000000000)


if __name__ == '__main__':
    unittest.main()
