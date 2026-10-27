"""Hand-object world model pieces (dextrah_lab/neurosim), CPU only, synthetic data."""
import math
import unittest

import torch

from dextrah_lab.neurosim.data import Layout, build_tensors, gather, split_envs
from dextrah_lab.neurosim.evaluate import auc, loss_prediction, open_loop
from dextrah_lab.neurosim.geometry import matrix_to_rot6, quat_to_matrix, rot6_to_matrix, rotation_angle, so3_exp, so3_log
from dextrah_lab.neurosim.model import HandObjectWorldModel, window_loss


def synthetic(T=80, N=6, H=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    q = torch.nn.functional.normalize(r(T, N, 4), dim=-1)
    done = torch.zeros(T, N, 1)
    end = torch.zeros(T, N, 1)
    done[30, 0] = 1; end[30, 0] = 2          # env 0: drop ends the episode at row 30
    done[50, 1] = 1; end[50, 1] = 1          # env 1: timeout at row 50
    data = dict(obj_pos=.01*r(T, N, 3), obj_quat=q, obj_linvel=r(T, N, 3), obj_angvel=r(T, N, 3), hand_q=r(T, N, H),
                hand_qd=r(T, N, H), hand_target=r(T, N, 6), palm_linvel=r(T, N, 3), palm_angvel=r(T, N, 3),
                gravity=torch.nn.functional.normalize(r(T, N, 3), dim=-1), table_pos=r(T, N, 3), table_parked=torch.zeros(T, N, 1),
                object_force=r(T, N, 3), finger_force=r(T, N, 5).abs(), palm_force=r(T, N, 1).abs(), table_force=torch.zeros(T, N, 1),
                ground_force=torch.zeros(T, N, 1), supported=torch.ones(T, N, 1), mass=torch.full((T, N, 1), .2),
                com_offset=torch.zeros(T, N, 3), done=done, end=end)
    context = dict(bps=r(N, 128), asset=torch.tensor([0, 1, 2, 3, 4, 5]), dt=1/60)
    return data, context


class NeurosimTests(unittest.TestCase):
    def test_geometry_round_trips(self):
        q = torch.nn.functional.normalize(torch.randn(50, 4), dim=-1)
        R = quat_to_matrix(q)
        self.assertTrue(torch.allclose(R@R.transpose(-1, -2), torch.eye(3).expand(50, 3, 3), atol=1e-5))
        self.assertTrue(torch.allclose(rot6_to_matrix(matrix_to_rot6(R)), R, atol=1e-5))
        v = torch.randn(50, 3)*.9
        self.assertTrue(torch.allclose(so3_log(so3_exp(v)), v, atol=1e-4))
        self.assertTrue(torch.allclose(so3_exp(torch.zeros(2, 3)), torch.eye(3).expand(2, 3, 3)))
        self.assertTrue(torch.allclose(rotation_angle(so3_exp(v), torch.eye(3).expand(50, 3, 3)), v.norm(dim=-1), atol=1e-4))

    def test_tensors_and_masks(self):
        data, context = synthetic()
        t = build_tensors(data, context)
        self.assertEqual(t['state'].shape[-1], 15+2*4)
        self.assertEqual(float(t['lost'][30, 0]), 1.)      # drop
        self.assertEqual(float(t['lost'][50, 1]), 0.)      # timeout is not a loss
        w = gather(t, torch.tensor([27, 27, 47]), torch.tensor([0, 2, 1]), 8)
        # env 0 from row 27: steps 1..2 continue, step 3 (row 30) ends with a drop, then nothing counts
        self.assertEqual(w['has_next'][0].tolist(), [True, True, False, False, False, False, False, False])
        self.assertEqual(w['alive'][0].tolist(), [True, True, True, False, False, False, False, False])
        self.assertTrue(bool(w['has_next'][1].all()))
        self.assertEqual(w['alive'][2].tolist(), [True, True, True, False, False, False, False, False])
        self.assertEqual(float(w['lost'][0, 3]), 1.)
        self.assertEqual(w['context'].shape[-1], 128+4)
        train, test = split_envs(context['asset'], .34, 0)
        self.assertEqual(int(test.sum()), 2)
        self.assertFalse(bool((train & test).any()))

    def test_model_loss_backward_and_eval(self):
        data, context = synthetic()
        t = build_tensors(data, context)
        t.pop('asset')
        L = Layout(hand=4)
        model = HandObjectWorldModel(L, hidden=(64, 64))
        model.dt = 1/60
        rows = (torch.randint(1, 80, (400,)), torch.randint(0, 6, (400,)))
        model.fit_normalizers(t, rows)
        w = gather(t, torch.tensor([0, 10, 27]), torch.tensor([3, 4, 0]), 8)
        loss, logs = window_loss(model, w, 8)
        self.assertTrue(math.isfinite(float(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        # zero-initialised heads: the first prediction is the mean one-step change (near persistence)
        xs, Rs, _, _ = model.rollout(w, 1)
        self.assertTrue(torch.allclose(Rs[0], w['rot'][:, 0]@so3_exp(model.norm_delta.mean[3:6].expand(3, 3)), atol=1e-4))
        res = open_loop(model, t, torch.arange(6), horizons=(1, 8), samples=64)
        self.assertIn(1, res)
        self.assertIn('pos_mm', res[1])
        lp = loss_prediction(model, t, torch.tensor([0, 1, 2]), horizon=10, random_samples=64, bins=4)
        self.assertEqual(lp['events'], 1)

    def test_auc(self):
        self.assertAlmostEqual(auc(torch.tensor([.1, .4, .35, .8]), torch.tensor([0, 0, 1, 1])), .75)
        self.assertAlmostEqual(auc(torch.tensor([1., 1., 1., 1.]), torch.tensor([0, 1, 0, 1])), .5)
        self.assertTrue(math.isnan(auc(torch.tensor([.1, .2]), torch.tensor([0, 0]))))


if __name__ == '__main__':
    unittest.main()
