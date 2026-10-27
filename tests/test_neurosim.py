"""Hand-object world model pieces (dextrah_lab/neurosim), CPU only, synthetic data."""
import math
import unittest

import torch

from dextrah_lab.neurosim.data import Layout, build_tensors, gather, phase_bucket, split_envs
from dextrah_lab.neurosim.evaluate import auc, auc_ci, drop_prediction_report, freeze_inputs, loss_prediction, open_loop
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
        g = torch.Generator().manual_seed(0)
        v = torch.nn.functional.normalize(torch.randn(50, 3, generator=g), dim=-1)*3.*torch.rand(50, 1, generator=g)   # angles < pi
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

    def test_context_width_follows_bps(self):
        data, context = synthetic()
        context['bps'] = torch.randn(6, 132)               # the env's BPS block is 132-D
        t = build_tensors(data, context)
        t.pop('asset')
        model = HandObjectWorldModel(Layout(hand=4, bps=132), hidden=(32, 32))
        model.fit_normalizers(t, (torch.randint(1, 80, (200,)), torch.randint(0, 6, (200,))))
        loss, _ = window_loss(model, gather(t, torch.tensor([0, 5]), torch.tensor([2, 3]), 4), 4)
        self.assertTrue(math.isfinite(float(loss)))

    def test_phase_buckets_and_frozen_inputs(self):
        b = phase_bucket(torch.tensor([0., 1., 2., 2., 2., 3.]), torch.tensor([.2, 1.5, 4., 10., 20., 30.]))
        self.assertEqual(b.tolist(), [0, 0, 1, 2, 3, 4])
        data, context = synthetic()
        t = build_tensors(data, context)
        w = freeze_inputs(gather(t, torch.tensor([3, 10]), torch.tensor([2, 4]), 6))
        self.assertTrue(torch.equal(w['action'][:, 1], w['action'][:, 6]))
        self.assertTrue(torch.equal(w['exo'][:, 3, 6:12], w['exo'][:, 1, 0:6]))   # after-step palm motion := before
        r = auc_ci(torch.tensor([.9, .8, .2, .1, .3]), torch.tensor([1, 1, 0, 0, 0]), n_boot=50)
        self.assertEqual(r['auc'], 1.)

    def test_drop_report_with_phases(self):
        data, context = synthetic(T=120, N=8)
        context['asset'] = torch.arange(8)
        T, N = 120, 8
        data['phase'] = torch.full((T, N, 1), 2.)
        data['carry_t'] = torch.arange(T, dtype=torch.float32)[:, None, None].expand(T, N, 1).clone()
        for env, row in ((2, 60), (3, 70), (5, 90)):
            data['done'][row, env] = 1; data['end'][row, env] = 2
        t = build_tensors(data, context)
        self.assertEqual(int(t['drop'].sum()), 4)                  # env 0 at row 30 + three
        self.assertEqual(int(t['censor'].sum()), 0)
        t.pop('asset')
        model = HandObjectWorldModel(Layout(hand=4), hidden=(32, 32))
        model.fit_normalizers(t, (torch.randint(1, T, (300,)), torch.randint(0, N, (300,))))
        rep = drop_prediction_report(model, t, torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7]), horizon=10,
                                     generator=torch.Generator().manual_seed(0), n_boot=20)
        self.assertIn('model_frozen', rep['random_negatives'])
        self.assertIn('logistic_phase', rep['phase_matched_negatives'])
        self.assertEqual(rep['random_negatives']['model_teacher']['positives'] >= 1, True)

    def test_auc(self):
        self.assertAlmostEqual(auc(torch.tensor([.1, .4, .35, .8]), torch.tensor([0, 0, 1, 1])), .75)
        self.assertAlmostEqual(auc(torch.tensor([1., 1., 1., 1.]), torch.tensor([0, 1, 0, 1])), .5)
        self.assertTrue(math.isnan(auc(torch.tensor([.1, .2]), torch.tensor([0, 0]))))


if __name__ == '__main__':
    unittest.main()
