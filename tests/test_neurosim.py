"""Hand-object world model pieces (dextrah_lab/neurosim), CPU only, synthetic data."""
import math
import unittest

import torch

from dextrah_lab.neurosim.data import EXO_DIM, Layout, build_tensors, gather, phase_bucket, pre_event_mask, split_envs
from dextrah_lab.neurosim.evaluate import (auc, auc_ci, compare_finite_differences, drop_prediction_report, freeze_inputs,
                                           loss_prediction, open_loop, open_loop_smoothed)
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
                com_offset=torch.zeros(T, N, 3), done=done, end=end, arm_now=r(T, N, 7), command=r(T, N, 3))
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

    def test_rotation_loss_gradient_bounded_near_pi(self):
        from dextrah_lab.neurosim.model import state_error
        model = HandObjectWorldModel(Layout(hand=4), hidden=(16,))
        R = torch.eye(3).expand(4, 3, 3).clone()
        v = torch.tensor([[0., 0., math.pi-1e-4], [0., 0., 3.], [0., 0., .1], [1e-6, 0., 0.]], requires_grad=True)
        x = torch.zeros(4, 15+8)
        err = state_error(model, x, so3_exp(v), x, R)['rot']
        err.sum().backward()
        self.assertTrue(torch.isfinite(v.grad).all() and float(v.grad.abs().max()) < 10.)
        self.assertAlmostEqual(float(err[2]), 2*(1-math.cos(.1))/float(model.norm_state.std[3:9].mean())**2, places=5)

    def test_review_fixes(self):
        data, context = synthetic(T=100, N=6)
        data['end'][30, 0] = 4                                         # make env 0's end a failed grasp check
        data['done'][70, 2] = 1; data['end'][70, 2] = 2               # and add a drop at env 2
        t = build_tensors(data, context, exo_mode='pre', lowpass_tau=.3, labels='drop')
        self.assertEqual(t['exo'].shape[-1], EXO_DIM['pre'])
        self.assertEqual(t['state'].shape[-1], 15+2*4+6)
        self.assertEqual(float(t['lost'][30, 0]), 0.)                  # censored, not a loss
        self.assertTrue(bool(t['censor'][30, 0]))
        self.assertEqual(float(t['lost'][70, 2]), 1.)
        lp = t['state'][..., -6:]
        self.assertTrue(torch.allclose(lp[30, 0], t['state'][30, 0, 9:15]))   # restarts at an episode start
        m = pre_event_mask(t, 8)
        self.assertEqual(torch.nonzero(m[:, 2]).flatten().tolist(), list(range(62, 70)))
        t.pop('asset')
        L = Layout(hand=4, lowpass=6, exo=EXO_DIM['pre'])
        model = HandObjectWorldModel(L, hidden=(32, 32), exo_mode='pre')
        model.fit_normalizers(t, (torch.randint(1, 100, (300,)), torch.randint(0, 6, (300,))))
        w = gather(t, torch.tensor([25, 65, 10]), torch.tensor([0, 2, 4]), 8)
        l1, logs1 = window_loss(model, w, 8)
        l2, _ = window_loss(model, w, 8, window_weight=torch.tensor([1., 1., 0.]))
        self.assertTrue(math.isfinite(float(l1)) and math.isfinite(float(l2)) and float(l1) != float(l2))
        self.assertIn('lowpass', logs1)
        fw = freeze_inputs(w, 'pre')
        self.assertTrue(torch.equal(fw['exo'][:, 5], fw['exo'][:, 1]))
        model.dt = 1/60
        sm = open_loop_smoothed(model, t, torch.arange(6), horizons=(8,), samples=64, generator=torch.Generator().manual_seed(0))
        self.assertIn(8, sm)

    def test_finite_difference_comparison(self):
        S, D = 5, 13
        g = torch.Generator().manual_seed(0)
        base = torch.randn(S, 1, generator=g)
        eff = torch.randn(S, 6, generator=g)
        sim = torch.cat((base, (base+torch.stack((eff, -eff), -1).reshape(S, 12))), 1).flatten()   # linear: +d, -d pairs
        r = compare_finite_differences(sim, 2*sim, S, D)                   # model: same direction, twice the size
        self.assertAlmostEqual(r['cosine_median'], 1., places=4)
        self.assertEqual(r['sign_agreement_clear'], 1.)
        self.assertAlmostEqual(r['magnitude_ratio_median'], 2., places=3)
        self.assertAlmostEqual(r['sim_antisymmetry'], 1., places=4)
        r2 = compare_finite_differences(sim, -sim, S, D)
        self.assertAlmostEqual(r2['cosine_median'], -1., places=4)
        self.assertEqual(r2['sign_agreement_clear'], 0.)

    def test_probabilistic_ensemble(self):
        from dextrah_lab.neurosim.data import combine
        from dextrah_lab.neurosim.ensemble import EnsembleMean, EnsembleWorldModel, ensemble_window_loss, uncertainty_report
        d1, c1 = synthetic(T=60, N=4, seed=1)
        d2, c2 = synthetic(T=60, N=4, seed=2)
        for d in (d1, d2):
            d['phase'] = torch.full((60, 4, 1), 2.); d['carry_t'] = torch.arange(60.)[:, None, None].expand(60, 4, 1).clone()
        c1['asset'], c2['asset'] = torch.arange(4), torch.arange(4, 8)
        t = combine([build_tensors(d1, c1, exo_mode='pre', lowpass_tau=.3, labels='drop'),
                     build_tensors(d2, c2, exo_mode='pre', lowpass_tau=.3, labels='drop')])
        self.assertEqual(t['state'].shape[1], 8)
        self.assertEqual(t['bps'].shape[0], 8)
        t.pop('asset')
        ens = EnsembleWorldModel(Layout(hand=4, lowpass=6, exo=EXO_DIM['pre']), members=3, hidden=(16, 16))
        ens.fit_normalizers(t, (torch.randint(1, 60, (200,)), torch.randint(0, 8, (200,))))
        w = gather(t, torch.tensor([3, 10, 20]), torch.tensor([0, 5, 7]), 6)
        loss, logs = ensemble_window_loss(ens, w, 6)
        self.assertTrue(math.isfinite(float(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in ens.parameters()))
        self.assertIn('nll', logs)
        mean = EnsembleMean(ens)
        xs, Rs, fs, ls = mean.rollout(w, 6)
        self.assertEqual(xs[0].shape, w['state'][:, 0].shape)
        self.assertTrue(torch.allclose(Rs[0]@Rs[0].transpose(-1, -2), torch.eye(3).expand(3, 3, 3), atol=1e-4))
        res = open_loop(mean, t, torch.arange(8), horizons=(1, 8), samples=32)
        self.assertIn(1, res)
        rep = uncertainty_report(ens, t, torch.arange(8), horizons=(1, 4), samples=64, generator=torch.Generator().manual_seed(0))
        self.assertIn('aleatoric_pos', rep)

    def test_counterfactual_pairs(self):
        import tempfile
        from pathlib import Path
        from dextrah_lab.neurosim.pairs import difference_scales, load_pairs, pair_loss
        G, K, T = 3, 4, 7                                   # rows t0-2 .. t0+H with H = 4
        data, _ = synthetic(N=G*K)
        data = {k: v[:T].clone() for k, v in data.items()}
        data['done'][:] = 0; data['end'][:] = 0
        model = HandObjectWorldModel(Layout(hand=4), hidden=(32, 32))
        model.dt = 1/60
        with tempfile.TemporaryDirectory() as d:
            for rnd in range(2):
                raw = {k: v.clone() for k, v in data.items()}
                for g in range(G):                          # copies identical to their baseline
                    for c in range(1, K):
                        for k_ in raw:
                            raw[k_][:, g*K+c] = raw[k_][:, g*K]
                raw.update(group_end=torch.zeros(G, dtype=torch.bool), bps=torch.randn(G*K, 128),
                           asset=torch.arange(G*K) % 5)
                torch.save(raw, Path(d)/f'round_{rnd:03d}.pt')
            tp = load_pairs(d, exo_mode='post', lowpass_tau=None, labels='lost', dt=1/60, copies=K)
        self.assertEqual(tp['state'].shape[:2], (T, 2*G*K))
        self.assertEqual(tp['group_end'].shape, (2*G,))
        t = {k: v for k, v in tp.items() if k not in ('copies', 'group_end', 'asset')}
        rows = (torch.randint(1, T, (200,)), torch.randint(0, 2*G*K, (200,)))
        model.fit_normalizers(t, rows)
        groups = torch.arange(2*G)
        ss, sf = difference_scales(tp, groups, T-2, torch.full((model.L.state,), 1e-3), torch.full((model.L.forces,), 1e-3))
        self.assertTrue(torch.allclose(ss, torch.full_like(ss, 1e-3)))     # identical copies: only the floor
        # identical copies with identical actions: predicted and simulated differences are both zero
        loss, logs = pair_loss(model, tp, groups, T-2, ss, sf)
        self.assertLess(float(loss), 1e-8)
        # a perturbed copy: a non-zero, differentiable loss
        tp['state'][2:, 1] += .01
        tp['action'][2:, 1] += .3
        loss, logs = pair_loss(model, tp, groups[:1].repeat(16), T-2, ss, sf, generator=torch.Generator().manual_seed(0))
        self.assertTrue(math.isfinite(float(loss)) and float(loss) > 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_auc(self):
        self.assertAlmostEqual(auc(torch.tensor([.1, .4, .35, .8]), torch.tensor([0, 0, 1, 1])), .75)
        self.assertAlmostEqual(auc(torch.tensor([1., 1., 1., 1.]), torch.tensor([0, 1, 0, 1])), .5)
        self.assertTrue(math.isnan(auc(torch.tensor([.1, .2]), torch.tensor([0, 0]))))


if __name__ == '__main__':
    unittest.main()
