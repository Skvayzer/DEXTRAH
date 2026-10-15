import math
import unittest

import torch

from dextrah_lab.sgs import MANIPULATION, SuccessGuidedSampler, sgs_logits


def manipulation(n=8, **kw):
    s = dict(MANIPULATION, num_configs=n)
    s.update(kw)
    return SuccessGuidedSampler(**s)


class KernelTests(unittest.TestCase):
    def test_project_page_weights(self):
        # sgs-rl.github.io: one nut-and-bolt configuration at 0/64, 27/64 and
        # 59/64 successes is shown with weights 0.14, 0.99 and 0.73, i.e.
        # softmax(l/T) relative to the kernel mode (p = t = .5).
        p = torch.tensor([0., 27/64, 59/64, .5], dtype=torch.float64)
        l = sgs_logits(p, .5, 1., 1e-4)/2.
        rel = torch.exp(l[:3]-l[3])
        self.assertTrue(torch.allclose(rel, torch.tensor([.14, .99, .73], dtype=torch.float64), atol=.005), rel)

    def test_formula_and_floor(self):
        # w = (p+eps)^(kt) (1-p+eps)^(k(1-t)); l = log(max(w, eps)+eps)  (Sec. 3.2, App. B.2)
        t, k, eps = .66, 5., 1e-8
        for p in (0., .1, .5, .66, .9, 1.):
            w = (p+eps)**(k*t)*(1-p+eps)**(k*(1-t))
            expected = math.log(max(w, eps)+eps)
            self.assertAlmostEqual(float(sgs_logits(torch.tensor(p, dtype=torch.float64), t, k, eps)), expected, places=10)
        # The floor bounds the logits of p in {0, 1} from below.
        self.assertGreaterEqual(float(sgs_logits(torch.tensor(0.), .5, 10., 1e-4)), math.log(2e-4)-1e-6)

    def test_mode_at_target(self):
        p = torch.linspace(0, 1, 1001, dtype=torch.float64)
        for t in (.33, .5, .66):
            self.assertAlmostEqual(float(p[sgs_logits(p, t, 5., 1e-8).argmax()]), t, delta=1e-3)

    def test_kappa_zero_is_uniform(self):
        p = torch.rand(100)
        l = sgs_logits(p, .5, 0., 1e-4)
        self.assertTrue(torch.allclose(l, l[0].expand_as(l)))


class SamplerTests(unittest.TestCase):
    def test_starts_uniform(self):
        s = manipulation(5)
        self.assertTrue(torch.allclose(s.probabilities(), torch.full((5,), .2)))

    def test_minimum_temperature_one(self):
        s = manipulation(4, temperature=.25)
        s.record(torch.tensor([0]*50), torch.ones(50, dtype=torch.bool))
        ref = manipulation(4, temperature=1.)
        ref.record(torch.tensor([0]*50), torch.ones(50, dtype=torch.bool))
        self.assertTrue(torch.allclose(s.probabilities(), ref.probabilities()))

    def test_success_rate_is_window_mean(self):
        s = manipulation(3, history=4)
        s.record(torch.tensor([0]), torch.tensor([True]))
        # One success in a window of four (the window starts as failures).
        self.assertAlmostEqual(float(s.success_rate()[0]), .25)
        s.record(torch.tensor([0, 0, 0, 0]), torch.tensor([False, False, True, True]))
        # Last four outcomes: F F T T (the first success has been overwritten).
        self.assertAlmostEqual(float(s.success_rate()[0]), .5)
        self.assertEqual(int(s.count[0]), 5)

    def test_duplicates_and_overflow_in_one_call(self):
        s = manipulation(2, history=3)
        ids = torch.tensor([1, 0, 1, 1, 1, 1])
        ok = torch.tensor([False, True, False, True, True, True])
        s.record(ids, ok)
        # Config 1 got F F T T T in call order: the newest three are T T T.
        self.assertAlmostEqual(float(s.success_rate()[1]), 1.)
        self.assertAlmostEqual(float(s.success_rate()[0]), 1/3)
        self.assertEqual(s.count.tolist(), [1, 5])
        self.assertEqual(s.head.tolist(), [1, 5 % 3])
        # Continuing writes wrap correctly.
        s.record(torch.tensor([1]), torch.tensor([False]))
        self.assertAlmostEqual(float(s.success_rate()[1]), 2/3)

    def test_sequential_equals_batched(self):
        torch.manual_seed(0)
        ids = torch.randint(0, 5, (400,))
        ok = torch.rand(400) < .4
        a, b = manipulation(5, history=7), manipulation(5, history=7)
        a.record(ids, ok)
        for i, o in zip(ids, ok):
            b.record(i[None], o[None])
        self.assertTrue(torch.equal(a.success_rate(), b.success_rate()))
        self.assertTrue(torch.equal(a.head, b.head) and torch.equal(a.count, b.count))

    def test_sampling_follows_softmax(self):
        s = manipulation(3, history=10)
        s.record(torch.tensor([0]*5+[1]*10), torch.tensor([True]*5+[True]*10))   # p = .5, 1, 0
        probs = s.probabilities()
        g = torch.Generator().manual_seed(1)
        draws = s.sample(200000, generator=g)
        freq = torch.bincount(draws, minlength=3).float()/len(draws)
        self.assertTrue(torch.allclose(freq, probs, atol=.005), (freq, probs))
        self.assertGreater(float(probs[0]), float(probs[1]))
        self.assertGreater(float(probs[2]), 0.)    # unsolved configurations keep non-zero mass

    def test_unavailable_get_zero_probability(self):
        s = manipulation(4)
        s.available[1:3] = False
        p = s.probabilities()
        self.assertEqual(p[1:3].tolist(), [0., 0.])
        self.assertTrue(set(s.sample(1000).tolist()) <= {0, 3})

    def test_state_roundtrip_and_retune(self):
        s = manipulation(4, history=5)
        s.record(torch.tensor([0, 1, 1, 3]), torch.tensor([True, False, True, True]))
        s.available[2] = False
        r = manipulation(4, history=5, kappa=5.)       # kernel retuned on resume
        r.load_state_dict(s.state_dict())
        self.assertTrue(torch.equal(r.outcomes, s.outcomes) and torch.equal(r.available, s.available))
        with self.assertRaises(ValueError):
            manipulation(4, history=6).load_state_dict(s.state_dict())


if __name__ == '__main__':
    unittest.main()
