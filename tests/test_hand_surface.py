"""Hand-to-surface observation: geometry against brute force, zero-column surgery."""
import math
import unittest

import torch

from dextrah_lab.mj.compat import install

install()
from rl_games.common.layers.recurrent import LSTMWithDones  # noqa: E402
from dextrah_lab.wholebody import hand_surface as hs  # noqa: E402
from dextrah_lab.wholebody.sapg_network import TaskBodyNormalizer  # noqa: E402
from dextrah_lab.wholebody.task_contract import BODY_EXTRA_DIM  # noqa: E402


def random_rotation(n, g):
    q = torch.randn(n, 4, generator=g)
    return hs.quat_to_mat(q)


def sample_box_surface(h, k, g):
    """k points on the surface of a box with half extents h (local frame)."""
    face = torch.randint(0, 6, (k,), generator=g)
    p = (torch.rand(k, 3, generator=g)*2-1)*h
    axis, sign = face//2, torch.where(face % 2 == 0, 1., -1.)
    p[torch.arange(k), axis] = sign*h[axis]
    return p


def sample_capsule_surface(r, half, k, g):
    u = torch.rand(k, generator=g)
    phi = torch.rand(k, generator=g)*2*math.pi
    # cylinder part
    z = (torch.rand(k, generator=g)*2-1)*half
    cyl = torch.stack((r*torch.cos(phi), r*torch.sin(phi), z), -1)
    d = torch.randn(k, 3, generator=g)
    d = d/d.norm(dim=-1, keepdim=True)*r
    caps = d+torch.stack((torch.zeros(k), torch.zeros(k), torch.where(d[:, 2] >= 0, half, -half)), -1)
    return torch.where((u < .6)[:, None], cyl, caps)


class GeometryTests(unittest.TestCase):
    def test_nearest_surface_matches_brute_force(self):
        g = torch.Generator().manual_seed(0)
        n, P, S, K = 6, 40, 3, 60000
        centre = torch.randn(n, S, 3, generator=g)*.05
        rot = random_rotation(n*S, g).reshape(n, S, 3, 3)
        size = torch.rand(n, S, 3, generator=g)*.04+.005
        is_box = torch.tensor([True, False, True])
        used = torch.ones(n, S, dtype=torch.bool)
        used[0, 2] = False
        points = torch.randn(n, P, 3, generator=g)*.08
        v = hs.nearest_surface_vectors(points, centre, rot, size, is_box, used)
        for i in range(n):
            surf = []
            for s in range(S):
                if not used[i, s]:
                    continue
                local = sample_box_surface(size[i, s], K, g) if is_box[s] else sample_capsule_surface(size[i, s, 0], size[i, s, 1], K, g)
                surf.append(local @ rot[i, s].T+centre[i, s])
            surf = torch.cat(surf)
            d = torch.cdist(points[i], surf).min(-1).values
            got = v[i].norm(dim=-1)
            # brute force over sampled surface points is an upper bound within sampling resolution
            self.assertTrue(bool((got <= d+1e-6).all()), (got-d).max())
            self.assertLess(float((d-got).max()), 2.5e-3)
            # the vector ends on the surface: its end point is ~as close to the sampled surface as sampling allows
            end = points[i]+v[i]
            self.assertLess(float(torch.cdist(end, surf).min(-1).values.max()), 2.5e-3)

    def test_inside_points_point_to_nearest_face(self):
        centre = torch.zeros(1, 1, 3)
        rot = torch.eye(3)[None, None]
        size = torch.tensor([[[.05, .02, .01]]])
        p = torch.tensor([[[.01, 0., .004], [0., -.015, 0.]]])
        v = hs.nearest_surface_vectors(p, centre, rot, size, torch.tensor([True]), torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(v[0, 0], torch.tensor([0., 0., .006]))
        torch.testing.assert_close(v[0, 1], torch.tensor([0., -.005, 0.]))

    def test_encoding_scale_and_clip(self):
        v = torch.tensor([[[.01, 0., 0.], [.6, .8, 0.]]])
        e = hs.encode(v)
        torch.testing.assert_close(e, torch.tensor([[.1, 0., 0., 1.8, 2.4, 0.]]))


class SurgeryTests(unittest.TestCase):
    def test_zero_columns_keep_lstm_outputs(self):
        torch.manual_seed(0)
        wrapper = LSTMWithDones(input_size=12, hidden_size=16, num_layers=1)
        x = torch.randn(5, 3, 12)
        h0 = (torch.zeros(1, 3, 16), torch.zeros(1, 3, 16))
        ref, _ = wrapper.rnn(x, h0)
        hs.expand_lstm_input(wrapper, 9, 4)
        self.assertEqual(wrapper.rnn.input_size, 16)
        x2 = torch.cat((x[..., :9], torch.randn(5, 3, 4)*5, x[..., 9:]), -1)
        out, _ = wrapper.rnn(x2, h0)
        torch.testing.assert_close(out, ref)

    def test_expand_state_dict(self):
        old = {'w': torch.arange(12.).reshape(3, 4), 'b': torch.ones(3)}
        model = {'w': torch.zeros(3, 6), 'b': torch.zeros(3), 'hand.running_mean': torch.full((2,), 7.)}
        out, added = hs.expand_state_dict(old, model, {'w': (1, 2)})
        torch.testing.assert_close(out['w'], torch.tensor([[0., 0., 0., 1., 2., 3.], [4., 0., 0., 5., 6., 7.], [8., 0., 0., 9., 10., 11.]]))
        self.assertEqual(added, ['hand.running_mean'])
        with self.assertRaises(ValueError):
            hs.expand_state_dict({**old, 'stray': torch.zeros(1)}, model, {'w': (1, 2)})

    def test_normalizer_with_hand_segment(self):
        from rl_games.algos_torch.running_mean_std import RunningMeanStd
        task = RunningMeanStd((5,))
        norm = TaskBodyNormalizer(task, 5, True, hand_dim=3).eval()
        x = torch.randn(4, 5+BODY_EXTRA_DIM+3)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)
        plain = TaskBodyNormalizer(task, 5, True).eval()
        torch.testing.assert_close(y[:, :5+BODY_EXTRA_DIM], plain(x[:, :5+BODY_EXTRA_DIM]))
        self.assertNotIn('hand.running_mean', {k for k in plain.state_dict()})


if __name__ == '__main__':
    unittest.main()
