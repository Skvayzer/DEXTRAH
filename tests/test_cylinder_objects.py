"""Opt-in cylinder object set: generator, bank (fifth slot), scene, nearest-surface vectors; flag off unchanged."""
import math
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from dextrah_lab.mj.compat import install

install()
from dextrah_lab.mj import objects  # noqa: E402
from dextrah_lab.mj.cylinder_objects import generate_cylinder_urdfs  # noqa: E402
from dextrah_lab.mj.robot import ContactParams  # noqa: E402
from dextrah_lab.mj.scene import object_spec  # noqa: E402
from dextrah_lab.wholebody import hand_surface as hs  # noqa: E402


def rotate(q, v):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3) @ v


class CylinderObjects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='test_cyl_')
        cls.paths, cls.scales, cls.meta = generate_cylinder_urdfs(cls.dir, radii=(.025, .03), per_radius=5, seed=3)

    def test_generator(self):
        self.assertEqual(len(self.paths), 10)
        self.assertTrue(all(p.stem.split('_')[1] == 'cylinder' for p in self.paths))
        r, L, m = self.meta.T
        self.assertTrue(np.all((L >= .08) & (L <= .2) & (m >= .1) & (m <= .5)))
        np.testing.assert_allclose(self.scales, np.stack((L, 2*r, 2*r), 1)/.04)
        state = np.random.get_state()[1][:5].copy()
        generate_cylinder_urdfs(tempfile.mkdtemp(), per_radius=1, seed=0)
        np.testing.assert_array_equal(np.random.get_state()[1][:5], state)      # global numpy state untouched

    def test_bank_fifth_slot(self):
        with self.assertRaises(ValueError):
            objects.build_object_bank([str(p) for p in self.paths])               # needs the flag
        bank = objects.build_object_bank([str(p) for p in self.paths], cylinder_slot=True)
        self.assertEqual(bank.size.shape, (10, 5, 3))
        self.assertEqual(bank.slot_types, objects.SLOT_TYPES+('cylinder',))
        self.assertTrue(np.all(bank.used[:, 4]) and not np.any(bank.used[:, :4]))
        r, L, m = self.meta.T
        np.testing.assert_allclose(bank.size[:, 4, :2], np.stack((r, L/2), 1), rtol=1e-6)
        np.testing.assert_allclose(bank.mass, m, rtol=1e-6)
        np.testing.assert_allclose(bank.inertia[:, 0], .5*m*r**2, rtol=1e-5)
        np.testing.assert_allclose(bank.inertia[:, 1], m*(3*r**2+L**2)/12, rtol=1e-5)
        for q in bank.quat[:, 4]:                                                     # cylinder axis along link x
            np.testing.assert_allclose(np.abs(rotate(q, np.array([0., 0., 1.]))), [1., 0., 0.], atol=1e-9)
        np.testing.assert_allclose(bank.rbound[:, 4], np.hypot(r, L/2), rtol=1e-6)
        np.testing.assert_allclose(bank.aabb[:, 4, 3:], np.stack((r, r, L/2), 1), rtol=1e-6)

    def test_flag_off_bank_unchanged(self):
        """Tool bank without the flag: four slots, rbound / aabb as the original formulas."""
        from dextrah_lab.mj.compat import p2p
        paths, _ = p2p('generate_objects').generate_handle_head_urdfs(
            handle_head_types=('hammer', 'marker', 'eraser', 'brush'), num_per_type=2, out_dir=tempfile.mkdtemp(),
            shuffle=False)
        bank = objects.build_object_bank([str(p) for p in paths])
        self.assertEqual(bank.size.shape[1:], (4, 3))
        box = np.array([t == 'box' for t in objects.SLOT_TYPES])
        rb = np.where(box, np.linalg.norm(bank.size, axis=-1), bank.size[..., 0]+bank.size[..., 1])
        np.testing.assert_array_equal(bank.rbound, rb)
        half = bank.size.copy()
        cap = ~box
        half[..., cap, 1] = bank.size[..., cap, 0]
        half[..., cap, 2] = bank.size[..., cap, 0]+bank.size[..., cap, 1]
        np.testing.assert_array_equal(bank.aabb, np.concatenate((np.zeros_like(half), half), -1))

    def test_object_spec(self):
        c = ContactParams(condim=4)
        m0 = object_spec(c).compile()
        m1 = object_spec(c, cylinder_slot=True).compile()
        self.assertEqual(m0.ngeom, 4)
        self.assertEqual(m1.ngeom, 5)
        self.assertEqual([int(t) for t in m0.geom_type], [int(t) for t in m1.geom_type[:4]])
        self.assertEqual(int(m1.geom_type[4]), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        self.assertEqual(m1.geom(4).name, 'cylinder')

    def test_nearest_surface_cylinder(self):
        """Against brute force over dense surface samples, outside and inside points; box / capsule slots kept."""
        g = torch.Generator().manual_seed(0)
        r, half = .03, .07
        k = 200000
        phi = torch.rand(k, generator=g)*2*math.pi
        u = torch.rand(k, generator=g)
        side = torch.stack((r*torch.cos(phi), r*torch.sin(phi), (torch.rand(k, generator=g)*2-1)*half), -1)
        rad = r*torch.sqrt(torch.rand(k, generator=g))
        cap = torch.stack((rad*torch.cos(phi), rad*torch.sin(phi), torch.where(u < .5, half, -half)), -1)
        surf = torch.cat((side, cap))
        P = 400
        pts = torch.randn(P, 3, generator=g)*torch.tensor([.04, .04, .09])
        size = torch.tensor([[[.01, .01, .01], [.01, .01, 0.], [r, half, 0.]]])
        centre = torch.tensor([[[.5, .5, .5], [.5, .5, .5], [0., 0., 0.]]])
        rot = torch.eye(3).expand(1, 3, 3, 3)
        used = torch.tensor([[True, True, True]])
        v = hs.nearest_surface_vectors(pts[None], centre, rot, size, torch.tensor([True, False, False]), used,
                                       torch.tensor([False, False, True]))[0]
        d_brute = torch.cdist(pts, surf).min(1).values
        torch.testing.assert_close(v.norm(dim=-1), d_brute, atol=2e-3, rtol=0)
        end = pts+v                                                     # on the surface
        on_side = (end[:, :2].norm(dim=-1)-r).abs() < 1e-6
        on_cap = (end[:, 2].abs()-half).abs() < 1e-6
        self.assertTrue(bool((on_side | on_cap).all()))
        v0 = hs.nearest_surface_vectors(pts[None], centre[:, :2], rot[:, :2], size[:, :2], torch.tensor([True, False]),
                                        used[:, :2])
        v1 = hs.nearest_surface_vectors(pts[None], centre[:, :2], rot[:, :2], size[:, :2], torch.tensor([True, False]),
                                        used[:, :2], torch.tensor([False, False]))
        torch.testing.assert_close(v0, v1, rtol=0, atol=0)               # no cylinder slot: identical


if __name__ == '__main__':
    unittest.main()
