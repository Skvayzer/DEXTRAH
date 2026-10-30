"""Structured hand-tool-table model (dextrah_lab/neurosim/structured) against the C MuJoCo stage-3 testbed.

CPU, float64. Needs the robot URDF (DEXTRAH_ROBOT_URDF on machines without /data1). Pass tests:
docs/neurosim_structured_plan.md §3.
"""
import unittest

import mujoco
import numpy as np
import torch

from dextrah_lab.neurosim.structured.kinematics import HandKinematics, palm_substeps
from dextrah_lab.neurosim.structured.rotations import mat_to_quat, quat_to_mat
from dextrah_lab.neurosim.structured.sdf import HullGrid, hull_sdf_exact, primitive_sdf
from dextrah_lab.neurosim.structured.spec import extract
from dextrah_lab.neurosim.testbed import Testbed, TestbedSpec

T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)


def random_palm(rng):
    q = rng.normal(size=4)
    return np.r_[rng.normal(size=3)*.1+(0., .25, 1.05), q/np.linalg.norm(q)]


def probe_model(kind, size=None, verts=None):
    """One geom at the origin + a 10 µm probe sphere on a mocap body (for mj_geomDistance)."""
    spec = mujoco.MjSpec()
    if kind == 'mesh':
        spec.add_mesh(name='hull', uservert=np.asarray(verts).reshape(-1).tolist())
        g = spec.worldbody.add_geom(name='g', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='hull')
    else:
        g = spec.worldbody.add_geom(name='g', type=getattr(mujoco.mjtGeom, 'mjGEOM_'+kind.upper()), size=size)
    probe = spec.worldbody.add_body(name='probe', mocap=True)
    probe.add_geom(name='p', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(1e-5, 0., 0.))
    m = spec.compile()
    return m, mujoco.MjData(m)


def mj_point_distance(m, d, points):
    out = []
    for p in points:
        d.mocap_pos[0] = p
        mujoco.mj_kinematics(m, d)
        out.append(mujoco.mj_geomDistance(m, d, 0, 1, 1., None)+1e-5)
    return np.array(out)


class StructuredWP1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = Testbed()
        cls.scene = extract(cls.tb)
        cls.hk = HandKinematics(cls.scene.hand, cls.scene)

    def _set(self, palm, q):
        tb = self.tb
        tb.d.qpos[tb.palm_qadr:tb.palm_qadr+7] = palm
        tb.d.qpos[tb.qadr] = q
        mujoco.mj_forward(tb.m, tb.d)

    def _fk(self, palm, q):
        return self.hk.fk(T(palm[None, :3]), quat_to_mat(T(palm[None, 3:])), T(q[None]))

    def test_fk_matches_mujoco(self):
        rng = np.random.default_rng(0)
        ids = [self.tb.m.body(n).id for n in self.scene.hand.body_names]
        r = self.scene.hand.jnt_range
        for _ in range(25):
            palm, q = random_palm(rng), rng.uniform(r[:, 0], r[:, 1])
            self._set(palm, q)
            k = self._fk(palm, q)
            d = self.tb.d
            self.assertLess(np.abs(k['xpos'][0].numpy()-d.xpos[ids]).max(), 1e-9)
            self.assertLess(np.abs(k['xR'][0].numpy().reshape(-1, 9)-d.xmat[ids]).max(), 1e-9)
            self.assertLess(np.abs(k['xipos'][0].numpy()-d.xipos[ids]).max(), 1e-9)
            ja = [self.tb.m.joint('hand/'+n).id for n in
                  [x.split('/')[-1] for x in [self.tb.m.joint(j).name for j in self.tb.joint_ids]]]
            self.assertLess(np.abs(k['anchor'][0].numpy()-d.xanchor[ja]).max(), 1e-9)
            self.assertLess(np.abs(k['axis'][0].numpy()-d.xaxis[ja]).max(), 1e-9)

    def test_point_jacobian_and_mass_matrix(self):
        rng = np.random.default_rng(1)
        m, d = self.tb.m, self.tb.d
        r = self.scene.hand.jnt_range
        for _ in range(5):
            palm, q = random_palm(rng), rng.uniform(r[:, 0], r[:, 1])
            self._set(palm, q)
            k = self._fk(palm, q)
            body = torch.arange(self.hk.nb)
            pts = k['xpos']+.01*T(rng.normal(size=(1, self.hk.nb, 3)))
            J = self.hk.point_jacobian(k, body, pts)[0].numpy()
            for b, name in enumerate(self.scene.hand.body_names):
                jacp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jacp, None, pts[0, b].numpy(), m.body(name).id)
                self.assertLess(np.abs(J[b]-jacp[:, self.tb.dadr]).max(), 1e-12)
            M = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, M)
            Mh = self.hk.mass_matrix(k, T(self.scene.hand.armature))[0].numpy()
            self.assertLess(np.abs(Mh-M[np.ix_(self.tb.dadr, self.tb.dadr)]).max(), 1e-12)

    def test_gravity_torque(self):
        rng = np.random.default_rng(2)
        m, d = self.tb.m, self.tb.d
        r = self.scene.hand.jnt_range
        palm, q = random_palm(rng), rng.uniform(r[:, 0], r[:, 1])
        self._set(palm, q)
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        g = self.hk.gravity_torque(self._fk(palm, q), T(self.scene.gravity))[0].numpy()
        self.assertLess(np.abs(g+d.qfrc_bias[self.tb.dadr]).max(), 1e-12)

    def test_palm_substeps_match_testbed(self):
        """The palm poses MuJoCo sees at each substep of Testbed.step (recorded by stepping) equal palm_substeps."""
        rng = np.random.default_rng(3)
        tb = self.tb
        for _ in range(5):
            p0 = random_palm(rng)
            w = rng.normal(size=3)*.3
            q1 = np.zeros(4)
            mujoco.mju_quatIntegrate(q1 := p0[3:].copy(), w, 1.)
            p1 = np.r_[p0[:3]+rng.normal(size=3)*.01, q1]
            seen, step = [], mujoco.mj_step

            def spy(m, d, *a):
                seen.append(d.qpos[tb.palm_qadr:tb.palm_qadr+7].copy())
                return step(m, d, *a)
            tb.reset(p0, np.zeros(11), np.r_[0., 0., .9, 1., 0., 0., 0.])
            mujoco.mj_step = spy
            try:
                tb.step(np.zeros(11), p0, p1)
            finally:
                mujoco.mj_step = step
            pm = palm_substeps(T(p0[None]), T(p1[None]), self.scene.substeps, self.scene.timestep)
            ours = torch.cat((pm.pos[:, 0], pm.quat[:, 0]), -1).numpy()
            self.assertLess(np.abs(ours-np.array(seen)).max(), 1e-12)
            self.assertLess(np.abs(torch.cat((pm.end_pos, pm.end_quat), -1)[0].numpy()-p1).max(), 1e-15)

    def test_mat_to_quat_roundtrip(self):
        q = torch.nn.functional.normalize(torch.randn(100, 4, dtype=torch.float64), dim=-1)
        q = q*torch.where(q[:, :1] < 0, -1., 1.)
        self.assertLess((mat_to_quat(quat_to_mat(q))-q).abs().max().item(), 1e-12)


class StructuredSDF(unittest.TestCase):
    def test_primitives_match_mj_geomDistance(self):
        rng = np.random.default_rng(4)
        cases = dict(sphere=(.02, 0., 0.), capsule=(.012, .04, 0.), box=(.012, .03, .05), cylinder=(.03, .07, 0.))
        for kind, size in cases.items():
            m, d = probe_model(kind, size)
            pts = rng.normal(size=(300, 3))*np.r_[size[:2], max(size[1], size[2]) or size[0]].max()*1.2
            ref = mj_point_distance(m, d, pts)
            geom = type('G', (), dict(kind=kind, size=np.array(size)))
            ours = primitive_sdf(geom, T(pts)).numpy()
            out = ref > 1e-4
            self.assertLess(np.abs(ours-ref)[out].max(), 1e-6, kind)
            # inside: MuJoCo's convex penetration depth equals the distance to the nearest surface point
            self.assertLess(np.abs(ours-ref)[~out].max(), 1e-5, kind)

    def test_hand_hulls_exact_and_grid(self):
        tb = Testbed()
        scene = extract(tb)
        rng = np.random.default_rng(5)
        for g in scene.hand.geoms[:6]:
            v, f = g.hull_vert, g.hull_face
            m, d = probe_model('mesh', verts=v)
            ext = v.max(0)-v.min(0)
            pts = v.mean(0)+rng.uniform(-1, 1, size=(300, 3))*(.5*ext+.006)
            ref = mj_point_distance(m, d, pts)
            ours = hull_sdf_exact(T(pts), v, f).numpy()
            self.assertLess(np.abs(ours-ref).max(), 2e-6, g.mujoco_id)
            grid = HullGrid(v, f, spacing=1e-3)
            near = np.abs(ref) < .005
            err = np.abs(grid(T(pts)).numpy()-ref)[near]
            self.assertLess(np.quantile(err, .95), 3e-4, g.mujoco_id)


if __name__ == '__main__':
    unittest.main()
