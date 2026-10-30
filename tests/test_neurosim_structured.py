"""Structured hand-tool-table model (dextrah_lab/neurosim/structured) against the C MuJoCo stage-3 testbed.

CPU, float64. Needs the robot URDF (DEXTRAH_ROBOT_URDF on machines without /data1). Pass tests:
docs/neurosim_structured_plan.md §3.
"""
import unittest

import mujoco
import numpy as np
import torch

from dextrah_lab.mj.robot import ContactParams
from dextrah_lab.neurosim.structured.contact import ContactModel
from dextrah_lab.neurosim.structured.dynamics import SimState, StructuredSim
from dextrah_lab.neurosim.structured.kinematics import HandKinematics, palm_substeps
from dextrah_lab.neurosim.structured.servo import servo_velocity
from dextrah_lab.neurosim.structured.rotations import mat_to_quat, quat_to_mat
from dextrah_lab.neurosim.structured.sdf import HullGrid, hull_sdf_exact, primitive_sdf
from dextrah_lab.neurosim.structured.spec import extract
from dextrah_lab.neurosim.testbed import Testbed, TestbedSpec

T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
torch.set_num_threads(4)                     # CPU jobs: <= 4 cores (CLAUDE.md)


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


class StructuredWP2(unittest.TestCase):
    """Contact-free dynamics. Free fingers limit-cycle around their targets (the 120 Hz velocity servo is unstable
    inside its 0.4-1.7 mrad band: ~8 mrad median gap, up to ~2 rad/s), so open-loop trajectories diverge from any
    tiny difference; the tests compare one substep / one control step from MuJoCo's own state."""

    @classmethod
    def setUpClass(cls):
        cls.tb = Testbed()
        cls.scene = extract(cls.tb)
        cls.sim = StructuredSim(cls.scene)

    def _random_state(self, rng, lo_pad=.05):
        tb, r = self.tb, self.scene.hand.jnt_range
        palm = np.r_[0., .8, 1.2, rng.normal(size=4)]
        palm[3:] /= np.linalg.norm(palm[3:])
        tb.reset(palm, rng.uniform(r[:, 0]-lo_pad, r[:, 1]), np.r_[1., 1., 10., 1., 0., 0., 0.])
        lin, ang = rng.normal(size=3)*.3, rng.normal(size=3)*2
        d = tb.d
        d.qvel[tb.dadr] = rng.normal(size=11)*2
        d.qvel[tb.palm_dadr:tb.palm_dadr+3], d.qvel[tb.palm_dadr+3:tb.palm_dadr+6] = lin, ang
        return palm, lin, ang

    def test_finger_substep(self):
        """One physics substep from random states (joint limits, clipped forces, moving palm): qvel within 1e-4."""
        rng = np.random.default_rng(10)
        tb, sim, worst, limits = self.tb, self.sim, 0., 0
        r = self.scene.hand.jnt_range
        for _ in range(150):
            palm, lin, ang = self._random_state(rng)
            tgt = rng.uniform(r[:, 0]-.1, r[:, 1]+.1)
            tb.d.ctrl[tb.act] = tb.servo_ctrl(tgt)
            mujoco.mj_forward(tb.m, tb.d)
            limits += tb.d.nefc > 0
            q, qd = T(tb.d.qpos[tb.qadr][None]), T(tb.d.qvel[tb.dadr][None])
            q1, qd1 = sim.finger_substep(q, qd, T(tgt[None]), T(palm[None, :3]), quat_to_mat(T(palm[None, 3:])),
                                         T(lin[None]), T(ang[None]))
            mujoco.mj_step(tb.m, tb.d)
            worst = max(worst, np.abs(qd1[0].numpy()-tb.d.qvel[tb.dadr]).max())
            self.assertLess(np.abs(q1[0].numpy()-tb.d.qpos[tb.qadr]).max(), 1e-6)
        self.assertGreater(limits, 20)
        self.assertLess(worst, 1e-4)

    def test_control_steps_resynced(self):
        """60 control steps with a moving palm and changing targets; each step starts from MuJoCo's state."""
        rng = np.random.default_rng(11)
        tb, sim, r = self.tb, self.sim, self.scene.hand.jnt_range
        palm = np.array([0., .8, 1.2, 1., 0., 0., 0.])
        tb.reset(palm, np.zeros(11), np.r_[1., 1., 10., 1., 0., 0., 0.])
        w = rng.normal(size=3)*2
        for k in range(60):
            if k % 10 == 0:
                tgt = rng.uniform(r[:, 0]-.1, r[:, 1]+.1)
            nxt = palm.copy()
            nxt[:3] += rng.normal(size=3)*.01
            mujoco.mju_quatIntegrate(nxt[3:], w, 1/60)
            s = sim.step(SimState.from_testbed(tb), T(tgt[None]), T(palm[None]), T(nxt[None]))
            tb.step(tgt, palm, nxt)
            palm = nxt
            self.assertLess(np.abs(s.numpy()['q']-tb.d.qpos[tb.qadr]).max(), 1e-6, k)
            self.assertLess(np.abs(s.numpy()['qd']-tb.d.qvel[tb.dadr]).max(), 1e-4, k)

    def test_object_free_flight(self):
        tb = self.tb
        tb.reset(np.array([0., .8, 1.2, 1., 0., 0., 0.]), np.zeros(11), np.r_[1., 1., 10., 1., 0., 0., 0.])
        tb.d.qvel[tb.object_dadr:tb.object_dadr+6] = (.3, -.2, 1., 3., -2., 5.)
        mujoco.mj_forward(tb.m, tb.d)
        s = SimState.from_testbed(tb)
        for _ in range(60):
            tb.step(np.zeros(11), np.array([0., .8, 1.2, 1., 0., 0., 0.]))
            s = self.sim.step(s, T(np.zeros((1, 11))), T(np.array([[0., .8, 1.2, 1., 0., 0., 0.]])))
        n, o, v = s.numpy(), tb.object_qadr, tb.object_dadr
        self.assertLess(np.abs(n['obj_pos']-tb.d.qpos[o:o+3]).max(), 1e-9)
        self.assertLess(np.abs(n['obj_quat']-tb.d.qpos[o+3:o+7]).max(), 1e-9)
        self.assertLess(np.abs(n['obj_w']-tb.d.qvel[v+3:v+6]).max(), 1e-9)

    def test_soft_servo_backward(self):
        """Forward values identical; d v_cmd / d target is 0 outside the band (exact), non-zero with band = 10 mrad."""
        h = self.scene.hand
        gain, vmax = T(h.servo_gain), T(h.servo_vmax)
        q = torch.zeros(1, 11, dtype=torch.float64)
        tgt = torch.full((1, 11), .01, dtype=torch.float64, requires_grad=True)       # 10 mrad gap: saturated
        exact = servo_velocity(tgt, q, gain, vmax)
        soft = servo_velocity(tgt, q, gain, vmax, band=.01)
        self.assertEqual((exact-soft).abs().max().item(), 0.)
        g_exact, = torch.autograd.grad(exact.sum(), tgt)
        g_soft, = torch.autograd.grad(soft.sum(), tgt)
        self.assertEqual(g_exact.abs().max().item(), 0.)
        self.assertTrue((g_soft > 0).all())


TOOL = dict(size=np.array([[.012, .012, .07], [1e-4, 0., 0.], [.03, .012, .02], [1e-4, 0., 0.]]),
            pos=np.array([[0., 0., -.02], [0., 0., 0.], [0., 0., .07], [0., 0., 0.]]), quat=np.tile([1., 0., 0., 0.], (4, 1)),
            used=np.array([1, 0, 1, 0], bool), mass=.3, inertia=np.array([5e-4, 5e-4, 1e-4]))


class StructuredWP3(unittest.TestCase):
    def test_tool_on_table_matches_mujoco(self):
        """Box tool (handle + head) landing / sliding / spinning on the table, 40 re-synced control steps: MuJoCo's
        8 corner contacts, normal law and (condim 4) elliptic friction reproduced to solver precision."""
        palm = np.array([0., .6, 1.3, 1., 0., 0., 0.])
        R = np.zeros(4)
        mujoco.mju_euler2Quat(R, np.radians([90., 3., 0.]), 'xyz')
        for condim, tol in ((1, 1e-12), (4, 1e-7)):
            tb = Testbed(TestbedSpec(object_kind='tool', tool=TOOL), contact=ContactParams(condim=condim))
            sc = extract(tb)
            sim = StructuredSim(sc, contact_model=ContactModel(sc))
            tb.reset(palm, np.zeros(11), np.r_[0., 0., .8125, R], table_top=.8)
            tb.d.qvel[tb.object_dadr:tb.object_dadr+6] = (.03, .01, -.05, 0., .5, .3)
            mujoco.mj_forward(tb.m, tb.d)
            o, v = tb.object_qadr, tb.object_dadr
            for k in range(40):
                s = sim.step(SimState.from_testbed(tb), T(np.zeros((1, 11))), T(palm[None]))
                tb.step(np.zeros(11), palm)
                n = s.numpy()
                self.assertLess(np.abs(n['obj_v']-tb.d.qvel[v:v+3]).max(), tol, (condim, k))
                self.assertLess(np.abs(n['obj_w']-tb.d.qvel[v+3:v+6]).max(), 10*tol, (condim, k))
            c = sim.last['contacts']
            act = [float(x) for x, a, kd in zip(c.dist[0], c.active[0], c.kind) if a and kd == 'object_table']
            self.assertEqual(sorted(np.round(act, 9)), sorted(np.round([tb.d.contact[i].dist for i in range(tb.d.ncon)], 9)))

    def test_finger_forces_on_held_cylinder(self):
        """Fingers close and press on an immovable cylinder (Testbed hold_object); per-finger normal force from one
        substep at MuJoCo's state. Status (WP3, one contact per pair): median relative error <= 15%."""
        from dextrah_lab.neurosim.reference.held_states import axis_angle, pose, random_unit, rotation_between
        from dextrah_lab.neurosim.testbed import CLOSE_TARGETS
        tb = Testbed(TestbedSpec(geometry='capsule'))
        sc = extract(tb)
        sim = StructuredSim(sc, contact_model=ContactModel(sc))
        r = float(tb.m.geom_size[tb.object_geom][0])
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']
        fb = np.array([fingers.index(f) for f in sc.hand.finger_of_body])
        rng, rel, scenes = np.random.default_rng(1), [], 0
        while scenes < 4:
            axis_h = axis_angle(random_unit(rng), np.radians(rng.uniform(0, 20))) @ np.array([0., 1., 0.])
            c_h = np.array([.021+r+rng.uniform(0, .008), rng.uniform(-.015, .025), rng.uniform(.045, .095)])
            R_hw = axis_angle(random_unit(rng), rng.uniform(0, np.pi))
            p_ow = np.array([0., 0., 1.2])
            palm, obj = pose(p_ow-R_hw @ c_h, R_hw), pose(p_ow, R_hw @ rotation_between(np.array([0., 0., 1.]), axis_h))
            tb.reset(palm, np.zeros(11), obj, table_top=.8)
            if any(c['pair'] == 'hand_object' and c['dist'] < 0 for c in tb.contacts()):
                continue
            tgt = CLOSE_TARGETS*rng.uniform(.6, 1.)
            for _ in range(45):
                tb.step(tgt, palm, hold_object=obj)
            tb.d.ctrl[tb.act] = tb.servo_ctrl(tgt)
            mujoco.mj_forward(tb.m, tb.d)
            mj = np.array([tb.finger_forces()[f] for f in fingers])
            sim.substep(SimState.from_testbed(tb), T(tgt[None]), T(palm[None, :3]), quat_to_mat(T(palm[None, 3:])),
                        T(np.zeros((1, 3))), T(np.zeros((1, 3))), hold_object=True)
            c, f = sim.last['contacts'], sim.last['forces']['elliptic'][0, :, 0]
            ours = np.zeros(6)
            for k, (kd, a) in enumerate(zip(c.kind, c.active[0])):
                if kd == 'hand_object' and a:
                    ours[fb[int(c.hand_body[k])]] += float(f[k])
            big = mj > 2.
            rel += list(np.abs(ours-mj)[big]/mj[big])
            scenes += 1
        self.assertGreater(len(rel), 8)
        self.assertLess(np.median(rel), .15)


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
