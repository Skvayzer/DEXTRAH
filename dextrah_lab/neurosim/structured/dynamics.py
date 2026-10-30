"""Dynamics of the fingers on the prescribed palm and of the free object, MuJoCo's step order (WP2, WP3/4).

One physics substep (Testbed.step: palm pose / velocity set, servo command from the current q, then mj_step):
    fingers: qfrc_smooth = clip(kv (v_cmd - qd), ±effort) + gravity - Coriolis/centrifugal (incl. the palm's motion)
        (the clip is the joint actuator force range MuJoCo's URDF import sets from the effort limit; the velocity
        servo itself is unclamped, see robot.hand_servo)
    object (free joint, body frame = principal frame at the COM; linear velocity in the world, angular in the body
        frame): qfrc_smooth = (m g, -ω × Iω)
    constraints (joint limits, contacts): MuJoCo's primal problem with the plain mass matrix
        (constraint.solve_primal) -> qfrc_constraint = M (qacc - a0)
    implicitfast: fingers (M + dt kv) qacc = qfrc_smooth + qfrc_constraint (MuJoCo keeps the -kv damping derivative
        even where the force is clipped; measured, 1e-15); object angular (I - dt ∂τ/∂ω) Δω = dt τ with
        τ = -ω × Iω + torque (MuJoCo 3.11 treats the gyroscopic term implicitly; measured);
    then qd += dt qacc; q += dt qd; quat ← quat · exp(ω dt / 2).
"""
from dataclasses import dataclass, replace

import torch

from .constraint import Rows, reference, solve_primal
from .kinematics import HandKinematics, palm_substeps
from .rotations import quat_integrate, quat_to_mat, skew
from .servo import servo_velocity


@dataclass
class SimState:
    q: torch.Tensor              # (B, 11) finger joints (HAND_JOINTS order)
    qd: torch.Tensor             # (B, 11)
    obj_pos: torch.Tensor        # (B, 3) object COM
    obj_quat: torch.Tensor       # (B, 4)
    obj_v: torch.Tensor          # (B, 3) world frame
    obj_w: torch.Tensor          # (B, 3) body frame (MuJoCo free-joint convention)

    @staticmethod
    def from_testbed(tb, dtype=torch.float64):
        d = tb.d
        t = lambda x: torch.as_tensor(x.copy(), dtype=dtype)[None]
        o, v = tb.object_qadr, tb.object_dadr
        return SimState(t(d.qpos[tb.qadr]), t(d.qvel[tb.dadr]), t(d.qpos[o:o+3]), t(d.qpos[o+3:o+7]),
                        t(d.qvel[v:v+3]), t(d.qvel[v+3:v+6]))

    def numpy(self, i=0):
        return {k: getattr(self, k)[i].detach().cpu().numpy() for k in self.__dataclass_fields__}


class StructuredSim:
    """contact_model: contact.ContactModel or None (contact-free)."""

    def __init__(self, scene, dtype=torch.float64, device='cpu', servo_band=None, coriolis=True, contact_model=None,
                 solver_iters=50):
        t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        h = scene.hand
        self.scene = scene
        self.kin = HandKinematics(h, scene, dtype, device)
        self.dt, self.substeps = scene.timestep, scene.substeps
        self.gravity = t(scene.gravity)
        self.armature = t(h.armature)
        self.gain, self.vmax, self.kv, self.effort = t(h.servo_gain), t(h.servo_vmax), t(h.servo_kv), t(h.servo_effort)
        self.lo, self.hi = t(h.jnt_range[:, 0]), t(h.jnt_range[:, 1])
        self.jnt_solref, self.jnt_solimp = t(h.jnt_solref), t(h.jnt_solimp)
        self.dof_invweight0 = t(h.dof_invweight0)
        self.obj_mass = t(scene.obj.mass)
        self.obj_inertia = t(scene.obj.inertia)
        self.c_solref, self.c_solimp = t(scene.solref), t(scene.solimp)
        self.refsafe = scene.refsafe
        self.servo_band = servo_band
        self.coriolis = coriolis
        self.contacts = contact_model
        self.solver_iters = solver_iters
        self.last = {}                                       # diagnostics of the last substep (contacts, forces)

    # --- smooth forces ------------------------------------------------------------------------------------------
    def finger_terms(self, q, qd, targets, palm_pos, palm_R, palm_lin=None, palm_ang=None):
        kin = self.kin.fk(palm_pos, palm_R, q)
        M = self.kin.mass_matrix(kin, self.armature)
        v_cmd = servo_velocity(targets, q, self.gain, self.vmax, self.servo_band)
        act = self.kv*(v_cmd-qd)
        act = torch.maximum(torch.minimum(act, self.effort), -self.effort)       # joint actuator force range (URDF)
        qfrc = act+self.kin.gravity_torque(kin, self.gravity)
        if self.coriolis:
            lin = torch.zeros_like(palm_pos) if palm_lin is None else palm_lin
            ang = torch.zeros_like(palm_pos) if palm_ang is None else palm_ang
            qfrc = qfrc-self.kin.bias_force(palm_pos, palm_R, q, qd, lin, ang)
        return kin, M, qfrc

    def limit_rows(self, q, qd, n):
        """Soft joint-limit rows (lower: pos = q - lo, J = +1; upper: pos = hi - q, J = -1) over n columns."""
        nq = q.shape[1]
        eye = torch.eye(nq, n, dtype=q.dtype, device=q.device)
        J = torch.cat((eye, -eye), 0).expand(q.shape[0], -1, -1)
        pos = torch.cat((q-self.lo, self.hi-q), -1)
        vel = torch.cat((qd, -qd), -1)
        aref, R, _ = reference(self.jnt_solref.repeat(2, 1), self.jnt_solimp.repeat(2, 1), pos, vel,
                               self.dof_invweight0.repeat(2), self.dt, self.refsafe)
        return J, aref, R, pos < 0

    # --- one substep --------------------------------------------------------------------------------------------
    def substep(self, s, targets, palm_pos, palm_R, palm_lin, palm_ang, hold_object=False):
        """hold_object: as Testbed.step(hold_object=...): the object's velocity is zeroed before the step, it stays a
        degree of freedom of the constraint solve, and its pose / velocity are not advanced."""
        B, nq = s.q.shape
        if hold_object:
            s = replace(s, obj_v=torch.zeros_like(s.obj_v), obj_w=torch.zeros_like(s.obj_w))
        kin, Mf, qf = self.finger_terms(s.q, s.qd, targets, palm_pos, palm_R, palm_lin, palm_ang)
        obj_R = quat_to_mat(s.obj_quat)
        I = self.obj_inertia
        Iw = I*s.obj_w
        gyro = -torch.cross(s.obj_w, Iw, dim=-1)
        n = nq+6
        M = torch.zeros(B, n, n, dtype=Mf.dtype, device=Mf.device)
        M[:, :nq, :nq] = Mf
        M[:, nq:nq+3, nq:nq+3] = torch.eye(3, dtype=M.dtype)*self.obj_mass
        M[:, nq+3:, nq+3:] = torch.diag_embed(I.expand(B, 3))
        qfrc = torch.cat((qf, (self.obj_mass*self.gravity).expand(B, 3), gyro), -1)
        a0 = torch.linalg.solve(M, qfrc[..., None]).squeeze(-1)
        uJ, uaref, uR, uact = self.limit_rows(s.q, s.qd, n)
        rows = Rows(uJ=uJ, uaref=uaref, uR=uR, uactive=uact)
        c = None
        if self.contacts is not None:
            c = self.contacts.generate(kin, s.obj_pos, obj_R)
            palm_w = (palm_R @ palm_ang[..., None]).squeeze(-1)
            J, vel, diag, mu, fric = self.contacts.rows(c, kin, self.kin, s.qd, palm_pos, palm_lin, palm_w, s.obj_pos,
                                                         obj_R, s.obj_v, s.obj_w, nq)
            dim = J.shape[2]
            # impedance of the contact on every row; position term only on the normal row (MuJoCo)
            aref, R, d = reference(self.c_solref, self.c_solimp, c.dist[..., None].expand_as(vel), vel, diag, self.dt,
                                   self.refsafe)
            if dim > 1:
                b = self._b()
                aref = torch.cat((aref[..., :1], -b*vel[..., 1:]), -1)
            if dim == 1:
                rows.uJ = torch.cat((rows.uJ, J[:, :, 0]), 1)
                rows.uaref = torch.cat((rows.uaref, aref[..., 0]), 1)
                rows.uR = torch.cat((rows.uR, R[..., 0]), 1)
                rows.uactive = torch.cat((rows.uactive, c.active), 1)
            else:
                rows.eJ, rows.earef, rows.eR, rows.eactive = J, aref, R, c.active
                rows.mu, rows.fric = mu.expand(B, -1), fric.expand(B, -1, -1)
        x, forces = solve_primal(M, a0, rows, self.solver_iters)
        qfrc_c = (M @ (x-a0)[..., None]).squeeze(-1)
        self.last = dict(contacts=c, forces=forces, qfrc_constraint=qfrc_c)
        # integrate (implicitfast)
        Mt = Mf+torch.diag_embed((self.dt*self.kv).expand_as(s.q))
        qacc = torch.linalg.solve(Mt, (qf+qfrc_c[:, :nq])[..., None]).squeeze(-1)
        qd = s.qd+self.dt*qacc
        q = s.q+self.dt*qd
        if hold_object:
            return replace(s, q=q, qd=qd)
        v = s.obj_v+self.dt*(self.gravity+qfrc_c[:, nq:nq+3]/self.obj_mass)
        tau = gyro+qfrc_c[:, nq+3:]
        D = -(skew(s.obj_w)*I-skew(Iw))                                         # ∂(-ω × Iω)/∂ω
        w = s.obj_w+self.dt*torch.linalg.solve(torch.diag_embed(I.expand(B, 3))-self.dt*D, tau[..., None]).squeeze(-1)
        return replace(s, q=q, qd=qd, obj_pos=s.obj_pos+self.dt*v, obj_quat=quat_integrate(s.obj_quat, w, self.dt),
                       obj_v=v, obj_w=w)

    def _b(self):
        return 2/(self.c_solimp[1].clamp(1e-4, .9999)*torch.clamp_min(self.c_solref[0], 2*self.dt if self.refsafe else 0.))

    # --- one control step ---------------------------------------------------------------------------------------
    def step(self, state, targets, palm, palm_next=None, hold_object=False):
        """targets (B, 11); palm, palm_next (B, 7)."""
        pm = palm_substeps(palm, palm if palm_next is None else palm_next, self.substeps, self.dt)
        s = state
        for k in range(self.substeps):
            s = self.substep(s, targets, pm.pos[k], quat_to_mat(pm.quat[k]), pm.lin, pm.ang, hold_object)
        return s

    # kept for the WP2 tests: one finger-only substep (no object, no contacts)
    def finger_substep(self, q, qd, targets, palm_pos, palm_R, palm_lin=None, palm_ang=None):
        B = q.shape[0]
        z = torch.zeros(B, 3, dtype=q.dtype, device=q.device)
        s = SimState(q, qd, z, torch.cat((torch.ones(B, 1, dtype=q.dtype), z), -1), z, z)
        cm, self.contacts = self.contacts, None
        try:
            s = self.substep(s, targets, palm_pos, palm_R, z if palm_lin is None else palm_lin,
                             z if palm_ang is None else palm_ang, hold_object=True)
        finally:
            self.contacts = cm
        return s.q, s.qd
