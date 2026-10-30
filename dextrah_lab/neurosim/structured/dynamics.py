"""Contact-free dynamics of the fingers on the prescribed palm and of the free object (WP2), MuJoCo's step order.

One physics substep (Testbed.step: palm pose / velocity set, servo command from the current q, then mj_step):
    qfrc_smooth = clip(kv (v_cmd - qd), ±effort) + gravity - Coriolis/centrifugal (incl. the palm's motion)
        (the clip is the joint actuator force range MuJoCo's URDF import sets from the effort limit; the velocity
        servo itself is unclamped, see robot.hand_servo)
    joint limits: soft unilateral rows solved with M (constraint.solve_unilateral)
    implicitfast: (M + dt kv) qacc = qfrc_smooth + qfrc_constraint;  qd += dt qacc;  q += dt qd
        (MuJoCo keeps the -kv damping derivative even where the force is clipped; measured, 1e-15)
Object (free joint, body frame = principal frame at the COM): linear velocity in the world, angular in the body frame;
    τ = -ω × Iω treated implicitly as MuJoCo 3.11's implicitfast does (measured): (I - dt ∂τ/∂ω) Δω = dt τ;
    v += dt g;  pos += dt v;  quat ← quat · exp(ω dt / 2).
"""
from dataclasses import dataclass, replace

import torch

from .constraint import reference, solve_unilateral
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
    def __init__(self, scene, dtype=torch.float64, device='cpu', servo_band=None, limit_iters=10, coriolis=True):
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
        self.refsafe = scene.refsafe
        self.servo_band = servo_band
        self.limit_iters = limit_iters
        self.coriolis = coriolis

    # --- fingers ------------------------------------------------------------------------------------------------
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

    def joint_limit_force(self, q, qd, M, qfrc_smooth):
        """Soft joint-limit rows (lower: pos = q - lo, J = +1; upper: pos = hi - q, J = -1), solved with M."""
        n = q.shape[1]
        eye = torch.eye(n, dtype=q.dtype, device=q.device)
        J = torch.cat((eye, -eye), 0)                                       # (2n, n)
        pos = torch.cat((q-self.lo, self.hi-q), -1)                         # (B, 2n)
        active = pos < 0
        if not bool(active.any()):
            return torch.zeros_like(q)
        Minv = torch.linalg.inv(M)
        A = J @ Minv @ J.T
        a0 = (J @ (Minv @ qfrc_smooth[..., None])).squeeze(-1)
        vel = qd @ J.T
        solref, solimp = self.jnt_solref.repeat(2, 1), self.jnt_solimp.repeat(2, 1)
        aref, R, _ = reference(solref, solimp, pos, vel, self.dof_invweight0.repeat(2), self.dt, self.refsafe)
        f = solve_unilateral(A, R, a0-aref, active, self.limit_iters)
        return f @ J

    def finger_substep(self, q, qd, targets, palm_pos, palm_R, palm_lin=None, palm_ang=None, qfrc_extra=None):
        kin, M, qfrc = self.finger_terms(q, qd, targets, palm_pos, palm_R, palm_lin, palm_ang)
        if qfrc_extra is not None:
            qfrc = qfrc+qfrc_extra
        qfrc = qfrc+self.joint_limit_force(q, qd, M, qfrc)
        Mt = M+torch.diag_embed((self.dt*self.kv).expand_as(q))
        qacc = torch.linalg.solve(Mt, qfrc[..., None]).squeeze(-1)
        qd = qd+self.dt*qacc
        return q+self.dt*qd, qd

    # --- object -------------------------------------------------------------------------------------------------
    def object_substep(self, pos, quat, v, w, force=None, torque=None):
        """force: world (B, 3); torque: body frame (B, 3) (both at the COM)."""
        acc = self.gravity.expand_as(v) if force is None else self.gravity+force/self.obj_mass
        I = self.obj_inertia
        Iw = I*w
        tau = -torch.cross(w, Iw, dim=-1) if torque is None else torque-torch.cross(w, Iw, dim=-1)
        D = -(skew(w)*I-skew(Iw))                                            # ∂(-ω × Iω)/∂ω
        v = v+self.dt*acc
        w = w+self.dt*torch.linalg.solve(torch.diag_embed(I.expand_as(w))-self.dt*D, tau[..., None]).squeeze(-1)
        return pos+self.dt*v, quat_integrate(quat, w, self.dt), v, w

    # --- one control step ---------------------------------------------------------------------------------------
    def step(self, state, targets, palm, palm_next=None):
        """targets (B, 11); palm, palm_next (B, 7). Contact-free (WP2)."""
        pm = palm_substeps(palm, palm if palm_next is None else palm_next, self.substeps, self.dt)
        s = state
        for k in range(self.substeps):
            R = quat_to_mat(pm.quat[k])
            q, qd = self.finger_substep(s.q, s.qd, targets, pm.pos[k], R, pm.lin, pm.ang)
            pos, quat, v, w = self.object_substep(s.obj_pos, s.obj_quat, s.obj_v, s.obj_w)
            s = replace(s, q=q, qd=qd, obj_pos=pos, obj_quat=quat, obj_v=v, obj_w=w)
        return s
