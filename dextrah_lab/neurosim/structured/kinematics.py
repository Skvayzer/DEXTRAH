"""Hand kinematics on a prescribed palm (batched torch, MuJoCo's conventions).

palm_substeps reproduces Testbed.step's palm trajectory exactly: constant world-frame linear velocity and constant
body-frame angular velocity (mju_quat2Vel of the step's relative rotation), pose set at the start of every physics
substep. hand_fk is mj_kinematics restricted to the hand subtree: body frame = parent frame * (body_pos, body_quat),
then each hinge rotates the body about its anchor by (q - qpos0).
"""
from dataclasses import dataclass

import torch

from .rotations import axis_angle_mat, quat_conj, quat_integrate, quat_mul, quat_to_mat, quat_to_vel


@dataclass
class PalmMotion:
    pos: torch.Tensor            # (S, B, 3) palm position at the start of each substep
    quat: torch.Tensor           # (S, B, 4)
    lin: torch.Tensor            # (B, 3) world-frame linear velocity (constant over the control step)
    ang: torch.Tensor            # (B, 3) body-frame angular velocity
    end_pos: torch.Tensor        # (B, 3) the next control step's palm pose (exact)
    end_quat: torch.Tensor


def palm_substeps(palm, palm_next, substeps, dt):
    """palm, palm_next: (B, 7) pos + wxyz quat; Testbed.step's interpolation."""
    T = substeps*dt
    lin = (palm_next[:, :3]-palm[:, :3])/T
    dq = quat_mul(quat_conj(palm[:, 3:]), palm_next[:, 3:])
    ang = quat_to_vel(dq, T)
    pos, quat = [], []
    for k in range(substeps):
        a = k/substeps
        pos.append(palm[:, :3]+a*(palm_next[:, :3]-palm[:, :3]))
        quat.append(quat_integrate(palm[:, 3:], ang, a*T) if k else palm[:, 3:])
    return PalmMotion(torch.stack(pos), torch.stack(quat), lin, ang, palm_next[:, :3], palm_next[:, 3:])


class HandKinematics:
    """Forward kinematics, COM / point Jacobians of the hand bodies. Build once per HandSpec."""

    def __init__(self, hand, scene=None, dtype=torch.float64, device='cpu'):
        t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        self.nb = hand.nbody
        self.parent = [int(p) for p in hand.parent]
        self.body_pos = t(hand.body_pos)
        self.body_R = quat_to_mat(t(hand.body_quat))
        self.ipos = t(hand.ipos)
        self.iR = quat_to_mat(t(hand.iquat))
        self.mass = t(hand.mass)
        self.inertia = t(hand.inertia)
        self.body_joint = [int(j) for j in hand.body_joint]
        self.jnt_pos = t(hand.jnt_pos)
        self.jnt_axis = t(hand.jnt_axis)
        self.qpos0 = t(hand.qpos0)
        # ancestors[b]: joints (HAND_JOINTS indices) that move body b
        self.ancestors = []
        for b in range(self.nb):
            js, x = [], b
            while x >= 0:
                if self.body_joint[x] >= 0:
                    js.append(self.body_joint[x])
                x = self.parent[x]
            self.ancestors.append(sorted(js))
        # mask (nb, 11): joint j moves body b
        self.moves = torch.zeros(self.nb, len(hand.qpos0), dtype=torch.bool, device=device)
        for b, js in enumerate(self.ancestors):
            self.moves[b, js] = True
        if scene is not None and (abs(scene.palm_body_pos).max() > 0 or abs(scene.palm_body_quat-(1, 0, 0, 0)).max() > 0):
            raise ValueError('hand root is offset from the palm frame (not implemented)')

    def fk(self, palm_pos, palm_R, q):
        """palm_pos (B, 3), palm_R (B, 3, 3), q (B, 11) -> dict(xpos (B, nb, 3), xR (B, nb, 3, 3),
        anchor (B, 11, 3), axis (B, 11, 3) world-frame joint anchors and axes, xipos (B, nb, 3) COMs,
        ximat (B, nb, 3, 3) inertia frames)."""
        B = q.shape[0]
        xpos, xR = [None]*self.nb, [None]*self.nb
        nj = q.shape[1]
        anchor, axis = [None]*nj, [None]*nj
        for b in range(self.nb):
            p = self.parent[b]
            pp, pR = (palm_pos, palm_R) if p < 0 else (xpos[p], xR[p])
            pos = pp+(pR @ self.body_pos[b])
            R = pR @ self.body_R[b]
            j = self.body_joint[b]
            if j >= 0:
                a = pos+R @ self.jnt_pos[j]
                ax = R @ self.jnt_axis[j]
                R = R @ axis_angle_mat(self.jnt_axis[j].expand(B, 3), q[:, j]-self.qpos0[j])
                pos = a-R @ self.jnt_pos[j]
                anchor[j], axis[j] = a, ax
            xpos[b], xR[b] = pos, R
        xpos, xR = torch.stack(xpos, 1), torch.stack(xR, 1)
        xipos = xpos+(xR @ self.ipos[..., None]).squeeze(-1)
        return dict(xpos=xpos, xR=xR, anchor=torch.stack(anchor, 1), axis=torch.stack(axis, 1), xipos=xipos,
                    ximat=xR @ self.iR)

    def point_jacobian(self, kin, body, points):
        """Translational Jacobian of world points attached to hand bodies.
        body (P,) long, points (B, P, 3) -> (B, P, 3, 11): column j = axis_j x (p - anchor_j) where joint j moves the body."""
        ax = kin['axis'][:, None]                         # (B, 1, 11, 3)
        r = points[:, :, None]-kin['anchor'][:, None]      # (B, P, 11, 3)
        J = torch.cross(ax.expand_as(r), r, dim=-1)*self.moves[body][None, :, :, None]
        return J.transpose(-1, -2)

    def rotation_jacobian(self, kin, body):
        """(B, P, 3, 11) angular-velocity Jacobian of bodies `body`."""
        return (kin['axis'][:, None]*self.moves[body][None, :, :, None]).transpose(-1, -2)

    def mass_matrix(self, kin, armature):
        """Joint-space inertia of the 11 finger joints with the palm fixed (CRBA via COM Jacobians) + armature."""
        bodies = torch.arange(self.nb, device=kin['xpos'].device)
        Jv = self.point_jacobian(kin, bodies, kin['xipos'])                    # (B, nb, 3, 11)
        Jw = self.rotation_jacobian(kin, bodies)
        Iw = kin['ximat'] @ torch.diag_embed(self.inertia) @ kin['ximat'].transpose(-1, -2)
        M = (self.mass[None, :, None, None]*Jv.transpose(-1, -2) @ Jv).sum(1)+(Jw.transpose(-1, -2) @ Iw @ Jw).sum(1)
        return M+torch.diag_embed(armature.expand(M.shape[0], -1))

    def velocities(self, kin, qd, palm_pos, palm_lin, palm_ang_world):
        """World COM linear velocities (B, nb, 3) and angular velocities (B, nb, 3) of the hand bodies."""
        w_j = kin['axis']*qd[..., None]                                          # (B, 11, 3)
        moves = self.moves.to(qd.dtype)
        omega = palm_ang_world[:, None]+torch.einsum('nj,bjk->bnk', moves, w_j)
        c = kin['xipos']
        v = palm_lin[:, None]+torch.cross(palm_ang_world[:, None].expand_as(c), c-palm_pos[:, None], dim=-1)
        r = c[:, :, None]-kin['anchor'][:, None]                                  # (B, nb, 11, 3)
        v = v+(torch.cross(w_j[:, None].expand_as(r), r, dim=-1)*moves[None, :, :, None]).sum(2)
        return v, omega

    def bias_force(self, palm_pos, palm_R, q, qd, palm_lin, palm_ang_local):
        """Coriolis / centrifugal generalised force on the 11 joints (qfrc_bias without gravity), including the
        prescribed palm's motion (constant linear and body-frame angular velocity, zero palm acceleration):
        sum_b Jv_bᵀ m_b a_b + Jw_bᵀ (I_b α_b + ω_b × I_b ω_b), with (a_b, α_b) the body accelerations at qdd = 0."""
        from .rotations import so3_exp

        def vel(q_, p_, th):
            R = palm_R @ so3_exp(th)
            k = self.fk(p_, R, q_)
            v, w = self.velocities(k, qd, p_, palm_lin, (R @ palm_ang_local[..., None]).squeeze(-1))
            return v, w
        zero = torch.zeros_like(palm_lin)
        (v, w), (a, alpha) = torch.func.jvp(vel, (q, palm_pos, zero), (qd, palm_lin, palm_ang_local))
        kin = self.fk(palm_pos, palm_R, q)
        bodies = torch.arange(self.nb, device=q.device)
        Jv = self.point_jacobian(kin, bodies, kin['xipos'])
        Jw = self.rotation_jacobian(kin, bodies)
        Iw = kin['ximat'] @ torch.diag_embed(self.inertia) @ kin['ximat'].transpose(-1, -2)
        Iw_w = (Iw @ w[..., None]).squeeze(-1)
        torque = (Iw @ alpha[..., None]).squeeze(-1)+torch.cross(w, Iw_w, dim=-1)
        f = self.mass[None, :, None]*a
        return ((Jv.transpose(-1, -2) @ f[..., None]).squeeze(-1)+(Jw.transpose(-1, -2) @ torque[..., None]).squeeze(-1)).sum(1)

    def gravity_torque(self, kin, gravity):
        """Generalised gravity force on the 11 joints (the -qfrc_bias gravity part)."""
        bodies = torch.arange(self.nb, device=kin['xpos'].device)
        Jv = self.point_jacobian(kin, bodies, kin['xipos'])
        f = self.mass[None, :, None]*gravity                                   # (1, nb, 3)
        return (Jv.transpose(-1, -2) @ f[..., None]).squeeze(-1).sum(1)
