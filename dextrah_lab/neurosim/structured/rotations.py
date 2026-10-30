"""Batched rotations in MuJoCo's conventions (quaternions wxyz, rotation matrices act on column vectors).

Everything is differentiable torch; shapes (..., 4) for quaternions, (..., 3, 3) for matrices.
"""
import math

import torch


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_conj(q):
    return q*q.new_tensor((1., -1., -1., -1.))


def quat_to_mat(q):
    """mju_quat2Mat (q need not be normalised exactly; MuJoCo assumes unit quaternions too)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion with w >= 0 (Shepperd's method, branch-free via the largest diagonal)."""
    m = R
    t = torch.stack((1+m[..., 0, 0]+m[..., 1, 1]+m[..., 2, 2], 1+m[..., 0, 0]-m[..., 1, 1]-m[..., 2, 2],
                     1-m[..., 0, 0]+m[..., 1, 1]-m[..., 2, 2], 1-m[..., 0, 0]-m[..., 1, 1]+m[..., 2, 2]), -1)
    k = t.argmax(-1, keepdim=True)
    s = 2*torch.sqrt(t.gather(-1, k).clamp_min(1e-30)).squeeze(-1)
    c = [torch.stack((s/4, (m[..., 2, 1]-m[..., 1, 2])/s, (m[..., 0, 2]-m[..., 2, 0])/s, (m[..., 1, 0]-m[..., 0, 1])/s), -1),
         torch.stack(((m[..., 2, 1]-m[..., 1, 2])/s, s/4, (m[..., 0, 1]+m[..., 1, 0])/s, (m[..., 0, 2]+m[..., 2, 0])/s), -1),
         torch.stack(((m[..., 0, 2]-m[..., 2, 0])/s, (m[..., 0, 1]+m[..., 1, 0])/s, s/4, (m[..., 1, 2]+m[..., 2, 1])/s), -1),
         torch.stack(((m[..., 1, 0]-m[..., 0, 1])/s, (m[..., 0, 2]+m[..., 2, 0])/s, (m[..., 1, 2]+m[..., 2, 1])/s, s/4), -1)]
    q = torch.stack(c, -2).gather(-2, k[..., None].expand(*k.shape[:-1], 1, 4)).squeeze(-2)
    return q*torch.where(q[..., :1] < 0, -1., 1.)


def axis_angle_quat(axis, angle):
    """Unit axis (..., 3), angle (...) -> quaternion (mju_axisAngle2Quat)."""
    h = .5*angle
    return torch.cat((torch.cos(h)[..., None], torch.sin(h)[..., None]*axis), -1)


def axis_angle_mat(axis, angle):
    """Rodrigues: rotation by angle (...) about unit axis (..., 3)."""
    c, s = torch.cos(angle)[..., None, None], torch.sin(angle)[..., None, None]
    K = skew(axis)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(K.shape)
    return eye+s*K+(1-c)*(K @ K)


def skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def so3_exp(w):
    """Rotation matrix of the rotation vector w (..., 3); smooth at 0."""
    th2 = (w*w).sum(-1)
    th = torch.sqrt(th2.clamp_min(1e-30))
    small = th2 < 1e-12
    a = torch.where(small, 1-th2/6, torch.sin(th)/th)
    b = torch.where(small, .5-th2/24, (1-torch.cos(th))/th2.clamp_min(1e-30))
    K = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return eye+a[..., None, None]*K+b[..., None, None]*(K @ K)


def quat_integrate(q, vel, scale):
    """mju_quatIntegrate: q * exp(vel * scale / 2) (vel in the local frame), normalised."""
    v = vel*(scale[..., None] if torch.is_tensor(scale) else scale)
    ang = torch.linalg.norm(v, dim=-1)
    axis = v/ang.clamp_min(1e-30)[..., None]
    dq = axis_angle_quat(axis, ang)
    out = quat_mul(q, dq)
    return out/torch.linalg.norm(out, dim=-1, keepdim=True)


def quat_to_vel(q, dt):
    """mju_quat2Vel: angular velocity (local frame) that rotates by q in time dt; angle wrapped to (-pi, pi]."""
    axis = q[..., 1:]
    s = torch.linalg.norm(axis, dim=-1)
    speed = 2*torch.atan2(s, q[..., 0])
    speed = torch.where(speed > math.pi, speed-2*math.pi, speed)
    return axis/s.clamp_min(1e-30)[..., None]*(speed/dt)[..., None]
