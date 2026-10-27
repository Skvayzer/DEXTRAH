"""Rotation utilities for the hand-object world model (torch, batched, wxyz quaternions).

The model state carries orientations as the first two columns of the rotation matrix ("6D", continuous;
Zhou et al. 2019) and predicts orientation changes as rotation vectors (axis * angle) composed on the
right: R_next = R exp([delta]x), delta expressed in the current object frame.
"""
import torch


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def matrix_to_rot6(R):
    """First two columns, (..., 6)."""
    return torch.cat((R[..., :, 0], R[..., :, 1]), -1)


def rot6_to_matrix(r6):
    """Gram-Schmidt on the two columns (any (..., 6) input maps to a rotation)."""
    a, b = r6[..., :3], r6[..., 3:]
    x = torch.nn.functional.normalize(a, dim=-1)
    b = b-(x*b).sum(-1, keepdim=True)*x
    y = torch.nn.functional.normalize(b, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), -1)


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack((z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z), -1).reshape(v.shape[:-1]+(3, 3))


def so3_exp(v):
    """Rotation vector (..., 3) -> rotation matrix (Rodrigues; series near zero)."""
    theta2 = (v*v).sum(-1, keepdim=True)[..., None]
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-8
    a = torch.where(small, 1-theta2/6, torch.sin(theta)/theta)
    b = torch.where(small, .5-theta2/24, (1-torch.cos(theta))/theta2.clamp_min(1e-12))
    K = skew(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(K.shape)
    return eye+a*K+b*(K@K)


def so3_log(R):
    """Rotation matrix -> rotation vector (angle in [0, pi])."""
    cos = ((R[..., 0, 0]+R[..., 1, 1]+R[..., 2, 2]-1)/2).clamp(-1+1e-7, 1-1e-7)
    theta = torch.acos(cos)
    w = torch.stack((R[..., 2, 1]-R[..., 1, 2], R[..., 0, 2]-R[..., 2, 0], R[..., 1, 0]-R[..., 0, 1]), -1)
    s = torch.sin(theta)
    factor = torch.where(theta < 1e-4, .5+theta*theta/12, theta/(2*s.clamp_min(1e-12)))
    return w*factor[..., None]


def rotation_angle(R1, R2):
    """Geodesic angle between rotation matrices (rad)."""
    cos = (((R1.transpose(-1, -2)@R2).diagonal(dim1=-2, dim2=-1).sum(-1))-1)/2
    return torch.acos(cos.clamp(-1., 1.))
