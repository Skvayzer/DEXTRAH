"""Signed distances (negative inside) of MuJoCo primitives in their geom frames, and geom-frame transforms.

MuJoCo sizes: sphere (r), capsule (r, half length along z), cylinder (r, half height along z), box (half extents).
All functions take points (..., 3) and return (...). Gradients are exact almost everywhere (the box and cylinder
have kinks on their medial surfaces, as any SDF).
"""
import torch

from .rotations import quat_to_mat


def sphere(p, size):
    return torch.linalg.norm(p, dim=-1)-size[0]


def capsule(p, size):
    z = p[..., 2].clamp(-size[1], size[1])
    d = p-torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), -1)
    return torch.linalg.norm(d, dim=-1)-size[0]


def _norm_safe(x):
    return torch.sqrt((x*x).sum(-1).clamp_min(1e-30))


def box(p, size):
    q = p.abs()-size[:3]
    outside = _norm_safe(q.clamp_min(0.))*(q.max(-1).values > 0)
    inside = q.max(-1).values.clamp_max(0.)
    return outside+inside


def cylinder(p, size):
    rho = _norm_safe(p[..., :2])
    q = torch.stack((rho-size[0], p[..., 2].abs()-size[1]), -1)
    outside = _norm_safe(q.clamp_min(0.))*(q.max(-1).values > 0)
    inside = q.max(-1).values.clamp_max(0.)
    return outside+inside


PRIMITIVES = dict(sphere=sphere, capsule=capsule, box=box, cylinder=cylinder)


def to_local(points, pos, R):
    """World points (..., 3) -> frame (pos (..., 3), R (..., 3, 3)) coordinates."""
    return ((points-pos)[..., None, :] @ R).squeeze(-2)


def geom_frame(body_pos, body_R, geom):
    """World pose of a GeomSpec attached to a body pose (..., 3), (..., 3, 3)."""
    gp = torch.as_tensor(geom.pos, dtype=body_pos.dtype, device=body_pos.device)
    gR = quat_to_mat(torch.as_tensor(geom.quat, dtype=body_pos.dtype, device=body_pos.device))
    return body_pos+(body_R @ gp[..., None]).squeeze(-1), body_R @ gR


def primitive_sdf(geom, points_local):
    size = torch.as_tensor(geom.size, dtype=points_local.dtype, device=points_local.device)
    return PRIMITIVES[geom.kind](points_local, size)


# --- convex hulls (MuJoCo's <= 64-vertex hulls of the hand meshes) ---------------------------------------------------

def point_triangle_distance(p, a, b, c):
    """Unsigned distance from points p (N, 1, 3) to triangles a, b, c (1, K, 3) -> (N, K) (Ericson's closest point;
    the region tests are applied in reverse order of his early returns, so the first matching region wins)."""
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = (ab*ap).sum(-1), (ac*ap).sum(-1)
    bp = p-b
    d3, d4 = (ab*bp).sum(-1), (ac*bp).sum(-1)
    cp = p-c
    d5, d6 = (ab*cp).sum(-1), (ac*cp).sum(-1)
    va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
    eps = 1e-300
    denom = 1/(va+vb+vc).clamp_min(eps)                        # face interior
    q = a+(vb*denom)[..., None]*ab+(vc*denom)[..., None]*ac
    bc = c-b
    e_bc = ((d4-d3)/((d4-d3)+(d5-d6)).clamp_min(eps))[..., None]
    q = torch.where(((va <= 0) & (d4-d3 >= 0) & (d5-d6 >= 0))[..., None], b+e_bc*bc, q)
    e_ac = (d2/(d2-d6).clamp_min(eps))[..., None]
    q = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a+e_ac*ac, q)
    e_ab = (d1/(d1-d3).clamp_min(eps))[..., None]
    q = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a+e_ab*ab, q)
    q = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c.expand_as(q), q)
    q = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b.expand_as(q), q)
    q = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a.expand_as(q), q)
    return torch.linalg.norm(p-q, dim=-1)


def hull_planes(verts, faces):
    """Outward unit normals (K, 3) and offsets (K,) of a convex hull's triangles (n . x <= off inside)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = torch.cross(b-a, c-a, dim=-1)
    n = n/torch.linalg.norm(n, dim=-1, keepdim=True)
    return n, (n*a).sum(-1)


def hull_sdf_exact(points, verts, faces, chunk=4096):
    """Exact signed distance (N,) of points (N, 3) to a convex hull (verts (n, 3), outward triangles (k, 3)).
    Inside, the distance to the nearest face plane is the distance to the boundary (convexity)."""
    verts = torch.as_tensor(verts, dtype=points.dtype)
    faces = torch.as_tensor(faces, dtype=torch.long)
    n, off = hull_planes(verts, faces)
    a, b, c = (verts[faces[:, i]][None] for i in range(3))
    out = []
    for p in points.split(chunk):
        inside = (p @ n.T-off).max(-1).values
        dist = point_triangle_distance(p[:, None], a, b, c).min(-1).values
        out.append(torch.where(inside <= 0, inside, dist))
    return torch.cat(out)


class HullGrid:
    """Signed-distance grid of one convex hull in its geom frame, trilinear lookup (differentiable in the points).
    Outside the grid box: value at the clamped point + distance to the box (an upper bound; the grid is padded so
    that contact-relevant queries stay inside)."""

    def __init__(self, verts, faces, spacing=1e-3, pad=5e-3, dtype=torch.float64, device='cpu'):
        v = torch.as_tensor(verts, dtype=torch.float64)
        lo, hi = v.min(0).values-pad, v.max(0).values+pad
        n = torch.ceil((hi-lo)/spacing).long()+1
        hi = lo+(n-1)*spacing
        axes = [torch.linspace(float(lo[i]), float(hi[i]), int(n[i]), dtype=torch.float64) for i in range(3)]
        pts = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        self.values = hull_sdf_exact(pts, v, faces).reshape(*n.tolist()).to(dtype=dtype, device=device)
        self.lo, self.hi = lo.to(dtype=dtype, device=device), hi.to(dtype=dtype, device=device)
        self.spacing = spacing
        self._vol = self.values.permute(2, 1, 0)[None, None]          # (1, 1, D=z, H=y, W=x): grid (x, y, z) order

    def __call__(self, p):
        shape = p.shape[:-1]
        p = p.reshape(-1, 3)
        pc = torch.minimum(torch.maximum(p, self.lo), self.hi)
        g = 2*(pc-self.lo)/(self.hi-self.lo)-1
        val = torch.nn.functional.grid_sample(self._vol, g[None, :, None, None, :].to(self._vol.dtype), mode='bilinear',
                                              align_corners=True).reshape(-1)
        return (val+torch.linalg.norm(p-pc, dim=-1)).reshape(shape)
