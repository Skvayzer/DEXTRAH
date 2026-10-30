"""MuJoCo's soft-constraint law (engine_core_constraint.c: getimpedance, mj_makeImpedance) and a small solver.

For a row with violation r = pos - margin (< 0 when active), velocity v and diagonal approximation Â:
    d = impedance(|r|) from solimp (dmin, dmax, width, midpoint, power)
    k = 1 / (dmax² tc² ζ²), b = 2 / (dmax tc), tc = max(solref[0], 2 dt) with refsafe
    a_ref = -b v - k d r,     R = max(1e-15, (1 - d) / d · Â)
The forces f of the active rows minimise ½ fᵀ(A + R)f + fᵀ(a0 - a_ref) with A = J M⁻¹ Jᵀ, a0 = J qacc_smooth, under
f >= 0 for unilateral (frictionless normal / limit) rows. Elliptic friction rows are added in WP4.
"""
import torch

MINIMP, MAXIMP, MINVAL = 1e-4, .9999, 1e-15


def impedance(solimp, r):
    """solimp (5,) or (..., 5); r (...) violation (pos - margin) -> d (...)."""
    dmin = solimp[..., 0].clamp(MINIMP, MAXIMP)
    dmax = solimp[..., 1].clamp(MINIMP, MAXIMP)
    width, mid, power = solimp[..., 2], solimp[..., 3], solimp[..., 4]
    x = (r/width).abs()
    a = 1/mid**(power-1)
    b = 1/(1-mid)**(power-1)
    xc = x.clamp(0., 1.)
    y = torch.where(xc <= mid, a*xc**power, 1-b*(1-xc)**power)
    y = torch.where(x >= 1, torch.ones_like(y), y)
    flat = (dmin == dmax) | (width <= MINVAL)
    return torch.where(flat, (dmin+dmax)/2*torch.ones_like(y), dmin+y*(dmax-dmin))


def stiffness_damping(solref, solimp, timestep, refsafe=True):
    """(k, b) of the standard (timeconst, dampratio) solref."""
    tc, zeta = solref[..., 0], solref[..., 1]
    if refsafe:
        tc = torch.clamp_min(tc, 2*timestep)
    dmax = solimp[..., 1].clamp(MINIMP, MAXIMP)
    return 1/(dmax*dmax*tc*tc*zeta*zeta), 2/(dmax*tc)


def reference(solref, solimp, r, v, diag, timestep, refsafe=True):
    """(a_ref, R, d) per row."""
    d = impedance(solimp, r)
    k, b = stiffness_damping(solref, solimp, timestep, refsafe)
    return -b*v-k*d*r, torch.clamp_min((1-d)/d*diag, MINVAL), d


def solve_unilateral(A, R, c, active, iters=50):
    """min ½ fᵀ(A + diag R)f + fᵀc, f >= 0 on active rows (f = 0 elsewhere); projected Gauss-Seidel, batched.
    A (B, n, n), R, c, active (B, n). Exact in one sweep for a single row (f = max(0, -c / (A + R)))."""
    n = c.shape[-1]
    H = A+torch.diag_embed(R)
    f = torch.zeros_like(c)
    act = active.to(c.dtype)
    for _ in range(iters if n > 1 else 1):
        for i in range(n):
            g = (H[:, i]*f).sum(-1)-H[:, i, i]*f[:, i]+c[:, i]
            fi = torch.clamp_min(-g/H[:, i, i], 0.)*act[:, i]
            f = torch.cat((f[:, :i], fi[:, None], f[:, i+1:]), -1)
    return f
