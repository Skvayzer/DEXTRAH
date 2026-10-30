"""MuJoCo's soft-constraint law (engine_core_constraint.c: getimpedance, mj_makeImpedance) and its primal solve.

Per row, with violation r = pos - margin (< 0 when active), constraint-space velocity v and diagonal approximation Â:
    d = impedance(|r|) from solimp (dmin, dmax, width, midpoint, power)
    k = 1 / (dmax² tc² ζ²), b = 2 / (dmax tc), tc = max(solref[0], 2 dt) with refsafe
    a_ref = -b v - k d r,     R = max(1e-15, (1 - d) / d · Â)
The constrained acceleration x minimises MuJoCo's primal cost (unique, convex)
    ½ (x - a0)ᵀ M (x - a0) + Σ s(J x - a_ref)
with, for y = J x - a_ref and D = 1/R:
    unilateral rows (joint limits, frictionless contacts): s = ½ D y² if y < 0 else 0
    elliptic contacts (condim 3 / 4: normal, 2 tangential, torsional): in the scaled space U_0 = μ y_0, U_j = μ_j y_j,
        N = U_0, T = |U_1..|: top zone N >= μ T: 0; bottom zone μ N + T <= 0: ½ Σ D_j y_j²;
        middle zone: ½ D_m (N - μ T)², D_m = D_0 / (μ² (1 + μ²))
and the constraint force is f = -∂s/∂y. M is the plain mass matrix (implicitfast's damping does not enter the solve).
Solved by Newton's method with a backtracking line search; batched over worlds. Forward exactness only: gradients
through the solve come later (implicit differentiation, WP7).
"""
from dataclasses import dataclass

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


@dataclass
class Rows:
    """Constraint rows of one solve (padded, masked).
    unilateral: J (B, nu, n), aref (B, nu), R (B, nu), active (B, nu)
    elliptic:   J (B, nc, dim, n), aref (B, nc, dim), R (B, nc, dim), active (B, nc), mu (B, nc) (= friction[0]),
                fric (B, nc, dim-1) (friction coefficients of the tangential / torsional rows)"""
    uJ: torch.Tensor = None
    uaref: torch.Tensor = None
    uR: torch.Tensor = None
    uactive: torch.Tensor = None
    eJ: torch.Tensor = None
    earef: torch.Tensor = None
    eR: torch.Tensor = None
    eactive: torch.Tensor = None
    mu: torch.Tensor = None
    fric: torch.Tensor = None


def _unilateral(y, D, active):
    neg = (y < 0)&active
    f = torch.where(neg, -D*y, torch.zeros_like(y))
    return (.5*D*y*y*neg).sum(-1), f, D*neg


def _elliptic(y, D, mu, fric, active):
    """Cost (B,), force (B, nc, dim), Hessian (B, nc, dim, dim) of the elliptic-cone rows."""
    scale = torch.cat((mu[..., None], fric), -1)                      # U = scale * y
    U = scale*y
    N = U[..., 0]
    Tv = U[..., 1:]
    T = torch.sqrt((Tv*Tv).sum(-1).clamp_min(1e-30))
    Tpos = (Tv*Tv).sum(-1) > 0
    top = (N >= mu*T) | (~Tpos & (N >= 0))
    bottom = ~top & ((mu*N+T <= 0) | (~Tpos & (N < 0)))
    middle = ~top & ~bottom
    top, bottom, middle = top & active, bottom & active, middle & active
    # bottom: quadratic in every row
    cost_b = .5*(D*y*y).sum(-1)
    f_b = -D*y
    H_b = torch.diag_embed(D)
    # middle: ½ Dm (N - μ T)² in U space
    Dm = D[..., 0]/(mu*mu*(1+mu*mu))
    NmT = N-mu*T
    cost_m = .5*Dm*NmT*NmT
    dU = torch.cat(((Dm*NmT)[..., None], -(Dm*NmT*mu/T)[..., None]*Tv), -1)     # ∂cost/∂U
    f_m = -dU*scale
    # Hessian in U space: Dm [g gᵀ + (N - μT) ∂²(N - μT)], g = (1, -μ Tv / T)
    g = torch.cat((torch.ones_like(N)[..., None], -(mu/T)[..., None]*Tv), -1)
    t = Tv/T[..., None]
    k = Tv.shape[-1]
    eye = torch.eye(k, dtype=y.dtype, device=y.device)
    curv = -(mu/T)[..., None, None]*(eye-t[..., :, None]*t[..., None, :])        # ∂²(N - μT)/∂Tv²
    HU = Dm[..., None, None]*g[..., :, None]*g[..., None, :]
    HU = HU+torch.nn.functional.pad((Dm*NmT)[..., None, None]*curv, (1, 0, 1, 0))
    H_m = scale[..., :, None]*HU*scale[..., None, :]
    w_b, w_m = bottom.to(y.dtype), middle.to(y.dtype)
    cost = (w_b*cost_b+w_m*cost_m).sum(-1)
    f = w_b[..., None]*f_b+w_m[..., None]*f_m
    H = w_b[..., None, None]*H_b+w_m[..., None, None]*H_m
    return cost, f, H


def evaluate(x, M, a0, rows):
    """Primal cost (B,), gradient (B, n), Hessian (B, n, n), constraint forces (dict)."""
    dx = x-a0
    Mdx = (M @ dx[..., None]).squeeze(-1)
    cost = .5*(dx*Mdx).sum(-1)
    grad, H = Mdx, M.clone()
    forces = {}
    if rows.uJ is not None and rows.uJ.shape[1]:
        y = (rows.uJ @ x[..., None]).squeeze(-1)-rows.uaref
        c, f, h = _unilateral(y, 1/rows.uR, rows.uactive)
        cost = cost+c
        grad = grad-(rows.uJ.transpose(-1, -2) @ f[..., None]).squeeze(-1)
        H = H+rows.uJ.transpose(-1, -2) @ (h[..., None]*rows.uJ)
        forces['unilateral'] = f
    if rows.eJ is not None and rows.eJ.shape[1]:
        y = (rows.eJ @ x[:, None, :, None]).squeeze(-1)-rows.earef
        c, f, h = _elliptic(y, 1/rows.eR, rows.mu, rows.fric, rows.eactive)
        cost = cost+c
        grad = grad-torch.einsum('bcdn,bcd->bn', rows.eJ, f)
        H = H+torch.einsum('bcdn,bcde,bcem->bnm', rows.eJ, h, rows.eJ)
        forces['elliptic'] = f
    return cost, grad, H, forces


def solve_primal(M, a0, rows, iters=50, tol=1e-12, ls_iters=30):
    """Newton on MuJoCo's primal; returns (qacc (B, n), forces dict). M (B, n, n), a0 = M⁻¹ qfrc_smooth (B, n)."""
    x = a0.clone()
    cost, grad, H, forces = evaluate(x, M, a0, rows)
    scale = torch.diagonal(M, dim1=-2, dim2=-1).mean(-1)
    for _ in range(iters):
        step = -torch.linalg.solve(H, grad[..., None]).squeeze(-1)
        dec = -(grad*step).sum(-1)                                   # Newton decrement
        if bool((dec <= tol*scale).all()):
            break
        alpha = torch.ones_like(cost)
        for _ in range(ls_iters):
            c_new = evaluate(x+alpha[..., None]*step, M, a0, rows)[0]
            ok = c_new <= cost-.25*alpha*dec+1e-300
            if bool(ok.all()):
                break
            alpha = torch.where(ok, alpha, .5*alpha)
        x = x+alpha[..., None]*step
        cost, grad, H, forces = evaluate(x, M, a0, rows)
    return x, forces
