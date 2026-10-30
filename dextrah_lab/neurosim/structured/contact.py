"""Contact generation and contact rows (WP3/WP4), mirroring MuJoCo where it is closed-form.

Pairs and generation:
- object part - table (box): box parts contribute their 8 corners, capsule parts their 2 segment ends (minus the
  radius), cylinders 2 rims of `rim_points` points; each candidate is a contact when inside the table box
  (MuJoCo's box-box / capsule-box give the penetrating corners / ends: 8 contacts for a box tool resting flat).
- hand geom - object part: up to `per_pair` contacts per pair: the deepest witness point, then farthest-point
  sampling among the penetrating witnesses (MuJoCo's multi-CCD gives 1-5 contacts per hand-cylinder pair, mostly 1
  per hull-box pair); candidates are the hand geom's witness points inside the object part and the part's witness
  points inside the hand geom (hull SDF grid or analytic primitive).
- hand geom - table: one contact per hand geom at its deepest witness point.
Normals point from body A (table / object) to body B (object / hand). Contact position: midway between surfaces.

Rows (MuJoCo condim 4, elliptic): normal, two tangents, torsion; Â from body_invweight0 (translational for the
first three rows, rotational for torsion), friction rows scaled R_j = R_0 (f_0 / f_j)² / impratio, contact
μ = f_0 / sqrt(impratio), friction combined by max over the two geoms.
"""
from dataclasses import dataclass

import math
import torch

from .rotations import quat_to_mat
from .sdf import HullGrid, PRIMITIVES, box as box_sdf


def _fibonacci_sphere(n, dtype):
    i = torch.arange(n, dtype=dtype)+.5
    phi = torch.acos(1-2*i/n)
    th = math.pi*(1+5**.5)*i
    return torch.stack((torch.cos(th)*torch.sin(phi), torch.sin(th)*torch.sin(phi), torch.cos(phi)), -1)


def witness_points(geom, dtype=torch.float64, density=2e-3):
    """Surface points of a geom in its geom frame (n, 3)."""
    s = torch.as_tensor(geom.size, dtype=dtype)
    if geom.kind == 'mesh':
        v = torch.as_tensor(geom.hull_vert, dtype=dtype)
        f = torch.as_tensor(geom.hull_face, dtype=torch.long)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        pts = [v, (a+b+c)/3, (a+b)/2, (b+c)/2, (c+a)/2]
        return torch.unique(torch.cat(pts), dim=0)
    if geom.kind == 'sphere':
        return s[0]*_fibonacci_sphere(64, dtype)
    if geom.kind == 'capsule':
        r, h = s[0], s[1]
        n_ax = max(2, int(2*h/density)+1)
        ring = 16
        z = torch.linspace(-float(h), float(h), n_ax, dtype=dtype)
        ang = torch.arange(ring, dtype=dtype)*2*math.pi/ring
        side = torch.stack((r*torch.cos(ang)[None].expand(n_ax, -1), r*torch.sin(ang)[None].expand(n_ax, -1),
                            z[:, None].expand(-1, ring)), -1).reshape(-1, 3)
        sph = r*_fibonacci_sphere(64, dtype)
        caps = torch.cat((sph[sph[:, 2] > 0]+torch.stack((0*h, 0*h, h)), sph[sph[:, 2] < 0]-torch.stack((0*h, 0*h, h))))
        return torch.cat((side, caps))
    if geom.kind == 'box':
        n = [max(2, int(2*float(x)/density)+1) for x in s]
        axes = [torch.linspace(-float(x), float(x), k, dtype=dtype) for x, k in zip(s, n)]
        g = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        on = (g.abs() >= s-1e-12).any(-1)
        return g[on]
    if geom.kind == 'cylinder':
        r, h = s[0], s[1]
        ring = 32
        ang = torch.arange(ring, dtype=dtype)*2*math.pi/ring
        rim = torch.stack((r*torch.cos(ang), r*torch.sin(ang)), -1)
        n_ax = max(2, int(2*h/density)+1)
        z = torch.linspace(-float(h), float(h), n_ax, dtype=dtype)
        side = torch.cat((rim[None].expand(n_ax, -1, -1), z[:, None, None].expand(-1, ring, 1)), -1).reshape(-1, 3)
        disc = [rim*f for f in (.25, .5, .75)]+[torch.zeros(1, 2, dtype=dtype)]
        disc = torch.cat(disc)
        faces = torch.cat((torch.cat((disc, torch.full_like(disc[:, :1], float(h))), -1),
                           torch.cat((disc, torch.full_like(disc[:, :1], -float(h))), -1)))
        return torch.cat((side, faces))
    raise ValueError(geom.kind)


def table_candidates(geom, dtype=torch.float64, rim_points=8):
    """(points (n, 3) in the geom frame, radius) of an object part's MuJoCo-style contact candidates on the table."""
    s = torch.as_tensor(geom.size, dtype=dtype)
    if geom.kind == 'box':
        c = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=dtype)
        return c*s, 0.
    if geom.kind == 'capsule':
        return torch.stack((torch.zeros(3, dtype=dtype), torch.zeros(3, dtype=dtype)))+torch.tensor([[0, 0, -1.], [0, 0, 1.]], dtype=dtype)*s[1], float(s[0])
    if geom.kind == 'cylinder':
        ang = torch.arange(rim_points, dtype=dtype)*2*math.pi/rim_points
        rim = torch.stack((s[0]*torch.cos(ang), s[0]*torch.sin(ang)), -1)
        return torch.cat((torch.cat((rim, torch.full_like(rim[:, :1], float(s[1]))), -1),
                          torch.cat((rim, torch.full_like(rim[:, :1], -float(s[1]))), -1))), 0.
    if geom.kind == 'sphere':
        return torch.zeros(1, 3, dtype=dtype), float(s[0])
    raise ValueError(geom.kind)


def make_frame(n):
    """Tangent basis (t1, t2) for unit normals n (..., 3) (any orthonormal choice: the elliptic cone is isotropic)."""
    a = torch.where((n[..., :1].abs() < .9), torch.tensor([1., 0., 0.], dtype=n.dtype),
                    torch.tensor([0., 1., 0.], dtype=n.dtype))
    t1 = torch.nn.functional.normalize(torch.cross(n, a.expand_as(n), dim=-1), dim=-1)
    return t1, torch.cross(n, t1, dim=-1)


@dataclass
class Contacts:
    pos: torch.Tensor            # (B, K, 3) world
    normal: torch.Tensor         # (B, K, 3) from A to B
    dist: torch.Tensor           # (B, K)
    active: torch.Tensor         # (B, K)
    hand_body: torch.Tensor      # (K,) hand body index of side B (hand pairs) else -1
    obj_side: torch.Tensor       # (K,) +1 object is B (object-table), -1 object is A (hand-object), 0 not involved
    friction: torch.Tensor       # (K, 3) combined (max) sliding, torsional, rolling
    kind: list                   # per slot: 'object_table' | 'hand_object' | 'hand_table'


def _grad(fn, p):
    with torch.enable_grad():
        q = p.detach().requires_grad_(True)
        d = fn(q)
        g, = torch.autograd.grad(d.sum(), q)
    return torch.nn.functional.normalize(g, dim=-1)


class ContactModel:
    def __init__(self, scene, dtype=torch.float64, device='cpu', grid_spacing=1e-3, rim_points=8, hand_table=True,
                 witness_density=2e-3, per_pair=1):
        self.scene, self.dtype, self.device = scene, dtype, device
        self.per_pair = per_pair
        t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        self.hand_geoms = scene.hand.geoms
        self.obj_geoms = scene.obj.geoms
        self.grids, self.hand_wit, self.obj_wit = [], [], []
        for g in self.hand_geoms:
            self.grids.append(HullGrid(g.hull_vert, g.hull_face, grid_spacing, dtype=dtype, device=device) if g.kind == 'mesh' else None)
            self.hand_wit.append(witness_points(g, dtype, witness_density).to(device))
        for g in self.obj_geoms:
            self.obj_wit.append(witness_points(g, dtype, witness_density).to(device))
        self.table_cand = [table_candidates(g, dtype, rim_points) for g in self.obj_geoms]
        self.table_pos, self.table_half = t(scene.table_pos), t(scene.table_half)
        self.hand_table = hand_table
        self.condim = scene.condim
        # slot layout
        kind, hand_body, obj_side, fric = [], [], [], []
        mx = lambda a, b: torch.maximum(t(a), t(b))
        table_f = t(scene.table_friction) if getattr(scene, 'table_friction', None) is not None else None
        for gi, g in enumerate(self.obj_geoms):
            n = self.table_cand[gi][0].shape[0]
            kind += ['object_table']*n
            hand_body += [-1]*n
            obj_side += [1]*n
            fric += [mx(g.friction, table_f if table_f is not None else g.friction)]*n
        for hi, h in enumerate(self.hand_geoms):
            for gi, g in enumerate(self.obj_geoms):
                kind += ['hand_object']*per_pair
                hand_body += [h.body]*per_pair
                obj_side += [-1]*per_pair
                fric += [mx(h.friction, g.friction)]*per_pair
        if hand_table:
            for h in self.hand_geoms:
                kind.append('hand_table')
                hand_body.append(h.body)
                obj_side.append(0)
                fric.append(mx(h.friction, table_f if table_f is not None else h.friction))
        self.kind = kind
        self.hand_body = torch.tensor(hand_body, device=device)
        self.obj_side = torch.tensor(obj_side, device=device)
        self.friction = torch.stack(fric)

    # --- geometry helpers ---------------------------------------------------------------------------------------
    def _table_sdf(self, p):
        return box_sdf(p-self.table_pos, self.table_half)

    def _hand_sdf(self, i, p_local):
        g = self.hand_geoms[i]
        if g.kind == 'mesh':
            return self.grids[i](p_local)
        return PRIMITIVES[g.kind](p_local, torch.as_tensor(g.size, dtype=p_local.dtype, device=p_local.device))

    def _obj_sdf(self, i, p_local):
        g = self.obj_geoms[i]
        return PRIMITIVES[g.kind](p_local, torch.as_tensor(g.size, dtype=p_local.dtype, device=p_local.device))

    @staticmethod
    def _frame(body_pos, body_R, geom):
        gp = torch.as_tensor(geom.pos, dtype=body_pos.dtype, device=body_pos.device)
        gR = quat_to_mat(torch.as_tensor(geom.quat, dtype=body_pos.dtype, device=body_pos.device))
        return body_pos+(body_R @ gp), body_R @ gR

    @staticmethod
    def _select(dist, world, K, min_sep=2e-3):
        """Up to K contact candidates per pair: the deepest, then farthest-point sampling among the penetrating
        candidates (at least min_sep apart). Returns (indices (B, K), valid (B, K))."""
        i0 = dist.argmin(1)
        sel, valid = [i0], [torch.ones_like(i0, dtype=torch.bool)]
        mind = torch.linalg.norm(world-world.gather(1, i0[:, None, None].expand(-1, 1, 3)), dim=-1)
        for _ in range(K-1):
            score = torch.where(dist < 0, mind, torch.full_like(mind, -1.))
            best, nxt = score.max(1)
            sel.append(nxt)
            valid.append(best >= min_sep)
            mind = torch.minimum(mind, torch.linalg.norm(world-world.gather(1, nxt[:, None, None].expand(-1, 1, 3)), dim=-1))
        return torch.stack(sel, 1), torch.stack(valid, 1)

    # --- generation ---------------------------------------------------------------------------------------------
    def generate(self, kin, obj_pos, obj_R):
        """kin: HandKinematics.fk output; obj_pos (B, 3) COM, obj_R (B, 3, 3). Returns Contacts."""
        P, N, D = [], [], []
        # object - table
        for gi, g in enumerate(self.obj_geoms):
            gpos, gR = self._frame(obj_pos, obj_R, g)
            cand, rad = self.table_cand[gi]
            w = gpos[:, None]+torch.einsum('bij,kj->bki', gR, cand)                     # (B, k, 3)
            d = self._table_sdf(w)-rad
            n = _grad(self._table_sdf, w)
            P.append(w-n*(rad+.5*d[..., None]))
            N.append(n)
            D.append(d)
        # hand - object
        hpose = [self._frame(kin['xpos'][:, h.body], kin['xR'][:, h.body], h) for h in self.hand_geoms]
        opose = [self._frame(obj_pos, obj_R, g) for g in self.obj_geoms]
        for hi, h in enumerate(self.hand_geoms):
            hp, hR = hpose[hi]
            wh = hp[:, None]+torch.einsum('bij,kj->bki', hR, self.hand_wit[hi])            # hand witnesses, world
            for gi, g in enumerate(self.obj_geoms):
                op, oR = opose[gi]
                # hand points in the object part, object points in the hand geom
                loc = torch.einsum('bki,bij->bkj', wh-op[:, None], oR)
                d1 = self._obj_sdf(gi, loc)
                wo = op[:, None]+torch.einsum('bij,kj->bki', oR, self.obj_wit[gi])
                loc2 = torch.einsum('bki,bij->bkj', wo-hp[:, None], hR)
                d2 = self._hand_sdf(hi, loc2)
                dist = torch.cat((d1, d2), 1)
                world = torch.cat((wh, wo), 1)
                sel, valid = self._select(dist, world, self.per_pair)                  # (B, K) candidate indices
                is1 = sel < d1.shape[1]
                i1, i2 = sel.clamp_max(d1.shape[1]-1), (sel-d1.shape[1]).clamp_min(0)
                p1 = loc.gather(1, i1[..., None].expand(-1, -1, 3))
                p2 = loc2.gather(1, i2[..., None].expand(-1, -1, 3))
                n1 = torch.einsum('bij,bkj->bki', oR, _grad(lambda x: self._obj_sdf(gi, x), p1))     # object -> hand
                n2 = -torch.einsum('bij,bkj->bki', hR, _grad(lambda x: self._hand_sdf(hi, x), p2))
                dk = torch.where(valid, dist.gather(1, sel), torch.ones_like(dist[:, :1]))     # unused slot: inactive
                wk = world.gather(1, sel[..., None].expand(-1, -1, 3))
                n = torch.where(is1[..., None], n1, n2)
                # midway between the surfaces (d < 0): hand witness moves against n, object witness along n
                pos = wk+torch.where(is1[..., None], -.5, .5)*dk[..., None]*n
                P.append(pos)
                N.append(n)
                D.append(dk)
        # hand - table
        if self.hand_table:
            for hi, h in enumerate(self.hand_geoms):
                hp, hR = hpose[hi]
                wh = hp[:, None]+torch.einsum('bij,kj->bki', hR, self.hand_wit[hi])
                d = self._table_sdf(wh)
                m, i = d.min(-1)
                w = wh.gather(1, i[:, None, None].expand(-1, 1, 3))[:, 0]
                n = _grad(self._table_sdf, w)
                P.append((w-.5*m[:, None]*n)[:, None])
                N.append(n[:, None])
                D.append(m[:, None])
        pos, normal, dist = torch.cat(P, 1), torch.cat(N, 1), torch.cat(D, 1)
        return Contacts(pos=pos, normal=normal, dist=dist, active=dist < 0, hand_body=self.hand_body,
                        obj_side=self.obj_side, friction=self.friction, kind=self.kind)

    # --- rows ---------------------------------------------------------------------------------------------------
    def rows(self, c, kin, hk, qd, palm_pos, palm_lin, palm_ang_world, obj_pos, obj_R, obj_v, obj_w, nq):
        """Constraint rows of contacts c over x = (finger qacc (nq), object linear acc (3, world), object angular
        acc (3, body)). Returns (J (B, K, dim, nq+6), vel (B, K, dim), diag Â (B, K, dim), mu (K,), fric (K, dim-1))."""
        sc = self.scene
        B, K = c.dist.shape
        dim = self.condim
        n = c.normal
        t1, t2 = make_frame(n)
        dirs = torch.stack((n, t1, t2), 2)[:, :, :min(dim, 3)]                         # (B, K, <=3, 3)
        hb = c.hand_body
        has_hand = hb >= 0
        hbc = hb.clamp_min(0)
        # hand side (B): finger columns and palm velocity
        Jp = hk.point_jacobian(kin, hbc, c.pos)*has_hand[None, :, None, None]           # (B, K, 3, nq)
        Jw = hk.rotation_jacobian(kin, hbc)*has_hand[None, :, None, None]
        lin_rows = torch.einsum('bkdi,bkin->bkdn', dirs, Jp)
        v_palm = palm_lin[:, None]+torch.cross(palm_ang_world[:, None].expand_as(c.pos), c.pos-palm_pos[:, None], dim=-1)
        v_hand = torch.einsum('bkdi,bki->bkd', dirs, v_palm)*has_hand[None, :, None]
        # object columns, sign s (+1 object is B, -1 object is A)
        s = c.obj_side.to(n.dtype)[None, :, None, None]
        r = c.pos-obj_pos[:, None]
        lin_o = s*dirs
        ang_o = s*torch.einsum('bkdi,bij->bkdj', torch.cross(r[:, :, None].expand_as(dirs), dirs, dim=-1), obj_R)
        J = torch.cat((lin_rows, lin_o, ang_o), -1)
        vel = (J @ torch.cat((qd, obj_v, obj_w), -1)[:, None, :, None]).squeeze(-1)+v_hand
        if dim == 4:
            tor_h = torch.einsum('bki,bkin->bkn', n, Jw)
            tor_o = c.obj_side.to(n.dtype)[None, :, None]*torch.einsum('bki,bij->bkj', n, obj_R)
            Jt = torch.cat((tor_h, torch.zeros_like(tor_o), tor_o), -1)[:, :, None]
            vt = (Jt @ torch.cat((qd, obj_v, obj_w), -1)[:, None, :, None]).squeeze(-1) \
                + (torch.einsum('bki,bi->bk', n, palm_ang_world)*has_hand[None])[..., None]
            J, vel = torch.cat((J, Jt), 2), torch.cat((vel, vt), 2)
        # diagonal approximation: translational invweight of both bodies; friction rows scaled
        inv_h = torch.as_tensor(sc.hand.body_invweight0[:, 0], dtype=n.dtype, device=n.device)[hbc]*has_hand
        inv_o = float(sc.obj.invweight0[0])*(c.obj_side != 0).to(n.dtype)
        A0 = (inv_h+inv_o)[None].expand(B, K)
        f = c.friction
        mu = f[:, 0]/math.sqrt(sc.impratio)
        fr = torch.stack((f[:, 0], f[:, 0], f[:, 1]), -1)[:, :dim-1]                    # tangential, tangential, torsion
        scale = torch.cat((torch.ones_like(f[:, :1]), (f[:, :1]/fr)**2/sc.impratio), -1)  # (K, dim)
        return J, vel, A0[..., None]*scale[None], mu, fr
