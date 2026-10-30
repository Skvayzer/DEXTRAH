"""Test and data scenes on the stage-3 testbed: random handle-head tools (M's 4-slot family) and grips.

Tools: a handle (box or capsule along the tool z axis) and a head (box or capsule) at the handle's top end,
body frame at the COM with principal axes = the tool axes (both parts are centred on the z axis, so the inertia
tensor is diagonal). Uniform density; mass drawn per tool. The real bank (play2perfect's URDF generator) is not
available locally; these tools span the same slot types and similar sizes.

Grip scenes: the handle lies across the palm (hand y axis) at the finger roots, the palm in a random orientation;
the fingers close on the tool while it is held fixed (Testbed.step hold_object), as in held_states.try_hold.
"""
from dataclasses import dataclass

import numpy as np

from ..reference.held_states import axis_angle, pose, random_unit, rotation_between
from ..testbed import CLOSE_TARGETS, Testbed, TestbedSpec

SQ = np.sqrt(.5)


def _part_mass_props(kind, size):
    """(volume, inertia diag about own centre per unit mass) of a box (half sizes) or a z capsule (r, half)."""
    if kind == 'box':
        hx, hy, hz = size
        return 8*hx*hy*hz, np.array([hy*hy+hz*hz, hx*hx+hz*hz, hx*hx+hy*hy])/3
    r, h = size[0], size[1]
    vc, vs = np.pi*r*r*2*h, 4/3*np.pi*r**3
    v = vc+vs
    # cylinder + two hemispheres (hemisphere COM offset 3r/8 from its flat face)
    ixx_c = vc*(r*r/4+(2*h)**2/12)
    izz_c = vc*r*r/2
    ixx_s = vs*(2*r*r/5)+vs*(h*h+3*h*r/4)
    izz_s = vs*2*r*r/5
    return v, np.array([ixx_c+ixx_s, ixx_c+ixx_s, izz_c+izz_s])/v


def random_tool(rng, handle=None, head=None, mass=None, handle_radius=None):
    """Tool dict for TestbedSpec(object_kind='tool', tool=...). handle_radius: capsule radius / box half width."""
    handle = handle or rng.choice(['box', 'capsule'])
    head = head or rng.choice(['box', 'capsule'])
    if handle == 'box':
        w = handle_radius if handle_radius is not None else rng.uniform(.008, .016)
        hs = np.array([w, w, rng.uniform(.04, .08)])
        top = hs[2]
        hsize = hs
    else:
        r = handle_radius if handle_radius is not None else rng.uniform(.008, .016)
        h = rng.uniform(.03, .07)
        hsize = np.array([r, h, 0.])
        top = h+r
    if head == 'box':
        es = np.array([rng.uniform(.01, .03), rng.uniform(.008, .016), rng.uniform(.01, .025)])
        hquat = np.array([1., 0., 0., 0.])
        ezsize = es[2]
        esize, ekind_size = es, es
    else:
        r, h = rng.uniform(.01, .02), rng.uniform(.015, .035)
        esize = np.array([r, h, 0.])
        hquat = np.array([SQ, SQ, 0., 0.])                       # capsule axis along the tool's y
        ezsize = r
        ekind_size = esize
    hpos, epos = np.zeros(3), np.array([0., 0., top+ezsize])
    vh, ih = _part_mass_props(handle, hsize)
    ve, ie = _part_mass_props(head, ekind_size)
    if head == 'capsule':                                        # rotated about x: y and z swap
        ie = ie[[0, 2, 1]]
    com = (vh*hpos+ve*epos)/(vh+ve)
    m = mass if mass is not None else rng.uniform(.1, .5)
    mh, me = m*vh/(vh+ve), m*ve/(vh+ve)
    dz_h, dz_e = hpos[2]-com[2], epos[2]-com[2]
    inertia = mh*ih+me*ie+np.array([mh*dz_h**2+me*dz_e**2, mh*dz_h**2+me*dz_e**2, 0.])
    size = np.zeros((4, 3))
    pos = np.zeros((4, 3))
    quat = np.tile([1., 0., 0., 0.], (4, 1))
    used = np.zeros(4, bool)
    hslot, eslot = (0 if handle == 'box' else 1), (2 if head == 'box' else 3)
    size[hslot], pos[hslot], used[hslot] = hsize, hpos-com, True
    size[eslot], pos[eslot], quat[eslot], used[eslot] = esize, epos-com, hquat, True
    for k in range(4):
        if not used[k]:
            size[k] = (1e-4, 1e-4, 1e-4) if k in (0, 2) else (1e-4, 1e-4, 0.)
    return dict(size=size, pos=pos, quat=quat, used=used, mass=float(m), inertia=inertia, handle=handle, head=head,
                handle_radius=float(hsize[0]), handle_centre=hpos-com)


@dataclass
class GripScene:
    palm: np.ndarray
    obj: np.ndarray              # object pose (7) held while closing
    targets: np.ndarray          # closing targets (11)
    tool: dict


def grip_scene(tb, rng, tool, close_scale=(.6, 1.), max_tries=200, table_top=.8, height=1.2, palm_up=None):
    """A non-overlapping start: handle across the palm at the finger roots, random palm orientation."""
    r = tool['handle_radius']
    for _ in range(max_tries):
        # measured on the hull hand (closing targets, 135 placements): the thumb opposes in 94% and the fingers reach
        # the handle when it lies within ~1 cm of the palm at 6-9 cm along the fingers
        axis_h = axis_angle(random_unit(rng), np.radians(rng.uniform(0, 10))) @ np.array([0., 1., 0.])
        c_h = np.array([.021+r+rng.uniform(0, .005), rng.uniform(-.015, .025), rng.uniform(.06, .09)])
        if palm_up:          # palm normal within palm_up degrees of vertical (the palm carries part of the load)
            from ..testbed import PALM_NORMAL
            pn = PALM_NORMAL/np.linalg.norm(PALM_NORMAL)
            up = axis_angle(random_unit(rng)*np.array([1., 1., 0.]), np.radians(rng.uniform(0, palm_up))) @ np.array([0., 0., 1.])
            R_hw = axis_angle(up, rng.uniform(0, 2*np.pi)) @ rotation_between(pn, up)
        else:
            R_hw = axis_angle(random_unit(rng), rng.uniform(0, np.pi))
        R_oh = rotation_between(np.array([0., 0., 1.]), axis_h) @ axis_angle(np.array([0., 0., 1.]), rng.uniform(0, 2*np.pi))
        handle_w = np.array([0., 0., height])
        R_ow = R_hw @ R_oh
        p_ow = handle_w-R_ow @ tool['handle_centre']+R_hw @ np.zeros(3)
        palm = pose(handle_w-R_hw @ c_h, R_hw)
        obj = pose(p_ow, R_ow)
        tb.reset(palm, np.zeros(11), obj, table_top=table_top)
        if any(c['pair'] == 'hand_object' and c['dist'] < 0 for c in tb.contacts()):
            continue
        targets = CLOSE_TARGETS*rng.uniform(*close_scale)
        # rev. 2.1 A3: the carry expert grips with thumb, index, middle; ring sometimes, pinky rarely. All four fingers
        # at their 2 N m effort overpower the 0.5-1.1 N m thumb and squeeze the tool out (measured)
        if rng.random() > .35:
            targets[7:9] = 0.
        if rng.random() > .05:
            targets[9:11] = 0.
        return GripScene(palm, obj, targets, tool)
    return None


@dataclass
class Hold:
    state: np.ndarray            # MuJoCo full-physics state after the settle (testbed.get_state)
    palm: np.ndarray
    targets: np.ndarray
    tool: dict
    gaps: np.ndarray             # targets - q at the settled state
    forces: dict                 # per-finger normal force at the settled state
    worst_mm: float
    worst_deg: float


def make_hold(tb, rng, tool, gap_mode="census", close_steps=90, settle_steps=30, shake_steps=30,
              shake_m=.003, shake_deg=3., max_mm=5., max_deg=5., palm_up=45., close_mode="locked"):
    """Close on the held tool, set the targets (gap_mode 'census': each touching joint q + |N(0, 10 mrad)| gap on the
    closing side, the size of the carry expert's in-contact gaps; 'deep': keep the closing targets), release under
    gravity, settle 0.5 s, then a ±3 mm / ±3 deg palm shake over 0.5 s (tool_holds.accept's test). Returns a Hold
    if the tool stays within 5 mm / 5 deg of its settled pose in the palm frame, else the rejection reason (str)."""
    from ..reference.held_states import mat
    from ..testbed import HAND_JOINTS
    g = grip_scene(tb, rng, tool, palm_up=palm_up)
    if g is None:
        return 'no_start'
    d = tb.d
    # Closing on a fixed tool at the closing targets lets the saturated fingers sink 10-20 mm into it, and the stored
    # penetration launches it when freed; closing on a free tool pushes it away before the thumb arrives (both
    # measured). So: tool held fixed, each finger's command ramps at 1 rad/s and is locked 10 mrad past its first
    # touch; then the tool is freed with gravity off and its 6 DoF damped, and gravity is ramped back.
    g0 = tb.m.opt.gravity.copy()
    dofs = slice(tb.object_dadr, tb.object_dadr+6)
    damp0 = tb.m.dof_damping[dofs].copy()
    ramp = np.zeros(11)
    locked = np.zeros(11, bool)
    rate = 2*tb.m.opt.timestep
    groups = dict(thumb=[0, 1, 2], index=[3, 4], middle=[5, 6], ring=[7, 8], pinky=[9, 10])
    if close_mode == 'fixed':          # close straight to the closing targets on the fixed tool
        for _ in range(45):
            tb.step(g.targets, g.palm, hold_object=g.obj)
        ramp = g.targets.copy()
    else:
        for _ in range(close_steps):
            forces = tb.finger_forces()
            q = d.qpos[tb.qadr]
            for f, js in groups.items():
                if forces[f] > .5 and not locked[js[0]]:
                    locked[js] = True
                    ramp[js] = np.minimum(q[js]+.01, g.targets[js])
            ramp = np.where(locked, ramp, np.minimum(ramp+rate, g.targets))
            tb.step(ramp, g.palm, hold_object=g.obj)
    targets = ramp.copy()
    forces = tb.finger_forces()
    touching = [f for f in ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm') if forces[f] > .5]
    if 'thumb' not in touching or len(touching) < 3:            # needs an opposing thumb and two more contacts
        tb.m.opt.gravity[:], tb.m.dof_damping[dofs] = g0, damp0
        return 'no_grip'
    if gap_mode in ('census', 'balanced'):
        q = d.qpos[tb.qadr].copy()
        for k, j in enumerate(HAND_JOINTS):
            f = next(x for x in ('thumb', 'index', 'middle', 'ring', 'pinky') if x in j)
            if gap_mode == 'census':
                gap = abs(rng.normal(0., .01))
            else:        # thumb saturated, fingers inside their 1.67 mrad band: blocked-joint torque kp*gap ~ the thumb's
                gap = .01 if f == 'thumb' else rng.uniform(3e-4, 1.2e-3)
            targets[k] = q[k]+gap if forces[f] > .5 else 0.
        for _ in range(10):                                    # fingers relax to the new torques, tool still fixed
            tb.step(targets, g.palm, hold_object=g.obj)
    tb.m.opt.gravity[:] = 0.
    tb.m.dof_damping[dofs] = 2.
    for _ in range(15):
        tb.step(targets, g.palm)
    tb.m.dof_damping[dofs] = damp0
    for k in range(10):
        tb.m.opt.gravity[:] = g0*(k+1)/10
        tb.step(targets, g.palm)
    tb.m.opt.gravity[:] = g0
    for _ in range(settle_steps):
        tb.step(targets, g.palm)
    if sum(v > .5 for v in tb.finger_forces().values()) < 2:
        return 'dropped'
    palm = g.palm
    Rp = mat(palm[3:])
    o = tb.object_qadr
    rel0 = (Rp.T @ (d.qpos[o:o+3]-palm[:3]), Rp.T @ mat(d.qpos[o+3:o+7]))
    state = tb.get_state()
    q_settled = d.qpos[tb.qadr].copy()
    forces = tb.finger_forces()
    sd, sa = random_unit(rng), random_unit(rng)
    worst_m = worst_deg = 0.
    prev = palm
    for k in range(shake_steps):
        ph = np.sin(2*np.pi*(k+1)/shake_steps)
        nxt = pose(palm[:3]+shake_m*ph*sd, axis_angle(sa, np.radians(shake_deg)*ph) @ Rp)
        tb.step(targets, prev, nxt)
        prev = nxt
        R = mat(d.qpos[tb.palm_qadr+3:tb.palm_qadr+7])
        p = R.T @ (d.qpos[o:o+3]-d.qpos[tb.palm_qadr:tb.palm_qadr+3])
        worst_m = max(worst_m, float(np.linalg.norm(p-rel0[0])))
        Rr = (R.T @ mat(d.qpos[o+3:o+7])).T @ rel0[1]
        worst_deg = max(worst_deg, float(np.degrees(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))))
    if 1000*worst_m > max_mm or worst_deg > max_deg:
        return 'shifted'
    return Hold(state, palm, targets, tool, targets-q_settled, forces, 1000*worst_m, worst_deg)


def tool_testbed(tool, geometry='mesh', contact=None):
    return Testbed(TestbedSpec(geometry=geometry, object_kind='tool', tool={k: tool[k] for k in
                                                                          ('size', 'pos', 'quat', 'used', 'mass', 'inertia')}),
                   contact=contact)


GROUPS = dict(thumb=[0, 1, 2], index=[3, 4], middle=[5, 6], ring=[7, 8], pinky=[9, 10])


def _hand_hulls_in_base(tb, q):
    """[(finger, vertices (n, 3), faces)] of the hand hulls in the hand-base frame at joint angles q."""
    import mujoco
    from .spec import extract
    sc = extract(tb)
    d = tb.d
    d.qpos[tb.palm_qadr:tb.palm_qadr+7] = (0., 0., 0., 1., 0., 0., 0.)
    d.qpos[tb.qadr] = q
    mujoco.mj_kinematics(tb.m, d)
    out = []
    for g in sc.hand.geoms:
        gid = g.mujoco_id
        out.append((sc.hand.finger_of_body[g.body], d.geom_xpos[gid]+g.hull_vert @ d.geom_xmat[gid].reshape(3, 3).T, g.hull_face))
    return out


def caged_hold(rng, gap_mode='saturated', palm_up=45., handle=None, posture=(.3, 1.), iters=300, settle_steps=30,
               shake_steps=30, shake_m=.003, shake_deg=3., max_mm=5., max_deg=5., trace=False):
    """A hold built without closing dynamics: the hand takes a closed posture (A3 finger choice), the largest handle
    segment that fits inside the finger cage is found (exact hull distances, random search in the base frame), a
    tool with that handle radius is placed there, the fingers press (targets = q + gap), gravity is ramped in, and
    the hold is tested with tool_holds.accept's settle + ±3 mm / ±3 deg shake. Returns a Hold or a reason (str)."""
    import torch
    from ..reference.held_states import mat
    from .sdf import hull_sdf_exact
    from ..testbed import CLOSE_TARGETS
    probe = tool_testbed(random_tool(rng))
    q = CLOSE_TARGETS*rng.uniform(*posture)
    use = dict(thumb=True, index=True, middle=True, ring=rng.random() < .35, pinky=rng.random() < .05)
    for f, js in GROUPS.items():
        if not use[f]:
            q[js] = 0.
    hulls = _hand_hulls_in_base(probe, q)
    half = rng.uniform(.03, .06)
    T = lambda x: torch.as_tensor(x, dtype=torch.float64)
    tt = torch.linspace(-1., 1., 9, dtype=torch.float64)

    def clearance(c, a):
        pts = T(c)[None]+tt[:, None]*half*T(a)[None]
        per, near = {}, {}
        for f, v, fc in hulls:
            dmin = float(hull_sdf_exact(pts, v, fc).min())
            if dmin < per.get(f, 1.):
                # nearest hull vertex to the axis segment: its direction around the axis (for force closure)
                rel = v-c
                s = np.clip(rel @ a, -half, half)
                perp = rel-np.outer(s, a)
                near[f] = perp[np.argmin(np.linalg.norm(perp, axis=1))]
            per[f] = min(per.get(f, 1.), dmin)
        return min(per.values()), per, near

    def max_gap_deg(a, near, per, r):
        """Largest angular gap (deg) around the handle axis between the contacts within r + 3 mm."""
        e1 = np.cross(a, [1., 0., 0.] if abs(a[0]) < .9 else [0., 0., 1.])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        ang = sorted(float(np.degrees(np.arctan2(near[f] @ e2, near[f] @ e1)) % 360) for f in near if per[f] < r+.003)
        if len(ang) < 2:
            return 360.
        return float(np.diff(ang+[ang[0]+360]).max())
    # inscribed handle touching thumb, index and middle together: maximise r = min clearance (<= 20 mm), penalising
    # any of those three more than 3 mm farther than r, and contacts that do not surround the axis (largest angular
    # gap > 150 deg: no force closure); plain max-clearance finds empty space outside the hand. Random search in the
    # base frame (x: palm side, z: along the fingers), axis near hand y, then local refinement
    req = ('thumb', 'index', 'middle')

    def score(c, a):
        cl, per, near = clearance(c, a)
        gap = max_gap_deg(a, near, per, cl)
        others = sorted(per[f] for f in per if f != "thumb")
        s = min(cl, .02)-3*max(0., cl-.02)-3*max(0., per["thumb"]-cl-.003)-3*max(0., others[1]-cl-.003)-1e-4*max(0., gap-150.)
        return s, cl, (per, gap)
    best = (-1e9, None, None, None, None)
    for k in range(iters):
        if k < iters//2 or best[1] is None:
            c = np.array([rng.uniform(.02, .08), rng.uniform(-.03, .06), rng.uniform(.03, .12)])
            a = axis_angle(random_unit(rng), np.radians(rng.uniform(0, 25))) @ np.array([0., 1., 0.])
        else:
            c = best[3]+rng.normal(0, .004, 3)
            a = axis_angle(random_unit(rng), np.radians(rng.normal(0, 4))) @ best[4]
        s, cl, per = score(c, a)
        if s > best[0]:
            best = (s, cl, per, c, a)
    _, r, (per, gap), c, a = best
    if r < .006:
        return 'no_cage'
    touching = [f for f, dd in per.items() if dd < r+.003]
    if 'thumb' not in touching or len(touching) < 3 or gap > 150.:
        return 'open_cage'
    radius = min(r-3e-4, .02)
    handle = handle or rng.choice(['box', 'capsule'])
    tool = random_tool(rng, handle=handle, handle_radius=radius if handle == 'capsule' else radius/np.sqrt(2))
    tb = tool_testbed(tool)
    if palm_up:
        from ..testbed import PALM_NORMAL
        pn = PALM_NORMAL/np.linalg.norm(PALM_NORMAL)
        up = axis_angle(random_unit(rng)*np.array([1., 1., 0.]), np.radians(rng.uniform(0, palm_up))) @ np.array([0., 0., 1.])
        R_hw = axis_angle(up, rng.uniform(0, 2*np.pi)) @ rotation_between(pn, up)
    else:
        R_hw = axis_angle(random_unit(rng), rng.uniform(0, np.pi))
    centre_w = np.array([0., 0., 1.2])
    palm = pose(centre_w-R_hw @ c, R_hw)
    R_ow = R_hw @ rotation_between(np.array([0., 0., 1.]), a)
    obj = pose(centre_w-R_ow @ tool['handle_centre'], R_ow)
    tb.reset(palm, q, obj, table_top=.8)
    if any(x['pair'] == 'hand_object' and x['dist'] < -1e-3 for x in tb.contacts()):
        return 'overlap'
    gap = dict(saturated=lambda f: .01, census=lambda f: abs(rng.normal(0., .01)),
               balanced=lambda f: .01 if f == 'thumb' else rng.uniform(3e-4, 1.2e-3))[gap_mode]
    targets = np.zeros(11)
    for f, js in GROUPS.items():
        if use[f]:
            for j in js:
                targets[j] = q[j]+gap(f)
    d = tb.d
    g0 = tb.m.opt.gravity.copy()
    o = tb.object_qadr
    p_start = d.qpos[o:o+3].copy()
    if trace:
        print('  cage r %.1f mm, clearances mm %s, contacts at start %s' % (
            1e3*r, {f: round(1e3*v, 1) for f, v in per.items()}, {k: round(v, 1) for k, v in tb.finger_forces().items() if v > 0}))
    for k in range(10):
        tb.m.opt.gravity[:] = g0*(k+1)/10
        tb.step(targets, palm)
        if trace:
            print('  ramp %d: tool moved %.1f mm, speed %.2f m/s, forces %s' % (
                k, 1e3*np.linalg.norm(d.qpos[o:o+3]-p_start), np.linalg.norm(d.qvel[tb.object_dadr:tb.object_dadr+3]),
                {f: round(v) for f, v in tb.finger_forces().items() if v > .5}))
    tb.m.opt.gravity[:] = g0
    for _ in range(settle_steps):
        tb.step(targets, palm)
    if sum(v > .5 for v in tb.finger_forces().values()) < 2:
        return 'dropped'
    Rp = mat(palm[3:])
    o = tb.object_qadr
    rel0 = (Rp.T @ (d.qpos[o:o+3]-palm[:3]), Rp.T @ mat(d.qpos[o+3:o+7]))
    state, q_settled, forces = tb.get_state(), d.qpos[tb.qadr].copy(), tb.finger_forces()
    sd, sa = random_unit(rng), random_unit(rng)
    worst_m = worst_deg = 0.
    prev = palm
    for k in range(shake_steps):
        ph = np.sin(2*np.pi*(k+1)/shake_steps)
        nxt = pose(palm[:3]+shake_m*ph*sd, axis_angle(sa, np.radians(shake_deg)*ph) @ Rp)
        tb.step(targets, prev, nxt)
        prev = nxt
        R = mat(d.qpos[tb.palm_qadr+3:tb.palm_qadr+7])
        p = R.T @ (d.qpos[o:o+3]-d.qpos[tb.palm_qadr:tb.palm_qadr+3])
        worst_m = max(worst_m, float(np.linalg.norm(p-rel0[0])))
        Rr = (R.T @ mat(d.qpos[o+3:o+7])).T @ rel0[1]
        worst_deg = max(worst_deg, float(np.degrees(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))))
    if 1000*worst_m > max_mm or worst_deg > max_deg:
        return 'shifted'
    return Hold(state, palm, targets, tool, targets-q_settled, forces, 1000*worst_m, worst_deg)


@dataclass
class EpisodeLog:
    tool: dict
    states: list                 # MuJoCo full-physics state at the start of each control step
    palm: np.ndarray             # (T+1, 7) palm pose at each control step
    targets: np.ndarray          # (T, 11)
    phase: list                  # script phase per step
    obj: np.ndarray              # (T+1, 7) object pose
    forces: list                 # per-finger normal forces per step
    table_force: np.ndarray      # (T,) object-table normal force


def table_episode(rng, tool=None, close_scale=(.6, 1.), lift=.06, speed=.15):
    """M-like scripted episode on the testbed: the tool lies flat on the table; the palm comes down over the handle
    (palm facing down, the handle across the palm), the fingers close (A3 finger choice), the palm lifts, holds
    with a small wiggle, lowers the tool back, the fingers open and the palm retreats. Every control step's state is
    logged; situations are labelled afterwards from contacts and relative motion (census.label)."""
    from ..reference.held_states import mat
    from ..testbed import CLOSE_TARGETS, FINGER_DIR, PALM_NORMAL
    tool = tool or random_tool(rng)
    tb = tool_testbed(tool)
    top = .8
    yaw = rng.uniform(0, 2*np.pi)
    # tool lying with its z axis horizontal along u
    u = np.array([np.cos(yaw), np.sin(yaw), 0.])
    R_ow = rotation_between(np.array([0., 0., 1.]), u)
    R_ow = axis_angle(u, rng.uniform(0, 2*np.pi)) @ R_ow
    # resting height: lowest point of the handle / head over the table
    tb.reset(np.r_[0., .6, 1.4, 1., 0., 0., 0.], np.zeros(11), np.r_[0., 0., 1.0, quat_wxyz(R_ow)], table_top=top)
    import mujoco
    m, d = tb.m, tb.d
    low = min(float((d.geom_xpos[g]+0).__getitem__(2)-_geom_extent_down(m, d, g)) for g in tb.object_geoms)
    obj = np.r_[0., 0., 1.0-(low-top)+5e-4, quat_wxyz(R_ow)]
    tb.reset(np.r_[0., .6, 1.4, 1., 0., 0., 0.], np.zeros(11), obj, table_top=top)
    for _ in range(20):
        tb.step(np.zeros(11), np.r_[0., .6, 1.4, 1., 0., 0., 0.])
    # palm: normal down, hand y along the handle, fingers horizontal
    pn = PALM_NORMAL/np.linalg.norm(PALM_NORMAL)
    R1 = rotation_between(pn, np.array([0., 0., -1.]))
    yh = R1 @ np.array([0., 1., 0.])
    yh_h = yh-yh[2]*np.array([0., 0., 1.])
    ang = np.arctan2(np.cross(yh_h, u)[2], yh_h @ u)
    R_hw = axis_angle(np.array([0., 0., 1.]), ang) @ R1
    hc = d.qpos[tb.object_qadr:tb.object_qadr+3]+mat(d.qpos[tb.object_qadr+3:tb.object_qadr+7]) @ tool['handle_centre']
    c_h = np.array([.021+tool['handle_radius']+rng.uniform(.001, .006), rng.uniform(-.015, .025), rng.uniform(.06, .09)])
    grasp = pose(hc-R_hw @ c_h, R_hw)
    above = grasp.copy()
    above[2] += .08
    targets_close = CLOSE_TARGETS*rng.uniform(*close_scale)
    if rng.random() > .35:
        targets_close[7:9] = 0.
    if rng.random() > .05:
        targets_close[9:11] = 0.
    dt = 2*m.opt.timestep
    log = dict(states=[], palm=[], targets=[], phase=[], obj=[], forces=[], table_force=[])
    tb.reset(above, np.zeros(11), d.qpos[tb.object_qadr:tb.object_qadr+7].copy(), table_top=top)
    cur = above.copy()

    def run(target_pose, n, tgt_fn, phase):
        nonlocal cur
        start = cur.copy()
        for k in range(n):
            a = (k+1)/n
            nxt = start.copy()
            nxt[:3] = start[:3]+a*(target_pose[:3]-start[:3])
            nxt[3:] = target_pose[3:]
            tgt = tgt_fn(a)
            log['states'].append(tb.get_state())
            log['palm'].append(cur.copy())
            log['targets'].append(tgt.copy())
            log['phase'].append(phase)
            log['obj'].append(d.qpos[tb.object_qadr:tb.object_qadr+7].copy())
            tb.step(tgt, cur, nxt)
            log['forces'].append(tb.finger_forces())
            log['table_force'].append(sum(float(c['force_local'][0]) for c in tb.contacts() if c['pair'] == 'object_table'))
            cur = nxt
    n_desc = max(2, int(.08/speed/dt))
    run(grasp, n_desc, lambda a: np.zeros(11), 'approach')
    run(grasp, 45, lambda a: np.minimum(targets_close, a*1.5*targets_close), 'close')
    up = grasp.copy()
    up[2] += lift
    run(up, max(2, int(lift/speed/dt)), lambda a: targets_close, 'lift')
    wig = up.copy()
    for k in range(3):
        w = up.copy()
        w[:3] += rng.normal(0, .01, 3)
        run(w, 10, lambda a: targets_close, 'hold')
    run(up, 10, lambda a: targets_close, 'hold')
    run(grasp+np.r_[0., 0., .003, 0., 0., 0., 0.], max(2, int(lift/speed/dt)), lambda a: targets_close, 'lower')
    run(cur, 30, lambda a: targets_close*(1-a), 'release')
    run(above, n_desc, lambda a: np.zeros(11), 'retreat')
    log['palm'].append(cur.copy())
    log['obj'].append(d.qpos[tb.object_qadr:tb.object_qadr+7].copy())
    return EpisodeLog(tool=tool, states=log['states'], palm=np.array(log['palm']), targets=np.array(log['targets']),
                      phase=log['phase'], obj=np.array(log['obj']), forces=log['forces'],
                      table_force=np.array(log['table_force'])), tb


def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def _geom_extent_down(m, d, g):
    """Distance from the geom centre to its lowest point along world -z."""
    R = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    if m.geom_type[g] == 6:                                       # box
        return float(np.abs(R[2]) @ s)
    return float(s[0]+abs(R[2, 2])*s[1])                          # capsule
