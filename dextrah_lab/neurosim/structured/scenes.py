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


def random_tool(rng, handle=None, head=None, mass=None):
    """Tool dict for TestbedSpec(object_kind='tool', tool=...)."""
    handle = handle or rng.choice(['box', 'capsule'])
    head = head or rng.choice(['box', 'capsule'])
    if handle == 'box':
        hs = np.array([rng.uniform(.008, .016), rng.uniform(.008, .016), rng.uniform(.04, .08)])
        top = hs[2]
        hsize = hs
    else:
        r, h = rng.uniform(.008, .016), rng.uniform(.03, .07)
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


def grip_scene(tb, rng, tool, close_scale=(.6, 1.), max_tries=200, table_top=.8, height=1.2):
    """A non-overlapping start: handle across the palm at the finger roots, random palm orientation."""
    r = tool['handle_radius']
    for _ in range(max_tries):
        axis_h = axis_angle(random_unit(rng), np.radians(rng.uniform(0, 20))) @ np.array([0., 1., 0.])
        c_h = np.array([.021+r+rng.uniform(0, .008), rng.uniform(-.015, .025), rng.uniform(.045, .095)])
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
        return GripScene(palm, obj, CLOSE_TARGETS*rng.uniform(*close_scale), tool)
    return None


def tool_testbed(tool, geometry='mesh', contact=None):
    return Testbed(TestbedSpec(geometry=geometry, object_kind='tool', tool={k: tool[k] for k in
                                                                          ('size', 'pos', 'quat', 'used', 'mass', 'inertia')}),
                   contact=contact)
