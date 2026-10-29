"""Held-state generator (stage 3 plan rev. 2.1, Phase 0.1; amendments A2, A3).

Rejection sampling with the object frozen while the fingers close, instead of a scripted grasp (scripted side grasps
failed 0/72: the thumb sweeps mostly along the hand's y axis and pushed the cylinder away before any finger arrived).

One trial:
1. A cylinder in a power-grasp pose (hand frame: axis across the palm along the hand's y axis, jittered 20 deg, its
   surface 0-8 mm from the palm at the finger roots); the palm is oriented so that the cylinder axis is near vertical
   in the world (tilt up to 15 deg, P's domain), in the air above the table. Initial hand-object penetration rejects
   the trial.
2. Gripping fingers drawn per trial (A3: thumb and index always, middle 0.9, ring 0.35, pinky 0.05); the others stay
   open. The object is held fixed; the fingers close at full speed, the thumb only after the first finger touches
   or 10 steps, and keep closing until blocked (they wrap the fixed object). Each touching finger's commands are then
   set to its joint angle plus a gap drawn from the closing side of the carry expert's in-contact gap distribution
   (A2; CENSUS_GAP), so the servos sit where the policy keeps them; fingers that never touched open. One command
   drives a proximal joint and its coupled distal joint, so the command is raised where needed to keep the distal
   joint pushing as well (else its target falls below the wrapped distal angle and the saturated servo pulls it open).
   The fingers close with gravity switched off and the object free, so the squeeze settles into its own force
   balance (a weld hid the imbalance between the 2 N m fingers and the 1.1 N m thumb: welded grips of ~110 N dropped
   the object on release).
3. Gravity ramped back on over 10 steps, settle 0.5 s, shake the palm (1 cm, 5 deg, one 2 Hz cycle), settle 0.25 s.
   Kept if the object moved less than 1 cm and 10 deg relative to the palm under gravity and the shake, with at
   least two contact sources (fingers or palm).

Commands follow the training interface: 6 actuated joint targets (rad), distal targets by coupling.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..testbed import HAND_JOINTS, TABLE_SIZE, Testbed, TestbedSpec, CylinderSpec, quat_from_matrix

COMMAND_JOINTS = ('right_thumb_metacarpal_joint', 'right_thumb_proximal_joint', 'right_index_proximal_joint',
                  'right_middle_proximal_joint', 'right_ring_proximal_joint', 'right_pinky_proximal_joint')
COUPLING = {'right_thumb_distal_joint': ('right_thumb_proximal_joint', 1.0),
            'right_index_distal_joint': ('right_index_proximal_joint', 1.155),
            'right_middle_distal_joint': ('right_middle_proximal_joint', 1.155),
            'right_ring_distal_joint': ('right_ring_proximal_joint', 1.155),
            'right_pinky_distal_joint': ('right_pinky_proximal_joint', 1.155)}
DISTAL_OF = {src: j for j, (src, _) in COUPLING.items()}
FINGER_COMMANDS = dict(thumb=(0, 1), index=(2,), middle=(3,), ring=(4,), pinky=(5,))
GRIP_PROBABILITY = dict(thumb=1., index=1., middle=.9, ring=.35, pinky=.05)
CONTACT_N = .5

# Carry expert C v2 (run 1587), stage 1 log, in-contact signed gap = target applied during the step - joint angle after
# it (rad), quantiles at CENSUS_P (2026-10-29; docs/neurosim_stage3_plan.md A). Pinky never touched: ring's is used.
CENSUS_P = np.clip(np.linspace(0., 1., 21), .005, .995)
CENSUS_GAP = {
    'right_thumb_metacarpal_joint': [-0.28824, -0.04278, -0.0265, -0.02048, -0.01672, -0.01398, -0.01175, -0.00979, -0.008,
                                     -0.00634, -0.00476, -0.00322, -0.00169, -0.00018, 0.00138, 0.00303, 0.00483, 0.00696,
                                     0.00982, 0.01595, 0.37751],
    'right_thumb_proximal_joint': [-0.00914, -0.00369, -0.00113, 0.00093, 0.00276, 0.00446, 0.00608, 0.00765, 0.00924,
                                   0.01095, 0.01287, 0.01509, 0.01777, 0.02115, 0.02553, 0.0321, 0.04521, 0.08476, 0.16613,
                                   0.28838, 0.95531],
    'right_index_proximal_joint': [-0.01652, -0.01017, -0.00699, -0.00467, -0.0027, -0.00089, 0.00087, 0.0026, 0.00428,
                                   0.00595, 0.00762, 0.00931, 0.01104, 0.01284, 0.01476, 0.01694, 0.01961, 0.02333, 0.02963,
                                   0.04589, 0.33712],
    'right_middle_proximal_joint': [-0.01933, -0.01269, -0.0097, -0.00728, -0.0052, -0.00334, -0.00157, 0.00017, 0.00187,
                                    0.00353, 0.00515, 0.00677, 0.0084, 0.01006, 0.01181, 0.01368, 0.01601, 0.01951, 0.0259,
                                    0.04216, 0.2158],
    'right_ring_proximal_joint': [-0.01972, -0.01242, -0.00947, -0.00714, -0.00512, -0.00327, -0.00152, 0.00018, 0.00182,
                                  0.00344, 0.00501, 0.00661, 0.00825, 0.00996, 0.01175, 0.01369, 0.01595, 0.01915, 0.0255,
                                  0.04959, 0.36598]}
CENSUS_GAP['right_pinky_proximal_joint'] = CENSUS_GAP['right_ring_proximal_joint']


def targets_from_commands(cmd):
    """6 actuated targets (COMMAND_JOINTS order, rad) -> 11 targets in HAND_JOINTS order (distals by coupling)."""
    value = dict(zip(COMMAND_JOINTS, cmd))
    for j, (src, k) in COUPLING.items():
        value[j] = k*value[src]
    return np.array([value[j] for j in HAND_JOINTS])


def sample_gap(joint, rng, closing=True):
    """A gap from the census distribution; closing=True draws from its closing side (gap > 0): a static hold needs
    every gripping joint to push (a saturated servo with its target on the opening side pulls away at full effort;
    in the policy's data those moments come with palm or dynamic support)."""
    q = np.asarray(CENSUS_GAP[joint])
    if closing:
        p0 = float(np.interp(0., q, CENSUS_P))              # probability mass below zero
        return float(np.interp(p0+(1-p0)*rng.random(), CENSUS_P, q))
    return float(np.interp(rng.random(), CENSUS_P, q))


def rotation_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.array([1., 0., 0.]) if abs(a[0]) < .9 else np.array([0., 1., 0.])
        p -= (p @ a)*a
        p /= np.linalg.norm(p)
        return 2*np.outer(p, p)-np.eye(3)
    K = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3)+K+K @ K*(1/(1+c))


def axis_angle(axis, angle):
    axis = axis/np.linalg.norm(axis)
    K = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*K @ K


def random_unit(rng):
    v = rng.normal(size=3)
    return v/np.linalg.norm(v)


def pose(p, R):
    return np.concatenate((p, quat_from_matrix(R)))


def mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


@dataclass
class HeldStateSpec:
    max_tilt_deg: float = 15.
    axis_jitter_deg: float = 20.              # cylinder axis around the hand's y axis
    palm_gap_m: tuple = (0., .008)            # cylinder surface to the palm surface (x = 0.021 m)
    across_m: tuple = (-.015, .025)           # cylinder centre along the hand's y axis
    along_m: tuple = (.045, .095)             # cylinder centre along the fingers (finger roots at z ~0.081 m)
    air_clearance: float = .15                # object bottom above the table top (m)
    close_steps: int = 90
    thumb_delay_steps: int = 0
    close_rate: float = 1.                    # rad/s ramp of the closing commands
    close_object_damping: float = 2.          # N s/m and N m s/rad on the object's 6 DoF while closing
    settle_steps: int = 30
    shake_steps: int = 30
    after_shake_steps: int = 15
    shake_m: float = .01
    shake_deg: float = 5.
    max_slip_m: float = .01
    max_slip_deg: float = 10.
    max_eject_m: float = .05                  # the gravity-free squeeze may move the object this far in the hand
    gravity_ramp_steps: int = 10


def relative_pose(tb):
    d = tb.d
    pp, pq = d.qpos[tb.palm_qadr:tb.palm_qadr+3], d.qpos[tb.palm_qadr+3:tb.palm_qadr+7]
    op, oq = d.qpos[tb.object_qadr:tb.object_qadr+3], d.qpos[tb.object_qadr+3:tb.object_qadr+7]
    Rp = mat(pq)
    return Rp.T @ (op-pp), Rp.T @ mat(oq)


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1., 1.))))


def try_hold(tb, rng, hs=HeldStateSpec(), table_top=None, mass=None):
    """One trial on testbed tb (its cylinder size). Returns (record or None, reason)."""
    m, d = tb.m, tb.d
    r, half = float(m.geom_size[tb.object_geom][0]), float(m.geom_size[tb.object_geom][1])
    if mass is not None:                                   # same shape, scaled density (from the compiled values)
        if not hasattr(tb, 'base_mass'):
            tb.base_mass, tb.base_inertia = float(m.body_mass[tb.object_body]), m.body_inertia[tb.object_body].copy()
        m.body_mass[tb.object_body] = mass
        m.body_inertia[tb.object_body] = tb.base_inertia*mass/tb.base_mass
    top = table_top if table_top is not None else rng.uniform(.70, .90)
    # 1. object pose in the hand frame, palm orientation making the cylinder axis near vertical
    # power-grasp prior (renders, 2026-10-29): axis across the palm (hand y, the index-pinky direction), against the
    # palm at the finger roots, so the fingers wrap it and the thumb closes over; jittered
    axis_h = axis_angle(random_unit(rng), np.radians(rng.uniform(0., hs.axis_jitter_deg))) @ np.array([0., 1., 0.])
    c_h = np.array([.021+r+rng.uniform(*hs.palm_gap_m), rng.uniform(*hs.across_m), rng.uniform(*hs.along_m)])
    tilt = np.radians(rng.uniform(0., hs.max_tilt_deg))
    up = axis_angle(random_unit(rng)*np.array([1., 1., 0.]), tilt) @ np.array([0., 0., 1.])
    R_hw = rotation_between(axis_h, up) @ axis_angle(axis_h, rng.uniform(0., 2*np.pi))
    R_oh = rotation_between(np.array([0., 0., 1.]), axis_h)
    R_ow = R_hw @ R_oh
    p_ow = np.array([0., 0., top+hs.air_clearance+half+r])
    palm = pose(p_ow-R_hw @ c_h, R_hw)
    obj = pose(p_ow, R_ow)
    tb.reset(palm, np.zeros(len(HAND_JOINTS)), obj, table_top=top)
    if any(c['pair'] == 'hand_object' and c['dist'] < 0 for c in tb.contacts()):
        return None, 'overlap'
    g0 = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.                                  # close with gravity off: the squeeze finds its own balance
    dofs = slice(tb.object_dadr, tb.object_dadr+6)
    damp0 = m.dof_damping[dofs].copy()
    m.dof_damping[dofs] = hs.close_object_damping          # the object drifts into balance instead of being knocked away
    # 2. fingers close on the fixed object; touching fingers get a census gap
    grip = [f for f, pr in GRIP_PROBABILITY.items() if rng.random() < pr]
    cmd = np.zeros(6)
    locked = {}
    lo = np.array([m.jnt_range[m.joint('hand/'+j).id][0] for j in COMMAND_JOINTS])
    hi = np.array([m.jnt_range[m.joint('hand/'+j).id][1] for j in COMMAND_JOINTS])
    qadr = np.array([m.jnt_qposadr[m.joint('hand/'+j).id] for j in COMMAND_JOINTS])
    first_touch = None
    for k in range(hs.close_steps):                        # close until blocked (the fingers wrap the fixed object)
        forces = tb.finger_forces()
        for f in grip:
            if forces[f] > CONTACT_N and f not in locked:
                locked[f] = k
                first_touch = k if first_touch is None else first_touch
            if f != 'thumb' or k >= hs.thumb_delay_steps or first_touch is not None:
                for i in FINGER_COMMANDS[f]:
                    cmd[i] = min(hi[i], cmd[i]+hs.close_rate*tb.m.opt.timestep*2)   # slow ramp toward closed
        tb.step(targets_from_commands(cmd), palm)
    forces = tb.finger_forces()
    touching = [f for f in grip if forces[f] > CONTACT_N]
    if len(touching)+(forces['palm'] > CONTACT_N) < 2:
        m.opt.gravity[:] = g0
        m.dof_damping[dofs] = damp0
        return None, 'no_grip'
    for f in grip:                                         # touching fingers: joint angle + a census gap; others open
        for i in FINGER_COMMANDS[f]:
            if f not in touching:
                cmd[i] = 0.
                continue
            gap = sample_gap(COMMAND_JOINTS[i], rng)
            c = d.qpos[qadr[i]]+gap
            distal = DISTAL_OF.get(COMMAND_JOINTS[i])
            if distal is not None:                         # one command drives both joints: keep the distal pushing too
                k = COUPLING[distal][1]
                c = max(c, (d.qpos[m.jnt_qposadr[m.joint('hand/'+distal).id]]+gap)/k)
            cmd[i] = np.clip(c, lo[i], hi[i])
    targets = targets_from_commands(cmd)
    for _ in range(10):                                    # the grip settles at the policy gaps, still without gravity
        tb.step(targets, palm)
    m.dof_damping[dofs] = damp0
    for _ in range(5):
        tb.step(targets, palm)
    pg, Rg = relative_pose(tb)
    if np.linalg.norm(pg-c_h) > hs.max_eject_m:
        m.opt.gravity[:] = g0
        return None, 'squeezed_out'
    # 3. gravity back on (ramped), settle, shake, settle
    for k in range(hs.gravity_ramp_steps):
        m.opt.gravity[:] = g0*(k+1)/hs.gravity_ramp_steps
        tb.step(targets, palm)
    m.opt.gravity[:] = g0
    for _ in range(hs.settle_steps):
        tb.step(targets, palm)
    p0, R0 = relative_pose(tb)
    if np.linalg.norm(p0-pg) > hs.max_slip_m or rotation_deg(R0.T @ Rg) > hs.max_slip_deg:
        return None, 'slipped_under_gravity'
    sd, sa = random_unit(rng), random_unit(rng)
    prev = palm
    for k in range(hs.shake_steps):
        ph = np.sin(2*np.pi*(k+1)/hs.shake_steps)
        nxt = pose(palm[:3]+hs.shake_m*ph*sd, axis_angle(sa, np.radians(hs.shake_deg)*ph) @ R_hw)
        tb.step(targets, prev, nxt)
        prev = nxt
    for _ in range(hs.after_shake_steps):
        tb.step(targets, palm)
    p1, R1 = relative_pose(tb)
    forces = tb.finger_forces()
    touching = [f for f in ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm') if forces[f] > CONTACT_N]
    if np.linalg.norm(p1-p0) > hs.max_slip_m or rotation_deg(R1.T @ R0) > hs.max_slip_deg:
        return None, 'slipped_in_shake'
    if len(touching) < 2:
        return None, 'lost_contact'
    q_cmd = d.qpos[qadr]
    record = dict(state=tb.get_state(), commands=cmd.copy(), gaps=(cmd-q_cmd).copy(), grip=grip, touching=touching,
                  forces=forces, table_top=top, mass=float(m.body_mass[tb.object_body]), radius=r, height=2*half,
                  palm=d.qpos[tb.palm_qadr:tb.palm_qadr+7].copy(), object_in_hand=(p1, R1),
                  tilt_deg=float(np.degrees(np.arccos(np.clip(mat(d.qpos[tb.object_qadr+3:tb.object_qadr+7])[2, 2], -1., 1.)))),
                  slip_mm=1000*float(np.linalg.norm(p1-p0)))
    return record, 'held'
