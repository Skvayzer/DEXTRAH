"""Stage 3 testbed (2026-10-29): Revo2 right hand + one solid cylinder + the table, in C MuJoCo, training physics.

The reference simulator for the physics-structured contact model (docs/neurosim_design.md, "Stage 3 proposal").
Everything that sets contact behaviour is copied from the whole-body manipulation task (mjlab, run 1454 and
after): physics step 1/120 s, 2 substeps per 60 Hz control step, implicitfast integrator, Newton solver, elliptic
friction cones, impratio 10, 20 (line-search) iterations, condim 4 with ContactParams friction / solref / solimp,
the training robot spec's hand (convex hulls <= 64 vertices, velocity servos robot.hand_servo, armature 1e-3),
robot / world collision masks (no finger self-collision), table = mocap box TABLE_SIZE.

The palm moves on a prescribed trajectory: each physics substep sets its free joint's pose AND velocity, and a
large armature on those 6 DoF keeps contacts from moving it within a step (an infinitely stiff arm). A mocap body
has no velocity in MuJoCo's contact solve, so a carried object would only be dragged by penetration; a weld to a
mocap target is either soft (sagged 12-55 mm under the fingers' weight) or unstable at dt = 1/120 s. Finger geometry: 'mesh' (the training hulls) or 'capsule' (one fitted capsule per collision
mesh, fit_capsules: an enclosing capsule or oriented box, whichever is tighter; the fingertip markers become
spheres).
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from dextrah_lab.mj.robot import ContactParams, WORLD_CONAFFINITY, WORLD_CONTYPE, build_robot_spec, hand_servo

TABLE_SIZE = (0.475, 0.4, 0.3)          # dextrah_lab.mj.scene.TABLE_SIZE (table_narrow.urdf)
PHYSICS_DT = 1/120
SUBSTEPS = 2                             # 60 Hz control
HAND_ROOT = 'right_base_link'
HAND_JOINTS = ('right_thumb_metacarpal_joint', 'right_thumb_proximal_joint', 'right_thumb_distal_joint',
               'right_index_proximal_joint', 'right_index_distal_joint', 'right_middle_proximal_joint',
               'right_middle_distal_joint', 'right_ring_proximal_joint', 'right_ring_distal_joint',
               'right_pinky_proximal_joint', 'right_pinky_distal_joint')
PREFIX = 'hand/'


def training_contact():
    return ContactParams(condim=4)


def set_training_options(spec):
    o = spec.option
    o.timestep = PHYSICS_DT
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    o.impratio = 10.
    o.iterations = 20
    o.ls_iterations = 20
    o.tolerance = 1e-8
    o.ls_tolerance = .01
    o.ccd_iterations = 50


def world_geom(geom, contact):
    """dextrah_lab.mj.scene._world_geom (object, table, ground)."""
    geom.contype, geom.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
    geom.condim = contact.condim
    geom.solref = contact.solref
    geom.solimp = contact.solimp
    geom.margin = geom.gap = 0.
    geom.friction = (contact.friction, contact.torsional, contact.rolling)


def hand_mesh_model(contact=None):
    """The training robot compiled alone (for the hand's collision meshes and capsule fits)."""
    return build_robot_spec(contact or training_contact()).compile()


def fit_capsule(points):
    """Smallest enclosing capsule along the principal axis of points (n, 3): (p0, p1, radius); a sphere when the
    caps cover the whole extent (p0 == p1)."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points-c, full_matrices=False)
    a = vt[0]
    t = (points-c) @ a
    rho = np.linalg.norm((points-c)-np.outer(t, a), axis=1)
    r = float(rho.max())
    reach = np.sqrt(np.maximum(r*r-rho*rho, 0.))
    lo, hi = float((t+reach).min()), float((t-reach).max())     # segment [lo, hi] keeps every point within r
    if lo > hi:
        lo = hi = .5*(lo+hi)
        r = float(np.max(np.linalg.norm(points-(c+lo*a), axis=1)))
    return c+lo*a, c+hi*a, r


def fit_box(points):
    """Oriented bounding box along the principal axes of points (n, 3): (centre, rotation (3, 3) columns = axes,
    half extents (3,))."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points-c, full_matrices=False)
    R = vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    local = (points-c) @ R
    lo, hi = local.min(0), local.max(0)
    centre = c+R @ ((lo+hi)/2)
    return centre, R, (hi-lo)/2


def hand_collision_meshes(m):
    """{body name: [(geom id, vertices (n, 3) in the body frame)]} of the right hand's collision meshes (the
    vertices MuJoCo builds the convex hull from)."""
    root = m.body(HAND_ROOT).id
    out = {}
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not (m.geom_contype[g] | m.geom_conaffinity[g]):
            continue
        x = b
        while x not in (0, root):
            x = int(m.body_parentid[x])
        if x != root:
            continue
        i = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i]+m.mesh_vertnum[i]].astype(np.float64)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        out.setdefault(m.body(b).name, []).append((g, m.geom_pos[g]+v @ R.reshape(3, 3).T))
    return out


def fit_capsules(m=None):
    """{body name: primitive} in the body frame, one per hand collision mesh: the enclosing capsule
    ('capsule', p0, p1, radius) or the oriented bounding box ('box', centre, R, half) of its convex hull,
    whichever is tighter (hull volume / primitive volume); + fit report."""
    from scipy.spatial import ConvexHull
    m = m or hand_mesh_model()
    caps, report = {}, {}
    for body, meshes in hand_collision_meshes(m).items():
        if len(meshes) != 1:
            raise ValueError(f'{body}: expected one collision mesh, got {len(meshes)}')
        v = meshes[0][1]
        hull = ConvexHull(v).volume
        p0, p1, r = fit_capsule(v)
        length = float(np.linalg.norm(p1-p0))
        cap_fill = hull/(np.pi*r*r*length+4/3*np.pi*r**3)
        c, R, half = fit_box(v)
        box_fill = hull/(8*float(np.prod(half)))
        if cap_fill >= box_fill:
            caps[body] = ('capsule', p0, p1, r)
            report[body] = dict(primitive='capsule', radius_mm=round(1000*r, 2), length_mm=round(1000*length, 2),
                                hull_fill=round(cap_fill, 3), box_fill=round(box_fill, 3))
        else:
            caps[body] = ('box', c, R, half)
            report[body] = dict(primitive='box', half_mm=np.round(1000*half, 2).tolist(), hull_fill=round(box_fill, 3),
                                capsule_fill=round(cap_fill, 3))
    return caps, report


@dataclass
class CylinderSpec:
    radius: float = .035
    height: float = .14
    mass: float = .3


@dataclass
class TestbedSpec:
    cylinder: CylinderSpec = field(default_factory=CylinderSpec)
    geometry: str = 'mesh'                     # 'mesh' | 'capsule'
    object_kind: str = 'cylinder'              # 'cylinder' | 'tool' (M's handle-head tools: compiled with tool=)
    # tool parts (object_kind 'tool'): dict(size (4, 3), pos (4, 3) at the COM, quat (4, 4), used (4,), mass, inertia (3,));
    # compiled into the model: writing per-object geometry into a compiled C MuJoCo model leaves stale compile data
    # (contact sets differed from a fresh compile in 30/30 restored holds, even with geom_sameframe cleared)
    tool: dict = None
    table_top: float = .80
    palm_armature: float = 1e3                 # kg / kg m^2 on the palm's 6 DoF (prescribed motion)
    substeps: int = SUBSTEPS                   # physics substeps per 60 Hz control step (timestep = 1/(60 substeps)); 2 = training
    # 'velocity': robot.hand_servo as trained (v_cmd = clip(g err, +-vmax), damping kv). Near the target the 120 Hz loop
    # is linearly unstable (growth 8.5-45x per substep for g dt = 11-52) and limit-cycles (~8 mrad, up to 2 rad/s).
    # 'deadbeat': the command never asks for more than the remaining error in one substep: the gain is capped at 1/dt,
    # v_cmd = clip(err / dt, +-vmax) (stable: spectral radius 0.35; g > 1/dt for every hand joint). Free speed unchanged; a blocked joint's torque is
    # kv |err| / dt up to the effort, i.e. linear up to ~vmax dt (19 mrad) instead of saturating at effort/kp (1.7 mrad).
    # (An implicit-position variant, damping kv (1 + g dt), was rejected: MuJoCo's implicitfast keeps that damping while
    # the joint force range clips, so free fingers creep at ~vmax / (1 + g dt).)
    hand_servo: str = 'velocity'


def build_spec(ts=TestbedSpec(), contact=None, capsules=None):
    contact = contact or training_contact()
    robot = build_robot_spec(contact)
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    set_training_options(spec)
    spec.option.timestep = 1/(60*ts.substeps)
    if ts.hand_servo not in ('velocity', 'deadbeat'):
        raise ValueError(f'hand_servo {ts.hand_servo}')
    ground = spec.worldbody.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0., 0., 1.))
    world_geom(ground, contact)
    table = spec.worldbody.add_body(name='table', mocap=True, pos=(0., 0., ts.table_top-TABLE_SIZE[2]/2))
    world_geom(table.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(s/2 for s in TABLE_SIZE)), contact)
    obj = spec.worldbody.add_body(name='object', pos=(0., 0., ts.table_top+ts.cylinder.height/2))
    obj.add_freejoint(name='object_free')
    if ts.object_kind == 'cylinder':
        g = obj.add_geom(name='object', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(ts.cylinder.radius, ts.cylinder.height/2, 0.))
        g.mass = ts.cylinder.mass                 # uniform density: inertia from the shape
        world_geom(g, contact)
    elif ts.object_kind == 'tool':
        # dextrah_lab.mj.scene.object_spec: body frame at the COM, 4 primitive slots (handle box / capsule, head box /
        # capsule) whose size, pose, rbound and aabb are written per object (set_tool), unused slots a 0.1 mm placeholder
        from dextrah_lab.mj.objects import PLACEHOLDER, SLOTS, SLOT_TYPES
        t = ts.tool
        if t is None:
            raise ValueError("object_kind 'tool' needs TestbedSpec.tool")
        obj.mass, obj.ipos, obj.inertia, obj.explicitinertial = float(t['mass']), (0., 0., 0.), tuple(t['inertia']), True
        for k, (name, kind) in enumerate(zip(SLOTS, SLOT_TYPES)):
            if not t['used'][k]:
                continue
            g = obj.add_geom(name='object_'+name, type=mujoco.mjtGeom.mjGEOM_BOX if kind == 'box' else mujoco.mjtGeom.mjGEOM_CAPSULE)
            g.size, g.pos, g.quat = tuple(t['size'][k]), tuple(t['pos'][k]), tuple(t['quat'][k])
            g.mass = 0.
            world_geom(g, contact)
    else:
        raise ValueError(f'object_kind {ts.object_kind}')
    # inactive weld of the object to the world: holds it (with its own mass and contact softness) while the fingers
    # close in the held-state generator; data = the object's pose, set when activated
    hold = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1='world', name2='object', objtype=mujoco.mjtObj.mjOBJ_BODY)
    hold.name = 'object_hold'
    hold.active = False
    hold.data[:] = (0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.)
    palm = spec.worldbody.add_body(name='palm', pos=(0., .25, ts.table_top+.25))
    palm_joint = palm.add_freejoint(name='palm_free')
    palm_joint.armature = ts.palm_armature
    palm.gravcomp = 1.
    root = palm.add_frame().attach_body(robot.body(HAND_ROOT), PREFIX, '')
    root.pos, root.quat = (0., 0., 0.), (1., 0., 0., 0.)        # palm pose = hand base pose
    for body in spec.bodies:
        if body.name.startswith(PREFIX) and body.name != PREFIX+HAND_ROOT:
            body.gravcomp = 0.                     # fingers feel gravity, as in training
    spec.body(PREFIX+HAND_ROOT).gravcomp = 1.
    existing = {a.name for a in spec.actuators}
    for j in HAND_JOINTS:
        if PREFIX+j in existing:
            continue
        act = spec.add_actuator(name=PREFIX+j, target=PREFIX+j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.set_to_velocity(kv=hand_servo(j)['kv'])
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    if ts.geometry == 'capsule':
        caps = capsules if capsules is not None else fit_capsules()[0]
        for geom in spec.geoms:
            body = geom.parent.name
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and body.startswith(PREFIX) and body[len(PREFIX):] in caps:
                prim = caps[body[len(PREFIX):]]
                geom.pos, geom.quat = (0., 0., 0.), (1., 0., 0., 0.)
                geom.meshname = ''
                if prim[0] == 'box':
                    _, c, R, half = prim
                    quat = np.zeros(4)
                    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R).reshape(-1))
                    geom.type = mujoco.mjtGeom.mjGEOM_BOX
                    geom.size = tuple(np.maximum(half, 1e-4))
                    geom.pos, geom.quat = tuple(c), tuple(quat)
                    continue
                _, p0, p1, r = prim
                if np.allclose(p0, p1):
                    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    geom.size = (max(r, 1e-4), 0., 0.)
                    geom.pos = tuple(p0)
                else:
                    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                    geom.fromto = (*p0, *p1)
                    geom.size = (max(r, 1e-4), 0., 0.)
    elif ts.geometry != 'mesh':
        raise ValueError(f'geometry {ts.geometry}')
    return spec


class Testbed:
    """One world: palm target (7: pos, wxyz), 11 finger joint targets per 60 Hz control step."""

    def __init__(self, ts=TestbedSpec(), contact=None, capsules=None):
        self.ts = ts
        self.spec = build_spec(ts, contact, capsules)
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.joint_ids = np.array([m.joint(PREFIX+j).id for j in HAND_JOINTS])
        self.qadr = m.jnt_qposadr[self.joint_ids]
        self.dadr = m.jnt_dofadr[self.joint_ids]
        self.act = np.array([m.actuator(PREFIX+j).id for j in HAND_JOINTS])
        servo = [hand_servo(j) for j in HAND_JOINTS]
        self.servo_gain = np.array([s['kp']/s['kv'] for s in servo])
        if ts.hand_servo == 'deadbeat':                  # gain capped at 1/dt: v_cmd = clip(err / dt, +-vmax) (g > 1/dt here)
            self.servo_gain = np.minimum(self.servo_gain, 1/self.m.opt.timestep)
        self.servo_vmax = np.array([s['vmax'] for s in servo])
        self.table_mocap = m.body('table').mocapid[0]
        self.object_body = m.body('object').id
        self.table_body = m.body('table').id
        self.palm_body = m.body('palm').id
        self.object_qadr = m.jnt_qposadr[m.joint('object_free').id]
        self.object_dadr = m.jnt_dofadr[m.joint('object_free').id]
        self.hold_eq = m.equality('object_hold').id
        self.finger_of_body = {}
        for b in range(m.nbody):
            n = m.body(b).name
            for f in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                if n.startswith(PREFIX+'right_'+f):
                    self.finger_of_body[b] = f
        self.palm_qadr = m.jnt_qposadr[m.joint('palm_free').id]
        self.palm_dadr = m.jnt_dofadr[m.joint('palm_free').id]
        root = m.body(PREFIX+HAND_ROOT).id
        hand_bodies = [b for b in range(m.nbody) if b == root or self._under(b, root)]
        self.hand_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_bodies
                                    and (m.geom_contype[g] | m.geom_conaffinity[g])])
        self.object_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.object_body])
        self.object_geom = int(self.object_geoms[0])
        self.table_geom = m.geom('table').id

    def _under(self, b, root):
        while b > 0:
            b = int(self.m.body_parentid[b])
            if b == root:
                return True
        return False

    def reset(self, palm_pose, q_hand, object_pose, table_top=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        if table_top is not None:
            d.mocap_pos[self.table_mocap] = (0., 0., table_top-TABLE_SIZE[2]/2)
        d.qpos[self.palm_qadr:self.palm_qadr+7] = palm_pose
        d.qpos[self.object_qadr:self.object_qadr+7] = object_pose
        d.qpos[self.qadr] = q_hand
        mujoco.mj_forward(m, d)

    def servo_ctrl(self, targets):
        """robot.hand_servo / MjArticulation.write_data_to_sim: velocity command from the position error."""
        err = targets-self.d.qpos[self.qadr]
        return np.clip(self.servo_gain*err, -self.servo_vmax, self.servo_vmax)

    def step(self, targets, palm_pose, palm_pose_next=None, hold_object=None):
        """One 60 Hz control step: the palm moves from palm_pose to palm_pose_next at constant linear and angular
        velocity (set with the pose at each physics substep); the servo command is recomputed each substep (as
        write_data_to_sim). hold_object: an object pose (7) kept fixed (pose re-set, velocity zeroed at every
        substep: an immovable object, e.g. while the fingers close on it)."""
        m, d = self.m, self.d
        nxt = palm_pose if palm_pose_next is None else palm_pose_next
        n_sub = self.ts.substeps
        T = n_sub*m.opt.timestep
        lin = (nxt[:3]-palm_pose[:3])/T
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, np.array([palm_pose[3], -palm_pose[4], -palm_pose[5], -palm_pose[6]]), nxt[3:7])
        ang = np.zeros(3)
        mujoco.mju_quat2Vel(ang, dq, T)                    # body-frame angular velocity (free joint convention)
        for k in range(n_sub):
            a = k/n_sub
            pose = np.empty(7)
            pose[:3] = palm_pose[:3]+a*(nxt[:3]-palm_pose[:3])
            q = palm_pose[3:7].copy()
            mujoco.mju_quatIntegrate(q, ang, a*T)
            pose[3:] = q
            d.qpos[self.palm_qadr:self.palm_qadr+7] = pose
            d.qvel[self.palm_dadr:self.palm_dadr+3] = lin
            d.qvel[self.palm_dadr+3:self.palm_dadr+6] = ang
            d.ctrl[self.act] = self.servo_ctrl(targets)
            if hold_object is not None:
                d.qpos[self.object_qadr:self.object_qadr+7] = hold_object
                d.qvel[self.object_dadr:self.object_dadr+6] = 0.
            mujoco.mj_step(m, d)
        if hold_object is not None:
            d.qpos[self.object_qadr:self.object_qadr+7] = hold_object
            d.qvel[self.object_dadr:self.object_dadr+6] = 0.
        d.qpos[self.palm_qadr:self.palm_qadr+7] = nxt                 # exactly on the trajectory
        d.qvel[self.palm_dadr:self.palm_dadr+3] = lin
        d.qvel[self.palm_dadr+3:self.palm_dadr+6] = ang
        mujoco.mj_forward(m, d)

    def finger_forces(self):
        """Hand-object normal force per finger (thumb, index, middle, ring, pinky) and on the palm (N)."""
        out = dict(thumb=0., index=0., middle=0., ring=0., pinky=0., palm=0.)
        for c in self.contacts():
            if c['pair'] == 'hand_object':
                f = self.finger_of_body.get(int(self.m.geom_bodyid[c['hand_geom']]), 'palm')
                out[f] += float(c['force_local'][0])
        return out

    def weld_object(self, object_pose=None):
        """Hold the object at object_pose (default: where it is) with a weld to the world, until release_object()."""
        d = self.d
        p = self.d.qpos[self.object_qadr:self.object_qadr+7] if object_pose is None else object_pose
        self.m.eq_data[self.hold_eq, 3:6] = p[:3]
        self.m.eq_data[self.hold_eq, 6:10] = p[3:7]
        d.eq_active[self.hold_eq] = 1

    def release_object(self):
        self.d.eq_active[self.hold_eq] = 0

    def get_state(self):
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        s = np.empty(mujoco.mj_stateSize(self.m, spec))
        mujoco.mj_getState(self.m, self.d, s, spec)
        return s

    def set_state(self, s):
        mujoco.mj_setState(self.m, self.d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.m, self.d)

    def contacts(self):
        """Contacts of the hand with the object, of the object with the table and of the hand with the table:
        list of dict(pair, hand_geom, dist, pos, normal, force_local (6: normal, 2 tangential, torsional, ...))."""
        m, d = self.m, self.d
        out, f = [], np.zeros(6)
        hand = set(self.hand_geoms.tolist())
        objg = set(self.object_geoms.tolist())
        for i in range(d.ncon):
            c = d.contact[i]
            g = {int(c.geom1), int(c.geom2)}
            if g & objg and g & hand:
                pair = 'hand_object'
            elif g & objg and self.table_geom in g:
                pair = 'object_table'
            elif self.table_geom in g and g & hand:
                pair = 'hand_table'
            else:
                continue
            mujoco.mj_contactForce(m, d, i, f)
            out.append(dict(pair=pair, hand_geom=next(iter(g & hand), -1), geom1=int(c.geom1), geom2=int(c.geom2),
                            dist=float(c.dist), pos=c.pos.copy(), frame=c.frame.copy(), force_local=f.copy()))
        return out


# Hand base frame (measured, fingers open -> closed): fingers point along FINGER_DIR, flex toward PALM_NORMAL
# (the palm side), the thumb sits on +y.
FINGER_DIR = np.array([.312, .001, .95])
PALM_NORMAL = np.array([.95, -.005, -.312])
CLOSE_TARGETS = np.array([1.2, .8, .8, 1.3, 1.5, 1.3, 1.5, 1.3, 1.5, 1.3, 1.5])


def quat_from_matrix(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(-1))
    return q


def side_grasp_pose(axis_point, approach, grasp_in_hand):
    """Hand base pose (7) for a power grasp of an upright cylinder: the thumb axis (+y) points up, the palm
    normal points along `approach` (horizontal unit vector, hand -> cylinder axis), and the cylinder axis passes
    through `grasp_in_hand` (3, base frame)."""
    u = np.array([approach[0], approach[1], 0.])
    u /= np.linalg.norm(u)
    y = np.array([0., 0., 1.])
    pn = PALM_NORMAL/np.linalg.norm(PALM_NORMAL)
    # base rotation R with R @ y_b = up and R @ palm_normal = u (palm normal has no y component)
    x_w = u*pn[0]-np.cross(u, y)*0.                                      # placeholder, solved below
    # in the base frame the palm normal lies in the x-z plane at angle a from +x; rotate so it maps to u
    a = np.arctan2(pn[2], pn[0])
    c, s = np.cos(-a), np.sin(-a)
    # R = [x_w, y_w, z_w]: x_w, z_w span the horizontal plane; x_b rotated by -a about y_b maps onto u
    z_dir = np.cross(u, y)                                               # horizontal, perpendicular to u
    x_w = c*u-s*(-z_dir)
    z_w = np.cross(x_w, y)
    R = np.stack((x_w, y, z_w), 1)
    pos = np.asarray(axis_point, dtype=np.float64)-R @ np.asarray(grasp_in_hand, dtype=np.float64)
    return np.concatenate((pos, quat_from_matrix(R)))
