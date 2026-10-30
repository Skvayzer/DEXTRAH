"""Everything the structured model knows, extracted from a compiled stage-3 testbed (never retyped).

HandSpec: the hand subtree below the prescribed palm (bodies in MuJoCo's topological order, one hinge or none per
body), inertias, joint axes / ranges / armature, the 11 velocity servos, collision geoms (MuJoCo's own convex hulls
for meshes, primitives for the 'capsule' variant). ObjectSpec: one free rigid body and its primitive geoms.
SceneSpec: table, ground, options and contact parameters.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from ..testbed import HAND_JOINTS, HAND_ROOT, PREFIX, SUBSTEPS, TABLE_SIZE

GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_PLANE): 'plane', int(mujoco.mjtGeom.mjGEOM_SPHERE): 'sphere',
              int(mujoco.mjtGeom.mjGEOM_CAPSULE): 'capsule', int(mujoco.mjtGeom.mjGEOM_CYLINDER): 'cylinder',
              int(mujoco.mjtGeom.mjGEOM_BOX): 'box', int(mujoco.mjtGeom.mjGEOM_MESH): 'mesh'}


@dataclass
class GeomSpec:
    name: str
    kind: str                    # 'sphere' | 'capsule' | 'cylinder' | 'box' | 'mesh' | 'plane'
    body: int                    # index into the owner's body list (hand) or 0 (object / world)
    size: np.ndarray             # MuJoCo geom_size (3,)
    pos: np.ndarray              # in the body frame (3,)
    quat: np.ndarray             # in the body frame (4,) wxyz
    friction: np.ndarray         # (3,) sliding, torsional, rolling
    mujoco_id: int
    hull_vert: np.ndarray = None     # mesh: hull vertices in the GEOM frame (n, 3)
    hull_face: np.ndarray = None     # mesh: hull triangles (k, 3), outward


@dataclass
class HandSpec:
    body_names: list
    parent: np.ndarray           # (nb,) parent index in this list, -1 = the palm frame
    body_pos: np.ndarray         # (nb, 3) in the parent frame (MuJoCo body_pos)
    body_quat: np.ndarray        # (nb, 4)
    ipos: np.ndarray             # (nb, 3) COM in the body frame
    iquat: np.ndarray            # (nb, 4) principal axes in the body frame
    mass: np.ndarray             # (nb,)
    inertia: np.ndarray          # (nb, 3) principal moments
    joint_body: np.ndarray       # (11,) body index of each HAND_JOINTS joint
    body_joint: np.ndarray       # (nb,) joint index (HAND_JOINTS order) of the body's hinge, -1 if none
    jnt_pos: np.ndarray          # (11, 3) anchor in the body frame
    jnt_axis: np.ndarray         # (11, 3) in the body frame
    qpos0: np.ndarray            # (11,)
    jnt_range: np.ndarray        # (11, 2)
    armature: np.ndarray         # (11,)
    servo_gain: np.ndarray       # (11,) kp / kv
    servo_vmax: np.ndarray       # (11,)
    servo_kv: np.ndarray         # (11,)
    servo_effort: np.ndarray     # (11,) effort = kv * vmax (blocked-joint torque cap)
    jnt_solref: np.ndarray       # (11, 2) joint-limit softness
    jnt_solimp: np.ndarray       # (11, 5)
    dof_invweight0: np.ndarray   # (11,) MuJoCo's diagonal approximation for joint-limit rows
    body_invweight0: np.ndarray  # (nb, 2) translational / rotational, for contact rows
    geoms: list = field(default_factory=list)
    finger_of_body: list = field(default_factory=list)    # 'thumb' | 'index' | ... | 'palm'

    @property
    def nbody(self):
        return len(self.body_names)


@dataclass
class ObjectSpec:
    mass: float
    inertia: np.ndarray          # (3,) principal moments, body frame = principal frame at the COM
    geoms: list
    invweight0: np.ndarray = None    # (2,) translational / rotational


@dataclass
class SceneSpec:
    hand: HandSpec
    obj: ObjectSpec
    table_top: float
    table_pos: np.ndarray        # (3,) box centre
    table_half: np.ndarray       # (3,)
    timestep: float
    substeps: int
    gravity: np.ndarray
    impratio: float
    solref: np.ndarray           # world/object/robot contact solref (all equal in the testbed)
    solimp: np.ndarray
    refsafe: bool
    palm_body_pos: np.ndarray    # hand root pose in the palm frame (identity in the testbed)
    palm_body_quat: np.ndarray
    table_friction: np.ndarray = None    # (3,)
    condim: int = 4              # contact dimensionality (equal for every contact geom in the testbed)


def _hull(m, g):
    """MuJoCo's convex hull of mesh geom g, in the geom frame: (vertices (n, 3), outward triangles (k, 3))."""
    i = int(m.geom_dataid[g])
    v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i]+m.mesh_vertnum[i]].astype(np.float64)
    adr = int(m.mesh_graphadr[i])
    if adr < 0:
        raise ValueError(f'geom {g}: mesh {i} has no convex hull graph')
    graph = m.mesh_graph[adr:]
    numvert, numface = int(graph[0]), int(graph[1])
    vert_globalid = graph[2+numvert:2+2*numvert]
    faces = graph[2+3*numvert+3*numface:2+3*numvert+6*numface].reshape(numface, 3)
    # face_globalid indexes the mesh's vertices directly
    used = np.unique(vert_globalid)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts, tri = v[used], remap[faces]
    c = verts.mean(0)
    n = np.cross(verts[tri[:, 1]]-verts[tri[:, 0]], verts[tri[:, 2]]-verts[tri[:, 0]])
    flip = ((verts[tri[:, 0]]-c)*n).sum(1) < 0
    tri[flip] = tri[flip][:, ::-1]
    return verts, tri


def _geom(m, g, body_index):
    kind = GEOM_TYPES[int(m.geom_type[g])]
    gs = GeomSpec(name=m.geom(g).name, kind=kind, body=body_index, size=m.geom_size[g].copy(), pos=m.geom_pos[g].copy(),
                  quat=m.geom_quat[g].copy(), friction=m.geom_friction[g].copy(), mujoco_id=g)
    if kind == 'mesh':
        gs.hull_vert, gs.hull_face = _hull(m, g)
    return gs


def extract(tb):
    """SceneSpec of a compiled Testbed."""
    m = tb.m
    root = m.body(PREFIX+HAND_ROOT).id
    palm = tb.palm_body
    bodies = [b for b in range(m.nbody) if b == root or tb._under(b, root)]
    index = {b: i for i, b in enumerate(bodies)}
    nb = len(bodies)
    parent = np.array([index.get(int(m.body_parentid[b]), -1) for b in bodies])
    if parent[0] != -1 or int(m.body_parentid[root]) != palm:
        raise ValueError('hand root must be a child of the palm body')
    joint_ids = [m.joint(PREFIX+j).id for j in HAND_JOINTS]
    body_joint = -np.ones(nb, dtype=np.int64)
    for k, j in enumerate(joint_ids):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f'{HAND_JOINTS[k]} is not a hinge')
        body_joint[index[int(m.jnt_bodyid[j])]] = k
    for b in bodies:
        if m.body_jntnum[b] > 1:
            raise ValueError(f'body {m.body(b).name} has {m.body_jntnum[b]} joints')
    dof = m.jnt_dofadr[joint_ids]
    act = [m.actuator(PREFIX+j).id for j in HAND_JOINTS]
    kv = np.array([-m.actuator_biasprm[a, 2] for a in act])          # velocity actuator: force = kv (ctrl - qd)
    if not np.allclose(kv, m.actuator_gainprm[act, 0]):
        raise ValueError('hand actuators are not velocity servos')
    # total actuator force per joint is clipped to the URDF effort (jnt_actfrcrange, set by MuJoCo's URDF import)
    lim = m.jnt_actfrcrange[joint_ids]
    if not (m.jnt_actfrclimited[joint_ids].all() and np.allclose(lim[:, 1], kv*tb.servo_vmax) and np.allclose(lim[:, 0], -lim[:, 1])):
        raise ValueError('hand joint actuator force range differs from the servo effort kv * vmax')
    finger = []
    for b in bodies:
        n = m.body(b).name
        finger.append(next((f for f in ('thumb', 'index', 'middle', 'ring', 'pinky') if n.startswith(PREFIX+'right_'+f)), 'palm'))
    hand = HandSpec(
        body_names=[m.body(b).name for b in bodies], parent=parent, body_pos=m.body_pos[bodies].copy(),
        body_quat=m.body_quat[bodies].copy(), ipos=m.body_ipos[bodies].copy(), iquat=m.body_iquat[bodies].copy(),
        mass=m.body_mass[bodies].copy(), inertia=m.body_inertia[bodies].copy(),
        joint_body=np.array([index[int(m.jnt_bodyid[j])] for j in joint_ids]), body_joint=body_joint,
        jnt_pos=m.jnt_pos[joint_ids].copy(), jnt_axis=m.jnt_axis[joint_ids].copy(),
        qpos0=m.qpos0[m.jnt_qposadr[joint_ids]].copy(), jnt_range=m.jnt_range[joint_ids].copy(),
        armature=m.dof_armature[dof].copy(), servo_gain=tb.servo_gain.copy(), servo_vmax=tb.servo_vmax.copy(),
        servo_kv=kv, servo_effort=kv*tb.servo_vmax, jnt_solref=m.jnt_solref[joint_ids].copy(),
        jnt_solimp=m.jnt_solimp[joint_ids].copy(), dof_invweight0=m.dof_invweight0[dof].copy(),
        body_invweight0=m.body_invweight0[bodies].copy(),
        geoms=[_geom(m, int(g), index[int(m.geom_bodyid[g])]) for g in tb.hand_geoms], finger_of_body=finger)
    ob = tb.object_body
    if np.abs(m.body_ipos[ob]).max() > 1e-12 or np.abs(m.body_iquat[ob]-(1, 0, 0, 0)).max() > 1e-12:
        raise ValueError('object body frame must be its principal frame at the COM')
    obj = ObjectSpec(mass=float(m.body_mass[ob]), inertia=m.body_inertia[ob].copy(),
                     geoms=[_geom(m, int(g), 0) for g in tb.object_geoms], invweight0=m.body_invweight0[ob].copy())
    tg = tb.table_geom
    contact_geoms = list(tb.hand_geoms)+list(tb.object_geoms)+[tg]
    solref, solimp = m.geom_solref[contact_geoms], m.geom_solimp[contact_geoms]
    if np.ptp(solref, 0).max() > 0 or np.ptp(solimp, 0).max() > 0:
        raise ValueError('contact solref / solimp differ between geoms (mixing rules not implemented)')
    condim = m.geom_condim[contact_geoms]
    if np.ptp(condim) > 0 or int(condim[0]) not in (1, 3, 4):
        raise ValueError(f'contact condim must be one of 1, 3, 4 and equal for every geom, got {sorted(set(condim))}')
    if m.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC and int(condim[0]) > 1:
        raise ValueError('only elliptic friction cones are implemented')
    return SceneSpec(
        hand=hand, obj=obj, table_top=float(tb.d.mocap_pos[tb.table_mocap][2]+TABLE_SIZE[2]/2),
        table_pos=tb.d.mocap_pos[tb.table_mocap].copy(), table_half=m.geom_size[tg].copy(), timestep=float(m.opt.timestep),
        substeps=SUBSTEPS, gravity=m.opt.gravity.copy(), impratio=float(m.opt.impratio), solref=solref[0].copy(),
        solimp=solimp[0].copy(), refsafe=not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE),
        palm_body_pos=m.body_pos[root].copy(), palm_body_quat=m.body_quat[root].copy(),
        table_friction=m.geom_friction[tg].copy(), condim=int(condim[0]))


def tensors(x, dtype=torch.float64, device='cpu'):
    """numpy fields of a spec dataclass -> dict of tensors (non-array fields kept)."""
    out = {}
    for k, v in vars(x).items():
        out[k] = torch.as_tensor(v, dtype=dtype if np.asarray(v).dtype.kind == 'f' else None, device=device) \
            if isinstance(v, np.ndarray) else v
    return out
