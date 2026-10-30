"""mjlab scene for the G1 reposing task: robot, per-env object, table, ground."""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .objects import PLACEHOLDER, slot_names, slot_types
from .robot import ContactParams, WORLD_CONAFFINITY, WORLD_CONTYPE, build_fixed_robot_spec, build_robot_spec

# play2perfect/assets/urdf/table_narrow.urdf: one box, kinematic in Isaac.
TABLE_SIZE = (0.475, 0.4, 0.3)
_GEOM = {'box': mujoco.mjtGeom.mjGEOM_BOX, 'capsule': mujoco.mjtGeom.mjGEOM_CAPSULE,
         'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER}


@dataclass
class SceneParams:
    contact: ContactParams = field(default_factory=ContactParams)
    cylinder_slot: bool = False          # opt-in fifth object slot: a true cylinder (objects.CYLINDER_SLOT)


def _world_geom(geom, contact):
    geom.contype, geom.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
    geom.condim = contact.condim
    geom.solref = contact.solref
    geom.solimp = contact.solimp
    geom.margin = geom.gap = 0.
    geom.friction = (contact.friction, contact.torsional, contact.rolling)


def object_spec(contact, cylinder_slot=False):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='object_root')
    body.add_freejoint(name='object_free')
    # Template inertials; the real per-env values are written per world.
    body.mass = 0.1
    body.ipos = (0., 0., 0.)
    body.inertia = (1e-4, 1e-4, 1e-4)
    body.explicitinertial = True
    for name, kind in zip(slot_names(cylinder_slot), slot_types(cylinder_slot)):
        geom = body.add_geom(name=name, type=_GEOM[kind])
        geom.size = (PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
        geom.mass = 0.
        _world_geom(geom, contact)
    return spec


def table_spec(contact):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='box')
    geom = body.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(s/2 for s in TABLE_SIZE))
    body.mass = 500.
    body.inertia = (1000., 1000., 1000.)
    body.explicitinertial = True
    _world_geom(geom, contact)
    return spec


def add_ground(contact):
    def edit(spec):
        geom = spec.worldbody.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0., 0., 1.))
        _world_geom(geom, contact)
        if contact.robot_ground_only:
            geom.conaffinity = WORLD_CONAFFINITY | 4
    return edit


def build_scene_cfg(num_envs, params=SceneParams(), fixed_body_reference=None, robot_pose=None):
    """fixed_body_reference: Isaac dump of the fixed-body task -> welded right-arm
    robot at robot_pose = (pos, wxyz rot) from the task config; else the floating
    whole-body robot (pose written at reset)."""
    from mjlab.entity import EntityCfg
    from mjlab.scene import SceneCfg
    contact = params.contact
    if fixed_body_reference is None:
        robot = EntityCfg(spec_fn=lambda: build_robot_spec(contact),
                          init_state=EntityCfg.InitialStateCfg(pos=(0., 0.42, 0.75)))
    else:
        pos, rot = robot_pose
        robot = EntityCfg(spec_fn=lambda: build_fixed_robot_spec(fixed_body_reference, contact),
                          init_state=EntityCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot)))
    return SceneCfg(num_envs=num_envs, env_spacing=0., spec_fn=add_ground(contact), entities={
        'robot': robot,
        'object': EntityCfg(spec_fn=lambda: object_spec(contact, params.cylinder_slot)),
        'table': EntityCfg(spec_fn=lambda: table_spec(contact), init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 0.65))),
    })


def object_indices(mj_model):
    body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'object/object_root')
    cylinder = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'object/cylinder') >= 0
    geoms = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f'object/{s}') for s in slot_names(cylinder)]
    if body < 0 or min(geoms) < 0:
        raise RuntimeError('Object body/slots not found in compiled scene')
    return body, np.array(geoms)
