"""Explicit compatibility with the installed IsaacLab mimic flag semantics."""
import ast
import inspect
import textwrap
import math


def mimic_flag_from_source(source):
    """Return the cfg value which makes the native importer parse_mimic=True.

    The workstation revision passes the misleadingly named
    convert_mimic_joints_to_normal_joints STRAIGHT THROUGH to set_parse_mimic.
    Other revisions invert it. Inspect that single expression and fail closed
    for unrecognized implementations; do not edit the shared IsaacLab checkout.
    """
    tree=ast.parse(textwrap.dedent(source))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=='set_parse_mimic']
    if len(calls)!=1 or len(calls[0].args)!=1:
        raise ValueError('Unrecognized IsaacLab mimic import configuration')
    arg=calls[0].args[0]
    invert=isinstance(arg,ast.UnaryOp) and isinstance(arg.op,ast.Not)
    value=arg.operand if invert else arg
    if not (isinstance(value,ast.Attribute) and value.attr=='convert_mimic_joints_to_normal_joints'
            and isinstance(value.value,ast.Attribute) and value.value.attr=='cfg'
            and isinstance(value.value.value,ast.Name) and value.value.value.id=='self'):
        raise ValueError('Unrecognized IsaacLab mimic flag expression')
    return not invert


def native_mimic_cfg_flag():
    from isaaclab.sim.converters import UrdfConverter
    return mimic_flag_from_source(inspect.getsource(UrdfConverter._get_urdf_import_config))


def configure_native_mimics(stage, relations, num_envs, frequency=100., damping_ratio=1.):
    """Author explicit, well-damped mechanical couplings BEFORE physics starts.

    The importer currently authors 25 Hz, damping ratio 0.005: an extremely
    underdamped compliant linkage, not the rigid relation in the URDF. These
    100 Hz/critical-damping defaults are a provisional numerical approximation
    of that mechanical relation, NOT fitted Revo2 compliance measurements.
    Native PhysX coupling propagates reaction impulses in both directions.
    """
    from pxr import UsdPhysics
    # Zero frequency requests the native non-compliant constraint. Keep this
    # available as an explicit diagnostic, not a silent tuning change.
    if not math.isfinite(frequency) or frequency<0 or not math.isfinite(damping_ratio) or damping_ratio<0:
        raise ValueError('Invalid coupling parameters')
    count=0
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not path.startswith('/World/envs/env_') or '/Robot/joints/' not in path:
            continue
        name=prim.GetName()
        if name not in relations:
            continue
        relation=relations[name]
        schemas=[s for s in prim.GetAppliedSchemas() if s.startswith('PhysxMimicJointAPI:')]
        if len(schemas)!=1:
            raise ValueError(f'Missing/ambiguous native coupling: {path}')
        instance=schemas[0].split(':',1)[1]
        prefix=f'physxMimicJoint:{instance}:'
        source=prim.GetParent().GetChild(relation['joint'])
        if not source.IsValid():
            raise ValueError(f'Missing coupling source: {path}')
        ref=prim.GetRelationship(prefix+'referenceJoint').GetTargets()
        if list(map(str,ref)) != [str(source.GetPath())]:
            raise ValueError(f'Wrong native coupling target: {path}')
        if not math.isclose(prim.GetAttribute(prefix+'gearing').Get(),-float(relation['multiplier']),rel_tol=1e-6):
            raise ValueError(f'Wrong native coupling gearing: {path}')
        # PhysX equation: q_target + gearing*q_source + offset_degrees = 0.
        if not math.isclose(prim.GetAttribute(prefix+'offset').Get(),
                            -math.degrees(float(relation.get('offset',0.))),abs_tol=1e-6):
            raise ValueError(f'Wrong native coupling offset: {path}')
        prim.GetAttribute(prefix+'referenceJointAxis').Set('rot'+UsdPhysics.RevoluteJoint(source).GetAxisAttr().Get())
        prim.GetAttribute(prefix+'naturalFrequency').Set(float(frequency))
        prim.GetAttribute(prefix+'dampingRatio').Set(float(damping_ratio))
        count+=1
    if count!=num_envs*len(relations):
        raise ValueError(f'Configured {count} couplings, expected {num_envs*len(relations)}')
    return dict(count=count,natural_frequency_hz=frequency,damping_ratio=damping_ratio,
                compliant=frequency>0,
                calibrated_hardware_compliance=False)


def filter_thumb_housing_pairs(stage, num_envs):
    """DIAGNOSTIC narrow exclusion for the measured palm/proximal-thumb pair.

    Do not mistake this for globally disabling hand self-contact. The merged
    wrist body contains the palm housing. Distal/palm, finger/finger, all robot
    body contacts, and all hand/object/table contacts remain enabled. Whether
    this pair should be permanently excluded requires the separate CAD audit.
    """
    from pxr import Usd, UsdPhysics
    pairs=[]
    for env_id in range(num_envs):
        root=stage.GetPrimAtPath(f'/World/envs/env_{env_id}/Robot')
        bodies={p.GetName().lower():p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        for side in ('left','right'):
            palm=bodies[f'{side}_wrist_yaw_link']
            thumb=bodies[f'{side}_thumb_proximal_link']
            relation=UsdPhysics.FilteredPairsAPI.Apply(palm).CreateFilteredPairsRel()
            relation.AddTarget(thumb.GetPath())
            if thumb.GetPath() not in relation.GetTargets():
                raise ValueError('Failed to author the specific housing-pair diagnostic')
            pairs.append([str(palm.GetPath()),str(thumb.GetPath())])
    return pairs


def decompose_palm_colliders(stage, num_envs):
    """Preserve the palm's concave thumb clearance instead of filling its hull.

    CAD audit 456/457 found no triangle-mesh intersections at the six measured
    poses, but palm convexification alone introduced intersections in all six.
    Change ONLY two palm collision approximations per robot: no pair filters,
    no visual/inertia/joint changes and no fingertip collider changes.
    """
    from pxr import UsdPhysics, PhysxSchema
    changed=[]
    for env_id in range(num_envs):
        for side in ('left','right'):
            path=f'/World/envs/env_{env_id}/Robot/{side}_wrist_yaw_link/collisions/{side}_base_link/mesh'
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f'Palm collider not found at audited path: {path}')
            if prim.IsInstanceProxy():
                ancestor=prim.GetParent()
                while ancestor.IsValid() and not ancestor.IsInstance():
                    ancestor=ancestor.GetParent()
                collision_root=f'/World/envs/env_{env_id}/Robot/{side}_wrist_yaw_link/collisions'
                if not ancestor.IsValid() or not (str(ancestor.GetPath())==collision_root or
                                                   str(ancestor.GetPath()).startswith(collision_root+'/')):
                    raise ValueError(f'Unexpected palm collision instance ancestor: {ancestor.GetPath()}')
                ancestor.SetInstanceable(False)
                prim=stage.GetPrimAtPath(path)
            if prim.IsInstanceProxy():
                raise ValueError('Palm collision approximation is not editable')
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set('convexDecomposition')
            decomposition=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            decomposition.CreateMaxConvexHullsAttr().Set(128)
            decomposition.CreateHullVertexLimitAttr().Set(64)
            decomposition.CreateVoxelResolutionAttr().Set(2000000)
            decomposition.CreateErrorPercentageAttr().Set(.1)
            decomposition.CreateMinThicknessAttr().Set(.0001)
            changed.append(path)
    return dict(paths=changed,max_hulls=128,hull_vertex_limit=64,voxel_resolution=2000000,
        error_percentage=.1,min_thickness_m=.0001,filtered_pairs=[],physics_validated=False)
