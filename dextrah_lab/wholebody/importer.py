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
    if not math.isfinite(frequency) or frequency<=0 or not math.isfinite(damping_ratio) or damping_ratio<=0:
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
                calibrated_hardware_compliance=False)
