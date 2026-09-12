"""Explicit compatibility with the installed IsaacLab mimic flag semantics."""
import ast
import inspect
import textwrap


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
