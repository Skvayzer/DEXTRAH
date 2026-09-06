"""G1 + BrainCo Revo2 components for ADEPT/SAPG training."""

from .collision_geometry import (
    G1_REVO2_CANONICAL_JOINT_NAMES,
    G1_REVO2_DYNAMIC_SPHERES,
    G1_REVO2_FIXED_SPHERES,
    G1_REVO2_MIMIC_RULES,
    G1_REVO2_SELF_COLLISION_PAIRS,
    build_g1_collision_batch,
    point_jacobian,
    reduce_jacobian_to_canonical,
)
from .reduced_fabric import (
    CollisionBatch,
    ReducedAdeptFabric,
    ReducedAdeptFabricConfig,
    ReducedAdeptFabricState,
)

__all__ = [
    "G1_REVO2_CANONICAL_JOINT_NAMES",
    "G1_REVO2_DYNAMIC_SPHERES",
    "G1_REVO2_FIXED_SPHERES",
    "G1_REVO2_MIMIC_RULES",
    "G1_REVO2_SELF_COLLISION_PAIRS",
    "build_g1_collision_batch",
    "point_jacobian",
    "reduce_jacobian_to_canonical",
    "CollisionBatch",
    "ReducedAdeptFabric",
    "ReducedAdeptFabricConfig",
    "ReducedAdeptFabricState",
]
