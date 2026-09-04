#!/usr/bin/env python3
"""Generate the 16 ADEPT Appendix Fig. 8 primitive USD assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from dextrah_lab.tasks.dextrah_kuka_allegro.adept_mdp import ADEPT_PRIMITIVES


COLORS = (
    (0.10, 0.42, 0.78),
    (0.02, 0.55, 0.42),
    (0.23, 0.35, 0.70),
    (0.05, 0.55, 0.20),
    (0.48, 0.16, 0.63),
    (0.70, 0.12, 0.38),
    (0.85, 0.45, 0.03),
    (0.92, 0.72, 0.02),
)


def _define_geometry(stage: Usd.Stage, shape: str, dimensions: tuple[float, ...]):
    path = "/baseLink/geometry"
    if shape == "cuboid":
        geometry = UsdGeom.Cube.Define(stage, path)
        geometry.GetSizeAttr().Set(1.0)
        geometry.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    elif shape == "sphere":
        geometry = UsdGeom.Sphere.Define(stage, path)
        geometry.GetRadiusAttr().Set(dimensions[0])
    elif shape == "capsule":
        geometry = UsdGeom.Capsule.Define(stage, path)
        geometry.GetAxisAttr().Set("Z")
        geometry.GetRadiusAttr().Set(dimensions[0])
        geometry.GetHeightAttr().Set(dimensions[1])
    elif shape == "cone":
        geometry = UsdGeom.Cone.Define(stage, path)
        geometry.GetAxisAttr().Set("Z")
        geometry.GetRadiusAttr().Set(dimensions[0])
        geometry.GetHeightAttr().Set(dimensions[1])
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    return geometry


def generate_asset(output_path: Path, shape: str, dimensions, color) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)

    root = UsdGeom.Xform.Define(stage, "/baseLink").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)

    geometry = _define_geometry(stage, shape, dimensions)
    geometry.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())

    stage.GetRootLayer().Save()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "dextrah_lab/assets/adept_primitives/USD",
    )
    args = parser.parse_args()

    for index, spec in enumerate(ADEPT_PRIMITIVES):
        output = args.output_dir / spec.name / f"{spec.name}.usd"
        generate_asset(
            output,
            spec.shape,
            spec.dimensions,
            COLORS[index % len(COLORS)],
        )
        print(output)


if __name__ == "__main__":
    main()
