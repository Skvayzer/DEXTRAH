#!/usr/bin/env python3
"""Generate the 16 ADEPT Appendix Fig. 8 primitive USD assets."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _geometry_usda(shape: str, dimensions: tuple[float, ...]) -> str:
    material_binding = 'prepend rel material:binding = </object/baseLink/Looks/material>'
    if shape == "cuboid":
        return (
            'def Cube "geometry" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {\n'
            '        double size = 1\n'
            f'        float3 xformOp:scale = ({dimensions[0]}, {dimensions[1]}, {dimensions[2]})\n'
            '        uniform token[] xformOpOrder = ["xformOp:scale"]\n'
            f'        {material_binding}\n'
            '    }'
        )
    usd_type = {"sphere": "Sphere", "capsule": "Capsule", "cone": "Cone"}.get(shape)
    if usd_type is None:
        raise ValueError(f"Unsupported shape: {shape}")
    attributes = [f"double radius = {dimensions[0]}"]
    if shape in {"capsule", "cone"}:
        attributes.extend((f"double height = {dimensions[1]}", 'uniform token axis = "Z"'))
    body = "\n        ".join(attributes)
    return (
        f'def {usd_type} "geometry" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{\n'
        f"        {body}\n"
        f'        {material_binding}\n'
        '    }'
    )


def generate_asset(output_path: Path, shape: str, dimensions, color) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry = _geometry_usda(shape, dimensions)
    content = f'''#usda 1.0
(
    defaultPrim = "object"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "object"
{{
def Xform "baseLink" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
{{
    def Scope "Looks"
    {{
        def Material "material"
        {{
            token outputs:surface.connect = </object/baseLink/Looks/material/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({color[0]}, {color[1]}, {color[2]})
                float inputs:roughness = 0.5
                token outputs:surface
            }}
        }}
    }}
    {geometry}
}}
}}
'''
    output_path.write_text(content)


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
