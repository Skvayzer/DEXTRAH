#!/usr/bin/env python3
"""Generate functional FMB proxy pegs and matching receptacles for ADEPT.

ADEPT does not publish its simulation CAD or dimensions. The public FMB
repository/spreadsheet names the physical families but provides no
simulation-ready meshes, so the centralized dimensions below are inferred.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


PEG_HEIGHT_M = 0.15
PEG_RADIUS_M = 0.025
BOARD_HALF_EXTENT_M = 0.12
BOARD_HEIGHT_M = 0.03
HOLE_CLEARANCE_M = 0.0015


def star_polygon(radius: float, inner_scale: float = 0.5):
    return [
        (
            (radius if index % 2 == 0 else radius * inner_scale)
            * math.cos(math.pi / 2 + index * math.pi / 5),
            (radius if index % 2 == 0 else radius * inner_scale)
            * math.sin(math.pi / 2 + index * math.pi / 5),
        )
        for index in range(10)
    ]


def rounded_square_polygon(radius: float, vertices: int = 16):
    result = []
    for index in range(vertices):
        angle = index * 2 * math.pi / vertices
        cosine, sine = math.cos(angle), math.sin(angle)
        result.append(
            (
                radius * math.copysign(abs(cosine) ** 0.5, cosine),
                radius * math.copysign(abs(sine) ** 0.5, sine),
            )
        )
    return result


def _material(color):
    return f'''    def Scope "Looks"
    {{
        def Material "material"
        {{
            token outputs:surface.connect = </object/baseLink/Looks/material/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({color[0]}, {color[1]}, {color[2]})
                float inputs:roughness = 0.6
                token outputs:surface
            }}
        }}
    }}'''


def _mesh(name, vertices, faces):
    point_text = ", ".join(f"({x:.8f}, {y:.8f}, {z:.8f})" for x, y, z in vertices)
    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(index) for face in faces for index in face)
    return f'''    def Mesh "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxMeshCollisionAPI"])
    {{
        point3f[] points = [{point_text}]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        uniform token subdivisionScheme = "none"
        uniform token physics:approximation = "convexDecomposition"
        prepend rel material:binding = </object/baseLink/Looks/material>
    }}'''


def _prism(polygon, height):
    count = len(polygon)
    vertices = [(x, y, -height / 2) for x, y in polygon]
    vertices += [(x, y, height / 2) for x, y in polygon]
    faces = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    faces += [
        (index, (index + 1) % count, (index + 1) % count + count, index + count)
        for index in range(count)
    ]
    return vertices, faces


def _ring(inner, outer_half_extent, height):
    count = len(inner)
    outer = []
    for index in range(count):
        angle = math.pi / 2 + index * 2 * math.pi / count
        scale = outer_half_extent / max(abs(math.cos(angle)), abs(math.sin(angle)))
        outer.append((scale * math.cos(angle), scale * math.sin(angle)))
    vertices = [(x, y, -height / 2) for x, y in outer]
    vertices += [(x, y, height / 2) for x, y in outer]
    vertices += [(x, y, -height / 2) for x, y in inner]
    vertices += [(x, y, height / 2) for x, y in inner]
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            (
                (index + count, following + count, following + 3 * count, index + 3 * count),
                (following, index, index + 2 * count, following + 2 * count),
                (index, following, following + count, index + count),
                (following + 2 * count, index + 2 * count, index + 3 * count, following + 3 * count),
            )
        )
    return vertices, faces


def _write_asset(path: Path, mesh: str, color, mass: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "object"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "object"
{{
def Xform "baseLink" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxContactReportAPI"])
{{
    bool physxArticulation:articulationEnabled = false
    float physics:mass = {mass}
{_material(color)}
{mesh}
}}
}}
'''
    )


def generate(output_root: Path):
    variants = {
        "star": star_polygon(PEG_RADIUS_M),
        "square_round": rounded_square_polygon(PEG_RADIUS_M),
    }
    for name, polygon in variants.items():
        peg_vertices, peg_faces = _prism(polygon, PEG_HEIGHT_M)
        peg_path = output_root / f"adept_fmb_{name}" / "USD" / f"fmb_{name}_peg" / f"fmb_{name}_peg.usd"
        _write_asset(peg_path, _mesh("geometry", peg_vertices, peg_faces), (0.95, 0.95, 0.95), 0.08)

        hole_polygon = [
            (
                x * (PEG_RADIUS_M + HOLE_CLEARANCE_M) / PEG_RADIUS_M,
                y * (PEG_RADIUS_M + HOLE_CLEARANCE_M) / PEG_RADIUS_M,
            )
            for x, y in polygon
        ]
        board_vertices, board_faces = _ring(hole_polygon, BOARD_HALF_EXTENT_M, BOARD_HEIGHT_M)
        board_path = output_root / "adept_fmb_receptacles" / f"fmb_{name}_board.usd"
        _write_asset(board_path, _mesh("geometry", board_vertices, board_faces), (0.08, 0.08, 0.08), 10.0)
        print(peg_path)
        print(board_path)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=repo_root / "dextrah_lab/assets")
    args = parser.parse_args()
    generate(args.output_root)


if __name__ == "__main__":
    main()
