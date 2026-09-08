#!/usr/bin/env python3
"""Inspect GRAIL-style BPS-128 on a real object mesh, with no simulator or RL."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw
import trimesh
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dextrah_lab.object_shape.bps import (  # noqa: E402
    GRAIL_COMMIT, encode_mesh, load_object_mesh, validate_representation,
)

AMBER = (249, 183, 55)
CYAN = (27, 211, 179)
MAGENTA = (242, 78, 151)
LINK = (118, 151, 187)


def descriptor_plot(distances, selected):
    """A compact index-to-distance plot, with the selected coordinate highlighted."""
    image = Image.new("RGB", (768, 236), (24, 31, 43))
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 48, 746, 36, 198
    ymax = max(1., float(np.ceil(distances.max() * 5) / 5))
    draw.text((left, 9), "128 scalar distances (normalized)", fill=(230, 238, 246), font_size=17)
    for y in np.linspace(0, ymax, 5):
        row = bottom - y / ymax * (bottom-top)
        draw.line((left, row, right, row), fill=(53, 63, 77))
        draw.text((3, row - 7), f"{y:.2f}", fill=(184, 198, 210), font_size=12)
    width = (right-left)/len(distances)
    for i, distance in enumerate(distances):
        x = left + i*width
        row = bottom - float(distance)/ymax*(bottom-top)
        draw.rectangle((x, row, x + max(1., width-1), bottom), fill=MAGENTA if i == selected else LINK)
    for i in (0, 32, 64, 96, 127):
        x = left + i * width
        draw.text((x-4, bottom+8), str(i), fill=(184, 198, 210), font_size=12)
    return np.asarray(image)


def sphere_circles():
    angle = np.linspace(0, 2*np.pi, 129)
    circle = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    curves = [circle, circle[:, [0, 2, 1]], circle[:, [2, 0, 1]]]
    return np.concatenate([np.stack([c[:-1], c[1:]], axis=1) for c in curves]).astype(np.float32)


def run_viewer(mesh, bps, report, port, object_name):
    server = viser.ViserServer(host="127.0.0.1", port=port, label="BPS-128 | Object shape")
    server.gui.configure_theme(titlebar_content=None, dark_mode=True, show_share_button=False)
    server.gui.main_panel.dock_right()
    server.scene.set_up_direction("+z")
    normalized_mesh = mesh.copy()
    normalized_mesh.vertices = (mesh.vertices - bps.centroid_m) / bps.scale_m
    mesh_handle = server.scene.add_mesh_simple(
        "/object/mesh", vertices=normalized_mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.uint32), color=(150, 168, 187), side="double", flat_shading=True)
    edges = normalized_mesh.vertices[normalized_mesh.edges_unique].astype(np.float32)
    edge_handle = server.scene.add_line_segments("/object/edges", points=edges,
        colors=(75, 96, 116), thickness=.7, thickness_units="screen", visible=False)
    basis_handle = server.scene.add_point_cloud("/basis/queries", points=bps.basis,
        colors=AMBER, point_size=.014, point_shape="circle", precision="float32")
    nearest_handle = server.scene.add_point_cloud("/surface/nearest", points=bps.nearest_normalized,
        colors=CYAN, point_size=.011, point_shape="circle", precision="float32")
    cloud_handle = server.scene.add_point_cloud("/surface/samples", points=bps.surface_normalized,
        colors=(147, 206, 205), point_size=.005, point_shape="circle", precision="float32", visible=False)
    segments = np.stack([bps.basis, bps.nearest_normalized], axis=1)
    lines_handle = server.scene.add_line_segments("/basis/distance_lines", points=segments,
        colors=LINK, thickness=1., thickness_units="screen")
    sphere_handle = server.scene.add_line_segments("/basis/unit_sphere", points=sphere_circles(),
        colors=(82, 100, 121), thickness=.75, thickness_units="screen")
    frame_handle = server.scene.add_frame("/normalized_frame", axes_length=.20, axes_radius=.003,
        visible=False)
    selected_query = server.scene.add_icosphere("/selected/query", radius=.023, color=MAGENTA)
    selected_nearest = server.scene.add_icosphere("/selected/nearest", radius=.016, color=MAGENTA)
    selected_line = server.scene.add_line_segments("/selected/line", points=segments[:1],
        colors=MAGENTA, thickness=4., thickness_units="screen")

    with server.gui.add_folder("Object shape | BPS-128"):
        server.gui.add_markdown(
            f"**{object_name}** — actual procedural SAPG object.\n\n"
            "**Amber:** 128 fixed query points. **Mint:** nearest sampled surface points. "
            "**Pink:** selected distance.\n\n"
            "These are **shape measurements, not collision spheres**. "
            "No policy, fabrics or PCA is running.")
        selected = server.gui.add_slider("Selected BPS index", min=0, max=127, step=1,
                                        initial_value=16)
        selected_status = server.gui.add_markdown("")
        plot = server.gui.add_image(descriptor_plot(bps.distances, selected.value), label="BPS descriptor")
        only_selected = server.gui.add_checkbox("Only selected connection", False)
        show_lines = server.gui.add_checkbox("All distance lines", True)
        show_queries = server.gui.add_checkbox("128 query points", True)
        show_nearest = server.gui.add_checkbox("Nearest surface points", True)
        show_samples = server.gui.add_checkbox("16,384 surface samples", False)
        show_mesh = server.gui.add_checkbox("Object mesh", True)
        show_edges = server.gui.add_checkbox("Mesh triangle edges", False)
        show_sphere = server.gui.add_checkbox("Sphere guide", True)
        show_axes = server.gui.add_checkbox("Normalized-frame axes", False)
        home = server.gui.add_button("Reset camera")
    with server.gui.add_folder("Encoding / source", expand_by_default=False):
        extents = " × ".join(f"{x*1000:.1f}" for x in mesh.extents)
        server.gui.add_markdown(
            "`d[i] = min_j || b[i] - x[j] ||`\n\n"
            "`x = (surface_point - sample_mean) / radius`\n\n"
            "The scene uses normalized object coordinates. No PCA rotation is applied. "
            "The sphere is fixed in the canonical object frame, not the world frame.\n\n"
            f"Mesh size: **{extents} mm**. Normalization radius: **{bps.scale_m*1000:.2f} mm**.\n\n"
            "One shared Fibonacci basis; 128 unsigned scalar distances. "
            "16,384 area-weighted surface samples, seed 42. "
            "This adapts GRAIL's default 10 queries and bypasses its small-mesh vertex shortcut.\n\n"
            f"Maximum distance approximation error on these queries: "
            f"**{report['max_sampling_distance_error_m']*1000:.3f} mm**, "
            "checked against continuous triangles.\n\n"
            "[Pinned GRAIL source](https://github.com/NVlabs/GRAIL/blob/"
            f"{GRAIL_COMMIT}/grail/retargeting/compute_bps.py). "
            "This is a shape-input preview; the SAPG observation has not been changed.")

    def camera(client):
        client.camera.position = (1.5, -2.6, 2.0)
        client.camera.look_at = (0., 0., 0.)
        client.camera.up_direction = (0., 0., 1.)
        client.camera.fov = .72

    server.on_client_connect(camera)

    @home.on_click
    def _home(event):
        if event.client is not None:
            camera(event.client)

    lock = threading.RLock()

    def update(_=None):
        with lock, server.atomic():
            index = int(selected.value)
            d = float(bps.distances[index])
            selected_query.position = bps.basis[index]
            selected_nearest.position = bps.nearest_normalized[index]
            selected_line.points = segments[index:index+1]
            selected_status.content = (
                f"**BPS[{index}] = {d:.4f}** (normalized)  \n"
                f"Physical distance: **{d*bps.scale_m*1000:.2f} mm**. "
                "The pink bar is this same number.")
            plot.image = descriptor_plot(bps.distances, index)
            lines_handle.visible = show_lines.value and not only_selected.value
            basis_handle.visible = show_queries.value and not only_selected.value
            nearest_handle.visible = show_nearest.value and not only_selected.value
            cloud_handle.visible = show_samples.value
            mesh_handle.visible = show_mesh.value
            edge_handle.visible = show_edges.value
            sphere_handle.visible = show_sphere.value
            frame_handle.visible = show_axes.value

    for control in (selected, only_selected, show_lines, show_queries, show_nearest,
                    show_samples, show_mesh, show_edges, show_sphere, show_axes):
        control.on_update(update)
    update()
    print(f"BPS_VISER_READY http://localhost:{server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="Canonical object mesh/GLB, in meters")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--name", default="Procedural hammer")
    parser.add_argument("--output", type=Path, required=True, help="NEW output directory")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    mesh = load_object_mesh(args.mesh)
    bps = encode_mesh(mesh)
    report = validate_representation(mesh, bps)
    report.update(mesh_path=str(args.mesh.resolve()),
                  mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                  grail_commit=GRAIL_COMMIT, seed=42,
                  surface_sampler="trimesh area-weighted triangle sampling (always)",
                  canonical_axes="input object frame; no principal-axis rotation",
                  descriptor="128 unsigned distances to normalized sampled surface",
                  viewer_units="normalized; millimeter distances displayed in GUI",
                  training_started=False, policy_observation_modified=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "bps_128.npz", basis_normalized=bps.basis,
        surface_m=bps.surface_m, surface_normalized=bps.surface_normalized,
        centroid_m=bps.centroid_m, scale_m=bps.scale_m, distances=bps.distances,
        nearest_indices=bps.nearest_indices, basis_m=bps.basis_m,
        nearest_surface_m=bps.nearest_m)
    np.savetxt(args.output / "descriptor.csv", np.c_[np.arange(128), bps.distances],
               delimiter=",", header="index,normalized_distance", comments="", fmt=["%d", "%.8f"])
    print("BPS_VALIDATED " + json.dumps(report), flush=True)
    if not args.validate_only:
        run_viewer(mesh, bps, report, args.port, args.name)


if __name__ == "__main__":
    main()
