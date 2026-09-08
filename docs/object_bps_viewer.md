# BPS-128 object-shape preview

This is a CPU-only Viser geometry viewer, not a policy rollout. It neither
starts training nor changes SAPG observations, rewards, physics, actions,
fabrics, or the hand PCA prior.

## Source and encoding

The unmodified GRAIL implementation is pinned at
`aa31d8242ac79b11545b9e3635f73014a227bdfc` in
`dextrah_lab/object_shape/grail`, with its original NVIDIA research/evaluation
license. The wrapper calls its Fibonacci-sphere generator, sample-mean/radius
normalization and nearest-point distance functions directly.

For metric surface samples `p_j`, let `c = mean(p_j)` and
`r = max_j ||p_j - c||`. Then:

```text
x_j = (p_j - c) / r
b_i = fixed Fibonacci-sphere query i
d_i = min_j ||b_i - x_j||, i = 0,...,127
```

The descriptor is **128 unsigned scalar distances**. The nearest points and
connecting vectors in the viewer explain those scalars; they are not extra
policy features. These points are not collision sensors or avoidance spheres.

Two deliberate adaptations from the upstream `process_usd` defaults:

- 128 basis points rather than 10.
- Always area-weighted triangle sampling: 16,384 points with seed 42, using
  trimesh's locally seeded sampler. GRAIL's small-mesh fallback would use only
  this mesh's 16 vertices, not its faces. Sampling is deterministic but is not
  intended to reproduce GRAIL's random sample sequence.

The GLB loader composes every scene-node transform and preserves instances.
It expects meters and the canonical object frame; no guessed unit conversions
or PCA/principal-axis rotation are performed. The sample centroid is not the
robot object's origin, center of mass, or bounding-box center. The saved
centroid offset and radius must accompany a later shape-aware observation
design. Pose/goal inputs remain necessary. A normalized BPS alone discards
absolute scale and is not intrinsically rotation invariant.

## Previewed object and validation

The hammer comes from the actual procedural Play2Perfect SAPG asset export:

`research/slides_assets/sapg_shape_examples_20260908/original_order/env_02/object.glb`

The neighboring metadata identifies training asset index 2 and source
`074_hammer_handle_(...)_head_(...).urdf`. This is the procedural training
hammer, not the separate DexToolBench claw-hammer evaluation asset.

- Actual size: 186.34 × 101.88 × 42.98 mm.
- GLB SHA256: `8300d17de2095d8b7419f1b3302ef7943feb7f1672d7c116897041adaf273b97`.
- Exported GLB bounds match the URDF boxes/origins within 2.20e-9 m;
  triangle surface areas also match.
- Normalization radius: 128.726 mm; descriptor range: 0.07119–0.91912.
- All 128 rendered nearest endpoints lie on triangles within 1.93e-9 m.
- Displayed normalized link lengths exactly equal the saved descriptor values.
- The sampled distance overestimates continuous-triangle nearest distance by
  at most 0.914 mm, mean 0.153 mm, over these 128 queries. This is measured
  approximation error for this asset, not a universal bound or reconstruction
  accuracy claim.

Triangle sampling follows the exported component surfaces, including component
interfaces; it does not perform a Boolean-union reconstruction. The explicit
triangle-distance check quantifies the approximation to that same input mesh.

## Run locally

Use an isolated environment, **not the workstation's Isaac Lab environment**.
The pinned viewer dependencies below were validated on local Python 3.14.

```bash
python3 -m venv .venv_bps_viewer
.venv_bps_viewer/bin/python -m pip install -r scripts/requirements-bps-viewer.txt
.venv_bps_viewer/bin/python scripts/visualize_object_bps.py \
  /absolute/path/to/object.glb --output outputs/bps128-new-preview --port 8088
```

The output directory must be new; existing artifacts are not overwritten.
The server binds to `127.0.0.1` only. On the laptop, open
<http://localhost:8088>; no workstation tunnel is needed.

- Amber dots: fixed queries. Mint dots: nearest sampled surface points.
- Pink segment/endpoints and pink plot bar: currently selected coordinate.
- Use **Selected BPS index** or its numeric input to inspect all 128 entries.
- **Only selected connection** removes the other connections and markers.
- **16,384 surface samples** and **Object mesh** show how the sampled input
  relates to the actual mesh. **Sphere guide** is only a visual reference.
- Orbit/zoom with the mouse; **Reset camera** restores the framing.

The scene is normalized; the sidebar shows both normalized distance and its
metric equivalent. It saves `bps_128.npz`, `descriptor.csv` and
`validation.json` with the basis, points, centroid, radius, nearest indices,
asset hash and upstream commit. Controls are shared across connected clients.

```bash
.venv_bps_viewer/bin/python -m pytest tests/test_object_bps.py -q
.venv_bps_viewer/bin/python -m playwright install chromium
.venv_bps_viewer/bin/python scripts/check_bps_viewer_browser.py outputs/bps128-new-preview
```

Ten geometry tests passed on 2026-09-08. Browser validation checks slider
extremes against the saved vector, mesh/cloud and selected-only controls,
JavaScript errors, and rendered amber-pixel changes. Screenshots and a JSON
report are saved alongside the descriptor. Browser screenshots are inspected
as well; these checks do not validate a learning experiment.

Current local preview artifacts: `outputs/bps128-hammer-viewer-20260908-v2`.
The full SAPG shape-input integration and training are still pending the
user's visual review. Workstation training queue was empty at launch.
