# G1/Revo2 Touch contact diagnostics

This is an **opt-in simulation diagnostic**, not a new training observation or
reward. The ongoing G1 ADEPT/SAPG run, its fabric/PCA controller, 131/153 actor/
critic dimensions, 24,576 environments, and 1,200-tool training pool are unchanged.
The viewer uses **12 environments and 12 procedural tools** (one per distribution).

## What follows ADEPT, and what does not

[ADEPT Appendix A](https://arxiv.org/html/2608.19182v1#A1) uses fingertip force
information and a grasp condition requiring the thumb and another finger to
exceed 1 N. The diagnostic exposes five world/local 3D normal-contact resultants
and the same strict threshold condition. It separately computes object-only and
pad/object-only gates: touching a table or calibration probe must not count as
grasping the manipulated object.

This is **not ADEPT's TacMap/FiLM visuotactile student**. Its vision-based Sharpa
tactile maps are not interchangeable with capacitive Revo2 Touch readings.
[BrainCo's capacitive protocol](https://www.brainco-hz.com/docs/revolimb-hand/en/revo2/modbus_touch.html)
reports normal force, tangential magnitude/direction, proximity and status;
normal/tangential force units are 0.01 N with a 25 N measurement range. This
implementation does not fabricate a pressure image, friction/shear or a slip
estimate from normal-only PhysX data. No hardware command is sent.

## Important model discovery

The training G1 URDF's `right_*_tip` collision meshes are only about **0.5 mm**
across. They are geometric tip markers, not the tactile pads. Attaching sensors
there would miss most real finger/object contacts.

Sensors instead bind to the five `right_*_distal_link` bodies. Individual PhysX
contact points are transformed into the corresponding Touch CAD frame and
filtered by its bounding volume, expanded by a provisional 2 mm tolerance.
The CAD comes from the pinned BrainCo submodule, commit
`92cc697c7fa691db59404ce52344f3969a5ef7a6`.

The standalone Touch model has different joint transforms/mimic ratios from
the training G1. We only align its pad geometry through matching distal-link
frames; we **do not replace the training robot's kinematics or collision mesh**.
Every test probe surface point is selected on the actual G1 collision mesh
inside that CAD window. This is a geometric proxy, not a calibrated sensing
footprint. Its registration and edge response need real-hand validation.

Related issue to review before the next training change: Play2Perfect's
`G1_BRAINCO_FINGERTIP_LINK_NAMES` also points to these tiny marker bodies, and
`apply_physx_material_properties` applies the special fingertip friction only
to that list. The actual distal contact surfaces therefore receive the generic
robot friction (configured 0.5), not the intended fingertip value (configured
1.5 before fingertip randomization). **This has not been changed in the running
training or diagnostic**, so comparisons remain against the existing physics.
Do not globally rename that list without auditing its observation/kinematic
uses; a material-specific mapping is the safer follow-up.

## Signal contract

Finger order is always **thumb, index, middle, ring, pinky**. Filter order is
**object, table, diagnostic probe**. Every finger has its own ContactSensor view
because Isaac Lab's filtered reporting is one sensor body to multiple targets.

| Signal | Meaning |
| --- | --- |
| `net_w`, `net_local` | All distal-body normal-contact resultant vectors, N |
| `pairs_w` | Distal-body vectors separated by object/table/probe |
| `pad_pairs_w` | Same vectors, summing only individual contacts inside the pad window |
| `pad_load_n` | Sum of normal-force magnitudes inside each pad/channel; no vector cancellation |
| `pad_points_w` | Normal-load-weighted pad contact centroid; valid only if load > 0 |
| `points_w`, `point_valid` | All-distal pair centroid and explicit validity mask |
| `unclassified_w` | All-distal force minus known pair forces; **not** a self-contact classification |
| `gate_all`, `gate_object`, `gate_pad_object` | Separate strict >1 N thumb-and-other conditions |

The raw/filtered display is the magnitude of the pad resultant across the
three known target classes. A 30 ms exponential filter and 0.15/0.08 N
on/off hysteresis provide contact duration and a filtered force derivative.
These are provisional engineering choices, not ADEPT or BrainCo calibration.
Raw data are never clipped to hide large simulated forces. Opposing normal
vectors can cancel; the scalar load display makes this distinction explicit.

The pad stream currently localizes contacts with the three filtered target
classes only. Contact with unclassified objects remains visible in the all-link
resultant; it is not silently treated as a valid localized tactile reading.
Self-collision remains disabled in the current training physics model.

Object identity and exact contact locations are **privileged simulation data**,
not observables available from the capacitive hand alone. The real sensor's
frame, packet freshness/status, baseline, units, saturation, bandwidth, lag and
noise remain to be calibrated and integrated. No 70 Hz hardware stream is
claimed or emulated by this 120 Hz simulation diagnostic.

## Validation performed on 2026-09-07

32 contact/fabric/collision/PCA regression tests passed. The PhysX validator
passed on 12 environments, with the same solver timestep and fabric/PCA path:

| Controlled test | Result |
| --- | --- |
| Free space | No unexpected distal contact |
| Thumb / index / middle / ring / pinky pad probes | Peak normal loads 5.58 / 0.58 / 0.59 / 0.59 / 0.49 N |
| Back of index finger | 2.27 N all-link contact, **0 N pad contact** |
| Actual procedural tool against index pad | 3.30 N pad contact |
| Table fixture | 70.85 N peak distal contact, no object-pad grasp gate |
| Release / subset reset | No lingering probe force; reset clears only selected filter state |
| Environment and target-class isolation | No probe leakage to other environments or object/table channels |
| Raw point aggregation | Reconstructed force agrees with PhysX pair-force matrix at every tested step |
| Observations | Original 131/153 dimensions retained; all finite |

These peak forces are **diagnostic fixture results, not hardware force targets**.
In particular, the table is deliberately repositioned into contact; that test
does not represent safe manipulation. No learned grasp success is claimed.
The web model uses measured distal-joint angles rather than ideal mimic values;
its five distal frame origins are checked against PhysX with a 2 mm tolerance
on every publication.

The browser test also exercises the actual frontend, not merely HTTP serving.
The installed Viser 0.1.34 cannot serialize NumPy boolean visibility flags and
has a React hook failure with our theme-toggle configuration (reproduced in an
empty CPU-only scene). The viewer uses native booleans, persistent glyphs and
the default theme; no installed dependency or training environment was upgraded.

## Running and viewing

On the workstation, from the repository root in the `adept_dextrah` environment:

```bash
python scripts/visualize_g1_contacts.py --headless --validate-only --num_envs=12
python scripts/visualize_g1_contacts.py --headless --num_envs=12 --viser-port=8089
```

Use a Slurm GPU allocation. The current viewer is a separate overlapping step
inside our training allocation, with two CPU cores and small PhysX buffers. It
uses about 1.6 GiB additional VRAM, is rate-limited to real time / 12 web updates
per second, and pauses physics when no browser is connected. It ends when that
allocation ends. It never loads the SAPG checkpoint or writes W&B training data.

From the laptop:

```bash
ssh -N -L 8089:127.0.0.1:8089 konstantin.smirnov@tl-server-0
```

Open `http://localhost:8089`. Start with **Pad probe**, choose a finger and use
**Focus hand**. Increase the fixture gap to release contact. Compare
**Back-of-finger probe**, **Tool contact fixture**, and **Table contact fixture**.
**Free physics** removes the imposed fixture; Open/Power grasp/Thumb-index pinch
and manual controls command the original fabric/PCA path, not a trained policy.

Cyan marks inactive CAD pads, yellow active pads. Orange/blue/purple arrows show
tool/table/probe normal-contact resultants. Red shows outside-pad contact.
Arrows have a fixed visual length and show direction; numeric force readings
remain uncapped. No force magnitude should be inferred from glyph length.

## Before using this in training or hardware

1. Inspect pad registration and contact/release behavior in the viewer.
2. Validate real-hand zero-load offsets, known-force response, sensor axes,
   edge coverage, timestamp/status handling, saturation and safe operating range.
3. Add a hardware-aligned, non-privileged observation stream, with realistic
   sampling/delay/noise. Keep object-filtered contacts critic/reward-only.
4. Benchmark an optimized aggregate contact path at 24k environments. The
   individual-point checks and CAD masking here favor auditability, not training
   throughput; they must not simply be enabled at full scale without profiling.
5. Run controlled no-touch / touch-critic / touch-actor ablations before claiming
   faster SAPG learning or better manipulation.
