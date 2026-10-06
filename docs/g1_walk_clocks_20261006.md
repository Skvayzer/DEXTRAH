# Phase 0: SONIC walking at the manipulation training clock — 2026-10-06

## Question

The 0.4 m/s two-table brush test (`g1_two_table_navigation_040_20260915.md`)
failed: 0 arrivals, motion opposite the command 57% of the time. Bare SONIC
had only been shown to walk at its native 200 Hz physics / 50 Hz control on
the harness floor. Before training a carry expert we needed to know whether
the manipulation training clock (120 Hz physics, decimation 2, 60 Hz control)
or the training floor/solver settings break walking by themselves.

## Setup

Job **1034**, source `084a16e`, outputs `outputs/walk_clocks_1034/`.
Same empty-floor harness as `g1_empty_floor_walking_20260915.md`: original
hash-pinned SONIC (`62f3e336…`), released planner (`39b553e1…`), no SAPG
residual, no fingers, no table, no object, no optimizer. Both the original
NVIDIA G1 asset and the source G1+Revo2 asset. 14 s per trial: idle 0–2 s,
command 2–10 s, idle 10–14 s. Metrics use 4–10 s.

| Factor | `native` / `harness` | `task` / `training` |
|---|---|---|
| Clock | 200 Hz physics, 50 Hz control, `SonicHistory` | 120 Hz physics, 60 Hz control, `TimedSonicHistory` (as in training env) |
| Physics | friction 1.0, Isaac default solver bounds | touch-383: friction 0.5 average, TGS 8 position / 0 velocity iterations |

All four combinations were run. Eight predeclared commands each.

## Result: training clock and physics do not break walking

All **64 trials** completed with **zero falls**. Minimum pelvis height
≥ 0.72 m, minimum upright cosine ≥ 0.988, mean speed after the stop command
≤ 0.05 m/s in every condition.

Revo2 asset, mean body-frame velocity over 4–10 s (m/s; turns in rad/s):

| Command | native/harness | native/training | task/harness | task/training |
|---|---:|---:|---:|---:|
| forward +0.40 | +0.366 | +0.370 | +0.408 | +0.381 |
| backward −0.40 | −0.269 | −0.276 | −0.268 | −0.277 |
| left +0.40 | +0.435 | +0.410 | +0.424 | +0.354 |
| right −0.40 | −0.371 | −0.429 | −0.388 | −0.393 |
| right −0.40, yaw −90° | −0.405 | −0.395 | −0.370 | −0.389 |
| turn left +0.20 | +0.112 | +0.128 | +0.106 | +0.097 |
| turn right −0.20 | −0.119 | −0.153 | −0.062 | −0.114 |

Direction cosine ≥ 0.99 for every translational command. The native/harness
column reproduces the 2026-09-15 result (forward +0.310 then, +0.366 now;
same sign and magnitude class). Backward walking is slower than commanded
(about −0.27 m/s) in every condition, as on 09-15; it is a SONIC/planner
property, not a clock effect. Turning right at the training clock with the
harness floor was weaker (−0.062 rad/s) but recovered with training physics.

Single seed per trial; descriptive, not a success-rate estimate.

## Interpretation

The training clock and the touch-383 floor/solver are **not** the cause of the
two-table failure. The remaining differences between this passing test and
the failing combined test are:

1. The learned 64-D SAPG latent residual (zero here, active there).
2. The trained standing arm-reference baseline used during carrying.
3. Waypoint-feedback commands: direction recomputed from measured root
   position each check (359 planner calls in 120 s, about 3.0/s, versus
   19–35 per 14 s trial here, about 1.4–2.5/s). The rate difference is
   modest, so this is a weaker candidate than items 1–2, but untested.
4. Table/object contacts, source domain randomization, and env-origin-relative
   `palm_pos` in the policy observation.

Next diagnostic: walk in the full manipulation environment at 0.4 m/s with
constant body-frame commands, first with zero residual/finger means (isolates
items 2 and 4), then with the trained residual (item 1), then with waypoint
feedback (item 3). That environment's standalone startup stalled in
`sim.reset()` on 2026-09-16 (jobs 616/617), so that must be resolved first.

## Operational notes

- The isolated planner runtime lives in
  `/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps`; recipes must
  append it to `PYTHONPATH`. Job 1033 omitted it and failed with
  `ModuleNotFoundError: onnxruntime`.
- Kit still hangs in `app.close()` after saving. The recipe now judges each run
  by `metadata.json` and kills the process 60 s after completion or error.
- Slurm partitions are now `gpu` and `cpu`; recipes were updated in `b1cac9b`.

Reproduce the table with
`python scripts/summarize_walk_clocks.py outputs/walk_clocks_1034`.
