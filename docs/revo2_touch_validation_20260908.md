# Revo2 Touch validation — 2026-09-08

Result: the optional tactile + arm motor-effort interface passes small-scene
CPU and GPU PhysX tests. **No tactile policy was trained.** This validates the
implementation and input contract, not calibrated sim-to-real accuracy or
large-scale training performance.

Implementation: `19d31ce` (readout/response), `bde2ca8` (SAPG task option),
`13926e6` (physics bench), `67e3a17` (visible force glyphs/browser checks).
See [configuration and limitations](revo2_touch_option.md).

## Scope and safeguards

Isaac Sim 5.0, installed Isaac Lab, PyTorch 2.7, RTX 6000 Ada on `tl-server-0`.
Each diagnostic scene has 12 environments and 12 object assets. Contact tests
use one indenter in environment 0; the other environments check isolation.
The contact fixture holds the G1 right hand with bench-only joint limits and
targets; it is not an infinitely rigid clamp. Actual link poses are used for
sensor frames and visualization. Normal task tests restore original limits
and default positions before stepping.

Diagnostics ran as bounded Slurm steps in the existing allocation, from an
isolated source copy. Existing BPS training job 355 was not stopped, restarted,
or modified. The diagnostic GPU shares its allocated GPU, so elapsed diagnostic
time is not a training-throughput benchmark. The CPU viewer avoids continuous
GPU physics contention and pauses when no browser is connected.

## Results

| Check | CPU PhysX | GPU PhysX |
|---|---:|---:|
| Five fingertip press/release tests | Pass | Pass |
| Index peak compression | 0.1996 N | 0.1998 N |
| Held-contact shear x, sensor frame | 0.0872 N | 0.0805 N |
| Held-contact friction, fixture tangent | 0.1176 N | 0.1111 N |
| Held-contact mean total-force balance error | 0.01372 N | 0.000260 N |
| Forward / reverse sliding friction | +0.1237 / −0.2537 N | +0.1296 / −0.2471 N |
| Backside contact, unmasked link peak | 1.0005 N | 0.9366 N |
| Backside contact, masked tactile output | 0 N | 0 N |
| SAPG actor / critic inputs, both options | 256 / 278 | 256 / 278 |
| Diagnostic optimizer updates | 0 | 0 |

Peak forces are fixture measurements, not calibration targets. The sensor-plane
shear includes projected total contact force; the friction-only diagnostic is
separate. Tilted/curved collision surfaces can therefore give nonzero sensor
shear during a nominal normal press.

The held test applies a constant 0.05 N lateral fixture drive without a moving
tangent target. From 2.5–3.8 s, relative tangential pose excursion is 17.3 µm
on CPU and 1.20 µm on GPU; net drift is below 0.36 µm on both. The simulator's
reported velocities and finite-difference pose motion differ in this fixture.
Both are retained in traces; held-contact qualification additionally bounds
actual relative pose motion, rather than calling any nonzero shear "static".
The GPU excursion/drift checks were also evaluated from its saved traces;
its sweep started before those two summary fields were added to the script.

All four observation layouts passed real environment steps, finite-value and
BPS-layout checks, partial resets, forced timeout/terminal-critic handling,
and forward passes through the actual SAPG actor and critic:

- Both off: 224 / 246 actor / critic dimensions.
- Tactile only: 249 / 271.
- Arm motor estimates only: 231 / 253.
- Both on: 256 / 278, plus SAPG's separate internal 32-dimensional embedding.

The targeted unit suite passed **37 tests**:

```sh
python -m pytest -q tests/test_touch_observations.py \
  tests/test_g1_tactile_forces.py tests/test_g1_contact.py
```

Browser automation passed: actual loaded contact readings, pause/resume,
test selection and release, with no JavaScript errors. The final press, shear
and release screenshots were saved locally; the shear view was inspected to
confirm probe/pad alignment and visible force arrows. Offset arrows have gray
connectors to actual contact locations and do not move the physical probe.

## Evidence

Committed machine-readable summaries:

- [CPU contact/contract checks](validation/revo2_touch_20260908/cpu_validation.json)
- [GPU contact/contract checks](validation/revo2_touch_20260908/gpu_validation.json)
- [Tactile-only contract](validation/revo2_touch_20260908/tactile_only_validation.json)
- [Torque-only contract](validation/revo2_touch_20260908/torque_only_validation.json)
- [Both-off contract](validation/revo2_touch_20260908/both_off_validation.json)
- [Browser checks](validation/revo2_touch_20260908/live_browser_validation.json)

Full traces, screenshots and copied summaries are in local
`outputs/revo2_touch_live/`. Workstation logs/traces are under
`/data1/users/konstantin.smirnov/revo2-tactile-live.XpwuhD/`, specifically
`outputs/cpu_final`, `outputs/gpu_final`, `outputs/touch_only_final`,
`outputs/torque_only`, and `outputs/both_off`. The live viewer uses
`outputs/viewer_final` and port 8091, forwarded to the laptop.

## Still not established

Pad stiffness/damping/friction and hardware shear-axis alignment are provisional.
The whole distal collider is compliant, not a modeled soft pad over a rigid core.
Motor efforts are clipped PD estimates, not exact solver reaction torques.
Self-contact remains disabled; unregistered collision partners fail visibly.
No 24k-environment tactile memory/throughput benchmark, trained-policy evaluation,
trained-checkpoint observation migration, or hardware transfer is claimed.
