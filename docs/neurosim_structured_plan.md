# Structured hand–tool–table simulator: build plan

2026-10-30. Branch `feature/g1-neurosim-structured` (from `feature/g1-neurosim` @ 2c3e35a). This is the build plan
for the structured ("gradient-faithful") model of `docs/neurosim_stage3_plan.md` (rev. 2.1, Phase 2 onward); that
document keeps the research design (data, reference gradients, gates G0–G4), this one the model, its work packages
and pass tests. Where they disagree on the model, this document wins; on data and gates, rev. 2.1 wins.

## Decisions (2026-10-30)

| Decision | Choice | Reason |
| --- | --- | --- |
| Branch | Build on `feature/g1-neurosim-structured`; merge into `feature/g1-neurosim` once WP1–4 pass | Avoids colliding with the session running the hold harvest on `feature/g1-neurosim` |
| Hand geometry for M's holds | Signed-distance grids of the training hulls (~1 mm, trilinear) | M's holds were made by the hull hand. The primitive hand fills only 35–68% of each hull, so it turns real grips into penetrations or different contacts; the harvest already shifts 86% of holds (median 17°) from Warp → C MuJoCo, and a second geometry change would add a second distortion |
| Hand geometry for development | Primitive hand (testbed 'capsule' variant) for WP1–4 unit tests and the first contact/friction fits | Fully analytic and fast; switch to hull grids for replaying M's holds (WP5) |
| First task slices | Carry, then place/release; M-grasp second, on the same simulator | Carry arrival (~30–35%) and placement are the largest losses; holding, slip, touchdown and release have continuous margins and a strong palm channel; the harvested holds are their start states. Grasp acquisition switches contact modes more often (weakest gradients) and needs its own approach/closing data |
| Compute | CPU only until the gradient benchmark (rev. 2.1 §7, gate G3) gives a go | One-GPU rule; nothing here needs a GPU before a policy test |

## 1. What the model is

One structured simulator of the Revo2 right hand on a prescribed palm trajectory, one rigid tool (M's 4-slot
handle–head bank) or test cylinder, and the table, stepping exactly as `Testbed.step` (60 Hz control, 2 × 1/120 s
substeps). Batched PyTorch, float64 on the CPU for validation (float32 option for speed). The tasks (carry,
place/release, later M-grasp) differ only in their start states and in which margins are judged.

| Part | What | Learned |
| --- | --- | --- |
| Spec | Extracted from the compiled testbed model (`MjModel`), never retyped: tree, joint axes/ranges, masses, inertias, armature, servo gains, geoms, contact parameters | no |
| Kinematics | Hand FK on the prescribed palm (pose and velocity interpolated exactly as `Testbed.step`), link/point Jacobians | no |
| Fingers | Exact velocity servo (`v_cmd = clip(g (q* − q), ±vmax)`, `τ = kv (v_cmd − q̇)`), implicitfast-style implicit damping, 11×11 joint mass block, gravity, joint limits | no |
| Object | Newton–Euler of one rigid body (mass, diagonal inertia at the COM, from the bank) | no |
| Geometry | Analytic SDFs (box, capsule, sphere, cylinder) for the tool, table and the primitive hand; SDF grids of the 17 hand hulls (MuJoCo's own ≤ 64-vertex hulls) | no |
| Contact points | Mirror MuJoCo's contact generation where it is closed-form (box/capsule on the table face: corner and end points); witness points + deepest-point selection per (hand link, tool part) | — |
| Normal contact | MuJoCo's soft-constraint law with the exact solref/solimp (below), coupled solve over active contacts | per-pair-type correction |
| Friction | MuJoCo's own soft elliptic cone, mirrored exactly (μ 0.5, fingertip markers 1.0, torsion 0.005 m, impratio 10); see §6 for why not a bristle model | none (exact where contacts match) |
| Residual | Small bounded network on local contact features | yes (WP6) |
| Backward-only smoothing | Soft servo band s ≈ σ (rev. 2.1 A1); pushing-only long-range contact tail; added as `f + (f_soft − stopgrad(f_soft))` so forward rollouts stay exact | band and tail chosen by the benchmark |
| Outputs | Next state, per-finger forces, margins: slip utilisation, grasp margin, tipping margin, distances, touchdown speed | — |

### 1.1 MuJoCo's contact law (verified in the testbed, MuJoCo 3.11)

For a contact with signed distance r (< 0 in penetration) and normal velocity v:

- solref (timeconst 0.02, dampratio 1) with `refsafe` (timeconst ≥ 2·dt): `k = 1 / (dmax² · tc² · ζ²)`, `b = 2 / (dmax · tc)`.
  Measured `efc_KBIP` = (2770.08, 105.26) = these values with dmax 0.95.
- Impedance `d(|r|)` from solimp (0.9, 0.95, 0.001, 0.5, 2): from dmin 0.9 to dmax 0.95 over the first 1 mm.
- Reference acceleration `a_ref = −b·v − k·d·r`; regulariser `R = (1 − d)/d · Â`, with Â (`efc_diagA`) the sum of the
  two bodies' `body_invweight0` translational terms for the normal row (object: 1/m) — measured 3.333 for a 0.3 kg tool.
- Elliptic friction rows (measured): cone coefficient `μ = f₀ / sqrt(impratio)` (0.158 for f₀ = 0.5);
  `Â_j = Â_0 (f₀ / f_j)² / impratio` and `R_j = (1 − d)/d · Â_j` (tangential ×0.1, torsional ×1000 for
  f = (0.5, 0.005)); `a_ref,j = −b·v_j` (no position term); the impedance d of the contact applies to every row.
- The forces solve MuJoCo's primal problem `min ½ (x − a₀)ᵀ M (x − a₀) + Σ s(Jx − a_ref)` with the plain M
  (implicitfast's damping does not enter the solve) and its three-zone elliptic cost; `constraint.solve_primal`
  implements it with Newton's method over the 17 DoF (11 fingers + 6 object), independent of the contact count.
- A single frictionless contact has the closed form `f = (a_ref − a_0)/(A + R)`; at rest `f = k·d·|r| / R`, so the
  static penetration depends on how many contacts share the load. Mirroring MuJoCo's contact count per pair type is
  therefore part of matching it, not a detail.
- Contact generation in this testbed: `mjDSBL_MULTICCD` and `mjDSBL_NATIVECCD` are off (multi-contact CCD on). A box
  tool resting on the table produces 8 contacts (4 corners per box part); capsule-on-box up to 2. Hand hulls on tool
  parts generate several contacts per pair via multi-CCD; this is where the per-pair-type correction is learned.

## 2. Code layout

`dextrah_lab/neurosim/structured/`:

| File | Content |
| --- | --- |
| `rotations.py` | Batched quaternion / rotation utilities (wxyz, MuJoCo conventions: `mju_quatIntegrate`, `mju_quat2Vel`) |
| `spec.py` | `HandSpec`, `ObjectSpec`, `SceneSpec` extracted from a compiled `Testbed` |
| `kinematics.py` | Palm interpolation, hand FK, point / rotation Jacobians, mass matrix, gravity, Coriolis (forward-mode) |
| `sdf.py` | Primitive SDFs, exact convex-hull SDF, hull SDF grids (trilinear lookup) |
| `servo.py` | Exact servo, soft backward surrogate |
| `constraint.py` | MuJoCo impedance law, `Rows`, primal Newton solve (unilateral + elliptic rows) |
| `contact.py` | Witness points, contact generation (object–table, hand–object, hand–table), contact rows |
| `dynamics.py` | `SimState`, `StructuredSim.step(state, targets, palm, palm_next, hold_object)` |

Tests: `tests/test_neurosim_structured.py` (CPU; every comparison is against the C MuJoCo testbed). Scripts:
`scripts/neurosim/structured_*.py`.

## 3. Work packages and pass tests

| WP | Content | Pass test (against the testbed, float64) | Estimate |
| --- | --- | --- | --- |
| WP1 | Spec extraction, rotations, palm interpolation, hand FK, point Jacobians, primitive SDFs | FK body poses within 1e-9 m / rad of `mj_kinematics` at random joint angles and palm poses; palm substep poses equal `Testbed.step`'s; SDF of box/capsule/sphere/cylinder equals `mj_geomDistance` (< 1e-6 m) outside and the analytic depth inside; Jacobians equal `mj_jac` | 2–3 d |
| WP2 | Contact-free dynamics: servo + implicit damping + joint mass block + gravity + Coriolis + limits; object free flight | One substep from random states (limits, clipped forces, moving palm): q̇ within 1e-4 rad/s; 60 re-synced control steps: q within 1e-6 rad; object free flight 1 s within 1e-9 (open-loop finger trajectories are not a valid test: free fingers limit-cycle, §5) | 1–2 d |
| WP3 | Contact points + normal law + coupled solve | Tool resting on the table: penetration and per-corner force within 5% of MuJoCo; fingers pressing on a welded tool: per-finger normal force within 10–20% | 2–3 d |
| WP4 | Friction (MuJoCo's elliptic cone, §6) | Slide / spin on the table matches MuJoCo (done with WP3); tool held 2 s by a hand grip: creep within 2× of MuJoCo's; stick → slip onset within one control step of MuJoCo's | 2–3 d |
| WP5 | Hull SDF grids (~1 mm, 0.5 mm check on flat faces), hull witness points; replay of harvested M holds | Grid SDF vs exact hull distance < 0.3 mm (1 mm grid) near the surface; held/shifted/dropped agreement with the testbed on ≥ 300 harvested holds | 2–3 d |
| WP6 | Fitting: per-pair-type correction, friction parameters, residual (staged, rev. 2.1 §6) | Gate G2 (rev. 2.1 §6) | 4–5 d |
| WP7 | Gradient benchmark + utility test on the carry and place/release slices (rev. 2.1 §7) | Gate G3 (go / no-go) | 3–4 d |
| WP8 | Policy-facing API, reflex layer | after a go | — |

Slices for WP5–7, in order: **C-carry** (harvested holds + logged walking palm motion), **place/release** (holds
lowered onto the table, fingers opened), then **M-repose** and **M-grasp** (reaching family; needs approach/closing data).

## 4. Status

| WP | Status |
| --- | --- |
| WP1 | done: FK / anchors / axes 1e-15; Jacobians, 11×11 mass matrix, gravity 1e-12; palm substeps exact; primitive SDFs = `mj_geomDistance` (< 1e-6 out, < 1e-5 in); exact hull SDF of all 17 hulls within 9e-7 m; 1 mm grids near the surface median 0.01 mm, p95 0.07 mm, max 0.29 mm |
| WP2 | done: finger substep q̇ within 1.5e-5 rad/s; re-synced control steps q within 1e-6 rad; object free flight 1e-14 |
| WP3 | partly done. **Object–table: exact** — box tool landing, sliding and spinning on the table, 40 re-synced steps: object velocity within 3e-15 (condim 1) and 4e-9 (condim 4, elliptic friction); same 8 contacts, depths and normal forces as MuJoCo. **Hand–object: approximate** — fingers pressing an immovable cylinder (8 scenes per hand): per-finger normal force median relative error 8% (primitive hand) / 6% (hull grids), p90 38%; MuJoCo makes 1–5 contacts per (link, cylinder) pair by multi-CCD (24–31 in total), we make 1 (8–12). Naive extra points (farthest-point, own normals, `per_pair=5`) are worse (median 71% / 124%). Next: mirror multi-CCD's patch (one normal per pair, clipped support polygon), then a release / drop test |
| WP4 | friction law done with WP3 (MuJoCo's elliptic cone, exact on the table); hold-creep and stick→slip tests pending on hand grips (needs WP3's contact patches) |
| WP5–8 | not started (WP5's hull grids exist and pass the accuracy test) |

## 5. Findings while building (MuJoCo 3.11, testbed)

1. **Joint actuator force range.** MuJoCo's URDF import sets `jnt_actfrcrange` = ±effort on every hand joint, so the
   applied servo torque is `clip(kv (v_cmd − q̇), ±effort)`, although the actuator itself is unclamped
   (`robot.hand_servo`'s docstring says "no force clamp"). The clip binds whenever q̇ opposes v_cmd (reversals),
   e.g. −0.5 instead of −1.07 N·m on the thumb metacarpal. implicitfast keeps the −kv damping derivative even while
   the force is clipped (verified to 1e-15).
2. **Free fingers never settle: the 120 Hz servo limit-cycles.** Inside the band the discrete loop gain is
   ≈ (dt·kv / (M + dt·kv))·g·dt ≈ 10–20 ≫ 2, so each joint chatters around its target. Free hand, constant targets,
   last 1 s: |gap| median 6–15 mrad (8 mrad typical), max 11–20 mrad, joint speeds up to ~2 rad/s. **This is the same
   size as the census gaps in rev. 2.1 A1/A2 (median 9–13 mrad)**, so part of the measured "saturation" may be chatter
   rather than grip; the census should be split by blocked (in contact, q̇ ≈ 0) vs moving joints before A2's gap
   distribution is used. The same servo law and dt run in mjlab training, so the chatter is in the training data too.
   Consequence for tests: open-loop finger trajectories diverge from 1e-7 differences, so pass tests re-sync to
   MuJoCo's state every step.
3. **Coriolis / centrifugal forces matter for the fingers** (per-substep q̇ error 2e-2 rad/s without, 1.5e-5 with),
   mostly from the palm's angular velocity. They are explicit in implicitfast (treating their derivative implicitly
   gives 5e-3).
4. **The object's gyroscopic term is implicit in implicitfast** (MuJoCo 3.11): `(I − dt ∂τ/∂ω) Δω = dt τ` with
   τ = −ω × Iω matches to 1e-14; explicit Euler is off by 2e-3 rad/s per substep at 6 rad/s.
5. Joint limits are soft (solref 0.02, solimp 0.9–0.95, `dof_invweight0` as Â): fingers pushed below 0 sink to
   −0.07 rad; the structured solve reproduces MuJoCo's limit forces.

6. **Friction: MuJoCo's soft elliptic cone instead of a bristle model** (change from rev. 2.1 §5.3). The reference
   simulator *is* MuJoCo, and its friction law could be mirrored exactly (object–table matches to 4e-9), so the
   forward model needs no fitted friction parameters wherever contacts match. MuJoCo's friction is itself soft: it
   creeps at a rate set by R_t, which the mirrored law reproduces by construction; a bristle model would have had to
   be fitted to that creep. A bristle state remains an option later for real-robot calibration.
7. **Contact count is the open issue for hand grips** (WP3 status): the coupled solve and the law are exact, so the
   remaining hand–object error comes from contact generation (count, placement, normals of multi-CCD patches).
8. The held-state generator (`reference/held_states.try_hold`) produced 0 holds in 400 trials here (primitive and
   hull hand; 'overlap' / 'no_grip'), consistent with the ≤ 1.5% reported on the server; WP3 tests therefore use
   fingers closing on an immovable cylinder. Harvested M holds (server) are the intended WP4/WP5 scenes.

Local development (no `/data1`): set `DEXTRAH_ROBOT_URDF` to a checkout of `unitreerobotics/unitree_ros`
(`robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf`). The upstream URDF may differ from the server's
training copy; every pass test compares against a testbed compiled from the same URDF, so the tests are unaffected,
but fitted parameters must be re-checked on the server's URDF before use.
