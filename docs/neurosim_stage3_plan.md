# Gradient-faithful hand–object simulator: implementation and build plan (rev. 2.1)

2026-10-29. Revision 2 was written by a research agent against `feature/g1-neurosim` @ b4a4d64 (the testbed, the servo law
in `dextrah_lab/mj/robot.py`, `docs/neurosim_stage3_setup_review.md`). Revision 2.1 adds the amendments in §A, from a
servo-saturation census of the trained carry expert's own data. Revision 2 is kept below unchanged; where §A and the
body disagree, §A wins.

## A. Amendments in rev. 2.1 (census of the carry expert, 2026-10-29)

**Census.** Carry expert C v2 (run 1587), stage 1 log `outputs/neurosim/go_nogo_v2/data`, 5 of 18 chunks
(3.1M steps). Gap = finger target applied during the step minus the joint angle after it, per actuated joint. (The log
stores `hand_target` in the environment's actuated order and `hand_q` in sorted joint order; columns were matched by
correlation, r = 0.99–1.00 for five joints.)

| Joint | Steps in contact | Outside the linear band | Median gap | 10th / 90th pct | Target on the closing side, in contact |
| --- | --- | --- | --- | --- | --- |
| Thumb metacarpal (band 0.42 mrad) | 77% | 97% | 8.6 mrad | 1.5 / 32 mrad | 33% |
| Thumb proximal (0.92 mrad) | 77% | 96% | 12.9 mrad | 2.1 / 166 mrad | 85% |
| Index proximal (1.67 mrad) | 23% | 90% | 9.6 mrad | 1.8 / 30 mrad | 68% |
| Middle proximal (1.67 mrad) | 21% | 90% | 8.9 mrad | 1.7 / 26 mrad | 61% |
| Ring proximal (1.67 mrad) | 8% | 90% | 8.7 mrad | 1.7 / 26 mrad | 61% |
| Pinky proximal (1.67 mrad) | 0% | 93% | — | — | — |

Saturation is confirmed (90–97% of steps), but the gap is small: median 9–13 mrad, 5–10× the band, about 10% of steps
inside it. The policy works near the band edge, not deep past it.

**A1. Finger gradients are not structurally zero under noise; derive the soft servo band from σ.** Reference gradients
average over executed-target noise σ = 5–50 mrad, larger than the typical gap, so the noise-averaged servo behaves like
a servo whose band is ~σ wide and its expected gradient in a hold is non-zero. Exactly zero holds only at σ = 0 and in
the deep tail (90th-percentile gaps 30–166 mrad). The soft backward servo's band scale is therefore s ≈ σ (matched to
the reference's noise), not a free knob; saturated in-contact states are gated like the rest (§7.3), with the exact
backward servo kept as the ablation. Stage 1.5's ±0.05–0.1 perturbations exceeded the gap, which explains the one-sided
responses (antisymmetry −0.15 to −0.44).

**A2. Held states copy the policy's gaps.** The held-state generator (0.1) closes each gripping finger until contact,
then sets its command past the contact angle by a gap drawn from the census distribution above, instead of random
closure levels (which would push targets deep and exaggerate saturation). Commands use the training interface: 6
actuated commands, distal targets by coupling (finger distal = 1.155 × proximal, thumb distal = 1.0 × thumb proximal).

**A3. Held states copy the policy's grasp type.** The carry expert holds with the thumb (77% of steps in contact), index
(23%) and middle (21%) fingers; ring 8%, pinky never. Held states favour thumb–index–middle grips (ring sometimes, pinky
rarely), not five-finger power wraps.

**A4. Palm channel: measure the arm-tracking approximation now.** The stage 1 log records the arm reference (`arm_now`,
`command`) and palm velocities, so the lag and bandwidth of SONIC's arm tracking (the "perfect tracking with a measured
lag" approximation of §9) can be measured on the CPU before Phase 5.

**A5. MuJoCo's impedance sets the reference acceleration, not the force.** solref / solimp define each contact's
reference acceleration and impedance; the force comes out of the solve with the effective inverse mass A and the
regulariser R = (1 − d)/d · Â (d the impedance). A single frictionless contact has a closed form, f = (a_ref − a_0)/(A + R);
with several contacts and elliptic friction cones the solve couples them. The learned "coupled-solve correction" in §5
is exactly the step from the per-contact closed form to the multi-contact result.

---

## Revision 2 (research agent), unchanged

### 0. What changed from revision 1

Adopted from the code and the review: prescribed palm motion with armature 1e3 (not mocap, not a weld); the exact
velocity-servo law and known distal coupling (1.155×, thumb 1.0×), not a learned actuator; per-link best primitive
(boxes on proximal links, capsules on distal links) as variant (a), plus signed-distance grids of the hulls as variant
(b); the testbed's own cylinders, so no dependency on the main plan before Phase 5; Warp parity by loading the testbed
model into MuJoCo Warp; holds generated in the testbed; radius 2.5–3.5 cm first; MJX in a separate environment, with
the collision-pair check in Phase 0.

New in this revision: the finger servos are saturated almost everywhere, so finger-command gradients are structurally
near zero during a hold (see A1 for the noise-averaged correction); pre-contact gradient targets are ill-posed as
specified, and the go criterion is reformulated; a utility test (short-horizon optimization in the true simulator) is
added before any policy training; the normal-contact law mirrors MuJoCo's known impedance rather than learning κ, b, β.

### 1. Goal and design

A learned hand–object–table simulator whose action-gradients help a policy, tested on setting a cylinder down upright.

| | State loss only | State + Jacobian loss |
| --- | --- | --- |
| Black box (stage-1.5 architecture, retrained on testbed data) | B0 | B1 |
| Structured (exact kinematics and servo, signed-distance contact, bristle friction, bounded residual) | S0 | S1 |
| Baseline | softened MJX (if the collision pairs are supported) | |

B0 retrained on the branched testbed data is the control for "better data". Phases 0–4 are CPU-only (4 cores, 19 GB
per job); Phase 0 also includes one short GPU job for Warp parity.

### 2. Facts from the code

2.1 Servo saturation: v_cmd = clamp((kp/kv)(q* − q), ±vmax), τ = kv(v_cmd − q̇), kp = 1200, kv = effort/vmax. The linear
band is |q* − q| ≤ effort/kp: 0.42 mrad (thumb metacarpal), 0.92 mrad (thumb proximal / distal), 1.67 mrad (fingers),
equal to vmax/gain. In a hold past the band, grip torque equals the effort limit and ∂(grip)/∂(command) = 0; in free
motion fingers move at vmax. The placement-deciding controls are the palm trajectory and opening timing, so palm motion
(a 6-D twist offset over the horizon) is added as a gradient channel in the benchmark, the utility test and the policy
test, reported separately from the 6 finger commands.

2.2 Other facts: contact rollouts diverge ~200× per 0.1 s (horizons 1–8 control steps); scripted grasps fail (0/72);
variant (a) primitives are fatter than the hulls (fill 0.35–0.68); training contact settings as in the setup review.

### 3. Phase 0: reference, held states, censuses

| Task | Detail | Output |
| --- | --- | --- |
| 0.1 Held-state generator | Cylinder at the grasp point, held fixed; fingers close (fingers before the thumb); release; settle 0.5 s; keep states that hold under gravity and a small palm shake. Vary grasp point, height, palm gap, size, mass. (Gaps and grip type per A2, A3.) | ≥ 5k held states per training size |
| 0.2 Determinism | Two identical runs bit-identical (C MuJoCo, float64, single thread) | Pass / fail |
| 0.3 Throughput | Control steps per second per core, mesh and capsule variants, with and without contact | Sizes Phase 1 |
| 0.4 Servo-saturation census | Fraction of in-contact finger joints outside the band, over held states and the episode script | Weight of finger-command gradients |
| 0.5 MJX check | Separate environment; collision support for cylinder–box, cylinder–plane, capsule–cylinder, box–cylinder, mesh–cylinder; same model file and options | Baseline feasible or not |
| 0.6 Warp parity | Load the testbed model into MuJoCo Warp; same scripted scenarios with replicas; compare noise-averaged outcome statistics with C MuJoCo | Reference question answered empirically |

Gate G0: deterministic reference; held-state bank built; saturation census and throughput recorded; MJX feasibility
resolved; Warp statistics within an agreed tolerance at the task σ.

### 4. Phase 1: data and reference gradients

Episodes as in the setup review (§7), with independent OU perturbations on every finger command (σ 0.05–0.15,
correlation 0.1–0.3 s), 20% near-misses, 10% tipping pushes; most episodes start from the held-state bank; the palm
trajectory gets its own OU perturbation (a few mm/s and deg/s). Sizes: radii {2.5, 3.0, 3.5} cm × heights {8, 12, 16,
20} cm; held out radius 3.25 cm, heights 10 and 18 cm, one mass band. 0.5–1M control steps, sized from 0.3.

Reference gradients: noise-averaged central differences with common random numbers, 32–64 draws, horizons 1/2/4/8, σ ∈
{0, 0.005, 0.02, 0.05} on executed targets plus palm noise {0, 1, 3} mm and 0.5°; action channels: the 6 finger
commands (coupled) and the 6 palm-twist components, random directions per group (matched with torch.func.jvp); the band
position of every finger recorded per state; pre-contact strata only where contact is reachable within the horizon;
outcomes: object position and rotation change, object velocity, per-finger normal force, grip force, slip margin,
stick-end time, tilt at the end of the horizon, object–table normal force. Gate G1: two independent estimates agree
(cosine ≥ 0.9) at the task σ, h ≤ 4, in-contact strata.

### 5. Phase 2: structured model

Exact forward servo with a smoothed backward surrogate (straight-through, band scale s; per A1, s ≈ σ); normal contact
mirroring MuJoCo's impedance law with a learned coupled-solve correction (A5) and the residual; LuGre/Dahl bristle
friction per witness point (tangential and torsional, μ_tors 0.005 m, marker μ 1.0), fitted also to stick→slip timing;
dynamic fingers (servo + contact torques); geometry (a) then (b) (1 mm grids, check 0.5 mm on flat faces; ~8 rim points
per cylinder face); palm as an input pose trajectory; semi-implicit Euler with N substeps from a stability sweep.
Layout: `dextrah_lab/neurosim/structured/{kinematics,servo,sdf,contact,friction,integrator,model}.py`,
`reference/{held_states,episodes,fd_targets}.py`, `losses.py`, `bench/{gradient_bench,utility_test}.py`,
`baselines/mjx_soft.py`; tests in `tests/test_neurosim_structured.py` (kinematics vs MuJoCo; SDF vs mj_geomDistance;
servo forward = testbed servo + actuator force; normal force vs MuJoCo on single static contacts; 2 s hold < 2 mm creep;
slip at μf_n; soft backward gives non-zero force gradients in saturated holds, exact gives zero; pre-step palm inputs only).

### 6. Phase 3: fitting and training

S staged: contact law on net per-link-pair force and torque (teacher-forced, Huber); one-step accelerations; 8-step
rollouts with the residual; S1 adds the Jacobian loss. B0/B1 on the same data. Jacobian loss: normalized error plus
(1 − cosine) on directional derivatives, finger and palm channels separately, at the matching (σ, h). Touchdown and
release slices weighted 3×. Gate G2: 2 s hold < 2 mm creep; touchdown force within 20%; 8-step error ≤ ½ persistence on
held-out sizes; release time within one control step on ≥ 80% of slow-opening trajectories.

### 7. Phase 4: evaluation

7.1 Gradient benchmark: all candidates × σ × horizon × stratum × held-out sizes; cosine, magnitude ratio, sign
agreement, finger vs palm channels, saturated vs in-band fingers; bootstrap intervals.

7.2 Utility test: from 500 held-out lowering states, optimize a short open-loop correction (palm twist and finger
commands, 8–16 steps) with K gradient steps from each model; execute in C MuJoCo (noise-averaged); measure tilt at
touchdown, touchdown speed, object at rest upright; baselines: no correction, CEM/MPPI at equal reference-simulator
rollouts.

7.3 Go criteria (fixed before running), task σ, h ≤ 4, held-out sizes: in contact, median cosine ≥ 0.7 on the palm
channel and on the finger channel (in-band; saturated states gated too per A1), magnitude ratio 0.5–2, sign agreement
≥ 80%; pre-contact (reachable strata) cosine ≥ 0.5, unreachable states judged by the utility test; S1's corrections
improve the placement objective over no correction and at least match CEM at equal real-simulator cost; S1 beats
softened MJX (if available) on the utility test or in-contact cosine.

### 8. Schedule

| Phase | Work | Effort | Gate |
| --- | --- | --- | --- |
| 0 | Held states, determinism, throughput, saturation census, MJX check, Warp parity | 3–4 d | G0 |
| 1 | Episodes with finger + palm OU, reference gradients, collection | 3–4 d + CPU time | G1 |
| 2 | Structured model, tests | 6–8 d (parallel with 1) | — |
| 3 | S0/S1 staged fit, B0/B1 training | 3–4 d | G2 |
| 4 | Benchmark, utility test, MJX baseline, report | 3–4 d | G3 |
| 5 | P policy test (GPU) | after G3 and cylinders in the training environment | G4 |

### 9. Phase 5: policy test (only after a go)

Finger commands plus palm motion through a differentiable palm-target interface: C v2's arm-reference channel, with
the palm trajectory as SONIC's tracked response (first approximation: perfect tracking with a measured lag; see A4).
Hybrid update ∇θ = ∇SAPG + λ·∇model, horizon ≤ 8 steps, critic beyond; λ gated by ensemble disagreement and live
finite-difference spot-checks in MuJoCo Warp; element-wise clip 0.1; equal GPU-hours vs SAPG alone, judged in Warp.

### 10. Answers to the review's questions (short)

C MuJoCo for clean targets, Warp parity checks noise-averaged agreement, the policy test is judged in Warp; gradient of
the expected outcome under σ over ≤ 8 steps; normal law mirrors MuJoCo's impedance (A5); LuGre/Dahl fitted to stick→slip
timing; backward-only tail graded by the utility test; dynamic fingers; variant (b) for transfer claims; utility test
added; upright placing with r ≤ 3.5 cm, decided mostly by palm trajectory and release timing; short-horizon hybrid with
gating; held-state generator plus OU perturbations; softened MJX if supported, else the utility test vs CEM.

### 11. Risks

| Risk | Mitigation |
| --- | --- |
| Finger gradients ~0 in saturated holds | Soft backward servo with s ≈ σ (A1), palm channel, saturation census |
| Weld release in the held-state generator injects a transient | Settle 0.5 s; keep only states that hold under a palm shake |
| Noisy finite-difference targets | Common random numbers, more draws, G1 self-consistency |
| MJX lacks cylinder pairs | Convex-mesh cylinder in all candidates, or drop the baseline and rely on the utility test vs CEM |
| C MuJoCo ≠ Warp | Phase 0.6 parity; final judgment in Warp |
| Effort competes with carry and placement | Phases 0–4 CPU-only; GPU only for 0.6 and Phase 5 |
