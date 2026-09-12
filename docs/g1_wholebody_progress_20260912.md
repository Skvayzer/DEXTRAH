# Whole-body SAPG implementation — 2026-09-12

Status: implementation in progress. **No whole-body SAPG training, teacher
selection, validated manipulation, or optimizer-inclusive capacity
result yet.** The four-environment probe is a diagnostic, not a training size.

## Architecture written into the research proposal

- Frozen SONIC reference encoder, FSQ quantizer and 29-joint body decoder.
- Trainable SAPG adapter: 64 pre-quantization residuals, scale 0.1; six
  continuous right-hand commands initially (70 outputs), later twelve hand
  commands (76 outputs). SONIC owns both arms; no competing arm target writer.
- Full floating-base physics with feet, gravity, both hands and load transfer.
  Frozen network weights do not mean frozen robot dynamics.
- Existing 13-action SAPG policies provide evaluated demonstrations/reference
  motion, not directly loadable 70-action actor/optimizer weights.
- 200 Hz physics, 50 Hz body controller, 70 Hz tactile acquisition planned.
  No fabrics, PCA or arm-torque observations introduced.

## Verified

CPU Slurm job 427 (`outputs/sonic_contract_20260912_v2/validation.json`):

- Downloaded official pinned 469 MB controller bundle; SHA-256 matches.
- Strictly loaded all 25,870,714 actor parameters into released SONIC code.
- Zero latent residual produces **exactly** the unchanged controller output.
- Random nonzero residual changes body actions (max difference 0.54266 in
  this synthetic check); outputs finite, weights frozen even under train().
- Preserved joint order, term-wise ten-sample history, grouped future q/qd
  packing, row-major first-two-columns orientation representation.
- Source full robot mass 33.81575472 kg. No removed leg/arm branches;
  all source masses/collision shapes and ten native finger mimic relations
  retained. Nonzero locked tip angles folded into fixed transforms with FK tests.
- All 29 body joints' parent, child, origin and axis attributes exactly match
  the pinned SONIC model-12 URDF (read-only comparison against both real files).

Tests: 23 local CPU tests for assets/contracts/actuators/importer/reference preparation
and teacher selection/runner; 2 remote Torch history/config tests (job 428).
These are not substitutes for physics or manipulation validation.

Independent runtime probe 431, CUDA stage:

- Torch 2.7.0+cu128 and bundled Warp 1.7.1 both allocate correctly on GPU.
- Single cube passes reset and 200 GPU PhysX steps, settles at z=0.050000008 m.
- Isaac emits the `cuDeviceGetUuid` warning but this test still passes; do
  not identify that warning alone as a failed driver/runtime.
- Probe subsequently hung during shutdown. The workstation's IsaacLab stop
  callback waits for interactive play; our headless probes now clear callbacks
  before close. No global framework or driver files modified.

Full-body GPU probe **434 passed**, four environments, ten simulated seconds:

- Floating pelvis (not fixed), **52 rigid bodies**, all **51 physical joints**
  and ten native PhysX mimic constraints.
- Imported mass 33.81575394 kg, matching source within numerical precision.
- Minimum pelvis height 0.752748 m; no fall or large tilt.
- Final-second foot support force/weight 0.9861–1.0150 across environments.
- Frozen SONIC ran at 50 Hz over 200 Hz physics. No arm/leg branches removed,
  no gravity disabling and no fabrics/PCA.
- This is **standing only** with an unchanged nominal reference and neutral
  fingers. It does not establish reaching, loaded grasping or disturbance recovery.

## Implemented, not yet fully validated

- Floating-base full robot config with original SONIC body gains/scales,
  gravity/self-contact on, both hand actuation sets and passive coupled distals.
  Hand PD 1200/25 matches the reduced teacher, **not hardware calibration**.
  Native distal coupling differs from the teacher's separate distal drives.
- Small SONIC standing-reference probe: runtime mass/coupling/dynamic-body
  checks, foot forces, no hidden resets, saved state/target traces and failure
  artifacts. A constant nominal reference is only an interface probe, not a
  claim of matching SONIC's complete original motion distribution.
- Offline 60→50 Hz teacher-source preparation: never interpolates across a
  reset; smooth achieved q/qd separately from held commanded targets; SLERP
  actual poses, hold discontinuous goal labels. Explicitly marks exports as
  **not feasible full-body references**. Whole-body retargeting/IK still needed.

## Evaluation and debugging status

Reboot fixed installed/loaded NVIDIA mismatch (both now 580.178.04).
Evaluation retry 419 made no visible rollout progress in ~20 min and was
cancelled. Dependent 420 was cancelled, not left orphaned. The 27-case
comparison remains implemented but incomplete; no winner or causal tactile
benefit may be reported.

Full-body 429 exposed a relative USD export-directory error; corrected to
absolute paths. Later probes revealed that shutdown could conceal an earlier
exception or even exit with status zero. We now save failure.json and print
the original exception before Kit shutdown. **A Slurm COMPLETED status alone
does not count as a validation pass; validation.json and its gates are required.**

Probe 433 then revealed zero imported mimic constraints. The workstation's
IsaacLab passes `convert_mimic_joints_to_normal_joints` directly to the native
`parse_mimic` flag, contrary to the configuration field's apparent meaning.
We inspect that expression and choose the value that actually enables native
couplings; do not replace follower drives with independent position drives.
Probe 434 confirmed ten `PhysxMimicJointAPI` instances (not tendons).
Probe **436** ran 768 full-body environments: standing passed, all independent
fingers moved, approximately **19,840 environment-steps/s** and **1.767 GiB**
sampled total device usage. **Moving native coupling failed**, with up to
1.737 rad deviation from the URDF relation. These are physics/controller-only
resource measurements, not a validated manipulation batch or optimizer budget.

Probes 437–439 isolated the coupling problem at four environments. Explicit
100 Hz/critical-damping parameters and disabling physics replication did not
resolve it. Those parameters remain provisional, not hardware calibration.
Do not hide this failure by resetting/clamping follower joint positions or
claim the full robot ready to grasp. Native reaction-force/coupling behavior
must be corrected or replaced by an explicitly validated physical model.

Reduced-teacher smoke **435** strictly loaded the BPS actor and executed 120
finite steps in 12 environments. Its final report hit an empty-family division
by zero (the tiny probe did not include every object family); fixed with a test.
Full 27-case teacher comparison resubmitted as **440**, immutable source
`42bcfe8`, output `outputs/teacher_selection_20260912_retry`. This is evaluation,
not a training/W&B run; no winner yet. Each attempt has a 30-minute timeout.

## Remaining gates

1. Complete reduced-teacher startup smoke test and rerun teacher suite.
2. Extend passing mass/standing checks to moving native couplings, contact
   geometry and reaching with compatible references; record quantitative traces.
3. Validate object load transfer and ten-pad tactile contact handling.
4. Build feasible teacher-to-body reference alignment/IK and evaluate tracking.
5. Benchmark increasing counts (768, 1536, then larger while useful) including
   **actual SAPG optimizer/rollout storage**, reset/contact bursts and headroom.
6. Only then launch the right-hand full-body SAPG pilot, with W&B and snapshots.

Use the agreed GPU budget for training throughput; small debug probes are
temporary. More GPU usage does not replace valid action/observation/physics
contracts. Do not silently reduce physical robot scope to reach a target count.
