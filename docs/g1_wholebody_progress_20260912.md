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

## Continuation: reuse the trained SAPG teacher before full-body RL

The user explicitly requires teacher-guided initialization, not grasping RL
from random adapter weights. Early SAPG-to-adapter behavior transfer is distinct
from the optional later perceptual/deployment distillation of a working composite
controller. We load `sonic_manipulation_base`, **not** a pretrained GRAIL pickup
adapter; our continuous Revo2 adapter is new and has no trained checkpoint yet.

Implemented and committed:

- `kinematics.py`, `teacher_bridge.py`, `align_g1_teacher_motion.py`: reconstruct
  the source wrist from the real URDF, verify it against simulator traces, apply
  one rigid torso/scene alignment to preserve hand-object transforms, retime to
  50 Hz and reject moving-source-torso/incorrect-frame data. Ten future frames
  are spaced **0.1 seconds**, not 0.02 seconds. Pose/velocity endpoints are held
  consistently. This same-robot conversion is **not** IK or dynamics validation.
- Two baseline-teacher hammer segments exported locally and remotely (CPU job
  **452**): 22.42 s / 2 hits and 32.08 s / 8 hits, no body-joint limit violations,
  maximum source wrist position discrepancy 0.01245 mm. Remote artifacts:
  `outputs/teacher_hammer_aligned_20260912`.
- `transfer.py`: recurrent student adapter, separate latent/finger heads,
  masked imitation losses, and Adam fitting of latent labels against decoded
  seven-joint physical arm targets. Other 22 body commands retain a baseline
  penalty and acceptance tolerance. Failed latent fits cannot supervise the
  body; valid six-channel finger labels remain usable. No 13-vector versus
  70-vector loss, no direct old-optimizer resume, no PhysX gradients.
- `FrozenSonic.decode_for_imitation`: frozen parameters with the released FSQ
  straight-through gradient. CPU **446**: ten tests passed, exact forward match
  with normal inference, nonzero residual gradients and no SONIC weight gradients.
- CPU **461** exercised actual SONIC target fitting (synthetic, not SAPG skill
  transfer): arm max error fell from 0.2380 to 0.0110 rad, but other body commands
  changed up to 0.0727 rad; **both labels were rejected** by the 0.05-rad retention
  gate. This confirms fitting and rejection behavior, not trained manipulation.
- Opt-in exact teacher-policy capture now records pre-action observations,
  original normalization, recurrent states, raw/clipped teacher actions,
  post-step applied joint targets and reset validity. Motion-only old captures
  contain **previous** applied targets and are not mislabeled as next actions.
  `teacher_data.py` checks causal alignment and action order. Capture **453**
  caught CPU termination flags indexed by CUDA IDs; fixed and retried as **459**.
- Added a teacher-reference tracking-only probe with real full-body state and
  wrist-error capture. It has no objects and neutral fingers; not yet executed
  at this entry, and never labeled a grasping validation.

### Hand-physics root cause isolated

The original compliant coupling failed (~1.2–1.7 rad). Rigid coupling reduced
error to 0.147 rad with self-collision on; disabling self-collision for diagnosis
reduced it to **0.001631 rad**, with standing and hand motion passing.
Solver iterations, timestep, artificial motor inertia and restoring passive
velocity caps did not provide an acceptable full-physics solution; some variants
became unstable. None of those failed diagnostic variants was made the default.
Job **450** was cancelled when its final compliant/zero-velocity-iteration case
stalled in simulation reset; earlier case artifacts remain. Diagnostic children
now have a three-minute timeout. Sweep completion alone is not a physics pass.

All-body contact measurements (**451**) identified palm/wrist versus proximal
thumb impulses of **548 N left / 534 N right**. Distal sensors were zero because
the contact was at the thumb base, not the fingertip. Merely measuring fingertips
would have missed the source of the linkage instability.

CPU CAD audits **456/457**, with original asset hashes and measured joint poses:

- Original triangle meshes: no palm/proximal-thumb intersection at any of six
  checked poses.
- Convex palm + original thumb: false intersection at **all six** poses.
- Original palm + convex thumb: no intersection at all six.
- Contact locations are near the proximal thumb bearing (~-5 to 0 mm in its
  longitudinal local coordinate), not at the distal tactile pad.

Thus the palm's single convex hull fills the thumb clearance. Narrowly filtering
just those two pairs (**454**) reproduces the 0.001631-rad pass while all other
self-contact remains enabled. This is a **diagnostic**, not a permanent filter.
`decompose_palm_colliders` instead changes only the two palm collision
approximations to preserve the concavity; no pair exclusions, no removed branches,
no mass/joint/visual/tip changes. First attempt **458** exposed instancing at the
whole wrist collision container; corrected and retried as **460**. Its physical
result is pending at this entry. CAD audits are in
`outputs/thumb_cad_audit_20260912.json` and `outputs/thumb_cad_audit_mixed_20260912.json`.

Dependencies: added `python-fcl==0.7.0.11` **only** to the isolated SONIC overlay
for CPU collision auditing (job 455). Original training environment unchanged.

### Resources and teacher comparison

The user's workstation quickstart explicitly allows **one GPU job per user**.
All GPU work is serialized by Slurm. Evaluation **440** was cancelled after its
three completed baseline training seeds to free the GPU for physics diagnosis.
The completed results and immutable runner snapshot remain intact, with resume
supported by their existing checksums/protocol; the 27-case selection is not
finished and no tactile-versus-BPS winner has been declared.

Remaining: verify repaired palm geometry and loaded hand mechanics, finish
teacher data/selection, validate reference tracking, collect paired real
full-body states/teacher labels, run supervised adapter initialization, integrate
the actual BPS/tactile full-body task and SAPG optimizer, benchmark and only then
fine-tune. **No full-body SAPG training or completed skill distillation yet.**
