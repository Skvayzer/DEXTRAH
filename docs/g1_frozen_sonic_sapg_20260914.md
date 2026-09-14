# Frozen pretrained SONIC + BPS128/touch SAPG restart

Requested 2026-09-14 after videos showed frequent post-lift body falls.

## Provenance and scope

Run 555 was gracefully stopped at epoch 8808 / 1,305,083,904 cumulative
whole-body transitions. Its complete checkpoint remains at
`outputs/0_sonic_sapg_touch_555/nn/complete_1305083904.pth`.
Neither that decoder nor the route-2 bootstrap 503 is loaded into this restart.

Original SONIC `last.pt` SHA-256:
`62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4`.
The encoder, FSQ quantizer, decoder and all other pretrained tensors remain
frozen, in evaluation mode, outside the SAPG optimizer. All state tensors and
buffers are checked bit-for-bit after each reported training update. The
external SONIC checkpoint is pinned in the saved run contract.

Manipulation initialization is the completed 17,364,025,344-frame touch-383
SAPG checkpoint, SHA-256:
`ed9b26a7ba335c25f15df6623c8e0a97f6d5b3285721ae7f8263f82d6227d2cd`.
Copy its recurrent task features, all six exploration embeddings, six finger
means/log standard deviations, normalizers, and asymmetric critic. New critic
body-observation columns are zero initialized as in the previous transfer.

**This is not an exact arm-policy transfer or an optimizer resume from 555.**
The previous 7-arm/6-finger teacher cannot directly initialize a 64-latent body
head. The new adaptor's final layer starts at zero, so its deterministic initial
body output equals original SONIC. It learns through SAPG, not a claimed
successful offline arm imitation. Earlier per-state latent fits did not meet
arm/other-body accuracy thresholds; their rejected labels are not reused.

## Action and training contract

`task observations -> copied SAPG LSTM/features + body history/reference`
`-> trainable latent adaptor (64) + copied/trainable finger head (6)`
`-> source meta-action delay -> frozen SONIC -> 29 body PD targets`

- Adaptor: MLP (512 SAPG features + 994 body/reference inputs), hidden widths
  512/256/128, SiLU, linear 64-D output. No modification inside SONIC's layers.
- Residual is scaled by 0.1 and injected **before** upstream FSQ quantization.
- SAPG likelihoods, importance ratios and exploration are in the 70-D
  latent/finger action space. There is no Gaussian joint noise after decoding.
- Latents are unbounded; source finger execution is clipped to [-1,1]. The
  source bounds loss applies to finger means only, not to latent coordinates.
- Initial latent Gaussian standard deviation is 1.0 for the leader, adjusted
  by relative source group scales; this is a design choice, not a claim of
  matching GRAIL's exploration hyperparameters. Effective pre-FSQ scale is 0.1.
- Source delay is applied to latent/finger meta-actions **before** decoding.
  SONIC therefore always receives current feedback, not delayed body targets.
  Finger delay, EMA and follower-joint arithmetic remain source-identical.
- Fresh actor/critic optimizer states. Reused task/finger parameters keep
  source LR 1e-5; new adaptor LR 3e-4; critic LR 1e-4. LR groups and both
  optimizers are saved/restored for subsequent same-architecture continuation.
- Same 1200-object bank, four orientation keypoints, BPS128, touch at 70 Hz,
  60 Hz task policy, 120 Hz physics, manipulation reward/goal logic and DR.
- Full floating body physics; all 29 body joints remain controlled. Right
  fingers only; left fingers neutral. Self-collision remains off. No fabrics,
  PCA, torque observations, new balance reward or online imitation loss.
- Continuous mode removes frame/epoch caps; no periodic evaluation resets.
  Source best-return selection, rolling latest and checkpointed-stop saves.

This adopts the **frozen latent-adaptor architecture** of GRAIL manipulation,
not its entire experiment. GRAIL also used motion references and motion-tracking
rewards; this task keeps the original SAPG reposing objective and a standing
reference. Freezing prevents pretrained weight forgetting; it does not prove
the adaptor cannot cause a fall or that the old arm skill is latent-reachable.

## Validation / launch

`tests/test_frozen_sonic_sapg.py` checks all six teacher distributions/recurrent
states, critic preservation, latent policy gradients, optimizer groups, bounds,
all-tensor freezing, rejection of the old bootstrap, and zero-residual
equivalence / residual gradients through the real pinned SONIC checkpoint.
`tests/test_sonic_source_actions.py` checks exact finger processing parity and
meta-action delay with current body feedback.

The Slurm recipe enables this mode with `FROZEN_PRETRAINED_SONIC=1`; no
`STUDENT_CHECKPOINT` is needed or accepted by the Python entry point. Future
resumes require the same architecture, SONIC hash, teacher, object bank and
action contract. Live validation and launch results will be added below.

Reference: [GRAIL §3.3 / Appendix B.1](https://arxiv.org/html/2606.05160v1).

### First preflight and startup correction

Job 556 passed all 20 initial tests, including real CUDA SONIC inference and
gradient/finger parity, but stopped before optimizer updates: Isaac's USD
availability check raised `Cannot run the event loop while another loop is
running`. Its captured W&B background stack entered Kit's globally patched
`asyncio.run`, which calls `asyncio._ov_loop.stop_if_running()` even from the
background thread. Start W&B before AppLauncher patches asyncio, then attach
the resolved task configuration later. Logging remains online throughout.
An ordering regression test guards this startup change. No physics/controller
settings or source asset checks were bypassed.

### Successful live smoke: 557

Source `ad3bad7`; all 21 preflight tests passed. On one RTX 6000 Ada, the
120-environment smoke completed 16 real SAPG updates / 30,720 transitions.
All **55 pretrained SONIC state tensors remained bitwise identical** after
every update; trainable SONIC parameter count was zero. Adaptor final-layer
weight change L2 was 0.30513; finger-head change L2 was 0.00272. No robot falls
or numerical-failure resets occurred. Mean pelvis height at the last update
was 0.75877 m. Touch publication/acquisition averaged 69.84375 Hz over the short
4.267-second simulation window (scheduled 70 Hz, finite-window rounding).

This is a short startup/gradient/balance check, **not learned grasping or goal
success**. The last instantaneous fingertip forces were zero. Maximum observed
joint speed was 351.66 rad/s; this includes all joints and is not a claim of
hardware-safe motion.

Full continuation uses `outputs/0_frozen_sonic_smoke_557/nn/complete_30720.pth`
with 9,216 environments / six groups of 1,536, one RTX 6000 Ada, 16 allocated
CPU threads (Kit/TBB pools 8), and a 48-hour Slurm allocation. Frame/epoch caps
are disabled; no periodic evaluations. Preserve the 70-action actor and both
optimizer states; initialize fresh physics episodes when changing batch size.
Live full-size results follow once optimizer updates are verified.

### Full-size 558 and per-update garbage collection

558 restored both optimizers from 557 and completed through epoch 300 /
41,908,224 transitions. All 55 SONIC tensors still matched the original;
adaptor weight-change L2 reached 1.31349. There were 93 robot-fall events and
29 finite numerical-failure resets (0.6925 per million new transitions), not
a demonstration of grasping success. The final instantaneous mean pelvis
height was 0.75621 m and touch publication rate 69.9956 Hz.

It failed with CUDA OOM during SAPG's observation-batch concatenation at
epoch 301. Live PyTorch memory had risen to 32.18 GiB; device memory was
46.99 GiB. The every-16-update full GC had reclaimed 103.07 GiB cumulatively,
including 15.48 GiB at epoch 273 and 5.04 GiB at epoch 289, returning live
allocation to about 3.70 GiB. The interval allowed garbage to accumulate
again before the next collection. This is evidence of collectible tensor
retention; the precise upstream cycle owner has not been isolated.

Resume the rolling complete checkpoint from 558 with
`BOUNDARY_GC_INTERVAL=1`: collect only between complete optimizer updates,
never empty the CUDA cache, change batches, reset training environments, or
detach live tensors. Keep 9,216 environments and all learning settings. This
adds CPU GC overhead in exchange for bounding between-update accumulation.
