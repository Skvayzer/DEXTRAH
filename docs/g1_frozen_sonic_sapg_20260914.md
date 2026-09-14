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
