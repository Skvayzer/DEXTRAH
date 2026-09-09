# Uninterrupted BPS-128 + Revo2 tactile continuation

Requested 2026-09-09: stop the BPS-only control, continue the tactile policy
longer, remove all periodic evaluations/restarts, and use 70 Hz tactile sensing.
This supersedes the bounded pilot launcher for the new experiment only.

- Parent: tactile pilot 381, `nn/complete_500170752.pth`, SHA-256
  `bbb38540baa8a5b1e47279a405548f8b92ff0df795dc427750471cf3a6a9e533`.
- Physics 120 Hz, tactile acquisition **70 Hz**, tactile delivery **70 Hz**, policy
  60 Hz. Acquisitions are quantized to 120 Hz physics ticks (one or two ticks
  apart), not fictitious interpolated force samples. Policy consumes the latest
  available sample. This is a nominal model, not newly measured hardware timing.
- All learned actor/critic parameters, both Adam optimizers, and input/value
  normalization are inherited. Original 10 Hz calibration is retained as
  provenance, not presented as a new 70 Hz calibration. Sample age remains in
  fixed 100 ms numerical units to preserve the checkpoint's learned interface.
- Only the initial resume reconstructs physics and clears recurrent state.
  Fresh rolling episode meters and a fresh best-reward baseline are used in the
  new run; learned normalization is not cleared.
- Same 24,576 environments, 1,200 objects, six SAPG groups, actions, rewards,
  tolerance, domain randomization and solver. No fabrics, PCA, arm torque inputs,
  added compliance, or reward changes.
- `--continuous` removes both frame and epoch limits and the reward-based win
  stop. There is **no evaluation subprocess and no 100M/500M training stop**.
- One Ada allocation lasts up to 48 hours. Slurm sends USR1 three minutes before
  the wall limit; the trainer saves and exits at an optimizer-update boundary.
  It does not automatically submit another job. Crashes remain possible.
- `nn/latest.pth`: first update and every 64 updates (~25.2M transitions).
  `nn/snapshot_<frame>.pth`: every 1,024 updates (~402.7M transitions).
  `best/model.pth` points to `nn/g1_sapg_bps128_touch.pth`, selected by upstream
  SAPG's leader rolling mean **training episode reward**, not evaluated success.
  All checkpoint writes use temporary files and atomic replacement, and include
  the central critic's optimizer. Saving does not recreate environments.
- W&B logs configured and observed acquisition/publication rates, mean/max
  sample age, force summaries, validity and normal training metrics.
  `run_transitions` counts samples since this resume; `additional_transitions`
  includes the prior tactile pilot; `cumulative_transitions` includes the 8B BPS
  parent. The rate change is recorded in `resume_provenance.json`.

Launch: `sbatch scripts/slurm/train_g1_touch_continuous.sbatch` from the isolated
remote worktree after tests. The earlier 10 Hz run is retained unchanged; this
run has a separate W&B identity. A 70 Hz continuation is not a matched-budget
comparison against the stopped 10 Hz-era control.
