# DEXTRAH-G1-WholeBody: working rules

## GPU use: ONE GPU, used AS MUCH AS POSSIBLE
- Exactly one GPU job at a time (running or queued); the cluster is shared.
  All GPU work goes through Slurm (`--partition=gpu`, 1x RTX 6000 Ada, 48 GB).
- That one GPU must be used to the full: size every training run to fill the
  48 GB of VRAM (we used to run 48k+ envs). Running 9k envs at ~25 GB is
  wasted capacity. Before a long run, measure memory at increasing
  `NUM_ENVS` (multiple of 6 for SAPG) and pick the largest that keeps
  ~2-4 GB headroom; report the chosen count and the measured VRAM.
- On mjlab, keep the MuJoCo Warp buffers at measured need (per world: <= 49
  contacts, <= 150 constraint rows), not oversized defaults, so the memory
  goes to environments.

## Training and evaluation
- W&B logging for every training run (project skvayzer/adept).
- Minimal checkpoints: one best + last resumable state + essential logs.
- Reuse trained weights; never restart from scratch without a reason.
- Best checkpoints must really be saved: after an MDP change resume with
  `RESET_BEST=1` and verify the best file updates once tracking starts.
- Never silently change rewards, goals, object bank, action semantics, rates,
  spawn rules or eval thresholds; changes are explicit, opt-in and recorded
  in the task contract.
- Don't stop or restart training unless asked. Keep evals quick and small.
- Hand-to-surface observation (`--hand-surface-obs`, 17 hand points -> nearest object
  surface, `dextrah_lab/wholebody/hand_surface.py`) is ON for all mjlab runs from
  2026-10-21 (sbatch default `HAND_SURFACE_OBS=1`; opt out with `HAND_SURFACE_OBS=0`).
- Conservative claims.

## Git
- Commit commit-by-commit and push right after each commit:
  `git push origin HEAD:feature/g1-wholebody-sapg` (fork github.com/Skvayzer/DEXTRAH).

## Backends
- Isaac Lab 2.2.1 / Isaac Sim 5.0 (PhysX): default, `BACKEND=isaac`.
- mjlab (MuJoCo Warp, `dextrah_lab/mj`, venv `/data1/users/konstantin.smirnov/venvs/mjlab`):
  `BACKEND=mjlab`; status in `docs/mjlab_port_plan_20261013.md`.
