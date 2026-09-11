# Whole-body SAPG preparation — 2026-09-11

Status: **partial implementation; simulation blocked by workstation driver mismatch.**
No teacher selected, no full-body physics validated, no new training launched.

## Isolation and upstream pin

Feature branch: `feature/g1-wholebody-sapg` in separate local/remote
`DEXTRAH-G1-WholeBody` worktrees. Original training worktrees/checkpoints and
unrelated uncommitted visualization changes are untouched.

GRAIL cloned separately to `robotics/GRAIL` locally and
`/data1/users/konstantin.smirnov/GRAIL` remotely; inspected revision
`aa31d8242ac79b11545b9e3635f73014a227bdfc`. No upstream installers run.
GRAIL root license restricts use to research/evaluation; vendored SONIC has
separate code/weight notices. Do not assume one license covers all artifacts.

Candidate body controller: released `sonic_manipulation_base` bundle, dataset
`nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL`, dataset revision
`40e795761302e611c1e7e3a6caefdd010d56c199`. Model weights not downloaded/validated.
Published checkpoint SHA256:
`62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4`.
Config declares 29 body actions, 930 proprioception inputs, 2x32 latent,
50 Hz control and 200 Hz physics. GRAIL injects the scaled latent correction
before FSQ. The model's actual joint ordering/history/reference conventions
must be preserved; our old 13-action checkpoint cannot directly initialize it.

## Implemented preparation

- Predeclared teacher comparison: BPS baseline, tactile best-training-reward,
  tactile final; seeds 42/43/44; original training bank, new procedural geometry
  seed 914271, and double object-wrench amplitude stress. Total 27 evaluations.
- Evaluation capture supports the alternate geometry seed and wrench multiplier.
  Held-out validation regenerates the exact training bank, checks its manifest,
  and rejects collision-geometry overlap independently of mass/density.
  New geometry uses the same families, not unseen object categories.
- Selection helper requires the entire suite. Rank by training-bank goal
  throughput, with two-percentage-point success/lift regression guards and
  stress throughput >=95% of baseline. Held-out results do not choose the winner.
  These are engineering margins, not statistical significance tests.
- Optional full-policy-rate capture retains achieved joint positions/velocities
  and previous applied joint targets separately for future reference conversion.
  This does not yet construct a feasible full-body trajectory.

No evaluation outputs or claimed policy improvement were produced. The
multi-process campaign launcher, rendering-label update, reference bridge,
full-body task, controller load test and resource benchmark remain to be built.
Pure tests cover geometry leakage, restoration of the generator hook, complete
suite requirements, invalid metrics and selection guards. Simulator integration
has not been tested; do not treat CPU tests as proof of valid physics.

## Workstation blocker

Read-only diagnostics on September 11 found:

- `/proc/driver/nvidia/version`: loaded driver **580.173.02**.
- Installed `nvidia-driver-580`, `libnvidia-compute-580`, `libnvidia-gl-580`:
  **580.178.04**.
- `/var/log/dpkg.log`: NVIDIA packages upgraded at **06:42 server time**.
- `uptime -s`: last boot **2026-08-27 11:22:21**.
- `nvidia-smi`: `NVML_ERROR_LIB_RM_VERSION_MISMATCH` (return code 18).
- Slurm 414: PyTorch CUDA allocation passed. CUDA availability alone therefore
  does not detect this Isaac Sim startup failure.
- Slurm 415: Isaac Sim could not enumerate/create any Vulkan GPU devices,
  `GPU Foundation is not initialized`. Diagnostic cancelled after 38 seconds.
  Log: `/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/slurm-bps-reposing-415.out`.

No driver/system package changes or reboot performed. Administrator intervention
is needed to align the running kernel module and installed userspace libraries.
After repair, rerun an Isaac Sim startup/step smoke test before submitting the
evaluation suite. Do not schedule a full-body training job until physical
standing, reaching, load transfer, contact/timing and optimizer-memory gates pass.
