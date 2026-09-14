# Workstation + Slurm: handoff for another Codex session

Verified on **2026-09-14, approximately 16:57–17:00 Dubai time (UTC+4)**.
Recheck live state; job IDs, branches and allocations below are not permanent.
This guide is operational context, **not permission to stop training or reboot**.

## 1. Start here: connect and inspect, without changing anything

Run on the **Mac/laptop**:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 konstantin.smirnov@tl-server-0
```

Tailscale must be connected. This working alias reaches hostname `ws013118`.
The Downloads quickstart uses `tl-server-1` in generic examples; do not replace
the verified `tl-server-0` alias without checking the user's SSH/Tailscale setup.
If batch authentication fails, diagnose SSH/Tailscale; do not disable host-key
verification, expose private keys, or request credentials in chat.

Run on the **workstation's SSH shell**:

```bash
hostname
date -Is
squeue -u konstantin.smirnov -o '%.12i %.24j %.10T %.12M %.12l %R'
sinfo -o '%P %a %l %D %G'
```

Before acting, read applicable `AGENTS.md`, this guide, the project's latest
handoff, and the active run's `task_contract.json` and `warmstart_validation.json`.
Prefer `rg`; it is installed locally but was unavailable in the remote shell.
Use `grep`/`find` there rather than installing packages just to search.

## 2. Resource rules: SSH is not a compute allocation

The lab quickstart is at
`/Users/konstantinsmirnov/Downloads/USER_QUICKSTART 1.pdf`.
Its restrictions agree with the inspected partitions and normal QoS:

| Resource | Current limit / setting |
|---|---|
| Training GPU request | `--gres=gpu:rtx_6000_ada:1` |
| Compute GPUs on node | 3 shared RTX 6000 Ada GPUs |
| Per job | At most 1 GPU |
| Per user | At most 1 allocated compute GPU across jobs |
| Interactive partition | At most 8 hours |
| Batch partition | At most 7 days; default 1 day if not overridden |
| Current whole-body recipe | 1 GPU, 16 CPUs, 96G system RAM, 48 hours |

`normal` QoS reported `MaxTRESPJ=gres/gpu=1` and
`MaxTRESPU=gres/gpu=1`; this limits concurrent GPU jobs even when another GPU
looks idle. CPU-only jobs may run separately if the scheduler has resources.
The additional `display_gpu:1` resource is not another training allocation.

Use the SSH shell for code inspection/editing, Git, file transfer, small log
reads, scheduler commands, and lightweight status queries. **Training, Isaac
simulation, GPU diagnostics, rendering and substantial CPU computation/tests
must run through Slurm** (or an allocated Open OnDemand session).

Do not bypass scheduling with `nohup`, `tmux`, a second SSH connection, or a
manually selected GPU. Do not change global drivers, power settings or packages.
Never overwrite `HOME`, `home`, or `CODEX_HOME` to hold a project path.

Recheck limits when necessary:

```bash
scontrol show partition batch
scontrol show partition interactive
sacctmgr -n -P show qos normal format=Name,MaxTRESPJ,MaxJobsPU,MaxTRESPU
```

## 3. Project paths and Python environments

`/home/konstantin.smirnov/data1` is a symlink to the fast workspace
`/data1/users/konstantin.smirnov`. Prefer the explicit `/data1/users/...` paths.
Archive storage is `/data2/users/konstantin.smirnov`; do not assume backups exist.

| Purpose | Workstation path |
|---|---|
| Whole-body SAPG integration | `/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody` |
| Original Play2Perfect task / SAPG | `/data1/users/konstantin.smirnov/play2perfect` |
| Completed BPS + tactile experiments | `/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH` |
| GRAIL source | `/data1/users/konstantin.smirnov/GRAIL` |
| Original SONIC bundle | `/data1/users/konstantin.smirnov/G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base` |
| Navigation project | `/data1/users/konstantin.smirnov/Click-and-Traverse` |
| Whole-body Python | `/data1/users/konstantin.smirnov/venvs/g1_sonic/bin/python` |
| Base Isaac/P2P Python | `/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python` |

The whole-body virtual environment's Python executable links to the conda
interpreter, but its environment/packages are still selected by invoking the
**venv path**. Do not replace it with system Python or casually reinstall torch.
For another project, inspect its own launcher rather than assuming this venv.

Local whole-body checkout:
`/Users/konstantinsmirnov/robotics/DEXTRAH-G1-WholeBody`.
Current branch: `feature/g1-wholebody-sapg`.
Remote checkout is a Git worktree whose common Git directory is under
`DEXTRAH-ADEPT/.git`; do not delete or reset that shared repository.

## 4. Interactive and CPU-only work

From the workstation SSH shell, request an interactive GPU shell:

```bash
srun --partition=interactive --qos=normal --time=00:30:00 \
  --cpus-per-task=4 --mem=16G --gres=gpu:rtx_6000_ada:1 --pty bash
```

It may wait while the user's training owns the permitted GPU allocation. Do
not cancel that training to accelerate this request without authorization.
Only once the allocation starts, check:

```bash
echo "$SLURM_JOB_ID"
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/data1/users/konstantin.smirnov/venvs/g1_sonic/bin/python -c \
  'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Use `cuda:0` **inside a one-GPU allocation**. Slurm may remap an allocated
physical GPU to logical index zero; identify hardware using UUID/PCI bus,
not the index alone. Leave Slurm's `CUDA_VISIBLE_DEVICES` untouched.
[Slurm GPU environment documentation](https://slurm.schedmd.com/gres.html).

Example allocated CPU-only test, with no GPU request:

```bash
srun --partition=batch --qos=normal --time=00:10:00 \
  --cpus-per-task=4 --mem=16G \
  --chdir=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody \
  /data1/users/konstantin.smirnov/venvs/g1_sonic/bin/python \
  -m pytest -q tests/test_wholebody_contract.py
```

Exit the allocated shell when finished. For unattended work use `sbatch`.
For agent tool calls, retain the returned session ID when `srun` is still
running; poll that session instead of accidentally submitting a duplicate.
[Slurm srun reference](https://slurm.schedmd.com/srun.html).

## 5. Minimal batch recipe

Prefer an existing project's `scripts/slurm/*.sbatch` recipe. For a new GPU
smoke test, create `gpu_smoke.sbatch` with `apply_patch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gpu_smoke
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
exec /data1/users/konstantin.smirnov/venvs/g1_sonic/bin/python -c \
  'import torch; x=torch.randn(512,512,device="cuda:0"); print(torch.cuda.get_device_name(0)); print((x@x).norm().item())'
```

Submit deliberately from a known working directory:

```bash
bash -n gpu_smoke.sbatch
sbatch --parsable gpu_smoke.sbatch
```

Record the returned job ID. `sbatch` acceptance is not proof that Python is
running. `--mem` requests **system RAM**, not GPU VRAM. In `#SBATCH` lines,
shell variables are not expanded; use literal paths or command-line options.
Output directories must already exist. [Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html).

## 6. Current whole-body run: preserve it unless asked otherwise

At the verification time, **job 560 was RUNNING**. This is a dated snapshot,
not an instruction to manipulate job 560 in a future session.

- Project: `DEXTRAH-G1-WholeBody`.
- Log: `slurm-sonic-sapg-560.out`.
- Run: `outputs/0_frozen_sonic_sapg_touch_560`.
- [W&B](https://wandb.ai/skvayzer/adept/runs/unique_id_0_frozen_sonic_sapg_touch_560).
- 9,216 environments, six SAPG groups, one RTX 6000 Ada.
- Original SONIC encoder/FSQ/decoder frozen; task/finger network and latent
  adaptor train. BPS128, touch 70 Hz, policy 60 Hz, physics 120 Hz.
- 48-hour allocation; no frame/epoch cap or periodic evaluations.
- Important settings: `FROZEN_PRETRAINED_SONIC=1`,
  `BOUNDARY_GC_INTERVAL=1`, `SIM_WORKER_THREADS=8`.
- Recipe: `scripts/slurm/train_g1_sonic_sapg.sbatch`.
- Detailed implementation/lineage: `docs/g1_frozen_sonic_sapg_20260914.md`.

This is **not** the earlier trainable-decoder experiment. Do not load bootstrap
503 or a 35-action checkpoint into the 70-action frozen-controller architecture.
Current checkpoints contain actor + critic + both optimizers; original SONIC
is external and hash-pinned. Reuse the checked continuation path, not ad-hoc
partial state loading.

The existing batch script snapshots committed source into
`outputs/sonic_sapg_source.*` **when the job starts**, not when submitted.
Uncommitted edits do not enter that snapshot. A queued job may therefore pick
up newer HEAD than at submission; inspect its saved `source_commit`. If exact
submission-time pinning matters, prepare an immutable checkout/commit-pinned
recipe before queuing. Never edit an active run's source snapshot.

### Deliberate continuation command — only after user authorization

Inspect the selected parent checkpoint and contract first. For the current
architecture, use a fresh output prefix and the existing recipe. This example
is intentionally blocked until an exact compatible checkpoint is selected:

```bash
cd /data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
resume_checkpoint='/REPLACE_WITH_VERIFIED_FROZEN_SONIC_CHECKPOINT.pth'
test -f "$resume_checkpoint" || { echo 'Select a verified checkpoint first'; exit 1; }
unset STUDENT_CHECKPOINT MEMORY_GC_PROBE_EPOCH DIAGNOSTIC_CAPTURE_SECONDS
sbatch --time=2-00:00:00 --job-name=g1_frozen_sapg \
  --export="ALL,WHOLEBODY_REPO=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody,FROZEN_PRETRAINED_SONIC=1,RUN_PREFIX=0_frozen_sonic_sapg_touch,RESUME_CHECKPOINT=$resume_checkpoint,NUM_ENVS=9216,CONTINUOUS=1,MEMORY_TRACE=1,BOUNDARY_GC_INTERVAL=1,SIM_WORKER_THREADS=8" \
  scripts/slurm/train_g1_sonic_sapg.sbatch
```

Do not blindly copy an old `SubmitLine`: it may point to an older checkpoint
or carry stale environment variables. Changing environment count still
requires memory validation. Do not start from scratch unless explicitly wanted.

## 7. Monitor: distinguish queued, initializing, training and failed

Replace `JOB_ID` with the verified numeric ID. These are read-only:

```bash
squeue -j JOB_ID -o '%.12i %.24j %.12T %.12M %.12l %R'
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS
sstat -j JOB_ID.batch --format=JobID,AveCPU,MaxRSS
```

`PENDING` commonly means resources, priority or the per-user GPU limit.
`RUNNING` includes asset loading, not just optimizer updates. If a job vanishes
from `squeue`, consult `sacct`; it may have completed, failed, timed out or been
cancelled. Check `.batch` and the application's result file, not only a parent
row. GPU OOM can appear as generic `FAILED` with a Python exception; CPU-memory
OOM may have a different accounting state.
[squeue reference](https://slurm.schedmd.com/squeue.html),
[sacct reference](https://slurm.schedmd.com/sacct.html).

For the dated current run, on the workstation:

```bash
cd /data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
tail -n 60 slurm-sonic-sapg-560.out
```

Inspect these files within the selected output directory:

| File | What it establishes |
|---|---|
| `task_contract.json` | Actual task, controller mode, hashes, resource-related settings |
| `warmstart_validation.json` | Checkpoint lineage and optimizer restoration |
| `progress.json` | Advancing epoch/frame, freezing audit, touch and memory telemetry |
| `training_result.json` | Completion, checkpointed stop, or full failure traceback |
| `wandb.json` | Correct run URL / identity |
| `memory_trace.jsonl` | Allocation growth, stage of failure, retries/OOM counts |
| `boundary_gc.jsonl` | Memory actually reclaimed between updates |
| `nn/latest.pth` | Rolling complete continuation state |

Read-only GPU status is useful, but a single utilization sample is not average
throughput. Confirm training with advancing counters **and** optimizer/loss
logs. Report the exact job, run URL, environment count, GPU identity, checkpoint,
source commit and remaining uncertainty. Poll at sensible intervals; do not
leave the user without progress updates while waiting.

## 8. Checkpointed stopping: project-specific, not generic scancel magic

Only do this when the user authorized stopping the exact job. First inspect
its ownership, command and launcher. The current SONIC batch launcher traps
USR1 and forwards it to Python; Python saves at a completed update boundary.

```bash
scontrol show job JOB_ID
# Only for this verified signal-aware training recipe:
scancel --signal=USR1 --batch JOB_ID
```

Then check the log for `CHECKPOINTED_STOP_REQUESTED`, the updated complete
checkpoint, `training_result.json` with `checkpointed_stop`, and job completion.
Do not immediately follow the signal with a hard cancellation.

USR1 is **not** universally safe: a program without a handler may terminate.
For an authorized non-checkpointable job, `scancel JOB_ID` cancels it, but may
lose unsaved work. Never use blanket `scancel -u`, `pkill python`, GPU reset or
reboot to free resources. Before resuming, verify the checkpoint exists and
matches the intended architecture. [Slurm signal semantics](https://slurm.schedmd.com/scancel.html).

## 9. Project-specific pitfalls already encountered

- **W&B + Isaac startup:** initialize W&B before `AppLauncher` starts Kit;
  afterwards attach the resolved config. Kit globally patches asyncio, and a
  later W&B background-loop startup caused a real asset-loading failure.
  Do not silently disable online logging as a workaround.
- **Startup time:** full robot/object scene creation takes several minutes.
  Inspect `stacks.log` and scene progress before declaring a hang. Existing
  launcher uses Kit/TBB worker counts of 8 within a 16-CPU allocation.
- **CUDA OOM:** first distinguish live tensors, reserved allocator memory and
  non-PyTorch PhysX memory. In this task, full GC every 16 updates was too sparse;
  every-update boundary GC bounds collectible batch retention. Do not revert
  it casually or claim its success guarantees indefinite memory stability.
- **CUDA not available over SSH:** test within an allocated GPU job before
  diagnosing a driver failure. `nvidia-smi` visibility alone does not grant
  compute access. Do not unset Slurm GPU restrictions or use another GPU ID.
- **Warnings:** headless GLFW/display and Warp driver-entry-point warnings
  have also occurred in working runs. Diagnose the traceback and actual test
  outcome, not an isolated warning line.
- **Checkpoint files:** rolling latest is atomically replaced, but live `scp`
  can race with that replacement. Prefer a completed immutable checkpoint;
  compare SHA-256 and repeat if the source changed. Do not read `.pending.pth`.
- **Secrecy:** never print `.netrc`, SSH keys, W&B tokens or the complete shell
  environment. Use existing authenticated configuration; no tokens in Git.

## 10. File transfer and Viser

Run transfers from the **laptop**. Substitute actual files; preserve originals:

```bash
scp konstantin.smirnov@tl-server-0:/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/RUN/recording.mp4 \
  /Users/konstantinsmirnov/Desktop/research_recordings/
```

Use `sha256sum FILE` remotely and `shasum -a 256 FILE` on macOS for verification.
GPU simulation/rendering for a viewer still needs a Slurm allocation. A viewer
does not justify stopping the active GPU training without asking. Check its
existing port and job first; do not start duplicate viewers.

For a viewer already listening on workstation loopback port 8080:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8080:127.0.0.1:8080 konstantin.smirnov@tl-server-0
```

Open `http://localhost:8080` on the laptop. Substitute the actual viewer port.
Prefer loopback binding; do not expose an unauthenticated Viser server publicly.
Closing this tunnel does not stop its Slurm job. If allocation nodes change in
future, forward to the actual allocated node, not an assumed login host.

## 11. Commit / deploy without disturbing worktrees or running jobs

Use `apply_patch` for source edits. Preserve unrelated edits and untracked
Slurm logs. Commit explicit files, not a blind `git add .`. Do not upgrade the
pinned P2P/GRAIL code or overwrite trained checkpoints to make imports work.

The established whole-body deployment path is:

```bash
# Laptop, after reviewing and committing the intended changes:
cd /Users/konstantinsmirnov/robotics/DEXTRAH-G1-WholeBody
git status --short
git remote -v
git push server HEAD:refs/heads/incoming/g1-wholebody-sapg
```

```bash
# Workstation: verify branch and status BEFORE merging.
cd /data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
git branch --show-current
git status --short
git merge --ff-only incoming/g1-wholebody-sapg
git push origin HEAD:feature/g1-wholebody-sapg
```

Stop and resolve unexpected divergence; never force-push or reset a worktree.
The local `server` remote currently points to the common `DEXTRAH-ADEPT` Git
repository, while the remote worktree's `origin` is
`https://github.com/Skvayzer/DEXTRAH.git`. Recheck rather than assuming a new
checkout has identical remotes. Committed changes do not update an already
running source snapshot.

## Copy-paste handoff instruction

> Read this workstation guide and the current project's handoff. SSH to
> konstantin.smirnov@tl-server-0, inspect Slurm and the active run before acting.
> Do not stop/restart any existing training unless I explicitly ask. Run heavy
> work only inside Slurm and use the correct project Python. Preserve original
> checkpoints and committed source provenance. A submitted/running Slurm job
> is not enough: verify actual optimizer updates and W&B logging before
> reporting that training has started.
