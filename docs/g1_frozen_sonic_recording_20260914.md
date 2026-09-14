# Recording the frozen-SONIC checkpoint without stopping training

The active request is diagnostic video, not a training restart or a new
evaluation schedule. Run 560 keeps its original process, immutable source
snapshot, optimizer state and 9,216 environments throughout recording.

`scripts/record_g1_sonic_concurrent.py` runs in a separate step of the existing
allocation, using `srun --jobid=560 --overlap --exact`, four CPUs and the same
allocated GPU. Slurm's [overlapping-step mechanism](https://slurm.schedmd.com/srun.html#OPT_overlap)
shares resources within that allocation; it does not reserve a second GPU or
bypass the user's GPU quota. Recording can temporarily reduce training speed.

The wrapper requires at least 12 GiB of free device memory before starting,
limits the recorder's PyTorch allocator to 4 GiB, and checks total device
headroom every 20 seconds. If free memory falls below 6 GiB, it terminates only
its own recorder/render process group. These are protective margins, not a
guarantee against instantaneous memory spikes. The 120-environment recorder
uses smaller PhysX buffers already supported by `source_task_config`.

The selected best-return checkpoint is copied through a read-only descriptor.
The copy rejects in-place modification, records a SHA-256, and is stored with
the source agent config and task contract. Training checkpoint files are never
modified or locked. There are no optimizer updates in the recording process.

The recorder now supports both the earlier 35-action trainable decoder and the
current 70-action frozen SONIC policy. For the latter it restores the entire
body configuration, strict-loads the trained actor and normalizers, and sends
**unclipped 64-dimensional latent residuals** plus six finger actions. Clamping
all 70 outputs to [-1, 1] would change the policy and is explicitly forbidden.
SONIC performs the same 0.1 pre-FSQ scaling and finger action execution as
training. All pretrained controller tensors are audited before and after.

The first assigned hammer, brush and spatula are preselected with seed 42,
before observing any outcomes. Each diagnostic covers 60 seconds at 60 Hz
control / 120 Hz physics, with tactile sensing at 70 Hz. Capture uses measured
PhysX link poses at 30 fps, retains resets, and renders full-body plus hand
close-up views. This is not a held-out success benchmark or a selected success
montage. Encoding validation checks dimensions, frame count and duration.

Five simulator-independent tests cover architecture matching, unbounded
latent commands and non-destructive checkpoint snapshots. Runtime validation
additionally requires strict checkpoint loading, finite observations/actions,
the controller audit and completed capture/encoding artifacts. Unit tests
alone are not evidence that the policy grasps or stays balanced.
