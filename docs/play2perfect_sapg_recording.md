# Recording the original G1 + Revo2 SAPG policy

This is a replay of **Play2Perfect**, not the ADEPT-inspired policy. The contact
bench is not involved, and neither training nor hardware control runs in the
recording scripts.

The selected checkpoint is the original run's best saved reward checkpoint:

```
play2perfect/outputs/2026-08-20/10-49-23/
  0_g1_brainco_right_mesh_collision_factorized_multiusd_objects_play_sapg_24576_scratch_20260820_retry1/
    nn/0_g1_brainco_right_mesh_collision_factorized_multiusd_objects_play_sapg_24576_scratch_20260820_retry1.pth
```

Its epoch is 146,869; its training frame counter is 57,751,240,704. The saved
best reward is not a fresh evaluation or a success probability.

## Replay contract

- Load the run's `.hydra/config.yaml`, rather than today's task defaults.
- Load actor weights and observation normalizers strictly: 92 observations,
  13 actions, recurrent actor, all six trained SAPG coefficient settings.
- Evaluate the zero-entropy leader (coefficient ID 0) using deterministic mean
  actions. The generic player's default spreads IDs across environments;
  it must not silently select an exploratory follower for the recording.
- Preserve the original action pipeline, solver, timestep, actuator and
  collision settings, and training domain randomization. No fabric/PCA prior
  is inserted into the original policy.
- Use six environments, recording environment 0 and seed 42 continuously.
  Generate the original procedural tool pool. The round-robin spawner need
  only materialize its first six templates: the other temporary prototypes
  have no assigned environments. This does not change the selected objects.
- The checkpoint has no environment/curriculum state. Pin the evaluation
  success criterion explicitly to the configured final target of 0.01 m;
  do not silently start the 0.075 m beginner curriculum. This criterion can
  be changed with `--success-tolerance` and is stored in the metadata.
- Reset recurrent state at episode termination. Keep resets in the recording;
  do not splice together selected successes.

## Capture and render separately

Run both steps through Slurm, following the workstation guide. Do not request
another GPU if the user's one-running-GPU-job limit is already occupied.
A small overlapping step in the user's existing allocation can capture the
rollout without stopping training. Rendering then uses a small EGL graphics
context on that same allocated GPU, after the physics capture has exited.
The renderer refuses to start outside Slurm or if more than one NVIDIA EGL
device is accessible. See the [pyrender offscreen-rendering documentation](https://pyrender.readthedocs.io/en/latest/examples/offscreen.html).

```
python scripts/record_play2perfect_sapg.py --headless \
  --play2perfect-root /path/to/play2perfect \
  --config /path/to/run/.hydra/config.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --output /path/to/new-recording-directory --seconds 20

python scripts/render_play2perfect_recording.py /path/to/new-recording-directory
```

The renderer requires pyrender, PyOpenGL/EGL, Pillow and FFmpeg. Neither script
overwrites an existing recording. Raw outputs stay outside Git.

The capture writes measured body positions/orientations, joint positions,
object/table/goal poses, per-step metrics and checkpoint provenance. Exact
generated object and table meshes are exported before temporary simulation
assets disappear. The renderer compares active-link URDF forward kinematics
against PhysX poses, including orientation. A mismatch above 2 mm or 0.01 rad
fails rendering rather than visually correcting the motion.

The full G1 body is shown only as **static visual context**. The checkpoint
does not control walking, balance, or whole-body disturbance rejection. The
video's tool motion and active arm/hand poses are recorded physics, not a
hand-authored animation. Green translucent geometry marks the current target.

Each recorded simulation sample is rendered exactly once. FFmpeg assigns the
capture FPS, so slow computation does
not alter the clip's playback speed. `render_validation.json` records the
maximum kinematic alignment error.

A short fixed-seed clip is qualitative evidence, not a success-rate benchmark.
Consult `metadata.json` and `metrics.json` for its settings and observed results.

## 2026-09-07 capture

The fixed-seed 20-second rollout loaded the checkpoint strictly and captured
600 frames at 30 FPS. The selected object was a procedural brush variant from
the original 1,200-object pool. It reached 10 goals with no episode resets;
recorded total reward was 10,809.97. Domain randomization remained enabled.
This longer continuous reward total is not directly comparable to a training
episode's mean reward.

The first recording attempt caught a batched-observation shape check in the
new adapter; it was corrected to validate the feature dimension. A software
browser-rendering attempt was stopped because it took seconds per frame.
The final renderer uses the same saved poses and does not rerun or select a
different rollout. `nvidia-smi` confirmed its graphics context on physical GPU
2, alongside training, using approximately 93 MiB.

Final video validation: H.264, 1280x720, 30 FPS, 600 frames, exactly 20.0 s.
All 600 frames passed URDF/PhysX alignment checks: maximum translation error
1.083e-6 m and maximum orientation error 2.618e-6 rad. Both capture and
rendering processes exited; the contact viewer remains stopped and training
continues independently.
