# Fabrics applied to G1 RL training

The video shows the actual sphere model configured for the running fabric-enabled
G1 + Revo2 policy: 14 moving right-arm/hand spheres and 7 fixed torso/left-arm
spheres. These are controller collision proxies, not tactile sensors, physical
collision meshes, or whole-body control.

The capture uses a separate small inference process with a stable checkpoint,
the saved training configuration, the fabric controller, and its existing soft
PCA prior enabled. It does not add spheres to an unrelated policy's recording.
Training and its files are not changed.

For each video sample, the training collision adapter records its measured
moving and fixed sphere centers, table-plane height, and ordered signed-clearance
vector. The renderer independently checks:

- Active robot-link poses against URDF forward kinematics (translation and rotation).
- Every moving sphere center against its measured link pose and local offset.
- Every fixed sphere center against its pelvis-frame definition and robot root pose.
- Every body, table, and configured self-collision clearance against the recorded
  controller value, preserving masks and pair ordering.

The unmodeled static body uses the documented G1 hold posture, including the
left-arm rest pose corresponding to the fixed-body proxy geometry. Spheres are
not shifted or enlarged for appearance. Three thin great-circle outlines make
the configured radii legible through translucent robot visuals.

Blue marks fixed body proxies; teal marks moving proxies. Moving spheres turn
amber inside the configured 8 cm avoidance range, and red for negative proxy
clearance. Proximity coloring does not claim a measured contact force or prove
that the controller is producing a particular repulsive force. Overlaps remain
visible; the fabric is not presented as a collision-free guarantee.

Use the existing capture/render scripts:

```
python scripts/record_play2perfect_sapg.py --headless --fabrics \
  --play2perfect-root /path/to/play2perfect \
  --config /path/to/fabric-run/params \
  --checkpoint /path/to/stable-checkpoint.pth \
  --output /path/to/new-recording-directory --seconds 20

python scripts/render_play2perfect_recording.py /path/to/new-recording-directory \
  --fabrics --video-name fabrics-g1-rl-training.mp4
```

Run these as Slurm steps on the user's allocated GPU, sequentially. Rendering
uses a small EGL graphics context on that GPU and does not launch a live viewer.
The output title is exactly “Fabrics applied to G1 RL training”. Raw trajectories,
source configuration, checkpoint provenance, and numerical render validation
are retained beside the MP4.

## Recorded example (2026-09-07)

The stable epoch-4,000 checkpoint loads strictly with 131 observations and 13
actions. The recording contains 600 frames at 30 FPS (20 seconds), all 21
sphere centers and all 125 enabled pair/plane clearances per frame. The
existing 5% soft PCA prior remains enabled. Eleven geometry/adapter tests pass.

The final H.264 MP4 is 1280 × 720 at 30 FPS, with 600 frames and a verified
20.000-second duration. Across all frames, maximum link-position disagreement
is 1.06 micrometers, maximum sphere-center disagreement is 0.15 micrometers,
and maximum clearance disagreement is 0.09 micrometers. The numeric results
are saved in `render_validation.json` beside the video.

This is an early-training geometry demonstration: the recorded environment
reached no goals and reset three times. Resets are retained and counted in the
video. Nine sampled frames contain negative proxy clearance (minimum -0.05741
m); those overlaps are rendered red rather than concealed. A proxy overlap
alone does not establish physical mesh contact. These observations are not a
success-rate evaluation or a collision-avoidance guarantee.

## Longer, success-selected recordings

Pass `--seconds 120 --select-successful-env` to capture all six environments
and retain one **continuous** rollout, selected by completed goal count, then
longest sustained lift above the configured reward threshold, then reward.
This selection does not splice episodes, hide resets, change objects after
capture, or relax the goal tolerance. The selected environment's exact object
and table geometry are exported. All candidate trajectories and per-step
metrics remain available in `candidates.npz` and `candidate_metrics.json`.

Goal events use the episode success counter captured before automatic reset,
including successes on terminal steps. Sustained lifting is measured using
height on every step, not the task's latched `lifted` flag. The reward's lift
metric includes a 5 cm offset; the video's object-rise readout removes that
offset. The 0.15 m reward threshold therefore corresponds to a root height
increase exceeding 0.10 m. Neither height nor a latched flag alone proves a
stable grasp; successful segments still need visual inspection.

The video shows completed goals and highlights goal events. Metadata includes
first-goal timestamps, the longest sustained-lift interval, and all candidate
summaries. No goal is claimed if none occurred. Fourteen selection and geometry
tests pass. The original short recording is preserved.
