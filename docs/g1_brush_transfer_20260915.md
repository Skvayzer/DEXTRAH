# Brush transfer probe — 2026-09-15

## Scope and provenance

An inference-only experiment requested while frozen-SONIC training job **560**
continues. No optimizer updates, controller changes, forced walking commands,
forced finger release, or object-state overrides were used. Normal episode
resets remain enabled and are retained in the recording.

- Checkpoint: epoch **14,856**, **2,188,277,760** cumulative transitions, the
  same immutable snapshot used for today's standard reposing videos.
- SHA-256: `d37ed661f291d2fc107cf3f80222d87e6ca3fac5e0d0ed43f3e64a73e298cf0b`.
- Capture/render source: `790c27dd4122aa755a0e312abb40225c8965f665`.
- Slurm step: **560.4**, overlapping the existing one-GPU allocation.
- Workstation output: `outputs/brush_transfer_560_20260915/`.
- Six simulated environments for SAPG group compatibility; **only brush
  environment 0** receives the transfer goal sequence and is shown/reported
  here. The other five are not transfer trials.
- Deterministic leader, 60 simulated seconds, 30 fps, measured PhysX link poses.
- BPS-128, touch 70 Hz, policy 60 Hz, physics 120 Hz. Original standing SONIC
  reference and all 55 pretrained SONIC tensors remain unchanged.

## What changes for this probe

A second physical, colliding tabletop is placed to the robot's right. Each
table block is 0.475 x 0.4 x 0.3 m, with centers separated by 0.625 m and a
**0.15 m edge gap**. Receiver height follows the source table's reset height.
The receiver is included in fingertip contact sensing; an additional physical
contact reporter measures brush contact with the receiver and all robot links.

A scripted **object-goal** sequence waits for a sustained contact-confirmed
lift, holds the measured grasp orientation, moves the goal toward the receiver
at up to 0.035 m/s, then lowers it at up to 0.02 m/s. Goal motion pauses if the
object trails it by more than 0.08 m. Actions are always produced by the loaded
policy. No learned table detector/planner or receiver-shape observation was
added. Thus this is assisted target following, not autonomous pick-and-place.

The usual 50-goal reset is disabled and timeout extended beyond the clip;
robot-fall, object-drop, hand-far and numerical terminations remain. No training
configuration or active source snapshot is modified by these inference options.

## Observed result

| Attempt | Time in clip | Grasped | Carried over receiver | Outcome |
| --- | --- | --- | --- | --- |
| 1 | 0–11.60 s | Yes | No | Robot-fall termination |
| 2 | 11.60–29.57 s | Yes | Yes, from 26.10 s | Brush drops; object-drop termination |
| 3 | 29.57–60.00 s | Yes | Yes, from 44.87 s | Still ongoing; lowering starts at 53.83 s |

**Three grasps, two carries over the receiving tabletop, zero stable placements
and releases, one robot-fall reset, one object-drop reset, zero numerical
failures.** These three sequential attempts are a diagnostic, not a success-rate
estimate. The final attempt is censored by the end of the video, not a failed
completed placement attempt.

At the final captured frame, the brush is about **6.88 cm above** the receiver
surface by its complete transformed visual-mesh bounds and remains in hand.
No receiver contact occurs during that last attempt. The earlier drop briefly
contacts the receiver near its edge, but never satisfies the full-footprint
support criterion. Raw receiver contact must not be equated with placement.

In the last attempt, the pelvis moves about 0.40 m horizontally from reset.
Both feet change location: left/right ankle maximum XY shifts are approximately
0.259/0.247 m; final shifts approximately 0.219/0.247 m. This demonstrates body
and foot motion, but position traces alone do not establish clean walking as
opposed to shuffling/sliding. The video is needed to judge motion quality.

The ordinary reposing counters still exist in capture metadata, but **are not
transfer-success metrics**: a moving/paused object target can repeatedly satisfy
the original reposing criterion. Use `transfer_report` and the actual resets.

## Measurement and validation

Carry requires prior contact-confirmed lift, the whole object footprint inside
the receiver with a 1 cm margin, height above the receiver, and robot contact.
Placement additionally requires receiver support, low object linear/angular
speed, and negligible object contact against all robot bodies continuously for
one second. There is no action override to make this criterion pass.

Five transfer tests passed locally and in the allocated recording step before
capture; seven recording-contract tests also passed. A subsequent sixth
transfer test tightens the support-height gate to reject contact underneath
the tabletop. That measurement-only fix does not enter this already captured
source and cannot change this result: no attempt was classified as supported
or placed even under the earlier, looser gate. It does not affect policy goals.

The 60-second capture took 480.19 wall seconds after scene startup and performed
zero optimizer updates. GPU headroom stayed above the recorder's guard; training
continued to advance. Export and laptop-transfer validation are recorded in
the output directory's `video_validation.json` and `provenance.json`.

## Interpretation

The existing controller can follow an assisted lateral object target while
moving its body, but this clip does **not** demonstrate completed table-to-table
placement. The original reposing policy has no explicit learned release phase.
The last attempt is still lowering at the cutoff, so a longer unchanged probe
would separate lack of time from a true terminal placement/release limitation.
Any subsequent training for reliable transfer should be a separately authorized
experiment, not an unannounced change to job 560.

## Delivered video

The laptop copy is
`/Users/konstantinsmirnov/Desktop/research_recordings/g1-frozen-sonic-sapg-brush-table-transfer-60s-20260915.mp4`.
It passed 1,800-frame / 60-second / 1600x900 export validation. Initial,
pre-fall and final previews were visually inspected; no physical-link visuals
were omitted. The complete uncut clip retains both resets.

Workstation and laptop SHA-256 match:
`d65f514e09b13ca302acb5b9d78eb9620f5213ce51b689aaace568e2db415051`.
The completed recording provenance shows training advancing from
2,432,612,352 to 2,470,508,544 transitions, with pretrained SONIC unchanged.
The post-transfer scheduler check confirmed job 560 still RUNNING and only
its batch step remaining; no recorder, renderer or viewer was left active
by this experiment.
