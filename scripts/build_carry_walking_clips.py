#!/usr/bin/env python3
"""Precompute released-planner walking references for carry-expert training.

Runs the pinned NVIDIA kinematic planner offline (CPU, no physics) from the
nominal standing pose: idle, a constant body-frame command, then idle. The
planner is fed its own integrated heading, i.e. an ideal tracker, so clips are
pure references. Training samples them on GPU instead of running 9k ONNX
planners. Requires G1-SONIC-planner/python-deps on PYTHONPATH.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


# name, (vx, vy, wz), [(start_s, end_s), ...] active intervals, total seconds
CLIPS = [
    ('idle', (0., 0., 0.), [], 10.),
    ('forward', (.4, 0., 0.), [(1., 7.)], 10.),
    ('backward', (-.4, 0., 0.), [(1., 7.)], 10.),
    ('left', (0., .4, 0.), [(1., 7.)], 10.),
    ('right', (0., -.4, 0.), [(1., 7.)], 10.),
    ('forward_left', (.28, .28, 0.), [(1., 7.)], 10.),
    ('forward_right', (.28, -.28, 0.), [(1., 7.)], 10.),
    ('backward_left', (-.28, .28, 0.), [(1., 7.)], 10.),
    ('backward_right', (-.28, -.28, 0.), [(1., 7.)], 10.),
    ('turn_left', (0., 0., .2), [(1., 7.)], 10.),
    ('turn_right', (0., 0., -.2), [(1., 7.)], 10.),
    ('arc_left', (.4, 0., .2), [(1., 7.)], 10.),
    ('arc_right', (.4, 0., -.2), [(1., 7.)], 10.),
    ('forward_stop_go', (.4, 0., 0.), [(1., 3.), (4., 6.)], 10.),
]


def command_at(spec, t):
    _, cmd, intervals, _ = spec
    return np.array(cmd if any(a <= t < b for a, b in intervals) else (0., 0., 0.), float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from dextrah_lab.wholebody.contract import nominal_body_pose
    from dextrah_lab.wholebody.navigation_reference import NavigationReference, ISAAC_FROM_MUJOCO, PLANNER_SHA256
    planner = args.workspace/'G1-SONIC-planner/planner_sonic.onnx'
    owner = NavigationReference(planner, speed_limit=.4)
    body = nominal_body_pose()
    frames = max(int(round(c[3]*50))+1 for c in CLIPS)
    out = dict(q=np.zeros((len(CLIPS), frames, 29), np.float32),
               root_quat=np.zeros((len(CLIPS), frames, 4), np.float32),
               root_pos=np.zeros((len(CLIPS), frames, 3), np.float32),
               velocity=np.zeros((len(CLIPS), frames, 3), np.float32),
               command=np.zeros((len(CLIPS), frames, 3), np.float32),
               lengths=np.zeros(len(CLIPS), np.int64))
    reports = {}
    for i, spec in enumerate(CLIPS):
        nav = NavigationReference(planner, session=owner.session, speed_limit=.4)
        nav.reset(np.array([0., 0., .76, 1., 0., 0., 0.]), body, 0.)
        n = int(round(spec[3]*50))+1
        for k in range(n):
            t = k/50
            cmd = command_at(spec, t)
            q, _, root = nav.reference(t, cmd, nav.desired_heading)
            out['q'][i, k], out['root_pos'][i, k], out['root_quat'][i, k] = q[0], root[0, :3], root[0, 3:7]
            out['command'][i, k] = cmd
        out['lengths'][i] = n
        # Planned body-frame velocity from the reference root trajectory.
        w, x, y, z = out['root_quat'][i, :n].T
        yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        dpos = np.gradient(out['root_pos'][i, :n, :2], 1/50, axis=0)
        c, s = np.cos(yaw), np.sin(yaw)
        out['velocity'][i, :n, 0] = c*dpos[:, 0]+s*dpos[:, 1]
        out['velocity'][i, :n, 1] = -s*dpos[:, 0]+c*dpos[:, 1]
        out['velocity'][i, :n, 2] = np.gradient(yaw, 1/50)
        for k in range(n, frames):  # pad with the final standing frame
            for key in ('q', 'root_quat', 'root_pos'):
                out[key][i, k] = out[key][i, n-1]
        active = out['command'][i, :n, :].any(-1)
        steady = active & (np.arange(n)/50 >= (spec[2][0][0]+2. if spec[2] else 0.))
        reports[spec[0]] = dict(command=list(spec[1]), intervals=spec[2], seconds=spec[3],
            planner_calls=nav.inferences,
            steady_planned_velocity=out['velocity'][i, :n][steady].mean(0).tolist() if steady.any() else None,
            final_planned_speed=float(np.linalg.norm(out['velocity'][i, n-1, :2])))
        print('CLIP', spec[0], json.dumps(reports[spec[0]]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, names=np.array([c[0] for c in CLIPS]), **out)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    meta = dict(planner_sha256=PLANNER_SHA256, clips=reports, hz=50, joint_order='Isaac BODY_JOINTS',
                isaac_from_mujoco=ISAAC_FROM_MUJOCO.tolist(), heading_feedback='integrated desired heading (ideal tracker)',
                output_sha256=digest)
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print('CLIPS_WRITTEN', args.output, digest, flush=True)


if __name__ == '__main__':
    main()
