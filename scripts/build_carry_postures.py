#!/usr/bin/env python3
"""Right-arm posture library for the carry posture prior (CPU, C MuJoCo forward kinematics).

Samples right-arm joint configurations (7 joints, BODY_JOINTS order: shoulder pitch / roll / yaw,
elbow, wrist roll / pitch / yaw) inside comfortable sub-ranges of the joint limits and stores each
palm pose (right_base_link) in the torso_link frame. The palm pose relative to the torso depends only
on these 7 angles (the shoulders hang from torso_link), so the whole-body state is irrelevant here.
At an episode start the carry env picks the posture that brings the held object, with its own
palm-to-object offset, nearest to the carry target (dextrah_lab/wholebody/carry_posture.py).
Postures with the palm inside the torso volume (self-collision is disabled in the task) are dropped.
Output (npz): q (N, 7) rad, palm_pos (N, 3) m and palm_quat (N, 4) wxyz in the torso frame, ranges.
"""
import argparse
from pathlib import Path

import numpy as np

JOINTS = ('right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
          'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
# Sub-ranges (rad): arm in front of the body, elbow from fully flexed to straight-ish.
RANGES = ((-1.4, 0.4), (-0.9, 0.25), (-0.9, 0.9), (-0.9, 1.3), (-1.6, 1.6), (-1.3, 1.3), (-1.3, 1.3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=60000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    from dextrah_lab.mj.compat import install
    install()
    import mujoco
    from mjlab.scene import Scene
    from dextrah_lab.mj.scene import SceneParams, build_scene_cfg
    from dextrah_lab.wholebody.contract import BODY_JOINTS
    m = Scene(build_scene_cfg(1, SceneParams(), robot_pose=((0., 0., .793), (1., 0., 0., 0.))), 'cpu').compile()
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint('robot/'+j).id] for j in JOINTS]
    torso, palm = m.body('robot/torso_link').id, m.body('robot/right_base_link').id
    rng = np.random.default_rng(args.seed)
    lo, hi = np.array(RANGES).T
    q = lo+(hi-lo)*rng.random((args.samples, 7))
    q = np.vstack([np.zeros((1, 7)), q])                     # include the zero pose (forearm forward)
    pos, quat = np.zeros((len(q), 3)), np.zeros((len(q), 4))
    for k, qq in enumerate(q):
        d.qpos[:] = m.qpos0
        d.qpos[adr] = qq
        mujoco.mj_kinematics(m, d)
        R = d.xmat[torso].reshape(3, 3)
        pos[k] = R.T @ (d.xpos[palm]-d.xpos[torso])
        Rp = R.T @ d.xmat[palm].reshape(3, 3)
        mujoco.mju_mat2Quat(quat[k], Rp.flatten())
    # Palm outside the torso volume (box in front of / beside the trunk), not behind the body.
    inside = (pos[:, 0] < .10) & (np.abs(pos[:, 1]) < .14) & (pos[:, 2] > -.20) & (pos[:, 2] < .40)
    keep = ~inside & (pos[:, 0] > -.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, q=q[keep].astype(np.float32), palm_pos=pos[keep].astype(np.float32),
             palm_quat=quat[keep].astype(np.float32), joints=np.array(JOINTS), ranges=np.array(RANGES),
             body_joint_index=np.array([BODY_JOINTS.index(j) for j in JOINTS]))
    print(f'postures {int(keep.sum())}/{len(q)}; palm x {pos[keep, 0].min():.2f}..{pos[keep, 0].max():.2f} '
          f'y {pos[keep, 1].min():.2f}..{pos[keep, 1].max():.2f} z {pos[keep, 2].min():.2f}..{pos[keep, 2].max():.2f}; '
          f'zero pose palm {pos[0].round(3).tolist()}')


if __name__ == '__main__':
    main()
