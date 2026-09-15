#!/usr/bin/env python3
"""Validate the actual ONNX planner interface, not physics or grasp success."""
import argparse
import json
import os
import numpy as np
from dextrah_lab.wholebody.navigation_reference import NavigationReference
from dextrah_lab.wholebody.contract import nominal_body_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated CPU/GPU step')
    planner = NavigationReference(args.planner)
    root = [0, .42, .75, np.sqrt(.5), 0, 0, -np.sqrt(.5)]
    planner.reset(root, nominal_body_pose(), 0.)
    max_qd = 0.
    for i in range(301):
        now = i/60
        command = [0, -.12 if 1. <= now < 4. else 0., 0]
        q, qd, ref = planner.reference(now, command, -np.pi/2)
        if not np.isfinite(np.r_[q.ravel(), qd.ravel(), ref.ravel()]).all():
            raise ValueError('Nonfinite planner sequence')
        max_qd = max(max_qd, float(np.abs(qd).max()))
        if i % 60 == 0:
            print('REFERENCE_SAMPLE '+json.dumps(dict(time=now, command=command,
                root=ref[0].tolist(), joint_velocity_abs_max=float(np.abs(qd).max()))), flush=True)
    print('NAVIGATION_REFERENCE_VALIDATED '+json.dumps(dict(**planner.report(),
        max_reference_joint_velocity=max_qd, physics_validated=False)), flush=True)


if __name__ == '__main__':
    main()
