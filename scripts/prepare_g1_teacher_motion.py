#!/usr/bin/env python3
"""Export reset-separated 50 Hz source motions for later whole-body retargeting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dextrah_lab.wholebody.reference import successful_segments, retime_segment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip',type=Path,required=True,help='Completed evaluation clip directory')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta=json.loads((args.clip/'metadata.json').read_text())
    if not meta.get('completed') or not meta.get('capture_reference') or meta.get('optimizer_updates')!=0:
        raise ValueError('Requires completed frozen-policy reference capture')
    dt=meta['reference_dt']
    if not np.isclose(dt,1/60):
        raise ValueError('Unexpected source controller rate')
    path=args.clip/'reference_trace.npz'
    with np.load(path,allow_pickle=False) as archive:
        trace=dict(archive)
    if len(meta['joint_names'])!=trace['joint_pos'].shape[-1]:
        raise ValueError('Joint-name/trace dimension mismatch')
    segments=successful_segments(trace,dt=dt)
    if not segments:
        raise ValueError('No sufficiently long success-containing segment without reset')
    args.output.mkdir(parents=True)
    reports=[]
    for i,segment in enumerate(segments):
        data=retime_segment(trace,segment,source_dt=dt)
        name=f'segment_{i:03d}.npz'
        np.savez_compressed(args.output/name,**data)
        reports.append(dict(file=name,source_start=segment.start,source_stop=segment.stop,
                            goal_hits=segment.goal_hits,duration_s=float(data['time_s'][-1])))
    report=dict(source_clip=str(args.clip.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_checkpoint_sha256=meta['checkpoint_sha256'],source_dt=dt,target_dt=1/50,
        joint_names=meta['joint_names'],robot_pose_convention='environment-local XYZ + world-axis wxyz',
        segments=reports,contains_full_body_reference=False,physics_validated=False,
        required_next_steps=['world-frame alignment','standing/reaching reference and IK',
                            'joint/velocity/contact feasibility','frozen SONIC tracking validation'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
