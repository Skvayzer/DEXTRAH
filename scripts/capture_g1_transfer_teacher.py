#!/usr/bin/env python3
"""Capture and validate old SAPG behavior for adapter initialization, no RL."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts.run_g1_teacher_suite import candidates
from dextrah_lab.wholebody.teacher_data import validate_policy_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--candidate',choices=['bps_baseline','touch_best','touch_final'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30.)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Capture requires a Slurm GPU allocation')
    run, checkpoint=candidates(args.workspace)[args.candidate]
    script=Path(__file__).with_name('record_g1_bps_reposing.py')
    subprocess.run([sys.executable,str(script),'--headless','--device','cuda:0',
        '--run',str(run),'--checkpoint',str(checkpoint),'--output',str(args.output),
        '--play2perfect-root',str(args.workspace/'play2perfect'),'--num-envs','1200',
        '--seconds',str(args.seconds),'--video-seconds',str(args.seconds),
        '--capture-reference','--capture-distillation'],check=True,timeout=1800)
    reports={}
    for clip in sorted((args.output/'clips').iterdir()):
        metadata=json.loads((clip/'metadata.json').read_text())
        with np.load(clip/'reference_trace.npz',allow_pickle=False) as archive:
            trace=dict(archive)
        with np.load(clip/'teacher_observation_normalizer.npz',allow_pickle=False) as archive:
            normalizer=dict(archive)
        reports[clip.name]=validate_policy_capture(trace,metadata,normalizer)
    report=dict(candidate=args.candidate,development_capture_not_teacher_selection=True,
        clips=reports,training_updates=0,fullbody_transfer_validated=False)
    (args.output/'distillation_capture_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
