#!/usr/bin/env python3
"""Sequential real-physics resume, evaluation, and full-size SAPG checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--smoke-run',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--play2perfect-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=24576)
    p.add_argument('--reuse-evaluation',type=Path)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Use a Slurm allocation and a new validation directory')
    args.output.mkdir(parents=True)
    previous=json.loads((args.smoke_run/'segment_complete.json').read_text())
    base=[sys.executable,'scripts/train_g1_touch_continuation.py','--headless','--device','cuda:0',
          '--source-run',str(args.source_run),'--source-checkpoint',str(args.checkpoint),
          '--calibration',str(args.calibration),'--max-frames','-1']
    if previous.get('resume') and previous['frame'] > previous['segment_start_frame']:
        resumed=previous  # Reuse an already completed real-physics resume test.
    else:
        subprocess.run(base+['--output',str(args.smoke_run),'--num-envs',str(previous['num_envs']),
            '--resume',previous['checkpoint'],'--max-epochs',str(previous['epoch']+3)],check=True)
        resumed=json.loads((args.smoke_run/'segment_complete.json').read_text())
        assert resumed['segment_start_frame']==previous['frame'] and resumed['frame']>previous['frame']
    assert resumed['resume']['actor_optimizer_restored'] and resumed['resume']['critic_optimizer_restored']
    evaluation=args.reuse_evaluation or args.output/'evaluation'
    if args.reuse_evaluation:
        import hashlib
        metadata=json.loads((evaluation/'metadata.json').read_text())
        assert metadata['checkpoint_sha256']==hashlib.file_digest(Path(resumed['checkpoint']).open('rb'),'sha256').hexdigest()
        assert metadata['num_envs']==1200 and metadata['evaluation_seconds']==120 and metadata['seed']==42
        assert metadata['tolerance_parameter']==.01 and metadata['tactile'] and metadata['optimizer_updates']==0
    else:
        subprocess.run([sys.executable,'scripts/record_g1_bps_reposing.py','--headless','--device','cuda:0',
            '--run',str(args.smoke_run),'--checkpoint',resumed['checkpoint'],'--output',str(evaluation),
            '--play2perfect-root',str(args.play2perfect_root),'--metrics-only',
            '--num-envs','1200','--seconds','120','--seed','42'],check=True)
    report=json.loads((evaluation/'evaluation.json').read_text())
    if report['episode_any_goal_success_rate'] < .56486 or report['goals_per_simulated_minute'] < 12.609:
        raise RuntimeError('Smoke policy did not preserve useful strict reposing performance')
    scale=args.output/'0_full_scale'
    subprocess.run(base+['--output',str(scale),'--num-envs',str(args.num_envs),'--max-epochs','30'],check=True)
    benchmark=json.loads((scale/'segment_complete.json').read_text())
    if benchmark['device_free_gib'] < 1.5:
        raise RuntimeError('Insufficient full-size VRAM safety margin')
    progress=json.loads((scale/'progress.json').read_text())
    if not progress['actor_touch_weight_norm']>0 or not progress['critic_touch_weight_norm']>0:
        raise RuntimeError('New tactile columns did not learn during the smoke test')
    result=dict(passed=True,small_resume=resumed,strict_evaluation=report,
        full_scale=benchmark,full_scale_last_update=progress,calibration=str(args.calibration),
        evaluation_directory=str(evaluation))
    (args.output/'validation.json').write_text(json.dumps(result,indent=2))
    print('TOUCH_FULL_VALIDATION_PASSED '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
