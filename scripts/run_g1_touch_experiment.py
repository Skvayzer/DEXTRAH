#!/usr/bin/env python3
"""Five bounded 100M SAPG segments with strict frozen-policy evaluations.

One process at a time on the same allocated GPU. Physics starts fresh between
segments; both optimizers, networks, statistics and step counters are retained.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--source-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--play2perfect-root',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=24576)
    p.add_argument('--control',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Require a GPU allocation and a new experiment directory')
    validation=json.loads(args.validation.read_text())
    if not validation['passed'] or validation['full_scale']['num_envs'] != args.num_envs:
        raise RuntimeError('Require successful physical validation at the requested training scale')
    if Path(validation['calibration']).resolve() != args.calibration.resolve():
        raise RuntimeError('Training must use the validated calibration artifact')
    import wandb
    run_id='unique_id_'+args.output.name
    default_excepthook=sys.excepthook
    def record_failure(kind,error,tb):
        if args.output.exists():
            (args.output/'pilot_failure.json').write_text(json.dumps(dict(error=str(error)),indent=2))
        if (args.output/'wandb.json').exists():
            try:
                with wandb.init(project='adept',entity='skvayzer',id=run_id,resume='must',dir=str(args.output)) as run:
                    # Keep a more informative regression-stop status if already set.
                    if run.summary.get('experiment_status') != 'stopped_regression':
                        run.summary['experiment_status']='failed'
                    run.summary['failure_reason']=str(error)
            except Exception:
                pass  # Reporting failure must not replace the original failure.
        default_excepthook(kind,error,tb)
    sys.excepthook=record_failure
    checkpoint=None
    results=[]
    for target in (100000000,200000000,300000000,400000000,500000000):
        command=[sys.executable,'scripts/train_g1_touch_continuation.py','--headless','--device','cuda:0',
            '--source-run',str(args.source_run),'--source-checkpoint',str(args.source_checkpoint),
            '--output',str(args.output),'--num-envs',str(args.num_envs),'--max-frames',str(target),
            '--calibration',str(args.calibration),'--wandb']
        if checkpoint:
            command += ['--resume',checkpoint]
        if args.control:
            command += ['--control']
        subprocess.run(command,check=True)
        segment=json.loads((args.output/'segment_complete.json').read_text())
        checkpoint=segment['checkpoint']
        evaluation=args.output/'evaluations'/f"frame_{segment['frame']}"
        subprocess.run([sys.executable,'scripts/record_g1_bps_reposing.py','--headless','--device','cuda:0',
            '--run',str(args.output),'--checkpoint',checkpoint,'--output',str(evaluation),
            '--play2perfect-root',str(args.play2perfect_root),'--metrics-only',
            '--num-envs','1200','--seconds','120','--seed','42'],check=True)
        report=json.loads((evaluation/'evaluation.json').read_text())
        # A coarse safety gate, not a statistical claim of improvement.
        # Reference final 8B BPS policy: 66.486% episode any-goal, 15.76125 goals/min.
        regression=(report['episode_any_goal_success_rate'] < .56486 or
                    report['goals_per_simulated_minute'] < 12.609)
        status='stopped_regression' if regression else ('pilot_complete' if target==500000000 else 'evaluated_continuing')
        record=dict(segment=segment,evaluation=report,status=status)
        results.append(record)
        (args.output/'experiment_results.json').write_text(json.dumps(results,indent=2))
        with wandb.init(project='adept',entity='skvayzer',id=run_id,resume='must',
                        dir=str(args.output),name=args.output.name) as run:
            wandb.define_metric('evaluation/additional_transitions')
            wandb.define_metric('evaluation/*',step_metric='evaluation/additional_transitions')
            metrics={'evaluation/'+k:v for k,v in report.items() if isinstance(v,(int,float))}
            metrics['evaluation/additional_transitions']=segment['frame']
            wandb.log(metrics)
            run.summary.update(dict(experiment_status=status,last_evaluation_frame=segment['frame'],
                latest_checkpoint=checkpoint,episode_any_goal_success_rate=report['episode_any_goal_success_rate'],
                goals_per_simulated_minute=report['goals_per_simulated_minute']))
        print('STRICT_EVALUATION '+json.dumps(dict(frame=segment['frame'],status=status,report=report)),flush=True)
        if regression:
            raise RuntimeError('Strict evaluation regressed beyond the safety gate; pilot stopped and checkpoint preserved')


if __name__=='__main__':
    main()
