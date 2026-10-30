#!/usr/bin/env python3
"""Evaluate a running whole-body training's checkpoints as they appear (learning curve on the deployment metric).

Every time nn/latest.pth of --run-dir advances by at least --interval-epochs, the checkpoint and its SGS state are
snapshotted (with the run's task_contract.json and params/agent.yaml, the layout eval_wholebody_checkpoint.py expects)
and evaluated with eval_wholebody_checkpoint.py (deterministic leader, fresh validated starts, fixed seed). One line per
evaluation: INFLIGHT {epoch, success, object_drop, goals, ...}. Stops when the training writes training_result.json
(after a final evaluation of the last checkpoint) or after --max-hours.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def snapshot(run, dest):
    dest.mkdir(parents=True)
    (dest/'params').mkdir()
    (dest/'nn').mkdir()
    shutil.copy(run/'task_contract.json', dest/'task_contract.json')
    shutil.copy(run/'params/agent.yaml', dest/'params/agent.yaml')
    shutil.copy(run/'params/env_resolved.yaml', dest/'params/env_resolved.yaml')
    shutil.copy(run/'nn/latest.sgs.pt', dest/'nn/ckpt.sgs.pt')
    shutil.copy(run/'nn/latest.pth', dest/'nn/ckpt.pth')
    return dest/'nn/ckpt.pth'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--interval-epochs', type=int, default=150)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-hours', type=float, default=12.)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eval_script = Path(__file__).with_name('eval_wholebody_checkpoint.py')
    last_epoch, t0, k = None, time.time(), 0
    print('CONFIG', json.dumps({k: str(v) for k, v in vars(a).items()}), flush=True)
    while time.time()-t0 < a.max_hours*3600:
        done = (a.run_dir/'training_result.json').exists()
        try:
            epoch = int(json.loads((a.run_dir/'progress.json').read_text())['epoch'])
        except Exception:
            epoch = None
        ready = (a.run_dir/'nn/latest.pth').exists() and epoch is not None
        if ready and (last_epoch is None or epoch-last_epoch >= a.interval_epochs or done):
            time.sleep(20)                                   # let a checkpoint write finish
            ckpt = snapshot(a.run_dir, a.out/f'snap_{k:03d}')
            payload_epoch = None
            res = subprocess.run([sys.executable, str(eval_script), '--checkpoint', str(ckpt), '--output',
                                  str(a.out/f'eval_{k:03d}'), '--num-envs', str(a.num_envs), '--episodes-per-env', '1',
                                  '--max-steps', str(a.max_steps), '--modes', 'det_leader', '--seed', str(a.seed)],
                                 capture_output=True, text=True)
            summary = a.out/f'eval_{k:03d}/summary.json'
            if summary.exists():
                d = json.loads(summary.read_text())
                if d.get('completed'):
                    r = d['results']['det_leader']['first_episode']['all']
                    payload_epoch = d.get('epoch')
                    row = dict(k=k, epoch=payload_epoch, success=r['success'], object_drop=r['object_drop'],
                               goals_mean=r['goals_mean_lower_bound'], goals_median=r['goals_median'],
                               lifted=r['lifted'], timeout=r['timeout'], robot_fall=r['robot_fall'], n=r['episodes'],
                               wall_minutes=(time.time()-t0)/60)
                    print('INFLIGHT', json.dumps(row), flush=True)
                    with (a.out/'inflight.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n')
                else:
                    print('INFLIGHT_ERROR', d.get('error'), flush=True)
            else:
                print('INFLIGHT_ERROR returncode', res.returncode, res.stderr[-800:], flush=True)
            shutil.rmtree(a.out/f'snap_{k:03d}/nn', ignore_errors=True)       # checkpoints are 0.8 GB each
            last_epoch, k = epoch, k+1
            if done:
                break
        time.sleep(30)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
