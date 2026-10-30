#!/usr/bin/env python3
"""Continue a whole-body SAPG fine-tune in chunks and evaluate after each chunk (early learning curve on the
deployment metric; the training run fills the GPU, so evaluation cannot run beside it).

Chunk i resumes from chunk i-1's nn/latest.pth (rl_games optimizer state + the SGS state X.sgs.pt) with the training
flags of M run 1454 (M_TRAIN_ARGS below, reconstructed from its task contract; see ~/robotics/m_train_args.sh) and
--max-epochs start + --chunk-epochs; then that checkpoint is evaluated with eval_wholebody_checkpoint.py (deterministic
leader, fresh validated starts, fixed seed). The finger servo comes from DEXTRAH_HAND_SERVO (recorded in each chunk's
task contract). Before the first chunk the resume checkpoint itself is evaluated. One line per evaluation:
CHUNK_EVAL {chunk, epoch, success, object_drop, goals_mean, ...}.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

M_TRAIN_ARGS = ['--backend', 'mjlab', '--mj-nconmax', '64', '--mj-njmax', '300', '--mj-cone-hessian-fix',
                '--frozen-pretrained-sonic', '--adapter-learning-rate', '0.0003', '--lean-sapg-batches', '--dual-clip', '3',
                '--grasp-reward-scale', '1.0', '--boundary-gc-freeze', '--boundary-gc-interval', '1',
                '--numerical-failure-mode', 'reset', '--per-env-object-physics', '--body-table-contact-monitor-only',
                '--hand-surface-obs', '--hand-surface-arm-links', '--hand-surface-goal', '--sgs', '--sgs-families',
                'reaching,stable_grasp,near_goal', '--sgs-validate-spawns', '--num-envs', '24960', '--minibatch-size', '39936',
                '--wandb', 'disabled']


def checkpoint_epoch(path):
    import torch
    p = torch.load(path, map_location='cpu', weights_only=False)
    p = p[0] if 0 in p else p
    return int(p['epoch'])


def evaluate(ckpt, out, num_envs, seed, label):
    run = Path(ckpt).parent.parent
    snap_ckpt = snapshot_from(run, Path(ckpt), out/f'snap_{label}')
    res = subprocess.run([sys.executable, str(Path(__file__).with_name('eval_wholebody_checkpoint.py')), '--checkpoint',
                          str(snap_ckpt), '--output', str(out/f'eval_{label}'), '--num-envs', str(num_envs),
                          '--episodes-per-env', '1', '--max-steps', '3000', '--modes', 'det_leader', '--seed', str(seed)],
                         capture_output=True, text=True)
    s = out/f'eval_{label}/summary.json'
    d = json.loads(s.read_text()) if s.exists() else dict(error=res.stderr[-800:])
    import shutil
    shutil.rmtree(out/f'snap_{label}/nn', ignore_errors=True)
    if not d.get('completed'):
        print('CHUNK_EVAL_ERROR', label, d.get('error'), flush=True)
        return None
    r = d['results']['det_leader']['first_episode']['all']
    return dict(epoch=d.get('epoch'), success=r['success'], object_drop=r['object_drop'], goals_mean=r['goals_mean_lower_bound'],
                goals_median=r['goals_median'], lifted=r['lifted'], timeout=r['timeout'], robot_fall=r['robot_fall'], n=r['episodes'])


def snapshot_from(run, ckpt, dest):
    import shutil
    dest.mkdir(parents=True)
    (dest/'params').mkdir()
    (dest/'nn').mkdir()
    shutil.copy(run/'task_contract.json', dest/'task_contract.json')
    for f in ('agent.yaml', 'env_resolved.yaml'):
        shutil.copy(run/'params'/f, dest/'params'/f)
    shutil.copy(ckpt.with_suffix('.sgs.pt'), dest/'nn/ckpt.sgs.pt')
    shutil.copy(ckpt, dest/'nn/ckpt.pth')
    return dest/'nn/ckpt.pth'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunks', type=int, default=6)
    p.add_argument('--chunk-epochs', type=int, default=150)
    p.add_argument('--eval-envs', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-initial-eval', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps({k: str(v) for k, v in vars(a).items()}), flush=True)
    t0 = time.time()
    prev = a.resume
    log = a.out/'chunk_evals.jsonl'

    def record(i, row):
        row = dict(chunk=i, wall_minutes=round((time.time()-t0)/60, 1), **row)
        print('CHUNK_EVAL', json.dumps(row), flush=True)
        with log.open('a') as f:
            f.write(json.dumps(row)+'\n')
    if not a.skip_initial_eval:
        r = evaluate(prev, a.out, a.eval_envs, a.seed, 'start')
        if r:
            record(0, r)
    for i in range(1, a.chunks+1):
        start = checkpoint_epoch(prev)
        out_i = a.out/f'0_chunk_{i:02d}'
        cmd = [sys.executable, str(Path(__file__).with_name('train_g1_sonic_sapg.py')), *M_TRAIN_ARGS,
               '--max-epochs', str(start+a.chunk_epochs), '--resume', str(prev), '--output', str(out_i)]
        print('CHUNK_TRAIN', i, 'from epoch', start, flush=True)
        with (a.out/f'train_{i:02d}.log').open('w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        result = json.loads((out_i/'training_result.json').read_text()) if (out_i/'training_result.json').exists() else {}
        if rc != 0 or not result.get('completed'):
            print('CHUNK_TRAIN_ERROR', i, rc, result.get('error'), flush=True)
            break
        prev = out_i/'nn/latest.pth'
        r = evaluate(prev, a.out, a.eval_envs, a.seed, f'{i:02d}')
        if r:
            record(i, r)
    print('DONE %.1f min' % ((time.time()-t0)/60), flush=True)


if __name__ == '__main__':
    main()
