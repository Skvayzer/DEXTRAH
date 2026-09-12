#!/usr/bin/env python3
"""Run the predeclared, post-training teacher comparison on ONE Slurm GPU.

No training/W&B initialization, no checkpoint edits. Each rollout is a fresh
Isaac process with strictly restored policy weights and recurrent state reset.
Resume skips only protocol- and checksum-validated completed evaluations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dextrah_lab.object_shape.teacher_suite import cases, summarize


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def candidates(workspace):
    bps = workspace/'DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355'
    touch = workspace/'DEXTRAH-BPS128-TOUCH/outputs/0_bps128_touch70_continuous_383'
    return {
        'bps_baseline': (bps, bps/'nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth'),
        'touch_best': (touch, touch/'nn/g1_sapg_bps128_touch.pth'),
        'touch_final': (touch, touch/'nn/complete_17364025344.pth'),
    }


def rollout_command(case, candidate, output, p2p):
    command = [sys.executable, str(Path(__file__).with_name('record_g1_bps_reposing.py')),
        '--headless', '--device', 'cuda:0', '--run', candidate['run'],
        '--checkpoint', candidate['checkpoint'], '--output', str(output),
        '--play2perfect-root', str(p2p), '--num-envs', '1200',
        '--seconds', '120', '--video-seconds', '60', '--fps', '30',
        '--seed', str(case['seed']), '--object-seed', str(case['object_seed']),
        '--wrench-multiplier', str(case['wrench_multiplier'])]
    if case['condition'] == 'training' and case['seed'] == 42:
        command += ['--capture-reference']
    else:
        command += ['--metrics-only']
    return command


def validate_result(path, case, candidate):
    meta = json.loads((path/'metadata.json').read_text())
    report = json.loads((path/'evaluation.json').read_text())
    checks = {'seed': case['seed'], 'object_seed': case['object_seed'],
        'wrench_multiplier': case['wrench_multiplier'], 'num_envs': 1200,
        'evaluation_seconds': 120., 'policy_dt': 1/60, 'optimizer_updates': 0,
        'checkpoint_sha256': candidate['checkpoint_sha256'],
        'fabrics': False, 'pca': False, 'success_steps': 10,
        'consecutive_success_steps': False, 'max_keypoint_error_m': .015,
        'tactile': case['candidate'] != 'bps_baseline'}
    if any(meta.get(key) != value for key, value in checks.items()):
        raise ValueError(f'Evaluation protocol mismatch: {path}')
    if case['condition'] == 'heldout' and meta['shape_audit']['training_geometry_overlap'] != 0:
        raise ValueError('Held-out shape leakage')
    if report.get('zero_training_updates') is not True or report.get('checkpoint_sha256') != candidate['checkpoint_sha256']:
        raise ValueError('Invalid frozen evaluation provenance')
    if not (path/'per_object_counts.npz').is_file():
        raise ValueError('Missing per-object counts')
    if case['condition'] == 'training' and case['seed'] == 42:
        for family in ('hammer', 'spatula', 'brush', 'eraser'):
            clip = path/'clips'/family
            if not json.loads((clip/'metadata.json').read_text()).get('completed'):
                raise ValueError('Incomplete clip')
            if not (clip/'reference_trace.npz').is_file():
                raise ValueError('Missing policy-rate reference trace')
    return report


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--render-winner', action='store_true')
    args = p.parse_args()
    if args.execute and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Execution requires a Slurm GPU allocation')
    project = Path(__file__).resolve().parents[1]
    p2p = args.workspace/'play2perfect'
    inputs = {}
    for name, (run, checkpoint) in candidates(args.workspace).items():
        inputs[name] = dict(run=str(run), checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            env_sha256=sha256(run/'params/env_resolved.yaml'),
            agent_sha256=sha256(run/'params/agent.yaml'),
            bank_manifest_sha256=sha256(run/'bps/manifest.json'))
    protocol = dict(version=1, candidates=inputs, cases=cases(),
        play2perfect_revision=subprocess.check_output(['git','-C',str(p2p),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(path.relative_to(project)):sha256(path) for path in
            sorted([*project.glob('dextrah_lab/**/*.py'),
                    project/'scripts/record_g1_bps_reposing.py', Path(__file__).resolve(),
                    project/'scripts/render_g1_bps_reposing.py'])},
        num_envs=1200, seconds=120., one_gpu=True, training_updates=0,
        selection='Training-bank throughput with success/lift and stress regression guards; held-out not used for selection')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('Immutable protocol changed; use a new output directory')
    else:
        atomic_json(manifest, protocol)
    if not args.execute:
        print(json.dumps(protocol, indent=2))
        return
    subprocess.run(['nvidia-smi'], check=True)
    records = []
    for index, case in enumerate(cases()):
        candidate = inputs[case['candidate']]
        directory = args.output/f"{case['candidate']}_{case['condition']}_seed{case['seed']}"
        directory.mkdir(exist_ok=True)
        marker = directory/'completed.json'
        if marker.exists():
            attempt = directory/json.loads(marker.read_text())['attempt']
            report = validate_result(attempt, case, candidate)
        else:
            attempt = directory/f'attempt_{time.time_ns()}'
            command = rollout_command(case, candidate, attempt, p2p)
            print('TEACHER_EVAL_START '+json.dumps(dict(index=index+1,total=27,case=case,output=str(attempt))),flush=True)
            with (directory/f'{attempt.name}.log').open('x') as log:
                result = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Evaluation failed ({result.returncode}); inspect {log.name}. No teacher selected.')
            report = validate_result(attempt, case, candidate)
            atomic_json(marker, dict(attempt=attempt.name))
        records.append(dict(case, report=report, path=str(attempt)))
        atomic_json(args.output/'progress.json', dict(completed=len(records),total=27,last_case=case))
        print('TEACHER_EVAL_COMPLETE '+json.dumps(dict(case=case,report=report)),flush=True)
    selection = summarize(records)
    selection['records'] = records
    selection['candidate'] = inputs[selection['selected_candidate']]
    atomic_json(args.output/'selection.json', selection)
    print('TEACHER_SELECTED '+json.dumps({k:v for k,v in selection.items() if k not in ('records','metrics')}),flush=True)
    if args.render_winner:
        winner = next(r for r in records if r['candidate'] == selection['selected_candidate']
                      and r['condition'] == 'training' and r['seed'] == 42)
        for family in ('hammer', 'spatula', 'brush', 'eraser'):
            recording = Path(winner['path'])/'clips'/family
            if (recording/'selected-teacher-validation.json').is_file():
                continue
            subprocess.run([sys.executable,str(project/'scripts/render_g1_bps_reposing.py'),
                str(recording),'--video-name','selected-teacher.mp4'],check=True)
    atomic_json(args.output/'completed.json', dict(selected=selection['selected_candidate'],
        evaluations=27,training_started=False))


if __name__ == '__main__':
    main()
