#!/usr/bin/env python3
"""Record in an overlapping Slurm step without stopping the training batch.

The caller must use srun --jobid=<training job> --overlap --exact. The recorder
shares only that allocation, copies a checkpoint, and has no optimizer. Memory
pressure terminates ONLY our own child process group, never the training job.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from dextrah_lab.wholebody.recording_contract import snapshot_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-name', default='g1_frozen_sonic_sapg_bps128_touch.pth')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--render-only', action='store_true', help='Render a completed capture, without simulation')
    parser.add_argument('--checkpoint-run', type=Path, help='Optional immutable snapshot; --source-run still monitors training')
    parser.add_argument('--brush-transfer', action='store_true')
    parser.add_argument('--navigation-planner', type=Path)
    parser.add_argument('--navigation-only', action='store_true')
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    if (args.navigation_planner and not args.brush_transfer) or (args.navigation_only and not args.navigation_planner):
        raise ValueError('Navigation requires --brush-transfer and --navigation-planner')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in (None, 'batch', 'extern'):
        raise RuntimeError('Use a separate step inside the existing training allocation')
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not device or ',' in device or args.output.exists() != args.render_only:
        raise ValueError('Require one allocated GPU and a fresh output (or existing --render-only capture)')
    if Path(args.checkpoint_name).name != args.checkpoint_name:
        raise ValueError('Checkpoint name must be a basename')

    def memory():
        raw = subprocess.check_output(['nvidia-smi', '-i', device,
            '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
        uuid, free = raw.split(',')
        return dict(gpu_uuid=uuid.strip(), free_mib=int(free.strip()))

    def progress():
        try:
            value = json.loads((args.source_run/'progress.json').read_text())
            return {k: value[k] for k in ('epoch', 'frame', 'pretrained_sonic_unchanged')}
        except (OSError, ValueError):
            return None

    initial_memory = memory()
    if initial_memory['free_mib'] < 12*1024:
        raise RuntimeError(f'Insufficient safe headroom; leave training alone: {initial_memory}')
    if args.render_only:
        provenance = json.loads((args.output/'provenance.json').read_text())
        result = json.loads((args.output/'capture/recording_result.json').read_text())
        if not result['completed'] or result['checkpoint_sha256'] != provenance['sha256']:
            raise ValueError('Require a completed capture matching the saved checkpoint')
        monitor = provenance.get('training_monitor_run', str(Path(provenance['original_checkpoint']).parent.parent))
        if Path(monitor).resolve() != args.source_run.resolve():
            raise ValueError('Render continuation must monitor the original source run')
        provenance.setdefault('render_continuations', []).append(dict(
            allocation=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID'],
            source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'),
            training_before=progress(), gpu_before=initial_memory))
    else:
        args.output.mkdir(parents=True)
        snapshot = args.output/'checkpoint_snapshot'
        (snapshot/'nn').mkdir(parents=True)
        (snapshot/'params').mkdir()
        copied = snapshot/'nn'/args.checkpoint_name
        checkpoint_run = args.checkpoint_run or args.source_run
        provenance = snapshot_checkpoint(checkpoint_run/'nn'/args.checkpoint_name, copied)
        for path in ('task_contract.json', 'params/agent.yaml'):
            shutil.copyfile(checkpoint_run/path, snapshot/path)
        provenance.update(training_before=progress(), gpu_before=initial_memory,
            allocation=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID'],
            source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'), training_stopped=False,
            training_monitor_run=str(args.source_run),
            experiment='brush_table_transfer' if args.brush_transfer else 'reposing')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    base = Path('/data1/users/konstantin.smirnov')
    sim_python = base/'venvs/g1_sonic/bin/python'
    render_python = base/'miniconda3/envs/adept_dextrah/bin/python'
    capture = args.output/'capture'

    def run(command):
        child = subprocess.Popen([str(x) for x in command], start_new_session=True)
        try:
            last_check = 0.
            while child.poll() is None:
                if time.monotonic()-last_check >= 20:
                    status = memory()
                    print('CONCURRENT_RECORDING_HEALTH '+json.dumps(dict(
                        **status, training=progress(), recorder_child=child.pid)), flush=True)
                    if status['free_mib'] < 6*1024:
                        raise RuntimeError('Recorder stopped to preserve training GPU headroom')
                    last_check = time.monotonic()
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f'Recorder child failed with status {child.returncode}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

    if not args.render_only:
        extra = ['--families', 'brush', '--brush-transfer'] if args.brush_transfer else []
        if args.navigation_planner:
            extra += ['--navigation-planner', args.navigation_planner]
        if args.navigation_only:
            extra += ['--navigation-only']
        run([sim_python, 'scripts/record_g1_sonic_checkpoint.py', '--headless', '--device', 'cuda:0',
             '--kit_args=--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4',
             '--checkpoint', copied, '--output', capture, '--num-envs', '6' if args.brush_transfer else '120', '--seconds', args.seconds,
             '--torch-memory-limit-gib', '4', *extra])
    for family in (() if args.capture_only else (('brush',) if args.brush_transfer else ('hammer', 'brush', 'spatula'))):
        run([render_python, 'scripts/render_g1_sonic_checkpoint.py', capture/family])
    provenance.update(training_after=progress(), gpu_after=memory(), completed=True,
                      video_rendered=not args.capture_only)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('CONCURRENT_RECORDING_COMPLETE '+json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
