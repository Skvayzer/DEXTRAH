#!/usr/bin/env python3
"""Summarize fixed-body teacher probes (scripts/train_g1_touch_continuation.py --stop-after-seconds).

For each probe output directory: env count, backend, startup seconds (output
directory creation -> resume_validation.json, written just before training),
steady throughput (median of the last epochs' rl_games "fps total" and
"fps step + policy inference" in the Slurm log section of that probe), epoch
time, and peak device memory = memory outside PyTorch (device used - PyTorch
reserved, at the last progress snapshot) + PyTorch peak reserved.
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def log_sections(log):
    """Split a mj_task/isaac_task log into {output_dir: text} by its '=== RUN' lines."""
    sections, current = {}, None
    for line in Path(log).read_text(errors='replace').splitlines():
        if line.startswith('=== RUN'):
            m = re.search(r'--output (\S+)', line)
            current = m.group(1) if m else None
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: '\n'.join(v) for k, v in sections.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--job', required=True, help='Slurm job id (the log prints the literal $SLURM_JOB_ID)')
    p.add_argument('--last', type=int, default=8, help='epochs for the steady-state median')
    args = p.parse_args()
    import subprocess, datetime
    start = subprocess.run(['sacct', '-j', str(args.job), '-o', 'Start', '-P', '-n'], capture_output=True, text=True).stdout.split()
    previous_end = datetime.datetime.fromisoformat(start[0]).timestamp() if start else None
    rows = []
    for out, text in log_sections(args.log).items():
        d = Path(out.replace('$SLURM_JOB_ID', str(args.job))) if out else None
        m = re.search(r'fixedprobe_(mjlab|isaac)_(\d+)_', out or '')
        row = dict(output=str(d), envs=int(m.group(2)) if m else None, backend=m.group(1) if m else None)
        tot = [int(x.replace(',', '')) for x in re.findall(r'fps total\s*:\s*([\d,]+)', text)]
        inf = [int(x.replace(',', '')) for x in re.findall(r'fps step \+ policy inference\s*:\s*([\d,]+)', text)]
        ep = [float(x) for x in re.findall(r'Time to train epoch\s*:\s*([\d.]+)', text)]
        row.update(epochs=len(tot), failed='Traceback' in text or 'out of memory' in text.lower(),
                   oom='out of memory' in text.lower() or 'OutOfMemory' in text)
        if tot:
            row.update(fps_total=statistics.median(tot[-args.last:]), fps_step_inference=statistics.median(inf[-args.last:]),
                       seconds_per_epoch=statistics.median(ep[-args.last:]))
        if d and d.is_dir():
            try:
                # Startup: previous probe's end (or job start) -> resume_validation.json,
                # written just before the first training epoch.
                resume = d/'resume_validation.json'
                if resume.exists() and previous_end:
                    row['startup_s'] = round(resume.stat().st_mtime-previous_end, 1)
                done = d/'segment_complete.json'
                if done.exists():
                    previous_end = done.stat().st_mtime
                prog = json.loads((d/'progress.json').read_text())
                non_torch = prog['device_used_gib']-prog.get('cuda_reserved_gib', prog['cuda_peak_reserved_gib'])
                row.update(non_torch_gib=round(non_torch, 2), torch_peak_reserved_gib=round(prog['cuda_peak_reserved_gib'], 2),
                           torch_peak_allocated_gib=round(prog.get('cuda_peak_allocated_gib', float('nan')), 2),
                           peak_device_gib=round(non_torch+prog['cuda_peak_reserved_gib'], 2),
                           total_gib=round(prog['device_used_gib']+prog['device_free_gib'], 2))
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                row['progress_error'] = repr(e)
        rows.append(row)
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
