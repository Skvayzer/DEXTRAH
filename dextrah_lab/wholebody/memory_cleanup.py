"""Bound cyclic-garbage retention between complete SAPG optimizer updates.

Run 547's live CUDA allocations repeatedly dropped by ~20 GiB at full Python
GC cycles. Collect at a short, explicit interval; do not flush the CUDA cache,
detach live training tensors, discard experience, or reset the simulator.
This is a mitigation, not attribution of the cycles to a specific component.
"""
import gc
import json
from pathlib import Path
import time
import torch


def install_boundary_gc(algo, output, interval=16):
    if interval < 1:
        raise ValueError('Require a positive GC interval')
    original = algo.train_epoch
    output = Path(output)
    first_epoch = int(algo.epoch_num)
    report = dict(interval_updates=interval, calls=0, collected_objects=0,
                  released_tensor_bytes=0, wall_seconds=0.)

    def train_epoch(*args, **kwargs):
        epoch = int(algo.epoch_num)
        if (epoch-first_epoch-1) % interval == 0:
            before = torch.cuda.memory_allocated()
            start = time.monotonic()
            collected = gc.collect(2)
            elapsed = time.monotonic()-start
            after = torch.cuda.memory_allocated()
            released = max(0, before-after)
            report['calls'] += 1
            report['collected_objects'] += collected
            report['released_tensor_bytes'] += released
            report['wall_seconds'] += elapsed
            sample = dict(epoch=epoch, frame=int(algo.frame), collected=collected,
                before_gib=before/2**30, after_gib=after/2**30,
                released_gib=released/2**30, wall_seconds=elapsed)
            with (output/'boundary_gc.jsonl').open('a') as file:
                file.write(json.dumps(sample)+'\n')
            (output/'boundary_gc_summary.json').write_text(json.dumps(report, indent=2))
            for name in ('before_gib', 'after_gib', 'released_gib', 'wall_seconds'):
                algo.writer.add_scalar('memory_cleanup/'+name, sample[name], int(algo.frame))
            if report['calls'] <= 4 or released > 2**30:
                print('SAPG_BOUNDARY_GC '+json.dumps(sample), flush=True)
        return original(*args, **kwargs)

    algo.train_epoch = train_epoch
    return report
