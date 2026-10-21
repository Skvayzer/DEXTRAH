"""Bound cyclic-garbage retention between complete SAPG optimizer updates.

Run 547's live CUDA allocations repeatedly dropped by ~20 GiB at full Python
GC cycles. Collect at a short, explicit interval; do not flush the CUDA cache,
detach live training tensors, discard experience, or reset the simulator.
This is a mitigation, not attribution of the cycles to a specific component.
"""
from collections import Counter
import gc
import json
from pathlib import Path
import time
import torch


def _attribute(garbage, limit=25):
    """Composition of a cyclic-garbage batch that holds CUDA tensors (for finding the cycle)."""
    tensors = [o for o in garbage if torch.is_tensor(o) and o.is_cuda]
    frames = [f'{o.f_code.co_filename.rsplit("/", 2)[-1]}:{o.f_lineno}:{o.f_code.co_name}'
              for o in garbage if type(o).__name__ == 'frame']
    dict_keys = [sorted(map(str, o))[:12] for o in garbage
                 if isinstance(o, dict) and any(torch.is_tensor(v) and v.is_cuda for v in o.values())]
    return dict(types=Counter(type(o).__qualname__ for o in garbage).most_common(limit),
                cuda_tensors=len(tensors),
                largest=[(list(t.shape), str(t.dtype)) for t in sorted(tensors, key=lambda t: -t.numel())[:10]],
                frames=Counter(frames).most_common(limit), tensor_dict_keys=dict_keys[:limit])


def install_boundary_gc(algo, output, interval=16, freeze=False):
    """freeze: at the first boundary, after collecting, move every surviving object to
    the permanent generation (gc.freeze) so later full collections scan only objects
    created during training (startup objects live for the whole run)."""
    if interval < 1:
        raise ValueError('Require a positive GC interval')
    original = algo.train_epoch
    output = Path(output)
    first_epoch = int(algo.epoch_num)
    report = dict(interval_updates=interval, calls=0, collected_objects=0,
                  released_tensor_bytes=0, wall_seconds=0., freeze=freeze, frozen_objects=None)

    def train_epoch(*args, **kwargs):
        epoch = int(algo.epoch_num)
        if (epoch-first_epoch-1) % interval == 0:
            before = torch.cuda.memory_allocated()
            start = time.monotonic()
            if freeze and report['frozen_objects'] is not None:
                # Cheap after the freeze: keep the garbage for attribution, then free it.
                gc.set_debug(gc.DEBUG_SAVEALL)
                collected = gc.collect(2)
                gc.set_debug(0)
                garbage = list(gc.garbage)
                gc.garbage.clear()
                held = sum(o.untyped_storage().nbytes() for o in {id(t.untyped_storage()): t for t in garbage
                           if torch.is_tensor(t) and t.is_cuda}.values())
                if held > 2**30:
                    row = dict(epoch=epoch, frame=int(algo.frame), cuda_gib=held/2**30, **_attribute(garbage))
                    with (output/'boundary_gc_garbage.jsonl').open('a') as file:
                        file.write(json.dumps(row)+'\n')
                del garbage
                gc.collect(2)
            else:
                collected = gc.collect(2)
            if freeze and report['frozen_objects'] is None:
                gc.freeze()
                report['frozen_objects'] = gc.get_freeze_count()
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
    if freeze:
        # After the freeze a full collection costs milliseconds: also collect right after
        # the rollout, so garbage made during it is gone before the SAPG batch peak.
        play_steps = algo.play_steps

        def play_steps_then_collect(*args, **kwargs):
            result = play_steps(*args, **kwargs)
            if report['frozen_objects'] is not None:
                report['post_rollout_collected'] = report.get('post_rollout_collected', 0)+gc.collect(2)
            return result
        algo.play_steps = play_steps_then_collect
    return report
