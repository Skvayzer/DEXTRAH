"""Bounded, tensor-free memory telemetry around the real SAPG update stages."""
import gc
import json
from pathlib import Path
import time
import torch


class SapgMemoryTrace:
    def __init__(self, algo, output, gc_probe_epoch=0):
        self.algo = algo
        self.output = Path(output)
        self.started = time.monotonic()
        self.first_epoch = int(algo.epoch_num)
        self.gc_probe_epoch = gc_probe_epoch
        self.gc_probed = False

    def sample(self, stage, **extra):
        epoch = int(self.algo.epoch_num)
        if epoch-self.first_epoch > 512 and epoch % 16 and not extra:
            return
        stats = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        row = dict(stage=stage, epoch=epoch, frame=int(self.algo.frame),
            elapsed_s=time.monotonic()-self.started,
            allocated_gib=torch.cuda.memory_allocated()/2**30,
            reserved_gib=torch.cuda.memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            device_used_gib=(total-free)/2**30, device_free_gib=free/2**30,
            inactive_split_gib=stats.get('inactive_split_bytes.all.current', 0)/2**30,
            active_gib=stats.get('active_bytes.all.current', 0)/2**30,
            allocation_retries=stats.get('num_alloc_retries', 0),
            oom_count=stats.get('num_ooms', 0), gc_enabled=gc.isenabled(),
            gc_counts=gc.get_count(), **extra)
        with (self.output/'memory_trace.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')

    def wrap(self, name):
        original = getattr(self.algo, name)
        def measured(*args, **kwargs):
            self.sample(name+':before')
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                self.sample(name+':error', error=type(error).__name__)
                (self.output/'memory_error_summary.txt').write_text(torch.cuda.memory_summary())
                raise
            self.sample(name+':after')
            if name == 'train_epoch' and self.gc_probe_epoch and not self.gc_probed and self.algo.epoch_num >= self.gc_probe_epoch:
                self.gc_probed = True
                self.sample('gc_probe:before', probe=True)
                collected = gc.collect()
                self.sample('gc_probe:after', collected=collected)
                print('MEMORY_GC_PROBE '+json.dumps(dict(epoch=int(self.algo.epoch_num), collected=collected)), flush=True)
            return result
        setattr(self.algo, name, measured)


def install_memory_trace(algo, output, gc_probe_epoch=0):
    trace = SapgMemoryTrace(algo, output, gc_probe_epoch)
    for name in ('play_steps', 'augment_batch_for_mixed_expl', 'train_central_value', 'train_epoch'):
        trace.wrap(name)
    trace.sample('installed')
    return trace
