#!/usr/bin/env python3
"""Pre-registered early stop and forgetting check of the M table fine-tune (reviews of 2026-10-28), from TensorBoard.

Inputs, per epoch: sgs/table_top_<lo>_<hi>/eval_success and .../eval_episodes = rolling evaluation success over the
last <= 3000 evaluation episodes of each table-top bin (1 in 16 envs; fresh validated resting starts; uniform table
heights; about 11-12 finished episodes per bin and epoch).

Reference: each bin's rolling value at the LAST critic warm-up epoch (--warmup, policy frozen). The warm-up is 300
epochs, so the windows then hold ~3000 frozen-policy episodes that finished after the start transient (episodes
end early only on failure, so the first ~38 epochs = one episode length are mostly failures).

FORGET (item 6): a bin touching 0.80 m below its reference by more than --forget points AND by more than the 95%
interval of the difference (binomial, both windows' episode counts), checked once the window has been refilled
since the warm-up (--refill epochs) -> narrow the range to 0.75-0.85 m, widen again on success.

STOP (item 7): epoch >= --min-epochs and the worst bin's slope is below --min-gain points per 200 epochs, now and
100 epochs earlier. Slope: least squares over the last --fit-epochs epochs of the worst-bin series (lowest of the
bins' rolling values).

Checkpoint (item 5): the 3 latest bests by BEST_METRIC=sgs_eval_table_worst (best/best_epoch<N>.pth: lowest bin,
mean as the tiebreak) are re-evaluated with scripts/eval_wholebody_checkpoint.py (deterministic leader, fresh
starts, condim 4, --table-height-range 0.10, ~3000 episodes per bin); the pick is the highest worst-bin success
there, mean as the tiebreak; it is compared with the frozen-M baseline of the same script (job 1631, pass 3).
Usage: table_stop_rule.py <run dir>
"""
import argparse
import math
import re
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def slope_per_200(y):
    """Least-squares slope of y (fraction per epoch) in points per 200 epochs."""
    x = np.arange(len(y), dtype=np.float64)
    return 100*200*np.polyfit(x, y, 1)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--refill', type=int, default=300, help='epochs after the warm-up before the forgetting check')
    p.add_argument('--min-epochs', type=int, default=1800, help='warm-up + 1500 training epochs')
    p.add_argument('--fit-epochs', type=int, default=600)
    p.add_argument('--min-gain', type=float, default=.5, help='points per 200 epochs')
    p.add_argument('--forget', type=float, default=3., help='points')
    a = p.parse_args()
    ea = EventAccumulator(a.run.rstrip('/')+'/summaries', size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    bins = sorted({m.group(1) for t in tags for m in [re.match(r'sgs/table_top_([\d.]+_[\d.]+)/eval_success$', t)] if m})
    if not bins:
        print('NO_TABLE_BINS')
        return
    read = lambda t: np.array([e.value for e in ea.Scalars(t)])
    succ = {b: read(f'sgs/table_top_{b}/eval_success') for b in bins}
    eps = {b: read(f'sgs/table_top_{b}/eval_episodes') for b in bins}
    n = min(len(v) for v in succ.values())
    wall = np.array([e.wall_time for e in ea.Scalars(f'sgs/table_top_{bins[0]}/eval_success')])
    epoch_s = float(np.median(np.diff(wall[-200:]))) if len(wall) > 2 else float('nan')
    print(f'EPOCH {n}  ({epoch_s:.1f} s/epoch)')
    if n < a.warmup:
        print(f'WARMUP ({n}/{a.warmup}): ' + ' | '.join(f'{b}: {100*succ[b][n-1]:.1f}% of {int(eps[b][n-1])}' for b in bins))
        return
    w0 = a.warmup-1
    rows, forget = [], []
    for b in bins:
        ref, n_ref, now, n_now = succ[b][w0], eps[b][w0], succ[b][n-1], eps[b][n-1]
        half = 1.96*math.sqrt(ref*(1-ref)/max(n_ref, 1)+now*(1-now)/max(n_now, 1))
        delta = now-ref
        rows.append(f'{b}: {100*now:5.1f}% (n {int(n_now)})  ref {100*ref:5.1f}% (n {int(n_ref)})  delta {100*delta:+5.1f} +- {100*half:.1f}')
        lo, hi = (float(x) for x in b.split('_'))
        if lo <= .80 <= hi and n >= a.warmup+a.refill and -delta > max(a.forget/100, half):
            forget.append(b)
    print('\n'.join(rows))
    if n < a.warmup+a.refill:
        print(f'forgetting check from epoch {a.warmup+a.refill} (window refilled since the warm-up)')
    elif forget:
        print('FORGET (bins at 0.80 m below the reference beyond the interval): ' + ', '.join(forget))
    worst = np.min(np.stack([succ[b][:n] for b in bins]), 0)
    if n < max(a.min_epochs, a.warmup+a.fit_epochs+100):
        print(f'CONTINUE (epoch {n} < {max(a.min_epochs, a.warmup+a.fit_epochs+100)})')
        return
    now_slope = slope_per_200(worst[n-a.fit_epochs:n])
    before = slope_per_200(worst[n-100-a.fit_epochs:n-100])
    print(f'worst bin {100*worst[-1]:.1f}%  slope {now_slope:+.2f} pts/200 epochs (100 epochs ago {before:+.2f})')
    print('STOP' if now_slope < a.min_gain and before < a.min_gain else 'CONTINUE')


if __name__ == '__main__':
    sys.exit(main())
