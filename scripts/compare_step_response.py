#!/usr/bin/env python3
"""Stage 2: compare PD step responses, Isaac (PhysX) vs mjlab (MuJoCo Warp).

Normalised response r(t) = (q(t) - q(t_step)) / offset per env (envs 0-7),
averaged; reports 63% rise time, peak overshoot and the value at 0.5 s.
"""
import json
import sys
import numpy as np


def summary(entry):
    trace = np.asarray(entry['trace'], dtype=float)          # (steps, 8)
    dt, step = entry['dt'], 12
    r = (trace[step:]-trace[step-1])/entry['offset']
    r = np.nanmean(r, axis=1)
    t = np.arange(len(r))*dt
    rise = t[np.argmax(r >= .63)] if (r >= .63).any() else float('nan')
    return dict(rise_63_s=float(rise), peak=float(np.nanmax(r)), at_0_5s=float(r[round(.5/dt)]),
                at_1_0s=float(r[min(round(1./dt), len(r)-1)]), finite=bool(np.isfinite(trace).all()))


isaac = json.load(open(sys.argv[1]))
mj = json.load(open(sys.argv[2]))
print(f"{'group':18s} {'joint':32s} {'Isaac rise/peak/0.5s':>26s}   {'mjlab rise/peak/0.5s':>26s}")
for group in isaac:
    a, b = summary(isaac[group]), summary(mj[group])
    fmt = lambda s: f"{s['rise_63_s']:.3f}s/{s['peak']:.2f}/{s['at_0_5s']:.2f}" + ('' if s['finite'] else ' NaN')
    print(f"{group:18s} {isaac[group]['joint']:32s} {fmt(a):>26s}   {fmt(b):>26s}")
