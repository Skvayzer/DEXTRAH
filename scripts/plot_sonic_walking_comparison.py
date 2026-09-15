#!/usr/bin/env python3
"""Plot the low-speed versus documented-speed tests from saved measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = {}
    for folder in args.recordings:
        meta = json.loads((folder/'metadata.json').read_text())
        if not meta['completed']: raise ValueError('Incomplete run')
        for case in meta['cases']: runs[case['name']] = folder
    names = ['forward_020', 'forward_040', 'right_012_yaw90', 'right_040_yaw90']
    labels = ['Forward: command +0.20 m/s', 'Forward: command +0.40 m/s',
              'Right: command -0.12 m/s', 'Right: command -0.40 m/s']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for index, (ax, name, label) in enumerate(zip(axes.flat, names, labels)):
        for kind, color in [('upstream', '#2081bd'), ('revo2', '#dd6b20')]:
            with np.load(runs[name]/kind/name/'trajectory.npz', allow_pickle=False) as d:
                axis = 0 if index < 2 else 1
                t, velocity = d['time_s'], d['velocity_body'][:, axis]
                if kind == 'upstream': ax.plot(t, d['command'][:, axis], 'k--', lw=1.8, label='Command')
                ax.plot(t, velocity, color=color, alpha=.15, lw=.6)
                average = np.convolve(velocity, np.ones(5)/5, mode='valid')
                ax.plot(t[2:-2], average, color=color, lw=1.4, label=f'{kind}: 0.2 s average')
                mask = (t >= 4) & (t < 10)
                mean = float(velocity[mask].mean())
                ax.text(.02, .08 if kind == 'revo2' else .16, f'{kind} mean, 4-10 s: {mean:+.3f} m/s',
                        color=color, transform=ax.transAxes, fontsize=10)
        ax.set_title(label)
        ax.set_ylabel('Measured body-frame velocity (m/s)')
        ax.set_xlabel('Simulation time (s)')
        ax.grid(alpha=.2)
        ax.axvspan(4, 10, color='#d5e2e7', alpha=.18)
    axes[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('Frozen SONIC: low-speed stalling is not reversed control\n'
                 'Original G1 versus Revo2 G1 | empty floor | same frozen weights | 50 Hz control / 200 Hz physics')
    fig.text(.5, .005, 'Faint lines: raw velocity. Shading: reported measurement window. No falls or within-trial resets.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .925))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    fig.savefig(args.output.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
