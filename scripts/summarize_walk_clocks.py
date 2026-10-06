#!/usr/bin/env python3
"""Tabulate Phase-0 empty-floor walking metrics across clock/physics conditions."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='outputs/walk_clocks_<job>')
    args = parser.parse_args()
    rows, summary = [], {}
    for folder in sorted(p for p in args.root.iterdir() if (p/'metadata.json').exists()):
        meta = json.loads((folder/'metadata.json').read_text())
        if not meta.get('completed'):
            print(f'{folder.name}: INCOMPLETE ({meta.get("error", "no error recorded")})')
            continue
        condition = f"{meta['clock']}/{meta['physics']}"
        for kind, robot in meta['robots'].items():
            for name, m in robot['tests'].items():
                vx, vy = m['mean_body_velocity_m_s']
                rows.append((condition, kind, name, m['command'], vx, vy, m['mean_body_yaw_rate_rad_s'],
                             m['direction_cosine'], m['first_fall_s'], m['planner']['inferences']))
                summary.setdefault(condition, {}).setdefault(kind, {})[name] = m
    print(f'{"condition":18} {"robot":8} {"case":16} {"command":>17} {"vx":>7} {"vy":>7} {"wz":>7} {"cos":>6} {"fall_s":>6} {"plans":>5}')
    for c, k, n, cmd, vx, vy, wz, cos, fall, plans in rows:
        cmd_s = '[' + ','.join(f'{v:+.2f}' for v in cmd) + ']'
        cos_s = '' if cos is None else f'{cos:+.2f}'
        fall_s = '' if fall is None else f'{fall:.1f}'
        print(f'{c:18} {k:8} {n:16} {cmd_s:>17} {vx:+.3f} {vy:+.3f} {wz:+.3f} {cos_s:>6} {fall_s:>6} {plans:>5}')
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
