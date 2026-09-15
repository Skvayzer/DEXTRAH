#!/usr/bin/env python3
"""Summarize only the recorded navigation robot, not other reposing environments."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    args = parser.parse_args()
    directory = args.recording
    meta = json.loads((directory/'metadata.json').read_text())
    if not meta['completed'] or meta['experiment']['type'] != 'brush_navigation_transfer':
        raise ValueError('Require a complete navigation-transfer recording')
    if meta['experiment'].get('navigation_only'):
        raise ValueError('This analysis requires the manipulation policy to be active')
    with np.load(directory/'trajectory.npz', allow_pickle=False) as values:
        trace = dict(values)
    if len(trace['time_s']) != meta['frames']:
        raise ValueError('Frame count mismatch')
    command = np.stack([trace['transfer_cmd_vx'], trace['transfer_cmd_vy']], axis=-1)
    measured = np.stack([trace['transfer_measured_vx'], trace['transfer_measured_vy']], axis=-1)
    if not np.isfinite(command).all() or not np.isfinite(measured).all():
        raise ValueError('Nonfinite command or measured velocity')
    if not np.allclose(np.diff(trace['time_s']), 1/meta['fps'], atol=1e-6):
        raise ValueError('Unexpected capture clock')
    speed = np.linalg.norm(command, axis=-1)
    moving = speed > .01
    projected = np.full(len(speed), np.nan)
    projected[moving] = np.sum(command[moving]*measured[moving], axis=-1)/speed[moving]
    summary = dict(checkpoint_sha256=meta['checkpoint_sha256'],
        controlled_env=meta['env_id'], seconds=meta['seconds'],
        counts=meta['transfer_report']['counts'], completed_attempts=int(meta['metrics']['episodes']),
        includes_ongoing_final_attempt=True, robot_falls=meta['robot_falls'],
        object_fall_resets=meta['metrics']['fall'], numerical_failures=meta['numerical_failures'],
        highest_zero_based_waypoint=int(trace['transfer_waypoint'].max()),
        nonzero_command_seconds=float(moving.sum()/meta['fps']),
        nonzero_speed_range_m_s=[float(speed[moving].min()), float(speed[moving].max())] if moving.any() else None,
        mean_velocity_along_command_m_s=float(projected[moving].mean()) if moving.any() else None,
        opposite_command_sample_fraction=float((projected[moving] < 0).mean()) if moving.any() else None,
        contact_hold_seconds=float(trace['transfer_contact_hold'].sum()/meta['fps']),
        velocity_summary_note='Includes transitions and falls; not a steady-state walking-speed estimate',
        success_note='Reposing goal hits are not transfer arrivals, support or release')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.55]})
    path_ax, velocity_ax = axes
    size = np.asarray(meta['experiment']['table_size_m'])[:2]
    for key, color, label in [('table', '#aab7c1', 'Source'), ('receiving_table', '#629ec7', 'Receiver')]:
        xy = trace[key][0, :2]
        path_ax.add_patch(Rectangle(xy-size/2, *size, facecolor=color, edgecolor='white'))
        path_ax.text(*xy, label, ha='center', va='center', fontsize=9)
    for attempt, entry in enumerate(meta['transfer_report']['attempts']):
        indices = np.flatnonzero(trace['resets'] == attempt)
        if len(indices):
            xy = trace['robot'][indices, :2]
            path_ax.plot(xy[:, 0], xy[:, 1], color='#aa513f', alpha=.55, linewidth=1,
                         label='Measured pelvis paths' if attempt == 0 else None)
        route = entry.get('navigation_route_m')
        if route:
            route = np.asarray(route)
            path_ax.plot(route[:, 0], route[:, 1], '--', color='#2466a5', alpha=.3,
                         label='Requested waypoint routes' if attempt == 0 else None)
    path_ax.set(xlabel='World X (m)', ylabel='World Y (m)', title='All attempts; paths broken at resets')
    path_ax.set_aspect('equal', adjustable='datalim')
    path_ax.legend(loc='lower left', fontsize=8)
    velocity_ax.plot(trace['time_s'], speed, color='#2466a5', linewidth=1.2, label='Requested speed magnitude')
    velocity_ax.plot(trace['time_s'], projected, color='#aa513f', linewidth=.7, label='Measured velocity along command')
    velocity_ax.axhline(0, color='#333333', linewidth=.6)
    velocity_ax.set(xlabel='Simulation time (s)', ylabel='m/s', title='Negative = moving opposite the command')
    velocity_ax.legend(loc='upper left', fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
    counts = summary['counts']
    fig.suptitle(f"Two-table brush trial | {counts['grasped']} contact-confirmed lifts | "
                 f"{counts['navigation_arrived']} arrivals | {meta['robot_falls']} robot falls", fontsize=14)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(directory/f'navigation_analysis.{extension}', dpi=180, bbox_inches='tight')
    (directory/'navigation_analysis.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
