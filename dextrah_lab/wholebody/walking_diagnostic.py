"""Predeclared empty-floor tests and descriptive (not benchmark) metrics."""
import numpy as np


def cases():
    # Separate fresh episodes; no chained commands or outcome-based selection.
    return [dict(name=name, command=cmd, yaw_deg=yaw) for name, cmd, yaw in (
        ('idle', [0., 0., 0.], 0.),
        ('forward_020', [.2, 0., 0.], 0.),
        ('backward_020', [-.2, 0., 0.], 0.),
        ('forward_040', [.4, 0., 0.], 0.),
        ('backward_040', [-.4, 0., 0.], 0.),
        ('left_040', [0., .4, 0.], 0.),
        ('right_040', [0., -.4, 0.], 0.),
        ('right_012_yaw90', [0., -.12, 0.], -90.),
        ('right_040_yaw90', [0., -.4, 0.], -90.),
        ('turn_left', [0., 0., .2], 0.),
        ('turn_right', [0., 0., -.2], 0.),
    )]


def command_at(case, seconds):
    return np.array(case['command'] if 2. <= seconds < 10. else [0., 0., 0.], float)


def metrics(trace, case):
    t = trace['time_s']
    walking = (t >= 4.) & (t < 10.)  # exclude command onset transient
    stopping = t >= 12.
    velocity = trace['velocity_body']
    cmd = np.asarray(case['command'])
    yaw = np.deg2rad(case['yaw_deg'])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    start = int(np.searchsorted(t, 2.))
    end = min(int(np.searchsorted(t, 10.)), len(t)-1)
    displacement = (trace['root'][end, :2]-trace['root'][start, :2]) @ rotation
    desired = cmd[:2]
    speed = np.linalg.norm(desired)
    actual = velocity[walking, :2].mean(0)
    alignment = float(np.dot(actual, desired)/(np.linalg.norm(actual)*speed)) if speed and np.linalg.norm(actual) > 1e-6 else None
    return dict(command=cmd.tolist(), mean_body_velocity_m_s=actual.tolist(),
        mean_body_yaw_rate_rad_s=float(velocity[walking, 5].mean()),
        command_window_displacement_start_body_m=displacement.tolist(),
        velocity_rmse_m_s=float(np.sqrt(np.mean((velocity[walking, :2]-desired)**2))),
        direction_cosine=alignment,
        minimum_pelvis_height_m=float(trace['root'][:, 2].min()),
        minimum_upright_cosine=float(trace['upright'].min()),
        first_fall_s=float(t[np.flatnonzero(trace['fell'])[0]]) if trace['fell'].any() else None,
        stop_speed_m_s=float(np.linalg.norm(velocity[stopping, :2], axis=1).mean()),
        joint_tracking_rmse_rad=float(np.sqrt(np.mean((trace['joint_q'][walking]-trace['reference_q'][walking, 0])**2))),
        resets_during_trial=0, interpretation='Descriptive single-seed diagnostic; not a success-rate estimate')
