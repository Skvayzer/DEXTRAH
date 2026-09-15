"""Velocity command -> released SONIC motion planner -> body references.

No simulated root writes or joint-target writes. CPU-only ONNX inference keeps
the planner separate from the allocated training GPU. Conventions follow the
pinned GRAIL/imports/SONIC deployment planner, not a hand-authored gait.
"""
import hashlib
import time
from pathlib import Path
import numpy as np

PLANNER_SHA256 = '39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea'
# For each IsaacLab joint index, its index in MuJoCo qpos[7:].
ISAAC_FROM_MUJOCO = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                            4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])


def yaw_of(quaternion):
    w, x, y, z = np.asarray(quaternion, float)
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def slerp(a, b, weight):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a*b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0., 1.)
    angle = np.arccos(dot)
    weight = np.asarray(weight)[..., None]
    sine = np.maximum(np.sin(angle), 1e-9)
    spherical = (np.sin((1-weight)*angle)*a + np.sin(weight*angle)*b)/sine
    linear = (1-weight)*a+weight*b
    result = np.where(dot > .9995, linear, spherical)
    return result/np.linalg.norm(result, axis=-1, keepdims=True)


def sample_motion(frames, times):
    """Interpolate 50 Hz reference, holding endpoints; no future physics reads."""
    frames, times = np.asarray(frames), np.asarray(times)
    t = np.clip(times*50., 0., len(frames)-1)
    lo = np.floor(t).astype(int); hi = np.minimum(lo+1, len(frames)-1)
    w = t-lo
    result = (1-w[..., None])*frames[lo]+w[..., None]*frames[hi]
    result[..., 3:7] = slerp(frames[lo, 3:7], frames[hi, 3:7], w)
    return result


def velocity_inputs(command_body, heading, context, seed=1234, *, facing_heading=None):
    """ROS-style (vx, vy, yaw_rate) to planner speed/world directions.

    Important: target_vel=0 means DEFAULT SPEED upstream, not stop. Select
    IDLE explicitly for zero translation. The bridge integrates commanded
    yaw-rate separately: measured heading rotates velocity, not desired facing.
    """
    command = np.asarray(command_body, float)
    context = np.asarray(context, np.float32)
    if command.shape != (3,) or context.shape != (4, 36) or not np.isfinite(np.r_[command, context.ravel(), heading]).all():
        raise ValueError('Invalid navigation command/context')
    speed = float(np.linalg.norm(command[:2]))
    if speed > .3 or abs(command[2]) > .3:
        raise ValueError('Diagnostic command exceeds conservative speed limits')
    c, s = np.cos(heading), np.sin(heading)
    world = np.array([c*command[0]-s*command[1], s*command[0]+c*command[1], 0.])
    facing = heading if facing_heading is None else float(facing_heading)
    if not np.isfinite(facing):
        raise ValueError('Invalid desired facing heading')
    return dict(context_mujoco_qpos=context[None],
        target_vel=np.array([speed if speed > 1e-4 else -1.], np.float32),
        mode=np.array([1 if speed > 1e-4 else 0], np.int64),
        movement_direction=(world/max(speed, 1e-8)).astype(np.float32)[None],
        facing_direction=np.array([[np.cos(facing), np.sin(facing), 0.]], np.float32),
        height=np.array([-1.], np.float32), random_seed=np.array([seed], np.int64),
        has_specific_target=np.zeros((1, 1), np.int64),
        specific_target_positions=np.zeros((1, 4, 3), np.float32),
        specific_target_headings=np.zeros((1, 4), np.float32),
        allowed_pred_num_tokens=np.ones((1, 11), np.int64))


class NavigationReference:
    """Small-batch inference bridge, with upstream-style planned context/blend.

    References are 50 Hz; ten future samples use 100 ms spacing. Replanning
    is checked at 10 Hz, from the previous planned trajectory (not invented
    measurements), with 40 ms lookahead and a 160 ms crossfade.
    """
    def __init__(self, path, *, session=None):
        self.path = str(path)
        if session is None:
            with Path(path).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != PLANNER_SHA256:
                    raise ValueError('Unpinned navigation planner')
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.session = session
        self.frames = None
        self.inferences = 0
        self.wall_seconds = 0.

    def _infer(self, context, command, heading, facing_heading=None):
        feeds = velocity_inputs(command, heading, context, facing_heading=facing_heading)
        types = {'tensor(float)': np.float32, 'tensor(int64)': np.int64,
                 'tensor(int32)': np.int32}
        actual = {}
        for entry in self.session.get_inputs():
            if entry.name not in feeds or entry.type not in types:
                raise ValueError(f'Unknown planner input {entry.name}: {entry.type}')
            value = feeds[entry.name].astype(types[entry.type])
            if list(value.shape) != entry.shape:
                raise ValueError(f'Planner shape mismatch: {entry.name} {entry.shape}')
            actual[entry.name] = value
        start = time.monotonic()
        out = dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, actual)))
        self.wall_seconds += time.monotonic()-start
        self.inferences += 1
        length = int(np.asarray(out['num_pred_frames']).item())
        result = np.asarray(out['mujoco_qpos'])[0, :length].copy()
        if result.shape != (length, 36) or not 4 <= length <= 64 or not np.isfinite(result).all():
            raise ValueError('Invalid planner output')
        norms = np.linalg.norm(result[:, 3:7], axis=1)
        if np.max(np.abs(norms-1)) > .02:
            raise ValueError('Invalid planner quaternion')
        # Explicit 30 -> 50 Hz conversion, same floor and endpoint treatment as upstream.
        query = np.arange(int(np.floor(length*50/30)))*30/50
        lo = np.minimum(query.astype(int), length-1); hi = np.minimum(lo+1, length-1)
        w = query-lo
        frames = (1-w[:, None])*result[lo]+w[:, None]*result[hi]
        frames[:, 3:7] = slerp(result[lo, 3:7], result[hi, 3:7], w)
        return frames

    def reset(self, root_pose, body_q, now):
        root_pose, body_q = np.asarray(root_pose), np.asarray(body_q)
        pose = np.r_[root_pose, np.zeros(29)]
        pose[7+ISAAC_FROM_MUJOCO] = body_q
        self.frames = self._infer(np.repeat(pose[None], 4, axis=0), np.zeros(3), yaw_of(root_pose[3:]))
        self.origin = self.last_plan = self.last_tick = now
        self.last_call = now
        self.command = np.zeros(3)
        self.heading = yaw_of(root_pose[3:])
        self.desired_heading = self.heading
        self.world_velocity = np.zeros(2)

    def reference(self, now, command, heading):
        if self.frames is None:
            raise RuntimeError('Initialize navigation reference first')
        if now < self.last_call-1e-6:
            raise ValueError('Navigation clock moved backwards')
        command = np.asarray(command, float)
        if command.shape != (3,) or not np.isfinite(np.r_[command, heading, now]).all():
            raise ValueError('Invalid navigation command/time')
        if np.linalg.norm(command[:2]) > .3 or abs(command[2]) > .3:
            raise ValueError('Diagnostic command exceeds conservative speed limits')
        self.desired_heading += command[2]*(now-self.last_call)
        self.last_call = now
        if now-self.last_tick >= .1-1e-6:
            self.last_tick = now
            moving = np.linalg.norm(command[:2]) > 1e-4
            old_moving = np.linalg.norm(self.command[:2]) > 1e-4
            c, s = np.cos(heading), np.sin(heading)
            world_velocity = np.array([c*command[0]-s*command[1], s*command[0]+c*command[1]])
            changed = moving != old_moving or np.linalg.norm(world_velocity-self.world_velocity) > .015
            turn = abs(np.arctan2(np.sin(self.desired_heading-self.heading),
                                  np.cos(self.desired_heading-self.heading))) > .04
            # Upstream checks commands at 10 Hz but replans walking at 1 Hz.
            # Short horizons are endpoint-padded, not replanned every check:
            # repeatedly restarting a 160 ms crossfade at 100 ms stalls gait.
            if changed or turn or (moving and now-self.last_plan >= 1.-1e-6):
                old_times = now-self.origin + np.arange(4)/30 + .04
                context = sample_motion(self.frames, old_times)
                new = self._infer(context, command, heading, self.desired_heading)
                # Retain old reference through the 40 ms lead, then 160 ms blend.
                times = np.arange(len(new)+2)/50
                old = sample_motion(self.frames, now-self.origin+times)
                fresh = sample_motion(new, times-.04)
                weight = np.clip((times-.04)/.16, 0., 1.)
                blended = (1-weight[:, None])*old+weight[:, None]*fresh
                blended[:, 3:7] = slerp(old[:, 3:7], fresh[:, 3:7], weight)
                self.frames = blended
                self.origin = self.last_plan = now
                self.command = command.copy(); self.heading = self.desired_heading
                self.world_velocity = world_velocity
        query = now-self.origin+np.arange(10)*.1
        reference = sample_motion(self.frames, query)
        forward = sample_motion(self.frames, query+.02)
        q = reference[:, 7+ISAAC_FROM_MUJOCO]
        qd = (forward[:, 7+ISAAC_FROM_MUJOCO]-q)*50
        return q.astype(np.float32), qd.astype(np.float32), reference[:, :7].astype(np.float32)

    def report(self):
        return dict(planner_sha256=PLANNER_SHA256, planner_path=self.path,
                    inference_backend='ONNX Runtime CPU', inferences=self.inferences,
                    inference_wall_seconds=self.wall_seconds, reference_hz=50,
                    future_frames=10, future_spacing_s=.1, command_check_hz=10,
                    walking_periodic_replan_s=1.,
                    facing='integrated commanded yaw-rate, initialized at measured yaw',
                    context='previous planned reference, initialized from measured robot',
                    robot_state_writes=False)
