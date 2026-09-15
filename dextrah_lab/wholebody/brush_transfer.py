"""Observation-only transfer checks and a scripted OBJECT-goal sequencer.

This is not a walking controller, hand controller, or learned planner. It never
writes robot/object states. A carry is explicitly distinct from a release.
"""
import numpy as np

TABLE_SIZE = (.475, .4, .3)
RECEIVER_OFFSET = (-.625, 0., 0.)  # robot faces -Y; -X is its right
PHASES = ('grasp', 'carry', 'lower', 'hold')


def move_goal(current, target, object_position, dt, speed=.035, max_lag=.08):
    current, target, position = map(lambda x: np.asarray(x, float), (current, target, object_position))
    if dt <= 0 or not np.isfinite(np.r_[current, target, position, dt]).all():
        raise ValueError('Invalid goal update')
    if np.linalg.norm(position-current) > max_lag:
        return current.copy()
    delta = target-current
    distance = np.linalg.norm(delta)
    return current + delta*min(1., speed*dt/max(distance, 1e-12))


def quat_matrix(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid wxyz orientation')
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class BrushTransferDriver:
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError('Need finite object geometry')
        self.attempts = []
        self.state = None

    def reset(self, goal_pose, receiver_pose, root_position):
        if self.state is not None:
            self.attempts.append(dict(self.state))
        self.goal = np.asarray(goal_pose, float).copy()
        self.receiver = np.asarray(receiver_pose, float).copy()
        self.initial_root = np.asarray(root_position, float).copy()
        self.phase = 0
        self.grasp_time = self.release_time = self.near_time = 0.
        self.state = dict(grasped=False, carried_over_receiver=False,
                          receiver_contact=False, placed_released=False,
                          max_root_xy_displacement_m=0., elapsed_s=0.)
        self.telemetry = dict(phase=0, receiver_force_n=0., robot_object_force_n=0.,
                              object_inside_receiver=False, receiver_distance_m=0.,
                              root_displacement_m=0., carried=0, placed=0)

    def update(self, pose, velocity, root_position, source_top, receiver_force, robot_force, dt):
        pose, velocity, root_position = map(np.asarray, (pose, velocity, root_position))
        if not np.isfinite(np.r_[pose, velocity, root_position, source_top, receiver_force, robot_force]).all():
            raise ValueError('Nonfinite transfer telemetry')
        points = self.vertices @ quat_matrix(pose[3:]).T + pose[:3]
        top = self.receiver[2]+TABLE_SIZE[2]/2
        inside = bool(np.all(np.abs(points[:, :2]-self.receiver[:2]) <= np.asarray(TABLE_SIZE[:2])/2-.01))
        lowest = float(points[:, 2].min())
        supported = inside and receiver_force > .05 and lowest < top+.025
        stable = np.linalg.norm(velocity[:3]) < .03 and np.linalg.norm(velocity[3:]) < .3
        released = supported and stable and robot_force < .02
        self.release_time = self.release_time+dt if released else 0.
        self.state['elapsed_s'] += dt
        self.state['max_root_xy_displacement_m'] = max(self.state['max_root_xy_displacement_m'],
            float(np.linalg.norm(root_position[:2]-self.initial_root[:2])))
        self.state['receiver_contact'] |= bool(supported)
        self.state['placed_released'] |= bool(self.state['grasped'] and self.release_time >= 1.)
        if self.state['grasped'] and inside and lowest > top+.015 and robot_force > .1:
            self.state['carried_over_receiver'] = True
        if self.phase == 0:
            lifted = lowest > source_top+.025 and robot_force > .1
            self.grasp_time = self.grasp_time+dt if lifted else 0.
            if self.grasp_time >= .5:
                self.state['grasped'] = True
                self.goal = pose.copy()  # continuity; freeze the measured grasp orientation
                offset_z = (self.vertices @ quat_matrix(pose[3:]).T)[:, 2].min()
                self.destination = np.r_[self.receiver[:2]+[0., .04], top-offset_z+.002]
                self.hover = self.destination.copy()
                self.hover[2] = max(1.03, self.destination[2]+.10, pose[2])
                self.phase = 1
        elif self.phase == 1:
            self.goal[:3] = move_goal(self.goal[:3], self.hover, pose[:3], dt)
            near = np.linalg.norm(pose[:3]-self.hover) < .04 and np.linalg.norm(self.goal[:3]-self.hover) < .005
            self.near_time = self.near_time+dt if near else 0.
            if self.near_time >= .5:
                self.phase = 2
        elif self.phase == 2:
            self.goal[:3] = move_goal(self.goal[:3], self.destination, pose[:3], dt, speed=.02)
            if np.linalg.norm(self.goal[:3]-self.destination) < .001:
                self.phase = 3
        self.telemetry = dict(phase=self.phase, receiver_force_n=float(receiver_force),
            robot_object_force_n=float(robot_force), object_inside_receiver=inside,
            receiver_distance_m=float(np.linalg.norm(pose[:2]-self.receiver[:2])),
            root_displacement_m=self.state['max_root_xy_displacement_m'],
            carried=int(self.state['carried_over_receiver']),
            placed=int(self.state['placed_released']))
        return self.goal.copy()

    def report(self):
        attempts = self.attempts + ([dict(self.state)] if self.state is not None else [])
        return dict(attempts=attempts, includes_ongoing_last_attempt=True,
            counts={key: sum(bool(a[key]) for a in attempts) for key in
                    ('grasped', 'carried_over_receiver', 'receiver_contact', 'placed_released')},
            release_definition='Inside receiver; receiver normal force >0.05 N; robot contact normal-force sum <0.02 N; linear speed <0.03 m/s and angular speed <0.3 rad/s continuously for 1 s',
            geometry_note='Visual mesh bounds gate footprint/height; physical contact is also required for placement',
            scripted_assistance='Object goal poses only; no action overrides or object teleportation')
