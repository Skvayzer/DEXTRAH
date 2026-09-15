"""Known-map waypoint navigation and independent body-relative carry target."""
import numpy as np
from .brush_transfer import BrushTransferDriver
from .navigation_reference import yaw_of


def rotation2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class WaypointVelocity:
    def __init__(self, waypoints, heading, speed=.12):
        self.waypoints = np.asarray(waypoints, float)
        if (self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2
                or not len(self.waypoints) or not np.isfinite(self.waypoints).all()):
            raise ValueError('Invalid waypoint route')
        if not np.isfinite([heading, speed]).all() or not 0 < speed <= .8:
            raise ValueError('Invalid navigation heading/speed')
        self.heading, self.speed, self.index = heading, speed, 0
        self.command = np.zeros(3)
        self.done = False
        self.settle = 0.
        self.stopping = False
        self.arrival_radius = .12
        self.distance = float('inf')

    def update(self, root_pose, root_velocity, dt, hold=False):
        if dt <= 0 or not np.isfinite(np.r_[root_pose, root_velocity, dt]).all():
            raise ValueError('Invalid navigation feedback')
        error = self.waypoints[self.index]-np.asarray(root_pose)[:2]
        self.distance = float(np.linalg.norm(error))
        direction = error/max(self.distance, 1e-9)
        approach_speed = max(0., float(np.dot(root_velocity[:2], direction)))
        # Cruise at the requested working gait speed, then select true IDLE.
        # Proportional tapering near the goal re-enters the observed slow-gait
        # stall. Anticipate ~0.35 s stopping travel; verify the stop physically.
        if self.distance <= .055 + min(.125, .35*approach_speed):
            self.stopping = True
        self.command[:] = 0.
        if hold or self.done:
            self.settle = 0.
            return self.command.copy()
        if self.stopping:
            stopped = np.linalg.norm(root_velocity[:2]) < .06
            self.settle = self.settle+dt if stopped else 0.
            if self.settle >= 1.:
                if self.distance <= self.arrival_radius:
                    if self.index == len(self.waypoints)-1:
                        self.done = True
                    else:
                        self.index += 1
                        self.stopping = False
                else:
                    # Overshoot/under-travel: another bounded cruise segment,
                    # not an assumed arrival or a simulated-state correction.
                    self.stopping = False
                self.settle = 0.
            return self.command.copy()
        yaw = yaw_of(np.asarray(root_pose)[3:])
        self.command[:2] = rotation2(-yaw)@direction*self.speed
        # Zero yaw rate holds the planner's initialized facing, independently
        # of measured yaw drift. Planner reference crossfades smooth transitions.
        return self.command.copy()


class NavigationTransferDriver(BrushTransferDriver):
    def __init__(self, vertices, *, navigation_speed=.12):
        if not np.isfinite(navigation_speed) or not 0 < navigation_speed <= .8:
            raise ValueError('Invalid navigation speed')
        self.navigation_speed = float(navigation_speed)
        super().__init__(vertices)

    def reset(self, goal_pose, receiver_pose, root_position):
        super().reset(goal_pose, receiver_pose, root_position)
        self.navigator = None
        self.mode = 0  # 0 grasp, 1 navigation, 2 reach/lower at stationary destination
        self.state['navigation_arrived'] = False
        self.command = np.zeros(3)
        self.lost_contact_s = 0.
        self.root_pose = np.r_[root_position, 1., 0., 0., 0.]
        self.root_velocity = np.zeros(6)
        self.telemetry.update(cmd_vx=0., cmd_vy=0., cmd_wz=0., measured_vx=0., measured_vy=0.,
                              waypoint=0, navigation_active=0, navigation_arrived=0,
                              waypoint_distance_m=0., navigation_stopping=0, contact_hold=0)

    def update(self, pose, velocity, root_position, source_top, receiver_force, robot_force, dt):
        # Reuse the independently tested grasp/carry/support/release counters.
        # During navigation, restore the carry target after this goal sequencer.
        super().update(pose, velocity, root_position, source_top, receiver_force, robot_force, dt)
        yaw = yaw_of(self.root_pose[3:])
        if self.state['grasped'] and self.navigator is None:
            self.carry_xy = rotation2(-yaw)@(np.asarray(pose)[:2]-self.root_pose[:2])
            self.carry_z = float(pose[2])
            self.carry_quaternion = np.asarray(pose)[3:].copy()
            self.grasp_yaw = yaw
            destination_root = self.receiver[:2]+[0., .04]-rotation2(yaw)@self.carry_xy
            # Known free-space route: back out from source, move laterally,
            # approach receiver. This is not perception or obstacle avoidance.
            corridor_y = max(.66, self.root_pose[1], destination_root[1])
            route = [[self.root_pose[0], corridor_y],
                     [destination_root[0], corridor_y], destination_root]
            self.navigator = WaypointVelocity(route, yaw, speed=self.navigation_speed)
            self.state['navigation_route_m'] = self.navigator.waypoints.tolist()
            self.mode = 1
        if self.mode == 1:
            self.phase = 1
            self.lost_contact_s = self.lost_contact_s+dt if robot_force < .1 else 0.
            self.command = self.navigator.update(self.root_pose, self.root_velocity, dt,
                                                   hold=self.lost_contact_s > .1)
            self.goal[:2] = self.root_pose[:2]+rotation2(yaw)@self.carry_xy
            self.goal[2] = self.carry_z  # world height, independent of gait bobbing
            delta = yaw-self.grasp_yaw
            w, x, y, z = self.carry_quaternion
            c, s = np.cos(delta/2), np.sin(delta/2)
            self.goal[3:] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
            if self.navigator.done:
                self.mode = 2
                self.state['navigation_arrived'] = True
                self.command[:] = 0.
                self.hover[:2] = self.receiver[:2]+[0., .04]
                self.hover[2] = self.carry_z
        elif self.mode == 2:
            self.command[:] = 0.
        measured = rotation2(-yaw)@self.root_velocity[:2]
        self.telemetry.update(phase=self.phase, cmd_vx=float(self.command[0]),
            cmd_vy=float(self.command[1]), cmd_wz=float(self.command[2]),
            measured_vx=float(measured[0]), measured_vy=float(measured[1]),
            waypoint=self.navigator.index if self.navigator else 0,
            navigation_active=int(self.mode == 1), navigation_arrived=int(self.mode == 2),
            waypoint_distance_m=self.navigator.distance if self.navigator else 0.,
            navigation_stopping=int(self.navigator.stopping) if self.navigator else 0,
            contact_hold=int(self.lost_contact_s > .1))
        return self.goal.copy()

    def report(self):
        value = super().report()
        value.update(navigation='Known-map waypoint velocity commands to released SONIC planner',
            carry_target='Frozen grasp offset in current base-yaw XY frame; fixed world Z',
            release_override=False, navigation_speed_limit_m_s=self.navigation_speed,
            waypoint_control='Constant-speed cruise, predictive IDLE braking, measured stop for 1 s within 0.12 m; retry if outside tolerance',
            route=None if self.navigator is None else self.navigator.waypoints.tolist(),
            navigation_arrived=self.mode == 2)
        value['counts']['navigation_arrived'] = sum(bool(a.get('navigation_arrived')) for a in value['attempts'])
        value['scripted_assistance'] = 'Object goal poses and planner velocity commands only; no action overrides or robot/object teleportation'
        return value
