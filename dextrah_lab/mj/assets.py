"""Isaac Lab asset API (the subset the task uses) on top of mjlab entities.

The task code (Play2Perfect utilities and our G1 hooks) reads Isaac Lab
fields and calls Isaac Lab writers. These adapters expose exactly those, in
Isaac Lab's conventions:

* joints and bodies in the order the Isaac articulation reported (dumped from
  the PhysX run, `ISAAC_ORDER`), so any index-based code sees the same layout;
* `root_state_w` / `body_state_w` = link pose (wxyz) + centre-of-mass velocity
  in the world frame (Isaac Lab 2.2 definitions), from mjlab's EntityData;
* joint position targets are buffered and written to the position actuators
  by `write_data_to_sim`, like Isaac's implicit drives.
"""
import torch


def _ids(env_ids, n, device):
    if env_ids is None:
        return torch.arange(n, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class ArticulationData:
    def __init__(self, owner):
        self._o = owner

    @property
    def _d(self):
        return self._o.entity.data

    joint_names = property(lambda self: self._o.joint_names)
    body_names = property(lambda self: self._o.body_names)
    joint_pos = property(lambda self: self._d.joint_pos[:, self._o.joint_perm])
    joint_vel = property(lambda self: self._d.joint_vel[:, self._o.joint_perm])
    root_pos_w = property(lambda self: self._d.root_link_pos_w)
    root_quat_w = property(lambda self: self._d.root_link_quat_w)
    root_lin_vel_w = property(lambda self: self._d.root_com_lin_vel_w)
    root_ang_vel_w = property(lambda self: self._d.root_com_ang_vel_w)
    root_ang_vel_b = property(lambda self: self._d.root_com_ang_vel_b)
    projected_gravity_b = property(lambda self: self._d.projected_gravity_b)

    @property
    def root_state_w(self):
        return torch.cat((self._d.root_link_pose_w, self._d.root_com_vel_w), -1)

    @property
    def body_state_w(self):
        p = self._o.body_perm
        return torch.cat((self._d.body_link_pose_w[:, p], self._d.body_com_vel_w[:, p]), -1)

    body_pos_w = property(lambda self: self._d.body_link_pos_w[:, self._o.body_perm])
    body_quat_w = property(lambda self: self._d.body_link_quat_w[:, self._o.body_perm])

    @property
    def applied_torque(self):
        return self._o.sim.data.actuator_force[:, self._o._ctrl_ids]

    def __getattr__(self, name):
        # Static per-env arrays set by the owner (limits, defaults).
        try:
            return self._o.static[name]
        except KeyError:
            raise AttributeError(name) from None


class MjArticulation:
    def __init__(self, entity, sim, joint_names, body_names, static):
        self.entity, self.sim = entity, sim
        n, device = sim.num_envs, sim.device
        self.num_instances, self.device = n, device
        self._ALL_INDICES = torch.arange(n, device=device)
        self.joint_names, self.body_names = list(joint_names), list(body_names)
        ent_joints = list(entity.joint_names)
        ent_bodies = list(entity.body_names)
        missing = set(self.joint_names) ^ set(ent_joints)
        if missing or set(self.body_names) != set(ent_bodies):
            raise RuntimeError(f'Isaac/MuJoCo robot mismatch: joints {sorted(missing)}, '
                               f'bodies {sorted(set(self.body_names) ^ set(ent_bodies))}')
        self.joint_perm = torch.tensor([ent_joints.index(j) for j in self.joint_names], device=device)
        self.body_perm = torch.tensor([ent_bodies.index(b) for b in self.body_names], device=device)
        # Position actuators are authored in the spec (one per joint, named
        # after it); resolve their global ctrl indices in Isaac joint order.
        model = sim.mj_model
        self._ctrl_ids = torch.tensor([model.actuator(f'robot/{j}').id for j in self.joint_names], device=device)
        self.static = static
        self.data = ArticulationData(self)
        self._targets = static['default_joint_pos'].clone()

    def find_bodies(self, names):
        names = [names] if isinstance(names, str) else list(names)
        return [self.body_names.index(n) for n in names], names

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        rows = slice(None) if env_ids is None else _ids(env_ids, self.num_instances, self.device)[:, None]
        cols = slice(None) if joint_ids is None else torch.as_tensor(joint_ids, device=self.device)
        self._targets[rows, cols] = target

    def configure_hand_servos(self, servos):
        """Revo2 joints use velocity servos (robot.hand_servo); ctrl is a velocity."""
        names = self.joint_names
        self._servo_ids = torch.tensor([names.index(n) for n in servos], device=self.device)
        self._servo_gain = torch.tensor([servos[n]['kp']/servos[n]['kv'] for n in servos], device=self.device)
        self._servo_vmax = torch.tensor([servos[n]['vmax'] for n in servos], device=self.device)

    def configure_velocity_limits(self, limits, dt):
        """Emulate PhysX max joint velocities of the PD-driven (body) joints.

        PhysX enforces the limit inside its solver. Clamping MuJoCo velocities
        after a step instead removes the motion the contact solver needs to
        resolve penetration. Here the position-actuator setpoint moves toward
        the target by at most limit*dt per physics step. (Hand joints are
        velocity servos whose command is already capped at their limit.)
        """
        self._step_limit = (limits*dt).to(self.device)
        self._setpoint = self._targets.clone()

    def reset_setpoint(self, env_ids, values):
        if getattr(self, '_setpoint', None) is not None:
            self._setpoint[env_ids] = values

    def write_data_to_sim(self, advance=True):
        ctrl = self._targets
        if getattr(self, '_setpoint', None) is not None:
            if advance:
                self._setpoint += (self._targets-self._setpoint).clamp(-self._step_limit, self._step_limit)
            ctrl = self._setpoint.clone()
        if getattr(self, '_servo_ids', None) is not None:
            error = self._targets[:, self._servo_ids]-self.data.joint_pos[:, self._servo_ids]
            ctrl[:, self._servo_ids] = torch.maximum(torch.minimum(self._servo_gain*error, self._servo_vmax), -self._servo_vmax)
        self.sim.data.ctrl[:, self._ctrl_ids] = ctrl

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        ids = _ids(env_ids, self.num_instances, self.device)
        if joint_ids is not None:
            raise NotImplementedError('Partial joint writes are not used by the task')
        inverse = torch.empty_like(self.joint_perm)
        inverse[self.joint_perm] = torch.arange(len(self.joint_perm), device=self.device)
        # Isaac order -> entity order
        pos = torch.empty_like(position)
        vel = torch.empty_like(velocity)
        pos[:, self.joint_perm] = position
        vel[:, self.joint_perm] = velocity
        self.entity.write_joint_state_to_sim(pos, vel, env_ids=ids)

    def write_root_state_to_sim(self, root_state, env_ids=None):
        ids = _ids(env_ids, self.num_instances, self.device)
        self.entity.write_root_link_pose_to_sim(root_state[:, :7], env_ids=ids)
        self.entity.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=ids)


class RigidObjectData:
    def __init__(self, owner):
        self._o = owner

    @property
    def _d(self):
        return self._o.entity.data

    root_pos_w = property(lambda self: self._d.root_link_pos_w)
    root_quat_w = property(lambda self: self._d.root_link_quat_w)
    root_lin_vel_w = property(lambda self: self._d.root_com_lin_vel_w)
    root_ang_vel_w = property(lambda self: self._d.root_com_ang_vel_w)

    @property
    def root_state_w(self):
        return torch.cat((self._d.root_link_pose_w, self._d.root_com_vel_w), -1)

    @property
    def default_mass(self):
        return self._o.default_mass


class MjRigidObject:
    """Floating rigid body (object) or mocap body (table)."""

    def __init__(self, entity, sim, default_mass=None):
        self.entity, self.sim = entity, sim
        self.num_instances, self.device = sim.num_envs, sim.device
        self.data = RigidObjectData(self)
        self.default_mass = default_mass
        self._mocap = entity.is_mocap if hasattr(entity, 'is_mocap') else entity.indexing.mocap_id is not None
        self._wrench = None
        self._free_dof = None
        if not self._mocap:
            model = sim.mj_model
            joint = [j for j in range(model.njnt) if model.jnt_bodyid[j] == entity.indexing.root_body_id]
            self._free_dof = int(model.jnt_dofadr[joint[0]])

    def write_root_pose_to_sim(self, pose, env_ids=None):
        ids = _ids(env_ids, self.num_instances, self.device)
        if self._mocap:
            self.entity.write_mocap_pose_to_sim(pose, env_ids=ids)
        else:
            self.entity.write_root_link_pose_to_sim(pose, env_ids=ids)

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        if self._mocap:
            raise RuntimeError('Kinematic table has no velocity state')
        self.entity.write_root_com_velocity_to_sim(velocity, env_ids=_ids(env_ids, self.num_instances, self.device))

    def limit_angular_velocity(self, max_rad_s):
        """PhysX caps each dynamic rigid body's angular speed (physxRigidBody:
        maxAngularVelocity, schema default 5729.58 deg/s = 100 rad/s; Isaac's
        object bake does not override it). MuJoCo has no cap, and its explicit
        gyroscopic term diverges for fast-spinning anisotropic bodies (an
        ejected eraser went 47 -> 2.8e6 rad/s in free flight). Same norm clamp.
        """
        dof = self._free_dof
        w = self.sim.data.qvel[:, dof+3:dof+6]          # body-frame angular velocity
        speed = w.norm(dim=-1, keepdim=True)
        self.sim.data.qvel[:, dof+3:dof+6] = w*torch.clamp(max_rad_s/speed.clamp_min(1e-9), max=1.)

    def set_external_force_and_torque(self, forces, torques, env_ids=None, body_ids=None, is_global=False):
        if not is_global or env_ids is not None or body_ids is not None:
            raise NotImplementedError('The task only applies global wrenches to all envs')
        # Isaac applies the stored wrench at the COM on every physics step; MuJoCo's
        # xfrc_applied is a world-frame force/torque at the body COM and persists.
        self._wrench = (forces[:, 0], torques[:, 0])

    def write_data_to_sim(self):
        if self._wrench is not None:
            body = self.entity.indexing.root_body_id
            self.sim.data.xfrc_applied[:, body, :3] = self._wrench[0]
            self.sim.data.xfrc_applied[:, body, 3:] = self._wrench[1]
