"""Inference-only navigation-reference injection; training remains unchanged."""
import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat
from .brush_transfer_env import BrushTransferEnv
from .navigation_transfer import NavigationTransferDriver
from .navigation_reference import NavigationReference, yaw_of
from .contract import BODY_JOINTS


class NavigationBrushEnv(BrushTransferEnv):
    def __init__(self, cfg, *, planner_path, navigation_only=False, **kwargs):
        self.navigation = None
        self.navigation_only = navigation_only
        self._navigation_observation_step = None
        super().__init__(cfg, **kwargs)
        self.navigation = NavigationReference(planner_path)
        self._standing_reference_q = self._reference_q.clone()
        self._arm_reference_ids = [i for i, name in enumerate(BODY_JOINTS)
                                   if any(part in name for part in ('shoulder', 'elbow', 'wrist'))]

    def configure_transfer(self, index, vertices):
        super().configure_transfer(index, vertices)
        self.transfer = NavigationTransferDriver(vertices)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        if self.transfer_index is not None and (ids == self.transfer_index).any():
            if self.navigation is not None:
                self.navigation.frames = None
                self._reference_q[self.transfer_index] = self._standing_reference_q[self.transfer_index]
                self._reference_qd[self.transfer_index] = 0.
            self._navigation_observation_step = None

    def _update_transfer(self):
        index = self.transfer_index
        self.transfer.root_pose = self._local_pose(self.robot)
        self.transfer.root_velocity = self.robot.data.root_vel_w[index].detach().cpu().numpy().copy()
        super()._update_transfer()

    @torch.no_grad()
    def _wholebody_observation(self):
        reference = None
        active = self.transfer is not None and self.transfer.state is not None and (
            self.transfer.state['grasped'] or self.navigation_only)
        if active:
            index = self.transfer_index
            now = self._sim_step_counter*self.cfg.sim.dt
            if self._navigation_observation_step != self._sim_step_counter:
                root = self._local_pose(self.robot)
                if self.navigation.frames is None:
                    self.navigation.reset(root, self.robot.data.joint_pos[index, self._body_ids].cpu().numpy(), now)
                command = self.transfer.command
                if self.navigation_only:
                    # Empty-hand locomotion check: settle, strafe right, stop.
                    elapsed = float(self.episode_length_buf[index])*self.step_dt
                    command = np.array([0., -.12 if 2. <= elapsed < 10. else 0., 0.])
                    self.transfer.telemetry.update(cmd_vx=0., cmd_vy=float(command[1]), cmd_wz=0., navigation_active=int(command[1] != 0))
                q, qd, root_ref = self.navigation.reference(now, command, yaw_of(root[3:]))
                if not self.navigation_only:
                    # Preserve the trained arm-reference baseline, not physical arm
                    # joints. SONIC + SAPG still generate all body joint targets.
                    q[:, self._arm_reference_ids] = self._standing_reference_q[index, :, self._arm_reference_ids].cpu().numpy()
                    qd[:, self._arm_reference_ids] = 0.
                self._reference_q[index] = torch.as_tensor(q, device=self.device)
                self._reference_qd[index] = torch.as_tensor(qd, device=self.device)
                self._navigation_root_ref = torch.as_tensor(root_ref[:, 3:], device=self.device)
                self._navigation_observation_step = self._sim_step_counter
            reference = self._navigation_root_ref
        extra = super()._wholebody_observation()
        if reference is not None:
            index = self.transfer_index
            delta = quat_mul(quat_conjugate(yaw_quat(self.robot.data.root_quat_w[index:index+1])).expand(10, -1), reference)
            ori = matrix_from_quat(delta)[:, :, :2].reshape(1, 10, 6)
            # Parent uses expanded views: clone before changing one environment.
            self._reference_ori6 = self._reference_ori6.clone()
            self._reference_ori6[index:index+1] = ori
            token = self._sonic_reference.reference_tokens(self._reference_q[index:index+1],
                self._reference_qd[index:index+1], ori)
            extra[index:index+1, 930:] = token
        return extra

    def transfer_contract(self):
        value = super().transfer_contract()
        value.update(type='brush_navigation_transfer', navigation_only=self.navigation_only,
            robot_reference_change='NVIDIA planner from body-frame cmd_vel; preserves trained arm-reference baseline during carrying',
            goal_change='Body-relative carry target while navigating; receiver target only after arrival',
            navigation_planner=self.navigation.report(),
            action_override='Zero latent and fingers only for empty-hand locomotion diagnostic' if self.navigation_only else False,
            limitation='Scripted known-map waypoints, not autonomous perception; standing-trained adapter is unvalidated under walking references')
        return value
