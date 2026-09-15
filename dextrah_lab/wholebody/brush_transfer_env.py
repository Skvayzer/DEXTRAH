"""Opt-in inference experiment; never registered as a training environment."""
import numpy as np
import torch
from pxr import Usd, UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
from .brush_transfer import BrushTransferDriver, TABLE_SIZE, RECEIVER_OFFSET, PHASES


class BrushTransferEnv(G1SonicTouchEnv):
    def __init__(self, cfg, **kwargs):
        self.transfer = None
        self.transfer_index = None
        self._transfer_last_step = None
        super().__init__(cfg, **kwargs)
        expected = (self.num_envs, 1, len(self.transfer_contact_paths), 3)
        if self.transfer_contact.data.force_matrix_w.shape != expected:
            raise RuntimeError(f'Incorrect object contact partner mapping: expected {expected}')
        if min(self.scene.cfg.env_spacing/2-abs(RECEIVER_OFFSET[0])-TABLE_SIZE[0]/2, .1) <= 0:
            raise ValueError('Receiving table exceeds the isolated environment footprint')

    def _extra_touch_partners(self):
        partners = super()._extra_touch_partners()
        self.receiving_table = RigidObject(RigidObjectCfg(
            prim_path='/World/envs/env_.*/ReceivingTable',
            spawn=sim_utils.CuboidCfg(size=TABLE_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=500.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .55, .72))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(RECEIVER_OFFSET[0], 0., .65))))
        self.scene.rigid_objects['receiving_table'] = self.receiving_table
        stage = sim_utils.get_current_stage()

        def bodies(root):
            return [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root))
                    if p.HasAPI(UsdPhysics.RigidBodyAPI)]

        receiver = bodies('/World/envs/env_0/ReceivingTable')
        objects = bodies('/World/envs/env_0/Object')
        robot = bodies('/World/envs/env_0/Robot')
        if len(receiver) != 1 or len(objects) != 1 or not robot:
            raise RuntimeError('Cannot resolve physical object/receiver/robot contact bodies')
        for env_path in self.scene.env_prim_paths:
            prim = stage.GetPrimAtPath(objects[0].replace('/World/envs/env_0', env_path, 1))
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
        expression = lambda path: path.replace('env_0', 'env_.*', 1)
        self.transfer_contact_paths = [expression(p) for p in receiver+robot]
        self.transfer_contact = ContactSensor(ContactSensorCfg(
            prim_path=expression(objects[0]), update_period=0.,
            filter_prim_paths_expr=self.transfer_contact_paths,
            max_contact_data_count_per_prim=128))
        self.scene.sensors['brush_transfer_contacts'] = self.transfer_contact
        print(f'BRUSH_TRANSFER_COLLISION_READY receiver={receiver} object={objects} robot_partners={len(robot)}', flush=True)
        return partners + [expression(receiver[0])]

    def configure_transfer(self, index, vertices):
        if self.transfer is not None:
            raise RuntimeError('Configure the experiment only once, before reset')
        self.transfer_index = int(index)
        self.transfer_ids = torch.tensor([index], device=self.device)
        self.transfer = BrushTransferDriver(vertices)

    def _local_pose(self, asset):
        index = self.transfer_index
        # The source task uses a tensor-only goal marker, not a RigidObject.
        pose = torch.cat((asset.data.root_pos_w[index], asset.data.root_quat_w[index])).detach().cpu().numpy().copy()
        pose[:3] -= self.scene.env_origins[index].detach().cpu().numpy()
        return pose

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        if hasattr(self, 'receiving_table'):
            pose = self.table.data.root_state_w[ids, :7].clone()
            pose[:, :3] += torch.tensor(RECEIVER_OFFSET, device=self.device)
            self.receiving_table.write_root_pose_to_sim(pose, env_ids=ids)
            self.receiving_table.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        if self.transfer is not None and (ids == self.transfer_index).any():
            self.transfer.reset(self._local_pose(self.goal_viz), self._local_pose(self.receiving_table),
                                self._local_pose(self.robot)[:3])
            self._transfer_last_step = None

    def _write_transfer_goal(self):
        goal = torch.as_tensor(self.transfer.goal, device=self.device, dtype=torch.float32).unsqueeze(0).clone()
        goal[:, :3] += self.scene.env_origins[self.transfer_ids]
        self.goal_viz.write_root_pose_to_sim(goal, env_ids=self.transfer_ids)

    def _get_dones(self):
        if self.transfer is not None and self.transfer.state is not None:
            if self._transfer_last_step != self._sim_step_counter:
                index = self.transfer_index
                forces = self.transfer_contact.data.force_matrix_w[index, 0]
                receiver_force = float(forces[0].norm())
                robot_force = float(forces[1:].norm(dim=-1).sum())
                self.transfer.update(self._local_pose(self.object),
                    self.object.data.root_vel_w[index].detach().cpu().numpy(),
                    self._local_pose(self.robot)[:3], self._local_pose(self.table)[2]+TABLE_SIZE[2]/2,
                    receiver_force, robot_force, self.step_dt)
                self._transfer_last_step = self._sim_step_counter
            self._write_transfer_goal()
        result = super()._get_dones()
        if self.transfer is not None and self.transfer.state is not None:
            # Source success bookkeeping may sample another reposing goal.
            # Preserve the experimental target before observations are built.
            self._write_transfer_goal()
        return result

    def transfer_contract(self):
        return dict(type='brush_table_transfer', table_size_m=TABLE_SIZE,
            receiver_offset_m=RECEIVER_OFFSET, table_edge_gap_m=.15,
            phase_names=PHASES, controlled_env=self.transfer_index,
            observation_change='None: policy sees object and goal, not receiving-table geometry',
            goal_change='After a contact-confirmed lift, slowly move goal toward receiver and lower; hold measured grasp orientation',
            robot_reference_change='None: original standing SONIC reference',
            action_override=False, object_teleportation=False, forced_release=False,
            optimizer_updates=0, physics_contact_paths=self.transfer_contact_paths,
            termination_change='Allow full recording duration; disable 50-goal reset; retain robot fall, object drop, hand-far and numerical safety resets',
            limitation='Goal sequencing is scripted; neither autonomous table perception nor a learned pick-and-place planner')
