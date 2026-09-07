"""Opt-in contact diagnostics. Training registration/configuration is unchanged."""
from __future__ import annotations

import torch
from pathlib import Path
from pxr import PhysxSchema, Usd, UsdPhysics

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from dextrah_lab.g1_adept.contact import (
    CONTACT_CHANNELS, FINGERS, TIP_BODIES, FingertipContactFilter,
    adept_grasp_gate, rotate_to_local,
    aggregate_pad_contacts,
)
from dextrah_lab.g1_adept.contact_geometry import load_pad_geometry
from .g1_revo2_adept_env import G1Revo2AdeptEnv


class G1ContactDiagnosticEnv(G1Revo2AdeptEnv):
    """Same controller/scene, five contact views, and a labelled test fixture.

    Contact points on distal bodies are masked with Touch CAD windows. Neither
    the footprint nor frame axes are a calibrated Revo2 capacitive sensor.
    """

    probe_radius = .008

    def _setup_scene(self):
        super()._setup_scene()
        stage = get_current_stage()
        env0 = self.scene.env_prim_paths[0]

        def rigid_path(asset):
            root = stage.GetPrimAtPath(f"{env0}/{asset}")
            paths = [str(p.GetPath()) for p in Usd.PrimRange(root)
                     if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(paths) != 1:
                raise RuntimeError(f"Expected one rigid body in {asset}, found {paths}")
            return paths[0].replace(env0, "/World/envs/env_.*", 1)

        self.contact_filter_paths = [rigid_path("Object"), rigid_path("Table"),
                                     "/World/envs/env_.*/ContactProbe"]
        self.contact_probe = RigidObject(RigidObjectCfg(
            prim_path=self.contact_filter_paths[2],
            spawn=sim_utils.SphereCfg(
                radius=self.probe_radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=.001, rest_offset=0.),
                mass_props=sim_utils.MassPropertiesCfg(mass=.01),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., -5.)),
        ))
        self.scene.rigid_objects["contact_probe"] = self.contact_probe
        self.tip_sensors = []
        for finger, body in zip(FINGERS, TIP_BODIES):
            paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(f"{env0}/Robot"))
                     if p.GetName() == body and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(paths) != 1:
                raise RuntimeError(f"Missing/unmerged fingertip body {body}: {paths}")
            expression = paths[0].replace(env0, "/World/envs/env_.*", 1)
            matches = find_matching_prim_paths(expression)
            if len(matches) != self.num_envs:
                raise RuntimeError(f"Tip count mismatch for {body}: {len(matches)}")
            for path in matches:
                report = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path))
                report.CreateThresholdAttr().Set(0.)
            # Filtered contacts require ONE sensor body per environment.
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=expression, update_period=0., track_pose=True,
                track_contact_points=True, max_contact_data_count_per_prim=64,
                filter_prim_paths_expr=self.contact_filter_paths,
            ))
            self.scene.sensors[f"touch_{finger}"] = sensor
            self.tip_sensors.append(sensor)

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.touch_filter = FingertipContactFilter(self.num_envs, self.device)
        self.g1_urdf = Path(__file__).resolve().parents[3].parent / "play2perfect" / self.cfg.assets.robot_urdf
        description = Path(__file__).resolve().parents[2] / "assets/revo2_description"
        self.pad_geometry = load_pad_geometry(self.g1_urdf, description)
        self.pad_tensors = [tuple(torch.as_tensor(x, device=self.device, dtype=torch.float32)
                                 for x in (pad.pad_to_distal[:3, 3], pad.pad_to_distal[:3, :3], pad.bounds))
                            for pad in self.pad_geometry]
        self.tip_body_ids = [self.robot.body_names.index(name) for name in TIP_BODIES]
        for name, sensor in zip(FINGERS, self.tip_sensors):
            expected = (self.num_envs, 1, len(CONTACT_CHANNELS), 3)
            if sensor.data.force_matrix_w.shape != expected:
                raise RuntimeError(f"{name}: filtered force shape {sensor.data.force_matrix_w.shape}, expected {expected}")
        print(f"CONTACT_BINDINGS fingers={TIP_BODIES}, filters={self.contact_filter_paths}", flush=True)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "touch_filter"):
            self.touch_filter.reset(env_ids)

    def read_contacts(self, advance_filter=True):
        data = [sensor.data for sensor in self.tip_sensors]
        net = torch.stack([d.net_forces_w[:, 0] for d in data], dim=1)
        pairs = torch.stack([d.force_matrix_w[:, 0] for d in data], dim=1)
        points = torch.stack([d.contact_pos_w[:, 0] for d in data], dim=1)
        pos = torch.stack([d.pos_w[:, 0] for d in data], dim=1)
        quat = torch.stack([d.quat_w[:, 0] for d in data], dim=1)
        pad_data = [aggregate_pad_contacts(
            sensor.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt),
            pos[:, i], quat[:, i], *self.pad_tensors[i],
        ) for i, sensor in enumerate(self.tip_sensors)]
        pad_pairs = torch.stack([row[0] for row in pad_data], dim=1)
        pad_points = torch.stack([row[1] for row in pad_data], dim=1)
        pad_load = torch.stack([row[2] for row in pad_data], dim=1)
        reconstructed = torch.stack([row[3] for row in pad_data], dim=1)
        # Catch force-unit, normal-sign, or buffer-index mistakes immediately.
        torch.testing.assert_close(reconstructed, pairs, atol=.02, rtol=.01)
        if advance_filter:
            self.touch_filter.update(pad_pairs.sum(dim=2), self.cfg.sim.dt)
        valid_point = torch.isfinite(points).all(dim=-1) & (pairs.norm(dim=-1) > 1.e-6)
        clean_points = torch.where(valid_point[..., None], points, pos[:, :, None])
        local_points = rotate_to_local(
            quat[:, :, None].expand(-1, -1, len(CONTACT_CHANNELS), -1),
            clean_points - pos[:, :, None],
        )
        # This residual is UNCLASSIFIED, not a reliable self-contact label.
        # The training model disables hand self-collisions.
        residual = net - pairs.sum(dim=2)
        return dict(
            net_w=net, net_local=rotate_to_local(quat, net), pairs_w=pairs,
            points_w=clean_points, points_local=local_points, point_valid=valid_point,
            tip_pos_w=pos, tip_quat_w=quat, unclassified_w=residual,
            pad_pairs_w=pad_pairs, pad_points_w=pad_points, pad_load_n=pad_load,
            gate_all=adept_grasp_gate(net.norm(dim=-1)),
            gate_object=adept_grasp_gate(pairs[:, :, 0].norm(dim=-1)),
            gate_pad_object=adept_grasp_gate(pad_pairs[:, :, 0].norm(dim=-1)),
        )


def diagnostic_cfg(num_envs=12, device="cuda:0"):
    """Small viewer; retain solver/timestep/controller, reduce buffer capacities."""
    from .g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg
    cfg = G1Revo2AdeptEnvCfg()
    cfg.seed = 42
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.assets.num_assets_per_type = 1
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 19
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
    cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2 ** 16
    cfg.sim.physx.gpu_collision_stack_size = 2 ** 24
    return cfg
