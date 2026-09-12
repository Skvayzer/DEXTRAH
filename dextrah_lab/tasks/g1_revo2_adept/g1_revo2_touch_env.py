"""Opt-in BPS + physical tactile sensing + optional estimated motor torques.

No change to default task, actions, rewards, fabrics or PCA. Distal compliance
is an explicit opt-in physics approximation, not calibrated BrainCo elastomer.
"""
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from dextrah_lab.g1_adept.contact import FINGERS, TIP_BODIES, rotate_to_local
from dextrah_lab.g1_adept.contact_geometry import load_pad_geometry
from dextrah_lab.g1_adept.tactile_forces import read_pad_contact_forces
from dextrah_lab.g1_adept.touch_frames import sensor_frames
from dextrah_lab.g1_adept.touch_observations import TouchObservationConfig, TouchObservationModel
from .g1_revo2_bps_env import G1Revo2BpsEnv, G1Revo2BpsEnvCfg


@configclass
class Revo2TouchCfg:
    enabled: bool = True
    arm_torques: bool = False
    sensor_hz: float = 70.
    publish_hz: float = 70.
    latency_s: float = 0.
    filter_tau_s: float = 0.
    noise_std_n: float = 0.
    gain_range: tuple = (1., 1.)
    bias_std_n: float = 0.
    dropout_probability: float = 0.
    force_scale_n: float = 25.
    torque_scale_nm: float = 40.
    material_override: bool = True  # False: read contacts without changing source physics.
    compliant: bool = True
    stiffness_n_m: float = 10000.  # provisional, NOT fitted to the recording
    damping_ns_m: float = 10.
    static_friction: float = 1.0
    dynamic_friction: float = 0.8
    max_contacts: int = 64
    touch_description: str = '/data1/users/konstantin.smirnov/DEXTRAH-ADEPT/dextrah_lab/assets/revo2_description'

    def observation_config(self):
        return TouchObservationConfig(**{k: getattr(self, k) for k in TouchObservationConfig.__dataclass_fields__})


@configclass
class G1Revo2TouchEnvCfg(G1Revo2BpsEnvCfg):
    touch: Revo2TouchCfg = Revo2TouchCfg()


class G1Revo2TouchEnv(G1Revo2BpsEnv):
    def _extra_touch_partners(self):
        return []

    def _setup_scene(self):
        started = time.monotonic()
        clone = self.scene.clone_environments
        def clone_with_reporting(*args, **kwargs):
            if self.cfg.touch.enabled and not self.cfg.touch.material_override:
                stage = sim_utils.get_current_stage()
                found = []
                for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot')):
                    if prim.GetName() in TIP_BODIES and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
                        found.append(prim.GetName())
                if sorted(found) != sorted(TIP_BODIES):
                    raise RuntimeError(f'Contact-report source bodies mismatch: {found}')
                print(f'TOUCH_SOURCE_REPORTING_READY before_clone bodies={len(found)}',flush=True)
            return clone(*args, **kwargs)
        # Local scene-instance hook only; never alter the external P2P module.
        self.scene.clone_environments = clone_with_reporting
        try:
            super()._setup_scene()
        finally:
            self.scene.clone_environments = clone
        print(f'TOUCH_SCENE_BASE_READY seconds={time.monotonic()-started:.1f}',flush=True)
        self.touch_sensors = []
        if not self.cfg.touch.enabled:
            return
        stage = sim_utils.get_current_stage()
        # One rigid body per environment per pattern is required by this PhysX
        # version. Enumerate every collision-enabled non-robot body; robot self
        # collisions are disabled in this task, so they cannot generate touch.
        self.touch_filter_paths = []
        root = stage.GetPrimAtPath('/World/envs/env_0')
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and '/Robot/' not in path:
                self.touch_filter_paths.append(path.replace('env_0','env_.*',1))
        self.touch_filter_paths += self._extra_touch_partners()
        if not self.touch_filter_paths:
            raise RuntimeError('No tactile collision partners were found')
        if self.cfg.assets.robot_self_collision:
            raise ValueError('Self-contact sensing requires an explicit robot-pair implementation')
        self._touch_paths = []
        for body in TIP_BODIES:
            paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'))
                     if p.GetName() == body and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(paths) != 1:
                raise RuntimeError(f'Missing physical distal body {body}: {paths}')
            self._touch_paths.append(paths[0])
            expression = paths[0].replace('env_0', 'env_.*', 1)
            sensor = ContactSensor(ContactSensorCfg(prim_path=expression,
                update_period=0., track_pose=True, track_contact_points=True,
                max_contact_data_count_per_prim=self.cfg.touch.max_contacts,
                filter_prim_paths_expr=self.touch_filter_paths))
            self.scene.sensors[f'revo2_touch_{body}'] = sensor
            self.touch_sensors.append(sensor)
        if self.cfg.touch.material_override:
            self._bind_compliant_material(stage)
        else:
            # Reporting is observational. Do not bind a material or change
            # friction/compliance/contact offsets in the continuation experiment.
            self.touch_material_binding_count = 0
            # P2P factorized clones inherit env_0 (copy_from_source=False).
            # Author the shared source first: redundant writes on 120k distal
            # bodies otherwise cause minutes of serial USD recomposition.
            for source in self._touch_paths:
                body = stage.GetPrimAtPath(source)
                api = PhysxSchema.PhysxContactReportAPI(body)
                if not body.HasAPI(PhysxSchema.PhysxContactReportAPI) or api.GetThresholdAttr().Get() != 0.:
                    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr().Set(0.)
            inherited, overrides = 0, 0
            for env_path in self.scene.env_prim_paths:
                for source in self._touch_paths:
                    body = stage.GetPrimAtPath(source.replace('/World/envs/env_0', env_path, 1))
                    api = PhysxSchema.PhysxContactReportAPI(body)
                    if not body.HasAPI(PhysxSchema.PhysxContactReportAPI) or api.GetThresholdAttr().Get() != 0.:
                        api = PhysxSchema.PhysxContactReportAPI.Apply(body)
                        api.CreateThresholdAttr().Set(0.)
                        overrides += 1
                    else:
                        inherited += 1
                    if not body.HasAPI(PhysxSchema.PhysxContactReportAPI) or api.GetThresholdAttr().Get() != 0.:
                        raise RuntimeError(f'Contact-report inheritance failed at {body.GetPath()}')
            print(f'TOUCH_REPORTING_VERIFIED bodies={inherited+overrides} inherited={inherited} '
                  f'fallback_overrides={overrides} material_bindings=0 '
                  f'setup_seconds={time.monotonic()-started:.1f}',flush=True)

    def _bind_compliant_material(self, stage):
        cfg = self.cfg.touch
        if cfg.stiffness_n_m <= 0 or cfg.damping_ns_m < 0 or not 0 <= cfg.dynamic_friction <= cfg.static_friction:
            raise ValueError('Invalid tactile material parameters')
        material_path = '/World/Revo2TouchMaterial'
        material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=cfg.static_friction,
            dynamic_friction=cfg.dynamic_friction, friction_combine_mode='multiply',
            compliant_contact_stiffness=cfg.stiffness_n_m if cfg.compliant else 0.,
            compliant_contact_damping=cfg.damping_ns_m if cfg.compliant else 0.)
        material_cfg.func(material_path, material_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        self.touch_material_binding_count = 0
        # Shared material, no per-clone material explosion. Whole distal collider
        # is compliant; pad-only sensing is geometrically cropped below.
        for env_path in self.scene.env_prim_paths:
            for source in self._touch_paths:
                path = source.replace('/World/envs/env_0', env_path, 1)
                body = stage.GetPrimAtPath(path)
                PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr().Set(0.)
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
                        self.touch_material_binding_count += 1

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._touch_ready = False
        self._touch_last_step = 0
        super().__init__(cfg, render_mode, **kwargs)
        self.touch_model = TouchObservationModel(self.num_envs, cfg.sim.dt, self.device, cfg.touch.observation_config())
        self.touch_raw = torch.zeros(self.num_envs, 5, 3, device=self.device)
        self.touch_contact_w = torch.zeros_like(self.touch_raw)
        self.touch_force_w = torch.zeros_like(self.touch_raw)
        self.touch_friction_w = torch.zeros_like(self.touch_raw)
        self.touch_link_force_w = torch.zeros_like(self.touch_raw)
        # Resolve from the actual imported P2P asset root, just as its scene
        # importer does. Our source may live in an immutable Slurm snapshot
        # under outputs/; checkout-relative sibling assumptions fail there.
        from isaacsimenvs.tasks.play.utils.scene_utils import _resolve_asset_path
        self.g1_urdf = Path(_resolve_asset_path(cfg.assets.robot_urdf))
        if cfg.touch.enabled and not self.g1_urdf.is_file():
            raise FileNotFoundError(f'Tactile source URDF not found: {self.g1_urdf}')
        if cfg.touch.enabled:
            self.pad_geometry = load_pad_geometry(self.g1_urdf, Path(cfg.touch.touch_description))
            self.pad_frames = torch.as_tensor(sensor_frames(self.g1_urdf, self.pad_geometry), device=self.device)
            self.pad_tensors = [tuple(torch.as_tensor(x, device=self.device, dtype=torch.float32)
                               for x in (p.pad_to_distal[:3, 3], p.pad_to_distal[:3, :3], p.bounds))
                               for p in self.pad_geometry]
            self.touch_body_ids = [self.robot.body_names.index(x) for x in TIP_BODIES]
            for sensor in self.touch_sensors:
                if sensor.data.force_matrix_w.shape != (self.num_envs, 1, len(self.touch_filter_paths), 3):
                    raise RuntimeError(f'Unsafe all-contact filter shape: {sensor.data.force_matrix_w.shape}')
            # Play2Perfect overwrites material friction after scene initialization.
            # Reapply ONLY distal shapes after it, never the marker-body list.
            if cfg.touch.material_override:
                self._restore_distal_friction()
            else:
                self.touch_resolved_materials = self.robot.root_physx_view.get_material_properties().clone()
        self._touch_extra_dim = cfg.touch.observation_config().dimension
        cfg.observation_space += self._touch_extra_dim
        cfg.state_space += self._touch_extra_dim
        for key, size in (('policy', cfg.observation_space), ('critic', cfg.state_space)):
            self.single_observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space['policy'], self.num_envs)
        self.state_space = gym.vector.utils.batch_space(self.single_observation_space['critic'], self.num_envs)
        self._touch_ready = True
        print(f'TOUCH_OPTION_READY actor={cfg.observation_space} critic={cfg.state_space} '
              f'actions=13 touch={cfg.touch.enabled} arm_motor_estimates={cfg.touch.arm_torques} '
              f'publish_hz={cfg.touch.publish_hz} material_override={cfg.touch.material_override}', flush=True)

    def _restore_distal_friction(self):
        view = self.robot.root_physx_view
        materials = view.get_material_properties()
        start = 0
        self.touch_shape_indices = []
        for name, path in zip(view.shared_metatype.link_names, view.link_paths[0]):
            count = self.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
            if name in TIP_BODIES:
                materials[:, start:start+count, 0] = self.cfg.touch.static_friction
                materials[:, start:start+count, 1] = self.cfg.touch.dynamic_friction
                # PhysX represents compliant force-spring stiffness as negative
                # restitution. P2P's material setter resets restitution to zero,
                # disabling that spring unless restored here. Damping is kept
                # from the shared USD material authored before initialization.
                materials[:, start:start+count, 2] = (-self.cfg.touch.stiffness_n_m
                                                     if self.cfg.touch.compliant else 0.)
                self.touch_shape_indices.extend(range(start, start+count))
            start += count
        if start != view.max_shapes or not self.touch_shape_indices:
            raise RuntimeError('Distal material shape mapping failed')
        view.set_material_properties(materials, torch.arange(self.num_envs, device='cpu', dtype=torch.int64))
        self.touch_resolved_materials = view.get_material_properties()[:, self.touch_shape_indices].clone()
        expected = materials[:, self.touch_shape_indices]
        torch.testing.assert_close(self.touch_resolved_materials, expected)
        print('TOUCH_DISTAL_MATERIAL', self.touch_resolved_materials[0].tolist(), flush=True)

    def capture_touch(self, step=None):
        if not self._touch_ready:
            return
        step = self._sim_step_counter if step is None else step
        if step <= self._touch_last_step:
            return
        if step != self._touch_last_step + 1:
            raise RuntimeError('Tactile acquisition skipped physics steps')
        self._touch_last_step = step
        if self.cfg.touch.enabled:
            for i, sensor in enumerate(self.touch_sensors):
                data = sensor.data
                p, q = data.pos_w[:, 0], data.quat_w[:, 0]
                row = read_pad_contact_forces(sensor.contact_physx_view, self.cfg.sim.dt,
                                             p, q, *self.pad_tensors[i])
                # Do not expose partner-specific force or identity to actor.
                force = row.total_pairs_w.sum(1)
                local = rotate_to_local(q, force) @ self.pad_frames[i]
                local[:, 0].clamp_min_(0)
                self.touch_raw[:, i] = local
                self.touch_force_w[:, i] = force
                self.touch_friction_w[:, i] = row.friction_pairs_w.sum(1)
                self.touch_link_force_w[:, i] = (row.reconstructed_normal_pairs_w + row.reconstructed_friction_pairs_w).sum(1)
                weights = row.normal_load_n
                self.touch_contact_w[:, i] = (row.normal_centroids_w * weights[...,None]).sum(1) / weights.sum(1).clamp_min(1e-12)[:,None]
                # Fail on missing partners (e.g. ground): never silently feed
                # target-only sensing as though it represented the real pad.
                torch.testing.assert_close(row.reconstructed_normal_pairs_w.sum(1),
                                           data.net_forces_w[:,0], atol=.03, rtol=.02)
        self.touch_model.advance(self.touch_raw)

    def arm_motor_torques(self):
        # ImplicitActuator computes an APPROXIMATE clipped PD motor effort;
        # not a measured six-axis reaction wrench or exact solver drive force.
        return self.robot.data.applied_torque[:, self._canonical_joint_ids_lab[:7]]

    def _touch_observation(self):
        return self.touch_model.observation(self.arm_motor_torques())

    def _apply_action(self):
        self.capture_touch(self._sim_step_counter - 1)
        super()._apply_action()

    def _get_dones(self):
        self.capture_touch()
        return super()._get_dones()

    def _get_observations(self):
        obs = super()._get_observations()
        if not self._touch_ready:
            return obs
        extra = self._touch_observation()
        return {k: torch.cat((v, extra), -1) for k, v in obs.items()}

    def _get_rewards(self):
        reward = super()._get_rewards()
        if 'final_observation' in self.extras:
            final = self.extras['final_observation']
            final['critic'] = torch.cat((final['critic'], self._touch_observation()), -1)
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._touch_ready:
            self.touch_model.reset(env_ids)
            self.touch_raw[env_ids] = 0
