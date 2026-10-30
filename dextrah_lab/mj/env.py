"""The G1 + Revo2 frozen-SONIC reposing task on mjlab (MuJoCo Warp).

Same MDP as `G1SonicTouchEnv` on Isaac (run 1105 configuration). The class
chain on Isaac is

    DirectRLEnv -> PlayEnv -> G1Revo2DirectEnv -> G1Revo2BpsEnv
                -> G1Revo2TouchEnv -> G1SonicTouchEnv

This class reproduces each hook of that chain in the same order, calling the
SAME pinned Play2Perfect utilities and the same wholebody/g1_adept modules.
Only the simulator-facing parts are new: scene construction, state access
(`assets.py`), contact sensing (`tactile.py`) and the physics loop below
(Isaac Lab `DirectRLEnv.step` semantics).

Physics-loop note: MuJoCo's `mj_step` leaves kinematics (xpos, cvel) at the
pre-integration state, whereas PhysX reports post-step poses. Contacts and
forces of a step are read right after it (consistent with those pre-step
poses), and `forward()` then refreshes kinematics once per policy step
before terminations/rewards/observations, as Isaac does after its substeps.
"""
from copy import deepcopy
from pathlib import Path
from dextrah_lab.paths import remap
import json
import math

import gymnasium as gym
import numpy as np
import torch

from .compat import p2p, install
install()

from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat  # noqa: E402
from dextrah_lab.g1_adept.contact import FINGERS, TIP_BODIES, rotate_to_local  # noqa: E402
from dextrah_lab.g1_adept.contact_geometry import load_pad_geometry  # noqa: E402
from dextrah_lab.g1_adept.touch_frames import sensor_frames  # noqa: E402
from dextrah_lab.g1_adept.touch_observations import TouchObservationConfig, TouchObservationModel  # noqa: E402
from dextrah_lab.object_shape.bank import SHAPE_DIM, build_bank, map_features  # noqa: E402
from dextrah_lab.wholebody.actuators import body_motors  # noqa: E402
from dextrah_lab.wholebody.body_termination import classify_body_state, combine_terminations  # noqa: E402
from dextrah_lab.wholebody.contract import BODY_JOINTS, joint_indices, nominal_body_pose  # noqa: E402
from dextrah_lab.wholebody.source_actions import apply_frozen_latent_action, apply_wholebody_action  # noqa: E402
from dextrah_lab.wholebody.task_contract import BANK_SHA256, BODY_EXTRA_DIM, TASK_ACTOR_DIM, TASK_CRITIC_DIM  # noqa: E402
from dextrah_lab.wholebody.timed_history import TimedSonicHistory  # noqa: E402

from .assets import MjArticulation, MjRigidObject  # noqa: E402
from .objects import build_object_bank  # noqa: E402
from .robot import ContactParams  # noqa: E402
from .scene import SceneParams, TABLE_SIZE, build_scene_cfg, object_indices  # noqa: E402
from .tactile import ContactReader  # noqa: E402

REFERENCE = Path(__file__).with_name('isaac_reference.json')
RUN_1105 = remap('/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/0_frozen_sonic_sapg_touch_dexterous_1105')
# Fixed-body touch teacher (run-383 task, per-env object physics): Isaac dump + resolved config.
REFERENCE_FIXED = Path(__file__).with_name('isaac_reference_fixed.json')
RUN_FIXED = remap('/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/0_bps128_touch70_fixed_smoke_1233')

obs_utils = p2p('obs_utils')
reset_utils = p2p('reset_utils')
action_utils = p2p('action_utils')
reward_utils = p2p('reward_utils')
termination_utils = p2p('termination_utils')
logging_utils = p2p('logging_utils')
scene_utils = p2p('scene_utils')
generate_objects = p2p('generate_objects')
size_distributions = p2p('object_size_distributions')


class Cfg:
    """Attribute view of the resolved Isaac env config (env_resolved.yaml)."""

    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, Cfg(value) if isinstance(value, dict) else value)

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Cfg) else v) for k, v in vars(self).items()}

    def __copy__(self):
        new = Cfg({})
        new.__dict__.update(self.__dict__)
        return new


def load_run_cfg(run=RUN_1105):
    import yaml
    from dextrah_lab.paths import SERVER_WORKSPACE, WORKSPACE
    text = (run/'params/env_resolved.yaml').read_text().replace(SERVER_WORKSPACE, str(WORKSPACE))   # DEXTRAH_WORKSPACE
    return Cfg(yaml.unsafe_load(text))


class _Scene:
    """`env.scene` surface used by the task: env origins only (worlds do not overlap)."""

    def __init__(self, n, device):
        self.env_origins = torch.zeros(n, 3, device=device)


class G1SonicTouchMjEnv:
    def __init__(self, cfg, *, sonic, num_envs, device='cuda:0', scene_params=None,
                 nconmax=None, njmax=None, mujoco_cfg=None, velocity_limits=True, seed=None, fixed_body=False,
                 hand_surface=False):
        from mjlab.scene import Scene
        from mjlab.sim import Simulation, SimulationCfg
        self.cfg, self.device, self.num_envs = cfg, device, num_envs
        # fixed_body: the fixed-body touch teacher (Isaac G1Revo2TouchEnv is the
        # leaf class: right arm + hand, welded base, no SONIC layers).
        self._fixed_body = fixed_body
        if hand_surface and not fixed_body:
            raise ValueError('Whole body: enable the hand-surface observation with cfg.sonic_body.hand_surface_obs')
        if fixed_body:
            if sonic is not None:
                raise ValueError('The fixed-body touch task has no SONIC controller')
            isaac_dims = (cfg.observation_space, cfg.state_space)   # as the Isaac run resolved them
        elif cfg.sonic_body.controller_mode != 'frozen_pretrained_latent':
            raise ValueError('The mjlab port implements the frozen-SONIC latent interface only')
        reference = json.loads((REFERENCE_FIXED if fixed_body else REFERENCE).read_text())
        seed = cfg.seed if seed is None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # ---- PlayEnv.__init__ (before DirectRLEnv) ----------------------------
        cfg.observation_space = obs_utils.compute_obs_dim(cfg.obs.obs_list, cfg)
        cfg.state_space = obs_utils.compute_obs_dim(cfg.obs.state_list, cfg)
        # ---- G1SonicTouchEnv.__init__ (before parent) ---------------------------
        self._wholebody_ready = False
        self._sonic_reference = sonic
        self._frozen_latent = True
        cfg.action_space = 13
        self._touch_ready = False
        self._touch_last_step = 0
        # ---- DirectRLEnv.__init__ ----------------------------------------------
        self.physics_dt = cfg.sim.dt
        self.step_dt = cfg.sim.dt*cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s/self.step_dt)
        self.scene_params = scene_params or SceneParams()
        self._mj_scene = Scene(build_scene_cfg(
            num_envs, self.scene_params, fixed_body_reference=reference if fixed_body else None,
            robot_pose=(cfg.assets.robot_init_pos, cfg.assets.robot_init_rot)), device)
        mj_model = self._mj_scene.compile()
        mujoco_cfg = mujoco_cfg or {}
        sim_cfg = SimulationCfg(nconmax=nconmax, njmax=njmax)
        sim_cfg.mujoco.timestep = cfg.sim.dt
        for key, value in mujoco_cfg.items():
            setattr(sim_cfg.mujoco, key, value)
        self.sim = Simulation(num_envs, sim_cfg, mj_model, device)
        self._mj_scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.mj_model = self.sim.mj_model
        self.scene = _Scene(num_envs, device)
        self._setup_scene(reference)
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self._nan_capture = None
        self._sgs = None
        self._fast_step = False
        self._grasp_reward = None
        self._body_penalty = None
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.reset_terminated = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.extras = {}
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space['policy'] = gym.spaces.Box(-np.inf, np.inf, shape=(cfg.observation_space,))
        self.single_observation_space['critic'] = gym.spaces.Box(-np.inf, np.inf, shape=(cfg.state_space,))
        # ---- PlayEnv.__init__ (after DirectRLEnv) ---------------------------------
        reset_utils.allocate_state_buffers(self)
        # ---- G1Revo2DirectEnv.__init__ -----------------------------------------
        requested = set(cfg.assets.handle_head_types)
        expected = sum(d.type in requested for d in size_distributions.OBJECT_SIZE_DISTRIBUTIONS)*cfg.assets.num_assets_per_type
        if len(self._object_urdf_paths) != expected:
            raise RuntimeError(f'Object pool mismatch: expected {expected}, got {len(self._object_urdf_paths)}')
        self._episode_ever_lifted = torch.zeros(num_envs, device=device, dtype=torch.bool)
        # ---- G1Revo2BpsEnv.__init__ --------------------------------------------
        if (cfg.observation_space, cfg.state_space) != (92, 114):
            raise ValueError('Warm-start task requires the proven 92/114 P2P observation layout')
        bank, manifest = build_bank(self._object_urdf_paths, cfg.bps_cache_dir)
        indices = self._object_asset_index_per_env.cpu().numpy()
        self._bps_features = torch.as_tensor(map_features(bank, indices), device=device)
        self._bps_bank = bank
        self._bps_manifest = manifest
        cfg.observation_space += SHAPE_DIM
        cfg.state_space += SHAPE_DIM
        # ---- G1Revo2TouchEnv.__init__ ------------------------------------------
        tcfg = cfg.touch
        obs_cfg = TouchObservationConfig(**{k: getattr(tcfg, k) for k in TouchObservationConfig.__dataclass_fields__})
        if tcfg.material_override:
            raise ValueError('Run 1105 reads touch without a distal material override')
        self.touch_model = TouchObservationModel(num_envs, cfg.sim.dt, device, obs_cfg)
        self.touch_raw = torch.zeros(num_envs, 5, 3, device=device)
        self.touch_contact_w = torch.zeros_like(self.touch_raw)
        self.touch_force_w = torch.zeros_like(self.touch_raw)
        self.touch_friction_w = torch.zeros_like(self.touch_raw)
        self.touch_link_force_w = torch.zeros_like(self.touch_raw)
        self.g1_urdf = Path(scene_utils._resolve_asset_path(cfg.assets.robot_urdf))
        self.pad_geometry = load_pad_geometry(self.g1_urdf, Path(tcfg.touch_description))
        self.pad_frames = torch.as_tensor(sensor_frames(self.g1_urdf, self.pad_geometry), device=device)
        self.pad_tensors = [tuple(torch.as_tensor(x, device=device, dtype=torch.float32)
                                  for x in (p.pad_to_distal[:3, 3], p.pad_to_distal[:3, :3], p.bounds))
                            for p in self.pad_geometry]
        self.touch_body_ids = [self.robot.body_names.index(x) for x in TIP_BODIES]
        self._touch_extra_dim = obs_cfg.dimension
        cfg.observation_space += self._touch_extra_dim
        cfg.state_space += self._touch_extra_dim
        self._touch_ready = True
        if fixed_body:
            # ---- fixed-body touch task: no SONIC layers ------------------------------
            if (cfg.observation_space, cfg.state_space) != tuple(isaac_dims):
                raise ValueError(f'Fixed-body interface {(cfg.observation_space, cfg.state_space)} != Isaac {isaac_dims}')
            if self._bps_manifest['features_sha256'] != BANK_SHA256:
                raise ValueError('mjlab port generated a different object bank')
            # Opt-in hand-to-surface observation (wholebody/hand_surface.py), after touch.
            self._hand_surface, self.hand_surface_dim = None, 0
            if hand_surface:
                from dextrah_lab.wholebody.hand_surface import HAND_SURFACE_DIM, HandSurface, install_actor_pose_capture
                self._hand_surface, self.hand_surface_dim = HandSurface(self), HAND_SURFACE_DIM
                install_actor_pose_capture(obs_utils)
                cfg.observation_space += HAND_SURFACE_DIM
                cfg.state_space += HAND_SURFACE_DIM
            self.single_action_space = gym.spaces.Box(-np.inf, np.inf, shape=(cfg.action_space,))
            for key, size in (('policy', cfg.observation_space), ('critic', cfg.state_space)):
                self.single_observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        else:
            # ---- G1SonicTouchEnv.__init__ (after parent) -------------------------------
            if (cfg.observation_space, cfg.state_space) != (TASK_ACTOR_DIM, TASK_CRITIC_DIM):
                raise ValueError('Expected the trained BPS128 + touch interface')
            if self._bps_manifest['features_sha256'] != BANK_SHA256:
                raise ValueError('mjlab port generated a different object bank')
            names = self.robot.joint_names
            self._body_ids = torch.tensor(joint_indices(names, BODY_JOINTS), device=device)
            self._body_nominal = torch.as_tensor(nominal_body_pose(), device=device)
            self._body_scales = torch.tensor([m.action_scale for m in body_motors().values()], device=device)
            self._body_lower = self.robot.data.joint_pos_limits[:, self._body_ids, 0].clone()
            self._body_upper = self.robot.data.joint_pos_limits[:, self._body_ids, 1].clone()
            for k, index in enumerate(self._arm_joint_ids.tolist()):
                body_index = self._body_ids.tolist().index(index)
                self._body_lower[:, body_index] = self._arm_lower[:, k]
                self._body_upper[:, body_index] = self._arm_upper[:, k]
            self._body_center = (self._body_lower+self._body_upper)/2
            self._body_half_range = (self._body_upper-self._body_lower)/2
            self._last_body_action = torch.zeros(num_envs, 29, device=device)
            self._wholebody_action_queue = torch.zeros(num_envs, self._action_queue.shape[1], 70, device=device)
            self._last_latent_action = torch.zeros(num_envs, 64, device=device)
            self._last_decoded_sonic_action = torch.zeros(num_envs, 29, device=device)
            self._body_history = TimedSonicHistory(num_envs, self.step_dt, device)
            self._body_extra_step = None
            self._body_extra = None
            self._body_fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._body_table_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._body_table_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._body_numerical_failure = torch.zeros_like(self._body_fallen)
            self._body_falls_total = torch.zeros((), dtype=torch.long, device=device)
            self._body_numerical_failures_total = torch.zeros_like(self._body_falls_total)
            self._body_max_joint_speed = torch.zeros((), device=device)
            self._body_checked_transitions = 0
            self._reference_q = self.robot.data.default_joint_pos[:, self._body_ids, None].transpose(1, 2).expand(-1, 10, -1).clone()
            self._reference_qd = torch.zeros_like(self._reference_q)
            self._reference_heading = torch.tensor(cfg.assets.robot_init_rot, device=device).expand(num_envs, -1)
            cfg.action_space = 70
            low = np.r_[np.full(64, -np.inf), np.full(6, -1.)].astype(np.float32)
            high = np.r_[np.full(64, np.inf), np.full(6, 1.)].astype(np.float32)
            self.single_action_space = gym.spaces.Box(low, high)
            # Opt-in hand-to-surface observation (wholebody/hand_surface.py), after the body extra.
            self._hand_surface, self.hand_surface_dim = None, 0
            if getattr(cfg.sonic_body, 'hand_surface_obs', False):
                from dextrah_lab.wholebody.hand_surface import HandSurface, install_actor_pose_capture
                # Opt-in InterMimicGen extensions: arm-link and goal-surface vectors (appended).
                self._hand_surface = HandSurface(self, arm_links=getattr(cfg.sonic_body, 'hand_surface_arm_links', False),
                                                 goal=getattr(cfg.sonic_body, 'hand_surface_goal', False))
                self.hand_surface_dim = self._hand_surface.dim
                install_actor_pose_capture(obs_utils)
            cfg.observation_space = TASK_ACTOR_DIM+BODY_EXTRA_DIM+self.hand_surface_dim
            cfg.state_space = TASK_CRITIC_DIM+BODY_EXTRA_DIM+self.hand_surface_dim
            for key, size in (('policy', cfg.observation_space), ('critic', cfg.state_space)):
                self.single_observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
            if getattr(cfg.sonic_body, 'spawn_on_table', False):
                from dextrah_lab.wholebody.spawn import TableSpawn
                self._table_spawn = TableSpawn(self._object_urdf_paths, self._object_asset_index_per_env, device)
            self._wholebody_ready = True
        # ---- backend extras ------------------------------------------------------
        self._velocity_limits = velocity_limits
        from .robot import hand_servo
        from dextrah_lab.wholebody.actuators import body_motors as _motors
        self.robot.configure_hand_servos({n: hand_servo(n) for n in self.robot.joint_names if n not in _motors()})
        # opt-in (evaluation / experiments): DEXTRAH_HAND_SERVO=deadbeat caps the hand servo gain at 1/dt,
        # v_cmd = clip(err / dt, +-vmax): the trained 120 Hz servo is linearly unstable near its target and limit-cycles
        # (~8 mrad); see docs/neurosim_structured_plan.md and the testbed's TestbedSpec.hand_servo
        import os
        if os.environ.get('DEXTRAH_HAND_SERVO', 'velocity') == 'deadbeat':
            self.robot._servo_gain = torch.clamp(self.robot._servo_gain, max=1/cfg.sim.dt)
            print('HAND_SERVO deadbeat: gain capped at 1/dt =', 1/cfg.sim.dt, flush=True)
        elif os.environ.get('DEXTRAH_HAND_SERVO', 'velocity') != 'velocity':
            raise ValueError('DEXTRAH_HAND_SERVO must be velocity or deadbeat')
        if velocity_limits:
            self.robot.configure_velocity_limits(torch.as_tensor(reference['joint_vel_limits'], device=device), self.physics_dt)
        self.contact_stats = dict(max_active=0)
        print(f'MJ_SONIC_TOUCH_READY envs={num_envs} actor={cfg.observation_space} critic={cfg.state_space} '
              f'actions={self.single_action_space.shape[0]} nconmax={nconmax} njmax={njmax} naconmax={self.sim.wp_data.naconmax} '
              f'graphs={self.sim.use_cuda_graph}', flush=True)

    # ======================================================================
    # Scene
    # ======================================================================
    def _setup_scene(self, reference):
        cfg, n, device = self.cfg, self.num_envs, self.device
        entities = self._mj_scene.entities
        # Same procedural pool, order and env assignment as Isaac (seed 42, shuffled).
        import tempfile
        self._tmp_asset_dir = tempfile.mkdtemp(prefix='mj_play_assets_')
        paths, scales = generate_objects.generate_handle_head_urdfs(
            handle_head_types=tuple(cfg.assets.handle_head_types), num_per_type=cfg.assets.num_assets_per_type,
            out_dir=self._tmp_asset_dir, shuffle=cfg.assets.shuffle_assets)
        self._object_urdf_paths = [str(p) for p in paths]
        asset = torch.arange(n, device=device) % len(paths)
        self._object_asset_index_per_env = asset
        self._object_scale_per_env = torch.as_tensor(np.asarray(scales, dtype=np.float32), device=device)[asset]
        self._table_variant_scales = None
        # Per-world object geometry and inertials.
        bank = build_object_bank(self._object_urdf_paths)
        self.object_bank = bank
        body, geoms = object_indices(self.sim.mj_model)
        self.sim.expand_model_fields(('geom_size', 'geom_pos', 'geom_quat', 'geom_rbound', 'geom_aabb',
                                      'body_mass', 'body_ipos', 'body_iquat', 'body_inertia'))
        a = asset.cpu().numpy()
        m = self.sim.model
        t = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
        m.geom_size[:, geoms] = t(bank.size[a])
        # Body frame at the COM (PhysX integrates rigid bodies about the COM).
        # With the URDF link origin as body frame, MuJoCo's free joint couples
        # a fast spin into translation (origin velocity v_com - w x r): at the
        # 100 rad/s cap that launched objects at ~30 m/s in free flight. The
        # adapter reports and writes the URDF link pose (`com_offset`).
        com = t(bank.com[a])
        m.geom_pos[:, geoms] = t(bank.pos[a])-com[:, None]
        m.geom_quat[:, geoms] = t(bank.quat[a])
        m.geom_rbound[:, geoms] = t(bank.rbound[a])
        m.geom_aabb[:, geoms] = t(bank.aabb[a]).view(n, 4, 2, 3)
        m.body_mass[:, body] = t(bank.mass[a])
        m.body_ipos[:, body] = 0.
        m.body_iquat[:, body] = t(np.tile([1., 0., 0., 0.], (n, 1)))
        m.body_inertia[:, body] = t(bank.inertia[a])
        from mjlab.managers.event_manager import RecomputeLevel
        self.sim.recompute_constants(RecomputeLevel.set_const)
        # Isaac-API assets.
        joint_names, body_names = reference['joint_names'], reference['body_names']
        lim = t(reference['joint_pos_limits'])
        static = dict(joint_pos_limits=lim.expand(n, -1, -1).clone(),
                      joint_vel_limits=t(reference['joint_vel_limits']).expand(n, -1).clone(),
                      default_joint_pos=t(reference['default_joint_pos']).expand(n, -1).clone(),
                      default_joint_vel=torch.zeros(n, len(joint_names), device=device),
                      default_root_state=t(reference['default_root_state']).expand(n, -1).clone())
        self.robot = MjArticulation(entities['robot'], self.sim, joint_names, body_names, static)
        self.object = MjRigidObject(entities['object'], self.sim, default_mass=t(bank.mass[a])[:, None], com_offset=com)
        self.table = MjRigidObject(entities['table'], self.sim)
        self.goal_viz = scene_utils._TensorGoalViz(n, device)
        # Contact reader (fingertip pads, monitored body-table links).
        from dextrah_lab.wholebody.source_scene import table_contact_monitored_links
        mj = self.sim.mj_model
        bid = lambda name: int(mj.body(name).id)
        robot_bodies = [bid(f'robot/{b}') for b in body_names]
        monitored = [] if self._fixed_body else table_contact_monitored_links(sorted(body_names))   # Isaac sensor order
        self.body_table_names = monitored
        self._contacts_cfg = dict(
            finger_bodies=[bid(f'robot/{b}') for b in TIP_BODIES],
            monitored_bodies=[bid(f'robot/{b}') for b in monitored],
            object_body=body, table_body=bid('table/box'),
            ground_geom=int(mj.geom('ground').id), robot_bodies=robot_bodies)
        self.body_table_sensors = ([None]*len(monitored) if not self._fixed_body
                                   and self.cfg.sonic_body.body_table_contact_termination else [])

    def _init_contacts(self):
        self.contacts = ContactReader(self.sim, self.sim.mj_model, pad_tensors=self.pad_tensors, **self._contacts_cfg)
        self.contacts.active_only = self._fast_step

    def enable_grasp_reward(self, scale, energy_scale=0.):
        """Opt-in reward terms after InterMimicGen (dextrah_lab/wholebody/grasp_reward.py):
        scale * contact-prioritized grasp score with thumb-finger opposition (scale 0: computed
        and logged only), minus energy_scale * bounded object-acceleration energy. Call before
        the first reset (the contact reader then also measures object-only pad/palm forces)."""
        from dextrah_lab.wholebody.grasp_reward import GraspReward
        if hasattr(self, 'contacts'):
            raise RuntimeError('Enable the grasp reward before the first reset')
        self._contacts_cfg['palm_body'] = int(self.sim.mj_model.body(f'robot/{self.robot.body_names[self._palm_body_id]}').id)
        self._grasp_reward = GraspReward(self, scale, energy_scale)
        return self._grasp_reward

    def enable_body_penalty(self, action_scale=0., action_bound=8., fall_penalty=0.):
        """Opt-in penalties on out-of-range frozen-SONIC body actions and robot falls
        (dextrah_lab/wholebody/body_penalty.py). Whole body only."""
        from dextrah_lab.wholebody.body_penalty import BodyPenalty
        if self._fixed_body:
            raise ValueError('Body penalties apply to the frozen-SONIC whole body')
        self._body_penalty = BodyPenalty(self, action_scale, action_bound, fall_penalty)
        return self._body_penalty

    def enable_fast_step(self):
        """Opt-in faster env.step with the same simulation and observations (checked by
        scripts/check_fast_env_step.py against the default path):

        * after the physics substeps and after resets, refresh only the derived kinematic
          quantities the task reads (poses, COM, body velocities) instead of a full
          sim.forward(); collision, constraints and the solver are recomputed from the
          state by the next physics step anyway (the solver warm start is written only by
          the step);
        * the touch contact reader processes the active contacts only (not the whole pool);
        * frozen SONIC runs its reference encoder once per policy step (the observation's
          reference tokens and the action decode share it) and only the g1_dyn decoder.
        """
        import mujoco_warp as mjwarp
        from mujoco_warp import EnableBit
        m = self.sim.wp_model
        if m.ntendon or m.nflex or (m.opt.enableflags & EnableBit.SLEEP):
            raise ValueError('The kinematics refresh does not cover tendons, flexes or sleeping')
        self._mjwarp = mjwarp
        self._fast_step = True
        if hasattr(self, 'contacts'):
            self.contacts.active_only = True
        if self._sonic_reference is not None:
            self._sonic_reference.check_fast_path()

    def _refresh_kinematics(self):
        """Kinematics (body/geom/site poses, xipos, subtree_com, cdof) and com_vel (cvel)
        for the current qpos/qvel: everything the task reads between steps."""
        import warp as wp
        m, d = self.sim.wp_model, self.sim.wp_data
        with wp.ScopedDevice(self.sim.wp_device):
            self._mjwarp.fwd_kinematics(m, d)
            self._mjwarp.com_vel(m, d)

    def _refresh(self):
        if self._fast_step:
            self._refresh_kinematics()
        else:
            self.sim.forward()

    # ======================================================================
    # DirectRLEnv.step / reset
    # ======================================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed)
        if not hasattr(self, 'contacts'):
            self._init_contacts()
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)
        self._write_data_to_sim(advance=False)
        self.sim.forward()
        return self._get_observations(), self.extras

    def step(self, action):
        action = action.to(self.device)
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self._write_data_to_sim()
            if self._nan_capture is not None:
                self._nan_capture.before_step(self._sim_step_counter)
            self.sim.step()
            if self._nan_capture is not None:
                self._nan_capture.after_step(self._sim_step_counter)
            self._after_physics_step()
        self._refresh()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        if self._sgs is not None:
            self._sgs.after_step()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self._write_data_to_sim(advance=False)
            self._refresh()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def enable_sgs(self, settings, state=None):
        """Opt-in Success Guided Sampling of episode configurations (dextrah_lab/sgs)."""
        from dextrah_lab.sgs.mj_task import MjSgsTask
        self._sgs = MjSgsTask(self, settings, state=state)
        return self._sgs

    def enable_nan_capture(self, out_dir, depth=4, max_abs_qvel=None):
        """Diagnostic only: record the first non-finite (or, with max_abs_qvel,
        exploding) world (mj/nan_capture.py)."""
        from .nan_capture import NanCapture
        self._nan_capture = NanCapture(self, out_dir, depth=depth, max_abs_qvel=max_abs_qvel)
        return self._nan_capture

    def _write_data_to_sim(self, advance=True):
        # advance=False after resets: no physics step follows the write yet.
        self.robot.write_data_to_sim(advance=advance)
        self.object.write_data_to_sim()

    # physxRigidBody:maxAngularVelocity schema default (5729.58 deg/s), unchanged by Isaac's object bake.
    PHYSX_MAX_ANGULAR_VELOCITY = 100.

    def _after_physics_step(self):
        self.object.limit_angular_velocity(self.PHYSX_MAX_ANGULAR_VELOCITY)
        self.capture_touch(self._sim_step_counter)

    # ======================================================================
    # Hooks, in Isaac MRO order
    # ======================================================================
    def _reset_idx(self, env_ids):
        # DirectRLEnv: scene.reset, episode length
        self.episode_length_buf[env_ids] = 0
        if self._sgs is not None:
            # SGS: record finished episodes, pick configurations and objects.
            self._sgs.before_reset(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        # PlayEnv
        reset_utils.reset_env_state(self, torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        # Drive setpoints restart at the reset joint targets (as PhysX drives do).
        self.robot.reset_setpoint(env_ids, self._cur_targets[env_ids])
        # G1Revo2DirectEnv
        self._episode_ever_lifted[env_ids] = False
        # G1Revo2TouchEnv
        if self._touch_ready:
            self.touch_model.reset(env_ids)
            self.touch_raw[env_ids] = 0
        if self._sgs is not None and not self._wholebody_ready:
            # SGS (fixed body): replace the rho reset of training envs by their configuration.
            self._sgs.after_reset(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        # G1SonicTouchEnv
        if not self._wholebody_ready:
            return
        ids = env_ids
        if getattr(self, '_table_spawn', None) is not None:
            from dextrah_lab.wholebody.spawn import apply_table_spawn
            apply_table_spawn(self, torch.as_tensor(ids, device=self.device, dtype=torch.long))
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self.scene.env_origins[ids]
        root[:, 7:] = 0
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self._body_history.reset(ids)
        self._last_body_action[ids] = (self._cur_targets[ids][:, self._body_ids]-self._body_nominal)/self._body_scales
        self._wholebody_action_queue[ids] = 0
        self._last_latent_action[ids] = 0
        self._last_decoded_sonic_action[ids] = 0
        self._body_fallen[ids] = False
        self._body_table_count[ids] = 0
        self._body_table_contact[ids] = False
        self._body_numerical_failure[ids] = False
        if self._sgs is not None:
            # SGS (whole body): after the standing reset, so a configuration also sets
            # the floating base, SONIC history and executed body action.
            self._sgs.after_reset(torch.as_tensor(ids, device=self.device, dtype=torch.long))

    def _pre_physics_step(self, actions):
        if self._fixed_body:
            action_utils.apply_action_pipeline(self, actions)
        else:
            apply_frozen_latent_action(self, actions, action_utils.apply_action_pipeline)
        action_utils.apply_wrench_dr(self)

    def _apply_action(self):
        # G1Revo2TouchEnv captures the previous physics step here; on this
        # backend that step was already captured right after it ran.
        self.capture_touch(self._sim_step_counter-1)
        target_ids = self._position_target_joint_ids
        self.robot.set_joint_position_target(self._cur_targets[:, target_ids],
                                             joint_ids=self._position_target_joint_ids_list)

    def capture_touch(self, step=None):
        if not self._touch_ready:
            return
        step = self._sim_step_counter if step is None else step
        if step <= self._touch_last_step:
            return
        if step != self._touch_last_step+1:
            raise RuntimeError('Tactile acquisition skipped physics steps')
        self._touch_last_step = step
        c = self.contacts.read()
        self._last_contacts = c
        q = self.sim.data.xquat[:, self.contacts.finger_bodies]
        force = c['normal_masked_w']+c['friction_masked_w']
        for i in range(5):
            local = rotate_to_local(q[:, i], force[:, i]) @ self.pad_frames[i]
            local[:, 0].clamp_min_(0)
            self.touch_raw[:, i] = local
        self.touch_force_w[:] = force
        self.touch_friction_w[:] = c['friction_masked_w']
        self.touch_link_force_w[:] = c['normal_all_w']+c['friction_all_w']
        self.touch_contact_w[:] = c['centroid_w']
        if 'body_table_force' in c:
            self._body_table_force_step = c['body_table_force']
        self.touch_model.advance(self.touch_raw)

    def _get_dones(self):
        if self._fixed_body:
            # G1Revo2TouchEnv -> PlayEnv. Non-finite states abort, as on the SONIC path.
            data = self.robot.data
            state = (data.joint_pos, data.joint_vel, self.object.data.root_state_w)
            if not bool(torch.stack([torch.isfinite(x).all() for x in state]).all()):
                if self._nan_capture is not None:
                    self._nan_capture.dump_env_state(self._sim_step_counter, state)
                raise RuntimeError('Non-finite simulation state; refusing to feed it to SAPG')
            self.capture_touch()
            termination_utils.update_tolerance_curriculum(self)
            obs_utils.compute_intermediate_values(self)
            return termination_utils.compute_terminations(self)
        data, cfg = self.robot.data, self.cfg.sonic_body
        # G1SonicTouchEnv (pre)
        state = (data.joint_pos, data.joint_vel, data.root_state_w, data.projected_gravity_b)
        try:
            self._body_fallen, self._body_numerical_failure, speed = classify_body_state(
                *state, self.scene.env_origins, self.object.data.root_state_w,
                minimum_height=cfg.minimum_pelvis_height, minimum_upright=cfg.minimum_upright_cosine,
                maximum_joint_speed=cfg.maximum_joint_speed, numerical_failure_mode=cfg.numerical_failure_mode)
        except RuntimeError:
            if self._nan_capture is not None:
                self._nan_capture.dump_env_state(self._sim_step_counter, state+(self.object.data.root_state_w,))
            raise
        self._body_falls_total += self._body_fallen.sum()
        self._body_numerical_failures_total += self._body_numerical_failure.sum()
        self._body_max_joint_speed = torch.maximum(self._body_max_joint_speed, speed.max())
        self._body_checked_transitions += self.num_envs
        # G1Revo2TouchEnv
        self.capture_touch()
        # PlayEnv
        termination_utils.update_tolerance_curriculum(self)
        obs_utils.compute_intermediate_values(self)
        task_terminated, truncated = termination_utils.compute_terminations(self)
        # G1SonicTouchEnv (post)
        if self.body_table_sensors:
            force = self._body_table_force_step
            pressing = force.amax(-1) > cfg.body_table_contact_force_n
            self._body_table_count = torch.where(pressing, self._body_table_count+1, torch.zeros_like(self._body_table_count))
            self._body_table_contact = self._body_table_count >= cfg.body_table_contact_steps
            self._body_table_force = force
            if cfg.body_table_contact_terminates:
                task_terminated = task_terminated | self._body_table_contact
                truncated = truncated & ~self._body_table_contact
        return combine_terminations(task_terminated, truncated, self._body_fallen, self._body_numerical_failure)

    def _get_rewards(self):
        # PlayEnv
        reward = reward_utils.compute_rewards(self)
        if self._grasp_reward is not None:
            # Opt-in InterMimicGen grasp score / object energy (enable_grasp_reward).
            extra, terms, logs = self._grasp_reward()
            self._grasp_logs = logs                     # per-env grasp score parts (carry v2 reads C_opp)
            reward = reward+extra
            self._reward_terms.update(terms)
            self._reward_terms['total_reward'] = reward
            lifted = self._lifted_object
            for k, v in logs.items():
                self.extras[f'grasp/{k}'] = v.mean()
            if bool(lifted.any()):
                self.extras['grasp/C_opp_lifted'] = logs['C_opp'][lifted].mean()
                self.extras['grasp/opposed_contact_rate_lifted'] = (logs['C_opp'][lifted] > .5).float().mean()
        if self._body_penalty is not None:
            # Opt-in SONIC action / fall penalties (enable_body_penalty).
            extra, terms, logs = self._body_penalty()
            reward = reward+extra
            self._reward_terms.update(terms)
            self._reward_terms['total_reward'] = reward
            for k, v in logs.items():
                self.extras[f'body/{k}'] = v.mean()
        logging_utils.log_step_metrics(self)
        if self.reset_buf.any():
            self.extras['final_observation'] = {'critic': obs_utils.build_critic_observation(self, reward=reward).clone()}
            self.extras['final_observation_mask'] = self.reset_buf.clone()
            self.extras['final_model_state_extras'] = {
                'lift_height': (0.05+self.object.data.root_pos_w[:, 2]-self.scene.env_origins[:, 2]-self._object_init_z).clone(),
                'near_goal_steps': (self._near_goal_steps.float()/max(float(self.cfg.termination.success_steps), 1.)).clone(),
                'success_tolerance': torch.full((self.num_envs,), float(self._current_success_tolerance), device=self.device),
                'action_queue': self._action_queue.flatten(start_dim=1).clone()}
        else:
            for key in ('final_observation', 'final_observation_mask', 'final_model_state_extras'):
                self.extras.pop(key, None)
        # G1Revo2DirectEnv
        self._episode_ever_lifted |= self._lifted_object
        self.extras['episode_final']['ever_lifted'] = self._episode_ever_lifted.float()
        self.extras['episode_final']['any_success'] = (self._successes > 0).float()
        # G1Revo2BpsEnv
        if 'final_observation' in self.extras:
            final = self.extras['final_observation']
            final['critic'] = torch.cat([final['critic'], self._bps_features], dim=-1)
        final = self.extras['episode_final']
        final['goals_reached_count'] = final['successes']
        final['any_goal_success_rate'] = final['any_success']
        final['ever_lifted_rate'] = final['ever_lifted']
        final['all_goals_completed_rate'] = final['all_goals_hit']
        # G1Revo2TouchEnv
        if 'final_observation' in self.extras:
            self.extras['final_observation']['critic'] = torch.cat(
                (self.extras['final_observation']['critic'], self._touch_observation()), -1)
        if self._fixed_body:
            if self._hand_surface is not None and 'final_observation' in self.extras:
                final = self.extras['final_observation']
                final['critic'] = torch.cat((final['critic'], self._hand_surface.observation(*self._true_object_pose())), -1)
            return reward
        # G1SonicTouchEnv
        self.extras['episode_final']['robot_fall'] = self._body_fallen.float()
        self.extras['episode_final']['numerical_failure'] = self._body_numerical_failure.float()
        if self.body_table_sensors and hasattr(self, '_body_table_force'):
            self.extras['episode_final']['done_body_table_contact'] = self._body_table_contact.float()
            touching = self._body_table_force > self.cfg.sonic_body.body_table_contact_force_n
            self.extras['body_table/any_link_touching'] = touching.any(-1).float().mean()
            self.extras['body_table/max_force_n'] = self._body_table_force.amax(-1).mean()
            for i, name in enumerate(self.body_table_names):
                if any(k in name for k in ('knee', 'pelvis', 'hip', 'torso')):
                    self.extras[f'body_table/touch_{name}'] = touching[:, i].float().mean()
        if 'final_observation' in self.extras:
            final = self.extras['final_observation']
            final['critic'] = torch.cat((final['critic'], self._wholebody_observation()), -1)
            if self._hand_surface is not None:
                final['critic'] = torch.cat((final['critic'], self._hand_surface.observation(*self._true_object_pose())), -1)
        return reward

    def _true_object_pose(self):
        return self.object.data.root_pos_w-self.scene.env_origins, self.object.data.root_quat_w

    def _hand_surface_observations(self):
        """(actor, critic) hand-to-surface observations: the actor sees the object pose its
        other object observations use (delayed/noised when that DR is on), the critic the true one."""
        true = self._true_object_pose()
        actor = true
        if self.cfg.domain_randomization.use_object_state_delay_noise:
            pose = getattr(self, '_actor_object_pose', None)
            if pose is None or pose[2] != self._sim_step_counter:
                raise RuntimeError('Actor object pose was not captured at this step')
            actor = pose[:2]
        critic = self._hand_surface.observation(*true)
        return (critic if actor is true else self._hand_surface.observation(*actor)), critic

    def _touch_observation(self):
        return self.touch_model.observation(None)

    def sonic_to_policy_body(self, sonic_action):
        targets = self._body_nominal+self._body_scales*sonic_action
        return (targets-self._body_center)/self._body_half_range

    def _body_terms(self):
        data = self.robot.data
        return (data.root_ang_vel_b, data.joint_pos[:, self._body_ids]-self._body_nominal,
                data.joint_vel[:, self._body_ids], self._last_body_action, data.projected_gravity_b)

    @torch.no_grad()
    def _wholebody_observation(self):
        if self._body_extra_step != self._sim_step_counter:
            proprio = self._body_history.push(*self._body_terms())
        else:
            self._body_history.initialize_fresh(*self._body_terms())
            proprio = self._body_history.value()
        heading_delta = quat_mul(quat_conjugate(yaw_quat(self.robot.data.root_quat_w)), self._reference_heading)
        ori6 = matrix_from_quat(heading_delta)[:, :, :2].reshape(self.num_envs, 1, 6).expand(-1, 10, -1)
        self._reference_ori6 = ori6
        if self._fast_step:
            # One encoder pass serves the reference tokens and the next action decode
            # (valid while _reference_ori6 is this ori6; the reference pose is fixed).
            tokens, self._sonic_latent = self._sonic_reference.reference_tokens_and_latent(
                self._reference_q, self._reference_qd, ori6)
            self._sonic_latent_ori6 = ori6
        else:
            tokens = self._sonic_reference.reference_tokens(self._reference_q, self._reference_qd, ori6)
        self._body_extra = torch.cat((proprio, tokens), -1)
        self._body_extra_step = self._sim_step_counter
        return self._body_extra

    def _get_observations(self):
        # PlayEnv
        obs = obs_utils.build_observations(self)
        # G1Revo2BpsEnv
        obs = {key: torch.cat([value, self._bps_features], dim=-1) for key, value in obs.items()}
        # G1Revo2TouchEnv
        if self._touch_ready:
            extra = self._touch_observation()
            obs = {k: torch.cat((v, extra), -1) for k, v in obs.items()}
        # G1SonicTouchEnv
        if not self._wholebody_ready:
            if self._hand_surface is not None:
                actor, critic = self._hand_surface_observations()
                obs = dict(policy=torch.cat((obs['policy'], actor), -1), critic=torch.cat((obs['critic'], critic), -1))
            return obs
        extra = self._wholebody_observation()
        obs = {k: torch.cat((v, extra), -1) for k, v in obs.items()}
        if self._hand_surface is not None:
            actor, critic = self._hand_surface_observations()
            obs = dict(policy=torch.cat((obs['policy'], actor), -1), critic=torch.cat((obs['critic'], critic), -1))
        return obs
