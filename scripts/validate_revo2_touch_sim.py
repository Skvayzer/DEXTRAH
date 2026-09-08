#!/usr/bin/env python3
"""Live PhysX contact bench and observation-contract checks. NEVER trains a policy.

The probe is a dynamic, rotation-constrained spherical indenter with an explicit
test-fixture drive. Its contact forces come from PhysX, not a supplied waveform.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=12)
parser.add_argument('--validate-only', action='store_true')
parser.add_argument('--validate-then-serve', action='store_true')
parser.add_argument('--contract-only', action='store_true')
parser.add_argument('--port', type=int, default=8091)
parser.add_argument('--output', type=Path, default=Path('outputs/revo2_touch_live'))
parser.add_argument('--no-tactile', action='store_true')
parser.add_argument('--no-arm-torques', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.no_tactile and not (args.validate_only or args.contract_only):
    parser.error('The live viewer requires tactile sensing; use --validate-only for torque-only checks')
if args.contract_only:
    args.validate_only = True
app = AppLauncher(args).app
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())

import numpy as np
import torch
import trimesh
import viser
from pxr import UsdPhysics, Sdf
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from dextrah_lab.g1_adept.contact import FINGERS, TIP_BODIES, rotate_to_local
from dextrah_lab.g1_adept.touch_frames import pose_matrix, display_pose, ray_surface_point
from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg

torch.set_num_threads(1)


def array(x):
    return x.detach().cpu().numpy()


class BenchEnv(G1Revo2TouchEnv):
    radius = .006

    def _extra_touch_partners(self):
        return ['/World/envs/env_.*/TouchIndenter']

    def _setup_scene(self):
        super()._setup_scene()
        self.probe = RigidObject(RigidObjectCfg(
            prim_path='/World/envs/env_.*/TouchIndenter',
            spawn=sim_utils.SphereCfg(radius=self.radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,
                    max_depenetration_velocity=1., solver_position_iteration_count=12),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0002, rest_offset=0.),
                mass_props=sim_utils.MassPropertiesCfg(mass=.05),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3., 3., 2.))))
        self.scene.rigid_objects['touch_indenter'] = self.probe
        # A translational fixture: free xyz, locked rotation. Avoid sphere rolling
        # confusing a static-shear test. These joints exist only in this bench.
        stage = sim_utils.get_current_stage()
        for path in self.scene.env_prim_paths:
            joint = UsdPhysics.Joint.Define(stage, path+'/ProbeRotationLock')
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path+'/TouchIndenter')])
            for axis in ('rotX', 'rotY', 'rotZ'):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr().Set(1.)
                limit.CreateHighAttr().Set(-1.)


def make_cfg():
    cfg = G1Revo2TouchEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.assets.num_assets_per_type = 1
    cfg.touch.enabled = not args.no_tactile
    cfg.touch.arm_torques = not args.no_arm_torques
    cfg.bps_cache_dir = '/data1/users/konstantin.smirnov/DEXTRAH-BPS128/outputs/bps128_cache'
    cfg.bps_artifact_dir = str(args.output/'bps')
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**19
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_collision_stack_size = 2**24
    return cfg


class Bench:
    def __init__(self):
        self.env = BenchEnv(make_cfg())
        self.commands = queue.SimpleQueue()
        self.palm_id = self.env.robot.body_names.index('right_base_link')
        self.hand_ids = self.env._canonical_joint_ids_lab[7:]
        self.original_joint_limits = self.env.robot.data.joint_pos_limits.clone()
        self.original_default_joints = self.env.robot.data.default_joint_pos.clone()
        self.case = 'Idle'
        self.finger = 1
        self.case_step = 0
        self.case_samples = []
        self.previous_probe_local = None
        self.completed = {}
        self.fixture_force = torch.zeros(self.env.num_envs, 1, 3, device=self.env.device)
        self.fixture_torque = torch.zeros_like(self.fixture_force)
        self.reset()

    def park(self, asset):
        p = asset.data.default_root_state[:, :7].clone()
        p[:, :3] = self.env.scene.env_origins + torch.tensor([3., 3., 2.], device=self.env.device)
        asset.write_root_pose_to_sim(p)
        if asset is not self.env.table:
            asset.write_root_velocity_to_sim(torch.zeros(self.env.num_envs, 6, device=self.env.device))

    def reset(self):
        env = self.env
        env.robot.write_joint_position_limit_to_sim(self.original_joint_limits)
        env.robot.data.default_joint_pos.copy_(self.original_default_joints)
        env.reset()
        for asset in (env.table, env.object, env.probe):
            self.park(asset)
        q = env.robot.data.default_joint_pos.clone()
        q[:, env._canonical_joint_ids_lab[:7]] = torch.tensor([-.45,-.25,0.,1.2,0.,0.,0.], device=env.device)
        q[:, self.hand_ids] = .08
        # Same mimic transmission as the actual direct-control task.
        q[:, env._mimic_target_joint_ids] = (
            q[:, self.hand_ids].index_select(1, env._mimic_source_hand_action_ids)
            * env._mimic_multipliers + env._mimic_offsets)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        # Mechanically clamped hand fixture for contact metrology. These narrow
        # joint limits are BENCH ONLY and restored before the normal task tests.
        # Forces still come from dynamic indenter / real collider interactions.
        env.robot.write_joint_position_limit_to_sim(torch.stack((q-.0001,q+.0001),-1))
        self.targets = q.clone()
        self.case = 'Idle'
        for _ in range(120):
            self.step()
        self.display = np.linalg.inv(pose_matrix(array(env.robot.data.body_pos_w[0,self.palm_id]),
                                                array(env.robot.data.body_quat_w[0,self.palm_id])))

    def start(self, case, finger=1):
        env = self.env
        self.park(env.probe)
        self.case, self.finger, self.case_step = case, finger, 0
        self.case_samples = []
        self.previous_probe_local = None
        pad = env.pad_geometry[finger]
        body = env.touch_body_ids[finger]
        q = env.robot.data.body_quat_w[0,body]
        normal = quat_apply(q, torch.as_tensor(pad.outward, device=env.device, dtype=torch.float32))
        self.normal = normal
        self.tangent = quat_apply(q, env.pad_frames[finger,:,1])
        # Ray cast the pad CENTER onto the actual collision mesh. The extremal
        # vertex can be a rounded edge and turns a press into unintended sliding.
        tree=ET.parse(env.g1_urdf).getroot()
        mesh_file=tree.find(f"link[@name='{TIP_BODIES[finger]}']/collision/geometry/mesh").get('filename')
        # The imported distal has one convex collision shape. Match that hull,
        # not tiny concave facets on the visual/URDF STL surface.
        mesh=trimesh.load(env.g1_urdf.parent/mesh_file,force='mesh').convex_hull
        origin=pad.mesh.centroid + pad.outward*.05
        point,actual_normal=ray_surface_point(mesh,origin,-pad.outward)
        if np.dot(actual_normal,pad.outward)<0:
            actual_normal=-actual_normal
        self.surface = env.robot.data.body_pos_w[0,body] + quat_apply(q,
            torch.as_tensor(pad.back_surface_point if case == 'Backside' else point,
                            device=env.device, dtype=torch.float32))
        if case == 'Backside':
            self.normal = -normal
        else:
            self.normal=quat_apply(q,torch.as_tensor(actual_normal,device=env.device,dtype=torch.float32))
        self.tangent -= torch.dot(self.tangent,self.normal)*self.normal
        self.tangent /= self.tangent.norm()
        self.anchor = self.surface + self.normal*(env.radius+.004)
        pose = env.probe.data.root_state_w[:1,:7].clone()
        pose[0,:3] = self.anchor
        env.probe.write_root_pose_to_sim(pose, torch.tensor([0], device=env.device))
        env.probe.write_root_velocity_to_sim(torch.zeros(1,6,device=env.device), torch.tensor([0],device=env.device))

    def step(self):
        if stop.is_set():
            raise RuntimeError('Diagnostic stop requested')
        env = self.env
        dt = env.cfg.sim.dt
        self.fixture_force.zero_()
        if self.case != 'Idle':
            t = self.case_step*dt
            p, v = env.probe.data.root_pos_w[0], env.probe.data.root_lin_vel_w[0]
            # A fixture drive applies forces; it NEVER supplies tactile readings.
            # Normal approach 0..1s, loaded 1..4s, withdrawal 4..5s.
            depth = min(t,1.)*.006 if t < 4 else .006-max(0.,t-4)*.016
            target = self.anchor-self.normal*depth
            tangent_offset = 0.
            if self.case == 'Slide +/-':
                if 1.5 <= t < 2.5:
                    tangent_offset = .004*(t-1.5)
                elif 2.5 <= t < 3.5:
                    tangent_offset = .004-.008*(t-2.5)
                elif 3.5 <= t < 4:
                    tangent_offset = -.004+.008*(t-3.5)
            target = target+self.tangent*tangent_offset
            delta = target-p
            normal_drive = 800.*torch.dot(delta,self.normal)*self.normal
            tangent_drive = 500.*(delta-torch.dot(delta,self.normal)*self.normal)
            # Static shear is a lateral FORCE during zero target motion.
            if self.case == 'Static shear' and 1.5 <= t < 4:
                # Sub-limit lateral force while the normal fixture holds depth.
                tangent_drive = .05*self.tangent
            self.fixture_force[0,0] = (normal_drive+tangent_drive-8.*v).clamp(-10,10)
            if (p-self.anchor).norm() > .06:
                (args.output/'fixture_failure.json').write_text(json.dumps(dict(
                    case=self.case,finger=FINGERS[self.finger],time=t,
                    position=array(p).tolist(),anchor=array(self.anchor).tolist(),
                    normal=array(self.normal).tolist(),surface=array(self.surface).tolist(),
                    joint_error=float((env.robot.data.joint_pos-self.targets).abs().max()),
                    force=array(env.touch_raw[0,self.finger]).tolist(),
                    samples=self.case_samples),indent=2))
                raise RuntimeError(f'Fixture travel exceeded 60 mm: {self.case} {FINGERS[self.finger]} t={t}')
            self.case_step += 1
        env.probe.set_external_force_and_torque(self.fixture_force, self.fixture_torque, is_global=True)
        ids = env._position_target_joint_ids
        env.robot.set_joint_position_target(self.targets[:,ids], joint_ids=env._position_target_joint_ids_list)
        env.robot.set_joint_velocity_target(torch.zeros_like(self.targets[:,ids]), joint_ids=env._position_target_joint_ids_list)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)
        env._sim_step_counter += 1
        env.capture_touch()
        if self.case != 'Idle':
            body=env.touch_body_ids[self.finger]
            point=env.touch_contact_w[0,self.finger]
            pad_v=env.robot.data.body_com_lin_vel_w[0,body]+torch.linalg.cross(
                env.robot.data.body_com_ang_vel_w[0,body],point-env.robot.data.body_com_pos_w[0,body])
            relative=env.probe.data.root_lin_vel_w[0]-pad_v
            relative-=torch.dot(relative,self.normal)*self.normal
            probe_local=rotate_to_local(env.robot.data.body_quat_w[0,body],
                env.probe.data.root_pos_w[0]-env.robot.data.body_pos_w[0,body]) @ env.pad_frames[self.finger]
            # Reported velocities and finite-difference pose motion differ in
            # this fixture (the task uses zero velocity-solver iterations).
            # Retain both and independently bound actual relative displacement.
            pose_speed=(0. if self.previous_probe_local is None else
                        float((probe_local[1:]-self.previous_probe_local[1:]).norm()/dt))
            self.previous_probe_local=probe_local.clone()
            self.case_samples.append(dict(time=self.case_step*dt,
                force=array(env.touch_raw[0,self.finger]).tolist(),
                friction_tangent=float(torch.dot(env.touch_friction_w[0,self.finger],self.tangent)),
                relative_tangent_speed=pose_speed,
                reported_relative_tangent_speed=float(relative.norm()),
                probe_in_pad=array(probe_local).tolist(),
                joint_error=float((env.robot.data.joint_pos-self.targets).abs().max()),
                link_force=array(env.touch_link_force_w[0,self.finger]).tolist(),
                sensor=array(env.touch_model.force[0,self.finger]).tolist(),
                fixture=array(self.fixture_force[0,0]).tolist(),
                velocity=array(env.probe.data.root_lin_vel_w[0]).tolist()))
            if self.case_step*dt >= 5.5:
                key = f'{FINGERS[self.finger]}:{self.case}'
                self.completed[key] = self.case_samples.copy()
                (args.output/'contact_traces_partial.json').write_text(json.dumps(self.completed)+'\n')
                print('TOUCH_CASE_FINISHED', key, flush=True)
                self.case = 'Idle'
                self.park(env.probe)
        if not torch.isfinite(env.robot.data.joint_pos).all() or not torch.isfinite(env.touch_raw).all():
            raise RuntimeError('Nonfinite physics state')
        if env.touch_link_force_w.norm(dim=-1).max() > 30:
            raise RuntimeError('Diagnostic force stop: 30 N')

    def contract(self):
        env = self.env
        env.robot.write_joint_position_limit_to_sim(self.original_joint_limits)
        env.robot.data.default_joint_pos.copy_(self.original_default_joints)
        obs, _ = env.reset()
        extra = env._touch_extra_dim
        assert obs['policy'].shape == (env.num_envs,224+extra)
        assert obs['critic'].shape == (env.num_envs,246+extra)
        resets = 0
        terminal = 0
        for _ in range(90):
            actions = torch.zeros(env.num_envs,13,device=env.device)
            obs, reward, done, timeout, info = env.step(actions)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all()
            assert torch.isfinite(reward).all()
            assert obs['policy'].shape[-1] == 224+extra
            torch.testing.assert_close(obs['policy'][:,92:224], env._bps_features)
            if 'final_observation' in info:
                assert info['final_observation']['critic'].shape == (env.num_envs,246+extra)
                terminal += 1
            resets += int((done|timeout).sum())
        # Force timeout, exercise pre-reset terminal critic and reset sensor state.
        env.episode_length_buf[:] = env.max_episode_length
        obs, _, _, timeout, info = env.step(torch.zeros(env.num_envs,13,device=env.device))
        assert timeout.all()
        assert info['final_observation']['critic'].shape[-1] == 246+extra
        assert not env.touch_model.valid.any()
        self.network_contract(obs)
        return dict(actor_dim=224+extra,critic_dim=246+extra,actions=13,
                    partial_resets=resets,terminal_steps=terminal+1,
                    arm_torque_source='IsaacLab implicit-actuator clipped PD estimate',
                    sensor_filter_paths=getattr(env,'touch_filter_paths',[]),
                    distal_materials=(array(env.touch_resolved_materials[0]).tolist()
                                      if env.cfg.touch.enabled else []),
                    optimizer_steps=0)

    def network_contract(self, obs):
        # Construct the actual SAPG actor/critic and run inference, but NEVER
        # construct an optimizer, Runner or training loop.
        from rl_games.algos_torch import model_builder
        from dextrah_lab.tasks.g1_revo2_adept.agents.touch_sapg import config
        from isaaclab_tasks.utils import load_cfg_from_registry
        import dextrah_lab.tasks.g1_revo2_adept.gym_setup
        loaded = load_cfg_from_registry('G1-Revo2-SimToolReal-Repose-BPS128-Touch','rl_games_cfg_entry_point')
        assert loaded['params']['config']['name']=='g1_sapg_bps128_touch'
        parameters=config()['params']
        for kind in ('policy','critic'):
            dim=obs[kind].shape[-1]
            params=parameters if kind=='policy' else dict(model={'name':'central_value'},network=parameters['config']['central_value_config']['network'])
            network=model_builder.ModelBuilder().load(params).build(dict(actions_num=13,
                input_shape=(dim+32,),num_seqs=self.env.num_envs,value_size=1,
                normalize_value=True,normalize_input=True,type='extra_param',
                coef_ids=torch.arange(6,dtype=torch.float32),coef_id_idx=dim)).eval()
            coefficients=(torch.arange(self.env.num_envs)%6).float()[:,None]
            inputs=torch.cat((obs[kind].cpu(),coefficients),-1)
            rnn=[x.expand(-1,self.env.num_envs,-1).contiguous() for x in network.get_default_rnn_state()] if network.is_rnn() else None
            out=network(dict(obs=inputs,is_train=False,rnn_states=rnn))
            assert torch.isfinite(out['values']).all()
            if kind=='policy':
                assert out['mus'].shape==(self.env.num_envs,13)
                assert torch.isfinite(out['mus']).all() and torch.isfinite(out['sigmas']).all()
        print('TOUCH_SAPG_FORWARD_PASS no_optimizer=true',flush=True)

    def validate(self):
        result = self.contract()
        if args.contract_only or not self.env.cfg.touch.enabled:
            return result
        self.reset()
        for finger in range(5):
            print('TOUCH_TEST_BEGIN', FINGERS[finger], flush=True)
            self.start('Press / release', finger)
            while self.case != 'Idle':
                self.step()
            data = self.completed[f'{FINGERS[finger]}:Press / release']
            force = np.array([x['force'] for x in data])
            assert force[:,0].max() > .03, (FINGERS[finger],force.max(0))
            assert np.abs(force[-24:]).max() < .03, ('release',FINGERS[finger])
            assert self.env.touch_raw[1:].abs().max() < .01, 'cross-environment contact leak'
            result[FINGERS[finger]+'_peak_n'] = float(force[:,0].max())
        for case in ('Static shear','Slide +/-','Backside'):
            self.start(case,1)
            while self.case != 'Idle':
                self.step()
        static = self.completed['index:Static shear']
        held = [x for x in static if 2.5 < x['time'] < 3.8]
        result['static_shear_n'] = float(np.mean([x['force'][1] for x in held]))
        result['static_probe_speed_m_s'] = float(np.mean([np.linalg.norm(x['velocity']) for x in held]))
        result['static_friction_n'] = float(np.mean([x['friction_tangent'] for x in held]))
        result['static_relative_speed_m_s'] = float(np.mean([x['relative_tangent_speed'] for x in held]))
        held_positions = np.array([x['probe_in_pad'][1:] for x in held])
        result['static_tangential_excursion_m'] = float(np.linalg.norm(np.ptp(held_positions,axis=0)))
        result['static_tangential_drift_m'] = float(np.linalg.norm(held_positions[-1]-held_positions[0]))
        result['static_force_balance_error_n'] = float(np.mean([
            np.linalg.norm(np.asarray(x['link_force'])-np.asarray(x['fixture'])) for x in held]))
        assert abs(result['static_friction_n']) > .01, result
        assert result['static_relative_speed_m_s'] < .003, result
        assert result['static_tangential_excursion_m'] < .0002, result
        assert result['static_tangential_drift_m'] < .0001, result
        assert result['static_force_balance_error_n'] < .03, result
        slide = self.completed['index:Slide +/-']
        positive = np.mean([x['friction_tangent'] for x in slide if 1.8 < x['time'] < 2.4])
        negative = np.mean([x['friction_tangent'] for x in slide if 2.9 < x['time'] < 3.4])
        result.update(slide_forward_shear_n=float(positive),slide_reverse_shear_n=float(negative))
        assert positive*negative < 0 and min(abs(positive),abs(negative))>.02, result
        back = self.completed['index:Backside']
        result['backside_pad_peak_n'] = float(np.abs([x['force'] for x in back]).max())
        result['backside_link_peak_n'] = float(np.linalg.norm([x['link_force'] for x in back],axis=-1).max())
        assert result['backside_pad_peak_n'] < .03
        assert result['backside_link_peak_n'] > .03
        return result

    def viewer(self):
        env = self.env
        server = viser.ViserServer(host='127.0.0.1',port=args.port)
        if server.get_port()!=args.port:
            raise RuntimeError('Requested viewer port is occupied')
        server.scene.set_up_direction('+z')
        server.gui.add_markdown('## Live Revo2 contact physics\n'
            'PhysX normal + friction forces. **Not recording replay; no policy training.** '
            'Hand mechanically clamped for contact tests; dynamic spherical indenter, rotation-locked fixture. '
            'Pad axes/materials provisional; whole distal collider compliant.')
        running = server.gui.add_checkbox('Run physics', initial_value=True)
        selected = server.gui.add_dropdown('Finger',options=FINGERS,initial_value='index')
        mode = server.gui.add_dropdown('Test',options=('Press / release','Static shear','Slide +/-','Backside'),initial_value='Press / release')
        server.gui.add_button('Run selected test').on_click(lambda _:self.commands.put(('test',mode.value,selected.value)))
        server.gui.add_button('Release / cancel').on_click(lambda _:self.commands.put(('release',)))
        loop = server.gui.add_checkbox('Repeat test',initial_value=False)
        offset_arrows = server.gui.add_checkbox('Offset force arrows',initial_value=True)
        state = server.gui.add_markdown('Ready')
        values = server.gui.add_markdown('')
        torque = server.gui.add_markdown('')
        server.gui.add_markdown('Blue = compression; orange = shear. Cyan pads turn gold on contact. '
            'Shear arrows use 5x visual magnification for visibility; numeric values remain N. '
            'Offset arrows sit 30 mm outside the pad, connected to the true contact by a gray line. '
            'Arrows use instantaneous physics; table shows raw and rate-limited 10 Hz outputs. '
            'No privileged contact partner or position is passed to the policy.')
        def camera(client):
            client.camera.position=(.20,.16,.18)
            client.camera.look_at=(.01,0.,.09)
            client.camera.up_direction=(0.,0.,1.)
        server.on_client_connect(camera)
        server.gui.add_button('Reset camera').on_click(lambda _:[camera(c) for c in server.get_clients().values()])
        tree=ET.parse(env.g1_urdf).getroot()
        descendants={'right_base_link'}
        for _ in range(15):
            for j in tree.findall('joint'):
                if j.find('parent').get('link') in descendants:
                    descendants.add(j.find('child').get('link'))
        handles=[]
        for body in env.robot.body_names:
            if body not in descendants:
                continue
            for index,visual in enumerate(tree.findall(f"link[@name='{body}']/visual")):
                g=visual.find('geometry/mesh')
                if g is None:
                    continue
                mesh=trimesh.load(env.g1_urdf.parent/g.get('filename'),force='mesh')
                mesh.apply_scale(np.fromstring(g.get('scale','1 1 1'),sep=' '))
                origin=visual.find('origin')
                if origin is not None:
                    tr=trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
                    tr[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                    mesh.apply_transform(tr)
                handle=server.scene.add_mesh_simple(f'/hand/{body}_{index}',mesh.vertices.astype(np.float32),mesh.faces.astype(np.uint32),color=(178,188,200))
                handles.append((env.robot.body_names.index(body),handle))
        pads=[]
        for i,pad in enumerate(env.pad_geometry):
            mesh=pad.mesh.copy()
            mesh.vertices += .0004*pad.outward
            pads.append(server.scene.add_mesh_simple(f'/pad/{FINGERS[i]}',mesh.vertices.astype(np.float32),mesh.faces.astype(np.uint32),color=(30,210,210),opacity=.85))
        ball=server.scene.add_icosphere('/indenter',radius=env.radius,color=(190,85,245))
        arrows={}
        self.start('Press / release',1)
        last_publish=0.
        fault = None
        try:
            while app.is_running() and not stop.is_set():
                start=time.monotonic()
                while not self.commands.empty():
                    command=self.commands.get()
                    if command[0]=='test':
                        if fault is not None:
                            self.reset()
                            fault = None
                        self.start(command[1],FINGERS.index(command[2]))
                        running.value=True
                    else:
                        self.case='Idle'
                        self.park(env.probe)
                        loop.value=False
                        running.value=fault is None
                if running.value and server.get_clients() and fault is None:
                    if self.case=='Idle' and loop.value:
                        self.start(mode.value,FINGERS.index(selected.value))
                    try:
                        self.step()
                    except (RuntimeError, AssertionError) as exc:
                        fault = str(exc)
                        print('TOUCH_VIEWER_SAFETY_STOP',fault,flush=True)
                        self.case='Idle'
                        self.park(env.probe)
                        running.value=False
                if start-last_publish>.08:
                    last_publish=start
                    with server.atomic():
                        for body,handle in handles:
                            handle.position,handle.wxyz=display_pose(self.display,array(env.robot.data.body_pos_w[0,body]),array(env.robot.data.body_quat_w[0,body]))
                        ball.position=trimesh.transform_points(array(env.probe.data.root_pos_w[:1]),self.display)[0]
                        ball.visible=self.case!='Idle'
                        for i,pad in enumerate(pads):
                            body=env.touch_body_ids[i]
                            p,q=array(env.robot.data.body_pos_w[0,body]),array(env.robot.data.body_quat_w[0,body])
                            pad.position,pad.wxyz=display_pose(self.display,p,q)
                            active=float(env.touch_raw[0,i,0])>.02
                            pad.color=(255,196,30) if active else (30,210,210)
                            pose=pose_matrix(p,q)
                            position=array(env.touch_contact_w[0,i]) if active else pose[:3,:3]@env.pad_geometry[i].surface_point+p
                            frame=pose[:3,:3]@array(env.pad_frames[i])
                            f=array(env.touch_raw[0,i])
                            glyph_position=position-frame[:,0]*.03 if offset_arrows.value else position
                            connector=f'/forces/{i}/offset'
                            if connector in arrows:
                                arrows.pop(connector).remove()
                            if active and offset_arrows.value:
                                points=trimesh.transform_points(np.array([position,glyph_position]),self.display)[None]
                                arrows[connector]=server.scene.add_line_segments(connector,
                                    points=points.astype(np.float32),colors=(145,145,155),
                                    thickness=1.5,thickness_units='screen')
                            for name,vector,color in [('normal',frame[:,0]*f[0],(60,145,255)),('shear',frame[:,1:]@f[1:],(255,145,30))]:
                                key=f'/forces/{i}/{name}'
                                if key in arrows:
                                    arrows.pop(key).remove()
                                length=np.linalg.norm(vector)
                                if length>.02:
                                    unit=vector/length
                                    arrow_length=min(.04 if name=='shear' else .02,
                                                     max(.004,length*(.30 if name=='shear' else .06)))
                                    end=glyph_position+unit*arrow_length
                                    side=np.cross(unit,[0.,0.,1.])
                                    if np.linalg.norm(side)<1e-5:
                                        side=np.cross(unit,[1.,0.,0.])
                                    side/=np.linalg.norm(side)
                                    head=min(.004,arrow_length*.35)
                                    segments=np.array([[glyph_position,end],[end,end-unit*head+side*head*.5],
                                                       [end,end-unit*head-side*head*.5]])
                                    points=trimesh.transform_points(segments.reshape(-1,3),self.display).reshape(-1,2,3)
                                    arrows[key]=server.scene.add_line_segments(key,points=points.astype(np.float32),colors=color,
                                        thickness=4.,thickness_units='screen')
                    state.content=(f'**{self.case}** · {FINGERS[self.finger]} · '
                        f'test time {self.case_step*env.cfg.sim.dt:.2f} s · '
                        f'physics time {env._sim_step_counter*env.cfg.sim.dt:.2f} s')
                    if fault is not None:
                        state.content += f'\n\n**SAFETY STOP:** {fault}. Run a new test to reset the fixture.'
                    rows=['| Finger | Raw Fn | Raw shear x/y | Delivered Fn / shear | Valid / age |','|---|---:|---|---|---|']
                    for i,name in enumerate(FINGERS):
                        f=array(env.touch_raw[0,i]); d=array(env.touch_model.force[0,i])
                        rows.append(f'| {name} | {f[0]:.2f} | {f[1]:+.2f}, {f[2]:+.2f} | {d[0]:.2f} / {d[1]:+.2f}, {d[2]:+.2f} | {bool(env.touch_model.valid[0,i])} / {float(env.touch_model.age_s[0,i])*1000:.0f} ms |')
                    values.content='\n'.join(rows)
                    torque.content='Arm motor torque **estimates** [Nm] (clamped bench, not grasping): '+', '.join(f'{x:+.2f}' for x in array(env.arm_motor_torques()[0]))
                time.sleep(max(0.,env.cfg.sim.dt-(time.monotonic()-start)))
        finally:
            server.stop()


try:
    args.output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        bench=Bench()
        try:
            if args.validate_only or args.validate_then_serve:
                result=bench.validate()
                (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
                (args.output/'contact_traces.json').write_text(json.dumps(bench.completed)+'\n')
                print('TOUCH_SIM_VALIDATED '+json.dumps(result),flush=True)
                if args.validate_then_serve:
                    bench.reset()
                    bench.viewer()
            else:
                bench.viewer()
        finally:
            bench.env.close()
except Exception:
    traceback.print_exc()
    os._exit(1)
app.close()
