#!/usr/bin/env python3
"""Compare the original GRAIL G1 asset and the source Revo2 asset on empty floor.

This is a controlled IsaacLab harness, NOT the untouched C++ deployment runtime.
Original pretrained controller only; no SAPG, objects, task rewards or optimizer.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=Path('/data1/users/konstantin.smirnov'))
    parser.add_argument('--robots', nargs='+', choices=['upstream', 'revo2'], default=['upstream', 'revo2'])
    parser.add_argument('--case', nargs='+', help='Explicit subset of predeclared tests')
    parser.add_argument('--packing', choices=['checkpoint', 'interleaved'], default='checkpoint',
                        help='Interleaved is an explicit diagnostic ablation, never a checkpoint migration')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise RuntimeError('Require exactly one allocated GPU')
    free_mib = int(subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free_mib < 12*1024:
        raise RuntimeError('Insufficient safe memory headroom; leave training alone')
    args.output.mkdir(parents=True, exist_ok=False)
    args.headless = True
    app = AppLauncher(args).app
    stack = (args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack)
    faulthandler.dump_traceback_later(90, repeat=True, file=stack)
    report = dict(completed=False, optimizer_updates=0, training_modified=False,
                  harness='Standalone IsaacLab, upstream robot config and pinned GRAIL manipulation base; not C++ runtime',
                  source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'), packing=args.packing)
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.prims import create_prim
        from isaaclab.assets import Articulation
        from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat
        from dextrah_lab.wholebody.sonic import FrozenSonic, SonicHistory, WEIGHTS_SHA256
        from dextrah_lab.wholebody.contract import BODY_JOINTS, nominal_body_pose, joint_indices
        from dextrah_lab.wholebody.actuators import body_motors
        from dextrah_lab.wholebody.navigation_reference import NavigationReference, yaw_of
        from dextrah_lab.wholebody.walking_diagnostic import cases, command_at, metrics
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
        sonic = FrozenSonic(args.workspace/'GRAIL', args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        # Load the real upstream config files without importing the unrelated
        # motion-dataset/training package initializer and its optional deps.
        import importlib.util
        import runpy
        import sys
        upstream = args.workspace/'GRAIL/imports/SONIC/gear_sonic/envs/manager_env'
        module_name = 'gear_sonic.envs.manager_env.mdp.actuators'
        spec = importlib.util.spec_from_file_location(module_name, upstream/'mdp/actuators.py')
        actuator_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = actuator_module
        spec.loader.exec_module(actuator_module)
        G1_CYLINDER_MODEL_12_DEX_CFG = runpy.run_path(str(upstream/'robots/g1.py'))['G1_CYLINDER_MODEL_12_DEX_CFG']
        tests = [c for c in cases() if args.case is None or c['name'] in args.case]
        if not tests or (args.case and set(args.case) != {c['name'] for c in tests}):
            raise ValueError('Unknown or empty test selection')
        material = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=args.device,
            render_interval=4, physics_material=material,
            physx=sim_utils.PhysxCfg(gpu_found_lost_pairs_capacity=2**20,
                gpu_found_lost_aggregate_pairs_capacity=2**21, gpu_total_aggregate_pairs_capacity=2**20,
                gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**18,
                gpu_collision_stack_size=2**26)))
        sim_utils.GroundPlaneCfg(physics_material=material).func('/World/ground', sim_utils.GroundPlaneCfg(physics_material=material))
        # Same measured reset pose for both assets, no table-specific arm pose.
        q0 = torch.tensor(nominal_body_pose(), device=args.device)
        scales = torch.tensor([m.action_scale for m in body_motors().values()], device=args.device)
        robots, configurations = {}, {}
        for robot_index, kind in enumerate(args.robots):
            if kind == 'upstream':
                cfg = G1_CYLINDER_MODEL_12_DEX_CFG.copy()
                cfg.spawn.usd_dir = str(args.output/'upstream_usd')
                cfg.spawn.usd_file_name = 'robot.usd'
                cfg.spawn.force_usd_conversion = True
                urdf = cfg.spawn.asset_path
            else:
                # Reuse the source importer/baker and actual trained actuator
                # configuration, not the older native-mimic Revo2 prototype.
                from dextrah_lab.wholebody.source_config import source_task_config
                from dextrah_lab.wholebody.source_scene import floating_sonic_robot_scene
                from isaacsimenvs.tasks.play.utils import scene_utils as source
                source_cfg, _ = source_task_config(args.output/'source', len(tests), args.device)
                assets = source_cfg.assets
                urdf = source._resolve_asset_path(assets.robot_urdf)
                raw = source._convert_urdf_to_usd(urdf, args.output/'revo2_conversion',
                    fix_base=False, self_collision=False, merge_fixed_joints=True,
                    fix_locked_joints=True, remove_fixed_links=source.G1_BRAINCO_UNUSED_FIXED_SENSOR_LINKS,
                    remove_link_subtrees=(), fix_joint_names=source.robot_fixed_joint_names(assets),
                    fix_joint_positions=source.robot_fixed_joint_positions(assets),
                    remove_collision_links=(), joint_drive=source._robot_joint_drive_cfg())
                with floating_sonic_robot_scene():
                    baked = source._bake_usd(raw, args.output/'baked', 'robot', props=dict(
                        disable_gravity=False, max_depenetration_velocity=1000.,
                        enabled_self_collisions=False, solver_position_iterations=8, solver_velocity_iterations=0),
                        apply_physx_articulation=True, strip_visuals=False)
                    cfg = source.build_robot_articulation_usd_cfg(baked, assets)
                cfg.init_state.joint_pos.update(dict(zip(BODY_JOINTS, map(float, nominal_body_pose()))))
            cfg.prim_path = f'/World/{kind}_trial_.*/Robot'
            for i in range(len(tests)):
                create_prim(f'/World/{kind}_trial_{i}', 'Xform', translation=(i*20., robot_index*30., 0.))
            cfg.init_state.pos = (0., 0., .76)
            cfg.init_state.rot = (1., 0., 0., 0.)
            robots[kind] = Articulation(cfg)
            configurations[kind] = dict(robot_urdf=str(urdf), articulation=cfg.to_dict())
        sim.reset()
        planner_path = args.workspace/'G1-SONIC-planner/planner_sonic.onnx'
        session_owner = NavigationReference(planner_path, speed_limit=.4)
        planners, histories, ids, offsets, traces, last_actions = {}, {}, {}, {}, {}, {}
        report.update(pretrained_sonic_sha256=WEIGHTS_SHA256, cases=tests, robots={},
                      physics_hz=200, control_hz=50, capture_hz=25, seconds=14,
                      nominal_body_pose=nominal_body_pose().tolist(), body_joint_names=list(BODY_JOINTS))
        for robot_index, (kind, robot) in enumerate(robots.items()):
            ids[kind] = torch.tensor(joint_indices(robot.joint_names, BODY_JOINTS), device=args.device)
            root = robot.data.default_root_state.clone()
            offsets[kind] = root.new_tensor([[i*20., robot_index*30., 0.] for i in range(len(tests))])
            root[:, :3] += offsets[kind]
            for i, case in enumerate(tests):
                a = np.deg2rad(case['yaw_deg'])/2
                root[i, 3:7] = root.new_tensor([np.cos(a), 0., 0., np.sin(a)])
            q = robot.data.default_joint_pos.clone()
            q[:, ids[kind]] = q0
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.reset()
            robot.write_data_to_sim()
            histories[kind] = SonicHistory(len(tests), args.device)
            last_actions[kind] = torch.zeros(len(tests), 29, device=args.device)
            planners[kind] = [NavigationReference(planner_path, session=session_owner.session, speed_limit=.4) for _ in tests]
            for i, planner in enumerate(planners[kind]):
                pose = root[i, :7].cpu().numpy().copy()
                pose[:3] -= offsets[kind][i].cpu().numpy()
                planner.reset(pose, nominal_body_pose(), 0.)
            traces[kind] = [[] for _ in tests]
            report['robots'][kind] = dict(body_names=robot.body_names, joint_names=robot.joint_names,
                robot_urdf=configurations[kind]['robot_urdf'], masses_kg=robot.root_physx_view.get_masses()[0].tolist(),
                body_q0_error=float((robot.data.default_joint_pos[:, ids[kind]]-q0).abs().max()))
        # Save resolved configurations before rollout, preserving the exact differences.
        (args.output/'configurations.json').write_text(json.dumps(configurations, indent=2, default=str))
        (args.output/'metadata.json').write_text(json.dumps(report, indent=2))
        start = time.monotonic()
        with torch.inference_mode():
            for step in range(700):
                now = step*.02
                for kind, robot in robots.items():
                    data = robot.data
                    q, qd, ref = [], [], []
                    commands = np.stack([command_at(c, now) for c in tests])
                    for i, planner in enumerate(planners[kind]):
                        rq, rv, rr = planner.reference(now, commands[i], yaw_of(data.root_quat_w[i].cpu().numpy()))
                        q.append(rq); qd.append(rv); ref.append(rr)
                    q, qd, ref = [torch.as_tensor(np.stack(x), device=args.device) for x in (q, qd, ref)]
                    ori = quat_mul(quat_conjugate(yaw_quat(data.root_quat_w))[:, None].expand(-1, 10, -1), ref[:, :, 3:])
                    ori6 = matrix_from_quat(ori)[:, :, :, :2].reshape(len(tests), 10, 6)
                    proprio = histories[kind].push(data.root_ang_vel_b,
                        data.joint_pos[:, ids[kind]]-q0, data.joint_vel[:, ids[kind]],
                        last_actions[kind], data.projected_gravity_b)
                    if args.packing == 'checkpoint':
                        action = sonic(proprio, q, qd, ori6)
                    else:
                        # Deliberate feature-layout A/B, never changes checkpoint weights.
                        joined = torch.cat((q, qd), -1).flatten(1)
                        action = sonic(proprio, joined[:, :290].reshape(-1, 10, 29),
                                       joined[:, 290:].reshape(-1, 10, 29), ori6)
                    target = q0+scales*action
                    if not torch.isfinite(target).all():
                        raise RuntimeError('Nonfinite SONIC motor targets')
                    robot.set_joint_position_target(target, joint_ids=ids[kind])
                    robot.write_data_to_sim()
                    if step % 2 == 0:
                        def array(v): return v.detach().cpu().numpy().copy()
                        for i, case in enumerate(tests):
                            root = array(data.root_state_w[i, :7]); root[:3] -= array(offsets[kind][i])
                            upright = float(-data.projected_gravity_b[i, 2])
                            traces[kind][i].append(dict(time_s=now, root=root,
                                velocity_body=array(torch.cat((data.root_lin_vel_b[i], data.root_ang_vel_b[i]))),
                                command=commands[i], upright=upright, fell=root[2] < .4 or upright < .5,
                                body_pos=array(data.body_pos_w[i]-offsets[kind][i]), body_quat=array(data.body_quat_w[i]),
                                joint_q=array(data.joint_pos[i, ids[kind]]), joint_qd=array(data.joint_vel[i, ids[kind]]),
                                reference_q=array(q[i]), reference_qd=array(qd[i]), reference_root=array(ref[i]),
                                action=array(action[i]), target=array(target[i])))
                    last_actions[kind] = action
                for _ in range(4):
                    sim.step(render=False)
                    for robot in robots.values(): robot.update(.005)
                if step % 50 == 0:
                    free = torch.cuda.mem_get_info()[0]/2**30
                    print('EMPTY_FLOOR_PROGRESS '+json.dumps(dict(sim_s=now, wall_s=time.monotonic()-start, free_gib=free)), flush=True)
                    if free < 6.:
                        raise RuntimeError('Stopping diagnostic to preserve training memory headroom')
        for kind in robots:
            report['robots'][kind]['tests'] = {}
            for i, case in enumerate(tests):
                folder = args.output/kind/case['name']; folder.mkdir(parents=True)
                trace = {k: np.stack([f[k] for f in traces[kind][i]]) for k in traces[kind][i][0]}
                np.savez_compressed(folder/'trajectory.npz', **trace)
                summary = metrics(trace, case)
                summary['planner'] = planners[kind][i].report()
                report['robots'][kind]['tests'][case['name']] = summary
                (folder/'metrics.json').write_text(json.dumps(summary, indent=2))
        report.update(completed=True, wall_seconds=time.monotonic()-start, root_state_writes='initialization only',
                      physical_joint_freezing=False, residual=False, object_or_table=False)
        print('EMPTY_FLOOR_COMPLETE '+json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'metadata.json').write_text(json.dumps(report, indent=2))
        faulthandler.cancel_dump_traceback_later()
        app.close()


if __name__ == '__main__':
    main()
