#!/usr/bin/env python3
"""Dump the Isaac/PhysX model actually simulated, as the mjlab port's reference.

Inference only. Builds the same environment as training run 1105 (per-env
object physics, table contact monitor) and saves what PhysX resolved:
joint/body order, limits, drive gains, armature, velocity/effort limits,
link masses/inertias/COMs, shape materials, per-env object inertials, and
the default robot state. A second, optional part records PD step responses
for the parity test: from the standing reset, every joint group receives a
fixed target offset while the other joints hold their targets.
"""
import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--step-response', action='store_true')
    p.add_argument('--fixed-body-run', type=Path,
                   help='Dump the fixed-body touch task (G1Revo2TouchEnv) exactly as this run resolved it '
                        '(params/env_resolved.yaml) instead of the run-1105 SONIC task')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True)
    args.headless = True
    AppLauncher(args)
    report = dict(completed=False)
    try:
        import numpy as np
        import torch
        from dextrah_lab.wholebody.sonic import FrozenSonic
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        workspace = Path('/data1/users/konstantin.smirnov')
        if args.fixed_body_run:
            import copy
            from isaaclab.utils import replace_strings_with_slices
            from dextrah_lab.g1_adept.touch_continuation import load_yaml
            from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg
            from dextrah_lab.wholebody import per_env_objects
            resolved = load_yaml(args.fixed_body_run/'params/env_resolved.yaml')
            cfg = G1Revo2TouchEnvCfg()
            cfg.seed = resolved['seed']   # None-typed default: set before from_dict, as the trainer does
            cfg.from_dict(replace_strings_with_slices(copy.deepcopy(resolved)))
            cfg.scene.num_envs = args.num_envs
            cfg.sim.device = args.device
            cfg.bps_artifact_dir = str(args.output/'bps')
            env_class = G1Revo2TouchEnv
            if not cfg.scene.replicate_physics:
                env_class = per_env_objects.with_per_env_objects(env_class)
            env = env_class(cfg)
            report['fixed_body_run'] = str(args.fixed_body_run)
            report['per_env_object_physics'] = per_env_objects.verify(env) if not cfg.scene.replicate_physics else None
        else:
            cfg, _ = source_task_config(args.output, args.num_envs, args.device)
            cfg.seed = 42
            contract = json.loads((workspace/'DEXTRAH-G1-WholeBody/outputs/0_frozen_sonic_sapg_touch_dexterous_1105/task_contract.json').read_text())
            cfg.sonic_body.from_dict(contract['body_termination'])
            sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
            env = G1SonicTouchEnv(cfg, sonic=sonic)
        r, o, t = env.robot, env.object, env.table
        view = r.root_physx_view
        arr = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        out = dict(
            joint_names=np.array(r.joint_names), body_names=np.array(r.body_names),
            joint_pos_limits=arr(r.data.joint_pos_limits[0]), joint_vel_limits=arr(r.data.joint_vel_limits[0]),
            joint_effort_limits=arr(r.data.joint_effort_limits[0]), joint_stiffness=arr(r.data.joint_stiffness[0]),
            joint_damping=arr(r.data.joint_damping[0]), joint_armature=arr(r.data.joint_armature[0]),
            joint_friction=arr(r.data.joint_friction_coeff[0]),
            dof_max_velocities=arr(view.get_dof_max_velocities()[0]), dof_max_forces=arr(view.get_dof_max_forces()[0]),
            default_joint_pos=arr(r.data.default_joint_pos[0]), default_root_state=arr(r.data.default_root_state[0]),
            body_masses=arr(view.get_masses()[0]), body_inertias=arr(view.get_inertias()[0]),
            body_coms=arr(view.get_coms()[0]), link_paths=np.array(view.link_paths[0]),
            shape_materials=arr(view.get_material_properties()[0]),
            object_masses=arr(o.root_physx_view.get_masses()[:, 0]),
            object_inertias=arr(o.root_physx_view.get_inertias()[:, 0]),
            object_coms=arr(o.root_physx_view.get_coms()[:, 0]),
            object_materials=arr(o.root_physx_view.get_material_properties()),
            object_asset_index=arr(env._object_asset_index_per_env),
            table_mass=arr(t.root_physx_view.get_masses()[0]),
            table_materials=arr(t.root_physx_view.get_material_properties()[0]),
            target_joint_ids=arr(env._position_target_joint_ids),
        )
        shapes = []
        for name, path in zip(view.shared_metatype.link_names, view.link_paths[0]):
            shapes.append((name, int(r._physics_sim_view.create_rigid_body_view(path).max_shapes)))
        out['shapes_per_link'] = np.array([s for _, s in shapes])
        out['shape_link_names'] = np.array([n for n, _ in shapes])
        np.savez_compressed(args.output/'isaac_reference.npz', **out)
        report.update(sim=dict(dt=cfg.sim.dt, decimation=cfg.decimation,
                               solver=dict(pos_iters=cfg.sim.physx.min_position_iteration_count if hasattr(cfg.sim.physx, 'min_position_iteration_count') else None)),
                      joints=len(r.joint_names), bodies=len(r.body_names), envs=env.num_envs)
        if args.step_response:
            report['step_response'] = step_responses(env, args.output)
        report['completed'] = True
        print('ISAAC_REFERENCE_DONE '+json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'reference.json').write_text(json.dumps(report, indent=2, default=str))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


STEP_GROUPS = {
    'right_arm': ('right_shoulder_pitch_joint', .3), 'right_elbow': ('right_elbow_joint', .3),
    'right_wrist': ('right_wrist_pitch_joint', .3), 'left_elbow': ('left_elbow_joint', .3),
    'waist_yaw': ('waist_yaw_joint', .15), 'right_index': ('right_index_proximal_joint', .8),
    'right_thumb_meta': ('right_thumb_metacarpal_joint', .8),
}


def step_responses(env, output):
    """Hold every target fixed at its reset value, then offset one joint.

    Bypasses the policy and SONIC: raw PD targets through the same
    `_apply_action` writer, with the robot's base free (gravity on).
    """
    import numpy as np
    import torch
    names = list(env.robot.joint_names)
    seconds, n = 1.5, env.num_envs
    results = {}
    for group, (joint, offset) in STEP_GROUPS.items():
        env.reset()
        j = names.index(joint)
        base = env._cur_targets.clone()
        trace = []
        for step in range(round(seconds/env.cfg.sim.dt)):
            target = base.clone()
            if step >= 12:   # 0.1 s of hold first
                target[:, j] += offset
            env._cur_targets[:] = target
            env._sim_step_counter += 1
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            trace.append(env.robot.data.joint_pos[:8, j].cpu().numpy().copy())
        results[group] = dict(joint=joint, offset=offset, dt=env.cfg.sim.dt,
                              trace=np.stack(trace).tolist(), initial_target=float(base[0, j]))
    (output/'step_response.json').write_text(json.dumps(results))
    return {k: dict(joint=v['joint'], offset=v['offset']) for k, v in results.items()}


if __name__ == '__main__':
    main()
