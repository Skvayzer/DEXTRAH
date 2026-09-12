#!/usr/bin/env python3
"""Small floating-base SONIC probe. No SAPG updates or claim of M0 completion.

A constant nominal reference tests the interface; a failure alone does not
establish failure on the controller's original reference-motion distribution.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in a Slurm GPU allocation')
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--controller',choices=['sonic','pd'],default='sonic')
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    # Isaac's URDF importer uses the export directory to author USD sublayers.
    # A relative directory can generate broken /configuration/... references.
    args.output=args.output.resolve()
    if args.num_envs < 1 or args.seconds < 2:
        raise ValueError('Invalid probe size/duration')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    args.headless=True
    app=AppLauncher(args).app
    try:
        import numpy as np
        import torch
        from pxr import UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils.math import matrix_from_quat, quat_conjugate, yaw_quat
        from dextrah_lab.wholebody.asset import prepare_urdf
        from dextrah_lab.wholebody.asset_cfg import fullbody_robot_cfg
        from dextrah_lab.wholebody.actuators import body_motors, actuator_manifest
        from dextrah_lab.wholebody.contract import BODY_JOINTS, hand_joints, joint_indices, PHYSICS_DT, CONTROL_DT
        from dextrah_lab.wholebody.sonic import FrozenSonic, SonicHistory
        torch.set_num_threads(2)
        torch.manual_seed(42)
        urdf=args.output/'g1_revo2_fullbody.urdf'
        audit=prepare_urdf(args.workspace/'play2perfect/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf',urdf)
        model=(FrozenSonic(args.workspace/'GRAIL',
            args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base',args.device)
            if args.controller=='sonic' else None)
        cfg=sim_utils.SimulationCfg(dt=PHYSICS_DT,device=args.device,render_interval=4,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
        cfg.physx.gpu_max_rigid_contact_count=2**20
        cfg.physx.gpu_max_rigid_patch_count=2**18
        cfg.physx.gpu_found_lost_pairs_capacity=2**20
        cfg.physx.gpu_found_lost_aggregate_pairs_capacity=2**20
        cfg.physx.gpu_total_aggregate_pairs_capacity=2**20
        cfg.physx.gpu_collision_stack_size=2**26
        sim=sim_utils.SimulationContext(cfg)
        scene_cfg=InteractiveSceneCfg(num_envs=args.num_envs,env_spacing=3.,replicate_physics=True)
        scene_cfg.ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg())
        scene_cfg.robot=fullbody_robot_cfg(urdf,args.output/'usd')
        scene_cfg.feet=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*ankle_roll_link',
            update_period=0.,history_length=1,debug_vis=False)
        print('FULLBODY_PROBE building scene',flush=True)
        begin=time.monotonic()
        scene=InteractiveScene(scene_cfg)
        print('FULLBODY_PROBE scene built; resetting simulation',flush=True)
        sim.reset()
        robot=scene['robot']
        assert not robot.is_fixed_base, 'Unexpected welded/fixed base'
        body_ids=joint_indices(robot.joint_names,BODY_JOINTS)
        hand_ids=joint_indices(robot.joint_names,(*hand_joints('left'),*hand_joints('right')))
        assert robot.num_joints==51, f'Expected 29 body + 12 independent + 10 coupled joints, got {robot.num_joints}'
        mass=robot.root_physx_view.get_masses().sum(-1)
        if not torch.allclose(mass,torch.full_like(mass,audit['total_mass_kg']),atol=.02,rtol=1e-4):
            raise ValueError(f'Imported mass differs from URDF: {mass}')
        # Inspect imported runtime schema, not just our Python configuration.
        mimic=[]
        rigid=[]
        for prim in sim.stage.Traverse():
            if not str(prim.GetPath()).startswith('/World/envs/env_0/Robot/'):
                continue
            schemas=list(prim.GetAppliedSchemas())
            mimic.extend((str(prim.GetPath()),s) for s in schemas if 'MimicJoint' in s)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                disabled=prim.GetAttribute('physxRigidBody:disableGravity').Get()
                kinematic=prim.GetAttribute('physics:kinematicEnabled').Get()
                if disabled or kinematic:
                    raise ValueError(f'Non-dynamic robot body: {prim.GetPath()}')
                rigid.append(str(prim.GetPath()))
        if len(mimic)!=10:
            raise ValueError(f'Expected 10 native mimic constraints; found {len(mimic)}')
        if len(rigid)<50:
            raise ValueError('Unexpectedly reduced robot rigid-body count')
        q0=robot.data.default_joint_pos.clone()
        root=robot.data.default_root_state.clone()
        root[:,:3]+=scene.env_origins
        robot.write_root_pose_to_sim(root[:,:7])
        robot.write_root_velocity_to_sim(root[:,7:])
        robot.write_joint_state_to_sim(q0,torch.zeros_like(q0))
        robot.set_joint_position_target(q0)
        scene.reset()
        history=SonicHistory(args.num_envs,args.device)
        last=torch.zeros(args.num_envs,29,device=args.device)
        refq=q0[:,body_ids,None].transpose(1,2).repeat(1,10,1)
        refqd=torch.zeros_like(refq)
        scales=torch.tensor([m.action_scale for m in body_motors().values()],device=args.device)
        records={k:[] for k in ('root_state','joint_pos','joint_vel','targets','foot_force','normalized_action')}
        failure=None
        wall=time.monotonic()
        steps=round(args.seconds/CONTROL_DT)
        print('FULLBODY_PROBE rollout started',flush=True)
        for step in range(steps):
            terms=(robot.data.root_ang_vel_b,robot.data.joint_pos[:,body_ids]-q0[:,body_ids],
                robot.data.joint_vel[:,body_ids],last,robot.data.projected_gravity_b)
            obs=history.push(*terms)
            # Planned reference has fixed WORLD heading zero. Normalize by
            # the CURRENT pelvis yaw, never its full roll/pitch orientation.
            ori=matrix_from_quat(quat_conjugate(yaw_quat(robot.data.root_quat_w)))[:,:,:2]
            ori=ori.reshape(args.num_envs,1,6).repeat(1,10,1)
            last=model(obs,refq,refqd,ori) if model is not None else torch.zeros_like(last)
            target=q0.clone()
            target[:,body_ids]=q0[:,body_ids]+last*scales
            target[:,hand_ids]=q0[:,hand_ids]
            robot.set_joint_position_target(target)
            for _ in range(round(CONTROL_DT/PHYSICS_DT)):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(PHYSICS_DT)
            values=dict(root_state=robot.data.root_state_w,joint_pos=robot.data.joint_pos,
                joint_vel=robot.data.joint_vel,targets=target,
                foot_force=scene['feet'].data.net_forces_w,normalized_action=last)
            for key,value in values.items():
                records[key].append(value.detach().cpu().numpy().copy())
            if not all(torch.isfinite(v).all() for v in values.values()):
                failure='nonfinite_state'
                break
            height=robot.data.root_pos_w[:,2]-scene.env_origins[:,2]
            upright=-robot.data.projected_gravity_b[:,2]
            if (height<.35).any() or (upright<.5).any():
                failure='fall_or_large_tilt'
                break
            if step%50==0:
                print(f'FULLBODY_PROBE step={step} z={height.min().item():.3f} upright={upright.min().item():.3f}',flush=True)
        elapsed=time.monotonic()-wall
        arrays={k:np.asarray(v) for k,v in records.items()}
        np.savez_compressed(args.output/'trace.npz',**arrays)
        z=arrays['root_state'][...,2]
        # Averages over the final second, not an instantaneous contact spike.
        load=arrays['foot_force'][-50:,...,2].sum(-1).mean(0)
        ratio=load/(mass.cpu().numpy()*9.81)
        standing=bool(failure is None and (z[-1]>.5).all() and (z[-1]<.95).all()
            and (ratio>.5).all() and (ratio<1.5).all())
        report=dict(controller=args.controller,reference='constant nominal pose, fixed world yaw; interface probe only',
            num_envs=args.num_envs,physics_hz=200,controller_hz=50,
            duration_simulated_s=len(z)*CONTROL_DT,rollout_wall_s=elapsed,
            env_steps_per_second=len(z)*args.num_envs/elapsed,total_wall_s=time.monotonic()-begin,
            fixed_base=robot.is_fixed_base,total_mass_kg=mass.tolist(),
            rigid_body_count=len(rigid),native_mimic_constraints=mimic,
            joint_names=robot.joint_names,body_joint_indices=body_ids,hand_joint_indices=hand_ids,
            actuators=actuator_manifest(),hand_gains=dict(stiffness=1200.,damping=25.,calibrated=False),
            foot_load_over_weight=ratio.tolist(),min_pelvis_height_m=float(z.min()),
            standing_probe_passed=standing,failure=failure,physics_import_validated=True,
            full_m0_validated=False,training_started=False,optimizer_memory_measured=False,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not standing:
            raise RuntimeError('Standing probe failed; see saved trace. Do not launch manipulation training.')
    finally:
        faulthandler.cancel_dump_traceback_later()
        app.close()


if __name__=='__main__':
    main()
