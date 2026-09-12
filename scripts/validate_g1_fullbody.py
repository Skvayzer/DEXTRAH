#!/usr/bin/env python3
"""Small floating-base SONIC probe. No SAPG updates or claim of M0 completion.

A constant nominal reference tests the interface; a failure alone does not
establish failure on the controller's original reference-motion distribution.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import sys


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in a Slurm GPU allocation')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--controller',choices=['sonic','pd'],default='sonic')
    p.add_argument('--student-checkpoint',type=Path,help='Route-2 student standing-regression probe; task inputs disabled, fingers neutral')
    p.add_argument('--task-clip',type=Path,help='Live one-object/table integration using this source clip and the student; no replay after reset')
    p.add_argument('--exercise-hands',action='store_true',help='Slowly close/open both hands to check native coupling')
    p.add_argument('--full-hand-range',action='store_true',help='Exercise 90 percent of each motor range at 0.15 Hz, not only 0.5 rad')
    p.add_argument('--record-envs',type=int,default=4,help='Cap trace copying; physics checks still cover all environments')
    p.add_argument('--no-physics-replication',action='store_true',help='Diagnose native constraint replication separately')
    p.add_argument('--mimic-frequency',type=float,default=100.,help='Hz; zero tests the non-compliant native constraint')
    p.add_argument('--mimic-damping',type=float,default=1.)
    p.add_argument('--hand-stiffness',type=float,default=1200.)
    p.add_argument('--hand-damping',type=float,default=25.)
    p.add_argument('--position-iterations',type=int,default=8)
    p.add_argument('--velocity-iterations',type=int,default=4)
    p.add_argument('--physics-hz',type=int,choices=[200,400,800,1000],default=200)
    p.add_argument('--bound-distal-speed',action='store_true',help='Restore importer-ignored passive URDF limits and compatible motor speeds')
    p.add_argument('--hand-armature',type=float,default=0.,help='DIAGNOSTIC reflected motor inertia kg m^2, not measured hardware data')
    p.add_argument('--diagnose-contacts',action='store_true',help='Identify thumb/index self-contact partners in a small probe')
    p.add_argument('--contact-offset',type=float,default=None,help='Diagnostic robot contact generation distance, metres')
    p.add_argument('--convex-decomposition',action='store_true',help='Diagnose convex-hull self-contact artifacts using closer collision geometry')
    p.add_argument('--reference',type=Path,help='Audited aligned teacher reference NPZ; tracking-only test without objects')
    p.add_argument('--tracking-rms-tolerance',type=float,default=.03,help='Metres, after the settling interval')
    p.add_argument('--tracking-max-tolerance',type=float,default=.08,help='Metres, after the settling interval')
    p.add_argument('--filter-thumb-housing',action='store_true',help='DIAGNOSTIC isolate the measured proximal-thumb/palm collision pair only')
    p.add_argument('--decompose-palms',action='store_true',help='Fix the audited palm convex-hull artifact without excluding contact pairs')
    p.add_argument('--palm-shrink-wrap',action='store_true',help='Project decomposed palm hulls onto original CAD surface')
    p.add_argument('--disable-self-collisions',action='store_true',help='Diagnostic ONLY; never a manipulation-ready result')
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    provenance=dict(commit=os.environ.get('FULLBODY_SOURCE_COMMIT'),source_root=str(project),
        sha256={str(path.relative_to(project)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__).resolve(),*sorted((project/'dextrah_lab/wholebody').glob('*.py'))]})
    # Isaac's URDF importer uses the export directory to author USD sublayers.
    # A relative directory can generate broken /configuration/... references.
    args.output=args.output.resolve()
    if args.num_envs < 1 or args.seconds < 2 or args.record_envs<1:
        raise ValueError('Invalid probe size/duration')
    if args.full_hand_range and not args.exercise_hands:
        raise ValueError('--full-hand-range requires --exercise-hands')
    if args.student_checkpoint and (args.reference or args.controller!='sonic'):
        raise ValueError('Student standing regression uses only a nominal reference and SONIC source')
    if args.task_clip and (not args.student_checkpoint or args.exercise_hands):
        raise ValueError('Live manipulation requires a student and cannot use diagnostic finger cycles')
    if min(args.tracking_rms_tolerance,args.tracking_max_tolerance)<=0:
        raise ValueError('Tracking tolerances must be positive')
    if args.diagnose_contacts and args.num_envs > 16:
        raise ValueError('Per-pair contact diagnostics are restricted to <=16 environments')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    args.headless=True
    app=AppLauncher(args).app
    stack_log=(args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(30, repeat=True, file=stack_log)
    try:
        import numpy as np
        import torch
        from pxr import Usd, UsdPhysics, PhysxSchema
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.utils.math import matrix_from_quat, quat_conjugate, yaw_quat
        from dextrah_lab.wholebody.asset import prepare_urdf
        from dextrah_lab.wholebody.asset_cfg import fullbody_robot_cfg
        from dextrah_lab.wholebody.actuators import body_motors, actuator_manifest
        from dextrah_lab.wholebody.importer import configure_native_mimics, filter_thumb_housing_pairs, decompose_palm_colliders, configure_collision_offsets
        from dextrah_lab.wholebody.contract import BODY_JOINTS, hand_joints, joint_indices, PHYSICS_DT, CONTROL_DT
        from dextrah_lab.wholebody.sonic import FrozenSonic, SonicHistory
        from dextrah_lab.wholebody.teacher_bridge import future_body_reference
        torch.set_num_threads(2)
        torch.manual_seed(42)
        urdf=args.output/'g1_revo2_fullbody.urdf'
        audit=prepare_urdf(args.workspace/'play2perfect/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf',urdf)
        model=(FrozenSonic(args.workspace/'GRAIL',
            args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base',args.device)
            if args.controller=='sonic' else None)
        student=student_metadata=None
        student_hidden=None
        if args.student_checkpoint:
            from dextrah_lab.wholebody.student_checkpoint import load_student_checkpoint
            student,student_metadata=load_student_checkpoint(args.student_checkpoint,model,args.device)
        physics_dt=1/args.physics_hz
        cfg=sim_utils.SimulationCfg(dt=physics_dt,device=args.device,render_interval=round(CONTROL_DT/physics_dt),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
        cfg.physx.gpu_max_rigid_contact_count=2**20
        cfg.physx.gpu_max_rigid_patch_count=2**18
        cfg.physx.gpu_found_lost_pairs_capacity=2**20
        cfg.physx.gpu_found_lost_aggregate_pairs_capacity=2**20
        cfg.physx.gpu_total_aggregate_pairs_capacity=2**20
        cfg.physx.gpu_collision_stack_size=2**26
        sim=sim_utils.SimulationContext(cfg)
        scene_cfg=InteractiveSceneCfg(num_envs=args.num_envs,env_spacing=3.,replicate_physics=not args.no_physics_replication)
        scene_cfg.ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg())
        scene_cfg.robot=fullbody_robot_cfg(urdf,args.output/'usd',
            hand_stiffness=args.hand_stiffness,hand_damping=args.hand_damping,
            self_collision=not args.disable_self_collisions,position_iterations=args.position_iterations,
            bound_distal_speed=args.bound_distal_speed,hand_armature=args.hand_armature,
            contact_offset=args.contact_offset,
            collider_type='convex_decomposition' if args.convex_decomposition else 'convex_hull',
            velocity_iterations=args.velocity_iterations)
        scene_cfg.feet=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*ankle_roll_link',
            update_period=0.,history_length=1,debug_vis=False)
        live_task=None
        if args.task_clip:
            from dextrah_lab.wholebody.clip_task import LiveClipTask
            live_task=LiveClipTask(args.task_clip,
                args.workspace/'play2perfect/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf',
                args.output,args.device)
            live_task.add_assets(scene_cfg)
        if args.diagnose_contacts:
            scene_cfg.all_body_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',
                update_period=0.,history_length=1,debug_vis=False)
        print('FULLBODY_PROBE building scene',flush=True)
        begin=time.monotonic()
        scene=InteractiveScene(scene_cfg)
        palm_collision_configuration=decompose_palm_colliders(sim.stage,args.num_envs,args.palm_shrink_wrap) if args.decompose_palms else None
        contact_offset_configuration=configure_collision_offsets(sim.stage,args.num_envs,args.contact_offset) if args.contact_offset is not None else None
        coupling_config=configure_native_mimics(sim.stage,audit['mimic_relations'],args.num_envs,
            frequency=args.mimic_frequency,damping_ratio=args.mimic_damping)
        filtered_housing_pairs=filter_thumb_housing_pairs(sim.stage,args.num_envs) if args.filter_thumb_housing else []
        contact_diagnostics={}
        if args.diagnose_contacts:
            paths=[str(p.GetPath()) for p in sim.stage.Traverse()
                   if str(p.GetPath()).startswith('/World/envs/env_0/Robot/') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for side in ('left','right'):
                for finger in ('thumb','index'):
                    source=next(path for path in paths if path.rsplit('/',1)[-1].lower()==f'{side}_{finger}_distal_link')
                    partners=[path for path in paths if path!=source]
                    sensor=ContactSensor(ContactSensorCfg(
                        prim_path=source.replace('/env_0/','/env_.*/'),update_period=0.,
                        filter_prim_paths_expr=[path.replace('/env_0/','/env_.*/') for path in partners]))
                    name=f'{side}_{finger}_contacts'
                    scene.sensors[name]=sensor
                    contact_diagnostics[name]=dict(sensor=sensor,partners=partners,peak=None,net_peak=0.)
                    # Make reporting observational and independent of threshold.
                    for env_path in scene.env_prim_paths:
                        prim=sim.stage.GetPrimAtPath(source.replace('/World/envs/env_0',env_path))
                        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
        print('FULLBODY_PROBE scene built; resetting simulation',flush=True)
        sim.reset()
        robot=scene['robot']
        assert not robot.is_fixed_base, 'Unexpected welded/fixed base'
        body_ids=joint_indices(robot.joint_names,BODY_JOINTS)
        hand_ids=joint_indices(robot.joint_names,(*hand_joints('left'),*hand_joints('right')))
        mimic_target_ids=joint_indices(robot.joint_names,tuple(audit['mimic_relations']))
        mimic_source_ids=joint_indices(robot.joint_names,tuple(
            r['joint'] for r in audit['mimic_relations'].values()))
        mimic_scale=torch.tensor([float(r['multiplier']) for r in audit['mimic_relations'].values()],device=args.device)
        mimic_offset=torch.tensor([float(r.get('offset',0)) for r in audit['mimic_relations'].values()],device=args.device)
        assert robot.num_joints==51, f'Expected 29 body + 12 independent + 10 coupled joints, got {robot.num_joints}'
        mass=robot.root_physx_view.get_masses().sum(-1)
        if not torch.allclose(mass,torch.full_like(mass,audit['total_mass_kg']),atol=.02,rtol=1e-4):
            raise ValueError(f'Imported mass differs from URDF: {mass}')
        # Inspect imported runtime schema, not just our Python configuration.
        mimic=[]
        mimic_properties=[]
        rigid=[]
        for prim in sim.stage.Traverse():
            if not str(prim.GetPath()).startswith('/World/envs/env_0/Robot/'):
                continue
            schemas=list(prim.GetAppliedSchemas())
            mimic.extend((str(prim.GetPath()),s) for s in schemas if 'MimicJoint' in s)
            if any('MimicJoint' in s for s in schemas):
                mimic_properties.append(dict(path=str(prim.GetPath()),
                    attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes()
                                if 'mimic' in a.GetName().lower()},
                    relations={r.GetName():[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()
                               if 'mimic' in r.GetName().lower()}))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                disabled=prim.GetAttribute('physxRigidBody:disableGravity').Get()
                kinematic=prim.GetAttribute('physics:kinematicEnabled').Get()
                if disabled or kinematic:
                    raise ValueError(f'Non-dynamic robot body: {prim.GetPath()}')
                rigid.append(str(prim.GetPath()))
        (args.output/'imported_couplings.json').write_text(json.dumps(mimic_properties,indent=2)+'\n')
        # Runtime values matter: IsaacLab can overwrite importer-authored gains.
        motor_state={key:getattr(robot.data,key)[0].cpu().tolist() for key in
            ('joint_stiffness','joint_damping','joint_armature','joint_effort_limits','joint_vel_limits')}
        (args.output/'runtime_motors.json').write_text(json.dumps(dict(
            joint_names=robot.joint_names,**motor_state),indent=2)+'\n')
        collision_audit=[]
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_audit.append(dict(path=str(prim.GetPath()), attributes={
                    name:str(prim.GetAttribute(name).Get()) for name in (
                        'physxCollision:contactOffset','physxCollision:restOffset','physics:approximation')}))
        (args.output/'collision_audit.json').write_text(json.dumps(collision_audit,indent=2)+'\n')
        if len(mimic)!=10:
            raise ValueError(f'Expected 10 native mimic constraints; found {len(mimic)}')
        if len(rigid)<50:
            raise ValueError('Unexpectedly reduced robot rigid-body count')
        q0=robot.data.default_joint_pos.clone()
        if args.full_hand_range:
            import xml.etree.ElementTree as ET
            joints={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')}
            hand_upper=q0.new_tensor([float(joints[robot.joint_names[i]].find('limit').get('upper')) for i in hand_ids])
            hand_amplitude=.9*(hand_upper-q0[:,hand_ids])
        else:
            hand_amplitude=.5
        root=robot.data.default_root_state.clone()
        root[:,:3]+=scene.env_origins
        robot.write_root_pose_to_sim(root[:,:7])
        robot.write_root_velocity_to_sim(root[:,7:])
        robot.write_joint_state_to_sim(q0,torch.zeros_like(q0))
        robot.set_joint_position_target(q0)
        scene.reset()
        if live_task is not None:
            live_task.reset(scene,audit)
            scene.reset()
        teacher_reference=None
        if args.reference:
            manifest=json.loads((args.reference.parent/'manifest.json').read_text())
            entries=[item for item in manifest['references'] if item['file']==args.reference.name]
            if len(entries)!=1 or entries[0]['limit_violations'] or not entries[0]['same_robot_kinematic_alignment']:
                raise ValueError('Reference must have passed the explicit kinematic audit')
            if entries[0]['body_joint_order']!=list(BODY_JOINTS):
                raise ValueError('Reference body order mismatch')
            with np.load(args.reference,allow_pickle=False) as archive:
                teacher_reference=dict(archive)
            if not np.allclose(teacher_reference['root_pose'],[0.,0.,.76,1.,0.,0.,0.],atol=1e-6):
                raise ValueError('This first tracking probe requires the nominal world-root reference')
            initial=q0.clone()
            initial[:,body_ids]=torch.as_tensor(teacher_reference['body_q'][0],device=args.device,dtype=q0.dtype)
            robot.write_joint_state_to_sim(initial,torch.zeros_like(initial))
            robot.set_joint_position_target(initial)
            scene.reset()
        history=SonicHistory(args.num_envs,args.device)
        last=torch.zeros(args.num_envs,29,device=args.device)
        refq=q0[:,body_ids,None].transpose(1,2).repeat(1,10,1)
        refqd=torch.zeros_like(refq)
        scales=torch.tensor([m.action_scale for m in body_motors().values()],device=args.device)
        records={k:[] for k in ('root_state','joint_pos','joint_vel','targets','foot_force','normalized_action')}
        if live_task is not None:
            records.update({k:[] for k in ('object_pose','object_force','task_goal_error')})
        if teacher_reference is not None:
            records.update({k:[] for k in ('sonic_proprio','reference_q','reference_qd','reference_ori6',
                'wrist_pose','reference_wrist_pose','reference_time_s')})
            wrist_id=robot.body_names.index('right_wrist_yaw_link')
        record_n=min(args.record_envs,args.num_envs)
        load_history=torch.zeros(50,args.num_envs,device=args.device)
        min_height=float('inf')
        max_coupling_error=0.
        all_body_contact_peak=None
        device_peak_used_bytes=0
        hand_range=torch.zeros(args.num_envs,len(hand_ids),device=args.device)
        failure=None
        clamped_body_targets=0
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
            if teacher_reference is not None:
                # First second is a stationary settling reference. Afterwards
                # follow a planned successful teacher segment (not live future
                # student states); fingers remain neutral in this body-only test.
                reference_time=max(0.,step*CONTROL_DT-1.)
                rq,rqd,_=future_body_reference(teacher_reference,reference_time)
                if step*CONTROL_DT<1.:
                    rq[:]=teacher_reference['body_q'][0]
                    rqd[:]=0.
                refq=torch.as_tensor(rq,device=args.device,dtype=q0.dtype)[None].repeat(args.num_envs,1,1)
                refqd=torch.as_tensor(rqd,device=args.device,dtype=q0.dtype)[None].repeat(args.num_envs,1,1)
            if student is not None:
                with torch.no_grad():
                    tokens=model.reference_tokens(refq,refqd,ori)
                    task=(live_task.observation()[:,None] if live_task is not None
                          else student.normalizer.mean[None,None].expand(args.num_envs,1,-1))
                    predicted,student_hidden=student(obs[:,None],task,tokens[:,None],student_hidden,task_active=live_task is not None)
                    last=predicted[:,0,:29]
                if not torch.isfinite(last).all() or (last.abs()>20).any():
                    raise ValueError('Student command is nonfinite or exceeds the SONIC action limit')
            else:
                last=model(obs,refq,refqd,ori) if model is not None else torch.zeros_like(last)
            target=q0.clone()
            target[:,body_ids]=q0[:,body_ids]+last*scales
            target[:,hand_ids]=q0[:,hand_ids]
            if live_task is not None:
                body_target=target[:,body_ids]
                limits=robot.data.joint_pos_limits[:,body_ids]
                bounded=body_target.clamp(limits[...,0],limits[...,1])
                clamped_body_targets+=int((bounded!=body_target).sum())
                target[:,body_ids]=bounded
                last=(bounded-q0[:,body_ids])/scales
                live_task.set_finger_targets(target,predicted[:,0,29:])
            if args.exercise_hands:
                frequency=.15 if args.full_hand_range else .3
                target[:,hand_ids]+=hand_amplitude*.5*(1-np.cos(2*np.pi*frequency*step*CONTROL_DT))
            robot.set_joint_position_target(target)
            for _ in range(round(CONTROL_DT/physics_dt)):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(physics_dt)
                for item in contact_diagnostics.values():
                    forces=item['sensor'].data.force_matrix_w.norm(dim=-1).amax(dim=(0,1))
                    item['peak']=forces if item['peak'] is None else torch.maximum(item['peak'],forces)
                    item['net_peak']=max(item['net_peak'],float(item['sensor'].data.net_forces_w.norm(dim=-1).max()))
                if args.diagnose_contacts:
                    forces=scene['all_body_contacts'].data.net_forces_w.norm(dim=-1).amax(0)
                    all_body_contact_peak=forces if all_body_contact_peak is None else torch.maximum(all_body_contact_peak,forces)
            values=dict(root_state=robot.data.root_state_w,joint_pos=robot.data.joint_pos,
                joint_vel=robot.data.joint_vel,targets=target,
                foot_force=scene['feet'].data.net_forces_w,normalized_action=last)
            if live_task is not None:
                values.update(live_task.record())
            if teacher_reference is not None:
                from dextrah_lab.wholebody.reference import _retime_pose
                query=min(max(0.,(step+1)*CONTROL_DT-1.),teacher_reference['time_s'][-1])
                reference_wrist=_retime_pose(teacher_reference['time_s'],teacher_reference['wrist_pose'],np.array([query]))[0]
                values.update(sonic_proprio=obs,reference_q=refq,reference_qd=refqd,reference_ori6=ori,
                    wrist_pose=torch.cat((robot.data.body_pos_w[:,wrist_id]-scene.env_origins,
                                          robot.data.body_quat_w[:,wrist_id]),-1),
                    reference_wrist_pose=torch.as_tensor(reference_wrist,device=args.device,dtype=q0.dtype)[None].repeat(args.num_envs,1),
                    reference_time_s=torch.full((args.num_envs,),reference_time,device=args.device))
            for key,value in values.items():
                records[key].append(value[:record_n].detach().cpu().numpy().copy())
            load_history[step%50]=values['foot_force'][...,2].sum(-1)
            coupling_error=(robot.data.joint_pos[:,mimic_target_ids]-
                robot.data.joint_pos[:,mimic_source_ids]*mimic_scale-mimic_offset).abs().max()
            max_coupling_error=max(max_coupling_error,float(coupling_error))
            hand_range=torch.maximum(hand_range,(robot.data.joint_pos[:,hand_ids]-q0[:,hand_ids]).abs())
            if not all(torch.isfinite(v).all() for v in values.values()):
                failure='nonfinite_state'
                break
            height=robot.data.root_pos_w[:,2]-scene.env_origins[:,2]
            min_height=min(min_height,float(height.min()))
            upright=-robot.data.projected_gravity_b[:,2]
            if (height<.35).any() or (upright<.5).any():
                failure='fall_or_large_tilt'
                break
            if step%50==0:
                free,total=torch.cuda.mem_get_info()
                device_peak_used_bytes=max(device_peak_used_bytes,total-free)
                print(f'FULLBODY_PROBE step={step} z={height.min().item():.3f} upright={upright.min().item():.3f}',flush=True)
        elapsed=time.monotonic()-wall
        arrays={k:np.asarray(v) for k,v in records.items()}
        np.savez_compressed(args.output/'trace.npz',**arrays)
        z=arrays['root_state'][...,2]
        # Averages over the final second, not an instantaneous contact spike.
        load=load_history[:min(step+1,50)].mean(0).cpu().numpy()
        ratio=load/(mass.cpu().numpy()*9.81)
        final_height=height.cpu().numpy()
        coupling_passed=max_coupling_error<.03
        hands_moved=bool((hand_range>.1).all()) if args.exercise_hands else None
        standing=bool(failure is None and (final_height>.5).all() and (final_height<.95).all()
            and (ratio>.5).all() and (ratio<1.5).all())
        report=dict(code_provenance=provenance,controller=args.controller,reference='constant nominal pose, fixed world yaw; interface probe only',
            num_envs=args.num_envs,physics_hz=args.physics_hz,controller_hz=50,
            duration_simulated_s=len(z)*CONTROL_DT,rollout_wall_s=elapsed,
            env_steps_per_second=len(z)*args.num_envs/elapsed,total_wall_s=time.monotonic()-begin,
            fixed_base=robot.is_fixed_base,total_mass_kg_min=float(mass.min()),total_mass_kg_max=float(mass.max()),
            rigid_body_count=len(rigid),native_mimic_constraints=mimic,
            joint_names=robot.joint_names,body_joint_indices=body_ids,hand_joint_indices=hand_ids,
            actuators=actuator_manifest(),hand_gains=dict(stiffness=args.hand_stiffness,damping=args.hand_damping,calibrated=False),
            foot_load_over_weight_min=float(ratio.min()),foot_load_over_weight_max=float(ratio.max()),
            min_pelvis_height_m=min_height,recorded_envs=record_n,
            hand_exercise=args.exercise_hands,all_independent_fingers_moved=hands_moved,
            full_hand_range_exercise=args.full_hand_range,
            max_native_coupling_error_rad=max_coupling_error,native_coupling_passed=coupling_passed,
            standing_probe_passed=standing,failure=failure,physics_import_validated=True,
            full_m0_validated=False,training_started=False,optimizer_memory_measured=False,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        report['device_peak_used_bytes_sampled']=device_peak_used_bytes
        report['student']=student_metadata
        report['student_task_inputs_disabled']=student is not None and live_task is None
        report['live_task']=live_task.report() if live_task is not None else None
        report['body_target_clamped_channels']=clamped_body_targets
        report['grasping_validated']=False
        report['native_coupling_configuration']=coupling_config
        report['replicate_physics']=not args.no_physics_replication
        report['self_collision']=not args.disable_self_collisions
        report['position_iterations']=args.position_iterations
        report['velocity_iterations']=args.velocity_iterations
        report['diagnostic_filtered_thumb_housing_pairs']=filtered_housing_pairs
        report['housing_collision_model_validated']=False
        report['palm_collision_configuration']=palm_collision_configuration
        report['bound_distal_speed']=args.bound_distal_speed
        report['hand_motor_armature_kg_m2']=args.hand_armature
        report['hand_motor_armature_calibrated']=False
        report['contact_offset_override_m']=args.contact_offset
        report['contact_offset_configuration']=contact_offset_configuration
        report['convex_decomposition']=args.convex_decomposition
        report['self_contact_peak_forces_n']={name:{path:float(force) for path,force in
            zip(item['partners'],item['peak'].cpu()) if force>.01}
            for name,item in contact_diagnostics.items()}
        report['finger_net_contact_peak_n']={name:item['net_peak'] for name,item in contact_diagnostics.items()}
        if all_body_contact_peak is not None:
            report['all_body_contact_peak_n']={name:float(force) for name,force in
                zip(scene['all_body_contacts'].body_names,all_body_contact_peak.cpu()) if force>.01}
        tracking_passed=None
        if teacher_reference is not None:
            wrist_error=np.linalg.norm(arrays['wrist_pose'][...,:3]-arrays['reference_wrist_pose'][...,:3],axis=-1)
            dot=np.abs(np.sum(arrays['wrist_pose'][...,3:]*arrays['reference_wrist_pose'][...,3:],axis=-1))
            angular_error=2*np.arccos(np.clip(dot,0.,1.))
            active=arrays['reference_time_s']>0.
            position_rms=float(np.sqrt(np.mean(wrist_error[active]**2))) if active.any() else float('inf')
            position_max=float(wrist_error[active].max()) if active.any() else float('inf')
            rotation_rms=float(np.sqrt(np.mean(angular_error[active]**2))) if active.any() else float('inf')
            rotation_max=float(angular_error[active].max()) if active.any() else float('inf')
            tracking_passed=(position_rms<=args.tracking_rms_tolerance and position_max<=args.tracking_max_tolerance
                             and rotation_rms<=.15 and rotation_max<=.3)
            report.update(reference=str(args.reference.resolve()),
                reference_type='aligned successful SAPG segment; body tracking only, neutral fingers and no objects',
                recorded_wrist_error_rms_m=float(np.sqrt(np.mean(wrist_error**2))),
                recorded_wrist_error_max_m=float(wrist_error.max()),
                tracking_error_after_settle=dict(position_rms_m=position_rms,position_max_m=position_max,
                    orientation_rms_rad=rotation_rms,orientation_max_rad=rotation_max),
                tracking_tolerances=dict(position_rms_m=args.tracking_rms_tolerance,
                    position_max_m=args.tracking_max_tolerance,orientation_rms_rad=.15,orientation_max_rad=.3),
                reference_tracking_passed=tracking_passed,
                grasping_validated=False)
        (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not standing or not coupling_passed or hands_moved is False or tracking_passed is False:
            raise RuntimeError(f'Probe failed: standing={standing}, coupling={coupling_passed}, fingers_moved={hands_moved}, tracking={tracking_passed}. Do not launch manipulation training.')
    except Exception as error:
        # Kit's fast shutdown can exit(0) before a pending exception is printed.
        # Persist the original error FIRST and return a failure to Slurm.
        (args.output/'failure.json').write_text(json.dumps(dict(
            error=repr(error),traceback=traceback.format_exc(),training_started=False),indent=2)+'\n')
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        # This workstation's IsaacLab stop callback deliberately renders until
        # play resumes. Unsubscribe it before App.close(), otherwise shutdown
        # hangs and can hide the ORIGINAL validation exception.
        if 'sim' in locals():
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close()
        faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':
    main()
