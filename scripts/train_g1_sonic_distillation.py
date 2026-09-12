#!/usr/bin/env python3
"""Supervised route-2 bootstrap: saved SAPG manipulation + SONIC rehearsal.

This trains a new SONIC-derived controller, not a latent adapter. Manipulation
body histories are explicitly KINEMATIC lifts, not simulated full-body data.
No RL, no simulator gradients, no claim of manipulation success. The following
gate is closed-loop full-body object-contact validation and SAPG fine-tuning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from dextrah_lab.wholebody.bootstrap import causal_episode, split_episode, standing_rehearsal
from dextrah_lab.wholebody.contract import nominal_body_pose, BODY_JOINTS
from dextrah_lab.wholebody.reference import Segment
from dextrah_lab.wholebody.sonic import FrozenSonic, WEIGHTS_SHA256, verify_file
from dextrah_lab.wholebody.student import ARCHITECTURE, ACTION_JOINTS, SonicManipulationStudent, direct_imitation_loss
from dextrah_lab.wholebody.teacher_bridge import align_teacher_segment
from dextrah_lab.wholebody.teacher_data import validate_policy_capture
from dextrah_lab.wholebody.task_contract import TOUCH_SHA256

TEACHER_SHA256='53c4b009cdd341b4a0e007111c4009da57898dcb00d264c12c81b18817637fec'


def archive(path):
    with np.load(path,allow_pickle=False) as data:
        return dict(data)


def checksum(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def add_body_teacher(data,sonic,device,standing=False,rest_pose=None):
    converted={k:torch.as_tensor(v,device=device) for k,v in data.items()
               if k in ('proprio','task','arm_targets','finger_actions','valid','sonic_action',
                        'teacher_rnn_state_0','teacher_rnn_state_1')}
    count=len(data['proprio'])
    q=torch.as_tensor(nominal_body_pose(),device=device)[None,None].repeat(1,10,1)
    if rest_pose is not None and not standing:
        q[:]=torch.as_tensor(rest_pose,device=device)
    zero=torch.zeros_like(q)
    ori=q.new_tensor([1.,0.,0.,1.,0.,0.])[None,None].repeat(1,10,1)
    if not standing:
        token=sonic.reference_tokens(q,zero,ori)
        tokens=token.repeat(count,1)
    else:
        orientations=torch.as_tensor(data['reference_ori6'],device=device)[:,None].repeat(1,10,1)
        tokens=[]
        for a in range(0,count,256):
            size=min(256,count-a)
            tokens.append(sonic.reference_tokens(q.repeat(size,1,1),zero.repeat(size,1,1),orientations[a:a+size]))
        tokens=torch.cat(tokens)
    decoder=sonic.actor.actor_module.decoders['g1_dyn']
    labels=[]
    for a in range(0,count,256):
        labels.append(decoder(torch.cat((tokens[a:a+256],converted['proprio'][a:a+256]),-1)))
    action=torch.cat(labels)
    if standing:
        error=float((action-converted['sonic_action']).abs().max())
        if error>2e-4:
            raise ValueError(f'Reconstructed standing observation/action alignment failed: {error}')
        converted['reconstruction_max_error']=error
    converted['sonic_action']=action
    converted['tokens']=tokens
    return converted


def prepare(args,sonic):
    training=[];validation=[];reports=[];normalizer=None;checkpoint=None
    rest=nominal_body_pose()
    rest[BODY_JOINTS.index('left_shoulder_roll_joint')]=args.left_clearance_roll
    if args.policy_hz==60:
        from dextrah_lab.wholebody.source_scene import standing_reset_pose
        rest=np.asarray([standing_reset_pose()[name] for name in BODY_JOINTS],dtype=np.float32)
    for clip in sorted((args.capture/'clips').iterdir()):
        if not clip.is_dir():
            continue
        print(f'DISTILLATION_PREPARE {clip.name}',flush=True)
        meta=json.loads((clip/'metadata.json').read_text())
        expected_dim=249 if args.teacher_sha256==TOUCH_SHA256 else 224
        if meta['checkpoint_sha256']!=args.teacher_sha256 or meta['actor_dim']!=expected_dim:
            raise ValueError('Capture does not match the explicitly selected pinned teacher')
        if checkpoint is None:
            checkpoint=Path(meta['checkpoint'])
            verify_file(checkpoint,args.teacher_sha256)
        elif Path(meta['checkpoint'])!=checkpoint:
            raise ValueError('Mixed teacher checkpoints')
        data=archive(clip/'reference_trace.npz')
        norm=archive(clip/'teacher_observation_normalizer.npz')
        capture_report=validate_policy_capture(data,meta,norm)
        if normalizer is None:
            normalizer=norm
        elif any(not np.array_equal(normalizer[k],norm[k]) for k in ('mean','variance')):
            raise ValueError('Mixed source normalizers')
        boundaries=np.r_[0,np.flatnonzero(np.diff(data['resets'])!=0)+1,len(data['step'])]
        entries=[]
        for start,stop in zip(boundaries[:-1],boundaries[1:]):
            if stop-start<2:
                continue
            episode=causal_episode(data,meta,int(start),int(stop),rest_pose=rest,policy_dt=1/args.policy_hz)
            split=split_episode(len(episode['proprio']),min_steps=args.sequence+args.burn_in)
            if split is None:
                continue
            # Audit source FK/constant torso/gravity-preserving alignment, but
            # do not feed the resulting future motion to the student.
            segment=Segment(int(start),int(stop),int(data['goal_hits'][stop-1]-data['goal_hits'][start]))
            _,alignment=align_teacher_segment(data,meta,segment,args.urdf)
            if alignment['limit_violations']:
                raise ValueError('Demonstration has unhandled body limit violations')
            for target,sl in zip((training,validation),split):
                part={k:v[sl].copy() for k,v in episode.items()}
                target.append(add_body_teacher(part,sonic,args.device,rest_pose=rest))
            entries.append(dict(source_start=int(start),source_stop=int(stop),
                training_interval=[split[0].start,split[0].stop],
                validation_interval=[split[1].start,split[1].stop],
                gap_steps=split[1].start-split[0].stop,alignment=alignment))
        reports.append(dict(clip=str(clip),source_trace_sha256=checksum(clip/'reference_trace.npz'),
            normalizer_sha256=checksum(clip/'teacher_observation_normalizer.npz'),
            capture=capture_report,episodes=entries))
    if not training or not validation or normalizer is None:
        raise ValueError('No valid reset-separated training/validation demonstrations')
    standing_report=json.loads((args.standing/'validation.json').read_text())
    standing=standing_rehearsal(archive(args.standing/'trace.npz'),standing_report)
    rehearsal=add_body_teacher(standing,sonic,args.device,standing=True)
    # Preserve a temporal rehearsal holdout too. Rehearsal carries NO task
    # observations or SAPG labels from an unrelated simulation.
    split=split_episode(len(rehearsal['proprio']),min_steps=32)
    if split is None:
        raise ValueError('Standing rehearsal is too short for separate validation')
    rehearsal_train={k:v[split[0]] for k,v in rehearsal.items() if torch.is_tensor(v)}
    rehearsal_val={k:v[split[1]] for k,v in rehearsal.items() if torch.is_tensor(v)}
    report=dict(teacher_checkpoint=str(checkpoint),teacher_checkpoint_sha256=args.teacher_sha256,
        sonic_checkpoint_sha256=WEIGHTS_SHA256,clips=reports,
        training_samples=sum(len(x['proprio']) for x in training),
        validation_samples=sum(len(x['proprio']) for x in validation),
        standing_reconstruction_max_error=rehearsal['reconstruction_max_error'],
        standing_source=str(args.standing),standing_source_sha256=checksum(args.standing/'trace.npz'),
        standing_train_samples=len(rehearsal_train['proprio']),standing_validation_samples=len(rehearsal_val['proprio']),
        task_state_frame='Original reduced-teacher world/task frame; online environment must map into this frame',
        manipulation_body_data='KINEMATIC lifted SAPG right arm, nominal legs and zero base motion; NOT real full-body rollouts',
        standing_body_data='Measured pre-action states reconstructed from passing real nominal SONIC standing probe',
        split='Within-episode contiguous time holdout, 50-step separation; NOT held-out objects',
        future_motion_input=False,student_control_hz=args.policy_hz,source_teacher_control_hz=60,
        body_history_spacing_s=.02,standing_rehearsal_control_hz=50,
        task_dim=expected_dim,tactile_training=(expected_dim==249),action_joint_order=list(ACTION_JOINTS),source_body_order=list(BODY_JOINTS),
        manipulation_rest_pose=rest.tolist(),left_clearance_roll=args.left_clearance_roll)
    return training,validation,rehearsal_train,rehearsal_val,normalizer,report


def sample_windows(episodes,batch,length,rng):
    # Choose trajectories in proportion to valid window counts, not number of
    # episodes; every chosen window stays inside one temporal split/episode.
    sizes=np.asarray([len(x['proprio'])-length+1 for x in episodes])
    if (sizes<1).any():
        raise ValueError('A sequence exceeds an available episode')
    chosen=rng.choice(len(episodes),size=batch,p=sizes/sizes.sum())
    keys=('proprio','task','tokens','arm_targets','finger_actions','sonic_action','valid')
    keys+=tuple(k for k in ('teacher_rnn_state_0','teacher_rnn_state_1') if k in episodes[0])
    out={k:[] for k in keys}
    for i in chosen:
        start=int(rng.integers(sizes[i]))
        for k in keys:
            out[k].append(episodes[i][k][start:start+length])
    return {k:torch.stack(v) for k,v in out.items()}


def copied_task_state(model,data,batched):
    """Source recurrent states are meaningful ONLY for the copied SAPG LSTM."""
    if not hasattr(model,'task_encoder'):
        return None
    values=[]
    for k in ('teacher_rnn_state_0','teacher_rnn_state_1'):
        x=data[k][:,0] if batched else data[k][0:1]
        values.append(x.transpose(0,1).contiguous())
    return tuple(values)


@torch.no_grad()
def evaluate(model,episodes,rehearsal,burn_in):
    model.eval()
    squared=np.zeros(3);count=0;arm_max=0.
    for e in episodes:
        hidden=copied_task_state(model,e,False)
        for start in range(0,len(e['proprio']),256):
            stop=min(start+256,len(e['proprio']))
            action,hidden=model(e['proprio'][None,start:stop],e['task'][None,start:stop],e['tokens'][None,start:stop],hidden)
            mask=e['valid'][None,start:stop].clone()
            if start<burn_in:
                mask[:,:min(burn_in-start,stop-start)]=False
            if not mask.any():
                continue
            _,m=direct_imitation_loss(model,action,e['arm_targets'][None,start:stop],
                e['finger_actions'][None,start:stop],e['sonic_action'][None,start:stop],mask)
            n=m['manipulation_samples'];count+=n
            squared+=np.asarray([m['arm_rmse_rad'],m['finger_rmse'],m['other_body_rmse_rad']])**2*n
            arm_max=max(arm_max,m['arm_max_error_rad'])
    if not count:
        raise ValueError('No valid evaluation samples')
    errors=np.sqrt(squared/count)
    # Task-disabled rehearsal uses normalizer mean => zero task features.
    # This preserves standing in the absence of an active manipulation goal.
    action,_=model(rehearsal['proprio'][None],model.normalizer.mean[None,None].expand(1,len(rehearsal['proprio']),-1),
        rehearsal['tokens'][None],task_active=False)
    body_error=(action[0,:,:29]-rehearsal['sonic_action'])*model.body_scales
    result=dict(arm_rmse_rad=float(errors[0]),finger_rmse=float(errors[1]),other_body_rmse_rad=float(errors[2]),
        arm_max_error_rad=arm_max,samples=count,standing_rmse_rad=float(body_error.square().mean().sqrt()),
        standing_max_error_rad=float(body_error.abs().max()))
    result['selection_loss']=sum(float(x*x) for x in errors)+10*result['standing_rmse_rad']**2
    model.train()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--standing',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--urdf',type=Path)
    p.add_argument('--updates',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--sequence',type=int,default=64)
    p.add_argument('--burn-in',type=int,default=32)
    p.add_argument('--decoder-lr',type=float,default=2e-5)
    p.add_argument('--task-lr',type=float,default=3e-4)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--wandb',choices=['online','disabled'],default='online')
    p.add_argument('--reuse-sapg-features',action='store_true',help='Also copy source LSTM/MLP/finger head; no random finger relearning')
    p.add_argument('--left-clearance-roll',type=float,default=.6,help='Planned left-arm rest reference to clear the table; no welded or overridden body joints')
    p.add_argument('--teacher-sha256',choices=[TEACHER_SHA256,TOUCH_SHA256],default=TEACHER_SHA256)
    p.add_argument('--policy-hz',type=int,choices=[50,60],default=50)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise ValueError('Use Slurm and a new output directory')
    if min(args.updates,args.batch_size,args.sequence,args.burn_in)<=0:
        raise ValueError('Require positive training sizes')
    if not .2<=args.left_clearance_roll<=.8:
        raise ValueError('Left clearance reference outside the audited candidate range')
    if args.urdf is None:
        args.urdf=args.workspace/'play2perfect/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf'
    contract=json.loads(args.contract.read_text())
    if not contract['passed'] or contract['source_weights_sha256']!=WEIGHTS_SHA256:
        raise ValueError('Real SONIC/student equivalence validation must pass first')
    args.output.mkdir(parents=True)
    run=None
    try:
        torch.set_num_threads(2);torch.manual_seed(42);rng=np.random.default_rng(42)
        # Preserve strict float32 equivalence; mixed precision is a later,
        # measured optimization rather than an untested transfer change.
        torch.backends.cuda.matmul.allow_tf32=False
        sonic=FrozenSonic(args.workspace/'GRAIL',args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base',args.device)
        train,val,standing,standing_val,norm,data_report=prepare(args,sonic)
        write_json(args.output/'dataset_manifest.json',data_report)
        feature_report=None
        architecture=ARCHITECTURE
        if args.reuse_sapg_features:
            from dextrah_lab.wholebody.sapg_features import load_sapg_actor,SonicSapgStudent,ARCHITECTURE_SAPG
            source_meta=json.loads((args.capture/'clips/hammer/metadata.json').read_text())
            sapg=load_sapg_actor(source_meta['checkpoint'],Path(source_meta['source_run'])/'params/agent.yaml',args.teacher_sha256,args.device)
            model=SonicSapgStudent(sonic,sapg,norm['mean'],norm['variance'],freeze_task=True).to(args.device)
            architecture=ARCHITECTURE_SAPG
            # Verify exact source-memory semantics against actual captured
            # deterministic actions, on original 60 Hz frames before resampling.
            capture=archive(args.capture/'clips/hammer/reference_trace.npz')
            ids=np.arange(0,len(capture['step']),17)
            raw=torch.as_tensor(capture['teacher_observation'][ids],device=args.device)
            state=tuple(torch.as_tensor(capture[k][ids],device=args.device).transpose(0,1).contiguous()
                for k in ('teacher_rnn_state_0','teacher_rnn_state_1'))
            with torch.no_grad():
                features,next_state=model.task_encoder(model.normalizer(raw[:,:model.task_dim,None].transpose(1,2)),state)
                copied=model.fingers(features[:,0])
                expected=torch.as_tensor(capture['teacher_action'][ids,7:13],device=args.device)
                # Capture records the player's output, which already clips
                # its Gaussian mean to the [-1,1] environment action range.
                torch.testing.assert_close(copied.clamp(-1,1),expected,rtol=1e-5,atol=5e-5)
                actual=sapg(dict(is_train=True,prev_actions=torch.zeros(len(ids),13,device=args.device),
                    obs=raw,rnn_states=state,seq_length=1))
                torch.testing.assert_close(copied,actual['mus'][:,7:13],rtol=1e-5,atol=5e-5)
                for a,b in zip(next_state,actual['rnn_states']):
                    torch.testing.assert_close(a,b,rtol=1e-5,atol=5e-5)
            feature_report=dict(samples=len(ids),clipped_finger_max_error=float((copied.clamp(-1,1)-expected).abs().max()),
                raw_mean_vs_source_model_max_error=float((copied-actual['mus'][:,7:13]).abs().max()),
                copied_source_recurrence=True,source_arm_output_not_used_for_control=True,
                source_task_weights_frozen_during_bootstrap=True)
            write_json(args.output/'sapg_feature_equivalence.json',feature_report)
            del sapg
        else:
            model=SonicManipulationStudent.from_sonic(sonic,norm['mean'],norm['variance']).to(args.device)
        decoder_params=list(model.decoder.parameters())
        new_params=[p for name,p in model.named_parameters() if not name.startswith('decoder.') and p.requires_grad]
        optimizer=torch.optim.Adam([dict(params=decoder_params,lr=args.decoder_lr),dict(params=new_params,lr=args.task_lr)])
        config=dict(architecture=architecture,stage='supervised_kinematic_bootstrap_not_fullbody_RL',
            source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'),action_dim=35,
            decoder_lr=args.decoder_lr,task_lr=args.task_lr,updates=args.updates,batch_size=args.batch_size,
            sequence=args.sequence,burn_in=args.burn_in,source_teacher_sha256=args.teacher_sha256,
            sonic_sha256=WEIGHTS_SHA256,task_dim=model.task_dim,hidden_dim=model.hidden_dim,
            standing_weight=10.,other_body_weight=1.,no_future_motion_input=True,
            reuse_sapg_features=args.reuse_sapg_features,feature_equivalence=feature_report,
            explicit_task_activity_gate=True,left_clearance_roll=args.left_clearance_roll,
            policy_hz=args.policy_hz,body_history_spacing_s=.02)
        write_json(args.output/'config.json',config)
        if args.wandb=='online':
            import wandb
            run=wandb.init(project='adept',entity='skvayzer',
                name=f'g1_sonic_sapg_distill_{os.environ["SLURM_JOB_ID"]}',
                job_type='supervised-distillation',config=config,dir=str(args.output),mode='online')
            write_json(args.output/'wandb_run.json',dict(id=run.id,url=run.url,mode='online'))
            print(f'WANDB_ONLINE {run.url}',flush=True)
        before=evaluate(model,val,standing_val,args.burn_in)
        write_json(args.output/'initial_metrics.json',before)
        if run:
            run.log({f'validation/{k}':v for k,v in before.items()},step=0)
        print('DISTILLATION_INITIAL '+json.dumps(before),flush=True)
        best=before['selection_loss'];best_update=0
        def save(name,step,metrics):
            path=args.output/name
            temporary=path.with_suffix('.tmp')
            torch.save(dict(architecture=architecture,model=model.state_dict(),optimizer=optimizer.state_dict(),
                update=step,metrics=metrics,config=config,task_mean=norm['mean'],task_variance=norm['variance'],
                dataset_manifest_sha256=checksum(args.output/'dataset_manifest.json'),
                fullbody_rl_updates=0,closed_loop_manipulation_validated=False,
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                numpy_rng_state=rng.bit_generator.state),temporary)
            temporary.replace(path)
        save('initial_student.pt',0,before)
        save('best_student.pt',0,before)
        wall=time.monotonic()
        with (args.output/'metrics.jsonl').open('w',buffering=1) as log:
            for update in range(1,args.updates+1):
                batch=sample_windows(train,args.batch_size,args.sequence+args.burn_in,rng)
                optimizer.zero_grad(set_to_none=True)
                # Reconstruct hidden state from past observations, with no
                # gradient/decoder storage for burn-in. Never use teacher RNN
                # state as the student's state.
                with torch.no_grad():
                    _,hidden=model(batch['proprio'][:,:args.burn_in],batch['task'][:,:args.burn_in],batch['tokens'][:,:args.burn_in],
                        copied_task_state(model,batch,True))
                sl=slice(args.burn_in,None)
                action,_=model(batch['proprio'][:,sl],batch['task'][:,sl],batch['tokens'][:,sl],hidden)
                loss,metrics=direct_imitation_loss(model,action,batch['arm_targets'][:,sl],batch['finger_actions'][:,sl],
                    batch['sonic_action'][:,sl],batch['valid'][:,sl])
                count=min(128,len(standing['proprio']))
                ids=torch.as_tensor(rng.integers(len(standing['proprio']),size=count),device=args.device)
                # Standing samples are independent one-step preservation
                # queries; body history already contains ten measured steps.
                stand_action,_=model(standing['proprio'][ids,None],model.normalizer.mean[None,None].expand(count,1,-1),
                    standing['tokens'][ids,None],task_active=False)
                stand_error=(stand_action[:,0,:29]-standing['sonic_action'][ids])*model.body_scales
                stand_loss=stand_error.square().mean()
                total=loss+10.*stand_loss
                if not torch.isfinite(total):
                    raise ValueError('Nonfinite distillation loss')
                total.backward()
                gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if update%20==0 or update==1:
                    elapsed=time.monotonic()-wall
                    free,capacity=torch.cuda.mem_get_info()
                    entry=dict(update=update,elapsed_s=elapsed,**metrics,
                        standing_rmse_rad=float(stand_loss.detach().sqrt()),gradient_norm=float(gradient_norm),
                        supervised_samples_per_s=update*args.batch_size*args.sequence/elapsed,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),device_used_bytes=capacity-free)
                    log.write(json.dumps(entry)+'\n')
                    print('DISTILLATION_UPDATE '+json.dumps(entry),flush=True)
                    if run:
                        run.log({f'train/{k}':v for k,v in entry.items()},step=update)
                if update%100==0 or update==args.updates:
                    metrics=evaluate(model,val,standing_val,args.burn_in)
                    print('DISTILLATION_VALIDATION '+json.dumps(dict(update=update,**metrics)),flush=True)
                    if run:
                        run.log({f'validation/{k}':v for k,v in metrics.items()},step=update)
                    if metrics['selection_loss']<best:
                        best=metrics['selection_loss'];best_update=update
                        save('best_student.pt',update,metrics)
                    save('last_student.pt',update,metrics)
        verify_file(Path(data_report['teacher_checkpoint']),args.teacher_sha256)
        verify_file(args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base/last.pt',WEIGHTS_SHA256)
        report=dict(completed=True,supervised_updates=args.updates,fullbody_rl_updates=0,
            initial_validation=before,final_validation=metrics,best_update=best_update,
            best_selection_loss=best,source_checkpoints_unchanged=True,
            kinematic_manipulation_bootstrap_only=True,standing_rehearsal_real_physics=True,
            manipulation_success_validated=False,tactile_training=False,
            next_gate='Online full-body object/table/contact observations and loaded closed-loop validation, then SAPG fine-tuning',
            wall_s=time.monotonic()-wall,source_commit=config['source_commit'])
        write_json(args.output/'result.json',report)
        if run:
            run.summary.update(report)
        print('DISTILLATION_RESULT '+json.dumps(report),flush=True)
    except BaseException as error:
        write_json(args.output/'failure.json',dict(error=repr(error),completed=False,fullbody_rl_updates=0))
        if run:
            run.finish(exit_code=1)
            run=None
        raise
    finally:
        if run:
            run.finish()


if __name__=='__main__':
    main()
