#!/usr/bin/env python3
"""Measure old SAPG target reachability through frozen SONIC on REAL states.

This optimizes per-state latent labels, not adapter weights. Object-free body
tracking states do not validate grasping or replace full-body task collection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from dextrah_lab.wholebody.sonic import FrozenSonic, WEIGHTS_SHA256
from dextrah_lab.wholebody.teacher_data import validate_policy_capture
from dextrah_lab.wholebody.transfer_labels import sample_teacher_commands
from dextrah_lab.wholebody.transfer import fit_teacher_arm_targets


def checksum(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def archive(path):
    with np.load(path,allow_pickle=False) as data:
        return dict(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run action fitting in a Slurm allocation')
    if min(args.samples,args.steps,args.batch_size)<1 or args.output.exists():
        raise ValueError('Require positive sizes and new output directory')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    validation=json.loads((args.probe/'validation.json').read_text())
    if not validation['standing_probe_passed'] or not validation['native_coupling_passed']:
        raise ValueError('Do not fit labels on a fallen or mechanically unstable body')
    reference_path=Path(validation['reference'])
    manifest=json.loads((reference_path.parent/'manifest.json').read_text())
    reference=archive(reference_path)
    clip=Path(manifest['source_clip'])
    meta=json.loads((clip/'metadata.json').read_text())
    source_path=clip/'reference_trace.npz'
    if checksum(source_path)!=manifest['source_trace_sha256'] or meta['checkpoint_sha256']!=manifest['checkpoint_sha256']:
        raise ValueError('Aligned reference/teacher provenance changed')
    teacher=archive(source_path)
    validate_policy_capture(teacher,meta,archive(clip/'teacher_observation_normalizer.npz'))
    trace=archive(args.probe/'trace.npz')
    # Env zero only: identical unperturbed copies must not inflate evidence.
    times=trace['reference_time_s'][:,0]
    labels=sample_teacher_commands(reference,teacher,meta,times)
    candidates=np.flatnonzero(labels['valid']&(times>0.))
    if not len(candidates):
        raise ValueError('No valid active tracking states')
    selected=candidates[np.unique(np.linspace(0,len(candidates)-1,min(args.samples,len(candidates))).astype(int))]
    sonic=FrozenSonic(args.workspace/'GRAIL',
        args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base',args.device)
    result={}
    for offset in range(0,len(selected),args.batch_size):
        ids=selected[offset:offset+args.batch_size]
        tensors=[torch.as_tensor(trace[key][ids,0],dtype=torch.float32,device=args.device)
                 for key in ('sonic_proprio','reference_q','reference_qd','reference_ori6')]
        targets=torch.as_tensor(labels['right_arm_targets'][ids],dtype=torch.float32,device=args.device)
        fit=fit_teacher_arm_targets(sonic,*tensors,targets,steps=args.steps)
        for key,value in vars(fit).items():
            result.setdefault(key,[]).append(value.cpu().numpy())
        print(f'Fitted {min(offset+args.batch_size,len(selected))}/{len(selected)} real-state targets',flush=True)
    result={key:np.concatenate(values) for key,values in result.items()}
    report=dict(source_probe=str(args.probe.resolve()),source_probe_trace_sha256=checksum(args.probe/'trace.npz'),
        source_reference=str(reference_path),source_reference_sha256=checksum(reference_path),
        teacher_checkpoint_sha256=meta['checkpoint_sha256'],sonic_checkpoint_sha256=WEIGHTS_SHA256,
        source_teacher_trace_sha256=manifest['source_trace_sha256'],samples=len(selected),steps=args.steps,
        accepted_labels=int(result['accepted'].sum()),
        arm_error_median_rad=float(np.median(result['arm_error_rad'])),
        arm_error_max_rad=float(result['arm_error_rad'].max()),
        baseline_arm_error_median_rad=float(np.median(result['baseline_arm_error_rad'])),
        other_body_error_max_rad=float(result['other_body_error_rad'].max()),
        tolerances_rad=dict(arm=.05,other_body=.05),
        real_fullbody_proprioception=True,source_tracking_passed=validation.get('reference_tracking_passed'),
        diagnostic_collision_filters=validation['diagnostic_filtered_thumb_housing_pairs'],
        source_has_objects=False,adapter_trained=False,sapg_updates=0,grasping_validated=False,
        usable_as_complete_adapter_training_dataset=False)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'fitted_labels.npz',**result,
        probe_frame=selected,teacher_frame=labels['source_sample'][selected],
        right_arm_targets=labels['right_arm_targets'][selected],
        right_finger_actions=labels['right_finger_actions'][selected])
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
