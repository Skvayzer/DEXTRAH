#!/usr/bin/env python3
"""Real checkpoint equivalence/gradient validation; no simulator or task RL."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from dextrah_lab.wholebody.contract import nominal_body_pose
from dextrah_lab.wholebody.sonic import FrozenSonic, WEIGHTS_SHA256, CONFIG_SHA256
from dextrah_lab.wholebody.student import SonicManipulationStudent, direct_imitation_loss


def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise ValueError('Use Slurm and a new output directory')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    sonic=FrozenSonic(args.workspace/'GRAIL',args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base',args.device)
    before=fingerprint(sonic)
    student=SonicManipulationStudent.from_sonic(sonic,torch.zeros(249),torch.ones(249)).to(args.device)
    q=torch.as_tensor(nominal_body_pose(),device=args.device)[None,None].repeat(4,10,1)
    qd=torch.zeros_like(q)
    ori=q.new_tensor([1.,0.,0.,1.,0.,0.])[None,None].repeat(4,10,1)
    obs=torch.randn(4,930,device=args.device)*.1
    obs[:,900:]=q.new_tensor([0.,0.,-1.]).repeat(10)
    tokens=sonic.reference_tokens(q,qd,ori)
    original=sonic(obs,q,qd,ori)
    task=torch.randn(4,1,249,device=args.device)
    task[...,224:]=0
    action,_=student(obs[:,None],task,tokens[:,None])
    error=float((action[:,0,:29]-original).abs().max())
    # Exact on CPU. CUDA GEMM shape/algorithm selection can change roundoff.
    torch.testing.assert_close(action[:,0,:29],original,rtol=1e-6,atol=1e-6)
    targets=student.body_targets(action.detach())[...,student.arm_indices]+.05
    optimizer=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=1e-5)
    loss,metrics=direct_imitation_loss(student,action,targets,
        torch.ones(4,1,6,device=args.device)*.1,original[:,None],torch.ones(4,1,dtype=torch.bool,device=args.device))
    loss.backward()
    gradients={name:float(p.grad.norm()) if p.grad is not None else 0.
        for name,p in student.named_parameters()}
    assert gradients['decoder.0.weight']>0 and gradients['decoder.12.weight']>0
    assert gradients['task_to_body.weight']>0 and gradients['fingers.weight']>0
    optimizer.step()
    assert fingerprint(sonic)==before
    assert all(not p.requires_grad and p.grad is None for p in sonic.parameters())
    report=dict(passed=True,initial_body_max_difference=error,
        source_teacher_unchanged=True,source_weights_sha256=WEIGHTS_SHA256,
        source_config_sha256=CONFIG_SHA256,trainable_decoder_parameters=sum(p.numel() for p in student.decoder.parameters()),
        total_trainable_student_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
        action_dim=35,latent_adapter=False,gradients=gradients,
        synthetic_gradient_probe=metrics,manipulation_trained=False,sapg_updates=0,
        source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'))
    args.output.mkdir(parents=True)
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
