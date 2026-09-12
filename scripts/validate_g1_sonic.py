#!/usr/bin/env python3
"""CPU model/asset contract validation; deliberately NOT a physics pass."""
import argparse
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run validation in Slurm')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/data1/users/konstantin.smirnov'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import torch
    from dextrah_lab.wholebody.sonic import FrozenSonic, SonicHistory
    from dextrah_lab.wholebody.contract import nominal_body_pose
    from dextrah_lab.wholebody.asset import prepare_urdf
    torch.set_num_threads(2)
    asset=prepare_urdf(args.workspace/'play2perfect/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf',
                       args.output/'g1_revo2_fullbody.urdf')
    bundle=args.workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base'
    model=FrozenSonic(args.workspace/'GRAIL',bundle)
    n=2
    history=SonicHistory(n)
    terms=[torch.zeros(n,d) for d in history.DIMENSIONS]
    terms[-1][:,2]=-1
    obs=history.push(*terms)
    q=torch.as_tensor(nominal_body_pose())[None,None,:].repeat(n,10,1)
    qd=torch.zeros_like(q)
    # First TWO COLUMNS, row-major; not the common column-concatenated 6D layout.
    ori=torch.tensor([1.,0.,0.,1.,0.,0.])[None,None,:].repeat(n,10,1)
    base=model(obs,q,qd,ori)
    zero=model(obs,q,qd,ori,torch.zeros(n,64))
    torch.testing.assert_close(base,zero,atol=1e-6,rtol=1e-6)
    torch.manual_seed(42)
    changed=model(obs,q,qd,ori,torch.randn(n,64))
    # Verify the actual released FSQ's surrogate gradient, not just a mock
    # decoder. These are synthetic interface tests, NOT grasping training.
    residual=torch.randn(n,64,requires_grad=True)
    differentiable=model.decode_for_imitation(obs,q,qd,ori,residual)
    inference=model(obs,q,qd,ori,residual.detach())
    torch.testing.assert_close(differentiable,inference,atol=1e-6,rtol=1e-6)
    differentiable.square().mean().backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all() and residual.grad.abs().max()>0
    assert all(parameter.grad is None for parameter in model.parameters())
    model.train(True)
    assert not model.training and not model.actor.training
    assert not any(p.requires_grad for p in model.parameters())
    report=dict(strict_checkpoint_load=True,model_parameters=sum(p.numel() for p in model.parameters()),
        zero_residual_max_error=float((base-zero).abs().max()),
        nonzero_residual_max_action_change=float((changed-base).abs().max()),
        finite_outputs=bool(torch.isfinite(changed).all()),body_actions=29,latent=64,
        imitation_gradient_verified=True,residual_gradient_max=float(residual.grad.abs().max()),
        imitation_forward_matches_inference=True,sonic_weight_gradients=False,
        source_asset_mass_kg=asset['total_mass_kg'],asset_warnings=asset['warnings'],
        physics_validated=False,training_started=False)
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
