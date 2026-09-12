#!/usr/bin/env python3
"""Fetch the pinned public SONIC manipulation body-controller bundle.

Only research/controller assets, not GRAIL's video-generation stack. This does
not launch physics or training, and never changes the existing SAPG environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

GRAIL_REVISION='aa31d8242ac79b11545b9e3635f73014a227bdfc'
MODEL_REPO='nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'
MODEL_REVISION='40e795761302e611c1e7e3a6caefdd010d56c199'
WEIGHTS_SHA256='62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grail',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm CPU allocation for setup')
    revision=subprocess.check_output(['git','-C',str(args.grail),'rev-parse','HEAD'],text=True).strip()
    if revision!=GRAIL_REVISION:
        raise ValueError(f'Unpinned GRAIL checkout: {revision}')
    from huggingface_hub import hf_hub_download
    import torch
    bundle={}
    for filename in ('model_config.yaml','last.pt'):
        path=Path(hf_hub_download(repo_id=MODEL_REPO,repo_type='dataset',revision=MODEL_REVISION,
            filename='checkpoint/SONIC/models/sonic_manipulation_base/'+filename,
            local_dir=args.output))
        with path.open('rb') as file:
            digest=hashlib.file_digest(file,'sha256').hexdigest()
        if filename=='last.pt' and digest!=WEIGHTS_SHA256:
            raise ValueError('SONIC model checksum mismatch')
        bundle[filename]=dict(path=str(path),sha256=digest,bytes=path.stat().st_size)
    report=dict(grail_revision=revision,model_repo=MODEL_REPO,model_revision=MODEL_REVISION,
        files=bundle,unsafe_pickle_globals=torch.serialization.get_unsafe_globals_in_checkpoint(bundle['last.pt']['path']),
        model_loaded=False,physics_validated=False,training_started=False)
    (args.output/'bundle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
