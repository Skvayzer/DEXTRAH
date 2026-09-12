"""Strict student loading, separate from immutable source controllers."""
import hashlib
import json
from pathlib import Path
import torch

from .sonic import WEIGHTS_SHA256
from .student import ARCHITECTURE,SonicManipulationStudent
from .sapg_features import ARCHITECTURE_SAPG,SonicSapgStudent,load_sapg_actor


def load_student_checkpoint(path,sonic,device):
    path=Path(path).resolve()
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg=payload['config']
    if cfg['sonic_sha256']!=WEIGHTS_SHA256 or cfg['action_dim']!=35 or cfg['no_future_motion_input'] is not True:
        raise ValueError('Student/source action or observation contract mismatch')
    manifest_path=path.parent/'dataset_manifest.json'
    with manifest_path.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=payload['dataset_manifest_sha256']:
            raise ValueError('Student dataset manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if payload['architecture']==ARCHITECTURE:
        model=SonicManipulationStudent.from_sonic(sonic,payload['task_mean'],payload['task_variance'],cfg['hidden_dim'])
    elif payload['architecture']==ARCHITECTURE_SAPG:
        checkpoint=Path(manifest['teacher_checkpoint'])
        sapg=load_sapg_actor(checkpoint,checkpoint.parent.parent/'params/agent.yaml',cfg['source_teacher_sha256'],device)
        model=SonicSapgStudent(sonic,sapg,payload['task_mean'],payload['task_variance'],freeze_task=True)
    else:
        raise ValueError('Unknown student architecture')
    model.load_state_dict(payload['model'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite student parameter')
    return model.to(device).eval().requires_grad_(False),dict(architecture=payload['architecture'],
        checkpoint=str(path),supervised_update=payload['update'],source_sonic_sha256=WEIGHTS_SHA256,
        source_sapg_sha256=cfg['source_teacher_sha256'],action_dim=35,fullbody_rl_updates=payload['fullbody_rl_updates'],
        left_clearance_roll=cfg.get('left_clearance_roll',.2))
