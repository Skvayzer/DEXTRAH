#!/usr/bin/env python3
"""Fetch only NVIDIA's pinned navigation planner; never replace SONIC weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = 'nvidia/GEAR-SONIC'
REVISION = '6733128a3d8a523b1418b06bca3cdf61c8b0987f'
SHA256 = '39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm allocation for planner preparation')
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(REPO, 'planner_sonic.onnx', revision=REVISION,
                               local_dir=args.output))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != SHA256:
        raise ValueError('Planner checksum mismatch')
    report = dict(repo=REPO, revision=REVISION, sha256=digest, path=str(path),
                  bytes=path.stat().st_size, pretrained_controller_changed=False)
    (args.output/'planner_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
