"""Small, simulator-independent checks for checkpoint recordings."""
import hashlib
import os
from pathlib import Path
import shutil


def action_layout(contract):
    mode = contract.get('controller_mode', 'trainable_decoder')
    if mode not in ('trainable_decoder', 'frozen_pretrained_latent'):
        raise ValueError(f'Unknown recorded controller: {mode}')
    frozen = mode == 'frozen_pretrained_latent'
    count = 70 if frozen else 35
    if contract.get('action_dim', count) != count:
        raise ValueError('Checkpoint controller and action dimension disagree')
    if frozen and (contract.get('architecture') != 'frozen_pretrained_sonic_sapg_latent64_fingers6_v1'
                   or contract.get('latent_residual_scale') != .1
                   or contract.get('latent_clipping') is not False):
        raise ValueError('Unsupported frozen SONIC action contract')
    return frozen, count


def execution_means(means, *, frozen):
    # Frozen SONIC consumes unbounded latent residuals; the environment clamps
    # only the six physical finger commands after applying source action delay.
    if means.shape[-1] != (70 if frozen else 35):
        raise ValueError('Wrong checkpoint action dimension')
    return means if frozen else means.clamp(-1, 1)


def snapshot_checkpoint(source, destination):
    """Copy an open checkpoint inode; reject concurrent in-place modification.

    Atomic replacement by the trainer is harmless: our open descriptor keeps
    referring to the original inode. Never write, rename or lock the source.
    """
    source, destination = Path(source), Path(destination)
    with source.open('rb') as src, destination.open('xb') as dst:
        before = os.fstat(src.fileno())
        shutil.copyfileobj(src, dst, length=2**20)
        after = os.fstat(src.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during copy; do not load this snapshot')
    if destination.stat().st_size != before.st_size:
        raise RuntimeError('Incomplete checkpoint snapshot')
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(original_checkpoint=str(source), snapshot=str(destination), sha256=digest,
                bytes=before.st_size, source_inode=before.st_ino)
