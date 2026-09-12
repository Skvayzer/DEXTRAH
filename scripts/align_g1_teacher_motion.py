#!/usr/bin/env python3
"""Export same-robot whole-body kinematic references from a frozen SAPG clip.

These are inputs for tracking validation, not trained adapters or dynamically
feasible trajectories. Keep the source checkpoint and frame audit alongside.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dextrah_lab.wholebody.reference import successful_segments
from dextrah_lab.wholebody.teacher_bridge import align_teacher_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.clip/'metadata.json').read_text())
    if not meta.get('completed') or not meta.get('capture_reference') or meta.get('optimizer_updates') != 0:
        raise ValueError('Requires a completed, frozen-policy reference capture')
    if not np.isclose(meta['reference_dt'], 1/60):
        raise ValueError('Unexpected teacher clock')
    path = args.clip/'reference_trace.npz'
    with np.load(path, allow_pickle=False) as archive:
        trace = dict(archive)
    segments = successful_segments(trace, dt=meta['reference_dt'])
    if not segments:
        raise ValueError('No success-containing reset-separated episodes')
    converted = [align_teacher_segment(trace, meta, segment, args.urdf) for segment in segments]
    args.output.mkdir(parents=True)
    reports = []
    for index, (segment, (data, audit)) in enumerate(zip(segments, converted)):
        filename = f'reference_{index:03d}.npz'
        np.savez_compressed(args.output/filename, **data)
        reports.append(dict(file=filename, source_start=segment.start, source_stop=segment.stop,
            goal_hits=segment.goal_hits, duration_s=float(data['time_s'][-1]), **audit))
    manifest = dict(source_clip=str(args.clip.resolve()),
        checkpoint_sha256=meta['checkpoint_sha256'],
        source_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        target_dt=.02, future_reference_spacing_s=.1, references=reports,
        physics_validated=False, adapter_trained=False,
        source_policy_reused=True, object_family=meta.get('object_family'),
        object_urdf_sha256=meta.get('object_urdf_sha256'))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
