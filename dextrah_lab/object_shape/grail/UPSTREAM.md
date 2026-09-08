# GRAIL BPS reference

`compute_bps.py` and `LICENSE` are unmodified files from
[NVlabs/GRAIL](https://github.com/NVlabs/GRAIL), commit
`aa31d8242ac79b11545b9e3635f73014a227bdfc`, retrieved 2026-09-08.

Original path: `grail/retargeting/compute_bps.py`.
These files retain the upstream NVIDIA research/evaluation-only license;
they are not relicensed under the surrounding project's license.

The viewer calls the upstream Fibonacci basis, normalization, and distance
functions. It does **not** call the USD loader or `process_usd`: our training
object is a transformed GLB export, and we always sample triangle surfaces
instead of taking the upstream small-mesh vertex-only shortcut. The default
basis count is deliberately increased from 10 to 128.
