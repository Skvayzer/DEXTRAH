# mujoco_warp 3.11.0 with the elliptic cone Hessian fix

Python sources of `mujoco-warp==3.11.0` (tests and test data omitted). Only
`mujoco_warp/_src/solver.py` differs: at the three elliptic cone Hessian sites
(sparse, compact, dense) `ttt = max(T^3, MJ_MINVAL)` became `T = max(T, 1e-12);
ttt = T^3`, matching C MuJoCo's `mu*N/(T*T*T)`.

Why: with T < 1e-5 the stock clamp under-weights the cone curvature, the
Newton Hessian becomes indefinite (seen in a light object's block) and Warp's
tile Cholesky (no pivot guard) returns NaN. This aborted runs 1157 and 1213.
Evidence and tools: `docs/mjlab_port_plan_20261013.md` ("NaN crash root
cause"), `scripts/mj_nan_*.py`. Replay of the captured step: stock 51-55% NaN,
fixed 0 of 481,300 solves, qacc otherwise unchanged.

Enabled with `MJ_CONE_HESSIAN_FIX=1` (sbatch prepends this directory to
PYTHONPATH; the trainer verifies it and records it in the task contract).
