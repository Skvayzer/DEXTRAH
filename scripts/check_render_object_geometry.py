#!/usr/bin/env python3
"""Compare the simulated object geometry with the URDF the video renderer draws.

Builds the unchanged recording scene (no policy, no stepping beyond reset) and,
for the predeclared hammer/brush/spatula environments, reports the USD world
bounding box of each environment's Object prim in the object's own frame
(including any spawn scale) next to the renderer's URDF mesh extents/offset.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=120)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.headless = True
    app = AppLauncher(args).app
    report = dict(completed=False)
    try:
        import numpy as np
        import torch
        import trimesh
        import yourdfpy
        from pxr import Usd, UsdGeom, Gf
        import isaaclab.sim as sim_utils
        from dextrah_lab.wholebody.sonic import FrozenSonic
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.object_shape.evaluation import select_family_envs
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        from isaacsimenvs.tasks.play.pose_viewer import object_urdf_for_env
        ws = Path('/data1/users/konstantin.smirnov')
        cfg, _ = source_task_config(args.output, args.num_envs, args.device)
        cfg.seed = 42
        cfg.sonic_body.controller_mode = 'frozen_pretrained_latent'
        sonic = FrozenSonic(ws/'GRAIL', ws/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        env = G1SonicTouchEnv(cfg, sonic=sonic)
        env.reset()
        stage = sim_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        assignment = env._object_asset_index_per_env.cpu().numpy()
        selected = select_family_envs(env._object_urdf_paths, assignment, ['hammer', 'brush', 'spatula'])
        out = {}
        for family, i in selected.items():
            text, path = object_urdf_for_env(env, i)
            mesh = yourdfpy.URDF.load(str(path)).scene.dump(concatenate=True)
            prim = stage.GetPrimAtPath(f'/World/envs/env_{i}/Object')
            world = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            # Object-frame bound: transform by inverse of the measured object pose.
            pos = env.object.data.root_pos_w[i].cpu().numpy(); quat = env.object.data.root_quat_w[i].cpu().numpy()
            local_box = cache.ComputeLocalBound(prim)
            rng = local_box.ComputeAlignedRange()
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            scale = [Gf.Vec3d(*xf.GetRow3(k)).GetLength() for k in range(3)]
            out[family] = dict(env=int(i), asset_index=int(assignment[i]), urdf=str(path),
                renderer_mesh_extents=mesh.extents.round(4).tolist(), renderer_mesh_center=mesh.bounds.mean(0).round(4).tolist(),
                usd_local_bound_min=list(map(float, rng.GetMin())), usd_local_bound_max=list(map(float, rng.GetMax())),
                usd_local_extents=[float(b-a) for a, b in zip(rng.GetMin(), rng.GetMax())],
                usd_world_extents=[float(b-a) for a, b in zip(world.GetMin(), world.GetMax())],
                prim_world_scale=scale, object_scale_per_env=env._object_scale_per_env[i].cpu().tolist(),
                measured_root_pos=pos.round(4).tolist(), measured_root_quat_wxyz=quat.round(4).tolist(),
                usd_prim_world_translation=list(map(float, xf.ExtractTranslation())))
            print('GEOMETRY '+json.dumps({family: out[family]}), flush=True)
        report.update(completed=True, objects=out)
    except BaseException as error:
        import traceback
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'geometry_check.json').write_text(json.dumps(report, indent=2))
        import sys; sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
