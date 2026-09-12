#!/usr/bin/env python3
"""Independent CUDA/Warp/PhysX probe: one falling cube, no robot or policy."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time
import traceback
import sys


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation, including CPU-physics mode')
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(args).app
    # Open a dedicated file AFTER Kit startup; retain its descriptor even if
    # Kit replaces process stdout/stderr or installs its own crash handler.
    with (args.output/'stacks.log').open('w') as stacks:
        faulthandler.enable(file=stacks)
        faulthandler.dump_traceback_later(30,repeat=True,file=stacks)
        try:
            import torch
            import warp as wp
            import isaaclab.sim as sim_utils
            from isaaclab.assets import RigidObject, RigidObjectCfg
            def stage(name,**extra):
                event=dict(stage=name,time=time.time(),**extra)
                with (args.output/'progress.jsonl').open('a') as file:
                    file.write(json.dumps(event)+'\n')
                print('RUNTIME_PROBE '+json.dumps(event),flush=True)
            stage('imports',torch=torch.__version__,warp=wp.__version__,warp_file=wp.__file__)
            if args.device.startswith('cuda'):
                x=torch.ones(64,device=args.device)
                torch.cuda.synchronize()
                stage('torch_ok',sum=float(x.sum()))
                w=wp.ones(64,device=args.device)
                wp.synchronize_device(args.device)
                stage('warp_ok',sum=float(w.numpy().sum()))
            sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
            ground=sim_utils.GroundPlaneCfg()
            ground.func('/World/ground',ground)
            cube=RigidObject(RigidObjectCfg(prim_path='/World/Cube',
                spawn=sim_utils.CuboidCfg(size=(.1,.1,.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                    collision_props=sim_utils.CollisionPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,1.))))
            stage('scene_built')
            sim.reset()
            stage('reset_complete')
            for i in range(200):
                cube.write_data_to_sim()
                sim.step(render=False)
                cube.update(.005)
                if i%50==0:
                    stage('step',step=i,z=float(cube.data.root_pos_w[0,2]))
            z=float(cube.data.root_pos_w[0,2])
            if not .035<z<.075:
                raise ValueError(f'Cube did not settle on ground: z={z}')
            stage('passed',z=z)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        finally:
            if 'sim' in locals():
                sim.clear_all_callbacks()
                sim.clear_instance()
            app.close()
            faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':
    main()
