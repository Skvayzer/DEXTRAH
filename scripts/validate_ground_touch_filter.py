#!/usr/bin/env python3
"""Check that the real source ground contributes filtered contact-point forces."""
import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    report = dict(passed=False)
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from dextrah_lab.wholebody.source_scene import ground_contact_partners
        from dextrah_lab.g1_adept.tactile_forces import read_pad_contact_forces
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device))
        ground = sim_utils.GroundPlaneCfg()
        ground.func('/World/ground', ground)
        partners = ground_contact_partners(sim_utils.get_current_stage())
        cube = RigidObject(RigidObjectCfg(prim_path='/World/Cube',
            spawn=sim_utils.CuboidCfg(size=(.1, .1, .1), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., .15))))
        sensor = ContactSensor(ContactSensorCfg(prim_path='/World/Cube',
            update_period=0., track_pose=True, track_contact_points=True,
            filter_prim_paths_expr=partners, max_contact_data_count_per_prim=64))
        sim.reset()
        eye, origin = torch.eye(3, device=args.device), torch.zeros(3, device=args.device)
        bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]], device=args.device)
        peak_error, supported_samples = 0., 0
        for step in range(180):
            cube.write_data_to_sim()
            sim.step(render=False)
            cube.update(sim.cfg.dt)
            sensor.update(sim.cfg.dt)
            data = sensor.data
            row = read_pad_contact_forces(sensor.contact_physx_view, sim.cfg.dt,
                data.pos_w[:, 0], data.quat_w[:, 0], origin, eye, bounds)
            reconstructed = row.reconstructed_normal_pairs_w.sum(1)
            net = data.net_forces_w[:, 0]
            torch.testing.assert_close(reconstructed, net, atol=.03, rtol=.02)
            peak_error = max(peak_error, float((reconstructed-net).abs().max()))
            supported_samples += int(net[:, 2].min() > 5.)
        if supported_samples < 100:
            raise RuntimeError('Ground-contact test did not produce sustained support')
        report.update(passed=True, ground_filters=partners,
            final_force_n=net.tolist(), supported_samples=supported_samples,
            maximum_reconstruction_error_n=peak_error)
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        (args.output/'validation.json').write_text(json.dumps(report, indent=2))
        print('GROUND_TOUCH_VALIDATION '+json.dumps(report), flush=True)
        # The small diagnostic has no state to preserve; avoid Kit's shutdown
        # rewriting error exit codes or waiting for an interactive timeline.
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
