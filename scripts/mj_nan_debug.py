#!/usr/bin/env python3
"""Find where the mjlab port first produces non-finite state (diagnostic only).

Passive robot (zero latent residual, open fingers). After every physics step,
checks qpos/qvel/qacc per world; on the first non-finite world prints the
step, which DOFs went bad (robot root, robot joints, object), its object,
contact count, nefc, overflow bits and the last finite state. Variants:
--no-velocity-limits, --mujoco '{"cone": "pyramidal", ...}'.
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=24)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--no-velocity-limits', action='store_true')
    p.add_argument('--mujoco', type=json.loads, default=None)
    p.add_argument('--label', default='default')
    args = p.parse_args()
    import torch
    import warp as wp
    from dextrah_lab.mj.factory import make_mj_env
    env = make_mj_env(args.num_envs, 'cuda:0', velocity_limits=not args.no_velocity_limits,
                      mujoco_cfg=args.mujoco, nconmax=128, njmax=1000)
    n = env.num_envs
    d = env.sim.wp_data
    m = env.sim.mj_model
    jq = lambda name: int(m.jnt_qposadr[m.joint(name).id])
    free_q = list(range(jq('robot/floating_base_joint'), jq('robot/floating_base_joint')+7))
    obj_q = list(range(jq('object/object_free'), jq('object/object_free')+7))
    robot_q = [i for i in range(free_q[-1]+1, obj_q[0])]
    env.reset()
    action = torch.zeros(n, 70, device='cuda:0')
    action[:, 64:] = -1.
    overflow = torch.zeros(n, dtype=torch.long, device='cuda:0')
    prev = None
    result = dict(label=args.label, first_nan=None)
    steps = round(args.seconds/env.step_dt)
    original_after = env._after_physics_step

    def checked():
        nonlocal prev
        qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
        overflow.__ior__(wp.to_torch(d.overflow).long())
        bad = ~(torch.isfinite(qpos).all(-1) & torch.isfinite(qvel).all(-1))
        if bad.any() and result['first_nan'] is None:
            w = int(bad.nonzero()[0])
            def which(q):
                q = q[w].cpu()
                v = q.shape[0] < m.nq   # qvel: free joints have 6 dofs
                f, o = (slice(0, 6), slice(m.nv-6, m.nv)) if v else (slice(free_q[0], free_q[-1]+1), slice(obj_q[0], obj_q[-1]+1))
                j = slice(f.stop, o.start)
                bad_joints = [m.joint(int(m.dof_jntid[k])).name for k in range(j.start, j.stop) if not torch.isfinite(q[k])] if v else None
                return dict(robot_root=bool((~torch.isfinite(q[f])).any()),
                            robot_joints=int((~torch.isfinite(q[j])).sum()), bad_joints=bad_joints,
                            object=bool((~torch.isfinite(q[o])).any()))
            nefc = wp.to_torch(d.nefc)
            world = wp.to_torch(d.contact.worldid)
            nacon = int(wp.to_torch(d.nacon))
            ncon_w = int((world[:nacon] == w).sum())
            result['first_nan'] = dict(
                sim_step=env._sim_step_counter, world=w, bad_worlds=int(bad.sum()),
                asset=int(env._object_asset_index_per_env[w]), object=env._object_urdf_paths[int(env._object_asset_index_per_env[w])].split('/')[-1][:80],
                qpos=which(qpos), qvel=which(qvel), contacts_in_world=ncon_w, nefc=int(nefc[w]),
                overflow_bits=int(overflow[w]), total_contacts=nacon,
                last_finite=None if prev is None else dict(
                    max_abs_qvel=float(prev[1][w].abs().max()), argmax_qvel=int(prev[1][w].abs().argmax()),
                    root_pos=prev[0][w, :3].tolist(), object_pos=prev[0][w].cpu()[obj_q[:3]].tolist()))
            dof = result['first_nan']['last_finite']['argmax_qvel'] if prev is not None else None
            if dof is not None:
                result['first_nan']['last_finite']['argmax_dof_joint'] = m.joint(int(m.dof_jntid[dof])).name
            print('NAN_FOUND '+json.dumps(result['first_nan']), flush=True)
        prev = (qpos.clone(), qvel.clone())
        original_after()

    env._after_physics_step = checked
    with torch.inference_mode():
        for step in range(steps):
            try:
                env.step(action)
            except RuntimeError as error:
                import traceback
                traceback.print_exc()
                result['stopped'] = f'{error} at policy step {step}'
                print('NAN_DEBUG '+json.dumps(result), flush=True)
                os._exit(1)
    result['overflow_worlds'] = int((overflow > 0).sum())
    result['overflow_bits'] = sorted(set(overflow[overflow > 0].tolist()))
    print('NAN_DEBUG '+json.dumps(result), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
