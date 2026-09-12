from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from dextrah_lab.wholebody.failure_recording import FailureRecorder


def fake_env():
    n = 2
    data = NS(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1),
        joint_pos=torch.zeros(n, 3), joint_vel=torch.zeros(n, 3),
        body_pos_w=torch.zeros(n, 2, 3), body_quat_w=torch.tensor([[[1., 0, 0, 0]]]).repeat(n, 2, 1))
    asset = NS(data=data)
    def stop():
        raise RuntimeError('test numerical stop')
    return NS(num_envs=n, device='cpu', cfg=NS(sim=NS(dt=.01)),
        scene=NS(env_origins=torch.zeros(n, 3)), robot=asset, object=asset, table=asset, goal_viz=asset,
        touch_raw=torch.zeros(n, 5, 3), _cur_targets=torch.zeros(n, 3),
        _successes=torch.zeros(n), episode_length_buf=torch.zeros(n), _sim_step_counter=1,
        _apply_action=lambda: None, _get_dones=stop, _reset_idx=lambda ids: None)


def test_ring_wrap_and_no_rng_or_physics_mutation(tmp_path):
    env = fake_env()
    recorder = FailureRecorder(env, tmp_path/'clip', seconds=.02)
    rng = torch.get_rng_state().clone()
    for step in range(7):
        env.robot.data.joint_pos[:] = step
        recorder.record(step)
        assert torch.all(env.robot.data.joint_pos == step)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    order = np.arange(recorder.count-recorder.capacity, recorder.count) % recorder.capacity
    np.testing.assert_array_equal(recorder.steps[order], [3, 4, 5, 6])
    np.testing.assert_array_equal(recorder.data[order, 0, 0], [3, 4, 5, 6])


def test_hooks_keep_reset_semantics_and_capture_rejected_frame(tmp_path):
    env = fake_env()
    recorder = FailureRecorder(env, tmp_path/'clip', seconds=.02)
    env._reset_idx(torch.tensor([1]))
    assert recorder.reset_counts.tolist() == [0, 1]
    env._apply_action()
    assert recorder.steps[0] == 0
    env._sim_step_counter = 2
    with pytest.raises(RuntimeError, match='test numerical stop'):
        env._get_dones()
    assert recorder.count == 2 and recorder.steps[1] == 2
