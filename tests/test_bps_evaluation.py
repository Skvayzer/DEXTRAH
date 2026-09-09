import numpy as np
import pytest
from dextrah_lab.object_shape.evaluation import ReposeStats, select_family_envs, bps_display_geometry


def final(counts, lifted=None, cap=None):
    n=len(counts)
    return dict(successes=np.array(counts), ever_lifted_rate=np.ones(n) if lifted is None else lifted,
                all_goals_hit=np.zeros(n) if cap is None else cap,
                done_fall=np.zeros(n), done_timeout=np.ones(n), done_hand_far=np.zeros(n))


def test_attempts_episodes_and_censoring_are_separate():
    s=ReposeStats(2,.5)
    s.update(final([1,0]),[False,True],[1,0])
    s.update(final([2,1]),[False,False],[1,1])
    s.update(final([2,2]),[True,True],[0,1])
    r=s.report()
    assert (r['hits'],r['failed_attempts'],r['episodes']) == (4,2,3)
    assert r['goal_success_fraction_resolved'] == pytest.approx(4/6)
    assert r['episode_any_goal_success_rate'] == pytest.approx(2/3)
    assert r['goals_per_simulated_minute'] == 80
    assert r['goals_per_completed_episode'] == pytest.approx(4/3)
    s.update(final([1,0]),[False,False],[1,0])
    assert s.report()['ongoing_episode_goals'] == 1
    assert s.report()['episodes'] == 3


def test_terminal_hit_is_not_failure_and_reset_is_clean():
    s=ReposeStats(1,1.)
    s.update(final([1],cap=[True]),[True],[1])
    s.update(final([0],lifted=[False]),[True],[0])
    r=s.report()
    assert r['hits']==1 and r['failed_attempts']==1 and r['all_goals']==1
    assert r['episode_ever_lifted_rate']==.5
    assert s.report([0])==r


def test_lost_counter_and_bad_state_fail():
    s=ReposeStats(1,1.)
    with pytest.raises(ValueError,match='double-counted'):
        s.update(final([2]),[False],[0])
    with pytest.raises(ValueError,match='Nonfinite'):
        s.update(final([0]),[False],[float('nan')])
    assert ReposeStats(1,1.).report()['goal_success_fraction_resolved'] is None


def test_preselection_uses_family_not_outcomes():
    paths=['0_brush_handle_x','1_hammer_handle_x','2_brush_handle_x']
    assert select_family_envs(paths,[2,1,0],['hammer','brush']) == {'hammer':1,'brush':0}
    with pytest.raises(ValueError,match='Missing'):
        select_family_envs(paths,[0,1],['eraser'])


def test_display_distances_equal_actual_policy_descriptor():
    import trimesh
    from dextrah_lab.object_shape.bank import mesh_features
    mesh=trimesh.creation.box(extents=[.04,.12,.02])
    features=mesh_features(mesh)
    data=bps_display_geometry(mesh,features)
    np.testing.assert_allclose(np.linalg.norm(data['basis']-data['nearest'],axis=1),features[:128])
    with pytest.raises(AssertionError):
        bps_display_geometry(mesh,features+.1)
