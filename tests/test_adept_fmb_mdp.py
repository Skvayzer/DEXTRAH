import torch

from dextrah_lab.tasks.dextrah_kuka_allegro.adept_fmb_mdp import (
    downstream_reward,
    extruded_polygon_surface_points,
    goal_tolerance,
    l_shaped_goal_path,
    rounded_square_polygon,
    star_polygon,
)


def test_downstream_reward_has_no_contact_gate_or_bonus():
    total, reach, goal = downstream_reward(torch.tensor([0.0]), torch.tensor([0.0]))
    assert total.item() == 6.0
    assert reach.item() == 1.0
    assert goal.item() == 5.0


def test_goal_tolerance_anneals_from_five_to_two_centimeters():
    assert goal_tolerance(0) == 0.05
    assert goal_tolerance(50) == 0.02
    assert goal_tolerance(100) == 0.02


def test_goal_path_is_horizontal_then_vertical():
    start = torch.tensor([0.0, 0.0, 1.0])
    preinsert = torch.tensor([1.0, 0.0, 1.0])
    insertion = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(l_shaped_goal_path(0, start, preinsert, insertion), start)
    assert torch.allclose(l_shaped_goal_path(25, start, preinsert, insertion), preinsert)
    assert torch.allclose(l_shaped_goal_path(50, start, preinsert, insertion), insertion)


def test_proxy_peg_pointclouds_have_exactly_64_surface_points():
    for polygon in (star_polygon(), rounded_square_polygon()):
        cloud = extruded_polygon_surface_points(polygon, 0.15, 64)
        assert cloud.shape == (64, 3)
        assert torch.isfinite(cloud).all()

