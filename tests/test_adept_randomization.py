from dataclasses import asdict

from dextrah_lab.adept.randomization import ADEPT_DISTILLATION_RANDOMIZATION


def test_appendix_i_randomization_ranges_are_complete_and_serializable():
    cfg = ADEPT_DISTILLATION_RANDOMIZATION
    assert cfg.physics.object_mass_scale == (0.5, 3.0)
    assert cfg.physics.robot_joint_friction_nm == (0.0, 5.0)
    assert cfg.scene.peg_spawn_xy_jitter_m == 0.15
    assert cfg.scene.board_x_jitter_m == (-0.07, 0.12)
    assert cfg.scene.board_y_jitter_m == (-0.30, 0.10)
    assert cfg.observation.joint_position_noise_std_rad == 0.08
    assert cfg.observation.object_orientation_noise_std_rad == 0.1
    assert cfg.visual.peg_diffuse_rgb == (0.90, 1.0)
    assert cfg.visual.board_diffuse_rgb == (0.04, 0.12)
    assert asdict(cfg)["adr_increments"] == 50

