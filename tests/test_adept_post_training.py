import torch

from dextrah_lab.adept.post_training import (
    PostTrainingConfig,
    PostTrainingPhase,
    STAGE1_ACTOR_FIELDS,
    STAGE2_ACTOR_FIELDS,
    distillation_loss,
    mahalanobis_distribution_loss,
    soft_height_mask,
    transfer_input_columns,
)


def test_post_training_phase_boundaries_and_mixed_rollout():
    cfg = PostTrainingConfig()
    assert cfg.phase(39_999, 0) == PostTrainingPhase.ACTOR_BC
    assert cfg.phase(40_000, 19) == PostTrainingPhase.CRITIC_WARMUP
    assert cfg.phase(40_000, 20) == PostTrainingPhase.PPO
    assert cfg.phase(40_000, 200_000) == PostTrainingPhase.COMPLETE
    assert [cfg.use_teacher_action(i) for i in range(4)] == [True, False, True, False]


def test_mahalanobis_loss_matches_equation_29():
    zeros = torch.zeros(2, 3)
    teacher_std = torch.full((2, 3), 2.0)
    loss = mahalanobis_distribution_loss(
        torch.ones(2, 3), torch.full((2, 3), 3.0), zeros, teacher_std
    )
    assert torch.allclose(loss, torch.full((2,), 3**0.5))


def test_distillation_weights_auxiliary_by_twenty():
    action = torch.zeros(1, 23)
    keypoints = torch.zeros(1, 8, 3)
    predicted = keypoints.clone()
    predicted[:, 0, 0] = 1.0
    total, terms = distillation_loss(
        action, torch.ones_like(action), action, torch.ones_like(action), predicted, keypoints
    )
    assert total.item() == 20.0
    assert terms["auxiliary"].item() == 1.0


def test_soft_height_mask_downweights_low_states_with_floor():
    weights = soft_height_mask(torch.tensor([-10.0, 0.08, 10.0]))
    assert torch.allclose(weights, torch.tensor([0.1, 0.55, 1.0]), atol=1e-5)


def test_observation_remap_copies_shared_fields_not_new_fields():
    source = torch.arange(2 * 391, dtype=torch.float32).reshape(2, 391)
    target = torch.full((2, 392), -1.0)
    remapped = transfer_input_columns(
        source, target, STAGE1_ACTOR_FIELDS, STAGE2_ACTOR_FIELDS
    )
    assert torch.equal(
        remapped[:, STAGE2_ACTOR_FIELDS["pointcloud"]],
        source[:, STAGE1_ACTOR_FIELDS["pointcloud"]],
    )
    assert torch.all(remapped[:, STAGE2_ACTOR_FIELDS["receptacle_pose"]] == -1.0)

