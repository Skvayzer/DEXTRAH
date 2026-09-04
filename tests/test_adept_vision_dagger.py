import torch

from dextrah_lab.adept.student import make_kuka_vision_student
from dextrah_lab.adept.vision_dagger import load_vision_backbone, set_backbone_trainable


def test_backbone_freeze_does_not_freeze_cross_attention():
    student = make_kuka_vision_student()
    set_backbone_trainable(student, False)
    assert not any(parameter.requires_grad for parameter in student.vision.backbone.parameters())
    assert all(parameter.requires_grad for parameter in student.vision.left_to_center.parameters())


def test_stage1_checkpoint_loads_only_vision_encoder():
    source = make_kuka_vision_student()
    target = make_kuka_vision_student()
    with torch.no_grad():
        next(source.vision.parameters()).fill_(0.25)
        target.action_mean.weight.fill_(7.0)
    load_vision_backbone(target, {"student": source.state_dict()})
    torch.testing.assert_close(
        next(target.vision.parameters()), next(source.vision.parameters())
    )
    assert torch.all(target.action_mean.weight == 7.0)
