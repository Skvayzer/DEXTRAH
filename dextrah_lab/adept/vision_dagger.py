"""ADEPT Stage-3 two-camera DAgger training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .actor_bc import RecurrentPolicyState, _default_state, _zero_done_state, policy_distribution
from .post_training import distillation_loss
from .student import AdeptStudentPolicy
from dextrah_lab.tasks.dextrah_kuka_allegro.adept_mdp import box_pose_keypoints


@dataclass(frozen=True)
class VisionDAggerConfig:
    # ADEPT refers to the public DextrAH-RGB unfreeze schedule without listing
    # its values. 5k is the checked-in DextrAH schedule; total iterations and
    # learning rate are likewise exposed because ADEPT v1 does not report them.
    iterations: int = 100_000
    learning_rate: float = 1e-4
    frozen_backbone_iterations: int = 5_000
    auxiliary_weight: float = 20.0
    keypoint_half_extents_m: tuple[float, float, float] = (0.025, 0.025, 0.075)


def set_backbone_trainable(student: AdeptStudentPolicy, trainable: bool) -> None:
    student.vision.backbone.train(trainable)
    for parameter in student.vision.backbone.parameters():
        parameter.requires_grad_(trainable)


class VisionDAggerTrainer:
    """Collect student rollouts and supervise with the frozen state teacher."""

    def __init__(
        self,
        env,
        teacher_model,
        student: AdeptStudentPolicy,
        *,
        cfg: VisionDAggerConfig | None = None,
        device: torch.device | str = "cuda:0",
        stage: str = "downstream",
    ):
        self.env = env
        self.task = env.unwrapped
        self.teacher = teacher_model.eval()
        self.student = student.to(device).train()
        self.cfg = cfg or VisionDAggerConfig()
        self.device = device
        if stage not in {"vision_pretrain", "downstream"}:
            raise ValueError("stage must be vision_pretrain or downstream")
        self.stage = stage
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.cfg.learning_rate, eps=1e-8
        )
        self.teacher_state = _default_state(self.teacher, device)
        self.student_state = None
        self.iteration = 0

    def run(self, iterations: int | None = None, callback=None):
        iterations = self.cfg.iterations if iterations is None else iterations
        self.task.set_student_distillation_phase(self.stage)
        observations, _ = self.env.reset()
        previous_teacher = torch.zeros(
            self.task.num_envs, self.task.num_actions, device=self.device
        )

        for self.iteration in range(self.iteration, iterations):
            set_backbone_trainable(
                self.student,
                self.iteration >= self.cfg.frozen_backbone_iterations,
            )
            with torch.no_grad():
                teacher_mean, teacher_std = policy_distribution(
                    self.teacher,
                    observations[
                        "pretraining_policy"
                        if self.stage == "vision_pretrain"
                        else "expert_policy"
                    ],
                    previous_teacher,
                    self.teacher_state,
                    train=False,
                )
            output = self.student(
                observations["img_left"],
                observations["img_right"],
                observations["policy"],
                recurrent_state=self.student_state,
            )
            self.student_state = tuple(
                tensor.detach() for tensor in output["recurrent_state"]
            )
            student_std = output["log_std"].exp()
            target_keypoints = box_pose_keypoints(
                self.task.object_pos,
                self.task.object_rot,
                self.cfg.keypoint_half_extents_m,
            )
            loss, terms = distillation_loss(
                output["mean"],
                student_std,
                teacher_mean,
                teacher_std,
                output["keypoints"],
                target_keypoints,
                object_height=self.task.object_pos[:, 2],
                auxiliary_weight=self.cfg.auxiliary_weight,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                action = torch.distributions.Normal(
                    output["mean"], student_std
                ).sample().clamp(-1.0, 1.0)
                observations, _, terminated, truncated, _ = self.env.step(action)
                done = torch.logical_or(terminated, truncated)
                _zero_done_state(self.teacher_state, done)
                if self.student_state is not None:
                    indices = done.nonzero(as_tuple=False).flatten()
                    for tensor in self.student_state:
                        tensor[:, indices] = 0.0
                previous_teacher = teacher_mean.detach()
                previous_teacher[done] = 0.0

            if callback is not None:
                callback(
                    self.iteration + 1,
                    {name: float(value.detach()) for name, value in terms.items()},
                )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "student": self.student.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration + 1,
            },
            temporary,
        )
        temporary.replace(path)


def load_vision_backbone(student: AdeptStudentPolicy, checkpoint: dict) -> None:
    """Initialize only the shared visual encoder from Stage-1 student pretraining."""

    state = checkpoint.get("student", checkpoint)
    prefix = "vision."
    vision_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not vision_state:
        raise KeyError("checkpoint contains no vision encoder weights")
    student.vision.load_state_dict(vision_state, strict=True)
