"""Executable core of ADEPT Algorithm 1, Stage 2 actor BC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .post_training import PostTrainingConfig, mahalanobis_distribution_loss


@dataclass
class RecurrentPolicyState:
    tensors: list[torch.Tensor] | None = None


def _default_state(model, device: torch.device | str) -> RecurrentPolicyState:
    if not model.is_rnn():
        return RecurrentPolicyState()
    return RecurrentPolicyState(
        [state.to(device) for state in model.get_default_rnn_state()]
    )


def _zero_done_state(state: RecurrentPolicyState, done: torch.Tensor) -> None:
    if state.tensors is None:
        return
    indices = done.nonzero(as_tuple=False).flatten()
    for tensor in state.tensors:
        # RL-Games recurrent state is layers x batch x hidden.
        tensor[:, indices] = 0.0


def policy_distribution(
    model,
    observations: torch.Tensor,
    previous_actions: torch.Tensor,
    state: RecurrentPolicyState,
    *,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = {
        "is_train": train,
        "obs": observations,
        "prev_actions": previous_actions,
    }
    if state.tensors is not None:
        batch.update(
            rnn_states=state.tensors,
            seq_length=1,
            rnn_masks=None,
        )
    result = model(batch)
    if state.tensors is not None:
        state.tensors = [tensor.detach() for tensor in result["rnn_states"]]
    return result["mus"], result["sigmas"]


class ActorBCTrainer:
    """On-policy mixed teacher/student BC exactly matching Appendix D."""

    def __init__(
        self,
        env,
        teacher_model,
        student_model,
        *,
        cfg: PostTrainingConfig | None = None,
        device: torch.device | str = "cuda:0",
    ):
        self.env = env
        self.task = env.unwrapped
        self.teacher = teacher_model.eval()
        self.student = student_model.train()
        self.cfg = cfg or PostTrainingConfig()
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.cfg.bc_actor_learning_rate, eps=1e-8
        )
        self.teacher_state = _default_state(self.teacher, device)
        self.student_state = _default_state(self.student, device)
        self.iteration = 0

    def run(self, iterations: int | None = None, callback=None):
        iterations = self.cfg.bc_iterations if iterations is None else iterations
        self.task.set_post_training_phase("actor_bc")
        observations, _ = self.env.reset()
        action_count = self.task.num_actions
        previous_teacher = torch.zeros(
            self.task.num_envs, action_count, device=self.device
        )
        previous_student = torch.zeros_like(previous_teacher)

        for self.iteration in range(self.iteration, iterations):
            with torch.no_grad():
                teacher_mean, teacher_std = policy_distribution(
                    self.teacher,
                    observations["pretraining_policy"],
                    previous_teacher,
                    self.teacher_state,
                    train=False,
                )
            student_mean, student_std = policy_distribution(
                self.student,
                observations["policy"],
                previous_student,
                self.student_state,
                train=True,
            )
            loss = mahalanobis_distribution_loss(
                student_mean,
                student_std,
                teacher_mean,
                teacher_std,
            ).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                teacher_action = torch.distributions.Normal(
                    teacher_mean, teacher_std
                ).sample()
                student_action = torch.distributions.Normal(
                    student_mean, student_std
                ).sample()
                action = (
                    teacher_action
                    if self.cfg.use_teacher_action(self.iteration)
                    else student_action
                ).clamp(-1.0, 1.0)
                observations, _, terminated, truncated, _ = self.env.step(action)
                done = torch.logical_or(terminated, truncated)
                _zero_done_state(self.teacher_state, done)
                _zero_done_state(self.student_state, done)
                previous_teacher = teacher_action.detach()
                previous_student = student_action.detach()
                previous_teacher[done] = 0.0
                previous_student[done] = 0.0

            if callback is not None:
                callback(self.iteration + 1, float(loss.detach()))

    def save(self, path: str | Path):
        """Save an actor checkpoint loadable by the normal RL-Games runner."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": self.student.state_dict(),
                "epoch": 0,
                "frame": 0,
                "adept_bc_iteration": self.iteration + 1,
            },
            temporary,
        )
        temporary.replace(path)


def checkpoint_weights(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract model weights from ordinary or rank-wrapped RL-Games saves."""

    if "model" in payload:
        return payload["model"]
    for rank in (0, "0"):
        if rank in payload and "model" in payload[rank]:
            return payload[rank]["model"]
    raise KeyError("checkpoint contains no RL-Games model state")
