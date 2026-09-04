"""RL-Games phase controller for ADEPT critic warm-up and conservative PPO."""

from __future__ import annotations

from typing import Any

from .post_training import PostTrainingConfig


def _find_task_env(root: Any):
    """Follow the small set of wrapper links used by Isaac Lab/RL-Games."""

    current = root
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "set_post_training_phase"):
            return current
        next_object = None
        for attribute in ("unwrapped", "env", "_env"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_object = candidate
                break
        current = next_object
    raise RuntimeError("could not locate the ADEPT task through RL-Games wrappers")


class AdeptPostTrainingObserver:
    """Hold actor updates for 20 epochs, then enable conservative PPO.

    The asymmetric central-value optimizer remains active while the policy
    optimizer is held at zero learning rate.  This is the least invasive way
    to preserve RL-Games' return/GAE and recurrent rollout implementation.
    """

    def __init__(
        self,
        cfg: PostTrainingConfig | None = None,
        actor_checkpoint: str | None = None,
    ):
        self.cfg = cfg or PostTrainingConfig()
        self.actor_checkpoint = actor_checkpoint
        self.algo = None
        self.task_env = None
        self._phase = None

    def before_init(self, *_args, **_kwargs):
        pass

    def after_init(self, algo):
        self.algo = algo
        if self.actor_checkpoint is not None:
            from rl_games.algos_torch import torch_ext

            payload = torch_ext.load_checkpoint(self.actor_checkpoint)
            weights = payload.get(
                getattr(algo, "global_rank", 0), payload.get(0, payload)
            )
            # Actor BC deliberately has no critic/optimizer state. Loading
            # policy weights only leaves the downstream critic freshly
            # initialized, as required by Algorithm 1 Stage 3.
            algo.set_weights(weights)
        self._reset_actor_log_std()
        self.task_env = _find_task_env(algo.vec_env)
        self._apply_phase("critic_warmup")

    def _reset_actor_log_std(self):
        """Pin RL-Games' input-independent Gaussian log-std to Appendix D."""

        model = getattr(self.algo, "model", None)
        if model is None:
            return
        matches = []
        for name, parameter in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"sigma", "logstd", "log_std"}:
                parameter.data.fill_(self.cfg.fixed_actor_log_std)
                matches.append(name)
        if not matches:
            raise RuntimeError("could not locate RL-Games actor log-std parameter")

    def _apply_phase(self, phase: str):
        if phase == self._phase:
            return
        self.task_env.set_post_training_phase(phase)
        learning_rate = 0.0 if phase == "critic_warmup" else self.cfg.actor_learning_rate
        self.algo.learning_rate = learning_rate
        self.algo.last_lr = learning_rate
        for parameter_group in self.algo.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        self._phase = phase

    def process_infos(self, _infos, _done_indices):
        pass

    def after_steps(self):
        # Schedulers may rewrite optimizer groups each epoch, so enforce the
        # phase immediately after rollout collection and before optimization.
        epoch = int(getattr(self.algo, "epoch_num", 0))
        phase = "critic_warmup" if epoch < self.cfg.critic_warmup_epochs else "ppo"
        self._apply_phase(phase)
        if phase == "critic_warmup":
            for parameter_group in self.algo.optimizer.param_groups:
                parameter_group["lr"] = 0.0

    def after_clear_stats(self):
        pass

    def after_print_stats(self, _frame, epoch_num, _total_time):
        phase = (
            "critic_warmup"
            if int(epoch_num) < self.cfg.critic_warmup_epochs
            else "ppo"
        )
        self._apply_phase(phase)
