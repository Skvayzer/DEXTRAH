"""Play2Perfect control baseline, with no geometric fabric or PCA action prior.

Inheriting PlayEnv directly is intentional: disabling repulsion weights in the
fabric would still leave its attractor dynamics, motion limits and feedforward.
This task inherits the original action/observation/termination implementations.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.play.play_env import PlayEnv
from isaacsimenvs.tasks.play.utils.object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
from isaaclab.utils import configclass

from .g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg


@configclass
class G1Revo2DirectEnvCfg(G1Revo2AdeptEnvCfg):
    """Keep the G1 physics/MDP configuration; disable every fabric/PCA path."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fabric.enabled = False
        self.fabric.pca_enabled = False
        self.fabric.append_state_to_observations = False
        self.fabric.pca_prior_initial_weight = 0.0
        self.fabric.pca_prior_final_weight = 0.0
        self.fabric.pca_artifact_path = ""
        self.fabric.pca_artifact_sha256 = ""
        self.fabric.velocity_target_factor = 0.0


class G1Revo2DirectEnv(PlayEnv):
    """Unchanged P2P control and rewards, plus terminal lift diagnostics."""

    cfg: G1Revo2DirectEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs) -> None:
        if any((cfg.fabric.enabled, cfg.fabric.pca_enabled,
                cfg.fabric.append_state_to_observations)):
            raise ValueError("The direct-control baseline requires fabric/PCA/state augmentation off")
        super().__init__(cfg, render_mode, **kwargs)
        requested = set(cfg.assets.handle_head_types)
        distributions = sum(d.type in requested for d in OBJECT_SIZE_DISTRIBUTIONS)
        expected = distributions * cfg.assets.num_assets_per_type
        actual = len(getattr(self, "_object_urdf_paths", ()))
        if actual != expected:
            raise RuntimeError(f"Object pool mismatch: expected {expected}, got {actual}")
        self._episode_ever_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        print(
            f"[INFO] DIRECT CONTROL: objects={actual}, actions={cfg.action_space}, "
            f"actor={cfg.observation_space}, critic={cfg.state_space}; "
            "geometric_fabric=false, PCA=false, shape_encoding=existing_keypoints_only",
            flush=True,
        )

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        self._episode_ever_lifted |= self._lifted_object
        # float() snapshots the booleans before DirectRLEnv auto-resets them.
        self.extras["episode_final"]["ever_lifted"] = self._episode_ever_lifted.float()
        self.extras["episode_final"]["any_success"] = (self._successes > 0).float()
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if hasattr(self, "_episode_ever_lifted"):
            self._episode_ever_lifted[env_ids] = False


__all__ = ["G1Revo2DirectEnv", "G1Revo2DirectEnvCfg"]
