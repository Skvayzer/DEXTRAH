"""Original G1 Play2Perfect control/reward with cached BPS-128 observations."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass

from dextrah_lab.object_shape.bank import SHAPE_DIM, build_bank, map_features
from .g1_revo2_direct_env import G1Revo2DirectEnv, G1Revo2DirectEnvCfg


@configclass
class G1Revo2BpsEnvCfg(G1Revo2DirectEnvCfg):
    bps_cache_dir: str = "outputs/bps128_cache"
    bps_artifact_dir: str = "outputs/bps128_last_bank"


class G1Revo2BpsEnv(G1Revo2DirectEnv):
    """Only observations/diagnostic names are changed, not robot dynamics."""

    cfg: G1Revo2BpsEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._bps_base_actor_dim = cfg.observation_space
        self._bps_base_critic_dim = cfg.state_space
        if (self._bps_base_actor_dim, self._bps_base_critic_dim) != (92, 114):
            raise ValueError("Warm-start task requires the proven 92/114 P2P observation layout")
        bank, manifest = build_bank(self._object_urdf_paths, cfg.bps_cache_dir)
        indices = self._object_asset_index_per_env.detach().cpu().numpy()
        # Independent check against composed USD references, not just the
        # round-robin index tensor that the descriptor lookup will use.
        from isaaclab.sim import get_current_stage
        stage = get_current_stage()
        for env_id, asset_id in enumerate(indices):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Object")
            references = prim.GetMetadata("references")
            paths = [ref.assetPath for ref in references.GetAddedOrExplicitItems()] if references else []
            expected_stem = Path(self._object_urdf_paths[asset_id]).stem
            if len(paths) != 1 or Path(paths[0]).stem != expected_stem:
                raise RuntimeError(f"BPS/USD asset identity mismatch: env={env_id}, "
                                   f"expected={expected_stem}, references={paths}")
        manifest["usd_reference_identity_checks"] = self.num_envs
        self._bps_bank = torch.as_tensor(bank, device=self.device)
        self._bps_features = torch.as_tensor(map_features(bank, indices), device=self.device)
        self._bps_manifest = manifest
        # PlayEnv allocates its own delay queues at original dimensions. BPS is
        # immutable per object; append AFTER the original delayed observation.
        cfg.observation_space = self._bps_base_actor_dim + SHAPE_DIM
        cfg.state_space = self._bps_base_critic_dim + SHAPE_DIM
        self.single_observation_space["policy"] = gym.spaces.Box(-np.inf, np.inf,
            shape=(cfg.observation_space,), dtype=np.float32)
        self.single_observation_space["critic"] = gym.spaces.Box(-np.inf, np.inf,
            shape=(cfg.state_space,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)
        artifact = Path(cfg.bps_artifact_dir)
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        np.savez_compressed(artifact / "bank.npz", features=bank, asset_index_per_env=indices)
        print(f"[BPS] READY objects={len(bank)} environments={self.num_envs} "
              f"actor={cfg.observation_space} critic={cfg.state_space} actions=13; "
              "128 distances + 4 metric normalization values; no fabrics/PCA/tactile", flush=True)

    def _get_observations(self):
        observations = super()._get_observations()
        return {key: torch.cat([value, self._bps_features], dim=-1)
                for key, value in observations.items()}

    def _get_rewards(self):
        reward = super()._get_rewards()
        if "final_observation" in self.extras:
            final = self.extras["final_observation"]
            final["critic"] = torch.cat([final["critic"], self._bps_features], dim=-1)
        final = self.extras["episode_final"]
        # Unambiguous aliases, aggregated over completed SAPG leader episodes
        # by the existing P2P observer. Values, reward and counters unchanged.
        final["goals_reached_count"] = final["successes"]
        final["any_goal_success_rate"] = final["any_success"]
        final["ever_lifted_rate"] = final["ever_lifted"]
        final["all_goals_completed_rate"] = final["all_goals_hit"]
        return reward
