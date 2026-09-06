"""G1/Revo2 reposing with an ADEPT-style controller and SAPG-compatible MDP."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from isaaclab.envs import DirectRLEnv

from isaacsimenvs.tasks.play.play_env import PlayEnv
from isaacsimenvs.tasks.play.utils.action_utils import apply_wrench_dr
from isaacsimenvs.tasks.play.utils.object_size_distributions import (
    OBJECT_SIZE_DISTRIBUTIONS,
)
from isaacsimenvs.tasks.play.utils.obs_utils import build_observations, compute_obs_dim
from isaacsimenvs.tasks.play.utils.reset_utils import allocate_state_buffers
from isaacsimenvs.tasks.play.utils.scene_utils import apply_physx_material_properties

from dextrah_lab.g1_adept import (
    CollisionBatch,
    IsaacCollisionAdapterConfig,
    IsaacG1CollisionAdapter,
    ReducedAdeptFabric,
)
from dextrah_lab.retargeting import (
    FrozenPCAHandActionMap,
    REVO2_RIGHT_ACTUATED_JOINTS,
)

from .g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg


FABRIC_OBSERVATION_DIM = 39


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class G1Revo2AdeptEnv(PlayEnv):
    """Play2Perfect reposing dynamics controlled through the reduced fabric."""

    cfg: G1Revo2AdeptEnvCfg

    def __init__(
        self,
        cfg: G1Revo2AdeptEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._base_policy_observation_dim = compute_obs_dim(cfg.obs.obs_list, cfg)
        self._base_critic_observation_dim = compute_obs_dim(cfg.obs.state_list, cfg)
        extra = FABRIC_OBSERVATION_DIM if cfg.fabric.append_state_to_observations else 0
        cfg.observation_space = self._base_policy_observation_dim + extra
        cfg.state_space = self._base_critic_observation_dim + extra

        # Reproduce PlayEnv.__init__, reserving augmented spaces before
        # DirectRLEnv constructs its Gym and RL-Games buffers.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        self._validate_object_pool()
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        # Play delays only its base observation. Keep that queue at base size
        # and delay the 39 fabric features in a stream with the same bounds.
        self._obs_queue = torch.zeros(
            self.num_envs,
            max(1, self.cfg.domain_randomization.obs_delay_max),
            self._base_policy_observation_dim,
            device=self.device,
        )
        self._fabric_obs_queue = torch.zeros(
            self.num_envs,
            max(1, self.cfg.domain_randomization.obs_delay_max),
            FABRIC_OBSERVATION_DIM,
            device=self.device,
        )
        self._setup_adept_controller()

    def _validate_object_pool(self) -> None:
        """Fail closed if training does not contain the requested full pool."""

        requested_types = set(self.cfg.assets.handle_head_types)
        distribution_count = sum(
            distribution.type in requested_types
            for distribution in OBJECT_SIZE_DISTRIBUTIONS
        )
        expected = distribution_count * self.cfg.assets.num_assets_per_type
        actual = len(getattr(self, "_object_urdf_paths", ()))
        if actual != expected:
            raise RuntimeError(
                "SimToolReal object-pool mismatch: "
                f"expected {expected} objects from {distribution_count} matching "
                f"distributions x {self.cfg.assets.num_assets_per_type}, got {actual}."
            )
        print(
            "[INFO] Verified SimToolReal pool: "
            f"{actual} objects, {distribution_count} distributions, "
            f"families={sorted(requested_types)}"
        )

    def _setup_adept_controller(self) -> None:
        fabric_cfg = self.cfg.fabric
        if not fabric_cfg.enabled:
            raise ValueError("this task requires fabric.enabled=true")
        if fabric_cfg.steps_per_policy_step <= 0:
            raise ValueError("fabric.steps_per_policy_step must be positive")
        if fabric_cfg.max_joint_delta <= 0.0:
            raise ValueError("fabric.max_joint_delta must be positive")
        if fabric_cfg.pca_prior_anneal_frames < 0:
            raise ValueError("fabric.pca_prior_anneal_frames cannot be negative")

        lower = torch.cat((self._arm_lower[0], self._hand_lower[0]))
        upper = torch.cat((self._arm_upper[0], self._hand_upper[0]))
        self.adept_fabric = ReducedAdeptFabric(
            lower,
            upper,
            config=fabric_cfg.reduced_config(),
        )
        measured_position, measured_velocity = self._measured_canonical_state()
        self.adept_fabric.reset(measured_position, velocity=measured_velocity)
        self.collision_adapter = IsaacG1CollisionAdapter(
            self,
            IsaacCollisionAdapterConfig(
                table_surface_offset=fabric_cfg.table_surface_offset,
                include_self_collision=fabric_cfg.include_self_collision,
            ),
        )
        self._pca_action_map = self._load_pca_action_map()
        self._cur_velocity_targets = torch.zeros_like(self._cur_targets)
        self._episode_min_fabric_clearance = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._last_tracking_error = torch.zeros(self.num_envs, device=self.device)
        self._last_collision: CollisionBatch | None = None
        self._last_pca_weight = 0.0

    def _load_pca_action_map(self) -> FrozenPCAHandActionMap | None:
        cfg = self.cfg.fabric
        if not cfg.pca_enabled:
            return None
        path = Path(cfg.pca_artifact_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "fabric.pca_enabled is true but its artifact is missing: "
                f"{path}. Run the DexYCB retargeting/PCA pipeline or override "
                "fabric.pca_artifact_path."
            )
        actual_sha = _sha256(path)
        if cfg.pca_artifact_sha256 and actual_sha != cfg.pca_artifact_sha256:
            raise RuntimeError(
                f"PCA artifact checksum mismatch: expected {cfg.pca_artifact_sha256}, "
                f"got {actual_sha} for {path}"
            )
        return FrozenPCAHandActionMap.from_file(
            path,
            expected_joint_names=REVO2_RIGHT_ACTUATED_JOINTS,
            device=self.device,
            dtype=torch.float32,
        )

    def _measured_canonical_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_ids = self._canonical_joint_ids_lab
        return (
            self.robot.data.joint_pos.index_select(1, joint_ids),
            self.robot.data.joint_vel.index_select(1, joint_ids),
        )

    def _delayed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        dr = self.cfg.domain_randomization
        if not dr.use_action_delay or dr.action_delay_max <= 0:
            return actions
        episode_start = (self.episode_length_buf == 0) & (self._successes == 0)
        if episode_start.any():
            self._action_queue[episode_start] = actions[episode_start].unsqueeze(1)
        self._action_queue = torch.roll(self._action_queue, shifts=1, dims=1)
        self._action_queue[:, 0, :] = actions
        delay_index = torch.randint(
            0, self._action_queue.shape[1], (self.num_envs,), device=self.device
        )
        return self._action_queue[
            torch.arange(self.num_envs, device=self.device), delay_index
        ]

    def _pca_prior_weight(self) -> float:
        cfg = self.cfg.fabric
        if self._pca_action_map is None:
            return 0.0
        frames = int(self.common_step_counter) * self.num_envs
        if cfg.pca_prior_anneal_frames == 0:
            fraction = 1.0
        else:
            fraction = min(1.0, frames / float(cfg.pca_prior_anneal_frames))
        return (
            cfg.pca_prior_initial_weight
            + fraction * (cfg.pca_prior_final_weight - cfg.pca_prior_initial_weight)
        )

    def _apply_pca_prior(self, target: torch.Tensor) -> torch.Tensor:
        weight = self._pca_prior_weight()
        self._last_pca_weight = weight
        if self._pca_action_map is None or weight == 0.0:
            return target
        hand = target[:, 7:]
        coordinates = self._pca_action_map(hand)
        projected = self._pca_action_map.reconstruct_clipped(coordinates)
        target = target.clone()
        target[:, 7:] = torch.lerp(hand, projected, weight)
        return target

    def _sync_tracking_outliers(
        self, measured_position: torch.Tensor, measured_velocity: torch.Tensor
    ) -> None:
        assert self.adept_fabric.state is not None
        self._last_tracking_error = torch.amax(
            torch.abs(self.adept_fabric.state.position - measured_position), dim=1
        )
        threshold = self.cfg.fabric.tracking_error_resync_threshold
        if threshold <= 0.0:
            return
        env_ids = torch.nonzero(
            self._last_tracking_error > threshold, as_tuple=False
        ).squeeze(-1)
        if env_ids.numel() > 0:
            self.adept_fabric.reset(
                measured_position.index_select(0, env_ids),
                env_ids,
                measured_velocity.index_select(0, env_ids),
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        delayed_actions = self._delayed_actions(actions)
        measured_position, measured_velocity = self._measured_canonical_state()
        self._sync_tracking_outliers(measured_position, measured_velocity)
        assert self.adept_fabric.state is not None
        target = self.adept_fabric.state.position + (
            self.cfg.fabric.max_joint_delta * delayed_actions
        )
        target = self._apply_pca_prior(target)

        measured_collision = self.collision_adapter.build()
        for _ in range(self.cfg.fabric.steps_per_policy_step):
            assert self.adept_fabric.state is not None
            displacement = self.adept_fabric.state.position - measured_position
            predicted_clearance = measured_collision.clearance + torch.einsum(
                "bcd,bd->bc", measured_collision.jacobian, displacement
            )
            predicted_collision = CollisionBatch(
                clearance=predicted_clearance,
                jacobian=measured_collision.jacobian,
                enabled=measured_collision.enabled,
            )
            self.adept_fabric.step(target, predicted_collision)
        self._last_collision = predicted_collision

        state = self.adept_fabric.state
        assert state is not None
        canonical_ids = self._canonical_joint_ids_lab
        self._cur_targets[:, canonical_ids] = state.position
        self._cur_velocity_targets[:, canonical_ids] = state.velocity
        if self._mimic_target_joint_ids.numel() > 0:
            source_position = state.position[:, 7:].index_select(
                1, self._mimic_source_hand_action_ids
            )
            source_velocity = state.velocity[:, 7:].index_select(
                1, self._mimic_source_hand_action_ids
            )
            mimic_position = (
                source_position * self._mimic_multipliers + self._mimic_offsets
            )
            mimic_velocity = source_velocity * self._mimic_multipliers
            mimic_lower = self.robot.data.joint_pos_limits[
                :, self._mimic_target_joint_ids, 0
            ]
            mimic_upper = self.robot.data.joint_pos_limits[
                :, self._mimic_target_joint_ids, 1
            ]
            self._cur_targets[:, self._mimic_target_joint_ids] = torch.clamp(
                mimic_position, mimic_lower, mimic_upper
            )
            self._cur_velocity_targets[:, self._mimic_target_joint_ids] = mimic_velocity
        self._prev_targets = self._cur_targets.clone()
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        target_ids = self._position_target_joint_ids
        target_ids_list = self._position_target_joint_ids_list
        self.robot.set_joint_position_target(
            self._cur_targets[:, target_ids], joint_ids=target_ids_list
        )
        self.robot.set_joint_velocity_target(
            self.cfg.fabric.velocity_target_factor
            * self._cur_velocity_targets[:, target_ids],
            joint_ids=target_ids_list,
        )

    def _normalized_fabric_state(self) -> torch.Tensor:
        state = self.adept_fabric.state
        assert state is not None
        lower = self.adept_fabric.lower_limits
        upper = self.adept_fabric.upper_limits
        position = 2.0 * (state.position - lower) / (upper - lower) - 1.0
        config = self.cfg.fabric
        velocity_scale = torch.tensor(
            [config.max_arm_velocity] * 7 + [config.max_hand_velocity] * 6,
            device=self.device,
        )
        acceleration_scale = torch.tensor(
            [config.max_arm_acceleration] * 7
            + [config.max_hand_acceleration] * 6,
            device=self.device,
        )
        return torch.cat(
            (
                position,
                state.velocity / velocity_scale,
                state.acceleration / acceleration_scale,
            ),
            dim=1,
        )

    def _delayed_fabric_observation(self, value: torch.Tensor) -> torch.Tensor:
        dr = self.cfg.domain_randomization
        if not dr.use_obs_delay or dr.obs_delay_max <= 0:
            return value
        episode_start = (self.episode_length_buf == 0) & (self._successes == 0)
        if episode_start.any():
            self._fabric_obs_queue[episode_start] = value[episode_start].unsqueeze(1)
        self._fabric_obs_queue = torch.roll(self._fabric_obs_queue, shifts=1, dims=1)
        self._fabric_obs_queue[:, 0, :] = value
        delay_index = torch.randint(
            0, self._fabric_obs_queue.shape[1], (self.num_envs,), device=self.device
        )
        return self._fabric_obs_queue[
            torch.arange(self.num_envs, device=self.device), delay_index
        ]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = build_observations(self)
        if not self.cfg.fabric.append_state_to_observations:
            return observations
        fabric_state = self._normalized_fabric_state()
        observations["policy"] = torch.cat(
            (observations["policy"], self._delayed_fabric_observation(fabric_state)),
            dim=1,
        )
        observations["critic"] = torch.cat(
            (observations["critic"], fabric_state), dim=1
        )
        clip = self.cfg.obs.clamp_abs_observations
        observations["policy"].clamp_(-clip, clip)
        observations["critic"].clamp_(-clip, clip)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        if self._last_collision is not None:
            min_clearance = torch.amin(self._last_collision.clearance, dim=1)
            self._episode_min_fabric_clearance = torch.minimum(
                self._episode_min_fabric_clearance, min_clearance
            )
            active = (
                self._last_collision.clearance
                < self.cfg.fabric.collision_influence_distance
            )
            penetration = self._last_collision.clearance < 0.0
            self.extras["fabric/min_clearance_m"] = min_clearance.mean()
            self.extras["fabric/active_constraint_fraction"] = active.float().mean()
            self.extras["fabric/penetration_fraction"] = penetration.float().mean()
            self.extras["fabric/tracking_error_rad"] = self._last_tracking_error.mean()
            self.extras["fabric/pca_prior_weight"] = float(self._last_pca_weight)
            self.extras["true_objective"] = self._is_success.float().mean()
            self.extras["episode_cumulative"]["fabric_penetration"] = (
                penetration.any(dim=1).float()
            )
            self.extras["episode_cumulative"]["fabric_tracking_error"] = (
                self._last_tracking_error
            )
            self.extras["episode_final"]["fabric_min_clearance_m"] = (
                self._episode_min_fabric_clearance
            )

        final = self.extras.get("final_observation")
        if (
            self.cfg.fabric.append_state_to_observations
            and isinstance(final, dict)
            and "critic" in final
        ):
            final["critic"] = torch.cat(
                (final["critic"], self._normalized_fabric_state()), dim=1
            )
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, "adept_fabric"):
            position, velocity = self._measured_canonical_state()
            self.adept_fabric.reset(
                position.index_select(0, env_ids),
                env_ids,
                velocity.index_select(0, env_ids),
            )
            self._fabric_obs_queue[env_ids] = 0.0
            self._episode_min_fabric_clearance[env_ids] = torch.inf
            self._cur_velocity_targets[env_ids] = 0.0


__all__ = ["FABRIC_OBSERVATION_DIM", "G1Revo2AdeptEnv"]
