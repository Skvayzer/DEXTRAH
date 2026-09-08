"""Strict, function-preserving SAPG observation expansion (no optimizer resume)."""
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_state_dict(source, target, base_dim, feature_mean, feature_var):
    """Insert features BEFORE the learned 32-D SAPG coefficient embedding.

    Unrecognized shape differences are errors, never silent partial loads.
    The shared normalizer count is retained; new feature statistics are
    initialized from the complete current object bank, not a near-zero count.
    """
    if set(source) != set(target):
        raise ValueError(f"Checkpoint key mismatch: {set(source)^set(target)}")
    extra = feature_mean.numel()
    result, expanded = {}, []
    for key, old in source.items():
        new = target[key]
        if old.shape == new.shape:
            result[key] = old.to(device=new.device, dtype=new.dtype)
            continue
        if key.endswith(("running_mean_std.running_mean", "running_mean_std.running_var")):
            if old.shape != (base_dim,) or new.shape != (base_dim+extra,):
                raise ValueError(f"Invalid input normalizer shape: {key}")
            values = feature_mean if key.endswith("running_mean") else feature_var
            result[key] = torch.cat([old.to(new), values.to(new)])
        elif key.endswith(("a2c_network.rnn.rnn.weight_ih_l0", "a2c_network.actor_mlp.0.weight")):
            if (old.ndim != 2 or new.shape != (old.shape[0], old.shape[1]+extra)
                    or old.shape[1] != base_dim+32):
                raise ValueError(f"Invalid SAPG input matrix shape: {key}")
            result[key] = torch.zeros_like(new)
            result[key][:, :base_dim] = old[:, :base_dim].to(new)
            result[key][:, base_dim+extra:] = old[:, base_dim:].to(new)
        else:
            raise ValueError(f"Unexpected checkpoint shape mismatch: {key}: {old.shape} -> {new.shape}")
        expanded.append(key)
    if len(expanded) != 3:
        raise ValueError(f"Expected input matrix + two normalizer buffers, got {expanded}")
    return result


def find_task_env(vec_env):
    node = vec_env
    for _ in range(8):
        if hasattr(node, "unwrapped"):
            node = node.unwrapped
        if hasattr(node, "_bps_features"):
            return node
        node = node.env
    raise RuntimeError("Could not locate BPS task")


def validate_source_contract(source_config, env):
    """Compare saved source MDP fields, not just compatible tensor lengths."""
    def plain(value):
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        return value
    differences = []
    for name in ("decimation", "episode_length_s", "is_finite_horizon"):
        if getattr(env.cfg, name) != source_config["env"][name]:
            differences.append(f"{name}: mismatch")
    for name in ("dt", "gravity"):
        if plain(getattr(env.cfg.sim, name)) != plain(source_config["env"]["sim"][name]):
            differences.append(f"sim.{name}: mismatch")
    for group in ("action", "reward", "reset", "termination", "domain_randomization", "assets", "obs"):
        current = getattr(env.cfg, group)
        for name, expected in source_config["env"][group].items():
            if not hasattr(current, name):
                differences.append(f"{group}.{name}: missing")
                continue
            actual = plain(getattr(current, name))
            if actual != plain(expected):
                differences.append(f"{group}.{name}: {actual!r} != {expected!r}")
    if differences:
        raise ValueError("Source checkpoint MDP differs:\n" + "\n".join(differences))


class BpsWarmstartObserver:
    """Loaded after network creation, before the first SAPG rollout."""

    def __init__(self, checkpoint, source_config, expected_sha256):
        self.checkpoint = Path(checkpoint)
        self.source_config = Path(source_config)
        self.expected_sha256 = expected_sha256

    def before_init(self, *args):
        pass

    def after_init(self, algo):
        import yaml
        from rl_games.algos_torch import model_builder

        self.algo = algo
        env = find_task_env(algo.vec_env)
        source_cfg = yaml.safe_load(self.source_config.read_text())
        validate_source_contract(source_cfg, env)
        digest = sha256_file(self.checkpoint)
        if digest != self.expected_sha256:
            raise ValueError(f"Wrong source checkpoint checksum: {digest}")
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        source = payload[0] if 0 in payload else payload
        source_actor = source["model"]
        source_critic = source["assymetric_vf_nets"]
        mean = env._bps_features.double().mean(0)
        variance = env._bps_features.double().var(0, unbiased=False).clamp_min(1e-8)
        algo.model.load_state_dict(expand_state_dict(source_actor, algo.model.state_dict(), 92, mean, variance), strict=True)
        algo.central_value_net.load_state_dict(expand_state_dict(source_critic,
            algo.central_value_net.state_dict(), 114, mean, variance), strict=True)
        if algo.optimizer.state or algo.central_value_net.optimizer.state:
            raise RuntimeError("Warm start must use fresh optimizers")

        # Real recorded source observations, all six SAPG coefficient groups,
        # nonzero recurrent states, plus arbitrary BPS values. Keep RNG intact.
        device = algo.ppo_device
        ids = torch.cat([torch.arange(i*4096, i*4096+8) for i in range(6)])
        coef_ids = algo.intr_reward_coef_embd[::algo.intr_coef_block_size, 0]
        if coef_ids.numel() != 6:
            raise ValueError("The source checkpoint requires six SAPG groups")
        old_params = source_cfg["agent"]["params"]
        maxima = dict(actor_mean=0., actor_sigma=0., actor_recurrent=0., critic_value=0.)
        rng_devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        was_training = algo.model.training
        was_critic_training = algo.central_value_net.model.training
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            actor_builder = model_builder.ModelBuilder().load(old_params)
            old_actor = actor_builder.build(dict(actions_num=13, input_shape=(124,), num_seqs=len(ids),
                value_size=1, normalize_value=True, normalize_input=True, type="extra_param",
                coef_ids=coef_ids, coef_id_idx=92)).to(device).eval()
            old_actor.load_state_dict(source_actor, strict=True)
            critic_builder = model_builder.ModelBuilder().load(dict(model={"name": "central_value"},
                network=old_params["config"]["central_value_config"]["network"]))
            old_critic = critic_builder.build(dict(actions_num=13, input_shape=(146,), num_seqs=len(ids),
                value_size=1, normalize_value=True, normalize_input=True, type="extra_param",
                coef_ids=coef_ids, coef_id_idx=114)).to(device).eval()
            old_critic.load_state_dict({k.removeprefix("model."): v for k, v in source_critic.items()}, strict=True)
            algo.model.eval()
            algo.central_value_net.model.eval()
            raw = source["obs"]["obs"][ids].to(device)
            states = source["obs"]["states"][ids].to(device)
            old_rnn = [v[:, ids].to(device) for v in source["rnn_states"]]
            new_rnn = [v.clone() for v in old_rnn]
            for _ in range(8):
                shape = env._bps_features[torch.arange(len(ids), device=device) % env.num_envs]
                augmented = torch.cat([raw[:, :92], shape, raw[:, 92:]], -1)
                options = dict(is_train=True, prev_actions=torch.zeros(len(ids), 13, device=device), seq_length=1)
                old = old_actor(dict(options, obs=raw, rnn_states=old_rnn))
                new = algo.model(dict(options, obs=augmented, rnn_states=new_rnn))
                for field, metric in (("mus", "actor_mean"), ("sigmas", "actor_sigma")):
                    error = float((old[field]-new[field]).abs().max())
                    maxima[metric] = max(maxima[metric], error)
                    torch.testing.assert_close(old[field], new[field], atol=5e-5, rtol=1e-5)
                for a, b in zip(old["rnn_states"], new["rnn_states"]):
                    maxima["actor_recurrent"] = max(maxima["actor_recurrent"], float((a-b).abs().max()))
                    torch.testing.assert_close(a, b, atol=5e-5, rtol=1e-5)
                old_rnn, new_rnn = old["rnn_states"], new["rnn_states"]
            augmented_states = torch.cat([states[:, :114], shape, states[:, 114:]], -1)
            old_value = old_critic(dict(obs=states, is_train=False))["values"]
            new_value = algo.central_value_net.model(dict(obs=augmented_states, is_train=False))["values"]
            maxima["critic_value"] = float((old_value-new_value).abs().max())
            torch.testing.assert_close(old_value, new_value, atol=5e-4, rtol=1e-5)
            del old_actor, old_critic
        algo.model.train(was_training)
        algo.central_value_net.model.train(was_critic_training)
        self.report = dict(source_checkpoint=str(self.checkpoint), source_sha256=digest,
            source_epoch=source["epoch"], source_frames=source["frame"], source_env_state=source.get("env_state"),
            fresh_optimizer=True, counters_reset=True, maximum_errors=maxima,
            actor_dim=224, critic_dim=246, shape_dim=132, sapg_groups=6,
            tolerance_initial=float(env._current_success_tolerance),
            tolerance_note="Source checkpoint has no saved curriculum state; original curriculum restarts explicitly")
        (Path(env.cfg.bps_artifact_dir) / "warmstart_validation.json").write_text(json.dumps(self.report, indent=2)+"\n")
        # Small auditable weights-only snapshot; no 24k-environment recurrent
        # states or old rollout buffers are restored into the new experiment.
        torch.save(dict(model=algo.model.state_dict(), assymetric_vf_nets=algo.central_value_net.state_dict(),
                        bps_warmstart=self.report), Path(env.cfg.bps_artifact_dir) / "initial_weights.pth")
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({"bps_warmstart": self.report,
                                    "bps_bank_sha256": env._bps_manifest["features_sha256"]})
        except ImportError:
            pass
        print("BPS_WARMSTART_VALIDATED " + json.dumps(self.report), flush=True)
        del payload, source

    def process_infos(self, *args, **kwargs):
        pass

    def after_steps(self):
        pass

    def after_clear_stats(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        actor = self.algo.model.a2c_network.rnn.rnn.weight_ih_l0[:, 92:224]
        critic = self.algo.central_value_net.model.a2c_network.actor_mlp[0].weight[:, 114:246]
        self.algo.writer.add_scalar("shape/actor_input_weight_norm", float(actor.detach().norm()), frame)
        self.algo.writer.add_scalar("shape/critic_input_weight_norm", float(critic.detach().norm()), frame)
