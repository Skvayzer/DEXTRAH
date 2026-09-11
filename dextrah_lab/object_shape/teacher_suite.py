"""Predeclared frozen-teacher evaluation; independent of Isaac imports."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

SEEDS = (42, 43, 44)
HELDOUT_OBJECT_SEED = 914271
CONDITIONS = {
    "training": {"object_seed": 42, "wrench_multiplier": 1.0},
    "heldout": {"object_seed": HELDOUT_OBJECT_SEED, "wrench_multiplier": 1.0},
    "stress": {"object_seed": 42, "wrench_multiplier": 2.0},
}


def geometry_hash(path):
    """Ignore mass, density, names and appearance when detecting shape leakage."""
    root = ET.parse(path).getroot()
    collisions = []
    for collision in root.findall("./link/collision"):
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError("Expected one primitive per collision")
        shape = geometry[0]
        if shape.tag not in ("box", "cylinder"):
            raise ValueError("Only procedural boxes/cylinders supported")
        def numbers(text):
            return tuple(float(x) for x in text.split())
        collisions.append((shape.tag,
            tuple(sorted((k, numbers(v)) for k, v in shape.attrib.items())),
            numbers(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
            numbers(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")))
    if not collisions:
        raise ValueError("No collision geometry")
    return hashlib.sha256(json.dumps(sorted(collisions)).encode()).hexdigest()


@contextmanager
def object_seed_override(scene_utils, seed, source_manifest):
    """Override only this scene's generator, then restore even on failure.

    Held-out uses the same six families/distributions but new sampled geometry,
    not unseen semantic families. Also reconstruct the exact training pool and
    reject geometry overlap (URDF hash alone would miss density-only changes).
    Generated files live only in newly allocated temporary directories.
    """
    original = scene_utils.generate_handle_head_urdfs
    audit = {"object_seed": seed, "training_geometry_overlap": None}
    def generate(**kwargs):
        paths, scales = original(**dict(kwargs, seed=seed))
        if seed != 42:
            import numpy as np
            state = np.random.get_state()
            try:
                with tempfile.TemporaryDirectory(prefix="g1_training_shape_audit_") as temp:
                    training, _ = original(**dict(kwargs, seed=42, out_dir=temp))
                    expected = {e["urdf_sha256"] for e in source_manifest["entries"]}
                    observed = {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in training}
                    if observed != expected:
                        raise ValueError("Reconstructed training pool differs from checkpoint manifest")
                    overlap = {geometry_hash(p) for p in paths} & {geometry_hash(p) for p in training}
                    audit["training_geometry_overlap"] = len(overlap)
                    if overlap:
                        raise ValueError("Held-out geometry overlaps training pool")
            finally:
                np.random.set_state(state)
        return paths, scales
    scene_utils.generate_handle_head_urdfs = generate
    try:
        yield audit
    finally:
        scene_utils.generate_handle_head_urdfs = original


def cases():
    return [dict(candidate=candidate, condition=condition, seed=seed,
                 **CONDITIONS[condition])
            for candidate in ("bps_baseline", "touch_best", "touch_final")
            for condition in CONDITIONS for seed in SEEDS]


def summarize(records):
    """Select using validation training-bank throughput, NOT held-out results.

    Minimum regression guards: mean episode success and lift each within two
    percentage points of baseline; stress throughput >=95% of baseline.
    Raw per-seed results remain authoritative. This is not an equal-budget
    causal tactile ablation and three seeds do not establish significance.
    """
    expected = {(c["candidate"], c["condition"], c["seed"]) for c in cases()}
    actual = [(r["candidate"], r["condition"], r["seed"]) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Need every unique predeclared evaluation before selection")
    metrics = ("episode_any_goal_success_rate", "episode_ever_lifted_rate",
               "goals_per_simulated_minute")
    import math
    grouped = {}
    for candidate, condition, _ in sorted(expected):
        subset = [r for r in records if (r["candidate"], r["condition"]) == (candidate, condition)]
        values = {}
        for metric in metrics:
            data = [r["report"][metric] for r in subset]
            if any(x is None or not math.isfinite(x) for x in data):
                raise ValueError("Invalid or empty evaluation metric")
            values[metric] = {"mean": sum(data)/len(data), "per_seed": data}
        grouped.setdefault(candidate, {})[condition] = values
    def mean(candidate, condition, metric):
        return grouped[candidate][condition][metric]["mean"]
    eligible = []
    for candidate in grouped:
        if (all(mean(candidate, "training", m) >= mean("bps_baseline", "training", m)-.02
                for m in metrics[:2]) and
            mean(candidate, "stress", metrics[2]) >= .95*mean("bps_baseline", "stress", metrics[2])):
            eligible.append(candidate)
    winner = max(eligible, key=lambda c: mean(c, "training", metrics[2]))
    return {"selected_candidate": winner, "eligible_candidates": eligible,
            "metrics": grouped, "heldout_used_for_selection": False,
            "causal_tactile_benefit_established": False}
