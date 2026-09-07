#!/usr/bin/env python3
"""Build evidence-backed PCA figures and motion inputs without changing training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dextrah_lab.retargeting import PCAArtifact, Revo2Kinematics, FrozenPCAHandActionMap
from dextrah_lab.retargeting.dexycb import palm_relative_points

NAVY, TEAL, PURPLE, ORANGE = "#193348", "#008f95", "#7755be", "#da8228"
JOINT_LABELS = ["Thumb\nrotation", "Thumb\nflexion", "Index", "Middle", "Ring", "Little"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def feasible_interval(mean, direction, lower, upper):
    """Scalar interval keeping mean + scalar * direction within joint limits."""
    active = np.abs(direction) > 1.e-12
    a = (lower[active] - mean[active]) / direction[active]
    b = (upper[active] - mean[active]) / direction[active]
    return max(np.minimum(a, b)), min(np.maximum(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hand-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    for sub in ("figures", "stills", "videos", "sources", "data"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / "summary.json").read_text())
    validation = json.loads((args.run / "validation.json").read_text())
    artifact_path = args.run / "revo2_human_motion_pca.npz"
    artifact = PCAArtifact.load(artifact_path)
    assert sha(artifact_path) == "8cea2fe7602958bbefec826fd331a100c15145c7dd837c68406a07895a2237b3"
    assert sha(summary["urdf"]) == artifact.metadata["urdf_sha256"]
    assert validation["passed"]
    hand = Revo2Kinematics(args.hand_urdf)
    paths = [args.run / item["trajectory"] for item in summary["trajectories"]]
    arrays = [load_npz(path) for path in paths]
    q = np.concatenate([a["joint_positions"] for a in arrays])
    mode = np.concatenate([np.full(len(a["gamma"]), str(a["grasp_mode"]) == "precision") for a in arrays])
    gamma = np.concatenate([a["gamma"] for a in arrays])
    errors = np.concatenate([a["fingertip_error"] for a in arrays])
    z = artifact.task_coordinates(q)
    projected = artifact.reconstruct(z)
    per_frame_rmse = np.sqrt(((q - projected)**2).mean(axis=1))
    singular = np.linalg.svd(q - q.mean(axis=0), compute_uv=False)
    ratios = singular**2 / (singular**2).sum()
    np.testing.assert_allclose(ratios, artifact.explained_variance_ratio, atol=1.e-12)
    np.testing.assert_allclose(q.mean(axis=0), artifact.mean, atol=1.e-12)
    assert len(q) == 62892
    # Verify the presentation's blend against the actual runtime implementation.
    runtime = FrozenPCAHandActionMap(artifact, dtype=torch.float64)
    soft = q + .05 * (np.clip(projected, artifact.joint_lower, artifact.joint_upper) - q)
    np.testing.assert_allclose(runtime.soft_prior(torch.tensor(q), .05).numpy(), soft, atol=1.e-12)
    cohort = Counter(item["subject"] for item in summary["trajectories"] if item["grasp_mode"] == "power")
    object_counts = Counter(item["ycb_grasp_id"] for item in summary["trajectories"] if item["grasp_mode"] == "power")
    info = dict(artifact_sha256=sha(artifact_path), source_urdf_sha256=sha(summary["urdf"]),
        hand_urdf=str(args.hand_urdf), subject_captures=dict(cohort), source_captures=501,
        retargeted_trajectories=len(paths), robot_configurations=len(q), alpha=summary["scale"],
        explained_variance_ratio=ratios.tolist(), retained_variance=float(ratios[:5].sum()),
        mean_per_trajectory_projection_rmse_rad=float(np.mean([
            np.sqrt(np.mean((a["joint_positions"] - artifact.reconstruct(artifact.task_coordinates(a["joint_positions"])))**2))
            for a in arrays])),
        mean_imitation_endpoint_error_mm=float(errors[gamma >= .999].mean()*1000),
        global_projection_rmse_rad=float(np.sqrt(((q - projected)**2).mean())),
        runtime_soft_prior_weight=.05, runtime_hand_actions=6,
        object_captures=dict(sorted(object_counts.items())), unique_objects=len(object_counts),
        source_frames=len(q)//2, cameras_per_capture_used=1,
        provenance="Actual saved robot-motion dataset; no RL improvement or human-style score inferred",
        motions=[], preset_motions=[])
    # Fixed, disclosed examples, not selected by minimum error.
    requested = [(1, "power"), (1, "precision"), (17, "power"), (17, "precision")]
    for index, (object_id, grasp_mode) in enumerate(requested):
        candidates = [i for i, item in enumerate(summary["trajectories"])
                      if item["ycb_grasp_id"] == object_id and item["grasp_mode"] == grasp_mode]
        if not candidates:
            raise ValueError(f"Missing example for object {object_id}")
        a = arrays[candidates[0]]
        entry = summary["trajectories"][candidates[0]]
        camera = args.dataset / entry["subject"] / entry["capture"] / entry["camera_serial"]
        joints, rgb_paths = [], []
        for frame in a["frame_indices"]:
            label = load_npz(camera / f"labels_{int(frame):06d}.npz")
            joints.append(label["joint_3d"].reshape(21, 3))
            rgb = camera / f"color_{int(frame):06d}.jpg"
            if not rgb.is_file():
                raise FileNotFoundError(rgb)
            rgb_paths.append(str(rgb))
        joints = np.stack(joints)
        human = summary["scale"] * palm_relative_points(joints, joints)
        np.testing.assert_allclose(human[:, [4, 8, 12, 16, 20]], a["scaled_human_fingertips"], atol=1.e-10)
        tips = hand.fingertip_positions(torch.tensor(a["joint_positions"])).numpy()
        np.testing.assert_allclose(tips, a["robot_fingertips"], atol=1.e-10)
        fname = f"motion_{index+1:02d}.npz"
        reconstructed = np.clip(artifact.reconstruct(artifact.task_coordinates(a["joint_positions"])),
                                artifact.joint_lower, artifact.joint_upper)
        np.savez_compressed(args.output / "data" / fname, **a, human_joints=human,
                            reconstructed=reconstructed, rgb_paths=np.asarray(rgb_paths))
        info["motions"].append(dict(file=fname, **entry,
            label=f"{'Coffee can' if object_id == 1 else 'Scissors'} / {grasp_mode}",
            timing="Each recorded frame shown twice at 30 FPS (0.5x), with 1-second endpoint holds"))
        for name, f in (("start", 0), ("middle", len(rgb_paths)//2), ("end", len(rgb_paths)-1)):
            shutil.copy2(rgb_paths[f], args.output / "sources" / f"matched_{index+1:02d}_{name}.jpg")
    preset_summary = json.loads((args.presets / "summary.json").read_text())
    for entry in preset_summary["trajectories"]:
        a = load_npz(args.presets / entry["trajectory"])
        np.testing.assert_allclose(hand.fingertip_positions(torch.tensor(a["joint_positions"])).numpy(),
                                   a["robot_fingertips"], atol=1.e-10)
        reconstructed = np.clip(artifact.reconstruct(artifact.task_coordinates(a["joint_positions"])),
                                artifact.joint_lower, artifact.joint_upper)
        name = f"preset_{entry['grasp_mode']}.npz"
        np.savez_compressed(args.output / "data" / name, **a, reconstructed=reconstructed)
        info["preset_motions"].append(dict(file=name, **entry))
    centered_z = (q - artifact.mean) @ artifact.components.T
    sweeps = []
    for pc, direction in enumerate(artifact.components):
        low, high = feasible_interval(artifact.mean, direction, artifact.joint_lower, artifact.joint_upper)
        observed = np.quantile(centered_z[:, pc], [.05, .95])
        sweeps.append([max(float(observed[0]), .98*low), min(float(observed[1]), .98*high)])
    _, _, vt = np.linalg.svd(artifact.components, full_matrices=True)
    residual = vt[-1]
    low, high = feasible_interval(artifact.mean, residual, artifact.joint_lower, artifact.joint_upper)
    np.savez_compressed(args.output / "data" / "analysis.npz", joint_positions=q,
        mode=mode, coordinates=z, components=artifact.components, mean=artifact.mean,
        lower=artifact.joint_lower, upper=artifact.joint_upper,
        per_frame_rmse=per_frame_rmse, explained_variance_ratio=ratios,
        sweeps=np.array(sweeps), residual=residual, residual_bounds=np.array([.8*low, .8*high]))
    info["pc_sweeps"] = sweeps
    info["pc_sweep_definition"] = "One component at a time; empirical 5–95% coordinates intersected with joint-feasible range; not a recorded human sequence"
    (args.output / "data" / "measurements.json").write_text(json.dumps(info, indent=2))
    (args.output / "data" / "explained_variance.csv").write_text(
        "component,individual_percent,cumulative_percent,retained\n" + "".join(
            f"{i+1},{100*v:.10f},{100*ratios[:i+1].sum():.10f},{i<5}\n" for i,v in enumerate(ratios)))
    for name in ("summary.json", "validation.json", "revo2_human_motion_pca.npz"):
        shutil.copy2(args.run / name, args.output / "data" / name)
    make_figures(args.output, artifact, info, q, mode, z, gamma, errors, per_frame_rmse)
    download_sources(args.output)
    print("PCA_PRESENTATION_PREPARED", json.dumps(info, indent=2), flush=True)


def save_figure(fig, output, name, footnote="Measured from our frozen G1 + Revo2 robot-motion dataset; not a policy success metric."):
    fig.text(.04, .025, footnote, fontsize=10, color="#596778")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / "figures" / f"{name}.{extension}", dpi=160, facecolor="white")
    plt.close(fig)


def make_figures(output, artifact, info, q, mode, z, gamma, errors, per_frame_rmse):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": NAVY, "text.color": NAVY, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.08, right=.96, bottom=.18, top=.80, wspace=.32)
    fig.suptitle("Five PCA coordinates retain 99.81% of robot-motion variance", fontsize=20, y=.94)
    values = np.array(info["explained_variance_ratio"])*100
    axes[0].bar(range(1, 7), values, color=[TEAL]*5 + ["#c9ced6"])
    for i, v in enumerate(values):
        axes[0].text(i+1, v+1.6, f"{v:.2f}%", ha="center", fontsize=11)
    axes[0].set(xlabel="Principal component", ylabel="Explained variance (%)", ylim=(0, 100), xticks=range(1, 7))
    axes[1].plot(range(1, 7), np.cumsum(values), "o-", color=PURPLE, lw=3)
    axes[1].axvline(5, color=TEAL, ls="--", alpha=.6)
    axes[1].axhline(98, color=ORANGE, ls=":", lw=1.8)
    axes[1].text(1.1, 97, "98% reference", fontsize=10, color=ORANGE)
    axes[1].set(xlabel="Components retained", ylabel="Cumulative variance (%)", ylim=(80, 101), xticks=range(1, 7))
    axes[1].text(1, 83, "6 Revo2 actuators → 5 components\n62,892 optimized configurations\nPower + precision retargeting", fontsize=12)
    save_figure(fig, output, "01_explained_variance", "Five components were retained by design; four already explain 99.29%. Variance is in retargeted robot angles, not raw human motion.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.07, right=.96, bottom=.20, top=.80, wspace=.30)
    subjects = list(info["subject_captures"])
    counts = list(info["subject_captures"].values())
    axes[0].bar(range(1, 11), counts, color=TEAL)
    for i,count in enumerate(counts):
        axes[0].text(i+1,count+1,str(count),ha="center",fontsize=11)
    axes[0].set(xlabel="DexYCB subject",ylabel="Unique right-hand captures",ylim=(0,60),xticks=range(1,11))
    axes[1].set_axis_off()
    lines = ["501 unique right-hand captures", f"{info['unique_objects']} object identities · 10 subjects", "One camera per physical capture",
             f"{info['source_frames']:,} retained source frames", "× 2 offline modes: power + precision", "= 62,892 six-joint configurations"]
    for i,line in enumerate(lines):
        axes[1].text(0, .95-i*.15, line,fontsize=15,color=TEAL if i in (0,5) else NAVY,transform=axes[1].transAxes)
    fig.suptitle("Which part of DexYCB was used to fit our PCA?",fontsize=22,y=.95)
    save_figure(fig, output, "06_dataset_subset", "Left-hand captures excluded. Invalid labels removed; motion window retained. No eight-view duplication and no held-out evaluation implied.")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=.12, right=.91, bottom=.19, top=.84)
    im = ax.imshow(artifact.components, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set(xticks=range(6), xticklabels=JOINT_LABELS, yticks=range(5), yticklabels=[f"PC {i+1}" for i in range(5)])
    for (r, c), v in np.ndenumerate(artifact.components):
        ax.text(c, r, f"{v:+.2f}", ha="center", va="center", color="white" if abs(v)>.55 else NAVY)
    fig.colorbar(im, ax=ax, label="Signed joint loading")
    fig.suptitle("The learned synergies coordinate six motor commands", fontsize=21, y=.95)
    save_figure(fig, output, "02_component_loadings", "Component signs are arbitrary; PCA axes are not semantic grasp labels. Distal joints follow the URDF couplings.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    fig.subplots_adjust(left=.08, right=.97, bottom=.19, top=.82, wspace=.28)
    rng = np.random.default_rng(42)
    for value, label, color in ((False, "Power retargeting", TEAL), (True, "Precision retargeting", PURPLE)):
        selected = np.flatnonzero(mode == value)
        selected = rng.choice(selected, min(5000, len(selected)), replace=False)
        axes[0].scatter(z[selected, 0], z[selected, 1], s=4, alpha=.16, color=color, label=label, rasterized=True)
    axes[0].set(xlabel="PC1 coordinate (rad)", ylabel="PC2 coordinate (rad)")
    axes[0].legend(markerscale=4, frameon=False)
    axes[1].hist(np.rad2deg(per_frame_rmse), bins=55, color=TEAL, alpha=.85)
    axes[1].set(xlabel="Per-frame joint reconstruction RMSE (degrees)", ylabel="Robot configurations")
    axes[1].text(.97, .94, f"Mean per-trajectory RMSE\n{info['mean_per_trajectory_projection_rmse_rad']:.5f} rad", transform=axes[1].transAxes, ha="right", va="top")
    fig.suptitle("The prior models retargeted robot postures—not task success", fontsize=20, y=.95)
    save_figure(fig, output, "03_latent_space_and_reconstruction")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    fig.subplots_adjust(left=.08, right=.96, bottom=.20, top=.80, wspace=.30)
    axes[0].plot([0, 1], [1, 0], lw=4, color=ORANGE, label="Human imitation weight γ")
    axes[0].plot([0, 1], [0, 1], lw=4, color=TEAL, label="Closure weight 1−γ")
    axes[0].set(xlabel="Normalized motion progress", ylabel="Loss weight", ylim=(-.05, 1.15))
    axes[0].legend(frameon=False, fontsize=11)
    bins = np.linspace(0, 1, 11)
    medians, p95 = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        e = errors[(1-gamma >= low) & (1-gamma <= high)]*1000
        medians.append(np.median(e)); p95.append(np.percentile(e, 95))
    axes[1].plot((bins[:-1]+bins[1:])/2, medians, "o-", color=TEAL, label="Median fingertip error")
    axes[1].plot((bins[:-1]+bins[1:])/2, p95, "--", color=ORANGE, label="95th percentile")
    axes[1].set(xlabel="Normalized motion progress", ylabel="Distance to scaled human tips (mm)")
    axes[1].legend(frameon=False, fontsize=11)
    fig.suptitle("Retargeting blends human imitation with grasp closure", fontsize=21, y=.95)
    save_figure(fig, output, "04_retargeting_objective", "Late frames intentionally favor closure. Larger late imitation error is not automatically optimizer failure.")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=.10, right=.94, bottom=.24, top=.78)
    x = np.arange(6)
    for offset, label, vals, color in ((-.24, "No prior", [1]*6, "#8793a3"),
        (0, "Current: 5% prior", [1]*5+[.95], TEAL), (.24, "Full projection (diagnostic)", [1]*5+[0], PURPLE)):
        ax.bar(x+offset, vals, width=.22, label=label, color=color)
    ax.set(xticks=x, xticklabels=[f"PC{i+1}" for i in range(5)]+["Residual\ndirection"],
           ylabel="Local target-motion gain", ylim=(0, 1.25))
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=11)
    fig.suptitle("SAPG keeps six hand actions; the PCA correction is only 5%", fontsize=20, y=.94)
    fig.text(.10, .11, "q_target = q_nominal + 0.05 × [project_PCA(q_nominal) − q_nominal]", fontsize=17, color=TEAL)
    save_figure(fig, output, "05_soft_prior_not_action_restriction", "Local gain interpretation holds away from joint clipping. Apply once to each nominal target; no recursive projection.")

    fig, ax = plt.subplots(figsize=(12, 6.75)); ax.set_axis_off()
    fig.suptitle("Human motion → offline robot data → a frozen prior for RL", fontsize=20, y=.95)
    boxes = [(0.025,.50,"DexYCB","501 right-hand captures\n21 labeled 3D joints\n5 fingertip targets",ORANGE),
             (.355,.50,"Offline Adam IK","6 Revo2 motor angles\nImitation + closure\n1,002 motion traces",TEAL),
             (.685,.50,"Frozen PCA","62,892 configurations\n5 × 6 projection matrix\n99.81% retained variance",PURPLE)]
    for x,y,title,body,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y), .29,.34,boxstyle="round,pad=0.015", ec=color, fc="#f4f7fa", lw=2, transform=ax.transAxes))
        ax.text(x+.145,y+.265,title,ha="center",fontsize=17,color=color,transform=ax.transAxes)
        ax.text(x+.145,y+.125,body,ha="center",va="center",fontsize=13,linespacing=1.6,transform=ax.transAxes)
    for x in (.32,.65):
        ax.annotate("",xy=(x+.03,.67),xytext=(x,.67),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=2,color=NAVY))
    ax.text(.5,.39,"OFFLINE: optimize robot joint configurations; fit PCA once",ha="center",fontsize=14,color=TEAL,transform=ax.transAxes)
    ax.annotate("",xy=(.82,.25),xytext=(.82,.48),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=2,color=PURPLE))
    ax.text(.50,.19,"ONLINE SAPG: 7 arm + 6 hand actions → 5% PCA correction → fabrics",ha="center",fontsize=15,transform=ax.transAxes)
    ax.text(.50,.09,"RL optimizes the policy. The offline retargeter and PCA remain frozen.",ha="center",fontsize=13,color="#596778",transform=ax.transAxes)
    save_figure(fig, output, "00_pipeline", "Our G1 + Revo2 implementation. Inspired by DextrAH-G Appendix D; current soft-prior integration differs from its action space.")


def download_sources(output):
    assets = [
        ("dexycb_official_overview.mp4", "https://dex-ycb.github.io/assets/dex-ycb-render.mp4", "Official dataset visualization; unchanged download"),
        ("dexycb_cvpr2021.pdf", "https://dex-ycb.github.io/assets/chao_cvpr2021.pdf", "Original DexYCB paper"),
        ("dexycb_supplementary.pdf", "https://dex-ycb.github.io/assets/chao_cvpr2021_supp.pdf", "Original DexYCB supplementary material"),
    ]
    manifest = []
    for name, url, description in assets:
        path = output / "sources" / name
        with urllib.request.urlopen(url, timeout=90) as response, path.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        manifest.append(dict(file=name, url=url, description=description, sha256=sha(path),
            attribution="Yu-Wei Chao et al., DexYCB, CVPR 2021; NVIDIA and University of Washington",
            usage="Dataset: CC BY-NC 4.0. Paper figures retain their original publication attribution."))
    (output / "sources" / "download_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
