#!/usr/bin/env python3
"""Package figures, source assets, citations, CSVs, and a local meeting gallery."""
import argparse
import csv
import html
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from prepare_pca_presentation import load_npz, save_figure, sha, NAVY, TEAL, ORANGE


def run(*args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True).stdout


def make_tables(root, summary, info):
    captures = [item for item in summary["trajectories"] if item["grasp_mode"] == "power"]
    assert len(captures) == len({(i["subject"],i["capture"]) for i in captures}) == 501
    assert sum(i["frames"] for i in captures) == 31446
    with (root / "data" / "source_capture_manifest.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject","capture","camera_serial","ycb_grasp_id","frames"])
        writer.writeheader()
        writer.writerows({key:item[key] for key in writer.fieldnames} for item in captures)
    for name,key,field in (("subject_counts","subject_captures","subject"),("object_counts","object_captures","ycb_object_id")):
        with (root / "data" / f"{name}.csv").open("w") as stream:
            writer = csv.writer(stream); writer.writerow([field,"unique_captures"])
            writer.writerows(info[key].items())
    analysis = load_npz(root / "data" / "analysis.npz")
    with (root / "data" / "pca_vectors.csv").open("w") as stream:
        writer = csv.writer(stream); writer.writerow(["component", *summary["joint_names"]])
        for i,row in enumerate(analysis["components"]):
            writer.writerow([f"PC{i+1}",*row])


def extra_figures(root, info):
    fig,ax = plt.subplots(figsize=(12,6.75))
    fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.81)
    ids,counts = zip(*info["object_captures"].items())
    ax.bar(np.arange(len(ids)),counts,color=TEAL)
    for i,c in enumerate(counts):
        ax.text(i,c+.5,str(c),ha="center",fontsize=10)
    ax.set(xticks=np.arange(len(ids)),xticklabels=ids,xlabel="Official YCB object ID",
           ylabel="Unique right-hand captures",ylim=(0,33))
    fig.suptitle("All 20 DexYCB object identities contribute to the prior",fontsize=20,y=.95)
    save_figure(fig,root,"07_object_coverage","Counts are physical captures, before generating power and precision retargetings; this is not an RL training object set.")
    a = load_npz(root / "data" / info["motions"][0]["file"])
    joints = a["human_joints"][0]*100
    tips = a["robot_fingertips"][0]*100
    fig = plt.figure(figsize=(12,6.75))
    fig.subplots_adjust(left=.02,right=.98,bottom=.14,top=.82,wspace=.06)
    colors = ["#cb7e29","#3988b5","#3c9b77","#8863b7","#c76b89"]
    for panel in range(2):
        ax = fig.add_subplot(1,2,panel+1,projection="3d")
        for finger in range(5):
            indices = [0,*range(1+4*finger,5+4*finger)]
            p = joints[indices]
            ax.plot(p[:,0],p[:,1],p[:,2],"o-",color=colors[finger],lw=2,alpha=.9 if panel==0 else .35,ms=4)
        if panel:
            human = joints[[4,8,12,16,20]]
            ax.scatter(*tips.T,c=TEAL,s=60,label="Revo2 fingertip FK")
            ax.scatter(*human.T,c=ORANGE,s=45,label="Scaled human tips")
            for x,y in zip(human,tips):
                ax.plot(*np.stack((x,y)).T,color="#cf6767",ls="--",lw=1.2)
            ax.legend(loc="upper left",fontsize=10,frameon=False)
        else:
            for axis,color,label in ((0,"#cc5555","Palm normal"),(1,"#53905b","Thumb side"),(2,"#557abb","Fingers")):
                endpoint=np.eye(3)[axis]*4
                ax.quiver(0,0,0,*endpoint,color=color)
        ax.set(xlabel="x (cm)",ylabel="y (cm)",zlabel="z (cm)",xlim=(-5,8),ylim=(-7,8),zlim=(-1,17))
        ax.view_init(elev=20,azim=35)
        ax.set_title("21 joints → a palm-fixed frame" if panel==0 else "Five fingertip targets → six robot angles",fontsize=14)
    fig.suptitle("What gets retargeted? Wrist-aligned fingertip geometry",fontsize=21,y=.95)
    save_figure(fig,root,"08_human_keypoints_and_retargeting","Actual first retained frame (γ=1) of the disclosed coffee-can capture; human scale α=0.90. Lines indicate geometric error, not forces.")


def sources(root):
    source = root / "sources"
    (source / "paper_images").mkdir(exist_ok=True)
    run("pdfimages","-f","1","-l","2","-png",source / "dexycb_cvpr2021.pdf",source / "paper_images" / "dexycb")
    run("pdftoppm","-f","3","-l","3","-singlefile","-r","240","-x","160","-y","1100","-W","860","-H","620","-png",
        source / "dexycb_supplementary.pdf",source / "dexycb_published_mano_diversity_plot")
    run("pdftoppm","-f","1","-l","1","-singlefile","-r","140","-png",source / "dexycb_cvpr2021.pdf",source / "dexycb_paper_overview_page")
    video = source / "dexycb_official_overview.mp4"
    duration = float(json.loads(run("ffprobe","-v","error","-show_entries","format=duration","-of","json",video))["format"]["duration"])
    for i,fraction in enumerate((.15,.5,.85)):
        target=source / f"dexycb_official_video_frame_{i+1:02d}.png"
        if not target.exists():
            run("ffmpeg","-nostdin","-v","error","-n","-ss",duration*fraction,"-i",video,"-frames:v","1","-threads","1",target)


def write_notes(root, info):
    caption = ("We retargeted 501 right-hand DexYCB captures from 10 subjects and 20 object identities "
        "into 62,892 six-actuator Revo2 configurations using power and precision optimization modes. "
        "The first five principal components explain 99.8081% of the variance in these robot joint configurations. "
        "The fitted PCA is frozen and used as a 5% soft prior during SAPG training; all six hand actions remain available.")
    table = "| Components retained | Individual variance | Cumulative variance |\n|---:|---:|---:|\n"
    cumulative=0.
    for i,v in enumerate(info["explained_variance_ratio"]):
        cumulative += v
        table += f"| {i+1} | {v*100:.4f}% | {cumulative*100:.4f}% |\n"
    (root / "SLIDE_TEXT.md").write_text(f"# Copy-ready slide caption\n\n{caption}\n\n"
        "Suggested title: **Five PCA vectors explain 99.81% of retargeted Revo2 motion variance.**\n\n"
        "The five vectors each have six joint coefficients; see `data/pca_vectors.csv`.\n\n"+table+
        "\nImportant qualification: four components already exceed 98% (99.2910%). Five were retained by design, not because five is the minimum to cross 98%.\n")
    dataset = f"""# Dataset subset and calculation

{caption}

## The actual subset

- The official release has 1,000 captures. We use all 501 captures whose metadata labels the acting hand as right; left-hand captures are excluded.
- All 10 subjects and all 20 object identities occur in this subset. Each subject contributes 50 captures except subject 08, which contributes 51 according to the release metadata.
- One camera is used per physical capture (the first available serial in sorted order), so the eight simultaneous views are not counted as eight demonstrations.
- Non-finite / missing hand labels are excluded. The loader retains the object-motion window with surrounding frames; the full run uses frame stride 1. The saved frame indices, rather than assumed video lengths, define the fitted data.
- There are 31,446 retained source frames. Each is optimized twice—power and precision—giving 1,002 robot-motion traces and 62,892 robot configurations. Power/precision are our optimization modes, not human grasp-class annotations supplied by DexYCB.
- Each sample is a vector of six independently actuated Revo2 joint angles in radians. Five additional distal joints follow the exact G1-integrated URDF mimic couplings and are not extra PCA variables.

## Retargeting and PCA

The five human fingertips are represented in a wrist/palm-fixed frame and scaled by α=0.90, calibrated for this Revo2 model. Offline Adam solves bounded IK with a human-imitation term, a closure term, and a posture regularizer. The human weight γ decreases from 1 to 0 along each trace. Thus the resulting data are human-inspired and closure-shaped, not an exact copy of raw human trajectories.

The fit uses centered robot joint configurations without per-joint standardization. If Q is the 62,892 × 6 configuration matrix, SVD of Q − mean(Q) yields the principal directions. Explained-variance fractions are squared singular values divided by their sum. The saved artifact contains a 5 × 6 matrix A and a six-angle mean. Its task map is x=Aq; affine reconstruction is q_hat=mean+(x−A mean)A in row-vector notation.

{table}

The variance values were recomputed from all saved configurations and agree with the frozen artifact. These are in-sample reconstruction statistics, not held-out performance, a measure of perceptual naturalness, or RL success rates. No DexYCB benchmark train/test split is claimed for this offline prior construction.

Mean per-trajectory PCA reconstruction RMSE: {info['mean_per_trajectory_projection_rmse_rad']:.8f} rad. Mean fingertip error at the pure-imitation endpoints (γ=1): {info['mean_imitation_endpoint_error_mm']:.4f} mm. The latter is not the error over all closure-biased frames.

## Current runtime use

SAPG predicts seven arm actions and six hand actions. A fixed 5% correction is applied once to each nominal hand target:

`q_target = q_nominal + 0.05 * (clip(project_PCA(q_nominal)) - q_nominal)`

Away from joint clipping, the five retained directions have gain 1 and the remaining direction has gain 0.95. This is a mild preference, not a hard five-action restriction. The policy can learn grasps outside the PCA subspace. Adam retargeting and PCA fitting are not run inside each RL environment. No training-speed or success-rate improvement is established by these visualizations; that requires a controlled ablation.

## Motion-media interpretation

- `01_human_to_revo2_pca.mp4`: matched source RGB frames beside their optimized Revo2 postures and frozen-PCA reconstructions. Coffee-can and scissors captures are fixed disclosed examples, not best-error selections. Frames are shown twice at 30 FPS, with one-second endpoint holds.
- `02_five_pca_motion_directions.mp4`: illustrative single-axis PCA sweeps. Other coordinates stay at the mean. Range is the empirical 5–95% interval intersected with joint-feasible bounds. PCA axes are not semantic grasp classes.
- `03_power_pinch_tripod_priors.mp4`: existing optional closure presets, projected by the full-run PCA. Pinch/tripod were not included in its fit and do not prove physically successful pinches.
- `04_actual_five_percent_prior.mp4`: synthetic omitted-direction sweep comparing no prior, the actual 5% setting, and diagnostic full projection. The deliberately small visual change at 5% is the real configured strength.

All robot media are validated kinematic replays/illustrations, not new physics simulations or policy rollouts. Joint limits and rendered fingertip FK are checked. Human and robot camera viewpoints are intentionally different: robot panels use a fixed palm-relative view.

Artifact SHA-256: `{info['artifact_sha256']}`.\nSource combined URDF SHA-256: `{info['source_urdf_sha256']}`.

Full capture IDs and counts are saved as CSV. The original run summary, validation, artifact, source indices and presentation measurements are included under `data/`.
"""
    (root / "DATASET_AND_PCA.md").write_text(dataset)
    (root / "SOURCES_AND_CREDITS.md").write_text("""# Sources and attribution

## Dataset and matched source images

Yu-Wei Chao et al. **DexYCB: A Benchmark for Capturing Hand Grasping of Objects.** CVPR 2021. NVIDIA and University of Washington.

- Project and dataset license: https://dex-ycb.github.io/
- Official paper: https://dex-ycb.github.io/assets/chao_cvpr2021.pdf
- Official supplement: https://dex-ycb.github.io/assets/chao_cvpr2021_supp.pdf
- Official website video: https://dex-ycb.github.io/assets/dex-ycb-render.mp4
- Toolkit: https://github.com/NVlabs/dex-ycb-toolkit
- Dataset license: https://creativecommons.org/licenses/by-nc/4.0/

Preserve attribution and the noncommercial condition when using dataset-derived images or media. This bundle does not grant new rights to the original papers; paper figures retain their publication attribution. The unchanged source PDFs are included for context. The website video is downloaded unchanged; three stills are extracted from it. `paper_images/` contains original raster images extracted from pages 1–2 of the paper, including its multi-view examples and capture apparatus.

**Reference-only plot:** `sources/dexycb_published_mano_diversity_plot.png` is Figure 2 on page 3 of the DexYCB supplement, rendered directly from the PDF with its caption. It compares MANO human-pose distributions. It is NOT our Revo2 PCA basis, NOT our 99.81% result, and NOT an RL-learning curve. The full source page is also retained.

Matched RGB image filenames correspond to the recorded source IDs in `data/measurements.json` and the saved motion NPZ files. The RGB photos are explanatory; PCA was fit to retargeted joint angles derived from the 3D labels, not to image pixels.

## Method inspiration

**DextrAH-G: Pixels-to-Action Dexterous Arm-Hand Grasping with Geometric Fabrics**, arXiv:2407.02274, Appendix D:
https://arxiv.org/html/2407.02274v1#A4

That work reports five components explaining approximately 98% for a 16-joint Allegro representation. Our six-actuator Revo2 fit is a different robot and dataset construction. Do not present its reported Allegro result as our measurement, or our current 5% soft prior as the paper's five-action policy interface.

## Our outputs

Figures under `figures/` are freshly generated from our saved artifact and robot-motion data, or explicitly labeled algebraic diagrams. Rendered robot geometry comes from the exact G1-integrated Revo2 URDF used during retargeting. Components are frozen throughout visualization. No training files or checkpoints were changed.
""")
    (root / "README.md").write_text("""# Revo2 PCA prior — research-meeting bundle

Open **index.html** for a local gallery, or **slide_figures.pdf** for the 16:9 figure deck.

Start with these assets:

1. `figures/01_explained_variance.png` — the 99.8081% result; also available as editable SVG and vector PDF.
2. `figures/06_dataset_subset.png` — exactly which part of DexYCB was used.
3. `videos/01_human_to_revo2_pca.mp4` — human RGB → offline Revo2 IK → PCA reconstruction.
4. `videos/02_five_pca_motion_directions.mp4` — what the five basis directions do.
5. `figures/05_soft_prior_not_action_restriction.png` — actual 5% SAPG integration, preserving six hand actions.

`SLIDE_TEXT.md` has a copy-ready caption and the exact variance table. `DATASET_AND_PCA.md` explains the subset, optimization, statistics, limitations and motion videos. `SOURCES_AND_CREDITS.md` provides attribution and separates published dataset figures from our own PCA results. CSVs and the full capture manifest are under `data/`.

All four new videos are 1920×1080 H.264 at 30 FPS. They are kinematic prior demonstrations, not trained-policy success recordings. PNGs are slide-ready; SVG/PDF versions preserve figure vectors and text. Original dataset website media and source PDFs are kept separately under `sources/`.
""")


def validate_and_index(root):
    validation = json.loads((root / "data" / "render_validation_all.json").read_text())
    assert validation["max_fk_error_m"] < 1.e-7
    for video in validation["videos"]:
        probe = json.loads(run("ffprobe","-v","error","-select_streams","v:0","-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration","-of","json",root / video["file"]))
        s=probe["streams"][0]
        assert (s["width"],s["height"],s["r_frame_rate"],int(s["nb_frames"])) == (1920,1080,"30/1",video["frames"])
        assert abs(float(probe["format"]["duration"])-video["seconds"]) < .002
        video["ffprobe"] = probe
    (root / "data" / "media_validation.json").write_text(json.dumps(validation,indent=2))
    run("pdfunite",*sorted((root / "figures").glob("*.pdf")),root / "slide_figures.pdf")
    cards=[]
    for path in sorted((root / "figures").glob("*.png")):
        rel=path.relative_to(root).as_posix(); name=path.stem.replace("_"," ")
        cards.append(f'<article><h3>{html.escape(name)}</h3><a href="{rel}"><img loading="lazy" src="{rel}"></a><p><a href="{rel[:-4]}.svg">SVG</a> · <a href="{rel[:-4]}.pdf">PDF</a></p></article>')
    for video in validation["videos"]:
        cards.append(f'<article><h3>{html.escape(Path(video["file"]).stem.replace("_"," "))}</h3><video controls preload="metadata" src="{video["file"]}"></video><p>{html.escape(video["description"])}</p></article>')
    for path in sorted((root / "stills").glob("*.png")):
        rel=path.relative_to(root).as_posix()
        cards.append(f'<article><h3>{html.escape(path.stem.replace("_"," "))}</h3><a href="{rel}"><img loading="lazy" src="{rel}"></a></article>')
    cards.append('<article><h3>Published dataset reference — not our PCA result</h3><img src="sources/dexycb_published_mano_diversity_plot.png"><p>Chao et al., DexYCB supplementary Figure 2: MANO human-pose diversity. Attribution and context in SOURCES_AND_CREDITS.md.</p></article>')
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Revo2 human-motion PCA prior</title><style>
    body{font:17px system-ui;background:#f3f6fa;color:#193348;margin:36px auto;max-width:1500px;padding:0 24px}
    h1{font-size:36px}a{color:#007f8b}.lead{background:white;border-left:5px solid #008f95;padding:20px;line-height:1.65}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px;margin-top:24px}
    article{background:white;border-radius:12px;padding:18px}img,video{width:100%}h3{font-size:19px}p{line-height:1.5}
    @media(max-width:700px){.grid{grid-template-columns:1fr}}</style></head><body>
    <h1>Human-motion PCA prior for G1 + Revo2</h1><div class="lead"><b>Five components retain 99.8081% of retargeted robot-motion variance.</b><br>
    501 right-hand captures · 10 subjects · 20 object identities · 31,446 source frames · 62,892 robot configurations.<br>
    Four components already retain 99.2910%; five were kept by design. Current SAPG uses a 5% correction, not a five-action restriction.<br>
    <a href="SLIDE_TEXT.md">Slide caption and exact table</a> · <a href="DATASET_AND_PCA.md">Dataset and method</a> ·
    <a href="SOURCES_AND_CREDITS.md">Sources and credits</a> · <a href="slide_figures.pdf">Figure deck PDF</a> ·
    <a href="data/explained_variance.csv">Variance CSV</a> · <a href="data/source_capture_manifest.csv">501-capture manifest</a></div><div class="grid">'''+"\n".join(cards)+"</div></body></html>"
    (root / "index.html").write_text(page)
    manifest=[dict(file=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(root.rglob("*")) if p.is_file() and p.name != "file_manifest.json"]
    (root / "file_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("PCA_BUNDLE_VALIDATED",len(manifest),"files",sum(m["bytes"] for m in manifest),"bytes",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("bundle",type=Path)
    root=parser.parse_args().bundle
    info=json.loads((root / "data" / "measurements.json").read_text())
    summary=json.loads((root / "data" / "summary.json").read_text())
    make_tables(root,summary,info)
    extra_figures(root,info)
    sources(root)
    write_notes(root,info)
    validate_and_index(root)


if __name__ == "__main__":
    main()
