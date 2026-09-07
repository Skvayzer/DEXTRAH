# Revo2 PCA prior: saved research-meeting material

The complete local/server bundle is `outputs/revo2-pca-research-meeting-20260907/`.
Open `index.html` for the gallery or `slide_figures.pdf` for the 16:9 figure deck.
The variance plot is `figures/01_explained_variance.{png,svg,pdf}` and the
dataset-summary figure is `figures/06_dataset_subset.{png,svg,pdf}`.

## Exact result and copy-ready explanation

We retargeted 501 right-hand DexYCB captures from 10 subjects and 20 object
identities into 62,892 six-actuator Revo2 configurations using power and
precision optimization modes. The first five principal components explain
**99.8081% of the variance in these robot joint configurations**. The fitted
PCA is frozen and used as a 5% soft prior during SAPG training; all six hand
actions remain available.

| Component | Individual variance | Cumulative variance |
| --- | ---: | ---: |
| 1 | 84.9633% | 84.9633% |
| 2 | 10.0767% | 95.0400% |
| 3 | 2.8076% | 97.8475% |
| 4 | 1.4435% | 99.2910% |
| 5 | 0.5170% | 99.8081% |
| 6 | 0.1919% | 100.0000% |

Four components already exceed 98%; five were retained by design, not because
five is the minimum needed to cross 98%. These are in-sample variance figures
for optimized robot angles, not raw human 3D motion, held-out reconstruction
accuracy, or a grasp-success rate. The approximately 98% result for Allegro in
DextrAH-G is a different result on a different robot.

## Which part of DexYCB?

- All 501 captures marked right-handed in the downloaded official release;
  left-hand captures are excluded. This is not a named DexYCB benchmark split.
- All 10 subjects: 50 captures each except subject 08, which has 51 right-hand
  captures in the release metadata. All 20 object identities are represented.
- One available camera per physical capture, selected by sorted serial number;
  the same movement is not duplicated across its eight simultaneous views.
- The saved trajectories contain 31,446 retained source frames after invalid
  label filtering and the object-motion-window selection. Frame stride is 1.
- Each retained source frame is optimized in two modes (power and precision),
  producing 1,002 traces and 62,892 six-angle robot configurations. These modes
  are retargeting objectives, not human grasp-class labels from the dataset.
- The human points are palm-relative, with five fingertip targets scaled by
  alpha 0.90. Offline Adam blends imitation with closure and posture terms;
  PCA is then fit to centered robot angles without per-joint standardization.
- The additional pinch/tripod inspection presets are not part of the PCA fit.

Capture IDs, frame counts, subject/object counts, all five six-entry PCA
vectors, and exact variance values are saved as CSV under the bundle's `data/`.
`SLIDE_TEXT.md` and `DATASET_AND_PCA.md` contain a reusable caption and details.

## Included motion videos

All four generated videos are 1920×1080, 30 FPS, H.264, and kinematic—not RL
rollouts or physical manipulation demonstrations:

1. Matched human RGB / Revo2 IK / PCA reconstruction: 25.6 seconds.
2. Five separate, joint-feasible PCA direction sweeps: 30 seconds.
3. Power, two-finger pinch, and tripod closure presets: 8.3 seconds.
4. Nominal / actual 5% correction / diagnostic 100% projection: 10 seconds.

The website's original DexYCB overview video, selected source RGB images,
paper images, published MANO-diversity plot, original paper and supplement are
under `sources/`, with attribution. That published MANO plot is explicitly
separated from our Revo2 PCA result. Dataset content is CC BY-NC 4.0; paper
figures retain their original publication attribution.

## Validation and reproduction

The variance was independently recomputed from all 62,892 saved configurations
and matches the frozen artifact. The presentation blend agrees numerically
with the runtime `soft_prior` implementation. Across 3,417 rendered hand poses,
maximum fingertip FK disagreement is 4.17e-17 m, with joint-limit checks.
FFprobe verifies all frame counts, durations and formats. Twenty-one PCA,
retargeting-data and presentation tests pass. These checks establish faithful
visualization, not physical task success or training-speed improvement.

Frozen artifact SHA-256:
`8cea2fe7602958bbefec826fd331a100c15145c7dd837c68406a07895a2237b3`.

Run these sequentially through the user's Slurm allocation, with a fresh output
directory. They do not change the artifact, checkpoints, or training:

```bash
python scripts/prepare_pca_presentation.py \
  --run /path/to/revo2-g1-full-v1 \
  --presets /path/to/revo2-g1-motion-presets-v1 \
  --dataset /path/to/dex-ycb/data \
  --hand-urdf /path/to/extracted-g1-revo2-hand.urdf \
  --output /path/to/new-bundle
python scripts/render_pca_presentation.py /path/to/new-bundle
python scripts/finish_pca_presentation.py /path/to/new-bundle
```

Sources: [DexYCB project](https://dex-ycb.github.io/),
[DexYCB paper](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf),
[DexYCB supplement](https://dex-ycb.github.io/assets/chao_cvpr2021_supp.pdf),
[DextrAH-G Appendix D](https://arxiv.org/html/2407.02274v1#A4).
