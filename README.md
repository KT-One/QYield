# QYield

QYield is a CLI for **quantum-inspired wafer-defect novelty detection** on WM-811K. It combines a frozen ResNet50 backbone with a trainable **Adaptive State Injection (ASI)** feature head (`asi_L4`) to flag rare defect patterns outside the labelled training classes.

**71.5 novelty AUROC** (11-seed mean) on the episodic 3-way/5-shot open-set benchmark. This is **+5.3** over the strongest classical control, spectral-normalized ProtoNet, and **+10.4** over the no-head baseline. Closed-set accuracy remains near 78 percent across the main controls.

## This repository has two parts

| | **MVP (the product)** | **Research (the evidence)** |
|---|---|---|
| Where | [`src/qyield/`](src/qyield/) + `checkpoints/`, `data/`, `tests/` | [`research/`](research/) |
| What | An installable novelty classifier with a CLI and TUI (`uv run qyield ...`). The feature head runs in pure PyTorch. | The R&D record, including experiment code and the documentation behind each reported result. |
| For | **Running** the demo and classifying wafer maps. | **Reviewing** the head design and matched baselines. |

If you need the model evidence, start at **[`research/README.md`](research/README.md)**.
It separates the active `asi_L4` record from archived DP-QCNN studies.

## What it predicts

8 real single-defect classes from WM-811K:

| Class | Description |
|---|---|
| Center | Defective dies clustered in the wafer's center. |
| Edge-Ring | Defective dies forming a ring near the wafer's edge. |
| Edge-Loc | Defective dies localized along one edge section. |
| Donut | Defective dies forming a ring offset from center/edge. |
| Loc | Defective dies localized in a small region. |
| Near-full | Defective dies covering almost the entire wafer. |
| Random | Defective dies scattered with no pattern. |
| Scratch | Defective dies forming a thin line/scratch pattern. |

**Center / Edge-Ring / Edge-Loc** are "base" classes (seen during training).
**Donut / Loc / Near-full / Random / Scratch** are "novel" classes, held out entirely at
training time. With a small labelled support set, QYield classifies known defect types and flags unfamiliar patterns as novel. It is a **few-shot ProtoNet** classifier. Each prediction uses the bundled support set (`data/kset_k10_s42.npz`) to compute class prototypes at load time and returns a novelty score based on nearest-prototype distance.

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

**Download the model + data bundle first** (kept outside the repo, Google Drive, not
GitHub, since the backbone exceeds GitHub's 100MB per-file limit):

https://drive.google.com/drive/folders/123QXisWom9kE7yqI5ppuGs2jCEvH7J2G?usp=drive_link

Download `checkpoints.zip` and `data.zip`, then extract both into this repo's root (so you
end up with `checkpoints/` and `data/` alongside `src/`):

```bash
unzip checkpoints.zip     # -> checkpoints/asi_l4/{asi_l4.pt, resnet50_jet_backbone.pt}
unzip data.zip            # -> data/kset_k10_s42.npz (+ manifest)
```

Then:

```bash
uv sync
uv run qyield info      # confirms setup
```

**GPU and CPU:** QYield selects `cuda` when available and otherwise uses CPU. For a `CUDA error: no kernel image ...` message, pass `--device cpu` or install the CUDA build that matches your GPU from https://pytorch.org. The model contains one ResNet50 and an approximately 0.7 MB head, so CPU inference is practical.

## Usage

**Classify your own wafer map:**

```bash
uv run qyield predict /path/to/your_wafer.npy
```

Accepts a raw `.npy` array with values in `{0, 1, 2}` for blank, good die, and defective die, or a grayscale `.png` or `.jpg`. Supply the wafer at its **native resolution**. The CLI resizes it to 224x224 internally. Pre-resizing and re-quantizing removes information and reduces accuracy. Each result includes the predicted class, full ranking, and **novelty score**. Higher scores indicate a closer match to a known class.

**Want something to try right now?** `data/samples/` ships with ten real, held-out wafer
maps (2 per novel class) as PNGs, genuine query images, *not* from the support set:

```bash
uv run qyield predict data/samples/Scratch_01.png
uv run qyield predict data/samples/Donut_01.png --n-way 3 --k-shot 5
```

The filename tells you the true class so you can check the prediction.

**No image at all? Draw a demo query from our bundled support set:**

```bash
uv run qyield demo                        # random class
uv run qyield demo --true-class Scratch   # pick a specific class
```

**Reproduce the reported 3-way/5-shot regime** (by default, both commands classify against
all 8 classes using every bundled shot):

```bash
uv run qyield predict wafer.npy --n-way 3 --k-shot 5 --seed 42
uv run qyield predict wafer.npy --ways Donut Scratch Loc --k-shot 5
```

**Prefer a visual, interactive interface?**

```bash
uv run qyield tui
```

Launches a terminal UI (mouse + keyboard) with two modes:

- **Demo**, pick a class from our K-set and classify a sample.
- **Label your own**, a two-phase, few-shot workflow:
  - *Phase 1, label:* step through a handful of wafers. Each shows its true label as a
    hint, but you assign whatever class you like (including your own custom class names).
    These build a personal support set.
  - *Phase 2, classify:* held-out query wafers are classified using *only* the labels you
    assigned, and each result shows **Predicted vs Actual**.

  This workflow lets operators create a support set for new defect types from a handful of labelled examples, with no retraining step.

Wafer previews render as green/red ANSI art (green = good die, red = defect). Class names are
shown with friendly labels (e.g. `Near-full` → "Near-full failure"), the terse keys remain
what you type for `--ways`.

## How it works

- **Backbone:** a frozen ImageNet **ResNet50** with `jet` colormap preprocessing. It produces a 2,048-dimensional global-average-pooled embedding.
- **Head (`asi_L4`):** a depth-4 **Adaptive State Injection** photonic head (~164k params).
  The 2048-d embedding is split into 512 four-value blocks. For each block, ASI computes
  Born-rule routing weights from the current values, applies four learned norm-preserving
  expert transforms, and mixes their outputs using those weights. Different inputs produce
  different mixtures. The product computes this exact expectation in PyTorch. See
  [`research/docs/asi-l4-architecture.md`](research/docs/asi-l4-architecture.md) for the full data flow.
- **Classification:** Euclidean nearest-prototype (few-shot ProtoNet) against the bundled
  K-shot support set, the **novelty score** is the negative min-prototype distance.

The head uses a single-photon, **classically simulable** formulation and runs in **pure PyTorch**. Its contribution is a *quantum-inspired* feature map. See
[`research/docs/qyield-technical.md`](research/docs/qyield-technical.md) §7 for the scope.

## Repo layout

```
src/qyield/       CLI + inference code (cli.py, model_l4.py [asi_L4 model],
                  model.py [shared helpers], constants.py, tui.py, preview.py,
                  wafer_render.py)
checkpoints/      asi_l4/ (asi_l4.pt head + resnet50_jet_backbone.pt), from Google Drive, gitignored
data/samples/     10 real held-out wafer maps (PNG) to try the predict flow, tracked in git
data/             kset_k10_s42.npz + manifest (bundled K-shot support set), from Google Drive, gitignored
tests/            smoke_test.py, tui_smoke_test.py
research/         R&D artifacts (experiment code, self-documenting write-ups, base
                  papers), the evidence behind the pitch, see research/README.md
```

## Honesty notes

- The headline is a **fair, same-backbone novelty-AUROC** result. Every head uses identical frozen ResNet50 features, and the +5.3 and +10.4 gaps measure the ASI head contribution. Closed-set accuracy remains near 78 percent across the main controls.
- **71.5** is the 11-seed mean. The shipped checkpoint is a **single trained head (seed 42)**:
  closed-set acc **80.7**, novelty AUROC **69.4**, expect call-to-call variance, especially at
  low `--k-shot` (each call is one sampled episode).
- At single-photon scale the head is **classically simulable**. The demonstrated value is a **quantum-inspired** feature map. Multi-photon hardware remains a separate research direction.
- `qyield demo` draws its query from the same pool used for support-set prototypes, useful to
  confirm the pipeline works. It is a product smoke test.
