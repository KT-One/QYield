# QYield

A CLI for **quantum wafer-defect novelty detection** on WM-811K, built on the DP-QCNN
photonic-head architecture: a frozen ResNet50 backbone feeding a trainable **Adaptive
State Injection (ASI)** photonic head (`asi_L4`). It flags *rare, never-seen* defect
patterns — the failure modes a classifier trained on known defects silently misses.

**71.5 novelty AUROC** (11-seed mean) on the episodic 3-way/5-shot open-set benchmark —
**+5.3** over the strongest classical control (spectral-normalized ProtoNet) and **+10.4**
over the no-head baseline, with closed-set accuracy held (~78%, tied — accuracy is not the
story here; novelty is).

## This repository has two parts

| | **MVP (the product)** | **Research (the evidence)** |
|---|---|---|
| Where | [`src/qyield/`](src/qyield/) + `checkpoints/`, `data/`, `tests/` | [`research/`](research/) |
| What | An installable, runnable novelty classifier — CLI + TUI (`uv run qyield ...`). Pure-PyTorch photonic head, no quantum-sim deps. | The full R&D trail: experiment code and self-documenting write-ups behind every pitched number. |
| For | **Running** — try the demo, classify a wafer. | **Reading** — how the head was designed and every baseline it beat. |

If you're here to **use** QYield, everything you need is below. If you're here to **review
the science** (the three QYield versions `p2`/`p3`/`L4`, and the classical
heads/architectures/backbones/SOTA anchors we tested against), start at
**[`research/README.md`](research/README.md)**.

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
training time. The pitch: from just a handful of labeled examples, QYield recognizes — and
**flags as novel** — defect types it never trained on. It's a **few-shot ProtoNet**
classifier: every prediction is a nearest-prototype lookup against a bundled support set
(`data/kset_k10_s42.npz`), computed fresh at load time, plus a **novelty score** (how
unlike any known class the query is).

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

**Download the model + data bundle first** (kept outside the repo — Google Drive, not
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

**GPU vs CPU:** auto-detects `cuda`, else falls back to CPU. The pinned CPU-safe wheels may
lack kernels for very new GPU architectures — if you hit `CUDA error: no kernel image ...`,
pass `--device cpu`, or install the matching CUDA build for your GPU from
https://pytorch.org. The model is small (one ResNet50 + a ~0.7 MB head) and runs comfortably
on CPU.

## Usage

**Classify your own wafer map:**

```bash
uv run qyield predict /path/to/your_wafer.npy
```

Accepts a raw `.npy` array with values in `{0, 1, 2}` (blank / good die / defective die —
recommended, native WM-811K format) or a grayscale `.png`/`.jpg`. Supply the wafer at its
**native resolution** — the CLI resizes it to 224x224 internally; do not pre-resize and
re-quantize yourself, that destroys information and measurably hurts accuracy. Each result
prints the predicted class, a full ranking, and a **novelty score** (higher = more like a
known class; lower = more likely a novel/unseen defect).

**Want something to try right now?** `data/samples/` ships with ten real, held-out wafer
maps (2 per novel class) as PNGs — genuine query images, *not* from the support set:

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

- **Demo** — pick a class from our K-set and classify a sample.
- **Label your own** — a two-phase, few-shot workflow:
  - *Phase 1 — label:* step through a handful of wafers. Each shows its true label as a
    hint, but you assign whatever class you like (including your own custom class names).
    These build a personal support set.
  - *Phase 2 — classify:* held-out query wafers are classified using *only* the labels you
    assigned, and each result shows **Predicted vs Actual**.

  This is the few-shot story made interactive — teach the model new defect types from a
  handful of examples, no retraining.

Wafer previews render as green/red ANSI art (green = good die, red = defect). Class names are
shown with friendly labels (e.g. `Near-full` → "Near-full failure"); the terse keys remain
what you type for `--ways`.

## How it works

- **Backbone:** a single frozen ImageNet **ResNet50** (`jet` colormap preprocessing),
  producing a 2048-d global-average-pooled embedding. Frozen — no fine-tuning.
- **Head (`asi_L4`):** a depth-4 **Adaptive State Injection** photonic head (~164k params).
  The 2048-d embedding is split across 512 tiny photonic circuits (QPUs); each does a learned
  norm-preserving rotation, then the ASI step — it *measures* part of the state and lets that
  outcome *choose* which learned circuit processes the rest (a data-dependent branch). All
  transforms are norm-preserving, so the head adds routing capacity **without** collapsing the
  feature geometry — which is exactly what improves open-set novelty detection.
- **Classification:** Euclidean nearest-prototype (few-shot ProtoNet) against the bundled
  K-shot support set; the **novelty score** is the negative min-prototype distance.

The head is single-photon and **classically simulable**, so it runs in **pure PyTorch** — no
`merlin`/`perceval`/quantum hardware required. The advantage is *quantum-inspired* (a better,
collapse-resistant feature map), not a hardware speedup; see
[`research/docs/qyield-technical.md`](research/docs/qyield-technical.md) §10 for the honest scope.

## Repo layout

```
src/qyield/       CLI + inference code (cli.py, model_l4.py [asi_L4 model],
                  model.py [shared helpers], constants.py, tui.py, preview.py,
                  wafer_render.py)
checkpoints/      asi_l4/ (asi_l4.pt head + resnet50_jet_backbone.pt) — from Google Drive, gitignored
data/samples/     10 real held-out wafer maps (PNG) to try the predict flow — tracked in git
data/             kset_k10_s42.npz + manifest (bundled K-shot support set) — from Google Drive, gitignored
tests/            smoke_test.py, tui_smoke_test.py
research/         R&D artifacts (experiment code, self-documenting write-ups, base
                  papers) — the evidence behind the pitch; see research/README.md
```

## Honesty notes

- The headline is a **fair, same-backbone novelty-AUROC** result (every head compared on
  identical frozen ResNet50 features; the +5.3/+10.4 gaps are the ASI head alone, CIs
  separated). Closed-set **accuracy is saturated** (~78%) across all heads — it's reported
  only to confirm no regression, not as the differentiator.
- **71.5** is the 11-seed mean. The shipped checkpoint is a **single trained head (seed 42)**:
  closed-set acc **80.7**, novelty AUROC **69.4** — expect call-to-call variance, especially at
  low `--k-shot` (each call is one sampled episode).
- At single-photon scale the head is **classically simulable**, so the advantage is
  **quantum-*inspired*** (a better feature map), not a proven quantum *hardware* moat — the
  hardware-moat regime is multi-photon and untested here.
- `qyield demo` draws its query from the same pool used for support-set prototypes — useful to
  confirm the pipeline works, not a benchmark result.
