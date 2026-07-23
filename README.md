# QYield

A CLI for quantum wafer-defect classification on WM-811K, built on the DP-QCNN
photonic-head architecture: a 3x SSL-pretrained ResNet50 ensemble feeding a photonic
(interferometer) head. **83.04% accuracy** (+/- 1.5%) on the episodic 3-way/5-shot benchmark,
beating the classical CNN-SOTA baseline (77.71%).

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
**Donut / Loc / Near-full / Random / Scratch** are "novel" classes, held out
entirely at training time — the headline 83.04% accuracy is measured only on
these 5, from just 5 labeled examples per class. That's the pitch: the model
recognizes rare/novel defect types it never trained on.

This is a **few-shot ProtoNet classifier**, not a fixed softmax model — every
prediction is a nearest-prototype lookup against a bundled 10-shot/class support
set (`data/kset_k10_s42.npz`), computed fresh at load time, not baked into the
checkpoint.

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

**Download the model + data bundle first** (kept outside the repo — Google Drive,
not GitHub, since the checkpoints exceed GitHub's 100MB per-file limit):

https://drive.google.com/drive/folders/123QXisWom9kE7yqI5ppuGs2jCEvH7J2G?usp=drive_link

Download `checkpoints.zip` and `data.zip`, then extract both into this repo's root
(so you end up with `checkpoints/` and `data/` alongside `src/`):

```bash
unzip checkpoints.zip
unzip data.zip
```

Then:

```bash
uv sync
uv run qyield info      # confirms setup
```

**GPU vs CPU:** auto-detects `cuda`, else falls back to CPU. The pinned CPU-safe
wheels may lack kernels for very new GPU architectures — if you hit
`CUDA error: no kernel image is available for execution on the device`, pass
`--device cpu`, or install the matching CUDA build for your GPU from
https://pytorch.org.

## Usage

**Classify your own wafer map:**

```bash
uv run qyield predict /path/to/your_wafer.npy
```

Accepts a raw `.npy` array with values in `{0, 1, 2}` (blank / good die /
defective die — recommended, native WM-811K format) or a grayscale `.png`/`.jpg`
(best-effort). Supply the wafer at its **native resolution** — the CLI resizes
it to 224x224 internally; do not pre-resize and re-quantize yourself, that
destroys information and measurably hurts accuracy.

**No data yet? Try a demo query from our bundled support set:**

```bash
uv run qyield demo                        # random class
uv run qyield demo --true-class Scratch   # pick a specific class
```

**Reproduce the reported 3-way/5-shot accuracy regime** (by default, both
commands classify against all 8 classes using every bundled shot):

```bash
uv run qyield predict wafer.npy --n-way 3 --k-shot 5 --seed 42
uv run qyield predict wafer.npy --ways Donut Scratch Loc --k-shot 5
```

## How it works

- **Backbone:** 3x ResNet50, each self-supervised-pretrained (SimCLR / Barlow
  Twins / VICReg) on WM-811K wafer maps, frozen. Per-block L2-normalized
  2048-d features are concatenated into a 6144-d embedding.
- **Head:** a photonic interferometer bank (2-photon, beamsplitter-mesh
  circuits) reshapes and reads out the embedding — this is the quantum
  component that provides the accuracy lift over a matched classical control.
- **Classification:** Euclidean nearest-prototype (few-shot ProtoNet), against
  the bundled K-shot support set.

Photonic circuits are simulated in pure PyTorch (no real quantum hardware
required); `merlinquantum`/`perceval-quandela` are only used once at
model-load time to extract the fixed circuit wiring schedule.

## Repo layout

```
src/qyield/       CLI + inference code (cli.py, model.py, constants.py)
checkpoints/      qresnet_ens/qresnet_ens.pt + stems/ (3 SSL backbones) — from Google Drive, gitignored
data/             kset_k10_s42.npz + manifest (bundled K-shot support set) — from Google Drive, gitignored
tests/            smoke_test.py
```

## Honesty notes

- Accuracy figures describe a 3-way/5-shot **episodic** protocol (few classes
  at a time, resampled support sets), averaged over 100 episodes x 6 seeds.
  A single CLI call is one sampled episode — expect call-to-call variance,
  especially at low `--k-shot`.
- The default (no `--n-way`/`--k-shot`/`--ways`) fixed 8-way classification is
  a different, not-directly-comparable task to the reported episodic number.
- `qyield demo` draws its query from the same pool used for support-set
  prototypes — useful to confirm the pipeline works, not a benchmark result.
