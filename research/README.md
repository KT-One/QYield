# QYield — Research & Development artifacts

This folder is the **evidence trail** behind the QYield pitch: the experiment code and
the self-documenting write-ups that produced every number on the slides. It is kept
**separate from the shipping product** (the CLI + TUI under [`../src/qyield/`](../src/qyield/)).

- **`../src/qyield/`** — the MVP: an installable, runnable wafer-defect classifier
  (`uv run qyield ...`). Pure-PyTorch photonic head, no research deps. Judges run this.
- **`research/`** (here) — the R&D: how the head was designed, every baseline it was
  measured against, and the recorded results. Judges read this.

> The results here are captured as **self-documenting markdown** (tables with per-model
> numbers, CIs, and provenance) under [`docs/`](docs/) — not raw run dumps. The code is a
> faithful copy of the research pipeline for review; see
> [Running the research code](#running-the-research-code) for what is and isn't
> self-contained. The MVP does **not** depend on any of it.

---

## The pitch → where it lives

### 1. QYield — the three model versions (`asi_p2`, `asi_p3`, `asi_L4`)

QYield is an **A**daptive **S**tate **I**njection (ASI) photonic head. The pitch shows
three versions along the depth/pooling-width progression; `asi_L4` is the one carried
forward as *the* QYield model.

| pitch name | id | structure | params | novelty AUROC (11 seeds) |
|---|---|---|---|---|
| **p2** | `asi_p2` | depth 1, pooling P=2 | 32,768 | 68.77 ± 0.85 |
| **p3** | `asi_p3` | depth 1, pooling P=3 | 40,960 | 69.21 ± 0.75 |
| **L4 (QYield)** | `asi_L4` | depth 4, pooling P=3 | 163,840 | **71.47 ± 0.95** |

- **Business framing:** [`docs/qyield-summary.md`](docs/qyield-summary.md) (problem,
  novelty-AUROC story, headline table).
- **Spec, math & full head landscape:** [`docs/qyield-technical.md`](docs/qyield-technical.md)
  — §2 per-layer ASI math, §3 why it works (norm-bounded + data-dependent gating), §8 the
  full head landscape including `asi_p2`/`asi_p3`/`asi_L4`.
- **Results write-up:** [`docs/results-novelty.md`](docs/results-novelty.md) — closed-set
  accuracy vs open-set novelty AUROC, per model, all three QYield versions.
- **Code:** [`src/dpqcnn/fswmpr/bench/qreg_bench.py`](src/dpqcnn/fswmpr/bench/qreg_bench.py)
  (defines the `asi` / `asi_deep` heads and the QReg bank) and
  [`src/dpqcnn/fswmpr/bench/novelty.py`](src/dpqcnn/fswmpr/bench/novelty.py)
  (the open-set novelty-AUROC benchmark that produced the headline).
- **Shipped checkpoint (product):** the MVP's `l4` model
  (`../src/qyield/model_l4.py`, run via `qyield predict`) is the depth-4
  head (`asi_deep`, seed 42) trained + exported by
  [`src/dpqcnn/fswmpr/bench/export_asi_l4.py`](src/dpqcnn/fswmpr/bench/export_asi_l4.py)
  onto a single frozen ImageNet ResNet50-jet backbone. That script trains the head
  (1000 updates, episodic prototype loss on the base classes), verifies a torchvision
  ResNet50 reproduces the research timm backbone **bit-for-bit** (so the MVP needs no
  `timm`), and saves `checkpoints/asi_l4/{asi_l4.pt, resnet50_jet_backbone.pt}`.
  Deployed seed-42 head: closed-set acc 80.7, novelty AUROC 69.4 (the 71.5 headline is
  the 11-seed mean). Run it in this research environment:
  `PYTHONPATH=src python -m dpqcnn.fswmpr.bench.export_asi_l4 --out-dir <QYield>/checkpoints/asi_l4`.

### 2. The classical versions we tested against

Every head sits on the **same frozen backbone → head → prototype classifier**, so any
gap is attributable to the head alone. This is the "we tried the classical way" story.

**(a) Different heads** — the control-head landscape ([`docs/qyield-technical.md`](docs/qyield-technical.md) §7–§8):

| head | class | code |
|---|---|---|
| ProtoNet (identity, no head) | baseline / floor | `bench/baselines.py` (`protonet_r50_frozen`) |
| `classical` (`zᵀWz` bilinear) | free quadratic | `bench/qreg_bench.py` |
| `SN-ProtoNet` (spectral-norm) | norm-preserving (strong classical control) | `bench/qreg_bench.py` |
| `orthogonal` | norm-preserving | `bench/qreg_bench.py` |
| `quantum` (non-adaptive photonic) | norm-preserving, no gating | `bench/qreg_bench.py` |
| `mlp_relu/gelu` (+ param-matched, residual) | free nonlinear | `bench/qreg_bench.py` |
| `dense1024` | free linear | `bench/qreg_bench.py` |

Full 11-seed numbers for every head: [`docs/results-novelty.md`](docs/results-novelty.md)
and [`docs/qyield-technical.md`](docs/qyield-technical.md) §8.

**(b) Different architectures / backbones** — the accuracy-and-SOTA campaign
([`docs/results-accuracy-sota.md`](docs/results-accuracy-sota.md)):

| line | backbone | code |
|---|---|---|
| **QConv4** | Conv4 (from scratch) | `bench/stageb_quantum.py` |
| **QResNet** | frozen ImageNet ResNet50 (grayscale/viridis/jet) | `bench/qresnet.py` |
| **QResNet-SSL** | in-domain SSL ResNet50 (SimCLR/Barlow/VICReg/SimSiam) | `bench/qresnet_ssl.py`, `qresnet_ssl2.py` |
| **QResNet-Ens** | top-3 SSL ensemble (6144-d) | `bench/qresnet_ens.py` |
| **QViT** | frozen ImageNet-MAE ViT-B/16 | `bench/stageb_quantum.py` (ViT path) |

Cost/compression analysis: [`docs/cost-runtime.md`](docs/cost-runtime.md).

**(c) Different SOTA anchors** — the literature bars we reproduced to have an honest,
external target (Liang et al. 2024, WM-811K few-shot: ProtoNet-Conv4 78.40,
ProtoNet-ResNet50 82.61 CNN-SOTA, MAE-ViT 84.40 overall-SOTA).

- Reproduction code: [`src/dpqcnn/fswmpr/bench/repro_c1.py`](src/dpqcnn/fswmpr/bench/repro_c1.py).
- The full anchored bar table is in [`docs/results-accuracy-sota.md`](docs/results-accuracy-sota.md) §1.

---

## Directory map

```
research/
├── README.md                 # this file — pitch → artifact index
├── docs/                     # self-documenting write-ups (results kept AS markdown)
│   ├── qyield-summary.md         # business pitch: problem, novelty-AUROC story, headline
│   ├── qyield-technical.md       # architecture, per-layer math, full head landscape (§8)
│   ├── results-novelty.md        # headline results: accuracy vs novelty AUROC, all heads
│   ├── results-accuracy-sota.md  # accuracy campaign: QConv4/QResNet*/QViT vs reproduced SOTA
│   └── cost-runtime.md           # cloud/on-prem cost & GPU-vs-QPU runtime analysis
├── src/dpqcnn/               # experiment source (faithful copy of the phase-3 pipeline)
│   ├── fswmpr/               # WM-811K few-shot wafer-map research (THE relevant code)
│   │   ├── bench/            # qreg_bench (asi heads), novelty (headline metric),
│   │   │                     #   baselines, repro_c1 (SOTA anchors), qresnet* (backbones),
│   │   │                     #   stageb_quantum (QConv4/QViT), build_kset, infer
│   │   ├── models/           # protonet, backbone, ssl_pretrain, multiphoton_core, dqproto, metrics
│   │   ├── data/             # preprocess (WM-811K), episodes, fscil, mixedwm38
│   │   └── constants.py
│   └── core/                 # low-level photonic primitives used by the multi-photon path
│                             #   (_baseline, blocks/MeshUnitary, ring, encoder, ...) — see deps note
└── papers/base/             # the two base papers QYield fuses (see below)
```

## Base papers

- **`papers/base/2504.20989v1.pdf`** — Monbroussou et al., *Photonic QCNN with Adaptive
  State Injection* (2025). Source of **ASI** (the "measure-then-choose" gadget) and the
  norm-preserving amplitude encoding.
- **`papers/base/2408.16327v1.pdf`** — Hwang et al., *Distributed QML via classical
  communication* (2024). Source of measurement-conditioned feedforward and the
  many-small-QPU decomposition.

How QYield fuses them: [`docs/qyield-technical.md`](docs/qyield-technical.md) §9.

## Running the research code

This code is provided as a **reviewable artifact**, not a second installable product.

- **Self-contained (external libs only — torch, numpy, timm, scikit-learn, matplotlib):**
  the classical + `n=1` ASI path. `bench/novelty.py`, `bench/baselines.py`, `bench/repro_c1.py`,
  `bench/qresnet*.py`, and everything under `models/` except the multi-photon core.
- **Needs the parent research environment** (`merlin` + `perceval` + a vendored photonic
  baseline): the **multi-photon (`n≥2`) Fock path** — `models/multiphoton_core.py`,
  `models/dqproto.py`, and the `...core._baseline` / `...core.blocks` imports.
  `src/dpqcnn/core/` is included so those imports resolve structurally, but `core/_baseline.py`
  itself pulls in `merlin` and a vendored monolith which are intentionally **not** copied here
  (that is the heavy baseline the MVP is separated from). The shipping QYield head is `n=1` and
  is reimplemented dependency-free in [`../src/qyield/model.py`](../src/qyield/model.py).

Reproduction entry points (in the parent env), for reference:

```bash
# headline novelty-AUROC landscape (produces asi_p2 / asi_p3 / asi_L4 + classical heads)
python -m dpqcnn.fswmpr.bench.novelty
# SOTA-anchor reproduction (Liang et al. 2024 bars)
python -m dpqcnn.fswmpr.bench.repro_c1
```

## Honest scope (carried from the write-ups)

- The headline is a **fair, same-backbone novelty-AUROC win** (+5.3 over the strongest
  classical control, +10.4 over the no-head floor; CIs separated). Closed-set accuracy is
  saturated (~78%) across all heads — it is reported only to confirm no regression.
- At `n=1` the head is **classically simulable**, so the advantage is **quantum-*inspired***
  (a better feature map), not a proven quantum-*hardware* moat. The hardware-moat regime is
  multi-photon (permanent-based) and is untested here — see
  [`docs/qyield-technical.md`](docs/qyield-technical.md) §10.
