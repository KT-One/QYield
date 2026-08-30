# QYield research

This directory documents the shipped QYield model and the experiments that support its claim.

**Shipped model:** `asi_L4` is a single-photon Adaptive State Injection head for open-set wafer-defect novelty detection. The CLI and TUI product lives in [`../src/qyield/`](../src/qyield/) and runs in pure PyTorch.

## Active QYield documentation

| Document | Purpose |
|---|---|
| [`docs/qyield-summary.md`](docs/qyield-summary.md) | Plain-language product and research summary. |
| [`docs/asi-l4-architecture.md`](docs/asi-l4-architecture.md) | Self-contained data flow and layer-by-layer `asi_L4` architecture. |
| [`docs/qyield-physics.md`](docs/qyield-physics.md) | Physical ASI primer first, then QYield's quantum-inspired adaptation and boundary. |
| [`docs/qyield-technical.md`](docs/qyield-technical.md) | Training protocol, controls, result interpretation, and scope. |
| [`docs/results-novelty.md`](docs/results-novelty.md) | Final 11-seed novelty results for `asi_L4` and matched controls. |

## What the product ships

- Frozen ImageNet ResNet50 with `jet` preprocessing.
- Depth-4 `asi_L4` head with 163,840 trainable parameters.
- Euclidean prototype classifier and a nearest-prototype novelty score.
- Checkpoint files in `checkpoints/asi_l4/`.

The shipped checkpoint is one trained seed. The headline result is the 11-seed evaluation mean. See the product README for both values.

## Research code

The `n=1` ASI path produced the QYield evidence:

- `src/dpqcnn/fswmpr/bench/qreg_bench.py`: ASI and control-head definitions.
- `src/dpqcnn/fswmpr/bench/novelty.py`: novelty-AUROC evaluation.
- `src/dpqcnn/fswmpr/bench/export_asi_l4.py`: checkpoint export and compatibility verification.
- `src/dpqcnn/fswmpr/data/`: WM-811K preprocessing and episodic sampling.

The product implementation has its own runtime layout:

- `../src/qyield/model_l4.py`: inference model.
- `../src/qyield/cli.py`: CLI.
- `../src/qyield/tui.py`: interactive few-shot workflow.

## DP-QCNN lineage and research boundary

QYield grew from the broader **Distributed Photonic Quantum Convolutional Neural Network (DP-QCNN)** research program. DP-QCNN connected two source ideas:

- Monbroussou et al., ["Photonic Quantum Convolutional Neural Networks with Adaptive State Injection"](papers/base/2504.20989v1.pdf), provides the photonic QCNN and physical ASI line.
- Hwang et al., ["Distributed quantum machine learning via classical communication"](papers/base/2408.16327v1.pdf), provides measurement-conditioned classical feedforward between quantum processors.

The parent program explored small photonic processors connected through adaptive classical communication. QYield carries forward the single-photon (`n=1`), depth-4 `asi_L4` feature head for wafer novelty detection. Product execution computes Born-weighted conditional transforms in PyTorch. Distributed DP-QCNN architecture, cross-QPU communication, physical state injection, and quantum hardware execution remain research-program capabilities.

Closed-set accuracy, multi-photon, hardware-cost, and compression studies live only in the canonical DP-QCNN repository:

- `/home/tam/Research/DP-QCNN/.ktcode/reports/rep-03_06/RESULTS.md`
- `/home/tam/Research/DP-QCNN/.ktcode/reports/rep-03_07/cost-runtime.md`

These reports cover QConv4, QResNet, QResNet-SSL, QResNet-Ens, and QViT. They document separate research models. QYield product performance, cost, runtime, and hardware results are reported in this repository.

The QYield repository intentionally contains no local copy of these DP-QCNN reports.

## Scope

QYield has a same-backbone novelty-detection result on WM-811K. Its validated scope is wafer-defect novelty detection. Accuracy SOTA, quantum hardware speedup, quantum computational advantage, and other domains require separate evidence.

The head is single-photon and classically simulable. The current claim is a structured, quantum-inspired photonic feature map that improves the tested novelty metric.
