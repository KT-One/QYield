# QYield technical overview

The exact layer sequence, tensor shapes, parameter count, and product-to-research source mapping are in [`asi-l4-architecture.md`](asi-l4-architecture.md).

## 1. System

QYield processes a wafer map with a frozen ImageNet ResNet50 and `jet` colormap preprocessing. The backbone returns a 2,048-dimensional global-average-pooled embedding `e`.

`asi_L4` reshapes `e` into 512 blocks of four values. It applies a depth-4 Adaptive State Injection head to each block. The head returns two squared-amplitude features per block, producing a 1,024-dimensional feature vector.

A ProtoNet classifier computes one Euclidean prototype per support-set class. The predicted class is the nearest prototype. The novelty score is the negative distance to that prototype.

## 2. ASI layer

For one four-dimensional block `v`, each layer applies:

```text
v <- U_l v
p_o <- v_o^2 for o in {1, 2, 3}
p_rest <- 1 - sum(p_o)
v <- sum_o p_o V_l^(o) v
v <- v / ||v||
```

`U_l` and each `V_l^(o)` are learned orthogonal maps:

```text
U_l     = exp(A_l - A_l^T)
V_l^(o) = exp(B_l^(o) - B_l^(o)^T)
```

The three observed modes plus the remainder produce four gate weights. The output is a weighted mixture of four expert transforms. The weights depend on the current input state.

The model has 163,840 trainable head parameters:

```text
512 blocks * 4 layers * 4 * 4     = 32,768 rotation parameters
512 blocks * 4 layers * 4 experts * 4 * 4 = 131,072 expert parameters
Total = 163,840
```

The skew-symmetric parameterisation makes each learned transform orthogonal, and the expert mixture is renormalised. This creates a constrained alternative to an unconstrained MLP. The novelty result is empirical.

## 3. Photonic interpretation

A normalised four-value block can be read as the amplitude of one photon across four modes. The orthogonal maps correspond to linear-optical interferometers. Squared amplitudes are Born probabilities.

The implementation uses the exact Born-weighted expectation over gate outcomes. Product execution evaluates every branch in PyTorch without detector sampling, photon injection, or quantum-state exchange between chips.

At one photon, the computation is small real-matrix algebra that runs in pure PyTorch. Multi-photon permanent-based simulation remains in the wider DP-QCNN research program.

## 4. Training

Only the ASI head is trained. The ResNet50 backbone is frozen.

- Optimiser: Adam, learning rate `1e-3`.
- Objective: episodic prototype loss, `CE(-cdist(query, class_prototypes))`.
- Training: 1,000 updates with meta-batch size 16.
- Input task: 3 base WM-811K classes.

The checkpoint export script is [`../src/dpqcnn/fswmpr/bench/export_asi_l4.py`](../src/dpqcnn/fswmpr/bench/export_asi_l4.py). It exports the head and a compatible torchvision ResNet50 backbone for the product bundle.

## 5. Evaluation

| Item | Setting |
|---|---|
| Dataset | WM-811K FSWMPR few-shot split |
| Base classes | Center, Edge-Ring, Edge-Loc |
| Held-out novel classes | Donut, Loc, Near-full, Random, Scratch |
| Episode | 3-way, 5-shot |
| Support per episode | 15 labelled selections, 3 classes x 5 shots |
| Known-class queries per episode | 60, 3 classes x 20 queries |
| Held-out novelty queries per episode | 20 |
| Evaluation | 1,100 episodes, 100 per seed across 11 seeds |
| Aggregate observations | 16,500 support selections and 88,000 query observations |
| Classifier | Euclidean nearest prototype |
| Novelty score | Negative nearest-prototype distance |
| Intervals | 95% confidence intervals |

The 88,000 query observations comprise 66,000 known-class and 22,000 held-out novelty queries. Source wafer maps can recur in different episodes, so these are sampled observations rather than a unique-image count.

The primary metric is novelty AUROC. Closed-set accuracy is a secondary check.

## 6. Controls and result

Every headline control uses the same frozen ResNet50 features, classifier, episode sampler, and novelty scorer. Only the head differs.

| Model | Constraint or structure | Closed-set accuracy | Novelty AUROC |
|---|---|---:|---:|
| ProtoNet-ResNet50 | no head | 76.15 +/- 0.67 | 61.12 +/- 0.97 |
| `classical` | unconstrained quadratic | 76.39 +/- 0.87 | 65.37 +/- 0.97 |
| `orthogonal` | orthogonal map | 77.78 +/- 0.65 | 66.00 +/- 0.88 |
| SN-ProtoNet | spectral-normalised classical map | 77.40 +/- 0.74 | 66.19 +/- 0.84 |
| `multi_photon` | non-adaptive multi-photon photonic-simulation control | 78.39 +/- 0.76 | 66.26 +/- 1.01 |
| `asi_p2` | adaptive ASI, depth 1, P=2 | 77.92 +/- 0.68 | 68.77 +/- 0.85 |
| `asi_p3` | adaptive ASI, depth 1, P=3 | 77.92 +/- 0.69 | 69.21 +/- 0.75 |
| **`asi_L4`** | **adaptive ASI, depth 4, P=3** | **78.47 +/- 0.75** | **71.47 +/- 0.95** |

`asi_L4` improves over the strongest classical control in this evaluation by 5.28 AUROC points. Closed-set accuracy ranges from 76.15 to 78.47 percent across these matched controls, with `asi_L4` at 78.47 percent. This supports competitive closed-set classification without making an accuracy SOTA claim. The novelty result supports the hypothesis that constrained, input-dependent routing preserves useful novelty structure better than the tested unconstrained heads.

The full result table is in [`results-novelty.md`](results-novelty.md).

## 7. Scope

Established:

- A same-backbone, same-protocol novelty-AUROC result on the WM-811K split.
- A measured advantage over the tested matched classical controls.
- A runnable pure-PyTorch product checkpoint and CLI/TUI.

Not established:

- Accuracy SOTA evidence.
- Generalisation evidence outside the tested wafer setting.
- Quantum hardware, energy, latency, and computational advantage evidence.
- Results from a multi-photon hardware implementation.

The base papers are in [`../papers/base/`](../papers/base/). They motivate the ASI and measurement-conditioned routing ideas. QYield implements a single-photon, classically simulated form of those ideas.
