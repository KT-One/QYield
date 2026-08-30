# Adaptive State Injection physics and QYield

Read this note in two parts:

1. **Physical Adaptive State Injection (ASI):** the detector, feedforward, and fresh-photon-injection mechanism described in the source paper.
2. **QYield:** a quantum-inspired software adaptation that uses mode mixing and Born-routing ideas in a PyTorch feature head.

The physical explanation comes first because it explains what ASI means in photonics. The QYield section then states exactly which parts the shipped model retains.

## 1. Physical Adaptive State Injection

The physical ASI mechanism comes from L. Monbroussou et al., ["Photonic Quantum Convolutional Neural Networks with Adaptive State Injection"](../papers/base/2504.20989v1.pdf), arXiv:2504.20989v1, 2025.

The paper's photonic PQCNN uses:

```text
single-photon sources
+ linear-optical interferometers
+ photon detectors
+ classical feedforward
+ conditional fresh-photon injection
```

Physical ASI uses a real detector outcome to control a later operation in the optical circuit.

### 1.1 One photon over several modes

A photon can occupy a coherent superposition of optical modes. For a simple four-mode register:

```text
|psi> = v0|1000> + v1|0100> + v2|0010> + v3|0001>
```

The basis state `|1000>` means one photon in mode 0 and vacuum in the other three modes. The coefficients are amplitudes, so their squared magnitudes give ideal detection probabilities:

```text
P(mode i) = |v_i|^2
sum_i |v_i|^2 = 1
```

Four modes can be physical waveguides, spatial paths, time bins, or another orthogonal mode basis. They are not four photons.

### 1.2 Linear-optical processing before measurement

A configurable interferometer, built from components such as beam splitters and phase shifters, mixes the mode amplitudes:

```text
a = Uv
```

For one photon, a lossless interferometer preserves total probability:

```text
sum_i |a_i|^2 = 1
```

This is the physical origin of the mode-mixing language used later in QYield.

### 1.3 A four-mode to two-mode ASI pooling example

The paper's pooling operation measures selected modes and retains others. A simple one-register, four-mode illustration is:

```text
M0  -> detector D0
R0  -> retained mode R0
M1  -> detector D1
R1  -> retained mode R1
```

`M0` and `M1` are measured modes. `R0` and `R1` are retained modes. This is one possible layout for explaining a four-mode register that is pooled down to two retained modes.

Because there is one photon in this register, the physical outcomes are mutually exclusive:

| Detector outcome | Meaning | Adaptive action |
|---|---|---|
| No click | The photon was already in a retained mode | Let the existing photon continue through `R0` and `R1` |
| `D0` click | The measured photon was consumed in `M0` | Inject a fresh photon into `R0` |
| `D1` click | The measured photon was consumed in `M1` | Inject a fresh photon into `R1` |

The output is again a one-photon state, but now over two retained modes. The injection step restores the required one-photon structure after a detector consumed a photon in a measured mode.

This is the physical intuition behind “measure, then conditionally inject.” The mechanism is a measurement-and-feedforward operation.

### 1.4 What happens in repeated shots

One hardware run, or **shot**, produces one detector outcome. Repeating the same prepared input many times estimates its outcome distribution:

```text
P(D0 click) approximately count(D0 clicks) / shots
P(D1 click) approximately count(D1 clicks) / shots
P(no click) approximately count(no clicks) / shots
```

The detector outcome drives a switch or controller that selects the correct physical action for that shot. A physical ASI circuit therefore follows one branch per shot.

The paper describes detector-controlled switching and conditional injection as the target adaptive hardware mechanism. Its reported experimental implementation emulated this adaptivity through postselection and separately configured circuit cases because coherent delay lines and fast feedforward switching were not available in that apparatus.

### 1.5 Why this is nonlinear and adaptive

A fixed interferometer applies one linear transformation to every input. ASI adds a measurement result and a later operation conditioned on that result:

```text
interferometer
→ detector result
→ classical feedforward
→ inject a fresh photon or preserve the existing photon
→ downstream optical circuit
```

The detector result is probabilistic for an individual shot, but its distribution depends on the input amplitudes. This gives the circuit adaptive, input-dependent behavior.

### 1.6 Paper scope

The source paper demonstrates a photonic PQCNN on 8-mode and 12-mode integrated interferometers. It uses multiple registers and photons for its tensor-encoded image setting. The four-mode, one-photon example above is a simplified teaching model for one register. The paper's full experimental layout uses 8-mode and 12-mode integrated interferometers with multiple registers and photons.

## 2. From physical ASI to QYield

QYield translates the physical ASI idea into deterministic PyTorch operations.

The shipped product uses a real four-value state per block, learned norm-preserving transforms, and Born-derived routing weights. It computes them deterministically in PyTorch.

```text
QYield product runtime:
- pure PyTorch tensor operations
- deterministic branch evaluation
- conventional CPU or GPU execution
```

### 2.1 What QYield keeps

QYield keeps three mathematical ideas motivated by photonics:

| Physical ASI idea | QYield adaptation |
|---|---|
| One photon distributed across modes | A normalized real four-dimensional feature state |
| Interferometer mixes mode amplitudes | Learned 4 x 4 orthogonal transform |
| Born rule gives outcome probabilities | Squared amplitudes become internal routing weights |
| Later action depends on an outcome | Input-dependent weighting of learned transforms |

QYield is a **quantum-inspired** feature map implemented as software.

### 2.2 What QYield changes

In a physical ASI circuit, a detector selects one branch on one shot. The detector consumes a measured photon, and a physical architecture may inject a fresh photon before later processing.

QYield evaluates all four learned transforms and combines them with Born-derived weights. The computation uses ordinary PyTorch tensor algebra.

The current routing has four software outcomes: three explicit squared amplitudes and one remainder outcome. This model-routing design differs from the paper's physical four-mode-to-two-mode pooling layout.

A useful shorthand is:

```text
physical ASI: measure one branch → conditionally inject or continue
QYield: compute all routing weights → evaluate all learned transforms → weighted combination
```

### 2.3 Why hardware shots would not reproduce the shipped forward pass

A physical shot-based ASI circuit creates a classical mixture of sampled detector branches. The shipped QYield model combines branch amplitudes before its final square readout.

Those operations are generally different:

```text
physical branch mixture:  sum_i g_i |y_i|^2
QYield amplitude mixture: |sum_i g_i y_i|^2
```

Many physical shots would estimate a related physical model. The current `asi_L4` forward pass uses a different amplitude-combination operation.

## 3. Where to find QYield implementation details

This note explains the physical ASI concept first. For QYield-specific details, use:

- [`asi-l4-architecture.md`](asi-l4-architecture.md): exact `asi_L4` equations, four routing outcomes, expert transforms, tensor shapes, and parameter counts.
- [`qyield-technical.md`](qyield-technical.md): training protocol, controls, result interpretation, and product scope.
- [`../papers/base/2504.20989v1.pdf`](../papers/base/2504.20989v1.pdf): physical photonic PQCNN and ASI source paper.
- [`../papers/base/2408.16327v1.pdf`](../papers/base/2408.16327v1.pdf): measurement-conditioned processing through classical communication.

## 4. Accurate summary

> Physical ASI measures selected optical modes, uses the detector result to control later action, and conditionally injects fresh photons to preserve the required photonic state structure. QYield is a pure-PyTorch, quantum-inspired adaptation that uses four-mode state geometry and Born-derived routing weights through deterministic tensor operations.
