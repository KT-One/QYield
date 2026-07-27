# Cost & runtime analysis — QResNet mainlines (SSL / Ens / SSL-Light), QConv4 & the monolithic baseline

**Model line-up & device/cost comparison is in §0** (Table 1: mainlines vs monolithic; Table 2:
QResNet-SSL-Light merge variants). §1–5 detail cloud pricing, on-prem hardware, and runtime.

Models: **QResNet-ensemble** (3× ResNet50 SSL ensemble, ≈25 GFLOP/image classical + 768-QPU photonic head,
3w5s **83.04 > CNN SOTA 82.61**) and **QConv4** (Conv4 + 3,136-QPU photonic head, 3w5s **81.20 > matched
classical +5.8**). **We are PHOTONIC (linear-optical)** → only photonic hardware (Quandela/Xanadu/ORCA) runs
it natively; gate-based QPUs are $-references only. **TL;DR:** 2-photon/≤9-mode circuits are classically
**simulable**, so both models deploy on a **$8.5k GPU** (train ~$4, serve sub-cent/ms). Real photonics is
optional and, per §5, ~1,000–10,000× slower.

**Shot budget:** recommend **S ≈ 5,000/QPU** (occupation SE ≈ 1%; > IonQ's 2,500-shot floor). Exact S needed
to preserve 83.04/81.20 is unmeasured → **shot-noise study = first item in the compression R&D plan.** All figures below
use S=5,000.

---

## 0. Model line-up — accuracy, photonic device counts & cloud cost

**Device counting rule:** a universal M-mode interferometer = M(M−1)/2 MZIs; **each MZI = 2 beam splitters
+ 2 phase shifters**. A distributed model's total = Σ over QPUs of that per-mesh count.
**Cloud $/query = (#circuits = n_qpus) × S × $1e-4** (Xanadu per-shot; each QPU is a separately-sampled
circuit). So cloud cost scales with **QPU count**, device count scales with **mesh size²**.

### Table 1 — the 3 mainlines vs the monolithic MerLin baseline
| Model | 3w5s acc | QPUs × M | total modes | MZIs | beam splitters | phase shifters | cloud $/query |
|---|---|---|---|---|---|---|---|
| **QResNet-SSL** (1× SimCLR-SSL RN50, 2048-d) | **81.7 ± 1.3** | 512 × 5 | 2,560 | 5,120 | 10,240 | 10,240 | **$256** |
| **QResNet-Ens** (3× SSL RN50, 6144-d) | **83.0 ± 1.5** | 1,536 × 5 | 7,680 | 15,360 | 30,720 | 30,720 | **$768** |
| **QResNet-SSL-Light** (merge, 128-QPU) | **79.7 ± 0.8** | 128 × 17 | 2,176 | 17,408 | 34,816 | 34,816 | **$64** |
| Monolithic — row/col rank-1 † | — (sep-limited) | 1 × 96 | 96 | 4,560 | 9,120 | 9,120 | ~$0.50 |
| Monolithic — full-rank † | — (unbuildable) | 1 × 2,048 | 2,048 | 2,096,128 | 4,192,256 | 4,192,256 | ~$0.50 |

† Monolithic anchored to the QResNet-SSL feature scale (D=2,048; row/col factorization 32×64 → M=R+C=96).
For QResNet-Ens scale (D=6,144=64×96): row/col M=160 → 12,720 MZI / 25,440 BS / 25,440 PS; full-rank
M=6,144 → ~18.9M MZI. Accuracy: QResNet-SSL/Ens are the headline numbers (RESULTS.md main table); the cost
line-up re-measured SSL@512 (read-all) at 80.7 ± 1.8 — same within seed noise. QResNet-Ens non-optimized =
1,536 QPU ($768); its Phase-2 **merged** variant (768 QPU, m=8) is ≈83.0–83.5 at **$384**. (Per-model 95%
CIs: QConv4 81.20 ± 1.10, QResNet-SSL 81.67 ± 1.32, QResNet-Ens design 83.20 ± 1.46 / fresh 82.88 ± 1.58.)

**Reading Table 1 honestly — DP-QCNN does NOT beat the monolithic on cost or raw device count:**
1. **Cloud cost:** the monolithic is **one** circuit → ~$0.50/query, ~500× cheaper than our distributed
   banks (many circuits × shots). Distribution is *cloud-expensive* by construction.
2. **Device count:** at matched expressiveness the monolithic must go **full-rank** (M=2,048 → ~2.1M MZI),
   ~400× *more* than QResNet-SSL — we crush it. But its **row/col rank-1** shortcut (4,560 MZI) is actually
   ~10% *fewer* devices than QResNet-SSL — at the price of representing only separable (rank-1) images.
3. **Where DP-QCNN wins = realizability + expressiveness:** our meshes are **M=5–17**, far under the
   ~64-mode monolithic realizability ceiling (`cost.yaml`); the row/col monolith (M=96) and full-rank
   (M=2,048) exceed it → not near-term buildable at usable loss (mesh depth → cumulative loss). And our
   full-rank per-QPU encoding escapes the row/col rank-1 (separability) limit — see phase-01 separability law.

### Rank-1 vs full-rank monolithic — what the labels mean (justification)
The MerLin monolithic loads a 2-D image by **row/col tensor-multiply**: a row-amplitude vector **r** (length
R) and a column-amplitude vector **c** (length C), encoded into R + C modes, whose reconstructed image is the
**outer product r·cᵀ** — i.e. pixel(i,j) = rᵢ · cⱼ.
- **An outer product of two vectors is a rank-1 matrix** (exactly one non-zero singular value). So the
  row/col monolithic can *only* represent **separable** images — a 2-D pattern that factorizes into
  (row-profile) × (col-profile). It is cheap: for D = 2,048 = 32 × 64, only **M = R + C = 96 modes**.
- **Full-rank** = drop the separability assumption and load the *entire* flattened D-vector into **M = D =
  2,048 modes**, so the interferometer can represent an arbitrary image of **rank up to min(R,C) = 32**
  (here), not just 1. Rank jumps **1 → 32**; modes jump **96 → 2,048** (~**460× more devices**: 4,560 →
  2.1M MZI). (An intermediate rank-k loader = sum of k outer products = k·(R+C) modes.)
- **Does rank-1 degrade accuracy a lot? Expected yes, for wafer defects.** Rings, scratches, edge-loc,
  donut patterns are spatially *correlated* → generally **not separable**, so a rank-1 encoding discards
  most of the discriminative structure. We did **not** run the monolithic on FSWMPR (hence the "—" in
  Table 1), but phase-01 measured exactly this failure mode: on **correlation-only** tasks the row/col
  (rank-1) encoder sat at **~50% (chance)** while a full/spatial encoder reached **~100%** — the
  *separability law*. Full-rank restores the accuracy but needs the unbuildable M=2,048 mesh; our
  distributed QPUs get full-rank expressiveness with buildable M=5–17 chips instead.

### Table 2 — QResNet-SSL-Light: the MERGE variants (512-base scaled down)
Merge = reshape 2,048 → (n_qpus, m=2048/n_qpus), **no Linear, read all m modes** — lossless repack into
fewer/bigger QPUs (total modes ≈ 2,048 throughout). 3 seeds each (`bench/cost_ablation.py`).
| variant | QPUs × M (m) | total modes | MZIs | cloud $/query | vs base | **3w5s acc (mean ± CI95)** |
|---|---|---|---|---|---|---|
| base-512 | 512 × 5 (m=4) | 2,560 | 5,120 | $256 | 1× | **80.7 ± 1.8** |
| cheap-256 | 256 × 9 (m=8) | 2,304 | 9,216 | $128 | 2× cheaper | **80.8 ± 1.7** |
| **cheap-128** ⭐ | 128 × 17 (m=16) | 2,176 | 17,408 | **$64** | **4× cheaper** | **79.7 ± 0.8** |
| cheap-64 | 64 × 33 (m=32) | 2,112 | 33,792 | $32 | 8× cheaper | **78.3 ± 1.1** |
| cheap-32 | 32 × 65 (m=64) | 2,080 | 66,560 | $16 | 16× cheaper | **74.5 ± 1.3** |

- **Merge beats lossy reductions (learned-Linear / PCA / random) by +4–6 pts at every tier** (those stay
  67–77; see results-accuracy-sota.md) — because it discards no information.
- **Recommended: cheap-128 ⭐ — $64/query (4× cheaper) at 79.7 ± 0.8**, essentially at the ~80% bar;
  cheap-256 ($128, 80.8) is the cheapest that clears 80 outright.
- **Tradeoff to note:** cloud $/query falls with fewer QPUs, but **MZI/BS/PS device count RISES** (5,120 →
  66,560) because merging builds bigger meshes — *cloud-cheap ≠ device-cheap*. And m≥32 (M≥33) is
  simulation-heavy (large Fock space), so m=8–16 is the practical sweet spot.

---

## 1. Cloud QPU pricing & cost (per 1 query, QResNet-ensemble = 768 QPU × 5,000 shots = 3.84M shots)
Photonic-native = correct fit; gate-based = $-reference only (need lossy boson→qubit compile).
| provider | tech | list price | **cost / query** |
|---|---|---|---|
| **Quandela** (our Perceval/MerLin stack) | photonic | per-shot not public (OVHcloud/Scaleway; request quote) | order-of Xanadu |
| **Xanadu** | photonic | $1e-4/shot ($100/1M) | **≈ $384** |
| ~~Rigetti~~ (gate, $-ref) | superconducting | $9e-4/shot + $0.30/task | ≈ $3,686 |
| ~~IBM~~ (gate, $-ref) | superconducting | $96/min PAYG | ≈ $1,200–6,000 |
| ~~IonQ~~ (gate, $-ref) | trapped-ion | $0.03/shot + $0.30/task | ≈ $115,430 |
Sources: xanadu.ai, aws.amazon.com/braket/pricing, ibm.com/quantum/pricing, quantumcomputingcost.com (2025–26).
**Verdict: even best-fit photonic cloud ≈ $384/query, plus minutes–hours cloud queue — impractical vs GPU.**

## 2. On-premise hardware price (web-grounded, bottom-up for OUR node)
We time-multiplex **one small physical node** (mesh only needs a handful of modes — tiny). Cryostat +
SNSPD + sources are **per-node, shared**, and dominate the cost:
| component | price |
|---|---|
| **QD single-photon source subsystem** (+pump laser) — generates the single photons the QPU needs | ~$150k |
| **SNSPD** (Superconducting Nanowire Single-Photon Detector) array + readout — counts photons per mode | ~$110k |
| **cryostat** (~2–5 K, shared by SNSPD + QD source) — both need cryogenic cooling | ~$200k |
| programmable MZI mesh chip + packaging + phase drivers | ~$80k |
| control/FPGA/timing + integration/calibration/facility | ~$250k |
| **parts subtotal** | **≈ $0.8M** |

**Precise estimate: ≈ $1.0M capital** to buy one deployed node (parts ~$0.8M + margin/full-stack; range
**$0.8–1.2M**); **≈ $2M over 5 years** (TCO ≈ 2×). Sources: Wikipedia SNSPD (€100k), Quandela (QD sources),
ORCA/Quandela (deployed systems), postquantum/umatechnology (TCO).

**This node price is THE SAME for QConv4 and QResNet-ensemble** — see §4. It does not scale with QPU count.

## 3. Cost-model provenance note
`configs/cost.yaml`'s per-QPU proxy (per_qpu = M(M−1)/2·$250 + $5000 + n·$1500) is kept only for *relative*
mode/photon scaling comparisons (e.g. the compression sweeps in the compression R&D plan) — it omits the
cryo/SNSPD/QD-source costs and is not a capital-cost estimate. **§2's ~$1.0M on-prem node is the price of
record** for any absolute hardware-cost claim.

## 4. QConv4 vs QResNet-ensemble — does the smaller checkpoint mean cheaper hardware? **NO**
| | QConv4 | QResNet-ensemble |
|---|---|---|
| QPUs (time-multiplexed on 1 node) | 3,136 | 768 |
| modes/QPU (M = m+add) | 5 | 9 |
| MZIs/QPU | 10 | 36 |
| trained head params (→ checkpoint size) | ~31k (smaller) | ~larger |
| **on-prem node price** (§2, shared cryo/SNSPD/source) | **≈ $1.0M — SAME node** | **≈ $1.0M — SAME node** |
| shots/query (S=5,000) | 15.68M | 3.84M |
| inference time/query (real QPU) | **~4× SLOWER** (~40–160 s) | ~10–40 s |
| inference cost/query (Xanadu cloud) | **~4× MORE** (≈ $1,568) | ≈ $384 |

**Answer: hardware capital cost is essentially the SAME (~$1.0M)** for both — dominated by the shared
cryo/SNSPD/QD-source node, independent of mesh size or QPU count, since both use one small node time-
multiplexed. **Checkpoint size tracks trained mesh-phase parameters, not hardware footprint.** What DOES
scale with QPU count is inference time/cost: QConv4 needs 3,136 reuse-rounds vs QResNet's 768 →
**~4× slower / ~4× costlier per query on real hardware** (GPU-simulated cost is unaffected — both ~ms,
sub-cent, regardless of QPU count).

## 5. Actual runtime, throughput & cost — GPU vs REAL photonic QPU (QResNet-ensemble)
**Train:** GPU-only (real-QPU training infeasible: parameter-shift → ~10¹⁵ shots). ≈ **60 PFLOP, ~1.5 GPU-hr,
≈ $3.75**, ~13 TFLOP/s. (QConv4 training is similarly GPU-only and cheaper — smaller backbone.)

**Inference (1 query, QResNet-ensemble)** — quantum work = 768 QPUs × S=5,000 = **3.84M 2-photon shots**:
| platform | wall-clock / query | throughput | cost / query |
|---|---|---|---|
| **GPU simulation** (exact expectations, no shots) | **~1–5 ms** | **~100s–1000s /s** | **≈ $2×10⁻⁶** |
| **Real photonic QPU — on-prem node** | **~10–40 s** (loss-limited) | **~0.03–0.1 /s** | **≈ $0.4** (amortized $2M 5-yr TCO, ~50% duty) |
| **Real photonic QPU — cloud (Xanadu)** | ~10–40 s compute **+ min–hrs queue** | queue-bound | **≈ $384** ($1e-4/shot × 3.84M) |
(QConv4: multiply real-QPU rows by ~4×, per §4.)

**Why real QPU is ~1,000–10,000× slower than GPU:** our features are *expectations* (⟨occupation⟩), so
hardware must **collect S≈5,000 shots per QPU**; the GPU computes the same expectation analytically in one
pass. **QPU timing assumptions:** QD source ≈80 MHz pump, ~5–8% source→detector efficiency ⇒ ~0.2–0.5 MHz
useful 2-photon rate; + mesh reconfigs (~0.5 ms each) → ~4 s (optimistic) to ~40 s (realistic) per query,
detection-dominated. On-prem `$/query` is amortized node TCO at good utilization; cloud `$/query` is
per-shot and additionally queue-bound.

---

## Bottom line
- **Cloud QPU:** ~$384/query (Xanadu, best-fit photonic, QResNet-ensemble) → impractical.
- **On-prem hardware:** ≈ **$1.0M capital** (one time-multiplexed photonic node; ~$2M over 5 yr) + $8.5k
  GPU — **the SAME node price for QConv4 and QResNet-ensemble** (§4); a smaller checkpoint does not mean
  cheaper hardware.
- **Our system (GPU):** train ~$4 / 1.5 hr; serve **~ms, sub-cent/query, 100s–1000s/s**.
- **Real photonic QPU (if built):** inference **~10–40 s/query** (QResNet-ensemble; ~4× more for QConv4),
  ~0.03–0.1/s, ≈$0.4/query on-prem (amortized) or ≈$384/query cloud — ~10³–10⁴× slower than GPU because
  expectations need ~5,000 shots/QPU.
- The quantum advantage is a **representational** benefit (photonic feature map > matched classical),
  computed **for free on a GPU**; real photonics only matters if pushed to the classically-hard
  multi-photon regime (see the (separate) compression R&D plan).

## Glossary (photonic hardware terms)
- **QD (quantum-dot) single-photon source:** a semiconductor "artificial atom" in a micro-cavity that
  emits exactly one photon per pump-laser pulse — generates the input photons our QPUs encode features into.
- **SNSPD (superconducting nanowire single-photon detector):** the readout — counts individual photons per
  output mode (our partial-measurement "occupation" reads off these counters).
- **Cryostat:** the closed-cycle fridge (~2–5 K) that cools the SNSPDs and QD sources — required even
  though the *mesh* itself can be near-room-temperature; "room-temperature photonics" refers to the chip,
  not the source/detector subsystem.
