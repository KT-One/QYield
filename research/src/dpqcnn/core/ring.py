"""Concentric-ring encoder — native resolution, universal across architectures.

Implements the design in ``architecture.md``: wafer
defects (rings, donuts, edge-rings, center blobs) are radial phenomena, so a
row/col axis-aligned partition is the wrong topology. Rings are cut in
**normalized radius space** so every wafer's own die area (whatever its native
``(h, w)``) maps onto a common ``[0, r_norm_max]`` scale, and **every pixel is
individually amplitude-encoded at native resolution** — no resize, no
pre-encoding pooling. ``RegisterConv``/``RegisterPool`` are already
mode-count-agnostic (the trainable filter is small and fixed; only the number
of slide/halving steps depends on the register's mode count ``d``), so a
variable-size ring register is not a new capability, just a new use of the
existing building blocks (see architecture.md §1 pt.3).

Two forms, both built on the same ring-geometry primitives:

* :class:`RingEncoder` — for the monolithic baseline. Ring-reorders a wafer's
  pixels (ring 0 first, ring 1 next, ...) into ONE flat list, amplitude-encodes
  the whole thing as a single joint 1-photon state over ``h*w`` modes, no
  partial trace, no distribution — the coherent ceiling case, at native
  resolution.
* :class:`RingPatchEncoder` — for DP-QCNN. Each ring's pixels become an
  independent register (one QPU per ring), amplitude-encoded locally.

Rings are grouped into **independent pairs** (ring 2k, ring 2k+1); within a
pair the DP model may use classical feed-forward (CC), but pairs never
communicate with each other (always NC across pairs) — see architecture.md
§2b. This keeps CC cost linear in ``n_rings`` (each pair's conditional bank
size depends only on its own two rings, never on how many other pairs exist).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .blocks import RegisterConv, RegisterDense, RegisterDenseBank, RegisterPool, marginal_readout


# ---------------------------------------------------------------------------
# Ring geometry (shared primitive).
# ---------------------------------------------------------------------------
def normalized_radius(h: int, w: int) -> torch.Tensor:
    """r_norm(i,j) for an h x w grid, normalized by each axis's own half-extent.

    Returns an (h, w) tensor. Center = ((h-1)/2, (w-1)/2). Normalizing by
    (h/2, w/2) separately (not by a single scalar) maps every wafer's own die
    area onto a comparable radius scale regardless of native size/aspect
    ratio, so ring bins defined once in this space are meaningful across every
    wafer without ever resizing the pixel grid (architecture.md §1 pt.1).
    """
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys = (torch.arange(h, dtype=torch.float32) - cy) / max(h / 2.0, 1e-9)
    xs = (torch.arange(w, dtype=torch.float32) - cx) / max(w / 2.0, 1e-9)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt(yy ** 2 + xx ** 2)


def ring_bin_edges(n_rings: int, r_max: float = math.sqrt(2.0)) -> torch.Tensor:
    """N+1 bin edges for equal-AREA annuli in normalized-radius space.

    Ring k covers r_norm in [edges[k], edges[k+1]). Equal-area (r ~ sqrt(k/N))
    rather than equal-width so each ring carries a comparable pixel count on a
    filled disk (architecture.md §1: fab engineers describe defects as
    "fraction of wafer radius", not raw pixel count).
    """
    k = torch.arange(n_rings + 1, dtype=torch.float32)
    return r_max * torch.sqrt(k / n_rings)


def ring_assignment(h: int, w: int, n_rings: int) -> torch.Tensor:
    """(h, w) long tensor of ring index in [0, n_rings-1] for each pixel."""
    r = normalized_radius(h, w)
    edges = ring_bin_edges(n_rings, r_max=float(r.max()) + 1e-6)
    # bucketize: idx = number of edges[1:-1] <= r, clamped to n_rings-1
    idx = torch.bucketize(r.reshape(-1), edges[1:-1], right=False)
    return idx.clamp_(0, n_rings - 1).reshape(h, w)


def ring_pixel_lists(h: int, w: int, n_rings: int) -> list[torch.Tensor]:
    """Per-ring flat pixel INDEX lists (into a flattened h*w image), ring 0..N-1.

    Precompute once per distinct (h, w, n_rings) and cache — this is what lets
    RingEncoder/RingPatchEncoder pull out "this ring's real native pixels" for
    any wafer of that shape without recomputing geometry per-sample.
    """
    assign = ring_assignment(h, w, n_rings).reshape(-1)
    return [torch.nonzero(assign == k, as_tuple=False).squeeze(-1) for k in range(n_rings)]


# ---------------------------------------------------------------------------
# Amplitude encoding helpers (shared convention: unit-norm real vector -> rank-1
# 1-photon density matrix, same as PatchEncoder/RegisterEncoder elsewhere).
# ---------------------------------------------------------------------------
def _amplitude_encode(v: torch.Tensor) -> torch.Tensor:
    """(b, m) real -> (b, m, m) complex rank-1 density matrix |v><v|, unit-norm."""
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    v = v.to(torch.complex64)
    return v.unsqueeze(2) @ v.unsqueeze(1).conj()


# ---------------------------------------------------------------------------
# 2a. RingEncoder — monolithic, native resolution, single joint state.
# ---------------------------------------------------------------------------
class RingEncoder(nn.Module):
    """Ring-reorder a native h*w wafer map into ONE flat pixel list and
    amplitude-encode it as a single joint 1-photon state over h*w modes.

    No partition into separate QPUs, no partial trace — this is the coherent,
    undistributed convention for the monolithic-ring baseline (architecture.md
    §2a). Built per distinct (h, w, n_rings); cache instances per shape.
    """

    def __init__(self, h: int, w: int, n_rings: int):
        super().__init__()
        self.h, self.w, self.n_rings = h, w, n_rings
        lists = ring_pixel_lists(h, w, n_rings)
        order = torch.cat(lists)  # ring 0's pixel indices first, then ring 1's, ...
        self.register_buffer("order", order, persistent=False)
        self.modes = h * w
        self.ring_sizes = [int(t.numel()) for t in lists]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (b, h, w) real -> rho (b, h*w, h*w) complex, ring-reordered."""
        if x.dim() == 4:
            x = x.squeeze(1)
        b = x.shape[0]
        flat = x.reshape(b, -1)[:, self.order]
        return _amplitude_encode(flat)


# ---------------------------------------------------------------------------
# 2b. RingPatchEncoder — DP-QCNN, native resolution, one register per ring.
# ---------------------------------------------------------------------------
class RingPatchEncoder(nn.Module):
    """Split a native h*w wafer map into N ring registers, each amplitude-
    encoding only its own ring's real pixels (native resolution, no resize).

    Returns a list of N (b, d_k, d_k) density matrices, d_k = ring k's native
    pixel count for this wafer shape.
    """

    def __init__(self, h: int, w: int, n_rings: int):
        super().__init__()
        if n_rings % 2 != 0:
            raise ValueError("n_rings must be even (paired-CC topology, architecture.md §2b)")
        self.h, self.w, self.n_rings = h, w, n_rings
        lists = ring_pixel_lists(h, w, n_rings)
        for k, idx in enumerate(lists):
            self.register_buffer(f"_idx_{k}", idx, persistent=False)
        self.ring_sizes = [int(t.numel()) for t in lists]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 4:
            x = x.squeeze(1)
        b = x.shape[0]
        flat = x.reshape(b, -1)
        out = []
        for k in range(self.n_rings):
            idx = getattr(self, f"_idx_{k}")
            out.append(_amplitude_encode(flat[:, idx]))
        return out


# ---------------------------------------------------------------------------
# RingQPU — one ring's independent chip (conv -> pool -> dense -> readout).
# Mirrors model.QPU / model.ConditionalQPU but built from an arbitrary native
# d (this wafer shape's ring pixel count) instead of a fixed image dimension,
# and pools down to a common `dense_width` before RegisterDense so different
# native ring sizes still land on comparable interpret-tensor axes.
# ---------------------------------------------------------------------------
def _padded_size(d: int, dense_width: int, kernel_size: int) -> tuple[int, list[int]]:
    """Smallest d_pad >= d such that repeated //kernel_size halving reaches
    EXACTLY dense_width with each stage evenly divisible (RegisterPool requires
    exact divisibility). Extra modes beyond the real d are zero-amplitude
    (physically: unused/empty modes) — same convention as RegisterDense's
    ``add_modes`` zero-padding, so no real pixel information is altered, only
    padded with vacuum. Real wafer-ring sizes are not powers of kernel_size
    (e.g. 338, 52, 24), so this padding is required for the pool chain to be
    well-formed; it never touches which pixels get encoded (architecture.md §1
    pt.3 remains "every pixel individually encoded" — padding only ADDS empty
    modes, never removes or averages real ones).
    """
    if d <= dense_width:
        return d, []
    n_stages = 0
    probe = dense_width
    while probe < d:
        probe *= kernel_size
        n_stages += 1
    return probe, [kernel_size] * n_stages


def _pool_stages(d: int, dense_width: int, kernel_size: int) -> list[int]:
    """Repeated halving schedule d -> ... -> dense_width (architecture.md §1
    pt.3: resolution adaptation happens through RegisterPool's existing halving,
    not a new downsampling primitive). Superseded by ``_padded_size`` for the
    actual stage count used in :class:`RingQPU`; kept for reference/tests."""
    stages = []
    cur = d
    while cur > dense_width:
        nxt = cur // kernel_size
        if nxt < dense_width or nxt == cur:
            break
        stages.append(kernel_size)
        cur = nxt
    return stages


def _pad_rho(rho: torch.Tensor, target: int) -> torch.Tensor:
    """Zero-pad a (b, d, d) density matrix up to (b, target, target). Extra
    modes carry zero amplitude (vacuum), matching RegisterDense's add_modes."""
    d = rho.shape[-1]
    if target == d:
        return rho
    b = rho.shape[0]
    out = torch.zeros(b, target, target, dtype=rho.dtype, device=rho.device)
    out[:, :d, :d] = rho
    return out


class RingQPU(nn.Module):
    """One ring's independent chip: conv -> repeated pool -> dense -> readout.

    Built for a specific native mode count ``d`` (this wafer-shape's ring pixel
    count); pools (possibly several times) down to ``dense_width`` before the
    fixed-size RegisterDense, so registers of different native ``d`` still
    expose comparable ``out_modes`` for the interpret tensor. Real ring sizes
    are rarely powers of ``kernel_size`` (e.g. 338, 52, 24), so ``d`` is padded
    with zero-amplitude (vacuum) modes up to the nearest size the pool chain
    can evenly halve down to ``dense_width`` — see ``_padded_size``. Padding
    only adds empty modes; every real pixel is still individually encoded.
    """

    def __init__(self, d: int, dense_width: int = 8, kernel_size: int = 2,
                 add_modes: int = 2, conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        self.d = d
        self.d_pad, stages = _padded_size(d, dense_width, kernel_size)
        convs, poolmods = [], []
        cur = self.d_pad
        for k in stages:
            convs.append(RegisterConv(cur, kernel_size, kernel_size, conv_circuit))
            poolmods.append(RegisterPool(cur, kernel_size))
            cur = cur // k
        if not stages:
            ks = min(kernel_size, cur)
            convs.append(RegisterConv(cur, ks, ks, conv_circuit))
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(poolmods)
        self.final_d = cur
        self.dense = RegisterDense(self.final_d, add_modes, dense_circuit)
        self.out_modes = self.dense.m

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        rho = _pad_rho(rho, self.d_pad)
        for i, conv in enumerate(self.convs):
            rho = conv(rho)
            if i < len(self.pools):
                rho = self.pools[i](rho)
        return marginal_readout(self.dense(rho))


class ConditionalRingQPU(nn.Module):
    """Ring QPU B of a pair: shared conv/pool, then a bank of dense layers
    selected by the pair-partner's classical outcome. Mirrors
    ``model.ConditionalQPU`` but for an arbitrary native ``d`` (zero-padded to
    the pool chain's required size, same convention as :class:`RingQPU`)."""

    def __init__(self, d: int, n_cond: int, dense_width: int = 8, kernel_size: int = 2,
                 add_modes: int = 2, conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        self.d = d
        self.d_pad, stages = _padded_size(d, dense_width, kernel_size)
        convs, poolmods = [], []
        cur = self.d_pad
        for k in stages:
            convs.append(RegisterConv(cur, kernel_size, kernel_size, conv_circuit))
            poolmods.append(RegisterPool(cur, kernel_size))
            cur = cur // k
        if not stages:
            ks = min(kernel_size, cur)
            convs.append(RegisterConv(cur, ks, ks, conv_circuit))
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(poolmods)
        self.final_d = cur
        self.bank = RegisterDenseBank(self.final_d, n_cond, add_modes, dense_circuit)
        self.n_cond = n_cond
        self.out_modes = self.bank.m

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        rho = _pad_rho(rho, self.d_pad)
        for i, conv in enumerate(self.convs):
            rho = conv(rho)
            if i < len(self.pools):
                rho = self.pools[i](rho)
        return self.bank(rho)  # (b, n_cond, out_modes)


# ---------------------------------------------------------------------------
# RingDPQCNN — paired-CC N-ring distributed model (architecture.md §2b/§3).
# ---------------------------------------------------------------------------
class RingDPQCNN(nn.Module):
    """N-ring distributed model, paired-CC topology.

    Rings are grouped into independent pairs (ring 2k, ring 2k+1). Within a
    pair, ``comm="CC"`` conditions ring 2k+1's dense bank on ring 2k's
    classical outcome (identical topology to the legacy 2-QPU row/col
    DPQCNN's CC); ``comm="NC"`` runs every ring fully independently. Pairs
    NEVER communicate with each other regardless of ``comm`` — cost stays
    linear in ``n_rings`` (architecture.md §2b, §5.1).

    All ``2 * n_pairs`` ring marginals feed into ONE interpret tensor ``W``
    (outer product across every line), generalizing the bilinear NC form.
    """

    def __init__(self, ring_sizes: list[int], comm: str = "CC", *,
                 dense_width: int = 8, kernel_size: int = 2, add_modes: int = 2,
                 num_classes: int = 2, conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        if comm not in {"NC", "CC"}:
            raise ValueError("comm must be 'NC' or 'CC'")
        if len(ring_sizes) % 2 != 0:
            raise ValueError("ring_sizes must have even length (paired-CC topology)")
        self.comm = comm
        self.n_rings = len(ring_sizes)
        self.n_pairs = self.n_rings // 2

        qpus = []
        out_modes = []
        for p in range(self.n_pairs):
            d0, d1 = ring_sizes[2 * p], ring_sizes[2 * p + 1]
            q0 = RingQPU(d0, dense_width, kernel_size, add_modes, conv_circuit, dense_circuit)
            m0 = q0.out_modes
            if comm == "NC":
                q1 = RingQPU(d1, dense_width, kernel_size, add_modes, conv_circuit, dense_circuit)
            else:
                q1 = ConditionalRingQPU(d1, m0, dense_width, kernel_size, add_modes,
                                        conv_circuit, dense_circuit)
            m1 = q1.out_modes
            qpus.append((q0, q1))
            out_modes.extend([m0, m1])
        self.qpus = nn.ModuleList([m for pair in qpus for m in pair])
        self.out_modes = out_modes
        # W: (num_classes, m_0, m_1, ..., m_{2*n_pairs-1})
        self.W = nn.Parameter(0.1 * torch.randn(num_classes, *out_modes))

    def forward(self, rhos: list[torch.Tensor]) -> torch.Tensor:
        marginals = []
        for p in range(self.n_pairs):
            q0, q1 = self.qpus[2 * p], self.qpus[2 * p + 1]
            rho0, rho1 = rhos[2 * p], rhos[2 * p + 1]
            p0 = q0(rho0)                      # (b, m0)
            marginals.append(p0)
            if self.comm == "NC":
                marginals.append(q1(rho1))     # (b, m1)
            else:
                p1_cond = q1(rho1)             # (b, m0, m1) = P(y|x)
                # fold P(x) into the conditional so the outer product below
                # still combines pure per-line marginals: p1_eff = sum_x P0(x) P1(y|x)
                p1_eff = torch.einsum("bx,bxy->by", p0, p1_cond)
                marginals.append(p1_eff)
        # outer product across all 2*n_pairs marginals, contracted with W.
        # Use uppercase letters for feature axes so "b" (batch) never collides.
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        joint = marginals[0]
        cur_letters = [letters[0]]
        for i, m in enumerate(marginals[1:], start=1):
            new_letter = letters[i]
            eq = f"b{''.join(cur_letters)},b{new_letter}->b{''.join(cur_letters)}{new_letter}"
            joint = torch.einsum(eq, joint, m)
            cur_letters.append(new_letter)
        w_letters = "".join(cur_letters)
        logits = torch.einsum(f"b{w_letters},c{w_letters}->bc", joint, self.W)
        return logits

    def extra_repr(self) -> str:
        return f"comm={self.comm}, n_rings={self.n_rings}, n_pairs={self.n_pairs}, out_modes={self.out_modes}"


class RingDPModel(nn.Module):
    """End-to-end ring-DP model: RingPatchEncoder -> RingDPQCNN. Built per
    native wafer shape (h, w) and n_rings; cache instances per shape bucket."""

    def __init__(self, h: int, w: int, n_rings: int, comm: str = "CC", *,
                 dense_width: int = 8, kernel_size: int = 2, add_modes: int = 2,
                 num_classes: int = 2, conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        self.encoder = RingPatchEncoder(h, w, n_rings)
        self.model = RingDPQCNN(self.encoder.ring_sizes, comm, dense_width=dense_width,
                                kernel_size=kernel_size, add_modes=add_modes,
                                num_classes=num_classes, conv_circuit=conv_circuit,
                                dense_circuit=dense_circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.encoder(x))

    def extra_repr(self) -> str:
        return f"h={self.encoder.h}, w={self.encoder.w}, n_rings={self.encoder.n_rings}"


# ---------------------------------------------------------------------------
# RingMonolithic — monolithic-ring baseline (architecture.md §2a).
# Single coherent circuit over the FULL ring-reordered h*w-mode register.
# ---------------------------------------------------------------------------
class RingMonolithic(nn.Module):
    """Monolithic baseline with the ring-reordered encoder: one joint 1-photon
    state over h*w modes (native resolution), one coherent conv/pool/dense
    chain, no partition, no partial trace (architecture.md §2a).
    """

    def __init__(self, h: int, w: int, n_rings: int, *,
                 dense_width: int = 8, kernel_size: int = 2, add_modes: int = 2,
                 num_classes: int = 2, conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        self.encoder = RingEncoder(h, w, n_rings)
        d = self.encoder.modes
        self.qpu = RingQPU(d, dense_width, kernel_size, add_modes, conv_circuit, dense_circuit)
        self.out_modes = self.qpu.out_modes
        self.W = nn.Parameter(0.1 * torch.randn(num_classes, self.out_modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = self.encoder(x)
        p = self.qpu(rho)                 # (b, m)
        return torch.einsum("bx,cx->bc", p, self.W)

    def extra_repr(self) -> str:
        return f"h={self.encoder.h}, w={self.encoder.w}, n_rings={self.encoder.n_rings}"


__all__ = [
    "normalized_radius", "ring_bin_edges", "ring_assignment", "ring_pixel_lists",
    "RingEncoder", "RingPatchEncoder",
    "RingQPU", "ConditionalRingQPU",
    "RingDPQCNN", "RingDPModel", "RingMonolithic",
    "_padded_size", "_pad_rho",
]
