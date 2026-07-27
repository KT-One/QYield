"""Single-register (1-photon, d-mode) building blocks for one QPU.

Each distributed QPU processes ONE register (rows or columns) holding a single
photon in ``d`` modes, represented by a density matrix ``rho`` of shape
``(b, d, d)``. These blocks are the single-register *factors* of the monolithic
baseline layers, which guarantees that on a separable (rank-1) input the
distributed pipeline reproduces the monolithic one up to the dense/interconnect
stage.

Key simplification from the 1-photon constraint: a linear-optical circuit on
``m`` modes acts on the 1-photon subspace as the ``m x m`` mode unitary ``U``
itself, so every block is just ``rho -> U rho U^dag`` (no multi-photon SLOS, no
Hong-Ou-Mandel bunching). This is exactly the hardware advantage the proposal
claims: isolated single-photon registers with deterministic single-click
readout.
"""

from __future__ import annotations

import io
import sys

import numpy as np
import torch
import torch.nn as nn

from ._baseline import CircuitConverter, get_circuit

try:  # Perceval Circuit for assembling the register interferometer.
    from perceval import Circuit
except Exception as exc:  # pragma: no cover
    raise ImportError("perceval is required for dpqcnn.blocks") from exc


def _converter(circuit) -> CircuitConverter:
    """Build a CircuitConverter, suppressing pcvl_pytorch chatter."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return CircuitConverter(circuit, ["phi"], dtype=torch.float32)
    finally:
        sys.stdout = original_stdout


def _evolve(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Apply U rho U^dag for a single (m x m) unitary to a batch of rho."""
    b = rho.shape[0]
    u = u.to(rho.dtype)
    u_b = u.unsqueeze(0).expand(b, -1, -1)
    u_dag = u_b.transpose(1, 2).conj()
    return torch.bmm(torch.bmm(u_b, rho), u_dag)


class MeshUnitary(nn.Module):
    """Vectorized, differentiable builder for the ``m x m`` mode unitary of a
    single-photon ``BS`` interferometer, replacing merlin's per-gate Python loop.

    Rationale (see ): for a single photon, a linear-optical circuit acts
    on the 1-photon subspace as its ``m x m`` mode unitary ``U``, which depends
    only on the trainable phases (not the input). merlin's ``CircuitConverter.to_tensor``
    builds ``U`` by looping over O(m^2) 2x2 gates in Python (~84 us/gate), which
    dominates training time. This module reproduces **merlin's exact** unitary
    (verified bit-identical up to fp round-off) but builds it in vectorized torch:
    the ``BS`` mesh is scheduled into ``nL ~ 2m`` layers of disjoint 2x2 rotations,
    the ``cos/sin(phi/2)`` are scattered into ``nL`` block-diagonal matrices in one
    vectorized op, and ``U`` is the ordered product of the layer matrices
    (``nL`` batched matmuls, no per-gate Python loop). ~10-60x faster; runs on GPU.

    Merlin's convention (extracted from ``list_rct``): each BS on modes ``(a, a+1)``
    with parameter ``phi_k`` contributes the 2x2 rotation
    ``[[cos(phi_k/2), -sin(phi_k/2)], [sin(phi_k/2), cos(phi_k/2)]]``, applied by
    left-multiplication in circuit order.

    Forward: ``phi`` of shape ``(n_param,)`` or ``(B, n_param)`` -> ``U`` of shape
    ``(m, m)`` or ``(B, m, m)`` (real dtype; callers cast to complex as needed).
    """

    def __init__(self, circ):
        super().__init__()
        self.m = circ.m
        self.nparam = len(circ.get_parameters())
        conv = _converter(circ)  # merlin used ONCE, at init, only for structure
        modes, kidx = [], []
        for r, c in conv.list_rct:
            if isinstance(c, torch.Tensor):
                raise NotImplementedError("MeshUnitary supports parametric BS meshes only")
            ps = c.get_parameters()
            if len(ps) != 1:
                raise NotImplementedError("MeshUnitary expects single-parameter BS gates")
            modes.append(list(r)[0])
            kidx.append(int(ps[0].name.split("_")[1]))
        # ASAP layer scheduling: disjoint 2x2 blocks share a layer; shared modes
        # force a later layer (preserves the sequential left-multiplication order).
        last: dict[int, int] = {}
        layer_of = []
        for a in modes:
            lvl = max(last.get(a, -1), last.get(a + 1, -1)) + 1
            layer_of.append(lvl)
            last[a] = last[a + 1] = lvl
        self.n_layers = (max(layer_of) + 1) if layer_of else 1
        # persistent buffers so .to(device) moves the index tensors too
        self.register_buffer("_layer", torch.tensor(layer_of, dtype=torch.long), persistent=False)
        self.register_buffer("_a", torch.tensor(modes, dtype=torch.long), persistent=False)
        self.register_buffer("_k", torch.tensor(kidx, dtype=torch.long), persistent=False)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        single = phi.dim() == 1
        if single:
            phi = phi.unsqueeze(0)
        b, m, nL = phi.shape[0], self.m, self.n_layers
        half = 0.5 * phi
        c, s = torch.cos(half), torch.sin(half)          # (b, nparam)
        eye = torch.eye(m, dtype=phi.dtype, device=phi.device)
        L = eye.reshape(1, 1, m, m).repeat(nL, b, 1, 1)  # (nL, b, m, m)
        li, a, k = self._layer, self._a, self._a + 1
        ck = c[:, self._k].transpose(0, 1)               # (nparam, b)
        sk = s[:, self._k].transpose(0, 1)
        L[li, :, self._a, self._a] = ck
        L[li, :, self._a, k] = -sk
        L[li, :, k, self._a] = sk
        L[li, :, k, k] = ck
        u = L[0]
        for l in range(1, nL):
            u = L[l] @ u                                  # ordered product of layers
        return u.squeeze(0) if single else u

    def extra_repr(self) -> str:
        return f"m={self.m}, n_param={self.nparam}, n_layers={self.n_layers}"


class RegisterConv(nn.Module):
    """Convolution on one register: a shared ``kernel``-mode filter applied with
    ``stride`` across ``d`` modes (the single-register factor of QConv2d).

    Parameters are shared across stride positions, matching the paper's
    translation-invariant convolutional filter.
    """

    def __init__(self, d: int, kernel_size: int = 2, stride: int | None = None,
                 circuit: str = "BS"):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

        reg = Circuit(d, name="RegConv")
        filt = get_circuit(kernel_size, circuit)  # shared param object
        n_pos = (d - kernel_size) // self.stride + 1
        for i in range(n_pos):
            reg.add(self.stride * i, filt)
        self._reg = reg
        self.mesh = MeshUnitary(reg)
        num_params = self.mesh.nparam
        self.phi = nn.Parameter(2 * np.pi * torch.rand(num_params))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        u = self.mesh(self.phi)
        return _evolve(rho, u)

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel={self.kernel_size}, stride={self.stride}"


class RegisterPool(nn.Module):
    """State-injection pooling on one register, halving ``d -> d/k``.

    This is the single-register factor of the joint QPooling: the baseline mask
    factorises as (X condition) x (Y condition), so applying this independently
    to each register is exactly consistent with the monolithic pooling on a
    separable state.

        mask(i, m) = (i%k == m%k) AND ( ((i%k != 0) AND (i == m)) OR (i%k == 0) )
        new index  = i // k
    """

    def __init__(self, d: int, kernel_size: int = 2):
        super().__init__()
        if d % kernel_size != 0:
            raise ValueError("register size must be divisible by kernel_size")
        self.d = d
        self.k = kernel_size
        self.new_d = d // kernel_size

        i = torch.arange(d)
        m = torch.arange(d)
        i_grid, m_grid = torch.meshgrid(i, m, indexing="ij")
        match_odd = ((i_grid % self.k != 0) & (i_grid == m_grid)) | (i_grid % self.k == 0)
        inject = (i_grid % self.k) == (m_grid % self.k)
        mask = inject & match_odd
        coords = torch.nonzero(mask, as_tuple=False)
        self.register_buffer("_src_i", coords[:, 0], persistent=False)
        self.register_buffer("_src_m", coords[:, 1], persistent=False)
        self.register_buffer("_new_i", i_grid[mask] // self.k, persistent=False)
        self.register_buffer("_new_m", m_grid[mask] // self.k, persistent=False)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        b = rho.shape[0]
        nvals = self._src_i.numel()
        b_idx = torch.arange(b, device=rho.device).unsqueeze(1).expand(-1, nvals).reshape(-1)
        new_i = self._new_i.expand(b, -1).reshape(-1)
        new_m = self._new_m.expand(b, -1).reshape(-1)
        out = torch.zeros(b, self.new_d, self.new_d, dtype=rho.dtype, device=rho.device)
        vals = rho[:, self._src_i, self._src_m].reshape(-1)
        out.index_put_((b_idx, new_i, new_m), vals, accumulate=True)
        return out

    def extra_repr(self) -> str:
        return f"d={self.d}->{self.new_d}, k={self.k}"


class RegisterDense(nn.Module):
    """Local dense layer on one register: pad with ``add_modes`` empty modes then
    apply the ``(p + add_modes)``-mode interferometer unitary (1-photon => the
    evolution operator is the mode unitary itself).
    """

    def __init__(self, p: int, add_modes: int = 0, circuit: str = "BS"):
        super().__init__()
        self.p = p
        self.add_modes = add_modes
        self.m = p + add_modes

        circ = get_circuit(self.m, circuit)
        self.mesh = MeshUnitary(circ)
        num_params = self.mesh.nparam
        self.phi = nn.Parameter(2 * np.pi * torch.rand(num_params))

    def _pad(self, rho: torch.Tensor) -> torch.Tensor:
        if self.add_modes == 0:
            return rho
        b = rho.shape[0]
        out = torch.zeros(b, self.m, self.m, dtype=rho.dtype, device=rho.device)
        out[:, : self.p, : self.p] = rho
        return out

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        rho = self._pad(rho)
        u = self.mesh(self.phi)
        return _evolve(rho, u)

    def extra_repr(self) -> str:
        return f"p={self.p}, add_modes={self.add_modes}, m={self.m}"


class RegisterDenseBank(nn.Module):
    """A bank of ``n_cond`` independent dense layers sharing one circuit structure.

    Mathematically identical to ``n_cond`` separate :class:`RegisterDense` modules
    (one per conditional outcome), but builds all ``n_cond`` mode unitaries in a
    **single batched** ``to_tensor`` call and applies them with a batched einsum.
    This amortises merlin's per-call component-loop overhead over the whole bank
    (~15x faster than a Python loop for n_cond~18) while producing bit-identical
    results (verified: max|diff| = 0 vs the per-dense loop for equal ``phi``).

    Forward: pooled ``rho`` ``(b, p, p)`` -> per-condition marginals ``(b, n_cond, m)``.
    """

    def __init__(self, p: int, n_cond: int, add_modes: int = 0, circuit: str = "BS"):
        super().__init__()
        self.p = p
        self.add_modes = add_modes
        self.m = p + add_modes
        self.n_cond = n_cond

        circ = get_circuit(self.m, circuit)
        self.mesh = MeshUnitary(circ)
        num_params = self.mesh.nparam
        self.phi = nn.Parameter(2 * np.pi * torch.rand(n_cond, num_params))

    def _pad(self, rho: torch.Tensor) -> torch.Tensor:
        if self.add_modes == 0:
            return rho
        b = rho.shape[0]
        out = torch.zeros(b, self.m, self.m, dtype=rho.dtype, device=rho.device)
        out[:, : self.p, : self.p] = rho
        return out

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        rho = self._pad(rho)                                   # (b, m, m)
        u = self.mesh(self.phi).to(rho.dtype)                  # (n_cond, m, m)
        # U_i rho_b U_i^dag for every condition i and batch element b:
        tmp = torch.einsum("nij,bjk->bnik", u, rho)            # (b, n_cond, m, m)
        out = torch.einsum("bnik,nlk->bnil", tmp, u.conj())    # (b, n_cond, m, m)
        diag = out.diagonal(dim1=2, dim2=3)                    # (b, n_cond, m)
        p = diag.real if torch.is_complex(diag) else diag
        return p.clamp_min(0.0)

    def extra_repr(self) -> str:
        return f"p={self.p}, add_modes={self.add_modes}, m={self.m}, n_cond={self.n_cond}"


class RegisterConvBank(nn.Module):
    """A bank of ``n_cond`` independent :class:`RegisterConv` filters (CC placement
    ablation: feed A's outcome into B's *convolution* stage instead of its Dense
    stage). Each condition gets its own filter phases, but the windowed/shared
    circuit *structure* (same ``kernel_size``, ``stride``, positions) is identical
    across conditions -- only the trained phases differ, mirroring how
    :class:`RegisterDenseBank` conditions Dense.

    Built the same batched way as ``RegisterDenseBank`` (one ``MeshUnitary`` over
    the ``(n_cond, num_params)`` phase tensor, one batched einsum) rather than a
    Python loop over ``n_cond`` separate ``RegisterConv`` modules.

    Forward: ``rho`` ``(b, d, d)`` -> per-condition evolved states ``(b, n_cond, d, d)``.
    """

    def __init__(self, d: int, n_cond: int, kernel_size: int = 2, stride: int | None = None,
                 circuit: str = "BS"):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.n_cond = n_cond

        reg = Circuit(d, name="RegConvBank")
        filt = get_circuit(kernel_size, circuit)
        n_pos = (d - kernel_size) // self.stride + 1
        for i in range(n_pos):
            reg.add(self.stride * i, filt)
        self.mesh = MeshUnitary(reg)
        num_params = self.mesh.nparam
        self.phi = nn.Parameter(2 * np.pi * torch.rand(n_cond, num_params))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        u = self.mesh(self.phi).to(rho.dtype)                   # (n_cond, d, d)
        rho_b = rho.unsqueeze(1).expand(-1, self.n_cond, -1, -1)  # (b, n_cond, d, d)
        u_dag = u.conj().transpose(-1, -2)
        tmp = torch.einsum("nij,bnjk->bnik", u, rho_b)
        return torch.einsum("bnik,nkl->bnil", tmp, u_dag)         # (b, n_cond, d, d)

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel={self.kernel_size}, stride={self.stride}, n_cond={self.n_cond}"


class RegisterLoaderBank(nn.Module):
    """A bank of ``n_cond`` independent single-layer phase circuits applied right
    after encoding, before Conv (CC placement ablation: feed A's outcome into
    B's *data-loading* stage). The bare one-hot encoder is untrainable (a fixed
    amplitude map), so this is the earliest point at which a *trainable*
    conditional gate can be inserted -- one ``d``-mode interferometer layer per
    condition, applied to the freshly encoded register state before any Conv
    filtering happens.

    Structurally identical to :class:`RegisterDenseBank` (a bank of ``d``-mode
    unitaries selected by A's outcome) but placed at the front of the QPU
    pipeline instead of the end.

    Forward: ``rho`` ``(b, d, d)`` -> per-condition evolved states ``(b, n_cond, d, d)``.
    """

    def __init__(self, d: int, n_cond: int, circuit: str = "BS"):
        super().__init__()
        self.d = d
        self.n_cond = n_cond
        circ = get_circuit(d, circuit)
        self.mesh = MeshUnitary(circ)
        num_params = self.mesh.nparam
        self.phi = nn.Parameter(2 * np.pi * torch.rand(n_cond, num_params))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        u = self.mesh(self.phi).to(rho.dtype)                    # (n_cond, d, d)
        rho_b = rho.unsqueeze(1).expand(-1, self.n_cond, -1, -1)
        u_dag = u.conj().transpose(-1, -2)
        tmp = torch.einsum("nij,bnjk->bnik", u, rho_b)
        return torch.einsum("bnik,nkl->bnil", tmp, u_dag)

    def extra_repr(self) -> str:
        return f"d={self.d}, n_cond={self.n_cond}"


def marginal_readout(rho: torch.Tensor) -> torch.Tensor:
    """Per-mode photon probability P(x) for a 1-photon register = diag(rho).

    Returns a real, non-negative tensor of shape ``(b, m)`` (rows sum to ~1).

    Note: ``rho`` is Hermitian (U rho U^dag), so its diagonal is physically real
    and non-negative. We take ``.real`` rather than ``torch.abs`` because the
    gradient of the complex modulus is NaN at 0, and a 1-photon register can
    legitimately drive a mode probability to exactly 0 during training (this was
    observed to NaN-out W around epoch ~18). ``.real`` is differentiable
    everywhere; ``clamp_min(0)`` guards against tiny negative fp noise.
    """
    diag = rho.diagonal(dim1=1, dim2=2)
    p = diag.real if torch.is_complex(diag) else diag
    return p.clamp_min(0.0)


__all__ = ["MeshUnitary", "RegisterConv", "RegisterPool", "RegisterDense", "RegisterDenseBank",
           "RegisterConvBank", "RegisterLoaderBank", "marginal_readout"]
