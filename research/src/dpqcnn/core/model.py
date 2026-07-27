"""The assembled DP-QCNN model with no-communication (NC) and classical
feed-forward (CC) variants.

Two independent photonic QPUs each process one register (rows / columns) and
produce a marginal measurement distribution. A trainable classical *interpret
tensor* ``W`` combines them into class logits:

    NC:  logit_c = sum_{x,y} W[c,x,y] * P_A(x) * P_B(y)
    CC:  logit_c = sum_{x,y} W[c,x,y] * P_A(x) * P_B(y | x)

``W`` generalises Hwang et al.'s interpret weights ``(w0,w1,w2,w3)`` to a full
``(num_classes, m_A, m_B)`` tensor over the two registers' marginal bins. The
CC variant realises genuine classical feed-forward: QPU A's classical outcome
``x`` selects one of ``m_A`` conditional phase-shift settings on QPU B, giving
``P_B(y | x)``. All branches are differentiable (expectation weighted by the
measured ``P_A``), so the whole model trains with plain autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    RegisterConv, RegisterConvBank, RegisterDense, RegisterDenseBank,
    RegisterLoaderBank, RegisterPool, marginal_readout,
)
from .encoder import RegisterEncoder


class QPU(nn.Module):
    """One photonic register: [conv -> pool -> dense] -> marginal readout.

    ``use_dense=False`` drops the local Dense stage entirely (ablation axis 2):
    the register is measured right after Conv+Pool, so ``out_modes`` becomes the
    post-pool width ``p`` instead of ``p + add_modes``.
    """

    def __init__(self, d: int, kernel_size: int = 2, add_modes: int = 2,
                 conv_circuit: str = "BS", dense_circuit: str = "BS",
                 use_dense: bool = True):
        super().__init__()
        self.use_dense = use_dense
        self.conv = RegisterConv(d, kernel_size, kernel_size, conv_circuit)
        self.pool = RegisterPool(d, kernel_size)
        if use_dense:
            self.dense = RegisterDense(self.pool.new_d, add_modes, dense_circuit)
            self.out_modes = self.dense.m
        else:
            self.dense = None
            self.out_modes = self.pool.new_d

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.conv(rho))
        if self.use_dense:
            return marginal_readout(self.dense(pooled))
        return marginal_readout(pooled)


class ConditionalQPU(nn.Module):
    """QPU B for the CC variant: A's classical outcome (``n_cond`` possible
    values) selects a conditional branch at one of three placements
    (``cc_target``, ablation axis 1):

      - ``"loader"``: a bank of ``n_cond`` single-layer phase gates applied right
        after encoding, *before* Conv (the earliest trainable point, since the
        bare one-hot encoder itself is untrainable).
      - ``"conv"``:   a bank of ``n_cond`` Conv filters (same windowed structure,
        different trained phases per condition).
      - ``"dense"``:  a bank of ``n_cond`` Dense unitaries (default, matches the
        original implementation).

    ``use_dense=False`` additionally drops the (unconditional or conditional)
    Dense stage everywhere except when ``cc_target="dense"`` is explicitly
    requested (Dense must exist for Dense to be the conditioning target).

    Returns ``P_B(y | x)`` of shape ``(b, n_cond, out_modes)``.
    """

    def __init__(self, d: int, n_cond: int, kernel_size: int = 2, add_modes: int = 2,
                 conv_circuit: str = "BS", dense_circuit: str = "BS",
                 use_dense: bool = True, cc_target: str = "dense"):
        super().__init__()
        if cc_target not in {"loader", "conv", "dense"}:
            raise ValueError("cc_target must be 'loader', 'conv' or 'dense'")
        if cc_target == "dense" and not use_dense:
            raise ValueError("cc_target='dense' requires use_dense=True (nothing to condition)")
        self.n_cond = n_cond
        self.cc_target = cc_target
        self.use_dense = use_dense

        if cc_target == "loader":
            self.loader_bank = RegisterLoaderBank(d, n_cond, conv_circuit)
            self.conv = RegisterConv(d, kernel_size, kernel_size, conv_circuit)
        else:
            self.loader_bank = None
            self.conv = None
            self.conv_bank = (RegisterConvBank(d, n_cond, kernel_size, kernel_size, conv_circuit)
                               if cc_target == "conv" else None)
            if cc_target != "conv":
                self.conv = RegisterConv(d, kernel_size, kernel_size, conv_circuit)

        self.pool = RegisterPool(d, kernel_size)

        if use_dense:
            if cc_target == "dense":
                self.bank = RegisterDenseBank(self.pool.new_d, n_cond, add_modes, dense_circuit)
                self.dense = None
                self.out_modes = self.bank.m
            else:
                self.bank = None
                self.dense = RegisterDense(self.pool.new_d, add_modes, dense_circuit)
                self.out_modes = self.dense.m
        else:
            self.bank = None
            self.dense = None
            self.out_modes = self.pool.new_d

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        b = rho.shape[0]

        if self.cc_target == "loader":
            rho_cond = self.loader_bank(rho)                      # (b, n_cond, d, d)
            u = self.conv.mesh(self.conv.phi).to(rho.dtype)       # (d, d), shared conv
            u_dag = u.conj().transpose(-1, -2)
            tmp = torch.einsum("ij,bnjk->bnik", u, rho_cond)
            evolved = torch.einsum("bnik,kl->bnil", tmp, u_dag)   # (b, n_cond, d, d)
        elif self.cc_target == "conv":
            evolved = self.conv_bank(rho)                          # (b, n_cond, d, d)
        else:  # "dense": conv is unconditional/shared
            u = self.conv.mesh(self.conv.phi).to(rho.dtype)
            u_dag = u.conj().transpose(-1, -2)
            rho_e = u @ rho.to(rho.dtype) @ u_dag                  # (b, d, d)
            evolved = rho_e.unsqueeze(1).expand(-1, self.n_cond, -1, -1)

        # pool each condition branch (RegisterPool works on (batch, d, d); flatten
        # the n_cond axis into batch, then reshape back)
        d = evolved.shape[-1]
        flat = evolved.reshape(b * self.n_cond, d, d)
        pooled_flat = self.pool(flat)                              # (b*n_cond, p, p)

        if not self.use_dense:
            p = pooled_flat.shape[-1]
            return marginal_readout(pooled_flat).reshape(b, self.n_cond, p)

        if self.cc_target == "dense":
            # self.bank already batches over n_cond internally (RegisterDenseBank
            # expects pooled shape (batch, p, p) and its OWN n_cond axis picks the
            # unitary); here the n_cond axis on `pooled_flat` (from A's outcome)
            # must line up 1:1 with the bank's n_cond axis (B's dense choice per
            # outcome), so reuse its forward directly on the (b*n_cond, p, p)
            # flattened batch is wrong (bank expects one pooled state and returns
            # ALL conditions). Instead pick out condition i's own dense per i.
            pooled = pooled_flat.reshape(b, self.n_cond, pooled_flat.shape[-2], pooled_flat.shape[-1])
            return self._dense_bank_paired(pooled)
        else:
            padded = self.dense._pad(pooled_flat)
            u = self.dense.mesh(self.dense.phi).to(padded.dtype)
            evolved_d = u @ padded @ u.conj().transpose(-1, -2)
            m = self.dense.m
            return marginal_readout(evolved_d).reshape(b, self.n_cond, m)

    def _dense_bank_paired(self, pooled: torch.Tensor) -> torch.Tensor:
        """``pooled``: (b, n_cond, p, p) — one pooled branch per A-outcome ``i``.
        ``self.bank.phi`` holds ``n_cond`` dense phase settings, one per outcome.
        Pairs branch ``i`` with dense unitary ``i`` (NOT an outer product): this
        reproduces the original ``ConditionalQPU`` semantics where A's outcome
        directly selects B's dense unitary.
        """
        b, n = pooled.shape[0], self.n_cond
        padded = self.bank._pad(pooled.reshape(b * n, pooled.shape[-2], pooled.shape[-1]))
        padded = padded.reshape(b, n, self.bank.m, self.bank.m)
        u = self.bank.mesh(self.bank.phi).to(padded.dtype)          # (n_cond, m, m)
        u_dag = u.conj().transpose(-1, -2)
        tmp = torch.einsum("nij,bnjk->bnik", u, padded)
        out = torch.einsum("bnik,nkl->bnil", tmp, u_dag)
        return marginal_readout(out.reshape(b * n, self.bank.m, self.bank.m)).reshape(b, n, self.bank.m)


class BiQPU(nn.Module):
    """QPU for bidirectional CC: conv+pool, exposes its own pooling marginal q,
    and a bank of dense layers conditioned on the PARTNER's pooling outcome.

    Returns (q, P_cond) with q of shape (b, p) and P_cond (b, n_cond, out_modes).
    """

    def __init__(self, d: int, n_cond: int, kernel_size: int = 2, add_modes: int = 2,
                 conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        self.conv = RegisterConv(d, kernel_size, kernel_size, conv_circuit)
        self.pool = RegisterPool(d, kernel_size)
        self.dense_bank = nn.ModuleList(
            [RegisterDense(self.pool.new_d, add_modes, dense_circuit) for _ in range(n_cond)]
        )
        self.out_modes = self.dense_bank[0].m
        self.p = self.pool.new_d

    def forward(self, rho: torch.Tensor):
        pooled = self.pool(self.conv(rho))                    # (b, p, p)
        q = pooled.diagonal(dim1=1, dim2=2).real.clamp_min(0)  # (b, p) own pooling marginal
        q = q / q.sum(1, keepdim=True).clamp_min(1e-12)
        cond = torch.stack([marginal_readout(dense(pooled)) for dense in self.dense_bank], dim=1)
        return q, cond                                        # cond: (b, n_cond, out_modes)


class DPQCNN(nn.Module):
    """Distributed Photonic QCNN.

    Args:
        dims: image dimensions (square).
        comm: "NC" (no communication), "CC" (one-way A->B classical feed-forward),
              or "CC2" (bidirectional/pooling-stage classical feed-forward).
        add_modes: empty modes added in each register's dense layer (ignored if
              ``use_dense=False``).
        num_classes: output classes.
        photons_a: 1 or 2; 2 gives QPU A a local 2-photon (ancilla) dense.
        use_dense: ablation axis 2 -- whether each QPU has a local Dense stage
              after Conv+Pool. ``False`` measures right after Pool.
        cc_target: ablation axis 1 (CC only) -- where A's outcome conditions
              QPU B: "loader" (pre-Conv phase gate), "conv" (Conv filter bank),
              or "dense" (Dense unitary bank, default / original behavior).
    """

    def __init__(self, dims: tuple[int, int], comm: str = "CC", *,
                 kernel_size: int = 2, add_modes: int = 2, num_classes: int = 2,
                 conv_circuit: str = "BS", dense_circuit: str = "BS",
                 photons_a: int = 1, use_dense: bool = True, cc_target: str = "dense"):
        super().__init__()
        if comm not in {"NC", "CC", "CC2"}:
            raise ValueError("comm must be 'NC', 'CC' or 'CC2'")
        if photons_a not in {1, 2}:
            raise ValueError("photons_a must be 1 or 2")
        if cc_target not in {"loader", "conv", "dense"}:
            raise ValueError("cc_target must be 'loader', 'conv' or 'dense'")
        self.comm = comm
        self.dims = dims
        self.photons_a = photons_a
        self.use_dense = use_dense
        self.cc_target = cc_target
        d = dims[0]

        self.encoder = RegisterEncoder(dims)

        if comm == "CC2":
            # symmetric: each QPU conditions on the partner's pooling outcome (p modes)
            p = d // kernel_size
            self.qpu_a = BiQPU(d, p, kernel_size, add_modes, conv_circuit, dense_circuit)
            self.qpu_b = BiQPU(d, p, kernel_size, add_modes, conv_circuit, dense_circuit)
            m_a, m_b = self.qpu_a.out_modes, self.qpu_b.out_modes
            self.m_a, self.m_b = m_a, m_b
            self.W = nn.Parameter(0.1 * torch.randn(num_classes, m_a, m_b))
            return

        # QPU A (rows). photons_a=2 injects an ancilla for local 2-photon interference.
        if photons_a == 2:
            from .multiphoton import QPU2Photon
            self.qpu_a = QPU2Photon(d, kernel_size, add_modes, conv_circuit, dense_circuit)
        else:
            self.qpu_a = QPU(d, kernel_size, add_modes, conv_circuit, dense_circuit,
                              use_dense=use_dense)
        m_a = self.qpu_a.out_modes

        if comm == "NC":
            self.qpu_b = QPU(d, kernel_size, add_modes, conv_circuit, dense_circuit,
                              use_dense=use_dense)
            m_b = self.qpu_b.out_modes
        else:  # CC: B is conditioned on A's m_a possible outcomes
            self.qpu_b = ConditionalQPU(d, m_a, kernel_size, add_modes,
                                        conv_circuit, dense_circuit,
                                        use_dense=use_dense, cc_target=cc_target)
            m_b = self.qpu_b.out_modes

        self.m_a, self.m_b = m_a, m_b
        # Trainable interpret tensor W[c, x, y].
        self.W = nn.Parameter(0.1 * torch.randn(num_classes, m_a, m_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho_x, rho_y = self.encoder(x)

        if self.comm == "CC2":
            q_a, pa_cond = self.qpu_a(rho_x)   # q_a (b,p_a); pa_cond (b, p_b, m_a)
            q_b, pb_cond = self.qpu_b(rho_y)   # q_b (b,p_b); pb_cond (b, p_a, m_b)
            # each QPU's effective marginal is conditioned on the PARTNER's pooling outcome
            p_a_eff = torch.einsum("nb,nbx->nx", q_b, pa_cond)   # (b, m_a)
            p_b_eff = torch.einsum("na,nay->ny", q_a, pb_cond)   # (b, m_b)
            joint = p_a_eff.unsqueeze(2) * p_b_eff.unsqueeze(1)
            return torch.einsum("bxy,cxy->bc", joint, self.W)

        p_a = self.qpu_a(rho_x)                        # (b, m_a)
        if self.comm == "NC":
            p_b = self.qpu_b(rho_y)                    # (b, m_b)
            joint = p_a.unsqueeze(2) * p_b.unsqueeze(1)        # (b, m_a, m_b)
        else:  # CC (one-way A->B)
            p_b_cond = self.qpu_b(rho_y)               # (b, m_a, m_b) = P_B(y|x)
            joint = p_a.unsqueeze(2) * p_b_cond                 # weight by P_A(x)

        logits = torch.einsum("bxy,cxy->bc", joint, self.W)
        return logits

    def extra_repr(self) -> str:
        return (f"comm={self.comm}, dims={self.dims}, m_a={self.m_a}, m_b={self.m_b}, "
                f"use_dense={self.use_dense}, cc_target={self.cc_target}")


__all__ = ["DPQCNN", "QPU", "ConditionalQPU", "BiQPU"]
