"""Spatial-patch distributed model (Idea 1, ).

Instead of the OG row/col one-hot encoder (which builds ONE global entangled
state and hands each QPU a lossy *marginal* after partial trace), each QPU here
independently **amplitude-encodes its own contiguous spatial patch** of the raw
pixels into a single-photon register. Each QPU therefore holds *real, lossless*
data about its region, so classical communication + the interpret tensor can
recover cross-region structure — the correlation-only failure of the row/col
split is expected to disappear when a good spatial cut exists.

Two-QPU version (parallels the row/col DP for a fair comparison): the image is
split into two halves (columns or rows); each half's pixels are amplitude-encoded
into a 1-photon state over ``m = d * d/2`` modes. The rest reuses the existing
single-photon QPU blocks and interpret tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import ConditionalQPU, QPU


class PatchEncoder(nn.Module):
    """Split a d×d image into two contiguous halves and amplitude-encode each
    half's pixels into a 1-photon density matrix (b, m, m), m = d*(d/2)."""

    def __init__(self, dims: tuple[int, int], split: str = "cols"):
        super().__init__()
        if dims[0] != dims[1]:
            raise NotImplementedError("square images only")
        if split not in {"cols", "rows"}:
            raise ValueError("split must be 'cols' or 'rows'")
        self.d = dims[0]
        self.split = split
        self.modes = self.d * (self.d // 2)

    def _encode(self, patch: torch.Tensor) -> torch.Tensor:
        v = patch.reshape(patch.shape[0], -1)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
        v = v.to(torch.complex64)
        return v.unsqueeze(2) @ v.unsqueeze(1).conj()   # (b, m, m), rank-1

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.squeeze(1)
        h = self.d // 2
        if self.split == "cols":
            a, b = x[:, :, :h], x[:, :, h:]
        else:
            a, b = x[:, :h, :], x[:, h:, :]
        return self._encode(a), self._encode(b)


class PatchDPQCNN(nn.Module):
    """Two-QPU distributed model over spatial patches (NC or CC).

    Same interpret-tensor head as :class:`DPQCNN`; only the encoder differs
    (independent per-patch amplitude encoding instead of row/col partial trace).
    """

    def __init__(self, dims: tuple[int, int], comm: str = "NC", *, split: str = "cols",
                 add_modes: int = 2, num_classes: int = 2,
                 conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        if comm not in {"NC", "CC"}:
            raise ValueError("comm must be 'NC' or 'CC'")
        self.comm = comm
        self.encoder = PatchEncoder(dims, split)
        m = self.encoder.modes

        self.qpu_a = QPU(m, 2, add_modes, conv_circuit, dense_circuit)
        m_a = self.qpu_a.out_modes
        if comm == "NC":
            self.qpu_b = QPU(m, 2, add_modes, conv_circuit, dense_circuit)
            m_b = self.qpu_b.out_modes
        else:
            self.qpu_b = ConditionalQPU(m, m_a, 2, add_modes, conv_circuit, dense_circuit)
            m_b = self.qpu_b.out_modes
        self.m_a, self.m_b = m_a, m_b
        self.W = nn.Parameter(0.1 * torch.randn(num_classes, m_a, m_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho_a, rho_b = self.encoder(x)
        p_a = self.qpu_a(rho_a)
        if self.comm == "NC":
            p_b = self.qpu_b(rho_b)
            joint = p_a.unsqueeze(2) * p_b.unsqueeze(1)
        else:
            p_b_cond = self.qpu_b(rho_b)
            joint = p_a.unsqueeze(2) * p_b_cond
        return torch.einsum("bxy,cxy->bc", joint, self.W)

    def extra_repr(self) -> str:
        return f"comm={self.comm}, split={self.encoder.split}, m_a={self.m_a}, m_b={self.m_b}"


__all__ = ["PatchEncoder", "PatchDPQCNN"]
