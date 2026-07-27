"""Multi-photon-per-QPU extension (Idea 1).

Injects an ancilla photon into ONE QPU's dense layer so that register performs
genuine 2-photon interference locally (state injection, a la Monbroussou et al.),
recovering the expressivity of the monolithic coherent dense WITHIN a single
chip -- at the cost of reintroducing (a controlled amount of) Hong-Ou-Mandel
bunching. This lets us map the core expressivity <-> hardware-robustness
trade-off of the whole project.

The 2-photon dense reuses the baseline SLOS amplitude engine. After pooling the
register holds 1 photon over ``p`` modes; we inject a 2nd photon into an added
mode, giving a 2-photon state over ``m = p + add_modes`` modes, and evolve it
through an m-mode interferometer.
"""

from __future__ import annotations

import io
import sys

import numpy as np
import torch
import torch.nn as nn

from ._baseline import (
    ComputationSpace,
    build_slos_graph,
    compute_amplitudes,
    generate_all_fock_states_list,
    get_circuit,
    marginalize_photon_presence,
)
from .blocks import RegisterConv, RegisterPool, _converter


class Register2PhotonDense(nn.Module):
    """Local dense with an injected ancilla photon -> 2-photon interference.

    Input: 1-photon data density matrix ``rho_p`` of shape ``(b, p, p)``.
    Output: 2-photon Fock-space density matrix ``(b, F, F)`` over ``m`` modes
    (F = number of 2-photon Fock states) plus the Fock-state ``keys``.
    """

    def __init__(self, p: int, add_modes: int = 2, circuit: str = "BS",
                 ancilla_mode: int | None = None):
        super().__init__()
        if add_modes < 1:
            raise ValueError("2-photon dense needs at least one added mode for the ancilla")
        self.p = p
        self.m = p + add_modes
        self.ancilla = p if ancilla_mode is None else ancilla_mode

        circ = get_circuit(self.m, circuit)
        self._conv = _converter(circ)
        self.phi = nn.Parameter(2 * np.pi * torch.rand(len(circ.get_parameters())))

        # input basis: data photon in mode x (0..p-1) + ancilla photon in ancilla mode
        self._input_states = [
            tuple(1 if k in (x, self.ancilla) else 0 for k in range(self.m))
            for x in range(p)
        ]
        original = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self._slos = build_slos_graph(m=self.m, n_photons=2,
                                          computation_space=ComputationSpace.FOCK)
        finally:
            sys.stdout = original
        self._slos.__class__.compute_amplitudes = compute_amplitudes
        self.keys = generate_all_fock_states_list(self.m, 2, true_order=True)

    def forward(self, rho_p: torch.Tensor) -> torch.Tensor:
        b = rho_p.shape[0]
        u = self._conv.to_tensor(self.phi)                    # (m, m) complex64
        amps = torch.stack([self._slos.compute_amplitudes(u, s)
                            for s in self._input_states]).squeeze(1)  # (p, F)
        u_evolve = amps.T                                     # (F, p)
        ue = u_evolve.unsqueeze(0).expand(b, -1, -1)
        rho_out = torch.bmm(torch.bmm(ue, rho_p.to(ue.dtype)),
                            ue.transpose(1, 2).conj())         # (b, F, F)
        return rho_out


def two_photon_marginal(rho_out: torch.Tensor, keys) -> torch.Tensor:
    """Per-mode occupation marginal (normalised to sum 1) for a 2-photon Fock rho."""
    probs = rho_out.diagonal(dim1=1, dim2=2).real.clamp_min(0)
    marg = marginalize_photon_presence(keys, probs)           # (b, m), rows ~ 2
    return marg / marg.sum(1, keepdim=True).clamp_min(1e-12)


def two_photon_bunching(rho_out: torch.Tensor, keys) -> torch.Tensor:
    """Prob mass on bunched (mode occupancy >=2) Fock states -> HOM reintroduced."""
    probs = rho_out.diagonal(dim1=1, dim2=2).real.clamp_min(0)
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-12)
    bunched = torch.tensor([1.0 if max(k) >= 2 else 0.0 for k in keys],
                           device=probs.device)
    return (probs * bunched).sum(1)                            # (b,)


class QPU2Photon(nn.Module):
    """One QPU with a 2-photon (ancilla-injected) local dense.

    conv -> pool -> 2-photon dense -> per-mode marginal. Exposes ``out_modes``
    (= m) and ``last_bunching`` (mean bunching prob of the last forward)."""

    def __init__(self, d: int, kernel_size: int = 2, add_modes: int = 2,
                 conv_circuit: str = "BS", dense_circuit: str = "BS"):
        super().__init__()
        self.conv = RegisterConv(d, kernel_size, kernel_size, conv_circuit)
        self.pool = RegisterPool(d, kernel_size)
        self.dense = Register2PhotonDense(self.pool.new_d, add_modes, dense_circuit)
        self.out_modes = self.dense.m
        self.last_bunching: float | None = None

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.conv(rho))
        rho_out = self.dense(pooled)
        with torch.no_grad():
            self.last_bunching = float(two_photon_bunching(rho_out, self.dense.keys).mean())
        return two_photon_marginal(rho_out, self.dense.keys)


__all__ = ["Register2PhotonDense", "QPU2Photon", "two_photon_marginal", "two_photon_bunching"]
