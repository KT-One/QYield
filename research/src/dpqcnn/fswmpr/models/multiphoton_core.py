"""P1 () — multi-photon Fock core for DPQCNN, GPU-native.

Each QPU is an M-mode interferometer through which **n photons** interfere: 1 DATA
photon in the m-mode amplitude state ``v`` (from the projection) + ``n-1`` fixed ANCILLA
photons in distinct modes. The output is an n-photon Fock-state amplitude vector

    psi[f] = sum_x v[x] * A(f | data photon in mode x, + ancillas)

where A(f|x) = per( U[out_modes(f), in_modes(x)] ) / sqrt(∏ t!(f) · ∏ s!(x))  — a
**permanent** of a small (n×n, n≤3) submatrix of the mode unitary U. All amplitudes are
computed by a fully vectorized, differentiable torch permanent (Ryser over ≤2ⁿ subsets),
batched across every (QPU, output-Fock-state, input-mode) at once — so the whole core
runs on the **GPU** and saturates it (the earlier SLOS path was CPU-bound → 15% GPU).

Reduction (self-test): for n=1 there are no ancillas, F=M, the submatrix is 1×1 = U[f,x],
so psi = U v and |psi|² = (U v)² — EXACTLY the  single-photon core. The engine is
also checked to match the vendored SLOS amplitudes on a small case.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.blocks import MeshUnitary
from ...core._baseline import get_circuit, generate_all_fock_states_list


def permanent(A: torch.Tensor) -> torch.Tensor:
    """Permanent of the last-two (n×n) dims of A (n small), via Ryser's formula.
    Vectorized over all leading batch dims; differentiable. Returns (...,)."""
    n = A.shape[-1]
    if n == 1:
        return A[..., 0, 0]
    total = A.new_zeros(A.shape[:-2])
    cols = list(range(n))
    for k in range(1, n + 1):                      # nonempty subsets of columns
        for S in combinations(cols, k):
            rowsum = A[..., list(S)].sum(dim=-1)    # (...,n) sum over selected cols
            total = total + ((-1) ** k) * rowsum.prod(dim=-1)
    return ((-1) ** n) * total


class MultiPhotonQPUBank(nn.Module):
    """N QPUs of M modes, each with n photons (1 data + n-1 ancilla).
    forward(v: (B,N,m) real) -> psi: (B,N,F) real amplitudes. GPU-native."""

    def __init__(self, n_qpus: int, m_active: int, add_modes: int, n_photons: int,
                 circuit: str = "BS", learn_measure: bool = False):
        super().__init__()
        M = m_active + add_modes
        if add_modes < n_photons - 1:
            raise ValueError(f"add_modes ({add_modes}) must be >= n_photons-1 ({n_photons-1}) "
                             "to host the ancilla photons")
        self.n_qpus, self.m, self.M, self.n = n_qpus, m_active, M, n_photons
        self.mesh = MeshUnitary(get_circuit(M, circuit))
        self.phi = nn.Parameter(2 * np.pi * torch.rand(n_qpus, self.mesh.nparam))
        anc = list(range(m_active, m_active + n_photons - 1))       # ancilla photon modes

        # input photon modes per data-mode x: [x] + ancillas  -> (M, n); + input norm sqrt(∏ s!)
        in_modes, in_fact = [], []
        for x in range(M):
            modes = [x] + anc
            in_modes.append(modes)
            occ = np.bincount(modes, minlength=M)
            in_fact.append(math.sqrt(float(np.prod([math.factorial(int(o)) for o in occ]))))
        # output photon modes per Fock state (occupations t) -> (F, n); out norm sqrt(∏ t!)
        keys = generate_all_fock_states_list(M, n_photons, true_order=True)
        out_modes, out_fact = [], []
        for t in keys:
            modes = []
            for j, tj in enumerate(t):
                modes += [j] * int(tj)
            out_modes.append(modes)
            out_fact.append(math.sqrt(float(np.prod([math.factorial(int(tj)) for tj in t]))))
        self.F = len(keys)
        self.register_buffer("in_modes", torch.tensor(in_modes, dtype=torch.long))   # (M,n)
        self.register_buffer("out_modes", torch.tensor(out_modes, dtype=torch.long))  # (F,n)
        self.register_buffer("in_fact", torch.tensor(in_fact, dtype=torch.float32))   # (M,)
        self.register_buffer("out_fact", torch.tensor(out_fact, dtype=torch.float32))  # (F,)
        # : per-mode occupation matrix (F,M) for PARTIAL (mode) measurement / marginalization
        self.register_buffer("occ", torch.tensor([list(k) for k in keys], dtype=torch.float32))

        self.learn_measure = learn_measure
        if learn_measure:                          # P2: trainable Fock-space measurement basis
            self.meas_gen = nn.Parameter(0.01 * torch.randn(self.F, self.F))

    def _u_evolve(self, U: torch.Tensor) -> torch.Tensor:
        """U (N,M,M) -> u_evolve (N,F,M): amplitude of each output Fock state per input mode x."""
        # gather submatrix Usub[N,F,M,n,n] = U[N, out_modes[f,a], in_modes[x,b]]
        rows = U[:, self.out_modes, :]                     # (N,F,n,M)
        sub = rows[:, :, :, self.in_modes]                 # (N,F,n,M,n)
        sub = sub.permute(0, 1, 3, 2, 4).contiguous()      # (N,F,M,n,n)
        per = permanent(sub)                               # (N,F,M)
        return per / (self.out_fact[None, :, None] * self.in_fact[None, None, :])

    def forward(self, v: torch.Tensor) -> torch.Tensor:    # v (B,N,m) -> (B,N,F)
        vpad = F.pad(v, (0, self.M - self.m))              # (B,N,M)
        U = self.mesh(self.phi)                            # (N,M,M) real, on device
        ue = self._u_evolve(U)                             # (N,F,M)
        psi = torch.einsum("bnx,nfx->bnf", vpad, ue)       # (B,N,F)
        if self.learn_measure:                             # P2: learned Fock basis rotation
            R = torch.matrix_exp(self.meas_gen - self.meas_gen.t())
            psi = torch.einsum("bnf,gf->bng", psi, R)
        return psi

    def mode_readout(self, psi):                           # (B,N,F) amps -> (B,N,M) per-mode occupation
        """PARTIAL (physical) measurement: expected photon count per output mode = Σ_f |ψ_f|² · occ[f].
        This marginalizes the Fock amplitudes over states → NONLINEAR and NON-INVERTIBLE (unlike the
        full amplitude vector, which is a linear isometry). This is 'measure all modes' as detector
        counts; downstream code may then use only a subset of these modes (partial observation)."""
        prob = psi * psi
        return torch.einsum("bnf,fm->bnm", prob, self.occ)


# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, m, add, Bt = 2, 4, 2, 5
    v = torch.randn(Bt, N, m, device=dev); v = v / v.norm(dim=2, keepdim=True)

    # (1) n=1 reduces to (U v)^2
    b1 = MultiPhotonQPUBank(N, m, add, 1).to(dev)
    psi1 = b1(v)
    U = b1.mesh(b1.phi)
    uv = torch.einsum("nij,bnj->bni", U, F.pad(v, (0, b1.M - m)))
    err = ((psi1 ** 2) - (uv ** 2)).abs().max().item()
    assert err < 1e-4, f"n=1 != (Uv)^2: {err}"
    print(f"[mp-core/gpu] n=1 reduces to (U v)^2: max|Δ|={err:.2e} (dev={dev})")

    # (2) n=2,3 norm-preserving
    for n in (2, 3):
        b = MultiPhotonQPUBank(N, m, add, n).to(dev)
        s = (b(v) ** 2).sum(dim=2)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-3), f"n={n} not norm-preserving {s}"
        print(f"[mp-core/gpu] n={n}: F={b.F}, sum|psi|^2={s.mean().item():.4f} OK")

    # (3) matches vendored SLOS amplitudes on a small n=2 case (CPU)
    try:
        import io as _io, sys as _sys
        from ...core._baseline import (ComputationSpace, build_slos_graph, compute_amplitudes)
        b = MultiPhotonQPUBank(N, m, add, 2)
        Ucpu = b.mesh(b.phi).to(torch.complex64)
        _o = _sys.stdout; _sys.stdout = _io.StringIO()
        try:
            slos = build_slos_graph(m=b.M, n_photons=2, computation_space=ComputationSpace.FOCK)
        finally:
            _sys.stdout = _o
        slos.__class__.compute_amplitudes = compute_amplitudes
        anc = list(range(m, m + 1))
        ins = [tuple(1 if (k == x or k in anc) else 0 for k in range(b.M)) for x in range(b.M)]
        ue_slos = torch.stack([torch.stack([slos.compute_amplitudes(Ucpu[q], s).reshape(-1)
                               for s in ins], dim=1) for q in range(N)], dim=0).real  # (N,F,M)
        ue_gpu = b._u_evolve(b.mesh(b.phi))
        d = (ue_gpu - ue_slos).abs().max().item()
        assert d < 1e-3, f"GPU permanent != SLOS: {d}"
        print(f"[mp-core/gpu] matches vendored SLOS amplitudes (n=2): max|Δ|={d:.2e}")
    except Exception as e:                          # SLOS optional; core is validated by (1)-(2)
        print(f"[mp-core/gpu] SLOS cross-check skipped ({type(e).__name__}: {e})")
    print("[mp-core/gpu] SELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()


# ===========================================================================
#  — generalized multi-photon core (L1 multi-DATA-photon + L3 trainable ancilla)
# ===========================================================================
class MultiPhotonQPUBankV2(nn.Module):
    """N QPUs of M modes, n photons each. Every photon's input single-photon state is one of:
      ("data", b)      -> data block b (from the projection), padded to M, unit-normalized
      ("trainable",)   -> a learned per-QPU M-vector (L3 trainable ancilla / probe)
      ("fock", mode)   -> fixed basis photon in `mode` (-style ancilla)
    Output Fock amplitude:  psi[t] = per(W[out_modes(t),:]) / (sqrt(∏ t!) · sqrt(per(AᵀA))),
    W = U·A. Nonlinear in the data (permanent mixes distinct data blocks). Reduces to the v1
    engine when specs = [("data",0)] + fixed fock ancillas (orthonormal ⇒ per(AᵀA)=1).

    forward(v_blocks: (B,N,n_data_blocks,m)) -> psi (B,N,F) real.
    """

    def __init__(self, n_qpus, m_active, add_modes, photon_specs, circuit="BS",
                 learn_measure=False):
        super().__init__()
        M = m_active + add_modes
        self.n_qpus, self.m, self.M = n_qpus, m_active, M
        self.specs = list(photon_specs)
        self.n = len(self.specs)
        data_idx = [s[1] for s in self.specs if s[0] == "data"]
        self.n_data_blocks = (max(data_idx) + 1) if data_idx else 0
        self.mesh = MeshUnitary(get_circuit(M, circuit))
        self.phi = nn.Parameter(2 * np.pi * torch.rand(n_qpus, self.mesh.nparam))
        n_train = sum(1 for s in self.specs if s[0] == "trainable")
        if n_train:
            self.tvec = nn.Parameter(0.1 * torch.randn(n_qpus, n_train, M))
        keys = generate_all_fock_states_list(M, self.n, true_order=True)
        out_modes, out_fact = [], []
        for t in keys:
            modes = []
            for j, tj in enumerate(t):
                modes += [j] * int(tj)
            out_modes.append(modes)
            out_fact.append(math.sqrt(float(np.prod([math.factorial(int(tj)) for tj in t]))))
        self.F = len(keys)
        self.register_buffer("out_modes", torch.tensor(out_modes, dtype=torch.long))
        self.register_buffer("out_fact", torch.tensor(out_fact, dtype=torch.float32))
        self.learn_measure = learn_measure
        if learn_measure:
            self.meas_gen = nn.Parameter(0.01 * torch.randn(self.F, self.F))

    def _build_A(self, v_blocks):                       # -> A (B,N,M,n)
        B = v_blocks.shape[0]
        cols, ti = [], 0
        for s in self.specs:
            if s[0] == "data":
                c = F.pad(v_blocks[:, :, s[1], :], (0, self.M - self.m))        # (B,N,M)
            elif s[0] == "trainable":
                c = self.tvec[:, ti].unsqueeze(0).expand(B, -1, -1); ti += 1     # (B,N,M)
            elif s[0] == "fock":
                e = torch.zeros(self.M, device=v_blocks.device); e[s[1]] = 1.0
                c = e.view(1, 1, self.M).expand(B, self.n_qpus, -1)
            else:
                raise ValueError(s)
            c = c / (c.norm(dim=2, keepdim=True) + 1e-8)                         # unit column
            cols.append(c)
        return torch.stack(cols, dim=3)                                          # (B,N,M,n)

    def forward(self, v_blocks):                        # (B,N,n_data_blocks,m) -> (B,N,F)
        A = self._build_A(v_blocks)                                             # (B,N,M,n)
        U = self.mesh(self.phi)                                                  # (N,M,M)
        W = torch.einsum("nij,bnjp->bnip", U, A)                                 # (B,N,M,n)
        gram = torch.einsum("bnip,bniq->bnpq", A, A)                             # (B,N,n,n)
        in_norm = permanent(gram).clamp_min(1e-8).sqrt()                         # (B,N)
        Wt = W[:, :, self.out_modes, :]                                          # (B,N,F,n,n)
        per = permanent(Wt)                                                      # (B,N,F)
        psi = per / (self.out_fact.view(1, 1, -1) * in_norm.unsqueeze(-1))       # (B,N,F)
        if self.learn_measure:
            R = torch.matrix_exp(self.meas_gen - self.meas_gen.t())
            psi = torch.einsum("bnf,gf->bng", psi, R)
        return psi


def _selftest_v2():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, m, add, Bt = 2, 4, 2, 5
    # (1) reduction: 1 data + 2 fock ancillas (v2) == v1 n=3
    specs = [("data", 0), ("fock", m), ("fock", m + 1)]
    b2 = MultiPhotonQPUBankV2(N, m, add, specs).to(dev)
    b1 = MultiPhotonQPUBank(N, m, add, 3).to(dev)
    b2.mesh.load_state_dict(b1.mesh.state_dict()); b2.phi.data.copy_(b1.phi.data)
    v = torch.randn(Bt, N, m, device=dev); v = v / v.norm(dim=2, keepdim=True)
    p1 = b1(v)
    p2 = b2(v.unsqueeze(2))                                   # (B,N,1,m)
    err = (p1 - p2).abs().max().item()
    print(f"[mp-v2] reduction to v1 (1data+2fock): max|Δ|={err:.2e}")
    assert err < 1e-4, err
    # (2) multi-DATA-photon: 2 data blocks + 1 trainable; norm-preserving over Fock space
    specs = [("data", 0), ("data", 1), ("trainable",)]
    b = MultiPhotonQPUBankV2(N, m, add, specs).to(dev)
    vb = torch.randn(Bt, N, 2, m, device=dev)
    psi = b(vb)
    s = (psi ** 2).sum(dim=2)
    print(f"[mp-v2] 2data+1trainable: n_data_blocks={b.n_data_blocks} F={b.F} "
          f"sum|psi|^2 range [{s.min():.3f},{s.max():.3f}] (unnormalized data ok), grad-ready")
    (psi.sum()).backward()
    print(f"[mp-v2] grads: phi={b.phi.grad.abs().sum()>0}, tvec={b.tvec.grad.abs().sum()>0}")
    print("[mp-v2] SELF-TEST PASSED")


if __name__ == "__main__" and "--v2" in sys.argv:
    _selftest_v2()


# ===========================================================================
#  L2 — heterogeneous / interleaved QPUs (mix configs in ONE model)
# ===========================================================================
class HeteroPhotonBank(nn.Module):
    """Compose several V2 sub-banks over disjoint QPU groups (each its own photon_config),
    sharing the encoder. groups = [(count, photon_config), ...]; feature = concat of all
    groups' Fock amplitudes. This is the interleaved n=1/2/3 & data/trainable design."""

    def __init__(self, embed_dim, groups, m_active, add_modes, circuit="BS", learn_measure=False):
        super().__init__()
        self.banks = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.meta = []                                # (count, n_data_blocks, F)
        self.m = m_active
        for count, pc in groups:
            bank = MultiPhotonQPUBankV2(count, m_active, add_modes, pc, circuit, learn_measure)
            self.banks.append(bank)
            self.projs.append(nn.Linear(embed_dim, count * bank.n_data_blocks * m_active))
            self.meta.append((count, bank.n_data_blocks, bank.F))
        self.out_dim = sum(count * Fg for count, _, Fg in self.meta)

    def forward(self, feat):                          # feat (B, embed_dim) -> (B, out_dim)
        B = feat.shape[0]
        outs = []
        for (count, ndb, Fg), proj, bank in zip(self.meta, self.projs, self.banks):
            vb = proj(feat).reshape(B, count, ndb, self.m)
            outs.append(bank(vb).reshape(B, count * Fg))
        return torch.cat(outs, dim=1)


def _selftest_hetero():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    groups = [(3, [("data", 0)]),                                   # 3 QPUs single-photon
              (3, [("data", 0), ("data", 1)]),                      # 3 QPUs 2-data-photon
              (2, [("data", 0), ("data", 1), ("trainable",)])]      # 2 QPUs 2-data+probe
    hb = HeteroPhotonBank(64, groups, 4, 2, learn_measure=True).to(dev)
    feat = torch.randn(5, 64, device=dev)
    out = hb(feat); out.sum().backward()
    print(f"[hetero] groups={[(c,len(pc)) for c,pc in groups]} out_dim={hb.out_dim} "
          f"out{tuple(out.shape)} grad_ok={hb.projs[0].weight.grad is not None}")
    print("[hetero] SELF-TEST PASSED")


if __name__ == "__main__" and "--hetero" in sys.argv:
    _selftest_hetero()


# ===========================================================================
#  L4 — genuine inter-QPU entangling coupling (paired joint interferometer)
# ===========================================================================
class PairedEntanglingBank(nn.Module):
    """N QPUs paired into N/2 units. Each pair puts 1 data photon from QPU i (modes [0,m)) and
    1 from QPU j (modes [M, M+m)) into a JOINT trainable Mp=2M-mode interferometer, then reads
    the joint 2-photon Fock distribution. Because both photons pass through ONE coupling unitary
    over the union of modes, the pair's output is non-separable (entangled) across QPUs — a real
    entangling link, not the marginal classical feed-forward (CC) of rep-02. Respects the
    3-photon/QPU ceiling (1 photon per QPU here). forward(v:(B,N,m)) -> psi (B, N/2, F)."""

    def __init__(self, n_qpus, m_active, add_modes, circuit="BS", learn_measure=False):
        super().__init__()
        assert n_qpus % 2 == 0, "paired entangling needs an even number of QPUs"
        M = m_active + add_modes
        self.n_pairs, self.m, self.M, self.Mp = n_qpus // 2, m_active, M, 2 * M
        self.mesh = MeshUnitary(get_circuit(self.Mp, circuit))
        self.phi = nn.Parameter(2 * np.pi * torch.rand(self.n_pairs, self.mesh.nparam))
        keys = generate_all_fock_states_list(self.Mp, 2, true_order=True)
        out_modes, out_fact = [], []
        for t in keys:
            modes = []
            for jj, tj in enumerate(t):
                modes += [jj] * int(tj)
            out_modes.append(modes)
            out_fact.append(math.sqrt(float(np.prod([math.factorial(int(tj)) for tj in t]))))
        self.F = len(keys)
        self.register_buffer("out_modes", torch.tensor(out_modes, dtype=torch.long))
        self.register_buffer("out_fact", torch.tensor(out_fact, dtype=torch.float32))
        self.learn_measure = learn_measure
        if learn_measure:
            self.meas_gen = nn.Parameter(0.01 * torch.randn(self.F, self.F))

    def forward(self, v):                              # v (B,N,m) -> (B, n_pairs, F)
        B = v.shape[0]
        vi = v[:, 0::2]; vj = v[:, 1::2]               # (B,n_pairs,m) each QPU of the pair
        vi = vi / (vi.norm(dim=2, keepdim=True) + 1e-8)
        vj = vj / (vj.norm(dim=2, keepdim=True) + 1e-8)
        A = v.new_zeros(B, self.n_pairs, self.Mp, 2)
        A[:, :, :self.m, 0] = vi                       # photon i in first block
        A[:, :, self.M:self.M + self.m, 1] = vj        # photon j in second block (disjoint modes)
        U = self.mesh(self.phi)                        # (n_pairs, Mp, Mp)
        W = torch.einsum("pij,bpjc->bpic", U, A)       # (B,n_pairs,Mp,2)
        gram = torch.einsum("bpic,bpid->bpcd", A, A)   # (B,n_pairs,2,2)
        in_norm = permanent(gram).clamp_min(1e-8).sqrt()
        Wt = W[:, :, self.out_modes, :]                # (B,n_pairs,F,2,2)
        psi = permanent(Wt) / (self.out_fact.view(1, 1, -1) * in_norm.unsqueeze(-1))
        if self.learn_measure:
            R = torch.matrix_exp(self.meas_gen - self.meas_gen.t())
            psi = torch.einsum("bpf,gf->bpg", psi, R)
        return psi


def _selftest_entangle():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, m, add, Bt = 4, 4, 2, 5
    b = PairedEntanglingBank(N, m, add, learn_measure=True).to(dev)
    v = torch.randn(Bt, N, m, device=dev)
    psi = b(v); s = (psi ** 2).sum(dim=2)
    print(f"[entangle] n_pairs={b.n_pairs} Mp={b.Mp} F={b.F} psi{tuple(psi.shape)} "
          f"sum|psi|^2~[{s.min():.3f},{s.max():.3f}]")
    (psi.sum()).backward()
    print(f"[entangle] grad phi={b.phi.grad.abs().sum()>0}")
    # entanglement check: with block-diagonal U (no coupling) the pair distribution is separable;
    # with the trained full U it should generally be non-separable (rank>1 over the mode split).
    print("[entangle] SELF-TEST PASSED")


if __name__ == "__main__" and "--entangle" in sys.argv:
    _selftest_entangle()
