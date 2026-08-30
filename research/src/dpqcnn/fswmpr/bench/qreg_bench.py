"""Path-A battery — "quantum-inspired regularization" mechanism study (GPU-optimized).

Central question (Path A framing): the photonic head's edge over a FREE classical control is a
REGULARIZATION benefit. If so, (a) tuned classical regularizers (weight decay / dropout / spectral
norm) should recover much of the gap [MAKE-OR-BREAK 2a], (b) an explicitly norm-preserving
ORTHOGONAL classical layer should match the photonic one [2b], (c) the benefit should not need a
*learned* unitary — a fixed random one should also help [2c], and (d) a FREE dense head's accuracy
should be non-monotonic in capacity (overfits when large) [1a].

All heads sit on a FROZEN cached backbone embedding (2048-d) → prototype (euclidean) few-shot head.
Only the small head trains, on cached embeddings.

GPU-optimization (math unchanged, protocol = META-BATCHED ProtoNet):
  * meta-batch: process `meta_batch` independent episodes per optimizer step as ONE big batched
    tensor → each CUDA kernel does meta_batch× more work (kills launch-overhead starvation).
  * ALL episode image indices are pre-sampled up front into GPU long tensors (vectorized argsort
    sampling) → zero per-step CPU RNG / host-device stalls in the training loop.
  * evaluation runs all `eval_episodes` at once (single batched forward).
  * TF32 tensor cores + cudnn.benchmark + high matmul precision.
  * optional torch.compile on the head.

Note: meta-batched training changes absolute numbers slightly vs the 1-episode/step reference, but
ALL heads share the identical protocol, so the RELATIVE (quantum vs classical vs regularized)
comparison — the whole point of this battery — is valid. Quantum's absolute is printed as a sanity
anchor against the reference (~81.7 SSL / ~79.6 jet).

Run:
  uv run python -m dpqcnn.fswmpr.bench.qreg_bench --feats resemb_ssl_jet_ep60_s42 resemb_resnet50_224_jet \
      --seeds 42 123 456 7 99 --tag qreg_full
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P
from ..models.protonet import _macro_recall_f1
from ..models.multiphoton_core import MultiPhotonQPUBank, MultiPhotonQPUBankV2
from ...core._baseline import generate_all_fock_states_list
from .novelty import auroc as _auroc, sample_class_images as _sample_class_images, evaluate_novelty as _evaluate_novelty


def perf_setup():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class RegHead(nn.Module):
    """One head. kind ∈ {baseline, quantum, classical, orthogonal, dense}.
    Regularizer knobs: dropout (input), spectral (classical only), train_phi (quantum only),
    n_photons (quantum only). weight_decay is applied by the optimizer, not here."""

    def __init__(self, kind, E, m=4, add_modes=1, n_photons=2, read_modes=2,
                 train_phi=True, dropout=0.0, spectral=False, hidden=1024, learn_measure=True,
                 photon_specs=None, readout="partial", n_pool=2, depth=2, act="relu", mlp_hidden=11,
                 readout_act="square"):
        super().__init__()
        self.kind = kind
        self.E = E
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.spectral = spectral
        self.readout_act = readout_act
        self.readout = readout
        if kind == "baseline":
            self.out_dim = E
            return
        if kind in ("qv2", "qfull"):
            # qv2: multi-DATA-photon V2 bank (genuine data-photon interference via a permanent).
            # qfull: V1 single-data-photon bank but FULL Fock |psi|^2 readout (richness control).
            specs = photon_specs or [("data", 0)]
            n_data = max([s[1] for s in specs if s[0] == "data"]) + 1
            self.n_data = n_data
            self.m = m
            self.n_qpus = E // (n_data * m)
            assert self.n_qpus * n_data * m == E, f"E={E} not divisible by n_data*m={n_data*m}"
            M = m + add_modes
            self.M = M
            n = len(specs)
            self.bankv2 = MultiPhotonQPUBankV2(self.n_qpus, m, add_modes, specs,
                                               C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            self.read = min(read_modes, M)
            if readout == "full":
                self.out_dim = self.n_qpus * self.bankv2.F
            else:
                keys = generate_all_fock_states_list(M, n, true_order=True)
                self.register_buffer("occ", torch.tensor([list(k) for k in keys], dtype=torch.float32))
                self.out_dim = self.n_qpus * self.read
            return
        assert E % m == 0
        self.m, self.n_qpus = m, E // m
        M = m + add_modes
        self.M = M
        self.read = min(read_modes, M)
        self.train_phi = train_phi
        if kind == "quantum":
            self.bank = MultiPhotonQPUBank(self.n_qpus, m, add_modes, n_photons,
                                           C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            if not train_phi:
                self.bank.phi.requires_grad_(False)
                if learn_measure:
                    self.bank.meas_gen.requires_grad_(False)
            self.out_dim = self.n_qpus * self.read
        elif kind == "classical":
            self.cW = nn.Parameter(0.1 * torch.randn(self.n_qpus, self.read, m, m))
            self.out_dim = self.n_qpus * self.read
        elif kind == "orthogonal":
            # quantum-inspired classical: per-QPU REAL orthogonal Q = expm(A - Aᵀ) (norm-preserving),
            # z' = Q z, feat = (z'²)[:read]. Exactly the n=1 single-photon analog with a trainable
            # real orthogonal "unitary" — the classical embodiment of the photonic structure.
            self.skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, m, m))
            self.out_dim = self.n_qpus * self.read
        elif kind == "asi":
            # ★ Direction 1 — Adaptive State Injection (differentiable branch-expectation reduction).
            # conv U → v'; measure `n_pool` pooling modes → Born gate g_o=|v'_o|² (+ rest bucket);
            # each outcome o selects a learned conditional unitary V^(o); read expected occupation:
            #   feat = Σ_o g_o(v) · (V^(o) v')²[:read]      (QUARTIC in data → escapes the
            # orthogonal/quadratic box; reduces to plain orthogonal head when all V^(o) are tied).
            self.n_pool = min(n_pool, m - 1)
            self.n_exp = self.n_pool + 1                       # pooling-mode experts + 1 "rest" expert
            self.conv_skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, m, m))
            self.exp_skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, self.n_exp, m, m))
            self.out_dim = self.n_qpus * self.read
        elif kind == "asi_cc":
            # ★ Direction 5 — Classical-Communication feedforward + trainable interpret.
            # QPUs paired (parent i, child j). PARENT's Born gate g_i selects the CHILD's conditional
            # unitary V_j^(o) (paper 2 Fig 1: one processor's measurement conditions another's gate).
            # Readout = [ parent self-gated asi ; child parent-gated (CC) ; trainable interpret(g) ].
            assert self.n_qpus % 2 == 0, "asi_cc needs an even number of QPUs"
            self.n_pairs = self.n_qpus // 2
            self.n_pool = min(n_pool, m - 1)
            self.n_exp = self.n_pool + 1
            self.conv_skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, m, m))
            self.exp_skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, self.n_exp, m, m))
            self.interpret = nn.Linear(2 * self.n_exp, self.read)     # f = Σ_k w_k P[pattern_k]
            self.out_dim = self.n_pairs * 3 * self.read
        elif kind == "asi_deep":
            # ★ Gate C — DEPTH-stacked adaptive injection (scale toward classically-hard regime).
            # L rounds of [conv U_l → Born gate g_l → gated mixture Σ_o g_l,o·(V_l^(o) v) → renorm];
            # final occupation readout. Depth L grows adaptive-circuit order. Does AUROC grow with L?
            self.n_pool = min(n_pool, m - 1)
            self.n_exp = self.n_pool + 1
            self.depth = depth
            self.conv_skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, depth, m, m))
            self.exp_skew = nn.Parameter(0.1 * torch.randn(self.n_qpus, depth, self.n_exp, m, m))
            self.out_dim = self.n_qpus * self.read
        elif kind == "dense":
            self.lin = nn.Linear(E, hidden)
            self.out_dim = hidden
        elif kind == "mlp":
            # matched-param per-QPU MLP with a STANDARD activation (relu/gelu) — control for
            # "is asi's edge just a generic nonlinearity?" Same block structure + ~param count as asi,
            # but not norm-preserving and not adaptive/gated (plain Linear→act→Linear per QPU).
            h = mlp_hidden
            self.W1 = nn.Parameter(torch.randn(self.n_qpus, m, h) / m ** 0.5)
            self.b1 = nn.Parameter(torch.zeros(self.n_qpus, h))
            self.W2 = nn.Parameter(torch.randn(self.n_qpus, h, self.read) / h ** 0.5)
            self.b2 = nn.Parameter(torch.zeros(self.n_qpus, self.read))
            self.act = act
            self.out_dim = self.n_qpus * self.read
        elif kind == "mlp_res":
            # fairer STRONG classical MLP: RESIDUAL (skip) so it can only ADD to the floor, never
            # collapse it — the classical analog of QYield's info-preserving design. feat = z + mlp(z),
            # output dim = input dim (m), W2 small-init so it starts ≈ the (norm-preserving) floor.
            h = mlp_hidden
            self.W1 = nn.Parameter(torch.randn(self.n_qpus, m, h) / m ** 0.5)
            self.b1 = nn.Parameter(torch.zeros(self.n_qpus, h))
            self.W2 = nn.Parameter(0.01 * torch.randn(self.n_qpus, h, m))
            self.b2 = nn.Parameter(torch.zeros(self.n_qpus, m))
            self.act = act
            self.out_dim = self.n_qpus * m
        else:
            raise ValueError(kind)

    def forward(self, e):
        if self.kind == "baseline":
            return e
        if self.kind == "dense":
            h = self.lin(self.dropout(e) if self.dropout else e)
            return h * h
        if self.kind in ("qv2", "qfull"):
            B = e.shape[0]
            vb = e.reshape(B, self.n_qpus, self.n_data, self.m)   # (B,N,n_data,m)
            if self.dropout is not None:
                vb = self.dropout(vb)
            psi = self.bankv2(vb)                                 # (B,N,F)
            if self.readout == "full":
                return (psi * psi).reshape(B, -1)
            occ = torch.einsum("bnf,fm->bnm", psi * psi, self.occ)  # (B,N,M) expected occupation
            return occ[:, :, :self.read].reshape(B, -1)
        B = e.shape[0]
        zc = e.reshape(B, self.n_qpus, self.m)
        zc = zc / (zc.norm(dim=2, keepdim=True) + 1e-8)
        if self.dropout is not None:
            zc = self.dropout(zc)
        if self.kind == "quantum":
            psi = self.bank(zc)                                          # photonic amplitude evolution (kept)
            if self.readout_act == "square":
                occ = self.bank.mode_readout(psi)                        # Born |ψ|² (native quantum activation)
            else:                                                        # keep ψ, swap ONLY the activation
                a = F.gelu(psi) if self.readout_act == "gelu" else F.relu(psi)
                occ = torch.einsum("bnf,fm->bnm", a, self.bank.occ)
            return occ[:, :, :self.read].reshape(B, -1)
        if self.kind == "classical":
            W = 0.5 * (self.cW + self.cW.transpose(-1, -2))
            if self.spectral:
                sv = torch.linalg.matrix_norm(W, ord=2, dim=(-2, -1)).clamp_min(1e-8)
                W = W / sv[..., None, None]
            return torch.einsum("bni,nkij,bnj->bnk", zc, W, zc).reshape(B, -1)
        if self.kind == "orthogonal":
            A = self.skew - self.skew.transpose(-1, -2)
            Q = torch.matrix_exp(A)
            zp = torch.einsum("nij,bnj->bni", Q, zc)
            return (zp * zp)[:, :, :self.read].reshape(B, -1)
        if self.kind == "asi":
            Uc = torch.matrix_exp(self.conv_skew - self.conv_skew.transpose(-1, -2))   # (N,m,m) conv
            vp = torch.einsum("nij,bnj->bni", Uc, zc)                                   # (B,N,m) unit
            g_pool = vp[:, :, :self.n_pool] ** 2                                        # (B,N,n_pool) Born
            g_rest = (1.0 - g_pool.sum(-1, keepdim=True)).clamp_min(0.0)                # (B,N,1)
            g = torch.cat([g_pool, g_rest], dim=-1)                                     # (B,N,n_exp)
            V = torch.matrix_exp(self.exp_skew - self.exp_skew.transpose(-1, -2))       # (N,n_exp,m,m)
            Vv = torch.einsum("noij,bnj->bnoi", V, vp)                                  # (B,N,n_exp,m)
            occ = (Vv * Vv)[:, :, :, :self.read]                                        # (B,N,n_exp,read)
            feat = torch.einsum("bno,bnor->bnr", g, occ)                                # (B,N,read) mixture
            return feat.reshape(B, -1)
        if self.kind == "asi_cc":
            Uc = torch.matrix_exp(self.conv_skew - self.conv_skew.transpose(-1, -2))     # (N,m,m)
            vp = torch.einsum("nij,bnj->bni", Uc, zc)                                    # (B,N,m)
            V = torch.matrix_exp(self.exp_skew - self.exp_skew.transpose(-1, -2))        # (N,n_exp,m,m)
            Vv = torch.einsum("noij,bnj->bnoi", V, vp)                                   # (B,N,n_exp,m)
            occ = (Vv * Vv)[:, :, :, :self.read]                                         # (B,N,n_exp,read)
            gp = vp[:, :, :self.n_pool] ** 2                                             # (B,N,n_pool)
            g = torch.cat([gp, (1.0 - gp.sum(-1, keepdim=True)).clamp_min(0.0)], -1)     # (B,N,n_exp) Born
            g_par, g_chi = g[:, 0::2], g[:, 1::2]                                        # (B,n_pairs,n_exp)
            occ_par, occ_chi = occ[:, 0::2], occ[:, 1::2]                                # (B,n_pairs,n_exp,read)
            feat_par = torch.einsum("bpo,bpor->bpr", g_par, occ_par)                     # parent self-gated
            feat_chi = torch.einsum("bpo,bpor->bpr", g_par, occ_chi)                     # ★ CC: PARENT gates CHILD
            f_int = self.interpret(torch.cat([g_par, g_chi], dim=-1))                    # trainable interpret
            return torch.cat([feat_par, feat_chi, f_int], dim=-1).reshape(B, -1)
        if self.kind == "asi_deep":
            v = zc                                                                       # (B,N,m) unit
            for l in range(self.depth):
                U = torch.matrix_exp(self.conv_skew[:, l] - self.conv_skew[:, l].transpose(-1, -2))
                v = torch.einsum("nij,bnj->bni", U, v)                                   # conv
                gp = v[:, :, :self.n_pool] ** 2
                g = torch.cat([gp, (1.0 - gp.sum(-1, keepdim=True)).clamp_min(0.0)], -1) # (B,N,n_exp) Born
                Vl = torch.matrix_exp(self.exp_skew[:, l] - self.exp_skew[:, l].transpose(-1, -2))
                Vv = torch.einsum("noij,bnj->bnoi", Vl, v)                               # (B,N,n_exp,m)
                v = torch.einsum("bno,bnoi->bni", g, Vv)                                 # gated mixture
                v = v / (v.norm(dim=2, keepdim=True) + 1e-8)                             # renorm for next layer
            return (v * v)[:, :, :self.read].reshape(B, -1)
        if self.kind == "mlp":
            h1 = torch.einsum("bnm,nmh->bnh", zc, self.W1) + self.b1
            h1 = F.gelu(h1) if self.act == "gelu" else F.relu(h1)
            out = torch.einsum("bnh,nhr->bnr", h1, self.W2) + self.b2
            return out.reshape(B, -1)
        if self.kind == "mlp_res":
            h1 = torch.einsum("bnm,nmh->bnh", zc, self.W1) + self.b1
            h1 = F.gelu(h1) if self.act == "gelu" else F.relu(h1)
            delta = torch.einsum("bnh,nhm->bnm", h1, self.W2) + self.b2
            return (zc + delta).reshape(B, -1)                                       # residual: floor + learned add


def pools(feat_name, device):
    E = torch.tensor(np.load(C.PROCESSED_DIR / f"{feat_name}.npy"), dtype=torch.float32, device=device)
    X, y, split, meta = P.load(img_size=224)
    cls = meta["classes"]
    base_by = {c: np.where((split == "base_train") & (y == cls.index(c)))[0] for c in C.BASE_CLASSES}
    novel_by = {c: np.where((split == "novel_pool") & (y == cls.index(c)))[0] for c in C.NOVEL_CLASSES}
    return E, base_by, novel_by


def build_head(cfg, E):
    return RegHead(cfg["kind"], E, m=cfg.get("m", 4), add_modes=cfg.get("add_modes", 1),
                   n_photons=cfg.get("n_photons", 2), read_modes=cfg.get("read_modes", 2),
                   train_phi=cfg.get("train_phi", True), dropout=cfg.get("dropout", 0.0),
                   spectral=cfg.get("spectral", False), hidden=cfg.get("hidden", 1024),
                   learn_measure=cfg.get("learn_measure", True),
                   photon_specs=cfg.get("photon_specs"), readout=cfg.get("readout", "partial"),
                   n_pool=cfg.get("n_pool", 2), depth=cfg.get("depth", 2),
                   act=cfg.get("act", "relu"), mlp_hidden=cfg.get("mlp_hidden", 11),
                   readout_act=cfg.get("readout_act", "square"))


def make_train_episodes(base_by, k, q, count, rng, device):
    """Pre-sample `count` 3-way (k+q)-shot BASE episodes. Support/query grouped by class:
    si=(count, 3k) as [c0*k,c1*k,c2*k]; qi=(count, 3q). Returns GPU long tensors."""
    sis, qis = [], []
    for c in C.BASE_CLASSES:
        picks = _sample_class_images(base_by[c], count, k + q, rng)   # (count, k+q)
        sis.append(picks[:, :k]); qis.append(picks[:, k:])
    si = np.concatenate(sis, axis=1); qi = np.concatenate(qis, axis=1)
    return (torch.as_tensor(si, dtype=torch.long, device=device),
            torch.as_tensor(qi, dtype=torch.long, device=device))


def meta_train(cfg, seed, device, E, base_by, updates, meta_batch, k, train_q, compile_head=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_head(cfg, E.shape[1]).to(device)
    npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if npar == 0:
        net.eval()
        return net, 0
    fwd = net
    if compile_head:
        try:
            fwd = torch.compile(net, mode="max-autotune")
        except Exception:
            fwd = net
    net.train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                           lr=1e-3, weight_decay=cfg.get("weight_decay", 0.0))
    n_way = 3
    si_all, qi_all = make_train_episodes(base_by, k, train_q, updates * meta_batch, rng, device)
    ql = torch.arange(n_way, device=device).repeat_interleave(train_q).repeat(meta_batch)
    for u in range(updates):
        s = si_all[u * meta_batch:(u + 1) * meta_batch]          # (mb, 3k)
        qq = qi_all[u * meta_batch:(u + 1) * meta_batch]         # (mb, 3q)
        se = fwd(E[s.reshape(-1)]).reshape(meta_batch, n_way * k, -1)
        qe = fwd(E[qq.reshape(-1)]).reshape(meta_batch, n_way * train_q, -1)
        protos = se.reshape(meta_batch, n_way, k, -1).mean(2)     # (mb, n_way, D)
        d = torch.cdist(qe, protos)                              # (mb, 3q, n_way)
        loss = F.cross_entropy((-d).reshape(-1, n_way), ql)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, npar


@torch.no_grad()
def evaluate(net, device, E, novel_by, n_way, k, q, n_episodes, eval_seed, exclude=None, q_ood=None):
    """Thin wrapper → the UNIFIED bench.novelty.evaluate_novelty (single eval seed, mindist headline).
    Photonic heads (here) and classical baselines (bench/baselines.py) now share ONE open-set episodic
    evaluator, so all rows are drawn from identical episodes per seed and are strictly comparable."""
    pool = [c for c in C.NOVEL_CLASSES if c not in (exclude or ())]
    def embed(idxs):
        return net(E[torch.as_tensor(idxs, device=device)])
    r = _evaluate_novelty(embed, novel_by, novel_classes=pool, seeds=[eval_seed],
                          n_way=n_way, k=k, q=q, n_ep=n_episodes, scorers=("mindist",))
    return {"acc": r["acc"]["mean"], "ci95": r["acc"]["ci95"],
            "macro_f1": float("nan"), "auroc": r["auroc_mindist"]["mean"]}


def head_configs():
    cfgs = {
        "baseline":           {"kind": "baseline"},     # identity → prototype = frozen-ResNet50 floor
        "multi_photon":      {"kind": "quantum"},
        "classical_free":     {"kind": "classical"},
        "orthogonal":         {"kind": "orthogonal"},
        "quantum_n1":         {"kind": "quantum", "n_photons": 1},
        "quantum_fixedphi":   {"kind": "quantum", "train_phi": False, "learn_measure": False},
    }
    for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
        cfgs[f"classical_wd{wd:g}"] = {"kind": "classical", "weight_decay": wd}
    for dp in [0.3, 0.5]:
        cfgs[f"classical_drop{dp:g}"] = {"kind": "classical", "dropout": dp}
    cfgs["classical_spectral"] = {"kind": "classical", "spectral": True}
    for h in [64, 256, 1024, 4096]:
        cfgs[f"dense{h}"] = {"kind": "dense", "hidden": h}
    # --- Option 1: genuine multi-DATA-photon interference (escape the orthogonal/quadratic box) ---
    D2 = [("data", 0), ("data", 1)]
    cfgs["q2data_partial"] = {"kind": "qv2", "photon_specs": D2, "readout": "partial"}
    cfgs["q2data_full"]    = {"kind": "qv2", "photon_specs": D2, "readout": "full"}
    cfgs["q2data_probe"]   = {"kind": "qv2", "photon_specs": D2 + [("trainable",)], "readout": "partial"}
    cfgs["q1_full"]        = {"kind": "qfull", "photon_specs": [("data", 0)], "readout": "full"}  # readout-richness control
    # --- Direction 1: Adaptive State Injection (Gate A: does adaptivity beat the orthogonal layer?) ---
    cfgs["asi_p1"] = {"kind": "asi", "n_pool": 1}
    cfgs["asi_p2"] = {"kind": "asi", "n_pool": 2}
    cfgs["asi_p3"] = {"kind": "asi", "n_pool": 3}
    # --- Direction 5: CC feedforward + trainable interpret (parent gate selects child unitary) ---
    cfgs["asi_cc_p1"] = {"kind": "asi_cc", "n_pool": 1}
    cfgs["asi_cc_p2"] = {"kind": "asi_cc", "n_pool": 2}
    # --- Gate C: depth-stacked adaptive injection (does novelty AUROC grow with adaptive depth?) ---
    for L in [1, 2, 3, 4]:
        cfgs[f"asi_L{L}"] = {"kind": "asi_deep", "n_pool": 3, "depth": L}
    # generic-activation controls (param-matched to asi_p3 ~41k): is the edge just a nonlinearity?
    cfgs["mlp_relu"] = {"kind": "mlp", "act": "relu", "mlp_hidden": 11}
    cfgs["mlp_gelu"] = {"kind": "mlp", "act": "gelu", "mlp_hidden": 11}
    # generic-activation controls param-matched to QYield/asi_L4 (~164k): mlp_hidden=45 →
    # n_qpus*(7*45+2) = 512*317 = 162,304 params (~99% of asi_L4's 163,840). Parameter-fairness
    # control: does a classical MLP sized to QYield beat the ~66 classical band? (expected: no)
    cfgs["mlp_relu_164k"] = {"kind": "mlp", "act": "relu", "mlp_hidden": 45}
    cfgs["mlp_gelu_164k"] = {"kind": "mlp", "act": "gelu", "mlp_hidden": 45}
    # fairest STRONG classical MLP: residual/skip (feat = z + mlp(z)) → can only add to the floor.
    # h=35 → n_qpus*(9*35+4)=512*319=163,328 params (~QYield 164k). The honest "SOTA + strong MLP".
    cfgs["mlp_res_relu"] = {"kind": "mlp_res", "act": "relu", "mlp_hidden": 35}
    cfgs["mlp_res_gelu"] = {"kind": "mlp_res", "act": "gelu", "mlp_hidden": 35}
    # SAME photonic quantum head, ONLY the Born |ψ|² activation swapped for relu/gelu:
    cfgs["quantum_relu"] = {"kind": "quantum", "readout_act": "relu"}
    cfgs["quantum_gelu"] = {"kind": "quantum", "readout_act": "gelu"}
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["resemb_ssl_jet_ep60_s42", "resemb_resnet50_224_jet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 7, 99])
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--updates", type=int, default=1000, help="optimizer steps (meta-batched)")
    ap.add_argument("--meta-batch", type=int, default=16, help="episodes per optimizer step")
    ap.add_argument("--train-q", type=int, default=15)
    ap.add_argument("--eval-q", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--tasks", nargs="+", default=["3w5s"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tag", default="qreg")
    args = ap.parse_args()

    perf_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cfgs = head_configs()
    names = args.heads or list(all_cfgs.keys())
    task_map = {"3w5s": (3, 5), "3w10s": (3, 10), "5w5s": (5, 5)}
    tasks = [(t, *task_map[t]) for t in args.tasks]

    out_all = {}
    for feat in args.feats:
        E, base_by, novel_by = pools(feat, device)
        print(f"\n===== FEAT {feat} E={tuple(E.shape)} heads={len(names)} seeds={args.seeds} "
              f"updates={args.updates} mb={args.meta_batch} =====", flush=True)
        res = {n: {t[0]: [] for t in tasks} for n in names}
        res_au = {n: {t[0]: [] for t in tasks} for n in names}
        nparam_of = {}
        for name in names:
            cfg = all_cfgs[name]
            t0 = time.time()
            for seed in args.seeds:
                net, npar = meta_train(cfg, seed, device, E, base_by, args.updates,
                                       args.meta_batch, 5, args.train_q, args.compile)
                nparam_of[name] = npar
                for tname, nw, k in tasks:
                    r = evaluate(net, device, E, novel_by, nw, k, args.eval_q, args.eval_episodes, seed)
                    res[name][tname].append(r["acc"])
                    res_au[name][tname].append(r["auroc"])
            line = " | ".join(f"{t[0]} {np.mean(res[name][t[0]]):.2f}±"
                              f"{1.96*np.std(res[name][t[0]],ddof=1)/np.sqrt(len(args.seeds)):.2f}"
                              f" AUROC {np.nanmean(res_au[name][t[0]]):.2f}" for t in tasks)
            print(f"  [{name:20s} p={nparam_of[name]:>8d}] {line}  ({time.time()-t0:.0f}s)", flush=True)

        summary = {}
        for name in names:
            summary[name] = {"n_params": nparam_of[name]}
            for tname, _, _ in tasks:
                v = np.array(res[name][tname]); a = np.array(res_au[name][tname])
                sci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                aci = 1.96 * np.nanstd(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
                summary[name][tname] = {"mean": float(v.mean()), "ci95": float(sci), "per_seed": v.tolist(),
                                        "auroc": float(np.nanmean(a)), "auroc_ci95": float(aci),
                                        "auroc_per_seed": a.tolist()}
        out_all[feat] = summary

    outp = C.RESULTS_DIR / f"{args.tag}.json"
    outp.write_text(json.dumps({"config": vars(args), "results": out_all}, indent=2))
    print(f"\n[qreg] saved {outp}", flush=True)


if __name__ == "__main__":
    main()
