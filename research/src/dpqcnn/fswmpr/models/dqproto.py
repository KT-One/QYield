"""OURS — DPQCNN: standalone Distributed Quantum ProtoNet (architecture.md §3.3).

A TOTALLY SEPARATE, self-contained model (no shared/frozen backbone): its own
small conv feature extractor -> trainable projection to D_total photonic modes
-> split across N small single-photon QPUs (NC; CC optional) -> per-QPU measured
marginals concatenated into a quantum embedding -> Euclidean PROTOTYPE metric
(no W outer-product tensor). Trained end-to-end episodically on BASE, evaluated
on the SAME shared novel episode files as every other model.

Photonic core reuses the merlin-free real single-photon fast path
(core.blocks.RegisterDense / MeshUnitary): for 1 photon a linear-optical circuit
acts on the m-mode 1-photon subspace as its m x m mode unitary U, so a QPU is
rho -> U rho U^T with measurement marginal = diag(U rho U^T).

Run: python -m dpqcnn.experiments.fswmpr_dqproto --n-qpus 8 --comm NC --seed 42
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
from .protonet import proto_predict, evaluate


def setup_perf():
    """GPU-saturation flags (Blackwell): TF32 matmuls, cuDNN autotune, high matmul precision."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class ConvEncoder(nn.Module):
    """Own small conv feature extractor (standalone; trained with the model)."""

    def __init__(self, embed_dim=64, in_ch=1, img_size=42, n_conv=3):
        super().__init__()
        ch = [32, 64, 64, 128, 128, 256][:max(1, n_conv)]
        layers, c = [], in_ch
        for o in ch:
            layers += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = o
        self.features = nn.Sequential(*layers)
        self.embed_dim = embed_dim
        self.proj = nn.Linear(ch[-1], embed_dim)

    def forward(self, x):
        h = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.proj(h)


class PatchAmplitudeEncoder(nn.Module):
    """: FIXED (non-learned) direct encoding — no conv, no linear.
    image (B,1,H,W) -> per-patch raw-pixel vectors (B, N_patches, m=patch^2). Pads H,W up to a
    multiple of `patch`. Each patch becomes one cheap QPU's amplitude input (normalized in the
    bank). This is the whole 'classical front-end': a reshape. All learning is in the quantum layer."""

    def __init__(self, img_size=42, patch=4):
        super().__init__()
        self.patch = patch
        self.pad = (patch - img_size % patch) % patch
        self.grid = (img_size + self.pad) // patch
        self.N = self.grid * self.grid          # number of QPUs (one per patch)
        self.m = patch * patch                  # modes per QPU (pixels per patch)

    def forward(self, x):                       # (B,1,H,W) -> (B, N, m)
        if self.pad:
            x = F.pad(x, (0, self.pad, 0, self.pad))
        p = x.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)  # (B,1,gh,gw,ph,pw)
        return p.contiguous().view(x.shape[0], self.N, self.m)


class InterpretHead(nn.Module):
    """ E3: class-agnostic factorized-bilinear cross-QPU aggregation → embedding.
    Restores the joint (multiplicative) cross-QPU correlations of the old interpret tensor W,
    but (a) outputs an EMBEDDING (no class axis → works for novel few-shot classes) and (b) is
    a LOW-RANK compact-bilinear approx (linear params, no M^N blowup that collapsed the N-ring W).
    e = (Σ_i A·g_i) ⊙ (Σ_i B·g_i) contains all pairwise cross-QPU products Σ_ij (A g_i)⊙(B g_j)."""

    def __init__(self, F, out=64, r=16):
        super().__init__()
        self.reduce = nn.Linear(F, r)
        self.A = nn.Linear(r, out, bias=False)
        self.B = nn.Linear(r, out, bias=False)

    def forward(self, feats):                    # (B,N,F) -> (B,out)
        g = torch.tanh(self.reduce(feats))       # (B,N,r)
        U = self.A(g).sum(1); V = self.B(g).sum(1)   # (B,out) each: additive over QPUs
        return U * V                             # multiplicative => cross-QPU joint (bilinear)


class HierarchicalPatchQCNN(nn.Module):
    """ E2: coarse→fine conditional QCNN. Layer-1 = LARGE patches -> coarse context per
    region; a classical feed-forward (CC) map turns each coarse feature into a context vector that
    CONDITIONS the Layer-2 fine (SMALL) patches inside it — injected as the fine QPU's 2nd DATA
    photon. Gives the quantum front-end a multi-scale receptive field. Output = fine-layer Fock amps."""

    def __init__(self, img_size=42, patch_large=6, patch_small=3, add_modes=2,
                 learn_measure=True, circuit="BS", interpret=0):
        super().__init__()
        from .multiphoton_core import MultiPhotonQPUBankV2
        self.enc1 = PatchAmplitudeEncoder(img_size, patch_large)
        self.enc2 = PatchAmplitudeEncoder(img_size, patch_small)
        self.bank1 = MultiPhotonQPUBankV2(self.enc1.N, self.enc1.m, add_modes,
                                          [("data", 0), ("trainable",)], circuit, learn_measure)
        self.cond = nn.Linear(self.bank1.F, self.enc2.m)     # coarse feature -> fine-mode context (CC)
        self.bank2 = MultiPhotonQPUBankV2(self.enc2.N, self.enc2.m, add_modes,
                                          [("data", 0), ("data", 1)], circuit, learn_measure)
        g1, g2 = self.enc1.grid, self.enc2.grid
        r = g2 // g1
        parent = [ (i // r) * g1 + (j // r) for i in range(g2) for j in range(g2) ]
        self.register_buffer("parent", torch.tensor(parent, dtype=torch.long))  # (N2,)
        self.interp = InterpretHead(self.bank2.F, interpret) if interpret else None
        self.out_dim = interpret if interpret else self.enc2.N * self.bank2.F
        self.m, self.add_modes, self.n_qpus = self.enc2.m, add_modes, self.enc2.N

    def forward(self, x):                                    # (B,1,H,W) -> (B, out_dim)
        B = x.shape[0]
        coarse = self.bank1(self.enc1(x).unsqueeze(2))       # (B,N1,F1)
        ctx = self.cond(coarse)[:, self.parent]              # (B,N2,m2) coarse context per fine patch
        fine = self.enc2(x)                                  # (B,N2,m2)
        vb = torch.stack([fine, ctx], dim=2)                 # (B,N2,2,m2): patch + coarse context
        psi = self.bank2(vb)                                 # (B,N2,F2)
        return self.interp(psi) if self.interp is not None else psi.reshape(B, -1)


def _conv_stem(n_conv, in_ch=1):
    """Shared conv stem builder (no GAP): n_conv [conv3-BN-ReLU-maxpool2] blocks -> (features, out_ch)."""
    ch = [32, 64, 64, 128, 128, 256][:max(1, n_conv)]
    layers, c = [], in_ch
    for o in ch:
        layers += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o),
                   nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        c = o
    return nn.Sequential(*layers), ch[-1]


class SpatialConvEncoder(nn.Module):
    """ Option B (#1): EXACT 1-1 conv->QPU encoding. Conv stem (NO global pool) -> 1x1 conv
    to m channels -> adaptive-pool to a grid x grid map -> (B, N=grid*grid, m). Each spatial cell is
    one QPU; its m channel values ARE that QPU's m mode amplitudes (bijective, no lossy Linear
    bottleneck and no spatial collapse — the two things that capped gs_c2)."""

    def __init__(self, img_size=42, m_modes=4, grid=6, n_conv=2, in_ch=1):
        super().__init__()
        self.features, out_ch = _conv_stem(n_conv, in_ch)
        self.to_modes = nn.Conv2d(out_ch, m_modes, 1)     # 1x1 -> exactly m channels per cell
        self.grid, self.m, self.N = grid, m_modes, grid * grid

    def forward(self, x):                                 # (B,1,H,W) -> (B, N, m)
        h = self.to_modes(self.features(x))               # (B, m, H', W')
        h = F.adaptive_avg_pool2d(h, self.grid)           # (B, m, grid, grid)
        return h.flatten(2).transpose(1, 2).contiguous()  # (B, grid*grid, m)


class SpatialHierConvQCNN(nn.Module):
    """ Option B (#2): spatial 1-1 conv encoding + 2-layer coarse->fine conditional QCNN.
    A shared conv stem feeds two 1x1 heads: a COARSE map (small grid, large receptive field) and a
    FINE map (large grid). Layer-1 QPUs run on the coarse cells -> a learned CC feed-forward turns
    each coarse Fock feature into a context vector that CONDITIONS the layer-2 fine-cell QPUs
    (injected as the fine QPU's 2nd data photon). Same conditioning idea as HierarchicalPatchQCNN,
    but on CONV features (1-1) instead of raw pixels."""

    def __init__(self, img_size=42, m_modes=4, grid_coarse=2, grid_fine=6, n_conv=2,
                 add_modes=2, learn_measure=True, circuit="BS", in_ch=1):
        super().__init__()
        from .multiphoton_core import MultiPhotonQPUBankV2
        assert grid_fine % grid_coarse == 0, "grid_fine must be a multiple of grid_coarse"
        self.features, out_ch = _conv_stem(n_conv, in_ch)
        self.to_modes_c = nn.Conv2d(out_ch, m_modes, 1)   # coarse cell -> m modes
        self.to_modes_f = nn.Conv2d(out_ch, m_modes, 1)   # fine cell   -> m modes
        self.gc, self.gf, self.m, self.add_modes = grid_coarse, grid_fine, m_modes, add_modes
        # layer-1 coarse QPUs: 1 data photon (coarse cell) + 1 trainable probe
        self.bank1 = MultiPhotonQPUBankV2(grid_coarse * grid_coarse, m_modes, add_modes,
                                          [("data", 0), ("trainable",)], circuit, learn_measure)
        self.cond = nn.Linear(self.bank1.F, m_modes)      # coarse Fock feature -> fine-mode context (CC)
        # layer-2 fine QPUs: photon-1 = fine cell, photon-2 = coarse context
        self.bank2 = MultiPhotonQPUBankV2(grid_fine * grid_fine, m_modes, add_modes,
                                          [("data", 0), ("data", 1)], circuit, learn_measure)
        r = grid_fine // grid_coarse
        parent = [(i // r) * grid_coarse + (j // r) for i in range(grid_fine) for j in range(grid_fine)]
        self.register_buffer("parent", torch.tensor(parent, dtype=torch.long))  # (N_fine,)
        self.n_qpus = grid_fine * grid_fine
        self.out_dim = self.n_qpus * self.bank2.F

    def _cells(self, fmap, to_modes, grid):               # conv map -> (B, grid*grid, m) raw (bank normalizes)
        h = F.adaptive_avg_pool2d(to_modes(fmap), grid)   # (B, m, grid, grid)
        return h.flatten(2).transpose(1, 2).contiguous()  # (B, grid*grid, m)

    def forward(self, x):                                 # (B,1,H,W) -> (B, out_dim)
        B = x.shape[0]
        fmap = self.features(x)
        coarse = self._cells(fmap, self.to_modes_c, self.gc)     # (B, Nc, m)
        c1 = self.bank1(coarse.unsqueeze(2))                     # (B, Nc, F1)
        ctx = self.cond(c1)[:, self.parent]                      # (B, Nf, m) coarse context per fine cell
        fine = self._cells(fmap, self.to_modes_f, self.gf)       # (B, Nf, m)
        vb = torch.stack([fine, ctx], dim=2)                     # (B, Nf, 2, m): cell + coarse context
        psi = self.bank2(vb)                                     # (B, Nf, F2)
        return psi.reshape(B, -1)


class PartialMeasHierQCNN(nn.Module):
    """ — 2-layer PARTIAL-MEASUREMENT conditional QCNN (user spec).
    Shared conv stem → coarse & fine 1-1 cell encodings (NO Linear compression → fixes the
    conv→pre_quantum info loss). LAYER-1 QPUs: encode coarse cells → interfere → MEASURE ALL MODES
    (per-mode occupation, `mode_readout` — nonlinear, non-invertible → breaks the isometry). A
    PORTION (`cond_modes`) of each L1 QPU's measured modes conditions the L2 QPUs (parent-local,
    additive on the fine encoding). LAYER-2 QPUs: measure all modes. INTERPRET: SOME (`read_modes`)
    measured modes of every L1 QPU + SOME of every L2 QPU → learnable MLP → embedding."""

    def __init__(self, img_size=42, m_modes=4, grid_coarse=2, grid_fine=6, n_conv=2, add_modes=2,
                 n_photons=2, cond_modes=2, read_modes=3, interpret=64, learn_measure=True, circuit="BS"):
        super().__init__()
        from .multiphoton_core import MultiPhotonQPUBank
        self.features, out_ch = _conv_stem(n_conv)
        self.to_c = nn.Conv2d(out_ch, m_modes, 1); self.to_f = nn.Conv2d(out_ch, m_modes, 1)
        self.gc, self.gf, self.m, self.add_modes = grid_coarse, grid_fine, m_modes, add_modes
        Nc, Nf = grid_coarse * grid_coarse, grid_fine * grid_fine
        self.bank1 = MultiPhotonQPUBank(Nc, m_modes, add_modes, n_photons, circuit, learn_measure)
        self.bank2 = MultiPhotonQPUBank(Nf, m_modes, add_modes, n_photons, circuit, learn_measure)
        M = self.bank1.M
        self.cond_modes = min(cond_modes, M); self.read_modes = min(read_modes, M)
        self.cond = nn.Linear(self.cond_modes, m_modes)        # portion of L1 modes → per-QPU context
        r = grid_fine // grid_coarse
        parent = [(i // r) * grid_coarse + (j // r) for i in range(grid_fine) for j in range(grid_fine)]
        self.register_buffer("parent", torch.tensor(parent, dtype=torch.long))
        self.interp = nn.Sequential(nn.Linear(Nc * self.read_modes + Nf * self.read_modes, interpret),
                                    nn.ReLU(inplace=True), nn.Linear(interpret, interpret))
        self.n_qpus, self.out_dim, self._Nc, self._Nf = Nf, interpret, Nc, Nf

    def _cells(self, fmap, to, grid):
        h = F.adaptive_avg_pool2d(to(fmap), grid)
        v = h.flatten(2).transpose(1, 2)                       # (B, grid*grid, m)
        return v / (v.norm(dim=2, keepdim=True) + 1e-8)

    def _enc(self, x):
        """→ (feat = post-measurement readouts, pre = raw conv cells fed to the quantum layers)."""
        B = x.shape[0]; fmap = self.features(x)
        c = self._cells(fmap, self.to_c, self.gc)              # (B,Nc,m) L1 input
        p1 = self.bank1.mode_readout(self.bank1(c))            # (B,Nc,M) L1 FULL mode measurement
        ctx = self.cond(p1[:, :, :self.cond_modes])[:, self.parent]   # (B,Nf,m) portion→condition
        f = self._cells(fmap, self.to_f, self.gf)              # (B,Nf,m) L2 input
        v2 = f + ctx; v2 = v2 / (v2.norm(dim=2, keepdim=True) + 1e-8)  # condition L2 encoding
        p2 = self.bank2.mode_readout(self.bank2(v2))           # (B,Nf,M) L2 FULL mode measurement
        feat = torch.cat([p1[:, :, :self.read_modes].reshape(B, -1),
                          p2[:, :, :self.read_modes].reshape(B, -1)], dim=1)   # SOME modes of each
        pre = torch.cat([c.reshape(B, -1), f.reshape(B, -1)], dim=1)          # raw conv cells (pre-quantum)
        return feat, pre

    def forward(self, x):
        feat, _ = self._enc(x)
        return self.interp(feat)


class DistributedQuantumProtoNet(nn.Module):
    """Standalone DPQCNN metric model. forward(x) -> quantum feature phi_q(x).

    Vectorized single-photon core: the state per QPU is pure (rank-1 rho=v v^T),
    so the measurement marginal diag(U rho U^T) = (U v)^2 elementwise — no density
    matrix is ever formed (O(N m^2), not O(N m^3)) and all N QPU unitaries are
    built + applied in ONE batched op (no Python loop over QPUs). Mathematically
    identical to the per-QPU RegisterDense loop.
    """

    def __init__(self, n_qpus=8, d_total=None, add_modes=None, comm="NC",
                 embed_dim=64, img_size=42, output_mode="prob", qpu_depth=1, n_photons=1,
                 learn_measure=False, photon_config=None, photon_groups=None,
                 entangle_pairs=False, patch_encode=None, hier_encode=None, sweep_cfg=None,
                 spatial_cfg=None, pm_cfg=None):
        super().__init__()
        from ...core.blocks import MeshUnitary
        from ...core._baseline import get_circuit
        d_total = d_total or C.QUANTUM["d_total"]
        add_modes = add_modes if add_modes is not None else C.QUANTUM["add_modes"]
        self.patch_encode = patch_encode           # : {"patch":p,"photon_config":[...]}
        self.hier_encode = hier_encode             #  E2: {"patch_large":..,"patch_small":..}
        self.sweep_cfg = sweep_cfg                  #  E5/6: conv-depth × quantum × dense sweep
        self.spatial_cfg = spatial_cfg              #  Option B: spatial 1-1 conv->QPU encoding
        self.pm_cfg = pm_cfg                        # : 2-layer PARTIAL-measurement QCNN
        self._taps = {}                             #  diagnostics: per-stage capture buffer
        if pm_cfg is not None:                      #  partial-measurement 2-layer conditional
            pc = pm_cfg
            self.pmhier = PartialMeasHierQCNN(img_size, pc.get("m_modes", 4), pc.get("grid_coarse", 2),
                pc.get("grid_fine", 6), pc.get("n_conv", 2), add_modes, pc.get("n_photons", 2),
                pc.get("cond_modes", 2), pc.get("read_modes", 3), pc.get("interpret", 64),
                pc.get("learn_measure", True), C.QUANTUM["conv_circuit"])
            self.n_qpus = self.pmhier.n_qpus; self.m = self.pmhier.m; self.add_modes = add_modes
            self.out_dim = self.pmhier.out_dim; self.output_mode = output_mode
            return
        if spatial_cfg is not None:                 # Option B: exact 1-1 conv-cell -> QPU (no GAP/Linear bottleneck)
            sc = spatial_cfg
            m_modes = sc.get("m_modes", 4); nc = sc.get("n_conv", 2)
            self.add_modes = add_modes; self.m = m_modes; self.output_mode = output_mode
            if sc.get("hier"):                      # #2: 2-layer coarse->fine on conv features
                self.sphier = SpatialHierConvQCNN(img_size, m_modes,
                                                  sc.get("grid_coarse", 2), sc.get("grid_fine", 6),
                                                  nc, add_modes, sc.get("learn_measure", True),
                                                  C.QUANTUM["conv_circuit"])
                self.n_qpus = self.sphier.n_qpus; self.out_dim = self.sphier.out_dim
            else:                                   # #1: single quantum layer on conv cells
                from .multiphoton_core import MultiPhotonQPUBank
                self.spenc = SpatialConvEncoder(img_size, m_modes, sc.get("grid", 6), nc)
                self.mpbank = MultiPhotonQPUBank(self.spenc.N, m_modes, add_modes,
                                                 sc.get("n_photons", 2), C.QUANTUM["conv_circuit"],
                                                 learn_measure=sc.get("learn_measure", True))
                self.n_qpus = self.spenc.N; self.out_dim = self.spenc.N * self.mpbank.F
            return
        if sweep_cfg is not None:                   # CNN(depth) [+ quantum head] [+ dense head] -> embedding
            nc = sweep_cfg.get("n_conv", 3); self.use_quantum = sweep_cfg.get("use_quantum", True)
            dh = sweep_cfg.get("dense_head", 0); npho = sweep_cfg.get("n_photons", 3)
            self.m = d_total // n_qpus; self.add_modes = add_modes; self.n_qpus = n_qpus
            self.enc = ConvEncoder(embed_dim, img_size=img_size, n_conv=nc)
            if self.use_quantum:
                from .multiphoton_core import MultiPhotonQPUBank
                self.proj = nn.Linear(embed_dim, d_total)
                self.mpbank = MultiPhotonQPUBank(n_qpus, self.m, add_modes, npho,
                                                 C.QUANTUM["conv_circuit"],
                                                 learn_measure=sweep_cfg.get("learn_measure", True))
                self.qdim = n_qpus * self.mpbank.F
                self.reupload = sweep_cfg.get("reupload", False)   #  iter2: data re-uploading
                self.readout = sweep_cfg.get("readout", "amp")     # : trainable-vs-lossy readout family
                if self.reupload:                                  # 2nd quantum layer, data re-injected
                    self._qdim1 = self.qdim; self.readout = "amp"
                    self.reproj = nn.Linear(self.qdim + embed_dim, d_total)  # [layer1 Fock ; data] -> modes
                    self.mpbank2 = MultiPhotonQPUBank(n_qpus, self.m, add_modes, npho,
                                                      C.QUANTUM["conv_circuit"],
                                                      learn_measure=sweep_cfg.get("learn_measure", True))
                    self.qdim = n_qpus * self.mpbank2.F
                else:
                    Fq = n_qpus * self.mpbank.F
                    if self.readout == "amp_prob": self.qdim = 2 * Fq          # info(ψ) + nonlinear(ψ²)
                    elif self.readout == "modes":  self.qdim = n_qpus * self.mpbank.M   # per-mode occupation
                    elif self.readout == "residual": self.qdim = Fq + d_total  # classical skip + quantum ψ²
                    elif self.readout == "obs":                                # learned observables on ψ²
                        self._odim = sweep_cfg.get("obs_dim", 64)
                        self.obs = nn.Linear(Fq, self._odim); self.qdim = self._odim
                    else: self.qdim = Fq                                       # amp | prob
            else:
                self.use_quantum = False; self.reupload = False; self.readout = "amp"
                self.qdim = embed_dim
            self.dense = nn.Sequential(*sum(([nn.Linear(self.qdim, self.qdim),
                                              nn.ReLU(inplace=True)] for _ in range(dh)), [])) if dh else None
            self.output_mode = output_mode; self.out_dim = self.qdim
            return
        if hier_encode is not None:                #  E2 hierarchical coarse->fine QCNN
            self.hier = HierarchicalPatchQCNN(img_size, hier_encode["patch_large"],
                                              hier_encode["patch_small"], add_modes,
                                              learn_measure=learn_measure,
                                              interpret=hier_encode.get("interpret", 0))
            self.output_mode = output_mode
            self.m, self.add_modes, self.n_qpus = self.hier.m, self.hier.add_modes, self.hier.n_qpus
            self.out_dim = self.hier.out_dim
            return
        if patch_encode is not None:               #  genuine QCNN (direct pixel encoding)
            from .multiphoton_core import MultiPhotonQPUBankV2
            self.patchenc = PatchAmplitudeEncoder(img_size, patch_encode["patch"])
            pc = patch_encode["photon_config"]
            self.mpbank = MultiPhotonQPUBankV2(self.patchenc.N, self.patchenc.m, add_modes, pc,
                                               C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            self.n_data_blocks = self.mpbank.n_data_blocks
            self.output_mode = output_mode
            self.m = self.patchenc.m; self.add_modes = add_modes
            self.n_qpus = self.patchenc.N
            self.out_dim = self.patchenc.N * self.mpbank.F
            return
        assert d_total % n_qpus == 0, f"d_total {d_total} not divisible by N={n_qpus}"
        assert not (comm == "CC" and qpu_depth > 1), "CC path is depth-1 (control comm ablation)"
        self.n_qpus = n_qpus
        self.m = d_total // n_qpus                 # active modes per QPU
        self.add_modes = add_modes
        self.M = self.m + add_modes                # total modes per QPU (padded)
        self.comm = comm
        self.output_mode = output_mode             # "prob" (measured (Uv)^2) | "amp" (Uv)
        self.qpu_depth = qpu_depth                 # photonic sublayers (|.| nonlinearity between)
        self.n_photons = n_photons                 # P1: 1=single-photon fast path, 2/3=Fock core
        self.photon_config = photon_config         # : per-photon input specs (V2 core)
        self.photon_groups = photon_groups         #  L2: heterogeneous QPU groups
        self.entangle_pairs = entangle_pairs       #  L4: paired entangling coupling
        self.enc = ConvEncoder(embed_dim, img_size=img_size)
        if entangle_pairs:                         #  L4 inter-QPU entangling coupling
            assert comm == "NC" and qpu_depth == 1, "entangling core: NC + depth-1 only"
            from .multiphoton_core import PairedEntanglingBank
            self.entbank = PairedEntanglingBank(n_qpus, self.m, add_modes,
                                                C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            self.proj = nn.Linear(embed_dim, d_total)
            self.out_dim = self.entbank.n_pairs * self.entbank.F
        elif photon_groups is not None:            #  L2 heterogeneous/interleaved QPUs
            assert comm == "NC" and qpu_depth == 1, "hetero core: NC + depth-1 only"
            from .multiphoton_core import HeteroPhotonBank
            self.heterobank = HeteroPhotonBank(embed_dim, photon_groups, self.m, add_modes,
                                               C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            self.out_dim = self.heterobank.out_dim
        elif photon_config is not None:            #  L1+L3 multi-DATA-photon core
            assert comm == "NC" and qpu_depth == 1, "V2 core: NC + depth-1 only"
            from .multiphoton_core import MultiPhotonQPUBankV2
            self.mpbank = MultiPhotonQPUBankV2(n_qpus, self.m, add_modes, photon_config,
                                               C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            self.n_data_blocks = self.mpbank.n_data_blocks
            self.proj = nn.Linear(embed_dim, n_qpus * self.n_data_blocks * self.m)
            self.out_dim = n_qpus * self.mpbank.F
        elif n_photons > 1:                        # P1 multi-photon Fock core ()
            assert comm == "NC" and qpu_depth == 1, "multi-photon core: NC + depth-1 only"
            from .multiphoton_core import MultiPhotonQPUBank
            self.proj = nn.Linear(embed_dim, d_total)
            self.mpbank = MultiPhotonQPUBank(n_qpus, self.m, add_modes, n_photons,
                                             C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            self.out_dim = n_qpus * self.mpbank.F  # Fock-space feature per QPU
        else:                                      # single-photon () fast path
            self.proj = nn.Linear(embed_dim, d_total)  # -> N*m photonic amplitudes
            self.mesh = MeshUnitary(get_circuit(self.M, C.QUANTUM["conv_circuit"]))
            self.phi = nn.Parameter(2 * np.pi * torch.rand(qpu_depth, n_qpus, self.mesh.nparam))
            self.out_dim = n_qpus * self.M
            if comm == "CC":                       # paired feed-forward (within pair only)
                self.cc = nn.Linear(self.M, self.m)

    def _amp(self, p):                             # p (B,N,m) -> padded unit amplitudes (B,N,M)
        v = p / (p.norm(dim=2, keepdim=True) + 1e-8)
        if self.add_modes:
            v = F.pad(v, (0, self.add_modes))
        return v

    def _measure(self, vpad, U):                   # (B,N,M),(N,M,M) -> (B,N,M) probs
        uv = torch.einsum("nij,bnj->bni", U, vpad)
        return uv * uv                             # (U v)^2 = diag(U rho U^T)

    def forward(self, x, output_mode=None, capture=False):  # x (B,1,H,W) -> (B, out_dim)
        om = output_mode or self.output_mode
        B = x.shape[0]
        taps = {} if capture else None
        def _tap(n, t):
            if taps is not None:
                taps[n] = t.detach().reshape(B, -1)
        if self.pm_cfg is not None:                 #  2-layer partial-measurement QCNN
            feat, pre = self.pmhier._enc(x)
            out = self.pmhier.interp(feat)
            _tap("input", x); _tap("pre_quantum", pre); _tap("post_quantum", feat); _tap("final", out)
            if taps is not None: self._taps = taps
            return out if om == "amp" else out * out
        if self.spatial_cfg is not None:            # Option B: spatial 1-1 conv->QPU
            _tap("input", x)
            if self.spatial_cfg.get("hier"):
                psi = self.sphier(x)
                _tap("post_quantum", psi); _tap("final", psi)
                if taps is not None: self._taps = taps
                return psi if om == "amp" else psi * psi
            v = self.spenc(x)                        # (B, N, m) raw conv-cell amplitudes
            _tap("conv", v); _tap("pre_quantum", v)
            v = v / (v.norm(dim=2, keepdim=True) + 1e-8)
            psi = self.mpbank(v).reshape(B, self.out_dim)
            _tap("post_quantum", psi); _tap("final", psi)
            if taps is not None: self._taps = taps
            return psi if om == "amp" else psi * psi
        if self.sweep_cfg is not None:              #  E5/6 conv×quantum×dense sweep
            _tap("input", x)
            feat = self.enc(x)                       # (B, embed_dim)
            _tap("conv", feat)
            if self.use_quantum:
                p = self.proj(feat).reshape(B, self.n_qpus, self.m)
                v = p / (p.norm(dim=2, keepdim=True) + 1e-8)
                _tap("pre_quantum", v)
                if getattr(self, "reupload", False):     # re-inject data into a 2nd quantum layer
                    e = self.mpbank(v).reshape(B, -1)
                    v2 = self.reproj(torch.cat([e, feat], dim=1)).reshape(B, self.n_qpus, self.m)
                    v2 = v2 / (v2.norm(dim=2, keepdim=True) + 1e-8)
                    e = self.mpbank2(v2).reshape(B, -1)
                else:
                    psi = self.mpbank(v)                 # (B,N,F) amplitudes
                    ro = self.readout
                    if ro == "prob":       e = (psi * psi).reshape(B, -1)          # nonlinear, keeps magnitude
                    elif ro == "amp_prob": e = torch.cat([psi.reshape(B, -1), (psi * psi).reshape(B, -1)], 1)
                    elif ro == "modes":    e = self.mpbank.mode_readout(psi).reshape(B, -1)
                    elif ro == "residual": e = torch.cat([v.reshape(B, -1), (psi * psi).reshape(B, -1)], 1)
                    elif ro == "obs":      e = self.obs((psi * psi).reshape(B, -1))  # learned observables
                    else:                  e = psi.reshape(B, -1)                    # amp (isometry baseline)
                _tap("post_quantum", e)
            else:
                e = feat
                _tap("pre_quantum", feat); _tap("post_quantum", feat)
            if self.dense is not None:
                e = self.dense(e)
            _tap("final", e)
            if taps is not None: self._taps = taps
            # readouts other than "amp" already ARE the measured feature (no extra squaring)
            if self.use_quantum and getattr(self, "readout", "amp") != "amp" and not getattr(self, "reupload", False):
                return e
            return e if om == "amp" else e * e
        if self.hier_encode is not None:            #  E2 hierarchical QCNN
            psi = self.hier(x)
            return psi if om == "amp" else psi * psi
        if self.patch_encode is not None:           #  genuine QCNN (direct patch encoding)
            vb = self.patchenc(x).unsqueeze(2)       # (B,N,1,m) — 1 data block (the patch)
            if self.n_data_blocks > 1:               # replicate patch across data blocks if needed
                vb = vb.expand(-1, -1, self.n_data_blocks, -1)
            psi = self.mpbank(vb).reshape(B, self.out_dim)
            return psi if om == "amp" else psi * psi
        if self.entangle_pairs:                     #  L4 paired entangling
            _tap("input", x); f = self.enc(x); _tap("conv", f)
            p = self.proj(f).reshape(B, self.n_qpus, self.m)
            _tap("pre_quantum", p)
            psi = self.entbank(p).reshape(B, self.out_dim)
            _tap("post_quantum", psi); _tap("final", psi)
            if taps is not None: self._taps = taps
            return psi if om == "amp" else psi * psi
        if self.photon_groups is not None:          #  L2 heterogeneous QPUs
            _tap("input", x); f = self.enc(x); _tap("conv", f); _tap("pre_quantum", f)
            psi = self.heterobank(f)                 # (B, out_dim)
            _tap("post_quantum", psi); _tap("final", psi)
            if taps is not None: self._taps = taps
            return psi if om == "amp" else psi * psi
        if self.photon_config is not None:          #  V2 multi-DATA-photon core
            _tap("input", x); f = self.enc(x); _tap("conv", f)
            vb = self.proj(f).reshape(B, self.n_qpus, self.n_data_blocks, self.m)
            _tap("pre_quantum", vb)
            psi = self.mpbank(vb).reshape(B, self.out_dim)
            _tap("post_quantum", psi); _tap("final", psi)
            if taps is not None: self._taps = taps
            return psi if om == "amp" else psi * psi
        _tap("input", x); f = self.enc(x); _tap("conv", f)
        p = self.proj(f).reshape(B, self.n_qpus, self.m)
        if self.n_photons > 1:                      # P1 multi-photon Fock core
            v = p / (p.norm(dim=2, keepdim=True) + 1e-8)   # unit data amplitudes per QPU
            _tap("pre_quantum", v)
            psi = self.mpbank(v).reshape(B, self.out_dim)  # (B, N*F) real Fock amplitudes
            _tap("post_quantum", psi); _tap("final", psi)
            if taps is not None: self._taps = taps
            return psi if om == "amp" else psi * psi
        if self.comm == "CC":                       # paired feed-forward (depth-1, prob)
            U = self.mesh(self.phi[0])
            vpad = self._amp(p)
            marg = self._measure(vpad, U)           # NC pass (all QPUs)
            even = marg[:, 0::2]                     # (B, N/2, M) partner marginals
            bias = self.cc(even)                     # (B, N/2, m)
            p_odd = p[:, 1::2] + bias                # feed-forward within pair
            v_odd = self._amp(p_odd)
            marg_odd = self._measure(v_odd, U[1::2])
            marg = marg.clone(); marg[:, 1::2] = marg_odd
            return marg.reshape(B, self.out_dim)
        # NC pass. qpu_depth photonic sublayers; between layers a MEASUREMENT
        # nonlinearity |a| (a linear stack of interferometers would collapse to a
        # single unitary — the |.| is what makes depth expressive). |a| keeps unit
        # norm (U orthogonal), so no renormalization needed.
        v = self._amp(p)                            # (B,N,M) unit amplitudes
        a = v
        for l in range(self.qpu_depth):
            U = self.mesh(self.phi[l])              # (N, M, M), one batched build
            a = torch.einsum("nij,bnj->bni", U, v)
            if l < self.qpu_depth - 1:
                v = a.abs()                         # nonlinear re-encode for next sublayer
        if om == "amp":
            return a.reshape(B, self.out_dim)       # pre-measurement amplitudes (fidelity)
        return (a * a).reshape(B, self.out_dim)     # measured marginals (Uv)^2


# ---------------------------------------------------------------------------
# Episodic training on BASE (end-to-end) + eval on shared novel episodes.
# ---------------------------------------------------------------------------
def _base_pool(device, img_size, dm=P):
    X, y, split, meta = dm.load(img_size=img_size)
    m = split == "base_train"
    Xb = torch.tensor(X[m], dtype=torch.float32, device=device)[:, None]
    yb = np.array([meta["classes"][int(v)] for v in y[m]])
    return Xb, {c: np.where(yb == c)[0] for c in C.BASE_CLASSES}


def _ce_pretrain(model, dev, seed, img_size, dm=P):
    """Train encoder+projection+QPUs with a CE classifier on BASE (3-class), then
    drop the classifier and use phi_q prototypes — the recipe that worked for the
    classical frozen path (~73%) vs episodic (~52% overfit)."""
    from .backbone import _augment
    torch.manual_seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    Xb, idx_by = _base_pool(dev, img_size, dm)
    order = torch.cat([torch.tensor(idx_by[c], device=dev) for c in C.BASE_CLASSES])
    labels = torch.cat([torch.full((len(idx_by[c]),), i, device=dev)
                        for i, c in enumerate(C.BASE_CLASSES)])
    clf = nn.Linear(model.out_dim, len(C.BASE_CLASSES)).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                           lr=C.QUANTUM.get("ce_lr", C.BACKBONE["lr"]),
                           weight_decay=C.BACKBONE["weight_decay"])
    bs, n = C.QUANTUM.get("ce_batch_size", C.BACKBONE["batch_size"]), order.shape[0]
    epochs = C.BACKBONE["pretrain_epochs"]
    steps_per_ep = (n + bs - 1) // bs
    total_steps = epochs * steps_per_ep
    warmup = max(1, int(C.QUANTUM.get("ce_warmup_frac", 0.0) * total_steps))
    use_sched = C.QUANTUM.get("ce_warmup_frac", 0.0) > 0
    def lr_at(step):                               # linear warmup -> cosine decay
        if step < warmup:
            return step / warmup
        import math
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at) if use_sched else None
    t0 = time.time(); model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = order[perm[i:i + bs]]
            opt.zero_grad()
            F.cross_entropy(clf(model(_augment(Xb[idx], gen))), labels[perm[i:i + bs]]).backward()
            opt.step()
            if sched is not None:
                sched.step()
    model.eval()
    return time.time() - t0


def train_and_eval(n_qpus=8, comm="NC", seed=42, n_episodes=None, img_size=None,
                   d_total=None, train_mode="episodic", qpu_depth=1, output_mode="prob",
                   return_model=False, ssl_encoder=False, data_module=None,
                   eval_episodic=True, n_photons=1, learn_measure=False, photon_config=None,
                   photon_groups=None, entangle_pairs=False, patch_encode=None, hier_encode=None,
                   ckpt_tag=None, ckpt_every=1000, sweep_cfg=None, spatial_cfg=None, pm_cfg=None):
    img_size = img_size or C.IMG_SIZE
    dm = data_module or P
    setup_perf()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    _cfg = dict(n_qpus=n_qpus, d_total=d_total, comm=comm, img_size=img_size, qpu_depth=qpu_depth,
                output_mode=output_mode, n_photons=n_photons, learn_measure=learn_measure,
                photon_config=photon_config, photon_groups=photon_groups,
                entangle_pairs=entangle_pairs, patch_encode=patch_encode, hier_encode=hier_encode,
                sweep_cfg=sweep_cfg, spatial_cfg=spatial_cfg, pm_cfg=pm_cfg)
    model = DistributedQuantumProtoNet(**_cfg).to(dev)

    def _ck(step, extra=None):                     # frequent checkpoints for debug/retrain/ablation
        if not ckpt_tag:
            return
        d = C.CKPT_DIR / ckpt_tag; d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": _cfg,
                    "meta": {"tag": ckpt_tag, "seed": seed, "train_mode": train_mode,
                             "out_dim": model.out_dim, "step": step, **(extra or {})}}, d / f"{step}.pt")
        (d / "config.json").write_text(json.dumps(
            {"cfg": _cfg, "tag": ckpt_tag, "seed": seed, "train_mode": train_mode,
             "out_dim": model.out_dim, "n_params": sum(p.numel() for p in model.parameters())},
            indent=2, default=str))
    if ssl_encoder:                                # E4: SSL-pretrain the conv encoder first
        from .ssl_pretrain import contrastive_pretrain
        contrastive_pretrain(model.enc, dev, seed, img_size=img_size, verbose=False)
    opt = torch.optim.Adam(model.parameters(), lr=C.QUANTUM["episodic_lr"])
    Xb, idx_by = _base_pool(dev, img_size, dm)
    nw, k, q = min(3, len(C.BASE_CLASSES)), 5, 15
    n_ep = n_episodes or C.QUANTUM["episodic_train_episodes"]
    if train_mode == "ce":
        train_s = _ce_pretrain(model, dev, seed, img_size, dm)
    elif train_mode == "proto":                    # : fidelity-matched prototypical loss
        t0 = time.time(); model.train()
        for it in range(n_ep):
            cls = list(rng.choice(C.BASE_CLASSES, nw, replace=False))
            si, sl, qi, ql = [], [], [], []
            for ci, c in enumerate(cls):
                pk = rng.choice(idx_by[c], k + q, replace=False)
                si += pk[:k].tolist(); sl += [ci] * k; qi += pk[k:].tolist(); ql += [ci] * q
            se = F.normalize(model(Xb[si], output_mode="amp"), dim=1)   # embeddings on hypersphere
            qe = F.normalize(model(Xb[qi], output_mode="amp"), dim=1)
            slt = torch.tensor(sl, device=dev)
            protos = F.normalize(torch.stack([se[slt == c].mean(0) for c in range(nw)]), dim=1)
            logits = 10.0 * (qe @ protos.t())       # temperature-scaled cosine (fidelity geometry)
            loss = F.cross_entropy(logits, torch.tensor(ql, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if (it + 1) % 5000 == 0:
                print(f"[dqproto proto] {it+1}/{n_ep} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
            if ckpt_tag and (it + 1) % ckpt_every == 0:
                _ck(f"ep{it+1}", {"loss": round(float(loss.item()), 4)})
        train_s = time.time() - t0
    else:
        t0 = time.time(); model.train()
        for it in range(n_ep):
            cls = list(rng.choice(C.BASE_CLASSES, nw, replace=False))
            si, sl, qi, ql = [], [], [], []
            for ci, c in enumerate(cls):
                pk = rng.choice(idx_by[c], k + q, replace=False)
                si += pk[:k].tolist(); sl += [ci] * k; qi += pk[k:].tolist(); ql += [ci] * q
            se, qe = model(Xb[si]), model(Xb[qi])
            protos = torch.stack([se[torch.tensor(sl, device=dev) == c].mean(0) for c in range(nw)])
            loss = F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(ql, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if (it + 1) % 5000 == 0:
                print(f"[dqproto N={n_qpus} {comm}] {it+1}/{n_ep} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        train_s = time.time() - t0
    model.eval()
    _ck("final", {"train_s": round(train_s, 1)})
    print(f"[dqproto N={n_qpus} {comm}] seed={seed} m={model.m}+{model.add_modes} "
          f"out={model.out_dim} params={sum(p.numel() for p in model.parameters())} train={train_s:.0f}s")
    if not eval_episodic:                          # FSCIL base-training: just return the frozen model
        return ([], model, dev) if return_model else []
    results = []
    for task in C.EPISODE_TASKS:
        r = evaluate(model, task["n_way"], task["k_shot"], seed, dev, img_size=img_size)
        r.update({"n_qpus": n_qpus, "comm": comm, "m_modes": model.m,
                  "add_modes": model.add_modes, "train_s": train_s})
        results.append(r)
        print(f"  {r['n_way']}w{r['k_shot']}s: acc={r['acc_mean']:.2f}±{r['acc_ci95']:.2f} "
              f"novel_recall={r['novel_recall_macro']:.2f} f1={r['macro_f1']:.2f} "
              f"({r['infer_s_per_query']*1e3:.3f} ms/query)")
    if return_model:
        return results, model, dev
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qpus", type=int, default=8)
    ap.add_argument("--comm", choices=["NC", "CC"], default="NC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--d-total", type=int, default=None)
    ap.add_argument("--train-mode", choices=["episodic", "ce"], default="ce")
    ap.add_argument("--qpu-depth", type=int, default=1)
    ap.add_argument("--output-mode", choices=["prob", "amp"], default="prob")
    ap.add_argument("--n-photons", type=int, default=1)
    a = ap.parse_args()
    train_and_eval(a.n_qpus, a.comm, a.seed, a.episodes, d_total=a.d_total,
                   train_mode=a.train_mode, qpu_depth=a.qpu_depth, output_mode=a.output_mode,
                   n_photons=a.n_photons)
