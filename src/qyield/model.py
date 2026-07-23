"""model.py — self-contained inference module for the QYield quantum wafer-defect
classifier (QResNet-ensemble: 3x SSL-pretrained ResNet50 + a photonic head).

Zero dependency on the DP-QCNN research package or its file layout — this module
only needs the checkpoint + SSL stems + K-set bundled in this repo's `checkpoints/`
and `data/` folders.

    from qyield.model import QYieldModel

    model = QYieldModel()                          # loads the one shipped checkpoint
    result = model.predict("/path/to/query_wafer.npy")
    # {"predicted_class": "Scratch", "ranking": [("Scratch", 12.4), ...], ...}

-----------------------------------------------------------------------------------
WHAT THIS IS / IS NOT
-----------------------------------------------------------------------------------
This is an episodic ProtoNet few-shot classifier: there is no fixed softmax head.
Classification = nearest-Euclidean-prototype distance to a K-shot SUPPORT SET,
embedded FRESH from the bundled K-set every time `QYieldModel` is constructed (not
baked into the checkpoint) — the genuine few-shot protocol this model was trained
and accuracy-verified with.

By default, `predict()`/`predict_array()` classify against all 8 known WM-811K
defect classes using every bundled support shot (a fixed, deterministic 8-way
classifier). Pass `n_way`/`k_shot`/`ways` to instead reproduce the exact episodic
regime the reported accuracy (83.04 @ 3-way/5-shot, beating the 77.71 CNN-SOTA bar)
describes — see README.md.
"""
from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    DEFAULT_CKPT_PATH,
    DEFAULT_KSET_PATH,
    DEFAULT_STEMS_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PIXEL_NORM_DIV,
    RESIZE_MODE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Photonic circuit primitives — copied verbatim from dpqcnn.core.blocks.MeshUnitary
# and dpqcnn.fswmpr.models.multiphoton_core.{permanent, MultiPhotonQPUBank}, using
# only perceval's public API (no dependency on the DP-QCNN research repo).
# ---------------------------------------------------------------------------
def _get_bs_circuit(m: int):
    """The 'BS' triangular beamsplitter-mesh circuit used by the model's photonic
    head. Requires `perceval` (pip install perceval-quandela) — used ONLY at
    model-construction time to extract the gate schedule; inference itself is
    pure torch (see MeshUnitary.forward)."""
    from perceval import BS
    from perceval.components import GenericInterferometer
    from perceval.utils import InterferometerShape, P
    return GenericInterferometer(m, lambda i: BS.Ry(theta=-2 * P(f"phi_{i}")),
                                 shape=InterferometerShape.TRIANGLE)


def _generate_all_fock_states_list(m: int, n: int, true_order: bool = True) -> list:
    def _gen(m, n):
        if n == 0:
            yield (0,) * m
            return
        if m == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for state in _gen(m - 1, n - i):
                yield (i,) + state
    states = list(_gen(m, n))
    if true_order:
        states.reverse()
    return states


class MeshUnitary(nn.Module):
    """Vectorized, differentiable builder for the m x m mode unitary of a single-photon
    'BS' interferometer. Uses `merlin` (pip install merlinquantum) ONLY at __init__ to
    extract the gate schedule from perceval's circuit structure; forward() has no
    perceval/merlin dependency (pure torch)."""

    def __init__(self, circ):
        super().__init__()
        import io
        import sys as _sys
        from merlin import CircuitConverter

        self.m = circ.m
        self.nparam = len(circ.get_parameters())
        _stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            conv = CircuitConverter(circ, ["phi"], dtype=torch.float32)
        finally:
            _sys.stdout = _stdout
        modes, kidx = [], []
        for r, c in conv.list_rct:
            if isinstance(c, torch.Tensor):
                raise NotImplementedError("MeshUnitary supports parametric BS meshes only")
            ps = c.get_parameters()
            if len(ps) != 1:
                raise NotImplementedError("MeshUnitary expects single-parameter BS gates")
            modes.append(list(r)[0])
            kidx.append(int(ps[0].name.split("_")[1]))
        last: dict[int, int] = {}
        layer_of = []
        for a in modes:
            lvl = max(last.get(a, -1), last.get(a + 1, -1)) + 1
            layer_of.append(lvl)
            last[a] = last[a + 1] = lvl
        self.n_layers = (max(layer_of) + 1) if layer_of else 1
        self.register_buffer("_layer", torch.tensor(layer_of, dtype=torch.long), persistent=False)
        self.register_buffer("_a", torch.tensor(modes, dtype=torch.long), persistent=False)
        self.register_buffer("_k", torch.tensor(kidx, dtype=torch.long), persistent=False)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        single = phi.dim() == 1
        if single:
            phi = phi.unsqueeze(0)
        b, m, nL = phi.shape[0], self.m, self.n_layers
        half = 0.5 * phi
        c, s = torch.cos(half), torch.sin(half)
        eye = torch.eye(m, dtype=phi.dtype, device=phi.device)
        L = eye.reshape(1, 1, m, m).repeat(nL, b, 1, 1)
        li, a, k = self._layer, self._a, self._a + 1
        ck = c[:, self._k].transpose(0, 1)
        sk = s[:, self._k].transpose(0, 1)
        L[li, :, self._a, self._a] = ck
        L[li, :, self._a, k] = -sk
        L[li, :, k, self._a] = sk
        L[li, :, k, k] = ck
        u = L[0]
        for l in range(1, nL):
            u = L[l] @ u
        return u.squeeze(0) if single else u


def _permanent(A: torch.Tensor) -> torch.Tensor:
    """Permanent of the last-two (n x n) dims of A (n small), via Ryser's formula."""
    n = A.shape[-1]
    if n == 1:
        return A[..., 0, 0]
    total = A.new_zeros(A.shape[:-2])
    cols = list(range(n))
    for k in range(1, n + 1):
        for S in combinations(cols, k):
            rowsum = A[..., list(S)].sum(dim=-1)
            total = total + ((-1) ** k) * rowsum.prod(dim=-1)
    return ((-1) ** n) * total


class MultiPhotonQPUBank(nn.Module):
    """N QPUs of M modes, each with n photons (1 data + n-1 ancilla)."""

    def __init__(self, n_qpus: int, m_active: int, add_modes: int, n_photons: int,
                 circuit: str = "BS", learn_measure: bool = False):
        super().__init__()
        M = m_active + add_modes
        if add_modes < n_photons - 1:
            raise ValueError(f"add_modes ({add_modes}) must be >= n_photons-1 ({n_photons-1})")
        self.n_qpus, self.m, self.M, self.n = n_qpus, m_active, M, n_photons
        self.mesh = MeshUnitary(_get_bs_circuit(M))
        self.phi = nn.Parameter(2 * np.pi * torch.rand(n_qpus, self.mesh.nparam))
        anc = list(range(m_active, m_active + n_photons - 1))

        in_modes, in_fact = [], []
        for x in range(M):
            modes = [x] + anc
            in_modes.append(modes)
            occ = np.bincount(modes, minlength=M)
            in_fact.append(math.sqrt(float(np.prod([math.factorial(int(o)) for o in occ]))))
        keys = _generate_all_fock_states_list(M, n_photons, true_order=True)
        out_modes, out_fact = [], []
        for t in keys:
            modes = []
            for j, tj in enumerate(t):
                modes += [j] * int(tj)
            out_modes.append(modes)
            out_fact.append(math.sqrt(float(np.prod([math.factorial(int(tj)) for tj in t]))))
        self.F = len(keys)
        self.register_buffer("in_modes", torch.tensor(in_modes, dtype=torch.long))
        self.register_buffer("out_modes", torch.tensor(out_modes, dtype=torch.long))
        self.register_buffer("in_fact", torch.tensor(in_fact, dtype=torch.float32))
        self.register_buffer("out_fact", torch.tensor(out_fact, dtype=torch.float32))
        self.register_buffer("occ", torch.tensor([list(k) for k in keys], dtype=torch.float32))

        self.learn_measure = learn_measure
        if learn_measure:
            self.meas_gen = nn.Parameter(0.01 * torch.randn(self.F, self.F))

    def _u_evolve(self, U: torch.Tensor) -> torch.Tensor:
        rows = U[:, self.out_modes, :]
        sub = rows[:, :, :, self.in_modes]
        sub = sub.permute(0, 1, 3, 2, 4).contiguous()
        per = _permanent(sub)
        return per / (self.out_fact[None, :, None] * self.in_fact[None, None, :])

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        vpad = F.pad(v, (0, self.M - self.m))
        U = self.mesh(self.phi)
        ue = self._u_evolve(U)
        psi = torch.einsum("bnx,nfx->bnf", vpad, ue)
        if self.learn_measure:
            R = torch.matrix_exp(self.meas_gen - self.meas_gen.t())
            psi = torch.einsum("bnf,gf->bng", psi, R)
        return psi

    def mode_readout(self, psi):
        prob = psi * psi
        return torch.einsum("bnf,fm->bnm", prob, self.occ)


# ---------------------------------------------------------------------------
# QResHead — the photonic head on top of the concatenated 3x SSL-ResNet50 embedding.
# ---------------------------------------------------------------------------
class QResHead(nn.Module):
    def __init__(self, head, E=2048, m_modes=4, add_modes=1, n_photons=2,
                 read_modes=2, learn_measure=True):
        super().__init__()
        self.head = head
        self.E = E
        if head == "baseline":
            self.out_dim = E
            return
        assert E % m_modes == 0
        self.m, self.n_qpus = m_modes, E // m_modes
        M = m_modes + add_modes
        self.M = M
        self.F = math.comb(M + n_photons - 1, n_photons)
        self.read_modes = min(read_modes, M)
        self.out_dim = self.n_qpus * self.read_modes
        if head == "quantum":
            self.bank = MultiPhotonQPUBank(self.n_qpus, self.m, add_modes, n_photons,
                                           "BS", learn_measure=learn_measure)
        else:
            self.cW = nn.Parameter(0.1 * torch.randn(self.n_qpus, self.read_modes, self.m, self.m))

    def forward(self, e):
        if self.head == "baseline":
            return e
        B = e.shape[0]
        zc = e.reshape(B, self.n_qpus, self.m)
        zc = zc / (zc.norm(dim=2, keepdim=True) + 1e-8)
        if self.head == "quantum":
            psi = self.bank(zc)
            occ = self.bank.mode_readout(psi)
            return occ[:, :, :self.read_modes].reshape(B, -1)
        Wsym = 0.5 * (self.cW + self.cW.transpose(-1, -2))
        return torch.einsum("bni,nkij,bnj->bnk", zc, Wsym, zc).reshape(B, -1)


# ---------------------------------------------------------------------------
# SSL ResNet50 stem loading — rebuilds the 3 SSL-pretrained backbones
# (SimCLR/Barlow/VICReg) from their checkpoints.
# ---------------------------------------------------------------------------
def _build_resnet50_stem(device):
    import torchvision
    net = torchvision.models.resnet50(weights=None)
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool).to(device)


def _load_stem(ckpt_path: Path, device):
    stem = _build_resnet50_stem(device)
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    stem.load_state_dict(c["stem_state_dict"])
    stem.eval()
    return stem


# ---------------------------------------------------------------------------
# Query image preprocessing — mirrors the training pipeline's normalization.
# ---------------------------------------------------------------------------
def load_query_image(path, img_size: int) -> np.ndarray:
    """Load+preprocess a query wafer map. Accepts:
      * .npy — raw wafer map, int array with values in {0,1,2} (canonical WM-811K
               format, RECOMMENDED) or an already-normalized float array.
      * .png/.jpg/... — grayscale image, best-effort (not the canonical format).
    Returns (img_size, img_size) float32 in [0,1]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"query image not found: {path}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.dtype.kind in "iu" and arr.max() <= 2:
            x = torch.tensor(arr.astype(np.float32) / PIXEL_NORM_DIV)[None, None]
        else:
            x = torch.tensor(arr.astype(np.float32))[None, None]
            if x.max() > 1.0:
                x = x / x.max()
    else:
        from PIL import Image
        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.tensor(arr)[None, None]
    x = F.interpolate(x, size=(img_size, img_size), mode=RESIZE_MODE, align_corners=False)
    return x[0, 0].numpy().astype(np.float32)


def load_kset(path):
    """Returns (images_224: (N,224,224) float32, labels: (N,) str array, classes: list[str])."""
    d = np.load(path, allow_pickle=True)
    return d["images_224"], d["labels"], list(d["classes"])


def compute_prototypes(embeddings: torch.Tensor, labels, classes):
    protos = {}
    labels_arr = np.asarray(labels)
    for c in classes:
        mask = labels_arr == c
        if mask.sum() == 0:
            continue
        protos[c] = embeddings[mask].mean(0)
    return protos


def _predict(query_emb: torch.Tensor, protos: dict):
    names = list(protos.keys())
    P_ = torch.stack([protos[c] for c in names])
    d = torch.cdist(query_emb[None], P_)[0]
    order = torch.argsort(d)
    return [(names[i], float(d[i])) for i in order.tolist()]


def _select_episode_classes(all_classes: list, n_way: int | None, ways: list[str] | None,
                            rng: np.random.Generator) -> list:
    """Resolve which classes participate in this prediction call.
    - `ways` (explicit class list) takes priority if given.
    - else `n_way` picks a random subset of that size from all_classes.
    - else (both None) -> all classes (default full 8-way behavior)."""
    if ways is not None:
        unknown = [c for c in ways if c not in all_classes]
        if unknown:
            raise ValueError(f"unknown class name(s) in `ways`: {unknown}; "
                             f"valid classes: {all_classes}")
        return list(ways)
    if n_way is not None:
        if not (1 <= n_way <= len(all_classes)):
            raise ValueError(f"n_way must be between 1 and {len(all_classes)}, got {n_way}")
        return list(rng.choice(all_classes, size=n_way, replace=False))
    return list(all_classes)


def _select_shots(labels_arr: np.ndarray, classes: list, k_shot: int | None,
                  rng: np.random.Generator) -> np.ndarray:
    """Return indices (into the full support pool) to use for prototype computation.
    If k_shot is None, use every available support example for each selected class.
    If k_shot is set, randomly subsample (without replacement) k_shot examples per
    class from what's available."""
    if k_shot is None:
        return np.where(np.isin(labels_arr, classes))[0]
    picked = []
    for c in classes:
        idx = np.where(labels_arr == c)[0]
        if len(idx) < k_shot:
            raise ValueError(f"k_shot={k_shot} exceeds available support shots for class "
                             f"'{c}' ({len(idx)} available in the bundled K-set)")
        picked.append(rng.choice(idx, size=k_shot, replace=False))
    return np.concatenate(picked)


# ---------------------------------------------------------------------------
# QYieldModel — the CLI-facing entry point.
# ---------------------------------------------------------------------------
class QYieldModel:
    """Loads the QYield checkpoint (+ its K-shot support set) ONCE, then serves
    predictions.

    Usage:
        model = QYieldModel()
        result = model.predict("/path/to/wafer.npy")
    """

    def __init__(self, device: str | None = None, ckpt_path: str | Path | None = None,
                 stems_dir: str | Path | None = None, kset_path: str | Path | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt_path = Path(ckpt_path) if ckpt_path else REPO_ROOT / DEFAULT_CKPT_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        kset_path = Path(kset_path) if kset_path else REPO_ROOT / DEFAULT_KSET_PATH
        if not kset_path.exists():
            raise FileNotFoundError(f"K-shot support set not found: {kset_path}")
        self.support_imgs_224, self.support_labels, self.classes = load_kset(kset_path)

        self.stems_dir = Path(stems_dir) if stems_dir else REPO_ROOT / DEFAULT_STEMS_DIR
        self._build_model()

        # Embed the full bundled K-set ONCE at load time. Prototypes are computed
        # PER PREDICTION CALL from a (possibly subsampled) slice of these cached
        # embeddings, so `n_way`/`k_shot`/`ways` can vary per call without re-embedding.
        with torch.no_grad():
            self.support_emb = self._embed(self.support_imgs_224)
        self.support_labels_arr = np.asarray(self.support_labels)
        self.protos = compute_prototypes(self.support_emb, list(self.support_labels), self.classes)

    # -- architecture construction -----------------------------------------------
    def _build_model(self):
        cfg = self.ckpt["config"]
        self.cfg = cfg
        net = QResHead(self.ckpt["head"], E=cfg["E"], m_modes=cfg["m_modes"],
                       add_modes=cfg["add_modes"], n_photons=cfg["n_photons"],
                       read_modes=cfg["read_modes"]).to(self.device)
        net.load_state_dict(self.ckpt["state_dict"])
        net.eval()
        self.net = net
        self.img_size = 224
        stem_paths = [self.stems_dir / Path(p).name for p in self.ckpt["ssl_stem_ckpts"]]
        missing = [p for p in stem_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing SSL stem checkpoint(s): {missing}")
        self.stems = [_load_stem(p, self.device) for p in stem_paths]
        self.colormap = self.ckpt["colormap"]

    # -- embedding -----------------------------------------------------------
    @torch.no_grad()
    def _embed(self, imgs_2d: np.ndarray) -> torch.Tensor:
        from matplotlib import colormaps
        lut = torch.tensor(colormaps[self.colormap](np.linspace(0, 1, 256))[:, :3],
                           dtype=torch.float32, device=self.device)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        x1 = torch.tensor(imgs_2d, dtype=torch.float32, device=self.device).unsqueeze(1)
        idx = (x1.squeeze(1).clamp(0, 1) * 255).long()
        rgb = lut[idx].permute(0, 3, 1, 2)
        xb = (rgb - mean) / std
        blocks = []
        for stem in self.stems:
            f = stem(xb).flatten(1)
            if self.cfg.get("norm_blocks", True):
                f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
            blocks.append(f)
        E = torch.cat(blocks, dim=1)
        return self.net(E)

    # -- few-shot episode resolution ---------------------------------------
    def _resolve_protos(self, n_way, k_shot, ways, seed):
        """Return the prototype dict to rank against for this call. With no
        n_way/k_shot/ways given, reuses the precomputed all-8-class/all-shots
        prototypes (fast path, no re-embedding)."""
        if n_way is None and k_shot is None and ways is None:
            return self.protos
        rng = np.random.default_rng(seed)
        classes = _select_episode_classes(self.classes, n_way, ways, rng)
        shot_idx = _select_shots(self.support_labels_arr, classes, k_shot, rng)
        return compute_prototypes(self.support_emb[shot_idx],
                                  self.support_labels_arr[shot_idx].tolist(), classes)

    # -- public API ------------------------------------------------------------
    def predict(self, image_path, n_way: int | None = None, k_shot: int | None = None,
               ways: list[str] | None = None, seed: int | None = None) -> dict:
        """image_path: path to a .npy raw wafer map ({0,1,2} ints, recommended) or a
        grayscale .png/.jpg.

        By default (n_way=k_shot=ways=None), classifies against ALL 8 bundled
        classes using every bundled support shot. Pass n_way/k_shot/ways to run a
        true few-shot episode (e.g. n_way=3, k_shot=5 for "3-way 5-shot") — see
        README.md."""
        query_img = load_query_image(image_path, self.img_size)
        with torch.no_grad():
            query_emb = self._embed(query_img[None])[0]
        protos = self._resolve_protos(n_way, k_shot, ways, seed)
        ranking = _predict(query_emb, protos)
        return {"predicted_class": ranking[0][0], "ranking": ranking,
                "episode_classes": list(protos.keys())}

    def predict_array(self, wafer_map: np.ndarray, n_way: int | None = None,
                      k_shot: int | None = None, ways: list[str] | None = None,
                      seed: int | None = None) -> dict:
        """Same as predict(), but takes an in-memory array directly instead of a
        file path. `wafer_map`: 2D array, either raw {0,1,2} ints or an
        already-normalized [0,1] float."""
        if wafer_map.dtype.kind in "iu" and wafer_map.max() <= 2:
            x = torch.tensor(wafer_map.astype(np.float32) / PIXEL_NORM_DIV)[None, None]
        else:
            x = torch.tensor(wafer_map.astype(np.float32))[None, None]
            if x.max() > 1.0:
                x = x / x.max()
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode=RESIZE_MODE, align_corners=False)
        query_img = x[0, 0].numpy().astype(np.float32)
        with torch.no_grad():
            query_emb = self._embed(query_img[None])[0]
        protos = self._resolve_protos(n_way, k_shot, ways, seed)
        ranking = _predict(query_emb, protos)
        return {"predicted_class": ranking[0][0], "ranking": ranking,
                "episode_classes": list(protos.keys())}
