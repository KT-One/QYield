"""model_l4.py — inference module for the QYield **L4** model (`asi_L4`): a single
frozen ResNet50-jet backbone + the depth-4 Adaptive State Injection (ASI) head.

This is the model behind the *novelty-detection* pitch (open-set AUROC), distinct
from the QResNet-ensemble in `model.py` (the closed-set accuracy model). It is
lighter — one ResNet50 backbone instead of three SSL stems — and its ~164k-param
photonic head is what improves flagging never-seen defect types.

    from qyield.model_l4 import QYieldL4Model

    model = QYieldL4Model()
    result = model.predict("/path/to/query_wafer.npy")
    # {"predicted_class": "Scratch", "ranking": [("Scratch", 0.42), ...], "novelty_score": ...}

Same few-shot ProtoNet protocol and public API as `QYieldModel` (predict /
predict_array / build_support_set* / n_way / k_shot / ways), against the same
bundled K-shot support set. It additionally returns a `novelty_score` (the
mindist open-set score the model was selected on).

Zero dependency on the DP-QCNN research package; needs only the bundled
`checkpoints/asi_l4/` files + the K-set. Backbone is a stock torchvision
ResNet50 loaded with the exact ImageNet weights the training features used
(verified bit-identical to the research pipeline's timm backbone), so no `timm`
dependency is required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .constants import (
    DEFAULT_KSET_PATH,
    DEFAULT_L4_BACKBONE_PATH,
    DEFAULT_L4_CKPT_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PIXEL_NORM_DIV,
    RESIZE_MODE,
)
# Reuse the shared, already-tested helpers from the ensemble module — no duplication.
from .model import (
    REPO_ROOT,
    _build_resnet50_stem,
    _predict,
    _select_episode_classes,
    _select_shots,
    compute_prototypes,
    load_kset,
    load_query_image,
)

import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ASIDeepHead — the depth-stacked Adaptive State Injection head (`asi_deep`),
# copied verbatim from the research RegHead(kind="asi_deep") forward so the
# bundled state_dict (conv_skew, exp_skew) loads and computes identically.
# ---------------------------------------------------------------------------
class ASIDeepHead(nn.Module):
    """N per-QPU adaptive circuits, depth L. Per layer: orthogonal conv U_l →
    Born gate g_o=v_o² (+ rest bucket) → gated mixture Σ_o g_o·(V_l^{(o)} v) →
    renormalize. Readout = squared occupation of the first `read_modes` modes.
    All transforms are norm-preserving (orthogonal + renorm), so the head adds
    data-dependent routing without collapsing the feature geometry."""

    def __init__(self, E=2048, m=4, n_pool=3, depth=4, read_modes=2, **_ignored):
        super().__init__()
        assert E % m == 0, f"E={E} not divisible by m={m}"
        self.E, self.m, self.n_qpus = E, m, E // m
        self.n_pool = min(n_pool, m - 1)
        self.n_exp = self.n_pool + 1
        self.depth = depth
        self.read = min(read_modes, m)
        self.conv_skew = nn.Parameter(torch.zeros(self.n_qpus, depth, m, m))
        self.exp_skew = nn.Parameter(torch.zeros(self.n_qpus, depth, self.n_exp, m, m))
        self.out_dim = self.n_qpus * self.read

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        B = e.shape[0]
        v = e.reshape(B, self.n_qpus, self.m)
        v = v / (v.norm(dim=2, keepdim=True) + 1e-8)                 # per-QPU L2 normalize
        for l in range(self.depth):
            U = torch.matrix_exp(self.conv_skew[:, l] - self.conv_skew[:, l].transpose(-1, -2))
            v = torch.einsum("nij,bnj->bni", U, v)                   # orthogonal conv
            gp = v[:, :, :self.n_pool] ** 2
            g = torch.cat([gp, (1.0 - gp.sum(-1, keepdim=True)).clamp_min(0.0)], -1)  # Born gate
            Vl = torch.matrix_exp(self.exp_skew[:, l] - self.exp_skew[:, l].transpose(-1, -2))
            Vv = torch.einsum("noij,bnj->bnoi", Vl, v)               # (B,N,n_exp,m) experts
            v = torch.einsum("bno,bnoi->bni", g, Vv)                 # gated mixture
            v = v / (v.norm(dim=2, keepdim=True) + 1e-8)             # renorm for next layer
        return (v * v)[:, :, :self.read].reshape(B, -1)              # squared-occupation readout


# ---------------------------------------------------------------------------
# QYieldL4Model — CLI-facing entry point for the asi_L4 novelty model.
# ---------------------------------------------------------------------------
class QYieldL4Model:
    """Loads the asi_L4 checkpoint (+ ResNet50-jet backbone + K-shot support set)
    once, then serves predictions with an open-set novelty score."""

    def __init__(self, device: str | None = None, ckpt_path: str | Path | None = None,
                 backbone_path: str | Path | None = None, kset_path: str | Path | None = None):
        ckpt_path = Path(ckpt_path) if ckpt_path else REPO_ROOT / DEFAULT_L4_CKPT_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(f"asi_L4 checkpoint not found: {ckpt_path}")
        self._backbone_path = Path(backbone_path) if backbone_path else REPO_ROOT / DEFAULT_L4_BACKBONE_PATH
        if not self._backbone_path.exists():
            raise FileNotFoundError(f"asi_L4 backbone not found: {self._backbone_path}")

        kset_path = Path(kset_path) if kset_path else REPO_ROOT / DEFAULT_KSET_PATH
        if not kset_path.exists():
            raise FileNotFoundError(f"K-shot support set not found: {kset_path}")
        self.support_imgs_224, self.support_labels, self.classes = load_kset(kset_path)
        self._ckpt_path = ckpt_path

        auto = device is None
        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._build_on_device(chosen)
        except RuntimeError as exc:
            if "CUDA" in str(exc) and (auto or chosen == "cuda"):
                import warnings
                warnings.warn(f"CUDA inference failed ({exc.__class__.__name__}); "
                              "falling back to CPU.", stacklevel=2)
                self._build_on_device("cpu")
            else:
                raise

    def _build_on_device(self, device: str) -> None:
        self.device = torch.device(device)
        self.ckpt = torch.load(self._ckpt_path, map_location=self.device, weights_only=False)
        self._build_model()
        with torch.no_grad():
            self.support_emb = self._embed(self.support_imgs_224)
        self.support_labels_arr = np.asarray(self.support_labels)
        self._bundled_pool = (self.support_emb, self.support_labels_arr, list(self.classes))
        self._set_active_pool(*self._bundled_pool)

    def _build_model(self):
        cfg = self.ckpt["config"]
        self.cfg = cfg
        self.colormap = cfg.get("colormap", "jet")
        self.img_size = 224
        head = ASIDeepHead(E=cfg["E"], m=cfg["m"], n_pool=cfg["n_pool"],
                           depth=cfg["depth"], read_modes=cfg["read_modes"]).to(self.device)
        head.load_state_dict(self.ckpt["state_dict"])
        head.eval()
        self.head = head
        # single frozen ResNet50-jet backbone (torchvision, ImageNet weights from the bundle)
        stem = _build_resnet50_stem(self.device)
        bb = torch.load(self._backbone_path, map_location=self.device, weights_only=False)
        stem.load_state_dict(bb["stem_state_dict"])
        stem.eval()
        self.stem = stem

    @torch.no_grad()
    def _embed(self, imgs_2d: np.ndarray) -> torch.Tensor:
        """gray (N,224,224)[0,1] -> jet RGB -> ImageNet norm -> ResNet50 GAP-2048 -> ASI head.
        Uses matplotlib `cmap(gray)` directly (matching how the training features were
        built); no post-normalization on the 2048-d embedding (the head normalizes per-QPU)."""
        from matplotlib import colormaps
        cmap = colormaps[self.colormap]
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        gray = np.asarray(imgs_2d, dtype=np.float32)
        rgb = cmap(gray)[..., :3]                                    # (N,224,224,3)
        xb = torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        xb = (xb - mean) / std
        feat = self.stem(xb).flatten(1)                              # GAP-2048
        return self.head(feat)

    # -- support-set management (mirrors QYieldModel) -----------------------
    def _set_active_pool(self, emb, labels, classes) -> None:
        self._active_emb = emb
        self._active_labels = np.asarray(labels)
        self._active_classes = list(classes)
        self._active_protos = compute_prototypes(
            self._active_emb, self._active_labels.tolist(), self._active_classes)
        self.protos = self._active_protos

    @property
    def active_classes(self) -> list[str]:
        return list(self._active_classes)

    @property
    def using_custom_support(self) -> bool:
        return self._active_emb is not self._bundled_pool[0]

    def build_support_set(self, images: np.ndarray, labels: list[str]) -> None:
        images = np.asarray(images, dtype=np.float32)
        labels = list(labels)
        if len(images) != len(labels):
            raise ValueError(f"images ({len(images)}) and labels ({len(labels)}) length mismatch")
        if len(labels) == 0:
            raise ValueError("need at least one labelled image to build a support set")
        with torch.no_grad():
            emb = self._embed(images)
        classes = list(dict.fromkeys(labels))
        self._set_active_pool(emb, labels, classes)

    def build_support_set_from_dir(self, root: str | Path) -> None:
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"support dir not found: {root}")
        imgs, labels = [], []
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".npy", ".png", ".jpg", ".jpeg"):
                    imgs.append(load_query_image(f, self.img_size))
                    labels.append(cls_dir.name)
        if not imgs:
            raise ValueError(f"no images found under {root}/<class>/ subfolders")
        self.build_support_set(np.stack(imgs), labels)

    def reset_support_set(self) -> None:
        self._set_active_pool(*self._bundled_pool)

    def _resolve_protos(self, n_way, k_shot, ways, seed):
        if n_way is None and k_shot is None and ways is None:
            return self._active_protos
        rng = np.random.default_rng(seed)
        classes = _select_episode_classes(self._active_classes, n_way, ways, rng)
        shot_idx = _select_shots(self._active_labels, classes, k_shot, rng)
        return compute_prototypes(self._active_emb[shot_idx],
                                  self._active_labels[shot_idx].tolist(), classes)

    # -- prediction ---------------------------------------------------------
    def _result(self, query_emb, protos) -> dict:
        ranking = _predict(query_emb, protos)
        # open-set novelty score = -min prototype distance (higher = more "known");
        # this is the mindist score the L4 model was selected on for novelty AUROC.
        novelty_score = -ranking[0][1]
        return {"predicted_class": ranking[0][0], "ranking": ranking,
                "episode_classes": list(protos.keys()), "novelty_score": float(novelty_score)}

    def predict(self, image_path, n_way: int | None = None, k_shot: int | None = None,
                ways: list[str] | None = None, seed: int | None = None) -> dict:
        query_img = load_query_image(image_path, self.img_size)
        with torch.no_grad():
            query_emb = self._embed(query_img[None])[0]
        return self._result(query_emb, self._resolve_protos(n_way, k_shot, ways, seed))

    def predict_array(self, wafer_map: np.ndarray, n_way: int | None = None,
                      k_shot: int | None = None, ways: list[str] | None = None,
                      seed: int | None = None) -> dict:
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
        return self._result(query_emb, self._resolve_protos(n_way, k_shot, ways, seed))
