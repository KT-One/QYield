""" — inference middleware CLI for the checkpointed quantum wafer-defect models.

True few-shot ProtoNet inference: both QConv4 and QResNet-ensemble classify a query by
NEAREST-PROTOTYPE distance to a K-shot support set, computed ON THE FLY (no fixed softmax
head, no baked-in prototypes in the checkpoint). This script:

  1. Loads the checkpoint (state_dict + config only) for the chosen model.
  2. Loads the bundled, pre-labeled K-shot support set (checkpoints/kset_k*.npz,
     built by bench/build_kset.py) — OR a user-supplied one with --kset.
  3. Embeds every support image through the model -> computes one prototype per class
     (mean embedding).
  4. Loads + preprocesses the user's query image (path to a .npy raw wafer map, {0,1,2}
     ints, OR a .png/.jpg grayscale image) the SAME way training data was preprocessed.
  5. Embeds the query, finds the nearest prototype (Euclidean), prints the predicted
     defect class + per-class distances (a rough confidence signal — smaller = closer).

Usage:
  uv run python -m dpqcnn.fswmpr.bench.infer --model qconv4 --image path/to/query.npy
  uv run python -m dpqcnn.fswmpr.bench.infer --model qresnet_ens --image path/to/query.npy

Note: this is a PROTOTYPE/DEMO CLI, not a reproduction of the paper's episodic accuracy
numbers — the K-set is fixed/curated (not resampled per query) and covers all 8 classes
(3 base + 5 novel), broader than the paper's 3-5-way episodic subsampling. See
README.md for the full checkpoint+inference workflow writeup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn

from .. import constants as C
from ..data import preprocess as P
from .build_kset import load_kset


# ---------------------------------------------------------------------------
# Query image loading + preprocessing (mirrors data/preprocess.py::_resize_norm,
# generalized to accept an arbitrary path instead of only WM-811K pickle rows).
# ---------------------------------------------------------------------------
def load_query_image(path: str, img_size: int) -> np.ndarray:
    """Load a query wafer map from disk and resize/normalize it EXACTLY like the
    training pipeline. Two accepted formats:
      * .npy  — raw wafer map, integer array with values in {0,1,2} (die states);
                this is the canonical WM-811K format and the recommended path.
      * .png/.jpg/... — a grayscale image already in [0,1] or [0,255]; treated as
                a rendered wafer map (best-effort; not the canonical format).
    Returns (img_size, img_size) float32 in [0,1], matching training normalization.
    """
    import torch.nn.functional as F
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"query image not found: {path}")

    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.dtype.kind in "iu" and arr.max() <= 2:      # canonical {0,1,2} wafer map
            x = torch.tensor(arr.astype(np.float32) / C.PIXEL_NORM_DIV)[None, None]
        else:                                               # already float/other scale
            x = torch.tensor(arr.astype(np.float32))[None, None]
            if x.max() > 1.0:
                x = x / x.max()
    else:
        from PIL import Image
        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.tensor(arr)[None, None]

    x = F.interpolate(x, size=(img_size, img_size), mode=C.RESIZE_MODE, align_corners=False)
    return x[0, 0].numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# QConv4 inference path
# ---------------------------------------------------------------------------
def build_qconv4(ckpt, device):
    from .stageb_quantum import Conv4Head
    cfg = ckpt["config"]
    net = Conv4Head(ckpt["head"], img_size=cfg["img_size"], channels=cfg["channels"],
                    d=cfg["d"], n_qpus=cfg["n_qpus"], add_modes=cfg["add_modes"],
                    n_photons=cfg["n_photons"], skip=cfg["skip"], readout=cfg["readout"],
                    read_modes=cfg["read_modes"], encode=cfg["encode"], m_modes=cfg["m_modes"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, cfg


@torch.no_grad()
def embed_qconv4(net, cfg, imgs_2d: np.ndarray, device):
    """imgs_2d: (N, H, W) float32 [0,1] already at cfg['img_size'] resolution."""
    x = torch.tensor(imgs_2d, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,H,W)
    if cfg["channels"] == 3:
        x = x.expand(-1, 3, -1, -1)
    return net(x)


# ---------------------------------------------------------------------------
# QResNet-ensemble inference path
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_qresnet_ens(ckpt, device):
    from . import qresnet_ssl as SSL1
    from . import qresnet_ssl2 as SSL2
    from .qresnet import QResHead
    cfg = ckpt["config"]
    net = QResHead(ckpt["head"], E=cfg["E"], m_modes=cfg["m_modes"], add_modes=cfg["add_modes"],
                   n_photons=cfg["n_photons"], read_modes=cfg["read_modes"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def load_any_stem(method):
        import torch.nn as _nn
        import torchvision
        if method == "simclr":
            cp_path = SSL1.stem_ckpt_path(ckpt["ssl_epochs"], ckpt["ssl_seed"], ckpt["colormap"])
        else:
            cp_path = SSL2.stem_ckpt_path(method, ckpt["ssl_epochs"], ckpt["ssl_seed"], ckpt["colormap"])
        c = torch.load(cp_path, map_location=device, weights_only=False)
        base = torchvision.models.resnet50(weights=None)
        stem = _nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                              base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool).to(device)
        stem.load_state_dict(c["stem_state_dict"])
        stem.eval()
        return stem

    stems = [load_any_stem(method) for method in ckpt["ssl_methods"]]
    return net, cfg, stems, ckpt["colormap"]


@torch.no_grad()
def embed_qresnet_ens(net, cfg, stems, colormap, imgs_2d: np.ndarray, device):
    """imgs_2d: (N, 224, 224) float32 [0,1]. Runs each SSL stem, jet-renders, GAP-2048,
    per-block L2-norm, concat -> QResHead."""
    from matplotlib import colormaps
    lut = torch.tensor(colormaps[colormap](np.linspace(0, 1, 256))[:, :3],
                       dtype=torch.float32, device=device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x1 = torch.tensor(imgs_2d, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,224,224)
    idx = (x1.squeeze(1).clamp(0, 1) * 255).long()
    rgb = lut[idx].permute(0, 3, 1, 2)                    # (N,3,224,224) jet-rendered
    xb = (rgb - mean) / std

    blocks = []
    for stem in stems:
        f = stem(xb).flatten(1)                            # (N, 2048)
        if cfg["norm_blocks"]:
            f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
        blocks.append(f)
    E = torch.cat(blocks, dim=1)                            # (N, 6144)
    return net(E)


# ---------------------------------------------------------------------------
# Shared few-shot prototype logic + CLI
# ---------------------------------------------------------------------------
def compute_prototypes(embeddings: torch.Tensor, labels: list[str], classes: list[str]):
    protos = {}
    labels_arr = np.asarray(labels)
    for c in classes:
        mask = labels_arr == c
        if mask.sum() == 0:
            continue
        protos[c] = embeddings[mask].mean(0)
    return protos


def predict(query_emb: torch.Tensor, protos: dict):
    names = list(protos.keys())
    P_ = torch.stack([protos[c] for c in names])            # (C, d)
    d = torch.cdist(query_emb[None], P_)[0]                 # (C,)
    order = torch.argsort(d)
    ranked = [(names[i], float(d[i])) for i in order.tolist()]
    return ranked


def main():
    ap = argparse.ArgumentParser(description="Few-shot inference CLI for QConv4 / QResNet-ensemble")
    ap.add_argument("--model", required=True, choices=["qconv4", "qresnet_ens"])
    ap.add_argument("--image", required=True, help="path to the unlabeled query image "
                    "(.npy raw wafer map {0,1,2} recommended, or grayscale .png/.jpg)")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: checkpoints/<model>.pt)")
    ap.add_argument("--kset", default=None, help="K-shot support set .npz "
                    "(default: newest checkpoints/kset_k*.npz — see bench/build_kset.py)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else C.CKPT_DIR / f"{args.model}.pt"
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}\n"
                 f"Train + save it first, e.g.:\n"
                 f"  uv run python -m dpqcnn.fswmpr.bench.stageb_quantum --heads quantum "
                 f"--seeds 42 --save-ckpt   # for qconv4\n"
                 f"  uv run python -m dpqcnn.fswmpr.bench.qresnet_ens --heads quantum "
                 f"--seeds 42 --save-ckpt --feats ...   # for qresnet_ens")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if args.kset:
        kset_path = Path(args.kset)
    else:
        candidates = sorted(C.CKPT_DIR.glob("kset_k*.npz"))
        if not candidates:
            sys.exit("no bundled K-shot support set found in checkpoints/ — build one first:\n"
                     "  uv run python -m dpqcnn.fswmpr.bench.build_kset --k 10 --seed 42")
        kset_path = candidates[-1]
    support_imgs_224, support_labels, classes = load_kset(kset_path)
    print(f"[infer] model={args.model} ckpt={ckpt_path.name} kset={kset_path.name} "
          f"({len(support_labels)} support imgs, {len(classes)} classes) device={device}")

    if args.model == "qconv4":
        net, cfg = build_qconv4(ckpt, device)
        img_size = cfg["img_size"]
        if img_size != 224:
            import torch.nn.functional as F
            support_imgs = np.stack([
                F.interpolate(torch.tensor(im)[None, None], size=(img_size, img_size),
                              mode=C.RESIZE_MODE, align_corners=False)[0, 0].numpy()
                for im in support_imgs_224
            ])
        else:
            support_imgs = support_imgs_224
        support_emb = embed_qconv4(net, cfg, support_imgs, device)
        query_img = load_query_image(args.image, img_size)
        query_emb = embed_qconv4(net, cfg, query_img[None], device)[0]
    else:
        net, cfg, stems, colormap = build_qresnet_ens(ckpt, device)
        support_emb = embed_qresnet_ens(net, cfg, stems, colormap, support_imgs_224, device)
        query_img = load_query_image(args.image, 224)
        query_emb = embed_qresnet_ens(net, cfg, stems, colormap, query_img[None], device)[0]

    protos = compute_prototypes(support_emb, list(support_labels), classes)
    ranked = predict(query_emb, protos)

    print(f"\n=== Prediction for {args.image} ===")
    print(f"  Predicted defect class : {ranked[0][0]}  (distance {ranked[0][1]:.3f})")
    print(f"  Runner-up              : {ranked[1][0]}  (distance {ranked[1][1]:.3f})")
    print(f"\n  Full ranking (nearest -> farthest):")
    for cls, dist in ranked:
        tag = "BASE " if cls in C.BASE_CLASSES else "NOVEL"
        print(f"    [{tag}] {cls:10s} dist={dist:.3f}")
    print(f"\n  (Prototypes computed on the fly from {len(support_labels)} bundled K-shot "
          f"support images — see bench/build_kset.py. This is a prototype/demo CLI, not "
          f"the paper's episodic-accuracy protocol.)")


if __name__ == "__main__":
    main()
