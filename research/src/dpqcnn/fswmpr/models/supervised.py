"""Supervised 8-class baseline (Gong-Lin / Yu2023 anchor).

The REPRODUCIBLE, same-data anchor: we extract the *exact* Gong-Lin subset
(25,519 real-defect wafers, per-class counts verified, 42x42 + /2), so their
supervised 8-class numbers are directly comparable. Reproducing them (~92%
Gong-Lin VGG16 / 95.4% Yu2023 WM-PeleeNet) validates the whole data+backbone
pipeline and gives a fully-supervised ceiling for the few-shot curves.

Unlike C1's few-shot 78.40% (a DIFFERENT, unreleased 17,805-wafer subset — not
reproducible from the paper), this anchor is on identical data and IS
reproducible; we hit ~94%.

Run: python -m dpqcnn.experiments.fswmpr_supervised --seed 42
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P
from .backbone import Conv4Backbone, _augment


def _split_all(y, seed, frac=0.7):
    """Stratified train/test over ALL 8 classes (Gong-Lin supervised task)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(len(C.ALL_DEFECT_CLASSES)):
        ci = np.where(y == c)[0]; rng.shuffle(ci); n = int(len(ci) * frac)
        tr += ci[:n].tolist(); te += ci[n:].tolist()
    return np.array(tr), np.array(te)


def run(seed=42, epochs=60, img_size=None):
    img_size = img_size or C.IMG_SIZE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X, y, split, meta = P.load(img_size=img_size)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)[:, None]
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    tr, te = _split_all(y, seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)

    bb = Conv4Backbone(embed_mode="gap", img_size=img_size).to(dev)
    clf = nn.Linear(bb.embed_dim, len(C.ALL_DEFECT_CLASSES)).to(dev)
    opt = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()),
                           lr=C.BACKBONE["lr"], weight_decay=C.BACKBONE["weight_decay"])
    t0 = time.time()
    for ep in range(epochs):
        bb.train(); clf.train()
        perm = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(perm), C.BACKBONE["batch_size"]):
            idx = torch.tensor(perm[i:i + C.BACKBONE["batch_size"]], device=dev)
            opt.zero_grad()
            F.cross_entropy(clf(bb(_augment(Xt[idx], gen))), yt[idx]).backward()
            opt.step()
    bb.eval(); clf.eval()
    with torch.no_grad():
        pred = torch.cat([clf(bb(Xt[torch.tensor(te[i:i + 512], device=dev)])).argmax(1)
                          for i in range(0, len(te), 512)])
        yte = yt[torch.tensor(te, device=dev)]
        acc = (pred == yte).float().mean().item()
        # per-class recall + macro-F1
        recs, f1s = [], []
        for c in range(len(C.ALL_DEFECT_CLASSES)):
            tp = ((pred == c) & (yte == c)).sum().item()
            fn = ((pred != c) & (yte == c)).sum().item()
            fp = ((pred == c) & (yte != c)).sum().item()
            r = tp / (tp + fn) if tp + fn else 0.0
            pr = tp / (tp + fp) if tp + fp else 0.0
            recs.append(r); f1s.append(2 * pr * r / (pr + r) if pr + r else 0.0)
    print(f"[supervised 8-class @{img_size}px seed{seed}] test_acc={acc*100:.2f}% "
          f"macro_recall={np.mean(recs)*100:.2f}% macro_f1={np.mean(f1s)*100:.2f}% ({time.time()-t0:.0f}s)")
    print("  vs Gong-Lin VGG16 ~92% / Yu2023 WM-PeleeNet 95.4% (same 25,519 subset)")
    return {"acc": acc * 100, "macro_recall": float(np.mean(recs)) * 100,
            "macro_f1": float(np.mean(f1s)) * 100, "seed": seed, "img_size": img_size}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    run(a.seed, a.epochs)
