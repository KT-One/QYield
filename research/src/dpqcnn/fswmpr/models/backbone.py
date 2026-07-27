"""Shared frozen Conv4 backbone (architecture.md §2).

Pretrains a standard few-shot Conv4 on the BASE classes (3-class cross-entropy),
then freezes it. Every head (A/B/C/A') consumes the SAME frozen embeddings; the
checkpoint hash is exposed so the driver can assert an identical backbone across
heads. Augmentation (flips + 90/180/270 rotations) is applied on-the-fly to the
base_train split ONLY (split-then-augment, architecture.md §1.1).

Run: python -m dpqcnn.experiments.fswmpr_backbone
"""

from __future__ import annotations

import hashlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P


class Conv4Backbone(nn.Module):
    """4× [conv3x3 - BN - ReLU - maxpool2] -> embedding.

    embed_mode="gap":     global-avg-pool -> embed_dim vector (shared/quantum regime).
    embed_mode="flatten": flattened conv map (rich; faithful ProtoNet-Conv4 anchor).
    """

    def __init__(self, channels=None, kernel_size=3, embed_dim=64, in_ch=1,
                 embed_mode="gap", img_size=None):
        super().__init__()
        channels = channels or C.BACKBONE["channels"]
        layers, c_prev = [], in_ch
        for c in channels:
            layers += [nn.Conv2d(c_prev, c, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.embed_mode = embed_mode
        if embed_mode == "gap":
            self.embed_dim = embed_dim
            self.proj = nn.Identity() if channels[-1] == embed_dim else nn.Linear(channels[-1], embed_dim)
        else:  # flatten
            img_size = img_size or C.IMG_SIZE
            with torch.no_grad():
                feat = self.features(torch.zeros(1, in_ch, img_size, img_size))
            self.embed_dim = int(feat.flatten(1).shape[1])
            self.proj = nn.Identity()

    def forward(self, x):                     # x: (B,1,H,W)
        h = self.features(x)
        if self.embed_mode == "gap":
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        else:
            h = h.flatten(1)
        return self.proj(h)


# ---------------------------------------------------------------------------
# On-the-fly base-train augmentation (split-then-augment).
# ---------------------------------------------------------------------------
def _augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    dev = x.device
    if C.BASE_AUGMENT.get("hflip") and torch.rand(1, generator=rng, device=dev).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if C.BASE_AUGMENT.get("vflip") and torch.rand(1, generator=rng, device=dev).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    rots = C.BASE_AUGMENT.get("rotations", [])
    if rots and torch.rand(1, generator=rng, device=dev).item() < 0.5:
        k = int(torch.randint(1, 4, (1,), generator=rng, device=dev).item())   # 90/180/270
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


def _load_split(split_name: str, device):
    X, y, split, meta = P.load()
    m = split == split_name
    x = torch.tensor(X[m], dtype=torch.float32, device=device)[:, None]  # (N,1,H,W)
    # relabel base classes to 0..len(BASE)-1 for the pretrain classifier
    base_map = {C.ALL_DEFECT_CLASSES.index(c): i for i, c in enumerate(C.BASE_CLASSES)}
    yy = torch.tensor([base_map.get(int(v), -1) for v in y[m]], dtype=torch.long, device=device)
    return x, yy


def ckpt_path(seed: int = None):
    seed = C.BACKBONE_SEED if seed is None else seed
    return C.CKPT_DIR / f"backbone_conv4_{P.preprocess_hash()}_seed{seed}.pt"


def backbone_hash(seed: int = None) -> str:
    sd = torch.load(ckpt_path(seed), map_location="cpu")["state_dict"]
    flat = b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())
    return hashlib.sha1(flat).hexdigest()[:12]


def pretrain(seed: int = None, force: bool = False):
    seed = C.BACKBONE_SEED if seed is None else seed
    cp = ckpt_path(seed)
    if cp.exists() and not force:
        print(f"[backbone] exists: {cp}")
        return cp
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)

    xtr, ytr = _load_split("base_train", dev)
    xva, yva = _load_split("base_val", dev)
    net = Conv4Backbone().to(dev)
    clf = nn.Linear(net.embed_dim, len(C.BASE_CLASSES)).to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()),
                           lr=C.BACKBONE["lr"], weight_decay=C.BACKBONE["weight_decay"])
    lossf = nn.CrossEntropyLoss()
    bs, n = C.BACKBONE["batch_size"], xtr.shape[0]
    best_va, best_state = -1.0, None
    print(f"[backbone] pretrain: {n} base_train, {xva.shape[0]} base_val, "
          f"{C.BACKBONE['pretrain_epochs']} epochs, dev={dev}")
    for ep in range(C.BACKBONE["pretrain_epochs"]):
        net.train(); clf.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = _augment(xtr[idx], gen)
            opt.zero_grad()
            loss = lossf(clf(net(xb)), ytr[idx])
            loss.backward(); opt.step()
        # val
        net.eval(); clf.eval()
        with torch.no_grad():
            va = (clf(net(xva)).argmax(1) == yva).float().mean().item()
        if va > best_va:
            best_va, best_state = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[backbone] epoch {ep+1}/{C.BACKBONE['pretrain_epochs']} val_acc={va:.4f} (best {best_va:.4f})")
    torch.save({"state_dict": best_state, "val_acc": best_va,
                "preprocess_hash": P.preprocess_hash(), "seed": seed,
                "embed_dim": net.embed_dim}, cp)
    print(f"[backbone] saved {cp} (best base_val acc {best_va:.4f}, hash {backbone_hash(seed)})")
    return cp


def load_backbone(seed: int = None, device=None) -> Conv4Backbone:
    seed = C.BACKBONE_SEED if seed is None else seed
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(ckpt_path(seed), map_location=dev)
    if d["preprocess_hash"] != P.preprocess_hash():
        raise RuntimeError("backbone/preprocess hash mismatch; retrain backbone")
    net = Conv4Backbone(embed_dim=d["embed_dim"]).to(dev)
    net.load_state_dict(d["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)                # FROZEN
    return net


if __name__ == "__main__":
    pretrain(force="--force" in sys.argv)
