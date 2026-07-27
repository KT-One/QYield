"""E4 — self-supervised encoder pretrain on BASE (architecture.md §1 E4).

SimCLR-style NT-Xent contrastive pretraining of an encoder on the BASE-class
images (labels UNUSED), producing a representation used as an initialization for
the downstream few-shot models. Works on any encoder exposing `.embed_dim` and
`forward(x)->(B,embed_dim)` — i.e. both the classical `Conv4Backbone` and the
DPQCNN `ConvEncoder` — so E4 is a clean one-factor change (encoder init) applied
symmetrically to both models.

Honest note: this lifts the *encoder* (classical part) of both models; it is
expected to move absolute accuracy, not necessarily the quantum-vs-classical gap.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P
from .backbone import _augment


def _proj_head(dim, out=64):
    return nn.Sequential(nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Linear(128, out))


def nt_xent(z, temp=0.5):
    """NT-Xent over 2N normalized embeddings (z = cat[view1, view2], (2N,d))."""
    n2 = z.shape[0]
    n = n2 // 2
    sim = (z @ z.t()) / temp                       # (2N,2N)
    sim.fill_diagonal_(-9e15)                      # mask self
    targets = torch.arange(n2, device=z.device)
    targets = (targets + n) % n2                   # positive = the other view
    return F.cross_entropy(sim, targets)


def _base_images(dev, img_size):
    X, y, split, meta = P.load(img_size=img_size)
    m = split == "base_train"
    return torch.tensor(X[m], dtype=torch.float32, device=dev)[:, None]


def contrastive_pretrain(encoder, dev, seed, epochs=40, bs=256, lr=1e-3, img_size=None,
                         verbose=True):
    """Train `encoder` in place with NT-Xent on BASE augmentations. Returns encoder."""
    img_size = img_size or C.IMG_SIZE
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    Xb = _base_images(dev, img_size)
    head = _proj_head(encoder.embed_dim).to(dev)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    n = Xb.shape[0]
    t0 = time.time(); encoder.train(); head.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = Xb[idx]
            v1, v2 = _augment(x, gen), _augment(x, gen)
            z1 = F.normalize(head(encoder(v1)), dim=1)
            z2 = F.normalize(head(encoder(v2)), dim=1)
            loss = nt_xent(torch.cat([z1, z2], 0))
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"[ssl] ep {ep+1}/{epochs} nt_xent={loss.item():.3f} ({time.time()-t0:.0f}s)")
    encoder.eval()
    return encoder


if __name__ == "__main__":
    import argparse
    from .backbone import Conv4Backbone
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Conv4Backbone(embed_mode="gap").to(dev)
    contrastive_pretrain(enc, dev, a.seed, epochs=a.epochs)
    print("[ssl] smoke pretrain done")
