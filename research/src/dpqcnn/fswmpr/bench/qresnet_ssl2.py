""" Phase 1 — SSL-method ABLATION for QResNet (find a more-stable feature extractor).

Compares SSL objectives for in-domain ResNet50 pretraining on base wafers, then attaches the same
baseline/quantum/classical prototype heads (reusing bench/qresnet.py). Methods:
  * simclr   — NT-Xent contrastive (needs negatives / large batch; higher variance)
  * barlow   — Barlow Twins (redundancy reduction; no negatives; very stable)
  * vicreg   — Variance-Invariance-Covariance (no negatives; stable)
  * simsiam  — stop-gradient predictor (no negatives, no momentum; simple, stable)

Same defect-preserving aug + jet-RGB as qresnet_ssl.py. SSL epochs chosen by loss convergence (~60), NOT
by novel accuracy. Run per method; a driver sweeps all four.

Run: uv run python -m dpqcnn.fswmpr.bench.qresnet_ssl2 --method barlow --ssl-epochs 60 --heads baseline quantum classical
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T2

from .. import constants as C
from ..data import preprocess as P
from ..models.ssl_pretrain import nt_xent
from . import qresnet as QR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _mlp(dims, bn=True, last_bn=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=not (bn and i < len(dims) - 2)))
        last = (i == len(dims) - 2)
        if (not last) and bn:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
        elif last and last_bn:
            layers.append(nn.BatchNorm1d(dims[i + 1], affine=False))
    return nn.Sequential(*layers)


# ---- SSL losses ------------------------------------------------------------
def barlow_loss(z1, z2, lmbda=5e-3):
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    Cc = (z1.T @ z2) / B
    on = ((torch.diagonal(Cc) - 1) ** 2).sum()
    off = (Cc ** 2).sum() - (torch.diagonal(Cc) ** 2).sum()
    return on + lmbda * off


def vicreg_loss(z1, z2, sim_w=25.0, std_w=25.0, cov_w=1.0):
    sim = F.mse_loss(z1, z2)

    def _std(z):
        return torch.mean(F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)))

    def _cov(z):
        B, D = z.shape
        z = z - z.mean(0)
        cov = (z.T @ z) / (B - 1)
        return ((cov ** 2).sum() - (torch.diagonal(cov) ** 2).sum()) / D

    return sim_w * sim + std_w * (_std(z1) + _std(z2)) + cov_w * (_cov(z1) + _cov(z2))


def simsiam_loss(p1, z1, p2, z2):
    z1, z2 = z1.detach(), z2.detach()
    return -0.5 * (F.cosine_similarity(p1, z2, dim=1).mean() + F.cosine_similarity(p2, z1, dim=1).mean())


def feat_path(method, epochs, seed, colormap):
    return C.PROCESSED_DIR / f"resemb_ssl2_{method}_{colormap}_ep{epochs}_s{seed}.npy"


def stem_ckpt_path(method, epochs, seed, colormap):
    """Checkpoint for the SSL-pretrained ResNet50 STEM itself (not just its cached
    embeddings) — required to embed genuinely NEW images at inference time, since
    the .npy cache only covers the fixed WM-811K rows seen during training/eval."""
    return C.CKPT_DIR / f"ssl2_stem_{method}_{colormap}_ep{epochs}_s{seed}.pt"


def load_stem(method, device, epochs=60, seed=42, colormap="jet"):
    """Rebuild a ResNet50 stem (conv1..avgpool, GAP-2048 output) and load SSL-pretrained
    weights saved by ssl_pretrain(..., save_stem=True). For inference on new images."""
    import torch.nn as _nn
    import torchvision
    ckpt = torch.load(stem_ckpt_path(method, epochs, seed, colormap), map_location=device, weights_only=False)
    net = torchvision.models.resnet50(weights=None)
    stem = _nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                          net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool).to(device)
    stem.load_state_dict(ckpt["stem_state_dict"])
    stem.eval()
    return stem, ckpt["colormap"]


def ssl_pretrain(method, device, epochs, bs, lr, seed, colormap="jet", force=False, save_stem=True):
    fp = feat_path(method, epochs, seed, colormap)
    if fp.exists() and not force:
        print(f"[ssl2/{method}] cached: {fp}")
        return fp
    import torchvision
    from matplotlib import colormaps
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, y, split, meta = P.load(img_size=224)
    xt = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
    base_idx = np.where(split == "base_train")[0]
    lut = torch.tensor(colormaps[colormap](np.linspace(0, 1, 256))[:, :3],
                       dtype=torch.float32, device=device) if colormap else None
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool).to(device)
    if method == "simclr":
        proj = _mlp([2048, 512, 128], bn=False).to(device); pred = None
    elif method in ("barlow", "vicreg"):
        proj = _mlp([2048, 2048, 2048, 2048], bn=True, last_bn=(method == "barlow")).to(device); pred = None
    elif method == "simsiam":
        proj = _mlp([2048, 2048, 2048, 2048], bn=True, last_bn=True).to(device)
        pred = _mlp([2048, 512, 2048], bn=True).to(device)
    else:
        raise ValueError(method)

    aug = T2.Compose([T2.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True),
                      T2.RandomHorizontalFlip(), T2.RandomVerticalFlip(),
                      T2.RandomRotation(30), T2.ColorJitter(0.2, 0.2, 0.2)])

    def render(x1):
        if lut is not None:
            idx = (x1.squeeze(1).clamp(0, 1) * 255).long()
            return lut[idx].permute(0, 3, 1, 2)
        return x1.expand(-1, 3, -1, -1)

    def feat(x1):
        rgb = aug(render(x1))
        return stem(((rgb - mean) / std)).flatten(1)

    params = list(stem.parameters()) + list(proj.parameters()) + (list(pred.parameters()) if pred else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    n = len(base_idx)
    stem.train(); proj.train()
    if pred:
        pred.train()
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            xb = xt[base_idx[perm[i:i + bs]]]
            if xb.shape[0] < 4:
                continue
            f1, f2 = feat(xb), feat(xb)
            if method == "simclr":
                z1 = F.normalize(proj(f1), dim=1); z2 = F.normalize(proj(f2), dim=1)
                loss = nt_xent(torch.cat([z1, z2], 0))
            elif method == "barlow":
                loss = barlow_loss(proj(f1), proj(f2))
            elif method == "vicreg":
                loss = vicreg_loss(proj(f1), proj(f2))
            else:  # simsiam
                z1, z2 = proj(f1), proj(f2)
                loss = simsiam_loss(pred(z1), z1, pred(z2), z2)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[ssl2/{method}] ep {ep+1}/{epochs} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    stem.eval()
    embs = np.zeros((X.shape[0], 2048), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], 256):
            rgb = render(xt[i:i + 256])
            embs[i:i + 256] = stem(((rgb - mean) / std)).flatten(1).float().cpu().numpy()
    np.save(fp, embs)
    print(f"[ssl2/{method}] saved {fp} ({time.time()-t0:.0f}s)")

    if save_stem:
        C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        cp = stem_ckpt_path(method, epochs, seed, colormap)
        torch.save({"stem_state_dict": stem.state_dict(), "method": method, "colormap": colormap,
                   "epochs": epochs, "seed": seed, "img_size": 224, "out_dim": 2048}, cp)
        print(f"[ssl2/{method}] saved stem checkpoint {cp} (needed for inference on new images)")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["simclr", "barlow", "vicreg", "simsiam"])
    ap.add_argument("--heads", nargs="+", default=["baseline", "quantum", "classical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--ssl-epochs", type=int, default=60)
    ap.add_argument("--ssl-bs", type=int, default=256)
    ap.add_argument("--ssl-lr", type=float, default=1e-3)
    ap.add_argument("--ssl-seed", type=int, default=42)
    ap.add_argument("--colormap", default="jet")
    ap.add_argument("--m-modes", type=int, default=4)
    ap.add_argument("--add-modes", type=int, default=1)
    ap.add_argument("--n-photons", type=int, default=2)
    ap.add_argument("--read-modes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train-q", type=int, default=15)
    ap.add_argument("--eval-q", type=int, default=20)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp = ssl_pretrain(args.method, device, args.ssl_epochs, args.ssl_bs, args.ssl_lr,
                      args.ssl_seed, args.colormap)
    E, base_by, novel_by = QR.pools(fp, device)
    tasks = [("3w5s", 3, 5), ("3w10s", 3, 10), ("5w5s", 5, 5)]
    print(f"[ssl2/{args.method}] E={E.shape} heads={args.heads} seeds={args.seeds}", flush=True)
    results = {h: {t[0]: [] for t in tasks} for h in args.heads}
    meta = {}
    for head in args.heads:
        for seed in args.seeds:
            net, nparams, out_dim = QR.meta_train(head, seed, device, E, base_by, args)
            meta[head] = {"n_params": nparams}
            for name, nw, k in tasks:
                r = QR.evaluate(net, device, E, novel_by, nw, k, args.eval_q, 100, seed)
                results[head][name].append(r["acc"])
                print(f"[ssl2/{args.method}:{head}] seed{seed} {name}: {r['acc']:.2f}±{r['ci95']:.2f}", flush=True)

    print(f"\n=== SSL-ablation method={args.method} — CNN SOTA 82.61 @ 3w5s ===")
    summary = {}
    for head in args.heads:
        summary[head] = {}
        line = [f"{head:9s}"]
        for name, _, _ in tasks:
            v = np.array(results[head][name]); m = v.mean()
            sci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            summary[head][name] = {"mean": float(m), "ci95": float(sci), "per_seed": v.tolist()}
            line.append(f"{name} {m:.2f}±{sci:.2f}")
        print("  " + " | ".join(line))
    out = C.RESULTS_DIR / f"{args.tag or ('qresnet_ssl2_' + args.method)}.json"
    out.write_text(json.dumps({"config": vars(args), "summary": summary}, indent=2))
    print(f"[ssl2/{args.method}] saved {out}")


if __name__ == "__main__":
    main()
