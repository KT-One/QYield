""" Phase 1 — QResNet: DP-QCNN photonic head on a ResNet50 (GAP-2048) backbone.

Goal: beat the CNN SOTA (C1 ProtoNet-ResNet50 = 82.61 @ 3w5s), quantum-attributed. Mirrors the confirmed
QConv4 recipe with the backbone swapped to ResNet50; the quantum head consumes the SAME GAP-2048 feature
the baseline uses (locked decision). Reports baseline / quantum / matched-classical at the same config.

Backbone: ResNet50 (ImageNet-pretrained, timm), FROZEN + cached GAP-2048 embedding (fast; the `baseline`
head on it = our reproduced frozen-feature ProtoNet-ResNet50). Quantum head: reshape 2048 → 512 QPUs × 4
modes (NO compression), 2 photons, partial measurement (read 2 of 5 modes), NO skip. Classical control:
matched free quadratic forms. Episodic 3-way 5-shot, 100-episode Q=20 eval, euclidean prototypes.

Run: uv run python -m dpqcnn.fswmpr.bench.qresnet --heads baseline quantum classical --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P
from ..models.protonet import proto_predict, _macro_recall_f1
from .repro_c1 import sample_episode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cache_path(model_name, img_size, colormap=None):
    cm = f"_{colormap}" if colormap else ""
    return C.PROCESSED_DIR / f"resemb_{model_name.replace('.', '_')}_{img_size}{cm}.npy"


def build_cache(model_name="resnet50", img_size=224, batch=256, force=False, colormap=None):
    out = cache_path(model_name, img_size, colormap)
    if out.exists() and not force:
        print(f"[res-cache] up-to-date: {out}")
        return out
    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, split, meta = P.load(img_size=img_size)
    cmap = None
    if colormap:                                     # render grayscale [0,1] -> true RGB via colormap
        from matplotlib import colormaps
        cmap = colormaps[colormap]
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    N = X.shape[0]
    embs = np.zeros((N, model.num_features), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch):
            gray = X[i:i + batch]                    # (b,224,224) in [0,1]
            if cmap is not None:
                rgb = cmap(gray)[..., :3]            # (b,224,224,3) via colormap
                xb = torch.tensor(rgb, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            else:
                xb = torch.tensor(gray, dtype=torch.float32, device=device)[:, None].expand(-1, 3, -1, -1)
            xb = (xb - mean) / std
            embs[i:i + batch] = model(xb).float().cpu().numpy()   # GAP-pooled 2048-d
            if (i // batch) % 20 == 0:
                print(f"[res-cache] {i+xb.shape[0]}/{N} ({time.time()-t0:.0f}s)", flush=True)
    np.save(out, embs)
    print(f"[res-cache] saved {out} shape={embs.shape}")
    return out


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
        self.out_dim = self.n_qpus * self.read_modes           # skip=none
        if head == "quantum":
            from ..models.multiphoton_core import MultiPhotonQPUBank
            self.bank = MultiPhotonQPUBank(self.n_qpus, self.m, add_modes, n_photons,
                                           C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
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


def pools(emb_path, device):
    X, y, split, meta = P.load(img_size=224)
    E = torch.tensor(np.load(emb_path), dtype=torch.float32, device=device)
    cls = meta["classes"]
    base_by = {c: np.where((split == "base_train") & (y == cls.index(c)))[0] for c in C.BASE_CLASSES}
    novel_by = {c: np.where((split == "novel_pool") & (y == cls.index(c)))[0] for c in C.NOVEL_CLASSES}
    return E, base_by, novel_by


def meta_train(head, seed, device, E, base_by, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = QResHead(head, E=E.shape[1], m_modes=args.m_modes, add_modes=args.add_modes,
                   n_photons=args.n_photons, read_modes=args.read_modes).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    if nparams == 0:                                            # baseline (frozen) — no training
        net.eval()
        return net, 0, net.out_dim
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    total = args.epochs * args.episodes
    for it in range(total):
        classes = list(rng.choice(C.BASE_CLASSES, size=3, replace=False))
        si, sl, qi, ql = sample_episode(base_by, classes, 5, args.train_q, rng)
        se = net(E[si]); qe = net(E[qi])
        sl_t = torch.tensor(sl, device=device)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(3)])
        loss = F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(ql, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, nparams, net.out_dim


@torch.no_grad()
def evaluate(net, device, E, novel_by, n_way, k, q, n_episodes, eval_seed):
    rng = np.random.default_rng(100000 + eval_seed)
    pool = list(C.NOVEL_CLASSES)
    accs, f1s = [], []
    for _ in range(n_episodes):
        classes = list(rng.choice(pool, size=n_way, replace=False))
        si, sl, qi, ql = sample_episode(novel_by, classes, k, q, rng)
        se = net(E[si]); qe = net(E[qi])
        pred = proto_predict(se, torch.tensor(sl, device=device), qe, n_way)
        true = torch.tensor(ql, device=device)
        accs.append((pred == true).float().mean().item())
        _, f = _macro_recall_f1(pred, true, n_way)
        f1s.append(f)
    a = np.array(accs)
    ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return {"acc": float(a.mean()) * 100, "ci95": float(ci) * 100,
            "macro_f1": float(np.mean(f1s)) * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["baseline", "quantum", "classical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--colormap", default=None, help="matplotlib colormap to render wafers as RGB (e.g. viridis, jet)")
    ap.add_argument("--m-modes", type=int, default=4)
    ap.add_argument("--add-modes", type=int, default=1)
    ap.add_argument("--n-photons", type=int, default=2)
    ap.add_argument("--read-modes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train-q", type=int, default=15)
    ap.add_argument("--eval-q", type=int, default=20)
    ap.add_argument("--tag", default="qresnet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_path = build_cache(args.model, colormap=args.colormap)
    E, base_by, novel_by = pools(emb_path, device)
    tasks = [("3w5s", 3, 5), ("3w10s", 3, 10), ("5w5s", 5, 5)]
    print(f"[qresnet] device={device} model={args.model} E={E.shape} heads={args.heads} "
          f"seeds={args.seeds} m={args.m_modes} n_ph={args.n_photons} read={args.read_modes} "
          f"(CNN SOTA bar = ProtoNet-ResNet50 82.61)", flush=True)

    results = {h: {t[0]: [] for t in tasks} for h in args.heads}
    meta = {}
    for head in args.heads:
        for seed in args.seeds:
            net, nparams, out_dim = meta_train(head, seed, device, E, base_by, args)
            meta[head] = {"n_params": nparams, "out_dim": out_dim}
            for name, nw, k in tasks:
                r = evaluate(net, device, E, novel_by, nw, k, args.eval_q, 100, seed)
                results[head][name].append(r["acc"])
                print(f"[qresnet:{head}] seed{seed} {name}: {r['acc']:.2f}±{r['ci95']:.2f} "
                      f"(f1 {r['macro_f1']:.2f})", flush=True)

    print("\n=== Phase 1 QResNet: quantum vs matched-classical vs baseline (ResNet50 GAP-2048) ===")
    print("(CNN SOTA bar = ProtoNet-ResNet50 84… → 82.61 @ 3w5s; baseline row = our frozen-ResNet50 repro)")
    summary = {}
    for head in args.heads:
        summary[head] = {"meta": meta.get(head, {})}
        line = [f"{head:9s} (params={meta.get(head,{}).get('n_params','?')}, "
                f"out={meta.get(head,{}).get('out_dim','?')})"]
        for name, _, _ in tasks:
            v = np.array(results[head][name])
            mean = v.mean()
            sci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            summary[head][name] = {"mean": float(mean), "ci95": float(sci), "per_seed": v.tolist()}
            line.append(f"{name} {mean:.2f}±{sci:.2f}")
        print("  " + " | ".join(line))

    out = C.RESULTS_DIR / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "summary": summary}, indent=2))
    print(f"[qresnet] saved {out}")


if __name__ == "__main__":
    main()
