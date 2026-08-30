"""Open-set novelty-detection utilities for the FSWMPR novelty sector ().

Consolidates the scattered scratch scripts (former temp/{novelty_baselines, tightest_baseline,
baseline_conv4, baselines_novelty}.py) into one reusable, deduplicated module. Everything is scored
with the SAME prototype-distance rule so photonic heads (asi/orthogonal/multi_photon, via qreg_bench) and
classical baselines (Baseline/ProtoNet-Conv4) are compared fairly.

Public API
----------
- auroc(pos, neg)                          rank/Mann-Whitney AUROC with tie handling
- sample_class_images(pool, count, per, rng)   vectorized no-replacement per-episode draws
- novelty_scores(support, labels, query, n_way) -> ({mindist,cosine,knn1,msp}, dist)
- evaluate_novelty(embed, novel_by, novel_classes, seeds, ...) -> per-scorer AUROC + proto accuracy
- load_wafer_224()                          (Xt, y, split, classes, base_by, novel_by)
- conv4_ce_backbone / conv4_protonet_backbone  Chen-2019 Baseline / ProtoNet-Conv4 feature extractors
- make_index_embedder(...)                  wrap a backbone as embed: global-idx -> features
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P
from ..models.backbone import Conv4Backbone

# --- shared protocol constants (open-set episodic, matches qreg_bench eval) -------------------
NW, K, Q, N_EPISODES = 3, 5, 20, 100
SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CosineClassifier(nn.Module):
    """Cosine-distance classifier head (Chen 2019 Baseline++): scale · Ŵ·x̂."""
    def __init__(self, d, n, scale=10.0):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(n, d)); self.scale = scale
    def forward(self, x):
        return self.scale * (F.normalize(x, dim=1) @ F.normalize(self.W, dim=1).T)


# --- metrics ----------------------------------------------------------------------------------
def auroc(pos, neg) -> float:
    """Rank (Mann-Whitney) AUROC with tie handling. pos/neg = 1-D confidence arrays."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    s = np.concatenate([pos, neg]); n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); s_sorted = s[order]
    base = np.arange(1, len(s) + 1, dtype=np.float64); ranks_sorted = base.copy()
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s_sorted[i] != s_sorted[start]:
            ranks_sorted[start:i] = base[start:i].mean(); start = i
    ranks = np.empty(len(s), dtype=np.float64); ranks[order] = ranks_sorted
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sample_class_images(pool_idx, count, per, rng):
    """Draw `per` distinct images (no replacement) from a 1-D pool for each of `count` episodes."""
    r = rng.random((count, len(pool_idx)))
    return pool_idx[np.argsort(r, axis=1)[:, :per]]


# --- novelty confidence scorers (higher = more "known") ---------------------------------------
def novelty_scores(support, labels, query, n_way):
    """Prototype-distance based known-confidence scores for `query` given labelled `support`.
    Returns (dict of scorer->np.array over queries, dist matrix (Nq,n_way))."""
    mus = torch.stack([support[labels == c].mean(0) for c in range(n_way)])       # (n_way,D)
    d = torch.cdist(query, mus)                                                    # (Nq,n_way)
    out = {"mindist": (-d.min(1).values).cpu().numpy()}
    qn, mn = F.normalize(query, dim=1), F.normalize(mus, dim=1)
    out["cosine"] = (qn @ mn.T).max(1).values.cpu().numpy()
    out["knn1"] = (-torch.cdist(query, support).min(1).values).cpu().numpy()
    out["msp"] = torch.softmax(-d, dim=1).max(1).values.cpu().numpy()
    return out, d


# --- unified open-set episodic evaluator ------------------------------------------------------
@torch.no_grad()
def evaluate_novelty(embed, novel_by, novel_classes=None, seeds=SEEDS,
                     n_way=NW, k=K, q=Q, n_ep=N_EPISODES, scorers=("mindist", "knn1", "cosine")):
    """Episodic open-set eval for ANY embedder.

    embed: callable(np.ndarray[global image indices]) -> torch.FloatTensor (len, D) on DEVICE.
    novel_by: {class_name: np.ndarray of global indices}. Per episode: n_way "known" classes give
    support+query; the held-out novel classes give OOD query. Returns per-seed accuracy (nearest
    prototype) and AUROC for each scorer.
    """
    pool = list(novel_classes or C.NOVEL_CLASSES)
    accs = []
    aus = {s: [] for s in scorers}
    for seed in seeds:
        rng = np.random.default_rng(100000 + seed)
        ep_acc, ep_au = [], {s: [] for s in scorers}
        for _ in range(n_ep):
            chosen = rng.choice(len(pool), size=n_way, replace=False)
            si, sl, qi = [], [], []
            for j, ci in enumerate(chosen):
                pick = sample_class_images(novel_by[pool[ci]], 1, k + q, rng)[0]
                si += pick[:k].tolist(); sl += [j] * k; qi += pick[k:].tolist()
            ncl = [c for c in range(len(pool)) if c not in chosen]
            allidx = np.concatenate([novel_by[pool[c]] for c in ncl])
            ood = rng.choice(allidx, size=q, replace=False)
            se = embed(np.array(si)); qe = embed(np.array(qi)); oe = embed(ood)
            lab = torch.arange(n_way, device=se.device).repeat_interleave(k)
            s_in, d_in = novelty_scores(se, lab, qe, n_way)
            s_ood, _ = novelty_scores(se, lab, oe, n_way)
            true = torch.arange(n_way, device=se.device).repeat_interleave(q)
            ep_acc.append(((-d_in).argmax(1) == true).float().mean().item())
            for s in scorers:
                ep_au[s].append(auroc(s_in[s], s_ood[s]))
        accs.append(float(np.mean(ep_acc)) * 100)
        for s in scorers:
            aus[s].append(float(np.nanmean(ep_au[s])) * 100)
    def stat(v):
        v = np.array(v); return {"mean": float(v.mean()),
                                 "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                                 "per_seed": v.tolist()}
    return {"acc": stat(accs), **{f"auroc_{s}": stat(aus[s]) for s in scorers}}


# --- data + classical backbones ---------------------------------------------------------------
def load_wafer_224(device=DEVICE):
    """Returns Xt(N,1,224,224 on CPU), y, split, classes, base_by, novel_by (global-index dicts)."""
    X, y, split, meta = P.load(img_size=224)
    cls = meta["classes"]
    Xt = torch.tensor(X, dtype=torch.float32)[:, None]
    base_by = {c: np.where((split == "base_train") & (y == cls.index(c)))[0] for c in C.BASE_CLASSES}
    novel_by = {c: np.where((split == "novel_pool") & (y == cls.index(c)))[0] for c in C.NOVEL_CLASSES}
    return Xt, y, split, cls, base_by, novel_by


def conv4_ce_backbone(seed, Xt, base_by, cls, y, device=DEVICE, epochs=40, bs=128, cosine_head=False):
    """Conv4 CE-pretrained on the base classes. cosine_head=False → Baseline (linear);
    cosine_head=True → Baseline++ (cosine classifier, Chen 2019). Returns frozen net."""
    torch.manual_seed(seed)
    lab = {c: i for i, c in enumerate(C.BASE_CLASSES)}
    bi = np.concatenate([base_by[c] for c in C.BASE_CLASSES])
    by = torch.tensor([lab[cls[int(y[i])]] for i in bi], device=device)
    net = Conv4Backbone(embed_mode="flatten", img_size=224, in_ch=1).to(device)
    clf = (CosineClassifier(net.embed_dim, len(C.BASE_CLASSES))
           if cosine_head else nn.Linear(net.embed_dim, len(C.BASE_CLASSES))).to(device)
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=1e-3)
    bt = torch.tensor(bi)
    for _ in range(epochs):
        perm = torch.randperm(len(bt))
        for s in range(0, len(bt), bs):
            idx = bt[perm[s:s + bs]]
            loss = F.cross_entropy(clf(net(Xt[idx].to(device))), by[perm[s:s + bs]])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def conv4_protonet_backbone(seed, device=DEVICE, epochs=100, episodes=100):
    """ProtoNet-Conv4 backbone (episodic meta-training), reusing the verified repro_c1 trainer."""
    from . import repro_c1 as R
    xt, base_by, _ = R.build_pools(device, 224)
    tf = R.make_transform(1, "none", device)
    net = R.meta_train(seed, device, xt, base_by, tf, 224, 1, n_way=3, k=5, q=15,
                       epochs=epochs, episodes=episodes)
    return net, xt, tf


def resnet50_protonet_embedder(seed, device=DEVICE, epochs=30, episodes=100, lr=1e-4):
    """ProtoNet-ResNet50 (Chen 2019 CNN-SOTA line): episodic meta-training of a timm ResNet50 on the
    base classes (GAP-2048, 1ch→3ch + ImageNet norm). Returns an `embed` callable keyed to GLOBAL
    image indices. NOTE: full-ResNet50 episodic finetuning @224 is the slow baseline (~20-40 min)."""
    import timm
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xt, y, split, cls, base_by, _ = load_wafer_224(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def prep(x1):                                            # (b,1,H,W)[0,1] -> imagenet-norm 3ch
        return (x1.expand(-1, 3, -1, -1) - mean) / std
    n_way, k, q = 3, 5, 15
    model.train()
    for _ in range(epochs * episodes):
        classes = list(rng.choice(C.BASE_CLASSES, size=n_way, replace=False))
        si, sl, qi, ql = [], [], [], []
        for j, c in enumerate(classes):
            pick = sample_class_images(base_by[c], 1, k + q, rng)[0]
            si += pick[:k].tolist(); sl += [j] * k; qi += pick[k:].tolist(); ql += [j] * q
        se = model(prep(Xt[torch.tensor(si)].to(device)))
        qe = model(prep(Xt[torch.tensor(qi)].to(device)))
        protos = torch.stack([se[torch.tensor(sl, device=device) == c].mean(0) for c in range(n_way)])
        loss = F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(ql, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def embed(idxs):
        outs = []
        with torch.no_grad():
            for s in range(0, len(idxs), 128):
                outs.append(model(prep(Xt[torch.as_tensor(idxs[s:s + 128])].to(device))))
        return torch.cat(outs)
    return embed


def make_index_embedder(net, images, device=DEVICE, tf=None, batch=256):
    """Wrap a frozen backbone as embed: global-idx array -> features (len, D) on device.
    Pre-caches features for all needed indices lazily per call (cheap for episodic use)."""
    def embed(idxs):
        outs = []
        with torch.no_grad():
            for s in range(0, len(idxs), batch):
                xb = images[torch.as_tensor(idxs[s:s + batch])].to(device)
                outs.append(net(tf(xb) if tf else xb))
        return torch.cat(outs)
    return embed


def cached_feature_embedder(feat_name, device=DEVICE):
    """Frozen path: embed = lookup into a cached feature matrix (e.g. resemb_resnet50_224_jet.npy),
    keyed by GLOBAL image index. No backbone, no training — the frozen-ResNet50 + prototype baseline,
    on the exact same features the photonic heads use."""
    E = torch.tensor(np.load(C.PROCESSED_DIR / f"{feat_name}.npy"), dtype=torch.float32, device=device)
    def embed(idxs):
        return E[torch.as_tensor(idxs, device=device)]
    return embed
