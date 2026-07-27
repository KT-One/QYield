"""Head B — Prototypical Network (architecture.md §3.2; C3 Snell 2017, C1 baseline).

Two modes:
  * "episodic": a fresh Conv4 trained end-to-end on BASE episodes (the faithful
    C1 ProtoNet-Conv4 reproduction; target 78.40% 3w5s / 76.23% 5w5s).
  * "frozen":  the shared frozen backbone used directly as the embedder (the
    controlled head-attribution variant, D3).

Prototype = mean support embedding; classify query by Euclidean distance
(softmax over -||.||^2). Evaluated on the shared episode JSONs so Head A/B/C/A'
see identical episodes.

Run: python -m dpqcnn.experiments.fswmpr_protonet --mode episodic --seed 42
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
from ..data import episodes as EP
from .backbone import Conv4Backbone, load_backbone


# ---------------------------------------------------------------------------
# Prototype metric + metrics.
# ---------------------------------------------------------------------------
def proto_predict(support_emb, sup_labels, query_emb, n_way):
    """support_emb (S,d), sup_labels (S,) in 0..n_way-1, query_emb (Q,d) ->
    predicted labels (Q,) by nearest Euclidean prototype."""
    protos = torch.stack([support_emb[sup_labels == c].mean(0) for c in range(n_way)])
    d = torch.cdist(query_emb, protos)          # (Q, n_way)
    return d.argmin(1)


def _macro_recall_f1(pred, true, n_way):
    recs, f1s = [], []
    for c in range(n_way):
        tp = ((pred == c) & (true == c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recs.append(rec); f1s.append(f1)
    return float(np.mean(recs)), float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Episodic sampling on BASE for meta-training.
# ---------------------------------------------------------------------------
def _base_pool(device, img_size, split="base_train", dm=P):
    X, y, split_arr, meta = dm.load(img_size=img_size)
    m = split_arr == split
    Xb = torch.tensor(X[m], dtype=torch.float32, device=device)[:, None]
    yb = np.array([meta["classes"][int(v)] for v in y[m]])
    idx_by = {c: np.where(yb == c)[0] for c in C.BASE_CLASSES}
    return Xb, idx_by


def _sample_episode(idx_by, classes, k, q, rng):
    sup_i, qry_i, sup_l, qry_l = [], [], [], []
    for ci, cls in enumerate(classes):
        pick = rng.choice(idx_by[cls], size=k + q, replace=False)
        sup_i += pick[:k].tolist(); sup_l += [ci] * k
        qry_i += pick[k:].tolist(); qry_l += [ci] * q
    return (np.array(sup_i), np.array(sup_l), np.array(qry_i), np.array(qry_l))


def _episode_acc(net, Xb, idx_by, classes, k, q, rng, device):
    si, sl, qi, ql = _sample_episode(idx_by, classes, k, q, rng)
    se, qe = net(Xb[si]), net(Xb[qi])
    pred = proto_predict(se, torch.tensor(sl, device=device), qe, len(classes))
    return (pred == torch.tensor(ql, device=device)).float().mean().item()


def episodic_train(seed, device, img_size=None, embed_mode="gap",
                   augment=False, early_stop=0, n_episodes=None):
    """Train a fresh Conv4 end-to-end on BASE episodes (C1 ProtoNet-Conv4).
    `augment`/`early_stop` curb the 3-base-class overfit for the faithful anchor."""
    from .backbone import _augment
    img_size = img_size or C.IMG_SIZE
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = Conv4Backbone(embed_mode=embed_mode, img_size=img_size).to(device); net.train()
    opt = torch.optim.Adam(net.parameters(), lr=C.PROTONET["episodic_lr"])
    Xb, idx_by = _base_pool(device, img_size, "base_train")
    Xv, idxv = _base_pool(device, img_size, "base_val")
    nw = min(C.PROTONET["train_n_way"], len(C.BASE_CLASSES))
    k, q = C.PROTONET["train_k_shot"], C.PROTONET["train_q"]
    n_ep = n_episodes or C.PROTONET["episodic_train_episodes"]
    best_va, best_state, since = -1.0, None, 0
    t0 = time.time()
    for it in range(n_ep):
        classes = list(rng.choice(C.BASE_CLASSES, size=nw, replace=False))
        si, sl, qi, ql = _sample_episode(idx_by, classes, k, q, rng)
        xs, xq = Xb[si], Xb[qi]
        if augment:
            xs, xq = _augment(xs, gen), _augment(xq, gen)
        se, qe = net(xs), net(xq)
        protos = torch.stack([se[torch.tensor(sl, device=device) == c].mean(0) for c in range(nw)])
        loss = F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(ql, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if early_stop and (it + 1) % 250 == 0:
            net.eval()
            va = np.mean([_episode_acc(net, Xv, idxv, list(rng.choice(C.BASE_CLASSES, nw, replace=False)),
                                       k, q, rng, device) for _ in range(20)])
            net.train()
            if va > best_va:
                best_va, since = va, 0
                best_state = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
            else:
                since += 1
                if since >= early_stop:
                    print(f"[protonet] early stop @ {it+1} (base_val ep-acc {best_va:.3f})")
                    break
        if (it + 1) % 5000 == 0:
            print(f"[protonet] meta-train {it+1}/{n_ep} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def anchor_pretrain(seed, device, img_size, embed_mode="flatten", init_net=None, dm=P):
    """CE-pretrain a Conv4(flatten) on BASE, freeze -> prototype eval (Chen-2019
    'Baseline' style; robust, avoids the 3-class episodic overfit).
    `init_net`: optional pre-initialized Conv4Backbone (e.g. SSL-pretrained) — E4."""
    from .backbone import _augment
    torch.manual_seed(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = init_net if init_net is not None else Conv4Backbone(embed_mode=embed_mode, img_size=img_size).to(device)
    clf = nn.Linear(net.embed_dim, len(C.BASE_CLASSES)).to(device)
    Xb, idx_by = _base_pool(device, img_size, "base_train", dm)
    labels = torch.cat([torch.full((len(idx_by[c]),), i, device=device)
                        for i, c in enumerate(C.BASE_CLASSES)])
    order = torch.cat([torch.tensor(idx_by[c], device=device) for c in C.BASE_CLASSES])
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()),
                           lr=C.BACKBONE["lr"], weight_decay=C.BACKBONE["weight_decay"])
    bs, n = C.BACKBONE["batch_size"], order.shape[0]
    for ep in range(C.ANCHOR["pretrain_epochs"]):
        net.train(); clf.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = order[perm[i:i + bs]]; yb = labels[perm[i:i + bs]]
            opt.zero_grad()
            loss = F.cross_entropy(clf(net(_augment(Xb[idx], gen))), yb)
            loss.backward(); opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Evaluate on shared novel episode files.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(embedder, n_way, k_shot, seed, device, img_size=None):
    img_size = img_size or C.IMG_SIZE
    X, y, split, meta = P.load(img_size=img_size)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    d = EP.load_episodes(n_way, k_shot, seed)
    accs, recs, f1s, tinfer = [], [], [], []
    for ep in d["episodes"]:
        classes = ep["classes"]
        si, sl, qi, ql = [], [], [], []
        for ci, cls in enumerate(classes):
            si += ep["support"][cls]; sl += [ci] * len(ep["support"][cls])
            qi += ep["query"][cls];   ql += [ci] * len(ep["query"][cls])
        if not qi:
            continue
        t0 = time.time()
        se = embedder(Xt[si][:, None]); qe = embedder(Xt[qi][:, None])
        pred = proto_predict(se, torch.tensor(sl, device=device), qe, n_way)
        tinfer.append((time.time() - t0) / len(qi))
        true = torch.tensor(ql, device=device)
        accs.append((pred == true).float().mean().item())
        r, f = _macro_recall_f1(pred, true, n_way)
        recs.append(r); f1s.append(f)
    def ci95(a):
        a = np.array(a); return 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return {
        "n_way": n_way, "k_shot": k_shot, "seed": seed, "n_episodes": len(accs),
        "acc_mean": float(np.mean(accs)) * 100, "acc_ci95": float(ci95(accs)) * 100,
        "novel_recall_macro": float(np.mean(recs)) * 100,
        "macro_f1": float(np.mean(f1s)) * 100,
        "infer_s_per_query": float(np.mean(tinfer)),
    }


def run(mode="anchor_frozen", seed=42):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "episodic":                      # 42px GAP episodic (weak, overfits)
        emb, img = episodic_train(seed, dev), C.IMG_SIZE
    elif mode == "frozen":                      # shared frozen backbone (regime b)
        emb, img = load_backbone(seed, dev), C.IMG_SIZE
    elif mode == "anchor_frozen":               # faithful: 84px flatten CE-pretrain
        img = C.ANCHOR["img_size"]
        emb = anchor_pretrain(seed, dev, img, C.ANCHOR["embed_mode"])
    elif mode == "anchor_episodic":             # faithful: 84px flatten episodic+aug+ES
        img = C.ANCHOR["img_size"]
        emb = episodic_train(seed, dev, img, C.ANCHOR["embed_mode"],
                             augment=C.ANCHOR["episodic_augment"],
                             early_stop=C.ANCHOR["episodic_early_stop_patience"],
                             n_episodes=C.ANCHOR["episodic_train_episodes"])
    else:
        raise ValueError(mode)
    print(f"[protonet:{mode}] seed={seed} img={img} embed={getattr(emb,'embed_mode','?')}/{emb.embed_dim}")
    for task in C.EPISODE_TASKS:
        r = evaluate(emb, task["n_way"], task["k_shot"], seed, dev, img_size=img)
        print(f"  {r['n_way']}w{r['k_shot']}s: acc={r['acc_mean']:.2f}±{r['acc_ci95']:.2f} "
              f"novel_recall={r['novel_recall_macro']:.2f} f1={r['macro_f1']:.2f} "
              f"({r['infer_s_per_query']*1e3:.3f} ms/query)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["episodic", "frozen", "anchor_frozen", "anchor_episodic"],
                    default="anchor_frozen")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.mode, a.seed)
