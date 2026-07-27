"""Prototype-registration FSCIL runner ( §3.2).

Dataset- and model-agnostic. Given a FROZEN base-trained embedder (classical Conv4
OR DPQCNN-v2) and a metric (euclidean / cosine / quantum-fidelity), it walks the
shared FSCIL sessions:
  * Session 0: register BASE-class prototypes = class means over base_train.
  * Session t: register the new novel class prototype from its K support shots.
  * After each session: classify the ALL-seen test set by nearest prototype.

No retraining at any session → base prototypes never move → zero base-forgetting by
construction (the honest question is the base-vs-novel balance, captured by HM).

Metrics per session t: ACS_t (overall acc on all seen classes), base_acc_t,
novel_acc_t, HM_t = harmonic mean(base,novel). Summary: AIA = mean_t ACS_t,
forgetting = base_acc_0 - base_acc_T, plus macro-F1 at the final session.

This is exactly CEC's decoupled paradigm minus the graph; DPQCNN-v2 plugs in via
its fidelity metric (output_mode='amp').
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as WM811K
from ..data import mixedwm38 as MIXEDWM38
from ..data import fscil as FS

DATA_MODULES = {"wm811k": WM811K, "mixedwm38": MIXEDWM38}


@torch.no_grad()
def _embed(embed_fn, Xt, idx, dev, metric, bs=512):
    """Embed the given indices; L2-normalize for cosine/fidelity (state space)."""
    outs = []
    idx = np.asarray(idx)
    for i in range(0, len(idx), bs):
        x = Xt[idx[i:i + bs]][:, None]
        e = embed_fn(x)
        if metric in ("cosine", "fidelity"):
            e = F.normalize(e, dim=1)
        outs.append(e)
    return torch.cat(outs, 0) if outs else torch.empty(0, device=dev)


def _classify(P_mat, q, metric):
    if metric == "euclidean":
        return torch.cdist(q, P_mat).argmin(1)
    ov = q @ P_mat.t()                       # both normalized (cosine/fidelity)
    return (ov if metric == "cosine" else ov * ov).argmax(1)


def _macro_f1(pred, true, n):
    f1s = []
    for c in range(n):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) * 100


@torch.no_grad()
def run_fscil(embed_fn, dataset, k_shot, seed, metric, img_size=None, dev=None, balance="none"):
    """balance: 'none' (default) | 'center' — subtract the prototype centroid from queries and
    prototypes before classifying (hubness reduction). Well-estimated BASE prototypes act as hubs
    that steal novel queries (base_acc high, novel_acc low); centering removes that shared dominant
    direction to rebalance base-vs-novel selection. Training-free, hyperparameter-free."""
    img_size = img_size or C.IMG_SIZE
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = DATA_MODULES[dataset]
    Xt, y, split, meta = P.load_gpu(img_size, dev)
    y = np.asarray(y); split = np.asarray(split)
    cls2idx = {c: i for i, c in enumerate(meta["classes"])}
    d = FS.load_sessions(dataset, k_shot, seed)
    base_classes = d["base_classes"]

    # base_train indices per base class (for the frozen embedder's base prototypes)
    bt_by = {c: np.where((split == "base_train") & (y == cls2idx[c]))[0] for c in base_classes}
    base_test = d["base_test"]; novel_test = d["novel_test"]

    protos = {}                                  # class_name -> prototype vector
    seen = []
    rows = []
    for s in d["sessions"]:
        for cls in s["new_classes"]:
            idx = bt_by[cls] if cls in base_classes else np.asarray(s["support"][cls])
            emb = _embed(embed_fn, Xt, idx, dev, metric)
            p = emb.mean(0)
            if metric in ("cosine", "fidelity"):
                p = F.normalize(p, dim=0)
            protos[cls] = p
            seen.append(cls)
        # evaluate on all seen classes
        names = list(protos.keys())
        P_mat = torch.stack([protos[n] for n in names])
        test_idx, test_lab, is_base = [], [], []
        for n in seen:
            tset = base_test[n] if n in base_classes else novel_test[n]
            test_idx += tset; test_lab += [names.index(n)] * len(tset)
            is_base += [n in base_classes] * len(tset)
        q = _embed(embed_fn, Xt, test_idx, dev, metric)
        Pm, qq = P_mat, q
        if balance == "center":                  # hubness reduction: remove prototype-centroid direction
            mu = P_mat.mean(0, keepdim=True)
            if metric in ("cosine", "fidelity"):
                Pm = F.normalize(P_mat - mu, dim=1); qq = F.normalize(q - mu, dim=1)
            else:
                Pm = P_mat - mu; qq = q - mu
        pred = _classify(Pm, qq, metric).cpu()
        true = torch.tensor(test_lab)
        is_base = np.asarray(is_base, bool)
        correct = (pred == true).numpy()
        acs = float(correct.mean()) * 100
        base_acc = float(correct[is_base].mean()) * 100 if is_base.any() else 0.0
        nov = ~is_base
        novel_acc = float(correct[nov].mean()) * 100 if nov.any() else 0.0
        hm = (2 * base_acc * novel_acc / (base_acc + novel_acc)) if (base_acc + novel_acc) else 0.0
        rows.append({"session": s["session"], "n_classes": len(names),
                     "acs": round(acs, 2), "base_acc": round(base_acc, 2),
                     "novel_acc": round(novel_acc, 2), "hm": round(hm, 2),
                     "macro_f1": round(_macro_f1(pred, true, len(names)), 2)})
    acs_curve = [r["acs"] for r in rows]
    return {
        "dataset": dataset, "k_shot": k_shot, "seed": seed, "metric": metric,
        "sessions": rows,
        "aia": round(float(np.mean(acs_curve)), 2),
        "final_acs": rows[-1]["acs"],
        "final_macro_f1": rows[-1]["macro_f1"],       # PRIMARY balanced metric (class-unbiased)
        "forgetting": round(rows[0]["base_acc"] - rows[-1]["base_acc"], 2),
        "final_hm": rows[-1]["hm"],
    }
