""" Stage A — faithful C1 ProtoNet-Conv4 reproduction (target 78.40% 3w5s).

Self-contained (does NOT touch the shared 42px episode JSONs / global constants) so
it can't disturb the quantum-regime pipeline. Implements C1's exact protocol:

  * 224x224 input (existing wm811k_224 bundle; 1-ch /2, optional 3-ch + C1 RGB norm).
  * Conv4 = 4x[conv3x3-BN-ReLU-maxpool2] + FLATTEN (Snell/Chen few-shot Conv4).
  * EPISODIC meta-training on the 3 BASE classes, 3-way 5-shot, Adam lr 1e-3,
    100 epochs x 100 episodes = 10,000 episode updates.
  * Euclidean prototype classification (softmax over -||.||^2).
  * Eval: 100 novel episodes, Q=20/class (-> 6000 pts @ 3w5s, matches C1), mean +/- 95% CI.
    3-way episodes draw from ALL 5 novel classes (incl Near-full), as in C1 Table 6.

Subset caveat: our WM-811K subset = 25,519 wafers (base ~75%); C1 = 17,805 (base ~82%).
Split CLASSES are identical; exact filtering differs (documented in  notes).

Run: uv run python -m dpqcnn.fswmpr.bench.repro_c1 --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import constants as C
from ..data import preprocess as P
from ..models.backbone import Conv4Backbone
from ..models.protonet import proto_predict, _macro_recall_f1

# C1 §4.2 reported RGB normalization (computed on their colormapped wafer images).
C1_MEAN = [0.2736, 0.4878, 0.4449]
C1_STD = [0.2056 ** 0.5, 0.2659 ** 0.5, 0.1510 ** 0.5]


def make_transform(channels: int, norm: str, device):
    mean = torch.tensor(C1_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(C1_STD, device=device).view(1, 3, 1, 1)

    def tf(x):                       # x: (B,1,H,W) in [0,1]
        if channels == 3:
            x = x.expand(-1, 3, -1, -1)
            if norm == "c1":
                x = (x - mean) / std
        return x
    return tf


def build_pools(device, img_size):
    """Whole 224 set on GPU (~5 GB); return xt(N,1,H,W) + {class: global idx} for
    base_train and novel_pool."""
    X, y, split, meta = P.load(img_size=img_size)
    cls = meta["classes"]
    xt = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
    base_by = {c: np.where((split == "base_train") & (y == cls.index(c)))[0]
               for c in C.BASE_CLASSES}
    novel_by = {c: np.where((split == "novel_pool") & (y == cls.index(c)))[0]
                for c in C.NOVEL_CLASSES}
    return xt, base_by, novel_by


def sample_episode(by, classes, k, q, rng):
    si, sl, qi, ql = [], [], [], []
    for ci, c in enumerate(classes):
        pick = rng.choice(by[c], size=k + q, replace=False)
        si += pick[:k].tolist(); sl += [ci] * k
        qi += pick[k:].tolist(); ql += [ci] * q
    return np.array(si), np.array(sl), np.array(qi), np.array(ql)


def meta_train(seed, device, xt, base_by, tf, img_size, channels,
               n_way=3, k=5, q=15, epochs=100, episodes=100, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Conv4Backbone(embed_mode="flatten", img_size=img_size, in_ch=channels).to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    base_classes = C.BASE_CLASSES
    nw = min(n_way, len(base_classes))
    total = epochs * episodes
    t0 = time.time()
    for it in range(total):
        classes = list(rng.choice(base_classes, size=nw, replace=False))
        si, sl, qi, ql = sample_episode(base_by, classes, k, q, rng)
        se = net(tf(xt[si])); qe = net(tf(xt[qi]))
        sl_t = torch.tensor(sl, device=device)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(nw)])
        loss = F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(ql, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 2000 == 0:
            print(f"[repro_c1] seed{seed} meta-train {it+1}/{total} "
                  f"loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def evaluate(net, device, xt, novel_by, tf, n_way, k, q, n_episodes, eval_seed, exclude=None):
    rng = np.random.default_rng(100000 + eval_seed)
    pool = [c for c in C.NOVEL_CLASSES if not exclude or c not in exclude]
    accs, f1s = [], []
    for _ in range(n_episodes):
        classes = list(rng.choice(pool, size=n_way, replace=False))
        si, sl, qi, ql = sample_episode(novel_by, classes, k, q, rng)
        se = net(tf(xt[si])); qe = net(tf(xt[qi]))
        pred = proto_predict(se, torch.tensor(sl, device=device), qe, n_way)
        true = torch.tensor(ql, device=device)
        accs.append((pred == true).float().mean().item())
        _, f = _macro_recall_f1(pred, true, n_way)
        f1s.append(f)
    a = np.array(accs)
    ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return {"acc": float(a.mean()) * 100, "ci95": float(ci) * 100,
            "macro_f1": float(np.mean(f1s)) * 100, "n_episodes": len(accs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--channels", type=int, choices=[1, 3], default=1)
    ap.add_argument("--norm", choices=["none", "c1"], default="none")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train-q", type=int, default=15)
    ap.add_argument("--eval-q", type=int, default=20)
    ap.add_argument("--tag", default="repro_c1")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = make_transform(a.channels, a.norm, device)
    xt, base_by, novel_by = build_pools(device, a.img)
    print(f"[repro_c1] device={device} img={a.img} ch={a.channels} norm={a.norm} "
          f"seeds={a.seeds} | base_train={sum(len(v) for v in base_by.values())} "
          f"novel={sum(len(v) for v in novel_by.values())}", flush=True)

    tasks = [("3w5s", 3, 5), ("3w10s", 3, 10), ("5w5s", 5, 5)]
    per_seed = {t[0]: [] for t in tasks}
    all_rows = []
    for seed in a.seeds:
        t0 = time.time()
        net = meta_train(seed, device, xt, base_by, tf, a.img, a.channels,
                         n_way=3, k=5, q=a.train_q, epochs=a.epochs, episodes=a.episodes)
        row = {"seed": seed}
        for name, nw, k in tasks:
            # 3-way includes ALL 5 novel (incl Near-full), as in C1 Table 6.
            r = evaluate(net, device, xt, novel_by, tf, nw, k, a.eval_q, 100, seed)
            per_seed[name].append(r["acc"])
            row[name] = r
            print(f"[repro_c1] seed{seed} {name}: {r['acc']:.2f}±{r['ci95']:.2f} "
                  f"(f1 {r['macro_f1']:.2f})", flush=True)
        row["train_s"] = time.time() - t0
        all_rows.append(row)
        print(f"[repro_c1] seed{seed} done in {row['train_s']:.0f}s", flush=True)

    print("\n=== C1 ProtoNet-Conv4 reproduction (target 3w5s=78.40, 3w10s=80.40, 5w5s=76.23) ===")
    summary = {}
    for name, _, _ in tasks:
        v = np.array(per_seed[name])
        mean = v.mean()
        sci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        summary[name] = {"seed_mean": float(mean), "seed_ci95": float(sci),
                         "per_seed": v.tolist()}
        print(f"  {name}: MEAN {mean:.2f} ± {sci:.2f} (seeds {v.round(2).tolist()})")

    out = C.RESULTS_DIR / f"{a.tag}_ch{a.channels}_{a.norm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "summary": summary, "rows": all_rows}, indent=2))
    print(f"[repro_c1] saved {out}")


if __name__ == "__main__":
    main()
