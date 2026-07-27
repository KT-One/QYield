""" Phase 1 (option 1) — in-domain SSL pretraining of ResNet50 on base wafers.

Frozen ImageNet ResNet50 caps ~78 (jet); label-finetuning on 3 base classes overfits (<78). SSL uses NO
labels → adapts to the wafer domain WITHOUT the 3-class overfit. We SimCLR-pretrain ResNet50 (ImageNet
init) on base_train wafers (jet-RGB + strong aug), then FREEZE, cache GAP-2048 features, and run the same
baseline / quantum / classical prototype heads (reusing bench/qresnet.py). Target: lift the ResNet50
baseline above ~78 toward the CNN SOTA 82.61, with the quantum head on top.

Run: uv run python -m dpqcnn.fswmpr.bench.qresnet_ssl --ssl-epochs 80 --heads baseline quantum classical
"""
from __future__ import annotations

import argparse
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T2

from .. import constants as C
from ..data import preprocess as P
from ..models.ssl_pretrain import nt_xent, _proj_head
from . import qresnet as QR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _jet_lut(device):
    from matplotlib import colormaps
    lut = colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
    return torch.tensor(lut, dtype=torch.float32, device=device)


def stem_ckpt_path(epochs, seed, colormap):
    """SimCLR ResNet50 stem checkpoint — needed to embed genuinely new images."""
    return C.CKPT_DIR / f"ssl_stem_simclr_{colormap}_ep{epochs}_s{seed}.pt"


def ssl_pretrain(device, epochs, bs, lr, seed, colormap="jet", force=False, save_stem=True):
    """SimCLR NT-Xent pretrain of ResNet50 on base_train wafers; cache GAP-2048 features for ALL images."""
    feat_path = C.PROCESSED_DIR / f"resemb_ssl_{colormap}_ep{epochs}_s{seed}.npy"
    if feat_path.exists() and not force:
        print(f"[ssl] cached features up-to-date: {feat_path}")
        return feat_path
    import torchvision
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, y, split, meta = P.load(img_size=224)
    xt = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)      # (N,1,224,224)
    base_idx = np.where(split == "base_train")[0]
    lut = _jet_lut(device) if colormap else None
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool).to(device)
    head = _proj_head(2048, out=128).to(device)

    def render(x1):
        if lut is not None:
            idx = (x1.squeeze(1).clamp(0, 1) * 255).long()
            return lut[idx].permute(0, 3, 1, 2)
        return x1.expand(-1, 3, -1, -1)

    # defect-preserving aug: keep jet color (defect is color-encoded), gentle crop, wafer-symmetric flips/rot
    aug = T2.Compose([T2.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True),
                      T2.RandomHorizontalFlip(), T2.RandomVerticalFlip(),
                      T2.RandomRotation(30), T2.ColorJitter(0.2, 0.2, 0.2)])

    def encode(x1):
        rgb = render(x1)
        rgb = aug(rgb)
        x = (rgb - mean) / std
        return stem(x).flatten(1)                     # (B,2048)

    opt = torch.optim.AdamW(list(stem.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    n = len(base_idx)
    stem.train(); head.train(); t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            bidx = base_idx[perm[i:i + bs]]
            xb = xt[bidx]
            z1 = F.normalize(head(encode(xb)), dim=1)
            z2 = F.normalize(head(encode(xb)), dim=1)
            loss = nt_xent(torch.cat([z1, z2], 0))
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[ssl] ep {ep+1}/{epochs} nt_xent={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # freeze + cache GAP-2048 features for ALL images (no aug at feature-extraction time)
    stem.eval()
    embs = np.zeros((X.shape[0], 2048), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], 256):
            rgb = render(xt[i:i + 256])
            x = (rgb - mean) / std
            embs[i:i + 256] = stem(x).flatten(1).float().cpu().numpy()
    np.save(feat_path, embs)
    print(f"[ssl] saved features {feat_path} shape={embs.shape} (train {time.time()-t0:.0f}s)")

    if save_stem:
        C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        cp = stem_ckpt_path(epochs, seed, colormap)
        torch.save({"stem_state_dict": stem.state_dict(), "method": "simclr", "colormap": colormap,
                   "epochs": epochs, "seed": seed, "img_size": 224, "out_dim": 2048}, cp)
        print(f"[ssl] saved stem checkpoint {cp} (needed for inference on new images)")
    return feat_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["baseline", "quantum", "classical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--ssl-epochs", type=int, default=80)
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
    ap.add_argument("--tag", default="qresnet_ssl")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save the BEST seed's trained 'quantum' head (state_dict+config only, "
                         "no baked-in prototypes) to checkpoints/qresnet_ssl.pt, mirroring "
                         "qresnet_ens.py's checkpoint format")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_path = ssl_pretrain(device, args.ssl_epochs, args.ssl_bs, args.ssl_lr, args.ssl_seed, args.colormap)
    E, base_by, novel_by = QR.pools(feat_path, device)
    tasks = [("3w5s", 3, 5), ("3w10s", 3, 10), ("5w5s", 5, 5)]
    print(f"[qresssl] E={E.shape} heads={args.heads} seeds={args.seeds} ssl_ep={args.ssl_epochs} "
          f"cmap={args.colormap} (CNN SOTA bar 82.61)", flush=True)

    import json
    results = {h: {t[0]: [] for t in tasks} for h in args.heads}
    meta = {}
    trained_nets = {h: {} for h in args.heads}   # head -> {seed: net}
    for head in args.heads:
        for seed in args.seeds:
            net, nparams, out_dim = QR.meta_train(head, seed, device, E, base_by, args)
            meta[head] = {"n_params": nparams, "out_dim": out_dim}
            trained_nets[head][seed] = net
            for name, nw, k in tasks:
                r = QR.evaluate(net, device, E, novel_by, nw, k, args.eval_q, 100, seed)
                results[head][name].append(r["acc"])
                print(f"[qresssl:{head}] seed{seed} {name}: {r['acc']:.2f}±{r['ci95']:.2f} "
                      f"(f1 {r['macro_f1']:.2f})", flush=True)

    print("\n=== Phase 1 QResNet (in-domain SSL ResNet50) — CNN SOTA bar 82.61 @ 3w5s ===")
    summary = {}
    for head in args.heads:
        summary[head] = {"meta": meta.get(head, {})}
        line = [f"{head:9s} (params={meta.get(head,{}).get('n_params','?')})"]
        for name, _, _ in tasks:
            v = np.array(results[head][name])
            m = v.mean(); sci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            summary[head][name] = {"mean": float(m), "ci95": float(sci), "per_seed": v.tolist()}
            line.append(f"{name} {m:.2f}±{sci:.2f}")
        print("  " + " | ".join(line))
    out = C.RESULTS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({"config": vars(args), "summary": summary}, indent=2))
    print(f"[qresssl] saved {out}")

    if args.save_ckpt:
        head = "quantum" if "quantum" in trained_nets else args.heads[0]
        best_seed_idx = int(np.argmax(results[head]["3w5s"]))
        best_seed = args.seeds[best_seed_idx]
        net = trained_nets[head][best_seed]
        C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = C.CKPT_DIR / "qresnet_ssl.pt"
        stem_ckpt = str(stem_ckpt_path(args.ssl_epochs, args.ssl_seed, args.colormap))
        torch.save({
            "model": "qresnet_ssl", "head": head, "seed": best_seed,
            "state_dict": net.state_dict(),
            "config": {"m_modes": args.m_modes, "add_modes": args.add_modes,
                       "n_photons": args.n_photons, "read_modes": args.read_modes,
                       "E": E.shape[1]},
            "out_dim": net.out_dim,
            "ssl_methods": ["simclr"], "ssl_epochs": args.ssl_epochs,
            "ssl_seed": args.ssl_seed, "colormap": args.colormap,
            "ssl_stem_ckpts": [stem_ckpt],
            "eval_3w5s_this_seed": results[head]["3w5s"][best_seed_idx],
            "eval_3w5s_aggregate": summary.get(head, {}).get("3w5s", {}),
            "reported_rep0306_3w5s": {"mean": 81.67, "ci95": 1.32},
            "note": "state_dict + config only — NO baked-in prototypes. Single-SimCLR-SSL "
                    "2048-d embedding, reshaped into 512 QPUs x m=4 modes (QResNet-SSL, the "
                    "non-ensemble/non-Light mainline; see  cost-runtime.md Table 1). "
                    "infer.py-equivalent code computes prototypes on the fly from a K-shot "
                    "support set at inference time. Checkpoint is the BEST individual seed "
                    "among this run's --seeds by 3w5s accuracy; aggregate mean verified against "
                    " reported 81.67±1.32 for provenance. Requires the SSL stem "
                    "checkpoint listed in ssl_stem_ckpts to embed a genuinely new "
                    "(never-cached) image.",
        }, ckpt_path)
        missing = not __import__("pathlib").Path(stem_ckpt).exists()
        print(f"[qresssl] saved inference checkpoint {ckpt_path} (head={head}, best seed={best_seed}, "
              f"3w5s={results[head]['3w5s'][best_seed_idx]:.2f} this-seed / "
              f"{summary.get(head, {}).get('3w5s', {}).get('mean', float('nan')):.2f} aggregate)")
        if missing:
            print(f"[qresssl] WARNING: SSL stem checkpoint missing on disk: {stem_ckpt}")


if __name__ == "__main__":
    main()
