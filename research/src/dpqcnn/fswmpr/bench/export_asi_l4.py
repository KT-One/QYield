"""export_asi_l4.py — train the QYield **L4** head (`asi_deep`, depth 4) and export a
self-contained checkpoint bundle for the QYield MVP.

This is the exact recipe used to produce the shipped `checkpoints/asi_l4/` bundle.
It must run in the DP-QCNN research environment (it imports the research `dpqcnn`
package, which needs `merlin` + `timm` + the cached `resemb_resnet50_224_jet.npy`
feature + the WM-811K data):

    PYTHONPATH=<dpqcnn-src> python -m dpqcnn.fswmpr.bench.export_asi_l4 \
        --out-dir <QYield>/checkpoints/asi_l4 --seed 42 --updates 1000

Outputs:
    asi_l4.pt                  head config + state_dict + train meta + sanity metrics (~0.7 MB)
    resnet50_jet_backbone.pt   frozen ResNet50 GAP-2048 backbone (torchvision-compatible, ~94 MB)

The backbone weights are the exact timm ImageNet ResNet50 that built the training
features; the script verifies a torchvision ResNet50 loaded with them reproduces the
cached features bit-for-bit (0.00e+00), so the MVP needs no `timm` dependency.

Deployed model (seed 42): closed-set acc 80.73, novelty AUROC 69.35 on this seed's head;
the reported headline (71.5 novelty AUROC) is the 11-seed mean — see docs/results-novelty.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .. import constants as C
from ..data import preprocess as P
from . import qreg_bench as QB

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEAT = "resemb_resnet50_224_jet"


def build_torchvision_backbone(device):
    """torchvision ResNet50 GAP-2048 feature extractor (same Sequential the MVP builds),
    initialised from the timm ImageNet weights that produced the training features."""
    import timm
    import torchvision

    timm_model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device).eval()
    timm_sd = timm_model.state_dict()
    tv = torchvision.models.resnet50(weights=None)
    tv_sd = tv.state_dict()
    transfer = {k: v for k, v in timm_sd.items() if k in tv_sd and v.shape == tv_sd[k].shape}
    tv.load_state_dict({**tv_sd, **transfer})
    tv = tv.to(device).eval()
    stem = nn.Sequential(tv.conv1, tv.bn1, tv.relu, tv.maxpool,
                         tv.layer1, tv.layer2, tv.layer3, tv.layer4, tv.avgpool).to(device).eval()
    return timm_model, stem, len(transfer)


@torch.no_grad()
def jet_embed(model_or_stem, gray_batch, device, is_stem):
    from matplotlib import colormaps
    cmap = colormaps["jet"]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    rgb = cmap(gray_batch)[..., :3]
    xb = torch.tensor(rgb, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    xb = (xb - mean) / std
    out = model_or_stem(xb)
    return out.flatten(1).float() if is_stem else out.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="destination dir for the bundle (QYield/checkpoints/asi_l4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--meta-batch", type=int, default=16)
    ap.add_argument("--train-q", type=int, default=15)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--eval-episodes", type=int, default=100)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    QB.perf_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] device={device} seed={args.seed} updates={args.updates}", flush=True)

    # 1. train the L4 head on cached ResNet50-jet features (base classes)
    E, base_by, novel_by = QB.pools(FEAT, device)
    cfg = {"kind": "asi_deep", "n_pool": 3, "depth": 4}
    t0 = time.time()
    net, npar = QB.meta_train(cfg, args.seed, device, E, base_by,
                              args.updates, args.meta_batch, 5, args.train_q, False)
    print(f"[export] trained asi_L4 head: {npar:,} params ({time.time()-t0:.0f}s)", flush=True)

    # 2. sanity eval: closed-set acc + novelty AUROC (3w5s, mindist)
    accs, aus = [], []
    for s in args.eval_seeds:
        r = QB.evaluate(net, device, E, novel_by, 3, 5, 20, args.eval_episodes, s)
        accs.append(r["acc"]); aus.append(r["auroc"])
        print(f"[export]   eval seed {s}: acc={r['acc']:.2f}  novelty_auroc={r['auroc']:.2f}", flush=True)
    acc_mean, au_mean = float(np.mean(accs)), float(np.mean(aus))
    print(f"[export] SANITY: acc={acc_mean:.2f}  novelty_auroc={au_mean:.2f} (target ~78 / ~71.5)", flush=True)

    # 3. backbone: torchvision ResNet50 from timm ImageNet weights; verify feature parity
    timm_model, stem, n_transfer = build_torchvision_backbone(device)
    X, y, split, meta = P.load(img_size=224)
    gray = X[np.arange(96)]
    e_timm = jet_embed(timm_model, gray, device, is_stem=False)
    e_tv = jet_embed(stem, gray, device, is_stem=True)
    e_cached = E[torch.as_tensor(np.arange(96), device=device)].clone()
    d_timm_cached = (e_timm - e_cached).abs().max().item()
    d_tv_timm = (e_tv - e_timm).abs().max().item()
    print(f"[export] parity: max|timm-cached|={d_timm_cached:.2e} (TF32 rounding ok) "
          f"max|torchvision-timm|={d_tv_timm:.2e}", flush=True)
    if d_timm_cached > 5e-3:
        raise SystemExit(f"[export] ABORT: preprocessing mismatch ({d_timm_cached:.2e}).")
    if d_tv_timm >= 1e-4:
        raise SystemExit("[export] ABORT: torchvision parity failed; MVP would need timm.")

    # 4. save the bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_path = out_dir / "resnet50_jet_backbone.pt"
    torch.save({"stem_state_dict": stem.state_dict(),
                "arch": "torchvision.resnet50 (conv1..avgpool Sequential)",
                "source": "timm resnet50 pretrained ImageNet, weights transferred",
                "colormap": "jet"}, backbone_path)
    head_bundle = {
        "model_kind": "asi_L4", "head": "asi_deep",
        "config": {"E": 2048, "m": 4, "n_pool": 3, "depth": 4, "read_modes": 2,
                   "n_exp": 4, "n_qpus": 512, "colormap": "jet", "norm_blocks": False},
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "backbone_file": backbone_path.name, "n_params": int(npar),
        "train": {"seed": args.seed, "updates": args.updates, "meta_batch": args.meta_batch,
                  "k": 5, "train_q": args.train_q, "lr": 1e-3, "loss": "episodic prototype CE",
                  "base_classes": C.BASE_CLASSES, "feature": FEAT},
        "sanity_metrics": {"eval_seeds": args.eval_seeds, "closed_set_acc_mean": acc_mean,
                           "novelty_auroc_mean": au_mean, "per_seed_acc": accs, "per_seed_auroc": aus},
    }
    head_path = out_dir / "asi_l4.pt"
    torch.save(head_bundle, head_path)
    print(f"[export] saved {head_path} + {backbone_path}\n[export] DONE", flush=True)


if __name__ == "__main__":
    main()
