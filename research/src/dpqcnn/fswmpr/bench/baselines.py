"""Fair classical baselines for the FSWMPR novelty sector (), built on bench.novelty.

Every method is a frozen feature extractor scored with the SAME prototype-distance novelty rule as
the photonic heads in qreg_bench (fixes the earlier softmax-scoring unfairness). Extend later
(e.g. ProtoNet-ResNet50) by adding a backbone branch — the evaluator never changes.

Run (single method+seed per process; simple foreground uv run):
  uv run python -m dpqcnn.fswmpr.bench.baselines --method baseline --seed 42
  uv run python -m dpqcnn.fswmpr.bench.baselines --method protonet --seed 42
"""
from __future__ import annotations

import argparse
import json

from .. import constants as C
from . import novelty as NV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["baseline", "baselinepp", "protonet", "protonet_r50", "protonet_r50_frozen"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce-epochs", type=int, default=40, help="Conv4 CE epochs (baseline/baselinepp)")
    ap.add_argument("--feat", default="resemb_resnet50_224_jet", help="cached feats (protonet_r50_frozen)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    dev = NV.DEVICE

    Xt, y, split, cls, base_by, novel_by = NV.load_wafer_224(dev)

    if args.method in ("baseline", "baselinepp"):
        net = NV.conv4_ce_backbone(args.seed, Xt, base_by, cls, y, dev, epochs=args.ce_epochs,
                                   cosine_head=(args.method == "baselinepp"))
        embed = NV.make_index_embedder(net, Xt, dev, tf=None)
        seeds = [args.seed]
    elif args.method == "protonet":
        net, xt, tf = NV.conv4_protonet_backbone(args.seed, dev)
        embed = NV.make_index_embedder(net, xt, dev, tf=tf)
        seeds = [args.seed]
    elif args.method == "protonet_r50_frozen":  # frozen ResNet50 + prototype (no training) — asi's feature pipeline
        embed = NV.cached_feature_embedder(args.feat, dev)
        seeds = NV.SEEDS                          # deterministic/no-train → all seeds in one fast call
    else:  # protonet_r50 (finetuned ProtoNet-ResNet50, slow, external SOTA ref)
        embed = NV.resnet50_protonet_embedder(args.seed, dev)
        seeds = [args.seed]

    res = NV.evaluate_novelty(embed, novel_by, seeds=seeds)
    line = (f"[{args.method} seed{args.seed}] acc={res['acc']['mean']:.2f} | "
            f"AUROC mindist={res['auroc_mindist']['mean']:.2f} "
            f"knn1={res['auroc_knn1']['mean']:.2f} cosine={res['auroc_cosine']['mean']:.2f}")
    print(line, flush=True)
    tag = args.tag or f"fair_{args.method}_{args.seed}"
    out = C.RESULTS_DIR / f"{tag}.json"
    out.write_text(json.dumps({f"{args.method}_{args.seed}": res}, indent=2))
    print(f"[baselines] saved {out}", flush=True)


if __name__ == "__main__":
    main()
