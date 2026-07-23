"""cli.py — QYield command-line interface.

Two ways to get a prediction:
  1. `qyield predict <wafer.npy>`   — classify YOUR OWN wafer map.
  2. `qyield demo`                 — no data yet? classify a sample wafer drawn
                                       from our bundled K-set instead.

Both print the predicted class + a full ranking, and support the same few-shot
episode flags (--n-way / --k-shot / --ways / --seed) to reproduce the reported
3-way/5-shot accuracy regime instead of the default fixed 8-way classification.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .constants import ALL_DEFECT_CLASSES, NOVEL_CLASSES
from .model import QYieldModel, load_kset


def _print_result(result: dict) -> None:
    print(f"\nPredicted class: {result['predicted_class']}")
    print(f"Classes in this episode: {', '.join(result['episode_classes'])}")
    print("\nRanking (closest prototype first):")
    for cls, dist in result["ranking"]:
        marker = " <-- predicted" if cls == result["predicted_class"] else ""
        print(f"  {cls:12s} distance={dist:.3f}{marker}")


def _add_episode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-way", type=int, default=None,
                   help="classify against N randomly-sampled classes instead of all 8 "
                        "(e.g. --n-way 3 for a 3-way episode). Ignored if --ways is given.")
    p.add_argument("--k-shot", type=int, default=None,
                   help="use only this many support shots/class (randomly subsampled, "
                        "max 10). Omit to use every bundled shot.")
    p.add_argument("--ways", nargs="+", default=None, metavar="CLASS",
                   help=f"explicit class list instead of --n-way, e.g. --ways Donut Scratch Loc. "
                        f"Valid: {', '.join(ALL_DEFECT_CLASSES)}")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducible --n-way/--k-shot subsampling.")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto-detect).")


def cmd_predict(args: argparse.Namespace) -> int:
    model = QYieldModel(device=args.device)
    result = model.predict(args.image, n_way=args.n_way, k_shot=args.k_shot,
                           ways=args.ways, seed=args.seed)
    _print_result(result)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .constants import DEFAULT_KSET_PATH
    from .model import REPO_ROOT

    model = QYieldModel(device=args.device)
    imgs, labels, classes = load_kset(REPO_ROOT / DEFAULT_KSET_PATH)
    labels = np.asarray(labels)
    rng = np.random.default_rng(args.seed)

    if args.true_class:
        idx = np.where(labels == args.true_class)[0]
        if len(idx) == 0:
            print(f"error: '{args.true_class}' not found in the bundled K-set. "
                  f"Valid classes: {', '.join(classes)}", file=sys.stderr)
            return 1
        i = int(rng.choice(idx))
    else:
        i = int(rng.integers(len(imgs)))

    print(f"Demo query drawn from the bundled K-set — true class: {labels[i]}")
    print("(this is a demo/smoke-test query from our own support pool, not a novel "
          "wafer — use `qyield predict <your_file.npy>` for real data)")
    result = model.predict_array(imgs[i], n_way=args.n_way, k_shot=args.k_shot,
                                 ways=args.ways, seed=args.seed)
    _print_result(result)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print("QYield — quantum wafer-defect classifier (QResNet-ensemble)")
    print(f"\nAll classes ({len(ALL_DEFECT_CLASSES)}): {', '.join(ALL_DEFECT_CLASSES)}")
    print(f"Novel classes (few-shot generalization target): {', '.join(NOVEL_CLASSES)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qyield", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="classify your own wafer map (.npy/.png/.jpg)")
    p_predict.add_argument("image", help="path to a wafer map file")
    _add_episode_args(p_predict)
    p_predict.set_defaults(func=cmd_predict)

    p_demo = sub.add_parser("demo", help="classify a sample wafer from our bundled K-set")
    p_demo.add_argument("--true-class", default=None, metavar="CLASS",
                        help=f"pick a demo query from this class specifically. "
                             f"Valid: {', '.join(ALL_DEFECT_CLASSES)}")
    _add_episode_args(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_info = sub.add_parser("info", help="show model/class info")
    p_info.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
