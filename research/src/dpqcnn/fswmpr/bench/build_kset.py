""" — curate a bundled, pre-labeled K-shot support set for CLI/inference-bundle use.

Both QConv4 and QResNet-ensemble are episodic ProtoNet models: they classify a query
by nearest-prototype distance to a K-shot SUPPORT SET, computed on the fly. There is
no fixed softmax head. For a simple "just give me an image path" CLI, we ship a
curated K-shot set (labeled by us, from the existing WM-811K labels) covering ALL 8
defect classes (3 base + 5 novel) so `infer.py` can compute prototypes at inference
time without the user supplying their own support images.

This is a genuine departure from the paper's few-shot episodic protocol (which
samples a NEW random K-shot subset per episode, only 3-5 classes at a time, only
from novel classes at eval time) — here the K-set is FIXED, curated once, covers all
8 classes for a usable general classifier. Documented in README.md.

Stores RAW wafer maps (pre-resize, {0,1,2} int arrays) + labels, resolution-independent
(each consumer resizes to whatever img_size it needs), so ONE k-set file serves both
QConv4 (224px) and QResNet-ensemble (224px + colormap). Also writes a human-readable
JSON manifest alongside the .npz (class descriptions, die-state legend, per-class shot
breakdown) — raw {0,1,2} pixel arrays are not self-explanatory to someone unfamiliar
with WM-811K, so the manifest exists specifically to make the K-set intuitive for
end users/devs browsing it, not just consumable by code.

Run: uv run python -m dpqcnn.fswmpr.bench.build_kset --k 10 --seed 42
"""
from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np

from .. import constants as C
from ..data import preprocess as P


def build_kset(k: int, seed: int, out_path=None):
    """Sample k examples per class (3 base + 5 novel) from their respective labeled
    pools (base classes from base_train; novel classes from novel_pool — the only
    split where they exist). Saves the .npz AND a companion human-readable .json
    manifest (same stem, `_manifest.json` suffix)."""
    X, y, split, meta = P.load(img_size=C.IMG_SIZE)   # native 42px cache; we only need
    cls = meta["classes"]                              # raw indices, not resized pixels
    rng = np.random.default_rng(seed)

    # Re-load the RAW (unresized) wafer maps directly from the pickle-derived arrays is
    # not available post-preprocess; instead we store indices into the wm811k_224 bundle
    # (the highest-res cache we maintain) so every consumer can slice at its own img_size.
    X224, y224, split224, meta224 = P.load(img_size=224)
    assert list(y224) == list(y) and list(split224) == list(split), \
        "42px and 224px caches must share row order (same preprocess split_hash)"

    chosen = {}
    for c in C.ALL_DEFECT_CLASSES:
        ci = cls.index(c)
        pool_split = "base_train" if c in C.BASE_CLASSES else "novel_pool"
        idx = np.where((split == pool_split) & (y == ci))[0]
        if len(idx) < k:
            raise ValueError(f"class {c} pool '{pool_split}' has only {len(idx)} < k={k} examples")
        chosen[c] = rng.choice(idx, size=k, replace=False)

    all_idx = np.concatenate([chosen[c] for c in C.ALL_DEFECT_CLASSES])
    all_labels = np.concatenate([[c] * k for c in C.ALL_DEFECT_CLASSES])
    imgs_224 = X224[all_idx]                            # (8k, 224, 224) float32 [0,1], /2-normed

    out_path = Path(out_path) if out_path else (C.CKPT_DIR / f"kset_k{k}_s{seed}.npz")
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images_224=imgs_224.astype(np.float32),
                        labels=all_labels, classes=C.ALL_DEFECT_CLASSES,
                        k=k, seed=seed, global_idx=all_idx)
    print(f"[kset] saved {out_path} — {k} shots x {len(C.ALL_DEFECT_CLASSES)} classes "
          f"({imgs_224.shape[0]} images, 224px, /2-normalized)")

    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    _write_manifest(manifest_path, imgs_224, all_labels, k, seed)
    print(f"[kset] saved human-readable manifest {manifest_path}")
    return out_path


def _write_manifest(manifest_path, imgs_224, labels, k, seed):
    """Human-readable companion to the .npz — describes each class + what the raw
    0/1/2 pixel values mean, so a user browsing the K-set isn't staring at bare
    integer arrays."""
    per_class = {}
    for c in C.ALL_DEFECT_CLASSES:
        mask = labels == c
        idxs = np.where(mask)[0].tolist()
        example = imgs_224[mask][0]
        vals, counts = np.unique(np.round(example * C.PIXEL_NORM_DIV).astype(int), return_counts=True)
        per_class[c] = {
            "description": C.CLASS_DESCRIPTIONS.get(c, ""),
            "group": "base (used in original training)" if c in C.BASE_CLASSES
                     else "novel (held out at training time)",
            "n_shots_in_bundle": int(mask.sum()),
            "image_indices_in_npz": idxs,
            "die_state_value_counts_example": {str(v): int(ct) for v, ct in zip(vals, counts)},
        }
    manifest = {
        "description": "Human-readable manifest for this K-shot support set .npz — used by "
                       "load_model.py / infer.py to compute few-shot classification prototypes. "
                       "Describes WHAT each class is and what the raw pixel values in the "
                       "images_224 array mean, since raw wafer-map arrays of 0/1/2 are not "
                       "self-explanatory without WM-811K domain context.",
        "k_shots_per_class": k, "build_seed": seed, "total_images": int(imgs_224.shape[0]),
        "image_shape": list(imgs_224.shape[1:]),
        "image_format": f"float32 in [0,1], each pixel = (raw die-state value 0/1/2) / "
                        f"{C.PIXEL_NORM_DIV} (see die_state_legend). Resolution 224x224.",
        "die_state_legend": {str(k_): v for k_, v in C.DIE_STATE_LEGEND.items()},
        "classes": per_class,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_kset(path):
    """Returns (images_224: (N,224,224) float32, labels: (N,) str array, classes: list[str])."""
    d = np.load(path, allow_pickle=True)
    return d["images_224"], d["labels"], list(d["classes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="shots per class (all 8 classes)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build_kset(args.k, args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
