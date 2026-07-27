"""MixedWM38 preprocessing ( §2 D2).

Loads the MixedWM38 bundle (`data/mixedwm38/Wafer_Map_Datasets.npz`), keeps the
SINGLE-defect maps (one-hot rows with exactly one active class), and emits a
hash-versioned tensor bundle with the SAME schema, class-index scheme and
base/novel split as the WM-811K pipeline (`data/preprocess.py`) so every
downstream module (episodes, FSCIL sessions, models) works unchanged across the
two datasets.

MixedWM38 label legend (from the bundled Description.pdf; C7/C9 relabel-corrected):
  dim 0..7 = [Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random]
These are the SAME 8 defect types as WM-811K, so we map by NAME onto
`constants.ALL_DEFECT_CLASSES` (identical indices) — enabling a clean cross-dataset
comparison and optional cross-dataset transfer.

Run:      python -m dpqcnn.fswmpr.data.mixedwm38
Self-test: python -m dpqcnn.fswmpr.data.mixedwm38 --test
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import constants as C

DATASET = "mixedwm38"
RAW = C.DATA_DIR / "mixedwm38" / "Wafer_Map_Datasets.npz"
# one-hot column order in arr_1 (Description.pdf)
MW38_LEGEND = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
               "Near-full", "Scratch", "Random"]


# ---------------------------------------------------------------------------
# Config hashes (dataset-tagged so its processed bundle + episodes are distinct).
# ---------------------------------------------------------------------------
def preprocess_hash(img_size: int = None) -> str:
    img_size = C.IMG_SIZE if img_size is None else img_size
    payload = {
        "dataset": DATASET, "img_size": img_size, "norm_div": C.PIXEL_NORM_DIV,
        "resize": C.RESIZE_MODE, "base": C.BASE_CLASSES, "novel": C.NOVEL_CLASSES,
        "base_split": C.BASE_SPLIT, "seed": C.BACKBONE_SEED,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def split_hash() -> str:
    """Resolution-independent identity of row selection + split (episode key)."""
    payload = {"dataset": DATASET, "base": C.BASE_CLASSES, "novel": C.NOVEL_CLASSES,
               "base_split": C.BASE_SPLIT, "seed": C.BACKBONE_SEED}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def processed_path(img_size: int = None) -> Path:
    img_size = C.IMG_SIZE if img_size is None else img_size
    return C.PROCESSED_DIR / f"{DATASET}_{img_size}_{preprocess_hash(img_size)}.npz"


def _resize_norm(wafer_map: np.ndarray, img_size: int) -> np.ndarray:
    """(52,52) int -> (img_size,img_size) float32 in [0,1] via clip[0,2] + /2 + bilinear.
    MixedWM38 doc says values {0,1,2}; the raw array has 214 stray '3' pixels
    (artifacts) — clipped to 2 ('fail') so the representation matches WM-811K exactly."""
    x = torch.tensor(np.clip(wafer_map, 0, 2).astype(np.float32) / C.PIXEL_NORM_DIV)[None, None]
    x = F.interpolate(x, size=(img_size, img_size), mode=C.RESIZE_MODE, align_corners=False)
    return x[0, 0].numpy().astype(np.float32)


def build(force: bool = False, img_size: int = None) -> Path:
    img_size = C.IMG_SIZE if img_size is None else img_size
    out = processed_path(img_size)
    if out.exists() and not force:
        print(f"[mixedwm38] up-to-date: {out}")
        return out
    if not RAW.exists():
        raise FileNotFoundError(f"missing {RAW} (download co1d7era/mixedtype-wafer-defect-datasets)")
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.BACKBONE_SEED)

    print(f"[mixedwm38] loading {RAW} (img_size={img_size}) ...")
    t0 = time.time()
    d = np.load(RAW)
    maps, labels = d["arr_0"], d["arr_1"]                 # (N,52,52) int, (N,8) one-hot
    single = labels.sum(1) == 1                            # keep single-defect maps only
    maps, oh = maps[single], labels[single]
    dim = oh.argmax(1)
    names = np.array([MW38_LEGEND[i] for i in dim])
    keep = np.array([n in C.ALL_DEFECT_CLASSES for n in names])
    maps, names = maps[keep], names[keep]
    print(f"[mixedwm38] {len(maps)} single-defect maps in {time.time()-t0:.1f}s")

    cls_to_idx = {c: i for i, c in enumerate(C.ALL_DEFECT_CLASSES)}   # SAME as WM-811K
    X = np.stack([_resize_norm(m, img_size) for m in maps])
    y = np.array([cls_to_idx[n] for n in names], dtype=np.int64)

    split = np.empty(len(maps), dtype=object)
    base_idx = np.array([i for i, n in enumerate(names) if n in C.BASE_CLASSES])
    novel_idx = np.array([i for i, n in enumerate(names) if n in C.NOVEL_CLASSES])
    split[novel_idx] = "novel_pool"
    for cls in C.BASE_CLASSES:                             # stratified base split
        ci = np.array([i for i in base_idx if names[i] == cls])
        rng.shuffle(ci)
        n = len(ci); n_tr = int(n * C.BASE_SPLIT["train"]); n_va = int(n * C.BASE_SPLIT["val"])
        split[ci[:n_tr]] = "base_train"
        split[ci[n_tr:n_tr + n_va]] = "base_val"
        split[ci[n_tr + n_va:]] = "base_test"

    meta = {
        "dataset": DATASET, "hash": preprocess_hash(img_size), "split_hash": split_hash(),
        "img_size": img_size, "n": int(len(maps)),
        "classes": C.ALL_DEFECT_CLASSES, "base": C.BASE_CLASSES, "novel": C.NOVEL_CLASSES,
        "counts": {c: int((y == cls_to_idx[c]).sum()) for c in C.ALL_DEFECT_CLASSES},
        "split_counts": {s: int((split == s).sum())
                         for s in ["base_train", "base_val", "base_test", "novel_pool"]},
    }
    np.savez_compressed(out, X=X, y=y, split=split.astype(str), meta=json.dumps(meta))
    print(f"[mixedwm38] saved {out}")
    print(f"[mixedwm38] counts: {meta['counts']}")
    print(f"[mixedwm38] splits: {meta['split_counts']}")
    return out


@lru_cache(maxsize=8)
def load(require_hash: bool = True, img_size: int = None):
    img_size = C.IMG_SIZE if img_size is None else img_size
    p = processed_path(img_size)
    if not p.exists():
        raise FileNotFoundError(f"run mixedwm38.build(img_size={img_size}) first (missing {p})")
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    if require_hash and meta["hash"] != preprocess_hash(img_size):
        raise RuntimeError(f"mixedwm38 hash mismatch: {meta['hash']} != {preprocess_hash(img_size)}")
    return d["X"], d["y"], d["split"], meta


_GPU_CACHE: dict = {}


def load_gpu(img_size, device):
    key = (img_size or C.IMG_SIZE, str(device))
    if key not in _GPU_CACHE:
        X, y, split, meta = load(img_size=img_size)
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        _GPU_CACHE[key] = (xt, y, split, meta)
    return _GPU_CACHE[key]


def _test():
    build(force=True)
    X, y, split, meta = load()
    # expected single-defect counts from Description.pdf
    exp = {"Center": 1000, "Donut": 1000, "Edge-Loc": 1000, "Edge-Ring": 1000,
           "Loc": 1000, "Near-full": 149, "Scratch": 1000, "Random": 866}
    assert meta["counts"] == exp, f"count mismatch: {meta['counts']} != {exp}"
    assert X.shape[1:] == (C.IMG_SIZE, C.IMG_SIZE), X.shape
    assert X.min() >= 0 and X.max() <= 1.0 + 1e-6, (X.min(), X.max())
    # split disjoint + covers base classes only in base_*; novel in novel_pool
    idx2cls = {i: c for i, c in enumerate(C.ALL_DEFECT_CLASSES)}
    for s in ["base_train", "base_val", "base_test"]:
        cls = {idx2cls[int(v)] for v in y[split == s]}
        assert cls <= set(C.BASE_CLASSES), f"{s} has non-base {cls}"
    novcls = {idx2cls[int(v)] for v in y[split == "novel_pool"]}
    assert novcls <= set(C.NOVEL_CLASSES), f"novel_pool has non-novel {novcls}"
    print(f"[mixedwm38] SELF-TEST PASSED (counts match legend, split disjoint, norm in [0,1])")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    else:
        build(force="--force" in sys.argv)
