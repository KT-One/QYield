"""FSWMPR preprocessing (architecture.md §1.1).

Loads WM-811K, keeps the real single-defect wafers (8 classes, drop none +
unlabeled), resizes to 42x42 (bilinear) with /2 normalization (Gong-Lin), and
writes a hash-versioned tensor bundle. Split-then-augment is honored by storing
CLEAN tensors + split labels here and applying augmentation ONLY on the base
train split at backbone-training time (never on val/test/novel episodes).

Run: python -m dpqcnn.experiments.fswmpr_preprocess
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import constants as C


# ---------------------------------------------------------------------------
# Legacy WM-811K pickle load (pre-0.20 pandas module paths + py2 pickle).
# ---------------------------------------------------------------------------
def _unpickle_legacy(path: Path):
    import pandas.core.indexes as _idx
    import pandas.core.indexes.base as _idxbase
    sys.modules.setdefault("pandas.indexes", _idx)
    sys.modules.setdefault("pandas.indexes.base", _idxbase)
    sys.modules.setdefault(
        "pandas.indexes.range",
        __import__("pandas.core.indexes.range", fromlist=["RangeIndex"]),
    )
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _extract_scalar(cell):
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return None
        return cell.flatten()[0]
    return cell


def _canon_label(raw) -> str | None:
    s = _extract_scalar(raw)
    if s is None:
        return None
    s = str(s).strip()
    return C.LABEL_ALIASES.get(s, s if s in C.ALL_DEFECT_CLASSES else None)


# ---------------------------------------------------------------------------
# Preprocessing config hash — refuse to consume a mismatched processed bundle.
# ---------------------------------------------------------------------------
def preprocess_hash(img_size: int = None) -> str:
    img_size = C.IMG_SIZE if img_size is None else img_size
    payload = {
        "img_size": img_size, "norm_div": C.PIXEL_NORM_DIV,
        "resize": C.RESIZE_MODE, "base": C.BASE_CLASSES, "novel": C.NOVEL_CLASSES,
        "base_split": C.BASE_SPLIT, "seed": C.BACKBONE_SEED,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def split_hash() -> str:
    """Resolution-INDEPENDENT identity of the row selection + split assignment.
    Episode files key on this so the same episodes are valid at any img_size
    (novel_pool row order is identical across resolutions)."""
    payload = {"base": C.BASE_CLASSES, "novel": C.NOVEL_CLASSES,
               "base_split": C.BASE_SPLIT, "seed": C.BACKBONE_SEED}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def processed_path(img_size: int = None) -> Path:
    img_size = C.IMG_SIZE if img_size is None else img_size
    return C.PROCESSED_DIR / f"wm811k_{img_size}_{preprocess_hash(img_size)}.npz"


def _resize_norm(wafer_map: np.ndarray, img_size: int) -> np.ndarray:
    """(h,w) int {0,1,2} -> (img_size,img_size) float32 in [0,1] via /2 + bilinear."""
    x = torch.tensor(wafer_map.astype(np.float32) / C.PIXEL_NORM_DIV)[None, None]
    x = F.interpolate(x, size=(img_size, img_size), mode=C.RESIZE_MODE,
                      align_corners=False)
    return x[0, 0].numpy().astype(np.float32)


def build(force: bool = False, img_size: int = None) -> Path:
    img_size = C.IMG_SIZE if img_size is None else img_size
    out = processed_path(img_size)
    if out.exists() and not force:
        print(f"[preprocess] up-to-date: {out}")
        return out
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.BACKBONE_SEED)

    print(f"[preprocess] loading {C.WM811K_RAW} (img_size={img_size}) ...")
    t0 = time.time()
    df = _unpickle_legacy(C.WM811K_RAW)
    df["_label"] = df["failureType"].apply(_canon_label)
    df = df[df["_label"].isin(C.ALL_DEFECT_CLASSES)].reset_index(drop=True)
    print(f"[preprocess] {len(df)} real-defect wafers in {time.time()-t0:.1f}s")

    cls_to_idx = {c: i for i, c in enumerate(C.ALL_DEFECT_CLASSES)}
    X = np.stack([_resize_norm(m, img_size) for m in df["waferMap"].values])
    y = np.array([cls_to_idx[l] for l in df["_label"].values], dtype=np.int64)

    # split label: base classes -> train/val/test; novel classes -> novel_pool
    split = np.empty(len(df), dtype=object)
    base_idx = np.array([i for i, l in enumerate(df["_label"].values)
                         if l in C.BASE_CLASSES])
    novel_idx = np.array([i for i, l in enumerate(df["_label"].values)
                          if l in C.NOVEL_CLASSES])
    split[novel_idx] = "novel_pool"
    # stratified base split by class
    for cls in C.BASE_CLASSES:
        ci = np.array([i for i in base_idx if df["_label"].values[i] == cls])
        rng.shuffle(ci)
        n = len(ci)
        n_tr = int(n * C.BASE_SPLIT["train"])
        n_va = int(n * C.BASE_SPLIT["val"])
        split[ci[:n_tr]] = "base_train"
        split[ci[n_tr:n_tr + n_va]] = "base_val"
        split[ci[n_tr + n_va:]] = "base_test"

    meta = {
        "hash": preprocess_hash(img_size), "split_hash": split_hash(),
        "img_size": img_size, "n": int(len(df)),
        "classes": C.ALL_DEFECT_CLASSES, "base": C.BASE_CLASSES,
        "novel": C.NOVEL_CLASSES,
        "counts": {c: int((y == cls_to_idx[c]).sum()) for c in C.ALL_DEFECT_CLASSES},
        "split_counts": {s: int((split == s).sum())
                         for s in ["base_train", "base_val", "base_test", "novel_pool"]},
    }
    np.savez_compressed(out, X=X, y=y, split=split.astype(str), meta=json.dumps(meta))
    print(f"[preprocess] saved {out}")
    print(f"[preprocess] counts: {meta['counts']}")
    print(f"[preprocess] splits: {meta['split_counts']}")
    return out


from functools import lru_cache


@lru_cache(maxsize=8)
def load(require_hash: bool = True, img_size: int = None):
    """Return (X, y, split, meta). Cached: repeated calls across configs reuse the
    decompressed arrays (the npz decompress is the slow part). Refuses on hash
    mismatch."""
    img_size = C.IMG_SIZE if img_size is None else img_size
    p = processed_path(img_size)
    if not p.exists():
        raise FileNotFoundError(f"run fswmpr_preprocess.build(img_size={img_size}) first (missing {p})")
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    if require_hash and meta["hash"] != preprocess_hash(img_size):
        raise RuntimeError(f"preprocess hash mismatch: {meta['hash']} != {preprocess_hash(img_size)} "
                           "(constants changed; rerun build(force=True))")
    return d["X"], d["y"], d["split"], meta


_GPU_CACHE: dict = {}


def subset_indices(y, split, meta, frac, seed=0, splits=None):
    """ fast lab: stratified per-(split,class) subset of GLOBAL indices.
    Returns {split_name: np.ndarray(global_idx)} keeping ~frac of every class in every split
    (>=1 per non-empty class). Deterministic in `seed`. Same rows across img_size (row order is
    resolution-independent — see split_hash)."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    y = _np.asarray(y); split = _np.asarray(split)
    splits = splits or ["base_train", "base_val", "base_test", "novel_pool"]
    ncls = len(meta["classes"])
    out = {}
    for sp in splits:
        keep = []
        for c in range(ncls):
            idx = _np.where((split == sp) & (y == c))[0]
            if len(idx) == 0:
                continue
            n = max(1, int(round(len(idx) * frac)))
            keep.append(rng.choice(idx, min(n, len(idx)), replace=False))
        out[sp] = _np.concatenate(keep) if keep else _np.array([], dtype=int)
    return out


def load_gpu(img_size, device):
    """GPU-resident cached (X_tensor(B,1,H,W), y, split, meta) — avoids re-transfer
    of the whole set to the GPU on every config/eval call."""
    key = (img_size or C.IMG_SIZE, str(device))
    if key not in _GPU_CACHE:
        import torch
        X, y, split, meta = load(img_size=img_size)
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        _GPU_CACHE[key] = (xt, y, split, meta)
    return _GPU_CACHE[key]


if __name__ == "__main__":
    force = "--force" in sys.argv
    build(force=force)                          # shared/quantum regime (C.IMG_SIZE=42)
    build(force=force, img_size=C.ANCHOR["img_size"])   # faithful anchor regime (84)
