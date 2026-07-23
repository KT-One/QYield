"""smoke_test.py — quick end-to-end smoke test for the QYield CLI package.

Covers both UX paths:
  1. "upload" path — predict_array() on an array standing in for a user-uploaded
     wafer map (drawn from the K-set here only so this test needs no external data;
     in real use this would be a genuinely new file via `qyield predict <path>`).
  2. "demo" path — the exact query-from-kset flow `qyield demo` uses.

Run: uv run python tests/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qyield.constants import DEFAULT_KSET_PATH, NOVEL_CLASSES
from qyield.model import QYieldModel, REPO_ROOT, load_kset


def main() -> int:
    print("QYield smoke test\n" + "=" * 40)
    # Use the DEFAULT device path (auto-detect + CPU fallback) — the same path
    # real users hit via the CLI — so this test catches device/fallback bugs
    # instead of masking them with an explicit device="cpu".
    model = QYieldModel()
    print(f"loaded OK on device={model.device}; classes={model.classes}")

    imgs, labels, classes = load_kset(REPO_ROOT / DEFAULT_KSET_PATH)
    labels = np.asarray(labels)
    rng = np.random.default_rng(0)

    # ---- 1) "upload" path: full default 8-way prediction ----
    i = int(rng.integers(len(imgs)))
    r = model.predict_array(imgs[i])
    ok1 = r["predicted_class"] == str(labels[i])
    print(f"\n[upload-path, full 8-way]  true={labels[i]:9s} -> predicted={r['predicted_class']} "
          f"[{'HIT' if ok1 else 'miss'}]")

    # ---- 2) "demo" path: true 3-way/5-shot episodes, novel classes only ----
    print(f"\n[demo-path, 3-way 5-shot, NOVEL classes only]")
    correct, trials = 0, 0
    novel_idx = np.where(np.isin(labels, NOVEL_CLASSES))[0]
    for i in rng.choice(novel_idx, size=6, replace=False):
        true = str(labels[i])
        for _ in range(20):
            ways = list(rng.choice(NOVEL_CLASSES, size=3, replace=False))
            if true in ways:
                break
        r = model.predict_array(imgs[i], ways=ways, k_shot=5, seed=int(rng.integers(1 << 30)))
        hit = r["predicted_class"] == true
        correct += hit
        trials += 1
        print(f"  true={true:9s} | ways={ways} -> {r['predicted_class']:9s} "
              f"[{'HIT ' if hit else 'miss'}]")

    print(f"\nsmoke hit rate: {correct}/{trials}")
    print("\nAll checks completed without error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
