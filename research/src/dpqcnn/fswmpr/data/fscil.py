"""Shared FSCIL session generator ( §3.2).

Builds the few-shot class-incremental session structure used by EVERY FSCIL model
(CEC-port, prototype-registration, DPQCNN-v2), so all see the identical sessions:
  * Session 0 (base): the BASE classes (trained on base_train at train time).
  * Sessions 1..T (incremental): each introduces ONE novel class with only K shots
    (support), sampled once and frozen; the rest of that class is held-out test.
After session t, evaluation covers ALL classes seen so far (base + novel up to t) —
the driver computes ACS_t, harmonic-mean(base,novel), forgetting, and AIA.

Dataset-agnostic: works on both WM-811K (`data.preprocess`) and MixedWM38
(`data.mixedwm38`) since both expose the same load()/split_hash() interface and
the SAME 8-class base/novel split.

Run:      python -m dpqcnn.fswmpr.data.fscil
Self-test: python -m dpqcnn.fswmpr.data.fscil --test
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import constants as C
from . import preprocess as WM811K
from . import mixedwm38 as MIXEDWM38

DATA_MODULES = {"wm811k": WM811K, "mixedwm38": MIXEDWM38}
K_SHOTS = (5, 10)


def _by_class(P, split_name):
    X, y, split, meta = P.load()
    idx2cls = {i: c for i, c in enumerate(meta["classes"])}
    by = defaultdict(list)
    for gi in np.where(split == split_name)[0]:
        by[idx2cls[int(y[gi])]].append(int(gi))
    return {k: np.array(v) for k, v in by.items()}


def session_file(dataset: str, k_shot: int, seed: int) -> Path:
    return C.EPISODES_DIR / f"fscil_{dataset}_{k_shot}shot_seed{seed}.json"


def build(dataset: str = "wm811k", k_shots=K_SHOTS, force: bool = False, seeds=None):
    P = DATA_MODULES[dataset]
    C.EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    base_test = _by_class(P, "base_test")
    novel = _by_class(P, "novel_pool")
    made = []
    for seed in (seeds if seeds is not None else C.SEEDS):
        for k in k_shots:
            fp = session_file(dataset, k, seed)
            if fp.exists() and not force:
                made.append(fp); continue
            rng = np.random.default_rng(seed * 1000 + k)
            order = list(C.NOVEL_CLASSES)
            rng.shuffle(order)                                  # seeded novel arrival order
            sessions = [{"session": 0, "new_classes": list(C.BASE_CLASSES), "support": {}}]
            novel_test = {}
            for si, cls in enumerate(order, start=1):
                pool = novel[cls].copy(); rng.shuffle(pool)
                sup = pool[:k].tolist()
                novel_test[cls] = pool[k:].tolist()             # held-out, disjoint from support
                sessions.append({"session": si, "new_classes": [cls], "support": {cls: sup}})
            payload = {
                "dataset": dataset, "k_shot": k, "seed": seed,
                "split_hash": P.split_hash(),
                "base_classes": list(C.BASE_CLASSES), "novel_order": order,
                "n_sessions": len(sessions),
                "sessions": sessions,
                "base_test": {c: base_test[c].tolist() for c in C.BASE_CLASSES},
                "novel_test": novel_test,
            }
            fp.write_text(json.dumps(payload))
            made.append(fp)
            print(f"[fscil] {fp.name}: {len(sessions)} sessions, order={order}")
    return made


def load_sessions(dataset: str, k_shot: int, seed: int) -> dict:
    fp = session_file(dataset, k_shot, seed)
    if not fp.exists():
        raise FileNotFoundError(f"run fscil.build('{dataset}') (missing {fp})")
    return json.loads(fp.read_text())


def checksum(dataset: str, k_shot: int, seed: int) -> str:
    return hashlib.sha1(session_file(dataset, k_shot, seed).read_bytes()).hexdigest()[:12]


def _test():
    for dataset in DATA_MODULES:
        P = DATA_MODULES[dataset]
        build(dataset, force=True)
        for seed in C.SEEDS:
            for k in K_SHOTS:
                d = load_sessions(dataset, k, seed)
                assert d["split_hash"] == P.split_hash(), "stale sessions vs split"
                # session 0 = base classes; sessions 1..T = one novel each, all novel covered
                assert d["sessions"][0]["new_classes"] == list(C.BASE_CLASSES)
                seen_novel = []
                for s in d["sessions"][1:]:
                    assert len(s["new_classes"]) == 1
                    cls = s["new_classes"][0]
                    seen_novel.append(cls)
                    sup = s["support"][cls]
                    assert len(sup) == k, f"{dataset} support size {len(sup)}!=k{k}"
                    # support disjoint from that class's held-out test
                    assert not (set(sup) & set(d["novel_test"][cls])), "support/test overlap"
                assert sorted(seen_novel) == sorted(C.NOVEL_CLASSES), "novel classes not all covered"
                # base_test only base classes
                assert set(d["base_test"]) == set(C.BASE_CLASSES)
        # determinism
        cs1 = {(k, s): checksum(dataset, k, s) for s in C.SEEDS for k in K_SHOTS}
        build(dataset, force=True)
        cs2 = {(k, s): checksum(dataset, k, s) for s in C.SEEDS for k in K_SHOTS}
        assert cs1 == cs2, f"{dataset} FSCIL sessions not deterministic!"
        print(f"[fscil] {dataset}: SELF-TEST PASSED (base S0, 1 novel/session, disjoint, determinism, hash)")
    return True


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    else:
        for ds in DATA_MODULES:
            build(ds, force="--force" in sys.argv)
