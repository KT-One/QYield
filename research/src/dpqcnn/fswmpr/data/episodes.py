"""Seeded episode generator (architecture.md §4.1; : dataset-parameterized).

Pre-generates N-way K-shot episode index lists ONCE per dataset and writes them to
`data/episodes/[{dataset}_]{N}way_{K}shot_seed{s}.json` (WM-811K keeps unprefixed
names for back-compat with 02; other datasets are prefixed). Every model
consumes the identical episodes. Indices are into the dataset bundle's global array
(novel_pool rows). Support/query disjoint; 3-way excludes Near-full.

Run:      python -m dpqcnn.fswmpr.data.episodes                 # wm811k (default)
          python -m dpqcnn.fswmpr.data.episodes --dataset mixedwm38
Self-test: python -m dpqcnn.fswmpr.data.episodes --test
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


def _dm(dataset):
    return DATA_MODULES[dataset]


def _novel_index_by_class(dataset="wm811k"):
    """Return {class_name: np.array(global row indices in novel_pool)}."""
    P = _dm(dataset)
    X, y, split, meta = P.load()
    idx2cls = {i: c for i, c in enumerate(meta["classes"])}
    by = defaultdict(list)
    for gi in np.where(split == "novel_pool")[0]:
        by[idx2cls[int(y[gi])]].append(int(gi))
    return {k: np.array(v) for k, v in by.items()}


def episode_file(n_way: int, k_shot: int, seed: int, dataset="wm811k") -> Path:
    pref = "" if dataset == "wm811k" else f"{dataset}_"
    return C.EPISODES_DIR / f"{pref}{n_way}way_{k_shot}shot_seed{seed}.json"


def build(force: bool = False, dataset="wm811k", seeds=None):
    C.EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    P = _dm(dataset)
    by_cls = _novel_index_by_class(dataset)
    made = []
    for seed in (seeds if seeds is not None else C.SEEDS):
        for task in C.EPISODE_TASKS:
            n_way, k_shot = task["n_way"], task["k_shot"]
            fp = episode_file(n_way, k_shot, seed, dataset)
            if fp.exists() and not force:
                made.append(fp); continue
            rng = np.random.default_rng(seed * 1000 + n_way * 100 + k_shot)
            pool = ({c: v for c, v in by_cls.items() if c not in C.THREE_WAY_EXCLUDE}
                    if n_way < len(C.NOVEL_CLASSES) else dict(by_cls))
            classes = sorted(pool)
            episodes = []
            need = k_shot + C.Q_QUERY
            for _ in range(C.N_EPISODES):
                ep_classes = list(rng.choice(classes, size=n_way, replace=False))
                support, query = {}, {}
                for cls in ep_classes:
                    avail = pool[cls]
                    take = min(need, len(avail))
                    pick = rng.choice(avail, size=take, replace=False)
                    support[cls] = pick[:k_shot].tolist()
                    query[cls] = pick[k_shot:take].tolist()
                episodes.append({"classes": ep_classes, "support": support, "query": query})
            payload = {"dataset": dataset, "n_way": n_way, "k_shot": k_shot, "q_query": C.Q_QUERY,
                       "seed": seed, "n_episodes": C.N_EPISODES,
                       "split_hash": P.split_hash(), "episodes": episodes}
            fp.write_text(json.dumps(payload))
            made.append(fp)
            print(f"[episodes:{dataset}] {fp.name}: {len(episodes)} episodes, classes={classes}")
    return made


def load_episodes(n_way: int, k_shot: int, seed: int, dataset="wm811k") -> dict:
    fp = episode_file(n_way, k_shot, seed, dataset)
    if not fp.exists():
        raise FileNotFoundError(f"run episodes.build(dataset='{dataset}') (missing {fp})")
    return json.loads(fp.read_text())


def checksum(n_way: int, k_shot: int, seed: int, dataset="wm811k") -> str:
    return hashlib.sha1(episode_file(n_way, k_shot, seed, dataset).read_bytes()).hexdigest()[:12]


def _test():
    ok = True
    for dataset in DATA_MODULES:
        P = _dm(dataset)
        build(force=True, dataset=dataset)
        for seed in C.SEEDS:
            for task in C.EPISODE_TASKS:
                d = load_episodes(task["n_way"], task["k_shot"], seed, dataset)
                assert d["split_hash"] == P.split_hash(), "stale episodes vs split"
                for ep in d["episodes"]:
                    assert len(ep["classes"]) == task["n_way"], "wrong way count"
                    if task["n_way"] < len(C.NOVEL_CLASSES):
                        assert all(c not in C.THREE_WAY_EXCLUDE for c in ep["classes"])
                    for cls in ep["classes"]:
                        s, q = set(ep["support"][cls]), set(ep["query"][cls])
                        assert not (s & q), "support/query overlap!"
                        assert len(s) <= task["k_shot"]
        cs1 = {(t["n_way"], t["k_shot"], s): checksum(t["n_way"], t["k_shot"], s, dataset)
               for s in C.SEEDS for t in C.EPISODE_TASKS}
        build(force=True, dataset=dataset)
        cs2 = {(t["n_way"], t["k_shot"], s): checksum(t["n_way"], t["k_shot"], s, dataset)
               for s in C.SEEDS for t in C.EPISODE_TASKS}
        assert cs1 == cs2, f"{dataset} episodes not deterministic!"
        print(f"[episodes:{dataset}] SELF-TEST PASSED (disjoint, way-count, determinism, hash-linked)")
    return ok


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    else:
        ds = "wm811k"
        if "--dataset" in sys.argv:
            ds = sys.argv[sys.argv.index("--dataset") + 1]
        build(force="--force" in sys.argv, dataset=ds)
