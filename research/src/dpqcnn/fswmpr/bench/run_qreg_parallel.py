"""Parallel orchestrator for the qreg battery. The per-head workload is tiny (a ~5k-param head on
~1k vectors/step) → a single process leaves the GPU almost idle (launch-latency bound). Running
several head-groups as CONCURRENT processes lets their kernels interleave on the SMs, raising GPU
utilization and cutting wall-clock. Each worker runs qreg_bench on a disjoint head subset; results
are merged into one JSON.

Run: uv run python -m dpqcnn.fswmpr.bench.run_qreg_parallel --tag qreg_full --max-parallel 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .. import constants as C
from .qreg_bench import head_configs


def run_group(feat, heads, seeds, updates, meta_batch, eval_episodes, tasks, tag):
    cmd = [sys.executable, "-m", "dpqcnn.fswmpr.bench.qreg_bench",
           "--feats", feat, "--heads", *heads, "--seeds", *map(str, seeds),
           "--updates", str(updates), "--meta-batch", str(meta_batch),
           "--eval-episodes", str(eval_episodes), "--tasks", *tasks, "--tag", tag]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return feat, tag, p.returncode, time.time() - t0, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["resemb_ssl_jet_ep60_s42", "resemb_resnet50_224_jet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 7, 99])
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--meta-batch", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--tasks", nargs="+", default=["3w5s", "3w10s"])
    ap.add_argument("--n-groups", type=int, default=6, help="head groups per feat (concurrency width)")
    ap.add_argument("--max-parallel", type=int, default=6, help="max concurrent GPU processes")
    ap.add_argument("--heads", nargs="+", default=None, help="subset of head names (default: all)")
    ap.add_argument("--tag", default="qreg_full")
    args = ap.parse_args()

    names = args.heads or list(head_configs().keys())
    # spread the 3 slow quantum heads across different groups for balanced load
    slow = [n for n in names if n.startswith("quantum") or n.startswith("multi_photon")]
    fast = [n for n in names if not (n.startswith("quantum") or n.startswith("multi_photon"))]
    groups = [[] for _ in range(args.n_groups)]
    for i, n in enumerate(slow + fast):
        groups[i % args.n_groups].append(n)
    groups = [g for g in groups if g]

    jobs = []
    for feat in args.feats:
        for gi, g in enumerate(groups):
            jobs.append((feat, g, f"{args.tag}__{feat}__g{gi}"))
    print(f"[orch] {len(jobs)} jobs, {args.max_parallel} concurrent, groups={[len(g) for g in groups]}", flush=True)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = [ex.submit(run_group, feat, g, args.seeds, args.updates, args.meta_batch,
                          args.eval_episodes, args.tasks, tag) for feat, g, tag in jobs]
        for f in futs:
            feat, tag, rc, dt, out, err = f.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[orch] {status} {tag} ({dt:.0f}s)", flush=True)
            if rc != 0:
                print(f"  stderr: {err[-800:]}", flush=True)
            else:
                for ln in out.splitlines():
                    if "] 3w5s" in ln or "] 3w" in ln:
                        print("   " + ln.strip(), flush=True)
            results.append((feat, tag, rc))

    # merge part JSONs
    merged = {}
    for feat, tag, rc in results:
        if rc != 0:
            continue
        part = C.RESULTS_DIR / f"{tag}.json"
        if not part.exists():
            continue
        d = json.loads(part.read_text())
        for fn, summ in d["results"].items():
            merged.setdefault(fn, {}).update(summ)
    outp = C.RESULTS_DIR / f"{args.tag}.json"
    outp.write_text(json.dumps({"merged_from": [t for _, t, _ in results], "results": merged}, indent=2))
    print(f"\n[orch] merged -> {outp} in {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
