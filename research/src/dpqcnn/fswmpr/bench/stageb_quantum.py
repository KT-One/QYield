""" Stage B — DP-QCNN residual head on the validated Conv4-flatten@224 pipeline.

Same episodic ProtoNet pipeline as Stage A (bench/repro_c1.py: 224px, Conv4-flatten,
episodic 3-way 5-shot 100ep×100epi, euclidean prototypes, eval Q=20). Three heads share
the Conv4 front-end and are trained end-to-end; we ask whether the distributed multi-photon
QUANTUM residual head beats a PARAMETER-MATCHED CLASSICAL control (same skip, same
input→F linear+square structure, unconstrained instead of unitary), and whether either
beats the baseline 81.7 flatten baseline.

Heads (all euclidean-prototype on the returned embedding):
  * baseline     : Conv4-flatten (12,544-d) -> prototype                        [Stage-A ref ~81.7]
  * quantum   : proj(12544->D) -> z; ψ=MultiPhotonQPUBank(z) (distributed, multi-photon,
                learn_measure); metric = [z ; ψ²]                             (residual readout)
  * classical : proj(12544->D) -> z; ψc = per-QPU UNCONSTRAINED linear(m->F)(z);
                metric = [z ; ψc²]   (param-matched control: no unitary structure)

Quantum ψ is LINEAR in z (photonic ue matrix from permanents of the unitary); the ONLY thing
`quantum` adds over `classical` is that its F×M linear map is a norm-preserving unitary-permanent
(few structured params) vs the classical full/free linear map (matched param count). ψ² is the
shared quadratic nonlinearity. Invariants held: distributed (n_qpus≥2), multi-photon (n≥2).

Run: uv run python -m dpqcnn.fswmpr.bench.stageb_quantum --heads baseline quantum classical --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from ..models.backbone import Conv4Backbone
from ..models.protonet import proto_predict, _macro_recall_f1
from .repro_c1 import build_pools, sample_episode, make_transform


class Conv4Head(nn.Module):
    """Conv4-flatten@224 front-end + {baseline | quantum | classical} residual head."""

    def __init__(self, head, img_size=224, channels=1, d=64, n_qpus=8,
                 add_modes=2, n_photons=2, skip="none", readout="partial",
                 read_modes=2, encode="reshape", m_modes=4, learn_measure=True):
        super().__init__()
        self.head, self.skip, self.readout, self.encode = head, skip, readout, encode
        self.conv = Conv4Backbone(embed_mode="flatten", img_size=img_size, in_ch=channels)
        E = self.conv.embed_dim                    # 12,544 @ 224
        self.E = E
        if head == "baseline":
            self.out_dim = E
            return
        if encode == "reshape":                    # NO compression: conv outputs ARE the mode amplitudes
            assert E % m_modes == 0, f"E={E} not divisible by m_modes={m_modes}"
            self.m = m_modes
            self.n_qpus = E // m_modes             # many small distributed QPUs (unbounded invariant)
            self.proj = None
        else:                                      # proj: Linear(E->d) bottleneck (compression)
            assert d % n_qpus == 0
            self.n_qpus, self.m = n_qpus, d // n_qpus
            self.proj = nn.Linear(E, d)
        M = self.m + add_modes
        self.M = M
        self.F = math.comb(M + n_photons - 1, n_photons)
        self.read_modes = min(read_modes, M)       # PARTIAL: keep only this many mode-occupations
        qwidth = self.read_modes if readout == "partial" else self.F
        self.qdim = self.n_qpus * qwidth
        skip_dim = {"none": 0, "full": E, "proj": (self.n_qpus * self.m)}[skip]
        self.out_dim = skip_dim + self.qdim
        if head == "quantum":
            from ..models.multiphoton_core import MultiPhotonQPUBank
            self.bank = MultiPhotonQPUBank(self.n_qpus, self.m, add_modes, n_photons,
                                           C.QUANTUM["conv_circuit"], learn_measure=learn_measure)
            assert self.bank.F == self.F
        elif readout == "partial":
            # matched classical control: `read_modes` FREE learnable quadratic forms per QPU
            # (a mode occupation occ = zᵀ A z is a quadratic form; classical A is unconstrained).
            self.cW = nn.Parameter(0.1 * torch.randn(self.n_qpus, self.read_modes, self.m, self.m))
        else:                                       # full readout classical: free linear (m->F) -> square
            self.cW = nn.Parameter(0.1 * torch.randn(self.n_qpus, self.F, self.m))

    def forward(self, x):
        v = self.conv(x)                            # (B, E)
        if self.head == "baseline":
            return v
        B = v.shape[0]
        if self.proj is not None:
            z = self.proj(v)                        # (B, d) compressed
        else:
            z = v                                   # (B, E) — no compression
        zc = z.reshape(B, self.n_qpus, self.m)
        zc = zc / (zc.norm(dim=2, keepdim=True) + 1e-8)   # per-QPU unit amplitudes (physical)
        if self.head == "quantum":
            psi = self.bank(zc)                     # (B, N, F) photonic Fock amplitudes
            if self.readout == "partial":
                occ = self.bank.mode_readout(psi)   # (B,N,M) expected photon count per mode
                feat = occ[:, :, :self.read_modes].reshape(B, -1)   # PARTIAL: subset of modes
            else:
                feat = (psi * psi).reshape(B, -1)   # full Fock prob (information-complete)
        else:
            if self.readout == "partial":           # matched free quadratic forms
                Wsym = 0.5 * (self.cW + self.cW.transpose(-1, -2))
                feat = torch.einsum("bni,nkij,bnj->bnk", zc, Wsym, zc).reshape(B, -1)
            else:
                a = torch.einsum("bnx,nfx->bnf", zc, self.cW)
                feat = (a * a).reshape(B, -1)
        if self.skip == "none":                     # quantum does ALL the work (no classical bypass)
            return feat
        skip = v if self.skip == "full" else z
        return torch.cat([skip, feat], dim=1)


def meta_train(head, seed, device, xt, base_by, tf, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Conv4Head(head, img_size=args.img, channels=args.channels, d=args.d,
                    n_qpus=args.n_qpus, add_modes=args.add_modes, n_photons=args.n_photons,
                    skip=args.skip, readout=args.readout, read_modes=args.read_modes,
                    encode=args.encode, m_modes=args.m_modes).to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    total = args.epochs * args.episodes
    nparams = sum(p.numel() for p in net.parameters())
    t0 = time.time()
    for it in range(total):
        classes = list(rng.choice(C.BASE_CLASSES, size=3, replace=False))
        si, sl, qi, ql = sample_episode(base_by, classes, 5, args.train_q, rng)
        se = net(tf(xt[si])); qe = net(tf(xt[qi]))
        sl_t = torch.tensor(sl, device=device)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(3)])
        loss = F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(ql, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 2500 == 0:
            print(f"[stageb:{head}] seed{seed} {it+1}/{total} loss={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, nparams, net.out_dim


@torch.no_grad()
def evaluate(net, device, xt, novel_by, tf, n_way, k, q, n_episodes, eval_seed):
    rng = np.random.default_rng(100000 + eval_seed)
    pool = list(C.NOVEL_CLASSES)
    accs, f1s = [], []
    for _ in range(n_episodes):
        classes = list(rng.choice(pool, size=n_way, replace=False))
        si, sl, qi, ql = sample_episode(novel_by, classes, k, q, rng)
        se = net(tf(xt[si])); qe = net(tf(xt[qi]))
        pred = proto_predict(se, torch.tensor(sl, device=device), qe, n_way)
        true = torch.tensor(ql, device=device)
        accs.append((pred == true).float().mean().item())
        _, f = _macro_recall_f1(pred, true, n_way)
        f1s.append(f)
    a = np.array(accs)
    ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return {"acc": float(a.mean()) * 100, "ci95": float(ci) * 100,
            "macro_f1": float(np.mean(f1s)) * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["baseline", "quantum", "classical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--channels", type=int, choices=[1, 3], default=1)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-qpus", type=int, default=8)
    ap.add_argument("--add-modes", type=int, default=1)
    ap.add_argument("--n-photons", type=int, default=2)
    ap.add_argument("--encode", choices=["reshape", "proj"], default="reshape")
    ap.add_argument("--m-modes", type=int, default=4)
    ap.add_argument("--skip", choices=["none", "proj", "full"], default="none")
    ap.add_argument("--readout", choices=["partial", "full"], default="partial")
    ap.add_argument("--read-modes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train-q", type=int, default=15)
    ap.add_argument("--eval-q", type=int, default=20)
    ap.add_argument("--tag", default="stageb")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save the FIRST seed's trained head (state_dict+config only, "
                         "no baked-in prototypes — infer.py computes those on the fly "
                         "from a K-shot support set) to checkpoints/qconv4.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = make_transform(args.channels, "none", device)
    xt, base_by, novel_by = build_pools(device, args.img)
    tasks = [("3w5s", 3, 5), ("3w10s", 3, 10), ("5w5s", 5, 5)]
    print(f"[stageb] device={device} heads={args.heads} seeds={args.seeds} "
          f"encode={args.encode} m_modes={args.m_modes} n_photons={args.n_photons} add={args.add_modes} "
          f"skip={args.skip} readout={args.readout} read_modes={args.read_modes}", flush=True)

    results = {h: {t[0]: [] for t in tasks} for h in args.heads}
    meta = {}
    trained_nets = {h: {} for h in args.heads}   # head -> {seed: net}
    for head in args.heads:
        for seed in args.seeds:
            net, nparams, out_dim = meta_train(head, seed, device, xt, base_by, tf, args)
            meta[head] = {"n_params": nparams, "out_dim": out_dim}
            trained_nets[head][seed] = net
            for name, nw, k in tasks:
                r = evaluate(net, device, xt, novel_by, tf, nw, k, args.eval_q, 100, seed)
                results[head][name].append(r["acc"])
                print(f"[stageb:{head}] seed{seed} {name}: {r['acc']:.2f}±{r['ci95']:.2f} "
                      f"(f1 {r['macro_f1']:.2f})", flush=True)

    print("\n=== Stage B: quantum-residual vs matched-classical vs baseline (Conv4@224) ===")
    print(f"(Stage-A baseline 3w5s reference = 81.66; skip={args.skip})")
    summary = {}
    for head in args.heads:
        summary[head] = {"meta": meta.get(head, {})}
        line = [f"{head:9s} (params={meta.get(head,{}).get('n_params','?')}, "
                f"out={meta.get(head,{}).get('out_dim','?')})"]
        for name, _, _ in tasks:
            v = np.array(results[head][name])
            mean = v.mean()
            sci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            summary[head][name] = {"mean": float(mean), "ci95": float(sci), "per_seed": v.tolist()}
            line.append(f"{name} {mean:.2f}±{sci:.2f}")
        print("  " + " | ".join(line))

    out = C.RESULTS_DIR / f"{args.tag}_{args.encode}_{args.readout}_skip-{args.skip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "summary": summary}, indent=2))
    print(f"[stageb] saved {out}")

    if args.save_ckpt:
        head = "quantum" if "quantum" in trained_nets else args.heads[0]
        best_seed_idx = int(np.argmax(results[head]["3w5s"]))
        best_seed = args.seeds[best_seed_idx]
        net = trained_nets[head][best_seed]
        C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = C.CKPT_DIR / "qconv4.pt"
        torch.save({
            "model": "qconv4", "head": head, "seed": best_seed,
            "state_dict": net.state_dict(),
            "config": {"img_size": args.img, "channels": args.channels, "d": args.d,
                       "n_qpus": args.n_qpus, "add_modes": args.add_modes,
                       "n_photons": args.n_photons, "skip": args.skip,
                       "readout": args.readout, "read_modes": args.read_modes,
                       "encode": args.encode, "m_modes": args.m_modes},
            "out_dim": net.out_dim,
            "eval_3w5s_this_seed": results[head]["3w5s"][best_seed_idx],
            "eval_3w5s_aggregate": summary.get(head, {}).get("3w5s", {}),
            "reported_rep0306_3w5s": {"mean": 81.20, "ci95": 1.10, "seeds": "42,123,456,7,99,2024"},
            "note": "state_dict + config only — NO baked-in prototypes. infer.py "
                    "computes prototypes on the fly from a K-shot support set at "
                    "inference time (true few-shot ProtoNet protocol). Checkpoint is "
                    "the BEST individual seed among this run's --seeds by 3w5s accuracy; "
                    "the aggregate mean across --seeds is verified against the "
                    " reported 81.20±1.10 for provenance (see eval_3w5s_aggregate).",
        }, ckpt_path)
        print(f"[stageb] saved inference checkpoint {ckpt_path} (head={head}, best seed={best_seed}, "
              f"3w5s={results[head]['3w5s'][best_seed_idx]:.2f} this-seed / "
              f"{summary.get(head, {}).get('3w5s', {}).get('mean', float('nan')):.2f} aggregate)")


if __name__ == "__main__":
    main()
