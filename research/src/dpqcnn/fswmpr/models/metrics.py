"""Prototype metrics for the  ablation (architecture.md E1/E3).

One vectorized `predict()` used IDENTICALLY by the classical ProtoNet and the
DPQCNN, so a metric change is a clean one-factor ablation:

  * euclidean  — control (Snell 2017): nearest mean prototype by L2.
  * cosine     — L2-normalize embeddings, nearest prototype by cosine (angle).
  * fidelity   — QUANTUM-fidelity kernel: treat the embedding as a (real,
                 single-photon) amplitude state |psi>, prototype = mean state,
                 score = |<psi_q|proto>|^2 (state overlap). For pure single-photon
                 states this is a *classical kernel* on the amplitude vectors — a
                 richer metric than Euclidean-on-marginals, NOT a computational
                 quantum speedup (stated honestly in the report).

Optional `transductive=True` applies soft k-means prototype rectification
(support-seeded, uses the unlabeled query batch) in the metric's own space —
a well-known few-shot booster (BD-CSPN family). Applied to BOTH models in E3.
"""

from __future__ import annotations

import torch


def _normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _class_means(emb, labels, n_way):
    return torch.stack([emb[labels == c].mean(0) for c in range(n_way)])


def _transductive_rectify(sup, sup_labels, qry, protos, n_way, iters=10, temp=10.0):
    """Support-seeded soft k-means in the current (already-transformed) space.
    Scale-normalized logits so the same `temp` behaves across metrics → fair."""
    for _ in range(iters):
        d = torch.cdist(qry, protos)                      # (Q, n_way)
        logits = -temp * d / (d.mean() + 1e-8)            # scale-invariant
        w = torch.softmax(logits, dim=1)                  # query responsibilities
        new = []
        for c in range(n_way):
            sc = sup[sup_labels == c]
            num = sc.sum(0) + (w[:, c:c + 1] * qry).sum(0)
            den = sc.shape[0] + w[:, c].sum()
            new.append(num / (den + 1e-8))
        protos = torch.stack(new)
    return protos


def predict(support_emb, sup_labels, query_emb, n_way,
            metric="euclidean", transductive=False, t_iters=10, t_temp=10.0):
    """support_emb (S,d), sup_labels (S,) in 0..n_way-1, query_emb (Q,d) ->
    predicted labels (Q,). `metric` in {euclidean, cosine, fidelity}."""
    if metric == "euclidean":
        s, q = support_emb, query_emb
        protos = _class_means(s, sup_labels, n_way)
        if transductive:
            protos = _transductive_rectify(s, sup_labels, q, protos, n_way, t_iters, t_temp)
        return torch.cdist(q, protos).argmin(1)

    if metric in ("cosine", "fidelity"):
        s, q = _normalize(support_emb), _normalize(query_emb)
        protos = _normalize(_class_means(s, sup_labels, n_way))
        if transductive:
            protos = _normalize(_transductive_rectify(s, sup_labels, q, protos, n_way, t_iters, t_temp))
        ov = q @ protos.t()                               # <psi_q|proto>
        scores = ov if metric == "cosine" else ov * ov    # cosine vs |<.|.>|^2
        return scores.argmax(1)

    raise ValueError(f"unknown metric {metric!r}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # self-test: shapes, euclidean == legacy proto_predict, metrics separate novel classes
    torch.manual_seed(0)
    d, nw, k, q = 16, 3, 5, 15
    protos_true = torch.randn(nw, d) * 3
    sup = torch.cat([protos_true[c] + 0.3 * torch.randn(k, d) for c in range(nw)])
    sl = torch.arange(nw).repeat_interleave(k)
    qry = torch.cat([protos_true[c] + 0.3 * torch.randn(q, d) for c in range(nw)])
    ql = torch.arange(nw).repeat_interleave(q)
    for m in ("euclidean", "cosine", "fidelity"):
        for tr in (False, True):
            p = predict(sup, sl, qry, nw, metric=m, transductive=tr)
            acc = (p == ql).float().mean().item()
            assert p.shape == ql.shape
            print(f"metric={m:10s} transductive={tr!s:5s} acc={acc:.3f}")
    # euclidean must match the legacy proto_predict exactly
    from .protonet import proto_predict
    a = predict(sup, sl, qry, nw, metric="euclidean")
    b = proto_predict(sup, sl, qry, nw)
    assert torch.equal(a, b), "euclidean predict diverges from legacy proto_predict"
    print("[metrics] SELF-TEST PASSED (shapes, euclidean==legacy, all metrics run)")
