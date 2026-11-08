"""E3b: is a retrained encoder the old one up to a rotation, and is the old store still usable after an update?

Retrain the good-view encoder on day1+day2 (v2), keep the day1 encoder (v1) and its store. Then:
  * per-direction alignment after one Procrustes rotation (paper, Extended Data Fig. 4): correlation along v1's principal axes
  * R^2 of the rotation restricted to the top-k directions
  * a practical test: new-quarter queries embedded by v2, rotated, searched against the OLD v1 store (same-customer
    precision). The queries' threads are hidden from v2's training, so neither model has trained on them.
Writes results/e3b.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import build_pools, load  # noqa: E402
from jepa import Encoder, train_jepa  # noqa: E402
from metrics import l2n, procrustes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def main():
    c1 = load(os.path.join(ROOT, "data/era/day1/observations.jsonl"))
    c2 = load(os.path.join(ROOT, "data/era/day2/observations.jsonl"), accounts=c1.accounts)
    X1 = np.load(f"{OUT}/emb_good_s0.npy")                      # v1 store (day1 records)
    enc1 = Encoder(len(c1.vocab) + 1); enc1.load_state_dict(torch.load(f"{OUT}/enc_good_s0.pt"))
    # Queries for the old-index test are new-quarter records hidden from BOTH models: the old model never saw them,
    # and the retrained model is trained with their whole threads hidden (never a view).
    seen = {o["src_id"] for o in c1.obs}
    new = np.array([i for i, o in enumerate(c2.obs) if o["src_id"] not in seen and c2.account[i] >= 0])
    qrng = np.random.default_rng(5)
    threads = qrng.permutation(np.unique(c2.thread[new]))
    hidden, k = [], 0
    while len(hidden) < 2000 and k < len(threads):
        hidden += [i for i in new if c2.thread[i] == threads[k]]; k += 1
    hidden = np.array(hidden)
    qsel = qrng.choice(hidden, min(2000, len(hidden)), replace=False)
    true2 = c2.state.copy(); c2.state[hidden] = -1; build_pools(c2)
    enc2, _ = train_jepa(c2, "good", steps=int(os.environ.get("STEPS", 1500)), seed=11, log=print)
    c2.state = true2; build_pools(c2)
    idx2 = {o["src_id"]: i for i, o in enumerate(c2.obs)}
    common = [i for i, o in enumerate(c1.obs) if o["src_id"] in idx2]
    Y = enc2.embed([c2.tok[idx2[c1.obs[i]["src_id"]]] for i in common])     # v2 view of the same day1 records
    X = X1[common]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(common))
    fit, test = perm[:2000], perm[2000:8000]
    mx, my = X[fit].mean(0), Y[fit].mean(0)
    R = procrustes(X[fit] - mx, Y[fit] - my)                    # v2 -> v1: one rotation...
    BR = (Y[fit] - my) @ R
    s = float(((X[fit] - mx) * BR).sum() / (BR ** 2).sum())    # ...and one global scale
    R = s * R
    Xt, Yt = X[test] - mx, (Y[test] - my) @ R
    _, S, Vt = np.linalg.svd(X[fit] - mx, full_matrices=False)  # v1 principal axes
    px, py = Xt @ Vt.T, Yt @ Vt.T
    corr = [float(np.corrcoef(px[:, j], py[:, j])[0, 1]) for j in range(Vt.shape[0])]
    var = (S ** 2) / (S ** 2).sum()
    r2_topk = {}
    for k in (4, 8, 16, 32, 64, 256):
        a, b = px[:, :k], py[:, :k]
        r2_topk[k] = float(1 - ((a - b) ** 2).sum() / (a ** 2).sum())
    r2_all = float(1 - ((Xt - Yt) ** 2).sum() / (Xt ** 2).sum())

    # practical: new-quarter queries embedded by v2 (rotated) searched against the old v1 store
    store = l2n(X1 - mx)
    acct_store = c1.account

    def precision(Q):
        sim = l2n(Q) @ store.T
        nn_ = np.argpartition(-sim, 10, axis=1)[:, :10]
        return float((acct_store[nn_] == c2.account[qsel][:, None]).mean())

    c2_as1 = load(os.path.join(ROOT, "data/era/day2/observations.jsonl"), vocab=c1.vocab, accounts=c1.accounts)
    q_v1 = enc1.embed([c2_as1.tok[i] for i in qsel]) - mx                  # old model (upper reference)
    q_v2 = enc2.embed([c2.tok[i] for i in qsel])
    res = {
        "r2_all_dims": r2_all, "r2_top_k_principal_dims": r2_topk,
        "corr_by_direction_first_10": corr[:10], "corr_dirs_11_to_32_mean": float(np.mean(corr[10:32])),
        "corr_dirs_33_to_256_mean": float(np.mean(corr[32:])), "variance_share_top16": float(var[:16].sum()),
        "old_store_precision@10": {"query_by_old_encoder": precision(q_v1),
                                   "query_by_new_encoder_rotated": precision((q_v2 - my) @ R),
                                   "query_by_new_encoder_unrotated": precision(q_v2 - my)},
        "chance_precision@10": float(np.mean([(acct_store == a).mean() for a in c2.account[qsel]])),
        "n_queries": int(len(qsel)), "n_hidden_from_retraining": int(len(hidden)),
    }
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/e3b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
