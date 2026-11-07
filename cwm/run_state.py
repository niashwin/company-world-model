"""State tracking and one-step forecasting from learned state vectors: a first step from a state space toward
dynamics.

For every (customer, quarter) the state vector is the mean of its records' embeddings, from the frozen day-1
encoders applied to the day-2 corpus (everything through January 2027, so the last quarter was never seen in
training). Linear probes, scored leave-one-customer-out so no customer's identity can be memorized:
  now   facts about quarter q from the state vector of quarter q
  next  facts about quarter q+1 from the state vector of quarter q
Facts come from Era's ground truth: support tickets (4 buckets), worst ticket priority, any sales call, deal stage.
Baselines: majority class (share of the most common answer) and oracle persistence (next quarter = this quarter's true fact,
which reads the ground truth directly). Writes results/state.json and results/state_vectors.npz.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(__file__))
from data import load  # noqa: E402
from jepa import Encoder  # noqa: E402
from memory import make_questions  # noqa: E402
from run_prh import D1, D2, OUT, tfidf_lsa  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

KINDS = ("tickets", "priority", "call", "stage")


def next_q(q):
    y, n = int(q[:4]), int(q[-1]) + 1
    return f"{y + 1}Q1" if n == 5 else f"{y}Q{n}"


def state_vectors(c, X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = np.zeros((len(c.states), X.shape[1]), np.float32)
    n = np.zeros(len(c.states), int)
    for s in range(len(c.states)):
        idx = c.pools.get(s)
        if idx is not None and len(idx):
            S[s] = Xn[idx].mean(0); n[s] = len(idx)
    return S, n


def loao(F, y, groups, dim=32):
    """Leave-one-customer-out accuracy of a linear probe, and of the training folds' majority class."""
    hit, maj = [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            continue
        k = min(dim, tr.sum() - 1, F.shape[1])
        clf = make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(C=0.5, max_iter=3000))
        clf.fit(F[tr], y[tr])
        hit += list(clf.predict(F[te]) == y[te])
        m = Counter(y[tr]).most_common(1)[0][0]
        maj += list(y[te] == m)
    return float(np.mean(hit)), float(np.mean(maj)), len(hit)


def bootstrap_ci(F, y, groups, reps=200, seed=0):
    """Customer-level bootstrap of the leave-one-customer-out accuracy (resample customers with replacement)."""
    rng = np.random.default_rng(seed)
    per = {}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        k = min(32, tr.sum() - 1, F.shape[1])
        clf = make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(C=0.5, max_iter=3000))
        clf.fit(F[tr], y[tr])
        per[g] = clf.predict(F[te]) == y[te]
    gs = list(per)
    accs = []
    for _ in range(reps):
        pick = rng.choice(gs, len(gs))
        accs.append(np.concatenate([per[g] for g in pick]).mean())
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


if __name__ == "__main__":
    c1 = load(D1)
    c2 = load(D2, vocab=c1.vocab, accounts=c1.accounts)
    gt2 = json.load(open(D2.replace("observations.jsonl", "ground_truth.json")))
    facts = {}
    for x in make_questions(gt2, c2.states):
        facts.setdefault((x["account"], x["quarter"]), {})[x["kind"]] = x["answer"]
    sid = {s: i for i, s in enumerate(c2.states)}
    now_rows = [i for i, s in enumerate(c2.states) if s in facts]
    trans = [(sid[s], sid[(s[0], next_q(s[1]))]) for s in c2.states
             if (s[0], next_q(s[1])) in sid and s in facts and (s[0], next_q(s[1])) in facts]
    res = {"n_states": len(c2.states), "n_transitions": len(trans), "quarters": sorted({q for _, q in c2.states}),
           "labels": {k: dict(Counter(facts[c2.states[i]].get(k) for i in now_rows)) for k in KINDS}}

    embs = {}
    for arm in ("good", "bad", "ntp"):
        enc = Encoder(len(c1.vocab) + 1)
        enc.load_state_dict(torch.load(f"{OUT}/enc_{arm}_s0.pt"))
        embs[arm] = enc.embed(c2.tok)
    embs["tfidf"] = tfidf_lsa(c2)
    torch.manual_seed(0)
    embs["random"] = Encoder(len(c1.vocab) + 1).embed(c2.tok)

    vecs = {}
    for arm, X in embs.items():
        S, n = state_vectors(c2, X)
        vecs[arm] = S
        out = {}
        for kind in KINDS:
            rows = [i for i in now_rows if kind in facts[c2.states[i]]]
            y = np.array([facts[c2.states[i]][kind] for i in rows])
            g = np.array([c2.states[i][0] for i in rows])
            acc, maj, n_ = loao(S[rows], y, g)
            tr_rows = [(a, b) for a, b in trans if kind in facts[c2.states[a]] and kind in facts[c2.states[b]]]
            Fa = S[[a for a, _ in tr_rows]]
            yb = np.array([facts[c2.states[b]][kind] for _, b in tr_rows])
            ya = np.array([facts[c2.states[a]][kind] for a, _ in tr_rows])
            gb = np.array([c2.states[a][0] for a, _ in tr_rows])
            acc_n, maj_n, nn_ = loao(Fa, yb, gb)
            out[kind] = {"now_acc": acc, "now_majority_loo": maj, "now_n": n_,
                         "now_majority": Counter(y).most_common(1)[0][1] / len(y),
                         "next_acc": acc_n, "next_majority_loo": maj_n, "next_n": nn_,
                         "next_majority": Counter(yb).most_common(1)[0][1] / len(yb),
                         "next_oracle_persistence": float((ya == yb).mean())}
            if arm in ("good", "ntp", "bad", "tfidf"):
                out[kind]["next_ci"] = bootstrap_ci(Fa, yb, gb)
                out[kind]["now_ci"] = bootstrap_ci(S[rows], y, g)
        res[arm] = out
        print(arm, json.dumps(out), flush=True)
    res["mean_over_facts"] = {arm: {"now": float(np.mean([res[arm][k]["now_acc"] for k in KINDS])),
                                    "next": float(np.mean([res[arm][k]["next_acc"] for k in KINDS]))}
                              for arm in embs}
    res["mean_over_facts"]["majority"] = {"now": float(np.mean([res["good"][k]["now_majority"] for k in KINDS])),
                                          "next": float(np.mean([res["good"][k]["next_majority"] for k in KINDS]))}
    res["mean_over_facts"]["oracle_persistence"] = {
        "next": float(np.mean([res["good"][k]["next_oracle_persistence"] for k in KINDS]))}
    json.dump(res, open(f"{OUT}/state.json", "w"), indent=1)
    np.savez(f"{OUT}/state_vectors.npz", states=np.array([f"{a}|{q}" for a, q in c2.states]),
             counts=np.array([len(c2.pools.get(s, [])) for s in range(len(c2.states))]), **vecs)
    print("mean", json.dumps(res["mean_over_facts"]))
