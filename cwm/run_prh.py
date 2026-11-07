"""E1-E3: does the Platonic Representation Theorem carry over to a company?

E1  One encoder over all seven systems, three objectives on identical texts (good views / bad views / next-token):
    does it encode the company's state (account x quarter) rather than which system or template wrote it?
E2  Separate encoders for messages (Slack+Gmail) and systems of record (Zendesk, Salesforce, Jira, Gong, Calendar),
    never sharing a pair: do they agree up to ONE rotation, and only with good views?
E3  New data arrives (Era day2 = one quarter later): does the representation carry over to the new quarter, and is a
    retrained encoder the old one up to a rotation (so a stored memory survives model updates)?
Writes results/prh.json and caches embeddings in results/emb_*.npy.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import load  # noqa: E402
from jepa import train_jepa, train_ntp  # noqa: E402
from metrics import (explained, knn_precision, mutual_knn, paired_retrieval, probe, procrustes,  # noqa: E402
                     whitener)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1 = os.path.join(ROOT, "data/era/day1/observations.jsonl")
D2 = os.path.join(ROOT, "data/era/day2/observations.jsonl")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 1500))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
LOG = open(os.path.join(OUT, "prh_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def train(c, arm, seed, modality=None, steps=STEPS, subset=None):
    if arm == "ntp":
        return train_ntp(c, modality=modality, steps=steps, seed=seed, log=log, obs_subset=subset)[0]
    return train_jepa(c, views=arm, modality=modality, steps=steps, seed=seed, log=log, obs_subset=subset)[0]


def tfidf_lsa(c, d=256):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit_transform([o["text"] for o in c.obs])
    return TruncatedSVD(d, random_state=0).fit_transform(X).astype(np.float32)


def e1_metrics(c, X, rng):
    linked = np.where(c.state >= 0)[0]
    q = rng.choice(linked, min(3000, len(linked)), replace=False)
    A, B = linked[c.modality[linked] == 0], linked[c.modality[linked] == 1]
    sub = lambda idx, n: rng.choice(idx, min(n, len(idx)), replace=False)  # noqa: E731
    a_tr, b_tr = sub(A, 6000), sub(B, 6000)
    ex = explained(X[linked], {"state": c.state[linked], "account": c.account[linked], "system": c.system[linked]})
    return {
        "account_p@10_cross_system": knn_precision(X, c.account, c.system, c.thread, q),
        "state_p@10_cross_system": knn_precision(X, c.state, c.system, c.thread, q),
        "probe_account_messages_to_records": probe(X[a_tr], c.account[a_tr], X[B], c.account[B]),
        "probe_account_records_to_messages": probe(X[b_tr], c.account[b_tr], X[A], c.account[A]),
        "rho_state": ex["state"]["rho"], "r2_state": ex["state"]["r2"], "r2_account": ex["account"]["r2"],
        "r2_system": ex["system"]["r2"],
    }


def e1(c, results):
    log("== E1: one encoder, three objectives, identical texts")
    rng = np.random.default_rng(0)
    res = results.setdefault("E1", {})
    X = tfidf_lsa(c); np.save(f"{OUT}/emb_tfidf.npy", X)
    res["tfidf_lsa"] = e1_metrics(c, X, rng); log("tfidf", res["tfidf_lsa"])
    from jepa import Encoder
    import torch
    torch.manual_seed(0)
    X = Encoder(len(c.vocab) + 1).embed(c.tok)
    res["random_init"] = e1_metrics(c, X, rng); log("random", res["random_init"])
    for arm in ("good", "bad", "ntp"):
        for seed in SEEDS:
            t = time.time()
            enc = train(c, arm, seed)
            X = enc.embed(c.tok)
            np.save(f"{OUT}/emb_{arm}_s{seed}.npy", X)
            import torch as _t
            _t.save(enc.state_dict(), f"{OUT}/enc_{arm}_s{seed}.pt")
            m = e1_metrics(c, X, np.random.default_rng(0))
            m["train_seconds"] = time.time() - t
            res[f"{arm}_s{seed}"] = m
            log(arm, seed, m)
            json.dump(results, open(f"{OUT}/prh.json", "w"), indent=1)


def pairs_for(c, states, n, rng):
    """(A obs, B obs) pairs of the same world state, states weighted by how often they are observed."""
    states = [s for s in states if (s, 0) in c.pools and (s, 1) in c.pools]
    w = np.array([min(len(c.pools[(s, 0)]), len(c.pools[(s, 1)])) for s in states], dtype=float)
    pick = rng.choice(len(states), n, p=w / w.sum())
    a = np.array([rng.choice(c.pools[(states[i], 0)]) for i in pick])
    b = np.array([rng.choice(c.pools[(states[i], 1)]) for i in pick])
    return a, b


def e2(c, results):
    log("== E2: separately trained encoders per modality, aligned by one rotation")
    rng = np.random.default_rng(1)
    both = [s for s in range(len(c.states)) if (s, 0) in c.pools and (s, 1) in c.pools]
    rng.shuffle(both)
    fit_states, test_states = both[: len(both) // 2], both[len(both) // 2:]
    fa, fb = pairs_for(c, fit_states, 2000, rng)
    ta, tb = pairs_for(c, test_states, 1000, rng)
    res = results.setdefault("E2", {"n_states_with_both": len(both)})
    A_all, B_all = np.where((c.state >= 0) & (c.modality == 0))[0], np.where((c.state >= 0) & (c.modality == 1))[0]

    def evaluate(XA, XB):
        wa, wb = whitener(XA[A_all], 64), whitener(XB[B_all], 64)   # unsupervised, per modality
        out = {"mutual_10nn_test": mutual_knn(wa(XA[ta]), wb(XB[tb]), 10)}
        for n_anchor in (64, 256, 2000):
            R = procrustes(wb(XB[fb[:n_anchor]]), wa(XA[fa[:n_anchor]]))   # maps A -> B
            r = paired_retrieval(wa(XA[ta]) @ R, wb(XB[tb]), c.state[ta], c.state[tb])
            ra = paired_retrieval(wa(XA[ta]) @ R, wb(XB[tb]), c.account[ta], c.account[tb], ks=(1,))
            out[f"anchors_{n_anchor}"] = {"state_R@1": r["R@1"], "state_R@10": r["R@10"], "account_R@1": ra["R@1"]}
        Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(64, 64)))
        out["random_rotation_state_R@1"] = paired_retrieval(wa(XA[ta]) @ Q, wb(XB[tb]), c.state[ta], c.state[tb])["R@1"]
        return out

    res["chance_state_R@1"] = float(np.mean([(c.state[tb] == s).mean() for s in c.state[ta]]))
    res["chance_account_R@1"] = float(np.mean([(c.account[tb] == s).mean() for s in c.account[ta]]))
    for arm in ("good", "bad", "ntp"):
        for seed in SEEDS:
            t = time.time()
            encA, encB = train(c, arm, seed, modality=0), train(c, arm, seed + 100, modality=1)
            m = evaluate(encA.embed(c.tok), encB.embed(c.tok))
            m["train_seconds"] = time.time() - t
            res[f"{arm}_s{seed}"] = m
            log(arm, seed, json.dumps(m))
            json.dump(results, open(f"{OUT}/prh.json", "w"), indent=1)


def e3(c1, results):
    log("== E3: a quarter of new data (day2)")
    import torch
    from jepa import Encoder
    res = results.setdefault("E3", {})
    c2 = load(D2, accounts=c1.accounts)                 # day2 vocabulary for retrained encoders
    seen = {o["src_id"] for o in c1.obs}
    new = np.array([i for i, o in enumerate(c2.obs) if o["src_id"] not in seen and c2.account[i] >= 0])
    old_in_2 = np.array([i for i, o in enumerate(c2.obs) if o["src_id"] in seen])
    res["n_new_linked"] = int(len(new))
    c2_as1 = load(D2, vocab=c1.vocab, accounts=c1.accounts)  # day2 text through day1's tokenizer
    rng = np.random.default_rng(2)
    for arm in ("good", "bad", "ntp"):
        enc = Encoder(len(c1.vocab) + 1)
        enc.load_state_dict(torch.load(f"{OUT}/enc_{arm}_s{SEEDS[0]}.pt"))
        X1 = np.load(f"{OUT}/emb_{arm}_s{SEEDS[0]}.npy")
        Xn = enc.embed([c2_as1.tok[i] for i in new])
        tr = rng.choice(len(c1.obs), 8000, replace=False)
        r = {"probe_account_day1_to_new_quarter": probe(X1[tr], c1.account[tr], Xn, c2.account[new])}
        # retrain on everything up to day2 and compare with the day1 model on the day1 observations
        if arm != "ntp":
            enc2 = train(c2, arm, SEEDS[0] + 7)
            Y = enc2.embed([c2.tok[i] for i in old_in_2])
            src2 = [c2.obs[i]["src_id"] for i in old_in_2]
            idx1 = {o["src_id"]: i for i, o in enumerate(c1.obs)}
            X = X1[[idx1[s] for s in src2]]
            wx, wy = whitener(X, 64), whitener(Y, 64)
            Xw, Yw = wx(X), wy(Y)
            perm = rng.permutation(len(Xw))
            fit, test = perm[:256], perm[256:4256]
            R = procrustes(Xw[fit], Yw[fit])            # new model -> old model, from 256 anchors
            resid = ((Yw[test] @ R - Xw[test]) ** 2).sum(1).mean() / (Xw[test] ** 2).sum(1).mean()
            r["v2_to_v1_rotation_r2"] = float(1 - resid)
            r["v2_to_v1_mutual_10nn"] = mutual_knn(Xw[test[:2000]], Yw[test[:2000]] @ R, 10)
        res[arm] = r
        log("E3", arm, r)
        json.dump(results, open(f"{OUT}/prh.json", "w"), indent=1)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["e1", "e2", "e3"]
    path = f"{OUT}/prh.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    c = load(D1)
    log(f"day1: {len(c.obs)} linked observations, {len(c.states)} states, {len(c.accounts)} accounts, vocab {len(c.vocab)}")
    results["data"] = {"observations": len(c.obs), "states": len(c.states), "accounts": len(c.accounts),
                       "per_system": {s: int((c.system == i).sum()) for i, s in enumerate(c.systems)}}
    for st in stages:
        {"e1": e1, "e2": e2, "e3": e3}[st](c, results)
        json.dump(results, open(path, "w"), indent=1)
    log("done", stages)
