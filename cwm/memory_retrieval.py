"""Which records the named-question memory stores retrieve (Table 7, version 3 keys): for each test question, the share
of the 16 retrieved records that belong to the asked customer and customer-quarter, with retrieval by the content of
the retrieval channel among records of the question's quarter, and, for comparison, by whole store rows (versions 1-2).
Writes results/memory2_q_retrieval.json."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, time_code, tokenize  # noqa: E402
from jepa import Encoder  # noqa: E402
from memory import fit_store_map, make_questions  # noqa: E402
from pretrained import minilm_embed  # noqa: E402
from run_prh import D1, OUT  # noqa: E402


def l2(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


if __name__ == "__main__":
    c = load(D1)
    gt = json.load(open(D1.replace("observations.jsonl", "ground_truth.json")))
    enc = Encoder(len(c.vocab) + 1)
    enc.load_state_dict(torch.load(os.path.join(OUT, "enc_good_s0.pt")))
    tok = lambda texts: [np.array([c.vocab.get(w, 0) for w in tokenize(s)][:96], dtype=np.int64) for s in texts]  # noqa: E731
    qs = make_questions(gt, c.states)
    rng = np.random.default_rng(0)                          # the split of run_memory2.py
    states = sorted({(q["account"], q["quarter"]) for q in qs})
    perm = rng.permutation(len(states))
    split = {states[i]: ("train" if k < 0.7 * len(states) else "val" if k < 0.8 * len(states) else "test")
             for k, i in enumerate(perm)}
    test = [q for q in qs if split[(q["account"], q["quarter"])] == "test"]
    texts, q_ts = [q["text"] for q in test], np.array([q["ts"] for q in test])
    n_sys = len(c.systems)
    acct = {a: i for i, a in enumerate(c.accounts)}
    sid = {s: i for i, s in enumerate(c.states)}
    qa = np.array([acct[q["account"]] for q in test]); qs_id = np.array([sid[(q["account"], q["quarter"])] for q in test])
    rec_q = np.array([c.states[s][1] for s in c.state])
    Xg, Xm = np.load(os.path.join(OUT, "emb_good_s0.npy")), np.load(os.path.join(OUT, "minilm_day1.npy"))
    Qg, Qm = enc.embed(tok(texts)), minilm_embed(texts)
    chans = {"good": (Xg, Qg), "minilm": (Xm, Qm),
             "dual": (np.concatenate([l2(Xg), l2(Xm)], 1), np.concatenate([l2(Qg), l2(Qm)], 1))}
    res = {"test_questions": len(test)}
    for name, (X, Q) in chans.items():
        f = fit_store_map(X)
        out = {}
        rows = lambda sem, ts, sy: np.concatenate(  # noqa: E731
            [sem, time_code(np.asarray(ts, float), 16), np.eye(n_sys)[sy] if sy is not None else np.zeros((len(sem), n_sys))], 1)
        for keys in ("quarter", "row"):
            if keys == "row":
                sim = l2(rows(f(Q), q_ts, None)) @ l2(rows(f(X), c.ts, c.system)).T
            else:
                sim = l2(f(Q)) @ l2(f(X)).T
                sim = np.where(rec_q[None, :] == np.array([q["quarter"] for q in test])[:, None], sim, -9.0)
            top = np.argsort(-sim, 1)[:, :16]
            out[keys] = {"same_customer": float(np.mean(c.account[top] == qa[:, None])),
                         "same_state": float(np.mean(c.state[top] == qs_id[:, None]))}
        res[name] = out
        print(name, json.dumps(out), flush=True)
    res["route"] = res["good"]                               # the routed store retrieves with the good-view channel
    json.dump(res, open(os.path.join(OUT, "memory2_q_retrieval.json"), "w"), indent=1)
