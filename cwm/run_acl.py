"""Access control built into the world model: leak tests, cost and non-interference (results/acl.json).

Readers (cwm/acl.py): CEO, account executive, support engineer, software engineer, financial analyst.
Canaries: five customers get secrets planted in 30 of their CRM and call records each, records that the support
engineer and the software engineer may not read: a unique codename per customer, and one shared sentence.
For each of these two readers we compare three configurations:
  open      one encoder trained on every record; search and states over every record      (no access control)
  filtered  the same encoder; search and states over the reader's records only            (retrieval-time filtering)
  built-in  an encoder trained only on the reader's records; search and states over them  (access control built in)
and measure the access advantage: how much planting the secrets changes what the reader can find (each configuration
is also built without the secrets: open_clean, and the reader's own encoder retrained on the clean corpus).
Cost: for every reader, placement of held-out readable records (customer), open encoder vs the reader's own encoder.
Non-interference: the support engineer's encoder is trained twice, with and without the canaries, and its training
data and outputs are compared byte for byte.

Stages:  train:open  train:open_clean  train:<reader>  train:support_engineer_clean   (results/acl_<name>.npz)
         eval                                                       (writes results/acl.json)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(__file__))
from acl import POLICY_TABLE, READERS, email_index, record_readers  # noqa: E402
from data import build_pools, load, tokenize  # noqa: E402
from jepa import train_jepa  # noqa: E402
from metrics import l2n  # noqa: E402
from run_extras import heldout_split  # noqa: E402
from run_prh import D1, OUT  # noqa: E402

STEPS = int(os.environ.get("STEPS", 1500))
CODENAMES = ["kestrelwind", "marlowgate", "tessaline", "corvanthe", "brightfenn"]
SECRET = "privately we expect them to churn to a competitor next quarter"
UNAUTHORIZED = ("support_engineer", "software_engineer")
CANARY_SYSTEMS = ("salesforce", "gong")
PER_CUSTOMER = 30


def setup():
    """Full linked corpus, readers per record, held-out split and canary plan (all deterministic)."""
    c = load(D1)
    gt = json.load(open(D1.replace("observations.jsonl", "ground_truth.json")))
    emails = email_index(gt)
    readers = [record_readers(o, emails) for o in c.obs]
    tr, te = heldout_split(c)
    is_test = np.zeros(len(c.obs), bool); is_test[te] = True
    for w in CODENAMES:
        assert w not in c.vocab, w
    # canary customers: the five with the most CRM and call records in the training split
    cand = {}
    for i in tr:
        if c.obs[i]["system"] in CANARY_SYSTEMS:
            cand.setdefault(int(c.account[i]), []).append(int(i))
    top = sorted(cand, key=lambda a: -len(cand[a]))[:len(CODENAMES)]
    rng = np.random.default_rng(7)
    plan = {}
    for k, a in enumerate(top):
        for i in rng.choice(cand[a], min(PER_CUSTOMER, len(cand[a])), replace=False):
            plan[int(i)] = k
    return c, readers, is_test, top, plan


def canary_text(o, k):
    return f"{o['text']} Internal note: {CODENAMES[k]}. {SECRET.capitalize()}."


def visible(readers, user):
    return np.array([user in r for r in readers])


def train_config(name):
    c, readers, is_test, top, plan = setup()
    clean = name.endswith("_clean")
    reader = name[:-len("_clean")] if clean else name
    idx = np.arange(len(c.obs)) if reader == "open" else np.where(visible(readers, READERS[reader]))[0]
    path = os.path.join(OUT, f"acl_corpus_{name}.jsonl")
    with open(path, "w") as f:
        for i in idx:
            o = dict(c.obs[i])
            if not clean and int(i) in plan:
                o["text"] = canary_text(o, plan[int(i)])
            f.write(json.dumps(o, sort_keys=True) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    cs = load(path)                                  # vocabulary from the readable records only
    assert len(cs.obs) == len(idx)
    true_state = cs.state.copy()
    cs.state[is_test[idx]] = -1; build_pools(cs)     # held-out threads are never a view
    t = time.time()
    enc, _ = train_jepa(cs, "good", steps=STEPS, seed=0)
    cs.state = true_state; build_pools(cs)
    X = enc.embed(cs.tok)
    queries = CODENAMES + [SECRET]
    Q = enc.embed([np.array([cs.vocab.get(w, 0) for w in tokenize(q)], dtype=np.int64) for q in queries])
    np.savez(os.path.join(OUT, f"acl_{name}.npz"), X=X, Q=Q, idx=idx, digest=digest,
             known=np.array([w in cs.vocab for w in CODENAMES]), seconds=time.time() - t)
    os.remove(path)
    print(name, "records", len(idx), "sha256", digest[:16], f"{time.time() - t:.0f}s", flush=True)


def leak_metrics(X, Q, rows, c, top, k_top=10):
    """X: embeddings of the candidate records `rows` (indices into the full corpus)."""
    Xn, Qn = l2n(X), l2n(Q)
    acct = c.account[rows]
    out = {"codename_precision": [], "codename_chance": []}
    for k, a in enumerate(top):
        sim = Xn @ Qn[k]
        nn = np.argsort(-sim)[:k_top]
        out["codename_precision"].append(float((acct[nn] == a).mean()))
        out["codename_chance"].append(float((acct == a).mean()))
    # customer states (mean of readable records), ranked by similarity to the secret sentence
    custs = np.unique(acct)
    S = l2n(np.stack([Xn[acct == a].mean(0) for a in custs]))
    order = custs[np.argsort(-(S @ Qn[-1]))]
    out["secret_precision@5"] = float(np.isin(order[:5], top).mean())
    out["secret_chance@5"] = float(len(set(top) & set(custs)) / len(custs))
    out["codename_precision"] = float(np.mean(out["codename_precision"]))
    out["codename_chance"] = float(np.mean(out["codename_chance"]))
    out["codename_advantage"] = out["codename_precision"] - out["codename_chance"]
    out["secret_advantage"] = out["secret_precision@5"] - out["secret_chance@5"]
    return out


def placement(X, rows, c, is_test):
    """Held-out readable records placed at the nearest readable training state (customer and state accuracy)."""
    Xn = l2n(X)
    tr = ~is_test[rows]
    st = c.state[rows]
    ids = np.unique(st[tr])
    S = l2n(np.stack([Xn[tr & (st == s)].mean(0) for s in ids]))
    pred = ids[np.argmax(Xn[~tr] @ S.T, axis=1)]
    acct_of = {s: c.accounts.index(c.states[s][0]) for s in ids}
    pa = np.array([acct_of[s] for s in pred])
    return {"customer_acc": float((pa == c.account[rows][~tr]).mean()),
            "state_acc": float((pred == st[~tr]).mean()), "n_test": int((~tr).sum())}


def evaluate():
    c, readers, is_test, top, plan = setup()
    z = {n[4:-4]: np.load(os.path.join(OUT, n)) for n in os.listdir(OUT) if n.startswith("acl_") and n.endswith(".npz")}
    res = {"policy": POLICY_TABLE, "canary_customers": [c.accounts[a] for a in top],
           "canary_records": len(plan), "codenames": CODENAMES, "secret": SECRET, "readers": {}}
    systems = sorted(set(o["system"] for o in c.obs))
    X_open, Q_open = z["open"]["X"], z["open"]["Q"]
    for name, user in READERS.items():
        vis = visible(readers, user)
        rows = np.where(vis)[0]
        r = {"user": user, "visible_share": float(vis.mean()), "visible_records": int(vis.sum()),
             "visible_share_by_system": {s: float(vis[[o["system"] == s for o in c.obs]].mean()) for s in systems},
             "sees_canaries": bool(all(vis[i] for i in plan))}
        own = z.get(name) if name != "ceo" else z["open"]      # the CEO reads everything: own encoder = open
        r["placement_open_encoder"] = placement(X_open[rows], rows, c, is_test)
        if own is not None:
            r["placement_own_encoder"] = placement(own["X"] if name != "ceo" else X_open, rows, c, is_test)
        if name in UNAUTHORIZED and own is not None:
            allrows = np.arange(len(c.obs))
            own_clean = z.get(f"{name}_clean", own)            # identical by construction; retrained for one reader
            oc = z.get("open_clean")
            confs = {"open": ((X_open, Q_open, allrows), (oc["X"], oc["Q"], allrows) if oc is not None else None),
                     "filtered": ((X_open[rows], Q_open, rows), (oc["X"][rows], oc["Q"], rows) if oc is not None else None),
                     "built_in": ((own["X"], own["Q"], rows), (own_clean["X"], own_clean["Q"], rows))}
            r["leak"] = {"codenames_in_own_vocabulary": int(own["known"].sum()),
                         "built_in_without_secrets_retrained": bool(f"{name}_clean" in z)}
            for conf, (w, wo) in confs.items():
                m = {"with": leak_metrics(*w, c, top)}
                if wo is not None:
                    m["without"] = leak_metrics(*wo, c, top)
                    m["codename_advantage"] = m["with"]["codename_precision"] - m["without"]["codename_precision"]
                    m["secret_advantage"] = m["with"]["secret_precision@5"] - m["without"]["secret_precision@5"]
                r["leak"][conf] = m
        res["readers"][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "visible_share_by_system"}), flush=True)
    if "support_engineer" in z and "support_engineer_clean" in z:
        a, b = z["support_engineer"], z["support_engineer_clean"]
        res["non_interference"] = {
            "reader": "support_engineer",
            "training_data_identical": bool(str(a["digest"]) == str(b["digest"])),
            "max_abs_diff_embeddings": float(np.abs(a["X"] - b["X"]).max()),
            "max_abs_diff_queries": float(np.abs(a["Q"] - b["Q"]).max()),
            "open_encoder_training_data_changed": True}
        print("non-interference", res["non_interference"], flush=True)
    json.dump(res, open(os.path.join(OUT, "acl.json"), "w"), indent=1)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        key, _, arg = st.partition(":")
        if key == "train":
            train_config(arg)
        elif key == "eval":
            evaluate()
