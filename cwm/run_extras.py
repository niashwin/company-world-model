"""Extra experiments for the paper.

seeds   E1 for more seeds (good / bad / ntp), for error bars.
mask    E1 with customer identity masked out of every record (names, contacts, domains, emails, ids):
        does the good-view encoder learn more than "which customer is this"?
noise   Good views built from corrupted links: a fraction p of records is assigned to a wrong state before
        training (a wrong merge joins two states, Thm 2). Evaluated against the true states.
heldout 25% of threads are hidden from training (their records are treated as unlinked: never a view, never a
        prefix). Evaluated only on those held-out records: cross-system neighbours among held-out records, and
        placement of each held-out record at the nearest training state (customer and quarter).
Usage: python run_extras.py OUTNAME stage[:arg] ...   e.g.  run_extras.py extras_a seeds:good,bad
Writes results/OUTNAME.json.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(__file__))
from data import build_pools, load  # noqa: E402
from metrics import knn_precision, l2n  # noqa: E402
from jepa import train_jepa, train_ntp  # noqa: E402
from run_prh import D1, OUT, e1_metrics, tfidf_lsa  # noqa: E402

STEPS = int(os.environ.get("STEPS", 1500))


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def train(c, arm, seed):
    if arm == "ntp":
        return train_ntp(c, steps=STEPS, seed=seed, log=None)[0]
    return train_jepa(c, views=arm, steps=STEPS, seed=seed, log=None)[0]


def identity_only_state_precision(c, n=3000, seed=0):
    """Expected same-state precision@k of a model that knows the customer and nothing else (all neighbours from the
    query's customer, drawn uniformly from its other-system records)."""
    rng = np.random.default_rng(seed)
    linked = np.where(c.state >= 0)[0]
    q = rng.choice(linked, min(n, len(linked)), replace=False)
    from collections import Counter
    ca = Counter(zip(c.account[linked], c.system[linked])); cs = Counter(zip(c.state[linked], c.system[linked]))
    na_ = Counter(c.account[linked]); ns_ = Counter(c.state[linked])
    v = []
    for i in q:
        na = na_[c.account[i]] - ca[(c.account[i], c.system[i])]
        ns = ns_[c.state[i]] - cs[(c.state[i], c.system[i])]
        if na > 0:
            v.append(ns / na)
    return float(np.mean(v))


# ------------------------------------------------------------------------------------------------ identity mask
GENERIC = {"bank", "labs", "lab", "payments", "pay", "systems", "system", "components", "transport", "logistics",
           "health", "capital", "group", "inc", "ltd", "llc", "co", "technologies", "technology", "tech", "software",
           "robotics", "cloud", "data", "ai", "analytics", "solutions", "partners", "global", "holdings", "networks",
           "energy", "foods", "freight", "industries", "insurance", "media", "medical", "motors", "retail", "the",
           "and", "of", "works", "studio", "studios", "finance", "financial", "bio", "labs.", "corp", "company"}


def identity_patterns(gt):
    """Regexes that remove customer identity: account names and their distinctive words, contact names, domains,
    email addresses and record ids. Internal staff names are kept (they are the company's own people)."""
    words, phrases = set(), set()
    acct_ids = {o["external_id"] for o in gt["zendesk_orgs"]} | {a["Id"] for a in gt["accounts"]}
    for a in gt["accounts"]:
        if a["Id"] not in acct_ids:
            continue
        phrases.add(a["Name"].lower())
        for w in re.findall(r"[a-z0-9]+", a["Name"].lower()):
            if w not in GENERIC and len(w) >= 3:
                words.add(w)
        site = (a.get("Website") or "").lower()
        m = re.search(r"(?:https?://)?(?:www\.)?([a-z0-9.-]+)", site)
        if m and m.group(1):
            phrases.add(m.group(1))
            words.add(m.group(1).split(".")[0])
    for o in gt["zendesk_orgs"]:
        phrases.add(o["name"].lower())
        for d in o.get("domain_names") or []:
            phrases.add(d.lower()); words.add(d.lower().split(".")[0])
    for ct in gt["contacts"]:
        for w in re.findall(r"[a-z]+", (ct.get("Name") or "").lower()):
            if len(w) >= 3:
                words.add(w)
    words = sorted(words, key=len, reverse=True)
    phrases = sorted(phrases, key=len, reverse=True)
    pats = [re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                                       # emails
            re.compile(r"\b(?:acct|deal|ticket|contact|call|case|opp)[-_]\d+\b"),          # record ids
            re.compile(r"\breq_[0-9a-f]{6,}\b"), re.compile(r"\b[0-9a-f]{12,}\b"),         # request / hex ids
            re.compile(r"\b\d{6,}\b")]                                                     # long numeric ids
    pats += [re.compile(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])") for p in phrases]
    pats += [re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(w) for w in words) + r")(?![a-z0-9])")]
    return pats


def mask_text(text, pats):
    t = (text or "").lower()
    for p in pats:
        t = p.sub(" xent ", t)
    return t


def masked_corpus(src, gt_path):
    gt = json.load(open(gt_path))
    pats = identity_patterns(gt)
    dst = os.path.join(OUT, "masked_observations.jsonl")
    n = {"all_records": 0, "all_records_changed": 0, "linked_records": 0, "linked_records_changed": 0}
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            o = json.loads(line)
            t = mask_text(o.get("text"), pats)
            changed = t != (o.get("text") or "").lower()
            n["all_records"] += 1; n["all_records_changed"] += changed
            if o.get("account_id") and o.get("text") and o.get("ts"):
                n["linked_records"] += 1; n["linked_records_changed"] += changed
            o["text"] = t
            g.write(json.dumps(o) + "\n")
    return dst, n


# ------------------------------------------------------------------------------------------------ stages
def stage_seeds(c, res, arg):
    arms = (arg or "good,bad,ntp").split(",")
    for seed in (1, 2):
        for arm in arms:
            t = time.time()
            enc = train(c, arm, seed)
            m = e1_metrics(c, enc.embed(c.tok), np.random.default_rng(0))
            m["train_seconds"] = time.time() - t
            res.setdefault("E1_seeds", {})[f"{arm}_s{seed}"] = m
            log("seeds", arm, seed, json.dumps(m))
            yield


def stage_mask(c, res, arg):
    path, counts = masked_corpus(D1, D1.replace("observations.jsonl", "ground_truth.json"))
    cm = load(path)
    out = res.setdefault("mask", dict(counts, **{"identity_only_state_p@10": identity_only_state_precision(cm)}))
    sample = [cm.obs[i]["text"][:200] for i in np.random.default_rng(3).choice(len(cm.obs), 5, replace=False)]
    out["examples"] = sample
    out["tfidf_lsa"] = e1_metrics(cm, tfidf_lsa(cm), np.random.default_rng(0)); log("mask tfidf", out["tfidf_lsa"])
    yield
    for arm in (arg or "good").split(","):
        t = time.time()
        enc = train(cm, arm, 0)
        m = e1_metrics(cm, enc.embed(cm.tok), np.random.default_rng(0))
        m["train_seconds"] = time.time() - t
        out[f"{arm}_s0"] = m
        log("mask", arm, json.dumps(m))
        yield


def stage_noise(c, res, arg):
    ps = [float(p) for p in (arg or "0.25,0.5").split(",")]
    linked = np.where(c.state >= 0)[0]
    true_state = c.state.copy()
    for p in ps:
        rng = np.random.default_rng(int(p * 1000))
        noisy = true_state.copy()
        flip = rng.choice(linked, int(round(p * len(linked))), replace=False)
        n_states = len(c.states)
        for i in flip:                                   # a wrong link: the record joins another state
            s = rng.integers(n_states - 1)
            noisy[i] = s if s < true_state[i] else s + 1
        c.state = noisy; build_pools(c)
        t = time.time()
        enc = train(c, "good", 0)
        c.state = true_state; build_pools(c)            # evaluate against the truth
        m = e1_metrics(c, enc.embed(c.tok), np.random.default_rng(0))
        m["train_seconds"] = time.time() - t
        res.setdefault("noise", {})[f"p{p}"] = m
        log("noise", p, json.dumps(m))
        yield


def heldout_split(c, share=0.25, seed=123):
    rng = np.random.default_rng(seed)
    linked = np.where(c.state >= 0)[0]
    threads = np.unique(c.thread[linked])
    test_threads = rng.choice(threads, int(round(share * len(threads))), replace=False)
    is_test = np.isin(c.thread, test_threads) & (c.state >= 0)
    return np.where((c.state >= 0) & ~is_test)[0], np.where(is_test)[0]


def placement(c, X, tr, te):
    """Assign each held-out record to the nearest training state centroid (cosine)."""
    Xn = l2n(X)
    S = np.zeros((len(c.states), X.shape[1])); n = np.zeros(len(c.states))
    np.add.at(S, c.state[tr], Xn[tr]); np.add.at(n, c.state[tr], 1)
    ok = n > 0
    S = l2n(S[ok] / n[ok][:, None]); ids = np.where(ok)[0]
    pred = ids[np.argmax(Xn[te] @ S.T, axis=1)]
    acct_of = np.array([c.accounts.index(a) for a, _ in c.states])
    return {"customer_acc": float((acct_of[pred] == c.account[te]).mean()),
            "state_acc": float((pred == c.state[te]).mean())}


def heldout_metrics(c, X, tr, te):
    """Neighbours are searched among held-out records only (other systems, other threads)."""
    q = np.random.default_rng(0).choice(len(te), min(3000, len(te)), replace=False)
    m = {"account_p@10_cross_system": knn_precision(X[te], c.account[te], c.system[te], c.thread[te], q),
         "state_p@10_cross_system": knn_precision(X[te], c.state[te], c.system[te], c.thread[te], q)}
    m.update(placement(c, X, tr, te))
    return m


def stage_heldout(c, res, arg):
    arms = (arg or "base,good,bad,ntp").split(",")
    tr, te = heldout_split(c)
    true_state = c.state.copy()
    out = res.setdefault("heldout", {"n_train_records": int(len(tr)), "n_test_records": int(len(te)),
                                     "thread_share": 0.25})
    if "base" in arms:
        from collections import Counter
        # references: knowing each record's customer and picking that customer's largest training state;
        # always the largest training state; always the largest customer
        best = {}
        for (a, s_), cnt in Counter((c.account[i], c.state[i]) for i in tr).items():
            if cnt > best.get(a, (None, -1))[1]:
                best[a] = (s_, cnt)
        out["reference_customer_known_state_acc"] = float(np.mean([best[c.account[i]][0] == c.state[i] for i in te]))
        out["reference_majority_state_acc"] = float(np.mean(c.state[te] == Counter(c.state[tr]).most_common(1)[0][0]))
        out["reference_majority_customer_acc"] = float(np.mean(c.account[te] ==
                                                               Counter(c.account[tr]).most_common(1)[0][0]))
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit([c.obs[i]["text"] for i in tr])
        svd = TruncatedSVD(256, random_state=0).fit(vec.transform([c.obs[i]["text"] for i in tr]))
        Xt = svd.transform(vec.transform([o["text"] for o in c.obs])).astype(np.float32)
        out["tfidf_lsa"] = heldout_metrics(c, Xt, tr, te); log("heldout tfidf", out["tfidf_lsa"])
        from jepa import Encoder
        torch.manual_seed(0)
        out["random_init"] = heldout_metrics(c, Encoder(len(c.vocab) + 1).embed(c.tok), tr, te)
        log("heldout random", out["random_init"])
        yield
    for arm in [a for a in arms if a != "base"]:
        c.state = true_state.copy(); c.state[te] = -1; build_pools(c)     # hidden: never a view or a prefix
        t = time.time()
        enc = train(c, arm, 0)
        c.state = true_state; build_pools(c)
        m = heldout_metrics(c, enc.embed(c.tok), tr, te)
        m["train_seconds"] = time.time() - t
        out[f"{arm}_s0"] = m
        log("heldout", arm, json.dumps(m))
        yield


if __name__ == "__main__":
    name, stages = sys.argv[1], sys.argv[2:]
    path = os.path.join(OUT, f"{name}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    c = load(D1)
    for st in stages:
        key, _, arg = st.partition(":")
        for _ in {"seeds": stage_seeds, "mask": stage_mask, "noise": stage_noise,
                  "heldout": stage_heldout}[key](c, res, arg):
            json.dump(res, open(path, "w"), indent=1)
    json.dump(res, open(path, "w"), indent=1)
    log("done", name, stages)
