"""Version-2 experiments, answering a review of version 1. Each stage writes results/v2_<stage>.json.

embed2      day-2 corpus embedded by the day-1 encoders (good, bad, next-token), for the analyses below
pretrained  a frozen pretrained backbone (all-MiniLM-L6-v2) with and without a good-view JEPA head:
            training-record metrics (as Extended Data Table 2) and held-out records (as Extended Data Table 3)
monthly     good views with customer-MONTH states (from scratch, a quarter of threads hidden), and the same with
            the pretrained backbone: held-out placement at month resolution
acl         access control with the pretrained backbone: each reader's head is trained only on records they may read
            (the backbone carries no company data); leak tests, cost, non-interference, and a public-records base
state       two channels: state vector (good views) plus detail vector (pretrained backbone), state tracking and
            next-quarter forecasting probes (as Appendix D)
time        monthly states: next-month forecasting and a linear dynamics predictor
primitives  operational primitives: drift detection against escalations and deal moves, cross-team misalignment,
            and situational nearest neighbours across customers
situations  similar situations across customers, with controls for time and a focus on customers in trouble
            (adds "situations_controls" to v2_primitives.json)
scope       why low-visibility readers lose accuracy: the same scratch encoder trained on a random sample of all
            records, as many as the reader may read (v2_scope.json)
acl_combo   a low-visibility reader's own two channels, scratch encoder and backbone head side by side (adds to
            v2_acl.json; needs run_acl.py's per-reader encoders)
acl_pub     the public-records base fine-tuned for 800 steps, as long as a head trained directly (adds to v2_acl.json)

Order: embed2 pretrained monthly acl state time primitives situations scope acl_combo acl_pub. The memory stores
(two channels concatenated and routed) are run by run_memory2.py with STORE=dual and STORE=route.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("THREADS", 1)))
sys.path.insert(0, os.path.dirname(__file__))
from data import build_pools, load, window_key  # noqa: E402
from jepa import Encoder, train_jepa  # noqa: E402
from memory import ANSWERS, _ts  # noqa: E402
from metrics import l2n  # noqa: E402
from pretrained import minilm_embed, train_head  # noqa: E402
from run_extras import heldout_metrics, heldout_split  # noqa: E402
from run_prh import D1, D2, OUT, e1_metrics  # noqa: E402
from run_state import loao  # noqa: E402

KINDS = ("tickets", "priority", "call", "stage")
SALES, SUPPORT = ("salesforce", "gong"), ("zendesk", "jira")


def save(name, res):
    json.dump(res, open(os.path.join(OUT, f"v2_{name}.json"), "w"), indent=1)


def corpora(window="quarter"):
    c1 = load(D1, window=window)
    c2 = load(D2, vocab=c1.vocab, accounts=c1.accounts, window=window)
    return c1, c2


def feats(name):
    return np.load(os.path.join(OUT, name))


# ------------------------------------------------------------------------------------------------ ground truth
def bounds(key, window):
    if window == "quarter":
        y, k = int(key[:4]), int(key[-1])
        lo = datetime(y, 3 * (k - 1) + 1, 1, tzinfo=timezone.utc)
        hi = datetime(y + (k == 4), (3 * k) % 12 + 1, 1, tzinfo=timezone.utc)
    elif window == "month":
        y, m = int(key[:4]), int(key[5:7])
        lo = datetime(y, m, 1, tzinfo=timezone.utc)
        hi = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
    else:
        y, w = int(key[:4]), int(key[6:])
        lo = datetime.fromisocalendar(y, w, 1).replace(tzinfo=timezone.utc)
        hi = lo + timedelta(days=7)
    return lo.timestamp(), hi.timestamp()


def next_key(key, window):
    lo, hi = bounds(key, window)
    return window_key(hi + 3600, window)


class Truth:
    """Era's ground truth per customer and window: tickets, worst priority, calls, deal stage, and dated events."""

    def __init__(self, gt):
        org2acct = {o["id"]: o["external_id"] for o in gt["zendesk_orgs"]}
        self.tickets = [(org2acct.get(t["organization_id"]), _ts(t["created_at"]), (t.get("priority") or "normal").lower())
                        for t in gt["tickets"]]
        self.calls = [(c["account_id"], _ts(c["started"])) for c in gt["calls"]]
        self.hist = defaultdict(list)
        for h in gt["opportunity_history"]:
            self.hist[h["OpportunityId"]].append((_ts(h["CreatedDate"]), h["StageName"]))
        self.opps = defaultdict(list)
        self.opp_acct = {}
        for o in gt["opportunities"]:
            self.opps[o["AccountId"]].append((_ts(o["CreatedDate"]), o["Id"]))
            self.opp_acct[o["Id"]] = o["AccountId"]

    def facts(self, acct, lo, hi):
        rank = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
        tk = [p for a, t, p in self.tickets if a == acct and t is not None and lo <= t < hi]
        n = len(tk)
        out = {"tickets": "none" if n == 0 else "one or two" if n <= 2 else "three to five" if n <= 5 else "six or more",
               "call": "yes" if any(a == acct and t is not None and lo <= t < hi for a, t in self.calls) else "no",
               "priority": "no tickets" if n == 0 else max(tk, key=lambda p: rank.get(p, 1))}
        stage = "no open deal"
        live = sorted([(cd, i) for cd, i in self.opps.get(acct, []) if cd is not None and cd < hi])
        if live:
            ev = sorted([(t, s) for t, s in self.hist.get(live[-1][1], []) if t is not None and t < hi])
            if ev:
                stage = ev[-1][1].lower()
        out["stage"] = stage if stage in ANSWERS["stage"] else "no open deal"
        return out

    def events(self, acct):
        """Dated events for a customer: urgent or high tickets, and deal stage changes."""
        ev = [("ticket_" + p, t) for a, t, p in self.tickets if a == acct and t is not None and p in ("urgent", "high")]
        for oid, rows in self.hist.items():
            if self.opp_acct.get(oid) != acct:
                continue
            rows = sorted(r for r in rows if r[0] is not None)
            for (t0, s0), (t1, s1) in zip(rows, rows[1:]):
                if s1 != s0:
                    ev.append(("stage_" + s1.lower().replace(" ", "_"), t1))
        return ev


def state_matrix(c, X, min_records=1):
    Xn = l2n(X)
    S = np.zeros((len(c.states), X.shape[1]), np.float32)
    n = np.zeros(len(c.states), int)
    for s in range(len(c.states)):
        idx = c.pools.get(s)
        if idx is not None and len(idx) >= min_records:
            S[s] = Xn[idx].mean(0)
            n[s] = len(idx)
    return S, n


def zscore(S):
    return (S - S.mean(0)) / (S.std(0) + 1e-6)


def two_channel(*mats):
    """Concatenate channels, each standardized and scaled to equal total variance."""
    return np.concatenate([zscore(m) / np.sqrt(m.shape[1]) for m in mats], 1)


# ------------------------------------------------------------------------------------------------ stages
def stage_embed2():
    c1, c2 = corpora()
    for arm in ("good", "bad", "ntp"):
        enc = Encoder(len(c1.vocab) + 1)
        enc.load_state_dict(torch.load(os.path.join(OUT, f"enc_{arm}_s0.pt")))
        np.save(os.path.join(OUT, f"emb2_{arm}.npy"), enc.embed(c2.tok))
        print("embed2", arm, flush=True)


def stage_pretrained():
    res = {}
    c1, _ = corpora()
    F1 = feats("minilm_day1.npy")
    res["minilm_raw"] = e1_metrics(c1, F1, np.random.default_rng(0))
    t = time.time()
    head = train_head(F1, c1, seed=0)
    H1 = head.embed(F1)
    res["minilm_head"] = e1_metrics(c1, H1, np.random.default_rng(0))
    res["minilm_head"]["train_seconds"] = time.time() - t
    np.save(os.path.join(OUT, "emb_minilm_head.npy"), H1)
    np.save(os.path.join(OUT, "emb2_minilm_head.npy"), head.embed(feats("minilm_day2.npy")))
    print("E1-style", json.dumps(res), flush=True)
    tr, te = heldout_split(c1)
    true_state = c1.state.copy()
    c1.state[te] = -1; build_pools(c1)
    head_ho = train_head(F1, c1, seed=0)
    c1.state = true_state; build_pools(c1)
    res["heldout"] = {"minilm_raw": heldout_metrics(c1, F1, tr, te),
                      "minilm_head": heldout_metrics(c1, head_ho.embed(F1), tr, te)}
    print("held-out", json.dumps(res["heldout"]), flush=True)
    save("pretrained", res)


def stage_monthly():
    """Good views with customer-month states, a quarter of the threads hidden; held-out placement by month."""
    res = {}
    c, c2 = corpora("month")
    tr, te = heldout_split(c)                       # same threads as the quarter experiments
    true_state = c.state.copy()
    best = {}
    for (a, s_), n in Counter((c.account[i], c.state[i]) for i in tr).items():
        if n > best.get(a, (None, -1))[1]:
            best[a] = (s_, n)
    res["n_states"] = len(c.states)
    res["reference_customer_known_month_acc"] = float(np.mean([best[c.account[i]][0] == c.state[i] for i in te]))
    c.state = true_state.copy(); c.state[te] = -1; build_pools(c)
    t = time.time()
    enc, _ = train_jepa(c, "good", steps=1500, seed=0)
    secs = time.time() - t
    c.state = true_state.copy(); build_pools(c)
    X = enc.embed(c.tok)
    np.save(os.path.join(OUT, "emb_good_month_ho.npy"), X)
    np.save(os.path.join(OUT, "emb2_good_month.npy"), enc.embed(c2.tok))
    res["good_month"] = heldout_metrics(c, X, tr, te); res["good_month"]["train_seconds"] = secs
    # the quarter-trained encoders, evaluated at month resolution (they saw every thread: an upper reference)
    res["good_quarter_all_threads"] = heldout_metrics(c, np.load(os.path.join(OUT, "emb_good_s0.npy")), tr, te)
    F1 = feats("minilm_day1.npy")
    res["minilm_raw"] = heldout_metrics(c, F1, tr, te)
    c.state = true_state.copy(); c.state[te] = -1; build_pools(c)
    head = train_head(F1, c, seed=0)
    c.state = true_state.copy(); build_pools(c)
    res["minilm_head_month"] = heldout_metrics(c, head.embed(F1), tr, te)
    np.save(os.path.join(OUT, "emb2_minilm_head_month.npy"), head.embed(feats("minilm_day2.npy")))
    print(json.dumps(res), flush=True)
    save("monthly", res)


def stage_acl():
    """Access control with a pretrained backbone: heads trained per reader on readable records only."""
    from acl import READERS
    from run_acl import CODENAMES, SECRET, canary_text, leak_metrics, placement, setup, visible
    c, readers, is_test, top, plan = setup()
    F_clean = feats("minilm_day1.npy")
    F_canary = F_clean.copy()
    ids = sorted(plan)
    F_canary[ids] = minilm_embed([canary_text(c.obs[i], plan[i]) for i in ids])
    Qf = minilm_embed(CODENAMES + [SECRET])
    true_state = c.state.copy()
    everyone = np.array([len(r) == 12 for r in readers])           # records every employee may read

    def head_for(mask, F, init=None, steps=800):
        c.state = true_state.copy(); c.state[~mask | is_test] = -1; build_pools(c)
        h = train_head(F, c, seed=0, init=init, steps=steps)
        c.state = true_state; build_pools(c)
        return h

    res = {"public_records": int(everyone.sum()), "readers": {}}
    allrows = np.arange(len(c.obs))
    open_h = head_for(np.ones(len(c.obs), bool), F_canary)
    open_clean_h = head_for(np.ones(len(c.obs), bool), F_clean)
    Xo, Qo = open_h.embed(F_canary), open_h.embed(Qf)
    Xoc, Qoc = open_clean_h.embed(F_clean), open_clean_h.embed(Qf)
    pub = head_for(everyone, F_canary)                             # public base: records every employee may read
    for name, user in READERS.items():
        vis = visible(readers, user)
        rows = np.where(vis)[0]
        own = head_for(vis, F_canary) if name != "ceo" else open_h
        Xr = own.embed(F_canary)
        r = {"visible_share": float(vis.mean()),
             "placement_open_head": placement(Xo[rows], rows, c, is_test),
             "placement_own_head": placement(Xr[rows], rows, c, is_test),
             "placement_raw_backbone": placement(F_canary[rows], rows, c, is_test)}
        if name in ("software_engineer", "financial_analyst"):
            import copy
            ft = head_for(vis, F_canary, init=copy.deepcopy(pub), steps=400)
            r["placement_public_base_finetuned"] = placement(ft.embed(F_canary)[rows], rows, c, is_test)
        if name in ("support_engineer", "software_engineer"):
            own_clean = head_for(vis, F_clean)
            Xrc = own_clean.embed(F_clean)
            leak = {}
            for conf, w, wo in (("open", (Xo, Qo, allrows), (Xoc, Qoc, allrows)),
                                ("filtered", (Xo[rows], Qo, rows), (Xoc[rows], Qoc, rows)),
                                ("built_in", (Xr[rows], own.embed(Qf), rows), (Xrc[rows], own_clean.embed(Qf), rows))):
                m = {"with": leak_metrics(*w, c, top), "without": leak_metrics(*wo, c, top)}
                m["codename_advantage"] = m["with"]["codename_precision"] - m["without"]["codename_precision"]
                m["secret_advantage"] = m["with"]["secret_precision@5"] - m["without"]["secret_precision@5"]
                leak[conf] = m
            leak["non_interference_max_abs_diff"] = float(np.abs(Xr[rows] - Xrc[rows]).max())
            r["leak"] = leak
        res["readers"][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "leak"}), flush=True)
        if "leak" in r:
            print("  leak", json.dumps({k: (v["codename_advantage"], v["secret_advantage"]) if isinstance(v, dict) else v
                                        for k, v in r["leak"].items()}), flush=True)
    save("acl", res)


def stage_state():
    """Two channels for state tracking and forecasting: good-view state plus pretrained detail."""
    import run_state as rs
    from memory import make_questions
    c1, c2 = corpora()
    gt2 = json.load(open(D2.replace("observations.jsonl", "ground_truth.json")))
    facts = {}
    for x in make_questions(gt2, c2.states):
        facts.setdefault((x["account"], x["quarter"]), {})[x["kind"]] = x["answer"]
    reps = {a: state_matrix(c2, feats(f"emb2_{a}.npy"))[0] for a in ("good", "bad", "ntp")}
    reps["minilm"] = state_matrix(c2, feats("minilm_day2.npy"))[0]
    reps["minilm_head"] = state_matrix(c2, feats("emb2_minilm_head.npy"))[0]
    prov = np.zeros((len(c2.states), len(c2.systems)), np.float32)
    for s in range(len(c2.states)):
        idx = c2.pools.get(s)
        if idx is not None:
            prov[s] = np.bincount(c2.system[idx], minlength=len(c2.systems)) / len(idx)
    reps["good+minilm"] = two_channel(reps["good"], reps["minilm"])
    reps["good+bad"] = two_channel(reps["good"], reps["bad"])
    reps["good+provenance"] = two_channel(reps["good"], prov)
    sid = {s: i for i, s in enumerate(c2.states)}
    now_rows = [i for i, s in enumerate(c2.states) if s in facts]
    trans = [(sid[s], sid[(s[0], rs.next_q(s[1]))]) for s in c2.states
             if (s[0], rs.next_q(s[1])) in sid and s in facts and (s[0], rs.next_q(s[1])) in facts]
    res = {}
    for name, S in reps.items():
        out = {}
        for kind in KINDS:
            rows = [i for i in now_rows if kind in facts[c2.states[i]]]
            y = np.array([facts[c2.states[i]][kind] for i in rows]); g = np.array([c2.states[i][0] for i in rows])
            acc, _, _ = loao(S[rows], y, g)
            a_ = [a for a, b in trans]; b_ = [b for a, b in trans]
            yb = np.array([facts[c2.states[b]][kind] for b in b_]); gb = np.array([c2.states[a][0] for a in a_])
            acc_n, _, _ = loao(S[a_], yb, gb)
            out[kind] = {"now": acc, "next": acc_n}
        out["mean_now"] = float(np.mean([out[k]["now"] for k in KINDS]))
        out["mean_next"] = float(np.mean([out[k]["next"] for k in KINDS]))
        res[name] = out
        print(name, json.dumps(out), flush=True)
    save("state", res)


def window_facts(truth, c, window):
    out = {}
    for (a, key) in c.states:
        lo, hi = bounds(key, window)
        out[(a, key)] = truth.facts(a, lo, hi)
    return out


ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def stage_time():
    """Monthly states: next-month forecasting and a linear dynamics predictor (day-2 corpus, day-1 encoders)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    _, c2 = corpora("month")
    truth = Truth(json.load(open(D2.replace("observations.jsonl", "ground_truth.json"))))
    facts = window_facts(truth, c2, "month")
    reps = {"good_month": state_matrix(c2, feats("emb2_good_month.npy")),
            "good_quarter": state_matrix(c2, feats("emb2_good.npy")),
            "bad": state_matrix(c2, feats("emb2_bad.npy")),
            "minilm": state_matrix(c2, feats("minilm_day2.npy")),
            "minilm_head_month": state_matrix(c2, feats("emb2_minilm_head_month.npy"))}
    reps["good_month+minilm"] = (two_channel(reps["good_month"][0], reps["minilm"][0]), reps["good_month"][1])
    sid = {s: i for i, s in enumerate(c2.states)}
    trans = [(sid[s], sid[(s[0], next_key(s[1], "month"))]) for s in c2.states
             if (s[0], next_key(s[1], "month")) in sid]
    res = {"n_states": len(c2.states), "n_transitions": len(trans)}
    a_ = np.array([a for a, b in trans]); b_ = np.array([b for a, b in trans])
    g = np.array([c2.states[a][0] for a in a_])
    for kind in KINDS:
        yb = np.array([facts[c2.states[b]][kind] for b in b_]); ya = np.array([facts[c2.states[a]][kind] for a in a_])
        res.setdefault("baselines", {})[kind] = {"majority": Counter(yb).most_common(1)[0][1] / len(yb),
                                                 "oracle_persistence": float((ya == yb).mean())}
    for name, (S, n) in reps.items():
        out = {}
        for kind in KINDS:
            yb = np.array([facts[c2.states[b]][kind] for b in b_])
            out[kind] = loao(S[a_], yb, g)[0]
        out["mean_next"] = float(np.mean([out[k] for k in KINDS]))
        # linear dynamics in the state space, leave one customer out: predict next month's state from this month's
        Z = zscore(S)
        cos_pers, cos_pred, cos_dir, err_pers, err_pred, err_shrink, shrink = [], [], [], [], [], [], []
        err_tuned, alphas = [], []
        for cust in np.unique(g):
            trm, tem = g != cust, g == cust
            pca = PCA(32, random_state=0).fit(Z[a_[trm]])
            A, B = pca.transform(Z[a_]), pca.transform(Z[b_])
            reg = Ridge(alpha=10.0).fit(A[trm], B[trm] - A[trm])     # residual model: next = this + learned change
            P = A[tem] + reg.predict(A[tem])
            k = float((A[trm] * B[trm]).sum() / (A[trm] * A[trm]).sum())   # control: shrink toward the mean state
            err_shrink.extend(((B[tem] - k * A[tem]) ** 2).sum(1).tolist()); shrink.append(k)
            inner = {}                                                    # ridge strength by inner leave-one-out
            for al in ALPHAS:
                e = 0.0
                for c_in in np.unique(g[trm]):
                    fit, val = trm & (g != c_in), g == c_in
                    r_in = Ridge(alpha=al).fit(A[fit], B[fit] - A[fit])
                    e += float(((B[val] - A[val] - r_in.predict(A[val])) ** 2).sum())
                inner[al] = e
            best = min(inner, key=inner.get); alphas.append(best)
            r_t = Ridge(alpha=best).fit(A[trm], B[trm] - A[trm])
            err_tuned.extend(((B[tem] - A[tem] - r_t.predict(A[tem])) ** 2).sum(1).tolist())
            for p, x0, x1 in zip(P, A[tem], B[tem]):
                cos_pers.append(float(x0 @ x1 / (np.linalg.norm(x0) * np.linalg.norm(x1) + 1e-9)))
                cos_pred.append(float(p @ x1 / (np.linalg.norm(p) * np.linalg.norm(x1) + 1e-9)))
                d_p, d_a = p - x0, x1 - x0
                cos_dir.append(float(d_p @ d_a / (np.linalg.norm(d_p) * np.linalg.norm(d_a) + 1e-9)))
                err_pers.append(float(((x1 - x0) ** 2).sum())); err_pred.append(float(((x1 - p) ** 2).sum()))
        out["dynamics"] = {"cos_persistence": float(np.mean(cos_pers)), "cos_predicted": float(np.mean(cos_pred)),
                           "cos_change_direction": float(np.mean(cos_dir)),
                           "error_reduction_vs_persistence": float(1 - np.sum(err_pred) / np.sum(err_pers)),
                           "shrink_error_reduction_vs_persistence": float(1 - np.sum(err_shrink) / np.sum(err_pers)),
                           "shrink_factor": float(np.mean(shrink)),
                           "tuned_error_reduction_vs_persistence": float(1 - np.sum(err_tuned) / np.sum(err_pers)),
                           "tuned_alphas": {str(a): alphas.count(a) for a in sorted(set(alphas))}}
        res[name] = out
        print(name, json.dumps(out), flush=True)
    save("time", res)


def stage_primitives():
    from sklearn.metrics import roc_auc_score
    c1, c2 = corpora()
    gt2 = json.load(open(D2.replace("observations.jsonl", "ground_truth.json")))
    truth = Truth(gt2)
    reps = {"good": feats("emb2_good.npy"), "good_month": feats("emb2_good_month.npy"), "bad": feats("emb2_bad.npy"),
            "minilm": feats("minilm_day2.npy"), "minilm_head": feats("emb2_minilm_head.npy")}
    reps["good+minilm"] = np.concatenate([l2n(reps["good"]), l2n(reps["minilm"])], 1)
    res = {}
    # ---------------- drift: weekly states per customer, distance from the previous four weeks
    wk = np.array([window_key(t, "week") for t in c2.ts])
    acct = c2.account
    weeks_all = sorted(set(wk))
    widx = {w: i for i, w in enumerate(weeks_all)}
    events = defaultdict(set)
    for a_i, a in enumerate(c2.accounts):
        for kind, t in truth.events(a):
            events[(a_i, window_key(t, "week"))].add(kind)
    drift = {}
    for name, X in list(reps.items()) + [("volume", None)]:
        scores, labels, lead = [], [], []
        for a_i in range(len(c2.accounts)):
            m = acct == a_i
            if m.sum() == 0:
                continue
            by_w = defaultdict(list)
            for i in np.where(m)[0]:
                by_w[widx[wk[i]]].append(i)
            series = {}
            for w, ids in by_w.items():
                if len(ids) >= 3:
                    series[w] = (l2n(X[ids]).mean(0) if X is not None else None, len(ids))
            ws = sorted(series)
            for j, w in enumerate(ws):
                prev = [p for p in ws[:j] if w - 8 <= p < w][-4:]
                if not prev:
                    continue
                if X is None:
                    cur, base = series[w][1], np.mean([series[p][1] for p in prev])
                    score = abs(np.log((cur + 1) / (base + 1)))
                else:
                    base = np.mean([series[p][0] for p in prev], 0)
                    v = series[w][0]
                    score = 1 - float(v @ base / (np.linalg.norm(v) * np.linalg.norm(base) + 1e-9))
                key = (a_i, weeks_all[w])
                scores.append(score); labels.append(int(bool(events.get(key))))
                nxt = (a_i, weeks_all[w + 1]) if w + 1 < len(weeks_all) else None
                lead.append(int(bool(events.get(nxt))) if nxt else 0)
        drift[name] = {"auroc_event_week": float(roc_auc_score(labels, scores)),
                       "auroc_next_week": float(roc_auc_score(lead, scores)),
                       "weeks": len(scores), "event_weeks": int(sum(labels))}
        print("drift", name, json.dumps(drift[name]), flush=True)
    res["drift"] = drift
    # ---------------- cross-team misalignment per customer-month: sales view vs support view
    mon = np.array([window_key(t, "month") for t in c2.ts])
    sysname = np.array([c2.obs[i]["system"] for i in range(len(c2.obs))])
    mis = {}
    keys = sorted({(acct[i], mon[i]) for i in range(len(c2.obs)) if acct[i] >= 0})
    label = {}
    for a_i, mk in keys:
        lo, hi = bounds(mk, "month")
        f = truth.facts(c2.accounts[a_i], lo, hi)
        hot = f["priority"] in ("urgent", "high")
        open_deal = f["stage"] in ("prospecting", "qualification", "proposal", "negotiation")
        label[(a_i, mk)] = int(hot and open_deal)
    groups = defaultdict(list)
    for i in range(len(c2.obs)):
        if acct[i] >= 0:
            side = "sales" if sysname[i] in SALES else "support" if sysname[i] in SUPPORT else None
            if side:
                groups[(acct[i], mon[i], side)].append(i)
    for name, X in reps.items():
        sc, lb = [], []
        for (a_i, mk) in keys:
            s_ids, p_ids = groups.get((a_i, mk, "sales"), []), groups.get((a_i, mk, "support"), [])
            if len(s_ids) >= 2 and len(p_ids) >= 2:
                u, v = l2n(X[s_ids]).mean(0), l2n(X[p_ids]).mean(0)
                sc.append(1 - float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
                lb.append(label[(a_i, mk)])
        mis[name] = {"auroc": float(roc_auc_score(lb, sc)) if 0 < sum(lb) < len(lb) else None,
                     "windows": len(sc), "at_risk": int(sum(lb))}
        print("misalignment", name, json.dumps(mis[name]), flush=True)
    res["misalignment"] = mis
    # ---------------- situational neighbours across customers (customer-quarter states)
    from memory import make_questions
    facts = {}
    for x in make_questions(gt2, c2.states):
        facts.setdefault((x["account"], x["quarter"]), {})[x["kind"]] = x["answer"]

    def situation(f):
        stage = f["stage"]
        grp = "open" if stage in ("prospecting", "qualification", "proposal", "negotiation") else stage
        return (f["tickets"], grp)

    rows = [i for i, s in enumerate(c2.states) if s in facts and len(c2.pools.get(i, [])) >= 5]
    lab = [situation(facts[c2.states[i]]) for i in rows]
    cust = np.array([c2.states[i][0] for i in rows])
    knn = {}
    for name, X in reps.items():
        S, _ = state_matrix(c2, X)
        Z = l2n(S[rows] - S[rows].mean(0))
        sim = Z @ Z.T
        prec, chance = [], []
        for j in range(len(rows)):
            other = np.where(cust != cust[j])[0]
            nn = other[np.argsort(-sim[j, other])[:5]]
            prec.append(np.mean([lab[k] == lab[j] for k in nn]))
            chance.append(np.mean([lab[k] == lab[j] for k in other]))
        knn[name] = {"precision@5": float(np.mean(prec)), "chance": float(np.mean(chance)), "states": len(rows)}
        print("situations", name, json.dumps(knn[name]), flush=True)
    res["situations"] = knn
    save("primitives", res)


def stage_situations():
    """Similar situations across customers, with controls for time and a focus on customers in trouble."""
    from memory import make_questions
    c1, c2 = corpora()
    gt2 = json.load(open(D2.replace("observations.jsonl", "ground_truth.json")))
    reps = {"good": feats("emb2_good.npy"), "good_month": feats("emb2_good_month.npy"), "bad": feats("emb2_bad.npy"),
            "minilm": feats("minilm_day2.npy"), "minilm_head": feats("emb2_minilm_head.npy")}
    reps["good+minilm"] = np.concatenate([l2n(reps["good"]), l2n(reps["minilm"])], 1)
    facts = {}
    for x in make_questions(gt2, c2.states):
        facts.setdefault((x["account"], x["quarter"]), {})[x["kind"]] = x["answer"]
    OPEN = ("prospecting", "qualification", "proposal", "negotiation")

    def situation(f):
        return (f["tickets"], "open" if f["stage"] in OPEN else f["stage"])

    def trouble(f):
        return f["priority"] in ("urgent", "high") and f["stage"] in OPEN

    rows = [i for i, s in enumerate(c2.states) if s in facts and len(c2.pools.get(i, [])) >= 5]
    lab = [situation(facts[c2.states[i]]) for i in rows]
    hot = np.array([trouble(facts[c2.states[i]]) for i in rows])
    cust = np.array([c2.states[i][0] for i in rows])
    qtr = np.array([c2.states[i][1] for i in rows])
    same = np.array([[a == b for b in lab] for a in lab])
    out = {"states": len(rows), "trouble_states": int(hot.sum())}
    sq = [same[j, (cust != cust[j]) & (qtr == qtr[j])].mean() for j in range(len(rows))
          if ((cust != cust[j]) & (qtr == qtr[j])).any()]
    out["same_quarter_baseline"] = float(np.mean(sq))
    for name, X in reps.items():
        S, _ = state_matrix(c2, X)
        Z = l2n(S[rows] - S[rows].mean(0))
        sim = Z @ Z.T
        r = {}
        for tag, ok in (("all", lambda j: cust != cust[j]), ("other_quarters", lambda j: (cust != cust[j]) & (qtr != qtr[j]))):
            prec, chance = [], []
            for j in range(len(rows)):
                other = np.where(ok(j))[0]
                nn = other[np.argsort(-sim[j, other])[:5]]
                prec.append(same[j, nn].mean()); chance.append(same[j, other].mean())
            r[f"precision@5_{tag}"] = float(np.mean(prec)); r[f"chance_{tag}"] = float(np.mean(chance))
        tp, tb = [], []
        for j in np.where(hot)[0]:
            other = np.where(cust != cust[j])[0]
            nn = other[np.argsort(-sim[j, other])[:5]]
            tp.append(hot[nn].mean()); tb.append(hot[other].mean())
        r["trouble_precision@5"] = float(np.mean(tp)); r["trouble_base_rate"] = float(np.mean(tb))
        out[name] = r
        print("situations", name, json.dumps(r), flush=True)
    path = os.path.join(OUT, "v2_primitives.json")
    res = json.load(open(path)); res["situations_controls"] = out
    json.dump(res, open(path, "w"), indent=1)


def stage_scope():
    """Why low-visibility readers lose accuracy: too few records, or the wrong kind? Train the same scratch encoder on a
    random sample of all records, as many as the reader may read, and place the reader's held-out readable records."""
    from acl import READERS
    from data import tokenize
    from run_acl import placement, setup, visible
    c, readers, is_test, top, plan = setup()
    true_state = c.state.copy()
    res = {}
    for name in ("software_engineer", "financial_analyst"):
        vis = visible(readers, READERS[name])
        rows = np.where(vis)[0]
        rng = np.random.default_rng(11)
        idx = np.sort(rng.choice(len(c.obs), int(vis.sum()), replace=False))
        path = os.path.join(OUT, f"scope_corpus_{name}.jsonl")
        with open(path, "w") as f:
            for i in idx:
                f.write(json.dumps(c.obs[i], sort_keys=True) + "\n")
        cs = load(path)
        os.remove(path)
        ts = cs.state.copy(); cs.state[is_test[idx]] = -1; build_pools(cs)
        enc, _ = train_jepa(cs, "good", steps=1500, seed=0)
        cs.state = ts
        tok = [np.array([cs.vocab.get(t, 0) for t in tokenize(c.obs[i]["text"])][:96], dtype=np.int64) for i in rows]
        Xr = enc.embed(tok)
        sys_all = np.array([c.obs[i]["system"] for i in range(len(c.obs))])
        res[name] = {"records": int(vis.sum()),
                     "reader_systems": {k: int(v) for k, v in zip(*np.unique(sys_all[rows], return_counts=True))},
                     "random_scope_systems": {k: int(v) for k, v in zip(*np.unique(sys_all[idx], return_counts=True))},
                     "placement_random_scope_same_size": placement(Xr, rows, c, is_test)}
        print(name, json.dumps(res[name]), flush=True)
    c.state = true_state
    save("scope", res)


def stage_acl_combo():
    """A reader's own two channels: their scratch encoder and their backbone head, both trained on readable records."""
    from acl import READERS
    from run_acl import canary_text, placement, setup, visible
    c, readers, is_test, top, plan = setup()
    F = feats("minilm_day1.npy").copy()
    ids = sorted(plan)
    F[ids] = minilm_embed([canary_text(c.obs[i], plan[i]) for i in ids])
    true_state = c.state.copy()
    path = os.path.join(OUT, "v2_acl.json")
    res = json.load(open(path))
    for name in ("software_engineer", "financial_analyst"):
        vis = visible(readers, READERS[name])
        rows = np.where(vis)[0]
        z = np.load(os.path.join(OUT, f"acl_{name}.npz"))
        assert np.array_equal(z["idx"], rows)
        c.state = true_state.copy(); c.state[~vis | is_test] = -1; build_pools(c)
        h = train_head(F, c, seed=0)
        c.state = true_state; build_pools(c)
        Xh = h.embed(F[rows])
        both = np.concatenate([l2n(z["X"]), l2n(Xh)], 1)
        r = res["readers"][name]
        r["placement_own_scratch"] = placement(z["X"], rows, c, is_test)
        r["placement_own_head_rerun"] = placement(Xh, rows, c, is_test)
        r["placement_own_two_channels"] = placement(both, rows, c, is_test)
        print(name, json.dumps({k: r[k]["customer_acc"] for k in ("placement_own_scratch", "placement_own_head_rerun",
                                                                  "placement_own_two_channels")}), flush=True)
    json.dump(res, open(path, "w"), indent=1)


def stage_acl_pub():
    """The public-records base, fine-tuned for 800 steps (as many as a head trained directly on the reader's records)."""
    import copy
    from acl import READERS
    from run_acl import canary_text, placement, setup, visible
    c, readers, is_test, top, plan = setup()
    F = feats("minilm_day1.npy").copy()
    ids = sorted(plan)
    F[ids] = minilm_embed([canary_text(c.obs[i], plan[i]) for i in ids])
    true_state = c.state.copy()
    everyone = np.array([len(r) == 12 for r in readers])

    def head_for(mask, init=None, steps=800):
        c.state = true_state.copy(); c.state[~mask | is_test] = -1; build_pools(c)
        h = train_head(F, c, seed=0, init=init, steps=steps)
        c.state = true_state; build_pools(c)
        return h

    pub = head_for(everyone)
    path = os.path.join(OUT, "v2_acl.json")
    res = json.load(open(path))
    for name in ("software_engineer", "financial_analyst"):
        vis = visible(readers, READERS[name])
        rows = np.where(vis)[0]
        ft = head_for(vis, init=copy.deepcopy(pub), steps=800)
        r = res["readers"][name]
        r["placement_public_base_finetuned_800"] = placement(ft.embed(F[rows]), rows, c, is_test)
        r["placement_public_base_only"] = placement(pub.embed(F[rows]), rows, c, is_test)
        print(name, json.dumps({k: r[k]["customer_acc"] for k in ("placement_public_base_finetuned_800",
                                                                  "placement_public_base_only")}), flush=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        {"embed2": stage_embed2, "pretrained": stage_pretrained, "monthly": stage_monthly, "acl": stage_acl,
         "state": stage_state, "time": stage_time, "primitives": stage_primitives,
         "situations": stage_situations, "scope": stage_scope,
         "acl_combo": stage_acl_combo, "acl_pub": stage_acl_pub}[st]()
