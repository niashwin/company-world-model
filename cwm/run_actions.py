"""Actions in the world model: what the company did with each customer, and what followed.

A world model's predictor is action-conditioned, s_{t+1} = h(s_t, a_t). Era's records already say what the company
did each month. We read six actions per customer and month from the exported records and Era's ground truth: a sales
call held, a leader (the CEO or the head of engineering) writing to or about the customer, a CRM task logged, an
engineering escalation (a Jira issue), support agents replying on tickets, and employees emailing the customer.
Outcomes come from the ground truth: a deal advancing or being won, and an urgent or high-priority ticket.

Stages (each adds a section to results/actions.json):
  table      actions and outcomes per customer-month, and how often each action is followed by each outcome
  dynamics   next month's state from this month's state, with and without this month's actions, against no change
             and shrinkage to the mean; controls with shuffled actions, with this or next month's mix of systems,
             and with next month's actions; how well the state predicts each action; and next month's outcomes
             from state and actions
  events     the state in the 14 days before and after a sales call, an engineering escalation or a leader's
             involvement, against matched times without one
  simulate   decision simulation for customers in trouble (an urgent or high ticket during an open deal): what
             followed similar states of other customers when the company intervened and when it did not
             (case-based, in the state space), and what an action-conditioned outcome model predicts
  checks     transitions whose next month comes after the day-2 export, where outcomes are zero by construction

All analyses are observational: the company's actions were not randomized, so what they estimate is what tends to
follow an action in a given state, not its causal effect.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import window_key  # noqa: E402
from memory import _ts  # noqa: E402
from metrics import l2n  # noqa: E402
from run_prh import D2, OUT  # noqa: E402
from run_v2 import bounds, corpora, feats, next_key, state_matrix, two_channel, zscore  # noqa: E402

ACTIONS = ("call", "leader", "crm_task", "eng_escalation", "support_reply", "outbound_email")
LEADERS = {"user-0", "user-1"}                       # the CEO and the head of engineering
OPEN = ("prospecting", "qualification", "proposal", "negotiation")
ORDER = {"prospecting": 0, "qualification": 1, "proposal": 2, "negotiation": 3, "closed won": 4}
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
PATH = os.path.join(OUT, "actions.json")
DAY = 86400.0


def save(key, value):
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    res[key] = value
    json.dump(res, open(PATH, "w"), indent=1)


class World:
    """Month states of the day-2 corpus, the company's actions and the outcomes, per customer and month."""

    def __init__(self):
        _, self.c = corpora("month")
        c = self.c
        self.gt = json.load(open(D2.replace("observations.jsonl", "ground_truth.json")))
        gt = self.gt
        self.acc = list(c.accounts)
        emails = {p["email"].lower(): p["id"] for p in gt["people"]["people"]}
        commenters = defaultdict(set)
        for i, o in enumerate(c.obs):
            if o["system"] == "zendesk" and o.get("kind") == "ticket_comment" and c.account[i] >= 0:
                commenters[str(o.get("author"))].add(c.account[i])
        agents = {a for a, s in commenters.items() if len(s) >= 3}       # agents answer many customers
        self.n_agents = len(agents)
        self.month = [window_key(t, "month") for t in c.ts]
        self.author = []
        A = defaultdict(Counter)
        for i, o in enumerate(c.obs):
            raw = str(o.get("author") or "")
            m = re.search(r"<([^>]+)>", raw)
            who = emails.get((m.group(1) if m else raw).strip().lower(), raw if raw.startswith("user-") else None)
            self.author.append(who)
            if c.account[i] < 0:
                continue
            k, s, kind = (c.account[i], self.month[i]), o["system"], o.get("kind")
            if s == "salesforce" and kind == "task":
                A[k]["crm_task"] += 1
            if s == "jira":
                A[k]["eng_escalation"] += 1
            if s == "zendesk" and kind == "ticket_comment" and str(o.get("author")) in agents:
                A[k]["support_reply"] += 1
            if s == "gmail" and who and who.startswith("user-"):
                A[k]["outbound_email"] += 1
            if s in ("gmail", "slack") and who in LEADERS:
                A[k]["leader"] += 1
        self.call_times = defaultdict(list)
        for cl in gt["calls"]:
            t = _ts(cl["started"])
            if t is not None and cl["account_id"] in self.acc:
                a = self.acc.index(cl["account_id"])
                A[(a, window_key(t, "month"))]["call"] += 1
                self.call_times[a].append(t)
        self.A = A
        org2acct = {o["id"]: o["external_id"] for o in gt["zendesk_orgs"]}
        self.esc = Counter()
        for t in gt["tickets"]:
            a, ts = org2acct.get(t["organization_id"]), _ts(t["created_at"])
            if a in self.acc and ts and (t.get("priority") or "normal").lower() in ("urgent", "high"):
                self.esc[(self.acc.index(a), window_key(ts, "month"))] += 1
        opp_acct = {o["Id"]: o["AccountId"] for o in gt["opportunities"]}
        hist = defaultdict(list)
        for h in gt["opportunity_history"]:
            t = _ts(h["CreatedDate"])
            if t:
                hist[h["OpportunityId"]].append((t, h["StageName"].lower()))
        self.adv, self.lost, self.stage_at = Counter(), Counter(), {}
        self.hist = {k: sorted(v) for k, v in hist.items()}
        self.opp_of = defaultdict(list)
        for oid, a in opp_acct.items():
            if a in self.acc:
                self.opp_of[self.acc.index(a)].append(oid)
        for oid, rows in self.hist.items():
            a = opp_acct.get(oid)
            if a not in self.acc:
                continue
            for (t0, s0), (t1, s1) in zip(rows, rows[1:]):
                k = (self.acc.index(a), window_key(t1, "month"))
                if s1 == "closed lost":
                    self.lost[k] += 1
                elif ORDER.get(s1, -1) > ORDER.get(s0, -1):
                    self.adv[k] += 1
        self.esc_times = defaultdict(list)
        for t in gt["tickets"]:
            a, ts_ = org2acct.get(t["organization_id"]), _ts(t["created_at"])
            if a in self.acc and ts_ and (t.get("priority") or "normal").lower() in ("urgent", "high"):
                self.esc_times[self.acc.index(a)].append(ts_)
        self.systems = sorted({o["system"] for o in c.obs})
        mix = defaultdict(Counter)
        for i, o in enumerate(c.obs):
            if c.account[i] >= 0:
                mix[(c.account[i], self.month[i])][o["system"]] += 1
        self.mix = mix
        self.states = [s for s in c.states if s[0] in self.acc]
        sid = {s: i for i, s in enumerate(c.states)}
        self.sid = sid
        self.trans = [(sid[s], sid[(s[0], next_key(s[1], "month"))]) for s in self.states
                      if (s[0], next_key(s[1], "month")) in sid]
        self.g = np.array([c.states[a][0] for a, b in self.trans])

    def key(self, i):
        s = self.c.states[i]
        return (self.acc.index(s[0]), s[1])

    def actions(self, i):
        return np.array([float(self.A[self.key(i)][a] > 0) for a in ACTIONS])

    def system_mix(self, i):
        m = self.mix[self.key(i)]
        tot = sum(m.values()) or 1
        return np.array([m[x] / tot for x in self.systems])

    def open_deal(self, i):
        a, m = self.key(i)
        lo, hi = bounds(m, "month")
        stage = None
        for oid in self.opp_of[a]:
            rows = [r for r in self.hist.get(oid, []) if r[0] < hi]
            if rows and rows[-1][1] in OPEN:
                stage = rows[-1][1]
        return stage is not None

    def outcomes(self, i):
        k = self.key(i)
        return {"deal_progress": float(self.adv[k] > 0), "escalation": float(self.esc[k] > 0)}

    def trouble(self, i):
        return self.esc[self.key(i)] > 0 and self.open_deal(i)


def reps(w):
    X = {"good_month": feats("emb2_good_month.npy"), "bad": feats("emb2_bad.npy"), "minilm": feats("minilm_day2.npy")}
    S = {k: state_matrix(w.c, v)[0] for k, v in X.items()}
    S["good_month+minilm"] = two_channel(S["good_month"], S["minilm"])
    return X, S


# ------------------------------------------------------------------------------------------------ table
def stage_table():
    w = World()
    F = np.array([w.actions(a) for a, b in w.trans])
    Fn = np.array([w.actions(b) for a, b in w.trans])
    Y = {k: np.array([w.outcomes(b)[k] for a, b in w.trans]) for k in ("deal_progress", "escalation")}
    out = {"month_states": len(w.states), "transitions": len(w.trans), "support_agents": w.n_agents,
           "action_share": {a: float(np.mean([w.actions(i)[j] for i in range(len(w.c.states))
                                              if w.c.states[i][0] in w.acc])) for j, a in enumerate(ACTIONS)},
           "outcome_rate_next_month": {k: float(v.mean()) for k, v in Y.items()},
           "trouble_transitions": int(sum(w.trouble(a) for a, b in w.trans)), "after_action": {}}
    for j, a in enumerate(ACTIONS):
        out["after_action"][a] = {f"{k}|{lag}": [float(y[M[:, j] > 0].mean()), float(y[M[:, j] == 0].mean())]
                                  for k, y in Y.items() for lag, M in (("this_month", F), ("next_month", Fn))}
    # support replies mark months with a ticket, and tickets persist: what the reply indicator stands in for
    from sklearn.metrics import roc_auc_score
    org2acct = {o["id"]: o["external_id"] for o in w.gt["zendesk_orgs"]}
    tick = Counter()
    for t in w.gt["tickets"]:
        a, ts = org2acct.get(t["organization_id"]), _ts(t["created_at"])
        if a in w.acc and ts:
            tick[(w.acc.index(a), window_key(ts, "month"))] += 1
    tk = np.array([tick[w.key(a)] > 0 for a, b in w.trans]).astype(float)
    sr = F[:, ACTIONS.index("support_reply")]
    out["support_reply_given_ticket"] = [float(sr[tk == 1].mean()), float(sr[tk == 0].mean())]
    out["ticket_this_month_auroc_for_escalation_next"] = float(roc_auc_score(Y["escalation"], tk))
    out["transitions_with_prospects"] = int(sum((s[0], next_key(s[1], "month")) in w.sid for s in w.c.states))
    print(json.dumps(out, indent=1))
    save("table", out)


# ------------------------------------------------------------------------------------------------ dynamics
def _ridge_tuned(Xtr, Ytr, gtr):
    """Ridge strength chosen by leave-one-customer-out validation inside the training customers."""
    from sklearn.linear_model import Ridge
    best, best_e = None, np.inf
    for al in ALPHAS:
        e = 0.0
        for cu in np.unique(gtr):
            m = gtr != cu
            r = Ridge(alpha=al).fit(Xtr[m], Ytr[m])
            e += float(((Ytr[~m] - r.predict(Xtr[~m])) ** 2).sum())
        if e < best_e:
            best, best_e = al, e
    return Ridge(alpha=best).fit(Xtr, Ytr), best


def _cluster_ci(per_item, groups, stat, n_boot=2000, seed=0):
    """Bootstrap over customers (clusters) of a statistic of per-item values."""
    rng = np.random.default_rng(seed)
    cust = np.unique(groups)
    idx = {cu: np.where(groups == cu)[0] for cu in cust}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[cu] for cu in rng.choice(cust, len(cust))])
        vals.append(stat(*[x[pick] for x in per_item]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stage_dynamics():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    w = World()
    _, S = reps(w)
    a_, b_, g = np.array([a for a, b in w.trans]), np.array([b for a, b in w.trans]), w.g
    F = np.array([w.actions(a) for a in a_])                  # actions this month (before the next state)
    Fn = np.array([w.actions(b) for b in b_])                 # actions next month (part of the next state itself)
    Mx = np.array([w.system_mix(a) for a in a_])              # this month's mix of systems (provenance, no actions)
    Mxn = np.array([w.system_mix(b) for b in b_])             # next month's mix of systems
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(F)) for _ in range(10)]
    out = {"transitions": len(w.trans), "actions": list(ACTIONS)}
    for name, Sm in S.items():
        Z = zscore(Sm)
        err = defaultdict(list)
        for cu in np.unique(g):
            trm, tem = g != cu, g == cu
            pca = PCA(32, random_state=0).fit(Z[a_[trm]])
            A, B = pca.transform(Z[a_]), pca.transform(Z[b_])
            D = B - A
            sc = A[trm].std()

            def feat(M):
                mu, sd = M[trm].mean(0), M[trm].std(0) + 1e-6
                return (M - mu) / sd * sc
            k = float((A[trm] * B[trm]).sum() / (A[trm] * A[trm]).sum())
            err["no_change"] += ((B[tem] - A[tem]) ** 2).sum(1).tolist()
            err["shrink"] += ((B[tem] - k * A[tem]) ** 2).sum(1).tolist()
            for label, X in (("state", A), ("state+actions", np.concatenate([A, feat(F)], 1)),
                             ("state+next_actions", np.concatenate([A, feat(Fn)], 1)),
                             ("state+mix", np.concatenate([A, feat(Mx)], 1)),
                             ("state+next_mix", np.concatenate([A, feat(Mxn)], 1)),
                             ("state+next_mix+next_actions", np.concatenate([A, feat(Mxn), feat(Fn)], 1))):
                r, _ = _ridge_tuned(X[trm], D[trm], g[trm])
                err[label] += ((D[tem] - r.predict(X[tem])) ** 2).sum(1).tolist()
            for p_i, p in enumerate(perms[:3]):                  # actions shuffled across transitions
                Xp = np.concatenate([A, feat(F[p])], 1)
                r, _ = _ridge_tuned(Xp[trm], D[trm], g[trm])
                err[f"state+shuffled_actions_{p_i}"] += ((D[tem] - r.predict(Xp[tem])) ** 2).sum(1).tolist()
            err["g"] += [cu] * int(tem.sum())
        gg = np.array(err.pop("g"))
        E = {k: np.array(v) for k, v in err.items()}
        base = E["no_change"].sum()
        r = {k: float(1 - v.sum() / base) for k, v in E.items() if not k.startswith("state+shuffled")}
        r["state+shuffled_actions"] = float(np.mean([1 - E[f"state+shuffled_actions_{i}"].sum() / base for i in range(3)]))
        gain = lambda e1, e0: 1 - e1.sum() / e0.sum()  # noqa: E731
        r["actions_vs_state"] = float(gain(E["state+actions"], E["state"]))
        r["actions_vs_state_ci"] = _cluster_ci((E["state+actions"], E["state"]), gg, gain)
        r["next_actions_vs_state"] = float(gain(E["state+next_actions"], E["state"]))
        r["next_actions_vs_state_ci"] = _cluster_ci((E["state+next_actions"], E["state"]), gg, gain)
        r["actions_vs_shrink"] = float(gain(E["state+actions"], E["shrink"]))
        r["mix_vs_state"] = float(gain(E["state+mix"], E["state"]))
        r["next_mix_vs_state"] = float(gain(E["state+next_mix"], E["state"]))
        r["next_mix_vs_state_ci"] = _cluster_ci((E["state+next_mix"], E["state"]), gg, gain)
        r["next_mix_vs_next_actions"] = float(gain(E["state+next_mix"], E["state+next_actions"]))
        r["next_mix_vs_next_actions_ci"] = _cluster_ci((E["state+next_mix"], E["state+next_actions"]), gg, gain)
        r["next_actions_beyond_next_mix"] = float(gain(E["state+next_mix+next_actions"], E["state+next_mix"]))
        r["next_actions_beyond_next_mix_ci"] = _cluster_ci((E["state+next_mix+next_actions"], E["state+next_mix"]),
                                                           gg, gain)
        r["shuffled_vs_state"] = float(r["state+shuffled_actions"] - r["state"]) / (1 - r["state"])
        out[name] = r
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    # how much of this month's actions the state already holds: each action predicted from the state
    def loao_auc(X, y, C=0.5):
        p = np.zeros(len(y))
        for cu in np.unique(g):
            trm, tem = g != cu, g == cu
            if len(np.unique(y[trm])) < 2:
                p[tem] = y[trm].mean(); continue
            p[tem] = LogisticRegression(C=C, max_iter=3000).fit(X[trm], y[trm]).predict_proba(X[tem])[:, 1]
        return p

    def comps(Sm, k=16):
        Z = zscore(Sm)
        A = np.zeros((len(a_), k))
        for cu in np.unique(g):
            trm, tem = g != cu, g == cu
            pca = PCA(k, random_state=0).fit(Z[a_[trm]])
            A[tem] = pca.transform(Z[a_[tem]]) / (pca.transform(Z[a_[trm]]).std() + 1e-9)
        return A

    C_ = {name: comps(Sm) for name, Sm in S.items()}
    out["action_from_state_auroc"] = {name: {act: float(roc_auc_score(F[:, j], loao_auc(C_[name], F[:, j])))
                                             for j, act in enumerate(ACTIONS) if 0 < F[:, j].sum() < len(F)}
                                      for name in S}
    print("actions from state", json.dumps(out["action_from_state_auroc"]), flush=True)
    # next month's outcomes from this month's state and actions, in every representation
    Y = {k: np.array([w.outcomes(b)[k] for b in b_]) for k in ("deal_progress", "escalation")}
    oc = {}
    for yk, y in Y.items():
        oc[yk] = {"base_rate": float(y.mean()), "actions": float(roc_auc_score(y, loao_auc(F, y)))}
        for name in S:
            p_s, p_sa = loao_auc(C_[name], y), loao_auc(np.concatenate([C_[name], F], 1), y)
            d = lambda yy, p1, p0: roc_auc_score(yy, p1) - roc_auc_score(yy, p0)  # noqa: E731
            oc[yk][name] = {"state": float(roc_auc_score(y, p_s)), "state+actions": float(roc_auc_score(y, p_sa)),
                            "difference": float(d(y, p_sa, p_s)), "difference_ci": _cluster_ci((y, p_sa, p_s), g, d)}
        sup = F[:, ACTIONS.index("support_reply")]
        oc[yk]["support_reply_only"] = float(roc_auc_score(y, sup)) if 0 < sup.sum() < len(sup) else None
        oc[yk]["support_reply_only_heldout"] = float(roc_auc_score(y, loao_auc(sup[:, None], y)))
        for name in S:
            p_ss = loao_auc(np.concatenate([C_[name], sup[:, None]], 1), y)
            p_sa = loao_auc(np.concatenate([C_[name], F], 1), y)
            d = lambda yy, p1, p0: roc_auc_score(yy, p1) - roc_auc_score(yy, p0)  # noqa: E731
            oc[yk][name]["state+support_reply"] = float(roc_auc_score(y, p_ss))
            oc[yk][name]["all_actions_minus_support_reply"] = float(d(y, p_sa, p_ss))
            oc[yk][name]["all_actions_minus_support_reply_ci"] = _cluster_ci((y, p_sa, p_ss), g, d)
        print("outcome", yk, json.dumps(oc[yk]), flush=True)
    out["outcomes"] = oc
    save("dynamics", out)


# ------------------------------------------------------------------------------------------------ events
def stage_events():
    """State in the 14 days before and after an action, against matched times without one. The action's own
    records (the call's transcript and invitation, the Jira issue, the leader's messages) are left out of the
    windows, so the comparison is about everything else the company and customer wrote."""
    from sklearn.decomposition import PCA
    w = World()
    c = w.c
    X = l2n(feats("emb2_good_month.npy"))
    sysn = np.array([o["system"] for o in c.obs])
    ts, acct = np.asarray(c.ts, float), c.account
    by_acct = {a: np.where(acct == a)[0] for a in range(len(w.acc))}
    events = []
    for a, times in w.call_times.items():
        events += [("call", a, t) for t in times]
    for i in np.where(sysn == "jira")[0]:
        if acct[i] >= 0:
            events.append(("eng_escalation", int(acct[i]), float(ts[i])))
    lead = defaultdict(list)
    for i in range(len(c.obs)):
        if acct[i] >= 0 and w.author[i] in LEADERS and sysn[i] in ("gmail", "slack"):
            lead[int(acct[i])].append(float(ts[i]))
    for a, tt in lead.items():                                   # first message of each leader episode
        tt.sort()
        last = -np.inf
        for t in tt:
            if t - last > 14 * DAY:
                events.append(("leader", a, t))
            last = t
    own = {"call": lambda i: sysn[i] in ("gong", "calendar"), "eng_escalation": lambda i: sysn[i] == "jira",
           "leader": lambda i: w.author[i] in LEADERS}
    ev_times = defaultdict(list)
    for kind, a, t in events:
        ev_times[(kind, a)].append(t)
    for a, tt in lead.items():                                   # matched times avoid every leader message
        ev_times[("leader", a)] = list(tt)

    def window(a, lo, hi, kind):
        ids = [i for i in by_acct[a] if lo <= ts[i] < hi and not own[kind](i)]
        return (X[ids].mean(0), len(ids)) if len(ids) >= 2 else None

    rng = np.random.default_rng(0)
    rows = []
    for kind, a, t in events:
        pre, post = window(a, t - 14 * DAY, t, kind), window(a, t, t + 14 * DAY, kind)
        if pre is None or post is None:
            continue
        rows.append((kind, a, pre[0], post[0], 1, pre[1], post[1]))
        # a matched time for the same customer with no event of this kind within 14 days
        cand = ts[by_acct[a]]
        for _ in range(20):
            u = float(rng.choice(cand))
            if all(abs(u - e) > 14 * DAY for e in ev_times[(kind, a)]):
                p0, p1 = window(a, u - 14 * DAY, u, kind), window(a, u, u + 14 * DAY, kind)
                if p0 is not None and p1 is not None:
                    rows.append((kind, a, p0[0], p1[0], 0, p0[1], p1[1]))
                    break
    out = {}
    for kind in ("call", "eng_escalation", "leader"):
        R = [r for r in rows if r[0] == kind]
        shift = np.array([1 - float(r[2] @ r[3] / (np.linalg.norm(r[2]) * np.linalg.norm(r[3]) + 1e-9)) for r in R])
        lab = np.array([r[4] for r in R]); grp = np.array([r[1] for r in R])
        npre = np.array([r[5] for r in R], float); npost = np.array([r[6] for r in R], float)
        diff = lambda s, l: s[l == 1].mean() - s[l == 0].mean()  # noqa: E731

        def adjusted(s, l, n0, n1):
            """Difference after adjusting for how many records each window holds (least squares)."""
            Xd = np.column_stack([np.ones(len(s)), l, np.log(n0), np.log(n1)])
            return float(np.linalg.lstsq(Xd, s, rcond=None)[0][1])
        out[kind] = {"events": int(lab.sum()), "matched": int((lab == 0).sum()),
                     "customers": int(len(np.unique(grp[lab == 1]))),
                     "shift_event": float(shift[lab == 1].mean()), "shift_matched": float(shift[lab == 0].mean()),
                     "records_before_event": float(np.median(npre[lab == 1])),
                     "records_before_matched": float(np.median(npre[lab == 0])),
                     "difference": float(diff(shift, lab)), "difference_ci": _cluster_ci((shift, lab), grp, diff),
                     "adjusted_difference": adjusted(shift, lab, npre, npost),
                     "adjusted_difference_ci": _cluster_ci((shift, lab, npre, npost), grp, adjusted)}
        print("events", kind, json.dumps(out[kind]), flush=True)
    # does knowing which action was taken help predict where the state goes?
    P0 = np.array([r[2] for r in rows]); P1 = np.array([r[3] for r in rows]); grp = np.array([r[1] for r in rows])
    kinds = ["call", "eng_escalation", "leader"]
    onehot = np.array([[float(r[0] == k and r[4] == 1) for k in kinds] for r in rows])
    err = defaultdict(list)
    for cu in np.unique(grp):
        trm, tem = grp != cu, grp == cu
        pca = PCA(32, random_state=0).fit(P0[trm])
        A, B = pca.transform(P0), pca.transform(P1)
        D = B - A
        sc = A[trm].std()
        H = onehot * sc
        err["no_change"] += ((B[tem] - A[tem]) ** 2).sum(1).tolist()
        for label, Xf in (("state", A), ("state+action", np.concatenate([A, H], 1))):
            r, _ = _ridge_tuned(Xf[trm], D[trm], grp[trm])
            err[label] += ((D[tem] - r.predict(Xf[tem])) ** 2).sum(1).tolist()
        err["g"] += [cu] * int(tem.sum())
    gg = np.array(err.pop("g")); E = {k: np.array(v) for k, v in err.items()}
    gain = lambda e1, e0: 1 - e1.sum() / e0.sum()  # noqa: E731
    out["prediction"] = {"pairs": len(rows), "state_vs_no_change": float(gain(E["state"], E["no_change"])),
                         "state+action_vs_no_change": float(gain(E["state+action"], E["no_change"])),
                         "action_vs_state": float(gain(E["state+action"], E["state"])),
                         "action_vs_state_ci": _cluster_ci((E["state+action"], E["state"]), gg, gain)}
    print("events prediction", json.dumps(out["prediction"]), flush=True)
    save("events", out)


# ------------------------------------------------------------------------------------------------ simulate
def stage_simulate(k_nn=15, pool=60, n_boot=300):
    """For each customer-month in trouble, compare what followed similar states of other customers when the company
    intervened (a sales call or an engineering escalation that month) and when it did not, what followed in all other
    customers' months, and what an action-conditioned outcome model predicts. Intervals come from resampling all
    customers and repeating the whole procedure. States are standardized and reduced to 16 components once, without
    labels; neighbours come from every month of the other customers."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    w = World()
    _, S = reps(w)
    a_, b_, g = np.array([a for a, b in w.trans]), np.array([b for a, b in w.trans]), w.g
    F = np.array([w.actions(a) for a in a_])
    I = ((F[:, ACTIONS.index("call")] + F[:, ACTIONS.index("eng_escalation")]) > 0).astype(float)
    Y = {k: np.array([w.outcomes(b)[k] for b in b_]) for k in ("deal_progress", "escalation")}
    esc_now = np.array([w.esc[w.key(a)] > 0 for a in a_]).astype(float)
    T = np.array([w.trouble(a) for a in a_])
    cust_t = g[T]
    A = {name: PCA(16, random_state=0).fit_transform(zscore(Sm[a_])) for name, Sm in S.items()}
    SIM = {name: l2n(M) @ l2n(M).T for name, M in A.items()}
    custs = np.unique(g)
    by_c = {cu: np.where(g == cu)[0] for cu in custs}

    def estimate(idx, rep, y, with_model=False):
        """Mean outcome after intervening and not, over the troubled months in idx (an index list, possibly with
        repeats), for case-based, unmatched and (optionally) model-based estimates."""
        gi = g[idx]
        case, unm, mod, bal = [], [], [], []
        models = {}
        for pos, i in enumerate(idx):
            if not T[i]:
                continue
            other = idx[gi != g[i]]
            on_all, off_all = other[I[other] == 1], other[I[other] == 0]
            unm.append((y[on_all].mean() if len(on_all) else np.nan, y[off_all].mean() if len(off_all) else np.nan))
            near = other[np.argsort(-SIM[rep][i, other], kind="stable")[:pool]]
            on, off = near[I[near] == 1][:k_nn], near[I[near] == 0][:k_nn]
            case.append((y[on].mean() if len(on) else np.nan, y[off].mean() if len(off) else np.nan))
            bal.append((T[on].mean() if len(on) else np.nan, T[off].mean() if len(off) else np.nan,
                        esc_now[on].mean() if len(on) else np.nan, esc_now[off].mean() if len(off) else np.nan))
            if with_model:
                if g[i] not in models:
                    Xo = np.concatenate([A[rep][other], I[other, None]], 1)
                    models[g[i]] = LogisticRegression(C=0.5, max_iter=2000).fit(Xo, y[other]) \
                        if len(np.unique(y[other])) == 2 else None
                m = models[g[i]]
                if m is not None:
                    x1 = np.concatenate([A[rep][i], [1.0]])[None]; x0 = np.concatenate([A[rep][i], [0.0]])[None]
                    mod.append((m.predict_proba(x1)[0, 1], m.predict_proba(x0)[0, 1]))
        f = lambda L: np.nanmean(np.array(L), 0) if L else np.array([np.nan, np.nan])  # noqa: E731
        return {"case": f(case), "unmatched": f(unm), "model": f(mod), "balance": f(bal) if bal else None}

    full = np.arange(len(g))
    out = {"trouble": int(T.sum()), "intervened_in_trouble": int(I[T].sum()), "k": k_nn, "pool": pool, "bootstrap": n_boot,
           "customers_in_trouble": int(len(np.unique(cust_t))),
           "most_trouble_months_one_customer": int(max(Counter(cust_t).values())),
           "decision": "intervene = a sales call or an engineering escalation in the month of trouble"}
    # when did the intervention come, relative to the month's first urgent ticket?
    before = after = 0
    for i in np.where(T & (I == 1))[0]:
        acc_i, mon = w.key(a_[i])
        lo, hi = bounds(mon, "month")
        tk = [t for t in w.esc_times[acc_i] if lo <= t < hi]
        acts = [t for t in w.call_times[acc_i] if lo <= t < hi]
        acts += [float(w.c.ts[j]) for j in range(len(w.c.obs)) if w.c.account[j] == acc_i and
                 w.c.obs[j]["system"] == "jira" and lo <= w.c.ts[j] < hi]
        if tk and acts:
            before += int(min(acts) < min(tk)); after += int(min(acts) >= min(tk))
    out["intervention_before_ticket"], out["intervention_after_ticket"] = before, after
    out["intervention_from_state_auroc"] = {}
    for name in S:
        p = np.zeros(len(g))
        for cu in custs:
            trm, tem = g != cu, g == cu
            p[tem] = LogisticRegression(C=0.5, max_iter=2000).fit(A[name][trm], I[trm]).predict_proba(A[name][tem])[:, 1]
        out["intervention_from_state_auroc"][name] = float(roc_auc_score(I, p))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([by_c[cu] for cu in rng.choice(custs, len(custs))]) for _ in range(n_boot)]
    for yk, y in Y.items():
        res = {"observed_in_trouble": [float(y[T & (I == 1)].mean()), float(y[T & (I == 0)].mean())]}
        for rep in S:
            pt = estimate(full, rep, y, with_model=(rep == "good_month"))
            bs = [estimate(ix, rep, y, with_model=(rep == "good_month")) for ix in boots]
            for label in (("case", "unmatched", "model") if rep == "good_month" else ("case",)):
                key = label if rep == "good_month" else f"case_{rep}"
                d = [b[label][0] - b[label][1] for b in bs]
                d = [x for x in d if np.isfinite(x)]
                res[key] = [float(pt[label][0]), float(pt[label][1])]
                res[key + "_difference"] = float(pt[label][0] - pt[label][1])
                res[key + "_difference_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            sfx = "" if rep == "good_month" else f"_{rep}"
            res["balance_in_trouble" + sfx] = [float(pt["balance"][0]), float(pt["balance"][1])]
            res["balance_urgent_ticket_this_month" + sfx] = [float(pt["balance"][2]), float(pt["balance"][3])]
        out[yk] = res
        print("simulate", yk, json.dumps(res), flush=True)
    print("simulate", json.dumps({k: v for k, v in out.items() if k not in Y}), flush=True)
    save("simulate", out)


# ------------------------------------------------------------------------------------------------ checks
def stage_checks(last_month="2027-01"):
    """Transitions whose next month comes after the day-2 export (30 January 2027). Such a month holds only records
    with later dates, such as planned calendar events, so its outcomes are zero by construction."""
    w = World()
    a_, b_ = [a for a, b in w.trans], [b for a, b in w.trans]
    late = [i for i, b in enumerate(b_) if str(w.key(b)[1]) > last_month]
    T = np.array([w.trouble(a) for a in a_]).astype(bool)
    F = np.array([w.actions(a) for a in a_])
    I = (F[:, ACTIONS.index("call")] + F[:, ACTIONS.index("eng_escalation")]) > 0
    keep = np.ones(len(a_), bool)
    keep[late] = False
    out = {"last_month_with_data": last_month, "transitions_into_later_months": len(late),
           "trouble_into_later_months": int(T[late].sum()) if late else 0,
           "later_next_months": [[str(w.key(a_[i])[0]), str(w.key(a_[i])[1]), str(w.key(b_[i])[1])] for i in late],
           "observed_in_trouble_without_them": {}}
    for yk in ("deal_progress", "escalation"):
        y = np.array([w.outcomes(b)[yk] for b in b_])
        out["observed_in_trouble_without_them"][yk] = [float(y[T & I & keep].mean()), float(y[T & ~I & keep].mean())]
    print(json.dumps(out, indent=1))
    save("checks", out)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        {"table": stage_table, "dynamics": stage_dynamics, "events": stage_events, "simulate": stage_simulate,
         "checks": stage_checks}[st]()
