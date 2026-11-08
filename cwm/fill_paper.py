"""Fill the numbers and appendix tables of paper/main.tex from results/*.json, so that every number in the paper
comes from a results file. Placeholders are ALL-CAPS tokens (E1GOODACC, MASKGOODPCT, APPSEEDTABLE, ...).

python fill_paper.py paper/main.in.tex paper/main.tex
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("CWM_RESULTS", os.path.join(ROOT, "results"))


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


PRH, XA, XB, ST, M2, M2S, E3B = (load(n) for n in ("prh.json", "extras_a.json", "extras_b.json", "state.json",
                                                    "memory2.json", "memory2_seeds.json", "e3b.json"))
ACL = load("acl.json")
V2 = {k: load(f"v2_{k}.json") for k in ("pretrained", "monthly", "acl", "state", "time", "primitives", "scope")}


def merged(prefix):
    """Merge result files written by parallel runs (prefix_a.json, prefix_b.json, ...): setup, runs and retrieval.
    A file without a suffix comes from one complete run of the same command (scripts/reproduce.sh writes it); when it
    exists, it alone is used, so that a new run replaces the stored parallel parts."""
    import re
    out = {}
    names = sorted(os.listdir(RES)) if os.path.isdir(RES) else []
    if prefix + ".json" in names:
        names = [prefix + ".json"]
    for n in names:
        if re.fullmatch(re.escape(prefix) + r"(_[a-z])?\.json", n):
            d = load(n)
            out.setdefault("setup", d.get("setup", {}))
            for k in ("runs", "retrieval"):
                out.setdefault(k, {}).update(d.get(k, {}))
            if "heldout_placement_customer" in d:
                out["heldout_placement_customer"] = d["heldout_placement_customer"]
    return out


ANCH = {k: merged(f"memory_anchor_{k}") for k in ("masked", "plain", "masked_xsys", "plain_xsys")}
XE = load("extras_e.json")
MEM = {"good": [], "minilm": [], "dual": [], "route": []}
MEMQ = {k: load(f"memory2_q_{k}.json") for k in MEM}     # version 3: keys are content within the question's quarter
_MEMSRC = ([(MEMQ[k], k) for k in MEM] if all(MEMQ.values()) else
           [(load("memory2.json"), "good"), (load("memory2_seeds.json"), "good"), (load("memory2_minilm4.json"), "minilm"),
            (load("memory2_dual.json"), "dual"), (load("memory2_route.json"), "route")])
for _d, _store in _MEMSRC:
    for _k, _v in _d.get("runs", {}).items():
        _name, _seed = _k.split("/")
        if _name in ("closed_book", "top16_pretrained"):
            MEM[_store].append((_seed, _name, _v["overall"]))
RTAG = {"ceo": "CEO", "account_executive": "AE", "support_engineer": "SUP", "software_engineer": "ENG",
        "financial_analyst": "FIN"}
RNAME = {"ceo": "CEO", "account_executive": "Account executive", "support_engineer": "Support engineer",
         "software_engineer": "Software engineer", "financial_analyst": "Financial analyst"}
HO = {}
for _n in ("extras_c.json", "extras_d.json"):
    for _k, _v in load(_n).get("heldout", {}).items():
        HO.setdefault(_k, _v)
METRICS = {"ACC": "account_p@10_cross_system", "ST": "state_p@10_cross_system",
           "MR": "probe_account_messages_to_records", "RM": "probe_account_records_to_messages",
           "RHO": "rho_state", "R2": "r2_state", "SYS": "r2_system"}


def runs(arm):
    """{seed: metrics} for one E1 arm."""
    out = {0: PRH["E1"][f"{arm}_s0"]}
    for d in (XA, XB):
        for k, v in d.get("E1_seeds", {}).items():
            a, s = k.rsplit("_s", 1)
            if a == arm:
                out[int(s)] = v
    return dict(sorted(out.items()))


def pm(vals, digits=3):
    vals = np.array(vals, dtype=float)
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{vals.mean():.{digits}f}{{\\scriptsize$\\,\\pm\\,${vals.std(ddof=1):.{digits}f}}}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def minus(text):
    """Replace a leading hyphen-minus on numbers with a typographic minus ($-$)."""
    return re.sub(r"(?<![\w$-])-(?=\d)", "$-$", text)


def signed(x):
    """A signed percentage with a typographic minus, e.g. +55% or -12%."""
    v = round(100 * x)
    return f"$-${abs(v)}\\%" if v < 0 else f"+{v}\\%"


def fill(text):
    sub = {}
    for arm, tag in (("good", "GOOD"), ("bad", "BAD"), ("ntp", "NTP")):
        r = runs(arm)
        for key, metric in METRICS.items():
            vals = [m[metric] for m in r.values()]
            sub[f"E1{tag}{key}"] = pm(vals, 2 if key in ("RHO", "R2", "SYS") else 3)
        sub[f"E1{tag}PCT"] = pct(np.mean([m[METRICS["ACC"]] for m in r.values()]))
        sub[f"E1{tag}STPCT"] = pct(np.mean([m[METRICS["ST"]] for m in r.values()]))
        sub[f"E1{tag}MRPCT"] = pct(np.mean([m[METRICS["MR"]] for m in r.values()]), 0)
        sub[f"E1{tag}NSEEDS"] = str(len(r))
    alt = [np.mean([m[METRICS["ACC"]] for m in runs(a).values()]) for a in ("bad", "ntp")]
    sub["EONEGOODROUND"] = pct(np.mean([m[METRICS["ACC"]] for m in runs("good").values()]), 0)
    sub["EONEALTROUND"] = f"{100 * min(alt):.0f}--{pct(max(alt), 0)}"
    if "tfidf_lsa" in PRH.get("E1", {}):
        sub["EONETFROUND"] = pct(PRH["E1"]["tfidf_lsa"][METRICS["ACC"]], 0)
    mask = XB.get("mask", {})
    if "good_s0" in mask:
        sub["MASKGOODPCT"] = pct(mask["good_s0"][METRICS["ACC"]], 0)
        sub["MASKSTPCT"] = pct(mask["good_s0"][METRICS["ST"]], 0)
        sub["MASKTFPCT"] = pct(mask["tfidf_lsa"][METRICS["ACC"]], 0)
    noise = XB.get("noise", {})
    if noise:
        ps = sorted(float(k[1:]) for k in noise)
        sub["NOISELOWPCT"] = pct(ps[0], 0)
        sub["NOISEHIGHPCT"] = pct(ps[-1], 0)
        clean = PRH["E1"]["good_s0"]
        lo, hi = noise[f"p{ps[0]}"], noise[f"p{ps[-1]}"]
        bad = np.mean([m[METRICS["ACC"]] for m in runs("bad").values()])
        ntp = np.mean([m[METRICS["ACC"]] for m in runs("ntp").values()])
        sub["NOISESENTENCE"] = (
            f"With {pct(ps[0], 0)} of the links wrong, same-customer precision is {pct(lo[METRICS['ACC']], 0)}, "
            f"against {pct(clean[METRICS['ACC']], 0)} with clean links. State precision is {pct(lo[METRICS['ST']], 0)}, "
            f"against {pct(clean[METRICS['ST']], 0)}. With {pct(ps[-1], 0)} of the links wrong, the two values are "
            f"{pct(hi[METRICS['ACC']], 0)} and {pct(hi[METRICS['ST']], 0)}. Even then, same-customer precision is far "
            f"above that of bad views and next-token prediction with clean links ({pct(bad, 0)} and "
            f"{pct(ntp, 0)}). Wrong links blur the space, but they do not break it.")
    if HO:
        sub["HONTEST"] = f"{HO['n_test_records']:,}".replace(",", "{,}")
        for arm, tag in (("good_s0", "GOOD"), ("bad_s0", "BAD"), ("ntp_s0", "NTP"), ("tfidf_lsa", "TF"),
                         ("random_init", "RAND")):
            if arm in HO:
                sub[f"HO{tag}CUST"] = pct(HO[arm]["customer_acc"], 0)
                sub[f"HO{tag}STATE"] = pct(HO[arm]["state_acc"], 0)
                sub[f"HO{tag}ACCPCT"] = pct(HO[arm][METRICS["ACC"]], 0)
        sub["HOREFSTATE"] = pct(HO["reference_customer_known_state_acc"], 0)
        sub["HOTABLE"] = heldout_table()
    if E3B:
        c10 = E3B["corr_by_direction_first_10"]
        sub["E3BCORR10"] = f"{min(c10):.2f}--{max(c10):.2f}"
        sub["E3BCORRTAIL"] = f"{E3B['corr_dirs_33_to_256_mean']:.2f}"
        sub["E3BVAR16"] = pct(E3B["variance_share_top16"], 0)
        sub["E3BR2TOP8"] = pct(E3B["r2_top_k_principal_dims"]["8"], 0)
        sub["E3BR2"] = pct(E3B["r2_all_dims"], 0)
        st = E3B["old_store_precision@10"]
        sub["E3BROTPCT"] = pct(st["query_by_new_encoder_rotated"], 0)
        sub["E3BUNROTPCT"] = pct(st["query_by_new_encoder_unrotated"], 0)
        sub["E3BROT"] = pct(st["query_by_new_encoder_rotated"])
        sub["E3BUNROT"] = pct(st["query_by_new_encoder_unrotated"])
        sub["E3BOLD"] = pct(st["query_by_old_encoder"])
        sub["E3BNHIDDEN"] = f"{E3B.get('n_hidden_from_retraining', 0):,}".replace(",", "{,}")
    if ACL:
        R = ACL["readers"]
        sub["ACLCANARYN"] = str(ACL["canary_records"])
        for r, tag in RTAG.items():
            if r not in R:
                continue
            sub[f"ACLVIS{tag}"] = pct(R[r]["visible_share"], 0)
            sub[f"ACLCOST{tag}OPEN"] = pct(R[r]["placement_open_encoder"]["customer_acc"], 0)
            sub[f"ACLCOST{tag}OWN"] = pct(R[r].get("placement_own_encoder", R[r]["placement_open_encoder"])["customer_acc"], 0)
            for conf, ctag in (("open", "OPEN"), ("filtered", "FILT"), ("built_in", "BUILT")):
                if "leak" in R[r]:
                    L = R[r]["leak"][conf]
                    sub[f"ACL{tag}{ctag}CODE"] = pct(L["with"]["codename_precision"], 0)
                    sub[f"ACL{tag}{ctag}SECRET"] = pct(L["with"]["secret_precision@5"], 0)
                    if "without" in L:
                        sub[f"ACL{tag}{ctag}NOCODE"] = pct(L["without"]["codename_precision"], 0)
                        sub[f"ACL{tag}{ctag}NOSECRET"] = pct(L["without"]["secret_precision@5"], 0)
            if "leak" in R[r] and "codename_advantage" in R[r]["leak"]["filtered"]:
                sub[f"ACL{tag}FILTCODEADVS"] = f"{R[r]['leak']['filtered']['codename_advantage']:+.2f}"
                sub[f"ACL{tag}FILTSECADVS"] = f"{R[r]['leak']['filtered']['secret_advantage']:+.2f}"
            if "leak" in R[r]:
                sub[f"ACL{tag}CODECHANCE"] = pct(R[r]["leak"]["built_in"]["with"]["codename_chance"], 0)
                sub[f"ACL{tag}SECRETCHANCE"] = pct(R[r]["leak"]["built_in"]["with"]["secret_chance@5"], 0)
        ni = ACL.get("non_interference", {})
        sub["ACLNIDIFF"] = f"{max(ni.get('max_abs_diff_embeddings', float('nan')), ni.get('max_abs_diff_queries', 0.0)):g}"
        sub["ACLPOLICYTABLE"] = acl_policy_table()
        sub["ACLSECRETSENTENCE"] = ACL["secret"].capitalize() + "."
        sub["ACLLEAKTABLE"] = acl_leak_table()
    fill_corpus(sub)
    fill_v2(sub)
    fill_anchor(sub)
    fill_actions(sub)
    sub["APPSEEDTABLE"] = seed_table()
    sub["APPSTATETABLE"] = state_table()
    sub["APPMEMTABLE"] = memory_table()
    LAST_SUB.clear()
    LAST_SUB.update(sub)
    for k in sorted(sub, key=len, reverse=True):
        text = text.replace(k, minus(sub[k]))
    return text


LAST_SUB = {}   # the last substitution map, written out when FILL_MAP names a file (used by tools/ste_check.py)


def mem_stats(store):
    rows = MEM[store]
    own_cb = any(n == "closed_book" for _, n, _ in rows)
    cb = {s: v for s, n, v in (rows if own_cb else MEM["minilm"]) if n == "closed_book"}
    tk = {s: v for s, n, v in rows if n == "top16_pretrained"}
    seeds = sorted(set(cb) & set(tk))
    a, b = np.array([cb[x] for x in seeds]), np.array([tk[x] for x in seeds])
    return a, b, b - a, len(seeds)


def fill_v2(sub):
    P = V2["pretrained"]
    if P:
        sub["PTRAWACC"] = pct(P["minilm_raw"][METRICS["ACC"]], 0)
        sub["PTHEADACC"] = pct(P["minilm_head"][METRICS["ACC"]], 0)
        sub["PTRAWHO"] = pct(P["heldout"]["minilm_raw"]["customer_acc"], 0)
        sub["PTHEADHO"] = pct(P["heldout"]["minilm_head"]["customer_acc"], 0)
        for tag, m in (("PTRAW", P["minilm_raw"]), ("PTHEAD", P["minilm_head"])):
            sub[f"{tag}ROW"] = (f"{m[METRICS['ACC']]:.3f} & {m[METRICS['ST']]:.3f} & {m[METRICS['MR']]:.3f} & "
                                f"{m[METRICS['RM']]:.3f} & {m['rho_state']:.2f} & {m['r2_state']:.2f} & {m['r2_system']:.2f}")
        for tag, m in (("PTRAW", P["heldout"]["minilm_raw"]), ("PTHEAD", P["heldout"]["minilm_head"])):
            sub[f"{tag}HOROW"] = (f"{m[METRICS['ACC']]:.3f} & {m[METRICS['ST']]:.3f} & {m['customer_acc']:.3f} & "
                                  f"{m['state_acc']:.3f}")
    if XE.get("mask_employees"):
        m = XE["mask_employees"]
        sub["MASKEMPGOODPCT"] = pct(m["good_s0"][METRICS["ACC"]], 0)
        sub["MASKEMPSTPCT"] = pct(m["good_s0"][METRICS["ST"]], 0)
        sub["MASKEMPTFPCT"] = pct(m["tfidf_lsa"][METRICS["ACC"]], 0)
        sub["MASKEMPCHANGED"] = pct(m["linked_records_changed"] / m["linked_records"], 0)
        sub["MASKEMPN"] = f"{m['linked_records_changed']:,}".replace(",", "{,}")
        sub["MASKCUSTN"] = f"{XB['mask']['linked_records_changed']:,}".replace(",", "{,}")
    M = V2["monthly"]
    if M:
        sub["MONSTATES"] = str(M["n_states"])
        sub["MONREF"] = pct(M["reference_customer_known_month_acc"], 0)
        sub["MONGOODCUST"] = pct(M["good_month"]["customer_acc"], 0)
        sub["MONGOODMONTH"] = pct(M["good_month"]["state_acc"], 0)
        sub["MONHEADMONTH"] = pct(M["minilm_head_month"]["state_acc"], 0)
        sub["MONRAWMONTH"] = pct(M["minilm_raw"]["state_acc"], 0)
        rows = ["\\setlength{\\tabcolsep}{5pt}\\begin{tabular}{llcc}", "\\toprule",
                "& & \\multicolumn{2}{c}{Held-out records placed with the right} \\\\", "\\cmidrule(lr){3-4}",
                "Windows & Representation & customer & customer and window \\\\", "\\midrule"]
        if HO:
            rows.append(f"Quarter (180 states) & good views & {HO['good_s0']['customer_acc']:.3f} & {HO['good_s0']['state_acc']:.3f} \\\\")
            rows.append(f" & customer known, its busiest quarter & 1.000 & {HO['reference_customer_known_state_acc']:.3f} \\\\")
            rows.append("\\addlinespace[2pt]")
        rows.append(f"Month ({M['n_states']} states) & good views & {M['good_month']['customer_acc']:.3f} & "
                    f"\\textbf{{{M['good_month']['state_acc']:.3f}}} \\\\")
        rows.append(f" & pretrained encoder & {M['minilm_raw']['customer_acc']:.3f} & {M['minilm_raw']['state_acc']:.3f} \\\\")
        rows.append(f" & pretrained + good-view head & {M['minilm_head_month']['customer_acc']:.3f} & "
                    f"{M['minilm_head_month']['state_acc']:.3f} \\\\")
        rows.append(f" & customer known, its busiest month & 1.000 & {M['reference_customer_known_month_acc']:.3f} \\\\")
        rows += ["\\bottomrule", "\\end{tabular}"]
        sub["MONTABLE"] = "\n".join(rows)
    T = V2["time"]
    if T:
        sub["TIMESTATES"] = str(T["n_states"]); sub["TIMETRANS"] = str(T["n_transitions"])
        B = T["baselines"]
        maj = np.mean([B[k]["majority"] for k in ("tickets", "priority", "call", "stage")])
        orc = np.mean([B[k]["oracle_persistence"] for k in ("tickets", "priority", "call", "stage")])
        sub["TIMEMAJ"] = pct(maj, 0); sub["TIMEORACLE"] = pct(orc, 0)
        names = [("good_month", "Good views, month states"), ("good_quarter", "Good views, quarter states"),
                 ("bad", "Bad views"), ("minilm", "Pretrained encoder"), ("minilm_head_month", "Pretrained + good-view head"),
                 ("good_month+minilm", "Two channels (good + pretrained)")]
        rows = ["\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lccccc}", "\\toprule",
                "& Next month & Cosine with & \\multicolumn{3}{c}{Squared error vs no change} \\\\",
                "\\cmidrule(lr){4-6}",
                "Representation & facts & next state & ridge, $\\alpha = 10$ & ridge, tuned $\\alpha$ & shrink \\\\",
                "\\midrule"]
        for k, n in names:
            if k in T:
                d = T[k]["dynamics"]
                rows.append(f"{n} & {T[k]['mean_next']:.2f} & {d['cos_persistence']:.2f} & "
                            f"{signed(-d['error_reduction_vs_persistence'])} & "
                            f"{signed(-d['tuned_error_reduction_vs_persistence'])} & "
                            f"{signed(-d['shrink_error_reduction_vs_persistence'])} \\\\")
        rows += ["\\midrule", f"Majority class & {maj:.2f} & & & & \\\\", f"Oracle persistence & {orc:.2f} & & & & \\\\",
                 "\\bottomrule", "\\end{tabular}"]
        sub["TIMETABLE"] = "\n".join(rows)
        gm = T["good_month"]["dynamics"]
        sub["DYNGOODERR"] = f"{100 * -gm['error_reduction_vs_persistence']:.0f}\\%"
        sub["DYNBADERR"] = f"{100 * T['bad']['dynamics']['error_reduction_vs_persistence']:.0f}\\%"
        sub["DYNMINIERR"] = f"{100 * T['minilm']['dynamics']['error_reduction_vs_persistence']:.0f}\\%"
        sub["SHRINKBAD"] = f"{100 * T['bad']['dynamics']['shrink_error_reduction_vs_persistence']:.0f}\\%"
        sub["SHRINKMINI"] = f"{100 * T['minilm']['dynamics']['shrink_error_reduction_vs_persistence']:.0f}\\%"
        sub["SHRINKGOOD"] = f"{100 * gm['shrink_error_reduction_vs_persistence']:.0f}\\%"
        sub["DYNGOODCOS"] = f"{gm['cos_persistence']:.2f}"; sub["DYNGOODDIR"] = f"{gm['cos_change_direction']:.2f}"
        dyn = {k: T[k]["dynamics"] for k, _ in names if k in T}
        gains = {k: 100 * (d["tuned_error_reduction_vs_persistence"] - d["shrink_error_reduction_vs_persistence"])
                 for k, d in dyn.items()}
        tuned = [100 * d["tuned_error_reduction_vs_persistence"] for d in dyn.values()]
        sub["TUNEDRANGE"] = f"{min(tuned):.0f}--{max(tuned):.0f}\\%"
        sub["TUNEDMAXGAIN"] = f"{max(gains.values()):.1f}"
        sub["TUNEDNWORSE"] = str(sum(v <= 0.5 for v in gains.values()))
        sub["SHRINKKBAD"] = f"{T['bad']['dynamics']['shrink_factor']:.2f}"
        sub["SHRINKKMINI"] = f"{T['minilm']['dynamics']['shrink_factor']:.2f}"
        sub["SHRINKKGOOD"] = f"{gm['shrink_factor']:.2f}"
        sub["FCGOODMONTH"] = pct(T["good_month"]["mean_next"], 0)
        sub["FCTWO"] = pct(T["good_month+minilm"]["mean_next"], 0)
        sub["FCBAD"] = pct(T["bad"]["mean_next"], 0)
    S = V2["state"]
    if S:
        names = [("good", "Good views (state)"), ("bad", "Bad views"), ("ntp", "Next-token"), ("minilm", "Pretrained encoder"),
                 ("minilm_head", "Pretrained + good-view head"), ("good+minilm", "Two channels: good + pretrained"),
                 ("good+bad", "Two channels: good + bad views"), ("good+provenance", "Good + system shares")]
        rows = ["\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lcccccc}", "\\toprule",
                "& \\multicolumn{5}{c}{This quarter} & Next quarter \\\\", "\\cmidrule(lr){2-6}\\cmidrule(lr){7-7}",
                "Representation & tickets & priority & call & stage & mean & mean \\\\", "\\midrule"]
        for k, n in names:
            if k in S:
                r = S[k]
                rows.append(f"{n} & " + " & ".join(f"{r[x]['now']:.2f}" for x in ("tickets", "priority", "call", "stage"))
                            + f" & {r['mean_now']:.2f} & {r['mean_next']:.2f} \\\\")
        rows += ["\\bottomrule", "\\end{tabular}"]
        sub["STATE2TABLE"] = "\n".join(rows)
        sub["CALLGOOD"] = pct(S["good"]["call"]["now"], 0); sub["CALLTWO"] = pct(S["good+minilm"]["call"]["now"], 0)
        sub["CALLGB"] = pct(S["good+bad"]["call"]["now"], 0); sub["CALLMINI"] = pct(S["minilm"]["call"]["now"], 0)
        sub["NOWTWO"] = pct(S["good+minilm"]["mean_now"], 0); sub["NOWMINI"] = pct(S["minilm"]["mean_now"], 0)
        sub["NOWBAD"] = pct(S["bad"]["mean_now"], 0); sub["NOWGOOD"] = pct(S["good"]["mean_now"], 0)
    if MEM["dual"]:
        rows = ["\\begin{tabular}{lcccc}", "\\toprule", "Store & Seeds & Closed book & 16 retrieved records & Gain \\\\", "\\midrule"]
        for k, n in (("good", "State channel only (good views)"), ("minilm", "Detail channel only (pretrained)"),
                     ("dual", "Two channels, concatenated"), ("route", "Two channels, routed$^{*}$")):
            if not MEM[k]:
                continue
            a, b, d, ns = mem_stats(k)
            rows.append(f"{n} & {ns} & {a.mean():.3f}{{\\scriptsize$\\,\\pm\\,${a.std(ddof=1):.3f}}} & "
                        f"{b.mean():.3f}{{\\scriptsize$\\,\\pm\\,${b.std(ddof=1):.3f}}} & "
                        f"{100 * d.mean():+.1f}{{\\scriptsize$\\,\\pm\\,${100 * d.std(ddof=1):.1f}}} \\\\")
            sub[f"MEM{k.upper()}"] = pct(b.mean(), 1)
            sub[f"MEMGAIN{k.upper()}"] = f"{100 * d.mean():.1f}"
        rows += ["\\bottomrule", "\\end{tabular}"]
        sub["MEM2TABLE"] = "\n".join(rows)
        RQ = load("memory2_q_retrieval.json")
        if RQ:
            for k in ("good", "minilm", "dual"):
                sub[f"MEMRETR{k.upper()}"] = pct(RQ[k]["quarter"]["same_customer"], 0)
                sub[f"MEMRETRROW{k.upper()}"] = pct(RQ[k]["row"]["same_customer"], 0)
        if all(MEMQ.values()):
            sub["MEMPAIRDUALDETAIL"] = fmt_paired(paired(runs_by_seed(MEMQ["dual"], "top16_pretrained"),
                                                         runs_by_seed(MEMQ["minilm"], "top16_pretrained")))
            sub["MEMPAIRDETAILSTATE"] = fmt_paired(paired(runs_by_seed(MEMQ["minilm"], "top16_pretrained"),
                                                          runs_by_seed(MEMQ["good"], "top16_pretrained")))
    PR = V2["primitives"]
    if PR:
        names = [("good", "Good views, quarter states"), ("good_month", "Good views, month states"), ("bad", "Bad views"),
                 ("minilm", "Pretrained encoder"), ("minilm_head", "Pretrained + good-view head"),
                 ("good+minilm", "Two channels"), ("volume", "Record volume only")]
        SC = PR.get("situations_controls", {})
        rows = ["\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{lcccccc}", "\\toprule",
                "& \\multicolumn{2}{c}{Drift, AUROC} & Misalignment & \\multicolumn{3}{c}{Similar situations, precision@5} \\\\",
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-7}",
                "Representation & same week & week ahead & AUROC & all & other quarters & in trouble \\\\", "\\midrule"]
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else "--"  # noqa: E731
        for k, n in names:
            d = PR["drift"].get(k, {}); m = PR["misalignment"].get(k, {}); sx = SC.get(k, {})
            rows.append(f"{n} & {fmt(d.get('auroc_event_week'))} & {fmt(d.get('auroc_next_week'))} & "
                        f"{fmt(m.get('auroc'))} & {fmt(sx.get('precision@5_all'))} & "
                        f"{fmt(sx.get('precision@5_other_quarters'))} & {fmt(sx.get('trouble_precision@5'))} \\\\")
        ch = PR["situations"]["good"]["chance"]
        rows += ["\\midrule", f"Chance or base rate & 0.50 & 0.50 & 0.50 & {ch:.2f} & "
                 f"{fmt(SC.get('good', {}).get('chance_other_quarters'))} & {fmt(SC.get('good', {}).get('trouble_base_rate'))} \\\\",
                 "\\bottomrule", "\\end{tabular}"]
        sub["PRIMTABLE"] = "\n".join(rows)
        dr = [v[x] for k, v in PR["drift"].items() if k != "volume" for x in ("auroc_event_week", "auroc_next_week")]
        sub["DRIFTRANGE"] = f"{min(dr):.2f}--{max(dr):.2f}"
        g = PR["drift"]["good"]
        n1, n2 = g["event_weeks"], g["weeks"] - g["event_weeks"]
        sub["DRIFTSE2"] = f"{2 * np.sqrt((n1 + n2 + 1) / (12 * n1 * n2)):.2f}"      # 2 SE of AUROC under no signal
        m = PR["misalignment"]["good"]
        n1, n2 = m["at_risk"], m["windows"] - m["at_risk"]
        sub["MISSE2"] = f"{2 * np.sqrt((n1 + n2 + 1) / (12 * n1 * n2)):.2f}"
        sub["DRIFTVOL"] = f"{PR['drift']['volume']['auroc_event_week']:.2f}"
        ms = [v["auroc"] for v in PR["misalignment"].values() if v.get("auroc") is not None]
        sub["MISRANGE"] = f"{min(ms):.2f}--{max(ms):.2f}"
        sx = [v["precision@5"] for v in PR["situations"].values()]
        sub["SITRANGE"] = f"{100 * min(sx):.0f}--{100 * max(sx):.0f}\\%"
        sub["SITCHANCE"] = pct(ch, 0)
        sub["SITGOOD"] = pct(PR["situations"]["good"]["precision@5"], 0)
        sub["SITBAD"] = pct(PR["situations"]["bad"]["precision@5"], 0)
        sub["PRIMWEEKS"] = f"{PR['drift']['good']['weeks']:,}".replace(",", "{,}")
        sub["PRIMEVENTS"] = str(PR["drift"]["good"]["event_weeks"])
        sub["MISWIN"] = str(PR["misalignment"]["good"]["windows"]); sub["MISRISK"] = str(PR["misalignment"]["good"]["at_risk"])
        SC = PR.get("situations_controls")
        if SC:
            sub["SITSTATES"] = str(SC["states"]); sub["TROUBLEN"] = str(SC["trouble_states"])
            sub["SITSAMEQ"] = pct(SC["same_quarter_baseline"], 0)
            sub["SITOTHERQGOOD"] = pct(SC["good"]["precision@5_other_quarters"], 0)
            sub["SITOTHERQCHANCE"] = pct(SC["good"]["chance_other_quarters"], 0)
            sub["TROUBLEBASE"] = pct(SC["good"]["trouble_base_rate"], 0)
            for k, tag in (("good", "GOOD"), ("bad", "BAD"), ("minilm", "MINI"), ("good+minilm", "TWO"),
                           ("minilm_head", "HEAD")):
                sub[f"TROUBLE{tag}"] = pct(SC[k]["trouble_precision@5"], 0)
    A2 = V2["acl"]
    if A2 and ACL:
        R1, R2 = ACL["readers"], A2["readers"]
        rows = ["\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{lccccccc}", "\\toprule",
                "& & \\multicolumn{4}{c}{Non-interfering: trained on readable records only} & \\multicolumn{2}{c}{Trained on everything} \\\\",
                "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}",
                "Reader & Readable & own encoder & backbone & backbone + head & both & encoder & backbone + head \\\\", "\\midrule"]
        for r, tag in RTAG.items():
            if r not in R2:
                continue
            own = R1[r].get("placement_own_encoder", R1[r]["placement_open_encoder"])["customer_acc"]
            raw = R2[r]["placement_raw_backbone"]["customer_acc"]
            head = R2[r]["placement_own_head"]["customer_acc"]
            both = R2[r].get("placement_own_two_channels", {}).get("customer_acc")
            vals = [own, raw, head] + ([both] if both is not None else [])
            best = max(vals)
            cell = lambda v: (f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}") if v is not None else "--"  # noqa: E731
            rows.append(f"{RNAME[r]} & {100 * R2[r]['visible_share']:.0f}\\% & {cell(own)} & {cell(raw)} & {cell(head)} & "
                        f"{cell(both)} & {R1[r]['placement_open_encoder']['customer_acc']:.2f} & "
                        f"{R2[r]['placement_open_head']['customer_acc']:.2f} \\\\")
            sub[f"ACL2{tag}HEAD"] = pct(head, 0)
            sub[f"ACL2{tag}RAW"] = pct(raw, 0)
            sub[f"ACL2{tag}OPENHEAD"] = pct(R2[r]["placement_open_head"]["customer_acc"], 0)
            if both is not None:
                sub[f"ACL2{tag}BOTH"] = pct(both, 0)
            pubk = ("placement_public_base_finetuned_800" if "placement_public_base_finetuned_800" in R2[r]
                    else "placement_public_base_finetuned")
            if pubk in R2[r]:
                sub[f"ACL2{tag}PUB"] = pct(R2[r][pubk]["customer_acc"], 0)
            if "placement_public_base_only" in R2[r]:
                sub[f"ACL2{tag}PUBONLY"] = pct(R2[r]["placement_public_base_only"]["customer_acc"], 0)
            if "leak" in R2[r]:
                L = R2[r]["leak"]
                for conf, ctag in (("open", "OPEN"), ("filtered", "FILT"), ("built_in", "BUILT")):
                    sub[f"ACL2{tag}{ctag}CODEADV"] = f"{L[conf]['codename_advantage']:+.2f}"
                    sub[f"ACL2{tag}{ctag}SECADV"] = f"{L[conf]['secret_advantage']:+.2f}"
                sub[f"ACL2{tag}NI"] = f"{L['non_interference_max_abs_diff']:g}"
        rows += ["\\bottomrule", "\\end{tabular}"]
        sub["ACL2TABLE"] = "\n".join(rows)
    SCO = V2.get("scope", {})
    for r, tag in (("software_engineer", "ENG"), ("financial_analyst", "FIN")):
        if r in SCO:
            x = SCO[r]
            sub[f"SCOPE{tag}"] = pct(x["placement_random_scope_same_size"]["customer_acc"], 0)
            tot = sum(x["reader_systems"].values())
            top_sys, top_n = max(x["reader_systems"].items(), key=lambda kv: kv[1])
            sub[f"SCOPE{tag}TOPSHARE"] = pct(top_n / tot, 0)
            sub[f"SCOPE{tag}TOPSYS"] = {"zendesk": "support tickets", "salesforce": "CRM records", "gmail": "emails",
                                        "slack": "chat messages", "gong": "call transcripts"}.get(top_sys, top_sys)
            sub[f"SCOPE{tag}N"] = f"{x['records']:,}".replace(",", "{,}")
    if A2:
        sub["ACL2PUBRECORDS"] = str(A2.get("public_records", ""))


def paired(runs_a, runs_b, n_boot=2000, seed=0):
    """Paired difference in accuracy (a - b) over the same questions: per-seed differences, and a bootstrap 95%
    interval over questions of the seed-averaged difference."""
    seeds = sorted(set(runs_a) & set(runs_b))
    A = np.array([runs_a[s_]["correct"] for s_ in seeds], float)
    B = np.array([runs_b[s_]["correct"] for s_ in seeds], float)
    d_q = (A - B).mean(0)                                   # per question, averaged over seeds
    rng = np.random.default_rng(seed)
    boots = [d_q[rng.integers(0, len(d_q), len(d_q))].mean() for _ in range(n_boot)]
    return {"mean": float(d_q.mean()), "per_seed": [float(x) for x in (A - B).mean(1)],
            "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))}


def runs_by_seed(d, store):
    return {k.split("/")[1]: v for k, v in d.get("runs", {}).items() if k.split("/")[0] == store and "correct" in v}


def fmt_paired(p):
    lo, hi = p["ci"]
    return (f"{100 * p['mean']:+.1f} points (95\\% interval {100 * lo:+.1f} to {100 * hi:+.1f}; per seed "
            + ", ".join(f"{100 * x:+.1f}" for x in p["per_seed"]) + ")")


def anchor_stats(corpus, store):
    runs = ANCH[corpus].get("runs", {})
    acc = np.array([v["overall"] for k, v in runs.items() if k.split("/")[0] == store])
    ret = [v for k, v in ANCH[corpus].get("retrieval", {}).items() if k.split("/")[0] == store]
    mean = lambda key: float(np.mean([r[key] for r in ret])) if ret else None  # noqa: E731
    return acc, {k: mean(k) for k in ("same_customer", "same_state", "same_state_answering_system")}


def fill_anchor(sub):
    M = ANCH["masked"]
    if not M.get("runs"):
        return
    st = M["setup"]
    sub["ANCHNTEST"] = str(st["test_questions"]); sub["ANCHNTRAIN"] = str(st["train_questions"])
    sub["ANCHNALLTEST"] = str(st["questions_before_anchoring"]["test"])
    rows_def = [("closed", "No store (closed book)", "--"),
                ("state", "Neighbours, state channel", "state"),
                ("detail", "Neighbours, detail channel", "detail"),
                ("concat", "Neighbours, both concatenated", "both"),
                ("routed", "Neighbours, state channel", "detail"),
                ("place_detail", "Placement, detail channel", "detail"),
                ("place_concat", "Placement, both concatenated", "detail"),
                ("place_state", "Placement, state channel", "detail"),
                ("link", "The anchor's own link (oracle)", "detail"),
                ("link_state", "The anchor's own link (oracle)", "state"),
                ("link_prov", "The anchor's own link (oracle)", "system, time")]
    rows = ["\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{llcccc}", "\\toprule",
            "& & \\multicolumn{2}{c}{Names masked} & \\multicolumn{2}{c}{Names kept} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Finding the records & Reading & accuracy & right customer & accuracy & right customer \\\\", "\\midrule"]
    vals = [anchor_stats("plain", k)[0].mean() for k, _, _ in rows_def
            if not k.startswith("link") and len(anchor_stats("plain", k)[0])]
    best_kept = max(vals) if vals else None
    f = lambda v: f"{v:.2f}" if v is not None else "--"  # noqa: E731
    for k, n, rd in rows_def:
        cells = []
        for corpus in ("masked", "plain"):
            acc, rt = anchor_stats(corpus, k)
            if not len(acc):
                cells += ["--", "--"]; continue
            cell = (f"{acc.mean():.3f}{{\\scriptsize$\\,\\pm\\,${acc.std(ddof=1):.3f}}}" if len(acc) > 1 else f"{acc.mean():.3f}")
            if corpus == "plain" and not k.startswith("link") and acc.mean() == best_kept:
                cell = "\\textbf{" + cell + "}"
            cells += [cell, f(rt["same_customer"])]
            tag = ("ANCHM" if corpus == "masked" else "ANCHP") + k.replace("_", "").upper()
            sub[f"{tag}ACC"] = pct(acc.mean(), 1)
            if rt["same_customer"] is not None:
                sub[f"{tag}CUST"] = pct(rt["same_customer"], 0)
                sub[f"{tag}USE"] = pct(rt["same_state_answering_system"], 1)
        if k in ("state", "place_detail", "link"):
            rows.append("\\addlinespace[3pt]")
        rows.append(f"{n} & {rd} & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    sub["ANCHORTABLE"] = "\n".join(rows)
    for corpus, tag in (("masked", "ANCHM"), ("plain", "ANCHP")):
        D = ANCH[corpus]
        ps, pd_ = runs_by_seed(D, "place_state"), runs_by_seed(D, "place_detail")
        if ps and pd_:
            sub[f"{tag}PAIRSTATEDETAIL"] = fmt_paired(paired(ps, pd_))
        knn = {k: anchor_stats(corpus, k)[0].mean() for k in ("state", "detail", "concat", "routed")
               if len(anchor_stats(corpus, k)[0])}
        if knn and ps:
            kb = max(knn, key=knn.get)
            sub[f"{tag}PAIRSTATEKNN"] = fmt_paired(paired(ps, runs_by_seed(D, kb)))
        cl = runs_by_seed(D, "closed")
        if cl and ps:
            sub[f"{tag}PAIRSTATECLOSED"] = fmt_paired(paired(ps, cl))
        if cl and runs_by_seed(D, "routed"):
            sub[f"{tag}PAIRROUTEDCLOSED"] = fmt_paired(paired(runs_by_seed(D, "routed"), cl))
    for corpus, tag in (("masked", "ANCHM"), ("plain", "ANCHP")):
        knn = [anchor_stats(corpus, k)[0].mean() for k in ("state", "detail", "concat", "routed")
               if len(anchor_stats(corpus, k)[0])]
        if knn:
            sub[f"{tag}KNNBEST"] = pct(max(knn), 1)
            sub[f"{tag}KNNRANGE"] = f"{100 * min(knn):.1f}--{100 * max(knn):.1f}\\%"
        hp = ANCH[corpus].get("heldout_placement_customer", {})
        for ch in ("state", "detail"):
            if ch in hp:
                sub[f"{tag}HO{ch.upper()}"] = pct(hp[ch], 0)
        nonoracle = [anchor_stats(corpus, k)[0].mean() for k, _, _ in rows_def
                     if not k.startswith("link") and len(anchor_stats(corpus, k)[0])]
        if nonoracle:
            sub[f"{tag}NONORACLERANGE"] = f"{100 * min(nonoracle):.1f}--{100 * max(nonoracle):.1f}\\%"
            la = anchor_stats(corpus, "link")[0]
            if len(la):
                sub[f"{tag}GAPORACLE"] = f"{100 * (la.mean() - max(nonoracle)):.1f}"
    # retrieval among other systems only (appendix)
    rows = ["\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llcccc}", "\\toprule",
            "& & \\multicolumn{2}{c}{Names masked} & \\multicolumn{2}{c}{Names kept} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Neighbours of the anchor & Reading & accuracy & right customer & accuracy & right customer \\\\", "\\midrule"]
    xs = []
    for k, n, rd in (("state", "state channel", "state"), ("detail", "detail channel", "detail"),
                     ("concat", "both concatenated", "both"), ("routed", "state channel", "detail"),
                     ("link", "the anchor's own link (oracle)", "detail")):
        cells = []
        for corpus in ("masked_xsys", "plain_xsys"):
            acc, rt = anchor_stats(corpus, k)
            if not len(acc):
                cells += ["--", "--"]; continue
            cells += [f"{acc.mean():.3f}{{\\scriptsize$\\,\\pm\\,${acc.std(ddof=1):.3f}}}", f(rt["same_customer"])]
            if k != "link":
                xs.append((corpus, rt["same_customer"]))
        rows.append(f"{n} & {rd} & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    sub["ANCHORXTABLE"] = "\n".join(rows)
    for corpus, tag in (("masked_xsys", "ANCHMX"), ("plain_xsys", "ANCHPX")):
        v = [x for cc, x in xs if cc == corpus and x is not None]
        if v:
            sub[f"{tag}CUSTRANGE"] = f"{100 * min(v):.0f}--{100 * max(v):.0f}\\%"
    thr = [v["anchor_thread"] for k, v in ANCH["masked"].get("retrieval", {}).items()
           if k.split("/")[0] in ("state", "detail", "concat", "routed") and "anchor_thread" in v]
    if thr:
        sub["ANCHTHREAD"] = pct(float(np.mean(thr)), 0)
    oth = [v["other_systems"] for k, v in ANCH["masked"].get("retrieval", {}).items()
           if k.split("/")[0] in ("state", "detail", "concat", "routed") and "other_systems" in v]
    if oth:
        sub["ANCHOTHERSYS"] = pct(float(np.mean(oth)), 0)


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def pp(x):
    """Percentage points, signed, with no sign on zero."""
    v = round(100 * x)
    return "0" if v == 0 else f"{v:+d}"


def pct_small(x):
    """A percentage with one decimal below 1%, none above."""
    return f"{100 * x:.1f}\\%" if abs(x) < 0.01 else pct(x, 0)


def rng_(vals, fmt="{:.2f}", sep="--"):
    return f"{fmt.format(min(vals))}{sep}{fmt.format(max(vals))}"


SYSNAME = {"slack": "Slack (chat)", "gmail": "Gmail (email)", "salesforce": "Salesforce (CRM)",
           "zendesk": "Zendesk (support)", "gong": "Gong (calls)", "jira": "Jira (issues)", "calendar": "Google Calendar"}


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def fill_corpus(sub):
    """Table 1 and the counts of Section 5, from results/corpus.json (cwm/corpus_stats.py)."""
    C = load("corpus.json")
    if not C:
        return
    d1 = C["day1"]
    rows = ["\\begin{tabular}{lrrr}", "\\toprule", "System & Records & Linked to a customer & Share linked \\\\",
            "\\midrule"]
    for k, v in sorted(d1["by_system"].items(), key=lambda kv: -kv[1]["records"]):
        rows.append(f"{SYSNAME.get(k, k)} & {thousands(v['records'])} & {thousands(v['linked'])} & "
                    f"{pct(v['linked'] / v['records'], 0)} \\\\")
    rows += ["\\midrule", f"All & {thousands(d1['records'])} & {thousands(d1['linked'])} & "
             f"{pct(d1['linked'] / d1['records'], 0)} \\\\", "\\bottomrule", "\\end{tabular}"]
    sub["DATATABLE"] = "\n".join(rows)
    sub["CORPUSRECORDS"] = thousands(d1["records"])
    sub["CORPUSLINKED"] = thousands(d1["linked"])
    sub["CORPUSSHARE"] = pct(d1["linked"] / d1["records"], 0)
    sub["CORPUSUNLINKED"] = pct(1 - d1["linked"] / d1["records"], 0)
    sub["CORPUSCUSTOMERS"] = str(C["customers"])
    sub["CORPUSSTATES"] = str(C["customer_quarter_states"])
    sub["CORPUSNEW"] = thousands(C["day2"]["new_linked_known_customers"])
    bm = C["day2"]["new_linked_by_month"]
    sub["CORPUSNEWQFOUR"] = thousands(sum(v for k, v in bm.items() if "2026-10" <= k <= "2026-12"))
    sub["CORPUSNEWJAN"] = thousands(bm.get("2027-01", 0))
    sub["CORPUSNEWLATER"] = thousands(sum(v for k, v in bm.items() if k > "2027-01"))
    sub["CORPUSFIRSTQ"], sub["CORPUSLASTQ"] = (f"{q[:4]}~Q{q[-1]}" for q in (C["first_quarter"], C["last_quarter"]))


def fill_actions(sub):
    """Section 6.11: actions read from records, action-conditioned dynamics, event windows, decision simulation.
    Error changes are printed as a share of the reference model's error, positive meaning worse."""
    AC = load("actions.json")
    if not AC:
        return
    T, D, EV, SI = AC.get("table", {}), AC.get("dynamics", {}), AC.get("events", {}), AC.get("simulate", {})
    names = {"call": "CALL", "leader": "LEADER", "crm_task": "CRMTASK", "eng_escalation": "ENGESC",
             "support_reply": "SUPPORT", "outbound_email": "OUTBOUND"}
    for a, tag in names.items():
        sub[f"ACTSHARE{tag}"] = pct(T["action_share"][a], 0)
    sub["ACTMONTHS"] = str(T["month_states"]); sub["ACTTRANS"] = str(T["transitions"])
    sub["SUPPORTAGENTS"] = WORDS.get(T["support_agents"], str(T["support_agents"]))
    sub["ACTTRANSPROSPECTS"] = str(T["transitions_with_prospects"])
    sub["ACTPROGRESSRATE"] = pct(T["outcome_rate_next_month"]["deal_progress"], 0)
    sub["ACTESCRATE"] = pct(T["outcome_rate_next_month"]["escalation"], 0)
    sub["ACTCALLPROGNEXT"] = pct(T["after_action"]["call"]["deal_progress|next_month"][0], 0)
    sub["ACTNOCALLPROGNEXT"] = pct(T["after_action"]["call"]["deal_progress|next_month"][1], 0)
    sub["ACTSUPTICKET"] = pct(T["support_reply_given_ticket"][0], 0)
    sub["ACTSUPNOTICKET"] = pct(T["support_reply_given_ticket"][1], 0)
    sub["ACTTICKETAUC"] = f"{T['ticket_this_month_auroc_for_escalation_next']:.2f}"
    reps_ = [("good_month", "Good views"), ("bad", "Bad views"), ("minilm", "Pretrained encoder"),
             ("good_month+minilm", "Two channels")]
    err = lambda gain: -100 * gain  # noqa: E731  (a gain relative to a reference -> change in error, % of it)
    if D:
        o = D["outcomes"]
        rows = ["\\setlength{\\tabcolsep}{4.5pt}\\begin{tabular}{lcccccc}", "\\toprule",
                "& \\multicolumn{6}{c}{Squared error of next month's state, relative to predicting no change} \\\\",
                "\\cmidrule(lr){2-7}",
                "& & & \\multicolumn{2}{c}{+ this month's actions} & \\multicolumn{2}{c}{+ next month's} \\\\",
                "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
                "Representation & shrink & state & actual & shuffled & actions & system mix \\\\", "\\midrule"]
        for k, n in reps_:
            r = D[k]
            rows.append(f"{n} & {signed(-r['shrink'])} & {signed(-r['state'])} & {signed(-r['state+actions'])} & "
                        f"{signed(-r['state+shuffled_actions'])} & {signed(-r['state+next_actions'])} & "
                        f"{signed(-r['state+next_mix'])} \\\\")
        rows += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{8pt}", "",
                 "\\setlength{\\tabcolsep}{4.5pt}\\begin{tabular}{lcccc}", "\\toprule",
                 "& \\multicolumn{4}{c}{Next month's outcome from this month, AUROC} \\\\", "\\cmidrule(lr){2-5}",
                 "& \\multicolumn{2}{c}{Deal advances or is won} & \\multicolumn{2}{c}{Urgent or high ticket} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                 "Representation & state & + actions & state & + actions \\\\", "\\midrule"]
        for k, n in reps_:
            cells = []
            for yk in ("deal_progress", "escalation"):
                v = o[yk][k]
                lo, hi = v["difference_ci"]
                star = "$^{*}$" if lo > 0 or hi < 0 else ""
                cells += [f"{v['state']:.2f}", f"{v['state+actions']:.2f}{star}"]
            rows.append(f"{n} & " + " & ".join(cells) + " \\\\")
        rows += ["\\midrule",
                 f"Actions alone & \\multicolumn{{2}}{{c}}{{{o['deal_progress']['actions']:.2f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{o['escalation']['actions']:.2f}}} \\\\",
                 f"Support reply indicator & \\multicolumn{{2}}{{c}}{{{o['deal_progress']['support_reply_only']:.2f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{o['escalation']['support_reply_only']:.2f}}} \\\\",
                 "\\bottomrule", "\\end{tabular}"]
        sub["ACTDYNTABLE"] = "\n".join(rows)
        lag = [err(D[k]["actions_vs_state"]) for k, _ in reps_]
        shuf = [err(D[k]["shuffled_vs_state"]) for k, _ in reps_]
        mix = [err(D[k]["mix_vs_state"]) for k, _ in reps_]
        nxt = [100 * D[k]["next_actions_vs_state"] for k, _ in reps_]
        nmix = [100 * D[k]["next_mix_vs_state"] for k, _ in reps_]
        sub["ACTLAGRANGE"] = f"{min(lag):+.1f}\\% to {max(lag):+.1f}\\%"
        sub["ACTSHUFRANGE"] = f"{min(shuf):.1f}--{max(shuf):.1f}\\%"
        sub["ACTMIXRANGE"] = f"{min(mix):+.1f}\\% to {max(mix):+.1f}\\%"
        sub["ACTNEXTRANGE"] = f"{min(nxt):.0f}--{max(nxt):.0f}\\%"
        sub["ACTNEXTMIXRANGE"] = f"{min(nmix):.0f}--{max(nmix):.0f}\\%"
        nice = dict(reps_)

        def sig_text(key, what="interval"):
            """Which representations' 95% intervals exclude zero, in words."""
            ex = [nice[k].lower() for k, _ in reps_ if D[k][key + "_ci"][0] > 0 or D[k][key + "_ci"][1] < 0]
            if not ex:
                return f"every {what} includes zero"
            if len(ex) == len(reps_):
                return f"every {what} excludes zero"
            return f"the {what} excludes zero only for " + " and ".join(ex)
        sub["ACTNEXTMIXSIG"] = sig_text("next_mix_vs_state")
        beyond = [100 * D[k]["next_actions_beyond_next_mix"] for k, _ in reps_]
        sub["ACTNEXTBEYONDRANGE"] = f"{min(beyond):.0f}--{max(beyond):.0f}\\%"
        sub["ACTNEXTBEYONDSIG"] = sig_text("next_actions_beyond_next_mix")
        g = D["good_month"]
        lo, hi = g["actions_vs_state_ci"]
        sub["ACTGOODLAG"] = f"{err(g['actions_vs_state']):+.1f}\\%"
        sub["ACTGOODLAGCI"] = f"{err(hi):+.1f}\\% to {err(lo):+.1f}\\%"
        sub["ACTGOODNEXT"] = f"{100 * g['next_actions_vs_state']:.1f}\\%"
        lo, hi = g["next_actions_vs_state_ci"]
        sub["ACTGOODNEXTCI"] = f"{100 * lo:.1f}--{100 * hi:.1f}\\%"
        fs = D["action_from_state_auroc"]
        for k, tag in (("good_month", "GOOD"), ("bad", "BAD"), ("minilm", "MINI"), ("good_month+minilm", "TWO")):
            sub[f"ACTFROMSTATE{tag}"] = rng_(list(fs[k].values()))
        for yk, tag in (("deal_progress", "PROG"), ("escalation", "ESC")):
            sub[f"ACTAUC{tag}STATERANGE"] = rng_([o[yk][k]["state"] for k, _ in reps_])
            sub[f"ACTAUC{tag}DIFFRANGE"] = rng_([o[yk][k]["difference"] for k, _ in reps_], "{:+.2f}", " to ")
            nsig = sum(o[yk][k]["difference_ci"][0] > 0 for k, _ in reps_)
            sub[f"ACTAUC{tag}NSIG"] = WORDS[nsig]
            sub[f"ACTAUC{tag}ACT"] = f"{o[yk]['actions']:.2f}"
            sub[f"ACTSUPONLY{tag}"] = f"{o[yk]['support_reply_only']:.2f}"
            sub[f"ACTAUC{tag}STATEMAX"] = f"{max(o[yk][k]['state'] for k, _ in reps_):.2f}"
        gaps = [o[yk][k]["all_actions_minus_support_reply"] for yk in o for k, _ in reps_]
        cis = [o[yk][k]["all_actions_minus_support_reply_ci"] for yk in o for k, _ in reps_]
        sub["ACTSUPVSALLRANGE"] = rng_(gaps, "{:+.2f}", " to ")
        nex = sum(lo > 0 or hi < 0 for lo, hi in cis)
        sub["ACTSUPVSALLSIG"] = "every interval including zero" if nex == 0 else \
            f"{WORDS[nex]} of the {WORDS[len(cis)]} intervals excluding zero"
        sub["ACTSUPHELDOUTESC"] = f"{o['escalation']['support_reply_only_heldout']:.2f}"
    if EV:
        for k, tag in (("call", "CALL"), ("eng_escalation", "ENGESC"), ("leader", "LEADER")):
            e = EV[k]
            sub[f"EV{tag}N"] = str(e["events"]); sub[f"EV{tag}CUST"] = str(e["customers"])
            sub[f"EV{tag}MATCHN"] = str(e["matched"])
            sub[f"EV{tag}SHIFT"] = f"{e['shift_event']:.3f}"; sub[f"EV{tag}MATCH"] = f"{e['shift_matched']:.3f}"
            sub[f"EV{tag}RECEV"] = f"{e['records_before_event']:.0f}"
            sub[f"EV{tag}RECMATCH"] = f"{e['records_before_matched']:.0f}"
            lo, hi = e["difference_ci"]
            sub[f"EV{tag}DIFF"] = f"{e['difference']:+.3f} (interval {lo:+.3f} to {hi:+.3f})"
            lo, hi = e["adjusted_difference_ci"]
            v = e["adjusted_difference"]
            sub[f"EV{tag}ADJ"] = f"{0 if abs(v) < 5e-4 else v:+.3f} (interval {lo:+.3f} to {hi:+.3f})".replace("+0.000", "0.000")
        pr = EV["prediction"]
        sub["EVPRED"] = f"{err(pr['action_vs_state']):+.2f}\\%"
        sub["EVPAIRS"] = str(pr["pairs"])
    if SI:
        sub["SIMTROUBLE"] = str(SI["trouble"]); sub["SIMINTERV"] = str(SI["intervened_in_trouble"])
        sub["SIMNOT"] = str(SI["trouble"] - SI["intervened_in_trouble"]); sub["SIMK"] = str(SI["k"])
        sub["SIMBOOT"] = str(SI["bootstrap"]); sub["SIMPOOL"] = str(SI["pool"]); sub["SIMCUST"] = WORDS.get(SI["customers_in_trouble"], str(SI["customers_in_trouble"]))
        sub["SIMMAXCUST"] = str(SI["most_trouble_months_one_customer"])
        sub["SIMBEFORE"] = str(SI["intervention_before_ticket"]); sub["SIMAFTER"] = str(SI["intervention_after_ticket"])
        P = SI["intervention_from_state_auroc"]
        sub["SIMPROPGOOD"] = f"{P['good_month']:.2f}"; sub["SIMPROPBAD"] = f"{P['bad']:.2f}"
        sub["SIMPROPRANGE"] = rng_(list(P.values()))
        bt, be = SI["escalation"]["balance_in_trouble"], SI["escalation"]["balance_urgent_ticket_this_month"]
        sub["SIMBALTROUBLEON"], sub["SIMBALTROUBLEOFF"] = pct_small(bt[0]), pct_small(bt[1])
        sub["SIMBALESCON"], sub["SIMBALESCOFF"] = pct_small(be[0]), pct_small(be[1])
        sfx = ["", "_bad", "_minilm", "_good_month+minilm"]
        num = lambda x: (f"{100 * x:.1f}" if x < 0.01 else f"{100 * x:.0f}")  # noqa: E731
        for key, tag in (("balance_in_trouble", "TROUBLE"), ("balance_urgent_ticket_this_month", "ESC")):
            for j, side in ((0, "ON"), (1, "OFF")):
                v = [SI["escalation"][key + x][j] for x in sfx]
                sub[f"SIMBAL{tag}{side}R"] = f"{num(min(v))}--{num(max(v))}\\%"
        rows = ["\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lcccccc}", "\\toprule",
                "& \\multicolumn{3}{c}{Deal advances or is won next month} & \\multicolumn{3}{c}{Urgent or high ticket next month} \\\\",
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                "Estimate & intervene & not & difference & intervene & not & difference \\\\", "\\midrule"]
        lab = [("observed_in_trouble", f"Observed in trouble ({SI['intervened_in_trouble']} vs "
                                       f"{SI['trouble'] - SI['intervened_in_trouble']})"),
               ("case", f"Similar states, good views ($k = {SI['k']}$)"),
               ("case_bad", "Similar states, bad views"),
               ("case_minilm", "Similar states, pretrained"),
               ("case_good_month+minilm", "Similar states, two channels"),
               ("unmatched", "All months of other customers"),
               ("model", "Outcome model (good views)")]
        for key, n in lab:
            cells = []
            for yk in ("deal_progress", "escalation"):
                v = SI[yk][key]
                ci = SI[yk].get(key + "_difference_ci")
                dcell = pp(v[0] - v[1]) + (f" [{pp(ci[0])}, {pp(ci[1])}]" if ci else "")
                cells += [pct(v[0], 0), pct(v[1], 0), dcell]
            rows.append(f"{n} & " + " & ".join(cells) + " \\\\")
        rows += ["\\bottomrule", "\\end{tabular}"]
        sub["SIMTABLE"] = "\n".join(rows)
        case_keys = ("case", "case_bad", "case_minilm", "case_good_month+minilm")
        nzero = 0
        for yk, tag in (("deal_progress", "PROG"), ("escalation", "ESC")):
            for key, t2 in (("case", "CASE"), ("unmatched", "UNM"), ("model", "MODEL")):
                v = SI[yk][key]
                sub[f"SIM{tag}{t2}"] = pp(v[0] - v[1])
            ci = SI[yk]["case_difference_ci"]
            sub[f"SIM{tag}CASECI"] = f"{pp(ci[0])} to {pp(ci[1])}"
            ds = [SI[yk][k][0] - SI[yk][k][1] for k in case_keys]
            sub[f"SIM{tag}CASERANGE"] = f"{pp(min(ds))} to {pp(max(ds))}"
            nzero += sum(SI[yk][k + "_difference_ci"][0] <= 0 <= SI[yk][k + "_difference_ci"][1] for k in case_keys)
        sub["SIMCASENZERO"] = WORDS[nzero]
    # one encoder or head per timing: E2 in prh.json times two per-system encoders and their evaluation together
    CK = AC.get("checks", {})
    if CK:
        sub["CHKLATE"] = WORDS.get(CK["transitions_into_later_months"], str(CK["transitions_into_later_months"])).capitalize()
        sub["CHKLATETROUBLE"] = WORDS.get(CK["trouble_into_later_months"], str(CK["trouble_into_later_months"]))
        o = CK["observed_in_trouble_without_them"]
        sub["CHKPROGON"], sub["CHKPROGOFF"] = pct(o["deal_progress"][0], 0), pct(o["deal_progress"][1], 0)
        sub["CHKESCON"], sub["CHKESCOFF"] = pct(o["escalation"][0], 0), pct(o["escalation"][1], 0)
    secs = [v for f in ("prh.json", "extras_a.json", "extras_b.json", "extras_c.json", "extras_d.json", "extras_e.json",
                        "v2_pretrained.json", "v2_monthly.json", "acl.json")
            for v in _seconds({k: x for k, x in load(f).items() if not (f == "prh.json" and k == "E2")})]
    if secs:
        sub["TRAINMINLO"] = f"{min(secs) / 60:.0f}"; sub["TRAINMINHI"] = f"{max(secs) / 60:.0f}"
    mem = [v for f in sorted(os.listdir(RES)) if f.startswith("memory") and f.endswith(".json") for v in _seconds(load(f))]
    if mem:
        sub["MEMREADERMIN"] = f"{np.ceil(max(mem) / 60):.0f}"


def _seconds(d):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ("train_seconds", "seconds") and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out += _seconds(v)
    return out


def acl_policy_table():
    R = ACL["readers"]
    systems = ["slack", "gmail", "zendesk", "salesforce", "gong", "jira", "calendar"]
    rows = ["\\begin{tabular}{p{0.5\\linewidth}p{0.42\\linewidth}}", "\\toprule", "Records & Who may read them \\\\",
            "\\midrule"]
    rows += [f"{a} & {b} \\\\" for a, b in ACL["policy"]]
    rows += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{6pt}", "",
             "\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{l" + "c" * (len(systems) + 1) + "}", "\\toprule",
             "Reader & all & " + " & ".join(systems) + " \\\\", "\\midrule"]
    for r in RTAG:
        if r in R:
            v = R[r]["visible_share_by_system"]
            rows.append(f"{RNAME[r]} & {100 * R[r]['visible_share']:.0f}\\% & "
                        + " & ".join(f"{100 * v.get(sy, 0):.0f}\\%" for sy in systems) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


def acl_leak_table():
    R = ACL["readers"]
    rows = ["\\setlength{\\tabcolsep}{5pt}\\begin{tabular}{llcccc}", "\\toprule",
            "& & \\multicolumn{2}{c}{Codename search, precision@10} & \\multicolumn{2}{c}{Secret sentence, precision@5} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Reader & Configuration & with secrets & without & with secrets & without \\\\", "\\midrule"]
    for r in ("support_engineer", "software_engineer"):
        if r not in R or "leak" not in R[r]:
            continue
        L = R[r]["leak"]
        for conf, name in (("open", "no access control"), ("filtered", "search and states filtered"),
                           ("built_in", "\\textbf{built in}")):
            first = RNAME[r] if conf == "open" else ""
            w, wo = L[conf]["with"], L[conf].get("without", {})
            rows.append(f"{first} & {name} & {w['codename_precision']:.2f} & {wo.get('codename_precision', float('nan')):.2f} & "
                        f"{w['secret_precision@5']:.2f} & {wo.get('secret_precision@5', float('nan')):.2f} \\\\")
        rows.append("\\addlinespace[3pt]")
    rows[-1] = "\\bottomrule"
    rows.append("\\end{tabular}")
    return "\n".join(rows)


def heldout_table():
    rows = ["\\setlength{\\tabcolsep}{6pt}\\begin{tabular}{lcccc}", "\\toprule",
            "& \\multicolumn{2}{c}{Precision@10, other systems} & \\multicolumn{2}{c}{Placed at the nearest state} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "Representation & customer & state & customer & state \\\\", "\\midrule"]
    for arm, name in (("good_s0", "\\textcolor{goodblue}{\\textbf{Good views}}"), ("bad_s0", "\\textcolor{badorange}{Bad views}"),
                      ("ntp_s0", "\\textcolor{ntpaqua}{Next-token}"), ("tfidf_lsa", "TF-IDF + LSA"),
                      ("random_init", "Random encoder")):
        if arm not in HO:
            continue
        m = HO[arm]
        cells = [m[METRICS["ACC"]], m[METRICS["ST"]], m["customer_acc"], m["state_acc"]]
        fmt = [f"\\textbf{{{v:.3f}}}" if arm == "good_s0" else f"{v:.3f}" for v in cells]
        rows.append(f"{name} & " + " & ".join(fmt) + " \\\\")
    P = V2["pretrained"].get("heldout", {})
    for arm, name in (("minilm_raw", "Pretrained encoder, frozen"), ("minilm_head", "Pretrained + good-view head")):
        if arm in P:
            m = P[arm]
            rows.append(f"{name} & {m[METRICS['ACC']]:.3f} & {m[METRICS['ST']]:.3f} & {m['customer_acc']:.3f} & "
                        f"{m['state_acc']:.3f} \\\\")
    rows.append("\\midrule")
    rows.append(f"Customer known, its busiest quarter & & & 1.000 & {HO['reference_customer_known_state_acc']:.3f} \\\\")
    rows.append(f"Largest customer / largest state & & & {HO['reference_majority_customer_acc']:.3f} & "
                f"{HO['reference_majority_state_acc']:.3f} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    return "\n".join(rows)


def seed_table():
    rows = ["\\begin{tabular}{llccccc}", "\\toprule",
            "Objective & Seed & Same customer & Same state & M$\\to$R & R$\\to$M & $\\rho_{\\text{state}}$ \\\\",
            "\\midrule"]
    for arm, name in (("good", "Good views"), ("bad", "Bad views"), ("ntp", "Next-token")):
        for s, m in runs(arm).items():
            rows.append(f"{name} & {s} & {m[METRICS['ACC']]:.3f} & {m[METRICS['ST']]:.3f} & {m[METRICS['MR']]:.3f} & "
                        f"{m[METRICS['RM']]:.3f} & {m[METRICS['RHO']]:.2f} \\\\")
        rows.append("\\addlinespace[2pt]")
    rows[-1] = "\\bottomrule"
    rows.append("\\end{tabular}")
    return "\n".join(rows)


def state_table():
    if not ST:
        return "(not run)"
    kinds = ("tickets", "priority", "call", "stage")
    head = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{l" + "c" * 10 + "}\n\\toprule\n"
            "& \\multicolumn{5}{c}{This quarter} & \\multicolumn{5}{c}{Next quarter} \\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
            "Representation & tickets & priority & call & stage & mean & tickets & priority & call & stage & mean \\\\\n"
            "\\midrule")
    rows = [head]
    for arm, name in (("good", "Good views"), ("bad", "Bad views"), ("ntp", "Next-token"), ("tfidf", "TF-IDF + LSA"),
                      ("random", "Random encoder")):
        now = [ST[arm][k]["now_acc"] for k in kinds]
        nxt = [ST[arm][k]["next_acc"] for k in kinds]
        rows.append(f"{name} & " + " & ".join(f"{v:.2f}" for v in now) + f" & {np.mean(now):.2f} & "
                    + " & ".join(f"{v:.2f}" for v in nxt) + f" & {np.mean(nxt):.2f} \\\\")
    rows.append("\\midrule")
    maj_now = [ST["good"][k]["now_majority"] for k in kinds]
    maj_nxt = [ST["good"][k]["next_majority"] for k in kinds]
    rows.append("Majority class & " + " & ".join(f"{v:.2f}" for v in maj_now) + f" & {np.mean(maj_now):.2f} & "
                + " & ".join(f"{v:.2f}" for v in maj_nxt) + f" & {np.mean(maj_nxt):.2f} \\\\")
    orc = [ST["good"][k]["next_oracle_persistence"] for k in kinds]
    rows.append("Oracle persistence & \\multicolumn{5}{c}{---} & " + " & ".join(f"{v:.2f}" for v in orc)
                + f" & {np.mean(orc):.2f} \\\\")
    rows.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(rows)


def memory_table():
    def collect(*ds, key):
        vals = []
        for d in ds:
            for k, v in d.get("runs", {}).items():
                if k.split("/")[0] == key:
                    vals.append(v)
        return vals

    def mean(vals, f):
        return np.mean([f(v) for v in vals]) if vals else float("nan")

    if all(MEMQ.values()):        # version 3: retrieval by content within the question's quarter
        G, L4, DU, RO = MEMQ["good"], MEMQ["minilm"], MEMQ["dual"], MEMQ["route"]
        jc = collect(G, key="closed_book"); jt = collect(G, key="top16_pretrained")
    else:
        L4, DU, RO = load("memory2_minilm4.json"), load("memory2_dual.json"), load("memory2_route.json")
        jc = collect(M2, M2S, key="closed_book"); jt = collect(M2, M2S, key="top16_pretrained")
    jm = collect(M2, key="memorilla_pretrained"); jn = collect(M2, key="memorilla_no_pretrain")
    lc = collect(L4, key="closed_book"); lt = collect(L4, key="top16_pretrained")
    dc = collect(DU, key="closed_book"); dt = collect(DU, key="top16_pretrained")
    rt = collect(RO, key="top16_pretrained")
    ov = lambda v: v["overall"]  # noqa: E731
    st = lambda v: v["growth"]["stale_store"]  # noqa: E731
    gr = lambda v: v["growth"]["grown_store"]  # noqa: E731
    cbn = lambda v: v["growth"]["closed_book"]  # noqa: E731
    rows = ["\\begin{tabular}{llcccc}", "\\toprule",
            "& & & & \\multicolumn{2}{c}{Next quarter: store} \\\\", "\\cmidrule(lr){5-6}",
            "Store & Reader & Seeds & Accuracy & not updated & updated \\\\", "\\midrule",
            f"Good views (state) & closed book (no store) & {len(jc)} & {mean(jc, ov):.3f} & \\multicolumn{{2}}{{c}}{{{mean(jc, cbn):.3f}}} \\\\",
            f" & 16 tokens over the 16 retrieved rows & {len(jt)} & {mean(jt, ov):.3f} & {mean(jt, st):.3f} & {mean(jt, gr):.3f} \\\\",
            f" & 16 tokens over the whole store & {len(jm)} & {mean(jm, ov):.3f} & {mean(jm, st):.3f} & {mean(jm, gr):.3f} \\\\",
            f" & the same, not pre-trained & {len(jn)} & {mean(jn, ov):.3f} & {mean(jn, st):.3f} & {mean(jn, gr):.3f} \\\\",
            "\\addlinespace[2pt]",
            f"Pretrained (detail) & closed book (no store) & {len(lc)} & {mean(lc, ov):.3f} & \\multicolumn{{2}}{{c}}{{{mean(lc, cbn):.3f}}} \\\\",
            f" & 16 tokens over the 16 retrieved rows & {len(lt)} & {mean(lt, ov):.3f} & {mean(lt, st):.3f} & {mean(lt, gr):.3f} \\\\",
            "\\addlinespace[2pt]",
            f"Two channels, concatenated & closed book (no store) & {len(dc)} & {mean(dc, ov):.3f} & \\multicolumn{{2}}{{c}}{{{mean(dc, cbn):.3f}}} \\\\",
            f" & 16 tokens over the 16 retrieved rows & {len(dt)} & {mean(dt, ov):.3f} & {mean(dt, st):.3f} & {mean(dt, gr):.3f} \\\\",
            "\\addlinespace[2pt]",
            f"Two channels, routed & 16 tokens over the 16 retrieved rows & {len(rt)} & {mean(rt, ov):.3f} & {mean(rt, st):.3f} & {mean(rt, gr):.3f} \\\\",
            "\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = fill(open(src).read())
    open(dst, "w").write(out)
    if os.environ.get("FILL_MAP"):
        json.dump(LAST_SUB, open(os.environ["FILL_MAP"], "w"), indent=0)
    left = [w for w in ("E1GOOD", "E1BAD", "E1NTP", "MASK", "NOISE", "APPSEED", "APPSTATE", "APPMEM", "HOGOOD", "HOTF",
                        "HOBAD", "HONTP", "HOREF", "HONTEST", "HOTABLE", "E3B", "ACL", "PTRAW", "PTHEAD", "MASKEMP",
                        "MON", "TIME", "DYN", "FCGOOD", "FCTWO", "FCBAD", "STATE2", "CALL", "NOWTWO", "NOWMINI",
                        "MEM2", "MEMGOOD", "MEMDUAL", "MEMMINI", "MEMROUTE", "SHRINK", "TUNED", "PRIM", "DRIFT", "MISR", "SIT", "ACL2",
                        "TROUBLE", "SCOPE", "MONTABLE", "ANCH", "ACT", "SIM", "EVP", "EVC", "EVE", "EVL", "TRAINMIN",
                        "CORPUS", "DATATABLE", "EONE", "MEMREADER", "SUPPORTAGENTS", "CHK") if w in out]
    print("unfilled:", left or "none")
    if left:
        sys.exit(1)
