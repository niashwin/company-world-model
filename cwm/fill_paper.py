"""Fill the numbers and appendix tables of paper/main.tex from results/*.json, so that every number in the paper
comes from a results file. Placeholders are ALL-CAPS tokens (E1GOODACC, MASKGOODPCT, APPSEEDTABLE, ...).

python fill_paper.py paper/main.in.tex paper/main.tex
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("CWM_RESULTS", os.path.join(ROOT, "results"))


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


PRH, XA, XB, ST, M2, M2S, M2L, E3B = (load(n) for n in ("prh.json", "extras_a.json", "extras_b.json", "state.json",
                                                         "memory2.json", "memory2_seeds.json", "memory2_minilm.json",
                                                         "e3b.json"))
ACL = load("acl.json")
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
            f"With {pct(ps[0], 0)} of the links wrong, same-customer precision is {pct(lo[METRICS['ACC']], 0)} "
            f"(from {pct(clean[METRICS['ACC']], 0)} with clean links) and state precision {pct(lo[METRICS['ST']], 0)} "
            f"(from {pct(clean[METRICS['ST']], 0)}). With {pct(ps[-1], 0)} wrong, they are "
            f"{pct(hi[METRICS['ACC']], 0)} and {pct(hi[METRICS['ST']], 0)}. Even then, same-customer precision is far "
            f"above that of bad views and next-token prediction trained on clean links ({pct(bad, 0)} and "
            f"{pct(ntp, 0)}). Wrong links blur the space; they do not break it.")
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
            if "leak" in R[r]:
                sub[f"ACL{tag}CODECHANCE"] = pct(R[r]["leak"]["built_in"]["with"]["codename_chance"], 0)
                sub[f"ACL{tag}SECRETCHANCE"] = pct(R[r]["leak"]["built_in"]["with"]["secret_chance@5"], 0)
        ni = ACL.get("non_interference", {})
        sub["ACLNIDIFF"] = f"{max(ni.get('max_abs_diff_embeddings', float('nan')), ni.get('max_abs_diff_queries', 0.0)):g}"
        sub["ACLPOLICYTABLE"] = acl_policy_table()
        sub["ACLSECRETSENTENCE"] = ACL["secret"].capitalize() + "."
        sub["ACLLEAKTABLE"] = acl_leak_table()
    sub["APPSEEDTABLE"] = seed_table()
    sub["APPSTATETABLE"] = state_table()
    sub["APPMEMTABLE"] = memory_table()
    for k in sorted(sub, key=len, reverse=True):
        text = text.replace(k, sub[k])
    return text


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

    jc = collect(M2, M2S, key="closed_book"); jt = collect(M2, M2S, key="top16_pretrained")
    jm = collect(M2, key="memorilla_pretrained"); jn = collect(M2, key="memorilla_no_pretrain")
    lc = collect(M2L, key="closed_book"); lt = collect(M2L, key="top16_pretrained")
    ov = lambda v: v["overall"]  # noqa: E731
    rows = ["\\begin{tabular}{llcc}", "\\toprule",
            "Store & Reader & Seeds & Accuracy \\\\", "\\midrule",
            f"Good-view encoder & closed book (no store) & {len(jc)} & {mean(jc, ov):.3f} \\\\",
            f" & 16 tokens over the 16 retrieved rows, pre-trained & {len(jt)} & {mean(jt, ov):.3f} \\\\",
            f" & 16 tokens over the whole store, pre-trained & {len(jm)} & {mean(jm, ov):.3f} \\\\",
            f" & 16 tokens over the whole store, not pre-trained & {len(jn)} & {mean(jn, ov):.3f} \\\\",
            "\\addlinespace[2pt]",
            f"Pretrained sentence encoder & closed book (no store) & {len(lc)} & {mean(lc, ov):.3f} \\\\",
            f" & 16 tokens over the 16 retrieved rows, pre-trained & {len(lt)} & {mean(lt, ov):.3f} \\\\",
            "\\midrule",
            "\\multicolumn{4}{l}{\\emph{Questions about the next quarter (" +
            f"{M2.get('growth_setup', {}).get('new_questions', '?')} questions, majority "
            f"{M2.get('growth_setup', {}).get('majority_on_new', float('nan')):.3f})" + "}} \\\\",
            f"Good-view encoder & closed book & {len(jc)} & {mean(jc, lambda v: v['growth']['closed_book']):.3f} \\\\",
            f" & retrieved rows, store not updated & {len(jt)} & {mean(jt, lambda v: v['growth']['stale_store']):.3f} \\\\",
            f" & retrieved rows, store updated & {len(jt)} & {mean(jt, lambda v: v['growth']['grown_store']):.3f} \\\\",
            "\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = fill(open(src).read())
    open(dst, "w").write(out)
    left = [w for w in ("E1GOOD", "E1BAD", "E1NTP", "MASK", "NOISE", "APPSEED", "APPSTATE", "APPMEM", "HOGOOD", "HOTF",
                        "HOBAD", "HONTP", "HOREF", "HONTEST", "HOTABLE", "E3B", "ACL") if w in out]
    print("unfilled:", left or "none")
