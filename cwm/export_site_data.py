"""Export the data of the project website to website/data/site-data.js.

The website shows the paper's own numbers: every table comes from the filled paper (paper/main.tex, written by
cwm/fill_paper.py from results/), and the headline figures come from the same substitution map. Three figures need
more than the paper prints, and this script computes them from the cached embeddings and the data:

  map        two-dimensional t-SNE maps of the 3,000 linked records of Fig. 2a-c, in six spaces
  hero       the records of one customer-quarter across the seven systems (times only, no text)
  decisions  for each customer-month in trouble, the similar months of other customers that the decision
             simulation of Fig. 5 compares, with their outcomes

Run after the experiments and `make paper`:  python cwm/export_site_data.py
No record text leaves the repository: the website gets times, systems, anonymous customer numbers and results.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_prh import D1, OUT, ROOT  # noqa: E402

SITE = os.path.join(ROOT, "website", "data", "site-data.js")
TEMPLATE, PAGE = os.path.join(ROOT, "website", "template.html"), os.path.join(ROOT, "website", "index.html")
TABLES = ["tab:data", "tab:e1", "tab:heldout", "tab:acl", "tab:aclrescue", "tab:twochannel", "tab:memory2",
          "tab:anchor", "tab:months", "tab:time", "tab:primitives", "tab:actions", "tab:simulate"]
SPACES = [("good", "emb_good_s0.npy", "Good views"), ("bad", "emb_bad_s0.npy", "Bad views"),
          ("ntp", "emb_ntp_s0.npy", "Next-token prediction"), ("tfidf", "emb_tfidf.npy", "TF-IDF + LSA"),
          ("minilm", "minilm_day1.npy", "Pretrained encoder, frozen"),
          ("minilm_head", "emb_minilm_head.npy", "Pretrained + good-view head")]


# ------------------------------------------------------------------------------------------------ LaTeX to text
SYMBOLS = {"alpha": "\u03b1", "rho": "\u03c1", "lambda": "\u03bb", "mu": "\u03bc", "sigma": "\u03c3", "tau": "\u03c4",
           "Delta": "\u0394", "pm": "\u00b1", "to": "\u2192", "le": "\u2264", "ge": "\u2265", "times": "\u00d7",
           "approx": "\u2248", "cdot": "\u00b7", "infty": "\u221e", "sim": "~"}
LABELS = {}          # label -> number as LaTeX prints it, from paper/main.aux (set in main)


def math_text(m):
    """The text of an inline formula: Greek letters and symbols, R^2 as R², subscripts as plain words."""
    m = m.replace("\\,", "\u2009")
    m = re.sub(r"\\(?:text|mathrm|mathit|textrm)\{([^}]*)\}", r"\1", m)
    m = re.sub(r"\^\{?2\}?", "\u00b2", m).replace("^{*}", "*")
    m = re.sub(r"_\{([^}]*)\}", r" \1", m)
    m = re.sub(r"\\([a-zA-Z]+)", lambda k: SYMBOLS.get(k.group(1), ""), m)
    return re.sub(r"(^|[\s(=])-", "\\1\u2212", m)


def ref(lab):
    return LABELS.get(lab, "?")


def plain(s):
    """LaTeX cell or phrase to plain text: Unicode symbols, resolved references, curly quotes, no markup."""
    s = s.replace("{\\scriptsize$\\,\\pm\\,$", " \u00b1 ").replace("$\\,\\pm\\,$", " \u00b1 ").replace("{,}", ",")
    s = re.sub(r"\\figref\{([^}]*)\}", lambda m: "Figure\u00a0" + ref(m.group(1)), s)
    s = re.sub(r"\\secref\{([^}]*)\}", lambda m: "\u00a7" + ref(m.group(1)), s)
    s = re.sub(r"\\(?:ref|autoref|eqref)\{([^}]*)\}", lambda m: ref(m.group(1)), s)
    s = s.replace("\\S ", "\u00a7").replace("\\S", "\u00a7")
    s = s.replace("``", "\u201c").replace("\'\'", "\u201d").replace("`", "\u2018")
    s = re.sub(r"\$([^$]*)\$", lambda m: math_text(m.group(1)), s)
    s = re.sub(r"\\textcolor\{[^}]*\}\{", "{", s)
    s = re.sub(r"\\(?:textbf|emph|textit|text|mbox|textsc|texttt)\{", "{", s)
    s = s.replace("\\%", "%").replace("\\,", "\u2009").replace("~", "\u00a0").replace("\\quad", "  ")
    s = s.replace("--", "\u2013").replace("\\&", "&")
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s).replace("{", "").replace("}", "")
    s = re.sub(r"(?<=\d)-(?=\d)", "\u2013", s)
    s = re.sub(r"(^|[\s(\[])-(?=\d)", "\\1\u2212", s)
    return re.sub(r"\s+", " ", s).strip()


def split_top(s, sep):
    """Split s on sep outside braces."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0 and s.startswith(sep, i) and (i == 0 or s[i - 1] != "\\" or sep == "\\\\"):
            out.append(cur)
            cur, i = "", i + len(sep)
            continue
        cur += ch
        i += 1
    out.append(cur)
    return out


def parse_tabular(body):
    """Rows of a LaTeX tabular as lists of {text, span}; 'rule' rows mark \\midrule."""
    body = re.sub(r"\\(?:toprule|bottomrule)", "", body)
    body = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", body)
    body = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", body)
    rows = []
    for raw in split_top(body, "\\\\"):
        if "\\midrule" in raw:
            rows.append("rule")
            raw = raw.replace("\\midrule", "")
        if not raw.strip():
            continue
        cells = []
        for cell in split_top(raw, "&"):
            m = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", cell, re.S)
            raw_cell = m.group(2) if m else cell
            c = {"text": plain(raw_cell), "span": int(m.group(1)) if m else 1}
            if "\\textbf{" in raw_cell:
                c["b"] = 1
            cells.append(c)
        rows.append(cells)
    while rows and rows[0] == "rule":
        rows.pop(0)
    return rows


def table_prefix(tex, i):
    """What the paper calls a table at position i: main text, Extended Data or Supplementary Information."""
    ed, si = tex.rfind("\\edsection"), tex.rfind("\\sisection")     # the last use, not the definition
    if si >= 0 and i > si:
        return "Supplementary Table"
    if ed >= 0 and i > ed:
        return "Extended Data Table"
    return "Table"


def paper_tables(tex):
    out = {}
    for lab in TABLES:
        i = tex.index("\\label{%s}" % lab)
        start = tex.rindex("\\begin{table}", 0, i)
        end = tex.index("\\end{table}", i)
        env = tex[start:end]
        cap = re.search(r"\\caption\{", env)
        caption = ""
        if cap:
            depth, j = 0, cap.end() - 1
            for k in range(j, len(env)):
                depth += env[k] == "{"
                depth -= env[k] == "}"
                if depth == 0:
                    caption = env[j + 1:k]
                    break
        tabs = re.findall(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", env, re.S)
        out[lab] = {"caption": plain(caption), "parts": [parse_tabular(t) for t in tabs], "prefix": table_prefix(tex, i)}
    return out


def aux_labels(aux):
    """Every label of the compiled paper, with the number that LaTeX prints for it (for example 6.11 or B)."""
    return dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))


# ------------------------------------------------------------------------------------------------ headline
HEADLINE = ["HOGOODCUST", "HOTFCUST", "HOBADCUST", "HONTPCUST", "PTRAWHO", "PTHEADHO", "E1GOODPCT", "E1BADPCT",
            "E1NTPPCT", "ACLNIDIFF", "ANCHPPLACESTATEACC", "ANCHPPLACEDETAILACC", "ANCHPCLOSEDACC", "ANCHPLINKACC",
            "ANCHPPAIRSTATEDETAIL", "SITGOOD", "SITCHANCE", "SITRANGE", "TRAINMINLO", "TRAINMINHI",
            "MEMREADERMIN", "SIMTROUBLE", "SIMINTERV", "SIMNOT", "SIMCUST", "SIMBALTROUBLEONR",
            "SIMBALTROUBLEOFFR", "SIMBALESCONR", "SIMBALESCOFFR", "SIMESCCASE", "SIMESCCASECI", "SIMPROPRANGE",
            "ACTLAGRANGE", "ACTNEXTRANGE", "ACTNEXTBEYONDRANGE", "ACTFROMSTATEGOOD", "CORPUSRECORDS",
            "CORPUSLINKED", "CORPUSCUSTOMERS", "CORPUSUNLINKED", "CORPUSNEW", "ACLCANARYN", "EVENGESCADJ",
            "EVCALLADJ", "EVLEADERADJ", "TUNEDMAXGAIN", "TUNEDNWORSE", "E1GOODNSEEDS", "ACLCOSTFINOWN", "ACL2FINHEAD",
            "ACLCOSTENGOWN", "ACL2ENGHEAD", "ANCHPPLACECONCATACC", "ANCHMHOSTATE", "ANCHMHODETAIL", "HOMAJCUST",
            "TRAINMINLO", "TRAINMINHI"]
REPO = "https://github.com/niashwin/company-world-model"
SITEURL = "https://niashwin.github.io/company-world-model/"


def headline(sub):
    return {k: plain(str(sub[k])) for k in HEADLINE if k in sub}


# ------------------------------------------------------------------------------------------------ map and hero
def state_map():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from data import load as load_corpus, window_key
    c = load_corpus(D1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(c.obs), 3000, replace=False)       # the records of Fig. 2a-c (cwm/make_figures.py)
    order = np.argsort(-np.bincount(c.account, minlength=len(c.accounts)), kind="stable")
    cust_no = {int(a): i + 1 for i, a in enumerate(order)}   # Customer 1 has the most linked records on day 1
    rank = {c.accounts[int(a)]: n for a, n in cust_no.items()}
    quarters = sorted({str(window_key(t, "quarter")) for t in c.ts})
    q_of = [str(window_key(t, "quarter")) for t in c.ts]
    coords = {}
    for key, fname, _ in SPACES:
        p = os.path.join(OUT, fname)
        if not os.path.exists(p):
            print("map: skipped", key, "(missing", fname + ")")
            continue
        X = np.load(p)[idx]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Y = TSNE(2, init="pca", perplexity=40, random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(X))
        Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
        coords[key] = np.round(Y * 1000).astype(int).ravel().tolist()
        print("map:", key, flush=True)
    sample = {
        "system": [int(c.system[i]) for i in idx],
        "customer": [cust_no[int(c.account[i])] for i in idx],
        "quarter": [quarters.index(q_of[i]) for i in idx],
    }
    # hero: the customer-quarter with records from the most systems among the sampled records, then the most records
    best, best_key = None, None
    for a in np.unique(c.account[idx]):
        for q in set(q_of[i] for i in idx if c.account[i] == a):
            members = [i for i in idx if c.account[i] == a and q_of[i] == q]
            key = (len({int(c.system[i]) for i in members}), len(members))
            if best_key is None or key > best_key:
                best, best_key = (int(a), q), key
    a, q = best
    allrec = [i for i in range(len(c.obs)) if c.account[i] == a and q_of[i] == q]
    t0, t1 = min(c.ts[i] for i in allrec), max(c.ts[i] for i in allrec)
    pos = {int(i): j for j, i in enumerate(idx)}
    hero = {
        "customer": cust_no[a], "quarter": q,
        "records": [[int(c.system[i]), round(float((c.ts[i] - t0) / max(1.0, t1 - t0)), 4), pos.get(int(i), -1)]
                    for i in sorted(allrec, key=lambda i: c.ts[i])],
        "start": float(t0), "end": float(t1),
        "pairs_across_systems": int(sum(1 for i in allrec for j in allrec if i < j and c.system[i] != c.system[j])),
    }
    return {"spaces": [[k, n] for k, _, n in SPACES if k in coords], "coords": coords, "sample": sample,
            "systems": list(c.systems), "quarters": quarters, "n_customers": len(c.accounts)}, hero, rank


# ------------------------------------------------------------------------------------------------ decisions
def decisions(rank):
    from sklearn.decomposition import PCA

    import run_actions as RA
    w = RA.World()
    _, S = RA.reps(w)
    a_, b_, g = np.array([a for a, b in w.trans]), np.array([b for a, b in w.trans]), w.g
    F = np.array([w.actions(a) for a in a_])
    I = ((F[:, RA.ACTIONS.index("call")] + F[:, RA.ACTIONS.index("eng_escalation")]) > 0).astype(int)
    Y = {k: np.array([w.outcomes(b)[k] for b in b_]).astype(int) for k in ("deal_progress", "escalation")}
    esc_now = np.array([w.esc[w.key(a)] > 0 for a in a_]).astype(int)
    T = np.array([w.trouble(a) for a in a_]).astype(int)
    cust_no = {str(a): rank[str(a)] for a in w.acc}          # the same customer numbers as the map
    month = [str(w.key(a)[1]) for a in a_]
    months = []
    for name, Sm in S.items():
        A = PCA(16, random_state=0).fit_transform(RA.zscore(Sm[a_]))
        sim = l2(A) @ l2(A).T
        for i in np.where(T == 1)[0]:
            other = np.where(g != g[i])[0]
            near = other[np.argsort(-sim[i, other], kind="stable")[:60]]
            on, off = near[I[near] == 1][:15], near[I[near] == 0][:15]
            months.append({"space": name, "i": int(i), "on": on.tolist(), "off": off.tolist()})
    items = [{"customer": cust_no[str(g[j])], "month": month[j], "intervened": int(I[j]), "trouble": int(T[j]),
              "urgent_now": int(esc_now[j]), "deal_next": int(Y["deal_progress"][j]),
              "urgent_next": int(Y["escalation"][j])} for j in range(len(a_))]
    troubled = [int(i) for i in np.where(T == 1)[0]]
    return {"items": items, "troubled": troubled, "neighbours": months,
            "spaces": [["good_month", "Good views"], ["bad", "Bad views"], ["minilm", "Pretrained encoder"],
                       ["good_month+minilm", "Two channels"]]}


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------------------------------------ main
def main():
    import fill_paper as FP
    tex_in = open(os.path.join(ROOT, "paper", "main.in.tex")).read()
    FP.fill(tex_in)
    tex = open(os.path.join(ROOT, "paper", "main.tex")).read()
    aux_p = os.path.join(ROOT, "paper", "main.aux")
    if not os.path.exists(aux_p):
        raise SystemExit("paper/main.aux is missing: run `make paper` first, so that the tables get their numbers")
    LABELS.update(aux_labels(open(aux_p).read()))
    tables = paper_tables(tex)
    for lab in tables:
        tables[lab]["number"] = LABELS.get(lab)
    acl = json.load(open(os.path.join(OUT, "acl.json")))
    readers = {r: {"visible_share": v.get("visible_share")} for r, v in acl["readers"].items()}
    data = {"headline": headline(FP.LAST_SUB), "tables": tables, "readers": readers}
    old = None
    if os.environ.get("REUSE_MAP") and os.path.exists(SITE):      # faster reruns while editing the page
        txt = open(SITE).read()
        old = json.loads(txt[txt.index("=") + 1:].rstrip().rstrip(";"))
    if old and "map" in old:
        data["map"], data["hero"], data["decisions"] = old["map"], old["hero"], old["decisions"]
    else:
        data["map"], data["hero"], rank = state_map()
        data["decisions"] = decisions(rank)
    os.makedirs(os.path.dirname(SITE), exist_ok=True)
    with open(SITE, "w") as f:
        f.write("// Generated by cwm/export_site_data.py from results/ and paper/main.tex. Do not edit.\n")
        f.write("window.SITE = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("wrote", SITE, f"{os.path.getsize(SITE) / 1e6:.2f} MB")
    # the page: every number in its text comes from the same values as the paper
    words = dict(data["headline"])
    hero = data["hero"]
    words.update({"HERON": f"{len(hero['records']):,}", "HEROC": str(hero["customer"]),
                  "HEROQ": f"{hero['quarter'][:4]} Q{hero['quarter'][-1]}",
                  "HEROSAMPLE": str(sum(1 for r in hero["records"] if r[2] >= 0)),
                  "FIGMAPS": LABELS.get("fig:maps", "?"), "SITEURL": SITEURL,
                  "PAPERURL": REPO + "/blob/main/paper/main.pdf", "CODEURL": REPO})
    page = open(TEMPLATE).read()
    for k, v in words.items():
        page = page.replace("{{" + k + "}}", v)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9]+)\}\}", page)))
    if left:
        raise SystemExit(f"website/template.html: no value for {left}")
    open(PAGE, "w").write(page)
    print("wrote", PAGE)


if __name__ == "__main__":
    main()
