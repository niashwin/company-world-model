"""Figures for the paper, from results/*.json (and, for the two maps, the cached embeddings in results/*.npy|npz).

python make_figures.py            -> paper/figures/*.pdf
Colours: one fixed role per method (good views blue, bad views orange, next-token aqua, baselines grey) and a
separate validated triple for the three highlighted customers; every coloured mark also carries a direct label.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("CWM_RESULTS", os.path.join(ROOT, "results"))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"good": "#2a78d6", "bad": "#eb6834", "ntp": "#1baf7a", "tfidf": "#7d7c78", "random": "#b9b8b2"}
NAME = {"good": "Good views", "bad": "Bad views", "ntp": "Next-token", "tfidf": "TF-IDF", "random": "Random init"}
CUST = ["#6250d6", "#eda100", "#e87ba4"]
MARK = ["o", "s", "^"]

META = {"CreationDate": None}   # no timestamp, so a rebuild gives the same file
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def seeds(metric, arm):
    """Values of an E1 metric for one arm over all available seeds."""
    prh, a, b = load("prh.json"), load("extras_a.json"), load("extras_b.json")
    vals = []
    if f"{arm}_s0" in prh.get("E1", {}):
        vals.append(prh["E1"][f"{arm}_s0"][metric])
    for d in (a, b):
        for k, v in d.get("E1_seeds", {}).items():
            if k.startswith(arm + "_s"):
                vals.append(v[metric])
    return np.array(vals)


def panel(ax, letter, x=-0.1, y=1.06):
    """Nature-style panel letter at the top left of an axis."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left", color=INK)


def bar_labels(ax, xs, ys, fmt="{:.2f}", dy=0.015, tops=None, bg=False):
    """Value labels: the text is the value, placed above `tops` (bar plus error bar) when given. With bg, a white
    background keeps a reference line from crossing the label."""
    tops = ys if tops is None else tops
    for x, y, t in zip(xs, ys, tops):
        ax.text(x, t + dy, fmt.format(y), ha="center", va="bottom", fontsize=7.5, color=INK, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.4) if bg else None)


# ---------------------------------------------------------------------------------------------- E1
def fig_e1():
    prh = load("prh.json")["E1"]
    arms = ["good", "bad", "ntp", "tfidf", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    for ax, metric, title in [(axes[0], "account_p@10_cross_system", "Same customer"),
                              (axes[1], "state_p@10_cross_system", "Same customer and quarter")]:
        means, errs = [], []
        for arm in arms:
            if arm in ("tfidf", "random"):
                means.append(prh[f"{arm}_lsa" if arm == "tfidf" else "random_init"][metric]); errs.append(0)
            else:
                v = seeds(metric, arm)
                means.append(v.mean()); errs.append(v.std(ddof=1) if len(v) > 1 else 0)
        xs = np.arange(len(arms))
        ax.bar(xs, means, width=0.62, color=[C[a] for a in arms], yerr=errs, capsize=2.5,
               error_kw=dict(ecolor=INK2, elinewidth=0.8, capthick=0.8))
        bar_labels(ax, xs, np.array(means), tops=np.array(means) + np.array(errs), bg=True)
        ax.set_xticks(xs, ["Good\nviews", "Bad\nviews", "Next-\ntoken", "TF-IDF", "Random\ninit"], rotation=0)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, loc="left", color=INK)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Precision@10, other systems")
    panel(axes[0], "a", x=-0.16); panel(axes[1], "b", x=-0.1)
    # identity-only ceiling on the unmasked corpus (run_extras.identity_only_state_precision)
    from_ident = load("extras_b.json").get("mask", {}).get("identity_only_state_p@10", 0.163)
    axes[1].axhline(from_ident, color=INK2, lw=0.9, ls=(0, (3, 2)))
    axes[1].text(4.4, from_ident + 0.03, "knowing only the customer", ha="right", va="bottom", fontsize=7, color=INK2)
    for ax in axes:
        for lab in ax.get_xticklabels():
            lab.set_fontsize(7.5)
    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG, "e1_precision.pdf"), metadata=META)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- maps
def tsne_maps():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from data import load as load_corpus
    need = [os.path.join(RES, f"emb_{a}_s0.npy") for a in ("good", "bad", "ntp")]
    path = os.path.join(ROOT, "data/era/day1/observations.jsonl")
    if not all(os.path.exists(p) for p in need) or not os.path.exists(path):
        print("maps: skipped (needs data/ and cached embeddings from run_prh.py e1)")
        return False
    c = load_corpus(path)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(c.obs), 3000, replace=False)
    counts = np.bincount(c.account[idx], minlength=len(c.accounts))
    top = np.argsort(-counts)[:12]
    sysmix = [len(set(c.system[idx][c.account[idx] == a])) for a in top]
    pick = [a for _, a in sorted(zip(sysmix, top), key=lambda t: -t[0])][:3]   # well-observed customers
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for ax, arm in zip(axes, ["good", "bad", "ntp"]):
        p = os.path.join(RES, f"emb_{arm}_s0.npy")
        if not os.path.exists(p):
            return
        X = np.load(p)[idx]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Y = TSNE(2, init="pca", perplexity=40, random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(X))
        rest = ~np.isin(c.account[idx], pick)
        ax.scatter(Y[rest, 0], Y[rest, 1], s=2.2, c="#cfcec9", lw=0, rasterized=True)
        for j, a in enumerate(pick):
            m = c.account[idx] == a
            ax.scatter(Y[m, 0], Y[m, 1], s=7, c=CUST[j], marker=MARK[j], lw=0.25, edgecolors="white",
                       rasterized=True, label=f"Customer {'ABC'[j]}")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(NAME[arm], color=C[arm], fontweight="bold")
    for ax, letter in zip(axes, "abc"):
        panel(ax, letter, x=0.0, y=1.02)
    axes[0].legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.1, markerscale=1.6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, "maps.pdf"), metadata=META, dpi=300)
    plt.close(fig)
    json.dump({"customers": [c.accounts[a] for a in pick]}, open(os.path.join(FIG, "maps_meta.json"), "w"))


# ---------------------------------------------------------------------------------------------- E2
def fig_e2():
    e2 = load("prh.json")["E2"]
    ks = [64, 256, 2000]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for arm in ["good", "bad", "ntp"]:
        ys = [e2[f"{arm}_s0"][f"anchors_{k}"]["state_R@1"] for k in ks]
        ax.plot(ks, ys, color=C[arm], lw=2, marker="o", ms=4.5, label=NAME[arm])
    ax.text(2400, e2["good_s0"]["anchors_2000"]["state_R@1"], "Good views", va="center", ha="left", fontsize=7.5)
    ax.text(2400, 0.082, "Bad views,\nnext-token", va="center", ha="left", fontsize=7.5)
    ax.axhline(e2["chance_state_R@1"], color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(60, e2["chance_state_R@1"] - 0.004, "chance", fontsize=7, color=INK2, va="top")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(50, 9000)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("Anchor pairs used to fit the rotation")
    ax.set_ylabel("Same-state recall@1")
    panel(ax, "a", x=-0.2)
    fig.savefig(os.path.join(FIG, "e2_anchors.pdf"), metadata=META)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- E3
def fig_e3():
    e3b = load("e3b.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    vals = list(e3b["corr_by_direction_first_10"]) + [e3b["corr_dirs_11_to_32_mean"], e3b["corr_dirs_33_to_256_mean"]]
    labels = [str(i) for i in range(1, 11)] + ["11-32", "33-256"]
    xs = np.arange(len(vals), dtype=float); xs[-2] = 11.0; xs[-1] = 12.6
    cols = [C["good"]] * 10 + ["#86b6ef", "#cde2fb"]
    ax.bar(xs, vals, width=0.7, color=cols)
    ax.set_xticks(xs, labels)
    for lab in ax.get_xticklabels()[-2:]:
        lab.set_fontsize(6.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correlation, old vs. new")
    ax.set_xlabel("Principal direction of the space (mean for the last two groups)")
    ax.set_title("Which directions survive retraining", loc="left")
    bar_labels(ax, xs[-2:], vals[-2:], dy=0.02)
    ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
    ax = axes[1]
    st = e3b["old_store_precision@10"]
    names = ["Old\nmodel", "New +\nrotation", "New,\nno rotation", "Chance"]
    ys = [st["query_by_old_encoder"], st["query_by_new_encoder_rotated"], st["query_by_new_encoder_unrotated"],
          e3b["chance_precision@10"]]
    cols = ["#86b6ef", C["good"], "#b9b8b2", "#e4e3df"]
    xs = np.arange(4)
    ax.bar(xs, ys, width=0.62, color=cols)
    bar_labels(ax, xs, ys)
    ax.set_xticks(xs, names)
    for lab in ax.get_xticklabels():
        lab.set_fontsize(7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Same-customer precision@10")
    ax.set_title("Searching the old index", loc="left")
    panel(axes[0], "a", x=-0.12); panel(axes[1], "b", x=-0.16)
    ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG, "e3_time.pdf"), metadata=META)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- robustness
def fig_robust():
    prh, b = load("prh.json")["E1"], load("extras_b.json")
    if "noise" not in b or "mask" not in b or "good_s0" not in b["mask"]:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    ps = [0.0] + sorted(float(k[1:]) for k in b["noise"])
    ys = [prh["good_s0"]["account_p@10_cross_system"]] + [b["noise"][f"p{p}"]["account_p@10_cross_system"] for p in ps[1:]]
    ys2 = [prh["good_s0"]["state_p@10_cross_system"]] + [b["noise"][f"p{p}"]["state_p@10_cross_system"] for p in ps[1:]]
    ax.plot(ps, ys, color=C["good"], lw=2, marker="o", ms=4.5)
    ax.plot(ps, ys2, color=C["good"], lw=1.4, marker="o", ms=3.5, ls=(0, (4, 2)))
    ax.text(ps[-1] + 0.02, ys[-1], "same customer", va="center", fontsize=7.5)
    ax.text(ps[-1] + 0.02, ys2[-1], "same customer\nand quarter", va="center", fontsize=7.5)
    for arm in ("bad", "ntp"):
        y = prh[f"{arm}_s0"]["account_p@10_cross_system"]
        ax.axhline(y, color=C[arm], lw=1, ls=(0, (2, 2)))
    ax.text(0.25, 0.035, "bad views, next-token (clean links)", fontsize=7, color=INK2, va="bottom", ha="center")
    ax.set_xlim(-0.02, 0.8)
    ax.set_xticks(ps, [f"{int(p * 100)}%" for p in ps])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Records attached to the wrong state during training")
    ax.set_ylabel("Precision@10, other systems")
    ax.set_title("Wrong links", loc="left")
    ax = axes[1]
    m = b["mask"]
    groups = [("Good views", prh["good_s0"]["account_p@10_cross_system"], m["good_s0"]["account_p@10_cross_system"]),
              ("TF-IDF", prh["tfidf_lsa"]["account_p@10_cross_system"], m["tfidf_lsa"]["account_p@10_cross_system"])]
    xs = np.arange(len(groups))
    w = 0.36
    for j, (name, a, bm) in enumerate(groups):
        col = C["good"] if j == 0 else C["tfidf"]
        ax.bar(xs[j] - w / 2, a, w * 0.92, color=col)
        ax.bar(xs[j] + w / 2, bm, w * 0.92, color=col, alpha=0.45, hatch="////", edgecolor="white", lw=0)
        bar_labels(ax, [xs[j] - w / 2, xs[j] + w / 2], [a, bm])
        ax.text(xs[j] - w / 2, -0.09, "original", ha="center", fontsize=7, color=INK2)
        ax.text(xs[j] + w / 2, -0.09, "masked", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xs, [g[0] for g in groups])
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Same customer, precision@10")
    ax.set_title("Customer names, contacts and ids masked", loc="left")
    panel(axes[0], "a", x=-0.13); panel(axes[1], "b", x=-0.15)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG, "robustness.pdf"), metadata=META)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------- persistence
def state_similarity(S, states, counts, min_records=5, max_gap=6):
    """Cosine similarity of centred state vectors: same customer k quarters apart, and different customers in the
    same quarter."""
    def qi(q):
        return int(q[:4]) * 4 + int(q[-1]) - 1
    ok = np.where(counts >= min_records)[0]
    X = S[ok].astype(float)
    X = X - X.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    same, other = {}, []
    for a in range(len(ok)):
        for b in range(a + 1, len(ok)):
            (ca, qa), (cb, qb) = states[ok[a]], states[ok[b]]
            c = float(X[a] @ X[b]); g = abs(qi(qa) - qi(qb))
            if ca == cb and 1 <= g <= max_gap:
                same.setdefault(g, []).append(c)
            elif ca != cb and g == 0:
                other.append(c)
    return {g: float(np.mean(v)) for g, v in sorted(same.items())}, float(np.mean(other))


def fig_persistence():
    p = os.path.join(RES, "state_vectors.npz")
    if not os.path.exists(p):
        print("persist: skipped (needs results/state_vectors.npz from run_state.py)")
        return False
    z = np.load(p, allow_pickle=True)
    states = [tuple(s.split("|")) for s in z["states"]]
    out = {}
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    for arm in ["good", "bad", "ntp", "tfidf"]:
        same, other = state_similarity(z[arm], states, z["counts"])
        out[arm] = {"same_customer_by_gap": same, "other_customer_same_quarter": other}
        gs = sorted(same)
        lw, ms = (2, 4.5) if arm != "tfidf" else (1.2, 3)
        ax.plot(gs, [same[g] for g in gs], color=C[arm], lw=lw, marker="o", ms=ms, label=NAME[arm])
        off = {"good": -0.18, "bad": -0.06, "ntp": 0.06, "tfidf": 0.18}[arm]
        ax.plot([off], [other], color=C[arm], marker="D", ms=ms, mfc="white", mew=1.3, ls="none")
    ax.set_xticks([0] + list(range(1, 7)), ["other\ncustomer"] + [str(g) for g in range(1, 7)])
    ax.set_xlim(-0.6, 6.4)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("Quarters apart, same customer")
    ax.set_ylabel("Cosine similarity of states")
    panel(ax, "b", x=-0.2)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, borderaxespad=0.1)
    fig.savefig(os.path.join(FIG, "persistence.pdf"), metadata=META)
    plt.close(fig)
    json.dump(out, open(os.path.join(RES, "persistence.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------- access control
READER_NAME = {"ceo": "CEO", "account_executive": "Account\nexecutive", "support_engineer": "Support\nengineer",
               "software_engineer": "Software\nengineer", "financial_analyst": "Financial\nanalyst"}


def fig_acl():
    a = load("acl.json")
    if not a or "readers" not in a:
        return
    R = a["readers"]
    short = {"ceo": "CEO", "account_executive": "Account\nexec.", "support_engineer": "Support\neng.",
             "software_engineer": "Software\neng.", "financial_analyst": "Financial\nanalyst"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    confs = [("open", "No access control", "#52514e", None), ("filtered", "Search and states filtered", "#b9b8b2", "////"),
             ("built_in", "Built in (this paper)", C["good"], None)]
    groups = [(r, t) for r in ("support_engineer", "software_engineer") if "leak" in R.get(r, {})
              for t in ("codename", "secret")]
    w = 0.26
    for j, (conf, name, col, hatch) in enumerate(confs):
        ys = [R[r]["leak"][conf].get(f"{t}_advantage", 0.0) for r, t in groups]
        xs = np.arange(len(groups)) + (j - 1) * w
        ax.bar(xs, ys, w * 0.92, color=col, hatch=hatch, edgecolor="white", lw=0, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, f"{y:.1f}" if y else "0", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(groups)), [f"{short[r]}\n{t}" for r, t in groups])
    for lab in ax.get_xticklabels():
        lab.set_fontsize(6.8)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Access advantage")
    ax.set_title("Planted secrets found without access", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=1.2, borderaxespad=0.1)
    ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
    ax = axes[1]
    R2 = load("v2_acl.json").get("readers", {})
    readers = [r for r in short if r in R]
    xs = np.arange(len(readers))
    o = [R[r]["placement_open_encoder"]["customer_acc"] for r in readers]
    b = [R[r].get("placement_own_encoder", R[r]["placement_open_encoder"])["customer_acc"] for r in readers]
    bars = [(o, "Trained on everything (leaks)", "#b9b8b2", None), (b, "Own encoder", C["good"], None)]
    if R2:
        h = [R2[r]["placement_own_head"]["customer_acc"] if r in R2 else np.nan for r in readers]
        bars.append((h, "Backbone + own head", "#9cc2ee", "////"))
    w = 0.8 / len(bars)
    for j, (ys, name, col, hatch) in enumerate(bars):
        xx = xs + (j - (len(bars) - 1) / 2) * w
        ax.bar(xx, ys, w * 0.92, color=col, hatch=hatch, edgecolor="white", lw=0, label=name)
        for x, y in zip(xx, ys):
            if np.isfinite(y):
                ax.text(x, y + 0.015, f"{100 * y:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK)
    ax.set_xticks(xs, [f"{short[r]}\n{100 * R[r]['visible_share']:.0f}%" for r in readers])
    for lab in ax.get_xticklabels():
        lab.set_fontsize(6.8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Held-out records, right customer")
    ax.set_title("Cost (under each reader: share readable)", loc="left")
    panel(axes[0], "a", x=-0.14); panel(axes[1], "b", x=-0.15)
    ax.legend(loc="upper right", frameon=False, fontsize=6.4, handlelength=1.2, borderaxespad=0.1, ncol=1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG, "acl.pdf"), metadata=META)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- Nature-style main figures
PRE, PRE_HEAD = "#5f6f82", "#9cc2ee"     # frozen pretrained encoder; the same with a good-view head (blue, hatched)


def _fp():
    """The paper's own aggregation of results (cwm/fill_paper.py), so figures and text use the same numbers."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fill_paper
    return fill_paper


def fig_heldout():
    """Fig. 2d: records whose links were withheld, placed with the right customer."""
    FP = _fp()
    HO, P = FP.HO, FP.V2["pretrained"].get("heldout", {})
    if "good_s0" not in HO or "minilm_head" not in P:
        return False
    rows = [("Good views", HO["good_s0"]["customer_acc"], C["good"], None),
            ("Bad views", HO["bad_s0"]["customer_acc"], C["bad"], None),
            ("Next-token", HO["ntp_s0"]["customer_acc"], C["ntp"], None),
            ("TF-IDF", HO["tfidf_lsa"]["customer_acc"], C["tfidf"], None),
            ("Random encoder", HO["random_init"]["customer_acc"], C["random"], None),
            ("Pretrained, frozen", P["minilm_raw"]["customer_acc"], PRE, None),
            ("Pretrained + good-view head", P["minilm_head"]["customer_acc"], PRE_HEAD, "////")]
    fig, ax = plt.subplots(figsize=(6.6, 2.05))
    ys = np.arange(len(rows))[::-1]
    for y, (name, v, col, hatch) in zip(ys, rows):
        ax.barh(y, v, height=0.62, color=col, hatch=hatch, edgecolor="white", lw=0)
        ax.text(v + 0.012, y, f"{100 * v:.0f}%", va="center", fontsize=7.5, color=INK)
    ref = HO["reference_majority_customer_acc"]
    ax.axvline(ref, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(ref + 0.008, ys[0] + 0.62, "largest customer", fontsize=7, color=INK2, va="center")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Records with withheld links placed with the right customer")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    panel(ax, "d", x=-0.34, y=1.0)
    fig.savefig(os.path.join(FIG, "heldout.pdf"), metadata=META)
    plt.close(fig)


def letter_at(fig, ax, letter, labels_left=True):
    """Panel letter above the axis: at the left edge of its tick labels, or just left of the plot."""
    fig.canvas.draw()
    pos = ax.get_position()
    if labels_left:
        x = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0
    else:
        x = pos.x0 - 0.035
    fig.text(x, pos.y1 + 0.035, letter, fontsize=10.5, fontweight="bold", va="bottom", ha="left", color=INK)


def _dots(ax, n, means, sds, colours, ref=None, ref_label=None):
    ys = np.arange(n)[::-1]
    for y, m, sd, col in zip(ys, means, sds, colours):
        if sd:
            ax.plot([m - sd, m + sd], [y, y], color=col, lw=1.6, alpha=0.55, solid_capstyle="butt")
        ax.plot([m], [y], marker="o", ms=5.2, color=col, mec="white", mew=0.8)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.text(ref + 0.004, n - 0.45, ref_label, fontsize=6.8, color=INK2, ha="left", va="bottom")
    ax.set_ylim(-0.6, n - 0.1 + 0.45)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return ys


def fig_memory():
    """Fig. 4: questions that come with one record of the customer instead of its name, with names kept (a) or
    masked (b); and questions that name the customer (c)."""
    FP = _fp()
    if not FP.ANCH["plain"].get("runs") or not FP.ANCH["masked"].get("runs"):
        return False
    fig = plt.figure(figsize=(6.6, 3.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.45, 1.0], hspace=0.62, wspace=0.07,
                          left=0.29, right=0.985, top=0.92, bottom=0.11)
    rows = [("closed", "No store"), ("state", "Neighbours, state channel"), ("detail", "Neighbours, detail channel"),
            ("concat", "Neighbours, both channels"), ("routed", "Neighbours by state, read by detail"),
            ("place_detail", "Placement, detail channel"), ("place_concat", "Placement, both channels"),
            ("place_state", "Placement, state channel"), ("link", "Oracle: the anchor's own link")]
    axes = []
    for j, (corpus, title) in enumerate((("plain", "Names kept in the records"), ("masked", "Names masked"))):
        ax = fig.add_subplot(gs[0, j])
        means, sds, cols = [], [], []
        for k, _ in rows:
            acc = FP.anchor_stats(corpus, k)[0]
            means.append(float(acc.mean())); sds.append(float(acc.std(ddof=1)) if len(acc) > 1 else 0.0)
            hl = k == "place_state" and corpus == "plain"
            cols.append(C["good"] if hl else ("#9a9890" if k in ("closed", "link") else INK2))
        ys = _dots(ax, len(rows), means, sds, cols, ref=means[0], ref_label="no store")
        ax.set_xlim(0.44, 0.76)
        ax.set_xticks([0.5, 0.6, 0.7], ["50%", "60%", "70%"])
        ax.set_yticks(ys, [r[1] for r in rows] if j == 0 else [""] * len(rows))
        if j == 0:
            for lab in ax.get_yticklabels():
                lab.set_fontsize(7.4)
        ax.set_xlabel("Questions answered")
        ax.set_title(title, loc="left", fontsize=8.5, x=0.0 if j == 0 else 0.07)
        axes.append(ax)
    # c: questions that name the customer
    ax = fig.add_subplot(gs[1, 0])
    stores = [("good", "State channel"), ("minilm", "Detail channel"), ("dual", "Both, concatenated"),
              ("route", "Found by state, read by detail")]
    closed_of = {"route": "minilm"}
    ys = np.arange(len(stores))[::-1]
    for y, (k, name) in zip(ys, stores):
        cb = [v for _, n, v in FP.MEM[closed_of.get(k, k)] if n == "closed_book"]
        st = [v for _, n, v in FP.MEM[k] if n == "top16_pretrained"]
        a, b, sd = float(np.mean(cb)), float(np.mean(st)), float(np.std(st, ddof=1))
        ax.plot([a, b], [y, y], color="#cfcec9", lw=1.6)
        ax.plot([b - sd, b + sd], [y, y], color=C["good"], lw=1.6, alpha=0.5, solid_capstyle="butt")
        ax.plot([a], [y], marker="o", ms=4.6, color="#9a9890", mec="white", mew=0.8)
        ax.plot([b], [y], marker="o", ms=5.2, color=C["good"], mec="white", mew=0.8)
    ax.set_yticks(ys, [n for _, n in stores])
    for lab in ax.get_yticklabels():
        lab.set_fontsize(7.4)
    ax.set_xlim(0.55, 0.8)
    ax.set_xticks([0.6, 0.7, 0.8], ["60%", "70%", "80%"])
    ax.set_ylim(-0.6, len(stores) - 0.4)
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Questions answered")
    ax.set_title("Questions that name the customer", loc="left", fontsize=8.5)
    key = fig.add_subplot(gs[1, 1]); key.axis("off")
    key.text(0.07, 0.95, "Grey dot: no store (closed book).\nBlue dot: the reader reads 16 records\nthat the store finds."
             "\nLines: one standard deviation\nover four seeds.", va="top", ha="left", fontsize=7, color=INK2,
             transform=key.transAxes, linespacing=1.45)
    for ax, letter, left in ((axes[0], "a", True), (axes[1], "b", False), (fig.axes[2], "c", True)):
        letter_at(fig, ax, letter, labels_left=left)
    fig.savefig(os.path.join(FIG, "memory.pdf"), metadata=META)
    plt.close(fig)


def fig_decisions():
    """Fig. 5: the decision simulation for customer-months in trouble. a, balance: similar months of other customers
    that were themselves in trouble, by decision. b, c, what followed: intervene minus not, with 95% intervals."""
    SI = load("actions.json").get("simulate", {})
    if not SI:
        return False
    spaces = [("", "Good\nviews"), ("_bad", "Bad\nviews"), ("_minilm", "Pre-\ntrained"), ("_good_month+minilm", "Two\nchannels")]
    fig = plt.figure(figsize=(6.6, 2.7))
    ax = fig.add_axes([0.075, 0.2, 0.25, 0.62])
    xs = np.arange(len(spaces))
    w = 0.36
    on = [SI["escalation"]["balance_in_trouble" + s][0] for s, _ in spaces]
    off = [SI["escalation"]["balance_in_trouble" + s][1] for s, _ in spaces]
    ax.bar(xs - w / 2, on, w * 0.92, color="#52514e", label="Company intervened")
    ax.bar(xs + w / 2, off, w * 0.92, color="#cfcec9", label="Did not intervene")
    for x, v in zip(xs - w / 2, on):
        ax.text(x, v + 0.008, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.6)
    for x, v in zip(xs + w / 2, off):
        ax.text(x, v + 0.008, f"{100 * v:.1f}" if v < 0.01 else f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(xs, [n for _, n in spaces])
    for lab in ax.get_xticklabels():
        lab.set_fontsize(7)
    ax.set_ylim(0, 0.4)
    ax.set_yticks([0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"])
    ax.set_ylabel("Similar months in trouble")
    ax.set_title("Who is compared", loc="left", fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.6, handlelength=1, borderaxespad=0.1)
    ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
    est = [("observed_in_trouble", "Observed"), ("case", "Similar months, good views"),
           ("case_bad", "Similar months, bad views"), ("case_minilm", "Similar months, pretrained"),
           ("case_good_month+minilm", "Similar months, two channels"), ("unmatched", "All other months"),
           ("model", "Outcome model")]
    axs = [ax]
    for j, (yk, title) in enumerate((("escalation", "Urgent ticket"), ("deal_progress", "Deal advances"))):
        ax = fig.add_axes([0.555 + 0.235 * j, 0.2, 0.2, 0.62])
        ys = np.arange(len(est))[::-1]
        for y, (k, name) in zip(ys, est):
            v = SI[yk][k]
            d = 100 * (v[0] - v[1])
            ci = SI[yk].get(k + "_difference_ci")
            col = C["good"] if k == "case" else ("#9a9890" if k == "observed_in_trouble" else INK2)
            if ci:
                ax.plot([100 * ci[0], 100 * ci[1]], [y, y], color=col, lw=1.6, alpha=0.55, solid_capstyle="butt")
            if k == "observed_in_trouble":
                ax.plot([d], [y], marker="D", ms=4.6, color=col, mfc="white", mec=col, mew=1.2)
            else:
                ax.plot([d], [y], marker="o", ms=5, color=col, mec="white", mew=0.8)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.set_xlim(-45, 65)
        ax.set_xticks([-40, 0, 40], ["−40", "0", "+40"])
        ax.set_xlabel("Difference (points)")
        ax.set_ylim(-0.6, len(est) - 0.4)
        ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
        ax.set_yticks(ys, [n for _, n in est] if j == 0 else [""] * len(est))
        if j == 0:
            for lab in ax.get_yticklabels():
                lab.set_fontsize(7)
            ax.get_yticklabels()[1].set_fontweight("bold")
        ax.set_title(title, loc="left", fontsize=8.5, x=0.0 if j == 0 else 0.07)
        axs.append(ax)
    for ax, letter, left in ((axs[0], "a", True), (axs[1], "b", True), (axs[2], "c", False)):
        letter_at(fig, ax, letter, labels_left=left)
    fig.savefig(os.path.join(FIG, "decisions.pdf"), metadata=META)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["e1", "maps", "e2", "e3", "robust", "persist", "acl", "heldout", "memory", "decisions"]
    for w in which:
        done = {"e1": fig_e1, "maps": tsne_maps, "e2": fig_e2, "e3": fig_e3, "robust": fig_robust,
                "persist": fig_persistence, "acl": fig_acl, "heldout": fig_heldout, "memory": fig_memory,
                "decisions": fig_decisions}[w]()
        if done is not False:
            print("made", w)
