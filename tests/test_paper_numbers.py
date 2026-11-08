"""The numbers that the text of the paper states directly agree with results/.

cwm/fill_paper.py writes most results into the paper from results/. A few numbers are typed in
paper/main.in.tex. Each test below finds one of them in the text and recomputes it from results/.
If you change a result or the text, this test tells you which sentence no longer agrees.
"""
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TEX = open(os.path.join(ROOT, "paper", "main.in.tex")).read()
ACC, ST = "account_p@10_cross_system", "state_p@10_cross_system"


def load(name):
    return json.load(open(os.path.join(RES, name)))


def find(pattern):
    m = re.search(pattern, TEX, re.S)
    assert m, f"the paper no longer contains: {pattern}"
    return m.groups()


def p1(x):
    return f"{100 * x:.1f}"


def p0(x):
    return f"{100 * x:.0f}"


def f2(x):
    return f"{x:.2f}"


def runs(arm):
    """Seed results of one arm of the main comparison (as in cwm/fill_paper.py)."""
    out = [load("prh.json")["E1"][f"{arm}_s0"]]
    for f in ("extras_a.json", "extras_b.json"):
        out += [v for k, v in load(f).get("E1_seeds", {}).items() if k.rsplit("_s", 1)[0] == arm]
    return out


def mean(arm, key):
    return float(np.mean([r[key] for r in runs(arm)]))


def test_table_2_rows_typed_in_the_text():
    E1 = load("prh.json")["E1"]
    keys = [ACC, ST, "probe_account_messages_to_records", "probe_account_records_to_messages", "rho_state",
            "r2_state", "r2_system"]
    for label, arm in ((r"TF-IDF \+ LSA", "tfidf_lsa"), ("Random encoder", "random_init")):
        got = find(label + r" & " + r" & ".join([r"(\d+(?:\.\d+)?)"] * 7) + r" \\\\")
        want = [f"{E1[arm][k]:.3f}" for k in keys[:4]] + [f2(E1[arm][k]) for k in keys[4:]]
        assert list(got) == want, (label, got, want)


def test_baselines_in_section_6_1():
    E1 = load("prh.json")["E1"]
    assert find(r"TF-IDF reaches (\d+(?:\.\d+)?)\\%")[0] == p1(E1["tfidf_lsa"][ACC])
    assert find(r"randomly initialized encoder reaches (\d+(?:\.\d+)?)\\%")[0] == p1(E1["random_init"][ACC])
    lo, hi = find(r"about (\d+)--(\d+)\\% for the other two objectives")
    alt = [mean(a, ST) for a in ("bad", "ntp")]
    assert (lo, hi) == (p0(min(alt)), p0(max(alt)))


def test_identity_ceiling():
    want = p1(load("extras_b.json")["mask"]["identity_only_state_p@10"])
    assert find(r"could reach at most (\d+(?:\.\d+)?)\\%")[0] == want
    assert find(r"still above the (\d+(?:\.\d+)?)\\%")[0] == want


def test_explained_fractions():
    rho_good = find(r"\$\\rho\$ is about (\d+(?:\.\d+)?) for good views")[0]
    assert f"{mean('good', 'rho_state'):.1f}" == rho_good
    bound = float(find(r"It is below (\d+(?:\.\d+)?) for every other representation")[0])
    E1 = load("prh.json")["E1"]
    others = [mean("bad", "rho_state"), mean("ntp", "rho_state"), E1["tfidf_lsa"]["rho_state"],
              E1["random_init"]["rho_state"]]
    assert max(others) < bound
    assert find(r"explains about (\d+)\\% of the good-view embedding")[0] == p0(mean("good", "r2_system"))


def test_alignment_of_separate_encoders():
    E2 = load("prh.json")["E2"]
    g = E2["good_s0"]
    top, chance = find(r"top match (\d+(?:\.\d+)?)\\% of the time, against (\d+(?:\.\d+)?)\\% by chance")
    assert (top, chance) == (p1(g["anchors_64"]["state_R@1"]), p1(E2["chance_state_R@1"]))
    assert find(r"With 2\{,\}000 anchors, the share is (\d+(?:\.\d+)?)\\%")[0] == p1(g["anchors_2000"]["state_R@1"])
    acc = [g[k]["account_R@1"] for k in ("anchors_64", "anchors_256", "anchors_2000")]
    lo, hi, ch = find(r"recall@1 is (\d+)--(\d+)\\% against (\d+)\\% by chance")
    assert (lo, hi, ch) == (p0(min(acc)), p0(max(acc)), p0(E2["chance_account_R@1"]))
    alt = [E2[a][k]["state_R@1"] for a in ("bad_s0", "ntp_s0") for k in ("anchors_64", "anchors_256", "anchors_2000")]
    assert find(r"recall stays at (\d+)--(\d+)\\%") == (p0(min(alt)), p0(max(alt)))
    good, other = find(r"is (\d+(?:\.\d+)?) with good views and (\d+(?:\.\d+)?) with either alternative")
    assert good == f2(g["mutual_10nn_test"])
    assert {f2(E2[a]["mutual_10nn_test"]) for a in ("bad_s0", "ntp_s0")} == {other}


def test_new_quarter_and_retraining():
    E3 = load("prh.json")["E3"]
    assert find(r"the probe assigns (\d+(?:\.\d+)?)\\% of these records")[0] == p1(E3["good"]["probe_account_day1_to_new_quarter"])
    ntp, bad = find(r"against (\d+(?:\.\d+)?)\\% for next-token prediction and (\d+(?:\.\d+)?)\\% for bad views")
    assert (ntp, bad) == (p1(E3["ntp"]["probe_account_day1_to_new_quarter"]),
                          p1(E3["bad"]["probe_account_day1_to_new_quarter"]))
    r2g, r2b = find(r"The \$R\^2\$ is \$(-?\d+(?:\.\d+)?)\$ against \$(-?\d+(?:\.\d+)?)\$")
    assert (r2g, r2b) == (f2(E3["good"]["v2_to_v1_rotation_r2"]), f2(E3["bad"]["v2_to_v1_rotation_r2"]))
    mg, mb = find(r"neighbour agreement is (\d+(?:\.\d+)?) against (\d+(?:\.\d+)?)")
    assert (mg, mb) == (f2(E3["good"]["v2_to_v1_mutual_10nn"]), f2(E3["bad"]["v2_to_v1_mutual_10nn"]))


def test_persistence_of_states():
    P = load("persistence.json")
    g = P["good"]
    assert find(r"cosine similarity of (\d+(?:\.\d+)?)")[0] == f2(g["same_customer_by_gap"]["1"])
    assert find(r"falls smoothly to (\d+(?:\.\d+)?) for states six quarters apart")[0] == f2(g["same_customer_by_gap"]["6"])
    assert find(r"same quarter have a similarity of (\d+(?:\.\d+)?)")[0] == f2(g["other_customer_same_quarter"])
    cons = [P[a]["same_customer_by_gap"]["1"] for a in ("bad", "ntp")]
    assert find(r"only (\d+(?:\.\d+)?)--(\d+(?:\.\d+)?) alike") == (f2(min(cons)), f2(max(cons)))
    other = [P[a]["other_customer_same_quarter"] for a in ("bad", "ntp")]
    assert find(r"Different customers are (\d+(?:\.\d+)?)--(\d+(?:\.\d+)?) alike") == (f2(min(other)), f2(max(other)))


def test_majority_classes():
    G = load("state.json")["good"]
    kinds = ("tickets", "priority", "call", "stage")
    got = find(r"tickets (\d+(?:\.\d+)?), priority (\d+(?:\.\d+)?), call (\d+(?:\.\d+)?), stage (\d+(?:\.\d+)?)\. "
               r"Majority class next quarter: (\d+(?:\.\d+)?)")
    want = tuple(f2(G[k]["now_majority"]) for k in kinds) + (f2(np.mean([G[k]["next_majority"] for k in kinds])),)
    assert got == want


def test_state_facts_and_transitions():
    S = load("v2_state.json")
    assert find(r"Bad-view states, however, read it (\d+)\\% of the time")[0] == p0(S["bad"]["call"]["now"])
    ratio = load("v2_time.json")["n_transitions"] / load("state.json")["n_transitions"]
    assert find(r"With (\d+(?:\.\d+)?) times as many transitions")[0] == f"{ratio:.1f}"


def test_new_records_on_day_2():
    C = load("corpus.json")["day2"]
    later = C["new_linked_later_by_system"]
    assert "The other CORPUSNEWLATER are calendar events for later months" in TEX   # the count is filled from results
    assert set(later) == {"calendar"}
