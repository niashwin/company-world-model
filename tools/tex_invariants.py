#!/usr/bin/env python3
"""Compare two versions of a LaTeX text and report what a rewrite must keep: placeholders filled by
cwm/fill_paper.py, labels, references, citations, environments and numbers.

Usage:  python tools/tex_invariants.py original.tex rewritten.tex [--map fill.json]
Exit code 1 if a placeholder, label, reference, citation or environment changed.
Numbers are reported but do not fail the check, because a rewrite may write "two" as "2".
"""
import argparse
import collections
import json
import re
import sys


def tokens(text, sub):
    keys = set(sub)
    t = re.sub(r"(?<!\\)%.*", "", text)
    out = {
        "placeholders": collections.Counter(w for w in re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", t) if w in keys)
        if keys else collections.Counter(w for w in re.findall(r"\b[A-Z][A-Z0-9]{5,}\b", t)),
        "labels": collections.Counter(re.findall(r"\\label\{([^}]*)\}", t)),
        "refs": collections.Counter(re.findall(r"\\(?:ref|secref|eqref|cref|autoref)\{([^}]*)\}", t)),
        "cites": collections.Counter(k.strip() for g in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", t)
                                     for k in g.split(",")),
        "environments": collections.Counter(re.findall(r"\\begin\{([^}]*)\}", t)),
        "numbers": collections.Counter(re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)*", t)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("rewritten")
    ap.add_argument("--map", help="placeholder map written by FILL_MAP=... python cwm/fill_paper.py")
    a = ap.parse_args()
    sub = json.load(open(a.map)) if a.map else {}
    A, B = tokens(open(a.original).read(), sub), tokens(open(a.rewritten).read(), sub)
    fail = False
    for kind in A:
        lost, added = A[kind] - B[kind], B[kind] - A[kind]
        if kind in ("refs", "cites", "placeholders"):
            lost = collections.Counter({k: v for k, v in lost.items() if k not in B[kind]})   # repeats may merge
            added = collections.Counter({k: v for k, v in added.items() if k not in A[kind]})
        if lost or added:
            hard = kind != "numbers"
            fail |= hard
            print(f"{'FAIL' if hard else 'note'} {kind}: lost {dict(lost)} added {dict(added)}")
    print("invariants:", "FAIL" if fail else "ok")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
