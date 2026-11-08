"""Write the full reference list of the paper into README.md, with a link for each entry.

The list goes between the markers `<!-- refs:start -->` and `<!-- refs:end -->`, in the order of the paper.
Run it after a change to paper/refs.bib and a build of the paper: `python tools/readme_refs.py`.
`make site` runs it too.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cwm"))
import bibrefs  # noqa: E402


def main():
    bib = bibrefs.parse_bib(os.path.join(ROOT, "paper", "refs.bib"))
    order = bibrefs.cite_order(os.path.join(ROOT, "paper", "main.aux"))
    missing = [k for k in order if k not in bib or not bibrefs.entry_url(bib[k])]
    if missing:
        sys.exit(f"These references have no entry or no link: {missing}")
    lines = [f"{i}. {bibrefs.entry_markdown(bib[k])}" for i, k in enumerate(order, 1)]
    block = ("<!-- refs:start -->\n<details>\n<summary>All " + str(len(order)) +
             " references of the paper, in its order, with links</summary>\n\n" + "\n".join(lines) +
             "\n\n</details>\n<!-- refs:end -->")
    path = os.path.join(ROOT, "README.md")
    text = open(path, encoding="utf-8").read()
    new, n = re.subn(r"<!-- refs:start -->.*?<!-- refs:end -->", lambda m: block, text, flags=re.S)
    if n != 1:
        sys.exit("README.md must have one block between <!-- refs:start --> and <!-- refs:end -->.")
    if new != text:
        open(path, "w", encoding="utf-8").write(new)
    print(f"README.md: {len(order)} references")


if __name__ == "__main__":
    main()
