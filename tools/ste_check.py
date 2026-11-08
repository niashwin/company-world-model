#!/usr/bin/env python3
"""Check LaTeX or Markdown prose against the main writing rules of ASD-STE100 (Simplified Technical English).

The checker is a set of heuristics. It does not replace the STE dictionary or a trained reviewer. It checks:

  LEN      a sentence has more than 25 words (descriptive text; rule 6.3). With --procedure, more than 20 (rule 5.1).
  PASSIVE  a form of "be" followed by a past participle (rule 3.6: use the active voice as much as possible).
  ING      a word that ends in -ing and is not on the list of technical names below (rule 3.2).
  TENSE    a perfect tense such as "has been" or "had shown" (rule 3.1: simple tenses only).
  PARA     a paragraph has more than six sentences (rule 6.6). Lists are not paragraphs.

A sentence complies when it has no LEN, PASSIVE, ING or TENSE flag. The score is the share of sentences that comply.

Usage:
  python tools/ste_check.py paper/main.tex                      # the filled paper (placeholders replaced)
  python tools/ste_check.py paper/main.in.tex --map fill.json   # the template, with a map from cwm/fill_paper.py
  python tools/ste_check.py README.md --show 40                 # list the first 40 sentences that do not comply
  python tools/ste_check.py chunk.tex --min 0.9                 # exit with code 1 below 90% compliance
"""
import argparse
import json
import re
import sys

# Words that end in -ing but are nouns, prepositions or technical names in this project.
ING_OK = set("""
during nothing something anything everything string strings thing things bring brings ring rings spring king
morning evening ceiling following training pretraining pre-training embedding embeddings learning forecasting
tracking masking clustering ranking routing sampling logging planning encoding modeling modelling reasoning timing
setting settings meeting meetings building buildings briefing onboarding pricing billing marketing engineering
accounting ticketing padding pooling heading headings listing listings rating ratings warning warnings mining
""".split())

IRREGULAR_PP = set("""
known shown seen given taken made done built found held kept left lost read run set split written chosen drawn
fitted put cut led sent spent told won begun broken driven fallen forgotten frozen hidden spoken thrown bought
brought caught sought taught bound meant dealt felt heard paid said sold understood withheld withdrawn grown
shown drawn shaken stolen worn torn sworn beaten bitten hung lit laid lain overlaid mistaken undertaken rewritten
retrained overwritten upheld
""".split())

BE = r"(?:am|is|are|was|were|be|been|being)"
ADV = r"(?:\s+(?:\w+ly|not|also|then|only|now|still|often|always|never|thus|therefore|first|then))*"
PASSIVE_RE = re.compile(rf"\b{BE}{ADV}\s+(\w+)\b", re.I)
PERFECT_RE = re.compile(r"\b(?:has|have|had)\s+(?:not\s+|also\s+|only\s+|never\s+)?(?:been|\w+ed|"
                        + "|".join(sorted(IRREGULAR_PP)) + r")\b", re.I)
ABBR = ["e.g.", "i.e.", "et al.", "vs.", "Fig.", "Figs.", "Sec.", "Eq.", "cf.", "approx.", "No.", "Tab.", "etc.",
        "Inc.", "Ltd.", "Dr.", "Mr.", "Ms.", "St."]


def is_pp(w):
    w = w.lower()
    if w in IRREGULAR_PP:
        return True
    if len(w) > 4 and w.endswith("ed") and w not in ("need", "seed", "speed", "feed", "embed", "red", "bed", "shed",
                                                      "indeed", "hundred", "exceed", "proceed", "succeed", "breed"):
        return True
    return False


# ------------------------------------------------------------------------------------------------ LaTeX to prose
def strip_env(text, env):
    return re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", text, flags=re.S)


def braced(text, i):
    """Return (content, end) of the brace group that starts at text[i] == '{'."""
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif text[j] == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return text[i + 1:], len(text)


def pull_command(text, cmd):
    """Remove every \\cmd{...} from text and return (text, [contents])."""
    out, found, i = [], [], 0
    pat = re.compile(rf"\\{cmd}\*?(?:\[[^\]]*\])?\{{")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        content, end = braced(text, m.end() - 1)
        found.append(content)
        i = end
    return " ".join(out), found


def latex_to_paragraphs(text):
    text = re.sub(r"(?<!\\)%.*", "", text)                                  # comments
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    # the whole body, without the bibliography commands (Extended Data and Supplementary Information come after them)
    text = re.sub(r"\\bibliography(?:style)?\{[^}]*\}", "\n\n", text)
    for env in ("tabular", "tabular\\*", "tabularx", "equation", "equation\\*", "align", "align\\*", "verbatim", "lstlisting",
                "thebibliography", "tikzpicture"):
        text = strip_env(text, env)
    text, captions = pull_command(text, "caption")
    text, footnotes = pull_command(text, "footnote")
    text = re.sub(r"\\begin\{(?:table|figure)\*?\}(\[[^\]]*\])?|\\end\{(?:table|figure)\*?\}", "\n\n", text)
    text += "\n\n" + "\n\n".join(captions + footnotes)
    text = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}|\\appendix|\\maketitle|\\tableofcontents", "\n\n", text)
    text = re.sub(r"\\paragraph\*?\{[^}]*\}", "\n\n", text)
    text = re.sub(r"\\suppnote\{[^}]*\}\{[^}]*\}", "\n\n", text)              # Supplementary Note headings
    text = re.sub(r"(?:\\textbf\{[a-z]\}(?:--|, )?)+,\s", "\n\n", text)          # Nature panel letters: "\textbf{a}, ..."
    text = re.sub(r"\\item\b(\[[^\]]*\])?", "\n\n", text)                   # each list item is its own block
    text = re.sub(r"\\begin\{(?:itemize|enumerate|description|abstract|center|small|quote)\}|"
                  r"\\end\{(?:itemize|enumerate|description|abstract|center|small|quote)\}", "\n\n", text)
    text = re.sub(r"\\(?:label|includegraphics|vspace|hspace|setlength|centering|small|footnotesize|"
                  r"scriptsize|noindent|newpage|clearpage|bibliography|bibliographystyle)\*?(\[[^\]]*\])?(\{[^}]*\})*",
                  " ", text)
    text = re.sub(r"\\cite[pt]?\*?(\[[^\]]*\])*\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:secref|ref|eqref|autoref|cref|Cref)\{[^}]*\}", "1", text)
    text = re.sub(r"\$[^$]*\$", " X ", text)                                 # inline math is one word
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    for _ in range(3):
        text = re.sub(r"\\(?:textbf|emph|textit|texttt|textsc|text|mbox|underline|url|href\{[^}]*\})\{([^{}]*)\}",
                      r"\1", text)
    text = text.replace("~", " ").replace("\\%", "%").replace("\\&", "&").replace("\\,", "").replace("\\ ", " ")
    text = text.replace("``", '"').replace("''", '"').replace("---", " - ").replace("--", "-")
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)                             # any command left
    text = text.replace("{", "").replace("}", "")
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def markdown_to_paragraphs(text):
    text = re.sub(r"```.*?```", "\n\n", text, flags=re.S)                    # code blocks
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        if all(l.lstrip().startswith("|") for l in lines):                   # tables: each cell is checked alone
            for l in lines:
                if re.match(r"^\s*\|[\s:|-]+\|\s*$", l):
                    continue
                blocks += [c.strip() for c in l.strip().strip("|").split("|") if len(c.split()) > 3]
            continue
        cur = []
        for l in lines:
            if re.match(r"^\s*(#+\s|[-*+]\s|\d+\.\s)", l):                   # headings and list items start blocks
                if cur:
                    blocks.append(" ".join(cur))
                cur = [] if l.lstrip().startswith("#") else [re.sub(r"^\s*([-*+]|\d+\.)\s", "", l)]
            else:
                cur.append(l.strip())
        if cur:
            blocks.append(" ".join(cur))
    out = []
    for b in blocks:
        b = re.sub(r"`[^`]*`", "X", b)                                      # inline code is one word
        b = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", b)                     # links keep their text
        b = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", b)
        b = re.sub(r"<[^>]+>", " ", b)
        out.append(re.sub(r"\s+", " ", b).strip())
    return [b for b in out if b]


def html_to_paragraphs(text):
    text = re.sub(r"<(script|style|svg|pre|code)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<(?:p|li|h[1-6]|figcaption|dd|dt|td|th|blockquote|div|section|br)[^>]*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&thinsp;", " ", text).replace("&amp;", "&").replace("&minus;", "-").replace("&ndash;", "-")
    text = re.sub(r"&[a-z]+;", " ", text)
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) > 3]


# ------------------------------------------------------------------------------------------------ sentences
def sentences(paragraph):
    p = paragraph
    for a in ABBR:
        p = p.replace(a, a.replace(".", "<DOT>"))
    p = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", p)                               # decimals and section numbers
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9§\"(\[])", p)
    return [s.replace("<DOT>", ".").strip() for s in parts if re.search(r"[A-Za-z]", s)]


def words(sentence):
    return [w for w in sentence.split() if re.search(r"[A-Za-z0-9]", w)]


def flags(sentence, limit):
    f = []
    n = len(words(sentence))
    if n > limit:
        f.append(f"LEN {n}")
    for m in PASSIVE_RE.finditer(sentence):
        if is_pp(m.group(1)):
            f.append(f"PASSIVE '{m.group(0)}'")
            break
    for w in re.findall(r"\b[A-Za-z][A-Za-z-]*ing\b", sentence):
        if len(w) > 4 and w.lower() not in ING_OK and w.lower().split("-")[-1] not in ING_OK:
            f.append(f"ING '{w}'")
            break
    m = PERFECT_RE.search(sentence)
    if m:
        f.append(f"TENSE '{m.group(0)}'")
    return f


def check(paragraphs, limit=25):
    rows, long_paras = [], 0
    for p in paragraphs:
        ss = sentences(p)
        if len(ss) > 6:
            long_paras += 1
        for s in ss:
            rows.append((s, flags(s, limit)))
    return rows, long_paras


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--map", help="JSON map from placeholder to value (FILL_MAP=... python cwm/fill_paper.py ...)")
    ap.add_argument("--procedure", action="store_true", help="use the 20-word limit of procedures")
    ap.add_argument("--show", type=int, default=0, help="list this many sentences that do not comply")
    ap.add_argument("--min", type=float, default=None, help="exit with code 1 if the share that complies is lower")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args()
    sub = json.load(open(a.map)) if a.map else {}
    limit = 20 if a.procedure else 25
    total, bad_all, report = 0, 0, {}
    for path in a.files:
        text = open(path).read()
        for k in sorted(sub, key=len, reverse=True):
            text = text.replace(k, str(sub[k]))
        if path.endswith(".tex"):
            paras = latex_to_paragraphs(text)
        elif path.endswith((".html", ".htm")):
            paras = html_to_paragraphs(text)
        else:
            paras = markdown_to_paragraphs(text)
        rows, long_paras = check(paras, limit)
        bad = [(s, f) for s, f in rows if f]
        kinds = {}
        for _, f in bad:
            for x in f:
                kinds[x.split()[0]] = kinds.get(x.split()[0], 0) + 1
        lens = [len(words(s)) for s, _ in rows]
        r = {"sentences": len(rows), "comply": 1 - len(bad) / max(1, len(rows)), "flags": kinds,
             "paragraphs": len(paras), "paragraphs_over_six_sentences": long_paras,
             "mean_words": sum(lens) / max(1, len(lens)), "max_words": max(lens or [0])}
        report[path] = r
        total += len(rows)
        bad_all += len(bad)
        if not a.json:
            print(f"{path}: {r['sentences']} sentences, {100 * r['comply']:.1f}% comply; flags {kinds}; "
                  f"mean {r['mean_words']:.1f} words, longest {r['max_words']}; "
                  f"{long_paras} of {len(paras)} paragraphs have more than six sentences")
            for s, f in bad[:a.show]:
                print(f"  - [{'; '.join(f)}] {s}")
    share = 1 - bad_all / max(1, total)
    if a.json:
        print(json.dumps({"files": report, "comply": share}, indent=1))
    elif len(a.files) > 1:
        print(f"all: {total} sentences, {100 * share:.1f}% comply")
    if a.min is not None and share < a.min:
        sys.exit(1)


if __name__ == "__main__":
    main()
