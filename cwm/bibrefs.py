"""Read paper/refs.bib and write the reference list of the website, with a link for each entry.

The website shows every reference of the paper, in the order of the paper, so that a reader can open each source.
The link of an entry is its `url` field. Without one, it is the DOI or the arXiv identifier of the entry.
"""
import html
import re

ACCENTS = {r'\"a': "ä", r'\"o': "ö", r'\"u': "ü", r'\"A': "Ä", r'\"O': "Ö", r'\"U': "Ü", r"\'e": "é", r"\'a": "á",
           r"\'i": "í", r"\'o": "ó", r"\'u": "ú", r"\'E": "É", r"\`e": "è", r"\`a": "à", r"\^e": "ê", r"\~n": "ñ",
           r"\c{c}": "ç", r"\v{s}": "š", r"\v{S}": "Š", r"\'\i": "í", r"\'{\i}": "í", r"\v{c}": "č", r"\v{z}": "ž", r"\ss": "ß", r"\o": "ø", r"\l": "ł"}


def tex_to_text(s):
    """Plain text from the LaTeX of a BibTeX field."""
    for k, v in ACCENTS.items():
        s = s.replace("{" + k + "}", v).replace(k, v)
    s = re.sub(r"\\(?:emph|textit|textbf|textsc|mathrm|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:url|nolinkurl)\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\&", "&").replace(r"\%", "%").replace(r"\_", "_").replace("---", "—").replace("--", "–")
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bib(path):
    """Return {key: {field: raw value}} for every entry of a BibTeX file."""
    text = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key, i = m.group(1).lower(), m.group(2), m.end()
        depth, j = 1, i
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        body = text[i:j - 1]
        fields, k = {"_type": kind}, 0
        while True:
            f = re.compile(r"\s*([A-Za-z_]+)\s*=\s*").match(body, k)
            if not f:
                break
            name, k = f.group(1).lower(), f.end()
            if body[k] == "{":
                depth, start = 1, k + 1
                k += 1
                while depth:
                    depth += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                value = body[start:k - 1]
            elif body[k] == '"':
                end = body.index('"', k + 1)
                value, k = body[k + 1:end], end + 1
            else:
                end = re.compile(r"[,\n]").search(body, k)
                end = end.start() if end else len(body)
                value, k = body[k:end].strip(), end
            fields[name] = value.strip()
            comma = body.find(",", k)
            if comma < 0:
                break
            k = comma + 1
        entries[key] = fields
    return entries


def entry_url(f):
    """The link of an entry: its url field, else its DOI, else its arXiv identifier."""
    if f.get("url"):
        return f["url"]
    if f.get("doi"):
        return "https://doi.org/" + f["doi"]
    for field in ("eprint", "journal", "howpublished", "note"):
        m = re.search(r"(\d{4}\.\d{4,5})", f.get(field, ""))
        if m:
            return "https://arxiv.org/abs/" + m.group(1)
    return None


def authors(f):
    names = [n.strip() for n in re.split(r"\s+and\s+", tex_to_text(f.get("author", "")))]
    out, others = [], False
    for n in names:
        if n.lower() == "others":
            others = True
            continue
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
        else:
            parts = n.split()
            last, first = parts[-1], " ".join(parts[:-1])
        initials = " ".join("-".join(q[0] + "." for q in p.split("-") if q) for p in first.split() if p and p[0].isalpha())
        out.append((initials + " " + last).strip() if initials else last)
    if len(out) > 6:
        out, others = out[:3], True
    if not out:
        return ""
    if others:
        return ", ".join(out) + " and others"
    return out[0] if len(out) == 1 else ", ".join(out[:-1]) + " and " + out[-1]


def venue(f):
    for field in ("journal", "booktitle", "howpublished", "publisher", "institution"):
        if f.get(field):
            v = tex_to_text(f[field])
            v = re.sub(r"^arXiv preprint\s*", "", v)
            return v
    return ""


def year(f):
    """The year, unless the venue already ends with it (as in "ACL 2025")."""
    y = tex_to_text(f.get("year", ""))
    return "" if y and venue(f).endswith(y) else y


def entry_html(f):
    url = entry_url(f)
    title = html.escape(tex_to_text(f.get("title", "")).rstrip("."))
    t = f'<a href="{html.escape(url)}">{title}</a>' if url else title
    parts = [html.escape(authors(f)) + ".", t + "."]
    tail = ", ".join(x for x in (html.escape(venue(f)), year(f)) if x)
    if tail:
        parts.append(tail + ".")
    return " ".join(p for p in parts if p.strip(". "))


def md_escape(s):
    return re.sub(r"([\\`*_\[\]<>|])", r"\\\1", s)


def entry_markdown(f):
    """One reference as a line of Markdown, with the title as a link."""
    url = entry_url(f)
    title = md_escape(tex_to_text(f.get("title", "")).rstrip("."))
    parts = [md_escape(authors(f)) + ".", (f"[{title}]({url})" if url else title) + "."]
    tail = ", ".join(x for x in (md_escape(venue(f)), year(f)) if x)
    if tail:
        parts.append(tail + ".")
    return " ".join(p for p in parts if p.strip(". "))


def cite_order(aux_path):
    order = []
    for m in re.finditer(r"\\citation\{([^}]*)\}", open(aux_path).read()):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    return order
