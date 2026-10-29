r"""Read-only scanner for the MAgent Theory 2.0 TeX sources (XC4-CONS lens).

Builds, for each compiled document, the ordered list of input files, then extracts
labels (\label and the result-heading macros of preamble.tex), references
(\cref, \Cref, \ref, \eqref, \crefrange), citations (\cite variants) and display
math blocks, each with file and line number.  Nothing is written outside the
lens scripts directory.
"""
from __future__ import annotations

import re
import pathlib
from dataclasses import dataclass, field

ROOT = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")

HEADING_TYPES = ["definition", "construction", "lemma", "proposition", "theorem",
                 "corollary", "remark", "openproblem", "hypothesis"]


def strip_comments(line: str) -> str:
    """Remove a TeX comment (an unescaped %) from one line."""
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def read_lines(rel: str) -> list[str]:
    p = ROOT / rel
    if not p.suffix:
        p = p.with_suffix(".tex")
    return [strip_comments(l.rstrip("\n")) for l in p.read_text(encoding="utf-8").split("\n")]


def expand_inputs(rel: str, seen: list[str] | None = None) -> list[str]:
    """Return rel followed by recursively input files in order (depth first at the input line)."""
    if seen is None:
        seen = []
    rel = rel if rel.endswith(".tex") else rel + ".tex"
    seen.append(rel)
    for line in read_lines(rel):
        for m in re.finditer(r"\\input\{([^}]+)\}", line):
            tgt = m.group(1)
            tgt = tgt if tgt.endswith(".tex") else tgt + ".tex"
            if tgt == "preamble.tex":
                continue
            expand_inputs(tgt, seen)
    return seen


DOCS = {
    "main": "MAgent_Theory_2.0.tex",
    "supplement": "MAgent_Theory_2.0_supplement.tex",
    "companion": "MAgent_Theory_2.0_companion.tex",
    "item1": "Item_1_communication_links.tex",
    "item2": "Item_2_link_dynamics.tex",
    "item3": "Item_3_curvature_hierarchy.tex",
    "item7": "Item_7_formation_feedback.tex",
}


def doc_files(doc: str) -> list[str]:
    return expand_inputs(DOCS[doc])


def balanced(s: str, i: int, open_c: str = "{", close_c: str = "}") -> tuple[str, int] | None:
    """Given s[i] == open_c, return (content, index after closing brace)."""
    if i >= len(s) or s[i] != open_c:
        return None
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return None


@dataclass
class Label:
    name: str
    file: str
    line: int
    kind: str          # heading type, 'chapter', 'section', 'equation', 'other'
    title: str = ""


@dataclass
class Ref:
    targets: list[str]
    file: str
    line: int
    macro: str
    context: str


@dataclass
class Cite:
    keys: list[str]
    file: str
    line: int
    macro: str
    context: str


def joined(files: list[str]) -> tuple[str, list[tuple[int, str, int]]]:
    """Concatenate files; return text and a list of (offset, file, line) for each line start."""
    parts = []
    index = []
    off = 0
    for f in files:
        for n, line in enumerate(read_lines(f), start=1):
            index.append((off, f, n))
            parts.append(line + "\n")
            off += len(line) + 1
    return "".join(parts), index


def locate(index, pos: int) -> tuple[str, int]:
    lo, hi = 0, len(index) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if index[mid][0] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return index[lo][1], index[lo][2]


def scan(files: list[str]):
    text, index = joined(files)
    labels: list[Label] = []
    refs: list[Ref] = []
    cites: list[Cite] = []
    # heading macros
    for m in re.finditer(r"\\(" + "|".join(HEADING_TYPES) + r")heading(?=\{)", text):
        b1 = balanced(text, m.end())
        if not b1:
            continue
        title, j = b1
        b2 = balanced(text, j)
        if not b2:
            continue
        lab, _ = b2
        f, ln = locate(index, m.start())
        labels.append(Label(lab.strip(), f, ln, m.group(1), " ".join(title.split())))
    # plain labels with the environment or sectioning command they belong to
    for m in re.finditer(r"\\label\{([^}]+)\}", text):
        f, ln = locate(index, m.start())
        before = text[max(0, m.start() - 400):m.start()]
        kind = "other"
        title = ""
        sec = list(re.finditer(r"\\(chapter|section|subsection|subsubsection)\*?\{", before))
        envs = list(re.finditer(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|figure|table|enumerate)\}", before))
        ends = list(re.finditer(r"\\end\{(equation\*?|align\*?|gather\*?|multline\*?|figure|table|enumerate)\}", before))
        last_sec = sec[-1].start() if sec else -1
        last_env = envs[-1].start() if envs else -1
        last_end = ends[-1].start() if ends else -1
        item_before = re.search(r"\\item(\[[^\]]*\])?[^\\]*$", before)
        if last_env > last_end and last_env > last_sec:
            kind = envs[-1].group(1).rstrip("*")
            if kind == "enumerate":
                kind = "item"
        elif last_sec > -1 and (m.start() - (max(0, m.start() - 400) + sec[-1].end())) < 300 and "\n\n" not in before[sec[-1].start():]:
            kind = sec[-1].group(1)
            b = balanced(before, sec[-1].end() - 1)
            if b:
                title = " ".join(b[0].split())
        labels.append(Label(m.group(1).strip(), f, ln, kind, title))
    for m in re.finditer(r"\\(cref|Cref|ref|eqref|autoref|pageref|labelcref|namecref|nameCref)\*?\{([^}]+)\}", text):
        f, ln = locate(index, m.start())
        ctx = text[max(0, m.start() - 160):m.end() + 60].replace("\n", " ")
        refs.append(Ref([t.strip() for t in m.group(2).split(",")], f, ln, m.group(1), ctx))
    for m in re.finditer(r"\\(crefrange|Crefrange)\{([^}]+)\}\{([^}]+)\}", text):
        f, ln = locate(index, m.start())
        ctx = text[max(0, m.start() - 160):m.end() + 60].replace("\n", " ")
        refs.append(Ref([m.group(2).strip(), m.group(3).strip()], f, ln, m.group(1), ctx))
    for m in re.finditer(r"\\(cite|citet|citep|citealp|citealt|citeauthor|citeyear|citeyearpar|Citet|Citep|nocite)\*?((?:\[[^\]]*\]){0,2})\{([^}]+)\}", text):
        f, ln = locate(index, m.start())
        ctx = text[max(0, m.start() - 200):m.end() + 40].replace("\n", " ")
        cites.append(Cite([k.strip() for k in m.group(3).split(",")], f, ln, m.group(1) + m.group(2), ctx))
    return text, index, labels, refs, cites


def parse_bib(rel: str) -> dict[str, dict]:
    """Minimal BibTeX parser: returns key -> {'type':..., fields..., '_line': n}."""
    p = ROOT / rel
    src = p.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", src):
        typ = m.group(1).lower()
        if typ in ("comment", "preamble", "string"):
            continue
        start = src.index("{", m.start())
        b = balanced(src, start)
        body = b[0] if b else src[start + 1:]
        line = src.count("\n", 0, m.start()) + 1
        fields = {"type": typ, "_line": line, "_file": rel}
        rest = body[len(m.group(2)) + 1:] if body.startswith(m.group(2)) else body.split(",", 1)[1]
        i = 0
        while i < len(rest):
            fm = re.match(r"\s*,?\s*([A-Za-z_-]+)\s*=\s*", rest[i:])
            if not fm:
                break
            name = fm.group(1).lower()
            i += fm.end()
            if i < len(rest) and rest[i] == "{":
                bb = balanced(rest, i)
                val, i = (bb[0], bb[1]) if bb else (rest[i + 1:], len(rest))
            elif i < len(rest) and rest[i] == '"':
                j = rest.index('"', i + 1)
                val, i = rest[i + 1:j], j + 1
            else:
                vm = re.match(r"[^,}\s]+", rest[i:])
                val = vm.group(0) if vm else ""
                i += len(val)
            fields[name] = " ".join(val.split())
        key = m.group(2)
        if key in entries:
            fields["_duplicate_in_same_file"] = True
        entries[key] = fields
    return entries


if __name__ == "__main__":
    for d in DOCS:
        print(d, doc_files(d))
