r"""AUTHORING.md prohibition scan over the scientific text of all seven documents (read-only).

Checks: spacing macros \, \; \! (and \:), list environments, decorative rules, status-badge
words, references to the author's other (unpublished) documents, repository/tool/workflow
vocabulary, numerical-check language, URLs, and unpunctuated display equations.
"""
from __future__ import annotations

import collections
import pathlib
import re

from texscan import DOCS, doc_files, read_lines

OUT = pathlib.Path(__file__).with_name("authoring_scan.out.txt")
out: list[str] = []

files = []
for d in DOCS:
    for f in doc_files(d):
        if f not in files:
            files.append(f)

CHECKS = {
    # NOTE: this lookahead wrongly skips \, followed by a letter (e.g. ess\,sup); the authoritative
    # spacing-macro count is spacing_recount.py.
    "spacing-macro": r"\\[,;!:](?![a-zA-Z])",
    "list-env": r"\\begin\{(itemize|enumerate|description)\}",
    "rule": r"\\(hrule|rule\{|hline|toprule|midrule|bottomrule|noindent\\rule|rule\[)",
    "badge-word": r"\b(INCONCLUSIVE|PROVED|PROVEN|VERIFIED|OPEN|CONFIRMED|PLAUSIBLE|REFUTED|TODO|FIXME|XXX)\b",
    "other-doc": r"(main manuscript|working document|the companion|the supplement|this manuscript|Item~?\s?[0-9]|Items~?\s?[0-9]|chapter~[0-9]+ of|Chapter~[0-9]+|chapters?~[0-9]|Chapter [0-9]|chapter [0-9])",
    "earlier-theory": r"(MultiAgentELBO|PIFB|VFE[_ ]?[0-9]|earlier manuscript|earlier theory|previous version|version 1\.0|2\.0 )",
    "repo-tool": r"(\.py\b|checks?\.py|item[0-9]_checks|github|sympy|numpy|scipy|mpmath|python|Lean\b|Mathlib|script|repository|commit|remediation|audit|review record|workflow|Codex|Claude|agent-produced)",
    "numeric-claim": r"(numerical check|numerical integration|numerically|integrations indicate|fixed-step|Runge|interval arithmetic|validated arithmetic|check~\(|checks~\(|to \$10\^\{-[0-9]+\}\$|residual)",
    "url": r"(\\url\{|\\href\{|https?://)",
    "summary-heading": r"\\(section|subsection)\*?\{[^}]*(Summary|Conclusion|Status|Scope|Remaining|remaining|What is still|Next steps|Outlook)[^}]*\}",
}

counts = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for f in files:
    if f.endswith(".bib"):
        continue
    for n, line in enumerate(read_lines(f), 1):
        for name, pat in CHECKS.items():
            for m in re.finditer(pat, line):
                counts[name][f] += 1
                s = max(0, m.start() - 90)
                examples[name].append(f"{f}:{n}: ...{line[s:m.end()+70]}...")

for name in CHECKS:
    out.append(f"===== {name}: total {sum(counts[name].values())}")
    for f, c in counts[name].items():
        out.append(f"   {f}: {c}")
    for e in examples[name][:400]:
        out.append("      " + e)

# display-equation punctuation: equation/align/gather/multline and \[ \]
out.append("===== unpunctuated display equations (last token before the end of the display)")
envpat = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.*?)\\end\{\1\}|\\\[(.*?)\\\]", re.S)
tot = 0
bad = []
for f in files:
    lines = read_lines(f)
    text = "\n".join(lines)
    for m in envpat.finditer(text):
        body = m.group(2) if m.group(2) is not None else m.group(3)
        body = re.sub(r"\\label\{[^}]*\}|\\tag\{[^}]*\}|\\nonumber|\\notag", "", body)
        b = body.rstrip()
        # drop trailing \end{split} etc.
        b = re.sub(r"(\\end\{(split|aligned|cases|pmatrix|bmatrix|array|gathered)\}\s*)+$", lambda mm: mm.group(0), b)
        tail = re.sub(r"\s+", "", b)
        tail = re.sub(r"(\\end\{(split|aligned|gathered)\})+$", "", tail)
        tail = re.sub(r"(\\right[.)\]\}|\\]?)$", "", tail) if False else tail
        tot += 1
        last = tail[-12:]
        ok = bool(re.search(r"([.,;:]|[.,;:]\}|[.,;:]\\\\?)$", tail)) or bool(re.search(r"[.,;:](\\right[.)\]|])?\}*$", tail)) or tail.endswith(".\\end{cases}") or re.search(r"[.,;:]\\end\{(cases|array|pmatrix|bmatrix)\}$", tail)
        # text after the display starting with a punctuation mark also counts as punctuated
        after = text[m.end():m.end() + 3].lstrip()
        if not ok and not (after[:1] in ".,;:"):
            line = text.count("\n", 0, m.start()) + 1
            bad.append(f"{f}:{line}: ...{tail[-60:]}  || next: {text[m.end():m.end()+50]!r}")
out.append(f"displays checked: {tot}; without terminal punctuation: {len(bad)}")
out.extend("   " + b for b in bad)

OUT.write_text("\n".join(out), encoding="utf-8")
print("\n".join(l for l in out if l.startswith("=====") or l.startswith("   ") and not l.startswith("      ")))
