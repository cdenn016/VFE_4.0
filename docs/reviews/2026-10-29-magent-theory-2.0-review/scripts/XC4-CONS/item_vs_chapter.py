"""Compare each result shared (by label) between a working document and its integrated chapter.

For every heading label present in both Item_k and chapter (17, 18, 19), extract the block from the
heading to the next result heading or sectioning command, normalize whitespace, and report the
similarity ratio and a word-level diff for blocks that differ.  Read-only.
"""
from __future__ import annotations

import difflib
import pathlib
import re
import sys

from texscan import ROOT, HEADING_TYPES

PAIRS = [
    ("Item_1_communication_links.tex", "chapters/17_communication_links.tex"),
    ("Item_2_link_dynamics.tex", "chapters/18_link_dynamics.tex"),
    ("Item_3_curvature_hierarchy.tex", "chapters/19_curvature_hierarchy.tex"),
]
HEAD = re.compile(r"\\(" + "|".join(HEADING_TYPES) + r")heading\{")
STOP = re.compile(r"\\(" + "|".join(HEADING_TYPES) + r")heading\{|\\(chapter|section|subsection)\*?\{|\\bibliography")


def blocks(path: str) -> dict[str, tuple[str, str]]:
    text = (ROOT / path).read_text(encoding="utf-8")
    out = {}
    for m in HEAD.finditer(text):
        # find label = second brace group
        i = m.end() - 1
        depth = 0
        j = i
        while True:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        title = text[i + 1:j]
        k = text.index("}", j + 1)
        label = text[j + 2:k]
        nxt = STOP.search(text, k + 1)
        body = text[k + 1:nxt.start() if nxt else len(text)]
        out[label] = (m.group(1), " ".join(title.split()), " ".join(body.split()))
    return out


report = []
for item, chap in PAIRS:
    a = blocks(item)
    b = blocks(chap)
    shared = [l for l in a if l in b]
    only_item = [l for l in a if l not in b]
    only_chap = [l for l in b if l not in a]
    report.append(f"===== {item} vs {chap}: shared {len(shared)}, only in item {only_item}, only in chapter {only_chap}")
    for l in shared:
        ka, ta, xa = a[l]
        kb, tb, xb = b[l]
        r = difflib.SequenceMatcher(None, xa, xb, autojunk=False).ratio()
        flag = "" if r > 0.995 and ka == kb and ta == tb else "  <-- differs"
        report.append(f"  {l}: item[{ka}] chapter[{kb}] ratio={r:.3f}{flag}")
        if flag:
            if ka != kb or ta != tb:
                report.append(f"     heading: item '{ka}: {ta}' / chapter '{kb}: {tb}'")
            wa, wb = xa.split(" "), xb.split(" ")
            sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    continue
                report.append(f"     {op}: ITEM<< {' '.join(wa[i1:i2])[:500]} >> CHAP<< {' '.join(wb[j1:j2])[:500]} >>")
out = pathlib.Path(__file__).with_name("item_vs_chapter.out.txt")
out.write_text("\n".join(report), encoding="utf-8")
print("\n".join(r for r in report if not r.startswith("     ")))
