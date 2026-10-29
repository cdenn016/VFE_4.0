"""Print every cross-reference with its sentence context and the target's kind, title and location,
so that each can be judged by hand for whether it points to the intended object.  Read-only.

Usage: python cref_table.py <doc> [kinds]   kinds: comma list of target kinds to include (default: results)
"""
from __future__ import annotations

import pathlib
import re
import sys

from texscan import doc_files, scan

doc = sys.argv[1] if len(sys.argv) > 1 else "main"
kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "definition", "construction", "lemma", "proposition", "theorem", "corollary", "remark",
    "openproblem", "hypothesis", "chapter", "section", "subsection"]
files = doc_files(doc)
text, index, labels, refs, cites = scan(files)
L = {l.name: l for l in labels}
rows = []
for r in refs:
    for t in r.targets:
        l = L.get(t)
        if l is None or l.kind not in kinds:
            continue
        # sentence context: from the previous period to the next period around the reference
        ctx = r.context
        ctx = re.sub(r"\s+", " ", ctx)
        same_file = "same" if l.file == r.file else "x-file"
        rows.append(f"{r.file}:{r.line} -> {t} [{l.kind}, {l.file}:{l.line}, {same_file}] '{l.title[:80]}'\n      ...{ctx[-260:]}")
out = pathlib.Path(__file__).with_name(f"cref_table_{doc}.out.txt")
out.write_text("\n".join(rows), encoding="utf-8")
print(len(rows), "rows written to", out.name)
