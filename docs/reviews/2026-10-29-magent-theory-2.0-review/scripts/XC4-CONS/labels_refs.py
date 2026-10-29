"""Label/reference integrity per document: duplicates, undefined targets, unused result labels,
label-prefix conventions, and labels shared between documents.  Read-only."""
from __future__ import annotations

import collections
import json
import pathlib
import sys

from texscan import DOCS, doc_files, scan

OUT = pathlib.Path(__file__).with_name("labels_refs.out.txt")
lines_out: list[str] = []


def emit(s: str = "") -> None:
    lines_out.append(s)


all_labels = {}
for doc in DOCS:
    files = doc_files(doc)
    text, index, labels, refs, cites = scan(files)
    names = collections.Counter(l.name for l in labels)
    dup = {n: c for n, c in names.items() if c > 1}
    defined = {l.name: l for l in labels}
    all_labels[doc] = defined
    used = collections.Counter()
    undefined = []
    for r in refs:
        for t in r.targets:
            used[t] += 1
            if t not in defined:
                undefined.append((t, r.file, r.line))
    emit(f"===== {doc}: {len(files)} files, {len(labels)} labels, {len(refs)} ref commands, "
         f"{sum(len(r.targets) for r in refs)} ref targets, {len(cites)} cite commands")
    kinds = collections.Counter(l.kind for l in labels)
    emit("label kinds: " + json.dumps(dict(kinds)))
    emit(f"duplicate labels: {dup}")
    emit(f"undefined ref targets: {undefined}")
    unused = [l for l in labels if used[l.name] == 0]
    emit(f"unused labels: {len(unused)} of {len(labels)}")
    uk = collections.Counter(l.kind for l in unused)
    emit("unused by kind: " + json.dumps(dict(uk)))
    for l in unused:
        if l.kind in ("theorem", "proposition", "lemma", "corollary"):
            emit(f"  UNUSED {l.kind:12s} {l.name:45s} {l.file}:{l.line}  ({l.title[:70]})")
    # prefix census per file
    pref = collections.defaultdict(collections.Counter)
    for l in labels:
        p = l.name.split(":")[0] if ":" in l.name else "(none)"
        pref[l.file][p] += 1
    emit("label prefixes per file:")
    for f, c in pref.items():
        emit(f"  {f}: {dict(c)}")
    emit()

# labels defined in more than one document
emit("===== label names defined in more than one document")
by_name = collections.defaultdict(list)
for doc, d in all_labels.items():
    for n, l in d.items():
        by_name[n].append((doc, l.file, l.line, l.kind, l.title[:60]))
multi = {n: v for n, v in by_name.items() if len({x[0] for x in v}) > 1}
emit(f"count: {len(multi)}")
for n, v in sorted(multi.items()):
    titles = {x[4] for x in v}
    flag = "SAME-TITLE" if len(titles) == 1 else "DIFF-TITLE"
    emit(f"  {flag} {n}: " + " | ".join(f"{x[0]}:{x[1]}:{x[2]} [{x[3]}] {x[4]}" for x in v))

OUT.write_text("\n".join(lines_out), encoding="utf-8")
print("\n".join(lines_out[:400]))
