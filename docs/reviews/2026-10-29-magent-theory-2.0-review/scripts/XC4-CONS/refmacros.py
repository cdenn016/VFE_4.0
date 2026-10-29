"""Count reference macros per file and list \eqref/\ref uses (AUTHORING: refer through \cref or \Cref)."""
import collections, pathlib
from texscan import DOCS, doc_files, scan
out = []
for doc in DOCS:
    text, index, labels, refs, cites = scan(doc_files(doc))
    c = collections.defaultdict(collections.Counter)
    for r in refs:
        c[r.file][r.macro] += 1
    for f, cc in c.items():
        if any(m not in ("cref", "Cref") for m in cc):
            out.append(f"{doc} {f}: {dict(cc)}")
pathlib.Path(__file__).with_name("refmacros.out.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
