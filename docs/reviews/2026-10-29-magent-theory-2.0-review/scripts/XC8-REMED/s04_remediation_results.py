r"""Join the remediation NEW LABELS with the heading index (read-only).

Prints, for every label a 2026-10-22 remediation log declares as new, its heading type and title
when the label is a result heading (theorem, proposition, lemma, corollary, definition, remark,
open problem, hypothesis, construction), sorted by file and line.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "s00_label_index.out.txt")
OUT = os.path.join(HERE, "s04_remediation_results.out.txt")


def main() -> None:
    lines = open(INDEX, encoding="utf-8").read().splitlines()
    heads = {}
    for ln in lines:
        m = re.match(r"^([^\t]+):(\d+)\t(\w+)\t([^\t]+)\t(.*)$", ln)
        if m:
            heads[m.group(4)] = (m.group(1), int(m.group(2)), m.group(3), m.group(5))
    rows = []
    started = False
    for ln in lines:
        if ln.startswith("# remediation NEW LABELS"):
            started = True
            continue
        if not started or not ln.strip():
            continue
        log, lab, loc, nref, nout = ln.split("\t")
        if lab in heads:
            f, n, kind, title = heads[lab]
            rows.append((f, n, kind, lab, title, log, int(nref), int(nout)))
    rows.sort()
    kinds = {}
    out = []
    for f, n, kind, lab, title, log, nref, nout in rows:
        kinds[kind] = kinds.get(kind, 0) + 1
        out.append("%s:%d\t%s\t%s\t%s\t%s\trefs=%d outside=%d" % (f, n, kind, lab, title, log, nref, nout))
    out.insert(0, "counts by heading kind: %s" % kinds)
    open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
