r"""Index result headings and labels of MAgent Theory 2.0 (read-only).

Writes a table: file:line, heading macro, title, label. Also reports which
labels named as "NEW LABELS" in the 2026-10-22 remediation logs still exist,
where they are, and how often they are cross-referenced outside their own file.
"""
import os
import re
import sys

ROOT = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0"
LOGS = r"C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-22-magent-theory-2.0-review/remediation/logs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s00_label_index.out.txt")

HEAD = re.compile(r"\\([a-zA-Z]+)heading\{([^}]*)\}\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
REF = re.compile(r"\\(?:cref|Cref|eqref|ref)\{([^}]*)\}")


def tex_files():
    for sub in ["", "chapters", "appendices", "supplements", "companion", "companion/figures"]:
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".tex"):
                yield os.path.join(sub, f).replace("\\", "/")


def main():
    heads = []
    labels = {}
    refs = {}
    for rel in tex_files():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                for m in HEAD.finditer(line):
                    heads.append((rel, n, m.group(1), m.group(2), m.group(3)))
                    labels.setdefault(m.group(3), []).append((rel, n))
                for m in LABEL.finditer(line):
                    labels.setdefault(m.group(1), []).append((rel, n))
                for m in REF.finditer(line):
                    for lab in m.group(1).split(","):
                        refs.setdefault(lab.strip(), []).append((rel, n))
    out = []
    out.append("# headings: %d" % len(heads))
    for rel, n, kind, title, lab in heads:
        out.append("%s:%d\t%s\t%s\t%s" % (rel, n, kind, lab, title))
    # remediation NEW LABELS
    new = []
    for f in sorted(os.listdir(LOGS)):
        txt = open(os.path.join(LOGS, f), encoding="utf-8").read()
        sec = txt.split("## NEW LABELS")
        if len(sec) < 2:
            continue
        body = sec[1].split("## ")[0]
        for lab in re.findall(r"`([a-z]+:[A-Za-z0-9\-]+)`", body):
            new.append((f, lab))
        for lab in re.findall(r"(?<![`\w])((?:lem|def|prop|thm|cor|rem|eq|sec|op|open|geo|inh|fus|smooth|erase|form|hyp|comp):[A-Za-z0-9\-]+)", body):
            new.append((f, lab))
    seen = set()
    out.append("")
    out.append("# remediation NEW LABELS: log, label, defined-at, n-refs-total, n-refs-outside-own-file")
    for f, lab in new:
        if (f, lab) in seen:
            continue
        seen.add((f, lab))
        locs = labels.get(lab, [])
        own = {r for r, _ in locs}
        rs = refs.get(lab, [])
        outside = [r for r in rs if r[0] not in own]
        out.append("%s\t%s\t%s\t%d\t%d" % (f, lab, ";".join("%s:%d" % l for l in locs) or "MISSING", len(rs), len(outside)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT, "headings", len(heads), "new-label rows", len(seen))


if __name__ == "__main__":
    sys.exit(main())
