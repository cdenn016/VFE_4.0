r"""Result-heading hygiene: hand-made headings outside the preamble macros, 'Lemma \ref' style
references, \eqref/\ref counts, sub/subsubsection use per document, and \square end marks.  Read-only."""
import collections
import re

from texscan import DOCS, doc_files, read_lines

out = []
for doc in DOCS:
    c = collections.Counter()
    ex = []
    for f in doc_files(doc):
        for n, l in enumerate(read_lines(f), 1):
            for pat, name in [(r"\\paragraph\{", "paragraph"), (r"\\textbf\{(Theorem|Proposition|Lemma|Corollary|Definition|Remark)", "bold-heading"),
                              (r"(Lemma|Theorem|Proposition|Corollary|Definition|Remark|Equation|Section|Chapter)~?\s*\\ref\{", "name+ref"),
                              (r"\\subsubsection\*?\{", "subsubsection"), (r"\\subsection\*?\{", "subsection"),
                              (r"\\square", "square"), (r"\\qed", "qed"), (r"\\begin\{proof\}", "proof")]:
                for m in re.finditer(pat, l):
                    c[name] += 1
                    if name in ("paragraph", "bold-heading", "name+ref", "square"):
                        ex.append(f"   {f}:{n}: {l[max(0, m.start() - 40):m.end() + 60]}")
    out.append(f"{doc}: {dict(c)}")
    out.extend(ex[:12])
txt = "\n".join(out)
print(txt)
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write(txt + "\n")
