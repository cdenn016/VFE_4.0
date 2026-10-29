r"""Find every stated bound on the evaluator coupling J in chapters 09 and 10 and companion c09,
to check the cross-chapter claim at chapters/10_boundary_closure.tex:321 that the range |J|<1/4
is the one used in the public-compression chapter.  Read-only."""
import re

from texscan import read_lines

out = []
for f in ["chapters/09_public_compression.tex", "companion/c09_public_compression.tex",
          "chapters/10_boundary_closure.tex"]:
    for n, l in enumerate(read_lines(f), 1):
        for m in re.finditer(r"(\|J\||J)\s*(<|\\le\b|\\leq\b|\\in\b|\\notin\b)", l):
            out.append(f"{f}:{n}: ...{l[max(0, m.start() - 70):m.end() + 45]}...")
txt = "\n".join(out)
print(txt)
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write(txt + "\n")
