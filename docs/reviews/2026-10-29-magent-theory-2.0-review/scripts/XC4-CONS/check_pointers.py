r"""Count pointers from the working documents into their external check scripts: numbered 'check (n)'
references and 'numerical check' sentences.  Read-only."""
import re

from texscan import read_lines

out = []
for f in ["Item_1_communication_links.tex", "Item_2_link_dynamics.tex", "Item_3_curvature_hierarchy.tex",
          "Item_7_formation_feedback.tex"]:
    numbered = 0
    numerical = 0
    lines_numbered = []
    for n, l in enumerate(read_lines(f), 1):
        k = len(re.findall(r"checks?~\(", l))
        numbered += k
        if k:
            lines_numbered.append(n)
        numerical += len(re.findall(r"numerical(ly)? check", l, re.I))
    out.append(f"{f}: numbered check pointers {numbered} on lines {lines_numbered}; 'numerical check' phrases {numerical}")
txt = "\n".join(out)
print(txt)
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write(txt + "\n")
