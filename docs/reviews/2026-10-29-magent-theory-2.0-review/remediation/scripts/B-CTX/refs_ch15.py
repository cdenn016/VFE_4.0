r"""Rename generated labels and convert \eqref to \cref in one .tex file."""
import re
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
for a, b in [("ctx:f:statement:1", "ctx:f:transport-equation"),
             ("ctx:f:statement:3", "ctx:f:elimination"),
             ("ctx:c:statement:2", "ctx:c:reduced-energy"),
             ("ctx:c:statement:5", "ctx:c:energy-identity")]:
    s = s.replace(a, b)
s = re.sub(r"\\eqref\{([^}]*)\}--\\eqref\{([^}]*)\}", r"\\crefrange{\1}{\2}", s)
s = re.sub(r"Equation \\eqref\{", r"\\Cref{", s)
s = re.sub(r"(^|\. )\\eqref\{", lambda m: m.group(1) + r"\Cref{", s, flags=re.M)
s = s.replace(r"\eqref{", r"\cref{")
open(p, "w", encoding="utf-8").write(s)
print(s.count(r"\eqref"), s.count(r"\crefrange"))
