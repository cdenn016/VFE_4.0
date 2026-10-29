r"""Restore backslashes lost in a sed rename of the chapter 15 continuum section."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
i = s.index(r"\section{A periodic continuum base}")
a, b = s[:i], s[i:]
BS = "\\"
for w in ["mathsf B", "mathbb T^d", "Phi(z)"]:
    b = b.replace(BS + w, "@@" + w).replace(w, BS + w).replace("@@" + w, BS + w)
b = b.replace("-V_r," + BS + "mathsf Bb_t&=-V_b", "-V_r,& " + BS + "mathsf Bb_t&=-V_b")
b = b.replace(r"\ell_U\ell_V]", r"\ell_U\ell_W]")
open(p, "w", encoding="utf-8").write(a + b)
