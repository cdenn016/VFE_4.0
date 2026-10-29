"""Cross-document arithmetic consistency of the chapter 18 / companion c18 critical-mode constants.

Checks (exact rational arithmetic) that the numbers printed in chapters/18_link_dynamics.tex and
companion/c18_link_dynamics.tex agree with each other:
  T_c = g^T H^{-1} g / 4 with g^T H^{-1} g = 6538/5059 (c18:326)  versus  T_c = 3269/10118 (ch18:457, c18:331);
  D = 19189456/25593481 (c18:critical-length) and D/(D+4) = 4797364/30390845 (ch18:623, c18 text);
  time-scale factor (D+4)/D ~ 6.34 (both);
  branch amplitude sqrt(-4/c) ~ 6.15297 with c = -657534393590529472/6223404588394626861 (c18).
This checks internal arithmetic only, not the derivation of g, H, D or c.
"""
from fractions import Fraction as F
import math

gHg = F(6538, 5059)
Tc = F(3269, 10118)
D = F(19189456, 25593481)
ratio = F(4797364, 30390845)
c = F(-657534393590529472, 6223404588394626861)
out = [
    f"gHg/4 == T_c: {gHg / 4 == Tc}  (T_c = {float(Tc):.10f})",
    f"D/(D+4) == printed ratio: {D / (D + 4) == ratio}",
    f"(D+4)/D = {float((D + 4) / D):.4f} (printed 'about 6.34')",
    f"c = {float(c):.7f} (printed -0.1056551); sqrt(-4/c) = {math.sqrt(float(-4 / c)):.5f} (printed 6.15297)",
]
print("\n".join(out))
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
