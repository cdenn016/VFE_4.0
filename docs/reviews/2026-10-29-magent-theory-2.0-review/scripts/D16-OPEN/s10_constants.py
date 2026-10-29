r"""D16-OPEN s10: temperature implied by the written h-step of open:routing (D16-OPEN-07).

open:persistence bounds the trajectory deviation by Delta = e (e^{CL}-1)/C.  For the far pair the
uniform field error on the tube contains the common-mode term (2/3) h^2 (R+3)^3 (open:constants).
Keeping Delta < 1/2 therefore needs h^2 < 0.5 C / ((2/3)(R+3)^3 (e^{CL}-1)).  We evaluate this for
C = 1 (the common mode of the far pair has slope -1, so C >= 1), C = 4.32 (sampled maximum of the
limit-field Jacobian norm, 2026-10-22 G1-04) and C = 64.5 (absolute bound, G1-04), and the
admissibility bound h < 1/(R+3).  Also 2(A/4)^{-0.975} at A = 4096 (the epsilon_b that A = 4096 needs).
"""
import mpmath as mp

A = 4096
L = mp.log(A / 4) + mp.mpf("1.426696")
R = (12 + A + 6) * mp.e ** L
out = ["L = %s, R = %s" % (mp.nstr(L, 6), mp.nstr(R, 7))]
for C in (1, 4.32, 64.5):
    C = mp.mpf(C)
    h2 = 0.5 * C / (mp.mpf(2) / 3 * (R + 3) ** 3 * (mp.e ** (C * L) - 1))
    out.append("C = %s: T = h^2 < %s" % (mp.nstr(C, 4), mp.nstr(h2, 3)))
out.append("admissibility h(R+3) < 1: T < %s" % mp.nstr(1 / (R + 3) ** 2, 3))
out.append("tube without Gronwall (0.1283 h^2 R^3 < 1/2): T < %s" % mp.nstr(0.5 / (mp.mpf("0.1283") * R ** 3), 3))
out.append("2(A/4)^(-0.975) at A=4096: %s" % mp.nstr(2 * (mp.mpf(A) / 4) ** mp.mpf("-0.975"), 4))
print("\n".join(out))
