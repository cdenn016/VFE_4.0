r"""GAP02-S2 s1: recompute every closed-form constant of chapters/12_open_formation.tex:186-252
and companion/c12_open_formation.tex:56-103 at 40 significant digits (mpmath).

Items
  [1] open:early-bounds: 68 e^{-8}, 1-68e^{-8} vs 0.977, 3+68e^{-8} vs 3.03, 1+4e^{-32}+4e^{-8} vs 1.0014,
      the derivative bound |g_c'(r)| <= (1+4r^2)e^{-2r^2} checked against the exact g_c' on a grid.
  [2] early phase: tau bound log4/0.977, a(tau) >= A e^{-1.0014*1.42}, 0.239*34, p(tau) lower bound,
      |vdot| bound at a = 8, exponent 0.977/1.0014 vs 0.975, b(t_A) bound at A = 34 and 4096.
  [3] late comparison: a_c = sqrt(log 9 / 2), a_* = sqrt(log 2), S by quadrature, log(4/a_*),
      Bdot(a_c), crossing time 4 -> a_c.
  [4] open:constants: t_A, L at A = 4096; R for D_R = 10; admissibility T bound; radius-1/2 tube
      bound without Gronwall; eps_b needed for A = 4096; A needed for eps_b = 1, 1/4.
  [5] companion c12:96-103: far-pair position, event shifts (null vs full, limit-field values
      quoted in the text), final-merge limits.
Run: OMP_NUM_THREADS=1 ... timeout 30 python s1_constants.py
"""
from mpmath import mp, mpf, exp, log, sqrt, quad, nstr

mp.dps = 40
E8 = exp(-8)
print("[1] open:early-bounds")
print("  68e^-8              =", nstr(68 * E8, 12))
print("  1-68e^-8            =", nstr(1 - 68 * E8, 12), "> 0.977 :", 1 - 68 * E8 > mpf("0.977"))
print("  3+68e^-8            =", nstr(3 + 68 * E8, 12), "< 3.03  :", 3 + 68 * E8 < mpf("3.03"))
up = 1 + 4 * exp(-32) + 4 * E8
print("  1+4e^-32+4e^-8      =", nstr(up, 12), "< 1.0014:", up < mpf("1.0014"))


def gprime(c, r):
    E = exp(-2 * r * r)
    return E * (c * (1 - 4 * r * r) + E) / (c + E) ** 2


worst = mpf(0)
for c in (mpf(1), mpf("1.0003"), mpf("1.5"), mpf(2), mpf(3)):
    for k in range(0, 2001):
        r = mpf(k) / 200  # r in [0, 10]
        bound = (1 + 4 * r * r) * exp(-2 * r * r)
        worst = max(worst, abs(gprime(c, r)) / bound)
print("  max |g_c'|/((1+4r^2)e^{-2r^2}) over r in [0,10], c in {1,..,3}:", nstr(worst, 8), "(<= 1 required)")
print("  34 b e^-8 coefficient: 2*(1+4*2^2)*e^-8 =", nstr(2 * 17 * E8, 10), "= 34e^-8")

print("[2] early phase")
tau_b = log(4) / mpf("0.977")
print("  log4/0.977 =", nstr(tau_b, 10), "< 1.42:", tau_b < mpf("1.42"))
fac = exp(-mpf("1.0014") * mpf("1.42"))
print("  e^{-1.0014*1.42} =", nstr(fac, 10), "> 0.239:", fac > mpf("0.239"), "; 0.239*34 =", nstr(mpf("0.239") * 34, 6))
print("  A_min for a(tau) >= 8 with this factor:", nstr(8 / fac, 8))
p_tau = exp(mpf(-0.5)) / (1 + exp(mpf(-0.5)) + 2 * E8)
print("  p(tau) >= e^{-1/2}/(1+e^{-1/2}+2e^-8) =", nstr(p_tau, 10), "> 1/4:", p_tau > mpf(1) / 4)
a = mpf(8)
vb = 8 * a * a * exp(-2 * a * a) + 4 * (a + 2) * (2 * a + 8) * exp(-2 * (a - 2) ** 2)
print("  |v0dot + vpmdot| bound at a = 8:", nstr(vb, 6), "< 1/100:", vb < mpf(1) / 100)
expo = mpf("0.977") / mpf("1.0014")
print("  0.977/1.0014 =", nstr(expo, 10), ">= 0.975:", expo >= mpf("0.975"))
for A in (34, 4096):
    print(f"  A={A}: 2(A/4)^-0.975 =", nstr(2 * (mpf(A) / 4) ** (-mpf("0.975")), 8),
          "; t_A in [", nstr(log(mpf(A) / 4) / mpf("1.0014"), 8), ",", nstr(log(mpf(A) / 4), 8), "]")

print("[3] late comparison (b = 0 reference)")
a_c = sqrt(log(9) / 2)
a_s = sqrt(log(4) / 2)
print("  a_c = sqrt(log9/2) =", nstr(a_c, 15), "; a_* = sqrt(log4/2) = sqrt(log2) =", nstr(a_s, 15))


def Bbar(a):
    v = exp(-2 * a * a)
    return v / (1 + v)


S = quad(lambda s: 1 / (1 + 4 * Bbar(exp(s))), [log(a_s), log(4)])
t_c = quad(lambda s: 1 / (1 + 4 * Bbar(exp(s))), [log(a_c), log(4)])
v = exp(-2 * a_c * a_c)
Bdot = 4 * a_c ** 2 * v * (1 + 4 * Bbar(a_c)) / (1 + v) ** 2
print("  S =", nstr(S, 12), "<= log(4/a_*) =", nstr(log(4 / a_s), 12), ":", S <= log(4 / a_s))
print("  time 4 -> a_c =", nstr(t_c, 12), "; Bdot(a_c) =", nstr(Bdot, 12), "; B(a_c) =", nstr(Bbar(a_c), 6))
print("  check p = (1-B)/2 >= 2/5 and contrast p - B >= 1/5 at B = 1/5:", (1 - mpf(1) / 5) / 2, (1 - mpf(1) / 5) / 2 - mpf(1) / 5)

print("[4] open:constants at A = 4096")
A = mpf(4096)
tA = log(A / 4)
L = tA + S
print("  t_A (upper) =", nstr(tA, 8), "; L = t_A + S =", nstr(L, 8))
for DR in (10, 12, 33):
    R = (DR + A + 6) * exp(L)
    print(f"  D_R={DR}: R = (D_R+A+6)e^L =", nstr(R, 8), "; admissibility T < 1/(R+3)^2 =", nstr(1 / (R + 3) ** 2, 4),
          "; no-Gronwall radius-1/2 tube T < 0.5/(0.1283 R^3) =", nstr(mpf("0.5") / (mpf("0.1283") * R ** 3), 4))
# G1-01 deviation constant: max_t (1/3) e^{-t}(1-e^{-2t}) at t = log(3)/2
t = log(3) / 2
print("  max_t (1/3)e^{-t}(1-e^{-2t}) =", nstr(exp(-t) * (1 - exp(-2 * t)) / 3, 10), "at t = log3/2")
print("  eps_b needed for A=4096: 2(1024)^-0.975 =", nstr(2 * mpf(1024) ** (-mpf("0.975")), 8))
for eb in (mpf(1), mpf(1) / 4):
    print(f"  A needed for eps_b={nstr(eb,3)} via 2(A/4)^-0.975 < eps_b: A >", nstr(4 * (2 / eb) ** (1 / mpf("0.975")), 8))

print("[5] companion c12:96-103")
print("  final-merge limits: first-stage contrast 1/6-4/6 =", mpf(1) / 6 - mpf(4) / 6,
      "; core-parent contrast 2/6-2/6 = 0; two-block rows (1/4)*4*2/6 =", mpf(4 * 2) / (4 * 6), ", (1/2)*2*4/6 =", mpf(2 * 4) / (2 * 6))
print("  null pair crossing (scaled limit, w/(1+w)=1/10 with b=2e^-t): t =", nstr(log(2 / a_c), 10))
print("  quoted shifts 0.589->0.647:", nstr((mpf("0.647") - mpf("0.589")) / mpf("0.589") * 100, 4), "% ; 8.214->8.271:",
      nstr((mpf("8.271") - mpf("8.214")) / mpf("8.214") * 100, 4), "%")
print("  far pair at A=4096, D_R=10..33: R in [", nstr((10 + A + 6) * exp(L), 4), ",", nstr((33 + A + 6) * exp(L), 4), "]")
