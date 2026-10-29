r"""GAP07-S3, focus question 1: interval-arithmetic certification of the gap equation it7:gap and the radius it7:radius.

Gap equation for two parents of n members each (it7:reduced, it7:depth, check (13)):
    F(d) = rho (s - d) - 2 d w(d),   w(d) = n e^{-a d^2} / (n - 1 + n e^{-a d^2}) = n / (n + (n - 1) e^{a d^2}),
with rho = 0.005 and a = n lam / (2T) = 3n (exact Bayes), or a = lam/(2T) = 3 with n = 3 (control at synchrony).
Radius (it7:radius-fn): Gamma_n(d) = log((2n-1)/pi_p) - 2 log(n - 1 + n e^{-n lam d^2/2}) - n lam d^2 / 4, pi_p = 0.95/11.

Method (mpmath iv, 30 digits, outward rounding). Zeros of F in (0, infinity): F(0) = rho s > 0; for d >= s both terms of F
are <= 0 and -2 d w(d) < 0, so F < 0 on [s, infinity). On [0, s] an adaptive bisection classifies every subinterval I:
either 0 is not in F(I) (no zero), or 0 is not in F'(I) (F strictly monotone on I, so the number of zeros in I is 0 or 1
according to the signs of F at the endpoints, evaluated as point intervals). Every zero is then enclosed by bisection on a
monotone bracket. Every printed enclosure is rigorous for the displayed formula; it does not certify the reduction r(t).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
from mpmath import iv, mp, mpf, nstr

iv.dps = 30
mp.dps = 30                     # midpoints and endpoints carry the interval precision exactly
RHO = iv.mpf("0.005")
PP = iv.mpf("0.95") / 11
LAM = iv.mpf(6)


def make_F(n, s, a):
    n = iv.mpf(n); s = iv.mpf(s); a = iv.mpf(a)

    def F(d):
        w = n / (n + (n - 1) * iv.exp(a * d * d))
        return RHO * (s - d) - 2 * d * w

    def dF(d):
        e = iv.exp(a * d * d)
        den = n + (n - 1) * e
        w = n / den
        wp = -n * (n - 1) * e * 2 * a * d / den ** 2
        return -RHO - 2 * w - 2 * d * wp
    return F, dF


def lo(x):
    r"""Exact lower endpoint of an interval as an mpf (mp.prec equals iv.prec, so the conversion is exact)."""
    return mpf(x._mpi_[0])


def hi(x):
    return mpf(x._mpi_[1])


def zeros_on(F, dF, A, B, maxdepth=60):
    r"""Rigorous count of zeros of F on [A, B] and monotone brackets containing them."""
    stack = [(mpf(A), mpf(B), 0)]
    brackets = []
    nboxes = 0
    while stack:
        a_, b_, dep = stack.pop()
        nboxes += 1
        I = iv.mpf([a_, b_])
        v = F(I)
        if lo(v) > 0 or hi(v) < 0:
            continue
        dv = dF(I)
        if lo(dv) > 0 or hi(dv) < 0:
            fa, fb = F(iv.mpf(a_)), F(iv.mpf(b_))
            if (hi(fa) < 0 < lo(fb)) or (hi(fb) < 0 < lo(fa)):
                brackets.append((a_, b_, 1 if lo(dv) > 0 else -1))
                continue
            if (lo(fa) > 0 and lo(fb) > 0) or (hi(fa) < 0 and hi(fb) < 0):
                continue                                   # monotone with equal end signs: no zero
        if dep >= maxdepth:
            raise RuntimeError(f"undecided box [{a_}, {b_}]")
        m = (a_ + b_) / 2
        stack.append((a_, m, dep + 1)); stack.append((m, b_, dep + 1))
    # merge brackets that share an endpoint (a zero exactly at a split point is counted twice otherwise)
    brackets.sort()
    merged = []
    for br in brackets:
        if merged and merged[-1][1] == br[0] and merged[-1][2] == br[2]:
            fm = F(iv.mpf(br[0]))
            if lo(fm) > 0 or hi(fm) < 0:
                merged.append(br)
            else:
                raise RuntimeError("zero at a split point")
        else:
            merged.append(br)
    return merged, nboxes


def refine(F, a_, b_, sgn, width=mpf("1e-12")):
    a_, b_ = mpf(a_), mpf(b_)
    while b_ - a_ > width:
        m = (a_ + b_) / 2
        fm = F(iv.mpf(m))
        if lo(fm) > 0:
            if sgn > 0: b_ = m
            else: a_ = m
        elif hi(fm) < 0:
            if sgn > 0: a_ = m
            else: b_ = m
        else:
            break
    return a_, b_


def certify_sign(F, A, B, test, maxdepth=40):
    stack = [(mpf(A), mpf(B), 0)]
    worst = None
    while stack:
        a_, b_, dep = stack.pop()
        v = F(iv.mpf([a_, b_]))
        if test(v):
            worst = v if worst is None else worst
            continue
        if dep >= maxdepth:
            return False
        m = (a_ + b_) / 2
        stack.append((a_, m, dep + 1)); stack.append((m, b_, dep + 1))
    return True


def report_gap(n, s, a, label, extra=()):
    F, dF = make_F(n, s, a)
    f0 = F(iv.mpf(0))
    assert lo(f0) > 0
    fs = F(iv.mpf(s))
    assert hi(fs) < 0
    brs, nb = zeros_on(F, dF, "0", s)
    roots = []
    for a_, b_, sg in brs:
        ra, rb = refine(F, a_, b_, sg)
        roots.append((ra, rb, sg))
    print(f"[{label}] n={n}, s={s}, a={a}: F(0)={nstr(lo(f0), 6)}>0, F(s)={nstr(hi(fs), 6)}<0, F<0 on [s,inf);"
          f" zeros in (0,inf): exactly {len(roots)} ({nb} boxes)")
    for ra, rb, sg in roots:
        slope = dF(iv.mpf([ra, rb]))
        print(f"    zero in [{nstr(ra, 12)}, {nstr(rb, 12)}], F' there in [{nstr(lo(slope), 6)}, {nstr(hi(slope), 6)}],"
              f" crossing {'up' if sg > 0 else 'down'}")
    for (A, B, kind, bound) in extra:
        if kind == "neg":
            ok = certify_sign(F, A, B, lambda v, c=mpf(bound): hi(v) < c)
            print(f"    F < {bound} on [{A}, {B}]: {ok}")
        else:
            ok = certify_sign(F, A, B, lambda v, c=mpf(bound): lo(v) > c)
            print(f"    F > {bound} on [{A}, {B}]: {ok}")
    return roots


print("== Gap equation, exact Bayes (a = 3n) ==")
R = {}
R[(3, "1.1")] = report_gap(3, "1.1", 9, "check (9)/(13) n=3")
R[(2, "1.1")] = report_gap(2, "1.1", 6, "check (13) n=2")
R[(4, "1.1")] = report_gap(4, "1.1", 12, "check (13) n=4")
R[(6, "1.1")] = report_gap(6, "1.1", 18, "check (13) n=6")
R[(3, "1.2")] = report_gap(3, "1.2", 9, "check (14) s=1.2")
R[(3, "1.3")] = report_gap(3, "1.3", 9, "check (14) s=1.3")
R[(3, "1.0")] = report_gap(3, "1.0", 9, "s=1.0 (not claimed)")

print("== Trapping certificates for the depth-one claims at s = 1.2, 1.3 (check (14) handoff gaps 0.996, 1.226) ==")
for s, hg in (("1.2", "0.996"), ("1.3", "1.2256")):
    F, dF = make_F(3, s, 9)
    (_, _, _), (ua, ub, _), (sa, sb, _) = R[(3, s)]
    lo_band = ua + mpf("0.005")
    okp = certify_sign(F, lo_band, sa - mpf("1e-9"), lambda v: lo(v) > 0)
    okn = certify_sign(F, sb + mpf("1e-9"), mpf(s), lambda v: hi(v) < 0)
    # smallest positive value of F on [d_u + 0.005, d_s - 0.005], as a margin against |r| <= 1e-10
    m1 = None
    for c in ("1e-5", "5e-6", "2e-6", "1e-6", "5e-7"):
        if certify_sign(F, lo_band, sa - mpf("0.005"), lambda v, c=mpf(c): lo(v) > c):
            m1 = c; break
    m2 = "n/a: d_s + 0.005 > s, and F < 0 on [s, inf) by sign"
    if sb + mpf("0.005") < mpf(s):
        for c in ("1e-5", "5e-6", "2e-6", "1e-6", "5e-7"):
            if certify_sign(F, sb + mpf("0.005"), mpf(s), lambda v, c=mpf(c): hi(v) < -c):
                m2 = c; break
    print(f"  s={s}: handoff gap {hg} > d_u + 0.005 = {nstr(ub + mpf('0.005'), 8)}: {mpf(hg) > ub + mpf('0.005')};"
          f" F > 0 on [d_u+0.005, d_s): {okp}; F < 0 on (d_s, s]: {okn};"
          f" F > {m1} on [d_u+0.005, d_s-0.005]; F < -{m2} on [d_s+0.005, s]")

print("== Control without feedback at synchrony (a = 3, prior ratio n = 3): only the merged zero? ==")
for s in ("1.1", "1.2", "1.3"):
    report_gap(3, s, 3, f"off/tempered s={s}")

print("== Large-n approximation it7:capture-distance: d^2 = (2T/(n lam)) log(2 n d / ((n-1) rho (s - d))) ==")
for n in (3, 4, 6):
    s = mpf("1.1"); rho = mpf("0.005")
    # fixed-point iteration on the displayed approximation (floating mpf, not interval: it is an approximation)
    from mpmath import log as mlog, sqrt as msqrt
    d = mpf("0.6")
    for _ in range(200):
        d = msqrt(2 / (n * mpf(6)) * mlog(2 * n * d / ((n - 1) * rho * (s - d))))
    du = R[(n, "1.1")][1]
    print(f"  n={n}: approximation {nstr(d, 6)}; certified middle root in [{nstr(du[0], 8)}, {nstr(du[1], 8)}];"
          f" d_u sqrt(n) = {nstr(du[0] * msqrt(n), 5)}")

print("== Radius it7:radius ==")


def make_G(n):
    n_ = iv.mpf(n)

    def G(d):
        return iv.log((2 * n_ - 1) / PP) - 2 * iv.log(n_ - 1 + n_ * iv.exp(-n_ * LAM * d * d / 2)) - n_ * LAM * d * d / 4
    return G


for n, guess in ((2, "1.087"), (3, "0.7685"), (4, "0.6004"), (6, "0.4104")):
    G = make_G(n)
    cond = (2 * n - 1) * PP
    g0 = G(iv.mpf(0))
    # bracket and bisect: G is strictly concave in d^2 with G(0) > 0 (proof of it7:radius), so one sign change
    a_, b_ = mpf(guess) - mpf("0.01"), mpf(guess) + mpf("0.01")
    assert lo(G(iv.mpf(a_))) > 0 and hi(G(iv.mpf(b_))) < 0
    while b_ - a_ > mpf("1e-12"):
        m = (a_ + b_) / 2
        gm = G(iv.mpf(m))
        if lo(gm) > 0: a_ = m
        elif hi(gm) < 0: b_ = m
        else: break
    print(f"  n={n}: (2n-1) pi_p in [{nstr(lo(cond), 8)}, {nstr(hi(cond), 8)}] < 1; Gamma_n(0) in "
          f"[{nstr(lo(g0), 8)}, {nstr(hi(g0), 8)}]; d_*(n) in [{nstr(a_, 12)}, {nstr(b_, 12)}]")
print("  -log(1 - pi_s) =", nstr(-iv.log(iv.mpf("0.95")), 8))
G3 = make_G(3)
print("  Gamma_3 < -2.5 on [1.09, 1.11]:", certify_sign(G3, "1.09", "1.11", lambda v: hi(v) < -2.5))
# ordering claims of the remark after it7:radius (n = 3)
dm, du, dsep = R[(3, "1.1")]
print("  d_m < d_*(3) < d_u - 0.005:", dm[1] < mpf("0.76849") and mpf("0.76851") < du[0] - mpf("0.005"))
# radius below the middle root for other n (relevant to the depth question, not claimed in the text)
for n, (ga, gb) in ((4, ("0.6003", "0.6005")), (6, ("0.4103", "0.4105"))):
    print(f"  n={n}: d_*(n) < d_u(n) - 0.005:", mpf(gb) < R[(n, "1.1")][1][0] - mpf("0.005"))

print("== Saddle-node of the n = 3 gap equation in the record gap s (gap07_window.py locates s_c near 1.0722) ==")
for s in ("1.07", "1.0722", "1.0723", "1.075"):
    F, dF = make_F(3, s, 9)
    brs, nb = zeros_on(F, dF, "0", s)
    print(f"  s = {s}: exactly {len(brs)} zero(s) of F in (0, inf)")
