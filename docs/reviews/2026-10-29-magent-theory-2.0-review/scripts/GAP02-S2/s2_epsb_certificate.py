r"""GAP02-S2 s2: validated (mpmath.iv, directed rounding) certificate of the late comparison in the
proof of open:routing (chapters/12_open_formation.tex:222-228), replacing the nonconstructive eps_b.

Reduced core system open:core (12:176-183), state (a, b), a = half distance of pair centres,
b = internal half-gap; w = e^{-2b^2}, v0 = e^{-2a^2}, v_pm = e^{-2(a pm b)^2},
Z_pm = 1 + w + v0 + v_pm,  c_p = w(1/Z_+ + 1/Z_-),  c_0 = v0(1/Z_+ + 1/Z_-),
G_pm = (a pm b) v_pm / Z_pm,
  f_a = -a - 2 a c_0 - 2 (G_+ + G_-),     f_b = -b - 2 b c_p - 2 (G_+ - G_-).
Parent row  B = (1/2)[(v0+v_+)/Z_+ + (v0+v_-)/Z_-]  (= B^q_{P1P2}), first-pair weights p_pm = w/Z_pm,
first-pair contrasts (w - v0 - v_pm)/Z_pm.

Region  Omega_eps = { a_* <= a <= 4, 0 <= b <= eps a / 4 },  a_* = sqrt(log 2) (B = 1/5 at b = 0).
Certificate (each item verified by outward-rounded interval enclosures on an adaptive partition):
  (I)   on the face b = eps a/4:  Phi_I := (eps/4) f_a - f_b > 0   (i.e. d/dt (b - eps a/4) < 0);
        b = 0 is invariant and f_a <= -a < 0 on Omega (a > b), so a trajectory from (4, b0), b0 <= eps,
        stays in Omega until a = a_*, which it reaches within log(4/a_*).
  (III) on every box of Omega whose B-enclosure contains 1/10:  Bdot = B_a f_a + B_b f_b > 0;
        so B = 1/10 is crossed at most once, transversally.
  (IV)  B(4, b) < 1/10 for 0 <= b <= eps (start) and B(a_*, b) > 1/10 for 0 <= b <= eps a_*/4 (exit);
        so the crossing occurs, at some a >= a_min (minimum a over crossing boxes), i.e. within
        log(4/a_min) of t_A.
  (V)   on Omega: p_pm >= 1/10 + s and contrasts >= 1/20 + s, s = 2e-3 (first-pair windows persist
        through the parent event with room for the third-pair exterior < 1e-3); uses monotonicity of
        w/(1+w+s) and (w-s)/(1+w+s) in w (increasing) and s (decreasing), w >= e^{-2(eps a/4)^2},
        v0 + v_pm <= e^{-2a^2} + e^{-2(a - eps a/4)^2}.
  (K)   K = sup_Omega |f_a|/a, giving a lower bound log(a_min/a_*)/K on the window after the crossing.
Run: OMP_NUM_THREADS=1 ... timeout 30 python s2_epsb_certificate.py
"""
from __future__ import annotations

import time

import mpmath
from mpmath import iv, mpf

iv.prec = 64
T0 = time.perf_counter()

# sanity: tight even power for intervals containing 0
_t = iv.mpf([-1, 2]) ** 2
assert _t.a == 0 and _t.b == 4, _t

A_STAR = iv.sqrt(iv.log(2))  # tight enclosure of sqrt(log 2)
FOUR = iv.mpf(4)
TENTH = iv.mpf(1) / 10
TWENTIETH = iv.mpf(1) / 20
SLACK = iv.mpf("0.002")


def LO(x):
    """Lower endpoint of an interval as an exact mpf."""
    return mpf(x.a)


def HI(x):
    """Upper endpoint of an interval as an exact mpf."""
    return mpf(x.b)


def core(a, b):
    w = iv.exp(-2 * b ** 2)
    v0 = iv.exp(-2 * a ** 2)
    ap, am = a + b, a - b
    vp = iv.exp(-2 * ap ** 2)
    vm = iv.exp(-2 * am ** 2)
    sp, sm = v0 + vp, v0 + vm
    Zp, Zm = 1 + w + sp, 1 + w + sm
    iZp, iZm = 1 / Zp, 1 / Zm
    cp = w * (iZp + iZm)
    c0 = v0 * (iZp + iZm)
    Gp, Gm = ap * vp * iZp, am * vm * iZm
    fa = -a - 2 * a * c0 - 2 * (Gp + Gm)
    fb = -b - 2 * b * cp - 2 * (Gp - Gm)
    B = (sp * iZp + sm * iZm) / 2
    dBa = ((1 + w) * (-4 * a * v0 - 4 * ap * vp) * iZp ** 2 + (1 + w) * (-4 * a * v0 - 4 * am * vm) * iZm ** 2) / 2
    dBb = ((-4 * ap * vp * (1 + w) + 4 * b * w * sp) * iZp ** 2 + (4 * am * vm * (1 + w) + 4 * b * w * sm) * iZm ** 2) / 2
    Bdot = dBa * fa + dBb * fb
    return dict(fa=fa, fb=fb, B=B, Bdot=Bdot, faa=(-fa) / a)


def phiI_over_a(a, eps):
    """Closed form of ((eps/4) f_a - f_b)/a on the face b = eps a/4 (function of a only)."""
    e = iv.mpf(eps)
    w = iv.exp(-(e ** 2) * a ** 2 / 8)
    v0 = iv.exp(-2 * a ** 2)
    vp = iv.exp(-2 * a ** 2 * (1 + e / 4) ** 2)
    vm = iv.exp(-2 * a ** 2 * (1 - e / 4) ** 2)
    Zp, Zm = 1 + w + v0 + vp, 1 + w + v0 + vm
    return (e / 2) * (w - v0) * (1 / Zp + 1 / Zm) + (2 - e / 2) * (1 + e / 4) * vp / Zp - (2 + e / 2) * (1 - e / 4) * vm / Zm


def bisect_1d(lo, hi, test, max_depth=40, n0=32, wmax=None):
    """Adaptive 1-D verification. test(interval) -> (ok, value). A box is accepted when ok and its width is
    at most wmax (forcing sharp reported margins). Returns (ok, nboxes, min accepted value or failure)."""
    lo, hi = mpf(lo), mpf(hi)
    edges = [lo] + [lo + (hi - lo) * k / n0 for k in range(1, n0)] + [hi]
    stack = [(edges[k], edges[k + 1], 0) for k in range(n0)]
    nbox, minval = 0, None
    while stack:
        x0, x1, d = stack.pop()
        nbox += 1
        ok, val = test(iv.mpf([x0, x1]))
        if ok and (wmax is None or x1 - x0 <= wmax):
            minval = val if minval is None or val < minval else minval
            continue
        if d >= max_depth:
            return False, nbox, (x0, x1, val)
        xm = (x0 + x1) / 2
        stack.append((x0, xm, d + 1))
        stack.append((xm, x1, d + 1))
    return True, nbox, minval


def certify(eps_str: str) -> dict:
    eps = mpf(eps_str)
    E = iv.mpf(eps_str)
    out: dict = {"eps": eps_str}
    a_lo = LO(A_STAR)  # lower end of the a_* enclosure; [a_*.a, 4] covers the a-range of Omega
    a_hi = mpf(4)

    # (I) face condition
    def tI(A):
        v = phiI_over_a(A, eps_str)
        return LO(v) > 0, LO(v)

    okI, nI, mI = bisect_1d(a_lo, a_hi, tI, wmax=mpf(1) / 512)
    chk = []
    for x in ("0.9", "2.0", "3.7"):
        A = iv.mpf(x)
        q = core(A, E * A / 4)
        gen = (E / 4 * q["fa"] - q["fb"]) / A
        cf = phiI_over_a(A, eps_str)
        chk.append(float(abs(gen.mid - cf.mid)))
    out["I"] = (okI, nI, mI, max(chk))

    # (IV) start and exit
    def tstart(Bv):
        q = core(FOUR, Bv)
        return HI(q["B"]) < LO(TENTH), HI(q["B"])

    okS, nS, mS = bisect_1d(mpf(0), eps, tstart, wmax=mpf(1) / 256)

    def texit(Bv):
        q = core(A_STAR, Bv)
        return LO(q["B"]) > HI(TENTH), LO(q["B"])

    okE, nE, mE = bisect_1d(mpf(0), HI(E * A_STAR / 4), texit, wmax=mpf(1) / 256)
    out["IV"] = (okS, nS, mS, okE, nE, mE)

    # (III) transversality on the crossing set, 2-D adaptive; band = 0 is the crossing set itself,
    # band = 0.01 certifies Bdot on {|B - 1/10| <= 0.01}, which gives the margins (m, s) = (0.01, min Bdot)
    # used by open:persistence for the parent crossing.
    out["III"] = crossing_pass(E, a_lo, a_hi, mpf(0))
    out["IIIband"] = crossing_pass(E, a_lo, a_hi, mpf("0.01"))
    return finish(out, E, a_lo, a_hi, eps)


def crossing_pass(E, a_lo, a_hi, band):
    lo_t = LO(TENTH - iv.mpf(band))
    hi_t = HI(TENTH + iv.mpf(band))
    stack = []
    NA = 64
    aed = [a_lo] + [a_lo + (a_hi - a_lo) * k / NA for k in range(1, NA)] + [a_hi]
    for k in range(NA):
        x0, x1 = aed[k], aed[k + 1]
        stack.append((x0, x1, mpf(0), HI(E * x1 / 4), 0))
    nbox, ncross, minBdot, amin, amax, fail = 0, 0, None, None, None, None
    while stack:
        x0, x1, y0, y1, d = stack.pop()
        if y0 > HI(E * x1 / 4):
            continue  # box entirely above the face: outside Omega
        nbox += 1
        q = core(iv.mpf([x0, x1]), iv.mpf([y0, y1]))
        if HI(q["B"]) < lo_t or LO(q["B"]) > hi_t:
            continue  # B outside the band on the whole box
        if LO(q["Bdot"]) > 0 and x1 - x0 <= mpf(1) / 1024 and y1 - y0 <= mpf(1) / 256:
            ncross += 1
            minBdot = LO(q["Bdot"]) if minBdot is None or LO(q["Bdot"]) < minBdot else minBdot
            amin = x0 if amin is None or x0 < amin else amin
            amax = x1 if amax is None or x1 > amax else amax
            continue
        if d >= 60:
            fail = (x0, x1, y0, y1, q["Bdot"])
            break
        if 4 * (x1 - x0) >= (y1 - y0):
            xm = (x0 + x1) / 2
            stack += [(x0, xm, y0, y1, d + 1), (xm, x1, y0, y1, d + 1)]
        else:
            ym = (y0 + y1) / 2
            stack += [(x0, x1, y0, ym, d + 1), (x0, x1, ym, y1, d + 1)]
    return (fail is None, nbox, ncross, minBdot, amin, amax, fail)


def finish(out, E, a_lo, a_hi, eps):
    # (V) first-pair margins, 1-D in a with monotone worst cases in b
    def tV(A):
        x0, x1 = iv.mpf(LO(A)), iv.mpf(HI(A))
        wmin = LO(iv.exp(-2 * (E * x1 / 4) ** 2))
        smax = HI(iv.exp(-2 * x0 ** 2) + iv.exp(-2 * (x0 - E * x1 / 4) ** 2))
        W, S = iv.mpf(wmin), iv.mpf(smax)
        p = W / (1 + W + S)
        c = (W - S) / (1 + W + S)
        okp = LO(p) > HI(TENTH + SLACK)
        okc = LO(c) > HI(TWENTIETH + SLACK)
        return (okp and okc), min(LO(p) - HI(TENTH + SLACK), LO(c) - HI(TWENTIETH + SLACK))

    okV, nV, mV = bisect_1d(a_lo, a_hi, tV, wmax=mpf(1) / 512)
    out["V"] = (okV, nV, mV)

    # (K) sup |f_a|/a over Omega on a fixed 200 x 20 cover (upper bound only)
    K = None
    aed = [a_lo] + [a_lo + (a_hi - a_lo) * k / 200 for k in range(1, 200)] + [a_hi]
    for k in range(200):
        x0, x1 = aed[k], aed[k + 1]
        ytop = HI(E * x1 / 4)
        bed = [mpf(0)] + [ytop * j / 20 for j in range(1, 20)] + [ytop]
        for j in range(20):
            q = core(iv.mpf([x0, x1]), iv.mpf([bed[j], bed[j + 1]]))
            K = HI(q["faa"]) if K is None or HI(q["faa"]) > K else K
    out["K"] = K
    return out


for eps_s in ("1", "0.25"):
    t1 = time.perf_counter()
    r = certify(eps_s)
    okI, nI, mI, chk = r["I"]
    okS, nS, mS, okE, nE, mE = r["IV"]
    ok3, nb3, nc3, mB, amin, amax, fail = r["III"]
    okV, nV, mV = r["V"]
    okb, nbb, ncb, mBb, aminb, amaxb, failb = r["IIIband"]
    allok = okI and okS and okE and ok3 and okV and okb
    print(f"eps = {eps_s}: {'PASS' if allok else 'FAIL'}   ({time.perf_counter() - t1:.1f} s)")
    print(f"  (I)   face (eps/4)f_a - f_b > 0: {okI}; {nI} boxes; min lower bound of Phi_I/a = {mpmath.nstr(mI, 6)},"
          f" so k - |f_a|/a >= {mpmath.nstr(4 * mI / mpf(eps_s), 6)}"
          f" (k - |f_a|/a = (4/eps) Phi_I/a on the face); closed form vs generic field |diff| = {chk:.1e}")
    print(f"  (IV)  start B(4,b) < 1/10 for b<=eps: {okS}; max upper bound {mpmath.nstr(mS, 6)}")
    print(f"        exit  B(a_*,b) > 1/10 for b<=eps a_*/4: {okE}; min lower bound {mpmath.nstr(mE, 6)}")
    print(f"  (III) Bdot > 0 on crossing boxes: {ok3}; {nb3} boxes evaluated, {nc3} crossing boxes;"
          f" min Bdot >= {mpmath.nstr(mB, 6)}; crossing a in [{mpmath.nstr(amin, 6)}, {mpmath.nstr(amax, 6)}]"
          + ("" if fail is None else f"; FAIL at {fail}"))
    print(f"  (III') Bdot > 0 on band |B-1/10| <= 0.01: {okb}; {ncb} band boxes; min Bdot >= {mpmath.nstr(mBb, 6)};"
          f" band a in [{mpmath.nstr(aminb, 6)}, {mpmath.nstr(amaxb, 6)}]")
    tc = mpmath.log(4 / amin)
    print(f"        crossing within log(4/a_min) = {mpmath.nstr(tc, 6)} of t_A; exit within log(4/a_*) = "
          f"{mpmath.nstr(HI(iv.log(FOUR / A_STAR)), 6)}")
    print(f"  (V)   first-pair p >= 0.102 and contrast >= 0.052 on Omega: {okV}; {nV} boxes; min margin {mpmath.nstr(mV, 6)}")
    print(f"  (K)   sup |f_a|/a <= {mpmath.nstr(r['K'], 6)}; window after crossing >= log(a_min/a_*)/K = "
          f"{mpmath.nstr(mpmath.log(amin / HI(A_STAR)) / r['K'], 6)}")
print(f"total time {time.perf_counter() - T0:.1f} s; mpmath {mpmath.__version__}, iv.prec = {iv.prec}")
