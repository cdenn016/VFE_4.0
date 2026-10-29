r"""D13-ERAPUB s01: constants and closed-form identities of chapters/08_smooth_erasure.tex.

Checks (independent of the 2026-10-22 scripts):
  (a) M_chi = max |chi'| for the flat step erase:flat-step;
  (b) B = 3 s^2 / [2 (1 - s^2/4)] for s = sigma_max = 3/4 and the grid maximum of
      |d_alpha d(z_a, alpha rho z_j)| / rho  (erase:gate-derivative);
  (c) m_K, M_K, V_K on K = [1/8, 1/2] and the admissible gate width Delta from erase:time-choice;
  (d) the well margins erase:well-margins;
  (e) sympy: erase:public-derivatives and their O(eps^2) order, erase:crossed-pair-gap,
      erase:cost-comparison constants, erase:row-bound constant Delta_eps / eps^2;
  (f) sympy: the cumulant mixed term erase:mixed-term and the witness erase:mixed-witness;
  (g) d x*/dJ at x = 1/2 for the heterogeneous-coupling remark (1/(8/3 - 2 log 3)).
Single-threaded, a few seconds.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

J = np.log(3.0)
HSTAR = -0.5
H0 = 0.25
L_BOX, U_BOX = 0.49, 0.51


def chi(r: float) -> float:
    if r <= 0:
        return 0.0
    if r >= 1:
        return 1.0
    e1, e2 = np.exp(-1 / r), np.exp(-1 / (1 - r))
    return e1 / (e1 + e2)


def dchi(r: float) -> float:
    if r <= 0 or r >= 1:
        return 0.0
    phi = 1 / r - 1 / (1 - r)
    s = 1 / (1 + np.exp(phi))
    return s * (1 - s) * (1 / r**2 + 1 / (1 - r) ** 2)


def part_a() -> None:
    rs = np.linspace(1e-4, 1 - 1e-4, 200001)
    vals = np.array([dchi(r) for r in rs])
    k = int(np.argmax(vals))
    print(f"(a) M_chi = max chi' = {vals[k]:.12f} at r = {rs[k]:.6f}; chi(1/2) = {chi(0.5):.6f}, chi(3/4) = {chi(0.75):.6f}")


def part_b() -> None:
    s = 0.75
    B = 3 * s**2 / (2 * (1 - s**2 / 4))
    worst = 0.0
    for rho in (1e-4, 1e-2, 0.1, 0.5):
        za = np.linspace(-s, s, 301)
        zj = np.linspace(-s, s, 301)
        A, Z = np.meshgrid(za, zj)
        for al in np.linspace(0, 1, 41):
            c = al * rho * Z
            val = np.abs(rho * Z * (c - A) / (1 - c**2)) / rho
            worst = max(worst, float(val.max()))
    print(f"(b) B = {B:.10f}; grid max of |d_alpha d|/rho over rho<=1/2 = {worst:.10f} (<= B: {worst <= B + 1e-12})")


def g_clock(h: float) -> float:
    return 2 * (1 - h**2) * (np.arctanh(h) - np.arctanh(HSTAR))


def part_c() -> None:
    K = (1 / 8, 1 / 2)
    hs = np.linspace(*K, 200001)
    gv = g_clock(hs)
    mK, MK = float(gv.min()), float(gv.max())
    VK = MK + mK / 2
    dmax = (H0 - 1 / 8) * 2 * mK / (3 * VK)
    print(f"(c) m_K = {mK:.6f} (at h = {hs[np.argmin(gv)]:.4f}), M_K = {MK:.6f} (at h = {hs[np.argmax(gv)]:.4f}), V_K = {VK:.6f}")
    print(f"(c) erase:time-choice admits Delta < {dmax:.6f}; initial interval in K needs Delta <= 1/3; closure bound 3 Delta/(2 m_K) at Delta=0.04: {3*0.04/(2*mK):.5f}")
    print(f"(c) rho bound from B rho M_chi / Delta <= m_K/2 with M_chi=2, B=0.981818: rho <= {mK/2/(0.981818*2):.5f} * Delta")


def part_d() -> None:
    m1 = J * L_BOX - np.arctanh(L_BOX)
    m2 = np.arctanh(U_BOX) - J * U_BOX
    print(f"(d) J l - atanh l = {m1:.6e}, atanh u - J u = {m2:.6e}; J/2 - atanh(1/2) = {J/2-np.arctanh(0.5):.2e}")


def part_e() -> None:
    x, v, c, e, s, al, rho = sp.symbols('x v c epsilon sigma alpha rho', real=True)
    d = (1 + x) / 2 * sp.log((1 + x) / (1 + c)) + (1 - x) / 2 * sp.log((1 - x) / (1 - c))
    cc = (1 - e) * x + e * v
    D = d.subs(c, cc)
    d1 = sp.diff(D, x)
    d2 = sp.diff(D, v)
    claim1 = sp.atanh(x) - sp.atanh(cc) + (1 - e) * (cc - x) / (1 - cc**2)
    claim2 = e * (cc - x) / (1 - cc**2)
    r1 = sp.simplify(sp.expand_log(d1 - claim1.rewrite(sp.log), force=True))
    r2 = sp.simplify(d2 - claim2)
    print(f"(e) erase:public-derivatives residuals: {r1}, {r2}")
    ser1 = sp.series(claim1, e, 0, 3).removeO()
    ser2 = sp.series(claim2, e, 0, 3).removeO()
    print(f"(e) series in eps: d1 D = {sp.simplify(ser1)} ; d2 D = {sp.simplify(ser2)}")
    # crossed-pair gap
    gap = d.subs({x: s, c: -al * rho * s}) - d.subs({x: s, c: al * rho * s})
    r3 = sp.simplify(sp.expand_log(gap - 2 * s * sp.atanh(al * rho * s).rewrite(sp.log), force=True))
    print(f"(e) erase:crossed-pair-gap residual: {r3}")
    # second derivative in first argument
    print(f"(e) d_1^2 d(x,c) = {sp.simplify(sp.diff(d, x, 2))}")
    lb = 2 * L_BOX**2 - (U_BOX - L_BOX) ** 2 / (2 * (1 - U_BOX**2))
    print(f"(e) erase:row-bound Delta_eps/eps^2 = {lb:.6f}; same-box bound (u-l)^2/(2(1-u^2)) = {(U_BOX-L_BOX)**2/(2*(1-U_BOX**2)):.3e}, cross bound (2l)^2/2 = {2*L_BOX**2:.4f}")


def part_f() -> None:
    T, eps, al, rho = sp.symbols('T epsilon alpha rho', positive=True)
    # generic row with three sources: X_j = eps^2 A_j - al rho zi zj
    A = sp.symbols('A1:4', real=True)
    Z = sp.symbols('z1:4', real=True)
    zi = sp.symbols('z_i', real=True)
    lam = sp.symbols('lam', real=True)
    X = [lam * (eps**2 * A[k] - al * rho * zi * Z[k]) for k in range(3)]
    row = -T * sp.log(sum(sp.exp(-Xk / T) for Xk in X) / 3)
    ser = sp.series(row, lam, 0, 3).removeO()
    second = sp.expand(ser.coeff(lam, 2))
    mixed = sp.expand(sp.diff(second, eps, 2).subs(eps, 0) / 2)  # coefficient of eps^2 ...
    mixed = sp.expand(sp.diff(mixed, rho).subs(rho, 0))       # ... times rho
    meanA = sum(A) / 3
    meanZ = sum(Z) / 3
    cov = sum(A[k] * Z[k] for k in range(3)) / 3 - meanA * meanZ
    claim = al * zi * cov / T
    print(f"(f) erase:mixed-term residual (coefficient of eps^2 rho): {sp.simplify(mixed - claim)}")
    # witness: x = r v, v = (-b,-a,a,b), z = sigma (1,-1,1,-1)
    a, b, r, s = sp.symbols('a b r sigma', positive=True)
    v = [-b, -a, a, b]
    zz = [s, -s, s, -s]
    xs = [r * vi for vi in v]
    total = 0
    for i in range(4):
        js = [j for j in range(4) if j != i]
        Aij = [(xs[j] - xs[i]) ** 2 / (2 * (1 - xs[i] ** 2)) for j in js]
        zj = [zz[j] for j in js]
        mA = sum(Aij) / 3
        mZ = sum(zj) / 3
        total += zz[i] * (sum(Aij[k] * zj[k] for k in range(3)) / 3 - mA * mZ)
    claimw = 2 * s**2 * r**2 / 9 * ((a**2 - 3 * b**2 + 6 * a * b) / (1 - r**2 * b**2) + (b**2 - 3 * a**2 + 6 * a * b) / (1 - r**2 * a**2))
    print(f"(f) erase:mixed-witness residual: {sp.simplify(total - claimw)}")
    f1 = sp.lambdify((a, b), a**2 - 3 * b**2 + 6 * a * b)
    f2 = sp.lambdify((a, b), b**2 - 3 * a**2 + 6 * a * b)
    print(f"(f) witness numerators at corner (a,b)=(l,u): {f1(L_BOX, U_BOX):.4f}, {f2(L_BOX, U_BOX):.4f}; at (u,u): {f1(U_BOX,U_BOX):.4f}, {f2(U_BOX,U_BOX):.4f}")


def part_g() -> None:
    xs = 0.5
    dxdJ = xs / (1 / (1 - xs**2) - J)
    print(f"(g) dx*/dJ at x=1/2: {dxdJ:.6f}; 1/(8/3 - 2 log 3) = {1/(8/3-2*np.log(3)):.6f}")
    print(f"(g) det of Jacobian (atanh x - J y, atanh y - J x) at (1/2,1/2): 16/9 - J^2 = {16/9 - J**2:.6f}")


if __name__ == '__main__':
    part_a(); part_b(); part_c(); part_d(); part_e(); part_f(); part_g()
