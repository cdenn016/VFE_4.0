r"""GAP10-S3 s01: second-order causal phase jet (lp:causal-phase-jets), the slice-jet
prolongation with W_loc, and the law jets (lp:causal-law-jets).

Part A (sympy, exact): for generic smooth omega(x, c) and x(c),
  d_a d_b [omega(x(c), c)] = D omega x_ab + D^2 omega[x_a, x_b] + D(omega_a) x_b + D(omega_b) x_a + omega_ab,
and for B~ = R(theta(c)) B(c) the second derivative formula of lp:causal-law-jets.

Part B (numerics on the chapter 14 field with a context-dependent marked frame):
  theta(c,t) := arg S(x(c,t)) - arg S(x(c,0))   (lp:frozen-local-initial-frame / lp:initial-phase-datum)
is differentiated in c by complex step + central differences of fixed-step RK4 solutions.
Independently, the jet ODE right-hand side of lp:causal-phase-jets is evaluated term by term
(directional derivatives of omega_loc at the slice state) and integrated in time (Simpson).
Also checked: second-order slice-jet prolongation with W_loc = V - omega_loc J, and
lp:causal-law-jets against differences of x~ = R_{U(S(x(c,0)))} x.
"""
from __future__ import annotations

import time
import numpy as np
import sympy as sp

from gap10_model import (Model, initial_section, rk4, omega_loc, Umat, rotB, Ssum, argS,
                         unpack, rot)

t_start = time.time()
print("=== Part A: symbolic chain-rule identities ===")
c1, c2 = sp.symbols('c1 c2')
cs = (c1, c2)
y = sp.symbols('y0:3')
# generic cubic omega(y, c) with symbolic coefficients and generic quadratic x_k(c):
# the second-order chain rule at a point is a polynomial identity in these jets.
allv = list(y) + [c1, c2]
mons = sorted(sp.itermonomials(allv, 3), key=sp.default_sort_key)
wco = sp.symbols(f'w0:{len(mons)}')
w = sum(k * mnm for k, mnm in zip(wco, mons))
qmons = sorted(sp.itermonomials([c1, c2], 2), key=sp.default_sort_key)
X = []
for k in range(3):
    co = sp.symbols(f'p{k}_0:{len(qmons)}')
    X.append(sum(a_ * mm for a_, mm in zip(co, qmons)))
sub = {y[k]: X[k] for k in range(3)}
expr = w.subs(sub)
at0 = {c1: 0, c2: 0}
ok = True
for a in range(2):
    for b in range(2):
        lhs = sp.diff(expr, cs[a], cs[b]).subs(at0)
        x0 = {y[k]: X[k].subs(at0) for k in range(3)}
        def P(e):
            return e.subs(at0).subs(x0) if False else e.subs(x0).subs(at0)
        T1 = sum(P(sp.diff(w, y[k])) * sp.diff(X[k], cs[a], cs[b]).subs(at0) for k in range(3))
        T2 = sum(P(sp.diff(w, y[k], y[m])) * sp.diff(X[k], cs[a]).subs(at0) * sp.diff(X[m], cs[b]).subs(at0)
                 for k in range(3) for m in range(3))
        T3 = sum(P(sp.diff(w, cs[a], y[k])) * sp.diff(X[k], cs[b]).subs(at0) for k in range(3))
        T4 = sum(P(sp.diff(w, cs[b], y[k])) * sp.diff(X[k], cs[a]).subs(at0) for k in range(3))
        T5 = P(sp.diff(w, cs[a], cs[b]))
        res = sp.expand(lhs - (T1 + T2 + T3 + T4 + T5))
        print(f"  phase-jet chain rule (a,b)=({a+1},{b+1}) [generic cubic omega, quadratic x]: residual = {res}")
        ok = ok and (res == 0)
# law jets
th = sp.Function('th')(c1, c2)
Bb = sp.Matrix([sp.Function('B1')(c1, c2), sp.Function('B2')(c1, c2)])
R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
J2 = sp.Matrix([[0, -1], [1, 0]])
Bt = R * Bb
for a in range(2):
    for b in range(2):
        lhs = sp.diff(Bt, cs[a], cs[b])
        rhs = R * (sp.diff(Bb, cs[a], cs[b]) + sp.diff(th, cs[a]) * J2 * sp.diff(Bb, cs[b])
                   + sp.diff(th, cs[b]) * J2 * sp.diff(Bb, cs[a]) + sp.diff(th, cs[a], cs[b]) * J2 * Bb
                   - sp.diff(th, cs[a]) * sp.diff(th, cs[b]) * Bb)
        res = sp.simplify(lhs - rhs)
        print(f"  law-jet formula (a,b)=({a+1},{b+1}): residual = {list(res)}")
        ok = ok and all(r == 0 for r in res)
    lhs1 = sp.diff(Bt, cs[a])
    rhs1 = R * (sp.diff(Bb, cs[a]) + sp.diff(th, cs[a]) * J2 * Bb)
    print(f"  first law jet (a={a+1}): residual = {list(sp.simplify(lhs1 - rhs1))}")
print("  Part A all exact:", ok)

print("\n=== Part B: numerics on the chapter 14 field ===")
M = Model()
ce = np.array([0.15, -0.1])
T, N = 1.0, 400
hc, hr = 1e-20, 1e-4


def traj(c):
    c = np.asarray(c)
    x0 = initial_section(c).astype(complex)
    return rk4(lambda t, x: M.field(x, c), x0, 0.0, T, N)


def derived(xs):
    r"""From a trajectory: slice state xbar, phase theta, frozen-frame state xtilde."""
    U0 = Umat(Ssum(xs[0]))
    a0 = argS(Ssum(xs[0]))
    xb = np.array([rotB(x, Umat(Ssum(x))) for x in xs])
    ph = np.array([argS(Ssum(x)) - a0 for x in xs])
    xt = np.array([rotB(x, U0) for x in xs])
    return xb, ph, xt


E = np.eye(2)
cache = {}


def get(key, c):
    if key not in cache:
        cache[key] = derived(traj(c))
    return cache[key]


base = get('0', ce + 0j)
d1 = {a: get(f'i{a}', ce + 1j * hc * E[a]) for a in range(2)}
d2 = {}
for a in range(2):
    for b in range(2):
        p = get(f'i{a}+{b}', ce + 1j * hc * E[a] + hr * E[b])
        m = get(f'i{a}-{b}', ce + 1j * hc * E[a] - hr * E[b])
        d2[(a, b)] = (p, m)
print(f"  trajectories computed: {len(cache)}  ({time.time()-t_start:.1f} s)")


def first(q_idx, a):
    return np.imag(d1[a][q_idx]) / hc


def second(q_idx, a, b):
    p, m = d2[(a, b)]
    return np.imag(p[q_idx] - m[q_idx]) / (2 * hr * hc)


xb = np.real(base[0])
ph = np.real(base[1])
xbar_a = {a: first(0, a) for a in range(2)}
xbar_ab = {(a, b): second(0, a, b) for a in range(2) for b in range(2)}
th_a = {a: first(1, a) for a in range(2)}
th_ab = {(a, b): second(1, a, b) for a in range(2) for b in range(2)}
xt_a = {a: first(2, a) for a in range(2)}
xt_ab = {(a, b): second(2, a, b) for a in range(2) for b in range(2)}

print("  symmetry of FD second jets: |th_12 - th_21|_max =",
      f"{np.max(np.abs(th_ab[(0,1)] - th_ab[(1,0)])):.2e}",
      " scale", f"{np.max(np.abs(th_ab[(0,1)])):.3e}")

ts = np.linspace(0, T, N + 1)


def simpson_cum_end(f):
    h = T / N
    return h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())


def omega_at(x, c):
    return omega_loc(M, x, c)


# zeroth order: theta(T) = int omega_loc(xbar, c) dt
om0 = np.array([np.real(omega_at(xb[k], ce)) for k in range(N + 1)])
print(f"  zeroth order: theta(T) = {ph[-1]:.12f}, int omega_loc dt = {simpson_cum_end(om0):.12f}, "
      f"diff {abs(ph[-1]-simpson_cum_end(om0)):.2e}")

# first order
for a in range(2):
    rhs = np.empty(N + 1)
    for k in range(N + 1):
        t1 = np.imag(omega_at(xb[k] + 1j * hc * xbar_a[a][k], ce)) / hc
        t2 = np.imag(omega_at(xb[k] + 0j, ce + 1j * hc * E[a])) / hc
        rhs[k] = t1 + t2
    print(f"  first-order jet a={a+1}: FD theta_a(T) = {th_a[a][-1]: .12f}, jet ODE = {simpson_cum_end(rhs): .12f},"
          f" diff {abs(th_a[a][-1]-simpson_cum_end(rhs)):.2e}")

# second order: five terms
maxrel = 0.0
for (a, b) in [(0, 0), (0, 1), (1, 1)]:
    T1 = np.empty(N + 1); T2 = np.empty(N + 1); T3 = np.empty(N + 1); T4 = np.empty(N + 1); T5 = np.empty(N + 1)
    for k in range(N + 1):
        x0k = xb[k]
        xa, xbb, xab = xbar_a[a][k], xbar_a[b][k], xbar_ab[(a, b)][k]
        T1[k] = np.imag(omega_at(x0k + 1j * hc * xab, ce)) / hc
        T2[k] = np.imag(omega_at(x0k + 1j * hc * xa + hr * xbb, ce) - omega_at(x0k + 1j * hc * xa - hr * xbb, ce)) / (2 * hr * hc)
        T3[k] = np.imag(omega_at(x0k + hr * xbb + 0j, ce + 1j * hc * E[a]) - omega_at(x0k - hr * xbb + 0j, ce + 1j * hc * E[a])) / (2 * hr * hc)
        T4[k] = np.imag(omega_at(x0k + hr * xa + 0j, ce + 1j * hc * E[b]) - omega_at(x0k - hr * xa + 0j, ce + 1j * hc * E[b])) / (2 * hr * hc)
        T5[k] = np.imag(omega_at(x0k + 0j, ce + 1j * hc * E[a] + hr * E[b]) - omega_at(x0k + 0j, ce + 1j * hc * E[a] - hr * E[b])) / (2 * hr * hc)
    rhs = T1 + T2 + T3 + T4 + T5
    jet = simpson_cum_end(rhs)
    fd = th_ab[(a, b)][-1]
    rel = abs(fd - jet) / max(abs(fd), 1e-300)
    maxrel = max(maxrel, rel)
    print(f"  second-order jet (a,b)=({a+1},{b+1}): FD theta_ab(T) = {fd: .12f}, jet ODE = {jet: .12f}, "
          f"abs diff {abs(fd-jet):.2e}, rel {rel:.2e}")
    print(f"      term integrals: Dw.x_ab {simpson_cum_end(T1): .6e}, D2w[x_a,x_b] {simpson_cum_end(T2): .6e}, "
          f"D(w_a)x_b {simpson_cum_end(T3): .6e}, D(w_b)x_a {simpson_cum_end(T4): .6e}, w_ab {simpson_cum_end(T5): .6e}")
    # dropping the explicit terms shows they matter
    print(f"      without explicit-partial terms (T3+T4+T5) the jet would be {simpson_cum_end(T1+T2): .6e}")

# slice-jet prolongation with W_loc (second order), integrated
def Wloc(x, c):
    om = omega_at(x, c)
    l, A, Cc, B, n = unpack(M.field(x, c))
    _, _, _, Bx, _ = unpack(x)
    J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
    B = B - om * (Bx @ J2.T)
    from gap10_model import pack
    return pack(l, A, Cc, B, n)


worst = 0.0
for (a, b) in [(0, 1)]:
    R = np.empty((N + 1, 24))
    for k in range(N + 1):
        x0k = xb[k]
        xa, xbb, xab = xbar_a[a][k], xbar_a[b][k], xbar_ab[(a, b)][k]
        t1 = np.imag(Wloc(x0k + 1j * hc * xab, ce)) / hc
        t2 = np.imag(Wloc(x0k + 1j * hc * xa + hr * xbb, ce) - Wloc(x0k + 1j * hc * xa - hr * xbb, ce)) / (2 * hr * hc)
        t3 = np.imag(Wloc(x0k + hr * xbb + 0j, ce + 1j * hc * E[a]) - Wloc(x0k - hr * xbb + 0j, ce + 1j * hc * E[a])) / (2 * hr * hc)
        t4 = np.imag(Wloc(x0k + hr * xa + 0j, ce + 1j * hc * E[b]) - Wloc(x0k - hr * xa + 0j, ce + 1j * hc * E[b])) / (2 * hr * hc)
        t5 = np.imag(Wloc(x0k + 0j, ce + 1j * hc * E[a] + hr * E[b]) - Wloc(x0k + 0j, ce + 1j * hc * E[a] - hr * E[b])) / (2 * hr * hc)
        R[k] = t1 + t2 + t3 + t4 + t5
    integ = np.array([simpson_cum_end(R[:, q]) for q in range(24)])
    dx = xbar_ab[(a, b)][-1] - xbar_ab[(a, b)][0]
    err = np.max(np.abs(integ - dx))
    worst = max(worst, err)
    print(f"  slice second jet prolongation (a,b)=({a+1},{b+1}): max |Delta x_ab - int RHS| = {err:.2e}, "
          f"scale {np.max(np.abs(dx)):.3e}")

# law jets: compare xtilde second jets (B blocks) with lp:causal-law-jets
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
k = N
Rth = rot(ph[k])
lawmax = 0.0
for (a, b) in [(0, 0), (0, 1), (1, 1)]:
    Bb = unpack(xb[k])[3]
    Bba, Bbb = unpack(xbar_a[a][k])[3], unpack(xbar_a[b][k])[3]
    Bbab = unpack(xbar_ab[(a, b)][k])[3]
    pred = (Rth @ (Bbab + th_a[a][k] * Bbb @ J2.T + th_a[b][k] * Bba @ J2.T + th_ab[(a, b)][k] * Bb @ J2.T
                   - th_a[a][k] * th_a[b][k] * Bb).T).T
    act = unpack(xt_ab[(a, b)][k])[3]
    e = np.max(np.abs(pred - act))
    lawmax = max(lawmax, e)
    print(f"  law jet (a,b)=({a+1},{b+1}) at T: max |B~_ab(FD) - formula| = {e:.2e}, scale {np.max(np.abs(act)):.3e}")
# identity x~ = R_theta xbar
idres = max(np.max(np.abs(rotB(xb[kk], rot(ph[kk])) - np.real(base[2][kk]))) for kk in range(0, N + 1, 50))
print(f"  identity x~ = R_theta xbar along trajectory: max residual {idres:.2e}")
Sx = [np.real(Ssum(base[0][kk]))[0] for kk in range(N + 1)]
print(f"  min_t S_x(x(c_e,t)) = {min(Sx):.3f} (analytic arg branch valid), |S(T)| = {abs(np.linalg.norm(np.real(Ssum(base[0][-1])))):.3f}")
print(f"\nmax relative discrepancy of second-order phase jet: {maxrel:.2e}")
print(f"runtime {time.time()-t_start:.1f} s")
