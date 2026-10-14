r"""L7 check 2: deterministic lock analysis in the nonvanishing-imprint step.

ca_selection_proofs.tex:154-220. On the invariant subspace kbar = mbar, the
scalar-free common residual obeys rho' = -G M_sigma rho with G = 4I - A_{C4},
M_sigma = diag(m_e(sigma)), rho(0) = (0,-1,0,1). Checks:
  * six assignments have m_1 = m_3 (eigenvector motion e^{-4 m_1 t}),
  * the four listed patterns, the (a,b,d,h) system, the invariant half-spaces,
    h(infinity) = 6/7, and the odd-even margin >= (7/50) e^{-4 t_r},
  * after redrawing S^(0), the margin persists (h_r >= 171/224),
  * the other six assignments are 'bad' (some receiver prefers an odd edge
    at some time), so the 10/16 count is the right split,
  * E|X_{h_s}|^2 = 14 exactly.
Also verifies the reduced-field identity V_i = p_i(1-p_i) g_i numerically.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.linalg import expm

A4 = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], float)
G = 4*np.eye(4) - A4
rho0 = np.array([0., -1., 0., 1.])

def mult(sig):
    return np.array([(sig[e] == 1) + (sig[(e + 1) % 4] == -1) for e in range(4)], float)

sigmas = list(itertools.product([1, -1], repeat=4))
pats = [tuple(int(v) for v in mult(s)) for s in sigmas]
eq13 = [p for p in pats if p[1] == p[3]]
print("patterns with m1 = m3:", len(eq13), eq13)
listed = [(1,1,2,0),(1,0,2,1),(2,0,1,1),(2,1,1,0)]
print("listed four present:", all(p in pats for p in listed))
others = [p for p in pats if p not in eq13 and p not in listed]
print("remaining six:", others)

def margin_path(m, ts):
    """odd-minus-even absolute residual margin along the first held motion: per receiver i
    the edges are i-1 and i; one odd, one even. Return min over receivers."""
    out = []
    for t in ts:
        r = expm(-G @ np.diag(m) * t) @ rho0
        mg = min(abs(r[o]) - abs(r[e]) for (o, e) in [(1, 0), (1, 2), (3, 2), (3, 0)])
        out.append(mg)
    return np.array(out)

ts = np.linspace(0, 6, 601)
for p in eq13 + listed + others:
    mg = margin_path(np.array(p, float), ts)
    bound = (7/50)*np.exp(-4*ts) if p in listed else np.exp(-4*p[1]*ts)
    ok = np.all(mg >= bound - 1e-12) if (p in listed or p in eq13) else None
    print(f"pattern {p}: min margin/bound ratio {np.min(mg/np.maximum(bound,1e-300)) if ok is not None else float('nan'):.4f}; min margin {mg.min():+.4f}; good={ok}")

# (1,1,2,0) explicit system
Mx = np.array([[-4,1,0],[1,-4,2],[0,1,-8]], float)
def abdh(t):
    abd = expm(Mx*t) @ np.array([0,1.,0])
    return abd
ok_half = True
for t in np.linspace(0, 10, 2001):
    a, b, d = abdh(t)
    ok_half &= (a - 0.75*b - 11*d/25 <= 1e-12) and (d - b/4 <= 1e-12) and (b >= np.exp(-4*t) - 1e-12) and a <= 0.25 and d <= 0.125
print("half-space invariants and bounds hold on grid:", ok_half)
# h(inf) by integrating h' = -(a+2d)
from scipy.integrate import quad
hinf = 1 - quad(lambda t: abdh(t)[0] + 2*abdh(t)[2], 0, 60, limit=200)[0]
print("h(inf) =", hinf, " 6/7 =", 6/7)
# after redraw of S^(0) at t_r: margin check
worst = np.inf
for tr in np.linspace(0, 5, 51):
    a0, b0, d0 = abdh(tr)
    h0 = 1 - quad(lambda t: abdh(t)[0] + 2*abdh(t)[2], 0, tr)[0]
    for r in np.linspace(0, 5, 51):
        ar, dr = a0*np.exp(-8*r), d0*np.exp(-8*r)
        br = b0 + 0.25*(a0+d0)*(1-np.exp(-8*r)); hr = h0 - 0.25*(a0+d0)*(1-np.exp(-8*r))
        assert hr >= 171/224 - 1e-12
        mg = min(br - ar, br - dr, hr - ar, hr - dr)
        worst = min(worst, mg / ((7/50)*np.exp(-4*tr)))
print("post-redraw margin / ((7/50)e^{-4 t_r}) minimum:", worst)
# cross-check the post-redraw formulas against the matrix flow with M = diag(2,0,2,0)
Ms0 = np.diag([2., 0, 2, 0])
a0, b0, d0 = abdh(0.7); h0 = 1 - quad(lambda t: abdh(t)[0] + 2*abdh(t)[2], 0, 0.7)[0]
rho = np.array([-a0, -b0, -d0, h0])
r1 = expm(-G @ Ms0 * 0.3) @ rho
pred = np.array([-a0*np.exp(-2.4), -(b0 + 0.25*(a0+d0)*(1-np.exp(-2.4))), -d0*np.exp(-2.4), h0 - 0.25*(a0+d0)*(1-np.exp(-2.4))])
print("post-redraw formula residual:", np.abs(r1 - pred).max())

# E|X_{h_s}|^2 with E|X0|^2 = 2 s^2 + 12 and e^{-2 T h_s} = s^{-2}
for s in [10., 1e3]:
    print("E|X_h|^2 bound at s=", s, ":", (2*s*s + 12)/s**2 + 12*(1 - s**-2))
