"""D24-LDYN script 4: stability of the per-agent mean flow (it2:per-agent-flow (ii),
open problem it2:open-convergence). Two-agent K=2 counterexample with rank-one codes and
correlated self precisions, checked in the full tangent model (dictionaries, anchor, loop
prior on the two-cycle, grounding), plus a symbolic loop-gain formula."""
import numpy as np
import sympy as sp
from tangent_lib import Model, spd

qs = sp.symbols('q', positive=True)
A0 = sp.Matrix([[10, 3], [3, 1]]) + qs * sp.Matrix([[1, 0], [0, 0]])
A1 = sp.Matrix([[1, 3], [3, 10]]) + qs * sp.Matrix([[0, 0], [0, 1]])
gain = sp.simplify(qs**2 * A0.inv()[1, 0] * A1.inv()[0, 1])
print("symbolic loop gain eig(X0 X1) =", sp.factor(gain), "; per-agent flow eigenvalue -1 + sqrt(gain) =", sp.simplify(-1 + sp.sqrt(gain)))
print("unstable iff q >", sp.solve(sp.Eq(gain, 1), qs))

def build(q, cval, sigma2, lam, grounded):
    rng = np.random.default_rng(3)
    P = [np.array([[10.0, 3.0], [3.0, 1.0]]), np.array([[1.0, 3.0], [3.0, 10.0]])]
    Tl = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    Rl = [np.array([[1.0 / q]]), np.array([[1.0 / q]])]
    cv = [np.array([cval, 0.5 * cval]), np.array([-0.3 * cval, cval])]
    return Model(2, 2, [(1, 0), (0, 1)], rng, sigma2=sigma2, lam=lam, P=P, Tlist=Tl, Rlist=Rl, cvec=cv, grounded=grounded)

for (q, cval, sigma2, lam, grounded) in [(1.0, 0.0, 1.0, 0.0, False), (1.0, 0.0, 1.0, 2.0, True),
                                         (1.0, 1.0, 0.05, 2.0, True), (3.0, 1.0, 0.2, 1.0, True), (0.5, 0.0, 1.0, 1.0, False)]:
    M = build(q, cval, sigma2, lam, grounded)
    J, eta = M.global_J_eta()
    D = np.zeros_like(J)
    for bl in M.blocks:
        D[np.ix_(bl, bl)] = J[np.ix_(bl, bl)]
    evg = np.linalg.eigvals(-np.linalg.solve(D, J))
    H, b, A = M.per_agent_H_b('plug')
    evp = np.linalg.eigvals(-np.linalg.solve(A, H))
    print(f"q={q} c={cval} sigma2={sigma2} lam={lam} grounded={grounded}: loops {M.loops}; global max Re {evg.real.max():+.4f}; "
          f"plug-in per-agent max Re {evp.real.max():+.4f}; min |sv(H)| {np.linalg.svd(H, compute_uv=False).min():.2e}")
    if q == 1.0 and cval == 1.0:
        # simulate the plug-in per-agent mean flow (RK4) from a small perturbation of its fixed point
        mu_star = np.linalg.solve(H, b)
        F = lambda mu: -np.linalg.solve(A, H @ mu - b)
        mu = mu_star + 1e-6 * np.random.default_rng(0).normal(size=M.n)
        dt = 0.01
        for step in range(3000):
            k1 = F(mu); k2 = F(mu + dt / 2 * k1); k3 = F(mu + dt / 2 * k2); k4 = F(mu + dt * k3)
            mu = mu + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        print("   RK4 plug-in per-agent flow, |mu(30) - mu*| / |mu(0) - mu*| =", np.abs(mu - mu_star).max() / 1e-6)
        xs = np.linalg.solve(J, eta)
        Fg = lambda x: -np.linalg.solve(D, J @ x - eta)
        x = xs + 1e-6 * np.random.default_rng(0).normal(size=M.n)
        for step in range(3000):
            k1 = Fg(x); k2 = Fg(x + dt / 2 * k1); k3 = Fg(x + dt / 2 * k2); k4 = Fg(x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        print("   RK4 global flow, |x(30) - x*| / |x(0) - x*| =", np.abs(x - xs).max() / 1e-6)
    Sg, Vg, it = M.law_level_fixed_point(0.0)
    Hl, bl_, Al = M.per_agent_H_b('law', Sg, Vg)
    evl = np.linalg.eigvals(-np.linalg.solve(Al, Hl))
    print(f"   law-level per-agent (covariance fixed point after {it} iterations): max Re {evl.real.max():+.4f}")
