r"""D04-DYN s3: prop:dyn-morse (full and Gibbs-reduced potentials share critical points and inertia).

Three binary agents, trivial model sector.  Agent 0 (chooser) has a flat evaluator and attends to the two
anchors 1 and 2 through identity channels with a learnable row b = beta_01; the anchors attend only to
themselves and have evaluators at opposite states, so they also feel source-role forces from agent 0.
At small T the mirror-symmetric critical point is a saddle.  Exact symbolic Hessians (sympy), critical
points by nsolve, inertia and the Schur-complement identity checked; also the asymmetric minima.
"""
import sympy as sp
import numpy as np

x0, x1, x2, b = sp.symbols("x0 x1 x2 b", positive=True)
T = sp.Rational(1, 10)
g = [sp.Rational(1, 2), sp.Rational(1, 10), sp.Rational(9, 10)]   # evaluator P(state 1)
pi0 = sp.Rational(1, 2)


def KLb(p, r):
    return p * sp.log(p / r) + (1 - p) * sp.log((1 - p) / (1 - r))


E0 = KLb(x0, g[0]) + KLb(x1, g[1]) + KLb(x2, g[2])
D01 = KLb(x0, x1)
D02 = KLb(x0, x2)
Erow = b * D01 + (1 - b) * D02 + T * (b * sp.log(b / pi0) + (1 - b) * sp.log((1 - b) / (1 - pi0)))
E = E0 + Erow
Z = pi0 * sp.exp(-D01 / T) + (1 - pi0) * sp.exp(-D02 / T)
Ered = E0 - T * sp.log(Z)

U = [x0, x1, x2]
gradE = [sp.diff(E, v) for v in U + [b]]
gradR = [sp.diff(Ered, v) for v in U]
HE = sp.hessian(E, U + [b])
HR = sp.hessian(Ered, U)
fE = sp.lambdify(U + [b], gradE, "mpmath")
fHE = sp.lambdify(U + [b], HE, "numpy")
fHR = sp.lambdify(U, HR, "numpy")
fbstar = sp.lambdify(U, pi0 * sp.exp(-D01 / T) / Z, "numpy")
fgR = sp.lambdify(U, gradR, "numpy")


def inertia(H, tol=1e-9):
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    return int(np.sum(ev < -tol)), int(np.sum(np.abs(ev) <= tol)), int(np.sum(ev > tol)), ev


from scipy.optimize import minimize
fR = sp.lambdify(U, Ered, "numpy")
guesses = [("mirror-symmetric", (0.5, 0.2, 0.8, 0.5))]
for start in ((0.15, 0.15, 0.85), (0.85, 0.15, 0.85)):
    m = minimize(lambda z: fR(*z), start, jac=lambda z: np.array(fgR(*z), dtype=float), bounds=[(1e-6, 1 - 1e-6)] * 3, tol=1e-14)
    guesses.append(("local minimum of E_red from " + str(start), tuple(m.x) + (float(fbstar(*m.x)),)))
for name, guess in guesses:
    sol = sp.nsolve(gradE, U + [b], guess, tol=1e-30, prec=40)
    pt = [float(v) for v in sol]
    HEv = np.array(fHE(*pt), dtype=float)
    HRv = np.array(fHR(*pt[:3]), dtype=float)
    H_uu, H_ub, H_bb = HEv[:3, :3], HEv[:3, 3:], HEv[3:, 3:]
    schur = H_uu - H_ub @ np.linalg.inv(H_bb) @ H_ub.T
    nE = inertia(HEv)
    nR = inertia(HRv)
    print(f"== {name} critical point (x0, x1, x2, b) = {np.round(pt, 8)}")
    print(f"   |b - beta*(u)| = {abs(pt[3] - fbstar(*pt[:3])):.2e}; |grad E_red(u)| = {np.max(np.abs(fgR(*pt[:3]))):.2e}")
    print(f"   row Hessian H_bb = {H_bb[0,0]:.6f}; T times Fisher of the row = {float(T)/(pt[3]*(1-pt[3])):.6f}")
    print(f"   inertia (neg, zero, pos) of Hess E = {nE[:3]}, of Hess E_red = {nR[:3]}")
    print(f"   max abs(Hess E_red - Schur complement) = {np.max(np.abs(HRv - schur)):.2e}; eigenvalues Hess E_red = {np.round(nR[3], 6)}")
