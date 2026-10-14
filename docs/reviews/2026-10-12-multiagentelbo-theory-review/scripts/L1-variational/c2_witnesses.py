r"""L1-variational check C2: correlated metric remark (c08), partial refresh (c08), access-restriction floor (c03),
ca_measure witness (c), random-link VFE (c05), appended energy (c04)."""
import numpy as np
import sympy as sp
from fractions import Fraction as Fr
from scipy.optimize import minimize_scalar

f, s, c, eps = sp.symbols('f s c epsilon', real=True)
Fc = ((f - s) ** 2 + s ** 2) / 2
G = sp.Matrix([[1, c], [c, 1]]); dF = sp.Matrix([sp.diff(Fc, f), sp.diff(Fc, s)])
v = G.inv() * dF
print("[3] critical set f =", sp.solve(sp.Eq(v[0], 0), f), "; d/df =", sp.simplify(sp.diff(v[0], f)))
dlt = (1 + eps) / 4
dFdt = (-dF[0] * v[0] / eps - dF[1] * v[1]).subs(c, sp.Rational(1, 2)).subs({f: 1 + 2 * dlt, s: 1 + dlt})
print("    dF/dt - [(1+eps)^2/(12 eps) - 4/3] =", sp.simplify(dFdt - ((1 + eps) ** 2 / (12 * eps) - sp.Rational(4, 3))),
      "; roots of eps^2-14eps+1:", sp.solve(eps ** 2 - 14 * eps + 1, eps))
Dl = sp.diag(sp.sqrt(eps), 1)
vc = (Dl * G * Dl).inv() * dF
print("    congruence fast component residual:", sp.simplify(eps * vc[0] - (dF[0] - c * sp.sqrt(eps) * dF[1]) / (1 - c ** 2)))

for om in [Fr(0), Fr(3, 10), Fr(1)]:
    Mk = [[Fr(1), Fr(0), Fr(0)], [Fr(0), Fr(1, 2), Fr(1, 2)], [Fr(0), Fr(1, 2), Fr(1, 2)]]
    Dw = [[1 - om / 2, om / 2, Fr(0)], [om / 2, 1 - om / 2, Fr(0)], [Fr(0), Fr(0), Fr(1)]]
    Tm = [[sum(Mk[i][l] * Dw[l][j] for l in range(3)) for j in range(3)] for i in range(3)]
    par = [0, 0, 1]
    res = {}
    for z0, start in [(0, [Fr(1, 2), Fr(1, 2), Fr(0)]), (1, [Fr(0), Fr(0), Fr(1)])]:
        num = Fr(0); den = Fr(0)
        for a in range(3):
            for b in range(3):
                for cc in range(3):
                    p = start[a] * Tm[a][b] * Tm[b][cc]
                    if par[b] == 0:
                        den += p
                        if par[cc] == 1:
                            num += p
        res[z0] = num / den
    print("[4] omega=%s: P(Z2=z2|z1,z1)=%s (claim %s); P(Z2=z2|z1,z2)=%s (claim %s); diff=%s"
          % (om, res[0], Fr(1, 6) + om / 12, res[1], Fr(1, 2) - om / 4, res[1] - res[0]))

P0 = np.array([37 / 89, 52 / 89]); P1 = np.array([13 / 161, 148 / 161])
logZ = lambda xi: np.log(np.sum(P0 ** xi * P1 ** (1 - xi)))
rx = minimize_scalar(logZ, bounds=(0, 1), method='bounded', options=dict(xatol=1e-12))
kl = lambda t, Pp: t * np.log(t / Pp[1]) + (1 - t) * np.log((1 - t) / Pp[0])
rt = minimize_scalar(lambda t: max(kl(t, P0), kl(t, P1)), bounds=(0.5, 0.99), method='bounded', options=dict(xatol=1e-12))
print("[5] Chernoff sup(-log Z) = %.8f at xi = %.5f; minimax over t = %.8f at t = %.5f" % (-rx.fun, rx.x, rt.fun, rt.x))
print("[6] sum 2^-n exp(-2^n) =", sum(2.0 ** -n * np.exp(-2.0 ** n) for n in range(1, 40)), "; e^-2 =", np.exp(-2))

rng = np.random.default_rng(11)
nT, nY = 4, 5
nu = rng.dirichlet(np.ones(nT)); Sact = rng.normal(size=nT); ZS = np.sum(nu * np.exp(-Sact))
prior = nu * np.exp(-Sact) / ZS
PY = rng.dirichlet(np.ones(nY), size=nT); Lik = rng.uniform(0.1, 1, size=(nT, nY))
pot = (PY * Lik).sum(1); po = np.sum(prior * pot)
R = rng.dirichlet(np.ones(nT)); QT = rng.dirichlet(np.ones(nY), size=nT)
joint_post = prior[:, None] * PY * Lik / po
Qj = R[:, None] * QT
Ftot = -np.log(po) + np.sum(Qj * np.log(Qj / joint_post))
Fmat = np.array([-np.log(pot[t]) + np.sum(QT[t] * np.log(QT[t] / (PY[t] * Lik[t] / pot[t]))) for t in range(nT)])
rhs = R @ Fmat + R @ Sact + np.sum(R * np.log(R / nu)) + np.log(ZS)
print("[7] random-link VFE residual:", Ftot - rhs)

Uy = rng.normal(size=nY) + 1.0
p0 = rng.dirichlet(np.ones(nY)); Lo = rng.uniform(0.1, 1, nY); q = rng.dirichlet(np.ones(nY))
Fv = np.sum(q * (np.log(q) - np.log(p0 * Lo)))
ZU = np.sum(p0 * np.exp(-Uy))
Ftil = np.sum(q * (np.log(q) - np.log(p0 * Lo * np.exp(-Uy) / ZU)))
print("[8] appended-energy residual:", Fv + q @ Uy - (Ftil - np.log(ZU)))
