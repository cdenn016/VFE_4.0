r"""D18-LP check 7 (illustration, not proof): grouping window and prescribed null of chapter 14.

eps = 0.01 (so eps <= 1/100 as the null box needs), J = eps^3, records H = eps o, o=(+,+,-),
Lam_i = eps e_{omega_i}; initial u = l/eps = (4, 1/2, -1); small marked amplitudes; B_i = delta b_i
(noncollinear, common cone), delta = 0.05.  Selector (pa:selector) on public odds L_i.
Null: deletes nonself costs, keeps labels/priors/entropy:  beta0 = a e^{-D_ii}/(a e^{-D_ii} + 1 - a).
Also evaluates the selection-limit matrix and u_* of lp:selection-limit and the null-box numbers.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0e, i1e
import lp_model as M

eps, delta = 0.01, 0.05
J = eps ** 3
o = np.array([1.0, 1.0, -1.0])
om = np.array([0.4, 2.1, -1.3])
H = eps * o
Lam = eps * np.stack([np.cos(om), np.sin(om)], axis=1)
x0 = np.zeros((3, 8))
x0[:, 0] = eps * np.array([4.0, 0.5, -1.0])
x0[0, 1:3] = 0.05 * np.array([np.cos(0.3), np.sin(0.3)]); x0[1, 1:3] = [0.04, 0.0]; x0[2, 1:3] = 0.03 * np.array([np.cos(2.0), np.sin(2.0)])
bdir = np.array([0.2, 0.7, 1.1])
x0[:, 5:7] = delta * np.stack([np.cos(bdir), np.sin(bdir)], axis=1)
x0[:, 7] = [0.02, -0.01, 0.015]


def selector(L):
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            if np.sign(L[i]) == np.sign(L[j]) and L[i] != 0 and min(abs(L[i]), abs(L[j])) / max(abs(L[i]), abs(L[j])) > 0.25:
                edges.append((i, j))
    comp = list(range(3))
    for i, j in edges:
        ci, cj = comp[i], comp[j]
        comp = [ci if cc == cj else cc for cc in comp]
    blocks = {}
    for i, cc in enumerate(comp):
        blocks.setdefault(cc, []).append(i)
    return sorted(blocks.values())


def self_cost(x):
    x = x.reshape(3, 8)
    D = np.zeros(3)
    for i in range(3):
        l, A, C, B = x[i, 0], x[i, 1:3], x[i, 3:5], x[i, 5:7]
        lp, lm = l / 2 + M.logI0(np.hypot(*(A + C))), -l / 2 + M.logI0(np.hypot(*(A - C)))
        pp = 1 / (1 + np.exp(lm - lp))
        Ee = pp * M.Mvec(A + C) + (1 - pp) * M.Mvec(A - C)
        Eze = pp * M.Mvec(A + C) - (1 - pp) * M.Mvec(A - C)
        m = np.tanh(M.public_odds(x)[i] / 2)
        logZ = M.logI0(np.hypot(*B)) + np.logaddexp(lp, lm) - np.log(2)
        Sent = l / 2 * m + A @ Ee + C @ Eze + B @ M.Mvec(B) - logZ
        D[i] = Sent - J * Eze[0] + M.logI0(J)
    return D


def Vnull(x):
    x = x.reshape(3, 8)
    D = self_cost(x)
    b0 = 0.2 * np.exp(-D) / (0.2 * np.exp(-D) + 0.8)
    n = x[:, 7]
    pp = 1 / (1 + np.exp(-n)); Ds = pp * np.log(2 * pp) + (1 - pp) * np.log(2 * (1 - pp))
    g0 = 0.2 * np.exp(-Ds) / (0.2 * np.exp(-Ds) + 0.8)
    dx = np.empty_like(x)
    dx[:, 0] = -b0 * x[:, 0] + H
    dx[:, 1:3] = -b0[:, None] * x[:, 1:3] + Lam
    dx[:, 3:5] = b0[:, None] * (np.array([J, 0.0]) - x[:, 3:5])
    dx[:, 5:7] = -b0[:, None] * x[:, 5:7]
    dx[:, 7] = -g0 * n
    return dx, b0


Tend = 60.0
ts = np.array([0, 2, 5, 10, 20, 40, 60.0])
full = solve_ivp(lambda t, y: M.V(y, H, Lam, J).ravel(), (0, Tend), x0.ravel(), t_eval=ts, method="DOP853", rtol=1e-11, atol=1e-14)
null = solve_ivp(lambda t, y: Vnull(y)[0].ravel(), (0, Tend), x0.ravel(), t_eval=ts, method="DOP853", rtol=1e-11, atol=1e-14)
for k, t in enumerate(ts):
    xf, xn = full.y[:, k].reshape(3, 8), null.y[:, k].reshape(3, 8)
    Lf, Ln = M.public_odds(xf), M.public_odds(xn)
    print(f"t={t:5.1f}  full L/eps={np.round(Lf/eps,3)} blocks={selector(Lf)} | null L/eps={np.round(Ln/eps,3)} "
          f"blocks={selector(Ln)} beta0={np.round(Vnull(xn)[1],4)}  |B|full={np.round(np.hypot(*xf[:,5:7].T),5)}")
Mm = -1.4 * np.eye(3) + 0.4 * np.ones((3, 3))
print("lp:selection-limit: eigenvalues", np.round(np.linalg.eigvalsh(Mm), 6), " u_* =", -np.linalg.solve(Mm, o), " (15/7,15/7,5/7) =", [15/7, 15/7, 5/7])
print("null box bound 72e^2+1/25+18e^6 =", 72 * eps ** 2 + 1 / 25 + 18 * eps ** 6, "< log 2 =", np.log(2),
      "; beta0 range [", np.exp(-np.log(2)) / (np.exp(-np.log(2)) + 4), ",", 1 / 5, "]")
print("|L-l| bound 2/5 e^3 + 8 e^6 =", 0.4 * eps ** 3 + 8 * eps ** 6, "< e/8 =", eps / 8)
