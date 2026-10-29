r"""D15-WISH s2: the null flow of form:kappa-independence (iii) under two row conventions.

inh:null-flow deletes every peer cost D_ij (j not equal to i) and keeps all other data.
Chapter 13 (pa:null-row), companion c13 (cp:null-row) and Item 7 (it7:null-row)
read this as: labels and priors kept, so with exact Gibbs rows the self weight is
   beta0_ii = pi_ii exp(-D_ii/T) / (pi_ii exp(-D_ii/T) + 1 - pi_ii)        (R2).
Chapter 11 (iii) instead uses self weight one (R1):
   E0 = sum_i [ KL(q_i || W(alpha0, I)) + KL(q_i || W(alpha0, Lambda0_i)) ],
   Lambda0_i = I + (kappa/N)(alpha_i P_i^{-1} + M0_{-i}),  M0_{-i} frozen initial moment.
(a) mpmath natural-gradient check of both nulls on the Wishart family.
    Prediction for R2:  d alpha_i/dt = -(1+beta_i)(alpha_i - alpha0),
                        d P_i/dt = -(1+beta_i) P_i + I + beta_i C0_i,
                        C0_i = Lambda0_i + (kappa/N)(alpha_i P_i^{-1} - alpha0 Lambda0_i^{-1}).
(b) Integrate the R2 null from admitted preparations (common precision) and compare
    its partition sequence and event times with the full flow (closed form).
"""
import itertools
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammaln, digamma

mp.mp.dps = 30
N = 4


def mlg2(a):
    return mp.log(mp.sqrt(mp.pi)) + mp.loggamma(a) + mp.loggamma(a - mp.mpf(1) / 2)


def mpsi2(a):
    return mp.digamma(a) + mp.digamma(a - mp.mpf(1) / 2)


def mkl(a, P, b, Q):
    Pi = P ** -1
    return ((a - b) * mpsi2(a) - mlg2(a) + mlg2(b) + b * mp.log(mp.det(P) / mp.det(Q))
            + a * ((Q * Pi)[0, 0] + (Q * Pi)[1, 1] - 2))


def unpack(th):
    return th[0], mp.matrix([[th[1], th[2]], [th[2], th[3]]])


def lam0(i, th, Mrest, kap):
    al, P = unpack(th)
    return mp.eye(2) + (kap / N) * (al * P ** -1 + Mrest[i])


def energy_null(TH, Mrest, a0, kap, T, conv):
    E = 0
    for i, th in enumerate(TH):
        al, P = unpack(th)
        Dii = mkl(al, P, a0, lam0(i, th, Mrest, kap))
        E += mkl(al, P, a0, mp.eye(2))
        if conv == "R1":
            E += Dii
        else:
            pi = mp.mpf(1) / N
            E += -T * mp.log(pi * mp.exp(-Dii / T) + 1 - pi)
    return E


def fisher(th):
    def psi(*x):
        al, P = unpack(x)
        return mlg2(al) - al * mp.log(mp.det(P))
    G = mp.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            o = [0] * 4
            o[i] += 1
            o[j] += 1
            G[i, j] = mp.diff(psi, tuple(th), tuple(o))
    return G


def natgrad_agent(TH, i, Efun):
    g = mp.matrix(4, 1)
    for k in range(4):
        def f(x, k=k):
            T2 = [list(t) for t in TH]
            T2[i][k] = x
            return Efun(T2)
        g[k] = mp.diff(f, TH[i][k])
    return -(fisher(TH[i]) ** -1) * g


TH = [[mp.mpf("1.0"), mp.mpf("1.3"), mp.mpf("0.2"), mp.mpf("0.9")],
      [mp.mpf("1.2"), mp.mpf("0.8"), mp.mpf("-0.1"), mp.mpf("1.1")],
      [mp.mpf("1.8"), mp.mpf("1.0"), mp.mpf("0.3"), mp.mpf("1.4")],
      [mp.mpf("2.1"), mp.mpf("1.6"), mp.mpf("0.0"), mp.mpf("0.7")]]
a0, T = mp.mpf(1), mp.mpf(1)
print("(a) natural-gradient check of the two nulls (unequal precisions, T_q = 1)")
for kap in [mp.mpf(0), mp.mpf(1)]:
    Mrest = []
    for i in range(N):
        M = mp.zeros(2, 2)
        for j in range(N):
            if j != i:
                al, P = unpack(TH[j])
                M += al * P ** -1
        Mrest.append(M)
    for conv in ["R1", "R2"]:
        worst_a, worst_P, rates = 0, 0, []
        for i in range(N):
            v = natgrad_agent(TH, i, lambda X: energy_null(X, Mrest, a0, kap, T, conv))
            al, P = unpack(TH[i])
            L0 = lam0(i, TH[i], Mrest, kap)
            Dii = mkl(al, P, a0, L0)
            pi = mp.mpf(1) / N
            beta = 1 if conv == "R1" else pi * mp.exp(-Dii / T) / (pi * mp.exp(-Dii / T) + 1 - pi)
            C0 = L0 + (kap / N) * (al * P ** -1 - a0 * L0 ** -1)
            R = -(1 + beta) * P + mp.eye(2) + beta * C0
            worst_a = max(worst_a, abs(v[0] + (1 + beta) * (al - a0)))
            worst_P = max(worst_P, abs(v[1] - R[0, 0]), abs(v[2] - R[0, 1]), abs(v[3] - R[1, 1]))
            rates.append(float(-v[0] / (al - a0)) if al != a0 else float("nan"))
        print(f"    kappa={float(kap)} {conv}: shape residual {float(worst_a):.1e}, precision residual "
              f"{float(worst_P):.1e}; shape rates -(d alpha/dt)/(alpha-alpha0) = {np.round(rates, 5)}")

# ---------------- (b) partition sequences: full flow vs R2 null ----------------


def lg2(a):
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a):
    return digamma(a) + digamma(a - 0.5)


def kl(a, P, b, Q):
    return ((a - b) * psi2(a) - lg2(a) + lg2(b) + b * np.log(np.linalg.det(P) / np.linalg.det(Q))
            + a * (np.trace(Q @ np.linalg.inv(P)) - 2.0))


def null_rhs(t, y, Mrest, a0, kap, T):
    dy = np.zeros_like(y)
    for i in range(N):
        al = y[4 * i]
        P = np.array([[y[4 * i + 1], y[4 * i + 2]], [y[4 * i + 2], y[4 * i + 3]]])
        Pinv = np.linalg.inv(P)
        L0 = np.eye(2) + kap / N * (al * Pinv + Mrest[i])
        D = kl(al, P, a0, L0)
        pi = 1.0 / N
        beta = pi * np.exp(-D / T) / (pi * np.exp(-D / T) + 1 - pi)
        C0 = L0 + kap / N * (al * Pinv - a0 * np.linalg.inv(L0))
        dP = -(1 + beta) * P + np.eye(2) + beta * C0
        dy[4 * i] = -(1 + beta) * (al - a0)
        dy[4 * i + 1], dy[4 * i + 2], dy[4 * i + 3] = dP[0, 0], dP[0, 1], dP[1, 1]
    return dy


def run_selector(shape_fn, tmax, delta, a0, n_grid=4000):
    """Threshold rule of form:threshold-rule on blocks read from fine shapes alpha(t)."""
    blocks = [frozenset([i]) for i in range(N)]
    hist = []
    ts = list(np.linspace(0, tmax, n_grid))

    def dist(I, J, t):
        a = shape_fn(t)
        return abs(np.mean([a[i] for i in I]) - np.mean([a[j] for j in J]))
    k, t_last = 0, 0.0
    while len(blocks) > 1 and k < len(ts) - 1:
        k += 1
        lo, hi = max(ts[k - 1], t_last), ts[k]
        crossed = []
        for I, J in itertools.combinations(blocks, 2):
            if dist(I, J, lo) > delta >= dist(I, J, hi):
                tc = brentq(lambda s: dist(I, J, s) - delta, lo, hi, xtol=1e-13)
                crossed.append((tc, I, J))
        if crossed:
            crossed.sort(key=lambda c: c[0])
            tc, I, J = crossed[0]
            blocks = [B for B in blocks if B not in (I, J)] + [I | J]
            hist.append((round(tc, 6), tuple(sorted(I | J))))
            t_last = tc + 1e-12
            k -= 1
    return hist


def full_shapes(alpha_init, a0):
    return lambda t: a0 + np.exp(-2 * t) * (np.asarray(alpha_init) - a0)


def r2_shapes(alpha_init, P0, a0, kap, T, tmax):
    y0 = np.concatenate([[a, P0[0, 0], P0[0, 1], P0[1, 1]] for a in alpha_init])
    Mrest = [sum(alpha_init[j] * np.linalg.inv(P0) for j in range(N) if j != i) for i in range(N)]
    sol = solve_ivp(null_rhs, (0, tmax), y0, args=(Mrest, a0, kap, T), dense_output=True,
                    rtol=1e-11, atol=1e-13, method="DOP853")
    return lambda t: sol.sol(t)[0::4]


a0, delta, Tq = 1.0, 0.1, 1.0
P0 = np.eye(2)
print()
print("(b) membership histories (time, joined block); alpha0 = 1, delta = 0.1, T_q = 1, P(0) = I")
cases = {"chapter example (1,1.2,1.8,2.1)": [1.0, 1.2, 1.8, 2.1],
         "gaps (0.30,0.60,0.29)": list(1 + np.concatenate([[0], np.cumsum([0.30, 0.60, 0.29])]))}
for name, al0 in cases.items():
    Hf = run_selector(full_shapes(al0, a0), 6.0, delta, a0)
    print(f"  {name}: full flow (every kappa) {Hf}")
    for kap in [0.0, 1.0, 5.0]:
        H0 = run_selector(r2_shapes(al0, P0, a0, kap, Tq, 12.0), 12.0, delta, a0)
        same = [b for _, b in H0] == [b for _, b in Hf]
        print(f"     R2 null kappa={kap}: {H0}   partition sequence equal: {same}")

print()
print("(c) scan of d3 at d1 = 0.30, d2 = 0.60 (kappa = 1): first join under full flow vs R2 null")
for d3 in [0.27, 0.28, 0.285, 0.29, 0.295, 0.299, 0.301, 0.31]:
    al0 = list(1 + np.concatenate([[0], np.cumsum([0.30, 0.60, d3])]))
    Hf = run_selector(full_shapes(al0, a0), 6.0, delta, a0)
    H0 = run_selector(r2_shapes(al0, P0, a0, 1.0, Tq, 12.0), 12.0, delta, a0)
    same = [b for _, b in H0] == [b for _, b in Hf]
    print(f"   d3={d3:.3f}: full first {Hf[0][1]} at {Hf[0][0]:.4f}; R2 null first {H0[0][1]} at {H0[0][0]:.4f}; "
          f"sequences equal: {same}")
