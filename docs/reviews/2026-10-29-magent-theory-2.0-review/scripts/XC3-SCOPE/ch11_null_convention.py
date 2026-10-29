"""XC3-SCOPE deep check: chapter 11 (form:kappa-independence (iii)) null-flow convention.

Chapter 11 computes the null flow with the nonidentity self cost at weight one:
    alpha_i' = -2 (alpha_i - alpha0),  P_i' = -2 P_i + I + C_i^0.
Chapter 13 (pa:null-row) and chapter 14 declare the null of inh:null-flow as
"deletes the nonself costs, not their labels or priors" and keeps the row entropy, so
with Gibbs rows at temperature T and priors 1/N the self weight is
    b_i = (1/N) e^{-D_ii/T} / ((1/N) e^{-D_ii/T} + (N-1)/N)  < 1,
giving alpha_i' = -(1+b_i)(alpha_i - alpha0),  P_i' = I + b_i C_i^0 - (1+b_i) P_i.
This script integrates both conventions for the chapter 11 example and for a nearby
admitted configuration, applies the threshold rule form:threshold-rule, and reports
partition sequences and event times.  Single-threaded, a few seconds.
"""
import numpy as np
from math import lgamma, log, pi
from scipy.special import digamma
from scipy.integrate import solve_ivp

def lgamma2(a):
    return 0.5 * log(pi) + lgamma(a) + lgamma(a - 0.5)

def psi2(a):
    return digamma(a) + digamma(a - 0.5)

def kl_wishart(al, P, a0, Lam):
    """KL(W2(al,P) || W2(a0,Lam)) for density ~ det(X)^(nu-3/2) exp(-tr(Prec X))."""
    Pinv = np.linalg.inv(P)
    return (a0 * np.log(np.linalg.det(P) / np.linalg.det(Lam)) + lgamma2(a0) - lgamma2(al)
            + (al - a0) * psi2(al) + al * np.trace(Lam @ Pinv) - 2.0 * al)

def make_rhs(conv, a0, kappa, N, T, alpha_init, P_init):
    I2 = np.eye(2)
    M0 = [alpha_init[j] * np.linalg.inv(P_init[j]) for j in range(N)]
    def rhs(t, y):
        al = y[:N]
        Ps = [y[N + 4 * i: N + 4 * i + 4].reshape(2, 2) for i in range(N)]
        dal = np.zeros(N); dP = []
        for i in range(N):
            P = 0.5 * (Ps[i] + Ps[i].T)
            Mi = al[i] * np.linalg.inv(P)
            Mm = sum(M0[j] for j in range(N) if j != i)
            Lam = I2 + (kappa / N) * (Mi + Mm)
            C0 = Lam + (kappa / N) * (Mi - a0 * np.linalg.inv(Lam))
            if conv == "A":
                b = 1.0
            else:
                D = kl_wishart(al[i], P, a0, Lam)
                w = (1.0 / N) * np.exp(-D / T)
                b = w / (w + (N - 1.0) / N)
            dal[i] = -(1.0 + b) * (al[i] - a0)
            dP.append((I2 + b * C0 - (1.0 + b) * P).ravel())
        return np.concatenate([dal] + dP)
    return rhs

def events(times, alphas, a0, delta):
    """Threshold rule: join current blocks when |b_I-b_J| first reaches delta from above."""
    N = alphas.shape[0]
    blocks = [[i] for i in range(N)]
    hist = []
    def score(B, k):
        return np.mean(alphas[B, k]) - a0
    for k in range(1, len(times)):
        changed = True
        while changed and len(blocks) > 1:
            changed = False
            best = None
            for x in range(len(blocks)):
                for z in range(x + 1, len(blocks)):
                    dprev = abs(score(blocks[x], k - 1) - score(blocks[z], k - 1))
                    dnow = abs(score(blocks[x], k) - score(blocks[z], k))
                    if dprev > delta >= dnow:
                        # linear interpolation of crossing time
                        tc = times[k - 1] + (dprev - delta) / (dprev - dnow) * (times[k] - times[k - 1])
                        if best is None or tc < best[0]:
                            best = (tc, x, z)
            if best is not None:
                tc, x, z = best
                hist.append((round(tc, 4), sorted(blocks[x] + blocks[z])))
                newb = blocks[x] + blocks[z]
                blocks = [b for q, b in enumerate(blocks) if q not in (x, z)] + [newb]
                changed = True
    return hist

def run(u, a0=1.0, kappa=1.0, delta=0.1, T=1.0, tmax=4.0):
    N = len(u)
    P_init = [np.eye(2) for _ in range(N)]
    y0 = np.concatenate([np.array(u, float)] + [P.ravel() for P in P_init])
    times = np.linspace(0.0, tmax, 40001)
    out = {}
    for conv in ("A", "B"):
        sol = solve_ivp(make_rhs(conv, a0, kappa, N, T, list(u), P_init), (0, tmax), y0,
                        t_eval=times, rtol=1e-10, atol=1e-12)
        out[conv] = events(times, sol.y[:N], a0, delta)
        if conv == "B":
            # report the self weights at t=0
            rhs = make_rhs("B", a0, kappa, N, T, list(u), P_init)
            al = np.array(u, float); bs = []
            for i in range(N):
                Mi = al[i] * np.eye(2); Mm = sum(u[j] * np.eye(2) for j in range(N) if j != i)
                Lam = np.eye(2) + (kappa / N) * (Mi + Mm)
                D = kl_wishart(al[i], np.eye(2), a0, Lam)
                w = (1.0 / N) * np.exp(-D / T); bs.append(w / (w + (N - 1.0) / N))
            out["b0"] = np.round(bs, 4)
    return out

if __name__ == "__main__":
    print("full flow (= convention A shape law): events at t=0.5 log(D/delta)")
    print("  chapter 11 example: 0.5log2=%.4f 0.5log3=%.4f 0.5log(17/2)=%.4f" % (0.5*np.log(2), 0.5*np.log(3), 0.5*np.log(8.5)))
    for T in (0.1, 1.0, 10.0):
        r = run([1.0, 1.2, 1.8, 2.1], T=T)
        print("T=%g example u=(1,6/5,9/5,21/10): self weights b_i(0) under B:" % T, r["b0"])
        print("   A (ch11 null):", r["A"])
        print("   B (ch13 null):", r["B"])
    # a configuration inside the admitted class form:gaps, near its boundary
    u2 = [1.0, 1.2, 1.45, 1.79]  # gaps d=(0.2,0.25,0.34): delta<0.2<0.25, 0.34<0.25+0.1
    for T in (0.1, 1.0):
        r = run(u2, T=T, tmax=4.0)
        print("T=%g near-boundary u=%s: b_i(0) under B:" % (T, u2), r["b0"])
        print("   A (ch11 null):", r["A"])
        print("   B (ch13 null):", r["B"])
