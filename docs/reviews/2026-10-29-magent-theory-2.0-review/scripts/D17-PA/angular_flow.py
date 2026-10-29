r"""D17-PA check 4: null box and three-agent angular parameter flows (pa:angular-grouping).

  (d) at random points of the box pa:null-box: D_ii (closed form) <= 2|eta-eta_G|^2 <= 72e^2+1/25+18e^6 < log 2,
      1/9 <= beta0 <= 1/5, inward-pointing faces, |L-l| <= 2|C|(|A|+|C|) < eps/8;
  (e) full and null parameter ODEs for three agents with o=(1,1,-1) and random angular records:
      partition history of pa:selector applied to the actual public odds L_i.
"""
import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import solve_ivp

rng = np.random.default_rng(5)
a_pr, p_pr = 0.2, 0.4
o = np.array([1.0, 1.0, -1.0])

def logI0(r):
    return np.log(i0e(r)) + np.abs(r)

def h_of(A):
    return logI0(np.hypot(A[0], A[1]))

def grad_h(A):
    r = np.hypot(A[0], A[1])
    return np.zeros(2) if r < 1e-300 else (i1e(r) / i0e(r)) * A / r

def L_of(l, A, C):
    return l + h_of(A + C) - h_of(A - C)

def stats(l, A, C, B):
    Pp = np.exp(l / 2 + h_of(A + C)); Pm = np.exp(-l / 2 + h_of(A - C)); s = Pp + Pm
    gp, gm = grad_h(A + C), grad_h(A - C)
    Ez = (Pp - Pm) / s
    Ee = (Pp * gp + Pm * gm) / s
    Eze = (Pp * gp - Pm * gm) / s
    Eps = grad_h(B)
    logZ = logI0(np.hypot(*B)) + np.log(s / 2)
    S = (l / 2) * Ez + A @ Ee + C @ Eze + B @ Eps - logZ   # E log q
    return Ez, Ee, Eze, Eps, S

def Dself(l, A, C, B, J):
    Ez, Ee, Eze, Eps, S = stats(l, A, C, B)
    return S - J * Eze[0] + logI0(J)

print("(d) null box checks")
worst_bound = 0.0; b0min, b0max = 1.0, 0.0; faces_ok = True; Lmax = 0.0
for _ in range(4000):
    eps = rng.uniform(1e-4, 0.01); J = eps ** 3
    oi = rng.choice([-1.0, 1.0])
    r = rng.uniform(0.25, 12.0); l = oi * r * eps
    A = rng.normal(size=2); A *= rng.uniform(0, 0.1) / np.linalg.norm(A)
    C = rng.normal(size=2); C *= rng.uniform(0, 2 * eps ** 3) / np.linalg.norm(C)
    B = rng.normal(size=2); B *= rng.uniform(0, 0.1) / np.linalg.norm(B)
    D = Dself(l, A, C, B, J)
    eta = np.concatenate([[l / 2], A, C, B]); etaG = np.array([0, 0, 0, J, 0, 0, 0.0])
    bnd = 2 * np.sum((eta - etaG) ** 2)
    worst_bound = max(worst_bound, D - bnd)
    b0 = a_pr * np.exp(-D) / (a_pr * np.exp(-D) + 2 * p_pr)
    b0min, b0max = min(b0min, b0), max(b0max, b0)
    Lmax = max(Lmax, abs(L_of(l, A, C) - l) / (eps / 8))
    if bnd > 72 * eps**2 + 1 / 25 + 18 * eps**6 + 1e-15 or 72 * eps**2 + 1 / 25 + 18 * eps**6 >= np.log(2):
        faces_ok = False
print("  max (D_ii - 2|eta-eta_G|^2) = %.2e (must be <= 0); beta0 range [%.4f, %.4f] (claim [1/9,1/5]=[%.4f,%.4f])" % (worst_bound, b0min, b0max, 1 / 9, 0.2))
print("  max |L-l|/(eps/8) = %.2e (must be < 1); bound chain consistent: %s" % (Lmax, faces_ok))

def unpack(y):
    s = y.reshape(3, 8)    # l, A1, A2, C1, C2, B1, B2, n
    return s

def rows_full(S, eps):
    J = eps ** 3
    Ls = np.array([L_of(S[i, 0], S[i, 1:3], S[i, 3:5]) for i in range(3)])
    beta = np.zeros((3, 3))
    for i in range(3):
        Ez, Ee, Eze, Eps, _ = stats(S[i, 0], S[i, 1:3], S[i, 3:5], S[i, 5:7])
        m_i = np.tanh(Ls[i] / 2)
        for j in range(3):
            beta[i, j] = a_pr * np.exp(J * Eze[0] - logI0(J)) if j == i else p_pr * np.exp(Ls[j] * m_i / 2) / np.cosh(Ls[j] / 2)
        beta[i] /= beta[i].sum()
    return beta, Ls

def gamma_rows(n):
    # presentation sector: Bernoulli log odds n, uniform self reference, identity peer channels
    def klb(x, y):
        px, py = 1 / (1 + np.exp(-x)), 1 / (1 + np.exp(-y))
        return px * np.log(px / py) + (1 - px) * np.log((1 - px) / (1 - py))
    g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            g[i, j] = (a_pr if i == j else p_pr) * np.exp(-(klb(n[i], 0.0) if i == j else klb(n[i], n[j])))
        g[i] /= g[i].sum()
    return g

def field(y, eps, om, null=False):
    S = unpack(y); J = eps ** 3; lam = eps
    out = np.zeros_like(S)
    e_om = np.stack([np.cos(om), np.sin(om)], axis=1)
    if not null:
        beta, Ls = rows_full(S, eps)
        g = gamma_rows(S[:, 7])
        for i in range(3):
            out[i, 0] = -S[i, 0] + sum(beta[i, j] * Ls[j] for j in range(3) if j != i) + eps * o[i]
            out[i, 1:3] = -S[i, 1:3] + lam * e_om[i]
            out[i, 3:5] = -S[i, 3:5] + beta[i, i] * J * np.array([1.0, 0.0])
            out[i, 5:7] = -S[i, 5:7]
            out[i, 7] = -S[i, 7] + sum(g[i, j] * S[j, 7] for j in range(3) if j != i)
    else:
        for i in range(3):
            D = Dself(S[i, 0], S[i, 1:3], S[i, 3:5], S[i, 5:7], J)
            b0 = a_pr * np.exp(-D) / (a_pr * np.exp(-D) + 2 * p_pr)
            out[i, 0] = -b0 * S[i, 0] + eps * o[i]
            out[i, 1:3] = -b0 * S[i, 1:3] + lam * e_om[i]
            out[i, 3:5] = b0 * (J * np.array([1.0, 0.0]) - S[i, 3:5])
            out[i, 5:7] = -b0 * S[i, 5:7]
            px = 1 / (1 + np.exp(-S[i, 7]))
            Ds = px * np.log(2 * px) + (1 - px) * np.log(2 * (1 - px))
            g0 = a_pr * np.exp(-Ds) / (a_pr * np.exp(-Ds) + 2 * p_pr)
            out[i, 7] = -g0 * S[i, 7]
    return out.ravel()

def partition(L):
    comp = [0, 1, 2]
    for i in range(3):
        for j in range(i + 1, 3):
            if L[i] * L[j] > 0 and min(abs(L[i]), abs(L[j])) / max(abs(L[i]), abs(L[j])) > 0.25:
                ci, cj = comp[i], comp[j]
                comp = [ci if c == cj else c for c in comp]
    blocks = {}
    for k, c in enumerate(comp):
        blocks.setdefault(c, []).append(k + 1)
    return tuple(sorted(tuple(b) for b in blocks.values()))

def run(y0, eps, om, T, null, n=2001):
    sol = solve_ivp(lambda t, y: field(y, eps, om, null), (0, T), y0, t_eval=np.linspace(0, T, n), rtol=1e-9, atol=1e-14)
    seq = []
    for k in range(sol.y.shape[1]):
        S = unpack(sol.y[:, k])
        P = partition([L_of(S[i, 0], S[i, 1:3], S[i, 3:5]) for i in range(3)])
        if not seq or seq[-1][1] != P:
            seq.append((round(float(sol.t[k]), 3), P))
    return seq, unpack(sol.y[:, -1])

print("(e) three-agent angular parameter flows")
u_in = np.array([4.0, 0.5, -1.0])
for eps in (0.01, 0.05, 0.2):
    nf = nn = 0
    NS = 6
    for trial in range(NS):
        om = rng.uniform(0, 2 * np.pi, size=3)
        S0 = np.zeros((3, 8))
        S0[:, 0] = eps * (u_in + 0.2 * rng.normal(size=3))
        for i in range(3):
            A = rng.normal(size=2); S0[i, 1:3] = A / np.linalg.norm(A) * rng.uniform(0.005, 0.05)
            C = rng.normal(size=2); S0[i, 3:5] = C / np.linalg.norm(C) * rng.uniform(0, 1) * eps ** 3
            th = rng.uniform(0, 2 * np.pi); S0[i, 5:7] = rng.uniform(0.005, 0.05) * np.array([np.cos(th), np.sin(th)])
            S0[i, 7] = rng.normal()
        seq, Sf = run(S0.ravel(), eps, om, 80.0, False)
        seq0, Sf0 = run(S0.ravel(), eps, om, 400.0, True)
        ok = seq[0][1] == ((1,), (2,), (3,)) and seq[-1][1] == ((1, 2, 3),)
        nf += int(ok); nn += int(any(P == ((1, 2, 3),) for _, P in seq0))
        if trial == 0:
            print("  eps=%.2f sample full history %s" % (eps, seq))
            print("           sample null history %s" % (seq0,))
            print("           final b (full) %s, final theta (full) %s" % (np.round(np.hypot(Sf[:, 5], Sf[:, 6]), 6), np.round(np.arctan2(Sf[:, 6], Sf[:, 5]), 4)))
    print("  eps=%.2f: full singletons-to-full-block %d/%d; null ever full block %d/%d" % (eps, nf, NS, nn, NS))
