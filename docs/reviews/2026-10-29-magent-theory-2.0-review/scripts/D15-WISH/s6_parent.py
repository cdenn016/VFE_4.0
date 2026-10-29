r"""D15-WISH s6: form:parent. Parent natural gradient versus summed fine velocities.

Parent energy  E^c = sum_I [ KL(W(A_I,P_I) || W(n_I alpha0, I)) + KL(W(A_I,P_I) || W(n_I alpha0, Lambda^c)) ],
Lambda^c = I + (kappa/N) sum_I A_I P_I^{-1} with the PRIMITIVE N = 4.
Fine prediction: dA_I/dt = -2(A_I - n_I alpha0), dP/dt = -2P + I + C_q (same C_q as the fine flow).
Variants that the chapter warns against: current parent count in Lambda^c, frozen channel, parent
target shape alpha0 instead of n_I alpha0.
"""
import mpmath as mp

mp.mp.dps = 30


def lg2(a):
    return mp.log(mp.sqrt(mp.pi)) + mp.loggamma(a) + mp.loggamma(a - mp.mpf(1) / 2)


def psi2(a):
    return mp.digamma(a) + mp.digamma(a - mp.mpf(1) / 2)


def kl(a, P, b, Q):
    Pi = P ** -1
    return ((a - b) * psi2(a) - lg2(a) + lg2(b) + b * mp.log(mp.det(P) / mp.det(Q))
            + a * ((Q * Pi)[0, 0] + (Q * Pi)[1, 1] - 2))


def unpack(th):
    return th[0], mp.matrix([[th[1], th[2]], [th[2], th[3]]])


def E_parent(TH, ns, a0, kap, Ncount, frozen=None, shape_mode="n"):
    M = mp.zeros(2, 2)
    for th in TH:
        A, P = unpack(th)
        M += A * P ** -1
    Lam = mp.eye(2) + (kap / Ncount) * M if frozen is None else frozen
    E = 0
    for th, n in zip(TH, ns):
        A, P = unpack(th)
        b = n * a0 if shape_mode == "n" else a0
        E += kl(A, P, b, mp.eye(2)) + kl(A, P, b, Lam)
    return E


def fisher(th):
    def psi(*x):
        A, P = unpack(x)
        return lg2(A) - A * mp.log(mp.det(P))
    G = mp.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            o = [0] * 4
            o[i] += 1
            o[j] += 1
            G[i, j] = mp.diff(psi, tuple(th), tuple(o))
    return G


def natgrad(TH, Efun):
    out = []
    for i in range(len(TH)):
        g = mp.matrix(4, 1)
        for k in range(4):
            def f(x, i=i, k=k):
                T2 = [list(t) for t in TH]
                T2[i][k] = x
                return Efun(T2)
            g[k] = mp.diff(f, TH[i][k])
        out.append(-(fisher(TH[i]) ** -1) * g)
    return out


alphas = [mp.mpf("1.0"), mp.mpf("1.2"), mp.mpf("1.8"), mp.mpf("2.1")]
P = mp.matrix([[mp.mpf("1.3"), mp.mpf("0.2")], [mp.mpf("0.2"), mp.mpf("0.9")]])
a0, kap, N = mp.mpf(1), mp.mpf(1), 4
M = sum(alphas) * P ** -1
Lam = mp.eye(2) + (kap / N) * M
C = mp.eye(2) + (2 * kap / N) * M - kap * a0 * Lam ** -1
Pdot = -2 * P + mp.eye(2) + C
for blocks in ([[0, 1], [2, 3]], [[0, 1, 2, 3]], [[0], [1, 2], [3]]):
    ns = [len(b) for b in blocks]
    TH = [[sum(alphas[i] for i in b), P[0, 0], P[0, 1], P[1, 1]] for b in blocks]
    pred_A = [-2 * (sum(alphas[i] for i in b) - len(b) * a0) for b in blocks]
    variants = {"declared (primitive N, live channel, shapes n_I alpha0)": dict(Ncount=N),
                "current parent count in Lambda": dict(Ncount=len(blocks)),
                "frozen channel": dict(Ncount=N, frozen=Lam),
                "parent target shape alpha0": dict(Ncount=N, shape_mode="1")}
    print(f"blocks {blocks}:")
    for name, kw in variants.items():
        V = natgrad(TH, lambda T: E_parent(T, ns, a0, kap, **kw))
        rA = max(abs(v[0] - pa) for v, pa in zip(V, pred_A))
        rP = max(max(abs(v[1] - Pdot[0, 0]), abs(v[2] - Pdot[0, 1]), abs(v[3] - Pdot[1, 1])) for v in V)
        print(f"   {name}: max |A velocity - summed fine| = {float(rA):.2e}, max |P velocity - fine| = {float(rP):.2e}")
