# XC3-SCOPE extension check: which exponential families does the identity-channel
# reciprocal Fisher flow (Gibbs-reduced, with source-role terms) leave invariant?
# Claim tested: invariant iff span{1,T} is closed under products on the support,
# i.e. iff the family is the block-constant-density (Jeffrey) family of a partition.
import numpy as np
rng = np.random.default_rng(7)

def kl(p, q):
    return float(np.sum(p * np.log(p / q)))

def forces(Q, R, alpha, T, PI):
    # Q: (N, n) laws; R: (N, n) references; returns first variations h (N, n)
    N, n = Q.shape
    D = np.array([[kl(Q[i], Q[j]) if i != j else 0.0 for j in range(N)] for i in range(N)])
    B = np.zeros((N, N))
    for i in range(N):
        w = np.array([PI[i, j] * np.exp(-D[i, j] / T) if j != i else 0.0 for j in range(N)])
        B[i] = w / w.sum()
    H = np.zeros((N, n))
    for a in range(N):
        h = alpha * (np.log(Q[a] / R[a]) + 1.0)
        for j in range(N):
            if j != a:
                h = h + B[a, j] * (np.log(Q[a] / Q[j]) + 1.0)   # receiver role
        for i in range(N):
            if i != a:
                h = h - B[i, a] * Q[i] / Q[a]                    # source role
        H[a] = h
    return H

def velocity(L, R, alpha, T, PI):
    Q = np.exp(L); Q = Q / Q.sum(axis=1, keepdims=True)
    H = forces(Q, R, alpha, T, PI)
    return -(H - np.sum(Q * H, axis=1, keepdims=True)), Q

def outside_fraction(q, v, Tstat):
    # Fisher (L2(q)) norm of the part of v outside span{T_k - E_q T_k}
    A = Tstat - (q[:, None] * Tstat).sum(axis=0)
    W = np.sqrt(q)
    coef, *_ = np.linalg.lstsq(W[:, None] * A, W * v, rcond=None)
    r = v - A @ coef
    return float(np.sqrt(np.sum(q * r * r)) / np.sqrt(np.sum(q * v * v)))

def member(eta, Tstat, rho):
    p = rho * np.exp(Tstat @ eta)
    return p / p.sum()

def run(name, Tstat, rho, N=3, alpha=0.7, T=0.8, steps=400, dt=0.01):
    Tstat = Tstat / np.max(np.abs(Tstat), axis=0)  # same family, bounded natural parameters
    d = Tstat.shape[1]
    PI = np.full((N, N), 1.0 / (N - 1))
    Q0 = np.array([member(rng.normal(size=d), Tstat, rho) for _ in range(N)])
    R = np.array([member(rng.normal(size=d), Tstat, rho) for _ in range(N)])
    L = np.log(Q0)
    v0, _ = velocity(L, R, alpha, T, PI)
    frac0 = max(outside_fraction(Q0[a], v0[a], Tstat) for a in range(N))
    for _ in range(steps):
        k1, _ = velocity(L, R, alpha, T, PI)
        k2, _ = velocity(L + 0.5 * dt * k1, R, alpha, T, PI)
        k3, _ = velocity(L + 0.5 * dt * k2, R, alpha, T, PI)
        k4, _ = velocity(L + dt * k3, R, alpha, T, PI)
        L = L + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        L = L - np.log(np.exp(L).sum(axis=1, keepdims=True))
    Qt = np.exp(L)
    # distance of the final laws from the family: refit eta by least squares on log(q/rho)
    dist = 0.0
    for a in range(N):
        y = np.log(Qt[a] / rho)
        A = np.hstack([np.ones((len(rho), 1)), Tstat])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        dist = max(dist, float(np.max(np.abs(y - A @ c))))
    print(name, "| max outside fraction at t=0:", round(frac0, 6), "| sup log-distance from family at t=4:", "%.2e" % dist)

n = 6
rho = rng.uniform(0.5, 1.5, size=n); rho = rho / rho.sum()
blocks = [[0, 1], [2, 3, 4], [5]]
Tblock = np.zeros((n, 2)); Tblock[blocks[0], 0] = 1.0; Tblock[blocks[1], 1] = 1.0
run("block-constant family, 3 blocks on 6 points (d=2)", Tblock, rho)

t = np.arange(n, dtype=float)
run("one-parameter family T=t on 6 points (d=1)", t[:, None], rho)
run("two-parameter family T=(t,t^2) on 6 points (d=2)", np.stack([t, t * t], axis=1), rho)

t3 = np.array([0.0, 1.0, 2.0]); rho3 = np.ones(3) / 3
run("one-parameter family T=t on 3 points (d=1)", t3[:, None], rho3)
t2 = np.array([0.0, 1.0]); rho2 = np.ones(2) / 2
run("binary alphabet (full family)", t2[:, None], rho2)

# three-point identity behind the necessity argument: e^{eta t} = a + b t at t=0,1,2
for eta in [0.5, 0.1, 0.01]:
    a = 1.0; b = np.exp(eta) - 1.0
    print("eta", eta, "| residual e^{2 eta} - (a + 2 b) =", "%.3e" % (np.exp(2 * eta) - (a + 2 * b)), "| (e^eta - 1)^2 =", "%.3e" % ((np.exp(eta) - 1) ** 2))
