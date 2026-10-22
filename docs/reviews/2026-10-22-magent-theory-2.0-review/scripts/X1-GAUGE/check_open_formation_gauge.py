"""X1-GAUGE: chapter 12 (open-law formation) gauge structure on an exact grid discretization
of the bit-circle model. The Haar-angle channel on an M-point angle grid is the uniform
average over the M grid rotations, so the discretized model has the same structure and the
same exact formulas (open:cost, open:row, open:full-field). Checks:
 (a) equivariance under independent per-agent AND per-sector grid rotations,
 (b) conditional angular law f(.|z,t) proportional to f(.|z,0)^{exp(-2t)} for correlated laws,
 (c) exact reduction of the bit marginals to the binary field plus an exogenous forcing,
 (d) theta_dot = omega (open:moving-flow coefficient),
 (e) Haar channel = S^1 twirl; H_i = KL(q || twirl q) = mutual information I(g; X),
 (f) the aggregate response is H_I-invariant and not (S^1)^I-invariant,
 (g) monodromy of the equal-phase representative versus a global one-agent anchor."""
import numpy as np

rng = np.random.default_rng(11)
N, M, T = 3, 24, 0.7
th = 2 * np.pi * np.arange(M) / M

def rand_law():
    a = rng.normal(size=(2, 1)) * 0.6
    phase = rng.uniform(0, 2 * np.pi, size=(2, 1))
    kap = rng.uniform(0.5, 1.5, size=(2, 1))
    q = np.exp(a + kap * np.cos(th[None, :] - phase) + 0.2 * rng.normal(size=(2, M)))
    return q / q.mean()

def marg(q):
    return q.mean(axis=1)

def rows(Q):
    mus = [marg(q) for q in Q]
    D = np.array([[np.mean(Q[i] * np.log(Q[i] / mus[j][:, None])) for j in range(N)] for i in range(N)])
    W = np.exp(-D / T)
    return W / W.sum(axis=1, keepdims=True), D

def field(Q):
    beta, _ = rows(Q)
    mus = [marg(q) for q in Q]
    V = []
    for i in range(N):
        h = 2 * np.log(Q[i])
        h = h - sum(beta[i, j] * np.log(mus[j])[:, None] for j in range(N))
        h = h - sum(beta[k, i] * (mus[k] / mus[i])[:, None] for k in range(N))
        V.append(-Q[i] * (h - np.mean(Q[i] * h)))
    return V

def rk4(Q, t1, dt=0.002):
    Q = [q.copy() for q in Q]
    traj = [Q]
    n = int(round(t1 / dt))
    for _ in range(n):
        k1 = field(Q)
        k2 = field([q + 0.5 * dt * k for q, k in zip(Q, k1)])
        k3 = field([q + 0.5 * dt * k for q, k in zip(Q, k2)])
        k4 = field([q + dt * k for q, k in zip(Q, k3)])
        Q = [q + dt / 6 * (a + 2 * b + 2 * c + d) for q, a, b, c, d in zip(Q, k1, k2, k3, k4)]
        traj.append(Q)
    return traj

Q0 = [rand_law() for _ in range(N)]
S0 = [rand_law() for _ in range(N)]
t1, dt = 0.3, 0.002
trQ = rk4(Q0, t1, dt)
trS = rk4(S0, t1, dt)

kq = [3, 7, 11]
ks = [5, 0, 19]
rot = lambda q, k: np.roll(q, k, axis=1)
trQr = rk4([rot(q, k) for q, k in zip(Q0, kq)], t1, dt)
trSr = rk4([rot(s, k) for s, k in zip(S0, ks)], t1, dt)
errQ = max(np.abs(rot(trQ[-1][i], kq[i]) - trQr[-1][i]).max() for i in range(N))
errS = max(np.abs(rot(trS[-1][i], ks[i]) - trSr[-1][i]).max() for i in range(N))
print("(a) per-agent per-sector rotation equivariance residuals:", errQ, errS)

err_b = 0.0
for i in range(N):
    f0 = Q0[i] / marg(Q0[i])[:, None]
    ft = trQ[-1][i] / marg(trQ[-1][i])[:, None]
    pred = f0 ** np.exp(-2 * t1)
    pred = pred / pred.mean(axis=1, keepdims=True)
    err_b = max(err_b, np.abs(ft - pred).max())
print("(b) conditional angular law vs f0^{exp(-2t)} residual:", err_b)

def c_forcing(i, t):
    f0 = Q0[i] / marg(Q0[i])[:, None]
    f = f0 ** np.exp(-2 * t)
    f = f / f.mean(axis=1, keepdims=True)
    return 2 * np.mean(f * np.log(f), axis=1)

def bin_field(x, t, forcing=True):
    a = np.arctanh(x)
    mu = [np.array([1 + xi, 1 - xi]) for xi in x]
    d = np.array([[np.sum(0.5 * mu[i] * np.log(mu[i] / mu[j])) for j in range(N)] for i in range(N)])
    W = np.exp(-d / T)
    beta = W / W.sum(axis=1, keepdims=True)
    out = np.zeros(N)
    for i in range(N):
        out[i] = (1 - x[i] ** 2) * (-a[i] - np.sum(beta[i] * (a[i] - a))) + np.sum(beta[:, i] * (x - x[i]))
        if forcing:
            cz = c_forcing(i, t)
            out[i] -= (1 - x[i] ** 2) * (cz[0] - cz[1]) / 2
    return out

def integrate_bits(forcing):
    x = np.array([0.5 * (marg(q)[0] - marg(q)[1]) for q in Q0])
    t = 0.0
    for _ in range(int(round(t1 / dt))):
        k1 = bin_field(x, t, forcing)
        k2 = bin_field(x + dt / 2 * k1, t + dt / 2, forcing)
        k3 = bin_field(x + dt / 2 * k2, t + dt / 2, forcing)
        k4 = bin_field(x + dt * k3, t + dt, forcing)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return x

x_full = np.array([0.5 * (marg(q)[0] - marg(q)[1]) for q in trQ[-1]])
print("(c) bits: full flow vs reduced binary field + exogenous forcing:", np.abs(integrate_bits(True) - x_full).max())
print("    without the forcing term the discrepancy is:", np.abs(integrate_bits(False) - x_full).max())

u = lambda q: np.mean(q * np.exp(1j * th)[None, :])
k_mid = len(trQ) // 2
for i in range(N):
    qm = trQ[k_mid][i]
    Vm = field(trQ[k_mid])[i]
    omega = np.imag(np.mean(Vm * np.exp(1j * th)[None, :]) / u(qm))
    fd = (np.angle(u(trQ[k_mid + 1][i])) - np.angle(u(trQ[k_mid - 1][i]))) / (2 * dt)
    print(f"(d) agent {i}: omega = {omega:.8f}, finite-difference d arg u/dt = {fd:.8f}")

q = Q0[0]
mu1 = marg(Q0[1])
tw = np.mean([rot(q, k) for k in range(M)], axis=0)
print("(e) Haar channel output = twirl:", np.abs(tw - marg(q)[:, None]).max())
H = np.mean(q * np.log(q / tw))
D01 = np.mean(q * np.log(q / mu1[:, None]))
KLmu = np.sum(0.5 * marg(q) * np.log(marg(q) / mu1))
print("    Pythagorean D_01 - (H_0 + KL(mu_0||mu_1)):", D01 - (H + KLmu))
MI = np.mean([np.mean(rot(q, k) * np.log(rot(q, k) / tw)) for k in range(M)])
print("    H_0 =", H, " mutual information I(g;X) =", MI)

def response_law(Qs):
    P = np.zeros((2, 2, 2, M))
    p = [qq / (2 * M) for qq in Qs]
    for z1 in range(2):
        for z2 in range(2):
            for z3 in range(2):
                P[z1, z2, z3] = np.real(np.fft.ifft(np.fft.fft(p[0][z1]) * np.fft.fft(p[1][z2]) * np.fft.fft(p[2][z3])))
    return P
R0 = response_law(Q0)
Rh = response_law([rot(Q0[0], 5), rot(Q0[1], 7), rot(Q0[2], M - 12)])
R1 = response_law([rot(Q0[0], 5), Q0[1], Q0[2]])
print("(f) response law change under H_I element:", np.abs(Rh - R0).max(), " under single rotation:", np.abs(R1 - R0).max())

n = 3
varth = np.array([0.4, 1.1, -0.9])
ts = np.linspace(0, 2 * np.pi, 401)
lift1 = varth[0] + ts
phis = [np.mean([l, varth[1], varth[2]]) for l in lift1]
print("(g) equal-phase representative: start", phis[0] % (2 * np.pi), " end", phis[-1] % (2 * np.pi),
      " shift", phis[-1] - phis[0], " 2pi/n =", 2 * np.pi / n)
Phi = [(l + varth[1] + varth[2]) % (2 * np.pi) for l in lift1]
print("    one-agent anchor representative (Phi,0,0) closes: start", Phi[0], " end", Phi[-1])
