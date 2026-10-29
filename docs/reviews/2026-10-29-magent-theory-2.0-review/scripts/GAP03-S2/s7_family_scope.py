r"""GAP03-S2 s7: scope of the parent mechanism (Extensions / GAP03-S2-07).
The proof of form:parent uses only (A) Bregman linearity of Fisher-Rao KL flows in natural
parameters, (B) the natural gradient of a function of the mean moment, and (C) a sum map that is
affine in natural parameters on the declared common family, with parent targets equal to the
sum pushforwards.  Check (C)-type families by finite-difference natural gradients:
  Gaussian N(m_i, v) with common v (sum map affine): exact intertwining expected;
  Poisson(lambda_i) (sum map not affine in log lambda): failure expected.
Potential (receiver-independent targets, total-moment channel), N primitive agents:
  E = sum_l [KL(q_l || ref) + KL(q_l || tgt(M))],  M = sum_l E X_l.
Parent targets: n_I-fold sum pushforwards of the fine targets.
"""
import numpy as np

kap, N = 0.7, 4


def num_grad(fun, x, h=1e-6):
    g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x); e[k] = h
        g[k] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


# ---------------- Gaussian, 1D ----------------
m0, v0, V = 0.3, 1.5, 0.8
def kl_g(m, v, a, b):
    return 0.5 * (v / b + (m - a) ** 2 / b - 1 + np.log(b / v))
def mu_of(M):
    return kap * M / N            # channel mean depends on the total first moment


def E_fine_g(x):                  # x = (m_1..m_N, v_1..v_N)
    m, v = x[:N], x[N:]
    M = m.sum()
    return sum(kl_g(m[i], v[i], m0, v0) + kl_g(m[i], v[i], mu_of(M), V) for i in range(N))


def E_par_g(x, ns):               # parents: means m_I, variances v_I, counts ns
    k = len(ns); m, v = x[:k], x[k:]
    M = m.sum()
    return sum(kl_g(m[I], v[I], ns[I] * m0, ns[I] * v0) + kl_g(m[I], v[I], ns[I] * mu_of(M), ns[I] * V) for I in range(k))


rng = np.random.default_rng(3)
m = rng.normal(size=N); v = 0.9                      # common variance
x = np.concatenate([m, np.full(N, v)])
g = num_grad(E_fine_g, x)
vel_fine = np.concatenate([-x[N:] * g[:N], -2 * x[N:] ** 2 * g[N:]])   # Fisher metric diag(1/v, 1/(2v^2))
for ns in ([2, 2], [1, 2, 1], [4]):
    idx = np.cumsum([0] + ns)
    mI = np.array([m[idx[I]:idx[I + 1]].sum() for I in range(len(ns))])
    vI = np.array([n * v for n in ns])
    xp = np.concatenate([mI, vI])
    gp = num_grad(lambda y: E_par_g(y, ns), xp)
    vel_par = np.concatenate([-xp[len(ns):] * gp[:len(ns)], -2 * xp[len(ns):] ** 2 * gp[len(ns):]])
    push_m = np.array([vel_fine[idx[I]:idx[I + 1]].sum() for I in range(len(ns))])
    push_v = np.array([n * vel_fine[N] for n in ns])
    print("Gaussian common variance, blocks %s: |parent - pushforward| mean %.2e, variance %.2e; fine variance velocities spread %.2e"
          % (ns, np.abs(vel_par[:len(ns)] - push_m).max(), np.abs(vel_par[len(ns):] - push_v).max(), np.ptp(vel_fine[N:])))

# ---------------- Poisson ----------------
l0 = 2.0
def kl_p(l, a):
    return l * np.log(l / a) - l + a
def E_fine_p(lam):
    M = lam.sum()
    return sum(kl_p(lam[i], l0) + kl_p(lam[i], 1 + kap * M / N) for i in range(N))
def E_par_p(lam, ns):
    M = lam.sum()
    return sum(kl_p(lam[I], ns[I] * l0) + kl_p(lam[I], ns[I] * (1 + kap * M / N)) for I in range(len(ns)))
lam = np.array([1.0, 3.0, 2.5, 0.7])
gl = num_grad(E_fine_p, lam)
vel_f = -lam * gl                                       # Fisher metric 1/lambda
for ns in ([2, 2], [4]):
    idx = np.cumsum([0] + ns)
    lI = np.array([lam[idx[I]:idx[I + 1]].sum() for I in range(len(ns))])
    gpl = num_grad(lambda y: E_par_p(y, ns), lI)
    vel_p = -lI * gpl
    push = np.array([vel_f[idx[I]:idx[I + 1]].sum() for I in range(len(ns))])
    print("Poisson, blocks %s: parent velocity %s vs pushforward %s" % (ns, np.round(vel_p, 6), np.round(push, 6)))
