r"""L06-ERA script 8: rate check for a block-phase extension (no gate, identity channels).

Binary agents (A = B uniform on {-1,1}, phi = psi = identity), q_i = Ber(x_i), s_i = Ber(y_i),
N = 4, J = log 3 (bistable local energy), identity peer channels, uniform nonself priors,
temperature T.  Find the stable equilibrium with agents {0,2} in the + well and {1,3} in
the - well, linearize the original unit-Fisher flow (mean coordinates, which give the
same eigenvalues as any chart at an equilibrium), and split the spectrum into
  tangential: modes inside Fix(S_2 x S_2) (block-equal configurations),
  normal:     intra-block antisymmetric modes.
A C^1 asymptotic-phase fibration onto the block-equal manifold near the equilibrium needs
min(normal rate) > max(tangential rate) (normal contraction dominates).
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

J = np.log(3.0)
N = 4


def d(x, z):
    return 0.5 * (1 + x) * np.log((1 + x) / (1 + z)) + 0.5 * (1 - x) * np.log((1 - x) / (1 - z))


def grad_sector(p, o, T):
    """Envelope gradient of one sector: local + own row + incoming."""
    g = np.arctanh(p) - J * o
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j != i:
                B[i, j] = np.exp(-d(p[i], p[j]) / T)
        B[i] /= B[i].sum()
    for i in range(N):
        for j in range(N):
            if j != i:
                g[i] += B[i, j] * (np.arctanh(p[i]) - np.arctanh(p[j]))
                g[i] += B[j, i] * (p[i] - p[j]) / (1 - p[i]**2)
    return g


def field(X, T):
    x, y = X[:N], X[N:]
    return np.concatenate([-(1 - x**2) * grad_sector(x, y, T), -(1 - y**2) * grad_sector(y, x, T)])


for T in (0.05, 0.1, 0.2):
    X0 = np.array([0.5, -0.5, 0.5, -0.5] * 2)
    X0 = X0 + 1e-3 * np.array([1, -2, -1, 2, 2, -1, -2, 1])
    X0 = solve_ivp(lambda t, X: field(X, T), (0, 300), X0, method='LSODA', rtol=1e-11, atol=1e-13).y[:, -1]
    Xs = fsolve(lambda X: field(X, T), X0, xtol=1e-14)
    h = 1e-6
    Jac = np.zeros((2 * N, 2 * N))
    for k in range(2 * N):
        e = np.zeros(2 * N)
        e[k] = h
        Jac[:, k] = (field(Xs + e, T) - field(Xs - e, T)) / (2 * h)
    ev, V = np.linalg.eig(Jac)
    # classify by intra-block antisymmetry: normal if x0+x2 = 0 and x1+x3 = 0 (same for y)
    rows = []
    for k in range(2 * N):
        v = np.real(V[:, k])
        sym = np.linalg.norm([v[0] - v[2], v[1] - v[3], v[4] - v[6], v[5] - v[7]])
        anti = np.linalg.norm([v[0] + v[2], v[1] + v[3], v[4] + v[6], v[5] + v[7]])
        rows.append(('tangential' if sym < 1e-6 else 'normal' if anti < 1e-6 else 'mixed', -np.real(ev[k])))
    tan = [r for c, r in rows if c == 'tangential']
    nor = [r for c, r in rows if c == 'normal']
    print(f'T={T}: equilibrium x={np.round(Xs[:N], 5)}, |field|={np.abs(field(Xs, T)).max():.1e}')
    print(f'T={T}: tangential decay rates {np.round(sorted(tan), 4)}; normal decay rates {np.round(sorted(nor), 4)}; '
          f'other {[c for c, r in rows if c == "mixed"]}')
    if nor and tan:
        print(f'T={T}: stable={all(-np.real(ev) > 0)}, min normal / max tangential = {min(nor)/max(tan):.3f}, '
              f'2*min normal / max tangential = {2*min(nor)/max(tan):.3f}')
    print(f'T={T}: full spectrum decay rates {np.round(np.sort(-np.real(ev)), 4)}')
