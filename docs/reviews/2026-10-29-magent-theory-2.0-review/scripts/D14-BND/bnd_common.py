r"""Shared helpers for lens D14-BND (chapter 10, boundary closure).

The fine system is built directly from the reciprocal potential of def:dyn-potential
with identity peer channels, a base-only evaluator density rho(r, w) and
unit-mobility Fisher rows. Nothing here uses the chapter's profile equations; the
profile system is implemented separately in ``profile_*`` functions so the two can be
compared.

Sample space in each sector: a finite base (weights A_r) times one public coordinate
per internal node. Two realizations:
  * coin:   theta_v in {-1, 1} uniform, sigma_v = theta_v * s_v(r)
  * circle: theta_v on an M_c-point uniform grid of the circle, theta_m = 2 pi (m + 1/2)/M_c,
            sigma_v = sign cos(theta_v - alpha_v(r)), alpha_v(r) a multiple of 2 pi / M_c.
Internal nodes are in heap order (root 0, children 2v+1 = left (eps=+1), 2v+2 = right (eps=-1)).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- tree
def tree(d: int):
    """Return N, M (internal nodes), paths[i] = [(ancestor node at depth j-1, eps_j)]_{j=1..d}."""
    N = 2 ** d
    M = N - 1
    paths = []
    for i in range(N):
        node = M + i
        anc = []
        while node > 0:
            parent = (node - 1) // 2
            eps = 1 if node == 2 * parent + 1 else -1
            anc.append((parent, eps))
            node = parent
        anc.reverse()
        paths.append(anc)
    return N, M, paths


def lca_class(paths, i: int, j: int) -> int:
    """Number k of shared initial edges between leaves i != j."""
    k = 0
    for (a, e), (b, f) in zip(paths[i], paths[j]):
        if a == b and e == f:
            k += 1
        else:
            break
    return k


def bits_to_idx(u) -> int:
    d = len(u)
    return int(sum(((1 - u[j]) // 2) << (d - 1 - j) for j in range(d)))


def all_bits(d: int):
    return [tuple(u) for u in itertools.product([1, -1], repeat=d)]  # idx order matches bits_to_idx


def P_matrix(k: int, d: int) -> np.ndarray:
    r"""P_k: keep bits 1..k, reverse bit k+1, average bits k+2..d (rows = u, cols = v)."""
    U = all_bits(d)
    n = 2 ** d
    P = np.zeros((n, n))
    tails = list(itertools.product([1, -1], repeat=d - k - 1))
    for u in U:
        for t in tails:
            v = u[:k] + (-u[k],) + tuple(t)
            P[bits_to_idx(u), bits_to_idx(v)] += 1.0 / len(tails)
    return P


# ----------------------------------------------------------------------------- sample space
@dataclass
class Space:
    d: int
    R: int                    # base points
    A: np.ndarray             # base weights (R,)
    vals: np.ndarray          # node coordinate values per cell, shape (C, M)
    Uc: np.ndarray            # cell weights (C,), uniform
    sig: np.ndarray           # sigma[r, c, v] in {-1,1}, shape (R, C, M)
    uidx: np.ndarray          # bit-string index of leaf i at (r, c): shape (N, R, C)


def make_space(d: int, A: np.ndarray, realization: str = "coin", orient=None, Mc: int = 8, alpha=None) -> Space:
    """Build the fine sample space and the leaf path-bit maps.

    orient[r, v] in {-1,1} (coin) or alpha[r, v] in {0..Mc-1} (circle, units of 2 pi / Mc).
    """
    N, M, paths = tree(d)
    R = len(A)
    if realization == "coin":
        vals = np.array(list(itertools.product([1, -1], repeat=M)), dtype=float)  # (C, M)
        if orient is None:
            orient = np.ones((R, M))
        sig = vals[None, :, :] * orient[:, None, :]
    elif realization == "circle":
        grid = np.arange(Mc)
        vals = np.array(list(itertools.product(grid, repeat=M)), dtype=float)  # grid index per node
        if alpha is None:
            alpha = np.zeros((R, M))
        theta = 2 * np.pi * (vals + 0.5) / Mc
        sig = np.sign(np.cos(theta[None, :, :] - 2 * np.pi * alpha[:, None, :] / Mc))
        assert np.all(sig != 0)
    else:
        raise ValueError(realization)
    C = vals.shape[0]
    Uc = np.full(C, 1.0 / C)
    uidx = np.zeros((N, R, C), dtype=int)
    for i in range(N):
        bits = np.stack([eps * sig[:, :, v] for (v, eps) in paths[i]], axis=-1)  # (R, C, d)
        idx = np.zeros((R, C), dtype=int)
        for j in range(d):
            idx += (((1 - bits[:, :, j]) // 2).astype(int)) << (d - 1 - j)
        uidx[i] = idx
    return Space(d, R, A, vals, Uc, sig, uidx)


def lift(f: np.ndarray, sp: Space) -> np.ndarray:
    """q_i(r, c) = f(r, u(i, r, c)); f has shape (R, 2^d)."""
    N = sp.uidx.shape[0]
    q = np.empty((N, sp.R, len(sp.Uc)))
    for i in range(N):
        q[i] = np.take_along_axis(f, sp.uidx[i], axis=1)
    return q


# ----------------------------------------------------------------------------- fine system
@dataclass
class FineData:
    spq: Space
    sps: Space
    logrho: np.ndarray        # (R, W)
    Tq: float
    Ts: float
    pi_row: np.ndarray        # (N, N) source priors, zero diagonal
    edges: np.ndarray         # (N, N) bool, i attends to j
    peer: bool = True         # False gives the null flow of inh:null-flow


def fine_energy(q, s, beta, gamma, D: FineData) -> float:
    lam = D.spq.A[:, None] * D.spq.Uc[None, :]
    mu = D.sps.A[:, None] * D.sps.Uc[None, :]
    N = q.shape[0]
    E = 0.0
    for i in range(N):
        E += np.sum(lam * q[i] * np.log(q[i])) + np.sum(mu * s[i] * np.log(s[i]))
        q0 = q[i] @ D.spq.Uc
        s0 = s[i] @ D.sps.Uc
        E -= np.sum(D.spq.A[:, None] * D.sps.A[None, :] * q0[:, None] * s0[None, :] * D.logrho)
    if D.peer:
        for i in range(N):
            for j in range(N):
                if D.edges[i, j]:
                    Dq = np.sum(lam * q[i] * (np.log(q[i]) - np.log(q[j])))
                    Ds = np.sum(mu * s[i] * (np.log(s[i]) - np.log(s[j])))
                    E += beta[i, j] * Dq + gamma[i, j] * Ds
    for i in range(N):
        m = D.edges[i]
        E += D.Tq * np.sum(beta[i, m] * np.log(beta[i, m] / D.pi_row[i, m]))
        E += D.Ts * np.sum(gamma[i, m] * np.log(gamma[i, m] / D.pi_row[i, m]))
    return E


def fine_forces(q, s, beta, gamma, D: FineData):
    r"""First variations h_i^q (w.r.t. the density against lambda) and row forces.

    h_i^q = log q_i + 1 + V_i + sum_j beta_ij (log q_i - log q_j + 1) - sum_l beta_li q_l / q_i
    """
    N = q.shape[0]
    lam = D.spq.A[:, None] * D.spq.Uc[None, :]
    mu = D.sps.A[:, None] * D.sps.Uc[None, :]
    lq, ls = np.log(q), np.log(s)
    hq = np.empty_like(q)
    hs = np.empty_like(s)
    Dq = np.zeros((N, N))
    Ds = np.zeros((N, N))
    for i in range(N):
        s0 = s[i] @ D.sps.Uc
        q0 = q[i] @ D.spq.Uc
        Vq = -(D.logrho * (D.sps.A * s0)[None, :]).sum(axis=1)   # (R,)
        Vs = -(D.logrho * (D.spq.A * q0)[:, None]).sum(axis=0)   # (W,)
        hq[i] = lq[i] + 1 + Vq[:, None]
        hs[i] = ls[i] + 1 + Vs[:, None]
    if D.peer:
        for i in range(N):
            for j in range(N):
                if D.edges[i, j]:
                    Dq[i, j] = np.sum(lam * q[i] * (lq[i] - lq[j]))
                    Ds[i, j] = np.sum(mu * s[i] * (ls[i] - ls[j]))
                    hq[i] += beta[i, j] * (lq[i] - lq[j] + 1)
                    hs[i] += gamma[i, j] * (ls[i] - ls[j] + 1)
                    hq[j] += -beta[i, j] * q[i] / q[j]
                    hs[j] += -gamma[i, j] * s[i] / s[j]
    hb = np.zeros((N, N))
    hg = np.zeros((N, N))
    for i in range(N):
        m = D.edges[i]
        hb[i, m] = Dq[i, m] + D.Tq * (np.log(beta[i, m] / D.pi_row[i, m]) + 1)
        hg[i, m] = Ds[i, m] + D.Ts * (np.log(gamma[i, m] / D.pi_row[i, m]) + 1)
    return hq, hs, hb, hg, Dq, Ds


def fine_velocity(state, D: FineData):
    q, s, beta, gamma = state
    hq, hs, hb, hg, _, _ = fine_forces(q, s, beta, gamma, D)
    lam = D.spq.A[:, None] * D.spq.Uc[None, :]
    mu = D.sps.A[:, None] * D.sps.Uc[None, :]
    dq = np.empty_like(q)
    ds = np.empty_like(s)
    for i in range(q.shape[0]):
        dq[i] = -q[i] * (hq[i] - np.sum(lam * q[i] * hq[i]))
        ds[i] = -s[i] * (hs[i] - np.sum(mu * s[i] * hs[i]))
    db = np.zeros_like(beta)
    dg = np.zeros_like(gamma)
    for i in range(q.shape[0]):
        m = D.edges[i]
        db[i, m] = -beta[i, m] * (hb[i, m] - np.sum(beta[i, m] * hb[i, m]))
        dg[i, m] = -gamma[i, m] * (hg[i, m] - np.sum(gamma[i, m] * hg[i, m]))
    return (dq, ds, db, dg)


def rk4(vel, state, dt: float, nsteps: int, args):
    x = state
    for _ in range(nsteps):
        k1 = vel(x, args)
        k2 = vel(tuple(a + 0.5 * dt * b for a, b in zip(x, k1)), args)
        k3 = vel(tuple(a + 0.5 * dt * b for a, b in zip(x, k2)), args)
        k4 = vel(tuple(a + dt * b for a, b in zip(x, k3)), args)
        x = tuple(a + dt / 6 * (b + 2 * c + 2 * e + g) for a, b, c, e, g in zip(x, k1, k2, k3, k4))
    return x


# ----------------------------------------------------------------------------- profile system
@dataclass
class ProfileData:
    d: int
    A: np.ndarray
    B: np.ndarray
    logrho: np.ndarray
    Tq: float
    Ts: float
    pik: np.ndarray           # class priors pi_k = n_k/(N-1)
    P: list                   # P_k matrices
    peer: bool = True


def profile_D(f, PD: ProfileData, base=None):
    """D_k(f) = int f (log f - P_k log f) d(base) dU_d; base defaults to PD.A."""
    base = PD.A if base is None else base
    lf = np.log(f)
    w = base[:, None] / f.shape[1]
    return np.array([np.sum(w * f * (lf - lf @ Pk.T)) for Pk in PD.P])


def profile_energy(f, g, bq, bs, PD: ProfileData) -> float:
    wq = PD.A[:, None] / f.shape[1]
    ws = PD.B[:, None] / g.shape[1]
    f0, g0 = f.mean(axis=1), g.mean(axis=1)
    F = np.sum(wq * f * np.log(f)) + np.sum(ws * g * np.log(g))
    F -= np.sum(PD.A[:, None] * PD.B[None, :] * f0[:, None] * g0[None, :] * PD.logrho)
    if PD.peer:
        F += bq @ profile_D(f, PD) + bs @ profile_D(g, PD, PD.B)
    F += PD.Tq * np.sum(bq * np.log(bq / PD.pik)) + PD.Ts * np.sum(bs * np.log(bs / PD.pik))
    return F


def profile_velocity(state, PD: ProfileData):
    f, g, bq, bs = state
    f0, g0 = f.mean(axis=1), g.mean(axis=1)
    Vq = -(PD.logrho * (PD.B * g0)[None, :]).sum(axis=1)
    Vs = -(PD.logrho * (PD.A * f0)[:, None]).sum(axis=0)

    def force(p, V, b):
        lp = np.log(p)
        if not PD.peer:
            return lp + V[:, None]
        h = 2 * lp + V[:, None]
        for bk, Pk in zip(b, PD.P):
            h -= bk * (lp @ Pk.T + (p @ Pk.T) / p)
        return h

    hq = force(f, Vq, bq)
    hs = force(g, Vs, bs)
    wq = PD.A[:, None] / f.shape[1]
    ws = PD.B[:, None] / g.shape[1]
    df = -f * (hq - np.sum(wq * f * hq))
    dg = -g * (hs - np.sum(ws * g * hs))
    Dq = profile_D(f, PD) if PD.peer else np.zeros(len(bq))
    Ds = profile_D(g, PD, PD.B) if PD.peer else np.zeros(len(bs))
    rq = Dq + PD.Tq * np.log(bq / PD.pik)
    rs = Ds + PD.Ts * np.log(bs / PD.pik)
    dbq = -bq * (rq - bq @ rq)
    dbs = -bs * (rs - bs @ rs)
    return (df, dg, dbq, dbs)


def class_rows(b, d, paths):
    """Fine rows beta_ij = b_k / n_k for j in class k of receiver i."""
    N = 2 ** d
    beta = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                k = lca_class(paths, i, j)
                beta[i, j] = b[k] / 2 ** (d - k - 1)
    return beta


def random_profile(rng, R: int, d: int, spread: float = 0.6) -> np.ndarray:
    f = np.exp(spread * rng.normal(size=(R, 2 ** d)))
    return f


def normalize_profile(f, A):
    return f / np.sum(A[:, None] * f / f.shape[1])


def setup(d: int, R: int, W: int, rng, Tq=0.7, Ts=1.3, realization="coin", Mc=8, J=0.8, peer=True):
    """A random coupled instance: base weights, rho, orientations/phases, prior rows."""
    N, M, paths = tree(d)
    A = rng.dirichlet(np.ones(R) * 3)
    B = rng.dirichlet(np.ones(W) * 3)
    logrho = J * rng.normal(size=(R, W))
    # normalize rho against A x B
    logrho -= np.log(np.sum(A[:, None] * B[None, :] * np.exp(logrho)))
    if realization == "coin":
        oq = rng.choice([-1.0, 1.0], size=(R, M))
        os_ = rng.choice([-1.0, 1.0], size=(W, M))
        spq = make_space(d, A, "coin", orient=oq)
        sps = make_space(d, B, "coin", orient=os_)
    else:
        aq = rng.integers(0, Mc, size=(R, M)).astype(float)
        as_ = rng.integers(0, Mc, size=(W, M)).astype(float)
        spq = make_space(d, A, "circle", Mc=Mc, alpha=aq)
        sps = make_space(d, B, "circle", Mc=Mc, alpha=as_)
    pi_row = np.full((N, N), 1.0 / (N - 1))
    np.fill_diagonal(pi_row, 0.0)
    edges = ~np.eye(N, dtype=bool)
    FD = FineData(spq, sps, logrho, Tq, Ts, pi_row, edges, peer)
    pik = np.array([2 ** (d - k - 1) / (N - 1) for k in range(d)])
    PD = ProfileData(d, A, B, logrho, Tq, Ts, pik, [P_matrix(k, d) for k in range(d)], peer)
    return N, M, paths, FD, PD
