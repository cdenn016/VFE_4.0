r"""Shared linear-algebra helpers for the D23-LEARN checks of chapters/17_communication_links.tex.

Conventions (independent of the manuscript's own item1_checks/checks.py):
  * An edge is a pair (j, i) meaning j -> i: receiver i, sender j.  Edge e carries the
    receiver's generator omega_e in gl(K,R), flattened row-major, so that the Frobenius
    inner product on X = gl(K)^E is the Euclidean inner product of the flattened vector.
  * For a K x K matrix delta, row-major vec satisfies vec(A delta C) = (A kron C^T) vec(delta).
  * N_e = {delta : T_e delta = 0} = null(T_e kron I_K).
  * (B omega)_gamma = sum_{e in gamma} omega_e (an edge counted with multiplicity).
  * Grounded tangent Fisher form: F(delta,delta) = Tr(R^{-1} T delta M delta^T T^T)
    = vec(delta)^T ((T^T R^{-1} T) kron M) vec(delta).
  * Dressed loop norm ||X||_Sigma^2 = Tr(X^T Sigma^{-1} X Sigma) = vec(X)^T (Sigma^{-1} kron Sigma) vec(X).
"""
from __future__ import annotations

import numpy as np

TOL = 1e-9


def null_space(A: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Orthonormal basis (columns) of ker A."""
    if A.shape[0] == 0:
        return np.eye(A.shape[1])
    u, s, vt = np.linalg.svd(A)
    rank = int((s > tol * max(1.0, s.max() if s.size else 1.0)).sum())
    return vt[rank:].T.copy()


def orth(A: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Orthonormal basis (columns) of range A."""
    if A.size == 0:
        return np.zeros((A.shape[0], 0))
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > tol * max(1.0, s.max())).sum())
    return u[:, :rank].copy()


def rank(A: np.ndarray, tol: float = TOL) -> int:
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol * max(1.0, s.max())).sum())


def spd(n: int, rng: np.random.Generator, cond: float = 5.0) -> np.ndarray:
    """Random symmetric positive-definite matrix with eigenvalues in [1, cond]."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.linspace(1.0, cond, n)
    rng.shuffle(ev)
    return (q * ev) @ q.T


def blockdiag(blocks: list[np.ndarray]) -> np.ndarray:
    n = sum(b.shape[0] for b in blocks)
    m = sum(b.shape[1] for b in blocks)
    out = np.zeros((n, m))
    r = c = 0
    for b in blocks:
        out[r:r + b.shape[0], c:c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    return out


def invisible_basis(codes: list[np.ndarray], K: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal bases Z of N = (+)_e N_e and Y of its Frobenius complement in X."""
    m = K * K
    Ns = [null_space(np.kron(T, np.eye(K))) for T in codes]
    blocksZ = []
    E = len(codes)
    Z = np.zeros((E * m, sum(n.shape[1] for n in Ns)))
    col = 0
    for e, n in enumerate(Ns):
        Z[e * m:(e + 1) * m, col:col + n.shape[1]] = n
        col += n.shape[1]
    Y = null_space(Z.T) if Z.shape[1] > 0 else np.eye(E * m)
    return Z, Y


def loop_matrix(loops: list[list[int]], nE: int, K: int) -> np.ndarray:
    """Linearized holonomy B: X -> g^L, (B omega)_gamma = sum_{e in gamma} omega_e."""
    m = K * K
    B = np.zeros((len(loops) * m, nE * m))
    for g, loop in enumerate(loops):
        for e in loop:
            B[g * m:(g + 1) * m, e * m:(e + 1) * m] += np.eye(m)
    return B


def fisher_edge(T: np.ndarray, R: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-exchange tangent Fisher form ((T^T R^{-1} T) kron M) on vec_row(delta)."""
    return np.kron(T.T @ np.linalg.solve(R, T), M)


def coboundary_gl(edges: list[tuple[int, int]], nV: int, K: int) -> np.ndarray:
    """delta^0: gl(K)^V -> gl(K)^E, (delta a)_e = a_i - a_j for e: j -> i."""
    m = K * K
    D = np.zeros((len(edges) * m, nV * m))
    for e, (j, i) in enumerate(edges):
        D[e * m:(e + 1) * m, i * m:(i + 1) * m] += np.eye(m)
        D[e * m:(e + 1) * m, j * m:(j + 1) * m] -= np.eye(m)
    return D


def sheaf_coboundary(edges: list[tuple[int, int]], codes: list[np.ndarray], nV: int, K: int) -> np.ndarray:
    """Code-sheaf coboundary on (R^K)^V: (delta u)_e = T_e (u_i - u_j)."""
    rows = []
    for e, (j, i) in enumerate(edges):
        T = codes[e]
        row = np.zeros((T.shape[0], nV * K))
        row[:, i * K:(i + 1) * K] += T
        row[:, j * K:(j + 1) * K] -= T
        rows.append(row)
    return np.vstack(rows)


def dim_H0(edges, codes, nV, K) -> int:
    return nV * K - rank(sheaf_coboundary(edges, codes, nV, K))


def gaussian_condition(Sigma: np.ndarray, Y: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Law N(0, Sigma) conditioned on Y^T w = c (covariance-form Gaussian conditioning)."""
    SY = Sigma @ Y
    G = np.linalg.solve(Y.T @ SY, np.eye(Y.shape[1]))
    mean = SY @ G @ c
    cov = Sigma - SY @ G @ SY.T
    return mean, cov


def detectable_lstsq(omega: np.ndarray, B: np.ndarray, W: np.ndarray, Z: np.ndarray) -> float:
    """D(omega) = min_{nu in N} ||B(omega + nu)||_W^2 by direct weighted least squares."""
    Wh = np.linalg.cholesky(W).T  # W = Wh^T Wh
    A = Wh @ B @ Z
    b = -Wh @ B @ omega
    if A.shape[1] == 0:
        return float(np.sum((Wh @ B @ omega) ** 2))
    z, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = Wh @ B @ (omega + Z @ z)
    return float(r @ r)


def W_projection_complement(B: np.ndarray, W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Q: W-orthogonal projection of g^L onto the W-orthogonal complement of B(N)."""
    BN = orth(B @ Z) if Z.shape[1] else np.zeros((B.shape[0], 0))
    if BN.shape[1] == 0:
        return np.eye(B.shape[0])
    P = BN @ np.linalg.solve(BN.T @ W @ BN, BN.T @ W)  # W-orthogonal projector onto B(N)
    return np.eye(B.shape[0]) - P
