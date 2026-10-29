"""D03-GEOM s09: can fewer than four proper complex Gaussian targets give an aggregate
jet e = dM_p r that is a bundle isomorphism on an open four-dimensional base?

geo:det-rank bounds rank(dM|_H) <= 3 only on the maximal-orbit stratum of triples and
remarks that off it H grows (two agents (M, 2M): dM|_H surjective).  Here e is computed
for actual sections:
 (1) two agents, section through the proportional pair (I, 2I) with transverse jet:
     rank e at the crossing point versus nearby points (discontinuity of e);
 (2) two agents kept on the proportional stratum (M(c), lambda(c) M(c)): rank e;
 (3) three agents kept on the jointly diagonalizable stratum (orbit dimension 6),
     M_i = g(c) D_i(c) g(c)^dagger with D_i diagonal: rank e;
 (4) three agents, generic section: rank e (maximal stratum).
"""
import numpy as np
rng = np.random.default_rng(3)
basisH = [np.array([[1, 0], [0, 0]], complex), np.array([[0, 0], [0, 1]], complex),
          np.array([[0, 1], [1, 0]], complex), np.array([[0, 1j], [-1j, 0]])]
def hv(Hm): return np.array([Hm[0, 0].real, Hm[1, 1].real, Hm[0, 1].real, Hm[0, 1].imag])
gl = []
for i in range(2):
    for j in range(2):
        Eij = np.zeros((2, 2), complex); Eij[i, j] = 1; gl += [Eij, 1j*Eij]
def e_matrix(Ms, dMs, w=None, tol=1e-9):
    """Ms: list of n covariances; dMs[mu][i]: derivative of M_i along base coordinate mu.
    Returns the 4x4 matrix of e (columns e(d_mu) in hv coordinates) and rank J."""
    n = len(Ms); w = np.ones(n) if w is None else w
    Jm = np.column_stack([np.concatenate([hv(xi@M + M@xi.conj().T) for M in Ms]) for xi in gl])
    Gm = np.zeros((4*n, 4*n))
    for i, M in enumerate(Ms):
        Mi = np.linalg.inv(M)
        for a in range(4):
            for b in range(4):
                Gm[4*i + a, 4*i + b] = w[i]*np.trace(Mi@basisH[a]@Mi@basisH[b]).real
    k = Jm.T @ Gm @ Jm
    cols = []
    for dm in dMs:
        v = np.concatenate([hv(d) for d in dm])
        a = -np.linalg.pinv(k, rcond=tol) @ (Jm.T @ Gm @ v)
        r = v + Jm @ a
        cols.append(sum(w[i]*r[4*i:4*i + 4] for i in range(n)))
    return np.column_stack(cols), np.linalg.matrix_rank(Jm, tol=tol)
def rank_sv(E): s = np.linalg.svd(E, compute_uv=False); return int(np.sum(s > 1e-8*max(1, s[0]))), s

# (1) two agents through (I, 2I): M1(c) = I + c-terms, M2(c) = 2I + sum c_mu S_mu
S = basisH
def sec1(cv):
    M1 = np.eye(2) + 0.3*cv[0]*cv[1]*basisH[2] + 0.2*cv[2]**2*basisH[0]
    M2 = 2*np.eye(2) + sum(cv[m]*S[m] for m in range(4))
    return [M1.astype(complex), M2.astype(complex)]
def jet(sec, cv, h=1e-6):
    out = []
    for m in range(4):
        e = np.zeros(4); e[m] = h
        Mp, Mm = sec(cv + e), sec(cv - e)
        out.append([(a - b)/(2*h) for a, b in zip(Mp, Mm)])
    return out
for cv in [np.zeros(4), np.array([1e-3, 0, 0, 0]), np.array([0, 2e-3, -1e-3, 5e-4]), np.array([1e-5, 1e-5, 1e-5, 1e-5])]:
    E, rJ = e_matrix(sec1(cv), jet(sec1, cv))
    rk, s = rank_sv(E)
    print(f"(1) c={cv}: rank J={rJ}, rank e={rk}, singular values {np.array2string(s, precision=4)}")

# (2) two agents on the proportional stratum
def g_of(cv): return np.array([[1 + 0.3*cv[0], 0.2*cv[1] + 0.1j*cv[2]], [0.1*cv[3] - 0.2j*cv[0], 1.2 + 0.25*cv[2]*cv[1]]])
def sec2(cv):
    g = g_of(cv); M = g @ g.conj().T; lam = 2 + 0.5*np.sin(cv[0]) + 0.3*cv[3]
    return [M, lam*M]
cv = rng.normal(size=4)*0.3
E, rJ = e_matrix(sec2(cv), jet(sec2, cv)); rk, s = rank_sv(E)
print(f"(2) proportional stratum: rank J={rJ}, rank e={rk}, singular values {np.array2string(s, precision=4)}")

# (3) three agents on the jointly diagonalizable stratum
def sec3(cv):
    g = g_of(cv)
    D = [np.diag([1 + 0.2*cv[0], 1.5 - 0.1*cv[1]]), np.diag([2 + 0.3*cv[2], 0.7 + 0.2*cv[3]**2]), np.diag([0.9 + 0.1*cv[1]*cv[2], 2.4 + 0.3*cv[0]])]
    return [g @ Dk @ g.conj().T for Dk in D]
for _ in range(3):
    cv = rng.normal(size=4)*0.3
    E, rJ = e_matrix(sec3(cv), jet(sec3, cv)); rk, s = rank_sv(E)
    print(f"(3) jointly diagonalizable stratum: rank J={rJ}, rank e={rk}, singular values {np.array2string(s, precision=4)}")

# (4) generic three-agent section
def sec4(cv):
    out = []
    for i in range(3):
        A = np.array([[1 + 0.3*cv[i] + i, 0.2*cv[(i+1) % 4] + 0.4j], [0.1*cv[3] - 0.2j*cv[0] + 0.3*i, 1.2 + 0.25*cv[2]*cv[1] - 0.2*i]])
        out.append(A @ A.conj().T)
    return out
for _ in range(3):
    cv = rng.normal(size=4)*0.3
    E, rJ = e_matrix(sec4(cv), jet(sec4, cv)); rk, s = rank_sv(E)
    print(f"(4) generic triple: rank J={rJ}, rank e={rk}, singular values {np.array2string(s, precision=4)}")

# tolerance check near the crossing: the extra orbit directions have locked-inertia
# eigenvalues of order |c|^2; a pseudoinverse cutoff above them mimics the crossing point.
for cv in [np.array([1e-5, 1e-5, 1e-5, 1e-5]), np.array([1e-4, -2e-4, 1e-4, 3e-4])]:
    for tol in (1e-9, 1e-15):
        E, rJ = e_matrix(sec1(cv), jet(sec1, cv, h=1e-8), tol=tol)
        rk, s = rank_sv(E)
        print(f"(1') c={cv}, pinv cutoff {tol:.0e}: rank e={rk}, singular values {np.array2string(s, precision=3)}")
