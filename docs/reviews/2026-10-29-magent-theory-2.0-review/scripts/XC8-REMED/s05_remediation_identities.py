r"""Independent numerical checks of remediation-added results outside chapters 3, 5, 6, 8, 11, 12.

(a) prop:int-reciprocal (02_interactions.tex:192): Omega_ji = Omega_ij^{-1}, transformation law,
    and the Laplace example Omega_12 = 5^{-1/2}[[2,1],[-1,2]], N_2 = [[1,1],[1,2]].
(b) prop:int-loop-defect (02_interactions.tex:219): invariance of
    Q = 1/2 Tr(Sigma (I-W)^T Sigma^{-1} (I-W)) under W -> gWg^{-1}, Sigma -> g Sigma g^T; values 1/2, 3/2, 2.
(c) geo:minimal-contraction (02_geometry.tex:405): h_min - hbar_min = Delta(r,r) + kbar (A^F - Abar^F)^2
    for planar unit-covariance Gaussian means under SO(2), with a fixed isotropic noise channel on one agent.
(d) prop:boundary-contrast-decay (10_boundary_closure.tex:277): spectrum of 3 - 2 sum_k pi_k P_k on mean-zero
    bit profiles equals 3 + 2/(N-1); Walsh rates lambda_m for random rows.
(e) geo:det-rank (determinant_example.tex:61): rank of dM restricted to H for random triples (<= 3) and for the
    four-agent tuple (= 4), orbit dimensions.
(f) cor:public-phase-unique witness numbers (09_public_compression.tex:225): intercepts and velocities.
"""
import itertools

import numpy as np
from scipy.linalg import sqrtm

rng = np.random.default_rng(29)
out = []


def psqrt_inv(N):
    S = sqrtm(N)
    return np.linalg.inv(S)


# (a)
T12 = np.array([[1.0, 1.0], [0.0, 1.0]])
T21 = np.array([[1.0, 0.0], [1.0, 1.0]])
N2 = T21 @ T12
O12 = T12 @ psqrt_inv(N2)
N1 = T12 @ T21
O21 = T21 @ psqrt_inv(N1)
out.append("(a) N2=%s ; Omega12*sqrt5=%s ; |Omega12 Omega21 - I|=%.2e ; Omega12 orthogonal err=%.2e" % (
    N2.tolist(), np.round(O12 * np.sqrt(5), 12).tolist(), np.abs(O12 @ O21 - np.eye(2)).max(),
    np.abs(O12.T @ O12 - np.eye(2)).max()))
worst_inv = worst_tr = 0.0
for _ in range(200):
    A = np.eye(3) + 0.4 * rng.standard_normal((3, 3))
    B = np.eye(3) + 0.4 * rng.standard_normal((3, 3))
    Nj = B @ A
    ev = np.linalg.eigvals(Nj)
    if np.any((np.abs(ev.imag) < 1e-9) & (ev.real <= 0)):
        continue
    Oij = A @ psqrt_inv(Nj)
    Oji = B @ psqrt_inv(A @ B)
    worst_inv = max(worst_inv, np.abs(Oji @ Oij - np.eye(3)).max())
    gi = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    gj = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    A2, B2 = gi @ A @ np.linalg.inv(gj), gj @ B @ np.linalg.inv(gi)
    O2 = A2 @ psqrt_inv(B2 @ A2)
    worst_tr = max(worst_tr, np.abs(O2 - gi @ Oij @ np.linalg.inv(gj)).max())
out.append("    random GL(3): max |Omega_ji Omega_ij - I| = %.2e ; max transformation-law residual = %.2e" % (worst_inv, worst_tr))


# (b)
def Q(W, S):
    I = np.eye(W.shape[0])
    return 0.5 * np.trace(S @ (I - W).T @ np.linalg.inv(S) @ (I - W))


Ws = [np.array([[1.0, 1.0], [0.0, 1.0]]), T12 @ T21, np.array([[0.0, -1.0], [1.0, 0.0]])]
vals = [Q(W, np.eye(2)) for W in Ws]
worst = 0.0
for _ in range(200):
    W = rng.standard_normal((3, 3))
    L = rng.standard_normal((3, 3))
    S = L @ L.T + 0.5 * np.eye(3)
    g = np.eye(3) + 0.5 * rng.standard_normal((3, 3))
    gi = np.linalg.inv(g)
    worst = max(worst, abs(Q(g @ W @ gi, g @ S @ g.T) - Q(W, S)) / max(1.0, Q(W, S)))
out.append("(b) Q(unipotent)=%.6f Q(eq:int-nongaussian-loop)=%.6f Q(quarter turn)=%.6f ; GL(3) invariance rel. err %.2e" % (
    vals[0], vals[1], vals[2], worst))

# (c) SO(2) on planar means, unit covariance; fixed isotropic noise on agent 0's sample (variance s2).
Rg = np.array([[0.0, -1.0], [1.0, 0.0]])
worst = 0.0
for _ in range(300):
    n = 3
    mu = rng.standard_normal((n, 2))
    dmu = rng.standard_normal((n, 2))       # d mu_i (X) for one contextual direction X
    s2 = rng.uniform(0.1, 3.0)
    wfine = np.ones(n)
    wcoarse = np.ones(n)
    wcoarse[0] = 1.0 / (1.0 + s2)            # Fisher metric of the mean after adding N(0, s2 I)
    J = np.array([Rg @ m for m in mu])       # J xi = xi * (R mu_i)_i
    def solve(wts):
        k = np.sum(wts * np.sum(J * J, axis=1))
        a = -np.sum(wts * np.sum(dmu * J, axis=1)) / k
        r = dmu + a * J
        return a, r, k, np.sum(wts * np.sum(r * r, axis=1))
    aF, r, k, hmin = solve(wfine)
    aC, rC, kC, hbar = solve(wcoarse)
    Delta = np.sum((wfine - wcoarse) * np.sum(r * r, axis=1))
    rhs = Delta + kC * (aF - aC) ** 2
    worst = max(worst, abs((hmin - hbar) - rhs))
# manuscript example: two 1-D location laws, J xi=(xi,xi), dmu=(1,-1), noise on first: 2 - 4/3 = 1/2 + 1/6
out.append("(c) max |h_min - hbar_min - Delta(r,r) - kbar(A-Abar)^2| over 300 random cases = %.2e" % worst)

# (d) bit profiles of depth d: P_k keeps first k bits, reverses bit k+1, averages the rest.
for d in (2, 3, 4):
    bits = list(itertools.product([-1, 1], repeat=d))
    idx = {b: i for i, b in enumerate(bits)}
    M = len(bits)
    Ps = []
    for k in range(d):
        P = np.zeros((M, M))
        for b in bits:
            tails = list(itertools.product([-1, 1], repeat=d - k - 1))
            for t in tails:
                c = b[:k] + (-b[k],) + t
                P[idx[b], idx[c]] += 1.0 / len(tails)
        Ps.append(P)
    Nl = 2 ** d
    pis = [2 ** (d - k - 1) / (Nl - 1) for k in range(d)]
    Lop = 3 * np.eye(M) - 2 * sum(p * P for p, P in zip(pis, Ps))
    ones = np.ones(M) / np.sqrt(M)
    Pz = np.eye(M) - np.outer(ones, ones)
    ev = np.linalg.eigvalsh(Pz @ Lop @ Pz)
    ev = np.sort(ev)[1:]  # drop the constant mode (eigenvalue 0 after projection)
    b = rng.dirichlet(np.ones(d))
    Lb = 3 * np.eye(M) - 2 * sum(bb * P for bb, P in zip(b, Ps))
    evb = np.sort(np.linalg.eigvals(Pz @ Lb @ Pz).real)[1:]
    pred = sorted(3 + 2 * b[m - 1] - 2 * sum(b[m:]) for m in range(1, d + 1) for _ in range(2 ** (m - 1)))
    out.append("(d) d=%d: class-prior rates min/max %.10f/%.10f vs 3+2/(N-1)=%.10f ; random rows max |rate-pred|=%.2e, min rate %.4f" % (
        d, ev.min(), ev.max(), 3 + 2 / (Nl - 1), np.abs(np.array(evb) - np.array(pred)).max(), min(evb)))


# (e) proper complex Gaussian targets, collective metric sum_i w_i Tr(M_i^{-1} h_i M_i^{-1} k_i)
def herm_basis():
    E = []
    E.append(np.array([[1, 0], [0, 0]], dtype=complex))
    E.append(np.array([[0, 0], [0, 1]], dtype=complex))
    E.append(np.array([[0, 1], [1, 0]], dtype=complex))
    E.append(np.array([[0, -1j], [1j, 0]], dtype=complex))
    return E


HB = herm_basis()


def gl2_basis():
    B = []
    for i in range(2):
        for j in range(2):
            e = np.zeros((2, 2), dtype=complex)
            e[i, j] = 1
            B.append(e)
            B.append(1j * e)
    return B


GB = gl2_basis()


def rank_info(Ms, w):
    n = len(Ms)
    # orbit directions: (xi M_i + M_i xi^dag)_i in real coordinates of Herm^n
    def herm_coords(H):
        return np.array([H[0, 0].real, H[1, 1].real, H[0, 1].real, H[0, 1].imag])
    orbit = np.array([np.concatenate([herm_coords(x @ M + M @ x.conj().T) for M in Ms]) for x in GB]).T
    odim = np.linalg.matrix_rank(orbit, tol=1e-9)
    # metric matrix on Herm^n in the basis HB^n
    dim = 4 * n
    basis = []
    for i in range(n):
        for E in HB:
            v = [np.zeros((2, 2), dtype=complex) for _ in range(n)]
            v[i] = E
            basis.append(v)
    Gm = np.zeros((dim, dim))
    Minv = [np.linalg.inv(M) for M in Ms]
    for a in range(dim):
        for b2 in range(dim):
            Gm[a, b2] = sum(w[i] * np.trace(Minv[i] @ basis[a][i] @ Minv[i] @ basis[b2][i]).real for i in range(n))
    # horizontal = G-orthogonal complement of orbit tangent; orbit tangent coords in this basis
    def coords(v):
        c = []
        for i in range(n):
            H = v[i]
            c += [H[0, 0].real, H[1, 1].real, 0.5 * (H[0, 1] + H[1, 0]).real, 0.5 * (H[1, 0] - H[0, 1]).imag]
        return np.array(c)
    Ob = np.array([coords([x @ M + M @ x.conj().T for M in Ms]) for x in GB]).T  # dim x 8
    C = Ob.T @ Gm                         # constraints: c^T G h = 0
    _, s, Vt = np.linalg.svd(C)
    r = int(np.sum(s > 1e-9))
    Hbasis = Vt[r:].T                     # dim x (dim - r)
    # dM(h) = sum_i w_i h_i
    dM = np.zeros((4, dim))
    for a in range(dim):
        i, e = divmod(a, 4)
        dM[:, a] = w[i] * herm_coords(basis[a][i])
    rk = np.linalg.matrix_rank(dM @ Hbasis, tol=1e-9)
    return odim, Hbasis.shape[1], rk


def rand_pd():
    X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return X @ X.conj().T + 0.3 * np.eye(2)


ranks3 = [rank_info([rand_pd() for _ in range(3)], np.ones(3)) for _ in range(30)]
four = rank_info([np.eye(2, dtype=complex), 2 * np.eye(2, dtype=complex), np.diag([1, 2]).astype(complex),
                  np.array([[2, 1], [1, 2]], dtype=complex)], np.ones(4))
two = rank_info([np.diag([1.0, 3.0]).astype(complex), 2 * np.diag([1.0, 3.0]).astype(complex)], np.ones(2))
out.append("(e) n=3 random: (orbit dim, dim H, rank dM|H) set = %s ; n=4 example = %s ; n=2 proportional = %s" % (
    sorted(set(ranks3)), four, two))

# (f) three-point witness
lam = np.ones(3) / 3
f1 = 3 * np.array([0.25, 0.25, 0.5])
c = (1 + np.sqrt(5)) / 8
f2 = 3 * np.array([c, c, 1 - 2 * c])
iota = [np.sum(lam * np.log(f)) for f in (f1, f2)]
vel = [np.sum(lam / f) - 1 for f in (f1, f2)]
out.append("(f) intercepts %.15f %.15f ; velocities %.12f %.12f (claimed 1/9=%.12f, (5sqrt5-10)/9=%.12f)" % (
    iota[0], iota[1], vel[0], vel[1], 1 / 9, (5 * np.sqrt(5) - 10) / 9))

text = "\n".join(out)
print(text)
with open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
