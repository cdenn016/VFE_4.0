"""XC1-GAUGE check 3: does the tangent dictionary-learning limit of ch17
(it1:learning, eq it1:limit-mean, it1:complementary) transform covariantly
under a common reference change h in GL(2)?

Truth omega*, codes T_e, anchor Pi_0 = sigma^-2 I (Frobenius) and loop form
W = Frobenius.  Compare omega_inf computed in the new presentation with the
anchor/loop metric left Frobenius in the NEW coordinates, against
h^-1 omega_inf h.  Then repeat with the metric carried along (dressed) and
with a covariance-dressed anchor ||X||_Sigma^2 = Tr(X^T Sigma^-1 X Sigma).
"""
import numpy as np

rng = np.random.default_rng(7)
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

K = 2
def vec(M): return M.reshape(-1)
def unvec(v): return v.reshape(K, K)

def conj_op(h):
    """matrix of X -> h^-1 X h acting on vec(X) (row-major)."""
    hi = np.linalg.inv(h)
    return np.kron(hi, h.T)

def null_basis(T):
    """basis (columns, vec) of {delta in gl(K): T delta = 0}."""
    M = np.kron(T, np.eye(K))  # vec(T delta) = (T kron I) vec(delta) for row-major vec
    u, s, vt = np.linalg.svd(M)
    r = (s > 1e-12).sum()
    return vt[r:].T

def limit_mean(T_list, omega_star, Gram_edge, lam):
    """omega_inf = (I - Pi_N) omega* with Lambda0 = blockdiag(Gram_edge) + lam B^T W B,
    W given by the same Gram on gl(K); two edges forming one two-cycle."""
    nE = len(T_list)
    Z = np.zeros((nE * K * K, 0))
    for e, T in enumerate(T_list):
        Ne = null_basis(T)
        blk = np.zeros((nE * K * K, Ne.shape[1])); blk[e * K * K:(e + 1) * K * K] = Ne
        Z = np.hstack([Z, blk])
    Bmat = np.hstack([np.eye(K * K)] * nE)  # (B omega) = omega_1 + omega_2
    L0 = np.kron(np.eye(nE), Gram_edge) + lam * Bmat.T @ Gram_edge @ Bmat
    S = Z.T @ L0 @ Z
    PiN = Z @ np.linalg.solve(S, Z.T @ L0)
    return (np.eye(nE * K * K) - PiN) @ omega_star

sigma2 = 1.0; lam = 3.0
T1 = np.array([[1.0, 0.4]]); T0 = np.array([[0.2, 1.0]])
w1 = rng.standard_normal((K, K)); w0 = rng.standard_normal((K, K))
omega_star = np.concatenate([vec(w1), vec(w0)])
Frob = np.eye(K * K) / sigma2
om_inf = limit_mean([T1, T0], omega_star, Frob, lam)

h = np.array([[1.0, 0.9], [0.0, 1.6]])     # nonorthogonal reference change
Ch = conj_op(h)
T1p, T0p = T1 @ h, T0 @ h
omega_star_p = np.concatenate([Ch @ vec(w1), Ch @ vec(w0)])
# (i) Frobenius anchor and loop form in the new coordinates (not carried)
om_inf_p = limit_mean([T1p, T0p], omega_star_p, Frob, lam)
target = np.concatenate([Ch @ om_inf[:4], Ch @ om_inf[4:]])
p("(i) fixed Frobenius anchor/loop form: |omega_inf' - h^-1 omega_inf h| =", np.abs(om_inf_p - target).max())
# (ii) carried metric: Gram' = Ch^{-T} Gram Ch^{-1}
Chi = np.linalg.inv(Ch)
Gram_c = Chi.T @ Frob @ Chi
om_inf_c = limit_mean([T1p, T0p], omega_star_p, Gram_c, lam)
p("(ii) metric carried with the data:      |omega_inf' - h^-1 omega_inf h| =", np.abs(om_inf_c - target).max())
# (iii) orthogonal h with fixed Frobenius
th = 0.7; ho = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Co = conj_op(ho)
om_o = limit_mean([T1 @ ho, T0 @ ho], np.concatenate([Co @ vec(w1), Co @ vec(w0)]), Frob, lam)
p("(iii) orthogonal h, fixed Frobenius:   residual =",
  np.abs(om_o - np.concatenate([Co @ om_inf[:4], Co @ om_inf[4:]])).max())
# (iv) covariance-dressed anchor: ||X||_Sigma^2 = Tr(X^T Sigma^-1 X Sigma) = vec(X)^T (Sigma^-1 kron Sigma) vec(X)
Sigma = np.array([[1.3, 0.4], [0.4, 0.8]])
def dressed_gram(Sig):
    return np.kron(np.linalg.inv(Sig), Sig) / sigma2
om_d = limit_mean([T1, T0], omega_star, dressed_gram(Sigma), lam)
hi = np.linalg.inv(h)
Sig_p = hi @ Sigma @ hi.T
om_dp = limit_mean([T1p, T0p], omega_star_p, dressed_gram(Sig_p), lam)
p("(iv) covariance-dressed anchor:        residual =",
  np.abs(om_dp - np.concatenate([Ch @ om_d[:4], Ch @ om_d[4:]])).max())

# detectable dissonance D(omega*) = ||Q B omega*||_W^2 with Frobenius W, before and after h
def detectable(T_list, omega_star, Gram):
    Z = []
    for e, T in enumerate(T_list):
        Ne = null_basis(T); blk = np.zeros((2 * K * K, Ne.shape[1])); blk[e * K * K:(e + 1) * K * K] = Ne; Z.append(blk)
    Z = np.hstack(Z)
    Bmat = np.hstack([np.eye(K * K)] * 2)
    BZ = Bmat @ Z
    # W-orthogonal projection of B omega onto complement of range(BZ)
    G = Gram
    P = BZ @ np.linalg.pinv(BZ.T @ G @ BZ) @ BZ.T @ G
    r = (np.eye(K * K) - P) @ (Bmat @ omega_star)
    return r @ G @ r
p("(v) detectable dissonance, Frobenius W: before", detectable([T1, T0], omega_star, np.eye(4)),
  " after fixed-Frobenius reference change", detectable([T1p, T0p], omega_star_p, np.eye(4)))
p("    zero set check: D=0 for a truth in B(N)? before",
  detectable([T1, T0], np.concatenate([null_basis(T1)[:, 0], null_basis(T0)[:, 1]]), np.eye(4)))

with open(__file__.replace(".py", "_output.txt"), "w") as f:
    f.write("\n".join(out) + "\n")

# ---------------------------------------------------------------- (vi) K=3, rank-2 codes sharing one row direction (it1:pair)
K3 = 3
def nullb(T, K):
    M = np.kron(T, np.eye(K)); u, s, vt = np.linalg.svd(M); r = (s > 1e-12).sum(); return vt[r:].T
def D_pair(T1, T0, w01, w10, K, Gram):
    Z = []
    for e, T in enumerate((T1, T0)):
        Ne = nullb(T, K); blk = np.zeros((2 * K * K, Ne.shape[1])); blk[e * K * K:(e + 1) * K * K] = Ne; Z.append(blk)
    Z = np.hstack(Z); Bm = np.hstack([np.eye(K * K)] * 2); BZ = Bm @ Z
    P = BZ @ np.linalg.pinv(BZ.T @ Gram @ BZ) @ BZ.T @ Gram
    r = (np.eye(K * K) - P) @ (Bm @ np.concatenate([w01.reshape(-1), w10.reshape(-1)]))
    return r @ Gram @ r
T1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]); T0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
w01 = rng.standard_normal((3, 3)); w10 = rng.standard_normal((3, 3))
h3 = np.array([[1.0, 0.8, 0.0], [0.0, 1.0, 0.5], [0.3, 0.0, 1.2]]); h3i = np.linalg.inv(h3)
D0 = D_pair(T1, T0, w01, w10, 3, np.eye(9))
D1 = D_pair(T1 @ h3, T0 @ h3, h3i @ w01 @ h3, h3i @ w10 @ h3, 3, np.eye(9))
p("(vi) it1:pair detectable dissonance with Frobenius W: original", D0, " after nonorthogonal h", D1)
# closed form of it1:pair: P_cap projects onto R0 cap R1 = span(e1): D = ||e1 e1^T (w01+w10)||_F^2
p("     closed form ||P_cap(w01+w10)||^2 =", np.sum((np.outer([1, 0, 0], [1, 0, 0]) @ (w01 + w10))**2))
with open(__file__.replace(".py", "_output.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
