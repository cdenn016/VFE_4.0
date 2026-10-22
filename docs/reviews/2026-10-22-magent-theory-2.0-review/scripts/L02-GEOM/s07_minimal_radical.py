r"""L02-GEOM s07: the radical of the collective minimal tensor h^min is integrable and h^min is basic.

Claim (proposed improvement): for the collective minimizing connection A^F, with V = r = D^{A^F} p,
F^{A^F}(Z, X) = 0 whenever Z lies in ker r (free action).  Reason: A^F = -p^* theta_mech,
F^{A^F} = -p^* Omega_mech, and Z in ker r makes dp(Z) = -J A^F(Z) vertical, where Omega vanishes.
By s02 identities (c),(d): ker r is involutive and L_Z h^min = 0 for Z in ker r.
Contrast: a generic non-minimal connection A = A^F + beta has F(Z, .) != 0 on its own radical.
Also checked: h^min(X,X) depends on dp(X) only modulo im J (shape-space quotient norm).

Model: Aff(1) acting diagonally on n=2 normal laws, coords (mu_i, s_i = log sigma_i),
G = sum_i w_i diag(e^{-2 s_i}, 2), J_i(xi) = (xi_d mu_i + xi_t, xi_d), [delta, tau] = tau.
Base R^3 (dim H = 2, so ker r has dimension >= 1 at every point).  Derivatives of A^F by central
differences (step 1e-5), so residuals at the 1e-9 level are expected.
"""
import numpy as np

rng = np.random.default_rng(2)
w = np.array([2 / 3, 5 / 4])
coef = rng.normal(size=(4, 10)) * np.array([1, 1, 1, 1]).reshape(4, 1)
coef[2:] *= 0.3                                   # s_i milder
def feats(c):
    x, y, z = c
    return np.array([1, x, y, z, x * y, y * z, x * z, x * x, y * y, z * z])
def dfeats(c):
    x, y, z = c
    return np.array([[0, 1, 0, 0, y, 0, z, 2 * x, 0, 0],
                     [0, 0, 1, 0, x, z, 0, 0, 2 * y, 0],
                     [0, 0, 0, 1, 0, y, x, 0, 0, 2 * z]]).T   # 10 x 3
def law(c):
    v = coef @ feats(c)                            # mu1, mu2, s1, s2
    return v[[0, 2, 1, 3]]                         # p = (mu1, s1, mu2, s2)
def dlaw(c):
    D = coef @ dfeats(c)                           # 4 x 3
    return D[[0, 2, 1, 3]]
def GJ(pv):
    G = np.diag([w[0] * np.exp(-2 * pv[1]), 2 * w[0], w[1] * np.exp(-2 * pv[3]), 2 * w[1]])
    J = np.array([[pv[0], 1], [1, 0], [pv[2], 1], [1, 0]])
    return G, J
beta_c = rng.normal(size=(2, 3))                   # a fixed non-minimal perturbation
def Aform(c, pert=False):
    pv = law(c); G, J = GJ(pv)
    A = -np.linalg.solve(J.T @ G @ J, J.T @ G @ dlaw(c))   # 2 x 3: (A_d, A_t) along e_a
    if pert:
        A = A + beta_c * (1 + 0.5 * c[0])                  # smooth, non-minimal
    return A
def curvature(c, pert=False, h=1e-5):
    dA = np.zeros((3, 2, 3))                       # dA[a] = d/dc_a of A (2 x 3)
    for a in range(3):
        e = np.zeros(3); e[a] = h
        dA[a] = (Aform(c + e, pert) - Aform(c - e, pert)) / (2 * h)
    A = Aform(c, pert)
    F = np.zeros((3, 3, 2))
    for a in range(3):
        for b in range(3):
            F[a, b] = dA[a][:, b] - dA[b][:, a]
            F[a, b, 1] += A[0, a] * A[1, b] - A[0, b] * A[1, a]
    return F
c0 = np.array([0.25, -1 / 3, 0.4])
for pert in (False, True):
    pv = law(c0); G, J = GJ(pv)
    r = dlaw(c0) + J @ Aform(c0, pert)
    U_, S_, Vt_ = np.linalg.svd(r)
    Kb = Vt_[(S_ > 1e-10).sum():]
    F = curvature(c0, pert)
    label = 'A = A^F (minimal)' if not pert else 'A = A^F + beta (non-minimal)'
    if len(Kb) == 0:
        print('%-30s rank r = %d; radical is zero here (contrast case is the contact example in s02 (e))' % (label, (S_ > 1e-10).sum()))
    for Zv in Kb:
        FZ = np.einsum('a,abk->bk', Zv, F)
        JFZ = np.array([J @ FZ[b] for b in range(3)])
        print('%-30s rank r = %d; max |F(Z,e_b)| = %.2e; max |J F(Z,e_b)| = %.2e; max |F| = %.2e'
              % (label, (S_ > 1e-10).sum(), np.abs(FZ).max(), np.abs(JFZ).max(), np.abs(F).max()))
G, J = GJ(law(c0)); k = J.T @ G @ J
v = rng.normal(size=4); xi = rng.normal(size=2)
def hmin(v):
    a = -np.linalg.solve(k, J.T @ G @ v); rr = v + J @ a
    return rr @ G @ rr
print('h^min(v + J xi) - h^min(v):', hmin(v + J @ xi) - hmin(v), ' (h^min(v) =', hmin(v), ')')
