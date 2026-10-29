r"""D18-LP check 1: lp:weights and lp:field from first principles.

Independent of the closed forms: build each q_i on a periodic grid, form the channelled law
nu_j by numerically marginalizing q_j over phi (copy z, psi; resample phi), the evaluator
G = e^{J z cos phi}/Z_G, the Gibbs-reduced per-agent content objective
  F_i(eta) = -log sum_j pi_ij exp(-KL(q_i || kappa_ij)) - E_{q_i}[H_i z/2 + Lam_i . e_phi],
and take the Fisher natural gradient  eta_dot = -Cov_q(T)^{-1} grad F_i  with T the sufficient
statistic of the coordinates eta=(l,A,C,B) (so T=(z/2, e_phi, z e_phi, e_psi)).
Presentation sector likewise with statistic w/2.  Compare with lp_model.content_rows / V.
"""
import numpy as np
import lp_model as M

rng = np.random.default_rng(14)
N = 64
phi = 2 * np.pi * np.arange(N) / N
psi = 2 * np.pi * np.arange(N) / N
Z_, PHI, PSI = np.meshgrid(np.array([-1.0, 1.0]), phi, psi, indexing="ij")
W = np.full(Z_.shape, 0.5 / N / N)  # uniform sign (1/2 each) x normalized Haar x Haar


def law(p):
    l, A, C, B = p[0], p[1:3], p[3:5], p[5:7]
    lg = (l * Z_ / 2 + (A[0] + Z_ * C[0]) * np.cos(PHI) + (A[1] + Z_ * C[1]) * np.sin(PHI)
          + B[0] * np.cos(PSI) + B[1] * np.sin(PSI))
    q = np.exp(lg)
    return q / np.sum(W * q)


def channel(q):
    """Copy z and psi, resample phi from Haar: average over the phi axis."""
    nu = q.mean(axis=1, keepdims=True)
    return np.broadcast_to(nu, q.shape)


J = 0.37
x = rng.normal(size=(3, 8)) * 0.6
x[:, 5:7] = np.array([[0.8, 0.3], [0.5, 0.9], [0.9, -0.2]])
H = np.array([0.3, -0.5, 0.2])
Lam = rng.normal(size=(3, 2)) * 0.4
G = np.exp(J * Z_ * np.cos(PHI))
G = G / np.sum(W * G)
pri = np.full((3, 3), 0.4)
np.fill_diagonal(pri, 0.2)


def KL(q, r):
    return np.sum(W * q * np.log(q / r))


def F_content(i, p, xx):
    q = law(p)
    D = []
    for j in range(3):
        if j == i:
            D.append(KL(q, G))
        else:
            D.append(KL(q, channel(law(xx[j, :7]))))
    D = np.array(D)
    red = -np.log(np.sum(pri[i] * np.exp(-D)))
    rec = np.sum(W * q * (H[i] * Z_ / 2 + Lam[i, 0] * np.cos(PHI) + Lam[i, 1] * np.sin(PHI)))
    return red - rec, D


def rows_quad(xx):
    beta = np.zeros((3, 3))
    for i in range(3):
        _, D = F_content(i, xx[i, :7], xx)
        w = pri[i] * np.exp(-D)
        beta[i] = w / w.sum()
    return beta


beta_q = rows_quad(x)
beta_c = M.content_rows(x, J)
print("rows: max |beta_quadrature - beta_closed| =", np.abs(beta_q - beta_c).max())

Vc = M.V(x, H, Lam, J)
res = []
h = 1e-5
for i in range(3):
    p0 = x[i, :7].copy()
    g = np.zeros(7)
    for k in range(7):
        e = np.zeros(7); e[k] = h
        g[k] = (F_content(i, p0 + e, x)[0] - F_content(i, p0 - e, x)[0]) / (2 * h)
    q = law(p0)
    T = np.stack([Z_ / 2, np.cos(PHI), np.sin(PHI), Z_ * np.cos(PHI), Z_ * np.sin(PHI), np.cos(PSI), np.sin(PSI)])
    mT = np.array([np.sum(W * q * t) for t in T])
    Cov = np.array([[np.sum(W * q * (T[a] - mT[a]) * (T[b] - mT[b])) for b in range(7)] for a in range(7)])
    eta_dot = -np.linalg.solve(Cov, g)
    res.append(np.abs(eta_dot - Vc[i, :7]).max())
    print(f"agent {i}: natural-gradient field vs lp:field max residual {res[-1]:.2e}; min eig Cov {np.linalg.eigvalsh(Cov).min():.3e}")

# presentation sector
def pres(n):
    return np.array([np.exp(n / 2), np.exp(-n / 2)]) / (2 * np.cosh(n / 2))


def F_pres(i, n, xx):
    s = pres(n)
    D = []
    for j in range(3):
        ref = np.array([0.5, 0.5]) if j == i else pres(xx[j, 7])
        D.append(np.sum(s * np.log(s / ref)))
    return -np.log(np.sum(pri[i] * np.exp(-np.array(D))))


for i in range(3):
    n0 = x[i, 7]
    g = (F_pres(i, n0 + h, x) - F_pres(i, n0 - h, x)) / (2 * h)
    s = pres(n0)
    var = s[0] * s[1] * 4 / 4  # Var(w/2) = (1 - tanh^2(n/2))/4 = s(+)s(-)
    ndot = -g / var
    print(f"agent {i}: presentation natural gradient {ndot:+.10f} vs lp:field {Vc[i,7]:+.10f}  diff {abs(ndot-Vc[i,7]):.2e}")
print("max content residual", max(res))
