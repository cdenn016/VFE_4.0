"""L8 checks for ga (decision metatheory, quotient Gaussians) and g11 prop:glink-law-level-energy."""
import numpy as np
from numpy.linalg import inv, slogdet, eigvalsh, eigh, det, pinv
from scipy.linalg import expm, sqrtm, null_space
from scipy.integrate import dblquad
from common import rng, out, rep, note, spd, gl_plus, kl

# prop:cg-quotient-covariance-polynomial: weighted MP inverse with non-identity G_ref
n = 4; Bq = rng.standard_normal((n, 2)); Q = Bq @ Bq.T; G = spd(n, 30.)
Gh = sqrtm(G).real; Gmh = inv(Gh)
Qh = Gmh @ Q @ Gmh; w, V = eigh(Qh); wp = np.where(w > 1e-10, 1 / np.where(w > 1e-10, w, 1), 0)
SQ = Gmh @ V @ np.diag(wp) @ V.T @ Gmh
rep("ga QSQ = Q", np.abs(Q @ SQ @ Q - Q).max(), np.abs(Q).max())
rep("ga SQS = S", np.abs(SQ @ Q @ SQ - SQ).max(), np.abs(SQ).max())
rep("ga G S Q = Q S G", np.abs(G @ SQ @ Q - Q @ SQ @ G).max(), np.abs(G).max())
Pi = SQ @ Q
note(f"ga Pi_Q: idempotent {np.abs(Pi@Pi-Pi).max():.2e}, kernel ker Q {np.abs(Pi @ null_space(Q)).max():.2e}, G-self-adjoint {np.abs(G@Pi - Pi.T@G).max():.2e}")
note(f"ga Sigma_Q differs from Euclidean pinv(Q) when G != I: {np.abs(SQ - pinv(Q)).max():.3e}")
# Z_Q numerically for n=2, rank 1, G nonidentity: integrate over slice S = (ker Q)^{perp_G}
Q2 = np.array([[1., 0], [0, 0]]); G2 = np.array([[2., 0.5], [0.5, 1.]]); h2 = np.array([0.7, 0.])
k = np.array([0, 1.]); s_dir = inv(G2) @ np.array([1., 0]); s_dir = s_dir / np.sqrt(s_dir @ G2 @ s_dir)  # G-unit vector G-orthogonal to ker
assert abs(k @ G2 @ s_dir) < 1e-14
from scipy.integrate import quad
Zn = quad(lambda u: np.exp(-(0.5 * (u * s_dir) @ Q2 @ (u * s_dir) - h2 @ (u * s_dir))), -50, 50)[0]
Gh2 = sqrtm(G2).real; Qh2 = inv(Gh2) @ Q2 @ inv(Gh2); dp = np.prod([e for e in eigvalsh(Qh2) if e > 1e-12])
w2, V2 = eigh(Qh2); S2 = inv(Gh2) @ V2 @ np.diag([0 if e < 1e-12 else 1 / e for e in w2]) @ V2.T @ inv(Gh2)
Zf = np.sqrt(2 * np.pi) * dp ** -0.5 * np.exp(0.5 * h2 @ S2 @ h2)
rep("ga Z_Q formula vs quadrature on G-slice", abs(Zn - Zf), Zf)
# gauge counterexample
g = np.array([[1, 1], [0, 1.]]); Qp = inv(g).T @ np.diag([1., 0]) @ inv(g)
note(f"ga counterexample Q' = {Qp.tolist()}, pinv = {(pinv(Qp)*4).tolist()}/4, pushed cov {(g @ np.diag([1.,0]) @ g.T).tolist()}")
# chain example
Qf = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1.]]); Sf = pinv(Qf)
note(f"ga chain: 9*Sigma_f = {(9*Sf).round(12).tolist()}; Q_f Sigma_f = I - 11^T/3 residual {np.abs(Qf@Sf - (np.eye(3)-np.ones((3,3))/3)).max():.1e}")
Fm = np.array([[1, 0, 0], [0, 0, 1.]]); Qc = 0.5 * np.array([[1, -1], [-1, 1.]]); Pc = pinv(Qc) @ Qc
note(f"ga chain: pinv(Qc)=Qc? {np.abs(pinv(Qc)-Qc).max():.1e}; Pi_c F Sf F^T Pi_c - Sigma_c = {np.abs(Pc@Fm@Sf@Fm.T@Pc.T - pinv(Qc)).max():.1e}; descent Qc F (I - Sf Qf) = {np.abs(Qc@Fm@(np.eye(3)-Sf@Qf)).max():.1e}")
# rank change: E exp(-X2^2)
for t in [0.5, 0.1]:
    val = quad(lambda x: np.exp(-x * x) * t / np.sqrt(2 * np.pi) * np.exp(-0.5 * t * t * x * x), -np.inf, np.inf)[0]
    rep(f"ga E exp(-X2^2) at t={t}", abs(val - abs(t) / np.sqrt(t * t + 2)), val)

# g11 prop:glink-law-level-energy
for scale in [0.3, 3.0]:
    K = 3; om = scale * rng.standard_normal((K, K)); mi, mj = rng.standard_normal(K), rng.standard_normal(K); Si, Sj = spd(K, 1e3), spd(K, 1e3)
    E = expm(om); Ei = expm(-om)
    direct = kl(mi, Si, E @ mj, E @ Sj @ E.T)
    formula = 0.5 * np.trace(inv(Sj) @ Ei @ Si @ Ei.T) + 0.5 * (Ei @ mi - mj) @ inv(Sj) @ (Ei @ mi - mj) - K / 2 + 0.5 * (slogdet(Sj)[1] - slogdet(Si)[1]) + np.trace(om)
    rep(f"g11 transported KL closed form (|omega|~{scale}, cond Sigma 1e3)", abs(direct - formula), abs(direct))
    # gradient via FD vs I - L(-omega^T, A_e)
    Ae = inv(Sj) @ (Ei @ (Si + np.outer(mi, mi)) - np.outer(mj, mi))
    def Lfre(B, T, nq=60):
        xs, ws = np.polynomial.legendre.leggauss(nq); xs = 0.5 * (xs + 1); ws = 0.5 * ws
        return sum(wq * expm(x * B) @ T @ expm((1 - x) * B) for x, wq in zip(xs, ws))
    grad = np.eye(K) - Lfre(-om.T, Ae)
    f = lambda o: 0.5 * np.trace(inv(Sj) @ expm(-o) @ Si @ expm(-o).T) + 0.5 * (expm(-o) @ mi - mj) @ inv(Sj) @ (expm(-o) @ mi - mj) + np.trace(o)
    fd = np.zeros((K, K)); eps = 1e-6
    for a in range(K):
        for b in range(K):
            Eab = np.zeros((K, K)); Eab[a, b] = eps
            fd[a, b] = (f(om + Eab) - f(om - Eab)) / (2 * eps)
    rep(f"g11 gradient I - L(-omega^T, A_e) vs FD (|omega|~{scale})", np.abs(grad - fd).max(), np.abs(fd).max())
# orbit Fisher form
K = 3; mu = rng.standard_normal(K); S = spd(K); up = rng.standard_normal((K, K))
kq = (up @ mu) @ inv(S) @ (up @ mu) + np.trace(up @ up) + np.trace(inv(S) @ up @ S @ up.T)
for t in [1e-2, 1e-3]:
    E = expm(t * up); val = kl(mu, S, E @ mu, E @ S @ E.T)
    note(f"g11 orbit Fisher: KL/(t^2/2) = {val/(t*t/2):.8f} vs k_q = {kq:.8f} at t={t}")
print("\n".join(out))
