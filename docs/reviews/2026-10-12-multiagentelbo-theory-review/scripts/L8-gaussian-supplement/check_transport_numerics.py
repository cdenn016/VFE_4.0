"""L8 numerical-hazard probe: transported Gaussian KL for large-norm GL(K) transports, float64 vs 50-digit mpmath."""
import numpy as np
import mpmath as mp
from numpy.linalg import inv, slogdet, cholesky, cond
from scipy.linalg import expm, solve_triangular, cho_factor, cho_solve
from common import rng, spd, kl

mp.mp.dps = 50
def mpkl_transported(mi, Si, mj, Sj, om):
    K = len(mi); O = mp.expm(mp.matrix(om.tolist()))
    Sp = O * mp.matrix(Sj.tolist()) * O.T; mup = O * mp.matrix(mj.tolist())
    Spi = Sp ** -1; d = mp.matrix(mi.tolist()) - mup
    tr = sum((Spi * mp.matrix(Si.tolist()))[a, a] for a in range(K))
    q = (d.T * Spi * d)[0]
    return 0.5 * (tr + q - K + mp.log(mp.det(Sp)) - mp.log(mp.det(mp.matrix(Si.tolist()))))

def kl_pullback(mi, Si, mj, Sj, om):
    """Closed form of g11 eq:glink-law-level-energy, evaluated with Cholesky solves, no inverse of Omega Sigma Omega^T."""
    K = len(mi); Ei = expm(-om)
    Lj = cholesky(Sj)
    A = solve_triangular(Lj, Ei @ cholesky(Si), lower=True)          # L_j^{-1} e^{-w} L_i
    d = solve_triangular(Lj, Ei @ mi - mj, lower=True)
    logdet_j = 2 * np.sum(np.log(np.diag(Lj))); logdet_i = 2 * np.sum(np.log(np.diag(cholesky(Si))))
    return 0.5 * (np.sum(A * A) + d @ d - K + logdet_j - logdet_i) + np.trace(om)

def kl_naive(mi, Si, mj, Sj, om):
    E = expm(om); return kl(mi, Si, E @ mj, E @ Sj @ E.T)

K = 3
print("scale  cond(Omega)   cond(Omega Sj Omega^T)   mp50 value        naive rel.err   pullback rel.err")
for scale in [0.3, 1.0, 2.0, 3.0]:
    om = scale * rng.standard_normal((K, K)); mi, mj = rng.standard_normal(K), rng.standard_normal(K); Si, Sj = spd(K, 1e3), spd(K, 1e3)
    ref = float(mpkl_transported(mi, Si, mj, Sj, om))
    E = expm(om)
    print(f"{scale:4.1f}  {cond(E):10.2e}   {cond(E @ Sj @ E.T):12.2e}   {ref: .10e}   {abs(kl_naive(mi,Si,mj,Sj,om)-ref)/abs(ref):10.2e}   {abs(kl_pullback(mi,Si,mj,Sj,om)-ref)/abs(ref):10.2e}")

# near-zero KL cancellation: q_i = transported q_j up to a tiny perturbation
print("\nsmall-KL cancellation (true KL ~ eps^2):")
for eps in [1e-3, 1e-5, 1e-7]:
    om = 0.5 * rng.standard_normal((K, K)); mj = rng.standard_normal(K); Sj = spd(K, 1e2)
    E = expm(om); mi = E @ mj + eps * rng.standard_normal(K); Si = E @ Sj @ E.T + eps * np.eye(K)
    ref = float(mpkl_transported(mi, Si, mj, Sj, om))
    # stable form: 1/2[sum (lam - 1 - log lam) + |d|^2] with lam = eig of whitened Si
    Lp = cholesky(E @ Sj @ E.T); Wm = solve_triangular(Lp, cholesky(Si), lower=True); lam = np.linalg.eigvalsh(Wm @ Wm.T)
    d = solve_triangular(Lp, mi - E @ mj, lower=True)
    stable = 0.5 * (np.sum(lam - 1 - np.log(lam)) + d @ d)
    print(f"eps={eps:.0e}: mp50 {ref:.6e}; pullback-trace form rel.err {abs(kl_pullback(mi,Si,mj,Sj,om)-ref)/ref:.2e}; eigen (lam-1-log lam) form rel.err {abs(stable-ref)/ref:.2e}")
