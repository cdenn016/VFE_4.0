r"""L05-FUS: symbolic and numeric checks of the static identities in chapters/06_fusion.tex.

Checks (all in log coordinates, where LN laws become Gaussians):
  1. precision of r_n(dm) G_n(m,dk) equals n A, A = [[a,-c],[-c,a]]      (fus:precision)
  2. KL(q_n x s_n || r_n G_n) = C + n U                                    (fus:local-formulas)
  3. predictive law s_n G_n                                               (fus:predictive-law)
  4. B_n restores unit variance; G_n B_n^h = B_n G_1                      (fus:message-kernel, fus:evaluator-square)
  5. sqrt(2 KL)/(1-chi) = |mu_i - mu_j|                                    (fus:distance)
  6. weighted geometric averaging: variance 1/n_I, joint reference r_{n_I}G_{n_I},
     conditional noise 1/(a n_I)                                          (fus:sample-retention)
  7. residual KL of synchronized children equals (n-1) C                   (handoff paragraph)
  8. reconstruction kernel covariance I - 11^T/n, residual independent of mean (fus:reconstruction)
Single-threaded, a few seconds.
"""
import numpy as np
import sympy as sp

a, c, n, mq, ms, chi = sp.symbols('a c n mu_q mu_s chi', positive=True)
mq, ms = sp.symbols('mu_q mu_s', real=True)

# 1. precision of joint reference in coordinates (x=log m, y=log k)
vx = a / (n * (a**2 - c**2))
b = c / a
w = 1 / (a * n)
P = sp.Matrix([[1 / vx + b**2 / w, -b / w], [-b / w, 1 / w]])
print("1. precision residual:", sp.simplify(P - n * sp.Matrix([[a, -c], [-c, a]])))

# 2. Gaussian KL in (log k, log m): recognition N((mq,ms), I/n), reference N(0,(nA)^{-1})
A = sp.Matrix([[a, -c], [-c, a]])
S0 = sp.eye(2) / n
S1inv = n * A
mvec = sp.Matrix([mq, ms])
KL = sp.Rational(1, 2) * ((S1inv * S0).trace() - 2 + (mvec.T * S1inv * mvec)[0]
                          + sp.log(1 / (S1inv.det())) - sp.log(S0.det()))
C = a - 1 - sp.Rational(1, 2) * sp.log(a**2 - c**2)
U = a / 2 * (mq**2 + ms**2) - c * mq * ms
print("2. KL - (C + nU):", sp.simplify(sp.expand_log(KL - (C + n * U), force=True)))

# 3. predictive variance
h = (c / a)**2 + 1 / a
pred_var = (c / a)**2 / n + 1 / (a * n)
print("3. predictive variance residual:", sp.simplify(pred_var - h / n))

# 4. message kernel and evaluator square
print("4a. q_n B_n variance residual:", sp.simplify(1 / n + (1 - 1 / n) - 1))
lhs = 1 / (a * n) + h * (1 - 1 / n)          # G_n then B_n^h
rhs = (c / a)**2 * (1 - 1 / n) + 1 / a       # B_n then G_1
print("4b. G_n B_n^h vs B_n G_1 variance residual:", sp.simplify(lhs - rhs),
      "; equals h-(c/a)^2/n:", sp.simplify(lhs - (h - (c / a)**2 / n)))
print("4c. predictive square variances:", sp.simplify(1 * (c / a)**2 + 1 / a - h),
      sp.simplify(h / n + h * (1 - 1 / n) - h))

# 5. distance
mi, mj = sp.symbols('m_i m_j', real=True)
kl_unit = sp.Rational(1, 2) * (mi - (mj + chi * (mi - mj)))**2
print("5. 2KL/(1-chi)^2 - (mi-mj)^2:", sp.simplify(2 * kl_unit / (1 - chi)**2 - (mi - mj)**2))

# 6. weighted averages (numeric, random multiplicities)
rng = np.random.default_rng(0)
an, cn = 1.3, 0.7
Anum = np.array([[an, -cn], [-cn, an]])
nj = np.array([1, 2, 4])
nI = nj.sum()
wts = nj / nI
var_avg = np.sum(wts**2 / nj)
cov_joint = sum(wt**2 * np.linalg.inv(k * Anum) for wt, k in zip(wts, nj))
cond_noise = np.sum(wts**2 / (an * nj))
print("6. 1/n_I residual:", var_avg - 1 / nI,
      "; joint cov residual:", np.abs(cov_joint - np.linalg.inv(nI * Anum)).max(),
      "; conditional noise residual:", cond_noise - 1 / (an * nI))

# 7. residual KL of n synchronized unit children (2n-dim Gaussian) minus parent KL
def gauss_kl(m0, S0, m1, S1):
    d = len(m0)
    S1i = np.linalg.inv(S1)
    dm = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) - d + dm @ S1i @ dm
                  + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])

Cnum = an - 1 - 0.5 * np.log(an**2 - cn**2)
for nn in (2, 3, 5):
    muq, mus = 0.4, -1.1
    m0 = np.tile([muq, mus], nn)                 # children (logk_j, logm_j) interleaved
    S0 = np.eye(2 * nn)
    S1 = np.kron(np.eye(nn), np.linalg.inv(Anum))
    kl_fine = gauss_kl(m0, S0, np.zeros(2 * nn), S1)
    Unum = an / 2 * (muq**2 + mus**2) - cn * muq * mus
    kl_parent = Cnum + nn * Unum
    print(f"7. n={nn}: fine KL - parent KL - (n-1)C =", kl_fine - kl_parent - (nn - 1) * Cnum,
          "; fine KL - n(C+U) =", kl_fine - nn * (Cnum + Unum))

# 8. reconstruction kernel: Z ~ N(mu 1, I); conditional on mean
for nn in (2, 4):
    Sigma = np.eye(nn)
    one = np.ones(nn)
    L = one / nn                                  # mean functional
    cov_ZZbar = Sigma @ L
    var_Zbar = L @ Sigma @ L
    cond_cov = Sigma - np.outer(cov_ZZbar, cov_ZZbar) / var_Zbar
    resid = np.eye(nn) - np.outer(one, one) / nn
    # residual R = Z - Zbar 1; cross covariance with Zbar
    Rmap = np.eye(nn) - np.outer(one, L)
    cross = Rmap @ Sigma @ L
    print(f"8. n={nn}: cond cov residual {np.abs(cond_cov - resid).max():.2e},",
          f"cross cov(R, Zbar) {np.abs(cross).max():.2e}, one-child var {cond_cov[0,0]:.4f} vs 1-1/n {1-1/nn:.4f},",
          f"independent-residual off-diag cov of reused-sample children {var_Zbar:.4f} (true 0)")

# 8b. generative joint residual kernel: children (log m_j, log k_j) iid under s_1 G_1
Sig_mk = np.array([[1.0, cn / an], [cn / an, (cn / an)**2 + 1 / an]])
nn = 3
SigBig = np.kron(np.eye(nn), Sig_mk)
Lbig = np.kron(np.ones((1, nn)) / nn, np.eye(2))      # 2 x 2n averaging
cond = SigBig - SigBig @ Lbig.T @ np.linalg.inv(Lbig @ SigBig @ Lbig.T) @ Lbig @ SigBig
target = np.kron(np.eye(nn) - np.ones((nn, nn)) / nn, Sig_mk)
print("8b. generative residual covariance (I-11^T/n) x Sigma_{s1G1} residual:", np.abs(cond - target).max())
print("    parent generative law s_nG_n = averaged child law: cov residual",
      np.abs(Lbig @ SigBig @ Lbig.T - Sig_mk / nn).max())
