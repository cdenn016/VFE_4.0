r"""D11-FUS identity checks for chapters/06_fusion.tex (single-threaded, a few seconds).

Each block prints a residual that should be exactly zero (sympy) or at machine precision.
Numerical agreement supports an identity; it does not prove a theorem.
"""
import sympy as sp
import numpy as np

a, c, n, chi, lam = sp.symbols('a c n chi lambda', positive=True)
mq, ms, mi, mj, v = sp.symbols('mu_q mu_s mu_i mu_j v', real=True)

def gauss_kl(m0, S0, m1, S1):
    r"""KL(N(m0,S0) || N(m1,S1)) = 1/2 [tr(S1^{-1}S0) - k + (m1-m0)^T S1^{-1}(m1-m0) + log det S1/det S0]."""
    k = S0.shape[0]
    S1i = S1.inv()
    d = m1 - m0
    return sp.Rational(1, 2)*((S1i*S0).trace() - k + (d.T*S1i*d)[0, 0] + sp.log(S1.det()/S0.det()))

A = sp.Matrix([[a, -c], [-c, a]])
C = a - 1 - sp.log(a**2 - c**2)/2
U = a/2*(mq**2 + ms**2) - c*mq*ms

# 1. Joint precision of r_n(dm) G_n(m,dk) in (log m, log k): X~N(0, a/(n(a^2-c^2))), Y|X~N((c/a)X, 1/(a n)).
sX2 = a/(n*(a**2 - c**2)); b = c/a; t2 = 1/(a*n)
P = sp.Matrix([[1/sX2 + b**2/t2, -b/t2], [-b/t2, 1/t2]])
print("1  precision - nA (should be zero matrix):", sp.simplify(P - n*A))

# 2. Local energy E_{0,n} = KL(q_n x s_n || r_n G_n) = C + n U  (fus:local-formulas)
E = gauss_kl(sp.Matrix([mq, ms]), sp.eye(2)/n, sp.zeros(2, 1), (n*A).inv())
print("2  E_{0,n} - (C + nU):", sp.simplify(sp.expand(sp.logcombine(E - (C + n*U), force=True))))

# 3. Predictive law s_n G_n: log k ~ N((c/a) mu_s, ((c/a)^2 + 1/a)/n)   (fus:predictive-law)
h = (c/a)**2 + 1/a
print("3  Var(log k) - h/n:", sp.simplify((c/a)**2/n + 1/(a*n) - h/n))

# 4. Message kernel: 1/n + (1 - 1/n) = 1; evaluator square variances (fus:evaluator-square)
lhs = 1/(a*n) + h*(1 - 1/n)                # G_n then B_n^h
rhs = (c/a)**2*(1 - 1/n) + 1/a             # B_n then G_1
print("4  G_n B_n^h vs B_n G_1 variance residual:", sp.simplify(lhs - rhs),
      "; equals h-(c/a)^2/n:", sp.simplify(lhs - (h - (c/a)**2/n)))
print("4b predictive square: h/n + h(1-1/n) - h =", sp.simplify(h/n + h*(1 - 1/n) - h))

# 5. Normalized square-root cost (fus:distance) and its chi-independence
L = chi*(mi - mj)
kl_restored = (mi - (mj + L))**2/2      # KL(LN(mi,1) || LN(mj+L,1))
d = sp.sqrt(2*kl_restored)/(1 - chi)
print("5  d^2 - (mi-mj)^2:", sp.simplify(d**2 - (mi - mj)**2))

# 6. Variance release derivative (common log variance v in both sectors) at v=1/n
Ev = gauss_kl(sp.Matrix([mq, ms]), v*sp.eye(2), sp.zeros(2, 1), (n*A).inv())
dEv = sp.simplify(sp.diff(Ev, v).subs(v, 1/n))
print("6  dE/dv at v=1/n:", dEv, " (text: n(a-1))")

# 7. The def:dyn-potential peer cost with an UNRESTORED receiver law LN(mu_i,1/n_i) against the
#    transported restored source LN(mu_j+L,1): sqrt(2D)/(1-chi) is NOT |mu_i-mu_j| for n_i>=2.
ni = sp.symbols('n_i', positive=True)
D_unrestored = gauss_kl(sp.Matrix([mi]), sp.Matrix([[1/ni]]), sp.Matrix([mj + L]), sp.Matrix([[1]]))
d_unr = sp.sqrt(2*D_unrestored)/(1 - chi)
print("7  2*D_unrestored - [(1/n_i - 1 + log n_i) + (1-chi)^2 (mi-mj)^2]:",
      sp.simplify(sp.expand(sp.logcombine(2*D_unrestored - ((1/ni - 1 + sp.log(ni)) + (1 - chi)**2*(mi - mj)**2), force=True))))
dd = sp.diff(d_unr, mi).subs(mi, mj)
print("7b derivative of unrestored cost at mu_i=mu_j:", sp.simplify(dd), "(smooth: no corner, no sticking)")
print("7c value at n_i=2, chi=0.3, gap 0: ", float(d_unr.subs({ni: 2, chi: sp.Rational(3, 10), mi: 0, mj: 0})),
      " vs at chi=0.6:", float(d_unr.subs({ni: 2, chi: sp.Rational(3, 5), mi: 0, mj: 0})), "(chi-dependent)")

# 8. Joint reconstruction: Z ~ N(mu 1, I_n); Z | Zbar ~ N(Zbar 1, I - 11^T/n); residual independent of Zbar
for nn in (2, 3, 5):
    one = np.ones((nn, 1)); S = np.eye(nn)
    cov_zbar = (one.T @ S @ one)[0, 0]/nn**2
    cross = S @ one/nn
    cond = S - cross @ cross.T/cov_zbar
    resid_cross = (np.eye(nn) - one @ one.T/nn) @ cross
    print(f"8  n={nn}: ||cond - (I-11^T/n)||={np.abs(cond - (np.eye(nn) - one@one.T/nn)).max():.1e}, "
          f"||Cov(resid, Zbar)||={np.abs(resid_cross).max():.1e}, one-child var={cond[0,0]:.6f} (1-1/n={1-1/nn:.6f})")

# 9. Residual information (n-1)C: chain rule KL(prod children || prod ref) = KL(stat) + E KL(cond)
for nn in (2, 3, 5):
    av, cv = 1.7, 0.6
    Am = np.array([[av, -cv], [-cv, av]])
    mu = np.array([0.3, -0.8])
    def kl(m0, S0, m1, S1):
        S1i = np.linalg.inv(S1); dlt = m1 - m0
        return 0.5*(np.trace(S1i @ S0) - len(m0) + dlt @ S1i @ dlt + np.log(np.linalg.det(S1)/np.linalg.det(S0)))
    full = nn*kl(mu, np.eye(2), np.zeros(2), np.linalg.inv(Am))          # n unit children
    stat = kl(mu, np.eye(2)/nn, np.zeros(2), np.linalg.inv(Am)/nn)        # retained parent law
    Cn = av - 1 - 0.5*np.log(av**2 - cv**2)
    print(f"9  n={nn}: full - stat - (n-1)C = {full - stat - (nn-1)*Cn:.2e}")

# 10. Sample retention variance sum_j (n_j/n_I)^2 / n_j = 1/n_I and evaluator noise 1/(a n_I)
nj = np.array([1, 2, 4, 3]); nI = nj.sum(); av = 1.7
print("10 weighted log-average variance - 1/n_I:", (nj/nI)**2 @ (1/nj) - 1/nI,
      "; evaluator noise - 1/(a n_I):", (nj/nI)**2 @ (1/(av*nj)) - 1/(av*nI))

# 11. Peer-term bookkeeping: (1/2) lam N n_i sum_j (n_j/N) d_ij summed over i equals lam sum_{i<j} n_i n_j d_ij
rng = np.random.default_rng(1)
nvec = rng.integers(1, 5, size=5).astype(float); N = nvec.sum(); x = rng.normal(size=5); lv = 0.7
Dm = np.abs(x[:, None] - x[None, :])
lhs11 = sum(0.5*lv*N*nvec[i]*sum((nvec[j]/N)*Dm[i, j] for j in range(5)) for i in range(5))
rhs11 = lv*sum(nvec[i]*nvec[j]*Dm[i, j] for i in range(5) for j in range(i + 1, 5))
print("11 peer-term bookkeeping residual:", lhs11 - rhs11)
