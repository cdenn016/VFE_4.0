r"""GAP05-S3 s4: hypotheses of it3:general-induced and the general-law it3:rigid.

A. it3:general-induced, non-Gaussian scalar instance: contents y_i, y_j with Laplace and
   Student-t self laws, dictionary prior Student-t, logistic record noise on
   o = y_i - y_j - w c + eps. (A1) Bayes formula Z(w) pi(w) / int Z pi against the
   w-marginal of the joint grid posterior. (A2) Version dependence at a fixed registered o:
   a second jointly measurable density version that differs from the first only at o = o0
   (a Lebesgue-null set of records) gives a different "posterior at o0".
B. it3:rigid. (B1) the parent precision for Gaussian self laws under three reference
   measures mu (Lebesgue, N(0, 1), density e^{b z}); (B2) a case where Z_A is finite for
   Lebesgue and infinite for mu = N(0, 1); (B3) the literal reading of the bridge normalizer
   "against this product measure" without the parent self densities diverges for a Gaussian
   bridge; (C) soft-to-rigid limit of the normalized parent marginal for non-Gaussian self
   densities with a Lebesgue-normalized cross-scale kernel.
"""
import numpy as np
from scipy.stats import laplace, t as student_t, logistic, norm, gumbel_r
from scipy.integrate import quad

print("== A1. Bayes formula for the induced action, non-Gaussian instance ==")
c = 1.3
o0 = 1.9
y = np.linspace(-14, 14, 561); dy = y[1] - y[0]
w = np.linspace(-10, 10, 401); dw = w[1] - w[0]
p_i = laplace.pdf(y, loc=0.5, scale=0.8)
p_j = student_t.pdf(y, 4, loc=-0.3, scale=0.7)
prior_w = student_t.pdf(w, 3, scale=0.9)
Yi, Yj = np.meshgrid(y, y, indexing='ij')
Z = np.empty_like(w)
for k, wk in enumerate(w):
    lik = logistic.pdf(o0 - (Yi - Yj - wk * c), scale=0.4)
    Z[k] = np.sum(lik * p_i[:, None] * p_j[None, :]) * dy * dy
post_formula = Z * prior_w / (np.sum(Z * prior_w) * dw)
joint_marg = np.empty_like(w)
for k, wk in enumerate(w):
    joint = p_i[:, None] * p_j[None, :] * prior_w[k] * logistic.pdf(o0 - (Yi - Yj - wk * c), scale=0.4)
    joint_marg[k] = joint.sum(axis=1).sum() * dy * dy
joint_marg /= np.sum(joint_marg) * dw
print("max |posterior from Z - joint-grid marginal| = %.1e (Fubini on the grid)" % np.abs(post_formula - joint_marg).max())
print("posterior mean of w = %.6f; Z > 0 on the grid: %s" % (np.sum(w * post_formula) * dw, bool(np.all(Z > 0))))

print("== A2. a second density version, equal to the first except at o = o0 ==")
Z2 = Z * (w <= 0)
post2 = Z2 * prior_w / (np.sum(Z2 * prior_w) * dw)
print("posterior mean at o0: version 1 %.6f, version 2 %.6f; P(w > 0 | o0): %.4f vs %.4f"
      % (np.sum(w * post_formula) * dw, np.sum(w * post2) * dw, np.sum(post_formula[w > 0]) * dw, np.sum(post2[w > 0]) * dw))
print("Both versions give the same joint law of (y, w, o), so both posteriors are versions of the")
print("conditional law; the value at one registered o is fixed only after a version is chosen.")

print("== B1. Gaussian parent precision under different reference measures mu ==")
PA, mA, m1, m2 = 0.5, 0.4, -1.0, 2.0
def parent(P1, P2, log_mu, L=40.0, n=400001):
    z = np.linspace(-L, L, n); dz = z[1] - z[0]
    logs = [norm.logpdf(z, mA, PA ** -0.5), norm.logpdf(z, m1, P1 ** -0.5), norm.logpdf(z, m2, P2 ** -0.5)]
    log_law = sum(ls - log_mu(z) for ls in logs) + log_mu(z)   # (p_A p_1 p_2 w.r.t. mu) times d mu / dz
    f = np.exp(log_law - log_law.max()); Zf = np.sum(f) * dz
    mean = np.sum(z * f) * dz / Zf; var = np.sum((z - mean) ** 2 * f) * dz / Zf
    return 1 / var, mean
for (P1, P2) in [(0.5, 0.5), (1.5, 1.5)]:
    pf = PA + P1 + P2; mf = (PA * mA + P1 * m1 + P2 * m2) / pf
    pl, ml = parent(P1, P2, lambda z: 0 * z)
    print("P_1 = P_2 = %.1f: it3:parent-gaussian precision %.6f mean %.6f; mu = Lebesgue: %.6f, %.6f" % (P1, pf, mf, pl, ml))
pf = PA + 3.0; mf = (PA * mA + 1.5 * m1 + 1.5 * m2) / pf
pg, mg = parent(1.5, 1.5, lambda z: norm.logpdf(z))
print("mu = N(0,1): precision %.6f (formula - |I| = %.6f), mean %.6f (formula %.6f)" % (pg, pf - 2, mg, mf))
b = 0.7
pe, me = parent(1.5, 1.5, lambda z: b * z)
print("mu = e^{%.1f z} dz: precision %.6f, mean %.6f (formula %.6f; predicted shift -|I| b / P' = %.6f)" % (b, pe, me, mf, -2 * b / pf))

print("== B2. Z_A finite for Lebesgue, infinite for mu = N(0,1) (P_A = P_1 = P_2 = 0.5) ==")
for L in (10.0, 20.0, 40.0):
    z = np.linspace(-L, L, 200001); dz = z[1] - z[0]
    logs = [norm.logpdf(z, mA, PA ** -0.5), norm.logpdf(z, m1, 0.5 ** -0.5), norm.logpdf(z, m2, 0.5 ** -0.5)]
    ZA_leb = np.sum(np.exp(sum(logs))) * dz
    lg = sum(ls - norm.logpdf(z) for ls in logs)                              # log of p_A p_1 p_2 w.r.t. mu, times d mu/dz
    log10_ZA_gau = (lg.max() + np.log(np.sum(np.exp(lg - lg.max())) * dz)) / np.log(10)
    print("  box [-%g, %g]: Z_A(Lebesgue) = %.6e   log10 Z_A(N(0,1)) = %.1f" % (L, L, ZA_leb, log10_ZA_gau))

print("== B3. literal bridge normalizer against mu x mu x pi_b, no parent self densities ==")
R, cb = 0.5, 1.0
wq = np.linspace(-6, 6, 121); dwq = wq[1] - wq[0]; piw = norm.pdf(wq)
for L in (5.0, 10.0, 20.0, 40.0):
    zA = np.linspace(-L, L, 801); dz = zA[1] - zA[0]
    zB = np.linspace(-L, L, 801)
    tot = 0.0
    for k in range(len(wq)):
        g = np.exp(-(0.3 - (zA[:, None] - zB[None, :] - wq[k] * cb)) ** 2 / (2 * R))
        tot += piw[k] * np.sum(g) * dz * dz * dwq
    print("  box half-width %4.0f: integral %.4f, per unit length %.4f" % (L, tot, tot / (2 * L)))
inner = lambda zz: quad(lambda u: norm.pdf(u) * np.exp(-(0.3 - (zz - u)) ** 2 / (2 * (R + cb ** 2))) * np.sqrt(R / (R + cb ** 2)), -12, 12)[0]
print("  with parent self laws N(0,1) included: %.6f (finite)" % quad(lambda zz: norm.pdf(zz) * inner(zz), -12, 12)[0])

print("== C. soft-to-rigid limit of the normalized parent marginal, non-Gaussian self densities ==")
z = np.linspace(-12, 12, 4801); dz = z[1] - z[0]
pA = gumbel_r.pdf(z, loc=0.2, scale=1.1)
pc = [laplace.pdf(z, loc=-0.5, scale=0.9), student_t.pdf(z, 5, loc=0.8, scale=1.0)]
rigid = pA * pc[0] * pc[1]; rigid /= np.sum(rigid) * dz
prev = None
for tau in (4.0, 16.0, 64.0, 256.0, 1024.0):
    s = tau ** -0.5
    kgrid = np.arange(-int(8 * s / dz) - 1, int(8 * s / dz) + 2) * dz
    ker = norm.pdf(kgrid, scale=s); ker /= np.sum(ker) * dz
    sm = [np.convolve(p, ker, mode='same') * dz for p in pc]
    soft = pA * sm[0] * sm[1]; soft /= np.sum(soft) * dz
    tv = 0.5 * np.sum(np.abs(soft - rigid)) * dz
    print("  tau = %6.0f: TV(soft, rigid with Lebesgue) = %.3e%s" % (tau, tv, "" if prev is None else "   ratio %.2f" % (prev / tv)))
    prev = tv
