"""GAP09-S1 s1: prop:dyn-per-agent on a continuous (non-finite) model.

Model (one agent i, self + one peer in each sector), all one-dimensional:
  presentation sources rho_l(dm): l=0 (self) r = N(0,1); l=1 (peer) N(mu_p, v_p)
  content sources kappa_j(m, dk): j=0 (self) G(m,.) = N(a*m, g); j=1 (peer) N(c_p, w_p)
  record l(o | k, m) = N(o; k + b*m, sig2)  (presentation-dependent likelihood)
Recognition Q = s (x) q (x) beta (x) gamma with s = N(ms, vs), q = N(mq, vq).

Checks
 (a) F_direct = E_Q[log Q - log P(m,k,J,L,o)] by tensor Gauss-Hermite quadrature
     equals the separated formula eq:dyn-per-agent-free-energy (closed-form KLs).
 (b) the closed-form evidence P(o) agrees with quadrature of E_{P^0} l(o|K,M), and
     F >= -log P(o) (that is, KL(Q || Pi_o) >= 0, eq:agent-free-energy) on every instance.
 (c) At fixed (s, q) the unit-temperature Gibbs rows minimize F over 2000 random rows.
 (d) Donsker-Varadhan control of the record term:
     E_Q[(log l)^+] <= KL(Q || P^0) + log(1 + P(o)) on random instances with narrow records.
"""
import numpy as np

rng = np.random.default_rng(20261029)
xg, wg = np.polynomial.hermite_e.hermegauss(40)  # probabilists' Hermite: weight exp(-x^2/2)
wg = wg / wg.sum()


def logN(x, m, v):
    return -0.5 * np.log(2 * np.pi * v) - 0.5 * (x - m) ** 2 / v


def kl_gauss(m1, v1, m2, v2):
    return 0.5 * (v1 / v2 + (m1 - m2) ** 2 / v2 - 1.0 + np.log(v2 / v1))


def instance(rng):
    p = dict(a=rng.normal(), g=rng.uniform(0.3, 2.0), mu_p=rng.normal(), v_p=rng.uniform(0.3, 2.0),
             c_p=rng.normal(), w_p=rng.uniform(0.3, 2.0), b=rng.normal(), sig2=rng.uniform(0.05, 1.0),
             o=rng.normal(), ms=rng.normal(), vs=rng.uniform(0.2, 1.5), mq=rng.normal(),
             vq=rng.uniform(0.2, 1.5))
    piq = rng.dirichlet([1, 1]); pis = rng.dirichlet([1, 1])
    beta = rng.dirichlet([1, 1]); gamma = rng.dirichlet([1, 1])
    return p, piq, pis, beta, gamma


def costs(p):
    # presentation costs KL(s || rho_l)
    cs = np.array([kl_gauss(p['ms'], p['vs'], 0.0, 1.0), kl_gauss(p['ms'], p['vs'], p['mu_p'], p['v_p'])])
    # content costs int KL(q || kappa_j(m,.)) s(dm)
    self_c = 0.5 * (p['vq'] / p['g'] + ((p['mq'] - p['a'] * p['ms']) ** 2 + p['a'] ** 2 * p['vs']) / p['g']
                    - 1.0 + np.log(p['g'] / p['vq']))
    cq = np.array([self_c, kl_gauss(p['mq'], p['vq'], p['c_p'], p['w_p'])])
    # record term E[-log N(o; k + b m, sig2)]
    e2 = (p['o'] - p['mq'] - p['b'] * p['ms']) ** 2 + p['vq'] + p['b'] ** 2 * p['vs']
    rec = 0.5 * np.log(2 * np.pi * p['sig2']) + 0.5 * e2 / p['sig2']
    return cs, cq, rec


def kl_cat(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return float(np.sum(x * np.log(x / y)))


def separated(p, piq, pis, beta, gamma):
    cs, cq, rec = costs(p)
    return float(gamma @ cs + kl_cat(gamma, pis) + beta @ cq + kl_cat(beta, piq) + rec)


def grid(p):
    M = p['ms'] + np.sqrt(p['vs']) * xg[:, None]
    K = p['mq'] + np.sqrt(p['vq']) * xg[None, :]
    W = wg[:, None] * wg[None, :]
    return M, K, W


def log_joint(p, piq, pis, M, K, j, l):
    lr = logN(M, 0.0, 1.0) if l == 0 else logN(M, p['mu_p'], p['v_p'])
    lk = logN(K, p['a'] * M, p['g']) if j == 0 else logN(K, p['c_p'], p['w_p'])
    lo = logN(p['o'], K + p['b'] * M, p['sig2'])
    return np.log(pis[l]) + lr + np.log(piq[j]) + lk + lo


def direct(p, piq, pis, beta, gamma):
    M, K, W = grid(p)
    logQmk = logN(M, p['ms'], p['vs']) + logN(K, p['mq'], p['vq'])
    tot = 0.0
    for j in range(2):
        for l in range(2):
            integrand = logQmk + np.log(beta[j]) + np.log(gamma[l]) - log_joint(p, piq, pis, M, K, j, l)
            tot += beta[j] * gamma[l] * np.sum(W * integrand)
    return float(tot)


def evidence(p, piq, pis):
    # Z_jl(o) = int int N(o; k + b m, sig2) kappa_j(m,dk) rho_l(dm), closed form
    Z = 0.0
    for l in range(2):
        mm, vm = (0.0, 1.0) if l == 0 else (p['mu_p'], p['v_p'])
        for j in range(2):
            if j == 0:  # k | m ~ N(a m, g): o ~ N((a+b) mm, sig2 + g + (a+b)^2 vm)
                mo = (p['a'] + p['b']) * mm; vo = p['sig2'] + p['g'] + (p['a'] + p['b']) ** 2 * vm
            else:       # k ~ N(c_p, w_p) indep of m: o ~ N(c_p + b mm, sig2 + w_p + b^2 vm)
                mo = p['c_p'] + p['b'] * mm; vo = p['sig2'] + p['w_p'] + p['b'] ** 2 * vm
            Z += pis[l] * piq[j] * np.exp(logN(p['o'], mo, vo))
    return Z


def kl_to_posterior(p, piq, pis, beta, gamma, Po):
    M, K, W = grid(p)
    logQmk = logN(M, p['ms'], p['vs']) + logN(K, p['mq'], p['vq'])
    tot = 0.0
    for j in range(2):
        for l in range(2):
            logPi = log_joint(p, piq, pis, M, K, j, l) - np.log(Po)
            tot += beta[j] * gamma[l] * np.sum(W * (logQmk + np.log(beta[j]) + np.log(gamma[l]) - logPi))
    return float(tot)


def evidence_quadrature(p, piq, pis):
    Z = 0.0
    for l in range(2):
        mm, vm = (0.0, 1.0) if l == 0 else (p['mu_p'], p['v_p'])
        zz = np.linspace(-12.0, 12.0, 40001)
        wz = np.exp(-0.5 * zz ** 2) / np.sqrt(2 * np.pi) * (zz[1] - zz[0])
        Mn = mm + np.sqrt(vm) * zz
        for j in range(2):
            # integrate k exactly (Gaussian convolution), then m by a fine trapezoid rule (independent check
            # of the m-convolution in the closed form)
            if j == 0:
                L = np.exp(logN(p['o'], (p['a'] + p['b']) * Mn, p['sig2'] + p['g']))
            else:
                L = np.exp(logN(p['o'], p['c_p'] + p['b'] * Mn, p['sig2'] + p['w_p']))
            Z += pis[l] * piq[j] * np.sum(wz * L)
    return Z


res_a, res_b, gaps, worst_row, dv_ok = [], [], [], np.inf, True
for trial in range(200):
    p, piq, pis, beta, gamma = instance(rng)
    Fs = separated(p, piq, pis, beta, gamma)
    Fd = direct(p, piq, pis, beta, gamma)
    Po = evidence(p, piq, pis)
    Poq = evidence_quadrature(p, piq, pis)
    res_a.append(abs(Fs - Fd) / max(1.0, abs(Fd)))
    res_b.append(abs(Po - Poq) / Po)
    gaps.append(Fd + np.log(Po))
    # (c) Gibbs rows at unit temperature
    cs, cq, rec = costs(p)
    bstar = piq * np.exp(-cq); bstar /= bstar.sum()
    gstar = pis * np.exp(-cs); gstar /= gstar.sum()
    Fstar = separated(p, piq, pis, bstar, gstar)
    for _ in range(10):
        b2 = rng.dirichlet([0.5, 0.5]); g2 = rng.dirichlet([0.5, 0.5])
        worst_row = min(worst_row, separated(p, piq, pis, b2, g2) - Fstar)
    # (d) DV bound: E_Q[(log l)^+] <= KL(Q||P0) + log(1 + P(o)); make records narrow
    p2 = dict(p); p2['sig2'] = rng.uniform(1e-4, 1e-2)
    Po2 = evidence(p2, piq, pis)
    M, K, W = grid(p2)
    lo = logN(p2['o'], K + p2['b'] * M, p2['sig2'])
    lhs = float(np.sum(W * np.maximum(lo, 0.0)))
    cs2, cq2, _ = costs(p2)
    klP0 = float(gamma @ cs2 + kl_cat(gamma, pis) + beta @ cq2 + kl_cat(beta, piq))
    if lhs > klP0 + np.log1p(Po2) + 1e-12:
        dv_ok = False

print('(a) max relative |separated - direct quadrature| over 200 instances: %.2e' % max(res_a))
print('(b) max relative |closed-form P(o) - quadrature P(o)|: %.2e; min of F + log P(o) = KL(Q||Pi_o): %.3e (>= 0 expected)' % (max(res_b), min(gaps)))
print('(c) min over random rows of F(rows) - F(Gibbs rows): %.3e (>= 0 expected)' % worst_row)
print('(d) DV bound E_Q[(log l)^+] <= KL(Q||P0) + log(1+P(o)) held on all 200 narrow-record instances:', dv_ok)
