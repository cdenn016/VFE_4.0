r"""L1-variational check A: evidence-bound identities of c05 on a linear-Gaussian
three-agent DAG with cyclic records (scopes {1,2}, {2,3}, {3}).

Latent y = (k1, m1, k2, m2, k3, m3); agent blocks i -> indices (2i, 2i+1).
"""
import itertools
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

rng = np.random.default_rng(7)
LOG2PI = np.log(2 * np.pi)


def model(theta):
    n = 6
    B = np.zeros((n, n)); c = np.zeros(n); d = np.zeros(n)
    K1, M1, K2, M2, K3, M3 = range(6)
    c[M1] = theta; d[M1] = 1.0
    B[K1, M1] = 0.5; d[K1] = 0.8
    B[M2, M1] = 0.6; d[M2] = 1.0
    B[K2, K1] = 0.9; B[K2, M2] = 0.3; d[K2] = 0.7
    B[M3, M1] = -0.4; d[M3] = 1.2
    B[K3, K1] = 0.9; B[K3, M3] = -0.2; d[K3] = 0.6
    A = np.linalg.inv(np.eye(n) - B)
    mu0 = A @ c; S0 = A @ np.diag(d) @ A.T
    H = np.zeros((3, n)); r = np.array([0.5, 0.4, 0.3])
    H[0, K1] = 1; H[0, K2] = -1
    H[1, K2] = 1; H[1, K3] = -0.8
    H[2, K3] = 1; H[2, M3] = 1
    return dict(B=B, c=c, d=d, mu0=mu0, S0=S0, H=H, r=r)


def kl_gauss(m1, S1, m2, S2):
    k = len(m1); S2i = np.linalg.inv(S2); dm = m2 - m1
    return 0.5 * (np.trace(S2i @ S1) - k + dm @ S2i @ dm
                  + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1])


def evidence_posterior(M, o):
    S0, mu0, H, r = M['S0'], M['mu0'], M['H'], M['r']
    So = H @ S0 @ H.T + np.diag(r)
    do = o - H @ mu0
    logp = -0.5 * (len(o) * LOG2PI + np.linalg.slogdet(So)[1] + do @ np.linalg.solve(So, do))
    J = np.linalg.inv(S0) + H.T @ np.diag(1 / r) @ H
    Sp = np.linalg.inv(J)
    mp = Sp @ (np.linalg.solve(S0, mu0) + H.T @ (o / r))
    return logp, mp, Sp


def E_quad(mu, S, A, b=None, c0=0.0):
    v = np.trace(A @ S) + mu @ A @ mu + c0
    if b is not None:
        v += b @ mu
    return v


def factor_terms(M, o, mq, Sq):
    B, c, d, H, r = M['B'], M['c'], M['d'], M['H'], M['r']
    out = []
    for i in range(6):
        a = np.zeros(6); a[i] = 1.0; a = a - B[i]
        A = np.outer(a, a) / (2 * d[i])
        out.append(0.5 * np.log(2 * np.pi * d[i]) + E_quad(mq, Sq, A, b=-2 * c[i] * a / (2 * d[i]),
                                                              c0=c[i] ** 2 / (2 * d[i])))
    rec = []
    for a_ in range(len(o)):
        h = H[a_]
        A = np.outer(h, h) / (2 * r[a_])
        rec.append(0.5 * np.log(2 * np.pi * r[a_]) + E_quad(mq, Sq, A, b=-2 * o[a_] * h / (2 * r[a_]),
                                                               c0=o[a_] ** 2 / (2 * r[a_])))
    return np.array(out), np.array(rec)


def neg_entropy(Sq):
    return -0.5 * (len(Sq) * (1 + LOG2PI) + np.linalg.slogdet(Sq)[1])


def rand_spd(k, scale=1.0):
    A = rng.normal(size=(k, k)) * scale
    return A @ A.T + 0.3 * np.eye(k)


theta = 0.4
M = model(theta)
o = np.array([0.7, -1.1, 0.9])
logp, mp, Sp = evidence_posterior(M, o)
mq = mp + rng.normal(size=6) * 0.5
Sq = rand_spd(6, 0.5)

Lext = logp - kl_gauss(mq, Sq, mp, Sp)
fac, rec = factor_terms(M, o, mq, Sq)
F_class = fac.sum() + rec.sum() + neg_entropy(Sq)
print("[1] L_ext + F_factorwise (should be 0)        :", Lext + F_class)
F_energy = kl_gauss(mq, Sq, M['mu0'], M['S0']) + rec.sum()
print("[2] F_ext - (KL(Q||P^Y)+E_Q H_o)               :", -Lext - F_energy)
blocks = [[0, 1], [2, 3], [4, 5]]
rho = [(rng.normal(size=2), rand_spd(2)) for _ in blocks]
mrho = np.concatenate([x_[0] for x_ in rho]); Srho = np.zeros((6, 6))
for bl, (m_, S_) in zip(blocks, rho):
    Srho[np.ix_(bl, bl)] = S_
Sprod = np.zeros((6, 6))
for bl in blocks:
    Sprod[np.ix_(bl, bl)] = Sq[np.ix_(bl, bl)]
TC = kl_gauss(mq, Sq, mq, Sprod)
lhs = kl_gauss(mq, Sq, mrho, Srho)
rhs = TC + sum(kl_gauss(mq[bl], Sq[np.ix_(bl, bl)], m_, S_) for bl, (m_, S_) in zip(blocks, rho))
print("[3] TC chain residual                         :", lhs - rhs, " TC =", TC)
Ftilde = fac.sum() + rec.sum() + neg_entropy(Sprod)
print("    pseudo-ELBO: Ltilde - (L + TC)              :", -Ftilde - (Lext + TC))
Spp = np.zeros((6, 6))
for bl in blocks:
    Spp[np.ix_(bl, bl)] = Sp[np.ix_(bl, bl)]
fac_p, rec_p = factor_terms(M, o, mp, Sp)
Lt_post = -(fac_p.sum() + rec_p.sum() + neg_entropy(Spp))
print("    at Q=posterior: Ltilde - log p, TC(post)    :", Lt_post - logp, kl_gauss(mp, Sp, mp, Spp))

Bi = [2, 3]; Oi = [0, 1, 4, 5]
recB = [0, 1]
recO = [2]


def cond_gauss(mu, S, I, Jx):
    SJJ = S[np.ix_(Jx, Jx)]; SIJ = S[np.ix_(I, Jx)]
    G = SIJ @ np.linalg.inv(SJJ)
    return mu[I] - G @ mu[Jx], G, S[np.ix_(I, I)] - G @ SIJ.T


aP, GP, CP = cond_gauss(M['mu0'], M['S0'], Bi, Oi)
aPi, GPi, CPi = cond_gauss(mp, Sp, Bi, Oi)


def H_terms(y, recs):
    H, r = M['H'], M['r']
    return sum(0.5 * np.log(2 * np.pi * r[a]) + (o[a] - H[a] @ y) ** 2 / (2 * r[a]) for a in recs)


def logZB(ym):
    H, r = M['H'], M['r']
    mB = aP + GP @ ym
    Hb = H[np.ix_(recB, Bi)]; Hm = H[np.ix_(recB, Oi)]
    mean = Hb @ mB + Hm @ ym
    Cov = Hb @ CP @ Hb.T + np.diag(r[recB])
    dv = o[recB] - mean
    return -0.5 * (len(recB) * LOG2PI + np.linalg.slogdet(Cov)[1] + dv @ np.linalg.solve(Cov, dv))


def E_r_HB(ym, rm, rC):
    H, r = M['H'], M['r']
    tot = 0.0
    for a in recB:
        hb = H[a, Bi]; hm = H[a, Oi]
        mean = hb @ rm + hm @ ym
        tot += 0.5 * np.log(2 * np.pi * r[a]) + ((o[a] - mean) ** 2 + hb @ rC @ hb) / (2 * r[a])
    return tot


def F_B_two_ways(ym, ra, rG, rC):
    rm = ra + rG @ ym
    v1 = -logZB(ym) + kl_gauss(rm, rC, aPi + GPi @ ym, CPi)
    v2 = kl_gauss(rm, rC, aP + GP @ ym, CP) + E_r_HB(ym, rm, rC)
    return v1, v2


mO = mq[Oi]; SO = Sq[np.ix_(Oi, Oi)]
ra, rG, rC = rng.normal(size=2), rng.normal(size=(2, 4)) * 0.3, rand_spd(2, 0.4)
ra2, rG2, rC2 = rng.normal(size=2), rng.normal(size=(2, 4)) * 0.3, rand_spd(2, 0.4)


def joint_from(mO, SO, ra, rG, rC):
    mu = np.zeros(6); S = np.zeros((6, 6))
    mu[Oi] = mO; mu[Bi] = ra + rG @ mO
    S[np.ix_(Oi, Oi)] = SO
    S[np.ix_(Bi, Oi)] = rG @ SO; S[np.ix_(Oi, Bi)] = (rG @ SO).T
    S[np.ix_(Bi, Bi)] = rC + rG @ SO @ rG.T
    return mu, S


x, w = hermegauss(6); w = w / w.sum()


def gh_nodes(m_, S_):
    L_ = np.linalg.cholesky(S_)
    for idx in itertools.product(range(6), repeat=len(m_)):
        yield m_ + L_ @ x[list(idx)], np.prod(w[list(idx)])


maxdiff = 0.0
EFB1 = EFB2 = EHbar = EKL = 0.0
for ym, wt in gh_nodes(mO, SO):
    v1, v2 = F_B_two_ways(ym, ra, rG, rC)
    u1, _ = F_B_two_ways(ym, ra2, rG2, rC2)
    maxdiff = max(maxdiff, abs(v1 - v2))
    EFB1 += wt * v1; EFB2 += wt * u1
    y = np.zeros(6); y[Oi] = ym
    EHbar += wt * H_terms(y, recO)
    EKL += wt * kl_gauss(ra + rG @ ym, rC, aPi + GPi @ ym, CPi)
print("[4] local energy form max |(-logZ+KL) - (KL_base + E H_B)| :", maxdiff)
muQ, SQ = joint_from(mO, SO, ra, rG, rC)
muQ2, SQ2 = joint_from(mO, SO, ra2, rG2, rC2)
F1 = -logp + kl_gauss(muQ, SQ, mp, Sp)
F2 = -logp + kl_gauss(muQ2, SQ2, mp, Sp)
print("    potential identity: dF - E dF_B              :", (F2 - F1) - (EFB2 - EFB1), " dF =", F2 - F1)
dec = kl_gauss(mO, SO, M['mu0'][Oi], M['S0'][np.ix_(Oi, Oi)]) + EHbar + EFB1
print("    decomposition residual                       :", F1 - dec)
print("    posterior block chain residual               :",
      kl_gauss(muQ, SQ, mp, Sp) - kl_gauss(mO, SO, mp[Oi], Sp[np.ix_(Oi, Oi)]) - EKL)
muE, SE = joint_from(mO, SO, aPi, GPi, CPi)
opt = kl_gauss(muE, SE, mp, Sp)
best_alt = min(kl_gauss(*joint_from(mO, SO, aPi + rng.normal(size=2) * 0.2,
                                    GPi + rng.normal(size=(2, 4)) * 0.2, CPi + 0.05 * rand_spd(2, 0.3)), mp, Sp)
               for _ in range(2000))
print("[5] E-coordinate: KL(kappa R||Pi) - KL(R||Pi_-B):", opt - kl_gauss(mO, SO, mp[Oi], Sp[np.ix_(Oi, Oi)]),
      "; best of 2000 perturbed kernels minus optimum:", best_alt - opt)

mus = [rng.normal(size=2) for _ in range(3)]; Ss = [rand_spd(2, 0.4) for _ in range(3)]


def prod_law(mus, Ss):
    mu = np.concatenate(mus); S = np.zeros((6, 6))
    for bl, S_ in zip(blocks, Ss):
        S[np.ix_(bl, bl)] = S_
    return mu, S


def F_prod(mus, Ss):
    mu, S = prod_law(mus, Ss)
    return -logp + kl_gauss(mu, S, mp, Sp)


def Fbar2(mus, Ss):
    mO_ = np.concatenate([mus[0], mus[2]]); SO_ = np.zeros((4, 4))
    SO_[:2, :2] = Ss[0]; SO_[2:, 2:] = Ss[2]
    return sum(wt * F_B_two_ways(ym, mus[1], np.zeros((2, 4)), Ss[1])[0] for ym, wt in gh_nodes(mO_, SO_))


h = 1e-5
gdiff = []
for j in range(2):
    e = np.zeros(2); e[j] = h
    mp_ = [mus[0], mus[1] + e, mus[2]]; mm_ = [mus[0], mus[1] - e, mus[2]]
    gdiff.append((F_prod(mp_, Ss) - F_prod(mm_, Ss)) / (2 * h) - (Fbar2(mp_, Ss) - Fbar2(mm_, Ss)) / (2 * h))
print("[6] product family: grad_mu2 F - grad_mu2 Fbar_2:", np.array(gdiff))
alt = [mus[0], rng.normal(size=2), mus[2]]; altS = [Ss[0], rand_spd(2, 0.4), Ss[2]]
print("    exact potential game: dF - dFbar_2           :",
      (F_prod(alt, altS) - F_prod(mus, Ss)) - (Fbar2(alt, altS) - Fbar2(mus, Ss)))

th_old, th_new = 0.4, -0.3
Mo, Mn = model(th_old), model(th_new)
lpo, mpo, Spo = evidence_posterior(Mo, o); lpn, mpn, Spn = evidence_posterior(Mn, o)
Lo = lpo - kl_gauss(mq, Sq, mpo, Spo); Ln = lpn - kl_gauss(mq, Sq, mpn, Spn)
fo, ro_ = factor_terms(Mo, o, mq, Sq); fn, rn_ = factor_terms(Mn, o, mq, Sq)
print("[7] acceptance: (Ln-Lo) - E_Q log(p_new/p_old)  :", (Ln - Lo) - ((fo.sum() + ro_.sum()) - (fn.sum() + rn_.sum())))
th = 3.0; ev = []
for it in range(25):
    Mt = model(th); lp, mpt, Spt = evidence_posterior(Mt, o); ev.append(lp)
    th = mpt[1]
print("    EM: min increment of log evidence over 24 steps:", np.diff(ev).min(), " final theta:", th)

J = np.array([[1, .6, .6], [.6, 1, .6], [.6, .6, 1.]]); mstar = np.zeros(3)
Fq = lambda m: 0.5 * (m - mstar) @ J @ (m - mstar)
m_j = np.array([1.0, -0.5, 0.3]); m_g = m_j.copy(); Fj = [Fq(m_j)]; Fg = [Fq(m_g)]
for _ in range(10):
    m_j = mstar - (J - np.diag(np.diag(J))) @ (m_j - mstar) / np.diag(J); Fj.append(Fq(m_j))
    for i in range(3):
        m_g[i] = mstar[i] - (J[i] @ (m_g - mstar) - J[i, i] * (m_g[i] - mstar[i])) / J[i, i]
    Fg.append(Fq(m_g))
print("[8] Jacobi mean-field F trace:", np.round(Fj[:6], 4), "...", np.round(Fj[-1], 3))
print("    Gauss-Seidel F trace     :", np.round(Fg[:6], 6))
