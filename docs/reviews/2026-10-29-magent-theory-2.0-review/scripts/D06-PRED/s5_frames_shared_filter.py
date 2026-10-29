r"""D06-PRED s5: rebasing lemma, shared-model condition, carried-label filter and its failure cases.

  rebasing lemma (cmem frame section)   (h D h^T)_{ij} = sum_k h_ik h_jk d_k; monomial h preserve diagonality,
                                        a non-monomial h does not
  shared-model condition                record kernels N(.;u,C_i), N(.;u,C_j) with C_i != C_j differ for every
                                        content bijection psi (variance is psi-independent); augmented record
                                        (O, C) with an ancillary law for C gives one shared model and the same
                                        posterior as the C-indexed mechanism
  cmem:history / cmem:filter            brute-force enumeration of label paths vs the forward recursion,
                                        and the one-step predictive law of the next record
  failure cases                         (i) persistent cue observed twice, (ii) deterministic copy O_1 = O_0,
                                        (iii) the next cue is the previous result (a composition through depth):
                                        identical label filters w_0 with different next-record laws
  Dirichlet record                      normalizer independent of b; posterior propto pi_b v_b^theta;
                                        equals the HMM prediction for uniform pi and theta = 1
"""
from __future__ import annotations

import itertools

import numpy as np
import sympy as sp
from scipy import stats

rng = np.random.default_rng(42)

# rebasing lemma
h = sp.Matrix(3, 3, sp.symbols("h0:9"))
d = sp.symbols("d0:3", positive=True)
Dm = sp.diag(*d)
M = h * Dm * h.T
print("(h D h^T)_01 =", sp.expand(M[0, 1]))
perm = np.array([[0, 2.0, 0], [0, 0, -0.5], [3.0, 0, 0]])
gen = rng.normal(size=(3, 3))
Dn = np.diag([1.0, 2.0, 3.0])
offd = lambda A: np.abs(A - np.diag(np.diag(A))).max()
print("monomial h: max off-diagonal of h D h^T =", offd(perm @ Dn @ perm.T), "; generic h:", offd(gen @ Dn @ gen.T))

# shared model condition: Gaussian record kernels with different variances never agree
print("record-kernel variances C_i = 1, C_j = 2 are psi-independent; KL(N(0,1)||N(0,2)) =",
      0.5 * (1 / 2 - 1 + np.log(2)))
# augmented record (O, C) with ancillary kappa(dC): label posterior equals the C-indexed one
pis = np.array([0.3, 0.7]); nus = np.array([-1.0, 1.0]); Ls = np.array([1.0, 0.5])
kappa = {1.0: 0.4, 2.0: 0.6}
for Ci, xi in [(1.0, 0.3), (2.0, -0.8)]:
    indexed = pis * stats.norm.pdf(xi, nus, np.sqrt(Ls + Ci)); indexed /= indexed.sum()
    aug = pis * kappa[Ci] * stats.norm.pdf(xi, nus, np.sqrt(Ls + Ci)); aug /= aug.sum()
    print(f"augmented-record posterior residual (C={Ci}) =", np.abs(indexed - aug).max())

# carried-label filter: Gaussian bank, fresh emissions
Mlab = 3
pi_hat = np.array([0.2, 0.5, 0.3])
Tm = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5]])
nu_ = np.array([-1.5, 0.0, 2.0]); L_ = np.array([0.6, 1.2, 0.4])
Cs = [0.5, 1.0, 0.3, 2.0]
obs = [0.4, -1.1, 1.9, 0.2]
n = len(obs)
Z = lambda b, o, C: stats.norm.pdf(o, nu_[b], np.sqrt(L_[b] + C))
# recursion
w = pi_hat * np.array([Z(b, obs[0], Cs[0]) for b in range(Mlab)]); w /= w.sum()
ws = [w]
for l in range(1, n):
    vpred = ws[-1] @ Tm
    w = vpred * np.array([Z(b, obs[l], Cs[l]) for b in range(Mlab)]); w /= w.sum()
    ws.append(w)
# brute force over label paths
err = 0.0
for l in range(n):
    post = np.zeros(Mlab)
    for path in itertools.product(range(Mlab), repeat=l + 1):
        pr = pi_hat[path[0]] * np.prod([Tm[path[k - 1], path[k]] for k in range(1, l + 1)])
        pr *= np.prod([Z(path[k], obs[k], Cs[k]) for k in range(l + 1)])
        post[path[-1]] += pr
    err = max(err, np.abs(post / post.sum() - ws[l]).max())
print("cmem:filter vs brute-force enumeration, max error =", err)
# one-step predictive density of O_{l+1} given O_{0:l}: (w_l T) . Z
onew, Cnew = 0.7, 0.8
pred_rec = (ws[-1] @ Tm) @ np.array([Z(b, onew, Cnew) for b in range(Mlab)])
num = den = 0.0
for path in itertools.product(range(Mlab), repeat=n + 1):
    pr = pi_hat[path[0]] * np.prod([Tm[path[k - 1], path[k]] for k in range(1, n + 1)])
    pr *= np.prod([Z(path[k], obs[k], Cs[k]) for k in range(n)])
    den += pr
    num += pr * Z(path[n], onew, Cnew)
print("next-record predictive density: recursion vs brute force, error =", abs(pred_rec - num / den))

# failure case (i): persistent cue U observed twice with independent noise, labels fixed
pi2 = np.array([0.5, 0.5]); nu2 = np.array([0.0, 1.0]); L2, C2 = 1.0, 1.0
o0 = o1 = 1.0
exact = pi2 * np.array([stats.multivariate_normal.pdf([o0, o1], [m, m], [[L2 + C2, L2], [L2, L2 + C2]]) for m in nu2])
exact /= exact.sum()
fresh = pi2 * np.array([stats.norm.pdf(o0, m, np.sqrt(L2 + C2)) * stats.norm.pdf(o1, m, np.sqrt(L2 + C2)) for m in nu2])
fresh /= fresh.sum()
print("persistent cue: exact P(A|o0,o1) =", exact.round(6), " fresh-emission filter =", fresh.round(6))
# failure case (ii): deterministic copy O_1 = O_0
once = pi2 * np.array([stats.norm.pdf(o0, m, np.sqrt(L2 + C2)) for m in nu2]); once /= once.sum()
twice = pi2 * np.array([stats.norm.pdf(o0, m, np.sqrt(L2 + C2)) ** 2 for m in nu2]); twice /= twice.sum()
print("deterministic copy: exact (= one record) =", once.round(6), " replicate likelihood =", twice.round(6))
# failure case (iii): the next cue is the previous result, U_1 = V_0, label fixed
# identical cue laws make w_0 = pi for every o_0; the next-record law still depends on o_0
nuc = np.array([0.0, 0.0]); Lc = np.array([1.0, 1.0]); omc = np.array([1.0, -1.0])
Fc = np.array([2.0, -0.5]); Qc = np.array([0.3, 0.3]); Cc, Cn = 1.0, 0.5
for o_first in [-1.0, 1.0]:
    w0 = pi2 * stats.norm.pdf(o_first, nuc, np.sqrt(Lc + Cc)); w0 /= w0.sum()
    mX = nuc + Lc / (Lc + Cc) * (o_first - nuc); PX = Lc * Cc / (Lc + Cc)
    mY = omc + Fc * (mX - nuc); PY = Qc + Fc ** 2 * PX
    pred_mean = float(w0 @ mY)
    fresh_mean = float(w0 @ nuc)  # what a fresh cue from mu_a would predict for O_1
    print(f"o_0={o_first}: w_0 = {w0.round(6)}, exact E[O_1|o_0] = {pred_mean:.6f}, "
          f"fresh-emission E[O_1|o_0] = {fresh_mean:.6f}")

# Dirichlet record
alpha, theta = 0.7, 1.3
vint = np.array([0.2, 0.5, 0.3])
norms = []
dens = []
for b in range(3):
    par = alpha * np.ones(3) + theta * np.eye(3)[b]
    from scipy.special import gammaln
    norms.append(gammaln(par.sum()) - gammaln(par).sum())
    dens.append(stats.dirichlet.pdf(vint, par))
print("Dirichlet log-normalizers (must be equal) =", np.round(norms, 12))
pri = np.array([0.1, 0.6, 0.3])
post = pri * np.array(dens); post /= post.sum()
target = pri * vint ** theta; target /= target.sum()
print("Dirichlet posterior vs pi_b v_b^theta, error =", np.abs(post - target).max())
post_u = np.ones(3) / 3 * np.array([stats.dirichlet.pdf(vint, alpha * np.ones(3) + np.eye(3)[b]) for b in range(3)])
post_u /= post_u.sum()
print("uniform pi, theta = 1: posterior - v =", np.abs(post_u - vint).max())
