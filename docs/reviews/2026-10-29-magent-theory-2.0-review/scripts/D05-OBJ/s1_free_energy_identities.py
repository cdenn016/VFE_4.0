"""D05-OBJ s1: exact checks of prop:dyn-global-local, prop:dyn-per-agent, prop:dyn-label-model.

Finite alphabets, random positive tables, natural logs.  Every free energy is computed
directly as E_Q[log Q - log P(.,o)] on the full product space and compared with the
closed forms of the manuscript.  Single-threaded, < 1 s.
"""
import numpy as np
rng = np.random.default_rng(20261029)

def simplex(*shape):
    x = rng.random(shape) + 0.05
    return x / x.sum(axis=-1, keepdims=True)

def kl(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = a > 0
    return float(np.sum(a[m] * (np.log(a[m]) - np.log(b[m]))))

print("=== A. prop:dyn-global-local (3 blocks, sizes 3,2,4; record likelihood) ===")
sizes = (3, 2, 4)
P0 = rng.random(sizes) + 0.02; P0 /= P0.sum()
lik = rng.random(sizes) * 0.9 + 0.05
Zo = float((lik * P0).sum()); Pi = lik * P0 / Zo
Q = [simplex(n) for n in sizes]
def joint(Q): return np.einsum('a,b,c->abc', *Q)
def F(Q):
    QQ = joint(Q)
    return kl(QQ, P0) - float((QQ * np.log(lik)).sum())
f_split = F(Q); f_post = kl(joint(Q), Pi) - np.log(Zo)
print("eq:dyn-fixed-vfe residual:", abs(f_split - f_post))
logp = np.log(Pi)
def ebar(i, Q):
    others = [j for j in range(3) if j != i]
    t = np.moveaxis(logp, i, 0)
    rest = [Q[j] for j in others]
    return np.einsum('abc,b,c->a', t, *rest)
worst = 0.0
for i in range(3):
    e = ebar(i, Q); Zi = np.exp(e).sum(); Qs = np.exp(e) / Zi
    ent = sum(float(np.sum(Q[j]*np.log(Q[j]))) for j in range(3) if j != i)
    rhs = kl(Q[i], Qs) - np.log(Zi) + ent - np.log(Zo)
    worst = max(worst, abs(F(Q) - rhs))
print("eq:dyn-local-free-energy max residual over i:", worst)
Qc = [q.copy() for q in Q]; vals = [F(Qc)]
for sweep in range(6):
    for i in range(3):
        e = ebar(i, Qc); Qc[i] = np.exp(e) / np.exp(e).sum(); vals.append(F(Qc))
print("block-update schedule, first F values:", np.round(vals[:7], 6), " largest step (must be <= 0):", np.max(np.diff(vals)))
e = ebar(0, Qc); Qs = np.exp(e)/np.exp(e).sum(); fstar = F([Qs, Qc[1], Qc[2]])
comp = min(F([simplex(sizes[0]), Qc[1], Qc[2]]) for _ in range(2000))
print("F at Q_0^*:", fstar, " best of 2000 random competitors:", comp, " competitor >= minimizer:", comp >= fstar)
A, B = simplex(2), simplex(2); t = 0.37
lhs = ebar(0, [Q[0], t*A+(1-t)*B, Q[2]]); rhs = t*ebar(0, [Q[0], A, Q[2]]) + (1-t)*ebar(0, [Q[0], B, Q[2]])
print("ebar_0 affine in Q_1, residual:", float(np.max(np.abs(lhs - rhs))))

print("\n=== B. prop:dyn-per-agent (|M|=2, |K|=3, J in {self,p1,p2}, L in {self,p1}, ell(o|k,m)) ===")
nM, nK = 2, 3
r = simplex(nM); G = simplex(nM, nK)
nu_q = simplex(2, nK)
nu_s = simplex(1, nM)
piq, pis = simplex(3), simplex(2)
ell = rng.random((nK, nM)) * 0.9 + 0.05
kappa = np.zeros((3, nM, nK)); kappa[0] = G; kappa[1] = nu_q[0][None, :]; kappa[2] = nu_q[1][None, :]
rho = np.vstack([r, nu_s])
P = np.einsum('l,lm,j,jmk,km->mkjl', pis, rho, piq, kappa, ell)
Po = P.sum(); Ppost = P / Po
s, q, be, ga = simplex(nM), simplex(nK), simplex(3), simplex(2)
def Fi(be, ga):
    Qa = np.einsum('m,k,j,l->mkjl', s, q, be, ga); return kl(Qa, Ppost) - np.log(Po)
F_direct = Fi(be, ga)
F_formula = (sum(ga[l]*kl(s, rho[l]) for l in range(2)) + kl(ga, pis)
             + sum(be[j]*sum(s[m]*kl(q, kappa[j, m]) for m in range(nM)) for j in range(3))
             + kl(be, piq) + float(-np.einsum('k,m,km->', q, s, np.log(ell))))
print("eq:dyn-per-agent-free-energy residual:", abs(F_direct - F_formula))
cq = np.array([sum(s[m]*kl(q, kappa[j, m]) for m in range(nM)) for j in range(3)])
cs = np.array([kl(s, rho[l]) for l in range(2)])
bstar = piq*np.exp(-cq); bstar /= bstar.sum(); gstar = pis*np.exp(-cs); gstar /= gstar.sum()
gap_id = Fi(be, ga) - Fi(bstar, gstar) - kl(be, bstar) - kl(ga, gstar)
print("row-gap identity residual F(b,g)-F(b*,g*)-KL(b||b*)-KL(g||g*):", gap_id)
print("peer costs vs KL(q||nu):", np.round(cq[1:], 12), np.round([kl(q, nu_q[0]), kl(q, nu_q[1])], 12))
PJ_o = Ppost.sum(axis=(0, 1, 3)); PL_o = Ppost.sum(axis=(0, 1, 2))
print("Gibbs content row b*:", np.round(bstar, 4), "  exact posterior P(J|o):", np.round(PJ_o, 4))
print("Gibbs model row   g*:", np.round(gstar, 4), "  exact posterior P(L|o):", np.round(PL_o, 4))
rt = pis @ rho
Gt = np.einsum('j,jmk->mk', piq, kappa)
Pm = np.einsum('m,mk,km->mk', rt, Gt, ell)
print("label marginal equals rt*Gt*ell (def:generative-model form), residual:", float(np.max(np.abs(Pm - P.sum(axis=(2, 3))))))
Fm = kl(np.einsum('m,k->mk', s, q), Pm / Pm.sum()) - np.log(Pm.sum())
PJ_mk = np.einsum('j,jmk->mkj', piq, kappa); PJ_mk = PJ_mk / PJ_mk.sum(axis=2, keepdims=True)
PL_m = (pis[:, None]*rho).T; PL_m = PL_m / PL_m.sum(axis=1, keepdims=True)
gap = sum(s[m]*q[k]*kl(bstar, PJ_mk[m, k]) for m in range(nM) for k in range(nK)) + sum(s[m]*kl(gstar, PL_m[m]) for m in range(nM))
print("F_i(b*,g*) - F_marg(s x q):", Fi(bstar, gstar) - Fm, "  expected label gap:", gap)

print("\n=== C. prop:dyn-label-model (3 agents, |K|=3, nonself labels, T=0.7) ===")
N, nK2, T = 3, 3, 0.7
rr = [simplex(nK2) for _ in range(N)]; qq = [simplex(nK2) for _ in range(N)]
src = {i: [j for j in range(N) if j != i] for i in range(N)}
pi = {i: simplex(len(src[i])) for i in range(N)}; bet = {i: simplex(len(src[i])) for i in range(N)}
K = {(i, j): simplex(nK2, nK2) for i in range(N) for j in src[i]}
Fmod = sum(kl(qq[i], rr[i]) for i in range(N)) + sum(kl(bet[i], pi[i]) for i in range(N))
for i in range(N):
    for a, j in enumerate(src[i]):
        Fmod += bet[i][a] * float(-np.einsum('w,v,vw->', qq[i], qq[j], np.log(K[(i, j)])) / T)
E = sum(kl(qq[i], rr[i]) for i in range(N)); Gsum = 0.0; Hsum = 0.0; Dm = {}; Gm = {}
for i in range(N):
    Hsum += float(-np.sum(qq[i]*np.log(qq[i])))
    for a, j in enumerate(src[i]):
        kij = K[(i, j)]; b = qq[j] @ kij; D = kl(qq[i], b)
        Gij = sum(qq[i][w] * kl(qq[j], kij[:, w]*qq[j]/b[w]) for w in range(nK2))
        Dm[(i, j)] = D; Gm[(i, j)] = Gij; E += bet[i][a]*D; Gsum += bet[i][a]*Gij
    E += T*kl(bet[i], pi[i])
rhs = E + Hsum + (T-1)*sum(kl(qq[i], rr[i]) for i in range(N)) + Gsum
print("eq:dyn-label-model residual |T F - rhs|:", abs(T*Fmod - rhs))
cost = np.array([Dm[(0, j)] + Gm[(0, j)] for j in src[0]])
bopt = pi[0]*np.exp(-cost/T); bopt /= bopt.sum()
bG = pi[0]*np.exp(-np.array([Dm[(0, j)] for j in src[0]])/T); bG /= bG.sum()
print("recognition-optimal row:", np.round(bopt, 5), " Gibbs row:", np.round(bG, 5), " G_0j:", np.round([Gm[(0, j)] for j in src[0]], 5))
