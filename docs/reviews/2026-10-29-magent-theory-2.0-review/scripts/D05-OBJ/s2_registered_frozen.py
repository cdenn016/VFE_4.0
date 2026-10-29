"""D05-OBJ s2: brute-force checks of thm:dyn-registered (i)-(iv) and cor:dyn-frozen-source.

Binary alphabets, trivial model sector, n = 4 messages per batch (T = 1/4), three agents,
a law-dependent channel on one edge.  Registration probabilities are computed by summing
over all 2^n message sequences, not from the type formula.  Single-threaded, < 2 s.
"""
import numpy as np
from itertools import product
from math import comb, log

n = 4; T = 1.0 / n; N = 3
types = [np.array([k/n, 1-k/n]) for k in (1, 2, 3)]          # interior n-types on {0,1}
Rref = [np.array([0.3, 0.7]), np.array([0.55, 0.45]), np.array([0.8, 0.2])]   # R_i = G_i (trivial model sector)
src = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
pi = {0: np.array([0.6, 0.4]), 1: np.array([0.3, 0.7]), 2: np.array([0.5, 0.5])}

def kl(a, b): return float(np.sum(a*np.log(a/b)))
def H(p): return float(-np.sum(p[p > 0]*np.log(p[p > 0])))

def channel(i, j, x):
    # law-dependent binary channel: identity mixed with uniform, mixing weight depends on the configuration
    eps = 0.1 + 0.5*x[i][0]*x[(i+1) % N][0] if (i, j) == (0, 1) else 0.2
    return (1-eps)*np.eye(2) + eps*0.5*np.ones((2, 2))

def nu(i, j, x): return x[j] @ channel(i, j, x)

def p_register(target, law):
    # brute force: probability that n iid draws from law have empirical type equal to target
    tot = 0.0
    for seq in product((0, 1), repeat=n):
        c = np.bincount(seq, minlength=2)/n
        if np.allclose(c, target): tot += float(np.prod([law[s] for s in seq]))
    return tot

def size_type(p):
    k = int(round(p[0]*n)); return comb(n, k)

def c_n(p): return H(p) - log(size_type(p))/n

Xn = list(product(types, repeat=N)); nX = len(Xn)
rng = np.random.default_rng(7)
max_i = max_ii = max_iii = max_iv = 0.0; bound_ok = True
logPo_terms = []
for x in Xn:
    Pox = 1.0
    for i in range(N):
        Pox *= p_register(x[i], Rref[i]) * sum(pi[i][a]*p_register(x[i], nu(i, j, x)) for a, j in enumerate(src[i]))
    logPo_terms.append(Pox)
Po = sum(logPo_terms)/nX
for idx, x in enumerate(Xn):
    beta = {i: (lambda v: v/v.sum())(rng.random(2)+0.1) for i in range(N)}
    F = log(nX); Eval = 0.0; Cn = 0.0; Ered = 0.0
    for i in range(N):
        p0 = p_register(x[i], Rref[i])
        f0 = size_type(x[i])*np.exp(-n*(H(x[i]) + kl(x[i], Rref[i])))
        max_i = max(max_i, abs(p0 - f0)/f0)
        pq = np.array([p_register(x[i], nu(i, j, x)) for j in src[i]])
        D = np.array([kl(x[i], nu(i, j, x)) for j in src[i]])
        fq = size_type(x[i])*np.exp(-n*(H(x[i]) + D))
        max_i = max(max_i, float(np.max(np.abs(pq - fq)/fq)))
        post = pi[i]*pq; post /= post.sum()
        gibbs = pi[i]*np.exp(-D/T); gibbs /= gibbs.sum()
        max_ii = max(max_ii, float(np.max(np.abs(post - gibbs))))
        F += kl(beta[i], pi[i]) - log(p0) - float(np.sum(beta[i]*np.log(pq)))
        Eval += kl(x[i], Rref[i]) + float(np.sum(beta[i]*D)) + T*kl(beta[i], pi[i])
        Cn += 2*c_n(x[i])            # reference-batch target q_i (x) s_i = q_i, and belief-batch target q_i
        Ered += kl(x[i], Rref[i]) - T*log(float(np.sum(pi[i]*np.exp(-D/T))))
    max_iii = max(max_iii, abs(T*F - (Eval + Cn + T*log(nX))))
    max_iv = max(max_iv, abs(Ered - (-T*log(logPo_terms[idx]) - Cn)))
    bound_ok &= (Ered + Cn + T*log(nX) >= -T*log(Po) - 1e-12)
print("thm:dyn-registered, n=4, |X_n| =", nX)
print("(i)   max relative error of registration probability vs |T_q| exp(-n[H+D]):", max_i)
print("(ii)  max |posterior of label - Gibbs row at T=1/n|:", max_ii)
print("(iii) max |T F - (E + C_n + T log|X_n|)|:", max_iii)
print("(iv)  max |E_red - (-T log P(o|x) - C_n)|:", max_iv, "  evidence bound holds at every x:", bound_ok)
cvals = [c_n(p) for p in types]
print("c_n on the interior types:", np.round(cvals, 4), " upper bound |supp| log(n+1)/n =", round(2*log(n+1)/n, 4))

print("\ncor:dyn-frozen-source, N = 5 copies, three frozen sources on a ternary alphabet")
Nc = 5; Tc = 1.0/Nc
q = np.array([0.2, 0.4, 0.4]); nus = [np.array([0.3, 0.3, 0.4]), np.array([0.1, 0.6, 0.3]), np.array([0.5, 0.25, 0.25])]
pic = np.array([0.2, 0.5, 0.3]); bc = np.array([0.3, 0.3, 0.4])
lhs = 0.0
for a in range(3):
    for seq in product(range(3), repeat=Nc):
        Qv = bc[a]*np.prod(q[list(seq)]); Pv = pic[a]*np.prod(nus[a][list(seq)])
        lhs += Qv*np.log(Qv/Pv)
lhs *= Tc
rhs = float(sum(bc[a]*kl(q, nus[a]) for a in range(3))) + Tc*kl(bc, pic)
print("eq:dyn-frozen-source residual:", abs(lhs - rhs))
seq = (0, 1, 1, 2, 2)            # a sequence of type q = (1/5, 2/5, 2/5)
post = np.array([pic[a]*np.prod(nus[a][list(seq)]) for a in range(3)]); post /= post.sum()
qt = np.bincount(seq, minlength=3)/Nc
g = pic*np.exp(-np.array([kl(qt, nus[a]) for a in range(3)])/Tc); g /= g.sum()
print("posterior of J given a sequence of type (1/5,2/5,2/5) vs Gibbs row at T=1/5, max diff:", float(np.max(np.abs(post - g))))
