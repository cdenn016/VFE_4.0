"""XC3-SCOPE deep checks of chapter 03 identities that relate the three objectives.
(1) prop:dyn-label-model: T F = E + sum_i[H(q_i)+(T-1)KL(q_i||r_i)] + sum_ij beta_ij G_ij.
(2) thm:dyn-registered (i) by brute-force sequence enumeration; (iii) and (iv) on a small type lattice.
(3) prop:dyn-global-local: F(Q;o) = KL(Q_i||Q_i*) - log Z_i + sum_{j!=i} int a_j log a_j - log Z_o.
(4) prop:dyn-nonreciprocal: eigenvalues of the three-cycle per-agent linearization.
Single-threaded, < 5 s.  Numerical agreement supports, but does not prove, the identities."""
import itertools, math
import numpy as np
rng = np.random.default_rng(0)
def kl(a, b): a = np.asarray(a, float); b = np.asarray(b, float); return float(np.sum(a*np.log(a/b)))
def H(a): a = np.asarray(a, float); return float(-np.sum(a*np.log(a)))

# (1) label model --------------------------------------------------------------
def label_model_check(N=3, A=3, T=0.7):
    q = [rng.dirichlet(np.ones(A)) for _ in range(N)]
    r = [rng.dirichlet(np.ones(A)) for _ in range(N)]
    K = {(i, j): rng.dirichlet(np.ones(A), size=A) for i in range(N) for j in range(N) if i != j}  # K[v,w]
    pi = {i: rng.dirichlet(np.ones(N-1)) for i in range(N)}
    beta = {i: rng.dirichlet(np.ones(N-1)) for i in range(N)}
    # free energy of the normalized label model, product recognition prod q_i x prod beta_i
    F = sum(kl(q[i], r[i]) for i in range(N)) + sum(kl(beta[i], pi[i]) for i in range(N))
    E = sum(kl(q[i], r[i]) for i in range(N)); Gsum = 0.0; Esrc = 0.0
    for i in range(N):
        js = [j for j in range(N) if j != i]
        rows = 0.0
        for a, j in enumerate(js):
            k = K[(i, j)]                        # k_ij(w|v) = K[v,w]
            expneglogk = -sum(q[i][w]*q[j][v]*math.log(k[v, w]) for v in range(A) for w in range(A))
            F += beta[i][a] * expneglogk / T     # success prob k^(1/T): -log lik = -(1/T) log k
            nu = q[j] @ k
            D = kl(q[i], nu)
            G = 0.0
            for w in range(A):
                pw = q[j]*k[:, w]/nu[w]          # source conditional q_j^w
                G += q[i][w]*kl(q[j], pw)
            rows += beta[i][a]*D; Gsum += beta[i][a]*G
        E += rows + T*kl(beta[i], pi[i])
    rhs = E + sum(H(q[i]) + (T-1)*kl(q[i], r[i]) for i in range(N)) + Gsum
    return T*F - rhs

# (2) registered agreement -------------------------------------------------------
def type_prob_bruteforce(qcounts, nu):
    n = sum(qcounts); A = len(nu); tot = 0.0
    for seq in itertools.product(range(A), repeat=n):
        c = [seq.count(a) for a in range(A)]
        if c == list(qcounts): tot += math.prod(nu[s] for s in seq)
    return tot
def type_prob_formula(qcounts, nu):
    n = sum(qcounts); q = np.array(qcounts)/n
    size = math.factorial(n)/math.prod(math.factorial(c) for c in qcounts)
    Hq = -sum(x*math.log(x) for x in q if x > 0)
    D = sum(x*math.log(x/nu[a]) for a, x in enumerate(q) if x > 0)
    return size*math.exp(-n*(Hq + D)), size, Hq
def registered_check(n=4):
    nu = [0.3, 0.7]; errs = []
    for c0 in range(n+1):
        bf = type_prob_bruteforce([c0, n-c0], nu); fo = type_prob_formula([c0, n-c0], nu)[0]
        errs.append(abs(bf-fo))
    # small two-agent lattice, belief sector only, model space a point; sources: self (identity) and peer (identity)
    T = 1.0/n; G = [np.array([0.4, 0.6]), np.array([0.55, 0.45])]; pi = [np.array([0.5, 0.5])]*2
    types = [np.array([c, n-c])/n for c in range(1, n)]          # interior types
    X = list(itertools.product(range(len(types)), repeat=2))
    def logP_o_given(x, beta=None):
        qs = [types[x[0]], types[x[1]]]; lp = 0.0; parts = []
        for i in range(2):
            ref, _, _ = type_prob_formula(list((qs[i]*n).round().astype(int)), G[i])
            lp += math.log(ref)
            comps = []
            for src in (i, 1-i):
                p, _, _ = type_prob_formula(list((qs[i]*n).round().astype(int)), qs[src])
                comps.append(p)
            parts.append(comps)
            lp += math.log(pi[i][0]*comps[0] + pi[i][1]*comps[1])
        return lp, parts, qs
    worst3 = 0.0; worst4 = 0.0; logPo_list = []
    for x in X:
        lp, parts, qs = logP_o_given(x)
        logPo_list.append(lp)
        Cn = 0.0; E0 = 0.0; Ered = 0.0
        for i in range(2):
            cnt = list((qs[i]*n).round().astype(int))
            _, size, Hq = type_prob_formula(cnt, G[i])
            cn = Hq - math.log(size)/n
            Cn += 2*cn                                           # reference batch + belief batch (point model sector)
            E0 += kl(qs[i], G[i])
            Z = pi[i][0]*math.exp(-0.0/T) + pi[i][1]*math.exp(-kl(qs[i], qs[1-i])/T)
            Ered += -T*math.log(Z)
        Ered += E0
        beta = [rng.dirichlet([1, 1]) for _ in range(2)]
        Ex = E0 + sum(beta[i][1]*kl(qs[i], qs[1-i]) + T*kl(beta[i], pi[i]) for i in range(2))
        F = math.log(len(X)) + sum(kl(beta[i], pi[i]) for i in range(2))
        F += -sum(math.log(type_prob_formula(list((qs[i]*n).round().astype(int)), G[i])[0]) for i in range(2))
        F += -sum(beta[i][0]*math.log(parts[i][0]) + beta[i][1]*math.log(parts[i][1]) for i in range(2))
        worst3 = max(worst3, abs(T*F - (Ex + Cn + T*math.log(len(X)))))
        worst4 = max(worst4, abs(Ered - (-T*lp - Cn)))
    logPo = math.log(np.mean(np.exp(logPo_list)))
    bound_ok = all(Ered_ + 0 >= 0 for Ered_ in [0])
    return max(errs), worst3, worst4

# (3) global/local mean-field identity ------------------------------------------
def global_local_check(sizes=(3, 4, 2)):
    p = rng.dirichlet(np.ones(np.prod(sizes))).reshape(sizes)   # posterior Pi_o density (counting ref)
    Zo = 0.37                                                    # arbitrary evidence constant
    a = [rng.dirichlet(np.ones(s)) for s in sizes]
    Q = np.einsum('i,j,k->ijk', *a)
    F = float(np.sum(Q*np.log(Q/p))) - math.log(Zo)
    i = 1
    ebar = np.einsum('ijk,i,k->j', np.log(p), a[0], a[2])
    Zi = float(np.sum(np.exp(ebar))); Qs = np.exp(ebar)/Zi
    rhs = kl(a[i], Qs) - math.log(Zi) + sum(float(np.sum(a[j]*np.log(a[j]))) for j in (0, 2)) - math.log(Zo)
    return F - rhs

# (4) three-cycle eigenvalues ---------------------------------------------------
def three_cycle(w=0.8):
    P = np.roll(np.eye(3), 1, axis=1)
    A = (-(1+w)*np.eye(3) + w*P)/(1+w)
    ev = np.sort_complex(np.linalg.eigvals(A))
    pred = np.sort_complex(np.array([-1/(1+w), (-1-1.5*w+1j*math.sqrt(3)*w/2)/(1+w), (-1-1.5*w-1j*math.sqrt(3)*w/2)/(1+w)]))
    return np.max(np.abs(ev-pred))

if __name__ == "__main__":
    print("(1) label-model residual:", [label_model_check(T=T) for T in (0.3, 1.0, 2.5)])
    e1, e3, e4 = registered_check(4)
    print("(2) registered: type-class formula max err %.2e, (iii) residual %.2e, (iv) residual %.2e" % (e1, e3, e4))
    print("(3) global/local residual:", global_local_check())
    print("(4) three-cycle eigenvalue error:", three_cycle())
