"""XC3-SCOPE checks of chapter 18 statements relating objectives.
(1) it2:potential-forces (i): at fixed covariances the tangent reciprocal potential differs from the
    widened global energy E^S (R_e -> S_{j,e}/beta_e, o_e = 0) by a mean-independent constant.
    The edge expectation over the dictionary law is computed by exact Gauss-Hermite quadrature,
    and KL(r||pi_omega) by the full Gaussian KL formula, independently of the proof's algebra.
(2) it2:ring: eigenvalues of the per-agent mean flow on a directed ring.
Single-threaded, < 5 s."""
import itertools, numpy as np
rng = np.random.default_rng(3)
K = 2; d = K*K
def spd(n, s=1.0):
    A = rng.normal(size=(n, n)); return s*(A@A.T) + n*np.eye(n)*0.5
def gkl(m1, S1, m2, S2):
    n = len(m1); S2i = np.linalg.inv(S2); dm = m1-m2
    return 0.5*(np.trace(S2i@S1) - n + dm@S2i@dm + np.log(np.linalg.det(S2)/np.linalg.det(S1)))
V3 = range(3); E = [(1, 0), (2, 0), (0, 1)]          # edges e=(j,i): j -> i
c = [rng.normal(size=K) for _ in V3]
T = {e: rng.normal(size=(1 if e != (2, 0) else 2, K)) for e in E}
R = {e: spd(T[e].shape[0]) for e in E}
P = [spd(K) for _ in V3]; m = [rng.normal(size=K) for _ in V3]
beta = {e: rng.uniform(0.3, 1.0) for e in E}
sigma2 = 0.7; lam = 0.9
Ge = {e: (lambda B: B@B.T)(rng.normal(size=(d, 1))) for e in E}; omhat = {e: rng.normal(size=d) for e in E}
loops = [[(1, 0), (0, 1)]]                            # the two-cycle 0 -> 1 -> 0
idx = {e: k for k, e in enumerate(E)}
B = np.zeros((d*len(loops), d*len(E)))
for g, L in enumerate(loops):
    for e in L: B[g*d:(g+1)*d, idx[e]*d:(idx[e]+1)*d] = np.eye(d)
Aw = np.eye(d*len(E))/sigma2 + lam*B.T@B + np.block([[Ge[e] if e == f else np.zeros((d, d)) for f in E] for e in E])
bw = np.concatenate([Ge[e]@omhat[e] for e in E]); a0 = np.linalg.solve(Aw, bw)
def kappa(i, w):                                      # kappa_i omega = omega c_i, omega in R^{K x K} as vector
    return w.reshape(K, K)@c[i]
Sig = [spd(K, 0.3) for _ in V3]; Vd = {e: spd(d, 0.2) for e in E}
def S(i, e): return T[e]@Sig[i]@T[e].T + R[e]
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(3); gh_w = gh_w/np.sqrt(2*np.pi)
def edge_cost(e, mu, a):
    j, i = e; Lc = np.linalg.cholesky(Vd[e]); tot = 0.0
    for nodes in itertools.product(range(3), repeat=d):
        z = gh_x[list(nodes)]; wgt = np.prod(gh_w[list(nodes)])
        om = a[e] + Lc@z
        mrec = T[e]@(mu[i] - kappa(i, om)); msrc = T[e]@mu[j]
        tot += wgt*gkl(mrec, S(i, e), msrc, S(j, e))
    return tot
def recip(mu, a):
    val = sum(gkl(mu[i], Sig[i], m[i], np.linalg.inv(P[i])) for i in V3)
    val += sum(beta[e]*edge_cost(e, mu, a) for e in E)
    av = np.concatenate([a[e] for e in E]); Vb = np.zeros((d*len(E),)*2)
    for e in E: Vb[idx[e]*d:(idx[e]+1)*d, idx[e]*d:(idx[e]+1)*d] = Vd[e]
    val += gkl(av, Vb, a0, np.linalg.inv(Aw))
    return val
def ES(mu, a):
    val = sum(0.5*(mu[i]-m[i])@P[i]@(mu[i]-m[i]) for i in V3)
    for e in E:
        j, i = e; Le = T[e]@(mu[i]-mu[j]-kappa(i, a[e]))
        val += 0.5*a[e]@a[e]/sigma2 + 0.5*(a[e]-omhat[e])@Ge[e]@(a[e]-omhat[e])
        val += 0.5*beta[e]*Le@np.linalg.inv(S(j, e))@Le
    av = np.concatenate([a[e] for e in E]); val += 0.5*lam*(B@av)@(B@av)
    return val
diffs = []
for trial in range(4):
    mu = [rng.normal(size=K) for _ in V3]; a = {e: rng.normal(size=d) for e in E}
    diffs.append(recip(mu, a) - ES(mu, a))
print("(1) recip - E^S over random means (should be constant):", np.round(diffs, 12), " spread=%.2e" % (max(diffs)-min(diffs)))
# (2) ring
def ring(l=5, qt=1.3, p=0.7, s=2.0, cc=0.8):
    n = 2*l; H = np.zeros((n, n)); A = np.array([[p+qt, -qt*cc], [-qt*cc, s+qt*cc**2]])
    Cp = np.array([[-qt, 0], [qt*cc, 0]])
    for i in range(l):
        H[2*i:2*i+2, 2*i:2*i+2] = A; k = (i+1) % l
        H[2*i:2*i+2, 2*k:2*k+2] = Cp
    Ablk = np.kron(np.eye(l), A); M = -np.linalg.solve(Ablk, H)
    ev = np.sort_complex(np.linalg.eigvals(M))
    w = qt*s/(p*s + p*qt*cc**2 + qt*s)
    pred = np.sort_complex(np.concatenate([-np.ones(l), [-(1 - np.exp(2j*np.pi*k/l)*w) for k in range(l)]]))
    return np.max(np.abs(ev-pred)), w
err, w = ring()
print("(2) ring eigenvalue error %.2e (w=%.4f)" % (err, w))
