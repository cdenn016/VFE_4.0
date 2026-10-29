"""GAP09-S1 s3: the alternating structured updates of cmem:product-label.

Structured family Q(a, u, v) = s_a q(u) K_a(u, v) on a finite presentation set and finite cue
alphabet.  The result kernel cancels, and
  F(s, q) = KL(s || pihat) + sum_a s_a KL(q || mu_a) - E_q log l(o | X),
whose block minimizers are  s_a ∝ pihat_a exp(-KL(q||mu_a))  and
q(u) ∝ l(o|u) exp(sum_a s_a log mu_a(u))  (cmem:product-label and the sentence after it).

Checks
 (a) exact block decrease: F(s,q) - F(s*(q), q) = KL(s || s*(q)) and
     F(s,q) - F(s, q*(s)) = KL(q || q*(s))  (so every sweep decreases F by a sum of KLs);
 (b) from every initialization on a grid the iterates converge (successive change < 1e-13);
 (c) two attracting fixed points with different free energies exist, so the iteration does not
     converge to the structured minimizer from every start; a third (repelling) fixed point
     separates the basins; all are above the exact minimum -log Z(o);
 (d) the q-normalizer is bounded by sum_a s_a Z_a(o) (weighted AM-GM), and it vanishes when
     the cue components have disjoint supports, where the q-update is undefined.
"""
import numpy as np

pihat = np.array([0.4, 0.6])
mu = np.array([[0.98, 0.01, 0.01],
               [0.01, 0.01, 0.98]])
lik = np.array([0.5, 0.6, 0.5])          # l(o | u) at the observed record


def kl(x, y):
    m = x > 0
    return float(np.sum(x[m] * np.log(x[m] / y[m])))


def F(s, q):
    return kl(s, pihat) + sum(s[a] * kl(q, mu[a]) for a in range(2)) - float(q @ np.log(lik))


def s_update(q):
    w = pihat * np.exp(-np.array([kl(q, mu[a]) for a in range(2)]))
    return w / w.sum()


def q_update(s):
    w = lik * np.exp(s @ np.log(mu))
    return w / w.sum(), w.sum()


Z = float(pihat @ (mu @ lik))
print('exact minimum -log Z(o) = %.10f' % (-np.log(Z)))

# (a) exact block-decrease identities on random points
rng = np.random.default_rng(7)
ea = 0.0
for _ in range(1000):
    s = rng.dirichlet([1, 1]); q = rng.dirichlet([1, 1, 1])
    s1 = s_update(q); q1, _ = q_update(s)
    ea = max(ea, abs(F(s, q) - F(s1, q) - kl(s, s1)), abs(F(s, q) - F(s, q1) - kl(q, q1)))
print('(a) max |F(s,q) - F(block update) - KL(old||new)| over 1000 random points: %.2e' % ea)

# (b), (c) iterate from a grid of initial label laws
limits = []
for s10 in np.linspace(0.01, 0.99, 99):
    s = np.array([s10, 1 - s10]); q, _ = q_update(s)
    Fprev = F(s, q); mono = True
    for it in range(20000):
        s_new = s_update(q); q_new, _ = q_update(s_new)
        Fn = F(s_new, q_new)
        if Fn > Fprev + 1e-14:
            mono = False
        change = np.abs(s_new - s).sum() + np.abs(q_new - q).sum()
        s, q, Fprev = s_new, q_new, Fn
        if change < 1e-13:
            break
    limits.append((s10, s[0], Fprev, it, mono))
lim = np.array([(a, b, c) for a, b, c, _, _ in limits])
print('(b) all 99 runs monotone:', all(m for *_, m in limits), '; max sweeps used:', max(it for _, _, _, it, _ in limits))
fps = []
for _, s1, Fv, _, _ in limits:
    if not any(abs(s1 - f[0]) < 1e-8 for f in fps):
        fps.append((s1, Fv))
for s1, Fv in sorted(fps):
    basin = [a for a, b, c, _, _ in limits if abs(b - s1) < 1e-8]
    print('(c) attracting fixed point s_1 = %.10f, F = %.10f, reached from s_1(0) in [%.2f, %.2f] (%d starts)'
          % (s1, Fv, min(basin), max(basin), len(basin)))

# repelling fixed point between the basins: root of g(s1) = s_update(q_update(s))_1 - s1
def g(x):
    s = np.array([x, 1 - x]); q, _ = q_update(s)
    return s_update(q)[0] - x
xs = np.linspace(1e-6, 1 - 1e-6, 200001)
gv = np.array([g(x) for x in xs[::100]])
roots = []
xx = xs[::100]
for k in range(len(xx) - 1):
    if gv[k] == 0 or gv[k] * gv[k + 1] < 0:
        a, b = xx[k], xx[k + 1]
        for _ in range(100):
            c = 0.5 * (a + b)
            if g(a) * g(c) <= 0:
                b = c
            else:
                a = c
        r = 0.5 * (a + b)
        h = 1e-7
        deriv = (g(r + h) - g(r - h)) / (2 * h) + 1.0     # derivative of the sweep map s1 -> s1'
        s = np.array([r, 1 - r]); q, _ = q_update(s)
        roots.append((r, deriv, F(s, q)))
for r, d, Fv in roots:
    kind = 'attracting' if abs(d) < 1 else 'repelling'
    print('    fixed point of the sweep map: s_1 = %.10f, slope of sweep map = %.4f (%s), F = %.10f' % (r, d, kind, Fv))

# (d) normalizer: AM-GM bound and the disjoint-support failure
worst = -np.inf
for _ in range(1000):
    s = rng.dirichlet([1, 1]); _, nz = q_update(s)
    worst = max(worst, nz - float(s @ (mu @ lik)))
print('(d) max over 1000 random s of [normalizer - sum_a s_a Z_a(o)] = %.3e (<= 0 by AM-GM)' % worst)
mu_disj = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
s = np.array([0.4, 0.6])
with np.errstate(divide='ignore'):
    w = lik * np.exp(s @ np.log(mu_disj))
print('    disjoint cue supports, interior s = (0.4, 0.6): q-normalizer =', float(w.sum()),
      '(every q has sum_a s_a KL(q||mu_a) = +inf, so the structured family has finite F only at vertices of s)')


# (e) a weak-coupling criterion: if Lambda = max |log(mu_a(u) mu_b(v) / (mu_b(u) mu_a(v)))| <= 1,
# the tangent Hessian of F on the product of simplices is positive semidefinite everywhere (the
# doubly centered cross term has Hilbert-Schmidt norm <= Lambda under s (x) q), so F is jointly
# convex, the structured fixed point is unique and every run converges to it.
def cross_ratio_range(mu):
    lm = np.log(mu)
    best = 0.0
    for a in range(mu.shape[0]):
        for b in range(mu.shape[0]):
            for u_ in range(mu.shape[1]):
                for v_ in range(mu.shape[1]):
                    best = max(best, abs(lm[a, u_] + lm[b, v_] - lm[b, u_] - lm[a, v_]))
    return best


def min_tangent_hessian_eig(mu_, s, q):
    nA, nU = mu_.shape
    H = np.zeros((nA + nU, nA + nU))
    H[:nA, :nA] = np.diag(1 / s); H[nA:, nA:] = np.diag(1 / q)
    H[:nA, nA:] = -np.log(mu_); H[nA:, :nA] = -np.log(mu_).T
    # orthonormal basis of the tangent space {sum sigma = 0, sum eta = 0}
    T = np.zeros((nA + nU, nA + nU - 2))
    Ba = np.linalg.qr(np.vstack([np.ones(nA), np.eye(nA)[:-1]]).T)[0][:, 1:]
    Bu = np.linalg.qr(np.vstack([np.ones(nU), np.eye(nU)[:-1]]).T)[0][:, 1:]
    T[:nA, :nA - 1] = Ba; T[nA:, nA - 1:] = Bu
    return float(np.linalg.eigvalsh(T.T @ H @ T).min())


print('(e) cross-ratio range Lambda of the bimodal bank above: %.3f' % cross_ratio_range(mu))
mu_weak = np.array([[0.40, 0.30, 0.30], [0.30, 0.30, 0.40]])
print('    weak bank mu = %s: Lambda = %.3f' % (mu_weak.tolist(), cross_ratio_range(mu_weak)))
mins = min(min_tangent_hessian_eig(mu_weak, rng.dirichlet([0.3, 0.3]) * 0.999 + 0.0005,
                                   rng.dirichlet([0.3, 0.3, 0.3]) * 0.999 + 0.000333) for _ in range(3000))
print('    min tangent-Hessian eigenvalue over 3000 random interior points: %.4f (>= 0: jointly convex)' % mins)
mu_save = mu.copy(); mu[:] = mu_weak
ends = set()
for s10 in np.linspace(0.01, 0.99, 99):
    s = np.array([s10, 1 - s10]); q, _ = q_update(s)
    for it in range(20000):
        s_new = s_update(q); q_new, _ = q_update(s_new)
        ch = np.abs(s_new - s).sum() + np.abs(q_new - q).sum(); s, q = s_new, q_new
        if ch < 1e-14:
            break
    ends.add(round(float(s[0]), 10))
print('    weak bank: distinct limits of s_1 over 99 starts:', sorted(ends))
print('    bimodal bank: min tangent-Hessian eigenvalue at the repelling fixed point: %.4f' %
      min_tangent_hessian_eig(mu_save, np.array([roots[1][0], 1 - roots[1][0]]),
                              (lambda s_: q_update(s_)[0])(np.array([roots[1][0], 1 - roots[1][0]]))))
