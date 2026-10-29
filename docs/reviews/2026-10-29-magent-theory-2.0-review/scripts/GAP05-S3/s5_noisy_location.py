r"""GAP05-S3 s5: side conditions of it1:noisy and DQM in it1:location.

A. it1:noisy on S^1 (grid N = 256): q a von Mises mixture, p a cardioid-type law, and two
   kernels: (i) von Mises noise (a location kernel), (ii) a softmax code into a 3-letter
   alphabet (not a location family). Chain rule KL(q||p) = KL(qK||pK) + int KL(q^h||p^h),
   the convexity step and the bound KL(q || qhat) <= KL(q||p) - KL(qK||pK).
B. Version dependence of qhat when q K is not << p K (possible only if KL(q||p) = infinity),
   and well-definedness when q K << p K although KL(q||p) = infinity.
C. DQM of location families via the Hellinger expansion 4 H^2(t)/t^2 -> J_f, for
   f = x^2 phi(x) (C^1 with a zero), Student-t_5, Laplace (not C^1, finite J), uniform (not DQM);
   then the referent-averaged Fisher form it1:fisher-location for K = 2, r = 1.
"""
import numpy as np
from scipy.stats import norm, t as student_t, laplace
from scipy.linalg import expm

N = 256
x = 2 * np.pi * np.arange(N) / N; dx = 2 * np.pi / N
def nrm(f, w=dx): return f / (np.sum(f) * w)
q = nrm(0.6 * np.exp(2.5 * np.cos(x - 0.7)) + 0.4 * np.exp(1.2 * np.cos(x + 2.0)))
p = nrm(1 + 0.8 * np.cos(x - 0.2) + 0.15 * np.sin(3 * x))
def KL(a, b, w): return np.sum(a * (np.log(a) - np.log(b))) * w

def check(Kmat, wh, label):
    # Kmat[y, h]: kernel density/probability of message h given content y; wh: message quadrature weight
    qK = q @ Kmat * dx; pK = p @ Kmat * dx
    post_p = (p[:, None] * Kmat) / pK[None, :]            # p^h(y), columns indexed by h
    post_q = (q[:, None] * Kmat) / qK[None, :]
    qhat = post_p @ qK * wh
    lhs = KL(q, qhat, dx)
    kl_qp, kl_m = KL(q, p, dx), KL(qK, pK, wh)
    integ = np.sum(qK * np.array([KL(post_q[:, k], post_p[:, k], dx) for k in range(Kmat.shape[1])])) * wh
    print("%s: KL(q||p) = %.10f, KL(qK||pK) + int KL(q^h||p^h) = %.10f (chain-rule residual %.1e)" % (label, kl_qp, kl_m + integ, abs(kl_qp - kl_m - integ)))
    print("   KL(q||qhat) = %.6f <= int KL(q^h||p^h) = %.6f = KL(q||p) - KL(qK||pK); slack %.4f; mass of qhat %.12f"
          % (lhs, integ, integ - lhs, np.sum(qhat) * dx))

print("== A. it1:noisy on the circle ==")
Kvm = np.exp(1.8 * np.cos(x[None, :] - x[:, None])); Kvm = Kvm / (np.sum(Kvm, axis=1, keepdims=True) * dx)
check(Kvm, dx, "von Mises noise kernel")
centers = np.array([0.0, 2.1, 4.2])
logits = 2.0 * np.cos(x[:, None] - centers[None, :])
Ksm = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
check(Ksm, 1.0, "softmax code into 3 letters")

print("== B. side conditions: q K << p K ==")
pB = np.array([0.5, 0.5, 0.0]); qB = np.array([1, 1, 1]) / 3.0
Kid = np.eye(3)
pK = pB @ Kid
for name, v2 in [("version p^2 = delta_0", np.array([1.0, 0, 0])), ("version p^2 = delta_2", np.array([0, 0, 1.0]))]:
    posts = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), v2]          # p^h for h = 0, 1 and an arbitrary choice at h = 2
    qhat = sum(qB[h] * posts[h] for h in range(3))
    with np.errstate(divide='ignore'):
        kl = np.sum(qB * (np.log(qB) - np.log(qhat)))
    print("noiseless identity code, KL(q||p) = inf, pK(2) = 0: %s gives qhat = %s, KL(q||qhat) = %s" % (name, np.round(qhat, 4), kl))
Kn = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
pK = pB @ Kn; qK = qB @ Kn
posts = (pB[:, None] * Kn) / pK[None, :]
qhat = posts @ qK
print("noisy full-support kernel: pK = %s > 0, so qhat = %s is version independent although KL(q||p) = inf" % (np.round(pK, 4), np.round(qhat, 4)))

print("== C. DQM of location families through the Hellinger expansion ==")
xs = np.linspace(-14, 14, 280001); dxs = xs[1] - xs[0]
fams = {
    "x^2 phi(x) (C^1, zero at 0)": (lambda u: np.abs(u) * np.sqrt(norm.pdf(u)), 3.0),
    "Student-t_5": (lambda u: np.sqrt(student_t.pdf(u, 5)), 6.0 / 8.0),
    "Laplace (not C^1, J finite)": (lambda u: np.sqrt(laplace.pdf(u)), 1.0),
    "uniform[-1,1] (not DQM)": (lambda u: np.sqrt(0.5 * (np.abs(u) <= 1)), np.nan),
}
def H2(sqrtf, a):
    return np.sum((sqrtf(xs - a) - sqrtf(xs)) ** 2) * dxs
for name, (sf, J) in fams.items():
    vals = ["%.5f" % (4 * H2(sf, tt) / tt ** 2) for tt in (1e-1, 1e-2, 1e-3)]
    print("  %-30s 4H^2/t^2 at t = 0.1, 0.01, 0.001: %s   J_f = %s" % (name, ", ".join(vals), J))
print("  (x^2 phi: J_f = E(2/x - x)^2 = 3 >= 1/Var = 1/3; t_5: J = 3/4 >= 1/Var = 3/5)")

print("== C2. referent-averaged Fisher form it1:fisher-location, K = 2, r = 1 ==")
T = np.array([[1.0, 0.5]])
thb = np.array([[1.2, 0.3], [-0.2, 0.9]])
refs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]]); rho = np.full(4, 0.25)
M = (refs.T * rho) @ refs
C = np.linalg.solve(thb, np.linalg.solve(thb, M).T)          # thb^{-1} M thb^{-T}
rng = np.random.default_rng(3)
delta = rng.standard_normal((2, 2))
v = np.array([-0.5, 1.0]); delta0 = np.outer(v, rng.standard_normal(2))   # T delta0 = 0
for name, (sf, J) in list(fams.items())[:3]:
    F = J * np.trace(T @ delta @ C @ delta.T @ T.T)
    out = []
    for tt in (1e-2, 1e-3, 1e-4):
        shifts = [(T @ expm(-tt * delta) @ np.linalg.solve(thb, y) - T @ np.linalg.solve(thb, y))[0] for y in refs]
        out.append(4 * sum(r_ * H2(sf, a) for r_, a in zip(rho, shifts)) / tt ** 2)
    shifts0 = [(T @ expm(-0.5 * delta0) @ np.linalg.solve(thb, y) - T @ np.linalg.solve(thb, y))[0] for y in refs]
    print("  %-30s 4H^2/t^2 at t = 1e-2, 1e-3, 1e-4: %.5f, %.5f, %.5f   Tr(J_f T d C d^T T^T) = %.5f;  max shift for T d0 = 0 at t = 0.5: %.1e"
          % (name, out[0], out[1], out[2], F, max(abs(np.array(shifts0)))))

print("== C3. a two-dimensional location density with a zero: f(u) ~ |u|^2 exp(-|u|^2/2) ==")
g = np.linspace(-9, 9, 901); dg = g[1] - g[0]
U1, U2 = np.meshgrid(g, g, indexing='ij')
f2 = (U1 ** 2 + U2 ** 2) * np.exp(-(U1 ** 2 + U2 ** 2) / 2); f2 /= np.sum(f2) * dg * dg
with np.errstate(divide='ignore', invalid='ignore'):
    s1 = np.where(f2 > 0, 2 * U1 / (U1 ** 2 + U2 ** 2) - U1, 0.0)
    s2 = np.where(f2 > 0, 2 * U2 / (U1 ** 2 + U2 ** 2) - U2, 0.0)
Jm = np.array([[np.sum(f2 * s1 * s1), np.sum(f2 * s1 * s2)], [np.sum(f2 * s2 * s1), np.sum(f2 * s2 * s2)]]) * dg * dg
Cov = np.array([[np.sum(f2 * U1 * U1), np.sum(f2 * U1 * U2)], [np.sum(f2 * U2 * U1), np.sum(f2 * U2 * U2)]]) * dg * dg
print("  J_f = %s (exact I);  Cov = %s (exact 2I);  J_f - Cov^{-1} eigenvalues %s" % (np.round(Jm, 4).tolist(), np.round(Cov, 4).tolist(), np.round(np.linalg.eigvalsh(Jm - np.linalg.inv(Cov)), 4)))
sq = lambda a1, a2: np.sqrt(np.maximum(((U1 - a1) ** 2 + (U2 - a2) ** 2) * np.exp(-((U1 - a1) ** 2 + (U2 - a2) ** 2) / 2), 0) / (2 * np.pi * 2))
for tt in (0.05, 0.01):
    a = tt * np.array([0.6, 0.8])
    H2v = np.sum((sq(*a) - sq(0, 0)) ** 2) * dg * dg
    print("  t = %.2f along (0.6, 0.8): 4H^2/t^2 = %.5f, a^T J_f a / t^2 = %.5f" % (tt, 4 * H2v / tt ** 2, a @ Jm @ a / tt ** 2))
