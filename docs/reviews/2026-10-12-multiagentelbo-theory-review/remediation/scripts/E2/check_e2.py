r"""E2 checks for the c08b additions.

(1) rem:link-reciprocal-limit: the 2-cycles (e,e') based at i and (e',e) based at j vanish together,
    and P_c{Q_p >= eps} -> 0 as the 2-cycle coefficient c grows (self-normalized weights from nu).
(2) rem:link-normalizer-role: log Z_S(c) = -\int_0^1 sum_p c_p E_{P_{tc}}[Q_p] dt, and
    -E_nu S <= log Z_S <= -E_P S.
(3) rem:link-chart-injectivity: exp(2 pi J) = I = exp(0); exp is injective on
    D = {|Im eig| < pi} (principal logm recovers X); sum (Im lam)^2 <= ||skew||_F^2;
    base-law mass outside D <= (||skew alpha||_F^2 + K(K-1)s^2/2)/pi^2.
"""
import numpy as np
from scipy.linalg import expm, logm

rng = np.random.default_rng(12)


def Q(H: np.ndarray) -> np.ndarray:
    """Loop defect with G^fib = I: 0.5 ||H - I||_F^2, batched."""
    K = H.shape[-1]
    D = H - np.eye(K)
    return 0.5 * np.einsum("...ab,...ab->...", D, D)


def bexpm(W: np.ndarray) -> np.ndarray:
    return np.stack([expm(w) for w in W])


# ---------- (1) 2-cycle ----------
K, n = 2, 100000
we = rng.normal(size=(n, K, K))     # generator of copy e: j->i  (base N(0, I))
wf = rng.normal(size=(n, K, K))     # generator of copy e': i->j (distinct reversal pair)
Te, Tf = bexpm(we), bexpm(wf)
Qp = Q(Te @ Tf)        # based at i
Qpp = Q(Tf @ Te)       # based at j
print("(1) max |Q_p - Q_p'| =", np.abs(Qp - Qpp).max(), "(differ in general; same zero set)")
# zero sets coincide: at wf = -we both vanish
print("    at w_e' = -w_e:", Q(Te[:5] @ bexpm(-we[:5])).max(), Q(bexpm(-we[:5]) @ Te[:5]).max())
eps = 0.5
for c in (0.0, 1.0, 4.0, 16.0, 64.0):
    w = np.exp(-c * (Qp + Qpp))
    w /= w.sum()
    print(f"    c={c:5.1f}: P_c(Q_p>=eps)={w[Qp >= eps].sum():.3e}, "
          f"P_c(||T_e'-T_e^-1||>0.5)={w[np.linalg.norm(Tf - np.linalg.inv(Te), axis=(1, 2)) > 0.5].sum():.3e}")

# ---------- (2) thermodynamic integration ----------
# triangle of three copies, K = 2, one loop coefficient c; nu = N(0, 0.6^2 I)
n = 400000
W = 0.6 * rng.normal(size=(n, 3, K, K))
H = bexpm(W[:, 0]) @ bexpm(W[:, 1]) @ bexpm(W[:, 2])
S1 = Q(H)
c = 1.7
logZ = np.log(np.mean(np.exp(-c * S1)))
ts = np.concatenate([[0.0], np.geomspace(1e-7, 1.0, 4001)])
EP = np.array([np.sum(np.exp(-t * c * S1) * S1) / np.sum(np.exp(-t * c * S1)) for t in ts])
TI = -np.trapezoid(c * EP, ts)
print("(2) log Z_S =", logZ, " TI =", TI, " diff =", logZ - TI)
print("    -E_nu S =", -c * S1.mean(), " <= log Z_S <= -E_P S =", -c * EP[-1])

# ---------- (3) chart injectivity and domain bounds ----------
J = np.array([[0.0, -1.0], [1.0, 0.0]])
print("(3) ||exp(2 pi J) - I|| =", np.linalg.norm(expm(2 * np.pi * J) - np.eye(2)))
errs = []
for _ in range(2000):
    k = rng.integers(2, 6)
    X = rng.normal(size=(k, k))
    lam = np.linalg.eigvals(X)
    if np.abs(lam.imag).max() >= np.pi - 1e-3:
        continue
    errs.append(np.linalg.norm(logm(expm(X)).real - X) / max(1.0, np.linalg.norm(X)))
print("    principal logm(expm X) = X on D: max rel err", max(errs), "over", len(errs))
viol = 0
for _ in range(5000):
    k = rng.integers(2, 7)
    X = 2 * rng.normal(size=(k, k))
    lam = np.linalg.eigvals(X)
    A = (X - X.T) / 2
    viol += (np.sum(lam.imag ** 2) > np.sum(A * A) + 1e-9)
    viol += (np.abs(lam.imag).max() > np.linalg.norm(A, 2) + 1e-9)
print("    Schur/Bendixson violations:", viol)
for K in (2, 4, 8):
    for s in (0.25, 0.5, 1.0, 1.5):
        m = 200000 if K < 8 else 50000
        Wk = s * rng.normal(size=(m, K, K))
        lam = np.linalg.eigvals(Wk)
        out = (np.abs(lam.imag) >= np.pi).any(-1).mean()
        frac = (np.abs(lam.imag) >= np.pi).mean()
        bound = K * (K - 1) * s ** 2 / (2 * np.pi ** 2)
        print(f"    K={K} s={s}: P(out of D)={out:.2e}  bound={bound:.2e}  "
              f"eig fraction={frac:.2e} bound/K={bound / K:.2e}")
