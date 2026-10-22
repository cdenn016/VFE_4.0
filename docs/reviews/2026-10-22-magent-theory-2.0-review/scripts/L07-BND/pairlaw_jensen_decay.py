r"""Checks of eq:boundary-tree-jensen, the pair-law identity D_k = KL(J_k || J_k^rev),
an explicit instance where D_k differs from KL(f || P_k f), and the decay of the
public contrast under the profile flow (linearized rate 3 - 2 sum_k b_k P_k on
Walsh modes).

Depth d = 2 and d = 3 profiles on a trivial or 2-point base.
"""
import itertools
import numpy as np

def bits(d):
    return np.array(list(itertools.product([1, -1], repeat=d)))

def Pk_matrix(d, k):
    """Markov kernel P_k(u, v): keep first k bits, flip bit k+1, uniform on the rest."""
    U = bits(d); n = len(U)
    P = np.zeros((n, n))
    for a, u in enumerate(U):
        for b, v in enumerate(U):
            if np.all(u[:k] == v[:k]) and v[k] == -u[k]:
                P[a, b] = 2.0 ** (-(d - k - 1))
    return P

print("== explicit instance: base trivial, d=2, f(u1,u2) = 1 + 0.5 u2, k = 0 ==")
d = 2; U = bits(d); w = np.full(len(U), 1 / len(U))
f = 1 + 0.5 * U[:, 1]
P0 = Pk_matrix(d, 0)
lf = np.log(f)
Dk = (w * f * (lf - P0 @ lf)).sum()
Pf = P0 @ f
kl_avg = (w * f * np.log(f / Pf)).sum()
gap = (w * f * (np.log(Pf) - P0 @ lf)).sum()
J = w[:, None] * f[:, None] * P0          # J(u,v) = U(u) f(u) P(u,v)
Jrev = J.T                                  # exchange u, v
mask = J > 0
kl_pair = (J[mask] * np.log(J[mask] / Jrev[mask])).sum()
print(f"D_0 = {Dk:.12f}, KL(J||J^rev) = {kl_pair:.12f}, KL(f||P_0 f) = {kl_avg:.12f}")
print(f"D_0 - KL(f||P_0 f) = {Dk - kl_avg:.12f}, Jensen term = {gap:.12f}, closed form -0.5 log 0.75 = {-0.5*np.log(0.75):.12f}")
print("P_0 f =", Pf, "(constant: the averaged class source equals the reference here)")

print("== random d=3 profile on a 2-point base: all k ==")
rng = np.random.default_rng(3)
d = 3; U = bits(d); n = len(U); A = np.array([0.35, 0.65])
f = np.exp(rng.normal(size=(2, n)) * 0.7)
f /= (A[:, None] * f / n).sum()
for k in range(d):
    P = Pk_matrix(d, k)
    assert np.allclose(P.sum(1), 1) and np.allclose(P, P.T)
    lf = np.log(f)
    Dk = (A[:, None] / n * f * (lf - lf @ P.T)).sum()
    J = (A[:, None, None] / n) * f[:, :, None] * P[None, :, :]
    Jrev = np.transpose(J, (0, 2, 1)) * 0 + (A[:, None, None] / n) * f[:, None, :] * P.T[None, :, :]
    m = J > 0
    klp = (J[m] * np.log(J[m] / Jrev[m])).sum()
    Pf = f @ P.T
    kla = (A[:, None] / n * f * np.log(f / Pf)).sum()
    gap = (A[:, None] / n * f * (np.log(Pf) - lf @ P.T)).sum()
    print(f"k={k}: D_k={Dk:.10f} KL(J||Jrev)={klp:.10f} |diff|={abs(Dk-klp):.1e}; D_k-KL(f||P_kf)={Dk-kla:.10f} gap={gap:.10f}")

print("== gradient of D_k vs gradient of KL(f || P_k f) at the d=2 instance ==")
d = 2; U = bits(d); w = np.full(4, .25)
f = 1 + 0.5 * U[:, 1]; P0 = Pk_matrix(d, 0)
gD = np.log(f) + 1 - P0 @ np.log(f) - (P0 @ f) / f
gK = np.log(f) + 1 - np.log(P0 @ f) - P0 @ (f / (P0 @ f))
# centred (mass-preserving) forces
cD = gD - (w * f * gD).sum(); cK = gK - (w * f * gK).sum()
print("centred force from D_0:", np.round(cD, 6))
print("centred force from KL(f||P_0 f):", np.round(cK, 6))

print("== decay of public contrast: linearized operator 3 - 2 sum_k b_k P_k ==")
for d in (2, 3, 4):
    U = bits(d); n = len(U)
    nk = np.array([2 ** (d - k - 1) for k in range(d)]); pik = nk / nk.sum()
    Lop = 3 * np.eye(n) - 2 * sum(pik[k] * Pk_matrix(d, k) for k in range(d))
    ev = np.linalg.eigvalsh((Lop + Lop.T) / 2)
    # remove the constant mode (eigenvalue 3 - 2 = 1 for the constant? check)
    const = np.ones(n) / np.sqrt(n)
    Pperp = np.eye(n) - np.outer(const, const)
    evp = np.linalg.eigvalsh(Pperp @ Lop @ Pperp)
    evp = np.sort(evp)[1:]  # drop the zero from the projected constant
    print(f"d={d}: uniform-prior class weights {np.round(pik,4)}; contrast eigenvalues min {evp.min():.4f} max {evp.max():.4f}")
