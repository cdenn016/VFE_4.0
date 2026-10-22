r"""L02-GEOM s05: probe of the rank deficit of dM|_H for n=3 proper complex Gaussians.

H = {h in (+)Herm : sum_i w_i h_i M_i^{-1} = 0} (Fisher-orthogonal complement of the GL(2,C)
orbit, using G = sum_i w_i tr(M_i^{-1} h_i M_i^{-1} k_i)).  Image dM(h) = sum_i w_i h_i.
Find the Hermitian Q with tr(Q dM(h)) = 0 on H and test candidate closed forms.
"""
import numpy as np
rng = np.random.default_rng(5)
sig = [np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], complex),
       np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
def herm_from(v):
    return sum(v[m] * sig[m] for m in range(4))
for trial in range(3):
    n = 3
    Ms = []
    for _ in range(n):
        L = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); Ms.append(L @ L.conj().T + 0.3 * np.eye(2))
    w = rng.random(n) + 0.5
    # constraint matrix: real and imaginary parts of sum_i w_i h_i M_i^{-1}
    rows = []
    for col in range(4 * n):
        v = np.zeros(4 * n); v[col] = 1
        Y = sum(w[i] * herm_from(v[4 * i:4 * i + 4]) @ np.linalg.inv(Ms[i]) for i in range(n))
        rows.append(np.concatenate([Y.real.ravel(), Y.imag.ravel()]))
    Cm = np.array(rows).T                     # 8 x 12
    U, S, Vt = np.linalg.svd(Cm)
    rk = int((S > 1e-9).sum())
    Hb = Vt[rk:].T                            # 12 x dimH
    D = np.hstack([w[i] * np.eye(4) for i in range(n)])
    img = D @ Hb
    U2, S2, Vt2 = np.linalg.svd(img)
    rkimg = int((S2 > 1e-9).sum())
    qcoef = U2[:, rkimg:]                     # Pauli coords x of annihilator: tr(Q H)= 2 eta-like
    # annihilator in trace pairing: tr(Q (x^mu sigma_mu)) = sum_mu x^mu tr(Q sigma_mu); choose Q
    # with tr(Q sigma_mu) = qcoef[mu] -> Q = (1/2) sum qcoef[mu] sigma_mu
    Q = 0.5 * herm_from(qcoef[:, 0])
    # candidate: Q proportional to something like sum_i w_i M_i^{-1} ... test residual of tr(Q dM)=0
    Mtot = sum(w[i] * Ms[i] for i in range(n))
    cands = {'M^-1': np.linalg.inv(Mtot), 'sum w M_i^-1': sum(w[i] * np.linalg.inv(Ms[i]) for i in range(n)),
             'I': np.eye(2)}
    print('trial', trial, ': rank C =', rk, ' dim H =', Hb.shape[1], ' rank dM|_H =', rkimg)
    for nm, Cc in cands.items():
        val = [np.trace(Cc @ herm_from(img[:, k])).real for k in range(img.shape[1])]
        print('   tr(%s dM(h)) on H basis: max abs %.2e' % (nm, np.max(np.abs(val))))

print('singular values of dM|_H (orthonormal H basis in the Euclidean coordinate sense):')
for n in (2, 3, 4, 5):
    for trial in range(2):
        Ms = []
        for _ in range(n):
            L = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); Ms.append(L @ L.conj().T + 0.3 * np.eye(2))
        w = rng.random(n) + 0.5
        rows = []
        for col in range(4 * n):
            v = np.zeros(4 * n); v[col] = 1
            Y = sum(w[i] * herm_from(v[4 * i:4 * i + 4]) @ np.linalg.inv(Ms[i]) for i in range(n))
            rows.append(np.concatenate([Y.real.ravel(), Y.imag.ravel()]))
        Cm = np.array(rows).T
        U, S, Vt = np.linalg.svd(Cm)
        rk = int((S > 1e-9).sum())
        Hb = Vt[rk:].T
        D = np.hstack([w[i] * np.eye(4) for i in range(n)])
        S2 = np.linalg.svd(D @ Hb, compute_uv=False)
        print('  n=%d: constraint rank %d, dim H %d, sv(dM|_H) =' % (n, rk, Hb.shape[1]), np.array2string(S2, precision=3))

# structural explanation for n = 3: with k_i = w_i h_i, A = M1^-1 - M3^-1, B = M2^-1 - M3^-1,
# the kernel of k -> (sum k_i, sum k_i M_i^-1) always contains the two independent elements
#   (A^-1, -B^-1, B^-1 - A^-1)  and  (A^-1 B A^-1, -A^-1, A^-1 - A^-1 B A^-1),
# one more than the generic count 12 - 4 - 7 = 1, so rank dM|_H <= 5 - 2 = 3.
Ms = []
for _ in range(3):
    L = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); Ms.append(L @ L.conj().T + 0.3 * np.eye(2))
Mi = [np.linalg.inv(M_) for M_ in Ms]
A = Mi[0] - Mi[2]; B = Mi[1] - Mi[2]
Ai, Bi = np.linalg.inv(A), np.linalg.inv(B)
for ks in ([Ai, -Bi, Bi - Ai], [Ai @ B @ Ai, -Ai, Ai - Ai @ B @ Ai]):
    herm = max(np.abs(k_ - k_.conj().T).max() for k_ in ks)
    print('kernel element: hermiticity %.1e, |sum k| %.1e, |sum k M^-1| %.1e'
          % (herm, np.abs(sum(ks)).max(), np.abs(sum(k_ @ m_ for k_, m_ in zip(ks, Mi))).max()))
