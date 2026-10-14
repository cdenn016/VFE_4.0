r"""L1-variational check D: witnesses for findings.
(1) thm:rg-memory-inference-descent(iv): unmarked sweep T with a fixed mismatched record conditional psi'
    can increase eq:rg-memory-fixed-vfe; the marked sweep decreases it strictly even when q is already at the
    conditional target, so the stated strictness condition is incomplete when psi' != psi.
(2) prop:obs-slow-model-link-dichotomy closing sentence: with both rates small but e_w >> e_m the model
    disagreement is gone long before 1/e_m; small-rate asymptotics of the slow rate.
"""
import numpy as np
from scipy.linalg import expm

# (1) Xi = {0,1}, parents par(0)=z0, par(1)=z1 (singleton fibers, so D = I), uniform p, proposal a01 = 1/2.
p = np.array([0.5, 0.5])
M = np.array([[0.5, 0.5], [0.5, 0.5]])
T = M  # D = C J = I on singleton fibers
psi = np.array([[0.5, 0.5], [0.999, 0.001]])            # record conditionals psi_xi(r), r in {0,1}
psi_p = np.array([[0.5, 0.5], [0.001, 0.999]])      # recognition record conditionals psi'_xi


def kl(a, b):
    a = np.asarray(a); b = np.asarray(b); msk = a > 0
    return float(np.sum(a[msk] * np.log(a[msk] / b[msk])))


def F(q, psip):  # eq:rg-memory-fixed-vfe without the constant -log Z_o
    return kl(q, p) + sum(q[x] * kl(psip[x], psi[x]) for x in range(2))


q = np.array([1.0, 0.0])
print("[1a] unmarked T, psi' held: F before =", F(q, psi_p), " after =", F(q @ T, psi_p))
q2 = p.copy()  # already the conditional target on the single component
print("[1b] marked T at q = target, psi' != psi: F before =", F(q2, psi_p), " after (psi' -> psi) =", F(q2 @ T, psi))

# (2) slow rate asymptotics and a mixed-rate arm
A0 = np.array([[4, 0, -2 * np.sqrt(2)], [0, 4, -2 * np.sqrt(2)], [-2 * np.sqrt(2), -2 * np.sqrt(2), 4.0]])
for em, ew in [(1e-3, 1e-3), (1e-4, 1e-2), (1e-4, 1e-4), (1e-4, 0.0)]:
    sm = 2 * (1 + em + ew) - 2 * np.sqrt((1 - em) ** 2 + ew ** 2)
    Mf = np.diag([1, em, ew]) @ A0
    x0 = np.array([0.0, 0.0, 1.0])
    t = 1 / (12 * em)
    xt = expm(-Mf * t) @ x0
    print("[2] e_m=%.0e e_w=%.0e: slow rate=%.3e (4e_m+2e_w=%.3e); |r^m(1/(12 e_m))|=%.3e"
          % (em, ew, sm, 4 * em + 2 * ew, abs(np.sqrt(2) * xt[1] - xt[2])))
