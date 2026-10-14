"""L8: cut completed square (main-manuscript lemma cited by g00), balanced weight, and the information-form
evidence identity an implementer needs (derived here, not stated in Supplement G)."""
import numpy as np
from numpy.linalg import inv, slogdet, eigvalsh, det
from scipy.linalg import sqrtm
from common import rng, out, rep, note, spd, gl_plus

K = 3; Ws = [spd(K, 1e2) for _ in range(3)]; Ts = [gl_plus(K, 10.) for _ in range(3)]
LII = sum(Ws); LIJ = sum(W @ T for W, T in zip(Ws, Ts)); LJJ = sum(T.T @ W @ T for W, T in zip(Ws, Ts))
Tb = inv(LII) @ LIJ; Delta = LJJ - LIJ.T @ inv(LII) @ LIJ
rep("c10 cut excess Delta = sum (T_e - Tbar)^T W_e (T_e - Tbar)", np.abs(Delta - sum((T - Tb).T @ W @ (T - Tb) for W, T in zip(Ws, Ts))).max(), np.abs(Delta).max())
zI, zJ = rng.standard_normal(K), rng.standard_normal(K)
E = sum((zI - T @ zJ) @ W @ (zI - T @ zJ) for W, T in zip(Ws, Ts))
rep("c10 completed square", abs(E - ((zI - Tb @ zJ) @ LII @ (zI - Tb @ zJ) + zJ @ Delta @ zJ)), E)
Wlo = LIJ @ inv(LJJ) @ LIJ.T; Wlo_h = sqrtm(Wlo).real; Wlo_mh = inv(Wlo_h)
Wbal = Wlo_h @ sqrtm(Wlo_mh @ LII @ Wlo_mh).real @ Wlo_h
Tbal = inv(Wbal) @ LIJ
rep("c10 balanced: W Wlo^-1 W = Whi", np.abs(Wbal @ inv(Wlo) @ Wbal - LII).max(), np.abs(LII).max())
rep("c10 balanced: Theta^T LII Theta = LJJ", np.abs(Tbal.T @ LII @ Tbal - LJJ).max(), np.abs(LJJ).max())
note(f"c10 balanced self terms min eigs: S_I {eigvalsh(LII - Wbal).min():.3e}, S_J {eigvalsh(LJJ - LIJ.T @ inv(Wbal) @ LIJ).min():.3e}")

# Information-form evidence: joint (y,o) with precision Lh, info vector hh; log p(o) = h_o^T o - 1/2 o^T Loo o + A_n(h_Y - L_Yo o, L_YY) - A_{n+m}(hh, Lh)
def A(h, L):
    return 0.5 * h @ np.linalg.solve(L, h) - 0.5 * slogdet(L)[1] + len(h) / 2 * np.log(2 * np.pi)
n, m = 4, 2; Lh = spd(n + m, 1e3); hh = rng.standard_normal(n + m); o = rng.standard_normal(m)
LYY, LYo, Loo = Lh[:n, :n], Lh[:n, n:], Lh[n:, n:]
logp = hh[n:] @ o - 0.5 * o @ Loo @ o + A(hh[:n] - LYo @ o, LYY) - A(hh, Lh)
Sig = inv(Lh); mu = Sig @ hh; So = Sig[n:, n:]; d = o - mu[n:]
direct = -0.5 * (m * np.log(2 * np.pi) + slogdet(So)[1] + d @ inv(So) @ d)
rep("info-form evidence identity", abs(logp - direct), abs(direct))
print("\n".join(out))
