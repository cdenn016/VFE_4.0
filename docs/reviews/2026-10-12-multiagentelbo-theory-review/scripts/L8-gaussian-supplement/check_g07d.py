"""L8 checks for g07 part D: nonflat series pivot, path formula, gauge law, example, anchored hidden vertices."""
import numpy as np
import sympy as sp
from numpy.linalg import inv, slogdet, eigvalsh, det
from common import rng, out, rep, note, spd, gl_plus

def schur(L, r, hdn):
    return L[np.ix_(r, r)] - L[np.ix_(r, hdn)] @ inv(L[np.ix_(hdn, hdn)]) @ L[np.ix_(hdn, r)]

K = 3
Th1, Th2 = gl_plus(K, 30.), gl_plus(K, 30.); W1, W2 = spd(K, 1e4), spd(K, 1e4)
h0, h1, h2 = rng.standard_normal(K), rng.standard_normal(K), rng.standard_normal(K)
R1 = np.hstack([np.eye(K), -Th1, np.zeros((K, K))]); R2 = np.hstack([np.zeros((K, K)), np.eye(K), -Th2])
Lf = R1.T @ W1 @ R1 + R2.T @ W2 @ R2 + np.kron(np.diag([1., 0, 1.]), np.eye(K))
hf = np.concatenate([h1, h0, h2])
r = list(range(K)) + list(range(2*K, 3*K)); hd = list(range(K, 2*K))
Le = schur(Lf, r, hd); Mh = Lf[np.ix_(hd, hd)]
he = hf[r] - Lf[np.ix_(r, hd)] @ inv(Mh) @ hf[hd]
Wn = inv(inv(W1) + Th1 @ inv(W2) @ Th1.T); Tn = Th1 @ Th2
Rn = np.hstack([np.eye(K), -Tn]); pred = Rn.T @ Wn @ Rn + np.eye(2*K)
rep("g07 nonflat series pivot precision", np.abs(Le - pred).max(), np.abs(Le).max())
M = Th1.T @ W1 @ Th1 + W2
ph = np.concatenate([h1 + W1 @ Th1 @ inv(M) @ h0, h2 + Th2.T @ W2 @ inv(M) @ h0])
rep("g07 nonflat series source update", np.abs(he - ph).max(), np.abs(he).max())
c_direct = 0.5 * slogdet(Mh)[1] - K / 2 * np.log(2 * np.pi) - 0.5 * h0 @ inv(Mh) @ h0
c_formula = 0.5 * slogdet(M)[1] - K / 2 * np.log(2 * np.pi) - 0.5 * h0 @ inv(M) @ h0
rep("g07 nonflat series scalar update", abs(c_direct - c_formula), abs(c_direct))
note(f"   conditioning: cond(W1)=1e4, cond(Theta)=30, cond(M)={np.linalg.cond(M):.2e}")
g0, g1, g2 = gl_plus(K, 10.), gl_plus(K, 10.), gl_plus(K, 10.)
Th1p, Th2p = g1 @ Th1 @ inv(g0), g0 @ Th2 @ inv(g2)
W1p, W2p = inv(g1).T @ W1 @ inv(g1), inv(g0).T @ W2 @ inv(g0)
Wnp = inv(inv(W1p) + Th1p @ inv(W2p) @ Th1p.T)
rep("g07 series gauge law W_new' (relative)", np.abs(Wnp - inv(g1).T @ Wn @ inv(g1)).max() / np.abs(Wnp).max())
Mp = Th1p.T @ W1p @ Th1p + W2p
rep("g07 series gauge log det M shift", abs(slogdet(Mp)[1] - slogdet(M)[1] + 2 * np.log(abs(det(g0)))))
Ths = [gl_plus(K, 5.) for _ in range(3)]; Ws = [spd(K) for _ in range(3)]
C = inv(Ws[0]); Tc = Ths[0]
for k in range(1, 3):
    C = C + Tc @ inv(Ws[k]) @ Tc.T; Tc = Tc @ Ths[k]
nv = 4; Lp = np.zeros((nv*K, nv*K))
for k in range(3):
    R = np.zeros((K, nv*K)); R[:, k*K:(k+1)*K] = np.eye(K); R[:, (k+1)*K:(k+2)*K] = -Ths[k]; Lp += R.T @ Ws[k] @ R
Lp[:K, :K] += np.eye(K)
Sp = schur(Lp, list(range(K)) + list(range(3*K, 4*K)), list(range(K, 3*K)))
Rn = np.hstack([np.eye(K), -Tc]); pred = Rn.T @ inv(C) @ Rn; pred[:K, :K] += np.eye(K)
rep("g07 nonflat path formula (m=3)", np.abs(Sp - pred).max(), np.abs(Sp).max())
# loop: endpoints equal -> Q_loop = (I-Hol)^T W_path (I-Hol)
Ll = np.zeros((3*K, 3*K))
for (a, b, T, W) in [(0, 1, Ths[0], Ws[0]), (1, 2, Ths[1], Ws[1]), (2, 0, Ths[2], Ws[2])]:
    R = np.zeros((K, 3*K)); R[:, a*K:(a+1)*K] += np.eye(K); R[:, b*K:(b+1)*K] += -T; Ll += R.T @ W @ R
Sl = schur(Ll, list(range(K)), list(range(K, 3*K)))
Hol = Tc
rep("g07 retained loop quadratic", np.abs(Sl - (np.eye(K) - Hol).T @ inv(C) @ (np.eye(K) - Hol)).max(), np.abs(Sl).max())

# exact example
Ta = sp.Matrix([[1, 1], [0, 1]]); Tb = sp.Matrix([[1, 0], [1, 1]]); Wa = sp.Matrix([[1, 0], [0, 2]]); Wb = sp.Matrix([[2, 1], [1, 2]])
def rr(n, i, j, T):
    R = sp.zeros(2, 2*n); R[:, 2*i:2*i+2] = sp.eye(2); R[:, 2*j:2*j+2] = R[:, 2*j:2*j+2] - T; return R
Lx = sp.zeros(6, 6)
for (i, j, T, W) in [(0, 1, Ta, Wa), (1, 0, sp.eye(2), Wb), (0, 2, Tb, sp.eye(2)), (2, 0, sp.eye(2), sp.eye(2))]:
    R = rr(3, i, j, T); Lx += R.T * W * R
Lx[0:2, 0:2] += sp.eye(2)
hx = sp.Matrix([0, 0, 1, 0, 0, 0]); r = [0, 1]; hdn = [2, 3, 4, 5]
Lrh = Lx.extract(r, hdn); Lhh = Lx.extract(hdn, hdn)
Leff = sp.simplify(Lx.extract(r, r) - Lrh * Lhh.inv() * Lrh.T)
heff = sp.simplify(hx.extract(r, [0]) - Lrh * Lhh.inv() * hx.extract(hdn, [0]))
note(f"g07 nonflat example: M2={Lx.extract([2,3],[2,3]).tolist()}, M3={Lx.extract([4,5],[4,5]).tolist()}, Lambda_eff={Leff.tolist()}, h_eff={list(heff)}, det Lambda={Lx.det()}")
note(f"   h^T Lambda^-1 h = {(hx.T * Lx.inv() * hx)[0]} (claimed 17/14); mu_1 = {list(Leff.inv()*heff)}; [Ta,Tb]={(Ta*Tb-Tb*Ta).tolist()}")
al = sp.symbols('alpha', positive=True); Mm = 2 + al
target = sp.Matrix([[1 - 1/Mm, -1/Mm], [-1/Mm, 1 - 1/Mm]])
p1 = (1/Mm) * sp.Matrix([[1, -1], [-1, 1]]) + (al/Mm) * sp.eye(2)
lk = 1/(1+al); w = (1+al)/Mm
p2 = w * sp.Matrix([[1, -lk], [-lk, lk**2]]) + sp.diag(0, al/(1+al))
note(f"g07 anchored scalar chain: residuals {sp.simplify(p1-target).tolist()}, {sp.simplify(p2-target).tolist()}")
Th1, Th2 = gl_plus(K, 10.), gl_plus(K, 10.); W1, W2 = spd(K), spd(K); Ah = spd(K)
M = Th1.T @ W1 @ Th1 + W2 + Ah; Wbar = W1 - W1 @ Th1 @ inv(M) @ Th1.T @ W1
Rm = W1 @ Th1 @ inv(M) @ W2 @ Th2; Qm = Th2.T @ (W2 - W2 @ inv(M) @ W2) @ Th2
note(f"g07 anchored hidden vertex: min eig Wbar {eigvalsh(Wbar).min():.3e}, min eig (Q - R^T Wbar^-1 R) {eigvalsh(Qm - Rm.T @ inv(Wbar) @ Rm).min():.3e}, |det R| {abs(det(Rm)):.3e}")
print("\n".join(out))
