r"""L6-formation: numerical test of c13 prop:rg-formation-gaussian-core (b),(c).

Triangle-chain carrier of def:rg-formation-carrier for L = 2, 3. Builds B_1 (discrepancy and contrast
rows), bridge rows B_l, loop covectors, H = sum_l J_l B_l^T B_l + J_1 P_H, the flag columns Q_l spanning
U_{l-1} cap U_l^perp, and the blocks B_{sr} = B_s Q_r.  Then runs the Gaussian-core mean flow
xbar' = -H_{l'} xbar with projections at the activation times t^act_l = 2 t_l, t_l = k log(1/eps)/J_l,
and compares, on the window I_l = [t_l, 2 t_l],

    Delta_{l-1:l}(t) = (1/2T) v^T H_zz^{-1} v,  v = (H_{l-1} xbar)_z,

with the claimed asymptotic (A^2/T) J_{l+1}^4/J_l^3 y_l(v0),
y_l(v) = 1/2 |(B_ll^T B_ll)^{-3/2} Y_l B_{l+1} P_l v|^2,  Y_l = B_{l+1,l}^T B_{l+1,l+1} B_{l+1,l+1}^T.
A = T = 1 (the ratio is independent of both).  Also checks full row rank of B_{l+1,l} and the
carrier dimension formula.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy.linalg import null_space, orth

mp.mp.dps = 60


def v2(j: int) -> int:
    k = 0
    while j % 2 == 0:
        j //= 2; k += 1
    return k


def carrier(L: int, bridge_ends: str = "last-first"):
    Nt = 2 ** (L - 1)
    N = 3 * Nt
    edges = []  # (tail j, head i, level)
    for q in range(Nt):
        a, b, c = 3 * q, 3 * q + 1, 3 * q + 2
        edges += [(a, b, 1), (b, c, 1), (c, a, 1)]
    for j in range(1, Nt):
        if bridge_ends == "last-first":
            t, h = 3 * (j - 1) + 2, 3 * j
        else:
            t, h = 3 * (j - 1), 3 * j + 1
        edges.append((t, h, 2 + v2(j)))
    E = len(edges)
    dim = 2 * N + E

    def rcov(e):
        t, h, _ = edges[e]
        r = np.zeros(dim)
        r[h] += 1; r[t] -= 1; r[2 * N + e] -= 1
        return r

    O = np.array([[1 / math.sqrt(2), -1 / math.sqrt(2), 0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]])
    rows1 = []
    for i in range(N):
        d = np.zeros(dim); d[i] = 1; d[N + i] = -1
        rows1.append(d)
    for q in range(Nt):
        Rq = np.stack([rcov(3 * q + s) for s in range(3)])
        rows1 += list(O @ Rq)
    B = {1: np.array(rows1)}
    for l in range(2, L + 1):
        B[l] = np.array([rcov(e) for e in range(E) if edges[e][2] == l])
    loops = []
    for q in range(Nt):
        h = np.zeros(dim); h[2 * N + 3 * q: 2 * N + 3 * q + 3] = 1 / math.sqrt(3)
        loops.append(h)
    Hl = np.array(loops)
    return dict(L=L, Nt=Nt, N=N, E=E, dim=dim, B=B, loops=Hl, edges=edges)


def build(c, J):
    L, dim, B = c["L"], c["dim"], c["B"]
    Ball = np.vstack([B[l] for l in range(1, L + 1)])
    U = orth(Ball.T)  # basis of U = range B^T
    # flag: U_0 = U; U_l = U cap ker(B_1..B_l)
    Ulist = [U]
    for l in range(1, L + 1):
        Bs = np.vstack([B[s] for s in range(1, l + 1)])
        Wl = null_space(Bs)
        # U_l = W_l cap U
        M = np.hstack([U, -Wl])
        ns = null_space(M)
        Ul = orth(U @ ns[: U.shape[1]]) if ns.size else np.zeros((dim, 0))
        Ulist.append(Ul)
    Q = {}
    for l in range(1, L + 1):
        Uprev, Ucur = Ulist[l - 1], Ulist[l]
        P = Ucur @ Ucur.T if Ucur.shape[1] else np.zeros((dim, dim))
        X = Uprev - P @ Uprev
        Q[l] = orth(X)
    Hmat = sum(J[l - 1] * B[l].T @ B[l] for l in range(1, L + 1)) + J[0] * c["loops"].T @ c["loops"]
    return U, Ulist, Q, Hmat


def main() -> None:
    for L in (2, 3):
        c = carrier(L)
        Nt, N, E, dim = c["Nt"], c["N"], c["E"], c["dim"]
        rkB = np.linalg.matrix_rank(np.vstack([c["B"][l] for l in range(1, L + 1)]))
        dims = [dim - np.linalg.matrix_rank(np.vstack([c["B"][s] for s in range(1, l + 1)])) for l in range(1, L + 1)]
        pred = [4 * Nt - 1 + Nt // 2 ** (l - 1) for l in range(1, L + 1)]
        print(f"L={L}: dim={dim} (10Nt-1={10*Nt-1}), rank B={rkB} (6Nt-1={6*Nt-1}), dim W_l={dims} pred {pred}")
        for ends in ("last-first", "other"):
            cc = carrier(L, ends)
            _, Ul, Q, _ = build(cc, [1.0] * L)
            for l in range(1, L):
                Bl1l = cc["B"][l + 1] @ Q[l]
                print(f"   bridge ends {ends}: B_({l+1},{l}) shape {Bl1l.shape}, rank {np.linalg.matrix_rank(Bl1l)}, "
                      f"B_ll square {cc['B'][l].shape[0]=}=={Q[l].shape[1]}")
        rng = np.random.default_rng(1)
        for rho, kmode in ((1e-1, 'fixed'), (1e-2, 'fixed'), (1e-3, 'fixed'), (1e-5, 'adaptive'), (1e-6, 'adaptive')):
            J = [rho ** (-(L - l)) for l in range(1, L + 1)]  # J_l = rho^{-(L-l)}, J_L = 1, ratios rho
            U, Ulist, Q, H = build(c, J)
            jmin = min(np.linalg.eigvalsh((c['B'][l] @ Q[l]).T @ (c['B'][l] @ Q[l])).min() for l in range(1, L))
            if kmode == 'fixed':
                eps, k = rho, 40.0
            else:
                eps = 0.1
                k = 3 * 4 * math.log(1 / rho) / (jmin * math.log(1 / eps))
            print(f"  rho={rho:g}: min_l j_l={jmin:.4f}, eps(window)={eps:g}, k={k:.2f}, k log(1/eps) rho={k*math.log(1/eps)*rho:.2e}")
            v0 = U @ rng.normal(size=U.shape[1]); v0 /= np.linalg.norm(v0)
            # mean flow with activations: level l' active on [t^act_{l'}, t^act_{l'+1}]
            t_l = [k * math.log(1 / eps) / J[l - 1] for l in range(1, L + 1)]
            tact = [0.0] + [2 * t for t in t_l]
            xbar0 = v0.copy()
            for lp in range(0, L):  # active level lp
                # basis of active normal space U_{lp}
                Ua = Ulist[lp]
                Hlp = None
                # Schur complement of H|_U onto U_lp: eliminate U cap U_lp^perp
                Ub = U
                Pe = orth(Ub - Ua @ (Ua.T @ Ub)) if lp > 0 else np.zeros((dim, 0))
                if lp == 0:
                    Hs = Ua.T @ H @ Ua
                else:
                    Haa = Ua.T @ H @ Ua; Hae = Ua.T @ H @ Pe; Hee = Pe.T @ H @ Pe
                    Hs = Haa - Hae @ np.linalg.solve(Hee, Hae.T)
                ya = Ua.T @ xbar0  # coordinates on active space (projection P_lp)
                l = lp + 1
                if l > L - 1:
                    break
                # discarded z = layer l, retained y = layers > l, expressed in active coordinates
                Zc = Ua.T @ Q[l]  # columns: layer l in active coords (orthonormal)
                lam, V = np.linalg.eigh(Hs)
                Hzz = Zc.T @ Hs @ Zc
                Bll = c["B"][l] @ Q[l]
                Bl1l = c["B"][l + 1] @ Q[l]
                Bl1l1 = c["B"][l + 1] @ Q[l + 1]
                Y = Bl1l.T @ Bl1l1 @ Bl1l1.T
                G = Bll.T @ Bll
                ew, evv = np.linalg.eigh(G)
                Gm32 = evv @ np.diag(ew ** -1.5) @ evv.T
                # P_l v0 : projection onto W_l
                Wl = null_space(np.vstack([c["B"][s] for s in range(1, l + 1)]))
                Pl = Wl @ Wl.T
                y_v0 = 0.5 * np.linalg.norm(Gm32 @ Y @ c["B"][l + 1] @ Pl @ v0) ** 2
                scale = J[l] ** 4 / J[l - 1] ** 3
                ratios = []
                for s in np.linspace(t_l[l - 1], 2 * t_l[l - 1], 5):
                    tau = s - tact[lp]
                    # high-precision evaluation of v = (Hs ybar)_z with ybar = exp(-Hs tau) ya
                    lamm = [mp.mpf(x) for x in lam]
                    coef = [mp.mpf(x) for x in (V.T @ ya)]
                    g = [lamm[i] * mp.e ** (-lamm[i] * tau) * coef[i] for i in range(len(lamm))]
                    Vz = Zc.T @ V
                    vz = np.array([float(mp.fsum(Vz[r, i] * g[i] for i in range(len(g)))) for r in range(Vz.shape[0])])
                    Delta = 0.5 * vz @ np.linalg.solve(Hzz, vz)
                    ybar = V @ (np.exp(-lam * tau) * (V.T @ ya))
                    xb = Ua @ ybar
                    y_t = 0.5 * np.linalg.norm(Gm32 @ Y @ c["B"][l + 1] @ Pl @ xb) ** 2
                    ratios.append((Delta / (scale * y_v0), Delta / (scale * y_t)))
                r = np.array(ratios)
                print(f"   L={L} rho={rho:g} interface l={l}: Delta/(scale*y(v0)) on I_l in [{r[:,0].min():.4f},{r[:,0].max():.4f}], "
                      f"Delta/(scale*y(xbar/A)) in [{r[:,1].min():.4f},{r[:,1].max():.4f}]")
                # advance mean to activation of next level, project onto W_{l}
                tau_end = tact[l] - tact[lp]
                ybar = V @ (np.exp(-lam * tau_end) * (V.T @ ya))
                xbar0 = Ulist[l] @ (Ulist[l].T @ (Ua @ ybar))


if __name__ == "__main__":
    main()
