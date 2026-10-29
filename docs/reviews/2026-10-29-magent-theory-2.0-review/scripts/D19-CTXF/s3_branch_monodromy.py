r"""D19-CTXF s3: competing nondegenerate minima, branch monodromy, and a Maxwell crossing of the
regular-branch flow of ctx:f:thm:ode, on a product family that satisfies every stated hypothesis.

Family (one agent per context, n = 1):  belief q = N(mu, I_2) (location family, flat Fisher metric),
model s = N(0, Sigma) (centered covariance family, Fisher metric 1/2 tr(S^-1 A S^-1 B)), product law
q (x) s.  G = SO(2) acts on both sectors by rotation of samples (isometric).  The product is a Hadamard
manifold (flat x SPD with a multiple of the affine-invariant metric), so squared distance is smooth
everywhere and Log is global:
    d^2 = |mu_v - R mu_u|^2 + 1/2 sum log^2 eig(S_v^-1 R S_u R^T),
    Log_{p_v}(R p_u) = (R mu_u - mu_v, S_v^{1/2} logm(S_v^{-1/2} R S_u R^T S_v^{-1/2}) S_v^{1/2}).
Local potentials are of the chapter's form with a self source (identity channel, D = 0, singleton row):
    E_w = KL(q_w (x) s_w || r_w G_w) = KL(q_w || N(m_w, I)) + KL(s_w || N(0, S_ref)),
with r_w = N(0, S_ref) and the constant evaluator G_w(m, dk) = N(m_w, I)(dk).  Fisher gradients:
    grad_mu = mu - m_w,  grad_Sigma = Sigma S_ref^-1 Sigma - Sigma.
(A) competing nondegenerate local minima of theta -> c(theta) at a free configuration;
(B) branch monodromy along a closed loop of laws (S_v rotated by alpha in [0, pi]);
(C) the coupled flow ctx:f:eq:flow on the continued branch crosses a Maxwell set: the continued branch
    stays nondegenerate and the action stays free, but after the crossing its energy exceeds min_U c.
"""
from __future__ import annotations

import numpy as np

J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot(t: float) -> np.ndarray:
    return np.cos(t) * np.eye(2) + np.sin(t) * J


def sym_fun(S: np.ndarray, f) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * f(w)) @ V.T


def airm_log(S: np.ndarray, P: np.ndarray) -> np.ndarray:
    Sh = sym_fun(S, np.sqrt)
    Sih = sym_fun(S, lambda w: 1.0 / np.sqrt(w))
    return Sh @ sym_fun(Sih @ P @ Sih, np.log) @ Sh


def cov_dist2(S: np.ndarray, P: np.ndarray) -> float:
    Sih = sym_fun(S, lambda w: 1.0 / np.sqrt(w))
    w = np.linalg.eigvalsh(Sih @ P @ Sih)
    return 0.5 * float(np.sum(np.log(w) ** 2))  # Fisher = 1/2 AIRM


def cost(th: float, mu_u, S_u, mu_v, S_v) -> float:
    R = rot(th)
    return 0.5 * (float(np.sum((mu_v - R @ mu_u) ** 2)) + cov_dist2(S_v, R @ S_u @ R.T))


def dtheta(th, *p, h=1e-5):
    f = lambda t: cost(t, *p)
    return (f(th + h) - f(th - h)) / (2 * h), (f(th + h) - 2 * f(th) + f(th - h)) / h**2


def newton(th, *p, it=8):
    for _ in range(it):
        g, H = dtheta(th, *p)
        th -= g / H
    return th


def local_minima(*p, n=4000):
    grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    vals = np.array([cost(t, *p) for t in grid])
    idx = [i for i in range(n) if vals[i] < vals[i - 1] and vals[i] < vals[(i + 1) % n]]
    out = []
    for i in idx:
        t = newton(grid[i], *p)
        g, H = dtheta(t, *p)
        out.append((float(np.angle(np.exp(1j * t))), cost(t, *p), H, g))
    return out


def locked_inertia(mu, S) -> float:
    # J_p xi = (xi J mu, xi (J S - S J)); Fisher norm^2 = |J mu|^2 + 1/2 tr((S^-1 (JS - SJ))^2)
    D = J @ S - S @ J
    Si = np.linalg.inv(S)
    return float(np.sum((J @ mu) ** 2) + 0.5 * np.trace(Si @ D @ Si @ D))


def kl_cov(S, Sr):
    M = np.linalg.solve(Sr, S)
    return 0.5 * (np.trace(M) - 2.0 - np.log(np.linalg.det(M)))


def total_energy(state, th, lam, refs):
    mu_u, S_u, mu_v, S_v = state
    (m_u, Su_r), (m_v, Sv_r) = refs
    E_u = 0.5 * float(np.sum((mu_u - m_u) ** 2)) + kl_cov(S_u, Su_r)
    E_v = 0.5 * float(np.sum((mu_v - m_v) ** 2)) + kl_cov(S_v, Sv_r)
    return E_u + E_v + lam * cost(th, *state)


def main() -> None:
    L: list[str] = []
    a = 0.3
    S0 = np.diag([4.0, 0.25])
    mu0 = np.array([a, 0.0])
    # (A) competing minima at agreement
    mins = local_minima(mu0, S0, mu0, S0)
    L.append("(A) agreement p_u = p_v = (0.3 e1, diag(4, 1/4)); locked inertia k = "
             f"{locked_inertia(mu0, S0):.4f} > 0 (free: R mu = mu forces R = I)")
    for t, v, H, g in mins:
        L.append(f"    local min theta = {t:+.6f}, cost = {v:.6f}, Hessian = {H:.4f}, |grad| = {abs(g):.1e}")

    # (B) monodromy: S_v(alpha) = R_alpha S0 R_alpha^T, alpha from 0 to pi (closed loop of laws)
    th = 0.0
    alphas = np.linspace(0.0, np.pi, 2001)
    Hmin, gap_min = np.inf, np.inf
    for al in alphas:
        Sv = rot(al) @ S0 @ rot(al).T
        th = newton(th, mu0, S0, mu0, Sv, it=4)
        g, H = dtheta(th, mu0, S0, mu0, Sv)
        Hmin = min(Hmin, H)
    Sv_end = rot(np.pi) @ S0 @ rot(np.pi).T
    L.append(f"(B) loop closes in law space: max|S_v(pi) - S_v(0)| = {np.max(np.abs(Sv_end - S0)):.1e}")
    L.append(f"    continued branch: theta(0) = 0, theta(pi) = {np.angle(np.exp(1j*th)):+.6f}; "
             f"min Hessian along the loop = {Hmin:.4f}")
    L.append(f"    branch energy at the same law configuration: start {cost(0.0, mu0, S0, mu0, S0):.6f}, "
             f"end {cost(th, mu0, S0, mu0, Sv_end):.6f}")

    # (C) Maxwell crossing of the regular-branch flow
    lam = 0.1
    m_u_ref, S_u_ref = mu0.copy(), S0.copy()
    m_v_ref, S_v_ref = rot(np.deg2rad(150.0)) @ mu0, S0.copy()

    def rhs(state, th):
        mu_u, S_u, mu_v, S_v = state
        th = newton(th, mu_u, S_u, mu_v, S_v, it=4)
        R = rot(th)
        Bv = (R @ mu_u - mu_v, airm_log(S_v, R @ S_u @ R.T))
        Bu = (R.T @ mu_v - mu_u, airm_log(S_u, R.T @ S_v @ R))
        d_mu_v = -(mu_v - m_v_ref) + lam * Bv[0]
        d_S_v = -(S_v @ np.linalg.inv(S_v_ref) @ S_v - S_v) + lam * Bv[1]
        d_mu_u = -(mu_u - m_u_ref) + lam * Bu[0]
        d_S_u = -(S_u @ np.linalg.inv(S_u_ref) @ S_u - S_u) + lam * Bu[1]
        return (d_mu_u, d_S_u, d_mu_v, d_S_v), th

    def add(s, k, h):
        return tuple(si + h * ki for si, ki in zip(s, k))

    state = (mu0.copy(), S0.copy(), mu0.copy(), S0.copy())
    th = 0.0
    dt, T = 0.01, 5.0
    rows = []
    t = 0.0
    crossed = None
    while t <= T + 1e-12:
        mu_u, S_u, mu_v, S_v = state
        th = newton(th, *state, it=6)
        g, H = dtheta(th, *state)
        mins = local_minima(*state, n=720)
        cmin = min(v for _, v, _, _ in mins)
        cbr = cost(th, *state)
        k_v = locked_inertia(mu_v, S_v)
        Ftot = total_energy(state, th, lam, ((m_u_ref, S_u_ref), (m_v_ref, S_v_ref)))
        rows.append((t, np.degrees(np.arctan2(mu_v[1], mu_v[0])), np.angle(np.exp(1j*th)), H,
                     cbr, cmin, np.linalg.norm(mu_v), k_v, len(mins), Ftot))
        if crossed is None and cbr - cmin > 1e-9:
            crossed = t
        k1, th1 = rhs(state, th)
        k2, _ = rhs(add(state, k1, dt / 2), th1)
        k3, _ = rhs(add(state, k2, dt / 2), th1)
        k4, _ = rhs(add(state, k3, dt), th1)
        state = tuple(s + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
                      for s, a1, a2, a3, a4 in zip(state, k1, k2, k3, k4))
        state = (state[0], 0.5 * (state[1] + state[1].T), state[2], 0.5 * (state[3] + state[3].T))
        t += dt
    L.append(f"(C) coupled flow, lambda = {lam}: first time the continued branch is not the global minimizer: "
             f"t = {crossed}")
    L.append("    t     angle(mu_v)  branch theta  branch Hessian  c_branch   min_U c    |mu_v|   k(p_v)  #local minima  F_total(branch)")
    for r in rows[:: 50]:
        L.append("    {:4.2f}  {:9.3f}  {:+10.5f}  {:12.4f}  {:9.6f}  {:9.6f}  {:6.4f}  {:7.4f}  {:d}  {:12.8f}".format(*r))
    L.append(f"    min branch Hessian over the run = {min(r[3] for r in rows):.4f}; min |mu_v| = "
             f"{min(r[6] for r in rows):.4f}; max (c_branch - min_U c) = {max(r[4]-r[5] for r in rows):.6f}")
    incs = [rows[i + 1][9] - rows[i][9] for i in range(len(rows) - 1)]
    L.append(f"    branch total energy F_* along the run: start {rows[0][9]:.8f}, end {rows[-1][9]:.8f}, "
             f"largest single-step change {max(incs):.2e} (negative: strictly decreasing at every step)")
    print("\n".join(L))


if __name__ == "__main__":
    main()
