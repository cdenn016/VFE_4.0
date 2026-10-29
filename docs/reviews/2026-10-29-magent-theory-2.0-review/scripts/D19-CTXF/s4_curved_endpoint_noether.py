r"""D19-CTXF s4: ctx:f:statement:1 and ctx:f:statement:3 on a curved admitted family, plus the Noether
horizontality of the minimizing geodesic.

Same product family as s3 (belief N(mu, I_2), model N(0, Sigma), Fisher metric flat x 1/2 AIRM; SO(2)
acting on both sectors), but with n = 2 agents per context so that the collective is a 2-agent product.
Checks, each against an explicit baseline:
 (i)  endpoint gradients at FIXED U: d/ds c(x + s dx, y, U) = -G_x(Log_x(U y), dx) and
      d/ds c(x, y + s dy, U) = -G_y(Log_y(U^{-1} x), dy), by central differences on the closed-form cost;
 (ii) at the minimizing U: G_z(Log_z x, J_z xi) = 0 (stationarity) AND G_x(Log_x z, J_x xi) = 0
      (Noether: momentum <gamma', xi*> is constant along a geodesic for a Killing field xi*), so the
      contextual force B_v = Log_{p_v}(U p_u) is Fisher-orthogonal to the orbit at p_v;
      at a non-stationary U the two momenta are equal and opposite-signed as predicted, and nonzero;
 (iii) momentum conservation along the AIRM x Euclidean geodesic at several interior times;
 (iv) envelope ctx:f:eq:envelope: directional derivative of C_*(p) = c(p, U*(p)) (U re-minimized) equals
      the endpoint formula.
"""
from __future__ import annotations

import numpy as np

J = np.array([[0.0, -1.0], [1.0, 0.0]])
rng = np.random.default_rng(20261029)


def rot(t):
    return np.cos(t) * np.eye(2) + np.sin(t) * J


def sym_fun(S, f):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * f(w)) @ V.T


def airm_log(S, P):
    Sh, Sih = sym_fun(S, np.sqrt), sym_fun(S, lambda w: 1 / np.sqrt(w))
    return Sh @ sym_fun(Sih @ P @ Sih, np.log) @ Sh


def airm_exp(S, V):
    Sh, Sih = sym_fun(S, np.sqrt), sym_fun(S, lambda w: 1 / np.sqrt(w))
    return Sh @ sym_fun(Sih @ V @ Sih, np.exp) @ Sh


# a collective point: list of agents, each (mu, Sigma)
def act(R, p):
    return [(R @ m, R @ S @ R.T) for m, S in p]


def dist2(p, q):
    tot = 0.0
    for (m1, S1), (m2, S2) in zip(p, q):
        Sih = sym_fun(S1, lambda w: 1 / np.sqrt(w))
        tot += float(np.sum((m2 - m1) ** 2)) + 0.5 * float(np.sum(np.log(np.linalg.eigvalsh(Sih @ S2 @ Sih)) ** 2))
    return tot


def Log(p, q):
    return [(m2 - m1, airm_log(S1, S2)) for (m1, S1), (m2, S2) in zip(p, q)]


def G(p, a, b):
    tot = 0.0
    for (m, S), (am, aS), (bm, bS) in zip(p, a, b):
        Si = np.linalg.inv(S)
        tot += float(am @ bm) + 0.5 * float(np.trace(Si @ aS @ Si @ bS))
    return tot


def Jt(p):
    return [(J @ m, J @ S - S @ J) for m, S in p]


def perturb(p, d, s):
    return [(m + s * dm, S + s * dS) for (m, S), (dm, dS) in zip(p, d)]


def rand_spd():
    A = rng.normal(size=(2, 2))
    return A @ A.T + 0.3 * np.eye(2)


def rand_point():
    return [(rng.normal(size=2), rand_spd()) for _ in range(2)]


def rand_tangent():
    out = []
    for _ in range(2):
        B = rng.normal(size=(2, 2))
        out.append((rng.normal(size=2), 0.5 * (B + B.T)))
    return out


def cost(x, y, th):
    return 0.5 * dist2(x, act(rot(th), y))


def argmin_theta(x, y, th0=None):
    grid = np.linspace(-np.pi, np.pi, 721, endpoint=False) if th0 is None else [th0]
    th = min(grid, key=lambda t: cost(x, y, t))
    h = 1e-5
    for _ in range(30):
        g = (cost(x, y, th + h) - cost(x, y, th - h)) / (2 * h)
        H = (cost(x, y, th + h) - 2 * cost(x, y, th) + cost(x, y, th - h)) / h**2
        th -= g / H
    return th


def main():
    L = []
    h = 1e-6
    errs_x, errs_y, scale = [], [], []
    for _ in range(5):
        x, y = rand_point(), rand_point()
        th = rng.uniform(-np.pi, np.pi)
        U = rot(th)
        dx, dy = rand_tangent(), rand_tangent()
        fdx = (cost(perturb(x, dx, h), y, th) - cost(perturb(x, dx, -h), y, th)) / (2 * h)
        fdy = (cost(x, perturb(y, dy, h), th) - cost(x, perturb(y, dy, -h), th)) / (2 * h)
        ax = -G(x, Log(x, act(U, y)), dx)
        ay = -G(y, Log(y, act(U.T, x)), dy)
        errs_x.append(abs(fdx - ax)); errs_y.append(abs(fdy - ay)); scale.append(max(abs(ax), abs(ay)))
    L.append(f"(i) endpoint gradients at fixed U (5 random curved configurations): max |FD - formula| = "
             f"{max(errs_x):.2e} (target), {max(errs_y):.2e} (source); typical |derivative| = {np.median(scale):.2e}")

    m_stat, m_non = [], []
    for _ in range(5):
        x, y = rand_point(), rand_point()
        th = argmin_theta(x, y)
        z = act(rot(th), y)
        a_z = G(z, Log(z, x), Jt(z))
        a_x = G(x, Log(x, z), Jt(x))
        m_stat.append((abs(a_z), abs(a_x)))
        thn = th + 0.7
        zn = act(rot(thn), y)
        b_z = G(zn, Log(zn, x), Jt(zn))
        b_x = G(x, Log(x, zn), Jt(x))
        m_non.append((b_z, b_x))
    L.append(f"(ii) at the minimizing U: max |G_z(Log_z x, J_z)| = {max(a for a,_ in m_stat):.2e}, "
             f"max |G_x(Log_x z, J_x)| = {max(b for _,b in m_stat):.2e}")
    L.append("     at U rotated by 0.7 rad from the minimizer: (G_z(Log_z x,J_z), G_x(Log_x z,J_x)) = "
             + ", ".join(f"({bz:+.4f},{bx:+.4f})" for bz, bx in m_non))

    # (iii) momentum along the geodesic gamma(t) from x to z at a non-stationary U
    x, y = rand_point(), rand_point()
    th = argmin_theta(x, y) + 0.5
    z = act(rot(th), y)
    V = Log(x, z)
    moms = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        gam = [(m + t * vm, airm_exp(S, t * vS)) for (m, S), (vm, vS) in zip(x, V)]
        # velocity of the AIRM geodesic Sigma(t) = S^{1/2} exp(t L) S^{1/2}: transported V
        vel = []
        for (m, S), (vm, vS) in zip(x, V):
            Sh, Sih = sym_fun(S, np.sqrt), sym_fun(S, lambda w: 1 / np.sqrt(w))
            Lm = Sih @ vS @ Sih
            E = sym_fun(Lm, lambda w: np.exp(t * w))
            vel.append((vm, Sh @ (Lm @ E) @ Sh))
        moms.append(G(gam, vel, Jt(gam)))
    L.append(f"(iii) momentum <gamma'(t), xi*> at t = 0, .25, .5, .75, 1: {[round(v, 10) for v in moms]}")

    # (iv) envelope of the reduced energy on the minimizing branch
    errs, mags = [], []
    for _ in range(4):
        x, y = rand_point(), rand_point()
        th = argmin_theta(x, y)
        dx, dy = rand_tangent(), rand_tangent()
        Cst = lambda s: cost(perturb(x, dx, s), perturb(y, dy, s), argmin_theta(perturb(x, dx, s), perturb(y, dy, s), th))
        fd = (Cst(1e-5) - Cst(-1e-5)) / 2e-5
        U = rot(th)
        formula = -G(x, Log(x, act(U, y)), dx) - G(y, Log(y, act(U.T, x)), dy)
        errs.append(abs(fd - formula)); mags.append(abs(formula))
    L.append(f"(iv) envelope: max |FD of C_* - formula| = {max(errs):.2e}; typical |dC_*| = {np.median(mags):.2e}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
