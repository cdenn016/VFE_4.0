r"""L09-OPEN: contextual geometry (open:curvature, open:rank, open:cost-equality,
open:equivariant-lift) and the generated-loss coefficient (open:generated-loss).

G1 product reference: collective connection A_chi, curvature F_{kappa chi}, tensor h, from the
   discretized flow of s4 at contexts (kappa, chi); compare with the closed forms
   A = -a/(a+b) dchi, F = -b rho I'(rho kappa)/(a+b)^2, h_chichi = n ab/(a+b), h_kk = n rho^2 Var.
G2 correlated initial laws through the equivariant lift: rank two and F != 0 on [0, 3].
G3 product-law Fisher cost equals the sum of factor costs for every xi (two factors).
G4 two identity-coupled agents: Fisher loss / t^2 -> 32/3 as t -> 0.
Numerical evidence only.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0e, i1e

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location("s4", pathlib.Path(__file__).with_name("s4_full_density_model.py"))
s4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s4)
N, M, TH = s4.N, s4.M, s4.TH


def lift(q: np.ndarray, kap: float, kap_star: float) -> np.ndarray:
    r"""open:equivariant-lift at chi = 0: tilt by e^{(kappa-kappa_*) cos(theta - vartheta_i)}, normalize."""
    out = np.empty_like(q)
    for i in range(q.shape[0]):
        th_i = np.angle(s4.first_moment(q[i]))
        w = np.exp((kap - kap_star) * np.cos(TH - th_i))[None, :]
        out[i] = q[i] * w / (q[i] * w).mean()
    return out


def evolve(q0, s0, T, ts):
    y0 = np.concatenate([q0.ravel(), s0.ravel()])
    sol = solve_ivp(s4.full_field(T), (0, ts[-1]), y0, rtol=1e-11, atol=1e-13, dense_output=True, method="RK45")
    Y = sol.sol(ts).T
    return Y[:, : N * 2 * M].reshape(-1, N, 2, M), Y[:, N * 2 * M:].reshape(-1, N, 2, M)


def fisher(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return float((a * b / p).mean())


def connection_terms(q, s, dq_k, block):
    r"""Return (A_chi, A_kappa, h_chichi, h_kk, h_kchi) for the common S^1 action on the block.
    d_chi q = -d_theta q (beliefs), d_chi s = 0; J q = -d_theta q, J s = -d_psi s."""
    Jq = {i: -s4.dtheta(q[i]) for i in block}; Js = {i: -s4.dtheta(s[i]) for i in block}
    JJ = sum(fisher(Jq[i], Jq[i], q[i]) + fisher(Js[i], Js[i], s[i]) for i in block)
    Jchi = sum(fisher(Jq[i], Jq[i], q[i]) for i in block)          # <J p, d_chi p>
    Jk = sum(fisher(Jq[i], dq_k[i], q[i]) for i in block)          # <J p, d_kappa p>
    Achi, Ak = -Jchi / JJ, -Jk / JJ
    def cost(Xq_fn, A):
        return None
    # minimized tensor h(X,Y) = <dpX + J A(X), dpY + J A(Y)>
    def vec(i, which):
        if which == "chi":
            return (Jq[i] + Achi * Jq[i], Achi * Js[i])
        return (dq_k[i] + Ak * Jq[i], Ak * Js[i])
    def h(X, Y):
        tot = 0.0
        for i in block:
            aX, bX = vec(i, X); aY, bY = vec(i, Y)
            tot += fisher(aX, aY, q[i]) + fisher(bX, bY, s[i])
        return tot
    return Achi, Ak, h("chi", "chi"), h("k", "k"), h("k", "chi")


def Ifun(v: float) -> float:
    return v * i1e(v) / i0e(v)


def main() -> None:
    A0, R, hh = 6.0, 60.0, 5e-3
    T = hh * hh
    ref = np.array([-A0 - 2, -A0 + 2, A0 - 2, A0 + 2, R - 2, R + 2])
    rng = np.random.default_rng(7)
    xq = hh * (ref + 0.05 * rng.standard_normal(N)); xs = hh * (ref + 0.05 * rng.standard_normal(N))
    kap_s, lam = 1.5, 1.0
    ph_q = rng.uniform(0, 2 * np.pi, N); ph_s = rng.uniform(0, 2 * np.pi, N)
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    dk = 1e-2
    for label, eps_c in [("G1 product reference", 0.0), ("G2 correlated eps_c=1e-2", 1e-2)]:
        cph = ph_q + 0.9
        q0 = s4.make_laws(xq, kap_s, ph_q, eps_c, cph); s0 = s4.make_laws(xs, lam, ph_s, eps_c, cph - 2.0)
        runs = {}
        for off in (-2, -1, 0, 1, 2):
            runs[off] = evolve(lift(q0, kap_s + off * dk, kap_s), s0, T, ts)
        print(f"[{label}]")
        for it, t in enumerate(ts):
            for block in [(0, 1), (0, 1, 2, 3)]:
                def terms(off):
                    q = runs[off][0][it]; s = runs[off][1][it]
                    dq = (runs[off + 1][0][it] - runs[off - 1][0][it]) / (2 * dk)
                    return connection_terms(q, s, dq, block), q, s
                (Ach0, Ak0, hcc, hkk, hkc), q, s = terms(0)
                Ach_p = terms(1)[0][0]; Ach_m = terms(-1)[0][0]
                F = (Ach_p - Ach_m) / (2 * dk)
                det = hcc * hkk - hkc**2
                n = len(block); rho = np.exp(-2 * t)
                a = Ifun(rho * kap_s); b = Ifun(rho * lam)
                v = rho * kap_s
                Ip = i1e(v) / i0e(v) + v * (1 - i1e(v) / (v * i0e(v)) - (i1e(v) / i0e(v))**2)  # I'(v) = A + v A'
                varcos = 1 - i1e(v) / (v * i0e(v)) - (i1e(v) / i0e(v))**2
                if eps_c == 0.0:
                    print(f"  t={t:.0f} n={n}: A_chi={Ach0:.6f} (formula {-a/(a+b):.6f}); F={F:.6f} (formula {-b*rho*Ip/(a+b)**2:.6f});"
                          f" h_cc={hcc:.6f} ({n*a*b/(a+b):.6f}); h_kk={hkk:.6f} ({n*rho**2*varcos:.6f}); h_kc={hkc:.1e}")
                else:
                    print(f"  t={t:.0f} n={n}: A_chi={Ach0:.6f}, F={F:.6f}, h_cc={hcc:.6f}, h_kk={hkk:.6f}, det h={det:.3e}")

    # G3: product-law Fisher cost equals sum of factor costs
    q0 = s4.make_laws(xq, kap_s, ph_q, 1e-2, ph_q + 0.3)
    f1, f2 = q0[0].ravel(), q0[1].ravel()
    d1, d2 = rng.standard_normal(f1.size), rng.standard_normal(f2.size)
    d1 -= d1.mean(); d2 -= d2.mean()
    J1 = -s4.dtheta(q0[0]).ravel(); J2 = -s4.dtheta(q0[1]).ravel()
    P = np.outer(f1, f2)
    worst = 0.0
    for xi in [-1.3, 0.0, 0.7]:
        dP = np.outer(d1 + xi * J1, f2) + np.outer(f1, d2 + xi * J2)
        lhs = float((dP**2 / P).mean())
        rhs = float(((d1 + xi * J1)**2 / f1).mean() + ((d2 + xi * J2)**2 / f2).mean())
        worst = max(worst, abs(lhs - rhs) / rhs)
    print(f"[G3] product-law Fisher cost vs sum of factor costs, max rel diff = {worst:.2e}")

    # G4: generated loss for two identity-coupled agents (finite alphabet, counting measure)
    k1 = np.array([[0.75, 0.25], [0.75, 0.25]])  # kappa_1(y|z): P(y=1)=1/4
    k2 = np.array([[0.25, 0.75], [0.25, 0.75]])  # kappa_2(y|z): P(y=1)=3/4
    r = np.full((2, 2), 0.25)

    def flow(t, y):
        q1 = y[:4].reshape(2, 2); q2 = y[4:].reshape(2, 2)
        h1 = 2 * np.log(q1) - np.log(r) - np.log(q2) - q2 / q1
        h2 = 2 * np.log(q2) - np.log(r) - np.log(q1) - q1 / q2
        return np.concatenate([(-q1 * (h1 - (q1 * h1).sum())).ravel(), (-q2 * (h2 - (q2 * h2).sum())).ravel()])

    def traj(c, t):
        mu1 = np.array([0.5, 0.5]); mu2 = np.array([0.5 + c, 0.5 - c])
        y0 = np.concatenate([(mu1[:, None] * k1).ravel(), (mu2[:, None] * k2).ravel()])
        return solve_ivp(flow, (0, t), y0, rtol=1e-13, atol=1e-15).y[:, -1]

    for t in [0.02, 0.01, 0.005, 0.001, 0.0005]:
        dc = 1e-5
        yp, ym, y0 = traj(dc, t), traj(-dc, t), traj(0.0, t)
        dy = (yp - ym) / (2 * dc)
        loss = 0.0
        for a in (slice(0, 4), slice(4, 8)):
            q = y0[a].reshape(2, 2); dq = dy[a].reshape(2, 2)
            mu = q.sum(1); dmu = dq.sum(1)
            loss += (dq**2 / q).sum() - (dmu**2 / mu).sum()
        print(f"[G4] t={t}: Fisher loss / t^2 = {loss/t**2:.5f} (claimed limit 32/3 = {32/3:.5f})")


if __name__ == "__main__":
    main()
