r"""D20-CTXC witnesses for the scope statements of the continuum part of chapter 15.

(a) Total-energy dissipation does not make the individual summands monotone:
    exact solutions in which the contextual Dirichlet term increases, and in which the
    local reciprocal term increases, while the total reduced energy decreases.
(b) Finite-time arrival at the singular orbit with bounded Sobolev norms:
    (b1) the manuscript's polar fixed-reference example with phi = pi,
    (b2) the two-agent, two-sector witness model with shifted model references N(c, I).
(c) Frame regularity: in a coupled system of the theorem's form whose horizontal
    coefficient has curvature (Heisenberg type), the naive L^1_t H^s bound on the frame
    velocity grows without bound along H^s-bounded data, while the frame displacement
    itself stays bounded in H^s.  Illustration of the persistence argument, not a proof.

Single threaded; sympy for exact identities, numpy/scipy for small matrix exponentials
and one-dimensional quadrature.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import sympy as sp
from scipy.linalg import expm

OUT: dict[str, object] = {}


def part_a() -> None:
    r"""V = (u - cos x)^2/2 + r^2/2 on the translation-invariant location family (h = k = 1, C = 0)."""
    t, x = sp.symbols("t x", real=True)
    lam = sp.symbols("lam", positive=True)
    res = {}
    for label, c0 in (("u0=0, r0=1", 0), ("u0=cos x, r0=0", 1)):
        c = sp.Function("c")(t)
        sol = sp.dsolve(sp.Eq(c.diff(t), -(1 + lam) * c + 1), c, ics={c.subs(t, 0): c0}).rhs
        u = sol * sp.cos(x)
        r = sp.exp(-t) if c0 == 0 else sp.Integer(0)
        pde = sp.simplify(sp.diff(u, t) - (lam * sp.diff(u, x, 2) - (u - sp.cos(x))))
        mean = lambda f: sp.simplify(sp.integrate(sp.expand(f), (x, 0, 2 * sp.pi)) / (2 * sp.pi))  # noqa: E731
        dirichlet = mean(lam / 2 * sp.diff(u, x) ** 2)
        local = mean((u - sp.cos(x)) ** 2 / 2 + r**2 / 2)
        total = sp.simplify(dirichlet + local)
        kinetic = mean(sp.diff(u, t) ** 2 + sp.diff(r, t) ** 2)
        res[label] = {
            "pde_residual": str(pde),
            "dirichlet(t)": str(sp.simplify(dirichlet)),
            "local(t)": str(sp.simplify(local)),
            "dF/dt + kinetic": str(sp.simplify(sp.diff(total, t) + kinetic)),
            "d dirichlet/dt at t=0.5, lam=1": float(sp.diff(dirichlet, t).subs({t: 0.5, lam: 1})),
            "d local/dt at t=0.5, lam=1": float(sp.diff(local, t).subs({t: 0.5, lam: 1})),
            "d total/dt at t=0.5, lam=1": float(sp.diff(total, t).subs({t: 0.5, lam: 1})),
        }
    OUT["a_nonmonotone_summands"] = res


def part_b() -> None:
    # (b1) polar fixed reference a e1, phi = pi, spatially constant u.
    t, a, u0 = sp.symbols("t a u0", positive=True)
    u = -a + (u0 + a) * sp.exp(-t)
    tstar = sp.log(1 + u0 / a)
    b1 = {
        "u_t + u - a cos(pi)": str(sp.simplify(sp.diff(u, t) + u - a * sp.cos(sp.pi))),
        "phi_t = -(a/u) sin(pi)": str(sp.simplify(-(a / u) * sp.sin(sp.pi))),
        "u(t*)": str(sp.simplify(u.subs(t, tstar))),
        "t*": str(tstar),
        "energy (u+a)^2/2 derivative": str(sp.simplify(sp.diff((u + a) ** 2 / 2, t))),
        "Cartesian p(t) = (a - (u0+a) e^{-t}) e1 passes 0 at t*": str(sp.simplify((a - (u0 + a) * sp.exp(-t)).subs(t, tstar))),
    }
    OUT["b1_polar_singular_orbit"] = b1
    # (b2) two agents, two sectors, model references N(c, I_2): grad V = P H - c e^T, e = (0,1,0,1).
    hmat = np.array([[3.0, -1, -2, 0], [-1, 4, 0, -2], [-2, 0, 3, -1], [0, -2, -1, 4]])
    e = np.array([0.0, 1, 0, 1])
    cvec = np.array([0.7, -0.4])

    def potential(p: np.ndarray) -> float:
        mu1, nu1, mu2, nu2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        val = 0.0
        for mu, nu in ((mu1, nu1), (mu2, nu2)):
            val += 0.5 * np.sum((mu - nu) ** 2) + 0.5 * np.sum((nu - cvec) ** 2) + 1.0
        return val + np.sum((mu1 - mu2) ** 2) + np.sum((nu1 - nu2) ** 2)

    rng = np.random.default_rng(29)
    ptest = rng.normal(size=(2, 4))
    fd = np.zeros_like(ptest)
    for idx in np.ndindex(ptest.shape):
        dp = np.zeros_like(ptest)
        dp[idx] = 1e-6
        fd[idx] = (potential(ptest + dp) - potential(ptest - dp)) / 2e-6
    grad_err = float(np.max(np.abs(fd - (ptest @ hmat - np.outer(cvec, e)))))
    pinf = np.outer(cvec, e) @ np.linalg.inv(hmat)
    tstar_num = 1.0
    p0 = pinf @ (np.eye(4) - expm(hmat * tstar_num))
    ts = np.linspace(0.0, tstar_num, 2001)
    traj = [pinf + (p0 - pinf) @ expm(-hmat * tt) for tt in ts]
    norms = np.array([np.linalg.norm(q) for q in traj])
    energies = np.array([potential(q) for q in traj])
    # ODE residual P_t + P H - c e^T at a few times by centered differences.
    ode_res = 0.0
    for tt in (0.2, 0.5, 0.8):
        pp = pinf + (p0 - pinf) @ expm(-hmat * (tt + 1e-6))
        pm = pinf + (p0 - pinf) @ expm(-hmat * (tt - 1e-6))
        pc = pinf + (p0 - pinf) @ expm(-hmat * tt)
        ode_res = max(ode_res, float(np.max(np.abs((pp - pm) / 2e-6 + pc @ hmat - np.outer(cvec, e)))))
    OUT["b2_two_agent_singular_orbit"] = {
        "grad V = PH - c e^T, max fd error": grad_err,
        "ODE residual": ode_res,
        "||P(0)||": float(norms[0]),
        "||P(t*)||": float(norms[-1]),
        "min ||P|| on [0, 0.99 t*]": float(norms[ts <= 0.99 * tstar_num].min()),
        "energy decreasing on grid": bool(np.all(np.diff(energies) < 1e-12)),
        "V(P(0)), V(P(t*))": [float(energies[0]), float(energies[-1])],
        "rank of P(t) (columns parallel to c)": int(np.linalg.matrix_rank(traj[1000], tol=1e-10)),
    }


def part_c() -> None:
    r"""Heisenberg-type coupled system: h = I_2, k = 1, C = (-u2/2, u1/2), V = |u|^2/2 (basic).

    u_t = lam Delta u - u, r_t = -C(u) u_t.  With u1(0) = 1 and u2(0) = g:
    r(t) - r(0) = -(lam/2) int_0^t e^{-2 tau} Delta S_tau g d tau, a Fourier multiplier
    m_t(l) = (lam l^2 / 2)(1 - e^{-(2 + lam l^2) t}) / (2 + lam l^2) in [0, 1/2).
    """
    lam, s, tend = 0.7, 3, 1.0
    rows = []
    for jmax in (4, 8, 16, 24, 32, 40):
        js = np.arange(1, jmax + 1)
        freq = 2.0 ** js
        weight = (1 + freq**2) ** s
        coef = (1.0 / js) / np.sqrt(weight)          # c_j <K_j>^s = 1/j
        g_hs = np.sqrt(np.sum(coef**2 * weight / 2))  # ||cos(Kx)||^2 = 1/2 in normalized measure
        mult = (lam * freq**2 / 2) * (1 - np.exp(-(2 + lam * freq**2) * tend)) / (2 + lam * freq**2)
        frame_hs = np.sqrt(np.sum((mult * coef) ** 2 * weight / 2))
        taus = np.concatenate([[0.0], np.logspace(-30, np.log10(tend), 40000)])
        integrand = []
        for tau in taus:
            amp = (lam / 2) * np.exp(-2 * tau) * freq**2 * np.exp(-lam * tau * freq**2) * coef
            integrand.append(np.sqrt(np.sum(amp**2 * weight / 2)))
        naive = float(np.trapezoid(np.array(integrand), taus))
        rows.append({"J": int(jmax), "||g||_Hs": float(g_hs), "||r(T)-r0||_Hs": float(frame_hs),
                     "naive int_0^T ||r_t||_Hs dt": naive, "ratio naive/actual": naive / frame_hs})
    OUT["c_frame_persistence_illustration"] = {"lambda": lam, "s": s, "T": tend, "rows": rows,
                                              "bound ||r(T)-r0||_Hs <= ||g||_Hs / 2": all(
                                                  r["||r(T)-r0||_Hs"] <= 0.5 * r["||g||_Hs"] + 1e-15 for r in rows)}


def main() -> None:
    start = time.perf_counter()
    part_a()
    part_b()
    part_c()
    OUT["runtime_s"] = round(time.perf_counter() - start, 2)
    Path(__file__).with_name("scope_witnesses.out.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
    print(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
