r"""D20-CTXC symbolic checks of the continuum part of chapters/15_contextual_coupling.tex.

Each check is an exact sympy identity (simplified residual 0) unless stated.  These
are formula checks; they do not prove the Sobolev theorem.  Single threaded, small.

Checks
  1. Block inversion of the adapted Fisher metric and both dissipation forms
     (ctx:c:eq:shape, ctx:c:eq:fiber, ctx:c:eq:dissipation), generic 2x2 blocks.
  2. Polar fixed-reference example (ctx:c:eq:nonbasic): metric, V_u, V_r,
     zero-coupling Cartesian reconstruction, and the positive-coupling Cartesian
     flow p_t = -(p - a e1) + lambda D_x D_x p in polar form.
  3. Witness reference-joint KL and Hessian H (ctx:c:eq:witnessV).
  4. Witness eigenvector, w, ansatz alpha_*, D_x P, D_x D_x P, horizontality,
     exact energy (ctx:c:eq:exactenergy), its derivative, n=1 vs n=2 velocity.
  5. Zero-coupling counterexample and its realization as a normalized KL potential.
  6. Elementary heat-lemma inequalities.
  7. Hopf-cone adapted chart for SO(2) on C^2 (used by hopf_flow_check.py):
     metric decomposition, Christoffel symbols, mechanical connection, curvature.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sympy as sp

RESULTS: list[dict[str, str]] = []


def record(name: str, residual: sp.Expr | sp.Matrix | bool, note: str = "") -> None:
    """Simplify a residual and record whether it vanishes identically."""
    if isinstance(residual, bool):
        ok = residual
        shown = str(residual)
    else:
        simp = sp.simplify(residual)
        if isinstance(simp, sp.MatrixBase):
            ok = all(sp.simplify(e) == 0 for e in simp)
        else:
            ok = simp == 0
        shown = str(simp)
    RESULTS.append({"check": name, "zero_residual": str(ok), "residual": shown[:200], "note": note})
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {shown[:120]} {note}")


def check_block_inversion() -> None:
    r"""Metric M=[[h+C^T k C, C^T k],[k C, k]]; check M z_t = -(V_u - lambda h tau, V_r).

    Verified by multiplication (no symbolic solve): with hinv, kinv the adjugate inverses,
    the residual M v + euler is a rational function that must cancel to zero.  Sizes
    n_u = 2, n_r = 2 with fully generic symmetric h, k and generic C.
    """
    h11, h12, h22, k11, k12, k22 = sp.symbols("h11 h12 h22 k11 k12 k22", real=True)
    c = sp.Matrix(2, 2, sp.symbols("c11 c12 c21 c22", real=True))
    h = sp.Matrix([[h11, h12], [h12, h22]])
    k = sp.Matrix([[k11, k12], [k12, k22]])
    hinv = h.adjugate() / h.det()
    kinv = k.adjugate() / k.det()
    vu = sp.Matrix(sp.symbols("Vu1 Vu2", real=True))
    vr = sp.Matrix(sp.symbols("Vr1 Vr2", real=True))
    tau = sp.Matrix(sp.symbols("t1 t2", real=True))
    lam = sp.symbols("lam", positive=True)
    euler_u = vu - lam * h * tau
    ut = lam * tau - hinv * (vu - c.T * vr)
    rt = -c * ut - kinv * vr
    row_u = (h + c.T * k * c) * ut + c.T * k * rt + euler_u
    row_r = k * c * ut + k * rt + vr
    record("block inversion: first block row of M z_t + dF = 0", row_u.applyfunc(sp.cancel))
    record("block inversion: second block row of M z_t + dF = 0", row_r.applyfunc(sp.cancel))
    diss1 = -(ut.T * h * ut + (rt + c * ut).T * k * (rt + c * ut))[0]
    diss2 = -(ut.T * h * ut + vr.T * kinv * vr)[0]
    pairing = (euler_u.T * ut + vr.T * rt)[0]
    record("dissipation first form = pairing dF(z_t)", sp.cancel(pairing - diss1))
    record("dissipation second form = first form", sp.cancel(diss1 - diss2))
    # Horizontal derivative: u-component w_u of grad_G V solves M w = (V_u, V_r).
    wu = hinv * (vu - c.T * vr)
    wr = kinv * vr - c * wu
    res1 = (h + c.T * k * c) * wu + c.T * k * wr - vu
    res2 = k * c * wu + k * wr - vr
    record("grad V has u-part h^{-1}(V_u - C^T V_r) (horizontal derivative H_a V)",
           sp.Matrix.vstack(res1, res2).applyfunc(sp.cancel))


def check_polar() -> None:
    r"""p = u R_{r+eta u} e1, reference mean a e1, SO(2)."""
    u, a = sp.symbols("u a", positive=True)
    r, eta, lam = sp.symbols("r eta lam", real=True)
    phi = r + eta * u
    p = sp.Matrix([u * sp.cos(phi), u * sp.sin(phi)])
    jac = p.jacobian([u, r])
    g = sp.simplify(jac.T * jac)
    g_claim = sp.Matrix([[1 + u**2 * eta**2, u**2 * eta], [u**2 * eta, u**2]])
    record("polar metric = du^2 + u^2 (dr + eta du)^2", g - g_claim)
    e1 = sp.Matrix([1, 0])
    v = sp.Rational(1, 2) * ((p - a * e1).T * (p - a * e1))[0]
    record("polar V = (u^2 + a^2 - 2 a u cos phi)/2", v - (u**2 + a**2 - 2 * a * u * sp.cos(phi)) / 2)
    vu, vr = sp.diff(v, u), sp.diff(v, r)
    record("polar V_r = a u sin phi", vr - a * u * sp.sin(phi))
    record("polar V_u = u - a cos phi + a eta u sin phi", vu - (u - a * sp.cos(phi) + a * eta * u * sp.sin(phi)))
    h, k, cc = 1, u**2, eta
    ut = -(vu - cc * vr) / h
    rt = -cc * ut - vr / k
    record("polar zero-coupling u_t = -u + a cos phi", ut - (-u + a * sp.cos(phi)))
    record("polar zero-coupling r_t = -eta u_t - (a/u) sin phi", rt - (-eta * ut - a / u * sp.sin(phi)))
    pt_chart = jac * sp.Matrix([ut, rt])
    record("polar zero coupling reconstructs p_t = -(p - a e1)", pt_chart + (p - a * e1))
    record("omitting C^T V_r changes u_t by -a eta u sin phi", (-(vu) / h) - ut + a * eta * u * sp.sin(phi))
    # Positive coupling with spatial dependence u(x), phi(x).
    x = sp.symbols("x", real=True)
    uf = sp.Function("U")(x)
    ff = sp.Function("Phi")(x)
    jm = sp.Matrix([[0, -1], [1, 0]])
    px = sp.Matrix([uf * sp.cos(ff), uf * sp.sin(ff)])
    alpha = -((px.diff(x)).T * (jm * px))[0] / (px.T * px)[0]
    record("polar alpha_* = -phi_x", sp.simplify(alpha + ff.diff(x)))
    dxp = px.diff(x) + alpha * jm * px
    dxdxp = dxp.diff(x) + alpha * jm * dxp
    erad = sp.Matrix([sp.cos(ff), sp.sin(ff)])
    record("polar D_x D_x p = u_xx e_rad", sp.simplify(dxdxp - uf.diff(x, 2) * erad))
    # Cartesian flow vs chart flow at positive coupling (r = phi - eta u).
    pt_cart = -(px - a * e1) + lam * dxdxp
    ut_c = lam * uf.diff(x, 2) - uf + a * sp.cos(ff)
    phit_c = -(a / uf) * sp.sin(ff)  # = r_t + eta u_t from ctx:c:eq:nonbasic
    pt_from_chart = erad * ut_c + uf * (jm * erad) * phit_c
    record("polar positive coupling: chart system = Cartesian Fisher flow", sp.simplify(pt_cart - pt_from_chart))


def check_witness_potential() -> sp.Matrix:
    r"""KL(q x s || r G) and the 4x4 Hessian H."""
    m1, m2, n1, n2 = sp.symbols("m1 m2 n1 n2", real=True)
    mu = sp.Matrix([m1, m2])
    nu = sp.Matrix([n1, n2])
    # (k, m) joint reference: m ~ N(0, I), k | m ~ N(m, I): Sigma = [[2I, I], [I, I]].
    sig = sp.Matrix(sp.BlockMatrix([[2 * sp.eye(2), sp.eye(2)], [sp.eye(2), sp.eye(2)]]))
    sinv = sig.inv()
    mean = sp.Matrix.vstack(mu, nu)
    kl = sp.Rational(1, 2) * (sinv.trace() - 4 + (mean.T * sinv * mean)[0] + sp.log(sig.det()))
    claim = sp.Rational(1, 2) * ((mu - nu).T * (mu - nu))[0] + sp.Rational(1, 2) * (nu.T * nu)[0] + 1
    record("witness KL(q x s || r G) = |mu-nu|^2/2 + |nu|^2/2 + 1", kl - claim)
    c = sp.symbols("c1:5", real=True)  # one planar coordinate suffices for the column Gram form
    quad = (sp.Rational(1, 2) * (c[0] - c[1])**2 + sp.Rational(1, 2) * c[1]**2
            + sp.Rational(1, 2) * (c[2] - c[3])**2 + sp.Rational(1, 2) * c[3]**2
            + (c[0] - c[2])**2 + (c[1] - c[3])**2)
    hess = sp.hessian(quad, c)
    h_claim = sp.Matrix([[3, -1, -2, 0], [-1, 4, 0, -2], [-2, 0, 3, -1], [0, -2, -1, 4]])
    record("witness Hessian H (ctx:c:eq:witnessV)", hess - h_claim)
    return h_claim


def check_witness(hmat: sp.Matrix) -> None:
    tau = (sp.sqrt(5) - 1) / 2
    kappa = 5 - tau
    v = sp.Matrix([1, tau, -1, -tau])
    record("H v = kappa v", sp.expand(hmat * v - kappa * v))
    record("kappa = (11 - sqrt 5)/2", sp.simplify(kappa - (11 - sp.sqrt(5)) / 2))
    ev = sorted([sp.nsimplify(e) for e in hmat.eigenvals().keys()], key=lambda e: float(e))
    record("kappa is an eigenvalue of H (spectrum listed)", any(sp.simplify(e - kappa) == 0 for e in ev),
           f"eigenvalues {[sp.simplify(e) for e in ev]}")
    w = (v.T * v)[0]
    record("w = ||V0||^2 = 5 - sqrt 5", sp.simplify(w - (5 - sp.sqrt(5))))
    x, t = sp.symbols("x t", real=True)
    lam, a0, a1 = sp.symbols("lam a0 a1", positive=True)
    n = sp.symbols("n", positive=True, integer=True)
    jm = sp.Matrix([[0, -1], [1, 0]])
    phi = sp.Function("phi")(x)
    amp = sp.Function("A")(t, x)
    rot = sp.cos(phi) * sp.eye(2) + sp.sin(phi) * jm
    v0 = sp.Matrix([[1], [0]]) * v.T
    pm = rot * amp * v0

    def fro(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
        return sum(a[i, j] * b[i, j] for i in range(a.rows) for j in range(a.cols))

    px = pm.diff(x)
    alpha = sp.simplify(-fro(px, jm * pm) / fro(pm, pm))
    record("witness alpha_* = -phi_x", sp.simplify(alpha + phi.diff(x)))
    dxp = px + alpha * jm * pm
    record("witness D_x P = R_phi a_x V0", sp.simplify(dxp - rot * amp.diff(x) * v0))
    dxdxp = dxp.diff(x) + alpha * jm * dxp
    record("witness D_x D_x P = R_phi a_xx V0", sp.simplify(dxdxp - rot * amp.diff(x, 2) * v0))
    record("witness -PH = -kappa P on the ansatz", sp.simplify(-pm * hmat + kappa * pm))
    # Horizontality of PH for a general 2x4 P.
    pg = sp.Matrix(2, 4, sp.symbols("p0:8", real=True))
    record("<PH, JP>_F = 0 for symmetric H", sp.expand(fro(pg * hmat, jm * pg)))
    # Exact energy and dissipation (normalized measure on S^1).
    aexact = a0 * sp.exp(-kappa * t) + a1 * sp.exp(-(kappa + lam * n**2) * t) * sp.cos(n * x)
    pde = sp.diff(aexact, t) - (lam * sp.diff(aexact, x, 2) - kappa * aexact)
    record("a(t,x) solves a_t = lam a_xx - kappa a", sp.simplify(pde))
    dens = 2 + kappa * w / 2 * aexact**2 + lam * w / 2 * sp.diff(aexact, x)**2
    energy = sp.simplify(sp.integrate(sp.expand(dens), (x, 0, 2 * sp.pi)) / (2 * sp.pi))
    claim = (2 + w * kappa * a0**2 / 2 * sp.exp(-2 * kappa * t)
             + w * (kappa + lam * n**2) * a1**2 / 4 * sp.exp(-2 * (kappa + lam * n**2) * t))
    record("exact energy (ctx:c:eq:exactenergy)", sp.simplify(energy - claim))
    dclaim = (-w * kappa**2 * a0**2 * sp.exp(-2 * kappa * t)
              - w / 2 * (kappa + lam * n**2)**2 * a1**2 * sp.exp(-2 * (kappa + lam * n**2) * t))
    record("dF/dt formula", sp.simplify(sp.diff(claim, t) - dclaim))
    kinetic = sp.simplify(sp.integrate(sp.expand(w * sp.diff(aexact, t)**2), (x, 0, 2 * sp.pi)) / (2 * sp.pi))
    record("dF/dt = -int ||P_t||^2", sp.simplify(dclaim + kinetic))
    # Same law and first derivative at x=0 for n=1,2; velocity difference.
    a_n = lambda nn: a0 + a1 * sp.cos(nn * x)  # noqa: E731
    vel = lambda nn: (lam * sp.diff(a_n(nn), x, 2) - kappa * a_n(nn)).subs(x, 0)  # noqa: E731
    same = [sp.simplify((a_n(2) - a_n(1)).subs(x, 0)), sp.simplify(sp.diff(a_n(2) - a_n(1), x).subs(x, 0))]
    record("n=1,2 profiles agree with first derivative at x=0", sp.Matrix(same))
    record("velocity difference = -3 lam a1 (times R_phi(0) V0)", sp.simplify(vel(2) - vel(1) + 3 * lam * a1))
    record("lower bound a >= e^{-kappa t}(a0-|a1|)", sp.simplify(
        (a0 * sp.exp(-kappa * t) - a1 * sp.exp(-(kappa + lam * n**2) * t))
        - sp.exp(-kappa * t) * (a0 - a1 * sp.exp(-lam * n**2 * t))))


def check_zero_coupling() -> None:
    t, a, u0, r0 = sp.symbols("t a u0 r0", real=True)
    u = sp.exp(-t) * (sp.cosh(a * t) * u0 + sp.sinh(a * t) * r0)
    r = sp.exp(-t) * (sp.sinh(a * t) * u0 + sp.cosh(a * t) * r0)
    record("zero coupling: u_t = -(u - a r)", sp.simplify(sp.diff(u, t) + (u - a * r)))
    record("zero coupling: r_t = -(r - a u)", sp.simplify(sp.diff(r, t) + (r - a * u)))
    # Realization: belief N(u,1), model N(r,1), reference N(0, 1/(1-a^2)), evaluator N(a m, 1).
    aa = sp.symbols("aa", positive=True)
    var_m = 1 / (1 - aa**2)
    sig = sp.Matrix([[aa**2 * var_m + 1, aa * var_m], [aa * var_m, var_m]])  # (k, m) order
    record("correlated reference precision = [[1,-a],[-a,1]]", sp.simplify(sig.inv() - sp.Matrix([[1, -aa], [-aa, 1]])))


def check_heat_inequalities() -> None:
    a = sp.symbols("a", nonnegative=True)
    y = sp.symbols("y", real=True)
    # f(a) = a - (1-e^{-a})^2 has f(0)=0 and f'(a) = 1 - 2 y (1 - y), y=e^{-a} in (0,1].
    fprime = sp.diff(a - (1 - sp.exp(-a))**2, a)
    record("f'(a) = 1 - 2 e^{-a}(1 - e^{-a})", sp.simplify(fprime - (1 - 2 * sp.exp(-a) * (1 - sp.exp(-a)))))
    record("max over y of 2y(1-y) is 1/2 < 1, so (1-e^{-a})^2 <= a", sp.maximum(2 * y * (1 - y), y, sp.Interval(0, 1)) == sp.Rational(1, 2))
    kk = sp.symbols("kk", positive=True)
    record("|k|^2 >= (1+|k|^2)/2 for |k|^2 >= 1 (difference at kk=1+z)",
           sp.simplify((kk - (1 + kk) / 2).subs(kk, 1 + sp.Symbol("z", nonnegative=True))) - sp.Symbol("z", nonnegative=True) / 2)


def check_hopf_chart() -> None:
    rho = sp.symbols("rho", positive=True)
    th, ps, ch = sp.symbols("theta psi chi", real=True)
    c2, s2 = sp.cos(th / 2), sp.sin(th / 2)
    psi_map = sp.Matrix([rho * c2 * sp.cos(ch), rho * c2 * sp.sin(ch),
                         rho * s2 * sp.cos(ch + ps), rho * s2 * sp.sin(ch + ps)])
    coords = [rho, th, ps, ch]
    jac = psi_map.jacobian(coords)
    g = sp.simplify(jac.T * jac)
    h = sp.diag(1, rho**2 / 4, rho**2 * sp.sin(th)**2 / 4)
    k = rho**2
    cvec = sp.Matrix([[0, 0, s2**2]])
    g_claim = sp.zeros(4, 4)
    g_claim[:3, :3] = h + cvec.T * k * cvec
    g_claim[:3, 3] = cvec.T * k
    g_claim[3, :3] = k * cvec
    g_claim[3, 3] = k
    record("Hopf chart metric = h + k (dchi + sin^2(theta/2) dpsi)^2", sp.simplify(g - g_claim))
    jm4 = sp.Matrix([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    record("J P = d Psi/d chi (fiber is the SO(2) orbit)", sp.simplify(jm4 * psi_map - psi_map.diff(ch)))
    hpsi = psi_map.diff(ps) - s2**2 * psi_map.diff(ch)
    record("H_psi orthogonal to the orbit", sp.simplify((hpsi.T * psi_map.diff(ch))[0]))
    # Christoffel symbols of h in (rho, theta, psi).
    q = [rho, th, ps]
    hinv = h.inv()
    gam = [[[sp.simplify(sum(hinv[a, d] * (sp.diff(h[d, b], q[c]) + sp.diff(h[d, c], q[b]) - sp.diff(h[b, c], q[d]))
                             for d in range(3)) / 2) for c in range(3)] for b in range(3)] for a in range(3)]
    nonzero = {f"Gamma^{a}_{b}{c}": str(gam[a][b][c]) for a in range(3) for b in range(3) for c in range(3) if gam[a][b][c] != 0}
    RESULTS.append({"check": "Hopf quotient Christoffel symbols (rho,theta,psi)=(0,1,2)", "zero_residual": "n/a",
                    "residual": json.dumps(nonzero), "note": "used in hopf_flow_check.py"})
    print("Christoffel:", nonzero)
    # Mechanical connection theta = dchi + sin^2(theta/2) dpsi, curvature (1/2) sin(theta) dtheta^dpsi.
    curv = sp.diff(s2**2, th)
    record("d(sin^2(theta/2)) / d theta = sin(theta)/2", sp.simplify(curv - sp.sin(th) / 2))


def main() -> None:
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    start = time.perf_counter()
    if part in ("all", "a"):
        check_block_inversion()
        check_zero_coupling()
        check_heat_inequalities()
    if part in ("all", "b"):
        check_polar()
    if part in ("all", "c"):
        hmat = check_witness_potential()
        check_witness(hmat)
    if part in ("all", "d"):
        check_hopf_chart()
    out = {"script": "symbolic_checks.py", "part": part, "sympy": sp.__version__,
           "runtime_s": round(time.perf_counter() - start, 2),
           "all_zero": all(r["zero_residual"] in ("True", "n/a") for r in RESULTS), "results": RESULTS}
    Path(__file__).with_name(f"symbolic_checks_{part}.out.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ALL ZERO:", out["all_zero"], "runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
