r"""D19-CTXF s1: exact recomputation of the two-context witness of
chapters/15_contextual_coupling.tex:299-362 (ctx:f:eq:examplelocal, ctx:f:eq:procrustes,
ctx:f:eq:witnessvelocity) and of the dissipation identity ctx:f:eq:dissipation on it.

Everything is exact rational/symbolic arithmetic in sympy.  Checks:
  (1) KL(q (x) s || r G) = 1/2|mu-nu|^2 + 1/2|nu|^2 + 1 from the closed Gaussian KL on R^4;
  (2) E(X) and grad E at X = (e1,e1,e2,e2);
  (3) planar Procrustes: c(theta) = 1/2(|X|^2+|Y|^2) - C cos - S sin, R_*, c_*, Hessian rho;
  (4) envelope gradients of c_* = 1/2(|X|^2+|Y|^2) - sqrt(C^2+S^2) equal X - R_*^T Y and Y - R_* X;
  (5) source-context velocity with symbolic lambda, and the neighbor-change increment;
  (6) total energy 37.5 and dF/dt = -||velocity||^2 = -323 at lambda = 1, and the identity for
      symbolic lambda.
"""
from __future__ import annotations

import sympy as sp

half = sp.Rational(1, 2)


def gaussian_kl(m0: sp.Matrix, S0: sp.Matrix, m1: sp.Matrix, S1: sp.Matrix) -> sp.Expr:
    r"""KL(N(m0,S0) || N(m1,S1)) = 1/2[tr(S1^{-1}S0) + (m1-m0)^T S1^{-1}(m1-m0) - k + log det S1/det S0]."""
    k = S0.shape[0]
    S1i = S1.inv()
    d = m1 - m0
    return sp.simplify(half * ((S1i * S0).trace() + (d.T * S1i * d)[0, 0] - k
                               + sp.log(S1.det() / S0.det())))


def main() -> None:
    out: list[str] = []
    # (1) joint reference: m ~ N(0,I2), k | m ~ N(m, I2); order (k, m).
    mu = sp.Matrix(sp.symbols("mu1 mu2", real=True))
    nu = sp.Matrix(sp.symbols("nu1 nu2", real=True))
    I2 = sp.eye(2)
    Z2 = sp.zeros(2)
    S_ref = sp.BlockMatrix([[2 * I2, I2], [I2, I2]]).as_explicit()
    S_rec = sp.eye(4)
    m_rec = sp.Matrix.vstack(mu, nu)
    m_ref = sp.zeros(4, 1)
    kl = gaussian_kl(m_rec, S_rec, m_ref, S_ref)
    claim = half * (mu - nu).dot(mu - nu) + half * nu.dot(nu) + 1
    out.append(f"(1) KL(q x s || rG) - claim = {sp.simplify(sp.expand(kl - claim))}")

    # (2) local potential on X = (mu1, nu1, mu2, nu2) columns in R^2
    xs = sp.symbols("x0:8", real=True)
    Xs = sp.Matrix(2, 4, xs)
    def E_of(M: sp.Matrix) -> sp.Expr:
        m1, n1, m2, n2 = (M[:, j] for j in range(4))
        val = 0
        for (a, b) in ((m1, n1), (m2, n2)):
            val += half * (a - b).dot(a - b) + half * b.dot(b) + 1
        # two directed identity-channel KLs between unit-covariance Gaussians, each sector
        val += 2 * half * (m1 - m2).dot(m1 - m2) + 2 * half * (n1 - n2).dot(n1 - n2)
        return sp.expand(val)
    E = E_of(Xs)
    gradE = sp.Matrix(2, 4, [sp.diff(E, v) for v in xs])
    e1 = sp.Matrix([1, 0]); e2 = sp.Matrix([0, 1])
    X = sp.Matrix.hstack(e1, e1, e2, e2)
    Y = sp.Matrix.hstack(2 * e1, 3 * e1, e2, 2 * e2)
    subsX = dict(zip(xs, list(X.T.reshape(8, 1)) if False else [X[i, j] for i in range(2) for j in range(4)]))
    gX = gradE.subs(subsX)
    g_claim = sp.Matrix([[2, 3, -2, -2], [-2, -2, 2, 3]])
    out.append(f"(2) grad E(X) = {gX.tolist()} ; minus manuscript = {(gX - g_claim).tolist()}")

    # (3) planar Procrustes
    J = sp.Matrix([[0, -1], [1, 0]])
    th = sp.symbols("theta", real=True)
    R = sp.cos(th) * I2 + sp.sin(th) * J
    C = sum((Y.multiply_elementwise(X)))
    S = sum((Y.multiply_elementwise(J * X)))
    rho = sp.sqrt(C**2 + S**2)
    cfun = half * sum(((Y - R * X).applyfunc(lambda t: t**2)))
    cform = half * (sum(X.applyfunc(lambda t: t**2)) + sum(Y.applyfunc(lambda t: t**2))) - C * sp.cos(th) - S * sp.sin(th)
    out.append(f"(3) (C,S,rho) = ({C},{S},{rho}); c(theta) - closed form = {sp.simplify(cfun - cform)}")
    Rstar = (C * I2 + S * J) / rho
    cstar = sp.simplify(cfun.subs(th, 0))
    hess = sp.simplify(sp.diff(cfun, th, 2).subs(th, 0))
    stat = sp.simplify(sp.diff(cfun, th).subs(th, 0))
    out.append(f"    R_* = {Rstar.tolist()}, c_* = {cstar}, stationarity at 0 = {stat}, Hessian = {hess} (rho = {rho})")

    # (4) envelope gradients of the reduced Procrustes energy, symbolic generic X, Y
    ys = sp.symbols("y0:8", real=True)
    Ys = sp.Matrix(2, 4, ys)
    Cg = sum(Ys.multiply_elementwise(Xs)); Sg = sum(Ys.multiply_elementwise(J * Xs))
    rg = sp.sqrt(Cg**2 + Sg**2)
    cst = half * (sum(Xs.applyfunc(lambda t: t**2)) + sum(Ys.applyfunc(lambda t: t**2))) - rg
    Rg = (Cg * I2 + Sg * J) / rg
    gx = sp.Matrix(2, 4, [sp.diff(cst, v) for v in xs])
    gy = sp.Matrix(2, 4, [sp.diff(cst, v) for v in ys])
    rx = (gx - (Xs - Rg.T * Ys)).applyfunc(sp.simplify)
    ry = (gy - (Ys - Rg * Xs)).applyfunc(sp.simplify)
    out.append(f"(4) envelope residuals: X-part {set(rx)}, Y-part {set(ry)}")

    # (5) velocity at the source context with symbolic lambda
    lam = sp.symbols("lambda", nonnegative=True)
    Bu = Rstar.T * Y - X  # Log_X(R_*^T Y) in the flat location metric
    Xdot = -gX + lam * Bu
    Xdot_claim = sp.Matrix([[-2 + lam, -3 + 2 * lam, 2, 2], [2, 2, -2, -3 + lam]])
    out.append(f"(5) Xdot - manuscript = {(Xdot - Xdot_claim).applyfunc(sp.simplify).tolist()}")
    Y2 = Y.copy(); Y2[:, 0] = 3 * e1
    C2 = sum(Y2.multiply_elementwise(X)); S2 = sum(Y2.multiply_elementwise(J * X))
    R2 = (C2 * I2 + S2 * J) / sp.sqrt(C2**2 + S2**2)
    Xdot2 = -gX + lam * (R2.T * Y2 - X)
    out.append(f"    changed neighbor: R_* = {R2.tolist()}, Xdot change = {(Xdot2 - Xdot).tolist()}")

    # (6) total energy and dissipation; F(X,Y) = E(X) + E(Y) + lam * c_*(X,Y)
    F = E_of(Xs) + E_of(Ys) + lam * cst
    pt = {**{xs[i * 4 + j]: X[i, j] for i in range(2) for j in range(4)},
          **{ys[i * 4 + j]: Y[i, j] for i in range(2) for j in range(4)}}
    gradF = sp.Matrix([sp.diff(F, v) for v in list(xs) + list(ys)]).subs(pt)
    vel = -gradF  # unit masses, flat Fisher metric on means
    dFdt = sp.simplify((gradF.T * vel)[0, 0])
    normsq = sp.simplify((vel.T * vel)[0, 0])
    F1 = sp.nsimplify(F.subs(pt).subs(lam, 1))
    out.append(f"(6) F at lambda=1: {F1}; dF/dt + ||vel||^2 = {sp.simplify(dFdt + normsq)}; "
               f"||vel||^2 at lambda=1: {normsq.subs(lam, 1)}")
    Ydot = vel[8:, 0].reshape(4, 2).T if False else sp.Matrix(2, 4, list(vel[8:, 0]))
    out.append(f"    target-context velocity (not displayed in manuscript): {Ydot.applyfunc(sp.expand).tolist()}")
    out.append(f"    ||vel||^2 as polynomial in lambda: {sp.expand(normsq)}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
