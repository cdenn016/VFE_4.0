r"""D19-CTXF s5: holonomy of the finite minimizing transports around a small square versus the
curvature of the collective (mechanical) connection A^F of geo:collective / ctx:f:eq:normal.

Unit-covariance planar location family, n = 3 agents (a 2x3 matrix of means), SO(2) acting diagonally.
For a smooth section p(c), c in R^2, the infinitesimal normal equation gives
    k = ||J p||^2 = ||p||^2,   A^F(X) = -<dp(X), J p>_F / ||p||^2,
and (abelian) F = dA^F.  The finite edge transport from u to v is the Procrustes minimizer
U_{v<-u} = R_{atan2(S, C)}, C = <p_v, p_u>_F, S = <p_v, J p_u>_F.  For the square
c0 -> c0+eps e1 -> c0+eps(e1+e2) -> c0+eps e2 -> c0 the holonomy angle is the sum of edge angles.
Prediction (reviewer's sketch): hol(eps) = -eps^2 F_12(center) + O(eps^3), from the odd midpoint
expansion of each edge transport (reversal symmetry U_{u<-v} = U_{v<-u}^{-1}).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

c1, c2 = sp.symbols("c1 c2", real=True)
P0 = sp.Matrix([[1.0, 0.2, -0.5], [0.1, 0.9, 0.4]])
P1 = sp.Matrix([[0.3, -0.7, 0.2], [0.5, 0.1, -0.6]])
P2 = sp.Matrix([[-0.4, 0.6, 0.1], [0.2, -0.3, 0.8]])
P3 = sp.Matrix([[0.5, 0.1, -0.2], [-0.3, 0.4, 0.2]])
P4 = sp.Matrix([[0.2, 0.3, 0.1], [0.1, -0.2, 0.3]])
p = P0 + c1 * P1 + c2 * P2 + c1 * c2 * P3 + c1**2 * P4
Jm = sp.Matrix([[0, -1], [1, 0]])
fro = lambda A, B: sum(A.multiply_elementwise(B))
nrm2 = fro(p, p)
A1 = -fro(sp.diff(p, c1), Jm * p) / nrm2
A2 = -fro(sp.diff(p, c2), Jm * p) / nrm2
F12 = sp.simplify(sp.diff(A2, c1) - sp.diff(A1, c2))
p_num = sp.lambdify((c1, c2), p, "numpy")
F_num = sp.lambdify((c1, c2), F12, "numpy")
A_num = sp.lambdify((c1, c2), (A1, A2), "numpy")
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def edge_angle(cu, cv) -> float:
    pu, pv = np.array(p_num(*cu), dtype=float), np.array(p_num(*cv), dtype=float)
    C, S = float(np.sum(pv * pu)), float(np.sum(pv * (J @ pu)))
    return float(np.arctan2(S, C))


def main() -> None:
    c0 = np.array([0.3, -0.2])
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    lines = []
    lines.append(f"F_12 at c0 = {float(F_num(*c0)):+.6f}; A^F(c0) = {tuple(round(float(v), 6) for v in A_num(*c0))}")
    prev = None
    for eps in (0.2, 0.1, 0.05, 0.025, 0.0125):
        verts = [c0, c0 + eps * e1, c0 + eps * (e1 + e2), c0 + eps * e2, c0]
        hol = sum(edge_angle(verts[k], verts[k + 1]) for k in range(4))
        center = c0 + 0.5 * eps * (e1 + e2)
        pred = -eps**2 * float(F_num(*center))
        err = hol - pred
        rate = "" if prev is None else f", error ratio vs previous eps = {prev / err:.2f}"
        lines.append(f"eps = {eps:<7} holonomy = {hol:+.6e}, -eps^2 F_12(center) = {pred:+.6e}, "
                     f"relative error = {err / pred:+.2e}{rate}")
        prev = err
    # first-order edge check: angle / eps -> -A(X) at the midpoint, error O(eps^2)
    for eps in (0.1, 0.01):
        mid = c0
        ang = edge_angle(mid - 0.5 * eps * e1, mid + 0.5 * eps * e1)
        lines.append(f"edge (symmetric about c0, direction e1), eps = {eps}: angle/eps = {ang/eps:+.8f}, "
                     f"-A^F_1(c0) = {-float(A_num(*c0)[0]):+.8f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
