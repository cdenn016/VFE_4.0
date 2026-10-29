r"""D19-CTXF s2: the free-action branch singularity of chapters/15_contextual_coupling.tex:364-373
(ctx:f:eq:cusp) and what happens to the minimized-energy flow at the singular point.

(a) exact check of min_R 1/2||x_eps - R y||_F^2 = 2 + eps + eps^2/2 - |eps| and of its one-sided slopes;
(b) at eps = 0 every R in SO(2) minimizes.  By Danskin's theorem the directional derivative of the
    minimized energy C_min(x,y) = min_R c(x,y,R) is min over minimizers of dc_R; the steepest-descent
    rate along -grad c_R is -||grad c_R||^2, which is the same for every R here.  Integrate the pure
    contextual flow (lambda = 1, no local potential) from the singular point for several initial branch
    choices R_theta0 and check that (i) for t > 0 the chosen R_theta0 becomes the unique global minimizer,
    so each trajectory is a classical gradient-flow solution of C_min on (0, t], and (ii) the trajectories
    are distinct.  This is non-uniqueness of the gradient flow of the minimized energy at the singular point.
Single-threaded numpy; small sizes; deterministic.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot(t: float) -> np.ndarray:
    return np.cos(t) * np.eye(2) + np.sin(t) * J


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """C = <x, y>_F, S = <x, J y>_F for target x and source y (cost 1/2||x - R y||^2)."""
    return float(np.sum(x * y)), float(np.sum(x * (J @ y)))


def global_angle(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    C, S = corr(x, y)
    return float(np.arctan2(S, C)), float(np.hypot(C, S))


def main() -> None:
    lines: list[str] = []
    # (a) exact cusp
    eps = sp.symbols("epsilon", real=True)
    th = sp.symbols("theta", real=True)
    Jm = sp.Matrix([[0, -1], [1, 0]])
    R = sp.cos(th) * sp.eye(2) + sp.sin(th) * Jm
    x = sp.Matrix([[1, 0], [0, 1 + eps]])  # columns e1, (1+eps) e2
    y = sp.Matrix([[1, 0], [0, -1]])       # columns e1, -e2
    c = sp.expand(sp.Rational(1, 2) * sum((x - R * y).applyfunc(lambda t: t**2)))
    c = sp.simplify(sp.expand_trig(c))
    C = sp.simplify(sum(x.multiply_elementwise(y)))
    S = sp.simplify(sum(x.multiply_elementwise(Jm * y)))
    base = sp.simplify(c + C * sp.cos(th) + S * sp.sin(th))
    lines.append(f"(a) C = {C}, S = {S}, cost - [base - C cos - S sin] has base = {base}")
    cmin = base - sp.Abs(C)  # min over theta of base - C cos(theta)
    claim = 2 + eps + eps**2 / 2 - sp.Abs(eps)
    lines.append(f"    min_R cost - manuscript formula = {sp.simplify(cmin - claim)}")
    lines.append(f"    left slope at 0 = {sp.limit(sp.diff(2 + eps + eps**2/2 + eps, eps), eps, 0)}, "
                 f"right slope at 0 = {sp.limit(sp.diff(2 + eps + eps**2/2 - eps, eps), eps, 0)}")

    # (b) non-uniqueness at the singular point eps = 0
    x0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    y0 = np.array([[1.0, 0.0], [0.0, -1.0]])
    C0, S0 = corr(x0, y0)
    lines.append(f"(b) at eps=0: C = {C0}, S = {S0}; cost is constant on SO(2): "
                 f"{[round(0.5*np.sum((x0 - rot(t) @ y0)**2), 12) for t in (0.0, 0.7, 2.1, np.pi)]}")
    dt, nsteps = 1e-4, 200
    finals = {}
    for th0 in (0.0, np.pi / 3, 2 * np.pi / 3, np.pi, -np.pi / 2):
        xa, ya = x0.copy(), y0.copy()
        theta = th0  # branch continued from the chosen minimizer
        slope0 = None
        for k in range(nsteps):
            Rm = rot(theta)
            gx = xa - Rm @ ya          # grad_x c_R  (target endpoint)
            gy = ya - Rm.T @ xa        # grad_y c_R  (source endpoint)
            if k == 0:
                slope0 = -(np.sum(gx**2) + np.sum(gy**2))  # d/dt c along -grad c_R
            xa, ya = xa - dt * gx, ya - dt * gy
            # continue the branch: maximize <x, R y> locally (Newton on theta)
            for _ in range(3):
                Cc, Ss = corr(xa, ya)
                f1 = Cc * np.sin(theta) - Ss * np.cos(theta)      # d cost / d theta
                f2 = Cc * np.cos(theta) + Ss * np.sin(theta)      # second derivative
                theta -= f1 / f2
        ang, rho = global_angle(xa, ya)
        dist = float(np.angle(np.exp(1j * (ang - theta))))
        finals[th0] = (xa.copy(), ya.copy())
        lines.append(f"    theta0 = {th0:+.4f}: initial slope -||grad c||^2 = {slope0:.6f}; at t = {dt*nsteps:.3f}: "
                     f"global minimizer angle - branch angle = {dist:+.2e}, rho = {rho:.3e}, "
                     f"x = {np.round(xa, 5).tolist()}")
    keys = list(finals)
    seps = [float(np.linalg.norm(finals[a][0] - finals[b][0]) + np.linalg.norm(finals[a][1] - finals[b][1]))
            for i, a in enumerate(keys) for b in keys[i + 1:]]
    lines.append(f"    min pairwise separation of the endpoint states at t = {dt*nsteps:.3f}: {min(seps):.3e} "
                 f"(baseline displacement from start ~ {np.linalg.norm(finals[keys[0]][0]-x0):.3e})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
