"""Small CPU formula checks for the standalone Item 5 construction.

These checks concern the displayed examples, not the general existence proofs.
"""

from __future__ import annotations

import os

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import json
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

Array = NDArray[np.float64]
RESULTS: dict[str, object] = {}


def record(name: str, residual: float, tolerance: float) -> None:
    """Require a finite residual below the stated instance-specific tolerance."""
    RESULTS[name] = {"residual": float(residual), "tolerance": tolerance}
    assert np.isfinite(residual) and residual < tolerance, (name, residual)


def norm(matrix: Array) -> float:
    """Return the Frobenius norm."""
    return float(np.linalg.norm(matrix))


def X(point: Array) -> Array:
    """Return the 2 by 3 message matrix [I_2, v]."""
    return np.column_stack((np.eye(2), point))


def right_inverse(matrix: Array) -> Array:
    r"""Compute Q_X = X^T (X X^T)^{-1}."""
    return np.linalg.solve(matrix @ matrix.T, matrix).T


def cross(target: Array, source: Array) -> Array:
    r"""Compute C(c', c) = X(c') Q_X(c)."""
    return X(target) @ right_inverse(X(source))


def reciprocal(target: Array, source: Array) -> Array:
    r"""Compute T = C [C_reverse C]^{-1/2} on the tested local branch."""
    direct = cross(target, source)
    backtrack = cross(source, target) @ direct
    return np.asarray(direct @ np.linalg.inv(sqrtm(backtrack)), dtype=float)


def connection(point: Array) -> list[Array]:
    r"""Return the two coefficients of a = -dv v^T/(1+v^T v)."""
    return [-np.outer(unit, point) / (1 + point @ point) for unit in np.eye(2)]


def derivative(function: Callable[[Array], Array], point: Array, axis: int) -> Array:
    """Use a centered difference for a bounded numerical identity check."""
    step = np.eye(2)[axis] * 1e-6
    return (function(point + step) - function(point - step)) / 2e-6


def gauge(point: Array) -> Array:
    """A smooth nonorthogonal, nonconstant GL(2) reference change."""
    x, y = point
    return np.array([[np.exp(0.2 * x), 0.1 * y], [0.3 * x, 1 + 0.1 * x * y]])


def transformed_X(point: Array) -> Array:
    """Re-express the same message collection under h^{-1}."""
    return np.linalg.solve(gauge(point), X(point))


def skew(vector: Array) -> Array:
    """Return the matrix for cross product by the vector."""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_links(point: Array, beta: float) -> list[Array]:
    """The two reciprocal-link representatives in the analytical witness."""
    x, y = point
    rx = Rotation.from_rotvec([x, 0, 0]).as_matrix()
    rz = Rotation.from_rotvec([0, 0, y]).as_matrix()
    rb = Rotation.from_rotvec([beta, 0, 0]).as_matrix()
    return [rx, rz @ rb @ rz.T]


def align_so3(target: Array, source: Array, beta: float) -> Array:
    """Find the local SO(3) squared-distance alignment near identity."""
    targets = so3_links(target, beta)
    sources = so3_links(source, beta)

    def residual(parameters: Array) -> Array:
        rotation = Rotation.from_rotvec(parameters).as_matrix()
        return np.concatenate([
            Rotation.from_matrix(qt @ (rotation @ qs @ rotation.T).T).as_rotvec()
            for qt, qs in zip(targets, sources)
        ])

    fit = least_squares(residual, np.zeros(3), xtol=1e-13, ftol=1e-13,
                        gtol=1e-13, max_nfev=60)
    assert fit.success
    return fit.x


def main() -> None:
    """Run tiny matrix, symbolic, and finite-path checks once."""
    x, y = sp.symbols("x y", real=True)
    sx = sp.Matrix([[1, 0, x], [0, 1, y]])
    sh = sx * sx.T
    sq = sx.T * sh.inv()
    sa = [-sx.diff(axis) * sq for axis in (x, y)]
    sf = sp.simplify(sa[1].diff(x) - sa[0].diff(y) + sa[0] * sa[1] - sa[1] * sa[0])
    expected = sp.Matrix([[-x*y, 1+x*x], [-1-y*y, x*y]]) / (1+x*x+y*y)**2
    assert sp.simplify(sf - expected) == sp.zeros(2)
    projector = sq * sx
    projector_f = (sx.diff(x) * (sp.eye(3)-projector) * sx.diff(y).T
                   - sx.diff(y) * (sp.eye(3)-projector) * sx.diff(x).T) * sh.inv()
    assert sp.simplify(projector_f - sf) == sp.zeros(2)
    for axis, coefficient in zip((x, y), sa):
        assert sp.simplify(sh.diff(axis) + coefficient*sh + sh*coefficient.T) == sp.zeros(2)
    assert sp.simplify(sp.trace(sf)) == 0
    RESULTS["exact_rectangular_curvature_and_metric"] = "symbolic identities equal zero"

    point = np.array([0.4, -0.3])
    other = point + np.array([0.025, 0.01])
    matrix = X(point)
    inverse = right_inverse(matrix)
    record("zero_displacement", norm(cross(point, point)-np.eye(2)), 1e-13)
    record("right_inverse", norm(matrix@inverse-np.eye(2)), 1e-13)
    transformed_cross = transformed_X(other) @ right_inverse(transformed_X(point))
    expected_cross = np.linalg.solve(gauge(other), cross(other, point) @ gauge(point))
    record("nonorthogonal_endpoint_law", norm(transformed_cross-expected_cross), 1e-12)
    for axis in range(2):
        derived = -derivative(transformed_X, point, axis) @ right_inverse(transformed_X(point))
        expected_a = np.linalg.solve(gauge(point), connection(point)[axis] @ gauge(point)
                                    + derivative(gauge, point, axis))
        record(f"affine_connection_axis_{axis}", norm(derived-expected_a), 2e-9)
        residual = derivative(X, point, axis)+connection(point)[axis]@matrix
        record(f"normal_equation_axis_{axis}", norm(residual@matrix.T), 2e-10)
        delta = np.eye(2)[axis]*1e-5
        central = (reciprocal(point+delta, point)-reciprocal(point-delta, point))/2e-5
        record(f"reciprocal_first_jet_axis_{axis}", norm(central+connection(point)[axis]), 2e-9)
    record("reciprocal_reversal", norm(reciprocal(point, other)@reciprocal(other, point)-np.eye(2)), 1e-12)

    def curve(t: float) -> Array:
        return np.array([0.3+0.5*t, -0.2+0.4*np.sin(1.1*t)])

    def velocity(t: float) -> Array:
        return np.array([0.5, 0.44*np.cos(1.1*t)])

    def ode(t: float, flattened: Array) -> Array:
        coefficients = connection(curve(t))
        speed = velocity(t)
        return (-(coefficients[0]*speed[0]+coefficients[1]*speed[1]) @ flattened.reshape(2, 2)).ravel()

    solved = solve_ivp(ode, (0, 1), np.eye(2).ravel(), rtol=2e-12, atol=2e-13)
    assert solved.success
    endpoint = solved.y[:, -1].reshape(2, 2)
    for name, rule in (("cross", cross), ("reciprocal", reciprocal)):
        errors = []
        for count in (8, 16, 32, 64):
            product = np.eye(2)
            for j in range(count):
                product = rule(curve((j+1)/count), curve(j/count)) @ product
            errors.append(norm(product-endpoint))
        assert all(new < 0.65*old for old, new in zip(errors, errors[1:])), errors
        RESULTS[f"{name}_path_refinement"] = {"steps": [8, 16, 32, 64], "errors": errors}

    point3 = np.array([0.9, 1.1])
    beta = 1.2
    links = so3_links(point3, beta)
    operators = [np.eye(3)-link for link in links]
    locked = sum(operator.T@operator for operator in operators)
    assert np.linalg.eigvalsh(locked).min() > 0
    wx, wb = 4*np.sin(point3[0]/2)**2, 4*np.sin(beta/2)**2
    expected_a = [np.zeros(3), np.array([0, 0, -wb/(wx+wb)])]
    for axis in range(2):
        tangents = [derivative(lambda p, j=j: so3_links(p, beta)[j], point3, axis)@link.T
                    for j, link in enumerate(links)]
        vectors = [np.array([v[2, 1], v[0, 2], v[1, 0]]) for v in tangents]
        selected = np.linalg.solve(locked, -sum(c.T@v for c, v in zip(operators, vectors)))
        record(f"so3_normal_axis_{axis}", norm(selected-expected_a[axis]), 2e-9)
        delta = np.eye(2)[axis]*1e-4
        jet = (align_so3(point3+delta, point3, beta)-align_so3(point3-delta, point3, beta))/2e-4
        record(f"so3_finite_alignment_jet_axis_{axis}", norm(jet+expected_a[axis]), 3e-7)
    coefficient = 2*wb*np.sin(point3[0])/(wx+wb)**2
    assert coefficient > 0
    RESULTS["so3_curvature"] = {"coefficient": coefficient, "min_locked_eigenvalue": float(np.linalg.eigvalsh(locked).min())}

    ss = sp.symbols("s", positive=True)
    scalar_X = sp.Matrix([[1, ss]])
    scalar_a = -scalar_X.diff(ss)*scalar_X.T/(scalar_X*scalar_X.T)[0]
    assert sp.simplify(scalar_a[0]+ss/(1+ss**2)) == 0
    RESULTS["backtrack_scalar_connection"] = "symbolic identity equal zero"
    result = {"status": "passed", "check_groups": len(RESULTS), "scope": "Small illustrative formulas only; no GPU or training", "checks": RESULTS}
    Path(__file__).with_name("check-results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "check_groups": result["check_groups"]}))


if __name__ == "__main__":
    main()
