r"""D20-CTXC independent numerical check of the coupled continuum Fisher flow.

Geometry (not used in the manuscript's own checks): two planar unit-covariance Gaussian
location laws z_1, z_2 in R^2 = C, with one SO(2) rotation acting on both means
(P = (z_1, z_2) in C^2, J = multiplication by i).  On P != 0 the action is free and
proper; the quotient is the cone over the Bloch sphere S^2(1/2), which is curved, so
the quotient Christoffel symbols and the horizontal coefficient C do not vanish.

Adapted chart  Psi(rho, theta, psi, chi) = rho (cos(theta/2) e^{i chi}, sin(theta/2) e^{i(chi+psi)}),
shape u = (rho, theta, psi), fiber r = chi, with (symbolic_checks.py, part d)
  G = d rho^2 + (rho^2/4)(d theta^2 + sin^2 theta d psi^2) + rho^2 (d chi + sin^2(theta/2) d psi)^2.

Potential: fixed x-dependent references N(z_ref,i(x), I) (not rotation invariant, so the
potential is nonbasic, V_r != 0) plus an invariant reciprocal peer term:
  V(x, P) = |P - P_ref(x)|^2 / 2 + beta |z_1 - z_2|^2.

Compared on periodic grids (d = 1 and d = 2):
  (A) chart system ctx:c:eq:shape, ctx:c:eq:fiber mapped to Cartesian velocity, versus
  (B) the Cartesian law flow P_t = -grad V + lambda sum_mu D_mu D_mu P with the
      eliminated connection alpha_mu = -<d_mu P, JP>/|P|^2 (ctx:c:eq:intrinsic).
Also: reduced energy identity |P_H d_mu P|^2 = h(d_mu u, d_mu u); dissipation identity
(ctx:c:eq:dissipation, both forms) by a directional finite difference of the reduced
energy; curvature of the eliminated connection versus the pulled-back monopole form;
passive gauge covariance with an x-dependent rotation in d = 2.

These are formula checks at sampled smooth fields, not a proof of well-posedness.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

LAM = 0.7
BETA = 0.6


def spectral_derivative(f: np.ndarray, axis: int, order: int = 1) -> np.ndarray:
    """Periodic spectral derivative along one grid axis of a 2 pi periodic grid."""
    n = f.shape[axis]
    freq = np.fft.fftfreq(n, d=1.0 / n)
    shape = [1] * f.ndim
    shape[axis] = n
    mult = ((1j * freq) ** order).reshape(shape)
    return np.real(np.fft.ifft(mult * np.fft.fft(f, axis=axis), axis=axis))


def jmul(p: np.ndarray) -> np.ndarray:
    """J acting on P = (x1, y1, x2, y2) stored in the last axis: (-y1, x1, -y2, x2)."""
    return np.stack([-p[..., 1], p[..., 0], -p[..., 3], p[..., 2]], axis=-1)


def psi_map(rho, th, ps, ch):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.stack([rho * c * np.cos(ch), rho * c * np.sin(ch),
                     rho * s * np.cos(ch + ps), rho * s * np.sin(ch + ps)], axis=-1)


def psi_jacobian(rho, th, ps, ch):
    """Return d Psi / d(rho, theta, psi, chi), each of shape grid + (4,)."""
    c, s = np.cos(th / 2), np.sin(th / 2)
    d_rho = np.stack([c * np.cos(ch), c * np.sin(ch), s * np.cos(ch + ps), s * np.sin(ch + ps)], axis=-1)
    d_th = np.stack([-0.5 * rho * s * np.cos(ch), -0.5 * rho * s * np.sin(ch),
                     0.5 * rho * c * np.cos(ch + ps), 0.5 * rho * c * np.sin(ch + ps)], axis=-1)
    d_ps = np.stack([0 * rho, 0 * rho, -rho * s * np.sin(ch + ps), rho * s * np.cos(ch + ps)], axis=-1)
    d_ch = np.stack([-rho * c * np.sin(ch), rho * c * np.cos(ch),
                     -rho * s * np.sin(ch + ps), rho * s * np.cos(ch + ps)], axis=-1)
    return d_rho, d_th, d_ps, d_ch


def grad_v(p: np.ndarray, pref: np.ndarray) -> np.ndarray:
    """Cartesian gradient of |P - P_ref|^2/2 + beta |z1 - z2|^2."""
    z1, z2 = p[..., 0:2], p[..., 2:4]
    peer = 2 * BETA * np.concatenate([z1 - z2, z2 - z1], axis=-1)
    return (p - pref) + peer


def v_density(p: np.ndarray, pref: np.ndarray) -> np.ndarray:
    z1, z2 = p[..., 0:2], p[..., 2:4]
    return 0.5 * np.sum((p - pref) ** 2, axis=-1) + BETA * np.sum((z1 - z2) ** 2, axis=-1)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def cartesian_flow(p: np.ndarray, pref: np.ndarray, d: int) -> tuple[np.ndarray, list[np.ndarray]]:
    r"""P_t = -grad V + lambda sum_mu D_mu D_mu P with alpha_mu = -<d_mu P, JP>/|P|^2."""
    jp = jmul(p)
    norm2 = dot(p, p)
    lap = np.zeros_like(p)
    alphas = []
    for mu in range(d):
        p_mu = spectral_derivative(p, axis=mu)
        alpha = -dot(p_mu, jp) / norm2
        alphas.append(alpha)
        dp = p_mu + alpha[..., None] * jp
        ddp = spectral_derivative(dp, axis=mu) + alpha[..., None] * jmul(dp)
        lap += ddp
    return -grad_v(p, pref) + LAM * lap, alphas


def reduced_energy(p: np.ndarray, pref: np.ndarray, d: int) -> float:
    jp = jmul(p)
    norm2 = dot(p, p)
    dens = v_density(p, pref)
    for mu in range(d):
        p_mu = spectral_derivative(p, axis=mu)
        dens = dens + 0.5 * LAM * (dot(p_mu, p_mu) - dot(p_mu, jp) ** 2 / norm2)
    return float(np.mean(dens))


def christoffel_tension(u: list[np.ndarray], d: int) -> list[np.ndarray]:
    r"""tau^a = sum_mu [d_mu d_mu u^a + Gamma^a_bc d_mu u^b d_mu u^c] for h = diag(1, rho^2/4, rho^2 sin^2/4)."""
    rho, th, ps = u
    tau = [np.zeros_like(rho) for _ in range(3)]
    for mu in range(d):
        dr, dt, dp = (spectral_derivative(f, axis=mu) for f in (rho, th, ps))
        ddr, ddt, ddp = (spectral_derivative(f, axis=mu, order=2) for f in (rho, th, ps))
        tau[0] += ddr - rho / 4 * dt**2 - rho * np.sin(th) ** 2 / 4 * dp**2
        tau[1] += ddt + 2 / rho * dr * dt - np.sin(th) * np.cos(th) * dp**2
        tau[2] += ddp + 2 / rho * dr * dp + 2 * (np.cos(th) / np.sin(th)) * dt * dp
    return tau


def chart_flow(u: list[np.ndarray], ch: np.ndarray, pref: np.ndarray, d: int):
    r"""u_t = lam tau_h - h^{-1}(V_u - C^T V_r);  r_t = -C u_t - k^{-1} V_r  (ctx:c:eq:shape, fiber)."""
    rho, th, ps = u
    p = psi_map(rho, th, ps, ch)
    jac = psi_jacobian(rho, th, ps, ch)
    g = grad_v(p, pref)
    v_u = [dot(g, jac[0]), dot(g, jac[1]), dot(g, jac[2])]
    v_r = dot(g, jac[3])
    cpsi = np.sin(th / 2) ** 2
    hinv = [np.ones_like(rho), 4 / rho**2, 4 / (rho**2 * np.sin(th) ** 2)]
    hdiag = [np.ones_like(rho), rho**2 / 4, rho**2 * np.sin(th) ** 2 / 4]
    k = rho**2
    tau = christoffel_tension(u, d)
    horizontal = [v_u[0], v_u[1], v_u[2] - cpsi * v_r]  # V_u - C^T V_r = H_a V
    u_t = [LAM * tau[a] - hinv[a] * horizontal[a] for a in range(3)]
    r_t = -cpsi * u_t[2] - v_r / k
    p_t = sum(jac[a] * u_t[a][..., None] for a in range(3)) + jac[3] * r_t[..., None]
    kinetic_first = sum(hdiag[a] * u_t[a] ** 2 for a in range(3)) + k * (r_t + cpsi * u_t[2]) ** 2
    kinetic_second = sum(hdiag[a] * u_t[a] ** 2 for a in range(3)) + v_r**2 / k
    return p, p_t, float(np.mean(kinetic_first)), float(np.mean(kinetic_second)), hdiag


def make_fields(d: int, n: int):
    axes = np.meshgrid(*[2 * np.pi * np.arange(n) / n] * d, indexing="ij")
    x = axes[0]
    y = axes[1] if d == 2 else 0 * x
    rho = 1.3 + 0.2 * np.cos(x) + 0.1 * np.sin(2 * x) + 0.15 * np.sin(y)
    th = 1.1 + 0.3 * np.sin(x) + 0.1 * np.cos(3 * x) + 0.25 * np.cos(y + 0.3)
    ps = 0.4 + 0.5 * np.cos(x) - 0.2 * np.sin(2 * x) + 0.35 * np.sin(y - x)
    ch = 0.2 + 0.3 * np.sin(x) + 0.15 * np.cos(2 * x) - 0.2 * np.cos(2 * y)
    pref = np.stack([0.8 + 0.2 * np.cos(x), -0.3 + 0.1 * np.sin(x) + 0.1 * np.cos(y),
                     0.1 * np.sin(x), 0.5 + 0.2 * np.cos(2 * x) - 0.1 * np.sin(y)], axis=-1)
    return [rho, th, ps], ch, pref, axes


def run(d: int, n: int) -> dict[str, float]:
    u, ch, pref, axes = make_fields(d, n)
    p, pt_chart, kin1, kin2, hdiag = chart_flow(u, ch, pref, d)
    pt_cart, alphas = cartesian_flow(p, pref, d)
    out: dict[str, float] = {}
    out["max|P_t(chart) - P_t(Cartesian)|"] = float(np.max(np.abs(pt_chart - pt_cart)))
    out["max|P_t| (scale)"] = float(np.max(np.abs(pt_cart)))
    # Reduced energy identity: |P_H d_mu P|^2 = h(d_mu u, d_mu u).
    jp = jmul(p)
    res = 0.0
    for mu in range(d):
        p_mu = spectral_derivative(p, axis=mu)
        lhs = dot(p_mu, p_mu) - dot(p_mu, jp) ** 2 / dot(p, p)
        rhs = sum(hdiag[a] * spectral_derivative(u[a], axis=mu) ** 2 for a in range(3))
        res = max(res, float(np.max(np.abs(lhs - rhs))))
    out["max|reduced density - h(du,du)|"] = res
    # Dissipation: d/de F*[P + e P_t] at 0 versus -||P_t||^2 and the two chart forms.
    eps = 1e-5
    deriv = (reduced_energy(p + eps * pt_cart, pref, d) - reduced_energy(p - eps * pt_cart, pref, d)) / (2 * eps)
    cart_kin = float(np.mean(dot(pt_cart, pt_cart)))
    out["|dF*(P_t) + ||P_t||^2|"] = abs(deriv + cart_kin)
    out["|chart kinetic (first form) - Cartesian ||P_t||^2|"] = abs(kin1 - cart_kin)
    out["|chart kinetic first form - second form|"] = abs(kin1 - kin2)
    out["||P_t||^2"] = cart_kin
    # Eliminated connection in chart form: alpha_mu = -(d_mu chi + sin^2(theta/2) d_mu psi).
    rho, th, ps = u
    amax = 0.0
    for mu in range(d):
        pred = -(spectral_derivative(ch, axis=mu) + np.sin(th / 2) ** 2 * spectral_derivative(ps, axis=mu))
        amax = max(amax, float(np.max(np.abs(alphas[mu] - pred))))
    out["max|alpha_mu - (-p^*theta)_mu|"] = amax
    if d == 1:
        # Base-cycle holonomy exp(J * oint alpha dx): gauge invariant, equals minus the geometric
        # phase oint sin^2(theta/2) d psi of the closed shape loop, generally nonzero.
        x = axes[0]
        hol = 2 * np.pi * float(np.mean(alphas[0]))
        berry = -2 * np.pi * float(np.mean(np.sin(th / 2) ** 2 * spectral_derivative(ps, axis=0)))
        out["oint alpha dx"] = hol
        out["|oint alpha + oint sin^2(theta/2) dpsi|"] = abs(hol - berry)
        gam = 0.4 * np.sin(x) + 0.3 * np.cos(2 * x)

        def rot1(q: np.ndarray) -> np.ndarray:
            return np.cos(gam)[..., None] * q + np.sin(gam)[..., None] * jmul(q)

        _, alphas_rot = cartesian_flow(rot1(p), rot1(pref), d)
        out["|oint alpha' - oint alpha| under periodic gauge"] = abs(2 * np.pi * float(np.mean(alphas_rot[0])) - hol)
        # Time derivative of the holonomy along the chart flow versus the swept curvature flux
        # d/dt oint alpha = -(1/2) oint sin(theta) (theta_t psi_x - theta_x psi_t) dx.
        rho, _, _ = u
        _, _, _, _, _ = chart_flow(u, ch, pref, d)
        jac = psi_jacobian(rho, th, ps, ch)
        g = grad_v(p, pref)
        v_r = dot(g, jac[3])
        tau = christoffel_tension(u, d)
        hinv = [np.ones_like(rho), 4 / rho**2, 4 / (rho**2 * np.sin(th) ** 2)]
        horizontal = [dot(g, jac[0]), dot(g, jac[1]), dot(g, jac[2]) - np.sin(th / 2) ** 2 * v_r]
        u_t = [LAM * tau[a] - hinv[a] * horizontal[a] for a in range(3)]
        r_t = -np.sin(th / 2) ** 2 * u_t[2] - v_r / rho**2
        epsilon = 1e-6

        def holonomy_at(e: float) -> float:
            th_e, ps_e, ch_e = th + e * u_t[1], ps + e * u_t[2], ch + e * r_t
            alpha_e = -(spectral_derivative(ch_e, axis=0) + np.sin(th_e / 2) ** 2 * spectral_derivative(ps_e, axis=0))
            return 2 * np.pi * float(np.mean(alpha_e))

        dhol = (holonomy_at(epsilon) - holonomy_at(-epsilon)) / (2 * epsilon)
        flux = -0.5 * 2 * np.pi * float(np.mean(np.sin(th) * (u_t[1] * spectral_derivative(ps, axis=0)
                                                            - spectral_derivative(th, axis=0) * u_t[2])))
        out["d/dt oint alpha (finite difference)"] = dhol
        out["|d/dt oint alpha - swept curvature flux|"] = abs(dhol - flux)
    if d == 2:
        f12 = spectral_derivative(alphas[1], axis=0) - spectral_derivative(alphas[0], axis=1)
        th1, th2 = spectral_derivative(th, axis=0), spectral_derivative(th, axis=1)
        ps1, ps2 = spectral_derivative(ps, axis=0), spectral_derivative(ps, axis=1)
        pull = -0.5 * np.sin(th) * (th1 * ps2 - th2 * ps1)
        out["max|F_12 - (pulled-back monopole form)|"] = float(np.max(np.abs(f12 - pull)))
        out["max|F_12| (curvature is nonzero)"] = float(np.max(np.abs(f12)))
        out["|mean F_12| (zero total flux on the trivial bundle)"] = abs(float(np.mean(f12)))
        # Passive gauge covariance with an x-dependent rotation g = R_{gamma(x,y)} on both columns.
        x, y = axes
        gam = 0.4 * np.sin(x) + 0.3 * np.cos(x + y)

        def rot(q: np.ndarray) -> np.ndarray:
            return np.cos(gam)[..., None] * q + np.sin(gam)[..., None] * jmul(q)

        pt_rot, _ = cartesian_flow(rot(p), rot(pref), d)
        out["max|flow(gP; gP_ref) - g flow(P; P_ref)|"] = float(np.max(np.abs(pt_rot - rot(pt_cart))))
    return out


def main() -> None:
    start = time.perf_counter()
    record = {"script": "hopf_flow_check.py", "lambda": LAM, "beta": BETA, "numpy": np.__version__}
    record["d=1, N=256"] = run(1, 256)
    record["d=2, N=64"] = run(2, 64)
    record["runtime_s"] = round(time.perf_counter() - start, 3)
    Path(__file__).with_name("hopf_flow_check.out.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
