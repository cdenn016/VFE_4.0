r"""D16-OPEN s9: can full and null flows order the nearly simultaneous first-stage events differently?

inh:null-flow compares partition SEQUENCES P (event times removed).  In the reference of
open:routing the three first-stage pair events are nearly simultaneous: every pair starts with
half-gap 2.  If the full-minus-null event-time offset differs between the core pair and the far
pair, then perturbing one pair's initial gap by a relative amount between the two flip points
reverses the order of two disjoint pair events in one flow but not in the other; on that open
wedge P != P0 although the formed blocks agree.

We compute, at the open:routing reference (A = 4096, R from the proof's recipe), the crossing time
of beta_01 = 1/10 (core pair) and beta_45 = 1/10 (far pair) under
   exact full field / exact null field at h = 1e-9 and h = 2e-11, and the limit fields,
with RK4 dt = 1e-3 and cubic interpolation of the crossing.  Then we scan a relative perturbation
eta of the far pair's initial gap and report where the event order differs between the two flows.
For an isolated pair the event time is (to e^{-8} accuracy) the beta crossing; contrast margins
are large there (exterior masses < 1e-10).  Numerical evidence only.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

A = 4096.0
L = np.log(A / 4) + 1.426696
R = (12 + A + 6) * np.exp(L)


def breg_over_h2(D, y, h):
    s = 1 - y * y
    p2 = 1 / s; p3 = 2 * y / s ** 2; p4 = (2 + 6 * y * y) / s ** 3; p5 = (24 * y + 24 * y ** 3) / s ** 4
    e = h * D
    return D * D * (p2 / 2 + p3 * e / 6 + p4 * e * e / 24 + p5 * e ** 3 / 120)


def rows_exact(Dm, x, h):
    Dv = breg_over_h2(Dm, x[None, :], h)
    Dv = Dv - Dv.min(1, keepdims=True)
    n = np.exp(-Dv)
    return n / n.sum(1, keepdims=True)


def rows_limit(u):
    d = u[:, None] - u[None, :]
    n = np.exp(-d ** 2 / 2)
    return n / n.sum(1, keepdims=True)


def full_exact(u, h):
    x = h * u
    Dm = u[:, None] - u[None, :]
    B = rows_exact(Dm, x, h)
    at = np.arctanh(h * Dm / (1 - x[:, None] * x[None, :]))
    return (1 - x * x) * (-np.arctanh(x) - (B * at).sum(1)) / h + (B * (u[:, None] - u[None, :])).sum(0)


def full_limit(u):
    W = rows_limit(u)
    X = u[None, :] - u[:, None]
    return -u + ((W + W.T) * X).sum(1)


def crossings(u0, h, mode, tmax=1.0, dt=1e-3):
    ts = np.arange(0, tmax + dt / 2, dt)
    b01, b45 = [], []
    if mode == "full":
        u = u0.copy()
        F = (lambda v: full_exact(v, h)) if h > 0 else full_limit
        for k, t in enumerate(ts):
            if k > 0:
                k1 = F(u); k2 = F(u + dt / 2 * k1); k3 = F(u + dt / 2 * k2); k4 = F(u + dt * k3)
                u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            B = rows_exact(u[:, None] - u[None, :], h * u, h) if h > 0 else rows_limit(u)
            b01.append(B[0, 1]); b45.append(B[4, 5])
    else:
        th0 = np.arctanh(h * u0) if h > 0 else None
        for t in ts:
            if h > 0:
                al = np.exp(-t) * th0; x = np.tanh(al)
                dal = np.exp(-t) * np.arctanh(h * (u0[:, None] - u0[None, :]) / (1 - h * h * u0[:, None] * u0[None, :]))
                Dm = np.sinh(dal) / (np.cosh(al)[:, None] * np.cosh(al)[None, :]) / h
                B = rows_exact(Dm, x, h)
            else:
                B = rows_limit(np.exp(-t) * u0)
            b01.append(B[0, 1]); b45.append(B[4, 5])
    out = []
    for arr in (b01, b45):
        cs = CubicSpline(ts, np.array(arr) - 0.1)
        k = np.argmax(np.array(arr) > 0.1)
        out.append(brentq(cs, ts[k - 1], ts[k]))
    return out


res = ["A=%g R=%.6e" % (A, R)]
for h in (0.0, 2e-11, 1e-9):
    u0 = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
    cF = crossings(u0, h, "full"); cN = crossings(u0, h, "null")
    dcore, dfar = cF[0] - cN[0], cF[1] - cN[1]
    res.append("h=%g: full t(core)=%.8f t(far)=%.8f | null t(core)=%.8f t(far)=%.8f | offset core %.3e far %.3e, difference %.3e"
               % (h, cF[0], cF[1], cN[0], cN[1], dcore, dfar, dfar - dcore))
    if h == 1e-9:
        # scan a relative perturbation of the far pair's initial gap
        flips = []
        for eta in np.concatenate([np.linspace(-2e-3, 2e-3, 11), np.linspace(-2.80e-4, -2.40e-4, 17)]):
            up = u0.copy(); up[4] = R - 2 * (1 + eta); up[5] = R + 2 * (1 + eta)
            f = crossings(up, h, "full"); n = crossings(up, h, "null")
            flips.append((eta, np.sign(f[1] - f[0]), np.sign(n[1] - n[0])))
        diff = [e for e, a, b in flips if a != b]
        res.append("   far-gap scan (11 values in [-2e-3,2e-3] and 17 in [-2.8e-4,-2.4e-4]): order(full) != order(null) at eta = %s"
                   % np.round(diff, 7))
        res.append("   (predicted wedge: eta between -(tF_far - tF_core) = %.4e and -(tN_far - tN_core) = %.4e)"
                   % (-(cF[1] - cF[0]), -(cN[1] - cN[0])))
print("\n".join(res))
