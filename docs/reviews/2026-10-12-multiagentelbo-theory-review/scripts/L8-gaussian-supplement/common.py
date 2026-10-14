"""Shared helpers for the L8 (Gaussian supplement) checks. float64 throughout."""
import numpy as np
from numpy.linalg import inv, det, slogdet

rng = np.random.default_rng(20261012)
out = []


def rep(name, val, base=1.0):
    out.append(f"{name}: residual {val:.3e} (baseline {base:.3e})")


def note(s):
    out.append(s)


def spd(n, cond=None):
    A = rng.standard_normal((n, n))
    if cond is None:
        return A @ A.T + n * np.eye(n)
    Q, _ = np.linalg.qr(A)
    ev = np.logspace(0, np.log10(cond), n)
    return Q @ np.diag(ev) @ Q.T


def gl_plus(n, cond=None):
    A = rng.standard_normal((n, n)) + 0.3 * np.eye(n)
    if cond is not None:
        U, s, Vt = np.linalg.svd(A)
        A = U @ np.diag(np.logspace(0, np.log10(cond), n)) @ Vt
    if det(A) < 0:
        A[:, 0] *= -1
    return A


def kl(mu0, S0, mu1, S1):
    K = len(mu0)
    S1i = inv(S1)
    d = mu0 - mu1
    return 0.5 * (np.trace(S1i @ S0) + d @ S1i @ d - K + slogdet(S1)[1] - slogdet(S0)[1])
