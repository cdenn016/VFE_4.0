r"""D11-FUS: the companion's per-agent inclusion n_I mu_I' in -partial_{mu_I} E (c07_fusion.tex:64)
differs from the joint subdifferential inclusion W mu' in -partial F (06_fusion.tex:207).

Two unit agents, coincident at w0, trap U, pair term lam(|dq|+|ds|).
Per-agent partial subdifferentials allow eps_12 = eps_21 = theta in [-1,1]^2 (not antisymmetric),
so every curve w' = -A w - lam*theta, theta in [-1,1]^2, with both agents at w(t) is a solution.
The joint subdifferential forces eps_21 = -eps_12, hence theta = 0 (unique solution).
The Filippov set of the a.e. gradient of lam|x1-x2| at x1=x2 is the antidiagonal segment
{lam(s,-s): |s|<=1}, i.e. the joint subdifferential, not the square [-lam,lam]^2.
"""
import numpy as np
from scipy.linalg import expm

a, c, lam = 1.0, 0.5, 0.3
A = np.array([[a, -c], [-c, a]])
w0 = np.array([1.0, -0.5])

def curve(theta, t):
    # w(t) = e^{-At} w0 - lam * A^{-1}(I - e^{-At}) theta
    E = expm(-A*t)
    return E @ w0 - lam*np.linalg.solve(A, (np.eye(2) - E) @ theta)

for theta in (np.zeros(2), np.array([1.0, 0.0]), np.array([0.5, -1.0])):
    # verify the per-agent inclusion along the curve by finite differences
    ts = np.linspace(0.0, 2.0, 201)
    res = 0.0
    for t in ts[1:-1]:
        dw = (curve(theta, t + 1e-6) - curve(theta, t - 1e-6))/2e-6
        sel = -(dw + A @ curve(theta, t))/lam        # required selection for agent 1 (and agent 2)
        res = max(res, np.abs(sel - theta).max())
    print(f"theta={theta}: w(2)={curve(theta, 2.0)}, selection residual {res:.1e}; "
          f"selections in [-1,1]^2: {np.all(np.abs(theta) <= 1)}; antisymmetric (eps21=-eps12): {np.allclose(theta, -theta)}")

# Filippov set: limits of the a.e. gradient of lam|x1-x2| (one sector) at x1=x2
rng = np.random.default_rng(0)
pts = rng.normal(scale=1e-8, size=(2000, 2))
grads = np.array([lam*np.sign(p[0] - p[1])*np.array([1.0, -1.0]) for p in pts])
print("distinct limiting gradients:", np.unique(np.round(grads, 12), axis=0).tolist(),
      "-> convex hull is the segment {lam(s,-s)}, the joint subdifferential")
