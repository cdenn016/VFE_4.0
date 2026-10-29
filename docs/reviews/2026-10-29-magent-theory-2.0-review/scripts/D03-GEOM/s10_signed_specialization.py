"""D03-GEOM s10: geo:signed-identity specialized to collective minimizers on both levels.
With Psi induced by an equivariant context-independent R (here the identity on means of
SO(3)-rotated normal location laws, coarse metric from isotropic noise), fine connection
A = A^F and coarse connection Abar^F, the horizontal defect is a = Jbar(Abar^F - A^F),
the coarse derivative is dR r + a = rbar, and X_Psi + Q_Psi = -kbar(Abar - A, Abar - A),
so geo:signed-identity reduces to geo:minimal-contraction-identity.
"""
import numpy as np
rng = np.random.default_rng(99)
def hat(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def Jm(mus): return np.vstack([np.column_stack([hat(e) @ m for e in np.eye(3)]) for m in mus])
worst = [0.0, 0.0, 0.0]
for _ in range(200):
    mus = rng.normal(size=(3, 3)); w = rng.random(3) + 0.2; tau2 = rng.random(3)*2
    G = np.diag(np.repeat(w, 3)); Gb = np.diag(np.repeat(w/(1 + tau2), 3)); D = G - Gb
    J = Jm(mus); v = rng.normal(size=9)
    A = -np.linalg.solve(J.T @ G @ J, J.T @ G @ v); r = v + J @ A
    Ab = -np.linalg.solve(J.T @ Gb @ J, J.T @ Gb @ v); rb = v + J @ Ab
    a = J @ (Ab - A)
    X = 2*(r @ Gb @ a); Q = a @ Gb @ a; kb = (Ab - A) @ (J.T @ Gb @ J) @ (Ab - A)
    worst[0] = max(worst[0], np.max(np.abs(r + a - rb)))
    worst[1] = max(worst[1], abs(X + Q + kb))
    lhs = r @ G @ r - rb @ Gb @ rb
    worst[2] = max(worst[2], abs(lhs - (r @ D @ r - X - Q)))
print("max |dR r + a - rbar| =", f"{worst[0]:.1e}")
print("max |X + Q + kbar(Abar - A)| =", f"{worst[1]:.1e}")
print("max |(h - hbar) - (u*Delta - X - Q)| =", f"{worst[2]:.1e}")
