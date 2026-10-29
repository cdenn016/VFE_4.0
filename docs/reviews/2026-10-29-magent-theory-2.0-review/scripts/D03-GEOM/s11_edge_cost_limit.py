"""D03-GEOM s11: second-order limit of the finite alignment cost (chapter 15 edge cost)
to the collective minimal tensor (geo:collective / geo:context-energy), flat example.
Two planar unit-covariance normal location laws (Fisher metric Euclidean on R^4),
SO(2) acting diagonally.  For q = p(c + eps X), p = p(c):
  min_theta |q - R_theta p|^2 = |q|^2 + |p|^2 - 2 sqrt(<q,p>^2 + <q,Jp>^2),
and (1/eps^2) min_theta (1/2)|q - R_theta p|^2 should tend to (1/2) h^min(X,X).
"""
import numpy as np
def p_of(c):
    return np.array([1 + 0.3*c[0], 0.2*c[1] - 0.1*c[0]**2, -0.5 + 0.4*c[1], 0.7 + 0.2*c[0]*c[1]])
Jrot = np.kron(np.eye(2), np.array([[0, -1], [1, 0]]))
c0 = np.array([0.3, -0.2]); X = np.array([0.6, 0.8])
p = p_of(c0); h = 1e-6
dp = (p_of(c0 + h*X) - p_of(c0 - h*X))/(2*h)
Jp = Jrot @ p
hmin = dp @ dp - (dp @ Jp)**2/(Jp @ Jp)
print("h^min(X,X) =", f"{hmin:.12f}")
for eps in [1e-1, 1e-2, 1e-3]:
    q = p_of(c0 + eps*X)
    m = q @ q + p @ p - 2*np.sqrt((q @ p)**2 + (q @ Jp)**2)
    val = 0.5*m/eps**2
    print(f"eps={eps:.0e}: edge cost/eps^2 = {val:.12f}; relative error vs h/2 = {abs(val/(hmin/2) - 1):.2e}")
