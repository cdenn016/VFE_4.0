"""Ch.10 prop:boundary-observable: eq:boundary-response-derivatives and eq:boundary-observability-det."""
import sympy as sp
e, m0, n0, a0, b0, J, c1, c2 = sp.symbols('epsilon m0 n0 a0 b0 J c1 c2', real=True)
def vel(xi, xj, yi):
    return -(1 - xi**2) * (2 * sp.atanh(xi) - sp.atanh(xj) - J * yi) - (xi - xj)
m, n, a, b = e * m0, e * n0, e * a0, e * b0
x1, x2, y1, y2 = m + a, m - a, n + b, n - b
Fx = (vel(x1, x2, y1) + vel(x2, x1, y2)) / 2
Fy = (vel(y1, y2, x1) + vel(y2, y1, x2)) / 2
claimFx = -(1 - m**2) * (sp.atanh(m) - J * n) + (6 * m - J * n) * a**2 - 2 * J * m * a * b
claimFy = -(1 - n**2) * (sp.atanh(n) - J * m) + (6 * n - J * m) * b**2 - 2 * J * n * a * b
print('Fx residual through order 4:', sp.simplify(sp.series(Fx - claimFx, e, 0, 5).removeO()))
print('Fy residual through order 4:', sp.simplify(sp.series(Fy - claimFy, e, 0, 5).removeO()))
# determinant of D(m,n,Fx,Fy) w.r.t. (m,n,a,b) using the truncated fields
M_, N_, A_, B_ = sp.symbols('M N A B', real=True)
tFx = -(1 - M_**2) * (sp.atanh(M_) - J * N_) + (6 * M_ - J * N_) * A_**2 - 2 * J * M_ * A_ * B_
tFy = -(1 - N_**2) * (sp.atanh(N_) - J * M_) + (6 * N_ - J * M_) * B_**2 - 2 * J * N_ * A_ * B_
H = sp.Matrix([M_, N_, tFx, tFy])
D = H.jacobian([M_, N_, A_, B_]).subs({M_: e * c1, N_: e * c1, A_: e * c2, B_: e * c2})
det = sp.factor(sp.simplify(D.det()))
print('det (truncated fields):', det)
print('claimed - computed:', sp.simplify(det - 4 * e**4 * c1**2 * c2**2 * (6 - 3 * J) * (6 - J)))
