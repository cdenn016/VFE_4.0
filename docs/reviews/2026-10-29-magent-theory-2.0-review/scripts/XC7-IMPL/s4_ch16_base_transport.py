r"""XC7-IMPL s4: conformance checks for chapter 16 (base transport from contextual data).

(a) ibt:eq:witnessA / ibt:eq:witnessF and ibt:eq:metric-compatibility, symbolic.
(b) Laplace realization: curvature equals the pullback under (x,y) -> (x,1+xy).
(c) ibt:path: ordered products of the cross-maps C(c',c) = X(c')Q_X(c) along a path
    converge to the path transport of the derived first jet (rate check).
(d) ibt:eq:SO3-curvature: alignment connection of two SO(3) links from the normal
    equation ibt:eq:SO3-normal, its curvature by finite differences, vs closed form.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm

x, y = sp.symbols('x y', real=True)
def connection(Xm):
    Hm = Xm * Xm.T; Q = Xm.T * Hm.inv()
    ax, ay = -Xm.diff(x) * Q, -Xm.diff(y) * Q
    F = sp.simplify(ay.diff(x) - ax.diff(y) + ax * ay - ay * ax)
    comp = [sp.simplify(Xm.diff(z) * Xm.T + Xm * Xm.diff(z).T + az * Hm + Hm * az.T) for z, az in ((x, ax), (y, ay))]
    return ax, ay, F, comp, Hm
Xw = sp.Matrix([[1, 0, x], [0, 1, y]])
ax, ay, F, comp, Hm = connection(Xw)
s = 1 + x ** 2 + y ** 2
Fclaim = sp.Matrix([[-x * y, 1 + x ** 2], [-(1 + y ** 2), x * y]]) / s ** 2
v = sp.Matrix([x, y])
print("(a) a_x + (dv/dx)v^T/s =", sp.simplify(ax + sp.Matrix([1, 0]) * v.T / s), "; a_y + ... =", sp.simplify(ay + sp.Matrix([0, 1]) * v.T / s))
print("    F - claimed =", sp.simplify(F - Fclaim), "; metric compatibility residuals:", [sp.simplify(c) for c in comp],
      "; Tr F =", sp.simplify(F.trace()), "; Tr a_x + d/dx log det H /2 =", sp.simplify(ax.trace() + sp.diff(sp.log(Hm.det()), x) / 2))
# (b) Laplace realization: replace v by (x, 1+xy)
XL = sp.Matrix([[1, 0, x], [0, 1, 1 + x * y]])
_, _, FL, _, _ = connection(XL)
pull = Fclaim.subs({y: 1 + x * y}, simultaneous=True) * x      # Jacobian determinant of (x,y)->(x,1+xy) is x
print("(b) Laplace realization: F - pullback =", sp.simplify(FL - pull))
# (c) path transport vs ordered products of C
Xf = sp.lambdify((x, y), Xw, 'numpy')
axf, ayf = sp.lambdify((x, y), ax, 'numpy'), sp.lambdify((x, y), ay, 'numpy')
def Xn(c): return np.array(Xf(*c), float)
def Qn(c): Xc = Xn(c); return Xc.T @ np.linalg.inv(Xc @ Xc.T)
gam = lambda t: np.array([0.3 + np.cos(2 * np.pi * t), 0.2 + 0.7 * np.sin(2 * np.pi * t)])
dgam = lambda t: np.array([-2 * np.pi * np.sin(2 * np.pi * t), 1.4 * np.pi * np.cos(2 * np.pi * t)])
def ode(t, Tv):
    c, dc = gam(t), dgam(t)
    A = np.array(axf(*c), float) * dc[0] + np.array(ayf(*c), float) * dc[1]
    return (-A @ Tv.reshape(2, 2)).reshape(-1)
Tex = solve_ivp(ode, (0, 1), np.eye(2).reshape(-1), rtol=1e-12, atol=1e-14).y[:, -1].reshape(2, 2)
errs = []
for nstep in (50, 100, 200, 400):
    ts = np.linspace(0, 1, nstep + 1); Tp = np.eye(2)
    for k in range(nstep):
        Tp = Xn(gam(ts[k + 1])) @ Qn(gam(ts[k])) @ Tp
    errs.append(np.abs(Tp - Tex).max())
print("(c) loop holonomy (exact ODE):", np.round(Tex, 6).tolist(), " product errors n=50..400:", ["%.2e" % e for e in errs],
      " ratios:", ["%.2f" % (errs[i] / errs[i + 1]) for i in range(3)])
# (d) SO(3) alignment
def hat(vv): return np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
def vee(M): return np.array([M[2, 1], M[0, 2], M[1, 0]])
def Rx(t): return expm(hat([t, 0, 0]))
def Rz(t): return expm(hat([0, 0, t]))
beta = 1.1
def links(c): return [Rx(c[0]), Rz(c[1]) @ Rx(beta) @ Rz(-c[1])]
def a_of(c, h=1e-6):
    Qs = links(c); Cs = [np.eye(3) - Q for Q in Qs]           # Ad_Q acts on vectors as Q
    K = sum(Cm.T @ Cm for Cm in Cs)
    out = []
    for d in range(2):
        e = np.zeros(2); e[d] = h
        Vs = [vee((Qp - Qm) / (2 * h) @ Q.T) for Qp, Qm, Q in zip(links(c + e), links(c - e), Qs)]
        out.append(-np.linalg.solve(K, sum(Cm.T @ Vv for Cm, Vv in zip(Cs, Vs))))
    return np.array(out)          # rows: a_x, a_y as vectors in so(3)
for c in (np.array([0.7, 0.4]), np.array([2.1, 1.3])):
    a = a_of(c); h = 1e-4
    dax_dy = (a_of(c + [0, h])[0] - a_of(c - [0, h])[0]) / (2 * h)
    day_dx = (a_of(c + [h, 0])[1] - a_of(c - [h, 0])[1]) / (2 * h)
    Fv = day_dx - dax_dy + np.cross(a[0], a[1])
    wx, wb = 4 * np.sin(c[0] / 2) ** 2, 4 * np.sin(beta / 2) ** 2
    a_cl = np.array([[0, 0, 0], [0, 0, -wb / (wx + wb)]])
    F_cl = np.array([0, 0, 2 * wb * np.sin(c[0]) / (wx + wb) ** 2])
    print(f"(d) SO(3) at (x,y)={tuple(c)}: |a - closed| = {np.abs(a - a_cl).max():.1e}, |F - closed| = {np.abs(Fv - F_cl).max():.1e}, F_3 = {Fv[2]:.6f}")
