"""X1-GAUGE: appendix base_connection: prop:conn-curvature, prop:conn-patches, the curved
rank-one example, small-loop holonomy, and the circle holonomy. float64, single thread."""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
d = 2
C1, C2, C3 = rng.normal(size=(3, d, d)) * 0.4
A0, A1, A2 = rng.normal(size=(3, d, d)) * 0.5
Bx0, By0 = rng.normal(size=(2, d, d)) * 0.5

def U(c):            # smooth invertible presentation
    x, y = c
    return expm(C1 * x + C2 * y + C3 * x * y)
def a(c):            # reference connection form components (a_x, a_y)
    x, y = c
    return (A0 + A1 * y + Bx0 * x * y, A2 + A1 * x + By0 * np.sin(x))

def dfun(f, c, k, eps=1e-5):
    e = np.zeros(2); e[k] = eps
    return (f(c + e) - f(c - e)) / (2 * eps)

def B(c):
    Uc = U(c); Ui = np.linalg.inv(Uc); ax, ay = a(c)
    dUx = dfun(U, c, 0); dUy = dfun(U, c, 1)
    return (Uc @ ax @ Ui - dUx @ Ui, Uc @ ay @ Ui - dUy @ Ui)

c0 = np.array([0.3, -0.2])
# curvature from B: F_xy = d_x B_y - d_y B_x + [B_x, B_y]
Bx, By = B(c0)
dBy_dx = dfun(lambda c: B(c)[1], c0, 0, 1e-4); dBx_dy = dfun(lambda c: B(c)[0], c0, 1, 1e-4)
F_B = dBy_dx - dBx_dy + Bx @ By - By @ Bx
ax, ay = a(c0)
day_dx = dfun(lambda c: a(c)[1], c0, 0); dax_dy = dfun(lambda c: a(c)[0], c0, 1)
F_a = U(c0) @ (day_dx - dax_dy + ax @ ay - ay @ ax) @ np.linalg.inv(U(c0))
print("prop:conn-curvature |F(B) - U(da+a^a)U^-1| =", np.abs(F_B - F_a).max())

# first-order transport: Theta(c+dc,c) = U(c+dc) exp(-a.dc) U(c)^-1 = I - B(dc) + O(dc^2)
dc = np.array([1e-5, -2e-5])
Th = U(c0 + dc) @ expm(-(ax * dc[0] + ay * dc[1])) @ np.linalg.inv(U(c0))
print("first order |(Theta - I) + B(dc)| / |dc| =", np.abs((Th - np.eye(d)) + (Bx * dc[0] + By * dc[1])).max() / np.linalg.norm(dc))

# small-loop holonomy with dT/dt = -B(gamma') T, counterclockwise square of side eps at c0
def transport(path_pts):
    T = np.eye(d)
    for p, q in zip(path_pts[:-1], path_pts[1:]):
        def rhs(t, y):
            c = p + t * (q - p)
            Bx_, By_ = B(c)
            Bm = Bx_ * (q - p)[0] + By_ * (q - p)[1]
            return (-Bm @ y.reshape(d, d)).ravel()
        sol = solve_ivp(rhs, (0, 1), T.ravel(), rtol=1e-11, atol=1e-13)
        T = sol.y[:, -1].reshape(d, d)
    return T
for eps in [2e-2, 1e-2]:
    pts = [c0, c0 + [eps, 0], c0 + [eps, eps], c0 + [0, eps], c0]
    Hol = transport(pts)
    print(f"eps={eps}: |(Hol - I)/eps^2 + F_xy| =", np.abs((Hol - np.eye(d)) / eps**2 + F_B).max())

# prop:conn-patches: U' = g U gives B' = g B g^-1 - dg g^-1 ; U' = U h with a' = h^-1 a h + h^-1 dh gives B' = B
G1 = rng.normal(size=(d, d)) * 0.3
g = lambda c: expm(G1 * c[0] + G1.T * c[1] ** 2)
Bg_direct = []
Ug = lambda c: g(c) @ U(c)
for k in range(2):
    dUg = dfun(Ug, c0, k)
    Bk = Ug(c0) @ a(c0)[k] @ np.linalg.inv(Ug(c0)) - dUg @ np.linalg.inv(Ug(c0))
    pred = g(c0) @ B(c0)[k] @ np.linalg.inv(g(c0)) - dfun(g, c0, k) @ np.linalg.inv(g(c0))
    Bg_direct.append(np.abs(Bk - pred).max())
print("agent re-expression law residuals:", Bg_direct)
hfun = lambda c: expm(G1.T * c[0] - G1 * np.cos(c[1]))
Uh = lambda c: U(c) @ hfun(c)
res = []
for k in range(2):
    ap = np.linalg.inv(hfun(c0)) @ a(c0)[k] @ hfun(c0) + np.linalg.inv(hfun(c0)) @ dfun(hfun, c0, k)
    Bk = Uh(c0) @ ap @ np.linalg.inv(Uh(c0)) - dfun(Uh, c0, k) @ np.linalg.inv(Uh(c0))
    res.append(np.abs(Bk - B(c0)[k]).max())
print("reference affine law residuals (B unchanged):", res)

# curved rank-one example
E12 = np.array([[0., 1.], [0., 0.]]); E21 = E12.T
print("F_xy for a_x=E12, a_y=E21:", (E12 @ E21 - E21 @ E12).tolist())
# circle holonomy B = b dtheta: dT/dtheta = -b T over 2pi
b = 0.3
print("circle holonomy:", np.exp(-2 * np.pi * b), " (flat but != 1)")
