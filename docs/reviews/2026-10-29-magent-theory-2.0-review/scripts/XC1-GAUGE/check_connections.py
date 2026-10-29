"""XC1-GAUGE check 2: connection conventions and examples of app:connection,
ch:link-base-transport (16) and ch:contextual-coupling (15).

Symbolic where cheap (sympy), float64 otherwise. Single-threaded.
"""
import numpy as np
import sympy as sp

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

x, y, eps = sp.symbols('x y epsilon', real=True)

# ---------------------------------------------------------- app:connection, eq:conn-curvature and its sign conventions
U = sp.Matrix([[1 + x**2, y], [sp.sin(x), 2 + y]])
a_x = sp.Matrix([[x * y, 1], [0, -y]])
a_y = sp.Matrix([[0, x], [y**2, 1]])
Ui = U.inv()
B_x = U * a_x * Ui - U.diff(x) * Ui
B_y = U * a_y * Ui - U.diff(y) * Ui
F_B = B_y.diff(x) - B_x.diff(y) + B_x * B_y - B_y * B_x
F_a = a_y.diff(x) - a_x.diff(y) + a_x * a_y - a_y * a_x
res = sp.simplify(F_B - U * F_a * Ui)
p("app:connection F_i = U (da + a^a) U^-1 residual:", res == sp.zeros(2, 2))
# first-order transport: U(c+dc) exp(-a dc) U(c)^-1 = I - B dc
d = sp.symbols('d', real=True)
Theta = U.subs(x, x + d) * (sp.eye(2) - a_x * d) * Ui
lin = sp.simplify(sp.Matrix(Theta).diff(d).subs(d, 0) + B_x)
p("app:connection Theta(c+dc,c) = I - B(dc) + O(dc^2):", lin == sp.zeros(2, 2))
# reference change U'=Uh with a' = h^-1 a h + h^-1 dh leaves B invariant
h = sp.Matrix([[sp.exp(x), 0], [y, 1]])
hi = h.inv()
ap_x = hi * a_x * h + hi * h.diff(x)
Up = U * h
Bp_x = Up * ap_x * Up.inv() - Up.diff(x) * Up.inv()
p("prop:conn-patches reference law keeps B:", sp.simplify(Bp_x - B_x) == sp.zeros(2, 2))
# passive change U'=gU: B' = g B g^-1 - dg g^-1  (same law as geo:gauge-connection A' = Ad_g A - dg g^-1)
g = sp.Matrix([[1, x], [0, sp.exp(y)]])
gi = g.inv()
Ug = g * U
Bg_x = Ug * a_x * Ug.inv() - Ug.diff(x) * Ug.inv()
p("prop:conn-patches passive law:", sp.simplify(Bg_x - (g * B_x * gi - g.diff(x) * gi)) == sp.zeros(2, 2))
p("note: ch15 writes A' = Ad_h A - dh h^-1 for a target change p'=h.p; ch16/app write a' = h^-1 a h + h^-1 dh",
  "for a reference change nu'=(h^-1)_# nu; the two agree under h -> h^-1:",
  sp.simplify((hi * a_x * h + hi * h.diff(x)) - (hi * a_x * hi.inv() - hi.diff(x) * hi.inv())) == sp.zeros(2, 2))

# ---------------------------------------------------------- ch16 witness ibt:eq:witnessX
X = sp.Matrix([[1, 0, x], [0, 1, y]])
H = X * X.T
Q = X.T * H.inv()
ax = -X.diff(x) * Q
ay = -X.diff(y) * Q
F = sp.simplify(ay.diff(x) - ax.diff(y) + ax * ay - ay * ax)
s = 1 + x**2 + y**2
claimed = sp.Matrix([[-x * y, 1 + x**2], [-(1 + y**2), x * y]]) / s**2
p("ch16 ibt:eq:witnessF residual:", sp.simplify(F - claimed) == sp.zeros(2, 2))
mc = sp.simplify(H.diff(x) + ax * H + H * ax.T)
p("ch16 metric compatibility dH + aH + Ha^T = 0 (x-component):", mc == sp.zeros(2, 2))
p("ch16 Tr F = 0:", sp.simplify(F.trace()) == 0)
# Laplace realization: v -> (x, 1 + x y); curvature is pullback with Jacobian x
Xl = sp.Matrix([[1, 0, x], [0, 1, 1 + x * y]])
Hl = Xl * Xl.T; Ql = Xl.T * Hl.inv()
alx = -Xl.diff(x) * Ql; aly = -Xl.diff(y) * Ql
Fl = sp.simplify(aly.diff(x) - alx.diff(y) + alx * aly - aly * alx)
pull = claimed.subs({y: 1 + x * y}, simultaneous=True) * x
p("ch16 Laplace example: F = pullback * Jacobian x:", sp.simplify(Fl - pull) == sp.zeros(2, 2))
# scalar backtrack example
sfun = sp.Function('s')(x)
Xs = sp.Matrix([[1, sfun]])
a_s = sp.simplify(-(Xs.diff(x) * Xs.T * (Xs * Xs.T).inv())[0, 0])
p("ch16 scalar backtrack a =", a_s, "(claimed -s ds/(1+s^2))")

# ---------------------------------------------------------- ch16 SO(3) reciprocal-link example, numerically by finite differences
def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])
def Rx(t):
    c, s_ = np.cos(t), np.sin(t); return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
def Rz(t):
    c, s_ = np.cos(t), np.sin(t); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
beta = 1.1
def links(xx, yy):
    return Rx(xx), Rz(yy) @ Rx(beta) @ Rz(-yy)
def Cmat(Qr):
    # matrix of xi -> xi - Ad_Q xi in vee coordinates; Ad_Q hat(v) = hat(Q v)
    return np.eye(3) - Qr
def conn(xx, yy, hstep=1e-6):
    Q1, Q2 = links(xx, yy)
    K = sum(Cmat(Qr).T @ Cmat(Qr) for Qr in (Q1, Q2))
    A = []
    for dx, dy in ((hstep, 0), (0, hstep)):
        P1, P2 = links(xx + dx, yy + dy); M1, M2 = links(xx - dx, yy - dy)
        V = [vee((Pp - Mm) / (2 * hstep) @ Qr.T) for Pp, Mm, Qr in ((P1, M1, Q1), (P2, M2, Q2))]
        rhs = -sum(Cmat(Qr).T @ Vr for Qr, Vr in zip((Q1, Q2), V))
        A.append(np.linalg.solve(K, rhs))
    return np.array(A)  # rows: a_x, a_y in vee coordinates
xx, yy = 0.8, 0.6
A = conn(xx, yy)
wx = 4 * np.sin(xx / 2)**2; wb = 4 * np.sin(beta / 2)**2
p("ch16 SO(3): a_x =", np.round(A[0], 10), " a_y =", np.round(A[1], 10),
  " claimed a_y =", np.round([0, 0, -wb / (wx + wb)], 10))
hh = 1e-4
Ayx = (conn(xx + hh, yy)[1] - conn(xx - hh, yy)[1]) / (2 * hh)
Axy = (conn(xx, yy + hh)[0] - conn(xx, yy - hh)[0]) / (2 * hh)
Fxy = Ayx - Axy + vee(hat(A[0]) @ hat(A[1]) - hat(A[1]) @ hat(A[0]))
p("ch16 SO(3): F_xy =", np.round(Fxy, 7), " claimed", np.round([0, 0, 2 * wb * np.sin(xx) / (wx + wb)**2], 7))

# ---------------------------------------------------------- ch15 finite witness (Procrustes) and velocity
J = np.array([[0.0, -1.0], [1.0, 0.0]])
e1, e2 = np.eye(2)
Xs_ = np.column_stack([e1, e1, e2, e2]); Ys = np.column_stack([2 * e1, 3 * e1, e2, 2 * e2])
C = np.sum(Ys * Xs_); S = np.sum(Ys * (J @ Xs_)); rho = np.hypot(C, S)
Rs = (C * np.eye(2) + S * J) / rho
cstar = 0.5 * (np.sum(Xs_**2) + np.sum(Ys**2)) - rho
p("ch15 witness (C,S,rho) =", (C, S, rho), " R* =", Rs.tolist(), " c* =", cstar)
mu1, nu1, mu2, nu2 = Xs_.T
gE = np.column_stack([(mu1 - nu1) + 2 * (mu1 - mu2), -(mu1 - nu1) + nu1 + 2 * (nu1 - nu2),
                      (mu2 - nu2) - 2 * (mu1 - mu2), -(mu2 - nu2) + nu2 - 2 * (nu1 - nu2)])
lam = sp.symbols('lambda')
Xdot = -sp.Matrix(gE) + lam * sp.Matrix(Rs.T @ Ys - Xs_)
p("ch15 grad E =", gE.tolist(), " Xdot =", Xdot.tolist())
# cusp example
for e in (-0.1, 0.1):
    xe = np.column_stack([e1, (1 + e) * e2]); ye = np.column_stack([e1, -e2])
    th = np.linspace(-np.pi, np.pi, 200001)
    vals = 0.5 * np.array([np.sum((xe - (np.cos(t) * np.eye(2) + np.sin(t) * J) @ ye)**2) for t in th[::200]])
    p("ch15 cusp eps=%+.1f: grid min %.6f, formula %.6f" % (e, vals.min(), 2 + e + 0.5 * e * e - abs(e)))

# ---------------------------------------------------------- ch15 continuum witness eigenvector and energy
Hm = np.array([[3, -1, -2, 0], [-1, 4, 0, -2], [-2, 0, 3, -1], [0, -2, -1, 4]], float)
tau = (np.sqrt(5) - 1) / 2; kap = 5 - tau
vv = np.array([1, tau, -1, -tau])
p("ch15 continuum Hv - kappa v:", np.abs(Hm @ vv - kap * vv).max(), " w =", vv @ vv, " 5-sqrt5 =", 5 - np.sqrt(5))

with open(__file__.replace(".py", "_output.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
