r"""D14-BND s7: prop:boundary-observable, eq:boundary-binary-flow, eq:boundary-response-derivatives,
eq:boundary-observability-det (sympy, rederived from the reciprocal potential).

Two agents, binary state/model spaces, identity channels (each agent's single source is the other,
so rows are trivial), uniform model reference, evaluator G_J(k|k') = exp(J k k')/(2 cosh J).
Also: the same with a NONuniform model reference r(k') = (1 + rho k')/2, to locate the extra term.
"""
import sympy as sp

x1, x2, y1, y2, J, rho = sp.symbols('x1 x2 y1 y2 J rho', real=True)
m, n, a, e, t, eps, c1, c2 = sp.symbols('m n a e t epsilon c1 c2', real=True)


def p(x, k):
    return (1 + k * x) / 2


def KL(xa, xb):
    return sum(p(xa, k) * sp.log(p(xa, k) / p(xb, k)) for k in (1, -1))


def local(x, y, ref_rho=0):
    tot = 0
    for k in (1, -1):
        for kp in (1, -1):
            R = (1 + ref_rho * kp) / 2 * sp.exp(J * k * kp) / (2 * sp.cosh(J))
            tot += p(x, k) * p(y, kp) * sp.log(p(x, k) * p(y, kp) / R)
    return tot


for ref, label in ((0, "uniform model reference"), (rho, "model reference (1 + rho k')/2")):
    E = local(x1, y1, ref) + local(x2, y2, ref) + KL(x1, x2) + KL(x2, x1) + KL(y1, y2) + KL(y2, y1)
    xdot1 = sp.simplify(-(1 - x1 ** 2) * sp.diff(E, x1))
    ydot1 = sp.simplify(-(1 - y1 ** 2) * sp.diff(E, y1))
    claim_x = -(1 - x1 ** 2) * (2 * sp.atanh(x1) - sp.atanh(x2) - J * y1) - (x1 - x2)
    claim_y = -(1 - y1 ** 2) * (2 * sp.atanh(y1) - sp.atanh(y2) - J * x1) - (y1 - y2)
    vals = {x1: sp.Rational(3, 10), x2: -sp.Rational(1, 5), y1: sp.Rational(1, 7), y2: sp.Rational(2, 5),
            J: sp.Rational(7, 10), rho: sp.Rational(1, 3)}
    rx = sp.N((xdot1 - claim_x).subs(vals).rewrite(sp.log), 30)
    ry = sp.N((ydot1 - claim_y).subs(vals).rewrite(sp.log), 30)
    print(f"{label}: residual of eq:boundary-binary-flow (x) {float(rx):.2e}, exchanged (y) {float(ry):.2e}")
    if ref != 0:
        cand = (1 - y1 ** 2) * sp.atanh(rho)   # expected extra model term from KL(s || r_rho)
        rr = sp.N((ydot1 - claim_y - cand).subs(vals).rewrite(sp.log), 30)
        print(f"   extra model term equals (1 - y1^2) atanh(rho): residual {float(rr):.2e} (vanishes at rho = 0)")

# exact flow (uniform reference), mean velocities
def xdot(xi, xj, yi):
    return -(1 - xi ** 2) * (2 * sp.atanh(xi) - sp.atanh(xj) - J * yi) - (xi - xj)

Fx = (xdot(m + a, m - a, n + e) + xdot(m - a, m + a, n - e)) / 2
Fy = (xdot(n + e, n - e, m + a) + xdot(n - e, n + e, m - a)) / 2
claim_Fx_low = (6 * m - J * n) * a ** 2 - 2 * J * m * a * e
claim_Fy_low = (6 * n - J * m) * e ** 2 - 2 * J * n * a * e
sc = {m: t * m, n: t * n, a: t * a, e: t * e}
ser = sp.series((Fx - (-(1 - m ** 2) * (sp.atanh(m) - J * n))).subs(sc, simultaneous=True), t, 0, 5).removeO()
resid = sp.expand(ser - claim_Fx_low.subs(sc, simultaneous=True))
print(f"F_x minus claimed expansion, through total degree 4: {sp.simplify(resid)}")
ser5 = sp.series((Fx - (-(1 - m ** 2) * (sp.atanh(m) - J * n))).subs(sc, simultaneous=True), t, 0, 6).removeO()
deg5 = sp.expand(ser5 - ser)
print(f"   (degree-5 part present, as O(5) allows: {deg5 != 0})")
# check the stated uncoupled identity: -1/2 sum (1-(m+-a)^2) atanh(m+-a) + 2 m a [atanh(m+a) - atanh(m-a)]
unc = (xdot(m + a, m - a, 0) + xdot(m - a, m + a, 0)) / 2
unc_claim = -sp.Rational(1, 2) * ((1 - (m + a) ** 2) * sp.atanh(m + a) + (1 - (m - a) ** 2) * sp.atanh(m - a)) \
    + 2 * m * a * (sp.atanh(m + a) - sp.atanh(m - a))
print(f"uncoupled identity residual: {sp.simplify(sp.expand(unc - unc_claim))}")
print(f"uncoupled series through degree 4: {sp.expand(sp.series(unc.subs(sc, simultaneous=True), t, 0, 5).removeO().subs(t, 1))}")

# Jacobian determinant of H = (m, n, F_x, F_y) wrt (m, n, a, e) = det of the (a, e) block.
# Exact derivatives, evaluated at high precision (a symbolic eps-series is too slow here).
import mpmath as mp
mp.mp.dps = 60
blk = [sp.diff(Fx, a), sp.diff(Fx, e), sp.diff(Fy, a), sp.diff(Fy, e)]
fns = [sp.lambdify((m, n, a, e, J), b_, modules="mpmath") for b_ in blk]
def det_at(epsv, c1v, c2v, Jv):
    args = (epsv * c1v, epsv * c1v, epsv * c2v, epsv * c2v, Jv)
    b0, b1, b2, b3 = [fn(*args) for fn in fns]
    return b0 * b3 - b1 * b2
for Jv in (mp.mpf("0.2"), mp.mpf("-0.15"), mp.mpf("1.7"), mp.mpf(2), mp.mpf(6)):
    for c1v, c2v in ((mp.mpf("0.7"), mp.mpf("1.3")), (mp.mpf("-1.1"), mp.mpf("0.4"))):
        lead = 4 * c1v**2 * c2v**2 * (6 - 3 * Jv) * (6 - Jv)
        r1 = det_at(mp.mpf("1e-3"), c1v, c2v, Jv) / mp.mpf("1e-3")**4
        r2 = det_at(mp.mpf("1e-4"), c1v, c2v, Jv) / mp.mpf("1e-4")**4
        # (det/eps^4 - lead)/eps^2 should approach a constant if the remainder is O(eps^6)
        q1 = (r1 - lead) / mp.mpf("1e-3")**2; q2 = (r2 - lead) / mp.mpf("1e-4")**2
        print(f"J={float(Jv):+.2f} c=({float(c1v):+.1f},{float(c2v):+.1f}): det/eps^4 at eps=1e-3,1e-4: "
              f"{float(r1):.10f}, {float(r2):.10f}; claimed {float(lead):.10f}; (det/eps^4-lead)/eps^2: {float(q1):.4f}, {float(q2):.4f}")

# Off-diagonal base points: the exclusion J not in {2, 6} is an artifact of the point m = n, a = e.
# Leading block at (m, n, a, e) = eps (cm, cn, ca, ce): det = eps^4 * Q(c; J) + O(eps^6) with
# Q = [2(6cm - J cn) ca - 2J cm ce][2(6cn - J cm) ce - 2J cn ca] - 4 J^2 cm cn ca ce.
cm, cn, ca, ce = sp.symbols('cm cn ca ce', real=True)
Q = (2*(6*cm - J*cn)*ca - 2*J*cm*ce) * (2*(6*cn - J*cm)*ce - 2*J*cn*ca) - 4*J**2*cm*cn*ca*ce
print("off-diagonal leading coefficients Q(c; J):")
for Jv, cv in ((2, (1, 0, 1, 1)), (6, (1, 0, 1, sp.Rational(1, 2))), (0, (1, 1, 1, 1))):
    Qv = Q.subs({J: Jv, cm: cv[0], cn: cv[1], ca: cv[2], ce: cv[3]})
    epsv = mp.mpf("1e-4")
    args = tuple(epsv * mp.mpf(float(c)) for c in cv) + (mp.mpf(Jv),)
    b0, b1, b2, b3 = [fn(*args) for fn in fns]
    print(f"   J={Jv}, c={cv}: Q = {Qv}; exact det/eps^4 at eps=1e-4: {float((b0*b3 - b1*b2)/epsv**4):.8f}")
