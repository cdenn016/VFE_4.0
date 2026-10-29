r"""GAP06-S2 focus question 5 (continued): structure of the d = 1 minimality Jacobian of
op:boundary-minimality (chapters/10_boundary_closure.tex:264-265) and a global non-separation family.

Profile flow eq:boundary-tree-flow, d = 1, f(r, u) = m_r (1 + u a_r), two-point base with weights
A = (A_0, A_1), model g(w, u) = n_w (1 + u e_w), evaluator log rho(r, w) (general 2x2 table).
(1) sympy: a_r' depends on a_r only (no m, n, J, rho dependence); m_0' = -A_1 m_0 m_1 (H_0 - H_1)
    + (coupling depending on m, n only), H_r = log m_r + kappa(a_r) - 1,
    kappa(a) = ((1+3a) log(1+a) + (1-3a) log(1-a))/2.
(2) closed form det = det_q det_s with det_q = (A_1 m0 m1)^2 [kappa'(a1) psi(a0) - kappa'(a0) psi(a1)],
    psi = (A kappa')', compared with the interval value of g6 at the same rational point.
(3) invariant product family f = f_0(r) F(u): the root responses (belief and model base marginals)
    do not depend on the shape F, for d = 1 and for d = 2 (binary, evolving rows, coupled evaluator),
    including a base-only broadcast record at t = 0.5.  Numerical integration (DOP853).
Single-threaded, a few seconds.
"""
import itertools
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

m0, a0, a1, n0, e0, e1, A0 = sp.symbols('m0 a0 a1 n0 e0 e1 A0', positive=True)
L00, L01, L10, L11 = sp.symbols('L00 L01 L10 L11', real=True)   # log rho(r, w)
A1 = 1 - A0; B0 = sp.Rational(11, 20); B1 = 1 - B0
m1 = (1 - A0 * m0) / A1; n1 = (1 - B0 * n0) / B1
logrho = [[L00, L01], [L10, L11]]
Aw = [A0, A1]; Bw = [B0, B1]


def sector(mv, av, other, wself, wother, rho_rw):
    V = [-sum(wother[w] * other[w] * rho_rw(r, w) for w in range(2)) for r in range(2)]
    fv = {(r, u): mv[r] * (1 + u * av[r]) for r in range(2) for u in (1, -1)}
    hv = {(r, u): 2 * sp.log(fv[(r, u)]) + V[r] - (sp.log(fv[(r, -u)]) + fv[(r, -u)] / fv[(r, u)])
          for r in range(2) for u in (1, -1)}
    cq = sum(wself[r] * sp.Rational(1, 2) * fv[(r, u)] * hv[(r, u)] for r in range(2) for u in (1, -1))
    fd = {k: -fv[k] * (hv[k] - cq) for k in fv}
    md = [sp.Rational(1, 2) * (fd[(r, 1)] + fd[(r, -1)]) for r in range(2)]
    ad = [(sp.Rational(1, 2) * (fd[(r, 1)] - fd[(r, -1)]) - av[r] * md[r]) / mv[r] for r in range(2)]
    return md, ad


md, ad = sector([m0, m1], [a0, a1], [n0, n1], Aw, Bw, lambda r, w: logrho[r][w])
nd, ed = sector([n0, n1], [e0, e1], [m0, m1], Bw, Aw, lambda w, r: logrho[r][w])
print("(1) structure")
others = [m0, a1, n0, e0, e1, A0, L00, L01, L10, L11]
print("  d(a0')/d(others) all zero:", all(sp.simplify(sp.diff(ad[0], v)) == 0 for v in others))
aa = sp.symbols('aa')
Aa = sp.simplify(ad[0].subs(a0, aa))
print("  contrast field A(a) =", Aa)
kap = ((1 + 3 * aa) * sp.log(1 + aa) + (1 - 3 * aa) * sp.log(1 - aa)) / 2
H = lambda mv, av: sp.log(mv) + kap.subs(aa, av) - 1
F_formula = -A1 * m0 * m1 * (H(m0, a0) - H(m1, a1))
coup = sp.simplify(sp.expand_log(md[0] - F_formula, force=True))
print("  m0' + A1 m0 m1 (H0 - H1) depends on a0, a1?:",
      [sp.simplify(sp.diff(coup, v)) != 0 for v in (a0, a1)], "(False, False expected)")
print("  coupling part:", sp.factor(coup))

print("\n(2) closed-form determinant vs interval value (uniform bases, J = 4/5, point of g6)")
kp = sp.diff(kap, aa); psi = sp.diff(Aa * kp, aa)
det_q = (A1 * m0 * m1)**2 * (kp.subs(aa, a1) * psi.subs(aa, a0) - kp.subs(aa, a0) * psi.subs(aa, a1))
det_s = (B1 * n0 * n1)**2 * (kp.subs(aa, e1) * psi.subs(aa, e0) - kp.subs(aa, e0) * psi.subs(aa, e1))
pt = {A0: sp.Rational(1, 2), m0: sp.Rational(3, 5), a0: sp.Rational(1, 3), a1: sp.Rational(1, 4)}
ptm = {n0: sp.Rational(4, 5), e0: sp.Rational(1, 5), e1: sp.Rational(2, 5)}
dq = sp.N(det_q.subs(pt), 40)
ds = sp.N(det_s.subs({**ptm}).subs(B1, sp.Rational(1, 2)).subs(B0, sp.Rational(1, 2)), 40)
# note: det_s above uses B weights; recompute with uniform B for comparison with g6
det_s_unif = (sp.Rational(1, 2) * n0 * (2 - n0))**2 * (kp.subs(aa, e1) * psi.subs(aa, e0) - kp.subs(aa, e0) * psi.subs(aa, e1))
dsu = sp.N(det_s_unif.subs(ptm), 40)
print("  det_q = %s, det_s = %s, product = %s" % (sp.N(dq, 20), sp.N(dsu, 20), sp.N(dq * dsu, 25)))
print("  g6 interval enclosure: -0.0029444708376893736616604911082864...")
print("  omega(a) = psi/kappa' is even: omega(0.3) - omega(-0.3) =",
      sp.N((psi / kp).subs(aa, sp.Rational(3, 10)) - (psi / kp).subs(aa, -sp.Rational(3, 10)), 15))
print("  det_q vanishes on a0 = a1:", sp.simplify(det_q.subs(a1, a0)) == 0)
om = sp.lambdify(aa, psi / kp, 'mpmath')
import mpmath as mpm
vals = [float(om(mpm.mpf(x))) for x in np.linspace(0.05, 0.95, 19)]
print("  omega on (0,1) at 19 points (not monotone, so {omega(a0) = omega(a1)} has a branch off the diagonal):", np.round(vals, 4).tolist())

print("\n(3) invisibility of base-uniform shapes in the root responses")


def Pk_bits(gf, k, d):
    # binary P_k: keep first k bits, reverse bit k+1, average later bits; gf shape (2,)*(d) per base point
    out = np.zeros_like(gf)
    for u in itertools.product((0, 1), repeat=d):
        acc = 0.0; cnt = 0
        for v in itertools.product((0, 1), repeat=d):
            if v[:k] == u[:k] and v[k] != u[k]:
                acc += gf[v]; cnt += 1
        out[u] = acc / cnt
    return out


def make_flow(d, Aw_, Bw_, logrho_, Tq=0.7, Ts=1.3):
    nk = [2**(d - k - 1) for k in range(d)]; N = 2**d
    pi = np.array([nk[k] / (N - 1) for k in range(d)])
    shp = (2,) + (2,) * d

    def unpack(y):
        lf = y[:2**(d + 1)].reshape(shp); lg = y[2**(d + 1):2**(d + 2)].reshape(shp)
        yb = y[2**(d + 2):2**(d + 2) + d]; ys = y[2**(d + 2) + d:]
        return lf, lg, yb, ys

    def rhs(t, y):
        lf, lg, yb, ys = unpack(y)
        f = np.exp(lf); g = np.exp(lg)
        wA = np.array(Aw_).reshape((2,) + (1,) * d) / 2**d
        wB = np.array(Bw_).reshape((2,) + (1,) * d) / 2**d
        f = f / np.sum(wA * f); g = g / np.sum(wB * g)
        bq = np.exp(yb); bq /= bq.sum(); bs = np.exp(ys); bs /= bs.sum()
        f0 = f.reshape(2, -1).mean(1); g0 = g.reshape(2, -1).mean(1)
        Vq = -np.array([sum(Bw_[w] * g0[w] * logrho_[r][w] for w in range(2)) for r in range(2)])
        Vs = -np.array([sum(Aw_[r] * f0[r] * logrho_[r][w] for r in range(2)) for w in range(2)])
        out = []
        Ds = []
        for (p, V, bb, wX) in ((f, Vq, bq, wA), (g, Vs, bs, wB)):
            lp = np.log(p)
            hq = 2 * lp + V.reshape((2,) + (1,) * d)
            D = []
            for k in range(d):
                Plp = np.stack([Pk_bits(lp[r], k, d) for r in range(2)])
                Pp = np.stack([Pk_bits(p[r], k, d) for r in range(2)])
                hq = hq - bb[k] * (Plp + Pp / p)
                D.append(float(np.sum(wX * p * (lp - Plp))))
            cq = float(np.sum(wX * p * hq))
            out.append((-(hq - cq)).ravel())
            Ds.append(np.array(D))
        rows = []
        for (bb, D, T) in ((bq, Ds[0], Tq), (bs, Ds[1], Ts)):
            G = D + T * np.log(bb / pi)
            rows.append(-(G - bb @ G))       # d/dt log b_k (Fisher flow on the simplex)
        return np.concatenate(out + rows)
    return rhs, unpack, pi


def root_response(y, unpack, d, Aw_):
    lf, lg, _, _ = unpack(y)
    f = np.exp(lf); g = np.exp(lg)
    wA = np.array(Aw_).reshape((2,) + (1,) * d) / 2**d
    f = f / np.sum(wA * f)
    return f.reshape(2, -1).mean(1)


Aw_ = [0.4, 0.6]; Bw_ = [0.55, 0.45]
logrho_ = [[0.9 * 0.8 - 0.2, -0.5 * 0.8 - 0.2], [-0.7 * 0.8 - 0.2, 0.3 * 0.8 - 0.2]]
for d in (1, 2):
    rhs, unpack, pi = make_flow(d, Aw_, Bw_, logrho_)
    f0 = np.array([1.3, 0.8]); f0 /= Aw_ @ f0
    g0 = np.array([0.9, 1.12]); g0 /= Bw_ @ g0
    rng = np.random.default_rng(11)
    shapes = []
    for trial in range(2):
        F = np.exp(0.5 * rng.normal(size=(2,) * d)); F /= F.mean()
        shapes.append(F)
    Gshape = np.exp(0.4 * rng.normal(size=(2, 2) + (2,) * (d - 1))).reshape((2,) + (2,) * d)
    Gshape = Gshape / Gshape.reshape(2, -1).mean(1).reshape((2,) + (1,) * d)   # model: base-dependent shape
    traj = []
    for F in shapes:
        lf = np.log(f0.reshape((2,) + (1,) * d) * F[None])
        lg = np.log(g0.reshape((2,) + (1,) * d) * Gshape)
        y0 = np.concatenate([lf.ravel(), lg.ravel(), np.log(pi), np.log(pi)])
        s1 = solve_ivp(rhs, (0, 0.5), y0, method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
        y_mid = s1.y[:, -1].copy()
        # base-only broadcast record: multiply every belief value at r by L(o|r), renormalize in rhs
        Lr = np.array([0.3, 0.9])
        lf_mid, lg_mid, yb, ys = unpack(y_mid)
        lf_mid = lf_mid + np.log(Lr).reshape((2,) + (1,) * d)
        y_mid2 = np.concatenate([lf_mid.ravel(), lg_mid.ravel(), yb, ys])
        s2 = solve_ivp(rhs, (0.5, 3.0), y_mid2, method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
        traj.append((s1, s2))
    ts1 = np.linspace(0, 0.5, 6); ts2 = np.linspace(0.5, 3.0, 6)
    dev = 0.0; devshape = 0.0
    for tt in ts1:
        r0 = root_response(traj[0][0].sol(tt), unpack, d, Aw_); r1 = root_response(traj[1][0].sol(tt), unpack, d, Aw_)
        dev = max(dev, np.abs(r0 - r1).max())
    for tt in ts2:
        r0 = root_response(traj[0][1].sol(tt), unpack, d, Aw_); r1 = root_response(traj[1][1].sol(tt), unpack, d, Aw_)
        dev = max(dev, np.abs(r0 - r1).max())
        l0 = unpack(traj[0][1].sol(tt))[0]; l1 = unpack(traj[1][1].sol(tt))[0]
        devshape = max(devshape, np.abs(l0 - l1).max())
    print("  d = %d: two belief shapes F, F' (sorted values %s vs %s)" %
          (d, np.round(np.sort(shapes[0].ravel()), 3).tolist(), np.round(np.sort(shapes[1].ravel()), 3).tolist()))
    print("    max |root belief response difference| over t in [0, 3] (record at t = 0.5) = %.2e;"
          " the profiles themselves differ by up to %.3f in log" % (dev, devshape))
