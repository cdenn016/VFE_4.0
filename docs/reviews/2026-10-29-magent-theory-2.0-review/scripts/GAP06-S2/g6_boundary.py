r"""GAP06-S2 focus question 5: chapters/10_boundary_closure.tex:180-276.

(a) Ergodic arc-rotation step of prop:boundary-fixed-family: the product rotation of arcs by
    fractions (c_v) is ergodic iff 1, c_v are rationally independent (Kronecker-Weyl); the
    manuscript's "rationally independent irrational fractions" admits c = (sqrt2-1, 2-sqrt2),
    for which cos 2pi(x1+x2) is a nonconstant invariant function (check below), and a Birkhoff
    average illustration for c = (sqrt2-1, sqrt3-1).
(b) rem:boundary-wreath for branching b = 3 (depth d = 1 and d = 2): fine reciprocal Fisher force on
    finite tables (base {0,1} x S_3^{#nodes}, uniform random child arrangements) against the lifted
    profile force with the replaced operator P_k (keep k digits, replace digit k+1 by a uniformly
    chosen different digit, average the rest); class costs; dim Fix(H) by orbit counting; contrast
    rates 3 + 2 b_{m-1}/(b-1) - 2 sum_{k>=m} b_k.  Control: cyclic (C_3) arrangements break class
    constancy, so transitivity on siblings alone is not enough.
(c) op:boundary-minimality, d = 1, two-point bases: exact symbolic Jacobian of
    (m0, L m0, L^2 m0, n0, L n0, L^2 n0) in the descriptor (m0, a0, a1, n0, e0, e1) at a rational
    point, determinant enclosed by mpmath interval arithmetic.
Single-threaded, < 30 s.
"""
import itertools
import numpy as np
import sympy as sp
import mpmath as mp
from mpmath import iv

print("(a) Kronecker condition")
c1, c2 = np.sqrt(2) - 1, 2 - np.sqrt(2)
x = np.random.default_rng(0).random((5, 2))
F = lambda y: np.cos(2 * np.pi * (y[:, 0] + y[:, 1]))
print("  c = (sqrt2-1, 2-sqrt2): |F(x + c) - F(x)| max = %.2e (invariant, nonconstant: F ranges over %.3f..%.3f)"
      % (np.abs(F((x + [c1, c2]) % 1) - F(x)).max(), F(x).min(), F(x).max()))
c = np.array([np.sqrt(2) - 1, np.sqrt(3) - 1])
y = np.array([0.1, 0.7]); s = 0.0; nT = 200000
for k in range(nT):
    s += np.cos(2 * np.pi * (y[0] + y[1])) + np.sin(2 * np.pi * (y[0] - 2 * y[1]))
    y = (y + c) % 1
print("  c = (sqrt2-1, sqrt3-1): Birkhoff average of a mean-zero test function over %d steps = %.2e" % (nT, s / nT))

print("\n(b) branching b = 3")
b = 3
perms = list(itertools.permutations(range(b)))


def build(d, cyclic=False):
    leaves = list(itertools.product(range(b), repeat=d))
    nodes = [()]
    for l in range(1, d):
        nodes += list(itertools.product(range(b), repeat=l))
    node_index = {v: k for k, v in enumerate(nodes)}
    arr = [tuple((s0 + j) % b for j in range(b)) for s0 in range(b)] if cyclic else perms
    thetas = list(itertools.product(range(len(arr)), repeat=len(nodes)))
    U = np.zeros((len(leaves), len(thetas), d), dtype=int)
    for i, e in enumerate(leaves):
        for t, th in enumerate(thetas):
            for j in range(d):
                v = node_index[e[:j]]
                U[i, t, j] = arr[th[v]][e[j]]
    return leaves, nodes, thetas, U, arr


def Pk(gfun, k, d):
    """P_k on functions of digits u in {0..b-1}^d (array of shape (b,)*d, or (2,)+(b,)*d per base point)."""
    if gfun.ndim == d + 1:
        return np.stack([Pk(gfun[r], k, d) for r in range(gfun.shape[0])])
    out = np.zeros_like(gfun)
    for u in itertools.product(range(b), repeat=d):
        acc = 0.0; cnt = 0
        for v in itertools.product(range(b), repeat=d):
            if v[:k] == u[:k] and v[k] != u[k]:
                acc += gfun[v]; cnt += 1
        out[u] = acc / cnt
    return out


rng = np.random.default_rng(7)
Jv = 0.6
phi_r = np.array([1.0, -1.0]); psi_w = np.array([1.0, -1.0])
logrho = Jv * np.outer(phi_r, psi_w) - np.log(np.cosh(Jv))   # (r, w)
for d in (1, 2):
    leaves, nodes, thetas, U, arr = build(d)
    N = len(leaves); nth = len(thetas)
    f = np.exp(0.4 * rng.normal(size=(2,) + (b,) * d)); f /= f.mean()
    g = np.exp(0.4 * rng.normal(size=(2,) + (b,) * d)); g /= g.mean()
    # fine laws: densities against A x uniform(thetas); arrays (N, 2, nth)
    idx = (np.arange(2)[None, :, None],) + tuple(U[:, None, :, j] for j in range(d))
    q = f[idx]; sm = g[idx]          # shape (N, 2, nth)
    lam_w = np.full((2, nth), 1.0 / (2 * nth))
    cls = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j:
                cls[i, j] = next(k for k in range(d) if leaves[i][k] != leaves[j][k])
    nk = [(b - 1) * b**(d - k - 1) for k in range(d)]
    bq = rng.dirichlet(np.ones(d)); bs = rng.dirichlet(np.ones(d))
    beta = np.array([[bq[cls[i, j]] / nk[cls[i, j]] if i != j else 0 for j in range(N)] for i in range(N)])
    gamma = np.array([[bs[cls[i, j]] / nk[cls[i, j]] if i != j else 0 for j in range(N)] for i in range(N)])
    # (b1) class costs
    Dk = []
    for k in range(d):
        Dk.append(float(np.mean(f * (np.log(f) - Pk(np.log(f), k, d)))))
    maxdev = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                KLij = float(np.sum(lam_w * q[i] * np.log(q[i] / q[j])))
                maxdev = max(maxdev, abs(KLij - Dk[cls[i, j]]))
    print("  d = %d, N = %d, |nodes| = %d, table size per agent = %d" % (d, N, len(nodes), 2 * nth))
    print("    class costs D_k(f) =", np.round(Dk, 8), "; max |KL(q_i||q_j) - D_class| = %.2e" % maxdev)
    # (b2) fine force vs profile force (belief sector), identity channels, class-constant rows
    g0 = g.reshape(2, -1).mean(1)                    # base marginal of model profile, density vs B
    Vq = -np.array([np.sum(0.5 * g0 * logrho[r]) for r in range(2)])
    worst = 0.0
    for i in range(N):
        h = np.log(q[i]) + Vq[:, None]
        for j in range(N):
            if j != i:
                h = h + beta[i, j] * (np.log(q[i]) - np.log(q[j]))
        for l in range(N):
            if l != i:
                h = h - beta[l, i] * q[l] / q[i]
        vel = -q[i] * (h - np.sum(lam_w * q[i] * h))
        # profile force h^q = 2 log f + V^q - sum_k b_k [P_k log f + P_k f / f]
        hp = 2 * np.log(f) + Vq.reshape((2,) + (1,) * d)
        for k in range(d):
            hp = hp - bq[k] * (Pk(np.log(f), k, d) + Pk(f, k, d) / f)
        cq = float(np.mean(f * hp))
        velp = -f * (hp - cq)
        lift = velp[(np.arange(2)[:, None],) + tuple(U[i][None, :, j] for j in range(d))]
        worst = max(worst, np.abs(vel - lift).max())
    print("    max |fine velocity - lifted profile velocity| over agents and table = %.2e" % worst)
    # (b3) row costs class constant (both sectors)
    Dks = [float(np.mean(g * (np.log(g) - Pk(np.log(g), k, d)))) for k in range(d)]
    dev_s = max(abs(float(np.sum(lam_w * sm[i] * np.log(sm[i] / sm[j]))) - Dks[cls[i, j]])
                for i in range(N) for j in range(N) if i != j)
    print("    model-sector class-cost deviation = %.2e" % dev_s)
    # (b4) contrast rates at a u-independent profile: linearize the profile field numerically
    for brow in (np.array([nk[k] / (N - 1) for k in range(d)]), np.array([0.3, 0.7])[:d] / np.array([0.3, 0.7])[:d].sum()):
        f0 = np.array([0.8, 1.2]).reshape((2,) + (1,) * d) * np.ones((2,) + (b,) * d)

        def prof_vel_log(lf):
            ff = np.exp(lf); ff = ff / ff.mean()
            hp = 2 * np.log(ff) + Vq.reshape((2,) + (1,) * d)
            for k in range(d):
                hp = hp - brow[k] * (Pk(np.log(ff), k, d) + Pk(ff, k, d) / ff)
            return -(hp - np.mean(ff * hp))
        base = np.log(f0)
        rates = []
        # contrast direction chi(u) = cos(2 pi u_m / b): largest nonconstant digit index m,
        # an eigenfunction of every P_k (eigenvalue 1, -1/(b-1), 0 for k >= m, k = m-1, k < m-1)
        for mlev in range(1, d + 1):
            chi = np.zeros((b,) * d)
            for u in itertools.product(range(b), repeat=d):
                chi[u] = np.cos(2 * np.pi * u[mlev - 1] / b)
            direction = np.broadcast_to(chi, (2,) + (b,) * d)
            hstep = 1e-6
            dv = (prof_vel_log(base + hstep * direction) - prof_vel_log(base - hstep * direction)) / (2 * hstep)
            ratio = -(dv * direction).sum() / (direction * direction).sum()
            pred = 3 + 2 * brow[mlev - 1] / (b - 1) - 2 * brow[mlev:].sum()
            rates.append((mlev, ratio, pred))
        print("    rows b =", np.round(brow, 4), ": contrast rates (m, measured, predicted) =",
              [(mm, round(r_, 6), round(p_, 6)) for mm, r_, p_ in rates])
    # (b5) orbit count of H on (leaf, theta) = dim Fix(H) per base point
    parent = list(range(N * nth))

    def find(z):
        while parent[z] != z:
            parent[z] = parent[parent[z]]; z = parent[z]
        return z
    th_index = {th: t for t, th in enumerate(thetas)}
    perm_index = {p: k for k, p in enumerate(arr)}
    node_index = {v: k for k, v in enumerate(nodes)}
    gens = []
    for v in nodes:
        for sw in ((1, 0, 2), (0, 2, 1)):          # transpositions generate S_3
            gens.append((v, sw))
    for (v, sw) in gens:
        lv = len(v)
        for i, e in enumerate(leaves):
            if e[:lv] != v:
                pi_e = e
            else:
                pi_e = e[:lv] + (sw[e[lv]],) + e[lv + 1:]
            ip = leaves.index(pi_e)
            for t, th in enumerate(thetas):
                th2 = list(th)
                # compensating map: theta'_{pi(w)} = theta_w o tau_w^{-1}
                newth = list(th)
                for w in nodes:
                    if len(w) > lv and w[:lv] == v:
                        w2 = w[:lv] + (sw[w[lv]],) + w[lv + 1:]
                        newth[node_index[w2]] = th[node_index[w]]
                    elif w == v:
                        old = arr[th[node_index[w]]]
                        inv_sw = [sw.index(k) for k in range(b)]
                        newp = tuple(old[inv_sw[k]] for k in range(b))
                        newth[node_index[w]] = perm_index[newp]
                tp = th_index[tuple(newth)]
                # check invariance of the path digits
                assert all(U[ip, tp, j] == U[i, t, j] for j in range(d))
                za, zb = find(i * nth + t), find(ip * nth + tp)
                if za != zb:
                    parent[za] = zb
    norb = len({find(z) for z in range(N * nth)})
    print("    orbits of H on (leaf, theta): %d (profile values per base point: b^d = %d)" % (norb, b**d))

# control: cyclic arrangements, d = 1
leaves, nodes, thetas, U, arr = build(1, cyclic=True)
f1 = np.array([[1.3, 0.6, 1.1], [0.9, 1.25, 0.85]]); f1 /= f1.mean()
q = f1[np.arange(2)[None, :, None], U[:, None, :, 0]]   # (3, 2, nth)
lw = 1.0 / (2 * len(thetas))
print("  control, cyclic C_3 arrangements, d = 1: KL(q_0||q_1) = %.6f, KL(q_0||q_2) = %.6f (siblings, unequal)"
      % (np.sum(lw * q[0] * np.log(q[0] / q[1])), np.sum(lw * q[0] * np.log(q[0] / q[2]))))

print("\n(c) op:boundary-minimality, d = 1, two-point bases: exact Jacobian")
m0, a0, a1, n0, e0, e1, J = sp.symbols('m0 a0 a1 n0 e0 e1 J')
mvec = [m0, 2 - m0]; avec = [a0, a1]; nvec = [n0, 2 - n0]; evec = [e0, e1]
phiv = [1, -1]; psiv = [1, -1]


def sector_field(mv, av, other_marg, coupling_sign):
    # V(r) = -sum_w (1/2) other_marg_w log rho(r,w), log rho = J phi_r psi_w (+ const, cancels)
    V = [-sp.Rational(1, 2) * sum(other_marg[w] * J * phiv[r] * psiv[w] for w in range(2)) for r in range(2)]
    fv = {(r, u): mv[r] * (1 + u * av[r]) for r in range(2) for u in (1, -1)}
    hv = {(r, u): 2 * sp.log(fv[(r, u)]) + V[r] - (sp.log(fv[(r, -u)]) + fv[(r, -u)] / fv[(r, u)])
          for r in range(2) for u in (1, -1)}
    cq = sum(sp.Rational(1, 4) * fv[k] * hv[k] for k in fv)
    fdot = {k: -fv[k] * (hv[k] - cq) for k in fv}
    mdot = [sp.Rational(1, 2) * (fdot[(r, 1)] + fdot[(r, -1)]) for r in range(2)]
    adot = [(sp.Rational(1, 2) * (fdot[(r, 1)] - fdot[(r, -1)]) - av[r] * mdot[r]) / mv[r] for r in range(2)]
    return mdot, adot


md, ad = sector_field(mvec, avec, nvec, 1)
nd, ed = sector_field(nvec, evec, mvec, 1)
X = sp.Matrix([md[0], ad[0], ad[1], nd[0], ed[0], ed[1]])
z = sp.Matrix([m0, a0, a1, n0, e0, e1])
L1m = X[0]; L1n = X[3]
L2m = (sp.Matrix([L1m]).jacobian(z) * X)[0]
L2n = (sp.Matrix([L1n]).jacobian(z) * X)[0]
H = sp.Matrix([m0, L1m, L2m, n0, L1n, L2n])
point = {m0: sp.Rational(3, 5), a0: sp.Rational(1, 3), a1: sp.Rational(1, 4),
         n0: sp.Rational(4, 5), e0: sp.Rational(1, 5), e1: sp.Rational(2, 5)}
iv.dps = 40


def ival(e):
    """Rigorous interval evaluation of a sympy expression tree (Add, Mul, Pow, log, numbers)."""
    if e.is_Integer:
        return iv.mpf(int(e))
    if e.is_Rational:
        return iv.mpf(int(e.p)) / iv.mpf(int(e.q))
    if e.is_Add:
        out = iv.mpf(0)
        for t in e.args:
            out += ival(t)
        return out
    if e.is_Mul:
        out = iv.mpf(1)
        for t in e.args:
            out *= ival(t)
        return out
    if e.is_Pow:
        base, ex = e.args
        if ex.is_Integer:
            bv = ival(base); k = int(ex)
            if k >= 0:
                out = iv.mpf(1)
                for _ in range(k):
                    out *= bv
                return out
            out = iv.mpf(1)
            for _ in range(-k):
                out *= bv
            return 1 / out
        raise ValueError("non-integer power %s" % e)
    if isinstance(e, sp.log):
        return iv.log(ival(e.args[0]))
    raise ValueError("unsupported node %s" % type(e))


for Jval in (sp.Rational(4, 5), sp.Integer(0)):
    sub = dict(point); sub[J] = Jval
    Jac = H.jacobian(z)
    Jnum = [[ival(sp.expand(Jac[i, j].subs(sub))) if Jac[i, j] != 0 else iv.mpf(0) for j in range(6)] for i in range(6)]
    # Leibniz determinant in interval arithmetic
    det = iv.mpf(0)
    for p in itertools.permutations(range(6)):
        sgn = 1
        pl = list(p)
        for i in range(6):
            for j in range(i + 1, 6):
                if pl[i] > pl[j]:
                    sgn = -sgn
        term = iv.mpf(sgn)
        for i in range(6):
            term *= Jnum[i][p[i]]
        det += term
    print("  J = %s: det enclosure = [%s, %s]" % (Jval, mp.nstr(det.a, 15), mp.nstr(det.b, 15)),
          "-> nonzero" if (det.a > 0 or det.b < 0) else "-> not certified")
    if Jval == 0:
        def blockdet(rows, cols):
            M3 = [[Jnum[i][j] for j in cols] for i in rows]
            dd = iv.mpf(0)
            for p in itertools.permutations(range(3)):
                sgn = 1
                for i in range(3):
                    for j in range(i + 1, 3):
                        if p[i] > p[j]:
                            sgn = -sgn
                t_ = iv.mpf(sgn)
                for i in range(3):
                    t_ *= M3[i][p[i]]
                dd += t_
            return dd
        bq_ = blockdet([0, 1, 2], [0, 1, 2]); bs_ = blockdet([3, 4, 5], [3, 4, 5])
        off = max(float(abs(Jnum[i][j]).b) for i in range(3) for j in range(3, 6))
        print("    J = 0 blocks: belief det in [%s, %s], model det in [%s, %s], max |cross entry| = %.1e"
              % (mp.nstr(bq_.a, 12), mp.nstr(bq_.b, 12), mp.nstr(bs_.a, 12), mp.nstr(bs_.b, 12), off))

print("\n(b') group control: only the cyclic subgroup C_3 at the node (transitive on siblings), uniform S_3 realization, d = 1")
leaves, nodes, thetas, U, arr = build(1)
cyc = (1, 2, 0)                              # 3-cycle on child indices
perm_index = {p: k for k, p in enumerate(arr)}
parent = list(range(3 * len(thetas)))


def find2(z):
    while parent[z] != z:
        parent[z] = parent[parent[z]]; z = parent[z]
    return z


for i in range(3):
    ip = cyc[i]
    for t, th in enumerate(thetas):
        old = arr[th[0]]
        inv = [cyc.index(k) for k in range(3)]
        newp = tuple(old[inv[k]] for k in range(3))
        tp = perm_index[newp]
        assert U[ip, tp, 0] == U[i, t, 0]
        za, zb = find2(i * len(thetas) + t), find2(ip * len(thetas) + tp)
        if za != zb:
            parent[za] = zb
print("  orbits of C_3 on (leaf, theta): %d (full S_3 gives 3 = b^d); extra invariant = parity of the arrangement"
      % len({find2(z) for z in range(3 * len(thetas))}))
print("  ordered sibling pairs: C_3 has %d orbits on the 6 ordered pairs of distinct leaves (S_3 has 1)"
      % len({frozenset({(cyc[i] if k == 1 else (cyc[cyc[i]] if k == 2 else i), cyc[j] if k == 1 else (cyc[cyc[j]] if k == 2 else j)) for k in range(3)})
             for i in range(3) for j in range(3) if i != j}))
