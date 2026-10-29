r"""D14-BND s9: first case of op:boundary-minimality (d = 1, two-point bases), numerical.

Depth d = 1: two leaves, one class (k = 0), rows trivial. Belief profile f(r, u) = m_r (1 + a_r u),
r in {0, 1}, base weights A; model profile g(w, z) = n_w (1 + e_w z), weights B. The unordered profile
tree is {(m_r, a_r^2)}_r and {(n_w, e_w^2)}_w; with normalization the descriptor has local coordinates
(m_0, a_0, a_1) and (n_0, e_0, e_1) for a_r, e_w > 0.
Root interface: the fresh root responses f_0(r, t) = m_r(t), g_0(w, t) = n_w(t) (one free scalar each).
Question: is the map descriptor -> (m_0, m_0', m_0'', n_0, n_0', n_0'') at t = 0 locally injective?
Time derivatives are Lie derivatives of the exact profile vector field (eq:boundary-tree-flow with
P_0 = bit flip), computed by nested central differences in 80-digit arithmetic.
"""
import mpmath as mp

mp.mp.dps = 80
A = [mp.mpf("0.4"), mp.mpf("0.6")]
B = [mp.mpf("0.55"), mp.mpf("0.45")]


def make_logrho(J):
    raw = [[J * mp.mpf("0.9"), -J * mp.mpf("0.5")], [-J * mp.mpf("0.7"), J * mp.mpf("0.3")]]
    Z = sum(A[r] * B[w] * mp.e ** raw[r][w] for r in range(2) for w in range(2))
    return [[raw[r][w] - mp.log(Z) for w in range(2)] for r in range(2)]


def field(state, logrho):
    """state = (f[r][u], g[w][z]) with u, z in (0: +1, 1: -1). Returns the profile velocity."""
    f, g = state
    f0 = [(f[r][0] + f[r][1]) / 2 for r in range(2)]
    g0 = [(g[w][0] + g[w][1]) / 2 for w in range(2)]
    Vq = [-sum(B[w] * g0[w] * logrho[r][w] for w in range(2)) for r in range(2)]
    Vs = [-sum(A[r] * f0[r] * logrho[r][w] for r in range(2)) for w in range(2)]

    def vel(p, V, wts):
        h = [[2 * mp.log(p[x][u]) + V[x] - (mp.log(p[x][1 - u]) + p[x][1 - u] / p[x][u]) for u in range(2)]
             for x in range(2)]
        c = sum(wts[x] * p[x][u] * h[x][u] / 2 for x in range(2) for u in range(2))
        return [[-p[x][u] * (h[x][u] - c) for u in range(2)] for x in range(2)]
    return vel(f, Vq, A), vel(g, Vs, B)


def add(s, v, h):
    return tuple([[s[k][x][u] + h * v[k][x][u] for u in range(2)] for x in range(2)] for k in range(2))


def obs_jets(state, logrho, h=mp.mpf("1e-25")):
    """(m_0, m_0', m_0'', n_0, n_0', n_0'') via Lie derivatives along the exact field."""
    def m0(s):
        return (s[0][0][0] + s[0][0][1]) / 2
    def n0(s):
        return (s[1][0][0] + s[1][0][1]) / 2
    def d1(fun, s):
        v = field(s, logrho)
        return (fun(add(s, v, h)) - fun(add(s, v, -h))) / (2 * h)
    out = []
    for fun in (m0, n0):
        out += [fun(state), d1(fun, state), d1(lambda s: d1(fun, s), state)]
    return out


def state_from_desc(desc):
    m0, a0, a1, n0, e0, e1 = desc
    m1 = (1 - A[0] * m0) / A[1]
    n1 = (1 - B[0] * n0) / B[1]
    f = [[m0 * (1 + a0), m0 * (1 - a0)], [m1 * (1 + a1), m1 * (1 - a1)]]
    g = [[n0 * (1 + e0), n0 * (1 - e0)], [n1 * (1 + e1), n1 * (1 - e1)]]
    return (f, g)


def jacobian(desc, logrho, h=mp.mpf("1e-12")):
    cols = []
    for k in range(6):
        dp = list(desc); dm = list(desc)
        dp[k] += h; dm[k] -= h
        jp = obs_jets(state_from_desc(dp), logrho)
        jm = obs_jets(state_from_desc(dm), logrho)
        cols.append([(a - b) / (2 * h) for a, b in zip(jp, jm)])
    return mp.matrix(cols).T


for J in (mp.mpf("0.8"), mp.mpf(0)):
    logrho = make_logrho(J)
    for desc in ([mp.mpf("1.1"), mp.mpf("0.35"), mp.mpf("0.2"), mp.mpf("0.9"), mp.mpf("0.25"), mp.mpf("0.4")],
                 [mp.mpf("0.8"), mp.mpf("0.5"), mp.mpf("0.15"), mp.mpf("1.2"), mp.mpf("0.3"), mp.mpf("0.1")]):
        Jm = jacobian(desc, logrho)
        if J == 0:
            # decoupled sectors: belief block (rows 0..2 x cols 0..2) and model block separately
            bq = Jm[0:3, 0:3]; bs = Jm[3:6, 3:6]
            cross = max(abs(Jm[i, j]) for i in range(3) for j in range(3, 6))
            print(f"J=0, desc={[float(x) for x in desc]}: det(belief block) = {mp.nstr(mp.det(bq), 12)}, "
                  f"det(model block) = {mp.nstr(mp.det(bs), 12)}, max cross-block entry = {mp.nstr(cross, 3)}")
        else:
            sv = mp.svd_r(Jm, compute_uv=False)
            print(f"J={float(J)}, desc={[float(x) for x in desc]}: det = {mp.nstr(mp.det(Jm), 12)}; "
                  f"singular values = {[mp.nstr(s, 6) for s in sv]}")
    # contrast-free direction check: at a_r = 0 the swap fixes the point; report the m'' sensitivity to a
logrho = make_logrho(mp.mpf("0.8"))
desc = [mp.mpf("1.1"), mp.mpf("0.35"), mp.mpf("0.2"), mp.mpf("0.9"), mp.mpf("0.25"), mp.mpf("0.4")]
Jm = jacobian(desc, logrho)
Jm2 = jacobian(desc, logrho, h=mp.mpf("1e-15"))
print(f"FD step stability of the coupled determinant: {mp.nstr(mp.det(Jm), 15)} vs {mp.nstr(mp.det(Jm2), 15)}")
