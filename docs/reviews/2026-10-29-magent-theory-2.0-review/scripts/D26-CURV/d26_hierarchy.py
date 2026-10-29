r"""D26-CURV check 4: supplied hierarchies (it3:cap, it3:rigid, it3:flat-wall, it3:wall-residual).

(a) Cap: eliminating a free cross-scale dictionary leaves precision (1/tau + s2|c_A|^2)^-1 I.
(b) Normalizer counterexample (sympy): p = z^(-2/3)/3 on (0,1) integrates to 1, p^2 does not.
(c) Gaussian rigid limit: minimizer error times tau tends to a constant; the normalized soft Gaussian law
    (mean and covariance) also converges to the rigid restriction at rate O(1/tau).
(d) Flat wall: scalar counterexample without paths; a case where the pairing graph is connected although
    some bridge pair has no closing walk; a cross-scale path with non-identity dictionaries.
(e) Wall residual formula with nonzero internal disagreements.
"""
from d26_common import *
import sympy as sp

rng = np.random.default_rng(4242)

K = 3; s2 = 0.8; c_A = rng.standard_normal(K)
for tau in (1.0, 10.0, 1e4):
    kk = kap(c_A)
    Aw = np.eye(K * K) / s2 + tau * kk.T @ kk
    eff = tau * np.eye(K) - tau ** 2 * kk @ np.linalg.solve(Aw, kk.T)
    print("(a) tau =", tau, ": |effective precision - (1/tau + s2|c|^2)^-1 I| =",
          np.abs(eff - np.eye(K) / (1 / tau + s2 * c_A @ c_A)).max(), " cap 1/(s2|c|^2) =", 1 / (s2 * c_A @ c_A))

z = sp.symbols("z", positive=True)
p = sp.Rational(1, 3) * z ** sp.Rational(-2, 3)
print("(b) int_0^1 p =", sp.integrate(p, (z, 0, 1)), " int_0^1 p^2 =", sp.integrate(p ** 2, (z, 0, 1)))

K = 2; s2 = 0.6; mm = K * K
internal = [(0, 1), (1, 0), (2, 3)]
bridges = [(2, 0), (3, 1), (0, 3)]
cross = [(4, 0), (4, 1), (5, 2), (5, 3)]
Pch = [spd(rng, K) for _ in range(4)]; Ppa = [spd(rng, K, 0.2) for _ in range(2)]
mch = [rng.standard_normal(K) + (2.0 if v < 2 else -2.0) for v in range(4)]
mpa = [rng.standard_normal(K) for _ in range(2)]
Tint = [rng.standard_normal((2, K)) for _ in internal]; Rint = [spd(rng, 2, 0.5) for _ in internal]
oint = [rng.standard_normal(2) for _ in internal]
Tbr = [rng.standard_normal((2, K)) for _ in bridges]; Rbr = [spd(rng, 2, 0.5) for _ in bridges]
obr = [rng.standard_normal(2) for _ in bridges]
cvec = [rng.standard_normal(K) for _ in range(6)]
nY = 6 * K; nW = len(bridges) * mm; n = nY + nW
yi = lambda v: slice(v * K, (v + 1) * K)
wi = lambda b: slice(nY + b * mm, nY + (b + 1) * mm)
par = {0: 0, 1: 0, 2: 1, 3: 1}


def soft():
    """Quadratic part H, linear part h of the soft energy without the tau term; C lists child-parent differences."""
    H = np.zeros((n, n)); h = np.zeros(n)
    for v in range(4):
        H[yi(v), yi(v)] += Pch[v]; h[yi(v)] += Pch[v] @ mch[v]
    for a in range(2):
        H[yi(4 + a), yi(4 + a)] += Ppa[a]; h[yi(4 + a)] += Ppa[a] @ mpa[a]
    for k, (j, i) in enumerate(internal):
        L = np.zeros((2, n)); L[:, yi(i)] = Tint[k]; L[:, yi(j)] = -Tint[k]
        Ri = np.linalg.inv(Rint[k]); H += L.T @ Ri @ L; h += L.T @ Ri @ oint[k]
    for b, (j, i) in enumerate(bridges):
        L = np.zeros((2, n)); L[:, yi(i)] = Tbr[b]; L[:, yi(j)] = -Tbr[b]; L[:, wi(b)] = -Tbr[b] @ kap(cvec[i])
        Ri = np.linalg.inv(Rbr[b]); H += L.T @ Ri @ L; h += L.T @ Ri @ obr[b]
        H[wi(b), wi(b)] += np.eye(mm) / s2
    C = np.zeros((len(cross) * K, n))
    for k, (A, i) in enumerate(cross):
        C[k * K:(k + 1) * K, yi(i)] = np.eye(K); C[k * K:(k + 1) * K, yi(A)] = -np.eye(K)
    return H, h, C


def rigid():
    """Rigid parent model of it3:rigid, embedded back into soft coordinates (children copy their parent)."""
    PA = Ppa[0] + Pch[0] + Pch[1]; PB = Ppa[1] + Pch[2] + Pch[3]
    mA = np.linalg.solve(PA, Ppa[0] @ mpa[0] + Pch[0] @ mch[0] + Pch[1] @ mch[1])
    mB = np.linalg.solve(PB, Ppa[1] @ mpa[1] + Pch[2] @ mch[2] + Pch[3] @ mch[3])
    nr = 2 * K + nW; Hr = np.zeros((nr, nr)); hr = np.zeros(nr)
    ya = lambda a: slice(a * K, (a + 1) * K)
    wb = lambda b: slice(2 * K + b * mm, 2 * K + (b + 1) * mm)
    Hr[ya(0), ya(0)] += PA; hr[ya(0)] += PA @ mA; Hr[ya(1), ya(1)] += PB; hr[ya(1)] += PB @ mB
    for b, (j, i) in enumerate(bridges):
        L = np.zeros((2, nr)); L[:, ya(par[i])] += Tbr[b]; L[:, ya(par[j])] -= Tbr[b]
        L[:, wb(b)] = -Tbr[b] @ kap(cvec[i])
        Ri = np.linalg.inv(Rbr[b]); Hr += L.T @ Ri @ L; hr += L.T @ Ri @ obr[b]
        Hr[wb(b), wb(b)] += np.eye(mm) / s2
    xr = np.linalg.solve(Hr, hr); Sr = np.linalg.inv(Hr)
    E = np.zeros((n, nr))
    for v in range(6):
        E[yi(v), ya(par.get(v, v - 4))] = np.eye(K)
    E[nY:, 2 * K:] = np.eye(nW)
    return E @ xr, E @ Sr @ E.T


x_rig, S_rig = rigid()
H, h, C = soft()
for tau in (1e1, 1e2, 1e3, 1e4):
    J = tau * C.T @ C + H
    x = np.linalg.solve(J, h); S = np.linalg.inv(J)
    print("(c) tau =", tau, ": tau*|x_soft - x_rigid| =", round(float(tau * np.abs(x - x_rig).max()), 6),
          "  tau*|Cov_soft - Cov_rigid| =", round(float(tau * np.abs(S - S_rig).max()), 6))

walks = {"b1->a1->b1": 1 * 1, "b2->a2->b2": 2 * 0.5}
print("(d) no-path example: available walk holonomies", walks, " J->I bridges carry 1 and 2")
I_edges = {("a1", "a2")}; J_edges = {("b2", "b1")}


def path(u, v, E):
    return u == v or (u, v) in E


eb = {"e1": ("b1", "a1"), "e2": ("b2", "a2")}; gb = {"g1": ("a2", "b2"), "g2": ("a1", "b1")}
pairs = [(e, g) for e in eb for g in gb if path(eb[e][1], gb[g][0], I_edges) and path(gb[g][1], eb[e][0], J_edges)]
print("    pairs with a closing walk:", pairs, "(the manuscript hypothesis asks for all 4)")
th = {"e1": 1.7, "g1": 1 / 1.7, "g2": 1 / 1.7}
th["e2"] = 1 / th["g1"]
print("    flatness of the three available walks forces theta_e2 =", th["e2"], "= theta_e1 =", th["e1"],
      "; product on the unavailable pair (e2, g2) =", th["e2"] * th["g2"])
t1, t2, te = 2.0, 0.25, 3.0
tg = 1 / (t2 * t1 * te)
print("    cross-scale path a1 -> A -> a2 with dictionaries (2, 0.25): walk holonomy", te * t1 * t2 * tg,
      "but theta_g theta_e =", tg * te)

cc = np.ones(K)
cvec[:] = [cc] * 6
obr[:] = [np.zeros(2) for _ in bridges]
H, h, C = soft()
x = np.linalg.solve(5.0 * C.T @ C + H, h)


def F(T, R):
    a = s2 * (cc @ cc)
    return a * T.T @ np.linalg.solve(R + a * T @ T.T, T)


e, g = 0, 2
De = x[yi(0)] - x[yi(2)]; Dg = x[yi(3)] - x[yi(0)]
lhs = (x[wi(e)].reshape(K, K) + x[wi(g)].reshape(K, K)) @ cc
rhs = (F(Tbr[e], Rbr[e]) - F(Tbr[g], Rbr[g])) @ De + F(Tbr[g], Rbr[g]) @ (De + Dg)
print("(e) tau = 5: |(omega_e + omega_g) c - [(F_e - F_g) Delta_e + F_g (Delta_e + Delta_g)]| =", np.abs(lhs - rhs).max(),
      " |Delta_e + Delta_g| =", round(float(np.abs(De + Dg).max()), 6))
