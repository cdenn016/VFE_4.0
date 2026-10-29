r"""D16-OPEN s2: the open:routing analytic chain (focus question 2).

 [1] open:core against the four-agent limit field open:limit at random (a,b).
 [2] open:early-bounds on a >= 4, 0 < b <= 2: ranges of -bdot/b and -adot/a.
     Also: is -bdot/b > 1 throughout (the open:null-control claim "peer forces tighten")?
 [3] the derivative bound |g_c'(r)| <= (1+4r^2) e^{-2r^2} for c >= 1.
 [4] late b = 0 comparison: a_c, a_*, S bound, Bdot formula.
 [5] core integration from A = 34 and A = 4096: tau, t_A, b(t_A) against the stated bounds,
     then the late window from (4, b(t_A)) under open:core (numerical evidence only).
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
out = []


def limit_field(u):
    d = u[:, None] - u[None, :]
    n = np.exp(-d ** 2 / 2)
    W = n / n.sum(1, keepdims=True)
    X = u[None, :] - u[:, None]
    return -u + ((W + W.T) * X).sum(1)


def core(a, b):
    w = np.exp(-2 * b * b); v0 = np.exp(-2 * a * a)
    vp = np.exp(-2 * (a + b) ** 2); vm = np.exp(-2 * (a - b) ** 2)
    Zp = 1 + w + v0 + vp; Zm = 1 + w + v0 + vm
    cp = w * (1 / Zp + 1 / Zm); c0 = v0 * (1 / Zp + 1 / Zm)
    da = -a - 2 * a * c0 - 2 * ((a + b) * vp / Zp + (a - b) * vm / Zm)
    db = -b - 2 * b * cp - 2 * ((a + b) * vp / Zp - (a - b) * vm / Zm)
    return da, db


# [1]
res = 0.0
for _ in range(400):
    a, b = rng.uniform(0, 5), rng.uniform(-3, 3)
    u = np.array([-a - b, -a + b, a - b, a + b])
    f = limit_field(u)
    ad, bd = (f[2] + f[3]) / 2, (f[3] - f[2]) / 2
    da, db = core(a, b)
    res = max(res, abs(ad - da), abs(bd - db))
    # reflection invariance: f should be antisymmetric under u -> -reverse(u)
    res = max(res, abs(f[0] + f[3]), abs(f[1] + f[2]))
out.append("[1] open:core vs limit field (400 random (a,b)): max residual %.3e" % res)

# [2]
A_ = np.concatenate([np.linspace(4, 8, 801), np.linspace(8, 40, 321)])
B_ = np.linspace(1e-6, 2, 2001)
aa, bb = np.meshgrid(A_, B_, indexing="ij")
da, db = core(aa, bb)
rb = -db / bb; ra = -da / aa
lo_b, hi_b = 1 - 68 * np.exp(-8), 3 + 68 * np.exp(-8)
lo_a, hi_a = 1.0, 1 + 4 * np.exp(-32) + 4 * np.exp(-8)
out.append("[2] -bdot/b on a in [4,40], b in (0,2]: min %.9f, max %.9f; stated [%.6f, %.6f]"
           % (rb.min(), rb.max(), lo_b, hi_b))
out.append("    -adot/a: min %.12f, max %.9f; stated [%.6f, %.7f]" % (ra.min(), ra.max(), lo_a, hi_a))
i, j = np.unravel_index(np.argmin(rb), rb.shape)
out.append("    min of (-bdot/b - 1) = %.3e at a = %.4f, b = %.4f  (positive => full field contracts faster than null)"
           % (rb.min() - 1, A_[i], B_[j]))
# local refinement near the minimizer
from scipy.optimize import minimize
fun = lambda p: (-core(p[0], p[1])[1] / p[1]) if (p[0] >= 4 and 0 < p[1] <= 2) else 10.0
best = minimize(fun, [A_[i], B_[j]], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-16))
out.append("    refined min of -bdot/b - 1 = %.6e at (a,b) = (%.6f, %.6f)" % (best.fun - 1, *best.x))
# what the stated bounds alone imply
out.append("    stated lower bound 1 - 68e^-8 = %.6f < 1, so open:early-bounds alone does not give a rate above 1" % lo_b)

# [3]
rr = np.linspace(2, 12, 20001)
worst = 0.0
for c in (1.0, 1.0 + 1e-3, 1.5, 3.0):
    E = np.exp(-2 * rr ** 2)
    gp = E * ((1 - 4 * rr ** 2) * c + E) / (c + E) ** 2
    worst = max(worst, (np.abs(gp) / ((1 + 4 * rr ** 2) * E)).max())
out.append("[3] max |g_c'| / ((1+4r^2)e^{-2r^2}) on r in [2,12], c in {1,1.001,1.5,3}: %.6f (<= 1 required)" % worst)
r2 = np.linspace(2, 6, 4001); bnd = (1 + 4 * r2 ** 2) * np.exp(-2 * r2 ** 2)
out.append("    (1+4r^2)e^{-2r^2} decreasing on [2,6]: %s" % bool(np.all(np.diff(bnd) < 0)))

# [4]
ac = np.sqrt(np.log(9) / 2); astar = np.sqrt(np.log(4) / 2)
sol = solve_ivp(lambda t, y: [-y[0] * (1 + 4 * np.exp(-2 * y[0] ** 2) / (1 + np.exp(-2 * y[0] ** 2)))],
                (0, 3), [4.0], rtol=1e-12, atol=1e-14, dense_output=True,
                events=[lambda t, y: y[0] - ac, lambda t, y: y[0] - astar])
tc, ts_ = sol.t_events[0][0], sol.t_events[1][0]
v = np.exp(-2 * ac ** 2); Bdot = 4 * ac ** 2 * v * (1 + 4 * v / (1 + v)) / (1 + v) ** 2
out.append("[4] a_c = %.6f, B(a_c) = %.6f; a_* = %.6f, B(a_*) = %.6f" % (ac, v / (1 + v), astar,
           np.exp(-2 * astar ** 2) / (1 + np.exp(-2 * astar ** 2))))
out.append("    time 4 -> a_c = %.6f; S (4 -> a_*) = %.6f <= log(4/a_*) = %.6f; Bdot(a_c) = %.6f"
           % (tc, ts_, np.log(4 / astar), Bdot))
# Bdot formula vs numerical derivative
a0 = 1.3; h = 1e-6
Bf = lambda a: np.exp(-2 * a * a) / (1 + np.exp(-2 * a * a))
adot = -a0 * (1 + 4 * Bf(a0))
num = (Bf(a0 + h) - Bf(a0 - h)) / (2 * h) * adot
v0 = np.exp(-2 * a0 ** 2)
out.append("    Bdot formula check at a=1.3: closed %.12f, chain rule %.12f" %
           (4 * a0 ** 2 * v0 * (1 + 4 * Bf(a0)) / (1 + v0) ** 2, num))


# [5]
def run_core(A):
    ev_b = lambda t, y: y[1] - 0.5
    ev_a = lambda t, y: y[0] - 4.0
    ev_a.terminal = True
    s = solve_ivp(lambda t, y: core(y[0], y[1]), (0, 50), [A, 2.0], rtol=1e-11, atol=1e-14,
                  events=[ev_b, ev_a], dense_output=True)
    tau = s.t_events[0][0]; tA = s.t_events[1][0]; bA = s.y_events[1][0][1]
    return s, tau, tA, bA


for A in (34.0, 4096.0):
    s, tau, tA, bA = run_core(A)
    out.append("[5] A=%g: tau=%.5f (<1.42), a(tau)=%.4f (>=8), t_A=%.5f in [%.5f, %.5f], b(t_A)=%.4e <= %.4e"
               % (A, tau, s.sol(tau)[0], tA, np.log(A / 4) / 1.0014, np.log(A / 4), bA, 2 * (A / 4) ** -0.975))
    # pair weights during [0, t_A]
    tt = np.linspace(0, tA, 4000)
    a_, b_ = s.sol(tt)
    w = np.exp(-2 * b_ ** 2); v0 = np.exp(-2 * a_ ** 2)
    vp = np.exp(-2 * (a_ + b_) ** 2); vm = np.exp(-2 * (a_ - b_) ** 2)
    pp = w / (1 + w + v0 + vp); pm = w / (1 + w + v0 + vm)
    k = np.argmax(np.minimum(pp, pm) > 0.1)
    out.append("    first-pair weights cross 1/10 near t=%.4f; on [tau, t_A] min p = %.4f (>1/4); "
               "max cross weight on [0,t_A] = %.3e (<= e^-8 = %.3e)" %
               (tt[k], np.minimum(pp, pm)[tt >= tau].min(), max(vm.max(), vp.max(), v0.max()), np.exp(-8)))
    # late window from (4, b(t_A))
    late = solve_ivp(lambda t, y: core(y[0], y[1]), (0, 1.6), [4.0, bA], rtol=1e-11, atol=1e-14,
                     dense_output=True)
    tl = np.linspace(0, 1.6, 16001)
    a_, b_ = late.sol(tl)
    w = np.exp(-2 * b_ ** 2); v0 = np.exp(-2 * a_ ** 2)
    vp = np.exp(-2 * (a_ + b_) ** 2); vm = np.exp(-2 * (a_ - b_) ** 2)
    Zp = 1 + w + v0 + vp; Zm = 1 + w + v0 + vm
    Bpar = 0.5 * ((v0 + vp) / Zp + (vm + v0) / Zm)            # (1/2) sum_{i in P1} sum_{j in P2} beta_ij
    stop = np.argmax(Bpar >= 0.2)
    cross = np.argmax(Bpar > 0.1)
    p_min = np.minimum(w / Zp, w / Zm)[:stop].min()
    contrast = np.minimum((w - v0 - vp) / Zp, (w - v0 - vm) / Zm)[:stop].min()
    dB = np.gradient(Bpar, tl)
    out.append("    late: parent row crosses 1/10 at +%.4f (dB/dt=%.4f there), reaches 1/5 at +%.4f; "
               "min first-pair weight %.4f, min first-pair contrast %.4f before B=1/5"
               % (tl[cross], dB[cross], tl[stop], p_min, contrast))

print("\n".join(out))
