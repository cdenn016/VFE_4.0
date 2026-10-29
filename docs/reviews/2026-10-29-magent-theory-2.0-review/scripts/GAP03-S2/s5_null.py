r"""GAP03-S2 s5: null-flow control of form:kappa-independence (iii), form:selection, form:responses.

Two row conventions for the null of inh:null-flow (peer costs D_ij, j != i, deleted):
  (a) self weight one (the convention computed in the proof of form:kappa-independence (iii)):
        reduced belief potential sum_i [KL(q_i||W(a0,I)) + KL(q_i||W(a0,Lam_i^0))]
        => alpha_i' = -2(alpha_i - a0),  P_i' = -2 P_i + I + C_i^0.
  (b) labels and priors kept (pa:null-row, cp:null-row, it7:null-row):
        reduced row term -T log(pi_ii e^{-D_ii/T} + 1 - pi_ii)
        => alpha_i' = -(1+beta_i)(alpha_i - a0),  P_i' = -(1+beta_i) P_i + I + beta_i C_i^0,
        beta_i = pi_ii e^{-D_ii/T} / (pi_ii e^{-D_ii/T} + 1 - pi_ii).
  Lam_i^0 = I + (kappa/N)(alpha_i P_i^{-1} + M^0_{-i}),  C_i^0 = Lam_i^0 + (kappa/N)(alpha_i P_i^{-1} - a0 (Lam_i^0)^{-1}).
Null parent dynamics after a join:
  (N1) fine-level null throughout; the selector reads block means of fine shapes;
  (N2) at a join the block is replaced by a parent that follows the null of the inherited parent
       construction: self prior n_I/N, parent self channel W(n_I a0, Lam_I^0); exact only while all
       precisions are common (here: kappa = 0 and P(0) = I under (b); any case under (a)).
Diagonal precisions stay diagonal under both nulls, so P_i = diag(x_i, y_i).
Single-threaded; every ODE is 8- to 12-dimensional.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammaln, digamma

N, A0, DELTA = 4, 1.0, 0.1


def lG2(a):
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a):
    return digamma(a) + digamma(a - 0.5)


def kl_diag(a, x, y, b, lx, ly):
    """KL(W(a, diag(x,y)) || W(b, diag(lx,ly)))"""
    return (a - b) * psi2(a) - lG2(a) + lG2(b) + b * (np.log(x / lx) + np.log(y / ly)) + a * (lx / x + ly / y - 2)


def null_rhs_factory(conv, T, kap, alpha_init, x0, y0, prior=None):
    """fine null field for N agents; state = [alpha(4), x(4), y(4)]"""
    mx0 = np.sum(alpha_init / x0)
    my0 = np.sum(alpha_init / y0)
    mxo = mx0 - alpha_init / x0     # M^0_{-i}, frozen initial peers
    myo = my0 - alpha_init / y0
    pii = 1.0 / N if prior is None else prior

    def rhs(t, s):
        al, x, y = s[:N], s[N:2 * N], s[2 * N:]
        lx = 1 + (kap / N) * (al / x + mxo)
        ly = 1 + (kap / N) * (al / y + myo)
        Cx = lx + (kap / N) * (al / x - A0 / lx)
        Cy = ly + (kap / N) * (al / y - A0 / ly)
        if conv == 'a':
            beta = np.ones(N)
        else:
            D = kl_diag(al, x, y, A0, lx, ly)
            w = pii * np.exp(-D / T)
            beta = w / (w + 1 - pii)
        return np.concatenate([-(1 + beta) * (al - A0), -(1 + beta) * x + 1 + beta * Cx, -(1 + beta) * y + 1 + beta * Cy])
    return rhs


def history_from_shapes(shape_fun, tmax, n0=N, grid=4000):
    """Event-driven threshold rule (form:threshold-rule) along a shape trajectory shape_fun(t) -> array(N).
    Returns list of (time, frozenset(block I), frozenset(block J))."""
    blocks = [frozenset([i]) for i in range(n0)]
    tcur = 0.0
    hist = []
    ts = np.linspace(0, tmax, grid)
    S = np.array([shape_fun(t) for t in ts])
    while len(blocks) > 1:
        best = None
        for ii in range(len(blocks)):
            for jj in range(ii + 1, len(blocks)):
                I, J = sorted(blocks[ii]), sorted(blocks[jj])
                dist = np.abs(S[:, I].mean(axis=1) - S[:, J].mean(axis=1)) - DELTA
                idx = np.where((ts > tcur + 1e-12) & (dist[:] <= 0))[0]
                if len(idx) == 0:
                    continue
                k = idx[0]
                if k == 0 or ts[k - 1] < tcur:
                    lo = tcur + 1e-12
                else:
                    lo = ts[k - 1]
                g = lambda tt: abs(shape_fun(tt)[I].mean() - shape_fun(tt)[J].mean()) - DELTA
                if g(lo) <= 0:   # already below at the current time: not reached from above
                    continue
                te = brentq(g, lo, ts[k], xtol=1e-13)
                if best is None or te < best[0]:
                    best = (te, blocks[ii], blocks[jj])
        if best is None:
            break
        te, I, J = best
        hist.append((te, I, J))
        blocks = [b for b in blocks if b not in (I, J)] + [I | J]
        tcur = te
    return hist


def full_history(u):
    return history_from_shapes(lambda t: A0 + np.exp(-2 * t) * (u - A0), 6.0)


def null_history_N1(u, conv, T, kap, P0diag, tmax=8.0):
    x0 = np.full(N, P0diag[0]); y0 = np.full(N, P0diag[1])
    rhs = null_rhs_factory(conv, T, kap, u, x0, y0)
    sol = solve_ivp(rhs, (0, tmax), np.concatenate([u, x0, y0]), method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    return history_from_shapes(lambda t: sol.sol(t)[:N], tmax), sol


def null_history_N2_k0_identity(u, T, tmax=8.0):
    """(N2) under convention (b) at kappa = 0, P(0) = I: precisions stay I exactly, so parents are W(A_I, I).
    Units carry (A_U, n_U); unit null field: A' = -(1+beta_U)(A - n a0), beta_U from prior n_U/N and
    D_U = KL(W(A_U, I) || W(n_U a0, I))."""
    units = [(float(u[i]), 1, frozenset([i])) for i in range(N)]
    tcur = 0.0
    hist = []
    while len(units) > 1:
        A = np.array([x[0] for x in units]); nn = np.array([x[1] for x in units])

        def rhs(t, s):
            D = kl_diag(s, 1.0, 1.0, nn * A0, 1.0, 1.0)
            pi = nn / N
            w = pi * np.exp(-D / T)
            beta = np.where(pi < 1, w / (w + 1 - pi), 1.0)
            return -(1 + beta) * (s - nn * A0)
        sol = solve_ivp(rhs, (tcur, tcur + tmax), A, method='DOP853', rtol=1e-11, atol=1e-13, dense_output=True)
        best = None
        ts = np.linspace(tcur, tcur + tmax, 3000)
        Ssol = sol.sol(ts)
        for ii in range(len(units)):
            for jj in range(ii + 1, len(units)):
                dist = np.abs(Ssol[ii] / nn[ii] - Ssol[jj] / nn[jj]) - DELTA
                idx = np.where(dist <= 0)[0]
                if len(idx) == 0 or idx[0] == 0:
                    continue
                g = lambda tt: abs(sol.sol(tt)[ii] / nn[ii] - sol.sol(tt)[jj] / nn[jj]) - DELTA
                te = brentq(g, ts[idx[0] - 1], ts[idx[0]], xtol=1e-13)
                if best is None or te < best[0]:
                    best = (te, ii, jj)
        if best is None:
            break
        te, ii, jj = best
        st = sol.sol(te)
        hist.append((te, units[ii][2], units[jj][2]))
        new = (st[ii] + st[jj], units[ii][1] + units[jj][1], units[ii][2] | units[jj][2])
        units = [(st[k], units[k][1], units[k][2]) for k in range(len(units)) if k not in (ii, jj)] + [new]
        tcur = te
    return hist


def fmt(h):
    return " -> ".join("%s+%s@%.4f" % (sorted(i + 1 for i in I), sorted(j + 1 for j in J), t) for t, I, J in h)


def seq(h):
    return [frozenset([I | J]) for _, I, J in h]


out = []
ex1 = np.array([1.0, 1.2, 1.8, 2.1])                     # chapter example
ex2 = np.array([1.0, 1.3, 1.9, 2.19])                    # gaps (0.30, 0.60, 0.29), D15-WISH s2(b)
for name, u in (("chapter example", ex1), ("gaps (0.30,0.60,0.29)", ex2)):
    out.append("[%s] full flow: %s" % (name, fmt(full_history(u))))
    for kap in (0.0, 1.0):
        ha, sola = null_history_N1(u, 'a', 1.0, kap, (1.0, 1.0))
        hb, solb = null_history_N1(u, 'b', 1.0, kap, (1.0, 1.0))
        out.append("[%s] kappa=%g  null (a), N1: %s" % (name, kap, fmt(ha)))
        out.append("[%s] kappa=%g  null (b), N1: %s" % (name, kap, fmt(hb)))
        # precision spread along the null at the first join time (relevant to N2 and to responses)
        t1 = ha[0][0]
        st = sola.sol(t1)
        out.append("[%s] kappa=%g  null (a): belief precision spread max|x_i - x_j| at t=%.3f: %.3e"
                   % (name, kap, t1, np.ptp(st[N:2 * N])))
    out.append("[%s] kappa=0, P(0)=I, T=1, null (b), N2 (parent-level null): %s" % (name, fmt(null_history_N2_k0_identity(u, 1.0))))

# precision separation under null (b) at kappa = 0 when P(0) != I
hb, solb = null_history_N1(ex1, 'b', 1.0, 0.0, (2.0, 2.0))
st = solb.sol(hb[0][0])
out.append("[chapter example] kappa=0, P(0)=2I, null (b): precision spread at first join %.3e (parents not Wishart)" % np.ptp(st[N:2 * N]))

# ---- map of the first-join order reversal under null (b), N1 ----
# family u = (1, 1 + d1, 1 + d1 + d2, 1 + d1 + d2 + d3), d1 = 0.30, d2 = 0.60; full flow flips at d3 = d1.


def first_join_gap(d3, conv, T, kap, P0):
    u = np.array([1.0, 1.3, 1.9, 1.9 + d3])
    x0 = np.full(N, P0[0]); y0 = np.full(N, P0[1])
    rhs = null_rhs_factory(conv, T, kap, u, x0, y0)

    def ev12(t, s): return abs(s[1] - s[0]) - DELTA
    def ev34(t, s): return abs(s[3] - s[2]) - DELTA
    def ev23(t, s): return abs(s[2] - s[1]) - DELTA
    for e in (ev12, ev34, ev23):
        e.terminal = False; e.direction = -1
    sol = solve_ivp(rhs, (0, 10.0), np.concatenate([u, x0, y0]), method='RK45', rtol=1e-9, atol=1e-11, events=(ev12, ev34, ev23))
    t12 = sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
    t34 = sol.t_events[1][0] if len(sol.t_events[1]) else np.inf
    t23 = sol.t_events[2][0] if len(sol.t_events[2]) else np.inf
    return t12 - t34, min(t12, t34) < t23


out.append("Map: null (b), N1; d1 = 0.30, d2 = 0.60, u1 = 1; the full flow joins {3,4} first iff d3 < 0.30.")
out.append("     entry = d3* at which the null first join switches from {1,2} to {3,4}; the reversal set is d3 in (d3*, 0.30)")
out.append("     (or (0.30, d3*)); width |d3* - 0.30| measures it. an entry 0.3000 means |d3* - 0.30| < 2e-5 (bisection resolution; reversal not resolved).")
Ts = [0.1, 0.3, 1.0, 3.0, 10.0]
P0s = [("I", (1.0, 1.0)), ("0.5I", (0.5, 0.5)), ("2I", (2.0, 2.0)), ("diag(e^.3,e^-.3)", (np.exp(0.3), np.exp(-0.3)))]
for kap in (0.0, 1.0, 5.0):
    out.append("  kappa = %g" % kap)
    header = "    P(0) \\ T      " + "".join("%10g" % T for T in Ts)
    out.append(header)
    for lab, P0 in P0s:
        row = "    %-16s" % lab
        for T in Ts:
            g_lo, _ = first_join_gap(0.20, 'b', T, kap, P0)
            g_hi, _ = first_join_gap(0.40, 'b', T, kap, P0)
            if np.sign(g_lo) == np.sign(g_hi):
                row += "%10s" % "none"
                continue
            d3s = brentq(lambda d: first_join_gap(d, 'b', T, kap, P0)[0], 0.20, 0.40, xtol=2e-5)
            row += "%10.4f" % d3s
        out.append(row)
for line in out:
    print(line)
