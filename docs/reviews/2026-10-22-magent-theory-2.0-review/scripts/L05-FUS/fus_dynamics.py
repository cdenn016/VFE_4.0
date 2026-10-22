r"""L05-FUS: numerical checks of the fusion dynamics (chapters/06_fusion.tex).

Part A: event-driven exact solution of the block equations in the ORIGINAL time t
        (closed form u_I(t) = f_I/kappa + (u_I0 - f_I/kappa) e^{-kappa t} between events,
        f_I = lambda (M_right - M_left)); does not use the tau time change.
        Compared with fus:collision-times and the open-gap formula tau_final.
Part B: smoothed full 8-dimensional two-sector system, |x| ~ sqrt(x^2+eps^2), stiff Radau
        integration; first-contact times per sector compared with Part A; v(t) check.
Part C: perturbation OFF the common-offset class (numerical observation only).
Part D: two-agent single-sector transversal crossing (contact without sticking).
Part E: random initial data, all masses one: time to complete synchronization.
Single-threaded; runs in a few seconds.
"""
import numpy as np
from scipy.integrate import solve_ivp


def event_driven(u0, kappa, lam, masses=None, tmax=50.0):
    """Exact block dynamics u' = -kappa u + lam (M_right - M_left) with sticking."""
    order = np.argsort(u0)
    pos = list(np.asarray(u0, float)[order])
    mass = list(np.ones(len(u0)) if masses is None else np.asarray(masses, float)[order])
    members = [[int(k)] for k in order]
    t = 0.0
    events = []
    while len(pos) > 1:
        N = sum(mass)
        left = np.cumsum([0.0] + mass[:-1])
        right = N - left - np.array(mass)
        f = lam * (right - left)
        # gap between block k and k+1: g(s) = -F + (g0 + F) e^{-kappa s}, F = (f_k - f_{k+1})/kappa > 0
        best, kbest = np.inf, None
        for k in range(len(pos) - 1):
            g0 = pos[k + 1] - pos[k]
            F = (f[k] - f[k + 1]) / kappa
            s = np.log((g0 + F) / F) / kappa
            if s < best - 1e-14:
                best, kbest = s, [k]
            elif abs(s - best) <= 1e-14:
                kbest.append(k)
        s = best
        pos = [f[k] / kappa + (pos[k] - f[k] / kappa) * np.exp(-kappa * s) for k in range(len(pos))]
        t += s
        for k in sorted(kbest, reverse=True):
            mtot = mass[k] + mass[k + 1]
            pos[k] = pos[k]  # equal positions at contact
            mass[k] = mtot
            members[k] = members[k] + members[k + 1]
            del pos[k + 1], mass[k + 1], members[k + 1]
        events.append((t, [sorted(m) for m in members]))
    return events


a, c, lam, d, v0 = 1.0, 0.5, 0.3, 1.0, 0.7
kappa = a - c
print("=== Part A: event-driven exact solution, original time ===")
for (a_, c_, lam_, d_) in [(1.0, 0.5, 0.3, 1.0), (2.0, 1.5, 1.0, 0.5)]:
    k_ = a_ - c_
    ev = event_driven(np.array([-3, -2, 2, 3]) * d_, k_, lam_)
    t1 = np.log(1 + k_ * d_ / (2 * lam_)) / k_
    t2 = np.log(1 + 5 * k_ * d_ / (4 * lam_)) / k_
    print(f"a={a_},c={c_},lam={lam_},d={d_}: events {[(round(e[0], 12), e[1]) for e in ev]}")
    print(f"   t1 formula {t1:.12f}, t2 formula {t2:.12f}; residuals {ev[0][0]-t1:.2e}, {ev[-1][0]-t2:.2e}")

for gaps in [(0.7, 2.9, 1.1), (1.3, 3.0, 0.4), (1.0, 1.2, 1.3)]:
    g1, g2, g3 = gaps
    H = max(g1, g3)
    D = g2 + (g1 + g3) / 2 - 2 * H
    u0 = np.cumsum([0.0, g1, g2, g3]) - 1.7
    ev = event_driven(u0, kappa, lam)
    tau_final = (2 * g2 + g1 + g3) / (8 * lam)
    t_final = np.log(1 + kappa * tau_final) / kappa
    print(f"gaps {gaps}: D={D:+.3f}; event sequence {[e[1] for e in ev]}; final t {ev[-1][0]:.12f}",
          f"vs formula {t_final:.12f} (formula applies only if D>0)")

print("\n=== Part B: smoothed full two-sector system (eps-regularized sign) ===")


def rhs_factory(a, c, lam, eps, n):
    def rhs(t, y):
        mq, ms = y[:n], y[n:]
        dq = mq[:, None] - mq[None, :]
        ds = ms[:, None] - ms[None, :]
        sq = (dq / np.sqrt(dq**2 + eps**2)).sum(axis=1)
        ss = (ds / np.sqrt(ds**2 + eps**2)).sum(axis=1)
        return np.concatenate([-a * mq + c * ms - lam * sq, -a * ms + c * mq - lam * ss])
    return rhs


def first_contacts(sol, n, thr, tgrid):
    Y = sol.sol(tgrid)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            for sec, off in (("q", 0), ("s", n)):
                g = np.abs(Y[off + i] - Y[off + j])
                idx = np.nonzero(g < thr)[0]
                out[(i, j, sec)] = tgrid[idx[0]] if len(idx) else np.inf
    return out, Y


eps = 1e-6
n = 4
u0 = np.array([-3, -2, 2, 3]) * d
y0 = np.concatenate([u0 + v0, u0 - v0])
T = 4.0
sol = solve_ivp(rhs_factory(a, c, lam, eps, n), (0, T), y0, method="Radau", rtol=1e-10, atol=1e-12,
                dense_output=True)
tgrid = np.linspace(0, T, 400001)
fc, Y = first_contacts(sol, n, 50 * eps, tgrid)
t1 = np.log(1 + kappa * d / (2 * lam)) / kappa
t2 = np.log(1 + 5 * kappa * d / (4 * lam)) / kappa
print("nfev", sol.nfev, "status", sol.status)
for key in [(0, 1, "q"), (0, 1, "s"), (2, 3, "q"), (2, 3, "s"), (1, 2, "q"), (1, 2, "s")]:
    print("  first contact", key, f"{fc[key]:.5f}")
print(f"  formulas: t1={t1:.5f}, t2={t2:.5f}")
vt = (Y[:n] - Y[n:]) / 2
vexact = v0 * np.exp(-(a + c) * tgrid)
print("  max |v_i(t) - v0 e^{-(a+c)t}| over agents and grid:", np.abs(vt - vexact[None, :]).max())
print("  max |mu(T)| (u should be 0 after t2):", np.abs((Y[:n, -1] + Y[n:, -1]) / 2).max())

print("\n=== Part C: perturbation off the common-offset class (numerical observation) ===")
rng = np.random.default_rng(1)
for trial in range(3):
    dq = rng.normal(scale=0.05, size=n)
    ds = rng.normal(scale=0.05, size=n)
    y0p = np.concatenate([u0 + v0 + dq, u0 - v0 + ds])
    solp = solve_ivp(rhs_factory(a, c, lam, eps, n), (0, T), y0p, method="Radau", rtol=1e-10, atol=1e-12,
                     dense_output=True)
    fcp, Yp = first_contacts(solp, n, 50 * eps, tgrid)
    seq = sorted([(v, k) for k, v in fcp.items() if np.isfinite(v)])
    print(f" trial {trial}: contact order (time, pair, sector):",
          [(round(v, 4), k) for v, k in seq[:8]])

print("\n=== Part D: two agents, contact in one sector without sticking ===")
y0d = np.array([0.1, 0.0, -5.0, 0.0])   # (x1, x2, y1, y2): dx(0)=0.1, dy(0)=-5
sold = solve_ivp(rhs_factory(a, c, lam, eps, 2), (0, 12), y0d, method="Radau", rtol=1e-10, atol=1e-12,
                 dense_output=True)
tg = np.linspace(0, 12, 120001)
Yd = sold.sol(tg)
dx = Yd[0] - Yd[1]
dy = Yd[2] - Yd[3]
signchg = np.nonzero(np.diff(np.sign(dx)))[0]
print("  |c dy(0)| =", abs(c * -5.0), "> 2 lam =", 2 * lam)
print("  q-gap sign changes at t ~", tg[signchg][:5], "; dx just after first crossing:", dx[signchg[0] + 50])
thr = 50 * eps
def settle_time(g):
    above = np.nonzero(np.abs(g) >= thr)[0]
    return tg[above[-1] + 1] if len(above) and above[-1] + 1 < len(tg) else None
win = (tg > 0.2) & (tg < 2.0)
print("  max dx on [0.2, 2.0] (negative => crossed and separated):", dx[win].max(),
      "; min dx there:", dx[win].min())
print("  q-gap settles (|dx|<50eps thereafter) at t ~", settle_time(dx),
      "; s-gap settles at t ~", settle_time(dy))

print("\n=== Part E: random initial data, time to complete synchronization (both sectors) ===")
for trial in range(3):
    y0r = rng.normal(scale=2.0, size=2 * n)
    solr = solve_ivp(rhs_factory(a, c, lam, eps, n), (0, 15), y0r, method="Radau", rtol=1e-9, atol=1e-12,
                     dense_output=True)
    tg2 = np.linspace(0, 15, 150001)
    Yr = solr.sol(tg2)
    spread = np.maximum(Yr[:n].max(0) - Yr[:n].min(0), Yr[n:].max(0) - Yr[n:].min(0))
    idx = np.nonzero(spread < 100 * eps)[0]
    print(f" trial {trial}: complete synchronization (spread<1e-4) from t ~ {tg2[idx[0]] if len(idx) else None};"
          f" |mu| at that time {np.abs(Yr[:, idx[0]]).max() if len(idx) else None}")
