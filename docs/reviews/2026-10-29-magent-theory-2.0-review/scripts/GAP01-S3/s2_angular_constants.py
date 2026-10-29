r"""GAP01-S3 s2: explicit constants for pa:angular-grouping (chapters/13_per_agent_inheritance.tex:241-295).

Full flow (pa:angular-ode) in u = l/eps with A, C left unscaled:
    du_i/dt = -u_i + sum_{j!=i} beta_ij (L_j/eps) + o_i,  L_i = l_i + h(A_i+C_i) - h(A_i-C_i),
    dA_i/dt = -A_i + eps e(omega_i)   (exact: A_i(t) = e^{-t}A_i(0) + (1-e^{-t}) eps e(omega_i)),
    dC_i/dt = -C_i + beta_ii eps^3 e_1  (|C_i(t)| <= eps^3 if |C_i(0)| <= eps^3).
Rigorous bounds used (derived in the report):
    |L_i/eps - u_i| <= delta_bar := 2 eps^2 (Abar + eps^3),  Abar = max(A0, eps);
    row log-weights x_ii in [-J - J^2/4, J], x_ij in [-eps^2(Lam_i Lam_j/4 + Lam_j^2/8), eps^2 Lam_i Lam_j/4];
    |beta_ij - pi_ij| <= pi_ij (e^{osc_i} - 1);  |R_i| <= p (e^{osc_i}-1) sum_{j!=i}|u_j| + delta_bar.
Same Lyapunov/shadowing argument as s1; the selector reads L, so each margin also absorbs |n|_1 delta_bar/|n|_2.
Then floating-point spot checks of the exact field over random omega, A(0), C(0).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0e, i1e

A_PR, P_PR = 0.2, 0.4
o = np.array([1.0, 1.0, -1.0])
ustar = np.array([15 / 7, 15 / 7, 5 / 7])
uin = np.array([4.0, 0.5, -1.0])
e0 = uin - ustar
c1 = np.full(3, e0.mean()); cp = e0 - c1
NS = [np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [4, -1, 0], [-1, 4, 0], [0, 4, -1], [0, -1, 4])]

def Rbound(rho, eps, A0, crude=False):
    r"""Tight version: with c_i = KL(Bern(L_i)||Bern(0)), the reduced log-weights are
    x_ij - c_i = -KL(Bern(L_i)||Bern(L_j)) in [-(L_i-L_j)^2/8, 0] and
    x_ii - c_i = J t_i - log I_0(J) - KL(Bern(L_i)||Bern(0)) in [-J - J^2/4 - L_i^2/8, J],
    so beta_ij/pi_ij lies in [e^{-osc}, e^{osc}] with osc = eps^2 Delta_i^2/8 + 2J + J^2/4."""
    J = eps ** 3
    Abar = max(A0, eps)
    dbar = 2 * eps ** 2 * (Abar + eps ** 3)
    Rs = []
    for i in range(3):
        if crude:
            Lam = np.abs(ustar) + rho + dbar; Lmax = Lam.max()
            osc = max(J, eps ** 2 * Lam[i] * Lmax / 4) + max(J + J ** 2 / 4, eps ** 2 * (Lam[i] * Lmax / 4 + Lmax ** 2 / 8))
        else:
            Di = max([abs(ustar[i]) + rho + dbar] +
                     [abs(ustar[i] - ustar[j]) + np.sqrt(2) * rho + 2 * dbar for j in range(3) if j != i])
            osc = eps ** 2 * Di ** 2 / 8 + 2 * J + J ** 2 / 4
        so = sum(abs(ustar[j]) for j in range(3) if j != i) + np.sqrt(2) * rho
        Rs.append(P_PR * np.expm1(osc) * so + dbar)
    return float(np.sqrt(np.sum(np.square(Rs)))), dbar

def margin(n, t):
    ub = ustar + np.exp(-t / 5) * c1 + np.exp(-7 * t / 5) * cp
    return n @ ub / np.linalg.norm(n)

def entrance_time(r0, sh, dbar):
    T = 0.0
    for n in NS:
        extra = np.abs(n).sum() * dbar / np.linalg.norm(n)
        a1, ap = n @ c1, n @ cp
        if a1 <= 0 and ap <= 0:
            f = lambda t: margin(n, t) - (np.exp(-t / 5) * r0 + sh + extra)
            if f(200.0) <= 0:
                return np.inf
            if f(0.0) > 0:
                continue
            lo, hi = 0.0, 200.0
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(mid) <= 0 else (lo, mid)
            T = max(T, hi)
        else:
            lb = (n @ ustar + min(0, a1) + min(0, ap)) / np.linalg.norm(n)
            if lb <= r0 + sh + extra:
                return np.inf
    return T

print("(a) explicit constants, uniform over omega in T^3, |A_i(0)|<=A0, |C_i(0)|<=eps^3, any b_i(0)>0, theta")
for A0 in (0.05, 0.5):
    for r0 in (0.05, 0.2):
        rho0 = np.linalg.norm(e0) + r0
        print("  A0=%.2f r0=%.2f rho0=%.4f" % (A0, r0, rho0))
        for eps in (0.001, 0.005, 0.01, 0.02):
            Rb, db = Rbound(rho0, eps, A0)
            Rc, _ = Rbound(rho0, eps, A0, crude=True)
            inv = Rb < rho0 / 5
            T = entrance_time(r0, 5 * Rb, db)
            print("     eps=%.3f Rbar=%.3e (crude %.3e) delta_bar=%.3e ball invariant=%s  explicit T=%.4f"
                  % (eps, Rb, Rc, db, inv, T))
        def finite(e):
            Rb_, db_ = Rbound(rho0, e, A0)
            return Rb_ < rho0 / 5 and np.isfinite(entrance_time(r0, 5 * Rb_, db_))
        lo, hi = 1e-4, 0.2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if finite(mid) else (lo, mid)
        print("     sup of eps with a finite explicit T (bisection, Rbound increasing in eps): %.5f" % lo)

# (b) floating-point spot checks of the exact field ----------------------------------------
def logI0(r):
    return np.log(i0e(r)) + r

def grad_h(v):
    r = np.linalg.norm(v)
    if r < 1e-300:
        return np.zeros(2)
    return (i1e(r) / i0e(r)) * v / r

def state_quants(l, A, C, eps):
    J = eps ** 3
    L = np.empty(3); t = np.empty(3)
    for i in range(3):
        rp, rm = np.linalg.norm(A[i] + C[i]), np.linalg.norm(A[i] - C[i])
        L[i] = l[i] + logI0(rp) - logI0(rm)
        lp, lm = l[i] / 2 + logI0(rp), -l[i] / 2 + logI0(rm)
        mx = max(lp, lm); Pp, Pm = np.exp(lp - mx), np.exp(lm - mx); s = Pp + Pm
        t[i] = (Pp * grad_h(A[i] + C[i])[0] - Pm * grad_h(A[i] - C[i])[0]) / s
    m = np.tanh(L / 2)
    w = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            w[i, j] = A_PR * np.exp(J * t[i] - logI0(J)) if i == j else P_PR * np.exp(L[j] * m[i] / 2) / np.cosh(L[j] / 2)
    return L, w / w.sum(axis=1, keepdims=True)

def rhs(t, x, eps, om):
    l = x[0:3]; A = x[3:9].reshape(3, 2); C = x[9:15].reshape(3, 2)
    L, b = state_quants(l, A, C, eps)
    dl = np.array([-l[i] + sum(b[i, j] * L[j] for j in range(3) if j != i) + eps * o[i] for i in range(3)])
    dA = -A + eps * np.stack([np.cos(om), np.sin(om)], axis=1)
    dC = -C + np.outer(np.diag(b) * eps ** 3, [1.0, 0.0])
    return np.concatenate([dl, dA.ravel(), dC.ravel()])

def full_block(L):
    def edge(i, j):
        if L[i] * L[j] <= 0:
            return False
        return min(abs(L[i]), abs(L[j])) / max(abs(L[i]), abs(L[j])) > 0.25
    return [edge(0, 1), edge(0, 2), edge(1, 2)].count(True) >= 2

rng = np.random.default_rng(20261029)
tt = np.linspace(0, 40, 4001)
print("(b) exact field, random omega, |A(0)|<=0.05, |C(0)|<=eps^3, u(0) in B(u_in,0.05):")
for eps in (0.01, 0.05, 0.2):
    worst = 0.0; ok = True
    for trial in range(6):
        om = rng.uniform(0, 2 * np.pi, 3)
        A0 = rng.normal(size=(3, 2)); A0 *= (0.05 * rng.uniform(0, 1, (3, 1))) / np.linalg.norm(A0, axis=1, keepdims=True)
        C0 = rng.normal(size=(3, 2)); C0 *= (eps ** 3 * rng.uniform(0, 1, (3, 1))) / np.linalg.norm(C0, axis=1, keepdims=True)
        du = rng.normal(size=3); du *= 0.05 * rng.uniform() / np.linalg.norm(du)
        x0 = np.concatenate([eps * (uin + du), A0.ravel(), C0.ravel()])
        sol = solve_ivp(rhs, (0, 40), x0, t_eval=tt, args=(eps, om), method="DOP853", rtol=1e-10, atol=1e-13)
        fb = []
        for k in range(len(tt)):
            x = sol.y[:, k]
            L, _ = state_quants(x[0:3], x[3:9].reshape(3, 2), x[9:15].reshape(3, 2), eps)
            fb.append(full_block(L))
        fb = np.array(fb)
        if fb[0]:
            ok = False
        last = tt[np.where(~fb)[0].max()]
        ok = ok and bool(fb[tt > last].all())
        worst = max(worst, last)
    print("   eps=%.2f: 6 random records/amplitudes; singletons at 0 and persistent full block: %s; latest entrance %.3f"
          % (eps, ok, worst))
