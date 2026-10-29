r"""XC7-IMPL s2: finite conformance checks for chapters 13 and 14 (per-agent model).

(a) pa:four-linear: Jacobian of the exact two-agent four-state per-agent flow at the
    record-free uniform table (Gibbs rows at unit temperature), by central differences,
    versus the claimed spectrum {1/4,-3/4,-5/4,-9/4} U {-1+-kappa/2 (x4)}.
(b) pa:bit-limit: equilibrium of the exact Bernoulli per-agent flow with Gibbs rows,
    u_eps = l_eps/eps, against u_* = (15/7,15/7,5/7); the gap should scale as eps^2.
(c) pa:angular-flow and lp:field: quadrature on {+-1} x S^1 x S^1; the centered
    own-law Fisher velocity of agent 1 is projected on the family span and compared
    with the closed-form ODE coefficients.
(d) lp:sum-failure: symbolic Omega and dOmega/du.
(e) lp:causal-flow: the causal slice/phase system reproduces the native solution
    rotated by the frozen initial local frame (lp:frozen-local-initial-frame).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0, i1
import sympy as sp

np.set_printoptions(precision=6, suppress=True)

# ---------------------------------------------------------------- (a)
J, kap, alpha = 1.5, 0.3, 0.3   # kappa=0.3 separates -1+-kappa/2 from the sign eigenvalues
rho_q, rho_s = np.cosh(J) / (1 + np.cosh(J)), 0.5
A = np.array([1, 1, -1, -1]); X = np.array([1, -1, 1, -1])      # states (a,x) order
W = np.zeros((4, 4))   # W[(a,x'),(a,x)] = (1+kappa a x' x)/2 if same a else 0
for s1 in range(4):
    for s2_ in range(4):
        if A[s1] == A[s2_]:
            W[s1, s2_] = (1 + kap * A[s1] * X[s1] * X[s2_]) / 2

def tables(theta):
    t = theta.reshape(2, 2, 4)        # agent, sector(q,s), state
    p = np.exp(t - t.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)

def kl(p, r):
    return np.sum(p * np.log(p / r))

def field4(theta, Hq=(0.0, 0.0), Hs=(0.0, 0.0)):
    P = tables(theta)
    out = np.zeros_like(P)
    for i in range(2):
        j = 1 - i
        q, s = P[i, 0], P[i, 1]
        m_i, n_i = np.sum(q * A), np.sum(s * A)
        nu_q, nu_s = P[j, 0] @ W, P[j, 1] @ W
        Dq_ii = np.sum(q * np.log(q)) - J * m_i * n_i + np.log(4 * np.cosh(J))
        Dq_ij = kl(q, nu_q)
        Ds_ii = np.sum(s * np.log(s)) + np.log(4)
        Ds_ij = kl(s, nu_s)
        wb = np.array([rho_q * np.exp(-Dq_ii), (1 - rho_q) * np.exp(-Dq_ij)]); b = wb / wb.sum()
        wg = np.array([rho_s * np.exp(-Ds_ii), (1 - rho_s) * np.exp(-Ds_ij)]); g = wg / wg.sum()
        hq = np.log(q) - b[1] * np.log(nu_q) - b[0] * J * n_i * A - Hq[i] * A
        hs = np.log(s) - g[1] * np.log(nu_s) - b[0] * J * m_i * A - Hs[i] * A
        out[i, 0] = -(hq - np.sum(q * hq))
        out[i, 1] = -(hs - np.sum(s * hs))
    return out.reshape(-1)          # d/dt log u, centered (sum_z u_z * rate = 0)

# Basis of sum-zero log coordinates per table; coordinates are log-probabilities up to a constant.
E = np.eye(4) - 0.25
Bt = np.linalg.svd(E)[0][:, :3]                       # 4x3 orthonormal basis of sum-zero vectors
def embed(c):     # 12 reduced coords -> 16 log-table coords
    return np.concatenate([Bt @ c[3 * k:3 * k + 3] for k in range(4)])
def reduce(v):    # project log-velocity (centered under u, not under uniform) to sum-zero coords
    return np.concatenate([Bt.T @ (v[4 * k:4 * k + 4] - v[4 * k:4 * k + 4].mean()) for k in range(4)])
h = 1e-6
Jac = np.zeros((12, 12))
for k in range(12):
    e = np.zeros(12); e[k] = h
    Jac[:, k] = (reduce(field4(embed(e))) - reduce(field4(embed(-e)))) / (2 * h)
ev = np.sort(np.linalg.eigvals(Jac).real)
claimed = np.sort([0.25, -0.75, -1.25, -2.25] + [-1 + kap / 2] * 4 + [-1 - kap / 2] * 4)
print("(a) pa:four-linear eigenvalues, numerical:", ev)
print("    claimed:", claimed, " max |diff| =", np.abs(ev - claimed).max())

# ---------------------------------------------------------------- (b)
def kl_ber(li, lj):
    pi, pj = 1 / (1 + np.exp(-li)), 1 / (1 + np.exp(-lj))
    return pi * np.log(pi / pj) + (1 - pi) * np.log((1 - pi) / (1 - pj))
def bit_field(t, l, eps, o=np.array([1., 1., -1.]), a=0.2, p=0.4):
    dl = np.zeros(3)
    for i in range(3):
        costs = np.array([kl_ber(l[i], 0.0) if j == i else kl_ber(l[i], l[j]) for j in range(3)])
        prior = np.array([a if j == i else p for j in range(3)])
        w = prior * np.exp(-costs); b = w / w.sum()
        dl[i] = -l[i] + sum(b[j] * l[j] for j in range(3) if j != i) + eps * o[i]
    return dl
ustar = np.array([15 / 7, 15 / 7, 5 / 7])
gaps = []
for eps in (0.2, 0.1, 0.05, 0.025):
    sol = solve_ivp(bit_field, (0, 120), eps * np.array([4, 0.5, -1]), args=(eps,), rtol=1e-12, atol=1e-14)
    ueps = sol.y[:, -1] / eps
    gaps.append(np.abs(ueps - ustar).max())
print("(b) pa:bit-limit |u_eps-u_*|, eps=0.2,0.1,0.05,0.025:", np.array(gaps),
      " ratios:", np.array(gaps[:-1]) / np.array(gaps[1:]))

# ---------------------------------------------------------------- (c)
Nphi = 64
phi = 2 * np.pi * np.arange(Nphi) / Nphi
Z, PH, PS = np.meshgrid(np.array([1., -1.]), phi, phi, indexing='ij')
wref = np.full(Z.shape, 1.0 / Z.size)          # uniform sign x Haar x Haar reference
def e(th): return np.stack([np.cos(th), np.sin(th)])
def logq_family(l, Avec, Cvec, Bvec):
    lq = l * Z / 2 + Avec[0] * np.cos(PH) + Avec[1] * np.sin(PH) + Z * (Cvec[0] * np.cos(PH) + Cvec[1] * np.sin(PH)) \
         + Bvec[0] * np.cos(PS) + Bvec[1] * np.sin(PS)
    return lq - np.log(np.sum(wref * np.exp(lq)))
def project(v, q):
    feats = [np.ones_like(Z), Z, np.cos(PH), np.sin(PH), Z * np.cos(PH), Z * np.sin(PH), np.cos(PS), np.sin(PS)]
    Fm = np.stack([f.ravel() for f in feats], 1); wq = (wref * q).ravel()
    G = Fm.T @ (Fm * wq[:, None]); coef = np.linalg.solve(G, Fm.T @ (wq * v.ravel()))
    res = v.ravel() - Fm @ coef
    return coef, np.sum(wq * res ** 2) / np.sum(wq * v.ravel() ** 2)
def hfun(Avec): return np.log(i0(np.linalg.norm(Avec)))
rng = np.random.default_rng(7)
eps = 0.3; Jc = eps ** 3; a_self, p_peer = 0.2, 0.4
agents = [dict(l=rng.normal(), A=rng.normal(size=2) * 0.5, C=rng.normal(size=2) * 0.2, B=rng.normal(size=2) * 0.7)
          for _ in range(3)]
Hrec, Lam = 0.4, np.array([0.3, -0.2])
for model in ("pa:angular-flow (circles resampled)", "lp:field (psi transmitted)"):
    lq = [logq_family(ag['l'], ag['A'], ag['C'], ag['B']) for ag in agents]
    L = [ag['l'] + hfun(ag['A'] + ag['C']) - hfun(ag['A'] - ag['C']) for ag in agents]
    q1 = np.exp(lq[0])
    # costs of agent 1 (index 0): self evaluator G = e^{J z cos phi}/I0(J); peer channel outputs
    logG = Jc * Z * np.cos(PH) - np.log(i0(Jc))
    lognu = []
    for j in (1, 2):
        base = L[j] * Z / 2 - np.log(np.cosh(L[j] / 2))
        if model.startswith("lp"):
            Bj = agents[j]['B']
            base = base + Bj[0] * np.cos(PS) + Bj[1] * np.sin(PS) - np.log(i0(np.linalg.norm(Bj)))
        lognu.append(base)
    S1 = np.sum(wref * q1 * lq[0])
    D = [S1 - np.sum(wref * q1 * logG)] + [S1 - np.sum(wref * q1 * ln) for ln in lognu]
    wts = np.array([a_self, p_peer, p_peer]) * np.exp(-np.array(D)); beta = wts / wts.sum()
    loglik = Hrec * Z / 2 + Lam[0] * np.cos(PH) + Lam[1] * np.sin(PH)
    phi1 = lq[0] - beta[0] * logG - beta[1] * lognu[0] - beta[2] * lognu[1] - loglik
    vel = -(phi1 - np.sum(wref * q1 * phi1))
    coef, frac = project(vel, q1)
    # closed form: l-dot, A-dot, C-dot, B-dot coefficients (coef[1] multiplies z = (l-dot)/2)
    ag = agents[0]
    ldot = -ag['l'] + beta[1] * L[1] + beta[2] * L[2] + Hrec
    Adot = -ag['A'] + Lam
    Cdot = -ag['C'] + beta[0] * Jc * np.array([1., 0.])
    Bdot = -ag['B'] + ((beta[1] * agents[1]['B'] + beta[2] * agents[2]['B']) if model.startswith("lp") else 0)
    closed = np.concatenate([[ldot / 2], Adot, Cdot, Bdot])
    # check weights against lp:weights / pa:angular-weights closed forms
    m1 = np.tanh(L[0] / 2); t1 = np.sum(wref * q1 * Z * np.cos(PH))
    Mb = (i1(np.linalg.norm(ag['B'])) / i0(np.linalg.norm(ag['B']))) * ag['B'] / np.linalg.norm(ag['B'])
    wcf = [a_self * np.exp(Jc * t1) / i0(Jc)]
    for j in (1, 2):
        extra = (agents[j]['B'] @ Mb - np.log(i0(np.linalg.norm(agents[j]['B'])))) if model.startswith("lp") else 0
        wcf.append(p_peer * np.exp(L[j] * m1 / 2 + extra) / np.cosh(L[j] / 2))
    bcf = np.array(wcf) / np.sum(wcf)
    print(f"(c) {model}: residual outside family span {frac:.2e}; "
          f"max|ODE coef - projection| {np.abs(closed - coef[1:]).max():.2e}; max|rows - closed form| {np.abs(bcf - beta).max():.2e}")

# ---------------------------------------------------------------- (d)
s_, k_, E_, u_ = sp.symbols('s k E u', positive=True)
Dd = s_ + k_ * E_ + k_ * u_
Om = sp.Rational(1, 5) * ((4 * k_ * u_ - 2 * k_ * E_) / Dd - 2 * k_ / (s_ + 2 * k_))
print("(d) lp:sum-failure dOmega/du - (4ks+6k^2E)/(5D^2) =",
      sp.simplify(sp.diff(Om, u_) - (4 * k_ * s_ + 6 * k_ ** 2 * E_) / (5 * Dd ** 2)))
b_, v_ = 0.7, 0.9
Jt = 1e-3
sv, kv, Ev, uv = 1 / (5 * i0(Jt)), 2 / (5 * i0(b_)), np.exp(b_ * i1(b_) / i0(b_)), 1 / np.cosh(v_ / 2)
Bs = [b_ * np.array([1., 0.]), b_ * np.array([0., 1.]), b_ * np.array([1., 0.])]
Ls = [0.0, v_, 0.0]
def Mvec(B): n = np.linalg.norm(B); return (i1(n) / i0(n)) * B / n
VB = []
for i in range(3):
    w = []
    for j in range(3):
        if j == i: w.append(1 / (5 * i0(Jt)))       # t_i = 0 when A = C = 0
        else: w.append(2 * np.exp(Ls[j] * np.tanh(Ls[i] / 2) / 2 + Bs[j] @ Mvec(Bs[i])) / (5 * np.cosh(Ls[j] / 2) * i0(b_)))
    bb = np.array(w) / np.sum(w)
    VB.append(-Bs[i] + sum(bb[j] * Bs[j] for j in range(3) if j != i))
S = sum(Bs); VS = sum(VB)
Om_num = (S[0] * VS[1] - S[1] * VS[0]) / (S @ S)
Om_cf = float(Om.subs({s_: sv, k_: kv, E_: Ev, u_: uv}))
print(f"    Omega from rows {Om_num:.15f} vs closed form {Om_cf:.15f}")

# ---------------------------------------------------------------- (e)
J2 = np.array([[0., -1.], [1., 0.]])
def rot(th): return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
def mean_cos(Av):
    n = np.linalg.norm(Av)
    return (i1(n) / i0(n)) * Av[0] / n if n > 0 else 0.0
def native(t, xv, H=np.array([0.3, 0.3, -0.3]), Jc=1e-3):
    x = xv.reshape(3, 8)          # l, A(2), C(2), B(2), n
    out = np.zeros_like(x)
    L = [x[i, 0] + hfun(x[i, 1:3] + x[i, 3:5]) - hfun(x[i, 1:3] - x[i, 3:5]) for i in range(3)]
    for i in range(3):
        m_i = np.tanh(L[i] / 2)
        Ap, Am = x[i, 1:3] + x[i, 3:5], x[i, 1:3] - x[i, 3:5]
        wp = np.exp(x[i, 0] / 2) * i0(np.linalg.norm(Ap)); wm = np.exp(-x[i, 0] / 2) * i0(np.linalg.norm(Am))
        t_i = (wp * mean_cos(Ap) - wm * mean_cos(Am)) / (wp + wm)
        w = []
        for j in range(3):
            if j == i: w.append(np.exp(Jc * t_i) / (5 * i0(Jc)))
            else: w.append(2 * np.exp(L[j] * m_i / 2 + x[j, 5:7] @ Mvec(x[i, 5:7]))
                           / (5 * np.cosh(L[j] / 2) * i0(np.linalg.norm(x[j, 5:7]))))
        bb = np.array(w) / np.sum(w)
        out[i, 0] = -x[i, 0] + sum(bb[j] * L[j] for j in range(3) if j != i) + H[i]
        out[i, 1:3] = -x[i, 1:3]
        out[i, 3:5] = -x[i, 3:5] + bb[i] * Jc * np.array([1., 0.])
        out[i, 5:7] = -x[i, 5:7] + sum(bb[j] * x[j, 5:7] for j in range(3) if j != i)
        out[i, 7] = -x[i, 7]          # presentation rows omitted: decoupled from the B equivariance check
    return out.reshape(-1)
def Rb(xv, Umat):
    x = xv.reshape(3, 8).copy(); x[:, 5:7] = x[:, 5:7] @ Umat.T; return x.reshape(-1)
def Ucan(S): return np.array([[S[0], S[1]], [-S[1], S[0]]]) / np.linalg.norm(S)
def causal(t, y):
    xb = y[:-1]
    V = native(t, xb); x = xb.reshape(3, 8); Vx = V.reshape(3, 8)
    S = x[:, 5:7].sum(0); VS = Vx[:, 5:7].sum(0)
    om = (S[0] * VS[1] - S[1] * VS[0]) / (S @ S)
    corr = np.zeros_like(x); corr[:, 5:7] = x[:, 5:7] @ J2.T
    return np.concatenate([V - om * corr.reshape(-1), [om]])
x0 = np.zeros((3, 8))
x0[:, 0] = [1.2, 0.15, -0.3]; x0[:, 1:3] = rng.normal(size=(3, 2)) * 0.3
x0[:, 5:7] = np.array([[0.5, 0.2], [0.3, 0.6], [0.6, 0.45]]); x0[:, 7] = 0.1
x0 = x0.reshape(-1)
T = 3.0
nat = solve_ivp(native, (0, T), x0, rtol=1e-11, atol=1e-13, dense_output=True)
U0 = Ucan(x0.reshape(3, 8)[:, 5:7].sum(0))
cau = solve_ivp(causal, (0, T), np.concatenate([Rb(x0, U0), [0.0]]), rtol=1e-11, atol=1e-13, dense_output=True)
errs = []
for tt in np.linspace(0, T, 7):
    xb = cau.sol(tt)[:-1]; th = cau.sol(tt)[-1]
    errs.append(np.abs(Rb(xb, rot(th)) - Rb(nat.sol(tt), U0)).max())
xbT = cau.sol(T)[:-1].reshape(3, 8)
print(f"(e) lp:causal-flow max|R_theta xbar - R_U0 x| over [0,{T}] = {max(errs):.2e}; "
      f"slice constraint S_y(xbar(T)) = {xbT[:, 6].sum():.2e}, theta(T) = {cau.sol(T)[-1]:.4f}")

# ---------------------------------------------------------------- (b2) pa:persistent-grouping events at eps = 0.05
def null_field(t, l, eps, o=np.array([1., 1., -1.]), a=0.2, p=0.4):
    dl = np.zeros(3)
    for i in range(3):
        D0 = kl_ber(l[i], 0.0); b0 = a * np.exp(-D0) / (a * np.exp(-D0) + 2 * p)
        dl[i] = -b0 * l[i] + eps * o[i]
    return dl
def blocks(l):
    n = len(l); adj = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.sign(l[i]) == np.sign(l[j]) != 0 and min(abs(l[i]), abs(l[j])) / max(abs(l[i]), abs(l[j])) > 0.25:
                adj[i, j] = adj[j, i] = True
    lab = list(range(n))
    for _ in range(n):
        for i in range(n):
            for j in range(n):
                if adj[i, j]: lab[i] = lab[j] = min(lab[i], lab[j])
    return len(set(lab))
eps = 0.05; ts = np.linspace(0, 60, 6001)
full = solve_ivp(bit_field, (0, 60), eps * np.array([4, 0.5, -1]), args=(eps,), t_eval=ts, rtol=1e-10, atol=1e-13)
null = solve_ivp(null_field, (0, 60), eps * np.array([4, 0.5, -1]), args=(eps,), t_eval=ts, rtol=1e-10, atol=1e-13)
nb_full = np.array([blocks(full.y[:, k]) for k in range(len(ts))]); nb_null = np.array([blocks(null.y[:, k]) for k in range(len(ts))])
first = ts[np.argmax(nb_full == 1)] if (nb_full == 1).any() else None
print(f"(b2) eps={eps}: blocks at t=0 full/null = {nb_full[0]}/{nb_null[0]}; first time full flow has one block: {first}; "
      f"one block on [first, 60]: {bool((nb_full[ts >= first] == 1).all())}; null ever one block: {bool((nb_null == 1).any())}; "
      f"final full u = {full.y[:, -1] / eps}")
