r"""GAP01-S3 s4: explicit radius for pa:four-grouping (chapters/13_per_agent_inheritance.tex:96-126).

Chart: centered log tables x = (f_1, f_2, g_1, g_2), each in R^4_0 over (a,x) or (b,u) in {+-1}^2,
norm ||x||^2 = sum_blocks (1/4) sum_z v_z^2.  Exact Gibbs-reduced per-agent field (pa:four-potentials):
   f_i' = -f_i + beta_ij lnu_j + (beta_ii J n_i + H^q_i) a,   g_i' = -g_i + gamma_ij lsig_j + (beta_ii J m_i + H^s_i) b,
lnu_j = centered log(softmax(f_j) W), lsig_j = centered log(softmax(g_j) W), W_b(y|x) = (1 + kappa b x y)/2.
Rigorous ingredients (derived in the report): A = DV(0) is symmetric in this norm with top eigenvalue 1/4, so
||e^{At}|| = e^{t/4}; remainder bound ||N(x)|| <= 2 max_block Nb(R), R = 2||x||, with Nb from
   ||lnu - fW||_inf <= R^2/2, |n - <b,g>| <= R^2, |beta_ij - 1/2| <= X(R)/4, X = R^2/2 + J R^2 + (2R + R^2/2)^2/2.
Grönwall/Volterra bootstrap for e = x - e^{At}x(0) gives an explicit (eta, h, r0) for which every initial table
within r0 of the product-family point m = n = eta(3,-1) selects the pair at t* = log 4 and on an explicit window,
while the null never selects.  Then a floating-point radius test with the exact flow.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

J = 1.5; KAP = 0.5
RHO_Q = np.cosh(J) / (1 + np.cosh(J)); RHO_S = 0.5
av = np.array([1.0, 1.0, -1.0, -1.0]); xv = np.array([1.0, -1.0, 1.0, -1.0])
Wm = np.zeros((4, 4))  # (qW)[k'] = sum_k q[k] Wm[k,k'] ; k=(b,x) -> k'=(b,y)
for k in range(4):
    for kp in range(4):
        if av[k] == av[kp]:
            Wm[k, kp] = (1 + KAP * av[k] * xv[k] * xv[kp]) / 2

def softmax(f):
    e = np.exp(f - f.max()); return e / e.sum()

def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))

def field(x, H):
    f1, f2, g1, g2 = x[0:4], x[4:8], x[8:12], x[12:16]
    q = [softmax(f1), softmax(f2)]; s = [softmax(g1), softmax(g2)]
    m = [q[i] @ av for i in range(2)]; n = [s[i] @ av for i in range(2)]
    out = []
    betas = []
    for i in range(2):
        j = 1 - i
        nuj = q[j] @ Wm
        D11 = float(np.sum(q[i] * np.log(q[i]))) - J * m[i] * n[i] + np.log(4 * np.cosh(J))
        D12 = kl(q[i], nuj)
        w = np.array([RHO_Q * np.exp(-D11), (1 - RHO_Q) * np.exp(-D12)]); b = w / w.sum()
        betas.append(b)
    for i in range(2):
        j = 1 - i
        lnu = np.log(q[j] @ Wm); lnu -= lnu.mean()
        out.append(-x[4 * i:4 * i + 4] + betas[i][1] * lnu + (betas[i][0] * J * n[i] + H[0][i]) * av)
    for i in range(2):
        j = 1 - i
        sj = s[j] @ Wm
        E11 = kl(s[i], np.full(4, 0.25)); E12 = kl(s[i], sj)
        w = np.array([RHO_S * np.exp(-E11), (1 - RHO_S) * np.exp(-E12)]); gm = w / w.sum()
        lsig = np.log(sj); lsig -= lsig.mean()
        out.append(-x[8 + 4 * i:12 + 4 * i] + gm[1] * lsig + (betas[i][0] * J * m[i] + H[1][i]) * av)
    return np.concatenate(out)

def null_field(x, H):
    out = []
    q = [softmax(x[0:4]), softmax(x[4:8])]; s = [softmax(x[8:12]), softmax(x[12:16])]
    m = [q[i] @ av for i in range(2)]; n = [s[i] @ av for i in range(2)]
    b0 = []
    for i in range(2):
        D11 = float(np.sum(q[i] * np.log(q[i]))) - J * m[i] * n[i] + np.log(4 * np.cosh(J))
        b0.append(RHO_Q * np.exp(-D11) / (RHO_Q * np.exp(-D11) + 1 - RHO_Q))
    for i in range(2):
        out.append(-b0[i] * x[4 * i:4 * i + 4] + (b0[i] * J * n[i] + H[0][i]) * av)
    for i in range(2):
        E11 = kl(s[i], np.full(4, 0.25))
        g0 = RHO_S * np.exp(-E11) / (RHO_S * np.exp(-E11) + 1 - RHO_S)
        out.append(-g0 * x[8 + 4 * i:12 + 4 * i] + (b0[i] * J * m[i] + H[1][i]) * av)
    return np.concatenate(out)

def cond_means(x):
    res = []
    for blk in range(4):
        p = softmax(x[4 * blk:4 * blk + 4])
        for xx in (1.0, -1.0):
            ip = [k for k in range(4) if xv[k] == xx and av[k] > 0][0]; im = [k for k in range(4) if xv[k] == xx and av[k] < 0][0]
            res.append((p[ip] - p[im]) / (p[ip] + p[im]))
    return np.array(res)

# basis of the centered subspace orthonormal in the u-norm: columns 2*e/... ; use Hadamard-type basis
Bq = np.stack([av, xv, av * xv], 1)  # orthonormal in u-norm (each has mean-square 1)
Bfull = np.zeros((16, 12))
for blk in range(4):
    Bfull[4 * blk:4 * blk + 4, 3 * blk:3 * blk + 3] = Bq
H0 = [[0.0, 0.0], [0.0, 0.0]]
Amat = np.zeros((12, 12)); hstep = 1e-6
for k in range(12):
    dx = Bfull[:, k] * hstep
    Amat[:, k] = Bfull.T @ (field(dx, H0) - field(-dx, H0)) / (2 * hstep) / 4  # coordinates in u-orthonormal basis
print("(a) linearization at the uniform table in a u-orthonormal basis:")
print("    asymmetry ||A-A^T||_max = %.2e; eigenvalues %s" % (np.abs(Amat - Amat.T).max(), np.round(np.sort(np.linalg.eigvals(Amat).real), 6)))

def Nb(R):
    X = R ** 2 / 2 + J * R ** 2 + (2 * R + R ** 2 / 2) ** 2 / 2
    Y = R ** 2 / 2 + (2 * R + R ** 2 / 2) ** 2 / 2
    nq = (X / 4) * (R + R ** 2 / 2) + R ** 2 / 4 + (X / 4) * J * R + (J / 2) * R ** 2
    ns = (Y / 4) * (R + R ** 2 / 2) + R ** 2 / 4 + (X / 4) * J * R + (J / 2) * R ** 2
    return max(nq, ns)

def Nbound(r):
    return 2 * Nb(2 * r)

# numerical sanity check of the remainder bound
rng = np.random.default_rng(7)
worst = 0.0
for r in (1e-3, 1e-2, 5e-2, 0.2):
    mx = 0.0
    for _ in range(300):
        c = rng.normal(size=12); c *= r / np.linalg.norm(c)
        xx = Bfull @ c
        Nx = Bfull.T @ (field(xx, H0)) / 4 - Amat @ c
        mx = max(mx, np.linalg.norm(Nx) / r ** 2)
    print("    r=%.0e: sampled max ||N(x)||/||x||^2 = %.3f ; analytic bound/||x||^2 = %.3f" % (r, mx, Nbound(r) / r ** 2))

# sign block (phi1, phi2, psi1, psi2)
A4 = np.array([[-1, .5, .75, 0], [.5, -1, 0, .75], [.75, 0, -1, .5], [0, .75, .5, -1]])

def explicit_window(eta, h, r0, tgrid):
    phi0 = np.array([np.arctanh(3 * eta), np.arctanh(-eta), np.arctanh(3 * eta), np.arctanh(-eta)])
    xl = np.array([np.linalg.norm(expm(A4 * t) @ phi0) for t in tgrid])
    phl = np.array([expm(A4 * t) @ phi0 for t in tgrid])
    # note: the selection test below needs the minimal sign coordinate of the linear solution
    dt = tgrid[1] - tgrid[0]
    E = np.zeros_like(tgrid)
    for it in range(200):
        integrand = np.array([Nbound(xl[k] + np.exp(tgrid[k] / 4) * r0 + E[k]) + 2 * h for k in range(len(tgrid))])
        En = np.zeros_like(E)
        for k in range(1, len(tgrid)):
            s = tgrid[:k + 1]
            # upper sum: integrand evaluated at right endpoints, kernel at left endpoints
            ker = np.exp((tgrid[k] - s[:-1]) / 4)
            En[k] = np.sum(ker * np.maximum(integrand[1:k + 1], integrand[:k])) * dt
        if np.max(np.abs(En - E)) < 1e-15 or np.any(En > 1.0):
            E = En; break
        E = En
    dev = np.exp(tgrid / 4) * r0 + E
    thr = eta / (8 * np.sqrt(2))
    minc = np.array([np.tanh(min(phl[k]) - 2 * dev[k]) for k in range(len(tgrid))])
    ok = minc > thr
    return ok, E, dev, minc, thr

tg = np.linspace(0, 3.0, 301)
print("(b) explicit Gronwall window (all eight conditionals > eta/(8 sqrt2)), null cone needs 2 r0 < atanh(eta):")
for eta in (1e-3, 5e-4, 2e-4):
    best = None
    for frac in (0.02, 0.05, 0.1, 0.15, 0.2):
        r0 = frac * eta; h = 0.01 * eta
        ok, E, dev, minc, thr = explicit_window(eta, h, r0, tg)
        if ok.any() and 2 * r0 < np.arctanh(eta):
            best = (r0, h, tg[ok].min(), tg[ok].max(), E[np.argmin(np.abs(tg - np.log(4)))])
    if best:
        print("    eta=%.0e: r0=%.2e (=%.2f eta), h=%.1e: selection guaranteed for t in [%.3f, %.3f] (grid end %.3f); E(t*)=%.2e"
              % (eta, best[0], best[0] / eta, best[1], best[2], best[3], tg[-1], best[4]))
    else:
        print("    eta=%.0e: no admissible r0 in the scanned fractions" % eta)

# (c) floating-point radius test of the exact flow
print("(c) exact-flow radius test, eta=0.02, h=2e-4, kappa=0.5: random centered perturbations of u-norm r")
eta = 0.02; hh = 2e-4
Hrec = [[hh, -hh], [hh, -hh]]
phi0 = [np.arctanh(3 * eta), np.arctanh(-eta)]
xb = np.concatenate([phi0[0] * av, phi0[1] * av, phi0[0] * av, phi0[1] * av])
thr = eta / (8 * np.sqrt(2))
for r in (0.002, 0.005, 0.01, 0.02):
    nsel = 0; nnull = 0; ntot = 8
    for trial in range(ntot):
        c = rng.normal(size=12); c *= r / np.linalg.norm(c)
        x0 = xb + Bfull @ c
        sol = solve_ivp(lambda t, y: field(y, Hrec), (0, 3.0), x0, t_eval=np.linspace(0, 3.0, 301), rtol=1e-9, atol=1e-12)
        sel = [np.all(cond_means(sol.y[:, k]) > thr) or np.all(cond_means(sol.y[:, k]) < -thr) for k in range(sol.y.shape[1])]
        init_sel = sel[0]
        if (not init_sel) and any(sel):
            nsel += 1
        soln = solve_ivp(lambda t, y: null_field(y, Hrec), (0, 30.0), x0, t_eval=np.linspace(0, 30.0, 301), rtol=1e-9, atol=1e-12)
        seln = [np.all(cond_means(soln.y[:, k]) > thr) or np.all(cond_means(soln.y[:, k]) < -thr) for k in range(soln.y.shape[1])]
        if any(seln):
            nnull += 1
    cone = 2 * r < np.arctanh(eta)
    print("    r=%.3f (= %.2f eta; inside 2r<atanh(eta): %s): full selects on [0,3] in %d/%d; null ever selects in %d/%d"
          % (r, r / eta, cone, nsel, ntot, nnull, ntot))
