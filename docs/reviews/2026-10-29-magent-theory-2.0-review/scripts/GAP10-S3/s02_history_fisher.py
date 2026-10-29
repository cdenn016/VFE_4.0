r"""GAP10-S3 s02: history-inclusive joint Fisher tensor for lp:local-realization / lp:cost.

Adaptive protocol on the chapter 14 field (gap10_model), context c in R^2:
  t in [0, t1]: native flow with fixed persistent coefficients;
  t1: diagnostic joint relative query of agents (1,2): Delta = Psi_1 - Psi_2, one sample each,
      density p_c(Delta) = I_0(|R(-Delta)B_1 + B_2|)/(I_0(|B_1|) I_0(|B_2|)) w.r.t. dDelta/2pi;
      policy d = 3 if cos Delta >= 0 else d = 2 (history only, no context dependence);
      sign evidence on agent d with strength r_d, actual outcome o ~ (1 + o m_d tanh(r_d/2))/2,
      appended persistently: H_d += r_d o;
  t in [t1, T]: native flow with the updated coefficients;
  at T: current samples (z_i, phi_i, psi_i) ~ q_i, presentation w_i ~ s_i, content label J_i ~ beta_i.
Joint experiment P_c(dh, dY) = Pi_c(dh) Q^h_c(dY) (lp:actual-experiment).

Scores: contextual s_a (complex step in c), marked orbit l_m (rotation of phi: dA = J A, dC = J C),
physical l_p (common rotation of psi: dB = J B).  Gram matrix G over (s_1, s_2, l_m, l_p) =
history block + E_h[current block(h)] (cross terms vanish because current scores have
conditional mean zero).  Parents: reference (U from c_0, lp:retention / lp:actual-experiment),
pointwise canonical (U(S(x(c,T)))), causal frozen local initial frame (U(S(x(c,0)))).
"""
from __future__ import annotations

import time
import numpy as np
from scipy.special import iv

from gap10_model import Model, initial_section, rk4, Umat, rotB, Ssum, unpack, rot, cnorm

t0 = time.time()
M = Model()
ce = np.array([0.25, -0.15])
c0 = np.array([0.0, 0.0])
t1, T, dt = 0.4, 1.0, 1.0 / 400
n1, n2 = int(round(t1 / dt)), int(round((T - t1) / dt))
r_ev = {2: 0.8, 3: 1.1}          # agent labels 1..3; evidence strengths
hc = 1e-20
E2 = np.eye(2)
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def run(c, x0=None):
    r"""Return state at t1 and, for each branch (d, o), the state at T."""
    c = np.asarray(c)
    if x0 is None:
        x0 = initial_section(c).astype(complex)
    xs1 = rk4(lambda t, x: M.field(x, c), x0, 0.0, t1, n1)
    x1 = xs1[-1]
    out = {}
    for d in (2, 3):
        for o in (-1, 1):
            H = M.H.astype(complex).copy()
            H[d - 1] += r_ev[d] * o
            xs2 = rk4(lambda t, x: M.field(x, c, H), x1, t1, T, n2)
            out[(d, o)] = xs2[-1]
    return x0, x1, out


ctx = {'0': ce + 0j, 'i1': ce + 1j * hc * E2[0], 'i2': ce + 1j * hc * E2[1]}
runs = {k: run(v) for k, v in ctx.items()}
# physical-rotation complex step at c_e: rotate the initial B's by i*hc
x0c = initial_section(ce).astype(complex)
runs['phys'] = run(ce + 0j, rotB(x0c, rot(1j * hc)))
ref = run(c0 + 0j)
print(f"trajectories done ({time.time()-t0:.1f} s)")

# Gauss-Legendre nodes on the two arcs of Delta
gl_x, gl_w = np.polynomial.legendre.leggauss(48)
arcs = {3: (-np.pi / 2, np.pi / 2), 2: (np.pi / 2, 3 * np.pi / 2)}
nodes = {}
for d, (a, b) in arcs.items():
    nodes[d] = (0.5 * (b - a) * gl_x + 0.5 * (b + a), 0.5 * (b - a) * gl_w / (2 * np.pi))


def logp_delta(x1, Dl):
    B = unpack(x1)[3]
    v = B[0] @ rot(-Dl).T + B[1]
    return np.log(iv(0, cnorm(v))) - np.log(iv(0, cnorm(B[0]))) - np.log(iv(0, cnorm(B[1])))


def logP_o(x1, d, o, c):
    beta, gam, L, m, t, eev = M.rows(x1, c)
    return np.log((1 + o * m[d - 1] * np.tanh(r_ev[d] / 2)) / 2)


# ---------- current-sample Fisher pieces (exponential family, exact quadrature) ----------
NQ = 96
phis = 2 * np.pi * np.arange(NQ) / NQ


def fisher_q(theta):
    r"""Cov of T=(z/2, e_phi, z e_phi, e_psi) under q(z,phi,psi) ~ exp(l z/2 + A.e + z C.e + B.e_psi)."""
    l, A, C, B = theta[0], theta[1:3], theta[3:5], theta[5:7]
    e = np.stack([np.cos(phis), np.sin(phis)], axis=1)
    Tz = []
    W = []
    for z in (-1, 1):
        lw = l * z / 2 + e @ A + z * (e @ C)
        W.append(np.exp(lw))
        Tz.append(np.concatenate([np.full((NQ, 1), z / 2), e, z * e], axis=1))
    W = np.concatenate(W); Tz = np.concatenate(Tz)
    W = W / W.sum()
    mu = W @ Tz
    C1 = (Tz - mu).T @ ((Tz - mu) * W[:, None])
    wpsi = np.exp(e @ B); wpsi /= wpsi.sum()
    mp = wpsi @ e
    C2 = (e - mp).T @ ((e - mp) * wpsi[:, None])
    F = np.zeros((7, 7))
    F[:5, :5] = C1
    F[5:, 5:] = C2
    return F, mu, mp


def theta_i(x, i):
    l, A, C, B, n = unpack(x)
    return np.concatenate([[l[i]], A[i], C[i], B[i]])


def current_block(xr, xa_list, c_list, rotfun=None):
    r"""Gram matrix of (s_1, s_2, l_m, l_p) for current samples at one branch.

    xr: real state; xa_list: complex-step states for a=1,2; rotfun(x, c) -> parent rotation
    matrix applied to B blocks (None for the fine law).
    """
    def apply(x, c):
        return x if rotfun is None else rotB(x, rotfun(x, c))
    xr_p = np.real(apply(xr + 0j, ce + 0j))
    dth = []
    for a in range(2):
        xp = apply(xa_list[a], c_list[a])
        dth.append(np.imag(xp) / hc)
    G = np.zeros((4, 4))
    # laws
    for i in range(3):
        th = theta_i(xr_p, i)
        F, mu, mp = fisher_q(th)
        vecs = [theta_i(dth[0], i), theta_i(dth[1], i)]
        A, C, B = th[1:3], th[3:5], th[5:7]
        vm = np.concatenate([[0.0], J2 @ A, J2 @ C, [0.0, 0.0]])
        vp = np.concatenate([[0.0], [0, 0], [0, 0], J2 @ B])
        V = np.stack(vecs + [vm, vp])
        G += V @ F @ V.T
    # presentations (trivial marked and physical action)
    l, Aa, Cc, Bb, n = unpack(xr_p)
    Fp = (1 - np.tanh(n / 2) ** 2) / 4
    for a in range(2):
        for b in range(2):
            G[a, b] += np.sum(unpack(dth[a])[4] * unpack(dth[b])[4] * Fp)
    # content source labels: categorical rows (invariant under both actions)
    beta_r = np.real(M.rows(xr_p + 0j, ce + 0j)[0])
    dlogb = []
    for a in range(2):
        xp = apply(xa_list[a], c_list[a])
        bc = M.rows(xp, c_list[a])[0]
        dlogb.append(np.imag(np.log(bc)) / hc)
    for a in range(2):
        for b in range(2):
            G[a, b] += np.sum(beta_r * dlogb[a] * dlogb[b])
    return G


def rot_reference(branch):
    xref = np.real(ref[2][branch])
    U = Umat(Ssum(xref))
    return lambda x, c: U + 0j


def rot_pointwise(x, c):
    return Umat(Ssum(x))


def make_rot_causal(key):
    x0 = runs[key][0]
    U = Umat(Ssum(x0))
    return lambda x, c: U


keys = ['i1', 'i2']
results = {}
for name in ['fine', 'reference', 'pointwise', 'causal']:
    Ghist = np.zeros((4, 4))
    Gcur = np.zeros((4, 4))
    defect_terms = []
    Pd = {}
    for d in (2, 3):
        Dn, Dw = nodes[d]
        for o in (-1, 1):
            branch = (d, o)
            # history scores at nodes (parents: history computed from parent states at t1)
            def hist_state(key):
                x1 = runs[key][1]
                if name == 'fine':
                    return x1
                if name == 'reference':
                    return rotB(x1, Umat(Ssum(ref[1])) + 0j)
                if name == 'pointwise':
                    return rotB(x1, Umat(Ssum(x1)))
                return rotB(x1, Umat(Ssum(runs[key][0])))
            x1r = np.real(hist_state('0'))
            lp0 = np.real(np.array([logp_delta(x1r + 0j, Dl) for Dl in Dn]))
            lo0 = np.real(logP_o(x1r + 0j, d, o, ce + 0j))
            w = Dw * np.exp(lp0) * np.exp(lo0)
            sh = []
            for a, key in enumerate(keys):
                x1c = hist_state(key)
                sa = np.imag(np.array([logp_delta(x1c, Dl) for Dl in Dn])) / hc \
                    + np.imag(logP_o(x1c, d, o, ctx[key])) / hc
                sh.append(sa)
            sh = np.array(sh)
            Ghist[:2, :2] += (sh * w) @ sh.T
            pb = w.sum()
            Pd[branch] = pb
            # current block
            xr = np.real(runs['0'][2][branch])
            xa_list = [runs['i1'][2][branch], runs['i2'][2][branch]]
            c_list = [ctx['i1'], ctx['i2']]
            if name == 'fine':
                rf = None
            elif name == 'reference':
                rf = rot_reference(branch)
            elif name == 'pointwise':
                rf = rot_pointwise
            else:
                rf = None
            if name == 'causal':
                # causal parent: B rotated by U(S(x(c,0))) with x(c,0) from the same context
                xr_c = rotB(xr + 0j, Umat(Ssum(runs['0'][0])))
                xa_c = [rotB(xa_list[a], Umat(Ssum(runs[keys[a]][0]))) for a in range(2)]
                Gc = current_block(np.real(xr_c), xa_c, c_list, None)
            else:
                Gc = current_block(xr, xa_list, c_list, rf)
            Gcur += pb * Gc
    results[name] = (Ghist, Gcur, Pd)

print(f"Fisher blocks done ({time.time()-t0:.1f} s)\n")
Gf_h, Gf_c, Pd = results['fine']
print("branch probabilities P(d,o):", {k: round(v, 6) for k, v in Pd.items()}, " sum =", f"{sum(Pd.values()):.15f}")
Gfine = Gf_h + Gf_c
np.set_printoptions(precision=6, suppress=False, linewidth=140)
print("history block G_hist (s1, s2):\n", Gf_h[:2, :2])
print("current block E_h[G_cur] over (s1, s2, l_m, l_p):\n", Gf_c)
print(f"marked-physical cross E[l_m l_p] = {Gf_c[2,3]:.3e} (Fisher block-diagonal (z,phi) vs psi)")

# physical score of the history law: complex-step rotation of all B at preparation
x1p = runs['phys'][1]
x1r = runs['0'][1]
mx = 0.0
for d in (2, 3):
    Dn, _ = nodes[d]
    for o in (-1, 1):
        s_k = np.imag(np.array([logp_delta(x1p, Dl) for Dl in Dn])) / hc + np.imag(logP_o(x1p, d, o, ce + 0j)) / hc
        mx = max(mx, np.max(np.abs(s_k)))
print(f"history score along common physical rotation (should vanish): max |d/dkappa log Pi| = {mx:.2e}")
# equivariance of the flow under the rotation (sanity)
xT_p = runs['phys'][2][(3, 1)]
xT_0 = runs['0'][2][(3, 1)]
Bp = unpack(np.imag(xT_p) / hc)[3]
Bpred = (unpack(np.real(xT_0))[3]) @ J2.T
print(f"equivariance: |dB/dkappa - J B| at T = {np.max(np.abs(Bp - Bpred)):.2e}; other blocks |d/dkappa| = "
      f"{np.max(np.abs(np.imag(xT_p)[:15] / hc)):.2e}")

# conditional mean zero of current scores at one branch (quadrature)
th = theta_i(np.real(runs['0'][2][(3, 1)]), 0)
F, mu, mp = fisher_q(th)
print(f"E_Q[T - E T] = 0 by construction; cov positive: min eig F(agent 1) = {np.linalg.eigvalsh(F).min():.4f}\n")


def summarize(G):
    Gss = G[:2, :2]; Gsm = G[:2, 2]; Gmm = G[2, 2]
    a = -Gsm / Gmm
    h = Gss - np.outer(Gsm, Gsm) / Gmm
    return a, h


out = {}
for name in ['fine', 'reference', 'pointwise', 'causal']:
    Gh, Gc, _ = results[name]
    G = Gh + Gc
    a, h = summarize(G)
    out[name] = (G, a, h)
    print(f"{name:10s}: connection a(e1), a(e2) = {a[0]: .10f}, {a[1]: .10f};  "
          f"h = [[{h[0,0]:.10f}, {h[0,1]:.10f}], [{h[1,0]:.10f}, {h[1,1]:.10f}]]")
Gf = out['fine'][0][:3, :3]
for name in ['reference', 'pointwise', 'causal']:
    Gp = out[name][0][:3, :3]
    print(f"  max |G_{name} - G_fine| over (s1,s2,l_m): {np.max(np.abs(Gp - Gf)):.2e};  "
          f"history block diff {np.max(np.abs(results[name][0] - results['fine'][0])):.2e};  "
          f"connection diff {np.max(np.abs(out[name][1] - out['fine'][1])):.2e}")

# closed-form defect for phase-field parents: dh(X,X) = E_h[2 (X Theta) b(X) + (X Theta)^2 I]
print("\nclosed-form defect check, Theta = parent rotation angle:")
for name in ['pointwise', 'causal']:
    pred = np.zeros((2, 2))
    for d in (2, 3):
        for o in (-1, 1):
            br = (d, o)
            xr = np.real(runs['0'][2][br])
            dTh = []
            for a, key in enumerate(keys):
                if name == 'pointwise':
                    Sk = Ssum(runs[key][2][br])
                else:
                    Sk = Ssum(runs[key][0])
                dTh.append(-np.imag(np.arctan(Sk[1] / Sk[0])) / hc)
            dTh = np.array(dTh)
            # fine b and I at this branch
            Gb = current_block(xr, [runs['i1'][2][br], runs['i2'][2][br]], [ctx['i1'], ctx['i2']], None)
            b = Gb[:2, 3]; I = Gb[3, 3]
            pred += Pd[br] * (np.outer(dTh, b) + np.outer(b, dTh) + I * np.outer(dTh, dTh))
    diff = out[name][2] - out['fine'][2]
    print(f"  {name:9s}: h_parent - h_fine = [[{diff[0,0]: .10f}, {diff[0,1]: .10f}], [{diff[1,0]: .10f}, {diff[1,1]: .10f}]]")
    print(f"             closed form     = [[{pred[0,0]: .10f}, {pred[0,1]: .10f}], [{pred[1,0]: .10f}, {pred[1,1]: .10f}]]"
          f"   max residual {np.max(np.abs(diff - pred)):.2e}")

# conditioning on the decision d = 3 (positive-probability event): normalizer score retained
print("\nconditioning on {cos Delta >= 0}:")
for name in ['fine', 'reference']:
    Gh = np.zeros((2, 2)); mean = np.zeros(2); P3 = 0.0
    Dn, Dw = nodes[3]
    for o in (-1, 1):
        x1r = np.real(runs['0'][1])
        if name == 'reference':
            x1r = np.real(rotB(runs['0'][1], Umat(Ssum(ref[1])) + 0j))
        lp0 = np.real(np.array([logp_delta(x1r + 0j, Dl) for Dl in Dn]))
        lo0 = np.real(logP_o(x1r + 0j, 3, o, ce + 0j))
        w = Dw * np.exp(lp0 + lo0)
        sh = []
        for key in keys:
            x1c = runs[key][1] if name == 'fine' else rotB(runs[key][1], Umat(Ssum(ref[1])) + 0j)
            sh.append(np.imag(np.array([logp_delta(x1c, Dl) for Dl in Dn])) / hc + np.imag(logP_o(x1c, 3, o, ctx[key])) / hc)
        sh = np.array(sh)
        Gh += (sh * w) @ sh.T; mean += sh @ w; P3 += w.sum()
    Gh /= P3; mean /= P3
    cov = Gh - np.outer(mean, mean)
    print(f"  {name:9s}: P(d=3) = {P3:.12f}; normalizer score d log P(d=3) = {mean};"
          f" conditional history Fisher (centered) diag = {np.diag(cov)}; uncentered diag = {np.diag(Gh)}")
print(f"\nruntime {time.time()-t0:.1f} s")
