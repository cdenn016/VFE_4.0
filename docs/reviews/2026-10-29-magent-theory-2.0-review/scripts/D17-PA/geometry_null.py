r"""D17-PA check 7: null-flow control for the geometry of pa:nonflat-window, at eps=0.

At eps=0 (J=0, uniform records, l=0) the null of pa:angular-null keeps every law von Mises in phi with
fixed direction and amplitude |A_i(t)| obeying d|A_i|/dt = -beta0_i |A_i|, with beta0_i computed from the
complete self cost D_ii = KL(vM(|A_i|) x vM(b_i) || uniform). The full flow at eps=0 has |A_i(t)|=e^{-t}|A_i(0)|.
Computes the minimizing connection, curvature and minimized tensor of both sections at the same times.
"""
import numpy as np
from scipy.special import i0, i1
from scipy.integrate import solve_ivp

a_pr, p_pr = 0.2, 0.4
d, d3, al3 = 0.03, 0.025, 1.0
bh = np.array([0.03, 0.03, 0.03])

def I_(x):
    return x * i1(x) / i0(x)

def klvm(x):
    return x * i1(x) / i0(x) - np.log(i0(x))

def amps(a, t, null):
    r0 = np.array([a, d, d3])
    if not null:
        return np.exp(-t) * r0, np.exp(-t) * bh
    def f(_, y):
        r, b = y[:3], y[3:]
        D = klvm(r) + klvm(b)
        b0 = a_pr * np.exp(-D) / (a_pr * np.exp(-D) + 2 * p_pr)
        return np.concatenate([-b0 * r, -b0 * b])
    sol = solve_ivp(f, (0, t), np.concatenate([r0, bh]), rtol=1e-12, atol=1e-15)
    return sol.y[:3, -1], sol.y[3:, -1]

def geom(a, chi, t, null):
    # agent 1 law vM(r1(a,t), chi); agents 2,3 fixed directions. Scores: radial d log q/d a, angular d/d chi.
    r, _ = amps(a, t, null)
    da = 1e-6
    r_p, _ = amps(a + da, t, null); r_m, _ = amps(a - da, t, null)
    dr1 = (r_p[0] - r_m[0]) / (2 * da)
    # chi-score of agent 1 is r1 sin(phi-chi): inertia I(r1); orbit inertia K = sum I(r_i); momentum j_chi = I(r1)
    # radial scores of agents 2,3 are also a-dependent in the null (through nothing: their rates do not depend on a)
    K = I_(r[0]) + I_(r[1]) + I_(r[2])
    A_chi = -I_(r[0]) / K
    Vr = (i1(r[0]) / i0(r[0])) * 0 + 1.0  # placeholder, replaced below
    ph = 2 * np.pi * np.arange(512) / 512
    w = np.exp(r[0] * np.cos(ph)); w /= w.mean()
    V = np.mean(w * np.cos(ph) ** 2) - (i1(r[0]) / i0(r[0])) ** 2
    h_aa = dr1 ** 2 * V
    h_cc = I_(r[0]) * (I_(r[1]) + I_(r[2])) / K
    return A_chi, h_aa, h_cc

for t in (0.0, 1.0, 5.0, 20.0):
    row = []
    for null in (False, True):
        a, chi = 0.04, 0.3
        dd = 1e-5
        Ap = geom(a + dd, chi, t, null)[0]; Am = geom(a - dd, chi, t, null)[0]
        F = (Ap - Am) / (2 * dd)
        A_chi, h_aa, h_cc = geom(a, chi, t, null)
        row.append((A_chi, F, h_aa, h_cc))
    print("t=%5.1f  full: A_chi=%.6f F=%.4f h_aa=%.3e h_cc=%.3e | null: A_chi=%.6f F=%.4f h_aa=%.3e h_cc=%.3e" % ((t,) + row[0] + row[1]))
