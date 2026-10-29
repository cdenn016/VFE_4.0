r"""D18-LP check 8: small-amplitude linear limit of the causal-parent tensor defect.

In the eps, delta -> 0 limit the rows equal the priors, the B-equation is linear and
context-independent:  dB/dt = M_B B,  M_B = -7/5 I + 2/5 11^T (acting on the agent index).
Then S(c,t) = e^{-t/5} S(c,0) (so omega_loc = 0 and alpha(c,t) = alpha_0(c)), and with
I(r) ~ r^2/2:
   b_t(X)/I_t = [3 det(Bbar, X Bbar) + r sum_i det(Bt_i, X Bt_i)] / [3|Bbar|^2 + r sum_i |Bt_i|^2],
   r = e^{-12 t/5},  Bbar = S/3,  Bt_i = B_i - Bbar.
The causal-parent tensor defect is (X alpha0)[(X alpha0) I_t - 2 b_t(X)].  Check symbolically that
b_t/I_t is a Moebius function of r whose constant value (when constant) is X alpha0, so the defect
cannot vanish on a time interval unless X alpha0 = 0; and numerically that the formula matches the
exact linear evolution.
"""
import numpy as np
import sympy as sp

# numeric check of the decomposition along the exact linear flow
rng = np.random.default_rng(3)
MB = -1.4 * np.eye(3) + 0.4 * np.ones((3, 3))
B0 = rng.normal(size=(3, 2)); dB0 = rng.normal(size=(3, 2))   # values and contextual derivative X B
det = lambda a, b: a[0] * b[1] - a[1] * b[0]
worst = 0.0
for t in [0.0, 0.3, 1.0, 2.5]:
    w, Vv = np.linalg.eigh(MB)
    Et = Vv @ np.diag(np.exp(w * t)) @ Vv.T
    Bt, dBt = Et @ B0, Et @ dB0
    lhs = sum(det(Bt[i], dBt[i]) for i in range(3)) / sum(Bt[i] @ Bt[i] for i in range(3))
    Bbar, dBbar = B0.mean(0), dB0.mean(0)
    Bt_, dBt_ = B0 - Bbar, dB0 - dBbar
    r = np.exp(-12 * t / 5)
    rhs = (3 * det(Bbar, dBbar) + r * sum(det(Bt_[i], dBt_[i]) for i in range(3))) / \
          (3 * Bbar @ Bbar + r * sum(Bt_[i] @ Bt_[i] for i in range(3)))
    worst = max(worst, abs(lhs - rhs))
    S = Bt.sum(0); dS = dBt.sum(0)
    Xalpha = det(S, dS) / (S @ S)
    Xalpha0 = det(B0.sum(0), dB0.sum(0)) / (B0.sum(0) @ B0.sum(0))
    print(f"t={t}: b/I = {lhs:+.6f} (Moebius formula {rhs:+.6f}); X alpha(t) = {Xalpha:+.6f}, X alpha0 = {Xalpha0:+.6f}")
print("max |b/I - Moebius| =", worst)

P, Q, R, T, r = sp.symbols("P Q R T r", real=True)
f = (P + r * Q) / (R + r * T)
print("d f/dr =", sp.factor(sp.diff(f, r)), " (zero on an interval iff Q R - P T = 0)")
print("value when Q R = P T (T != 0):", sp.simplify(f.subs(Q, P * T / R)), " = P/R = X alpha0")
