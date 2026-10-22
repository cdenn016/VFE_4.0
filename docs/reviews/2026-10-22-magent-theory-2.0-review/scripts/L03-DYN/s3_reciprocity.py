"""L03-DYN script 3: one potential versus agent-wise potentials.

(a) Cross-derivative (Monderer-Shapley) test: agent-wise KL costs F_1=KL(q1||q2), F_2=KL(q2||q1)
    on a finite alphabet do not form an exact potential game unless q1=q2.
(b) Directed 3-cycle of Bernoulli agents, fixed unit rows, references r_i:
    agent-wise Fisher descent of F_i = KL(p_i||r_i) + w KL(p_i||p_{i+1}) is linear in log-odds with
    Jacobian -(1+w)I + wP (P cyclic), whose spectrum is complex: not a gradient flow for any metric.
    The chapter-3 flow (Fisher gradient of the sum E, with the incoming source force) has a real
    spectrum at its equilibrium, as every Riemannian gradient flow must.
"""
import numpy as np
import sympy as sp
from scipy.optimize import fsolve

# (a)
x1, x2, y1, y2 = sp.symbols('x1 x2 y1 y2', positive=True)   # q1=(x1,x2), q2=(y1,y2) unnormalized coords
F1 = x1 * sp.log(x1 / y1) + x2 * sp.log(x2 / y2)
F2 = y1 * sp.log(y1 / x1) + y2 * sp.log(y2 / x2)
c12 = sp.diff(F1, x1, y1)
c21 = sp.diff(F2, y1, x1)
print("(a) d2F1/dx1dy1 =", sp.simplify(c12), "  d2F2/dy1dx1 =", sp.simplify(c21),
      "  difference =", sp.simplify(c12 - c21))

# (b) agent-wise, in log-odds l_i: dl_i/dt = -(l_i - rho_i) - w (l_i - l_{i+1})
w = sp.symbols('w', positive=True)
Pc = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
J = -(1 + w) * sp.eye(3) + w * Pc
ev = J.eigenvals()
print("(b) agent-wise Jacobian eigenvalues:", [sp.simplify(e) for e in ev])

# numeric: agent-wise vs global at w=2, rho = logits of r
wn = 2.0
rho = np.array([-1.2, 0.3, 1.5])
sig = lambda l: 1 / (1 + np.exp(-l))


def agentwise(l):
    return -(l - rho) - wn * (l - np.roll(l, -1))


def global_field(l):
    # E = sum_i KL(p_i||r_i) + w KL(p_i||p_{i+1});  dE/dp_i = (l_i-rho_i) + w(l_i-l_{i+1}) + w (p_i-p_{i-1})/(p_i(1-p_i))
    p = sig(l)
    dEdp = (l - rho) + wn * (l - np.roll(l, -1)) + wn * (p - np.roll(p, 1)) / (p * (1 - p))
    pdot = -p * (1 - p) * dEdp            # Fisher (Shahshahani) gradient on each Bernoulli simplex
    return pdot / (p * (1 - p))           # convert to log-odds velocity


def jac(f, x, h=1e-6):
    n = len(x)
    M = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        M[:, k] = (f(x + e) - f(x - e)) / (2 * h)
    return M


la = fsolve(agentwise, np.zeros(3), xtol=1e-14)
lg = fsolve(global_field, np.zeros(3), xtol=1e-14)
print("agent-wise equilibrium log-odds", la, " residual", np.abs(agentwise(la)).max())
print("agent-wise Jacobian spectrum   ", np.linalg.eigvals(jac(agentwise, la)))
print("global equilibrium log-odds    ", lg, " residual", np.abs(global_field(lg)).max())
print("global Jacobian spectrum       ", np.linalg.eigvals(jac(global_field, lg)))
# self-adjointness check of the global linearization w.r.t. the Fisher metric in log-odds, g = diag(p(1-p))
p = sig(lg)
g = np.diag(p * (1 - p))
Jg = jac(global_field, lg)
print("asymmetry of g J (global; zero for a Fisher gradient flow):", np.abs(g @ Jg - (g @ Jg).T).max())
Ja = jac(agentwise, la)
pa = sig(la)
ga = np.diag(pa * (1 - pa))
print("asymmetry of g J (agent-wise):", np.abs(ga @ Ja - (ga @ Ja).T).max())
# spiral: agent-wise trajectory angle around equilibrium
