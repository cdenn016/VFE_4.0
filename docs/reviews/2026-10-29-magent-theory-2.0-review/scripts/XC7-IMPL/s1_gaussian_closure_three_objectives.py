r"""XC7-IMPL s1: Which of the three 2.0 objectives keeps Gaussian agents Gaussian?

For each objective we evaluate, on a 1-D grid, the first variation phi of the
objective in the law q_1 and the unit-mobility Fisher velocity
    v = -(phi - E_{q_1} phi),
then measure the fraction of its Fisher norm ||v||^2_{L^2(q_1)} lying outside the
Gaussian tangent space span{1, x, x^2}.  A fraction at roundoff level means the
full flow is tangent to the Gaussian family at this state (restricted flow =
full flow); an O(1) fraction means the Gaussian family is not invariant.

Objectives (three agents, fixed rows, identity or linear-Gaussian channels):
 (R) reciprocal law potential, def:dyn-potential: E = sum_i KL(q_i||r)
     + sum_{j->i} beta_ij KL(q_i || nu_ij), nu_ij = q_j K_ij.
 (P) per-agent free energy of agent 1, def:dyn-per-agent / prop:dyn-per-agent:
     F_1 = beta_11 KL(q_1||r) + sum_j beta_1j KL(q_1||nu_1j) - E log l_1(o|k),
     peers' laws held as data.
 (G) global free energy, eq:dyn-fixed-vfe / prop:dyn-global-local, with
     linear-Gaussian comparison records o_e ~ N(y_i - a y_j, tau) and Gaussian
     priors; the first variation in Q_1 is log(Q_1/P0_1) - E_{Q_-1} log l.
"""
import numpy as np

x = np.linspace(-14.0, 14.0, 40001)
dx = x[1] - x[0]

def npdf(m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)

def fraction_outside(q, phi):
    w = q * dx
    v = -(phi - np.sum(w * phi))
    basis = np.vstack([np.ones_like(x), x, x ** 2]).T
    G = basis.T @ (basis * w[:, None])
    coef = np.linalg.solve(G, basis.T @ (w * v))
    resid = v - basis @ coef
    return np.sum(w * resid ** 2) / np.sum(w * v ** 2)

# Three agents; agent 1 is the agent whose velocity we test.
m = {1: 0.0, 2: 0.5, 3: 1.0}
s2 = {1: 1.0, 2: 1.0, 3: 1.3 ** 2}
q = {i: npdf(m[i], s2[i]) for i in m}
r = npdf(0.0, 4.0)                       # local reference / self evaluator
beta = {(1, 2): 0.6, (2, 1): 0.5, (3, 1): 0.7}   # j->i stored as (i, j)? see below
# convention: beta[(i, j)] = weight of receiver i attending to source j

def channel(q_src, theta, lam, m_src, v_src):
    # linear-Gaussian channel K(v,.) = N(theta v, lam) applied to a Gaussian law
    return npdf(theta * m_src, theta ** 2 * v_src + lam)

out = []
for chan_name, theta, lam in [("identity", 1.0, 0.0), ("linear-Gaussian", 0.8, 0.3)]:
    if lam == 0.0:
        nu = {(i, j): q[j] for (i, j) in beta}
    else:
        nu = {(i, j): channel(q[j], theta, lam, m[j], s2[j]) for (i, j) in beta}
    # (R) reciprocal: receiver role of agent 1 (1 attends to 2) + source role (2 and 3 attend to 1)
    phi_R = np.log(q[1] / r) + beta[(1, 2)] * np.log(q[1] / nu[(1, 2)])
    for (i, j), b in beta.items():
        if j == 1 and i != 1:
            if lam == 0.0:
                phi_R = phi_R - b * q[i] / nu[(i, j)]
            else:
                # incoming term -b * int (q_i/nu)(w) K(v, dw), K(v,.) = N(theta v, lam)
                ratio = q[i] / nu[(i, j)]
                inc = np.array([np.sum(ratio * np.exp(-(x - theta * xv) ** 2 / (2 * lam))
                                       / np.sqrt(2 * np.pi * lam)) * dx for xv in x[::40]])
                phi_R_sub = np.interp(x, x[::40], inc)
                phi_R = phi_R - b * phi_R_sub
    # (P) per-agent free energy of agent 1 with a Gaussian record N(o; k, c)
    o, c = 0.7, 0.5
    loglik = -(o - x) ** 2 / (2 * c)
    b11, b12 = 0.4, 0.6
    phi_P = b11 * np.log(q[1] / r) + b12 * np.log(q[1] / nu[(1, 2)]) - loglik
    # (G) global free energy: Gaussian priors P0_i = N(0,4); records o_12 ~ N(y_1 - a y_2, tau)
    # (1 receives from 2) and o_31 ~ N(y_3 - a y_1, tau) (1 is the sender); mean-field Q.
    a, tau, o12, o31 = 0.8, 0.4, 0.3, -0.2
    E_y2, E_y2sq = m[2], s2[2] + m[2] ** 2
    E_y3, E_y3sq = m[3], s2[3] + m[3] ** 2
    exp_ll_12 = -((o12 - x) ** 2 + 2 * (o12 - x) * a * E_y2 + a ** 2 * E_y2sq) / (2 * tau)
    exp_ll_31 = -((o31 - E_y3) ** 2 + s2[3] + 2 * (o31 - E_y3) * a * x + a ** 2 * x ** 2) / (2 * tau)
    phi_G = np.log(q[1] / r) - exp_ll_12 - exp_ll_31
    out.append((chan_name, fraction_outside(q[1], phi_R), fraction_outside(q[1], phi_P),
                fraction_outside(q[1], phi_G)))

print("fraction of Fisher norm of d/dt log q_1 outside span{1,x,x^2}")
print("channel            reciprocal(E)   per-agent(F_1)   global(F)")
for row in out:
    print(f"{row[0]:<18} {row[1]:.3e}       {row[2]:.3e}        {row[3]:.3e}")

# Control: reciprocal potential with the source role deleted (receiver only) is Gaussian-closed.
phi_recv = np.log(q[1] / r) + beta[(1, 2)] * np.log(q[1] / q[2])
print(f"control, reciprocal with source role deleted: {fraction_outside(q[1], phi_recv):.3e}")
