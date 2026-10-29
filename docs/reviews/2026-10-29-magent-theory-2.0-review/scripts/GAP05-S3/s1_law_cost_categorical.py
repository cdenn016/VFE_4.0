r"""GAP05-S3 s1: a categorical (non-Gaussian) instance of it2:law-potential.

Checks, at a random interior state, by complex-step differentiation:
  (a) the cost variation it2:law-cost-variation
      dC_e = int Delta_e dr_e + int r_e [ int log(A/B) dA - int (A/B) dB ];
  (b) the own-block first variation
      delta E_T / delta r_e = log r_e - E_{r_{-e}} log pi_omega + beta_e Delta_e  (+ const),
      and that replacing pi_omega by the product of its marginals changes it;
  (c) the row identities it2:row-gap: E_T(var) - E_T(red) = sum_i T_i KL(beta_i || beta_i^*)
      and dE_T = dF_0 + sum_e beta_e^* dC_e;
  (d) the Fisher (replicator) velocity of r_e: dE/dt = -Var_{r_e}(g_e).
Model: 3 agents, contents in {0,1,2}, dictionaries in Z_3 acting by cyclic shift
(a nonlinear dictionary action), a noisy 3x3 code, edges e0:1->0, e1:2->0, e2:0->1,
and a joint dictionary prior that couples e0 and e2 (a two-cycle loop term).
"""
import numpy as np

rng = np.random.default_rng(20261029)
n = 3          # content alphabet
m = 3          # message alphabet
d = 3          # dictionary alphabet (Z_3)

def rand_simplex(k):
    x = rng.uniform(0.5, 2.0, size=k)
    return x / x.sum()

Q = np.array([rand_simplex(m) for _ in range(n)])            # sender code kernel Q[y, h]
edges = [(1, 0), (2, 0), (0, 1)]                              # (sender j, receiver i)
rows = {0: [0, 1], 1: [2]}                                    # receiver -> candidate edges
T_temp = {0: 0.7, 1: 1.3}
row_prior = {0: np.array([0.4, 0.6]), 1: np.array([1.0])}

def K_rec(w):
    """Receiver kernel: content y read through the dictionary inverse, then coded."""
    return np.array([Q[(y - w) % n] for y in range(n)])

p_ref = [rand_simplex(n) for _ in range(3)]
W = np.zeros((d, d, d))
for w0 in range(d):
    for w1 in range(d):
        for w2 in range(d):
            W[w0, w1, w2] = (0.8 * sum(np.cos(2 * np.pi * w / d) for w in (w0, w1, w2))
                             + 1.5 * np.cos(2 * np.pi * (w0 + w2) / d))
W += 0.3 * rng.standard_normal(W.shape)
pi_w = np.exp(W); pi_w /= pi_w.sum()

def KL(a, b):
    return np.sum(a * (np.log(a) - np.log(b)))

def unpack(x):
    q = [x[3 * k:3 * k + 3] for k in range(3)]
    r = [x[9 + 3 * k:9 + 3 * k + 3] for k in range(3)]
    beta = {0: x[18:20], 1: x[20:21]}
    return q, r, beta

def costs(q, r):
    C, Delta = [], []
    for e, (j, i) in enumerate(edges):
        B = q[j] @ Q
        De = np.array([KL(q[i] @ K_rec(w), B) for w in range(d)])
        Delta.append(De)
        C.append(np.sum(r[e] * De))
    return C, Delta

def F0(q, r):
    joint = np.einsum('a,b,c->abc', r[0], r[1], r[2])
    return sum(KL(q[k], p_ref[k]) for k in range(3)) + np.sum(joint * (np.log(joint) - np.log(pi_w)))

def E_var(x):
    q, r, beta = unpack(x)
    C, _ = costs(q, r)
    val = F0(q, r)
    for i, I in rows.items():
        val += sum(beta[i][k] * C[e] for k, e in enumerate(I)) + T_temp[i] * KL(beta[i], row_prior[i])
    return val

def E_red(x):
    q, r, beta = unpack(x)
    C, _ = costs(q, r)
    val = F0(q, r)
    for i, I in rows.items():
        Z = sum(row_prior[i][k] * np.exp(-C[e] / T_temp[i]) for k, e in enumerate(I))
        val -= T_temp[i] * np.log(Z)
    return val

def cstep(f, x, v, h=1e-30):
    return np.imag(f(x + 1j * h * v)) / h

x0 = np.concatenate([rand_simplex(3) for _ in range(6)] + [rand_simplex(2), np.array([1.0])])
q0, r0, b0 = unpack(x0)

def tangent(blocks):
    v = np.zeros_like(x0)
    for (s, l) in blocks:
        u = rng.standard_normal(l); u -= u.mean(); v[s:s + l] = u
    return v

print("== (a) it2:law-cost-variation, complex step vs formula ==")
worst = 0.0
_, Delta0 = costs(q0, r0)
for e, (j, i) in enumerate(edges):
    for trial in range(4):
        v = tangent([(3 * i, 3), (3 * j, 3), (9 + 3 * e, 3)])
        def Ce(x, e=e):
            q, r, _ = unpack(x)
            return costs(q, r)[0][e]
        num = cstep(Ce, x0, v)
        dq = [v[3 * k:3 * k + 3] for k in range(3)]
        dr = v[9 + 3 * e:9 + 3 * e + 3]
        B = q0[j] @ Q
        dB = dq[j] @ Q
        formula = np.sum(dr * Delta0[e])
        for w in range(d):
            A = q0[i] @ K_rec(w)
            dA = dq[i] @ K_rec(w)
            formula += r0[e][w] * (np.sum(np.log(A / B) * dA) - np.sum((A / B) * dB))
        worst = max(worst, abs(num - formula))
print("max |complex-step - formula| over 12 directions = %.2e" % worst)

print("== (b) own-block first variation of the variable-row potential ==")
beta_of_edge = {0: b0[0][0], 1: b0[0][1], 2: b0[1][0]}
worst = 0.0
worst_ind = 0.0
marg = [pi_w.sum(axis=(1, 2)), pi_w.sum(axis=(0, 2)), pi_w.sum(axis=(0, 1))]
logpi = np.log(pi_w)
Elogs = [np.einsum('abc,b,c->a', logpi, r0[1], r0[2]),
         np.einsum('abc,a,c->b', logpi, r0[0], r0[2]),
         np.einsum('abc,a,b->c', logpi, r0[0], r0[1])]
for e in range(3):
    g = np.log(r0[e]) - Elogs[e] + beta_of_edge[e] * Delta0[e]
    g_ind = np.log(r0[e]) - np.log(marg[e]) + beta_of_edge[e] * Delta0[e]
    for trial in range(4):
        v = tangent([(9 + 3 * e, 3)])
        num = cstep(E_var, x0, v)
        worst = max(worst, abs(num - np.sum(v[9 + 3 * e:9 + 3 * e + 3] * g)))
        worst_ind = max(worst_ind, abs(num - np.sum(v[9 + 3 * e:9 + 3 * e + 3] * g_ind)))
print("max |complex-step - (log r_e - E_r-e log pi + beta_e Delta_e)| = %.2e" % worst)
print("same with pi replaced by the product of its marginals: %.3e (nonzero, so the correlations matter)" % worst_ind)

print("== (c) it2:row-gap identities with these non-Gaussian costs ==")
C0, _ = costs(q0, r0)
gap = 0.0
bstar = {}
for i, I in rows.items():
    w_ = np.array([row_prior[i][k] * np.exp(-C0[e] / T_temp[i]) for k, e in enumerate(I)])
    bstar[i] = w_ / w_.sum()
    gap += T_temp[i] * KL(b0[i], bstar[i])
print("E_var - E_red - sum T KL(beta||beta*) = %.2e" % (E_var(x0) - E_red(x0) - gap))

def surrogate(x):
    q, r, _ = unpack(x)
    C, _ = costs(q, r)
    val = F0(q, r)
    for i, I in rows.items():
        w_ = [row_prior[i][k] * np.exp(-C[e] / T_temp[i]) for k, e in enumerate(I)]
        Zs = sum(w_)
        val += sum(w_[k] / Zs * C[e] for k, e in enumerate(I))
    return val

worst = 0.0
for trial in range(6):
    v = tangent([(0, 3), (3, 3), (6, 3), (9, 3), (12, 3), (15, 3)])
    num = cstep(E_red, x0, v)
    dF0 = cstep(lambda x: F0(*unpack(x)[:2]), x0, v)
    dC = [cstep(lambda x, e=e: costs(*unpack(x)[:2])[0][e], x0, v) for e in range(3)]
    form = dF0 + bstar[0][0] * dC[0] + bstar[0][1] * dC[1] + bstar[1][0] * dC[2]
    surr = cstep(surrogate, x0, v)
    worst = max(worst, abs(num - form))
print("max |dE_T - (dF0 + sum beta* dC)| over 6 directions = %.2e" % worst)
print("last direction: dE_T = %.6f, entropy-free surrogate derivative = %.6f" % (num, surr))

print("== (d) replicator (Fisher) velocity of r_0 and its dissipation ==")
e = 0
g = np.log(r0[e]) - Elogs[e] + beta_of_edge[e] * Delta0[e]
vel = -r0[e] * (g - np.sum(r0[e] * g))
v = np.zeros_like(x0)
v[9:12] = vel
dE = cstep(E_var, x0, v)
var = np.sum(r0[e] * g ** 2) - np.sum(r0[e] * g) ** 2
print("dE/dt along replicator velocity = %.12f; -Var_r(g) = %.12f; diff = %.2e" % (dE, -var, dE + var))
print("Interior finite tables: positive message entries and finite sums, so the dominated-chart")
print("and envelope hypotheses of it2:law-cost-variation hold trivially in this instance.")
