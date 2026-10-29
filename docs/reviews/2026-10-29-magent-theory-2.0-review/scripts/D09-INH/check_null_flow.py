r"""D09-INH check: interaction necessity (F5) and the common-clock architecture.

Setting: the hypotheses of inh:common-clock (chapters/05_inheritance.tex:1329-1373) on a binary
alphabet.  N = 3 agents with belief laws Ber(p_i), natural parameters theta_i = logit p_i, a shared
model-independent evaluator G_0 = Ber(sigmoid(theta_0)), self excluded from belief sources, uniform
rows over j != i (the Gibbs rows equal the prior because all costs of a row are equal), statistic
T(v) = v, moment M = sum_l p_l, and constant-output channels C[M] = Ber(c(M)), c(M) = sigmoid(a + b M).
From the proof of inh:common-clock (h_i = log q_i - log G_0 + log q_i - log C[M] + source term), the
Fisher (replicator) flow in natural parameters is
    theta_i' = -[(theta_i - theta_0) + (theta_i - a - b M) - b sum_l (p_l - c(M))],
and the null flow of inh:null-flow (peer costs deleted) is theta_i' = -(theta_i - theta_0).
  1. Verify the common-clock identity d/dt (theta_i - theta_j) = -2 (theta_i - theta_j) numerically.
  2. Selector (relabeling-equivariant, reads each agent's natural-parameter distance to the shared
     evaluator, d_i = |theta_i - theta_0|): join two current blocks when the difference of their mean
     d values first falls below delta.  Compare partition sequences P and P^0 on a grid around
     theta(0) = (-1, 1.2, 3).
     Case b = 0: the channel output is a constant law, so no agent's dynamics depends on any other
     agent, yet the null flow deletes the "peer" costs.  Case b = 1.5: genuinely coupled.
"""
import itertools
import numpy as np

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def field(theta, theta0, a, b):
    p = sig(theta)
    M = p.sum()
    c = sig(a + b * M)
    return -((theta - theta0) + (theta - (a + b * M)) - b * (p - c).sum())

def field0(theta, theta0):
    return -(theta - theta0)

def run(theta_init, f, delta, t_max=12.0, dt=2e-3):
    theta = np.array(theta_init, dtype=float)
    blocks = [[0], [1], [2]]
    seq = []
    t = 0.0
    prev = None
    def dvals(th):
        return np.abs(th - 0.0)
    while t < t_max and len(blocks) > 1:
        dv = dvals(theta)
        means = [dv[b].mean() for b in blocks]
        pairs = [(abs(means[i] - means[j]), i, j) for i, j in itertools.combinations(range(len(blocks)), 2)]
        gap, i, j = min(pairs)
        if gap < delta:
            new = blocks[i] + blocks[j]
            blocks = [b for k, b in enumerate(blocks) if k not in (i, j)] + [sorted(new)]
            seq.append(tuple(sorted(tuple(sorted(b)) for b in blocks)))
            continue
        k1 = f(theta); k2 = f(theta + dt / 2 * k1); k3 = f(theta + dt / 2 * k2); k4 = f(theta + dt * k3)
        theta = theta + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return seq

theta0 = 0.0
# 1. common clock
rng = np.random.default_rng(0)
th = rng.normal(size=3)
for a, b in ((2.0, 0.0), (2.0, 1.5)):
    v = field(th, theta0, a, b)
    print(f"1. (a,b)=({a},{b}): d/dt(theta_i - theta_j) + 2(theta_i - theta_j) residuals:",
          [f"{(v[i] - v[j]) + 2 * (th[i] - th[j]):.1e}" for i, j in ((0, 1), (0, 2), (1, 2))])

delta = 0.1
base = np.array([-1.0, 1.2, 3.0])
for a, b in ((2.0, 0.0), (2.0, 1.5)):
    f = lambda x: field(x, theta0, a, b)
    f0 = lambda x: field0(x, theta0)
    P = run(base, f, delta); P0 = run(base, f0, delta)
    print(f"2. (a,b)=({a},{b}) at theta(0) = {base.tolist()}: P = {P}; P0 = {P0}; differ: {P != P0}")
    differ = 0; total = 0
    for e in itertools.product((-0.05, 0.0, 0.05), repeat=3):
        x0 = base + np.array(e)
        total += 1
        differ += run(x0, f, delta) != run(x0, f0, delta)
    print(f"   grid of {total} perturbed initial configurations (+-0.05 per coordinate): P != P0 in {differ}")

# 3. finer neighborhood and the configurations where the coarse grid agreed
for a, b in ((2.0, 0.0),):
    f = lambda x: field(x, theta0, a, b)
    f0 = lambda x: field0(x, theta0)
    same = []
    for e in itertools.product((-0.05, 0.0, 0.05), repeat=3):
        x0 = base + np.array(e)
        if run(x0, f, delta) == run(x0, f0, delta):
            same.append(tuple(np.round(x0, 3)))
    print("3. coarse-grid configurations with P = P0 (b = 0):", same)
    cnt = 0
    for e in itertools.product((-0.02, 0.0, 0.02), repeat=3):
        x0 = base + np.array(e)
        cnt += run(x0, f, delta) != run(x0, f0, delta)
    print(f"   finer grid (+-0.02 per coordinate): P != P0 in {cnt}/27")

# 4. repaired control: frozen-environment null flow (every peer law held at its initial value in
#    every cost and channel of agent i; constant-output costs are kept).
def field_frozen(theta, theta_init, theta0, a, b):
    p = sig(theta); p_init = sig(np.array(theta_init))
    out = np.zeros(3)
    for i in range(3):
        # agent i sees peers frozen at initial values; its own law enters M through itself
        pp = p_init.copy(); pp[i] = p[i]
        M = pp.sum(); c = sig(a + b * M)
        # receiver term and the source term of agent i itself (other agents' costs depend on q_i via M)
        out[i] = -((theta[i] - theta0) + (theta[i] - (a + b * M)) - b * (pp - c).sum())
    return out

for a, b in ((2.0, 0.0), (2.0, 1.5)):
    f = lambda x: field(x, theta0, a, b)
    cnt = 0; tot = 0
    for e in itertools.product((-0.02, 0.0, 0.02), repeat=3):
        x0 = base + np.array(e)
        ff = lambda x, x0=x0: field_frozen(x, x0, theta0, a, b)
        tot += 1
        cnt += run(x0, f, delta) != run(x0, ff, delta)
    x0 = base
    ff = lambda x: field_frozen(x, x0, theta0, a, b)
    print(f"4. (a,b)=({a},{b}) frozen-environment control: P = {run(base, f, delta)}, P_frozen = {run(base, ff, delta)};"
          f" P != P_frozen on {cnt}/{tot} grid points (+-0.02)")
