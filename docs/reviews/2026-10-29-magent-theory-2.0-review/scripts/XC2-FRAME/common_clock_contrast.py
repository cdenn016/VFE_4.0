"""XC2-FRAME contrast check for inh:common-clock (05_inheritance:1329-1363).
Binary agents, log odds l_i, uniform evaluator, source sets exclude self, every channel equals
one shared target C.  Reciprocal potential E = sum_i [KL(q_i||u) + KL(q_i||C)], unit Fisher flow.
(A) additive-moment target as in the proposition: C = Bern(log odds f(M)), M = sum_i E_{q_i} z,
    f(M) = 0.7 M + 0.2 M^3.  The proposition predicts d/dt(l_i - l_j) = -2 (l_i - l_j) exactly.
(B) geometric-mean target: C = Bern(log odds mean(l)) -- shared by every receiver, but additive in
    the NATURAL parameters rather than in the moment.  No common clock is predicted.
The field is computed by central finite differences of E divided by the Bernoulli Fisher
information sigma'(l_i), independently of any hand-derived formula.
"""
import numpy as np

def sig(x): return 1 / (1 + np.exp(-x))
def kl(l, m):
    p, r = sig(l), sig(m)
    return p * np.log(p / r) + (1 - p) * np.log((1 - p) / (1 - r))
def E_A(l):
    M = np.sum(np.tanh(l / 2)); f = 0.7 * M + 0.2 * M ** 3
    return np.sum(kl(l, 0.0) + kl(l, f))
def E_B(l):
    return np.sum(kl(l, 0.0) + kl(l, l.mean()))
def field(E, l, h=1e-6):
    g = np.array([(E(l + h * e) - E(l - h * e)) / (2 * h) for e in np.eye(len(l))])
    s = sig(l); return -g / (s * (1 - s))

rng = np.random.default_rng(11)
for name, E in (("A additive moment", E_A), ("B geometric mean", E_B)):
    worst = 0.0
    for _ in range(20):
        l = rng.uniform(-3, 3, size=4)
        v = field(E, l)
        for i in range(4):
            for j in range(i + 1, 4):
                worst = max(worst, abs((v[i] - v[j]) + 2 * (l[i] - l[j])))
    print(f"{name:20s}: max over 20 random 4-agent states of |d/dt(l_i-l_j) + 2(l_i-l_j)| = {worst:.3e}")
