"""XC2-FRAME check of the scope of prop:dyn-no-global-vfe (03_dynamics:375-385) against the
abstract sentence 'we prove that it is not the free energy of any fixed normalized model
whose recognition laws are the agent laws'.

The proposition holds the rows FIXED.  Part (a) uses unboundedness of E (E >= beta_ij KL(q_i||q_j)).
The construction chapters use the Gibbs-reduced potential E_red.  Two agents, binary belief
laws q_i = Bern(p_i), uniform reference, identity self channel (D_ii = 0) and identity peer
channel, priors pi_ii = pi_ij = 1/2, temperature T = 1:
   E_red = sum_i [ KL(q_i || u) - T log( pi_ii + pi_ij exp(-KL(q_i||q_j)/T) ) ].
(i)  E_red is bounded on the open square, so the boundedness argument of part (a) does not apply.
(ii) The mixed third derivative d_{p1} d_{p2}^2 E_red is nonzero at an interior point, whereas
     the same derivative of F(q_1 (x) q_2; o) vanishes identically for product recognition, so the
     argument of part (b) does extend to E_red when channels are fixed (one identity edge).
(iii) If every channel is source independent (constant output law nu), the fixed-row potential
     IS twice a product-model free energy plus a constant: E = 2 KL(zeta_1 (x) zeta_2 || Pi_1 (x) Pi_2) + c.
     So the identity-channel hypothesis of (b) is load-bearing.
"""
import sympy as sp
import numpy as np

p1, p2 = sp.symbols('p1 p2', positive=True)
T = sp.Integer(1)
def kl(a, b):
    return a * sp.log(a / b) + (1 - a) * sp.log((1 - a) / (1 - b))
half = sp.Rational(1, 2)
E_red = (kl(p1, half) + kl(p2, half)
         - T * sp.log(half + half * sp.exp(-kl(p1, p2) / T))
         - T * sp.log(half + half * sp.exp(-kl(p2, p1) / T)))
f = sp.lambdify((p1, p2), E_red, 'numpy')
g = np.linspace(1e-9, 1 - 1e-9, 2001)
P1, P2 = np.meshgrid(g, g)
vals = f(P1, P2)
print(f"(i)  sup of E_red over a 2001^2 grid reaching 1e-9 from the boundary: {np.nanmax(vals):.6f}")
print(f"     analytic bound 2 log 2 + 2 T log 2 = {4*np.log(2):.6f}")
d3 = sp.diff(E_red, p1, 1, p2, 2)
pt = {p1: sp.Rational(3, 10), p2: sp.Rational(7, 10)}
print(f"(ii) d_p1 d_p2^2 E_red at (0.3, 0.7) = {float(d3.subs(pt)):.6f}  (nonzero)")
pt2 = {p1: sp.Rational(1, 5), p2: sp.Rational(2, 3)}
print(f"     d_p1 d_p2^2 E_red at (0.2, 2/3) = {float(d3.subs(pt2)):.6f}")

# (iii) constant channels: belief and model sectors, product laws, finite alphabets
rng = np.random.default_rng(3)
K, Mdim = 3, 2
def rand_simplex(n):
    x = rng.random(n) + 0.1; return x / x.sum()
R = [rng.random((K, Mdim)) + 0.1 for _ in range(2)]
R = [r / r.sum() for r in R]
nu_q, nu_s = rand_simplex(K), rand_simplex(Mdim)
def kl_np(a, b):
    return float(np.sum(a * (np.log(a) - np.log(b))))
Pi = []
for r in R:
    w = np.sqrt(r * nu_q[:, None] * nu_s[None, :]); Pi.append(w / w.sum())
diffs = []
for _ in range(6):
    q = [rand_simplex(K) for _ in range(2)]; s = [rand_simplex(Mdim) for _ in range(2)]
    E = 0.0; Fprod = 0.0
    for i in range(2):
        zeta = np.outer(q[i], s[i])
        E += kl_np(zeta, R[i]) + kl_np(q[i], nu_q) + kl_np(s[i], nu_s)   # rows sum to one over the single peer
        Fprod += kl_np(zeta, Pi[i])
    diffs.append(E - 2 * Fprod)
print(f"(iii) E - 2 KL(product || Pi) over 6 random configurations: {np.round(diffs, 12)}")
print(f"      spread = {max(diffs) - min(diffs):.2e} (constant up to rounding)")
