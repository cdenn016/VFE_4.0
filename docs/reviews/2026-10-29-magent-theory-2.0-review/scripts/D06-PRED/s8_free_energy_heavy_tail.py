r"""D06-PRED s8: a general-law cue model in which the exact posterior has infinite KL to the prior.

One presentation, cue U ~ Uniform(0,1), record O | U=u ~ N(0, sigma(u)^2) with
sigma(u) = u log(u)^2 / sqrt(2 pi) on (0, 1/e] and sigma(u) = 1/(e sqrt(2 pi)) on (1/e, 1).
At o = 0 the likelihood is l(0|u) = 1/(u log(u)^2) on (0,1/e] and e on (1/e,1), so
  Z(0) = int_0^1 l(0|u) du = 1 + (e - 1) = e  (finite, positive),
but E_Pi[log l] = (1/Z) int l log l du = +infinity, hence KL(Pi||mu) = E_Pi[log l] - log Z = +infinity.
With the substitution s = -log u, int_delta^{1/e} l du = int_1^S ds/s^2 and
int_delta^{1/e} l log l du = int_1^S (s - 2 log s)/s^2 ds = log S - 2 int_1^S log(s)/s^2 ds, S = -log delta.
The script evaluates both truncations with mpmath for S = 10, 10^3, 10^6, 10^12.
Consequence: the separated (energy - entropy) form of cmem:free-energy and the Gibbs cost f_a of cmem:gibbs
are infinity - infinity at the exact posterior, while eq:agent-free-energy gives F(Pi;0) = -log Z = -1.
"""
from __future__ import annotations

import mpmath as mp

mp.mp.dps = 30
for S in [10, 10 ** 3, 10 ** 6, 10 ** 12]:
    mass = mp.quad(lambda s: 1 / s ** 2, [1, S])
    ent = mp.quad(lambda s: (s - 2 * mp.log(s)) / s ** 2, [1, 10, 100, S] if S > 100 else [1, S])
    print(f"S = {S:>14}: int l du (tail part) = {mp.nstr(mass, 12)}, int l log l du (tail part) = {mp.nstr(ent, 12)}, log S = {mp.nstr(mp.log(S), 12)}")
print("Z(0) = 1 + (e - 1) =", mp.e)
