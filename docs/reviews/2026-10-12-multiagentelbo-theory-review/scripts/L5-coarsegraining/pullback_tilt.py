r"""Check of the sentence in lem:rg-interaction-step that only at g=0 may the tilted operator be
replaced by U: for a deterministic channel chi and psi = f o chi (a pullback direction, g != 0),
the tilted reverse kernel equals the untilted one."""
import numpy as np
from hoeffding_growth import BinarySpace, block_channel, det_kernel, reverse_kernel, tilted, rng

S0 = BinarySpace(4, rng.uniform(0.2, 0.8, size=4)); S1 = BinarySpace(2, rng.uniform(0.2, 0.8, size=2))
mbar = rng.dirichlet(np.ones(16))
K = det_kernel(lambda y: y[0] * y[1], 2)
C = block_channel(S0, S1, [[0, 1], [2, 3]], [K, K])
Cd, _ = reverse_kernel(mbar, C)
f = rng.normal(size=4)
chi = C.argmax(1)
psi = f[chi]
g = S0.hd(psi)
print("||g||_G =", S0.Gnorm(g), " max|Cdag_psi - Cdag| =", np.abs(tilted(Cd, psi - psi.mean()) - Cd).max())
