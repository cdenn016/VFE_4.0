r"""L11 check s4: how much generator-prior mass leaves the principal-logarithm domain.

For G = GL+(K), exp: gl(K) -> GL+(K) is injective with invertible differential on
{w : every eigenvalue has |Im| < pi} (principal branch). Outside it the Frechet
derivative L(w, .) can be singular (eigenvalue differences in 2 pi i Z \ {0}; g11
prop:glink-sample-record(iii)) and distinct generators share a link. This measures the
probability of leaving that domain under an isotropic Gaussian base law N(0, s^2 I) on
gl(K) in Frobenius coordinates, the setting of MultiAgentELBO latent_links (G^fib = I).
"""
import numpy as np
rng = np.random.default_rng(3)
n = 200000
for K in (2, 4, 8):
    row = []
    for s in (0.25, 0.5, 1.0, 1.5, 2.0):
        W = s*rng.normal(size=(n if K < 8 else n//4, K, K))
        lam = np.linalg.eigvals(W)
        out = (np.abs(lam.imag) >= np.pi).any(-1).mean()
        row.append(f"s={s}: {out:.2e}")
    print(f"K={K}: P(some |Im eig| >= pi) ->", ", ".join(row))
