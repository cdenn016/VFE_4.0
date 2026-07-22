# VFE 4.0

This repository implements the bounded H1 foundation and H2 information--moment
representation gate for a deterministic, evaluation-only VFE 4.0 verification
run. H1 and H2 are the implemented ordered gate prefix; H3 through H8 remain
unimplemented.

The user-facing surface is `verify_vfe4.py`. Its top-level `CONFIG` dictionary
is editable, and the file can be run directly from an IDE or file association;
there are no command-line flags or parser. Importing it performs no work.

H1 evaluates one frozen `T=2`, scalar-state, scalar-model fixture in three
separately assembled ways:

1. a monolithic full-joint Gaussian-mixture ELBO;
2. a local conditional-factor ELBO decomposition; and
3. an independent NumPy evidence-minus-posterior-KL identity.

At reviewed revision `b736d21`, all three values are approximately
`-3.911522906174741`. The maximum pairwise residual is
`4.440892098500626e-16`; the largest pair-specific allowance is
`7.582209551464146e-13`. Every pair and every homologous local term is decided
against its own calibrated allowance. The exact-head click-run gate reports
`H1: pass` and publishes an atomic, manifest-checked artifact under `runs/`.

H2 keeps that fixture and its four positive source components unchanged. It
verifies each recognition and generative Gaussian component separately, then
uses the exact source weights to compare the direct PyTorch information
representation, the unchanged PyTorch H1 moment representation, and an
independently parsed NumPy dense-moment oracle. It never moment-projects the
source-marginal mixture or describes that mixture as one Gaussian. Natural
coordinates are `(h,-J/2)`, expectation coordinates are `(mu,M)` with
`M=Sigma+mu mu^T`, and `(mu,Sigma)` is the moment representation rather than
the Fisher-dual expectation-coordinate pair.

The frozen H2 envelope is `D <= 6`, `lambda_min(J) >= 1e-4`,
`lambda_max(J) <= 1e4`, `kappa_2(J) <= 1e6`, and `||mu||_inf <= 4`. Its
absolute error budget fixes `eps=np.finfo(np.float64).eps`,
`gamma(n)=n*eps/(1-n*eps)`, `C=256`, and `N(D)=8*D+32`; at the observed
`kappa_2 <= 42.35`, the preregistered descriptive pair budget is approximately
`3.86e-10 * scale`, while every decision uses its own invariant-specific
allowance. The one editable `CONFIG` requests `H1` then `H2`, and one run
publishes both validation payloads through a single atomic, manifest-checked
artifact directory.

There is no dataset, language-model training path, optimizer, parameter update,
or gradient calculation in H1. This bounded evaluation performs no
backpropagation, but the VFE 4.0 codebase is not claimed to be
backpropagation-free. H2 establishes only bounded componentwise representation
verification; it makes no optimizer, gradient, performance, prediction,
scaling, or later-gate claim. H3 through H8 remain unimplemented.
