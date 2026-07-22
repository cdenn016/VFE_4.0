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

H2 is verified at implementation revision
`00de72b93ebcc504ef5652d11ad3012f80852aa0`. The revision-bound JUnit XML
contains 414 tests, 0 failures, 0 errors, and 0 skips; its SHA-256 is
`268902c66ab92955574526cd4bf1fcd7999611a88009d3c6edaa6ba8aa17a7b7`.
The one final click run published
`runs/verify-h1-h2-20260722T074944065126Z-cb17f1bb2893`, whose five manifest
hashes recompute and whose provenance binds the tested Git revision, dirty
digest, config hash, fixture hash, objective schema, and CPU float64
environment.

The H2 artifact reports 295/295 invariants and 282/282 per-quantity
comparisons passing, with zero obligations. Its direct-information, unchanged
H1-moment, and independent NumPy ELBO values are respectively
`-3.9115229061747425`, `-3.9115229061747407`, and
`-3.91152290617474`. The largest comparison residual is
`3.5527136788005009e-15` under its `5.4870414032774655e-09` allowance; the
largest residual/allowance ratio is approximately `5.018e-5`. All eight
precision-condition records remain inside the frozen envelope, and all four
exact negative controls are decisive, including direct `h`-as-`mu` semantic
records and an inverse audit with zero production violations and one detected
injected violation. Independent artifact, numerical, dependency, and
spec/reachability reviews found no remaining Critical or Important H2 issue.

There is no dataset, language-model training path, optimizer, parameter update,
or gradient calculation in H1. This bounded evaluation performs no
backpropagation, but the VFE 4.0 codebase is not claimed to be
backpropagation-free. H2 establishes only bounded componentwise representation
verification; it makes no optimizer, gradient, performance, prediction,
scaling, or later-gate claim. H3 through H8 remain unimplemented.
