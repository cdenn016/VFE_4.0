# VFE 4.0

This repository implements the bounded H1 foundation, H2 information--moment
representation gate, H3 structured-recognition adequacy gate, and one coupled
H1--H5 verification prefix that integrates the already frozen H4 cost artifact
with the H5 update-coherence gate. The implemented ordered prefixes are H1,
H1/H2, H1/H2/H3, and H1/H2/H3/H4/H5. H4 is not available as a standalone
prefix, and H6 through H8 and training remain deferred.

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
allowance.

H3 uses two separately authored and separately hashed, source-free,
four-dimensional Gaussian fixtures. The coupled law has a frozen analytic
factorized reverse-KL gap; the control removes the preregistered couplings and
has a diagonal exact posterior. Both the structured full-SPD and fine
factorized diagonal recognition arms start from the same zero mean and identity
precision. Only H3 recognition parameters enter its reverse-mode autograd
graph; the frozen generative factors and independent NumPy oracle do not.
Operand-local absolute allowances and two signed three-way thresholds determine
PASS, FAIL, or INCONCLUSIVE without relative tolerances or blanket `allclose`.
This bounded H3 protocol does not establish H4 cost, H5 update coherence, H6
prediction, H7 frame covariance, or H8 sparse scaling.

The one editable `CONFIG` now requests H1 through H5. One run captures each
required fixture exactly once, evaluates H1, H2, H3, H4, then H5, and publishes
separate `validation/h4.json` and `validation/h5.json` payloads in one atomic,
manifest-checked artifact directory. The runner serializes the preexisting
typed H4 validation artifact; it does not reconstruct H4 status or measurement
content. H4 and H5 statuses remain separate and may differ. Selecting H1,
H1/H2, or H1/H2/H3 and removing the later conditional sections preserves the
shorter compatibility surface and does not capture, evaluate, or publish H4/H5
data. The theory sources are
`Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex` and
`Manuscripts/MAgent_exact_elbo_whitepaper.tex`.

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
verification. H3 uses reverse-mode autograd internally to obtain gradients of
the direct ELBO with respect to its recognition mean and precision-Cholesky
parameters; this is ordinary backward differentiation through a forward ELBO
computation, not a claim of backpropagation-free learning. Neither implemented
gate makes a general optimizer, performance, prediction, scaling, or
later-gate claim.
