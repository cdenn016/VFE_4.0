# Blue Rebuttal — h7-raw-factor-vs-grouped-kl-trace

## Concession

Red is right on two points. First, variational-inference canon permits both
whole-objective representations. Blei, Kucukelbir, and McAuliffe define the
ELBO as
\(\mathbb E_q[\log p(z,x)]-\mathbb E_q[\log q(z)]\), then rewrite the same
quantity as expected log likelihood minus
\(\mathrm{KL}(q(z)\Vert p(z))\)
([Blei, Kucukelbir, and McAuliffe 2017, §2.2, Eq. 13 and the immediately
following display](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf)).
Canon does not privilege a software storage layout.

Second, the generic H6 protocol does not mechanically prove which
representation a provider supplied. `LanguageElboExpectation.contribution`
returns an untagged tensor (`vfe4/objective/language_elbo.py:425-461`), and
`H6LanguageElboTerms` records partitions, factor identities, values, and the
additive equality without a `raw_factor`/`grouped_kl` discriminator
(`vfe4/types/h6.py:2436-2507`). A deliberately different upstream provider
could therefore place signed grouped values in those slots and zero the
entropy contributions while satisfying the present structural checks. That is
a real schema-enforcement gap.

## Core attack

Red's conclusion changes the scope of the claim. The claim is not that every
ELBO implementation must use raw factors, nor that the current dataclass alone
proves their meaning. It concerns one versioned, provenance-preserving object:
the canonical 13-slot `CompleteLanguageELBOFactorTrace`. H7 creates that object
from an exact `H6LanguageElboTerms`, copies its factor identities, values, and
total, requires the exact \(T=2\) 13-slot order, and rejects any later change
(`vfe4/objective/language_elbo.py:1107-1200`). Red's synthetic grouped provider
would create a *different upstream H6 trace*. It does not show that H7 may take
an already produced raw trace and rename its slots.

The algebraic equivalence red cites reinforces this distinction. For one
latent block,

\[
-\mathrm{KL}(q\Vert p)
=\mathbb E_q[\log p]+\mathcal H(q).
\]

Moving from the raw representation to the grouped representation therefore
requires combining the matching expected-log-generative and entropy pieces.
It is not a one-slot relabeling. The live H7 code performs no such
recombination: it maps each `model_source`, `state_source`,
`model_transition`, and `state_transition` slot one-for-one to an identifier
ending in `_kl`, maps the initial slot to `initial_joint_kl`, and separately
retains the two entropy slots
(`vfe4/objective/h7_covariance.py:1399-1434`). The generic-protocol
counterexample does not rescue that mapping.

Red's strongest implementation evidence—the independent grouped oracle—also
supports separate derivation. The oracle computes source log ratios and
negative conditional Gaussian KLs from the complete \(q\) and \(p\) laws
(`verification/mp_oracles/h7_covariance.py:3770-3787`), records joint
recognition entropy as a diagnostic, and explicitly excludes it from the
grouped complete-ELBO sum
(`verification/mp_oracles/h7_covariance.py:3801-3807`). That is the separate
grouped view required by the claim, not evidence that raw slots may be
renamed.

## Defense

The repository's established complete-factor evaluator fixes the raw lineage
that H7 is being asked to preserve. Its initial factor is an expected
generative log density, its source factors are sums of \(q\log p\), its
transition factors are expected Gaussian log densities, and recognition
entropy is evaluated separately
(`vfe4/objective/h5_complete.py:894-1009`). A mechanically independent test
reconstructs all raw factor values, including the separate entropy total, and
compares every stored value against that reconstruction
(`tests/unit/test_h5_complete_objective.py:182-244`,
`tests/unit/test_h5_complete_objective.py:320-338`).

The H6 record then requires every one of the seven named partitions to be
present and requires their decomposition to equal the language ELBO
(`vfe4/types/h6.py:2458-2486`). The H7 wrapper is intentionally an immutable
view of that source record, not a regrouping constructor. Under this existing
trace version, preserving raw meanings and deriving grouped KLs from the full
law is the only design that simultaneously preserves provenance, retains the
separate additive entropy, and agrees with the independent oracle.

Red does identify a repair worth adopting: encode the representation in the
trace schema so a future provider cannot exploit the untagged protocol. That
repair strengthens the claim's implementation. It does not validate the
current one-to-one `_kl` relabeling.

## Falsification conditions

This defense fails if any of the following is established:

1. The actual canonical H6 producer—not a synthetic protocol witness—is shown
   to emit signed grouped KL values under a binding versioned contract, with
   absorbed entropy excluded from the additive total.
2. `CompleteLanguageELBOFactorTrace` is reversioned as a grouped trace with new
   factor identities and values, an explicit representation discriminator,
   and recognition entropy marked nonadditive.
3. The H7 builder is shown to derive every grouped term from the complete
   \(p\) and \(q\) laws and to prove one-time entropy accounting, rather than
   copying a single raw slot into each `_kl` label.

None of those conditions is satisfied by the cited implementation.

The Research-vault pages `[[Evidence lower bound (ELBO)]]` and
`[[VFE Transformer Program]]` were consulted for program context only; the
defense above rests on the primary paper and live source.
