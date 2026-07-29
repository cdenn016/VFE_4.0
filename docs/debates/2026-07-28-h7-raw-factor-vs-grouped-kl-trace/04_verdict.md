# Verdict — h7-raw-factor-vs-grouped-kl-trace

## Outcome

BLUE_WINS

## Decisive evidence

The binding source comparison is
`vfe4/objective/language_elbo.py:1122-1200` together with
`vfe4/objective/h7_covariance.py:1399-1434`. The trace factory copies the
exact H6 factor identities, values, and total and later rejects any change;
the H7 evaluator then maps those unchanged source values one-for-one into
identifiers ending in `_kl` while retaining the entropy slots. That operation
is a relabeling, not a KL derivation.

## Reasoning

Variational-inference canon permits both whole-objective representations:
\(\mathbb E_q[\log p(x,z)]+H(q)\) and expected log likelihood minus grouped
KL terms are algebraically equal [Blei, Kucukelbir, and McAuliffe 2017,
§2.2, Eq. 13 and the following display]. This equivalence does not make their
individual trace entries interchangeable, because
\(-\mathrm{KL}(q\Vert p)=\mathbb E_q[\log p]+H(q)\) absorbs the matching
entropy. The claim is scoped to the existing source-bound
`CompleteLanguageELBOFactorTrace`, not to every valid ELBO storage design.
The established complete-factor evaluator produces expected generative log
factors and recognition entropy separately
(`vfe4/objective/h5_complete.py:894-1009`), and the H7 wrapper preserves that
source record without regrouping it. The independent oracle supplies the
proper second representation: it computes \(K_0\), source log ratios, and
conditional Gaussian KLs from the complete laws and excludes its entropy
diagnostic from the grouped ELBO sum
(`verification/mp_oracles/h7_covariance.py:3742-3807`). Red's proposed trace
with new identities, explicit entropy ownership, and zero additive entropy
would be a reversioned grouped schema, not a valid reinterpretation of this
13-slot trace. The generic H6 protocol's missing representation tag is a
schema-enforcement gap, but it does not validate the current relabeling.

## Action

Keep the canonical 13-slot trace as the raw additive
\(\mathbb E_q[\log p]\)-plus-entropy record. Add an explicit raw
representation/provenance check so future providers cannot exploit the H6
typing gap. Build a separate H7 grouped record from the complete \(p\) and
\(q\) laws, store positive initial/source/transition KL quantities with an
explicit sign convention, assemble the grouped ELBO by subtraction, and keep
joint recognition entropy nonadditive in that grouped view. Replace the
current one-to-one `_kl` mapping and test equality among the raw total, grouped
law-derived total, monolithic total, and independent oracle.
