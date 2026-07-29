# Red Rebuttal — h7-raw-factor-vs-grouped-kl-trace

## Concession

Blue establishes the defect in the present one-to-one relabeling. The live H5
producer evaluates the initial, source, and transition slots as expected
generative log factors and evaluates recognition entropy separately
(`vfe4/objective/h5_complete.py:888-1009`). H7 then copies those stored values
into names ending in `_kl` without evaluating a log ratio or Gaussian KL
(`vfe4/objective/h7_covariance.py:1399-1433`). By contrast, the independent
oracle forms source log ratios and conditional Gaussian KL contributions from
the laws and omits its entropy diagnostic when assembling the grouped local
ELBO (`verification/mp_oracles/h7_covariance.py:3742-3807`). Since
\(-\mathrm{KL}(q\Vert p)=E_q[\log p]+H(q)\), adding the matching entropy again
would double count it [Blei, Kucukelbir, and McAuliffe 2017 §2.2]. I therefore
grant that the current `_kl` relabeling is not a valid grouped-KL derivation.

## Core attack

Blue does not establish the stronger necessity claim that every grouped view
must be recomputed “separately from the complete law.” Canon permits exact
algebraic regrouping of raw expected-log factors with their matching entropy
components; law-level recomputation is a strong independent check, but it is
not the only mathematically valid derivation [Blei, Kucukelbir, and McAuliffe
2017 §2.2]. The live interface leaves this distinction untyped:
`LanguageElboExpectation.contribution` returns an opaque scalar tensor
(`vfe4/objective/language_elbo.py:425-447`), the assembler checks the canonical
slot order and additive equality (`vfe4/objective/language_elbo.py:791-876`),
and `CompleteLanguageELBOFactorTrace` copies the source IDs, values, and total
without a representation-kind or entropy-ownership field
(`vfe4/objective/language_elbo.py:1107-1188`). The cited H5 behavior proves the
provenance of that producer; it does not turn “derive again from the complete
law” into a theorem of the trace type.

The falsifiable test is an exact regrouping with new identities, explicit
entropy ownership, and a declared sign convention. If that regrouping agrees
with the raw total and the independent oracle for every admitted law, the
claim’s exclusive derivation requirement fails even though the present
one-to-one relabeling remains invalid. The oracle itself shows why sign must be
declared: its source entries are \(E_q[\log p-\log q]\), its transition entries
are negated positive KLs, and only \(K_0\) is retained as a positive divergence
before subtraction (`verification/mp_oracles/h7_covariance.py:3742-3807`).

## Defense

The schema-under-specification objection survives Blue’s producer trace. The
factory accepts arbitrary signed values so long as all partitions exist and
their sum equals the reported total (`vfe4/types/h6.py:2451-2491`,
`vfe4/types/h6.py:2509-2551`). The H7 unit fixture exploits that freedom by
moving value between an emission slot and an entropy slot while preserving the
same total (`tests/unit/test_h7_complete_objective.py:137-170`), and the H7
result contract later verifies only that grouped records bind and sum the same
13 stored values (`vfe4/types/h7.py:4227-4279`). A `_kl` suffix therefore
cannot certify KL semantics, just as a partition label alone cannot certify an
entropy allocation.

This objection supports an explicit raw-versus-grouped representation marker
and separate grouped identities. It does not rescue the current relabeling. I
cannot falsify the claim’s narrow operational core: the existing
provenance-bound 13-slot trace should retain its producer’s raw meanings, and
any grouped representation must absorb each entropy contribution exactly once.
I can falsify only the added assertion that recomputation from the complete law
is the sole valid way to derive that grouped representation.
