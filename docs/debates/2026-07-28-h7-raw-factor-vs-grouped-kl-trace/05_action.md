# Action — h7-raw-factor-vs-grouped-kl-trace

**From verdict:** BLUE_WINS
**Reconciliation rule:** panel-lite single-judge rubric

## Recommended action

1. Keep the canonical 13-slot `CompleteLanguageELBOFactorTrace` as the raw
   additive \(E_q[\log p]\)-plus-recognition-entropy record.
2. Add an explicit raw representation and provenance discriminator so a
   future H6 provider cannot substitute grouped terms under the same schema.
3. Replace H7's current one-to-one `_kl` relabeling with a separate grouped
   record derived from the complete \(p\) and \(q\) laws.
4. Store positive initial, source, and transition KL quantities with an
   explicit sign convention; assemble the grouped ELBO by subtraction and
   keep joint recognition entropy nonadditive in that grouped view.
5. Require equality among the raw 13-slot total, the grouped law-derived
   total, the monolithic complete-law total, and the independent oracle.

## Follow-up debates

None. The exact entropy-slot partition and provenance-field schema are
implementation design details to freeze in focused tests before the H7
fixture-law builder is added.
