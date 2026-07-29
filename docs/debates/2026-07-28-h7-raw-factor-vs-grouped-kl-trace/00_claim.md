# Claim — h7-raw-factor-vs-grouped-kl-trace

**Mode:** implementation
**Panel:** lite
**Rounds:** 2
**Judging:** rubric
**Experts override:** none
**Evidence scope:** paths:VFE4 H7 ELBO implementation and theory
**Canon location:** embedded

## Claim

For H7, the canonical 13-slot `CompleteLanguageELBOFactorTrace` must retain
raw additive \(E_q[\log p]\) plus recognition-entropy semantics, while
positive grouped KL terms must be derived separately from the complete law
rather than relabeling those same raw slots.

## User context

The claim arose while implementing the missing H7 fixture-law factor-trace
builder. The current code requires the 13 slots to sum to the complete ELBO,
but another H7 path labels those slots as positive or signed grouped KL terms
while also requiring explicit entropy slots. The decision must preserve the
whitepaper's complete-ELBO equation, the independent-oracle comparison, and
the existing additive trace contract without compensating terms.
