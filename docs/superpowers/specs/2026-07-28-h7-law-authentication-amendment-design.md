# H7 Law-Authentication and Exact-ELBO Amendment

**Status:** Required correctness remediation within the authorized H2--H8
buildout

**Scope:** Replace synthetic or structurally authenticated H7 objective values
with values derived from the exact frozen H1/H7 complete-law snapshots. This
amendment does not authorize H7 calibration or a runtime H7 status decision.

## Problem

The existing Task-5 boundary has two distinct provenance layers but treats
them as one:

1. `BuiltArm.evaluate_complete_language_elbo` authenticates the H6 endpoint
   assembly route.
2. The supplied `LanguageElboExpectation` determines the numerical factor
   values.

The first layer is now authenticated by a live factory-issued arm and a
capture-only receipt. It does not prove that an arbitrary expectation derived
its values from the frozen H7 law.

The ordinary A5 fixed source prior is also not the H7-v1 law. Its receiver-2
DAG row has parents `(0, 1)`, whereas H7-v1 freezes the singleton predecessor
support `(1,)` with probability one. H1 uses the dense row but has nonuniform
fixed source probabilities, while the ordinary factory initializes uniform
logits. Borrowing either ordinary source-factor identity would therefore
authenticate a different law.

Finally, the current Task-5 adapter relabels raw `E_q log p` source and
transition factors as positive KLs. A positive grouped KL requires both the
recognition entropy and generative cross-entropy operands; it cannot equal a
raw generative factor by itself.

## Frozen Mathematical Semantics

H7 uses the augmented joint with explicit discrete source labels:

```text
q(s, y) = q(s) q(y | s)
```

The exact raw representation has 13 slots: the initial expected log
generative density, then model source, model transition, state source, state
transition, emission, and recognition entropy for receivers 1 and 2.

Because H6 has no `entropy@0` slot, the chronological entropy ownership is
frozen as:

```text
entropy@1 =
  H(initial_joint)
  + H(model_source@1) + H(model_transition@1)
  + H(state_source@1) + H(state_transition@1)

entropy@2 =
  H(model_source@2) + H(model_transition@2)
  + H(state_source@2) + H(state_transition@2)
```

Every child is retained in an immutable ownership record. Moving entropy
between the two raw slots is forbidden even though it would leave the total
unchanged.

The grouped representation has two expected-emission terms and nine
nonnegative KL terms: the joint initial KL plus model/state source and
transition KLs at both receivers. Recognition entropy is diagnostic-only in
this representation. Positive KL values enter the grouped ELBO with sign
minus.

The builder requires agreement among:

- the 13-slot raw sum;
- the 11-term grouped sum;
- the monolithic augmented-joint expectation; and
- the independent high-precision oracle, after explicitly negating the
  oracle's historically signed negative-KL fields.

## Chosen Provenance Design

Three capabilities remain separate:

1. `training.arms` authenticates a live factory-issued `BuiltArm`.
2. `objective.language_elbo` authenticates one H6 assembly receipt.
3. the H7 law-evaluation module authenticates one role-bound derivation from
   an exact `H7LawPairSnapshot`.

The H7 assembly factory constructs the exact fixed source structure and logits
before predictive-boundary, parameter-role, and arm-registration capture. It
never mutates an issued arm.

The law-evaluation evidence is bound to the trial, law pair, action, role,
complete-law snapshot, raw fixture, source structure, source-law identities,
assembly receipt, ordered raw and grouped operands, entropy ownership,
quadrature order, and derivation route. Structurally identical, copied,
pickled, cross-role, cross-trial, and unregistered records are rejected.

Grouped operand hashes identify complete-law components and assembled
recognition moments. Raw factor IDs may prove equality with the authenticated
H6 assembly, but they cannot serve as grouped-law provenance.

## Failure Semantics

- An ordinary dense/uniform A5 arm presented as H1 or H7 law evidence fails.
- A generic expectation may receive an assembly receipt but cannot receive H7
  law-evaluation evidence.
- Missing entropy children, a changed ownership partition, negative KL,
  raw-factor-as-KL provenance, cross-role replay, or disagreement among raw,
  grouped, monolithic, and oracle values fails before gate construction.
- A missing authorization, calibration value, predecessor, or independent
  oracle comparison remains `INCONCLUSIVE`; it is never converted into a
  synthetic PASS.

