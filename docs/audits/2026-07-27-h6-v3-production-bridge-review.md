# H6-Prediction v3 production-bridge review

Date: 2026-07-27
Scope: uncommitted H6-Prediction v3 Task 8--11 source candidate
Method: independent static review; no broad test suite or empirical gate run
Disposition: source blockers confirmed; remediation required before Task 12

## Confirmed findings

| ID | Severity | Finding | Required closure |
|---|---|---|---|
| H6B-01 | High | The public attempt runner treats an existing checkpoint as terminal and never hydrates a partial checkpoint. Only the last batch is published. | Persist declared partial boundaries, hydrate a fresh exact executable, skip persisted updates, and require process-loss A0/A5 resume to reproduce uninterrupted terminal bytes. |
| H6B-02 | High | Hydration authenticates attempt/runtime/layout but can accept a same-layout alternate-forward factory and checkpoint-owned optimizer values from another tuning cell. | Bind the planned attempt, selected tuning cell, closed factory identity, deterministic policy, and canonical seed-realized initial state before loading state or constructing the optimizer. |
| H6B-03 | High | `initialization_sha256` hashes config plus seed while actual module construction is deterministic and seed-independent. A0 replicates therefore begin identically. | Freeze a stateless CPU-float64 seed-to-parameter mapping and hash actual initialized model/recognition bytes. Shared permutation identity must remain independent of run seed. |
| H6B-04 | High | The emission-only endpoint computes the complete ELBO and filters factors afterward, executing operations excluded from the frozen objective and FLOP ledger. | Use a dedicated emission-only evaluator and require the exact ordered factor inventory before optimization and checkpoint publication. |
| H6B-05 | High | Ragged active receiver counts are omitted from the authenticated counter-consumption identity while objectives consume only active rows. | Bind exact active horizons and exact consumed normal bytes per example; make planner and live terminal identities agree. |
| H6B-06 | High | Validation boundaries are counted but no boundary validation record or ordered graph-free metric history is executed or persisted. | Publish authenticated boundary records and carry ordered metric/validation identities through partial and terminal checkpoints. |
| H6B-07 | High | Prior-feature receiver loops repeatedly clone and CPU-hash the complete generative model. At production shape this scales with examples, receivers, banks, and repeated recognition forwards. | Capture one stopped stateless mapping and authentication identity per declared phase/batch boundary; forbid complete-state cloning, CPU transfer, or hashing inside receiver loops. |
| H6B-08 | High | Direct exact A0 held-out scoring has no matching exact Prefix authority. Weighted-SMC bounded certificates certify another estimator path. | Produce a same-revision sibling direct-A0 certificate, bind it alongside the unchanged bounded set, and require it before any A0 target read. |
| H6B-09 | Incomplete | The three bounded integration nodes and exactly two CUDA nodes required by Task 12 do not yet exist. | Add the specified CPU integration fixture and CUDA resume/ownership nodes, then run only those bounded nodes. |

## Static positives

- The inspected source-prior path receives only the previously observed token
  prefix; no direct target-history flow was found.
- Phase ownership freezes the inactive module in the inspected engine path.
- The public production runner requires the installed runtime and refuses the
  synthetic CPU runtime. The dominant resource issue is repeated state
  cloning and hashing inside the recognition hot path.

## Source decision

The direct A0 path cannot inherit a weighted-SMC certificate. For identical
A0 particle emissions, weighted aggregation is equal to the direct
distribution in real arithmetic, but this does not make the estimator
identities equal and does not prove raw-byte identity after aggregation.
The July 27 normative section in
`docs/preregistrations/2026-07-21-h6-prefix-prediction.md` therefore adds a
sibling direct-exact A0 certificate while leaving
`BoundedPrefixCertificateSet` unchanged and prohibiting legacy conversion.

## Closure evidence required

This review is not closed by source edits alone. Closure requires:

1. focused machine-readable JUnit for each repaired seam;
2. an independent post-repair source review;
3. the bounded Task 12 integration nodes;
4. installed-CUDA identity verification and exactly the two frozen CUDA
   nodes;
5. a current-revision validated claim ledger.

No H6-Prediction PASS, held-out opening, H8 predecessor claim, or WikiText-103
scientific result is asserted by this review.
