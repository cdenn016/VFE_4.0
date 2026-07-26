# H8 Zero-Compute Preflight and Workload Audit

Date: 2026-07-25
Base revision: `78ce4967430fde8f541c952d92d1e8c51bf17d4a`
Method: parallel static source review plus metadata-only preflight; no H8
correctness, production, profiler, training, or data workload was run.

## Outcome

H8 is `blocked`; its scientific status is `not_evaluated`. The repository has
the H8 sparse backend, correctness producer, runtime child, budget decoder, and
source-only gate assembly, but it does not yet have the parent orchestration
and complete runtime cross-binding that can supply a PASS-capable H8 record.
The current runner explicitly supplies empty correctness, production,
profiler, and control tuples, and the gate retains
`h8_complete_runtime_cross_binding_not_implemented`.

The first live preflight observation on the implementation branch bound:

- candidate dirty digest
  `af73a110e0ed55b524f7c837fc28057606447b4d7c2ee17fdee621996b4923ec`;
- candidate source identity
  `bf59690c6339de6cffe4caf89e8d3aed3d5a165b8be5b94170c8f9aec1d5b461`;
- target scientific-config SHA-256
  `1a730a4e80c7a76eb22a06b2b7d05a399657020fec67d2691b00fb4cbdb25529`;
- disposition `blocked` and scientific status `not_evaluated`.

That dirty identity is an audit snapshot, not a reusable evidence identity.
Committing any source changes necessarily creates a new candidate.

## Bootstrap defects repaired

Two defects prevented even pure H8 target resolution on Windows:

1. `h1_v1.json`, `h7_v1.json`, and `h7_density_probes_v1.json` were checked
   out with CRLF bytes despite raw-byte SHA-256 contracts. They are now marked
   `-text`, and their committed LF bytes reproduce the frozen hashes
   `388e38...583b`, `d2ed12...66d4`, and `4857af...82c6`.
2. The centered fixed-decoder stabilizer is represented by a
   `diagonal_base` action, while `H7TrialSpec` previously classified every
   non-scalar/non-diagonal profile as `internal_product`.
   `matrix_fixed_decoder_stabilizer` is now in the diagonal-base family.

No fixture value or scientific configuration regime changed.

## Exact H8 forecast

The frozen correctness grid has 12 cells:
`T in {1,2,4,8}` by dimension in `{1,2,4}`. It implies:

- 36 source evaluations;
- 1,224 retained source endpoint records;
- 2,448 ordered-pair endpoint comparisons;
- 72 wrong-path control decisions.

The resource protocol has 15 production children, three profiler children, and
12 isolated allocation-control children: 30 children with no retry.

The production layout is `N=129`, `b=40`, `D=5,160`. Exact float64 storage
arithmetic is:

| Category | Scalars | Bytes |
|---|---:|---:|
| Information vector | 5,160 | 41,280 |
| Precision | 411,200 | 3,289,600 |
| Factor | 411,200 | 3,289,600 |
| Selected inverse | 411,200 | 3,289,600 |
| Maximum local workspace | 1,600 | 12,800 |
| Forbidden dense population matrix | 26,625,600 | 213,004,800 |

Per-child protocol caps are 60 seconds, 128 MiB incremental process HWM,
64 MiB live Torch population, RHS width 40, and sample width one. Summing 30
timeout caps gives 1,800 seconds, but this is not a runtime estimate and
excludes the uncapped correctness parent. Total wall time remains unknown until
measured.

## Predecessor workload census

### H1-Prefix-Prior scorer v2

The frozen fixture has two history cases and two suffix controls. It requires
four production scorer calls, two independent scorer calls, and eight
objective-route evaluations, with quadrature orders 21 and 17. It has no seed
grid or child processes.

### H6-Prefix

Authorized-full currently repeats four particle levels
`(128,256,512,1024)` for every semantic profile. Per semantic profile:

- 9,720 small plus 4,096 validation cases = 13,816 cases;
- five prediction calls per case = 69,080 calls;
- across four particle levels, 55,264 cases, 276,320 prediction calls, and
  132,633,600 nominal particle-call units.

If all 12 prediction endpoints require distinct semantic profiles, this rises
to 663,168 cases, 3,315,840 prediction calls, and 1,591,603,200 nominal
particle-call units. The exact profile count is not yet frozen because the
editable H6-Prefix config is empty.

The gate also fails to pass frozen source-mask observations into
`run_dynamic_prefix_checks`; those inventories currently execute zero
observations. This is a correctness blocker, not merely a performance issue.

### H6 finite-SMC accuracy

The frozen grid is four fixtures by 512 seeds by 256 particles over horizon
six:

- 2,048 SMC sequence runs;
- 12,288 next-token calls;
- 524,288 particle trajectories;
- 3,145,728 particle-position propagation units;
- 76 error cells with 512 observations each.

The current predictor revalidates immutable fixture/oracle state on every
next-token call. Caching one validated immutable fixture/oracle record per
fixture can remove this redundancy without changing the statistical design.

### H6-Prediction

The preregistered tuning phase has 72 quarter-pass runs, or 18 equivalent
corpus passes. Confirmatory testing has 96 two-pass runs, or 192 passes. The
total is 168 training attempts and 210 equivalent corpus passes.

The held-out SMC opening requests 24,576 corpus records and 11,796,480 particle
streams before multiplying by per-record target positions. This is the
dominant CPU workload. Current frozen decisions consume only PRIMARY and
OBJECTIVE comparisons even though the source requires full-test SMC at all 12
endpoints.

### H7

H7 contains eight trials, 12 negative controls, 486 matrix density-probe pairs,
and eight scalar probe pairs, for 494 probe pairs. The independent oracle uses
100 decimal digits and Gauss-Hermite orders 41 and 51. It has no repetition
grid or child processes.

## Current prerequisite states

The metadata-only preflight reports:

| Prerequisite | State |
|---|---|
| Active verification marker clear | `blocked` |
| H8 preregistration | `present_unvalidated` |
| Exact H8 registry v2 | `missing` |
| Same-candidate JUnit | `missing` |
| H1--H5 evidence | `missing` |
| H1-Prefix-Prior scorer-v2 evidence | `missing` |
| Independent H6-Prefix evidence | `missing` |
| Amended H6-Prediction v2 evidence | `missing` |
| H7 compatibility registry | `missing` |
| H7 evidence/pointer | `missing` |
| H8 runtime orchestrator | `blocked` |
| Complete H8 runtime cross-binding | `blocked` |

The active marker belongs to the current revision-bound verification session.
It must be closed before any scientific H8 execution; its presence is expected
during this code-verification task.

## Required amendments before expensive execution

1. Repair H6-Prefix source-mask observation plumbing.
2. Amend H6-Prefix so exhaustive structural/cache invariance runs at one
   representative particle count, with only estimator-dependent checks on a
   frozen stratified particle ladder. Do not silently change the preregistered
   contract.
3. Cache immutable finite-SMC fixture/oracle validation once per fixture.
4. Amend H6-Prediction before opening the test set: either restrict held-out
   SMC to endpoints consumed by frozen decisions or preregister a
   validation-powered sequential design.
5. Implement the H8 parent runtime orchestrator and complete cross-binding
   before producing an H8 registry or attempting resource children.

## Prudent execution order

After the amendments are frozen:

1. close H1--H5 and scorer-v2 H1-Prefix-Prior on one candidate/JUnit identity;
2. close independent H6-Prefix and finite-SMC accuracy;
3. produce amended H6-Prediction v2 evidence;
4. build H7's three-record compatibility registry and close H7;
5. build the sole exact H8 registry v2;
6. run the zero-compute H8 preflight and require no blocked/missing/stale state;
7. execute H8 correctness and resource protocols once;
8. proceed to WikiText-103 integration, recording, and figure generation only
   after exact H8 PASS.

No full suite or expensive scientific workload is justified before these
conditions are met.
