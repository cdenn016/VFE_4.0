# VFE4 H8 Zero-Compute Preflight Design

## Purpose

H8 must not begin scientific execution until its exact-current-candidate
predecessors, shared JUnit identity, amended H6-Prediction evidence, H7
compatibility closure, runtime orchestration, and resource contract are all
ready. The click-to-run verifier currently attempts to resolve the H8
scientific configuration and then requires an already complete registry. That
path raises on ordinary missing prerequisites and cannot summarize the frozen
workload before execution.

Add a separate `h8_preflight` operation to `verify_vfe4.py`. It is an advisory,
metadata-only operation. It may read repository metadata and existing evidence,
but it must not import or invoke an H8 gate, correctness runner, runtime child,
profiler, model, trainer, or data loader. It must not publish an artifact or
write any file. Its result can never be a scientific H8 `PASS`.

## Click-to-run contract

The editable operation is:

```python
"h8_preflight": {
    "enabled": False,
    "authorization": None,
    "config": {
        "schema_version": "h8-preflight-config-v1",
        "operation": "H8-Preflight",
        "target_operation": "h8",
        "inspection_policy": "metadata_only",
        "write_artifact": False,
    },
}
```

The exact authorization phrase is
`AUTHORIZE_VFE4_H8_ZERO_COMPUTE_PREFLIGHT_V1`. The target scientific mapping is
the existing editable `operations["h8"]["config"]`; the preflight must not
duplicate it.

## Result contract

`verification/h8_preflight.py` owns an advisory `H8PreflightResult`, separate
from all gate-result types. The result contains:

- schema, operation, disposition, and the literal scientific status
  `not_evaluated`;
- current candidate identity and a canonical hash of the target H8 scientific
  mapping;
- ordered prerequisite records with states `missing`,
  `present_unvalidated`, `malformed`, `stale`, or `blocked`;
- exact workload and resource forecasts derived from the frozen target
  configuration;
- an execution-policy record proving that the preflight launches zero tests,
  training runs, scientific evaluations, profilers, and runtime children and
  writes zero artifacts;
- ordered obligations and a canonical result hash.

`blocked` means at least one prerequisite is missing, malformed, stale, or
explicitly blocked. `metadata_complete_unvalidated` means metadata is present
but has not been reopened by the scientific H8 verifier. Neither disposition is
a scientific success state.

## Prerequisite inspection

The preflight reports, in dependency order:

1. whether `.verification/active.json` is absent before a scientific run;
2. whether the frozen H8 preregistration exists as a regular file;
3. whether the exact
   `.verification/h8-current-candidate-<HEAD>-refs.json` registry exists and
   declares schema `h8-current-candidate-refs-v3`;
4. whether the registry declares one same-candidate JUnit identity;
5. H1--H5 evidence;
6. scorer-v2 H1-Prefix-Prior evidence;
7. independent H6-Prefix evidence;
8. amended H6-Prediction v2 evidence;
9. H7's exact three-record compatibility registry;
10. H7 evidence and pointer metadata;
11. a nonempty H8 runtime orchestrator; and
12. complete runtime cross-binding.

The preflight performs only structural and identity-level inspection. Existing
artifacts are `present_unvalidated`, never silently promoted to verified. A
missing registry yields a complete blocked report instead of an exception.
Symlinked control files are rejected.

The runtime checks parse source text without importing runtime modules. A
source-only runner that supplies empty correctness, production, profiler, and
control sections remains blocked. The explicit
`h8_complete_runtime_cross_binding_not_implemented` marker also remains
blocked.

## Frozen workload forecast

For the current H8 configuration, the preflight must report:

- 12 correctness cells, 36 source evaluations, 1,224 retained source endpoint
  records, 2,448 ordered-pair endpoint comparisons, and 72 wrong-path control
  decisions;
- 15 production children, three profiler children, and 12 isolated allocation
  controls, for 30 resource children with no retries;
- `N=129`, `b=40`, and `D=5,160`;
- 5,160 information-vector scalars (41,280 bytes);
- 411,200 scalars (3,289,600 bytes) in each precision, factor, and selected
  inverse category;
- at most 1,600 local-workspace scalars (12,800 bytes);
- a forbidden dense population matrix of 26,625,600 scalars
  (213,004,800 bytes);
- per-child caps of 60 seconds, 128 MiB incremental process memory, 64 MiB
  Torch population, RHS width 40, and sample width one.

The 1,800-second sequential resource-child ceiling is arithmetic, not a runtime
prediction, and excludes the uncapped correctness parent. Total H8 wall time is
therefore unavailable until measured.

## Bootstrap repairs

The preflight target cannot currently resolve on Windows for two independent
reasons:

- raw-byte-pinned H1/H7 JSON fixtures are checked out with CRLF bytes; mark the
  three H7-consumed fixtures `-text` and restore their committed LF bytes;
- the centered fixed-decoder stabilizer is a `diagonal_base` action, but
  `H7TrialSpec` currently expects `internal_product`; include
  `matrix_fixed_decoder_stabilizer` in the diagonal-base profile family.

These repairs change no scientific fixture values or configured regime.

## Testing boundary

Use only exact, focused nodes:

- the existing raw H7 fixture parser/config test;
- unit tests for missing, malformed/stale, and present-unvalidated preflight
  metadata plus exact workload arithmetic;
- one click-run integration test that forbids scientific dispatch and verifies
  no filesystem mutation;
- the generic click-run dispatcher contract.

No full suite, training run, H8 correctness grid, profiler, or resource child is
part of this change.
