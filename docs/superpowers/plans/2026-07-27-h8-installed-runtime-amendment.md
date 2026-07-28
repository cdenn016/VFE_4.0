# H8 Installed-Runtime Amendment Implementation Plan

> Execute this plan in the isolated
> `codex/vfe4-installed-torch-wt103-buildout-20260727` worktree. Use
> `C:/anaconda/python.exe` for every command that imports Torch. Do not run a
> broad or full suite during the amendment.

**Goal:** Replace H8's unexecuted Torch 2.9.1 private-profiler pin with the
exact installed CUDA Torch contract, mint the v5 scientific protocol, and
prove behavioral compatibility with one tiny real-profiler preflight before
any H8 scientific workload.

**Architecture:** Keep `verification/h8_wire.py` as the standard-library
parent recovery authority and preserve the child's independent local mirrors.
Bind the v3 frozen config, v3 parent/child digest, v5 validation schema, exact
full Torch version, two source hashes, and descriptor-derived API hash across
config, preflight, runtime, provenance, fixtures, and documentation. Keep the
child-v2 envelope and internal v4 runtime-section shape unchanged.

**Verification discipline:** Each task runs only its named RED/GREEN nodes
with a unique `--basetemp` under `.verification/`. The real profiler preflight
runs once. Independent review follows GREEN. A closure ledger is created only
after the final amendment revision is committed so its artifact binding stays
current.

---

## Task 1: Freeze the executable profiler authority and protocol identities

**Files:**

- Modify: `verification/h8_wire.py`
- Modify: `vfe4/types/h8.py`
- Modify: `vfe4/config/schema.py`
- Modify: `verification/h8_protocol.py`
- Modify: `verification/h8_gate.py`
- Inspect: `verify_vfe4.py`
- Inspect: `verification/h8_runtime.py`
- Inspect: `verification/h8_budget.py`
- Inspect: `verification/h8_orchestrator.py`
- Inspect: `verification/h8_parent_authority.py`
- Test: `tests/unit/test_h8_parent_orchestrator.py`
- Test: `tests/unit/test_config.py`

### Step 1: Write the failing contract assertions

Update
`test_h8_v2_config_and_protocol_contract_are_complete_and_shared` without
renaming it; v2 still names the unchanged child envelope. Assert:

- `h8-validation-config-v3`;
- `vfe4.h8.parent-child-protocol.v3`;
- `h8-child-v2`;
- `h8-sparse-scale-v5`;
- full version `2.10.0.dev20251210+cu128`;
- both installed source hashes;
- exact 757-byte descriptor and derived SHA-256
  `2ee166166bab997499cc66da85146a031f458fbe0190a75b1a1a3ddea80efc38`.

Add a focused assertion that drifting any one of the four config profiler
identity fields is rejected by the protocol constructor. Update the existing
static-config test to expect v3 and the installed values.

### Step 2: Run RED

Run only:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h8_parent_orchestrator.py::test_h8_v2_config_and_protocol_contract_are_complete_and_shared `
  tests/unit/test_config.py::test_h8_static_protocol_and_h7_projection_are_exact_and_pure `
  -q --basetemp .verification/pytest-h8-runtime-task1-red
```

Expected: assertion failures showing the old v2/v4/2.9.1 identities.

### Step 3: Implement the canonical authority

In `verification/h8_wire.py`:

- replace the old version and source hashes;
- store the exact descriptor from the approved design as one ASCII string;
- derive the API SHA with `hashlib.sha256(descriptor.encode("ascii"))`;
- export the descriptor and hash.

Synchronize the public/type mirrors. Mint config v3 and protocol domain v3.
Before hashing the protocol, require exact equality for the version, both
source hashes, and API hash. Set `H8_VALIDATION_SCHEMA` to
`h8-sparse-scale-v5`. Inspect the five named downstream consumers for direct
v2/v4 assumptions and update only consumers with an actual identity
dependency; retain the deliberately internal v4 runtime-section function
names and shape.

### Step 4: Run GREEN

Repeat only the two Step 2 nodes with a new basetemp:
`.verification/pytest-h8-runtime-task1-green`.

### Step 5: Commit

Commit the executable authority and its focused tests.

## Task 2: Enforce exact child startup and preserve lossless evidence

**Files:**

- Modify: `verification/h8_child.py`
- Modify: `test_support/h8_runtime_fakes.py`
- Modify: `tests/unit/test_h8_allocation.py`

### Step 1: Write the failing full-version test

Add
`test_h8_profiler_pins_reject_full_torch_version_drift`. Build a fake Torch
module rooted in temporary profiler source files, monkeypatch only the
expected hashes for those fixture bytes, and prove:

- the child-local version, both source hashes, and descriptor hash initially
  equal the stable wire authority;
- the exact full version is accepted;
- the suffix-stripped version is rejected;
- a forged full version is rejected;
- either source-file hash drift is rejected.

Update the existing lossless child-v2 evidence test and fake runtime payloads
to the installed full version and current hashes.

### Step 2: Run RED

Run only:

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h8_allocation.py::test_h8_profiler_pins_reject_full_torch_version_drift `
  tests/unit/test_h8_allocation.py::test_h8_child_v2_retains_lossless_private_evidence_without_public_schema_drift `
  -q --basetemp .verification/pytest-h8-runtime-task2-red
```

### Step 3: Implement exact startup checks

Synchronize all child-local mirrors. Move the exact full-string version check
into `_verify_profiler_pins()` so every caller receives the same fail-closed
check. Remove base-version normalization from `_run_profiler()`. Preserve
raw-byte hashing, returned profiler evidence, the child-v2 envelope, and all
lossless join fields.

### Step 4: Run GREEN and commit

Repeat only the two Step 2 nodes with
`.verification/pytest-h8-runtime-task2-green`, then commit.

## Task 3: Bind preflight, source-only runtime, and provenance

**Files:**

- Modify: `verification/h8_preflight.py`
- Modify: `vfe4/artifacts/provenance.py`
- Modify: `verification/h8_gate.py`
- Test: `tests/unit/test_h8_preflight.py`
- Test: `tests/promotion/test_h8_gate.py`

### Step 1: Write the failing drift/provenance assertions

Extend `test_target_rejects_any_frozen_h8_protocol_drift` to perturb each
profiler identity independently and require rejection. Extend
`test_h8_payload_inventories_are_exact_and_private` to require the installed
full version in source-only environment and artifact provenance. The test must
exercise both consumers with a frozen config object; a source-text assertion
alone is insufficient.

### Step 2: Run RED

Run only:

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h8_preflight.py::test_target_rejects_any_frozen_h8_protocol_drift `
  tests/promotion/test_h8_gate.py::test_h8_payload_inventories_are_exact_and_private `
  -q --basetemp .verification/pytest-h8-runtime-task3-red
```

### Step 3: Implement bindings

- accept only `h8-validation-config-v3`;
- derive source-only environment version from the frozen config;
- compare provenance to the frozen config rather than a literal;
- recompute `H8_FROZEN_SECTION_SHA256` from the final canonical v3 H8 section.

### Step 4: Run GREEN and commit

Repeat only the two Step 2 nodes with
`.verification/pytest-h8-runtime-task3-green`, then commit.

## Task 4: Implement a bounded private-profiler compatibility inspector

**Files:**

- Modify: `verification/h8_child.py`
- Modify: `tests/unit/test_h8_allocation.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `verification/fixtures/h8_exact_test_nodes_v1.txt`

### Step 1: Add fake RED tests for the tiny inspector

Add `inspect_installed_h8_profiler_schema(torch)` beside the existing private
profiler helpers. It must:

- call the exact version/source pin verifier;
- run only a tiny CPU float64 tensor allocation/clone/mutation under the
  frozen profiler flags;
- read the nonempty private memory timeline and Kineto event tree;
- require exact four-tuple timeline rows, require every observed action to be
  a member of the frozen action enum, deterministically witness the minimum
  allocation/mutation actions, validate the complete four-action enum
  separately, require decodable complete TensorKey fields, and require the
  Allocation/TorchOp private event types;
- return a small immutable/canonical inspection record, not an H8 PASS.

Do not require the tiny workload to observe `PREEXISTING` or `DESTROY`, whose
appearance may depend on lifetime/GC behavior.

It must not call the H8 production operation graph, dense references,
correctness grid, 30-child orchestrator, or artifact publisher.

Use fake profiler objects in
`test_h8_profiler_schema_inspector_is_bounded_with_fakes` to test the
inspector's bounds and schema rejection without executing the installed
profiler during development. Add
`test_installed_h8_profiler_schema_preflight` as the sole real-runtime
integration node. Add it and Task 2's new full-version pin test—the only two
new exact-milestone nodes—to the H8 exact-node manifest.

### Step 2: Run fake GREEN only

Run only the fake inspector tests and the full-version pin test with a unique
basetemp. Do **not** run the real integration node in this task.

### Step 3: Commit without real-profiler evidence

Commit the tiny inspector, focused integration test, and exact-node inventory
change. The commit makes no behavioral compatibility claim yet.

## Task 5: Amend normative and planning documentation

**Files:**

- Modify: `docs/preregistrations/2026-07-21-h8-sparse-scale.md`
- Modify: `docs/superpowers/plans/2026-07-21-vfe4-h8-sparse-scale.md`
- Modify:
  `docs/superpowers/plans/2026-07-21-vfe4-post-h8-wikitext103-training.md`
- Modify: `docs/preregistrations/2026-07-25-post-h8-arm-gate-amendment.md`
- Modify: `README.md`

### Step 1: Preserve history and add the v5 amendment

Record that 2.9.1/v4 was superseded before any accepted H8 execution. Freeze
the installed full version, two source hashes, exact descriptor, descriptor
SHA, config v3, protocol v3, and validation protocol v5. State explicitly that
the two Python hashes do not identify compiled Kineto and that the real
preflight/scientific child supply behavioral evidence.

Update active H8 and post-H8 instructions to require v5 while retaining
historical v4 text only when labeled superseded. Do not rewrite prior measured
evidence because none exists.

### Step 2: Run static consistency checks

Use targeted `rg` searches for the superseded identities
`h8-validation-config-v2`, `vfe4.h8.parent-child-protocol.v2`, and
`h8-sparse-scale-v4`. Classify every remaining match as explicitly historical
or an error. Do not treat the intentionally retained `h8-child-v2` or internal
v4 runtime-section function names as drift. Run `git diff --check`. No pytest
run is needed for this documentation-only step.

### Step 3: Commit

Commit the preregistration and documentation amendment.

## Task 6: Independent review and bounded amendment closure

**Files:**

- Review all Task 1-5 diffs
- Create ignored evidence under `.verification/`

### Step 1: Independent review

Dispatch one implementation reviewer and one adversarial reviewer. Require
them to check:

- no Torch installation or environment mutation;
- full-version and source-byte checks are reachable;
- all four config identities are cross-bound;
- v5/config-v3/protocol-v3 are consistent;
- child-v2 and runtime-section structure are unchanged;
- the real preflight does not reach H8 scale;
- no active 2.9.1/v2/v4 split-brain remains.

Fix findings with only the directly affected focused test nodes.

### Step 2: Run one combined development check

Run only the focused fake/unit/promotion nodes from Tasks 1-4 in one process.
Exclude the real installed-runtime integration node. Do not run the
repository's broad/full suite.

### Step 3: Commit the final H8 selection

After review fixes are complete, set only `operations.h8.enabled=True` and its
existing exact authorization literal in the editable top-level `CONFIG`.
Commit that selection before generating any current-candidate JUnit,
predecessor artifact, profiler evidence, or closure ledger. Do not toggle the
tracked selection afterward.

### Step 4: Execute the real profiler exactly once in the candidate milestone

On the final clean activated revision, verify the Anaconda Torch/CUDA identity,
then run the frozen exact-node manifest once. This one JUnit run is the first
and only execution of
`test_installed_h8_profiler_schema_preflight`; reuse its evidence for
amendment closure and the H8 candidate chain. Do not run that node separately.

Expected: the tiny profiler test completes in seconds, the exact milestone has
no failures/errors/skips, and no scientific H8 child is launched.

### Step 5: Create and validate the closure ledger

After the final amendment commit, start the verification gate in closure mode,
record one claim that the installed-runtime amendment is internally bound and
behaviorally compatible, attach the current-candidate exact JUnit and reviewer
results, and validate the ledger. If evidence is missing, record
`INCONCLUSIVE`; do not launch scientific H8.

### Step 6: Proceed to H8

Only after Task 6 closes:

1. regenerate the H1-H7/H6-Prediction prerequisite chain in frozen order;
2. validate the H8 registry;
3. run the scientific H8 click launcher once.

WikiText-103 implementation begins only after the v5 H8 artifact and ledger
validate as PASS.
