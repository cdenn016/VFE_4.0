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
`h8-sparse-scale-v5`.

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
full version in source-only environment/provenance and no second hardcoded
version.

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

## Task 4: Exercise the real installed private profiler once

**Files:**

- Modify: `verification/h8_child.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `verification/fixtures/h8_exact_test_nodes_v1.txt`

### Step 1: Add the tiny compatibility inspector and test

Add `inspect_installed_h8_profiler_schema(torch)` beside the existing private
profiler helpers. It must:

- call the exact version/source pin verifier;
- run only a tiny CPU float64 tensor allocation/clone/mutation under the
  frozen profiler flags;
- read the nonempty private memory timeline and Kineto event tree;
- require exact four-tuple timeline rows, the frozen action union, decodable
  complete TensorKey fields, and the required Allocation/TorchOp private event
  types;
- return a small immutable/canonical inspection record, not an H8 PASS.

It must not call the H8 production operation graph, dense references,
correctness grid, 30-child orchestrator, or artifact publisher.

Add `test_installed_h8_profiler_schema_preflight`. Add it and Task 2's new
full-version pin test—the only two new nodes—to the H8 exact-node manifest.

### Step 2: Run the real preflight once

Verify the authoritative runtime immediately before the node:

```powershell
& 'C:/anaconda/python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& 'C:/anaconda/python.exe' -m pytest `
  tests/integration/test_verify_vfe4.py::test_installed_h8_profiler_schema_preflight `
  -q --basetemp .verification/pytest-h8-runtime-task4-real
```

Expected: one PASS, no H8 production child, and a runtime measured in seconds,
not minutes.

### Step 3: Commit

Commit the tiny inspector, focused integration test, and exact-node inventory
change.

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

Use `rg` to confirm no active source literal still requires 2.9.1, config v2,
protocol v2, or H8 v4. Run `git diff --check`. No pytest run is needed for this
documentation-only step.

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

### Step 2: Run one combined amendment check

Run only the five focused existing nodes plus the new full-version and real
preflight nodes in one process, writing JUnit XML under `.verification/`.
Do not run the repository's broad/full suite.

### Step 3: Create and validate the closure ledger

After the final amendment commit, start the verification gate in closure mode,
record one claim that the installed-runtime amendment is internally bound and
behaviorally compatible, attach the focused JUnit and reviewer results, and
validate the ledger. If evidence is missing, record `INCONCLUSIVE`; do not
launch scientific H8.

### Step 4: Proceed to H8

Only after Task 6 closes:

1. commit H8 selection and the existing authorization literal;
2. generate one current-candidate exact-node JUnit;
3. regenerate the H1-H7/H6-Prediction prerequisite chain in frozen order;
4. validate the H8 registry;
5. run the scientific H8 click launcher once.

WikiText-103 implementation begins only after the v5 H8 artifact and ledger
validate as PASS.
