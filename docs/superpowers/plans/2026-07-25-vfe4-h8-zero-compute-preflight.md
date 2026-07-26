# VFE4 H8 Zero-Compute Preflight Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a click-to-run, read-only H8 preflight that reports exact blockers
and frozen workload/resource arithmetic without launching scientific work.

**Architecture:** Keep `verify_vfe4.py` as the sole click launcher and add one
strict `h8_preflight` operation. Put all read-only inspection and advisory
result types in a standard-library-only `verification/h8_preflight.py` module.
Repair the two H7 configuration-resolution defects that otherwise prevent even
metadata preflight of the target.

**Tech Stack:** Python 3.14, standard-library dataclasses/JSON/AST/pathlib,
pytest, existing VFE4 editable configuration dictionaries.

---

### Task 1: Restore H7 raw-byte portability

**Files:**
- Modify: `.gitattributes`
- Verify: `vfe4/validation/fixtures/h1_v1.json`
- Verify: `vfe4/validation/fixtures/h7_v1.json`
- Verify: `vfe4/validation/fixtures/h7_density_probes_v1.json`
- Test: `tests/unit/test_h7_fixture.py`

- [ ] Record the existing exact H7 parser failure as RED.
- [ ] Mark all three H7-consumed, raw-byte-pinned fixtures `-text`.
- [ ] Mechanically restore the three paths from their committed LF blobs.
- [ ] Verify all observed SHA-256 values equal the frozen constants.

### Task 2: Correct the centered stabilizer action family

**Files:**
- Modify: `vfe4/types/h7.py`
- Test: `tests/unit/test_h7_fixture.py`

- [ ] Preserve the existing failing exact fixture/config node.
- [ ] Classify `matrix_fixed_decoder_stabilizer` as `diagonal_base`.
- [ ] Re-run only the exact H7 fixture/config node and capture JUnit evidence.

### Task 3: Specify the advisory preflight in tests

**Files:**
- Create: `tests/unit/test_h8_preflight.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `tests/integration/test_click_run_launchers.py`

- [ ] Require a blocked structured result when the current registry is absent.
- [ ] Require exact H8 workload/resource arithmetic and null measurements.
- [ ] Require malformed, stale, and v1 registries to remain nonauthorizing.
- [ ] Require a structurally complete v2 registry to remain
  `present_unvalidated`, never scientific `PASS`.
- [ ] Require click execution to launch no scientific path and write no file.
- [ ] Run one exact preflight test as RED before implementation.

### Task 4: Implement standard-library-only metadata inspection

**Files:**
- Create: `verification/h8_preflight.py`

- [ ] Add strict request/candidate validation and canonical JSON hashing.
- [ ] Add immutable prerequisite and advisory result records.
- [ ] Inspect the active marker, preregistration, exact registry schema and
  candidate, direct predecessors, H7 transitive compatibility, and H6
  Prediction v2 declarations without importing gate/runtime modules.
- [ ] Parse H8 runner source to report source-only empty runtime sections and
  the unimplemented cross-binding marker.
- [ ] Derive exact correctness, storage, child-count, and cap arithmetic from
  the target configuration.
- [ ] Guarantee no artifact publication or filesystem write.

### Task 5: Wire the click-to-run operation

**Files:**
- Modify: `verify_vfe4.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `tests/integration/test_click_run_launchers.py`

- [ ] Add the disabled `h8_preflight` editable config dictionary.
- [ ] Add the exact authorization phrase and strict operation inventory.
- [ ] Capture current source identity, call the advisory inspector, and print
  its structured report.
- [ ] Keep the target H8 scientific config in one place.
- [ ] Run only the exact unit, click integration, and generic dispatcher nodes.

### Task 6: Close revision-bound verification

**Files:**
- Update locally: `.verification/ledger.json`
- Produce locally: focused JUnit XML files under `.verification/`

- [ ] Commit the source/docs/tests on the dedicated branch without pushing.
- [ ] Restart the claim ledger at the committed revision.
- [ ] Re-run the exact H7, preflight unit, and click integration nodes with
  `C:/anaconda/python.exe`; do not run a full suite.
- [ ] Record one claim per code behavior with mechanical and independent static
  views.
- [ ] Validate the ledger and report the exact remaining blockers and amended
  execution order.
