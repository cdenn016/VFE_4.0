# H8 Installed-Runtime Amendment Design

**Date:** 2026-07-27
**Status:** Approved for implementation
**Scope:** H8 profiler reproducibility contract only; no VFE objective, model,
numerical method, scientific threshold, or workload change.

## Context

H8 was implemented before its first scientific execution with a private
PyTorch profiler contract pinned to `torch==2.9.1`. The machine's real
CUDA-enabled working interpreter is `C:/anaconda/python.exe`, whose installed
runtime is:

- Python `3.12.7`
- PyTorch `2.10.0.dev20251210+cu128`
- CUDA available
- CUDA runtime `12.8`
- NVIDIA GeForce RTX 5090

The private profiler surfaces required by H8 are present in that installed
runtime, but their source bytes differ from the unexecuted 2.9.1 pin:

- `torch/profiler/_memory_profiler.py` SHA-256
  `22de3b0790907b90053af829ebf1bff0b6add2745ac0381ec7de78812edacb47`
- `torch/profiler/profiler.py` SHA-256
  `543430b2e9b24df777f86415865fee250b35e3444a80920bcca0e8889b917956`

No accepted H8 scientific or milestone outcome exists under the 2.9.1 pin.
Installing an older Torch build solely to satisfy that pre-outcome pin would
replace the user's functioning CUDA environment without improving the VFE
theory, ELBO, gradients, or training path.

## Decision

Amend H8 before its first scientific run to freeze the exact installed
runtime. The amendment is fail-closed and auditable:

1. Pin the full `torch.__version__` string
   `2.10.0.dev20251210+cu128`; do not normalize away the local `+cu128`
   suffix.
2. Pin the raw bytes of both private profiler source files.
3. Store the complete canonical ASCII private-API descriptor in executable
   source and derive its SHA-256 from that descriptor instead of maintaining
   an unexplained hash literal.
4. Bump `H8ValidationConfig` to `h8-validation-config-v3`.
5. Bump the parent/child protocol digest domain to
   `vfe4.h8.parent-child-protocol.v3`.
6. Bump the scientific validation protocol/artifact schema to
   `h8-sparse-scale-v5`. Retain `h8-child-v2` because its wire envelope does
   not change. The internally named v4 runtime-section builder remains a
   structural sub-schema and does not define the amended scientific protocol
   identity.
7. Cross-bind all four profiler fields—exact Torch version, both source
   hashes, and API-contract hash—between the frozen config and executable
   wire authority.
8. Preserve the earlier 2.9.1 text as superseded pre-outcome history in the
   preregistration and append a dated normative amendment. Do not present the
   old pin as current.

The replacement descriptor is exactly the following 757 ASCII bytes, with no
leading or trailing whitespace and no terminal newline:

```text
torch==2.10.0.dev20251210+cu128|runtime=installed-exact|memory_profile_source_sha256=22de3b0790907b90053af829ebf1bff0b6add2745ac0381ec7de78812edacb47|profiler_source_sha256=543430b2e9b24df777f86415865fee250b35e3444a80920bcca0e8889b917956|flags=record_shapes:true,profile_memory:true,with_stack:true|timeline=profile._memory_profile().timeline:(timestamp_ns,action,key_and_version,numbytes)|actions=PREEXISTING,CREATE,INCREMENT_VERSION,DESTROY|event_tree=profile.profiler.kineto_results.experimental_event_tree()|allocation=_EventType.Allocation+_ExtraFields_Allocation|torchop=_EventType.TorchOp+_ExtraFields_TorchOp|join=TensorKey(id,storage.ptr,allocation_id,device)+version|raw_export=(timestamp_ns,action,numbytes,category)|join_unavailable=INCONCLUSIVE
```

Its SHA-256 is
`2ee166166bab997499cc66da85146a031f458fbe0190a75b1a1a3ddea80efc38`.
`runtime=installed-exact` avoids inventing an upstream release tag for a
development build. The full version and two source hashes identify the exact
declared Python-side contract, not the compiled Kineto implementation. The
descriptor freezes the private symbols, tuple shapes, actions, join rule, and
profiler flags relied upon by H8; the real schema preflight and scientific
profiler child separately exercise the installed compiled backend.

The stable standard-library wire module remains the canonical parent recovery
authority. The child deliberately retains independent local mirrors so drift
remains observable. The type/config mirrors, preflight frozen-section hash,
source-only gate environment, runtime consumers, fake-child fixtures, and
artifact provenance must all be synchronized and tested. The config-to-wire
protocol check rejects a mismatch in any of the four profiler identity fields,
preventing a partially amended split-brain contract.

## Runtime behavior

The child performs checks in this order before profiler evidence can support a
PASS:

1. exact full `torch.__version__` equality;
2. exact source-file discovery;
3. exact raw-byte SHA-256 equality for both source files;
4. availability and observed schema of the required private profiler
   surfaces;
5. the existing lossless timeline/event-tree join and allocation invariants.

The source-only gate reports the frozen config's version rather than a second
hardcoded string. Provenance compares the observed runtime to the same config
field. A forged or partially amended config is rejected while constructing the
protocol digest and again during preflight.

## Verification strategy

Testing remains bounded:

- focused unit tests first for v3 config/protocol binding, full-version
  rejection, source-hash rejection, preflight binding, and source-only
  provenance;
- one tiny real installed-runtime profiler schema preflight using
  `C:/anaconda/python.exe`;
- no production H8 workload until that preflight passes;
- one scientific H8 execution only after the existing H1-H7/current-candidate
  evidence chain is current and mechanically validated;
- no repeated full-suite runs during this amendment.

H8 remains a CPU float64, no-grad, one-thread scientific gate. The Anaconda
interpreter is required because it is the authoritative Torch environment,
not because the H8 workload uses CUDA.

## Click-to-run and activation

The user-facing workflow remains the editable top-level `CONFIG` dictionary in
`verify_vfe4.py`; no CLI is introduced. H8 activation and its exact existing
authorization literal will be committed in a dedicated frozen-candidate
revision before current-candidate evidence is generated. The activation will
not be toggled after candidate/JUnit hashing.

WikiText-103 work begins only after H8 produces a validated PASS under this
amended contract. Production training will use the existing CUDA Torch runtime
through click-to-run launchers with editable config dictionaries.

## Non-goals

This amendment does not:

- claim that H8 already passes;
- weaken allocation, memory, timing, correctness, or observability criteria;
- change the exact VFE ELBO or any mathematical result;
- make H8 depend on CUDA;
- add a package installer, dependency downgrade, command-line interface, or
  automatic environment mutation;
- begin WikiText-103 downloads or training.

## Failure handling

Any version, source, symbol, schema, tuple, action, join, or descriptor mismatch
makes the profiler channel `INCONCLUSIVE`; it is never silently adapted. If the
tiny installed-runtime preflight exposes an actual behavioral incompatibility,
implementation pauses at the profiler boundary and records the exact open
obligation rather than launching H8 or changing the scientific criteria.
