# H7 Predecessor Provenance Amendment

**Date:** 2026-07-25
**Scope:** H1--H5, H1-Prefix-Prior scorer-v2, and H6-Prefix inputs to H7

This amendment closes three provenance gaps without changing the H7
mathematical trials or decision thresholds.

1. Every predecessor source identity is derived as
   `source_candidate_sha256(git_head, dirty_digest)`. A caller-supplied
   `source_sha256` that is merely well formed, or that agrees only between
   payloads, is not evidence of candidate identity.
2. `H7PredecessorReference` records both `junit_path` and `junit_sha256`.
   H7 resolves the exact path, rejects symlinks and non-files, streams the
   bytes through SHA-256, and requires all three predecessor references to
   name the same JUnit file. Digest agreement without an available matching
   preimage is invalid.
3. Each predecessor ledger contains one dedicated
   `EVIDENCE_VERIFIED` closure claim. Its claim ID is frozen per predecessor,
   its statement contains the H7-owned closure-binding digest, and its
   mechanical evidence records name the exact artifact manifest and candidate
   JUnit paths at the live artifact revision. A valid but unrelated closure
   ledger cannot authorize a predecessor.

The closure-binding digest covers the predecessor key, resolved artifact path,
Git head, dirty digest, resolved JUnit path, JUnit digest, manifest digest, and
complete payload-hash mapping. It deliberately excludes the ledger path and
ledger digest so the claim can be created before the ledger is serialized,
avoiding a circular hash dependency. H7 separately hashes and validates the
final repository-contained ledger bytes.

The versioned H1 projectors are also strict: the legacy projector accepts only
`h1-prefix-prior-config-v1`, while the scorer-v2 projector accepts only
`h1-prefix-prior-config-v2`. An operation-name match cannot dispatch the wrong
producer version.
