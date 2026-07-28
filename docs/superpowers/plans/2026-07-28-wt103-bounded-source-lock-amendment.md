# Bounded WikiText-103 Source-Lock Implementation Plan

**Goal:** Make real WikiText-103 source locking bounded in memory while
preserving exact VFE4 source, tokenizer, payload, and durability authority.

**Design:** See
`docs/superpowers/specs/2026-07-28-wt103-bounded-source-lock-amendment-design.md`.

## Global Constraints

- No Task 13 execution, network access, production training, or held-out
  opening in this plan.
- Do not import or deserialize the legacy `.pt` token caches.
- Do not trust filename-only or V3 provenance.
- Keep the click-to-run editable dictionary and add no CLI.
- Use `C:/anaconda/python.exe` for every test importing Torch.
- Use focused tests only; do not run the full suite.
- Preserve all unrelated live/WIP files.

## Task 1: Bounded content-addressed durability

- [x] Add red tests for chunked same-volume staging, bounded streamed reopen
  hashing, digest-derived destination, no-overwrite collision behavior,
  idempotent exact-content recovery, and injected write/reopen/promotion
  failures.
- [x] Add typed verified-size/digest identity construction without requiring
  payload bytes.
- [x] Implement the narrow content-addressed stream publication API for both
  POSIX and Windows backends, retaining existing byte APIs unchanged.
- [x] Add a reusable bounded regular-nonlink size/SHA validator.
- [x] Run only the affected durability tests, Ruff, and `py_compile`.

## Task 2: Exact bounded tokenizer stream

- [ ] Add red tests comparing the streamed result with an independent
  ordinary-encoding fixture across every byte-chunk boundary of representative
  ASCII, Unicode, whitespace, newline, and special-token-looking text.
- [ ] Add red tests for strict UTF-8 failure, tokenizer-regex coverage failure,
  source drift, token-range failure, decode mismatch, and a forbidden
  corpus-sized output chunk.
- [ ] Extend the production tokenizer adapter with the exact regex-piece and
  bounded byte-decode operations required by the stream.
- [ ] Incrementally decode UTF-8, retain the final open regex piece, encode
  closed pieces, validate bounded round trips, and emit fixed-size int32le
  chunks with accumulated count/min/max/size/SHA facts.
- [ ] Cross-check the live adapter's piece-stream result against
  `encode_ordinary` for all frozen golden vectors.
- [ ] Run only the affected tokenizer/source-lock tests, Ruff, and
  `py_compile`.

## Task 3: Source-lock integration and local cache placement

- [ ] Add a red injected source-lock test proving no whole raw split or token
  payload is passed to the bytes-only publication API.
- [ ] Publish the three production caches through the streamed backend,
  construct the existing VFE4 content-addressed records from returned facts,
  and stream-validate raw/token payloads on reopen and training open.
- [ ] Move the editable default VFE4 cache root to
  `Path.home() / ".cache" / "vfe4" / "wikitext103"`.
- [ ] Document that existing V3 `.pt` files remain quarantined and untouched.
- [ ] Run the source-lock, launcher, and production-data focused tests only,
  then Ruff, `py_compile`, and `git diff --check`.

## Task 4: Independent review and proportional closure

- [ ] Have one reviewer audit exact ordinary-tokenization equivalence and
  provenance boundaries.
- [ ] Have another reviewer audit streaming durability, crash semantics, and
  peak-memory bounds.
- [ ] Fix every High/Critical finding with a focused regression and scoped
  re-review.
- [ ] Include the final focused tests in the single proportional Tasks 1-12
  verification pass before publication.
