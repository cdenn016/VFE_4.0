# Bounded WikiText-103 Source-Lock Amendment

**Status:** Approved remediation within the existing autonomous WikiText-103
buildout

**Scope:** Replace the whole-corpus token-cache construction and validation
path before Task 13. This amendment does not authorize Task 13, production
training, or held-out test opening.

## Problem

The production builder currently materializes the complete raw split, decoded
text, Python token list, NumPy array, serialized payload, and durable readback.
For the 116,840,318-token training split, plausible peak memory is several
gigabytes. The bytes-only durability API also requires corpus-sized copies.
That violates the buildout's bounded-resource intent even though synthetic
fixtures pass.

The existing files under
`C:\Users\chris and christine\.cache\tokenized_cache` are useful historical
artifacts but are not production source authority. They are PyTorch
ZIP/pickle-framed int64 tensors whose sidecars omit the raw-parent hash, exact
GPT-2 encoding and tiktoken distribution/table identities, payload hash,
builder identity, and VFE4 record identity. V3 itself classifies these caches
as filename-inferred and unverified.

## Chosen Design

Task 13 will still begin from the official staged raw archive and the
source-locked live GPT-2/tiktoken adapter. The existing `.pt` files will remain
untouched and will not be admitted as VFE4 cache authority.

Tokenization becomes a bounded stream:

1. Read each sealed raw split through a regular-nonlink, size- and
   SHA-validated byte stream with strict incremental UTF-8 decoding.
2. Preserve exact `encode_ordinary` semantics across byte-chunk boundaries by
   retaining the final incomplete tokenizer-regex piece and encoding only
   closed pieces with the pinned tokenizer's single-piece BPE operation.
3. Compare every encoded piece's bounded byte decode with the exact source
   piece, validate token range, and buffer only a fixed small token block.
4. Emit canonical little-endian int32 chunks while accumulating count,
   min/max, byte size, and SHA-256.

The durability backend gains one narrow content-addressed streaming
publication operation. It writes bounded chunks to a unique same-directory
staging file, flushes it, reopens and hashes it in bounded blocks, derives the
final digest filename, and promotes it without overwriting an existing
different target. It then performs another bounded reopen validation and
returns a typed identity derived from the verified size and digest. Existing
small-artifact byte APIs remain unchanged.

All later raw-split and token-cache identity checks use bounded streamed
size/SHA validation. Training continues to memory-map only the final
content-addressed int32 token payload and casts batch slices to `torch.long`.

## Cache Placement

The click-to-run default VFE4 cache root moves under
`Path.home() / ".cache" / "vfe4" / "wikitext103"`. This uses the user's
established cache area without mixing VFE4 authority with legacy V3 cache
files. The editable dictionary remains the sole configuration surface; there
is no CLI or environment-variable requirement.

## Failure Semantics

- Invalid UTF-8, uncovered tokenizer-regex text, token-range or bounded
  round-trip mismatch, raw identity drift, or streamed digest mismatch stops
  before a cache record is published.
- A conflicting final digest path is never overwritten.
- A crash may retain only a uniquely named backend staging file; it cannot
  create a finalized source record or source-lock bundle.
- Task 13 remains guarded by
  `AUTHORIZE_VFE4_WT103_SOURCE_LOCK_V1`.

## Verification

Focused tests must first fail against the whole-corpus implementation, then
cover every boundary of representative ASCII, Unicode, newline, whitespace,
and special-token-looking fixture text; invalid UTF-8; source mutation;
round-trip and range failures; streamed publication/reopen; exact-content
collision; conflicting collision; and a guard that rejects corpus-sized
single chunks. The live Task 13 adapter must cross-check its streaming result
against ordinary encoding on the already frozen golden vectors before it can
issue tokenizer authority.

