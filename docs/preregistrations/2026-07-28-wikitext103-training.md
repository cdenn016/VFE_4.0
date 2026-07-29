# WikiText-103 Training Preregistration

Date frozen: 2026-07-28

Status: outcome-free source protocol. This record authorizes no network
acquisition, live tokenizer inspection, production source lock, corpus
optimization, validation or test scoring, test opening, or scientific result.

Normative implementation authority:
`docs/superpowers/plans/2026-07-21-vfe4-post-h8-wikitext103-training.md`,
including its 2026-07-28 executable-build amendment, and
`docs/preregistrations/2026-07-25-post-h8-arm-gate-amendment.md`.
Conflicting historical H6-Prediction v2 or H8 v4 wording is ineligible.

## Scientific boundary and prerequisites

The experiment is the singleton-base, zero-dimensional language-model
specialization of VFE4. Token positions and source edges are a causal DAG, not
base points, base transport, base curvature, or base holonomy. The state
objective is one complete normalized ELBO. Held-out NLL and perplexity come
only from the causal target-blind prior predictor. Posterior reconstruction,
emission-only loss, samples, and free-energy diagnostics are separately
labeled outputs.

Hermetic Tasks 1--12 and the explicitly nonproduction synthetic smoke may use
generated fixtures only. Before Task 13 performs any network or live package
operation, the exact clean Tasks 1--12 revision must have:

1. the native executable H6-Prediction v3 authority, including its exact
   config, readiness, metric, checkpoint, validation, held-out, result,
   current-pointer, and claim-ledger identities;
2. `h8-sparse-scale-v5` PASS under `h8-validation-config-v3` and
   `vfe4.h8.parent-child-protocol.v3`, with the same-revision H1--H5,
   H1-Prefix-Prior, H6-Prefix, H6-Prediction v3, and H7 predecessors;
3. a distinct same-revision `TrainingSparsityCertificate` PASS;
4. the WT103 target-blind predictor-safety certificate; and
5. independent durability, capacity, environment, dependency-lock, resource,
   and authorization records.

After the finalized production source record is reviewed and committed, the
new clean revision must reproduce the full predecessor and H8 v5 chain.
Pre-source-lock H8 evidence cannot authorize the post-record revision. H8 is a
synthetic CPU-float64 systems certificate and cannot satisfy training
sparsity, GPU capacity, decoder/optimizer memory, or resource readiness.

## Candidate source and license contract

The following are candidate request strings, not verified facts:

- archive:
  `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip`;
- announcement/source page:
  `https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/`.

No mirror, Hugging Face transformation, TorchText download, prepared
vocabulary, V3 cache, redirect target, or filename-compatible alternate
response can substitute silently. Redirects are recorded in order and the
final origin must remain HTTPS. Changed or ambiguous bytes stop for a
preregistration revision before outcomes.

Only these central-directory entries are admissible:

```text
wikitext-103-raw/
wikitext-103-raw/wiki.train.raw
wikitext-103-raw/wiki.valid.raw
wikitext-103-raw/wiki.test.raw
```

The archive response is bounded to 268,435,456 bytes. Only ZIP methods 0 and 8
are allowed. Encryption, ambiguous data descriptors, multi-disk archives,
out-of-envelope ZIP64, links, devices, FIFOs, duplicate or case-colliding
names, alternate separators, absolute/drive/UNC paths, dot segments, NULs,
extra files, and unexpected directories are rejected. Each regular member
must have positive sizes, at most 671,088,640 uncompressed bytes, compression
ratio at most 100, and the total regular-file size at most 805,306,368 bytes.
Extraction is streamed into a newly reserved staging directory, never follows
a link or escapes containment, and verifies declared and observed sizes,
central-directory CRC32, recomputed CRC32, and SHA-256 before publication.

The source page is bounded to 4,194,304 bytes, must declare HTML/XHTML with
UTF-8-compatible decoding, and cannot use script-rendered text as license
evidence. License extraction searches raw bytes case-insensitively for ASCII
`creative commons`, requires exactly one occurrence inside one closed
`<p>...</p>` span no larger than 4,096 bytes, and records byte offsets,
raw-slice SHA-256, HTMLParser-visible text, and every contained `href`. Zero
or multiple matches, malformed containment, contradictory declarations,
multiple plausible license links, or response/content/redirect ambiguity
stops source lock.

Acquisition first produces
`StagedWikiText103AcquisitionObservation`, whose scope is
`nonproduction_staged_observation`. Only after archive, members, source page,
license, live tokenizer, production caches, schedules, and dependency lock
validate may Task 13 create `FinalizedWikiText103SourceRecord` and publish the
reviewed tracked JSON record. Offline reopening validates that record and all
bound bytes without a network call. A fresh observation can never silently
replace it.

The unresolved reviewed dependency-lock pair may undergo exactly one
fail-closed Task 13 transition to the fully resolved installed-RECORD pair.
The unresolved predecessor bytes, writer identity, and resolved replacement
bytes must all validate. A partial retry accepts only either exact endpoint
of that transition. The finalized source bundle and tracked source record are
exclusive-create-or-exact-idempotent publications; once the tracked final
record exists, source-lock mode reopens it before any network, tokenizer,
cache, lock, or staging mutation. Different bytes require a new
preregistration revision and path. From before that marker inspection through
the final durable reopen, the transaction holds nonblocking OS leases over the
canonical repository, cache root, and finalized-record path in deterministic
order. Transactions sharing any mutation root therefore cannot overlap even
when their other configured paths differ.

Unfrozen until Task 13:

- request/final URLs and redirect chains;
- response status, content type, and source/license bytes;
- archive, central-directory, and member size/CRC/hash identities;
- installed distribution RECORD and tokenizer-table identities;
- production token counts, payloads, and round-trip identities;
- window, permutation, cadence, and schedule hashes; and
- the final dependency-lock hash.

## Tokenizer, cache, and windows

Before Task 13, the complete production-scope candidate tokenizer contract is
only:

```text
distribution = "tiktoken"
version = "0.12.0"
encoding_name = "gpt2"
```

Tasks 1--12 do not import `tiktoken`, inspect `importlib.metadata`, read live
distribution/RECORD files, inspect live regex/rank/special-token tables, or
run production golden vectors. They may use only injected synthetic adapters
and may create only `SyntheticFixtureTokenizerSpec` and
`SyntheticFixtureTokenCacheIdentity`. These records are readiness-ineligible.
Task 13 alone may create `ProductionTokenizerSpec` and
`ProductionTokenCacheIdentity`.

The production facts proposed for source-lock verification are:

```text
vocabulary_size = 50_257
ordinary token IDs = 0..50_255
<|endoftext|> = 50_256
corpus method = encode_ordinary
allowed special tokens = none
inserted BOS/EOS = none
padding input token = 50_256
ignored target sentinel = -100
raw decoding = strict UTF-8 with no newline/Unicode normalization
```

Each split is encoded and round-tripped independently as one complete member;
no fitted state, vocabulary learning, or cross-split statistics are allowed.
Token payloads are contiguous little-endian int32. Production identities bind
the raw split, exact production tokenizer type/domain/hash, token count,
minimum/maximum ID, payload size/hash, round-trip hash, builder code hash, and
zero cross-split parents. V3 cache paths are rejected.

The VFE4 default cache root is exactly
`str(Path.home() / ".cache" / "vfe4" / "wikitext103")`.
`Path.home() / ".cache" / "tokenized_cache" / "*.pt"` denotes legacy V3
PyTorch ZIP/pickle caches. Those files remain untouched, unread,
un-deserialized, and inadmissible as VFE4 source, tokenizer, or cache
authority.

The canonical domains are disjoint:

```text
vfe4.wt103.staged-acquisition-observation.v1\0
vfe4.wt103.finalized-source-record.v1\0
vfe4.wt103.synthetic-fixture-tokenizer-spec.v1\0
vfe4.wt103.production-tokenizer-spec.v1\0
vfe4.wt103.synthetic-fixture-token-cache.v1\0
vfe4.wt103.production-token-cache.v1\0
```

Windows freeze `sequence_length=128` and `stride=128`. For a split with
`n>=2`, starts are `0,128,256,...` below `n-1`; inputs are
`tokens[start:start+128]`, targets are `tokens[start+1:start+129]`, and only
targets originating in that split are valid. The final partial window is
included exactly once. Padding inputs use 50,256; ignored targets use -100.
Across the split, counted targets equal `n-1`, each transition appears once,
and no transition crosses a split.

Training includes every window and final batch (`drop_last=False`). Each pass
permutes window IDs with NumPy PCG64 from the hashed tuple
`(2026072199, train_split_sha256, window_manifest_sha256, pass_index)`;
the little-endian uint64 schedule and NumPy version are bound. Validation and
test use ascending IDs, no shuffle/subsample, and the full padded final
window. The initial policy is `num_workers=0`. `DataCursor` binds the split,
pass, permutation hash, next batch ordinal and exact window IDs, and cumulative
valid targets. Resume validates stored schedule bytes and never regenerates a
different order from a nominal seed.

## Shared experiment profile

The shared profile is `wt103-experiment-profile-v1`:

```text
dataset_schema = "wikitext-103-raw-v1"
tokenizer_schema = "gpt2-tiktoken-v1"
vocabulary_size = 50_257
sequence_length = stride = batch_size = 128
gradient_accumulation_steps = 1
num_workers = 0
pin_memory = true
drop_last = false
model_depth = 1
d_z = d_m = K = 20
combined_latent_block = 40
source_lookback = 20
state/model parents(t) = range(max(0,t-20),t)
population_frame_profile = "h7-direct-glplus-v1"
decoder_profile = "categorical_linear_chunked"
decoder_train_token_chunk = 512
decoder_eval_token_chunk = 256
smc_particle_chunk = 32
dropout_probability = 0.0
input_output_embedding_tied = false
```

Arm-specific objective, prior, source mixture, latent/recognition
applicability, recognition family/iterations, update phases, scorer, and
result role never appear as shared defaults and are never inferred from an arm
family label.

The real profile uses only `cuda:0`, float32 parameters and optimizer states,
bfloat16 autocast, no GradScaler, float32 SPD factor/solve/logdet, float64 SMC
log weights, and `math.fsum` corpus accumulation. Deterministic algorithms and
cuDNN determinism are on; benchmark, TF32, and fp16 reduced-precision
reduction are off; `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA
initialization. There is no CPU or alternate-GPU fallback.

AdamW freezes betas `(0.9,0.999)`, epsilon `1e-8`, `amsgrad=false`,
`foreach=false`, `fused=false`, per-active-block global L2 clipping at 1.0,
and validity-only acceptance. Nonfinite objective/gradient, AMP overflow,
invalid support, non-SPD state, scope mismatch, snapshot alias, or optimizer
access mismatch rejects and exactly restores parameters, optimizer,
scheduler, RNG, AMP, and counters.

Each attempt uses 100 optimizer-step linear warmup followed by cosine decay to
0.1 of the initial rate, no restart, over the planned active optimizer steps.
Validation has 20 stable-deduplicated boundaries per pass. There are exactly
two passes, no early stopping, and no best-checkpoint selection. The newest
two boundary checkpoints are `resume_only`; only the post-pass
`terminal_scoring` checkpoint can be selected for confirmatory/test scoring.

## A0 architecture and match rule

A0 is one pre-norm decoder block with final LayerNorm, two heads, learned
absolute positions of capacity 128, token-plus-position input, tanh-approximate
GELU, explicit biased QKV/output/MLP projections, untied biased decoder, and
zero dropout. Attention is full causal inclusive-self and implemented only by
`torch.nn.functional.scaled_dot_product_attention` inside:

```text
torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
```

The project policy is `flash_attention_only_no_fallback`. Math,
memory-efficient, and cuDNN alternatives are disabled by the single-backend
context. `is_causal=true`, the mask argument is `None`, attention weights are
not returned, GQA is off, and fused full attention is allowed only if no
forward/backward aggregate pair-axis tensor is materialized. Task 13 freezes
the observed PyTorch/API/backend hashes; before then those values are
explicitly unresolved candidates.

The exact parameter formula is:

```text
P_A0(h) = 2*V*h + 128*h + 12*h^2 + 15*h + V
```

Candidate widths are:

```text
(20,24,28,32,36,40,44,48,52,56,60,64,72,80,96,112,128,160)
```

Eligibility requires parameter mismatch at most 1%, semantic training-FLOP
mismatch at most 5%, exact optimizer access, and no filler state relative to
the parent-specific complete PRIMARY endpoint. Selection minimizes
`(abs(log(P_A0/P_PRIMARY)), abs(log(F_A0/F_PRIMARY)), h)`. No corpus outcome
enters selection and no primary dimension moves to rescue matching.

## Immutable arms and gates

The exact ordered arm rows are:

| Arm | Factory | Objective | Prior | Latent/recognition | Scorer | Role |
|---|---|---|---|---|---|---|
| `WT103-A0-AR-v1` | `build_wt103_a0@wt103-arm-v1` | cross-entropy | absent | false/false | exact autoregressive | PRIMARY_REFERENCE |
| `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1` | `build_wt103_a5_parent_specific@wt103-arm-v1` | complete ELBO | parent-specific pooled prefix | true/true | weighted SMC | PRIMARY_ENDPOINT |
| `WT103-A5-FIXED-COMPLETE-v1` | `build_wt103_a5_fixed@wt103-arm-v1` | complete ELBO | fixed | true/true | weighted SMC | PRIOR_CONTROL |
| `WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1` | `build_wt103_a5_parent_specific@wt103-arm-v1` | emission-only ablation, non-ELBO | parent-specific pooled prefix | true/true | weighted SMC | OBJECTIVE_GATE |
| `WT103-A5-NOLATENT-v1` | `build_wt103_a5_nolatent@wt103-arm-v1` | cross-entropy | absent | false/false | exact autoregressive | LATENT_PATH_CONTROL |

All use grid `wt103-six-cell-v1`, confirmatory seeds
2026072101--2026072108, and terminal role `terminal_scoring`. Latent rows
freeze exact source mixture, structured block-tridiagonal smoothing, one
recognition iteration, and phases
`recognition_adam_proposal`, `immutable_detached_snapshot`,
`model_adam_proposal`. Nonlatent rows explicitly freeze absent source and
recognition with only `model_ce_adam_proposal`; no dormant latent work exists.

The ordered gates are:

```text
SOURCE_LOCK
H8_EXACT_REVISION
POST_H8_READINESS
OBJECTIVE
PRIMARY
PRIOR_CONTROL
LATENT_PATH_CONTROL
```

OBJECTIVE is a prerequisite of PRIMARY. Failure/inconclusiveness prevents the
PRIMARY claim but does not erase durable records. Controls never rescue,
reverse, or promote PRIMARY. `EndpointInventory.create` is the only producer
of tuning, checkpoint, validation/test endpoint, raw-score, result-row,
resource-work, figure-panel, and figure-series keys and counts.

## Update, evaluation, tuning, and statistics

A0 and no-latent use expected/observed autograd scope `m_step`. Latent arms use
`e_and_m`: recognition leaves are optimized while model parameters are frozen,
then an owned nonaliasing snapshot is cloned, detached, and hashed before the
model proposal. The exact update label is `adam_proposal`; no stronger H5
label or monotonicity claim is allowed.

Every arm tunes the six cells from learning rates
`(1e-4,3e-4,1e-3)` and weight decays `(0,1e-2)` with seeds 2026072199 and
2026072200 on exactly the first quarter-pass schedule. Selection minimizes the
mean full-validation prior-predictive NLL/token; ties choose lower learning
rate then lower weight decay. All cells are published.

Confirmatory seeds are 2026072101--2026072108, with shared data-order seed
2026072199. Exactly two passes run. Missing pairs, numerical divergence, or
failed scientific checks receive no replacement; at most one infrastructure
retry is allowed with proof of no advancement or exact restoration.
The full click-launcher hash includes `operation`, while the experiment
configuration hash excludes only that operational mode. Task 14 evidence,
plans, checkpoints, and scientific resume identity bind the stable experiment
hash; the launcher still requires the exact `train` or `resume` operation and
its separate authorization.

Validation weighted-SMC rows use 256 particles and streams 0--7. Exact rows
are evaluated once and may only be replayed for identity. Test weighted-SMC
rows use streams 0--63 and particles `(128,256,512,1024)`. The primary metric
is corpus-summed target-blind NLL divided by counted targets; perplexity is
`exp(NLL/token)`. Batch means are never averaged.

For every weighted endpoint:

```text
Q0 = 2*Y256 - Y128
Q1 = 2*Y512 - Y256
Q2 = 2*Y1024 - Y512
R1 = Q1 - Q0
R2 = Q2 - Q1
c = 4.5144904535377144
h(X) = c*s(X)/sqrt(64)
U1 = abs(mean(R1)) + h(R1)
U2 = abs(mean(R2)) + h(R2)
B = 4*U2
H = h(Q2)
delta = -log(0.99) = 0.01005033585350145
```

Eligibility requires `U2<=0.75*U1`,
`B<=0.00025125839633753625`, and
`H<=0.0005025167926750725`. The primary paired half-width also requires
`H_i<=delta/20` and `H_i+B_i<=delta/10`. The eight seed-error intervals are
propagated over all 256 error-box corners with `t_(0.975,7)=2.364624251592784`.
Inflated lower bound above delta is PASS; inflated upper bound at most zero is
FAIL; otherwise the result is INCONCLUSIVE.

## Sparsity, durability, checkpoints, and metrics

`TrainingSparsityCertificate` binds the exact revision, profile, factory set,
endpoint inventory, whitelist, forbidden shapes, traces, formula
reconciliation, and negative controls. Every distinct arm path is traced
through data transfer, forward, applicable proposal/snapshot, backward,
optimizer, evaluation, metric/failure write, and checkpoint serialization.
Population/source pair-axis storage outside the width-20 envelope is
forbidden. Vocabulary-axis density is separately declared and does not make
population inference dense. This structural certificate and the independent
85%-of-device shape-identical capacity preflight are both required.

Every scientific write uses a startup-probed same-volume
`DurabilityBackend`: exclusive staging/reservation, complete write and file
flush, prepublication byte/schema/hash validation, durable replace, directory
durability where supported by the platform contract, and reopen/hash
validation. Unknown network/FUSE/cloud semantics are INCONCLUSIVE.

Boundary checkpoints are `resume_only`; post-pass checkpoints are
`terminal_scoring`. Each binds model/recognition, applicable
optimizer/scheduler/AMP state, all RNG/counter streams, exact permutation and
cursor, config/objective/update/factory/predecessor/data/environment hashes,
metric/failure heads, accepted/rejected counts, and lineage. Executable pickle
is prohibited; loading is size/hash checked, CPU-first, `weights_only=True`,
and limited to the declared primitive/tensor whitelist. V3 migration is
permanently rejected.

`scientific_state_sha256` covers scientific tensors/primitives and excludes
paths, timestamps, durations, process IDs, serialization layout, and terminal
artifact hashes. `artifact_sha256` is domain-separated over exact payload and
manifest-body hashes. Faithful resumes require equal scientific state and next
predictions, not byte-identical operational serialization.

Each attempt holds a nonblocking per-attempt OS execution lease from
reservation through terminal finalization; another process must fail before
owner, lineage, metric, or checkpoint mutation. Process exit releases the
lease. The single allowed infrastructure retry is a plan-bound lineage
budget, not permission for repeated click-resume attempts. Immediately before
owner or ordinal mutation, the live lease exclusively creates and reopens an
immutable `resume-lineage-intent.json`; a pre-ledger crash must recover that
exact event, never synthesize a new timestamp or digest. Immediately before
resumed scientific execution, after the conservative resource pre-debit
commits, the lease exclusively creates and reopens a hashed
`resume-execution-started.json` transition bound to the plan, reservation,
ordinal, and lineage. Its presence permanently consumes the retry; the same
lineage and every new lineage are rejected before subsequent owner or lineage
mutation. Every rolling save authenticates the currently committed sidecar and
writes the opposite inactive slot before atomically replacing that sidecar, so
a crash cannot overwrite the last authenticated resume point. If terminal
manifest publication succeeds but the process exits before closing the resume
owner, recovery holds the attempt lease, validates the exact manifest,
artifacts, lineage intent, lease, and execution-start transition, durably closes
only that exact active owner against the manifest hash, then strictly
revalidates before the terminal rename.

`metrics.jsonl` and `failures.jsonl` are independently hash-chained, flushed,
and durably published. Every mean/rate stores numerator and denominator.
The complete-objective arms use the distinct
`wt103-structured-factor-elbo-v1` partition schema. Its raw factor record is
the expected log emission; initial, source, and transition cross-entropies for
both model and state paths; the analytic continuous-recognition entropy; the
one-sample conditional-source-entropy estimate; and their joint-recognition
entropy estimate. The objective is exactly emission minus all six generative
cross-entropies plus the joint entropy estimate. Only the conditional
categorical source terms are also recorded as genuine derived source-KL
diagnostics. H5's mean-field conditional-Gaussian KL labels are reference-only
and are not reused for the block-tridiagonal WT103 smoothing law.

The single reparameterized WT103 draw has no preregistered finite estimator
error bound. `estimator_error_bound` is therefore explicitly not applicable
with reason
`no_preregistered_finite_bound_for_single_sample_mc`; zero is prohibited.
Required metrics otherwise include the complete factor record or explicit
not-applicability; target-blind NLL/PPL and estimator identities; source
entropy/effective source count; proposal acceptance/rejection and exact
optimizer settings; SPD/solve/condition health; gradients and autograd scope;
work counts/timing; and host/device memory/failure records. Inapplicable data
is labeled with a reason, never fabricated as zero.

## Resources, figures, and the one opening

The immutable ceilings are 720 GPU hours, 840 wall hours, and 500 kWh.
Forecasts use minimum post-warmup throughput, maximum duration/power, a 1.25
headroom factor, 100 ms power sampling, exact inventory work counts, and
independent disk/host/device headroom. Device allocated and reserved peaks
must each be at most 85% of physical capacity. Insufficiency stops for an
explicit revision; no batch, sequence, particle, source, dimension, precision,
seed, or endpoint is silently reduced.

Resource debits are deliberately conservative and never refunded. Before
scientific work, each attempt durably prepays 60 seconds of device time, wall
time, and energy at the readiness-bound frozen conservative maximum of the
measured board-power peak and provider-reported limit. A separate
monotonic heartbeat, independent of the 100 ms sampler, durably debits elapsed
30-second intervals at that frozen maximum; heartbeat failure is polled inside every
training and validation loop and aborts within the prepaid runway. Measured
device seconds, wall seconds, and sampled energy are then appended without
offsetting the conservative debits. Headroom is checked before every ledger
publication, so an uncatchable process exit leaves a prepaid tail rather than
an unrecorded zero. A measured interruption is debited before propagation, and
a resumed segment appends rather than replacing or collapsing prior usage.

The deterministic required figure registry contains:

1. training objectives and validation NLL/PPL;
2. terminal NLL/PPL;
3. complete-ELBO/emission-only decomposition;
4. source entropy and effective source count;
5. accepted/rejected updates;
6. SPD/solve health;
7. throughput/time/host/device memory; and
8. seed variability.

Figures consume only finalized manifest-validated JSONL and the frozen result
table. CSV is regenerated and byte-checked, never trusted as a semantic input.
Each spec freezes inventory-derived panels/series, applicability, aggregation,
uncertainty, labels/units/style, DejaVu fonts, fixed SVG hashsalt/metadata, and
caption/alt text. Each output has SVG, PNG, PDF, plotted CSV/JSON, caption,
alt-text, and content-addressed manifest identities. Rendering imports no
training/checkpoint/data path and never repairs a partial result.

The test split is opened exactly once after every inventory-derived terminal
checkpoint, selection, source/schedule identity, estimator stream, analysis
hash, figure hash, and run-group manifest is frozen. The durability backend
must publish the exclusive reservation before a capability exists. A crash
after reservation is terminal. Partial endpoints are durably retained but
never aggregated or promoted; the transaction is not reopened for a fix,
replacement seed, extra metric, or prettier figure.

## Explicit nonclaims

This protocol does not claim:

- H8 proved training sparsity, GPU memory safety, decoder/optimizer sparsity,
  capacity, or an asymptotic scaling law;
- H6 byte-tokenizer or H6-Prediction results transfer to GPT-2/WikiText-103;
- VFE4 is backprop-free, forward-gradient trained, exact EM, or monotone under
  `adam_proposal`;
- posterior reconstruction or emission-only diagnostics are predictive
  evidence;
- the PRIMARY matched whole-architecture comparison isolates one mechanism;
- controls rescue or reverse PRIMARY;
- source entropy proves attention, causal discovery, gauge covariance, or
  useful source identification;
- two passes establish convergence, scaling, calibration, state of the art,
  or long-context generalization; or
- any V3 cache, checkpoint, objective, or config is compatible with VFE4.

## Authorization gates

Task 13 needs separate authorization for network/source lock and live tokenizer
inspection. Task 14 may produce generated-data readiness evidence but performs
no corpus optimization or held-out scoring. Task 15 needs separate
authorization after the frozen resource forecast to train on WikiText-103.
Task 16 needs explicit acknowledgment of the irreversible one-opening test
transaction unless that authorization already covered the complete frozen
experiment.
