# H6 Prefix and Prediction Preregistration

Date frozen: 2026-07-21  
Source-build revision: assigned after Task 11 review  
Evidence revisions: separately assigned for H6-Prefix and H6-Prediction

## Scope and separation

H6 implements the singleton base `C0={*}` and language modeling over labeled
population copies at that point. The causal token DAG is a separate discrete
object. Token edges, cache transitions, and source links are not base transport,
curvature, or holonomy. H6 therefore makes no H7 frame-covariance claim.

H6-Prefix and H6-Prediction are distinct operations with distinct schemas,
artifacts, ledgers, and decisions:

- `h6-prefix-v1` consumes only exact H6 source, config, model-family,
  vocabulary, estimator, fixture, and data-safety identities. It has no H1-H5
  predecessor and never invokes H4.
- `h6-prediction-readiness-v1` consumes exact current H1, H2, H3, and H5
  correctness artifacts, a conditional H1 prefix-prior artifact when active,
  the finite-SMC artifact, the H6-Prefix certificate set, exact H5 producer
  identities, the H6-owned schedule, data/access identities, and the frozen
  endpoint protocol. H4 correctness, timing, and cost are not prerequisites.
- `h6-prediction-v1` owns empirical attempts, checkpoints, scoring, uncertainty,
  and final decisions after readiness.

Source buildout is complete only when focused deterministic fixture tests and
static review close Tasks 1-12. It does not itself run exhaustive Prefix cases,
the 512-replicate SMC grid, corpus training, 96-checkpoint assessment, or the
one-time test opening. Those are separately authorized evidence operations at
one frozen `(git_head, dirty_digest)`.

## Theory and structural types

The normative sources are the VFE 4.0 whitepaper sections on the generative
model, structured information form, transformer crosswalk, limitations, and
appendices. The language specialization uses:

```text
ZeroDimensionalBase(base_id="C0", points=("*",), dimension=0)
CausalDag(labeling="zero_based", node_labels=(0,...,T), rows=(receiver,parents))
H6LanguageStructure.receiver_labels == tuple(row.receiver_t for row in dag.rows)
```

Every parent is a declared node with `parent < receiver_t`; receivers are
unique, ordered, and complete. One-based, gapped, duplicated, self, future, or
ambiguous labels are rejected.

The complete language ELBO is horizon-indexed and contains seven disjoint
partitions: emission, initial, state source, model source, state transition,
model transition, and recognition entropy. Every factor has an ordered identity.
The independently accumulated complete decomposition must equal the reported
total. Emission-only training is explicitly
`emission_only_ablation_non_elbo`; it is never relabeled as an ELBO.

Public tensor-bearing results use `FrozenTensorSnapshot`: a private contiguous
clone that remains connected to the autograd graph, immutable metadata and raw
bytes, storage-version validation, and clone-only public access. Mutating caller
storage or a returned clone cannot alter the record; mutating private storage
causes integrity failure before use.

Each record has at most one owned integrity digest. Its domain-separated
canonical preimage contains every semantic field and already-verified reference
digest but excludes the owned digest itself. Tokenizer, fixture, payload,
manifest, H5-producer, and other referenced/content digests are independently
verified against their named bytes or producer preimages.

## Predictor and Prefix certificate

The only public prediction call is:

```python
next_token_log_probs(prefix_tokens, estimator_rng, cache=None)
```

It has no target, suffix, full-window, recognition, posterior, or reconstruction
argument. It returns one `(V,)` log-probability vector bound to an immutable
`VocabularyIdentity`. The target is read only after that vector is returned.

Every source row uses the sole shared
`masked_log_softmax_from_parents(logits, declared_parents, receiver_t)`. Invalid
support is written as exact `-inf` before normalization. Empty support is an
error. Post-softmax masking, a second normalization helper, self/future parents,
or target/suffix/recognition flow into the predictor is forbidden.

`PrefixCaseKey` binds arm, predictor config, estimator, model family,
vocabulary, data safety, Git revision, and dirty digest. `PrefixCertificate`
binds the full key, immutable canonical validation bytes and their SHA-256,
PASS/FAIL/INCONCLUSIVE status, obligations, and its own integrity digest. PASS
requires the complete exact check inventory, every check true, and no obligation.

The production small fixture has `V=3`, `T=4`, and receiver rows
`(1:(0), 2:(0,1), 3:(0,1,2), 4:(0,1,2,3))`. Its mask inventory is 168 base
cases. The WikiText-2 property inventory is 16,384 base cases. Exhaustive
leakage comparison contains 9,720 comparisons per certified
model-family/estimator profile. Each dynamic case runs cold-cache, warm-cache,
and reverse-order/cache-rebuild modes with common counter streams. Leakage and
mask allowances are exactly zero on deterministic CPU float64, including dtype,
shape, device, contiguous bytes, and signed-zero-sensitive SHA-256 identity.

## H6-owned update schedule

H5 contributes only the exact producer labels `exact_coordinate`,
`generalized_em`, and `natural_gradient_proposal` and its producer hashes. It
does not certify AdamW or an H6 phase.

The H6 outer schedule uses AdamW, one model update opportunity per batch, two
full passes, and validation boundaries at
`ceil(k*batches_per_pass/20), k=1..20`, deduplicated in order for each pass.
No-latent endpoints use only `model_ce_adamw` and construct no recognition
optimizer. Latent endpoints use exactly
`recognition_adamw -> immutable_detached_snapshot -> model_adamw` once per
batch. No-op phases or dormant parameters are prohibited.

The shared training batch size is eight with no drop-last. A versioned
SHA-256-counter, rejection-sampled Fisher-Yates permutation keyed by seed
`2026072199` and zero-based pass index fixes training order independently of
Python, NumPy, and Torch RNG implementations. Evaluation order is sequential.

## Frozen tuning protocol

Every public arm A0--A5 uses the same six-cell grid

```text
learning_rate in {1e-4, 3e-4, 1e-3}
weight_decay in {0, 1e-2}
```

with exactly two quarter-pass runs per cell. The tuning seeds are
`2026072199` and `2026072200`; `2026072200` is the adjacent independent
companion frozen before outcomes because the source protocol required two
tuning seeds but named only `2026072199`. A quarter pass is the first
`ceil(number_of_batches/4)` batches of the frozen full-pass permutation.

Each arm publishes all twelve tuning runs and selects the cell with the lowest
mean validation prior NLL. A tie is resolved first by lower learning rate and
then by lower weight decay. PRIMARY therefore gives A0 and A5 equal six-cell
tuning, MAP gives A2 and A5 equal six-cell tuning, and A1, A3, and A4 remain
independently tuned descriptive controls. The six nonbase component endpoints
in STRUCTURE, PRIOR, MIXTURE, OBJECTIVE, LATENT, and RECOGNITION use the
selected A5-primary `(learning_rate, weight_decay)` exactly and are not
separately tuned. No validation or test outcome may expand the grid, replace a
seed, or alter this estimand.

## Data and access boundary

The bounded H6 corpus is the official WikiText-2 raw archive at exactly
`https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip`.
Public configuration cannot select another URL or opener. Synthetic tests
inject a byte-stream opener only through the non-exported internal
`_acquire_wikitext2_blinded(config, opener)` seam; public
`acquire_wikitext2_blinded(config)` always uses the exact URL. Training never
substitutes WikiText-103, a prepared vocabulary, synthetic text, or another
mirror.

Archive preparation accepts at most `16,777,216` downloaded compressed bytes
and exactly these four entries:

```text
wikitext-2-raw/
wikitext-2-raw/wiki.train.raw
wikitext-2-raw/wiki.valid.raw
wikitext-2-raw/wiki.test.raw
```

The directory entry is the sole directory. Each file has positive compressed
and uncompressed size, each size is at most `16,777,216`, total uncompressed
file bytes are at most `33,554,432`, and the uncompressed-to-compressed ratio
is at most `100`. Only ZIP_STORED (`0`) and ZIP_DEFLATED (`8`) are accepted.
Encryption, links, duplicate or case-colliding paths, missing or extra entries,
path traversal, unsupported methods, central-directory/streamed size or CRC
disagreement, and decompression beyond a bound are rejected. The exact archive
bytes, three streamed raw members, tokenizer specification, encoded streams,
and window manifests are independently SHA-256 bound.

Observed archive/member byte sizes, compression methods, CRC32 values, raw
SHA-256 values, encoded-token SHA-256 values, and window/fixture identities are
measured and copied into the exact config and this preregistration before the
separately authorized evidence revision. They are explicitly deferred during
source buildout and must match during bounded streaming extraction; no value is
inferred from or selected using prediction outcomes.

Raw bytes map to IDs `0..255`, `BOS=256`, `EOS=257`, vocabulary size 258, and
ignored target `-100`. Each split is `[BOS] + exact raw bytes + [EOS]` with no
text decoding or newline normalization. Encoded identity uses the domain
`VFE4-H6-U16LE-TOKENS-V1`, a little-endian uint64 token count, and unsigned
16-bit little-endian IDs.

Windows have length and stride 32. Inputs are `tokens[start:start+32]`; targets
are `tokens[start+1:start+33]`. The final partial window appears once, with BOS
input padding and `-100` target padding. Counts exclude `-100`.

Before readiness, only sealed split handles and
`validation_safety_fixture.bin` are available. That fixture contains exactly
4,096 distinct validation-only windows selected by domain-separated SHA-256
rank with index tie-break and serialized canonically with validation identity,
starts, real-target counts, and 33 uint16-LE IDs. It depends only on validation
content and the frozen policy. Train/test/archive changes cannot change it.

The binary publisher accepts five caller payloads: three sealed raw members,
the fixture, and `data_identity.json`. It then generates a self-excluding
manifest over ordered path, length, and content hashes. `data_identity.json`
cannot contain the enclosing manifest identity. Publication uses fixed paths,
same-volume owned staging, create-new writes, closed handles before Windows
rename, and no-overwrite installation. `ZipFile.extract`, `extractall`, and the
JSON-only run publisher are forbidden.

Training materialization requires exact readiness. Test mapping requires an
opaque capability issued only after a durable `O_EXCL` reservation; its
canonical reservation proof binds readiness, experiment, data/test, access
policy, and reservation identities and is independently revalidated before any
test mapping.

## Weighted SMC recursion

Before token `t`, particles carry normalized `log w_(t-1)` and histories.
Prediction is the weighted mixture

```text
log p_hat_t(v) = logsumexp_n(log w_(t-1)^n + ell_t^n(v)).
```

After the formerly predicted token is appended:

```text
log w_tilde^n = log w_(t-1)^n + ell_t^n(x_t)
log Z_hat_t   = logsumexp_n(log w_tilde^n)
log w_t^n     = log w_tilde^n - log Z_hat_t
ESS_t         = 1 / sum_n(exp(log w_t^n)^2)
```

Systematic resampling occurs only when `ESS_t < N/2`, after assimilation, and
uses one named counter draw in `[0,1/N)`. Otherwise no resampling counter is
consumed. The sequence likelihood is `sum_t log Z_hat_t`, equal to the sum of
selected public predictions from the same pending particles.

The finite gate uses four exact `V=3,T=6` fixtures, `N=256`, and 512 replicate
seeds `2026072300..2026072811`. It has 76 cells and simultaneous bias/variance
tails `a=0.01/304`. Frozen constants are:

```text
t_(1-a,511)       = 4.0243186150882195
chi2_(a,511)      = 393.23185025997486
chi2_(1-a,511)    = 648.65591595794933
delta             = 0.01005033585350145
bias_limit        = 0.001005033585350145
sd_limit          = 0.0025125839633753625
```

PASS requires every simultaneous upper absolute-bias bound at most
`bias_limit`, every upper SD bound at most `sd_limit`, and exact recursion,
normalization, replay, cache, and normalizer identity. A finite lower bound over
a limit or witnessed recursion defect is FAIL; unresolved intervals are
INCONCLUSIVE. This full grid is deferred evidence.

## Endpoint uncertainty and decisions

Actual assessment contains 12 endpoint configurations by eight seeds, hence 96
checkpoints. Each uses common replicate IDs `0..63` and particle counts
`(128,256,512,1024)`. For checkpoint `c`, replicate `r`, and level `N`:

```text
Y[c,r,N] = -fsum_t(log Z_hat[c,r,N,t]) / counted_test_targets
Q0 = 2*Y256 - Y128
Q1 = 2*Y512 - Y256
Q2 = 2*Y1024 - Y512
R1 = Q1 - Q0
R2 = Q2 - Q1
```

All means use `math.fsum/64`; variances and covariances use denominator 63.
Exactly 352 simultaneous intervals use frozen critical value
`4.5144904535377144`. `Q2` is the reported NLL. Missing streams, levels,
convergence, finite values, or uncertainty closure make the endpoint
INCONCLUSIVE.

For any 64-vector `X`, define the simultaneous half-width

```text
h(X) = 4.5144904535377144 * s(X) / sqrt(64)
```

where `s(X)` is the denominator-63 sample standard deviation. For checkpoint
`c`, define

```text
U1 = abs(mean(R1)) + h(R1)
U2 = abs(mean(R2)) + h(R2)
B_c = U2 / (1 - 0.75) = 4*U2
H_c = h(Q2)
```

Eligibility requires all three weak inequalities
`U2 <= 0.75*U1`,
`B_c <= delta/40 = 0.00025125839633753625`, and
`H_c <= delta/20 = 0.0005025167926750725`. These are empirical simultaneous
convergence bounds conditional on the preregistered geometric-remainder
assumption, not a proof of unbiasedness. A nonfinite value, missing replicate
or particle level, contraction failure, or crossed threshold makes every
consuming contrast INCONCLUSIVE; a witnessed recursion or identity defect is
FAIL.

For each matrix row and training seed `i`, define the 64 common-stream values

```text
D_i[r] = Q2_left_i[r] - Q2_right_i[r]
d_i    = mean_r(D_i[r])
H_i    = h(D_i)
e_i    = H_i + B_left_i + B_right_i
```

Require `H_i <= delta/20` and
`e_i <= delta/10 = 0.001005033585350145`. For each eight-seed row, enumerate
all `2^8 = 256` corner vectors `d_i + s_i*e_i`, with every
`s_i in {-1,+1}`. At every corner compute the ordinary denominator-7 training
seed interval using `t_(0.975,7)=2.364624251592784`; the estimator-aware row
interval is the minimum lower endpoint and maximum upper endpoint across all
256 corners. PRIMARY and MAP decisions use only this inflated interval. Any
endpoint or paired-uncertainty failure leaves the affected row INCONCLUSIVE,
even when the uninflated interval would pass.

The primary paired contrast is `d_i=NLL_A0,i-NLL_A5,i` over eight seeds. The
training-seed interval uses `t_(0.975,7)=2.364624251592784` and is conservatively
enveloped over estimator-error boxes. Lower bound greater than
`delta=-log(0.99)=0.01005033585350145` is PASS; upper bound at most zero is FAIL;
otherwise the complete result is INCONCLUSIVE. Perplexity is secondary.

## Frozen prediction and attribution matrix

Every row uses confirmatory seeds `2026072101..2026072108`, one terminal
checkpoint per endpoint/seed, exact endpoint Prefix-certificate keys, and the
all-or-none global test opening. Each endpoint must pass the 1% parameter, 5%
whole-schedule FLOP, and exact optimizer-access checks; otherwise the row is
ineligible/INCONCLUSIVE rather than relaxed. "Shared A5" means that both
endpoints use the A5-primary selected `(lr,wd)` and estimate a factor
intervention conditional on that optimizer setting.

| ID | Left exact config / factory | Right exact config / factory | Sole config factor changed | Hyperparameter estimand | Interpretation |
|---|---|---|---|---|---|
| `PRIMARY` | `h6-a0-ar-v1` / `build_a0@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Whole declared architecture | Equal six-cell tuning per endpoint | Primary A0--A5 predictive contrast; not component attribution. |
| `MAP` | `h6-a2-generic-map-v1` / `build_a2@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Generic map versus typed same-point map restriction | Equal six-cell tuning per endpoint | Conditional map attribution only; never H7 covariance. |
| `STRUCTURE` | `h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Recognition precision structure | Shared A5 | Recognition-family effect conditional on A5 tuning. |
| `PRIOR` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-prefix-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Fixed versus prefix-conditioned generative source prior | Shared A5 | Descriptive changed-joint contrast; right endpoint requires a separate H1 rerun. |
| `MIXTURE` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-projection-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Exact mixture versus declared moment projection | Shared A5 | Descriptive approximation contrast with a projection-error record. |
| `OBJECTIVE` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-emission-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Complete ELBO versus emission-only optimization | Shared A5 | Optimization-objective intervention; the emission endpoint is not an ELBO. |
| `LATENT` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1` / `build_a5@h6-arm-v1` | Latent channel enabled versus disabled | Shared A5 | Descriptive because disabling latents changes the model and active capacity allocation; recognition is structurally absent on the right. |
| `RECOGNITION` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-filtering-v1` / `build_a5@h6-arm-v1` | Training recognition conditioning | Shared A5 | Training-regime effect; neither endpoint supplies held-out predictions from recognition. |

A1, A3, and A4 remain independently tuned, matched descriptive controls in the
six-arm report. No endpoint, seed, checkpoint, or certificate may be silently
substituted. A descriptive row cannot support a causal or H7 claim even when
its interval excludes zero. MAP attribution additionally requires PRIMARY PASS
and an inflated MAP lower bound above zero.

## Arms, controls, and nonclaims

Arms are A0 conventional autoregressive, A1 ordinary latent, A2 matched generic
map, A3 immediate-predecessor/source-free, A4 model-channel-free, and A5 full
H6. Parameter counts must be within 1% of A5 and whole-schedule FLOPs within 5%,
with every active parameter in exactly one optimizer and no filler. Factorial
reports isolate recognition structure, source-prior form, source mixture,
complete ELBO versus emission-only, latent enablement, and smoothing versus
filtering.

H6 does not claim H7 covariance, orientation-reversing `GL(2)`, optimizer
superiority, exact mixture when projection is used, universal leakage from
finite cases alone, WikiText-103 behavior, GPT-2-tokenizer behavior, or empirical
prediction until the separately authorized artifacts and ledgers close.

## Separate artifact schemas and ledger paths

### Independent Prefix artifact: `h6-prefix-v1`

The immutable Prefix artifact contains exactly:

```text
config.json
provenance.json
environment.json
validation/h6_prefix.json
certificates/prefix_set.json
manifest.sha256
```

It is published with `predecessor_refs={}` and contains only exact H6
source/config/model-family/vocabulary/estimator/fixture/data-safety identities,
the complete dynamic/static/cache validation payload, concrete certificates,
and its manifest. It contains no H1--H5 reference or status, H1 prefix-prior or
SMC-accuracy input, H6 training schedule, matching/tuning/capacity result,
checkpoint, opening, or prediction claim. Its revision-specific claim ledger is

```text
.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json
```

### Prediction-readiness artifact: `h6-prediction-readiness-v1`

Prerequisite status is never a caller flag. Each referenced producer payload is
canonical JSON that binds its `gate`, `status`, `obligations`, `git_head`,
`dirty_digest`, and `config_sha256`; PASS is derived only when that parsed
payload says `status="pass"` with no obligation. Its sorted LF manifest must
content-bind `config.json` and the exact validation member. Correctness members
are exactly `validation/h1.json`, `validation/h2.json`,
`validation/h3.json`, and `validation/h5.json`. The H1 prefix-prior reference
additionally binds `schemas/generative_factor.json` and
`validation/h1_prefix_prior.json`. The finite-SMC reference additionally binds
`protocol/estimator.json`, `fixtures/finite_smc.json`, and
`validation/h6_smc_accuracy.json`. Readiness retains and revalidates these
typed references, the exact H5 update binding, H6 training schedule,
endpoint-SMC protocol, data identity, and Prefix certificates; bare hashes
cannot mint a PASS token.

Prediction readiness is a separate atomic `h6_prediction_readiness.json` under
its own immutable manifest. It records `status="PASS"`, exact `git_head` and
`dirty_digest`, experiment-config identity, the H1/H2/H3/H5 correctness
manifest tuple, the required H1 prefix-prior manifest for the frozen matrix,
the exact H5 producer binding, H6 training schedule, finite-SMC manifest,
critical-values and endpoint-SMC protocol identities, attribution matrix,
matching set, Prefix certificate set, data identity, access policy, and its own
readiness digest. It references prerequisite artifacts and their manifests; it
does not copy their payloads, include H4, alter Prefix closure, materialize
train data, or contain empirical metrics. Readiness introduces no third claim
ledger; its obligations close within the later H6-Prediction ledger.

### Empirical Prediction artifact: `h6-prediction-v1`

The final metrics child uses canonical schema `h6-prediction-metrics-v1` and
contains `estimator_complete` plus the exact `primary_interval` object with
finite float `lower` and `upper` endpoints (or `null` only while estimator
evidence is incomplete). `PredictionDecision` is derived from those parsed
metrics by the frozen interval rule. The final result hashes and retains the
same metrics bytes; no independent caller-supplied status or interval is
accepted.

The Prediction artifact references the exact readiness artifact and immutable
parent manifests. Its atomic child records cover sealed/materialized data
identities, `tuning_selection.json` with all candidate cells and the selection
trace, typed training attempts and failures, validation metrics, terminal
checkpoints, the complete checkpoint-set identity, the immutable exclusive test
reservation, all endpoint/stream/particle records, uncertainty aggregation, and
the final `test_opening_result.json`. The reservation is never rewritten or
deleted. A completed opening has exactly `96*64*4=24,576` corpus records; any
missing, duplicate, or nonfinite record is represented by the frozen failure
semantics rather than silently repaired. Its revision-specific claim ledger is

```text
.verification/h6-prediction-<FULL_HEAD>-<EXPERIMENT_SHA>-ledger.json
```

The existing `.verification/ledger.json` and every prior revision-specific
ledger remain byte-for-byte unchanged. Neither new ledger path may be
overwritten or repointed; a replacement evidence revision receives a new path.
No unified H1--H6 validation artifact or ledger is created.

## Deferred artifact fields

The following are measured and frozen only before their evidence revision, not
selected using prediction outcomes: official archive/member/token/window hashes,
validation-fixture hash, complete Prefix certificate-set hash, matched endpoint
dimensions/parameters/FLOPs, actual checkpoint hashes, endpoint stream registry,
experiment/checkpoint-set/reservation/result hashes, uncertainty bounds, and
final metrics. These fields remain explicitly `DEFERRED` during source buildout;
their absence is an evidence state, not a fabricated value, zero hash, inferred
measurement, or PASS. Each is filled only before or during its separately
authorized exact-revision evidence operation according to the frozen protocol.
