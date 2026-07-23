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

## Data and access boundary

The bounded H6 corpus is the exact official WikiText-2 raw archive URL. Public
configuration cannot select a mirror or opener. Synthetic tests inject bytes
only through a private, non-exported opener. Actual network acquisition and
measured archive/member hashes are deferred evidence.

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

The primary paired contrast is `d_i=NLL_A0,i-NLL_A5,i` over eight seeds. The
training-seed interval uses `t_(0.975,7)=2.364624251592784` and is conservatively
enveloped over estimator-error boxes. Lower bound greater than
`delta=-log(0.99)=0.01005033585350145` is PASS; upper bound at most zero is FAIL;
otherwise the complete result is INCONCLUSIVE. Perplexity is secondary.

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

## Deferred artifact fields

The following are measured and frozen only before their evidence revision, not
selected using prediction outcomes: official archive/member/token/window hashes,
validation-fixture hash, complete Prefix certificate-set hash, matched endpoint
dimensions/parameters/FLOPs, actual checkpoint hashes, endpoint stream registry,
and final metrics. Their absence during source buildout is an explicit deferred
evidence state, not a fabricated value or PASS.
