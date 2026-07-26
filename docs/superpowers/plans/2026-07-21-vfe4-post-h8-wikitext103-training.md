# VFE 4.0 Post-H8 WikiText-103 Training, Recording, and Figure Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After one exact, validated H8 PASS exists, build the click-to-run WikiText-103 data, training, evaluation, checkpoint, recording, artifact, and deterministic figure pipeline for the zero-dimensional VFE 4.0 language model, then execute the separately authorized preregistered multi-seed experiment without transferring H6 byte-tokenizer evidence or H8 synthetic allocation evidence.

**Architecture:** A new VFE4-owned WikiText-103 cache begins from the official raw archive, seals each split behind typed capabilities, binds a pinned GPT-2/tiktoken specification, and produces deterministic exactly-once causal windows. Immutable factories create the H6-defined target-blind VFE4 prior scorer and H5-labeled training phases; atomic run/checkpoint/metric manifests make every result resumable and auditable. A separate import-safe click launcher renders figures only from immutable recorded metrics and run-group manifests, never by importing training or recomputing metrics from checkpoints.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, a source-lock-validated candidate `tiktoken==0.12.0`, candidate-pinned `matplotlib==3.10.6` with the noninteractive `Agg` backend, ZIP/CRC/SHA-256 validation from the standard library, frozen dataclasses and `Literal` types, JSON/JSONL/CSV, OS-specific tested durability backends, pytest with one final JUnit XML, and the installed evidence-gated verification ledger.

## Global Constraints

- **The 2026-07-25 arm/gate amendment is normative.**
  `docs/preregistrations/2026-07-25-post-h8-arm-gate-amendment.md`
  supersedes every independently entered two-arm count in this plan. All
  tuning attempts, terminal checkpoints, scoring records, result rows,
  resource work counts, and figure series are derived from one immutable
  `EndpointInventory` and its SHA-256.
- **This plan is post-H8.** No implementation task, data acquisition, source-lock operation, smoke run, or experiment in this plan may begin until an existing `validation/h8.json` with `schema_version="h8-sparse-scale-v3"` has `status="pass"`, its atomic manifest validates, its exact `(git_head, dirty_digest, config_sha256)` is recorded, and its revision-specific H8 claim ledger validates. Plan authoring itself is not execution.
- The H8 artifact remains a synthetic `T=128`, `K=d_z=d_m=20`, CPU-float64 systems certificate. It is **not** evidence that the WikiText-103 decoder, batches, autograd graph, optimizer states, token caches, evaluation, GPU kernels, checkpoints, or figures are sparse or within any memory budget. The post-H8 candidate must measure those paths independently.
- The language model remains the VFE4 singleton-base specialization `C0={*}` with labeled population positions and a separate causal DAG. Token positions, data windows, and source edges are not base points, base transport, base curvature, or base holonomy.
- The normative state objective remains one complete normalized ELBO. Held-out NLL/PPL comes only from the causal target-blind generative prior predictor. Posterior reconstruction, emission-only loss, samples, and diagnostic free-energy quantities are separate labeled outputs.
- H5 labels control guarantees. The frozen first experiment uses `adam_proposal` for recognition and model blocks, plus `immutable_detached_snapshot`; these labels do not imply exact coordinate ascent, MM, GEM, or monotonic ELBO improvement. Every accepted/rejected proposal records its actual rule and objective evidence.
- The configured VFE4 path uses reverse-mode autograd in its active recognition and model proposal blocks. Record expected and observed scope exactly and call this backpropagation. Do not call it forward-mode, forward-gradient, clean EM, pure FEP, or backprop-free.
- H6-Prefix proved safety only for exact H6 model-family, vocabulary, predictor-config, estimator, and cache identities. Its byte tokenizer (`V=258`) does not certify GPT-2/tiktoken (`V=50,257`), WikiText-103 caches, the post-H8 model shape, or a new prefix cache. A new `WT103-Predictor-Safety` certificate is mandatory before empirical materialization.
- H6-Prediction remains predecessor evidence for the selected scorer/update/statistical design, not a transferable WikiText-103 predictive result. Post-H8 readiness binds exact H5, H6, H7, and H8 artifact/ledger identities and reruns every configuration-sensitive safety check at the final training candidate.
- Post-H8 readiness requires a distinct same-revision `TrainingSparsityCertificate`. H8 cannot satisfy it, and a device-capacity smoke below 85% cannot satisfy it. Structural shape/allocation safety and available-capacity safety are independent conjuncts.
- Tasks 1--12 are hermetic with respect to the candidate tokenizer package. They may store only the candidate strings `distribution="tiktoken"`, `version="0.12.0"`, and `encoding_name="gpt2"`, and may exercise only injected synthetic distribution/table/tokenizer adapters. They must not import `tiktoken`, query its live version or `importlib.metadata` distribution/RECORD files, read live regex/rank/special-token tables, or run the production golden vectors. Those live checks occur together, for the first time, in Task 13 Step 3; a pre-source-lock `TokenizerSpec` exists only as synthetic fixture evidence and cannot satisfy readiness.
- Every validation-boundary checkpoint is labeled `role="resume_only"`. It may restore the same attempt after a permitted interruption but cannot be selected, scored for confirmation/test, placed in the hash-derived endpoint inventory, or rendered. Only the post-pass terminal checkpoint labeled `role="terminal_scoring"` is eligible for confirmatory/test scoring.
- The user surface requires no CLI. `train_vfe4.py` and `generate_vfe4_figures.py` each expose one editable top-level `CONFIG`, one `main()`, and one `if __name__ == "__main__": main()` guard. Importing either file performs no parsing, I/O, device initialization, data access, run reservation, rendering, or training.
- Every launcher dictionary is resolved once into frozen typed sections. Unknown keys, missing required keys, wrong plain-Python scalar types, invalid combinations, stale schema/objective identities, or derived-field overrides fail before side effects. Runtime code never rereads or mutates the dictionary.
- V3 is a read-only rough engineering guide. VFE4 imports no V3 module, reads no V3 checkpoint, accepts no V3 config, and never discovers or reuses V3's filename-inferred/unverified `~/.cache/tokenized_cache/wikitext-103_*` files. V3 data/objective/checkpoint/metric schemas are not migration inputs.
- Before the user authorizes official acquisition and later real training, tests and launcher smokes use only tiny generated text/archive fixtures under pytest temporary directories. No task silently downloads, expands to the real corpus, maps a real test cache, or launches real optimization.
- Each implementation task runs only its named focused RED/GREEN commands and ends in one bounded review/commit. After the authorized source lock and all tracked code/docs are frozen, run one full suite exactly once with one JUnit XML at the final integration candidate. Reviewers consume the existing XML and artifacts; they do not rerun the broad suite.
- Generated caches, run artifacts, JUnit files, and ledgers are not committed. Source-lock and preregistration documents are committed before the integration candidate. Existing predecessor artifacts and ledgers remain immutable.

---

## Normative Sources and Read-Only Context

- VFE4 design, user surface, autograd policy, runtime flows, V3 boundary, artifact contract, and promotion ladder: `docs/superpowers/specs/2026-07-21-vfe4-codebase-design.md`.
- H5 update taxonomy, complete-objective evaluation, immutable snapshot, acceptance/rollback, and provenance: `docs/superpowers/plans/2026-07-21-vfe4-h4-h5-cost-update.md`.
- H6 target-blind predictor, explicit arms, data capabilities, update schedule, tuning/seeds/stopping, SMC uncertainty, one test opening, and nonclaims: `docs/superpowers/plans/2026-07-21-vfe4-h6-prefix-prediction.md`.
- H7 law/ELBO covariance and exact predecessor identity: `docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md`.
- H8 exact PASS schema, synthetic scope, and explicit post-H8 training boundary: `docs/superpowers/plans/2026-07-21-vfe4-h8-sparse-scale.md`.
- Normative language theory: `Manuscripts/vfe4_whitepaper/01_executive_scope.tex`, `03_bundle_geometry.tex`, `04_generative_model.tex`, `05_structured_information_form.tex`, `06_elbo_coordinate_updates.tex`, `07_transformer_crosswalk.tex`, `08_hypotheses_limitations.tex`, and `09_appendices.tex`.
- Related finite-law/update boundaries: `Manuscripts/magent_elbo_whitepaper/04_generative_model.tex`, `05_structured_recognition_elbo.tex`, `08_information_geometry_gauge.tex`, `10_executable_crosswalk.tex`, and `11_vfe4_comparison.tex`.
- Read-only Research wiki context: `[[VFE Transformer Program]]`, `[[Inference machinery -- variational EM and filtering]]`, and `[[Language Modeling]]`. The wiki cautions that V3's structural-EM schedule does not establish one shared ELBO, multi-seed provenance is required for claims, and optimizer/capacity/config changes defeat narrow causal attributions.
- Rough V3 engineering seams only: `C:/Users/chris and christine/Desktop/V3_Transformer/train_vfe3.py`, `vfe3/data/datasets.py`, `vfe3/run_artifacts.py`, `vfe3/viz/figure_worker.py`, `vfe3/viz/figures.py`, and `make_figures.py`.

## Frozen Evidence Claims and Nonclaims

### Claims this milestone may close

1. The exact official WikiText-103 raw archive and three split members were acquired, validated, source/licensed-recorded, and reused offline only under exact byte identities.
2. The pinned GPT-2/tiktoken tokenizer and each encoded split have exact, independently verifiable identities with no learned or fitted cross-split state.
3. Windowing scores every adjacent transition in a split exactly once, masks only final padding, deterministically shuffles complete training windows, evaluates full validation/test streams, and resumes at the exact next batch.
4. The final post-H8 predictor configuration is target-blind under the new vocabulary/cache/model identity.
5. A checkpoint resumes to the same scientific weights, optimizer/scheduler states, RNG/data cursor, update trace, metric numerator/denominator state, and next predictions as uninterrupted execution for the bounded smoke fixture, while operational timestamps and artifact bytes remain separately identifiable.
6. Recorded WikiText-103 NLL/PPL and diagnostic figures are reproducible from immutable run manifests and recorded metrics at the exact evidence revision.
7. The empirical PRIMARY contrast is limited to the preregistered A0 reference and parent-specific pooled-prefix complete-objective endpoint, seeds, optimizer settings, capacity/compute matching report, stopping rule, estimator, and data identity. The three additional inventory arms are separately labeled gates or controls and cannot widen or rescue the PRIMARY claim.

### Claims this milestone must not make

- H8 proved training sparsity, GPU memory safety, decoder sparsity, optimizer sparsity, or an asymptotic scaling law.
- A dense vocabulary table or bounded decoder-logit chunk makes the population inference path dense; permitted vocabulary-axis density and forbidden population-axis density are separate claims.
- H6's byte-tokenizer safety/prediction result transferred to GPT-2/tiktoken or WikiText-103.
- VFE4 is backprop-free, forward-gradient trained, exact EM, or monotone under `adam_proposal`.
- Perplexity is computed from a target-conditioned recognition state, posterior reconstruction, emission-only diagnostic, selected best checkpoint, or averaged batch means.
- The PRIMARY A0-versus-parent-specific-complete comparison is a whole-architecture comparison even when training-compute matched; it does not isolate one mechanism.
- An optimizer-only causal conclusion follows when capacity, active phases, update labels, regularization, data order, stopping, or evaluation differs.
- A source-entropy or effective-source curve proves attention, causal discovery, gauge covariance, or useful source identification.
- Two passes establish a scaling law, state-of-the-art result, convergence, posterior correctness, calibration, or long-context generalization.
- V3 caches/checkpoints/objectives are compatible with VFE4, or matching a V3 tokenizer vocabulary makes either model equivalent.

---

## Frozen WikiText-103 Source, License, Archive, and Cache Contract

### Candidate official locations, verified only by source lock

- Candidate dataset request URL: `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip`.
- Candidate dataset announcement/source-page request URL: `https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/`.
- No mirror, Hugging Face transformation, TorchText download, prepared vocabulary, V3 cache, redirect target, or filename-compatible alternate response may substitute silently. HTTP redirects are recorded as an ordered chain and require the final origin to remain HTTPS; changed bytes are never auto-accepted.

The plan-authoring task is network-prohibited, so the two URLs, archive/member identities, source-page content, license text, and installed tokenizer distribution/table identity remain **candidate facts** until Task 13. During the separately authorized source-lock task, the implementation streams the candidate responses, validates both request-to-final redirect chains, computes the exact archive SHA-256 and byte size, validates every ZIP record, validates the installed tokenizer distribution/tables, and atomically writes `docs/data/wikitext103-raw-v1-source-record.json`. That machine-produced record is reviewed and committed before the one integration JUnit or any training. Thereafter those observed values are required literals for reuse; any ambiguity or changed response stops source lock for an explicit preregistration revision before outcomes, never an automatic cache refresh.

### Exact member inventory and safety envelope

The only accepted central-directory entries are the directory `wikitext-103-raw/` and these three regular files:

```text
wikitext-103-raw/wiki.train.raw
wikitext-103-raw/wiki.valid.raw
wikitext-103-raw/wiki.test.raw
```

The archive validator enforces all of the following before extraction:

- archive response body `<=268,435,456` bytes;
- ZIP methods only `ZIP_STORED (0)` or `ZIP_DEFLATED (8)`;
- no encryption, data-descriptor ambiguity, multi-disk archive, ZIP64 size beyond the declared limits, symlink/device/FIFO entry, duplicate name, case-colliding name, alternate separator, absolute/drive/UNC path, `.`/`..` segment, NUL, extra file, or unexpected directory;
- each regular member has positive compressed and uncompressed size, uncompressed size `<=671,088,640` bytes, compression ratio `<=100`, and total uncompressed regular-file size `<=805,306,368` bytes;
- streamed extraction never writes outside a newly reserved cache staging directory, never follows a link, never exceeds the declared member/total bounds, and verifies actual byte count, central-directory CRC32, recomputed CRC32, and SHA-256 before publication;
- each split is published to its own immutable content-addressed directory and manifest; training code receives no path that can be rewritten to another split.

The source record stores exact request URL, ordered redirect status/location/final-URL chain, status/content headers, retrieval UTC, archive size/SHA-256, ordered central-directory metadata, per-member path/size/compressed-size/method/flags/CRC32/SHA-256, source-page request/final URLs and captured SHA-256, verbatim license declaration, license link if the source page supplies one, citation text, validator schema/hash, installed tokenizer distribution metadata/table identities, and cache-relative paths. The archive response must have an accepted ZIP content type or a documented absent content type plus valid ZIP signature. The source page is bounded to `4,194,304` bytes, must declare `text/html` or `application/xhtml+xml` with UTF-8-compatible decoding, and may not use a script-rendered license as evidence.

License extraction is deterministic and audit-visible. Search the raw source-page bytes case-insensitively for ASCII `creative commons`; require exactly one occurrence inside one syntactically closed `<p ...>...</p>` byte span no larger than `4,096` bytes. Record the paragraph start/end byte offsets, raw-slice SHA-256, HTMLParser-derived visible text, and every `href` wholly contained in that span. Zero matches, multiple matches, malformed containment, a contradictory license elsewhere in the page, more than one plausible license link, or a response/content-type/redirect ambiguity stops Task 13 for a preregistration revision before tokenizer/window manifests or outcomes. No human chooses among ambiguous extracts.

Offline reuse is the default after source lock. `acquire_wikitext103()` first validates the committed source record, archive bytes, every member, token manifests, and directory containment. If all match it performs no network call. If anything is absent or mismatched and `allow_network=False`, it fails closed. If `allow_network=True`, it downloads only the source-locked request URL into a new staging path and never overwrites the prior cache.

### Tested durability backend

Every cache, source record, schedule, metric export, checkpoint, reservation, run manifest, and figure index uses one startup-probed `DurabilityBackend`; calling raw `Path.write_*`, `open(...,"w")`, or `os.replace` from domain modules is forbidden. Staging and destination must resolve to the same volume. The probe runs independently in cache, run, and figure roots, records backend schema/implementation hash, OS/build, filesystem/volume identity, supported operations, and create/replace/reopen SHA-256 results, then removes only its own uniquely named probe files. Failure or unknown semantics blocks before scientific state.

- **POSIX backend:** create the sibling staging file with `O_CREAT|O_EXCL` and mode `0o600`, write fully, flush, call `os.fsync(file_fd)`, verify bytes, call same-filesystem `os.replace`, open the containing directory with `O_RDONLY|O_DIRECTORY`, and call `os.fsync(directory_fd)`. `EINVAL`, `ENOTSUP`, or an unavailable directory-fsync contract is not silently ignored for a scientific root; the probe returns INCONCLUSIVE and readiness fails.
- **Windows backend:** use `CreateFileW(..., CREATE_NEW, FILE_ATTRIBUTE_NORMAL|FILE_FLAG_WRITE_THROUGH)` for staging and exclusive reservations, `WriteFile`, `FlushFileBuffers`, close/reopen/hash validation, then same-volume `MoveFileExW(..., MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)` for publication. Resolve volume identity with `GetVolumePathNameW` plus `GetVolumeInformationW`; a cross-volume path fails. The probe verifies exclusive-create collision, replacement, write-through reopen, and preservation of the old target after an injected pre-replace failure. The exact Win32 error code is recorded.
- A backend cannot claim crash consistency beyond its tested platform/filesystem operations. The artifact records the narrower claim: the required create/flush/replace/reopen sequence completed under the recorded backend identity. Unsupported network/FUSE/cloud-synced semantics remain INCONCLUSIVE.

### Split capability boundary

- Acquisition may hash and seal all raw members and may tokenize all splits in an isolated preprocessing subprocess, but exposes only opaque `SealedSplitRef` records.
- `TrainDataCapability` opens train plus validation tokens after post-H8 readiness passes. It cannot open test tokens.
- Training/tuning code cannot import the unsealing primitive and cannot receive a `TestDataCapability`.
- `DurableTestOpeningCapability` is created only after the platform `DurabilityBackend` has exclusive-created, flushed, published, and reopen-validated the reservation and all terminal checkpoints, analysis code, estimator protocol, and run-group manifests are frozen. It opens the test token stream once for the complete assessment. A crash after reservation is terminal and cannot be retried.

---

## Frozen Tokenizer and Window Contract

### Tokenizer decision

The first post-H8 WikiText-103 experiment proposes the following candidate tokenizer contract; Task 13 must verify and freeze the installed distribution and tables before it becomes an executable scientific fact:

```text
library distribution: tiktoken==0.12.0
encoding name: gpt2
vocabulary size: 50,257
ordinary token IDs: 0..50,255
declared special token: <|endoftext|> -> 50,256
encoding call for corpus text: encode_ordinary
allowed special tokens during corpus encoding: none
inserted BOS/EOS tokens: none
padding input token: 50,256
ignored target sentinel: -100
raw decoding: strict UTF-8, no newline/Unicode normalization
```

This is the preregistered comparison decision conditional on source-lock verification: it targets the standard GPT-2 vocabulary used by the V3 WikiText-103 engineering baseline while leaving VFE4 probability semantics independent. It is not evidence that V3's cache was correct or that the H6 byte vocabulary generalizes.

`TokenizerSpec` hashes the exact installed distribution name/version and distribution-file RECORD hashes, encoding name, regex pattern, ordered mergeable-rank byte strings/IDs, ordered special-token map, vocabulary size, `encode_ordinary` policy, padding/mask policy, and golden encode/decode vectors. Task 13 requires candidate `tiktoken==0.12.0` and commits the observed distribution/table/spec hashes; a version, metadata, table, regex, or golden-vector mismatch stops for a preregistration revision before windows or outcomes. Each split is decoded and encoded independently as the entire strict-UTF-8 member; no tokenizer fit, vocabulary learning, normalization statistics, or mutable state crosses splits. A split's round trip `decode_bytes(encoded_ids) == raw_member_bytes` is mandatory.

Token payloads are little-endian contiguous `int32` files because all IDs fit. Each split manifest stores raw identity, tokenizer spec/hash, exact token count, minimum/maximum ID, payload size/SHA-256, dtype/endianness, round-trip hash, builder code hash, and zero cross-split parents. Publication uses the probed `DurabilityBackend` staging/flush/replace/reopen sequence. The VFE4 cache root defaults to a VFE4-owned directory and rejects any resolved path under V3's cache root.

### Windows and exact token counts

The first experiment freezes `sequence_length=128` and `stride=128`. For a split token stream of length `n>=2`, let `transitions=n-1` and starts be `0,128,256,...` below `transitions`. Every window is:

```text
inputs  = tokens[start:start+128]
targets = tokens[start+1:start+129]
valid_target_mask = targets that came from the split, not padding
```

The final partial window is included exactly once. Unused inputs are `50,256`; unused targets are `-100`; `valid_target_mask` is false exactly at those target positions. Across the full split, `sum(valid_target_mask)==n-1`, every transition index `1..n-1` appears once, and no transition crosses a split. Every metric stores the numerator and this counted-target denominator.

Training contains every window and every final batch (`drop_last=False`) but permutes window IDs each pass. The permutation is generated by NumPy `PCG64` from the hashed tuple `(data_order_seed=2026072199, train_split_sha256, window_manifest_sha256, epoch_index)`, saved as a little-endian `uint64` schedule with its SHA-256, and bound to the NumPy version. Validation and test use ascending window IDs, no shuffle, no subsampling, and the full padded final window. The initial implementation uses `num_workers=0`; changing worker/prefetch/distributed policy is a new schedule identity.

`DataCursor` records split, pass, permutation SHA, next batch ordinal, exact next window IDs, cumulative valid targets, batch size, and schedule schema. Resume validates the stored permutation bytes and replays the exact next batch; it never regenerates a different order from a nominal seed.

---

## Literal `WT103ExperimentProfile` v1

The scientific profile is executable without outcome-dependent choices. The only model-shape search is the finite A0 capacity-match rule below; it uses parameter/FLOP formulas only and is frozen in Task 1 before corpus outcomes.

```text
profile_schema                      = "wt103-experiment-profile-v1"
dataset_schema                      = "wikitext-103-raw-v1"
tokenizer_schema                    = "gpt2-tiktoken-v1"  # candidate until Task 13
vocabulary_size                     = 50_257
sequence_length                     = 128
stride                              = 128
batch_size                          = 128
gradient_accumulation_steps         = 1
num_workers                         = 0
pin_memory                          = true
drop_last                           = false
model_depth                         = 1
d_z                                 = 20
d_m                                 = 20
K                                   = 20
combined_latent_block               = 40
source_lookback                     = 20
state_parents(t)                    = range(max(0,t-20),t)
model_parents(t)                    = range(max(0,t-20),t)
population_frame_profile            = "h7-direct-glplus-v1"
decoder_profile                     = "categorical_linear_chunked"
decoder_train_token_chunk           = 512
decoder_eval_token_chunk            = 256
smc_particle_chunk                  = 32
dropout_probability                 = 0.0
input_output_embedding_tied         = false
real_training_device                = "cuda:0"
parameter_dtype                     = "float32"
optimizer_state_dtype               = "float32"
autocast_enabled                    = true
autocast_dtype                      = "bfloat16"
grad_scaler_enabled                 = false
grad_scaler_fixed_scale             = 1.0
spd_factor_solve_logdet_dtype       = "float32"
smc_log_weight_dtype                = "float64"
metric_corpus_accumulator           = "python_math_fsum_float64"
torch_deterministic_algorithms      = true
cudnn_deterministic                 = true
cudnn_benchmark                     = false
allow_tf32_matmul                   = false
allow_tf32_cudnn                    = false
allow_fp16_reduced_precision_reduce = false
cublas_workspace_config             = ":4096:8"  # launcher sets before CUDA initialization
optimizer                           = "AdamW"
adam_betas                          = (0.9,0.999)
adam_epsilon                        = 1.0e-8
adam_amsgrad                        = false
adam_foreach                        = false
adam_fused                          = false
gradient_clip                       = "per_active_block_global_l2"
gradient_clip_max_norm              = 1.0
learning_rate_grid                  = (1.0e-4,3.0e-4,1.0e-3)
weight_decay_grid                   = (0.0,1.0e-2)
scheduler                           = "linear_warmup_then_cosine"
scheduler_warmup_optimizer_steps    = 100
scheduler_min_lr_ratio              = 0.1
scheduler_restart_count             = 0
scheduler_horizon                   = "planned_active_optimizer_steps_for_attempt"
proposal_acceptance                 = "validity_only_no_monotonicity_claim"
reject_on                           = (nonfinite_objective,nonfinite_gradient,amp_overflow,
                                      invalid_support,non_spd,scope_mismatch,snapshot_alias,
                                      optimizer_access_mismatch)
validation_boundaries_per_pass      = 20
checkpoint_at_every_validation      = true
rolling_resume_checkpoints_retained = 2
rolling_checkpoint_role             = "resume_only"
terminal_checkpoint_retained        = true
terminal_checkpoint_role            = "terminal_scoring"
best_checkpoint_selection           = false
confirmatory_passes                 = 2
early_stopping                      = false
```

This profile contains only run-group fields shared across the inventory.
`training_objective`, `prior_variant`, `source_mixture`,
`recognition_family`, `recognition_iterations_per_batch`, latent/recognition
applicability, scorer kind, update phases, and result role live only in each
immutable `WT103ArmSpec`. A consumer that reconstructs one of those values
from an A0/A5 family label instead of the arm spec fails configuration
resolution.

The real profile requires CUDA and does not fall back to CPU or another GPU. CPU float64 appears only in generated-fixture tests and is a separate non-scientific `smoke-v1` profile. Parameters and optimizer states remain float32; bfloat16 autocast is limited to declared dense contractions. SPD factorizations/solves/log determinants stay float32, and SMC log weights plus corpus accumulation use the stated higher-precision paths. Because bfloat16 uses no dynamic scaler here, every update records `amp_scale=1.0`, `amp_overflow=false|true`, and `grad_scaler_applicability="disabled_bfloat16"`; an overflow/nonfinite event rejects and rolls back the proposal.

### Frozen `A0ArchitectureProfile` v1

The conventional comparator has one exact architectural meaning. Its selected hidden width `h` is the output of the already frozen finite, corpus-free match rule; every other field below is literal, and changing one changes `a0_architecture_sha256` and requires a preregistration revision.

```text
schema_version                      = "wt103-a0-architecture-v1"
block_count                         = 1
hidden_width                        = selected_h_from_finite_match_rule
attention_heads                     = 2
head_width                          = h // 2             # h must be even
attention_context                   = "full_causal_inclusive_self"
attention_allowed_keys(q)           = range(0,q+1)
attention_semantic_pair_count       = L*(L+1)//2
attention_implementation            = "torch.nn.functional.scaled_dot_product_attention"
attention_backend_policy            = "flash_attention_only_no_fallback"
source_lock_pytorch_api_binding      = "torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])"
source_lock_alternative_backends     = "disabled_by_single-backend_context"
source_lock_pytorch_version          = observed_and_frozen_in_Task_13
source_lock_sdpa_api_sha256          = observed_and_frozen_in_Task_13
source_lock_flash_backend_sha256     = observed_and_frozen_in_Task_13
attention_is_causal                 = true
attention_mask_argument             = null
attention_scale                     = 1/sqrt(head_width)
attention_dropout_probability       = 0.0
attention_returns_weights           = false
grouped_query_attention              = false
backend_fallback_allowed             = false
fused_full_attention_allowed         = true
fused_attention_materialization      = "forbidden"
token_embedding                      = "learned[V,h]_no_bias"
positional_encoding                  = "learned_absolute[L_max=128,h]"
position_interpolation               = false
input_composition                    = "token_embedding_plus_position_embedding"
normalization                        = "LayerNorm(eps=1e-5,elementwise_affine=true,bias=true)"
normalization_placement              = "pre_norm_with_final_norm"
residual_topology                    = "x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"
qkv_projection                       = "Linear(in=h,out=3h,weight[3h,h],bias[3h])"
attention_output_projection          = "Linear(in=h,out=h,weight[h,h],bias[h])"
mlp_input_projection                 = "Linear(in=h,out=4h,weight[4h,h],bias[4h])"
activation                           = "gelu_tanh_approximation"
mlp_output_projection                = "Linear(in=4h,out=h,weight[h,4h],bias[h])"
decoder_projection                   = "untied_Linear(in=h,out=V,weight[V,h],bias[V])"
all_dropout_probabilities            = 0.0
parameter_formula_schema             = "wt103-a0-parameter-formula-v1"
flop_formula_schema                  = "wt103-a0-semantic-train-flops-v1"
a0_formula_sha256                    = canonical_hash(parameter_and_flop_formula_records)
a0_architecture_sha256               = canonical_hash(all_resolved_fields_formula_hash_and_frozen_pytorch_api_backend_identity)
```

The project-owned policy literal `flash_attention_only_no_fallback` is semantically full causal attention but names no library enum. Task 13 binds it to the exact installed PyTorch version/API source identity and the real context `torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])`, whose single-backend list disables math, memory-efficient, and cuDNN alternatives; those identities enter `a0_architecture_sha256`. The call has no attention-weight output or explicit mask. It is admissible only if same-revision readiness proves exact Flash selection and the training-sparsity trace proves no `[L,L]`, `[B,L,L]`, `[B,2,L,L]`, flattened equivalent, or aggregate pair-axis attention storage in forward or backward. API/backend unavailability, another selected backend, fallback, nondeterminism, materialized weights/mask, or an opaque workspace that cannot be classified makes readiness INCONCLUSIVE.

For `V=50_257` and learned positional capacity `L_max=128`, the exact active-parameter formula is

```text
P_A0(h) = token_embedding Vh
        + position_embedding 128h
        + QKV (3h^2 + 3h)
        + attention_output (h^2 + h)
        + MLP (8h^2 + 5h)
        + three affine LayerNorms (6h)
        + untied_decoder (Vh + V)
        = 2Vh + 128h + 12h^2 + 15h + V.
```

`A0FormulaRecord` is an analytical semantic ledger, not a profiler estimate. It freezes `1 multiply=1 FLOP`, `1 add/subtract/divide/exp/log/tanh/rsqrt=1 FLOP`, comparisons and indexing `=0`, linear `F(m,k,n,bias)=2mkn+mn` when biased, causal QK-plus-AV `=4*B*[L(L+1)/2]*h`, causal softmax `=B*2*(4*[L(L+1)/2]-L)`, affine LayerNorm forward `=B*L*(7h+2)`, tanh-GELU forward `=9*B*L*4h`, and exact separately enumerated backward, cross-entropy, embedding-scatter, residual, and AdamW primitive records. Whole-attempt `F_A0` sums those records over the literal batch/window/cadence schedule; decoder chunking changes peak storage but not FLOPs. Task 6 must reconstruct `P_A0` from actual named parameter shapes and `F_A0` from an independent hand-enumerated tiny operator ledger, then match both byte-for-byte to the canonical formula record/hash. Profiler FLOP estimates cannot close this check.

Initialization is dispatched from the immutable arm spec. Every latent row uses seed-local Xavier uniform gain 1 for embeddings and decoder weights, zero categorical/linear biases and source logits, identity primary frame matrices, identity block-precision diagonal factors, and zero lower factors. A0 and no-latent instantiate only their declared parameters and may not allocate dormant source, frame, precision, or recognition state; they use the same applicable initializer classes. Every initializer consumes an explicitly named counter substream and records its terminal counter.

The A0 factory is `build_a0@wt103-arm-v1` and implements the exact `A0ArchitectureProfile` above, with no latent/recognition/source/frame state and hidden-width candidates exactly

```text
(20,24,28,32,36,40,44,48,52,56,60,64,72,80,96,112,128,160).
```

Evaluate every candidate without corpus tensors using the canonical parameter/FLOP formulas. Let `P_PRIMARY` and `F_PRIMARY` denote only the `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1` endpoint selected through its `PRIMARY_ENDPOINT` role. Eligible A0 candidates satisfy `abs(P_A0/P_PRIMARY-1)<=0.01`, `abs(F_A0/F_PRIMARY-1)<=0.05`, exact optimizer access, and no filler state. Select the unique minimum tuple `(abs(log(P_A0/P_PRIMARY)), abs(log(F_A0/F_PRIMARY)), h)`; only after selection does readiness test that exact `h` on the frozen Flash backend. Backend failure stops rather than selecting a different width. If no formula candidate is eligible, readiness is INCONCLUSIVE and this plan must be revised before outcomes. PRIMARY-endpoint dimensions never move to rescue A0 matching.

A0 and the no-latent control execute `model_ce_adam_proposal`. Each latent A5 row executes exactly one `recognition_adam_proposal`, one immutable detached snapshot, and one `model_adam_proposal` per batch under its frozen objective. Each `adam_proposal` applies AdamW and its scheduler only after all validity checks pass; objective decrease alone neither rejects nor licenses monotonicity. Rejection restores parameters, optimizer moments/step, scheduler ordinal/LR, RNG, AMP state, and update counters exactly.

Validation runs at the 20 boundaries already frozen; checkpointing occurs after the metric/failure ledgers are durable at each boundary. The two newest rolling checkpoints are `role="resume_only"`: they may resume that same attempt and are categorically ineligible for validation selection, confirmatory/test scoring, the hash-derived endpoint inventory, aggregation, or figures. Only the post-pass checkpoint is `role="terminal_scoring"`. Removing an older resume-only checkpoint is a run-owned lineage event performed only after its successor and terminal lineage record are durable; its identity remains in the immutable lineage, and no externally referenced checkpoint is removed.

---

## Frozen Experiment, Update, Stopping, and Statistical Policy

### Arms and interventions

The first WikiText-103 run group uses the exact ordered
`WT103ArmSpec` inventory from the 2026-07-25 amendment:

- `WT103-A0-AR-v1`;
- `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1`;
- `WT103-A5-FIXED-COMPLETE-v1`;
- `WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1`;
- `WT103-A5-NOLATENT-v1`.

The exact source rows are:

| Arm | Factory | Objective | Prior | Latent / recognition | Scorer | Role |
|---|---|---|---|---|---|---|
| `WT103-A0-AR-v1` | `build_wt103_a0@wt103-arm-v1` | `cross_entropy` | `absent` | false / false | `exact_autoregressive` | `PRIMARY_REFERENCE` |
| `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1` | `build_wt103_a5_parent_specific@wt103-arm-v1` | `complete_elbo` | `parent_specific_pooled_prefix` | true / true | `weighted_smc` | `PRIMARY_ENDPOINT` |
| `WT103-A5-FIXED-COMPLETE-v1` | `build_wt103_a5_fixed@wt103-arm-v1` | `complete_elbo` | `fixed` | true / true | `weighted_smc` | `PRIOR_CONTROL` |
| `WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1` | `build_wt103_a5_parent_specific@wt103-arm-v1` | `emission_only_ablation_non_elbo` | `parent_specific_pooled_prefix` | true / true | `weighted_smc` | `OBJECTIVE_GATE` |
| `WT103-A5-NOLATENT-v1` | `build_wt103_a5_nolatent@wt103-arm-v1` | `cross_entropy` | `absent` | false / false | `exact_autoregressive` | `LATENT_PATH_CONTROL` |

Every row uses `tuning_grid_id="wt103-six-cell-v1"`, confirmatory seeds
`2026072101..2026072108`, and
`terminal_checkpoint_role="terminal_scoring"`. The two parent-specific rows
share the selected capacity, model family, normalized prior law, scorer,
particle protocol, optimizer policy, data order, and checkpoint schedule;
their only scientific intervention is the objective. The no-latent row may
not retain dormant recognition parameters or fake SMC work.

Each latent row additionally freezes `source_mixture="exact"`,
`recognition_family="structured_block_tridiagonal_smoothing"`,
`recognition_iterations_per_batch=1`, and update phases
`("recognition_adam_proposal","immutable_detached_snapshot",
"model_adam_proposal")`. A0 and no-latent freeze those three latent fields as
`"absent"`, `0`, and update phases `("model_ce_adam_proposal",)`. These are
arm-spec fields, not inferred defaults.

Every applicable arm uses the same tokenizer, window IDs/order, batch size,
passes, model-update opportunities, validation/checkpoint boundaries,
terminal-selection rule, and test opening. Before tuning, corpus-free matching
selects the eligible nuisance allocation without inspecting corpus outcomes.
Every active parameter appears exactly once in its declared optimizer; no
dormant/filler/no-op parameters or phases are allowed.

The PRIMARY A0 versus parent-specific complete report is a
training-compute-matched whole-architecture contrast, not component
attribution. Fixed-prior complete is a changed-joint control,
parent-specific emission-only is the ordered OBJECTIVE gate, and no-latent is
a bundled latent-path control unless exact held-fixed semantics are later
proved. These rows retain distinct labels and never rescue or promote PRIMARY.

### Update/autograd contract

- A0 expected/observed autograd scope is `m_step` and has no recognition state or recognition optimizer.
- Every latent A5 inventory arm has expected/observed scope `e_and_m`. Recognition proposal tensors are the only active E-like leaves while model parameters are frozen. The accepted recognition result is cloned into an immutable nonaliasing `RecognitionSnapshot`, detached, and hashed. Model parameters are then the only active M-like leaves. The no-latent control has no recognition optimizer.
- Every applicable active optimizer label is `adam_proposal`, with the exact H5 objective/update/snapshot/factor-dependency schema hashes. A proposal is accepted only if the configured rule says so and all finite, gradient, support, SPD, and immutable-snapshot checks pass. Rejections restore every affected parameter/optimizer/scheduler state and record reason plus before/after objective numerators and estimator budget.
- Runtime instrumentation hard-fails if observed autograd scope differs from resolved scope, if a recognition snapshot aliases model storage, if a target/suffix reaches the prior scorer, if any active parameter is missing/duplicated across optimizers, or if an update is reported under a stronger label than executed.

### Tuning, confirmation, stopping, test, and inference

The H6 policy is refrozen for the new dataset identity:

- equal tuning grid for every trainable inventory arm: `learning_rate in {1e-4,3e-4,1e-3}` by `weight_decay in {0,1e-2}`;
- tuning seeds `2026072199` and `2026072200`, each exactly the first `ceil(training_batches/4)` batches of the frozen pass permutation;
- select lowest mean full-validation prior-predictive NLL/token; tie-break by lower learning rate, then lower weight decay; publish every cell;
- confirmatory initialization/run seeds `2026072101..2026072108`, shared data-order seed `2026072199`;
- exactly two complete passes, validation at `ceil(k*batches_per_pass/20)` for `k=1..20` in each pass after stable deduplication, no early stopping, no adaptive extension, and no best-validation checkpoint selection;
- rolling validation checkpoints are resume-only; only the post-pass terminal checkpoint is eligible for confirmation/test. Validation curves are diagnostics and do not select or promote a checkpoint;
- every tuning/validation score uses the arm's frozen `scorer_kind`. A weighted-SMC row uses exactly `N=256` particles and common validation stream IDs `0..7`, derived from `SHA256("post-h8-wt103-validation-v1|2026072198|stream_id|purpose")`; aggregate each stream by corpus-summed log normalizers and counted targets, then use the arithmetic mean of the eight corpus NLLs. An exact-autoregressive row is evaluated once and may be replayed only as an identity check, never assigned fake variance;
- at most one infrastructure retry, only with proof of no advancement or exact checkpoint restore. Numerical divergence, nonfinite values, model/estimator/prefix/capacity failure, or missing paired seed gets no replacement. Incomplete pairs are `INCONCLUSIVE`.

The primary score is corpus-summed target-blind prior NLL in nats divided by counted targets; PPL is `exp(NLL/token)` and secondary. Never average batch means. Every weighted-SMC arm retains the H6 recursion, 64 common counter-based streams rooted at `2026072198`, particle ladder `(128,256,512,1024)`, and `Q2` extrapolation/error rules. Exact autoregressive arms are paired with the same stream identities without inventing stochastic variation. The required exact and weighted record counts are derived solely from each `WT103ArmSpec.scorer_kind`, the confirmatory-seed inventory, the stream inventory, and the particle inventory. Missing, duplicate, nonfinite, failed, or separately entered records make the associated gate `INCONCLUSIVE`.

For each weighted-SMC checkpoint and each of the 64 test streams, compute `Q0=2Y256-Y128`, `Q1=2Y512-Y256`, `Q2=2Y1024-Y512`, `R1=Q1-Q0`, and `R2=Q2-Q1`. With `df=63`, use the frozen simultaneous constant `4.5144904535377144`, `h(X)=4.5144904535377144*s(X)/sqrt(64)`, `U1=abs(mean(R1))+h(R1)`, `U2=abs(mean(R2))+h(R2)`, conditional remainder bound `B=U2/(1-0.75)=4U2`, and random half-width `H=h(Q2)`. Eligibility requires `U2<=0.75*U1`, `B<=delta/40=0.00025125839633753625`, and `H<=delta/20=0.0005025167926750725`. For the PRIMARY pair only, use common-stream `D_i[r]=NLL_A0,i-Q2_parent-specific-complete,i[r]`, `H_i=h(D_i)`, and `e_i=H_i+B_parent-specific-complete,i`; require `H_i<=delta/20` and `e_i<=delta/10=0.001005033585350145`.

Propagate each `e_i` into `d_i=NLL_A0,i-mean(Q2_parent-specific-complete,i)` by enumerating all `2^8=256` error-box corners and using the frozen `t_(0.975,7)=2.364624251592784`. The practical threshold remains `delta=-log(0.99)=0.01005033585350145`: inflated lower bound `>delta` is PASS, inflated upper bound `<=0` is FAIL, otherwise INCONCLUSIVE. These bounds retain H6's independent-stream, finite-variance, approximate-Gaussian replicate-mean, independent-training-seed, and geometric-remainder assumptions; a failed contraction or assumption is INCONCLUSIVE. This paired decision applies only to PRIMARY A0 versus parent-specific complete. Per-arm PPL, uninflated intervals, estimator diagnostics, controls, and seed variability remain separately reported regardless of decision.

The test split is opened once after every terminal checkpoint derived by the immutable `EndpointInventory`, all tuning selections, metric/analysis/figure source hashes, estimator streams, source/token/window manifests, and run-group manifest are frozen. The one transaction attempts every required endpoint and is never reopened for a fix, extra seed, prettier figure, or missing metric. Every endpoint record completed before a crash or failure is durably retained with its disposition, but a partial endpoint set is never aggregated, promoted, or reported as a scientific comparison; any missing, duplicate, or nonfinite required endpoint makes the terminal result INCONCLUSIVE.

---

## Revision-Bound Training Sparsity Certificate

`TrainingSparsityCertificate` is a same-`(git_head,dirty_digest,profile_sha256,factory_set_sha256,endpoint_inventory_sha256)` structural certificate consumed by post-H8 readiness. It is not H8, not a memory-capacity smoke, and not an asymptotic complexity claim. PASS means only that every traced operation in the exact immutable arm-inventory training/evaluation/checkpoint vocabulary obeyed this finite shape/storage contract.

For the literal profile let `B=128`, `L=128`, `b=40`, `W=20`, `V=50_257`, `D=L*b=5_120`, train decoder chunk `C_train=512`, evaluation decoder chunk `C_eval=256`, and SMC particle chunk `C_particle=32`.

### Permitted dense axes and shapes

- Vocabulary density is explicitly permitted for embedding/decoder parameters `[V,h]` or `[V,b]`, decoder bias `[V]`, their gradients, and same-shaped AdamW moments. It is recorded separately and never called population sparse.
- Decoder logits/logit gradients are permitted only as `[C,V]` with `C<=512` during training and `C<=256` during evaluation. A materialized `[B*L,V]` tensor is forbidden even though its dense axis is vocabulary.
- A0 Q/K/V inputs and outputs `[B,2,L,h/2]`, packed QKV `[B,L,3h]`, and the output `[B,L,h]` are permitted. The single exact forced Flash SDPA operator may declare the semantic full-causal pair domain `L*(L+1)/2`, but it returns no weights and is the only exception to pair-domain language below; this exception permits no tensor or allocator storage with both query and key axes.
- Token IDs/masks `[B,L]`, latent means `[B,L,b]`, block-diagonal precision/factors/selected blocks `[B,L,b,b]`, single lower adjacent blocks `[B,L-1,b,b]`, and banded source indices/logits/probabilities `[B,L,W]` are permitted.
- Primary frames/local maps `[L,K,K]`, local workspaces no larger than `[B,b,b]`, width-`<=b` block solve RHS, bounded `[C_particle,...]` SMC state, scalar/row diagnostics, and checkpoint tensors identical to classified parameters/optimizer/scientific state are permitted.
- Aliases/views are counted once by unique storage span. An upper adjacent block may be a transpose view of the stored lower block; duplicate persistent upper storage is forbidden.

### Forbidden population-dense/equivalent storage

Except for the nonmaterialized semantic pair count of the exact A0 Flash operator, any request, result, retained/transient/aggregate storage, serialized tensor, or equivalent slab with pair-axis shape `[L,L]`, `[B,L,L]`, `[B,2,L,L]`, `[L,L,b,b]`, `[B,L,L,b,b]`, `[D,D]`, `[B,D,D]`, flat pair-axis equivalent, a full population covariance/precision/moment/Cholesky/identity/selector, full-width solve RHS, all block pairs, full causal source matrix, or decoder logits `[B*L,V]` is forbidden. Flattening, tiling retained across autograd, stacking, concatenating, sparse-to-dense conversion, custom CUDA allocation, NumPy allocation, or checkpoint serialization cannot evade classification.

### Required trace and reconciliation

In clean child processes, trace the exact scientific profile through every distinct arm path derived from `EndpointInventory`: A0 data transfer/forward/CE/backward/AdamW/evaluation/checkpoint; parent-specific complete and fixed-complete data transfer/forward, E-like proposal, immutable snapshot, complete-ELBO evaluation, model backward, both accepted optimizer/scheduler updates, weighted prior evaluation at each particle chunk, metric/failure writes, and checkpoint serialization; the parent-specific emission-only objective-gate path with its exact objective scope; and the no-latent control path without fabricating recognition or latent work. Use four complementary records: PyTorch dispatch pre-request/result shape and stack/concat tracing, PyTorch profiler operator/memory events, CUDA allocator snapshot plus unique-storage lifetime registry, and backend/checkpoint tensor inventories.

For every persistent or logical tensor, recompute `numel*element_size` from the frozen formula and require exact equality with its classified storage-byte record. Reconcile unique logical bytes to CUDA allocated bytes by recording allocator overhead/fragmentation separately; require `allocated_bytes>=classified_unique_storage_bytes` and zero unclassified live storage/events. Profiler and dispatch inventories must agree on every population- or vocabulary-shaped operation. Capacity percentages are not part of this certificate.

Assigned negative controls are executed through pre-allocation guards with reduced safe payloads but production logical-shape metadata: dense population `[D,D]` and `[B,D,D]` (population-shape guard plus dispatch); full source `[B,L,L]` (source-band guard); pair slab `[B,L,L,b,b]` (stack/concat guard); A0 math-SDPA fallback, explicit causal mask, and requested attention weights `[B,2,L,L]` (backend/dispatch/allocator guards); full decoder `[B*L,V]` (decoder-chunk guard); full selector/RHS `[D,D]` (factor backend counter); and an unclassified checkpoint tensor (serializer inventory). Every assigned detector must fire before the forbidden production allocation/serialization. A missed control, backend fallback, forbidden attempt, duplicate upper storage, or unclassified event is FAIL; missing profiler/allocator/serializer/backend observability is INCONCLUSIVE.

The certificate publishes the exact whitelist/forbidden schemas, formula table, per-path event/storage inventories, reconciled bytes, negative-control records, environment/durability identity, revision/profile/factory hashes, status, obligations, and explicit nonclaims. Readiness requires this certificate PASS **and** the independent 85%-capacity/resource forecast PASS.

---

## Metrics, Checkpoints, Runs, and Figures

### Checkpoint contract

Before any attempt, the run manager publishes one immutable `experiment-plan.json` containing the expected arms/seeds/attempts, literal profile, source/token/window/permutation/schedule identities, predecessor refs, resource ceilings, and schema hashes. Every checkpoint binds `experiment_plan_sha256`. A terminal `run-manifest.json` later references checkpoint identities; a checkpoint never embeds or depends on that not-yet-existent terminal manifest, and checkpoint I/O never publishes a parent/run manifest.

Every checkpoint contains:

- `checkpoint_role` exactly `resume_only` at validation boundaries or `terminal_scoring` after the complete planned passes. Load may use either role for compatible restoration, but evaluation, endpoint aggregation, and test-opening validators accept only `terminal_scoring`;
- model and active recognition state; active optimizer and scheduler state per parameter block; AMP scaler only if the exact precision profile uses one;
- Python, NumPy, PyTorch CPU, and every CUDA-device RNG state; estimator counter streams; train permutation bytes/hash and `DataCursor`;
- resolved config, objective/update/snapshot schema hashes, arm/factory/intervention identity, H5/H6/H7/H8 evidence references, data/tokenizer/window manifests, source lock, and environment identity;
- global step/pass, successful and rejected update counts, cumulative counted targets, metric next ordinal/hash-chain head plus scientific numerator/denominator projection, failure-ledger head, checkpoint parent identity, experiment-plan hash, and resume-lineage head.

Two identities are mandatory and never conflated:

- `scientific_state_sha256` canonically hashes exact tensor dtype/shape/bytes for model, recognition, optimizer, scheduler, AMP state, every RNG/counter stream, cursor/permutation, accepted/rejected update trace, metric numerators/denominators and next ordinal, plus the next-batch prediction fixture. It excludes path names, serialization layout, UTC/monotonic timestamps, elapsed durations, host process IDs, write order, and terminal artifact hashes.
- `checkpoint_payload_sha256` hashes the exact serialized checkpoint bytes. `checkpoint_manifest_body_sha256` hashes the canonical operational manifest body with all digest/self-identity fields excluded. `artifact_sha256` is `SHA256("vfe4-checkpoint-artifact-v1\0" || checkpoint_payload_sha256_bytes || checkpoint_manifest_body_sha256_bytes)`. The published checkpoint manifest carries those three values, but none is defined over a byte string containing itself; its exact file-byte identity is recorded externally by the downstream run manifest. Two faithful resumes may have distinct payload/body/artifact hashes and timestamps while the scientific state hash and next predictions are equal.

Resume appends a distinct immutable `resume-lineage.jsonl` event with parent artifact/scientific identities, new process/environment identity, cursor, reason, and timestamps. Operational events never rewrite scientific state. Exact smoke equivalence requires equal scientific hashes; elementwise equality of all scientific tensors/primitives; equal update/metric numerator-denominator state; and bitwise-equal next two batches of prior predictions under the same estimator streams. It does not require equal wall-clock durations, UTC values, terminal run-manifest hash, filesystem path, or serialized artifact hash.

Executable pickle is prohibited. Before deserialization, require a regular nonlink file, exact manifest-declared byte size and SHA-256, and `size_bytes<=checkpoint_max_bytes` from the immutable experiment plan. Load only with `torch.load(..., map_location="cpu", weights_only=True)`; adding safe globals or retrying with `weights_only=False` is forbidden. Recursively accept only exact `dict/list/tuple/str/int/float/bool/None/torch.Tensor` values, declared tensor dtypes/shapes/numel, CPU storage before explicit restore, and no custom class/global/reducer. A malicious reducer fixture must not execute or create its sentinel. A nonpickle tensor container may replace this schema only through an explicit schema revision.

Writes use the probed `DurabilityBackend`: same-volume unique sibling staging, durable file flush, pre-publication read-back/schema/hash validation, durable replacement, and reopen validation. Loading is fail-closed before mutating a live object. Any schema, config, objective, model shape, arm, optimizer, scheduler, precision, dependency, tokenizer/data/window, predecessor, experiment-plan, RNG, cursor, tensor-inventory, or scientific-state mismatch blocks resume. The default migration set is empty. An explicit future migration profile must name source/destination schema hashes, transform code hash, information-loss declaration, independent test, and a new run identity; no profile may load V3.

### Metric and failure records

The authoritative live stream is `metrics.jsonl`, one canonical JSON object per line with `schema_version`, ordinal, UTC/monotonic timestamps, run/arm/seed/phase/split/step/pass identities, `previous_record_sha256`, and `record_sha256`. Each append flushes and fsyncs. Resume validates the hash chain and may truncate only one proven incomplete final byte fragment; a malformed complete line or chain mismatch fails closed. `metrics.csv` is a deterministic atomic export from validated JSONL, with a fixed column order/schema and decimal round-trip strings.

Every recorded mean/rate carries its raw numerator and denominator. Required fields include:

- per-step/pass train complete ELBO and every `ElboTerms` partition: expected emission log likelihood, initial state/model, state/model source KL, state/model transitions, joint recognition entropy, estimator error, and complete scalar;
- held-out prior-predictive summed NLL, counted targets, NLL/token, PPL, estimator stream/particle level, and cache audit; separate emission-only diagnostic with `is_elbo=false`;
- state/model/source KL numerators, categorical source `entropy_sum`, `source_row_count`, support size, and effective source count defined exactly as `exp(entropy_sum/source_row_count)`; a zero row count is explicit `not_applicable`, never zero effective sources;
- accepted/rejected proposals by exact label/block/reason, before/after complete objective numerators/denominators, error allowance, snapshot hash, damping/projection/rollback facts, learning rate, scheduler ordinal/state, AMP scale/overflow/applicability, clipping threshold/pre/post norm/clipped flag, and every effective optimizer parameter (`betas`, `eps`, `weight_decay`, `amsgrad`, `foreach`, `fused`) for that update;
- minimum local Cholesky pivot, failed pivots, jitter/damping, SPD projections, local condition endpoints, sparse condition estimate, solve residual numerator/allowance, and nonfinite counts;
- gradient L2/inf norms and clipped/unclipped counts per active block; expected/observed autograd scope;
- examples, windows, batches, counted targets, tokens/second numerator and elapsed denominator, data-wait/forward/inference/backward/update/evaluation/checkpoint durations, and total wall clock;
- process RSS/HWM, CUDA allocated/reserved/peak allocated/peak reserved, allocation retry/OOM records, device/dtype, and resource-preflight identity;
- every warning/failure with exception type/message hash, phase/cursor/checkpoint, retry classification, state-advanced proof, and terminal disposition.

`failures.jsonl` is a separate hash-chained fsynced ledger so a training exception cannot disappear when metrics finalization fails.

### Run directory, provenance, and resource safeguards

The run manager first publishes immutable `experiment-plan.json`; only then is an attempt created under `runs/.inprogress/<run_id>` by the probed `DurabilityBackend` exclusive-create primitive. A terminal successful or failed attempt is fully integrity-validated and durably renamed to `runs/<run_id>`; crashes retain the in-progress directory for explicit resume. No existing path is overwritten. Checkpoint I/O returns a typed `CheckpointIdentity` and never edits a run manifest. The run manager alone publishes the terminal manifest referencing checkpoint/metric/failure identities and the atomic `experiment-index.json`; newest-directory or glob selection is forbidden.

Each run records Git HEAD, staged/unstaged/untracked content digest, canonical dependency lock and installed distributions, Python/PyTorch/CUDA/cuDNN/driver versions, OS/CPU/RAM/GPU names and capacities, device capability, BLAS/thread settings, determinism flags, locale/time zone, start/end UTC and monotonic duration, exact source/token/window/config/objective/factory/evidence hashes, parent checkpoint, and failure state. Dirty runs are labeled dirty and cannot be promoted as clean evidence; the scientific candidate requires a clean tracked source.

Before official acquisition, preprocessing, training, or evaluation, compute a byte forecast for archive staging, extracted members, int32 token caches, schedule files, all retained checkpoints, JSONL/CSV, test records, figures, and 25% temporary-write overhead. Require available bytes `>=2*forecast+10 GiB`; do not auto-delete another run or V3 cache. Tokenization runs one split at a time in an isolated process and requires available RAM/swap above its measured smoke-derived multiplier; otherwise fail before allocation.

Before real training, run a synthetic shape-identical allocation preflight through every distinct path in the immutable arm inventory: data transfer, forward, applicable recognition proposal, immutable snapshot, model backward/update, validation scorer, metric record, and checkpoint serialization. Require peak device allocated and reserved memory each `<=85%` of physical capacity and enough host/disk headroom for one atomic checkpoint duplicate. H8 values do not enter this decision, and this capacity check does not replace `TrainingSparsityCertificate`. OOM or over-budget fails; do not silently reduce batch size, sequence length, particles, sources, dimensions, precision, or checkpoint contents.

### Throughput, wall-time, GPU-hour, and energy authorization

After Task 13 knows exact corpus/window counts but before authorization, measure each distinct training and scorer path declared by the immutable arm inventory on generated shape-identical batches: five untimed warmups plus 20 timed updates per distinct train path, 10 timed full validation-window calls per scorer kind, 10 test-window calls at each applicable particle count, one durable checkpoint write per distinct checkpoint schema, and one complete tiny figure-set render. Sample GPU board power every `100 ms` through a provenance-bound NVML or `nvidia-smi` provider; provider absence/permission failure makes energy readiness INCONCLUSIVE.

For each component use the minimum observed post-warmup throughput and maximum observed duration/power, not a mean, then multiply predicted time by `forecast_headroom_factor=1.25`. Publish a component table with exact work units and formulas for:

- the `EndpointInventory`-derived tuning attempts at one quarter train pass, plus one applicable full-validation assessment per attempt;
- the inventory-derived confirmation attempts at two full train passes, 40 full-validation boundaries per attempt, and checkpoint/metric costs at those boundaries;
- the inventory-derived exact and weighted-SMC test corpus records;
- source/token/window preparation, final table aggregation, every frozen required-figure-registry entry with its inventory-derived panels/series, and review/export overhead (GPU hours zero where CPU-only).

Sum predicted GPU seconds/device-hours and wall time without overlapping components unless the executable schedule explicitly overlaps them. Conservative energy is `forecast_gpu_hours * max(measured_max_board_power_watts, reported_power_limit_watts)/1000`. The immutable profile ceilings are `max_gpu_hours=720`, `max_wall_hours=840`, and `max_energy_kwh=500`; authorization requires each `1.25*raw_forecast` to remain at or below its ceiling. Exceeding a ceiling stops for an explicit preregistration/user-authorization revision; no automatic batch/particle/seed/schedule reduction is permitted.

Every attempt records actual device-seconds, wall seconds, and sampled energy in an append-only `ResourceUsageLedger` bound to, but never mutating, the immutable experiment plan; remaining allowances are derived from plan ceilings minus validated ledger totals. Immediately before the irreversible test reservation, recompute the test-only forecast from actual validation throughput and require remaining GPU-hour/wall/energy ceilings and disk capacity each to exceed `1.25*test_transaction_forecast`. Insufficient headroom blocks before the durability backend's exclusive reservation; it never opens a partial test to discover cost.

### Figure contract

`generate_vfe4_figures.py` takes one explicit immutable run-group manifest path in its editable `CONFIG`. It never selects newest, imports `train_vfe4`, imports training/model/checkpoint modules, initializes CUDA, opens data, or mutates a run. The sole semantic inputs are terminal-manifest-validated, finalized `metrics.jsonl` files plus frozen final result-table JSON. `metrics.csv` is never a semantic input: regenerate it from validated JSONL, require byte equality with the published CSV, and fail on drift. A figure never recomputes a metric from a checkpoint or partial run. Missing required finalized records make that figure fail with an obligation.

`FigureSpec` freezes schema/version/hash, ordered panels/series, source columns, aggregation, uncertainty interval, axis scale/limits, units, colors/markers/fonts, captions, and alt text. Rendering uses `Agg`, pinned Matplotlib, DejaVu fonts, fixed `svg.hashsalt`, fixed metadata with no current timestamp, explicit sorted ordering, and no random layout. Each figure writes SVG, PNG, and PDF when the installed backend passes a startup format probe; otherwise the whole required-format preflight fails before rendering. Every figure also writes the exact plotted CSV and JSON sidecars, caption Markdown, alt text, and an output manifest with input/spec/environment/output hashes.

Every metric/result row has `applicability="applicable"|"not_applicable"` plus a reason derived from its exact arm spec. Complete-ELBO panels apply only to complete-objective latent rows; recognition, source-entropy/effective-source, and SPD-population panels apply only when the corresponding arm fields are active; CE applies only to cross-entropy rows; prior-NLL/PPL, model-update, throughput, and ordinary parameter-gradient panels apply wherever their frozen scorer/model path produces them. Every other field is explicit `not_applicable`, never zero, silently null-plotted, or fabricated. Shared plots omit inapplicable series and state the reason in caption/alt text.

The minimum required figure set is:

1. role-labeled training-objective curves for every applicable inventory arm plus every arm's validation prior-NLL/PPL curves, with all seed traces and preregistered across-seed uncertainty;
2. terminal prior-NLL and PPL for all ordered inventory arms, with the estimator-aware paired interval applied only to the PRIMARY A0-versus-parent-specific-complete contrast and the other rows visibly labeled as gates or controls;
3. complete-ELBO decompositions for the two complete-objective latent arms, plus the parent-specific emission-only non-ELBO objective-gate diagnostic; A0 and every nonapplicable control field are explicitly labeled not applicable;
4. state/model source entropy and `exp(entropy_sum/source_row_count)` effective source count for every applicable latent arm, with explicit not-applicable reasons for the other arms;
5. accepted/rejected update counts and reasons by label/block;
6. SPD, pivot, condition-estimate, solve-residual, damping, and projection health for every applicable latent arm, with explicit not-applicable reasons for the other arms;
7. throughput, phase wall time, host memory, and device allocated/reserved memory;
8. seed variability for terminal NLL/PPL, complete ELBO, acceptance rate, and peak memory.

Curve x-coordinates are the shared counted-training-target boundaries, never wall-clock interpolation. At each common boundary, show every confirmatory-seed trace and the pointwise arithmetic mean with the descriptive two-sided `mean +/- 2.364624251592784*sample_sd/sqrt(8)` band; do not smooth, resample, or treat the pointwise band as a simultaneous hypothesis test. Terminal arm plots use the estimator-inflated interval above, not the pointwise curve band.

Figure generation publishes into a new content-addressed `figures/<figure_set_sha256>/` directory and atomically updates only the experiment's figure index after validating every required output/sidecar. Training finalization may invoke the same pure renderer by package API, but a rendering failure records a failure and never changes the training result; one click on `generate_vfe4_figures.py` regenerates the identical set later from the immutable run manifest.

---

## File and Interface Map

Paths already present in the H1–H8/H6 implementation are modify-in-place
surfaces, not replacement modules. Post-H8 work must preserve their existing
public contracts and tests. Production WT103 model code is isolated in
`vfe4/training/wt103_models.py`; the H6 byte-vocabulary CPU/float64 models are
predecessor evidence and are not repurposed as the WT103 architecture.

| Path | Responsibility |
|---|---|
| `vfe4/types/training.py` | Frozen `WT103ArmSpec`, `WT103GateSpec`, `EndpointInventory`, shared profile, run, update, data cursor, sparsity, metric, checkpoint, evaluation, and experiment records. |
| `vfe4/types/figures.py` | Frozen figure input/spec/output/manifest records. |
| `vfe4/config/schema.py` | Extend frozen training/data/evaluation/checkpoint/recording/artifact/figure sections and closed literal values. |
| `vfe4/config/resolve.py` | Strict recursive unknown-key rejection, conditional training/figure resolution, derived fields, canonical hashes. |
| `vfe4/config/training.py` | Post-H8 default `TrainingConfig` construction and cross-section invariants; no launcher import. |
| `vfe4/data/wikitext103.py` | Exact URL acquisition, ZIP validation, bounded extraction, source/license manifest, offline reuse. |
| `vfe4/data/tokenizer.py` | Pinned GPT-2/tiktoken spec, split encoding, round trip, int32 cache publication. |
| `vfe4/data/windows.py` | Exactly-once shifted windows, masks, counts, permutations, batches, and cursor replay. |
| `vfe4/data/access.py` | Sealed split references and train/test capability boundary. |
| `vfe4/training/wt103_models.py` | Production WikiText-103 A0 and structured-arm modules; Flash-only A0 attention and WT103 dimensions remain isolated from the existing H6 CPU/float64 models. |
| `vfe4/training/factories.py` | Explicit factories for all five immutable `WT103ArmSpec` rows, including model/recognition/predictor/optimizer/scheduler construction and match reports. |
| `vfe4/training/formulas.py` | Frozen A0 architecture, exact parameter formula, analytical semantic FLOP operator ledger, and canonical hashes. |
| `vfe4/training/sparsity.py` | Revision-bound inventory-wide shape/allocation traces, formulas, classifications, negative controls, and certificate. |
| `vfe4/training/readiness.py` | Consume typed integrity records for H5/H6/H7/H8, predictor safety, training sparsity, allocation/data/resource readiness. |
| `vfe4/training/engine.py` | Typed phase schedule, complete ELBO updates, immutable snapshots, acceptance/rollback, loop. |
| `vfe4/checkpoint/schema.py` | Closed checkpoint schema and compatibility report. |
| `vfe4/checkpoint/io.py` | Safe bounded `weights_only` save/read-back/load, scientific/artifact identities, exact resume; returns identities and never publishes run manifests. |
| `vfe4/checkpoint/serialization.py` | Tensor/primitive whitelist, size/hash-before-read, shape inventory, malicious-reducer rejection. |
| `vfe4/checkpoint/migrations.py` | Empty-by-default explicit migration registry; permanent V3 rejection. |
| `vfe4/recording/metrics.py` | Canonical hash-chained JSONL records and validation. |
| `vfe4/recording/tables.py` | Stable deterministic CSV/result-table export. |
| `vfe4/recording/failures.py` | Independent append-only failure ledger and retry classification. |
| `vfe4/evaluation/prior_nll.py` | Target-blind corpus sums, cache audit, and exact-autoregressive versus weighted-SMC dispatch solely from `WT103ArmSpec.scorer_kind`. |
| `vfe4/evaluation/statistics.py` | Q2/error propagation, paired seed intervals, 256 corners, PASS/FAIL/INCONCLUSIVE. |
| `vfe4/evaluation/test_opening.py` | Durable exclusive one-opening reservation and test capability. |
| `vfe4/artifacts/run_directory.py` | Exclusive in-progress reservation, atomic terminal publication, run-group/index lifecycle. |
| `vfe4/artifacts/manifest.py` | Upstream generic closed-schema integrity reader/hasher returning typed records; no readiness/run-domain imports. |
| `vfe4/artifacts/durability.py` | Upstream POSIX/Windows durability backends, volume/startup probes, exclusive create, durable replace. |
| `vfe4/artifacts/environment.py` | Hardware/dependency/git/resource provenance, throughput/power forecasts, authorization ceilings. |
| `vfe4/artifacts/atomic.py` | Upstream generic canonical bytes and durability-backed atomic primitives; no domain/run imports. |
| `vfe4/artifacts/provenance.py` | Extend existing provenance with post-H8/data/training/evaluation/figure identities. |
| `vfe4/figures/spec.py` | Stable required figure registry as explicit functions/records, not dynamic signature dispatch. |
| `vfe4/figures/load.py` | Manifest/metrics/sidecar validation and immutable table construction. |
| `vfe4/figures/plots.py` | Eight required deterministic plotting functions. |
| `vfe4/figures/render.py` | Isolated no-training renderer, formats, sidecars, captions/alt text, atomic figure set. |
| `train_vfe4.py` | One editable click-to-run training `CONFIG`; import-safe orchestration only. |
| `generate_vfe4_figures.py` | One editable click-to-run figure `CONFIG`; import-safe orchestration only. |
| `docs/preregistrations/2026-07-21-post-h8-wikitext103-training.md` | Frozen source protocol, tokenizer, arms, schedules, statistics, opening, evidence, nonclaims. |
| `docs/data/wikitext103-raw-v1-source-record.json` | Machine-produced exact official archive/member/source/license identities, committed before training. |
| `docs/data/wikitext103-raw-v1-source-record.md` | Human-readable interpretation of the exact JSON record and offline rebuild instructions. |
| `tests/unit/test_wikitext103_source.py` | ZIP/URL/path/CRC/size/offline/V3-rejection tests with tiny archives only. |
| `tests/unit/test_wikitext103_tokenizer.py` | Pinned spec/golden/round-trip/split-isolation/token-cache tests. |
| `tests/unit/test_training_windows.py` | Shift/mask/count/permutation/batch/cursor tests. |
| `tests/unit/test_training_config.py` | Recursive strict config, import safety, derived hashes, incompatible combinations. |
| `tests/unit/test_training_factories.py` | Exact five-arm semantics, active parameters/phases, objective/prior interventions, H5 labels, capacity/FLOP matching, and result roles. |
| `tests/promotion/test_training_sparsity.py` | Exact inventory-wide path coverage, formula/byte reconciliation, forbidden shapes, assigned controls, and certificate status. |
| `tests/unit/test_training_engine.py` | Autograd scopes, snapshot nonaliasing, acceptance/rollback, complete objective. |
| `tests/unit/test_training_checkpoint.py` | Atomic schema, exact resume, corruption/mismatch/migration/V3 rejection. |
| `tests/unit/test_durability_backend.py` | POSIX/Windows probe, same-volume, flush/replace/reopen, failure injection, backend identity. |
| `tests/unit/test_training_metrics.py` | JSONL chain, numerator/denominator, stable CSV, incomplete-tail/failure behavior. |
| `tests/unit/test_training_artifacts.py` | Run reservation/publication/index/manifest/environment/resource safeguards. |
| `tests/unit/test_training_evaluation.py` | Target blindness, corpus sums, one opening, estimator records, paired decisions. |
| `tests/unit/test_training_figures.py` | No-training imports, stable specs, required plots/formats/sidecars/captions/alt text. |
| `tests/property/test_training_prefix_safety.py` | GPT-2 vocabulary target/suffix/cache invariance under exact final predictor identity. |
| `tests/integration/test_train_vfe4.py` | Import the live dictionary and run one tiny generated inventory-wide train/resume/evaluate flow without external data. |
| `tests/integration/test_generate_vfe4_figures.py` | Regenerate every frozen required-figure-registry entry from immutable tiny run metrics only. |
| `tests/promotion/test_post_h8_training_readiness.py` | Exact H8/prerequisite/data/predictor/resource readiness and stale-reference blocking. |

Dependency direction is `config + types -> artifacts.atomic + artifacts.durability + artifacts.manifest (generic integrity only) -> data/numerics -> generative/recognition/objective/predictive -> training/checkpoint/evaluation/recording -> artifacts.run_directory/environment/provenance -> figures -> launchers`. Generic integrity modules import no readiness, checkpoint, run-manager, figure, or domain module. Readiness consumes typed `ArtifactIntegrityRecord` values rather than hashing arbitrary paths. Checkpoint I/O consumes integrity/durability primitives and returns `CheckpointIdentity`/`LoadedCheckpoint`; only the downstream run manager publishes plan/terminal manifests. `figures` may import `types`, finalized `recording` readers, and generic artifact-integrity readers only. It cannot import `training`, `checkpoint`, `data`, `generative`, `recognition`, `objective`, `predictive`, run mutation, or either launcher. Production never imports `verification/`, `tests/`, or V3.

## Public Interfaces Frozen by This Plan

```python
@dataclass(frozen=True)
class WikiText103SourceRecord:
    schema_version: Literal["wikitext103-source-v1"]
    archive_request_url: str
    archive_final_url: str
    archive_redirect_chain: tuple[RedirectHop, ...]
    source_page_request_url: str
    source_page_final_url: str
    source_page_redirect_chain: tuple[RedirectHop, ...]
    archive_size_bytes: int
    archive_sha256: str
    archive_content_type: str | None
    central_directory_sha256: str
    members: tuple[ArchiveMemberIdentity, ArchiveMemberIdentity, ArchiveMemberIdentity]
    source_page_size_bytes: int
    source_page_content_type: str
    source_page_sha256: str
    license_paragraph_start_byte: int
    license_paragraph_end_byte: int
    license_raw_slice_sha256: str
    license_declaration: str
    license_hrefs: tuple[str, ...]
    installed_distribution_sha256: str
    tokenizer_tables_sha256: str
    validator_sha256: str
    record_sha256: str

@dataclass(frozen=True)
class WT103ArmSpec:
    schema_version: Literal["wt103-arm-spec-v1"]
    arm_id: Literal[
        "WT103-A0-AR-v1",
        "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
        "WT103-A5-FIXED-COMPLETE-v1",
        "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
        "WT103-A5-NOLATENT-v1",
    ]
    factory_id: str
    training_objective: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ]
    prior_variant: Literal[
        "absent", "fixed", "parent_specific_pooled_prefix"
    ]
    source_mixture: Literal["absent", "exact"]
    latent_enabled: bool
    recognition_enabled: bool
    recognition_family: Literal[
        "absent", "structured_block_tridiagonal_smoothing"
    ]
    recognition_iterations_per_batch: Literal[0, 1]
    update_phases: tuple[str, ...]
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    tuning_grid_id: Literal["wt103-six-cell-v1"]
    confirmatory_seed_ids: tuple[int, ...]
    terminal_checkpoint_role: Literal["terminal_scoring"]
    result_role: Literal[
        "PRIMARY_REFERENCE",
        "PRIMARY_ENDPOINT",
        "PRIOR_CONTROL",
        "OBJECTIVE_GATE",
        "LATENT_PATH_CONTROL",
    ]
    nonclaims: tuple[str, ...]
    arm_spec_sha256: str

@dataclass(frozen=True)
class WT103GateSpec:
    schema_version: Literal["wt103-gate-spec-v1"]
    gate_id: Literal[
        "SOURCE_LOCK",
        "H8_EXACT_REVISION",
        "POST_H8_READINESS",
        "OBJECTIVE",
        "PRIMARY",
        "PRIOR_CONTROL",
        "LATENT_PATH_CONTROL",
    ]
    ordinal: int
    prerequisite_gate_ids: tuple[str, ...]
    result_arm_ids: tuple[str, ...]
    disposition_rule_id: str
    gate_spec_sha256: str

@dataclass(frozen=True)
class EndpointInventory:
    schema_version: Literal["wt103-endpoint-inventory-v1"]
    arms: tuple[WT103ArmSpec, ...]
    gates: tuple[WT103GateSpec, ...]
    tuning_cells: tuple[tuple[float, float], ...]
    tuning_seed_ids: tuple[int, ...]
    confirmatory_seed_ids: tuple[int, ...]
    validation_stream_ids: tuple[int, ...]
    test_stream_ids: tuple[int, ...]
    particle_counts: tuple[int, ...]
    estimator_protocol_sha256: str
    tuning_attempt_keys: tuple[str, ...]
    terminal_checkpoint_keys: tuple[str, ...]
    validation_endpoint_keys: tuple[str, ...]
    test_endpoint_keys: tuple[str, ...]
    raw_score_record_keys: tuple[str, ...]
    result_row_keys: tuple[str, ...]
    figure_panel_keys: tuple[str, ...]
    figure_series_keys: tuple[str, ...]
    endpoint_inventory_sha256: str

    @classmethod
    def create(
        cls,
        arms: tuple[WT103ArmSpec, ...],
        gates: tuple[WT103GateSpec, ...],
        tuning_cells: tuple[tuple[float, float], ...],
        tuning_seeds: tuple[int, ...],
        confirmatory_seeds: tuple[int, ...],
        estimator_protocol: EstimatorProtocol,
    ) -> "EndpointInventory": ...

@dataclass(frozen=True)
class WT103ExperimentProfile:
    schema_version: Literal["wt103-experiment-profile-v1"]
    batch_size: Literal[128]
    sequence_length: Literal[128]
    d_z: Literal[20]
    d_m: Literal[20]
    K: Literal[20]
    source_lookback: Literal[20]
    model_depth: Literal[1]
    optimizer: AdamWProfile
    scheduler: SchedulerProfile
    precision: PrecisionProfile
    cadence: CadenceProfile
    profile_sha256: str

@dataclass(frozen=True)
class A0ArchitectureProfile:
    schema_version: Literal["wt103-a0-architecture-v1"]
    hidden_width: int
    attention_context: Literal["full_causal_inclusive_self"]
    attention_backend_policy: Literal["flash_attention_only_no_fallback"]
    pytorch_sdpa_api_binding: str
    pytorch_version: str
    sdpa_api_sha256: str
    flash_backend_sha256: str
    positional_encoding: Literal["learned_absolute"]
    normalization_placement: Literal["pre_norm_with_final_norm"]
    activation: Literal["gelu_tanh_approximation"]
    parameter_formula_schema: Literal["wt103-a0-parameter-formula-v1"]
    flop_formula_schema: Literal["wt103-a0-semantic-train-flops-v1"]
    formula_sha256: str
    architecture_sha256: str

@dataclass(frozen=True)
class CandidateTokenizerContract:
    distribution: Literal["tiktoken"]
    version: Literal["0.12.0"]
    encoding_name: Literal["gpt2"]

@dataclass(frozen=True)
class TokenizerSpec:
    schema_version: Literal["gpt2-tiktoken-v1"]
    distribution: Literal["tiktoken"]
    version: Literal["0.12.0"]
    encoding_name: Literal["gpt2"]
    vocabulary_size: Literal[50257]
    eot_token_id: Literal[50256]
    corpus_method: Literal["encode_ordinary"]
    distribution_record_sha256: str
    regex_pattern_sha256: str
    mergeable_ranks_sha256: str
    special_tokens_sha256: str
    golden_vectors_sha256: str
    spec_sha256: str

@dataclass(frozen=True)
class WindowManifest:
    split: Literal["train", "validation", "test"]
    token_payload_sha256: str
    sequence_length: Literal[128]
    stride: Literal[128]
    window_count: int
    counted_targets: int
    payload_sha256: str
    manifest_sha256: str

@dataclass(frozen=True)
class PermutationManifest:
    split: Literal["train"]
    pass_index: Literal[0, 1]
    data_order_seed: Literal[2026072199]
    bit_generator: Literal["PCG64"]
    numpy_version: str
    window_manifest_sha256: str
    payload_sha256: str
    manifest_sha256: str

@dataclass(frozen=True)
class DataCursor:
    split: Literal["train", "validation", "test"]
    pass_index: int
    permutation_sha256: str
    next_batch_ordinal: int
    next_window_ids: tuple[int, ...]
    counted_targets: int
    cursor_sha256: str

@dataclass(frozen=True)
class TrainingSparsityCertificate:
    schema_version: Literal["wt103-training-sparsity-v1"]
    git_head: str
    dirty_digest: str
    profile_sha256: str
    factory_set_sha256: str
    whitelist_sha256: str
    forbidden_shape_sha256: str
    trace_set_sha256: str
    formula_reconciliation_sha256: str
    negative_controls_sha256: str
    status: GateStatus
    obligations: tuple[str, ...]
    certificate_sha256: str

class DurabilityBackend(Protocol):
    def probe(self, root: Path) -> DurabilityIdentity: ...
    def create_exclusive(self, path: Path, payload: bytes) -> DurableFileIdentity: ...
    def replace_durable(self, path: Path, payload: bytes) -> DurableFileIdentity: ...

class PriorPredictor(Protocol):
    def next_token_log_probs(
        self,
        prefix_tokens: torch.Tensor,
        estimator_rng: EstimatorRng,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction: ...

def resolve_training_config(raw: Mapping[str, object]) -> TrainingConfig: ...
def resolve_figure_config(raw: Mapping[str, object]) -> FigureConfig: ...
def validate_endpoint_inventory(
    inventory: EndpointInventory,
    *,
    expected_sha256: str,
) -> None: ...
def acquire_wikitext103(config: AcquisitionConfig) -> SealedDatasetRef: ...
def materialize_train_data(
    sealed: SealedDatasetRef,
    readiness: PostH8ReadinessToken,
) -> TrainDataCapability: ...
def reserve_test_opening(plan: TestOpeningPlan) -> DurableTestOpeningCapability: ...
def build_training_arm(arm: ArmId, config: TrainingConfig) -> TrainingArm: ...
def train_step(arm: TrainingArm, batch: CausalBatch, state: TrainState) -> StepResult: ...
def save_checkpoint(path: Path, bundle: CheckpointBundle) -> CheckpointIdentity: ...
def load_checkpoint(path: Path, expected: ResumeContract) -> LoadedCheckpoint: ...
def score_prior_nll(
    predictor: PriorPredictor,
    batches: Iterable[CausalBatch],
    stream: EstimatorRng,
) -> NllTotals: ...
def append_metric(path: Path, record: MetricRecord) -> MetricIdentity: ...
def export_metrics_csv(jsonl: Path, csv_path: Path) -> TableIdentity: ...
def publish_experiment_plan(plan: ExperimentPlan) -> ExperimentPlanIdentity: ...
def finalize_run(run: ReservedRun, checkpoints: tuple[CheckpointIdentity, ...]) -> RunManifestIdentity: ...
def render_figure_set(config: FigureConfig) -> FigureSetManifest: ...
```

`EndpointInventory.create` validates the exact ordered five arm rows and seven
gate rows, derives every key tuple from scorer/checkpoint applicability, and
hashes the canonical payload. Attempt, checkpoint, endpoint, raw-record,
result-row, figure-panel, and figure-series counts are read-only `len(...)`
properties over those derived tuples; they are not constructor or
configuration fields.
Resolvers reject any count override, extra derived key, missing key, duplicate
key, inapplicable particle key, reordered arm/gate, or payload/hash mismatch.
`OBJECTIVE` must precede and be a prerequisite of `PRIMARY`.

Before Task 13, `CandidateTokenizerContract` is the only production-scope tokenizer record; any `TokenizerSpec` created by Tasks 1--12 is explicitly fixture-scoped through an injected synthetic adapter. `CheckpointIdentity` contains checkpoint role plus distinct `scientific_state_sha256` and `artifact_sha256`; `LoadedCheckpoint` contains the validated bundle plus that identity. Neither API writes a run manifest.

---

### Task 1: Freeze Post-H8 Scope, Types, Configuration, and Preregistration

**Files:**
- Create: `vfe4/types/training.py`
- Create: `vfe4/types/figures.py`
- Create: `vfe4/config/training.py`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Create: `docs/preregistrations/2026-07-21-post-h8-wikitext103-training.md`
- Create: `tests/unit/test_training_config.py`

**Interfaces:** Produce every frozen record and both strict resolvers named above, including `WT103ArmSpec`, `WT103GateSpec`, `EndpointInventory.create`, and `validate_endpoint_inventory`. Consume existing H5/H6/H7/H8 result/reference types without widening their states.

- [ ] **Step 1: Write failing config/type tests.** Assert the shared-only `WT103ExperimentProfile`, exact ordered five `WT103ArmSpec` rows, seven ordered `WT103GateSpec` rows, derived `EndpointInventory`, and `A0ArchitectureProfile` field-by-field: batch/dimensions/depth/banded parent envelope; per-arm objective/prior/mixture/latent/recognition/update/scorer/role choices; OBJECTIVE-before-PRIMARY dependency; A0 full-causal `flash_attention_only_no_fallback` policy, source-lock-resolved PyTorch/API/backend identity fields, learned absolute positions, pre-norm/final norm, residual topology, tanh-GELU, projection/bias/tie choices, candidate-width rule and parameter/FLOP schema hashes; proposals/validity acceptance; AdamW betas/epsilon/flags/clipping; per-attempt warmup/cosine scheduler; dtype/device/autocast/determinism; decoder/particle chunks; resume-only versus terminal-scoring checkpoint roles; evaluation cadence; H6 seeds/grid/stopping/constants; one-opening policy; metric/figure/resource/sparsity schemas; and all nonclaims. Require every tuning/checkpoint/validation/test/raw-score/result-row/figure-series key and count to be derived from arm/gate/scorer applicability. With an injected synthetic API-binding adapter, require the exact `sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])` mapping and reject math/memory-efficient/cuDNN inclusion or fallback. Assert exact nested key sets, plain scalar types, frozen dataclasses, and canonical hash stability. Reject unknown keys at every depth, mutable lists where tuples are required after resolution, `bool` as `int`, derived hash/count/key overrides, reordered/duplicate/missing arms or gates, family-inferred scorer/objective behavior, backend fallback/materialized A0 attention, V3 paths, H6 byte vocabulary under WT103, absent H8/sparsity references, and claims of backprop-free or H8 training-memory transfer. Candidate URL/package facts remain unverified strings until source lock. Block `tiktoken` and live distribution/table discovery imports, inject only synthetic tokenizer adapters, and import both future launcher module names through stubs with zero side effects.
- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_training_config.py -q`. Expected: FAIL on missing types/resolvers.
- [ ] **Step 3: Implement frozen schemas and canonical resolution.** Keep launcher-friendly raw dictionaries at the boundary, convert once into immutable records, derive every key/count/hash through `EndpointInventory.create`, use explicit architecture/arm/gate/figure/checkpoint-role literals, and emit field-qualified errors. Keep objective, prior, latent/recognition, update, scorer, and result-role choices in `WT103ArmSpec`; the shared profile contains no singular A5 behavior. The pre-source-lock tokenizer config contains only the three candidate strings and an injected-adapter interface, never observed package/table facts. Preserve all existing H1--H8 config behavior.
- [ ] **Step 4: Write the complete preregistration.** Copy the exact source/archive/tokenizer/window/literal-profile/sparsity/update/tuning/stopping/statistical/opening/checkpoint/durability/artifact/metric/figure/resource/evidence/nonclaim rules from this plan. Mark request URLs, redirect chains, source/license facts, installed tokenizer facts, archive/member hashes, token hashes, and window/permutation hashes as candidate/unfrozen until the exact source-lock operation; they cannot be guessed or selected after outcomes.
- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: PASS.
- [ ] **Step 6: Review and commit.** Reviewer checks type/signature consistency, recursive rejection, immutable resolution, no CLI, and no transferred evidence. Commit `feat(training): freeze post-H8 WikiText-103 protocol`.

### Task 2: Establish Generic Integrity Readers and POSIX/Windows Durability

**Files:**
- Create: `vfe4/artifacts/durability.py`
- Create: `vfe4/artifacts/manifest.py`
- Modify: `vfe4/artifacts/atomic.py`
- Create: `tests/unit/test_durability_backend.py`

**Interfaces:** Produce `DurabilityBackend`, `PosixDurabilityBackend`, `WindowsDurabilityBackend`, `probe_durability`, generic `ArtifactIntegrityRecord`, `validate_closed_manifest`, and durability-backed canonical write primitives. These modules import no domain/run/readiness/checkpoint code.

- [ ] **Step 1: Write failing backend tests.** On the active OS plus injected fake syscall adapters for the other OS, assert same-volume staging, exclusive-create collision, exact file flush, replacement, reopen/hash validation, POSIX directory fsync, Windows `CREATE_NEW`/`FILE_FLAG_WRITE_THROUGH`/`FlushFileBuffers`/`MoveFileExW` flags, volume identity, and exact error capture. Inject failures before/after each syscall and prove the old target remains or the operation returns an explicit indeterminate obligation; never claim success from existence alone.
- [ ] **Step 2: Write failing integrity/dependency tests.** Require closed key sets, regular nonlink files, size-before-hash, canonical bytes, recursive manifests, and typed identities. Statically prove generic integrity modules do not import readiness, checkpoint, run manager, figures, or domain packages.
- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_durability_backend.py tests/unit/test_atomic_artifacts.py -q`. Expected: FAIL on missing backends/integrity reader.
- [ ] **Step 4: Implement and startup-probe both contracts.** Select by platform explicitly, record implementation/source hash and filesystem/volume facts, and fail closed on unknown/network/FUSE/cloud-sync semantics. Route existing atomic callers through the compatible backend without weakening H1--H8 behavior.
- [ ] **Step 5: Run focused GREEN.** Run the Step 3 command. Expected: PASS on real and fake syscall seams without scientific data.
- [ ] **Step 6: Review and commit.** Reviewer checks syscall ordering, same-volume proof, failure states, dependency direction, and backward compatibility. Commit `feat(artifacts): add tested durability backends`.

### Task 3: Implement Candidate Archive Acquisition, Source/License Recording, and Offline Reuse

**Files:**
- Modify: `vfe4/data/__init__.py`
- Create: `vfe4/data/wikitext103.py`
- Modify: `vfe4/data/access.py`
- Create: `tests/unit/test_wikitext103_source.py`

**Interfaces:** Produce `acquire_wikitext103`, `WikiText103SourceRecord`, `SealedDatasetRef`, split identities, and bounded source-record generation. No model-facing tensor is returned.

- [ ] **Step 1: Write failing tiny-response/archive tests.** Build HTTP responses and archives in memory/temporary paths only. Prove request/final redirect-chain ordering, HTTPS final origin, status/content headers, archive signature/content-type rule, source-page `4,194,304`-byte/content-type/UTF-8 bounds, deterministic unique license paragraph/link extraction with exact byte offsets/slice hash, the exact four-entry archive inventory, method/size/ratio/CRC/SHA checks, directory containment, regular-file rules, durability-backed staging/publication, exact offline reuse, and `allow_network=False` failure. Add one negative case per redirect loop/downgrade/ambiguity, oversized/wrong-type source page, zero/multiple/contradictory license matches, encryption, duplicate/case collision/traversal/drive/UNC/symlink/device/extra member, unsupported compression, CRC/size mismatch, zip bomb, changed source record, split swap, and any V3-resolved cache root.
- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_wikitext103_source.py -q`. Expected: FAIL on missing acquisition module.
- [ ] **Step 3: Implement bounded candidate acquisition and sealed storage.** Use one explicit HTTPS client seam injected in tests, no import-time network, fixed-size streaming reads, explicit request/final redirect records, ZIP central-directory validation before extraction, second validation while extracting, the Task 2 durability backend, content-addressed split directories, deterministic license offsets/text/link, and opaque capabilities. Capture the source page separately and never treat it as model input. Ambiguity returns a source-lock obligation rather than guessing.
- [ ] **Step 4: Implement manifest revalidation and V3 quarantine.** Every offline open rehashes the committed source record plus archive/member bytes. Resolve paths before comparison and reject V3 roots, symlinks, ambiguous files, or legacy filename-only provenance.
- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: PASS without network.
- [ ] **Step 6: Review and commit.** Reviewer checks path traversal, archive bombs, CRC/SHA ordering, split isolation, and no real access. Commit `feat(data): add verified WikiText-103 acquisition`.

### Task 4: Implement the Hermetic Candidate Tokenizer Validator and Cache Builder

**Files:**
- Create: `vfe4/data/tokenizer.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/test_wikitext103_tokenizer.py`

**Interfaces:** Produce `CandidateTokenizerContract`, `TokenizerDistributionAdapter`, `validate_tokenizer_adapter`, `build_tokenizer_spec`, `encode_sealed_split`, `TokenCacheIdentity`, and `open_token_cache` gated by split capability. Before Task 13, every adapter is synthetic and every resulting identity is fixture-only.

- [ ] **Step 1: Write failing hermetic tokenizer tests.** Install an import blocker for `tiktoken` and live `importlib.metadata` distribution/RECORD discovery. Supply synthetic adapters representing an accepted candidate and one independent mutation of name/version/vocabulary/special map/regex/ranks/ordinary-encoding policy/golden ASCII-Unicode-newline outputs. Prove the pure validator accepts only the exact synthetic contract, derives a deterministic fixture spec hash, and rejects every mutation. With a tiny reversible synthetic byte tokenizer, assert strict UTF-8, no normalization/BOS/EOS, exact raw-byte round trip, int32 little-endian payload, size/hash/min/max/count, and independent split parents; reject out-of-range IDs, malformed payload, V3 cache/provenance, fitted state, and manifest mismatch. No test asserts a fact about the installed package.
- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_wikitext103_tokenizer.py -q`. Expected: FAIL on missing tokenizer module/dependency pin.
- [ ] **Step 3: Implement pure adapter validation and cache mechanics.** Make validation depend only on a passed `TokenizerDistributionAdapter`; do not import `tiktoken`, query installed metadata, read live tables, or execute production golden vectors. Implement isolated one-split encode/round-trip/publication against the injected adapter and durability backend. Keep the production adapter behind the explicit Task 13 source-lock orchestration seam, and make every pre-source-lock identity carry `evidence_scope="synthetic_fixture_only"` so readiness rejects it.
- [ ] **Step 4: Add preprocessing resource checks.** Forecast peak disk/RAM from raw size and a smoke-measured multiplier, report the estimate, and fail before encoding if `2*forecast+10 GiB` disk or required host headroom is unavailable. Never fall back to linewise/different segmentation after a memory failure.
- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: PASS on synthetic adapters/tiny strings only, with the live import blocker still active.
- [ ] **Step 6: Review and commit.** Reviewer checks no live package/distribution/table/golden access, synthetic evidence labeling, full-split mechanics, round trip, cache ownership, and cross-split isolation. Commit `feat(data): add hermetic tokenizer contract validator`.

### Task 5: Build Exactly-Once Windows, Deterministic Schedules, and Split Capabilities

**Files:**
- Modify: `vfe4/data/windows.py`
- Modify: `vfe4/data/access.py`
- Create: `tests/unit/test_training_windows.py`

**Interfaces:** Produce `CausalWindowSet`, `CausalBatch`, `WindowManifest`, `PermutationManifest`, `DataCursor`, `materialize_train_data`, and test-only opening through the evaluation capability.

- [ ] **Step 1: Write failing exhaustive small-stream tests.** Use only synthetic token-cache identities from the injected Task 4 adapter. For token lengths `2..520`, compare windows/masks against an independent transition enumeration; assert exactly `n-1` target IDs, one padded final window, EOT input padding, `-100` targets, no split crossing, no dropped final batch, ascending validation/test, and stable train permutation/hash. Test cursor save/restore at every batch boundary and prove the resumed next window IDs and cumulative denominator match uninterrupted execution. A manifest hash is not consumable until its payload has been durably closed and revalidated.
- [ ] **Step 2: Write failing access/static tests.** Training can open train/validation only after a valid readiness token; no training import reaches test unsealing; sealed test hashes may exist but bytes/tokens cannot be mapped. Reject stale readiness/data hashes, worker/distributed policy changes, altered permutation bytes, duplicate/missing IDs, and a cursor beyond the schedule.
- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_training_windows.py -q`. Expected: FAIL on missing window/schedule APIs.
- [ ] **Step 4: Implement windows and stored permutations.** Use memory-mapped int32 source, cast only each slice to `torch.long`, keep masks explicit, store PCG64 schedule bytes atomically, and set `num_workers=0`. Make counted targets a first-class integer, not a reconstructed mean denominator. This task implements and fixture-tests builders only; Task 13 instantiates and freezes the official train/validation/test `WindowManifest` values, pass-0/pass-1 train `PermutationManifest` values, ascending validation/test schedules, and batch/evaluation/checkpoint cadence manifests before any preregistration or experiment-plan field consumes their hashes.
- [ ] **Step 5: Run focused GREEN.** Run the Step 3 command. Expected: PASS.
- [ ] **Step 6: Review and commit.** Reviewer checks boundary arithmetic, final padding, shuffle reproducibility, resume cursor, and test capability. Commit `feat(data): add deterministic causal schedules`.

### Task 6: Create the Five Explicit Arm Factories, Training Sparsity, and Post-H8 Readiness

**Files:**
- Modify: `vfe4/training/__init__.py`
- Create: `vfe4/training/wt103_models.py`
- Create: `vfe4/training/factories.py`
- Create: `vfe4/training/formulas.py`
- Create: `vfe4/training/sparsity.py`
- Create: `vfe4/training/readiness.py`
- Create: `tests/unit/test_training_factories.py`
- Create: `tests/promotion/test_training_sparsity.py`
- Create: `tests/promotion/test_post_h8_training_readiness.py`

**Interfaces:** Consume the exact `WT103ArmSpec`/`WT103GateSpec`/`EndpointInventory` records from Task 1 and produce `A0ArchitectureProfile`, `A0FormulaRecord`, `reconstruct_a0_parameters`, `reconstruct_a0_flops`, `build_training_arm`, `audit_arm_matching`, `certify_training_sparsity`, `TrainingSparsityCertificate`, `validate_post_h8_readiness`, `PostH8ReadinessToken`, exact factory/config/inventory hashes, and `WT103PredictorSafetyCertificate`. Readiness consumes the bounded H6 Prefix v2 certificate set and the narrowed H6 Prediction v3 result; legacy H6 v1 evidence cannot satisfy either dependency.

- [ ] **Step 1: Write failing factory/formula tests.** Assert A0 exactly matches `A0ArchitectureProfile`: full causal two-head `flash_attention_only_no_fallback` policy mapped to the frozen `sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])` API with alternatives disabled and no mask/weights/fallback, learned absolute positions, pre-norm plus final norm, exact residual topology, tanh-GELU, and every projection/bias/tie choice. Assert the resolved PyTorch version/API/backend hashes enter the architecture hash; substitute another backend or fallback and require rejection. A0 has no latent/recognition/snapshot phase and exactly one CE/model optimizer phase. Reconstruct `P_A0(h)=2Vh+128h+12h^2+15h+V` from actual named tensor shapes with every parameter counted once. Independently hand-enumerate a tiny full-causal operator ledger, including semantic attention pairs, decoder chunks, backward, CE, and AdamW, and require exact equality with `A0FormulaRecord`, whole-schedule `F_A0`, and both canonical schema hashes; decoder rechunking must preserve FLOPs. Assert each of the five factories exactly follows its arm spec: both complete rows use complete ELBO, the parent-specific emission row uses only its bound non-ELBO objective intervention, no-latent has no recognition/snapshot/SMC work, and scorer dispatch follows `scorer_kind` rather than the A0/A5 label. Forbid filler/no-op/dormant capacity and enforce 1%/5% PRIMARY training matching from the deterministic finite search.
- [ ] **Step 2: Write failing training-sparsity tests.** Bind the literal profile and exact ordered five-arm factory inventory, then trace every distinct enumerated train, applicable E-like proposal, snapshot, backward, optimizer, scorer, and checkpoint path. Require permitted vocabulary/logit chunks, A0 Q/K/V/result tensors, the exact nonmaterialized Flash operator, and block/banded population shapes. Reject backend fallback, explicit/materialized attention masks or weights, `[B,2,L,L]` or aggregate pair-axis storage, and every forbidden population/source/pair/logit shape; reconcile every unique storage as `numel*element_size` with allocator overhead separate and zero unclassified bytes. Require each math-SDPA/materialized-attention, dense-population, batch-dense, full-source, pair-slab, full-decoder, selector/RHS, and unclassified-checkpoint negative control to fire before allocation or serialization. Prove H8 and the 85% capacity preflight cannot populate or replace this certificate.
- [ ] **Step 3: Write failing readiness tests.** Use only injected synthetic tokenizer records in this task and prove their `synthetic_fixture_only` scope cannot issue a production token. Require typed, manifest-validated integrity records for H5, H6-Prefix, H6-Prediction, H7, and exact prerequisite H8 PASS; exact source/tokenizer/window/permutation/cadence/profile/A0-architecture/formula/factory/objective/update/snapshot/estimator identities; a PASS same-revision `TrainingSparsityCertificate`; the independent 85% capacity record; throughput/power/resource authorization; clean source candidate; and no changed dependency closure. Prove H6 byte certificates and H8 allocation endpoints cannot satisfy GPT-2 predictor safety, training sparsity, or training capacity. Exhaustively perturb current target/suffix/cache traversal on bounded `V=50257` synthetic cases and run static signature/import/taint/cache audits against the final predictor. Readiness consumes typed integrity records and never hashes an arbitrary caller path.
- [ ] **Step 4: Run focused RED.** Run `python -m pytest tests/unit/test_training_factories.py tests/promotion/test_training_sparsity.py tests/promotion/test_post_h8_training_readiness.py -q`. Expected: FAIL on missing factories/sparsity/readiness.
- [ ] **Step 5: Implement explicit factories, formulas, and matching.** Use direct constructors, not signature inspection/registries. Implement the exact A0 architecture and analytical formula records; freeze the deterministic finite A0 width selection before outcomes and publish complete architecture/formula hashes, parameter-role, optimizer-ID, phase-FLOP, dimension, and margin tables. The selected width cannot change because Flash is unavailable; that condition fails readiness.
- [ ] **Step 6: Implement the revision-bound sparsity certificate.** Run each exact path in clean child processes; combine dispatch, profiler, CUDA allocator/unique-storage, exact A0 SDPA-backend, and serializer inventories; classify the sole nonmaterialized full-causal Flash semantic domain, permitted vocabulary/logit dense work, and block/banded population work separately from forbidden pair-axis/population storage. Execute all assigned controls and publish formulas, logical/physical byte reconciliation, status, and certificate hash. Missing backend/materialization observability is INCONCLUSIVE and any fallback, forbidden, or unclassified event is FAIL.
- [ ] **Step 7: Implement fail-closed readiness.** Bind exact typed evidence identities and issue an opaque token only after every predecessor, GPT-2 predictor-safety, finalized data/schedule identity, arm match, PASS training-sparsity certificate, independent capacity preflight, metric/figure source, durability probe, and throughput/disk/resource precondition passes. A prior FAIL becomes an INCONCLUSIVE readiness obligation; no data/training object is constructed.
- [ ] **Step 8: Run focused GREEN.** Run the Step 4 command. Expected: PASS on injected tiny predecessor/data artifacts and exact synthetic path traces.
- [ ] **Step 9: Review and commit.** Reviewer checks H8 exact precondition, H6 nontransfer, target blindness, structural sparsity versus capacity separation, active capacity, and no strengthened update labels. Commit `feat(training): add matched arms and readiness gate`.

### Task 7: Implement the Typed Training Engine, Complete ELBO Updates, and Recording

**Files:**
- Create: `vfe4/training/engine.py`
- Create: `vfe4/recording/__init__.py`
- Create: `vfe4/recording/metrics.py`
- Create: `vfe4/recording/tables.py`
- Create: `vfe4/recording/failures.py`
- Create: `tests/unit/test_training_engine.py`
- Create: `tests/unit/test_training_metrics.py`

**Interfaces:** Produce `train_step`, `train_attempt`, `append_metric`, `validate_metric_log`, `export_metrics_csv`, `append_failure`, and exact `StepResult`/`MetricRecord` identities.

- [ ] **Step 1: Write failing engine tests.** Use tiny injected arms to prove phase order, block freezing, reverse-mode scope, complete ELBO term equality, snapshot clone/detach/nonalias/hash, acceptance and exact rollback of parameters/optimizers/schedulers, finite/gradient/SPD checks, A0 absence of recognition, and failure classification. Prove an `adam_proposal` never reports monotonicity or exact/MM/GEM status.
- [ ] **Step 2: Write failing recording tests.** Require every metric family and raw numerator/denominator listed above, exact ordinals/hash chain, fsync seam, deterministic CSV columns/decimal strings, incomplete-final-fragment recovery, and hard failure on a malformed complete record/chain. Assert effective source count is exactly `exp(entropy_sum/source_row_count)` and zero rows are explicit `not_applicable`; every update records learning rate, scheduler ordinal/state, AMP scale/overflow/applicability, clipping threshold/pre/post norm/clipped flag, and all effective AdamW parameters. A0 complete-ELBO/recognition/source/SPD fields must be explicit `not_applicable`, never fabricated zero. Force a metric-write failure and prove the independent failure ledger retains the terminal event.
- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_training_engine.py tests/unit/test_training_metrics.py -q`. Expected: FAIL on missing engine/recorders.
- [ ] **Step 4: Implement only typed phase execution.** Dispatch only from `WT103ArmSpec.update_phases`: A0 and no-latent run their one CE/model phase, while each latent row runs recognition proposal, frozen snapshot, then its objective-bound model proposal. Centralize complete-ELBO assembly for applicable rows, the separately bound emission-only non-ELBO path, proposal labels, acceptance, rollback, gradient instrumentation, target-blind validation boundaries, and exact counted-target accumulation. A family label never adds an absent phase.
- [ ] **Step 5: Implement canonical logs and exports.** Keep JSONL authoritative, use one schema/field order, store raw terms, numerator/denominator state, exact update controls, applicability, and timing/resource counters, and derive CSV only after full validation. Define the checkpoint scientific metric projection separately from UTC/duration/artifact fields. Do not round away source values.
- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS.
- [ ] **Step 7: Review and commit.** Reviewer checks objective completeness, autograd terminology, rollback completeness, metric denominators, and failure durability. Commit `feat(training): add typed engine and metric ledger`.

### Task 8: Add Safe Atomic Checkpoints and Scientific-State Resume Validation

**Files:**
- Create: `vfe4/checkpoint/__init__.py`
- Create: `vfe4/checkpoint/schema.py`
- Create: `vfe4/checkpoint/io.py`
- Create: `vfe4/checkpoint/serialization.py`
- Create: `vfe4/checkpoint/migrations.py`
- Create: `tests/unit/test_training_checkpoint.py`

**Interfaces:** Produce `save_checkpoint`, `load_checkpoint`, `ResumeContract`, `CheckpointBundle`, `CheckpointIdentity`, `LoadedCheckpoint`, and empty `MIGRATION_PROFILES`. Checkpoint I/O returns identities/bundles only and never publishes a run or experiment manifest.

- [ ] **Step 1: Write failing scientific-resume and role tests.** Round-trip all active model/recognition/optimizer/scheduler/AMP/RNG/estimator/data-cursor/update-trace/metric-numerator-denominator/config/objective/schema/evidence fields, exact `checkpoint_role`, and the immutable pre-attempt `experiment-plan.json` hash. Compare uninterrupted versus split/resumed training at every small-fixture checkpoint: require equal `scientific_state_sha256`, elementwise-equal scientific tensors/primitives, equal next window IDs and metric ordinals, and bitwise-equal next two batches of prior predictions. Prove `resume_only` checkpoints restore the same attempt but are rejected by confirmation/test/endpoint/figure validators; only the complete post-pass checkpoint can carry `terminal_scoring`. Explicitly allow paths, PIDs, UTC/monotonic timestamps, durations, serialization order, terminal run-manifest hashes, and `artifact_sha256` to differ. Assert the payload/body/artifact formula has no self-reference, checkpoint manifests cannot depend on the later terminal run manifest, and resume events live in a separate immutable lineage.
- [ ] **Step 2: Write failing corruption/security/mismatch tests.** Before any read, reject nonregular/link files, over-plan size, declared/actual size mismatch, and payload SHA mismatch. Require `weights_only=True`, CPU mapping, no registered safe globals, no fallback, a recursive exact primitive/tensor whitelist, and declared dtype/shape/numel inventory; a malicious custom reducer fixture must neither execute nor create its sentinel. Truncate/flip bytes and vary each schema/config/objective/model/arm/optimizer/precision/dependency/tokenizer/data/window/permutation/experiment-plan/evidence/cursor field independently. Require fail-closed before mutation. Reject device-implicit loads, unknown migrations, V3 checkpoint keys/paths, and a migration that lacks exact source/destination/code/loss/test identity.
- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_training_checkpoint.py -q`. Expected: FAIL on missing checkpoint package.
- [ ] **Step 4: Implement bounded safe serialization and durability-backed I/O.** Capture all scientific state and schedule identities, compute distinct canonical scientific and exact artifact hashes, write through the Task 2 durability backend, read back only through the bounded `weights_only` validator, and return `CheckpointIdentity`/`LoadedCheckpoint`. Load into a fresh object only after full compatibility validation. The run manager, not checkpoint I/O, appends resume-lineage events and publishes terminal manifests from the returned identity.
- [ ] **Step 5: Keep migrations closed.** Ship an empty mapping and a permanent error explaining that V3 is not a source schema. Tests document the process a future explicit VFE4 migration must satisfy.
- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS.
- [ ] **Step 7: Review and commit.** Reviewer checks scientific-versus-artifact identity, exact optimizer/scheduler/RNG/data/metric resume, malicious reducer rejection, durability ordering, noncircular plan/terminal manifests, no parent-manifest mutation, and no V3 path. Commit `feat(checkpoint): add exact VFE4 resume bundles`.

### Task 9: Add Prior Evaluation, Estimator-Aware Statistics, and One Test Opening

**Files:**
- Modify: `vfe4/evaluation/__init__.py`
- Modify: `vfe4/evaluation/prior_nll.py`
- Create: `vfe4/evaluation/statistics.py`
- Create: `vfe4/evaluation/test_opening.py`
- Create: `tests/unit/test_training_evaluation.py`
- Create: `tests/property/test_training_prefix_safety.py`

**Interfaces:** Produce `score_prior_nll`, `aggregate_a5_smc`, `paired_prediction_decision`, `reserve_test_opening`, and exact raw/test result records.

- [ ] **Step 1: Write failing target-blind/corpus tests.** Use fake predictors to prove the exact public signature, no recognition/target/suffix import path, ascending complete evaluation windows, `-100` exclusion, `math.fsum` corpus numerator, exact denominator, PPL derivation, cache cold/warm/reverse equality, and separation of emission-only diagnostics. Perturb current targets/suffixes under GPT-2 vocabulary and require identical raw predictions/cache records for fixed prefix/stream.
- [ ] **Step 2: Write failing one-opening tests.** Require the exact terminal-checkpoint logical keys and count derived from the frozen `EndpointInventory`, every checkpoint role exactly `terminal_scoring`, a complete run group, analysis/figure/estimator/data/inventory hashes, no prior reservation, and no active verification marker before the durability backend's exclusive create. Substitute any rolling `resume_only` checkpoint, omit or duplicate any derived key, or alter a role and fail before test capability or scoring. Require durable reservation replacement/reopen before issuing capability. Prove training/tuning cannot map test, a second open fails, and any crash after reservation is terminal.
- [ ] **Step 3: Write failing statistics tests.** Hand-author the complete five-arm, eight-seed table with scorer-kind-derived exact or 64-stream/four-particle raw records. Assert exact inventory, ordered gate roles, Q0/Q1/Q2/R1/R2 and H6 bounds for weighted arms, common-stream covariance, parent-specific-complete estimator error radius, 256 PRIMARY corners, frozen constants/delta, objective-before-primary status rules, and no batch-mean averaging. Missing/duplicate/nonfinite records force the associated gate INCONCLUSIVE; completed partial endpoint records remain durably inspectable but cannot enter any aggregation, promotion, or scientific result.
- [ ] **Step 4: Run focused RED.** Run `python -m pytest tests/unit/test_training_evaluation.py tests/property/test_training_prefix_safety.py -q`. Expected: FAIL on missing evaluation APIs.
- [ ] **Step 5: Implement target-blind scoring and exclusive opening.** Keep the unsealer private to `test_opening.py`, pass only `DurableTestOpeningCapability`, append and validate every completed raw endpoint record independently, and never reopen after reservation. A crash preserves partial records and sets terminal INCONCLUSIVE rather than pretending no scoring occurred.
- [ ] **Step 6: Implement exact frozen aggregation.** Aggregate only a complete, unique, finite frozen inventory. Reuse the H6 algorithm and constants by public API where identity-compatible, but create a new WT103 protocol/hash and record that the dataset/tokenizer/checkpoint evidence is new. A0 exact scores remain exact and do not acquire fake Monte Carlo variance.
- [ ] **Step 7: Run focused GREEN.** Run the Step 4 command. Expected: PASS.
- [ ] **Step 8: Review and commit.** Reviewer checks target blindness, one opening, corpus weighting, estimator propagation, and claim scope. Commit `feat(evaluation): add one-opening prior scoring`.

### Task 10: Build Atomic Run Directories, Manifests, Provenance, and Resource Preflights

**Files:**
- Create: `vfe4/artifacts/run_directory.py`
- Create: `vfe4/artifacts/environment.py`
- Modify: `vfe4/artifacts/provenance.py`
- Create: `tests/unit/test_training_artifacts.py`

Consume unchanged from Task 2: `vfe4/artifacts/manifest.py`, `vfe4/artifacts/atomic.py`, and `vfe4/artifacts/durability.py`.

**Interfaces:** Produce `reserve_run`, `publish_experiment_plan`, `finalize_run`, `validate_run_manifest`, `publish_experiment_index`, `capture_environment`, `forecast_resources`, `ResourceForecast`, `ResourceUsageLedger`, and `run_allocation_preflight`.

- [ ] **Step 1: Write failing lifecycle tests.** Assert the run manager durably publishes immutable `experiment-plan.json` before any attempt, reserves exclusively with no overwrite/newest selection, resumes explicitly, atomically closes success/failure, consumes typed generic integrity records, preserves crashed attempts, publishes a terminal manifest from returned checkpoint identities, and then publishes the experiment index. Inject failures at every create/write/replace/reopen/manifest/index boundary and prove checkpoint I/O never publishes or rewrites a parent manifest.
- [ ] **Step 2: Write failing provenance/resource tests.** Require all git/dependency/hardware/runtime/data/evidence/inventory identities; distinguish clean and dirty; use the exact byte forecast formula; reject insufficient disk/host/device/checkpoint-duplicate headroom; and require independent shape-identical allocation records for every distinct arm path plus a PASS training-sparsity record. Benchmark every frozen preparation/tuning/confirmation/validation/checkpoint/test/table/figure component with the specified warmups/samples, minimum throughput, maximum duration/power, `1.25` headroom, and work counts derived only from `EndpointInventory`. Require a provenance-bound `100 ms` power provider, complete GPU-hour/wall/energy formulas, hard `720 h`/`840 h`/`500 kWh` ceilings, and pre-test remaining-budget recomputation. Missing power evidence is INCONCLUSIVE; H8 endpoints cannot populate either training preflight.
- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_training_artifacts.py -q`. Expected: FAIL on missing lifecycle/provenance APIs.
- [ ] **Step 4: Implement the run-manager lifecycle above generic integrity.** Consume Task 2 primitives without adding domain imports to them, use explicit paths and content-addressed identities, publish the pre-run plan before attempts, retain terminal failure artifacts, append resume lineage separately, and never delete another attempt as cleanup. The run manager alone publishes terminal run/experiment manifests after receiving checkpoint identities.
- [ ] **Step 5: Implement environment, capacity, forecast, and usage records.** Capture values before device work; measure the exact shape-identical allocation path and every throughput/power component; enforce the independent 85% device caps, disk formula, forecast headroom, immutable ceilings, and actual-usage debits. Require enough remaining forecast and disk headroom before test reservation. Any insufficiency requires an explicit config/preregistration/user-authorization revision rather than an automatic shrink.
- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS.
- [ ] **Step 7: Review and commit.** Reviewer checks crash recovery, manifest closure, dirty digest, exact index, and training-owned resource evidence. Commit `feat(artifacts): add atomic training run lifecycle`.

### Task 11: Implement Deterministic Figure Generation from Immutable Metrics Only

**Files:**
- Modify: `pyproject.toml`
- Create: `vfe4/figures/__init__.py`
- Create: `vfe4/figures/spec.py`
- Create: `vfe4/figures/load.py`
- Create: `vfe4/figures/plots.py`
- Create: `vfe4/figures/render.py`
- Create: `generate_vfe4_figures.py`
- Create: `tests/unit/test_training_figures.py`
- Create: `tests/integration/test_generate_vfe4_figures.py`

**Interfaces:** Pin `matplotlib==3.10.6`; produce eight explicit plot functions, `render_figure_set`, `FigureSpec`, `FigureSetManifest`, and import-safe figure `CONFIG` resolution.

- [ ] **Step 1: Write failing import/dependency tests.** Import the launcher and every figure module while blocking CUDA/data/model/training/checkpoint calls. Assert no side effects and statically reject forbidden imports. Require one explicit manifest path and reject newest/glob/path escape/unknown config.
- [ ] **Step 2: Write failing figure-schema tests.** Build terminal-manifest-validated finalized JSONL plus frozen final result-table JSON for the exact `EndpointInventory`. Require exact equality with its derived `figure_panel_keys` and ordered `figure_series_keys`, plus the frozen required-figure registry, aggregations, uncertainty, labels, units, applicability, SVG+PNG+PDF, plotted CSV+JSON, caption+alt text, stable spec/input/output hashes, and a content-addressed figure set. Regenerate published `metrics.csv` from JSONL and require byte equality before rendering. Delete one required finalized numerator, panel key, or series key; substitute a partial run; alter CSV; collapse controls into one VFE series; or mark an inapplicable field applicable and prove rendering fails rather than opening a checkpoint, trusting CSV, recomputing, or fabricating zero.
- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_training_figures.py tests/integration/test_generate_vfe4_figures.py -q`. Expected: FAIL on missing figure package/launcher.
- [ ] **Step 4: Implement deterministic style and readers.** Use `Agg`, fixed fonts/rcParams/hashsalt/metadata, sorted identities, terminal-manifest-validated finalized JSONL plus frozen result-table JSON as the only semantic inputs, explicit functions, and no registry/signature discovery. Regenerate CSV solely as an audit projection and require byte equality with the published export.
- [ ] **Step 5: Implement and render the frozen required specs.** Put raw seed traces beside uncertainty summaries; keep each ordered arm and result role distinct; label complete ELBO and the emission-only non-ELBO objective as different quantities; keep NLL and PPL axes distinct; show resource/conditioning allowances next to their endpoints; and render every inapplicable recognition/source/SPD/ELBO field with its explicit reason rather than a fake zero. Reject every partial attempt/result set. Write sidecars before image publication and validate all formats before indexing.
- [ ] **Step 6: Run focused GREEN twice.** Run the Step 3 command twice in separate temporary output roots. Expected: PASS and byte-identical spec/data/semantic hashes; image byte hashes must match within the same pinned environment.
- [ ] **Step 7: Review and commit.** Reviewer checks no training/checkpoint import, metric authority, statistical labeling, accessibility text, and atomic output. Commit `feat(figures): add reproducible VFE4 run figures`.

### Task 12: Add the Click Training Launcher and Tiny End-to-End Smoke

**Files:**
- Modify: `train_vfe4.py`
- Create: `tests/integration/test_train_vfe4.py`
- Modify: `README.md`

**Interfaces:** The launcher orchestrates resolver, exact refs, acquisition/readiness, the immutable arm inventory, training/evaluation/checkpoint/recording/artifacts, and no figure logic. `generate_vfe4_figures.py` is the separate pure figure launcher. Neither owns probability/data/metric logic.

- [ ] **Step 1: Write failing launcher tests.** Install blockers for `tiktoken`, live distribution/RECORD discovery, and live tokenizer tables/golden vectors; inject only synthetic tokenizer adapters. Import the live editable dictionary with every external side effect mocked and require none. Assert the complete literal `WT103ExperimentProfile`/`A0ArchitectureProfile`, exact five-arm/gate inventory, explicit `idle|source_lock|readiness|train|resume` modes, durability roots, and resource ceilings resolve without hidden defaults. Replace only paths/data sizes/arm dimensions/steps through a typed test helper, and run generated smoke data through every distinct arm path, interruption from a `resume_only` checkpoint, validation, terminal `terminal_scoring` scientific/artifact identity, and stable metrics CSV. Separately import and exercise the pure figure launcher from finalized tiny artifacts. Unknown live key, unresolved source-lock fact in training mode, stale H8/sparsity/inventory ref, failed durability probe, or failed forecast fails before reservation.
- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/integration/test_train_vfe4.py -q`. Expected: FAIL on missing launcher.
- [ ] **Step 3: Implement one-click orchestration.** Keep one editable dictionary, one main, one guard, clear constants for cache/run roots, `operation="idle"` by default, and exact `idle|source_lock|readiness|train|resume` operation values. Source-lock mode builds all derived manifests before exposing their hashes; readiness creates no corpus optimizer update; train/resume modes require the literal profile, frozen source/schedule/inventory identities, durability probes, training-sparsity certificate, capacity preflight, and resource authorization. Print resolved run identity, predecessor statuses, data/tokenizer/schedule/inventory identities, forecast/actual usage, arm/seed progress, terminal status, and artifact path. Add no argparse/env-required setting or hidden fallback. Keep figure roots and figure operations exclusively in the separate editable `generate_vfe4_figures.py` dictionary.
- [ ] **Step 4: Document operator workflow and V3 boundary.** README gives click Run instructions for source lock, training, resume, and figure regeneration; exact disk/device preflights; one-opening warning; cache location; and explicit statements that V3 files are only design references and H8 is not training-memory evidence.
- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: PASS on generated smoke data only.
- [ ] **Step 6: Review and commit.** Reviewer checks click UX, import safety, no real data access, no CLI, and artifact paths. Commit `feat(training): add click-run VFE4 experiment`.

### Task 13: Perform the Separately Authorized Source, Tokenizer, Window, and Schedule Lock

**Files:**
- Produce/update tracked: `docs/data/wikitext103-raw-v1-source-record.json`
- Produce/update tracked: `docs/data/wikitext103-raw-v1-source-record.md`
- Modify tracked: `docs/preregistrations/2026-07-21-post-h8-wikitext103-training.md`
- Produce ignored/external: official archive, sealed raw splits, token caches, and their atomic manifests.

**Precondition:** Tasks 1--12 are reviewed and committed; the user separately authorizes the real network acquisition. No training, optimizer update on corpus data, validation scoring, or test scoring is authorized by source-lock permission.

- [ ] **Step 1: Revalidate the initial exact H8 PASS.** Validate artifact, manifest, ledger, exact source/config tuple, and its `no_post_h8_training_memory_transfer` nonclaim. If absent/stale/invalid, stop before network.
- [ ] **Step 2: Validate candidate locations and acquire exactly once through `train_vfe4.py` source-lock mode.** Use only the preregistered candidate request URLs; record the complete request-to-final redirect chains, bounded response/content-type facts, archive inventory/CRC/size/SHA/path safety, and deterministic license paragraph byte offsets/slice hash/text/links. Seal the three splits. Any redirect, source-page, license, archive, or multiple-download ambiguity preserves immutable failure artifacts and stops for an explicit preregistration revision before tokenizer/window work; it is never resolved by human outcome-aware choice.
- [ ] **Step 3: Perform the first live tokenizer import, freeze the installed distribution, and build caches split-by-split.** Remove the hermetic blocker only inside source-lock mode, import `tiktoken` for the first time, query installed distribution name/version and every distribution-file RECORD hash, extract/hash the exact live regex/ranks/special-token tables, and execute the production ASCII/Unicode/newline golden encode/decode vectors. Validate all observations against the three candidate strings and pure Task 4 validator, then publish the final `TokenizerSpec`; any import, package, RECORD, table, regex, policy, or golden-vector mismatch stops before windows or outcomes. Verify strict UTF-8, full-split encode/decode round trip, token ranges/counts/payload hashes, and separate split parentage. Keep test sealed and prove no training import mapped it.
- [ ] **Step 4: Build and close every derived data/schedule manifest before consuming a derived hash.** From the closed token identities, durably build and revalidate train/validation/test `WindowManifest` values; pass-0 and pass-1 train `PermutationManifest` values from `data_order_seed=2026072199` (quarter-pass tuning consumes the pass-0 prefix); ascending validation/test schedules; and exact batch, 20-boundary-per-pass evaluation, resume-only boundary-checkpoint, and post-pass terminal-scoring checkpoint manifests. Validate coverage, counts, payload bytes, NumPy/PCG64 identity, checkpoint-role exclusivity, and independent split parentage. Until all are closed, no preregistration, readiness input, run-group record, or experiment plan may read or publish one of their hashes.
- [ ] **Step 5: Atomically publish the complete pre-outcome source lock.** Machine-write the JSON from validated bytes, generate the Markdown interpretation, and update the preregistration with the final request/final URLs, redirects, license offsets, archive/member/tokenizer/table/token/window/permutation/evaluation/checkpoint schedule hashes. Resolve `flash_attention_only_no_fallback` to the installed PyTorch version and exact `torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])` API/backend identities, record that alternatives are disabled, hash those identities into the final A0 architecture record, and reject any missing API or alternate selection. Set `freeze_completeness=true` only after every required field is closed and internally revalidated. No hash is hand-entered, guessed, or selected from multiple candidate downloads.
- [ ] **Step 6: Review and commit source lock.** Two reviewers independently compare archive/member/token/window/schedule manifests and the tracked record, license/source citation, installed distribution/table identity, path containment, no early hash consumption, and no V3 cache. Commit `docs(data): lock verified WikiText-103 source and schedules`.

### Task 14: Freeze the Final Integration Candidate and Run One Full JUnit

**Files:**
- Modify: none after candidate selection.
- Produce outside tracked source: `C:\tmp\vfe4-post-h8-wikitext103-integration.xml`, same-candidate predecessor compatibility artifacts/ledgers, predictor-safety/readiness artifacts, allocation preflight, and `.verification/post-h8-wt103-<FULL_HEAD>-<CONFIG_SHA>-ledger.json`.

- [ ] **Step 1: Freeze one clean exact revision.** Require full 40-character HEAD, no tracked/index diff, no nonignored untracked source, exact source/prereg/dependency/config hashes, no active verification marker, and immutable prior ledgers. Bind bootstrap H5/H6/H7/H8 refs.
- [ ] **Step 2: Run one full suite exactly once.** Run `python -m pytest -q --junitxml=C:\tmp\vfe4-post-h8-wikitext103-integration.xml`. Expected: exit 0. Parse tests/failures/errors/skips/duration only from XML. A source defect requires a new commit and one replacement XML; never combine revisions or run another broad suite for confidence.
- [ ] **Step 3: Reproduce same-candidate prerequisite compatibility without real training.** Through pure config projections, validate H5 update schema, H6 target-blind scorer/estimator, H7 covariance dependency, and rerun H8 at the exact candidate if required by its freshness contract. Validate each artifact/ledger and record why any immutable H6-Prediction one-opening artifact remains compatible and unreachable from append-only training branches.
- [ ] **Step 4: Run same-revision predictor, sparsity, capacity, durability, and resource readiness.** Produce the exact `WT103-Predictor-Safety` certificate, A0 architecture/parameter/FLOP reconstruction and finite training-match report, and proof that the frozen API context selected `SDPBackend.FLASH_ATTENTION` exactly with every alternative/fallback disabled and no attention materialization. Bind the observed PyTorch/API/backend identities into the A0 architecture hash, then produce typed source/token/window/permutation/cadence/checkpoint-role/endpoint-inventory integrity records, revision-bound `TrainingSparsityCertificate`, independent shape-identical 85%-capacity/checkpoint preflight for every distinct arm path, durability probes, disk forecast, and component-level throughput/power/GPU-hour/wall/energy forecast under the hard ceilings. The sparsity trace executes synthetic forward/applicable E-proposal/snapshot/backward/optimizer/evaluation/checkpoint paths but no corpus optimizer step or held-out score.
- [ ] **Step 5: Review and validate the integration ledger.** One reviewer checks data/split/token/window/cache safety; one checks model/update/autograd/checkpoint/resume; one checks evaluation/statistics/opening; one checks artifacts/metrics/figures/resources. Ledger claims bind exact eligible evidence; missing evidence is INCONCLUSIVE, never LLM consensus.
- [ ] **Step 6: Recheck immutability.** Recompute HEAD/diffs/source/config/manifests/JUnit/predecessor/ledger hashes. No tracked edit is permitted after readiness. Report exact candidate, XML totals, prerequisite identities, tokenizer/data/window identities, preflight maxima/allowances, and readiness status.

### Task 15: Run Authorized Equal Tuning and Confirmatory Training

**Files:**
- Modify: none.
- Produce ignored/external: atomic tuning/selection/confirmatory attempts, metrics, checkpoints, failures, manifests, and experiment index.

**Precondition:** Task 14 readiness and its claim ledger are PASS at unchanged revision/config/source/token/window/permutation/cadence/profile/factory/sparsity/capacity/durability/resource identities, and the user separately authorizes real WikiText-103 training with the published GPU-hour/wall/energy/disk forecast.

- [ ] **Step 1: Publish the immutable experiment plan, then reserve the group.** Before any attempt, create the canonical `EndpointInventory` from the exact five ordered `WT103ArmSpec` rows, ordered gates, six tuning cells, two tuning seeds, eight confirmatory seeds, and scorer protocol. Durably publish `experiment-plan.json` with that inventory and only its derived tuning-attempt, terminal-checkpoint, validation/test-record, result-row, and figure-series counts; independently entered counts are forbidden. Bind the exact source/token/window/permutation/cadence/profile/factory/checkpoint/estimator/opening/analysis/figure identities, resource ceilings/forecast, and expected artifact inventory. Every checkpoint binds both plan and endpoint-inventory hashes. Only after reopen validation may the run manager reserve attempts; existing identity blocks, with no overwrite or newest selection.
- [ ] **Step 2: Run equal tuning only.** Execute every inventory-derived two-seed quarter-pass grid cell for every trainable arm with the same pass-0 permutation prefix and the scorer selected by that arm's `scorer_kind`. Revalidate readiness/source/token/window/permutation/cadence/profile/factory/sparsity/resource/inventory identities before each launch and resume. Publish all metrics/failures and debit actual GPU seconds, wall seconds, sampled energy, and disk before selecting by the frozen per-arm rule.
- [ ] **Step 3: Freeze tuning selections and the confirmatory plan.** Atomically publish every arm's selection inputs, means, tie trace, selected settings, the exact terminal endpoint logical keys derived by `EndpointInventory` plus checkpoint schema/experiment-plan identities, and the one-opening plan. Runtime scientific/artifact hashes are intentionally unknown until each checkpoint closes. No confirmatory/test value can change selection.
- [ ] **Step 4: Run all inventory confirmation attempts.** Execute every ordered inventory arm for seeds `2026072101..2026072108`, two full passes, frozen pass-0/pass-1 permutations and validation/checkpoint boundaries, no early/best selection, and exact data order. Label every boundary checkpoint `resume_only`; it may restore that attempt but cannot be evaluated or promoted. After the complete passes, publish exactly one `terminal_scoring` checkpoint per arm/seed. Each checkpoint/resume and metric append revalidates experiment-plan/endpoint-inventory/parent/scientific/artifact identities; the run manager separately appends lineage and publishes terminal manifests. Debit actual resource use after every durable component. Pairing is used for the PRIMARY contrast only; controls retain their distinct result roles.
- [ ] **Step 5: Apply the frozen failure policy.** Retain every attempt. Permit only the one proved infrastructure retry; never replace a seed or modify config. If any required terminal checkpoint is missing, publish INCONCLUSIVE and stop before test opening.
- [ ] **Step 6: Freeze the complete derived terminal-checkpoint inventory.** Validate exact agreement with `EndpointInventory` and distinct checkpoint scientific/artifact identities, model/optimizer/scheduler/AMP/RNG/data-cursor/update-trace/metric-numerator-denominator/config/evidence hashes, counted targets/update opportunities, capacity/FLOP/optimizer access, resume lineage, and the final run-group manifest. Recompute conservative test-only throughput/power/GPU-hour/wall/energy and disk headroom from actual validation evidence; require at least `1.25*test_transaction_forecast` remaining before test authorization.

### Task 16: Perform the Single Test Opening, Publish Results, and Regenerate Figures

**Files:**
- Modify: none.
- Produce: immutable reservation, raw test estimator records, final result tables, figure set/sidecars, experiment manifest/index, and `.verification/post-h8-wt103-result-<FULL_HEAD>-<EXPERIMENT_SHA>-ledger.json`.

- [ ] **Step 1: Preflight the complete one-opening transaction.** Require unchanged Task 14 readiness/ledger/source/schedules, complete Task 15 group, and exact equality between all post-pass `terminal_scoring` checkpoint keys and the canonical `EndpointInventory`; require zero resume-only checkpoint in that inventory, exact sealed test identity, scorer-kind-derived stream/particle applicability, analysis/statistics/figure hashes, no prior reservation, and no active marker. Recompute test-only forecast from actual validation throughput/power and require remaining GPU-hour/wall/energy ceilings and disk each exceed `1.25*test_transaction_forecast`. Mechanically prove no training/tuning process mapped test.
- [ ] **Step 2: Durably reserve and open once.** Use the probed platform `DurabilityBackend` exclusive-create primitive (POSIX `O_CREAT|O_EXCL`; Windows `CreateFileW(CREATE_NEW, FILE_FLAG_WRITE_THROUGH)`), write/flush/reopen-validate exact identities and `state="RESERVED"`, durably publish without deleting or rewriting it, issue the sole capability, and open the sealed test tokens. A later crash is terminal INCONCLUSIVE.
- [ ] **Step 3: Attempt the complete fixed inventory and retain honest partial evidence.** Append and durability-validate every raw score record derived from each arm's `scorer_kind`, the eight confirmatory seeds, 64 stream IDs, and applicable particle ladder. Exact-autoregressive arms contribute one exact corpus total per seed; weighted-SMC arms contribute the full seed/stream/particle inventory. Every record binds `endpoint_inventory_sha256` and includes its numerator/denominator/cache/counter/failure fields. If execution stops, preserve every completed endpoint and failure record, mark the terminal transaction INCONCLUSIVE, never aggregate or promote the partial set, and never reopen. Missing/duplicate/nonfinite records cannot be repaired.
- [ ] **Step 4: Aggregate only a complete frozen inventory.** If and only if all required endpoint records are unique, finite, manifest-valid, and finalized, compute Q2/error bounds, eight paired seed effects, all 256 corners, inflated interval, PASS/FAIL/INCONCLUSIVE rule, NLL/PPL, ELBO/source/update/SPD/gradient/throughput/memory summaries, and frozen result-table JSON. Regenerate stable CSV projections from authoritative JSONL and require byte equality with every published CSV. State every nonclaim and applicability reason beside the result.
- [ ] **Step 5: Render all required figures from finalized immutable inputs.** Run `generate_vfe4_figures.py` by click or direct file execution with the explicit experiment manifest. Its only semantic inputs are finalized manifest-validated `metrics.jsonl` and frozen result-table JSON; it regenerates/checks CSV but never trusts it semantically, opens checkpoints/data, imports training, uses partial attempts, or fabricates inapplicable fields. Validate every registry- and inventory-derived SVG/PNG/PDF output, data sidecar, caption, alt text, spec/input/output hash, and the content-addressed figure index.
- [ ] **Step 6: Have fresh reviewers consume artifacts only.** Review one-opening completeness, raw corpus sums/estimator aggregation/statistics, metric numerator/denominator integrity, figure-to-sidecar agreement, provenance/resources, and claim scope. They do not rerun training, tests, scoring, or opening.
- [ ] **Step 7: Close and validate the result ledger.** Use one claim per data/tokenizer/window/schedule, predictor safety, training sparsity, capacity, durability, update/autograd, checkpoint scientific/artifact identity, resume, metric, opening, endpoint completeness, score/statistic, figure, artifact, and forecast/actual resource check. Evidence is revision/config/data/environment specific. Missing eligible evidence is INCONCLUSIVE.
- [ ] **Step 8: Report the evidence revision.** Report exact HEAD/config/experiment/reservation/result/figure/ledger hashes, JUnit totals, source/token/window identities, every arm/seed disposition, raw estimator uncertainty, inflated interval, measured training memory/allocation endpoints beside allowances, final artifact paths, and nonclaims. Do not edit tracked docs after closure.

---

## V3 Migration Boundary

Adapt only reviewed patterns: thin click orchestration, explicit data-loader boundaries, causal slicing arithmetic, dependency-injected train/evaluate functions, atomic writes, config-bound checkpoints, run provenance, and isolated figure workers. Reimplement them against VFE4 types and tests.

Do not copy or consume V3 `BeliefState`, moving-peer/free-energy objective, target-blind structural-EM claims, source weights as VFE4 posterior variables, 161-field config, registry/signature dispatch, cache filenames, filename-inferred cache provenance, checkpoint/best-model schema, newest-run figure selection, metric reconstruction, or model-specific figures. There is no direct checkpoint migration. A later one-way initializer requires a separate plan and cannot carry evidence.

## Disk and Operational Handoff

- Before execution, create a fresh dedicated implementation branch/worktree from current remote `main`, preserving all user WIP. This plan owns no implementation branch itself.
- Tasks 1--12 use generated fixtures and injected synthetic tokenizer distribution/table adapters only; no live `tiktoken`/distribution/table/golden access occurs. Task 13 requires explicit network/source-lock authorization and owns the first live tokenizer validation. Task 14 performs the one final integration JUnit and generated-data readiness evidence without corpus optimization or held-out scoring. Task 15 requires separate real-training authorization after the exact GPU-hour/wall/energy/disk forecast. Task 16 requires explicit acknowledgment of the irreversible one-opening transaction unless that authorization was already stated for the entire frozen experiment.
- If source bytes change, a dependency/table/hash changes, H8 or another predecessor becomes stale, resource headroom fails, or a reviewer finds a source defect, preserve all artifacts, mark the attempt INCONCLUSIVE, and return to the owning task on a new revision. Do not weaken bounds or patch a frozen candidate.
- Implementation should proceed task-by-task with a fresh review after each bounded commit. The final integration candidate gets one full JUnit; generated evidence is never committed.

## Final Self-Review Checklist

- [ ] **Spec coverage:** Exact H8 PASS precondition and nontransfer; candidate-to-verified URL/source/license/distribution/table lock with synthetic-only Tasks 1--12; archive/member SHA/size/CRC/compression/path safety; offline/V3-cache isolation; pinned GPT-2 tokenizer and H6 nontransfer; deterministic windows/masks/counts/shuffle/full eval/final pad/cursor; shared-only experiment profile, exact ordered five-arm/seven-gate inventory, and exact A0 full-causal architecture/formulas/nonmaterialized Flash backend; per-arm H5 labels/objective/prior/recognition/scorer/result roles; H6/H7/H8 identities; explicit autograd/snapshot policy; revision-bound training sparsity plus independent capacity; resume-only rolling versus terminal-scoring checkpoints; safe scientific-artifact resume/migration; every required metric with numerator/denominator/update controls/applicability; H6-refrozen seeds/tuning/stopping/statistics/opening; finalized-input figures/sidecars/captions/alt text and the frozen required-figure registry with inventory-derived panels/series; generic integrity and tested POSIX/Windows durability; atomic runs/manifests/environment/failures/index; click launchers; disk/GPU-hour/wall/energy safeguards; TDD/commits/one JUnit/authorization boundaries; evidence/nonclaims; V3 boundary.
- [ ] **Type consistency:** Every interface name and field used by later tasks is owned in the file/interface map; `WT103ArmSpec`, `WT103GateSpec`, `EndpointInventory`, `WT103ExperimentProfile`, `A0ArchitectureProfile`, `A0FormulaRecord`, `TrainingConfig`, `CandidateTokenizerContract`, `TokenizerSpec`, `WindowManifest`, `PermutationManifest`, `DataCursor`, `TrainingSparsityCertificate`, `PostH8ReadinessToken`, `PriorPredictor`, `CheckpointBundle`, `CheckpointIdentity`, `LoadedCheckpoint`, `MetricRecord`, `NllTotals`, `FigureSpec`, and manifests retain one meaning. No count or derived endpoint key is accepted outside `EndpointInventory.create`.
- [ ] **Data arithmetic:** For every split `counted_targets=n_tokens-1`; final padding contributes zero; no batch mean enters corpus NLL; train/validation/test schedules and capabilities are distinct.
- [ ] **Evidence audit:** H6 byte and H8 synthetic evidence never close GPT-2/training claims; one exact candidate/JUnit, one revision-bound sparsity certificate, independent capacity and resource records, one irreversible test opening, raw/partial endpoint records, manifest integrity, and revision-specific ledgers are required. Partial endpoints force INCONCLUSIVE and are never aggregated.
- [ ] **Figure audit:** No training/checkpoint/data import; explicit run-group input; finalized manifest-validated JSONL and frozen result-table JSON authoritative; CSV regenerated and byte-checked only; deterministic formats/style/specs; explicit applicability; data/caption/alt-text sidecars; all required plots.
- [ ] **Freeze-completeness audit:** Before Task 13, request/final URLs, redirects, archive/member bytes, source/license offsets, installed distribution/table identity, token identities, and derived window/permutation/cadence hashes are intentionally unresolved candidate facts; this plan must not claim that no placeholders remain. Task 13 may set `freeze_completeness=true` only after it validates and closes every one of those facts before outcomes. Any ambiguity stops for an explicit preregistration revision.
- [ ] **Hermetic-tokenizer audit:** Tasks 1--12 contain only the three candidate strings plus injected synthetic adapters; live package/version/RECORD/table/production-golden evidence first appears in Task 13 Step 3 and all synthetic identities are readiness-ineligible.
- [ ] **Checkpoint-role audit:** Every validation-boundary checkpoint is resume-only, every scored endpoint is a post-pass terminal-scoring checkpoint, and evaluator/opening/figure validators reject role substitution.
- [ ] **Path check:** This plan is saved at `docs/superpowers/plans/2026-07-21-vfe4-post-h8-wikitext103-training.md`. This authoring task changes no other file and performs no network access, test, training, or commit.

## Execution Handoff

Plan implementation begins only after the exact H8 artifact and ledger described above are PASS and validated. Execute with `superpowers:subagent-driven-development` for task-by-task review or `superpowers:executing-plans` for controlled inline batches. Stop at the Task 13 source-lock, Task 15 GPU-training, and Task 16 irreversible-opening authorization boundaries; neither ordinary implementation permission nor H8 PASS authorizes a real download, corpus training run, or test opening.
