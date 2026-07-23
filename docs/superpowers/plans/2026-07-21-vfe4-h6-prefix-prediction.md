# VFE 4.0 H6 Prefix Safety and Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the bounded zero-dimensional H6 language sidecar, prove its finite and statically audited prefix-safety contract for every exact predictor configuration, and only then run a separately revision-bound, compute-matched WikiText-2 prediction experiment.

**Architecture:** H6 is two evidence products, not one gate. `H6-Prefix` verifies the target-blind public prior-predictor boundary, source support, cache behavior, and static dataflow at an exact source/config/estimator identity; `H6-Prediction` consumes those immutable certificates and separately evaluates six matched arms with a frozen target-blind filter/SMC scorer. The language specialization remains a labeled causal population over the singleton base `C0={*}`; the internal causal DAG is a probability graph and never a surrogate bundle base.

**Tech Stack:** Python 3.10+, PyTorch, deterministic CPU float64 for H6-Prefix and estimator validation, byte-level WikiText-2 caches, NumPy independent oracles, SciPy-free frozen critical values, pytest, SHA-256 provenance, atomic JSON/checkpoint publication, JUnit XML.

## Global Constraints

- The normative theory is `Manuscripts/vfe4_whitepaper/02_observations_related_work.tex`, `04_generative_model.tex`, `05_structured_information_form.tex`, `07_transformer_crosswalk.tex`, `08_hypotheses_limitations.tex`, and `09_appendices.tex`. The corresponding Research WIP files were byte-identical when this plan was authored. The relevant wiki context is `[[VFE Transformer Program]]`, `[[Inference machinery -- variational EM and filtering]]`, and `[[vfe-population-generative-status-2026-07-12]]`; V3 supplies rough launcher/data/artifact mechanics only.
- The implemented geometric base is exactly the singleton `C0={*}`. Token positions are labeled population copies over that point. `CausalDag` is stored separately from `ZeroDimensionalBase`; neither a token edge nor a cache transition is base transport, base curvature, or base holonomy.
- H6 does not establish H7. At the singleton base, A5 uses shared vertex-coboundary internal maps with `U_t=exp(Phi_t)`, `Omega_tj=U_t U_j^-1`, and a full same-receiver `B_t`; A2 replaces only each `Omega_tj` with independent dense fixed-frame edge maps `A^z_tj` and `A^m_tj` and retains the same `B_t`. This is a matched shared-vertex-coboundary versus generic-fixed-frame/non-coboundary model contrast, not evidence of frame covariance, connection transport, curvature, or holonomy. H7 and every covariance label remain unimplemented/unverified until the independent H7 plan passes.
- `H6-Prefix` and `H6-Prediction` have different schemas, artifact roots, evidence revision identifiers, claim ledgers, and closure decisions. Prefix runs and publishes solely from its own exact source/config/model-family/vocabulary/estimator/data-safety identities. H1--H5 status, predecessor publication, estimator-accuracy evidence, the H6 training schedule, arm matching, tuning, checkpoints, and predictive outcomes are not Prefix inputs, preflight conditions, status terms, publication conditions, or ledger claims.
- Prediction cannot be reported, launched, resumed, or scored unless every exact arm/config/estimator/model-family/vocabulary tuple it consumes has a PASS prefix certificate. A missing, stale, FAIL, or INCONCLUSIVE certificate blocks that tuple and makes the aggregate prediction result INCONCLUSIVE. Checkpoint hashes are bound separately in empirical provenance and may not alter the certified predictor safety contract.
- Task 11 completes the tracked H6 source surface with focused RED/GREEN commands only. Task 12 performs a source-build closeout using focused deterministic fixtures and records the deferred evidence operations; it does not run a broad suite, H4 timing benchmark, large estimator grid, corpus training, test opening, or `.verification/` lifecycle. The full Prefix and Prediction evidence revisions are later, separately authorized operations at a frozen `(git_head, dirty_digest)` produced by `vfe4.artifacts.provenance.dirty_content_digest`.
- Development has a hard resource boundary. Every focused RED or GREEN command uses synthetic, deterministic, no-download, no-training fixtures and must finish in less than 10 seconds on CPU; a command that reaches 10 seconds is stopped and narrowed before any retry. Do not start background workers, corpus acquisition, training, a broad/full suite, the H4 timing benchmark, the 9,720/4,096 Prefix inventories, the 76-cell/512-replicate SMC grid, or endpoint scoring during Tasks 1--12. The large H4/H6 operations exist only behind editable click-to-run dictionaries in `verify_vfe4.py` or `train_vfe4.py`, are `False` by default, execute only inside `main(CONFIG)` after an explicit operation-specific authorization field is present, and are unreachable from package imports, launcher imports, and ordinary pytest collection/execution. No CLI is added.
- H6-Prefix publication preflight validates only the exact H6 source/config/model-family/vocabulary/estimator/data-safety identities and complete Prefix case/static-audit inventory for that evidence revision. It neither reads nor references an H1--H5 artifact, and it cannot be blocked by any H1--H5 status or publication state.
- Before any empirical Prediction split materialization/access, tuning, training, validation scoring, checkpointing, or test scoring, Prediction readiness may require exact current H1, H2, H3, and H5 correctness artifacts plus the exact H1-prefix-prior, finite-SMC, and H6-Prefix inputs used by the selected Prediction matrix. H4 correctness/timing/cost evidence is not a Prediction prerequisite; the frozen green H4 correctness provenance (`911` tests, zero failures/errors/skips) may be referenced as nonblocking history, while its deferred timing benchmark must never be triggered by H6.
- Prefix-conditioned-prior variants consume a separate current-candidate H1 rerun artifact keyed to the exact prefix-prior generative-factor/config schema. The bounded SMC recursion gate likewise has a separate current-candidate artifact. These full evidence artifacts are produced only by the deferred Prediction-evidence operation, never by source buildout. The H1 variant does not replace or mutate ordinary H1/H2/H3/H5 correctness evidence, and fixed-prior variants do not require it.
- H6 owns the complete immutable `H6TrainingSchedule`, including AdamW class, optimizer policy, and phase names/order. H5 recognizes only `exact_coordinate`, `generalized_em`, and `natural_gradient_proposal`; those labels and the exact H5 producer fields are correctness provenance for Prediction readiness, never names or certifications of H6 Adam/AdamW phases, schedule composition, repetition count, optimizer behavior, or monotonicity.
- H6 uses separate artifacts. An H6-Prefix artifact contains no predecessor reference and publishes only H6 identities, validation, certificates, and manifest. Deferred Prediction readiness references exact current H1/H2/H3/H5, H1-prefix-prior, finite-SMC, and H6-Prefix evidence without copying payloads. No unified H1--H6 validation payload is created.
- The bounded corpus is the official WikiText-2 raw archive at exactly `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip`. Public configuration cannot select another URL or opener. Synthetic tests inject a byte-stream opener only through the non-exported internal `_acquire_wikitext2_blinded(config, opener)` seam; public `acquire_wikitext2_blinded(config)` always uses the exact URL. The exact downloaded archive bytes, three streamed raw members, tokenizer specification, encoded streams, and window manifests are SHA-256 bound. Training never substitutes WikiText-103, a prepared vocabulary, synthetic text, or another mirror.
- Archive preparation accepts at most `16,777,216` compressed bytes; exactly one directory entry and the three files `wikitext-2-raw/wiki.train.raw`, `wikitext-2-raw/wiki.valid.raw`, and `wikitext-2-raw/wiki.test.raw`; compression methods ZIP_STORED (`0`) or ZIP_DEFLATED (`8`) only; positive per-member compressed/uncompressed sizes at most `16,777,216`; total uncompressed bytes at most `33,554,432`; and compression ratio at most `100`. It rejects encryption, links, duplicate/case-colliding paths, extra files, path traversal, CRC mismatch, size mismatch, and decompression beyond a bound. The observed archive/member byte sizes, compression methods, CRC32 values, and SHA-256 values are copied into config/preregistration before evidence and must match during streaming extraction.
- Data access has one capability boundary. Before Prediction readiness, acquisition may bounded-stream-validate, hash, and seal all three raw members and materialize only the frozen validation safety fixture. It returns hashes/metadata and opaque sealed handles, never train tensors/windows, ordinary validation tensors/windows, or model-facing test bytes. After readiness PASS, `materialize_prediction_train` may decode/materialize train/validation data. Only after the durable `O_EXCL` reservation may `open_test_for_scoring` map/decode test bytes. `DurableTestOpeningCapability` is opaque and privately constructed only by the sole reservation issuer after the reservation bytes are durable. Its canonical proof binds readiness, experiment, data, sealed test, access policy, reservation path/state, and proof digest. The store's private validator independently retains the exact registered proof/identities and re-reads the immutable reservation bytes; forged strings or proof bytes fail before mapping. Training, tuning, and analysis cannot import the private unsealer or issuer.
- The H6 tokenizer is fixed: raw UTF-8 bytes map to IDs `0..255`, `BOS=256`, `EOS=257`, vocabulary size `258`, and ignored target padding `-100`. Each split is encoded independently as `[BOS] + exact_raw_member_bytes + [EOS]`; bytes and newline sequences are not normalized. Encoded-token identity is `SHA256(b"VFE4-H6-U16LE-TOKENS-V1\x00" || uint64_le(token_count) || concat(uint16_le(token_id)))`; every ID is range-checked `0..257`, and the exact preimage bytes are streamed in that order. Native-endian tensor/storage bytes are never serialized or hashed. No learned state crosses splits.
- Sequence length and stride are both exactly `32`. A window uses `inputs=tokens[start:start+32]` and `targets=tokens[start+1:start+33]`; the last partial window is included once, fills unused inputs with `BOS`, and fills unused targets with `-100`. Token counts always count targets not equal to `-100`.
- `validation_safety_fixture.bin` contains exactly 4,096 distinct validation-only stride-32 windows and fails closed if fewer exist. Rank zero-based candidate window index `i` by ascending bytes of `SHA256(b"VFE4-H6-VALIDATION-SAFETY-RANK-V1\x00" || validation_token_sha256_raw32 || uint64_le(i))`, breaking a digest tie by ascending `i`; select the first 4,096. Serialize `b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00" || validation_token_sha256_raw32 || uint32_le(4096)` followed in rank order by `uint64_le(start) || uint16_le(real_target_count) || 33*uint16_le(encoded_id)`. Each 33-ID row is the exact causal token slice padded with `BOS`; `real_target_count` deterministically restores target `-100` masking. The fixture digest depends only on validation token content/identity and this policy, never train, test, archive, or aggregate-data identity. It is the sole pre-readiness validation material; Task 9 derives perturbations from it with seed `2026072197`.
- Blinded binary publication uses a narrow whole-directory writer, not JSON-only `publish_run_directory`. The caller supplies exactly five payloads in this canonical order: `sealed/wiki.train.raw`, `sealed/wiki.valid.raw`, `sealed/wiki.test.raw`, `validation_safety_fixture.bin`, `data_identity.json`. `data_identity.json` contains no enclosing `manifest_sha256`, manifest path, or directory-manifest identity. After bounded streaming, the publisher itself computes each raw payload length/hash and generates `manifest.sha256`; callers cannot supply it. Its digest is `SHA256(b"VFE4-H6-BINARY-DIRECTORY-MANIFEST-V1\x00" || uint32_le(5) || concat(uint16_le(path_utf8_length) || path_utf8 || uint64_le(raw_length) || raw_content_sha256_raw32))` over that exact order; `manifest.sha256` contains the 64 lowercase ASCII hex digest plus LF and is excluded from its own preimage. Use a same-volume owned stage, create-new/O_EXCL files, flush/fsync where supported, close every handle, and install with an OS no-replace primitive. Existing destinations and unsupported no-replace platforms fail closed; never call `ZipFile.extract`/`extractall`, and clean up only the owned stage.
- WikiText-103 and the GPT-2 tokenizer are reserved until after H8. They do not appear as supported H6 configuration values, fallback paths, tests presented as H6 evidence, or secondary experiment arms.
- Training uses smoothing recognition as the primary regime and filtering recognition as a required ablation. Held-out validation/test scoring uses only the causal generative prior predictor. Recognition may consume the current target or complete observed window during training, but no recognition object, activation, parameter, target, or suffix may enter the prior predictor.
- The public bound call is exactly `next_token_log_probs(prefix_tokens: CausalPrefix, estimator_rng: EstimatorStream, cache: PrefixCache | None = None) -> PriorPrediction`. Its bound signature contains those three parameters in that order and rejects raw tensors. It has no target, suffix, full-window, recognition, posterior, or reconstruction parameter. `PriorPrediction.log_probs` has shape `(V,)`, where `V` comes from an immutable `VocabularyIdentity` included in the prefix, predictor, cache, prefix key, and artifact. WikiText-2 uses `V=258`; no generic interface hardcodes 258.
- Every source normalization in both state and model banks calls the single shared `masked_log_softmax_from_parents(logits, declared_parents, receiver_t)`. It derives the Boolean mask only from declared parents satisfying `j<t`, writes exact `-inf` before normalization, rejects an empty/all-invalid row with `AllInvalidSourceRowError`, and returns exact zero mass outside support. Post-softmax masking, renormalization in another helper, direct `softmax`/`log_softmax` in a source-prior module, or unresolved/dynamic dispatch is respectively FAIL or INCONCLUSIVE; no alternate normalization path exists.
- `T_mask` identities are sorted `MaskCaseKey(fixture_sha256, vocabulary_sha256, predictor_config_sha256, model_family_sha256, prior_variant, bank, receiver_t, context_sha256)`. The separate production-path `h6-prefix-small-v1` fixture has vocabulary 3, horizon 4, and parent rows `((0,), (0,1), (0,1,2), (0,1,2,3))` for both banks. Its fixed-prior manifest has exactly 4 contexts per bank; its prefix-conditioned manifest has exactly `2*(1+3+9+27)=80` contexts per bank (all token prefixes times the two frozen latent-history contexts `zero` and `seeded`). Across both banks/variants the small base inventory is `4+4+80+80=168` mask cases.
- The WikiText-2 property manifest has exactly 4,096 contexts for each active `(prior_variant, bank)` cell: fixed/state 4,096, fixed/model 4,096, prefix/state 4,096, prefix/model 4,096, for a two-bank/two-variant base inventory of 16,384. A model family with no bank has no fake mask row; A4 contributes only its state bank; A0/A3 contribute none. The gate records exact per-configuration counts and the sorted manifest SHA.
- H6-Prefix uses deterministic CPU float64. The leakage allowance is exactly `0`, the mask allowance is exactly `0`, and equality is stricter than tensor-value equality: compare dtype, shape, device, contiguous raw storage bytes viewed as `uint8`, and SHA-256. This preserves signed-zero distinctions. Metadata or raw-byte differences FAIL even if numeric comparison would call `+0.0` and `-0.0` equal.
- The exhaustive leakage fixture is the separate `h6-prefix-small-v1` production model path with `V=3`, `T=4`; it is not a size-mutated WikiText-2 identity. For each one-indexed position `t`, enumerate every prefix in `V^(t-1)` and every ordered pair of tails in `V^(5-t) x V^(5-t)`, including equal tails. The exact count is `sum_t 3^(t-1) * 3^(2*(5-t)) = 9,720` comparisons per certified model-family/estimator profile.
- The language-property fixture contains exactly `4,096` WikiText-2 validation perturbations at `V=258`, generated from identity-bound validation windows with seed `2026072197`. Each pair preserves `x_<t` and varies current targets/suffixes independently. The case file records generator version, `VocabularyIdentity`, and exact validation-token/window hashes.
- Every dynamic prefix case runs cold-cache, warm-cache, and reverse-order/cache-rebuild modes. Compared calls use common counter-based estimator streams. Cache results must equal cold recomputation exactly and must be independent of case traversal order.
- H6-Prefix also requires a static import/signature audit, a taint/dataflow audit from targets/suffixes/recognition to generative sinks, and a cache-key/value audit. A universal source claim is never inferred from the finite cases alone.
- H6-Prefix status is PASS only when every exact case, source-mask check, cache check, signature/import rule, taint obligation, and artifact identity passes. A witnessed violation is FAIL. Missing fixtures, unsupported dynamic dispatch, incomplete taint coverage, absent hashes, or an unauditable cache is INCONCLUSIVE.
- Prediction arms are fixed. A0 is a conventional normalized autoregressive baseline with no latent, source, map, or recognition sector. A1 is one ordinary Gaussian state chain with no categorical sources, internal maps, or model channel. A2 is identical to A5 in both channels, categorical state/model source banks, fixed source priors, exact source mixture, recognition family/conditioning, objective, and full same-receiver `B_t`; only the shared vertex-coboundary `Omega_tj=U_t U_j^-1` maps are replaced by independent dense fixed-frame/non-coboundary `A^z_tj,A^m_tj`. A3 is a typed dual-channel immediate-predecessor model with no categorical source variables. A4 is a typed shared-vertex-coboundary state-only model with one categorical state-source bank and no model channel, `B_t`, or model-source bank. A5 is the full dual-channel, dual-source-bank shared vertex-coboundary H6 model.
- All six arms consume identical encoded tokens, batch/window order, pass count, model-update opportunities, validation boundaries, checkpoint boundaries, and test-opening transaction. Their trainable parameter counts are within `1%` of the A5 reference, their counted whole-schedule training FLOPs are within `5%`, every active trainable parameter is present exactly once in its declared optimizer, and no arm uses dormant/no-op/filler parameters or phases.
- `H6TrainingSchedule` is one hashed common outer schedule plus a hashed typed phase schedule for every exact endpoint. Its `AdamWPolicyRecord` fixes `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`, `maximize=False`, `foreach=False`, `capturable=False`, `differentiable=False`, `fused=False`, `zero_grad(set_to_none=True)`, all-active-parameter decay, and an always-evaluated L2 global-gradient scale with `max_norm=1.0`; only learning rate and weight decay vary over the frozen tuning grid. A0 and every `latent_enabled=false` endpoint use only `model_ce_adamw`; they construct no recognition parameter store, law, optimizer, or fake recognition step. Every latent endpoint owns a trainable recognition parameter store and uses exactly `recognition_adamw -> immutable_detached_snapshot -> model_adamw` once per batch. Ephemeral `StructuredLanguageRecognition`/`FactorizedLanguageRecognition` records are normalized laws emitted from that store, never parameter owners. FLOP matching counts the actual active phases, so extra latent inference/update work is matched structurally rather than hidden by no-ops. H5's enabled labels remain exactly `exact_coordinate`, `generalized_em`, and `natural_gradient_proposal`; none is renamed to Adam or used to certify the H6 schedule or its monotonicity.
- A5's frozen reference capacity is `(emission_width=64, latent_width=16, recognition_width=64)`. Every latent endpoint searches, in field order `(emission_width, latent_width, recognition_width)`, only the Cartesian product of the literal values `emission_width=(48,64,80,96)`, `latent_width=(8,16,24,32)`, and `recognition_width=(32,64,96)`, for at most 48 formula-only candidates. A0 and every no-latent endpoint search exactly the four `emission_width` candidates with `latent_width=None` and `recognition_width=None`. Every present allocation field must determine a live tensor shape and live forward/training computation; an inapplicable field is `None`, and a filler, dormant, identity-only, or no-op use is forbidden. The first lexicographically eligible allocation is frozen without reading data, loss, gradients, or metrics.
- Capacity/compute matching covers active whole-schedule **training arithmetic only**. Prediction/scoring FLOPs depend on the target-blind predictor, prefix, estimator, particle count, and cache protocol; they are computed and reported in a separate prediction ledger and never enter the 5% training-FLOP eligibility test.
- Capacity allocation is a declared outcome-blind nuisance adjustment. For a component row, the resolved semantic configs after deleting only `capacity_allocation` must differ in exactly the row's named factor; the raw configs may additionally differ in the mechanically selected allocation. Any other difference invalidates the row. Matching therefore does not turn PRIOR, MIXTURE, OBJECTIVE, LATENT, or other descriptive rows into causal claims, and a row with no eligible literal allocation is INCONCLUSIVE rather than granted a tolerance or filler exception.
- Required factorial reports are structured versus population-factorized recognition, fixed versus prefix-conditioned generative source prior, exact source mixture versus the declared projection, complete-ELBO versus emission-only training, latent enabled versus disabled, and smoothing versus filtering training. Each comparison changes only its named factor on the same arm factory/config family. A1, A3, A4, and A5 have different live generative factors, so downstream objective/training work must define and hash a family-specific ELBO factor/term inventory for each latent family; the full A5 `1+6T` inventory may not be applied unchanged to a family with absent factors.
- Prefix-conditioned source priors are a new normalized generative model. Prediction readiness requires the separate exact H1 rerun PASS artifact for those variants. That prerequisite is not part of Prefix safety closure. Emission-only is labeled an ablation, not another ELBO. Projection is labeled approximation and records projection error; it is never called exact mixture marginalization.
- Before empirical scoring, freeze and validate the weighted bootstrap filter/SMC estimator specified below: 256 particles for the bounded finite gate, carried normalized float64 log weights, systematic resampling after observation only when ESS is below `0.5 * particle_count`, `logsumexp` normalization, and counter-based streams. The proposal is exactly the causal generative source/transition law, so no omitted proposal correction exists. An unweighted emission average is forbidden whenever carried weights are nonuniform. The finite `V=3,T=6` gate validates recursion only; it cannot close actual WikiText-2 checkpoint estimator error.
- Tuning is the equal grid `learning_rate in {1e-4, 3e-4, 1e-3}` by `weight_decay in {0, 1e-2}` for every arm, using exactly two quarter-pass runs per cell. The specified tuning/train seed is `2026072199`; because the source protocol fixed two tuning seeds but named only one, freeze the adjacent independent companion `2026072200` in the preregistration before any tuning. This explicit resolution is not evidence from outcomes.
- Confirmatory initialization/run seeds are exactly `2026072101..2026072108`. The shared data-order seed is `2026072199`. Actual test scoring uses the frozen 64-entry common paired stream registry derived from root `2026072198`, never one selected estimator stream. No replacement seed or adaptive replicate is permitted.
- Batch size is exactly `8`, with no drop-last. Training order for zero-based pass index `p` is a versioned Fisher-Yates permutation of window indices keyed by shared seed `2026072199`, independent of Python/NumPy/Torch RNGs. For draw-block counter `c=0,1,...`, compute `SHA256(b"VFE4-H6-BATCH-PERMUTATION-DRAW-V1\x00" || uint64_le(2026072199) || uint64_le(p) || uint64_le(c))`, consume its four unsigned little-endian 64-bit words in byte-offset order `0,8,16,24`, then advance `c`. For Fisher-Yates `i=n-1,...,1`, let `m=i+1`, `limit=2^64-(2^64 mod m)`, reject words `x>=limit`, otherwise set `j=x mod m` and swap positions `i,j`. The schedule preimage is `b"VFE4-H6-FROZEN-BATCH-SCHEDULE-V1\x00" || uint64_le(seed) || uint64_le(p) || uint64_le(n) || uint16_le(8) || uint8(0) || concat(uint64_le(permuted_window_index))`; its SHA-256 is the schedule digest. Consecutive groups of eight form batches and the final short batch is retained. A full pass visits every window exactly once; a quarter pass is the first `ceil(number_of_batches/4)` complete-or-final-short batches. Evaluation order is sequential window index `0..n-1`, batched by eight with the final short batch retained.
- The phrase “validation every twentieth pass” is operationalized as validation at every twentieth of a corpus pass: boundaries `ceil(k * batches_per_pass / 20)` for `k=1..20`, deduplicated while preserving order, on each of two passes. This is the only reading compatible with both “two full passes” and periodic validation; it is frozen before outcomes.
- The test split is opened once, globally, after tuning choices, all eight-seed terminal checkpoints, prefix certificates, analysis code hashes, and the complete actual-endpoint SMC protocol are frozen. Blinded acquisition/hash and sealed storage are not a model-facing opening. The irreversible opening begins only after the durable `O_EXCL` reservation and unsealing capability are recorded; that one transaction scores every endpoint/checkpoint across the complete 64-stream, four-particle-count assessment or scores none. Validation does not choose early checkpoints.
- An infrastructure failure may receive one exact retry only when the attempt artifact proves no optimizer/checkpoint state advanced or proves an exact checkpoint restore. Numerical divergence, nonfinite loss, estimator failure, model failure, prefix failure, capacity mismatch, or a missing pair is not infrastructure and receives no replacement run. Any incomplete paired seed set makes the affected decision INCONCLUSIVE.
- The primary metric is the actual-endpoint SMC-qualified corpus-summed, token-counted prior negative log likelihood in nats/token defined below. Per-batch means are not averaged. Perplexity is `exp(NLL)` and is secondary.
- The primary paired contrast is `d_i = NLL_A0,i - NLL_A5,i` over the eight confirmatory seeds. The practical threshold is `delta = -log(0.99) = 0.01005033585350145`. The training-seed interval uses the frozen `t_(0.975,7)=2.364624251592784`, then is conservatively enveloped over all estimator-error boxes as defined below. Lower bound greater than `delta` is PASS; upper bound less than or equal to `0` is FAIL; every other complete result is INCONCLUSIVE.
- Report the paired `NLL_A2 - NLL_A5` interval separately. Attribution to the shared-vertex-coboundary versus generic-fixed-frame/non-coboundary map parameterization is permitted only if the primary A0--A5 result passes and this secondary interval has lower bound greater than zero. It is not an H7 covariance, connection, curvature, or holonomy inference.
- Deferred evidence artifacts are atomic and identity-bound. Independent Prefix validation and Prediction-only H1/H2/H3/H5, H1-prefix-prior, finite-SMC, readiness, train materialization, tuning, typed-phase attempts, checkpoints, validation scoring, immutable test reservation, endpoint records, uncertainty aggregation, and final metrics each have immutable manifests with `git_head`/`dirty_digest` plus applicable config/data/access/estimator/RNG/parent hashes.
- Each implementation task runs only its named focused RED/GREEN commands on deterministic shrunken fixtures. Do not run cumulative/broad tests, gates, training, timing, or test opening during source buildout. Exact-revision Prefix/Prediction artifacts and ledgers are created only under the separate Task 13--14 authorization; a later source change invalidates affected evidence and requires a new evidence revision, never an in-place artifact patch.
- Source reviewers consume focused outputs only. Separately authorized evidence reviewers consume exact-revision artifacts, manifests, and claim ledgers without rerunning training, scoring, the H4 benchmark, or other evidence workloads for confidence.
- Preserve `.verification/ledger.json` and every prior revision-specific ledger byte-for-byte. H6-Prefix uses `.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json`. H6-Prediction uses `.verification/h6-prediction-<FULL_HEAD>-<EXPERIMENT_SHA>-ledger.json`. Never overwrite or repoint an existing ledger; a replacement revision gets a new path.

---

## File Map and Dependency Boundaries

| Path | Responsibility |
|---|---|
| `vfe4/types/h6.py` | Immutable, domain-separated hashed `ZeroDimensionalBase`, `CausalDagRow`/`CausalDag`, H6 language structure, `FrozenTensorSnapshot`, arm, data identity, predictor/cache, complete prefix certificate, estimator, ELBO, NLL, checkpoint, attempt, and decision records. |
| `vfe4/config/schema.py` | Conditional frozen H6 Prefix/Prediction sections, H6-owned AdamW `H6TrainingSchedule`, Prediction-only H1/H2/H3/H5 references, estimator/matrix config, and closed literals. |
| `vfe4/config/resolve.py` | Existing resolver extended with H6-Prefix and Prediction operations, exact own Prefix identities, Prediction-only prerequisites, conditional H6 sections, and one canonical config/hash. |
| `vfe4/types/results.py` | Existing explicit result union extended with fail-closed `H6PrefixGateResult` and `H6PredictionResult` without weakening earlier result types. |
| `vfe4/data/byte_tokenizer.py` | Stateless byte tokenizer with IDs `0..257`; no corpus fitting. |
| `vfe4/data/wikitext2.py` | Official archive retrieval/extraction, member allowlist, blinded hashing, sealed split storage, and identity-bound materialization. |
| `vfe4/data/access.py` | Capability boundary for blinded acquisition, post-readiness train materialization, and post-reservation test unsealing. |
| `vfe4/data/windows.py` | Exact stride-32 causal windows, ignored-target counting, immutable window and batch schedules. |
| `vfe4/numerics/critical_values.py` | Frozen SciPy-free Student-t/chi-square critical constants and protocol identities; no runtime quantile solver. |
| `vfe4/generative/source_priors.py` | Fixed and prefix-conditioned normalized causal source priors with pre-normalization masks. |
| `vfe4/generative/language.py` | Normalized initial, model/state transition, and categorical emission factors for the singleton-base language model. |
| `vfe4/recognition/language.py` | Normalized structured/factorized filtering and smoothing recognition families used only for training/inference. |
| `vfe4/objective/language_elbo.py` | Horizon-indexed `H6LanguageElboTerms` assembly with exact factor identities/partition equality plus a separately typed non-ELBO emission-only ablation. |
| `vfe4/predictive/cache.py` | Immutable prefix cache and exact cache identity/key validation. |
| `vfe4/predictive/smc.py` | Frozen weighted bootstrap filter/SMC recursion, normalizing-constant estimator, cache state, and counter-based stream. |
| `vfe4/predictive/prior.py` | `PriorPredictor` and the only public target-blind `next_token_log_probs` boundary. |
| `vfe4/recognition/parameter_store.py` | Trainable structured/factorized recognition parameter stores that emit normalized Task 4 recognition laws; predictor code cannot import them. |
| `vfe4/training/arms.py` | Six explicit arm factories; no registry/signature dispatch. |
| `vfe4/training/matching.py` | Active parameter, optimizer-access, and counted-FLOP audits with hard 1%/5% gates. |
| `vfe4/training/checkpoint.py` | Atomic exact-resume checkpoint schema including horizon-indexed ELBO terms, RNG/data cursor, prefix certificate references, and checkpoint hashes. |
| `vfe4/training/language.py` | Shared H6-owned AdamW schedule training step/pass engine, typed language-ELBO metrics, and failure classification. |
| `vfe4/training/h6_readiness.py` | Fail-closed Prediction-readiness validator and opaque PASS-token issuer; it owns no Prefix closure claim. |
| `vfe4/training/h6_experiment.py` | Equal-grid tuning, fixed confirmatory schedule, infrastructure-only retry, and one-test-opening orchestration. |
| `vfe4/evaluation/prior_nll.py` | Target-blind corpus NLL scorer and paired t decisions. |
| `vfe4/evaluation/smc_uncertainty.py` | Actual-checkpoint 64-stream/four-particle-level aggregation, convergence/bias envelopes, simultaneous bounds, and conservative interval inflation. |
| `vfe4/validation/h6_prefix.py` | Dynamic leakage/mask/cache case execution and exact PASS/FAIL/INCONCLUSIVE precedence. |
| `vfe4/validation/h6_static_audit.py` | AST import/signature, taint/dataflow, and cache audits. |
| `verification/numpy_oracles/h6_prefix.py` | Independent exhaustive enumeration and expected count/reference comparisons. |
| `verification/h1_prefix_prior_gate.py` | New H1 normalization/ELBO audit for prefix-conditioned source priors. |
| `verification/h6_smc_gate.py` | Independent small-model estimator normalization, reproducibility, cache, and error-analysis gate. |
| `verification/h6_prefix_gate.py` | Atomic H6-Prefix gate and certificate-set publisher. |
| `vfe4/artifacts/h6.py` | H6-specific atomic experiment/checkpoint/metrics/failure/test-opening writers. |
| `vfe4/artifacts/provenance.py` | Extend provenance with data, estimator, prefix certificate, checkpoint, and parent evidence revisions. |
| `verify_vfe4.py` | Extend the one-editable-dictionary/one-main/no-required-CLI verifier through independent H6-Prefix and own the pure H1-prefix-prior/H6-Prefix projections plus generic current-candidate runner intended for the synchronized H7 consumer. |
| `train_vfe4.py` | New click-to-run editable prediction dictionary; no required CLI. |
| `docs/preregistrations/2026-07-21-h6-prefix-prediction.md` | Frozen source resolutions, hashes, arm matrix, estimator, schedules, statuses, decisions, nonclaims, and artifact schemas. |
| `tests/unit/test_h6_*.py` | H6 types/config/data/model/recognition/predictor/matcher/checkpoint/statistics unit contracts. |
| `tests/oracle/test_h1_prefix_prior.py` | Independent new-H1 prefix-prior comparisons. |
| `tests/oracle/test_h6_smc_oracle.py` | SMC versus exact bounded model and estimator-stream tests. |
| `tests/unit/test_h6_critical_values.py` | Exact frozen constant/hash/CDF-certificate assumptions without importing SciPy. |
| `tests/unit/test_h6_smc_uncertainty.py` | Hand-table tests for endpoint aggregation, common-stream covariance, convergence, bounds, and 256-corner interval inflation. |
| `tests/property/test_h6_prefix.py` | 9,720 exhaustive and 4,096 frozen perturbation properties. |
| `tests/promotion/test_h6_prefix_gate.py` | Prefix gate status, blocking, stale-certificate, and artifact tests. |
| `tests/integration/test_h6_language.py` | Six-arm tiny train/evaluate/resume and one-opening integration. |
| `tests/integration/test_train_vfe4.py` | Imports and shrinks the live editable dictionary without adding CLI requirements. |
| `tests/unit/test_config.py` | Existing ordered-prefix/conditional-H6/canonical-hash and predecessor fail-closed coverage. |
| `tests/unit/test_atomic_artifacts.py` | Existing atomic writer tests extended for separate H6 references and exclusive durable test opening. |
| `tests/integration/test_verify_vfe4.py` | Existing click-run integration extended through independent H6-Prefix plus the three H6-owned lifecycle adapters, with no Prefix predecessor read/rerun/copy; synchronized H7-consumer compatibility is a separate focused contract test. |

Dependency direction is `config + types -> data/numerics -> generative -> recognition/objective -> predictive/inference -> training/evaluation -> launchers/artifacts`. Production never imports `verification/` or `tests/`. `predictive/` and `generative/` never import `recognition/`, `objective/`, `training/`, or a target-bearing batch type. V3 code is not imported.

## Public Interfaces Frozen by This Plan

```python
@dataclass(frozen=True)
class VocabularyIdentity:
    vocabulary_id: str
    size: int
    tokenizer_spec_sha256: str

@dataclass(frozen=True)
class ZeroDimensionalBase:
    base_id: Literal["C0"]
    points: tuple[Literal["*"],]
    dimension: Literal[0]
    canonical_sha256: str

@dataclass(frozen=True)
class CausalDagRow:
    receiver_t: int
    parents: tuple[int, ...]

@dataclass(frozen=True)
class CausalDag:
    labeling: Literal["zero_based"]
    node_labels: tuple[int, ...]
    rows: tuple[CausalDagRow, ...]
    canonical_sha256: str

@dataclass(frozen=True)
class H6LanguageStructure:
    base: ZeroDimensionalBase
    dag: CausalDag
    receiver_labels: tuple[int, ...]
    structure_sha256: str

@dataclass(frozen=True, init=False)
class FrozenTensorSnapshot:
    __owned: torch.Tensor
    dtype: str
    shape: tuple[int, ...]
    device: str
    contiguous: bool
    requires_grad: bool
    storage_version: int
    raw_bytes_sha256: str

    @classmethod
    def capture(cls, value: torch.Tensor) -> "FrozenTensorSnapshot": ...
    def value(self) -> torch.Tensor: ...  # fresh clone; autograd path preserved
    def assert_intact(self) -> None: ...

@dataclass(frozen=True)
class H6FactorTerm:
    receiver_t: int
    partition: Literal[
        "emission", "initial", "state_source", "model_source",
        "state_transition", "model_transition", "entropy"
    ]
    factor_identity_sha256: str
    value: FrozenTensorSnapshot

@dataclass(frozen=True)
class H6LanguageElboTerms:
    horizon: int
    ordered_factor_terms: tuple[H6FactorTerm, ...]
    emission_terms: tuple[H6FactorTerm, ...]
    initial_terms: tuple[H6FactorTerm, ...]
    state_source_terms: tuple[H6FactorTerm, ...]
    model_source_terms: tuple[H6FactorTerm, ...]
    state_transition_terms: tuple[H6FactorTerm, ...]
    model_transition_terms: tuple[H6FactorTerm, ...]
    entropy_terms: tuple[H6FactorTerm, ...]
    complete_decomposition: FrozenTensorSnapshot
    total_language_elbo: FrozenTensorSnapshot
    equality_checked: Literal[True]
    canonical_sha256: str

@dataclass(frozen=True)
class EmissionOnlyAblationTerms:
    objective_kind: Literal["emission_only_ablation_non_elbo"]
    ordered_emission_terms: tuple[H6FactorTerm, ...]
    total: FrozenTensorSnapshot
    canonical_sha256: str

class PriorPredictor(Protocol):
    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction: ...

@dataclass(frozen=True)
class PriorPrediction:
    vocabulary: VocabularyIdentity
    log_probs: FrozenTensorSnapshot  # shape (vocabulary.size,)
    cache: PrefixCache
    estimator_record: EstimatorRecord

class LanguageRecognitionParameterStore(nn.Module):
    """Owns trainable parameters and emits an ephemeral normalized law."""
    def recognition_law(
        self,
        conditioning: RecognitionConditioning,
    ) -> StructuredLanguageRecognition | FactorizedLanguageRecognition: ...

@dataclass(frozen=True)
class CapacityAllocation:
    emission_width: Literal[48, 64, 80, 96]
    latent_width: Literal[8, 16, 24, 32] | None
    recognition_width: Literal[32, 64, 96] | None
    allocation_sha256: str

@dataclass(frozen=True)
class ArmConfig:
    arm: ArmId
    config_id: str
    latent_enabled: bool
    state_channel_enabled: bool
    model_channel_enabled: bool
    source_mode: Literal["absent", "immediate_predecessor", "categorical"]
    map_mode: Literal[
        "absent",
        "generic_fixed_frame_non_coboundary",
        "shared_vertex_coboundary",
    ]
    recognition_family: Literal["absent", "structured", "factorized"]
    recognition_conditioning: Literal["absent", "filtering", "smoothing"]
    prior_variant: Literal["absent", "fixed", "prefix_conditioned"]
    mixture_mode: Literal["absent", "exact", "moment_projection"]
    objective_kind: Literal["cross_entropy", "complete_elbo", "emission_only_ablation"]
    capacity_allocation: CapacityAllocation
    config_sha256: str

@dataclass(frozen=True)
class ParameterRoleRecord:
    qualified_name: str
    parameter_id: int
    role: str
    phase: Literal["model_ce_adamw", "recognition_adamw", "model_adamw"]
    scalar_count: int
    trainable: Literal[True]
    record_sha256: str

@dataclass(frozen=True)
class OptimizerBinding:
    phase: Literal["model_ce_adamw", "recognition_adamw", "model_adamw"]
    optimizer_class: Literal["AdamW"]
    optimizer_policy_sha256: str
    parameter_ids: tuple[int, ...]
    binding_sha256: str

@dataclass(frozen=True)
class FlopTerm:
    phase: Literal["model_ce_adamw", "recognition_adamw", "immutable_detached_snapshot", "model_adamw"]
    operation: str
    repetitions: int
    arithmetic_flops_per_repetition: int
    bytes_copied_per_repetition: int
    total_arithmetic_flops: int
    term_sha256: str

@dataclass(frozen=True)
class MatchingReport:
    endpoint_config_sha256: str
    reference_config_sha256: str
    parameter_roles: tuple[ParameterRoleRecord, ...]
    optimizer_bindings: tuple[OptimizerBinding, ...]
    flop_terms: tuple[FlopTerm, ...]
    parameter_relative_difference: float
    flop_relative_difference: float
    eligible: bool
    obligations: tuple[str, ...]
    report_sha256: str

@dataclass(frozen=True)
class ArmMatrixRow:
    row_id: Literal["PRIMARY", "MAP", "STRUCTURE", "PRIOR", "MIXTURE", "OBJECTIVE", "LATENT", "RECOGNITION"]
    left_config_sha256: str
    right_config_sha256: str
    named_factor: str
    nuisance_capacity_fields: tuple[str, ...]
    tuning_estimand: Literal["equal_grid", "shared_a5"]
    interpretation: Literal["primary", "conditional", "descriptive"]
    row_sha256: str

@dataclass(frozen=True)
class BuiltArm:
    config: ArmConfig
    model: nn.Module
    recognition_store: LanguageRecognitionParameterStore | None
    proposal: TargetFreeProposalAdapter
    predictor: PriorPredictor
    parameter_roles: tuple[ParameterRoleRecord, ...]
    optimizer_bindings: tuple[OptimizerBinding, ...]
    flop_terms: tuple[FlopTerm, ...]
    model_family_sha256: str

@dataclass(frozen=True)
class PrefixCaseKey:
    arm: ArmId
    predictor_config_sha256: str
    estimator_sha256: str
    model_family_sha256: str
    vocabulary_sha256: str
    data_safety_sha256: str
    git_head: str
    dirty_digest: str

@dataclass(frozen=True)
class PrefixCertificate:
    key: PrefixCaseKey
    validation_payload_canonical_json: bytes
    validation_payload_sha256: str
    status: EvidenceStatus
    obligations: tuple[str, ...]
    certificate_sha256: str

@dataclass(frozen=True)
class PredictionCorrectnessArtifactRef:
    gate: Literal["H1", "H2", "H3", "H5"]
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    config_sha256: str
    validation_payload_sha256: str
    status: GateStatus

@dataclass(frozen=True)
class H1PrefixPriorArtifactRef:
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    generative_factor_schema_sha256: str
    config_sha256: str
    validation_payload_sha256: str
    status: GateStatus

@dataclass(frozen=True)
class SmcAccuracyArtifactRef:
    artifact_path: Path
    manifest_sha256: str
    git_head: str
    dirty_digest: str
    estimator_sha256: str
    fixture_set_sha256: str
    validation_payload_sha256: str
    status: GateStatus

@dataclass(frozen=True)
class H5UpdateBinding:
    h5_manifest_sha256: str
    h5_payload_sha256: str
    update_spec_raw_sha256: str
    update_spec_canonical_sha256: str
    objective_schema_sha256: str
    factor_input_schema_sha256: str
    reference_sha256: str
    recognition_state_sha256: str
    model_state_sha256: str
    validation_payload_sha256: str
    enabled_update_labels: tuple[
        Literal["exact_coordinate", "generalized_em", "natural_gradient_proposal"], ...
    ]
    binding_sha256: str

@dataclass(frozen=True)
class H6OuterSchedule:
    schedule_schema: Literal["h6-outer-schedule-v1"]
    optimizer_class: Literal["AdamW"]
    optimizer_policy_sha256: str
    model_updates_per_batch: Literal[1]
    validation_twentieths_per_pass: Literal[20]
    full_passes: Literal[2]
    outer_schedule_sha256: str

@dataclass(frozen=True)
class H6ArmPhaseSchedule:
    endpoint_config_sha256: str
    latent_enabled: bool
    phases: tuple[TrainingPhase, ...]
    recognition_updates_per_batch: Literal[0, 1]
    model_updates_per_batch: Literal[1]
    no_op_phases: Literal[0]
    phase_schedule_sha256: str

@dataclass(frozen=True)
class H6TrainingSchedule:
    schedule_schema: Literal["h6-training-schedule-v2"]
    outer: H6OuterSchedule
    endpoint_phases: tuple[H6ArmPhaseSchedule, ...]
    schedule_sha256: str

@dataclass(frozen=True)
class H6PredictionReadinessToken:
    readiness_schema: Literal["h6-prediction-readiness-v1"]
    git_head: str
    dirty_digest: str
    experiment_config_sha256: str
    correctness_manifests: tuple[tuple[Literal["H1", "H2", "H3", "H5"], str], ...]
    h1_prefix_prior_manifest_sha256: str | None
    h5_update_binding_sha256: str
    h6_training_schedule_sha256: str
    smc_validation_manifest_sha256: str
    critical_values_sha256: str
    endpoint_smc_protocol_sha256: str
    attribution_matrix_sha256: str
    matching_set_sha256: str
    prefix_certificate_set_sha256: str
    data_identity_sha256: str
    access_policy_sha256: str
    readiness_sha256: str
    status: Literal["PASS"]

@dataclass(frozen=True)
class ProjectedCurrentCandidateConfig:
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"]
    raw_config: Mapping[str, object]
    canonical_sha256: str

@dataclass(frozen=True)
class CandidateArtifactReference:
    artifact_path: Path
    git_head: str
    dirty_digest: str
    manifest_sha256: str
    payload_hashes: Mapping[str, str]

def project_h1_prefix_prior_config(
    raw_config: Mapping[str, object],
) -> ProjectedCurrentCandidateConfig: ...

def project_h6_prefix_config(
    raw_config: Mapping[str, object],
) -> ProjectedCurrentCandidateConfig: ...

def run_projected_current_candidate(
    *,
    config: ProjectedCurrentCandidateConfig,
    junit_sha256: str | None,
    predecessor_refs: Mapping[str, CandidateArtifactReference],
) -> CandidateArtifactReference: ...

@dataclass(frozen=True)
class EncodedTokenStorageIdentity:
    storage_schema: Literal["vfe4-h6-u16le-tokens-v1"]
    token_count: int
    byte_length: int
    encoded_token_sha256: str

@dataclass(frozen=True)
class ValidationSafetyFixture:
    policy: Literal["vfe4-h6-validation-safety-fixture-v1"]
    validation_token_sha256: str
    starts: tuple[int, ...]                 # exactly 4,096
    real_target_counts: tuple[int, ...]     # exactly 4,096
    fixture_sha256: str

@dataclass(frozen=True)
class FrozenBatchSchedule:
    schedule_schema: Literal["vfe4-h6-frozen-batch-schedule-v1"]
    shared_seed: Literal[2026072199]
    zero_based_pass_index: int
    window_count: int
    batch_size: Literal[8]
    drop_last: Literal[False]
    permutation: tuple[int, ...]
    schedule_sha256: str

class _OpeningProofValidator:
    __registered_proof_canonical_bytes: bytes | None
    __registered_identities: tuple[str, ...] | None

@dataclass(frozen=True)
class ExperimentIdentity:
    checkpoint_set_sha256: str
    current_candidate_sha256: str
    sealed_data_sha256: str
    access_policy_sha256: str
    analysis_sha256: str
    stream_protocol_sha256: str
    experiment_identity_sha256: str

class DurableTestOpeningCapability(Protocol):
    @property
    def proof_identity_sha256(self) -> str: ...

class ValidatedTestOpening(Protocol):
    @property
    def proof_identity_sha256(self) -> str: ...

@dataclass(frozen=True)
class BlindedCorpusStore:
    data_identity_sha256: str
    sealed_train_handle: SealedSplitHandle
    sealed_validation_handle: SealedSplitHandle
    frozen_validation_fixture: ValidationSafetyFixture
    sealed_test_handle: SealedSplitHandle
    _opening_validator: _OpeningProofValidator = field(repr=False, compare=False)

def acquire_wikitext2_blinded(config: H6DataConfig) -> BlindedCorpusStore: ...

def publish_blinded_binary_directory(
    destination: Path,
    payloads: Mapping[Literal[
        "sealed/wiki.train.raw",
        "sealed/wiki.valid.raw",
        "sealed/wiki.test.raw",
        "validation_safety_fixture.bin",
        "data_identity.json",
    ], BoundedBinarySource],
) -> BinaryDirectoryReference: ...

def reserve_and_issue_durable_test_opening_capability(
    *,
    store: BlindedCorpusStore,
    readiness: H6PredictionReadinessToken,
    experiment_identity: ExperimentIdentity,
    reservation_path: Path,
) -> DurableTestOpeningCapability: ...

def validate_durable_test_opening_capability(
    store: BlindedCorpusStore,
    opening: DurableTestOpeningCapability,
) -> ValidatedTestOpening: ...

def materialize_prediction_train(
    store: BlindedCorpusStore,
    readiness: H6PredictionReadinessToken,
) -> MaterializedPredictionData: ...

def open_test_for_scoring(
    store: BlindedCorpusStore,
    opening: DurableTestOpeningCapability,
) -> TestCausalWindows: ...

def require_prefix_pass(
    key: PrefixCaseKey,
    certificates: Mapping[PrefixCaseKey, PrefixCertificate],
) -> PrefixCertificate: ...

def score_prior_nll_replicate(
    predictor: PriorPredictor,
    windows: Iterable[CausalWindow],
    estimator_stream: EstimatorStream,
    particle_count: Literal[128, 256, 512, 1024],
    certificate: PrefixCertificate,
) -> NllTotals: ...
```

`ZeroDimensionalBase` accepts exactly `base_id="C0"`, `points=("*",)`, and `dimension=0`. `CausalDag` owns immutable `CausalDagRow(receiver_t, parents)` records, so its separately computed hash intrinsically binds every receiver-to-parent mapping rather than relying on positional tuple interpretation. It accepts only explicit zero-based integer node labels with no gaps or duplicates; rows have unique strictly increasing receivers, unique declared parents, and strict `j < receiver_t`. `H6LanguageStructure.receiver_labels` must equal `tuple(row.receiver_t for row in dag.rows)` exactly, in order. Self/future parents, duplicate parents, missing/duplicate receivers, a one-based alternative, and any labeling that could be interpreted as either zero- or one-based are rejected. Token/DAG edges never enter base transport, curvature, or holonomy records.

Digest roles are explicit. Each record has at most one **owned integrity digest** (for example `canonical_sha256`, `structure_sha256`, `binding_sha256`, `schedule_sha256`, `readiness_sha256`, or `certificate_sha256`): compute that digest from a domain-separated canonical preimage containing every semantic field of the record, including all already-verified reference/content digests, while excluding only the owned digest field itself. By contrast, **referenced/content digests** such as `tokenizer_spec_sha256`, `raw_bytes_sha256`, fixture/payload/manifest hashes, and H5 producer hashes are independently recomputed from and checked against their named external bytes or producer-defined canonical preimages; they are not recomputed from the containing record. Construction rejects either a wrong owned digest or an unverified/mismatched reference digest. A one-bit semantic mutation changes the owned digest, and a one-bit external-byte mutation invalidates its reference digest without creating a self-referential hash.

`FrozenTensorSnapshot.capture` privately owns a contiguous clone without detaching it from the autograd graph, records dtype/shape/device/contiguity/`requires_grad`/tensor storage-version/raw-byte SHA metadata, and exposes tensor values only as fresh clones that preserve the autograd path. Construction, every public access, canonical serialization, hashing, metrics, and checkpoint publication call `assert_intact`; any in-place mutation or metadata/raw-byte/version mismatch is rejected. `H6FactorTerm`, `H6LanguageElboTerms`, `EmissionOnlyAblationTerms`, and `PriorPrediction` hash immutable snapshot metadata and bytes, never a caller-owned mutable tensor, so their canonical identity cannot diverge after mutation.

`PrefixCertificate` binds the complete `PrefixCaseKey`, including `data_safety_sha256`, plus immutable canonical validation bytes, their independently checked hash, status, obligations, and its own domain-separated certificate hash. PASS is fail-closed: it requires an exact PASS validation payload for the same complete key, empty obligations, all required dynamic/static checks present and passing, matching payload bytes/hash, and a valid certificate hash; otherwise construction returns/requires FAIL or INCONCLUSIVE and `require_prefix_pass` rejects it.

`EncodedTokenStorageIdentity` hashes only the exact unsigned-16-bit little-endian preimage frozen in Global Constraints; `byte_length` is the ASCII domain-header length plus `8 + 2*token_count`. `ValidationSafetyFixture` validates exactly 4,096 unique starts and its binary file reproduces every stored input/target without consulting train/test data. `FrozenBatchSchedule` validates a complete permutation of `range(window_count)` and derives batches only by contiguous groups of eight with no drop-last.

`publish_blinded_binary_directory` accepts exactly the five caller payload names above and rejects missing, extra, duplicate, or caller-supplied `manifest.sha256` entries. Regardless of mapping insertion order, it stages and manifest-hashes them in the one frozen path order. It streams/hash-counts those payloads, then creates the manifest file itself from the frozen ordered `(path, uint64 length, raw content SHA-256)` preimage; the manifest excludes itself. `data_identity.json` is independently canonicalized and must not contain the enclosing manifest identity. Publication retains the same-volume/O_EXCL/no-replace/closed-handle/owned-cleanup rules and never delegates to `publish_run_directory`, `ZipFile.extract`, or `extractall`. The public acquisition function has no URL/opener argument; only synthetic unit tests call the non-exported internal opener seam.

Define `reservation_path_utf8` as UTF-8 of `unicodedata.normalize("NFC", str(reservation_path.resolve(strict=False))).replace("\\", "/")`. The reservation file contains exactly `b"VFE4-H6-DURABLE-TEST-OPENING-PROOF-V1\x00" || uint32_le(len(reservation_path_utf8)) || reservation_path_utf8 || readiness_sha256_raw32 || experiment_identity_sha256_raw32 || data_identity_sha256_raw32 || sealed_test_sha256_raw32 || access_policy_sha256_raw32 || b"RESERVED\x00"`; its SHA-256 is the reservation/proof identity. The public capability/validated-opening names are non-instantiable Protocols; their concrete classes, proof bytes, constructor token, and `_OpeningProofValidator` are module-private and accepted only by exact concrete type plus module-private issuer token identity. `reserve_and_issue_durable_test_opening_capability` is the sole constructor path: it validates `ExperimentIdentity`, derives fields from exact readiness/experiment/store objects, writes these bytes with durable O_EXCL semantics, fsyncs file/directory, registers an immutable copy plus decoded identities in the store's private one-shot validator, and only then constructs the private capability. The sole validator compares exact concrete type, capability, independently retained registry, and re-read immutable file bytes/hashes/fields before returning the private `ValidatedTestOpening` implementation. `open_test_for_scoring` accepts no strings, structural impostors, or alternate proof path; any forged/reconstructed capability or mismatch fails before mapping.

`TrainingPhase` is a closed H6 enum with exactly `MODEL_CE_ADAMW`, `RECOGNITION_ADAMW`, `IMMUTABLE_DETACHED_SNAPSHOT`, and `MODEL_ADAMW`. An endpoint with `latent_enabled=false` has phases `(MODEL_CE_ADAMW,)` and zero recognition updates. A latent endpoint has phases `(RECOGNITION_ADAMW, IMMUTABLE_DETACHED_SNAPSHOT, MODEL_ADAMW)` and one recognition update. These are H6 schedule labels, not H5 labels. Any other tuple, phase reordering, dummy phase, recognition object on a no-latent endpoint, or mismatch between the phase schedule and endpoint config is rejected during resolution.

The three lifecycle adapters above are H6-owned independent compatibility interfaces. Both projections are pure and never mutate the one editable root `CONFIG`. `project_h6_prefix_config` includes only H6 Prefix identities and has no predecessor/PASS input. `run_projected_current_candidate` is keyword-only and requires `predecessor_refs == {}` for `H6-Prefix`; a nonempty mapping is rejected rather than recorded as Prefix provenance. The current H7/H8 plans must be synchronized and focused-tested against this frozen contract before Task 11 may close; this is a required documentation/consumer-compatibility task, not H6 Prefix or Prediction evidence. H8's `project_h7_compatibility_config` remains H8-owned.

The scorer calls `next_token_log_probs(prefix, rng, cache)` before reading the target for that position. It then selects the target log probability, updates token totals, and only on the next call includes that formerly scored token in the prefix. A cache can accelerate this sequence but cannot change the call contract.

## Frozen Weighted SMC Recursion and Accuracy Gate

For token position `t`, the filtered cache before prediction stores particle histories `U_(t-1)^n`, normalized carried log weights `log w_(t-1)^n` satisfying `logsumexp(log_w)=0`, the assimilated prefix/vocabulary/config/model/estimator hashes, cumulative `log_Z_hat`, and exact counter positions. At `t=1`, draw `U_0^n` from the normalized initial law and set `log w_0^n=-log N`.

For each particle, draw source variables and the next continuous latents from the exact generative prior transition conditioned on its own history; retain the parent's carried weight. If `ell_t^n(v)=log L_t(v|U_t^n)`, the pre-observation vocabulary law returned publicly is

```text
log p_hat_t(v) = logsumexp_n(log w_(t-1)^n + ell_t^n(v)),  v=0,...,V-1.
```

This is a weighted mixture. The implementation may use an unweighted mean only after a recorded resampling step has made every carried weight exactly `1/N`.

The immutable pending cache stores predicted `U_t^n`, `ell_t^n(:)`, parent weights, and the prefix digest. Only when the next call contains the formerly predicted token `x_t` as the newly appended prefix element does assimilation occur:

```text
log w_tilde^n = log w_(t-1)^n + ell_t^n(x_t)
log Z_hat_t   = logsumexp_n(log w_tilde^n)
log w_t^n     = log w_tilde^n - log Z_hat_t
ESS_t         = 1 / sum_n(exp(log w_t^n)**2)
```

Add `log Z_hat_t` to the cumulative normalizing-constant estimate. If `ESS_t < N/2`, systematic resampling consumes exactly one named `r~Uniform[0,1/N)` counter, forms ordered points `r+k/N` for `k=0..N-1`, constructs the float64 cumulative sum of `exp(log w_t)` in particle-index order with its final entry forced to exactly one, and chooses each ancestor by the smallest cumulative index greater than or equal to the point (`searchsorted(..., side="left")`). Carry the selected complete histories in point order and reset every log weight to `-log(N)`; otherwise carry `U_t^n,log w_t^n` unchanged and consume no resampling counter. The finite gate below fixes `N=256`, hence threshold 128; actual-endpoint assessment uses the same recursion at its frozen particle ladder. Resampling occurs after normalization/recording, never before prediction and never merely to make an unweighted average convenient. Cold reconstruction performs these operations from the initial law; warm cache performs the identical state transitions. The teacher-forced log-likelihood estimate is exactly `sum_t log Z_hat_t`, which equals the sum of the public selected `log p_hat_t(x_t)` values from the same pending particles.

The independent SMC gate freezes four raw JSON finite-state/source-mixture fixtures `h6-smc-finite-01` through `04`, each with `V=3`, `T=6`, strictly positive exact token probabilities at least `0.10`, emission likelihood ratio at most `1.25`, both source-bank variants where applicable, and exact enumeration of every latent/source history. Raw fixture hashes are frozen before running production. It uses exactly 512 replicate seeds `2026072300..2026072811`, the same 256-particle configuration, and 76 simultaneous cells: all `4*6*3=72` token-log-probability cells plus four sequence log-normalizers.

For each cell record all 512 replicate errors, mean error `m` (bias estimate), unbiased sample variance `s^2` with denominator 511, standard deviation, and exact value. Control one familywise `alpha=0.01` jointly across both the 76 bias intervals and 76 variance intervals by allocating `a=0.01/(2*76*2)=0.01/304` to each tail. Freeze `t_(1-a,511)=4.0243186150882195`, `chi2_(a,511)=393.23185025997486`, and `chi2_(1-a,511)=648.65591595794933`. The mean-error interval is `[m-t*s/sqrt(512), m+t*s/sqrt(512)]`; its upper absolute-bias bound is the maximum absolute endpoint, and its lower absolute-bias bound is zero if the interval contains zero and otherwise the minimum absolute endpoint. The variance interval is `[511*s^2/648.65591595794933, 511*s^2/393.23185025997486]`; take endpoint square roots for the SD interval. Coverage is model-based on independent replicate streams, finite second moments, and Gaussian replicate errors (the chi-square variance interval needs normality); the artifact states these assumptions and may not relabel them as distribution-free. Define `delta=0.01005033585350145`, `bias_limit=delta/10=0.001005033585350145`, and `sd_limit=delta/4=0.0025125839633753625`. PASS requires every cell's simultaneous upper absolute-bias bound at most `bias_limit` and simultaneous upper SD bound at most `sd_limit`, plus normalization/replay/cache/normalizer identities. FAIL requires a finite simultaneous lower bound above either limit or a witnessed recursion/identity defect. Otherwise the estimator gate is INCONCLUSIVE. Raw errors, bounds, assumptions, exact critical constants, per-cell bias/variance, and familywise decision are published; thresholds cannot be changed after outcomes.

The critical-value fixture freezes the exact probability/df/tail conventions and decimal constants `t_(0.975,7)=2.364624251592784`, `t_(1-0.01/304,511)=4.0243186150882195`, `chi2_(0.01/304,511)=393.23185025997486`, `chi2_(1-0.01/304,511)=648.65591595794933`, and the endpoint constant below. Its provenance records independent R `qt`/`qchisq` evaluation and 80-decimal incomplete-beta/incomplete-gamma CDF checks. Production performs no runtime quantile inversion and imports no SciPy; tests assert the exact literals, fixture hash, df, two-sided/tail allocation, unbiased-variance convention, and CDF residual certificates.

## Frozen Actual-Endpoint SMC Uncertainty Protocol

The finite gate above proves the recursion on bounded exact models; it does not establish estimator error for trained WikiText-2 checkpoints. Inside the one irreversible test opening, assess exactly 12 unique endpoint configurations (A0--A5 plus the six nonbase A5 component endpoints) times eight training seeds, for 96 checkpoints. Every checkpoint uses exactly 64 Monte Carlo replicate IDs `0..63` and particle counts `(128,256,512,1024)`. The common paired registry derives every counter key from `SHA256("h6-wt2-endpoint-mc-v1|2026072198|replicate_id|purpose")`; the same replicate ID is used across particle levels, endpoints, and paired contrasts. There is no chosen best stream, adaptive replicate, or post-opening extension.

For checkpoint `c`, replicate `r`, and particle count `N`, score the complete test corpus sequentially and record

```text
Y[c,r,N] = -fsum_t(log Z_hat[c,r,N,t]) / counted_test_targets.
```

The aggregation never averages batch means. For each `N`, record all 64 `Y` values, their `math.fsum(...)/64` mean, unbiased sample variance with denominator 63, and cross-level/common-endpoint covariance. Define per-replicate Richardson values

```text
Q0[r] = 2*Y[r,256]  - Y[r,128]
Q1[r] = 2*Y[r,512]  - Y[r,256]
Q2[r] = 2*Y[r,1024] - Y[r,512]
R1[r] = Q1[r] - Q0[r]
R2[r] = Q2[r] - Q1[r].
```

The reported estimator-qualified checkpoint NLL is `mean_r(Q2[r])`. This is a preregistered first-order `1/N` extrapolation, not an exact-likelihood claim. Its convergence/bias interpretation is conditional on a geometric remainder contraction no larger than `0.75` beyond the observed ladder. If that assumption is not supported by the simultaneous checks, the checkpoint and every consuming contrast are INCONCLUSIVE rather than silently treating `Q2` as exact.

There are exactly 352 simultaneous two-sided intervals: for each of 96 checkpoints, one interval each for `Q2`, `R1`, and `R2` (288), plus one common-stream paired `Q2_left-Q2_right` interval for each of eight matrix rows and eight training seeds (64). Control familywise `alpha=0.01` with per-tail `a_endpoint=0.01/(2*352)=0.000014204545454545455`, `df=63`, and frozen `t_(1-a_endpoint,63)=4.5144904535377144`. For any 64-vector `X`, its simultaneous half-width is `h(X)=4.5144904535377144*s(X)/sqrt(64)` and its upper absolute-mean bound is `abs(mean(X))+h(X)`. These t bounds assume independent registry entries, finite variance, and approximate Gaussian replicate means; common random numbers induce the intended within-ID cross-endpoint covariance. The final df=7 interval additionally assumes independent training seeds and the usual approximate normality of seed-level effects. None is described as distribution-free.

For checkpoint `c`, set `U1=abs(mean(R1))+h(R1)`, `U2=abs(mean(R2))+h(R2)`, conditional remaining-bias bound `B_c=U2/(1-0.75)=4*U2`, and random half-width `H_c=h(Q2)`. Eligibility requires `U2<=0.75*U1`, `B_c<=delta/40=0.00025125839633753625`, and `H_c<=delta/20=0.0005025167926750725`; equality is allowed. These are empirical simultaneous convergence bounds under the stated remainder assumption, not a proof of unbiasedness. A nonfinite value, missing replicate/particle level, contraction failure, or crossed threshold is INCONCLUSIVE; a witnessed recursion/identity defect is FAIL.

For matrix row and training seed `i`, form 64 common-stream values `D_i[r]=Q2_left_i[r]-Q2_right_i[r]`, point contrast `d_i=mean(D_i)`, paired Monte Carlo half-width `H_i=h(D_i)`, and total estimator-error radius `e_i=H_i+B_left_i+B_right_i`. Require `H_i<=delta/20` and `e_i<=delta/10=0.001005033585350145`. For each eight-seed row, enumerate all `2^8=256` corner vectors `(d_i+s_i*e_i)` for `s_i in {-1,+1}`. Compute the usual df=7 training-seed t interval with frozen `2.364624251592784` at every corner; the final estimator-aware interval is the minimum lower endpoint and maximum upper endpoint over all corners. PRIMARY/MAP decisions use only this inflated interval. Any endpoint or paired uncertainty failure makes the affected row INCONCLUSIVE, even if the uninflated interval would pass.

## Frozen Prediction and Attribution Matrix

Every row below uses all eight confirmatory seeds, one terminal checkpoint per endpoint/seed, exact endpoint prefix-certificate keys, and the all-or-none global test opening. Every endpoint must satisfy the 1% parameter, 5% whole-schedule training-FLOP, and exact optimizer-access checks; otherwise that row is ineligible/INCONCLUSIVE rather than relaxed. Prediction FLOPs are reported separately and do not affect matching. “Shared A5” means both endpoints use the A5-primary selected `(lr,wd)` and estimates a factor intervention conditional on that optimizer setting; it is not unequal tuning disguised as an architecture-wide optimum.

| ID | Left exact config / factory | Right exact config / factory | Sole config factor changed | Hyperparameter estimand | Interpretation |
|---|---|---|---|---|---|
| `PRIMARY` | `h6-a0-ar-v1` / `build_a0@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Whole declared architecture | Equal six-cell tuning per endpoint | Primary A0--A5 predictive contrast; not component attribution. |
| `MAP` | `h6-a2-generic-map-v1` / `build_a2@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Shared vertex-coboundary versus generic fixed-frame/non-coboundary maps (right versus left) | Equal six-cell tuning per endpoint | Conditional map-parameterization attribution only. A2 and A5 are otherwise identical, including both source banks, fixed priors, exact mixture, recognition, objective, both channels, and `B_t`; this is never an H7 covariance, connection, curvature, or holonomy claim. |
| `STRUCTURE` | `h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Recognition precision structure | Shared A5 | Recognition-family effect conditional on A5 tuning. |
| `PRIOR` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-prefix-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Fixed versus prefix-conditioned generative source prior | Shared A5 | Descriptive changed-joint contrast; right endpoint requires separate H1 rerun. |
| `MIXTURE` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-projection-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Exact mixture versus declared moment projection | Shared A5 | Descriptive approximation contrast with projection-error record. |
| `OBJECTIVE` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-emission-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Complete ELBO versus emission-only optimization | Shared A5 | Optimization-objective intervention; emission endpoint is not an ELBO. |
| `LATENT` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-nolatent-norecognition-v1` / `build_a5@h6-arm-v1` | Latent channel enabled versus disabled | Shared A5 | Descriptive because disabling latents changes the model and active capacity allocation; recognition is structurally absent on the right, not an inactive setting. |
| `RECOGNITION` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-filtering-v1` / `build_a5@h6-arm-v1` | Training recognition conditioning | Shared A5 | Training-regime effect; neither endpoint supplies held-out predictions from recognition. |

A1, A3, and A4 remain independently tuned, matched descriptive controls in the six-arm report. No pair may be silently substituted for a missing endpoint, seed, checkpoint, or certificate. Rows labeled descriptive cannot support a causal or H7 claim even if their intervals exclude zero.

## Source Ambiguities Resolved Before Evidence

- The manuscript specifies a finite language sample but not a corpus version; this plan binds the exact official raw URL and archive contract above and records the observed archive/member size/compression/CRC/hash identities before any H6 calculation. A changed response is INCONCLUSIVE pending an explicit preregistration revision, never auto-accepted.
- The source protocol gives two tuning seeds but only one identifier. This plan freezes `2026072199` and `2026072200`; the latter is the adjacent companion selected before outcomes.
- “Validation every twentieth pass” cannot literally mean every twentieth complete pass when confirmatory training ends after two passes. This plan freezes the only coherent schedule: every twentieth of each pass.
- The manuscript permits exact prediction or a declared estimator but constructs neither an SMC algorithm nor an error threshold. This plan freezes the weighted recursion, 256-particle finite gate, after-observation ESS-half systematic resampling, 512-replicate exact finite oracle, and separate actual-endpoint 64-stream `(128,256,512,1024)` assessment. The empirical NLL is the exact preregistered `Q2` aggregation and remains estimator-qualified; uncertainty is propagated into every paired decision.
- The manuscript requires matched capacity/compute but gives no widths. This plan freezes the selection algorithm and hard tolerances now; exact per-arm dimensions, counts, and FLOPs are measured and copied into the preregistration before tuning, without using predictive outcomes.

### Task 1: Freeze H6 types, configuration, status precedence, and preregistration

**Files:**
- Create: `vfe4/types/h6.py`
- Modify: `vfe4/types/__init__.py`
- Modify: `vfe4/types/results.py`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `vfe4/config/__init__.py`
- Modify: `tests/unit/test_config.py`
- Create: `tests/unit/test_h6_config.py`
- Create: `tests/unit/test_h6_types.py`
- Create: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`

**Interfaces:**
- Produces: `ZeroDimensionalBase`, `CausalDagRow`, `CausalDag`, `H6LanguageStructure`, `FrozenTensorSnapshot`, `ArmId`, `EvidenceStatus`, `VocabularyIdentity`, `PrefixCaseKey`, concrete `PrefixCertificate`, `PredictionCorrectnessArtifactRef`, `H1PrefixPriorArtifactRef`, `SmcAccuracyArtifactRef`, exact-producer-field `H5UpdateBinding`, `TrainingPhase`, `H6OuterSchedule`, `H6ArmPhaseSchedule`, `H6TrainingSchedule`, `H6PredictionReadinessToken`, blinded/materialized/test-opening data capabilities, `EstimatorSpec`, endpoint-SMC protocol types, `DataIdentity`, `CheckpointIdentity`, `NllTotals`, `PredictionDecision`, conditional H6 sections on existing `ResolvedConfig`, and the explicit `H6PrefixGateResult` / `H6PredictionResult` union members.
- Consumes: existing `GateStatus`, H1/H2/H3/H5 result types for Prediction only, and existing `resolve_config`; earlier result records remain unchanged. It does not consume H4 or any predecessor result for Prefix.

- [ ] **Step 1: Write failing immutable-type and strict-config tests.** Require a separately hashed exact-one-point `ZeroDimensionalBase`, immutable `CausalDagRow(receiver_t,parents)` records inside a separately hashed explicitly zero-based `CausalDag`, and exact structure/DAG receiver equality. For each record's single owned integrity digest, recompute the domain-separated semantic preimage including already-verified reference digests and excluding only that owned digest; test one-bit semantic mutations and supplied-wrong-digest rejection. Separately mutate every named external byte/preimage and prove its tokenizer/raw-data/fixture/payload/manifest/H5 producer digest fails independent verification rather than being recomputed from the container. Test `FrozenTensorSnapshot` ownership/integrity and fail-closed data-safety-bound certificates. Require exact schedules/protocols and independent Prefix versus Prediction prerequisites; reject H4 as required.

```python
def test_prediction_config_is_blocked_without_exact_prefix_set() -> None:
    raw = copy.deepcopy(H6_PREDICTION_CONFIG)
    raw["prerequisites"]["prefix_certificate_set_sha256"] = None
    with pytest.raises(ValueError, match="exact H6-Prefix certificate set"):
        resolve_h6_prediction_config(raw, repo_root=REPO_ROOT)

def test_public_arm_inventory_is_exact() -> None:
    assert tuple(ArmId) == tuple(ArmId(f"A{i}") for i in range(6))
```

- [ ] **Step 2: Run the focused RED tests.** Run `python -m pytest tests/unit/test_h6_types.py tests/unit/test_h6_config.py tests/unit/test_config.py -q`. Expected: FAIL because H6 records and conditional existing-resolver support do not exist.

- [ ] **Step 3: Extend the existing schema, resolver, immutable ownership boundary, and explicit result union.** Implement structural canonicalization, snapshots, and complete certificates. Give each record one domain-separated owned integrity digest whose preimage excludes only itself and includes verified reference digests; independently verify tokenizer/data/fixture/payload/manifest/H5 producer digests against their named bytes or producer preimages before container canonicalization. Prefix remains predecessor-free; Prediction adds exact H1/H2/H3/H5 and its other frozen inputs. Missing obligations cannot become PASS; do not add a parallel parser.

- [ ] **Step 4: Write the preregistration before any H6 computation.** Copy every Global Constraint, public interface, weighted SMC equation/bound, actual-endpoint `Y/Q/R/B/e` formula, 352-interval family, all frozen critical constants, attribution-matrix row, source/access resolution, source-freeze/current-candidate lifecycle, separate artifact schema, statistical formula, failure rule, and nonclaim. Define distinct `h6-prefix-v1`, `h6-prediction-readiness-v1`, and `h6-prediction-v1` schemas plus revision-specific deferred Prefix/Prediction ledger paths. Mark the exhaustive Prefix cases, 512-replicate finite-SMC grid, corpus tuning/training, 96-checkpoint assessment, and one-time test opening as separately authorized evidence operations, not source-build completion. State that measured archive hashes, arm dimensions/counts/FLOPs, and certificate-set hash are frozen before their evidence revision and are not selected using prediction results.

- [ ] **Step 5: Run the focused GREEN tests.** Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit.**

```text
git add vfe4/types/h6.py vfe4/types/__init__.py vfe4/types/results.py vfe4/config/schema.py vfe4/config/resolve.py vfe4/config/__init__.py tests/unit/test_h6_types.py tests/unit/test_h6_config.py tests/unit/test_config.py docs/preregistrations/2026-07-21-h6-prefix-prediction.md
git commit -m "feat: freeze H6 prefix and prediction protocol"
```

### Task 2: Build the identity-bound blinded WikiText-2 store and capability-gated windows

**Files:**
- Create: `vfe4/data/__init__.py`
- Create: `vfe4/data/byte_tokenizer.py`
- Create: `vfe4/data/wikitext2.py`
- Create: `vfe4/data/access.py`
- Create: `vfe4/data/windows.py`
- Create: `tests/unit/test_h6_byte_tokenizer.py`
- Create: `tests/unit/test_h6_wikitext2.py`
- Create: `tests/unit/test_h6_data_access.py`
- Create: `tests/unit/test_h6_windows.py`
- Read only: `vfe4/config/schema.py`, `vfe4/config/resolve.py`, and `docs/preregistrations/2026-07-21-h6-prefix-prediction.md` (Task 1-owned surfaces).

**Interfaces:**
- Produces: `ByteTokenizerV1`, `EncodedTokenStorageIdentity`, public `acquire_wikitext2_blinded(config) -> BlindedCorpusStore`, non-exported `_acquire_wikitext2_blinded(config, opener)`, five-payload/self-manifesting `publish_blinded_binary_directory`, `materialize_validation_safety_fixture`, `materialize_prediction_train(store, readiness)`, opaque `DurableTestOpeningCapability`, sole `reserve_and_issue_durable_test_opening_capability`, sole `validate_durable_test_opening_capability`, `open_test_for_scoring(store, opening)`, `CausalWindows`, and `FrozenBatchSchedule`.
- Consumes: `DataIdentity` and the canonical data config from Task 1.

- [ ] **Step 1: Write failing synthetic archive/storage/fixture/schedule/capability tests.** Retain the exact URL/opener, archive, uint16-le token, 4,096-window fixture, batch/permutation, and no-download tests. For publication, supply exactly five payloads in multiple mapping orders and require identical staging/manifest order; reject missing/extra/duplicate/caller-manifest inputs. Verify the exact manifest preimage/file bytes, raw lengths/content hashes, self-exclusion, and that `data_identity.json` rejects any enclosing manifest identity. Retain same-volume/O_EXCL/no-replace/closed-handle/owned-cleanup mutants. Prove direct/deserialized construction of `DurableTestOpeningCapability` is unavailable; only the issuer after durable O_EXCL proof bytes succeeds. Mutate every readiness/experiment/data/test/access/path/state/proof byte, registry copy, and on-disk reservation byte and require the sole validator/open path to fail before mapping.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_h6_byte_tokenizer.py tests/unit/test_h6_wikitext2.py tests/unit/test_h6_data_access.py tests/unit/test_h6_windows.py -q`. Expected: FAIL on missing data/capability modules.

- [ ] **Step 3: Implement exact acquisition, token bytes, validation fixture, and self-manifesting binary publication.** Public acquisition uses the official URL/private test seam; stream/close members, encode uint16-le IDs, and serialize the fixture exactly. Accept/stage exactly five caller payloads, canonicalize their fixed order, forbid enclosing-manifest fields in `data_identity.json`, compute raw lengths/hashes, and generate `manifest.sha256` internally from the frozen self-excluding preimage. Do not use `publish_run_directory`, `extract`, or `extractall`, and do not publish model-readable token files.

- [ ] **Step 4: Implement capability-gated materialization, schedules, and opaque opening proof.** Retain readiness-gated materialization and exact schedules. Give each store a private one-shot proof validator. The sole issuer derives canonical proof bytes from independently supplied readiness/experiment plus store identities, durably writes them with O_EXCL, registers an immutable proof/decoded identity copy, then privately constructs the capability. The sole validator compares capability, registry, and re-read file bytes/hashes/fields. `open_test_for_scoring` must obtain `ValidatedTestOpening` from that validator before its only read-only mapping path; it accepts no loose identity strings or alternate constructor.

- [ ] **Step 5: Review Task 1-owned measured-value slots without editing them.** Source buildout uses only synthetic fixtures; actual download remains deferred. If `schema.py`, `resolve.py`, or the preregistration lacks a required measured-value slot, stop Task 2 and record a reviewed omission proof for the Task 1 owner; Task 2 may modify/add those paths only after that review explicitly revises its file and commit lists. Otherwise they remain read-only and absent from the Task 2 commit.

- [ ] **Step 6: Run focused GREEN.** Run the Step 2 command. Expected: PASS on synthetic identity fixtures only; do not access an official/local corpus cache.

- [x] **Step 7: Commit.**

```text
git add vfe4/data tests/unit/test_h6_byte_tokenizer.py tests/unit/test_h6_wikitext2.py tests/unit/test_h6_data_access.py tests/unit/test_h6_windows.py
git commit -m "feat: add identity-bound WikiText-2 byte data"
```

### Task 3: Implement normalized language generative factors and source priors

**Files:**
- Create: `vfe4/generative/source_priors.py`
- Create: `vfe4/generative/language.py`
- Modify: `vfe4/generative/__init__.py`
- Modify: `vfe4/numerics/categorical.py`
- Create: `tests/unit/test_h6_source_priors.py`
- Create: `tests/unit/test_h6_language_generative.py`

**Interfaces:**
- Produces: the sole `masked_log_softmax_from_parents`, `AllInvalidSourceRowError`, `MaskCaseKey`, `FixedSourcePrior`, `PrefixConditionedSourcePrior`, and `LanguageGenerativeModel` normalized factor methods.
- Consumes: Task 1 structural config and Task 2 prefix token tensors only; it never consumes `RecognitionLaw`.

- [ ] **Step 1: Write failing normalization/support/causality tests.** Exercise exact masks before normalization, all-invalid errors, nonempty parents strictly `j<t`, fixed and prefix-conditioned priors, both source banks, vocabulary-sized normalized emissions for V=3 and V=258, and rejection of current-target/suffix-shaped inputs. Assert the exact 168-case small and 16,384-case WikiText-2 base mask manifests/counts. Mutants using post-softmax masking, direct alternate normalization, undeclared/self/future parents, or a second helper must be detected. Test that the singleton base and DAG are separate fields.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_h6_source_priors.py tests/unit/test_h6_language_generative.py -q`. Expected: FAIL on missing factors.

- [ ] **Step 3: Implement minimal normalized factors.** Use typed `CausalPrefix` rather than a full batch. Prefix-conditioned scores consume only prior token embeddings and earlier generated latent history. Every bank/variant passes raw logits and declared parents through `masked_log_softmax_from_parents`; no source module calls another normalization primitive. Apply `masked_fill(~mask, -inf)` before `log_softmax`, explicitly reject a row with no `True` mask, and return the mask identity. A3 bypasses source variables with immediate predecessor transitions rather than a degenerate learned categorical row.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit.**

```text
git add vfe4/generative/source_priors.py vfe4/generative/language.py vfe4/generative/__init__.py vfe4/numerics/categorical.py tests/unit/test_h6_source_priors.py tests/unit/test_h6_language_generative.py
git commit -m "feat: add normalized H6 language factors"
```

### Task 4: Add smoothing/filtering recognition and one complete language ELBO

**Files:**
- Create: `vfe4/recognition/language.py`
- Modify: `vfe4/recognition/__init__.py`
- Create: `vfe4/objective/language_elbo.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `tests/unit/test_h6_language_recognition.py`
- Create: `tests/unit/test_h6_language_elbo.py`

**Interfaces:**
- Produces: `StructuredLanguageRecognition`, `FactorizedLanguageRecognition`, `RecognitionConditioning`, `H6FactorTerm`, `H6LanguageElboTerms`, `EmissionOnlyAblationTerms`, `evaluate_language_elbo(...) -> H6LanguageElboTerms`, and `evaluate_emission_only_ablation(...) -> EmissionOnlyAblationTerms`.
- Consumes: generative factors from Task 3 and existing precision/information interfaces; predictive modules do not import these outputs.

- [ ] **Step 1: Write failing access/normalization/term tests.** Prove the recognition access rules and distinct exact/projection types. Require exact-horizon ordered factor identities, all seven partitions, no duplicate/missing term, and checked equality between complete decomposition and total language ELBO. Reject the two-step `ElboTerms`. Require every tensor field to be a `FrozenTensorSnapshot`; mutate the original tensor, a returned clone, and (in a negative control) private storage, proving canonical bytes/hash remain stable or access fails before use. Test clone-only access preserves autograd. Emission-only remains a distinct non-ELBO type.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_h6_language_recognition.py tests/unit/test_h6_language_elbo.py -q`. Expected: FAIL on missing language recognition/objective.

- [ ] **Step 3: Implement the families and horizon-indexed canonical assembler.** Build `H6FactorTerm` entries in canonical order, capturing each value through `FrozenTensorSnapshot`; derive partitions from that tuple, independently accumulate the total, capture both totals, assert snapshot integrity, and check equality. Canonical hashes bind verified snapshot metadata/bytes. Reject missing/duplicate/mutated terms and never adapt the two-step `ElboTerms`; emission-only stays outside the ELBO API.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit.**

```text
git add vfe4/recognition/language.py vfe4/recognition/__init__.py vfe4/objective/language_elbo.py vfe4/objective/__init__.py tests/unit/test_h6_language_recognition.py tests/unit/test_h6_language_elbo.py
git commit -m "feat: add H6 language recognition and ELBO"
```

### Task 5: Implement the target-blind prior predictor, immutable cache, and SMC estimator

**Files:**
- Create: `vfe4/predictive/__init__.py`
- Create: `vfe4/predictive/cache.py`
- Create: `vfe4/predictive/smc.py`
- Create: `vfe4/predictive/prior.py`
- Create: `vfe4/numerics/critical_values.py`
- Create: `verification/h6_smc_gate.py`
- Create: `verification/fixtures/h6_critical_values_v1.json`
- Create: `verification/fixtures/h6_smc_finite_01.json`
- Create: `verification/fixtures/h6_smc_finite_02.json`
- Create: `verification/fixtures/h6_smc_finite_03.json`
- Create: `verification/fixtures/h6_smc_finite_04.json`
- Create: `tests/unit/test_h6_prior_predictor.py`
- Create: `tests/unit/test_h6_predictive_cache.py`
- Create: `tests/unit/test_h6_critical_values.py`
- Create: `tests/oracle/test_h6_smc_oracle.py`

**Interfaces:**
- Produces: the frozen `PriorPredictor` implementation boundary, `EstimatorStream`, weighted `PrefixCache`/pending state, `BootstrapSmcPredictor`, `SmcAccuracyReport`, and `run_h6_smc_gate` plus its immutable artifact.
- Consumes: only Task 3 generative interfaces, Task 1 `VocabularyIdentity`/estimator/`SmcAccuracyArtifactRef` types, and prefix tokens.

- [ ] **Step 1: Write failing signature, target-blind, and snapshot tests.** Inspect the exact bound signature/import boundary; require `PriorPrediction.log_probs` to be a `FrozenTensorSnapshot` of shape `(V,)`; compare metadata/raw bytes/hash across common streams; prove caller mutation of the source or returned clone cannot alter the record, autograd is preserved, and a private-storage/version mutation fails on access. Reject wrong cache/prefix/vocabulary/config/model-state/estimator/data-safety identities and prove the target is not read before scoring.

- [ ] **Step 2: Write failing independent estimator/constant tests on a shrunken deterministic grid.** Raw-hash the four frozen finite-model schemas, but use a small fixed subset of cells/seeds/particles to test carried nonuniform weights, weighted mixtures, incremental weights, `log Z_hat_t`, normalization/resampling order, ESS, systematic ancestors, cache state, cold/warm replay, and counters. Unit-test the complete 76-cell/512-replicate inventory formulas, df=511, familywise allocation, literal constants, thresholds, and status boundaries without executing that grid. Assert no SciPy import and fail on a one-ULP constant mutation. The full seeds `2026072300..2026072811` run only in deferred Task 13 evidence.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_h6_prior_predictor.py tests/unit/test_h6_predictive_cache.py tests/unit/test_h6_critical_values.py tests/oracle/test_h6_smc_oracle.py -q`. Expected: FAIL on missing predictive/critical-value modules.

- [ ] **Step 4: Implement the exact weighted bootstrap recursion.** Implement the equations in “Frozen Weighted SMC Recursion and Accuracy Gate” literally: propagate each history from the generative proposal while carrying its normalized parent weight; return `logsumexp(log_weight + emission_log_prob)` for each vocabulary item; store pending particles/emission rows/weights; assimilate only the newly appended formerly scored token; add the incremental log normalizer; normalize; compute ESS; then resample ancestors and reset weights only below `N/2` (128 for the finite gate's `N=256`). Use counter keys `(stream_seed, prefix_digest, position, purpose, particle_index)` so paired arms/cases share streams without global RNG dependence. Reject proposal modes other than the declared generative bootstrap.

- [ ] **Step 5: Implement frozen constants and the estimator gate.** Store the five literal critical constants in `vfe4/numerics/critical_values.py`; load no quantile package at runtime. Implement the exact joint Bonferroni t bias and chi-square variance calculations over 76 cells with `a=0.01/304`, but exercise them during buildout only on shrunken deterministic fixtures. The separately authorized Task 13 evidence run requires every upper absolute-bias bound `<=0.001005033585350145`, every upper SD bound `<=0.0025125839633753625`, exact identities, and the complete 512-replicate inventory before publishing `validation/h6_smc_accuracy.json`. Focused Task 5 output is not that evidence.

- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS.

- [ ] **Step 7: Commit.**

```text
git add vfe4/predictive vfe4/numerics/critical_values.py verification/h6_smc_gate.py verification/fixtures/h6_critical_values_v1.json verification/fixtures/h6_smc_finite_01.json verification/fixtures/h6_smc_finite_02.json verification/fixtures/h6_smc_finite_03.json verification/fixtures/h6_smc_finite_04.json tests/unit/test_h6_prior_predictor.py tests/unit/test_h6_predictive_cache.py tests/unit/test_h6_critical_values.py tests/oracle/test_h6_smc_oracle.py
git commit -m "feat: add target-blind H6 prior predictor"
```

### Task 6: Add the new H1 prefix-conditioned-prior prerequisite

**Files:**
- Create: `vfe4/validation/fixtures/h1_prefix_prior_v1.json`
- Create: `verification/numpy_oracles/h1_prefix_prior.py`
- Create: `verification/h1_prefix_prior_gate.py`
- Create: `tests/oracle/test_h1_prefix_prior.py`
- Modify: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`

**Interfaces:**
- Produces: `run_h1_prefix_prior(config) -> (GateResult, Path)` and an immutable separate artifact keyed by generative-factor schema/config plus the existing `git_head`/`dirty_digest` provenance pair.
- Consumes: Task 1 `H1PrefixPriorArtifactRef`, H1 public assembly/oracle patterns, and Task 3 prefix prior; it does not overwrite `validation/h1.json` or earlier H1 evidence.

- [ ] **Step 1: Freeze a bounded target-blind prefix-prior fixture and write failing tests.** The fixture has `T=2`, `d_z=d_m=1`, vocabulary 3, both parents at `t=2`, and at least two distinct prefixes that produce distinct normalized priors. Compare monolithic, local, and evidence-plus-posterior-KL identities independently; add a negative control that supplies the current target to the prior and must be detected.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/oracle/test_h1_prefix_prior.py -q`. Expected: FAIL because the fixture/oracle/gate are absent.

- [ ] **Step 3: Implement and bind the new H1 evidence surface.** Raw-byte hash the fixture, record exact prefix-prior generative/config hashes and producer revision/digest fields, retain the H1 calibrated allowance machinery, and implement separate atomic `validation/h1_prefix_prior.json` publication. Focused Task 6 fixtures are source tests, not evidence; deferred Prediction readiness consumes only a separately authorized exact-candidate H1-prefix-prior artifact.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit.**

```text
git add vfe4/validation/fixtures/h1_prefix_prior_v1.json verification/numpy_oracles/h1_prefix_prior.py verification/h1_prefix_prior_gate.py tests/oracle/test_h1_prefix_prior.py docs/preregistrations/2026-07-21-h6-prefix-prediction.md
git commit -m "test: verify prefix-conditioned source priors"
```

### Task 7: Build six explicit arm factories and hard capacity/compute matching

**Files:**
- Create: `vfe4/training/__init__.py`
- Create: `vfe4/training/arms.py`
- Create: `vfe4/training/matching.py`
- Create: `vfe4/recognition/parameter_store.py`
- Modify: `vfe4/recognition/__init__.py`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `vfe4/config/__init__.py`
- Create: `tests/unit/test_h6_arms.py`
- Create: `tests/unit/test_h6_matching.py`
- Modify: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`

**Interfaces:**
- Produces: immutable `ArmConfig`, `CapacityAllocation`, `BuiltArm`, `ParameterRoleRecord`, `OptimizerBinding`, `FlopTerm`, `MatchingReport`, and `ArmMatrixRow`; trainable `LanguageRecognitionParameterStore` implementations; `build_a0` through `build_a5`; `build_arm(ArmId, ArmConfig) -> BuiltArm`; the eight exact matrix rows; `audit_arm_matching(...) -> MatchingReport`; and exact `arm_matrix_sha256` / per-row hashes.
- Consumes: Task 3 normalized model factors, Task 4 immutable recognition-law value records, and Task 5 `CausalPrefix`, `PriorPredictor`, `EstimatorStream`, and arm-agnostic `TargetFreeProposalAdapter`. `LanguageGenerativeProposalAdapter` is one Task 3 adapter, not the required concrete type for every arm; each arm factory supplies a target-free adapter matching its own semantics.

- [ ] **Step 1: Write one bounded failing semantic test file.** Use only `V=3`, `T=2`, one deterministic forward call, and constructor/static checks; do not train. Assert the following literal families:
  - A0 has normalized AR logits, only conventional CE/model parameters, and no latent, source, map, recognition store/law, or recognition optimizer.
  - A1 is one ordinary Gaussian state chain with a trainable recognition store and no categorical source bank, internal map, model channel, `B_t`, or model-source bank.
  - A2 and A5 are identical after deleting only `map_mode` and the map-parameter payload: both have state and model channels, categorical state/model source banks, fixed priors, exact mixture, the same recognition family/conditioning, the same complete objective, and the same full same-receiver `B_t`. A5 computes shared vertex frames `U_t=exp(Phi_t)` and `Omega_tj=U_t U_j^-1`; A2 replaces only `Omega_tj` with independent dense fixed-frame/non-coboundary edge maps `A^z_tj` and `A^m_tj`.
  - A3 is typed and dual-channel, uses only the fixed immediate predecessor, and constructs no categorical source variable or source bank.
  - A4 is typed shared-vertex-coboundary and state-only, with one categorical state-source bank and no model channel, `B_t`, or model-source bank.
  - A5 is the full dual-channel, dual-source-bank shared vertex-coboundary family.

  The no-latent component endpoint likewise has only its model phase and canonical `nolatent-norecognition` identity. Each latent store must own every recognition `nn.Parameter` and emit the appropriate normalized Task 4 law; the law owns no parameters. Require every `BuiltArm.proposal` to satisfy `TargetFreeProposalAdapter`, and require the exact predictor signature `next_token_log_probs(prefix_tokens: CausalPrefix, estimator_rng: EstimatorStream, cache: PrefixCache | None = None) -> PriorPrediction`; raw tensors fail. Require all eight literal matrix identities, one semantic intervention after deleting only `capacity_allocation`, declared nuisance allocation fields, tuning estimands, seeds/checkpoint/certificate templates, opening group, and descriptive/nonclaim labels. Freeze a required family-specific ELBO inventory identity/factor schema for each latent family for downstream objective/training work; do not reuse A5's full `1+6T` inventory for A1, A3, or A4 when factors are structurally absent.

- [ ] **Step 2: Write one bounded failing matcher test file.** Reconstruct only the literal formula records; do not load data, execute a training step, or enumerate outcomes. Freeze A5 at `(emission_width=64, latent_width=16, recognition_width=64)`. In field/lexicographic order, latent endpoints enumerate exactly `emission_width=(48,64,80,96)`, `latent_width=(8,16,24,32)`, and `recognition_width=(32,64,96)`, for at most 48 candidates. A0 and every no-latent endpoint enumerate exactly the four emission widths with both other fields `None`. Prove every present field controls a live tensor shape and a live forward/training operation; reject any field used only in identity/provenance, any unbound/duplicate/dormant/filler parameter, and any dummy/no-op phase.

  Freeze the arithmetic ledger rather than using a profiler: dense matmul `(m,n)@(n,k)` costs `2mnk`; dense matvec costs `2mn`; every scalar add, subtract, multiply, divide, exp, log, sqrt, comparison, or select costs one; length-`n` `log_softmax` costs `5n-1`; backward costs exactly `2 * differentiable_forward_flops`; the always-evaluated L2 clip/scale costs `3P+3` for `P` active gradient scalars; AdamW costs `18P` per update; and an immutable detached snapshot costs zero arithmetic FLOPs while recording exact bytes copied. A `FlopTerm` records phase, operation, repetitions, arithmetic FLOPs per repetition, copied bytes, total, and digest. Whole-schedule training FLOPs include only actual active training phases over the common batches/passes and exclude data I/O, validation, checkpoint serialization, test scoring, and all estimator particle/cache work. Require identical passes, batches/model-update opportunities, data/validation/checkpoint boundaries, and the exact AdamW policy; do not require a nonexistent recognition phase for A0/no-latent endpoints. Enforce `abs(P_endpoint/P_A5-1)<=0.01` and `abs(F_train_endpoint/F_train_A5-1)<=0.05`. Prediction FLOPs are a separate reported quantity keyed by endpoint, prefix protocol, estimator, particle count, and cache mode; they are never used to pass/fail training-compute matching. Verify that capacity changes are labeled outcome-blind nuisance reallocations, never a second semantic intervention, and that unmatched or otherwise different endpoints are ineligible/INCONCLUSIVE.

- [ ] **Step 3: Run one focused RED.** Run `python -m pytest tests/unit/test_h6_arms.py tests/unit/test_h6_matching.py -q`. Expected: FAIL on missing factories/matcher in less than 10 seconds. If the command reaches 10 seconds, stop it, statically locate the slow collector/case, shrink the fixture, and rerun only that exact test node once.

- [ ] **Step 4: Implement explicit factories and the trainable recognition boundary.** `build_a0` through `build_a5` each construct only their literal family modules above, a matching concrete `TargetFreeProposalAdapter`, and a `PriorPredictor`; `build_arm` uses an explicit `if arm is ArmId.A0` through `A5` chain and exact config-arm equality, never registry/signature dispatch. A2 and A5 share every constructor input and semantic record except the exact map mode/parameter payload; mechanically compare those payloads and fail closed on any other difference. Latent builders construct one recognition parameter store whose named `nn.Parameter`s appear in the recognition AdamW binding exactly once; each batch call emits an ephemeral normalized structured/factorized law connected to those parameters. Predictive/generative modules never import the store or emitted law. Hash every exact semantic config and built model/proposal/predictor identity. Bind the required family-specific ELBO inventory schema for later implementation, but do not invent absent factors or train in Task 7.

- [ ] **Step 5: Implement deterministic matching, matrix records, and config resolution.** Resolve A5 to `(64,16,64)`. For each latent endpoint enumerate only the 48 applicable literal candidates; for each no-latent endpoint enumerate only the four emission-width candidates; select the first report passing both tolerances and every live-field/active-parameter/optimizer rule. Never inspect corpus bytes, loss, gradients, validation, test values, or prediction FLOPs. For a component row, compare semantic canonical payloads after deleting only `capacity_allocation`; require the remaining difference to equal the named factor and record any allocation differences in `nuisance_capacity_fields`. The MAP row's sole semantic difference is exactly shared vertex-coboundary maps versus independent dense generic fixed-frame/non-coboundary maps; `B_t`, both source banks, fixed priors, exact mixture, recognition, objective, and both channels must remain byte-identical at the semantic-record boundary. Hash each endpoint's full config, explicit factory identity, semantic diff, nuisance allocation diff, match report, tuning estimand, seeds, certificate-key template, and opening group. Add the closed candidate/policy/matrix records to `schema.py`, canonical resolution to `resolve.py`, and exports to `config/__init__.py`.

- [ ] **Step 6: Freeze eligible source profiles, then run one focused GREEN.** The no-corpus config/matcher resolver records exact live dimensions, parameter-role tables, optimizer bindings, training-FLOP terms, copied-byte terms, margins, family-specific ELBO inventory requirements, and hashes in config/preregistration before tuning. Prediction-FLOP reporting remains a distinct downstream record and is not folded into matching. These are source-frozen candidates, not Prefix or Prediction evidence. If any arm or row has no eligible allocation, retain INCONCLUSIVE and do not loosen tolerances. Run the Step 3 command once; expected PASS in less than 10 seconds. Do not run another H6 test file or a broad suite.

- [ ] **Step 7: Commit.**

```text
git add vfe4/training/__init__.py vfe4/training/arms.py vfe4/training/matching.py vfe4/recognition/parameter_store.py vfe4/recognition/__init__.py tests/unit/test_h6_arms.py tests/unit/test_h6_matching.py vfe4/config/schema.py vfe4/config/resolve.py vfe4/config/__init__.py docs/preregistrations/2026-07-21-h6-prefix-prediction.md
git commit -m "feat: add matched H6 arm factories"
```

### Task 8: Add shared training, exact checkpoints, prior NLL, and statistics

**Files:**
- Create: `vfe4/training/checkpoint.py`
- Create: `vfe4/training/language.py`
- Create: `vfe4/evaluation/__init__.py`
- Create: `vfe4/evaluation/prior_nll.py`
- Create: `vfe4/evaluation/smc_uncertainty.py`
- Create: `tests/unit/test_h6_checkpoint.py`
- Create: `tests/unit/test_h6_statistics.py`
- Create: `tests/unit/test_h6_smc_uncertainty.py`
- Create: `tests/integration/test_h6_language.py`

**Interfaces:**
- Produces: immutable `ArmObjectiveInventory`, `DetachedRecognitionLawSnapshot`, `H6AttemptSpec`, `H6AttemptCursor`, `H6ObjectiveManifest`, and `H6CheckpointManifest`; family-specific `ArmTrainingObjectiveAdapter`; `plan_h6_attempt`, `train_h6_attempt`, `save_h6_checkpoint`, `load_h6_checkpoint`, `score_prior_nll_replicate`, `aggregate_endpoint_smc`, `inflate_paired_interval`, `paired_t_interval`, and `decide_primary_prediction`.
- Consumes: Task 1 H6-owned AdamW `H6TrainingSchedule` and Prediction-only exact H5 producer-field binding, Task 2 batch schedules, Task 4 `H6LanguageElboTerms`/typed non-ELBO ablation, Task 5 predictor, and Task 7 arms/matching report.

- [x] **Step 1: Write failing synthetic planning/refusal and manifest tests.** Use no corpus and execute no optimizer step. Freeze one `ArmObjectiveInventory` per literal family: A0 contains CE emission only; A1 contains initial/state-transition/emission/entropy; A2 and A5 contain initial/state-source/model-source/state-transition/model-transition/emission/entropy; A3 omits both source factors; A4 omits the model source, model transition, and every model-channel term. `ArmTrainingObjectiveAdapter` must reject an absent, extra, reordered, duplicated, wrong-family, wrong-horizon, or stale-hash factor rather than filling it with zero or reusing A5's `1+6T` schema.

  Freeze a distinct `DetachedRecognitionLawSnapshot` for the recognition-to-model phase boundary. It owns detached, clone-only, `requires_grad=False` mean/precision-Cholesky bytes plus family, conditioning, parameter-store-state, dtype/shape/device, and digest identities. It is not Task 4's `FrozenTensorSnapshot`: Task 4 remains autograd-preserving and is not changed or reused as the detached phase record. Prove the detached record has no recognition graph edge, while mutation of its source or a returned clone cannot change its bytes.

  `plan_h6_attempt` and `train_h6_attempt` accept only a current exact `MatchingReport` with `status="ELIGIBLE"`, `eligible=True`, a complete operator-level whole-schedule FLOP ledger, empty FLOP obligations, exact parameter ownership, and the common schedule/policy hashes. Task 7 is presently FLOP-incomplete/INCONCLUSIVE, so every Task 8 development attempt must refuse before corpus access, optimizer construction, gradient evaluation, or parameter mutation. Tests may exercise typed phase planning and this refusal boundary only; they do not authorize a training step.

  Define hash-bound `H6AttemptSpec`, `H6AttemptCursor`, `H6ObjectiveManifest`, and `H6CheckpointManifest`. The attempt spec binds the exact source revision/digest, readiness, endpoint config/factory/model-family, eligible match report and complete FLOP ledger, objective inventory/adapter, H5 producer binding, H6 outer/phase/AdamW policy, tuning cell, seed, data/window/batch schedule, estimator, and Prefix-certificate identities. The cursor binds zero-based pass, batch, next phase, model/recognition update counts, validation/checkpoint boundary counts, data permutation/cursor, and RNG/counter state. The objective manifest binds the family inventory, every ordered live term and identity, totals/equality record, objective kind, and detached-recognition snapshot when applicable. The checkpoint manifest binds those records plus only active model/recognition/optimizer state manifests and exact raw-byte hashes. Use one tiny fake checkpoint to prove atomic save/load, mutation resistance, and exact resume at the next declared phase with no replay, skip, duplicated update, caller override, or identity substitution.

- [x] **Step 2: Write bounded scorer/uncertainty/statistics tests.** Use fake target-blind predictors and exactly one tiny fake checkpoint. For arithmetic only, use one synthetic `64 x 4` table (64 common streams by particle counts `128,256,512,1024`) to prove `Y/Q0/Q1/Q2/R1/R2`, denominator-63 variance/covariance, the frozen critical value, contraction, and `B/H/e` formulas. Use exactly eight paired scalar values and enumerate the 256 scalar corner vectors to test the df=7 interval inflation and PRIMARY/MAP rules. Test corpus-summed token weighting, `-100` exclusion, no recognition argument, missing/nonfinite/duplicate inputs, and wrong-boundary equality. Do not construct a 96-checkpoint development table, 24,576 corpus records, 352 materialized production intervals, or a real endpoint inventory; those remain separately authorized evidence operations.

- [x] **Step 3: Run one focused RED.** Run `python -m pytest tests/unit/test_h6_checkpoint.py tests/unit/test_h6_statistics.py tests/unit/test_h6_smc_uncertainty.py tests/integration/test_h6_language.py -q`. Expected: fail on the missing typed adapters/manifests/refusal path and finish in less than 10 seconds. At 10 seconds, stop, shrink, and rerun only the exact failing node once. The command must not load a corpus or execute an optimizer step.

- [x] **Step 4: Implement the typed objective, phase-plan, checkpoint, and refusal boundary.** Resolve the family-specific adapter from the exact arm semantic profile and inventory hash; never registry-dispatch or synthesize absent factors. Accept only immutable H6 common outer/endpoint-phase hashes; H5 labels remain provenance and never become H6 optimizer phases. A0/no-latent plans only `model_ce_adamw`; eligible latent endpoints plan `recognition_adamw -> DetachedRecognitionLawSnapshot -> model_adamw`. Emission-only accepts only its distinct non-ELBO record. Before any data or optimizer access, fail closed unless the match report is ELIGIBLE and its whole-schedule FLOP proof is complete. Implement atomic manifest validation and exact-resume reconstruction, but keep the development path training-disabled while Task 7 remains INCONCLUSIVE.

- [x] **Step 5: Implement bounded scoring and decision arithmetic.** Use `math.fsum`, explicit counted-target totals, exact typed stream/particle identities, and literal constants only. Production functions retain the full evidence interfaces and fail closed on incomplete inventories, but development tests exercise only the one-checkpoint `64 x 4`, eight-value, and 256-corner arithmetic fixtures. Production imports no SciPy and never substitutes an uninflated interval.

- [x] **Step 6: Run one focused GREEN.** Run the Step 3 command once. Expected: PASS for synthetic planning/manifests/arithmetic and exact refusal of all training because Task 7 is not ELIGIBLE/FLOP-complete. This is source verification, not training or Prediction evidence.

- [x] **Step 7: Commit.**

```text
git add vfe4/training/checkpoint.py vfe4/training/language.py vfe4/evaluation tests/unit/test_h6_checkpoint.py tests/unit/test_h6_statistics.py tests/unit/test_h6_smc_uncertainty.py tests/integration/test_h6_language.py
git commit -m "feat: add H6 training and prior scoring"
```

### Task 9: Implement the 9,720-case and 4,096-case dynamic prefix oracle

**Development safety:** `9,720` and `4,096` are closed-form/full-evidence inventory metadata, not routine CPU test counts. Focused execution is hard-capped at 16 supplied cases; this task's tests use four V=3 cases and two V=258 cases. Complete materialization remains Task 13-only and separately authorized.

**Files:**
- Create: `verification/numpy_oracles/h6_prefix.py`
- Modify: `verification/numpy_oracles/__init__.py`
- Create: `vfe4/validation/h6_prefix.py`
- Create: `vfe4/validation/fixtures/h6_prefix_small_v1.json`
- Create: `vfe4/validation/fixtures/h6_validation_perturbations_v1.json`
- Create: `tests/property/test_h6_prefix.py`

**Interfaces:**
- Produces: `enumerate_ordered_tail_pairs`, `load_frozen_validation_perturbations`, `run_dynamic_prefix_checks`, and `DynamicPrefixReport`.
- Consumes: exact predictor/config/estimator case key from Tasks 1/5/7 and only Task 2's identity-bound frozen validation-safety fixture; it has no train/test capability.

- [x] **Step 1: Write failing enumeration and perturbation tests on shrunken fixtures.** Exercise the real production predictor/factories with distinct V=3 and V=258 identities on a small deterministic subset. Unit-test the closed-form full inventory counts `9,720`, `(6561,2187,729,243)`, and `4,096`, generator identities, and no config mutation without executing the complete inventories during buildout.

- [x] **Step 2: Write failing leak/cache/mask tests.** Inject one target-reading predictor, one suffix-reading wrapper, one cache missing config identity, one post-softmax mask, and one all-invalid fallback. Each witnessed defect must FAIL. Remove a fixture/audit field and require INCONCLUSIVE. The correct predictor must produce exact zero residual in cold/warm/reverse modes.

- [x] **Step 3: Run focused RED.** The three-node focused file ran in `1.303 s`; two nodes passed and the one Windows `Path` branch failed. No other test file or inventory ran.

- [x] **Step 4: Implement independent enumeration and frozen perturbation generation.** The NumPy oracle constructs sequence pairs independently of production helpers and invokes the production predictor under the separately resolved small fixture. Focused tests generate only a deterministic subset. The complete 4,096 V=258 records and full 9,720 inventory are materialized and identity-bound only in the separately authorized Prefix evidence run.

- [x] **Step 5: Implement dynamic comparisons.** For each exact case key, instantiate a fresh counter stream for the pair and capture log probabilities before selecting any target. Require equal vocabulary identity, dtype, shape `(V,)`, device, contiguity, raw `uint8` bytes, and SHA-256; do not use `torch.equal`, which hides signed-zero differences. Calculate exact mask mass, audit all-invalid behavior, and compare cache modes/traversal order. Record first counterexample and complete inventory; never short-circuit artifact accounting after a failure.

- [x] **Step 6: Run focused GREEN.** After the semantic tail-harness correction, the unaffected V=258 node passed in the three-node run and the two nodes affected by one duplicated sparse-index literal passed together in `1.353 s`. No broad rerun was performed.

- [ ] **Step 7: Commit.**

```text
git add verification/numpy_oracles/h6_prefix.py verification/numpy_oracles/__init__.py vfe4/validation/h6_prefix.py vfe4/validation/fixtures/h6_prefix_small_v1.json vfe4/validation/fixtures/h6_validation_perturbations_v1.json tests/property/test_h6_prefix.py
git commit -m "test: add exhaustive H6 prefix oracle"
```

### Task 10: Add static import/signature, taint/dataflow, and cache audits

**Files:**
- Create: `vfe4/validation/h6_static_audit.py`
- Create: `tests/unit/test_h6_static_audit.py`
- Create: `tests/unit/test_h6_taint_audit.py`
- Create: `tests/unit/test_h6_mask_audit.py`

**Interfaces:**
- Produces: `audit_h6_static_source(repo_root, exact_case_keys) -> StaticAuditReport`.
- Consumes: the explicit production module allowlist and case keys; it never imports test/oracle helpers into production.

- [ ] **Step 1: Write failing mutant-fixture tests.** Create temporary source trees containing forbidden recognition imports, extra predictor parameters, `**kwargs`, target-to-prior flow, suffix-to-emission flow, recognition-to-cache flow, cache values with target data, post-softmax masking, a second normalization helper, direct source-bank `softmax`/`log_softmax`, self/future parents, wrong 168/16,384 base counts, incomplete sink inventories, a preprocessing return that leaks sealed bytes, pre-readiness train materialization, training/tuning/analysis import of the private unsealer, and test mapping without a durable opening capability. Require FAIL for proven flows/count/support/access defects and INCONCLUSIVE for unresolved dynamic dispatch/reflection/import analysis.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_h6_static_audit.py tests/unit/test_h6_taint_audit.py tests/unit/test_h6_mask_audit.py -q`. Expected: FAIL because the auditor is absent.

- [ ] **Step 3: Implement the import/signature/normalization/access audit.** Preserve the generative/import checks. Prove only `vfe4.data.access` reaches the private unsealer, issuer, capability constructor, and proof validator; train materialization requires readiness; `open_test_for_scoring` requires the sole validator's `ValidatedTestOpening`; and blinded preprocessing returns no raw/token tensor. Hash audited source/rules; unresolved dispatch is INCONCLUSIVE.

- [ ] **Step 4: Implement the taint, cache, and split-capability audit.** Sources are target tensors, suffix tensors, complete windows beyond the slicing boundary, recognition objects/activations/parameters, posterior reconstructions, and sealed train/test contents. Sinks are source-prior logits, transition parameters, emission logits, estimator proposals/weights before observation, predictor outputs, cache keys/values, training/tuning/analysis inputs, and any public preprocessing return. Follow assignments, calls, returns, attributes, containers, capabilities, and explicit factory calls; unresolved reflection/eval/import makes the report INCONCLUSIVE. Cache keys require source/config/model-state/estimator/prefix identities; cache payload schema permits only causal filter state/counter positions. Test data may reach scoring only after the durable opening capability; it may never reach training/tuning.

- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: PASS for production and exact expected states for mutants.

- [ ] **Step 6: Commit.**

```text
git add vfe4/validation/h6_static_audit.py tests/unit/test_h6_static_audit.py tests/unit/test_h6_taint_audit.py tests/unit/test_h6_mask_audit.py
git commit -m "test: audit H6 causal dataflow"
```

### Task 11: Publish H6-Prefix, experiment artifacts, and both click-run launchers

**Files:**
- Create: `vfe4/artifacts/h6.py`
- Modify: `vfe4/artifacts/provenance.py`
- Modify: `vfe4/artifacts/__init__.py`
- Create: `verification/h6_prefix_gate.py`
- Modify: `verification/run_gates.py`
- Modify: `verify_vfe4.py`
- Create: `train_vfe4.py`
- Create: `vfe4/training/h6_experiment.py`
- Create: `vfe4/training/h6_readiness.py`
- Create: `tests/promotion/test_h6_prefix_gate.py`
- Create: `tests/unit/test_h6_prediction_readiness.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_atomic_artifacts.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Create: `tests/integration/test_train_vfe4.py`
- Modify: `docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md`
- Modify: `docs/superpowers/plans/2026-07-21-vfe4-h8-sparse-scale.md`

**Interfaces:**
- Produces: exact public `project_h1_prefix_prior_config`, `project_h6_prefix_config`, and `run_projected_current_candidate` lifecycle adapters with the return records/signatures frozen under Public Interfaces; `CurrentPredictionPrerequisiteRefs`, `run_h6_prefix`, `validate_h6_prediction_readiness`, `run_h6_experiment`, separate atomic Prediction-prerequisite/certificate/readiness/failure/checkpoint/test-opening schemas, and two editable root dictionaries.
- Consumes: all earlier tasks; launchers orchestrate only.

**Precondition:** Before Task 11 closes, update the H7 and H8 plan consumers to the exact independent-Prefix projector/runner signatures above and add a focused cross-plan compatibility test. Until that docs/consumer sync passes, H6 retains the desired frozen contract but does not claim the current H7 plan consumes it. This synchronization is required documentation/integration work, not H6 evidence.

- [ ] **Step 1: Write failing independent Prefix gate/artifact tests.** Require one prefix certificate per exact source/config/model-family/vocabulary/estimator/data-safety key, PASS/FAIL/INCONCLUSIVE precedence, stale-own-hash rejection, exact mask/case/static-audit inventories, atomic manifests, and no overwrite. Prove Prefix runs and publishes with no predecessor artifact or PASS state present. Reject any Prefix config, preflight, result, reference file, artifact, or ledger schema containing an H1--H5 status/reference. The artifact contains only `config.json`, `provenance.json`, `environment.json`, `validation/h6_prefix.json`, `certificates/prefix_set.json`, and `manifest.sha256`. Assert Prefix closure contains no H1 variant, SMC accuracy, H6 schedule, matching, tuning, capacity, checkpoint, opening, or prediction claim.

- [ ] **Step 2: Write failing Prediction-readiness/access/launcher and lifecycle-adapter tests.** Test fail-closed H1/H2/H3/H5 readiness and exact H5 fields/labels; H4 is absent. Freeze the three adapter signatures/records, pure projectors, keyword-only runner, `project_h6_prefix_config(CONFIG)` without predecessors, and `predecessor_refs={}` for Prefix. Add a focused consumer-contract fixture used by the synchronized H7/H8 plan text; reject their old predecessor-accepting Prefix signature. Prove H8 alone owns `project_h7_compatibility_config`. Retain one root `CONFIG`, one `main`, one guard, and no required CLI per launcher.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/promotion/test_h6_prefix_gate.py tests/unit/test_h6_prediction_readiness.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py tests/integration/test_train_vfe4.py -q`. Expected: FAIL on missing gate/readiness/launcher/orchestrator and conditional existing-surface wiring.

- [ ] **Step 4: Implement the H6-owned lifecycle adapters and independent H6-Prefix publication.** Implement the frozen pure projectors and keyword-only runner; Prefix requires `predecessor_refs={}` and publishes no predecessor reference. Publish the complete fail-closed certificates with verified `data_safety_sha256`, payload bytes/hash/status/obligations, and domain-separated certificate hash. Synchronize the H7/H8 plan consumer signatures and focused contract test in the same source task; leave `project_h7_compatibility_config` H8-owned.

- [ ] **Step 5: Implement Prediction readiness before experiment access.** `validate_h6_prediction_readiness` first revalidates the exact deferred-evidence H1/H2/H3/H5 artifacts at one `git_head`/`dirty_digest`, then the separate same-candidate H1-prefix-prior and finite-SMC artifacts/ledgers and independent H6-Prefix certificate set. It validates H5's actual fields/labels as correctness provenance while taking AdamW class/policy/phases solely from the common/typed H6 schedule. It also validates critical-value and actual-endpoint protocol hashes, the literal matrix, every prefix key, and the blinded-data access policy; then it reconstructs all endpoints without corpus access and mechanically reproduces/freeze-hashes every match report. It never requests H4 or launches an H4 benchmark. It publishes separate `h6_prediction_readiness.json` and returns an opaque PASS token. Only that token can materialize train data or start empirical operations; matching is a Prediction-readiness phase, not a Prefix claim.

- [ ] **Step 6: Implement atomic experiment/opening surfaces and the two launchers.** Keep package boundaries and launcher constraints. Construct an immutable `ExperimentIdentity` binding checkpoint/current-candidate/sealed-data/access-policy/analysis/stream identities, then use only Task 2's `reserve_and_issue_durable_test_opening_capability` and validator; no launcher constructs or decodes the opaque capability. Implement endpoint-MC/result schemas from Task 14. No argparse, Typer, or Hydra.

- [ ] **Step 7: Run focused GREEN.** Run the Step 3 command. Expected: PASS.

- [ ] **Step 8: Commit.**

```text
git add vfe4/artifacts/h6.py vfe4/artifacts/provenance.py vfe4/artifacts/__init__.py vfe4/types/results.py vfe4/config/schema.py vfe4/config/resolve.py verification/h6_prefix_gate.py verification/run_gates.py verify_vfe4.py train_vfe4.py vfe4/training/h6_experiment.py vfe4/training/h6_readiness.py tests/promotion/test_h6_prefix_gate.py tests/unit/test_h6_prediction_readiness.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py tests/integration/test_train_vfe4.py docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md docs/superpowers/plans/2026-07-21-vfe4-h8-sparse-scale.md
git commit -m "feat: publish H6 prefix and experiment surfaces"
```

### Task 12: Close the H6 source build with focused tests and defer evidence workloads

**Files:**
- Read only: every tracked plan/preregistration/source/config/launcher/test file at the final Task 11 revision.
- Produce: no gate artifact, corpus artifact, benchmark output, checkpoint, test-opening record, or `.verification/` ledger.

**Buildout policy:** Task 12 closes source buildout using focused deterministic tests only. It must not run a full suite, any H4 benchmark, the full 9,720/4,096 Prefix inventories, the 512-replicate SMC grid, corpus acquisition/training, 96-checkpoint scoring, or the one-time test opening. Those are separately authorized exact-revision evidence operations in Tasks 13--14. Focused tests may shrink dimensions, horizons, case counts, particles, and corpus fixtures while preserving the production interfaces; their results cannot be reported as full-gate evidence.

- [ ] **Step 1: Review the bounded source deliverable.** Confirm Tasks 1--11 changed only their mapped source/tests/docs, every launcher retains one editable root `CONFIG`, one `main`, one guard, and no required CLI, and no source path invokes an H4 timing benchmark or evidence workload during import, resolution, or focused tests. Preserve unrelated work and never create or touch `.verification/`.

- [ ] **Step 2: Run only the final focused compatibility tests with shrunken deterministic fixtures.**

```text
python -m pytest tests/promotion/test_h6_prefix_gate.py tests/unit/test_h6_prediction_readiness.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py tests/integration/test_train_vfe4.py -q
```

Expected: PASS on shrunken fixtures. This is focused source verification only, not full Prefix, SMC, corpus, checkpoint, timing, or Prediction evidence.

- [ ] **Step 3: Review lifecycle adapters and consumer synchronization.** Confirm the exact three H6-owned signatures/records, pure projections, keyword-only runner, and empty Prefix predecessor mapping. Confirm the H7/H8 plan consumers were updated and focused-tested against that contract before Task 11 closed; do not describe this docs sync as H6 evidence. Confirm `project_h7_compatibility_config` remains H8-owned.

- [ ] **Step 4: Review the Prefix/Prediction boundary.** Confirm Prefix config, status, preflight, artifact, publication, and focused tests contain only exact H6 source/config/model-family/vocabulary/estimator/data-safety identities. Confirm Prediction readiness alone names exact H1/H2/H3/H5 correctness, H1-prefix-prior, finite-SMC, and Prefix inputs; it neither requires H4 nor calls the deferred H4 benchmark.

- [ ] **Step 5: Record deferred evidence obligations in the preregistration.** List the full Prefix inventories, exact current H1/H2/H3/H5 and H1-prefix-prior inputs, 512-replicate finite-SMC grid, corpus tuning/training, 96 checkpoints, and one-time test opening with their exact-revision identity requirements. Do not execute any of them.

- [ ] **Step 6: Close source buildout.** Report the focused commands actually run and any remaining source defect. Source build completion does not imply Prefix PASS, finite-SMC PASS, corpus completion, Prediction readiness, H6-Prediction, H7, or H8.

### Task 13: Deferred, separately authorized Prefix and Prediction evidence campaign

**Files:**
- Read only: the exact Task 12 source/config/preregistration; Prefix certificates and Prediction prerequisites are produced within this separately authorized task, not by Task 12.
- Produce ignored/external atomic experiment attempts, checkpoints, validation metrics, and failure artifacts only.

**Authorization boundary:** This task is not part of H6 source build completion. Run it only under separate authorization for an exact frozen revision. Prefix may run and publish first from its own identities alone. Prediction readiness separately establishes exact H1/H2/H3/H5, H1-prefix-prior, finite-SMC, Prefix, schedule/matching/estimator/data eligibility before corpus materialization. H4 timing/cost is nonblocking provenance and its stopped benchmark is never rerun.

- [ ] **Step 1: Produce independent Prefix evidence, then validate Prediction readiness before empirical access.** First run/publish full H6-Prefix from its own exact identities with `predecessor_refs={}` and close only its safety ledger. Separately produce exact current H1/H2/H3/H5 correctness, H1-prefix-prior, and full 512-replicate finite-SMC artifacts. Invoke `validate_h6_prediction_readiness` before train materialization or model construction; validate exact revision/digest/manifests/payloads, the actual H5 producer fields/labels, H6-owned schedules, estimator protocols, matrix, blinded access policy, Prefix set, and reconstructed match reports. Missing/stale/non-PASS/nonreproducible input returns no token. H4 is neither produced nor consumed. PASS publishes separate `h6_prediction_readiness.json`; it never alters Prefix closure.

- [ ] **Step 2: Materialize train and reserve the experiment only after readiness PASS.** Pass the opaque readiness token to `materialize_prediction_train`; verify the sealed train and validation identities while decoding, create train/ordinary-validation tensors/windows/schedules, and prove the test handle remains sealed/unmapped. Hash the readiness token, `(git_head,dirty_digest)`, materialized data identities, canonical config, literal matrix, exact current prerequisite/Prefix manifests, H5 binding, common/typed schedules, match reports, finite-SMC and actual-endpoint protocol, 64-stream registry, analysis source, and failure policy. Atomically reserve one experiment root; an existing root is never overwritten. Every later operation revalidates readiness/data hashes before empirical state.

- [ ] **Step 3: Run only the preregistered tuning estimands.** For each of A0--A5, each of the six `(lr, wd)` cells, and tuning seeds `2026072199` and `2026072200`, run exactly a quarter pass with identical frozen batch schedules and validation access. This gives PRIMARY equal tuning for A0/A5, MAP equal tuning for A2/A5, and independent descriptive tuning for A1/A3/A4. Require the exact endpoint/config/estimator/model-family/vocabulary prefix certificate and readiness token before launch and revalidate both hashes before validation scoring. Select the lowest mean validation prior NLL; deterministic tie-break is lower learning rate then lower weight decay. Publish all cells, not only winners. The six nonbase component endpoints in STRUCTURE, PRIOR, MIXTURE, OBJECTIVE, LATENT, and RECOGNITION receive the selected A5-primary `(lr,wd)` exactly and are not separately tuned.

- [ ] **Step 4: Freeze selections and the full endpoint plan before confirmation.** Atomically publish `tuning_selection.json` with every candidate metric/hash and deterministic selection trace. In the same immutable parent set, freeze every literal matrix row's left/right config ID, factory ID, sole changed factor, tuning estimand, match-report SHA, eight seeds, exact prefix-certificate keys, and common global opening group. No confirmatory/test metric can alter a choice, and no matrix endpoint may be substituted or silently dropped.

- [ ] **Step 5: Run confirmatory training for every matrix endpoint.** For each seed `2026072101..2026072108`, run each unique endpoint required by PRIMARY, MAP, STRUCTURE, PRIOR, MIXTURE, OBJECTIVE, LATENT, and RECOGNITION, plus A1/A3/A4 descriptive controls, for exactly two complete passes with shared data-order seed `2026072199`, validations at every twentieth of each pass, terminal checkpoints, and no early/best selection. Each launch checks readiness, prefix certificate, common outer schedule, exact typed phase schedule, H5 binding, and match report. A0/no-latent have no recognition phase/state; latent endpoints use recognition/snapshot/model. PRIMARY/MAP use independently selected settings; component rows use shared A5. Every row must have eight paired terminal checkpoints or the opening remains ineligible.

- [ ] **Step 6: Apply the frozen failure policy.** Publish every attempt. Permit at most one exact infrastructure retry with an unchanged config and proved cursor/checkpoint semantics. Do not replace seeds or retry numerical/model/prefix/capacity/estimator failures. If any required terminal endpoint/seed is absent, mark Prediction INCONCLUSIVE and stop before test opening.

- [ ] **Step 7: Freeze the complete all-or-none terminal checkpoint set.** Verify readiness/current-candidate/config hashes, manifests, data schedules, counted train tokens/model-update opportunities, active phase/update inventories, common/typed schedules and H5 identities, parameter/whole-schedule-FLOP margins, optimizer access, validation boundaries, exact prefix certificates, and the frozen 64-stream/four-particle-level opening protocol. Publish one immutable checkpoint-set SHA containing all 96 endpoint/seed checkpoints. Source and dirty digest remain exactly Task 12.

### Task 14: Deferred, separately authorized single test opening and H6-Prediction closure

**Files:**
- Read only: exact Task 12 source and Task 13 frozen experiment artifacts.
- Produce: one atomic test-opening artifact, final metrics artifact, and `.verification/h6-prediction-<FULL_HEAD>-<EXPERIMENT_SHA>-ledger.json`.

**Authorization boundary:** This irreversible evidence operation is not source buildout and does not run automatically after Task 12. It requires separate authorization after the full Task 13 evidence set exists at the exact frozen revision; otherwise H6-Prediction remains unrun/INCONCLUSIVE without opening test data.

- [ ] **Step 1: Preflight one-opening eligibility for the complete frozen matrix.** Require the exact PASS readiness token and its H1/H2/H3/H5, H1-prefix-prior, finite-SMC, and independent Prefix parents; exact H5 producer binding; H6-owned schedules; critical-value/access-policy/endpoint protocol; matrix/matching/sealed-data hashes; all 96 terminal checkpoints; unchanged revision/config/analysis hashes; the exact stream registry/particle ladder; no prior reservation; and no active marker. H4 is absent. Prove train/tuning never mapped the sealed test handle. If any condition fails, publish Prediction INCONCLUSIVE without model-facing test access.

- [ ] **Step 2: Durably reserve exactly one opening through the sole issuer.** Freeze `ExperimentIdentity` over the complete checkpoint/current-candidate/sealed-data/access-policy/analysis/stream protocol, then call `reserve_and_issue_durable_test_opening_capability(store=store, readiness=readiness, experiment_identity=experiment_identity, reservation_path=reservation_path)` exactly once. That function alone writes/fsyncs the frozen canonical O_EXCL proof bytes, registers the independently retained proof, and privately constructs the opaque capability. `FileExistsError` or any post-reservation crash is terminal Prediction INCONCLUSIVE. Never truncate, replace, rename over, delete, reconstruct, or deserialize a capability. `open_test_for_scoring` must first obtain the sole validator's `ValidatedTestOpening`; blinded storage before that validation is not an opening.

- [ ] **Step 3: Run the complete fixed actual-endpoint assessment in that transaction.** Score every checkpoint at all four particle counts for all 64 common stream IDs using target-blind weighted prior prediction, sequential cold-then-warm cache audit, and corpus-summed token totals. Publish every `log Z_hat`, counted-target total, `Y[c,r,N]`, counter-consumption record, cache audit, checkpoint/stream/particle identity, and failure. The transaction is complete only with exactly `96*64*4=24,576` corpus records. A missing/duplicate/nonfinite record or scorer defect makes Prediction INCONCLUSIVE/FAIL as specified; never reopen or add replicates.

- [ ] **Step 4: Aggregate uncertainty and compute every frozen matrix report.** Recompute all checkpoint means/variances/covariances, `Q0/Q1/Q2/R1/R2`, exactly 352 simultaneous bounds, convergence/bias/random-error gates, and the reported estimator-qualified `Q2` NLLs. For each matrix row compute common-stream `D_i`, `e_i`, all 256 training-seed error-box corners, and the inflated df=7 interval. Apply PRIMARY `lower>delta -> PASS`, `upper<=0 -> FAIL`, else INCONCLUSIVE only to the inflated interval. MAP attribution requires PRIMARY PASS and inflated MAP lower bound above zero. Report STRUCTURE/PRIOR/MIXTURE/OBJECTIVE/LATENT/RECOGNITION with inflated descriptive intervals and A1/A3/A4 checkpoint uncertainty descriptively. Any uncertainty not materially below the frozen limits is INCONCLUSIVE. State explicitly that no H7 result follows.

- [ ] **Step 5: Have fresh reviewers consume artifacts only.** One checks reservation/capability/failure/tuning protocol; one independently recomputes corpus sums, 64-stream/four-level aggregation, variances/covariances, convergence/bias envelopes, 352 simultaneous bounds, and 256-corner intervals; one checks current-prerequisite/prefix/checkpoint/sealed-data/estimator provenance and nonclaims. Reviewers do not rerun training, scoring, or tests. An evidence defect requiring another model-facing test access makes the result INCONCLUSIVE.

- [ ] **Step 6: Start and validate the separate Prediction ledger.** Use one claim per check: exact current-candidate prerequisite set; typed active schedules/no fake phases; blinded/train/test capability boundary; matrix/matching/tuning/checkpoint completeness; durable exclusive reservation/terminal-after-reservation rule; weighted recursion finite gate; exact 24,576-record endpoint inventory; common streams; corpus/checkpoint variance/covariance; convergence assumption and bias envelopes; 352 simultaneous bounds; materially-sub-delta gates; `Q2` aggregation; 256-corner interval inflation; PRIMARY/MAP rules; descriptive labels; atomic provenance; and H7/H8 nonclaims. No Prefix claim is rewritten. Missing eligible evidence is INCONCLUSIVE, never majority-vote closure.

- [ ] **Step 7: Report the empirical evidence revision.** Report `prediction_evidence_revision`, exact `(git_head,dirty_digest)`, experiment/checkpoint-set/reservation/result SHAs, sealed/materialized data/access-policy hashes, finite/endpoint estimator and 64-stream hashes, all uncertainty bounds/failures, uninflated and inflated PRIMARY/MAP intervals, attribution disposition, artifact path, and validated ledger. Completion is a separate immutable `test_opening_result.json`; the reservation remains unchanged. Do not commit generated evidence or edit the preregistration.

## Out of Scope for This Plan

- Any H7 covariance/change-of-variables implementation, residual, or claim.
- Any H8 sparse-scale execution or permission to expand context/parent-set complexity.
- Positive-dimensional base geometry, base connection, curvature, or holonomy.
- Optional independent graph links or graph-holonomy attribution.
- WikiText-103, GPT-2/tiktoken tokenization, long-context language modeling, or post-H8 scaling.
- V3 checkpoint loading, V3 objective semantics, V3 `BeliefState`, or claims that V3 is an exact limit.
- Posterior-predictive reconstruction reported as validation/test perplexity.
- Free-running samples as a substitute for teacher-forced held-out likelihood.
- A claim that smoothing recognition itself predicts held-out tokens, that an emission-only ablation is an ELBO, or that a projection is exact mixture marginalization.
- Hyperparameter rescue, replacement seeds, early stopping, best-validation checkpoint selection, or repeated test openings.
- Running or requiring the stopped H4 timing benchmark as any H6 source-build, Prefix, or Prediction condition.
- Treating the deferred full Prefix inventories, 512-replicate SMC grid, corpus training, 96 checkpoints, or one-time test opening as source-build completion.

## Self-Review of Plan Completeness

- **Spec coverage:** Tasks 1--11 build separate Prefix/Prediction surfaces with explicit hashed DAG rows, exact receiver equality, complete data-safety-bound certificates, immutable tensor snapshots, domain-separated non-self-referential hashes, the actual H5 boundary, H6-owned AdamW schedules, and the frozen independent-Prefix lifecycle contract. Task 11 requires H7/H8 consumer-plan synchronization and focused compatibility before closure but does not claim the current H7 plan already consumes it. Task 12 closes source with focused tests; Tasks 13--14 defer evidence workloads and never rerun H4 timing.
- **Task ordering:** Prefix can run and publish independently from its own identities. Prediction readiness alone consumes exact H1/H2/H3/H5, H1-prefix-prior, finite-SMC, and Prefix evidence before corpus materialization. Source buildout ends at Task 12 without a broad suite, gate, training run, checkpoint grid, or opening. Under separate authorization, tuning precedes confirmation, all checkpoints precede the durable opening, and Prediction closure never mutates Prefix evidence.
- **Type consistency:** `CausalDagRow`/`CausalDag` bind receiver edges intrinsically; structure receiver labels equal DAG receivers. `FrozenTensorSnapshot` owns every public tensor-bearing result. `PrefixCaseKey` includes data safety and `PrefixCertificate` binds the complete fail-closed validation record. Each record's single owned integrity digest excludes only itself and includes verified reference digests; content/reference digests are independently verified against their named external bytes/preimages. The independent lifecycle signatures remain consistent across H6 and the required synchronized H7/H8 consumer docs.
- **Placeholder scan:** Every protocol choice that affects evidence is fixed here or has an exact pre-outcome measurement-and-freeze procedure. No threshold, seed, status rule, estimator algorithm, dataset substitution, or statistical decision is selected after predictive outcomes.
- **Path check:** The plan is saved at `docs/superpowers/plans/2026-07-21-vfe4-h6-prefix-prediction.md`. Implementation must use a fresh dedicated branch/worktree, preserve the user's live/WIP, and follow the bounded commit sequence above. This authoring task itself performs no code change, test, training run, network action, or commit.
