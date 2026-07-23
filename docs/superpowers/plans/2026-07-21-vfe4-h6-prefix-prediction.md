# VFE 4.0 H6 Prefix Safety and Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the bounded zero-dimensional H6 language sidecar, prove its finite and statically audited prefix-safety contract for every exact predictor configuration, and only then run a separately revision-bound, compute-matched WikiText-2 prediction experiment.

**Architecture:** H6 is two evidence products, not one gate. `H6-Prefix` verifies the target-blind public prior-predictor boundary, source support, cache behavior, and static dataflow at an exact source/config/estimator identity; `H6-Prediction` consumes those immutable certificates and separately evaluates six matched arms with a frozen target-blind filter/SMC scorer. The language specialization remains a labeled causal population over the singleton base `C0={*}`; the internal causal DAG is a probability graph and never a surrogate bundle base.

**Tech Stack:** Python 3.10+, PyTorch, deterministic CPU float64 for H6-Prefix and estimator validation, byte-level WikiText-2 caches, NumPy independent oracles, SciPy-free frozen critical values, pytest, SHA-256 provenance, atomic JSON/checkpoint publication, JUnit XML.

## Global Constraints

- The normative theory is `Manuscripts/vfe4_whitepaper/02_observations_related_work.tex`, `04_generative_model.tex`, `05_structured_information_form.tex`, `07_transformer_crosswalk.tex`, `08_hypotheses_limitations.tex`, and `09_appendices.tex`. The corresponding Research WIP files were byte-identical when this plan was authored. The relevant wiki context is `[[VFE Transformer Program]]`, `[[Inference machinery -- variational EM and filtering]]`, and `[[vfe-population-generative-status-2026-07-12]]`; V3 supplies rough launcher/data/artifact mechanics only.
- The implemented geometric base is exactly the singleton `C0={*}`. Token positions are labeled population copies over that point. `CausalDag` is stored separately from `ZeroDimensionalBase`; neither a token edge nor a cache transition is base transport, base curvature, or base holonomy.
- H6 does not establish H7. A5 may use the declared typed same-point internal-map restriction, and A2 may replace it with a matched generic map, but neither H6 result is evidence of population-frame covariance. H7 and every covariance label remain unimplemented/unverified until the independent H7 plan passes.
- `H6-Prefix` and `H6-Prediction` have different schemas, artifact roots, evidence revision identifiers, claim ledgers, and closure decisions. Prefix runs and publishes solely from its own exact source/config/model-family/vocabulary/estimator/data-safety identities. H1--H5 status, predecessor publication, estimator-accuracy evidence, the H6 training schedule, arm matching, tuning, checkpoints, and predictive outcomes are not Prefix inputs, preflight conditions, status terms, publication conditions, or ledger claims.
- Prediction cannot be reported, launched, resumed, or scored unless every exact arm/config/estimator/model-family/vocabulary tuple it consumes has a PASS prefix certificate. A missing, stale, FAIL, or INCONCLUSIVE certificate blocks that tuple and makes the aggregate prediction result INCONCLUSIVE. Checkpoint hashes are bound separately in empirical provenance and may not alter the certified predictor safety contract.
- Task 11 completes the tracked H6 source surface with focused RED/GREEN commands only. Task 12 performs a source-build closeout using focused deterministic fixtures and records the deferred evidence operations; it does not run a broad suite, H4 timing benchmark, large estimator grid, corpus training, test opening, or `.verification/` lifecycle. The full Prefix and Prediction evidence revisions are later, separately authorized operations at a frozen `(git_head, dirty_digest)` produced by `vfe4.artifacts.provenance.dirty_content_digest`.
- H6-Prefix publication preflight validates only the exact H6 source/config/model-family/vocabulary/estimator/data-safety identities and complete Prefix case/static-audit inventory for that evidence revision. It neither reads nor references an H1--H5 artifact, and it cannot be blocked by any H1--H5 status or publication state.
- Before any empirical Prediction split materialization/access, tuning, training, validation scoring, checkpointing, or test scoring, Prediction readiness may require exact current H1, H2, H3, and H5 correctness artifacts plus the exact H1-prefix-prior, finite-SMC, and H6-Prefix inputs used by the selected Prediction matrix. H4 correctness/timing/cost evidence is not a Prediction prerequisite; the frozen green H4 correctness provenance (`911` tests, zero failures/errors/skips) may be referenced as nonblocking history, while its deferred timing benchmark must never be triggered by H6.
- Prefix-conditioned-prior variants consume a separate current-candidate H1 rerun artifact keyed to the exact prefix-prior generative-factor/config schema. The bounded SMC recursion gate likewise has a separate current-candidate artifact. These full evidence artifacts are produced only by the deferred Prediction-evidence operation, never by source buildout. The H1 variant does not replace or mutate ordinary H1/H2/H3/H5 correctness evidence, and fixed-prior variants do not require it.
- H6 owns the complete immutable `H6TrainingSchedule`, including AdamW class, optimizer policy, and phase names/order. H5 recognizes only `exact_coordinate`, `generalized_em`, and `natural_gradient_proposal`; those labels and the exact H5 producer fields are correctness provenance for Prediction readiness, never names or certifications of H6 Adam/AdamW phases, schedule composition, repetition count, optimizer behavior, or monotonicity.
- H6 uses separate artifacts. An H6-Prefix artifact contains no predecessor reference and publishes only H6 identities, validation, certificates, and manifest. Deferred Prediction readiness references exact current H1/H2/H3/H5, H1-prefix-prior, finite-SMC, and H6-Prefix evidence without copying payloads. No unified H1--H6 validation payload is created.
- The bounded corpus is the official WikiText-2 raw archive at exactly `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip`. The exact downloaded archive bytes, the three extracted raw member bytes, tokenizer specification, encoded streams, and window manifests are SHA-256 bound. Training never silently substitutes WikiText-103, a prepared vocabulary, synthetic text, or another mirror.
- Archive preparation accepts at most `16,777,216` compressed bytes; exactly one directory entry and the three files `wikitext-2-raw/wiki.train.raw`, `wikitext-2-raw/wiki.valid.raw`, and `wikitext-2-raw/wiki.test.raw`; compression methods ZIP_STORED (`0`) or ZIP_DEFLATED (`8`) only; positive per-member compressed/uncompressed sizes at most `16,777,216`; total uncompressed bytes at most `33,554,432`; and compression ratio at most `100`. It rejects encryption, links, duplicate/case-colliding paths, extra files, path traversal, CRC mismatch, size mismatch, and decompression beyond a bound. The observed archive/member byte sizes, compression methods, CRC32 values, and SHA-256 values are copied into config/preregistration before evidence and must match during streaming extraction.
- Data access has one capability boundary. Before Prediction readiness, the acquisition path may download, bounded-stream-validate, hash, and seal all three raw members and may materialize only the frozen validation safety fixture. It returns hashes/metadata and opaque sealed train/validation/test handles, never train tensors/windows, ordinary validation tensors/windows, or model-facing test bytes. After readiness PASS, `materialize_prediction_train` may decode/materialize train tokens, train windows, batch schedules, and ordinary validation data from the sealed train/validation handles. Only after the durable `O_EXCL` test-opening reservation may `open_test_for_scoring` map/decode the sealed test token bytes for a model. Training, tuning, and analysis modules cannot import or call the unsealing primitive; static call-graph/capability tests prove preprocessing cannot expose test targets to them.
- The H6 tokenizer is fixed: raw UTF-8 bytes map to IDs `0..255`, `BOS=256`, `EOS=257`, vocabulary size `258`, and ignored target padding `-100`. Each split is encoded independently as `[BOS] + exact_raw_member_bytes + [EOS]`; bytes and newline sequences are not normalized. No learned state crosses splits.
- Sequence length and stride are both exactly `32`. A window uses `inputs=tokens[start:start+32]` and `targets=tokens[start+1:start+33]`; the last partial window is included once, fills unused inputs with `BOS`, and fills unused targets with `-100`. Token counts always count targets not equal to `-100`.
- WikiText-103 and the GPT-2 tokenizer are reserved until after H8. They do not appear as supported H6 configuration values, fallback paths, tests presented as H6 evidence, or secondary experiment arms.
- Training uses smoothing recognition as the primary regime and filtering recognition as a required ablation. Held-out validation/test scoring uses only the causal generative prior predictor. Recognition may consume the current target or complete observed window during training, but no recognition object, activation, parameter, target, or suffix may enter the prior predictor.
- The public bound call is exactly `next_token_log_probs(prefix_tokens, estimator_rng, cache=None)`. Its bound signature contains those three parameters in that order and has no target, suffix, full-window, recognition, posterior, or reconstruction parameter. `PriorPrediction.log_probs` has shape `(V,)`, where `V` comes from an immutable `VocabularyIdentity` included in the predictor, cache, prefix key, and artifact. WikiText-2 uses `V=258`; no generic interface hardcodes 258.
- Every source normalization in both state and model banks calls the single shared `masked_log_softmax_from_parents(logits, declared_parents, receiver_t)`. It derives the Boolean mask only from declared parents satisfying `j<t`, writes exact `-inf` before normalization, rejects an empty/all-invalid row with `AllInvalidSourceRowError`, and returns exact zero mass outside support. Post-softmax masking, renormalization in another helper, direct `softmax`/`log_softmax` in a source-prior module, or unresolved/dynamic dispatch is respectively FAIL or INCONCLUSIVE; no alternate normalization path exists.
- `T_mask` identities are sorted `MaskCaseKey(fixture_sha256, vocabulary_sha256, predictor_config_sha256, model_family_sha256, prior_variant, bank, receiver_t, context_sha256)`. The separate production-path `h6-prefix-small-v1` fixture has vocabulary 3, horizon 4, and parent rows `((0,), (0,1), (0,1,2), (0,1,2,3))` for both banks. Its fixed-prior manifest has exactly 4 contexts per bank; its prefix-conditioned manifest has exactly `2*(1+3+9+27)=80` contexts per bank (all token prefixes times the two frozen latent-history contexts `zero` and `seeded`). Across both banks/variants the small base inventory is `4+4+80+80=168` mask cases.
- The WikiText-2 property manifest has exactly 4,096 contexts for each active `(prior_variant, bank)` cell: fixed/state 4,096, fixed/model 4,096, prefix/state 4,096, prefix/model 4,096, for a two-bank/two-variant base inventory of 16,384. A model family with no bank has no fake mask row; A4 contributes only its state bank; A0/A3 contribute none. The gate records exact per-configuration counts and the sorted manifest SHA.
- H6-Prefix uses deterministic CPU float64. The leakage allowance is exactly `0`, the mask allowance is exactly `0`, and equality is stricter than tensor-value equality: compare dtype, shape, device, contiguous raw storage bytes viewed as `uint8`, and SHA-256. This preserves signed-zero distinctions. Metadata or raw-byte differences FAIL even if numeric comparison would call `+0.0` and `-0.0` equal.
- The exhaustive leakage fixture is the separate `h6-prefix-small-v1` production model path with `V=3`, `T=4`; it is not a size-mutated WikiText-2 identity. For each one-indexed position `t`, enumerate every prefix in `V^(t-1)` and every ordered pair of tails in `V^(5-t) x V^(5-t)`, including equal tails. The exact count is `sum_t 3^(t-1) * 3^(2*(5-t)) = 9,720` comparisons per certified model-family/estimator profile.
- The language-property fixture contains exactly `4,096` WikiText-2 validation perturbations at `V=258`, generated from identity-bound validation windows with seed `2026072197`. Each pair preserves `x_<t` and varies current targets/suffixes independently. The case file records generator version, `VocabularyIdentity`, and exact validation-token/window hashes.
- Every dynamic prefix case runs cold-cache, warm-cache, and reverse-order/cache-rebuild modes. Compared calls use common counter-based estimator streams. Cache results must equal cold recomputation exactly and must be independent of case traversal order.
- H6-Prefix also requires a static import/signature audit, a taint/dataflow audit from targets/suffixes/recognition to generative sinks, and a cache-key/value audit. A universal source claim is never inferred from the finite cases alone.
- H6-Prefix status is PASS only when every exact case, source-mask check, cache check, signature/import rule, taint obligation, and artifact identity passes. A witnessed violation is FAIL. Missing fixtures, unsupported dynamic dispatch, incomplete taint coverage, absent hashes, or an unauditable cache is INCONCLUSIVE.
- Prediction arms are fixed: A0 conventional autoregressive baseline; A1 ordinary latent sequence model without the typed internal-map sector; A2 capacity-matched generic non-equivariant maps; A3 fixed immediate-predecessor/source-free model; A4 model-channel-free model; A5 full VFE 4.0 H6 model.
- All six arms consume identical encoded tokens, batch/window order, pass count, model-update opportunities, validation boundaries, checkpoint boundaries, and test-opening transaction. Their trainable parameter counts are within `1%` of the A5 reference, their counted whole-schedule training FLOPs are within `5%`, every active trainable parameter is present exactly once in its declared optimizer, and no arm uses dormant/no-op/filler parameters or phases.
- `H6TrainingSchedule` is one hashed common outer schedule plus a hashed typed phase schedule for every exact endpoint. The H6 record itself fixes AdamW class, betas, epsilon, clipping/decay policy, batches, passes, model-update opportunities, validation/checkpoint boundaries, and failure semantics. A0 and every `latent_enabled=false` endpoint use only `model_ce_adamw`; they construct no recognition object/optimizer and receive no fake recognition step. Every latent endpoint uses exactly `recognition_adamw -> immutable_detached_snapshot -> model_adamw` once per batch. FLOP matching counts the actual active phases, so extra latent inference/update work is matched structurally rather than hidden by no-ops. H5's enabled labels remain exactly `exact_coordinate`, `generalized_em`, and `natural_gradient_proposal`; none is renamed to Adam or used to certify the H6 schedule or its monotonicity.
- Required factorial reports are structured versus population-factorized recognition, fixed versus prefix-conditioned generative source prior, exact source mixture versus the declared projection, complete-ELBO versus emission-only training, latent enabled versus disabled, and smoothing versus filtering training. Each comparison changes only its named factor on the same arm factory/config family.
- Prefix-conditioned source priors are a new normalized generative model. Prediction readiness requires the separate exact H1 rerun PASS artifact for those variants. That prerequisite is not part of Prefix safety closure. Emission-only is labeled an ablation, not another ELBO. Projection is labeled approximation and records projection error; it is never called exact mixture marginalization.
- Before empirical scoring, freeze and validate the weighted bootstrap filter/SMC estimator specified below: 256 particles for the bounded finite gate, carried normalized float64 log weights, systematic resampling after observation only when ESS is below `0.5 * particle_count`, `logsumexp` normalization, and counter-based streams. The proposal is exactly the causal generative source/transition law, so no omitted proposal correction exists. An unweighted emission average is forbidden whenever carried weights are nonuniform. The finite `V=3,T=6` gate validates recursion only; it cannot close actual WikiText-2 checkpoint estimator error.
- Tuning is the equal grid `learning_rate in {1e-4, 3e-4, 1e-3}` by `weight_decay in {0, 1e-2}` for every arm, using exactly two quarter-pass runs per cell. The specified tuning/train seed is `2026072199`; because the source protocol fixed two tuning seeds but named only one, freeze the adjacent independent companion `2026072200` in the preregistration before any tuning. This explicit resolution is not evidence from outcomes.
- Confirmatory initialization/run seeds are exactly `2026072101..2026072108`. The shared data-order seed is `2026072199`. Actual test scoring uses the frozen 64-entry common paired stream registry derived from root `2026072198`, never one selected estimator stream. No replacement seed or adaptive replicate is permitted.
- A quarter pass is the first `ceil(number_of_training_batches / 4)` batches of the frozen pass permutation. A full pass is every training window exactly once. Confirmatory runs execute exactly two full passes and never early-stop or select a best-validation checkpoint.
- The phrase “validation every twentieth pass” is operationalized as validation at every twentieth of a corpus pass: boundaries `ceil(k * batches_per_pass / 20)` for `k=1..20`, deduplicated while preserving order, on each of two passes. This is the only reading compatible with both “two full passes” and periodic validation; it is frozen before outcomes.
- The test split is opened once, globally, after tuning choices, all eight-seed terminal checkpoints, prefix certificates, analysis code hashes, and the complete actual-endpoint SMC protocol are frozen. Blinded acquisition/hash and sealed storage are not a model-facing opening. The irreversible opening begins only after the durable `O_EXCL` reservation and unsealing capability are recorded; that one transaction scores every endpoint/checkpoint across the complete 64-stream, four-particle-count assessment or scores none. Validation does not choose early checkpoints.
- An infrastructure failure may receive one exact retry only when the attempt artifact proves no optimizer/checkpoint state advanced or proves an exact checkpoint restore. Numerical divergence, nonfinite loss, estimator failure, model failure, prefix failure, capacity mismatch, or a missing pair is not infrastructure and receives no replacement run. Any incomplete paired seed set makes the affected decision INCONCLUSIVE.
- The primary metric is the actual-endpoint SMC-qualified corpus-summed, token-counted prior negative log likelihood in nats/token defined below. Per-batch means are not averaged. Perplexity is `exp(NLL)` and is secondary.
- The primary paired contrast is `d_i = NLL_A0,i - NLL_A5,i` over the eight confirmatory seeds. The practical threshold is `delta = -log(0.99) = 0.01005033585350145`. The training-seed interval uses the frozen `t_(0.975,7)=2.364624251592784`, then is conservatively enveloped over all estimator-error boxes as defined below. Lower bound greater than `delta` is PASS; upper bound less than or equal to `0` is FAIL; every other complete result is INCONCLUSIVE.
- Report the paired `NLL_A2 - NLL_A5` interval separately. Attribution to the typed map restriction is permitted only if the primary A0--A5 result passes and this secondary interval has lower bound greater than zero. It is not an H7 covariance inference.
- Deferred evidence artifacts are atomic and identity-bound. Independent Prefix validation and Prediction-only H1/H2/H3/H5, H1-prefix-prior, finite-SMC, readiness, train materialization, tuning, typed-phase attempts, checkpoints, validation scoring, immutable test reservation, endpoint records, uncertainty aggregation, and final metrics each have immutable manifests with `git_head`/`dirty_digest` plus applicable config/data/access/estimator/RNG/parent hashes.
- Each implementation task runs only its named focused RED/GREEN commands on deterministic shrunken fixtures. Do not run cumulative/broad tests, gates, training, timing, or test opening during source buildout. Exact-revision Prefix/Prediction artifacts and ledgers are created only under the separate Task 13--14 authorization; a later source change invalidates affected evidence and requires a new evidence revision, never an in-place artifact patch.
- Source reviewers consume focused outputs only. Separately authorized evidence reviewers consume exact-revision artifacts, manifests, and claim ledgers without rerunning training, scoring, the H4 benchmark, or other evidence workloads for confidence.
- Preserve `.verification/ledger.json` and every prior revision-specific ledger byte-for-byte. H6-Prefix uses `.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json`. H6-Prediction uses `.verification/h6-prediction-<FULL_HEAD>-<EXPERIMENT_SHA>-ledger.json`. Never overwrite or repoint an existing ledger; a replacement revision gets a new path.

---

## File Map and Dependency Boundaries

| Path | Responsibility |
|---|---|
| `vfe4/types/h6.py` | Immutable, separately hashed `ZeroDimensionalBase`, `CausalDag`, H6 language structure, arm, data identity, predictor/cache, prefix certificate, estimator, ELBO, NLL, checkpoint, attempt, and decision records. |
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
| `verify_vfe4.py` | Extend the one-editable-dictionary/one-main/no-required-CLI verifier through independent H6-Prefix and own the pure H1-prefix-prior/H6-Prefix projections plus generic current-candidate runner consumed by H7. |
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
| `tests/integration/test_verify_vfe4.py` | Existing click-run integration extended through independent H6-Prefix plus the three H6-owned H7 lifecycle adapters, with no Prefix predecessor read/rerun/copy. |

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
class CausalDag:
    labeling: Literal["zero_based"]
    node_labels: tuple[int, ...]
    parent_rows: tuple[tuple[int, ...], ...]
    canonical_sha256: str

@dataclass(frozen=True)
class H6LanguageStructure:
    base: ZeroDimensionalBase
    dag: CausalDag
    receiver_labels: tuple[int, ...]
    structure_sha256: str

@dataclass(frozen=True)
class H6FactorTerm:
    receiver_t: int
    partition: Literal[
        "emission", "initial", "state_source", "model_source",
        "state_transition", "model_transition", "entropy"
    ]
    factor_identity_sha256: str
    value: torch.Tensor

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
    complete_decomposition: torch.Tensor
    total_language_elbo: torch.Tensor
    equality_checked: Literal[True]
    canonical_sha256: str

@dataclass(frozen=True)
class EmissionOnlyAblationTerms:
    objective_kind: Literal["emission_only_ablation_non_elbo"]
    ordered_emission_terms: tuple[H6FactorTerm, ...]
    total: torch.Tensor
    canonical_sha256: str

class PriorPredictor(Protocol):
    def next_token_log_probs(
        self,
        prefix_tokens: torch.Tensor,
        estimator_rng: EstimatorRng,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction: ...

@dataclass(frozen=True)
class PriorPrediction:
    vocabulary: VocabularyIdentity
    log_probs: torch.Tensor          # shape (vocabulary.size,)
    cache: PrefixCache
    estimator_record: EstimatorRecord

@dataclass(frozen=True)
class PrefixCaseKey:
    arm: ArmId
    predictor_config_sha256: str
    estimator_sha256: str
    model_family_sha256: str
    vocabulary_sha256: str
    git_head: str
    dirty_digest: str

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
class BlindedCorpusStore:
    data_identity_sha256: str
    sealed_train_handle: SealedSplitHandle
    sealed_validation_handle: SealedSplitHandle
    frozen_validation_fixture: ValidationSafetyFixture
    sealed_test_handle: SealedSplitHandle

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

`ZeroDimensionalBase` accepts exactly `base_id="C0"`, `points=("*",)`, and `dimension=0`; its canonical hash excludes no field. `CausalDag` is separately canonicalized and accepts only explicit zero-based integer node labels with no gaps or duplicates. `H6LanguageStructure.receiver_labels` is explicit, unique, strictly increasing, and names every receiver row exactly once. For every `(receiver_t, parents)` pair, parents are unique declared nodes and satisfy strict `j < receiver_t`; self/future parents, duplicate parents, missing/duplicate receivers, a one-based alternative, and any labeling that could be interpreted as either zero- or one-based are rejected. Token/DAG edges never enter base transport, curvature, or holonomy records.

`TrainingPhase` is a closed H6 enum with exactly `MODEL_CE_ADAMW`, `RECOGNITION_ADAMW`, `IMMUTABLE_DETACHED_SNAPSHOT`, and `MODEL_ADAMW`. An endpoint with `latent_enabled=false` has phases `(MODEL_CE_ADAMW,)` and zero recognition updates. A latent endpoint has phases `(RECOGNITION_ADAMW, IMMUTABLE_DETACHED_SNAPSHOT, MODEL_ADAMW)` and one recognition update. These are H6 schedule labels, not H5 labels. Any other tuple, phase reordering, dummy phase, recognition object on a no-latent endpoint, or mismatch between the phase schedule and endpoint config is rejected during resolution.

The three lifecycle adapters above are H6-owned public compatibility interfaces used by H7. Both projections are pure and never mutate the one editable root `CONFIG`. `project_h6_prefix_config` includes only H6 Prefix identities and has no predecessor/PASS input. `run_projected_current_candidate` requires `predecessor_refs == {}` for `H6-Prefix`; a nonempty mapping is rejected rather than recorded as Prefix provenance. H8's `project_h7_compatibility_config` is not implemented or owned by H6.

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

Every row below uses all eight confirmatory seeds, one terminal checkpoint per endpoint/seed, exact endpoint prefix-certificate keys, and the all-or-none global test opening. Every endpoint must satisfy the 1% parameter, 5% FLOP, and exact optimizer-access checks; otherwise that row is ineligible/INCONCLUSIVE rather than relaxed. “Shared A5” means both endpoints use the A5-primary selected `(lr,wd)` and estimates a factor intervention conditional on that optimizer setting; it is not unequal tuning disguised as an architecture-wide optimum.

| ID | Left exact config / factory | Right exact config / factory | Sole config factor changed | Hyperparameter estimand | Interpretation |
|---|---|---|---|---|---|
| `PRIMARY` | `h6-a0-ar-v1` / `build_a0@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Whole declared architecture | Equal six-cell tuning per endpoint | Primary A0--A5 predictive contrast; not component attribution. |
| `MAP` | `h6-a2-generic-map-v1` / `build_a2@h6-arm-v1` | `h6-a5-structured-fixed-exact-complete-latent-smoothing-v1` / `build_a5@h6-arm-v1` | Generic map versus typed same-point map restriction | Equal six-cell tuning per endpoint | Conditional map attribution only; never H7 covariance. |
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
- Produces: `ZeroDimensionalBase`, `CausalDag`, `H6LanguageStructure`, `ArmId`, `EvidenceStatus`, `VocabularyIdentity`, `PrefixCaseKey`, `PrefixCertificate`, `PredictionCorrectnessArtifactRef`, `H1PrefixPriorArtifactRef`, `SmcAccuracyArtifactRef`, exact-producer-field `H5UpdateBinding`, `TrainingPhase`, `H6OuterSchedule`, `H6ArmPhaseSchedule`, `H6TrainingSchedule`, `H6PredictionReadinessToken`, blinded/materialized/test-opening data capabilities, `EstimatorSpec`, endpoint-SMC protocol types, `DataIdentity`, `CheckpointIdentity`, `NllTotals`, `PredictionDecision`, conditional H6 sections on existing `ResolvedConfig`, and the explicit `H6PrefixGateResult` / `H6PredictionResult` union members.
- Consumes: existing `GateStatus`, H1/H2/H3/H5 result types for Prediction only, and existing `resolve_config`; earlier result records remain unchanged. It does not consume H4 or any predecessor result for Prefix.

- [ ] **Step 1: Write failing immutable-type and strict-config tests.** Require a separately hashed exact-one-point `ZeroDimensionalBase`, a separately hashed explicitly zero-based `CausalDag`, and an `H6LanguageStructure` that owns all receiver labels. Accept only complete ordered receiver rows with unique parents satisfying `j < receiver_t`; reject self/future parents, duplicate parents, missing/duplicate receivers, gapped node labels, one-based labels, and ambiguous mixed labeling. Require all six arm IDs, vocabulary identities, exact seeds, exact tokenizer/window/access-capability values, the 64-stream/four-particle-level endpoint estimator, exact grid, H6-owned AdamW class/policy, typed per-endpoint phase schedules, two/quarter and eight/two-pass schedules, three status states, unknown-key rejection, no H7/H8 dataset values, no CLI-only fields, and round-trip canonical hashes. A0 and every no-latent endpoint resolve only the H6 model-CE AdamW phase and zero recognition updates; every latent endpoint resolves H6 recognition-AdamW/snapshot/model-AdamW in order. Reject no-op/filler phases, a recognition object for no-latent arms, an H5 label used as an H6 phase, and a phase/config mismatch. Preserve every existing operation unchanged; independent `H6-Prefix` requires only its own conditional section and identities, while Prediction requires exact current H1/H2/H3/H5, H1-prefix-prior/SMC/Prefix, estimator/matrix/schedule/access references. Reject H4 as a required Prediction reference.

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

- [ ] **Step 3: Extend the existing schema, resolver, and explicit result union.** Keep Prefix and Prediction conditional sections distinct inside the one canonical config. Prefix canonical JSON/hash contains the structural, vocabulary, source, model-family, estimator, fixture/data-safety, mask-manifest, case-inventory, static-audit, and publication identities only; it contains no H1--H5 status/reference field. Prediction canonical JSON additionally includes exact current-candidate H1/H2/H3/H5 references, H1 variant/SMC/Prefix references, the H5 exact fields `update_spec_raw_sha256`, `update_spec_canonical_sha256`, `objective_schema_sha256`, `factor_input_schema_sha256`, `reference_sha256`, `recognition_state_sha256`, `model_state_sha256`, and `validation_payload_sha256`, common/typed H6 AdamW schedules, data capabilities, the 64-stream/four-level endpoint protocol, arm matrix, retry count, and O_EXCL test-opening policy. H5 enabled labels are restricted to `exact_coordinate`, `generalized_em`, and `natural_gradient_proposal` and never populate H6 phase names. Absent conditional sections contribute nothing to other operation hashes. `EvidenceStatus` accepts only PASS/FAIL/INCONCLUSIVE; missing obligations cannot become PASS. Do not add a nonexistent predecessor dependency-closure field or create a parallel config parser.

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
- Modify: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`

**Interfaces:**
- Produces: `ByteTokenizerV1`, `acquire_wikitext2_blinded(config) -> BlindedCorpusStore`, `materialize_validation_safety_fixture`, `materialize_prediction_train(store, readiness)`, `open_test_for_scoring(store, opening)`, `CausalWindows`, and `FrozenBatchSchedule`.
- Consumes: `DataIdentity` and the canonical data config from Task 1.

- [ ] **Step 1: Write failing archive, capability, and exposure tests with a synthetic fixture.** Assert exact URL, archive/member/count/size/method/ratio/encryption/path/CRC limits, exact byte mapping, one BOS/EOS per split, no UTF-8 decoding or newline normalization, changed-byte/hash rejection, bounded streaming extraction, split-independent hashes, exact final padding, stride/length, counted targets, and stable identities. Before readiness, require only opaque sealed train/test handles plus the frozen validation-safety fixture; attempts to request train tensors/windows or test bytes/tensors fail before opening a file. A forged/wrong-revision readiness token cannot materialize train. A forged/non-durable/wrong-experiment opening capability cannot unseal test. Include a test proving the validation fixture is unchanged when train bytes change and AST/import tests proving training/tuning/analysis cannot import the private unsealer.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_h6_byte_tokenizer.py tests/unit/test_h6_wikitext2.py tests/unit/test_h6_data_access.py tests/unit/test_h6_windows.py -q`. Expected: FAIL on missing data/capability modules.

- [ ] **Step 3: Implement blinded acquisition and sealed publication.** Allow only the exact `wikitext-2-raw/` entries and bounds in Global Constraints; reject traversal, duplicate/case-colliding members, links, encryption, unsupported methods, central-directory/streamed size or CRC disagreement, and missing/extra files. Hash archive/raw/encoded-token identities in bounded streaming mode, but return no raw/token content. Atomically publish sealed train, validation, and test members, `validation_safety_fixture.bin`, `data_identity.json`, an access-policy hash, and `manifest.sha256`. Do not publish model-readable `train.tokens.bin`, ordinary-validation token/window files, or `test.tokens.bin` in this phase.

- [ ] **Step 4: Implement capability-gated materialization, windows, and schedules.** `materialize_prediction_train` verifies the exact readiness `git_head`, `dirty_digest`, data identity, and manifest before decoding train/validation and creating any train tensor/window. `open_test_for_scoring` accepts only the durable reservation-derived capability and maps the sealed test stream read-only; it is the sole model-facing test path. Hash each window manifest from `(split_token_sha256, seq_len, stride, starts, real_target_counts, padding_policy)`. `FrozenBatchSchedule` takes the shared data-order seed and pass index; its digest is embedded in every training attempt/checkpoint. Evaluation order is sequential and complete.

- [ ] **Step 5: Freeze the acquisition protocol without fetching corpus data.** Record the official URL, bounds, member allowlist, tokenizer/access-policy schemas, and measured-value slots in typed config/preregistration. Source buildout uses only synthetic archive fixtures. Actual download, measured archive/member/token hashes, and sealed official handles are produced only by the separately authorized Task 13 evidence operation; no substitute dataset is allowed.

- [ ] **Step 6: Run focused GREEN.** Run the Step 2 command. Expected: PASS on synthetic identity fixtures only; do not access an official/local corpus cache.

- [ ] **Step 7: Commit.**

```text
git add vfe4/data vfe4/config/schema.py vfe4/config/resolve.py tests/unit/test_h6_byte_tokenizer.py tests/unit/test_h6_wikitext2.py tests/unit/test_h6_data_access.py tests/unit/test_h6_windows.py docs/preregistrations/2026-07-21-h6-prefix-prediction.md
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

- [ ] **Step 1: Write failing access/normalization/term tests.** Prove smoothing may use the whole observed window, filtering at position `t` may use `x_<=t` but no suffix, structured versus factorized are separate families, and exact mixtures versus projections return different tagged types. Require `H6LanguageElboTerms` to carry the exact horizon; a deterministically ordered factor term for every `(receiver_t, partition, factor_identity_sha256)`; explicit emission, initial, state-source, model-source, state-transition, model-transition, and entropy partitions; no duplicate or missing factor identity; and an exact checked equality between `complete_decomposition` and `total_language_elbo`. Reject the existing two-time-step `ElboTerms` type at this boundary. Test emission-only returns `EmissionOnlyAblationTerms` with `objective_kind="emission_only_ablation_non_elbo"` and cannot be passed where `H6LanguageElboTerms` is required.

- [ ] **Step 2: Run focused RED.** Run `python -m pytest tests/unit/test_h6_language_recognition.py tests/unit/test_h6_language_elbo.py -q`. Expected: FAIL on missing language recognition/objective.

- [ ] **Step 3: Implement the families and horizon-indexed canonical assembler.** Reuse the sparse precision protocol; do not form a production dense inverse. Exact source treatment retains a normalized mixture. Projection returns a record with source mixture identity, projected moments, projection method, and error diagnostics. Build immutable `H6FactorTerm` entries in canonical horizon/partition/factor-identity order, derive each named partition from that same ordered tuple, compute the complete decomposition from all partitions exactly once, compare it to the independently accumulated total language ELBO under the frozen equality rule, and reject any missing/duplicate/mismatched term. Do not adapt or alias the existing two-time-step `ElboTerms`. Keep the primary training config on smoothing and the emission-only return type explicitly outside the ELBO API.

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

- [ ] **Step 1: Write failing signature and target-blind tests.** Inspect the bound method for exactly `prefix_tokens`, `estimator_rng`, `cache`; assert imports contain no recognition/objective/training module; compare repeated common-stream calls by dtype/shape/device plus contiguous raw `uint8` bytes/hash; reject a cache with wrong prefix/vocabulary/config/model-state/estimator digest; require `(3,)` and `(258,)` normalized outputs on their distinct vocabulary identities; distinguish signed zero; and test that the target is not read before scoring.

- [ ] **Step 2: Write failing independent estimator/constant tests on a shrunken deterministic grid.** Raw-hash the four frozen finite-model schemas, but use a small fixed subset of cells/seeds/particles to test carried nonuniform weights, weighted mixtures, incremental weights, `log Z_hat_t`, normalization/resampling order, ESS, systematic ancestors, cache state, cold/warm replay, and counters. Unit-test the complete 76-cell/512-replicate inventory formulas, df=511, familywise allocation, literal constants, thresholds, and status boundaries without executing that grid. Assert no SciPy import and fail on a one-ULP constant mutation. The full seeds `2026072300..2026072811` run only in deferred Task 13 evidence.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_h6_prior_predictor.py tests/unit/test_h6_predictive_cache.py tests/unit/test_h6_critical_values.py tests/oracle/test_h6_smc_oracle.py -q`. Expected: FAIL on missing predictive/critical-value modules.

- [ ] **Step 4: Implement the exact weighted bootstrap recursion.** Implement the equations in “Frozen Weighted SMC Recursion and Accuracy Gate” literally: propagate each history from the generative proposal while carrying its normalized parent weight; return `logsumexp(log_weight + emission_log_prob)` for each vocabulary item; store pending particles/emission rows/weights; assimilate only the newly appended formerly scored token; add the incremental log normalizer; normalize; compute ESS; then resample ancestors and reset weights only below `N/2` (128 for the finite gate's `N=256`). Use counter keys `(stream_seed, prefix_digest, position, purpose, particle_index)` so paired arms/cases share streams without global RNG dependence. Reject proposal modes other than the declared generative bootstrap.

- [ ] **Step 5: Implement frozen constants and the estimator gate.** Store the five literal critical constants in `vfe4/numerics/critical_values.py`; load no quantile package at runtime. Implement the exact joint Bonferroni t bias and chi-square variance calculations over 76 cells with `a=0.01/304`, but exercise them during buildout only on shrunken deterministic fixtures. The separately authorized Task 13 evidence run requires every upper absolute-bias bound `<=0.001005033585350145`, every upper SD bound `<=0.0025125839633753625`, exact identities, and the complete 512-replicate inventory before publishing `validation/h6_smc.json`. Focused Task 5 output is not that evidence.

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
- Create: `tests/unit/test_h6_arms.py`
- Create: `tests/unit/test_h6_matching.py`
- Modify: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`

**Interfaces:**
- Produces: `build_a0` through `build_a5`, `build_arm(ArmId, ArmConfig)`, the eight exact matrix row records/config identities above, `audit_arm_matching(...) -> MatchingReport`, and exact `arm_matrix_sha256` / per-row hashes.
- Consumes: Tasks 3--5 model/recognition/predictor interfaces.

- [ ] **Step 1: Write failing semantic arm/matrix tests.** Assert A0 has normalized AR logits, only the conventional CE/model phase, and no latent/recognition object/optimizer; A1 has latents but no typed-map sector; A2 has unconstrained generic maps; A3 has fixed predecessor and no source categorical variables; A4 has no model channel; A5 has both channels, sources, and typed same-point maps. The no-latent component endpoint must likewise have only its model phase and the canonical `nolatent-norecognition` identity; its recognition section is structurally absent when `latent_enabled=false`, never retained as an inactive smoothing/filler setting. Require every literal matrix config/factory identity, exactly one named intervention per component row after conditional resolution, PRIMARY/MAP equal tuning, remaining rows' shared-A5 estimand, all eight seeds/checkpoints/certificate keys, all-or-none opening, and exact descriptive/nonclaim labels. Assert each predictor retains the exact public signature and vocabulary identity.

- [ ] **Step 2: Write failing matcher tests.** Count active trainable parameters by named role and typed phase, require every active ID exactly once in its declared AdamW optimizer, reject dormant/frozen filler and dummy/no-op recognition phases, and compute whole-outer-schedule FLOPs from each endpoint's actual active phases. Require identical passes, batch/model-update opportunities, data/validation/checkpoint access, and optimizer policy; do not require a nonexistent recognition phase for A0/no-latent endpoints. Enforce `abs(P_endpoint/P_A5-1)<=0.01` and `abs(F_endpoint/F_A5-1)<=0.05` for all six arms and every matrix endpoint. Test violations make the exact arm/row ineligible; they never relax bounds. Assert irreducibly changed-joint/approximation/latent-capacity rows remain descriptive despite matching.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_h6_arms.py tests/unit/test_h6_matching.py -q`. Expected: FAIL on missing factories/matcher.

- [ ] **Step 4: Implement explicit factories, matrix records, and deterministic capacity search.** Search only a preregistered finite tuple of width/rank/hidden-dimension candidates, order lexicographically, and choose the first candidate satisfying both tolerances and active-parameter rules. Do not inspect loss/NLL. Use explicit function calls, not registry signature inspection. Hash each endpoint's full config, factory identity, sole-change diff, match report, tuning estimand, seeds, certificate-key template, and opening group.

- [ ] **Step 5: Freeze candidate arm profiles before tuning.** Run only the no-corpus matcher/config preparation operation, record exact dimensions, parameter-role tables, optimizer IDs, FLOP term tables, margins, and hashes in the preregistration/config, and commit them. These are source-frozen candidates, not Prefix or closed Prediction evidence; deferred Prediction readiness mechanically reconstructs every endpoint after exact H1/H2/H3/H5 validation. If any arm cannot match, H6-Prediction remains INCONCLUSIVE until the architecture plan is revised; do not loosen tolerances.

- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS for the frozen arm profiles and negative controls.

- [ ] **Step 7: Commit.**

```text
git add vfe4/training/__init__.py vfe4/training/arms.py vfe4/training/matching.py tests/unit/test_h6_arms.py tests/unit/test_h6_matching.py vfe4/config/schema.py vfe4/config/resolve.py docs/preregistrations/2026-07-21-h6-prefix-prediction.md
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
- Produces: `train_h6_attempt`, `save_h6_checkpoint`, `load_h6_checkpoint`, `score_prior_nll_replicate`, `aggregate_endpoint_smc`, `inflate_paired_interval`, `paired_t_interval`, and `decide_primary_prediction`.
- Consumes: Task 1 H6-owned AdamW `H6TrainingSchedule` and Prediction-only exact H5 producer-field binding, Task 2 batch schedules, Task 4 `H6LanguageElboTerms`/typed non-ELBO ablation, Task 5 predictor, and Task 7 arms/matching report.

- [ ] **Step 1: Write failing tiny training/resume tests.** No training object may be constructed before a valid readiness capability materializes train data. All endpoints receive identical window IDs, counted tokens, passes, model-update/validation/checkpoint indices, common outer-schedule hash, H6 AdamW class/policy/phase identities, Prediction-only exact H5 producer-field reference, and terminal-checkpoint policy. Split/resume reproduces uninterrupted active weights/optimizers, estimator stream, sampler cursor, and metric hashes. Assert A0/no-latent endpoints run only `MODEL_CE_ADAMW`, allocate no recognition state/optimizer, and contain no filler phase; latent endpoints run H6 recognition AdamW, detached/nonalias snapshot, then H6 model AdamW. Metrics and checkpoints round-trip the exact horizon, ordered factor identities, all seven ELBO partitions, decomposition total, equality-check record, and canonical `H6LanguageElboTerms` hash. Emission-only checkpoints use the distinct non-ELBO type/tag. No H6 phase is accepted as an H5 update label, and no schedule makes an exact-coordinate/GEM/natural-gradient/monotonicity claim.

- [ ] **Step 2: Write failing scorer/uncertainty/statistics tests.** Use fake target-blind predictors and hand-authored 96-checkpoint/64-stream tables to prove corpus-summed token weighting, `-100` exclusion, no recognition argument, certificate/opening enforcement, exact particle/replicate inventory, registry pairing, `Y/Q0/Q1/Q2/R1/R2`, unbiased denominator-63 variance/covariance, exactly 352 simultaneous intervals, frozen `4.5144904535377144`, contraction and `B/H/e` thresholds, `Q2` as the reported NLL, and 256-corner interval inflation. Prove one-stream scoring, missing/nonfinite/duplicate streams, particle-level omission, failed convergence, or uncertainty above/equal-to the wrong side of a boundary is INCONCLUSIVE. Test exact `delta`, frozen df=7 constant `2.364624251592784`, PRIMARY/MAP rules, and descriptive rows.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/unit/test_h6_checkpoint.py tests/unit/test_h6_statistics.py tests/unit/test_h6_smc_uncertainty.py tests/integration/test_h6_language.py -q`. Expected: FAIL on missing training/evaluation modules.

- [ ] **Step 4: Implement the typed H6 schedule engine.** Accept only the immutable H6 common outer/endpoint-phase hashes resolved in Task 1; instantiate AdamW solely from the H6 optimizer class/policy record. Prediction readiness separately validates H5's actual `exact_coordinate`, `generalized_em`, and `natural_gradient_proposal` labels and the eight exact producer fields, but the trainer never maps those labels to AdamW or asks H5 to certify the schedule. A0/no-latent executes conventional CE/model AdamW directly; latent endpoints alone execute recognition AdamW, immutable snapshot, then model AdamW. Primary training uses smoothing and `H6LanguageElboTerms`; named endpoints change only their frozen field, and emission-only is accepted only through `EmissionOnlyAblationTerms`. Validate every active parameter/gradient/update, reject any absent/extra/no-op phase, and record the H6 nonmonotonic schedule nonclaim. Checkpoints include only active model/recognition/optimizer states plus RNG/estimator/data cursor/config/objective/exact H5 provenance/H6 optimizer/outer/phase/prefix identities and the complete typed horizon-indexed objective record, then publish atomically.

- [ ] **Step 5: Implement scoring, endpoint uncertainty, and decision math.** Use `math.fsum` for corpus and replicate aggregation, explicit counted-target totals, the exact 64-entry common stream registry, and literal critical constants only. Implement all `Y/Q/R/U/B/H/e` formulas and status precedence from the frozen protocol. Return raw per-checkpoint/per-particle/per-stream values, means, sample variances/covariances, convergence trace, all 352 bounds, all individual training-seed contrasts/error radii, all 256 corner intervals, final inflated interval, and rule trace. Production imports no SciPy and never substitutes an uninflated interval.

- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS.

- [ ] **Step 7: Commit.**

```text
git add vfe4/training/checkpoint.py vfe4/training/language.py vfe4/evaluation tests/unit/test_h6_checkpoint.py tests/unit/test_h6_statistics.py tests/unit/test_h6_smc_uncertainty.py tests/integration/test_h6_language.py
git commit -m "feat: add H6 training and prior scoring"
```

### Task 9: Implement the 9,720-case and 4,096-case dynamic prefix oracle

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

- [ ] **Step 1: Write failing enumeration and perturbation tests on shrunken fixtures.** Exercise the real production predictor/factories with distinct V=3 and V=258 identities on a small deterministic subset. Unit-test the closed-form full inventory counts `9,720`, `(6561,2187,729,243)`, and `4,096`, generator identities, and no config mutation without executing the complete inventories during buildout.

- [ ] **Step 2: Write failing leak/cache/mask tests.** Inject one target-reading predictor, one suffix-reading wrapper, one cache missing config identity, one post-softmax mask, and one all-invalid fallback. Each witnessed defect must FAIL. Remove a fixture/audit field and require INCONCLUSIVE. The correct predictor must produce exact zero residual in cold/warm/reverse modes.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/property/test_h6_prefix.py -q`. Expected: FAIL on missing oracle/fixture/runner.

- [ ] **Step 4: Implement independent enumeration and frozen perturbation generation.** The NumPy oracle constructs sequence pairs independently of production helpers and invokes the production predictor under the separately resolved small fixture. Focused tests generate only a deterministic subset. The complete 4,096 V=258 records and full 9,720 inventory are materialized and identity-bound only in the separately authorized Prefix evidence run.

- [ ] **Step 5: Implement dynamic comparisons.** For each exact case key, instantiate a fresh counter stream for the pair and capture log probabilities before selecting any target. Require equal vocabulary identity, dtype, shape `(V,)`, device, contiguity, raw `uint8` bytes, and SHA-256; do not use `torch.equal`, which hides signed-zero differences. Calculate exact mask mass, audit all-invalid behavior, and compare cache modes/traversal order. Record first counterexample and complete inventory; never short-circuit artifact accounting after a failure.

- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: PASS for correct fixtures and expected FAIL/INCONCLUSIVE for injected controls.

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

- [ ] **Step 3: Implement the import/signature/normalization/access audit.** Parse ASTs without importing launchers, build the in-repo import/call graph, enforce forbidden dependency edges, resolve the six explicit factories/matrix configs, and inspect the bound predictor signature. Prove every state/model fixed/prefix source row reaches the one shared pre-softmax parent-mask helper, every declared parent is `<receiver_t`, and no post-softmax/alternate normalization is reachable. Prove only `vfe4.data.access` reaches the private unsealer, train materialization requires readiness, model-facing test mapping requires `DurableTestOpeningCapability`, and blinded preprocessing returns no raw/token tensor. Hash every audited source file, exact `T_mask` manifest/count, access-policy graph, and audit rule set; unresolved dispatch is INCONCLUSIVE.

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

**Interfaces:**
- Produces: exact public `project_h1_prefix_prior_config`, `project_h6_prefix_config`, and `run_projected_current_candidate` lifecycle adapters with the return records/signatures frozen under Public Interfaces; `CurrentPredictionPrerequisiteRefs`, `run_h6_prefix`, `validate_h6_prediction_readiness`, `run_h6_experiment`, separate atomic Prediction-prerequisite/certificate/readiness/failure/checkpoint/test-opening schemas, and two editable root dictionaries.
- Consumes: all earlier tasks; launchers orchestrate only.

- [ ] **Step 1: Write failing independent Prefix gate/artifact tests.** Require one prefix certificate per exact source/config/model-family/vocabulary/estimator/data-safety key, PASS/FAIL/INCONCLUSIVE precedence, stale-own-hash rejection, exact mask/case/static-audit inventories, atomic manifests, and no overwrite. Prove Prefix runs and publishes with no predecessor artifact or PASS state present. Reject any Prefix config, preflight, result, reference file, artifact, or ledger schema containing an H1--H5 status/reference. The artifact contains only `config.json`, `provenance.json`, `environment.json`, `validation/h6_prefix.json`, `certificates/prefix_set.json`, and `manifest.sha256`. Assert Prefix closure contains no H1 variant, SMC accuracy, H6 schedule, matching, tuning, capacity, checkpoint, opening, or prediction claim.

- [ ] **Step 2: Write failing Prediction-readiness/access/launcher and lifecycle-adapter tests.** For each Prediction correctness input H1, H2, H3, and H5 independently, test absent payload, duplicate, `fail`, `inconclusive`, stale manifest, wrong `git_head`, wrong `dirty_digest`, wrong config/schema, and changed payload hash; each stops before endpoint factory/matcher, train materialization, optimizer, or trainer. Assert no H4 input is requested or can trigger an H4 benchmark. Validate H5 only through its payload/manifest identity and exact fields `update_spec_raw_sha256`, `update_spec_canonical_sha256`, `objective_schema_sha256`, `factor_input_schema_sha256`, `reference_sha256`, `recognition_state_sha256`, `model_state_sha256`, and `validation_payload_sha256`, with enabled labels restricted to the three actual H5 labels; reject invented schedule/snapshot/dependency hashes and Adam/AdamW labels. Reject Task 5/6 development artifacts whose revision/digest differs from the frozen Prediction candidate. After valid current inputs, require readiness to reconstruct every endpoint without corpus data, reproduce every parameter/FLOP/optimizer match report, and freeze H6-owned typed AdamW schedules. Test missing/stale/non-PASS prefix cert, current SMC/H1-prefix artifacts, critical-value/protocol hash, matrix key, and sealed-data access policy likewise. Focused compatibility tests freeze the exact three adapter signatures/return records, prove both projections are pure/nonmutating, require `project_h6_prefix_config(CONFIG)` to contain no predecessor fields, require `run_projected_current_candidate(..., predecessor_refs={})` for Prefix, and prove no H6 module owns `project_h7_compatibility_config`. `verify_vfe4.py` retains one editable root `CONFIG`, one `main`, and no required CLI; `train_vfe4.py` does likewise, imports without side effects, and its focused test shrinks dimensions/pass counts while preserving readiness/capability/factory/artifact paths.

- [ ] **Step 3: Run focused RED.** Run `python -m pytest tests/promotion/test_h6_prefix_gate.py tests/unit/test_h6_prediction_readiness.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py tests/integration/test_train_vfe4.py -q`. Expected: FAIL on missing gate/readiness/launcher/orchestrator and conditional existing-surface wiring.

- [ ] **Step 4: Implement the H6-owned lifecycle adapters and independent H6-Prefix publication.** Add pure, nonmutating `project_h1_prefix_prior_config(CONFIG) -> ProjectedCurrentCandidateConfig` and `project_h6_prefix_config(CONFIG) -> ProjectedCurrentCandidateConfig`, plus the generic keyword-only `run_projected_current_candidate(config,junit_sha256,predecessor_refs) -> CandidateArtifactReference` exactly as frozen above. H6-Prefix resolution and preflight validate only its exact own identities and require `predecessor_refs={}`. Publish only `config.json`, `provenance.json`, `environment.json`, `validation/h6_prefix.json`, `certificates/prefix_set.json`, and `manifest.sha256`; there is no predecessor reference file. The certificate set hashes sorted exact keys/payloads; `vfe4/types/results.py` carries the explicit H6 result. Leave H8's `project_h7_compatibility_config` to H8.

- [ ] **Step 5: Implement Prediction readiness before experiment access.** `validate_h6_prediction_readiness` first revalidates the exact deferred-evidence H1/H2/H3/H5 artifacts at one `git_head`/`dirty_digest`, then the separate same-candidate H1-prefix-prior and finite-SMC artifacts/ledgers and independent H6-Prefix certificate set. It validates H5's actual fields/labels as correctness provenance while taking AdamW class/policy/phases solely from the common/typed H6 schedule. It also validates critical-value and actual-endpoint protocol hashes, the literal matrix, every prefix key, and the blinded-data access policy; then it reconstructs all endpoints without corpus access and mechanically reproduces/freeze-hashes every match report. It never requests H4 or launches an H4 benchmark. It publishes separate `h6_prediction_readiness.json` and returns an opaque PASS token. Only that token can materialize train data or start empirical operations; matching is a Prediction-readiness phase, not a Prefix claim.

- [ ] **Step 6: Implement atomic experiment/opening surfaces and the two launchers.** Keep probability, capability/data, training, uncertainty, and artifact logic in package modules. The readiness token and every attempt reference, rather than copy, Prediction prerequisite/Prefix artifacts. Each launcher has exactly one editable root `CONFIG`, one `main`, one guard, no required CLI or environment variable, and no auto-run on import; `train_vfe4.py` prints resolved operation/status/artifact paths and returns structured results. Implement the exclusive durable reservation that issues the only `DurableTestOpeningCapability`, plus immutable raw endpoint-MC/result schemas from Task 14. No argparse, Typer, or Hydra.

- [ ] **Step 7: Run focused GREEN.** Run the Step 3 command. Expected: PASS.

- [ ] **Step 8: Commit.**

```text
git add vfe4/artifacts/h6.py vfe4/artifacts/provenance.py vfe4/artifacts/__init__.py vfe4/types/results.py vfe4/config/schema.py vfe4/config/resolve.py verification/h6_prefix_gate.py verification/run_gates.py verify_vfe4.py train_vfe4.py vfe4/training/h6_experiment.py vfe4/training/h6_readiness.py tests/promotion/test_h6_prefix_gate.py tests/unit/test_h6_prediction_readiness.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py tests/integration/test_train_vfe4.py
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

- [ ] **Step 3: Review the three H6-owned lifecycle adapters.** Confirm the exact public signatures and return records for `project_h1_prefix_prior_config`, `project_h6_prefix_config`, and `run_projected_current_candidate`; Prefix projection/runner tests require no predecessor reference or PASS state. Confirm `project_h7_compatibility_config` is absent because H8 owns it.

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

- [ ] **Step 2: Durably reserve exactly one opening and issue the sole test capability.** Before model-facing mapping/unsealing, create the canonical immutable reservation with `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)`. Write canonical JSON containing experiment/readiness/checkpoint-set/current-candidate/sealed-data/access-policy/analysis hashes, all 96 checkpoints, replicate IDs `0..63`, particle counts `(128,256,512,1024)`, and `state="RESERVED"`; flush and `os.fsync` the file, close it, then fsync the containing directory (or documented durable equivalent). `FileExistsError` blocks without model-facing test access. Never reopen, truncate, replace, rename over, or delete the reservation. A crash/exception after reservation is terminal Prediction INCONCLUSIVE and never retryable. Derive `DurableTestOpeningCapability` from the exact durable record, then and only then call `open_test_for_scoring`; blinded sealed storage before this point is not an opening.

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

- **Spec coverage:** Tasks 1--11 build separate Prefix/Prediction surfaces with exact singleton/DAG types, horizon-indexed `H6LanguageElboTerms`, the actual H5 producer boundary, H6-owned AdamW schedules, and all three H7-consumed lifecycle adapters. Task 12 closes source with focused shrunken tests only. Tasks 13--14 explicitly defer full Prefix inventories, exact H1/H2/H3/H5 plus H1-prefix-prior/finite-SMC readiness, corpus training, 96 checkpoints, and one test opening; none requires or reruns H4 timing.
- **Task ordering:** Prefix can run and publish independently from its own identities. Prediction readiness alone consumes exact H1/H2/H3/H5, H1-prefix-prior, finite-SMC, and Prefix evidence before corpus materialization. Source buildout ends at Task 12 without a broad suite, gate, training run, checkpoint grid, or opening. Under separate authorization, tuning precedes confirmation, all checkpoints precede the durable opening, and Prediction closure never mutates Prefix evidence.
- **Type consistency:** `ZeroDimensionalBase`, `CausalDag`, `H6LanguageStructure`, `H6FactorTerm`, `H6LanguageElboTerms`, `EmissionOnlyAblationTerms`, `ArmId`, `VocabularyIdentity`, `PrefixCaseKey`, `PrefixCertificate`, Prediction-only artifact refs, exact-field `H5UpdateBinding`, H6 AdamW schedules, the three lifecycle adapter records, capabilities, estimator records, and evidence revisions have one owner and consistent consumers. The public predictor signature, ELBO partitions, adapter signatures, vocabulary-sized result, access capabilities, and typed schedule are identical across config, factories, metrics, checkpoints, tests, and static audit.
- **Placeholder scan:** Every protocol choice that affects evidence is fixed here or has an exact pre-outcome measurement-and-freeze procedure. No threshold, seed, status rule, estimator algorithm, dataset substitution, or statistical decision is selected after predictive outcomes.
- **Path check:** The plan is saved at `docs/superpowers/plans/2026-07-21-vfe4-h6-prefix-prediction.md`. Implementation must use a fresh dedicated branch/worktree, preserve the user's live/WIP, and follow the bounded commit sequence above. This authoring task itself performs no code change, test, training run, network action, or commit.
