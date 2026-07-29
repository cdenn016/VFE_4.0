# VFE 4.0

This repository implements the bounded H1 foundation, H2 information--moment
representation gate, H3 structured-recognition adequacy gate, and one coupled
H1--H5 verification prefix that integrates the already frozen H4 cost artifact
with the H5 update-coherence gate, plus a separate reference-only H7
publication path. The implemented compatibility prefixes remain H1, H1/H2,
H1/H2/H3, and H1/H2/H3/H4/H5. H4 is not available as a standalone prefix.
H7 does not rerun those gates; H6-Prediction, H8, and training remain outside
its frozen frame-covariance claim.

The user-facing surface is `verify_vfe4.py`. Its top-level `CONFIG` dictionary
is editable, and the file can be run directly from an IDE or file association;
there are no command-line flags or parser. Importing it performs no work.

## WikiText-103 operator workflow

`train_vfe4.py` is the separate click-to-run WikiText-103 surface. Edit its
single top-level `CONFIG` dictionary and click Run; do not add command-line
arguments or depend on environment-only settings. The default operation is
`"idle"`, so an unchanged click performs no data access, device
initialization, reservation, checkpoint I/O, or training. Its exact operations
are `idle`, `synthetic_smoke`, `source_lock`, `readiness`, `train`, and
`resume`. The default VFE4-owned cache root is exactly
`str(Path.home() / ".cache" / "vfe4" / "wikitext103")`; the default run root
is `artifacts/wt103-runs`. Paths must be absolute, explicit, free of globs,
and outside every V3 cache or checkout.

`Path.home() / ".cache" / "tokenized_cache" / "*.pt"` denotes legacy V3
PyTorch ZIP/pickle caches. VFE4 leaves those files untouched and unread,
never deserializes them, and never admits them as source, tokenizer, or cache
authority.

For the generated-data integration check, set
`CONFIG["training"]["operation"] = "synthetic_smoke"` and leave
`CONFIG["authorization"] = None`. This runs one CPU thread over generated
scalars through all five frozen arms, validation, metric export, terminal
checkpoints, manifests, and one real resume continuation. Every output is
labeled `nonproduction_synthetic_smoke`; the result is a partial `pretest`
index, is ineligible for production readiness, and cannot open held-out data.
Reusing the same `smoke_run_id` or output roots is rejected rather than
overwritten.

Official source locking is a separate network-capable transaction. Set the
operation to `"source_lock"` only after authorizing the exact phrase
`AUTHORIZE_VFE4_WT103_SOURCE_LOCK_V1` in `CONFIG["authorization"]`. The
transaction downloads only the pinned official sources, builds the exact
token and schedule products, publishes one durable typed source bundle, and
reopens every byte before returning. Do not substitute an ad hoc download or
an unverified cache. The bundle also records the actual PRIMARY A5 parameter
inventory and a phase-by-phase FLOP ledger. That ledger remains explicitly
inconclusive wherever the semantic primitive policy is not frozen; it never
invents a scalar FLOP total.

After a finalized source record exists, `"readiness"` uses no authorization
and performs no corpus optimizer update. It first requires an exact A0 match.
The current preregistered width grid contains no candidate within the 1%
PRIMARY parameter margin, and the PRIMARY semantic FLOP total remains open,
so readiness returns typed capacity-matching obligations before opening any
Task 14 evidence. Later readiness work must also reopen the exact source,
dependency lock, predecessor evidence, training-sparsity certificate,
durability records, environment, and resource forecast before it can issue a
PASS token.

Real `"train"` and `"resume"` operations require the exact authorization
phrase `AUTHORIZE_VFE4_WT103_PRODUCTION_TRAINING_V1` after readiness is
complete. Resume must name one absolute, explicit
`experiment-plan.json` in `CONFIG["paths"]["resume_experiment_plan_path"]`;
that exact immutable plan is reopened before its plan-bound ledger and
authenticated rolling sidecar/checkpoint. There is no glob, newest-run
discovery, or fallback checkpoint selection, and the terminal
`experiment-index.json` remains absent until the complete pretest
attempt inventory is finalized. Launcher schema v2 migration: rename
`resume_experiment_index_path` to `resume_experiment_plan_path` and point it
to the experiment's exact `experiment-plan.json`; copied launcher-v1 configs
must also set `launcher_schema` to `wt103-click-launcher-v2`, and the legacy
key is rejected.
Both operations remain sealed while any capacity-matching or readiness
obligation is open.

Before acquisition, preprocessing, training, or evaluation, forecast archive
staging, extracted data, int32 token caches, schedules, every retained
checkpoint, records, figures, and 25% temporary-write overhead. Available disk
space must be at least `2 * forecast + 10 GiB`, with enough host headroom for
the measured preprocessing multiplier and one atomic checkpoint duplicate.
Before real training, run the shape-identical allocation preflight through
every distinct frozen arm path. Both peak device allocated memory and peak
device reserved memory must be at most 85% of physical capacity. A failed
disk, RAM, dependency, durability, sparsity, device, throughput, power, or
resource-authorization check stops before reservation or allocation; it never
silently changes batch size, sequence length, particles, source width,
precision, or checkpoint contents.

Figure regeneration remains isolated in `generate_vfe4_figures.py`. Edit its
own `CONFIG`, set `operation` to `"render"`, provide one absolute finalized
`experiment-index.json`, set `figure_root` to that experiment's exact
`figures` directory, and click Run. It renders only manifest-validated
finalized metrics and the frozen result table. It never opens data or
checkpoints, trains, chooses a newest run, or consumes partial attempts.

The held-out test is an irreversible one-opening transaction. It may occur
exactly once, only after the complete ordered inventory of `terminal_scoring`
checkpoint keys and all preflight identities validate. Synthetic smoke,
readiness, resume, and figure rendering cannot reserve or open it, and an
interrupted opening cannot be retried or repaired by substituting a
`resume_only` checkpoint.

V3 files are read-only design references. VFE4 imports no V3 module and
transfers no V3 config, objective, cache, tokenizer evidence, metric schema,
checkpoint, or scientific authority. Likewise, the H8 sparse-scale result is
synthetic structural evidence, not WikiText-103 training-memory evidence. Real
training independently requires a same-revision
`TrainingSparsityCertificate` and the separate 85%-capacity preflight.

The separate H8 selection reads only
`.verification/h8-current-candidate-<FULL_HEAD>-refs.json`, binds its exact
current-candidate `h8-current-candidate-refs-v3` variants, and executes only
H8. The direct H1--H5, active H1-prefix-prior, and independently produced
H6-Prefix variants must match H7's transitive references exactly.
H6-Prediction remains a separate scientific prerequisite bound to its own
frozen producer and dependency closure. H8 reopens and validates those
references but never reruns or copies their payloads, certificates, or
ledgers.

For direct click execution, `verify_vfe4.py` establishes the recorded H8
startup contract before importing project numerical code: the five thread
limits equal one and `MKL_THREADING_LAYER=SEQUENTIAL`.
`KMP_DUPLICATE_LIB_OK` is forbidden rather than used to mask duplicate OpenMP
runtimes. Package callers must establish the same environment before selecting
H8. Parent and children record/revalidate the exact values and forbidden
absence; this is a runtime-safety contract, not scientific H8 PASS evidence.
The direct verifier applies this bounded startup preamble before resolving any
editable operation because operation resolution imports numerical project
types; programmatic H8 use additionally requires a fresh numerical runtime and
an environment that was already valid when `verify_vfe4` was imported.

The active installed-runtime `h8-sparse-scale-v5` selected route uses
`h8-validation-config-v3` and `vfe4.h8.parent-child-protocol.v3`; the 2.9.1/v4
pin was superseded before any accepted H8 scientific or milestone run. It
runs the frozen 12-cell
correctness grid, mints parent authority only after a valid prerequisite
start, and retains the exact issued prefix of the 30-slot request plan:
15 cold production children in seed-major/repetition order, three separate
profiler children in seed-major order, and 12 isolated controls in frozen
order using seed `20260721`. PASS requires all 30 attempts and the complete
15/3/12 decoded inventories. A witnessed timeout, abnormal exit, forbidden
operation or allocation, off-band fill, nonfinite value, solver failure,
resource or pivot breach, identity mismatch, invalid profiler transition, or
missed control dominates as FAIL. Missing or nonunique evidence without a
witnessed violation is INCONCLUSIVE. Direct evaluation, incomplete
inventories, or caller-supplied runtime sections cannot clear the runtime PASS
locks.

The inclusive per-child limits are 60.0 seconds, 134,217,728 bytes of
incremental process high-water memory, 67,108,864 bytes of live PyTorch
population storage, 411,200 float64-equivalent scalars in each precision,
factor, and selected-inverse category, solve RHS width at most 40, sample width
exactly 1, and zero forbidden attempts or off-band fill. The four primary
observability channels are PyTorch dispatch/live-storage tracing, a separate
profiler child with lossless raw-event joins, backend
fill/workspace/RHS/sample/selected-block counters, and clean-subprocess OS
high-water memory. The NumPy guard supplies its assigned controls.
`tracemalloc_supplementary` is literal JSON `null` and cannot affect status.
Actual parent timing is retained on the attempt and never overwrites the
child-authored `resources.parent_elapsed_ns=0`.

One selected publication contains exactly `config.json`, `provenance.json`,
`environment.json`, `references/h7.json`,
`references/h6_prediction.json`, `validation/h8.json`, and
`manifest.sha256`. The post-publication current-candidate result pointer is
external to that manifest. The exact ordered nonclaims are
`no_language_result`, `no_training_result`, `no_prediction_result`,
`no_large_language_model_scale`, `no_asymptotic_scaling_law`,
`no_gpu_claim`, `no_exact_global_spectrum`, and
`no_post_h8_training_memory_transfer`.

This is an implemented, PASS-capable protocol surface, not a measured result.
It freezes installed `torch.__version__` `2.10.0.dev20251210+cu128`, profiler
source hashes `22de3b0790907b90053af829ebf1bff0b6add2745ac0381ec7de78812edacb47`
and `543430b2e9b24df777f86415865fee250b35e3444a80920bcca0e8889b917956`, and
descriptor SHA-256 `2ee166166bab997499cc66da85146a031f458fbe0190a75b1a1a3ddea80efc38`.
Those Python-side identities do not identify compiled Kineto; the schema
preflight and scientific profiler child provide behavioral evidence. No v5
scientific or milestone execution, JUnit total, residual, resource
endpoint, or H8 PASS is reported here.

`verification/h8_preflight.py` remains a separate metadata-only, zero-compute
advisor. It launches no tests, correctness cells, runtime children, profilers,
controls, training, or data work; it only reports whether the parent
orchestrator and prerequisite identities are ready.

The selectable H7 operation accepts only
`("H1","H2","H3","H4","H5","H6-Prefix","H7")`. It derives the exact
`.verification/h7-current-candidate-<FULL_HEAD>-refs.json` registry, validates
the ordered H1--H5, H1-Prefix-Prior, and predecessor-free H6-Prefix sibling
references, captures the raw H1 and H7 fixture bytes, and publishes only
`config.json`, `provenance.json`, `environment.json`, the three reference
records, and `validation/h7.json` under one manifest. It never reruns or copies
a predecessor payload, certificate, or ledger.

H7's scalar `GL+(1,R)` replay is a complete-law regression only. The primary
claim is limited to the preregistered direct `GL+(2,R)` matrix elements and
does not cover `det(g)<0`. The frozen source scorer is
`alpha_b,t,j(prefix)+r_z^T z_j+r_m^T m_j`; both covectors use the source-frame
inverse transpose, and `history_scorer_wrong_source_inverse` is its exact
control. A fixed decoder is positive only on
`C_V W g^-1=C_V W`. Continuous recognition entropy shifts by `+logJ_G`; it is
not invariant. This unevaluated Task-8 surface is `INCONCLUSIVE`: no H7 test
totals, residuals, or PASS result have been measured. Optimizer/training
equivariance, H6-Prediction, predictive benefit, and H8 scale remain open.

H1 evaluates one frozen `T=2`, scalar-state, scalar-model fixture in three
separately assembled ways:

1. a monolithic full-joint Gaussian-mixture ELBO;
2. a local conditional-factor ELBO decomposition; and
3. an independent NumPy evidence-minus-posterior-KL identity.

At reviewed revision `b736d21`, all three values are approximately
`-3.911522906174741`. The maximum pairwise residual is
`4.440892098500626e-16`; the largest pair-specific allowance is
`7.582209551464146e-13`. Every pair and every homologous local term is decided
against its own calibrated allowance. The exact-head click-run gate reports
`H1: pass` and publishes an atomic, manifest-checked artifact under `runs/`.

H2 keeps that fixture and its four positive source components unchanged. It
verifies each recognition and generative Gaussian component separately, then
uses the exact source weights to compare the direct PyTorch information
representation, the unchanged PyTorch H1 moment representation, and an
independently parsed NumPy dense-moment oracle. It never moment-projects the
source-marginal mixture or describes that mixture as one Gaussian. Natural
coordinates are `(h,-J/2)`, expectation coordinates are `(mu,M)` with
`M=Sigma+mu mu^T`, and `(mu,Sigma)` is the moment representation rather than
the Fisher-dual expectation-coordinate pair.

The frozen H2 envelope is `D <= 6`, `lambda_min(J) >= 1e-4`,
`lambda_max(J) <= 1e4`, `kappa_2(J) <= 1e6`, and `||mu||_inf <= 4`. Its
absolute error budget fixes `eps=np.finfo(np.float64).eps`,
`gamma(n)=n*eps/(1-n*eps)`, `C=256`, and `N(D)=8*D+32`; at the observed
`kappa_2 <= 42.35`, the preregistered descriptive pair budget is approximately
`3.86e-10 * scale`, while every decision uses its own invariant-specific
allowance.

H3 uses two separately authored and separately hashed, source-free,
four-dimensional Gaussian fixtures. The coupled law has a frozen analytic
factorized reverse-KL gap; the control removes the preregistered couplings and
has a diagonal exact posterior. Both the structured full-SPD and fine
factorized diagonal recognition arms start from the same zero mean and identity
precision. Only H3 recognition parameters enter its reverse-mode autograd
graph; the frozen generative factors and independent NumPy oracle do not.
Operand-local absolute allowances and two signed three-way thresholds determine
PASS, FAIL, or INCONCLUSIVE without relative tolerances or blanket `allclose`.
This bounded H3 protocol does not establish H4 cost, H5 update coherence, H6
prediction, H7 frame covariance, or H8 sparse scaling.

The one editable `CONFIG` now requests H1 through H5. One run captures each
required fixture exactly once, evaluates H1, H2, H3, H4, then H5, and publishes
separate `validation/h4.json` and `validation/h5.json` payloads in one atomic,
manifest-checked artifact directory. The runner serializes the preexisting
typed H4 validation artifact; it does not reconstruct H4 status or measurement
content. H4 and H5 statuses remain separate and may differ. Selecting H1,
H1/H2, or H1/H2/H3 and removing the later conditional sections preserves the
shorter compatibility surface and does not capture, evaluate, or publish H4/H5
data. The theory sources are
`Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex` and
`Manuscripts/MAgent_exact_elbo_whitepaper.tex`.

H2 is verified at implementation revision
`00de72b93ebcc504ef5652d11ad3012f80852aa0`. The revision-bound JUnit XML
contains 414 tests, 0 failures, 0 errors, and 0 skips; its SHA-256 is
`268902c66ab92955574526cd4bf1fcd7999611a88009d3c6edaa6ba8aa17a7b7`.
The one final click run published
`runs/verify-h1-h2-20260722T074944065126Z-cb17f1bb2893`, whose five manifest
hashes recompute and whose provenance binds the tested Git revision, dirty
digest, config hash, fixture hash, objective schema, and CPU float64
environment.

The H2 artifact reports 295/295 invariants and 282/282 per-quantity
comparisons passing, with zero obligations. Its direct-information, unchanged
H1-moment, and independent NumPy ELBO values are respectively
`-3.9115229061747425`, `-3.9115229061747407`, and
`-3.91152290617474`. The largest comparison residual is
`3.5527136788005009e-15` under its `5.4870414032774655e-09` allowance; the
largest residual/allowance ratio is approximately `5.018e-5`. All eight
precision-condition records remain inside the frozen envelope, and all four
exact negative controls are decisive, including direct `h`-as-`mu` semantic
records and an inverse audit with zero production violations and one detected
injected violation. Independent artifact, numerical, dependency, and
spec/reachability reviews found no remaining Critical or Important H2 issue.

There is no dataset, language-model training path, optimizer, parameter update,
or gradient calculation in H1. This bounded evaluation performs no
backpropagation, but the VFE 4.0 codebase is not claimed to be
backpropagation-free. H2 establishes only bounded componentwise representation
verification. H3 uses reverse-mode autograd internally to obtain gradients of
the direct ELBO with respect to its recognition mean and precision-Cholesky
parameters; this is ordinary backward differentiation through a forward ELBO
computation, not a claim of backpropagation-free learning. Neither implemented
gate makes a general optimizer, performance, prediction, scaling, or
later-gate claim.
