# H6 Executable Prediction Amendment Design

**Date:** 2026-07-27
**Status:** pre-outcome design; no corpus training, validation result, test
opening, or H6 scientific result exists at this revision

## Purpose

The amended H6 source currently validates configurations, readiness,
matching, checkpoints, statistics, and artifact consumers, but it deliberately
refuses every experiment operation. It also has no executable provider for the
categorical source posterior required by the VFE 4.0 joint. This amendment
freezes the missing recognition, estimator, device, resume, orchestration, and
held-out-record contracts before an H6 training result exists.

The design keeps the existing click-to-run workflow. `train_vfe4.py` remains
the only user-facing H6 launcher, with one editable top-level `CONFIG`
dictionary, one selected operation, and no required command-line interface.
It does not install, replace, or downgrade Torch.

## Normative authority and supersession

The following order is normative:

1. the VFE 4.0 normalized joint, recognition disintegration, and ELBO in
   `Manuscripts/vfe4_whitepaper/04_generative_model.tex`,
   `05_structured_information_form.tex`, and
   `06_elbo_coordinate_updates.tex`;
2. `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`;
3. `docs/preregistrations/2026-07-25-h6-audit-amendment.md`;
4. `docs/preregistrations/2026-07-25-h6-workload-amendment.md`;
5. this pre-outcome executable amendment.

This amendment changes no generative factor and no held-out target-blindness
rule. It supersedes only the missing or conflicting execution clauses:

- a marker-only `ExactSourceMixtureLaw` is not an evaluated source posterior;
- one window-level Gaussian cannot stand in for receiver-indexed filtering;
- an RNG-state hash without restorable state is not exact resume;
- CPU-only training is not required;
- the old all-endpoint held-out inventory is superseded by exactly 4,104
  logical scoring rows.

This amendment completes the existing sealed WikiText-2 H6 experiment. The
post-H8 WikiText-103 build is a separate dataset and experiment schema. It
must define and authenticate its own archive/member inventory, tokenizer,
split/window manifests, row counts, and data identity; a WikiText-2 store or
identity is never relabeled as WikiText-103.

## Selected recognition family

### Mixed law

For one observed training window, the selected restricted recognition family
is

```text
Q_psi(y,a,b | x,Gamma)
  = product_t [
      beta_psi,t(a_t | I_t)
      gamma_psi,t(b_t | I_t)
      q_psi,t(y_t | a_t,b_t,I_t)
    ].
```

An absent source bank is represented by one typed singleton choice with
probability one and contributes neither entropy nor parameters. The continuous
conditional is receiver-factorized with shared parameters:

```text
q_psi(y | a,b,x,Gamma)
  = product_t q_psi,t(y_t | a_t,b_t,I_t),
y_t = (z_t,m_t) for two-channel arms,
y_t = z_t       for one-channel arms.
```

`structured` means a full SPD covariance between the live channels within one
receiver. `factorized` means a block-diagonal covariance between those
channels. Neither H6 family claims a full dense population covariance across
receivers. This is a declared restricted posterior within the manuscript's
general normalized mixed recognition family, not a change to the generative
model and not evidence that the restriction is adequate.

The categorical rows are explicit recognition factors. They are not the
generative priors and are never sampled during ELBO evaluation. Every
supported parent is summed exactly.

### Receiver-indexed continuous laws

`LanguageRecognitionParameterStore` remains the sole owner of recognition
parameters. It produces one immutable `LanguageRecognitionTrajectory` with:

- receiver labels `0..T`;
- one live mean row per receiver;
- one shared live precision-Cholesky parameterization;
- one base component mean for receivers `0..T-1` and one
  source-conditioned component-mean row for every local positive-support
  source tuple at the terminal receiver `T`;
- the exact within-receiver block structure;
- one state-source row and, when applicable, one model-source row for each
  receiver `1..T`;
- the conditioning mode, support rows, and complete identities.

The store reuses its token embedding, mean head, and precision parameters. It
does not create a separate parameter bank per receiver. For receiver `t`, the
context is:

```text
filtering: deterministic_position(t) + mean(E[x_1],...,E[x_t])
smoothing: deterministic_position(t) + mean(E[x_1],...,E[x_T])
```

The empty filtering context at the initial receiver uses only
`deterministic_position(0)`. The position map is a frozen, parameter-free
float64 sinusoidal map with a versioned descriptor. It exists so smoothing
does not collapse every receiver to the same mean. Filtering receiver `t`
never observes a token after `t`.

The bounded H6 restriction makes continuous components source-conditioned
only at the terminal receiver. For bank `c` in `{state,model}`, define a
shared rank-one source shift

```text
s_c,t,j = v_c^T(context_t-context_j) + rho_c*log1p(t-j)
delta_c,t,j = u_c * s_c,t,j,
```

where `u_c` has the latent width of its own channel. At `t=T`, the component
mean is the base receiver mean plus the state shift in the state block and the
model shift in the model block. At `t<T`, the shift is exactly zero and the
single normalized base Gaussian is reused rather than duplicated per source
tuple. The precision-Cholesky is shared across receivers and terminal
components. This terminal-mixture restriction is nontrivial but ensures that
every parent-specific generative prior reads a source-independent earlier
continuous history, so all categorical expectations remain exactly
enumerable without a global assignment product. `v_c`, `rho_c`, and `u_c`
use deterministic nonzero initialization and are trainable recognition
parameters. No state-source shift is inserted into the model block or vice
versa.

### Compact categorical recognition

The source rows use ragged causal supports and never allocate a dense
`T x T` matrix. For bank `c` in `{state,model}`, receiver `t`, and declared
parent `j`,

```text
r_c,t,j = v_c^T (context_t - context_j)
          + rho_c * log1p(t-j)

log q_c,t(j)
  = log_softmax_support(
      stop_model_gradient(log pi_c,t(j) evaluated at the recognition means)
      + r_c,t,j
    ).
```

This mean-evaluated row is a recognition-proposal feature only. It uses the
exact target-blind token-prefix argument of the corresponding prior but the
earlier recognition means as continuous inputs. Model-parameter gradients
through the proposal baseline are stopped; gradients with respect to the
recognition means remain live. Under smoothing, those means may depend on the
full observed window, which is permitted for recognition but never
reclassified as a target-blind generative factor. The actual generative
`log pi_theta` term in the ELBO is evaluated on the same sampled component
history used by its transition term, with model gradients live only in the
model phase. Fixed-prior endpoints use their exact fixed prior rows for both
the proposal baseline and generative evaluation.

Each live bank adds one vector of length `recognition_width`, one lag scalar,
and one channel shift vector of length `latent_width`. A one-bank arm adds
`R+1+d` parameters; a two-bank arm adds `2(R+1+d)`.
These are real recognition parameters, participate in the recognition
optimizer and forward calculation, and enter parameter and whole-schedule
FLOP matching. The prospective matching selector must be rerun from formulas;
no prior candidate is grandfathered and no filler parameter is allowed.

Rows are normalized only over the declared positive-prior support. A missing
support, an all-invalid row, a self/future parent, a positive recognition mass
on a zero-prior parent, or a second normalization path is an error.

The trainable-bank count is frozen by arm: zero for A0, A1, and A3; two for
A2 and A5; and one for A4. Fixed-prior rows remain part of the generative
model; "fixed" does not mean that the corresponding recognition row is tied
to that prior.

## Executable ELBO estimator

### Exact categorical reduction

For receiver `t`, the production estimator draws one base
`epsilon_t ~ N(0,I)` and uses common random numbers across all of its local
source components:

```text
y_t^(j,k) = mu_t^(j,k) + L_t^(-T) epsilon_t.
```

For one state-source row,

```text
E_beta[log pi^z_t(a_t)]
  = sum_j beta_t(j) log pi^z_t(j)

E_beta[log K^z_t,a_t]
  = sum_j beta_t(j) log K^z_t,j

H(beta_t) = -sum_j beta_t(j) log beta_t(j).
```

The model-source row is identical with `gamma`, `pi^m`, and `K^m`. The exact
row contribution to the ELBO is

```text
sum_j beta_t(j) [
  log pi^z_t(j) + log K^z_t,j - log beta_t(j)
]
```

and analogously for `gamma`. The categorical entropy occurs exactly once
through `-sum q log q`; its coefficient is one. It is not duplicated in the
Gaussian entropy partition and has no temperature or beta multiplier.
For receivers `t<T`, the continuous law is source-independent and these
separate row sums are exact conditional on the continuous base draws. At the
terminal receiver the state transition and emission depend jointly on
`y_T^(j,k)`, so the canonical terminal contribution uses the explicit finite
double sum

```text
sum_(j,k) beta_T(j) gamma_T(k) [
  log pi^z_T(j) + log K^z_T,j(y_T^(j,k), y_j)
  + log pi^m_T(k) + log K^m_T,k(y_T^(j,k), y_k)
  + log p(x_T | y_T^(j,k))
  - log beta_T(j) - log gamma_T(k)
].
```

Structurally absent banks reduce this to the corresponding single sum.
Canonical partition records may split the bracket into source, transition,
emission, and entropy slots, but their live sum must equal this same tensor.
The finite source sum is exact conditional on `epsilon`; expectation over
`epsilon` makes the one-sample pathwise continuous estimator unbiased. The
sampled `log pi` and `log K` values are not evaluated only at the base mean
and are not plug-in recognition-prior values.

The categorical coordinate oracle remains

```text
q_star(j) proportional to exp(
  E_q[log pi_t(j) + log K_t,j]
).
```

It is a tiny exact-oracle check, not the production update rule and not an
extra training phase.

### Rao-Blackwellized continuous estimate

The production evaluator is:

```text
evaluation_method = "reparameterized_mc"
continuous_base_samples_per_receiver_per_example_per_phase = 1
categorical_evaluation = "exact_support_sum"
gaussian_entropy = "analytic"
component_sampling = "common_random_numbers_per_receiver"
```

One continuous noise tensor containing one base draw per receiver is consumed
for an entire example and phase. Every local component at that receiver uses
the same base draw; every occurrence of that component in an initial,
transition, source-prior, or emission term reuses the identical realized
value. Source labels are never sampled. Required expectations are computed by
nested exact sums over only the local source rows on which a factor depends;
the implementation never enumerates or materializes the global Cartesian
product of assignments. Gaussian component entropy is analytic. The estimator
therefore Rao-Blackwellizes every finite source choice while using one
pathwise continuous base sample per receiver for nonlinear expectations.

The independently accumulated total and canonical ordered term total must be
the same live scalar tensor. Reverse-mode autograd computes gradients of that
scalar with respect to the active parameter block. This is ordinary
backpropagation through the forward ELBO computation; it is not forward-mode
AD and the build is not described as backpropagation-free.

`ExactSourceMixtureLaw` becomes an immutable evaluated record containing the
exact supports, posterior rows, normalized component means/shared
precisions, generative prior rows at sampled component histories, per-parent
transition contributions, categorical entropies/KLs, and identities. A marker
with only an endpoint identity cannot authorize an objective.

For the `moment_projection` endpoint, the exact terminal Gaussian mixture is
replaced by the manuscript's moment-matched Gaussian
`N(bar_mu_T,bar_Sigma_T)` before objective evaluation. Earlier receivers are
already single Gaussians and are unchanged. Its joint recognition law retains
the same explicit categorical rows but makes the projected terminal
continuous receiver independent of the local source choice. The projection
uses

```text
omega_jk = beta_T(j) gamma_T(k)
bar_mu_T = sum_(j,k) omega_jk mu_T^(j,k)
bar_Sigma_T = Sigma_T
  + sum_(j,k) omega_jk
      (mu_T^(j,k)-bar_mu_T)(mu_T^(j,k)-bar_mu_T)^T
C_bar = cholesky(bar_Sigma_T)
y_bar_T = bar_mu_T + C_bar epsilon_T.
```

It uses the analytic entropy
`0.5 * [D*(1+log(2*pi)) + logdet(bar_Sigma_T)]`, not the shared component
entropy. Gradients remain live through the mixture moments, Cholesky
factorization, sample, and entropy. The projection record stores the exact
first two moments and the analytic
`sum_(j,k) omega_jk KL(N_jk || N_bar)` dispersion upper bound; it does not
call that upper bound the generally intractable mixture KL. Exact and
projected endpoints therefore differ by a real, normalized recognition
approximation.

## Phase ownership

Every latent batch has exactly these three boundaries:

1. **recognition AdamW**
   - model parameters have `requires_grad=False`;
   - continuous-recognition and categorical-residual parameters are active;
   - the complete endpoint objective is evaluated with the recognition-phase
     noise domain;
   - the minimized scalar is `loss = -elbo_estimate` (or negative live
     emission objective for the declared non-ELBO ablation);
   - backward, finite-gradient validation, global-norm clipping, and one
     recognition AdamW step occur.
2. **immutable detached snapshot**
   - after the recognition optimizer step, perform a fresh deterministic
     recognition forward with the updated parameters and the still-frozen
     model;
   - clone and detach every updated receiver/component mean, shared
     precision-Cholesky value, categorical row, support, context identity,
     recognition-state identity, source-model identity, and law identity;
   - no recognition graph edge or mutable alias remains.
3. **model AdamW**
   - recognition parameters are inactive;
   - the model consumes only the detached recognition snapshot;
   - the complete endpoint objective is reevaluated with the distinct
     model-phase noise domain;
   - the minimized scalar is again `loss = -elbo_estimate` (or the negative
     declared live emission objective);
   - backward, finite-gradient validation, global-norm clipping, and one model
     AdamW step occur.

No-latent endpoints retain one model AdamW phase minimizing exact
cross-entropy/negative sequence log probability and construct no recognition
state. The emission-only endpoint remains explicitly non-ELBO and uses only
its live emission terms in both active phases.

The model phase never recomputes posterior rows after model parameters change.
The recognition phase never leaves a gradient on model storage. A parameter
cannot belong to two optimizer bindings.

## Deterministic noise and exact resume

Training noise is counter-based and has no mutable global generator. The
domain is:

```text
vfe4.h6.training-rmc-normal.v1
```

The key contains:

```text
attempt_spec_sha256
pass_index
batch_index
phase
example_ordinal
sample_ordinal=0
draw_block
```

SHA-256 blocks expand to four little-endian uint64 words. Open uniforms and
paired Box-Muller normals reuse the already implemented
`EstimatorStream` mapping, under the new domain and a training-only purpose.
Normals are generated as CPU float64, reshaped in exact receiver/channel
order, then transferred to the training device. Recognition and model phases
have different keys and never reuse noise.

The cursor records the next exact phase and key coordinates plus the
counter-consumption digest. It does not claim that a digest alone is a
restorable RNG state. Resume reconstructs the next noise directly from the
key.

Checkpoint v3 stores:

- attempt spec v3 and cursor v3;
- objective/estimator/runtime identities;
- sorted named model and recognition tensors;
- sorted optimizer states keyed by stable qualified parameter name, not
  ephemeral object ID;
- every tensor as canonical contiguous CPU little-endian bytes with dtype,
  shape, and SHA-256;
- data permutation and next-batch coordinates;
- validation/checkpoint boundary counts.

Loading reconstructs fresh modules and optimizers, revalidates all names and
bytes, restores the cursor, and either resumes exactly or refuses. Tiny
uninterrupted and checkpoint/resume runs must produce byte-identical terminal
state and metric records.

## Runtime and device contract

The installed runtime is used as-is:

```text
python = C:/anaconda/python.exe
training_device = "cuda:0"
training_dtype = "float64"
validation_device = "cpu"
heldout_scoring_device = "cpu"
scoring_dtype = "float64"
```

The exact Python, Torch full version, CUDA runtime, device name, compute
capability, and deterministic-policy fields enter the runtime identity and
attempt spec. A training launch refuses when CUDA is unavailable or a required
float64 operation cannot execute. It does not silently fall back to CPU.

Canonical initialization is created on CPU float64, hashed, and then moved to
CUDA before optimizers are constructed. Checkpoints are published in canonical
CPU form. Validation and held-out scoring reconstruct a fresh CPU model from
that canonical state; recognition is never used for prior-predictive scoring.

Only bounded synthetic development tests may use CPU training fixtures. No CPU
production training campaign is authorized.

The existing `H6CausalTransformer` and its math-SDPA boundary remain the
strict CPU float64 scoring/reference implementation. A v3 training module is
explicitly device-aware and accepts only `cuda:0` float64 in production. At
process startup it enables deterministic algorithms, requires the frozen
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, calls
`torch.use_deterministic_algorithms(True)`, sets cuDNN benchmark false and
cuDNN deterministic true, disables CUDA/cuDNN TF32 and reduced-precision
float32 reductions, and disables flash and memory-efficient SDPA while
enabling math SDPA. It records and revalidates those live settings. A missing setting,
unsupported deterministic kernel, runtime/device mismatch, or attempted
fallback is a refusal. Exact-resume equality is promised only for the same
recorded Python/Torch/CUDA/device capability and deterministic-policy
identity.

## Experiment operations

`run_h6_experiment` replaces each blocker with one lazy bound implementation:

```text
plan
  -> reconstruct 12 endpoints, matching reports, tuning cells,
     attempt specs, and immutable experiment plan

train
  -> run or exactly resume one planned tuning/confirmatory attempt,
     publish validation boundaries and terminal checkpoint

score_validation
  -> CPU target-blind prior NLL for planned checkpoints,
     publish tuning selection and complete checkpoint set

score_test_transaction
  -> prove complete eligibility, durably reserve the sole opening,
     consume its process-local capability, score the exact discriminated
     inventory, close results, and publish pointers in the same process
```

The v3 launcher replaces the old split `reserve_test_opening`/`score_test`
surface with the single
`AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1` operation. The opening
capability is opaque, process-local, and nonserializable, so a reservation
created by one click cannot be scored by another. Historical split operations
cannot authorize v3.

Every other operation retains its existing exact authorization phrase. The
click-run launcher resolves a complete editable scientific dictionary and
immutable reference-registry path; it does not require CLI flags or hidden
environment scientific settings.

Training and scoring remain separable so a crash before the irreversible
opening does not consume it. After reservation, a crash or incomplete result
is terminally inconclusive and the opening is never recreated.

The transaction persists an authenticated journal bound to the experiment
config, readiness, data/test inventory, selected checkpoints, and opening
proof. Its only transitions are `RESERVED -> FINALIZED` or
`RESERVED -> INCONCLUSIVE`. The reservation uses the existing exclusive
create primitive. Result files may be individually atomic, but the design
does not claim an impossible cross-directory atomic rename. A restart that
observes `RESERVED` without the complete bound final result writes or
validates terminal `INCONCLUSIVE`; it never issues another capability.

## Data-store lifecycle across clicks

Acquisition remains a separate, explicit data operation. After the official
archive/member hashes and tokenized split hashes have been published, later
clicks reopen the blinded store through a new authenticated manifest API.
Reopening:

- verifies the archive, extracted members, tokenizer, split, and window
  manifests before returning a handle;
- exposes train and validation through their existing typed capabilities;
- retains test bytes behind the one-opening issuer;
- never republishes, replaces, or redownloads an existing sealed store;
- refuses a partial directory, a hash mismatch, or an already consumed
  opening.

The authenticated manifest is serializable. The test-opening capability is
not.

The v3 factory is
`reopen_authenticated_blinded_store_v3(manifest_path, artifact_root)`. It
reuses the existing archive/member/token/split revalidation, then constructs
and privately registers a fresh `BlindedCorpusStore` only after every
identity succeeds. Returning only a `DataIdentity` is insufficient. The
factory never opens test bytes and never reconstructs an opening capability.

## Held-out workload

The only test-byte consumers are:

- eight exact A0 corpus totals, one per confirmatory seed checkpoint;
- complete A5: `8 * 64 * 4 = 2,048` weighted-SMC corpus rows;
- emission-only A5: `8 * 64 * 4 = 2,048` weighted-SMC corpus rows.

The one opening therefore produces exactly:

```text
8 + 2,048 + 2,048 = 4,104 logical scoring rows.
```

The other nine trained endpoints cannot map or receive test bytes. Complete
A5 rows are reused unchanged for PRIMARY and OBJECTIVE; they are not rescored.
A0 has no particle count, replicate stream, Monte Carlo half-width, or SMC bias
bound. The weighted rows retain common streams and particle levels
`(128,256,512,1024)`.

The raw inventory becomes a discriminated union:

```text
exact_a0_corpus_total
weighted_a5_smc_corpus_total
```

It rejects 24,576-row legacy assumptions, a weighted A0 row, an exact A5 row,
or any held-out record for another endpoint. The final analysis reports only
claims supported by the new inventory; the remaining endpoint results are
validation-only disclosures.

## Schema changes

Active executable records use new identities:

```text
h6-prediction-config-v3
h6-prediction-readiness-v3
h6-training-schedule-v3
h6-attempt-spec-v3
h6-attempt-cursor-v3
h6-objective-manifest-v3
h6-checkpoint-v3
h6-raw-endpoint-inventory-v4
h6-prediction-metrics-v3
h6-prediction-result-v3
```

Legacy records remain readable only where an existing historical consumer
requires them. They cannot authorize the amended executable path.

The migration is additive and type-exact:

- add `H6PredictionV3ResolvedConfig` and
  `resolve_h6_prediction_v3_config`;
- add v3 readers/writers for readiness, schedule, attempt, cursor, objective,
  checkpoint, inventory, metrics, result, and the click-run envelope;
- keep v1/v2 parsers unchanged for historical reads;
- reject a v1/v2 typed object at every v3 dispatcher before any effect;
- bump `train_vfe4.py`'s editable dictionary and returned result envelope to
  v3, removing the two split test operations from the v3 operation union;
- add a v3 H6 reference and exact reopen validator to H8; do not weaken the
  existing H8 v2/legacy parsers.

Config v3 binds:

- recognition trajectory and categorical posterior schemas;
- exact categorical reduction and one-sample continuous estimator;
- training-noise domain and counter mapping;
- phase ownership;
- training/scoring devices and dtype;
- runtime and deterministic-policy identity;
- checkpoint codec;
- exact 4,104-row held-out contract.

## Matching v3

`h6-amended-matching-policy-v3` retains the v2 candidate grids, one-percent
parameter tolerance, five-percent whole-training arithmetic-FLOP tolerance,
two passes, batch size eight, sequence/stride 32, and first-lexicographic
hard-eligible selection. For an arm with recognition width `R` and trainable
source-bank count `B`, the parameter formula is exactly:

```text
P_v3(arm) = P_v2(arm) + B * (R + 1 + d).
```

The v3 FLOP ledger extends the existing named analytical terms with every
categorical-context residual dot product, lag scalar, support log-softmax,
rank-one component shift, component realization, exact nested
`sum q(log pi + log K - log q)` reduction, moment-projection construction
where selected, its recognition-phase backward, gradient clipping, and AdamW
update for those added parameters.
Multiplicities come only from the declared ragged support lengths, receiver
count, batches, phases, and passes. CPU-to-CUDA noise transfer, validation,
data I/O, checkpoint serialization, test scoring, prediction particle
propagation, and cache work remain explicitly excluded, matching the
capacity-comparison estimand rather than measured wall time. The formulas and
term inventory are hashed; no measured loss, gradient, timing, validation, or
test value may enter selection. All matching artifacts and readiness
identities are regenerated under v3.

## Checkpoint v3 hydration codec

The checkpoint codec enumerates `module.state_dict()` entries and optimizer
state by stable qualified parameter name in Unicode-codepoint sort order.
Every tensor record contains role, qualified name, dtype, shape, contiguous
row-major little-endian bytes, byte length, and SHA-256. Shared-storage aliases,
duplicate/case-colliding names, sparse tensors, unsupported dtypes, and
unbound optimizer parameters are rejected. Optimizer groups record their
ordered stable names and exact hyperparameters; AdamW moments and step values
are encoded as named tensor/scalar records rather than object IDs.

Hydration order is fixed:

1. construct fresh CPU float64 modules from the bound attempt spec;
2. validate the complete named state and parameter inventories;
3. construct optimizer groups from the stable names;
4. decode and load canonical module and optimizer state on CPU;
5. validate every byte/hash and the next-phase counter cursor;
6. move active modules and optimizer tensors to the authorized device;
7. resume at the exact recorded next phase.

The v3 cursor includes pass, batch, next phase, example/sample ordinals,
counter draw-block and consumption digest, permutation identity, and
validation/checkpoint boundary counts. Missing or extra state is a hard
failure.

## Verification strategy

Development remains milestone-scoped:

1. pure config/type/source-row tests;
2. tiny ELBO enumeration versus a monolithic log-ratio oracle;
3. gradient-ownership and snapshot tests;
4. counter-noise and checkpoint round-trip tests;
5. fake-data planner/orchestrator tests;
6. 4,104-contract artifact-consumer tests using tiny records and arithmetic
   counts, never materializing the production table;
7. exactly two tiny CUDA tests on the final stable implementation:
   - A0 uninterrupted versus checkpoint/resume;
   - A5 recognition update, detached snapshot, model update, and resume.

The two CUDA tests use `C:/anaconda/python.exe`, tiny synthetic batches, and
finish before any corpus campaign. No full suite is run after each task. One
broader bounded readiness/fixture milestone is reserved for the frozen source
candidate.

## Rejected alternatives

- **Tie categorical recognition to the generative prior.** This is a valid
  ablation but removes observation-driven source inference and is not the
  primary H6 law.
- **Sample categorical sources or use Gumbel-softmax.** Exact support summation
  is available and lower variance.
- **Unrestricted assignment-specific population Gaussians.** They require a
  global Cartesian component bank and defeat the bounded experiment. H6 uses
  normalized local source-conditioned components with shared precision and
  rank-one channel-specific mean shifts, then performs only factor-local exact
  sums.
- **Full dense population covariance.** It would invalidate H6 capacity
  matching and contradict the bounded sparse direction. H6 uses the declared
  receiver-factorized family.
- **Hand-derived complete gradients.** Reverse-mode autograd over the explicit
  ELBO is the safer implementation. Hand derivations remain independent
  oracles for tiny factors and categorical rows.
- **Install a pinned Torch.** The installed CUDA runtime is the execution
  authority; environment mutation is out of scope.

## Completion boundary

This amendment is source-complete only when every operation is implemented,
the two tiny CUDA paths pass at the final source revision, the 4,104-row
consumer contract is enforced, the click-to-run dictionary is populated with
the full schema, and independent review closes the implementation ledger.

Source completion does not itself claim a successful H6 empirical result.
Tuning, confirmatory training, the sole held-out opening, and scientific
publication remain explicit evidence operations at their recorded revisions.
