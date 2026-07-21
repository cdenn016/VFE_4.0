# VFE 4.0 Codebase Design

- **Date:** 2026-07-21
- **Status:** Approved architecture; written specification awaiting review
- **Target:** A greenfield, zero-dimensional VFE 4.0 language-model implementation
- **Primary user experience:** Click-to-run Python files with editable configuration dictionaries

## 1. Executive decision

VFE 4.0 will be implemented as a new probabilistic codebase. It will not be a
fork, refactor, or semantic relabeling of V3. The language-model implementation
will begin with the whitepaper's zero-dimensional base
$\mathcal C_0=\{\ast\}$, one normalized causal generative joint, one
normalized recognition law, and one authoritative state ELBO.

V3 is an engineering guide for launcher ergonomics, cached-token data loading,
training-loop boundaries, artifact publication, checkpointing, evaluation, and
report generation. V3's `BeliefState`, moving-peer objective, checkpoint
schema, configuration monolith, and split belief/decode objective are not VFE 4.0
model components.

The initial implementation uses block-coordinate variational EM:

1. exact CAVI or analytic Gaussian/source updates when the selected family
   admits them;
2. PyTorch reverse-mode autograd for coordinate derivatives of general E-like
   or M-like scalar-objective proposals, followed by the separately declared
   statistical or group-geometric update map;
3. a fixed-recognition M-like update by default, using an immutable recognition
   snapshot with disjoint parameter ownership and storage before
   model-parameter optimization;
4. no differentiation through an unrolled inference trajectory in the default
   profile; and
5. hand-derived formulas as independent oracles, not a second production
   gradient engine.

This is a hybrid design. Individual phases can be autograd-free, but the whole
language model will not be advertised as "backpropagation-free" when any
configured update uses autograd.

Terminology is strict throughout this design. A reverse-mode VJP or backward
sweep through the scalar objective is backpropagation. It is not a
"forward-gradient" method merely because the differentiated variables occur in
the forward recognition computation. Forward-mode AD means a JVP and is a
different derivative algorithm.

## 2. Source hierarchy and scope

The following precedence governs implementation questions:

1. `Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex` and its
   `Manuscripts/vfe4_whitepaper/` modules are the normative language-model
   theory.
2. `Manuscripts/MAgent_exact_elbo_whitepaper.tex` and
   `Manuscripts/magent_elbo_whitepaper/verification/` provide related finite
   ELBO derivations and reusable oracle patterns. They do not replace VFE 4.0's
   typed causal language model.
3. `C:/Users/chris and christine/Desktop/V3_Transformer` supplies rough
   engineering patterns from the live working tree. It does not supply VFE 4.0
   probability semantics.
4. The Research vault supplies cross-project context. The two target whitepaper
   trees in this repository were hash-matched to the current Research WIPs
   during the design investigation.

The normative source locations include:

- normalized causal factors and joint:
  `Manuscripts/vfe4_whitepaper/04_generative_model.tex:12` and `:69`;
- filtering, smoothing, information form, and sparse precision:
  `Manuscripts/vfe4_whitepaper/05_structured_information_form.tex:1`,
  `:75`, and `:392`;
- the authoritative ELBO and update semantics:
  `Manuscripts/vfe4_whitepaper/06_elbo_coordinate_updates.tex:42`,
  `:249`, `:355`, and `:857`;
- the V3 boundary and one-way initializer rule:
  `Manuscripts/vfe4_whitepaper/07_transformer_crosswalk.tex:113`;
- the H1--H8 promotion ladder:
  `Manuscripts/vfe4_whitepaper/08_hypotheses_limitations.tex:1`; and
- factor-assembly and analytic-oracle specifications:
  `Manuscripts/vfe4_whitepaper/09_appendices.tex:280` and `:349`.

## 3. Goals

The codebase must:

1. implement a normalized posterior-independent causal generative model;
2. implement normalized filtering and smoothing recognition families without
   confusing either with prior prediction;
3. evaluate the same complete ELBO for every update labeled as an E-like,
   M-like, exact-coordinate, MM, or generalized-EM update;
4. support the zero-dimensional language specialization while preserving
   enough type separation to add a positive-dimensional base later without
   retrofitting current objects;
5. represent correlated population recognition in sparse information form;
6. support exact small-problem enumeration and independent NumPy/SymPy oracles;
7. provide a click-to-run experience with editable configuration dictionaries
   and no required user-facing CLI;
8. produce revision-, configuration-, data-, and environment-bound artifacts;
9. fail closed on invalid support, target leakage, non-SPD factors, nonfinite
   objective values, stale schemas, or mislabeled update guarantees; and
10. advance through the manuscript's H1--H8 promotion gates rather than
    beginning with a large training run.

## 4. Non-goals for the first implementation

The initial implementation will not:

- implement a positive-dimensional contextual base, smooth base connection,
  base curvature, or base holonomy;
- identify internal node-frame coboundaries with a physical gauge field;
- implement the separate configuration-Gibbs random variable or its partition
  function;
- retrofit the V3 moving-peer energy into the fixed VFE 4.0 generative joint;
- directly load a V3 checkpoint;
- claim that source-posterior softmax is scaled dot-product attention;
- claim exactness for a moment-matched Gaussian replacement of a source
  mixture;
- use a global dense population covariance on a promoted execution path;
- introduce a registry/plugin system before H3 identifies stable extension
  seams;
- require a dashboard, notebook, shell flags, or configuration language; or
- claim predictive, computational, or scaling benefit before the corresponding
  preregistered gate passes.

## 5. User experience and entry points

Two root-level files define the initial user surface:

### 5.1 `verify_vfe4.py`

This file contains one editable nested `CONFIG` dictionary and a conventional
`if __name__ == "__main__": main()` block. Its modes run H1--H8 reference
fixtures and promotion checks. Clicking the IDE's Run button is sufficient.

### 5.2 `train_vfe4.py`

This file contains one editable nested `CONFIG` dictionary and the same
click-to-run entry pattern. It constructs the selected dataset, model,
recognition family, optimizer policy, artifact writer, and training operation
through importable package APIs.

Neither entry point uses `argparse`, Typer, Hydra, required environment
variables, or required shell flags. Importing an entry point does not start a
run. The launchers contain orchestration only; probability calculations,
training behavior, data access, artifacts, and reports live in the `vfe4`
package.

The launchers may expose clearly named constants such as `DATASET`,
`CACHE_ROOT`, `RUN_ROOT`, and `SEEDS` when that improves editability. Derived
values are computed during typed configuration resolution rather than by
mutating the dictionary after import.

## 6. Repository architecture

The planned repository shape is:

```text
VFE_4.0/
  train_vfe4.py
  verify_vfe4.py
  pyproject.toml
  README.md
  vfe4/
    config/
    types/
    numerics/
    geometry/
    generative/
    recognition/
    objective/
    inference/
    predictive/
    data/
    training/
    evaluation/
    artifacts/
    validation/
  verification/
    numpy_oracles/
    sympy_oracles/
  tests/
    unit/
    oracle/
    property/
    integration/
    promotion/
  docs/
    preregistrations/
    superpowers/specs/
    superpowers/plans/
  Manuscripts/
```

The package remains at repository root, as V3 does, so click-to-run launchers do
not require an editable package installation merely to resolve imports.
`pyproject.toml` defines dependencies and test/tool configuration, but the
entry-point contract remains a normal Python file.

### 6.1 Dependency direction

Dependencies flow in one direction:

```text
config + types
  -> numerics
  -> geometry + generative factors
  -> recognition + objective
  -> inference + prior prediction
  -> training + evaluation
  -> launchers and reporting
```

`verification/` can import public production interfaces to compare results.
Production modules cannot import `verification/` or `tests/`. Artifacts and
reporting consume immutable result objects; they do not reach back into model
internals.

Initial files should remain focused. Directories need not contain many modules
at bootstrap; a file is split when it acquires a second responsibility or
cannot be understood independently.

## 7. Configuration contract

The editable dictionary is parsed once into frozen typed sections:

- `RunConfig`: mode, seeds, device, dtype, deterministic policy;
- `DataConfig`: dataset identity, split, tokenizer/cache identity, windowing;
- `ModelConfig`: dimensions, vocabulary, parent sets, generative kernels,
  source priors, decoder, geometry sector;
- `RecognitionConfig`: filtering or smoothing, structured or factorized
  family, source treatment, estimator;
- `InferenceConfig`: coordinate schedule, iteration limits, convergence,
  damping, acceptance budget;
- `OptimizationConfig`: E-like and M-like update methods, learning rates,
  autograd scope, gradient clipping;
- `ValidationConfig`: requested H gates, oracle precision, error budgets; and
- `ArtifactConfig`: run root, checkpointing, reports, figure policy.

Every resolved configuration contains:

- `schema_version`;
- `objective_schema_version`;
- explicit latent dimensions `d_z` and `d_m`;
- parent sets and source support;
- geometry and recognition tags;
- update labels;
- `expected_autograd_scope`;
- estimator and precision settings; and
- artifact/provenance policy.

The parser rejects:

- unknown keys;
- incompatible dimensions or representations;
- empty causal parent sets where a source is required;
- nonnormalized or negative source priors;
- all-invalid source rows;
- target-dependent generative source priors;
- unsupported combinations of recognition family and solver;
- claims of exact coordinate ascent for an approximate update;
- general `GL(K)` covariance paired with a diagonal-only family unless an
  explicit projection/approximation profile is selected; and
- resume artifacts with incompatible schema or objective versions.

The canonical resolved configuration is serialized before execution and
hashed into the run identity. Runtime code receives typed objects and never
reads the mutable global dictionary.

## 8. Core mathematical and software types

The implementation must keep the following concepts distinct.

### 8.1 Structural and geometric data

`StructuralData` contains the causal graph, parent sets, sequence labels,
declared representations, and deterministic geometry. These data condition
the model; they are not posterior variables.

`ZeroDimensionalBase` is the only implemented base in the first program. It
has no nontrivial base connection, curvature, or holonomy.

`PopulationFrames` represents separately declared same-point internal frames.
Its full `G^{T+1}` action is an internal reparameterization; the diagonal `G`
subgroup is the base gauge action in the singleton specialization.

The first reference fixtures keep the primary group elements `U_t` fixed as
deterministic geometry. Learned population frames are introduced only after the
analytic oracles pass and belong to an explicit M-like deterministic-geometry
parameter block. A local chart `U_t = exp(phi_t)` may be selected for a bounded
profile, but `U_t` remains the primary object; the profile must record the chart
domain and cannot imply that one real exponential chart covers all of
`GL^+(K)`.

Optional independent graph links are a separate future type. They cannot be
stored in `PopulationFrames` or inferred from a causal graph.

### 8.2 Generative model

`GenerativeModel` owns:

- normalized initial state and model factors;
- normalized categorical source priors;
- normalized state and model transition kernels;
- normalized categorical emissions; and
- parameters required to transform the complete law under an internal frame
  change.

The generative model cannot receive a `RecognitionLaw` argument. Changing a
source prior from fixed to prefix-conditioned changes the model and invalidates
all affected H-gate evidence, including H1.

### 8.3 Recognition law

`RecognitionLaw` is a normalized distribution over the complete latent state,
model state, and source variables. Filtering and smoothing are different
types or tagged variants with explicit conditioning contracts.

Structured Gaussian components use joint information coordinates. A source
assignment can index a component, so marginalization generally yields a
mixture. A moment projection is represented as an approximation result with
an error record, not as `RecognitionLaw` equality.

### 8.4 Sparse precision interface

Promoted code depends on a factor interface with operations equivalent to:

```python
class PrecisionFactor(Protocol):
    def solve(self, rhs: Tensor) -> Tensor: ...
    def logdet(self) -> Tensor: ...
    def selected_inverse(self, blocks: Sequence[Block]) -> Mapping[Block, Tensor]: ...
    def sample(self, noise: Tensor) -> Tensor: ...
```

Dense inverse and covariance construction are allowed only inside bounded
small-problem oracles. Production objective, inference, and diagnostics use
solves, factorizations, selected blocks, log determinants, or samples.

### 8.5 ELBO result

Every evaluation returns an immutable `ElboTerms` record containing at least:

- expected categorical log likelihood;
- initial-state and initial-model contributions;
- state-source categorical KL;
- model-source categorical KL;
- state-transition contribution;
- model-transition contribution;
- joint recognition entropy or its correctly partitioned equivalent;
- estimator error metadata; and
- the complete scalar ELBO.

The complete scalar is computed from the terms in one place. Training,
validation, and reporting do not reconstruct different versions of it.

### 8.6 Prior predictor

`PriorPredictor` computes the causal next-token distribution before the target
is observed. Its public call has no `RecognitionLaw` parameter. Posterior
predictive reconstruction is exposed separately and cannot be reported as
held-out prior prediction.

### 8.7 Update attempt

Every nontrivial update returns an `UpdateAttempt` with:

- the update label: exact coordinate, MM, generalized EM, natural-gradient
  proposal, SGD, Adam, or other explicit class;
- variables and parameter blocks affected;
- observed autograd scope;
- objective/estimator state before and after the proposal;
- deterministic or stochastic error budget;
- acceptance or rejection;
- convergence/line-search metadata; and
- any SPD projection, damping, or approximation applied.

This prevents an optimizer name or coordinate system from implying a guarantee
the executed update does not possess.

## 9. Autograd and gradient policy

### 9.1 Default rule

Let `S` denote the scalar being differentiated: either the ELBO to maximize or
the VFE `F = -ELBO` to minimize. When `S` depends on many active coordinates,
PyTorch reverse-mode autograd is the default derivative engine. A call of the
form

```python
g_mu, g_sigma_parameter, g_phi = torch.autograd.grad(
    S, (mu, sigma_parameter, phi)
)
```

performs one reverse-mode VJP with scalar seed one and returns the coordinate
components of the differential with respect to the tensors that are actually
stored. It computes gradients *of a scalar produced by the forward
calculation*; it is not forward-mode AD and the result is not itself an update
rule. Only tensors owned by the active block appear in the call.

This derivative engine is easier to keep aligned with the implemented scalar
and less error-prone than maintaining hand-derived derivatives for every
coupled transition, source, decoder, and frame-dependent term. It is still
backpropagation. Exact coordinate solutions take precedence, and independent
hand derivations remain verification oracles.

### 9.2 Exact coordinates remain explicit

When a coordinate optimum is available in closed form, production code uses
the exact update rather than taking a generic gradient step. Examples include
small conjugate Gaussian blocks and normalized source-coordinate updates under
their complete applicable score.

Exact formulas are also implemented independently in NumPy/SymPy oracle code.
The oracle does not share the PyTorch derivative graph.

### 9.3 General E-like updates

When a nonconjugate emission or restricted recognition family removes a
closed-form coordinate:

1. evaluate the complete ELBO or a declared valid surrogate;
2. obtain the proposal gradient with `torch.autograd.grad`;
3. apply the declared optimizer, natural-gradient preconditioner, damping, or
   line search;
4. re-evaluate the complete objective under the declared error budget; and
5. label and accept the proposal according to its actual guarantee.

Using natural coordinates does not by itself make the step a natural gradient.
Using autograd does not make a finite step coordinate ascent.

### 9.3.1 Coordinate covectors are not geometric update vectors

For a directly parameterized Gaussian `q = N(mu, Sigma)`, reverse-mode AD can
return the Euclidean/Frobenius coordinate covectors

```text
g_mu    = partial F / partial mu
g_Sigma = sym(partial F / partial Sigma).
```

Under the Gaussian Fisher metric, the corresponding natural-gradient vectors
for VFE minimization are

```text
ng_mu    = Sigma @ g_mu
ng_Sigma = 2 * Sigma @ g_Sigma @ Sigma,
```

and the descent direction is their negative. For ELBO maximization the sign is
reversed. A finite covariance proposal must still use the declared SPD-safe
path and acceptance rule.

The promoted structured-Gaussian path stores information coordinates and a
sparse precision factor, not a global dense `Sigma`. Autograd therefore returns
derivatives with respect to the stored `(h, J)` representation or its stable
factor parameters. Conjugate blocks use the exact information-coordinate target.
For a nonconjugate block, the initial promoted profile labels its proposal as
Euclidean descent in the stored chart and subjects it to sparse SPD
backtracking and complete-objective acceptance. It does not claim a natural
gradient. A Fisher label for a sparse precision subfamily is permitted only
after the restricted Fisher operator, its solve or projection, and preservation
of the declared sparsity pattern have independent derivations and H-gate
coverage; the full-Gaussian Fisher map alone is insufficient.

Symmetric precision covectors use the Frobenius pairing, implemented with an
`svec` convention whose independent off-diagonal entries are scaled by
`sqrt(2)`. An unscaled `vech` implementation is permitted only with its explicit
factor-two off-diagonal pullback and matching oracle; the two conventions cannot
be mixed.

For an exact target `(h_star, J_star)`, damping uses the SPD convex path
`(1-rho)*(h, J) + rho*(h_star, J_star)`. For a general sparse proposal it tests
`h + rho*delta_h` and `J + rho*sym(delta_J)` over `rho = 2**(-k)`. The step is
accepted only if the declared sparse pattern is preserved, sparse
factorization succeeds within the pivot/condition envelope, and the complete
objective passes its error-bounded acceptance test. The bounded dense oracle
also checks equivalence with the manuscript's whitened-increment formulation.

If a bounded profile stores a Cholesky factor `L`, autograd returns
`partial F / partial L`; that is not interchangeable with
`partial F / partial Sigma` or `partial F / partial J`, and Euclidean
optimization of `L` is not labeled an SPD natural-gradient step.

### 9.3.2 Frame-coordinate derivatives

The first reference profile keeps population frames fixed, so it has no
`g_phi`. The default later learned-frame M-like profile stores the primary group
element `U` directly. Reverse mode supplies its ambient Frobenius covector
`g_U`. For the initial learned-frame scope `G = GL^+(K)`, use the explicitly
selected left-invariant Frobenius group metric and the left trivialization
`delta_U = U @ xi`. The body gradient is `xi = U.T @ g_U`; VFE descent proposes
`U_new = U @ matrix_exp(-eta * xi)`. This is labeled left-trivialized
Frobenius group descent, not a Fisher natural gradient. Its covariance under
the complete declared internal-frame action remains an H7 obligation rather
than an optimizer-name guarantee. The complete-objective acceptance and
frame-condition checks still apply. A later proper subgroup requires an
explicit projection of `U.T @ g_U` onto its Lie algebra and separate evidence.

An opt-in bounded local-chart profile may instead use
`U = matrix_exp(phi)`. Differentiating through that map gives

```text
g_phi = D exp_phi^* [g_U],
```

the pullback of the group-element covector through the exact Frechet derivative
of the matrix exponential under the chosen coordinate pairing. PyTorch can
compute this pullback by reverse mode. It is an ordinary local-chart covector,
not automatically a Fisher natural gradient, a gauge-invariant direction, or a
left/right-trivialized group velocity. Any geometric frame update must name a
group metric or preconditioner, convert the covector to the appropriate tangent
vector, and apply a declared group retraction. A direct Euclidean step on
`phi` is labeled as chart descent. A chart-pullback group profile additionally
requires a declared group-metric operator `M_U`, the pullback metric
`M_phi = D exp_phi^* @ M_U @ D exp_phi`, and a solve
`M_phi @ v_phi = g_phi`. It must verify that `D exp_phi` is injective and within
a frozen conditioning envelope, then name the retraction and any recharting
operation before the profile can be enabled.

### 9.4 M-like updates

The default M-like step treats the accepted recognition law as fixed. It
materializes an immutable `RecognitionSnapshot`, verifies that the snapshot
does not alias trainable model storage, detaches its tensors, and differentiates
the complete expected joint terms with respect to selected model parameters.
Detachment alone is insufficient when recognition and model blocks share
trainable storage or recognition is recomputed during the M-step. Disjoint
parameter ownership and the immutable snapshot are part of the default
block-coordinate variational-EM contract.

Learned recognition parameters, when present, are updated in an E-like
parameter block with the model parameters frozen. Decoder and transition
parameters are updated in the M-like block with recognition fixed.

### 9.5 Unrolled and implicit differentiation

Differentiating through an inference trajectory is deferred to a separately
named `unrolled_inference` or `implicit_inference` profile. Enabling one
changes memory use, optimization semantics, and evidence freshness. It requires
its own tests and cannot inherit the default profile's H5 record.

Forward-mode JVPs through `torch.func.jvp` return directional derivatives such
as `DS[delta_mu, delta_Sigma, delta_phi]`. They are reserved for implicit
solvers or local sensitivity diagnostics where that shape is advantageous.
Recovering every component of the gradient of one scalar by forward mode would
require a basis sweep (or its vectorized equivalent), so it is not the default
for the high-dimensional scalar ELBO/VFE.

### 9.6 Manual/custom backward

A custom `autograd.Function` or hand-derived production gradient is introduced
only when:

1. profiling identifies a concrete memory or runtime bottleneck;
2. the forward operation and derivative domain are explicit;
3. double-precision `gradcheck` passes;
4. the derivative matches an independent analytic or finite-difference oracle
   inside a declared conditioning envelope; and
5. boundary, singular, and failed-factorization behavior is tested.

### 9.7 Gradient observability

Every run records:

- `expected_autograd_scope` from the resolved configuration;
- `observed_autograd_scope = "none" | "e_step" | "m_step" | "e_and_m"`
  from runtime instrumentation, with a hard failure on mismatch;
- whether accepted inference outputs were detached;
- whether any trajectory was unrolled;
- optimizer/update labels by parameter block;
- gradient norms and nonfinite counts;
- proposal acceptance counts; and
- objective changes with estimator error budgets.

## 10. Numerical policy

The implementation uses:

- `logsumexp` and `log_softmax` for categorical normalization;
- support masks before normalization;
- log ratios only on positive-prior support;
- hard failure for an all-invalid row;
- Cholesky or another declared stable factorization for SPD operations;
- linear solves instead of explicit inverses;
- float64 for normative identity oracles;
- an explicit condition-number envelope for frame-covariance tests; and
- fail-closed checks for NaN, Inf, singular frames, non-SPD precision, invalid
  determinants, and unsupported reference-measure changes.

Training precision can later use float32, bfloat16, or mixed precision only
after the float64 reference path passes and the lower-precision error budget is
calibrated against it.

## 11. Runtime flows

### 11.1 Verification flow

1. Parse and snapshot configuration.
2. Construct the exact bounded fixture for the selected H gate.
3. Evaluate the independent oracle.
4. Evaluate the production implementation through public APIs.
5. Compare all declared terms, not only the final scalar.
6. Write a machine-readable gate result and JUnit record.
7. Mark the gate pass, fail, or inconclusive using its frozen error budget.
8. Permit the next promotion stage only when required predecessors pass on the
   same relevant artifact/configuration family.

### 11.2 Training flow

1. Load identity-bound cached tokens and deterministic split metadata.
2. Form causal token windows and separate prefix inputs from scored targets.
3. Construct target-blind generative priors and source support.
4. Construct or update filtering/smoothing recognition under its declared
   observation access.
5. Evaluate the complete ELBO and term record.
6. Perform exact or accepted E-like updates with model parameters frozen.
7. Materialize the immutable, nonaliasing recognition snapshot and detach its
   tensors in the default profile.
8. Perform the autograd M-like proposal on selected model parameters.
9. Apply finite-value, gradient, SPD, and acceptance checks.
10. Log metrics, update checkpoints atomically, and preserve resume state.
11. Run prior-predictive validation without passing target-conditioned
    recognition into the predictor.

### 11.3 Evaluation flow

Teacher-forced held-out log likelihood is computed from the causal
prior-predictive distribution. Per-token negative log likelihood and
perplexity derive from that score. The ELBO, posterior-predictive
reconstruction, free-running ancestral samples, and approximation diagnostics
are reported separately.

## 12. V3 reuse boundary

VFE 4.0 can copy or adapt the following V3 patterns after line-level review:

- thin click-run orchestration from
  `C:/Users/chris and christine/Desktop/V3_Transformer/train_vfe3.py`;
- frozen typed configuration construction and unknown-key failure;
- split-aware cached-token loaders and causal windows;
- dependency-injected `train`, `train_step`, and `evaluate` boundaries;
- atomic JSON/CSV publication;
- config-bound best and resume checkpoints;
- run identity and provenance;
- isolated figure/report workers; and
- launcher tests that instantiate the live editable dictionary and shrink only
  dimensions for a one-step fixture.

The following are prohibited as direct reuse:

- V3 `BeliefState` as the VFE 4.0 joint state;
- V3 moving-peer divergences as normalized generative transitions;
- V3 source weights as persistent source posterior variables;
- V3 objective assembly or checkpoint schema;
- the 161-field V3 configuration monolith;
- signature-inspection registry dispatch;
- V3 model-specific toggle families; and
- direct V3 checkpoint loading.

A later one-way initializer can copy dimension-compatible embeddings, decoder
weights, token statistics, or internal frame data. It must record the map and
cannot manufacture joint precision blocks, normalized transitions, source
posterior state, or evidence that the two models are equivalent.

## 13. Artifacts and provenance

Each run owns:

```text
runs/<run_id>/
  config.json
  provenance.json
  environment.json
  metrics.jsonl
  validation/
    h1.json
    ...
    h8.json
  checkpoints/
  figures/
  logs/
  manifest.sha256
```

`provenance.json` records:

- Git revision and dirty-state digest;
- canonical configuration and objective-schema hashes;
- Python, PyTorch, CUDA, and dependency versions;
- device and numerical precision;
- exact model, data, and estimator seeds;
- dataset, tokenizer, and cache hashes;
- start/end timestamps and wall time;
- passed, failed, inconclusive, and invalidated gates;
- update labels and autograd scope; and
- parent run/checkpoint identity for resume.

Writes use temporary files plus atomic replacement. A checkpoint contains
model, recognition, optimizer, scheduler, RNG, data cursor, schema, and
provenance state needed for a faithful resume. Best checkpoints are optional;
final evaluation explicitly records whether it used a best checkpoint, a
requested checkpoint, or the live terminal state.

## 14. Error handling

The code fails closed in the following cases:

- invalid or unknown configuration;
- missing or identity-mismatched data cache;
- empty or all-invalid source support;
- target or suffix access in prior prediction;
- invalid recognition normalization or support;
- non-SPD precision/covariance;
- singular or inadmissibly conditioned frame in an identity oracle;
- nonfinite loss or gradients;
- an objective-decreasing update labeled exact coordinate or valid MM beyond
  its error budget;
- an unaccepted generalized-EM proposal;
- dense population-covariance allocation on the H8 path;
- stale checkpoint, objective, or config schema;
- artifact identity mismatch; or
- attempted reuse of invalidated gate evidence after a relevant source,
  dependency, estimator, data, or configuration change.

Expected experimental failures are written as structured results, not hidden
by fallback behavior. In particular, an all-invalid source row never becomes a
uniform row, and failed factorization never becomes an unchecked dense inverse.

## 15. Verification and promotion ladder

All gates are initially unimplemented and unverified.

| Gate | Purpose | Required implementation outcome |
|---|---|---|
| H1 | Analytic ELBO identity | On the specified `T=2`, `d_z=d_m=1`, vocabulary-3 fixture, monolithic log ratio, local decomposition, and evidence-plus-posterior-KL agree within a calibrated budget. |
| H2 | Information/moment equality | The same Gaussian law and complete ELBO agree between `(h,J)` and `(\mu,\Sigma)` without using dense inverse on the promoted path. |
| H3 | Structured-posterior adequacy | Structured recognition closes more of the known gap on a coupled synthetic target, and the gain disappears in an independently generated zero-coupling control. |
| H4 | Information-form cost | Information and moment solvers reach the same optimum; a preregistered primary cost endpoint determines whether a benefit exists. |
| H5 | Update coherence | Exact-coordinate and valid-MM updates do not decrease the complete ELBO; generalized-EM proposals are directly accepted or rejected. |
| H6 | Prefix safety and prediction | Finite leakage and source-mask oracles pass, a static causal-dataflow audit passes, and only then do matched predictive controls run. |
| H7 | Internal frame covariance | The complete law and ELBO, including decoder, transition, covariance/precision, morphism, and Jacobian transformations, pass bounded float64 residual tests. |
| H8 | Sparse scale | The `T=128`, `K=20` reference completes without a global dense population covariance or equivalent quadratic buffer. |

The promotion order is H1/H2, H3, H4/H5, H6, H7, then H8. Minimal
frame-coboundary operations exist earlier because H1 exercises them, but the
implementation is not called covariant until H7 passes.

Every test total and failure count used for closure comes from current
machine-readable output such as JUnit XML. Numerical agreement supports an
implementation identity; it does not replace a mathematical derivation.

## 16. Test architecture

### 16.1 Unit tests

Cover typed dimensions, masks, normalized kernels, SPD validation, factor
assembly, coordinate conversion, ELBO term accounting, config parsing,
checkpoint schemas, and artifact atomicity.

### 16.2 Independent oracle tests

Use NumPy/SymPy implementations derived from the manuscripts. They compare
against PyTorch production results on small fixtures without sharing
production helper functions that could reproduce the same bug.

### 16.3 Gradient tests

For smooth bounded fixtures:

- compare autograd with hand-derived analytic gradients where available;
- compare both with central finite differences inside a declared conditioning
  envelope;
- use double-precision `gradcheck` for custom differentiable operations;
- check zero, masked, singular-near-boundary, and source-mixture cases; and
- verify that the default M-step has no tensor or parameter alias and no
  unintended gradient path into accepted recognition state.

### 16.4 Property tests

Cover normalization, source support, prefix invariance, information/moment
round trips, inverse congruence for precision, population-frame
change-of-variables, and monotonicity/acceptance contracts.

### 16.5 Integration tests

Cover both click-run dictionaries, one complete small verification run, one
tiny cached-data training run, checkpoint/resume equivalence, prior-predictive
evaluation, artifact manifests, and figure-worker isolation.

### 16.6 Promotion tests

H1--H8 tests remain separate from routine unit tests because they bind exact
fixtures, configurations, numerical budgets, and evidence records. Promotion
tests can fail, pass, or be inconclusive; routine tests are ordinary software
regressions.

## 17. Milestone decomposition

The implementation plan will decompose the work into these self-contained
milestones after this specification is approved:

1. repository/package/launcher/config/provenance foundation;
2. H1/H2 exact finite reference core;
3. H3 structured and factorized recognition;
4. H4/H5 sparse precision and update semantics;
5. H6 zero-dimensional language sidecar and matched controls;
6. H7 complete internal-frame covariance;
7. H8 sparse scale and allocation audit; and
8. separately approved extensions.

Each milestone will cite exact manuscript/V3 patterns, identify files and
public interfaces, include a verification checklist, and include anti-pattern
guards. A later milestone cannot silently weaken an earlier gate.

## 18. Alternatives considered

### 18.1 V3 fork with objective replacement

Rejected. It offers the fastest path to a training loop but carries the wrong
state, objective, config, and checkpoint semantics. Hidden V3 assumptions would
make exact ELBO closure harder to establish than a greenfield core.

### 18.2 Sealed analytic sidecar promoted later

Not selected as the primary architecture. It minimizes the first probe but
creates an avoidable migration boundary from dense prototype objects to the
production package. Instead, independent oracles remain sealed while
production interfaces exist from H1 onward.

### 18.3 Registry/plugin architecture from day one

Deferred. It expands invalid configuration combinations and encourages
toggle-driven objective drift before stable families exist. Registries can be
introduced after H3 at seams demonstrated by at least two real
implementations.

### 18.4 Fully hand-derived production gradients

Rejected as the default. They create a second implementation of every forward
term, increase maintenance burden, and are especially fragile across source
mixtures, nonconjugate emissions, and geometry-dependent factors. Analytic
gradients remain valuable as independent oracles and for exact coordinates.

## 19. Specification acceptance criteria

The design is ready for implementation planning when the reviewer agrees that:

- VFE4, MAgent, and V3 roles are unambiguous;
- click-to-run/no-required-CLI behavior is explicit;
- the normalized joint, recognition, prior-prediction, and configuration-Gibbs
  boundaries cannot be conflated by the proposed interfaces;
- the default gradient and detachment semantics are explicit;
- exact, generalized-EM, natural-gradient, SGD, and Adam labels cannot imply
  false guarantees;
- H1--H8 control promotion;
- sparse precision is the promoted interface;
- artifacts bind code, config, data, estimator, and environment;
- there are no placeholders, deferred design choices disguised as
  requirements, or claims of current implementation success; and
- implementation remains out of scope until a separate phased plan is
  reviewed.

## 20. Current evidence status

At specification time:

- the repository has no VFE 4.0 production package or launcher;
- the MAgent finite oracle source is present and supplies reusable reference
  patterns;
- the revision-bound source hashes, V3 inspection state, oracle command, JUnit
  counts, and collection failure are recorded in
  `docs/investigations/2026-07-21-vfe4-buildout-investigation.md`;
- the same tests do not collect from this repository's current
  `Manuscripts`-capitalized layout because their import expects lowercase
  `manuscripts`;
- none of H1--H8 is implemented for VFE 4.0; and
- no predictive, computational, covariance, or scaling claim is closed.

These facts define the starting point. They are not evidence that the planned
implementation already works.
