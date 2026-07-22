# VFE 4.0 H4 Cost and H5 Update-Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a preregistered empirical H4 comparison of independent information- and moment-form Gaussian solvers and a separate deterministic H5 verification gate that proves every labeled update evaluates the complete affected objective and obeys its label-specific acceptance or rollback contract.

**Architecture:** H4 builds one immutable neutral Gaussian problem per fixed seed, passes the same factors and protocol to two genuinely independent production solvers, verifies terminal-law equivalence outside the timer, and uses seed-level paired timing ratios for inference. H5 keeps H2's detached immutable evaluation seam intact, constructs gradient proposals in a separate differentiable working representation, freezes each candidate before complete-objective evaluation, and records a closed update taxonomy plus factor-dependency and rollback evidence. The unified click-run publishes distinct `validation/h4.json` and `validation/h5.json` payloads and distinct `GateResult` objects, even though one exact revision, one JUnit run, one click artifact, and one revision-specific claim ledger form the coupled H4/H5 milestone.

**Tech Stack:** Python 3.10+, PyTorch float64 on CPU with one intra-op thread, NumPy float64, SciPy-free deterministic statistics, `time.perf_counter_ns`, pytest, JUnit XML, SHA-256 provenance, atomic JSON artifacts.

## Global Constraints

- Begin only after the approved H3 plan is implemented and H1--H3 pass through the single ordered click-run at the candidate source revision. Preserve every H1/H2/H3 fixture, result meaning, artifact payload, and ledger file.
- H4 and H5 are separate gates. They have separate immutable result types, status decisions, invariant sets, obligations, preregistrations, and `validation/h4.json` / `validation/h5.json` payloads. A shared implementation milestone must never collapse them into one status or let one gate compensate for the other.
- H4 is an empirical cost hypothesis. H5 is implementation verification. Neither licenses H6 language/predictive claims, H7 covariance claims, H8 sparse-scaling claims, or training.
- Preserve the immutable detached H2 seam: `PrecisionFactor`, `DenseCholeskyPrecision`, `InformationGaussian`, `assemble_*_information`, and H2 evaluation results remain non-differentiable snapshot/evaluation objects. Do not add `requires_grad` leaves, optimizer ownership, or a mutable covariance property to them.
- H5 gradient proposals use a separate differentiable working representation. Before complete-objective comparison, convert the proposed working state to a finite, detached, cloned, nonaliasing immutable candidate snapshot. Accepted snapshots replace live state only through the acceptance controller; rejected candidates never become live.
- Reuse H3's neutral normalized Gaussian-factor semantics and independent exact-posterior oracle as the small correctness anchor. Do not alter H3 fixture bytes or select H4 problem coefficients from observed timings.
- H4 correctness anchor: the frozen H3 coupled and independently authored zero-coupling fixtures, `T=1`, `d_z=d_m=1`, `D=4`. Both H4 solvers must reproduce the same exact optimum and H3 reference quantities before any scaled timing is eligible.
- H4 scaled suite: `d_z=d_m=4`, `T in {7,15,31}`, hence `D in {64,128,256}` in population-major `[z_t,m_t]` order. `D=256` on the coupled problem family is the sole primary timing endpoint. The smaller dimensions and every zero-control timing are secondary diagnostics.
- H4 uses exactly 20 fixed shared problem seeds, frozen in the preregistration and typed configuration. Each seed has one independently generated coupled problem and one matched zero-coupling control with the same marginal noise, offsets, observation map, factor order, and seed identity; only the declared transition coupling blocks differ.
- Both H4 arms consume the same immutable neutral factor object, initial law, objective definition, factor schedule, stopping rule, dtype, CPU device, thread setting, and process environment. Arm order is deterministic AB/BA. An arm must not regenerate, reorder, mutate, or cache a different problem.
- The information arm assembles and solves in information coordinates. The moment arm constructs and updates a joint moment law directly through affine-Gaussian propagation and conditioning. The moment arm must not call the information solver, canonical assembler, `InformationGaussian`, or obtain a covariance by inverting the information arm's final precision.
- H4 performs common validation, tensor materialization, exact-oracle construction, CPU/thread/affinity inspection, factor hashing, and condition diagnostics outside the timer. Do not time H2's diagnostic `eigvalsh`, and do not run an unbalanced diagnostic in only one arm.
- The timed region begins immediately before construction of fresh arm-native solver state and ends after the arm exhausts the identical one-pass factor schedule, performs its arm-native finite/SPD checks, and evaluates the common objective in its native representation. It includes arm-native factor assembly/propagation, solves/factorizations, and objective evaluation. It excludes problem generation, exact-oracle work, hashing, condition-envelope checks, conversion of either native terminal law into the common H4 comparison record, selected-moment extraction for equivalence, garbage collection setup, artifact serialization, bootstrap statistics, and diagnostic memory passes. The information arm therefore is not rewarded for already storing `J`, and the moment arm is not penalized by timing a comparison-only conversion to `J`.
- Freeze scaled-problem traversal as `for horizon_index, horizon in enumerate((7,15,31))`, then `for seed_index, seed in enumerate(H4_PROBLEM_SEEDS)`, then `for kind_index, kind in enumerate(("coupled","zero_control"))`; assign zero-based `problem_index` in exactly that order. For each problem, run all three untimed warmup pairs and then all 11 timed pairs. Pair `pair_index` uses information-then-moment (AB) exactly when `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, otherwise moment-then-information (BA), where warmups use pair indices `0,1,2` and timed repetitions use `3+repetition_index`. Warmups verify execution only and never count toward timed or inferential order balance. For the primary `horizon_index=2`, `kind_index=0` (`D=256`, coupled) endpoint, odd `seed_index` values have exactly six AB and five BA timed pairs, even `seed_index` values have exactly five AB and six BA timed pairs: ten seeds of each pattern and exactly 110 AB plus 110 BA primary timed pairs in aggregate. Retain every raw nanosecond timing and its exact order. Repetitions are not inferential units.
- CPU float64 and one PyTorch intra-op thread are mandatory for H4. Record PyTorch inter-op threads, NumPy/PyTorch BLAS configuration, relevant thread environment variables, processor identity, OS, process affinity, clock name/resolution, power-policy fields when available, and observed thread counts. Missing mandatory CPU/float64/one-thread facts makes H4 `INCONCLUSIVE`; do not silently substitute a different environment.
- H4's per-seed primary statistic is `median(11 information times) / median(11 moment times)` for that seed's `D=256` coupled problem. The aggregate estimate is `exp(mean(log(seed_ratio)))`. Compute a deterministic 95% paired percentile-bootstrap interval by resampling the 20 seed-level log ratios with replacement for exactly 100,000 replicates using frozen bootstrap seed `20260721`, then exponentiating the 2.5th and 97.5th percentiles. Seeds, not repetitions, are the inferential units.
- H4 support threshold: if terminal equivalence passes and the upper 95% bound is below or equal to `0.80` while the interval is not the degenerate point `[0.80,0.80]`, H4 is `PASS`. If equivalence passes and the lower bound is greater than or equal to `0.80` while the interval is not that degenerate point, H4 is `FAIL` (no supported benefit). An interval crossing `0.80`, including the degenerate exact-boundary case, is `INCONCLUSIVE` with an explicit precision obligation.
- H4 equivalence is a prerequisite on exact-posterior gap, terminal information vector `h`, precision `J`, selected means/covariance blocks, and complete objective. Every comparison uses its own operand-shaped allowance. A finite decisive miss is `FAIL`; missing/nonfinite output, an indecisive allowance, an environment/protocol mismatch, or an incomplete repetition table is `INCONCLUSIVE`.
- Freeze the H4 scaled-suite admissibility envelope at `lambda_min(J) >= 1e-6`, `lambda_max(J) <= 1e6`, `kappa_2(J) <= 1e8`, minimum Cholesky pivot `>= 1e-3`, `||mu||_inf <= 16`, and every moment-arm innovation covariance satisfying the same eigenvalue/condition bounds at its local dimension. Bounds are inclusive. Any problem, exact oracle, or terminal arm outside the envelope is `INCONCLUSIVE`; do not jitter, clip, pseudo-invert, repair, or silently omit it.
- Freeze `H4_SOLVER_RELATIVE_BUDGET=1e-9` and `H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION=1e-4`. Each solver-produced operand contributes exactly `1e-9 * invariant_scale` once; oracle operands contribute no solver term. Each comparison records `invariant_scale=max(1, every compared scalar absolute value or vector/matrix infinity norm)`, its rounding and solver contributions, final allowance, and `allowance/invariant_scale`. Eligibility requires the ratio to be strictly less than `1e-4`; equality or a larger ratio is `INCONCLUSIVE`. No invariant borrows another invariant's scale or condition number.
- H4 retains raw times. Peak memory and real-operation counts are secondary and are collected in separate untimed diagnostic passes using the same arm wrappers. They cannot rescue or overturn the primary timing decision and are not H8 sparse-allocation evidence.
- Instrument real operations symmetrically through one shared `InstrumentedLinearAlgebra` facade used by both arms. A `NullOperationRecorder` is used in timed runs and a `CountingOperationRecorder` in untimed diagnostic runs. The recorder may observe an operation only inside the wrapper that actually executes it; no solver may emit estimated or formula-derived counts as if they were runtime operations.
- H5's closed update taxonomy is exactly `exact_coordinate`, `valid_mm`, `generalized_em`, `natural_gradient_proposal`, `sgd_proposal`, `adam_proposal`, and `truncated_iteration`. Unknown aliases and optimizer-name inference fail configuration resolution.
- An `exact_coordinate` or enabled `valid_mm` attempt may be accepted only when `delta_elbo >= -epsilon_delta`. A `generalized_em` attempt may be accepted only when the observed complete-objective change is resolved positive: `delta_elbo > epsilon_delta`. Natural-gradient, SGD, Adam, and truncated-iteration labels carry no monotonic theorem; any accepted instance still requires the declared proposal acceptance rule and remains labeled as a proposal/iteration.
- `valid_mm` is disabled in the initial H5 profile. Configuration resolution rejects it unless a revision-bound proof artifact identifies the surrogate, touching/equality property, global minorization domain for ELBO (or majorization for VFE), maximization rule, and current derivation evidence. H5 contains the label and rejection test but no positive MM fixture.
- A rejected valid proposal is a successful H5 outcome only when live model-state, recognition-state, optimizer-state (if present), and RNG hashes are byte-for-byte unchanged before and after rejection. Rejection with mutation is `FAIL`.
- Every `UpdateAttempt` records: label; affected variables and parameter blocks; the complete expected factor IDs; the expected affected-factor subset; observed reevaluated factor IDs; before/after factor-input hashes; `observed_affected_factor_ids` derived only from unequal input hashes; diagnostic-only `value_changed_factor_ids`; complete factor-universe IDs; full `ElboTerms` before and after; objective schema hash; frozen-complement hash; candidate snapshot hash; live state/recognition/optimizer/RNG hashes before proposal, after proposal, and after acceptance/rollback; missing, extra, cache-hit, and reused factor IDs; delta; every operand/term-shaped allowance; `epsilon_delta`; decision; convergence/line-search metadata; autograd scope; and any damping/projection.
- H5's dependency graph is explicit and fail-closed. It maps every recognition variable and model parameter block to the complete set of objective factor IDs whose inputs they can affect. Define `observed_affected_factor_ids` as the factor-universe-ordered IDs whose canonical input hash differs before versus after, and require exact ordered equality with `expected_affected_factor_ids`. Scalar equality never hides an input change, and scalar roundoff/change never invents a dependency; `value_changed_factor_ids` is diagnostic only. An attempt is ineligible if this equality fails, an expected reevaluation is missing, an unexpected factor appears without schema declaration, a changed-input factor is served from stale cache, a reused factor lacks matching input/frozen-complement hashes, or before/after objective schemas differ.
- H5 minimum positive cases are: one exact conjugate Gaussian E-block; one exact normalized categorical source row where the bounded fixture exposes it; one exact Gaussian M-block with an immutable nonaliasing recognition snapshot; one accepted resolved-positive generalized-EM proposal; and one deliberately rejected objective-decreasing proposal with proven rollback.
- H5 mandatory controls are: omit a child transition factor; omit an emission factor; accept an unresolved generalized-EM delta; mislabel a natural-gradient proposal as exact; mutate live state or RNG during rejection; change a factor input while preserving its scalar value to catch a value-based false negative; and perturb a scalar value while preserving its factor-input hash to catch a value-based false positive. Each control must be detected or classified by the intended independent dependency/acceptance/hash invariant, not by a generic exception alone.
- H5 starts deterministic on CPU float64. Stochastic estimators and lower precision are outside this milestone. There is no stochastic error-budget branch in the initial gate.
- H5 numerical budgets are term- and operand-shaped. Evaluate every before/after `ElboTerms` field at frozen deterministic quadrature orders `21` and `17`; its `convergence_estimate` is `abs(term_order_21-term_order_17)`, including an explicit zero for analytic terms, and its rounding allowance uses only that field's actual summands, dimensions, condition numbers, and operation count. Each term's total allowance is `convergence_estimate + rounding_allowance`. The complete before/after allowance sums every signed term's total allowance and one final reduction-rounding term. Freeze `epsilon_delta = before_total_allowance + after_total_allowance + subtraction_rounding`; the stochastic contribution is exactly zero in H5 v1. An emission-touching update whose decision does not clear this total is `INCONCLUSIVE`; no run-wide maximum or unrelated condition number may be borrowed.
- H5 `PASS` requires every mandatory positive case and every control to pass. A finite, complete, decisive label/acceptance/dependency/rollback violation is `FAIL`. Missing proof, missing factor evidence, nonfinite output, stale schema/cache ambiguity, or an allowance that cannot resolve the required comparison is `INCONCLUSIVE` with a named obligation.
- Each task runs only its named focused RED/GREEN command. Do not run cumulative suites between tasks. The finished H4/H5 candidate uses exactly one full pytest run with JUnit because both gates, both payloads, and both configs are committed at the same exact revision and evaluated by one click artifact. If any source, test, configuration, fixture, preregistration, or artifact-schema change follows review, invalidate both gates' coupled evidence and run one replacement full suite at the new joint candidate revision.
- Fresh reviewers inspect existing focused output, the one exact-revision JUnit XML, raw H4 timings, H5 attempts, manifest, payloads, and claim ledger. They do not rerun implementer tests, the full suite, or benchmark repetitions.
- Preserve `.verification/ledger.json`, every `.verification/h3-<FULL_HEAD>-ledger.json`, and every prior H4/H5 ledger byte-for-byte. The coupled milestone uses only `.verification/h4-h5-<FULL_HEAD>-ledger.json`. An existing `.verification/active.json` blocks activation; never delete, overwrite, or repoint it manually.
- A milestone candidate requires every plan, preregistration, source, configuration, launcher, and test file named by Tasks 1--9 to be tracked; tracked content must be clean; and `git ls-files --others --exclude-standard` must return no path outside `.verification/`. Record the repository `dirty_content_digest` before JUnit, recheck it after JUnit/click-run/review, bind it in artifact provenance and the coupled ledger, and recheck it after ledger validation. Run artifacts are excluded only through the configured run-root rule; no other nonignored untracked evidence is silently tolerated.
- If a source-changing defect is found after coupled-ledger activation, do not edit while that ledger is active. Close affected/current claims as `INCONCLUSIVE` with the exact repair obligation, validate and report the current ledger so verification tooling can retire its marker, preserve that ledger, then repair and commit at a new revision. The new revision receives one replacement joint JUnit/click evidence run and a new `.verification/h4-h5-<NEW_FULL_HEAD>-ledger.json`; never overwrite or delete the first attempt.
- Manuscript references use the live repository's exact case, `Manuscripts/...`; any differently cased variant is invalid in this plan, preregistrations, and source-location assertions.
- Preserve one editable root `CONFIG`, one `verify_vfe4.py`, one `main`, and one `if __name__ == "__main__"` block. No required CLI, environment variables, notebooks, dashboards, second launchers, or second editable configuration dictionaries.

## Normative Sources and Read-Only Context

- Whitepaper H4/H5: `Manuscripts/vfe4_whitepaper/08_hypotheses_limitations.tex:51` through `:57`.
- Complete update semantics and factor blankets: `Manuscripts/vfe4_whitepaper/06_elbo_coordinate_updates.tex:355` through `:462`.
- Information identities and verification appendix: `Manuscripts/vfe4_whitepaper/09_appendices.tex:1` through `:207` and factor assembly beginning at `:280`.
- Approved code design: `docs/superpowers/specs/2026-07-21-vfe4-codebase-design.md`, especially sections 6.1, 7, 8.7, 9.2--9.5, 11.1, 13, and 15--17.
- H2 seam/plan: `docs/superpowers/plans/2026-07-21-vfe4-h2-information-moment.md` and live `vfe4/types/information.py`, `vfe4/numerics/precision.py`, `vfe4/numerics/information.py`, `vfe4/objective/h2_information.py`.
- Approved H3 plan: `docs/superpowers/plans/2026-07-21-vfe4-h3-structured-adequacy.md`, especially its neutral scalar-factor model, exact NumPy posterior oracle, differentiable recognition path, and revision-specific evidence policy.
- Research vault, read only: `[[Variational EM]]`, `[[Natural gradient]]`, `[[neal-1998-variational-em]]`, and `[[dempster-1977-em-algorithm]]`. These support the same-objective and finite-step-label distinctions; they do not supply empirical H4 evidence or executable H5 closure.

## File Map and Dependency Boundaries

| File | Responsibility |
|---|---|
| `vfe4/types/h4.py` | Immutable neutral-problem identity, solver-arm, terminal-law, timing, operation-count, and H4 gate-result records. |
| `vfe4/generative/reference_h4.py` | Deterministic coupled/control Gaussian problem generator and H3-fixture adapter; no solver or timing logic. |
| `vfe4/inference/h4_instrumentation.py` | Real-operation facade, null/counting recorders, operation shapes, and untimed memory diagnostics. |
| `vfe4/inference/h4_solvers.py` | Independent information and direct-moment solver implementations behind one protocol. |
| `verification/numpy_oracles/h4_gaussian.py` | NumPy-only exact posterior/objective from immutable neutral factors; no production solver imports. |
| `verification/h4_budget.py` | Operand-shaped terminal equivalence allowances only. |
| `verification/h4_statistics.py` | Primary per-seed/aggregate timed-order balance, seed-level medians, geometric mean, fixed paired bootstrap, and three-way threshold decision. |
| `verification/h4_gate.py` | Preflight, correctness anchor, scaled equivalence, independently indexed timed AB/BA harness, balance gate, diagnostics, status mapping, and H4 payload. |
| `vfe4/types/updates.py` | Closed H5 taxonomy, factor IDs, immutable snapshots, hashes, `UpdateAttempt`, and H5 gate-result records. |
| `vfe4/objective/dependency_graph.py` | Static variable/parameter-to-factor graph and expected affected-factor calculation. |
| `vfe4/objective/h5_complete.py` | One authoritative full `ElboTerms` evaluation with factor-level trace and cache provenance. |
| `vfe4/inference/h5_updates.py` | Differentiable proposal construction, exact/source/M/GEM operations, freeze-before-evaluate, acceptance, and rollback controller. |
| `verification/numpy_oracles/h5_updates.py` | Independent exact E/source/M reference updates and complete-objective deltas. |
| `verification/h5_budget.py` | Per-term deterministic-convergence plus rounding records, complete before/after totals, subtraction rounding, and exact `epsilon_delta`. |
| `verification/h5_gate.py` | Positive cases, adversarial controls, dependency/acceptance/hash decisions, and H5 payload. |
| `vfe4/config/schema.py` | Frozen `H4ValidationConfig` and `H5ValidationConfig` sections plus closed literals. |
| `vfe4/config/resolve.py` | Exact protocol and derived timed-balance validation, coupled prefix validation, and canonical config hashing. |
| `verification/run_gates.py` | Conditional one-time fixture capture, ordered H1--H5 evaluation, and one atomic artifact family. |
| `vfe4/artifacts/provenance.py` | Timing clock, BLAS/thread/affinity, gate/config/fixture, and update-schema provenance. |
| `verify_vfe4.py` | The one editable click-run configuration and orchestration entry point. |
| `docs/preregistrations/2026-07-21-h4-information-cost.md` | Frozen H4 factors, seeds, timing, statistics, equivalence, statuses, payload, and nonclaims. |
| `docs/preregistrations/2026-07-21-h5-update-coherence.md` | Frozen taxonomy, dependency graph, cases/controls, budgets, statuses, payload, and nonclaims. |

Dependency direction remains `config + types -> generative/numerics -> recognition/objective -> inference -> verification/runner/artifacts`. Production `vfe4` never imports `verification` or `tests`. NumPy oracles import only Python, NumPy, and raw immutable records/bytes represented without PyTorch. `verification/h4_gate.py` and `verification/h5_gate.py` are the only layers that compare production and oracle outputs.

## Public Interface Map

```python
# vfe4/types/h4.py
H4SolverArm = Literal["information", "moment"]
H4ProblemKind = Literal["coupled", "zero_control"]
H4PairOrder = Literal["information_then_moment", "moment_then_information"]

@dataclass(frozen=True)
class H4NeutralProblem:
    problem_id: str
    seed: int
    kind: H4ProblemKind
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    transitions: tuple[H4AffineGaussianFactor, ...]
    observations: tuple[H4AffineGaussianFactor, ...]
    canonical_sha256: str

@dataclass(frozen=True)
class H4SelectedMoment:
    name: str
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

@dataclass(frozen=True)
class H4TerminalLaw:
    arm: H4SolverArm
    h: tuple[float, ...]
    J: tuple[tuple[float, ...], ...]
    mean: tuple[float, ...]
    selected_moments: tuple[H4SelectedMoment, ...]
    complete_objective: float
    stopping_residual: float

@dataclass(frozen=True)
class H4NativeInformationState:
    h: tuple[float, ...]
    J: tuple[tuple[float, ...], ...]
    mean: tuple[float, ...]
    complete_objective: float

@dataclass(frozen=True)
class H4NativeMomentState:
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    complete_objective: float

@dataclass(frozen=True)
class H4TimingRecord:
    problem_id: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    seed: int
    kind: H4ProblemKind
    horizon: int
    repetition_index: int
    pair_index: int
    order: H4PairOrder
    information_nanoseconds: int
    moment_nanoseconds: int

# vfe4/inference/h4_solvers.py
class H4GaussianSolver(Protocol):
    def solve(
        self,
        problem: H4NeutralProblem,
        protocol: H4SolveProtocol,
        linalg: InstrumentedLinearAlgebra,
    ) -> H4SolverResult: ...

def solve_information_form(...) -> H4SolverResult: ...
def solve_moment_form(...) -> H4SolverResult: ...

# verification/h4_statistics.py
def summarize_seed_ratios(records: tuple[H4TimingRecord, ...]) -> H4TimingSummary: ...
def summarize_primary_timed_order(
    records: tuple[H4TimingRecord, ...],
    *,
    expected: tuple[tuple[int, int, int], ...],
) -> H4PrimaryTimedOrderBalance: ...
def paired_log_bootstrap_interval(
    seed_ratios: tuple[float, ...], *, replicates: int, seed: int
) -> H4BootstrapInterval: ...
def decide_h4_interval(interval: H4BootstrapInterval, threshold: float) -> H4IntervalDecision: ...
```

```python
# vfe4/types/updates.py
class UpdateLabel(str, Enum):
    EXACT_COORDINATE = "exact_coordinate"
    VALID_MM = "valid_mm"
    GENERALIZED_EM = "generalized_em"
    NATURAL_GRADIENT_PROPOSAL = "natural_gradient_proposal"
    SGD_PROPOSAL = "sgd_proposal"
    ADAM_PROPOSAL = "adam_proposal"
    TRUNCATED_ITERATION = "truncated_iteration"

@dataclass(frozen=True)
class RecognitionSnapshot:
    schema_version: str
    tensors: Mapping[str, FrozenTensorValue]
    state_sha256: str

@dataclass(frozen=True)
class CompleteElboEvaluation:
    terms: ElboTerms
    factor_values: Mapping[str, float]
    factor_input_hashes: Mapping[str, str]
    factor_ids: tuple[str, ...]
    cache_hits: tuple[str, ...]
    reused_factor_ids: tuple[str, ...]
    objective_schema_sha256: str
    frozen_complement_sha256: str

@dataclass(frozen=True)
class UpdateAttempt:
    label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    expected_factor_ids: tuple[str, ...]
    expected_affected_factor_ids: tuple[str, ...]
    observed_factor_ids: tuple[str, ...]
    observed_affected_factor_ids: tuple[str, ...]
    value_changed_factor_ids: tuple[str, ...]
    complete_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]
    cache_hits: tuple[str, ...]
    reused_factor_ids: tuple[str, ...]
    before: CompleteElboEvaluation
    after: CompleteElboEvaluation
    delta_elbo: float
    epsilon_delta: float
    accepted: bool
    decision_reason: str
    hashes: UpdateHashRecord
    autograd_scope: str

# vfe4/objective/dependency_graph.py
def expected_affected_factors(
    graph: FactorDependencyGraph,
    *,
    variables: tuple[str, ...],
    parameters: tuple[str, ...],
) -> tuple[str, ...]: ...

# vfe4/inference/h5_updates.py
def freeze_recognition_candidate(working: DifferentiableRecognitionState) -> RecognitionSnapshot: ...
def execute_update(
    live: H5LiveState,
    specification: UpdateSpecification,
    evaluator: CompleteElboEvaluator,
    budget: H5BudgetConfig,
) -> UpdateAttempt: ...
```

The exact field order above is part of canonical JSON and hashing. Add `H4GateResult` and `H5GateResult` as separate fail-closed records rather than widening H1/H2's singular-residual assumptions or H3's allowance mapping into an untyped union. The unified runner accepts the explicit result union.

## Design Rationale and Nonclaims

- A direct moment solver is the scientific control for an information solver. Calling the information path and inverting only at the end would compare presentation formats, not independent algorithms.
- The H3 anchor verifies that both solvers preserve the already frozen small Gaussian semantics. The scaled H4 suite tests runtime under a structured family with the exact requested dimensions; it does not turn the H3 adequacy result into a performance result.
- Reusing identical immutable factor objects eliminates model drift between arms. Constraining thread count, order, warmups, repetitions, and raw-data retention makes the timing claim auditable without pretending Windows scheduling noise has vanished.
- Seed-level aggregation avoids pseudoreplication. The 11 repetitions estimate each seed's timing; they do not create 220 independent problem instances.
- The `0.80` rule is a proposed practical-effect threshold, not a theorem derived from information geometry. Passing establishes only this preregistered CPU/float64 implementation result.
- The exact `[0.80,0.80]` bootstrap boundary satisfies both non-strict verbal inequalities, so the implementation resolves the overlap conservatively as `INCONCLUSIVE` rather than allowing branch order to choose the scientific conclusion.
- H4 memory/count diagnostics are separated from the primary timer because Python/native allocation tracking and counting wrappers add different overheads. They remain useful implementation evidence but not primary inferential endpoints.
- H5 verifies labels against the complete implemented ELBO. It does not prove the whitepaper's exact-arithmetic coordinate theorem, and numerical monotonicity tests cannot replace the derivation in the manuscript.
- A natural coordinate, natural-gradient direction, reverse-mode gradient, optimizer class, or accepted line search does not imply exact coordinate ascent. The stored label describes the executed operation.
- Exact/MM allow a resolved rounding-scale nonincrease (`delta >= -epsilon_delta`). GEM is stricter: it must resolve a positive increase (`delta > epsilon_delta`). This prevents a numerically unresolved tie from being advertised as generalized-EM progress.
- H5's rejected-proposal case is positive verification of transactional behavior, not a claim that the proposal was a good update.
- The initial H5 profile proves no MM implementation because no current revision-bound minorization artifact exists. Rejecting `valid_mm` configuration is the correct closed-world behavior.

---

### Task 1: Freeze H4 Protocol Types, Generator Contract, and Preregistration

**Files:**

- Track: `docs/superpowers/plans/2026-07-21-vfe4-h4-h5-cost-update.md`
- Create: `vfe4/types/h4.py`
- Modify: `vfe4/types/__init__.py`
- Create: `vfe4/generative/reference_h4.py`
- Modify: `vfe4/generative/__init__.py`
- Create: `tests/unit/test_h4_problem.py`
- Create: `docs/preregistrations/2026-07-21-h4-information-cost.md`

**Interfaces:**

- Produce immutable `H4AffineGaussianFactor`, `H4NeutralProblem`, `H4SolveProtocol`, `H4NativeInformationState`, `H4NativeMomentState`, `H4SelectedMoment`, `H4TerminalLaw`, `H4SolverResult`, `H4TimingRecord`, `H4OperationRecord`, `H4MemoryRecord`, `H4GateResult`, and exact `Literal` aliases from the public interface map. `H4TerminalLaw.selected_moments` is an exact-name tuple in canonical order, never a mutable dictionary. Every `H4SelectedMoment` contains an immutable mean and covariance block.
- Produce `make_h4_problem(*, seed: int, kind: H4ProblemKind, horizon: Literal[7,15,31], d_z: Literal[4], d_m: Literal[4]) -> H4NeutralProblem` and `h4_anchor_from_h3(fixture: H3Fixture) -> H4NeutralProblem`.
- The generator returns one fully materialized immutable problem. Neither solver receives a seed or generator callback.

- [ ] **Step 1: Write the H4 preregistration before any timing exists.** Copy every H4 global constraint, the exact 20 seed values, zero-based horizon/seed/kind indices, exact traversal, the dimension table, factor-generation formulas, three warmup-pair and 11 timed-pair AB/BA formulas, primary per-seed and aggregate timed-order balance, timer/batched-postflight boundaries, primary endpoint, bootstrap algorithm, `0.80` decision table, scaled conditioning envelope, `1e-9` solver budget, strict `1e-4` allowance/scale cap, equivalence fields, status precedence, operation/memory secondary status, JSON schema, and H5/H6/H7/H8/training nonclaims. State explicitly that warmups do not enter inferential balance and that no coefficient, seed, order, repetition count, envelope, budget, cap, bootstrap setting, or threshold was chosen from H4 measurements.

  Freeze the 20 problem seeds as:

  ```python
  H4_PROBLEM_SEEDS = (
      104729, 130363, 155921, 181081, 206369,
      231779, 257053, 282407, 307831, 333271,
      358747, 384253, 409891, 435437, 461009,
      486587, 512161, 537793, 563359, 588937,
  )

  # Each row is (seed, primary_timed_ab_count, primary_timed_ba_count).
  H4_PRIMARY_TIMED_BALANCE = (
      (104729, 5, 6), (130363, 6, 5),
      (155921, 5, 6), (181081, 6, 5),
      (206369, 5, 6), (231779, 6, 5),
      (257053, 5, 6), (282407, 6, 5),
      (307831, 5, 6), (333271, 6, 5),
      (358747, 5, 6), (384253, 6, 5),
      (409891, 5, 6), (435437, 6, 5),
      (461009, 5, 6), (486587, 6, 5),
      (512161, 5, 6), (537793, 6, 5),
      (563359, 5, 6), (588937, 6, 5),
  )
  H4_PRIMARY_TIMED_AB_TOTAL = 110
  H4_PRIMARY_TIMED_BA_TOTAL = 110
  ```

  Define the scaled problem constructively. Storage remains population-major `[z_0,m_0,z_1,m_1,...,z_T,m_T]`, with `D=(T+1)*(d_z+d_m)`, `d_z=d_m=4`; the initial joint `[z_0,m_0]` is fixed `N(0,I_8)` and consumes no RNG draw. The normalized factor schedule is exactly `initial_joint`, then for every ascending `t=1..T`, the normalized `m_t|m_{t-1}` transition, the normalized `z_t|z_{t-1},m_t` transition, and the normalized local observation factor. Thus `m_t` is generated and consumed before `z_t|m_t`, without changing storage order.

  The only generator is `numpy.random.Generator(numpy.random.PCG64(seed))`; neither solver receives it. For each ascending `t`, draw exactly and only in this order: `A_m`, `A_z`, and `B`, each `standard_normal((4,4))`; `c_m` then `c_z`, each `uniform(-0.25,0.25,size=4)`; `R_m` then `R_z`, each `uniform(0.5,1.5,size=4)`; raw `G=standard_normal((8,8))`; observation offset `uniform(-0.25,0.25,size=8)`; observation noise `uniform(0.75,1.25,size=8)`; and observed target `uniform(-1,1,size=8)`. Define `spectral_clip(M)=M*min(1,0.65/||M||_2)`, apply it to `A_m` and to the horizontally concatenated `[A_z B]`, and split the latter back into `A_z` and `B`. Set `H=I_8+0.05*G/max(1,||G||_2)`. The factors are therefore `m_t ~ N(A_m m_{t-1}+c_m,diag(R_m))`, `z_t ~ N(A_z z_{t-1}+B m_t+c_z,diag(R_z))`, and `y_t ~ N(H[z_t,m_t]+offset,diag(observation_noise))` at the drawn target.

  The zero control is derived from the same draws and records: it zeros all active `A_m`, `A_z`, and `B` blocks for every `t`, while retaining every other generated value, raw-draw record, offset, diagonal noise, `H`, target, factor ID/order, seed, shape, and canonical serialization field unchanged. The sole objective is the exact normalized Gaussian log evidence/log normalizer of this complete schedule, including every factor constant; `J,h` and selected moments are comparison records, not another objective. Serialize every generated float and raw-draw provenance into the immutable problem, then hash canonical JSON.

  The H3 anchor adapter is a generic normalized-affine schedule: it reads only public H3 normalized-factor records in fixture order and preserves their rows, targets, variances, normalizers, IDs, and H3 coordinate order. It must not infer initial/transition/observation roles or synthesize a state-space factorization. Both raw H3 fixtures must reproduce their H3 canonical `(J,h,objective)` under H3's own allowances.

  The exact selected-moment labels are `("initial", "terminal", "observation[1]", ..., "observation[T]")`. `initial` and `terminal` are the full joint `[z_t,m_t]` blocks at `t=0` and `t=T`; every `observation[t]` is the full local `[z_t,m_t]` block in ascending time. Keep all labels even when `T=1` makes blocks overlap; do not deduplicate, map, or alias them.

  H4 thread control is process-scoped and mandatory. After H1--H3 work and before H4 preflight/timing, capture `torch.get_num_threads()`, call `torch.set_num_threads(1)`, and verify the observed intra-op count is one. In a `finally` block attempt to restore the captured count and record prior, effective, restored, and restoration-error fields. A set/verify failure suppresses timed records and makes H4 `INCONCLUSIVE`; a restoration failure is an environment/protocol obligation that prevents H4 `PASS`. Do not change inter-op threads.

  Resolve H4 status in this fixed precedence: protocol/environment/thread/fixture/condition/table-completeness/nonfinite ambiguity is `INCONCLUSIVE`; otherwise a finite decisive H3-anchor or terminal-law miss is `FAIL`; otherwise apply the primary interval rule (`PASS` only when upper bound `<=0.80`, `FAIL` only when lower bound `>=0.80`, and `[0.80,0.80]` or a crossing interval `INCONCLUSIVE`). Operation and memory diagnostics are secondary and never rescue or overturn that status.

- [ ] **Step 2: Write strict type/generator tests.** Assert exact field sets, tuple immutability, finite float64-representable values, exact coordinate order, `D` table `(64,128,256)`, fixed no-RNG initial `N(0,I_8)`, the exact PCG64 draw order/distributions, causal `m_t`-then-`z_t|m_t` parent indices, SPD noises, factor-ID uniqueness, the separate `A_m` and joint `[A_z B]` spectral-clip envelope, exact `H` construction, and a zero control that zeros all and only active `A_m`, `A_z`, and `B` blocks while sharing every other draw/record. Require deterministic canonical bytes/hash for repeat construction, distinct hashes across kind/seed/size, and no H4 field on H1/H2/H3 records. Require each `H4TimingRecord` to carry independent `problem_index`, `horizon_index`, `seed_index`, `kind_index`, timed `repetition_index`, absolute `pair_index`, exact order label, and both positive native-arm durations; reject an order inconsistent with the independent-index parity formula. Require the exact immutable selected-moment labels `("initial","terminal","observation[1]",...,"observation[T]")`, with immutable mean/covariance rows; reject reordered, duplicate, missing, mapping, mutable, or aliased values. Adapt both raw H3 fixtures through the generic normalized-affine adapter and require their canonical `(J,h,objective)` to agree with the existing H3 generator/oracle under H3's own allowances.

- [ ] **Step 3: Run the Task 1 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h4_problem.py -q
  ```

  Expected: collection fails because `vfe4.types.h4` and `vfe4.generative.reference_h4` do not exist.

- [ ] **Step 4: Implement the immutable records and deterministic generator.** Validate all constructor invariants before hashing. Canonical JSON uses UTF-8, sorted keys, compact separators, finite JSON numbers, row-major arrays, and exact seed/kind/shape fields. The H3 adapter reads only public H3 normalized factors, neither changes H3 bytes nor imports an H3 oracle, and never assigns a factor a role by name.

- [ ] **Step 5: Run the Task 1 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h4_problem.py -q
  ```

  Expected: all type, deterministic-generation, control-matching, dimension, hash, and H3-anchor tests pass without timing.

- [ ] **Step 6: Commit Task 1.**

  ```powershell
  git add docs/superpowers/plans/2026-07-21-vfe4-h4-h5-cost-update.md vfe4/types/h4.py vfe4/types/__init__.py vfe4/generative/reference_h4.py vfe4/generative/__init__.py tests/unit/test_h4_problem.py docs/preregistrations/2026-07-21-h4-information-cost.md
  git commit -m "test: freeze H4 information cost protocol"
  ```

---

### Task 2: Implement Independent H4 Solver Arms and Symmetric Real-Operation Instrumentation

**Files:**

- Create: `vfe4/inference/h4_instrumentation.py`
- Create: `vfe4/inference/h4_solvers.py`
- Modify: `vfe4/inference/__init__.py`
- Create: `tests/unit/test_h4_solvers.py`
- Create: `tests/unit/test_h4_instrumentation.py`

**Interfaces:**

- Produce `OperationKind`, `OperationRecorder`, `NullOperationRecorder`, `CountingOperationRecorder`, `InstrumentedLinearAlgebra`, and `measure_untimed_memory(callable) -> H4MemoryRecord`.
- Produce `H4GaussianSolver`, `InformationFormH4Solver`, `MomentFormH4Solver`, `solve_information_form`, and `solve_moment_form`.
- `H4GaussianSolver` lives in `vfe4/inference/h4_solvers.py` beside the facade functions; `vfe4/types/h4.py` contains only dependency-light immutable data. Both solvers return the identical `H4SolverResult` envelope with an exactly one-of native information/moment state. The untimed common converter produces `H4TerminalLaw` and the canonical selected-moment tuple named `initial`, `terminal`, and every local observation block.

- [ ] **Step 1: Write solver independence and instrumentation tests.** On hand-checkable `D=4` and adapted H3 fixtures, assert both arms match the exact law. Monkeypatch the information solver, canonical factor assembler, `InformationGaussian`, `torch.linalg.inv`, `torch.linalg.pinv`, and `torch.cholesky_inverse` to raise while the moment arm still succeeds. Monkeypatch moment propagation/conditioning entry points to raise while the information arm still succeeds. Assert each arm receives the same problem object identity and never mutates it.

  Test the facade with counting and null recorders. For every wrapped Cholesky, triangular solve, matrix multiply, rank update, and selected-block extraction, require one real operation and one correctly shaped count under the counting recorder, the same numerical output under the null recorder, and no public `record_only` method. A fake solver that tries to report an operation without executing the facade must have no way to increment counts.

- [ ] **Step 2: Run the Task 2 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h4_solvers.py tests/unit/test_h4_instrumentation.py -q
  ```

  Expected: collection fails because the H4 solver and instrumentation modules do not exist.

- [ ] **Step 3: Implement the information arm.** Starting from the neutral initial law, assemble `J` and `h` directly by adding every normalized affine-Gaussian factor in frozen schedule order. Use the facade for Cholesky, solves, quadratics, and log determinants. Never form a complete covariance. Materialize only the native terminal `h`, `J`, mean, and complete normalized Gaussian objective within the timed solver call; selected inverse blocks belong to the common untimed equivalence converter.

- [ ] **Step 4: Implement the moment arm independently.** Construct the joint mean/covariance directly by affine Gaussian propagation from the initial moment law, then apply each observation factor with the Gaussian conditioning identities

  ```python
  innovation_covariance = noise + A @ covariance @ A.T
  gain = covariance @ A.T @ solve(innovation_covariance, I)
  mean = mean + gain @ (target - A @ mean)
  covariance = covariance - gain @ A @ covariance
  ```

  Symmetrize only the roundoff-level result as `0.5*(Sigma+Sigma.T)` after the symmetric rank update; reject a non-SPD result rather than jittering. Return only the native terminal mean, covariance, and common objective within the timed solver call. The untimed common converter derives `J` and `h` with this arm's own Cholesky solve; it does not call the information arm or share canonical tensors.

- [ ] **Step 5: Implement the common one-pass stopping rule, untimed conversion, and memory diagnostics.** Both arms stop only after the same frozen factor schedule is exhausted exactly once, every native finite/SPD check passes, and the common objective is evaluated; neither has a looser convergence loop. After the complete timed batch for a problem, `to_common_terminal_law` computes `h`, `J`, mean, the canonical immutable selected-moment tuple, objective, and `||J*mu-h||_inf/scale` for both arms under the same comparison instrumentation. `measure_untimed_memory` records Python peak bytes and process working-set delta when available, labels unsupported fields as unavailable, and never substitutes them into timing.

- [ ] **Step 6: Run the Task 2 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h4_solvers.py tests/unit/test_h4_instrumentation.py -q
  ```

  Expected: both independent arms close the H3 anchor and hand examples; independence monkeypatches hold; real-operation counts are symmetric in mechanism; no forbidden inverse path is used.

- [ ] **Step 7: Commit Task 2.**

  ```powershell
  git add vfe4/inference/h4_instrumentation.py vfe4/inference/h4_solvers.py vfe4/inference/__init__.py tests/unit/test_h4_solvers.py tests/unit/test_h4_instrumentation.py
  git commit -m "feat: add independent H4 Gaussian solver arms"
  ```

---

### Task 3: Add the Independent H4 Oracle, Operand-Shaped Equivalence Budgets, and Paired Statistics

**Files:**

- Create: `verification/numpy_oracles/h4_gaussian.py`
- Modify: `verification/numpy_oracles/__init__.py`
- Create: `verification/h4_budget.py`
- Create: `verification/h4_statistics.py`
- Create: `tests/oracle/test_h4_numpy_oracle.py`
- Create: `tests/unit/test_h4_statistics.py`

**Interfaces:**

- Produce `H4OracleEvaluation` and `evaluate_h4_oracle(problem_payload: bytes) -> H4OracleEvaluation` from canonical neutral-problem bytes.
- Produce `scalar_allowance`, `vector_allowance`, `matrix_allowance`, `selected_moment_allowance`, `objective_allowance`, `pair_allowance`, `condition_envelope_eligible`, and `allowance_is_decisive`; each record retains dimensions, operands, absolute summands, condition numbers, operation counts, solver contribution, invariant scale, final allowance, and allowance/scale ratio.
- Produce `H4TimingSummary`, `H4PrimaryTimedOrderBalance`, `H4BootstrapInterval`, `H4IntervalDecision`, `summarize_seed_ratios`, `summarize_primary_timed_order`, `paired_log_bootstrap_interval`, and `decide_h4_interval`.

- [ ] **Step 1: Write oracle, budget, and statistics tests.** Require the NumPy module to import neither `torch` nor `vfe4`; parse only canonical problem bytes; independently assemble the exact posterior; and match the H3 anchor/frozen references. Reject noncanonical schemas, wrong hashes, noncausal factors, non-SPD noise, and nonfinite values. Exercise every scaled conditioning-envelope boundary exactly: equality at each inclusive lower/upper limit is eligible, one representable float outside is ineligible, and any single ineligible problem/oracle/arm produces an H4 `INCONCLUSIVE` obligation rather than repair.

  Use hand-authored timing tables to prove: traversal is exactly horizon/seed/kind with independent zero-based indices; each problem has three warmup pairs followed by 11 timed pairs; AB occurs exactly when `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`; warmups are excluded from timed/inferential balance; no conversion/diagnostic event occurs inside the timed-batch event span; each seed/arm timing group has exactly 11 positive raw nanosecond values; and `summarize_primary_timed_order` reports the primary `D=256` coupled table equal to `H4_PRIMARY_TIMED_BALANCE` row-for-row, with ten `6 AB/5 BA` seeds, ten `5 AB/6 BA` seeds, and totals `AB=110`, `BA=110`. Reject a swapped seed row, a per-seed `7/4` distribution even if aggregate totals remain `110/110`, an aggregate `111/109` distribution, or any attempt to use warmups to repair timed imbalance. Prove per-seed medians are used, the aggregate is the geometric mean, resampling is over exactly 20 seed ratios, fixed seed/replicate count is byte-deterministic, and a table with 220 repetitions treated as independent is rejected. Cover interval entirely below, entirely above, crossing, touching from each side, and degenerate at `0.80`. Test `allowance/invariant_scale` immediately below `1e-4`, exactly `1e-4`, and immediately above; only the first is decisive. Test the solver contribution as exactly `1e-9*scale` once per solver operand and reject duplicate, omitted, oracle-side, or pooled solver contributions.

- [ ] **Step 2: Run the Task 3 tests for RED.**

  ```powershell
  python -m pytest tests/oracle/test_h4_numpy_oracle.py tests/unit/test_h4_statistics.py -q
  ```

  Expected: collection fails because the H4 oracle, budget, and statistics modules do not exist.

- [ ] **Step 3: Implement the independent oracle.** Reconstruct the joint mean/covariance of the generative chain from raw numeric records, condition on observations, then compute exact `J`, `h`, selected moments, normalized objective, eigenvalue/condition diagnostics, and absolute-summand metadata. Do not call either production solver or reuse production factor-assembly helpers.

- [ ] **Step 4: Implement operand-shaped budgets and eligibility literally.** Freeze `eps=np.finfo(np.float64).eps`, `gamma(n)=n*eps/(1-n*eps)`, `C=4096`, `H4_SOLVER_RELATIVE_BUDGET=1e-9`, and `H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION=1e-4`. Every primitive allowance uses only the current operand's `dimension`, actual absolute summands, observed condition numbers, declared operation count, and a one-bit solver-produced flag. A solver-produced operand adds exactly `1e-9*invariant_scale`; an oracle operand adds zero. A pair allowance is exactly the left allowance plus right allowance plus one comparison-reduction term. `allowance_is_decisive` uses strict `<1e-4`; `condition_envelope_eligible` uses the inclusive frozen bounds. Reject empty condition collections, nonfinite/negative inputs, dimension mismatch, duplicate solver contributions, or a request to supply a global maximum condition number.

- [ ] **Step 5: Implement statistics literally.** Use `statistics.median` on each 11-value integer sequence, `math.log`/`math.exp` for seed ratios, `numpy.random.Generator(numpy.random.PCG64(20260721))` for exactly 100,000 paired resamples, and NumPy's explicitly preregistered linear percentile interpolation. Retain the bootstrap seed, replicate count, seed indices, and resample-index SHA-256 in the result.

- [ ] **Step 6: Run the Task 3 tests for GREEN.**

  ```powershell
  python -m pytest tests/oracle/test_h4_numpy_oracle.py tests/unit/test_h4_statistics.py -q
  ```

  Expected: independent exact laws match the anchors; malformed operands fail closed; paired summaries and all `0.80` boundary decisions match the preregistration exactly.

- [ ] **Step 7: Commit Task 3.**

  ```powershell
  git add verification/numpy_oracles/h4_gaussian.py verification/numpy_oracles/__init__.py verification/h4_budget.py verification/h4_statistics.py tests/oracle/test_h4_numpy_oracle.py tests/unit/test_h4_statistics.py
  git commit -m "test: add H4 oracle budgets and paired statistics"
  ```

---

### Task 4: Build the H4 Preflight, Timed Harness, Gate Decision, and Artifact Payload

**Files:**

- Create: `verification/h4_gate.py`
- Create: `tests/promotion/test_h4_gate.py`
- Modify: `docs/preregistrations/2026-07-21-h4-information-cost.md`

**Interfaces:**

- Produce `H4GateEvaluation(result, anchors, problems, oracle_by_problem, solver_results, equivalence, raw_timings, primary_timed_order_balance, timing_summary, operation_counts, memory_records, environment, validation_payload)`.
- Produce `evaluate_h4(config: ResolvedConfig, *, h3_coupled_bytes: bytes, h3_zero_bytes: bytes) -> H4GateEvaluation` and `h4_validation_payload(evaluation) -> dict[str, object]`.

- [ ] **Step 1: Write the promotion test with exact invariant names.** Require this ordered tuple:

  ```text
  h3_anchor_identity
  fixed_seed_problem_identity
  coupled_zero_control_contract
  cpu_float64_one_thread
  shared_protocol_identity
  scaled_condition_envelope
  complete_repetition_table
  primary_timed_order_balance
  exact_posterior_gap_equivalence
  terminal_h_equivalence
  terminal_J_equivalence
  selected_moment_equivalence
  complete_objective_equivalence
  all_equivalence_allowances_decisive
  real_operation_instrumentation
  primary_seed_level_inference
  primary_effect_threshold
  ```

  Assert H4 can be `PASS`, `FAIL`, or `INCONCLUSIVE` independently of an H5 result. A decisive terminal-equivalence miss and an interval wholly at/above the no-support region are finite `FAIL` cases. Missing environment facts, wrong traversal/pair formula, incomplete raw timings, any primary per-seed balance mismatch, aggregate primary totals other than exactly `110/110`, warmups counted toward balance, nonfinite solver output, any exact/problem/terminal condition outside the inclusive envelope, allowance/scale ratio at or above `1e-4`, or crossing interval is `INCONCLUSIVE`. Test the exact 20-row primary balance table plus all single-row/aggregate failure controls, every exact envelope boundary, and every decisiveness boundary. No secondary size/control/memory/count value can change `primary_effect_threshold`.

- [ ] **Step 2: Run the Task 4 test for RED.**

  ```powershell
  python -m pytest tests/promotion/test_h4_gate.py -q
  ```

  Expected: collection fails because `verification.h4_gate` does not exist.

- [ ] **Step 3: Implement the common preflight outside timing.** Capture and hash both H3 fixture byte sequences, adapt them once, generate/hash all `20*3*2` scaled problems once in exact horizon/seed/kind traversal, compute NumPy exact results, materialize solver inputs, set/verify one intra-op thread, and record inter-op threads/affinity/BLAS/clock. Check every exact problem against the inclusive condition envelope before warmups. Reject any problem/oracle/protocol/traversal mismatch before timing. Do not run counting/memory diagnostics here and do not call H2 diagnostic eigenvalue functions in the timed callable.

- [ ] **Step 4: Implement whole-problem warmup and timed batches.** Traverse problems only in frozen horizon/seed/kind order while retaining independent zero-based `horizon_index`, `seed_index`, and `kind_index`. For one problem, create fresh solver state inside every arm call, run warmup pairs `0,1,2` using AB exactly when `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, then preallocate the 22 native-result/timing slots, disable cyclic garbage collection once, and run timed pair indices `3..13` consecutively with the same formula. Around each native solver call, call `perf_counter_ns`; between repetitions perform only timer reads, native-result reference assignment, and preallocated scalar timing assignment. Do not construct timing records, convert terminals, hash outputs, compare equivalence, count operations, measure memory, serialize, log, print, validate shapes, or run diagnostics until all 11 timed pairs for that problem finish. Restore garbage collection once after the timed batch, then materialize `H4TimingRecord` objects from the preallocated indices/orders/durations. Warmup order is recorded in the separate event trace but excluded from every `H4TimingRecord` and timed-balance count. Reject order/table problems only in postflight.

- [ ] **Step 5: Batch terminal conversion, diagnostics, equivalence, and the H4 decision after timing.** For each completed problem, convert retained native terminals in fixed arm-independent order `(repetition_index 0..10, arm information then moment)`, regardless of timed AB/BA order; only then hash, validate, compare to the exact oracle/paired arm, and append canonical immutable selected moments. After every problem's timing is complete, run separate untimed counting and memory passes in frozen problem/arm order. A solver may not pass because its counterpart shares the same error. Before computing primary statistics, call `summarize_primary_timed_order` on raw records only; require exact row equality with `H4_PRIMARY_TIMED_BALANCE`, ten rows of each `6/5` and `5/6` pattern, exact aggregate totals `110/110`, and zero warmup contribution. Compute primary statistics only after timed-order balance, condition-envelope, and per-invariant allowance-decisiveness eligibility pass. Apply status precedence: protocol/environment/traversal/availability ambiguity, timed-balance mismatch, out-of-envelope value, or nondecisive allowance -> `INCONCLUSIVE`; decisive equivalence miss -> `FAIL`; primary interval crossing -> `INCONCLUSIVE`; lower no-support bound -> `FAIL`; upper support bound -> `PASS`.

- [ ] **Step 6: Emit the complete `validation/h4.json` schema.** Include finite JSON records for gate/status/obligations; exact config hash/profile; both H3 anchor hashes; exact horizon/seed/kind traversal and independent indices; the parity formula; separate warmup and timed AB/BA sequences; an explicit `warmups_count_toward_balance=false`; the expected and observed 20-row primary timed balance tables; expected/observed counts of ten `6/5` rows and ten `5/6` rows; expected/observed aggregate `AB=110`, `BA=110`; the `primary_timed_order_balance` invariant; timer and postflight event spans proving no inter-repetition conversion/diagnostic event; every problem identity/factor hash; environment/clock/thread/BLAS/affinity data; solver implementations and common protocol; warmup counts; every raw timing/order; seed-level medians/ratios; bootstrap indices digest/estimate/interval; threshold decision; frozen condition envelope and each exact/problem/terminal diagnostic; exact/oracle/arm terminal quantities; every invariant's scale, rounding/solver contributions, allowance, allowance/scale ratio, and strict decisiveness result; untimed real-operation counts; untimed memory diagnostics; canonical immutable selected-moment order; ordered invariants; H4 bounded claim; and H5--H8/training nonclaims.

- [ ] **Step 7: Run the Task 4 test for GREEN.**

  ```powershell
  python -m pytest tests/promotion/test_h4_gate.py -q
  ```

  Expected: fixture-sized deterministic/mocked clocks cover every status, exact condition/decisiveness boundary, traversal/order, no-between-repetition-work, batched-conversion, and schema branch; a real reduced repetition fixture proves both solver arms and equivalence without treating its timing as H4 promotion evidence. The full 20-seed/11-repetition protocol is reserved for the click-run milestone.

- [ ] **Step 8: Commit Task 4.**

  ```powershell
  git add verification/h4_gate.py tests/promotion/test_h4_gate.py docs/preregistrations/2026-07-21-h4-information-cost.md
  git commit -m "test: add the H4 information cost gate"
  ```

---
### Task 5: Freeze the H5 Taxonomy, Factor-Dependency Graph, Immutable Snapshot Boundary, and Preregistration

**Files:**

- Create: `vfe4/types/updates.py`
- Modify: `vfe4/types/__init__.py`
- Create: `vfe4/objective/dependency_graph.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `vfe4/validation/fixtures/h5_factorized_update_v1.json`
- Create: `vfe4/validation/h5_update_spec.py`
- Create: `tests/unit/test_h5_update_types.py`
- Create: `tests/unit/test_h5_dependency_graph.py`
- Create: `tests/unit/test_h5_update_spec.py`
- Create: `docs/preregistrations/2026-07-21-h5-update-coherence.md`

**Interfaces:**

- Produce `UpdateLabel`, `FactorId`, `UpdateSpecification`, `FrozenTensorValue`, `RecognitionSnapshot`, `H5ModelSnapshot`, `H5ReferenceState`, `FactorInputHashRecord`, `UpdateHashRecord`, `CompleteElboEvaluation`, `UpdateAttempt`, `H5PositiveCaseResult`, `H5ControlResult`, and `H5GateResult`. `CompleteElboEvaluation.factor_values` and `.factor_input_hashes` are defensive-copy `MappingProxyType` mappings in exact factor-universe order. `H5ModelSnapshot` owns only immutable model parameter-block values, normalized-factor reconstruction metadata, declared shared-storage groups, and schema hashes; `H5ReferenceState` owns the parsed update specification plus immutable initial recognition/model snapshots.
- Produce `FactorDependencyGraph(factor_universe, variable_dependencies, parameter_dependencies)`, `build_h5_reference_dependency_graph()`, and `expected_affected_factors(...)`.
- Freeze factor IDs in exact order: `initial_joint`, `model_source[1]`, `model_transition[1]`, `state_source[1]`, `state_transition[1]`, `emission[1]`, `model_source[2]`, `model_transition[2]`, `state_source[2]`, `state_transition[2]`, `emission[2]`, `recognition_entropy`.

- [ ] **Step 1: Write the H5 preregistration before executing any update.** Copy the closed taxonomy, factor universe, variable/parameter dependency table, canonical factor-input hash schema, exact `observed_affected = ordered(input_hash_before != input_hash_after)` rule, diagnostic-only value-change rule, positive cases, seven controls, snapshot/alias rules, cache/reuse rules, hash state machine, frozen quadrature orders `21/17`, per-term deterministic convergence estimates, rounding formulas, complete before/after totals, exact `epsilon_delta` formula, zero stochastic contribution, exact/MM/GEM/proposal decisions, emission-touching indecision rule, status precedence, JSON schema, and H4/H6/H7/H8/training nonclaims. Record `valid_mm` as configuration-disabled pending a named proof artifact; do not reserve a blank proof path.

  Freeze the bounded H5 state as the unchanged `h1-v1` generative factors plus `vfe4/validation/fixtures/h5_factorized_update_v1.json`, a separately declared fully factorized continuous/categorical recognition and model-state specification parsed only by `vfe4/validation/h5_update_spec.py`. The fixture contains its fixture ID, fixture schema version, factor-input-hash schema version, exact ordered factor universe, fully factorized CPU-float64 recognition initialization, immutable model parameter blocks, normalized-factor reconstruction schema, source-support masks, the exact `source_row_a2` coordinate/parent conditioning, and every declared shared-storage group. The parser reads raw bytes once, validates every required field and finite value, rejects unknown/extra fields and undeclared/ambiguous aliases, and returns canonical bytes plus hash before any dependency graph is built. Task 5 authors the bytes and pins their exact SHA-256 literal in the parser and resolved configuration before GREEN/commit; do not invent that digest before the fixture exists. Its zero cross-block recognition slopes make the `q[z0]` Markov-blanket coordinate conjugate while retaining positive source support, child transitions, and both categorical emissions elsewhere in the complete objective. The fixture/update-spec hashes, not mutable runtime constructors, identify this initial state.

  `H5LiveState` owns accepted recognition and model snapshots only by whole-snapshot replacement. `DifferentiableRecognitionState` and any differentiable model working values are ephemeral proposal inputs; they cannot be accepted state, serialized snapshots, or aliases of H1/H2 records. Accepted `H5ModelSnapshot` values are immutable `FrozenTensorValue` records and rebuild normalized factor inputs from the declared reconstruction schema. This supplies explicit ownership for `theta[state_transition_2]` and every shared parameter block without mutating H1 generative factors.

  The preregistered dependency table must include at least:

  ```text
  q[z0] -> initial_joint, state_transition[1], state_transition[2], recognition_entropy
  q[m0] -> initial_joint, model_transition[1], model_transition[2], recognition_entropy
  q[z1] -> state_transition[1], emission[1], state_transition[2], recognition_entropy
  q[m1] -> model_transition[1], state_transition[1], emission[1], model_transition[2], recognition_entropy
  q[source_row_a2] -> state_source[2], state_transition[2], recognition_entropy
  theta[state_transition_2] -> state_transition[2]
  theta[emission_1] -> emission[1]
  theta[shared_decoder_transition] -> emission[1], emission[2], state_transition[2]
  ```

  The graph builder expands source-supported child edges from the frozen parent sets, so the implementation does not hard-code only one realized source assignment.

- [ ] **Step 2: Write strict type, parser, hash, and graph tests.** Reject unknown labels, string aliases, duplicate/unknown factor IDs, unsorted affected blocks, missing complete factor IDs, mutable mappings, nonfinite terms/deltas/allowances, inconsistent accept/reject reasons, and PASS/FAIL/INCONCLUSIVE result contradictions. The update-spec parser tests reject a raw-digest mismatch, missing/extra/schema-invalid fields, source-row ambiguity, and undeclared shared storage, and require exact canonical raw-byte/hash capture. Prove `RecognitionSnapshot` and `H5ModelSnapshot` clone and detach CPU float64 values, share no storage with differentiable sources or each other except declared reconstructible sharing, expose no optimizer methods, and have stable canonical bytes/hash. Require before/after factor-input hashes for every universe ID, exact universe order, 64-hex digests, `observed_affected_factor_ids` equal to the ordered unequal-hash set, and `value_changed_factor_ids` to be accepted only as diagnostic metadata. Reject a value-derived affected set even when it happens to match one example.

  Assert the graph's exact factor universe and dependency sets. Cover unions across multiple variables/parameters, source-supported children, shared parameters, empty update blocks, unknown coordinates, and a deliberately incomplete graph. Require the incomplete graph to fail construction because each declared variable/parameter lacks its preregistered factors.

- [ ] **Step 3: Run the Task 5 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h5_update_types.py tests/unit/test_h5_dependency_graph.py tests/unit/test_h5_update_spec.py -q
  ```

  Expected: collection fails because the update types, dependency graph, and H5 update-spec parser do not exist.

- [ ] **Step 4: Implement immutable records, parser, and canonical hashing.** `FrozenTensorValue` stores dtype, shape, and finite row-major numeric tuples, not a live tensor. `RecognitionSnapshot` hashes schema plus all tensor records. `H5ModelSnapshot` hashes its schema, parameter blocks, reconstruction mapping, and declared shared-storage groups; `H5ReferenceState` binds it to the parsed raw update-spec bytes/hash. `FactorInputHashRecord` canonicalizes the exact normalized factor inputs, recognition moments/source probabilities, model parameters, and frozen complement that determine one factor expectation. `UpdateAttempt` validates that before/after schemas match, `expected_factor_ids` equals the complete declared factor universe for a full comparison, `expected_affected_factor_ids` is the dependency-graph subset, `observed_affected_factor_ids` is derived from input hashes and equals that subset exactly, `missing_factor_ids` and `extra_factor_ids` are exact differences between complete expected and observed reevaluated IDs, factor lists preserve universe order, and acceptance is compatible with the label-specific reason. It never derives affectedness from scalar values.

- [ ] **Step 5: Implement the static dependency graph.** Store both forward mappings and a validated factor universe. `expected_affected_factors` returns the factor-universe-ordered union. Recognition entropy is included for every recognition-coordinate update and excluded from model-only updates with frozen recognition. Model parameters that share storage or feed multiple normalized factors must name every factor. The graph predicts input dependencies only; evaluator scalar equality/difference is never consulted.

- [ ] **Step 6: Run the Task 5 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h5_update_types.py tests/unit/test_h5_dependency_graph.py tests/unit/test_h5_update_spec.py -q
  ```

  Expected: closed taxonomy, raw-byte update-spec capture, immutable recognition/model snapshots, complete dependency graph, and fail-closed result semantics pass.

- [ ] **Step 7: Commit Task 5.**

  ```powershell
  git add vfe4/types/updates.py vfe4/types/__init__.py vfe4/objective/dependency_graph.py vfe4/objective/__init__.py vfe4/validation/fixtures/h5_factorized_update_v1.json vfe4/validation/h5_update_spec.py tests/unit/test_h5_update_types.py tests/unit/test_h5_dependency_graph.py tests/unit/test_h5_update_spec.py docs/preregistrations/2026-07-21-h5-update-coherence.md
  git commit -m "test: freeze H5 update coherence contract"
  ```

---

### Task 6: Implement One Complete H5 Objective, Factor Trace, Cache Proofs, and Operand-Shaped Delta Budget

**Files:**

- Create: `vfe4/objective/h5_complete.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `verification/h5_budget.py`
- Create: `tests/unit/test_h5_complete_objective.py`
- Create: `tests/unit/test_h5_budget.py`

**Interfaces:**

- Produce `H5ReferenceState`, `H5LiveState`, `DifferentiableRecognitionState`, `CompleteElboEvaluator`, and `evaluate_h5_complete_elbo(state, *, cache=None) -> CompleteElboEvaluation`.
- Produce `term_allowance(value_order_21, value_order_17, rounding_inputs)`, `complete_elbo_allowance(term_allowances, signed_terms)`, `subtraction_rounding_allowance`, and `epsilon_delta(before, after) -> H5DeltaAllowance`. Every returned record exposes `convergence_estimate`, `rounding_allowance`, and `total`.
- Reuse the existing `ElboTerms` partition and its one-place complete scalar. Add factor-level trace metadata around it; do not construct a second scalar objective.

- [ ] **Step 1: Write complete-objective and budget tests.** Build the bounded H5 state from captured `h1-v1` generative factors plus the preregistered factorized recognition update-spec bytes in CPU float64. Construct the equivalent frozen H1 recognition record only inside the test and compare all 12 ordered factor IDs plus every existing `ElboTerms` field to `evaluate_local_elbo` at that common state. Require the raw expected-log-generative-factor sum plus recognition entropy and the KL-partitioned view to reconstruct exactly the one `ElboTerms.complete_elbo` field; neither view may become a second training objective. For every before/after term, independently compute order-21 and order-17 values and require `convergence_estimate=abs(v21-v17)`; analytic terms still carry explicit zero estimates rather than omitting the field.

  Update each preregistered recognition/model block in isolation. Derive `observed_affected_factor_ids` only by ordered comparison of every factor's canonical before/after input hashes, then assert exact ordered equality with the dependency graph's `expected_affected_factor_ids`. Specifically require child transitions for `q[z0]`, emission for `q[z1]` and emission parameters, and every shared factor for `theta[shared_decoder_transition]`. Record scalar `value_changed_factor_ids` separately as diagnostic evidence that cannot add, remove, or excuse an affected factor. A cache may reuse a factor only when its inputs and frozen-complement hash match; mutate one input behind the cache and require stale-cache rejection.

  Test the budget with unequal term scales, unequal deterministic quadrature differences, and condition numbers. Assert each term record contains only its own inputs and `total=convergence_estimate+rounding_allowance`; each complete before/after allowance equals the sum of every signed term total plus one final reduction-rounding term; stochastic contribution is exactly zero; and

  ```python
  epsilon_delta = (
      before_total_allowance
      + after_total_allowance
      + subtraction_rounding_allowance(before_elbo, after_elbo)
  )
  ```

  Reject a global-kappa argument, a missing convergence estimate/term allowance, wrong arity, a nonzero stochastic contribution, or negative/nonfinite scale. Add an emission-touching boundary table for `delta` immediately below, exactly at, and immediately above `epsilon_delta`: only the resolved side allowed by the label is eligible; the unresolved boundary is `INCONCLUSIVE`.

- [ ] **Step 2: Run the Task 6 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h5_complete_objective.py tests/unit/test_h5_budget.py -q
  ```

  Expected: collection fails because `vfe4.objective.h5_complete` and `verification.h5_budget` do not exist.

- [ ] **Step 3: Implement the reference live/working states without weakening H2.** Construct H5-specific differentiable leaves from immutable H1 generative values and the canonical factorized H5 recognition specification; do not reuse the H1 fixture's structured recognition law as if it were the conjugate test family. Never place autograd tensors into H1/H2 records or `InformationGaussian`. `H5LiveState` owns immutable accepted recognition/model snapshots plus explicit optimizer/RNG states. `DifferentiableRecognitionState` is ephemeral and cannot be serialized as accepted state.

- [ ] **Step 4: Implement the complete evaluator and factor trace.** Evaluate every factor in the exact universe order at both frozen quadrature orders on every uncached full comparison. Before evaluation, canonicalize and hash each factor's actual inputs; record the finite order-21/order-17 scalar values, input hash, cache disposition, absolute summands, and convergence estimate. Build `ElboTerms` once from order-21 values and return its complete scalar. A requested cache entry is accepted only when factor schema, exact input hash, quadrature orders, and frozen-complement hash all match; otherwise raise a typed stale-cache error. Defensive-copy factor values/input hashes into universe-ordered `MappingProxyType` mappings.

- [ ] **Step 5: Implement the budget functions literally.** Freeze quadrature orders `21/17`, `eps`, `gamma`, `C=4096`, and operation counts by term shape in the preregistration. Each allowance record names the term, both quadrature values, `convergence_estimate=abs(v21-v17)`, signed reported value, absolute summands, dimensions, actual SPD condition numbers, rounding allowance, and total. Complete before/after totals sum all term totals plus their own final reduction-rounding term. Initial H5 has exactly zero stochastic contribution. `epsilon_delta` is exactly `before_total + after_total + subtraction_rounding` and may not recompute operands using pooled metadata.

- [ ] **Step 6: Run the Task 6 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h5_complete_objective.py tests/unit/test_h5_budget.py -q
  ```

  Expected: one complete `ElboTerms` scalar, factor-level dependency changes, cache proofs, and operand-shaped delta budgets all pass; H2 objects remain detached and unchanged.

- [ ] **Step 7: Commit Task 6.**

  ```powershell
  git add vfe4/objective/h5_complete.py vfe4/objective/__init__.py verification/h5_budget.py tests/unit/test_h5_complete_objective.py tests/unit/test_h5_budget.py
  git commit -m "feat: add complete H5 objective accounting"
  ```

---

### Task 7: Implement Exact and Proposal Updates, Freeze-Before-Evaluate Acceptance, Rollback, and Independent Oracle

**Files:**

- Create: `vfe4/inference/h5_updates.py`
- Modify: `vfe4/inference/__init__.py`
- Create: `verification/numpy_oracles/h5_updates.py`
- Modify: `verification/numpy_oracles/__init__.py`
- Create: `tests/unit/test_h5_updates.py`
- Create: `tests/oracle/test_h5_update_oracle.py`

**Interfaces:**

- Produce `exact_conjugate_gaussian_e_update`, `exact_source_row_update`, `exact_gaussian_m_update`, `propose_generalized_em`, `propose_natural_gradient`, `freeze_recognition_candidate`, and `execute_update`.
- Produce NumPy-only `H5OracleUpdate`, `oracle_exact_e_block`, `oracle_exact_source_row`, `oracle_exact_m_block`, and `oracle_complete_delta` from raw H1 fixture bytes plus canonical H5 update-spec bytes.
- `execute_update` is the only function allowed to replace a live accepted snapshot.

- [ ] **Step 1: Write exact-update, proposal, snapshot, and oracle tests.** Require the five positive cases:

  1. exact conjugate Gaussian update of the frozen `q[z0]` block includes the initial factor, every source-supported child state transition, and recognition entropy;
  2. exact normalized categorical `q[source_row_a2]` uses prior plus complete expected transition score and sums to one on positive support;
  3. exact Gaussian M update of `theta[state_transition_2]` holds a detached, nonaliasing `RecognitionSnapshot` fixed;
  4. a backtracked gradient proposal with complete resolved `delta > epsilon_delta` is accepted as `generalized_em`;
  5. a deliberately oversized natural-gradient proposal decreases the complete objective, is rejected, and leaves all live hashes unchanged.

  For every case compare candidate parameters and complete delta to the NumPy oracle. Assert gradient proposals are built in `DifferentiableRecognitionState`, then frozen before `CompleteElboEvaluator` sees them. Monkeypatch the evaluator to inspect `requires_grad=False`, no storage alias, finite CPU float64, and stable hash. For each attempt, independently recompute before/after factor-input hashes, derive `observed_affected_factor_ids`, and require exact ordered equality with the dependency graph even when one affected scalar happens to remain numerically equal. Monkeypatch H2 `InformationGaussian` mutation/autograd paths to raise; H5 still works without changing H2.

- [ ] **Step 2: Run the Task 7 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py -q
  ```

  Expected: collection fails because the H5 update and oracle modules do not exist.

- [ ] **Step 3: Implement exact coordinates.** Derive the conjugate Gaussian target from all expected Markov-blanket factors and solve it directly. Compute the categorical source row with support mask plus `logsumexp` normalization. Compute the Gaussian M optimum from sufficient statistics of the immutable recognition snapshot. Exact implementations do not call autograd or generic optimizers.

- [ ] **Step 4: Implement gradient proposals in the separate working representation.** Use `torch.autograd.grad` only on declared active leaves. Natural-gradient proposals retain `UpdateLabel.NATURAL_GRADIENT_PROPOSAL`; they never enter the exact-coordinate branch. GEM backtracking proposes finite step sizes in frozen order, freezes each candidate, evaluates the complete objective at quadrature orders 21/17, and accepts only the first `delta > epsilon_delta` candidate. Any emission-touching candidate inside or on the total allowance boundary is rejected as unresolved and surfaced to H5 as `INCONCLUSIVE`, not ordinary successful rollback evidence.

- [ ] **Step 5: Implement transactional acceptance and rollback.** Before proposal, canonical-hash live model, recognition, optimizer, and RNG states. Evaluate `before`; compute the expected dependency set; create/freeze candidate; evaluate `after`; derive `observed_affected_factor_ids` solely from before/after input hashes; compute diagnostic value-change IDs separately; require exact expected/observed affected equality; calculate missing/extra/cache/reuse sets and `epsilon_delta` from the two complete total allowances; then decide by label. On rejection, discard candidate and recompute all live hashes. On acceptance, atomically replace only declared blocks and verify frozen-complement hashes. Return an `UpdateAttempt` for success, rejection, or typed failure; never mutate first and attempt to reverse floating-point operations.

- [ ] **Step 6: Implement the independent NumPy oracle.** Parse raw fixture/update-spec bytes independently, compute exact conditional Gaussian/source/M updates from dense moments and normalized scores, and evaluate the full H1-shaped objective. The oracle imports neither PyTorch nor production H5 modules and shares no dependency-graph implementation.

- [ ] **Step 7: Run the Task 7 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py -q
  ```

  Expected: all five positive cases match the independent oracle; snapshot separation, label rules, resolved GEM acceptance, and mutation-free rejection pass.

- [ ] **Step 8: Commit Task 7.**

  ```powershell
  git add vfe4/inference/h5_updates.py vfe4/inference/__init__.py verification/numpy_oracles/h5_updates.py verification/numpy_oracles/__init__.py tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py
  git commit -m "feat: add transactional H5 update semantics"
  ```

---

### Task 8: Add the H5 Gate, Mandatory Adversarial Controls, Status Mapping, and Payload

**Files:**

- Create: `verification/h5_gate.py`
- Create: `tests/promotion/test_h5_gate.py`
- Modify: `docs/preregistrations/2026-07-21-h5-update-coherence.md`

**Interfaces:**

- Produce `H5GateEvaluation(result, fixture_hash, positive_attempts, controls, oracle_results, allowances, validation_payload)`.
- Produce `evaluate_h5(config: ResolvedConfig, *, h1_fixture_bytes: bytes) -> H5GateEvaluation` and `h5_validation_payload(evaluation) -> dict[str, object]`.

- [ ] **Step 1: Write the promotion test with exact invariant names.** Require this ordered tuple:

  ```text
  fixture_and_objective_schema_identity
  closed_update_taxonomy
  dependency_graph_complete
  exact_gaussian_e_coordinate
  exact_categorical_source_coordinate
  exact_gaussian_m_coordinate_fixed_recognition
  accepted_resolved_generalized_em
  rejected_proposal_rollback
  child_factor_omission_detected
  emission_factor_omission_detected
  unresolved_gem_acceptance_detected
  natural_gradient_mislabel_detected
  rejection_mutation_detected
  changed_input_equal_value_detected
  changed_value_unchanged_input_not_affected
  all_delta_allowances_operand_shaped
  ```

  Require a complete `UpdateAttempt` for each positive case and each control, including before/after full `ElboTerms`, order-21/order-17 convergence estimates, before/after factor-input hashes, expected/observed affected IDs, diagnostic value-change IDs, schemas, complement/candidate/live/RNG hashes, delta/total allowance, decision, and label. Assert H5 status is independent of the H4 timing result.

- [ ] **Step 2: Run the Task 8 test for RED.**

  ```powershell
  python -m pytest tests/promotion/test_h5_gate.py -q
  ```

  Expected: collection fails because `verification.h5_gate` does not exist.

- [ ] **Step 3: Implement the seven mandatory controls as targeted fault injection.** Inject each fault behind a test/gate-only evaluator or controller seam:

  - suppress one expected child `state_transition[2]` evaluation while keeping other factors;
  - suppress `emission[1]` for an emission-touching proposal;
  - force controller acceptance when `abs(delta) <= epsilon_delta` under `generalized_em`;
  - submit a natural-gradient proposal with requested label `exact_coordinate`;
  - mutate one live-state field and advance RNG after a rejected proposal.
  - change one factor's canonical input bytes while constructing a mathematically equal scalar value, proving input-hash affectedness catches the value-based false negative;
  - perturb one reported factor scalar while keeping canonical factor inputs byte-identical, proving the factor is absent from `observed_affected_factor_ids` despite appearing in diagnostic `value_changed_factor_ids`; the separate deterministic reevaluation check still detects the corrupted scalar.

  Each control passes only when the intended invariant detects the fault and returns the expected typed reason. Do not catch all faults as one generic exception or alter production behavior globally.

- [ ] **Step 4: Implement H5 status precedence.** Validate fixture/schema/taxonomy/graph availability first, then complete before/after factor-input hash coverage and exact expected/observed affected equality, positive-case finite/oracle/factor completeness, every term's deterministic convergence estimate and total allowance, label-specific delta decisions, rollback hashes, and controls. A decisive positive-case, affectedness, or control miss is `FAIL`. Missing evidence, nonfinite state, schema mismatch with unknown cause, stale-cache ambiguity, absent MM proof request, nonzero stochastic contribution, or an emission-touching/other required delta that does not clear the complete total allowance is `INCONCLUSIVE`. H5 is `PASS` only when every positive and control invariant passes.

- [ ] **Step 5: Emit the complete `validation/h5.json` schema.** Include gate/status/obligations; raw fixture hash; config/objective/update schema hashes; exact taxonomy; dependency graph; factor universe; positive/seven-control specifications; every complete before/after term and factor trace; every factor's before/after input hash; expected/observed reevaluated/observed affected/diagnostic value-changed/missing/extra/cache/reuse IDs; variables/parameters; candidate/complement/live/recognition/optimizer/RNG hashes; exact label/autograd/damping/line-search facts; quadrature orders; every term's two values, deterministic convergence estimate, rounding allowance, and total; complete before/after totals; zero stochastic contribution; subtraction rounding; exact epsilon formula; delta/decision and emission-touching decisiveness; oracle values; ordered invariants; bounded H5 claim; disabled-MM reason; and H6--H8/training nonclaims.

- [ ] **Step 6: Run the Task 8 test for GREEN.**

  ```powershell
  python -m pytest tests/promotion/test_h5_gate.py -q
  ```

  Expected: all positive cases pass, each of seven faults is detected/classified by its named control, value-based affectedness false negatives/positives are impossible, every term carries deterministic convergence plus rounding, every attempt contains the complete evidence record, and PASS/FAIL/INCONCLUSIVE precedence matches the preregistration.

- [ ] **Step 7: Commit Task 8.**

  ```powershell
  git add verification/h5_gate.py tests/promotion/test_h5_gate.py docs/preregistrations/2026-07-21-h5-update-coherence.md
  git commit -m "test: add the H5 update coherence gate"
  ```

---

### Task 9: Extend Typed Configuration, One Click-Run, Atomic Artifact Family, and Environment Provenance Through H5

**Files:**

- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `verification/run_gates.py`
- Modify: `vfe4/artifacts/provenance.py`
- Modify: `verify_vfe4.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_atomic_artifacts.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h4-information-cost.md`
- Modify: `docs/preregistrations/2026-07-21-h5-update-coherence.md`

**Interfaces and compatibility:**

- Extend accepted gate prefixes only to `("H1",)`, `("H1","H2")`, `("H1","H2","H3")`, and `("H1","H2","H3","H4","H5")`. Do not accept H4 without H5, H5 without H4, reordered/duplicate gates, or any H6--H8 prefix in this milestone.
- Add `h4: H4ValidationConfig | None` and `h5: H5ValidationConfig | None`. Both are absent for shorter prefixes and both are required for the H5 prefix. They remain separately hashed sections and produce separate results. `H5ValidationConfig` freezes `update_spec_fixture_id`, the exact post-authoring `update_spec_expected_sha256`, update-spec schema version, factor-input-hash schema version, and the ordered factor-universe IDs; resolution rejects any disagreement.
- `H4ValidationConfig` exposes the exact parity expression, warmup/timed pair-index tuples, `warmups_count_toward_balance=False`, the canonical `H4_PRIMARY_TIMED_BALANCE` 20-row tuple `(seed, AB, BA)`, `primary_timed_ab_total=110`, and `primary_timed_ba_total=110`; resolution recomputes these values from independent horizon/seed/kind/pair indices and rejects any disagreement.
- Extend the explicit result union to include `H4GateResult` and `H5GateResult`; do not merge their measurements or status.

- [ ] **Step 1: Write focused configuration, integration, and artifact tests.** Assert the one editable `CONFIG` resolves to ordered H1--H5 and includes exact H4 horizon/seed/kind traversal with independent zero-based indices, AB exactly when `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, warmup/timed pair indices, `warmups_count_toward_balance=False`, the exact 20-row `H4_PRIMARY_TIMED_BALANCE`, exactly ten primary `6/5` rows and ten `5/6` rows, exact aggregate timed totals `AB=110` and `BA=110`, seeds/dimensions/protocol/statistics/environment constraints, inclusive condition envelope, `1e-9` solver budget, and strict `1e-4` decisiveness cap; plus exact H5 update-spec fixture ID/digest/schema fields, taxonomy/dependency/input-hash rule/cases/seven controls, quadrature orders `21/17`, deterministic-convergence-plus-rounding budgets, zero stochastic contribution, and epsilon formula. Reject a formula based on flattened `problem_index`, either swapped per-seed count, a per-seed imbalance masked by correct aggregate totals, aggregate totals other than `110/110`, or a true warmup-balance flag. Test every envelope/cap/delta boundary. Resolve every shorter compatibility prefix and prove it contains no H4/H5 config, does not read/hash/capture H4/H5/update-spec inputs, does not run timing/updates, and publishes no H4/H5 payload/provenance keys.

  One mocked H1--H5 `main()` call evaluates each gate once and publishes exactly one manifest-checked directory containing:

  ```text
  config.json
  provenance.json
  environment.json
  validation/h1.json
  validation/h2.json
  validation/h3.json
  validation/h4.json
  validation/h5.json
  manifest.sha256
  ```

  Assert H4 and H5 statuses may differ and both survive round-trip publication. Preserve path containment, alias/reparse-point defenses, no-overwrite atomic publication, and prior manifests.

- [ ] **Step 2: Run the Task 9 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: failures show the resolver/runner currently stop at H3 and no H4/H5 payloads or environment fields exist.

- [ ] **Step 3: Add exact typed H4/H5 sections and fail-closed resolution.** Canonicalize every frozen literal. Derive arm order from independent indices, recompute the primary 20-row timed balance, pattern counts, and aggregate totals, and require exact equality with the configured literals before returning `ResolvedConfig`. Reject changed horizon/seed/kind traversal, seed order/count, size order, primary dimension, warmup/timed pair indices, parity formula, flattened-`problem_index` parity, warmup inclusion, any per-seed primary balance row, pattern-count or `110/110` aggregate mismatch, no-between-repetitions/postflight tag, bootstrap settings, threshold, condition-envelope bound/inclusivity, solver budget, decisiveness cap/strictness, timer boundary tag, solver labels, thread/dtype/device, H5 update-spec fixture/digest/schema fields, H5 label order, factor/input-hash IDs, positive/seven-control IDs, quadrature orders, deterministic convergence rule, nonzero stochastic contribution, disabled-MM policy, or delta formula. Reject H4/H5 section presence for shorter prefixes and absence of either section for the coupled prefix.

- [ ] **Step 4: Extend conditional one-time capture and ordered evaluation.** Capture `h1-v1` once for H1/H2/H5, H3 coupled/zero bytes once for H3/H4 only when consumed, and `h5_factorized_update_v1.json` bytes once only for the coupled H1/H2/H3/H4/H5 prefix. Pass the same captured H5 byte object to every H5 production evaluator and oracle adapter. Evaluate H1, H2, H3, H4, H5 in order. H4 receives H3 bytes; H5 receives H1 bytes and update-spec bytes. Publish only after both expensive gates return. Shorter prefixes must neither read, hash, capture, nor publish the H5 update-spec. Aggregate status is `fail` if any gate fails, otherwise `inconclusive` if any is inconclusive, otherwise `pass`.

- [ ] **Step 5: Extend environment and provenance.** Preserve current source/config/dirty-content security fields and expose the canonical `dirty_content_digest` used by milestone preflight/rechecks. Add timing clock implementation/resolution/monotonicity, process CPU affinity, logical/physical CPU counts when available, processor/platform, PyTorch intra/inter-op threads, `torch.__config__.show()` digest/text, NumPy BLAS configuration digest/text, CUDA availability (expected false for H4), and exact values/presence of `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. Record H4 prior/effective/restored intra-op thread values and any restoration error, ordered gate states, distinct H4/H5 config hashes, factor/update/model-snapshot schema hashes, `fixture_hashes["h5-factorized-update-v1"]`, `gate_fixture_consumers["H5"]=("h1-v1","h5-factorized-update-v1")`, H5 raw update-spec digest/schema values, H4 traversal/problem-factor hashes/parity formula/warmup-exclusion flag/expected and observed primary per-seed plus `110/110` aggregate timed balance/envelope/budget/cap, H5 factor-input hash schema/quadrature/allowance rules, and H4/H5 bounded-claim/nonclaim tags. Shorter prefixes contain none of the H5 update-spec fields.

- [ ] **Step 6: Extend the one launcher and bounded documentation.** Keep one `CONFIG`, `main`, and script guard. Print H1--H5 statuses separately and one artifact path. README and the H4 preregistration state the independent-index parity formula, literal primary 20-row timed balance, ten/ten pattern split, exact `110/110` totals, and that warmups are excluded from balance; they do not prestate H4 speed or H5 pass results. Use exact live path case `Manuscripts/...` in every source citation and add a focused documentation assertion that rejects any differently cased variant. Explicitly defer H6--H8 and training.

- [ ] **Step 7: Run the Task 9 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: all accepted prefixes preserve their exact surfaces; one mocked H1--H5 click-run publishes eight JSON files plus one valid manifest; H4/H5 remain separate; prior prefix artifacts contain no future-gate data.

- [ ] **Step 8: Commit Task 9.**

  ```powershell
  git add vfe4/config/schema.py vfe4/config/resolve.py verification/run_gates.py vfe4/artifacts/provenance.py verify_vfe4.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py README.md docs/preregistrations/2026-07-21-h4-information-cost.md docs/preregistrations/2026-07-21-h5-update-coherence.md
  git commit -m "feat: publish separate H4 and H5 verification gates"
  ```

---

### Task 10: Produce One Coupled Exact-Revision H4/H5 Milestone Record

**Files:**

- Modify: none. Every tracked protocol, implementation, test, preregistration, launcher, and artifact-schema file is committed before selecting the candidate revision.
- Produce outside tracked source: `C:\tmp\vfe4-h4-h5-preflight.json`, `C:\tmp\vfe4-h4-h5-milestone.xml`, one atomic run directory under the configured run root, and `.verification/h4-h5-<FULL_HEAD>-ledger.json`.
- Preserve `.verification/ledger.json`, all `.verification/h3-*-ledger.json`, and any prior `.verification/h4-h5-*-ledger.json` byte-for-byte. Do not commit `.verification` or generated run artifacts.

**Why one milestone is allowed:** H4 and H5 are evaluated by the same ordered click-run, config snapshot, source revision, environment capture, JUnit revision, artifact manifest, and fixture-byte snapshot. Their gate results, payloads, statuses, and ledger claims remain separate. If either implementation changes, both evidence sets are invalidated and the replacement candidate again uses one common revision.

- [ ] **Step 1: Fail-closed on tracked scope, unexpected untracked content, revision, dirty-content identity, activation, and preserved ledgers.**

  ```powershell
  $candidateHead = (git rev-parse HEAD).Trim()
  if ($candidateHead.Length -ne 40) { throw 'H4/H5 requires a full 40-character HEAD' }
  $requiredTracked = @(
      'docs/superpowers/plans/2026-07-21-vfe4-h4-h5-cost-update.md',
       'docs/preregistrations/2026-07-21-h4-information-cost.md',
       'docs/preregistrations/2026-07-21-h5-update-coherence.md',
       'vfe4/validation/fixtures/h5_factorized_update_v1.json',
       'vfe4/validation/h5_update_spec.py',
       'vfe4/types/h4.py', 'vfe4/types/updates.py', 'vfe4/types/__init__.py',
      'vfe4/generative/reference_h4.py', 'vfe4/generative/__init__.py',
      'vfe4/inference/h4_instrumentation.py', 'vfe4/inference/h4_solvers.py',
      'vfe4/inference/h5_updates.py', 'vfe4/inference/__init__.py',
      'vfe4/objective/dependency_graph.py', 'vfe4/objective/h5_complete.py',
      'vfe4/objective/__init__.py',
      'verification/numpy_oracles/h4_gaussian.py',
      'verification/numpy_oracles/h5_updates.py',
      'verification/numpy_oracles/__init__.py',
      'verification/h4_budget.py', 'verification/h4_statistics.py',
      'verification/h4_gate.py', 'verification/h5_budget.py',
      'verification/h5_gate.py', 'verification/run_gates.py',
      'vfe4/config/schema.py', 'vfe4/config/resolve.py',
      'vfe4/artifacts/provenance.py', 'verify_vfe4.py', 'README.md',
      'tests/unit/test_h4_problem.py', 'tests/unit/test_h4_solvers.py',
      'tests/unit/test_h4_instrumentation.py', 'tests/unit/test_h4_statistics.py',
      'tests/oracle/test_h4_numpy_oracle.py', 'tests/promotion/test_h4_gate.py',
       'tests/unit/test_h5_update_types.py', 'tests/unit/test_h5_dependency_graph.py',
       'tests/unit/test_h5_update_spec.py',
       'tests/unit/test_h5_complete_objective.py', 'tests/unit/test_h5_budget.py',
      'tests/unit/test_h5_updates.py', 'tests/oracle/test_h5_update_oracle.py',
      'tests/promotion/test_h5_gate.py', 'tests/unit/test_config.py',
      'tests/unit/test_atomic_artifacts.py', 'tests/integration/test_verify_vfe4.py'
  )
  $tracked = @(git ls-files -- $requiredTracked)
  $missingTracked = @($requiredTracked | Where-Object { $_ -notin $tracked })
  if ($missingTracked.Count -ne 0) {
      throw "H4/H5 candidate has missing or untracked required files: $($missingTracked -join ', ')"
  }
  git diff --exit-code
  git diff --cached --exit-code
  $nonignoredUntracked = @(git ls-files --others --exclude-standard)
  $unexpectedUntracked = @(
      $nonignoredUntracked |
          Where-Object { $_ -ne '.verification' -and -not $_.StartsWith('.verification/') }
  )
  if ($unexpectedUntracked.Count -ne 0) {
      throw "nonignored untracked content outside .verification blocks H4/H5: $($unexpectedUntracked -join ', ')"
  }
  if (Test-Path -LiteralPath '.verification/active.json') {
      throw 'existing verification activation blocks H4/H5; preserve and resolve its owning workflow'
  }
  $preflightDirtyDigest = (& python -c "from pathlib import Path; from verify_vfe4 import CONFIG; from vfe4.artifacts.provenance import dirty_content_digest; print(dirty_content_digest(Path.cwd(), Path(CONFIG['artifacts']['run_root'])))").Trim()
  if ($preflightDirtyDigest -notmatch '^[0-9a-f]{64}$') { throw 'invalid preflight dirty-content digest' }
  $ledgerHashes = @(
      Get-ChildItem -LiteralPath '.verification' -File -Filter '*ledger.json' -ErrorAction SilentlyContinue |
      Sort-Object FullName |
      Get-FileHash -Algorithm SHA256
  )
  [ordered]@{
      candidate_head = $candidateHead
      dirty_content_digest = $preflightDirtyDigest
      required_tracked = $requiredTracked
      nonignored_untracked = $nonignoredUntracked
      prior_ledgers = @($ledgerHashes | ForEach-Object {
          [ordered]@{ path = $_.Path; sha256 = $_.Hash.ToLowerInvariant() }
      })
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 -LiteralPath C:\tmp\vfe4-h4-h5-preflight.json
  ```

  Expected: exact 40-character revision; every named plan/preregistration/source/config/launcher/test is tracked; no tracked modification; no nonignored untracked content outside `.verification`; no active marker; valid dirty-content digest; and a machine-readable retained SHA-256 table for every prior ledger. Do not delete unexpected content to make this pass; preserve it and resolve ownership.

- [ ] **Step 2: Run the only coupled milestone full regression and parse JUnit.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h4-h5-milestone.xml
  ```

  Expected: pytest exits zero. Parse suite/test/failure/error/skip totals only from `C:\tmp\vfe4-h4-h5-milestone.xml`. Immediately recheck `HEAD`, both diffs, the no-unexpected-untracked rule, and `dirty_content_digest == preflightDirtyDigest`; any mismatch invalidates the run. Do not report terminal dots or remembered earlier totals. Do not run another full suite unless a subsequent source/test/config/protocol change invalidates this candidate.

- [ ] **Step 3: Run the single full click-run and verify the artifact without rerunning H4.**

  ```powershell
  python verify_vfe4.py
  ```

  Expected: the launcher prints separate H1, H2, H3, H4, and H5 statuses and one run directory. Independently recompute `manifest.sha256`; verify exact source/config/environment identity; artifact `dirty_content_digest == preflightDirtyDigest`; raw fixture and H4 problem hashes; H4 prior/effective/restored thread fields; exact indexed traversal and parity formula; separate warmup/timed event order with warmups excluded from balance; complete H4 raw timing table; primary observed balance equal to the literal 20-row table, ten `6/5` plus ten `5/6` seeds, and exactly `110 AB/110 BA`; bootstrap/envelope/budget/decisiveness metadata; raw H5 update-spec fixture ID/SHA-256/schema and H5-only fixture-consumer provenance; complete H5 factor-input/affectedness/quadrature/allowance attempts, immutable model-snapshot ownership/no-alias evidence, and seven controls; and separate `validation/h4.json` / `validation/h5.json`. Recheck `HEAD`, both diffs, unexpected-untracked rule, and dirty digest after inspection. This is the only full 20-seed H4 timing execution for the candidate.

- [ ] **Step 4: Have fresh reviewers inspect existing evidence only.** Assign at least these independent reviews:

  - H4 protocol/statistics reviewer: exact horizon/seed/kind traversal with independent indices, common factors, independent arms, all three warmup pairs before all 11 timed pairs per problem, AB iff `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, warmups excluded from balance, primary observed rows exactly equal to the literal 20-row balance table, ten `6/5` plus ten `5/6` timed seeds, exact aggregate `110 AB/110 BA`, no conversion/hashing/diagnostics between timed repetitions, fixed batched conversion order, seeds as inferential units, bootstrap implementation, and threshold/status mapping;
  - H4 numerical/runtime reviewer: exact optimum, inclusive scaled conditioning envelope and boundaries, `h/J`/moment/objective equivalence, per-invariant scales, exact solver contribution, strict allowance/scale cap, no unbalanced H2 diagnostic, canonical immutable selected moments, real-operation instrumentation, raw timing/environment/BLAS/affinity provenance;
  - H5 theory/dependency reviewer: exact-case `Manuscripts/...` Markov blankets, same complete ELBO, dependency prediction versus input-hash-derived observed affected sets, exact/source/M/GEM semantics, MM rejection, factor-universe completeness;
  - H5 implementation/transaction reviewer: captured update-spec raw bytes/digest/parser/schema, immutable recognition and model-snapshot ownership with declared shared storage only, differentiable-working versus immutable-snapshot boundary, fixed recognition M-block, order-21/order-17 convergence estimates for every term, complete total allowances and exact epsilon formula, emission-touching indecision, acceptance/rollback hashes, cache/reuse proofs, seven controls, and value-change diagnostic nonauthority;
  - artifact/compatibility reviewer: required tracked-file list, no unexpected untracked content, stable dirty-content digest, separate H4/H5 statuses/payloads, exact prefix behavior, atomic manifest, prior-ledger hashes, H6--H8/training nonclaims.

  Reviewers cite source lines, focused command outputs, JUnit XML, click artifact fields, and preregistrations. They do not rerun tests or timings. Before ledger activation, resolve every Critical/Important issue by returning to its owning task; any tracked change invalidates the candidate and requires one replacement coupled milestone run at the new revision. Recheck exact `HEAD`, tracked diffs, unexpected-untracked rule, and dirty-content digest after review and before Step 5.

- [ ] **Step 5: Start, populate, and validate the coupled revision-specific ledger.** Read the verification contract and code, mathematics, evidence, experiment, and general criterion files before assigning states. Recheck revision/activation, then:

  ```powershell
  $preflight = Get-Content -Raw -LiteralPath C:\tmp\vfe4-h4-h5-preflight.json | ConvertFrom-Json
  $candidateHead = (git rev-parse HEAD).Trim()
  if ($candidateHead -ne $preflight.candidate_head) { throw 'candidate HEAD changed before ledger activation' }
  git diff --exit-code
  git diff --cached --exit-code
  $unexpectedUntracked = @(
      git ls-files --others --exclude-standard |
          Where-Object { $_ -ne '.verification' -and -not $_.StartsWith('.verification/') }
  )
  if ($unexpectedUntracked.Count -ne 0) { throw 'unexpected untracked content before ledger activation' }
  $currentDirtyDigest = (& python -c "from pathlib import Path; from verify_vfe4 import CONFIG; from vfe4.artifacts.provenance import dirty_content_digest; print(dirty_content_digest(Path.cwd(), Path(CONFIG['artifacts']['run_root'])))").Trim()
  if ($currentDirtyDigest -ne $preflight.dirty_content_digest) { throw 'dirty-content digest changed before ledger activation' }
  $ledger = ".verification/h4-h5-$candidateHead-ledger.json"
  if (Test-Path -LiteralPath '.verification/active.json') {
      throw 'existing verification activation blocks H4/H5'
  }
  if (Test-Path -LiteralPath $ledger) {
      throw "revision-specific H4/H5 ledger exists and must not be overwritten: $ledger"
  }
  & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" start --cwd . --ledger $ledger --mode closure
  ```

  Populate one claim per check, keeping H4 experiment claims and H5 code/mathematics/evidence claims distinct. H4 claims cover indexed traversal/parity/no-inter-repetition-work identity; warmup exclusion; exact primary per-seed `H4_PRIMARY_TIMED_BALANCE`, ten/ten pattern counts, and `110/110` aggregate timed balance; independent solver reachability; scaled condition-envelope eligibility; terminal equivalence; per-invariant solver budgets and strict allowance decisiveness; canonical immutable selected moments; raw repetition completeness; seed-level statistics; bootstrap interval; threshold decision; thread set/verify/restore environment identity; and secondary count/memory nonclaim. H5 claims cover update-spec fixture raw digest/parser/schema/provenance, immutable model-snapshot ownership, taxonomy, graph completeness, complete before/after factor-input hashes, exact expected/observed affected equality, diagnostic-only value changes, each positive case, all seven adversarial controls, every term's deterministic quadrature convergence plus rounding, complete before/after total allowances, exact zero-stochastic epsilon formula, emission-touching decisiveness, full factor/term evaluation, snapshot separation, fixed recognition, label-specific acceptance, rollback hashes, and cache/reuse proof. Add separate required-tracked-scope, exact-case manuscript source, artifact/JUnit, dirty-content-digest, and prior-ledger-preservation claims.

  Give every assessed claim at least two unique views and exactly one adjudicator. Escalate to four/eight for recorded triggers; high/critical closure also links a skeptic. H4 experiment/code claims close only with current mechanical or reproduced-output evidence. Mathematical correctness claims close only with a derivation/formal proof, never timing or numerical agreement. Source/general claims require current primary/reproduced source evidence. Missing evidence or unresolved disagreement is `INCONCLUSIVE`, not a vote.

  Populate the exact generated ledger with `apply_patch` if the installed tool has no claim-add operation, then validate:

  ```powershell
  $candidateHead = (git rev-parse HEAD).Trim()
  $ledger = ".verification/h4-h5-$candidateHead-ledger.json"
  & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" validate $ledger --cwd .
  ```

  Expected: validation exits zero; every claim matches the current artifact revision; H4/H5 outcomes remain separate; the ledger records the unchanged preflight dirty-content digest; prior ledgers retain their Step 1 hashes. Do not manually remove the active marker or reuse the ledger path for another revision.

  If a source-changing defect is discovered after `start`, stop implementation immediately. Populate every affected/current claim as `INCONCLUSIVE` with the exact repair obligation, validate and report this revision-specific ledger so verification tooling—not a manual delete—retires `.verification/active.json`, and preserve the ledger. Only after that retirement may the owning task be repaired and committed. The repaired revision gets a new full 40-character path, one replacement joint JUnit/click-run, fresh preflight JSON, and a new ledger; it never reuses evidence or the old ledger.

- [ ] **Step 6: Perform the final read-only cross-check and report exact evidence surfaces.**

  ```powershell
  git rev-parse HEAD
  git diff --exit-code
  git diff --cached --exit-code
  $preflight = Get-Content -Raw -LiteralPath C:\tmp\vfe4-h4-h5-preflight.json | ConvertFrom-Json
  $closureHead = (git rev-parse HEAD).Trim()
  if ($closureHead -ne $preflight.candidate_head) { throw 'candidate HEAD changed at closure' }
  $requiredTracked = @($preflight.required_tracked)
  $stillTracked = @(git ls-files -- $requiredTracked)
  $missingTracked = @($requiredTracked | Where-Object { $_ -notin $stillTracked })
  if ($missingTracked.Count -ne 0) { throw "required tracked file missing at closure: $($missingTracked -join ', ')" }
  $unexpectedUntracked = @(
      git ls-files --others --exclude-standard |
          Where-Object { $_ -ne '.verification' -and -not $_.StartsWith('.verification/') }
  )
  if ($unexpectedUntracked.Count -ne 0) { throw 'unexpected untracked content at closure' }
  $closureDirtyDigest = (& python -c "from pathlib import Path; from verify_vfe4 import CONFIG; from vfe4.artifacts.provenance import dirty_content_digest; print(dirty_content_digest(Path.cwd(), Path(CONFIG['artifacts']['run_root'])))").Trim()
  if ($closureDirtyDigest -ne $preflight.dirty_content_digest) { throw 'dirty-content digest changed at closure' }
  foreach ($prior in @($preflight.prior_ledgers)) {
      if (-not (Test-Path -LiteralPath $prior.path)) { throw "prior ledger missing at closure: $($prior.path)" }
      $actualPriorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $prior.path).Hash.ToLowerInvariant()
      if ($actualPriorHash -ne $prior.sha256) { throw "prior ledger changed at closure: $($prior.path)" }
  }
  Get-FileHash -Algorithm SHA256 -LiteralPath C:\tmp\vfe4-h4-h5-milestone.xml
  Get-ChildItem -LiteralPath '.verification' -File -Filter '*ledger.json' -ErrorAction SilentlyContinue |
      Sort-Object FullName |
      Get-FileHash -Algorithm SHA256
  ```

  Expected: `HEAD` matches the revision in preflight JSON, JUnit, artifact provenance, and ledger; every required file remains tracked; tracked source is unchanged; no nonignored untracked content exists outside `.verification`; preflight/artifact/ledger/closure dirty-content digests agree; prior ledger hashes match Step 1; the only new ledger is `.verification/h4-h5-<FULL_HEAD>-ledger.json`. Report JUnit totals from XML, artifact path, H4 status/interval, H5 status, dirty-content digest, and validated ledger path. Do not add a post-evidence documentation commit or rerun the suite/timing.

## Out of Scope for This Plan

- H6 prefix-measurability, language modeling, held-out prediction, matched predictive controls, or dataset/tokenizer work.
- H7 internal population-frame covariance, decoder contragredience, Jacobian covariance, or gauge claims.
- H8 `T=128,K=20` sparse-scale/allocation promotion or a claim that H4's `D=256` diagnostic is H8 evidence.
- Any training launcher, checkpoint, optimizer schedule for training, WikiText-103 run, perplexity result, or predictive experiment.
- GPU, mixed precision, float32/bfloat16, multi-thread benchmark claims, cross-machine generalization, or energy-use claims.
- A general theorem that information coordinates are faster, use less memory, or converge better. H4 is bounded to the frozen hardware/protocol and exact primary endpoint.
- A positive valid-MM implementation without its complete revision-bound proof artifact.
- Stochastic H5 objectives, Monte Carlo error budgets, unrolled/implicit inference, learned frames, or H7-sensitive group updates.
- Treating natural-gradient, SGD, Adam, or truncated iterations as exact coordinate ascent.
- Research-vault ingestion. The vault was consulted read-only; any new result is offered for separate user-confirmed ingest only after the coupled milestone exists.

## Self-Review of Plan Completeness

- **Spec coverage:** H4 and H5 remain separate results/payloads/statuses. H4 freezes the H3 anchor as a generic normalized-affine schedule, exact scaled dimensions, fixed `N(0,I_8)` initial law, PCG64 draw order and distributions, `m_t`-then-`z_t|m_t` factorization, all-zero transition-block control, independent arms, three per-problem warmup pairs followed by 11 timed pairs, independent-index parity, warmup exclusion, the literal primary 20-row balance table, ten/ten per-seed pattern split, exact `110 AB/110 BA` timed aggregate, batch-post-timing conversions, seed-level bootstrap threshold, inclusive scaled conditioning envelope, per-invariant solver budget and strict allowance/scale cap, mean/covariance selected moments, raw times, secondary memory/count nonclaim, set/verify/finally-restore one-thread CPU float64, and provenance. H5 now specifies a tracked parsed raw-byte update fixture, post-authoring exact digest pin, immutable model-snapshot ownership, and declared shared storage while preserving immutable H2 state, separating differentiable working state from frozen snapshots, closing the update taxonomy, rejecting unsupported MM, deriving affected factors only from before/after input hashes, retaining value changes as diagnostics, including all seven controls, recording order-21/order-17 convergence for every term, summing complete before/after total allowances, freezing zero stochastic contribution, and using the exact subtraction-rounding `epsilon_delta` rule.
- **Interface consistency:** `H4GaussianSolver` lives in the inference layer while `H4NeutralProblem`, solver results, and tuple-ordered selected moments remain dependency-light protocol types. One generated problem feeds both independent solver arms and the oracle; no conversion, hash, count, or diagnostic work occurs between timed representations. `CompleteElboEvaluation` is produced only by the complete evaluator and embedded twice in every `UpdateAttempt`; `observed_affected_factor_ids` comes only from its ordered factor-input hashes, and `execute_update` alone accepts or rolls back. H5's parser binds raw update-spec bytes to `H5ReferenceState`, which owns immutable recognition/model snapshots rather than H1/H2 mutation. The runner captures that fixture only for H5 and publishes separate payloads.
- **Evidence discipline:** Focused RED/GREEN commands are noncumulative. The milestone preflight requires every plan, preregistration, source, config, launcher, and test file to be tracked; rejects nonignored untracked content outside `.verification`; records the exact dirty-content digest and prior-ledger hashes; and rechecks them through closure. The only full suite and full timing occur at one shared exact revision. Reviewers inspect rather than rerun. A defect found after ledger activation closes the current revision `INCONCLUSIVE`, preserves it, repairs only after tool-driven retirement, and permits exactly one replacement revision/run/ledger. The coupled ledger separates claims by gate and preserves every earlier ledger.
- **Placeholder scan:** The plan contains no unspecified H4 generator distributions, control blocks, selected-moment inventory, thread restoration rule, H5 fixture producer/parser, model-snapshot owner, or required-tracked surface. The exact H5 fixture SHA-256 is intentionally not invented: Task 5 authors the tracked bytes and pins its measured digest before GREEN/commit. H6--H8/training and MM activation remain explicit nonclaims, not hidden implementation gaps.
- **American English:** Terminology uses American English throughout.

Plan implementation must stop after Task 10 with the exact H4 result, exact H5 result, JUnit totals, atomic artifact path, and validated `.verification/h4-h5-<FULL_HEAD>-ledger.json` reported from current evidence.
