# VFE 4.0 H4 Cost and H5 Update-Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a preregistered empirical H4 comparison of independent information- and moment-form Gaussian solvers and a separate deterministic H5 verification gate that proves every labeled update evaluates the complete affected objective and obeys its label-specific acceptance or rollback contract.

**Architecture:** H4 builds one immutable neutral Gaussian problem per fixed seed, materializes its raw factors once into one identity-stable owned tensor object, passes that same materialized object and protocol to two genuinely independent production solvers, verifies terminal-law equivalence outside the timer, and uses seed-level paired timing ratios for inference. H5 keeps H2's detached immutable evaluation seam intact, constructs gradient proposals in a separate differentiable working representation, freezes each candidate before complete-objective evaluation, and records a closed update taxonomy plus factor-dependency and rollback evidence. The unified click-run publishes distinct `validation/h4.json` and `validation/h5.json` payloads and distinct `GateResult` objects, even though one exact revision, one JUnit run, one click artifact, and one revision-specific claim ledger form the coupled H4/H5 milestone.

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
- Both H4 arms consume the same immutable raw-only `H4MaterializedProblem` object by identity, together with the same initial law, objective definition, factor schedule, stopping rule, dtype, CPU device, thread setting, and process environment. The oracle alone consumes canonical neutral-problem bytes. Arm order is deterministic AB/BA. An arm must not regenerate, rematerialize, reorder, mutate, or cache a different problem.
- The information arm assembles and solves in information coordinates. The moment arm constructs and updates a joint moment law directly through affine-Gaussian propagation and conditioning. The moment arm must not call the information solver, canonical assembler, `InformationGaussian`, or obtain a covariance by inverting the information arm's final precision.
- H4 performs common validation, tensor materialization, exact-oracle construction, CPU/thread/affinity inspection, factor hashing, and condition diagnostics outside the timer. Do not time H2's diagnostic `eigvalsh`, and do not run an unbalanced diagnostic in only one arm.
- The timed region begins immediately before construction of fresh arm-native solver state and ends after the arm exhausts the identical one-pass factor schedule, performs its arm-native finite/SPD checks, and evaluates the common objective in its native representation. It includes arm-native factor assembly/propagation, solves/factorizations, and objective evaluation. It excludes problem generation, exact-oracle work, hashing, condition-envelope checks, conversion of either native terminal law into the common H4 comparison record, selected-moment extraction for equivalence, garbage collection setup, artifact serialization, bootstrap statistics, and diagnostic memory passes. The information arm therefore is not rewarded for already storing `J`, and the moment arm is not penalized by timing a comparison-only conversion to `J`.
- Freeze scaled-problem traversal as `for horizon_index, horizon in enumerate((7,15,31))`, then `for seed_index, seed in enumerate(H4_PROBLEM_SEEDS)`, then `for kind_index, kind in enumerate(("coupled","zero_control"))`; assign zero-based `problem_index` in exactly that order. For each problem, run all three untimed warmup pairs and then all 11 timed pairs. Pair `pair_index` uses information-then-moment (AB) exactly when `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, otherwise moment-then-information (BA), where warmups use pair indices `0,1,2` and timed repetitions use `3+repetition_index`. Warmups verify execution only and never count toward timed or inferential order balance. For the primary `horizon_index=2`, `kind_index=0` (`D=256`, coupled) endpoint, odd `seed_index` values have exactly six AB and five BA timed pairs, even `seed_index` values have exactly five AB and six BA timed pairs: ten seeds of each pattern and exactly 110 AB plus 110 BA primary timed pairs in aggregate. Retain every raw nanosecond timing and its exact order. Repetitions are not inferential units.
- CPU float64 and one PyTorch intra-op thread are mandatory for H4. Record PyTorch inter-op threads, NumPy/PyTorch BLAS configuration, relevant thread environment variables, processor identity, OS, process affinity, clock name/resolution, power-policy fields when available, and observed thread counts. Missing mandatory CPU/float64/one-thread facts makes H4 `INCONCLUSIVE`; do not silently substitute a different environment.
- H4's per-seed primary statistic is `median(11 information times) / median(11 moment times)` for that seed's `D=256` coupled problem. The aggregate estimate is `exp(mean(log(seed_ratio)))`. Compute a deterministic 95% paired percentile-bootstrap interval by resampling the 20 seed-level log ratios with replacement for exactly 100,000 replicates using frozen bootstrap seed `20260721`, then exponentiating the 2.5th and 97.5th percentiles. Seeds, not repetitions, are the inferential units.
- H4 support threshold: if terminal equivalence passes and the upper 95% bound is below or equal to `0.80` while the interval is not the exact boundary point `[0.80,0.80]`, H4 is `PASS`. If equivalence passes and the lower bound is greater than or equal to `0.80` while the interval is not that exact boundary point, H4 is `FAIL` (no supported benefit). An interval crossing `0.80`, including the exact-boundary case, is `INCONCLUSIVE` with an explicit precision obligation.
- H4 equivalence is a prerequisite on exact-posterior gap, terminal information vector `h`, precision `J`, selected means/covariance blocks, and complete objective. Every comparison uses its own operand-shaped allowance. A finite decisive miss is `FAIL`; missing/nonfinite output, an indecisive allowance, an environment/protocol mismatch, or an incomplete repetition table is `INCONCLUSIVE`.
- Freeze the H4 scaled-suite admissibility envelope at `lambda_min(J) >= 1e-6`, `lambda_max(J) <= 1e6`, `kappa_2(J) <= 1e8`, minimum Cholesky pivot `>= 1e-3`, `||mu||_inf <= 16`, and every moment-arm innovation covariance satisfying the same eigenvalue/condition bounds at its local dimension. Bounds are inclusive. Any problem, exact oracle, or terminal arm outside the envelope is `INCONCLUSIVE`; do not jitter, clip, pseudo-invert, repair, or silently omit it.
- Freeze `H4_SOLVER_RELATIVE_BUDGET=1e-9` and `H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION=1e-4`. Each solver-produced operand contributes exactly `1e-9 * invariant_scale` once; oracle operands contribute no solver term. Each comparison records `invariant_scale=max(1, every compared scalar absolute value or vector/matrix infinity norm)`, its rounding and solver contributions, final allowance, and `allowance/invariant_scale`. Eligibility requires the ratio to be strictly less than `1e-4`; equality or a larger ratio is `INCONCLUSIVE`. No invariant borrows another invariant's scale or condition number.
- H4 retains raw times. Peak memory and real-operation counts are secondary and are collected in separate untimed diagnostic passes using the same arm wrappers. They cannot rescue or overturn the primary timing decision and are not H8 sparse-allocation evidence.
- Instrument real operations symmetrically through one shared `InstrumentedLinearAlgebra` facade used by both arms. A `NullOperationRecorder` is used in timed runs and a `CountingOperationRecorder` in untimed diagnostic runs. The recorder may observe an operation only inside the wrapper that actually executes it; no solver may emit estimated or formula-derived counts as if they were runtime operations.
- H5 v1 uses horizon `T=2`, the exact raw `h1-v1` fixture whose full SHA-256 is `388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b`, and one separately tracked H5 update-spec fixture.
- The recognition law is exactly
  \[
  Q_{\mathrm{H5}}=\prod_{t=0}^{2}q_t^z(z_t)q_t^m(m_t)
  \prod_{t=1}^{2}\gamma_t(b_t)\beta_t(a_t\mid b_t).
  \]
  It is continuous mean-field with conditional categorical state-source rows. It must never be described as fully factorized categorical recognition.
- Every continuous factor is source-independent. In the equivalent H1 structured record, every model kernel has slope zero and identical offset/variance across all `b` slots; every state kernel has both slopes zero and identical offset/variance across all `(a,b)` slots. Equality includes offsets and variances, not slopes alone.
- The equivalent H1 reconstruction is exact: its initial mean is `(mu_z0,mu_m0)` with diagonal covariance `diag(V_z0,V_m0)`; at each `t in (1,2)`, every model-source slot is `Normal(offset=mu_mt, slope=0, variance=V_mt)` and every state-source `(a,b)` slot is `Normal(offset=mu_zt, z_slope=0, m_slope=0, variance=V_zt)`. The parser constructs every repeated slot from the one corresponding H5 coordinate, then validates literal equality of every repeated offset and variance.
- The exact initial continuous recognition values are, in order `(z0,m0,z1,m1,z2,m2)`, means `(-0.10,0.25,0.05,0.175,-0.04,0.14)` and variances `(0.65,0.78,0.96,1.21,0.90,1.40)`.
- The exact categorical values are `gamma_1=(1)`, `beta_1(.|b1=0)=(1)`, `gamma_2=(0.4,0.6)`, `beta_2(.|b2=0)=(0.75,0.25)`, and `beta_2(.|b2=1)=(0.2,0.8)`. Every listed probability is finite, nonnegative, normalized, and positive on its declared support.
- `q[source_row_a2]` means only `beta_2(.|b2=0)` on ordered support `(0,1)`. No alias names are accepted.
- The exact mutable model-block universe is `theta[state_transition_2]`, `theta[emission_1]`, and `theta[shared_decoder_transition]`. All other H1 model values are frozen reference inputs.
- `theta[state_transition_2]` stores the full scalar-Gaussian base block `(alpha_0,alpha_1,B_base,c,R)`, initially `(0.8,0.64,-0.35,0.08,0.48)`; the effective normalized-factor block is `(alpha_0,alpha_1,B_effective=B_base+s,c,R)`. H5 promotes these copied transition coefficients to independently mutable parameters; an accepted H5 candidate does not claim to remain derived from the frozen H1 frame object.
- `theta[emission_1]` initially owns `w_z=(0.2,-0.4,0.1)`, `w_m=(0.3,0.2,-0.5)`, and `bias=(0.05,-0.1,0.15)`.
- `theta[shared_decoder_transition]` is one scalar `s`, initially `0.0`. Reconstruction uses `B_effective=B_base+s`, `emission[1].w_z[0]=w_z_base[0]+s`, and `emission[2].w_z[0]=-0.1+s`. The remaining emission-2 values stay frozen from H1. This one primitive block, not accidental tensor aliasing, creates the declared three-factor dependency.
- H5 v1 is deterministic CPU float64. Quadrature orders are exactly `21` and `17`; stochastic contribution is exactly zero.
- The initial optimizer state is `FrozenByteState("h5-no-optimizer-v1", b'{"kind":"none"}')`. The initial RNG state is `FrozenByteState("h5-deterministic-rng-v1", b'{"algorithm":"none","counter":0}')`; production H5 v1 never advances it. The rejection-mutation control changes only its test-local counter to one so rollback hashing has a declared target.
- The existing `ElboTerms.complete_elbo` is the sole scalar objective. Raw expected-log-factor values plus recognition entropy and the KL-partitioned `ElboTerms` are two reconstructions of that scalar, never two objectives.
- Accepted state changes only by whole `RecognitionSnapshot` and `H5ModelSnapshot` replacement inside `execute_update`. No accepted snapshot contains a live tensor, autograd graph, optimizer method, or alias of H1/H2 records.
- `valid_mm` remains in `UpdateLabel` but has no H5 v1 producer. A request for it is rejected during configuration resolution unless a revision-bound proof artifact is present. Absence of an MM request is irrelevant to H5 gate status; H5 attempt and gate code have no missing-MM-proof branch.
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
| `vfe4/inference/h4_solvers.py` | Raw-only materialization, exact inference-layer runtime records, independent information/direct-moment solvers, common conversion, and null-bound native diagnostic replay behind one protocol. |
| `verification/numpy_oracles/h4_gaussian.py` | NumPy-only exact posterior/objective from immutable neutral factors; no production solver imports. |
| `verification/h4_budget.py` | Operand-shaped terminal equivalence allowances only. |
| `verification/h4_statistics.py` | Primary per-seed/aggregate timed-order balance, seed-level medians, geometric mean, fixed paired bootstrap, and three-way threshold decision. |
| `verification/h4_gate.py` | Preflight, correctness anchor, scaled equivalence, independently indexed timed AB/BA harness, balance gate, diagnostics, status mapping, and H4 payload. |
| `.gitattributes` | Freezes `h5_conditional_update_v1.json` as raw non-text bytes so Git cannot rewrite the pinned fixture. |
| `vfe4/types/updates.py` | Closed H5 labels/rules, immutable recognition/model/reference/live/candidate/request records, transaction-outcome union, and canonical state hashes. |
| `vfe4/types/h5_schema.py` | Dependency-neutral identifier universes, reconstruction records, term signs, hash domains, operation-count tables, and objective/factor-input schema hashes. |
| `vfe4/validation/fixtures/h5_conditional_update_v1.json` | Exact conditional-categorical H5 update specification with pinned raw bytes. |
| `vfe4/validation/h5_update_spec.py` | Full-digest-first strict parser, canonical specification encoding, exact H1 reconstruction, and reference-state builder. |
| `vfe4/objective/dependency_graph.py` | Closed variable/parameter-to-factor graph and exact ordered affected-factor calculation. |
| `vfe4/objective/h5_complete.py` | One authoritative complete `ElboTerms` evaluation with raw factor trace, factor-input hashes, and fail-closed cache provenance. |
| `vfe4/numerics/h5_budget.py` | Production-owned term, complete-objective, delta, and exact-candidate operand-shaped numerical allowances. |
| `vfe4/inference/h5_updates.py` | Differentiable proposal construction, exact/source/M/GEM/natural operations, freeze-before-evaluate, acceptance, and transactional rollback. |
| `verification/numpy_oracles/h5_updates.py` | NumPy-only independent exact E/source/M candidates and complete-objective deltas from raw captured bytes. |
| `verification/h5_gate.py` | Five positive cases, seven adversarial controls, independent candidate comparison, status mapping, and byte-bound H5 payload. |
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
H4ProblemSource = Literal["scaled_pcg64", "h3_anchor"]
H4PairOrder = Literal["information_then_moment", "moment_then_information"]
H4FactorRole = Literal["initial", "transition", "observation"]
H4OperationKind = Literal[
    "cholesky", "triangular_solve", "matrix_multiply",
    "symmetric_rank_update", "selected_block_extract",
]
H4JsonScalar: TypeAlias = str | int | float | bool | None
H4JsonValue: TypeAlias = H4JsonScalar | tuple["H4JsonValue", ...] | Mapping[str, "H4JsonValue"]
H4JsonMapping: TypeAlias = Mapping[str, H4JsonValue]
H4MeasurementName = Literal[
    "primary_seed_ratio_geometric_mean", "primary_bootstrap_lower",
    "primary_bootstrap_upper", "primary_effect_threshold",
    "primary_timed_ab_total", "primary_timed_ba_total",
    "maximum_solver_stopping_residual", "maximum_allowance_scale_fraction",
]
H4AllowanceInvariantName = Literal[
    "h3_anchor_identity", "exact_posterior_gap_equivalence",
    "terminal_h_equivalence", "terminal_J_equivalence",
    "selected_moment_equivalence", "complete_objective_equivalence",
]
H4_INVARIANT_NAMES = (
    "h3_anchor_identity",
    "fixed_seed_problem_identity",
    "coupled_zero_control_contract",
    "cpu_float64_one_thread",
    "shared_protocol_identity",
    "scaled_condition_envelope",
    "complete_repetition_table",
    "primary_timed_order_balance",
    "exact_posterior_gap_equivalence",
    "terminal_h_equivalence",
    "terminal_J_equivalence",
    "selected_moment_equivalence",
    "complete_objective_equivalence",
    "all_equivalence_allowances_decisive",
    "real_operation_instrumentation",
    "primary_seed_level_inference",
    "primary_effect_threshold",
)
H4_MEASUREMENT_NAMES = (
    "primary_seed_ratio_geometric_mean",
    "primary_bootstrap_lower",
    "primary_bootstrap_upper",
    "primary_effect_threshold",
    "primary_timed_ab_total",
    "primary_timed_ba_total",
    "maximum_solver_stopping_residual",
    "maximum_allowance_scale_fraction",
)
H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL = (
    "primary_seed_ratio_geometric_mean",
    "primary_bootstrap_lower",
    "primary_bootstrap_upper",
    "primary_timed_ab_total",
    "primary_timed_ba_total",
)
H4_ALLOWANCE_INVARIANT_NAMES = (
    "h3_anchor_identity",
    "exact_posterior_gap_equivalence",
    "terminal_h_equivalence",
    "terminal_J_equivalence",
    "selected_moment_equivalence",
    "complete_objective_equivalence",
)

@dataclass(frozen=True)
class H4RawDraw:
    draw_index: int
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

@dataclass(frozen=True)
class H4AffineGaussianFactor:
    factor_id: str
    role: H4FactorRole
    time_index: int
    normalized_coordinate_indices: tuple[int, ...]
    parent_coordinate_indices: tuple[int, ...]
    matrix: tuple[tuple[float, ...], ...]
    target: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    raw_draws: tuple[H4RawDraw, ...]

@dataclass(frozen=True)
class H4NeutralProblem:
    problem_id: str
    source_kind: H4ProblemSource
    seed: int
    kind: H4ProblemKind
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    factor_schedule: tuple[H4AffineGaussianFactor, ...]
    canonical_sha256: str

@dataclass(frozen=True)
class H4SolveProtocol:
    protocol_id: Literal["h4-single-pass-v1"]
    dtype: Literal["float64"]
    device: Literal["cpu"]
    factor_passes: Literal[1]
    solver_relative_budget: float
    stopping_rule: Literal["complete_schedule_finite_spd"]

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
class H4SolverResult:
    problem_id: str
    problem_sha256: str
    arm: H4SolverArm
    protocol_id: Literal["h4-single-pass-v1"]
    factor_count: int
    native_information: H4NativeInformationState | None
    native_moment: H4NativeMomentState | None

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

@dataclass(frozen=True)
class H4OperationRecord:
    problem_id: str
    arm: H4SolverArm
    operation: H4OperationKind
    operand_shapes: tuple[tuple[int, ...], ...]
    result_shape: tuple[int, ...]
    count: int

@dataclass(frozen=True)
class H4MemoryRecord:
    problem_id: str
    arm: H4SolverArm
    python_peak_bytes: int | None
    process_working_set_delta_bytes: int | None
    unavailable_fields: tuple[
        Literal["python_peak_bytes", "process_working_set_delta_bytes"], ...
    ]

H4AllowanceSentinelReason = Literal[
    "not_evaluated_after_decisive_h3_anchor_failure",
    "not_evaluated_after_inconclusive_eligibility",
]

@dataclass(frozen=True, slots=True)
class H4AllowanceOperationCount:
    label: str
    count: int

@dataclass(frozen=True, slots=True)
class H4AllowanceOperand:
    label: str
    value: float
    value_norm: float
    absolute_summand_accumulation: float
    condition_numbers: tuple[float, ...]
    operation_counts: tuple[H4AllowanceOperationCount, ...]
    solver_produced: bool
    rounding_allowance: float
    solver_allowance: float
    total_allowance: float

@dataclass(frozen=True, slots=True)
class H4AllowanceElement:
    stream_index: int
    invariant: H4AllowanceInvariantName
    problem_id: str
    comparison_source: Literal[
        "solver_to_oracle",
        "adapter_to_h3_reference",
        "adapter_to_oracle",
    ]
    repetition_index: int | None
    arm: H4SolverArm | None
    path: str
    shape: tuple[int, ...]
    flat_index: int
    invariant_scale: float
    left: H4AllowanceOperand
    right: H4AllowanceOperand
    comparison_reduction_allowance: float
    residual: float
    normalized_residual: float
    final_allowance: float
    allowance_scale_ratio: float
    decisive: bool
    passed: bool

@dataclass(frozen=True, slots=True)
class H4ApplicableAllowance:
    applicable: Literal[True]
    invariant: H4AllowanceInvariantName
    element_stream_domain: Literal["vfe4.h4.allowance-element-stream.v1"]
    expected_element_count: int
    observed_element_count: int
    element_stream_sha256: str
    maximum_normalized_residual: float
    maximum_normalized_residual_element: H4AllowanceElement
    maximum_allowance_scale_ratio: float
    maximum_allowance_scale_ratio_element: H4AllowanceElement
    first_failed_element: H4AllowanceElement | None
    first_indecisive_element: H4AllowanceElement | None
    decisive: bool
    passed: bool

@dataclass(frozen=True, slots=True)
class H4InapplicableAllowance:
    applicable: Literal[False]
    reason: H4AllowanceSentinelReason

H4AllowanceRecord: TypeAlias = H4ApplicableAllowance | H4InapplicableAllowance

H4_ALLOWANCE_ELEMENT_COUNTS = (
    ("h3_anchor_identity", 184),
    ("exact_posterior_gap_equivalence", 2640),
    ("terminal_h_equivalence", 394240),
    ("terminal_J_equivalence", 75694080),
    ("selected_moment_equivalence", 3738240),
    ("complete_objective_equivalence", 2640),
)

@dataclass(frozen=True, slots=True)
class H4IntervalDecision:
    lower: float
    upper: float
    threshold: float
    classification: H4IntervalClass
    invariant_passed: bool
    invariant_value: float
    invariant_limit: float
    invariant_detail: Literal[
        "bootstrap_interval_supports_effect",
        "bootstrap_interval_excludes_support",
        "bootstrap_interval_crosses_threshold",
        "bootstrap_interval_equals_threshold",
    ]
    status_if_other_invariants_eligible: GateStatus
    obligation: str | None

@dataclass(frozen=True)
class H4GateResult:
    gate: Literal["H4"]
    status: GateStatus
    measurements: Mapping[H4MeasurementName, float | None]
    invariants: tuple[InvariantResult, ...]
    allowances_by_invariant: Mapping[H4AllowanceInvariantName, H4AllowanceRecord]
    obligations: tuple[str, ...]

def canonical_h4_problem_bytes(problem: H4NeutralProblem) -> bytes: ...

# vfe4/inference/h4_solvers.py
@dataclass(frozen=True, slots=True)
class H4MaterializedProblem:
    materialization_version: Literal["h4-materialized-problem-v1"]
    problem_id: str
    problem_sha256: str
    protocol_id: Literal["h4-single-pass-v1"]
    dtype: Literal["float64"]
    device: Literal["cpu"]
    source_kind: H4ProblemSource
    seed: int
    kind: H4ProblemKind
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    factor_ids: tuple[str, ...]
    factor_roles: tuple[H4FactorRole, ...]
    factor_time_indices: tuple[int, ...]
    factor_normalized_coordinate_indices: tuple[tuple[int, ...], ...]
    factor_parent_coordinate_indices: tuple[tuple[int, ...], ...]
    _factor_matrices: tuple[Tensor, ...]
    _factor_targets: tuple[Tensor, ...]
    _factor_covariances: tuple[Tensor, ...]
    tensor_sha256: str

@dataclass(frozen=True, slots=True)
class H4InnovationDiagnostic:
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    innovation_dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    minimum_cholesky_pivot: float

@dataclass(frozen=True, slots=True)
class H4NativeDiagnostics:
    problem_id: str
    problem_sha256: str
    protocol_id: Literal["h4-single-pass-v1"]
    arm: H4SolverArm
    factor_count: int
    replayed_result: H4SolverResult
    innovation_diagnostics: tuple[H4InnovationDiagnostic, ...]
    finite: Literal[True]
    spd: Literal[True]
    replay_matches_result: Literal[True]

def materialize_h4_problem(
    problem: H4NeutralProblem,
    protocol: H4SolveProtocol,
) -> H4MaterializedProblem: ...

class H4GaussianSolver(Protocol):
    def solve(
        self,
        materialized: H4MaterializedProblem,
        protocol: H4SolveProtocol,
        linalg: InstrumentedLinearAlgebra,
    ) -> H4SolverResult: ...

def solve_information_form(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    linalg: InstrumentedLinearAlgebra,
) -> H4SolverResult: ...

def solve_moment_form(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    linalg: InstrumentedLinearAlgebra,
) -> H4SolverResult: ...

def to_common_terminal_law(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> H4TerminalLaw: ...

def evaluate_h4_native_diagnostics(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> H4NativeDiagnostics: ...

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
def classify_h4_interval(lower: float, upper: float) -> H4IntervalDecision: ...
def decide_h4_interval(interval: H4BootstrapInterval) -> H4IntervalDecision: ...
```

### Public H4 allowance and interval contract

These records remain dependency-light in `vfe4/types/h4.py` and are re-exported
from `vfe4/types/__init__.py`. Remove the flat
`H4_APPLICABLE_ALLOWANCE_FIELDS` schema and make
`H4GateResult.allowances_by_invariant` contain exact typed records rather than
arbitrary recursive JSON mappings.



`H4AllowanceElement` remains the exact nested element-local proof schema, but
the artifact never retains the roughly 80 million-element stream. Every path
is nonempty and unique in the canonical stream, every shape is positive, and
`flat_index` is in row-major range. Each element's local `invariant_scale` is
defined independently as
`max(1, abs(left.value), left.value_norm, abs(right.value),
right.value_norm)`; no global maximum is pooled back into another element's
budget. Constructors recompute all derived values rather than trusting
caller-supplied arithmetic.

`comparison_source="solver_to_oracle"` requires a real arm; it requires
repetition `0..10` for scaled problems and `None` for the two anchor problems.
`adapter_to_h3_reference` is allowed only for the 22 coupled-anchor paths and
`adapter_to_oracle` only for the two zero-control `c/logZ` paths; both require
`arm=None` and `repetition_index=None`. Every other source/arm/repetition
combination is rejected, so no non-solver anchor path receives a fake arm.

The exact arithmetic is:

```text
eps = 2.220446049250313e-16
C = 4096
gamma(n) = n*eps/(1-n*eps), with integer n >= 0 and n*eps < 1
n_operand = sum(term.count for term in operation_counts)
kappa = max((1, *condition_numbers))
rounding_allowance = C*gamma(n_operand)*kappa
                     * max(1, value_norm, absolute_summand_accumulation)
solver_allowance = 1e-9*invariant_scale if solver_produced else 0
total_allowance = rounding_allowance + solver_allowance
comparison_reduction_allowance = C*gamma(3)
    * max(1, abs(left.value), abs(right.value),
             abs(left.value)+abs(right.value))
residual = abs(left.value-right.value)
final_allowance = left.total_allowance + right.total_allowance
                  + comparison_reduction_allowance
allowance_scale_ratio = final_allowance/invariant_scale
decisive = allowance_scale_ratio < 1e-4
passed = residual <= final_allowance
normalized_residual = residual/final_allowance
```

`comparison_reduction_allowance` is strictly positive for finite operands, so
`final_allowance` is strictly positive and every stored normalized residual is
finite.

The operation-count table is an ordered tuple of unique labels and exact
nonnegative scalar-operation counts. The budget module owns the closed count
formulas:

```text
dot(k) = max(0, 2*k-1)
matmul(m,k,n) = m*n*dot(k)
triangular_solve(n,r) = r*n*n
cholesky(n) = ceil(n*n*n/3)
reduction(n) = max(0,n-1)
log_terms(n) = n
elementwise(n, operations_per_element) = n*operations_per_element
selected_extract(...) = 0
```

Each operand producer records the table for the arithmetic that produced the
retained comparison value. Native solver arithmetic is covered exactly once by
the solver term; postflight conversion arithmetic is still listed in the
rounding table. Oracle operands have `solver_produced=False`. The literal zero
operand used by the posterior-gap comparison has `value_norm=0`, zero absolute
accumulation, `condition_numbers=(1.0,)`, an empty operation table, and no
solver term. Duplicate, omitted, pooled, or oracle-side solver terms are
rejected.

Composite aggregation never compares `max(residual)` with `max(allowance)`.
Instead a module-private NumPy producer/accumulator consumes read-only
float64 operand groups in canonical order, slices them into at most 4,096
row-major scalar lanes, checks every lane vectorially, updates a
domain-separated stream digest, and retains only deterministic witnesses:

```text
maximum_normalized_residual = max(element.normalized_residual)
maximum_allowance_scale_ratio = max(element.allowance_scale_ratio)
record.passed = all(element.passed)
record.decisive = all(element.decisive)
```

Ties for either maximum choose the lowest `stream_index`. `first_failed_element`
and `first_indecisive_element` are the lowest-index matching elements and are
`None` exactly when `passed` or `decisive` is true. The two maximum witnesses
are always present for a nonempty applicable stream. The accumulator stores
only four private scalar witness candidates and creates
`H4AllowanceElement` dataclasses only once, for the final distinct witnesses;
it never creates a Python object per checked scalar. It rejects the first
path/order/index mismatch and requires
`observed_element_count == expected_element_count`; it never converts the
input iterable or scalar lanes to a tuple or list.

`H4ApplicableAllowance.__post_init__` selects the frozen expected count from
its invariant, requires observed equality and a lowercase 64-hex digest,
requires every witness to name that invariant and an in-range stream index,
requires both maximum fields to equal their witness fields bit-for-bit, and
requires first-failure/indecisive presence to agree with the conjunction
booleans. The module-private accumulator factory additionally owns the live
SHA-256 object and independently generated expected group-header iterator; it
is the only production construction path. Gate/artifact constructors accept
only its completed record and cross-check the six literal counts/order.

The expected complete-run counts are frozen and independently recomputed from
the traversal, repetition, arm, dimension, and selected-block tables:

```text
h3_anchor_identity = 184
exact_posterior_gap_equivalence = 120*11*2 = 2,640
terminal_h_equivalence = 40*11*2*(64+128+256) = 394,240
terminal_J_equivalence = 40*11*2*(64^2+128^2+256^2) = 75,694,080
selected_moment_equivalence
  = 40*11*2*((9+17+33)*8 + (9+17+33)*8^2)
  = 3,738,240
complete_objective_equivalence = 120*11*2 = 2,640
total = 79,832,024
```

The 184 anchor elements comprise four arm-to-oracle comparisons at
`D=4,T=1,B=2` (`KL`, four `h`, sixteen `J`, three selected means of two
scalars, three selected covariances of four scalars, and objective: 40 per
arm result, 160 total), plus the coupled adapter/reference `J(16),h(4),c,
logZ` paths and the zero-control adapter/oracle `c,logZ` paths (24 total).
These paths are an exact frozen tuple, not inferred from record contents.

The stream digest begins with
`b"vfe4.h4.allowance-element-stream.v1\x00"`. Each comparison group then
adds an unsigned eight-byte big-endian length and compact sorted-key UTF-8
header containing invariant,
problem/hash, comparison source, repetition, optional arm, path prefix, shape,
element-count, both ordered
operation tables, condition-number tuples encoded with `float.hex()`, and
solver flags. The header also binds each operand's value norm as `float.hex()`
and, separately for `left` and `right`, the element count and SHA-256 of both
the value vector and the absolute-summand vector. Each vector digest begins
with `b"vfe4.h4.allowance-group-vector.v1\x00"`, the ASCII lane name
(`left_value`, `right_value`, `left_absolute_summand`, or
`right_absolute_summand`), a zero byte, and its unsigned eight-byte
big-endian element count, followed by the exact contiguous little-endian
`<f8` bytes. The expected-header path and observed-group path independently
recompute all four vector digests and both norms; a caller cannot supply them.
Thus equal metadata around different numerical operands is a header mismatch,
not a self-consistent observed stream. Its scalar rows are appended in
row-major order as a packed,
unaligned little-endian structured array with the exact fields
`left_value,right_value,left_value_norm,right_value_norm,left_absolute_sum,
right_absolute_sum,left_rounding,left_solver,right_rounding,right_solver,
comparison_allowance,residual,normalized_residual,final_allowance,
allowance_scale_ratio` as `<f8`, followed by `decisive,passed` as `u1`.
The unaligned row itemsize is exactly 122 bytes. Chunks contain at most 4,096
rows and chunk boundaries do not enter the hash.
All values are checked finite before packing. This makes the digest independent
of Python object repr and prevents a compact artifact from hiding unchecked
scalars.

`H4GateResult` keeps the exact six allowance keys and exact early-failure
sentinels, but applicable values must be `H4ApplicableAllowance` and
inapplicable values must be `H4InapplicableAllowance`. Conclusive post-timing
results require all six applicable records. No mutable mapping or free-form
allowance field survives construction.

The prerequisite also closes one post-`d0a53b1` restoration seam. A decisive
anchor miss remains the exact early `FAIL` only when all process-global state
that was changed has been restored. If the anchor was evaluated but thread or
GC restoration later fails, `H4GateResult` permits exactly one alternative:
`INCONCLUSIVE`, the same five unavailable primary measurements, an applicable
numerical anchor allowance, later allowances using
`not_evaluated_after_inconclusive_eligibility`, and the exact obligation
`restore H4 process-global state before closing anchor result`. This branch
cannot become `PASS` and does not erase the recorded anchor miss. It is the
only exception to the current "anchor miss implies early FAIL" validator and
is required by the rule that no conclusive result is finalized before
successful restoration.

Task 1 also exposes exactly one classifier and one decision record:



The only inequalities are in this public function:

```text
if lower == upper == 0.80: boundary / INCONCLUSIVE
elif upper <= 0.80:        support / PASS
elif lower >= 0.80:        no_support / FAIL
else:                      crossing / INCONCLUSIVE
```

The decision freezes the exact invariant value/limit/detail already required
by `H4GateResult`; crossing and boundary name the precision obligation and the
other two use `obligation=None`. `H4GateResult.__post_init__` calls this public
function. The former private classifier is removed. Task 3's convenience
`decide_h4_interval(interval)` must only delegate to
`classify_h4_interval(interval.lower, interval.upper)`.

Finally, Task 1 exports one authoritative `H4_PRIMARY_TIMED_BALANCE` tuple,
copied exactly from the preregistration, together with totals `110/110`. Config,
statistics, and the gate import it; none owns another hand-written classifier
or primary-balance universe.

`H4NeutralProblem.factor_schedule` is the authoritative ordered generic normalized-factor schedule. Any initial, transition, or observation partition is a validated derived view over that one tuple and is never an independent canonical source of coefficients, offsets, covariances, IDs, or order. Availability in this ordered schedule, rather than numeric index order or a factor ID/name, determines the schedule validation.

`H4RawDraw` has a nonnegative zero-based `draw_index` first, a nonempty name, a tuple of nonnegative integer dimensions, row-major finite float64-representable values, and `product(shape) == len(values)` (with scalar shape `()` having product one). Within a factor, raw draws are strictly increasing by draw index and have unique names; over a scaled problem their draw indices are globally unique. H3 anchors have `raw_draws=()`. For each scaled time `t`, the eleven names are exactly `A_m[t]`, `A_z[t]`, `B[t]`, `c_m[t]`, `c_z[t]`, `R_m[t]`, `R_z[t]`, `G[t]`, `observation_offset[t]`, `observation_noise[t]`, `observed_target[t]`, in that order, with `draw_index=11*(t-1)+local_index` for zero-based `local_index`. Each factor's raw-draw tuple remains increasing even though those indices interleave across factors.

`H4AffineGaussianFactor` requires a nonempty unique factor ID, a nonnegative time index, finite tuple-owned values, matrix `A` of exact shape `d x D`, target `b` of shape `d`, and exactly symmetric positive-definite covariance `R` of shape `d x d`; it encodes `A y-b` and stores no normalizer, because the normalizer is derived only from `d` and `R`. Metadata tuples are ordered, unique, disjoint, and in `[0,D)`. Initial and transition normalized-coordinate tuples have length `d`, while observation normalized coordinates are empty. Initial parent metadata is empty; transition and observation parent metadata is explicit. Initial/transition factors require `A[:, normalized_coordinate_indices] = I_d` and support only normalized and parent columns. Observation factors use local latent coordinates as parents and support only parent columns. Initial normalized indices are ascending `z_0` components then ascending `m_0` components. For every scaled `t`, `m_transition[t]` has ascending `m_t` normalized indices and ascending `m_{t-1}` parents; `z_transition[t]` has ascending `z_t` normalized indices and parents `z_{t-1}` then `m_t`, each ascending; `observation[t]` has no normalized indices and parents `z_t` then `m_t`, each ascending. This metadata is schedule-causal even though storage puts `z_t` before `m_t`. For scaled controls, zero every designated transition-parent column, not merely a parent coefficient that happens to be nonzero.

Scaled coordinates are exactly `z[t,i]` for `i=0..3`, then `m[t,i]` for `i=0..3`, for each `t=0..T`, in storage order. Scaled factor IDs are exactly `initial_joint`, then `m_transition[t]`, `z_transition[t]`, and `observation[t]` for each ascending `t`; H3 IDs remain unchanged. A scaled problem ID is exactly `h4-{kind}-T{horizon}-dz4-dm4-seed{seed}-v1`; an H3 anchor ID is exactly `h4-anchor-{fixture.fixture_id}`. `H4NeutralProblem.source_kind` is exact and never inferred from an ID: `scaled_pcg64` validates a requested positive PCG64 seed, supported kind, `horizon in {7,15,31}`, `d_z=d_m=4`, `dimension=(horizon+1)*(d_z+d_m)`, the scaled coordinate tuple, and the exact scaled factor-ID schedule; `h3_anchor` validates its exact anchor ID together with `seed=0`, and retains public `('z0','m0','z1','m1')` spelling, IDs, and H3 dimensions rather than being rewritten as scaled. Both source-specific seed values are included in core serialization/digest validation, and both sources require a nonempty schedule with unique IDs and a lowercase 64-hex-character `canonical_sha256`.

`H4SolveProtocol` accepts only `protocol_id="h4-single-pass-v1"`, `dtype="float64"`, `device="cpu"`, `factor_passes=1`, `solver_relative_budget=1e-9`, and `stopping_rule="complete_schedule_finite_spd"`. `H4SolverResult` requires its nonempty problem ID, lowercase 64-hex problem hash, one of the two arms, the frozen protocol ID, and a factor count equal to the consumed schedule count; exactly one native state is non-`None`, and it must be the state matching the declared arm. `H4OperationRecord` admits only the listed operation literal, has a positive count, and has only positive operand/result dimensions; records are aggregates of real shared-facade calls, never estimates. `H4MemoryRecord.unavailable_fields` is an ordered duplicate-free subset of exactly `("python_peak_bytes", "process_working_set_delta_bytes")`; each metric is `None` if and only if named unavailable. Python peak is nonnegative, whereas process working-set delta may be signed.

`H4GateResult` follows the existing fail-closed H3 record semantics with exact H4 schemas: invariant names must equal `H4_INVARIANT_NAMES` in that order; measurement mapping keys must equal `H4_MEASUREMENT_NAMES` in that order; and allowance mapping keys must equal `H4_ALLOWANCE_INVARIANT_NAMES` in that order. `measurements` and applicable allowance records contain only recursive owned immutable H4 JSON: scalars are `str|int|float|bool|None`, values may be immutable tuples or owned immutable string-key mappings, and every float is finite. Construction rejects lists, mutable mappings after ownership conversion, empty or non-string mapping keys, duplicate keys, and nonfinite floats.

`PASS` and every `FAIL` reached after completed timed statistics require all eight `H4_MEASUREMENT_NAMES` values to be finite. A decisive finite pre-timing `FAIL` whose sole decisive cause is `h3_anchor_identity` instead requires exactly `H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL` to be `None`; `primary_effect_threshold` is finite exactly `0.80`, and `maximum_solver_stopping_residual` plus `maximum_allowance_scale_fraction` are finite values from the anchor comparison. No timing or statistic value may be fabricated. Every later unevaluated invariant has exactly `value=None`, `limit=None`, `passed=False`, and detail `not_evaluated_after_decisive_h3_anchor_failure`. `INCONCLUSIVE` may use `None` only for a measurement whose producing phase did not complete; its associated invariant/detail and a named obligation must identify that phase. Finite-but-unavailable fabrication is rejected.

The allowance outer key universe remains exact even when a comparison is inapplicable. An inapplicable allowance record is exactly `{"applicable": False, "reason": <closed reason>}` with closed reasons `not_evaluated_after_decisive_h3_anchor_failure` and `not_evaluated_after_inconclusive_eligibility`; missing/extra sentinel keys or an inapplicable record carrying numeric values is rejected. For a decisive H3-anchor failure, `h3_anchor_identity` remains applicable and numerical, and every other `H4_ALLOWANCE_INVARIANT_NAMES` record uses the first sentinel. Applicable allowance records state `applicable=True` and carry their Task 3 numerical fields. Gate is exactly `"H4"`, and obligations are an immutable duplicate-free string tuple.

Canonical problem bytes are exact. Let `core` be the ordered public `H4NeutralProblem` content excluding `canonical_sha256`, recursively represented by compact finite JSON objects/arrays and including every factor field and raw draw. Compute `digest = SHA256(b"vfe4.h4.neutral-problem.v1\\x00" + compact UTF-8 sorted-key finite JSON(core))`. The published bytes returned by `canonical_h4_problem_bytes` are compact UTF-8 sorted-key finite JSON of `{"schema_version":"h4-neutral-problem-v1","canonical_sha256":digest,"problem":core}`. Parsing first requires the exact envelope schema literal, then recomputes the domain-separated core digest before accepting it. Embedded `canonical_sha256` is that core digest, not `SHA256` of full envelope bytes. The envelope is not self-hashed, and reserialization cannot substitute for core-digest validation.

For a matched coupled/control pair with `source_kind="scaled_pcg64"`, allowed differences are only `kind`, `problem_id`, `canonical_sha256`, and designated transition-parent matrix columns. Raw draws, targets, covariances, roles, time indices, normalized metadata, parent metadata, factor IDs/order, seed, horizon, shape, and every other factor field are identical; parent metadata remains identical when its designated coefficient columns are zero. This invariant does not apply to H3 anchors: the independently authored H3 zero fixture changes observation targets. The adapter maps each H3 scalar structurally as `matrix=(row,)`, `target=(target,)`, and `covariance=((variance,),)`. It derives role/time/normalized/parent metadata from the public `initial_factors`, `transition_factors`, and `observation_factors` group and declared position only, never IDs/names. It reproduces derived scalar normalizers as `-.5*log(2*pi*variance)`. For every H4 schedule, assemble `J=sum(A^T R^-1 A)`, `h=sum(A^T R^-1 b)`, and `c=-.5 sum(b^T R^-1 b + d*log(2*pi) + logdet(R))`. The coupled anchor compares its frozen reference log evidence; the zero H3 fixture has no frozen reference log evidence and compares independently derived adapter/oracle `c/logZ` only, never a frozen-reference logZ.

## H5 Authoritative Identifier Universes

```python
H5_FACTOR_UNIVERSE = (
    "initial_joint",
    "model_source[1]", "model_transition[1]", "state_source[1]",
    "state_transition[1]", "emission[1]",
    "model_source[2]", "model_transition[2]", "state_source[2]",
    "state_transition[2]", "emission[2]", "recognition_entropy",
)

H5_RECOGNITION_COORDINATE_UNIVERSE = (
    "q[z0]", "q[m0]", "q[z1]", "q[m1]", "q[z2]", "q[m2]",
    "q[model_source_b1]", "q[state_source_a1_b0]",
    "q[model_source_b2]", "q[source_row_a2]", "q[state_source_a2_b1]",
)

H5_MODEL_BLOCK_UNIVERSE = (
    "theta[state_transition_2]",
    "theta[emission_1]",
    "theta[shared_decoder_transition]",
)

H5_SIGNED_TERM_IDS = (
    "expected_log_emission[1]", "expected_log_emission[2]",
    "initial_model_kl", "initial_state_kl",
    "model_source_kl[1]", "model_source_kl[2]",
    "model_transition_kl[1]", "model_transition_kl[2]",
    "state_source_kl[1]", "state_source_kl[2]",
    "state_transition_kl[1]", "state_transition_kl[2]",
)
H5_SIGNED_TERM_SIGNS = (+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
H5_DIAGNOSTIC_TERM_IDS = ("joint_recognition_entropy",)
H5_DERIVED_TERM_IDS = ("complete_elbo",)
H5_NONCLAIM_IDS = (
    "no_h4_cost_claim",
    "no_h6_prediction_claim",
    "no_h7_scaling_claim",
    "no_h8_training_or_readiness_claim",
)
```

`ElboTerms.allowances` is metadata, not an evaluated term. `joint_recognition_entropy` is evaluated and budgeted diagnostically but is not added again to the KL-partitioned complete scalar. `complete_elbo` is derived once from `H5_SIGNED_TERM_IDS`.

## H5 Closed Dependency Graph

```text
q[z0] -> initial_joint, state_transition[1], state_transition[2], recognition_entropy
q[m0] -> initial_joint, model_transition[1], model_transition[2], recognition_entropy
q[z1] -> state_transition[1], emission[1], state_transition[2], recognition_entropy
q[m1] -> model_transition[1], state_transition[1], emission[1], model_transition[2], recognition_entropy
q[z2] -> state_transition[2], emission[2], recognition_entropy
q[m2] -> model_transition[2], state_transition[2], emission[2], recognition_entropy
q[model_source_b1] -> model_source[1], model_transition[1], state_source[1], state_transition[1], recognition_entropy
q[state_source_a1_b0] -> state_source[1], state_transition[1], recognition_entropy
q[model_source_b2] -> model_source[2], model_transition[2], state_source[2], state_transition[2], recognition_entropy
q[source_row_a2] -> state_source[2], state_transition[2], recognition_entropy
q[state_source_a2_b1] -> state_source[2], state_transition[2], recognition_entropy
theta[state_transition_2] -> state_transition[2]
theta[emission_1] -> emission[1]
theta[shared_decoder_transition] -> state_transition[2], emission[1], emission[2]
```

Singleton categorical coordinates remain in the universe and graph even though their only valid probability vector is `(1)`. Public update-request resolution rejects attempts to change a singleton coordinate.

## H5 Canonical Encoding and Hash Domains

All runtime snapshot/hash encoders recursively convert each finite binary64 value to `float.hex()`, bytes to `{"length":len(payload),"hex":payload.hex()}`, tuples to arrays, enums to their exact string values, and mappings to exact-schema sorted-key compact UTF-8 JSON with `ensure_ascii=True`, separators `(',', ':')`, and no trailing newline. This distinguishes signed zero and avoids interpreter-dependent decimal formatting. Constructors compute hashes; callers never supply trusted state hashes.

```text
vfe4.h5.update-spec.v1\0
vfe4.h5.update-request.v1\0
vfe4.h5.reference-state.v1\0
vfe4.h5.recognition-snapshot.v1\0
vfe4.h5.model-snapshot.v1\0
vfe4.h5.live-state.v1\0
vfe4.h5.candidate.v1\0
vfe4.h5.semantic-state.v1\0
vfe4.h5.attempt.v1\0
vfe4.h5.transaction.v1\0
vfe4.h5.factor-input-schema.v1\0
vfe4.h5.factor-input.v1\0
vfe4.h5.frozen-complement.v1\0
vfe4.h5.optimizer-state.v1\0
vfe4.h5.rng-state.v1\0
vfe4.h5.objective-schema.v1\0
vfe4.h5.validation-payload.v1\0
```

Task 7 adds the sole prevalidation-only hash domain `vfe4.h5.candidate-draft.v1\0`, owned locally by `vfe4/inference/h5_updates.py`. It is distinct from `vfe4.h5.candidate.v1\0`: a draft hash proves exactly what reached final candidate validation, while `candidate_sha256` continues to identify only a successfully constructed `H5CandidateSnapshot`.

The H5 update-spec parser computes `SHA256(raw_bytes)` before UTF-8 decoding and compares all 64 hexadecimal characters with its pinned literal. It then rejects duplicate keys, unknown/missing fields, nonfinite constants, wrong sequence types/order, aliases, and schema drift. `UpdateSpecification` accepts the already checked immutable raw bytes but no digest/canonical metadata; its constructor recomputes `raw_sha256`, encodes only the decoded semantic fields (never `raw_bytes` or derived hashes) into `canonical_bytes`, and derives the domain-separated canonical hash. `H5ReferenceState` requires byte equality between its update-spec bytes and `specification.raw_bytes`, verifies the exact H1/update IDs, and likewise recomputes all four displayed hashes from its bytes/specification/schema constants instead of accepting them from callers.

The reference-state hash core is exactly `(h1_fixture_sha256, update_spec_raw_sha256, specification.canonical_sha256, objective_schema_sha256, factor_input_schema_sha256, initial_recognition.state_sha256, initial_model.state_sha256, initial_optimizer_state.state_sha256, initial_rng_state.state_sha256)` in that order under the reference-state domain. `initial_live(reference)` returns a new complete `H5LiveState` containing exactly those four initial state objects after defensive reconstruction; its resulting nested hashes equal the four recorded reference hashes.

The request-independent semantic-state digest is `SHA256(semantic-state-domain || uint64_be(len(canonical_recognition_bytes)) || canonical_recognition_bytes || uint64_be(len(canonical_model_bytes)) || canonical_model_bytes)`. Production and the independent oracle both retain it as provenance, but independent NumPy/PyTorch arithmetic is not required to produce bit-identical digests. Candidate correctness uses the frozen fieldwise allowance below. This digest deliberately excludes rule/request/label/active-block/damping provenance: final snapshots protect it with `candidate_sha256` and transaction validation, while Task 7's rejected prevalidation input is protected separately by `candidate_draft_sha256` and can never stand in for a final snapshot.

`factor_input_schema_sha256` is the hash of the factor-input-schema domain followed by canonical JSON of `(H5_FACTOR_UNIVERSE, ("schema_version","factor_id","normalized_factor","observation","recognition_inputs"), reconstruction_records)`. For each factor, `normalized_factor` is its reconstructed effective generative factor after applying shared groups, `observation` is the exact time-labeled observation or `null`, and `recognition_inputs` is the ordered `(coordinate_id, canonical coordinate value)` tuple named by its reconstruction record; `recognition_entropy` instead uses `normalized_factor=null`, `observation=null`, and all recognition coordinates in universe order. `objective_schema_sha256` is the hash of the objective-schema domain followed by canonical JSON of the factor-input schema hash, all three identifier sets and signs, the complete dependency graph, reconstruction/shared-group records, quadrature orders, all three operation-count tables, and literal formula tags `h5-term-budget-v1`, `h5-complete-budget-v1`, `h5-delta-budget-v1`, and `h5-candidate-comparison-v1`.

For one request, the frozen-complement core is exactly `(h1_raw_sha256, update_spec_raw_sha256, objective_schema_sha256, every recognition coordinate not named in request.variables, every model block not named in request.parameters, optimizer_state_sha256, rng_state_sha256)` in the universe orders above. Its domain-separated hash is computed before proposal construction and must be identical in the candidate evaluation. Active blocks never appear in the complement; undeclared blocks cannot be omitted.

## H5 Exact Numerical Budget

```python
H5_EPS = float(np.finfo(np.float64).eps)
H5_C = 4096.0

def gamma_n(n: int) -> float:
    return (n * H5_EPS) / (1.0 - n * H5_EPS)

H5_ANALYTIC_OPERATION_COUNTS = {
    "initial_model_kl": 192,
    "initial_state_kl": 192,
    "model_source_kl[1]": 32,
    "model_source_kl[2]": 64,
    "model_transition_kl[1]": 192,
    "model_transition_kl[2]": 320,
    "state_source_kl[1]": 32,
    "state_source_kl[2]": 96,
    "state_transition_kl[1]": 256,
    "state_transition_kl[2]": 448,
    "joint_recognition_entropy": 320,
}

H5_ANALYTIC_FACTOR_OPERATION_COUNTS = {
    "initial_joint": 256,
    "model_source[1]": 32,
    "model_transition[1]": 192,
    "state_source[1]": 32,
    "state_transition[1]": 256,
    "model_source[2]": 64,
    "model_transition[2]": 320,
    "state_source[2]": 96,
    "state_transition[2]": 448,
    "recognition_entropy": 320,
}

H5_CANDIDATE_COMPARISON_OPERATION_COUNTS = {
    "exact_z0.mean": 512,
    "exact_z0.variance": 512,
    "exact_source_row_a2.probability[0]": 512,
    "exact_source_row_a2.probability[1]": 512,
    "exact_state_transition_2_m.alpha_0": 4096,
    "exact_state_transition_2_m.alpha_1": 4096,
    "exact_state_transition_2_m.B_base": 4096,
    "exact_state_transition_2_m.c": 4096,
    "exact_state_transition_2_m.R": 4096,
}

def emission_operation_count(order: int) -> int:
    if order not in (21, 17):
        raise ValueError("H5 quadrature order must be 21 or 17")
    return 32 * order * order + 8 * order + 32
```

These are frozen conservative scalar-operation upper bounds for the stated evaluator and independent-candidate algebra, not runtime profiler counts. The candidate table applies separately to the production and independent oracle value. Changing either algebra invalidates the table and evidence.

For each term and order `r`:

\[
\rho_r=4096\,\gamma_{N_r}\max(1,\kappa_{r,1},\ldots)
\max(1,|v_r|,S_r),
\]

where `S_r` is that term's actual absolute-summand accumulation and the kappas are only its actual SPD solve/precision condition numbers; terms without an SPD solve use `(1.0,)`. Define

\[
\rho_{21-17}=4096\,\gamma_3
\max(1,|v_{21}|,|v_{17}|,|v_{21}|+|v_{17}|),
\]
\[
A_{\mathrm{term}}=|v_{21}-v_{17}|+\rho_{21}+\rho_{17}+\rho_{21-17}.
\]

Analytic terms have identical order values and explicit zero convergence, while retaining rounding allowances. The complete before/after allowance is

\[
A_{\mathrm{complete}}=\sum_{i=1}^{12}A_i+
4096\,\gamma_{13}\max\left(1,\sum_{i=1}^{12}|s_iv_i|\right).
\]

For `delta=after-before`,

\[
A_{\mathrm{sub}}=4096\,\gamma_3
\max(1,|before|,|after|,|delta|,|before|+|after|),
\]
\[
\epsilon_\Delta=A_{before}+A_{after}+A_{sub}.
\]

For an independently generated exact-candidate scalar pair `(p_j,o_j)` with frozen count `N_j`, define

\[
\rho^p_j=4096\gamma_{N_j}\max(1,\kappa^p_j)\max(1,|p_j|),\qquad
\rho^o_j=4096\gamma_{N_j}\max(1,\kappa^o_j)\max(1,|o_j|),
\]
\[
\rho^{cmp}_j=4096\gamma_3\max(1,|p_j|,|o_j|,|p_j|+|o_j|),\qquad
A^{candidate}_j=\rho^p_j+\rho^o_j+\rho^{cmp}_j.
\]

For z0 and source-row fields both kappas are `1.0`; for every M-block field both are the recorded condition number of `G`. Exact-candidate agreement requires each `abs(p_j-o_j) <= A_candidate_j`. Proposal cases GEM/natural have no independently generated parameter candidate and therefore no candidate-field comparison; their independent check is the complete before/after delta.

Every positive case also carries an operand-shaped production/oracle complete-delta agreement. For either implementation `x in {production, oracle}`, its `before` and `after` operand records are derived from that implementation's own ordered 12-term order-21/order-17 evaluation, never copied from the other implementation. A complete operand retains that implementation's exact ordered 12-term trace and records the reported order-21 value; the sum of both-order term operation counts plus exactly `13` for complete reduction; the ordered flattening of both-order term-local condition numbers; the ordered flattening of both-order absolute summands followed by the 12 absolute signed reported terms; the sum of the 12 term convergence estimates; the sum of all term order-21, order-17, and cross-order roundings plus the complete reduction rounding; and `allowance=convergence+rounding`. Every aggregate is recomputed from the retained trace. Its `delta` operand has an empty complete-term trace and is derived only from its own two complete operands:

\[
\Delta_x=after_x-before_x,\qquad N_{\Delta_x}=3,\qquad
\kappa_{\Delta_x}=(1),\qquad S_{\Delta_x}=(|before_x|,|after_x|),
\]
\[
\rho_{\Delta_x}=4096\,\gamma_3
\max(1,|before_x|,|after_x|,|\Delta_x|,|before_x|+|after_x|),
\qquad A_{\Delta_x}=A_{before_x}+A_{after_x}+\rho_{\Delta_x}.
\]

The independent agreement comparison is

\[
\rho_{cmp}=4096\,\gamma_3
\max(1,|\Delta_p|,|\Delta_o|,|\Delta_p|+|\Delta_o|),
\qquad A_{agreement}=A_{\Delta_p}+A_{\Delta_o}+\rho_{cmp},
\]

and passes exactly when `abs(delta_p-delta_o) <= A_agreement`. Production `A_delta_p` must equal the transaction's `H5DeltaAllowance.epsilon_delta`; the byte-only oracle independently derives `A_delta_o` from its own term traces and local frozen numerical constants. A missing, nonfinite, incorrectly ordered, formula-inconsistent, or cross-side-copied operand record is not agreement and makes `all_delta_allowances_operand_shaped` false.

No global kappa, run-wide maximum, unrelated term scale, stochastic allowance, or solver contribution may be added.


## H5 Public Interface Seams

The complete immutable schemas and exact canonical field orders are owned by Tasks 5–8 below. These top-level seams are normative:

```python
H5AttemptOutcome: TypeAlias = CompletedUpdateAttempt | FailedUpdateAttempt

def build_h5_reference_state(
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5ReferenceState: ...

def evaluate_h5_complete_elbo(
    reference: H5ReferenceState,
    state: H5LiveState | H5CandidateSnapshot,
    *,
    frozen_complement_sha256: str,
    cache: Mapping[FactorCacheKey, FactorCacheEntry] | None = None,
) -> CompleteElboEvaluation: ...

def execute_update(
    reference: H5ReferenceState,
    live: H5LiveState,
    request: UpdateRequest,
    evaluator: CompleteElboEvaluator,
    budget: H5BudgetConfig,
    *,
    fault_injection: H5FaultInjection | None = None,
) -> H5TransactionResult: ...

def evaluate_h5(
    config: ResolvedConfig,
    *,
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5GateEvaluation: ...
```

The unified runner accepts the explicit H1–H5 result union. H4 and H5 retain separate fail-closed result records, measurements, statuses, payloads, and provenance.


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
- Exact-coordinate attempts allow a resolved rounding-scale nonincrease (`delta >= -epsilon_delta`). GEM is stricter: it must resolve a positive increase (`delta > epsilon_delta`). A future proof-bound MM rule could use the former policy, but no MM request reaches H5 attempt or gate code in v1. This prevents a numerically unresolved tie from being advertised as generalized-EM progress.
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

- Produce immutable `H4RawDraw`, `H4AffineGaussianFactor`, `H4NeutralProblem`, `H4SolveProtocol`, `H4NativeInformationState`, `H4NativeMomentState`, `H4SelectedMoment`, `H4TerminalLaw`, `H4SolverResult`, `H4TimingRecord`, `H4OperationRecord`, `H4MemoryRecord`, `H4GateResult`, the exact `Literal` aliases, and exported `canonical_h4_problem_bytes` with the explicit schemas and validations in the Public Interface Map. `H4TerminalLaw.selected_moments` is an exact-name tuple in canonical order, never a mutable dictionary. Every `H4SelectedMoment` contains an immutable mean and covariance block.
- Produce `make_h4_problem(*, seed: int, kind: H4ProblemKind, horizon: Literal[7,15,31], d_z: Literal[4], d_m: Literal[4]) -> H4NeutralProblem` and `h4_anchor_from_h3(fixture: H3Fixture) -> H4NeutralProblem`.
- The generator returns one fully materialized immutable problem. Neither solver receives a seed or generator callback.

- [ ] **Step 1: Write the H4 preregistration before any timing exists.** Copy every H4 global constraint, the exact 20 seed values, zero-based horizon/seed/kind indices, exact traversal, the dimension table, factor-generation formulas, three warmup-pair and 11 timed-pair AB/BA formulas, primary per-seed and aggregate timed-order balance, timer/batched-postflight boundaries, primary endpoint, bootstrap algorithm, `0.80` decision table, scaled conditioning envelope, `1e-9` solver budget, strict `1e-4` allowance/scale cap, equivalence fields, status precedence, operation/memory secondary status, JSON schema, and H5/H6/H7/H8/training nonclaims. Before timing, copy the exact raw-draw schema/order/names, factor metadata/support rules, `source_kind`/seed rule, coordinate/problem/factor ID spellings, scoped scaled zero-control exceptions, exact `J`/`h`/`c` and derived-normalizer assembly, digest domain/core/envelope contract, and exact `H4GateResult` measurement/invariant/allowance schemas, including the pre-timing H3-anchor-failure measurement applicability, unevaluated-invariant detail, and allowance sentinel records. State explicitly that warmups do not enter inferential balance and that no coefficient, seed, order, repetition count, envelope, budget, cap, bootstrap setting, or threshold was chosen from H4 measurements.

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

  Define the scaled problem constructively. Its `source_kind` is exactly `scaled_pcg64`; storage remains population-major `[z_0,m_0,z_1,m_1,...,z_T,m_T]`, with `D=(T+1)*(d_z+d_m)`, `d_z=d_m=4`; the initial joint `[z_0,m_0]` is fixed `N(0,I_8)` and consumes no RNG draw. The normalized factor schedule and IDs are exactly `initial_joint`, then for every ascending `t=1..T`, `m_transition[t]`, `z_transition[t]`, and `observation[t]`. Its metadata is exact: initial normalized indices are `z_0` then `m_0`; `m_transition[t]` normalizes `m_t` and parents `m_{t-1}`; `z_transition[t]` normalizes `z_t` and parents `z_{t-1}` then `m_t`; `observation[t]` has no normalized indices and parents `z_t` then `m_t`; every listed coordinate block is ascending. Thus `m_t` is generated and consumed before `z_t|m_t`, without changing storage order.

  The only generator is `numpy.random.Generator(numpy.random.PCG64(seed))`; neither solver receives it. For each ascending `t`, draw exactly and only in this order: `A_m`, `A_z`, and `B`, each `standard_normal((4,4))`; `c_m` then `c_z`, each `uniform(-0.25,0.25,size=4)`; `R_m` then `R_z`, each `uniform(0.5,1.5,size=4)`; raw `G=standard_normal((8,8))`; observation offset `uniform(-0.25,0.25,size=8)`; observation noise `uniform(0.75,1.25,size=8)`; and observed target `uniform(-1,1,size=8)`. Their exact raw-draw names are `A_m[t]`, `A_z[t]`, `B[t]`, `c_m[t]`, `c_z[t]`, `R_m[t]`, `R_z[t]`, `G[t]`, `observation_offset[t]`, `observation_noise[t]`, and `observed_target[t]`, with global draw index `11*(t-1)+local_index`. Define `spectral_clip(M)=M*min(1,0.65/||M||_2)`, apply it to `A_m` and to the horizontally concatenated `[A_z B]`, and split the latter back into `A_z` and `B`. Set `H=I_8+0.05*G/max(1,||G||_2)`. The factors are therefore `m_t ~ N(A_m m_{t-1}+c_m,diag(R_m))`, `z_t ~ N(A_z z_{t-1}+B m_t+c_z,diag(R_z))`, and `y_t ~ N(H[z_t,m_t]+offset,diag(observation_noise))` at the drawn target; its normalized residual target is exactly `b=observed_target-offset` while both raw values remain provenance.

  The `source_kind="scaled_pcg64"` zero control is derived from the same draws and records: allowed differences from its coupled peer are only kind, problem ID, canonical SHA-256, and designated transition-parent matrix columns. It zeros all designated `A_m`, `A_z`, and `B` parent columns for every `t`, including columns whose coupled values happen already to be zero, while retaining raw draws, targets, covariances, roles, time indices, normalized and parent metadata, factor IDs/order, seed, horizon, shape, and all other factor fields unchanged. This matched-pair invariant never applies to an H3 anchor. Serialize every generated float and raw-draw provenance into immutable `core`; hash exactly `b"vfe4.h4.neutral-problem.v1\\x00" + compact UTF-8 sorted-key finite JSON(core)`; then publish the separately digest-checked schema envelope, whose literal schema version is required before core-digest verification and whose embedded hash is the domain-separated core digest, never a self-referential or full-envelope hash.

  Freeze the exact H4 objective, sign, and constants. For every schedule factor `r` with residual dimension `d_r`, use

  ```text
  log f_r(y) = -1/2 (A_r y-b_r)^T R_r^{-1}(A_r y-b_r)
               - d_r/2 log(2π) - 1/2 logdet R_r.
  Σ_r log f_r(y) = -1/2 y^T J y + h^T y + c.
  complete_objective = log Z
                     = c + 1/2 h^T J^{-1}h - 1/2 logdet J + D/2 log(2π),
  ```

  for SPD `J`. Higher is better. This is the unrestricted Gaussian optimum/evidence, not negative VFE and not a second ELBO. The information and moment arms independently compute this same scalar, including all factor constants; `J,h` and selected moments are comparison records, not alternate objectives.

  The H3 anchor adapter maps the explicit structural fixture groups `initial_factors`, `transition_factors`, and `observation_factors`, each in declared order, into `H4NeutralProblem.factor_schedule`. Its `source_kind` is exactly `h3_anchor` and its retained integer seed is exactly `0`, which enters the anchor core and digest validation. Each scalar becomes `matrix=(row,)`, `target=(target,)`, and `covariance=((variance,),)` with a normalizer derived as `-0.5*log(2*pi*variance)`; role/time/normalized/parent metadata comes only from group and declared position. It preserves IDs and exact H3 coordinate spelling, does not infer groups or roles from IDs/names, and does not synthesize a state-space factorization. The coupled fixture compares its frozen reference log evidence; the zero fixture has no frozen reference log evidence and only compares independently derived `c/logZ` under H3 allowances.

  The exact selected-moment labels are `("initial", "terminal", "observation[1]", ..., "observation[T]")`. `initial` and `terminal` are the full joint `[z_t,m_t]` blocks at `t=0` and `t=T`; every `observation[t]` is the full local `[z_t,m_t]` block in ascending time. Keep all labels even when `T=1` makes blocks overlap; do not deduplicate, map, or alias them.

  H4 thread control is process-scoped and mandatory. After H1--H3 work and before H4 preflight/timing, read-only capture both intra-op and inter-op counts. Any capture error permits no set and requires no restore. Only after both succeed, call `torch.set_num_threads(1)` and verify the observed intra-op count is one; after any set attempt, restore the captured intra-op count in `finally` and verify inter-op remained exact. A capture/set/verify failure suppresses timed records and makes H4 `INCONCLUSIVE`; a restoration failure is an environment/protocol obligation that prevents H4 `PASS`. Never set inter-op threads.

  Resolve H4 status in this fixed precedence: protocol/environment/thread/fixture/condition/table-completeness/nonfinite ambiguity is `INCONCLUSIVE`; otherwise a finite decisive H3-anchor or terminal-law miss is `FAIL`; otherwise apply the primary interval rule (`PASS` only when upper bound `<=0.80`, `FAIL` only when lower bound `>=0.80`, and `[0.80,0.80]` or a crossing interval `INCONCLUSIVE`). Operation and memory diagnostics are secondary and never rescue or overturn that status.

#### Prerequisite C: exact bootstrap preregistration bytes

Before any H4 measurement exists, amend the H4 preregistration to freeze:

```python
rng = np.random.Generator(np.random.PCG64(20260721))
indices = rng.integers(
    0,
    20,
    size=(100000,20),
    endpoint=False,
    dtype=np.int64,
)
seed_log_ratios = np.asarray(tuple(math.log(r) for r in seed_ratios),
                             dtype=np.float64)
replicate_mean_log_ratios = np.mean(
    seed_log_ratios[indices],
    axis=1,
    dtype=np.float64,
)
log_lower, log_upper = np.percentile(
    replicate_mean_log_ratios,
    (2.5,97.5),
    method="linear",
)
lower, upper = math.exp(float(log_lower)), math.exp(float(log_upper))
```

The exact compact sorted-key UTF-8 digest header is:

```json
{"dtype":"<i8","endpoint":false,"high":20,"low":0,"seed":20260721,"shape":[100000,20]}
```

The exact digest is:

```text
SHA256(
  b"vfe4.h4.bootstrap-indices.v1\x00"
  + header
  + b"\x00"
  + ascontiguousarray(indices,dtype="<i8").tobytes(order="C")
)
= a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14
```

Percentiles are never computed in ratio space and the 220 repetition times
are never bootstrap units.

- [ ] **Step 2: Write strict type/generator tests.** Assert exact ordered fields and validations for every defined H4 record, including explicit `source_kind` rather than ID inference; scaled positive PCG64 seeds and H3-anchor `seed=0` exactly in IDs/core/digests; `H4RawDraw` global zero-based indices, row-major values, shape product, finite values, factor-local ordering/names, and H3 empty raw draws; the factor residual/no-normalizer, exact `A:d x D`, `b:d`, SPD `R:d x d`, metadata order/disjointness/support/identity-column contracts; immutable tuple/mapping ownership; schedule availability rather than ID/numeric-role inference; and `factor_schedule` as the sole canonical source. Assert exact scaled coordinate strings, factor IDs, causal metadata tuple orders, and `D` table `(64,128,256)`, H3's unchanged coordinate spelling/IDs, fixed no-RNG initial `N(0,I_8)`, the exact PCG64 names/indices/draw order/distributions, separate `A_m` and joint `[A_z B]` spectral-clip envelopes, exact `H`, and scaled observation `b=observed_target-offset` with raw provenance retained. Require that only scaled matched controls have exactly the listed exceptions and that every designated transition-parent column is zeroed; do not apply this invariant to H3 anchors. Require deterministic core digest and published envelope bytes, exact hash domain/schema literal before digest verification, parser recomputation rather than self-reference/full-envelope hash, and distinct hashes across kind/seed/size. Require exact `J`, `h`, and `c` factor assembly including derived normalizers. Require exact `H4_INVARIANT_NAMES`, `H4_MEASUREMENT_NAMES`, `H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL`, and `H4_ALLOWANCE_INVARIANT_NAMES` ordering; test the sole pre-timing decisive H3-anchor failure with its five-and-only-five unavailable measurements, finite threshold/residual/allowance fraction, exact unevaluated-invariant records, and applicable/inapplicable allowance shapes; reject fabricated values, wrong phase/obligation evidence, malformed sentinels, or numeric inapplicable records. Require closed result aliases, finite conclusive measurements, `INCONCLUSIVE` `None` values only with phase obligations, and recursive owned immutable finite allowance records. Require each `H4TimingRecord` to carry independent `problem_index`, `horizon_index`, `seed_index`, `kind_index`, timed `repetition_index`, absolute `pair_index`, exact order label, and both positive native-arm durations; reject an order inconsistent with the independent-index parity formula. Require the exact immutable selected-moment labels `("initial","terminal","observation[1]",...,"observation[T]")`, with immutable mean/covariance rows; reject reordered, duplicate, missing, mapping, mutable, or aliased values. Adapt both raw H3 fixtures through the explicit structural-group adapter; require the coupled canonical `(J,h,c,logZ)` to agree with H3/reference allowances and the zero anchor to compare independently derived adapter/oracle `c/logZ` only, without asserting a nonexistent frozen reference.

  For the Prerequisite A records, assert exact frozen/slotted field order, deep immutable ownership, the six literal allowance counts and their recomputed total of 79,832,024, source/arm/repetition legality, nonempty unique row-major paths, element-local scales, exact operation-count arithmetic, finite positive final allowances, strict decisiveness, residual pass logic, bounded 4,096-lane accumulation, digest and witness consistency, exact inapplicable sentinels, and the restoration-failure anchor branch. Exercise `classify_h4_interval` at support, no-support, crossing, and exact-boundary cases; require `decide_h4_interval` to delegate without owning threshold inequalities; reject the removed flat allowance schema and every free-form allowance mapping.

- [ ] **Step 3: Run the Task 1 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h4_problem.py -q
  ```

  Expected: collection fails because `vfe4.types.h4` and `vfe4.generative.reference_h4` do not exist.

- [ ] **Step 4: Implement the immutable records and deterministic generator.** Validate all constructor invariants before hashing. Canonical JSON uses UTF-8, sorted keys, compact separators, finite JSON numbers, row-major arrays, and exact seed/kind/shape fields. The H3 adapter reads only public H3 normalized factors, neither changes H3 bytes nor imports an H3 oracle, and never assigns a factor a role by name.

  Implement the Prerequisite A typed allowance constructors, streaming accumulator construction boundary, public interval classifier, delegating interval helper, exact primary balance exports, and restoration-failure anchor semantics in the same Task 1 files; constructors recompute every derived arithmetic field and accept no free-form replacement schema.

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

- `vfe4.types.h4` remains dependency-light and unchanged by Task 2. `H4MaterializedProblem`, `H4InnovationDiagnostic`, and `H4NativeDiagnostics` are inference-layer runtime records in `vfe4.inference.h4_solvers`; they are not neutral protocol types and are not added to `vfe4/types/h4.py`.
- Consume and re-export the existing public `H4OperationKind`; no second alias or operation universe is allowed.
- Produce `H4MaterializedProblem`, `H4InnovationDiagnostic`, `H4NativeDiagnostics`, `H4GaussianSolver`, `InformationFormH4Solver`, `MomentFormH4Solver`, `materialize_h4_problem`, `solve_information_form`, `solve_moment_form`, `to_common_terminal_law`, and `evaluate_h4_native_diagnostics`.
- Produce `NullOperationRecorder`, `CountingOperationRecorder`, `InstrumentedLinearAlgebra`, and `measure_untimed_memory` in `vfe4.inference.h4_instrumentation`; recorder mutation remains module-private and capability-guarded.

#### Raw-only materialization and exact runtime records

`H4MaterializedProblem` is an immutable, frozen, slotted inference-layer record with exactly these logical fields in order:

```text
materialization_version
problem_id
problem_sha256
protocol_id
dtype
device
source_kind
seed
kind
horizon
d_z
d_m
dimension
coordinate_order
factor_ids
factor_roles
factor_time_indices
factor_normalized_coordinate_indices
factor_parent_coordinate_indices
_factor_matrices
_factor_targets
_factor_covariances
tensor_sha256
```

`materialization_version` is exactly `h4-materialized-problem-v1`, `dtype` is exactly `float64`, and `device` is exactly `cpu`. The three private tensor tuples are parallel to the factor metadata and contain only the raw neutral-factor `A`, `b`, and `R` values. Each stored tensor is detached, cloned, contiguous, CPU `torch.float64`, has `requires_grad=False`, owns distinct nonaliasing storage, and cannot alias a protocol tuple or another stored tensor. Materialization stores no Cholesky factor, whitening, solve, inverse, product, log determinant, eigenvalue, condition number, selected block, or other derived numerical value.

The record exposes no public tensor getter, setter, mutable collection, or mutative method. A module-private accessor callable only with a module-private capability held by the solvers and converter returns the owned tensors directly without cloning. Internal reads are clone-free; neither solver may mutate a materialized tensor or its view. `tensor_sha256` is domain-separated by the materialization-version literal and binds problem/protocol identity, complete ordered factor metadata, every tensor's role, shape, dtype, device, and canonical contiguous bytes. Construction recomputes and validates the digest rather than trusting a supplied field. Digest validation is outside every timed call.

Materialization exposes exactly:

```python
def materialize_h4_problem(
    problem: H4NeutralProblem,
    protocol: H4SolveProtocol,
) -> H4MaterializedProblem: ...
```

It validates the complete frozen protocol, canonical problem digest and identity, factor order and metadata, tuple/tensor shape agreement, and tensor ownership. It performs no matrix operation and no numerical derivation. The harness invokes it exactly once per neutral problem before any warmup, timed, counting, memory, conversion, or diagnostic pass. Both arms receive that same `H4MaterializedProblem` object by `is` identity for every warmup and timed pair; neither arm accepts a neutral problem, seed, generator, raw-draw record, callback, or separately materialized tensor.

`H4InnovationDiagnostic` is a frozen, slotted record with exactly these ordered fields:

```text
factor_id
time_index
parent_coordinate_indices
innovation_dimension
minimum_eigenvalue
maximum_eigenvalue
condition_number
minimum_cholesky_pivot
```

There is one record per observation factor in factor-schedule order. Identity fields equal the factor's declared metadata. The four numerical diagnostics are finite; dimension is the actual local innovation dimension; minimum/maximum eigenvalues and minimum pivot are positive; and `condition_number` equals `maximum_eigenvalue/minimum_eigenvalue` within an explicit float64 rounding check. Task 3 owns eligibility thresholds.

`H4NativeDiagnostics` is a frozen, slotted record with exactly these ordered fields:

```text
problem_id
problem_sha256
protocol_id
arm
factor_count
replayed_result
innovation_diagnostics
finite
spd
replay_matches_result
```

`replayed_result` is independently reconstructed by diagnostic replay. The final three fields are literal `True`; the producer raises rather than constructing a record when replay is nonfinite, non-SPD, or not exactly equal to the supplied result. Information diagnostics have an empty innovation tuple. Moment diagnostics contain every observation factor, including both scalar H3 observation factors and every scaled joint observation factor.

The inference layer exposes exactly:

```python
def to_common_terminal_law(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> H4TerminalLaw: ...

def evaluate_h4_native_diagnostics(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> H4NativeDiagnostics: ...
```

Both functions require exact materialization version/digest, problem ID/hash, protocol ID, arm, factor count, factor IDs/order, metadata, native-state class, and facade-binding agreement. Result factor identity is the exact materialized factor count plus the canonical problem hash that binds the complete ordered schedule; a detached count is insufficient.

#### Identity-bound real-operation instrumentation

The public instrumentation surface is exactly:

```python
class NullOperationRecorder:
    def snapshot(self) -> tuple[H4OperationRecord, ...]: ...

class CountingOperationRecorder:
    def snapshot(self) -> tuple[H4OperationRecord, ...]: ...

class InstrumentedLinearAlgebra:
    def __init__(
        self,
        *,
        problem_id: str,
        arm: H4SolverArm,
        recorder: NullOperationRecorder | CountingOperationRecorder,
    ) -> None: ...

    def cholesky(self, value: Tensor) -> Tensor: ...
    def triangular_solve(
        self,
        triangular: Tensor,
        rhs: Tensor,
        *,
        upper: bool = False,
    ) -> Tensor: ...
    def matrix_multiply(self, left: Tensor, right: Tensor) -> Tensor: ...
    def symmetric_rank_update(
        self,
        covariance: Tensor,
        gain: Tensor,
        innovation_covariance: Tensor,
    ) -> Tensor: ...
    def selected_block_extract(
        self,
        value: Tensor,
        row_indices: tuple[int, ...],
        column_indices: tuple[int, ...] | None = None,
    ) -> Tensor: ...

def measure_untimed_memory(
    problem_id: str,
    arm: H4SolverArm,
    callable: Callable[[], object],
) -> H4MemoryRecord: ...
```

The facade binds `problem_id` and `arm` immutably at construction. Numerical methods accept only operands; callers cannot supply or spoof identity per operation. `selected_block_extract` accepts a vector when `column_indices=None` and a matrix otherwise. `symmetric_rank_update` performs the real `covariance - gain @ innovation_covariance @ gain.T` operation and returns its roundoff-symmetrized result.

Recorder mutation is a module-private endpoint guarded by a module-private capability created and held only by `InstrumentedLinearAlgebra`. Public recorders expose only immutable `snapshot()` output; they expose no `record`, `observe`, `increment`, `record_only`, or capability getter. A facade method emits one `H4OperationRecord` only after its underlying operation succeeds. The counting recorder aggregates in first-successful-call order by `(problem_id, arm, operation, operand_shapes, result_shape)` and increments only the matching count. The null recorder returns identical numerical results and an empty snapshot.

Every Cholesky, triangular solve, matrix multiplication, symmetric rank update, and selected extraction in either solver, the common converter, or diagnostic replay goes through the supplied facade. Direct `torch.linalg` calls or `@` for these operations are forbidden. Elementwise arithmetic, accumulation, transpose views, finite checks, symmetrization, `diag`, `log`, norms, and scalar reductions are allowed directly. The only additional diagnostic matrix routine is untimed `eigvalsh` on an innovation covariance after that covariance was constructed through the facade.

`evaluate_h4_native_diagnostics` accepts only a null-recorder facade bound to the supplied result's exact problem and arm. Its replay is outside timing and the counting pass. Counting and memory are separate untimed passes; neither contributes operations to diagnostic replay or the timed result.

#### Frozen solver protocol and native objectives

`H4GaussianSolver.solve(materialized, protocol, linalg) -> H4SolverResult` is the only solver protocol. `InformationFormH4Solver`, `MomentFormH4Solver`, `solve_information_form`, and `solve_moment_form` accept only the same materialized object, protocol, and correctly identity-bound facade. Fresh arm-native state is allocated after the timer starts. Every solver checks the literal materialization version, protocol identity, facade binding, and complete factor schedule before using the private clone-free tensor view.

The information arm initializes fresh `J=zeros(D,D)`, `h=zeros(D)`, and `c=0` inside the timed call. For each raw materialized factor `(A,b,R)` in exact schedule order, it performs through the facade:

```text
L = cholesky(R)
A_w = triangular_solve(L, A)
b_w = triangular_solve(L, b)
J += matrix_multiply(A_w.T, A_w)
h += matrix_multiply(A_w.T, b_w)
c += -0.5 * (
    sum(b_w*b_w)
    + d*log(2*pi)
    + 2*sum(log(diag(L)))
)
```

After the complete pass it computes `L_J=cholesky(J)`, solves `J*mu=h` by two facade triangular solves, and returns

```text
logZ = c + 0.5*sum(h*mu) - sum(log(diag(L_J))) + D/2*log(2*pi)
```

which is exactly `c + .5*h.T*J^-1*h - .5*logdet(J) + D/2*log(2*pi)`. The native information state contains only `h`, `J`, `mu`, and this complete objective. Native finite/SPD checks occur after the full pass. It never constructs a complete covariance. Whitening, Cholesky, assembly, solve, and objective work all remain inside the timed call; materialization precomputes none of them.

The moment arm initializes `objective=0`, a fresh length-`D` mean buffer, a fresh `D x D` covariance buffer, and an empty active-coordinate set inside the timed call. Whenever a tensor subblock is read, active coordinates are the strictly ascending tuple of their declared global indices; the set is used only for membership and completeness checks.

It consumes every consecutive initial factor exactly once. For each initial factor with declared normalized global indices `C`, it scatters `b` into `mu[C]`, scatters `R` into `Sigma[C,C]`, sets cross-covariance with all previously active coordinates to zero, rejects overlap, and adds `C` to the active set. The scaled case consumes its one joint eight-dimensional initial factor. Each H3 anchor consumes both scalar initial factors in schedule order. Initial factors add no objective increment because their normalized laws are represented by direct construction.

It then consumes every transition and observation factor in the remaining exact schedule order. For a transition with normalized global child indices `C` and declared parent indices `P`, it requires all parents active and all children inactive, obtains `F=-A[:,P]`, `mu_P`, and every covariance subblock through `selected_block_extract`, and computes through the facade:

```text
mu_C = matrix_multiply(F, mu_P) + b
Sigma_C_active = matrix_multiply(F, Sigma[P,active])
Sigma_CC = matrix_multiply(Sigma_C_active[:,P-position], F.T) + R
```

The last line is algebraically `F*Sigma[P,P]*F.T+R`; parent positions come from the declared global active-coordinate map, never factor names or append order. It scatters `mu_C`, `Sigma_CC`, and the exact child-to-active cross-covariance `F*Sigma[P,active]` plus its transpose into declared global child positions, then marks `C` active. This fixed global scatter handles scaled `m_t` then `z_t|m_t` and H3 `m1` then `z1|m1` without appending or reordering coordinates. Transition factors add no objective increment because their normalized conditional laws are represented by direct construction.

For each observation factor, it obtains `A_active`, `mu_active`, and `Sigma_active` for the currently active declared coordinates through `selected_block_extract` and computes before conditioning:

```text
r = b - matrix_multiply(A_active, mu_active)
C = matrix_multiply(Sigma_active, A_active.T)
S = R + matrix_multiply(A_active, C)
L_S = cholesky(S)
v = triangular_solve(L_S, r)
increment = -0.5 * (
    sum(v*v)
    + d*log(2*pi)
    + 2*sum(log(diag(L_S)))
)
K_T = triangular_solve(L_S.T, triangular_solve(L_S, C.T), upper=True)
K = K_T.T
mu_active = mu_active + matrix_multiply(K, r)
Sigma_active = symmetric_rank_update(Sigma_active, K, S)
mu[active] = mu_active
Sigma[active,active] = Sigma_active
objective += increment
```

The updated active mean and covariance are scattered back into the declared global `mu`/`Sigma` buffers before the next factor is consumed; rebinding only the extracted local tensors is forbidden. Consequently, H3's second scalar observation conditions the posterior produced by its first observation, and each later scaled transition consumes the posterior produced by the preceding joint observation. The two H3 scalar observation factors are conditioned sequentially in declared schedule order; each scaled joint observation is conditioned in its declared position. The sum of sequential observation predictive log densities is the native normalized Gaussian objective. After the complete schedule, every global coordinate must be active and native finite/SPD checks run. The arm returns only full mean, full covariance, and accumulated objective. It never jitters, clips, pseudoinverts, calls an inverse API, calls the information arm or its assembler, instantiates `InformationGaussian`, or reads or mutates information-arm tensors.

#### Common conversion, selected blocks, residual, and replay

Selected coordinate blocks are derived only from declared global indices:

- `initial` is the global-coordinate-ordered union of every initial factor's declared normalized indices.
- `terminal` is the global-coordinate-ordered union of every transition factor's declared normalized indices at `time_index=horizon`.
- `observation[t]` is the global-coordinate-ordered union of every observation factor's declared parent indices at that time.

A missing, duplicate, undeclared, or wrongly dimensioned coordinate is rejected. Every block has dimension `d_z+d_m`: two for an H3 anchor and eight for a scaled problem. The two H3 observation factors jointly define the one `observation[1]` block, while each scaled observation factor defines its local eight-dimensional block. Labels are exactly `("initial", "terminal", "observation[1]", ..., "observation[T]")`, retaining overlap when `T=1`.

`to_common_terminal_law(materialized, result, linalg)` validates all identity and factor obligations before conversion. It propagates the native objective verbatim and never recomputes or normalizes it. For the information arm, it solves only the requested columns of `J^-1` through the facade and extracts only requested covariance blocks; it never constructs a complete covariance. For the moment arm, it derives full `J` with a Cholesky of its own covariance and facade triangular solves against the identity, then sets `h=matrix_multiply(J,mu)`; it never calls the information arm or shares its tensors. Both arms extract all selected means/covariance blocks through the facade and convert them into immutable nonaliasing H4 tuples.

For both arms the converter computes exactly:

```text
numerator = ||J*mu - h||_inf
scale = max(1, ||J||_inf*||mu||_inf + ||h||_inf)
stopping_residual = numerator/scale
```

The matrix infinity norm is the maximum absolute row sum, the vector infinity norm is the maximum absolute component, and `J*mu` goes through the facade. Conversion is outside timing.

`evaluate_h4_native_diagnostics(materialized, result, linalg)` reruns the matching arm from the same raw materialized tensors with a null-recorder facade. During moment replay it captures every already-computed innovation covariance before conditioning and evaluates eigenvalue, condition, and Cholesky-pivot diagnostics. It must reconstruct an `H4SolverResult` exactly equal field-for-field and float-for-float to the supplied result before returning `H4NativeDiagnostics`; mismatch raises and produces no usable diagnostic. Replay never substitutes for, mutates, or rewrites the native result and remains outside timing and counting.

- [ ] **Step 1: Write exact runtime-record and instrumentation tests.** Assert exact ordered fields and public signatures for all three runtime records, materialization, recorders, facade, solvers, converter, diagnostics, and memory helper. Require immutable recorder snapshots, immutable facade identity/arm, no public recorder mutation/capability endpoint, one count only after each successful operation, first-successful-call aggregation, and null/counting numerical equivalence.

- [ ] **Step 2: Prove materialization is raw-only and owned.** Compare every private tensor byte-for-byte to its protocol tuple; require detached cloned contiguous nonaliasing CPU-float64 storage, exact literal version/digest, and identical digest before/after each arm. Patch Cholesky, solve, multiply, inverse, log determinant, eigenvalue, condition, and whitening entry points to raise during materialization. Patch cloning to raise after materialization, require both arms to receive the same materialized object by `is`, and require deliberate private-storage tampering to fail digest validation.

- [ ] **Step 3: Add hand-checkable and H3/scaled solver cases.** Check frozen information `J/h/c/logZ`, both H3 initial factors and sequential scalar observations, the scaled joint initial factor, fixed global transition scatter, observation predictive objective, exact native-objective propagation, and selected-block dimensions two for H3 and eight for scaled problems. Perturb the first H3 observation and require the second observation's innovation/objective to change through the scattered posterior; perturb one scaled observation and require the next transition's propagated law to change. These regressions must fail if an implementation updates only extracted local tensors without scattering them back into global `mu`/`Sigma`.

- [ ] **Step 4: Enforce facade completeness.** Patch direct Cholesky, triangular solve, matrix multiply/`@`, symmetric update, and selected extraction alternatives to raise while facade calls remain available. Require every successful defined operation to produce exactly one count. Permit `eigvalsh` only in explicit null-bound diagnostic replay.

- [ ] **Step 5: Add exact independence seams.** Patch `solve_information_form`, every information assembler, `InformationGaussian`, `torch.linalg.inv`, `torch.linalg.pinv`, and `torch.cholesky_inverse` to raise while the moment arm succeeds. Patch moment growth/conditioning helpers to raise while the information arm succeeds. Assert neither arm reads the other's native state.

- [ ] **Step 6: Test common conversion and residual.** Reject every materialization-version/digest, problem/hash, protocol, arm, factor, native-state, and facade-binding mismatch. Require information conversion to solve only requested inverse columns, moment conversion to use only its own covariance, exact selected labels/blocks, exact residual scaling, and verbatim native objective.

- [ ] **Step 7: Test null-bound diagnostic replay.** Require replay to use a null-bound facade outside counts, return every observation diagnostic in exact schedule order, reconstruct the supplied native result exactly, and reject a one-ulp replayed-result mismatch or a counting/wrong-identity facade.

- [ ] **Step 8: Run the Task 2 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h4_solvers.py tests/unit/test_h4_instrumentation.py -q
  ```

  Expected: collection fails because the two new inference modules do not exist.

- [ ] **Step 9: Implement instrumentation, raw-only materialization, both independent arms, common conversion, and null-bound diagnostic replay.** Implement the identity-bound capability-guarded facade, raw tensor ownership/digest, untimed memory helper without numerical precomputation, exact information and moment pseudocode, one-pass schedule, native objectives, global-coordinate scatter, finite/SPD checks, independence prohibitions, exact selected blocks and residual, information selected-column solves, moment-owned precision conversion, verbatim objective propagation, innovation diagnostics, and exact replay equality outside timing/counting.

- [ ] **Step 10: Run the Task 2 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h4_solvers.py tests/unit/test_h4_instrumentation.py -q
  ```

  Expected: materialization ownership/digest, independent native algorithms/objectives, H3/scaled global scatter and selected blocks, exact residual, identity-bound real-operation accounting, and untimed null-bound replay all pass.

- [ ] **Step 11: Commit Task 2.**

  ```powershell
  git add vfe4/inference/h4_instrumentation.py vfe4/inference/h4_solvers.py vfe4/inference/__init__.py tests/unit/test_h4_solvers.py tests/unit/test_h4_instrumentation.py
  git commit -m "feat: add independent H4 Gaussian solver arms"
  ```

---

### Task 3: Add the Independent H4 Oracle, Operand-Shaped Equivalence Budgets, and Paired Statistics

##### Task 3 prerequisite and numerical authority

- Modify: `docs/preregistrations/2026-07-21-h4-information-cost.md`
- Modify: `vfe4/types/h4.py`
- Modify: `vfe4/types/__init__.py`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `vfe4/config/__init__.py`
- Modify: `tests/unit/test_h4_problem.py`
- Modify: `tests/unit/test_config.py`
- Create: `verification/h4_records.py`
- Create: `verification/numpy_oracles/h4_gaussian.py`
- Modify: `verification/numpy_oracles/__init__.py`
- Create: `verification/h4_budget.py`
- Create: `verification/h4_statistics.py`
- Create: `tests/unit/test_h4_records.py`
- Create: `tests/oracle/test_h4_numpy_oracle.py`
- Create: `tests/unit/test_h4_budget.py`
- Create: `tests/unit/test_h4_statistics.py`

#### Prerequisite B: freeze and resolve the H4 section before Task 4

Add these frozen, slotted records in `vfe4/config/schema.py` and re-export them
from `vfe4/config/__init__.py`:

```python
@dataclass(frozen=True, slots=True)
class H4TraversalConfig:
    horizons: tuple[int, int, int]
    seeds: tuple[int, ...]
    kinds: tuple[H4ProblemKind, H4ProblemKind]
    d_z: Literal[4]
    d_m: Literal[4]
    dimensions: tuple[int, int, int]
    primary_horizon: Literal[31]
    primary_kind: Literal["coupled"]
    primary_dimension: Literal[256]

@dataclass(frozen=True, slots=True)
class H4TimingConfig:
    parity_expression: Literal[
        "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0"
    ]
    warmup_pair_indices: tuple[int, int, int]
    timed_pair_indices: tuple[int, ...]
    timed_repetitions_per_problem: Literal[11]
    warmups_count_toward_balance: Literal[False]
    primary_timed_balance: tuple[tuple[int, int, int], ...]
    primary_5_ab_6_ba_rows: Literal[10]
    primary_6_ab_5_ba_rows: Literal[10]
    primary_timed_ab_total: Literal[110]
    primary_timed_ba_total: Literal[110]
    clock: Literal["time.perf_counter_ns"]
    timer_boundary: Literal["fresh_native_solver_call_v1"]
    between_repetitions: Literal["timer_reads_and_preallocated_assignments_only"]

@dataclass(frozen=True, slots=True)
class H4BootstrapConfig:
    seed: Literal[20260721]
    replicates: Literal[100000]
    inferential_units: Literal[20]
    index_low: Literal[0]
    index_high: Literal[20]
    endpoint: Literal[False]
    index_dtype: Literal["<i8"]
    index_shape: tuple[Literal[100000], Literal[20]]
    statistic: Literal["mean_log_seed_ratio"]
    percentiles: tuple[float, float]
    percentile_method: Literal["linear"]
    percentile_space: Literal["log_then_exp"]
    digest_domain: Literal["vfe4.h4.bootstrap-indices.v1"]
    expected_index_sha256: Literal[
        "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14"
    ]

@dataclass(frozen=True, slots=True)
class H4ConditionEnvelopeConfig:
    posterior_minimum_eigenvalue: float
    posterior_maximum_eigenvalue: float
    posterior_maximum_condition_number: float
    posterior_minimum_cholesky_pivot: float
    posterior_maximum_mean_infinity_norm: float
    innovation_minimum_eigenvalue: float
    innovation_maximum_eigenvalue: float
    innovation_maximum_condition_number: float
    inclusive: Literal[True]

@dataclass(frozen=True, slots=True)
class H4AllowanceConfig:
    float64_epsilon: float
    rounding_constant: Literal[4096]
    solver_relative_budget: float
    maximum_allowance_scale_fraction: float
    decisiveness_comparison: Literal["strict_less_than"]
    element_stream_domain: Literal["vfe4.h4.allowance-element-stream.v1"]
    maximum_chunk_rows: Literal[4096]

@dataclass(frozen=True, slots=True)
class H4EnvironmentConfig:
    device: Literal["cpu"]
    dtype: Literal["float64"]
    intra_op_threads: Literal[1]
    alter_inter_op_threads: Literal[False]
    cuda_expected: Literal[False]
    gc_policy: Literal["restore_exact_prior_enabled_state"]
    power_policy_field_order: tuple[
        Literal["active_power_scheme"],
        Literal["cpu_frequency_governor"],
        Literal["energy_performance_preference"],
        Literal["low_power_mode"],
    ]
    power_policy_capture: Literal["typed_best_effort_outside_timing"]

@dataclass(frozen=True, slots=True)
class H4ValidationConfig:
    schema_version: Literal["h4-validation-config-v1"]
    solve_protocol: H4SolveProtocol
    traversal: H4TraversalConfig
    timing: H4TimingConfig
    bootstrap: H4BootstrapConfig
    condition_envelope: H4ConditionEnvelopeConfig
    allowance: H4AllowanceConfig
    environment: H4EnvironmentConfig
    primary_effect_threshold: float
    maximum_validation_payload_bytes: Literal[67108864]
    canonical_json: str
    config_sha256: str
```

`vfe4/config/schema.py` imports the dependency-light public
`H4SolveProtocol` from `vfe4.types.h4`. Canonical H4 JSON expands all six
protocol fields in the dataclass order shown below; it never serializes only
`protocol_id` and never accepts a mapping in the resolved field.

Exact resolved values are:

```text
horizons = (7,15,31)
seeds = H4_PROBLEM_SEEDS
kinds = ("coupled","zero_control")
dimensions = (64,128,256)
warmup_pair_indices = (0,1,2)
timed_pair_indices = (3,4,5,6,7,8,9,10,11,12,13)
percentiles = (2.5,97.5)
posterior limits = (lambda_min >= 1e-6, lambda_max <= 1e6,
                    kappa_2 <= 1e8, pivot >= 1e-3,
                    ||mu||_inf <= 16)
innovation limits = (lambda_min >= 1e-6, lambda_max <= 1e6,
                     kappa_2 <= 1e8)
float64_epsilon = 2.220446049250313e-16
solver_relative_budget = 1e-9
maximum_allowance_scale_fraction = 1e-4
allowance stream domain = "vfe4.h4.allowance-element-stream.v1"
maximum allowance chunk rows = 4096
power policy field order = ("active_power_scheme",
                            "cpu_frequency_governor",
                            "energy_performance_preference",
                            "low_power_mode")
maximum_validation_payload_bytes = 67108864 (64 MiB)
primary_effect_threshold = 0.80
```

The complete nested solve protocol is exactly:

```text
solve_protocol.protocol_id = "h4-single-pass-v1"
solve_protocol.dtype = "float64"
solve_protocol.device = "cpu"
solve_protocol.factor_passes = 1
solve_protocol.solver_relative_budget = 1e-9
solve_protocol.stopping_rule = "complete_schedule_finite_spd"
```

`H4ValidationConfig.__post_init__` requires
`type(solve_protocol) is H4SolveProtocol` and reconstructs
`H4SolveProtocol(**asdict(solve_protocol))` to exercise its own frozen
validator. It then proves these cross-field identities rather than merely
checking each field separately:

```text
solve_protocol.dtype == environment.dtype == "float64"
solve_protocol.device == environment.device == "cpu"
solve_protocol.solver_relative_budget == allowance.solver_relative_budget
solve_protocol.factor_passes == 1
len(timing.timed_pair_indices) == timing.timed_repetitions_per_problem == 11
timing.warmup_pair_indices == (0,1,2)
timing.timed_pair_indices == tuple(range(3,14))
bootstrap.inferential_units == bootstrap.index_high == len(traversal.seeds) == 20
traversal.dimensions
  == tuple((T+1)*(traversal.d_z+traversal.d_m) for T in traversal.horizons)
traversal.primary_dimension
  == (traversal.primary_horizon+1)*(traversal.d_z+traversal.d_m)
maximum_validation_payload_bytes == 67108864
```

Task 3 adds
`resolve_h4_validation_config(raw_h4: Mapping[str, object]) ->
H4ValidationConfig`. This is a standalone exact H4-section resolver; it does
not add an accepted runner prefix, alter the root `CONFIG`, or add `h4` to
`ResolvedConfig`. The accepted runner prefixes remain exactly `("H1",)`,
`("H1","H2")`, and `("H1","H2","H3")` until Task 9 atomically adds the
existing coupled `("H1","H2","H3","H4","H5")` milestone. There is never
an H4-only runner prefix.

The standalone resolver rejects unknown/missing keys, mutable aliases, wrong types,
reordered or duplicate values, any changed seed/horizon/kind order, flattened
`problem_index` parity, any changed pair index, a true warmup-balance flag,
any primary row/pattern/total disagreement, a noninclusive envelope, a changed
formula tag, digest, threshold, budget, cap, allowance-stream domain/chunk
bound, ordered power-policy category, payload-size ceiling, an incomplete or
subclassed solve protocol, and any failed cross-field identity above.

The resolver independently recomputes all 120 problem indices, every warmup
and timed order from the four independent indices, the exact primary 20-row
table, ten rows of each pattern, and totals `110/110`. It canonicalizes the H4
raw section as compact sorted-key finite JSON, computes the section SHA-256,
stores both in `H4ValidationConfig`, and also includes the same section in the
future full `ResolvedConfig.canonical_json/config_sha256` when Task 9 invokes
the same resolver for the coupled milestone. The caller cannot supply either
derived H4 hash.

Task 4 accepts only:

```python
def evaluate_h4(
    config: H4ValidationConfig,
    *,
    h3_coupled_bytes: bytes,
    h3_zero_bytes: bytes,
) -> H4GateEvaluation: ...
```

It raises before any H4 work unless `type(config) is H4ValidationConfig`, its
canonical JSON and SHA-256 revalidate, and both H3 byte objects parse to the
two frozen anchor identities and full raw hashes. Every gate-originated Task 2
entry point that accepts a protocol argument receives the same
`config.solve_protocol` object by identity, including materialization and every
timed, counting, and memory solver call. Task 2's diagnostic-internal replay is
the sole object-identity exemption: its private replay protocol need not be
`config.solve_protocol` by `is`, but it must equal all six configured protocol
fields exactly. Converter/diagnostic entry points that do not accept a protocol
remain bound by the materialized/result protocol ID and are checked against all
six complete configured fields before and after the call. No gate or runner
reconstructs `H4SolveProtocol()` or passes a protocol ID in place of the
complete config object.

---

#### Shared Task 3 verification records

Create `verification/h4_records.py` with stdlib-only frozen, slotted records.
It must not import `torch`, `vfe4.inference`, or a production solver.

##### Condition and coverage records

```python
@dataclass(frozen=True, slots=True)
class H4PosteriorConditionRecord:
    problem_id: str
    problem_sha256: str
    source: Literal["numpy_oracle", "information", "moment"]
    repetition_index: int | None
    dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    minimum_cholesky_pivot: float
    mean_infinity_norm: float
    finite: Literal[True]
    spd: Literal[True]
    eligible: bool

@dataclass(frozen=True, slots=True)
class H4InnovationConditionRecord:
    problem_id: str
    problem_sha256: str
    source: Literal["numpy_oracle", "moment"]
    repetition_index: int | None
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    innovation_dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    finite: Literal[True]
    spd: Literal[True]
    eligible: bool

@dataclass(frozen=True, slots=True)
class H4ConditionWitness:
    metric: Literal[
        "minimum_eigenvalue",
        "maximum_eigenvalue",
        "maximum_condition_number",
        "minimum_cholesky_pivot",
        "maximum_mean_infinity_norm",
        "first_ineligible",
    ]
    stream_index: int
    record: H4PosteriorConditionRecord | H4InnovationConditionRecord

@dataclass(frozen=True, slots=True)
class H4ConditionStreamSummary:
    name: Literal[
        "oracle_posterior",
        "terminal_posterior",
        "oracle_innovation",
        "moment_innovation",
    ]
    stream_domain: Literal["vfe4.h4.condition-record-stream.v1"]
    expected_record_count: int
    observed_record_count: int
    record_stream_sha256: str
    eligible_record_count: int
    ineligible_record_count: int
    witnesses: tuple[H4ConditionWitness, ...]
    all_eligible: bool

@dataclass(frozen=True, slots=True)
class H4ProblemConditionSummary:
    problem_id: str
    problem_sha256: str
    name: Literal[
        "oracle_posterior",
        "terminal_posterior",
        "oracle_innovation",
        "moment_innovation",
    ]
    stream_domain: Literal["vfe4.h4.problem-condition-record-stream.v1"]
    expected_record_count: int
    observed_record_count: int
    record_stream_sha256: str
    eligible_record_count: int
    ineligible_record_count: int
    witnesses: tuple[H4ConditionWitness, ...]
    all_eligible: bool

@dataclass(frozen=True, slots=True)
class H4CoverageRecord:
    name: Literal[
        "oracle_posterior",
        "terminal_posterior",
        "oracle_innovation",
        "moment_innovation",
        "native_replay",
        "operation_pass",
        "memory_pass",
        "execution_trace",
        "postflight_schedule",
    ]
    key_stream_domain: Literal["vfe4.h4.coverage-key-stream.v1"]
    expected_key_count: int
    observed_key_count: int
    expected_key_stream_sha256: str
    observed_key_stream_sha256: str
    missing_key_count: int
    extra_key_count: int
    duplicate_key_count: int
    first_missing_key: str | None
    first_extra_key: str | None
    first_duplicate_key: str | None
    complete: bool
```

Posterior eligibility applies all five posterior limits. Innovation
eligibility applies only its declared eigenvalue and condition limits. The
Task 2 innovation pivot remains in `H4NativeDiagnostics` as positivity/SPD
evidence but is not silently subjected to the posterior `1e-3` pivot limit;
innovations have no mean limit.

For the 120 scaled problems, complete coverage is exact:

```text
oracle posterior records = 120
retained terminal posterior records = 120*11*2 = 2640
oracle innovation records = 40*(7+15+31) = 2120
moment replay innovation records = 11*2120 = 23320
native replay wrappers = 120*11*2 = 2640
operation passes = 120*2 = 240
memory passes = 120*2 = 240
execution traces = 120
postflight event keys = 40*(636+1076+1956) = 146720
```

Canonical coverage keys are generated in traversal order, then repetition
`0..10`, then arm `("information","moment")`, then factor-schedule order.
Every retained terminal is checked; there is no cross-repetition shortcut.
The `postflight_schedule` coverage record instead consumes every
`H4PostflightEventKey` in the exact per-problem order frozen below and therefore
has expected key count 146,720; this global record is additional to the 120
per-problem schedule summaries.
Expected and observed keys are consumed as streams, never retained as tuples.
Both SHA-256 values start from
`b"vfe4.h4.coverage-key-stream.v1\x00" + name.encode("ascii") + b"\x00"`
and append each UTF-8 key as an unsigned eight-byte big-endian length followed
by its bytes. For `postflight_schedule`, that UTF-8 key is the compact
sorted-key JSON mapping of every `H4PostflightEventKey` field, with JSON `null`
for inapplicable optionals and no timing value. `complete` is true exactly when counts and digests agree and all
three discrepancy counts are zero. First discrepancy witnesses use canonical
lowest expected/observed position.

Condition records are likewise consumed once in canonical key order. The
global `H4ConditionStreamSummary` is never used as a per-problem carrier. Its
stream digest begins with
`b"vfe4.h4.condition-record-stream.v1\x00" + name.encode("ascii") + b"\x00"`
and consumes the complete traversal with exact counts `120`, `2640`, `2120`,
and `23320` in the four-name order above. The distinct compact
`H4ProblemConditionSummary` binds exactly one problem ID/full SHA-256; its
digest begins with
`b"vfe4.h4.problem-condition-record-stream.v1\x00" +
problem_sha256.encode("ascii") + b"\x00" + name.encode("ascii") + b"\x00"`
and has exact counts `1`, `22`, `T`, and `11*T`. It rejects a record carrying
another problem identity. One exact anchor-only exception makes the Task 4
compact oracle constructible: problem IDs `h4-anchor-h3-coupled-v1` and
`h4-anchor-h3-zero-control-v1` accept only `name="oracle_innovation"` with exact
expected/observed count `2`, one for each frozen H3 observation factor. No
anchor summary may use the other three names or any scaled count. Anchor
innovation summaries remain nested in their `H4CompactOracleRecord`; they are
excluded from the four scaled global accumulators, whose counts remain exactly
`120`, `2640`, `2120`, and `23320`, and from the scaled coverage-count sums.
Both forms append length-prefixed compact sorted-key
JSON with every float encoded as `float.hex()`. Posterior summaries have the
exact ordered metric witnesses
minimum eigenvalue, maximum eigenvalue, maximum condition number, minimum
Cholesky pivot, maximum mean infinity norm, then optional first ineligible.
Innovation summaries omit the inapplicable pivot and mean metrics. Extremum
ties select the lowest stream index. `all_eligible` is equivalent to
`ineligible_record_count == 0`, never to a sampled witness. A global summary
is finalized from the global accumulator, not by concatenating compact
per-problem digests; the gate cross-checks their counts and eligibility only.

##### Execution and restoration records

```python
@dataclass(frozen=True, slots=True)
class H4ArmCallSpan:
    problem_id: str
    phase: Literal["warmup", "timed"]
    pair_index: int
    repetition_index: int | None
    order: Literal["information_then_moment", "moment_then_information"]
    order_position: Literal[0, 1]
    arm: Literal["information", "moment"]
    start_nanoseconds: int
    end_nanoseconds: int
    duration_nanoseconds: int

@dataclass(frozen=True, slots=True)
class H4PostflightEventKey:
    problem_id: str
    problem_sha256: str
    event_index: int
    phase: Literal[
        "materialized_integrity",
        "terminal_conversion",
        "native_diagnostic_replay",
        "terminal_posterior_condition",
        "moment_innovation_condition",
        "oracle_rehydration",
        "oracle_route_agreement",
        "equivalence_group",
        "operation_pass",
        "memory_pass",
        "stream_compaction",
    ]
    repetition_index: int | None
    arm: Literal["information", "moment"] | None
    factor_id: str | None
    selected_moment_name: str | None
    equivalence_component: Literal[
        "kl_to_zero",
        "h",
        "J",
        "selected_mean",
        "selected_covariance",
        "objective",
    ] | None
    integrity_phase: Literal["after_timed_batch", "after_postflight"] | None

@dataclass(frozen=True, slots=True)
class H4PostflightTimingWitness:
    event: H4PostflightEventKey
    timed_batch_end_nanoseconds: int
    start_nanoseconds: int
    end_nanoseconds: int

@dataclass(frozen=True, slots=True)
class H4PostflightScheduleSummary:
    stream_domain: Literal["vfe4.h4.postflight-event-key-stream.v1"]
    expected_event_count: int
    observed_event_count: int
    expected_key_stream_sha256: str
    observed_key_stream_sha256: str
    first_mismatch_index: int | None
    first_expected_key: H4PostflightEventKey | None
    first_observed_key: H4PostflightEventKey | None
    timing_violation_count: int
    first_timing_violation: H4PostflightTimingWitness | None
    complete: bool

@dataclass(frozen=True, slots=True)
class H4GarbageCollectorRecord:
    problem_id: str
    capture_attempted: Literal[True]
    capture_error: str | None
    prior_enabled: bool | None
    disable_required: bool | None
    disable_attempted: bool
    disable_error: str | None
    effective_state_capture_error: str | None
    disabled_during_batch: bool | None
    restore_attempted: bool
    restored_enabled: bool | None
    restoration_error: str | None
    restored_exact_prior_state: bool

@dataclass(frozen=True, slots=True)
class H4ExecutionTrace:
    problem_id: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    warmup_spans: tuple[H4ArmCallSpan, ...]
    timed_batch_start_nanoseconds: int
    timed_spans: tuple[H4ArmCallSpan, ...]
    timed_batch_end_nanoseconds: int
    postflight_schedule: H4PostflightScheduleSummary
    garbage_collector: H4GarbageCollectorRecord
    warmups_count_toward_balance: Literal[False]
    timed_guard_violations: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class H4ThreadStateRecord:
    capture_error: str | None
    prior_intra_op_threads: int | None
    set_attempted: bool
    set_error: str | None
    effective_intra_op_threads: int | None
    verified_one: bool
    prior_inter_op_threads: int | None
    final_inter_op_threads: int | None
    inter_op_unchanged: bool
    restore_attempted: bool
    restored_intra_op_threads: int | None
    restoration_error: str | None
    restored_exact_prior_state: bool
```

A completed trace has exactly six warmup spans and 22 timed spans, with exact
pair/order identities. Each `H4TimingRecord.information_nanoseconds` and
`moment_nanoseconds` value equals its matching arm span's
`duration_nanoseconds`. The postflight schedule must be complete and every
postflight timing witness starts at or after the timed-batch end.
Trace objects are synthesized only after the batch from preallocated scalar
slots; constructing a dataclass inside the timer is forbidden.

Every postflight key repeats the enclosing problem ID/full SHA-256 and uses its
zero-based canonical `event_index`. Integrity keys require only their matching
`integrity_phase`. Conversion, replay, and terminal-posterior keys require
repetition `0..10` and a real arm. Moment-innovation keys require repetition,
`arm="moment"`, and the exact observation factor ID. Oracle-rehydration and
route-agreement keys have no optional discriminator. Equivalence keys require
repetition, arm, and component; `selected_moment_name` is present exactly for
the two selected components. Operation and memory keys require only an arm.
Compaction has no optional discriminator. Every unmentioned optional is
`None`; a fake arm, repetition, factor, selected label, or integrity phase is
rejected.

The gate owns a module-private timed-batch guard. Gate-level materialization,
terminal conversion, diagnostic replay, hashing, postflight shape/result
validation, serialization, operation counting, memory measurement, logging,
and printing all reject entry while the guard is active. Task 2's structural
protocol/factor-schedule checks and its algorithmically required,
facade-visible native factorizations remain inside the native timed call;
materialized-byte hashing and record-constructor `_spd` do not. Focused tests
patch every excluded gate callable and
prove the guard, rather than inferring exclusion from the absence of a trace
event.

Thread capture is a read-only preliminary phase: call
`torch.get_num_threads()` and then `torch.get_num_interop_threads()` exactly
once each before any setter. If either getter raises, retain the phase-valid
prior value (if any) plus one bounded `capture_error`, set
`set_attempted=False`, perform no process-global mutation, and therefore set
`restore_attempted=False`, `restored_intra_op_threads=None`,
`restoration_error=None`, and `restored_exact_prior_state=False`. A partial
capture is not permission to set or restore. Only after both priors were
captured does the guard attempt `torch.set_num_threads(1)`. Once that setter
has been attempted, restoration to the captured prior intra-op value is
mandatory in the outer `finally`, even if the setter or subsequent
verification raised, because a failed setter may have mutated state. Inter-op
threads are read again after restoration and must exactly equal the captured
prior value; they are never set.

Thread restoration is an outer `try/finally` around all post-set H4 work. Each
timed batch has an inner `try/finally` that restores cyclic GC to the exact
prior enabled state, including the case where GC was already disabled. GC
capture is always attempted exactly once. A capture exception gives
`prior_enabled=None`, suppresses disable and timing, and records the stable
error. With a captured `True`, disable is required and attempted exactly once;
with captured `False`, disable is not required and is not called. A disable
exception or an exception while checking the effective state suppresses
timing, but restoration is still attempted because the prior state is known.
All capture/disable/effective-state/restore errors are capped at 512 Unicode
code points and retained. A solver or GC exception may yield the typed scaled
incomplete-phase record defined below only after both restoration attempts
finish; it may not leak process-global state or fabricate timing rows. Thread
capture/set/verify or restore failure and GC restore failure preclude a conclusive
result, including the narrowly typed anchor-miss/restoration branch above.
Inter-op thread count is never changed.

---

#### Corrected Task 3: independent oracle, element-local budgets, traces, and paired statistics

##### Public interfaces

The independent module imports neither `torch` nor `vfe4` and consumes only
canonical neutral-problem bytes:

```python
@dataclass(frozen=True, slots=True)
class H4OracleSelectedMoment:
    name: str
    coordinate_indices: tuple[int, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

H4OracleRouteOperationLabel = Literal[
    "factor_covariance_cholesky",
    "factor_triangular_solves",
    "factor_assembly_matmuls",
    "factor_quadratics",
    "factor_logdet_reductions",
    "factor_J_sum_reduction",
    "factor_h_sum_reduction",
    "factor_c_scalar_combinations",
    "factor_c_sum_reduction",
    "posterior_precision_symmetrization",
    "posterior_precision_cholesky",
    "posterior_natural_solve",
    "posterior_quadratic",
    "posterior_logdet_reduction",
    "affine_propagation_matmuls",
    "innovation_assembly",
    "innovation_cholesky",
    "innovation_triangular_solves",
    "innovation_quadratics",
    "innovation_logdet_reductions",
    "kalman_gain_solves",
    "mean_updates",
    "covariance_updates",
    "route_sum_reduction",
]

@dataclass(frozen=True, slots=True)
class H4OracleOperandEvidence:
    path: str
    value: float
    value_norm: float
    absolute_summand_accumulation: float
    condition_numbers: tuple[float, ...]
    operation_counts: tuple[tuple[H4OracleRouteOperationLabel, int], ...]

@dataclass(frozen=True, slots=True)
class H4OraclePosteriorDiagnostic:
    dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    minimum_cholesky_pivot: float
    mean_infinity_norm: float

@dataclass(frozen=True, slots=True)
class H4OracleInnovationDiagnostic:
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    innovation_dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float

@dataclass(frozen=True, slots=True)
class H4OracleRouteAgreement:
    problem_id: str
    problem_sha256: str
    canonical_operand: H4OracleOperandEvidence
    predictive_operand: H4OracleOperandEvidence
    float64_epsilon: float
    rounding_constant: Literal[4096]
    solver_allowance: Literal[0.0]
    maximum_allowance_scale_fraction: float
    invariant_scale: float
    canonical_rounding_allowance: float
    predictive_rounding_allowance: float
    comparison_reduction_allowance: float
    residual: float
    normalized_residual: float
    final_allowance: float
    allowance_scale_ratio: float
    decisiveness_rule: Literal["allowance_scale_ratio_strictly_less_than_1e-4"]
    pass_rule: Literal["residual_less_than_or_equal_to_final_allowance"]
    decisive: bool
    passed: bool
    eligible: bool

@dataclass(frozen=True, slots=True)
class H4OracleEvaluation:
    schema_version: Literal["h4-numpy-oracle-v1"]
    problem_id: str
    problem_sha256: str
    source_kind: Literal["scaled_pcg64", "h3_anchor"]
    seed: int
    kind: Literal["coupled", "zero_control"]
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    factor_ids: tuple[str, ...]
    precision: tuple[tuple[float, ...], ...]
    natural: tuple[float, ...]
    constant: float
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    canonical_log_normalizer: float
    predictive_log_normalizer: float
    route_agreement: H4OracleRouteAgreement
    selected_moments: tuple[H4OracleSelectedMoment, ...]
    posterior_diagnostic: H4OraclePosteriorDiagnostic
    innovation_diagnostics: tuple[H4OracleInnovationDiagnostic, ...]
    operand_evidence: tuple[H4OracleOperandEvidence, ...]

@dataclass(frozen=True, slots=True)
class H4OracleKLEvaluation:
    value: float
    trace_term: float
    quadratic_mean_term: float
    minus_dimension_term: float
    candidate_logdet_precision_term: float
    minus_oracle_logdet_precision_term: float
    absolute_summand_accumulation: float
    candidate_condition_number: float
    oracle_condition_number: float
    operation_counts: tuple[tuple[str, int], ...]

def evaluate_h4_oracle(problem_payload: bytes) -> H4OracleEvaluation: ...

def reverse_kl_to_h4_oracle(
    oracle: H4OracleEvaluation,
    *,
    mean: tuple[float, ...],
    precision: tuple[tuple[float, ...], ...],
) -> H4OracleKLEvaluation: ...
```

The parser rejects duplicate/unknown/missing keys, a wrong schema literal,
wrong core digest, noncausal metadata, unsupported source-specific identity,
nonfinite values, non-SPD factor covariance, or an invalid schedule. It never
imports or calls a generator, Task 2 materializer, solver, converter, or
production factor assembler.

##### Two independent oracle routes

The canonical route assembles directly from normalized factors:

```text
J = sum(A.T @ solve(R,A))
h = sum(A.T @ solve(R,b))
c = -.5*sum(b.T@solve(R,b) + d*log(2*pi) + logdet(R))
mu_star = solve(J,h)
Sigma_star = solve(J,I)
logZ_canonical = c + .5*h.T@mu_star
                 - sum(log(diag(cholesky(J))))
                 + D/2*log(2*pi)
```

The predictive route independently constructs the moment law in declared
global coordinates, including both H3 initial factors or the scaled joint
initial factor, scatters every transition into its declared child/parent
indices, and for each observation before conditioning computes:

```text
r = b-A_active@mu_active
C = Sigma_active@A_active.T
S = R+A_active@C
L = cholesky(S)
v = solve(L,r)
increment = -.5*(sum(v*v)+d*log(2*pi)+2*sum(log(diag(L))))
K = solve(S,C.T).T
mu_active = mu_active+K@r
Sigma_active = symmetrize(Sigma_active-K@S@K.T)
logZ_predictive += increment
```

The two scalar routes close through `H4OracleRouteAgreement`; a bare residual
is not accepted. The canonical operand path is exactly
`"canonical_log_normalizer"`, the predictive path is exactly
`"predictive_log_normalizer"`, and their values equal the two enclosing
log-normalizer fields bit-for-bit. Their ordered operation-count tables and
condition-number tuples are the actual independently derived route evidence.
The enclosing `operand_evidence` tuple contains these two route operands
exactly once in canonical-then-predictive order, and the route agreement must
reference equal owned records; a duplicate, omitted, reordered, or divergent
copy is rejected.
For both scalar operands `value_norm=abs(value)`. Canonical
`absolute_summand_accumulation` is the sum of the absolute values of `c`,
`.5*h.T@mu_star`, the negated Cholesky log-diagonal reduction, and
`D/2*log(2*pi)`; its condition tuple is every factor-covariance condition
number in schedule order followed by `condition_number(J)`. Predictive
accumulation is the sum of `abs(increment)` in observation order; its condition
tuple is every propagated covariance then innovation covariance in the exact
order used by the route. Neither tuple is pooled with the other route.
The policy fields must equal the H4 allowance config bit-for-bit:
`float64_epsilon=2.220446049250313e-16`, `rounding_constant=4096`,
`solver_allowance=0.0`, and `maximum_allowance_scale_fraction=1e-4`.
Oracle-to-oracle route comparison never receives a solver term.
For a scalar route operand `x`, with its own operation count `n_x`, condition
maximum `kappa_x`, and absolute summand accumulation `a_x`, compute:

```text
rounding_x = 4096*gamma(n_x)*kappa_x*max(1,abs(x),a_x)
invariant_scale = max(1,abs(canonical),abs(predictive))
comparison_reduction_allowance = 4096*gamma(3)
  * max(1,abs(canonical),abs(predictive),abs(canonical)+abs(predictive))
residual = abs(canonical-predictive)
final_allowance = rounding_canonical + rounding_predictive
                  + comparison_reduction_allowance
allowance_scale_ratio = final_allowance/invariant_scale
normalized_residual = residual/final_allowance
decisive = allowance_scale_ratio < maximum_allowance_scale_fraction
passed = residual <= final_allowance
eligible = passed and decisive
```

The canonical route table is exactly
`factor_covariance_cholesky`, `factor_triangular_solves`,
`factor_assembly_matmuls`, `factor_quadratics`,
`factor_logdet_reductions`, `factor_J_sum_reduction`,
`factor_h_sum_reduction`, `factor_c_scalar_combinations`,
`factor_c_sum_reduction`, `posterior_precision_symmetrization`,
`posterior_precision_cholesky`,
`posterior_natural_solve`, `posterior_quadratic`,
`posterior_logdet_reduction`, `route_sum_reduction`. The predictive table is
exactly `affine_propagation_matmuls`,
`innovation_assembly`, `innovation_cholesky`,
`innovation_triangular_solves`, `innovation_quadratics`,
`innovation_logdet_reductions`, `kalman_gain_solves`, `mean_updates`,
`covariance_updates`, `route_sum_reduction`; zero counts remain explicit. Each nonnegative count is derived
from the actual factor dimensions and declared schedule using the closed
operation formulas below; missing, duplicate, reordered, or extra labels are
rejected.

For `F` normalized factors and posterior dimension `D`, the canonical counts
introduced for the executable accumulations are exact:
`factor_J_sum_reduction = F*D*D`,
`factor_h_sum_reduction = F*D`,
`factor_c_scalar_combinations = 4*F`,
`factor_c_sum_reduction = F`, and
`posterior_precision_symmetrization = 2*D*D`. These counts match zero-array/
zero-scalar initialization followed by one in-place accumulation per factor;
they must not use the first contribution as an implicit initializer. The four
c scalar-combination operations per factor are one multiplication for
`d_f*log(2*pi)`, two additions forming
`quadratic + d_f*log(2*pi) + logdet`, and one multiplication by `-0.5`;
the separate c-sum count is the following `constant += contribution`.

All derived fields are recomputed in `__post_init__`. `passed` is only the
numerical predicate `residual <= final_allowance`; equality at the final
allowance therefore sets `passed=True` even when the allowance ratio is
indecisive. Equality at `1e-4` sets `decisive=False`. Overall route
`eligible` is exactly `passed and decisive`. A record with either predicate
false is ineligible and makes H4 `INCONCLUSIVE`; neither route is selected
opportunistically.
The record is retained once in every compact oracle record and therefore in
both the gate evaluation and validation artifact.

Selected indices are derived only from factor metadata and exactly match Task
2: initial normalized-index union, terminal transition-child union at the
horizon, and each time's observation-parent union, all in global-coordinate
order. Every block is dimension `d_z+d_m`, labels are exactly
`("initial","terminal","observation[1]",...,"observation[T]")`, and overlap
at `T=1` is retained.

For a candidate `q=N(mu_q,J_q^-1)` and oracle posterior
`p*=N(mu_star,J_star^-1)`, compute without clipping:

```text
KL(q || p*) = .5*(
    trace(J_star @ inverse(J_q))
    + (mu_star-mu_q).T @ J_star @ (mu_star-mu_q)
    - D
    + logdet(J_q)
    - logdet(J_star)
)
```

Every retained information and moment result is evaluated independently. The
candidate covariance and both log determinants are obtained by Cholesky solves
and diagonal-log reductions; the implementation does not call
`numpy.linalg.inv`.
The
`exact_posterior_gap_equivalence` allowance compares each signed KL result to
the literal zero operand; a small negative roundoff value is retained, not
clipped. Arm-to-arm agreement cannot satisfy this invariant.

##### Budget interfaces

```python
def gamma_n(n: int) -> float: ...
def dot_operation_count(k: int) -> int: ...
def matrix_multiply_operation_count(m: int, k: int, n: int) -> int: ...
def triangular_solve_operation_count(n: int, rhs_columns: int) -> int: ...
def cholesky_operation_count(n: int) -> int: ...
def operand_allowance(...) -> H4AllowanceOperand: ...
def pair_element_allowance(...) -> H4AllowanceElement: ...

@dataclass(frozen=True, slots=True)
class _H4AllowanceOperandGroup:
    label: str
    values: NDArray[np.float64]
    value_norm: float
    absolute_summand_accumulations: NDArray[np.float64]
    condition_numbers: tuple[float, ...]
    operation_counts: tuple[H4AllowanceOperationCount, ...]
    solver_produced: bool

@dataclass(frozen=True, slots=True)
class _H4AllowanceGroupInput:
    problem_id: str
    problem_sha256: str
    comparison_source: Literal[
        "solver_to_oracle",
        "adapter_to_h3_reference",
        "adapter_to_oracle",
    ]
    repetition_index: int | None
    arm: H4SolverArm | None
    path_prefix: str
    shape: tuple[int, ...]
    left: _H4AllowanceOperandGroup
    right: _H4AllowanceOperandGroup

def aggregate_allowance_groups(
    invariant: H4AllowanceInvariantName,
    *,
    expected_element_count: int,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def build_h4_anchor_identity_allowance(
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def build_h4_exact_posterior_gap_allowance(
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def build_h4_terminal_h_allowance(
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def build_h4_terminal_j_allowance(
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def build_h4_selected_moment_allowance(
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def build_h4_complete_objective_allowance(
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance: ...

def allowance_is_decisive(record: H4ApplicableAllowance) -> bool: ...

def posterior_condition_record(
    *,
    problem_id: str,
    problem_sha256: str,
    source: Literal["numpy_oracle", "information", "moment"],
    repetition_index: int | None,
    dimension: int,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    condition_number: float,
    minimum_cholesky_pivot: float,
    mean_infinity_norm: float,
    envelope: H4ConditionEnvelopeConfig,
) -> H4PosteriorConditionRecord: ...

def innovation_condition_record(
    *,
    problem_id: str,
    problem_sha256: str,
    source: Literal["numpy_oracle", "moment"],
    repetition_index: int | None,
    factor_id: str,
    time_index: int,
    parent_coordinate_indices: tuple[int, ...],
    innovation_dimension: int,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    condition_number: float,
    envelope: H4ConditionEnvelopeConfig,
) -> H4InnovationConditionRecord: ...
```

`operand_allowance` and `pair_element_allowance` are scalar reference/witness
constructors used by focused tests and by final witness materialization. The
production complete stream uses only `aggregate_allowance_groups`; it may not
call either function once per scalar.

The six public builders above are the only single-invariant allowance
aggregation entry points. Each delegates to `aggregate_allowance_groups` with
the caller's two keyword-only iterables and its own frozen literal
invariant/count pair; the caller cannot supply or override either value. They
exist for focused tests and bounded direct fixtures. The production gate uses
the unified one-pass Task 3 accumulator frozen below so it never replays or
buffers the complete source stream:

```text
build_h4_anchor_identity_allowance -> ("h3_anchor_identity", 184)
build_h4_exact_posterior_gap_allowance -> ("exact_posterior_gap_equivalence", 2640)
build_h4_terminal_h_allowance -> ("terminal_h_equivalence", 394240)
build_h4_terminal_j_allowance -> ("terminal_J_equivalence", 75694080)
build_h4_selected_moment_allowance -> ("selected_moment_equivalence", 3738240)
build_h4_complete_objective_allowance -> ("complete_objective_equivalence", 2640)
```

They stream one scalar lane per canonical path and compare every solver arm to
an independent oracle/frozen reference. Selected mean and covariance paths
include the selected label and row/column. Builders operate in chunks of at
most 4,096 row-major scalars and retain only the four possible witness records
required by `H4ApplicableAllowance`. They reject a missing/duplicate path,
shape mismatch, global condition maximum, pooled solver flag, count mismatch,
or a record whose
recomputed arithmetic differs by any bit.

Every group array is exact `numpy.float64`, one-dimensional, C-contiguous,
read-only, finite, and has `prod(shape)` entries; the two value and two
absolute-summand arrays have identical length. The accumulator compares each
compact canonical group header to the independently generated expected header,
then takes read-only slices of at most 4,096 rows. For each slice it applies the
published arithmetic in the written order using `np.abs`, sequential
`np.maximum`, `np.multiply`, `np.add`, `np.subtract`, `np.divide`, `np.less`,
and `np.less_equal`, with no reduction reassociation. `np.argmax` supplies the
first local maximum and `np.flatnonzero(mask)[0]` the first failure; global
updates use strict `>` so ties retain the earliest stream index. One packed
structured scratch array of at most 4,096 rows feeds SHA-256 and is discarded
before the next slice. Only final witness lanes are converted to Python floats
and nested records.

Tests patch the `H4AllowanceElement` constructor to count calls and require no
more than the number of distinct persisted witnesses, feed a synthetic stream
larger than two chunks, assert every scratch array's `.nbytes` is at most
`4096 * packed_row_itemsize`, and use `tracemalloc` to prove Python-object
growth is not linear in total element count. They independently recompute
group order, all six exact element counts, the canonical digest, maxima, first
witnesses, pass/decisive conjunctions, and a late final-chunk miss. This is the
no-per-scalar-object guarantee; instantiating 79,832,024 dataclasses is a test
failure.

The operand operation tables incorporate the repaired Task 2 return paths,
not the pre-repair constructor behavior. In information conversion, the
already-required facade Cholesky of native `J` supplies both the covariance
solve and the terminal-`J` proof receipt. In moment conversion, the
already-required facade Cholesky of native covariance supplies the inversion
path's proof receipt, and the derived terminal precision receives one further
explicit facade Cholesky. Every selected covariance block receives its own
explicit untimed facade Cholesky. For `S=T+2` selected blocks, converter
evidence therefore contains exactly `S+1` Choleskys for information and `S+2`
for moment. These operations and their scalar counts are included in the
matching operand rounding tables.
There is no allowance row for `_spd` or a hidden dataclass-constructor
factorization; focused tests make `_spd` raise and require the public solver
and converter paths plus these explicit facade counts to remain valid.

The two condition builders are the only eligibility classifiers. They apply
the inclusive config comparisons literally, recompute `finite`/`spd`, and
return immutable records. The oracle emits raw diagnostics and never imports
configuration or embeds a second copy of the envelope.

##### Task 3-owned allowance group producers

Task 4 must not instantiate `_H4AllowanceOperandGroup` or
`_H4AllowanceGroupInput`, choose an operation count, condition number, norm,
absolute-summand value, solver flag, path, or expected header. Task 3 therefore
adds these bounded helpers to `verification/h4_budget.py`; the private group
types are removed from `__all__`:

```python
@dataclass(frozen=True, slots=True)
class H4ResultAllowanceGroupBundle:
    kl_to_zero: _H4AllowanceGroupInput
    terminal_h: _H4AllowanceGroupInput
    terminal_J: _H4AllowanceGroupInput
    selected_mean_and_covariance: tuple[_H4AllowanceGroupInput, ...]
    complete_objective: _H4AllowanceGroupInput

@dataclass(frozen=True, slots=True)
class H4AllowanceResultSource:
    problem_payload: bytes
    repetition_index: int | None
    oracle: H4OracleEvaluation
    result: H4SolverResult
    terminal: H4TerminalLaw
    kl_to_oracle: H4OracleKLEvaluation

@dataclass(frozen=True, slots=True)
class H4AnchorAllowanceSource:
    h3_fixture_bytes: bytes
    information: H4AllowanceResultSource
    moment: H4AllowanceResultSource

def h4_result_allowance_group_bundle(
    *,
    source: H4AllowanceResultSource,
) -> H4ResultAllowanceGroupBundle: ...

def h4_anchor_identity_groups(
    *,
    source: H4AnchorAllowanceSource,
) -> tuple[_H4AllowanceGroupInput, ...]: ...

class H4SixInvariantAllowanceAccumulator:
    def consume(
        self,
        source: H4AnchorAllowanceSource | H4AllowanceResultSource,
    ) -> None: ...

    def anchor_identity_record(self) -> H4ApplicableAllowance: ...

    def finalize(self) -> tuple[
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
    ]: ...

def new_h4_six_invariant_allowance_accumulator(
) -> H4SixInvariantAllowanceAccumulator: ...
```

The source records own no budget field; they only bind the original bytes and
full current-problem numerical records needed by both independent paths.
`H4AllowanceResultSource.repetition_index` is exactly `None` for a result
inside either `H4AnchorAllowanceSource` and exactly `0..10` for a scaled
result. It is part of every group identity and header. The constructors reject
an anchor repetition, a scaled `None`, a value outside `0..10`, or disagreement
with the accumulator's canonical problem/repetition/arm position; neither
`H4SolverResult` nor `H4TerminalLaw` is treated as an implicit repetition
carrier.
`h4_result_allowance_group_bundle` validates byte/object identity and exact
problem/hash/arm/dimension/selected-label agreement. It creates read-only
one-dimensional C-contiguous `float64` arrays and never retains a complete-run
array collection. `h4_anchor_identity_groups` reparses the supplied raw H3
bytes and canonical problem bytes internally; no caller supplies a reference
scalar or a `coupled`/`zero` switch.

Task 4 constructs one `H4SixInvariantAllowanceAccumulator`, calls `consume`
first for the coupled and zero-control anchor sources, then once for each
scaled result in exact problem, repetition `0..10`, arm
`information,moment` order, and calls `finalize` once. `consume` first invokes
a module-private expected-path producer that reparses the immutable bytes and
independently recomputes values, norms, absolute-summand vectors, conditions,
operation tables, vector digests, and group headers. It then invokes the
separate observed-path bundle producer and advances only the applicable
invariant accumulators. Anchor sources advance only `h3_anchor_identity`;
scaled result sources advance the other five invariants. No source is replayed,
`tee`d, or retained after `consume` returns. Each private invariant accumulator
retains only its digest state, counters, maxima, and bounded witnesses.
`finalize` refuses missing, duplicate, reordered, or extra sources and returns
the six records in the frozen `H4AllowanceInvariantName` order.

Immediately after exactly the coupled and zero-control anchor sources have
been consumed—and before any scaled source—Task 4 calls
`anchor_identity_record()`. The one-shot boundary freezes and caches the
complete 184-element `h3_anchor_identity` record without closing the unified
accumulator or the other five invariant states. Calling it before both anchors,
after any scaled source, or after a failed/reordered anchor consume is an
error. Repeated calls return the same immutable cached object only while the
accumulator remains at the exact two-anchor boundary before its first scaled
consume. Any premature, post-scaled, post-finalize, or failed/reordered call
fail-closes the accumulator permanently; no later `consume`, snapshot, or
`finalize` may recover it. If the anchor is decisive and fails, Task 4 may
close the preregistered early-FAIL branch
without consuming scaled sources. If it passes, scaled one-pass consumption
continues normally; final `finalize()` must return that exact cached anchor
record as its first tuple member. This snapshot neither replays an anchor
source nor copies/retains its operand arrays.

The six literal single-invariant builders remain public for focused unit tests
and direct bounded fixtures. Task 4 must use only the unified accumulator; it
may not call those builders sequentially, construct a private group, or access
a private accumulator.

The result bundle order is exactly `kl_to_zero`, `terminal_h`, `terminal_J`,
then `selected_mean,selected_covariance` for every declared selected label,
then `complete_objective`. The complete anchor stream is coupled information,
coupled moment, coupled adapter `J,h,c,logZ`, zero-control information,
zero-control moment, zero-control adapter `c,logZ`; its respective scalar
subtotals are `40,40,22,40,40,2`. Scaled streams traverse problem, repetition
`0..10`, arm `information,moment`; each invariant selects only its matching
bundle field. This order alone yields the six frozen counts.

The exact operand vectors and provenance are:

- `kl_to_zero`: left is the one signed `kl_to_oracle.value`; right is literal
  zero. Left uses `abs(value)`, the KL record's absolute-summand scalar,
  `(candidate_condition_number, oracle_condition_number)`, and the KL record's
  ordered operation table; right uses norm/summand zero, conditions `(1.0,)`,
  and an empty table.
- `terminal_h` and `terminal_J`: left is terminal `h` or row-major `J`; right
  is oracle `natural` or row-major `precision`. Vector norm is infinity norm
  and matrix norm is maximum absolute row sum. Per-lane absolute summands are
  the sum of absolute factor contributions for an assembly, the sum of
  absolute products for a matrix product, and otherwise `abs(output_lane)` for
  an indivisible Cholesky/triangular-solve output. Oracle operation tables and
  conditions are recomputed from the canonical factor route. Information-left
  `h` has an empty postflight table and information-left `J` has its required
  `cholesky(D)` proof receipt. Moment-left `J` has native-covariance
  `cholesky(D)`, two `triangular_solve(D,D)` receipts, and the derived-precision
  `cholesky(D)` proof; moment-left `h` appends
  `matrix_multiply(D,D,1)`. Conditions list the corresponding Cholesky-input
  condition numbers in that same order.
- each selected mean/covariance uses the declared label and shape `(s,)` or
  `(s,s)` with row-major covariance. Means use infinity norm; covariances use
  maximum absolute row sum. A selected extraction contributes the explicit
  zero-count `selected_extract` receipt. Information covariance appends the
  native-precision `cholesky(D)`, two `triangular_solve(D,s)`, and selected
  `cholesky(s)` receipts; moment covariance appends selected
  `cholesky(s)`. The oracle side uses the canonical-posterior solve receipt and
  the same selected extraction/block-proof order. Absolute-summand lanes use
  the exact product sums where a product occurs and `abs(output_lane)` for a
  solve/extract result.
- `complete_objective`: left is terminal `complete_objective` propagated
  verbatim with norm/summand `abs(value)`, conditions `(1.0,)`, and an empty
  postflight table; right is `oracle.canonical_log_normalizer` with the exact
  canonical route operand norm, absolute-summand scalar, condition tuple, and
  ordered operation table already validated by `H4OracleRouteAgreement`.
- the coupled adapter groups use canonical factor assembly for row-major
  `J`, `h`, `c`, and canonical `logZ`; frozen H3-reference operands use
  `abs(value)`, conditions `(1.0,)`, an empty operation table, and no solver
  term. The zero-control adapter uses only canonical `c,logZ` against the
  independently evaluated oracle operands; no nonexistent frozen zero-control
  reference is synthesized.

Operation labels are emitted in the written order with zero-count
`selected_extract` retained; all other zero counts are omitted. Counts use
only the closed Task 3 scalar formulas and the exact operand shapes above.
The literal conversion-table labels/counts are: information `h` `()`;
information `J` `(("terminal_information_precision_proof_cholesky",
cholesky(D)),)`; moment `J`
`("terminal_moment_covariance_cholesky",cholesky(D)),
("terminal_moment_precision_solves",2*triangular_solve(D,D)),
("terminal_moment_precision_proof_cholesky",cholesky(D))`; moment `h` is
that tuple plus `("terminal_moment_natural_matmul",matmul(D,D,1))`.
A selected mean table is exactly `(("selected_extract",0),)`. An information
selected-covariance table is `("selected_information_precision_cholesky",
cholesky(D)),("selected_information_covariance_solves",
2*triangular_solve(D,s)),("selected_extract",0),
("selected_covariance_proof_cholesky",cholesky(s))`; a moment selected-
covariance table is `("selected_extract",0),
("selected_covariance_proof_cholesky",cholesky(s))`. Here the abbreviated
functions are exactly the public Task 3 count functions, not new formulas.
Oracle and adapter operands use these distinct operand-local tables; the full
canonical-`logZ` table is never copied onto an upstream `J`, `h`, `c`, or
selected operand. Let `F` be the normalized factor schedule and `d_f` the row
dimension of factor `f`. Each written label remains present even when its
derived count is zero:

```text
oracle_or_adapter_J =
  factor_covariance_cholesky
  factor_precision_solves_A
  factor_J_assembly_matmuls
  factor_J_sum_reduction
  posterior_precision_symmetrization

oracle_or_adapter_h =
  factor_covariance_cholesky
  factor_precision_solves_b
  factor_h_assembly_matmuls
  factor_h_sum_reduction

oracle_or_adapter_c =
  factor_covariance_cholesky
  factor_precision_solves_b
  factor_c_quadratics
  factor_c_logdet_reductions
  factor_c_scalar_combinations
  factor_c_sum_reduction

oracle_selected_mean =
  factor_covariance_cholesky
  factor_precision_solves_A
  factor_precision_solves_b
  factor_J_assembly_matmuls
  factor_h_assembly_matmuls
  factor_J_sum_reduction
  factor_h_sum_reduction
  posterior_precision_symmetrization
  posterior_precision_cholesky
  posterior_natural_solve
  selected_extract

oracle_selected_covariance =
  factor_covariance_cholesky
  factor_precision_solves_A
  factor_J_assembly_matmuls
  factor_J_sum_reduction
  posterior_precision_symmetrization
  posterior_precision_cholesky
  posterior_covariance_solves
  selected_extract
  selected_covariance_proof_cholesky

oracle_logZ =
  factor_covariance_cholesky
  factor_triangular_solves
  factor_assembly_matmuls
  factor_quadratics
  factor_logdet_reductions
  factor_J_sum_reduction
  factor_h_sum_reduction
  factor_c_scalar_combinations
  factor_c_sum_reduction
  posterior_precision_symmetrization
  posterior_precision_cholesky
  posterior_natural_solve
  posterior_quadratic
  posterior_logdet_reduction
  route_sum_reduction
```

The count for each factor Cholesky is `cholesky_operation_count(d_f)`.
`factor_precision_solves_A` and `factor_precision_solves_b` use respectively
two triangular solves with `D` and one right-hand-side column per factor;
their union, in factor order and A-then-b order, is the already frozen
`factor_triangular_solves` count. J/h assembly uses respectively
`matrix_multiply_operation_count(D,d_f,D)` and
`matrix_multiply_operation_count(D,d_f,1)` per factor.
`factor_J_sum_reduction` and `factor_h_sum_reduction` contain exactly
`F*D*D` and `F*D` scalar additions because the executable route initializes
zero arrays and applies `+=` once per factor. `factor_c_scalar_combinations`
is exactly `4*F` and `factor_c_sum_reduction` is exactly `F`, using the four
per-factor operations and subsequent accumulation defined with the canonical
table above. `posterior_precision_symmetrization` is exactly `2*D*D` for one
lane-wise addition and one multiplication by `0.5`. The c labels plus the
quadratic/log-diagonal labels are bit-for-bit the c-producing subset of the
frozen full canonical route table. Posterior Cholesky/solve and selected-block
counts use the public Task 3 count functions and exact dimensions. Tests
require the de-duplicated union of the upstream subsets plus
`posterior_precision_symmetrization` and the posterior/logZ-only suffixes to
reconstruct the full canonical table with the exact counts above; shared
factor Choleskys are counted once in that union. A missing, duplicated, or
downstream operation in an upstream table is an error. KL alone uses its
record's exact literal-label order.

The per-lane absolute-summand vectors are also distinct and recomputed from
the normalized factors: J lane `(i,j)` is
`sum_f(abs((A_f.T @ solve(R_f,A_f))[i,j]))`; h lane `i` is
`sum_f(abs((A_f.T @ solve(R_f,b_f))[i]))`; c is the one-element vector whose
value is the sum of the absolute per-factor c contributions. A selected mean
or covariance is the exact absolute value of each selected solve/extraction
output lane, because the final triangular solve is treated as indivisible;
the selected-block proof does not change the covariance lanes. Canonical
`logZ` uses exactly the four-term absolute accumulation already frozen in
`H4OracleRouteAgreement`. Adapter J/h/c/logZ use the same respective oracle
vectors and tables on the adapter side; the frozen H3 reference side remains
an empty table with `abs(reference_lane)`.

The chronological condition tuple for J, h, or c is the factor-covariance
condition number in factor order and contains no posterior-J condition. For a
selected mean, selected covariance, or `logZ`, append
`condition_number(J)` exactly once after the factor conditions. A selected
block proof does not append a second condition because it validates the
already produced operand and cannot affect its lanes. When `F=0` (for a
literal reference operand only), the tuple is `(1.0,)`. These rules are used
independently by the expected-header and observed-group producers.
Every chronological Cholesky input that can affect the operand lanes
contributes its own condition number;
when no such input exists the tuple is exactly `(1.0,)`. No global/run-wide
condition number or value norm is accepted. `solver_produced=True` appears on
every left operand derived from a solver result, including KL, and nowhere
else, so each compared solver operand receives exactly one solver term. The
producer recomputes every vector from full current-problem records and rejects
a caller-supplied receipt, missing factor contribution, changed label/order,
or mismatch with the repaired Task 2 converter shape/count contract.

##### Statistics interfaces

```python
@dataclass(frozen=True, slots=True)
class H4SeedTimingSummary:
    seed_index: int
    seed: int
    information_median_nanoseconds: int
    moment_median_nanoseconds: int
    ratio: float
    log_ratio: float

@dataclass(frozen=True, slots=True)
class H4TimingSummary:
    primary_problem_ids: tuple[str, ...]
    seed_summaries: tuple[H4SeedTimingSummary, ...]
    geometric_mean_ratio: float

@dataclass(frozen=True, slots=True)
class H4PrimaryTimedOrderBalance:
    expected_rows: tuple[tuple[int, int, int], ...]
    observed_rows: tuple[tuple[int, int, int], ...]
    expected_pattern_counts: tuple[tuple[int, int, int], ...]
    observed_pattern_counts: tuple[tuple[int, int, int], ...]
    expected_ab_total: int
    expected_ba_total: int
    observed_ab_total: int
    observed_ba_total: int
    warmup_contribution: Literal[0]
    matches: bool

@dataclass(frozen=True, slots=True)
class H4BootstrapInterval:
    bootstrap_seed: int
    replicate_count: int
    inferential_seed_indices: tuple[int, ...]
    resample_index_shape: tuple[int, int]
    resample_index_dtype: Literal["<i8"]
    resample_index_sha256: str
    statistic: Literal["mean_log_seed_ratio"]
    percentile_method: Literal["linear"]
    percentile_space: Literal["log_then_exp"]
    estimate: float
    lower: float
    upper: float

def summarize_seed_ratios(records: tuple[H4TimingRecord, ...]) -> H4TimingSummary: ...
def summarize_primary_timed_order(
    records: tuple[H4TimingRecord, ...],
    traces: tuple[H4ExecutionTrace, ...],
) -> H4PrimaryTimedOrderBalance: ...
def paired_log_bootstrap_interval(summary: H4TimingSummary) -> H4BootstrapInterval: ...
def decide_h4_interval(interval: H4BootstrapInterval) -> H4IntervalDecision: ...
```

The summary requires exactly 11 positive integer times for each of the 20
primary coupled seeds and rejects any warmup or nonprimary row. Medians use
`statistics.median` on the exact 11-value integer tuple. The estimate is
`exp(fsum(log_ratio)/20)`. Bootstrap uses the exact preregistered NumPy route
and digest above. `decide_h4_interval` contains no inequality.

##### Corrected Task 3 focused steps

- [ ] **Step 1: Amend the preregistration before measurement.** Add the exact
  bootstrap algorithm/header/digest, element-local allowance aggregation,
  bounded scalar-stream domains/counts/witnesses, compact payload ceiling,
  complete solve protocol, typed oracle-route agreement, selected-coordinate
  retention, typed power-policy category, condition coverage, full-repetition
  trace rule, exact postflight schedule/count/digest, and exception-safe
  thread/GC boundaries and incomplete phases above. Do not record an H4
  result. Replace every obsolete flat/free-form applicable-allowance field
  list with the typed `H4ApplicableAllowance` aggregate, its nested witness
  records, exact six invariant counts, and typed inapplicable sentinels; Task 4
  may not begin while the preregistration still describes the removed flat
  schema.

- [ ] **Step 2: Write prerequisite type/config/record tests.** Freeze exact
  dataclass field order, slots/frozen/deep ownership, Task 1 allowance
  arithmetic, public interval boundary cases, the standalone H4-section
  resolver and unchanged H1/H2/H3 prefix set, H4 section hashes, the complete
  nested solve protocol and every cross-field rejection, all config literals
  including 4,096-row/64 MiB bounds and derived
  balance, condition limits, compact condition/coverage stream schemas, trace
  cardinalities, coverage-key order, power-policy order, and restoration
  records. Assert distinct per-problem/global condition-summary types and
  counts, including two exact anchor-only oracle-innovation summaries (one per
  anchor, each with count two) and their exclusion from scaled global totals,
  the Task 3-owned
  allowance group producers and private group
  constructors, exact anchor/scaled repetition ownership, numeric-vector-
  bound independent headers, one-pass six-invariant consumption without
  source retention, the exact two-anchor early identity snapshot that remains
  identical through full finalization, same-object repeats only at that
  boundary, and fail-closed premature/post-scaled/failed/post-finalize calls,
  operand-local oracle/adapter tables, and read-only
  partial-thread-capture semantics. Reject every old flat/full-element
  allowance mapping and every duplicate classifier.

- [ ] **Step 3: Run prerequisite tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h4_problem.py tests/unit/test_config.py tests/unit/test_h4_records.py -q
  ```

  Expected: failures identify the missing typed allowance/decision records,
  standalone H4-section resolver, and verification records; no H4 runner
  prefix is accepted.

- [ ] **Step 4: Implement the prerequisite public types, H4 resolver subset,
  and shared records.** Preserve all earlier H1-H3 prefixes and Task 2 types.

- [ ] **Step 5: Run prerequisite tests for GREEN.** Use the Step 3 command.
  Expected: exact schemas, formulas, prefixes, and record validation pass.

- [ ] **Step 6: Write independent oracle, budget, and statistics tests.** Test
  no `torch`/`vfe4` import; byte parsing; both H3 anchors; hand-authored `D=4`;
  scaled joint initial/global scatter; both logZ routes; exact typed
  `H4OracleRouteAgreement` operand evidence/allowance/strict boundary/pass
  arithmetic, including numerical `passed`, independent `decisive`, and
  `eligible == (passed and decisive)`; selected labels and indices; oriented KL at the exact posterior
  and a perturbed candidate; all
  element-local solver-flag controls; exact six stream counts and digest
  encoding; vectorized multi-chunk arithmetic with only final witness objects;
  late-chunk failure/indecisive controls; repaired visible converter
  Cholesky-count tables with no hidden `_spd`; all six exact producer vector,
  norm, absolute-summand, condition, operation-table, solver-flag, path, and
  independent expected-header rules; complete global and per-problem condition counts; exact
  3-warmup/11-timed trace validation; exact balance; bootstrap first/last row,
  header, digest, log-space percentile; and all interval boundaries.

- [ ] **Step 7: Run numerical-authority tests for RED.**

  ```powershell
  python -m pytest tests/oracle/test_h4_numpy_oracle.py tests/unit/test_h4_budget.py tests/unit/test_h4_statistics.py -q
  ```

  Expected: collection fails because the H4 oracle, budget, and statistics
  modules do not exist.

- [ ] **Step 8: Implement the independent oracle, element-local budget, and
  statistics modules.** No production solver/materializer implementation
  module import is allowed; dependency-light public H4 record types are valid
  producer inputs. Complete the typed preregistration replacement in Step 1
  before Task 4 RED.

- [ ] **Step 9: Run numerical-authority tests for GREEN.** Use the Step 7
  command. Expected: both independent logZ routes, oracle-zero KL, exact
  selected blocks, bounded vectorized local budgets and witnesses, full
  traces, balance, bootstrap bytes, and delegated decisions pass.

- [ ] **Step 10: Commit corrected Task 3 only.**

  ```powershell
  git add docs/preregistrations/2026-07-21-h4-information-cost.md vfe4/types/h4.py vfe4/types/__init__.py vfe4/config/schema.py vfe4/config/resolve.py vfe4/config/__init__.py verification/h4_records.py verification/numpy_oracles/h4_gaussian.py verification/numpy_oracles/__init__.py verification/h4_budget.py verification/h4_statistics.py tests/unit/test_h4_problem.py tests/unit/test_config.py tests/unit/test_h4_records.py tests/oracle/test_h4_numpy_oracle.py tests/unit/test_h4_budget.py tests/unit/test_h4_statistics.py
  git commit -m "test: add H4 oracle budgets and paired statistics"
  ```

---

### Task 4: Build the H4 Preflight, Timed Harness, Gate Decision, and Artifact Payload

##### Task 4 harness and gate

- Create: `verification/h4_gate.py`
- Create: `tests/promotion/test_h4_gate.py`

Task 4 does not reopen the preregistration or H4-section resolver. Task 9 later
adds H5, attaches the already resolved H4 section to `ResolvedConfig` only for
the coupled H1-H5 prefix, and adds runner/artifact wiring while consuming the
H4 records frozen here.
The prerequisite Task 2 repair owns the private integrity implementation and
trusted record factories; they are deliberately absent from the Tasks 3-4
file lists.

#### Corrected Task 4: exact H4 preflight, harness, gate, and artifact

##### Immutable gate-side records

Create these frozen, slotted records in `verification/h4_gate.py`:

```python
H4MaterializedIntegrityPhase = Literal[
    "after_materialization",
    "after_anchor_information",
    "after_anchor_moment",
    "before_timed_batch",
    "after_timed_batch",
    "after_postflight",
]

H4ScaledMaterializedIntegrityCheckpoint = Literal[
    "after_materialization",
    "before_timed_batch",
    "after_timed_batch",
    "after_postflight",
]

@dataclass(frozen=True, slots=True)
class H4MaterializedIntegrityCheck:
    phase: H4MaterializedIntegrityPhase
    expected_tensor_sha256: str
    observed_tensor_sha256: str
    exact_match: Literal[True]

@dataclass(frozen=True, slots=True)
class H4MaterializationIdentity:
    problem_id: str
    problem_sha256: str
    materialization_version: Literal["h4-materialized-problem-v1"]
    protocol_id: Literal["h4-single-pass-v1"]
    tensor_sha256: str
    materialization_count: Literal[1]
    shared_by_identity: Literal[True]
    integrity_checks: tuple[H4MaterializedIntegrityCheck, ...]

@dataclass(frozen=True, slots=True)
class H4CanonicalStreamDigest:
    domain: Literal[
        "vfe4.h4.oracle-evaluation-stream.v1",
        "vfe4.h4.native-result-stream.v1",
        "vfe4.h4.terminal-law-stream.v1",
        "vfe4.h4.native-diagnostic-stream.v1",
    ]
    record_count: int
    scalar_count: int
    byte_count: int
    sha256: str

@dataclass(frozen=True, slots=True)
class H4SelectedMomentSummary:
    name: str
    coordinate_indices: tuple[int, ...]
    dimension: int
    mean_scalar_count: int
    mean_sha256: str
    mean_infinity_norm: float
    covariance_scalar_count: int
    covariance_sha256: str
    covariance_trace: float
    covariance_maximum_absolute_value: float

@dataclass(frozen=True, slots=True)
class H4CompactKLSummary:
    value: float
    trace_term: float
    quadratic_mean_term: float
    minus_dimension_term: float
    candidate_logdet_precision_term: float
    minus_oracle_logdet_precision_term: float
    absolute_summand_accumulation: float
    candidate_condition_number: float
    oracle_condition_number: float
    operation_counts: tuple[tuple[str, int], ...]

@dataclass(frozen=True, slots=True)
class H4CompactResultRecord:
    problem_id: str
    problem_sha256: str
    source_kind: Literal["scaled_pcg64", "h3_anchor"]
    repetition_index: int | None
    arm: H4SolverArm
    native_stream: H4CanonicalStreamDigest
    terminal_stream: H4CanonicalStreamDigest
    oracle_kl_q_to_p: H4CompactKLSummary
    native_complete_objective: float
    terminal_complete_objective: float
    stopping_residual: float
    selected_moments: tuple[H4SelectedMomentSummary, ...]

@dataclass(frozen=True, slots=True)
class H4CompactOracleRecord:
    problem_id: str
    problem_sha256: str
    source_kind: Literal["scaled_pcg64", "h3_anchor"]
    dimension: int
    oracle_stream: H4CanonicalStreamDigest
    canonical_log_normalizer: float
    predictive_log_normalizer: float
    route_agreement: H4OracleRouteAgreement
    selected_moments: tuple[H4SelectedMomentSummary, ...]
    posterior_condition: H4PosteriorConditionRecord
    innovation_conditions: H4ProblemConditionSummary

@dataclass(frozen=True, slots=True)
class H4NativeReplayRecord:
    problem_id: str
    problem_sha256: str
    repetition_index: int
    arm: H4SolverArm
    reference_native_sha256: str
    replayed_native_sha256: str
    diagnostic_stream: H4CanonicalStreamDigest
    innovation_record_count: int
    exact_result_match: Literal[True]

@dataclass(frozen=True, slots=True)
class H4CountingPassRecord:
    problem_id: str
    problem_sha256: str
    arm: H4SolverArm
    reference_repetition_index: Literal[0]
    reference_native_sha256: str
    replayed_native_sha256: str
    reference_terminal_sha256: str
    replayed_terminal_sha256: str
    exact_result_match: Literal[True]
    solver_operations: tuple[H4OperationRecord, ...]
    terminal_conversion_operations: tuple[H4OperationRecord, ...]

@dataclass(frozen=True, slots=True)
class H4MemoryPassRecord:
    problem_id: str
    problem_sha256: str
    arm: H4SolverArm
    reference_repetition_index: Literal[0]
    reference_native_sha256: str
    replayed_native_sha256: str
    exact_result_match: Literal[True]
    memory: H4MemoryRecord

@dataclass(frozen=True, slots=True)
class H4ProblemEvaluation:
    problem_id: str
    problem_sha256: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    oracle: H4CompactOracleRecord
    materialization: H4MaterializationIdentity
    execution_trace: H4ExecutionTrace
    retained_results: tuple[H4CompactResultRecord, ...]
    native_replays: tuple[H4NativeReplayRecord, ...]
    condition_summaries: tuple[
        H4ProblemConditionSummary,
        H4ProblemConditionSummary,
        H4ProblemConditionSummary,
        H4ProblemConditionSummary,
    ]
    counting_passes: tuple[H4CountingPassRecord, H4CountingPassRecord]
    memory_passes: tuple[H4MemoryPassRecord, H4MemoryPassRecord]

@dataclass(frozen=True, slots=True)
class H4ScaledIncompletePhaseRecord:
    problem_id: str
    problem_sha256: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    phase: Literal[
        "warmup",
        "gc_capture",
        "gc_disable",
        "timed_batch",
        "gc_restore",
        "postflight",
    ]
    materialization: H4MaterializationIdentity
    warmup_spans: tuple[H4ArmCallSpan, ...]
    partial_timed_spans: tuple[H4ArmCallSpan, ...]
    garbage_collector: H4GarbageCollectorRecord | None
    postflight_schedule: H4PostflightScheduleSummary | None
    stable_error: str
    obligation: str

@dataclass(frozen=True, slots=True)
class H4ScaledMaterializedIntegrityFailureRecord:
    problem_id: str
    problem_sha256: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    materialization_version: Literal["h4-materialized-problem-v1"]
    protocol_id: Literal["h4-single-pass-v1"]
    materialization_count: Literal[1]
    shared_by_identity: Literal[True]
    checkpoint: H4ScaledMaterializedIntegrityCheckpoint
    expected_tensor_sha256: str
    completed_integrity_checks: tuple[H4MaterializedIntegrityCheck, ...]
    failure_kind: Literal["seam_exception", "digest_mismatch"]
    observed_tensor_sha256: str | None
    seam_error: str | None
    warmup_spans: tuple[H4ArmCallSpan, ...]
    timed_spans: tuple[H4ArmCallSpan, ...]
    garbage_collector: H4GarbageCollectorRecord | None
    postflight_schedule: H4PostflightScheduleSummary | None
    obligation: Literal["materialized_integrity"]

@dataclass(frozen=True, slots=True)
class H4AnchorEvaluation:
    problem_id: str
    problem_sha256: str
    oracle: H4CompactOracleRecord
    materialization: H4MaterializationIdentity
    information_result: H4CompactResultRecord
    information_diagnostic_stream: H4CanonicalStreamDigest
    moment_result: H4CompactResultRecord
    moment_diagnostic_stream: H4CanonicalStreamDigest

@dataclass(frozen=True, slots=True)
class H4UnavailablePhaseRecord:
    phase: Literal[
        "anchor_coupled",
        "anchor_zero_control",
        "scaled_preflight",
        "statistics",
    ]
    reason: str
    obligation: str

@dataclass(frozen=True, slots=True)
class H4PowerPolicyField:
    name: Literal[
        "active_power_scheme",
        "cpu_frequency_governor",
        "energy_performance_preference",
        "low_power_mode",
    ]
    availability: Literal["available", "not_applicable", "unavailable"]
    source: Literal["powercfg", "linux_sysfs", "pmset", "none"]
    value: str | None
    unavailable_reason: str | None

@dataclass(frozen=True, slots=True)
class H4EnvironmentRecord:
    clock_implementation: str
    clock_resolution_seconds: float
    clock_monotonic: bool
    processor: str
    platform: str
    platform_system: Literal["Windows", "Linux", "Darwin", "Other"]
    affinity_cpu_ids: tuple[int, ...] | None
    logical_cpu_count: int
    physical_cpu_count: int | None
    torch_version: str
    numpy_version: str
    torch_config_text: str
    torch_config_sha256: str
    numpy_blas_text: str
    numpy_blas_sha256: str
    cuda_available: Literal[False]
    environment_variables: tuple[tuple[str, bool, str | None], ...]
    power_policy_fields: tuple[
        H4PowerPolicyField,
        H4PowerPolicyField,
        H4PowerPolicyField,
        H4PowerPolicyField,
    ]
    power_policy_category_complete: Literal[True]
    unavailable_fields: tuple[str, ...]
    mandatory_facts_complete: bool

@dataclass(frozen=True, slots=True)
class H4PayloadSizeRecord:
    encoding: Literal["utf8-compact-sorted-key-json-v1"]
    observed_bytes: int
    maximum_bytes: Literal[67108864]
    fixed_point_iterations: int
    within_limit: bool

@dataclass(frozen=True, slots=True)
class H4GateEvaluation:
    schema_version: Literal["h4-gate-evaluation-v1"]
    payload_representation: Literal["bounded-stream-summaries-v1"]
    maximum_payload_bytes: Literal[67108864]
    result: H4GateResult
    h4_config_sha256: str
    anchors: tuple[
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
    ]
    unavailable_phases: tuple[H4UnavailablePhaseRecord, ...]
    problems: tuple[
        H4ProblemEvaluation
        | H4ScaledIncompletePhaseRecord
        | H4ScaledMaterializedIntegrityFailureRecord,
        ...,
    ]
    allowances: tuple[H4AllowanceRecord, ...]
    coverage: tuple[H4CoverageRecord, ...]
    condition_summaries: tuple[
        H4ConditionStreamSummary,
        H4ConditionStreamSummary,
        H4ConditionStreamSummary,
        H4ConditionStreamSummary,
    ]
    raw_timings: tuple[H4TimingRecord, ...]
    primary_timed_order_balance: H4PrimaryTimedOrderBalance | None
    timing_summary: H4TimingSummary | None
    bootstrap_interval: H4BootstrapInterval | None
    interval_decision: H4IntervalDecision | None
    thread_state: H4ThreadStateRecord
    environment: H4EnvironmentRecord
    payload_size: H4PayloadSizeRecord
    bounded_claim: str
    nonclaims: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class H4ValidationArtifact:
    schema_version: Literal["vfe4-validation-h4-v1"]
    payload_representation: Literal["bounded-stream-summaries-v1"]
    maximum_payload_bytes: Literal[67108864]
    gate: Literal["H4"]
    status: GateStatus
    h4_config_sha256: str
    result: H4GateResult
    anchors: tuple[
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
    ]
    unavailable_phases: tuple[H4UnavailablePhaseRecord, ...]
    problems: tuple[
        H4ProblemEvaluation
        | H4ScaledIncompletePhaseRecord
        | H4ScaledMaterializedIntegrityFailureRecord,
        ...,
    ]
    allowances: tuple[H4AllowanceRecord, ...]
    coverage: tuple[H4CoverageRecord, ...]
    condition_summaries: tuple[
        H4ConditionStreamSummary,
        H4ConditionStreamSummary,
        H4ConditionStreamSummary,
        H4ConditionStreamSummary,
    ]
    raw_timings: tuple[H4TimingRecord, ...]
    primary_timed_order_balance: H4PrimaryTimedOrderBalance | None
    timing_summary: H4TimingSummary | None
    bootstrap_interval: H4BootstrapInterval | None
    interval_decision: H4IntervalDecision | None
    thread_state: H4ThreadStateRecord
    environment: H4EnvironmentRecord
    payload_size: H4PayloadSizeRecord
    bounded_claim: str
    nonclaims: tuple[str, ...]

def evaluate_h4(
    config: H4ValidationConfig,
    *,
    h3_coupled_bytes: bytes,
    h3_zero_bytes: bytes,
) -> H4GateEvaluation: ...

def h4_validation_artifact(evaluation: H4GateEvaluation) -> H4ValidationArtifact: ...
def h4_validation_payload(artifact: H4ValidationArtifact) -> dict[str, object]: ...
```

`h4_validation_artifact` is the sole evaluation-to-publication boundary.
`h4_validation_payload` accepts only that already constructed immutable
artifact, never an evaluation or a union, and recursively thaws it without
rebuilding status, coverage, payload-size evidence, or any nested record. The
returned mapping must reserialize to the artifact's exact fixed-point byte
count. Passing an `H4GateEvaluation` is a type/value error. This makes the
Task 9 publication flow identity-preserving:
`evaluation -> h4_validation_artifact(evaluation) ->
h4_validation_payload(artifact)`.

All tuples have fixed canonical order. The two anchor slots are always coupled
then zero-control and contain either the completed typed evaluation or an
explicit unavailable-phase record whose phase is respectively
`anchor_coupled` or `anchor_zero_control`; no placeholder numeric value is
fabricated. Anchor-phase records are valid only in their matching anchor slots.
The top-level `unavailable_phases` is an ordered duplicate-free tuple containing
at most one `scaled_preflight` record followed by at most one `statistics`
record. Those two phases are valid only in this top-level field. There is no
generic `timing` or `postflight` unavailable variant: an execution failure after
a scaled problem begins is owned by its typed `H4ScaledIncompletePhaseRecord`,
while scaled materialized-integrity failures retain their dedicated typed
record. Both `H4GateEvaluation` and `H4ValidationArtifact` independently enforce
these placement and order rules. `allowances` has exactly the six public
invariant names. `coverage` has exactly the nine coverage names above.
`problems` has exactly 120 scaled evaluations in traversal order when timing
completes. A complete problem has 22 compact result records, 22 replay digest
wrappers, one compact oracle, four exact condition-stream summaries, two
counting passes, and two memory passes. The artifact never owns a full
`H4SolverResult`, full `H4TerminalLaw`, full `D x D` oracle array, replayed
duplicate, complete coverage-key tuple, or complete allowance-element tuple.
The artifact record is the only source for the JSON mapping; serialization
recursively thaws owned compact records and checks the exact top-level field
order. Broad untyped containers are forbidden.

If a scaled execution fails during or after its warmups, `problems` contains
the exact completed traversal prefix followed by one and only one
`H4ScaledIncompletePhaseRecord`, whose indices are the next canonical problem
and whose `phase` is the first incomplete boundary. Traversal then stops. The
record has a strict canonical prefix of zero through six warmup spans and zero
through 21 timed arm-call spans. The GC record is absent exactly for a warmup
failure and otherwise present; the postflight summary is present exactly when
postflight began. Its `stable_error` is nonempty and capped
at 512 Unicode code points. It contributes no fabricated
`H4TimingRecord`; only fully closed information/moment pairs from complete
problems enter `raw_timings`. A warmup exception is phase `warmup`; a GC
capture failure is phase `gc_capture`; a disable/effective-state failure is
`gc_disable`; a timed arm exception is `timed_batch`; any inexact GC
restoration is `gc_restore`; and any later failure is `postflight`. Every
branch is `INCONCLUSIVE` with the phase-specific repair obligation after every
applicable thread and GC restoration has been attempted.

The phase-to-obligation mapping is exact:

```text
warmup -> "complete all six H4 warmup arm calls without exception"
gc_capture -> "capture cyclic GC state before H4 timing"
gc_disable -> "disable and verify cyclic GC before H4 timing"
timed_batch -> "complete all 22 H4 timed arm calls and restore process-global state"
gc_restore -> "restore exact prior cyclic GC state after H4 timing"
postflight -> "complete exact H4 postflight schedule and release full problem objects"
```

The constructor rejects any other obligation for the selected phase.

Scaled gate-owned integrity failures use the separate
`H4ScaledMaterializedIntegrityFailureRecord`; they never masquerade as a GC,
timing, postflight, or generic unavailable phase. Its checkpoint fixes the
exact successful-check prefix and execution evidence:

```text
after_materialization -> checks (), warmups 0, timed 0, GC None, schedule None
before_timed_batch -> checks (after_materialization), warmups 6, timed 0,
                      GC None, schedule None
after_timed_batch -> checks (after_materialization,before_timed_batch),
                     warmups 6, timed 22, GC complete, schedule present/incomplete
after_postflight -> checks (after_materialization,before_timed_batch,
                            after_timed_batch), warmups 6, timed 22,
                    GC complete, schedule present/incomplete
```

The before-timed checkpoint occurs after all warmups and before GC capture.
The after-timed checkpoint occurs only after the timed guard exits and GC has
been restored. Every completed check has its exact checkpoint phase and
matching expected/observed digest. `failure_kind="seam_exception"` requires
`observed_tensor_sha256=None` and a nonempty, 512-code-point-capped
`seam_error`; `failure_kind="digest_mismatch"` requires a valid observed
digest different from the expected digest and `seam_error=None`. Both require
the one closed obligation `materialized_integrity`.

An `after_materialization` failure during all-problem preflight makes
`problems` exactly the single failure record for its canonical problem index;
previously preflighted problems are not fabricated as completed evaluations.
A later checkpoint failure follows the exact completed-problem traversal
prefix and is its final record. Either shape stops traversal and statistics,
is `INCONCLUSIVE`, and serializes through both `H4GateEvaluation` and
`H4ValidationArtifact` because both problem unions accept the dedicated
carrier.

Each problem's four `H4ProblemConditionSummary` values are in exact order
`oracle_posterior`, `terminal_posterior`, `oracle_innovation`,
`moment_innovation`, with expected counts `1`, `22`, `T`, and `11*T`.
The top-level `condition_summaries` field of both `H4GateEvaluation` and
`H4ValidationArtifact`, and the identically positioned JSON payload field, is
the exact ordered tuple of four global `H4ConditionStreamSummary` values with
counts `120`, `2640`, `2120`, and `23320`. Their global counts must equal the
sums of the compact per-problem counts and the four condition coverage totals
before the corresponding global coverage record can be complete. The global
digest is accumulated directly over canonical records and is not derived by
hashing per-problem summary digests.

The oracle-evaluation stream header is one length-prefixed compact sorted-key
JSON mapping with exactly `schema_version,problem_id,problem_sha256,
source_kind,seed,kind,horizon,d_z,d_m,dimension,coordinate_order,factor_ids,
selected_blocks`; each selected-block header is exactly
`name,coordinate_indices,dimension` in declared order. Its numeric lanes are,
in order, row-major `precision`, `natural`, `constant`, `mean`, row-major
`covariance`, `canonical_log_normalizer`, `predictive_log_normalizer`, then for
each selected block its mean and row-major covariance. Thus
`record_count=1` and
`scalar_count=2*D*D + 2*D + 3 + (T+2)*(s+s*s)`, where `s=d_z+d_m`.
Route-agreement allowance fields, operand-evidence tables, condition
diagnostics, and their derived booleans are excluded from these lanes because
they are independently retained and fieldwise checked in the typed route and
condition records; no other `H4OracleEvaluation` numeric field is excluded.

The native-diagnostic stream header is one length-prefixed compact sorted-key
JSON mapping with exactly `problem_id,problem_sha256,protocol_id,arm,
factor_count,replayed_native_sha256,innovation_records`; each innovation
header is exactly `factor_id,time_index,parent_coordinate_indices,
innovation_dimension` in factor-schedule order. Its lanes for each innovation
are exactly `minimum_eigenvalue,maximum_eigenvalue,condition_number,
minimum_cholesky_pivot`. Therefore `record_count=N_innovation` and
`scalar_count=4*N_innovation`: information has `0/0`, a scaled moment replay
has `T/4*T`, and an H3 moment replay has `2/8`. The replayed native numeric
state is excluded because `replayed_native_sha256` binds the separate native
stream; literal `finite/spd/replay_matches_result` proofs are excluded because
construction requires them to be true. A missing, extra, reordered, or
renamed header key/lane is a stream mismatch. `byte_count` is the total bytes
fed to SHA-256, including literal domain plus NUL, header length/header, and
numeric lane bytes.

`H4CanonicalStreamDigest` uses its literal ASCII domain plus `b"\x00"`, then
unsigned eight-byte big-endian-length-prefixed compact sorted-key UTF-8
headers and raw canonical scalar lanes. Floating lanes are contiguous
little-endian `<f8`; integer lanes are
little-endian `<i8`; strings are UTF-8 with unsigned eight-byte big-endian
lengths; booleans are `u1`; and `None` is a dedicated zero-byte tag. Record and
scalar counts and hashed byte count are checked while streaming. Native-result
order is exact public field order followed by row-major tuple scalars;
terminal order is `arm,h,J,mean,selected_moments,complete_objective,
stopping_residual`, with each selected label and coordinate-index tuple in its
header followed by means/covariances in declared order. The expected native
scalar counts are `D^2+2D+1` for information and
`D^2+D+1` for moment; terminal count is
`D^2+2D+(T+2)*(8+8^2)+2`. The summary constructor rejects any mismatch.

Each `H4SelectedMomentSummary` retains the exact ordered
`coordinate_indices`, requires them to be unique, strictly increasing,
in-range, and of length `dimension`, hashes the complete mean and covariance
arrays, checks counts `8` and `64` for scaled problems (and `2` and `4` for
anchors), and retains only bounded scalar summaries. Oracle and every arm must
carry identical indices for a selected label; equal numerical arrays under
different indices are a protocol mismatch. Per-result native and terminal
objectives, stopping residual, and every selected label remain directly
visible. Each result also retains the full signed oriented `KL(q || p*)` term
summary used by its zero comparison. Exact native/terminal stream hashes prove
replay/count/memory equality without retaining duplicate matrices.

Selected mean and covariance hashes use domains
`vfe4.h4.selected-mean.v1` and `vfe4.h4.selected-covariance.v1`, followed by a
length-prefixed header binding problem hash, repetition/anchor identity, arm,
label, the complete ordered coordinate-index tuple, parent dimension, selected
dimension, and scalar count, then contiguous `<f8` row-major values. The
native/terminal/oracle stream header also includes the index tuple before its
selected numeric lanes. `H4CanonicalStreamDigest.scalar_count` counts numeric
float lanes only; the encoded index metadata contributes to `byte_count` and
SHA-256 but does not change the frozen numeric scalar-count formulas. Mutating
only an index changes both selected hashes and the enclosing stream digest.

`H4CompactOracleRecord.route_agreement` is the exact typed record constructed
from the full oracle. Its problem identity and both operand values must match
the compact oracle identity and its two retained log normalizers. Compacting
or serializing an oracle without the agreement, or replacing it with a scalar
residual, is rejected.

`H4CompactOracleRecord.innovation_conditions` has the same problem identity
as the compact oracle. A scaled oracle requires its exact per-problem
`oracle_innovation` summary with count `T`; an H3 anchor oracle requires the
anchor-only `oracle_innovation` summary with count `2`. An anchor summary never
enters the scaled global condition summaries or coverage sums.

`H4CompactResultRecord` requires `source_kind="scaled_pcg64"` with repetition
`0..10`, or `source_kind="h3_anchor"` with `repetition_index=None`; no anchor
is assigned a fake timed repetition. `H4ProblemEvaluation` accepts only the
former exact 22-record order, while `H4AnchorEvaluation` accepts exactly one
information and one moment record of the latter kind.

Constructors cross-check every repeated identity and value: evaluation and
artifact status equal `result.status`; allowance tuple values equal the six
`result.allowances_by_invariant` values in public key order; each nested
problem/hash/protocol/arm/factor count agrees; each terminal/replay/count/memory
stream count/hash agrees with its wrapper; trace and raw timing
identities/durations agree;
every integrity check's expected and observed digest equals its enclosing
materialization's `tensor_sha256`; and the H4 config hash equals the supplied
standalone typed section. Task 9 separately binds that same hash to the full
coupled config and manifest. A completed scaled materialization has the exact
phase tuple `("after_materialization", "before_timed_batch",
"after_timed_batch", "after_postflight")`. A completed anchor has the exact
phase tuple `("after_materialization", "after_anchor_information",
"after_anchor_moment")`. No phase is duplicated or reordered. Literal `True`
proof fields are emitted only by module-private checked factories, never
accepted as unaudited caller assertions.

Full oracle/native/terminal objects are live only for the current problem's
postflight. After every scalar comparison, condition, digest, replay equality,
and witness is closed, the gate constructs the compact records and drops all
22 result references, replay duplicates, and the rehydrated full oracle before
advancing. The tests retain weak references or equivalent lifetime sentinels
to prove this per-problem release; no complete-run list of full numerical
objects may exist.

Missing memory values remain explicit in `H4MemoryRecord.unavailable_fields`
and never influence the primary status. A missing/mismatched memory pass
identity is a protocol incompleteness, but an unavailable platform metric is
not. Operation-count magnitudes and memory magnitudes are secondary; they do
not rescue or overturn the primary decision.

`environment_variables` is in the exact order `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS`. `platform_system` is derived from
`platform.system()` by mapping every value outside Windows/Linux/Darwin to
`Other`. `power_policy_fields` is always present in the exact
order `active_power_scheme`, `cpu_frequency_governor`,
`energy_performance_preference`, `low_power_mode`. On Windows only the first is
applicable and its source is `powercfg`; on Linux the middle two use sorted
per-CPU `cpu_id=value` observations from sysfs; on macOS the last uses `pmset`;
all other OS/field pairs are explicit `not_applicable/source="none"` records.
An applicable probe either has `available`, a nonempty canonical value, and no
reason, or `unavailable`, no value, and a nonempty stable reason. No field may
be omitted, duplicated, reordered, or represented by an empty string. A
canonical value is capped at 4,096 Unicode code points; a larger observation is
recorded as unavailable with a bounded reason rather than silently truncated.

All `stable_error`/unavailable-reason strings use
`module.qualname + ": " + str(error)`, normalize CRLF/CR to LF, replace NUL
with `U+FFFD`, and retain at most the first 512 Unicode code points. This same
bounded formatter is used for thread, GC, integrity, power-policy, and
incomplete-phase errors.

Power-policy capture happens with the other environment probes before the
first warmup and outside every timer. Platform absence is allowed because the
global contract says "when available"; it remains visible as `unavailable`
and does not fabricate a value. Omission/malformed ordering makes
`power_policy_category_complete` impossible and H4 `INCONCLUSIVE`. The exact
four records serialize as ordinary ordered typed children of
`H4EnvironmentRecord`; there is no platform-dependent key set.

Process affinity is captured by the one shared `process_cpu_affinity()`
provider: `os.sched_getaffinity(0)` where present, Win32
`GetProcessAffinityMask` through `ctypes` on Windows, and optional `psutil` only
as the last fallback. A successful provider result is a sorted, unique,
nonempty tuple of real CPU IDs. The provider never substitutes
`range(os.cpu_count())`. If every provider is unavailable, the serialized
`affinity_cpu_ids` is JSON null and `unavailable_fields` contains
`"affinity_cpu_ids"`; the tuple/null value and unavailable marker must agree in
both directions. Because affinity is mandatory, that state makes
`mandatory_facts_complete=False` and H4 is `INCONCLUSIVE` before any problem or
timing record is produced. Physical CPU count remains conditionally unavailable
without itself invalidating completeness. Clock identity/resolution/monotonicity,
processor/platform/platform-system category, process affinity, logical count,
library versions, configuration texts/digests, CUDA state, and every variable
presence/value are mandatory for `mandatory_facts_complete=True`.

##### Preflight and materialized API boundary

Inside the outer thread-restoration boundary, capture environment facts,
validate both H3 byte hashes, adapt both anchors once, generate all 120 scaled
neutral problems once, and emit canonical bytes once. For each neutral problem:

1. pass canonical bytes to `evaluate_h4_oracle`, close both routes and all
   oracle conditions, compute the full oracle stream digest/compact summary,
   then release the full oracle arrays;
2. retain the canonical bytes and compact oracle summary, not the full oracle;
3. call `materialize_h4_problem(neutral, config.solve_protocol)` exactly once;
4. call Task 2's private, untimed
   `_assert_h4_materialized_integrity(materialized)` seam and require its
   returned digest to equal `materialized.tensor_sha256`;
5. retain only a `H4MaterializationIdentity` plus its immutable integrity
   checks in the artifact;
6. pass the same in-memory `H4MaterializedProblem` object by `is` to every
   warmup, timed, diagnostic, counting, memory, and conversion operation, and
   pass the one `config.solve_protocol` object by `is` to every Task 2 entry
   point whose signature accepts a protocol.

All 120 oracle summaries, condition summaries, materializations, and initial
integrity checks must close before the first scaled warmup. At each problem's
postflight, re-evaluate the independent oracle from the retained canonical
bytes, require its complete `oracle_stream` count/byte-count/SHA-256 to equal
the preflight summary, require its `H4OracleRouteAgreement` to equal the
preflight compact agreement field-for-field, use that one full oracle for
every retained-scalar comparison, and release it with the 22 full results at
problem completion.
This performs deterministic untimed oracle work twice but prevents a
complete-run collection of nested full `D x D` tuple arrays.

The production calls are exactly:

```python
solve_information_form(materialized, protocol, linalg)
solve_moment_form(materialized, protocol, linalg)
to_common_terminal_law(materialized, result, linalg)
evaluate_h4_native_diagnostics(materialized, result, null_linalg)
```

The Task 2 seam has the exact private signature:

```python
def _assert_h4_materialized_integrity(
    materialized: H4MaterializedProblem,
) -> str: ...
```

It validates the exact materialized structure, ownership/nonaliasing rules,
metadata, and current raw private-tensor bytes, recomputes the domain-separated
materialized digest, raises on any mismatch, and otherwise returns that
lowercase digest. The gate still compares the return explicitly with
`tensor_sha256`; neither a stale stored digest nor a no-op seam can close the
check.

For each scaled problem, the gate invokes the seam explicitly at exactly four
artifact checkpoints:
immediately after materialization; after all three warmup pairs and immediately
before entering the timed guard; immediately after leaving the guard and
before constructing timing/trace records; and after the complete fixed-order
postflight. Thus the pre-batch checkpoint also closes the warmup mutation
window, the post-batch checkpoint closes all 22 timed arm calls as one batch,
and the final checkpoint closes conversion, replay, counting, and memory. For
each anchor, invoke it immediately after materialization and after completing
the information and moment arm respectively. Every call and digest comparison
is outside every timer. The solver entry path must not hash, clone, or traverse
all materialized bytes while the timing guard is active.

Task 2's untimed converter/diagnostic entry validation may also call the same
seam during postflight. Those internal calls remain outside every timer and do
not replace, increment, or weaken the four gate-owned artifact checkpoints;
the native timed solver entry uses only Task 2's metadata-only identity,
storage-pointer, shape, and tensor-version receipt and never recomputes the
byte digest.

Any seam exception or digest mismatch stops that branch before statistics.
An anchor uses its matching typed anchor unavailable record; a scaled problem
uses `H4ScaledMaterializedIntegrityFailureRecord` at the exact failing
checkpoint. Both carry the exact closed obligation `materialized_integrity`
and make the gate `INCONCLUSIVE`; neither is treated as an invariant miss. A
malicious in-place edit through
`materialized._factor_matrices`, `_factor_targets`, or
`_factor_covariances` therefore cannot survive either the timed batch or
postflight boundary.

Task 4 consumes the public `H4SolverResult`, native-state, selected-moment, and
`H4TerminalLaw` records exactly as returned. Task 2 owns the exact private
names/signatures of the proven-SPD factories that construct those public
records after facade/native proof. Direct public constructors retain their
full untrusted-input validation, but Task 2's solver and converter return paths
must not call `_spd`, eigendecomposition, inversion, or another hidden cubic
SPD proof merely to package already-proven values. The converter's repaired
algorithm does perform the explicit facade-counted selected-block and
moment-derived-precision Choleskys frozen above before invoking the private
factory; those visible operations are not constructor validation. Tasks 3-4
do not import or invoke the factories; their contract is observable through
the public return paths and enforced by focused tests that make `_spd` fail if
it is reached there while still requiring the explicit facade counts.

No gate call supplies `H4NeutralProblem` to a solver or rematerializes per arm,
repetition, diagnostic, count, or memory pass. The NumPy oracle never receives
the materialized object.

Warmup and timed solver calls, terminal conversion, native replay, and memory
passes use a correctly problem/arm-bound `NullOperationRecorder` facade.
Exactly the two explicit operation passes per problem use a
`CountingOperationRecorder` facade. A facade is never rebound or reused for a
different problem or arm.

All 120 oracle posterior records and 2,120 oracle innovation records must be
eligible before the first warmup. The H3 anchors close before scaled timing;
a gate calls the unified accumulator's `anchor_identity_record()` immediately
after the second anchor and uses that record for the early decision. A decisive
anchor miss follows the already frozen early-FAIL schema without consuming a
scaled allowance source. An oracle
route mismatch, invalid config, incomplete environment, or out-of-envelope
preflight is `INCONCLUSIVE`, never repaired.

##### Exception-safe execution

The implementation has an outer shape equivalent to the following only after
both read-only captures succeed:

```python
try:
    set_attempted = True
    torch.set_num_threads(1)
    effective = torch.get_num_threads()
    if effective != 1:
        raise H4EligibilityError("intra-op thread verification failed")
    # preflight, anchors, all problem batches and postflight work
finally:
    restore_attempted = True
    try:
        torch.set_num_threads(prior_threads)
        restored = torch.get_num_threads()
    except Exception as error:
        restoration_error = stable_error(error)
```

The actual guard captures intra-op and then inter-op counts inside a protected
read-only preliminary step. Any capture failure populates `capture_error`,
suppresses set/timing, and returns `INCONCLUSIVE` without restoration because
no mutation occurred. It may retain an intra-op value captured before an
inter-op getter failure, but that partial value cannot authorize a setter.
After both captures succeed, any set attempt makes restoration mandatory,
even if set or verification fails.

Each problem's timed section has an inner shape equivalent to:

```python
gc_capture_attempted = True
try:
    prior_gc_enabled = gc.isenabled()
except Exception as error:
    prior_gc_enabled = None
    gc_capture_error = stable_error(error)
try:
    if prior_gc_enabled is None:
        raise H4EligibilityError("cyclic GC state capture failed")
    gc_disable_required = prior_gc_enabled
    if gc_disable_required:
        gc_disable_attempted = True
        try:
            gc.disable()
        except Exception as error:
            gc_disable_error = stable_error(error)
            raise H4EligibilityError("cyclic GC disable failed") from error
    try:
        gc_disabled_during_batch = not gc.isenabled()
    except Exception as error:
        gc_effective_state_capture_error = stable_error(error)
        raise H4EligibilityError("cyclic GC effective-state capture failed") from error
    if not gc_disabled_during_batch:
        raise H4EligibilityError("cyclic GC remained enabled")
    # enter timed guard; run pair_index 3..13 only
finally:
    if prior_gc_enabled is not None:
        gc_restore_attempted = True
        try:
            if prior_gc_enabled:
                gc.enable()
            else:
                gc.disable()
            gc_restored_enabled = gc.isenabled()
        except Exception as error:
            gc_restoration_error = stable_error(error)
```

All GC record fields are initialized before capture. A complete timed batch
requires `capture_error`, `disable_error`,
`effective_state_capture_error`, and `restoration_error` all `None`;
`prior_enabled` and `restored_enabled` are booleans and equal;
`disable_required == prior_enabled`; `disable_attempted == prior_enabled`;
`disabled_during_batch is True`; `restore_attempted is True`; and
`restored_exact_prior_state is True`. On capture failure, every later observed
state is `None`, both later attempt fields are false, and the scaled incomplete
phase is `gc_capture`. On a later failure, fields retain exactly the phase-valid
prefix above; no false boolean substitutes for an unavailable observation.

Warmup pairs `0,1,2` run before GC setup. Timed pairs are exactly `3..13`.
Before entering the timed guard, preallocate 22 result-reference slots and 44
integer start/end slots. Between calls perform only `perf_counter_ns`, result
reference assignment, and integer slot assignment. Synthesize trace and
`H4TimingRecord` objects after leaving the guard and restoring GC. No focused
test lowers the repetition count; real tests select one or two problems while
keeping all three warmup and eleven timed pairs.

##### Fixed postflight order and coverage

For each completed problem, after the timed batch:

1. record the `after_timed_batch` materialized-integrity event;
2. convert all retained results in `(repetition 0..10, information, moment)`
   order using null-bound facades, compute native/terminal stream digests and
   selected summaries, and retain the full terminal objects only through this
   problem's postflight;
3. replay diagnostics in the same repetition/arm order, require the replayed
   native digest to equal its retained native digest, update the
   diagnostic/innovation streams, and discard each replayed full result
   immediately;
4. compute terminal posterior conditions in repetition/arm order, then moment
   innovation conditions in repetition order and declared observation-factor
   order;
5. re-evaluate the full oracle from canonical bytes and require its stream
   digest to equal preflight;
6. construct and validate its typed `H4OracleRouteAgreement`;
7. for each `(repetition 0..10, information, moment)` result, append allowance
   groups in the exact order `kl_to_zero`, `h`, `J`, then
   `(selected_mean,selected_covariance)` for each selected label in declared
   order, then `objective`;
8. run one counting pass for information and one for moment, each from the same
   materialized object; pass the replayed result through terminal conversion on
   the same counting facade so the repaired converter Choleskys are visible,
   and require both native and terminal digests to equal retained repetition 0;
9. run one memory pass in information-then-moment order with the same equality
   requirement;
10. record the `after_postflight` materialized-integrity event; and
11. compact the problem, clear all full oracle/native/terminal/replay
    references, and only then advance.

The exact event-key count for a problem of horizon `T` is:

```text
after_timed_batch integrity                         1
terminal conversion                               22
native diagnostic replay                          22
terminal posterior condition                      22
moment innovation condition                     11*T
oracle rehydration                                  1
oracle route agreement                              1
equivalence groups              22*(4 + 2*(T+2))
operation passes                                    2
memory passes                                       2
after_postflight integrity                          1
stream compaction                                    1
total                                      251 + 55*T
```

Thus each `T=7`, `T=15`, and `T=31` problem requires exactly 636, 1,076, and
1,956 keys, respectively, and the 120-problem scaled suite requires exactly
`40*(636+1076+1956) = 146720` keys. A module-private independent expected-key
iterator generates the order above from only the resolved traversal and factor
schedule. The observed path streams one key immediately around each actual
call. Both digests begin with
`b"vfe4.h4.postflight-event-key-stream.v1\x00"`, the problem SHA-256, and a
zero byte, then append every compact sorted-key UTF-8 event mapping as an
unsigned eight-byte big-endian length plus bytes. Event start/end timestamps
are checked for nonnegative, monotone, nonoverlapping post-timed spans but are
not part of the identity digest. Chunking is forbidden: one logical call emits
one key.

`H4PostflightScheduleSummary.complete` is true exactly when expected and
observed counts and hashes agree, both first-mismatch witnesses are absent,
and the timing-violation count is zero. On mismatch it retains only the lowest
event-index expected/observed witness; on timing failure it retains only the
lowest event-index timing witness. Constructors independently require the
per-horizon counts above, while the gate/artifact require 120 complete
summaries, global event count 146,720, and their canonical traversal-order
digest before statistics. No list or tuple of all event keys survives
postflight.

Because the approved `measure_untimed_memory` returns only `H4MemoryRecord`,
its callable stores exactly one solver return in a gate-local one-slot holder;
the gate rejects zero or multiple writes, hashes that one value, compares it
to the retained repetition-zero native digest, and drops it after freezing the
compact `H4MemoryPassRecord`.

Information native diagnostics have the exact empty innovation tuple. Moment
replay contains every scheduled observation. Counting and memory use their own
facades/passes and do not add records to diagnostic replay or timed traces.
The counting recorder is snapshotted immediately after the solver and again
after conversion; the first snapshot becomes `solver_operations` and the
strict ordered delta becomes `terminal_conversion_operations`. For `S=T+2`,
the latter must expose one native-`J` Cholesky plus `S` selected-block
Choleskys for information, and one native-covariance Cholesky, one
derived-precision Cholesky, plus `S` selected-block Choleskys for moment,
together with the converter's other declared facade operations. Thus the exact
converter totals are `S+1` and `S+2`.

After all problems, finalize the four condition accumulators, all nine
coverage key streams, all 120 postflight schedule summaries and their
146,720-event aggregate, the unified allowance accumulator's six private
invariant states, and the exact primary
timed balance before statistics. Their exact counts and stream digests must
close even when every retained numerical scalar passed; no witness can replace
coverage. Only after every condition, schedule, and allowance is eligible may
the gate compute seed statistics, bootstrap, and the public Task 1 interval
decision.

##### Status and artifact authority

Status precedence is exact:

1. A sole decisive H3 anchor miss with successful process-global restoration
   uses the existing finite pre-timing `FAIL` record and five exact unavailable
   measurements. The typed anchor-miss/restoration exception instead proceeds
   to item 2.
2. Any config/protocol/environment/restoration/materialized-integrity/
   traversal/trace/table/coverage/stream-count/stream-digest/payload-size,
   oracle-route, condition, finiteness, or allowance-decisiveness ambiguity is
   `INCONCLUSIVE` with a named obligation.
3. Any decisive element-local oracle comparison miss is `FAIL`.
4. Otherwise use `H4IntervalDecision.status_if_other_invariants_eligible`:
   support is `PASS`, no-support is `FAIL`, crossing/boundary is
   `INCONCLUSIVE`.

Task 4 never reimplements `0.80` inequalities. It constructs invariant 16 and
the gate result directly from the same `H4IntervalDecision` record.

`h4_validation_payload` contains only the exact versioned artifact fields
above and their typed descendants. It thereby publishes exact config hashes,
anchor/problem/materialization identities, both oracle logZ routes, selected
indices and per-result summaries, native/terminal/oracle/diagnostic stream
counts and digests, condition/coverage stream counts and witnesses,
element-local allowance counts/digests/maxima/witnesses, full traces and raw
timings, bootstrap parameters and index digest, decision,
restoration/environment/power policy, replay/count/memory hash equality,
bounded claim, and H5-H8/training nonclaims. It publishes no complete `D x D`
repetition matrix or repeated replay payload and cannot invent an unavailable
finite value.

Before `evaluate_h4` finalizes status, it constructs the exact prospective
`H4ValidationArtifact` mapping and solves the self-inclusive payload-size field
by iteration: begin with `observed_bytes=0`, serialize compact sorted-key JSON
with `allow_nan=False` to UTF-8, replace `observed_bytes` by the new byte
length, and repeat until the stored value equals the serialized length. The
digit-length fixed point must converge within four iterations. The resulting
`H4PayloadSizeRecord` is identical in evaluation and artifact; its constructor
requires `1 <= fixed_point_iterations <= 4` and
`within_limit == (observed_bytes <= maximum_bytes)`. A value above
67,108,864 forces `INCONCLUSIVE` with obligation
`reduce H4 validation payload below 67108864 bytes without dropping scalar
coverage`; rebuilding that status is included in the final fixed-point pass.
`h4_validation_payload` returns the compact mapping only when its independently
reserialized byte length equals the record and `within_limit=True`; otherwise
publication is refused. A complete 120-problem synthetic
max-shape regression must serialize below that ceiling, contain no keys named
`native_result`, `terminal_law`, `replayed_result`, `precision`, or
`covariance` except bounded selected-summary scalar fields, and show peak
construction memory bounded by compact-record count rather than 79,832,024
comparison elements. A late scalar mutation in the last J chunk must change
the digest/witness/status while all expected coverage counts remain exact.

##### Corrected Task 4 focused steps

- [ ] **Step 1: Write exact gate and artifact tests first.** Assert every
  record field/order/type, deep immutability, exact complete nested protocol
  ownership, two H3
  anchors with their exact two-record oracle-innovation summaries excluded
  from scaled global condition totals, the exact 184-element early anchor
  snapshot and no scaled-source consumption on decisive anchor failure,
  120-problem complete branch,
  materialization count/identity,
  3/11 traces, guard exclusions, exact 146,720-event postflight schedule
  counts/order/digests, distinct compact per-problem and top-level global
  posterior/innovation condition summaries/counts/digests, exact
  oracle-evaluation and native-diagnostic headers/lane orders/scalar counts,
  typed retained
  oracle-route agreements, replay/count/memory digest equality, exact
  six allowance group counts/digests/witnesses, compact per-result summaries,
  persisted selected coordinate indices and index-bound hashes,
  public decision identity, status precedence, restoration, ordered power
  policy, payload-size ceiling, artifact-only payload signature and exact
  no-reconstruction round trip, no-per-scalar-object behavior, and exact
  artifact keys. Include read-only intra/inter-op capture failures (including
  partial capture with no set/restore), mandatory restore after every set
  attempt, GC capture/disable/effective-state/restore exceptions,
  every typed scaled incomplete phase, and both integrity failure kinds at all
  four scaled checkpoints with the exact `materialized_integrity` carrier and
  obligation, without fabricated records.

- [ ] **Step 2: Run the Task 4 test for RED.**

  ```powershell
  python -m pytest tests/promotion/test_h4_gate.py -q
  ```

  Expected: collection fails because `verification.h4_gate` does not exist.

- [ ] **Step 3: Implement config-bound preflight and anchor closure.** Capture
  environment, enter the outer thread boundary, verify H3 bytes, generate and
  hash all problems, run both NumPy routes, require
  `route_agreement.eligible is True` and
  `route_agreement.eligible == (route_agreement.passed and
  route_agreement.decisive)`, retain it in each compact oracle, and
  compact/release every full oracle. Materialize each problem once with
  `config.solve_protocol`, record
  initial integrity, and close every preflight condition before any warmup.

- [ ] **Step 4: Implement the guarded warmup and timed harness.** Use exact
  independent-index parity, preallocated slots, nested GC restoration, and
  post-batch trace/timing construction. Retain every GC capture/disable/error
  field and construct the exact scaled incomplete-phase union branch after
  restoration. Construct the dedicated scaled integrity-failure union branch
  at its exact checkpoint rather than routing it through a generic unavailable
  phase. Never reduce repetitions.

- [ ] **Step 5: Implement fixed-order conversion, replay, condition,
  equivalence, count, memory, and compaction postflight.** Require exact
  reference-result stream equality, coordinate-index identity, complete
  canonical coverage, bounded allowance chunks, the independent exact
  `251+55*T` event-key stream and 146,720-event global closure, final
  integrity, and release of full current-problem objects.

- [ ] **Step 6: Implement typed decision and artifact construction.** Use only
  the public Task 1 interval decision and exact status precedence. Deeply thaw
  only the already constructed `H4ValidationArtifact` at JSON serialization,
  reject an evaluation passed to `h4_validation_payload`, preserve artifact
  identity without reconstruction, and enforce the 64 MiB ceiling.

- [ ] **Step 7: Run the Task 4 test for GREEN.**

  ```powershell
  python -m pytest tests/promotion/test_h4_gate.py -q
  ```

  Expected: mocked-clock branches plus a real reduced-problem/full-repetition
  fixture prove all schemas, boundaries, restoration, guard, coverage,
  scalar-stream equivalence, route agreement, selected-index retention,
  postflight schedule, scaled incomplete-phase compaction lifetime, bounded
  chunks, Task 3-owned allowance group producers without constructing private
  groups or budget provenance in Task 4, one-pass source consumption and
  numeric-vector-bound expected headers, exact power policy serialization,
  payload size, and artifact
  behavior. No test result is H4 promotion evidence.

- [ ] **Step 8: Commit corrected Task 4 only.**

  ```powershell
  git add verification/h4_gate.py tests/promotion/test_h4_gate.py
  git commit -m "test: add the H4 information cost gate"
  ```

---

### Task 5: Freeze H5 Types, Raw Specification, Snapshot Ownership, Hashes, and Complete Dependency Graph

**Files:**

- Create: `.gitattributes` containing exactly `vfe4/validation/fixtures/h5_conditional_update_v1.json -text` so Git never rewrites the raw hashed fixture bytes.
- Create: `vfe4/types/updates.py`
- Modify after H4 export serialization: `vfe4/types/__init__.py`
- Create: `vfe4/types/h5_schema.py`
- Create: `vfe4/objective/dependency_graph.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `vfe4/validation/fixtures/h5_conditional_update_v1.json`
- Create: `vfe4/validation/h5_update_spec.py`
- Create: `tests/unit/test_h5_update_types.py`
- Create: `tests/unit/test_h5_objective_schema.py`
- Create: `tests/unit/test_h5_dependency_graph.py`
- Create: `tests/unit/test_h5_update_spec.py`
- Create: `docs/preregistrations/2026-07-21-h5-update-coherence.md`

**Interfaces:**

`vfe4/types/updates.py` owns every displayed snapshot/request/value record and update enum plus canonical recognition/model/request/live/reference/candidate/semantic-state encoders and `initial_live`. `vfe4/types/h5_schema.py` owns every Task 5 identifier/dependency/reconstruction literal tuple, sign, operation-count table/function, hash domain, factor-input/objective schema encoder, and resulting schema hash as plain immutable data, importing no update dataclass; lower-level types and higher-level objective/numerical modules therefore share one dependency-neutral source. Task 7 owns its provisional candidate-draft type, encoder, and distinct draft domain in `vfe4/inference/h5_updates.py`; those do not weaken or replace the Task 5 final-candidate constructor. `vfe4/objective/dependency_graph.py` owns `FactorDependencyGraph` plus dependency resolution. `vfe4/validation/h5_update_spec.py` owns the expected raw-digest literal, strict parser, reference builder, H1 reconstruction, and update-spec canonical encoder.

```python
class UpdateLabel(str, Enum):
    EXACT_COORDINATE = "exact_coordinate"
    VALID_MM = "valid_mm"
    GENERALIZED_EM = "generalized_em"
    NATURAL_GRADIENT_PROPOSAL = "natural_gradient_proposal"
    SGD_PROPOSAL = "sgd_proposal"
    ADAM_PROPOSAL = "adam_proposal"
    TRUNCATED_ITERATION = "truncated_iteration"

class H5UpdateRule(str, Enum):
    EXACT_Z0 = "exact_z0"
    EXACT_SOURCE_ROW_A2 = "exact_source_row_a2"
    EXACT_STATE_TRANSITION_2_M = "exact_state_transition_2_m"
    GENERALIZED_EM_EMISSION_1 = "generalized_em_emission_1"
    NATURAL_GRADIENT_Z1 = "natural_gradient_z1"

@dataclass(frozen=True)
class FrozenTensorValue:
    dtype: Literal["float64"]
    shape: tuple[int, ...]
    values: tuple[float, ...]

@dataclass(frozen=True)
class FrozenByteState:
    schema_version: str
    payload: bytes
    state_sha256: str = field(init=False)

@dataclass(frozen=True)
class GaussianRecognitionCoordinate:
    coordinate_id: str
    mean: FrozenTensorValue       # scalar shape ()
    variance: FrozenTensorValue   # scalar shape (), strictly positive

@dataclass(frozen=True)
class CategoricalRecognitionCoordinate:
    coordinate_id: str
    support: tuple[int, ...]
    conditioned_on: tuple[tuple[str, int], ...]
    probabilities: FrozenTensorValue

@dataclass(frozen=True)
class RecognitionSnapshot:
    schema_version: Literal["h5-recognition-snapshot-v1"]
    gaussians: tuple[GaussianRecognitionCoordinate, ...]
    categoricals: tuple[CategoricalRecognitionCoordinate, ...]
    state_sha256: str = field(init=False)

@dataclass(frozen=True)
class ModelParameterBlock:
    block_id: str
    values: tuple[tuple[str, FrozenTensorValue], ...]

@dataclass(frozen=True)
class FactorReconstructionRecord:
    factor_id: str
    bindings: tuple[str, ...]

@dataclass(frozen=True)
class SharedParameterGroup:
    group_id: Literal["shared_decoder_transition"]
    source: Literal["theta[shared_decoder_transition].s"]
    consumers: tuple[str, ...]

@dataclass(frozen=True)
class H5ModelSnapshot:
    schema_version: Literal["h5-model-snapshot-v1"]
    parameter_blocks: tuple[ModelParameterBlock, ...]
    reconstruction_records: tuple[FactorReconstructionRecord, ...]
    shared_groups: tuple[SharedParameterGroup, ...]
    objective_schema_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

@dataclass(frozen=True)
class UpdateSpecification:
    raw_bytes: bytes = field(repr=False)
    fixture_id: Literal["h5-conditional-update-v1"]
    fixture_schema_version: Literal[1]
    recognition_family: Literal["continuous_mean_field_conditional_categorical"]
    h1_fixture_id: Literal["h1-v1"]
    h1_fixture_sha256: str
    factor_input_schema_version: Literal["h5-factor-input-v1"]
    factor_input_schema_sha256: str = field(init=False)
    factor_universe: tuple[str, ...]
    recognition_coordinate_universe: tuple[str, ...]
    model_block_universe: tuple[str, ...]
    quadrature_orders: tuple[Literal[21], Literal[17]]
    reconstruction_records: tuple[FactorReconstructionRecord, ...]
    shared_groups: tuple[SharedParameterGroup, ...]
    initial_recognition: RecognitionSnapshot
    initial_model: H5ModelSnapshot
    canonical_bytes: bytes = field(init=False)
    canonical_sha256: str = field(init=False)
    raw_sha256: str = field(init=False)

    def as_h1_recognition_record(self) -> H1RecognitionFactorRecord: ...

@dataclass(frozen=True)
class H5ReferenceState:
    schema_version: Literal["h5-reference-state-v1"]
    raw_h1_fixture_bytes: bytes
    raw_update_spec_bytes: bytes
    h1_fixture_sha256: str = field(init=False)
    update_spec_raw_sha256: str = field(init=False)
    objective_schema_sha256: str = field(init=False)
    factor_input_schema_sha256: str = field(init=False)
    specification: UpdateSpecification
    initial_recognition: RecognitionSnapshot
    initial_model: H5ModelSnapshot
    initial_optimizer_state: FrozenByteState
    initial_rng_state: FrozenByteState
    reference_sha256: str = field(init=False)

@dataclass(frozen=True)
class H5LiveState:
    schema_version: Literal["h5-live-state-v1"]
    recognition: RecognitionSnapshot
    model: H5ModelSnapshot
    optimizer_state: FrozenByteState
    rng_state: FrozenByteState
    state_sha256: str = field(init=False)

@dataclass(frozen=True)
class UpdateRequest:
    schema_version: Literal["h5-update-request-v1"]
    request_id: str
    rule: H5UpdateRule
    requested_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    damping_schedule: tuple[float, ...]
    request_sha256: str = field(init=False)

@dataclass(frozen=True)
class H5CandidateSnapshot:
    schema_version: Literal["h5-candidate-v1"]
    rule: H5UpdateRule
    request_sha256: str
    producer_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    damping: float
    numerical_diagnostics: tuple[tuple[str, float], ...]
    recognition: RecognitionSnapshot
    model: H5ModelSnapshot
    candidate_sha256: str = field(init=False)
```

All snapshot/specification constructors require the displayed schema literals, exact universe and field order, exact block names/shapes, finite binary64 values, positive variances/`R`, normalized positive-supported categorical rows, exact reconstruction/shared-group equality, and recomputed hashes. `H5LiveState` and `H5CandidateSnapshot` defensively rebuild their nested immutable values; supplying an existing mutable tensor, mapping, hash, or alias cannot bypass validation. `initial_live(reference)` uses exactly the reference's four initial state records.

`UpdateRequest` validates the exact rule contract:

```python
H5_RULE_CONTRACTS = {
    H5UpdateRule.EXACT_Z0:
        (UpdateLabel.EXACT_COORDINATE, ("q[z0]",), (), (1.0,)),
    H5UpdateRule.EXACT_SOURCE_ROW_A2:
        (UpdateLabel.EXACT_COORDINATE, ("q[source_row_a2]",), (), (1.0,)),
    H5UpdateRule.EXACT_STATE_TRANSITION_2_M:
        (UpdateLabel.EXACT_COORDINATE, (), ("theta[state_transition_2]",), (1.0,)),
    H5UpdateRule.GENERALIZED_EM_EMISSION_1:
        (UpdateLabel.GENERALIZED_EM, (), ("theta[emission_1]",),
         (1.0, .5, .25, .125, .0625, .03125, .015625, .0078125,
          .00390625, .001953125, .0009765625)),
    H5UpdateRule.NATURAL_GRADIENT_Z1:
        (UpdateLabel.NATURAL_GRADIENT_PROPOSAL, ("q[z1]",), (), (64.0,)),
}
```

Production `H5CandidateSnapshot` construction rejects any mismatch between rule, requested label, active blocks, and schedule. `request_sha256` is recomputed from every displayed request field under the update-request domain. A final candidate copies the request's rule/hash/active blocks, records the one selected damping, and recomputes its own hash over those provenance fields, its diagnostics, and both snapshots; a final candidate can therefore neither migrate between requests nor conceal its line-search step. Only the exact M candidate carries `(("G_condition_number", kappa_2(G)),)`; all other final candidates require an empty diagnostics tuple. Task 7's immutable prevalidation draft records test/gate-only fault input before this constructor, so the mislabel control can prove the attempted provenance and observe typed rejection without ever constructing an invalid `H5CandidateSnapshot`. `valid_mm` never enters this mapping.

`FactorDependencyGraph` has exact tuple fields `factor_universe`, `recognition_coordinate_universe`, `model_block_universe`, `variable_dependencies`, and `parameter_dependencies`. Construction requires every coordinate/block exactly once, exact factor ordering, no unknown factor, and exact equality with the closed graph above. `expected_affected_factors` returns the universe-ordered union.

The fixture has exact top-level fields in this order before canonical sorted-key encoding:

```json
{
  "fixture_id": "h5-conditional-update-v1",
  "fixture_schema_version": 1,
  "recognition_family": "continuous_mean_field_conditional_categorical",
  "h1_fixture_id": "h1-v1",
  "h1_fixture_sha256": "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b",
  "factor_input_schema_version": "h5-factor-input-v1",
  "factor_universe": ["initial_joint","model_source[1]","model_transition[1]","state_source[1]","state_transition[1]","emission[1]","model_source[2]","model_transition[2]","state_source[2]","state_transition[2]","emission[2]","recognition_entropy"],
  "recognition_coordinate_universe": ["q[z0]","q[m0]","q[z1]","q[m1]","q[z2]","q[m2]","q[model_source_b1]","q[state_source_a1_b0]","q[model_source_b2]","q[source_row_a2]","q[state_source_a2_b1]"],
  "model_block_universe": ["theta[state_transition_2]","theta[emission_1]","theta[shared_decoder_transition]"],
  "quadrature_orders": [21,17],
  "continuous_recognition": [
    ["q[z0]",-0.10,0.65],["q[m0]",0.25,0.78],
    ["q[z1]",0.05,0.96],["q[m1]",0.175,1.21],
    ["q[z2]",-0.04,0.90],["q[m2]",0.14,1.40]
  ],
  "categorical_recognition": [
    ["q[model_source_b1]",[0],[],[1.0]],
    ["q[state_source_a1_b0]",[0],[["b1",0]],[1.0]],
    ["q[model_source_b2]",[0,1],[],[0.4,0.6]],
    ["q[source_row_a2]",[0,1],[["b2",0]],[0.75,0.25]],
    ["q[state_source_a2_b1]",[0,1],[["b2",1]],[0.2,0.8]]
  ],
  "model_parameter_blocks": [
    ["theta[state_transition_2]",[["alpha_0",0.8],["alpha_1",0.64],["B_base",-0.35],["c",0.08],["R",0.48]]],
    ["theta[emission_1]",[["w_z",[0.2,-0.4,0.1]],["w_m",[0.3,0.2,-0.5]],["bias",[0.05,-0.1,0.15]]]],
    ["theta[shared_decoder_transition]",[["s",0.0]]]
  ],
  "factor_reconstruction": [
    ["initial_joint",["h1.initial_joint","q[z0]","q[m0]"]],
    ["model_source[1]",["h1.model_source_priors[1]","q[model_source_b1]"]],
    ["model_transition[1]",["h1.model_transition[1]","q[m0]","q[m1]","q[model_source_b1]"]],
    ["state_source[1]",["h1.state_source_priors[1]","q[model_source_b1]","q[state_source_a1_b0]"]],
    ["state_transition[1]",["h1.state_transition[1]","q[z0]","q[z1]","q[m1]","q[model_source_b1]","q[state_source_a1_b0]"]],
    ["emission[1]",["theta[emission_1]","theta[shared_decoder_transition]","q[z1]","q[m1]","h1.observation_label[t=1]"]],
    ["model_source[2]",["h1.model_source_priors[2]","q[model_source_b2]"]],
    ["model_transition[2]",["h1.model_transition[2]","q[m0]","q[m1]","q[m2]","q[model_source_b2]"]],
    ["state_source[2]",["h1.state_source_priors[2]","q[model_source_b2]","q[source_row_a2]","q[state_source_a2_b1]"]],
    ["state_transition[2]",["theta[state_transition_2]","theta[shared_decoder_transition]","q[z0]","q[z1]","q[z2]","q[m2]","q[model_source_b2]","q[source_row_a2]","q[state_source_a2_b1]"]],
    ["emission[2]",["h1.emission[2]","theta[shared_decoder_transition]","q[z2]","q[m2]","h1.observation_label[t=2]"]],
    ["recognition_entropy",["recognition_snapshot"]]
  ],
  "shared_parameter_groups": [
    ["shared_decoder_transition","theta[shared_decoder_transition].s",["state_transition[2].B:add","emission[1].w_z[0]:add","emission[2].w_z[0]:add"]]
  ],
  "source_row_a2": {"coordinate_id":"q[source_row_a2]","time":2,"condition":["b2",0],"support":[0,1],"initial_probabilities":[0.75,0.25]}
}
```

- [ ] **Step 1: Write the preregistration with every Global Constraint, identifier tuple, dependency row, hash domain, fixture value, rule contract, numerical formula, positive case, control, status predicate, and nonclaim in this proposal.**

- [ ] **Step 2: Write strict failing type/hash/parser/schema/graph tests.** Include exact field-order assertions; source-independent reconstruction; exact rule mapping; immutable byte/tensor ownership; signed-zero-distinguishing snapshot/reference/semantic-state hashes; exact `initial_live` four-hash reconstruction; duplicate/extra/missing/alias rejection; raw hash before decode; wrong H1 hash; unnormalized/zero-supported rows; unequal slot offsets/variances; exact schema-hash cores and operation counts; wrong shared consumers; every missing/extra graph edge; singleton update rejection; and proof that the five-member `H5UpdateRule` universe has no `valid_mm` producer. Task 9 configuration tests, not attempt/gate tests, own rejection of a requested `valid_mm` label without a proof artifact.

```python
def test_conditional_family_reconstructs_source_independent_h1_record():
    state = build_h5_reference_state(H1_BYTES, H5_BYTES)
    equivalent = state.specification.as_h1_recognition_record()
    assert equivalent.initial_joint.covariance[0, 1].item() == 0.0
    for record in equivalent.model_kernels:
        assert torch.equal(record.slopes, torch.zeros_like(record.slopes))
        assert len(set(zip(record.offsets.tolist(), record.variances.tolist(), strict=True))) == 1
    for record in equivalent.state_kernels:
        assert torch.equal(record.z_slopes, torch.zeros_like(record.z_slopes))
        assert torch.equal(record.m_slopes, torch.zeros_like(record.m_slopes))
        assert len(set(zip(record.offsets.tolist(), record.variances.tolist(), strict=True))) == 1
    assert tuple(equivalent.state_source_probabilities_given_model_source[1][0].tolist()) == (0.75, 0.25)
```

- [ ] **Step 3: Run Task 5 RED.**

```powershell
python -m pytest tests/unit/test_h5_update_types.py tests/unit/test_h5_objective_schema.py tests/unit/test_h5_dependency_graph.py tests/unit/test_h5_update_spec.py -q
```

Expected: collection fails because the H5 types, graph, and parser do not exist.

- [ ] **Step 4: Author the exact JSON fixture above as UTF-8 without BOM, LF line endings, and one final LF; add the exact `.gitattributes` nonconversion rule before staging, then measure its raw SHA-256.**

```powershell
Get-FileHash -Algorithm SHA256 vfe4\validation\fixtures\h5_conditional_update_v1.json
```

Copy the command's complete lowercase 64-hex digest verbatim into `EXPECTED_H5_UPDATE_SPEC_RAW_SHA256` and the parser test. Do not truncate it, derive it at import time, or place a provisional digest in committed code.

- [ ] **Step 5: Implement the records, canonical encoders, raw-byte parser, reference-state builder, exact H1 reconstruction, and closed dependency graph.** Constructors defensively copy bytes and tuple data; `FrozenTensorValue.from_tensor` performs `detach().to(device="cpu",dtype=torch.float64).contiguous().clone()` before row-major extraction.

```python
def build_h5_reference_state(
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5ReferenceState: ...

def parse_h5_update_spec_bytes(data: bytes) -> UpdateSpecification: ...

def canonical_h5_recognition_snapshot_bytes(snapshot: RecognitionSnapshot) -> bytes: ...
def canonical_h5_model_snapshot_bytes(snapshot: H5ModelSnapshot) -> bytes: ...
def canonical_h5_live_state_bytes(state: H5LiveState) -> bytes: ...
def canonical_h5_reference_state_bytes(state: H5ReferenceState) -> bytes: ...
def canonical_h5_semantic_state_bytes(recognition: RecognitionSnapshot, model: H5ModelSnapshot) -> bytes: ...
def initial_live(reference: H5ReferenceState) -> H5LiveState: ...

def build_h5_reference_dependency_graph(specification: UpdateSpecification) -> FactorDependencyGraph: ...
def expected_affected_factors(
    graph: FactorDependencyGraph,
    *,
    variables: tuple[str, ...],
    parameters: tuple[str, ...],
) -> tuple[str, ...]: ...
```

- [ ] **Step 6: Run Task 5 GREEN.**

```powershell
python -m pytest tests/unit/test_h5_update_types.py tests/unit/test_h5_objective_schema.py tests/unit/test_h5_dependency_graph.py tests/unit/test_h5_update_spec.py -q
```

Expected: all Task 5 tests pass.

- [ ] **Step 7: Serialize `vfe4/types/__init__.py` export edits after the H4 Task 1 export commit is present, then commit Task 5.**

```powershell
git add .gitattributes vfe4/types/updates.py vfe4/types/h5_schema.py vfe4/types/__init__.py vfe4/objective/dependency_graph.py vfe4/objective/__init__.py vfe4/validation/fixtures/h5_conditional_update_v1.json vfe4/validation/h5_update_spec.py tests/unit/test_h5_update_types.py tests/unit/test_h5_objective_schema.py tests/unit/test_h5_dependency_graph.py tests/unit/test_h5_update_spec.py docs/preregistrations/2026-07-21-h5-update-coherence.md
git commit -m "feat: freeze closed H5 update protocol"
```

---

### Task 6: Implement the Complete H5 Objective, Ordered Factor Trace, Cache Proof, and Operand-Shaped Budget

**Files:**

- Create: `vfe4/objective/h5_complete.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `vfe4/numerics/h5_budget.py`
- Modify: `vfe4/numerics/__init__.py`
- Create: `tests/unit/test_h5_complete_objective.py`
- Create: `tests/unit/test_h5_budget.py`

**Interfaces:**

`vfe4/objective/h5_complete.py` owns factor input/evaluation/cache records, `CompleteElboEvaluation`, the evaluator protocol, and complete-objective implementation. `vfe4/numerics/h5_budget.py` owns the three allowance records, `H5BudgetConfig`, and all allowance functions while importing the already frozen constants/counts from Task 5's `h5_schema`; production inference imports this one-way numerical module, never `verification`.

```python
class CacheDisposition(str, Enum):
    REEVALUATED = "reevaluated"
    REUSED = "reused"

@dataclass(frozen=True)
class FactorInputHashRecord:
    factor_id: str
    input_schema_version: Literal["h5-factor-input-v1"]
    input_schema_sha256: str = field(init=False)
    canonical_input_bytes: bytes
    input_sha256: str = field(init=False)

@dataclass(frozen=True)
class FactorEvaluationRecord:
    factor_id: str
    input_hash: FactorInputHashRecord
    frozen_complement_sha256: str
    value_order_21: float
    value_order_17: float
    absolute_summands_order_21: tuple[float, ...]
    absolute_summands_order_17: tuple[float, ...]
    condition_numbers_order_21: tuple[float, ...]
    condition_numbers_order_17: tuple[float, ...]
    operation_count_order_21: int
    operation_count_order_17: int
    cache_disposition: CacheDisposition

@dataclass(frozen=True)
class H5TermAllowance:
    term_id: str
    objective_sign: Literal[-1, 0, 1]
    value_order_21: float
    value_order_17: float
    signed_reported_value: float
    absolute_summands_order_21: tuple[float, ...]
    absolute_summands_order_17: tuple[float, ...]
    condition_numbers_order_21: tuple[float, ...]
    condition_numbers_order_17: tuple[float, ...]
    operation_count_order_21: int
    operation_count_order_17: int
    convergence_estimate: float
    rounding_order_21: float
    rounding_order_17: float
    comparison_rounding: float
    total: float

@dataclass(frozen=True)
class H5CompleteAllowance:
    term_allowances: tuple[H5TermAllowance, ...]
    reduction_rounding: float
    total: float
    stochastic_contribution: float  # constructor requires exactly 0.0

@dataclass(frozen=True)
class H5DeltaAllowance:
    before_total: float
    after_total: float
    subtraction_rounding: float
    stochastic_contribution: float  # constructor requires exactly 0.0
    epsilon_delta: float

@dataclass(frozen=True)
class H5BudgetConfig:
    quadrature_orders: tuple[Literal[21], Literal[17]]
    epsilon: float
    C: float
    signed_term_ids: tuple[str, ...]
    analytic_operation_counts: Mapping[str, int]
    analytic_factor_operation_counts: Mapping[str, int]

@dataclass(frozen=True, init=False)
class CompleteElboEvaluation:
    terms: ElboTerms
    factor_records: tuple[FactorEvaluationRecord, ...]
    term_allowances: tuple[H5TermAllowance, ...]
    diagnostic_allowances: tuple[H5TermAllowance, ...]
    complete_allowance: H5CompleteAllowance
    objective_schema_sha256: str = field(init=False)
    evaluated_state_sha256: str = field(init=False)
    frozen_complement_sha256: str

    @classmethod
    def build(cls, *, state: H5LiveState | H5CandidateSnapshot, terms: ElboTerms, factor_records: tuple[FactorEvaluationRecord, ...], term_allowances: tuple[H5TermAllowance, ...], diagnostic_allowances: tuple[H5TermAllowance, ...], complete_allowance: H5CompleteAllowance, frozen_complement_sha256: str) -> Self: ...

@dataclass(frozen=True)
class FactorCacheKey:
    factor_id: str
    input_hash: FactorInputHashRecord
    quadrature_orders: tuple[Literal[21], Literal[17]]
    frozen_complement_sha256: str

@dataclass(frozen=True)
class FactorCacheEntry:
    key: FactorCacheKey
    record: FactorEvaluationRecord

class CompleteElboEvaluator(Protocol):
    def evaluate(
        self,
        state: H5LiveState | H5CandidateSnapshot,
        *,
        frozen_complement_sha256: str,
        cache: Mapping[FactorCacheKey, FactorCacheEntry] | None = None,
    ) -> CompleteElboEvaluation: ...

def evaluate_h5_complete_elbo(
    reference: H5ReferenceState,
    state: H5LiveState | H5CandidateSnapshot,
    *,
    frozen_complement_sha256: str,
    cache: Mapping[FactorCacheKey, FactorCacheEntry] | None = None,
) -> CompleteElboEvaluation: ...
```

`H5BudgetConfig.__post_init__` defensively copies both mappings into sorted `MappingProxyType` instances, requires exact equality with the displayed analytic-term and analytic-factor key sets/counts, rejects nonpositive counts, and requires the exact frozen constants `(21, 17)`, binary64 epsilon, and `C=4096.0`; callers cannot mutate the budget after construction. The two emission terms/factors are intentionally absent from those analytic maps and use `emission_operation_count(order)`.

`H5DeltaAllowance.__post_init__` requires finite nonnegative components, `stochastic_contribution == 0.0`, and exact recomputation of `epsilon_delta = before_total + after_total + subtraction_rounding`.

`factor_records` is exactly `H5_FACTOR_UNIVERSE` order. Its values are raw signed ELBO contributions: expected `log p` for the eleven generative factors and positive recognition entropy for `recognition_entropy`; `math.fsum(record.value_order_21)` equals `terms.complete_elbo` within its reduction allowance. `ElboTerms` is built separately from the same state and must yield the same scalar.

For signed `H5TermAllowance` records, `value_order_21` and `value_order_17` are the ordinary existing `ElboTerms` component values, `objective_sign` is the corresponding `H5_SIGNED_TERM_SIGNS[index]`, and `signed_reported_value = objective_sign * value_order_21`. The diagnostic `joint_recognition_entropy` record has `objective_sign=0` and `signed_reported_value=0.0`; it retains its full operand-shaped allowance but never enters complete reduction. Complete reduction requires exactly the signed records and uses their signed values once.

Each `canonical_input_bytes` value is the factor-input domain followed by canonical JSON with exactly the five ordered fields frozen in the factor-input schema: schema version, factor ID, reconstructed effective normalized factor, observation-or-null, and ordered recognition inputs. It contains the observation/support metadata, categorical weights, and moments needed by that factor expectation and excludes unrelated state and scalar outputs. `input_schema_sha256` is recomputed from the closed schema selected by `h5-factor-input-v1`; it is never trusted from a cache entry. The cache key includes that schema hash and both quadrature orders.

Cache resolution performs one exact full-key lookup. No entry, or only entries sharing a strict subset of key fields, is an ordinary miss and reevaluates both orders. An exact-key entry is reusable only if its embedded record recomputes the same factor ID, schema version/hash, input hash, complement hash, both order values/operands/counts are finite, and the stored record was originally `REEVALUATED`; reuse returns a validated copy marked `REUSED`. An exact-key entry whose embedded record disagrees with any key or payload invariant is `STALE_CACHE` and yields `FailedUpdateAttempt(DEPENDENCY_VALIDATION, STALE_CACHE)`; it is never silently treated as a miss.

`CompleteElboEvaluation.build` derives the evaluated-state and objective-schema hashes from its state/schema constants and is the only constructor. `FactorCacheKey` requires `factor_id == input_hash.factor_id`; its schema/input hashes are therefore derived through the nested immutable record rather than accepted independently.

For a full evaluation:

```python
available_factor_ids = ordered_union(reevaluated_factor_ids, reused_factor_ids)
missing_factor_ids = ordered_difference(H5_FACTOR_UNIVERSE, available_factor_ids)
extra_factor_ids = ordered_difference(available_factor_ids, H5_FACTOR_UNIVERSE)
```

An expected affected factor must be reevaluated and may never be reused. Unaffected factors may be reused only on an exact key match.

For universe-ordered before/after records, affectedness and the separate value diagnostic are exactly:

```python
observed_affected_factor_ids = tuple(
    factor_id for factor_id in H5_FACTOR_UNIVERSE
    if before_by_id[factor_id].input_hash.input_sha256
       != after_by_id[factor_id].input_hash.input_sha256
)
value_changed_factor_ids = tuple(
    factor_id for factor_id in H5_FACTOR_UNIVERSE
    if (
        before_by_id[factor_id].value_order_21.hex(),
        before_by_id[factor_id].value_order_17.hex(),
    ) != (
        after_by_id[factor_id].value_order_21.hex(),
        after_by_id[factor_id].value_order_17.hex(),
    )
)
```

No scalar tolerance participates in either set. `expected_affected_factor_ids` is independently the universe-ordered dependency-graph union for the request and must equal `observed_affected_factor_ids`; value-change equality is never an acceptance substitute.

- [ ] **Step 1: Write failing complete-objective, factor-input, cache, and budget tests.** Compare the source-independent H5 state against an equivalent H1 structured record at both quadrature orders; check all raw factor values, all signed/diagnostic/derived term sets, one complete scalar, exact input-hash changes for every dependency row, unchanged-scalar/changed-input behavior, cache hit/miss/stale-key paths, and exact operation counts/formulas/boundaries.

```python
def test_h5_complete_objective_has_one_scalar_and_complete_factor_trace():
    reference = build_h5_reference_state(H1_BYTES, H5_BYTES)
    evaluation = evaluate_h5_complete_elbo(reference, initial_live(reference), frozen_complement_sha256=COMPLEMENT)
    assert tuple(record.factor_id for record in evaluation.factor_records) == H5_FACTOR_UNIVERSE
    assert math.fsum(record.value_order_21 for record in evaluation.factor_records) == pytest.approx(evaluation.terms.complete_elbo, abs=evaluation.complete_allowance.total)
    assert tuple(item.term_id for item in evaluation.term_allowances) == H5_SIGNED_TERM_IDS
    assert tuple(item.term_id for item in evaluation.diagnostic_allowances) == H5_DIAGNOSTIC_TERM_IDS
    assert evaluation.complete_allowance.stochastic_contribution == 0.0
```

- [ ] **Step 2: Run Task 6 RED.**

```powershell
python -m pytest tests/unit/test_h5_complete_objective.py tests/unit/test_h5_budget.py -q
```

Expected: collection fails because the complete evaluator and H5 budget do not exist.

- [ ] **Step 3: Implement source-independent H1 reconstruction, factor-specific moment extraction, both-order evaluation, raw trace reconstruction, and the one `ElboTerms` construction.** Emission quadrature is the only order-dependent factor in H5 v1; analytic values are independently returned for both orders with zero convergence.

- [ ] **Step 4: Implement exact cache validation and operand-local allowances.**

```python
def term_allowance(
    term_id: str,
    *,
    objective_sign: Literal[-1, 0, 1],
    value_order_21: float,
    value_order_17: float,
    absolute_summands_order_21: tuple[float, ...],
    absolute_summands_order_17: tuple[float, ...],
    condition_numbers_order_21: tuple[float, ...],
    condition_numbers_order_17: tuple[float, ...],
    operation_count_order_21: int,
    operation_count_order_17: int,
) -> H5TermAllowance: ...

def complete_elbo_allowance(
    term_allowances: tuple[H5TermAllowance, ...],
    signed_terms: tuple[float, ...],
) -> H5CompleteAllowance: ...

def subtraction_rounding_allowance(before_elbo: float, after_elbo: float) -> float: ...

def epsilon_delta(
    before: H5CompleteAllowance,
    after: H5CompleteAllowance,
    *,
    before_elbo: float,
    after_elbo: float,
) -> H5DeltaAllowance: ...
```

- [ ] **Step 5: Run Task 6 GREEN.**

```powershell
python -m pytest tests/unit/test_h5_complete_objective.py tests/unit/test_h5_budget.py -q
```

Expected: all Task 6 tests pass.

- [ ] **Step 6: Commit Task 6.**

```powershell
git add vfe4/objective/h5_complete.py vfe4/objective/__init__.py vfe4/numerics/h5_budget.py vfe4/numerics/__init__.py tests/unit/test_h5_complete_objective.py tests/unit/test_h5_budget.py
git commit -m "feat: add complete H5 objective and budget"
```

---

### Task 7: Implement Exact Coordinates, Proposal Provenance, Freeze-Before-Evaluate Transactions, Rollback, and Independent Oracle

**Files:**

- Create: `vfe4/inference/h5_updates.py`
- Modify after H4 Task 2 export serialization: `vfe4/inference/__init__.py`
- Create: `verification/numpy_oracles/h5_updates.py`
- Modify after H4 Task 3 export serialization: `verification/numpy_oracles/__init__.py`
- Create: `tests/unit/test_h5_updates.py`
- Create: `tests/oracle/test_h5_update_oracle.py`

**Interfaces:**

`vfe4/inference/h5_updates.py` owns the attempt, fault, provenance, working-state, transaction records and all production update/controller functions below. `verification/numpy_oracles/h5_updates.py` owns only `H5OracleUpdate` and the four byte-only independent oracle functions.

```python
class AttemptPhase(str, Enum):
    REQUEST = "request"
    BEFORE_EVALUATION = "before_evaluation"
    PROPOSAL = "proposal"
    FREEZE = "freeze"
    AFTER_EVALUATION = "after_evaluation"
    DEPENDENCY_VALIDATION = "dependency_validation"
    DECISION = "decision"
    COMMIT_OR_ROLLBACK = "commit_or_rollback"

class AttemptFailureReason(str, Enum):
    LABEL_PROVENANCE_MISMATCH = "label_provenance_mismatch"
    FACTOR_COVERAGE_MISMATCH = "factor_coverage_mismatch"
    AFFECTED_FACTOR_MISMATCH = "affected_factor_mismatch"
    STALE_CACHE = "stale_cache"
    NONFINITE_OR_INVALID_CANDIDATE = "nonfinite_or_invalid_candidate"
    DECISION_POLICY_VIOLATION = "decision_policy_violation"
    ROLLBACK_HASH_MISMATCH = "rollback_hash_mismatch"
    DETERMINISTIC_REEVALUATION_MISMATCH = "deterministic_reevaluation_mismatch"

class DecisionReason(str, Enum):
    EXACT_WITHIN_ALLOWANCE = "exact_within_allowance"
    RESOLVED_POSITIVE = "resolved_positive"
    RESOLVED_DECREASE_REJECTED = "resolved_decrease_rejected"
    UNRESOLVED_DELTA_REJECTED = "unresolved_delta_rejected"

class H5FaultKind(str, Enum):
    OMIT_CHILD = "omit_child"
    OMIT_EMISSION = "omit_emission"
    FORCE_UNRESOLVED_GEM_ACCEPT = "force_unresolved_gem_accept"
    MISLABEL_NATURAL_AS_EXACT = "mislabel_natural_as_exact"
    MUTATE_REJECTED_LIVE_AND_RNG = "mutate_rejected_live_and_rng"
    CHANGE_INPUT_KEEP_VALUE = "change_input_keep_value"
    CHANGE_VALUE_KEEP_INPUT = "change_value_keep_input"

@dataclass(frozen=True)
class H5FaultInjection:
    kind: H5FaultKind
    target_factor_id: str | None
    scalar_delta: float | None

H5_CANDIDATE_DRAFT_DOMAIN: Final[bytes] = b"vfe4.h5.candidate-draft.v1\x00"

@dataclass(frozen=True)
class H5CandidateDraft:
    schema_version: Literal["h5-candidate-draft-v1"]
    rule: H5UpdateRule
    request_sha256: str
    producer_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    damping: float
    numerical_diagnostics: tuple[tuple[str, float], ...]
    recognition: RecognitionSnapshot
    model: H5ModelSnapshot
    candidate_draft_sha256: str = field(init=False)

@dataclass(frozen=True)
class UpdateHashRecord:
    schema_version: Literal["h5-update-hash-record-v1"]
    request_sha256: str
    before_live_sha256: str
    before_recognition_sha256: str
    before_model_sha256: str
    before_optimizer_sha256: str
    before_rng_sha256: str
    predecision_live_sha256: str | None
    predecision_optimizer_sha256: str | None
    predecision_rng_sha256: str | None
    candidate_draft_sha256: str | None
    candidate_sha256: str | None
    candidate_recognition_sha256: str | None
    candidate_model_sha256: str | None
    frozen_complement_sha256: str
    final_live_sha256: str
    final_recognition_sha256: str
    final_model_sha256: str
    final_optimizer_sha256: str
    final_rng_sha256: str

@dataclass(frozen=True)
class PartialFactorEvaluation:
    observed_records: tuple[FactorEvaluationRecord, ...]
    expected_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]

@dataclass(frozen=True)
class DeterministicReevaluationRecord:
    factor_id: str
    input_sha256: str
    reported_value_order_21: float
    reported_value_order_17: float
    recomputed_value_order_21: float
    recomputed_value_order_17: float
    matched: bool

@dataclass(frozen=True)
class CompletedUpdateAttempt:
    schema_version: Literal["h5-completed-attempt-v1"]
    request: UpdateRequest
    producer_label: UpdateLabel
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    expected_factor_ids: tuple[str, ...]
    expected_affected_factor_ids: tuple[str, ...]
    reevaluated_factor_ids: tuple[str, ...]
    reused_factor_ids: tuple[str, ...]
    observed_affected_factor_ids: tuple[str, ...]
    value_changed_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]
    before: CompleteElboEvaluation
    after: CompleteElboEvaluation
    delta_elbo: float
    allowance: H5DeltaAllowance
    accepted: bool
    decision_reason: DecisionReason
    line_search_step: int | None
    damping: float
    autograd_scope: tuple[str, ...]
    hashes: UpdateHashRecord

@dataclass(frozen=True)
class FailedUpdateAttempt:
    schema_version: Literal["h5-failed-attempt-v1"]
    request: UpdateRequest
    producer_label: UpdateLabel | None
    phase: AttemptPhase
    reason: AttemptFailureReason
    before: CompleteElboEvaluation | None
    partial_after: PartialFactorEvaluation | None
    expected_factor_ids: tuple[str, ...]
    expected_affected_factor_ids: tuple[str, ...]
    observed_factor_ids: tuple[str, ...]
    observed_affected_factor_ids: tuple[str, ...]
    value_changed_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    extra_factor_ids: tuple[str, ...]
    decision_delta_elbo: float | None
    decision_epsilon_delta: float | None
    attempted_accept: bool | None
    deterministic_reevaluation: DeterministicReevaluationRecord | None
    hashes: UpdateHashRecord
    obligations: tuple[str, ...]

H5AttemptOutcome: TypeAlias = CompletedUpdateAttempt | FailedUpdateAttempt

@dataclass(frozen=True)
class H5TransactionResult:
    schema_version: Literal["h5-transaction-result-v1"]
    live: H5LiveState
    outcome: H5AttemptOutcome

@dataclass
class DifferentiableRecognitionState:
    active_coordinate_ids: tuple[str, ...]
    mean_leaves: Mapping[str, torch.Tensor]
    log_variance_leaves: Mapping[str, torch.Tensor]
    categorical_logit_leaves: Mapping[str, torch.Tensor]

@dataclass
class DifferentiableModelState:
    active_block_ids: tuple[str, ...]
    unconstrained_leaves: Mapping[str, torch.Tensor]
```

Working-state constructors require exact active keys, CPU float64 scalar/vector leaves, `requires_grad=True` only for declared active leaves, and no storage alias with either live snapshot. Variance leaves use `exp(log_variance)` at freeze; categorical leaves use masked `log_softmax`/`softmax` on the fixed support. `freeze_candidate` first recomputes `SHA256(H5_FROZEN_COMPLEMENT_DOMAIN || canonical_frozen_complement_bytes(reference, live, request))` and requires exact equality with `expected_frozen_complement_sha256`; this check occurs before any draft or final candidate is constructed. It then takes the unchanged complete `live` base, replaces exactly the active request blocks from detached working leaves, and defensively copies every inactive recognition/model block from `live` into an `H5CandidateDraft`.

`H5CandidateDraft` has exactly the displayed field order, owns defensive immutable copies of both snapshots, requires the displayed schema literal, valid enum values, a 64-lowercase-hex request hash, exact tuples, finite binary64 damping/diagnostics, and valid finite recognition/model structures, but deliberately does not enforce the rule/producer/active-block/schedule relation. `freeze_candidate` still requires `draft.rule`, `draft.request_sha256`, `draft.variables`, and `draft.parameters` to be exact copies of the constructor-recomputed request before final construction; only the injected producer-label mismatch remains representable. Its canonical core is the exact-schema sorted-key compact JSON projection of its ten nonderived fields, using the common H5 recursive binary64/tuple/enum encoder and the full nested recognition/model semantic fields, never their caller-supplied or derived hashes. `candidate_draft_sha256` is recomputed as `SHA256(H5_CANDIDATE_DRAFT_DOMAIN || canonical_h5_candidate_draft_bytes(draft))`. The draft is never a live state, evaluator input, accepted candidate, or artifact-level substitute for a final candidate.

After recording the draft hash, `freeze_candidate` constructs `H5CandidateSnapshot` with `schema_version="h5-candidate-v1"` and the draft's exact remaining nine provenance/diagnostic/snapshot inputs. The existing final constructor remains the sole rule/producer/active-block/damping/diagnostic validator and independently deep-copies and hashes the final snapshot. A private typed rejection carrier may retain the immutable draft only long enough for `execute_update` to create a typed `FREEZE` failure; it is not exported or serialized. Working states are never accepted, hashed as live state, or serialized into artifacts.

`UpdateHashRecord` has exactly the displayed field order and `request_sha256` must equal the constructor-recomputed request hash. Before/final live, recognition, model, optimizer, RNG, and frozen-complement hashes are mandatory for every transaction outcome, including early typed failure. Before a draft exists, `candidate_draft_sha256`, `candidate_sha256`, `candidate_recognition_sha256`, and `candidate_model_sha256` are all `None`. Once a structurally valid draft exists, its draft/recognition/model hashes are mandatory. `candidate_sha256` becomes mandatory only after successful `H5CandidateSnapshot` construction and is `None` for every final-constructor rejection, including the mislabel `FREEZE` failure. Predecision hashes become mandatory only after successful final freeze and before after-evaluation/decision; phase constructors require every unavailable later hash to be `None`. A completed outcome requires every hash field, including both distinct draft and final candidate hashes.

`CompletedUpdateAttempt` requires complete before/after universes, no missing/extra IDs, exact producer/request label agreement, exact expected/observed affected equality, and exact decision logic. `FailedUpdateAttempt` is the only representation for a missing factor, pre-evaluation label rejection, stale cache, invalid candidate, forced invalid decision, mutation, or deterministic scalar corruption; phase-specific constructors require producer/decision/affectedness/recheck fields exactly when that phase has observed them and require `None`/empty tuples before observation. A complement mismatch or structurally invalid working-state freeze has no draft or final candidate hash. The `MISLABEL_NATURAL_AS_EXACT` path has a recomputed draft/recognition/model hash, `candidate_sha256 is None`, all predecision hashes `None`, no after evaluation or acceptance decision, and final live/recognition/model/optimizer/RNG hashes exactly equal to the before hashes; `H5TransactionResult.live` is the original caller-owned live object. No failure fabricates an after-`ElboTerms` value.

The transition symbols below are not free: at `t=1`, `(alpha_10,B_1,c_1,R_1)=(1.25,0.45,-0.12,0.37)` from the frozen H1 factor; at `t=2`, `(alpha_20,alpha_21,B_2,c_2,R_2)=(alpha_0,alpha_1,B_base+s,c,R)` from the current live H5 model snapshot. Every formula uses those effective values and the displayed H5 recognition means/variances.

The exact coordinates are:

1. For `q[z0]`, let `(J0,h0)` be the information form of the H1 initial joint, and let
   \[
   w_{t0}=P_Q(a_t=0),\quad w_{10}=1,\quad
   w_{20}=\sum_b\gamma_2(b)\beta_2(0\mid b).
   \]
   Then
   \[
   J^\star=J_{0,zz}+\sum_{t=1}^{2}w_{t0}\alpha_{t0}^2/R_t,
   \]
   \[
   h^\star=h_{0,z}-J_{0,zm}\mu_{m0}+
   \sum_{t=1}^{2}w_{t0}\alpha_{t0}
   (\mu_{z_t}-B_t\mu_{m_t}-c_t)/R_t.
   \]
   Return variance `1/J*` and mean `h*/J*`. Recognition entropy supplies the coordinate normalizer; it is not added as an information factor.

2. For `q[source_row_a2]`, for `a in (0,1)` use `alpha_a=(alpha_0,alpha_1)[a]`, `B_2=B_base+s`, and the live `(c,R)`, and define
   \[
   \ell_a=-\frac12\left[\log(2\pi R_2)+
   \frac{V_{z2}+\alpha_a^2V_{za}+B_2^2V_{m2}+
   (\mu_{z2}-\alpha_a\mu_{za}-B_2\mu_{m2}-c_2)^2}{R_2}\right].
   \]
   Return `softmax(log(state_source_prior_2[a]) + ell_a)` in support order `(0,1)`. The positive fixed `gamma_2(0)` multiplier cancels from this row optimum.

3. For the full `theta[state_transition_2]` M block, set
   \[
   x_a=(1_{a=0}z_0,1_{a=1}z_1,m_2,1)^T,
   \qquad w_{ab}=\gamma_2(b)\beta_2(a\mid b).
   \]
   With the detached recognition snapshot fixed,
   \[
   G=\sum_{a,b}w_{ab}E[x_ax_a^T],\qquad
   g=\sum_{a,b}w_{ab}E[x_az_2],
   \]
   \[
   (\alpha_0^\star,\alpha_1^\star,B_{effective}^\star,c^\star)^T=G^{-1}g,
   \]
   \[
   R^\star=\sum_{a,b}w_{ab}E[(z_2-\theta^{\star T}x_a)^2].
   \]
   Every expectation uses the frozen mean-field moments `E[u^2]=V_u+mu_u^2` and `E[uv]=mu_u*mu_v` for distinct continuous coordinates. Accumulate in support order `b=(0,1)`, then `a=(0,1)`, set `G = 0.5*(G+G.T)`, and solve the displayed system by Cholesky without a pseudoinverse or ridge term. Require successful SPD factorization, finite results, and `R*>0`. Store `B_base*=B_effective*-s` with the shared scalar fixed.

`differentiable_h5_complete_elbo_order_21` reconstructs every inactive value from `live`, substitutes exactly the active leaves, and evaluates the identical Task 6 factor algebra at order 21; its detached scalar must equal the Task 6 order-21 complete scalar within that record's rounding allowance before any gradient is used.

For generalized EM, compute one Euclidean direction at the unchanged live state from the order-21 complete objective,
`g = grad_(w_z,w_m,bias) L_21(live)`, with recognition, the shared scalar, and every other model block detached. For each damping in the rule's exact schedule, form `(w_z,w_m,bias)_d = (w_z,w_m,bias)_live + d*g` from that same direction, freeze and evaluate it, and select only the first finite candidate with `delta_elbo > epsilon_delta`. No optimizer/RNG state changes, no direction recomputation between damping values, and no order-17-only acceptance are allowed.

The H5 v1 natural-gradient control differentiates the same order-21 complete objective and updates only the `q[z1]` mean with its variance frozen: `mu_new = mu_old + 64.0 * variance * dL_21/dmu`. The fixed multiplier `64.0` is the preregistered oversized proposal; it is not backtracked or relabeled. Its candidate must remain finite and is accepted or rejected only by the complete-delta policy; order 17 remains part of the complete allowance.

```python
def exact_conjugate_gaussian_e_update(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest) -> H5CandidateSnapshot: ...
def exact_source_row_update(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest) -> H5CandidateSnapshot: ...
def exact_gaussian_m_update(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest) -> H5CandidateSnapshot: ...
def differentiable_h5_complete_elbo_order_21(reference: H5ReferenceState, live: H5LiveState, recognition_working: DifferentiableRecognitionState, model_working: DifferentiableModelState) -> torch.Tensor: ...
def propose_generalized_em(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest, damping: float) -> H5CandidateSnapshot: ...
def propose_natural_gradient(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest, step_size: float) -> H5CandidateSnapshot: ...
def canonical_h5_candidate_draft_bytes(draft: H5CandidateDraft) -> bytes: ...
def freeze_candidate(reference: H5ReferenceState, live: H5LiveState, recognition_working: DifferentiableRecognitionState, model_working: DifferentiableModelState, *, request: UpdateRequest, producer_label: UpdateLabel, damping: float, expected_frozen_complement_sha256: str) -> H5CandidateSnapshot: ...
def canonical_frozen_complement_bytes(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest) -> bytes: ...
def execute_update(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest, evaluator: CompleteElboEvaluator, budget: H5BudgetConfig, *, fault_injection: H5FaultInjection | None = None) -> H5TransactionResult: ...
```

After the existing H4 names, `vfe4/inference/__init__.py` appends exactly `H5_CANDIDATE_DRAFT_DOMAIN`, `AttemptPhase`, `AttemptFailureReason`, `DecisionReason`, `H5FaultKind`, `H5FaultInjection`, `H5CandidateDraft`, `UpdateHashRecord`, `PartialFactorEvaluation`, `DeterministicReevaluationRecord`, `CompletedUpdateAttempt`, `FailedUpdateAttempt`, `H5AttemptOutcome`, `H5TransactionResult`, `DifferentiableRecognitionState`, `DifferentiableModelState`, `canonical_h5_candidate_draft_bytes`, `exact_conjugate_gaussian_e_update`, `exact_source_row_update`, `exact_gaussian_m_update`, `differentiable_h5_complete_elbo_order_21`, `propose_generalized_em`, `propose_natural_gradient`, `freeze_candidate`, `canonical_frozen_complement_bytes`, and `execute_update`, in that order. The private draft-rejection carrier is absent from both imports and `__all__`. After the existing H4 oracle names, `verification/numpy_oracles/__init__.py` appends exactly `H5OracleTermEvidence`, `H5OracleOperandEvidence`, `H5OracleUpdate`, `oracle_exact_e_block`, `oracle_exact_source_row`, `oracle_exact_m_block`, and `oracle_complete_delta`, in that order.

Decision rules are exact: an exact coordinate is eligible iff `delta_elbo >= -epsilon_delta`; generalized-EM and other proposal labels are eligible iff `delta_elbo > epsilon_delta`; `delta_elbo < -epsilon_delta` is a resolved rejection; the closed boundary is rejected as unresolved. Any required emission-touching positive case on that boundary makes the gate inconclusive, not failed. A rejected resolved-decrease natural-gradient positive case passes only with identical final live/recognition/model/optimizer/RNG hashes. `execute_update` returns a new immutable live state on acceptance and the original object on rejection or failure; no caller-visible in-place mutation occurs.

The independent oracle accepts bytes, never production types. It parses its two candidate JSON byte fields into the same closed semantic snapshot schemas and independently computes `semantic_state_sha256`; it neither constructs nor claims equality of the provenance-bearing production `candidate_sha256`. Only `oracle_exact_m_block` records `(("G_condition_number", kappa_2(G)),)`; every other oracle result requires an empty candidate-condition-number tuple. It retains an oracle-owned exact complete-term trace for both the before and after evaluations so Task 8 can reproduce every aggregate without borrowing a production allowance. `verification/numpy_oracles/h5_updates.py` owns module-local immutable copies of the exact signed-term IDs/signs, operation-count tables, `epsilon`, `C=4096.0`, and emission count formula; it imports neither PyTorch nor a production H5 module.

`H5OracleTermEvidence` has exactly the displayed field order. Its constructor recomputes and validates `signed_reported_value`, `convergence_estimate`, all three rounding fields, and `total` from that record's own both-order values, absolute summands, condition numbers, and operation counts; analytic records require identical both-order values and zero convergence. `H5OracleOperandEvidence` is `init=False`: callers can construct it only through `from_complete_terms` or `from_delta`. `from_complete_terms` requires `operand in {"before","after"}`, the exact 12 signed-term IDs/order/signs, and independently recomputes `value`, aggregate operation count, ordered condition/absolute-summand traces, convergence, rounding, and allowance. `from_delta` requires validated before/after oracle operands, stores an empty term trace, and independently derives the exact subtraction evidence and allowance. Any caller-supplied or post-copy aggregate perturbation is rejected. `H5OracleUpdate` additionally requires exact `before,after,delta` roles and recomputes the delta record from its two complete operands; it never accepts a production `CompleteElboEvaluation`, `H5CompleteAllowance`, or `H5DeltaAllowance`:

```python
@dataclass(frozen=True)
class H5OracleTermEvidence:
    schema_version: Literal["h5-oracle-term-evidence-v1"]
    term_id: str
    objective_sign: Literal[-1, 1]
    value_order_21: float
    value_order_17: float
    signed_reported_value: float
    absolute_summands_order_21: tuple[float, ...]
    absolute_summands_order_17: tuple[float, ...]
    condition_numbers_order_21: tuple[float, ...]
    condition_numbers_order_17: tuple[float, ...]
    operation_count_order_21: int
    operation_count_order_17: int
    convergence_estimate: float
    rounding_order_21: float
    rounding_order_17: float
    comparison_rounding: float
    total: float

@dataclass(frozen=True, init=False)
class H5OracleOperandEvidence:
    schema_version: Literal["h5-oracle-operand-evidence-v1"]
    operand: Literal["before", "after", "delta"]
    complete_term_trace: tuple[H5OracleTermEvidence, ...]
    value: float
    operation_count: int
    condition_numbers: tuple[float, ...]
    absolute_summands: tuple[float, ...]
    convergence: float
    rounding: float
    allowance: float

    @classmethod
    def from_complete_terms(cls, *, operand: Literal["before", "after"], complete_term_trace: tuple[H5OracleTermEvidence, ...]) -> Self: ...

    @classmethod
    def from_delta(cls, *, before: Self, after: Self) -> Self: ...

@dataclass(frozen=True)
class H5OracleUpdate:
    schema_version: Literal["h5-oracle-update-v1"]
    rule: str
    candidate_recognition_json: bytes
    candidate_model_json: bytes
    candidate_condition_numbers: tuple[tuple[str, float], ...]
    semantic_state_sha256: str = field(init=False)
    before: H5OracleOperandEvidence
    after: H5OracleOperandEvidence
    delta: H5OracleOperandEvidence

def oracle_exact_e_block(h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes) -> H5OracleUpdate: ...
def oracle_exact_source_row(h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes) -> H5OracleUpdate: ...
def oracle_exact_m_block(h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes) -> H5OracleUpdate: ...
def oracle_complete_delta(h1_fixture_bytes: bytes, update_spec_bytes: bytes, before_state_bytes: bytes, after_state_bytes: bytes, *, rule: str) -> H5OracleUpdate: ...
```

- [ ] **Step 1: Write failing tests for the five positives, every transaction phase, and independent oracle.** Assert exact z0 blanket contributions, the row formula, full five-parameter M solve, fixed detached recognition, fieldwise production/oracle exact-candidate agreement under the frozen operand-shaped allowances while retaining nonbinding semantic hashes, detached equality between differentiable and Task 6 order-21 complete objectives, first resolved GEM damping, oversized natural-gradient rejection, freeze-before-evaluate, no H2 mutation/autograd, input-hash affectedness, valid unaffected reuse, and byte-identical rollback. Assert the exact `H5CandidateDraft` and revised `UpdateHashRecord` field orders, draft domain/encoder/hash recomputation, defensive nonaliasing ownership, the corrected keyword-only `freeze_candidate` signature, and complement recomputation before draft construction. Prove that a valid freeze records distinct mandatory draft/final hashes, while the mislabel path records its recomputed draft/recognition/model hashes, keeps `candidate_sha256` and every predecision hash `None`, performs no evaluation, and returns the original live object with exact before/final rollback hashes. Assert the exact `H5OracleTermEvidence` and `H5OracleOperandEvidence` field/export orders, exact local 12-term ID/sign/count tables, exact `before,after,delta` roles, both-order values and all rounding inputs, independent aggregate recomputation, and `delta.allowance == before.allowance + after.allowance + delta.rounding`. Reconstruct every aggregate from the serialized oracle trace, reject each one-field term/aggregate perturbation, prove direct `H5OracleOperandEvidence(...)` construction is unavailable, and prove perturbing a production allowance object cannot change any oracle record. Assert the exact Task 7 production and oracle export orders and absence of the private rejection carrier.

- [ ] **Step 2: Run Task 7 RED.**

```powershell
python -m pytest tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py -q
```

Expected: collection fails because H5 updates and oracle do not exist.

- [ ] **Step 3: Implement the three exact coordinates without autograd or generic optimizers.** Use solve/Cholesky operations directly and freeze candidates immediately.

- [ ] **Step 4: Implement differentiable GEM/natural proposals with exact active-leaf scopes and the frozen damping order.** `torch.autograd.grad` may receive only declared active leaves. Candidate freeze must precede evaluator entry.

- [ ] **Step 5: Implement transactional `execute_update`.** Capture before hashes, evaluate before, derive expected dependencies, compute the expected frozen-complement hash, propose/freeze through the corrected reference-and-expected-hash signature, prove live state unchanged predecision, evaluate after, derive hash-affected IDs, validate complete/reused sets, compute the exact delta allowance, apply label policy, then atomically replace whole snapshots or retain the original live state. Convert every typed phase failure into `FailedUpdateAttempt`; retain a draft hash on final-constructor rejection without fabricating a final candidate hash.

- [ ] **Step 6: Implement the NumPy-only oracle with an independent strict parser and dense moment formulas.** It imports neither PyTorch nor `vfe4` production H5 modules and shares no production dependency graph.

- [ ] **Step 7: Run Task 7 GREEN.**

```powershell
python -m pytest tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py -q
```

Expected: all Task 7 tests pass.

- [ ] **Step 8: Serialize export edits after H4 Tasks 2 and 3 exports are merged, then commit Task 7.**

```powershell
git add vfe4/inference/h5_updates.py vfe4/inference/__init__.py verification/numpy_oracles/h5_updates.py verification/numpy_oracles/__init__.py tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py
git commit -m "feat: add transactional H5 updates"
```

---

### Task 8: Add the H5 Gate, Five Positive Results, Seven Independent Controls, Status Mapping, and Byte-Bearing Evaluation

**Files:**

- Create: `verification/h5_gate.py`
- Create: `tests/promotion/test_h5_gate.py`
- Modify: `docs/preregistrations/2026-07-21-h5-update-coherence.md`

**Interfaces:**

`verification/h5_gate.py` owns all positive/control/gate records and both gate functions below; no `vfe4` module imports it.

```python
class H5PositiveCaseId(str, Enum):
    EXACT_GAUSSIAN_E = "exact_gaussian_e_coordinate"
    EXACT_SOURCE_ROW = "exact_categorical_source_coordinate"
    EXACT_GAUSSIAN_M = "exact_gaussian_m_coordinate_fixed_recognition"
    ACCEPTED_GEM = "accepted_resolved_generalized_em"
    REJECTED_NATURAL = "rejected_proposal_rollback"

class H5ControlId(str, Enum):
    OMIT_CHILD = "child_factor_omission_detected"
    OMIT_EMISSION = "emission_factor_omission_detected"
    FORCE_UNRESOLVED_GEM = "unresolved_gem_acceptance_detected"
    MISLABEL_NATURAL = "natural_gradient_mislabel_detected"
    MUTATE_REJECTION = "rejection_mutation_detected"
    CHANGED_INPUT_EQUAL_VALUE = "changed_input_equal_value_detected"
    CHANGED_VALUE_SAME_INPUT = "changed_value_unchanged_input_not_affected"

class H5ControlDetection(str, Enum):
    CHILD_FACTOR_COVERAGE_FAILURE = "child_factor_coverage_failure"
    EMISSION_FACTOR_COVERAGE_FAILURE = "emission_factor_coverage_failure"
    UNRESOLVED_GEM_POLICY_FAILURE = "unresolved_gem_policy_failure"
    NATURAL_LABEL_PROVENANCE_FAILURE = "natural_label_provenance_failure"
    REJECTION_ROLLBACK_HASH_FAILURE = "rejection_rollback_hash_failure"
    INPUT_HASH_CHANGE_WITH_EQUAL_VALUE = "input_hash_change_with_equal_value"
    VALUE_CHANGE_WITH_SAME_INPUT = "value_change_with_same_input"

H5_CONTROL_DETECTION_BY_ID = MappingProxyType(dict(zip(H5ControlId, H5ControlDetection, strict=True)))

H5_POSITIVE_RULE_BY_ID = MappingProxyType({
    H5PositiveCaseId.EXACT_GAUSSIAN_E: H5UpdateRule.EXACT_Z0,
    H5PositiveCaseId.EXACT_SOURCE_ROW: H5UpdateRule.EXACT_SOURCE_ROW_A2,
    H5PositiveCaseId.EXACT_GAUSSIAN_M: H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
    H5PositiveCaseId.ACCEPTED_GEM: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
    H5PositiveCaseId.REJECTED_NATURAL: H5UpdateRule.NATURAL_GRADIENT_Z1,
})

H5_CONTROL_BASE_RULE_BY_ID = MappingProxyType({
    H5ControlId.OMIT_CHILD: H5UpdateRule.EXACT_Z0,
    H5ControlId.OMIT_EMISSION: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
    H5ControlId.FORCE_UNRESOLVED_GEM: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
    H5ControlId.MISLABEL_NATURAL: H5UpdateRule.NATURAL_GRADIENT_Z1,
    H5ControlId.MUTATE_REJECTION: H5UpdateRule.NATURAL_GRADIENT_Z1,
    H5ControlId.CHANGED_INPUT_EQUAL_VALUE: H5UpdateRule.EXACT_STATE_TRANSITION_2_M,
    H5ControlId.CHANGED_VALUE_SAME_INPUT: H5UpdateRule.GENERALIZED_EM_EMISSION_1,
})

class H5PreflightPhase(str, Enum):
    H1_FIXTURE_VALIDATION = "h1_fixture_validation"
    UPDATE_SPEC_VALIDATION = "update_spec_validation"
    REFERENCE_CONSTRUCTION = "reference_construction"
    READY = "ready"

class H5PreflightErrorKind(str, Enum):
    INVALID_H1_FIXTURE = "invalid_h1_fixture"
    UPDATE_SPEC_RAW_DIGEST_MISMATCH = "update_spec_raw_digest_mismatch"
    INVALID_UPDATE_SPEC_SCHEMA = "invalid_update_spec_schema"
    REFERENCE_CONSTRUCTION_FAILED = "reference_construction_failed"
    OBJECTIVE_SCHEMA_IDENTITY_FAILED = "objective_schema_identity_failed"
    FACTOR_INPUT_SCHEMA_IDENTITY_FAILED = "factor_input_schema_identity_failed"

H5_PREFLIGHT_ERROR_KINDS_BY_PHASE = MappingProxyType({
    H5PreflightPhase.H1_FIXTURE_VALIDATION: (
        H5PreflightErrorKind.INVALID_H1_FIXTURE,
    ),
    H5PreflightPhase.UPDATE_SPEC_VALIDATION: (
        H5PreflightErrorKind.UPDATE_SPEC_RAW_DIGEST_MISMATCH,
        H5PreflightErrorKind.INVALID_UPDATE_SPEC_SCHEMA,
    ),
    H5PreflightPhase.REFERENCE_CONSTRUCTION: (
        H5PreflightErrorKind.REFERENCE_CONSTRUCTION_FAILED,
        H5PreflightErrorKind.OBJECTIVE_SCHEMA_IDENTITY_FAILED,
        H5PreflightErrorKind.FACTOR_INPUT_SCHEMA_IDENTITY_FAILED,
    ),
    H5PreflightPhase.READY: (),
})

class H5UnavailableField(str, Enum):
    UPDATE_SPEC_CANONICAL_SHA256 = "update_spec_canonical_sha256"
    OBJECTIVE_SCHEMA_SHA256 = "objective_schema_sha256"
    FACTOR_INPUT_SCHEMA_VERSION = "factor_input_schema_version"
    FACTOR_INPUT_SCHEMA_SHA256 = "factor_input_schema_sha256"
    REFERENCE = "reference"
    REFERENCE_SHA256 = "reference_sha256"
    FACTOR_UNIVERSE = "factor_universe"
    RECOGNITION_COORDINATE_UNIVERSE = "recognition_coordinate_universe"
    MODEL_BLOCK_UNIVERSE = "model_block_universe"
    VARIABLE_DEPENDENCY_ROWS = "variable_dependency_rows"
    PARAMETER_DEPENDENCY_ROWS = "parameter_dependency_rows"
    POSITIVE_CASES = "positive_cases"
    POSITIVE_ATTEMPTS = "positive_attempts"
    CONTROLS = "controls"
    ORACLE_RESULTS = "oracle_results"

@dataclass(frozen=True)
class H5PreflightError:
    phase: H5PreflightPhase
    kind: H5PreflightErrorKind
    detail: str

@dataclass(frozen=True)
class H5PreflightRecord:
    schema_version: Literal["h5-preflight-record-v1"]
    phase: H5PreflightPhase
    h1_fixture_raw_sha256: str
    update_spec_raw_sha256: str
    errors: tuple[H5PreflightError, ...]
    unavailable_fields: tuple[H5UnavailableField, ...]
    obligation: str | None

@dataclass(frozen=True)
class H5DeltaOperandEvidence:
    schema_version: Literal["h5-delta-operand-evidence-v1"]
    operand: Literal["before", "after", "delta"]
    value: float
    operation_count: int
    condition_numbers: tuple[float, ...]
    absolute_summands: tuple[float, ...]
    rounding: float
    allowance: float

@dataclass(frozen=True)
class H5DeltaImplementationEvidence:
    schema_version: Literal["h5-delta-implementation-evidence-v1"]
    implementation: Literal["production", "oracle"]
    before: H5DeltaOperandEvidence
    after: H5DeltaOperandEvidence
    delta: H5DeltaOperandEvidence
    operand_shaped: bool

@dataclass(frozen=True)
class H5DeltaAgreement:
    schema_version: Literal["h5-delta-agreement-v1"]
    rule: H5UpdateRule
    production: H5DeltaImplementationEvidence
    oracle: H5DeltaImplementationEvidence
    comparison_rounding: float
    allowance: float
    absolute_error: float
    passed: bool

@dataclass(frozen=True)
class H5CandidateScalarComparison:
    field_id: str
    production_value: float
    oracle_value: float
    operation_count: int
    production_condition_number: float
    oracle_condition_number: float
    production_rounding: float
    oracle_rounding: float
    comparison_rounding: float
    allowance: float
    absolute_error: float
    passed: bool

@dataclass(frozen=True)
class H5CandidateComparison:
    rule: H5UpdateRule
    scalar_comparisons: tuple[H5CandidateScalarComparison, ...]
    max_absolute_error: float
    max_allowance: float
    passed: bool

@dataclass(frozen=True)
class H5PositiveCaseResult:
    schema_version: Literal["h5-positive-case-result-v1"]
    case_id: H5PositiveCaseId
    outcome: H5AttemptOutcome
    production_semantic_state_sha256: str
    oracle_semantic_state_sha256: str
    candidate_comparison: H5CandidateComparison | None
    delta_agreement: H5DeltaAgreement
    passed: bool
    detail: str

@dataclass(frozen=True)
class H5ControlResult:
    schema_version: Literal["h5-control-result-v1"]
    control_id: H5ControlId
    expected_detection: H5ControlDetection
    observed_detection: H5ControlDetection | None
    outcome: H5AttemptOutcome
    passed: bool
    detail: str

@dataclass(frozen=True)
class H5GateResult:
    schema_version: Literal["h5-gate-result-v1"]
    gate: Literal["H5"]
    status: GateStatus
    preflight: H5PreflightRecord
    h1_fixture_raw_sha256: str
    update_spec_raw_sha256: str
    update_spec_canonical_sha256: str | None
    objective_schema_sha256: str | None
    factor_input_schema_version: Literal["h5-factor-input-v1"] | None
    factor_input_schema_sha256: str | None
    reference_sha256: str | None
    positive_cases: tuple[H5PositiveCaseResult, ...] | None
    controls: tuple[H5ControlResult, ...] | None
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

@dataclass(frozen=True)
class H5ValidationPayloadRecord:
    schema_version: Literal[1]
    result: H5GateResult
    reference_sha256: str | None
    factor_universe: tuple[str, ...] | None
    recognition_coordinate_universe: tuple[str, ...] | None
    model_block_universe: tuple[str, ...] | None
    variable_dependency_rows: tuple[tuple[str, tuple[str, ...]], ...] | None
    parameter_dependency_rows: tuple[tuple[str, tuple[str, ...]], ...] | None
    positive_attempts: tuple[H5AttemptOutcome, ...] | None
    controls: tuple[H5ControlResult, ...] | None
    oracle_results: tuple[H5OracleUpdate, ...] | None
    nonclaims: tuple[str, ...]
    canonical_bytes: bytes = field(init=False, repr=False)
    payload_sha256: str = field(init=False)

@dataclass(frozen=True)
class H5GateEvaluation:
    schema_version: Literal["h5-gate-evaluation-v1"]
    result: H5GateResult
    reference: H5ReferenceState | None
    positive_attempts: tuple[H5AttemptOutcome, ...] | None
    controls: tuple[H5ControlResult, ...] | None
    oracle_results: tuple[H5OracleUpdate, ...] | None
    validation_payload: H5ValidationPayloadRecord

def compare_h5_exact_candidate(production: H5CandidateSnapshot, oracle: H5OracleUpdate) -> H5CandidateComparison: ...
def compare_h5_complete_delta(outcome: CompletedUpdateAttempt, oracle: H5OracleUpdate) -> H5DeltaAgreement: ...

def evaluate_h5(
    config: ResolvedConfig,
    *,
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5GateEvaluation: ...

def h5_validation_payload(evaluation: H5GateEvaluation) -> dict[str, object]: ...
```

The caller must pass the same captured immutable H5 byte object to production and oracle adapters. In Task 8, `evaluate_h5` reads only the already-existing resolved common CPU/float64/determinism fields; all H5 protocol identity comes from Task 5 constants and the captured bytes. It never rereads either fixture path. Task 9 later adds and validates the H5 config section without changing this byte-bearing signature.

Before parsing, `evaluate_h5` computes the two raw SHA-256 digests from the captured byte objects. `H5_PREFLIGHT_ERROR_KINDS_BY_PHASE` is the exact exhaustive phase/error validity map: every error kind appears once, an error is valid only in its mapped phase, and `READY` admits none. A successful preflight has `phase=READY`, no errors, no unavailable fields, and `obligation is None`; every optional result/evaluation/payload field is then non-`None`. Any H1 fixture validation, full update-spec digest/schema validation, objective/factor-schema identity, or reference-construction exception is caught only at this preflight boundary and returned as `GateStatus.INCONCLUSIVE`: `H5PreflightRecord` contains the exact attempted phase, exactly one mapped typed error, the two actually computed raw digests, the full `H5UnavailableField` enum tuple in declaration order, and one nonempty repair obligation. On that branch, every derived schema/reference hash, reference, universe/dependency table, positive case/attempt, control, and oracle field is exactly `None`; `invariants == ()`; `result.obligations == (preflight.obligation,)`; and `validation_payload.nonclaims == H5_NONCLAIM_IDS` exactly, the same frozen tuple used by a completed H5 payload. No zero digest, empty string, dummy reference, empty successful case tuple, shortened preflight-specific nonclaim tuple, or partially trusted derived hash may be fabricated.

The displayed field orders are exact. `H5ValidationPayloadRecord` requires exact universe/dependency order, raw attempt/control/oracle equality with the result/evaluation, and `nonclaims == H5_NONCLAIM_IDS` on both completed and preflight-inconclusive branches. On the preflight-inconclusive branch it requires all optional fields to be `None` exactly as named by `preflight.unavailable_fields`. It canonicalizes every nested frozen record under the validation-payload domain, includes unavailable optionals explicitly as JSON `null` rather than omitting keys, and computes its own hash; `h5_validation_payload` is only the deterministic JSON-primitive projection of that closed record, not an independently assembled mapping.

For every positive and control execution, construct a fresh `UpdateRequest` from the mapped base rule contract and set `request_id` exactly to that case/control enum's `.value`; no positive request object or request hash is reused by a control. The two mappings displayed above are exhaustive and exact. In particular, control 6 (`CHANGED_INPUT_EQUAL_VALUE`) uses base rule `EXACT_STATE_TRANSITION_2_M`, and control 7 (`CHANGED_VALUE_SAME_INPUT`) uses base rule `GENERALIZED_EM_EMISSION_1`.

`H5DeltaOperandEvidence`, `H5DeltaImplementationEvidence`, and `H5DeltaAgreement` have exactly the displayed field orders. The production adapter derives its three operand records from the outcome's own complete before/after term traces and `H5DeltaAllowance`; the oracle adapter first reconstructs every `H5OracleTermEvidence` formula and every aggregate from the byte-only oracle's retained complete-term traces, then derives its gate records only from those validated oracle objects. Both adapters enforce the global formulas, exact `before,after,delta` roles, finite local evidence, and `operand_shaped=True`. `compare_h5_complete_delta` then stores the exact comparison rounding, total allowance, absolute error, and closed `<=` result. `all_delta_allowances_operand_shaped` is true if and only if all five positive cases have a finite passing `H5DeltaAgreement`, both implementations' `operand_shaped` flags are true, the production evidence revalidates against its outcome traces, the oracle evidence independently revalidates against its serialized oracle term traces and local constants, every aggregate/formula validates, and no evidence object, term trace, or allowance is shared by identity across implementations. No production value may fill a missing oracle trace field.

`compare_h5_exact_candidate` compares exactly the active fields listed in `H5_CANDIDATE_COMPARISON_OPERATION_COUNTS`, in that order, using the frozen scalar formula and conditions above. Exact z0/source comparisons record condition `1.0`; the M comparison records each implementation's `G` condition number supplied in its update diagnostics. Because `H5TransactionResult` deliberately does not expose a candidate object, Task 8 deterministically reconstructs each positive candidate from the same initial live state, exact `outcome.request`, and exact selected `outcome.damping`, dispatching to the corresponding exact/GEM/natural Task 7 proposal. Before comparison or serialization it requires the reconstructed provenance-bearing `candidate_sha256`, recognition hash, and model hash to equal `outcome.hashes.candidate_sha256`, `candidate_recognition_sha256`, and `candidate_model_sha256`. Separately, it recomputes `canonical_h5_semantic_state_bytes(reconstructed.recognition, reconstructed.model)` and its domain-separated semantic-state hash and requires that semantic hash to equal `outcome.after.evaluated_state_sha256`. A provenance candidate hash is never compared with a semantic evaluated-state hash. Any same-domain mismatch is `INCONCLUSIVE` with a reconstruction obligation. It stores the recomputed production semantic-state hash in `H5PositiveCaseResult`. The reconstructed candidate's canonical semantic bytes, not the final live state on a rejected natural case, are the `after_state_bytes` supplied to the byte-only delta oracle. This does not add a candidate field to any Task 7 result. Exact positive cases require a non-`None` passing candidate comparison; GEM/natural positive cases require `candidate_comparison is None`. All five require a passing independent complete-delta agreement. Production/oracle semantic hashes are always retained, but cross-implementation equality or inequality never determines PASS/FAIL/INCONCLUSIVE.

The exact ordered invariant tuple is:

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

The seven fault injections are exact and test/gate-only:

1. During `EXACT_Z0`, the after-evaluator omits expected affected factor `state_transition[2]`; outcome must be `FailedUpdateAttempt(AFTER_EVALUATION, FACTOR_COVERAGE_MISMATCH)` with the other eleven ordered records in `partial_after` and no fabricated after terms.
2. During `GENERALIZED_EM_EMISSION_1`, the after-evaluator omits its expected affected `emission[1]`; the same typed failure records the other eleven factors.
3. Starting from the valid GEM request, the proposal seam changes only `emission[1].bias[0]` to `math.nextafter(old, math.inf)`, producing distinct factor-input bytes and a finite complete delta inside the positive allowance. A decision seam then returns accept despite `abs(delta_elbo)<=epsilon_delta`; validation converts it to `FailedUpdateAttempt(DECISION, DECISION_POLICY_VIOLATION)` before commit.
4. Use the valid `NATURAL_GRADIENT_Z1` request and its normal detached working proposal, preserving its rule, request hash, `q[z1]` active block, snapshots, damping, and expected frozen-complement hash; pass only `producer_label=EXACT_COORDINATE` instead of `NATURAL_GRADIENT_PROPOSAL` into the freeze seam. This constructs an immutable mislabeled `H5CandidateDraft` and recomputes its draft hash, then the unchanged `H5CandidateSnapshot` constructor rejects the provenance mismatch. The result is `FailedUpdateAttempt(FREEZE, LABEL_PROVENANCE_MISMATCH)` before final-candidate construction, predecision capture, evaluation, or acceptance: `candidate_draft_sha256` and candidate recognition/model hashes are present, `candidate_sha256` and all predecision hashes are `None`, and exact before/final live/recognition/model/optimizer/RNG hashes prove rollback.
5. After the resolved-decrease natural candidate is rejected, the rollback seam returns a copied live state with only `q[z1].mean=math.nextafter(old, math.inf)` and RNG payload counter changed from zero to one. Final live/recognition/RNG hashes yield `FailedUpdateAttempt(COMMIT_OR_ROLLBACK, ROLLBACK_HASH_MISMATCH)`; the original caller-owned live object remains unchanged.
6. Starting from a fresh request with `request_id=H5ControlId.CHANGED_INPUT_EQUAL_VALUE.value` and every other field exactly equal to the `EXACT_STATE_TRANSITION_2_M` rule contract, for `state_transition[2]` reflect `alpha_0` about its fixed-complement scalar least-squares optimum
   `alpha_hat = E[z0*(z2-(B_base+s)*m2-c)] / E[z0**2]`, using `alpha_0' = 2*alpha_hat-alpha_0`. Evaluate both sides through the same completed-square quadratic form; their order-21/order-17 `float.hex()` pairs must be exactly equal while canonical input bytes differ. The factor must appear in `observed_affected_factor_ids` and must be absent from `value_changed_factor_ids`; failure to realize this exact fixture property is `INCONCLUSIVE`, not a relaxed control.
7. Starting from a fresh request with `request_id=H5ControlId.CHANGED_VALUE_SAME_INPUT.value` and every other field exactly equal to the `GENERALIZED_EM_EMISSION_1` rule contract, keep the `state_transition[2]` canonical input bytes unchanged while the factor-record seam adds exactly `1.0e-6` to its reported order-21 and order-17 values. Before any internally inconsistent `CompleteElboEvaluation` can be constructed, independent deterministic reevaluation of that same input produces the unmodified pair. The resulting `FailedUpdateAttempt(AFTER_EVALUATION, DETERMINISTIC_REEVALUATION_MISMATCH)` retains the corrupted record in `partial_after`, keeps the factor absent from `observed_affected_factor_ids`, includes it in diagnostic `value_changed_factor_ids`, and stores the unequal pair in `deterministic_reevaluation`.

Status construction is fail-closed with exact precedence `INCONCLUSIVE`, then `FAIL`, then `PASS`:

- `INCONCLUSIVE` takes precedence whenever preflight cannot construct the exact fixture/schema/reference (using the typed unavailable branch above), any required evidence is unavailable/missing/nonfinite, deterministic candidate reconstruction disagrees with the completed outcome hashes, a schema/cache cause is unresolved, or a required emission-touching comparison remains inside/on its complete allowance; obligations are nonempty and name each open phase. This remains `INCONCLUSIVE` when some other currently available evidence simultaneously shows a decisive failure. Preflight failures never become `FAIL` by inventing derived evidence.
- `FAIL` is considered only after all required evidence is available, finite, complete, and decisive, and then applies iff that complete evidence decisively falsifies a required positive, dependency, decision, rollback, oracle, or control invariant. A control is successful when it detects its injected fault; the injected fault itself does not make H5 fail.
- `PASS` is considered only after the first two predicates are false and requires preflight `READY` with no errors/unavailable fields, exact raw fixture/schema/graph identity, all five positives passing, all seven controls detecting their intended fault, every numerical record finite/complete/operand-shaped, and empty obligations.
- Unsupported MM is absent from this status mapping. Configuration resolution rejects it before `evaluate_h5`; a normal H5 run has no MM obligation.

- [ ] **Step 1: Write the failing promotion test for exact five-positive/seven-control order, every typed outcome, payload schema, byte identity, and all PASS/FAIL/INCONCLUSIVE contradictions.** Assert the exact preflight/error/unavailable/result/payload/evaluation field orders, the exhaustive one-owner `H5_PREFLIGHT_ERROR_KINDS_BY_PHASE` map, and every phase-valid optional. Malformed H1 bytes, wrong full update-spec digest, malformed update schema, and reference/schema construction failure must each return `INCONCLUSIVE` with two real raw digests, one phase-valid typed error/obligation, the full ordered unavailable tuple, the exact unchanged `H5_NONCLAIM_IDS`, JSON `null` for every derived optional, and no fabricated hash/reference/invariant/attempt/control/oracle. Reject every cross-phase error kind and any preflight-specific nonclaim shortening. Test exact status precedence by combining one decisive finite failed invariant with one unavailable required field and requiring `INCONCLUSIVE`, then restoring completeness and requiring `FAIL`. Include exact candidate-comparison field/count/order/formulas, deterministic candidate reconstruction from request+damping, same-domain equality with all recorded outcome hashes, an explicit rejection of candidate-hash versus semantic-hash comparison, permitted unequal cross-implementation semantic hashes, all five exact operand-shaped delta agreements reconstructed from the independent oracle term traces, and within/outside comparison allowance, exact boundaries `delta=-epsilon`, `delta=epsilon`, and just outside them; malformed partial/full outcomes; changed-input/equal-value and same-input/changed-value separation; rollback optimizer/RNG hashes; the mislabel control's present draft/recognition/model hashes and absent final-candidate/predecision hashes; no MM obligation; and a proof that path rereads are not performed. Assert every positive/control `outcome.request.request_id == case_or_control_id.value`, the complete two rule maps, control 6's exact-M base rule, and control 7's GEM base rule.

```python
def test_h5_gate_consumes_captured_bytes_and_requires_all_cases_and_controls():
    evaluation = evaluate_h5(CONFIG, h1_fixture_bytes=H1_BYTES, h5_update_spec_bytes=H5_BYTES)
    assert tuple(case.case_id for case in evaluation.result.positive_cases) == tuple(H5PositiveCaseId)
    assert tuple(control.control_id for control in evaluation.result.controls) == tuple(H5ControlId)
    assert evaluation.result.status is GateStatus.PASS
    assert evaluation.result.obligations == ()
    assert "valid_mm" not in " ".join(evaluation.result.obligations)

def test_h5_preflight_failure_is_typed_inconclusive_without_fabrication():
    evaluation = evaluate_h5(CONFIG, h1_fixture_bytes=H1_BYTES, h5_update_spec_bytes=b"invalid")
    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert evaluation.result.preflight.errors[0].kind is H5PreflightErrorKind.UPDATE_SPEC_RAW_DIGEST_MISMATCH
    assert tuple(evaluation.result.preflight.unavailable_fields) == tuple(H5UnavailableField)
    assert evaluation.reference is None
    assert evaluation.positive_attempts is None
    assert evaluation.controls is None
    assert evaluation.oracle_results is None
```

- [ ] **Step 2: Run Task 8 RED.**

```powershell
python -m pytest tests/promotion/test_h5_gate.py -q
```

Expected: collection fails because the H5 gate does not exist.

- [ ] **Step 3: Implement test/gate-only fault seams and the seven controls without changing production globals.** Each seam is injected into one evaluator/controller instance and restored by object disposal, not monkeypatch leakage.

- [ ] **Step 4: Implement the typed preflight branch, five positive cases, gate invariant/status constructor, byte-bearing `evaluate_h5`, and complete payload.** Build every fresh request from the exact case/control enum value and mapped base rule. Reconstruct positive candidates deterministically from request plus selected damping and require exact outcome hashes before comparison. Construct both sides of every delta agreement from their independent evidence. Payload includes the preflight carrier; phase-valid optional raw/canonical/schema/reference hashes; universes/graph; every complete or partial outcome; both-order factor/term values; operand-shaped production/oracle delta evidence and agreement; allowances; dependency/value diagnostics; producer/request labels/IDs; line search; all snapshot/live/optimizer/RNG hashes; oracle comparisons; controls; invariants; status/obligations; and H6–H8/training nonclaims. It contains no MM configuration field or obligation; Task 9 owns that later config-only resolution surface.

  The payload must also preserve distinct `candidate_draft_sha256` and `candidate_sha256` values, including the exact phase-valid `None` states specified by `UpdateHashRecord`; it never substitutes a draft hash for a final candidate hash.

- [ ] **Step 5: Run Task 8 GREEN.**

```powershell
python -m pytest tests/promotion/test_h5_gate.py -q
```

Expected: all Task 8 tests pass.

- [ ] **Step 6: Update the H5 preregistration only to copy the now-implemented exact raw digest and any mechanically generated schema hash; do not change formulas, cases, controls, thresholds, or status rules after observing outcomes.** Rerun the focused Task 8 test if this documentation is imported by the payload test.

- [ ] **Step 7: Commit Task 8.**

```powershell
git add verification/h5_gate.py tests/promotion/test_h5_gate.py docs/preregistrations/2026-07-21-h5-update-coherence.md
git commit -m "feat: add fail-closed H5 gate"
```

## H4/H5 Export-File Serialization

- `vfe4/types/__init__.py`: apply Task 5 exports only after the repaired H4 Task 1 exports are present. Rebase first and append H5 names without replacing H4 names.
- `vfe4/inference/__init__.py`: H4 Task 2 owns its first edit. Implement `h5_updates.py` and tests independently, then serialize the H5 export patch after H4 Task 2 merges.
- `verification/numpy_oracles/__init__.py`: H4 Task 3 owns its first edit. Implement the H5 oracle module independently, then serialize exports after H4 Task 3 merges.
- Do not run H4 and H5 workers concurrently in the same worktree. Never resolve these export collisions by reverting, stashing, or overwriting another task's changes.
- H5 Tasks 5–8 import only stable H1/H2/H3 types and `GateStatus`/`InvariantResult`; they do not import H4 solver, timing, statistics, or gate modules.

---


### Task 9: Extend Typed Configuration, One Click-Run, Atomic Artifact Family, and Environment Provenance Through H5

**Files:**

- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `vfe4/config/__init__.py`
- Modify: `verification/run_gates.py`
- Modify: `vfe4/artifacts/provenance.py`
- Modify: `verify_vfe4.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_atomic_artifacts.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h5-update-coherence.md`

**Interfaces and compatibility:**

#### Task 9 H4 integration boundary

Task 9 no longer defines any H4 schema, H4-section resolver formula, numerical
literal, preregistration correction, classifier, trace, condition, allowance,
or gate payload shape. It owns only the existing coupled H1-H5 presence rule,
full-config projection check, and runner/artifact integration.

Its corrected scope is:

- add `H5ValidationConfig`, attach both independently typed `h4` and `h5`
  sections to `ResolvedConfig`, and accept exactly the existing prefix set
  `("H1",)`, `("H1","H2")`, `("H1","H2","H3")`, and
  `("H1","H2","H3","H4","H5")`; H4 and H5 are both present exactly for
  the last prefix and both absent for every shorter prefix;
- reuse Task 3's `resolve_h4_validation_config` byte-for-byte inside full
  resolution, prove its canonical JSON/SHA-256 and complete nested
  `H4SolveProtocol` are unchanged, and add no second H4 resolver;
- reject `("H1","H2","H3","H4")`, every H5-without-H4 or H4-without-H5
  selection, and every reordered, duplicate, or H6-H8 prefix;
- wire the already implemented `evaluate_h4(config.h4,
  h3_coupled_bytes=..., h3_zero_bytes=...)` and `h4_validation_payload` into
  the runner exactly once, after H3 and before H5;
- publish the preexisting `H4ValidationArtifact`; do not rebuild its payload or
  recompute H4 status/config/coverage in the runner; the coupled manifest and
  provenance record both its `h4_config_sha256` and the distinct full
  resolved-config SHA-256, and require the latter's canonical H4 projection to
  hash to the former;
- add H5 configuration, H5 fixture capture, H5 evaluation, coupled manifest,
  provenance, launcher, integration tests, and documentation only;
- remove `docs/preregistrations/2026-07-21-h4-information-cost.md` from Task 9's
  file list and remove all H4 resolver-formula RED/GREEN work from Task 9.

Thus Task 9 shrinks from "define H4/H5 config and H4 payload" to "integrate
the already resolved and typed H4 section with H5 under the one coupled
H1-H5 milestone, then wire two already typed gates into one artifact family."

- Extend accepted gate prefixes only to `("H1",)`, `("H1","H2")`, `("H1","H2","H3")`, and `("H1","H2","H3","H4","H5")`. Do not accept H4 without H5, H5 without H4, reordered/duplicate gates, or any H6--H8 prefix in this milestone.
- Add `h4: H4ValidationConfig | None` and `h5: H5ValidationConfig | None`. Both are absent for shorter prefixes and both are required for the H5 prefix. They remain separately hashed sections and produce separate results. `H5ValidationConfig` copies Task 5's exact fixture ID, raw SHA-256, canonical SHA-256, objective-schema SHA-256, factor-input schema version/SHA-256, all three ordered identifier universes, and ordered rule/positive/control IDs. Resolution validates the declared full 64-hex raw digest against the frozen Task 5 constant and rejects any mismatch, truncation, short digest prefix, canonical/schema drift, or ordering change. `resolve_config` never reads, captures, hashes, parses, or receives update-spec bytes. After resolution, the runner is the sole capture owner: it reads the fixture exactly once, immediately recomputes `SHA256(captured_bytes)`, compares it to the resolved full digest before any H1--H5 gate evaluation or parser decode, and passes that same bytes object by identity to H5. The comparison is retained as runner preflight evidence, not converted into a global artifact exception. On a mismatch, ordered evaluation still reaches H5, whose typed preflight publishes `INCONCLUSIVE` atomically with the real observed digest, unavailable derived fields as JSON null, empty update-hash records, and unchanged `H5_NONCLAIM_IDS`; H1--H4 payloads remain publishable. A completed/`READY` H5 result must still agree exactly with the configured digest.
- H5 v1 config contains `enabled_update_rules=tuple(H5UpdateRule)`, `enabled_update_labels=(UpdateLabel.EXACT_COORDINATE, UpdateLabel.GENERALIZED_EM, UpdateLabel.NATURAL_GRADIENT_PROPOSAL)`, and `mm_proof_artifact=None`. Resolution requires those labels to equal the ordered unique producer labels induced by the enabled rule contracts. Adding `UpdateLabel.VALID_MM` is the concrete unsupported-MM request and is rejected before constructing an `UpdateRequest`, before reading update state, and before calling `evaluate_h5`. Absence of an MM artifact or request never creates an H5 attempt, invariant, obligation, or gate status.
- Reuse Task 3's `resolve_h4_validation_config` byte-for-byte inside full resolution; require its canonical JSON/SHA-256 and complete nested `H4SolveProtocol` to remain unchanged, require the full resolved configuration's canonical H4 projection to hash to that same H4 digest, and add no second H4 resolver.
- Extend the explicit result union to include `H4GateResult` and `H5GateResult`; do not merge their measurements or status.

- [ ] **Step 1: Write focused configuration, integration, and artifact tests.** Assert the one editable `CONFIG` resolves to ordered H1--H5, reuses Task 3's exact resolved H4 section without changing its canonical JSON/SHA-256 or nested `H4SolveProtocol`, and projects that section byte-for-byte from the full resolved config; plus exact H5 conditional update-spec fixture ID, full raw/canonical digests, objective/factor-input schema fields, identifier universes, five rule contracts, producer-label order, five positive cases, seven controls, quadrature orders `21/17`, deterministic-convergence-plus-rounding budgets, zero stochastic contribution, and epsilon formula. Reject any second H4 resolver, canonical H4 projection drift, H4-only prefix, H5-without-H4 prefix, or coupled-prefix section absence. Reject `VALID_MM` with `mm_proof_artifact=None` before request/state/evaluator construction, and assert that the supported config creates no missing-MM obligation. Test every H5 delta boundary. Resolve every shorter compatibility prefix and prove it contains no H4/H5 config, does not read/hash/capture H4/H5/update-spec inputs, does not run timing/updates, and publishes no H4/H5 payload/provenance keys.

  One mocked H1--H5 `main()` call evaluates each gate once and publishes exactly one manifest-checked directory containing:

  ```text
  config.json
  environment.json
  provenance.json
  validation/h1.json
  validation/h2.json
  validation/h3.json
  validation/h4.json
  validation/h5.json
  manifest.sha256
  ```

  The eight JSON payload paths above are the exact lexicographic manifest
  order; `manifest.sha256` is written last and is not one of its own entries.

  Assert H4 and H5 statuses may differ and both survive round-trip publication. Preserve path containment, alias/reparse-point defenses, no-overwrite atomic publication, and prior manifests.

- [ ] **Step 2: Run the Task 9 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: failures show full resolution and the runner do not yet integrate the already resolved H4 section with H5, and no coupled H4/H5 payloads or environment fields exist.

- [ ] **Step 3: Add H5 configuration and integrate the existing typed H4 section with fail-closed resolution.** Reuse Task 3's `resolve_h4_validation_config` byte-for-byte and prove that the full-config H4 projection has the same canonical JSON/SHA-256 and complete nested `H4SolveProtocol`; do not define or recompute any H4 formula, literal, classifier, condition, allowance, trace, or payload here. Canonicalize every H5 frozen literal. Reject an H5 fixture/raw/canonical/objective-schema/factor-input-schema field that differs from Task 5, any raw digest shorter than the exact 64-hex SHA-256 literal, any universe/rule/label/positive/control order change, quadrature-order drift, deterministic-budget drift, nonzero stochastic contribution, or delta-formula drift. Require exact equality between enabled rule producer labels and `enabled_update_labels`. Reject unsupported `VALID_MM` at configuration resolution when `mm_proof_artifact=None`; do not pass that condition into attempt or gate status logic. Reject H4/H5 section presence for shorter prefixes and absence of either section for the coupled prefix; reject `("H1","H2","H3","H4")`, every reordered or duplicate selection, and every H6--H8 prefix.

  Export `H5ValidationConfig` from `vfe4.config` symmetrically with the
  existing H4 type. This step compares only typed literals and frozen
  constants; it performs no fixture I/O and no actual raw-byte digest
  computation. Focused tests patch the fixture reader to fail if resolution
  attempts any capture.

- [ ] **Step 4: Extend conditional one-time capture and ordered evaluation.** Capture `h1-v1` once for H1/H2/H5, H3 coupled/zero bytes once for H3/H4 only when consumed, and `h5_conditional_update_v1.json` bytes once only for the coupled H1/H2/H3/H4/H5 prefix. Immediately compute and compare its full raw-byte SHA-256 against the exact Task 5/config literal before any gate evaluation or parser decode; no short digest prefix is accepted or exposed in configuration, gate arguments, provenance, payloads, or artifacts. Retain that comparison without raising a global publication error. Pass the same captured H5 byte object by identity to `evaluate_h5` and every production/oracle adapter; H5 receives both `h1_fixture_bytes` and `h5_update_spec_bytes`. Evaluate H1, H2, H3, H4, H5 in order. A raw-digest mismatch is owned by H5's typed preflight and produces an atomic H5 `INCONCLUSIVE` payload with the real observed digest, exact unavailable/null fields, empty positive/control update-hash records, and unchanged nonclaims while preserving the H1--H4 payloads; a `READY` H5 result must match the configured digest. Invoke the already implemented `evaluate_h4(config.h4, h3_coupled_bytes=..., h3_zero_bytes=...)` exactly once after H3 and before H5, then pass its preexisting `H4ValidationArtifact` to `h4_validation_payload`; the runner does not rebuild H4 status, coverage, or payload content. Publish only after both expensive gates return. Shorter prefixes must neither read, hash, capture, nor publish the H5 update-spec. Aggregate status is `fail` if any gate fails, otherwise `inconclusive` if any is inconclusive, otherwise `pass`.

- [ ] **Step 5: Extend environment and provenance.** Preserve current source/config/dirty-content security fields and expose the canonical `dirty_content_digest` used by milestone preflight/rechecks. Add timing clock implementation/resolution/monotonicity, process CPU affinity, logical/physical CPU counts when available, processor/platform, PyTorch intra/inter-op threads, `torch.__config__.show()` digest/text, NumPy BLAS configuration digest/text, CUDA availability (expected false for H4), and exact values/presence of `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. Publish the preexisting `H4ValidationArtifact` without rebuilding it. Record ordered gate states, the Task 3 `h4_config_sha256`, the distinct full resolved-config SHA-256, and proof that the latter's canonical H4 projection hashes to the former. Record the distinct H5 config hash, H5 raw/canonical update-spec, objective-schema, factor-input-schema, recognition/model/reference/payload hashes, and the ordered per-positive/per-control `UpdateHashRecord` payloads as transaction evidence; Task 9 does not invent an aggregate transaction SHA-256 absent from the frozen Task 7 API. Also record `fixture_hashes["h5-conditional-update-v1"]`, `gate_fixture_consumers["H5"]=("h1-v1","h5-conditional-update-v1")`, H5 universes/rule/control orders/quadrature/allowance rules, and H4/H5 bounded-claim/nonclaim tags. Shorter prefixes contain none of the H5 update-spec fields.

- [ ] **Step 6: Extend the one launcher and bounded documentation.** Keep one `CONFIG`, `main`, and script guard. Print H1--H5 statuses separately and one artifact path. README documents only the coupled integration of the already frozen H4 artifact with H5 and does not restate or revise H4 numerical formulas, preregistration, classifier, trace, condition, allowance, or payload shape; it does not prestate H4 speed or H5 pass results. The H5 preregistration states the exact conditional recognition law, raw/canonical/schema bindings, five rules/positives, seven controls, and config-only unsupported-MM rejection without inventing a missing-MM gate obligation. Use exact live path case `Manuscripts/...` in every source citation and add a focused documentation assertion that rejects any differently cased variant. Explicitly defer H6--H8 and training.

- [ ] **Step 7: Run the Task 9 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: all accepted prefixes preserve their exact surfaces; one mocked H1--H5 click-run publishes eight JSON files plus one valid manifest; H4/H5 remain separate; prior prefix artifacts contain no future-gate data.

- [ ] **Step 8: Commit Task 9.**

  ```powershell
  git add vfe4/config/schema.py vfe4/config/resolve.py vfe4/config/__init__.py verification/run_gates.py vfe4/artifacts/provenance.py verify_vfe4.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py README.md docs/preregistrations/2026-07-21-h5-update-coherence.md
  git commit -m "feat: publish separate H4 and H5 verification gates"
  ```

---

### Task 10: Produce One Coupled Exact-Revision H4/H5 Milestone Record

**Files:**

- Modify: none. Every tracked protocol, implementation, test, preregistration, launcher, and artifact-schema file is committed before selecting the candidate revision.
- Produce outside tracked source: `C:\tmp\vfe4-h4-h5-preflight.json`, `C:\tmp\vfe4-h4-h5-milestone.xml`, one atomic run directory under the configured run root, and `.verification/h4-h5-<FULL_HEAD>-ledger.json`.
- Preserve `.verification/ledger.json`, all `.verification/h3-*-ledger.json`, and any prior `.verification/h4-h5-*-ledger.json` byte-for-byte. Do not commit `.verification` or generated run artifacts.

**Why one milestone is allowed:** H4 and H5 are evaluated by the same ordered click-run, config snapshot, source revision, environment capture, JUnit revision, artifact manifest, and fixture-byte snapshot. Their gate results, payloads, statuses, and ledger claims remain separate. If either implementation changes, both evidence sets are invalidated and the replacement candidate again uses one common revision.

Run every Task 10 code block in one PowerShell 5.1 session from the repository root. The retained variables bind the one JUnit run, one click invocation, one artifact, and one ledger; never rerun a native command merely to reconstruct lost shell state.

- [ ] **Step 1: Fail-closed on tracked scope, unexpected untracked content, revision, dirty-content identity, activation, and preserved ledgers.**

  ```powershell
  $candidateHeadOutput = @(git rev-parse HEAD)
  $candidateHeadExit = $LASTEXITCODE
  if ($candidateHeadExit -ne 0) { throw 'git rev-parse HEAD failed during H4/H5 preflight' }
  $candidateHead = ($candidateHeadOutput -join '').Trim()
  if ($candidateHead.Length -ne 40) { throw 'H4/H5 requires a full 40-character HEAD' }
  $requiredTracked = @(
      '.gitattributes',
      'docs/superpowers/plans/2026-07-21-vfe4-h4-h5-cost-update.md',
       'docs/preregistrations/2026-07-21-h4-information-cost.md',
       'docs/preregistrations/2026-07-21-h5-update-coherence.md',
       'vfe4/validation/fixtures/h5_conditional_update_v1.json',
       'vfe4/validation/h5_update_spec.py',
       'vfe4/types/h4.py', 'vfe4/types/updates.py', 'vfe4/types/h5_schema.py',
       'vfe4/types/__init__.py',
       'vfe4/generative/reference_h4.py', 'vfe4/generative/__init__.py',
       'vfe4/inference/h4_instrumentation.py', 'vfe4/inference/h4_solvers.py',
       'vfe4/inference/h5_updates.py', 'vfe4/inference/__init__.py',
       'vfe4/objective/dependency_graph.py', 'vfe4/objective/h5_complete.py',
       'vfe4/objective/__init__.py',
       'vfe4/numerics/h5_budget.py', 'vfe4/numerics/__init__.py',
       'verification/numpy_oracles/h4_gaussian.py',
       'verification/numpy_oracles/h5_updates.py',
       'verification/numpy_oracles/__init__.py',
       'verification/h4_records.py', 'verification/h4_budget.py',
       'verification/h4_statistics.py',
       'verification/h4_gate.py', 'verification/h5_gate.py',
       'verification/run_gates.py',
      'vfe4/config/__init__.py', 'vfe4/config/schema.py',
      'vfe4/config/resolve.py',
      'vfe4/artifacts/__init__.py', 'vfe4/artifacts/provenance.py',
      'verify_vfe4.py', 'README.md',
      'tests/unit/test_h4_problem.py', 'tests/unit/test_h4_solvers.py',
      'tests/unit/test_h4_instrumentation.py', 'tests/unit/test_h4_records.py',
      'tests/unit/test_h4_budget.py', 'tests/unit/test_h4_statistics.py',
      'tests/oracle/test_h4_numpy_oracle.py', 'tests/promotion/test_h4_gate.py',
        'tests/unit/test_h5_update_types.py', 'tests/unit/test_h5_objective_schema.py',
        'tests/unit/test_h5_dependency_graph.py', 'tests/unit/test_h5_update_spec.py',
       'tests/unit/test_h5_complete_objective.py', 'tests/unit/test_h5_budget.py',
      'tests/unit/test_h5_updates.py', 'tests/oracle/test_h5_update_oracle.py',
      'tests/promotion/test_h5_gate.py', 'tests/unit/test_config.py',
      'tests/unit/test_atomic_artifacts.py', 'tests/integration/test_verify_vfe4.py'
  )
  $tracked = @(git ls-files -- $requiredTracked)
  if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed during H4/H5 tracked-scope preflight' }
  $missingTracked = @($requiredTracked | Where-Object { $_ -notin $tracked })
  if ($missingTracked.Count -ne 0) {
      throw "H4/H5 candidate has missing or untracked required files: $($missingTracked -join ', ')"
  }
  git diff --exit-code
  if ($LASTEXITCODE -ne 0) { throw 'tracked worktree changes block H4/H5 preflight' }
  git diff --cached --exit-code
  if ($LASTEXITCODE -ne 0) { throw 'staged changes block H4/H5 preflight' }
  $nonignoredUntracked = @(git ls-files --others --exclude-standard)
  if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed while checking untracked H4/H5 content' }
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
  $ledger = ".verification/h4-h5-$candidateHead-ledger.json"
  if (Test-Path -LiteralPath $ledger) {
      throw "revision-specific H4/H5 ledger exists and must not be overwritten: $ledger"
  }
  $ledgerFullPath = [System.IO.Path]::GetFullPath($ledger)
  $preflightDirtyOutput = @(& python -c "from pathlib import Path; from verify_vfe4 import CONFIG; from vfe4.artifacts.provenance import dirty_content_digest; print(dirty_content_digest(Path.cwd(), Path(CONFIG['artifacts']['run_root'])))")
  $preflightDirtyExit = $LASTEXITCODE
  if ($preflightDirtyExit -ne 0) { throw 'dirty-content digest capture failed during H4/H5 preflight' }
  $preflightDirtyDigest = ($preflightDirtyOutput -join '').Trim()
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
      expected_ledger = $ledger
      expected_ledger_path = $ledgerFullPath
      prior_ledgers = @($ledgerHashes | ForEach-Object {
          [ordered]@{ path = $_.Path; sha256 = $_.Hash.ToLowerInvariant() }
      })
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 -LiteralPath C:\tmp\vfe4-h4-h5-preflight.json

  function Assert-H4H5CandidateState {
      param(
          [Parameter(Mandatory = $true)][string]$Stage,
          [switch]$RequireNoActiveMarker
      )

      $currentHeadOutput = @(git rev-parse HEAD)
      $currentHeadExit = $LASTEXITCODE
      if ($currentHeadExit -ne 0) { throw ("{0}: git rev-parse HEAD failed" -f $Stage) }
      $currentHead = ($currentHeadOutput -join '').Trim()
      if ($currentHead -ne $candidateHead) { throw ("{0}: candidate HEAD changed" -f $Stage) }

      git diff --exit-code
      if ($LASTEXITCODE -ne 0) { throw ("{0}: tracked worktree changed" -f $Stage) }
      git diff --cached --exit-code
      if ($LASTEXITCODE -ne 0) { throw ("{0}: staged index changed" -f $Stage) }

      $currentUntracked = @(git ls-files --others --exclude-standard)
      if ($LASTEXITCODE -ne 0) { throw ("{0}: git ls-files failed" -f $Stage) }
      $currentUnexpected = @(
          $currentUntracked |
              Where-Object { $_ -ne '.verification' -and -not $_.StartsWith('.verification/') }
      )
      if ($currentUnexpected.Count -ne 0) {
          throw ("{0}: unexpected untracked content: {1}" -f $Stage, ($currentUnexpected -join ', '))
      }

      $currentDirtyOutput = @(& python -c "from pathlib import Path; from verify_vfe4 import CONFIG; from vfe4.artifacts.provenance import dirty_content_digest; print(dirty_content_digest(Path.cwd(), Path(CONFIG['artifacts']['run_root'])))")
      $currentDirtyExit = $LASTEXITCODE
      if ($currentDirtyExit -ne 0) { throw ("{0}: dirty-content digest capture failed" -f $Stage) }
      $currentDirtyDigest = ($currentDirtyOutput -join '').Trim()
      if ($currentDirtyDigest -ne $preflightDirtyDigest) {
          throw ("{0}: dirty-content digest changed" -f $Stage)
      }
      if ($RequireNoActiveMarker -and (Test-Path -LiteralPath '.verification/active.json')) {
          throw ("{0}: verification activation marker must not exist" -f $Stage)
      }
  }
  ```

  Expected: exact 40-character revision; every named plan/preregistration/source/config/launcher/test is tracked; no tracked modification; no nonignored untracked content outside `.verification`; no active marker; the revision-specific ledger does not yet exist; valid dirty-content digest; and a machine-readable retained SHA-256 table for every prior ledger. Do not delete unexpected content to make this pass; preserve it and resolve ownership.

- [ ] **Step 2: Run the only coupled milestone full regression and parse JUnit.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h4-h5-milestone.xml
  if ($LASTEXITCODE -ne 0) { throw 'H4/H5 milestone pytest failed' }

  [xml]$junitDocument = Get-Content -Raw -LiteralPath C:\tmp\vfe4-h4-h5-milestone.xml
  $junitSuites = @($junitDocument.SelectNodes('/testsuites/testsuite | /testsuite'))
  if ($junitSuites.Count -eq 0) { throw 'JUnit contains no top-level testsuite' }
  $junitTotals = [ordered]@{ suites = $junitSuites.Count; tests = 0L; failures = 0L; errors = 0L; skipped = 0L }
  foreach ($suite in $junitSuites) {
      foreach ($field in @('tests', 'failures', 'errors', 'skipped')) {
          $parsed = 0L
          $raw = [string]$suite.GetAttribute($field)
          if (-not [long]::TryParse($raw, [ref]$parsed) -or $parsed -lt 0) {
              throw "invalid JUnit $field total: $raw"
          }
          $junitTotals[$field] += $parsed
      }
  }
  if ($junitTotals.tests -le 0 -or $junitTotals.failures -ne 0 -or $junitTotals.errors -ne 0) {
      throw 'JUnit totals are inconsistent with successful pytest completion'
  }
  $junitSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath C:\tmp\vfe4-h4-h5-milestone.xml).Hash.ToLowerInvariant()
  Assert-H4H5CandidateState -Stage 'after JUnit' -RequireNoActiveMarker
  ```

  Expected: pytest exits zero; suite/test/failure/error/skip totals and the retained JUnit hash come only from `C:\tmp\vfe4-h4-h5-milestone.xml`; and the executable identity recheck passes. Do not report terminal dots or remembered earlier totals. Do not run another full suite unless a subsequent source/test/config/protocol change invalidates this candidate.

- [ ] **Step 3: Run the single full click-run and verify the artifact without rerunning H4.**

  ```powershell
  $clickOutput = @(python verify_vfe4.py 2>&1)
  $clickExitCode = $LASTEXITCODE
  if ($clickExitCode -notin @(0, 1)) { throw "click-run raised or returned unsupported exit code $clickExitCode" }
  $clickLines = @($clickOutput | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_.Length -ne 0 })
  if ($clickLines.Count -ne 6) { throw 'click-run must emit exactly five status lines and one artifact line' }

  $gateStatuses = [ordered]@{}
  for ($index = 0; $index -lt 5; $index++) {
      $gate = "H$($index + 1)"
      if ($clickLines[$index] -notmatch ('^{0}: (pass|fail|inconclusive)$' -f [regex]::Escape($gate))) {
          throw "invalid or out-of-order click status line: $($clickLines[$index])"
      }
      $gateStatuses[$gate] = $Matches[1].ToUpperInvariant()
  }
  if ($clickLines[5] -notmatch '^artifact: (.+)$') { throw 'click-run emitted no unique artifact path' }
  $artifactText = $Matches[1].Trim()
  $artifactItem = Get-Item -LiteralPath $artifactText -Force -ErrorAction Stop
  if (-not $artifactItem.PSIsContainer) { throw 'click artifact path is not a directory' }
  $artifactPath = $artifactItem.FullName

  $allFivePass = @($gateStatuses.Values | Where-Object { $_ -eq 'PASS' }).Count -eq 5
  $nonpassCount = @($gateStatuses.Values | Where-Object { $_ -in @('FAIL', 'INCONCLUSIVE') }).Count
  if ($clickExitCode -eq 0 -and -not $allFivePass) {
      throw 'click exit 0 is permitted only when all five gates are PASS'
  }
  if ($clickExitCode -eq 1 -and ($allFivePass -or $nonpassCount -lt 1)) {
      throw 'click exit 1 requires the artifact plus at least one FAIL or INCONCLUSIVE gate'
  }

  $expectedManifestEntries = @(
      'config.json',
      'environment.json',
      'provenance.json',
      'validation/h1.json',
      'validation/h2.json',
      'validation/h3.json',
      'validation/h4.json',
      'validation/h5.json'
  )
  $manifestPath = Join-Path $artifactPath 'manifest.sha256'
  $manifestLines = @(Get-Content -LiteralPath $manifestPath -Encoding UTF8 -ErrorAction Stop)
  if ($manifestLines.Count -ne $expectedManifestEntries.Count) { throw 'manifest must contain exactly eight entries' }
  for ($index = 0; $index -lt $expectedManifestEntries.Count; $index++) {
      $line = [string]$manifestLines[$index]
      if ($line -notmatch '^([0-9a-f]{64})  (.+)$') { throw "invalid manifest line: $line" }
      $declaredHash = $Matches[1]
      $relativePath = $Matches[2]
      if ($relativePath -ne $expectedManifestEntries[$index]) {
          throw "manifest entry is missing, extra, or out of lexicographic order: $relativePath"
      }
      $payloadPath = Join-Path $artifactPath ($relativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar))
      if (-not (Test-Path -LiteralPath $payloadPath -PathType Leaf)) { throw "manifest payload missing: $relativePath" }
      $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $payloadPath).Hash.ToLowerInvariant()
      if ($actualHash -ne $declaredHash) { throw "manifest payload hash mismatch: $relativePath" }
  }
  $artifactPrefix = $artifactPath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
  $actualArtifactFiles = @(
      Get-ChildItem -LiteralPath $artifactPath -File -Recurse -Force |
          ForEach-Object { $_.FullName.Substring($artifactPrefix.Length).Replace('\', '/') } |
          Sort-Object
  )
  $expectedArtifactFiles = @(($expectedManifestEntries + 'manifest.sha256') | Sort-Object)
  if ($actualArtifactFiles.Count -ne $expectedArtifactFiles.Count -or
      (($actualArtifactFiles -join "`n") -cne ($expectedArtifactFiles -join "`n"))) {
      throw 'artifact contains missing or extra files beyond the exact eight JSON payloads and manifest'
  }
  $manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
  ```

  Expected: the launcher prints separate H1, H2, H3, H4, and H5 statuses and one run directory. Independently recompute `manifest.sha256`; verify exact source/config/environment identity; artifact `dirty_content_digest == preflightDirtyDigest`; raw fixture and H4 problem hashes; H4 prior/effective/restored thread fields; exact indexed traversal and parity formula; separate warmup/timed event order with warmups excluded from balance; complete H4 raw timing table; primary observed balance equal to the literal 20-row table, ten `6/5` plus ten `5/6` seeds, and exactly `110 AB/110 BA`; bootstrap/envelope/budget/decisiveness metadata; raw H5 update-spec fixture ID/SHA-256/schema and H5-only fixture-consumer provenance; successful H5 preflight with exact phase/error map, no unavailable fields, and exact `H5_NONCLAIM_IDS`; complete H5 factor-input/affectedness/quadrature/allowance attempts; five independently derived operand-shaped production/oracle delta agreements whose oracle aggregates reproduce from the serialized both-order 12-term traces; deterministic candidate reconstructions with provenance hashes matched only to outcome provenance hashes and semantic hash matched only to `after.evaluated_state_sha256`; exact case/control request IDs and base-rule mappings; immutable model-snapshot ownership/no-alias evidence; and seven controls; and separate `validation/h4.json` / `validation/h5.json`. Recheck `HEAD`, both diffs, unexpected-untracked rule, and dirty digest after inspection. This is the only full 20-seed H4 timing execution for the candidate.
  For H4 also require the four ordered top-level global condition summaries to
  have counts `120/2640/2120/23320`, their per-problem sums to agree, and the
  oracle/native-diagnostic stream scalar counts to match the frozen formulas.

  After every listed artifact-field inspection succeeds, execute:

  ```powershell
  Assert-H4H5CandidateState -Stage 'after click inspection' -RequireNoActiveMarker
  ```

- [ ] **Step 4: Have fresh reviewers inspect existing evidence only.** Assign at least these independent reviews:

  - H4 protocol/statistics reviewer: exact horizon/seed/kind traversal with independent indices, common factors, independent arms, all three warmup pairs before all 11 timed pairs per problem, AB iff `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, warmups excluded from balance, primary observed rows exactly equal to the literal 20-row balance table, ten `6/5` plus ten `5/6` timed seeds, exact aggregate `110 AB/110 BA`, no conversion/hashing/diagnostics between timed repetitions, fixed batched conversion order, seeds as inferential units, bootstrap implementation, and threshold/status mapping;
  - H4 numerical/runtime reviewer: exact optimum, inclusive scaled conditioning envelope and boundaries, distinct per-problem/global condition-summary counts/digests, exact oracle/native-diagnostic headers/lane counts, `h/J`/moment/objective equivalence, Task 3-owned one-pass six-invariant allowance consumption, explicit anchor/scaled repetition identity, numeric-vector-bound independent headers, operand-local oracle/adapter operation routes and conditions, exact solver contribution, strict allowance/scale cap, no unbalanced H2 diagnostic, canonical immutable selected moments, real-operation instrumentation, raw timing/environment/BLAS/affinity provenance;
  - H5 theory/dependency reviewer: exact-case `Manuscripts/...` Markov blankets, the conditional categorical recognition law, source-independent continuous reconstruction, same complete ELBO, dependency prediction versus input-hash-derived observed affected sets, exact/source/M/GEM semantics, factor-universe completeness, and proof that MM is absent from attempt/gate paths;
  - H5 implementation/transaction reviewer: captured update-spec raw bytes/digest/parser/schema; typed pre-reference `INCONCLUSIVE` with the exact phase/error map, real raw digests, exact unavailable fields/JSON nulls, unchanged `H5_NONCLAIM_IDS`, `INCONCLUSIVE`-before-`FAIL` precedence, and no fabricated reference/schema/case/control/oracle evidence; proof that no short fixture-digest prefix is accepted or exposed anywhere in the H5 parser/config/gate/artifact path; immutable recognition and model-snapshot ownership with declared shared storage only; differentiable-working versus immutable-snapshot boundary; fixed recognition M-block; order-21/order-17 convergence estimates for every term; independently derived production/oracle before/after/delta operand evidence, self-sufficient oracle 12-term traces, and exact comparison allowance; deterministic candidate reconstruction with strict provenance-to-provenance and semantic-to-semantic hash checks; exact case/control request IDs, including exact-M control 6 and GEM control 7; complete total allowances and exact epsilon formula; emission-touching indecision; acceptance/rollback hashes; cache/reuse proofs; seven controls; and value-change diagnostic nonauthority;
  - artifact/compatibility reviewer: required tracked-file list, no unexpected untracked content, stable dirty-content digest, separate H4/H5 statuses/payloads, exact prefix behavior, H5 full raw-digest-only provenance with no accepted/exposed short prefix, atomic manifest, prior-ledger hashes, H6--H8/training nonclaims.

  Reviewers cite source lines, focused command outputs, JUnit XML, click artifact fields, and preregistrations. They do not rerun tests or timings. Before ledger activation, resolve every Critical/Important issue by returning to its owning task; any tracked change invalidates the candidate and requires one replacement coupled milestone run at the new revision. Then execute the review-boundary recheck:

  ```powershell
  Assert-H4H5CandidateState -Stage 'after reviews' -RequireNoActiveMarker
  ```

- [ ] **Step 5: Start, populate, and validate the coupled revision-specific ledger.** Read the verification contract and code, mathematics, evidence, experiment, and general criterion files before assigning states. Recheck revision/activation, then:

  ```powershell
  $preflight = Get-Content -Raw -LiteralPath C:\tmp\vfe4-h4-h5-preflight.json | ConvertFrom-Json
  if ([string]$preflight.candidate_head -ne $candidateHead) { throw 'preflight candidate HEAD changed in memory' }
  if ([string]$preflight.dirty_content_digest -ne $preflightDirtyDigest) { throw 'preflight dirty digest changed in memory' }
  $ledger = [string]$preflight.expected_ledger
  if ([System.IO.Path]::GetFullPath($ledger) -ne [string]$preflight.expected_ledger_path) {
      throw 'preflight ledger path is inconsistent'
  }
  Assert-H4H5CandidateState -Stage 'before ledger activation' -RequireNoActiveMarker
  if (Test-Path -LiteralPath $ledger) {
      throw "revision-specific H4/H5 ledger exists and must not be overwritten: $ledger"
  }
  $startOutput = @(& "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" start --cwd . --ledger $ledger --mode closure)
  if ($LASTEXITCODE -ne 0) { throw "verification start failed: $($startOutput -join [Environment]::NewLine)" }
  if ($startOutput.Count -ne 1 -or ([string]$startOutput[0]).Trim() -ne $ledger) {
      throw 'verification start output does not identify the exact revision-specific ledger'
  }
  if (-not (Test-Path -LiteralPath $ledger -PathType Leaf) -or
      -not (Test-Path -LiteralPath '.verification/active.json' -PathType Leaf)) {
      throw 'verification start did not create both ledger and activation marker'
  }
  ```

  Populate one claim per check, keeping H4 experiment claims and H5 code/mathematics/evidence claims distinct. H4 claims cover indexed traversal/parity/no-inter-repetition-work identity; warmup exclusion; exact primary per-seed `H4_PRIMARY_TIMED_BALANCE`, ten/ten pattern counts, and `110/110` aggregate timed balance; independent solver reachability; scaled condition-envelope eligibility; terminal equivalence; per-invariant solver budgets and strict allowance decisiveness; canonical immutable selected moments; raw repetition completeness; seed-level statistics; bootstrap interval; threshold decision; thread set/verify/restore environment identity; and secondary count/memory nonclaim. H5 claims cover typed no-fabrication preflight behavior, exact phase/error validity, unchanged inconclusive nonclaims, and `INCONCLUSIVE` precedence over simultaneous decisive failure; update-spec fixture full raw SHA-256/parser/schema/provenance; proof that no short fixture-digest prefix is accepted or exposed in the parser/config/gate/artifact path; immutable model-snapshot ownership; taxonomy; exact positive/control request IDs and base-rule map; graph completeness; complete before/after factor-input hashes; exact expected/observed affected equality; diagnostic-only value changes; deterministic candidate reconstruction with same-domain provenance/semantic hash comparisons; each positive case; all seven adversarial controls; every term's deterministic quadrature convergence plus rounding; independently derived production/oracle before/after/delta evidence, self-sufficient oracle term traces, and comparison; complete before/after total allowances; exact zero-stochastic epsilon formula; emission-touching decisiveness; full factor/term evaluation; snapshot separation; fixed recognition; label-specific acceptance; rollback hashes; and cache/reuse proof. Add separate required-tracked-scope, exact-case manuscript source, artifact/JUnit, dirty-content-digest, and prior-ledger-preservation claims.

  Give every assessed claim at least two unique views and exactly one adjudicator. Escalate to four/eight for recorded triggers; high/critical closure also links a skeptic. H4 experiment/code claims close only with current mechanical or reproduced-output evidence. Mathematical correctness claims close only with a derivation/formal proof, never timing or numerical agreement. Source/general claims require current primary/reproduced source evidence. Missing evidence or unresolved disagreement is `INCONCLUSIVE`, not a vote.

  Populate the exact generated ledger with `apply_patch` if the installed tool has no claim-add operation, then validate:

  ```powershell
  $validateOutput = @(& "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" validate $ledger --cwd .)
  if ($LASTEXITCODE -ne 0) { throw "verification validate failed: $($validateOutput -join [Environment]::NewLine)" }
  $validatedLedgerSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ledger).Hash.ToLowerInvariant()
  ```

  Expected: validation exits zero; every claim matches the current artifact revision; H4/H5 outcomes remain separate; the ledger records the unchanged preflight dirty-content digest; prior ledgers retain their Step 1 hashes. Do not manually remove the active marker or reuse the ledger path for another revision.

  If a source-changing defect is discovered after `start`, stop implementation immediately. Populate every affected/current claim as `INCONCLUSIVE` with the exact repair obligation, validate it, and use Step 6 with a message that names this revision-specific ledger so the installed verification hook—not a manual delete—retires `.verification/active.json`; preserve the ledger. Only after that retirement may the owning task be repaired and committed. The repaired revision gets a new full 40-character path, one replacement joint JUnit/click-run, fresh preflight JSON, and a new ledger; it never reuses evidence or the old ledger.

- [ ] **Step 6: Retire the installed verification hook cleanly before any H6 continuation.** Invoke the installed tool's hook with a final-message surrogate that names the validated ledger, parse its JSON response, require the exact successful empty object, and require tool-driven marker removal. Never delete or edit `.verification/active.json` manually.

  ```powershell
  $hookPayload = [ordered]@{
      cwd = (Get-Location).Path
      stop_hook_active = $true
      last_assistant_message = "Validated ledger: $ledger"
  } | ConvertTo-Json -Compress
  $hookOutput = @($hookPayload | & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" hook)
  if ($LASTEXITCODE -ne 0) { throw "verification hook process failed: $($hookOutput -join [Environment]::NewLine)" }
  $hookText = ($hookOutput -join [Environment]::NewLine).Trim()
  if ($hookText.Length -eq 0) { throw 'verification hook emitted no JSON response' }
  try {
      $hookResult = $hookText | ConvertFrom-Json -ErrorAction Stop
  } catch {
      throw "verification hook emitted invalid JSON: $hookText"
  }
  if ($null -eq $hookResult) { throw 'verification hook JSON decoded to null' }
  $hookProperties = @($hookResult.PSObject.Properties)
  if ($hookProperties.Count -ne 0) {
      if ([string]$hookResult.decision -eq 'block' -and ([string]$hookResult.reason).Length -ne 0) {
          throw "verification hook blocked retirement: $($hookResult.reason)"
      }
      throw "verification hook emitted an inconsistent success response: $hookText"
  }
  if (Test-Path -LiteralPath '.verification/active.json') {
      throw 'installed verification hook did not retire the activation marker'
  }
  ```

  Expected: hook process exit zero, exactly one parseable `{}` response, and no active marker. This is the only permitted retirement path before starting an H6 verification workflow.

- [ ] **Step 7: Perform the final read-only cross-check and report exact evidence surfaces.**

  ```powershell
  $preflight = Get-Content -Raw -LiteralPath C:\tmp\vfe4-h4-h5-preflight.json | ConvertFrom-Json
  Assert-H4H5CandidateState -Stage 'at closure' -RequireNoActiveMarker
  $requiredTracked = @($preflight.required_tracked)
  $stillTracked = @(git ls-files -- $requiredTracked)
  if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed during closure tracked-scope check' }
  $missingTracked = @($requiredTracked | Where-Object { $_ -notin $stillTracked })
  if ($missingTracked.Count -ne 0) { throw "required tracked file missing at closure: $($missingTracked -join ', ')" }

  $currentJunitSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath C:\tmp\vfe4-h4-h5-milestone.xml).Hash.ToLowerInvariant()
  if ($currentJunitSha256 -ne $junitSha256) { throw 'JUnit XML changed after parsing' }
  $currentManifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
  if ($currentManifestSha256 -ne $manifestSha256) { throw 'artifact manifest changed after inspection' }
  $closureManifestLines = @(Get-Content -LiteralPath $manifestPath -Encoding UTF8 -ErrorAction Stop)
  if ($closureManifestLines.Count -ne $expectedManifestEntries.Count) {
      throw 'artifact manifest entry count changed after inspection'
  }
  for ($index = 0; $index -lt $expectedManifestEntries.Count; $index++) {
      $line = [string]$closureManifestLines[$index]
      if ($line -notmatch '^([0-9a-f]{64})  (.+)$') { throw "invalid closure manifest line: $line" }
      $declaredHash = $Matches[1]
      $relativePath = $Matches[2]
      if ($relativePath -ne $expectedManifestEntries[$index]) {
          throw "closure manifest order changed: $relativePath"
      }
      $payloadPath = Join-Path $artifactPath ($relativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar))
      if (-not (Test-Path -LiteralPath $payloadPath -PathType Leaf)) {
          throw "closure manifest payload missing: $relativePath"
      }
      $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $payloadPath).Hash.ToLowerInvariant()
      if ($actualHash -ne $declaredHash) { throw "closure payload hash mismatch: $relativePath" }
  }
  $closureArtifactFiles = @(
      Get-ChildItem -LiteralPath $artifactPath -File -Recurse -Force |
          ForEach-Object { $_.FullName.Substring($artifactPrefix.Length).Replace('\', '/') } |
          Sort-Object
  )
  if ($closureArtifactFiles.Count -ne $expectedArtifactFiles.Count -or
      (($closureArtifactFiles -join "`n") -cne ($expectedArtifactFiles -join "`n"))) {
      throw 'artifact file set changed after inspection'
  }

  $currentValidatedLedgerSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ledger).Hash.ToLowerInvariant()
  if ($currentValidatedLedgerSha256 -ne $validatedLedgerSha256) {
      throw 'validated H4/H5 ledger changed after hook retirement'
  }

  foreach ($prior in @($preflight.prior_ledgers)) {
      if (-not (Test-Path -LiteralPath $prior.path)) { throw "prior ledger missing at closure: $($prior.path)" }
      $actualPriorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $prior.path).Hash.ToLowerInvariant()
      if ($actualPriorHash -ne $prior.sha256) { throw "prior ledger changed at closure: $($prior.path)" }
  }

  $priorLedgerPaths = @($preflight.prior_ledgers | ForEach-Object { [System.IO.Path]::GetFullPath([string]$_.path) })
  $currentLedgerItems = @(
      Get-ChildItem -LiteralPath '.verification' -File -Filter '*ledger.json' -ErrorAction SilentlyContinue |
          Sort-Object FullName
  )
  $currentLedgerPaths = @($currentLedgerItems | ForEach-Object { [System.IO.Path]::GetFullPath($_.FullName) })
  $removedLedgerPaths = @($priorLedgerPaths | Where-Object { $_ -notin $currentLedgerPaths })
  $newLedgerPaths = @($currentLedgerPaths | Where-Object { $_ -notin $priorLedgerPaths })
  $expectedLedgerPath = [System.IO.Path]::GetFullPath([string]$preflight.expected_ledger_path)
  if ($removedLedgerPaths.Count -ne 0) { throw "prior ledger set shrank: $($removedLedgerPaths -join ', ')" }
  if ($newLedgerPaths.Count -ne 1 -or $newLedgerPaths[0] -ne $expectedLedgerPath) {
      throw "closure requires exactly one new ledger, the revision-specific H4/H5 ledger: $($newLedgerPaths -join ', ')"
  }
  $closureLedgerHashes = @($currentLedgerItems | Get-FileHash -Algorithm SHA256)
  ```

  Expected: `HEAD` matches the revision in preflight JSON, JUnit, artifact provenance, and ledger; every required file remains tracked; tracked source is unchanged; no nonignored untracked content exists outside `.verification`; preflight/artifact/ledger/closure dirty-content digests agree; JUnit and manifest hashes remain unchanged; prior ledger hashes match Step 1; and set difference proves the only new ledger is `.verification/h4-h5-<FULL_HEAD>-ledger.json`. Report JUnit totals from XML, artifact path, manifest hash, H4 status/interval, H5 status, dirty-content digest, and validated ledger path. Do not add a post-evidence documentation commit or rerun the suite/timing.

## Out of Scope for This Plan

- H6 prefix-measurability, language modeling, held-out prediction, matched predictive controls, or dataset/tokenizer work.
- H7 internal population-frame covariance, decoder contragredience, Jacobian covariance, or gauge claims.
- H8 `T=128,K=20` sparse-scale/allocation promotion or a claim that H4's `D=256` diagnostic is H8 evidence.
- Any training launcher, checkpoint, optimizer schedule for training, WikiText-103 run, perplexity result, or predictive experiment.
- GPU, mixed precision, float32/bfloat16, multi-thread benchmark claims, cross-machine generalization, or energy-use claims.
- A general theorem that information coordinates are faster, use less memory, or converge better. H4 is bounded to the frozen hardware/protocol and exact primary endpoint.
- Any positive valid-MM implementation in H5 v1; a future revision requires a complete revision-bound proof artifact and a newly verified configuration contract.
- Stochastic H5 objectives, Monte Carlo error budgets, unrolled/implicit inference, learned frames, or H7-sensitive group updates.
- Treating natural-gradient, SGD, Adam, or truncated iterations as exact coordinate ascent.
- Research-vault ingestion. The vault was consulted read-only; any new result is offered for separate user-confirmed ingest only after the coupled milestone exists.

## Self-Review of Plan Completeness

- **Task 2 consistency:** every production call uses the materialized API;
  neutral bytes are oracle-only; materialization remains raw-only and outside
  timing; integrity hashes bracket mutation windows; repaired converter
  Choleskys are facade-visible; replay/count/memory remain distinct untimed
  passes.
- **Configuration and milestone closure:** the standalone H4 section owns all
  six `H4SolveProtocol` fields and validates their cross-field identities. It
  adds no runner prefix; Task 9 integrates it only with H5 under the existing
  coupled H1-H5 milestone and rejects H4 alone.
- **Oracle closure:** canonical and predictive `logZ`, exact selected blocks,
  typed route operand/allowance/strict-decision evidence, and signed
  `KL(q || p*)` to zero are explicit. No solver pair can corroborate a shared
  error.
- **Allowance closure:** every scalar path has its own operands, flags,
  operation table, rounding, solver term, comparison term, residual, pass, and
  strict ratio. Bounded vector chunks check 79,832,024 lanes without
  per-scalar dataclasses; counts, digest, maxima, and deterministic witnesses
  cannot mismatch elements.
- **Execution closure:** exact 3/11 cardinalities, preallocated timing slots,
  a mechanical guard, GC capture/disable/restore errors, typed incomplete
  phases, dedicated four-checkpoint materialized-integrity failure carriers,
  and the exact 146,720-event postflight schedule are all closed.
- **Condition closure:** posterior and innovation rules and complete counts are
  distinct and exact; record/key streams retain digests and worst witnesses
  without unbounded tuples.
- **Classifier closure:** only Task 1 contains threshold inequalities.
- **Schema closure:** gate, artifact, replay, operation, memory, environment,
  restoration, power policy, and coverage data are exact immutable records;
  selected coordinate indices survive compaction and bind their hashes; full
  current-problem objects are compacted and released before traversal advances,
  and canonical payload size is capped at 64 MiB.
- **Placeholder scan:** no `TBD`, deferred field, reduced repetition, unnamed
  callable, free-form allowance mapping, or deferred Task 9 H4 definition
  remains; Task 9 performs integration only.

- **H5 spec coverage:** H5 specifies the exact conditional categorical law and source-independent continuous reconstruction, a raw-byte-frozen conditional update fixture, full raw/canonical/schema hashes, closed identifier universes and dependency graph, immutable model/recognition/reference/live/candidate ownership, exact E/source/M formulas, complete objective/cache records, the completed/failed attempt union, five positives, seven controls, fieldwise exact-candidate allowances, order-21/order-17 term budgets, exact zero-stochastic delta allowance, byte-bearing gate evaluation, and configuration-only rejection of unsupported MM.
- **H5 interface consistency:** `CompleteElboEvaluation` is produced only by the complete evaluator. Completed outcomes embed complete before/after evaluations; failed outcomes carry only phase-valid complete or partial evidence through `H5AttemptOutcome`. `observed_affected_factor_ids` comes only from ordered factor-input hashes, and `execute_update` alone accepts or rolls back. H5's parser binds both raw fixture byte objects to `H5ReferenceState`, while the runner passes those same captured bytes into byte-bearing production and oracle seams and publishes a separate H5 payload.
- **Evidence discipline:** Focused RED/GREEN commands are noncumulative. The milestone preflight requires every plan, preregistration, source, config, launcher, and test file to be tracked; rejects nonignored untracked content outside `.verification`; records the exact dirty-content digest and prior-ledger hashes; and rechecks them through closure. The only full suite and full timing occur at one shared exact revision. Reviewers inspect rather than rerun. A defect found after ledger activation closes the current revision `INCONCLUSIVE`, preserves it, repairs only after tool-driven retirement, and permits exactly one replacement revision/run/ledger. The coupled ledger separates claims by gate and preserves every earlier ledger.
- **H5 placeholder scan:** The plan contains no unspecified H5 fixture producer/parser, raw-byte digest comparison, source-row identity, model-snapshot owner, short-digest rejection rule, or required-tracked surface. The exact H5 fixture SHA-256 is intentionally not invented: Task 5 authors the tracked bytes and pins `SHA256(raw_fixture_bytes)` in parser/tests before GREEN/commit; Task 9 copies and verifies that literal. H6--H8/training and MM activation remain explicit nonclaims, not hidden implementation gaps.
- **American English:** Terminology uses American English throughout.

Plan implementation must stop after Task 10 with the exact H4 result, exact H5 result, JUnit totals, atomic artifact path, and validated `.verification/h4-h5-<FULL_HEAD>-ledger.json` reported from current evidence.
