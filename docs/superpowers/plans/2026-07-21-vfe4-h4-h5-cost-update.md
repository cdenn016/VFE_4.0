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

@dataclass(frozen=True)
class H4GateResult:
    gate: Literal["H4"]
    status: GateStatus
    measurements: Mapping[H4MeasurementName, float | None]
    invariants: tuple[InvariantResult, ...]
    allowances_by_invariant: Mapping[H4AllowanceInvariantName, H4JsonMapping]
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
def decide_h4_interval(interval: H4BootstrapInterval, threshold: float) -> H4IntervalDecision: ...
```

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

The H5 update-spec parser computes `SHA256(raw_bytes)` before UTF-8 decoding and compares all 64 hexadecimal characters with its pinned literal. It then rejects duplicate keys, unknown/missing fields, nonfinite constants, wrong sequence types/order, aliases, and schema drift. `UpdateSpecification` accepts the already checked immutable raw bytes but no digest/canonical metadata; its constructor recomputes `raw_sha256`, encodes only the decoded semantic fields (never `raw_bytes` or derived hashes) into `canonical_bytes`, and derives the domain-separated canonical hash. `H5ReferenceState` requires byte equality between its update-spec bytes and `specification.raw_bytes`, verifies the exact H1/update IDs, and likewise recomputes all four displayed hashes from its bytes/specification/schema constants instead of accepting them from callers.

The reference-state hash core is exactly `(h1_fixture_sha256, update_spec_raw_sha256, specification.canonical_sha256, objective_schema_sha256, factor_input_schema_sha256, initial_recognition.state_sha256, initial_model.state_sha256, initial_optimizer_state.state_sha256, initial_rng_state.state_sha256)` in that order under the reference-state domain. `initial_live(reference)` returns a new complete `H5LiveState` containing exactly those four initial state objects after defensive reconstruction; its resulting nested hashes equal the four recorded reference hashes.

The request-independent semantic-state digest is `SHA256(semantic-state-domain || uint64_be(len(canonical_recognition_bytes)) || canonical_recognition_bytes || uint64_be(len(canonical_model_bytes)) || canonical_model_bytes)`. Production and the independent oracle both retain it as provenance, but independent NumPy/PyTorch arithmetic is not required to produce bit-identical digests. Candidate correctness uses the frozen fieldwise allowance below. This digest deliberately excludes rule/request/label/active-block/damping provenance, which remains protected only by `candidate_sha256` and transaction validation.

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

  H4 thread control is process-scoped and mandatory. After H1--H3 work and before H4 preflight/timing, capture `torch.get_num_threads()`, call `torch.set_num_threads(1)`, and verify the observed intra-op count is one. In a `finally` block attempt to restore the captured count and record prior, effective, restored, and restoration-error fields. A set/verify failure suppresses timed records and makes H4 `INCONCLUSIVE`; a restoration failure is an environment/protocol obligation that prevents H4 `PASS`. Do not change inter-op threads.

  Resolve H4 status in this fixed precedence: protocol/environment/thread/fixture/condition/table-completeness/nonfinite ambiguity is `INCONCLUSIVE`; otherwise a finite decisive H3-anchor or terminal-law miss is `FAIL`; otherwise apply the primary interval rule (`PASS` only when upper bound `<=0.80`, `FAIL` only when lower bound `>=0.80`, and `[0.80,0.80]` or a crossing interval `INCONCLUSIVE`). Operation and memory diagnostics are secondary and never rescue or overturn that status.

- [ ] **Step 2: Write strict type/generator tests.** Assert exact ordered fields and validations for every defined H4 record, including explicit `source_kind` rather than ID inference; scaled positive PCG64 seeds and H3-anchor `seed=0` exactly in IDs/core/digests; `H4RawDraw` global zero-based indices, row-major values, shape product, finite values, factor-local ordering/names, and H3 empty raw draws; the factor residual/no-normalizer, exact `A:d x D`, `b:d`, SPD `R:d x d`, metadata order/disjointness/support/identity-column contracts; immutable tuple/mapping ownership; schedule availability rather than ID/numeric-role inference; and `factor_schedule` as the sole canonical source. Assert exact scaled coordinate strings, factor IDs, causal metadata tuple orders, and `D` table `(64,128,256)`, H3's unchanged coordinate spelling/IDs, fixed no-RNG initial `N(0,I_8)`, the exact PCG64 names/indices/draw order/distributions, separate `A_m` and joint `[A_z B]` spectral-clip envelopes, exact `H`, and scaled observation `b=observed_target-offset` with raw provenance retained. Require that only scaled matched controls have exactly the listed exceptions and that every designated transition-parent column is zeroed; do not apply this invariant to H3 anchors. Require deterministic core digest and published envelope bytes, exact hash domain/schema literal before digest verification, parser recomputation rather than self-reference/full-envelope hash, and distinct hashes across kind/seed/size. Require exact `J`, `h`, and `c` factor assembly including derived normalizers. Require exact `H4_INVARIANT_NAMES`, `H4_MEASUREMENT_NAMES`, `H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL`, and `H4_ALLOWANCE_INVARIANT_NAMES` ordering; test the sole pre-timing decisive H3-anchor failure with its five-and-only-five unavailable measurements, finite threshold/residual/allowance fraction, exact unevaluated-invariant records, and applicable/inapplicable allowance shapes; reject fabricated values, wrong phase/obligation evidence, malformed sentinels, or numeric inapplicable records. Require closed result aliases, finite conclusive measurements, `INCONCLUSIVE` `None` values only with phase obligations, and recursive owned immutable finite allowance records. Require each `H4TimingRecord` to carry independent `problem_index`, `horizon_index`, `seed_index`, `kind_index`, timed `repetition_index`, absolute `pair_index`, exact order label, and both positive native-arm durations; reject an order inconsistent with the independent-index parity formula. Require the exact immutable selected-moment labels `("initial","terminal","observation[1]",...,"observation[T]")`, with immutable mean/covariance rows; reject reordered, duplicate, missing, mapping, mutable, or aliased values. Adapt both raw H3 fixtures through the explicit structural-group adapter; require the coupled canonical `(J,h,c,logZ)` to agree with H3/reference allowances and the zero anchor to compare independently derived adapter/oracle `c/logZ` only, without asserting a nonexistent frozen reference.

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

- [ ] **Step 1: Write the promotion test with exact invariant names.** Require `H4_INVARIANT_NAMES` from the public type contract exactly and in order; do not duplicate this literal in Task 4. Require `H4_MEASUREMENT_NAMES` and `H4_ALLOWANCE_INVARIANT_NAMES` from that same contract as the exact ordered outer mapping key sets, and validate their closed aliases and recursive immutable finite allowance-record JSON.

  Include the early-fail branch: a sole decisive finite `h3_anchor_identity` failure is pre-timing, has exactly `H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL` as `None`, has finite `0.80` threshold/residual/allowance-fraction values as specified by the public contract, marks every later invariant with the exact unevaluated tuple/detail, and has one applicable numerical H3 allowance plus five exact inapplicable sentinels. Reject a timed/statistical fabrication, a different unavailable set, malformed/extra/missing sentinel content, numeric content in an inapplicable record, or an `INCONCLUSIVE` `None` without its producing-phase invariant/detail and named obligation.

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

`vfe4/types/updates.py` owns every displayed snapshot/request/value record and update enum plus canonical recognition/model/request/live/reference/candidate/semantic-state encoders and `initial_live`. `vfe4/types/h5_schema.py` owns every identifier/dependency/reconstruction literal tuple, sign, operation-count table/function, hash domain, factor-input/objective schema encoder, and resulting schema hash as plain immutable data, importing no update dataclass; lower-level types and higher-level objective/numerical modules therefore share one dependency-neutral source. `vfe4/objective/dependency_graph.py` owns `FactorDependencyGraph` plus dependency resolution. `vfe4/validation/h5_update_spec.py` owns the expected raw-digest literal, strict parser, reference builder, H1 reconstruction, and update-spec canonical encoder.

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

Production construction rejects any mismatch between rule, requested label, active blocks, and schedule. `request_sha256` is recomputed from every displayed request field under the update-request domain. A candidate copies the request's rule/hash/active blocks, records the one selected damping, and recomputes its own hash over those provenance fields, its diagnostics, and both snapshots; a candidate can therefore neither migrate between requests nor conceal its line-search step. Only the exact M candidate carries `(("G_condition_number", kappa_2(G)),)`; all other candidates require an empty diagnostics tuple. Test/gate-only fault injection operates before candidate validation so the mislabel control can observe the typed rejection. `valid_mm` never enters this mapping.

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

Working-state constructors require exact active keys, CPU float64 scalar/vector leaves, `requires_grad=True` only for declared active leaves, and no storage alias with either live snapshot. Variance leaves use `exp(log_variance)` at freeze; categorical leaves use masked `log_softmax`/`softmax` on the fixed support. `freeze_candidate` takes the unchanged complete `live` base, replaces exactly the active request blocks from detached working leaves, defensively copies every inactive recognition/model block from `live`, and asserts the frozen-complement hash before returning the complete candidate. Working states are never accepted, hashed as live state, or serialized into artifacts.

`UpdateHashRecord.request_sha256` must equal the constructor-recomputed request hash. Before/final live, recognition, model, optimizer, RNG, and frozen-complement hashes are mandatory for every transaction outcome, including early typed failure. Candidate hashes become mandatory at `FREEZE`; predecision hashes become mandatory after freeze and before after-evaluation/decision; phase constructors require the unavailable later hashes to be `None`. A completed outcome requires every hash field.

`CompletedUpdateAttempt` requires complete before/after universes, no missing/extra IDs, exact producer/request label agreement, exact expected/observed affected equality, and exact decision logic. `FailedUpdateAttempt` is the only representation for a missing factor, pre-evaluation label rejection, stale cache, invalid candidate, forced invalid decision, mutation, or deterministic scalar corruption; phase-specific constructors require producer/decision/affectedness/recheck fields exactly when that phase has observed them and require `None`/empty tuples before observation. No failure fabricates an after-`ElboTerms` value.

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
def freeze_candidate(live: H5LiveState, recognition_working: DifferentiableRecognitionState, model_working: DifferentiableModelState, *, request: UpdateRequest, producer_label: UpdateLabel, damping: float) -> H5CandidateSnapshot: ...
def canonical_frozen_complement_bytes(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest) -> bytes: ...
def execute_update(reference: H5ReferenceState, live: H5LiveState, request: UpdateRequest, evaluator: CompleteElboEvaluator, budget: H5BudgetConfig, *, fault_injection: H5FaultInjection | None = None) -> H5TransactionResult: ...
```

Decision rules are exact: an exact coordinate is eligible iff `delta_elbo >= -epsilon_delta`; generalized-EM and other proposal labels are eligible iff `delta_elbo > epsilon_delta`; `delta_elbo < -epsilon_delta` is a resolved rejection; the closed boundary is rejected as unresolved. Any required emission-touching positive case on that boundary makes the gate inconclusive, not failed. A rejected resolved-decrease natural-gradient positive case passes only with identical final live/recognition/model/optimizer/RNG hashes. `execute_update` returns a new immutable live state on acceptance and the original object on rejection or failure; no caller-visible in-place mutation occurs.

The independent oracle accepts bytes, never production types. It parses its two candidate JSON byte fields into the same closed semantic snapshot schemas and independently computes `semantic_state_sha256`; it neither constructs nor claims equality of the provenance-bearing production `candidate_sha256`. Only `oracle_exact_m_block` records `(("G_condition_number", kappa_2(G)),)`; every other oracle result requires an empty condition-number tuple:

```python
@dataclass(frozen=True)
class H5OracleUpdate:
    schema_version: Literal["h5-oracle-update-v1"]
    rule: str
    candidate_recognition_json: bytes
    candidate_model_json: bytes
    candidate_condition_numbers: tuple[tuple[str, float], ...]
    semantic_state_sha256: str = field(init=False)
    before_elbo: float
    after_elbo: float
    delta_elbo: float

def oracle_exact_e_block(h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes) -> H5OracleUpdate: ...
def oracle_exact_source_row(h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes) -> H5OracleUpdate: ...
def oracle_exact_m_block(h1_fixture_bytes: bytes, update_spec_bytes: bytes, live_state_bytes: bytes) -> H5OracleUpdate: ...
def oracle_complete_delta(h1_fixture_bytes: bytes, update_spec_bytes: bytes, before_state_bytes: bytes, after_state_bytes: bytes, *, rule: str) -> H5OracleUpdate: ...
```

- [ ] **Step 1: Write failing tests for the five positives, every transaction phase, and independent oracle.** Assert exact z0 blanket contributions, the row formula, full five-parameter M solve, fixed detached recognition, fieldwise production/oracle exact-candidate agreement under the frozen operand-shaped allowances while retaining nonbinding semantic hashes, detached equality between differentiable and Task 6 order-21 complete objectives, first resolved GEM damping, oversized natural-gradient rejection, freeze-before-evaluate, no H2 mutation/autograd, input-hash affectedness, valid unaffected reuse, and byte-identical rollback.

- [ ] **Step 2: Run Task 7 RED.**

```powershell
python -m pytest tests/unit/test_h5_updates.py tests/oracle/test_h5_update_oracle.py -q
```

Expected: collection fails because H5 updates and oracle do not exist.

- [ ] **Step 3: Implement the three exact coordinates without autograd or generic optimizers.** Use solve/Cholesky operations directly and freeze candidates immediately.

- [ ] **Step 4: Implement differentiable GEM/natural proposals with exact active-leaf scopes and the frozen damping order.** `torch.autograd.grad` may receive only declared active leaves. Candidate freeze must precede evaluator entry.

- [ ] **Step 5: Implement transactional `execute_update`.** Capture before hashes, evaluate before, derive expected dependencies, propose/freeze, prove live state unchanged predecision, evaluate after, derive hash-affected IDs, validate complete/reused sets, compute the exact delta allowance, apply label policy, then atomically replace whole snapshots or retain the original live state. Convert every typed phase failure into `FailedUpdateAttempt`.

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
    oracle_delta: float
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
    h1_fixture_raw_sha256: str
    update_spec_raw_sha256: str
    update_spec_canonical_sha256: str
    objective_schema_sha256: str
    factor_input_schema_version: Literal["h5-factor-input-v1"]
    factor_input_schema_sha256: str
    positive_cases: tuple[H5PositiveCaseResult, ...]
    controls: tuple[H5ControlResult, ...]
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]

@dataclass(frozen=True)
class H5ValidationPayloadRecord:
    schema_version: Literal[1]
    result: H5GateResult
    reference_sha256: str
    factor_universe: tuple[str, ...]
    recognition_coordinate_universe: tuple[str, ...]
    model_block_universe: tuple[str, ...]
    variable_dependency_rows: tuple[tuple[str, tuple[str, ...]], ...]
    parameter_dependency_rows: tuple[tuple[str, tuple[str, ...]], ...]
    positive_attempts: tuple[H5AttemptOutcome, ...]
    controls: tuple[H5ControlResult, ...]
    oracle_results: tuple[H5OracleUpdate, ...]
    nonclaims: tuple[str, ...]
    canonical_bytes: bytes = field(init=False, repr=False)
    payload_sha256: str = field(init=False)

@dataclass(frozen=True)
class H5GateEvaluation:
    schema_version: Literal["h5-gate-evaluation-v1"]
    result: H5GateResult
    reference: H5ReferenceState
    positive_attempts: tuple[H5AttemptOutcome, ...]
    controls: tuple[H5ControlResult, ...]
    oracle_results: tuple[H5OracleUpdate, ...]
    validation_payload: H5ValidationPayloadRecord

def compare_h5_exact_candidate(production: H5CandidateSnapshot, oracle: H5OracleUpdate) -> H5CandidateComparison: ...

def evaluate_h5(
    config: ResolvedConfig,
    *,
    h1_fixture_bytes: bytes,
    h5_update_spec_bytes: bytes,
) -> H5GateEvaluation: ...

def h5_validation_payload(evaluation: H5GateEvaluation) -> dict[str, object]: ...
```

The caller must pass the same captured immutable H5 byte object to production and oracle adapters. In Task 8, `evaluate_h5` reads only the already-existing resolved common CPU/float64/determinism fields; all H5 protocol identity comes from Task 5 constants and the captured bytes. It never rereads either fixture path. Task 9 later adds and validates the H5 config section without changing this byte-bearing signature.

`H5ValidationPayloadRecord` requires exact universe/dependency order, raw attempt/control/oracle equality with the result/evaluation, and `nonclaims == H5_NONCLAIM_IDS`. It canonicalizes every nested frozen record under the validation-payload domain and computes its own hash; `h5_validation_payload` is only the deterministic JSON-primitive projection of that closed record, not an independently assembled mapping.

`compare_h5_exact_candidate` compares exactly the active fields listed in `H5_CANDIDATE_COMPARISON_OPERATION_COUNTS`, in that order, using the frozen scalar formula and conditions above. Exact z0/source comparisons record condition `1.0`; the M comparison records each implementation's `G` condition number supplied in its update diagnostics. Exact positive cases require non-`None` passing comparisons. GEM/natural positive cases require `candidate_comparison is None` and independent complete-delta agreement. Production/oracle semantic hashes are always retained but their equality or inequality never determines PASS/FAIL/INCONCLUSIVE.

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
4. Use the valid `NATURAL_GRADIENT_Z1` request and its normal frozen candidate, preserving its rule, request hash, `q[z1]` active block, snapshots, and damping; mutate only `candidate.producer_label` from `NATURAL_GRADIENT_PROPOSAL` to `EXACT_COORDINATE` and recompute the candidate hash. Producer/request-label validation alone yields `FailedUpdateAttempt(FREEZE, LABEL_PROVENANCE_MISMATCH)` before evaluation or acceptance.
5. After the resolved-decrease natural candidate is rejected, the rollback seam returns a copied live state with only `q[z1].mean=math.nextafter(old, math.inf)` and RNG payload counter changed from zero to one. Final live/recognition/RNG hashes yield `FailedUpdateAttempt(COMMIT_OR_ROLLBACK, ROLLBACK_HASH_MISMATCH)`; the original caller-owned live object remains unchanged.
6. For `state_transition[2]`, reflect `alpha_0` about its fixed-complement scalar least-squares optimum
   `alpha_hat = E[z0*(z2-(B_base+s)*m2-c)] / E[z0**2]`, using `alpha_0' = 2*alpha_hat-alpha_0`. Evaluate both sides through the same completed-square quadratic form; their order-21/order-17 `float.hex()` pairs must be exactly equal while canonical input bytes differ. The factor must appear in `observed_affected_factor_ids` and must be absent from `value_changed_factor_ids`; failure to realize this exact fixture property is `INCONCLUSIVE`, not a relaxed control.
7. With the `state_transition[2]` canonical input bytes unchanged, the factor-record seam adds exactly `1.0e-6` to its reported order-21 and order-17 values. Before any internally inconsistent `CompleteElboEvaluation` can be constructed, independent deterministic reevaluation of that same input produces the unmodified pair. The resulting `FailedUpdateAttempt(AFTER_EVALUATION, DETERMINISTIC_REEVALUATION_MISMATCH)` retains the corrupted record in `partial_after`, keeps the factor absent from `observed_affected_factor_ids`, includes it in diagnostic `value_changed_factor_ids`, and stores the unequal pair in `deterministic_reevaluation`.

Status construction is fail-closed:

- `PASS` iff the raw fixture/schema/graph are exact, all five positives pass, all seven controls detect their intended fault, every numerical record is finite/complete/operand-shaped, and obligations are empty.
- `FAIL` iff current finite complete evidence decisively falsifies a required positive, dependency, decision, rollback, oracle, or control invariant. A control is successful when it detects its injected fault; the injected fault itself does not make H5 fail.
- `INCONCLUSIVE` iff required evidence is missing/nonfinite, a schema/cache cause is unresolved, or a required emission-touching comparison remains inside/on its complete allowance; obligations are nonempty and name each open phase.
- Unsupported MM is absent from this status mapping. Configuration resolution rejects it before `evaluate_h5`; a normal H5 run has no MM obligation.

- [ ] **Step 1: Write the failing promotion test for exact five-positive/seven-control order, every typed outcome, payload schema, byte identity, and all PASS/FAIL/INCONCLUSIVE contradictions.** Include exact candidate-comparison field/count/order/formulas, both within/outside allowance, permitted unequal semantic hashes, exact boundaries `delta=-epsilon`, `delta=epsilon`, and just outside them; malformed partial/full outcomes; changed-input/equal-value and same-input/changed-value separation; rollback optimizer/RNG hashes; no MM obligation; and a proof that path rereads are not performed.

```python
def test_h5_gate_consumes_captured_bytes_and_requires_all_cases_and_controls():
    evaluation = evaluate_h5(CONFIG, h1_fixture_bytes=H1_BYTES, h5_update_spec_bytes=H5_BYTES)
    assert tuple(case.case_id for case in evaluation.result.positive_cases) == tuple(H5PositiveCaseId)
    assert tuple(control.control_id for control in evaluation.result.controls) == tuple(H5ControlId)
    assert evaluation.result.status is GateStatus.PASS
    assert evaluation.result.obligations == ()
    assert "valid_mm" not in " ".join(evaluation.result.obligations)
```

- [ ] **Step 2: Run Task 8 RED.**

```powershell
python -m pytest tests/promotion/test_h5_gate.py -q
```

Expected: collection fails because the H5 gate does not exist.

- [ ] **Step 3: Implement test/gate-only fault seams and the seven controls without changing production globals.** Each seam is injected into one evaluator/controller instance and restored by object disposal, not monkeypatch leakage.

- [ ] **Step 4: Implement the five positive cases, gate invariant/status constructor, byte-bearing `evaluate_h5`, and complete payload.** Payload includes raw/canonical/schema hashes; universes/graph; every complete or partial outcome; both-order factor/term values; allowances; dependency/value diagnostics; producer/request labels; line search; all snapshot/live/optimizer/RNG hashes; oracle comparisons; controls; invariants; status/obligations; and H6–H8/training nonclaims. It contains no MM configuration field or obligation; Task 9 owns that later config-only resolution surface.

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
- Add `h4: H4ValidationConfig | None` and `h5: H5ValidationConfig | None`. Both are absent for shorter prefixes and both are required for the H5 prefix. They remain separately hashed sections and produce separate results. `H5ValidationConfig` copies Task 5's exact fixture ID, raw SHA-256, canonical SHA-256, objective-schema SHA-256, factor-input schema version/SHA-256, all three ordered identifier universes, and ordered rule/positive/control IDs. Resolution recomputes `SHA256(raw_fixture_bytes)` after the coupled-prefix capture and rejects any mismatch, truncation, short digest prefix, canonical/schema drift, or ordering change.
- H5 v1 config contains `enabled_update_rules=tuple(H5UpdateRule)`, `enabled_update_labels=(UpdateLabel.EXACT_COORDINATE, UpdateLabel.GENERALIZED_EM, UpdateLabel.NATURAL_GRADIENT_PROPOSAL)`, and `mm_proof_artifact=None`. Resolution requires those labels to equal the ordered unique producer labels induced by the enabled rule contracts. Adding `UpdateLabel.VALID_MM` is the concrete unsupported-MM request and is rejected before constructing an `UpdateRequest`, before reading update state, and before calling `evaluate_h5`. Absence of an MM artifact or request never creates an H5 attempt, invariant, obligation, or gate status.
- `H4ValidationConfig` exposes the exact parity expression, warmup/timed pair-index tuples, `warmups_count_toward_balance=False`, the canonical `H4_PRIMARY_TIMED_BALANCE` 20-row tuple `(seed, AB, BA)`, `primary_timed_ab_total=110`, and `primary_timed_ba_total=110`; resolution recomputes these values from independent horizon/seed/kind/pair indices and rejects any disagreement.
- Extend the explicit result union to include `H4GateResult` and `H5GateResult`; do not merge their measurements or status.

- [ ] **Step 1: Write focused configuration, integration, and artifact tests.** Assert the one editable `CONFIG` resolves to ordered H1--H5 and includes exact H4 horizon/seed/kind traversal with independent zero-based indices, AB exactly when `(horizon_index + seed_index + kind_index + pair_index) % 2 == 0`, warmup/timed pair indices, `warmups_count_toward_balance=False`, the exact 20-row `H4_PRIMARY_TIMED_BALANCE`, exactly ten primary `6/5` rows and ten `5/6` rows, exact aggregate timed totals `AB=110` and `BA=110`, seeds/dimensions/protocol/statistics/environment constraints, inclusive condition envelope, `1e-9` solver budget, and strict `1e-4` decisiveness cap; plus exact H5 conditional update-spec fixture ID, full raw/canonical digests, objective/factor-input schema fields, identifier universes, five rule contracts, producer-label order, five positive cases, seven controls, quadrature orders `21/17`, deterministic-convergence-plus-rounding budgets, zero stochastic contribution, and epsilon formula. Reject a formula based on flattened `problem_index`, either swapped per-seed count, a per-seed imbalance masked by correct aggregate totals, aggregate totals other than `110/110`, or a true warmup-balance flag. Reject `VALID_MM` with `mm_proof_artifact=None` before request/state/evaluator construction, and assert that the supported config creates no missing-MM obligation. Test every envelope/cap/delta boundary. Resolve every shorter compatibility prefix and prove it contains no H4/H5 config, does not read/hash/capture H4/H5/update-spec inputs, does not run timing/updates, and publishes no H4/H5 payload/provenance keys.

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

- [ ] **Step 3: Add exact typed H4/H5 sections and fail-closed resolution.** Canonicalize every frozen literal. Derive arm order from independent indices, recompute the primary 20-row timed balance, pattern counts, and aggregate totals, and require exact equality with the configured literals before returning `ResolvedConfig`. Reject changed horizon/seed/kind traversal, seed order/count, size order, primary dimension, warmup/timed pair indices, parity formula, flattened-`problem_index` parity, warmup inclusion, any per-seed primary balance row, pattern-count or `110/110` aggregate mismatch, no-between-repetitions/postflight tag, bootstrap settings, threshold, condition-envelope bound/inclusivity, solver budget, decisiveness cap/strictness, timer boundary tag, solver labels, thread/dtype/device, an H5 fixture/raw/canonical/objective-schema/factor-input-schema field that differs from Task 5, any raw digest shorter than the exact 64-hex SHA-256 literal, any universe/rule/label/positive/control order change, quadrature-order drift, deterministic-budget drift, nonzero stochastic contribution, or delta-formula drift. Require exact equality between enabled rule producer labels and `enabled_update_labels`. Reject unsupported `VALID_MM` at configuration resolution when `mm_proof_artifact=None`; do not pass that condition into attempt or gate status logic. Reject H4/H5 section presence for shorter prefixes and absence of either section for the coupled prefix.

- [ ] **Step 4: Extend conditional one-time capture and ordered evaluation.** Capture `h1-v1` once for H1/H2/H5, H3 coupled/zero bytes once for H3/H4 only when consumed, and `h5_conditional_update_v1.json` bytes once only for the coupled H1/H2/H3/H4/H5 prefix. Immediately compute and compare its full raw-byte SHA-256 against the exact Task 5/config literal before parser decode; no short digest prefix is accepted or exposed in configuration, gate arguments, provenance, payloads, or artifacts. Pass the same captured H5 byte object by identity to `evaluate_h5` and every production/oracle adapter; H5 receives both `h1_fixture_bytes` and `h5_update_spec_bytes`. Evaluate H1, H2, H3, H4, H5 in order. H4 receives H3 bytes. Publish only after both expensive gates return. Shorter prefixes must neither read, hash, capture, nor publish the H5 update-spec. Aggregate status is `fail` if any gate fails, otherwise `inconclusive` if any is inconclusive, otherwise `pass`.

- [ ] **Step 5: Extend environment and provenance.** Preserve current source/config/dirty-content security fields and expose the canonical `dirty_content_digest` used by milestone preflight/rechecks. Add timing clock implementation/resolution/monotonicity, process CPU affinity, logical/physical CPU counts when available, processor/platform, PyTorch intra/inter-op threads, `torch.__config__.show()` digest/text, NumPy BLAS configuration digest/text, CUDA availability (expected false for H4), and exact values/presence of `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. Record H4 prior/effective/restored intra-op thread values and any restoration error, ordered gate states, distinct H4/H5 config hashes, H5 raw/canonical update-spec, objective-schema, factor-input-schema, recognition/model/reference/transaction/payload hashes, `fixture_hashes["h5-conditional-update-v1"]`, `gate_fixture_consumers["H5"]=("h1-v1","h5-conditional-update-v1")`, H5 universes/rule/control orders/quadrature/allowance rules, H4 traversal/problem-factor hashes/parity formula/warmup-exclusion flag/expected and observed primary per-seed plus `110/110` aggregate timed balance/envelope/budget/cap, and H4/H5 bounded-claim/nonclaim tags. Shorter prefixes contain none of the H5 update-spec fields.

- [ ] **Step 6: Extend the one launcher and bounded documentation.** Keep one `CONFIG`, `main`, and script guard. Print H1--H5 statuses separately and one artifact path. README and the H4 preregistration state the independent-index parity formula, literal primary 20-row timed balance, ten/ten pattern split, exact `110/110` totals, and that warmups are excluded from balance; they do not prestate H4 speed or H5 pass results. The H5 preregistration states the exact conditional recognition law, raw/canonical/schema bindings, five rules/positives, seven controls, and config-only unsupported-MM rejection without inventing a missing-MM gate obligation. Use exact live path case `Manuscripts/...` in every source citation and add a focused documentation assertion that rejects any differently cased variant. Explicitly defer H6--H8 and training.

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
       'verification/h4_budget.py', 'verification/h4_statistics.py',
       'verification/h4_gate.py', 'verification/h5_gate.py',
       'verification/run_gates.py',
      'vfe4/config/schema.py', 'vfe4/config/resolve.py',
      'vfe4/artifacts/provenance.py', 'verify_vfe4.py', 'README.md',
      'tests/unit/test_h4_problem.py', 'tests/unit/test_h4_solvers.py',
      'tests/unit/test_h4_instrumentation.py', 'tests/unit/test_h4_statistics.py',
      'tests/oracle/test_h4_numpy_oracle.py', 'tests/promotion/test_h4_gate.py',
        'tests/unit/test_h5_update_types.py', 'tests/unit/test_h5_objective_schema.py',
        'tests/unit/test_h5_dependency_graph.py', 'tests/unit/test_h5_update_spec.py',
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
  - H5 theory/dependency reviewer: exact-case `Manuscripts/...` Markov blankets, the conditional categorical recognition law, source-independent continuous reconstruction, same complete ELBO, dependency prediction versus input-hash-derived observed affected sets, exact/source/M/GEM semantics, factor-universe completeness, and proof that MM is absent from attempt/gate paths;
  - H5 implementation/transaction reviewer: captured update-spec raw bytes/digest/parser/schema, proof that no short fixture-digest prefix is accepted or exposed anywhere in the H5 parser/config/gate/artifact path, immutable recognition and model-snapshot ownership with declared shared storage only, differentiable-working versus immutable-snapshot boundary, fixed recognition M-block, order-21/order-17 convergence estimates for every term, complete total allowances and exact epsilon formula, emission-touching indecision, acceptance/rollback hashes, cache/reuse proofs, seven controls, and value-change diagnostic nonauthority;
  - artifact/compatibility reviewer: required tracked-file list, no unexpected untracked content, stable dirty-content digest, separate H4/H5 statuses/payloads, exact prefix behavior, H5 full raw-digest-only provenance with no accepted/exposed short prefix, atomic manifest, prior-ledger hashes, H6--H8/training nonclaims.

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

  Populate one claim per check, keeping H4 experiment claims and H5 code/mathematics/evidence claims distinct. H4 claims cover indexed traversal/parity/no-inter-repetition-work identity; warmup exclusion; exact primary per-seed `H4_PRIMARY_TIMED_BALANCE`, ten/ten pattern counts, and `110/110` aggregate timed balance; independent solver reachability; scaled condition-envelope eligibility; terminal equivalence; per-invariant solver budgets and strict allowance decisiveness; canonical immutable selected moments; raw repetition completeness; seed-level statistics; bootstrap interval; threshold decision; thread set/verify/restore environment identity; and secondary count/memory nonclaim. H5 claims cover update-spec fixture full raw SHA-256/parser/schema/provenance, proof that no short fixture-digest prefix is accepted or exposed in the parser/config/gate/artifact path, immutable model-snapshot ownership, taxonomy, graph completeness, complete before/after factor-input hashes, exact expected/observed affected equality, diagnostic-only value changes, each positive case, all seven adversarial controls, every term's deterministic quadrature convergence plus rounding, complete before/after total allowances, exact zero-stochastic epsilon formula, emission-touching decisiveness, full factor/term evaluation, snapshot separation, fixed recognition, label-specific acceptance, rollback hashes, and cache/reuse proof. Add separate required-tracked-scope, exact-case manuscript source, artifact/JUnit, dirty-content-digest, and prior-ledger-preservation claims.

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
- Any positive valid-MM implementation in H5 v1; a future revision requires a complete revision-bound proof artifact and a newly verified configuration contract.
- Stochastic H5 objectives, Monte Carlo error budgets, unrolled/implicit inference, learned frames, or H7-sensitive group updates.
- Treating natural-gradient, SGD, Adam, or truncated iterations as exact coordinate ascent.
- Research-vault ingestion. The vault was consulted read-only; any new result is offered for separate user-confirmed ingest only after the coupled milestone exists.

## Self-Review of Plan Completeness

- **Spec coverage:** H4 and H5 remain separate results/payloads/statuses. H4 freezes one authoritative generic normalized-factor schedule with validated derived partitions, the H3 structural-group anchor mapping, the exact normalized log-evidence sign/constants, exact scaled dimensions, fixed `N(0,I_8)` initial law, PCG64 draw order and distributions, `m_t`-then-`z_t|m_t` factorization, all-zero transition-block control, independent arms, three per-problem warmup pairs followed by 11 timed pairs, independent-index parity, warmup exclusion, the literal primary 20-row balance table, ten/ten per-seed pattern split, exact `110 AB/110 BA` timed aggregate, batch-post-timing conversions, seed-level bootstrap threshold, inclusive scaled conditioning envelope, per-invariant solver budget and strict allowance/scale cap, mean/covariance selected moments, raw times, secondary memory/count nonclaim, set/verify/finally-restore one-thread CPU float64, and provenance. H5 specifies the exact conditional categorical law and source-independent continuous reconstruction, a raw-byte-frozen conditional update fixture, full raw/canonical/schema hashes, closed identifier universes and dependency graph, immutable model/recognition/reference/live/candidate ownership, exact E/source/M formulas, complete objective/cache records, the completed/failed attempt union, five positives, seven controls, fieldwise exact-candidate allowances, order-21/order-17 term budgets, exact zero-stochastic delta allowance, byte-bearing gate evaluation, and configuration-only rejection of unsupported MM.
- **Interface consistency:** `H4GaussianSolver`, `H4MaterializedProblem`, and the native diagnostic records live in the inference layer, while `H4NeutralProblem`, solver results, and tuple-ordered selected moments remain dependency-light protocol types. One generated neutral problem is materialized exactly once into raw owned tensors; both independent solver arms receive that same materialized object by identity, while the oracle receives its canonical bytes. No conversion, hash, count, memory, or diagnostic-replay work occurs between timed representations. `CompleteElboEvaluation` is produced only by the complete evaluator. Completed outcomes embed complete before/after evaluations; failed outcomes carry only phase-valid complete or partial evidence through `H5AttemptOutcome`. `observed_affected_factor_ids` comes only from ordered factor-input hashes, and `execute_update` alone accepts or rolls back. H5's parser binds both raw fixture byte objects to `H5ReferenceState`, while the runner passes those same captured bytes into byte-bearing production and oracle seams and publishes a separate H5 payload.
- **Evidence discipline:** Focused RED/GREEN commands are noncumulative. The milestone preflight requires every plan, preregistration, source, config, launcher, and test file to be tracked; rejects nonignored untracked content outside `.verification`; records the exact dirty-content digest and prior-ledger hashes; and rechecks them through closure. The only full suite and full timing occur at one shared exact revision. Reviewers inspect rather than rerun. A defect found after ledger activation closes the current revision `INCONCLUSIVE`, preserves it, repairs only after tool-driven retirement, and permits exactly one replacement revision/run/ledger. The coupled ledger separates claims by gate and preserves every earlier ledger.
- **Placeholder scan:** The plan contains no unspecified H4 canonical factor source, objective sign/constants, generator distributions, control blocks, selected-moment inventory, thread restoration rule, H5 fixture producer/parser, raw-byte digest comparison, source-row identity, model-snapshot owner, short-digest rejection rule, or required-tracked surface. The exact H5 fixture SHA-256 is intentionally not invented: Task 5 authors the tracked bytes and pins `SHA256(raw_fixture_bytes)` in parser/tests before GREEN/commit; Task 9 copies and verifies that literal. H6--H8/training and MM activation remain explicit nonclaims, not hidden implementation gaps.
- **American English:** Terminology uses American English throughout.

Plan implementation must stop after Task 10 with the exact H4 result, exact H5 result, JUnit totals, atomic artifact path, and validated `.verification/h4-h5-<FULL_HEAD>-ledger.json` reported from current evidence.
