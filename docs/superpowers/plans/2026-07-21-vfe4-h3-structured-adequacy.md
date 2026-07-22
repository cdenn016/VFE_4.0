# VFE 4.0 H3 Structured-Posterior Adequacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic synthetic H3 gate showing that a differentiable full-SPD Gaussian recognition family closes the frozen coupled posterior gap that a fine factorized diagonal Gaussian cannot represent, while that advantage disappears on a separately authored zero-coupling control whose exact posterior factorizes.

**Architecture:** Keep H3 independent of the H1/H2 source-mixture fixture and immutable H1 literal records. Two frozen source-free Gaussian JSON fixtures feed a direct PyTorch normalized-factor model, two differentiable PyTorch recognition arms, and an independent NumPy posterior/evidence/mean-field oracle; only `verification/h3_gate.py` may compare all three paths. Extend the H2 unified click-run and its one editable configuration dictionary to the ordered prefix `("H1","H2","H3")`, capture each fixture required by the requested prefix exactly once, never touch H3 fixtures for shorter prefixes, and publish one atomic `validation/h3.json` alongside H1/H2 only for the H3 prefix.

**Tech Stack:** Python 3.10+, PyTorch float64 on CPU, deterministic `torch.optim.LBFGS` with strong-Wolfe line search, NumPy float64, pytest, atomic JSON artifacts, SHA-256 provenance, JUnit XML.

## Global Constraints

- Begin only after H1 and H2 are implemented and their ordered gate artifact is passing at the current source revision. H3 consumes the H2 unified runner/factor conventions but does not change either earlier fixture, objective, or result semantics.
- H3 is a bounded synthetic adequacy gate, not a language-model, predictive, training, scaling, or generic theorem about correlation.
- The coupled and zero-control fixtures are source-free, `T=1`, `d_z=d_m=1`, float64 CPU laws in the exact coordinate order `[z0,m0,z1,m1]`.
- Coupled fixture: `z0,m0` are independent `N(0,1)`; `m1|m0 ~ N(0.8*m0,0.36)`; `z1|z0,m1 ~ N(0.7*z0+0.6*m1,0.25)`; `x1|(z1,m1) ~ N((z1,m1),0.64 I2)`; observed `x=(1.1,0.2)`.
- The coupled reference posterior freezes

  ```text
  J = [[ 2.96,              0.0,   -2.8,               1.68             ],
       [ 0.0,               2.77777777777778, 0.0,    -2.22222222222222],
       [-2.8,               0.0,    5.5625,           -2.4              ],
       [ 1.68,             -2.22222222222222, -2.4,    5.78027777777778]]
  h = [0.0, 0.0, 1.71875, 0.3125]
  log p(x) = -2.6536596233553
  analytic fine-factorized reverse-KL gap = 0.6815463199745935 nats
  ```

- The zero control is a separately authored and separately hashed JSON file, never a runtime transformation of the coupled object. It keeps the same initial law, observation law, dimensions, and coordinate order; declares `m1 ~ N(0,0.36)` and `z1 ~ N(0,0.25)` with no parents; observes `x=(0.4,-0.7)`; and freezes exact posterior precision `diag(1,1,5.5625,4.34027777777778)`.
- The three computational paths are: differentiable PyTorch structured full-SPD recognition, differentiable PyTorch fine-factorized diagonal recognition, and an independent NumPy exact-posterior/evidence/analytic reverse-KL mean-field oracle.
- The production objective evaluates `E_q[log p(y,x)] + H(q)` directly as a sum of normalized initial, transition, and observation factor expectations. It must not evaluate `log_evidence - KL`, accept oracle outputs, or import `verification`.
- Both PyTorch arms parameterize precision as `J_q=L_q L_q.T`. Every Cholesky diagonal is `exp(raw_diagonal)`. The structured arm learns all six strict-lower entries of a `4x4` `L_q`; the factorized arm has no off-diagonal parameters.
- Every optimization uses a fresh parameter module and fresh optimizer, common initialization `mu=zeros(4)` and `J_q=I_4`, CPU float64, and no randomness.
- Freeze the optimizer exactly: maximize ELBO by minimizing its negative with `torch.optim.LBFGS(lr=1.0, max_iter=1, max_eval=25, tolerance_grad=1e-12, tolerance_change=1e-18, history_size=20, line_search_fn="strong_wolfe")`; allow at most 200 accepted outer iterations and 5,000 closure evaluations; enforce that closure cap before every closure evaluation with a dedicated budget-exhaustion exception; require finite values, terminal gradient infinity norm at most `1e-8`, absolute accepted-objective change at most `1e-12` for three consecutive accepted iterations, and no source of nondeterminism.
- Freeze the H3 admissibility envelope at `D=4`, `lambda_min(J) >= 1e-4`, `lambda_max(J) <= 1e4`, `kappa_2(J) <= 1e6`, and `||mu||_inf <= 4` for each exact posterior and terminal recognition law. Reject rather than jitter, clip, pseudo-invert, regularize, or repair.
- Freeze `eps=np.finfo(np.float64).eps`, `gamma(n)=n*eps/(1-n*eps)`, `C=4096`, `N(D)=16*D+64`, and a per-optimized-operand solver contribution of `1e-7` nats. There is no singular/global H3 allowance. `scalar_allowance` uses only that scalar's value, absolute-summand accumulation, SPD operand condition numbers, and whether the scalar came from an optimized arm. `pair_allowance` adds the two scalar allowances and one `C*gamma(D+2)` comparison reduction. `three_operand_identity_allowance` and `four_operand_identity_allowance` add exactly their three/four scalar allowances and one `C*gamma(D+3)`/`C*gamma(D+4)` signed-reduction term. No invariant may borrow unrelated condition numbers or a run-wide maximum. There is no `rtol`, blanket `allclose`, residual-tuned threshold, or post-result adjustment.
- Freeze the two one-sided adequacy thresholds as signed three-way decisions. For the coupled gap, `margin_gap=G-0.50` and `A_gap` is the operand-local pair allowance for `(G,0.50)`. For structured closure, `margin_resolve=0.01*G-KL_cs` and `A_resolve` is the operand-local pair allowance for `(0.01*G,KL_cs)`. In either case, `margin > allowance` is PASS eligibility, `margin < -allowance` is a finite FAIL, and `-allowance <= margin <= allowance` is INCONCLUSIVE with a threshold-specific open obligation. An allowance may never be added to the favorable side to make an uncleared threshold pass.
- H3 can pass only when: both signed adequacy thresholds have PASS eligibility; all four arms converge; coupled factorized terminal KL matches the analytic gap under their pair allowance; every arm closes `log p(x)-ELBO=KL(q||p)` under its own three-operand identity allowance; the coupled delta identity closes under its four-operand identity allowance; both zero-control KLs, their two-operand ELBO delta, and both zero ELBO/KL identities close under their own allowances; and every decision allowance used by an H3 invariant is strictly less than `1%` of the coupled gap.

**Pre-promotion optimizer-coherence amendment (2026-07-22):** During focused Task 4 implementation, before any H3 promotion run, gate decision, or milestone evidence, the installed PyTorch L-BFGS implementation was confirmed to apply `tolerance_change` to a directional-derivative stop. The original `1e-15` value could stop above the separately frozen `1e-8` terminal gradient target. The internal stop is therefore amended to `tolerance_change=(1e-8)^2/100=1e-18`. The gradient and accepted-objective convergence targets and every other optimizer, budget, threshold, and decision setting remain unchanged; this is a protocol-coherence correction rather than a post-result adjustment.

The same focused pre-promotion review identified full-loss float64 quantization near the factorizing zero-control optimum. For each outer step, the first closure captures a detached immutable reference `q0` from that same live `q`, without another objective evaluation, and all closures in that step minimize the negative stable direct difference. Per factor use `a=row@q.mean-target`, `a0=row@q0.mean-target`, `s=solve_triangular(L_q,row)`, `s0=solve_triangular(L_q0,row)`, `variance_delta=dot(s-s0,s+s0)`, `expected_square_delta=(a-a0)*(a+a0)+variance_delta`, and `factor_delta=-0.5*expected_square_delta/variance`; use `entropy_delta=-sum(log(diag(L_q))-log(diag(L_q0)))`; combine them by ordinary tensor addition. Reset `q0` every outer step. Because `q0` is detached and fixed, this is a q-independent additive recentering: its q-gradient equals the full direct ELBO gradient and it preserves the strong-Wolfe conditions. The factored mean identity avoids subtracting rounded full scalars, and the solve-vector identity evaluates `||s||^2-||s0||^2` without subtracting two rounded order-one variances, completing stabilization of covariance and strict-lower precision-Cholesky directions. Accepted diagnostics still evaluate and record the full direct ELBO. The difference path may use no normalized constants, oracle, canonical form, evidence, inverse, repair, changed tolerance, or relaxed convergence/decision rule. It was adopted before promotion and gate evidence.
- A finite, converged invariant outside its allowance or a signed threshold with `margin < -allowance` is `FAIL`. A finite signed threshold inside `[-allowance,+allowance]` is not false and not passing: it is `INCONCLUSIVE` with `resolve coupled gap threshold outside allowance` or `resolve structured closure threshold outside allowance`. A missing/mismatched fixture hash, parse failure, PyTorch/NumPy/frozen-reference disagreement, nonconvergence, nonfinite trajectory, out-of-envelope law, nonfactorizing control, or allowance too large to decide is also `INCONCLUSIVE` with an explicit open obligation.
- Each task runs only its named focused RED/GREEN tests. Do not run cumulative tests after tasks. Run the full pytest suite with JUnit exactly once at the H3 milestone candidate revision; if later review changes source, tests, configuration, or fixtures, invalidate that evidence and run one replacement full suite at the new candidate revision.
- Reviewers inspect implementer command output, the exact-revision JUnit XML, the click-run artifact, manifest hashes, and the verification claim ledger. Reviewers do not rerun implementer tests or the full suite.
- Preserve every existing verification ledger, especially the H2 `.verification/ledger.json`. H3 uses only `.verification/h3-<FULL_HEAD>-ledger.json`, where `<FULL_HEAD>` is the exact 40-character candidate revision. Before H3 activation, an existing `.verification/active.json` is a fail-closed stop: do not delete, overwrite, or repoint it. A replacement candidate gets a new revision-specific H3 ledger path; prior H2/H3 ledgers remain untouched.
- Preserve the click-to-run/no-required-CLI contract. Do not add `argparse`, required environment variables, a second launcher, or a second editable configuration dictionary.

## File Map and Dependency Boundaries

| File | Responsibility |
|---|---|
| `vfe4/types/h3.py` | Immutable H3-only fixture, optimizer-config, arm-result, and gate-result records; no H1 literal mutation. |
| `vfe4/validation/h3_fixture.py` | Strict byte parser and independent-control relationship validator for the two H3 fixtures. |
| `vfe4/validation/fixtures/h3_coupled_v1.json` | Immutable coupled Gaussian law and frozen posterior reference. |
| `vfe4/validation/fixtures/h3_zero_control_v1.json` | Independently authored zero-coupling law and frozen diagonal posterior reference. |
| `vfe4/generative/reference_h3.py` | Normalized scalar Gaussian factors, PyTorch log joint, and diagnostic canonical assembly. |
| `vfe4/recognition/reference_h3.py` | Autograd-preserving structured and factorized precision-Cholesky parameterizations. |
| `vfe4/objective/h3_gaussian.py` | Direct differentiable `E_q[log p(y,x)] + H(q)` factor evaluation. |
| `vfe4/inference/h3_optimize.py` | Fresh-arm deterministic L-BFGS optimization and convergence records. |
| `verification/numpy_oracles/h3_posterior.py` | Independent NumPy JSON parser, exact posterior/evidence, analytic mean-field optimum/gap, and terminal reverse KL. |
| `verification/h3_budget.py` | Frozen operand-local scalar, pair, three-/four-operand identity, and decisiveness allowances. |
| `verification/h3_gate.py` | Fixture identity, path agreement, optimization, adequacy/control decisions, and H3 payload. |
| `verification/run_gates.py` | Extend the H2 ordered runner to H1/H2/H3 and conditional one-time capture of only requested fixture bytes. |
| `docs/preregistrations/2026-07-21-h3-structured-adequacy.md` | Frozen fixtures, optimizer, envelope, operand-local allowance functions/records, decision table, artifact schema, and nonclaims. |

Production imports stop at `vfe4`. The independent NumPy oracle imports only Python, NumPy, and JSON. `verification/h3_gate.py` is the only module that receives both production and oracle outputs. H3 runtime tensors never pass through H1/H2 immutable tensor records because those validation boundaries may intentionally detach and clone.

## Design Rationale and Nonclaims

- The fine comparison family is coordinatewise `q(z0)q(m0)q(z1)q(m1)`, represented by diagonal precision. It is variational mean field, not a thermodynamic population mean field and not an agent-block family.
- For a Gaussian target with precision `J_p`, the reverse-KL optimum in that fine family has `mu_q=mu_p`, diagonal variances `1/J_p[ii]`, and gap `0.5*(sum(log(diag(J_p)))-logdet(J_p))`. The NumPy oracle derives this; production never substitutes the formula for optimization.
- The structured full-SPD family contains the exact four-dimensional posterior. A successful result demonstrates adequacy and optimizer reachability only on these two bounded conjugate fixtures.
- The control changes the generative model and observation in a matched, independently serialized way. It is not generated by zeroing fields in the coupled fixture at runtime and is not a recognition-family ablation.
- Dense `4x4` diagnostic matrices are allowed at H3. No H4 timing/allocation benefit, H5 monotonic-update guarantee, H6 language/prefix result, H7 covariance claim, or H8 sparse-scaling result follows.
- L-BFGS accepted-iterate traces are convergence evidence, not H5 evidence. H3 makes no general monotonicity statement about coordinate, MM, generalized-EM, SGD, or Adam updates.

---

### Task 1: Freeze H3-only types, fixtures, control independence, and preregistration

**Files:**

- Create: `vfe4/types/h3.py`
- Modify: `vfe4/types/__init__.py`
- Create: `vfe4/validation/h3_fixture.py`
- Modify: `vfe4/validation/__init__.py`
- Create: `vfe4/validation/fixtures/h3_coupled_v1.json`
- Create: `vfe4/validation/fixtures/h3_zero_control_v1.json`
- Create: `tests/unit/test_h3_fixture.py`
- Create: `docs/preregistrations/2026-07-21-h3-structured-adequacy.md`

**Interfaces:**

- Produce immutable `H3Fixture`, `H3ScalarFactorRecord`, `H3InitializationConfig`, `H3OptimizationConfig`, `H3DecisionConfig`, `H3ArmResult`, `H3GateResult`, and `H3FixtureHashes` without importing or changing `H1Fixture`, H1 factor records, or their literal fields.
- `H3GateResult` owns `gate: Literal["H3"]`, both fixture IDs, `GateStatus`, measurements, invariants, immutable `allowances_by_invariant`, obligations, and the same fail-closed PASS/FAIL/INCONCLUSIVE consistency checks as the shared H1/H2 result. It deliberately has no singular `residual` or `calibrated_allowance`: every allowance-bearing comparison invariant names its own residual/margin and operand-local allowance record. Hash/control availability, condition-envelope, and optimizer-convergence eligibility invariants retain their exact configured rules but do not fabricate a rounding allowance. The runner later accepts `GateResult | H3GateResult`; it does not widen H1 fixture literals to pretend H3 is an H1 fixture.
- Produce `parse_h3_fixture_bytes(data: bytes, *, expected_fixture_id: Literal["h3-coupled-v1","h3-zero-control-v1"]) -> H3Fixture` and `validate_independent_control(coupled: H3Fixture, zero: H3Fixture) -> None`.
- The parser stores immutable Python numeric tuples. It does not construct an H1 object, mutate input, or retain mutable JSON containers.

- [ ] **Step 1: Write the preregistration and the two fixture files before running any H3 calculation.** Use identical root schemas with exact fields `fixture_schema_version`, `fixture_id`, `kind`, `horizon`, `dimensions`, `continuous_order`, `initial`, `transitions`, `observation`, and `reference`. The coupled reference contains the exact `J`, `h`, evidence, and mean-field gap from Global Constraints. The zero reference contains the exact diagonal precision and no copied coupled transition object. Encode every factor as normalized scalar Gaussian data: a coordinate-row vector, target scalar, and variance.

  The six coupled rows in `[z0,m0,z1,m1]` order are:

  ```text
  ([1,0,0,0], 0.0, 1.0)
  ([0,1,0,0], 0.0, 1.0)
  ([0,-0.8,0,1], 0.0, 0.36)
  ([-0.7,0,1,-0.6], 0.0, 0.25)
  ([0,0,1,0], 1.1, 0.64)
  ([0,0,0,1], 0.2, 0.64)
  ```

  The six independently written control rows are:

  ```text
  ([1,0,0,0], 0.0, 1.0)
  ([0,1,0,0], 0.0, 1.0)
  ([0,0,0,1], 0.0, 0.36)
  ([0,0,1,0], 0.0, 0.25)
  ([0,0,1,0], 0.4, 0.64)
  ([0,0,0,1], -0.7, 0.64)
  ```

  The preregistration copies every global threshold and status rule, states that neither JSON was selected from optimizer residuals, defines the artifact schema below, and explicitly defers H4--H8.

- [ ] **Step 2: Write the focused fixture/type tests.** Require exact root fields, exact IDs/order/dimensions, positive finite variances, six factors, frozen reference shapes, immutable copies, and fail-closed H3 result states. Assert `H3GateResult` accepts an immutable exact-name allowance mapping and exposes no singular `calibrated_allowance`; reject a missing/extra allowance for any allowance-bearing comparison invariant and reject invented allowances on hash/control/envelope/convergence eligibility invariants. Reject unknown/missing fields, booleans as numbers, NaN/Inf, wrong row length, nonpositive variance, duplicate IDs, changed coordinate order, control parents/couplings, changed common initial law, changed observation covariance/map, and a control posterior with any off-diagonal entry. Assert the two fixture paths and raw SHA-256 digests differ.

- [ ] **Step 3: Run the Task 1 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h3_fixture.py -q
  ```

  Expected: collection fails because `vfe4.types.h3` and `vfe4.validation.h3_fixture` do not exist.

- [ ] **Step 4: Implement the H3 records and strict byte parser.** Validate exact schema sets rather than accepting optional extras. `validate_independent_control` must compare the two parsed records field by field and allow differences only in `fixture_id`, `kind`, the two transition factor rows, observation values, and reference values. It must additionally require zero transition parent coefficients, diagonal frozen posterior precision, and exact factor order. Never expose `zero_couplings(coupled)` or another transformation helper.

- [ ] **Step 5: Freeze and record raw fixture digests.**

  ```powershell
  Get-FileHash -Algorithm SHA256 vfe4/validation/fixtures/h3_coupled_v1.json
  Get-FileHash -Algorithm SHA256 vfe4/validation/fixtures/h3_zero_control_v1.json
  ```

  Expected: two distinct 64-hex digests. Copy those measured values into named constants in `h3_fixture.py` and the preregistration in the same commit; do not normalize/re-serialize fixture content before hashing.

- [ ] **Step 6: Run the Task 1 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h3_fixture.py -q
  ```

  Expected: all fixture, independence, hash, and H3-result consistency tests pass.

- [ ] **Step 7: Commit Task 1.**

  ```powershell
  git add vfe4/types/h3.py vfe4/types/__init__.py vfe4/validation/h3_fixture.py vfe4/validation/__init__.py vfe4/validation/fixtures/h3_coupled_v1.json vfe4/validation/fixtures/h3_zero_control_v1.json tests/unit/test_h3_fixture.py docs/preregistrations/2026-07-21-h3-structured-adequacy.md
  git commit -m "test: freeze H3 structured adequacy fixtures"
  ```

---

### Task 2: Add the normalized PyTorch model, independent NumPy oracle, and frozen budget

**Files:**

- Create: `vfe4/generative/reference_h3.py`
- Modify: `vfe4/generative/__init__.py`
- Create: `verification/numpy_oracles/h3_posterior.py`
- Modify: `verification/numpy_oracles/__init__.py`
- Create: `verification/h3_budget.py`
- Create: `tests/unit/test_h3_reference_model.py`
- Create: `tests/oracle/test_h3_numpy_oracle.py`

**Interfaces:**

- Produce `H3ScalarGaussianFactor(row: Tensor, target: Tensor, variance: Tensor)`, `H3CanonicalJoint(precision, natural, log_constant)`, and `H3GenerativeModel.from_fixture(fixture)` with `factors`, `log_joint(y)`, and `canonical_joint()`.
- Produce NumPy-only `H3PosteriorOracleEvaluation(fixture_id, precision, natural, mean, covariance, log_evidence, analytic_factorized_precision, analytic_factorized_mean, analytic_factorized_reverse_kl, diagnostics)` and `evaluate_h3_posterior_oracle(data: bytes, *, expected_fixture_id: str) -> H3PosteriorOracleEvaluation`.
- Produce `reverse_kl_to_oracle(oracle, *, mean: np.ndarray, precision: np.ndarray) -> float` for gate-side evaluation of terminal PyTorch laws.
- `verification.h3_budget` exposes `gamma_n`, `scalar_allowance`, `pair_allowance`, `three_operand_identity_allowance`, `four_operand_identity_allowance`, and `allowance_is_decisive`, using only the frozen constants and the operands belonging to the invariant being checked.

- [ ] **Step 1: Write the PyTorch model and NumPy oracle tests.** For both fixtures, compare the sum of six scalar `log_prob` values against `model.log_joint(y)` at fixed finite `y`, and compare PyTorch diagnostic canonical `(J,h,c)` with a hand accumulation of `row outer row / variance`, `target*row/variance`, and normalized constants. The NumPy test independently parses raw JSON bytes and verifies the coupled `J`, `h`, evidence, and analytic gap plus the control diagonal precision and zero analytic gap. Assert the oracle module imports neither `torch` nor `vfe4`.

- [ ] **Step 2: Run the Task 2 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_h3_reference_model.py tests/oracle/test_h3_numpy_oracle.py -q
  ```

  Expected: collection fails because the H3 generative, oracle, and budget modules do not exist.

- [ ] **Step 3: Implement normalized PyTorch factors.** Each factor must evaluate

  ```python
  residual = row @ y - target
  return -0.5 * (
      residual.square() / variance
      + torch.log(2.0 * torch.pi * variance)
  )
  ```

  `canonical_joint()` is diagnostic assembly from those same public normalized factors. It may not read the frozen posterior reference fields. Validate float64 CPU, exact `(4,)` rows, positive finite variance, and six factors. Do not add source variables or categorical terms.

- [ ] **Step 4: Implement the independent NumPy oracle from raw factor rows.** Assemble

  ```python
  J = sum(np.outer(row, row) / variance for each factor)
  h = sum(target * row / variance for each factor)
  c = sum(-0.5 * (target**2 / variance + np.log(2*np.pi*variance)))
  L = np.linalg.cholesky(J)
  mu = np.linalg.solve(J, h)
  Sigma = np.linalg.solve(J, np.eye(4, dtype=np.float64))
  log_evidence = c + 0.5*h@mu + 0.5*4*np.log(2*np.pi) - np.log(np.diag(L)).sum()
  J_mf = np.diag(np.diag(J))
  gap = 0.5 * (np.log(np.diag(J)).sum() - 2*np.log(np.diag(L)).sum())
  ```

  Symmetrize only the solve result for reporting, not to repair an invalid input. Reject failed Cholesky, nonfinite outputs, and reference disagreement rather than using jitter/pseudoinverse. `reverse_kl_to_oracle` implements the oriented dense Gaussian formula with Cholesky solves and slog-determinants; it never calls production.

- [ ] **Step 5: Implement the operand-local allowance functions literally.** Use:

  ```python
  EPS = float(np.finfo(np.float64).eps)
  C = 4096.0
  SOLVER_ALLOWANCE_NATS = 1.0e-7

  def operation_count(dimension: int) -> int:
      return 16 * dimension + 64

  def scalar_allowance(
      dimension: int,
      *,
      value: float,
      absolute_sum: float,
      kappas: tuple[float, ...],
      optimized: bool,
  ) -> float:
      rounding = C * gamma_n(operation_count(dimension)) * max(
          1.0, *kappas
      ) * max(1.0, abs(value), absolute_sum)
      return (SOLVER_ALLOWANCE_NATS if optimized else 0.0) + rounding

  def pair_allowance(
      dimension: int,
      *,
      left: float,
      right: float,
      left_allowance: float,
      right_allowance: float,
  ) -> float:
      comparison = C * gamma_n(dimension + 2) * max(
          1.0, abs(left), abs(right), abs(left) + abs(right)
      )
      return left_allowance + right_allowance + comparison

  def three_operand_identity_allowance(
      dimension: int,
      *,
      operands: tuple[float, float, float],
      operand_allowances: tuple[float, float, float],
  ) -> float:
      reduction = C * gamma_n(dimension + 3) * max(
          1.0, sum(abs(value) for value in operands)
      )
      return sum(operand_allowances) + reduction

  def four_operand_identity_allowance(
      dimension: int,
      *,
      operands: tuple[float, float, float, float],
      operand_allowances: tuple[float, float, float, float],
  ) -> float:
      reduction = C * gamma_n(dimension + 4) * max(
          1.0, sum(abs(value) for value in operands)
      )
      return sum(operand_allowances) + reduction
  ```

  Reject bools, empty condition-number collections, arity mismatches, nonfinite values, negative allowances/absolute sums/condition numbers, and dimensions other than four. `scalar_allowance` receives only condition numbers from the scalar's actual SPD operations; `optimized=True` only for a terminal-arm quantity that carries optimizer error. The pair and identity functions receive already computed operand-local allowances and may not accept a condition-number collection of their own. `allowance_is_decisive(allowance,decisiveness_scale)` is true only for finite nonnegative allowance, finite positive scale, and `allowance < 0.01*decisiveness_scale`.

- [ ] **Step 6: Run the Task 2 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h3_reference_model.py tests/oracle/test_h3_numpy_oracle.py -q
  ```

  Expected: both fixtures agree with their frozen references under operand-local scalar/pair allowances; the independent control gap is zero under its pair allowance; all three-/four-operand arities and malformed inputs fail closed.

- [ ] **Step 7: Commit Task 2.**

  ```powershell
  git add vfe4/generative/reference_h3.py vfe4/generative/__init__.py verification/numpy_oracles/h3_posterior.py verification/numpy_oracles/__init__.py verification/h3_budget.py tests/unit/test_h3_reference_model.py tests/oracle/test_h3_numpy_oracle.py
  git commit -m "feat: add H3 Gaussian model and posterior oracle"
  ```

---

### Task 3: Add autograd-preserving structured/factorized recognition and the direct ELBO

**Files:**

- Create: `vfe4/recognition/reference_h3.py`
- Modify: `vfe4/recognition/__init__.py`
- Create: `vfe4/objective/h3_gaussian.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `tests/unit/test_h3_autograd_objective.py`

**Interfaces:**

- Produce `H3RecognitionFamily = Literal["structured_full_spd","fine_factorized_diagonal"]` and `H3VariationalGaussian(family, mean, precision_cholesky)` with differentiable `linear_variance(row)`, `entropy()`, and `precision()`.
- Produce `StructuredH3Parameters(initialization: H3InitializationConfig)` and `FactorizedH3Parameters(initialization: H3InitializationConfig)`, each a fresh `torch.nn.Module`; `make_h3_parameters(family, initialization)` returns a new instance on every call and materializes exactly the configured four-zero mean/identity precision.
- Produce `H3ObjectiveEvaluation(expected_log_factors: tuple[Tensor,...], entropy: Tensor, elbo: Tensor)` and `evaluate_h3_elbo(model: H3GenerativeModel, q: H3VariationalGaussian) -> H3ObjectiveEvaluation`.
- Runtime tensor records validate without `.detach()`, `.clone()` of a graph-breaking source, `.item()`, NumPy conversion, H1/H2 `GaussianLaw`, or H2 `InformationGaussian.from_information`. Detachment occurs only when `h3_optimize` creates the terminal immutable result.

- [ ] **Step 1: Write the early autograd-retention and objective tests.** At common initialization on the coupled model, assert `evaluation.elbo.requires_grad`; `torch.autograd.grad` reaches every mean/diagonal parameter and all six structured strict-lower parameters with finite gradients; at least one coupled off-diagonal gradient is nonzero. Monkeypatch H2 `InformationGaussian.from_information` and the H1 `GaussianLaw` constructor to raise and prove H3 evaluation still works. Add `torch.autograd.gradcheck` over explicit mean/raw-Cholesky tensors for both families.

  Also assert factor reconstruction exactly:

  ```python
  assert evaluation.elbo == sum(evaluation.expected_log_factors) + evaluation.entropy
  assert tuple(q.precision().shape) == (4, 4)
  assert torch.equal(q.precision(), torch.eye(4, dtype=torch.float64))
  ```

  Reject wrong dtype/device/shape, nonfinite parameters, a non-lower Cholesky, an illegal factorized off-diagonal, and a model/q dimension mismatch.

- [ ] **Step 2: Run the Task 3 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h3_autograd_objective.py -q
  ```

  Expected: collection fails because the H3 recognition and objective modules do not exist.

- [ ] **Step 3: Implement both precision-Cholesky families.** Consume the immutable `H3InitializationConfig`, reject any non-frozen mean/precision value, and derive raw diagonal/lower leaves from that common law rather than keeping an unproven constructor default. Structured parameters contain a float64 mean leaf `(4,)`, raw diagonal `(4,)`, and six raw strict-lower leaves in row-major order `(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)`. Build

  ```python
  L = torch.zeros((4, 4), dtype=torch.float64)
  L[strict_lower_rows, strict_lower_columns] = raw_lower
  L = L + torch.diag(torch.exp(raw_diagonal))
  ```

  Factorized parameters build only `diag(exp(raw_diagonal))`; they must not allocate or mask learnable off-diagonal values. `linear_variance(row)` returns `||solve_triangular(L,row,column=True)||_2^2`, and entropy is

  ```python
  0.5 * dimension * (1.0 + math.log(2.0 * math.pi)) \
      - torch.log(torch.diagonal(L)).sum()
  ```

  These operations remain differentiable.

- [ ] **Step 4: Implement the direct production objective.** For every normalized scalar factor with residual `r@y-target`, compute

  ```python
  mean_residual = row @ q.mean - target
  expected_square = mean_residual.square() + q.linear_variance(row)
  expected_log_factor = -0.5 * (
      expected_square / variance + torch.log(2.0 * torch.pi * variance)
  )
  ```

  Sum the six expected factor logs and `q.entropy()` with ordinary tensor addition so autograd remains live. Do not call `canonical_joint`, `log_evidence`, `reverse_kl`, the NumPy oracle, or a complete-covariance constructor.

- [ ] **Step 5: Run the Task 3 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h3_autograd_objective.py -q
  ```

  Expected: both families retain autograd through the direct factor objective; monkeypatched detaching paths are unused; gradcheck passes.

- [ ] **Step 6: Commit Task 3.**

  ```powershell
  git add vfe4/recognition/reference_h3.py vfe4/recognition/__init__.py vfe4/objective/h3_gaussian.py vfe4/objective/__init__.py tests/unit/test_h3_autograd_objective.py
  git commit -m "feat: add differentiable H3 recognition objectives"
  ```

---

### Task 4: Add deterministic fresh-arm L-BFGS optimization

**Files:**

- Create: `vfe4/inference/__init__.py`
- Create: `vfe4/inference/h3_optimize.py`
- Create: `tests/unit/test_h3_optimize.py`

**Interfaces:**

- Produce `optimize_h3_arm(model: H3GenerativeModel, family: H3RecognitionFamily, initialization: H3InitializationConfig, config: H3OptimizationConfig) -> H3ArmResult`.
- The function owns parameter and optimizer construction, so callers cannot accidentally reuse optimizer history or parameters across arms.
- `H3ArmResult` records family, convergence/failure reason, accepted iteration count, closure evaluations, terminal ELBO, terminal gradient infinity norm, terminal objective change, terminal mean, terminal precision Cholesky, terminal precision, accepted ELBO tuple, and SHA-256 of the canonical JSON trace.

- [ ] **Step 1: Write focused optimizer tests.** Patch `torch.optim.LBFGS` with a counting wrapper and prove four calls over two fixtures/two families receive disjoint parameter identities and the exact frozen keyword settings. Pass the resolved H3 common-initialization record and assert every arm starts with its exact zero mean/identity precision; reject a missing or altered initialization rather than silently defaulting. Run each real arm twice and require byte-identical canonical result JSON/trace digest. Test a forced max-iteration path, a NaN closure, a line-search exception, and a terminal out-of-envelope law as nonconverged results with explicit failure reasons rather than false PASS data. Add a fake line search that invokes the closure 5,001 times: assert exactly 5,000 objective evaluations occur, the 5,001st invocation raises the dedicated internal `H3ClosureBudgetExhausted` before objective evaluation, and the public result freezes `failure_reason="closure_evaluation_budget_exhausted"` with `converged=False`.

- [ ] **Step 2: Run the Task 4 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h3_optimize.py -q
  ```

  Expected: collection fails because `vfe4.inference.h3_optimize` does not exist.

- [ ] **Step 3: Implement one accepted-iteration loop.** Construct one fresh module from the supplied immutable common initialization and one fresh L-BFGS per call. Define private `H3ClosureBudgetExhausted`. At the very first line of every closure, before zeroing gradients or evaluating the model, perform:

  ```python
  if closure_evaluations >= config.maximum_closure_evaluations:
      raise H3ClosureBudgetExhausted
  closure_evaluations += 1
  ```

  Then call `optimizer.zero_grad(set_to_none=True)`, evaluate the direct ELBO, minimize `-elbo`, reject nonfinite loss/gradients, call `backward()`, and return the scalar loss. Catch `H3ClosureBudgetExhausted` only at the `optimizer.step(closure)` boundary and freeze the exact public failure reason `closure_evaluation_budget_exhausted`; do not allow L-BFGS or a generic exception handler to swallow/relabel it. After every completed `optimizer.step(closure)`, reevaluate the accepted ELBO and gradient without retaining an old graph, append its finite Python value, and update the consecutive convergence counter.

  Converge only after three consecutive accepted iterates satisfy both `gradient_inf <= 1e-8` and `abs(elbo_t-elbo_(t-1)) <= 1e-12`. Stop inconclusively at 200 accepted iterations. The closure-local check is the authoritative 5,000-evaluation cap, including repeated strong-Wolfe calls inside one outer iteration. Do not reuse a closure graph, use `retain_graph=True`, seed randomness, warm-start one arm from another, or substitute the oracle optimum.

- [ ] **Step 4: Snapshot the terminal law only after optimization.** Convert tensors to finite tuples under `torch.no_grad()`, calculate eigenvalue/condition diagnostics without changing the law, serialize the accepted trace with sorted keys and compact separators, and hash those exact canonical bytes. A finite but unconverged result retains measurements for diagnosis but has `converged=False` and a nonempty failure reason.

- [ ] **Step 5: Run the Task 4 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h3_optimize.py -q
  ```

  Expected: all four real arms converge deterministically from the common initialization; repeated results/digests agree; forced failures remain explicit and finite where available.

- [ ] **Step 6: Commit Task 4.**

  ```powershell
  git add vfe4/inference/__init__.py vfe4/inference/h3_optimize.py tests/unit/test_h3_optimize.py
  git commit -m "feat: optimize H3 recognition arms deterministically"
  ```

---

### Task 5: Add the fail-closed H3 gate, adequacy decision, and artifact payload

**Files:**

- Create: `verification/h3_gate.py`
- Create: `tests/promotion/test_h3_gate.py`
- Modify: `docs/preregistrations/2026-07-21-h3-structured-adequacy.md`

**Interfaces:**

- Produce `H3GateEvaluation(result, fixture_hashes, oracle_by_fixture, arms_by_fixture, comparisons, allowances_by_invariant, validation_payload)`; `allowances_by_invariant` is immutable and has exactly one operand-local allowance record for every allowance-bearing comparison invariant, while nonnumerical and eligibility-only invariants remain outside that mapping.
- Produce `evaluate_h3(config: ResolvedConfig, *, coupled_fixture_bytes: bytes | None = None, zero_control_fixture_bytes: bytes | None = None) -> H3GateEvaluation` and `h3_validation_payload(evaluation) -> dict[str, object]`.
- `None` exists for focused standalone tests and reads each named file once. The unified runner in Task 6 always supplies already captured bytes.

- [ ] **Step 1: Write the promotion test with exact invariant names.** Require the ordered invariant families:

  ```text
  fixture_hashes_match
  independent_control_contract
  coupled_frozen_reference_agreement
  zero_frozen_reference_agreement
  pytorch_numpy_canonical_agreement
  posterior_condition_envelope
  all_arms_converged
  coupled_oracle_gap_minimum
  all_invariant_allowances_decisive
  coupled_structured_fraction_resolved
  coupled_factorized_analytic_gap
  coupled_structured_elbo_kl_identity
  coupled_factorized_elbo_kl_identity
  coupled_delta_adequacy_identity
  zero_structured_kl
  zero_factorized_kl
  zero_delta_adequacy
  zero_structured_elbo_kl_identity
  zero_factorized_elbo_kl_identity
  ```

  Assert this tuple exactly so an adequacy/control obligation cannot disappear silently. Add status tests proving a finite converged adequacy miss is FAIL, while missing hash, reference disagreement, nonconvergence, nonfinite output, out-of-envelope precision, control off-diagonal, or any one nondecisive invariant allowance is INCONCLUSIVE. Assert each allowance-bearing comparison invariant records its allowance kind (`scalar`, `pair`, `three_operand_identity`, or `four_operand_identity`), exact operands, exact operand condition numbers, exact operand allowances, final allowance, residual or signed decision margin, named decisiveness scale, and `decisiveness_ratio=allowance/decisiveness_scale`. The nat-valued adequacy/control decisions use `G` as their scale; canonical reference/path comparisons use `max(1,abs(each compared operand))` in their own units.

  Add exact boundary tables for both signed thresholds. For `margin_gap=G-0.50` and `margin_resolve=0.01*G-KL_cs`, use `np.nextafter` or `math.nextafter` to assert: a margin immediately above `+A` is PASS eligibility; margins exactly `+A`, `0`, and `-A` are INCONCLUSIVE with the correct named obligation; and a margin immediately below `-A` is FAIL. Also assert an overall gate with all other evidence passing maps threshold PASS eligibility to PASS, threshold FAIL to FAIL, and threshold INCONCLUSIVE to INCONCLUSIVE rather than treating a finite indecisive margin as a false invariant.

- [ ] **Step 2: Run the Task 5 test for RED.**

  ```powershell
  python -m pytest tests/promotion/test_h3_gate.py -q
  ```

  Expected: collection fails because `verification.h3_gate` does not exist.

- [ ] **Step 3: Implement capture, identity, and oracle agreement first.** Hash the supplied raw bytes before parsing. Require exact expected digests, different fixture digests, strict production parsing, strict independent-control validation, and independent NumPy parsing of the same captured bytes. For every frozen/NumPy/PyTorch scalar or matrix element, compute a `scalar_allowance` from only that calculation's values, absolute summands, and target/assembled SPD condition numbers; compare two paths with `pair_allowance` from those two scalar allowances. Aggregate an invariant by its maximum normalized residual while retaining every element's own allowance record. Do not optimize if this preflight is missing or disagrees; return INCONCLUSIVE.

- [ ] **Step 4: Optimize all four arms and compute gate-side comparisons.** Before construction, assert the already resolved ordered family tuple, common initialization, `optimization_operation="maximize_direct_h3_elbo_lbfgs"`, and `expected_autograd_scope="h3_recognition_only"`; the resolver rejects a mismatch as invalid configuration before gate evaluation, so no arm may run under mislabeled provenance. Pass `config.h3.common_initialization` and the exact H3 optimizer config to every fresh `optimize_h3_arm` call independently in fixed order:

  ```text
  coupled/structured_full_spd
  coupled/fine_factorized_diagonal
  zero/structured_full_spd
  zero/fine_factorized_diagonal
  ```

  Convert terminal tuples to fresh NumPy arrays only in verification and use `reverse_kl_to_oracle`. Define

  ```text
  G = coupled_oracle.analytic_factorized_reverse_kl
  KL_cs, KL_cf = coupled terminal reverse KLs
  KL_zs, KL_zf = zero terminal reverse KLs
  Delta_c = ELBO_cs - ELBO_cf
  Delta_z = ELBO_zs - ELBO_zf
  resolved_fraction = (G - KL_cs) / G
  ```

  First build operand-local scalar allowance records: oracle evidence/gaps use `optimized=False` and only exact-posterior condition numbers; terminal ELBO/KL values use `optimized=True` and only that arm plus its exact posterior condition numbers. Then require:

  - compute a scalar record for `G` and an exact-zero-allowance protocol constant record for `0.50`; set `A_gap=pair_allowance(G,0.50,A_G,0)` and `margin_gap=G-0.50`; classify PASS eligibility only when `margin_gap>A_gap`, FAIL only when `margin_gap < -A_gap`, and otherwise INCONCLUSIVE with obligation `resolve coupled gap threshold outside allowance`;
  - compute a separate scalar record for the direct derived operand `0.01*G` from that calculation's absolute accumulation/target condition numbers and the terminal scalar record for `KL_cs`; set `A_resolve=pair_allowance(0.01*G,KL_cs,A_0.01G,A_KL_cs)` and `margin_resolve=0.01*G-KL_cs`; classify PASS eligibility only when `margin_resolve>A_resolve`, FAIL only when `margin_resolve < -A_resolve`, and otherwise INCONCLUSIVE with obligation `resolve structured closure threshold outside allowance`; report `resolved_fraction` descriptively but never use `KL_cs <= 0.01*G + allowance` as a pass rule;
  - `abs(KL_cf-G) <= A_factorized`, using their pair allowance;
  - for each arm, `abs(log_evidence-ELBO-KL) <= A_identity`, using `three_operand_identity_allowance((log_evidence,ELBO,KL), ...)`;
  - `abs((ELBO_cs-ELBO_cf)-(KL_cf-KL_cs)) <= A_delta_c`, using `four_operand_identity_allowance((ELBO_cs,ELBO_cf,KL_cf,KL_cs), ...)`;
  - `abs(KL_zs-0) <= A_zero_structured` and `abs(KL_zf-0) <= A_zero_factorized`, each using a pair allowance whose exact-zero operand has allowance zero;
  - `abs(ELBO_zs-ELBO_zf) <= A_delta_z`, using the pair allowance for the two ELBO operands; and
  - both zero-arm `log_evidence-ELBO-KL` residuals under their own three-operand identity allowances.

  Use direct comparisons, not `allclose`. No comparison may receive condition numbers or absolute sums from another arm/invariant.

- [ ] **Step 5: Implement the status precedence and signed-threshold eligibility exactly.** Evaluation order is: fixture/hash availability; parser/control validity; frozen/PyTorch/NumPy agreement; finite/envelope checks; convergence; per-invariant allowance construction/decisiveness; signed-threshold eligibility; then remaining finite decision invariants. Every allowance-bearing comparison invariant must satisfy `allowance_is_decisive(its_allowance,its_decisiveness_scale)` independently; one nondecisive allowance makes `all_invariant_allowances_decisive` false and the gate INCONCLUSIVE with that invariant named. Next classify each signed threshold as `PASS_ELIGIBLE`, `FAIL`, or `INCONCLUSIVE` from its signed margin and pair allowance. A threshold `INCONCLUSIVE` returns overall INCONCLUSIVE with its named obligation; it is not stored as a finite failed invariant. A threshold `FAIL` or any remaining finite equality/control invariant outside its allowance returns FAIL. Only two threshold `PASS_ELIGIBLE` records plus all remaining passing invariants can yield PASS. Never turn nonconvergence or a near-boundary finite threshold into evidence that a family is inadequate.

- [ ] **Step 6: Emit the complete `validation/h3.json` schema.** Include:

  ```text
  schema_version and gate/status/obligations
  coupled and zero fixture IDs, relative paths, byte counts, expected/observed SHA-256
  canonical config SHA-256 and exact H3 gate profile: ordered recognition families, common zero/identity initialization, optimization operation, expected autograd scope, optimizer settings, and decision config
  frozen reference constants and independent oracle outputs/diagnostics
  four arm initializations, terminal laws, convergence facts, accepted trace digests
  every KL, ELBO, evidence difference, adequacy delta, resolved fraction
  allowance constants plus an `allowances_by_invariant` object recording each allowance-bearing comparison invariant's allowance kind, exact operands, per-operand absolute sums/condition numbers/optimized flag/scalar allowance, final pair or three-/four-operand allowance, residual or signed margin, named decisiveness scale, and `allowance/decisiveness_scale` ratio
  `threshold_decisions` entries for `coupled_oracle_gap_minimum` and `coupled_structured_fraction_resolved`, each with ordered operands, favorable margin formula/direction, signed margin, pair allowance, lower/upper boundary, eligibility in `PASS_ELIGIBLE|FAIL|INCONCLUSIVE`, and the exact open obligation when indecisive
  exact ordered invariant records
  explicit H3 bounded claim and H4--H8 nonclaims
  ```

  JSON contains terminal matrices as row-major arrays and uses finite JSON numbers only. Hash provenance always refers to raw bytes, while trace digests refer to canonical trace JSON; label both domains so they cannot be confused.

- [ ] **Step 7: Run the Task 5 test for GREEN.**

  ```powershell
  python -m pytest tests/promotion/test_h3_gate.py -q
  ```

  Expected: H3 is PASS on the frozen pair; the coupled gap decisively exceeds `0.50`; structured resolves at least `99%`; the factorized gap and all four ELBO/KL identities close; control KLs/delta close; every allowance-bearing comparison uses its own operand-local allowance, every nat-valued decision records `allowance/G<1%`, and every canonical comparison records its own-unit decisiveness ratio below `1%`. Both threshold boundary tables prove the exact three-way mapping: immediately above `+A` is PASS eligibility, the closed `[-A,+A]` band is INCONCLUSIVE with the threshold-specific obligation, and immediately below `-A` is FAIL.

- [ ] **Step 8: Commit Task 5.**

  ```powershell
  git add verification/h3_gate.py tests/promotion/test_h3_gate.py docs/preregistrations/2026-07-21-h3-structured-adequacy.md
  git commit -m "test: add the H3 structured adequacy gate"
  ```

---

### Task 6: Extend the single H2 click-run/config/artifact path to ordered H1/H2/H3

**Files:**

- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `verification/run_gates.py`
- Modify: `verify_vfe4.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h3-structured-adequacy.md`

**Interfaces and compatibility:**

- Extend ordered accepted gate prefixes only to `("H1",)`, `("H1","H2")`, and `("H1","H2","H3")`. Empty, H2/H3 without predecessors, reversed, duplicate, or unknown gates fail closed.
- Add optional `h3: H3ValidationConfig | None` to the resolved configuration. It must be absent for H1/H2 prefixes and required with exact frozen values for the H3 prefix. Its canonical fields include ordered `recognition_families=("structured_full_spd","fine_factorized_diagonal")`, immutable `common_initialization` with `mean=(0.0,0.0,0.0,0.0)` and `precision=((1.0,0.0,0.0,0.0),(0.0,1.0,0.0,0.0),(0.0,0.0,1.0,0.0),(0.0,0.0,0.0,1.0))`, `optimization_operation="maximize_direct_h3_elbo_lbfgs"`, and `expected_autograd_scope="h3_recognition_only"`. This is a separate immutable gate profile; do not add H3 fields to H1 fixture/model/recognition literals.
- Extend `VerificationRunResult.gate_results` to `tuple[GateResult | H3GateResult,...]` while keeping H1/H2 result validation intact.
- `verify_vfe4.main(config=CONFIG) -> VerificationRunResult` prints H1, H2, H3 in order and one artifact path. `_script_main` returns zero only when every requested gate passes.

- [ ] **Step 1: Update focused config/integration tests.** Assert import is side-effect free; the one editable `CONFIG` resolves exactly to `("H1","H2","H3")`; its canonical H3 profile records both ordered families, the explicit four-zero mean and `4x4` identity precision common initialization, `optimization_operation="maximize_direct_h3_elbo_lbfgs"`, `expected_autograd_scope="h3_recognition_only"`, and `threshold_decision_rule="signed_margin_three_way"`. Resolve both compatibility prefixes and prove they preserve the existing H1/H2 values (`recognition.family="structured_linear_gaussian_mixture"`, `inference.operation="evaluate_only"`, `optimization.e_like_update="none"`, `optimization.m_like_update="none"`, `optimization.expected_autograd_scope="none"`) with no H3 profile. Invalid prefixes and mismatched H3-section presence fail. One H3-prefix `main()` call evaluates each gate once and publishes exactly one manifest-checked run containing `config.json`, `provenance.json`, `environment.json`, `validation/h1.json`, `validation/h2.json`, and `validation/h3.json`. Preserve existing run-root/security cases.

  For both H1 and H1/H2 prefixes, monkeypatch both H3 fixture `Path.read_bytes` calls to raise if touched; assert the runs succeed, contain no H3 fixture hash/consumer/provenance key, and publish no `validation/h3.json`.

- [ ] **Step 2: Run the Task 6 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: failures show only H1/H2 prefixes are supported and no H3 payload is published.

- [ ] **Step 3: Add the exact H3 editable dictionary section.** It contains the two fixture IDs, both raw expected digests, `recognition_families=["structured_full_spd","fine_factorized_diagonal"]` in that order, `common_initialization={"mean":[0.0,0.0,0.0,0.0],"precision":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]}`, `optimization_operation="maximize_direct_h3_elbo_lbfgs"`, `expected_autograd_scope="h3_recognition_only"`, the exact optimizer settings, condition envelope, solver contribution, `threshold_decision_rule="signed_margin_three_way"`, `minimum_coupled_gap=0.50`, `minimum_resolved_fraction=0.99`, `coupled_gap_inconclusive_obligation="resolve coupled gap threshold outside allowance"`, `structured_closure_inconclusive_obligation="resolve structured closure threshold outside allowance"`, and `maximum_allowance_gap_fraction=0.01`. Resolution rejects any unknown, reordered, missing, or changed protocol literal. Canonicalization includes this H3 gate profile in `config_sha256` only when H3 is requested and never mutates `CONFIG`. The existing recognition/inference/optimization records remain explicitly scoped to H1/H2 evaluation and retain their old values for compatibility prefixes; the H3 profile is the authoritative provenance for H3 optimization.

- [ ] **Step 4: Extend the unified runner with conditional one-time byte capture.** Capture only fixtures consumed by the requested prefix:

  ```python
  h1_bytes = H1_FIXTURE_PATH.read_bytes()
  h3_coupled_bytes: bytes | None = None
  h3_zero_bytes: bytes | None = None
  if "H3" in config.validation.gates:
      h3_coupled_bytes = H3_COUPLED_FIXTURE_PATH.read_bytes()
      h3_zero_bytes = H3_ZERO_FIXTURE_PATH.read_bytes()
  ```

  Pass `h1_bytes` to requested H1/H2 evaluation and pass the two non-`None` H3 byte sequences to `evaluate_h3` only for the H3 prefix. Do not let a gate reread a fixture or recapture it after another gate runs. H1 and H1/H2 prefixes must neither stat/read H3 fixture content nor construct an H3 evaluation. Publish only after all requested gates return. Aggregate status is fail if any gate fails, inconclusive if none fails and any is inconclusive, otherwise pass.

- [ ] **Step 5: Extend provenance without weakening H2.** Keep the existing source/config/environment/dirty-content security path. For the H3 prefix, add ordered `gate_states`, a `fixture_hashes` mapping with raw expected/observed hashes for `h1-v1`, `h3-coupled-v1`, and `h3-zero-control-v1`, `gate_fixture_consumers={"H1":["h1-v1"],"H2":["h1-v1"],"H3":["h3-coupled-v1","h3-zero-control-v1"]}`, and the canonical H3 recognition-family/initialization/operation/autograd-scope profile. Manifest hashing covers all three validation payloads. H1 and H1/H2 runs contain only requested gate states, `h1-v1` hashes/consumers, and their existing evaluate-only/no-autograd profile; they must not provenance-bind H3 paths, hashes, IDs, settings, or consumers.

- [ ] **Step 6: Update the launcher and bounded documentation.** Keep one `CONFIG`, one `main`, and one `if __name__ == "__main__"` block. Before the milestone, README and preregistration describe only the frozen H3 synthetic coupled/control protocol, exact operand-local allowance contract, prefix-conditioned config/provenance, and H4--H8 nonclaims; they do not prestate measured H3/JUnit results. Do not add a family-selection CLI or a second H3 launcher.

- [ ] **Step 7: Run the Task 6 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: one mocked/default click-run evaluates ordered H1/H2/H3 from one captured fixture snapshot set and atomically publishes all three payloads with valid manifest/provenance; focused H1 and H1/H2 prefix runs do not read or provenance-bind either H3 fixture and preserve their evaluate-only/no-autograd configuration values.

- [ ] **Step 8: Commit Task 6.**

  ```powershell
  git add vfe4/config/schema.py vfe4/config/resolve.py verification/run_gates.py verify_vfe4.py tests/unit/test_config.py tests/integration/test_verify_vfe4.py README.md docs/preregistrations/2026-07-21-h3-structured-adequacy.md
  git commit -m "feat: publish ordered H1 H2 H3 verification"
  ```

---

### Task 7: Produce the one exact-revision H3 milestone record

**Files:**

- Modify: none. All tracked protocol and documentation content was completed and committed in Tasks 1, 5, and 6 before this candidate revision.
- Produce outside tracked source: `C:\tmp\vfe4-h3-milestone.xml`, one atomic run directory under the configured run root, and `.verification/h3-<FULL_HEAD>-ledger.json`, with `<FULL_HEAD>` replaced by the exact 40-character Task 6 candidate revision. Do not commit `.verification` or generated run artifacts.
- Verification-skill claim ledger: initialize and validate only the revision-specific H3 path through the installed skill tooling; populate one claim per check using the skill's schema. Preserve `.verification/ledger.json` and every prior revision-specific ledger byte-for-byte. The skill-owned `.verification/active.json` may be created only after the preflight/review steps below and must never be manually removed or overwritten.

**Evidence policy:** Task 7 is read-only with respect to every tracked file. The JUnit XML, click artifact, and revision-specific validated H3 ledger are bound to the exact clean Task 6 candidate revision; measured facts remain in the click artifact and H3 ledger rather than being copied into a later tracked documentation commit. If `.verification/active.json` exists at either H3 activation preflight, stop fail-closed and report the existing activation; never remove/repoint it or overwrite the H2 ledger. Reviewer-requested fixes are resolved before H3 activation. If a source-changing issue is discovered after activation, do not edit source: close the current revision's H3 ledger as INCONCLUSIVE with the exact repair obligation, validate/report that ledger so the verification hook can retire its own marker, preserve the ledger, then return to the owning task. A later fixed candidate uses `.verification/h3-<NEW_FULL_HEAD>-ledger.json` and one replacement full-suite run; no earlier ledger is deleted. Reviewers inspect evidence and do not rerun commands.

- [ ] **Step 1: Record the exact candidate revision and verify tracked cleanliness.**

  ```powershell
  git rev-parse HEAD
  git status --short
  if (Test-Path -LiteralPath '.verification/active.json') {
      throw 'existing verification activation blocks H3; preserve it and resolve its owning workflow'
  }
  if (Test-Path -LiteralPath '.verification/ledger.json') {
      Get-FileHash -Algorithm SHA256 -LiteralPath '.verification/ledger.json'
  }
  Get-ChildItem -LiteralPath '.verification' -File -Filter 'h3-*-ledger.json' -ErrorAction SilentlyContinue |
      Sort-Object FullName |
      Get-FileHash -Algorithm SHA256
  ```

  Expected: a 40-character commit ID, no tracked modifications, no active verification marker, and a recorded pre-H3 SHA-256 table for the H2 ledger and any prior H3 ledgers. Existing `.verification/ledger.json` and prior H3 ledgers are permitted and remain untouched. Generated `runs/` and `.verification/` state may be ignored/untracked under repository policy.

- [ ] **Step 2: Run the only H3 milestone full regression and write machine-readable evidence.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h3-milestone.xml
  ```

  Expected: pytest exits zero. Parse suite/test/failure/error/skip totals only from the JUnit XML. Do not report terminal dots, a remembered H2 total, or reviewer reruns.

- [ ] **Step 3: Run the single click-run and inspect the atomic artifact.**

  ```powershell
  python verify_vfe4.py
  ```

  Expected: the launcher prints `H1: pass`, `H2: pass`, `H3: pass`, and one run directory. Independently recompute `manifest.sha256`; verify source/config/environment identities, all three raw fixture hashes, gate-consumer mapping, ordered gate states, and the existence/schema of `validation/h3.json`.

- [ ] **Step 4: Have fresh reviewers inspect existing evidence only.** One reviewer checks protocol/spec/nonclaim coverage and control independence; one checks Gaussian canonical algebra, reverse-KL orientation, direct objective, operand-local allowance formulas, and both signed three-way threshold decisions; one checks autograd/optimizer isolation, status precedence, fixture capture, artifact/provenance, click-run compatibility, and preservation of the H2/prior-H3 ledger hashes. They cite focused outputs, JUnit XML, artifact files, and source lines. Resolve any Critical/Important issue by returning to its task; reviewers must not rerun implementer tests or the full suite.

- [ ] **Step 5: Start, populate, and validate the revision-specific evidence-gated H3 ledger without touching H2.** Read the verification contract plus code, mathematics, evidence, experiment, and general criterion files before assigning states. Recheck the exact revision and activation state, derive the path, and start closure mode:

  ```powershell
  $h3Head = (git rev-parse HEAD).Trim()
  if ($h3Head.Length -ne 40) { throw 'H3 requires a full 40-character HEAD' }
  if (Test-Path -LiteralPath '.verification/active.json') {
      throw 'existing verification activation blocks H3; preserve it and resolve its owning workflow'
  }
  $h3Ledger = ".verification/h3-$h3Head-ledger.json"
  if (Test-Path -LiteralPath $h3Ledger) {
      throw "revision-specific H3 ledger already exists and must not be overwritten: $h3Ledger"
  }
  & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" start --cwd . --ledger $h3Ledger --mode closure
  ```

  Use separate claims for: fixture hashes/control independence; frozen/NumPy/PyTorch canonical agreement; autograd retention; fresh common initialization; four-arm convergence; both signed threshold margins/three-way eligibility records; factorized analytic gap; four ELBO/KL identities; coupled delta identity; zero-control KL/delta; every operand-local allowance/decisiveness record; ordered publication/manifest; exact JUnit totals; and unchanged Step 1 hashes for `.verification/ledger.json` plus every prior H3 ledger. Give every assessed claim at least two uniquely identified views and one structured adjudicator; escalate to four/eight views when the skill's recorded triggers apply. Close only with current domain-eligible evidence at the candidate revision. Use INCONCLUSIVE for any missing obligation; LLM agreement is not closure. Populate exactly the printed revision-specific H3 ledger with `apply_patch` if the installed tool still has no claim-add operation. Re-derive the same path and validate it:

  ```powershell
  $h3Head = (git rev-parse HEAD).Trim()
  $h3Ledger = ".verification/h3-$h3Head-ledger.json"
  & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" validate $h3Ledger --cwd .
  ```

  Expected: validation exits zero and `.verification/h3-<FULL_HEAD>-ledger.json` remains bound to the exact candidate artifact revision. `.verification/ledger.json` and prior H3 ledgers have unchanged hashes. Do not commit `.verification`, manually delete `.verification/active.json`, or reuse this ledger path for another revision.

- [ ] **Step 6: Perform a final read-only revision/evidence cross-check and leave tracked source unchanged.**

  ```powershell
  git rev-parse HEAD
  git diff --exit-code
  git diff --cached --exit-code
  git status --short
  if (Test-Path -LiteralPath '.verification/ledger.json') {
      Get-FileHash -Algorithm SHA256 -LiteralPath '.verification/ledger.json'
  }
  Get-ChildItem -LiteralPath '.verification' -File -Filter 'h3-*-ledger.json' -ErrorAction SilentlyContinue |
      Sort-Object FullName |
      Get-FileHash -Algorithm SHA256
  ```

  Expected: `HEAD` is byte-for-byte the candidate revision recorded in Step 1 and in provenance/H3 ledger; both diff commands exit zero; no tracked file changed during JUnit, click-run, review, or ledger validation; the H2 ledger and every preexisting H3 ledger retain their Step 1 SHA-256; the only new ledger is the current revision-specific H3 ledger. Generated ignored/untracked evidence may appear according to repository policy. Report the exact revision, JUnit totals, artifact path, and validated `.verification/h3-<FULL_HEAD>-ledger.json` location from their evidence surfaces so the verification hook can validate the named active ledger; do not name `.verification/ledger.json` as the H3 ledger and do not edit or commit tracked documentation after validation.

## Out of Scope for This Plan

- Categorical source variables, source mixtures, language tokens, vocabulary emissions, datasets, checkpoints, learned model parameters, or amortized recognition networks.
- A claim that structured inference always outperforms factorized inference, that reverse KL is universally underdispersed, or that any language posterior has the coupled fixture's geometry.
- Runtime construction of the control by editing/copying the coupled fixture.
- Timing, memory, allocation, sparse-factor, information-versus-moment cost, or H4 claims.
- Coordinate/MM/generalized-EM monotonicity or H5 claims.
- Prefix safety, held-out prediction, attention equivalence, source attribution, or H6 claims.
- Population-frame covariance, base curvature/holonomy, decoder transformation, or H7 claims.
- `T=128`, `K=20`, sparse execution, or H8 claims.
- A registry/plugin architecture; H3 may inform a later separately reviewed registry design but does not introduce one.

## Self-Review of Plan Completeness

- **Spec coverage:** Tasks 1--7 cover the exact coupled/control laws, separate immutable H3 types, three computational paths, direct production ELBO, autograd-retention regression, exact deterministic optimizer settings and closure-local cap, independent oracle, analytic mean-field gap, operand-local scalar/pair/three-/four-operand allowances, signed three-way threshold decisions with boundary tests, status precedence, control independence, raw-byte/trace digest provenance, prefix-conditioned H3 config/autograd provenance, conditional fixture capture, ordered H1/H2/H3 runner, atomic `validation/h3.json`, reviewer evidence, a revision-specific H3 ledger that preserves H2, and one exact-revision JUnit run.
- **Task ordering:** Fixture/preregistration freeze precedes any H3 computation; model/oracle precede differentiable recognition/objective; autograd is proved before optimization; optimization precedes gate decisions; all tracked documentation is committed before runner integration closes; milestone tests and review precede H3 ledger activation so reviewer fixes cannot strand an active marker; final evidence leaves the tested revision and every earlier ledger unchanged.
- **Type/interface consistency:** H3 uses `H3OptimizationConfig`, `H3ArmResult`, and `H3GateResult` throughout; H1/H2 fixture literals remain unchanged; the runner consumes the explicit `GateResult | H3GateResult` union; production never imports the oracle.
- **Decision completeness:** Every PASS condition has a named invariant; every allowance-bearing comparison invariant has its own exact-operand allowance record and named-scale decisiveness ratio, while hash/control/envelope/convergence eligibility checks retain their exact non-allowance rules. Both one-sided thresholds use signed `PASS_ELIGIBLE`/`FAIL`/`INCONCLUSIVE` mappings: favorable margins must exceed the pair allowance, unfavorable margins must be below its negative, and the closed boundary band is inconclusive. Other finite converged misses map to FAIL; missing identity, disagreement, nonconvergence, nonfinite/out-of-envelope data, control invalidity, and any indecisive invariant allowance map to INCONCLUSIVE.
- **Nonclaim completeness:** H4 cost, H5 update, H6 language/prefix, H7 covariance, and H8 scale claims are explicitly deferred in global constraints, artifact payload, docs, and out-of-scope text.
- **Placeholder scan:** No implementation choice or threshold remains to be selected during execution. Fixture hashes are frozen before gate calculation; milestone measurements are recorded only in the exact-revision click artifact/JUnit/ledger surfaces because they are evidence, not design parameters or post-validation tracked-document edits.

Plan complete and saved to `docs/superpowers/plans/2026-07-21-vfe4-h3-structured-adequacy.md`. Implementation should proceed task-by-task only after H2 is green at the current revision.
