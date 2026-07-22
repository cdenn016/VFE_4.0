# VFE 4.0 H2 Information--Moment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify, component by component and after exact source-weighted aggregation, that the unchanged positive-weight `h1-v1` Gaussian-mixture law and complete ELBO agree between a new PyTorch information path, the unchanged PyTorch H1 moment path, and an independent NumPy dense-moment oracle.

**Architecture:** Add a Cholesky-backed precision-factor seam and direct canonical-factor assembly without calling either H1 `joint_component` method. The production information evaluator may retain dense `6x6` precision matrices for this bounded reference, but it obtains means, selected moment blocks, log determinants, quadratic forms, traces, and samples only through the factor interface; it never forms a covariance or inverse. Verification remains one-way: `verification/` may import `vfe4`, while production `vfe4` may never import `verification`.

**Tech Stack:** Python 3.10+, PyTorch float64, NumPy float64, pytest, atomic JSON artifacts, JUnit XML.

## Global Constraints

- Work on the existing dedicated H1-foundation branch/worktree. Preserve unrelated work and make exactly one commit after each task's named focused tests are green.
- H2 is representation verification only. It makes no optimizer, performance, gradient, H3/H4/H5/H7/H8, prediction, or scaling claim.
- Reuse the unchanged `vfe4/validation/fixtures/h1_v1.json` law: `T=2`, `D=6`, coordinate order `[z0,m0,z1,m1,z2,m2]`, source-path order `[(0,0),(1,0),(0,1),(1,1)]`, and positive recognition weights `(0.30,0.10,0.12,0.48)`.
- Evaluate all four fixed-source recognition and generative Gaussian components separately and then aggregate with the exact recognition source weights. Never moment-project the source-marginal mixture and never describe it as one Gaussian.
- Use the terminology exactly: natural coordinates are `(h,-J/2)`, expectation coordinates are `(mu,M)` with `M=Sigma+mu mu^T`, and `(mu,Sigma)` is the moment representation, not the Fisher-dual pair.
- Float64 only. Reject rather than jitter, pseudo-invert, clip, repair, rescale, or regularize.
- The frozen admissibility envelope is `D <= 6`, `lambda_min(J) >= 1e-4`, `lambda_max(J) <= 1e4`, `kappa_2(J) <= 1e6`, and `||mu||_inf <= 4`. Record the minimum Cholesky pivot, both extreme eigenvalues, `kappa_2`, and `||mu||_inf` for every `q` and `p` component.
- The preregistration records, without tuning the envelope, the observed eight-component calibration: `lambda_min >= 0.22580664749462973`, `lambda_max <= 10.189172671632396`, `kappa_2 <= 42.348335716414404`, and `||mu||_inf <= 0.29950000000000004`.
- Freeze `eps=np.finfo(np.float64).eps`, `gamma(n)=n*eps/(1-n*eps)`, `C=256`, and `N(D)=8*D+32`. A path allowance is `C*gamma(N(D))*max(1, every SPD operand kappa_2)*max(1, output inf norm, absolute-summand accumulation inf norm)`.
- Backward residual allowances for `J@mu-h` and `J@Sigma[:,B]-E_B` omit condition number and equal `C*gamma(N(D))*max(1, ||J||_inf*||solution||_inf+||rhs||_inf)`.
- A complete-ELBO allowance is the sum of its `K=12` signed local-term allowances plus `C*gamma(K+1)*max(1,sum(abs(term) for term in signed_terms))`.
- A pair allowance is `left_allowance + right_allowance + C*gamma(D+2)*max(1,||left||_inf,||right||_inf)`. There is no `rtol`, blanket `allclose`, empirical post-tuning, or threshold tuning on `h1-v1`. At the observed `kappa_2 <= 42.35`, the preregistered pair budget is approximately `3.86e-10 * scale`; this is descriptive, not a replacement for the invariant-specific calculation.
- The only promotion negative controls are `h=mu`, reversed log-determinant sign, conditional covariance substituted for an emission marginal covariance, and an instrumented inverse/full-identity-solve attempt. Each mathematical wrong-path residual must be at least `1e-3*scale`; otherwise H2 is `INCONCLUSIVE`, not passed. Malformed input coverage belongs in focused unit tests.
- Each task runs only its named new or directly modified tests for RED/GREEN. Do not run a cumulative suite after each task. Normally run the full pytest suite with JUnit once, at the exact milestone candidate revision, followed by one click-run. If a later review forces a source/test/config fix, invalidate that evidence and run one replacement full suite at the new candidate revision; never rerun it merely for confidence. Reviewers inspect implementer evidence and do not rerun identical commands.
- Preserve the click-to-run/no-required-CLI surface. Do not add `argparse`, environment requirements, a second launcher, or a second editable configuration dictionary.

## File Map and Dependency Boundaries

| File | Responsibility |
|---|---|
| `vfe4/types/information.py` | Immutable block/diagnostic records and the `PrecisionFactor` protocol. |
| `vfe4/numerics/precision.py` | Checked dense-Cholesky implementation of the approved factor seam. |
| `vfe4/numerics/information.py` | One fixed-source information Gaussian; no covariance property. |
| `vfe4/numerics/linear_gaussian.py` | Add normalized initial and affine-conditional factors directly into `(h,J)`. |
| `vfe4/recognition/reference_h2.py` | Direct H2 recognition-component assembly from `H1RecognitionFactorRecord`. |
| `vfe4/generative/reference_h2.py` | Direct H2 generative-component assembly from `H1GenerativeFactorRecord`. |
| `vfe4/objective/h2_information.py` | Information-form component diagnostics, selected emission moments, every local term, and complete ELBO. |
| `verification/numpy_oracles/h2_moment.py` | Independent NumPy-only parser, dense moment assembly, terms, and traces. |
| `verification/h2_budget.py` | The frozen scale-aware allowance formulas only. |
| `verification/h2_gate.py` | Three-path comparisons, envelope checks, four negative controls, and fail-closed H2 result. |
| `verification/run_gates.py` | Evaluate requested gates, then publish one atomic run. |
| `docs/preregistrations/2026-07-21-h2-information-moment.md` | Frozen H2 fixture, budget, controls, conditioning facts, and later closure record. |

Production imports stop at `vfe4`. The independent oracle imports only Python/NumPy. `verification/h2_gate.py` is the only layer that sees all three paths.

---

### Task 1: Add the precision factor and information-Gaussian primitives

**Files:**

- Create: `vfe4/types/information.py`
- Modify: `vfe4/types/__init__.py`
- Create: `vfe4/numerics/precision.py`
- Create: `vfe4/numerics/information.py`
- Modify: `vfe4/numerics/__init__.py`
- Create: `tests/unit/test_precision_factor.py`

**Interfaces:**

- Produces `MatrixBlock(rows: tuple[int,...], columns: tuple[int,...])`, `PrecisionDiagnostics(dimension, min_cholesky_pivot, lambda_min, lambda_max, kappa_2)`, and the approved `PrecisionFactor` methods `solve`, `logdet`, `selected_inverse`, and `sample`.
- Adds only two KL/log-density operations: `quadratic(value)` and `trace_inverse_product(left)`, where the latter returns `tr(J_left @ J_self^{-1})` without forming `J_self^{-1}`.
- Produces `DenseCholeskyPrecision(matrix)` and `InformationGaussian.from_information(h,J,factor_factory=DenseCholeskyPrecision)` with `mean()`, `log_normalizer()`, `entropy()`, `log_prob(value)`, `oriented_kl(other)`, and `selected_moment_blocks(blocks)`.
- `InformationGaussian` exposes cloned `h` and `J` for bounded H2 diagnostics, but deliberately has no `covariance`, `inverse`, or `moment_matrix` property.

- [ ] **Step 1: Write the focused unit tests.** Cover float64/symmetry/SPD validation; exact solves; `2*sum(log(diag(L)))`; rectangular selected blocks; deterministic sampling; log density; log normalizer; entropy; oriented `KL(q||p)`; and rejection of nonfinite, nonsymmetric, non-SPD, wrong-dimension, duplicate-index, and full-identity selected-block requests. Use hand-checkable diagonal and correlated `3x3` matrices, not randomized sweeps.

  The central tests must assert the oriented formula and absence of a covariance surface:

  ```python
  q = InformationGaussian.from_information(
      torch.tensor([0.4, -0.2], dtype=torch.float64),
      torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=torch.float64),
  )
  p = InformationGaussian.from_information(
      torch.tensor([-0.1, 0.5], dtype=torch.float64),
      torch.tensor([[1.4, -0.2], [-0.2, 2.2]], dtype=torch.float64),
  )
  assert q.oriented_kl(p).item() >= 0.0
  assert not hasattr(q, "covariance")
  assert torch.equal(q.factor.solve(q.h), q.mean())
  ```

- [ ] **Step 2: Run the Task 1 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_precision_factor.py -q
  ```

  Expected: collection fails because `vfe4.types.information` and `vfe4.numerics.precision` do not exist.

- [ ] **Step 3: Implement the seam exactly.** The protocol and core dense implementation must have these signatures:

  ```python
  @runtime_checkable
  class PrecisionFactor(Protocol):
      @property
      def dimension(self) -> int:
          raise NotImplementedError
      @property
      def diagnostics(self) -> PrecisionDiagnostics:
          raise NotImplementedError
      def solve(self, rhs: Tensor) -> Tensor:
          raise NotImplementedError
      def logdet(self) -> Tensor:
          raise NotImplementedError
      def selected_inverse(
          self, blocks: Sequence[MatrixBlock]
      ) -> Mapping[MatrixBlock, Tensor]:
          raise NotImplementedError
      def sample(self, noise: Tensor) -> Tensor:
          raise NotImplementedError
      def quadratic(self, value: Tensor) -> Tensor:
          raise NotImplementedError
      def trace_inverse_product(self, left: "PrecisionFactor") -> Tensor:
          raise NotImplementedError
  ```

  `selected_inverse` constructs only the selector columns requested by a call, solves them through the Cholesky factor, and rejects a request whose distinct column set is all `D` columns. `trace_inverse_product` uses the stable factor identity

  ```python
  whitened = torch.linalg.solve_triangular(self._chol, left._chol, upper=False)
  return torch.sum(whitened * whitened)
  ```

  after checking that `left` is the same supported factor implementation and dimension. `sample(noise)` solves `L.T @ centered = noise`; it does not call `solve(I)`.

  `InformationGaussian.oriented_kl(p)` implements

  ```python
  delta = p.mean() - self.mean()
  return 0.5 * (
      self.factor.trace_inverse_product(p.factor)
      + p.factor.quadratic(delta)
      - self.dimension
      + self.factor.logdet()
      - p.factor.logdet()
  )
  ```

  and validates every derived tensor as finite. Construction uses `torch.linalg.cholesky_ex` and fails on any nonzero info code. Do not import NumPy or `verification` here.

- [ ] **Step 4: Run the Task 1 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_precision_factor.py -q
  ```

  Expected: all tests in this file pass with zero failures and zero errors.

- [ ] **Step 5: Commit Task 1.**

  ```powershell
  git add vfe4/types/information.py vfe4/types/__init__.py vfe4/numerics/precision.py vfe4/numerics/information.py vfe4/numerics/__init__.py tests/unit/test_precision_factor.py
  git commit -m "feat: add H2 precision factor primitives"
  ```

---

### Task 2: Assemble recognition and generative `(h,J)` directly from normalized factors

**Files:**

- Create: `vfe4/numerics/linear_gaussian.py`
- Modify: `vfe4/numerics/__init__.py`
- Create: `vfe4/recognition/reference_h2.py`
- Modify: `vfe4/recognition/__init__.py`
- Create: `vfe4/generative/reference_h2.py`
- Modify: `vfe4/generative/__init__.py`
- Create: `tests/unit/test_h2_information_assembly.py`

**Interfaces:**

- Consumes the Task 1 `InformationGaussian` and the existing immutable H1 factor records and `SourcePath`.
- Produces `add_initial_gaussian(h,J,indices,mean,covariance)` and `add_scalar_conditional(h,J,target_index,parent_coefficients,offset,variance)`.
- Produces `assemble_recognition_information(factors: H1RecognitionFactorRecord, path: SourcePath) -> InformationGaussian` and `assemble_generative_information(factors: H1GenerativeFactorRecord, path: SourcePath) -> InformationGaussian`.
- Neither assembler accepts an `H1RecognitionLaw`/`H1GenerativeModel`, imports `reference_h1`, nor calls `joint_component`.

- [ ] **Step 1: Write the focused assembly tests.** For all four paths, compare `J@mu-h` with zero and compare direct-information log densities at fixed deterministic `y` values with the corresponding normalized H1 continuous factor log densities after subtracting source/emission terms. Monkeypatch both H1 `joint_component` methods to raise and prove both H2 assemblers still succeed. Add deterministic malformed tests for a repeated/forward parent index, nonpositive variance, out-of-support source, and wrong `h/J` shape.

- [ ] **Step 2: Run the Task 2 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h2_information_assembly.py -q
  ```

  Expected: collection fails because the H2 assembly modules do not exist.

- [ ] **Step 3: Implement canonical accumulation.** For an initial Gaussian, factor its declared covariance with Cholesky, solve its small `2x2` initial-block identity to obtain the precision that is being stored as `J`, and add `P` and `P@mean` at the declared indices. This one input-normalization solve is required because the unchanged fixture declares its initial law in moments. The forbidden evaluator operation is solving a component precision against `I_6` to materialize `Sigma`; the gate records that distinction explicitly.

  For every scalar normalized conditional

  ```text
  y[target] = sum(coefficient[index] * y[index]) + offset + noise,
  noise ~ Normal(0, variance),
  ```

  build `v` with `v[target]=1` and `v[parent]=-coefficient`, then add exactly

  ```python
  precision = 1.0 / variance
  J.add_(precision * torch.outer(v, v))
  h.add_(precision * offset * v)
  ```

  The recognition assembler uses its selected model kernel followed by its selected state kernel at each time. The generative assembler uses its selected model transition followed by its selected state transition. Both use `[z0,m0,z1,m1,z2,m2]`; both symmetrize only by exact mirrored accumulation, not by repairing a nonsymmetric result after the fact.

  End each assembler with:

  ```python
  return InformationGaussian.from_information(h, J)
  ```

  and reject any result outside finite float64/SPD requirements. Do not use `torch.linalg.inv`, `torch.linalg.pinv`, `torch.cholesky_inverse`, jitter, or clipping.

- [ ] **Step 4: Run the Task 2 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h2_information_assembly.py -q
  ```

  Expected: all four recognition and all four generative components assemble directly; all tests pass.

- [ ] **Step 5: Commit Task 2.**

  ```powershell
  git add vfe4/numerics/linear_gaussian.py vfe4/numerics/__init__.py vfe4/recognition/reference_h2.py vfe4/recognition/__init__.py vfe4/generative/reference_h2.py vfe4/generative/__init__.py tests/unit/test_h2_information_assembly.py
  git commit -m "feat: assemble H2 information components directly"
  ```

---

### Task 3: Evaluate every information-form component and complete ELBO term

**Files:**

- Create: `vfe4/objective/h2_information.py`
- Modify: `vfe4/objective/__init__.py`
- Create: `tests/unit/test_h2_information_elbo.py`

**Interfaces:**

- Consumes the two Task 2 assemblers plus public H1 factors, source weights, source priors, emissions, and the existing deterministic Gauss--Hermite nodes.
- Produces immutable `RoundingInputs(output_inf_norm, absolute_summand_accumulation_inf, spd_kappa2)`; `H2ComponentTerms`; and `H2InformationEvaluation(components, local_terms, source_entropy, weighted_component_entropy, joint_recognition_entropy, complete_elbo, rounding_inputs, component_diagnostics)`.
- Produces `evaluate_information_elbo(model: H1GenerativeModel, recognition: H1RecognitionLaw, *, quadrature_order: int) -> H2InformationEvaluation`.
- The evaluator reads `model.factors` and `recognition.factors`; it never calls either H1 `joint_component`, `evaluate_local_elbo`, or `evaluate_monolithic_elbo`.

- [ ] **Step 1: Write the focused evaluator tests.** Patch both H1 `joint_component` methods to raise, evaluate all four positive-weight paths, and assert exact source order and weights. Check the component reconstruction

  ```python
  component.complete_value == math.fsum((
      component.gaussian_log_ratio,
      component.source_log_ratio,
      *component.expected_log_emission,
  ))
  ```

  and the source-weighted aggregate. Check that `joint_recognition_entropy` equals categorical source entropy plus weighted conditional-Gaussian entropy. Check the 12 signed local scalars reconstruct `local_terms.complete_elbo`. Add unit failures for order other than 21, a zero/negative path weight, nonfinite emission output, and an instrumented request for all six inverse columns.

- [ ] **Step 2: Run the Task 3 test for RED.**

  ```powershell
  python -m pytest tests/unit/test_h2_information_elbo.py -q
  ```

  Expected: collection fails because `vfe4.objective.h2_information` does not exist.

- [ ] **Step 3: Implement the direct information evaluation.** For each path:

  1. assemble `q_info` and `p_info` directly;
  2. obtain `mu_q=q_info.mean()`;
  3. request only the small principal moment blocks needed by that calculation, one request at a time: initial `[0,1]`, each model-transition pair, each state-transition triple, and emission blocks `[2,3]` and `[4,5]`;
  4. compute `q`/`p` log normalizers, `q` entropy, and `KL(q||p)` from Task 1 operations;
  5. evaluate every scalar conditional-Gaussian KL in the existing H1 local partition from the selected `q` moments and the original normalized q/p conditional factors;
  6. evaluate selected categorical log-softmax by order-21 two-dimensional deterministic quadrature over each selected emission marginal; and
  7. record every reduction's output, sum of absolute scalar summands, and all SPD operand condition numbers.

  The aggregate must use `math.fsum` over the four source-weighted component values and over the 12 signed local terms. Use the existing `ElboTerms` only for the identical H1 local partition; store H2-only component/log-normalizer/rounding metadata in the new result records. Do not reconstruct the complete scalar anywhere else.

  The implementation must contain no call matching any of:

  ```python
  torch.linalg.inv
  torch.linalg.pinv
  torch.cholesky_inverse
  factor.solve(torch.eye(6, dtype=torch.float64))
  ```

- [ ] **Step 4: Run the Task 3 test for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_h2_information_elbo.py -q
  ```

  Expected: every component and local term reconstructs and all tests pass without a full inverse request.

- [ ] **Step 5: Commit Task 3.**

  ```powershell
  git add vfe4/objective/h2_information.py vfe4/objective/__init__.py tests/unit/test_h2_information_elbo.py
  git commit -m "feat: evaluate the H2 information ELBO"
  ```

---

### Task 4: Add the independent NumPy moment oracle and fail-closed H2 gate

**Files:**

- Create: `verification/h2_budget.py`
- Create: `verification/numpy_oracles/h2_moment.py`
- Modify: `verification/numpy_oracles/__init__.py`
- Create: `verification/h2_gate.py`
- Modify: `vfe4/types/results.py`
- Create: `tests/oracle/test_h2_numpy_oracle.py`
- Create: `tests/promotion/test_h2_gate.py`
- Create: `docs/preregistrations/2026-07-21-h2-information-moment.md`

**Interfaces:**

- The NumPy oracle exposes `evaluate_h2_moment_oracle(fixture_path: Path, *, quadrature_order: int=21) -> H2MomentOracleEvaluation`. It imports no `torch`, `vfe4`, or existing H1 oracle helper.
- `verification.h2_budget` exposes `gamma_n`, `path_allowance`, `backward_residual_allowance`, `complete_elbo_allowance`, and `pair_allowance` with scalar/array infinity norms and no relative tolerance.
- Generalize the existing immutable `GateResult.gate` field from `Literal["H1"]` to `Literal["H1","H2"]`; all status/invariant/obligation validation remains unchanged. H2 uses this shared result type rather than introducing an untyped parallel result.
- `verification.h2_gate` exposes `H2GateEvaluation(result: GateResult, fixture_observed_sha256, information, moment, oracle, comparisons, negative_controls)`, `evaluate_h2(config: ResolvedConfig) -> H2GateEvaluation`, and `h2_validation_payload(evaluation) -> dict[str,object]`. It does not publish a run; Task 5 owns publication. Task 5 extends `evaluate_h2` with an optional immutable fixture-byte snapshot so the unified runner can capture the fixture once.

- [ ] **Step 1: Write and freeze the preregistration before running the promotion calculation.** Copy the exact global constraints above, all four positive source weights, all eight observed conditioning facts, the broader rejection envelope, every invariant family, the four and only four negative controls, the `1e-3*scale` decisiveness rule, and the H2 nonclaims. State explicitly that the fixture data were not selected or altered using H2 residuals.

- [ ] **Step 2: Write the oracle and promotion tests.** The oracle test checks independent JSON parsing, NumPy-only imports, all four dense affine-moment components, law conversions, selected emission marginals, log normalizers, entropies, oriented KLs, local terms, component values, and complete ELBO. The promotion test requires exactly these comparisons:

  - direct-information versus unchanged H1 moment component means and selected blocks for all eight `q/p` components;
  - `J@mu-h` and `J@Sigma[:,B]-E_B` backward residuals;
  - information versus NumPy log normalizer and entropy;
  - information versus H1 and NumPy oriented `KL(q||p)`/Gaussian log ratio;
  - information versus H1 and NumPy for every component emission, source ratio, Gaussian contribution, component value, every aggregate local contribution, joint recognition entropy, and complete ELBO;
  - all eight condition-envelope/Cholesky-pivot records; and
  - exactly four decisive negative controls.

  Assert the invariant-name tuple exactly so controls cannot silently disappear. Construct one valid H2 `GateResult` and verify that the unchanged generic status rules reject inconsistent PASS, FAIL, and INCONCLUSIVE records for both H1 and H2.

- [ ] **Step 3: Run the Task 4 tests for RED.**

  ```powershell
  python -m pytest tests/oracle/test_h2_numpy_oracle.py tests/promotion/test_h2_gate.py -q
  ```

  Expected: collection fails because the oracle, budget, and H2 gate modules do not exist.

- [ ] **Step 4: Implement the independent dense moment oracle.** Parse `h1_v1.json` independently, assemble each `q` and `p` component from affine transforms and innovation covariances, and use `numpy.linalg.cholesky`/`numpy.linalg.solve`. Dense `6x6` covariance is allowed only here. Use `numpy.polynomial.hermite_e.hermegauss` independently for the order-21 emission calculation. Record every output's absolute summand accumulation and SPD condition numbers. Never import or call production assembly, H1 oracle assembly, or PyTorch.

- [ ] **Step 5: Implement the frozen budget literally.** The module's core is:

  ```python
  EPS = float(np.finfo(np.float64).eps)
  C = 256.0

  def gamma_n(n: int) -> float:
      numerator = n * EPS
      if n <= 0 or numerator >= 1.0:
          raise ValueError("n must be positive and n*eps must be below one")
      return numerator / (1.0 - numerator)

  def operation_count(dimension: int) -> int:
      return 8 * dimension + 32

  def path_allowance(dimension, kappas, output_inf, absolute_sum_inf):
      return C * gamma_n(operation_count(dimension)) * max(1.0, *kappas) * max(
          1.0, output_inf, absolute_sum_inf
      )

  def backward_residual_allowance(dimension, matrix_inf, solution_inf, rhs_inf):
      return C * gamma_n(operation_count(dimension)) * max(
          1.0, matrix_inf * solution_inf + rhs_inf
      )
  ```

  `complete_elbo_allowance` rejects any term/allowance length other than 12. `pair_allowance` adds the two local allowances plus the `C*gamma_n(D+2)` comparison rounding term. Every function rejects nonfinite/negative inputs.

- [ ] **Step 6: Implement the gate and controls.** Use ordinary `<=` against the computed invariant-specific allowance; do not call `allclose`. Set status to `FAIL` for a finite exceeded invariant, `INCONCLUSIVE` for missing evidence, an indecisive wrong-path residual, fixture/hash mismatch, factor failure, or inability to instrument the inverse path.

  The mathematical controls deliberately recompute the affected invariant with `(h,J)` misread as `(mu,J)`, the determinant ratio reversed, and `J_ii^{-1}` substituted for the Schur/marginal emission covariance. For each, require

  ```python
  wrong_residual >= 1.0e-3 * max(1.0, abs(correct_value), abs(wrong_value))
  ```

  The inverse control wraps every production factor, records `solve` RHS widths and `selected_inverse` column sets, and raises on width `D` or all `D` columns. During the production information evaluation, patch `torch.linalg.inv`, `torch.linalg.pinv`, and `torch.cholesky_inverse` to raise. Passing requires zero forbidden attempts during the real evaluation and successful detection of one deliberately injected `solve(I_D)` attempt.

- [ ] **Step 7: Run the Task 4 tests for GREEN.**

  ```powershell
  python -m pytest tests/oracle/test_h2_numpy_oracle.py tests/promotion/test_h2_gate.py -q
  ```

  Expected: both files pass; H2 is `PASS`; all named invariants pass; all four controls are decisive/detected; no random sweep or performance test runs.

- [ ] **Step 8: Commit Task 4.**

  ```powershell
  git add verification/h2_budget.py verification/numpy_oracles/h2_moment.py verification/numpy_oracles/__init__.py verification/h2_gate.py vfe4/types/results.py tests/oracle/test_h2_numpy_oracle.py tests/promotion/test_h2_gate.py docs/preregistrations/2026-07-21-h2-information-moment.md
  git commit -m "test: add the H2 information moment gate"
  ```

---

### Task 5: Publish ordered H1/H2 results through the single click-run surface

**Files:**

- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `verification/h1_gate.py`
- Create: `verification/run_gates.py`
- Modify: `verify_vfe4.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `tests/promotion/test_h1_gate.py`
- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h2-information-moment.md`

**Interfaces and compatibility:**

- `ValidationConfig.gates` accepts only the ordered implemented prefixes `("H1",)` and `("H1","H2")`; the editable launcher dictionary advances to `"gates": ["H1","H2"]`. Empty, H2-without-H1, reversed, duplicate, or unknown gates fail closed. The H1-only prefix exists solely for the backward-compatible `run_h1` wrapper and its focused tests, so a single-H1 artifact never claims that H2 was requested.
- Add `H1GateEvaluation(result, validation_payload, fixture_observed_sha256)` and `evaluate_h1(config, *, fixture_bytes: bytes | None = None)` to `verification/h1_gate.py`. Give `evaluate_h2` the identical optional keyword. `None` preserves standalone behavior; supplied bytes are hashed and evaluated without rereading the live fixture. Keep `run_h1(config)` as a backward-compatible single-gate wrapper implemented by calling `evaluate_h1`; do not duplicate H1 math.
- Add `VerificationRunResult(gate_results, run_directory)` and `run_verification(config)` in `verification/run_gates.py`. Its private `_config_payload` reads `config.canonical_json` and adds `config_sha256`; its private `_combined_provenance` calls the existing provenance builder and then adds the ordered gate metadata.
- `verify_vfe4.main(config=CONFIG) -> VerificationRunResult` prints ordered H1 and H2 statuses and one artifact path. `_script_main` returns zero only when both are `PASS`.

- [ ] **Step 1: Update focused integration/config tests.** Assert import remains side-effect free, `CONFIG` resolves to exactly `("H1","H2")`, the H1-only ordered prefix still resolves for `run_h1`, invalid/non-prefix gate lists fail, one `main()` call evaluates both gates once, and one manifest-checked run contains `config.json`, `provenance.json`, `environment.json`, `validation/h1.json`, and `validation/h2.json`. Update the H1 promotion helper to request only `("H1",)` and preserve its single-gate provenance assertions. Preserve the existing malicious/outside/repository-control `run_root` cases already in these files; do not add a new Windows path-edge matrix.

- [ ] **Step 2: Run the Task 5 tests for RED.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/integration/test_verify_vfe4.py tests/promotion/test_h1_gate.py -q
  ```

  Expected: failures show the configuration still permits only H1 and the launcher publishes only `validation/h1.json`.

- [ ] **Step 3: Split H1 evaluation from publication without changing H1 math.** Move only the capture/revalidation/try-except portion of `run_h1` into `evaluate_h1`; leave `_evaluate`, invariants, controls, and payload construction byte-for-byte behaviorally unchanged. `run_h1` calls the new function and retains its existing single-H1 publication contract for direct callers.

- [ ] **Step 4: Implement the atomic ordered runner.** `run_verification` must:

  ```python
  fixture_bytes = FIXTURE_PATH.read_bytes()
  h1 = evaluate_h1(config, fixture_bytes=fixture_bytes)
  h2 = evaluate_h2(config, fixture_bytes=fixture_bytes)
  results = (h1.result, h2.result)
  payloads = {
      "config.json": config_payload(config),
      "provenance.json": combined_provenance(config, h1, h2, started, ended),
      "environment.json": build_environment(config),
      "validation/h1.json": h1.validation_payload,
      "validation/h2.json": h2_validation_payload(h2),
  }
  run_dir = publish_run_directory(config.artifacts.run_root, run_name, payloads)
  return VerificationRunResult(results, run_dir)
  ```

  Require `config.validation.gates == ("H1", "H2")` before evaluation. Both gates consume the same captured `h1_v1.json` bytes and must report the same observed hash. Build combined provenance by calling the existing `build_provenance` security path once with the aggregate state (`pass` only if both pass; `fail` if either fails; otherwise `inconclusive`), then add ordered `gate_states={"H1": h1.result.status.value, "H2": h2.result.status.value}` and `fixture_consumers=("H1","H2")`. Do not duplicate or weaken `dirty_content_digest`, repository-control exclusions, path containment, atomic replacement, or manifest validation.

- [ ] **Step 5: Update the launcher and documentation.** Change the module docstring to H1/H2, use only `run_verification`, and keep the one editable `CONFIG`. README and preregistration must state that H2 is componentwise representation verification over the unchanged source mixture, quote the frozen envelope/budget, distinguish `(mu,Sigma)` from expectation coordinates, and leave H3--H8 unimplemented.

- [ ] **Step 6: Run the Task 5 tests for GREEN.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/integration/test_verify_vfe4.py tests/promotion/test_h1_gate.py -q
  ```

  Expected: both files pass; no second launcher/config dictionary exists; one mocked click-run publishes both validation payloads atomically.

- [ ] **Step 7: Commit Task 5.**

  ```powershell
  git add vfe4/config/schema.py vfe4/config/resolve.py verification/h1_gate.py verification/run_gates.py verify_vfe4.py tests/unit/test_config.py tests/integration/test_verify_vfe4.py tests/promotion/test_h1_gate.py README.md docs/preregistrations/2026-07-21-h2-information-moment.md
  git commit -m "feat: publish ordered H1 and H2 verification"
  ```

---

### Task 6: Produce the one exact-revision milestone record

**Files:**

- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h2-information-moment.md`
- Verification-skill claim ledger: start and validate it through the installed skill tooling and populate the schema-conformant JSON with `apply_patch` because the installed tool exposes no claim-add command; do not commit `.verification`.

**Evidence policy:** This task changes documentation only after evidence is captured at the exact Task 5 commit. Any source/test/config change invalidates that evidence and returns work to the affected task's focused RED/GREEN cycle before creating a new milestone candidate. Reviewers consume the JUnit XML, click-run artifact, manifest, and claim ledger; they do not rerun identical tests.

- [ ] **Step 1: Record the exact candidate revision and confirm the tracked tree is clean.**

  ```powershell
  git rev-parse HEAD
  git status --short
  ```

  Expected: a 40-character commit ID and no tracked modifications. Generated `runs/` or verification-control state may be ignored/untracked according to repository policy.

- [ ] **Step 2: Run the milestone full regression at that exact revision and write machine-readable evidence.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h2-milestone.xml
  ```

  Expected: pytest exits zero. Read totals only from `C:\tmp\vfe4-h2-milestone.xml`; do not report a visual progress count or a remembered H1 total. Do not repeat this run unless a subsequent review requires a source/test/config change, in which case discard this evidence and replace it once at the new exact revision.

- [ ] **Step 3: Run the one click-run and inspect the atomic artifact.**

  ```powershell
  python verify_vfe4.py
  ```

  Expected: the launcher prints `H1: pass`, `H2: pass`, and one run directory. Independently recompute the manifest hashes and verify that its `provenance.json` records the exact candidate revision/config/fixture identity and ordered gate states, and that both `validation/h1.json` and `validation/h2.json` exist.

- [ ] **Step 4: Have fresh reviewers inspect existing evidence only.** One reviewer checks spec/plan coverage and dependency direction; one checks canonical assembly/KL signs and mixture semantics; one checks artifact/config/run-root behavior. They cite the implementer's focused command outputs, milestone JUnit XML, and click artifact. Resolve any Critical/Important issue by returning to the owning task; do not authorize reviewers to rerun the same suite.

- [ ] **Step 5: Populate and validate the evidence-gated claim ledger.** Use one claim per check: direct assembly independence, no covariance/inverse in production evaluation, componentwise law conversions, selected marginal correctness, entropy/log-normalizer/oriented-KL equality, all component/local/complete-ELBO comparisons, envelope metadata, decisive controls, ordered click-run publication, and JUnit totals. Close only with current eligible evidence at the candidate revision; otherwise use `INCONCLUSIVE`.

- [ ] **Step 6: Update documentation with measured facts from that evidence.** In both files record the exact tested source revision, JUnit totals parsed from XML, artifact directory, fixture hash, maximum residual only alongside the allowance belonging to that same invariant, eight-component conditioning extrema, negative-control residuals/detection result, and reviewer disposition. State that the final documentation commit is a documentation-only child of the tested source revision and that H3--H8 remain unimplemented.

- [ ] **Step 7: Check documentation and commit the closure record without rerunning tests.**

  ```powershell
  rg -n "allclose|rtol|H3.*pass|H4.*pass|H5.*pass|H7.*pass|H8.*pass" README.md docs/preregistrations/2026-07-21-h2-information-moment.md
  git diff --check
  git add README.md docs/preregistrations/2026-07-21-h2-information-moment.md
  git commit -m "docs: record H2 representation verification"
  ```

  Expected: `rg` finds no placeholder, blanket tolerance, or later-gate pass claim; `git diff --check` exits zero; the commit contains documentation only. Do not rerun the full suite after this documentation-only commit, and report the tested parent revision explicitly.

## Out of Scope for This Plan

- Any source-mixture moment projection or claim that the source-marginal law has one global precision.
- Training, datasets, WikiText-103, prediction, checkpoints, optimizers, gradients, parameter updates, or figures.
- Random stress sweeps, performance benchmarks, allocation-cost claims, or new Windows path-edge inventories.
- H3 posterior adequacy, H4 information-form cost, H5 update coherence, H6 prefix/predictive work, H7 frame covariance, H8 sparse scale, and any post-H8 experiment or Research-vault ingestion.
- A sparse production backend beyond the approved factor protocol; H2's dense `6x6` stored precision is a bounded reference implementation, not H8 evidence.

After this plan closes, the next implementation plan must begin from the exact H1/H2 artifact family and may not weaken this gate's componentwise mixture semantics, factor seam, or frozen error contract.
