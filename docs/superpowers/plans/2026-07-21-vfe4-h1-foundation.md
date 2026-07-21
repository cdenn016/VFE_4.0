# VFE 4.0 H1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable VFE 4.0 increment: a strict typed/configured package, a frozen `T=2`, `d_z=d_m=1`, vocabulary-3 reference model, three independently assembled H1 ELBO identities, a fail-closed promotion gate, revision-bound artifacts, and a root-level click-to-run verifier.

**Architecture:** The production path is a root `vfe4` package using PyTorch float64 tensors; the independent oracle is a root `verification` package using NumPy and separately written quadrature/factor assembly. Both consume the same versioned data-only fixture, but production code cannot import oracle code and the oracle cannot import production numerical helpers. `verify_vfe4.py` resolves one editable dictionary and calls public package APIs. This increment keeps frames deterministic and performs probability/ELBO evaluation only: it does not add training, learned frames, an autograd update, H2+, or a misleading `train_vfe4.py` placeholder.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, frozen dataclasses, pytest with JUnit XML, standard-library JSON/hash/platform/path/file APIs.

## Global Constraints

- The normative semantics are `Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex` and `Manuscripts/vfe4_whitepaper/`; MAgent supplies independent-oracle patterns only, and V3 supplies launcher/artifact ergonomics only.
- The user surface is click-to-run. `verify_vfe4.py` may not use `argparse`, Typer, Hydra, required environment variables, or required shell flags. Developer test commands below do not change that user contract.
- Use red-green-refactor for every task. Observe the named test fail for the expected missing behavior before writing production code.
- Ellipses in the interface snippets below mean "signature specified here" only. No committed implementation may contain an ellipsis body, `NotImplementedError`, skipped test, or placeholder return.
- Keep all normative probability calculations in float64. Validate shapes, finite values, normalization, positive source support, and SPD matrices at boundaries.
- The H1 continuous stack is exactly `y = [z0, m0, z1, m1, z2, m2]`. The observations are `x = [x1, x2]`; there is no `x0` emission.
- The model slice order is `b_t -> m_t -> a_t -> z_t -> x_t`. At `t=1`, both sources are fixed at parent 0. At `t=2`, both parents 0 and 1 are admitted.
- The generative transitions are the whitepaper kernels
  `m_t ~ N((U_t/U_j) m_j + c_m[t], R_m[t])` and
  `z_t ~ N((U_t/U_j) z_j + B[t] m_t + c_z[t], R_z[t])`.
- Frames are fixed deterministic structure in H1. There is no `phi`, `g_phi`, frame optimizer, or covariance claim in this increment.
- Production monolithic ELBO, production local decomposition, and independent evidence/posterior-KL must be separate calculation paths. A helper that collapses them to the same arithmetic invalidates H1.
- Dense `6x6` covariance is allowed only in this bounded H1 fixture and independent oracle. Do not generalize it into the future promoted sparse interface.
- `ElboTerms.complete_elbo` is assembled in exactly one production location. Reporting and promotion code consume that value and do not reconstruct an objective.
- The data-only fixture may be shared. Mathematical helpers, log-density routines, component assembly, source enumeration, and quadrature code may not be shared across production and independent oracle paths.
- Dependencies remain one-way: `verification/` may import public `vfe4` interfaces, but no `vfe4.*` module may import `verification.*`. Cross-boundary H1 orchestration lives in `verification/h1_gate.py`.
- Order-to-order quadrature differences are named fixture-specific convergence estimates, not proved numerical error bounds. Every comparison uses the sum of the two participating calibrated allowances plus a separate compensated-summation rounding allowance.
- `.verification/` is the agent verification control plane. The tracked root `verification/` package is product code. Never import one from the other or write product artifacts into `.verification/`.
- Use `pytest --junitxml=<path>` for canonical machine-readable test totals. Do not create a custom JUnit serializer.
- Each task is one commit after its focused tests pass. Do not bundle later tasks into an earlier commit.

---

### Task 1: Bootstrap the package and strict H1 configuration boundary

**Files:**

- Create: `pyproject.toml`
- Create: `README.md`
- Create: `vfe4/__init__.py`
- Create: `vfe4/config/__init__.py`
- Create: `vfe4/config/schema.py`
- Create: `vfe4/config/resolve.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_config.py`

**Public contract:**

```python
@dataclass(frozen=True)
class RunConfig:
    mode: Literal["verify"]
    seed: int
    device: Literal["cpu"]
    dtype: Literal["float64"]
    deterministic: bool

@dataclass(frozen=True)
class ValidationConfig:
    gates: tuple[Literal["H1"]]
    fixture_id: Literal["h1-v1"]
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    maximum_convergence_estimate: float  # resolver requires exactly 1e-9

@dataclass(frozen=True)
class DataConfig:
    kind: Literal["frozen_fixture"]
    identity: Literal["h1-v1"]

@dataclass(frozen=True)
class ModelConfig:
    horizon: Literal[2]
    d_z: Literal[1]
    d_m: Literal[1]
    vocabulary_size: Literal[3]
    state_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    model_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    state_source_support: tuple[tuple[int, ...], tuple[int, ...]]
    model_source_support: tuple[tuple[int, ...], tuple[int, ...]]
    geometry: Literal["fixed_population_frames"]

@dataclass(frozen=True)
class RecognitionConfig:
    conditioning: Literal["smoothing"]
    family: Literal["structured_linear_gaussian_mixture"]
    source_treatment: Literal["exact_enumeration"]

@dataclass(frozen=True)
class InferenceConfig:
    operation: Literal["evaluate_only"]
    estimator: Literal["deterministic_quadrature"]

@dataclass(frozen=True)
class OptimizationConfig:
    e_like_update: Literal["none"]
    m_like_update: Literal["none"]
    expected_autograd_scope: Literal["none"]

@dataclass(frozen=True)
class ArtifactConfig:
    run_root: Path

@dataclass(frozen=True)
class ResolvedConfig:
    schema_version: Literal[1]
    objective_schema_version: Literal["vfe4-state-elbo-v1"]
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    recognition: RecognitionConfig
    inference: InferenceConfig
    optimization: OptimizationConfig
    validation: ValidationConfig
    artifacts: ArtifactConfig
    canonical_json: str
    config_sha256: str

def resolve_config(raw: Mapping[str, object], *, repo_root: Path) -> ResolvedConfig: ...
```

The initial editable mapping is exactly:

```python
{
    "schema_version": 1,
    "objective_schema_version": "vfe4-state-elbo-v1",
    "run": {
        "mode": "verify",
        "seed": 20260721,
        "device": "cpu",
        "dtype": "float64",
        "deterministic": True,
    },
    "data": {"kind": "frozen_fixture", "identity": "h1-v1"},
    "model": {
        "horizon": 2,
        "d_z": 1,
        "d_m": 1,
        "vocabulary_size": 3,
        "state_parent_sets": [[0], [0, 1]],
        "model_parent_sets": [[0], [0, 1]],
        "state_source_support": [[0], [0, 1]],
        "model_source_support": [[0], [0, 1]],
        "geometry": "fixed_population_frames",
    },
    "recognition": {
        "conditioning": "smoothing",
        "family": "structured_linear_gaussian_mixture",
        "source_treatment": "exact_enumeration",
    },
    "inference": {
        "operation": "evaluate_only",
        "estimator": "deterministic_quadrature",
    },
    "optimization": {
        "e_like_update": "none",
        "m_like_update": "none",
        "expected_autograd_scope": "none",
    },
    "validation": {
        "gates": ["H1"],
        "fixture_id": "h1-v1",
        "quadrature_order": 21,
        "convergence_check_order": 17,
        "maximum_convergence_estimate": 1e-9,
    },
    "artifacts": {"run_root": "runs"},
}
```

`resolve_config` must recursively reject unknown or missing keys, booleans supplied where integers are required, any gate list other than exactly `["H1"]` (including empty or duplicate lists), a non-CPU device, a non-float64 dtype, or any value that differs from the frozen structural/recognition/evaluation literals above. Resolve relative `run_root` against `repo_root`; canonicalize paths with forward slashes before sorted compact JSON serialization; hash that exact serialization, excluding the derived `canonical_json` and `config_sha256` fields themselves. Never mutate `raw`.

- [ ] **Step 1: Write the failing configuration tests.** Cover successful resolution, frozen nested records, stable hash under mapping-key reordering, input nonmutation, relative-path resolution, and parametrized rejection of unknown keys/invalid values.
- [ ] **Step 2: Run RED.**

  ```powershell
  python -m pytest tests/unit/test_config.py -q --junitxml=C:\tmp\vfe4-task1-red.xml
  ```

  Expected: collection fails because `vfe4.config` does not exist.

- [ ] **Step 3: Implement the minimal package and resolver.** Use explicit allowed-key sets at each mapping level and small typed extraction helpers; do not use permissive `**kwargs` construction. Set `requires-python = ">=3.10"`, declare `torch` and `numpy` runtime dependencies, and configure pytest test paths in `pyproject.toml`.
- [ ] **Step 4: Run GREEN and package compilation.**

  ```powershell
  python -m pytest tests/unit/test_config.py -q --junitxml=C:\tmp\vfe4-task1.xml
  python -m compileall -q vfe4 tests
  ```

  Expected: all configuration tests pass; compilation exits zero.

- [ ] **Step 5: Document the scope.** `README.md` must say H1 is the only implemented gate, show click-to-run as the intended surface, state that no training path exists yet, and avoid claims that VFE 4.0 is backpropagation-free.
- [ ] **Step 6: Commit.**

  ```powershell
  git add pyproject.toml README.md vfe4 tests
  git commit -m "build: bootstrap VFE 4.0 configuration core"
  ```

---

### Task 2: Add immutable mathematical types and fail-closed numerics

**Files:**

- Create: `vfe4/types/__init__.py`
- Create: `vfe4/types/structural.py`
- Create: `vfe4/types/results.py`
- Create: `vfe4/numerics/__init__.py`
- Create: `vfe4/numerics/categorical.py`
- Create: `vfe4/numerics/gaussian.py`
- Create: `vfe4/numerics/quadrature.py`
- Create: `tests/unit/test_structural_types.py`
- Create: `tests/unit/test_categorical_numerics.py`
- Create: `tests/unit/test_gaussian_numerics.py`
- Create: `tests/unit/test_quadrature.py`

**Public contract:**

```python
@dataclass(frozen=True)
class StructuralData:
    horizon: Literal[2]
    d_z: Literal[1]
    d_m: Literal[1]
    vocabulary_size: Literal[3]
    state_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    model_parent_sets: tuple[tuple[int, ...], tuple[int, ...]]
    state_source_support: tuple[tuple[int, ...], tuple[int, ...]]
    model_source_support: tuple[tuple[int, ...], tuple[int, ...]]

@dataclass(frozen=True)
class PopulationFrames:
    values: torch.Tensor  # shape (3,), finite and nonzero

    def omega(self, receiver: int, source: int) -> torch.Tensor: ...

@dataclass(frozen=True)
class SourcePath:
    a: tuple[int, int]
    b: tuple[int, int]

@dataclass(frozen=True)
class NumericalAllowance:
    convergence_estimate: float
    rounding_allowance: float

    @property
    def total(self) -> float:
        return self.convergence_estimate + self.rounding_allowance

@dataclass(frozen=True)
class ElboTermAllowances:
    expected_log_emission: tuple[NumericalAllowance, NumericalAllowance]
    initial_model_kl: NumericalAllowance
    initial_state_kl: NumericalAllowance
    model_source_kl: tuple[NumericalAllowance, NumericalAllowance]
    model_transition_kl: tuple[NumericalAllowance, NumericalAllowance]
    state_source_kl: tuple[NumericalAllowance, NumericalAllowance]
    state_transition_kl: tuple[NumericalAllowance, NumericalAllowance]
    joint_recognition_entropy: NumericalAllowance
    complete_elbo: NumericalAllowance

@dataclass(frozen=True)
class ElboTerms:
    expected_log_emission: tuple[float, float]
    initial_model_kl: float
    initial_state_kl: float
    model_source_kl: tuple[float, float]
    model_transition_kl: tuple[float, float]
    state_source_kl: tuple[float, float]
    state_transition_kl: tuple[float, float]
    joint_recognition_entropy: float
    allowances: ElboTermAllowances
    complete_elbo: float

@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    value: float | None
    limit: float | None
    detail: str

class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

@dataclass(frozen=True)
class GateResult:
    gate: Literal["H1"]
    status: GateStatus
    fixture_id: Literal["h1-v1"]
    residual: float | None
    calibrated_allowance: float | None
    measurements: Mapping[str, float | None]
    invariants: tuple[InvariantResult, ...]
    obligations: tuple[str, ...]
```

Numerical functions:

```python
def require_probability_vector(value: Tensor, *, name: str) -> Tensor: ...
def categorical_kl(q: Tensor, p: Tensor, *, name: str) -> Tensor: ...
def selected_log_softmax(logits: Tensor, index: int) -> Tensor: ...
def require_spd(matrix: Tensor, *, name: str) -> Tensor: ...
def gaussian_log_prob(value: Tensor, mean: Tensor, covariance: Tensor) -> Tensor: ...
def probabilists_gauss_hermite(order: int, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]: ...
```

`PopulationFrames.omega(t,j)` returns `U[t] / U[j]` for this scalar fixture after index checks; it must not imply H7 covariance. `categorical_kl` permits `q_i=0`, requires `p_i>0` wherever `q_i>0`, and evaluates log ratios only on that support. `require_spd` uses `torch.linalg.cholesky_ex` and reports the named matrix. `gaussian_log_prob` uses Cholesky solve and diagonal log determinant, never an explicit inverse. Quadrature weights must integrate a standard-normal expectation and satisfy `sum(weights) == 1` within a float64 rounding bound.

`ElboTerms.__post_init__` checks all fields are finite and verifies, using a fresh `256 * eps * max(1, absolute term sum)` arithmetic allowance rather than the broader quadrature allowance, that

```text
complete_elbo
= sum(expected_log_emission)
  - initial_model_kl
  - initial_state_kl
  - sum(model_source_kl)
  - sum(model_transition_kl)
  - sum(state_source_kl)
  - sum(state_transition_kl).
```

The entropy field is required diagnostic metadata and is not added a second time because it is already partitioned inside the KL terms. Analytic terms have zero convergence estimate and their own weighted-operation rounding allowance; each emission and the complete scalar has its own order-17/order-21 convergence estimate and rounding allowance. No complete-objective allowance is reused for a term comparison. `GateResult` defensively copies its mappings into immutable views. `INCONCLUSIVE` requires at least one obligation and permits unavailable measurements; `PASS`/`FAIL` require finite residual, allowance, and measurements.

- [ ] **Step 1: Write failing unit/property tests.** Include malformed shapes, out-of-range parents, zero/singular frames, nonnormalized/negative probabilities, `q` mass outside `p` support, non-SPD matrices, Gaussian normalizer agreement with `torch.distributions.MultivariateNormal`, quadrature exactness on moments through degree six, and a deliberately inconsistent `ElboTerms` total.
- [ ] **Step 2: Run RED.**

  ```powershell
  python -m pytest tests/unit/test_structural_types.py tests/unit/test_categorical_numerics.py tests/unit/test_gaussian_numerics.py tests/unit/test_quadrature.py -q --junitxml=C:\tmp\vfe4-task2-red.xml
  ```

  Expected: collection fails on missing `vfe4.types`/`vfe4.numerics` modules.

- [ ] **Step 3: Implement the minimal immutable records and numerical functions.** Clone/detach tensors on construction so frozen records cannot be changed through caller aliases. Return tensors on the requested dtype/device and reject implicit float32 inputs on the normative path.
- [ ] **Step 4: Run GREEN and the cumulative unit suite.**

  ```powershell
  python -m pytest tests/unit -q --junitxml=C:\tmp\vfe4-task2.xml
  ```

  Expected: all Task 1-2 unit tests pass.

- [ ] **Step 5: Commit.**

  ```powershell
  git add vfe4/types vfe4/numerics tests/unit
  git commit -m "feat: add H1 mathematical types and numerics"
  ```

---

### Task 3: Freeze and implement the normalized H1 model and recognition law

**Files:**

- Create: `docs/preregistrations/2026-07-21-h1-reference-fixture.md`
- Create: `vfe4/types/h1.py`
- Create: `vfe4/generative/__init__.py`
- Create: `vfe4/generative/reference_h1.py`
- Create: `vfe4/recognition/__init__.py`
- Create: `vfe4/recognition/reference_h1.py`
- Create: `vfe4/validation/__init__.py`
- Create: `vfe4/validation/h1_fixture.py`
- Create: `vfe4/validation/fixtures/h1_v1.json`
- Create: `tests/unit/test_h1_fixture.py`
- Create: `tests/unit/test_h1_generative.py`
- Create: `tests/unit/test_h1_recognition.py`

**Frozen `h1-v1` fixture:**

```json
{
  "fixture_schema_version": 1,
  "fixture_id": "h1-v1",
  "continuous_order": ["z0", "m0", "z1", "m1", "z2", "m2"],
  "vocabulary_labels": [1, 2, 3],
  "observation_label_base": 1,
  "observation_labels": [1, 2],
  "frames": [1.0, 1.25, 0.8],
  "initial_joint": {
    "mean": [0.2, -0.15],
    "covariance": [[0.8, 0.18], [0.18, 0.65]]
  },
  "model_source_priors": [[1.0], [0.35, 0.65]],
  "state_source_priors": [[1.0], [0.55, 0.45]],
  "model_offsets": [0.1, -0.05],
  "model_variances": [0.42, 0.55],
  "state_offsets": [-0.12, 0.08],
  "state_variances": [0.37, 0.48],
  "state_model_slopes": [0.45, -0.35],
  "decoder": [
    {"w_z": [0.2, -0.4, 0.1], "w_m": [0.3, 0.2, -0.5], "bias": [0.05, -0.1, 0.15]},
    {"w_z": [-0.1, 0.35, -0.25], "w_m": [0.25, -0.2, 0.15], "bias": [-0.05, 0.12, -0.07]}
  ],
  "recognition": {
    "initial_mean": [-0.1, 0.25],
    "initial_covariance": [[0.65, 0.11], [0.11, 0.78]],
    "model_source_probabilities": [[1.0], [0.4, 0.6]],
    "state_source_probabilities_given_model_source": [
      [[1.0]],
      [[0.75, 0.25], [0.2, 0.8]]
    ],
    "model_kernels": [
      [{"slope": 0.9, "offset": -0.05, "variance": 0.58}],
      [
        {"slope": 0.7, "offset": 0.06, "variance": 0.6},
        {"slope": 0.95, "offset": -0.09, "variance": 0.5}
      ]
    ],
    "state_kernels": [
      [{"z_slope": 0.75, "m_slope": 0.25, "offset": 0.08, "variance": 0.52}],
      [
        {"a": 0, "b": 0, "z_slope": 0.65, "m_slope": 0.35, "offset": -0.02, "variance": 0.55},
        {"a": 1, "b": 0, "z_slope": 0.85, "m_slope": 0.3, "offset": 0.04, "variance": 0.48},
        {"a": 0, "b": 1, "z_slope": 0.55, "m_slope": 0.4, "offset": -0.07, "variance": 0.58},
        {"a": 1, "b": 1, "z_slope": 0.8, "m_slope": 0.2, "offset": 0.05, "variance": 0.46}
      ]
    ]
  },
  "quadrature": {
    "order": 21,
    "convergence_check_order": 17,
    "maximum_convergence_estimate": 1e-9
  }
}
```

The preregistration explains the source hierarchy, formulas, source ordering, three H1 calculations, calibrated-allowance rule, failure injections, and nonclaims. Record that observations use the manuscript labels `{1,2,3}` and cross into tensor indices only through a checked `label_to_index(label) = label - 1`; label 1 selects decoder row 0, label 2 selects row 1, and labels 0 and 4 fail. Record that the four `(a2,b2)` paths are ordered `(0,0),(1,0),(0,1),(1,1)` and that recognition path weights are `(0.30,0.10,0.12,0.48)`.

**Production interfaces:**

```python
def load_h1_fixture(path: Path | None = None) -> H1Fixture: ...
def enumerate_source_paths(fixture: H1Fixture) -> tuple[SourcePath, ...]: ...
def label_to_index(label: int, *, vocabulary_size: int = 3) -> int: ...

@dataclass(frozen=True)
class GaussianLaw:
    mean: Tensor
    covariance: Tensor

@dataclass(frozen=True)
class H1Fixture:
    fixture_schema_version: Literal[1]
    fixture_id: Literal["h1-v1"]
    structural: StructuralData
    frames: PopulationFrames
    observation_labels: tuple[int, int]
    initial_joint: GaussianLaw
    model_source_priors: tuple[Tensor, Tensor]
    state_source_priors: tuple[Tensor, Tensor]
    model_transitions: tuple[ModelTransitionRecord, ModelTransitionRecord]
    state_transitions: tuple[StateTransitionRecord, StateTransitionRecord]
    emissions: tuple[EmissionRecord, EmissionRecord]
    recognition: RecognitionParameterRecord
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    maximum_convergence_estimate: float  # loader requires exactly 1e-9 for h1-v1

class H1GenerativeModel:
    @classmethod
    def from_fixture(cls, fixture: H1Fixture) -> "H1GenerativeModel": ...

    @property
    def factors(self) -> H1GenerativeFactorRecord: ...
    def source_log_prob(self, path: SourcePath) -> Tensor: ...
    def log_joint(self, y: Tensor, path: SourcePath) -> Tensor: ...
    def joint_component(self, path: SourcePath) -> GaussianLaw: ...
    def emission_log_prob(self, y: Tensor, observations: tuple[int, int]) -> Tensor: ...

class H1RecognitionLaw:
    @classmethod
    def from_fixture(cls, fixture: H1Fixture) -> "H1RecognitionLaw": ...

    @property
    def factors(self) -> H1RecognitionFactorRecord: ...
    def source_probability(self, path: SourcePath) -> Tensor: ...
    def log_prob(self, y: Tensor, path: SourcePath) -> Tensor: ...
    def joint_component(self, path: SourcePath) -> GaussianLaw: ...
```

`ModelTransitionRecord`, `StateTransitionRecord`, `EmissionRecord`, `RecognitionParameterRecord`, `H1GenerativeFactorRecord`, and `H1RecognitionFactorRecord` are frozen, shape-checked records defined in `vfe4/types/h1.py`; tensor fields use private cloned storage and clone-returning accessors. They expose exactly the initial law, source rows, normalized scalar conditional kernels, and emission parameters required by the local evaluator without exposing mutable model internals. `log_joint` includes initial density, both source priors, both normalized transition pairs, and both selected `log_softmax` emissions. `log_prob` includes the normalized initial recognition density, recognition source probabilities, and all recognition conditional kernels. Joint components are assembled independently from each declared directed linear-Gaussian chain and checked against a separately constructed affine-noise matrix. Recognition component assembly may not call generative component assembly.

- [ ] **Step 1: Write failing fixture/model/recognition tests.** Assert the exact four paths and weights; exact `omega(t,j)=U_t/U_j`; normalized source tables; one-based label-to-index behavior and selected decoder rows; finite normalized log densities; correct factor inclusion; component means/covariances against a direct affine-noise matrix construction; and rejection of malformed JSON, zero prior support with positive recognition mass, labels 0/4, non-SPD initial covariances, and nonpositive variances.
- [ ] **Step 2: Run RED.**

  ```powershell
  python -m pytest tests/unit/test_h1_fixture.py tests/unit/test_h1_generative.py tests/unit/test_h1_recognition.py -q --junitxml=C:\tmp\vfe4-task3-red.xml
  ```

  Expected: collection fails because the H1 fixture/model modules do not exist.

- [ ] **Step 3: Add the preregistration and data-only JSON, then implement production loading and laws.** Parse every field explicitly into float64 tensors. The loader returns immutable cloned records. Configuration/fixture identity and quadrature-order compatibility are checked later at the single `verification.h1_gate.run_h1` boundary, because the fixture loader has no configuration argument.
- [ ] **Step 4: Run GREEN and cumulative units.**

  ```powershell
  python -m pytest tests/unit -q --junitxml=C:\tmp\vfe4-task3.xml
  ```

  Expected: all Task 1-3 unit tests pass.

- [ ] **Step 5: Commit.**

  ```powershell
  git add docs/preregistrations vfe4/types/h1.py vfe4/generative vfe4/recognition vfe4/validation tests/unit
  git commit -m "feat: freeze normalized H1 reference law"
  ```

---

### Task 4: Implement the two production ELBO calculations

**Files:**

- Create: `vfe4/objective/__init__.py`
- Create: `vfe4/objective/h1_monolithic.py`
- Create: `vfe4/objective/h1_local.py`
- Create: `tests/oracle/test_h1_production_identities.py`
- Create: `tests/unit/test_elbo_terms.py`

**Interfaces and separation:**

```python
@dataclass(frozen=True)
class MonolithicElboResult:
    value: float
    component_values: tuple[float, float, float, float]
    component_gaussian_log_ratios: tuple[float, float, float, float]
    component_source_log_ratios: tuple[float, float, float, float]
    expected_log_emission: tuple[float, float]
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    numerical_allowance: NumericalAllowance

def evaluate_monolithic_elbo(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> MonolithicElboResult: ...

def evaluate_local_elbo(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> ElboTerms: ...
```

The monolithic implementation enumerates paths and evaluates

```text
sum_path q(path) [
  E_q log p_complete_6d(y|path) - E_q log q_complete_6d(y|path)
  + log p(path) - log q(path)
  + E_q log L_1 + E_q log L_2
]
```

Compute the complete continuous log-ratio analytically from the two full `6x6` joint Gaussian components using Cholesky solves and log determinants. This is one complete-component Gaussian KL calculation, not a sum of local conditional KLs. Evaluate each emission expectation only on its `2x2` `(z_t,m_t)` marginal with probabilists' Gauss-Hermite nodes. This preserves a genuinely monolithic calculation while avoiding the default `13^6`/`15^6` tensor-product explosion. It may consume `joint_component`, `source_log_prob`, `source_probability`, and emissions, but it may not call either complete pointwise log-density method or any local-KL helper.

The local implementation follows the manuscript term order:

```text
sum_t E_Q log L_t
- KL(q(m0) || p(m0))
- E_q(m0) KL(q(z0|m0) || p(z0|m0))
- sum_t [KL(Q(b_t)||pi_m_t) + E KL(Q(m_t|...)||K_m_t)]
- sum_t [E KL(Q(a_t|...)||pi_z_t) + E KL(Q(z_t|...)||K_z_t)].
```

Use analytic Gaussian conditional KL expectations from public immutable factor records for all Gaussian factors. Factor each correlated initial bivariate Gaussian as `m0` followed by `z0|m0` so the result records separate initial-model and initial-state KLs whose sum equals the full initial joint KL. Only the expected `log_softmax` emissions use deterministic `2x2` marginal quadrature. Compute and record the complete joint-recognition entropy independently from source entropy plus Gaussian conditional entropies. Do not obtain a local term by subtracting the others from the monolithic scalar.

For each production evaluator, calculate the order-21 reported value and the independently repeated order-17 check value. Record

```python
convergence_estimate = abs(value_order_21 - value_order_17)
rounding_allowance = 32.0 * np.finfo(np.float64).eps * weighted_absolute_sum
numerical_allowance = NumericalAllowance(convergence_estimate, rounding_allowance)
```

Every weighted reduction uses lexicographically generated nodes and `math.fsum`; `weighted_absolute_sum` is the `math.fsum` of the absolute weighted contributions from that same evaluation. The fixture fails its convergence invariant when either production convergence estimate exceeds the preregistered `1e-9`. These are fixture-calibrated allowances, not universal forward-error proofs. The stochastic estimation budget is zero.

- [ ] **Step 1: Write failing tests.** Include monolithic/local agreement under the sum of their two allowances plus comparison roundoff, order-17/order-21 convergence, per-term finiteness and sign checks, source/transition array lengths, entropy equality with direct component entropy, complete-`6x6` versus conditional-factor Gaussian KL chain-rule agreement, and failure-injection tests proving that omitting source entropy or replacing selected `log_softmax` with the raw logit exceeds the paired calibrated allowance.
- [ ] **Step 2: Run RED.**

  ```powershell
  python -m pytest tests/unit/test_elbo_terms.py tests/oracle/test_h1_production_identities.py -q --junitxml=C:\tmp\vfe4-task4-red.xml
  ```

  Expected: collection fails because `vfe4.objective` does not exist.

- [ ] **Step 3: Implement monolithic and local paths separately.** Keep internal helpers private to their own module. If both need fixture data, consume public immutable model/recognition records rather than sharing objective arithmetic.
- [ ] **Step 4: Run GREEN and cumulative nonpromotion tests.**

  ```powershell
  python -m pytest tests/unit tests/oracle -q --junitxml=C:\tmp\vfe4-task4.xml
  ```

  Expected: all Task 1-4 tests pass.

- [ ] **Step 5: Commit.**

  ```powershell
  git add vfe4/objective tests/unit/test_elbo_terms.py tests/oracle
  git commit -m "feat: implement H1 production ELBO identities"
  ```

---

### Task 5: Add the independent NumPy evidence/posterior-KL oracle

**Files:**

- Create: `verification/__init__.py`
- Create: `verification/numpy_oracles/__init__.py`
- Create: `verification/numpy_oracles/h1_elbo.py`
- Create: `tests/oracle/test_h1_numpy_oracle.py`
- Create: `tests/property/test_h1_normalization.py`

**Independent API:**

```python
@dataclass(frozen=True)
class IndependentNumericalAllowance:
    convergence_estimate: float
    rounding_allowance: float

    @property
    def total(self) -> float:
        return self.convergence_estimate + self.rounding_allowance

@dataclass(frozen=True)
class H1EvidenceRecord:
    observation_labels: tuple[int, int]
    probability: float
    log_probability: float
    probability_allowance: IndependentNumericalAllowance
    log_probability_allowance: IndependentNumericalAllowance

@dataclass(frozen=True)
class H1IdentityRecord:
    evidence: H1EvidenceRecord
    posterior_kl: float
    elbo_from_identity: float
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    posterior_kl_allowance: IndependentNumericalAllowance
    identity_allowance: IndependentNumericalAllowance

@dataclass(frozen=True)
class IndependentTermAllowances:
    expected_log_emission: tuple[IndependentNumericalAllowance, IndependentNumericalAllowance]
    initial_model_kl: IndependentNumericalAllowance
    initial_state_kl: IndependentNumericalAllowance
    model_source_kl: tuple[IndependentNumericalAllowance, IndependentNumericalAllowance]
    model_transition_kl: tuple[IndependentNumericalAllowance, IndependentNumericalAllowance]
    state_source_kl: tuple[IndependentNumericalAllowance, IndependentNumericalAllowance]
    state_transition_kl: tuple[IndependentNumericalAllowance, IndependentNumericalAllowance]
    joint_recognition_entropy: IndependentNumericalAllowance
    complete_elbo: IndependentNumericalAllowance

@dataclass(frozen=True)
class IndependentTermRecord:
    expected_log_emission: tuple[float, float]
    initial_model_kl: float
    initial_state_kl: float
    model_source_kl: tuple[float, float]
    model_transition_kl: tuple[float, float]
    state_source_kl: tuple[float, float]
    state_transition_kl: tuple[float, float]
    joint_recognition_entropy: float
    complete_elbo: float
    allowances: IndependentTermAllowances

def h1_log_evidence(
    fixture_path: Path,
    observation_labels: tuple[int, int],
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1EvidenceRecord: ...

def h1_local_diagnostics(
    fixture_path: Path,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> IndependentTermRecord: ...

def h1_evidence_and_posterior_kl(
    fixture_path: Path,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> H1IdentityRecord: ...
```

The module imports only the standard library and NumPy. It reads and validates the JSON independently; it may not import `vfe4`. Build each of the four generative and recognition `6x6` Gaussian components using separately coded affine-noise matrices. `h1_log_evidence` and its private source enumerator accept only raw generative fixture records (initial joint, generative priors, generative transitions, frames, and decoder); no recognition record is in their call graph. Tests mutate recognition fields and require evidence to remain bitwise unchanged, then mutate generative priors/kernels and require evidence to change. Use NumPy physicists' Hermite nodes with the explicit `sqrt(2)` standard-normal scaling, not the production probabilists' implementation. Marginalize those full components to `(z_t,m_t)` for log-emission expectations and to `(z1,m1,z2,m2)` for evidence; never materialize a default six-dimensional tensor-product node grid.

Compute evidence directly:

```text
p(x) = sum_path p(path) E_p(y|path)[L_1(x1|z1,m1) L_2(x2|z2,m2)]
```

Then evaluate the posterior KL from its own Bayes-normalized density:

```text
KL(Q || p(.|x))
= sum_path q(path) E_q[
    log q(path,y) - log p(path,y,x) + log p(x)
  ].
```

Return `log_evidence - posterior_kl`; do not import or call either production ELBO evaluator. This algebraically cancels the same computed log-evidence contribution, so it independently checks the complete log-density expectation but does **not** by itself validate evidence. Evaluate order 21 and order 17 separately, use lexicographic nodes plus `math.fsum`, and record the independent convergence estimate and weighted-absolute-sum rounding allowance with the preregistered formulas. Compute the probability-domain evidence allowance from the two probability evaluations and their probability-domain weighted sums; independently compute the log-domain allowance from the two `log(probability)` evaluations and log-domain rounding scale. Never add a probability allowance to a log quantity. `h1_local_diagnostics` independently assembles every homologous local term and term-shaped allowance so the gate can compare every declared term, not only final scalars.

- [ ] **Step 1: Write failing oracle and normalization tests.** Compare independent identity ELBO with both production values under their own paired allowances; compare every independent local diagnostic with its production homolog using its term-shaped allowance; assert direct source-loop and prebuilt component-table enumeration agree; assert order-17/order-21 convergence estimates are finite and at most `1e-9`; calculate all `3^2=9` observation-pair evidences in one vectorized pass, require each probability in `(0,1]`, and require their sum to equal one within the sum of their probability-domain allowances; require `posterior_kl >= -posterior_kl_allowance` and `elbo_from_identity <= log_evidence + identity_allowance + log_probability_allowance`; assert evidence is unchanged by recognition-field mutations but changes under generative-prior/kernel mutations; require a deliberate recognition-mixture-for-generative-evidence substitution to differ beyond its probability-domain allowance; assert the exact one-based label mapping; and reject source mass outside positive prior support.
- [ ] **Step 2: Run RED.**

  ```powershell
  python -m pytest tests/oracle/test_h1_numpy_oracle.py tests/property/test_h1_normalization.py -q --junitxml=C:\tmp\vfe4-task5-red.xml
  ```

  Expected: collection fails because `verification.numpy_oracles.h1_elbo` does not exist.

- [ ] **Step 3: Implement the independent oracle.** Use `np.linalg.cholesky`, `np.linalg.solve`, and `np.linalg.slogdet`; do not use explicit matrix inverses. Keep source enumeration and log-density formulas in this module even when production has analogous code.
- [ ] **Step 4: Run GREEN and cumulative tests.**

  ```powershell
  python -m pytest tests/unit tests/oracle tests/property -q --junitxml=C:\tmp\vfe4-task5.xml
  ```

  Expected: all Task 1-5 tests pass and the JUnit file contains zero failures/errors.

- [ ] **Step 5: Commit.**

  ```powershell
  git add verification tests/oracle/test_h1_numpy_oracle.py tests/property
  git commit -m "test: add independent H1 evidence oracle"
  ```

---

### Task 6: Build the fail-closed H1 gate, atomic artifacts, and click-to-run verifier

**Files:**

- Create: `vfe4/artifacts/__init__.py`
- Create: `vfe4/artifacts/atomic.py`
- Create: `vfe4/artifacts/provenance.py`
- Create: `verification/h1_gate.py`
- Create: `verify_vfe4.py`
- Create: `tests/unit/test_atomic_artifacts.py`
- Create: `tests/promotion/test_h1_gate.py`
- Create: `tests/integration/test_verify_vfe4.py`

**Gate and launcher contract:**

```python
def run_h1(config: ResolvedConfig) -> tuple[GateResult, Path]: ...
```

`run_h1` evaluates the two production paths and independent oracle, then sets

```python
measurements = {
    "monolithic_elbo": monolithic.value,
    "local_elbo": local.complete_elbo,
    "evidence_minus_posterior_kl": identity.elbo_from_identity,
}
comparison_rounding = (
    64.0
    * np.finfo(np.float64).eps
    * max(1.0, *(abs(v) for v in measurements.values()))
)
pairwise_residuals = {
    "monolithic_vs_local": abs(monolithic.value - local.complete_elbo),
    "monolithic_vs_identity": abs(monolithic.value - identity.elbo_from_identity),
    "local_vs_identity": abs(local.complete_elbo - identity.elbo_from_identity),
}
pairwise_allowances = {
    "monolithic_vs_local": monolithic.numerical_allowance.total + local.allowances.complete_elbo.total + comparison_rounding,
    "monolithic_vs_identity": monolithic.numerical_allowance.total + identity.identity_allowance.total + comparison_rounding,
    "local_vs_identity": local.allowances.complete_elbo.total + identity.identity_allowance.total + comparison_rounding,
}
residual = max(pairwise_residuals.values())
calibrated_allowance = max(pairwise_allowances.values())  # summary fields only
term_comparisons = build_named_term_comparisons(
    local,
    independent_terms,
    # For each scalar or tuple element, use only the matching allowance from
    # local.allowances and independent_terms.allowances plus term-local roundoff.
)
```

Before calculation, this boundary checks that fixture identity/schema, dimensions, parent sets, source support, geometry/recognition tags, evaluation-only update labels, autograd scope, and quadrature orders exactly match `ResolvedConfig`. It also executes three named negative controls: omitting both source-entropy/KL contributions, replacing selected `log_softmax` values with raw selected logits, and substituting recognition mixture components/weights for the generative evidence mixture must each create a residual larger than its own paired allowance. The summary `residual` and `calibrated_allowance` are never compared to each other for the decision: every named pairwise residual must satisfy its own named allowance. Each scalar or indexed homologous term similarly has its own named residual and the sum of only its two term-shaped allowances plus term-local roundoff. Status is `PASS` only when every available measurement/allowance is finite, all three pairwise comparisons pass separately, every term comparison passes separately, both production and independent convergence estimates are at most `1e-9`, all nine evidence values and their sum pass using probability-domain allowances, posterior KL is nonnegative within its own allowance, the ELBO does not exceed log evidence using the identity plus log-probability allowances, all three negative controls are detected, and every named invariant passes. A finite disagreement is `FAIL`.

`verification.h1_gate.run_h1` is the single computation catch boundary. A failed factorization, invalid fixture, incompatible frozen configuration, or nonfinite calculation becomes and publishes an `INCONCLUSIVE` `GateResult` with unavailable floats set to `None` and a nonempty obligation. Invalid raw configuration fails before this function and creates no run. Artifact reservation/publication failures raise `ArtifactPublicationError`, print `artifact unavailable`, and exit nonzero; they cannot be represented as a successfully published result. Never turn an implementation exception into a pass.

The run directory is `runs/verify-h1-<UTC timestamp>-<config hash prefix>/` and contains:

```text
config.json
provenance.json
environment.json
validation/h1.json
manifest.sha256
```

Write every JSON file through a temporary sibling, flush and `fsync`, then `os.replace`. Build the manifest last from sorted relative POSIX paths and SHA-256 digests; exclude the manifest itself. Provenance records Git revision, dirty-state digest, canonical config/objective hashes, Python/PyTorch/NumPy versions, device/dtype, seeds, fixture file hash, start/end UTC timestamps, and gate state. Do not overwrite an existing run directory.

`validation/h1.json` contains the complete `ElboTerms`, monolithic per-path continuous/source/emission contributions, independent homologous term record, evidence/posterior-KL record, all calibrated allowances, all term and scalar residuals, and named invariant results. `verify_vfe4.py` contains the exact Task 1 mapping as an editable top-level `CONFIG`, defines `main(config: Mapping[str, object] = CONFIG) -> GateResult`, prints the artifact path and concise status, exits nonzero on fail/inconclusive when run as a script, and does no work on import.

- [ ] **Step 1: Write failing artifact, promotion, and integration tests.** Test atomic replacement cleanup after an injected write failure, stable manifest ordering/digests, required provenance keys, config/fixture compatibility, scalar and every homologous term comparison, all evidence invariants, optional unavailable values plus nonempty obligations for inconclusive state, the single computation catch boundary, artifact-publication error propagation, import safety, absence of CLI parser imports, editable-dictionary resolution, a temporary run root, exact artifact files, and repeat-run nonoverwrite behavior.
- [ ] **Step 2: Run RED.**

  ```powershell
  python -m pytest tests/unit/test_atomic_artifacts.py tests/promotion/test_h1_gate.py tests/integration/test_verify_vfe4.py -q --junitxml=C:\tmp\vfe4-task6-red.xml
  ```

  Expected: collection fails because the artifact/gate modules and launcher do not exist.

- [ ] **Step 3: Implement atomic publication, provenance, H1 orchestration, and the thin launcher.** Keep `verify_vfe4.py` orchestration-only. `verification.h1_gate` imports public production results plus the independent oracle; no `vfe4.*` module imports `verification.*`.
- [ ] **Step 4: Run GREEN, the full suite, and the click-run file.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h1-full.xml
  python verify_vfe4.py
  ```

  Expected: JUnit reports zero failures/errors; the launcher prints `H1: pass` and a new artifact directory containing the five declared files.

- [ ] **Step 5: Inspect the artifact and manifest mechanically.** Recompute hashes independently in a test or one-off read-only check and verify the recorded revision/config/fixture identities match the current run.
- [ ] **Step 6: Commit.**

  ```powershell
  git add vfe4/artifacts verification/h1_gate.py verify_vfe4.py tests
  git commit -m "feat: add click-to-run H1 promotion gate"
  ```

---

### Task 7: Review and close the bounded H1 increment

**Files:**

- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h1-reference-fixture.md`
- Create: `.verification/active.json` and ledger through the verification-skill tooling only; do not hand-edit them.

- [ ] **Step 1: Run a preliminary mechanical/artifact check at the Task 6 revision.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h1-final.xml
  python -m compileall -q vfe4 verification verify_vfe4.py tests
  python verify_vfe4.py
  git diff --check
  git status --short
  ```

  Expected: JUnit has zero failures/errors, compilation and diff check exit zero, the click-run artifact passes, and only intentional generated `runs/`/verification-control artifacts are untracked or ignored.

- [ ] **Step 2: Run a fresh reviewer against the plan/spec and exact base/head revisions.** Resolve every Critical or Important finding before closure; rerun affected focused tests plus the full JUnit suite after each source change.
- [ ] **Step 3: Update documentation with measured facts only after review fixes are complete.** Record the actual residual, actual calibrated allowance, fixture hash, preliminary JUnit totals, and the exact implementation revision that produced them. Keep H2-H8 explicitly unimplemented. Do not call the codebase backpropagation-free; this increment simply performs no gradient update.
- [ ] **Step 4: Commit all review fixes and documentation changes.**

  ```powershell
  git add vfe4 verification verify_vfe4.py tests README.md docs/preregistrations
  git commit -m "docs: record H1 foundation verification"
  ```

- [ ] **Step 5: At the final committed `HEAD`, rerun the authoritative checks and click-run artifact.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h1-final-head.xml
  python -m compileall -q vfe4 verification verify_vfe4.py tests
  python verify_vfe4.py
  git diff --check
  git status --short
  ```

  Expected: JUnit has zero failures/errors, the final click-run artifact records the final `HEAD`, and the tracked worktree is clean. If any command fails, fix, commit, and repeat this step before opening a ledger.

- [ ] **Step 6: Populate and validate the final-revision claim ledger, then make no further edits or commits.** Close separate claims for configuration strictness, H1 three-way identity, local-term agreement, independent oracle separation, categorical/full-joint normalization, evidence inequalities, the three runtime/JUnit negative controls, click-to-run behavior, artifact identity, and final JUnit counts. Code/math claims require current eligible evidence at the exact final revision; otherwise mark them `INCONCLUSIVE`.

## Out of Scope for This Plan

- `train_vfe4.py`, datasets, token caches, checkpoints, optimization, and language-model training.
- H2 information/moment equivalence and the promoted sparse precision interface.
- H3 structured-vs-factorized adequacy experiments.
- H4/H5 solver cost and update-coherence promotion.
- H6 prefix-safety/predictive evaluation.
- H7 learned-frame updates or covariance.
- H8 scale/allocation evidence.
- Claims of predictive advantage, computational advantage, gauge covariance, exact inference beyond the frozen fixture, or backpropagation-free language-model training.

The next plan begins with H2 only after this exact H1 artifact family closes at a reviewed revision.
