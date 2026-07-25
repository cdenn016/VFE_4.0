# VFE 4.0 H4-H6 Audit Amendments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the July 24 audit findings before any H6 evidence campaign, preserve the already-correct H7/H8 source work, and leave a versioned path into the amended post-H8 WikiText-103 buildout.

**Architecture:** Replace H6's order-invariant primary comparator with a small causal transformer, replace the ambiguous pooled-history source prior with a target-blind parent-specific pooled-prefix prior, and version the H6 arm/gate/matching identities around those two changes. Keep the raw SMC Jensen statement separate from the two-sided Richardson endpoint, promote complete-versus-emission-only NLL to a logical blocking gate, add the missing rectangular Gaussian oracle, and derive downstream predecessor inventories from immutable records instead of editing evidence in place.

**Tech Stack:** Python 3.12 through `C:/anaconda/python.exe`, PyTorch `torch.nn.functional.scaled_dot_product_attention`, NumPy independent oracles, frozen dataclasses and canonical SHA-256 records, Ruff, pytest node-level checks, editable click-run dictionaries.

## Global Constraints

- Product operation remains click-to-run through editable dictionaries in `train_vfe4.py`, `verify_vfe4.py`, and the future figure launcher. Add no `argparse`, required shell variable, or product CLI.
- Preserve the user's live `train_vfe4.py` toggle and preserve `.verification/` byte-for-byte. Neither path may be staged by an amendment task.
- H6 A0 uses `V=258`, maximum receiver horizon and learned-position capacity `L_max=32`, one block, two equal heads, pre-norm, tanh-GELU, untied decoder, no dropout, and CPU float64 SDPA-math semantics.
- The exact A0 parameter formula is `P_A0(h)=2*V*h+32*h+12*h^2+15*h+V`. The copied post-H8 term `128*h` is invalid at H6 scale and would count 96 dormant position rows.
- `h=53` is inadmissible because two equal heads require `h % 2 == 0`. The exact amended parameter witness is `h=52`, two heads of width `26`, and `P_A0=61,982`.
- Do not freeze the obsolete `(emission_width=64, latent_width=16, recognition_width=32, context_width=6)` A5 allocation: its parameter count is close but its current whole-schedule arithmetic is not. First remove only exact shared-frame common-subexpression redundancy, then run a prospective formula-only joint search over `D=(2,4,8)`, `C=(4,6,8)`, `E=(84,85,86,87,88,89)`, and `R=(113,114,115,116,117,118)`, enumerated in ascending lexicographic `(d,c,e,r)` order.
- After both hard matching gates are applied, select the minimum key `(abs(log(P_A5/P_A0)), abs(log(F_A5/F_A0)), d,c,e,r)`. The tuple suffix is the deterministic tie-break; an empty eligible set means PRIMARY is ineligible.
- `(emission_width=89, latent_width=2, recognition_width=113, context_width=6)` gives `P_A5=62,112` and is a provisional parameter-feasibility probe against A0's `61,982`, not a frozen endpoint. Final eligibility requires implemented operator ledgers to close both the unchanged one-percent parameter gate and five-percent whole-schedule training-FLOP gate.
- Raw particle-filter `log Z_hat` is downward-biased in expectation when `Z_hat` is a nonnegative unbiased estimator of `Z`; raw NLL is upward-biased. The H6 endpoint is Richardson `Q2=2*Y_1024-Y_512`; Jensen does not determine the sign of its remaining bias. Keep the current two-sided `Q2` remainder unless a signed expansion is proved.
- The complete-versus-emission gate uses `d_obj=NLL(parent-specific complete ELBO)-NLL(parent-specific emission-only)` and `delta_obj=-log(0.99)=0.01005033585350145`. PASS requires the estimator-aware interval upper bound `<= delta_obj`; FAIL requires its lower bound `> delta_obj`; every other disposition is INCONCLUSIVE.
- Preserve the single all-or-none H6 test opening. OBJECTIVE is adjudicated logically before PRIMARY but does not save confirmatory training or test-scoring compute.
- H4's operand-shaped allowance implementation and amended preregistration already exist. C3 requires fresh, separately authorized evidence; it does not authorize another source rewrite and it does not become an H6-Prediction prerequisite.
- Keep scalar `h1-v1` unchanged. Add the asymmetric `d_z=2,d_m=3` check as a sibling H2/H5 objective-and-update oracle.
- Any operation importing Torch, constructing a model, or making a CUDA statement uses `C:/anaconda/python.exe`. Do not use bare `python`.
- No broad or full pytest run, parameter grid, training run, data download, profiler, numerical campaign, or repeated gate run belongs to ordinary implementation. Each task may run its named test node once after RED and once after GREEN, with a ten-second stop limit for ordinary unit nodes.
- H7/H8 remain source-only and INCONCLUSIVE until separately authorized exact-revision evidence exists. Do not claim H7 PASS, H8 PASS, or post-H8 readiness from source inspection.

---

## File and Responsibility Map

| Surface | Files | Responsibility |
|---|---|---|
| Normative amendments | `docs/preregistrations/2026-07-25-h6-audit-amendment.md`, existing H6 and post-H8 plans/preregistrations | Freeze corrected architecture, prior, estimator, gate, risk, and nonclaim semantics before implementation. |
| H6 transformer | `vfe4/training/h6_transformer.py`, `vfe4/training/arms.py`, `vfe4/training/parameter_counts.py`, `vfe4/training/matching.py` | Order-sensitive A0, exact formulas, active-parameter inventory, metric-blind matching. |
| Parent-content source prior | `vfe4/generative/source_priors.py`, H1 prefix-prior projector/gate, H6 config/types | A target-blind normalized source prior whose score varies with each candidate parent's realized content. |
| SMC semantics/calibration | `vfe4/evaluation/smc_uncertainty.py`, `verification/h6_smc_gate.py`, new `verification/numpy_oracles/h6_linear_gaussian_smc.py`, new `verification/h6_continuous_smc_gate.py` | Raw Jensen boundary, unchanged two-sided Q2 endpoint, independent continuous sensitivity/coverage artifact. |
| Objective gate | `vfe4/types/h6.py`, `vfe4/evaluation/smc_uncertainty.py`, `vfe4/training/h6_readiness.py`, H6 artifacts | Directional logical blocker under one test opening. |
| Rectangular oracle | new `vfe4/validation/h2_h5_rectangular_fixture.py`, new `verification/numpy_oracles/h2_h5_rectangular.py`, new `verification/h2_h5_rectangular_gate.py` | Exercise rectangular `B`, `B^T P_z B`, state/model recoil, and both coordinate updates without mutating H1. |
| Compute disclosure | new `vfe4/evaluation/compute_ledger.py`, `vfe4/artifacts/h6_matching.py` | Keep training eligibility separate from inference-inclusive reporting. |
| Depth and post-H8 protocol | dated preregistration amendment plus existing post-H8 plan | Define normalized depth before code, five-arm inventory, cost/context boundary, risk and attribution limits. |
| Downstream references | H7/H8 reference schemas and readiness validators only | Bind fresh artifacts; do not alter H7/H8 mathematics or overwrite historical evidence. |

---

### Task 1: Freeze One Coherent H6 Amendment

**Files:**
- Create: `docs/preregistrations/2026-07-25-h6-audit-amendment.md`
- Modify: `docs/superpowers/plans/2026-07-21-vfe4-h6-prefix-prediction.md`
- Modify: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`

**Interfaces:**
- Consumes: July 24 peer review and amendment record.
- Produces: versioned literals consumed by Tasks 2-8: `h6-a0-transformer-v2`, `parent_specific_pooled_prefix`, `h6-objective-gate-v1`, `h6-smc-bias-semantics-v2`.

- [ ] **Step 1: Write the dated amendment and revise the live H6 plan/preregistration in place.**

  Freeze:

  ```text
  PRIMARY: h6-a0-transformer-v2 vs
           h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2
  OBJECTIVE: parent-specific pooled-prefix complete vs matching emission-only
  PRIOR control: parent-specific pooled-prefix complete vs fixed complete
  raw estimator: Y_N=log Z_hat_N, Jensen direction known
  reported endpoint: Q2=2Y_1024-Y_512, remainder direction unknown
  H4 relation: separately rerun, nonblocking for H6-Prediction
  ```

  State that the current pooled-latent-history implementation is a distinct legacy/descriptive mechanism and cannot be relabeled as parent-specific addressing. State that the repaired token query is still mean pooled and therefore is not standard transformer attention. State that the objective gate is logically first under the one-opening design and therefore does not reduce already-frozen training/scoring work.

- [ ] **Step 2: Review the normative literals directly.**

  Confirm the architecture, source-prior, matching-inventory, Q2, OBJECTIVE,
  and consumer-specific invalidation clauses against the July 24 review.
  Human prose is not executable behavior and does not receive a grep-only
  pytest contract.

- [ ] **Step 3: Run `git diff --check` and commit only the amendment slice.**

  ```powershell
  git add docs/preregistrations/2026-07-25-h6-audit-amendment.md docs/superpowers/plans/2026-07-21-vfe4-h6-prefix-prediction.md docs/preregistrations/2026-07-21-h6-prefix-prediction.md
  git commit -m "docs: freeze H6 audit amendment"
  ```

---

### Task 2: Replace Primary A0 With the H6 Causal Transformer

**Files:**
- Create: `vfe4/training/h6_transformer.py`
- Modify: `vfe4/training/arms.py`
- Modify: `vfe4/training/parameter_counts.py`
- Modify: `vfe4/training/matching.py`
- Modify: `vfe4/training/__init__.py`
- Test: `tests/unit/test_h6_arms.py`
- Test: `tests/unit/test_h6_parameter_counts.py`
- Test: `tests/unit/test_h6_matching.py`

**Interfaces:**
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class H6A0ArchitectureProfile:
      schema_version: Literal["h6-a0-architecture-v2"]
      vocabulary_size: Literal[258]
      position_capacity: Literal[32]
      hidden_width: Literal[52]
      attention_heads: Literal[2]
      head_width: Literal[26]
      block_count: Literal[1]
      architecture_sha256: str

  def h6_a0_parameter_count(profile: H6A0ArchitectureProfile) -> int: ...

  class H6CausalTransformer(nn.Module):
      def prefix_log_probs(self, prefix: CausalPrefix) -> Tensor: ...
  ```

- [ ] **Step 1: Add three bounded failing tests.**

  Add exact nodes for:

  ```python
  assert h6_a0_parameter_count(profile) == 61_982
  assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 61_982
  assert profile.hidden_width % profile.attention_heads == 0
  ```

  For order sensitivity, compare two prefixes with the same multiset and different order after setting a deterministic non-symmetric parameter fixture; require different logits. For future blindness, compare two windows with identical `CausalPrefix` and different receiver/suffix bytes; require raw-byte-identical logits. Inspect the module tree for exactly one SDPA block, three affine LayerNorms, QKV, attention output, `h->4h->h` MLP, learned `[32,52]` positions, and untied `[258,52]` embedding/decoder storage.

- [ ] **Step 2: Run only the three new RED nodes.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_arms.py::test_a0_transformer_is_order_sensitive_and_future_blind tests/unit/test_h6_arms.py::test_a0_transformer_inventory_is_exact tests/unit/test_h6_parameter_counts.py::test_h6_a0_transformer_formula_matches_live_tensors -q
  ```

  Expected: FAIL on missing profile/model/count function in less than ten seconds.

- [ ] **Step 3: Implement `H6CausalTransformer`.**

  Use a zero, non-parameter BOS row at learned position `0`; place prefix tokens at positions `1..len(prefix)`. This activates every position row over receiver horizons `1..32` without adding a BOS parameter. Apply:

  ```python
  x = bos_plus_prefix_embedding
  x = x + position_embedding(position_ids)
  x = x + attention_output(sdpa(qkv(ln1(x)), is_causal=True, dropout_p=0.0))
  x = x + mlp_out(F.gelu(mlp_in(ln2(x)), approximate="tanh"))
  logits = decoder(final_norm(x)[-1])
  return F.log_softmax(logits, dim=-1)
  ```

  Force the H6 CPU float64 math-SDPA policy in the model-owned call boundary. Do not materialize attention weights or an `[L,L]` mask. Keep the old mean-pool class under an explicit `MeanPooledPrefixFloor` name for legacy/no-latent descriptive use; `build_a0` must never return it.

- [ ] **Step 4: Split transformer and mean-pool parameter formulas.**

  Implement exact integer arithmetic:

  ```python
  def h6_a0_parameter_count(*, vocabulary_size: int, position_capacity: int, hidden_width: int) -> int:
      if hidden_width % 2:
          raise ValueError("H6 A0 hidden width must split into two equal heads")
      return (
          2 * vocabulary_size * hidden_width
          + position_capacity * hidden_width
          + 12 * hidden_width * hidden_width
          + 15 * hidden_width
          + vocabulary_size
      )
  ```

  Keep a separate `mean_pooled_no_latent_parameter_count` for the descriptive floor. Reject `h=53`; do not rescue it by unequal heads, dormant position rows, or filler parameters.

- [ ] **Step 5: Add the analytical A0 training-FLOP profile without declaring eligibility.**

  Add the one-block operator terms to `analytical_training_flop_ledger`: embedding/BOS-position assembly, three LayerNorms, QKV, causal `QK`, softmax, `AV`, output projection, two residuals, tanh-GELU MLP, decoder/log-softmax/CE, backward, clipping, and AdamW. Record the parameter witness separately from `flop_within_tolerance`. If the final source formula exceeds five percent against Task 5's A5 ledger, return `INELIGIBLE` and an obligation; do not alter the threshold or add no-op work.

- [ ] **Step 6: Run the three GREEN nodes once.**

  Run the Step 2 command once. Then:

  ```powershell
  C:/anaconda/python.exe -m ruff check vfe4/training/h6_transformer.py vfe4/training/arms.py vfe4/training/parameter_counts.py vfe4/training/matching.py tests/unit/test_h6_arms.py tests/unit/test_h6_parameter_counts.py tests/unit/test_h6_matching.py
  git diff --check
  ```

- [ ] **Step 7: Commit the A0 slice without user config or evidence.**

  ```powershell
  git add vfe4/training/h6_transformer.py vfe4/training/arms.py vfe4/training/parameter_counts.py vfe4/training/matching.py vfe4/training/__init__.py tests/unit/test_h6_arms.py tests/unit/test_h6_parameter_counts.py tests/unit/test_h6_matching.py
  git commit -m "feat: add causal H6 transformer baseline"
  ```

---

### Task 3: Close C3 With Fresh H4 Evidence, Not Another Rewrite

**Files:**
- Inspect only: `verification/h4_budget.py`
- Inspect only: `vfe4/types/h4.py`
- Inspect only: `docs/preregistrations/2026-07-21-h4-information-cost.md`
- Test only if the static mirror differs: `tests/unit/test_h4_budget.py`
- Evidence, separately authorized: a new immutable H4 run artifact and revision-specific ledger

**Interfaces:**
- Consumes: existing element-local `H4AllowanceOperand`, `H4AllowanceElement`, and `H4_ALLOWANCE_STREAM_DOMAIN`.
- Produces: no source change when the two mirror formulas are identical; later, a fresh H4 result bound to the amended revision.

- [ ] **Step 1: Review the production/verification mirror statically.**

  Confirm both implementations derive:

  ```text
  element_scale = max(1, abs(left), left_norm, abs(right), right_norm)
  solver_allowance = 1e-9 * element_scale only for solver-produced operands
  final_allowance = left_total + right_total + comparison_reduction
  decisive = final_allowance / element_scale < 1e-4
  ```

  If those expressions and field identities match, make no source edit.

- [ ] **Step 2: If and only if the mirror differs, add one boundary node and repair the smaller side.**

  The node must show exact equality is eligible when the operand-shaped allowance is decisive and must show an oversized allowance remains INCONCLUSIVE. Run only:

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h4_budget.py::test_h4_operand_local_mirror_and_decisiveness_boundaries -q
  ```

- [ ] **Step 3: Record the evidence authorization boundary.**

  The ordinary implementation session stops after static review. In a separately authorized click-run evidence turn, produce one exact-revision H4 artifact and ledger. Do not rerun H1-H5 merely for reassurance; regenerate downstream predecessor evidence only when Task 11 prepares a frozen H7/H8 candidate.

---

### Task 4: Correct the SMC Bias Semantics and Add Continuous Calibration

**Files:**
- Modify: `vfe4/evaluation/smc_uncertainty.py`
- Modify: `verification/h6_smc_gate.py`
- Create: `verification/numpy_oracles/h6_linear_gaussian_smc.py`
- Create: `verification/h6_continuous_smc_gate.py`
- Modify: `vfe4/types/h6.py`
- Test: `tests/unit/test_h6_smc_uncertainty.py`
- Test: `tests/oracle/test_h6_smc_oracle.py`

**Interfaces:**
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class SmcBiasSemantics:
      schema_version: Literal["h6-smc-bias-semantics-v2"]
      raw_log_normalizer_direction: Literal["downward_or_equal_in_expectation"]
      raw_nll_direction: Literal["upward_or_equal_in_expectation"]
      reported_endpoint: Literal["Q2=2Y_1024-Y_512"]
      q2_remainder_direction: Literal["unknown_without_signed_expansion"]
      q2_bound_kind: Literal["two_sided_conditional_geometric_remainder"]
      contraction: Literal[0.75]
      semantics_sha256: str

  @dataclass(frozen=True, slots=True)
  class LinearGaussianSmcCalibrationReport:
      fixture_sha256: str
      exact_log_likelihood: float
      raw_records: tuple[...]
      q2_record: ...
      q2_interval_contains_exact: bool
      transfer_claim: Literal["sensitivity_control_not_trained_endpoint_bound"]
      report_sha256: str
  ```

- [ ] **Step 1: Add a unit test that separates raw and extrapolated signs.**

  Require construction to reject:

  ```python
  SmcBiasSemantics(q2_remainder_direction="downward")
  SmcBiasSemantics(q2_bound_kind="one_sided")
  ```

  Retain the existing 256-corner two-sided `inflate_paired_interval` arithmetic for Q2.

- [ ] **Step 2: Run the one RED node.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_smc_uncertainty.py::test_raw_jensen_sign_does_not_claim_a_q2_sign -q
  ```

- [ ] **Step 3: Add the typed semantic record without changing Q2 interval geometry.**

  Document the expansion:

  ```text
  E[Y_N] = Y + c/N + d/N^2 + ...
  E[2Y_2N-Y_N] = Y - d/(2N^2) + ...
  ```

  Therefore `c<=0` from Jensen does not determine the sign of the Q2 remainder. Keep `bias_bound=u2/(1-0.75)` two-sided and retain INCONCLUSIVE when empirical contraction fails.

- [ ] **Step 4: Build an independent continuous calibration fixture.**

  Freeze a deterministic linear-Gaussian state-space fixture with `T=32`, state dimension `16`, and observation dimension `16`. Use `m0=0`, `P0=I`, `A=0.75*I+0.05*S` where `S[k,k+1]=1` and every other entry is zero, `Q=0.2*I`, `C=I`, `R=0.3*I`, and

  ```text
  x_t[k] = sin((t+1)*(k+1)/17) + 0.1*cos((t+1+2*k)/11)
  ```

  for zero-based `t=0..31`, `k=0..15`. Store the resulting decimal arrays and checked-in canonical JSON bytes. The NumPy oracle computes exact `log p(x)` by a Kalman innovation recursion and does not import `vfe4.numerics.linear_gaussian`, Torch, or the production particle recursion.

  The production calibration uses the same particle ladder `(128,256,512,1024)` and reports raw `Y_N`, Q2, contraction, and two-sided coverage. Its full replicate inventory is an evidence operation, not a pytest parameter grid. Unit coverage uses only `T=3`, dimension `2`, particle counts `(8,16)`, and two deterministic streams.

- [ ] **Step 5: State the calibration's limited conclusion.**

  Passing means the implementation's continuous recursion and declared Q2 envelope cover the independent fixture under that frozen profile. It does not prove a finite-sample raw sign, a signed Q2 remainder, geometric contraction on trained checkpoints, or transfer of a bias bound to WikiText.

- [ ] **Step 6: Run only the tiny GREEN nodes.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_smc_uncertainty.py::test_raw_jensen_sign_does_not_claim_a_q2_sign tests/oracle/test_h6_smc_oracle.py::test_tiny_linear_gaussian_calibration_matches_independent_kalman -q
  ```

  Then run Ruff on the changed files and `git diff --check`. Do not run the `T=32` evidence inventory.

---

### Task 5: Promote a Parent-Specific Pooled-Prefix Source Prior and Recompute Matching

**Files:**
- Modify: `vfe4/generative/source_priors.py`
- Modify: `vfe4/types/h6.py`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `vfe4/training/arms.py`
- Modify: `vfe4/training/parameter_counts.py`
- Modify: `vfe4/training/matching.py`
- Modify: `verification/h1_prefix_prior_gate.py`
- Modify: `verification/numpy_oracles/h1_prefix_prior.py`
- Test: `tests/unit/test_h6_source_priors.py`
- Test: `tests/oracle/test_h1_prefix_prior.py`
- Test: `tests/unit/test_h6_matching.py`

**Interfaces:**
- Produces:

  ```python
  PriorVariant = Literal[
      "fixed",
      "pooled_history_conditioned",
      "parent_specific_pooled_prefix",
  ]

  class ParentSpecificPooledPrefixSourcePrior(_SourcePriorBase):
      def state_source_log_probs(
          self, *, prefix: CausalPrefix, earlier_latents: Tensor
      ) -> NormalizedSourceFactor: ...
      def model_source_log_probs(
          self, *, prefix: CausalPrefix, earlier_latents: Tensor
      ) -> NormalizedSourceFactor: ...
  ```

- [ ] **Step 1: Add a failing parent-addressability test.**

  At receiver `t=2`, use a nonzero prefix query and bank projection, zero slot keys/biases, and distinct parent latent rows. Swap the realized latents assigned to `j=0` and `j=1`; require the two supported probabilities to swap in both banks. Perturb `x_t` and every suffix token while preserving `x_<t` and earlier generated latents; require raw-byte-identical probabilities and context identity. On a sparse row, perturb a latent outside the declared parent support and require no change.

- [ ] **Step 2: Run the single RED node.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_source_priors.py::test_parent_specific_prior_addresses_each_candidate_without_target_leakage -q
  ```

- [ ] **Step 3: Replace pooled-latent-history scoring with a separately named parent-specific implementation.**

  Preserve the current parameter families but change their semantics. For bank `b`, receiver `t`, and parent `j`, define:

  ```text
  q_t = 0                                             when t=1
  q_t = mean(E[x_1],...,E[x_(t-1)])                  when t>1
  raw_score_b,t,j = q_t^T (k_b,t,j + W_b y_b,j) + beta_b,t,j
  ```

  Store free `(k,beta)` rows for `parents[:-1]`; define the last declared parent's slot key and bias as exact zeros, compute every parent's complete raw score including the anchor's `q_t^T W_b y_b,anchor`, subtract that complete anchor score, and then call the existing gauge-anchored masked normalization. This uses the unchanged count

  ```text
  V*c + 2*d*c + T*(T-1)*c + T*(T-1)
  ```

  while making the score depend on each candidate parent's own latent content. Rename the old mean-projected-latent behavior `pooled_history_conditioned`; never deserialize it as the new variant. Freeze scorer schema `parent-specific-pooled-prefix-bilinear-v1`, context-hash domain v2, `token_summary=mean-prior-token-embeddings-v1`, `parent_content=bank-projection-of-candidate-row-v1`, and `anchor=last-declared-parent-complete-score-subtraction-v1`. State explicitly that the token query remains order-invariant and the stochastic selector is not transformer self-attention.

- [ ] **Step 4: Promote the new variant in H1 and H6.**

  Add an H1-Prefix scorer-v2 sibling fixture with `T=2`, `d_z=d_m=1`, `V=3`, nonzero bank projections, distinct nonzero parent latents, and active/swapped histories under one fixed target-free prefix. Extend the independent NumPy oracle/projector to recheck production probabilities, active/swapped complete-ELBO decompositions, normalization, and target blindness of the changed joint. In H6, make parent-specific pooled-prefix complete A5 the PRIMARY right arm and fixed complete the PRIOR control. Use versioned config/factory/model-family hashes and reject fallback to fixed or pooled-history rows.

- [ ] **Step 5: Remove exact shared-frame redundancy before matching.**

  In `LatentLanguageArmModel`, build every `U_t=matrix_exp(phi_t)` once per channel and live forward graph, reuse each source pullback/solve across receivers, and invalidate the cache after every parameter update. Apply the same semantics-preserving common-subexpression elimination to every shared-frame endpoint. Add a tiny cached-versus-uncached test requiring identical values, gradients, source support, and active parameter bytes. Do not cache across detached snapshots or optimizer updates.

- [ ] **Step 6: Run a prospective formula-only joint search and fail closed.**

  Keep A0 fixed at `h=52`. Search only the Cartesian inventory
  `D=(2,4,8)`, `C=(4,6,8)`, `E=(84,85,86,87,88,89)`, and
  `R=(113,114,115,116,117,118)` in ascending lexicographic `(d,c,e,r)`
  order, preserving the normalized joint, objective, source
  support, recognition regime, token exposure, update schedule, and optimizer
  policy. Require:

  ```python
  assert h6_a0_parameter_count(..., hidden_width=52) == 61_982
  assert 53 % 2 != 0
  assert parameter_relative_difference <= 0.01
  assert whole_schedule_training_flop_relative_difference <= 0.05
  ```

  Filter on both hard gates, then select the minimum key
  `(abs(log(P_A5/P_A0)),abs(log(F_A5/F_A0)),d,c,e,r)`. Record
  `(e,d,r,c)=(89,2,113,6)` and `P_A5=62,112` only as a provisional
  parameter-feasibility probe: its absolute gap from A0 is `130`, about
  `0.209%`. It becomes a frozen endpoint only if the implemented cached-A5
  and transformer operator ledgers select it under that rule. If no candidate
  closes both gates, PRIMARY is ineligible/INCONCLUSIVE; do not add redundant
  recomputation, dormant parameters, fake phases, no-op optimizer work, or a
  wider tolerance.

- [ ] **Step 7: Run four tiny GREEN nodes once.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_source_priors.py::test_parent_specific_prior_addresses_each_candidate_without_target_leakage tests/oracle/test_h1_prefix_prior.py::test_parent_specific_joint_is_normalized_and_target_blind tests/unit/test_h6_matching.py::test_shared_frame_cache_preserves_values_and_gradients tests/unit/test_h6_matching.py::test_parent_specific_primary_joint_search_closes_both_gates_or_fails_closed -q
  ```

  Run exact-file Ruff and `git diff --check`; do not run an H1 or H6 evidence campaign.

---

### Task 6: Make OBJECTIVE a Logical Blocking Gate and Record the V3 Risk

**Files:**
- Modify: `vfe4/types/h6.py`
- Modify: `vfe4/evaluation/smc_uncertainty.py`
- Modify: `vfe4/training/h6_readiness.py`
- Modify: `vfe4/artifacts/h6.py`
- Modify: `docs/superpowers/plans/2026-07-21-vfe4-h6-prefix-prediction.md`
- Modify: `docs/preregistrations/2026-07-21-h6-prefix-prediction.md`
- Test: `tests/unit/test_h6_statistics.py`
- Test: `tests/unit/test_h6_prediction_readiness.py`

**Interfaces:**
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class ObjectiveGateSpec:
      schema_version: Literal["h6-objective-gate-v1"]
      complete_arm_id: str
      emission_arm_id: str
      orientation: Literal["nll_complete_minus_nll_emission"]
      delta_obj: Literal[0.01005033585350145]
      opening_policy: Literal["single_all_or_none"]
      evaluation_order: Literal["OBJECTIVE_then_PRIMARY"]
      spec_sha256: str

  def decide_objective_gate(
      interval: InflatedPairedInterval, spec: ObjectiveGateSpec
  ) -> Literal["PASS", "FAIL", "INCONCLUSIVE"]: ...
  ```

- [ ] **Step 1: Add exact boundary tests.**

  Test upper equal to `delta_obj` as PASS, lower strictly greater as FAIL, an interval crossing the boundary as INCONCLUSIVE, and any ineligible Q2 endpoint as INCONCLUSIVE.

- [ ] **Step 2: Run the RED boundary node.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_statistics.py::test_objective_gate_direction_and_exact_margin_boundaries -q
  ```

- [ ] **Step 3: Implement the gate over parent-specific pooled-prefix arms only.**

  The two endpoints must differ only in:

  ```text
  objective_kind: complete_elbo -> emission_only_ablation_non_elbo
  ```

  Keep prior variant, latent path, capacity allocation, data order, seed, optimizer policy, checkpoint role, scorer, particle streams, and terminal opening identical. Preserve `is_elbo=False` on the emission arm.

- [ ] **Step 4: Enforce logical ordering without a second test opening.**

  Score the complete frozen endpoint inventory once. Evaluate OBJECTIVE first. If it is FAIL or INCONCLUSIVE, record PRIMARY as `NOT_EVALUATED_AFTER_OBJECTIVE_GATE` even though its already-opened raw endpoint records remain immutable. Do not claim this ordering saved training or scoring compute.

- [ ] **Step 5: Add the provenance-bounded C5 risk.**

  Record:

  ```text
  risk_id = "v3-free-energy-versus-cross-entropy-tension-v1"
  status = "historical_risk_signal_not_vfe4_evidence"
  mitigation_gate = "h6-objective-gate-v1"
  ```

  Include exact V3 numeric sequences only when a primary artifact path, revision, config, and digest have been verified and recorded. Until then, cite the qualitative tension and the peer-review source, not the transcribed values as evidence.

- [ ] **Step 6: Run two GREEN nodes and static checks.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_statistics.py::test_objective_gate_direction_and_exact_margin_boundaries tests/unit/test_h6_prediction_readiness.py::test_objective_gate_blocks_primary_without_a_second_opening -q
  ```

---

### Task 7: Add the Rectangular H2/H5 Objective-and-Update Oracle

**Files:**
- Create: `vfe4/validation/h2_h5_rectangular_fixture.py`
- Create: `verification/numpy_oracles/h2_h5_rectangular.py`
- Create: `verification/h2_h5_rectangular_gate.py`
- Modify: `vfe4/numerics/linear_gaussian.py` only through new vector/matrix sibling functions
- Modify: `vfe4/inference/h5_updates.py`
- Test: `tests/oracle/test_h2_numpy_oracle.py`
- Test: `tests/unit/test_h5_updates.py`

**Interfaces:**
- Consumes: peer-review C5 construction with `T=3`, `d_z=2`, `d_m=3`, dense parent sets, Gaussian emission.
- Produces: `H2H5RectangularFixture`, `RectangularInformationAssembly`, and `RectangularUpdateOracleReport`.

- [ ] **Step 1: Freeze the sibling fixture without changing `h1-v1`.**

  Use PCG64 seed `31337`, `T=3`, `d_z=2`, `d_m=3`, observation dimension `2`, and dense causal parents `range(t)`. Store exact arrays and raw/canonical hashes in a new fixture identity. H1 config/resolution remains scalar and byte-identical.

- [ ] **Step 2: Add two failing nodes.**

  Require independent NumPy and production routes to agree on the rectangular `B_s.T @ P_z @ B_s` pullback and the model-channel recoil vector. Require both state and model natural updates `(J,h)` to match, with explicit transpose-shape rejection.

- [ ] **Step 3: Run only the two RED nodes.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/oracle/test_h2_numpy_oracle.py::test_rectangular_information_assembly_matches_independent_dense_oracle tests/unit/test_h5_updates.py::test_rectangular_state_and_model_updates_match_independent_oracle -q
  ```

- [ ] **Step 4: Add vector/matrix sibling assembly and update paths.**

  Do not widen historical scalar payloads. The independent oracle directly evaluates the dense quadratic and closed-form natural parameters; it must not import production assembly/update functions. The production path must reject `B` with `(d_m,d_z)` orientation and accept only `(d_z,d_m)`.

- [ ] **Step 5: Run the two GREEN nodes once and static checks.**

  Use the Step 3 command. Do not run the peer-review script's derivative-free 4,000-iteration minimizer in pytest.

---

### Task 8: Add Inference-Inclusive Reporting Without Changing Match Eligibility

**Files:**
- Create: `vfe4/evaluation/compute_ledger.py`
- Modify: `vfe4/artifacts/h6_matching.py`
- Modify: `vfe4/types/h6.py`
- Test: `tests/unit/test_h6_matching_artifact.py`

**Interfaces:**
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class InferenceComputeRecord:
      endpoint_id: str
      scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
      particle_count: int | None
      replicate_count: int
      prefix_cache_mode: str
      checkpoint_load_flops: int
      cache_build_flops: int
      scoring_flops: int
      total_flops: int
      wall_time_seconds: float | None
      record_sha256: str
  ```

- [ ] **Step 1: Add one failing ledger-separation node.**

  Require the training match result to be byte-identical when inference records are added or changed. Require A5 scoring FLOPs to increase monotonically over `(128,256,512,1024)` particles and require exact A0 to carry `particle_count=None`.

- [ ] **Step 2: Implement the separate inference-inclusive table.**

  Report `(training FLOPs, scoring FLOPs by N, declared-workload total, measured wall time when available)` and label the scientific match claim exactly `training-compute-matched`. Never feed an `InferenceComputeRecord` into the one-percent/five-percent eligibility predicate.

- [ ] **Step 3: Run the single node once after GREEN.**

  ```powershell
  C:/anaconda/python.exe -m pytest tests/unit/test_h6_matching_artifact.py::test_inference_ledger_is_complete_and_cannot_change_training_eligibility -q
  ```

---

### Task 9: Define Depth Before Adding the H6 Depth-2 Risk Arm

**Files:**
- Create: `docs/preregistrations/2026-07-25-h6-depth2-cascade-amendment.md`
- Create: `vfe4/types/h6_depth.py`
- Create: `vfe4/generative/h6_depth.py`
- Create: `vfe4/objective/h6_depth.py`
- Create: `verification/numpy_oracles/h6_depth.py`
- Test: `tests/oracle/test_h6_depth_oracle.py`

**Interfaces:**
- Produces a normalized two-layer cascade, not a dead `model_depth=2` config field.

- [ ] **Step 1: Freeze the exact cascade semantics.**

  Layer 1 owns `(z1_t,m1_t,a1_t,b1_t)`. Layer 2 owns `(z2_t,m2_t,a2_t,b2_t)`. The normalized joint factorization is:

  ```text
  p(y1_0)
  p(y2_0 | y1_0)
  product_t p(a1_t|prefix1) p(b1_t|prefix1)
            p(m1_t|m1_parent) p(z1_t|z1_parent,m1_t)
            p(a2_t|prefix2) p(b2_t|prefix2)
            p(m2_t|m2_parent,m1_t)
            p(z2_t|z2_parent,m2_t,z1_t)
            p(x_t|z2_t,m2_t)
  ```

  Every Gaussian conditional is normalized; source rows are normalized and target-blind; the complete ELBO contains both layers' initial/source/transition terms, the one top-layer emission term, and the full recognition entropy. Parameters are independent by layer and optimizer ownership is disjoint.

- [ ] **Step 2: Freeze the scientific disposition.**

  `H6-DEPTH2-CASCADE-v1` is a nonblocking composition-risk probe. Its result cannot promote H6, establish scalable depth, or change the first WT103 profile, which remains depth 1. It receives its own metric-blind match report and no fallback to repeated inference steps.

- [ ] **Step 3: Add a tiny independent normalization/objective node.**

  At `T=2`, scalar channels, and `V=3`, enumerate source choices and use independent Gaussian reductions to verify normalization and local-sum/monolithic complete-objective equality. Run only:

  ```powershell
  C:/anaconda/python.exe -m pytest tests/oracle/test_h6_depth_oracle.py::test_depth2_cascade_is_normalized_and_complete_objective_matches -q
  ```

  This source gate must pass before a trainable depth-2 arm or config literal is added. A failed or inconclusive gate leaves the depth arm absent rather than treating depth as a harmless integer.

---

### Task 10: Amend the Post-H8 Protocol Before WikiText-103 Code

**Files:**
- Modify: `docs/superpowers/plans/2026-07-21-vfe4-post-h8-wikitext103-training.md`
- Create: `docs/preregistrations/2026-07-25-post-h8-arm-gate-amendment.md`
- Test: `tests/unit/test_h8_allocation.py` only for source-contract lint added here

**Interfaces:**
- Produces the ordered minimum inventory:

  ```text
  WT103-A0-AR-v1
  WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1
  WT103-A5-FIXED-COMPLETE-v1
  WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1
  WT103-A5-NOLATENT-v1
  ```

- [ ] **Step 1: Replace all hard-coded two-arm counts with an immutable arm/gate inventory.**

  Define `WT103ArmSpec`, `WT103GateSpec`, and derived `EndpointInventory`. Tuning attempts, terminal checkpoints, SMC records, resource forecasts, result rows, and figure series must consume the inventory hash rather than independently entered constants.

- [ ] **Step 2: Carry B1/B2/B3 forward with honest attribution.**

  Parent-specific pooled-prefix complete is primary; fixed complete is a changed-joint control; parent-specific pooled-prefix emission-only is the objective-only gate; no-latent is an explicitly bundled latent-path control unless its held-fixed semantics are mechanically proven. Do not collapse the four VFE rows into one plot label.

- [ ] **Step 3: Freeze C1 and C5 claim boundaries.**

  State:

  ```text
  WT103 training: L=128, b=40, D=L*b=5,120
  H8 synthetic fixture: N=129, b=40, D=5,160
  A5 direct source lookback: W=20
  A0 direct attention reach: full causal 128
  ```

  The dense `O(D^3)` value is an unbanded counterfactual, not measured H8 runtime. The implemented width-20 algorithm must expose every lag-`1..20` cross-moment needed by exact transition expectations or explicitly declare an approximation and its error. Describe A5 as finite direct-address width with recursively propagated state, not as categorically incapable of long-range influence. Record the V3 F-versus-CE observation as a provenance-bounded risk linked to the objective gate.

- [ ] **Step 4: Preserve the click-run contract.**

  The post-H8 launcher remains one editable dictionary with `operation="idle"` by default and explicit `source_lock|readiness|train|resume` operations. Figure generation uses a separate editable dictionary. Import performs no I/O, device initialization, data discovery, run selection, or rendering.

- [ ] **Step 5: Stop at protocol readiness.**

  No WikiText-103 loader, training engine, data download, source lock, or figure generation starts until an exact H8 PASS exists for the implementation revision and the amended endpoint inventory is frozen.

---

### Task 11: Rebuild Reference Identities Without Editing Historical Evidence

**Files:**
- Modify only as required by schema versions: H7/H8 reference types and readiness validators
- Do not modify for bookkeeping alone: `docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md`
- Modify the H8 plan/preregistration SHA literal only if H7 semantic text actually changes
- Preserve: `.verification/**`

**Interfaces:**
- Produces fresh future references for H4, rectangular H2/H5, parent-specific H1-Prefix, H6-Prefix, H6-Prediction, H7, and H8.

- [ ] **Step 1: Classify invalidation by interface.**

  ```text
  B1 scorer -> fresh H1-Prefix and H6-Prefix -> future H7 reference
  A1/A2/B2/A3/C2 -> fresh H6-Prediction -> future H8 reference only
  C3 -> fresh H4 result -> only consumers that explicitly bind that result
  C4 -> fresh rectangular sibling result; scalar H1 stays unchanged
  ```

  H6-Prediction is not an H7 premise. None of these changes the committed
  H7/H8 mathematical algorithms.

- [ ] **Step 2: Preserve the H7 plan pin unless its semantic text changes.**

  H8 currently pins the canonical UTF-8/LF SHA-256 of the H7 plan. If H7 plan text remains untouched, preserve that literal. If an H7-specific semantic edit is unavoidable, recompute the canonical digest, update every H8 plan/preregistration/config/test literal in one commit, and make all prior H8 references stale.

- [ ] **Step 3: Update validators to require new versioned references.**

  Missing, old fixed/pooled-prior, old A0, unsigned-bias-semantics, or descriptive-objective references yield INCONCLUSIVE. Validators reference immutable predecessor artifacts; they never copy, overwrite, or synthesize predecessor payloads.

---

### Task 12: Perform Prudent Source Verification and Leave Evidence Campaigns Separate

**Files:**
- All files changed by Tasks 1-11
- No generated evidence staged

**Interfaces:**
- Produces a source-reviewed amendment candidate, not scientific PASS evidence.

- [ ] **Step 1: Review each task immediately after its bounded commit.**

  Use one fresh reviewer for specification compliance and one for code quality. Resolve findings before starting the next dependency task. Reviewers use static source and the named node output; they do not launch additional suites.

- [ ] **Step 2: Run changed-file static checks once at the end.**

  ```powershell
  git diff --check
  $vfe4ChangedPython = git diff --name-only --diff-filter=ACMR -- '*.py'
  C:/anaconda/python.exe -m ruff check $vfe4ChangedPython
  ```

  Inspect `$vfe4ChangedPython` before running Ruff; do not point Ruff at the whole repository.

- [ ] **Step 3: Run one amendment smoke selection only if all prior nodes are fast.**

  Select one node each for A0 architecture, parent-specific target blindness, Q2 sign separation, objective-gate boundary, and rectangular update. Stop the selection if collection or execution reaches 30 seconds total. Do not run a full test file or broad suite.

- [ ] **Step 4: Report source status precisely.**

  Report exact commits and static/node checks. State:

  ```text
  H4 evidence: INCONCLUSIVE until separately rerun
  H6 evidence: not run
  H7 evidence: INCONCLUSIVE/source-only
  H8 evidence: INCONCLUSIVE/source-only
  WikiText-103: protocol-only until H8 PASS
  ```

- [ ] **Step 5: Keep scientific runs separately authorized and click-driven.**

  The future order is: one fresh H4 artifact; parent-specific H1-Prefix and H6-Prefix prerequisites; finite plus continuous SMC calibration; H6 readiness; one frozen H6 experiment/opening with OBJECTIVE adjudicated before PRIMARY; H7 exact-revision evidence; H8 exact-revision evidence; then post-H8 source lock and WikiText-103 buildout. Ordinary implementation does not silently start any of them.

---

## Self-Review Checklist

- [ ] A1 uses H6 `L_max=32`, even `h=52`, exact `61,982`, and no dormant position rows.
- [ ] No A5 allocation is frozen from parameter proximity alone; `(89,2,113,6)`/`62,112` remains provisional until the cached whole-schedule FLOP ledger closes.
- [ ] C3 recognizes the current operand-local source repair and requests only fresh evidence.
- [ ] A2 signs raw `log Z_hat`/NLL but keeps Richardson Q2 two-sided absent a proof.
- [ ] B2 freezes `delta_obj`, exact interval rules, single opening, and logical-not-compute ordering.
- [ ] B1 is parent-specific in latent content, retains the honest pooled-token-prefix label, and is not the current pooled-latent-history mechanism under a new label.
- [ ] C4 preserves scalar H1 and covers rectangular H2/H5 objective/update terms.
- [ ] A3 leaves training-match eligibility unchanged.
- [ ] C1 distinguishes `D=5,120` WT103 from `D=5,160` H8 and does not call a dense counterfactual measured runtime.
- [ ] C2 defines a normalized depth-2 joint before exposing a depth field.
- [ ] C5 remains a provenance-bounded V3 risk, not VFE4 evidence.
- [ ] B3 drives post-H8 counts from a hashed inventory and retains prefix, objective, and latent controls.
- [ ] H7/H8 source mathematics remains untouched; only future evidence/reference identities are regenerated.
- [ ] No product CLI, broad CPU suite, Torch command through bare Python, training, download, or evidence campaign appears in an implementation task.
