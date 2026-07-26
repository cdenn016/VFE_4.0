# VFE 4.0 H7 Internal Population-Frame Covariance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify at one exact revision that the complete normalized VFE 4.0 generative and recognition laws, every local ELBO term, the monolithic log-ratio ELBO, and the available evidence/posterior-KL identities transform covariantly under the declared direct `GL+(2)` population-frame action, while differential entropy and coordinate densities receive their required Jacobian shifts.

**Architecture:** Add a pure PyTorch tensor-action layer whose outputs retain autograd graphs, then add immutable generative and recognition pushforward adapters that transform every factor of a frozen complete-law snapshot without mutating the post-H6 model. H7 uses a separately typed `GL+(1,R)` replay of the scalar `h1-v1` complete law as a regression and a primary full-matrix `GL+(2,R)` `h7-v1` law as the H7 group claim, compares the production float64 complete objective with an independent 100-decimal mpmath oracle, and publishes one fail-closed `validation/h7.json`; the verification runner accepts no gate beyond H7.

**Tech Stack:** Python 3.10+, PyTorch float64, mpmath at 100 decimal digits, exact discrete source enumeration, deterministic Gauss--Hermite orders 41 and 51, pytest, SHA-256 provenance, atomic JSON artifacts, JUnit XML.

## Global Constraints

- Implement this plan only after the H6 plan is complete at a clean candidate revision. Consume the post-H6 normalized model, recognition, objective, source-prior, artifact, and runner interfaces without changing their probability semantics, trained parameter bytes, update schedule, predictor contract, or empirical results.
- H7 is a forward change-of-variables verification. It is not optimizer equivariance, gradient-flow equivariance, training invariance, a predictive benefit, an information-form cost result, or evidence for H8 scale.
- Preserve the detached H2 snapshot/evaluation seam byte-for-byte in behavior. Do not add `requires_grad`, mutable state, a covariance property, or optimizer ownership to `PrecisionFactor`, `DenseCholeskyPrecision`, `InformationGaussian`, H2 assemblers, or H2 results.
- Add a separate pure differentiable tensor transform layer. It must not call `detach`, `clone`, `torch.no_grad`, NumPy, JSON, the oracle, or artifact code. Its unit tests require finite reverse-mode derivatives with respect to both the input tensor and direct group element. H7 closure nevertheless concerns forward covariance only.
- The primary group is exactly `G=GL+(2,R)`, with `d_z=d_m=2`, standard state and model representations, `T=2`, `D=12`, and `V=3`. Group elements are supplied directly. The scalar replay has its own exact `GL+(1,R)` action type and is a complete-law regression only; it does not establish any `GL+(2)` result. Only the matrix trials support the primary `GL+(2)` claim. Do not parameterize either action with `phi`, `exp(phi)`, BCH, a Lie-algebra chart, or `torch.matrix_exp`.
- A single diagonal direct element at every population label is the base action inherited from the one principal bundle over `C0={*}`. Independent direct elements `(g0,g1,g2)` are the separately declared internal product action. Report these actions separately; never call the product action three independent base gauge transformations.
- The exact core uses population frames `U_t` and receiver/source links `Omega_tj=U_t U_j^{-1}`. The transformed laws are `U_t'=g_t U_t` and `Omega_tj'=g_t Omega_tj g_j^{-1}`. Products act rightmost first. Require `Omega_21 Omega_10=Omega_20` and the closed relation-walk product `Omega_02 Omega_21 Omega_10=I`; this relation-walk check is not a causal-DAG cycle or base holonomy.
- Freeze and test the typed morphism law `B_t'=g_t B_t g_t^{-1}` for the standard equal-dimensional representations. Implement the general receiver/model form `G_z,t B_t G_m,t^{-1}` even though this fixture has `G_z,t=G_m,t=g_t`. The required equal-dimensional mutant is the genuinely distinct reverse-arrow expression `G_m,t^{-1} B_t G_z,t`, not a state/model channel-label swap that collapses to the correct law when both standard representations use the same `g_t`. A separate non-square `d_z=2,d_m=3` type test must reject transposed operands and channel swaps before arithmetic.
- Freeze and test all Gaussian laws: `mu'=G mu`, `Sigma'=G Sigma G^T`, `M'=G M G^T`, `h'=G^{-T}h`, and `J'=G^{-T}J G^{-1}`. The global `G` is block diagonal in exact order `[z0,m0,z1,m1,z2,m2]`, with one two-dimensional state block and one two-dimensional model block per population label.
- Transform every generative and recognition transition offset, receiver covariance, receiver precision, affine parent map, receiver Jacobian, frame link, and `B` morphism. A model transition map from source `j` to receiver `t` transforms as `A_tj'=g_t A_tj g_j^{-1}`. The state transition's same-receiver model map transforms as `B_t'=g_t B_t g_t^{-1}`.
- Decoder maps transform contragrediently: `W_z,t'=W_z,t g_t^{-1}`, `W_m,t'=W_m,t g_t^{-1}`, and `bias_t'=bias_t`. The complete transformed-decoder trials require invariant logits. A held-fixed decoder is positive only on the centered-softmax emission-kernel stabilizer `C_V W g^{-1}=C_V W`, where `C_V=I-(1/V)11^T`; outside that stabilizer it is a required negative control.
- Discrete source variables, support, exact source order, and categorical probabilities are unchanged. Fixed source tables must be raw-byte identical after pushforward. H7 exercises the frozen linear history scorer `s_{b,t,j}(x_{<t},z_j,m_j)=alpha_{b,t,j}(x_{<t})+(r^z_{b,t,j})^T z_j+(r^m_{b,t,j})^T m_j`, where `b` is the state or model bank. Its exact prefix term is unchanged, while `r_z'=G_z,j^{-T}r_z` and `r_m'=G_m,j^{-T}r_m`; therefore every raw score, mask, normalized probability, support entry, and source order is invariant. A changed probability/support, changed prefix term, omitted source inverse, or receiver-frame inverse substituted for the source-frame inverse is a violation.
- A normalized coordinate density is not invariant. Let `logJ_G=sum_t(log|det G_z,t|+log|det G_m,t|)=2*sum_t log(det g_t)` for the standard representations. The complete generative and recognition log densities each shift by `-logJ_G`; their pointwise log ratio is invariant; the continuous recognition entropy shifts by `+logJ_G`. Initial and receiver-conditional density shifts must also be recorded separately. Do not label entropy invariant.
- H7 compares the existing complete local objective, a separate monolithic `E_q[log p-log q]` evaluation, every local term, complete `p` and `q` density shifts, the pointwise log ratio, and the scalar ELBO. A latent-only, transition-only, decoder-only, final-scalar-only, or partial-term test cannot pass H7.
- The initial continuous contribution is exactly one joint term `K0 = KL(Q0(z0,m0) || p0(z0,m0))`. The primary structured fixture has nonzero cross-channel covariance, so H7 may not replace `K0` by `KL(Q0(z0)||p0(z0)) + KL(Q0(m0)||p0(m0))`. H7-v1 uses the joint term directly; a future marginal-plus-conditional chain form would require its own typed record and a mechanical equality check against the same joint KL before use.
- H7 exercises both exact post-H6 recognition families. `structured_full_block` is represented by unrestricted joint/receiver blocks. `factorized_diagonal_within_fiber` starts with the explicitly frozen diagonal-within-fiber fixture below, but a generic non-diagonal `GL+(2)` congruence does not remain diagonal. Its transformed value is therefore promoted to `H7RecognitionTensorLaw(representation="unrestricted_full_block_pushforward", origin_family="factorized_diagonal_within_fiber")`; H7 never casts it back to `FactorizedLanguageRecognition`, projects it to a diagonal law, or claims same-family closure.
- Separate graph-preserving borrowed tensor/action views from owned evidence snapshots. A borrowed view retains the caller's exact tensor object, storage identity, version, dtype, shape, device, contiguity, and autograd graph and is never serialized or published. Every fixture snapshot, trial result, gate result, and artifact owns a contiguous cloned value plus capture identity, owned-storage version, raw bytes, raw-byte SHA-256, and a domain-separated canonical integrity SHA-256. Published results never retain a caller-owned tensor or mutable mapping.
- Source/scorer evidence is row-typed by bank, receiver, source, and channel. The context carries separate `z_history` and `m_history`; each row binds exact prefix bytes and prefix term, alpha coefficients/value, channel-specific covectors, Boolean mask, ordered support, normalized probabilities, raw scores, raw bytes, and canonical identity. A combined untyped latent-history tuple or receiver-only scorer record is invalid.
- Required trials have closed IDs, a closed role (`scalar_regression`, `positive_covariance`, or `expected_negative`), an expected predicate, an owned frozen action snapshot, and an action SHA-256. `matrix-fixed-decoder-outside-stabilizer` is an expected-negative trial whose success predicate is a decisive change; it is never counted as a positive covariance trial.
- Density checks consume the frozen typed probe-pair inventory parsed from `h7_v1.json` and the scalar adapter, never probes selected or independently whitened at evaluation time. Every pair binds fixture/component/source identity, one shared original anchor and provenance, `x`, `x_prime=G_component @ x`, and separate expected initial, receiver-conditional, and global Jacobian shifts. The transformed point is not re-anchored or re-whitened under the transformed covariance.
- Where an independent evidence calculation exists, require evidence and posterior-KL invariance as well. The scalar `h1-v1` replay must retain its independent evidence-plus-posterior-KL identity. The matrix fixture must not fabricate an analytic evidence claim for the nonconjugate categorical emission; it reports that obligation as not applicable while still closing both complete ELBO paths.
- Production identity calculations use CPU float64. The independent oracle parses raw fixture bytes itself, imports no `vfe4`, PyTorch, NumPy, or production budget code, uses mpmath at exactly 100 decimal digits, enumerates exact source paths, and evaluates categorical-emission expectations at Gauss--Hermite orders 41 and 51.
- The required envelope is inclusive: every direct group element satisfies `||g_t||_2<=2` and `||g_t^{-1}||_2<=2`; every original and transformed SPD operand used by a required calculation satisfies `kappa_2<=1e3`. Record determinants, both group norms, SPD extreme eigenvalues, and condition numbers per operand. Never jitter, clip, pseudo-invert, regularize, resample, project, repair, or silently exclude a required operand.
- A required trial that is missing, stale, outside the envelope, has an unresolved 41/51 oracle comparison, or lacks an eligible predecessor is `INCONCLUSIVE`, not `FAIL`. A finite, valid, in-envelope scalar/positive covariance, density, or objective violation is `FAIL`. H7 is `PASS` only when every scalar-regression and positive-covariance trial satisfies its predicate, the expected-negative trial changes decisively, every tensor/law/local/complete-objective invariant and independent oracle comparison passes, and every injected control is decisive.
- Every numerical comparison owns a category-typed, operand-local budget. Its immutable operand records name the exact original/transformed/reference operands, shapes, value hashes, scales, condition numbers, normalizations, oracle values, operation counts, quadrature contributions, and individual allowance contributions. Backward recovery is recorded per operand before the trial maximum is aggregated. No pooled condition maximum, run-wide scale, or aggregate-only `H7BudgetRecord` is admissible.
- Every public fixture/law/objective/envelope/trial/control/gate record named below is concrete and immutable. Constructors defensively own tensors and mappings, recompute domain-separated canonical hashes excluding only their own integrity field, and validate PASS/FAIL/INCONCLUSIVE consistency. `H7GateResult` is defined only in `vfe4/types/results.py` and re-exported exactly once from `vfe4/types/__init__.py`; `vfe4/types/h7.py` owns its auxiliary records but no competing result class.
- The complete negative-control inventory is fixed and ordered by the exact IDs `wrong_covariance_congruence`, `wrong_precision_congruence`, `history_scorer_wrong_source_inverse`, `reversed_link_order`, `reverse_arrow_B`, `wrong_decoder_dual_action`, `fixed_decoder_outside_stabilizer`, `omitted_density_jacobian`, `reversed_logdet_sign`, `entropy_false_invariance`, `changed_h1_source_probability`, and `diagonal_for_internal_action`. Source-support preservation is an additional exact required invariant with its own malformed-input tests. Every control must be decisive under its own matching operand-local budget; preregistration, tests, result records, and `validation/h7.json` use these IDs verbatim and in this order.
- `det(g)<0` is outside H7's declared `GL+(2)` domain. The parser and action constructor reject it before evaluation, and the artifact states that H7 makes no claim for the orientation-reversing component of `GL(2)`. Do not count rejection of a reflection as a covariance PASS over full `GL(2)`.
- Each implementation task runs only its named new or directly modified focused tests for RED/GREEN, then creates exactly one bounded commit. A task reviewer inspects that diff and the implementer's focused output without rerunning the same tests. Do not run a cumulative suite after each task.
- After every tracked H7 source/test/config/fixture/preregistration/schema edit and bounded review is complete, freeze one exact clean H7 candidate revision and prohibit every later tracked edit or commit in that candidate lifecycle. Run the full pytest suite with JUnit exactly once first. At that same revision and dirty-content digest, produce exactly once and in order: the current H1--H5 compatibility artifact/ledger for H7; the H1-prefix-prior artifact/ledger only when the frozen H7 scorer profile is consumed (the required `h7-linear-history-source-v1` profile activates it); the independently projected current H6-Prefix certificate set/artifact/ledger with `predecessor_refs={}`; and finally the H7 artifact/ledger. H1--H5, H1-prefix-prior, and H6-Prefix become sibling H7 inputs; neither H1--H5 nor H4 is a predecessor of H6-Prefix. If a later review discovers a source defect, preserve all evidence as history, commit a new candidate, and repeat the complete lifecycle with one replacement JUnit; never patch the frozen candidate or rerun for confidence.
- H7 references the current predecessor artifact, manifest, payload/certificate-set, and ledger hashes and never copies their payloads or reruns them inside the H7 click operation. H6-Prediction is not required unless an empirical checkpoint is explicitly added as a new H7 trial; the frozen reference H7-v1 protocol adds none.
- Preserve `.verification/ledger.json` and every historical revision-specific ledger byte-for-byte. Task 9 creates separate current-candidate predecessor ledgers in their own scopes and H7 claims close only in `.verification/h7-<FULL_HEAD>-<FIXTURE_SET_SHA>-ledger.json`; H7 references predecessor ledger hashes instead of duplicating their claims. An existing `.verification/active.json` blocks activation; never delete, overwrite, or repoint it manually.
- Preserve one editable `CONFIG` in `verify_vfe4.py`, one `main`, one script guard, and no required CLI. H7 extends the verification surface only through H7. Do not add H8 keys, a second verifier, `argparse`, Typer, Hydra, a required environment variable, notebook, or dashboard.

## Normative Sources and Read-Only Context

- H7 protocol and decision rules: `Manuscripts/vfe4_whitepaper/08_hypotheses_limitations.tex`, H7 paragraph.
- Population torsor, direct frames, link/morphism laws, and coboundary order: `Manuscripts/vfe4_whitepaper/03_bundle_geometry.tex`, especially `population-frame-torsor`, `link-morphism-gauge-laws`, `regime-i-cocycle-identity`, and `regime-i-loop-holonomy`.
- Transition offsets, receiver covariances, density Jacobians, decoder dual action, and strict fixed-readout stabilizer: `Manuscripts/vfe4_whitepaper/04_generative_model.tex`, especially `transition-offset-covariance-gauge-laws`, `receiver-density-kernel-measure-invariance`, `dual-decoder-gauge-laws`, and `fixed-decoder-stabilizer`.
- Joint `(h,J)/(mu,Sigma,M)` transformations and entropy shift: `Manuscripts/vfe4_whitepaper/05_structured_information_form.tex`, especially `joint-gaussian-gauge-laws`, `joint-density-log-normalizer-jacobian`, and `entropy-shift-kl-invariance`.
- Complete `p/q` pushforwards, ELBO/evidence/KL identity, and the exact centered-softmax fixed-decoder stabilizer: `Manuscripts/vfe4_whitepaper/06_elbo_coordinate_updates.tex`, especially `complete-generative-density-pushforward` through `fixed-decoder-emission-kernel-stabilizer`.
- Independent derivation and type/cocycle oracles: `Manuscripts/vfe4_whitepaper/09_appendices.tex`, especially the population-frame pushforward, population-copy dimension/type oracle, and expanded coboundary composition check.
- Approved architecture and one-way dependency rules: `docs/superpowers/specs/2026-07-21-vfe4-codebase-design.md`.
- Predecessor plans and live seams: the approved H2/H3 plans plus the implemented H2 factor/objective code, and the current H4/H5 and H6 plans. H7 preserves the H2 detached factor seam, consumes the H5 complete-objective trace, and snapshots the post-H6 language model without mutating it.
- Research wiki, read only: `[[Gauge transformation]]`, `[[Parallel transport]]`, `[[Symmetric spaces and the SPD cone]]`, `[[Variational free energy]]`, and `[[VFE Transformer Program]]`. These support the frame/covariance distinction and nonclaims; they are not executable H7 evidence.

## Frozen H7-v1 Laws and Trial Matrix

### Scalar replay

The scalar replay consumes the unchanged raw `vfe4/validation/fixtures/h1_v1.json` bytes and existing four exact source paths. It adds no new scalar model parameters. Its positive direct elements are `GL+(1,R)` values wrapped in `H7ScalarReplayAction`; this is a complete-law/source/evidence/KL regression and is never counted as a `GL+(2)` matrix trial.

- diagonal base trial: `g0=g1=g2=1.25`;
- internal product trial: `(g0,g1,g2)=(0.8,1.1,1.4)`;
- both trials transform frames, links, initial and transition laws, recognition kernels, decoder weights, coordinate densities, entropy, local terms, monolithic ELBO, evidence, and posterior KL;
- the scalar replay is the decisive source-probability/support control because `h1-v1` has nontrivial positive source tables.

### Matrix fixture

Task 1 writes the following exact JSON values before any H7 calculation. The file is `vfe4/validation/fixtures/h7_v1.json`; JSON field order and bytes are immutable after its raw SHA-256 is frozen.

```json
{
  "fixture_schema_version": 1,
  "fixture_id": "h7-v1",
  "group": "GL+(2,R)",
  "representations": {"state": "standard", "model": "standard"},
  "horizon": 2,
  "dimensions": {"d_z": 2, "d_m": 2, "D": 12, "V": 3},
  "continuous_order": ["z0[0]", "z0[1]", "m0[0]", "m0[1]", "z1[0]", "z1[1]", "m1[0]", "m1[1]", "z2[0]", "z2[1]", "m2[0]", "m2[1]"],
  "state_parent_sets": [[0], [1]],
  "model_parent_sets": [[0], [1]],
  "state_source_support": [[0], [1]],
  "model_source_support": [[0], [1]],
  "observation_label_base": 0,
  "observation_labels": [0, 2],
  "frame_profiles": {
    "identity": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    "nonidentity": [[[1.0, 0.0], [0.0, 1.0]], [[1.1, 0.15], [-0.05, 0.95]], [[0.9, -0.1], [0.2, 1.05]]]
  },
  "actions": {
    "diagonal": [[[1.2, 0.2], [-0.1, 0.9]], [[1.2, 0.2], [-0.1, 0.9]], [[1.2, 0.2], [-0.1, 0.9]]],
    "internal": [[[1.25, 0.1], [0.05, 0.95]], [[0.85, -0.2], [0.1, 1.15]], [[1.05, 0.25], [-0.15, 0.9]]],
    "fixed_decoder_stabilizer": [[[1.0, 0.0], [0.2, 1.1]], [[1.0, 0.0], [0.2, 1.1]], [[1.0, 0.0], [0.2, 1.1]]]
  },
  "generative": {
    "initial_mean": [0.2, -0.1, 0.15, 0.05],
    "initial_covariance": [[0.9, 0.12, 0.08, -0.03], [0.12, 0.75, 0.02, 0.04], [0.08, 0.02, 0.8, 0.1], [-0.03, 0.04, 0.1, 0.7]],
    "model_source_probabilities": [[1.0], [1.0]],
    "state_source_probabilities": [[1.0], [1.0]],
    "source_scorer_profile": {
      "profile_id": "h7-linear-history-source-v1",
      "law": "alpha(prefix)+r_z^T z_j+r_m^T m_j",
      "prefix_tokens": [0, 2],
      "alpha_bias": {"model": [0.17, -0.09], "state": [-0.04, 0.13]},
      "alpha_token_scale": {"model": [0.03, -0.02], "state": [0.01, 0.025]},
      "z_history": [[0.3, -0.2], [-0.15, 0.25]],
      "m_history": [[0.1, 0.4], [0.2, -0.3]],
      "r_z": {
        "model": [[0.2, -0.15], [-0.1, 0.25]],
        "state": [[-0.12, 0.09], [0.16, 0.07]]
      },
      "r_m": {
        "model": [[0.05, 0.18], [0.22, -0.08]],
        "state": [[0.14, -0.06], [-0.05, 0.19]]
      }
    },
    "model_offsets": [[0.1, -0.05], [-0.08, 0.12]],
    "model_receiver_covariances": [[[0.5, 0.07], [0.07, 0.4]], [[0.45, -0.05], [-0.05, 0.55]]],
    "state_offsets": [[0.04, 0.09], [-0.06, 0.03]],
    "state_receiver_covariances": [[[0.38, 0.04], [0.04, 0.42]], [[0.52, 0.06], [0.06, 0.47]]],
    "B": [[[0.35, -0.1], [0.2, 0.25]], [[-0.15, 0.3], [0.1, 0.4]]],
    "decoder": [
      {"W_z": [[-0.1, -0.2], [0.4, -0.2], [0.6, -0.2]], "W_m": [[0.05, 0.25], [-0.5, 0.25], [0.0, 0.25]], "bias": [0.05, -0.08, 0.03]},
      {"W_z": [[0.0, 0.1], [-0.35, 0.1], [-0.4, 0.1]], "W_m": [[-0.1, -0.15], [0.6, -0.15], [0.1, -0.15]], "bias": [-0.02, 0.11, -0.09]}
    ]
  },
  "recognition": {
    "family_id": "structured-full-block-v1",
    "initial_mean": [-0.05, 0.12, 0.2, -0.08],
    "initial_covariance": [[0.72, 0.06, 0.04, 0.01], [0.06, 0.68, -0.02, 0.05], [0.04, -0.02, 0.74, 0.08], [0.01, 0.05, 0.08, 0.66]],
    "model_source_probabilities": [[1.0], [1.0]],
    "state_source_probabilities_given_model_source": [[[1.0]], [[1.0]]],
    "model_parent_maps": [[[0.82, 0.1], [-0.05, 0.9]], [[0.88, -0.08], [0.12, 0.84]]],
    "model_offsets": [[-0.04, 0.07], [0.03, -0.09]],
    "model_receiver_covariances": [[[0.6, 0.05], [0.05, 0.48]], [[0.55, -0.04], [-0.04, 0.5]]],
    "state_parent_maps": [[[0.78, 0.06], [-0.09, 0.86]], [[0.81, 0.11], [0.04, 0.79]]],
    "state_model_maps": [[[0.22, -0.05], [0.13, 0.18]], [[-0.1, 0.2], [0.07, 0.26]]],
    "state_offsets": [[0.06, -0.02], [-0.05, 0.08]],
    "state_receiver_covariances": [[[0.5, 0.03], [0.03, 0.46]], [[0.58, 0.02], [0.02, 0.44]]],
    "factorized_fixture": {
      "family_id": "factorized-diagonal-within-fiber-v1",
      "representation": "factorized_diagonal_within_fiber",
      "shared_fields": ["model_source_probabilities", "state_source_probabilities_given_model_source", "model_parent_maps", "model_offsets", "state_parent_maps", "state_model_maps", "state_offsets"],
      "initial_mean": [-0.05, 0.12, 0.2, -0.08],
      "initial_diagonal_covariance": [0.72, 0.68, 0.74, 0.66],
      "model_receiver_diagonal_covariances": [[0.6, 0.48], [0.55, 0.5]],
      "state_receiver_diagonal_covariances": [[0.5, 0.46], [0.58, 0.44]],
      "generic_gl_plus_2_output_representation": "unrestricted_full_block_pushforward"
    }
  },
  "density_probes": {
    "probe_set_schema": "h7-density-probe-pairs-v1",
    "whitened_scale": 0.25,
    "anchor_policy": "original_component_mean",
    "anchor_provenance": "raw_fixture_component_mean_and_lower_cholesky_v1",
    "pair_law": "x=anchor+L@(scale*direction);x_prime=G_component@x",
    "direction_ids_by_dimension": {
      "2": ["zero", "+e0", "-e0", "+e1", "-e1"],
      "4": ["zero", "+e0", "-e0", "+e1", "-e1", "+e2", "-e2", "+e3", "-e3"],
      "12": ["zero", "+e0", "-e0", "+e1", "-e1", "+e2", "-e2", "+e3", "-e3", "+e4", "-e4", "+e5", "-e5", "+e6", "-e6", "+e7", "-e7", "+e8", "-e8", "+e9", "-e9", "+e10", "-e10", "+e11", "-e11"]
    },
    "components": [
      {"component_id": "p.initial_joint", "source_id": "initial", "dimension": 4, "shift_scope": "initial_joint"},
      {"component_id": "p.model.receiver_1", "source_id": "model:1<-0", "dimension": 2, "shift_scope": "receiver_model"},
      {"component_id": "p.model.receiver_2", "source_id": "model:2<-1", "dimension": 2, "shift_scope": "receiver_model"},
      {"component_id": "p.state.receiver_1", "source_id": "state:1<-0", "dimension": 2, "shift_scope": "receiver_state"},
      {"component_id": "p.state.receiver_2", "source_id": "state:2<-1", "dimension": 2, "shift_scope": "receiver_state"},
      {"component_id": "q.structured.initial_joint", "source_id": "initial", "dimension": 4, "shift_scope": "initial_joint"},
      {"component_id": "q.structured.model.receiver_1", "source_id": "model:1<-0", "dimension": 2, "shift_scope": "receiver_model"},
      {"component_id": "q.structured.model.receiver_2", "source_id": "model:2<-1", "dimension": 2, "shift_scope": "receiver_model"},
      {"component_id": "q.structured.state.receiver_1", "source_id": "state:1<-0", "dimension": 2, "shift_scope": "receiver_state"},
      {"component_id": "q.structured.state.receiver_2", "source_id": "state:2<-1", "dimension": 2, "shift_scope": "receiver_state"},
      {"component_id": "q.factorized.initial_joint", "source_id": "initial", "dimension": 4, "shift_scope": "initial_joint"},
      {"component_id": "q.factorized.model.receiver_1", "source_id": "model:1<-0", "dimension": 2, "shift_scope": "receiver_model"},
      {"component_id": "q.factorized.model.receiver_2", "source_id": "model:2<-1", "dimension": 2, "shift_scope": "receiver_model"},
      {"component_id": "q.factorized.state.receiver_1", "source_id": "state:1<-0", "dimension": 2, "shift_scope": "receiver_state"},
      {"component_id": "q.factorized.state.receiver_2", "source_id": "state:2<-1", "dimension": 2, "shift_scope": "receiver_state"},
      {"component_id": "p.global", "source_id": "matrix-singleton-path", "dimension": 12, "shift_scope": "global"},
      {"component_id": "q.structured.global", "source_id": "matrix-singleton-path", "dimension": 12, "shift_scope": "global"},
      {"component_id": "q.factorized.global", "source_id": "matrix-singleton-path", "dimension": 12, "shift_scope": "global"}
    ]
  },
  "oracle": {"decimal_precision": 100, "gauss_hermite_orders": [41, 51]}
}
```

For every frame profile, generative parent maps are the represented direct coboundaries `Omega_tj=U_t U_j^{-1}` for the exact chain `0->1->2`. The parser also constructs all six ordered pair links solely for the cocycle/open-walk/closed-walk audit; those extra relation links are not generative factors. Receiver precisions are derived by checked linear solves from the frozen covariances and compared with independently transformed precision laws; they are not serialized as redundant adjustable inputs.

The source-scorer profile has bank order `(model,state)`, receiver order `t=(1,2)`, and the sole permitted source `j=t-1` in each frozen support row. For a bank `b`, `alpha_{b,t,j}(x_{<t})=a_{b,t}+c_{b,t}*sum_{ell=1}^{t} ell*(x_{ell-1}+1)` using the serialized `alpha_bias=a`, `alpha_token_scale=c`, and the first `t` entries of `prefix_tokens`; this arithmetic and the prefix bytes do not transform. The history vectors transform as `z_j'=g_j z_j` and `m_j'=g_j m_j`, while each serialized covector transforms by the source frame as `r_z'=solve(g_j^T,r_z)` and `r_m'=solve(g_j^T,r_m)`. For every bank/receiver row, H7 records separate covector-law and raw-score residuals before checking the unchanged singleton mask/probability. The `history_scorer_wrong_source_inverse` control runs under `matrix-nonidentity-internal-transformed` and deliberately uses the receiver inverse `g_t^{-T}` instead of `g_j^{-T}`; its raw-score residual, not the trivially unit singleton probability, must cross the control limit.

The parent `recognition` record is the structured full-block fixture. The nested factorized fixture shares only the seven explicitly named non-covariance fields and carries its own mean and every diagonal covariance value; no unnamed field is inherited. Its original covariance representation is diagonal within each state/model fiber. For a generic non-diagonal direct element, the parser/action layer constructs the full congruence and returns an unrestricted `H7RecognitionTensorLaw` tagged with the factorized origin. Tests must witness a nonzero transformed off-diagonal entry and reject diagonal projection, same-family reconstruction, or a claim that `FactorizedLanguageRecognition` is closed under the trial group.

The density-probe JSON is an ordered finite inventory, not a runtime recipe. During Task 1's one-time preregistration freeze, after the raw fixture and preregistration skeleton exist but before any H7 trial, objective, or oracle calculation, expand each listed component against the listed direction IDs in exact order, interpret `+ei/-ei` as the signed coordinate basis in the declared dimension, form `x=anchor+L@(0.25*direction)` from the one recorded original anchor/Cholesky provenance, and bind `x_prime=G_component@x`. Canonically serialize and hash the expanded `H7DensityProbePair` tuple, then copy that exact pair-table hash into the preregistration and `H7ValidationConfig`. The ordinary runtime parser only reconstructs the preregistered table and verifies its canonical bytes/hash; evaluation may consume only that frozen tuple. Neither parser nor evaluator may choose new directions, independently whiten either side, recompute a transformed whitening, or substitute a transformed anchor. The scalar H1 adapter follows the same one-time freeze for each of its four ordered source paths and binds the raw H1 fixture SHA as provenance.

The centered decoder rows deliberately have rank two after concatenating state and model columns, while every centered row reads only the first coordinate in each channel. The nonidentity direct element `[[1,0],[0.2,1.1]]` therefore belongs to the centered-softmax emission stabilizer but not generally to the strict raw-readout stabilizer because the row-common linear functional changes. The diagonal matrix `[[1.2,0.2],[-0.1,0.9]]` lies outside that centered stabilizer and is the held-fixed-decoder negative trial.

### Required positive and negative trials

| Trial ID | Law/frame profile | Action | Decoder | Closed role | Expected predicate |
|---|---|---|---|---|---|
| `scalar-base-transformed` | unchanged `h1-v1` | `GL+(1)` scalar base `1.25` | transform | `scalar_regression` | `complete_covariance` including sources and evidence/KL; no `GL+(2)` evidence. |
| `scalar-internal-transformed` | unchanged `h1-v1` | `GL+(1)` scalars `(0.8,1.1,1.4)` | transform | `scalar_regression` | `complete_covariance`; no `GL+(2)` evidence. |
| `matrix-identity-base-transformed` | `h7-v1`, identity `U` | frozen diagonal matrix | transform | `positive_covariance` | `complete_covariance`. |
| `matrix-identity-internal-transformed` | `h7-v1`, identity `U` | frozen internal matrices | transform | `positive_covariance` | `complete_covariance`. |
| `matrix-nonidentity-base-transformed` | `h7-v1`, nonidentity `U` | frozen diagonal matrix | transform | `positive_covariance` | `complete_covariance`. |
| `matrix-nonidentity-internal-transformed` | `h7-v1`, nonidentity `U` | frozen internal matrices | transform | `positive_covariance` | `complete_covariance` for both recognition-family origins. |
| `matrix-fixed-decoder-centered-stabilizer` | `h7-v1`, nonidentity `U` | frozen stabilizer element | fixed | `positive_covariance` | `centered_decoder_stabilizer_invariance`; raw logits may differ only by a row-common scalar. |
| `matrix-fixed-decoder-outside-stabilizer` | `h7-v1`, nonidentity `U` | frozen diagonal matrix | fixed | `expected_negative` | `decisive_outside_stabilizer_change` in emission and complete objective. |

Every `H7TrialSpec` owns the exact action elements parsed from the frozen fixture (or the two scalar literals), their raw-byte identities, and one domain-separated `action_sha256`. Every transformed-decoder positive trial evaluates both original and transformed laws. The expected-negative trial succeeds only on its decisive-change predicate and cannot satisfy the positive trial inventory. Every control is injected into a fresh transformed copy; controls never alter the correct production transformation or share mutated state.

## File Map and Dependency Boundaries

| Path | Responsibility |
|---|---|
| `vfe4/types/h7.py` | Borrowed tensor/action identity views; owned tensor, action, source/scorer, generative, recognition, probe, objective, envelope, trial/control, budget, predecessor, and gate-evaluation records with canonical integrity hashes. It does not define `H7GateResult`. |
| `vfe4/types/results.py` | Sole owner of the immutable fail-closed `H7GateResult`; `vfe4/types/__init__.py` re-exports it once. |
| `vfe4/validation/fixtures/h7_v1.json` | Frozen full-matrix law, frame profiles, direct actions, decoders, and oracle settings. |
| `vfe4/validation/h7_fixture.py` | Strict raw-byte parser, optional-H1 byte adapter, both recognition-family fixtures, matrix complete-law builder, frame-profile builder, frozen typed-probe expansion, and post-H6 snapshot parity checks. |
| `vfe4/geometry/group_action.py` | Pure differentiable direct-group tensor transformations using left/right solves without materialized inverses, block assembly, log-Jacobian, frame/link composition, and centered-softmax projector. |
| `vfe4/geometry/__init__.py` | Public export of the H7 tensor-action seam only. |
| `vfe4/generative/pushforward.py` | Complete borrowed tensor-law view plus owned generative evidence snapshot: frames, links, maps, `B`, offsets, covariance/precision, typed linear-history scorer rows, decoder, and density-shift metadata. |
| `vfe4/generative/language.py` | Export the existing post-H6 concrete normalized model as `LanguageGenerativeModel` if that exact public name is absent; no factor, parameter, or arithmetic change. |
| `vfe4/recognition/pushforward.py` | Complete borrowed tensor-law view plus owned evidence snapshots for exactly `StructuredLanguageRecognition` and `FactorizedLanguageRecognition`; generic actions promote factorized inputs to unrestricted full-block tensor laws without projection or same-family reconstruction. |
| `vfe4/recognition/language.py` | Read-only owner of the exact post-H6 `StructuredLanguageRecognition` and `FactorizedLanguageRecognition` concrete types. H7 adds no alias and makes no arithmetic or conditioning edit here. |
| `vfe4/objective/h7_covariance.py` | Calls the existing complete local objective, binds joint `K0`, evaluates the frozen corresponding density-probe pairs, adds the independent monolithic expectation, and emits immutable term/density/entropy/evidence/KL diagnostics without defining a new training objective. |
| `verification/mp_oracles/h7_covariance.py` | Independent JSON/mpmath-only 100-decimal exact-source oracle with analytic Gaussian reductions and GH41/GH51 emission expectations. |
| `verification/h7_budget.py` | Frozen category- and operand-typed forward, comparison, per-operand backward, oracle/quadrature, and control-decisiveness allowances. |
| `verification/h7_gate.py` | Predecessor validation, fixture capture, trial execution, envelope checks, oracle comparisons, controls, status precedence, and `validation/h7.json`. |
| `vfe4/config/schema.py` | Frozen `H7ValidationConfig` and predecessor-reference section. |
| `vfe4/config/resolve.py` | Exact H7 literals, ordered prefix through H7, canonical hashing, and rejection of H8/det-negative/unsupported profiles. |
| `verification/run_gates.py` | The H7-owned pure current-candidate H1--H5 projection, the H6-owned conditional H1-prefix-prior and independent H6-Prefix projectors/producer, and the H7-only selected operation after predecessor-reference validation; one-time H1/H7 byte capture and atomic H7 publication. |
| `vfe4/artifacts/provenance.py` | H7 source/config/fixture/predecessor/action/oracle/budget/dependency-closure provenance. |
| `verify_vfe4.py` | The one editable verification mapping extended only through H7. |
| `docs/preregistrations/2026-07-21-h7-frame-covariance.md` | Frozen laws, trial matrix, transformations, budgets, controls, statuses, artifact schema, and nonclaims. |
| `tests/unit/test_h7_fixture.py` | Raw schema, exact bytes, types, SPD/group validation, det-negative rejection, and post-H6 snapshot parity. |
| `tests/unit/test_h7_group_action.py` | Tensor laws, link/cocycle/order, centered projector, direct-element/no-exp rule, and autograd preservation. |
| `tests/unit/test_h7_generative_pushforward.py` | Every generative factor, fixed/history source rule, decoder policies, density shifts, and nonmutation. |
| `tests/unit/test_h7_recognition_pushforward.py` | Every recognition factor, source probabilities, conditional covariance/precision, entropy shift, and nonmutation. |
| `tests/unit/test_h7_complete_objective.py` | Local/monolithic term completeness, density probes, log ratio, complete ELBO, evidence/KL applicability, and no partial-pass path. |
| `tests/oracle/test_h7_mp_oracle.py` | Independent 100-decimal law values, exact sources, GH convergence, and production comparisons. |
| `tests/promotion/test_h7_gate.py` | Required trials, all controls, envelope/status precedence, predecessor freshness, and payload. |
| `tests/unit/test_config.py` | Ordered H7 typed config, compatibility prefixes, and no H8 support. |
| `tests/unit/test_atomic_artifacts.py` | H7 reference/manifest/no-overwrite and predecessor-link preservation. |
| `tests/integration/test_verify_vfe4.py` | One click-run publishes the H7 artifact without rerunning/copying predecessor payloads. |

Dependency direction remains `config + types -> numerics/geometry -> generative -> recognition/objective -> verification/runner/artifacts`. Production `vfe4` never imports `verification` or `tests`. `verification/mp_oracles/h7_covariance.py` imports only Python standard-library modules and mpmath. The pure geometry module imports only PyTorch and H7 tensor-facing types; it does not import generative, recognition, objective, verification, or artifact modules.

## Public Interfaces Frozen by This Plan

```python
# vfe4/types/h7.py
H7ActionKind = Literal["diagonal_base", "internal_product"]
H7DecoderPolicy = Literal["transform", "fixed"]
H7FrameProfile = Literal["identity", "nonidentity", "h1_v1"]
H7RecognitionFamily = Literal[
    "structured_full_block", "factorized_diagonal_within_fiber"
]
H7RecognitionRepresentation = Literal[
    "structured_full_block", "factorized_diagonal_within_fiber",
    "unrestricted_full_block_pushforward",
]
H7TrialRole = Literal["scalar_regression", "positive_covariance", "expected_negative"]
H7ExpectedPredicate = Literal[
    "complete_covariance", "centered_decoder_stabilizer_invariance",
    "decisive_outside_stabilizer_change",
]
H7TrialId = Literal[
    "scalar-base-transformed", "scalar-internal-transformed",
    "matrix-identity-base-transformed", "matrix-identity-internal-transformed",
    "matrix-nonidentity-base-transformed", "matrix-nonidentity-internal-transformed",
    "matrix-fixed-decoder-centered-stabilizer",
    "matrix-fixed-decoder-outside-stabilizer",
]
H7ControlId = Literal[
    "wrong_covariance_congruence", "wrong_precision_congruence",
    "history_scorer_wrong_source_inverse", "reversed_link_order",
    "reverse_arrow_B", "wrong_decoder_dual_action",
    "fixed_decoder_outside_stabilizer", "omitted_density_jacobian",
    "reversed_logdet_sign", "entropy_false_invariance",
    "changed_h1_source_probability", "diagonal_for_internal_action",
]
H7SourceBank = Literal["model", "state"]
H7Channel = Literal["z", "m"]
H7BudgetCategory = Literal[
    "vector", "information", "offset", "decoder", "covariance", "precision",
    "second_moment", "map", "cocycle", "density", "local_term",
    "complete_objective", "backward",
]
H7OperandRole = Literal["original", "transformed", "reference", "recovered", "oracle"]
H7OperationKind = Literal[
    "exact_identity", "direct_solve", "matrix_product", "quadratic_form",
    "logdet", "analytic_density", "gauss_hermite", "pair_comparison",
]
H7AllowanceKind = Literal[
    "operation_rounding", "quadrature_convergence", "reference_rounding"
]

@dataclass(frozen=True)
class H7RawTensorIdentity:
    object_id: int
    storage_data_ptr: int
    storage_version: int
    dtype: str
    shape: tuple[int, ...]
    device: str
    contiguous: bool
    requires_grad: bool

@dataclass(frozen=True)
class H7BorrowedTensorView:
    tensor: torch.Tensor             # exact caller object; never serialized
    identity: H7RawTensorIdentity

@dataclass(frozen=True, init=False)
class H7OwnedTensorSnapshot:
    __owned: torch.Tensor            # private contiguous clone
    capture_identity: H7RawTensorIdentity
    owned_storage_version: int
    dtype: str
    shape: tuple[int, ...]
    device: str
    raw_bytes: bytes
    raw_bytes_sha256: str
    snapshot_sha256: str

    @classmethod
    def capture(cls, value: torch.Tensor) -> "H7OwnedTensorSnapshot": ...
    def value(self) -> torch.Tensor: ...       # fresh clone
    def assert_intact(self) -> None: ...

@dataclass(frozen=True)
class H7BorrowedActionView:
    elements: tuple[H7BorrowedTensorView, H7BorrowedTensorView, H7BorrowedTensorView]
    kind: H7ActionKind
    dimension: Literal[1, 2]
    group: Literal["GL+(1,R)", "GL+(2,R)"]

@dataclass(frozen=True)
class H7ScalarReplayAction:
    elements: tuple[H7OwnedTensorSnapshot, H7OwnedTensorSnapshot, H7OwnedTensorSnapshot]
    kind: H7ActionKind
    dimension: Literal[1]
    group: Literal["GL+(1,R)"]
    representation: Literal["standard_scalar"]
    action_sha256: str

@dataclass(frozen=True)
class H7GLPlus2Action:
    elements: tuple[H7OwnedTensorSnapshot, H7OwnedTensorSnapshot, H7OwnedTensorSnapshot]
    kind: H7ActionKind
    dimension: Literal[2]
    group: Literal["GL+(2,R)"]
    representation: Literal["direct_gl_plus_2"]
    action_sha256: str

H7TensorActionSnapshot: TypeAlias = H7ScalarReplayAction | H7GLPlus2Action

@dataclass(frozen=True)
class H7TrialSpec:
    trial_id: H7TrialId
    role: H7TrialRole
    expected_predicate: H7ExpectedPredicate
    fixture_id: Literal["h1-v1", "h7-v1"]
    frame_profile: H7FrameProfile
    decoder_policy: H7DecoderPolicy
    action: H7TensorActionSnapshot
    action_sha256: str
    trial_sha256: str

@dataclass(frozen=True)
class H7HistoryValueView:
    channel: H7Channel
    population_label: int
    value: H7BorrowedTensorView

@dataclass(frozen=True)
class H7HistoryValueSnapshot:
    channel: H7Channel
    population_label: int
    value: H7OwnedTensorSnapshot
    history_sha256: str

@dataclass(frozen=True)
class H7SourceCovectorSnapshot:
    bank: H7SourceBank
    channel: H7Channel
    receiver_t: int
    source_j: int
    value: H7OwnedTensorSnapshot
    covector_sha256: str

@dataclass(frozen=True)
class H7SourceScorerRowView:
    bank: H7SourceBank
    receiver_t: int
    source_j: int
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    alpha_bias: float
    alpha_token_scale: float
    prefix_term: float
    z_history: tuple[H7HistoryValueView, ...]
    m_history: tuple[H7HistoryValueView, ...]
    z_covector: H7BorrowedTensorView
    m_covector: H7BorrowedTensorView
    mask: tuple[bool, ...]
    support: tuple[int, ...]
    raw_scores: H7BorrowedTensorView
    probabilities: H7BorrowedTensorView
    semantic_row_sha256: str

@dataclass(frozen=True)
class H7SourceContextView:
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    z_history: tuple[H7HistoryValueView, ...]
    m_history: tuple[H7HistoryValueView, ...]
    scorer_rows: tuple[H7SourceScorerRowView, ...]
    source_scorer_profile: Literal["h7-linear-history-source-v1"] | None
    semantic_context_sha256: str

@dataclass(frozen=True)
class H7SourceScorerRowSnapshot:
    bank: H7SourceBank
    receiver_t: int
    source_j: int
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    alpha_bias: float
    alpha_token_scale: float
    prefix_term: float
    z_history: tuple[H7HistoryValueSnapshot, ...]
    m_history: tuple[H7HistoryValueSnapshot, ...]
    z_covector: H7SourceCovectorSnapshot
    m_covector: H7SourceCovectorSnapshot
    mask: tuple[bool, ...]
    support: tuple[int, ...]
    raw_scores: H7OwnedTensorSnapshot
    probabilities: H7OwnedTensorSnapshot
    source_row_raw_bytes: bytes
    row_raw_bytes_sha256: str
    row_sha256: str

@dataclass(frozen=True)
class H7SourceContextSnapshot:
    prefix_tokens: tuple[int, ...]
    prefix_bytes: bytes
    prefix_bytes_sha256: str
    z_history: tuple[H7HistoryValueSnapshot, ...]
    m_history: tuple[H7HistoryValueSnapshot, ...]
    scorer_rows: tuple[H7SourceScorerRowSnapshot, ...]
    source_scorer_profile: Literal["h7-linear-history-source-v1"] | None
    source_scorer_sha256: str | None
    context_sha256: str

@dataclass(frozen=True)
class H7RecognitionContextSnapshot:
    observation_labels: tuple[int, ...]
    conditioning: Literal["filtering", "smoothing"]
    context_sha256: str

@dataclass(frozen=True)
class H7GaussianComponentSnapshot:
    component_id: str
    receiver_t: int | None
    source_j: int | None
    mean: H7OwnedTensorSnapshot
    covariance: H7OwnedTensorSnapshot
    precision: H7OwnedTensorSnapshot
    information_vector: H7OwnedTensorSnapshot
    second_moment: H7OwnedTensorSnapshot
    component_sha256: str

@dataclass(frozen=True)
class H7AffineComponentSnapshot:
    component_id: str
    bank: H7SourceBank
    receiver_t: int
    source_j: int
    parent_map: H7OwnedTensorSnapshot
    same_receiver_model_map: H7OwnedTensorSnapshot | None
    offset: H7OwnedTensorSnapshot
    receiver_law: H7GaussianComponentSnapshot
    component_sha256: str

@dataclass(frozen=True)
class H7TensorLawComponent:
    component_id: str
    receiver_t: int | None
    source_j: int | None
    tensors: Mapping[str, H7BorrowedTensorView]
    component_identity_sha256: str

@dataclass(frozen=True)
class H7DecoderSnapshot:
    receiver_t: int
    state_weight: H7OwnedTensorSnapshot
    model_weight: H7OwnedTensorSnapshot
    bias: H7OwnedTensorSnapshot
    centered_stabilizer_class: Literal["transformed", "inside", "outside"]
    decoder_sha256: str

@dataclass(frozen=True)
class H7GenerativeSnapshot:
    frames: tuple[H7OwnedTensorSnapshot, ...]
    ordered_links: Mapping[tuple[int, int], H7OwnedTensorSnapshot]
    initial_joint: H7GaussianComponentSnapshot
    transitions: tuple[H7AffineComponentSnapshot, ...]
    source_context: H7SourceContextSnapshot | None
    decoders: tuple[H7DecoderSnapshot, ...]
    support_sha256: str
    snapshot_sha256: str

@dataclass(frozen=True)
class H7GenerativeTensorLaw:
    components: tuple[H7TensorLawComponent, ...]
    source_context: H7SourceContextView | None
    decoder_policy: H7DecoderPolicy
    law_identity_sha256: str

@dataclass(frozen=True)
class H7RecognitionSnapshot:
    origin_family: H7RecognitionFamily
    representation: H7RecognitionRepresentation
    initial_joint: H7GaussianComponentSnapshot
    model_conditionals: tuple[H7AffineComponentSnapshot, ...]
    state_conditionals: tuple[H7AffineComponentSnapshot, ...]
    source_rows: tuple[H7SourceScorerRowSnapshot, ...]
    context: H7RecognitionContextSnapshot
    snapshot_sha256: str

@dataclass(frozen=True)
class H7RecognitionTensorLaw:
    origin_family: H7RecognitionFamily
    representation: H7RecognitionRepresentation
    components: tuple[H7TensorLawComponent, ...]
    source_rows: tuple[H7SourceScorerRowView, ...]
    context: H7RecognitionContextSnapshot
    law_identity_sha256: str

@dataclass(frozen=True)
class H7CompleteLawSnapshot:
    fixture_id: Literal["h1-v1", "h7-v1"]
    generative: H7GenerativeSnapshot
    recognition: H7RecognitionSnapshot
    raw_fixture_sha256: str
    snapshot_sha256: str

@dataclass(frozen=True)
class H7LawPairSnapshot:
    original: H7CompleteLawSnapshot
    transformed: H7CompleteLawSnapshot
    action_sha256: str
    law_pair_sha256: str

@dataclass(frozen=True)
class H7Fixture:
    fixture_id: Literal["h7-v1"]
    raw_fixture_sha256: str
    frame_profiles: Mapping[H7FrameProfile, tuple[H7OwnedTensorSnapshot, ...]]
    actions: Mapping[str, H7GLPlus2Action]
    generative: H7GenerativeSnapshot
    recognition_families: tuple[H7RecognitionSnapshot, H7RecognitionSnapshot]
    matrix_trial_specs: tuple[H7TrialSpec, ...]
    density_probe_pairs: tuple["H7DensityProbePair", ...]
    density_probe_set_sha256: str
    fixture_sha256: str

@dataclass(frozen=True)
class H7ValidationConfig:
    schema_version: Literal["h7-validation-config-v1"]
    required_trial_specs: tuple[H7TrialSpec, ...]
    required_control_ids: tuple[H7ControlId, ...]
    recognition_families: tuple[
        Literal["structured_full_block"],
        Literal["factorized_diagonal_within_fiber"],
    ]
    h1_fixture_raw_sha256: str
    h7_fixture_raw_sha256: str
    density_probe_set_sha256: str
    oracle_decimal_precision: Literal[100]
    gauss_hermite_orders: tuple[Literal[41], Literal[51]]
    group_norm_limit: float
    group_inverse_norm_limit: float
    spd_condition_limit: float
    predecessor_keys: tuple[Literal["h1_h5"], Literal["h1_prefix_prior"], Literal["h6_prefix"]]
    canonical_json: str
    config_sha256: str

@dataclass(frozen=True)
class H7DensityProbePair:
    probe_id: str
    fixture_id: Literal["h1-v1", "h7-v1"]
    component_id: str
    source_id: str
    action_sha256: str
    anchor_sha256: str
    anchor_provenance: str
    x: H7OwnedTensorSnapshot
    x_prime: H7OwnedTensorSnapshot       # exactly G_component @ x
    initial_log_jacobian_shift: float
    receiver_log_jacobian_shift: float
    global_log_jacobian_shift: float
    probe_sha256: str

@dataclass(frozen=True)
class H7OperandRecord:
    operand_id: str
    category: H7BudgetCategory
    role: H7OperandRole
    dtype: str
    shape: tuple[int, ...]
    value_sha256: str
    scale: float
    condition_number: float
    normalization: float
    oracle_value: str | None
    operand_sha256: str

@dataclass(frozen=True)
class H7AllowanceContribution:
    kind: H7AllowanceKind
    operation_id: str
    operation_kind: H7OperationKind
    operation_count: int
    quadrature_order: Literal[41, 51] | None
    unit_allowance: float
    value: float
    contribution_sha256: str

@dataclass(frozen=True)
class H7BudgetRecord:
    invariant_id: str
    category: H7BudgetCategory
    operands: tuple[H7OperandRecord, ...]
    contributions: tuple[H7AllowanceContribution, ...]
    comparison_normalization: float
    total_allowance: float
    budget_sha256: str

@dataclass(frozen=True)
class H7ResidualRecord:
    invariant_id: str
    category: Literal[
        "tensor", "law", "cocycle", "density", "jacobian", "source",
        "decoder", "local_term", "monolithic", "evidence", "posterior_kl",
        "absolute", "relative", "backward"
    ]
    value: float
    budget: H7BudgetRecord
    passed: bool
    residual_sha256: str

@dataclass(frozen=True)
class H7BackwardResidualRecord:
    operand_id: str
    original_sha256: str
    transformed_sha256: str
    recovered_sha256: str
    numerator: float
    normalization: float
    value: float
    budget: H7BudgetRecord
    passed: bool
    backward_sha256: str

@dataclass(frozen=True)
class H7EnvelopeOperandRecord:
    operand_id: str
    determinant: float | None
    norm_2: float | None
    inverse_norm_2: float | None
    minimum_eigenvalue: float | None
    maximum_eigenvalue: float | None
    condition_number_2: float
    within_envelope: bool
    record_sha256: str

@dataclass(frozen=True)
class H7EnvelopeRecord:
    group_operands: tuple[H7EnvelopeOperandRecord, ...]
    spd_operands: tuple[H7EnvelopeOperandRecord, ...]
    inclusive: Literal[True]
    passed: bool
    envelope_sha256: str

@dataclass(frozen=True)
class H7InitialJointKlRecord:
    term_id: Literal["K0_joint_z0_m0"]
    original_value: float
    transformed_value: float
    residual: H7ResidualRecord
    chain_decomposition: None
    record_sha256: str

@dataclass(frozen=True)
class H7LocalTermRecord:
    term_id: str
    original_value: float
    transformed_value: float
    signed_child_ids: tuple[str, ...]
    residual: H7ResidualRecord
    term_sha256: str

@dataclass(frozen=True)
class H7ObjectiveCovarianceEvaluation:
    initial_joint_kl: H7InitialJointKlRecord
    local_terms: tuple[H7LocalTermRecord, ...]
    density_probes: tuple[H7DensityProbePair, ...]
    scorer_residuals: tuple[H7ResidualRecord, ...]
    complete_local: H7ResidualRecord
    complete_monolithic: H7ResidualRecord
    log_ratio: H7ResidualRecord
    entropy_shift: H7ResidualRecord
    evidence: H7ResidualRecord | None
    posterior_kl: H7ResidualRecord | None
    not_applicable_reason: str | None
    evaluation_sha256: str

@dataclass(frozen=True)
class H7TrialResult:
    spec: H7TrialSpec
    observed_predicate: H7ExpectedPredicate
    predicate_satisfied: bool
    logabsdet_measure_shift: float
    r_abs: H7ResidualRecord
    r_rel: H7ResidualRecord
    backward_by_operand: tuple[H7BackwardResidualRecord, ...]
    r_back_max: H7ResidualRecord
    residuals: tuple[H7ResidualRecord, ...]
    envelope: H7EnvelopeRecord
    law_pairs_by_recognition_family: Mapping[
        H7RecognitionFamily, H7LawPairSnapshot
    ]
    objective_by_recognition_family: Mapping[
        H7RecognitionFamily, H7ObjectiveCovarianceEvaluation
    ]
    trial_result_sha256: str

@dataclass(frozen=True)
class H7ControlResult:
    control_id: H7ControlId
    target_invariant_id: str
    wrong_residual: float
    invariant_scale: float
    matching_correct_allowance: float
    decisiveness_limit: float
    detected: bool
    control_sha256: str

@dataclass(frozen=True)
class H7PredecessorReference:
    artifact_path: str
    git_head: str
    dirty_digest: str
    junit_sha256: str
    manifest_sha256: str
    payload_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    reference_sha256: str

@dataclass(frozen=True)
class H7GateEvaluation:
    result: "H7GateResult"
    validation_payload_canonical_json: bytes
    validation_payload_sha256: str
    fixture_set_sha256: str
    dependency_closure_sha256: str
    evaluation_sha256: str

@dataclass(frozen=True)
class H7PassOutcome:
    kind: Literal["PASS"]
    scalar_trial_ids: tuple[H7TrialId, H7TrialId]
    positive_trial_ids: tuple[H7TrialId, H7TrialId, H7TrialId, H7TrialId, H7TrialId]
    expected_negative_trial_id: H7TrialId
    control_ids: tuple[H7ControlId, ...]
    outcome_sha256: str

@dataclass(frozen=True)
class H7FailOutcome:
    kind: Literal["FAIL"]
    failed_invariant_ids: tuple[str, ...]
    expected_negative_false_acceptance: bool
    outcome_sha256: str

@dataclass(frozen=True)
class H7InconclusiveOutcome:
    kind: Literal["INCONCLUSIVE"]
    obligations: tuple[str, ...]
    outcome_sha256: str

H7GateOutcome: TypeAlias = H7PassOutcome | H7FailOutcome | H7InconclusiveOutcome

# vfe4/types/results.py -- sole definition; re-export once from vfe4/types/__init__.py
@dataclass(frozen=True)
class H7GateResult:
    gate: Literal["H7"]
    status: GateStatus
    fixture_hashes: Mapping[str, str]
    predecessor_references: Mapping[str, H7PredecessorReference]
    trials: tuple[H7TrialResult, ...]
    controls: tuple[H7ControlResult, ...]
    outcome: H7GateOutcome
    obligations: tuple[str, ...]
    result_sha256: str
```

Every owned record above recomputes a domain-separated canonical SHA-256 from all semantic fields except its own integrity field. `H7BorrowedTensorView` and `H7BorrowedActionView` are deliberately unhashed ephemeral graph views; they assert live object/storage/version metadata before every action and are forbidden from artifact/result schemas. Mapping fields are copied into sorted immutable mappings before hashing. Every published trial owns its exact original/transformed `H7LawPairSnapshot` values as well as its objective evaluations; hashes alone never substitute for the complete law evidence. `H7GateResult` is a separate explicit result type defined only in `results.py`, and its `status`, closed `H7GateOutcome` variant, top-level obligations, trial roles, and nested predicates must agree during construction. PASS requires `H7PassOutcome`, the exact two scalar-regression specs, five positive-covariance specs, one expected-negative spec, exact twelve control IDs, no obligations, all positive/scalar residuals and envelopes passing, and the expected-negative decisive-change predicate satisfied. FAIL requires `H7FailOutcome` plus at least one finite in-envelope failed positive/scalar invariant or a gate that incorrectly accepts the outside-stabilizer trial as covariant. INCONCLUSIVE requires `H7InconclusiveOutcome` and the same nonempty named obligations at both levels, including a missing/nondecisive expected-negative outcome, and cannot contain a claimed finite refutation unless that refutation is separately represented.

Every matrix `H7TrialResult.law_pairs_by_recognition_family` and `objective_by_recognition_family` mapping has exactly the ordered keys `("structured_full_block","factorized_diagonal_within_fiber")`; the second pair's transformed recognition snapshot is tagged `unrestricted_full_block_pushforward` for every non-diagonal primary action. Each scalar H1 regression has only its exact adapted H1 recognition-family key and cannot be used to satisfy this two-family matrix inventory.

```python
# vfe4/geometry/group_action.py
def require_direct_gl_plus(element: torch.Tensor, *, dimension: int) -> torch.Tensor: ...
def borrow_h7_action(
    elements: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *, kind: H7ActionKind, dimension: Literal[1, 2],
) -> H7BorrowedActionView: ...
def freeze_h7_action(action: H7BorrowedActionView) -> H7TensorActionSnapshot: ...
def block_population_action(action: H7BorrowedActionView) -> torch.Tensor: ...
def right_solve(value: torch.Tensor, right: torch.Tensor) -> torch.Tensor: ...
def push_vector(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_covariance(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_precision(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_information_vector(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_second_moment(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_receiver_source_map(value: torch.Tensor, receiver: torch.Tensor, source: torch.Tensor) -> torch.Tensor: ...
def push_same_receiver_morphism(value: torch.Tensor, state_receiver: torch.Tensor, model_receiver: torch.Tensor) -> torch.Tensor: ...
def push_decoder(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def compose_reframed_frames(action: H7BorrowedActionView, frames: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...
def frame_links(frames: tuple[torch.Tensor, ...]) -> Mapping[tuple[int, int], torch.Tensor]: ...
def logabsdet_measure_shift(action: H7BorrowedActionView) -> torch.Tensor: ...
def centered_logit_projector(vocabulary_size: int, *, like: torch.Tensor) -> torch.Tensor: ...
```

Every inverse action uses a solve applied directly to the actual operand. Right multiplication is exactly `right_solve(value,G)=torch.linalg.solve(G.T,value.T).T`; covectors use `torch.linalg.solve(G.T,value)`; a two-sided precision push is `right=torch.linalg.solve(G.T,J.T).T` followed by `torch.linalg.solve(G.T,right)`. Links, affine maps, `B`, decoders, precision laws, and backward recovery compose those right/left solves. Production must not call `torch.linalg.inv`/`pinv`, must not compute `torch.linalg.solve(G,I)`, and must not create an identity right-hand side solely to materialize an inverse. `require_direct_gl_plus` and `borrow_h7_action` return/view the exact caller tensors and record `H7RawTensorIdentity`; they do not clone or detach. Validation checks are differentiable comparisons/guards and may inspect detached scalar booleans only to reject invalid domains; returned tensors retain the original graph. Only `freeze_h7_action` clones into the owned evidence type, after action evaluation and before a result/artifact is constructed.

```python
# vfe4/generative/pushforward.py
def borrow_h7_generative(
    model: LanguageGenerativeModel,
    *,
    context: H7SourceContextView | None,
) -> H7GenerativeTensorLaw: ...
def pushforward_generative(
    law: H7GenerativeTensorLaw,
    action: H7BorrowedActionView,
    *,
    decoder_policy: H7DecoderPolicy,
) -> H7GenerativeTensorLaw: ...
def freeze_h7_generative(law: H7GenerativeTensorLaw) -> H7GenerativeSnapshot: ...

# vfe4/recognition/pushforward.py
H7RecognitionInput: TypeAlias = (
    StructuredLanguageRecognition | FactorizedLanguageRecognition
)
def borrow_h7_recognition(
    law: H7RecognitionInput,
    *,
    context: H7RecognitionContextSnapshot,
) -> H7RecognitionTensorLaw: ...
def pushforward_recognition(
    law: H7RecognitionTensorLaw,
    action: H7BorrowedActionView,
) -> H7RecognitionTensorLaw: ...
def freeze_h7_recognition(law: H7RecognitionTensorLaw) -> H7RecognitionSnapshot: ...
```

`LanguageGenerativeModel` is the normalized public model exported by post-H6 `vfe4/generative/language.py`. Recognition accepts the explicit read-only union of the exact post-H6 `StructuredLanguageRecognition` and `FactorizedLanguageRecognition` concrete classes; H7 must not invent a `LanguageRecognitionLaw` alias. Dispatch is by those exact classes (or overloads with those exact parameters), and rejects projection-tagged/moment-projected recognition records, emission-only objective records, arbitrary duck-typed objects, and unsupported dynamic factor types before borrowing. Borrowed tensor laws retain graph and caller identity and cannot enter a result or artifact. Pushforward always returns a new tensor law. A structured origin remains unrestricted full-block. A factorized origin is tagged `unrestricted_full_block_pushforward` after any generic non-diagonal matrix action; only the scalar/diagonal subgroup case may retain a diagonal representation after an explicit zero-off-diagonal check, and no H7 primary trial relies on that exception. `freeze_h7_*` is the only borrowed-to-owned boundary and captures complete cloned factor bytes/hashes after transformation. The H1 replay enters through the strict optional-byte adapter below, not through an `object`-typed branch. The original post-H6 model and its pre-action owned snapshot retain identical hashes.

```python
# vfe4/validation/h7_fixture.py
def adapt_optional_h1_fixture_bytes(
    data: bytes | None,
    *,
    required_scalar_trials: tuple[H7TrialId, ...],
) -> H7CompleteLawSnapshot | None: ...
```

`adapt_optional_h1_fixture_bytes` performs no file I/O. `data` must be present and match the frozen raw H1 SHA exactly when `required_scalar_trials` is the exact two scalar IDs; it must be absent when that tuple is empty. Any other tuple, unexpected bytes, missing required bytes, or supplied-but-unused bytes is rejected. H7-v1 uses the exact two scalar trials, so its H1 bytes are required and captured once by the runner.

```python
# vfe4/objective/h7_covariance.py
def evaluate_h7_complete_covariance(
    original: H7CompleteLawSnapshot,
    transformed: H7CompleteLawSnapshot,
    action: H7TensorActionSnapshot,
    *,
    quadrature_orders: tuple[int, int],
) -> H7ObjectiveCovarianceEvaluation: ...

# verification/h7_gate.py
def evaluate_h7(
    config: ResolvedConfig,
    *,
    h1_fixture_bytes: bytes,
    h7_fixture_bytes: bytes,
    predecessor_bytes: Mapping[str, bytes],
) -> H7GateEvaluation: ...
```

`evaluate_h7_complete_covariance` calls the existing post-H6 complete local objective and its factor trace. It binds the initial contribution once as `H7InitialJointKlRecord(term_id="K0_joint_z0_m0")`; the matrix structured fixture's cross-channel covariance makes a marginal-KL sum invalid. Its monolithic path independently evaluates `E_q[log p-log q]` from the same complete law and exact source paths. It consumes only the pre-frozen corresponding probe pairs. It never supplies a scalar used by training or H5 acceptance.

## Frozen Numerical Budget and Status Rules

Use the following exact functions in `verification/h7_budget.py` and copy them verbatim into the preregistration:

```python
EPS64 = 2.0 ** -52
ROUNDING_CONSTANT = 4096.0
MAX_ORACLE_RELATIVE_DELTA = 1.0e-18
CONTROL_MINIMUM_RELATIVE_RESIDUAL = 1.0e-8
CONTROL_ALLOWANCE_MULTIPLE = 100.0

def gamma_n(n: int) -> float:
    return (n * EPS64) / (1.0 - n * EPS64)

def rounding_allowance(operation_count: int, scale: float, condition_product: float) -> float:
    return ROUNDING_CONSTANT * gamma_n(operation_count) * condition_product * scale

def reference_rounding_allowance(reference_value: float) -> float:
    return 64.0 * EPS64 * max(1.0, abs(reference_value))

def control_decisiveness_limit(correct_allowance: float, scale: float) -> float:
    return max(
        CONTROL_ALLOWANCE_MULTIPLE * correct_allowance,
        CONTROL_MINIMUM_RELATIVE_RESIDUAL * scale,
    )
```

Each invariant constructs one `H7BudgetRecord(category,operands,contributions,comparison_normalization,...)`. There is no constructor that accepts only an aggregate scale/condition product. Every `H7OperandRecord` names the exact operand and role, its value hash/shape/dtype, its own scale and condition number, the normalization used by this comparison, and its oracle decimal string when applicable. The budget builder derives the category operation count from those typed shapes, records it in the `operation_rounding` contribution, adds a `quadrature_convergence` contribution only for the named GH-dependent operand, and adds reference rounding only for the named oracle value. Each invariant records its exact actual operation count from the frozen category formulas below; it may not borrow another invariant's scale, SPD condition number, oracle value/delta, or normalization.

| Category | Frozen operation count | Condition product | Scale operands |
|---|---:|---|---|
| vector/information/offset/decoder law in dimension `n` | `32*n+64` | product of the participating direct-element `kappa_2` values | only the two compared outputs and original operand |
| covariance/precision/second moment/map in dimension `n` | `64*n**3+128*n**2+64*n+256` | participating direct-element `kappa_2` times only the named SPD operand `kappa_2` when applicable | only compared matrices and source operand |
| cocycle/open/closed walk in dimension `n` | `96*n**3+128*n**2+256` | product of direct-element and frame `kappa_2` values on that walk | links, composite, expected endpoint map |
| one Gaussian density/probe in dimension `n` | `128*n**3+192*n**2+128*n+512` | only the density covariance/precision and direct block action conditions | log density, expected shifted log density, quadratic/logdet summands |
| one local ELBO term | sum of that term's child factor counts plus `32*k+64` for `k` signed summands | only SPD operands touched by that term | original/transformed term and its signed summands |
| complete monolithic/local ELBO | sum of its exact child counts plus `32*k+64` | only operands touched by that path | original/transformed complete value and exact signed term list |
| backward transform | forward count plus inverse-action count | direct-element condition product only | original, transformed, recovered operand |

For a high-precision comparison, `oracle_convergence_allowance=2*abs(value_51-value_41)` for a GH-dependent scalar and zero for analytic scalars. Require `abs(value_51-value_41) <= MAX_ORACLE_RELATIVE_DELTA*max(1,abs(value_51))`; otherwise the affected required trial is INCONCLUSIVE. Total comparison allowance is the sum of production rounding, oracle convergence, and reference rounding. An invariant with no oracle uses zero oracle/reference terms except the ordinary pair-comparison rounding for its two float64 outputs.

For each complete trial, calculate exactly:

```text
r_abs  = abs(L_prime-L)
r_rel  = r_abs/max(1,abs(L),abs(L_prime))
r_back_max = max_u ||T_G^-1(T_G(u))-u||_F/max(1,||u||_F)
```

The `r_back` operand inventory is exact: all `U`; all ordered-pair `Omega`; all initial/transition/recognition means, offsets, parent maps, `B`, covariances, precisions, `(h,J,M)` objects, decoder weights, and any history-reading source covectors. Construct and publish one `H7BackwardResidualRecord` per named operand with its original/transformed/recovered hashes, numerator, normalization, value, and own budget before computing `r_back_max`; the aggregate is only the maximum of that frozen tuple. Categorical probabilities, source support, observation labels, and biases are compared by exact identity and do not enter the normalized Frobenius maximum.

Status precedence is exact:

1. Validate source/config/fixture/predecessor identities and required inventory. Missing or stale input is INCONCLUSIVE.
2. Validate `GL+`, norm, SPD, and condition envelopes. A required out-of-envelope trial is INCONCLUSIVE; an invalid serialized fixture is INCONCLUSIVE because the intended law was not evaluated.
3. Validate the 100-decimal oracle and GH41/51 convergence. Missing or unresolved oracle evidence is INCONCLUSIVE.
4. Evaluate every scalar-regression and positive-covariance invariant. A finite, valid, in-envelope residual above its own allowance is FAIL.
5. Evaluate the expected-negative trial and every injected control against their matching `control_decisiveness_limit`. A finite wrong path not separated from the correct path is INCONCLUSIVE because the negative check is nondecisive; an expected-negative/control path that the production gate incorrectly accepts as covariant is FAIL.
6. PASS requires all current compatible predecessor references; both exact `role="scalar_regression"` specs satisfying `complete_covariance`; all five exact `role="positive_covariance"` matrix specs satisfying their role-appropriate covariance/stabilizer predicate; the sole `role="expected_negative"` outside-stabilizer spec satisfying `decisive_outside_stabilizer_change`; all twelve exact control IDs; all exact source/scorer identities; valid canonical hashes; and no obligation. The expected-negative trial never increments the positive-covariance count.

"Current" compatible predecessor evidence means artifacts and validated revision-specific ledgers produced exactly once *after the sole full JUnit* at the final frozen H7 `(git_head,dirty_digest)`: first one ordered H1--H5 PASS artifact/ledger; then a separate H1-prefix-prior PASS artifact/ledger only when the selected H7 snapshot consumes a prefix/history-conditioned generative scorer; then one H6-Prefix PASS artifact/certificate-set/ledger covering every post-H6 model/config/vocabulary identity consumed by H7. The required `h7-linear-history-source-v1` profile does consume that scorer, so the conditional H1-prefix-prior step is active for this preregistration. H7 validates and references those exact artifact/manifest/payload/certificate-set/ledger hashes and copies none of them. H6-Prediction is an empirical result and is not a mathematical prerequisite for the frozen reference laws; if an empirical H6 checkpoint is selected as a future additional H7 trial, its exact checkpoint/Prediction artifact becomes required for that added trial. The frozen H7-v1 gate selects no empirical checkpoint and makes no H6 predictive claim.

## Task 1: Freeze H7 Types, Matrix Fixture, Typed Configuration, and Preregistration

**Files:**

- Create: `vfe4/types/h7.py`
- Modify: `vfe4/types/__init__.py`
- Create: `vfe4/validation/h7_fixture.py`
- Modify: `vfe4/validation/__init__.py`
- Create: `vfe4/validation/fixtures/h7_v1.json`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Create: `tests/unit/test_h7_fixture.py`
- Modify: `tests/unit/test_config.py`
- Create: `docs/preregistrations/2026-07-21-h7-frame-covariance.md`
- Modify: `pyproject.toml`

**Interfaces:** Produce the auxiliary H7 records and parser named above, including raw tensor identity, borrowed tensor/action views, owned tensor/action snapshots, exact trial specs/roles, typed source/scorer rows, both recognition-family fixture snapshots, frozen density-probe pairs, operand/budget/envelope/objective records, the disjoint `H7ScalarReplayAction | H7GLPlus2Action` owned union, `parse_h7_fixture_bytes(data: bytes) -> H7Fixture`, `adapt_optional_h1_fixture_bytes(data: bytes | None, *, required_scalar_trials: tuple[H7TrialId, ...]) -> H7CompleteLawSnapshot | None`, and a frozen `H7ValidationConfig`. `H7GateResult` remains Task 7's sole `results.py` type. Add `mpmath>=1.3` as an oracle dependency; production modules never import it.

- [ ] **Step 1: Write the exact preregistration and `h7_v1.json` bytes.** Copy the complete JSON, both structured/full-block and factorized/diagonal-within-fiber recognition fixtures, promotion-to-unrestricted closure rule, linear history-scorer equation/coefficient arrays, typed source-row inventory, fixed density-probe component/direction inventory, joint-`K0` rule, and every global/budget/status/control literal from this plan. Record the exact required trial/control ID order and roles, owned action hashes, the separate scalar `GL+(1)` regression versus primary matrix `GL+(2)` claim, the `det<0` nonclaim, the scorer source-frame rule, and the centered-softmax stabilizer distinction. Do not run a parser, determinant, condition-number, objective, probe expansion, or oracle calculation before both files exist.

- [ ] **Step 2: Write failing fixture/type/config tests.** Require exact root fields, group/representation/dimension/order, chain parent/support, exact matrices, both recognition fixtures and family tags, factorized diagonal bytes, scorer profile/law/prefix/alpha/separate-z-and-m-history/covector/mask/support/probability/raw-score bytes, exact trial IDs/roles/predicates/action hashes and twelve-control ID order, fixed probe component/direction expansion and pair hash, immutable defensive access, SPD factors, positive group determinants, direct-element literals, exact GH/precision settings, and per-record wrong-integrity-hash rejection. Require raw borrowed views to retain caller object/storage/version/dtype/shape and autograd identity while owned snapshots clone and reject later private-byte/version mutation. Require scalar fixture adaptation to construct only `H7ScalarReplayAction(dimension=1,group="GL+(1,R)")` with `(1,1)` owned elements and matrix parsing to construct only `H7GLPlus2Action(dimension=2,group="GL+(2,R)")`; reject cross-construction, a claimed scalar `GL+(2)` result, or an action whose declared dimension/group disagrees with its shapes. Test the optional-H1 adapter's exact four cases: required+matching bytes succeeds, required+missing rejects, unused+absent returns `None`, unused+supplied rejects. Reject unknown/missing fields, booleans as numbers, NaN/Inf, wrong shapes/order, non-SPD covariances, any `det<=0` action/frame, any changed direct matrix/scorer coefficient, nonstandard representation, changed parent/source support, a decoder that loses the stated centered-stabilizer property, or any H8 gate/config key. Assert every shorter predecessor prefix still resolves without reading H7 bytes or carrying H7 config/provenance fields.

- [ ] **Step 3: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_fixture.py tests/unit/test_config.py -q
  ```

  Expected: collection/config failures show the H7 types/parser/config do not exist and the resolver stops at the post-H6 prefix.

- [ ] **Step 4: Implement strict records, parser, H1 adapter, and config.** Parse exact schema sets. Defensively copy mappings and capture owned tensors for fixture/evidence records; keep borrowed graph views separate and unhashed. Build `U_t U_j^{-1}` as `torch.linalg.solve(U_j.T,U_t.T).T`; build derived precisions with the two-sided operand solves frozen above; never pass an identity right-hand side to `solve`. Enumerate exact chain sources and construct bank/receiver/source/channel-typed scorer rows with separate `z_history`/`m_history`; evaluate the exact scorer prefix law without transforming `alpha`. Parse both recognition families. The factorized original must be diagonal within fiber, while its declared generic-action output is unrestricted full-block. Reconstruct the preregistered typed probe-pair table from its serialized finite inventory and reject any canonical-byte or pair-set-hash mismatch; do not choose or expand probes during trial/objective runtime. H1 adaptation must preserve the raw H1 fixture hash, exact four-path order, one-based labels, every existing factor value, preregistered probe-pair hash, and optional-byte contract while returning only the scalar action type. Config resolution accepts only the prior exact prefixes plus the full ordered tuple `("H1","H2","H3","H4","H5","H6-Prefix","H7")`; the H7 operation references predecessor artifacts and runs H7 only.

- [ ] **Step 5: Freeze raw fixture hashes before any H7 calculation.**

  ```powershell
  Get-FileHash -Algorithm SHA256 vfe4/validation/fixtures/h1_v1.json
  Get-FileHash -Algorithm SHA256 vfe4/validation/fixtures/h7_v1.json
  ```

  Copy the two exact lowercase digests into named parser/config constants and the preregistration in this same task. In the same preregistration-freeze step, materialize the deterministic corresponding-probe pairs once from the already written finite inventories, serialize the exact ordered pair table, and copy its lowercase canonical SHA-256 into the preregistration and config. This identity-only freeze selects no threshold and evaluates no objective/oracle. No normalized/re-serialized raw-fixture hash, runtime probe generation, or unpinned pair-table hash is accepted.

- [ ] **Step 6: Run focused GREEN.** Run the Step 3 command. Expected: all exact schema/hash/config/type/domain tests pass, including `det<0` rejection and no H7 read for shorter prefixes.

- [ ] **Step 7: Have one reviewer inspect the frozen protocol and bytes.** The reviewer checks this plan against the preregistration/fixture/config, confirms all binding matrices and controls are frozen, and inspects the focused output only. Resolve any Important issue before commit.

- [ ] **Step 8: Commit Task 1.**

  ```powershell
  git add pyproject.toml vfe4/types/h7.py vfe4/types/__init__.py vfe4/validation/h7_fixture.py vfe4/validation/__init__.py vfe4/validation/fixtures/h7_v1.json vfe4/config/schema.py vfe4/config/resolve.py tests/unit/test_h7_fixture.py tests/unit/test_config.py docs/preregistrations/2026-07-21-h7-frame-covariance.md
  git commit -m "test: freeze H7 frame covariance protocol"
  ```

## Task 2: Implement the Pure Differentiable Direct Group-Action Layer

**Files:**

- Create: `vfe4/geometry/group_action.py`
- Create: `vfe4/geometry/__init__.py`
- Create: `tests/unit/test_h7_group_action.py`

**Consumes:** raw float64 tensors plus `H7BorrowedTensorView`/`H7BorrowedActionView`; owned `H7ScalarReplayAction | H7GLPlus2Action` records are evidence snapshots only. **Produces:** every public graph-preserving tensor transform and the one `freeze_h7_action` evidence boundary; no law snapshot or objective dependency.

- [ ] **Step 1: Write failing group-action tests.** Cover all exact vector/covariance/precision/information/second-moment/map/`B`/decoder laws against hand-computed `GL+(1)` scalar and `GL+(2)` matrix values; require borrowed scalar/matrix action views and owned action snapshots, dimensions, shapes, group labels, and hashes to remain disjoint. Assert `require_direct_gl_plus`, `borrow_h7_action`, and every direct action result retain the exact caller tensor objects and live storage/version/dtype/shape metadata; mutating a borrowed operand invalidates the view before reuse. Assert `freeze_h7_action` alone clones owned bytes and a caller mutation cannot change the snapshot/hash. Cover `U'=gU`; receiver/source link order; `Omega_21 Omega_10=Omega_20`; the closed relation walk; diagonal versus internal actions; global block order; `logJ_G`; centered projection; positive determinants; `det<0` rejection; and no `matrix_exp` call. For the frozen equal-dimensional matrix operand, prove the required reverse-arrow mutant `G_m^{-1} B G_z` differs from correct `G_z B G_m^{-1}`. Add an independent non-square type case with `B` shape `(d_z,d_m)=(2,3)`, `G_z` shape `(2,2)`, and `G_m` shape `(3,3)`; require the correct output shape `(2,3)` and reject `B.T`, swapped state/model receivers, or any channel-swap adapter before arithmetic.

  Patch `torch.linalg.inv`, `torch.linalg.pinv`, and `torch.matrix_exp` to raise during correct transforms. Wrap `torch.linalg.solve` and raise if a call receives a square identity right-hand side whose sole purpose would be forming an inverse; still allow ordinary vector/matrix operand right-hand sides. Assert `right_solve(value,G)` is exactly `solve(G.T,value.T).T` and inspect the two-sided precision/link/map/decoder paths to prove none creates `eye_like(G)` for a solve.

  Add a gradient test with independent `requires_grad=True` operands and direct elements. Sum nonconstant entries from every transform and call `torch.autograd.grad`; require finite, non-`None`, nonzero gradients for the operand and group element. Inspect `grad_fn` and prohibit detached/cloned outputs.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_group_action.py -q
  ```

  Expected: collection fails because `vfe4.geometry.group_action` does not exist.

- [ ] **Step 3: Implement the minimal pure tensor laws.** Implement `right_solve(value,G)` only as `torch.linalg.solve(G.T,value.T).T`; use `torch.linalg.solve(G.T,value)` for covectors; form two-sided inverse actions by composing right solve with a left solve on the actual operand. Never call `solve(G,I)` or construct an identity RHS to obtain an inverse. Preserve dtype/device, caller identity, and autograd in borrowed/direct paths. Validate borrowed scalar actions as exactly three `(1,1)` positive elements and borrowed matrix actions as exactly three `(2,2)` direct elements; freeze them to the matching owned action type with exact raw bytes and `action_sha256`. `require_direct_gl_plus` rejects wrong rank/shape/dtype, nonfinite entries, singular elements, and nonpositive `slogdet` sign; it returns the original tensor reference rather than a detached copy.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: every tensor law/order/domain/autograd assertion passes and all forbidden inverse/exponential patches remain untouched.

- [ ] **Step 5: Have one reviewer inspect only the algebra and autograd boundary.** Check scalar/matrix action-type separation, receiver/source order, reverse-arrow `B` mutant decisiveness, non-square channel typing, transpose placement, direct right/two-sided solves with no identity RHS, log-Jacobian multiplicity, direct-element domain, and absence of detach/oracle/objective imports.

- [ ] **Step 6: Commit Task 2.**

  ```powershell
  git add vfe4/geometry/group_action.py vfe4/geometry/__init__.py tests/unit/test_h7_group_action.py
  git commit -m "feat: add differentiable direct frame actions"
  ```

## Task 3: Push Forward the Complete Generative Law

**Files:**

- Create: `vfe4/generative/pushforward.py`
- Modify: `vfe4/generative/language.py`
- Modify: `vfe4/generative/__init__.py`
- Modify: `vfe4/generative/source_priors.py`
- Create: `tests/unit/test_h7_generative_pushforward.py`

**Consumes:** the post-H6 normalized generative model or H1/H7 fixture adapter plus Task 2 borrowed action/tensor laws. **Produces:** graph-preserving `H7GenerativeTensorLaw`, immutable owned `H7GenerativeSnapshot`, typed scorer rows, and complete pushed-forward law/density metadata.

- [ ] **Step 1: Write failing generative tests.** For both separately typed scalar-regression actions and every primary matrix frame/action profile, assert transformed `U`, every ordered link, causal transition map, `B`, offset, covariance, derived precision, initial law, `(h,J,M)`, decoder, bias, support, and source probability. Require each transition location to satisfy `mu_tj'=g_t mu_tj` on the exact density probes and each normalized receiver log density to shift by `-logdet(g_t)` per channel.

  Fixed source tables must have identical canonical bytes. Exercise exactly `h7-linear-history-source-v1`: for both banks and `t=(1,2)`, require one `H7SourceScorerRowSnapshot(bank,receiver_t,source_j)` with separate typed `z_history`/`m_history`, prefix bytes/term, alpha values, channel covectors, mask, support, raw scores, probabilities, and row hash. Recompute the serialized `alpha` formula from the frozen prefix, require `r_z'=solve(g_j.T,r_z)` and `r_m'=solve(g_j.T,r_m)`, and record separate covector and raw-score residuals before proving mask/support/order/probabilities unchanged. Under `matrix-nonidentity-internal-transformed`, inject unchanged covectors and the exact gate mutant `r'=solve(g_t.T,r)` in place of the source-frame solve; both unit cases must change a raw score, and the latter supplies control ID `history_scorer_wrong_source_inverse`. Test transformed decoder logits exactly, fixed decoder centered-stabilizer probabilities, and outside-stabilizer probability change. Hash the source model before and after every transformation and require no mutation. Prove borrowed tensor laws remain graph-connected and that only `freeze_h7_generative` creates owned clone/hash evidence.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_generative_pushforward.py -q
  ```

  Expected: collection fails because the generative snapshot/pushforward interface is absent.

- [ ] **Step 3: Implement borrowed law, complete pushforward, and evidence freeze.** Consume the existing post-H6 `LanguageGenerativeModel`. Borrow every graph tensor required by the complete objective while recording raw identity/version metadata. Build exact bank/channel/receiver/source records with separate z/m histories, prefix/context bytes, scorer profile ID, `alpha` coefficients/value, source covectors, support, mask, raw scores, probabilities, and scorer canonical hash. Transform scorer covectors only with their source block through left solves; keep `alpha` and categorical identities byte-identical. Transform all other continuous/model/geometric factors with direct operand solves. For `decoder_policy="fixed"`, leave weights/bias untouched and record whether the exact centered-stabilizer equations hold; do not silently coerce an outside action into a positive trial. Freeze original/transformed tensor laws only after calculation into complete owned snapshots; no borrowed object enters a result or artifact.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: every factor/location/density/source/decoder/nonmutation check passes for H1 and H7 laws.

- [ ] **Step 5: Have one reviewer inspect complete-factor coverage against the post-H6 model.** The reviewer checks that no generative factor consumed by `language_elbo.py` or the source-prior path is absent, that every history covector uses source rather than receiver frame, that the exact prefix term is untouched, and that no recognition or verification import entered production.

- [ ] **Step 6: Commit Task 3.**

  ```powershell
  git add vfe4/generative/pushforward.py vfe4/generative/source_priors.py vfe4/generative/language.py vfe4/generative/__init__.py tests/unit/test_h7_generative_pushforward.py
  git commit -m "feat: push forward complete generative laws"
  ```

## Task 4: Push Forward the Complete Recognition Law Without Altering H2 Snapshots

**Files:**

- Create: `vfe4/recognition/pushforward.py`
- Read only: `vfe4/recognition/language.py`
- Modify: `vfe4/recognition/__init__.py`
- Create: `tests/unit/test_h7_recognition_pushforward.py`

**Consumes:** exactly post-H6 `StructuredLanguageRecognition | FactorizedLanguageRecognition` filtering/smoothing laws, the two exact Task 1 fixtures, and H1/H7 adapters. **Produces:** borrowed complete tensor laws plus owned snapshots; generic matrix actions promote a factorized origin to an unrestricted full-block `H7RecognitionTensorLaw` without introducing a recognition alias or a projected same-family result.

- [ ] **Step 1: Write failing recognition tests.** Parameterize the same complete checks over exact instances/fixtures of `StructuredLanguageRecognition` and `FactorizedLanguageRecognition`. Cover separately typed scalar and matrix initial means/covariances/precisions/information vectors/second moments; every model and state conditional parent map, state-model map, offset, covariance, and precision; exact source order/probabilities/support; joint component `(mu,Sigma,h,J,M)` laws; per-conditional and complete recognition density shifts; and the `+logJ_G` continuous entropy shift. Require original snapshot/model/optimizer hashes unchanged. For the structured origin, require `representation="structured_full_block"` before and after. For the factorized origin, require the frozen diagonal bytes initially, then apply the non-diagonal internal action and witness at least one nonzero within-fiber off-diagonal entry; require `representation="unrestricted_full_block_pushforward"`, full congruence equality, and unchanged `origin_family`. Reject a returned `FactorizedLanguageRecognition`, diagonal truncation, moment projection, or `representation="factorized_diagonal_within_fiber"` after that action.

  Assert the public signature is the explicit concrete union (or overloads with those exact two parameter types), not `object`, `Any`, a nonexistent `LanguageRecognitionLaw`, or an open structural alias. Pass a projection-tagged/moment-projected recognition record, an emission-only objective record, and a lookalike object exposing only moments/emissions; each must fail closed before factor access. This prevents H7 from silently proving covariance of a projected Gaussian or partial emission record instead of the normalized complete recognition law.

  Import and exercise every public H2 type/factor/evaluator before and after the H7 transformation. Require identical H2 canonical values, detached ownership, public signatures, and no `requires_grad`; H7 tensor gradients must use only the separate borrowed tensor-law record. Mutate a caller tensor and a returned clone to prove owned snapshots remain stable; mutate private owned storage in a negative control and require integrity failure before evaluation/publication.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_recognition_pushforward.py -q
  ```

  Expected: collection fails because the recognition pushforward does not exist.

- [ ] **Step 3: Implement complete recognition borrowing, pushforward, promotion, and freeze.** Import the existing concrete `StructuredLanguageRecognition` and `FactorizedLanguageRecognition` types read-only and dispatch through the exact union/overloads. Borrow every complete factor for the matched family and reject all other runtime classes; do not edit `language.py` or create an alias. Transform each source-conditioned component before exact source marginalization; never moment-project the mixture. Keep fixed categorical weights byte-identical. A structured origin remains full-block. A factorized origin under a generic non-diagonal `GL+(2)` action returns an unrestricted full-block tensor law carrying the factorized origin tag and cannot be reconstructed as the concrete factorized family. Freeze only at the evidence boundary, with owned bytes and canonical hashes. Compute entropy shifts from conditional receiver Jacobians and source weights, then independently compare with the global `logJ_G` shift.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: all component/joint/density/entropy/source/H2-preservation tests pass.

- [ ] **Step 5: Have one reviewer inspect concrete-type dispatch, representation closure, mixture semantics, and the H2 boundary.** Reject any same-family closure claim for generic actions, missing factorized fixture, diagonal truncation, invented alias/open duck type, projection/emission-only acceptance, component omission, source-marginal Gaussian projection, borrowed tensor in an evidence record, in-place mutation, or H2 differentiability change.

- [ ] **Step 6: Commit Task 4.**

  ```powershell
  git add vfe4/recognition/pushforward.py vfe4/recognition/__init__.py tests/unit/test_h7_recognition_pushforward.py
  git commit -m "feat: push forward complete recognition laws"
  ```

## Task 5: Evaluate Complete Local, Monolithic, Density, and Evidence/KL Covariance

**Files:**

- Create: `vfe4/objective/h7_covariance.py`
- Modify: `vfe4/objective/__init__.py`
- Modify: `vfe4/objective/language_elbo.py`
- Create: `tests/unit/test_h7_complete_objective.py`

**Consumes:** the unchanged existing complete local objective/factor trace plus Tasks 3--4 snapshots. **Produces:** `H7ObjectiveCovarianceEvaluation` only; no trainable loss or acceptance decision.

- [ ] **Step 1: Write failing complete-objective tests.** Require the exact term inventory: expected emissions by time; exactly one `H7InitialJointKlRecord(term_id="K0_joint_z0_m0")`; model/state source terms; model/state transition terms by time; joint recognition entropy; complete local ELBO; monolithic ELBO; complete `p` and `q` density shifts; pointwise log ratio; `r_abs`; `r_rel`; and every tensor-law diagnostic. On the structured fixture with nonzero initial z/m cross covariance, compute the joint Gaussian KL directly and prove it differs from the sum of the two marginal KLs; reject the marginal sum, duplicate initial terms, or an unchecked marginal-plus-conditional chain. The matrix path additionally requires all twelve `h7-linear-history-source-v1` residuals--eight covector-law residuals (`r_z` and `r_m` for two banks at two receiver times) plus four raw-score residuals--before accepting its unchanged source terms. The separately labeled scalar regression additionally requires evidence, posterior KL, and `log evidence-ELBO=KL` before/after. The matrix fixture marks evidence/KL not applicable with a fixed reason and no fabricated value.

  Consume the exact pre-expanded `H7DensityProbePair` tuple for every exact source component. Require each pair's fixture/component/source/action/anchor provenance and `x_prime=G_component@x`, then require its separate initial/receiver/global shift and the aggregate `log p' = log p-logJ_G`, `log q'=log q-logJ_G`, and pointwise log-ratio equality. A mutant that independently whitens or re-anchors `x_prime`, changes a probe, omits a source component, or creates a probe during evaluation must fail before a density comparison. Prove that removing any local term, using only transitions/latents, or checking only the final scalar makes the completeness validator reject the evaluation.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_complete_objective.py -q
  ```

  Expected: collection fails because the H7 complete-objective diagnostic is absent.

- [ ] **Step 3: Expose, do not replace, the existing complete objective.** If `language_elbo.py` lacks an importable immutable complete-term/factor trace, expose its existing evaluator without changing its scalar arithmetic or training call sites. Adapt its two initial channel entries only at the H7 diagnostic boundary into one mechanically computed joint `K0`; do not alter the post-H6 training objective. Reject the adaptation unless the joint density inputs are complete and the resulting H7 trace contains exactly one initial term. Implement H7's monolithic expectation separately from the local partition, using exact source enumeration and deterministic quadrature. The H7 module compares the two; it never supplies a loss to training or H5.

- [ ] **Step 4: Implement density, entropy, and available evidence/KL diagnostics.** Validate and consume only the fixture/parser-owned corresponding probe pairs; record their shared anchor provenance, `x/x_prime` hashes, separate initial and receiver shifts before the global shift, exact component/source identities, complete density/log-ratio probes, local terms including joint `K0`, both complete scalars, and applicability. Preserve the H1 independent evidence path and its original normalization conventions.

- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: complete scalar/local/density/term/evidence checks pass; every intentionally partial inventory is rejected.

- [ ] **Step 6: Have one reviewer inspect objective completeness, joint-initial semantics, corresponding probes, and signs.** Compare the term inventory with the authoritative post-H6 objective and manuscript pushforward proof; independently check joint `K0` against the nonzero cross-channel fixture and reject a marginal-KL sum; verify every `x_prime` is the declared action on the same original anchored point; inspect source entropy, emission normalizers, determinant/Jacobian signs, and H1 evidence/KL orientation.

- [ ] **Step 7: Commit Task 5.**

  ```powershell
  git add vfe4/objective/h7_covariance.py vfe4/objective/__init__.py vfe4/objective/language_elbo.py tests/unit/test_h7_complete_objective.py
  git commit -m "feat: evaluate complete H7 objective covariance"
  ```

### July 26 Task 4B1.5 amendment: assemble global canonical precisions

The original Task-5 implementation did not retain the 40 fixed-path global
precision matrices needed by the independent precision-operand inventory. A
temporary capture seam required those matrices as injected owned values. This
amendment replaces that temporary source with direct production canonical
assembly; it does not authorize serialization, calibration, trials, controls,
or an H7 status decision.

For each original-law fixed source path, use exact global order
`[z0,m0,z1,m1,z2,m2]`. Scatter the joint initial `(J0,h0)`. For model and state
conditionals respectively, assemble

```text
R_m,t = E_m,t - A_m,t,b_t E_m,b_t
R_z,t = E_z,t - A_z,t,a_t E_z,a_t - B_t E_m,t
J += R^T Lambda R
h += R^T Lambda offset
```

using only already-owned component precisions, information vectors, affine
maps/offsets, multiplication, addition, and block scatter. The new assembler
must not call `inv`, `pinv`, `cholesky_inverse`, solve against an identity, or
derive `J` from the propagated covariance. Source probabilities do not enter
a fixed-path canonical pair.

Capture order is every `q` path followed by every `p` path. Each of the two
scalar batches contributes eight globals in the frozen four-path order. Each
of the twelve matrix-family batches contributes singleton `q` then `p`.
Together with 152 component rows, the 40 globals close the ordered 192-row
inventory.

- [ ] **Step 4B1.5.1: Write two focused failing tests.** Cover exact scalar and
  matrix-family path/order/provenance, both `J Sigma` and `Sigma J`, `h=J mu`,
  factor/global quadratic agreement, defensive ownership, stale factor
  rejection, and source-scan/runtime bans on inverse APIs in the new assembler.
- [ ] **Step 4B1.5.2: Add an immutable assembled-global record and direct
  assembler.** Bind trial, `q`/`p` role, path, original-law snapshot, the exact
  ordered initial/model-1/state-1/model-2/state-2 component hashes, propagated
  covariance snapshot, assembled precision/information snapshots, and one
  domain-separated assembly hash. Global precision operands use
  `source_kind="assembled_global"`; the live capture accepts no injected global
  values.
- [ ] **Step 4B1.5.3: Run only the two exact focused nodes, review, and
  commit.** Do not run the complete file or suite.

This amendment makes only the bounded claim that the new global assembler
performs no inverse synthesis. The fixture parser currently materializes
already-owned local receiver precisions separately; their provenance remains a
pre-H7-closure obligation if the final claim is end-to-end inverse-free.

### July 26 Task 4B2 amendment: serialize production covariance and precision

Replace the current `h7-mp-precision-operands-v1` covariance-hash equality
with the reviewed root schema `h7-mp-precision-operands-v2`, source contract
`task5-production-covariance-and-precision-v2`, and binary64 text policy
`python-repr-binary64-roundtrip-v1`. The root fields are exactly

```text
precision_table_schema
h1_raw_fixture_sha256
h7_raw_fixture_sha256
ordered_trial_ids
source_contract
binary64_text_policy
precision_set_sha256
records
```

Each record has exactly

```text
row_index
trial_id
gaussian_id
source_kind
shape
covariance_values
covariance_values_sha256
covariance_snapshot_sha256
precision_values
precision_values_sha256
precision_snapshot_sha256
record_sha256
```

Serialize both production tensors as row-major arrays of canonical Python
`repr` strings. Every leaf must be finite, parse to one binary64 value, and
round-trip to the identical token and binary64 bits under the declared text
policy. Emit exactly the closed 192-row Task-5 inventory in its exact capture
order: every batch's `owned_component` rows precede its `assembled_global`
rows, for complete totals of 152 and 40. Reject a missing, duplicated,
reordered, or extra row; a source kind outside those two literals; any global
Gaussian ID not marked
`assembled_global`; and every legacy `injected` source.

The exact hash preimages are:

- `covariance_values_sha256`: domain
  `vfe4.h7.mp-serialized-covariance-values.v2` over
  `{trial_id, gaussian_id, source_kind, shape, covariance_values}`;
- `precision_values_sha256`: domain
  `vfe4.h7.mp-serialized-precision-values.v2` over
  `{trial_id, gaussian_id, source_kind, shape, precision_values}`;
- `record_sha256`: domain
  `vfe4.h7.mp-serialized-precision-operand.v2` over every row field except
  `record_sha256`;
- `precision_set_sha256`: domain
  `vfe4.h7.mp-serialized-precision-set.v2` over every root field except
  `precision_set_sha256`, including the complete ordered records.

Use the existing H7 canonical hashing semantics for every domain. The parser
reconstructs the exact row-major binary64 bytes and recomputes both
`vfe4.h7.owned-tensor-snapshot.v1` hashes. It independently assembles the
100-decimal covariance, compares it numerically with the serialized production
covariance without requiring representation-hash equality, verifies both
`Sigma J = I` and `J Sigma = I` on the serialized production pair, and
requires exactly 192 consumed rows.

- [ ] **Step 4B2.1: Add two focused failing tests.** Freeze the exact v2
  root/row fields, domains, canonical binary64 text round trip, 192-row
  ordering and 152/40 source split. Require the independent consumer to accept
  a numerically equal high-precision covariance whose representation hash
  differs. Reject v1, injected sources, stale value/snapshot/row/set hashes,
  nonfinite or noncanonical tokens, and incomplete/reordered inventories.
- [ ] **Step 4B2.2: Add the production table writer and v2 oracle consumer.**
  Serialize only the owned Task-5 capture batches. Recompute every hash and
  snapshot from the exact values rather than trusting caller metadata. Keep
  the oracle covariance independent while validating the serialized
  production covariance/precision pair.
- [ ] **Step 4B2.3: Run only the two exact focused nodes, review, and commit.**
  Do not run the complete file or suite.

This amendment freezes no concrete v2 raw-file, set, or oracle-inventory hash.
Those measurements remain `UNMEASURED`, and their expected constants remain
`None`, until a separately authorized serialization/calibration run.

## Task 6: Add the Independent 100-Decimal Oracle and Operand-Local Budgets

**Files:**

- Create: `verification/mp_oracles/__init__.py`
- Create: `verification/mp_oracles/h7_covariance.py`
- Create: `verification/mp_oracles/h7_budget_protocol.py`
- Create: `verification/h7_budget.py`
- Create: `tests/oracle/test_h7_mp_oracle.py`
- Modify: `docs/preregistrations/2026-07-21-h7-frame-covariance.md`

**Consumes:** raw H1/H7 JSON bytes only. **Produces:** independent original/transformed tensor/law/local/monolithic/density/oracle values and exact budget records. The complete transitive oracle closure is restricted to the Python standard library plus `mpmath`; it imports no `vfe4`, Torch, NumPy, or general verification budget module.

- [ ] **Step 1: Write failing oracle/budget tests.** Require no `vfe4`, torch, or NumPy imports in the oracle module. Verify the 100-decimal Jacobi-matrix Gauss--Hermite construction: the physicists-Hermite Jacobi matrix has zero diagonal and off-diagonal entry `sqrt(k/2)` between zero-based rows `k-1` and `k` for `k=1..n-1`; mpmath `eigsy` returns nodes `x_i` and orthonormal eigenvectors; standard-normal nodes are `sqrt(2)*x_i`; normalized weights are the squared first eigenvector components and sum to one. Test exact polynomial moments through degree 12.

  Require exact source enumeration, independent evaluation of the serialized `alpha` formula and every original/transformed history-covector/raw-score value, independent affine Gaussian assembly, independent covariance/precision/information transformations, receiver/global log-Jacobian shifts, joint `K0`, all local and monolithic terms, corresponding frozen probe pairs, scalar evidence/KL, and the two-dimensional logit-contrast emission integral at orders 41/51. For every category, test the complete `H7OperandRecord` tuple (typed category, ID, role, shape, hash, scale, condition, normalization, oracle decimal) and every `H7AllowanceContribution` field (kind, operation ID/kind/count, optional exact quadrature order, unit allowance, value), then the total allowance and canonical budget hash. Require one `H7BackwardResidualRecord` per operand before the maximum. Reject an aggregate-only budget, global condition maximum, borrowed scale/normalization, unnamed oracle value, GH contribution on an analytic invariant, GH convergence boundary, and control-decisiveness boundary.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/oracle/test_h7_mp_oracle.py -q
  ```

  Expected: collection fails because the mpmath oracle and H7 budget do not exist.

- [ ] **Step 3: Implement the independent oracle.** Set `mp.mp.dps=100` inside the entry call and restore the caller's precision afterward. Parse JSON numbers as decimal strings into `mp.mpf`. Use mpmath matrices and `lu_solve` directly on each actual vector/matrix right-hand side, never an identity RHS created solely to form an inverse. Independently evaluate the exact scorer prefix arithmetic and source-covector solves, Cholesky/eigendecomposition checks, analytic Gaussian expectations, exact categorical sums, and local/monolithic reductions. Reduce each selected emission log-softmax to its two Gaussian logit contrasts; do not call a production projector or quadrature helper.

- [ ] **Step 4: Implement exact category- and operand-local budget records.** A budget constructor receives immutable named `H7OperandRecord` values rather than scalar aggregate scale/condition arguments or a run-wide summary. Derive positive operation counts from the typed category and shapes; bind per-operand scales/conditions/normalizations/value hashes; store oracle values as exact decimal strings; record the exact operation ID/kind/count, optional GH order, and unit allowance; add operation, quadrature, and reference contributions exactly once where applicable; and validate the domain-separated budget hash. Build every backward record separately from original/transformed/recovered operand hashes and its own budget, then take the maximum without discarding the tuple.

- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: independent analytic/GH/oracle/budget tests pass; every required GH41/51 delta clears the frozen relative limit.

- [ ] **Step 6: Record only preregistered calibration facts.** Add the measured raw fixture hashes, required operand condition extrema, and GH41/51 deltas to the preregistration. Do not alter a threshold, fixture byte, action, trial, or control after seeing those values. If a frozen trial is outside the envelope or GH convergence is unresolved, stop this candidate as INCONCLUSIVE rather than tuning it.

- [ ] **Step 7: Have one reviewer inspect oracle independence and numerical contracts.** Check imports, source enumeration, contrast reduction, 100-decimal arithmetic, quadrature construction, budget locality, and no threshold tuning.

- [ ] **Step 8: Commit Task 6.**

  ```powershell
  git add verification/mp_oracles/__init__.py verification/mp_oracles/h7_covariance.py verification/mp_oracles/h7_budget_protocol.py verification/h7_budget.py tests/oracle/test_h7_mp_oracle.py docs/preregistrations/2026-07-21-h7-frame-covariance.md
  git commit -m "test: add the H7 high precision oracle"
  ```

## Task 7: Build the Fail-Closed H7 Gate and All Decisive Controls

**Files:**

- Create: `verification/h7_gate.py`
- Modify: `vfe4/types/results.py`
- Modify: `vfe4/types/__init__.py`
- Create: `tests/promotion/test_h7_gate.py`
- Modify: `docs/preregistrations/2026-07-21-h7-frame-covariance.md`

**Consumes:** Tasks 1--6 plus immutable predecessor references. **Produces:** the sole `vfe4/types/results.py::H7GateResult` and one immutable `H7GateEvaluation(result, validation_payload_canonical_json, validation_payload_sha256, fixture_set_sha256, dependency_closure_sha256, evaluation_sha256)`; `vfe4/types/__init__.py` re-exports the result exactly once.

- [ ] **Step 1: Write failing gate/result tests for every role and residual family.** Require the exact two `role="scalar_regression"` `GL+(1)` specs, exact five `role="positive_covariance"` `GL+(2)` specs, and one outside-stabilizer `role="expected_negative"` spec with their exact predicates/action hashes. Require scalar actions to carry only dimension 1/group `GL+(1,R)` and matrix actions only dimension 2/group `GL+(2,R)`; prove scalar results and the expected-negative result cannot satisfy the positive matrix inventory. For each scalar/positive trial require the immutable envelope, `r_abs`, `r_rel`, every per-operand backward record before `r_back_max`, complete owned original/transformed `H7LawPairSnapshot` values, cocycle/open/closed products, joint `K0`, all local terms, both recognition-family origins (including factorized-to-full promotion for generic actions), source identities, all twelve scorer residuals where applicable, decoder/logit/probability laws, exact corresponding probe pairs, density/Jacobian/entropy shifts, local/monolithic complete values, and applicable evidence/KL. The expected-negative trial requires a decisive emission/complete-objective change and must reject `observed_predicate="complete_covariance"`. Assert the matrix internal trial cannot pass when only latent/transition residuals, snapshot hashes without the owned law pairs, a same-family factorized result, independently whitened probes, marginal initial KLs, aggregate-only backward/budget data, or source probabilities without raw-score residuals are supplied.

  Construct every public owned snapshot/envelope/objective/trial/control/result/evaluation record from mutable tensor/mapping inputs, mutate the originals, and require stable canonical bytes/hashes. Supply a wrong owned hash, duplicate/missing map key, borrowed tensor view, private-storage mutation, inconsistent nested status, or noncanonical payload bytes and require construction/access to fail. Import `H7GateResult` from `vfe4.types.results` and `vfe4.types`; require object identity and prove `vfe4.types.h7` has no competing definition.

- [ ] **Step 2: Write one isolated mutant per required control.** Inject exactly:

  1. `wrong_covariance_congruence`: `Sigma'=g^{-T} Sigma g^{-1}`;
  2. `wrong_precision_congruence`: `J'=g J g^T`;
  3. `history_scorer_wrong_source_inverse`: under `matrix-nonidentity-internal-transformed`, use `r_z'=g_t^{-T}r_z` and `r_m'=g_t^{-T}r_m` instead of the source-frame `g_j^{-T}` laws, targeting the frozen raw-score residual;
  4. `reversed_link_order`: `Omega'=g_j^{-1} Omega g_t` instead of the receiver/source coboundary order;
  5. `reverse_arrow_B`: `B'=G_m,t^{-1} B G_z,t` instead of `G_z,t B G_m,t^{-1}` on the frozen equal-dimensional operand;
  6. `wrong_decoder_dual_action`: `W'=W G`;
  7. `fixed_decoder_outside_stabilizer`: hold the decoder fixed under the frozen outside-stabilizer diagonal action;
  8. `omitted_density_jacobian`: use zero receiver/global Jacobian shift;
  9. `reversed_logdet_sign`: use `+logdet(g_t)` instead of the inverse-density sign;
  10. `entropy_false_invariance`: assert `H(q')=H(q)`;
  11. `changed_h1_source_probability`: swap the positive H1 `t=2` source probabilities while leaving support fixed and normalized;
  12. `diagonal_for_internal_action`: substitute the frozen diagonal action for the internal `(g0,g1,g2)` action.

  Require this exact ID/order tuple in preregistration, tests, `H7ControlResult`, and artifact JSON. Each mutant must exceed `max(100*correct_allowance,1e-8*scale)` on its named invariant. Test a synthetic boundary exactly at the limit as nondecisive/INCONCLUSIVE and immediately above it as detected. Controls use fresh copies and cannot be caught only by a generic exception. The independent non-square `B` channel-swap rejection is a type test, not a thirteenth numeric control.

- [ ] **Step 3: Write fail-closed status tests.** Independently inject missing/stale predecessor, fixture hash mismatch, missing/duplicate/wrong-role trial or control, wrong action hash, out-of-envelope group/SPD, unresolved GH delta, nonfinite calculation, finite in-envelope covariance failure, finite local-term or joint-`K0` failure, same-family factorized closure, incomplete/mismatched density-probe pairs, and nondecisive expected-negative change. Require the exact PASS/FAIL/INCONCLUSIVE precedence from this plan and construction-time status consistency. A decisive outside-stabilizer change contributes only expected-negative success; covariance acceptance of it is FAIL; a nondecisive change is INCONCLUSIVE. `det<0` remains rejected/outside-domain and never creates a full-GL PASS claim.

- [ ] **Step 4: Run focused RED.**

  ```powershell
  python -m pytest tests/promotion/test_h7_gate.py -q
  ```

  Expected: collection fails because `verification.h7_gate` does not exist.

- [ ] **Step 5: Implement predecessor and dependency-closure validation.** Validate predecessor references only when each artifact and its validated ledger has the exact final H7 `git_head`, `dirty_digest`, candidate JUnit SHA, producer schema, manifest, payload/certificate-set hash, and PASS status. Require the ordered current H1--H5 artifact/ledger first, then the conditional H1-prefix-prior artifact/ledger when a scorer profile is consumed, then the independently produced current H6-Prefix artifact/certificate-set/ledger. This order belongs only to the H7 reference registry: H6-Prefix must contain no H1--H5/H4 predecessor identity and must have been produced with `predecessor_refs={}`. `h7-linear-history-source-v1` activates the conditional input. Reject historical, development, reordered, missing-ledger, wrong-revision, post-H7-produced, or predecessor-bearing H6-Prefix references. Hash every consumed source/config/fixture/objective/adapter file into the H7 dependency closure. Do not copy or rerun predecessor payloads, and do not require H6-Prediction absent an explicit empirical checkpoint trial.

- [ ] **Step 6: Execute trials, oracle, controls, freeze evidence, and determine status.** Capture both fixture byte sequences once, borrow graph-preserving tensor laws/actions for calculation, execute correct trials and independent controls, then freeze every tensor/action/law/probe/objective/envelope/result into owned canonical records before status construction. Reject any borrowed view in the evidence tree. Validate every nested integrity hash and exact role/predicate inventory. The expected outside-stabilizer trial succeeds only when the held-fixed decoder causes a decisive emission/complete-objective change; it is not a positive covariance trial.

- [ ] **Step 7: Emit the complete `validation/h7.json`.** Include gate/status/obligations/result hash; exact source/dependency closure; references to the current H1--H5, active H1-prefix-prior, and H6-Prefix artifact/manifest/payload/certificate-set/ledger hashes; raw fixture hashes; separate scalar `GL+(1)` regression and primary matrix `GL+(2)` group/representation/base/product/nonclaim tags; every owned action snapshot/raw-byte/action hash; every trial role/expected/observed predicate; every frame/determinant/norm; every original/transformed SPD diagnostic; both recognition origins and the factorized-to-unrestricted promotion; exact scorer law/profile/prefix/alpha/separate-z-and-m-history/covector/mask/support/raw-score/probability identities plus all twelve scorer residuals; tensor/law/cocycle/closed-walk residuals; corresponding density probe pair IDs, shared anchor provenance, `x/x_prime` hashes and separate expected shifts; entropy shifts; joint `K0`; every local term; monolithic/local/evidence/KL records and applicability; `r_abs/r_rel`, every per-operand backward record and `r_back_max`; every operand/category/operation/quadrature/reference budget input/contribution; GH41/51 values/deltas; every exact control ID/residual/limit/detection; canonical nested/evaluation hashes; and explicit H8/training/optimizer/det-negative/base-curvature/predictive nonclaims. The H7 run directory contains only owned immutable reference/evidence records and no borrowed tensor view, predecessor validation payload, certificate-set copy, ledger copy, or H6-Prediction record.

- [ ] **Step 8: Run focused GREEN.** Run the Step 4 command. Expected: the frozen gate is PASS; every correct invariant clears its own allowance; all twelve controls are decisive; every status mutant maps exactly.

- [ ] **Step 9: Have two bounded reviewers inspect existing evidence.** One checks gauge/type/decoder/Jacobian mathematics; one checks complete objective, oracle, controls, predecessor freshness, and status precedence. They inspect focused output and source only and do not rerun tests.

- [ ] **Step 10: Commit Task 7.**

  ```powershell
  git add verification/h7_gate.py vfe4/types/results.py vfe4/types/__init__.py tests/promotion/test_h7_gate.py docs/preregistrations/2026-07-21-h7-frame-covariance.md
  git commit -m "test: add the complete H7 covariance gate"
  ```

## Task 8: Publish H7 Through the One Ordered Verification Surface

**Files:**

- Modify: `verification/run_gates.py`
- Modify: `vfe4/artifacts/provenance.py`
- Modify: `verify_vfe4.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_atomic_artifacts.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `README.md`
- Modify: `docs/preregistrations/2026-07-21-h7-frame-covariance.md`

**Interface:** The one editable verifier accepts the exact ordered tuple through H7 and selected operation `H7`. Before the final freeze, H7 owns the pure nonmutating `project_h1_h5_compatibility_config(CONFIG)` projection and reuses H6's frozen `project_h1_prefix_prior_config(CONFIG)`, `project_h6_prefix_config(CONFIG)`, and keyword-only `run_projected_current_candidate(*, config, junit_sha256, predecessor_refs) -> CandidateArtifactReference`. The H6 producer accepts only `ProjectedCurrentCandidateConfig`; H6-Prefix is always invoked with `predecessor_refs={}` and rejects every nonempty mapping. Task 9 runs H1--H5 once through the existing ordered verifier, calls each H6-owned projection/producer once, and never edits tracked CONFIG. The H7 operation validates those sibling external references, captures H1/H7 bytes once, runs H7 only, and returns the existing `VerificationRunResult` explicit union containing one `H7GateResult` for this selected operation.

- [ ] **Step 1: Write failing config/artifact/integration tests.** Require exact H7 config/action/trial-role/recognition-family/probe/oracle/operand-budget/predecessor literals and reject H7 without the full ordered tuple, reordered/duplicate gates, H8, changed group matrices, changed action bytes/hash, `phi`/exponential mode, weakened envelope, lower precision, changed GH orders, or altered control thresholds. Preserve every shorter prefix's existing behavior and prove it does not read H7 fixture bytes, import mpmath, run H7, or publish H7 keys. Require artifact publication to reject a borrowed tensor/action view, mutable mapping, wrong nested/result hash, same-family generic factorized output, marginal initial-KL split, or wrong-role outside-stabilizer result. Add a focused lifecycle consumer-contract test that freezes the one-argument `project_h6_prefix_config(CONFIG)` signature, the keyword-only H6 producer and `CandidateArtifactReference` return type, `predecessor_refs={}` for H6-Prefix, rejection of the old predecessor-accepting projector/nonempty runner mapping, and absence of H1--H5/H4 provenance from the Prefix artifact.

  One mocked H7 `main()` call publishes exactly:

  ```text
  config.json
  provenance.json
  environment.json
  references/h1_h5.json
  references/h1_prefix_prior.json   # present only when consumed
  references/h6_prefix.json
  validation/h7.json
  manifest.sha256
  ```

  Each reference record contains the exact predecessor artifact path plus manifest, relevant payload/certificate-set, and validated ledger hashes. It must not copy `validation/h1.json` through `validation/h6_prefix.json`, copy a predecessor certificate/ledger, rerun predecessors, emit prediction metrics, or create `validation/h8.json`.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py -q
  ```

  Expected: failures show the unified surface currently stops at H6-Prefix and has no H7 artifact/provenance path.

- [ ] **Step 3: Extend conditional config resolution and runner orchestration.** Keep one editable dictionary. H7 owns or reuses pure `project_h1_h5_compatibility_config(CONFIG)` for its separate H1--H5 compatibility run. Reuse H6's exact `project_h1_prefix_prior_config(CONFIG)` and one-argument `project_h6_prefix_config(CONFIG)` without wrapping, widening, or mutating CONFIG, and call the keyword-only H6 producer with `predecessor_refs={}` for both independent projected operations. H6-Prefix consumes no H1--H5, H4, H1-prefix-prior, H7, or H6-Prediction reference and never starts an H4 timing run. The H7 path validates the three sibling reference records, captures H1/H7 fixture bytes once, and passes them into `evaluate_h7`; a gate may not reread them. Publish only after H7 returns. `_script_main` returns zero only for H7 PASS and prints `H7: pass` plus one artifact path.

- [ ] **Step 4: Extend provenance without weakening earlier artifacts.** Record exact Git revision, dirty-content/dependency closure, candidate JUnit SHA, config/objective schema, fixture expected/observed hashes, predecessor manifest/payload/certificate-set/ledger hashes, exact production order, scalar `GL+(1)` replay versus primary `GL+(2)` type tags, group/action bytes/hashes, closed trial roles/predicates, both recognition origins and promotion rule, typed scorer/probe-set identities, joint-`K0` schema, owned snapshot/result hashes, oracle version/precision/orders, envelope/operand-budget/exact control-ID constants, H7 status, and bounded nonclaims. Manifest hashing covers every immutable reference and `validation/h7.json`; borrowed graph views are unpublishable.

- [ ] **Step 5: Update bounded documentation.** README and preregistration describe the implemented H7 surface and exact protocol only; before the milestone they do not prestate JUnit totals or measured residuals. State that only the direct matrix trials cover selected `GL+(2)` elements, while the scalar `GL+(1)` path is a complete-law regression; neither covers the det-negative component. Freeze the exact history-scorer law/control ID, fixed-decoder centered-softmax stabilizer restriction, entropy shifts, one-revision predecessor production order, and open optimizer/training/H6-Prediction/H8 claims.

- [ ] **Step 6: Run focused GREEN.** Run the Step 2 command. Expected: all compatibility prefixes remain isolated; the focused lifecycle assertions match H6's frozen signatures; and one H7 click-run publishes exactly the files above with a valid manifest and no predecessor rerun/copy.

- [ ] **Step 7: Have one reviewer inspect the public/config/artifact boundary.** Check single-CONFIG/no-CLI behavior, the H7-owned H1--H5 projection, exact unwrapped H6-owned signatures and `CandidateArtifactReference`, independent H6-Prefix publication with an empty predecessor mapping, conditional scorer prerequisite, exact production/reference order, optional-H1 fixture capture, sole `H7GateResult` export, owned-only nested records/mappings/tensors, exact roles/representation/K0/probe/budget payload, reference-only H7 artifact, and no H8 or empirical widening.

- [ ] **Step 8: Commit Task 8.**

  ```powershell
  git add verification/run_gates.py vfe4/artifacts/provenance.py verify_vfe4.py tests/unit/test_config.py tests/unit/test_atomic_artifacts.py tests/integration/test_verify_vfe4.py README.md docs/preregistrations/2026-07-21-h7-frame-covariance.md
  git commit -m "feat: publish H7 frame covariance verification"
  ```

## Task 9: Produce the One Exact-Revision H7 Milestone Record

**Files:**

- Modify: none. Every tracked protocol, source, test, config, launcher, README, preregistration, and artifact schema is committed and reviewed before candidate selection.
- Produce outside tracked source: `C:\tmp\vfe4-h7-current-candidate-preflight.json`, `C:\tmp\vfe4-h7-milestone.xml`, one current H1--H5 compatibility run, one current H1-prefix-prior run because `h7-linear-history-source-v1` is active, one current H6-Prefix run, one atomic H7 run, `.verification/h7-current-candidate-<FULL_HEAD>-refs.json`, `.verification/h7-current-candidate-<FULL_HEAD>-result.json`, and four new revision-specific ledgers.
- New ledgers, in producer order: `.verification/h1-h5-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json`, `.verification/h1-prefix-prior-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json`, `.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json`, and `.verification/h7-<FULL_HEAD>-<FIXTURE_SET_SHA>-ledger.json`.
- Preserve every prior ledger and generated artifact byte-for-byte. Do not commit `.verification`, run directories, preflight/JUnit output, or measured records.

**Evidence policy:** Task 9 is entirely tracked-source read-only. The one JUnit is produced first; every prerequisite artifact/ledger and H7 artifact/ledger is then produced exactly once, sequentially, at the same frozen `(git_head,dirty_digest)` and JUnit SHA. No tracked edit or commit is permitted after Step 1. A source/test/config/fixture/preregistration/artifact-schema defect makes affected evidence INCONCLUSIVE: preserve it as history, allow the active verification Stop hook to close, return to the owning task, commit a new candidate, and repeat all of Task 9 including one replacement JUnit. Never patch or rerun an artifact at the frozen revision for confidence.

Because the verification control plane permits one `.verification/active.json`, close the four ledgers in four sequential fresh verifier turns. Each turn starts only after the prior ledger validates and its successful recursive Stop-hook invocation removes the activation marker. Never delete, overwrite, rename, or manually repoint that marker.

- [ ] **Step 1: Freeze the final H7 source before producing any evidence.** Record a full 40-character `HEAD`; require every file from Tasks 1--8 tracked; require `git diff --exit-code` and `git diff --cached --exit-code`; require no nonignored untracked path outside `.verification/`; require no active marker; compute the existing `dirty_content_digest`, H7 dependency closure, and raw H1/H7 fixture-set SHA; and hash `.verification/ledger.json` plus every historical revision-specific ledger/artifact reference into `C:\tmp\vfe4-h7-current-candidate-preflight.json`. Require the JUnit destination and every new revision-specific output path to be absent rather than overwriting an earlier attempt. This is the final source freeze; do not validate a historical predecessor as "current" here because current artifacts are produced only after Step 2.

- [ ] **Step 2: Run the only H7 milestone full suite with machine-readable totals.**

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h7-milestone.xml
  ```

  Expected: pytest exits zero. Parse tests/failures/errors/skips only from the XML; do not use terminal dots or remembered predecessor totals. Compute `$junitSha=(Get-FileHash -Algorithm SHA256 C:\tmp\vfe4-h7-milestone.xml).Hash.ToLowerInvariant()`, then immediately recheck `HEAD`, both diffs, dirty digest, dependency closure, and fixture-set SHA. A failure stops this candidate before any prerequisite publication.

- [ ] **Step 3: Produce and close the current H1--H5 compatibility artifact exactly once.** In memory, resolve `project_h1_h5_compatibility_config(CONFIG)` and call the existing ordered `run_verification` once; do not route this operation through H6's two-operation `run_projected_current_candidate`, edit CONFIG, or edit tracked source. Independently validate the artifact manifest, exact `(git_head,dirty_digest)`, JUnit SHA, compatibility config/objective/update/fixture identities, five distinct ordered `validation/h1.json` through `validation/h5.json` payload hashes, and exactly five PASS states. Start, populate one claim per gate/identity check, and validate `.verification/h1-h5-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json` with current mechanical/reproduced evidence. Only after its Stop-hook closure removes the marker may Step 4 begin.

- [ ] **Step 4: Produce and close the conditional H1-prefix-prior artifact exactly once.** The condition is explicit: `h7-linear-history-source-v1` is consumed, so this preregistration requires the step. In memory, call `run_projected_current_candidate(config=project_h1_prefix_prior_config(CONFIG),junit_sha256=junit_sha256,predecessor_refs={})` once at the same revision/digest/JUnit SHA. Validate the exact prefix-prior config, generative-factor/scorer schema, raw fixture, manifest/payload, and PASS status; close `.verification/h1-prefix-prior-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json` in its own fresh verifier turn. If a future H7 protocol removes every prefix/history scorer, this entire artifact/reference is absent rather than replaced by a fake certificate.

- [ ] **Step 5: Produce and close the current H6-Prefix certificate set exactly once.** Independently call `run_projected_current_candidate(config=project_h6_prefix_config(CONFIG), junit_sha256=junit_sha256, predecessor_refs={})` once using keyword arguments. Independently validate the same revision/digest/JUnit SHA, H6 prefix/config/model-family/vocabulary/data-safety identities, every required certificate key, the immutable certificate-set SHA, manifest, PASS state, and absence of any H1--H5/H4 predecessor field or H4 timing invocation. Close `.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json` in a fresh verifier turn. The Step 3 and Step 4 results remain separate sibling inputs to H7 and are not Prefix premises. Do not produce finite-SMC, readiness, checkpoint, metric, or H6-Prediction evidence: none is consumed by frozen H7-v1.

- [ ] **Step 6: Bind predecessor references and run the H7 click verifier exactly once.** Atomically write `.verification/h7-current-candidate-<FULL_HEAD>-refs.json` in exact order H1--H5, active H1-prefix-prior, H6-Prefix, with each artifact path plus manifest, relevant payload/certificate-set, validated ledger, revision/digest, and JUnit SHA. The H7 click path derives this exact full-HEAD registry path and fails closed on absence or extra keys; it never selects "latest" evidence.

  ```powershell
  $h7ClickOutput = @(& python verify_vfe4.py)
  $h7ClickOutput
  $h7ArtifactLine = @($h7ClickOutput | Where-Object { $_ -like 'artifact: *' })
  if ($LASTEXITCODE -ne 0 -or $h7ArtifactLine.Count -ne 1) {
      throw 'H7 click-run did not produce exactly one successful artifact line'
  }
  $h7Artifact = [System.IO.Path]::GetFullPath(
      $h7ArtifactLine[0].Substring('artifact: '.Length)
  )
  $h7Payload = Get-Content -Raw -LiteralPath (Join-Path $h7Artifact 'validation\h7.json') |
      ConvertFrom-Json
  $fixtureSetSha = [string]$h7Payload.fixture_set_sha256
  if ($fixtureSetSha -notmatch '^[0-9a-f]{64}$') {
      throw 'H7 artifact fixture_set_sha256 is invalid'
  }
  ```

  Expected: `H7: pass` and one artifact path. Independently recompute `manifest.sha256`; validate exact source/config/dependency/fixture/action/oracle identities and every nested canonical hash; require the three reference records and their exact current hashes; require `validation/h7.json`, both exact scalar-regression roles, all five positive-covariance roles, the sole expected-negative role with decisive-change predicate, both recognition origins and factorized-to-full promotion, joint `K0`, the complete frozen corresponding-probe set, typed source rows/all twelve scorer residuals, per-operand backward/budget records, and the twelve ordered control IDs. Require no borrowed tensor/action view, copied predecessor validation/certificate/ledger, H6-Prediction payload, or H8 file. Atomically record the exact artifact path, artifact manifest SHA, fixture-set SHA, refs-registry SHA, revision/digest, and JUnit SHA in `.verification/h7-current-candidate-<FULL_HEAD>-result.json`; this result pointer is not part of the H7 artifact manifest and therefore introduces no hash cycle.

- [ ] **Step 7: Have fresh reviewers consume the immutable evidence set only.** One checks scalar-versus-matrix typing, group/cocycle/`B`/decoder/stabilizer mathematics, direct solves, raw-versus-owned tensor boundaries, action hashes, and non-square type safety; one checks both recognition representations and generic-action promotion, joint `K0`, probability measures, exact typed history-scorer source-frame law, corresponding density probes, Jacobians/entropy, and local-monolithic/evidence-KL completeness; one checks the 100-decimal oracle, category/operand-local and per-backward budgets, immutable result/status consistency, exact role/control inventory, and predecessor/ledger provenance. They cite focused outputs, JUnit XML, the four current artifacts, source, and closed predecessor ledgers. They do not rerun tests or gates. A Critical/Important source defect invalidates this candidate and triggers the complete replacement lifecycle; it is never repaired in place.

- [ ] **Step 8: Start, populate, and validate only the revision-specific H7 closure ledger.** Re-read `.verification/h7-current-candidate-<FULL_HEAD>-result.json`, independently revalidate its artifact/manifest/fixture/ref/revision/JUnit hashes, and set `$h7Artifact` from that exact record. Recheck full `HEAD`, dirty digest, dependency closure, every predecessor ledger hash, and no active marker; then derive the new path:

  ```powershell
  $h7Head = (git rev-parse HEAD).Trim()
  if ($h7Head.Length -ne 40) { throw 'H7 requires a full 40-character HEAD' }
  if (-not (Test-Path -LiteralPath $h7Artifact)) { throw 'validated H7 artifact path is unavailable' }
  $fixtureSetSha = [string](
      Get-Content -Raw -LiteralPath (Join-Path $h7Artifact 'validation\h7.json') |
          ConvertFrom-Json
  ).fixture_set_sha256
  if ($fixtureSetSha -notmatch '^[0-9a-f]{64}$') { throw 'H7 fixture-set SHA is invalid' }
  $h7Ledger = ".verification/h7-$h7Head-$fixtureSetSha-ledger.json"
  if (Test-Path -LiteralPath '.verification/active.json') { throw 'existing verification activation blocks H7' }
  if (Test-Path -LiteralPath $h7Ledger) { throw "H7 ledger already exists: $h7Ledger" }
  & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" start --cwd . --ledger $h7Ledger --mode closure
  ```

  The fixture-set SHA and H7 artifact path are read from the validated immutable result record and artifact, not rediscovered by directory order or chosen as editable protocol fields.

  Populate one claim per H7 check: current reference-only predecessors and their closed ledgers; raw fixture/dependency closure; separately typed scalar `GL+(1)` regression and primary direct `GL+(2)` domain/envelope/det-negative nonclaim; exact trial roles/predicates/action hashes; diagonal-base/internal-product distinction; borrowed graph/autograd versus owned evidence/no-materialized-inverse boundary; `U`/receiver-source/cocycle/closed-walk order; correct and reverse-arrow `B`; all `(mu,Sigma,M,h,J)` laws; every generative transition law; structured recognition law and factorized-origin unrestricted-full-block promotion; exact typed source identity/separate z-m histories and twelve scorer residuals; decoder contragredience and centered-stabilizer scope; corresponding density probes and complete density/Jacobian/entropy/log-ratio laws; joint `K0` and every other local term; local/monolithic complete ELBO; scalar evidence/posterior KL; GH41/51 and 100-decimal oracle; category/operand budgets; `r_abs/r_rel`, every backward operand and `r_back_max`; immutable envelope/objective/result hashes and status consistency; all twelve exact controls; exact JUnit totals; atomic H7 artifact/manifest; and preservation of every predecessor ledger hash. Do not duplicate H1--H5 or H6-Prefix correctness claims in the H7 ledger; record their validated ledgers as provenance evidence.

  Use mathematics-domain derivation evidence for transformation identities, current mechanical/reproduced-output evidence for code and gate claims, and current source evidence for artifact/provenance claims. Every assessed claim has at least two distinct views and one structured adjudicator; high/critical or disputed claims follow the verification skill's escalation/skeptic rules. Missing eligible evidence is INCONCLUSIVE, never LLM-supported closure or majority vote. Validate the exact printed ledger:

  ```powershell
  & "C:\Python314\python.exe" "C:\Users\chris and christine\.codex\skills\verification\scripts\verification_gate.py" validate $h7Ledger --cwd .
  ```

- [ ] **Step 9: Recheck immutability and report the complete evidence revision.** Recompute `HEAD`, dirty digest, tracked/index diffs, dependency closure, fixture-set/JUnit/artifact/manifest/reference hashes, all four new ledger hashes, and every historical ledger hash. Expected: tracked source is unchanged; the only new evidence is the exact ordered current-candidate set; every ledger validates at the same artifact revision. Report exact `(git_head,dirty_digest)`, XML-derived totals, all predecessor artifact/ledger paths, H6-Prefix certificate-set SHA, H7 artifact/fixture-set SHA, maximum residual only beside its own allowance, exact control inventory, and validated H7 ledger path. State explicitly that H7 references rather than copies predecessors and that H6-Prediction was not required or produced. Do not edit tracked documentation or rerun any test/gate after closure.

## Out of Scope for This Plan

- H8 sparse-scale execution, `T=128`, `K=20`, dense-allocation policing, or permission to expand sequence/parent-set complexity.
- Training, optimizer equivariance, natural-gradient/frame-update equivalence, SGD/Adam behavior, parameter-update covariance, or gradient-trajectory claims.
- Predictive improvement, H6 metric reinterpretation, new tuning, checkpoint selection, or test-set reopening.
- Positive-dimensional base geometry, base parallel transport, base curvature, base holonomy, or independent graph-link curvature.
- Orientation-reversing `det<0` frame changes or a claim over full disconnected `GL(2)`.
- A claim that differential entropy is invariant, that coordinate densities are scalars, or that a fixed decoder respects the full internal product group.
- A source-mixture moment projection, projecting a factorized-origin generic pushforward back to diagonal/same-family form, jitter, clipping, pseudo-inverse, resampling, or numerical repair.
- A second training/objective implementation, changes to H2 detached snapshots, or use of the H7 monolithic diagnostic as a training loss.
- Research-vault ingestion. The implementation may offer the completed H7 result for later ingest, but no vault write belongs to this plan.

## Self-Review of Plan Completeness

- **Spec coverage:** Tasks 1--9 cover separately typed `GL+(1)` scalar complete-law regression and primary direct `GL+(2)` matrix evidence; standard `d_z=d_m=2` representations; `T=2,D=12,V=3`; frozen diagonal/internal action bytes and hashes; identity/nonidentity frames; receiver/source link and cocycle order; correct versus reverse-arrow `B` plus non-square channel typing; all moment/information laws through direct operand solves with no materialized inverse; graph-preserving borrowed views versus owned evidence snapshots; exact `StructuredLanguageRecognition | FactorizedLanguageRecognition` dispatch and fixtures; unrestricted full-block promotion of a factorized origin under generic actions with projection/same-family rejection; transition offsets/covariances/precisions/Jacobians; decoder contragredience/bias/centered stabilizer; bank/channel/receiver/source-typed scorer rows, separate z/m histories, twelve scorer residuals, and source-frame mutant; frozen corresponding density-probe pairs; joint initial `K0`; density/entropy/log-ratio shifts; local/monolithic/evidence-KL obligations; immutable envelope/objective/gate records; 100-decimal GH41/51 oracle; category/operand-local and per-backward budgets; `r_abs/r_rel/r_back_max`; closed positive/expected-negative roles; the twelve ordered control IDs; status precedence; optional-H1 byte behavior; reference-only predecessors; one JUnit; and revision-specific ledgers.
- **Task ordering:** Protocol and bytes freeze before any calculation; pure tensor laws precede factor pushforwards; generative and recognition completeness precede objective comparison; objective completeness precedes the independent oracle/gate; all tracked work and reviews finish before the final source freeze. The sole candidate JUnit then precedes exact-revision H1--H5, active H1-prefix-prior, independent H6-Prefix with `predecessor_refs={}`, and H7 artifact/ledger production in that order, with no later tracked edit. That order is an H7 registry order, not an H1--H5/H4 dependency inside Prefix.
- **Type consistency:** `H7ScalarReplayAction` and `H7GLPlus2Action` remain disjoint owned members of `H7TensorActionSnapshot`; direct transforms consume only `H7BorrowedActionView`. Trial specs carry matching dimension/group/action hash, closed role, and expected predicate. Recognition consumes only the explicit two-class post-H6 union, and a factorized origin has an explicit unrestricted-full-block output representation under generic actions. Typed source rows retain separate channel histories/covectors. H6 projected producers return `CandidateArtifactReference`; H7 converts validated sibling producer references into its exact predecessor-reference type without changing payload hashes. The same owned snapshots, probe pairs, joint-`K0`, scorer records, predecessor-reference type, operand budget/backward/residual/trial/control records, and sole `results.py::H7GateResult` flow through every task. H2 types remain unchanged and detached.
- **Decision completeness:** Every required scalar/positive invariant has a category- and operand-local allowance; every backward operand is recorded before aggregation; every GH-dependent invariant has a 41/51 convergence record; the expected-negative trial and every numeric control have an exact decisive-change rule; finite valid positive violations FAIL; covariance acceptance of the outside-stabilizer trial FAILS; missing/stale/out-of-envelope/unresolved/nondecisive evidence is INCONCLUSIVE; and PASS requires the complete role-aware, integrity-valid inventory.
- **Nonclaim completeness:** The plan explicitly excludes treating the scalar replay as `GL+(2)` evidence, det-negative `GL(2)`, optimizer/training equivariance, H6-Prediction/predictive benefit, H8 scale, base curvature/holonomy, fixed-decoder full-group symmetry, entropy invariance, and Research-vault writes.
- **Placeholder scan:** No fixture coefficient, action matrix, trial, control, threshold, status rule, oracle precision/order, or commit boundary remains to be chosen from outcomes. Runtime hashes/revisions/artifact paths are measured identities rather than adjustable scientific parameters.
- **Path check:** This plan is saved at `docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md`. The authoring task makes no code, test, config, fixture, preregistration, ledger, or commit change outside this file.
