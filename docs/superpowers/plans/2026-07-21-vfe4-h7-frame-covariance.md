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
- Where an independent evidence calculation exists, require evidence and posterior-KL invariance as well. The scalar `h1-v1` replay must retain its independent evidence-plus-posterior-KL identity. The matrix fixture must not fabricate an analytic evidence claim for the nonconjugate categorical emission; it reports that obligation as not applicable while still closing both complete ELBO paths.
- Production identity calculations use CPU float64. The independent oracle parses raw fixture bytes itself, imports no `vfe4`, PyTorch, NumPy, or production budget code, uses mpmath at exactly 100 decimal digits, enumerates exact source paths, and evaluates categorical-emission expectations at Gauss--Hermite orders 41 and 51.
- The required envelope is inclusive: every direct group element satisfies `||g_t||_2<=2` and `||g_t^{-1}||_2<=2`; every original and transformed SPD operand used by a required calculation satisfies `kappa_2<=1e3`. Record determinants, both group norms, SPD extreme eigenvalues, and condition numbers per operand. Never jitter, clip, pseudo-invert, regularize, resample, project, repair, or silently exclude a required operand.
- A required trial that is missing, stale, outside the envelope, has an unresolved 41/51 oracle comparison, or lacks an eligible predecessor is `INCONCLUSIVE`, not `FAIL`. A finite, valid, in-envelope covariance/density/objective violation is `FAIL`. H7 is `PASS` only when every required positive trial, tensor/law/local/complete-objective invariant, independent oracle comparison, and negative control passes.
- The complete negative-control inventory is fixed and ordered by the exact IDs `wrong_covariance_congruence`, `wrong_precision_congruence`, `history_scorer_wrong_source_inverse`, `reversed_link_order`, `reverse_arrow_B`, `wrong_decoder_dual_action`, `fixed_decoder_outside_stabilizer`, `omitted_density_jacobian`, `reversed_logdet_sign`, `entropy_false_invariance`, `changed_h1_source_probability`, and `diagonal_for_internal_action`. Source-support preservation is an additional exact required invariant with its own malformed-input tests. Every control must be decisive under its own matching operand-local budget; preregistration, tests, result records, and `validation/h7.json` use these IDs verbatim and in this order.
- `det(g)<0` is outside H7's declared `GL+(2)` domain. The parser and action constructor reject it before evaluation, and the artifact states that H7 makes no claim for the orientation-reversing component of `GL(2)`. Do not count rejection of a reflection as a covariance PASS over full `GL(2)`.
- Each implementation task runs only its named new or directly modified focused tests for RED/GREEN, then creates exactly one bounded commit. A task reviewer inspects that diff and the implementer's focused output without rerunning the same tests. Do not run a cumulative suite after each task.
- After every tracked H7 source/test/config/fixture/preregistration/schema edit and bounded review is complete, freeze one exact clean H7 candidate revision and prohibit every later tracked edit or commit in that candidate lifecycle. Run the full pytest suite with JUnit exactly once first. At that same revision and dirty-content digest, produce exactly once and in order: the current H1--H5 compatibility artifact/ledger; the H1-prefix-prior artifact/ledger only when the frozen H7 scorer profile is consumed (the required `h7-linear-history-source-v1` profile activates it); the current H6-Prefix certificate set/artifact/ledger; and finally the H7 artifact/ledger. If a later review discovers a source defect, preserve all evidence as history, commit a new candidate, and repeat the complete lifecycle with one replacement JUnit; never patch the frozen candidate or rerun for confidence.
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
      "latent_history": {
        "z": [[0.3, -0.2], [-0.15, 0.25]],
        "m": [[0.1, 0.4], [0.2, -0.3]]
      },
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
    "state_receiver_covariances": [[[0.5, 0.03], [0.03, 0.46]], [[0.58, 0.02], [0.02, 0.44]]]
  },
  "density_probes": {"whitened_scale": 0.25, "directions": "zero_and_signed_coordinate_basis"},
  "oracle": {"decimal_precision": 100, "gauss_hermite_orders": [41, 51]}
}
```

For every frame profile, generative parent maps are the represented direct coboundaries `Omega_tj=U_t U_j^{-1}` for the exact chain `0->1->2`. The parser also constructs all six ordered pair links solely for the cocycle/open-walk/closed-walk audit; those extra relation links are not generative factors. Receiver precisions are derived by checked linear solves from the frozen covariances and compared with independently transformed precision laws; they are not serialized as redundant adjustable inputs.

The source-scorer profile has bank order `(model,state)`, receiver order `t=(1,2)`, and the sole permitted source `j=t-1` in each frozen support row. For a bank `b`, `alpha_{b,t,j}(x_{<t})=a_{b,t}+c_{b,t}*sum_{ell=1}^{t} ell*(x_{ell-1}+1)` using the serialized `alpha_bias=a`, `alpha_token_scale=c`, and the first `t` entries of `prefix_tokens`; this arithmetic and the prefix bytes do not transform. The history vectors transform as `z_j'=g_j z_j` and `m_j'=g_j m_j`, while each serialized covector transforms by the source frame as `r_z'=solve(g_j^T,r_z)` and `r_m'=solve(g_j^T,r_m)`. For every bank/receiver row, H7 records separate covector-law and raw-score residuals before checking the unchanged singleton mask/probability. The `history_scorer_wrong_source_inverse` control runs under `matrix-nonidentity-internal-transformed` and deliberately uses the receiver inverse `g_t^{-T}` instead of `g_j^{-T}`; its raw-score residual, not the trivially unit singleton probability, must cross the control limit.

The centered decoder rows deliberately have rank two after concatenating state and model columns, while every centered row reads only the first coordinate in each channel. The nonidentity direct element `[[1,0],[0.2,1.1]]` therefore belongs to the centered-softmax emission stabilizer but not generally to the strict raw-readout stabilizer because the row-common linear functional changes. The diagonal matrix `[[1.2,0.2],[-0.1,0.9]]` lies outside that centered stabilizer and is the held-fixed-decoder negative trial.

### Required positive and negative trials

| Trial ID | Law/frame profile | Action | Decoder policy | Expected role |
|---|---|---|---|---|
| `scalar-base-transformed` | unchanged `h1-v1` | `GL+(1)` scalar base `1.25` | transform | Complete scalar-law regression, sources, evidence/KL; no `GL+(2)` evidence. |
| `scalar-internal-transformed` | unchanged `h1-v1` | `GL+(1)` scalars `(0.8,1.1,1.4)` | transform | Complete scalar internal-product regression; no `GL+(2)` evidence. |
| `matrix-identity-base-transformed` | `h7-v1`, identity `U` | frozen diagonal matrix | transform | Direct `GL+(2)` base action from identity frames. |
| `matrix-identity-internal-transformed` | `h7-v1`, identity `U` | frozen internal matrices | transform | Independent receiver/source action from identity frames. |
| `matrix-nonidentity-base-transformed` | `h7-v1`, nonidentity `U` | frozen diagonal matrix | transform | Direct base action on nonidentity frames. |
| `matrix-nonidentity-internal-transformed` | `h7-v1`, nonidentity `U` | frozen internal matrices | transform | Strongest complete internal-product trial. |
| `matrix-fixed-decoder-centered-stabilizer` | `h7-v1`, nonidentity `U` | frozen stabilizer element | fixed | Positive emission-kernel stabilizer trial; raw logits may differ by a row-common scalar, centered logits and probabilities may not. |
| `matrix-fixed-decoder-outside-stabilizer` | `h7-v1`, nonidentity `U` | frozen diagonal matrix | fixed | Required finite negative control; complete objective must change decisively. |

Every transformed-decoder positive trial evaluates both original and transformed laws. Every control is injected into a fresh transformed copy; controls never alter the correct production transformation or share mutated state.

## File Map and Dependency Boundaries

| Path | Responsibility |
|---|---|
| `vfe4/types/h7.py` | Immutable, dimension-honest `H7ScalarReplayAction`/`H7GLPlus2Action` types plus fixture/snapshot metadata, action/trial IDs, budget records, residuals, controls, and fail-closed gate result. |
| `vfe4/validation/fixtures/h7_v1.json` | Frozen full-matrix law, frame profiles, direct actions, decoders, and oracle settings. |
| `vfe4/validation/h7_fixture.py` | Strict raw-byte parser, scalar adapter, matrix complete-law builder, frame-profile builder, probe builder, and post-H6 snapshot parity checks. |
| `vfe4/geometry/group_action.py` | Pure differentiable direct-group tensor transformations using left/right solves without materialized inverses, block assembly, log-Jacobian, frame/link composition, and centered-softmax projector. |
| `vfe4/geometry/__init__.py` | Public export of the H7 tensor-action seam only. |
| `vfe4/generative/pushforward.py` | Complete immutable generative pushforward: frames, links, maps, `B`, offsets, covariance/precision, the exact frozen linear history scorer, decoder, and density-shift metadata. |
| `vfe4/generative/language.py` | Export the existing post-H6 concrete normalized model as `LanguageGenerativeModel` if that exact public name is absent; no factor, parameter, or arithmetic change. |
| `vfe4/recognition/pushforward.py` | Complete immutable recognition pushforward with overloads for exactly `StructuredLanguageRecognition` and `FactorizedLanguageRecognition`; source tables/scorers and every continuous initial/conditional component are preserved without projection. |
| `vfe4/recognition/language.py` | Read-only owner of the exact post-H6 `StructuredLanguageRecognition` and `FactorizedLanguageRecognition` concrete types. H7 adds no alias and makes no arithmetic or conditioning edit here. |
| `vfe4/objective/h7_covariance.py` | Calls the existing complete local objective, adds the independent monolithic expectation and density probes, and emits term/density/entropy/evidence/KL covariance diagnostics without defining a new training objective. |
| `verification/mp_oracles/h7_covariance.py` | Independent JSON/mpmath-only 100-decimal exact-source oracle with analytic Gaussian reductions and GH41/GH51 emission expectations. |
| `verification/h7_budget.py` | Frozen operand-local forward, comparison, backward, oracle, and control-decisiveness allowances. |
| `verification/h7_gate.py` | Predecessor validation, fixture capture, trial execution, envelope checks, oracle comparisons, controls, status precedence, and `validation/h7.json`. |
| `vfe4/config/schema.py` | Frozen `H7ValidationConfig` and predecessor-reference section. |
| `vfe4/config/resolve.py` | Exact H7 literals, ordered prefix through H7, canonical hashing, and rejection of H8/det-negative/unsupported profiles. |
| `verification/run_gates.py` | Pure current-candidate H1--H5, conditional H1-prefix-prior, and H6-Prefix projection entry points plus the H7-only selected operation after predecessor-reference validation; one-time H1/H7 byte capture and atomic H7 publication. |
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

@dataclass(frozen=True)
class H7ScalarReplayAction:
    elements: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # each shape (1, 1)
    kind: H7ActionKind
    dimension: Literal[1]
    group: Literal["GL+(1,R)"]
    representation: Literal["standard_scalar"]

@dataclass(frozen=True)
class H7GLPlus2Action:
    elements: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    kind: H7ActionKind
    dimension: Literal[2]
    group: Literal["GL+(2,R)"]
    representation: Literal["direct_gl_plus_2"]

H7TensorAction: TypeAlias = H7ScalarReplayAction | H7GLPlus2Action

@dataclass(frozen=True)
class H7SourceContextSnapshot:
    prefix_tokens: tuple[int, ...]
    latent_history: tuple[tuple[float, ...], ...]
    source_scorer_profile: Literal["h7-linear-history-source-v1"] | None
    source_scorer_sha256: str | None
    context_sha256: str

@dataclass(frozen=True)
class H7RecognitionContextSnapshot:
    observation_labels: tuple[int, ...]
    conditioning: Literal["filtering", "smoothing"]
    context_sha256: str

@dataclass(frozen=True)
class H7BudgetRecord:
    invariant_id: str
    operation_count: int
    scale: float
    condition_product: float
    rounding_allowance: float
    oracle_convergence_allowance: float
    reference_rounding_allowance: float
    total_allowance: float

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

@dataclass(frozen=True)
class H7TrialResult:
    trial_id: str
    fixture_id: Literal["h1-v1", "h7-v1"]
    frame_profile: H7FrameProfile
    action_kind: H7ActionKind
    action_dimension: Literal[1, 2]
    group_domain: Literal["GL+(1,R)_scalar_replay", "GL+(2,R)_primary"]
    decoder_policy: H7DecoderPolicy
    logabsdet_measure_shift: float
    r_abs: H7ResidualRecord
    r_rel: H7ResidualRecord
    r_back: H7ResidualRecord
    residuals: tuple[H7ResidualRecord, ...]
    envelope: H7EnvelopeRecord

@dataclass(frozen=True)
class H7ControlResult:
    control_id: str
    target_invariant_id: str
    wrong_residual: float
    invariant_scale: float
    matching_correct_allowance: float
    decisiveness_limit: float
    detected: bool

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

@dataclass(frozen=True)
class H7GateResult:
    gate: Literal["H7"]
    status: GateStatus
    fixture_hashes: Mapping[str, str]
    predecessor_references: Mapping[str, H7PredecessorReference]
    trials: tuple[H7TrialResult, ...]
    controls: tuple[H7ControlResult, ...]
    obligations: tuple[str, ...]
```

`H7GateResult` is a separate explicit result type. PASS requires the exact positive trial IDs and exact control IDs from this plan, no obligations, and every nested residual/envelope/control passing. FAIL requires at least one finite in-envelope failed positive invariant. INCONCLUSIVE requires at least one named obligation and cannot contain a claimed finite refutation unless that refutation is separately represented.

```python
# vfe4/geometry/group_action.py
def require_direct_gl_plus(element: torch.Tensor, *, dimension: int) -> torch.Tensor: ...
def block_population_action(action: H7TensorAction) -> torch.Tensor: ...
def right_solve(value: torch.Tensor, right: torch.Tensor) -> torch.Tensor: ...
def push_vector(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_covariance(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_precision(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_information_vector(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_second_moment(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def push_receiver_source_map(value: torch.Tensor, receiver: torch.Tensor, source: torch.Tensor) -> torch.Tensor: ...
def push_same_receiver_morphism(value: torch.Tensor, state_receiver: torch.Tensor, model_receiver: torch.Tensor) -> torch.Tensor: ...
def push_decoder(value: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor: ...
def compose_reframed_frames(action: H7TensorAction, frames: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...
def frame_links(frames: tuple[torch.Tensor, ...]) -> Mapping[tuple[int, int], torch.Tensor]: ...
def logabsdet_measure_shift(action: H7TensorAction) -> torch.Tensor: ...
def centered_logit_projector(vocabulary_size: int, *, like: torch.Tensor) -> torch.Tensor: ...
```

Every inverse action uses a solve applied directly to the actual operand. Right multiplication is exactly `right_solve(value,G)=torch.linalg.solve(G.T,value.T).T`; covectors use `torch.linalg.solve(G.T,value)`; a two-sided precision push is `right=torch.linalg.solve(G.T,J.T).T` followed by `torch.linalg.solve(G.T,right)`. Links, affine maps, `B`, decoders, precision laws, and backward recovery compose those right/left solves. Production must not call `torch.linalg.inv`/`pinv`, must not compute `torch.linalg.solve(G,I)`, and must not create an identity right-hand side solely to materialize an inverse. Validation checks are differentiable comparisons/guards and may inspect detached scalar booleans only to reject invalid domains; returned tensors retain the original graph.

```python
# vfe4/generative/pushforward.py
def snapshot_h7_generative(
    model: LanguageGenerativeModel,
    *,
    context: H7SourceContextSnapshot | None,
) -> H7GenerativeSnapshot: ...
def pushforward_generative(
    snapshot: H7GenerativeTensorLaw,
    action: H7TensorAction,
    *,
    decoder_policy: H7DecoderPolicy,
) -> H7GenerativeTensorLaw: ...

# vfe4/recognition/pushforward.py
H7RecognitionInput: TypeAlias = (
    StructuredLanguageRecognition | FactorizedLanguageRecognition
)
def snapshot_h7_recognition(
    law: H7RecognitionInput,
    *,
    context: H7RecognitionContextSnapshot,
) -> H7RecognitionSnapshot: ...
def pushforward_recognition(
    snapshot: H7RecognitionTensorLaw,
    action: H7TensorAction,
) -> H7RecognitionTensorLaw: ...
```

`LanguageGenerativeModel` is the normalized public model exported by post-H6 `vfe4/generative/language.py`. Recognition accepts the explicit read-only union of the exact post-H6 `StructuredLanguageRecognition` and `FactorizedLanguageRecognition` concrete classes; H7 must not invent a `LanguageRecognitionLaw` alias. Dispatch is by those exact classes (or overloads with those exact parameters), and rejects projection-tagged/moment-projected recognition records, emission-only objective records, arbitrary duck-typed objects, and unsupported dynamic factor types before snapshotting. The H1 replay enters through `adapt_h1_fixture_bytes`, not through an `object`-typed branch. Snapshot construction freezes the complete factor bytes and canonical hash before transformation. Tensor-law conversion is separate and does not detach. The transformed law is a new object; the original snapshot and post-H6 model must retain identical hashes.

```python
# vfe4/objective/h7_covariance.py
def evaluate_h7_complete_covariance(
    original: H7CompleteLawSnapshot,
    transformed: H7CompleteLawSnapshot,
    action: H7TensorAction,
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

`evaluate_h7_complete_covariance` calls the existing post-H6 complete local objective and its factor trace. Its monolithic path independently evaluates `E_q[log p-log q]` from the same complete law and exact source paths. It never supplies a scalar used by training or H5 acceptance.

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

Each invariant records its exact actual operation count from the frozen category formulas below; it may not borrow another invariant's scale, SPD condition number, or oracle delta.

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
r_back = max_u ||T_G^-1(T_G(u))-u||_F/max(1,||u||_F)
```

The `r_back` operand inventory is exact: all `U`; all ordered-pair `Omega`; all initial/transition/recognition means, offsets, parent maps, `B`, covariances, precisions, `(h,J,M)` objects, decoder weights, and any history-reading source covectors. Categorical probabilities, source support, observation labels, and biases are compared by exact identity and do not enter the normalized Frobenius maximum.

Status precedence is exact:

1. Validate source/config/fixture/predecessor identities and required inventory. Missing or stale input is INCONCLUSIVE.
2. Validate `GL+`, norm, SPD, and condition envelopes. A required out-of-envelope trial is INCONCLUSIVE; an invalid serialized fixture is INCONCLUSIVE because the intended law was not evaluated.
3. Validate the 100-decimal oracle and GH41/51 convergence. Missing or unresolved oracle evidence is INCONCLUSIVE.
4. Evaluate every positive invariant. A finite, valid, in-envelope residual above its own allowance is FAIL.
5. Evaluate every negative control against `control_decisiveness_limit`. A finite wrong path not separated from the correct path is INCONCLUSIVE because the control is nondecisive; a control that the production gate incorrectly accepts as covariant is FAIL.
6. PASS requires all current compatible predecessor references, both separately labeled scalar-regression trials, all five primary matrix positive trials, the outside-stabilizer expected-negative trial, all twelve exact control IDs, all exact source/scorer identities, and no obligation.

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

**Interfaces:** Produce the H7 records and parser named above, the disjoint `H7ScalarReplayAction | H7GLPlus2Action` union, `parse_h7_fixture_bytes(data: bytes) -> H7Fixture`, `adapt_h1_fixture_bytes(data: bytes) -> H7CompleteLawSnapshot`, and a frozen `H7ValidationConfig`. Add `mpmath>=1.3` as an oracle dependency; production modules never import it.

- [ ] **Step 1: Write the exact preregistration and `h7_v1.json` bytes.** Copy the complete JSON, linear history-scorer equation/coefficient arrays, and every global/budget/status/control literal from this plan. Record the exact required trial/control ID order, the separate scalar `GL+(1)` regression versus primary matrix `GL+(2)` claim, the `det<0` nonclaim, the scorer source-frame rule, and the centered-softmax stabilizer distinction. Do not run a parser, determinant, condition-number, objective, or oracle calculation before both files exist.

- [ ] **Step 2: Write failing fixture/type/config tests.** Require exact root fields, group/representation/dimension/order, chain parent/support, exact matrices, scorer profile/law/prefix/coefficient/history bytes, exact trial and twelve-control ID order, immutable defensive access, SPD factors, positive group determinants, direct-element literals, exact GH/precision settings, and fail-closed result consistency. Require scalar fixture adaptation to construct only `H7ScalarReplayAction(dimension=1,group="GL+(1,R)")` with `(1,1)` elements and matrix parsing to construct only `H7GLPlus2Action(dimension=2,group="GL+(2,R)")`; reject cross-construction, a claimed scalar `GL+(2)` result, or an action whose declared dimension/group disagrees with its shapes. Reject unknown/missing fields, booleans as numbers, NaN/Inf, wrong shapes/order, non-SPD covariances, any `det<=0` action/frame, any changed direct matrix/scorer coefficient, nonstandard representation, changed parent/source support, a decoder that loses the stated centered-stabilizer property, or any H8 gate/config key. Assert every shorter predecessor prefix still resolves without reading H7 bytes or carrying H7 config/provenance fields.

- [ ] **Step 3: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_fixture.py tests/unit/test_config.py -q
  ```

  Expected: collection/config failures show the H7 types/parser/config do not exist and the resolver stops at the post-H6 prefix.

- [ ] **Step 4: Implement strict records, parser, H1 adapter, and config.** Parse exact schema sets. Build `U_t U_j^{-1}` as `torch.linalg.solve(U_j.T,U_t.T).T`; build derived precisions with the two-sided operand solves frozen above; never pass an identity right-hand side to `solve`. Enumerate exact chain sources, evaluate the exact scorer prefix law without transforming `alpha`, and create density probes from zero plus signed coordinate-basis whitened directions at scale `0.25`. H1 adaptation must preserve the raw H1 fixture hash, exact four-path order, one-based labels, and every existing factor value while returning only the scalar action type. Config resolution accepts only the prior exact prefixes plus the full ordered tuple `("H1","H2","H3","H4","H5","H6-Prefix","H7")`; the H7 operation references predecessor artifacts and runs H7 only.

- [ ] **Step 5: Freeze raw fixture hashes before any H7 calculation.**

  ```powershell
  Get-FileHash -Algorithm SHA256 vfe4/validation/fixtures/h1_v1.json
  Get-FileHash -Algorithm SHA256 vfe4/validation/fixtures/h7_v1.json
  ```

  Copy the two exact lowercase digests into named parser/config constants and the preregistration in this same task. No normalized/re-serialized hash is accepted.

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

**Consumes:** the explicit `H7ScalarReplayAction | H7GLPlus2Action` union and raw float64 tensors. **Produces:** every public tensor transform frozen above; no snapshot or objective dependency.

- [ ] **Step 1: Write failing group-action tests.** Cover all exact vector/covariance/precision/information/second-moment/map/`B`/decoder laws against hand-computed `GL+(1)` scalar and `GL+(2)` matrix values; require the two action classes, dimensions, shapes, and group labels to remain disjoint. Cover `U'=gU`; receiver/source link order; `Omega_21 Omega_10=Omega_20`; the closed relation walk; diagonal versus internal actions; global block order; `logJ_G`; centered projection; positive determinants; `det<0` rejection; and no `matrix_exp` call. For the frozen equal-dimensional matrix operand, prove the required reverse-arrow mutant `G_m^{-1} B G_z` differs from correct `G_z B G_m^{-1}`. Add an independent non-square type case with `B` shape `(d_z,d_m)=(2,3)`, `G_z` shape `(2,2)`, and `G_m` shape `(3,3)`; require the correct output shape `(2,3)` and reject `B.T`, swapped state/model receivers, or any channel-swap adapter before arithmetic.

  Patch `torch.linalg.inv`, `torch.linalg.pinv`, and `torch.matrix_exp` to raise during correct transforms. Wrap `torch.linalg.solve` and raise if a call receives a square identity right-hand side whose sole purpose would be forming an inverse; still allow ordinary vector/matrix operand right-hand sides. Assert `right_solve(value,G)` is exactly `solve(G.T,value.T).T` and inspect the two-sided precision/link/map/decoder paths to prove none creates `eye_like(G)` for a solve.

  Add a gradient test with independent `requires_grad=True` operands and direct elements. Sum nonconstant entries from every transform and call `torch.autograd.grad`; require finite, non-`None`, nonzero gradients for the operand and group element. Inspect `grad_fn` and prohibit detached/cloned outputs.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_group_action.py -q
  ```

  Expected: collection fails because `vfe4.geometry.group_action` does not exist.

- [ ] **Step 3: Implement the minimal pure tensor laws.** Implement `right_solve(value,G)` only as `torch.linalg.solve(G.T,value.T).T`; use `torch.linalg.solve(G.T,value)` for covectors; form two-sided inverse actions by composing right solve with a left solve on the actual operand. Never call `solve(G,I)` or construct an identity RHS to obtain an inverse. Preserve dtype/device and autograd. Validate `H7ScalarReplayAction` as exactly three `(1,1)` positive elements and `H7GLPlus2Action` as exactly three `(2,2)` direct elements. `require_direct_gl_plus` rejects wrong rank/shape/dtype, nonfinite entries, singular elements, and nonpositive `slogdet` sign; it returns the original tensor reference rather than a detached copy.

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

**Consumes:** the post-H6 normalized generative model or H1/H7 fixture adapter plus Task 2 tensor laws. **Produces:** immutable `H7GenerativeSnapshot`, graph-preserving `H7GenerativeTensorLaw`, and complete pushed-forward law/density metadata.

- [ ] **Step 1: Write failing generative tests.** For both separately typed scalar-regression actions and every primary matrix frame/action profile, assert transformed `U`, every ordered link, causal transition map, `B`, offset, covariance, derived precision, initial law, `(h,J,M)`, decoder, bias, support, and source probability. Require each transition location to satisfy `mu_tj'=g_t mu_tj` on the exact density probes and each normalized receiver log density to shift by `-logdet(g_t)` per channel.

  Fixed source tables must have identical canonical bytes. Exercise exactly `h7-linear-history-source-v1`: for both banks and `t=(1,2)`, recompute the serialized `alpha` formula from the frozen prefix, require `r_z'=solve(g_j.T,r_z)` and `r_m'=solve(g_j.T,r_m)`, and record separate covector and raw-score residuals before proving mask/support/order/probabilities unchanged. Under `matrix-nonidentity-internal-transformed`, inject unchanged covectors and the exact gate mutant `r'=solve(g_t.T,r)` in place of the source-frame solve; both unit cases must change a raw score, and the latter supplies control ID `history_scorer_wrong_source_inverse`. Test transformed decoder logits exactly, fixed decoder centered-stabilizer probabilities, and outside-stabilizer probability change. Hash the source model before and after every transformation and require no mutation.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_generative_pushforward.py -q
  ```

  Expected: collection fails because the generative snapshot/pushforward interface is absent.

- [ ] **Step 3: Implement snapshot and complete pushforward.** Consume the existing post-H6 `LanguageGenerativeModel`. Snapshot every model factor required by the existing complete objective, including exact prefix/context bytes, scorer profile ID, `alpha` coefficients, latent histories, source covectors, support, mask identity, and source-scorer canonical hash. Transform scorer covectors only with their source block through left solves; keep `alpha` and categorical identities byte-identical. Transform all other continuous/model/geometric factors with direct operand solves. For `decoder_policy="fixed"`, leave weights/bias untouched and record whether the exact centered-stabilizer equations hold; do not silently coerce an outside action into a positive trial.

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

**Consumes:** exactly post-H6 `StructuredLanguageRecognition | FactorizedLanguageRecognition` filtering/smoothing laws and H1/H7 adapters. **Produces:** immutable and tensor-facing complete recognition pushforwards without introducing a recognition alias.

- [ ] **Step 1: Write failing recognition tests.** Parameterize the same complete checks over exact instances of `StructuredLanguageRecognition` and `FactorizedLanguageRecognition`. Cover separately typed scalar and matrix initial means/covariances/precisions/information vectors/second moments; every model and state conditional parent map, state-model map, offset, covariance, and precision; exact source order/probabilities/support; joint component `(mu,Sigma,h,J,M)` laws; per-conditional and complete recognition density shifts; and the `+logJ_G` continuous entropy shift. Require original snapshot/model/optimizer hashes unchanged.

  Assert the public signature is the explicit concrete union (or overloads with those exact two parameter types), not `object`, `Any`, a nonexistent `LanguageRecognitionLaw`, or an open structural alias. Pass a projection-tagged/moment-projected recognition record, an emission-only objective record, and a lookalike object exposing only moments/emissions; each must fail closed before factor access. This prevents H7 from silently proving covariance of a projected Gaussian or partial emission record instead of the normalized complete recognition law.

  Import and exercise every public H2 type/factor/evaluator before and after the H7 transformation. Require identical H2 canonical values, detached ownership, public signatures, and no `requires_grad`; H7 tensor gradients must use only the separate tensor-law record.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_recognition_pushforward.py -q
  ```

  Expected: collection fails because the recognition pushforward does not exist.

- [ ] **Step 3: Implement complete recognition pushforward.** Import the existing concrete `StructuredLanguageRecognition` and `FactorizedLanguageRecognition` types read-only and dispatch through the exact union/overloads. Snapshot every complete factor for the matched family and reject all other runtime classes; do not edit `language.py` or create an alias. Transform each source-conditioned component before exact source marginalization; never moment-project the mixture. Keep fixed categorical weights byte-identical. Compute entropy shifts from conditional receiver Jacobians and source weights, then independently compare with the global `logJ_G` shift.

- [ ] **Step 4: Run focused GREEN.** Run the Step 2 command. Expected: all component/joint/density/entropy/source/H2-preservation tests pass.

- [ ] **Step 5: Have one reviewer inspect concrete-type dispatch, mixture semantics, and the H2 boundary.** Reject any invented alias/open duck type, projection/emission-only acceptance, component omission, source-marginal Gaussian projection, in-place mutation, or H2 differentiability change.

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

- [ ] **Step 1: Write failing complete-objective tests.** Require the exact term inventory: expected emissions by time, initial model/state contributions, model/state source terms, model/state transition terms by time, joint recognition entropy, complete local ELBO, monolithic ELBO, complete `p` and `q` density shifts, pointwise log ratio, `r_abs`, `r_rel`, and every tensor-law diagnostic. The matrix path additionally requires all twelve `h7-linear-history-source-v1` residuals--eight covector-law residuals (`r_z` and `r_m` for two banks at two receiver times) plus four raw-score residuals--before accepting its unchanged source terms. The separately labeled scalar regression additionally requires evidence, posterior KL, and `log evidence-ELBO=KL` before/after. The matrix fixture marks evidence/KL not applicable with a fixed reason and no fabricated value.

  Evaluate zero plus signed-basis density probes for every exact source component. Require `log p' = log p-logJ_G`, `log q'=log q-logJ_G`, and pointwise log-ratio equality. Prove that removing any local term, using only transitions/latents, or checking only the final scalar makes the completeness validator reject the evaluation.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/unit/test_h7_complete_objective.py -q
  ```

  Expected: collection fails because the H7 complete-objective diagnostic is absent.

- [ ] **Step 3: Expose, do not replace, the existing complete objective.** If `language_elbo.py` lacks an importable immutable complete-term/factor trace, expose its existing evaluator without changing its scalar arithmetic or training call sites. Implement H7's monolithic expectation separately from the local partition, using exact source enumeration and deterministic quadrature. The H7 module compares the two; it never supplies a loss to training or H5.

- [ ] **Step 4: Implement density, entropy, and available evidence/KL diagnostics.** Record initial and receiver shifts before the global shift, exact source identities, complete density/log-ratio probes, local terms, both complete scalars, and applicability. Preserve the H1 independent evidence path and its original normalization conventions.

- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: complete scalar/local/density/term/evidence checks pass; every intentionally partial inventory is rejected.

- [ ] **Step 6: Have one reviewer inspect objective completeness and signs.** Compare the term inventory with the authoritative post-H6 objective and manuscript pushforward proof; inspect source entropy, emission normalizers, determinant/Jacobian signs, and H1 evidence/KL orientation.

- [ ] **Step 7: Commit Task 5.**

  ```powershell
  git add vfe4/objective/h7_covariance.py vfe4/objective/__init__.py vfe4/objective/language_elbo.py tests/unit/test_h7_complete_objective.py
  git commit -m "feat: evaluate complete H7 objective covariance"
  ```

## Task 6: Add the Independent 100-Decimal Oracle and Operand-Local Budgets

**Files:**

- Create: `verification/mp_oracles/__init__.py`
- Create: `verification/mp_oracles/h7_covariance.py`
- Create: `verification/h7_budget.py`
- Create: `tests/oracle/test_h7_mp_oracle.py`
- Modify: `docs/preregistrations/2026-07-21-h7-frame-covariance.md`

**Consumes:** raw H1/H7 JSON bytes only. **Produces:** independent original/transformed tensor/law/local/monolithic/density/oracle values and exact budget records.

- [ ] **Step 1: Write failing oracle/budget tests.** Require no `vfe4`, torch, or NumPy imports in the oracle module. Verify the 100-decimal Jacobi-matrix Gauss--Hermite construction: the physicists-Hermite Jacobi matrix has zero diagonal and off-diagonal entry `sqrt(k/2)` between zero-based rows `k-1` and `k` for `k=1..n-1`; mpmath `eigsy` returns nodes `x_i` and orthonormal eigenvectors; standard-normal nodes are `sqrt(2)*x_i`; normalized weights are the squared first eigenvector components and sum to one. Test exact polynomial moments through degree 12.

  Require exact source enumeration, independent evaluation of the serialized `alpha` formula and every original/transformed history-covector/raw-score value, independent affine Gaussian assembly, independent covariance/precision/information transformations, receiver/global log-Jacobian shifts, all local and monolithic terms, scalar evidence/KL, and the two-dimensional logit-contrast emission integral at orders 41/51. Test every budget formula/category, exact operand names, no global condition maximum, GH convergence boundary, and control decisiveness boundary.

- [ ] **Step 2: Run focused RED.**

  ```powershell
  python -m pytest tests/oracle/test_h7_mp_oracle.py -q
  ```

  Expected: collection fails because the mpmath oracle and H7 budget do not exist.

- [ ] **Step 3: Implement the independent oracle.** Set `mp.mp.dps=100` inside the entry call and restore the caller's precision afterward. Parse JSON numbers as decimal strings into `mp.mpf`. Use mpmath matrices and `lu_solve` directly on each actual vector/matrix right-hand side, never an identity RHS created solely to form an inverse. Independently evaluate the exact scorer prefix arithmetic and source-covector solves, Cholesky/eigendecomposition checks, analytic Gaussian expectations, exact categorical sums, and local/monolithic reductions. Reduce each selected emission log-softmax to its two Gaussian logit contrasts; do not call a production projector or quadrature helper.

- [ ] **Step 4: Implement exact operand-local budget records.** Require positive operation counts/scales/conditions, exact category formulas, finite allowances, and oracle deltas only on GH-dependent invariants. A budget constructor receives named operands rather than a run-wide condition summary.

- [ ] **Step 5: Run focused GREEN.** Run the Step 2 command. Expected: independent analytic/GH/oracle/budget tests pass; every required GH41/51 delta clears the frozen relative limit.

- [ ] **Step 6: Record only preregistered calibration facts.** Add the measured raw fixture hashes, required operand condition extrema, and GH41/51 deltas to the preregistration. Do not alter a threshold, fixture byte, action, trial, or control after seeing those values. If a frozen trial is outside the envelope or GH convergence is unresolved, stop this candidate as INCONCLUSIVE rather than tuning it.

- [ ] **Step 7: Have one reviewer inspect oracle independence and numerical contracts.** Check imports, source enumeration, contrast reduction, 100-decimal arithmetic, quadrature construction, budget locality, and no threshold tuning.

- [ ] **Step 8: Commit Task 6.**

  ```powershell
  git add verification/mp_oracles/__init__.py verification/mp_oracles/h7_covariance.py verification/h7_budget.py tests/oracle/test_h7_mp_oracle.py docs/preregistrations/2026-07-21-h7-frame-covariance.md
  git commit -m "test: add the H7 high precision oracle"
  ```

## Task 7: Build the Fail-Closed H7 Gate and All Decisive Controls

**Files:**

- Create: `verification/h7_gate.py`
- Modify: `vfe4/types/results.py`
- Create: `tests/promotion/test_h7_gate.py`
- Modify: `docs/preregistrations/2026-07-21-h7-frame-covariance.md`

**Consumes:** Tasks 1--6 plus immutable predecessor references. **Produces:** one `H7GateEvaluation(result, validation_payload, fixture_set_sha256, dependency_closure_sha256)`.

- [ ] **Step 1: Write failing gate tests for every positive trial and residual family.** Require the exact two `GL+(1)` scalar-regression IDs, exact five primary `GL+(2)` positive IDs, and the outside-stabilizer expected-negative trial. Require scalar records to carry only `action_dimension=1/group_domain="GL+(1,R)_scalar_replay"` and matrix records only `action_dimension=2/group_domain="GL+(2,R)_primary"`; prove scalar results cannot satisfy the matrix inventory. For each positive trial require envelope, `r_abs`, `r_rel`, `r_back`, all tensor laws, cocycle/open/closed products, all local terms, source identities, all twelve scorer residuals where applicable, decoder/logit/probability laws, density/Jacobian/entropy shifts, local/monolithic complete values, and applicable evidence/KL. Assert the matrix internal trial cannot pass when only latent/transition residuals or source probabilities without raw-score residuals are supplied.

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

- [ ] **Step 3: Write fail-closed status tests.** Independently inject missing/stale predecessor, fixture hash mismatch, missing trial/control, out-of-envelope group/SPD, unresolved GH delta, nonfinite calculation, finite in-envelope covariance failure, finite local-term failure, and incomplete density probes. Require the exact PASS/FAIL/INCONCLUSIVE precedence from this plan. `det<0` remains rejected/outside-domain and never creates a full-GL PASS claim.

- [ ] **Step 4: Run focused RED.**

  ```powershell
  python -m pytest tests/promotion/test_h7_gate.py -q
  ```

  Expected: collection fails because `verification.h7_gate` does not exist.

- [ ] **Step 5: Implement predecessor and dependency-closure validation.** Validate predecessor references only when each artifact and its validated ledger has the exact final H7 `git_head`, `dirty_digest`, candidate JUnit SHA, producer schema, manifest, payload/certificate-set hash, and PASS status. Require the ordered current H1--H5 artifact/ledger first, then the conditional H1-prefix-prior artifact/ledger when a scorer profile is consumed, then the current H6-Prefix artifact/certificate-set/ledger. `h7-linear-history-source-v1` activates the conditional input. Reject historical, development, reordered, missing-ledger, wrong-revision, or post-H7-produced references. Hash every consumed source/config/fixture/objective/adapter file into the H7 dependency closure. Do not copy or rerun predecessor payloads, and do not require H6-Prediction absent an explicit empirical checkpoint trial.

- [ ] **Step 6: Execute trials, oracle, controls, and status.** Capture both fixture byte sequences once, freeze snapshots before calculation, execute correct trials, independently execute controls, and construct immutable nested results. The expected outside-stabilizer trial passes as a control only when the held-fixed decoder causes a decisive emission/complete-objective change; it is not a positive covariance trial.

- [ ] **Step 7: Emit the complete `validation/h7.json`.** Include gate/status/obligations; exact source/dependency closure; references to the current H1--H5, active H1-prefix-prior, and H6-Prefix artifact/manifest/payload/certificate-set/ledger hashes; raw fixture hashes; separate scalar `GL+(1)` regression and primary matrix `GL+(2)` group/representation/base/product/nonclaim tags; every frame/action/determinant/norm; every original/transformed SPD diagnostic; every trial and decoder policy; exact scorer law/profile/prefix/alpha/covector/history identities plus all twelve scorer residuals; source/support identities; tensor/law/cocycle/closed-walk residuals; density probes and expected shifts; entropy shifts; every local term; monolithic/local/evidence/KL records and applicability; `r_abs/r_rel/r_back`; every budget input/component; GH41/51 values/deltas; every exact control ID/residual/limit/detection; and explicit H8/training/optimizer/det-negative/base-curvature/predictive nonclaims. The H7 run directory contains reference records only and no predecessor validation payload, certificate-set copy, ledger copy, or H6-Prediction record.

- [ ] **Step 8: Run focused GREEN.** Run the Step 4 command. Expected: the frozen gate is PASS; every correct invariant clears its own allowance; all twelve controls are decisive; every status mutant maps exactly.

- [ ] **Step 9: Have two bounded reviewers inspect existing evidence.** One checks gauge/type/decoder/Jacobian mathematics; one checks complete objective, oracle, controls, predecessor freshness, and status precedence. They inspect focused output and source only and do not rerun tests.

- [ ] **Step 10: Commit Task 7.**

  ```powershell
  git add verification/h7_gate.py vfe4/types/results.py tests/promotion/test_h7_gate.py docs/preregistrations/2026-07-21-h7-frame-covariance.md
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

**Interface:** The one editable verifier accepts the exact ordered tuple through H7 and selected operation `H7`. Before the final freeze, the runner exposes pure nonmutating `project_h1_h5_compatibility_config`, `project_h1_prefix_prior_config`, and `project_h6_prefix_config` functions plus `run_projected_current_candidate(config,junit_sha256,predecessor_refs) -> AtomicArtifactRef`; Task 9 calls each required projection exactly once without editing tracked CONFIG. The H7 operation validates those external predecessor references, captures H1/H7 bytes once, runs H7 only, and returns the existing `VerificationRunResult` explicit union containing one `H7GateResult` for this selected operation.

- [ ] **Step 1: Write failing config/artifact/integration tests.** Require exact H7 config/action/oracle/budget/predecessor literals and reject H7 without the full ordered tuple, reordered/duplicate gates, H8, changed group matrices, `phi`/exponential mode, weakened envelope, lower precision, changed GH orders, or altered control thresholds. Preserve every shorter prefix's existing behavior and prove it does not read H7 fixture bytes, import mpmath, run H7, or publish H7 keys.

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

- [ ] **Step 3: Extend conditional config resolution and runner orchestration.** Keep one editable dictionary. Reuse or add pure `project_h1_h5_compatibility_config(CONFIG)`, `project_h1_prefix_prior_config(CONFIG)`, and `project_h6_prefix_config(CONFIG,current_refs)` functions whose outputs omit H7 fields and reproduce the predecessor schemas without mutating CONFIG; the prefix-prior projection is callable only when the scorer profile requires it. H6-Prefix consumes the just-produced current H1--H5 reference, not H7 and not H6-Prediction. The H7 path validates raw predecessor reference bytes/hashes, captures H1/H7 fixture bytes once, and passes them into `evaluate_h7`; a gate may not reread them. Publish only after H7 returns. `_script_main` returns zero only for H7 PASS and prints `H7: pass` plus one artifact path.

- [ ] **Step 4: Extend provenance without weakening earlier artifacts.** Record exact Git revision, dirty-content/dependency closure, candidate JUnit SHA, config/objective schema, fixture expected/observed hashes, predecessor manifest/payload/certificate-set/ledger hashes, exact production order, scalar `GL+(1)` replay versus primary `GL+(2)` type tags, group/action/trial matrices, standard representations, scorer identity/law, decoder policies/stabilizer classification, oracle version/precision/orders, envelope/budget/exact control-ID constants, H7 status, and bounded nonclaims. Manifest hashing covers every reference and `validation/h7.json`.

- [ ] **Step 5: Update bounded documentation.** README and preregistration describe the implemented H7 surface and exact protocol only; before the milestone they do not prestate JUnit totals or measured residuals. State that only the direct matrix trials cover selected `GL+(2)` elements, while the scalar `GL+(1)` path is a complete-law regression; neither covers the det-negative component. Freeze the exact history-scorer law/control ID, fixed-decoder centered-softmax stabilizer restriction, entropy shifts, one-revision predecessor production order, and open optimizer/training/H6-Prediction/H8 claims.

- [ ] **Step 6: Run focused GREEN.** Run the Step 2 command. Expected: all compatibility prefixes remain isolated and one H7 click-run publishes exactly the files above with a valid manifest and no predecessor rerun/copy.

- [ ] **Step 7: Have one reviewer inspect the public/config/artifact boundary.** Check single-CONFIG/no-CLI behavior, pure predecessor projections, conditional scorer prerequisite, exact production/reference order, conditional fixture capture, reference-only H7 artifact, exact H7 payload, and no H8 or empirical widening.

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

- [ ] **Step 3: Produce and close the current H1--H5 compatibility artifact exactly once.** In memory, call `run_projected_current_candidate(config=project_h1_h5_compatibility_config(CONFIG),junit_sha256=junit_sha256,predecessor_refs={})`; do not edit CONFIG or tracked source. Independently validate the artifact manifest, exact `(git_head,dirty_digest)`, JUnit SHA, compatibility config/objective/update/fixture identities, five distinct ordered `validation/h1.json` through `validation/h5.json` payload hashes, and exactly five PASS states. Start, populate one claim per gate/identity check, and validate `.verification/h1-h5-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json` with current mechanical/reproduced evidence. Only after its Stop-hook closure removes the marker may Step 4 begin.

- [ ] **Step 4: Produce and close the conditional H1-prefix-prior artifact exactly once.** The condition is explicit: `h7-linear-history-source-v1` is consumed, so this preregistration requires the step. In memory, call `run_projected_current_candidate(config=project_h1_prefix_prior_config(CONFIG),junit_sha256=junit_sha256,predecessor_refs={})` once at the same revision/digest/JUnit SHA. Validate the exact prefix-prior config, generative-factor/scorer schema, raw fixture, manifest/payload, and PASS status; close `.verification/h1-prefix-prior-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json` in its own fresh verifier turn. If a future H7 protocol removes every prefix/history scorer, this entire artifact/reference is absent rather than replaced by a fake certificate.

- [ ] **Step 5: Produce and close the current H6-Prefix certificate set exactly once.** Build exact `current_h1_h5_refs` from the Step 3 H1--H5 artifact/ledger; H6-Prefix does not consume the H1-prefix-prior result as a safety premise. In memory, call `run_projected_current_candidate(config=project_h6_prefix_config(CONFIG,current_h1_h5_refs),junit_sha256=junit_sha256,predecessor_refs=current_h1_h5_refs)` once. Independently validate the same revision/digest/JUnit SHA, H1--H5 reference, H6 prefix/config/model-family/vocabulary identities, every required certificate key, the immutable certificate-set SHA, manifest, and PASS state. Close `.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json` in a fresh verifier turn. Do not produce finite-SMC, readiness, checkpoint, metric, or H6-Prediction evidence: none is consumed by frozen H7-v1.

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

  Expected: `H7: pass` and one artifact path. Independently recompute `manifest.sha256`; validate exact source/config/dependency/fixture/action/oracle identities; require the three reference records and their exact current hashes; require `validation/h7.json`, both scalar-regression plus all five primary matrix positives, the expected-negative trial, all twelve scorer residuals, and the twelve ordered control IDs. Require no copied predecessor validation/certificate/ledger, no H6-Prediction payload, and no H8 file. Atomically record the exact artifact path, artifact manifest SHA, fixture-set SHA, refs-registry SHA, revision/digest, and JUnit SHA in `.verification/h7-current-candidate-<FULL_HEAD>-result.json`; this result pointer is not part of the H7 artifact manifest and therefore introduces no hash cycle.

- [ ] **Step 7: Have fresh reviewers consume the immutable evidence set only.** One checks scalar-versus-matrix typing, group/cocycle/`B`/decoder/stabilizer mathematics, direct solves, and non-square type safety; one checks probability measures, exact history-scorer source-frame law, Jacobians/entropy, and local-monolithic/evidence-KL completeness; one checks the 100-decimal oracle, budgets, exact control inventory, status, and predecessor/ledger provenance. They cite focused outputs, JUnit XML, the four current artifacts, source, and closed predecessor ledgers. They do not rerun tests or gates. A Critical/Important source defect invalidates this candidate and triggers the complete replacement lifecycle; it is never repaired in place.

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

  Populate one claim per H7 check: current reference-only predecessors and their closed ledgers; raw fixture/dependency closure; separately typed scalar `GL+(1)` regression and primary direct `GL+(2)` domain/envelope/det-negative nonclaim; diagonal-base/internal-product distinction; pure tensor/autograd/no-materialized-inverse layer; `U`/receiver-source/cocycle/closed-walk order; correct and reverse-arrow `B`; all `(mu,Sigma,M,h,J)` laws; every generative/recognition transition law; exact source identity and twelve history-scorer residuals; decoder contragredience and centered-stabilizer scope; complete density/Jacobian/entropy/log-ratio laws; every local term; local/monolithic complete ELBO; scalar evidence/posterior KL; GH41/51 and 100-decimal oracle; `r_abs/r_rel/r_back`; all twelve exact controls; exact JUnit totals; atomic H7 artifact/manifest; and preservation of every predecessor ledger hash. Do not duplicate H1--H5 or H6-Prefix correctness claims in the H7 ledger; record their validated ledgers as provenance evidence.

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
- A source-mixture moment projection, diagonal within-block covariance substitution, jitter, clipping, pseudo-inverse, resampling, or numerical repair.
- A second training/objective implementation, changes to H2 detached snapshots, or use of the H7 monolithic diagnostic as a training loss.
- Research-vault ingestion. The implementation may offer the completed H7 result for later ingest, but no vault write belongs to this plan.

## Self-Review of Plan Completeness

- **Spec coverage:** Tasks 1--9 cover separately typed `GL+(1)` scalar complete-law regression and primary direct `GL+(2)` matrix evidence; standard `d_z=d_m=2` representations; `T=2,D=12,V=3`; frozen diagonal/internal matrices; identity/nonidentity frames; receiver/source link and cocycle order; correct versus reverse-arrow `B` plus non-square channel typing; all moment/information laws through direct operand solves with no materialized inverse; exact `StructuredLanguageRecognition | FactorizedLanguageRecognition` dispatch with projection/emission-only rejection; transition offsets/covariances/precisions/Jacobians; decoder contragredience/bias/centered stabilizer; the exact linear history scorer, twelve scorer residuals, and source-frame mutant; density/entropy/log-ratio shifts; local/monolithic/evidence-KL obligations; envelope; 100-decimal GH41/51 oracle; operand-local budgets; `r_abs/r_rel/r_back`; the twelve ordered control IDs; status precedence; reference-only predecessors; one JUnit; and revision-specific ledgers.
- **Task ordering:** Protocol and bytes freeze before any calculation; pure tensor laws precede factor pushforwards; generative and recognition completeness precede objective comparison; objective completeness precedes the independent oracle/gate; all tracked work and reviews finish before the final source freeze. The sole candidate JUnit then precedes exact-revision H1--H5, active H1-prefix-prior, H6-Prefix, and H7 artifact/ledger production in that order, with no later tracked edit.
- **Type consistency:** `H7ScalarReplayAction` and `H7GLPlus2Action` remain disjoint members of `H7TensorAction`; trial records carry matching dimension/group tags. Recognition consumes only the explicit two-class post-H6 union. The same snapshots, scorer records, predecessor-reference type, budget/residual/trial/control records, `H7GateResult`, direct action IDs, decoder policies, and fixture hashes flow through every task. H2 types remain unchanged and detached.
- **Decision completeness:** Every required positive invariant has an operand-local allowance; every GH-dependent invariant has a 41/51 convergence record; every negative control has an exact decisiveness rule; finite valid violations FAIL; missing/stale/out-of-envelope/unresolved evidence is INCONCLUSIVE; and PASS requires the complete inventory.
- **Nonclaim completeness:** The plan explicitly excludes treating the scalar replay as `GL+(2)` evidence, det-negative `GL(2)`, optimizer/training equivariance, H6-Prediction/predictive benefit, H8 scale, base curvature/holonomy, fixed-decoder full-group symmetry, entropy invariance, and Research-vault writes.
- **Placeholder scan:** No fixture coefficient, action matrix, trial, control, threshold, status rule, oracle precision/order, or commit boundary remains to be chosen from outcomes. Runtime hashes/revisions/artifact paths are measured identities rather than adjustable scientific parameters.
- **Path check:** This plan is saved at `docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md`. The authoring task makes no code, test, config, fixture, preregistration, ledger, or commit change outside this file.
