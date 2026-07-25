# H7 Internal Population-Frame Covariance Preregistration

Protocol revision: `h7-frame-covariance-v1`

This preregistration freezes a forward change-of-variables check for the
complete normalized VFE 4.0 law. It does not preregister optimizer or
gradient-flow equivariance, a training benefit, a predictive benefit, base
curvature or holonomy, H8 scaling, or the orientation-reversing component of
`GL(2)`.

## Frozen domain and fixtures

- Primary domain: direct `GL+(2,R)` elements, standard state and model
  representations, `T=2`, `d_z=d_m=2`, `D=12`, and `V=3`.
- Scalar regression: the unchanged raw `h1-v1` fixture under separately typed
  `GL+(1,R)` actions. It is not evidence for `GL+(2,R)`.
- Matrix fixture: the exact raw bytes of
  `vfe4/validation/fixtures/h7_v1.json`.
- Matrix density probes: the exact raw bytes of
  `vfe4/validation/fixtures/h7_density_probes_v1.json`; ordinary parsing
  consumes this table, and its probe-loading path never runs a Cholesky
  factorization, whitening, or transformed-anchor reconstruction.
- Required recognition origins, in order:
  `structured_full_block`, `factorized_diagonal_within_fiber`.
- A generic non-diagonal primary action promotes the factorized origin to
  `unrestricted_full_block_pushforward`; it is never projected back to a
  diagonal or same-family law.
- The initial continuous contribution is the joint
  `K0_joint_z0_m0`, never a sum of marginal KL terms.

The exact raw H1/H7 fixture hashes and the deterministic density-probe-set
hash are source constants in `vfe4.validation.h7_fixture` and frozen below
after the two raw fixture files exist:

- `h1_fixture_raw_sha256`: `388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b`
- `h7_fixture_raw_sha256`: `d2ed126c3deab3eafc7b94f81f13152be63eb854e3e62e03f1494dea163666d4`
- `density_probe_table_raw_sha256`: `4857af296e84a33f47964c3bca65e0d42967009aa5c79a52bcc98d6db04382c6`
- `density_probe_set_sha256`: `f002618a32270846c83fedf9888bc06a01d755019edc6421526aee33f89fb42f`

The one-time preregistration command was
`python _freeze_h7_probe_table.py`. It completed in 2.1 seconds, emitted
exactly 486 records and 405264 bytes, printed both hashes above, and the
temporary generator was then removed. This command is a historical freeze
record, not a runtime or test entry point.

## Frame, factor, source, and density laws

Frames and links transform as
`U_t'=g_t U_t` and `Omega_tj'=g_t Omega_tj g_j^-1`, with products acting
rightmost first. The exact relation checks are
`Omega_21 Omega_10=Omega_20` and
`Omega_02 Omega_21 Omega_10=I`; the latter is a relation walk, not a
causal-DAG cycle or a base-holonomy claim.

The standard Gaussian laws are
`mu'=G mu`, `Sigma'=G Sigma G^T`, `M'=G M G^T`,
`h'=G^-T h`, and `J'=G^-T J G^-1`.
Receiver/source maps use `G_receiver A G_source^-1`.
The same-receiver morphism uses the general law
`G_z,t B_t G_m,t^-1`.
Decoder maps use `W_z,t'=W_z,t g_t^-1`,
`W_m,t'=W_m,t g_t^-1`, and unchanged bias.

The frozen source scorer is
`alpha_b,t,j(prefix) + r_z^T z_j + r_m^T m_j`, with
`alpha=a+b*sum_{ell=1}^t ell*(x_{ell-1}+1)`.
Prefix bytes and alpha do not transform. Histories transform in source frame;
both covectors use the matching source inverse transpose. Masks, ordered
support, normalized probabilities, and raw scores must remain invariant.

For standard representations,
`logJ_G=2*sum_t log(det(g_t))`.
Complete generative and recognition coordinate log densities each shift by
`-logJ_G`; their pointwise log ratio is invariant and continuous recognition
entropy shifts by `+logJ_G`. Initial and receiver-conditional shifts are
reported separately. Entropy is not called invariant.

The metadata inventory and one-time construction law remain serialized in
`h7_v1.json`. The resulting `x`, `x_prime`, shift metadata, per-row canonical
hash, and exact row order are serialized in
`h7_density_probes_v1.json`. Each pair was frozen from one original anchor
under `x=anchor+L@(0.25*direction)` and `x_prime=G_component@x`.
Ordinary parsing reads the exact pair values and checks the raw-table hash,
all 486 typed row hashes, their closed action/component/direction order, and
the domain-separated tuple hash. Evaluation may not select, rewhiten,
re-anchor, regenerate, or numerically reconstruct probes.

The scalar H1 adapter separately preserves the typed generative prior rows,
typed recognition probability tables, one-based observation labels `(1,2)`,
decoder indices `(0,1)`, and the exact ordered paths
`h1-path-0:a0-b0`, `h1-path-1:a1-b0`, `h1-path-2:a0-b1`,
`h1-path-3:a1-b1`. Its second-time recognition state-kernel selector is
`b=(0,0,1,1)` in the frozen `(a,b)` row order, not the varying `a` selector.
For each scalar trial and each ordered path, the adapter owns one frozen
global-density pair anchored at that path's original generative conditional
mean in `(z0,m0,z1,m1,z2,m2)` order. The eight pairs are trial-major, bind
the raw H1 hash, path ID, action hash, original-anchor profile, transformed
point, and global log-Jacobian shift in one self-hashing scalar probe set.

## Required trials

| Trial ID | Role | Frame/action/decoder | Expected predicate |
|---|---|---|---|
| `scalar-base-transformed` | `scalar_regression` | H1 / scalar base / transform | `complete_covariance` |
| `scalar-internal-transformed` | `scalar_regression` | H1 / scalar product / transform | `complete_covariance` |
| `matrix-identity-base-transformed` | `positive_covariance` | identity / diagonal base / transform | `complete_covariance` |
| `matrix-identity-internal-transformed` | `positive_covariance` | identity / internal product / transform | `complete_covariance` |
| `matrix-nonidentity-base-transformed` | `positive_covariance` | nonidentity / diagonal base / transform | `complete_covariance` |
| `matrix-nonidentity-internal-transformed` | `positive_covariance` | nonidentity / internal product / transform | `complete_covariance` |
| `matrix-fixed-decoder-centered-stabilizer` | `positive_covariance` | nonidentity / stabilizer / fixed | `centered_decoder_stabilizer_invariance` |
| `matrix-fixed-decoder-outside-stabilizer` | `expected_negative` | nonidentity / diagonal base / fixed | `decisive_outside_stabilizer_change` |

The centered-softmax stabilizer is exactly
`C_V W g^-1=C_V W`, with `C_V=I-(1/V)11^T`. Raw logits may differ by a
row-common scalar inside this stabilizer. The outside-stabilizer trial must
change the emission and complete objective decisively and never counts as a
positive covariance trial.

## Required negative controls

The exact ordered inventory is:

1. `wrong_covariance_congruence`
2. `wrong_precision_congruence`
3. `history_scorer_wrong_source_inverse`
4. `reversed_link_order`
5. `reverse_arrow_B`
6. `wrong_decoder_dual_action`
7. `fixed_decoder_outside_stabilizer`
8. `omitted_density_jacobian`
9. `reversed_logdet_sign`
10. `entropy_false_invariance`
11. `changed_h1_source_probability`
12. `diagonal_for_internal_action`

Source-support preservation is an additional exact invariant. Each control is
injected into a fresh transformed copy and must be decisive under its own
operand-local allowance.

## Numerical envelope and budgets

- Production: CPU float64.
- Oracle: mpmath at exactly 100 decimal digits.
- Exact-source emission checks: Gauss--Hermite orders 41 and 51.
- Inclusive action envelope:
  `||g_t||_2 <= 2` and `||g_t^-1||_2 <= 2`.
- Inclusive SPD envelope: every required original/transformed operand has
  `kappa_2 <= 1e3`.
- `EPS64=2**-52`, `ROUNDING_CONSTANT=4096`,
  `MAX_ORACLE_RELATIVE_DELTA=1e-18`,
  `CONTROL_MINIMUM_RELATIVE_RESIDUAL=1e-8`, and
  `CONTROL_ALLOWANCE_MULTIPLE=100`.
- `gamma_n(n)=(n*EPS64)/(1-n*EPS64)`.
- `rounding_allowance=ROUNDING_CONSTANT*gamma_n(operation_count)
  *condition_product*scale`.
- `reference_rounding_allowance=64*EPS64*max(1,abs(reference_value))`.
- `control_decisiveness_limit=max(100*correct_allowance,1e-8*scale)`.

Every invariant owns its exact typed operands, shapes, value hashes, scales,
condition numbers, normalization, operation count, optional quadrature
contribution, optional oracle value, and allowance contributions. No pooled
condition maximum, run-wide scale, or aggregate-only budget is admissible.
Backward recovery directly inverse-transforms all 218 matrix-trial inventory
operands. Each record binds its category/shape-specific forward and inverse
budgets before the complete record tuple is frozen and reduced to `r_back_max`.
The Task-5 comparison seam requires the exact frozen `H7TrialSpec`, binds its
trial, fixture, frame profile, decoder policy, and action SHA-256 to the
selected oracle trial, and retains comparisons for complete local and
monolithic ELBOs, joint initial and local terms, all three density diagnostics,
the entropy shift, scalar evidence/posterior KL when applicable, and the fixed
matrix evidence-not-applicable status.

### Task-6 source-only calibration status

Task 6 was authored under an explicit source-only restriction. No fixture was
parsed, no mpmath or project module was imported or executed, and no
quadrature, probe, tensor, or numerical calculation was performed during this
authoring pass. Consequently, the earlier frozen source constants above are
not represented as fresh Task-6 empirical measurements.

- Independent raw-byte fixture hash measurements: `UNMEASURED`.
- Independent raw scalar-probe table/set hashes: `UNMEASURED`; no scalar
  probe is regenerated or substituted from H1 moments. A supplied serialized
  anchor must nevertheless equal the independently assembled direct H1
  generative conditional global mean before its row/hash can be accepted.
- Independent serialized precision-operand table/set hashes: `UNMEASURED`.
  Precision and information-form inventory fields require the approved,
  source-bound Task-5 precision rows; the oracle does not synthesize an
  identity RHS or algebraically cancel solves to materialize an inverse.
- Required original/transformed SPD condition extrema: `UNMEASURED`.
- Required GH41/GH51 emission deltas: `UNMEASURED`.
- Final independent oracle inventory hash: `UNMEASURED`.
- Task-6 calibration closure: `INCONCLUSIVE`.

No empirical threshold, envelope, fixture byte, action, trial, or control was
changed or inferred from an unexecuted calculation. The source-only
implementation may proceed to mechanical syntax and lint checks, but Task-6
numerical closure requires a later authorized exact-revision run that records
these measurements without tuning the frozen protocol. Until those identities
are frozen, the oracle validates supplied raw schemas/probe inventories and
returns `INCONCLUSIVE`; it never promotes its observed hashes to expected
hashes.

### Task-7 source-only gate status

Task 7 was authored under the same explicit source-only restriction. The sole
`H7GateResult` now lives in `vfe4/types/results.py` and is re-exported once;
`verification/h7_gate.py` owns exact inventory checks, current-candidate
predecessor and validated-ledger checks, the closed source-dependency hash,
status precedence, and canonical `validation/h7.json` assembly. The compact
promotion contract does not execute a fixture, oracle, tensor operation,
control, import, or numerical calculation during this authoring pass.

The public assembly boundary captures its source dependency closure and
validates the ordered predecessor registry itself; callers cannot supply a
prevalidated closure or registry token. Predecessor ledgers are checked by the
installed deterministic verification-ledger validator and must bind the exact
live artifact revision captured by that validator. The validator source hash
is itself part of the H7 dependency closure. Expected-negative status is
derived only from the owned outside-stabilizer trial: decisive change is
success, complete covariance acceptance is `FAIL`, and any partial or
nondecisive change is `INCONCLUSIVE`. No caller-provided boolean can select
among those outcomes.

No H7 PASS or FAIL is claimed. Task-7 runtime closure is `INCONCLUSIVE` with
the following open obligations:

1. freeze the raw and set hashes for the scalar density-probe inventory;
2. freeze the raw and set hashes for the serialized precision operands;
3. measure every required original/transformed SPD condition extreme;
4. resolve and record every GH41/GH51 comparison;
5. freeze the independent oracle inventory hash;
6. validate the exact current H1--H5, H1-Prefix-Prior, and independently
   predecessor-free H6-Prefix artifact/ledger references;
7. execute the exact eight trials and twelve controls at one frozen candidate
   revision and capture current machine-readable evidence.

Any one of these open obligations dominates a finite candidate violation under
the preregistered status order. A later authorized run may close them only at
the exact recorded revision and dirty-content digest; it may not tune a
threshold, action, fixture, trial, control, or inventory from the results.

### Task-8 source-only publication status

Task 8 adds H7 as a separate selected operation without changing the legacy
H1--H5 prefix runner. The launcher contains one editable `CONFIG`, one `main`,
one script guard, no CLI, and no H8 operation. Its pure
`project_h1_h5_compatibility_config(CONFIG)` returns the unchanged H1--H5
scientific prefix without mutating the launcher mapping.

The H7 click path derives exactly
`.verification/h7-current-candidate-<FULL_HEAD>-refs.json`. The registry is one
canonical JSON mapping in the exact order `h1_h5`, `h1_prefix_prior`,
`h6_prefix`; each value is the complete `H7PredecessorReference`. Each
H6-produced `CandidateArtifactReference` is adapted losslessly by adding only
the candidate JUnit and validated-ledger path/hash, then canonicalized and
round-tripped before use. H7 validates those sibling references but does not
run H1--H5, H1-Prefix-Prior, or H6-Prefix and does not copy any predecessor
validation, certificate, or ledger.

The current-candidate H1--H5 producer supplies the candidate XML digest through
`run_verification(..., candidate_junit_sha256=<lowercase SHA-256>)`; omitting
that keyword preserves the legacy provenance schema exactly. At the H7
boundary, the raw H1 and H7 fixture files are each read once. Those captured
bytes are then reused for observed fixture identities, the source-dependency
closure, and gate assembly; those consumers do not reopen either fixture.

One H7 publication contains only `config.json`, `provenance.json`,
`environment.json`, `references/h1_h5.json`, the conditionally required
`references/h1_prefix_prior.json`, `references/h6_prefix.json`,
`validation/h7.json`, and `manifest.sha256`. The active frozen scorer profile
requires the H1-Prefix-Prior reference, so all three reference files are
present for H7-v1. Provenance binds the revision, dirty-content digest,
candidate JUnit, raw fixture identities, dependency closure, registry and
predecessor hashes, action bytes/hashes, trial roles, recognition promotion,
scorer/probe/joint-`K0` identities, oracle settings, operand-local budget
constants, controls, status, and nonclaims.

This authoring pass executed no tests, fixtures, imports, oracle, tensor
calculation, or gate. Task-8 runtime status therefore remains `INCONCLUSIVE`.
There are no H7 measured test totals, residuals, or PASS claim. The scalar
`GL+(1,R)` path remains only a complete-law regression; only the selected
direct matrix trials can support the bounded `GL+(2,R)` claim, and neither
covers `det(g)<0`. Optimizer/training equivariance, H6-Prediction, predictive
benefit, and H8 scale remain open.

## Status precedence and predecessors

`INCONCLUSIVE` dominates for a missing/stale predecessor, missing required
trial, envelope violation, nonfinite required result, unresolved GH41/GH51
comparison, or nondecisive/missing required negative control. Otherwise a
finite in-envelope failed covariance invariant or false acceptance of the
outside-stabilizer trial is `FAIL`. `PASS` requires both scalar regressions,
all five positive matrix trials, the sole expected-negative predicate, both
recognition origins, the complete local and monolithic objectives, all
corresponding density/source checks, and all twelve controls.

The exact predecessor registry order is `h1_h5`, `h1_prefix_prior`,
`h6_prefix`. Each reference binds the same final H7 candidate revision,
dirty digest, JUnit hash, artifact manifest, payload or certificate hashes,
and validated revision-specific ledger. H7 references these immutable
identities; it neither copies nor reruns predecessor payloads.

The accepted verification prefix is exactly
`("H1","H2","H3","H4","H5","H6-Prefix","H7")`. H6-Prediction is not a
prerequisite for this frozen mathematical fixture because no empirical
checkpoint is selected.
