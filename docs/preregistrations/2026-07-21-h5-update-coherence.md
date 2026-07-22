# H5 Update-Coherence Preregistration

Date frozen: 2026-07-21
Implementation gate: H5
Fixture: `h5-conditional-update-v1`
Raw fixture SHA-256: `9dd42603419952a2ffa4b6602971240ec00572283557d672ae6ee106c31dd91c`

## Scope and claim

H5 is a deterministic implementation-verification gate. It tests whether each
labeled update evaluates the same complete H1 ELBO over every factor whose
canonical input changes, and whether acceptance or rollback follows the label's
declared contract. It is separate from the empirical H4 cost gate and cannot
compensate for H4. It makes no H4 cost, H6 prediction, H7 scaling/covariance, H8
readiness, or training claim.

The recognition law is

$$
Q_{\mathrm{H5}}=\prod_{t=0}^{2}q_t^z(z_t)q_t^m(m_t)
\prod_{t=1}^{2}\gamma_t(b_t)\beta_t(a_t\mid b_t).
$$

This is continuous mean-field recognition with conditional categorical
state-source rows. It is not fully factorized categorical recognition.

The H2 records remain detached immutable evaluation snapshots. Gradient
proposals use a separate differentiable working representation and are detached,
cloned, checked for finite values, and frozen before complete-objective
evaluation. Accepted state changes only by whole-snapshot replacement inside the
transaction controller. Rejected candidates never become live.

The same raw H1 and H5 byte objects are captured once and supplied to production
and oracle seams. H5 v1 uses CPU binary64, quadrature orders 21 and 17, and zero
stochastic contribution. Any source, fixture, test, configuration, plan, or
artifact-schema edit invalidates revision-bound evidence.

## Immutable identifiers

The factor universe, in evaluation order, is:

```text
initial_joint
model_source[1]
model_transition[1]
state_source[1]
state_transition[1]
emission[1]
model_source[2]
model_transition[2]
state_source[2]
state_transition[2]
emission[2]
recognition_entropy
```

The recognition-coordinate universe is:

```text
q[z0]
q[m0]
q[z1]
q[m1]
q[z2]
q[m2]
q[model_source_b1]
q[state_source_a1_b0]
q[model_source_b2]
q[source_row_a2]
q[state_source_a2_b1]
```

The mutable model-block universe is:

```text
theta[state_transition_2]
theta[emission_1]
theta[shared_decoder_transition]
```

The signed complete-ELBO terms are, in order,
`expected_log_emission[1]`, `expected_log_emission[2]`,
`initial_model_kl`, `initial_state_kl`, `model_source_kl[1]`,
`model_source_kl[2]`, `model_transition_kl[1]`,
`model_transition_kl[2]`, `state_source_kl[1]`,
`state_source_kl[2]`, `state_transition_kl[1]`, and
`state_transition_kl[2]`, with signs
`(+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)`.
`joint_recognition_entropy` is a diagnostic term and `complete_elbo` is derived
once from the 12 signed terms. `ElboTerms.allowances` is metadata, not a term.

## Exact fixture values

The full raw JSON is tracked at
`vfe4/validation/fixtures/h5_conditional_update_v1.json` as UTF-8 without a BOM,
with LF line endings and one final LF. Git attributes mark it `-text`.

- Identity: schema version `1`, family
  `continuous_mean_field_conditional_categorical`, H1 fixture `h1-v1`, H1 raw
  SHA-256 `388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b`,
  and factor-input schema `h5-factor-input-v1`.
- Continuous coordinates `(q[z0],q[m0],q[z1],q[m1],q[z2],q[m2])` have means
  `(-0.10,0.25,0.05,0.175,-0.04,0.14)` and variances
  `(0.65,0.78,0.96,1.21,0.90,1.40)`.
- Categorical coordinates are
  `q[model_source_b1]=(1)`,
  `q[state_source_a1_b0](.|b1=0)=(1)`,
  `q[model_source_b2]=(0.4,0.6)`,
  `q[source_row_a2](.|b2=0)=(0.75,0.25)`, and
  `q[state_source_a2_b1](.|b2=1)=(0.2,0.8)`, on the literal ordered supports
  recorded by the fixture. Every probability is finite, nonnegative,
  normalized, and strictly positive on its support.
- `theta[state_transition_2]` owns
  `(alpha_0,alpha_1,B_base,c,R)=(0.8,0.64,-0.35,0.08,0.48)`.
- `theta[emission_1]` owns `w_z=(0.2,-0.4,0.1)`,
  `w_m=(0.3,0.2,-0.5)`, and `bias=(0.05,-0.1,0.15)`.
- `theta[shared_decoder_transition]` owns `s=0.0`. Reconstruction uses
  `B_effective=B_base+s`, `emission[1].w_z[0]=w_z_base[0]+s`, and
  `emission[2].w_z[0]=-0.1+s`. No storage alias creates this sharing.
- The exact source-row record is coordinate `q[source_row_a2]`, time `2`,
  condition `("b2",0)`, support `(0,1)`, and initial probabilities
  `(0.75,0.25)`. No alias is accepted.

Equivalent H1 reconstruction uses initial mean `(mu_z0,mu_m0)` and diagonal
covariance `diag(V_z0,V_m0)`. At each time 1 and 2, all model-source slots have
zero slope and the same corresponding `mu_mt,V_mt`; all state-source slots have
zero z and m slopes and the same corresponding `mu_zt,V_zt`. The parser builds
every repeated slot from one H5 coordinate and checks literal equality of every
repeated offset and variance.

## Factor reconstruction and dependency graph

The reconstruction bindings are frozen as:

```text
initial_joint -> h1.initial_joint, q[z0], q[m0]
model_source[1] -> h1.model_source_priors[1], q[model_source_b1]
model_transition[1] -> h1.model_transition[1], q[m0], q[m1], q[model_source_b1]
state_source[1] -> h1.state_source_priors[1], q[model_source_b1], q[state_source_a1_b0]
state_transition[1] -> h1.state_transition[1], q[z0], q[z1], q[m1], q[model_source_b1], q[state_source_a1_b0]
emission[1] -> theta[emission_1], theta[shared_decoder_transition], q[z1], q[m1], h1.observation_label[t=1]
model_source[2] -> h1.model_source_priors[2], q[model_source_b2]
model_transition[2] -> h1.model_transition[2], q[m0], q[m1], q[m2], q[model_source_b2]
state_source[2] -> h1.state_source_priors[2], q[model_source_b2], q[source_row_a2], q[state_source_a2_b1]
state_transition[2] -> theta[state_transition_2], theta[shared_decoder_transition], q[z0], q[z1], q[z2], q[m2], q[model_source_b2], q[source_row_a2], q[state_source_a2_b1]
emission[2] -> h1.emission[2], theta[shared_decoder_transition], q[z2], q[m2], h1.observation_label[t=2]
recognition_entropy -> recognition_snapshot
```

The one shared group is
`("shared_decoder_transition","theta[shared_decoder_transition].s",
("state_transition[2].B:add","emission[1].w_z[0]:add",
"emission[2].w_z[0]:add"))`.

The complete dependency graph is:

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

Affected factors are the factor-universe-ordered union of the active rows.
Singleton categorical coordinates stay in the graph but public update resolution
rejects attempts to change them.

## Update taxonomy and exact rule contracts

`UpdateLabel` contains `exact_coordinate`, `valid_mm`, `generalized_em`,
`natural_gradient_proposal`, `sgd_proposal`, `adam_proposal`, and
`truncated_iteration`. The only H5 v1 producers are:

```text
exact_z0 -> exact_coordinate; variables=(q[z0]); parameters=(); damping=(1)
exact_source_row_a2 -> exact_coordinate; variables=(q[source_row_a2]); parameters=(); damping=(1)
exact_state_transition_2_m -> exact_coordinate; variables=(); parameters=(theta[state_transition_2]); damping=(1)
generalized_em_emission_1 -> generalized_em; variables=(); parameters=(theta[emission_1]); damping=(1,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625)
natural_gradient_z1 -> natural_gradient_proposal; variables=(q[z1]); parameters=(); damping=(64)
```

`valid_mm` deliberately has no H5 v1 producer. Configuration resolution must
reject an MM request unless a revision-bound proof artifact exists. MM absence is
not a gate obligation and does not enter attempt or status logic.

The initial optimizer state is `FrozenByteState("h5-no-optimizer-v1",
b'{"kind":"none"}')`. The initial RNG state is
`FrozenByteState("h5-deterministic-rng-v1",
b'{"algorithm":"none","counter":0}')`. Production never advances the RNG.

## Canonical encodings and hashes

All finite binary64 values encode as `float.hex()`, bytes as a length and hex
pair, tuples as arrays, enums as exact string values, and mappings as compact
sorted-key ASCII JSON with no trailing newline. This distinguishes signed zero.
Callers never supply trusted derived hashes.

The domains, each including a terminal NUL byte, are:

```text
vfe4.h5.update-spec.v1
vfe4.h5.update-request.v1
vfe4.h5.reference-state.v1
vfe4.h5.recognition-snapshot.v1
vfe4.h5.model-snapshot.v1
vfe4.h5.live-state.v1
vfe4.h5.candidate.v1
vfe4.h5.semantic-state.v1
vfe4.h5.attempt.v1
vfe4.h5.transaction.v1
vfe4.h5.factor-input-schema.v1
vfe4.h5.factor-input.v1
vfe4.h5.frozen-complement.v1
vfe4.h5.optimizer-state.v1
vfe4.h5.rng-state.v1
vfe4.h5.objective-schema.v1
vfe4.h5.validation-payload.v1
```

The parser hashes raw bytes before UTF-8 decoding and compares all 64 lowercase
hexadecimal digits. It rejects duplicate keys, reordered object fields,
unknown/missing fields, nonfinite values, wrong sequence types/order, aliases,
and schema drift. The canonical update-spec hash excludes raw bytes and all
derived hashes.

The implemented canonical update-spec SHA-256 is
`0e4e870dd725aeaec77ffd128ba85dbf619df5b0261b2178e6a115a8970715d6`.
The implemented factor-input-schema SHA-256 is
`2ae8a66776760dc5b3e1e73d4e41f1c0fdc137ed9a972fe31d4187adc5a94642`,
and the implemented objective-schema SHA-256 is
`b6af943b135b5acc01f9950502fb4a68554eab10005e44e43a7e7f213e4357ac`.

The factor-input schema core is `(factor_universe,
("schema_version","factor_id","normalized_factor","observation",
"recognition_inputs"), reconstruction_records)`. The objective-schema core adds
the factor-input schema hash, identifier/term universes and signs, dependency
graph, reconstruction/shared records, quadrature orders, the three operation
count tables, and formula tags `h5-term-budget-v1`,
`h5-complete-budget-v1`, `h5-delta-budget-v1`, and
`h5-candidate-comparison-v1`.

## Numerical budgets

Let `eps` be binary64 machine epsilon, `C=4096`, and
`gamma(n)=n*eps/(1-n*eps)`. Analytic operation counts are:

```text
initial_model_kl=192; initial_state_kl=192
model_source_kl[1]=32; model_source_kl[2]=64
model_transition_kl[1]=192; model_transition_kl[2]=320
state_source_kl[1]=32; state_source_kl[2]=96
state_transition_kl[1]=256; state_transition_kl[2]=448
joint_recognition_entropy=320
```

Analytic factor counts are `initial_joint=256`, `model_source[1]=32`,
`model_transition[1]=192`, `state_source[1]=32`,
`state_transition[1]=256`, `model_source[2]=64`,
`model_transition[2]=320`, `state_source[2]=96`,
`state_transition[2]=448`, and `recognition_entropy=320`.
Emission count is `32*r^2+8*r+32` for `r` exactly 21 or 17.

Exact-candidate counts are 512 for z0 mean/variance and each source-row
probability, and 4096 for each M-block field `alpha_0`, `alpha_1`, `B_base`, `c`,
and `R`.

For each term/order `r`,

$$\rho_r=4096\gamma_{N_r}\max(1,\kappa_{r,1},\ldots)
\max(1,|v_r|,S_r),$$
$$\rho_{21-17}=4096\gamma_3
\max(1,|v_{21}|,|v_{17}|,|v_{21}|+|v_{17}|),$$
$$A_{term}=|v_{21}-v_{17}|+\rho_{21}+\rho_{17}+\rho_{21-17}.$$

The complete allowance is

$$A_{complete}=\sum_{i=1}^{12}A_i+4096\gamma_{13}
\max(1,\sum_{i=1}^{12}|s_i v_i|).$$

For `delta=after-before`,

$$A_{sub}=4096\gamma_3\max(1,|before|,|after|,|delta|,
|before|+|after|),$$
$$\epsilon_\Delta=A_{before}+A_{after}+A_{sub}.$$

For independent exact-candidate scalars `(p_j,o_j)` with count `N_j`, each side
gets its own `4096*gamma(N_j)*max(1,kappa)*max(1,abs(value))`; comparison adds
`4096*gamma(3)*max(1,abs(p),abs(o),abs(p)+abs(o))`. Agreement requires
`abs(p_j-o_j)` no greater than their sum. z0/source-row conditions are 1; M
conditions are each implementation's recorded `G` condition number. No global
condition number, solver term, stochastic term, or unrelated scale is allowed.

## Positive cases, controls, and decisions

The five positives, in order, are exact Gaussian z0 E-coordinate, exact
categorical source row, exact Gaussian fixed-recognition M-coordinate, accepted
resolved GEM emission update, and rejected oversized natural-gradient proposal
with byte-identical rollback.

The seven test-only controls, in order, are: omit affected
`state_transition[2]`; omit affected `emission[1]`; force acceptance of an
unresolved nextafter-sized GEM delta; relabel the natural proposal as exact;
mutate rejected z1 and RNG counter during rollback; reflect `alpha_0` across its
fixed-complement least-squares optimum so changed input has exactly equal value;
and add exactly `1e-6` to both reported state-transition values while preserving
the same input bytes. Each control passes only by detecting its injected fault.

Acceptance is label-specific. Exact-coordinate candidates require the exact
producer and complete evidence. GEM accepts only a resolved positive increase,
`delta > epsilon_delta`. The oversized natural proposal is rejected when its
complete delta is not acceptable. Any accepted unresolved update, label
provenance mismatch, incomplete affected-factor evaluation, deterministic
reevaluation mismatch, or rollback-hash mismatch is a typed failure before live
state mutation.

H5 is `PASS` only when fixture/schema/graph identity is exact, all five positives
pass, all seven controls detect their intended faults, every numerical record is
finite, complete, and operand-shaped, and obligations are empty. It is `FAIL`
only when current finite complete evidence decisively falsifies a required
positive, dependency, decision, rollback, oracle, or control invariant. It is
`INCONCLUSIVE` when evidence is missing/nonfinite, a schema/cache cause is
unresolved, or an emission-touching comparison is inside or on its complete
allowance; each open phase is named in obligations.

## Evidence and nonclaims

The coupled H4/H5 milestone uses one exact tracked revision, one full JUnit run,
one click artifact, separate `validation/h4.json` and `validation/h5.json`
payloads, and a revision-specific coupled claim ledger. Prior ledgers are
preserved byte-for-byte. Focused implementer runs are noncumulative; reviewers
inspect retained output rather than rerunning tests or timings.

This preregistration draws read-only context from [[Variational EM]],
[[Natural gradient]], [[neal-1998-variational-em]], and
[[dempster-1977-em-algorithm]]. Those sources support the same-objective and
finite-step-label distinctions; they are not executable H5 closure evidence.
