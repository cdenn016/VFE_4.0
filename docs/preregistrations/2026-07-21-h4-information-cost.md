# H4 information-cost protocol preregistration

## Scope and nonclaims

H4 is a bounded CPU/float64, one-intra-op-thread empirical comparison of two
independent Gaussian implementations. It makes no H5 update-coherence claim,
no H6 predictive claim, no H7 covariance/gauge claim, no H8 sparse-scaling
claim, and no training, GPU, lower-precision, multi-thread, energy, or
cross-machine claim. Coefficients, seeds, orders, repetition counts,
conditioning envelope, budgets, caps, bootstrap settings, and thresholds were
fixed before collecting H4 measurements.

The primary endpoint is the coupled, `T=31`, `D=256` family. The two arms use
the same immutable, fully materialized neutral problem and exact factor
schedule; neither arm receives a random generator or regenerates a problem.
The information arm works in information coordinates. The moment arm performs
affine-Gaussian propagation and conditioning directly and must not call the
information arm, its canonical assembler, or invert the information arm's
precision. H3 coupled and independently authored zero-control fixtures remain
the `T=1`, `d_z=d_m=1`, `D=4` correctness anchors.

## Frozen suite and generation

```python
H4_PROBLEM_SEEDS = (
    104729, 130363, 155921, 181081, 206369,
    231779, 257053, 282407, 307831, 333271,
    358747, 384253, 409891, 435437, 461009,
    486587, 512161, 537793, 563359, 588937,
)
H4_PRIMARY_TIMED_BALANCE = (
    (104729, 5, 6), (130363, 6, 5), (155921, 5, 6),
    (181081, 6, 5), (206369, 5, 6), (231779, 6, 5),
    (257053, 5, 6), (282407, 6, 5), (307831, 5, 6),
    (333271, 6, 5), (358747, 5, 6), (384253, 6, 5),
    (409891, 5, 6), (435437, 6, 5), (461009, 5, 6),
    (486587, 6, 5), (512161, 5, 6), (537793, 6, 5),
    (563359, 5, 6), (588937, 6, 5),
)
H4_PRIMARY_TIMED_AB_TOTAL = 110
H4_PRIMARY_TIMED_BA_TOTAL = 110
```

Traversal is zero-based, first `horizon_index, horizon` in
`enumerate((7, 15, 31))`, then `seed_index, seed`, then `kind_index, kind` in
`enumerate(("coupled", "zero_control"))`; `problem_index` follows exactly
that traversal. With `d_z=d_m=4`, population-major storage is
`[z_0,m_0,z_1,m_1,...,z_T,m_T]`, and dimensions are `(T+1)*8`, namely
`64, 128, 256`. The initial `[z_0,m_0] ~ N(0,I_8)` is fixed and consumes no
random draw. The sole canonical schedule is `initial_joint`, followed at each
ascending `t=1..T` by `m_t|m_{t-1}`, `z_t|z_{t-1},m_t`, and the local
observation. Derived partitions, if exposed, are validated views only.

For every problem, the only generator is
`numpy.random.Generator(numpy.random.PCG64(seed))`. Per ascending time step it
draws, exactly in this order: `A_m`, `A_z`, `B` as three
`standard_normal((4,4))`; `c_m`, `c_z` as two `uniform(-.25,.25,4)`;
`R_m`, `R_z` as two `uniform(.5,1.5,4)`; `G` as
`standard_normal((8,8))`; observation offset as `uniform(-.25,.25,8)`;
observation noise as `uniform(.75,1.25,8)`; and target as
`uniform(-1,1,8)`. Let
`spectral_clip(M)=M*min(1,.65/||M||_2)`. Apply it separately to `A_m` and to
the concatenated `[A_z B]`, then split the latter. Set
`H=I_8+.05*G/max(1,||G||_2)`. The factors are

```text
m_t ~ N(A_m m_(t-1)+c_m, diag(R_m))
z_t ~ N(A_z z_(t-1)+B m_t+c_z, diag(R_z))
y_t ~ N(H[z_t,m_t]+offset, diag(observation_noise))
```

at the drawn target. The zero control uses the same raw draws and provenance
but zeros all and only active `A_m`, `A_z`, and `B`; offsets, noises, `H`,
targets, IDs, order, seed, shape, and every other generated value remain
identical. Canonical UTF-8 JSON uses sorted keys, compact separators, finite
row-major numbers, and serializes all generated values and raw-draw provenance;
its SHA-256 is recorded with seed, kind, and shape.

## Objective and comparison records

For every normalized factor with residual `A_r y-b_r`, covariance `R_r`, and
dimension `d_r`,

```text
log f_r(y) = -.5(A_r y-b_r)^T R_r^-1(A_r y-b_r)
             -d_r/2 log(2*pi) -.5 logdet(R_r)
sum log f_r(y) = -.5 y^T J y + h^T y + c
complete_objective = log Z
                   = c + .5 h^T J^-1 h -.5 logdet(J) + D/2 log(2*pi).
```

`J` must be SPD and higher is better. This is the unrestricted Gaussian
optimum/evidence, not negative VFE or a second ELBO. Both arms independently
produce this scalar including constants. `J`, `h`, and exact-name selected
moments are comparison records, not alternative objectives. Selected moments
are immutable tuples in the exact order
`("initial", "terminal", "observation[1]", ..., "observation[T]")`; initial
and terminal are the full `[z_t,m_t]` blocks, each observation is its full
local block, and overlapping blocks are retained rather than deduplicated.

The H3 adapter reads only public structural groups `initial_factors`,
`transition_factors`, and `observation_factors`, in their declared order. It
preserves rows, targets, variances, normalizers, IDs, and H3 coordinate order,
does not infer roles from names/IDs, and does not synthesize a state-space
factorization. Each raw H3 fixture must reproduce its H3 canonical `(J,h,logZ)`
under H3 allowances.

## Execution, timing, and eligibility

After H1--H3 and before preflight, capture `torch.get_num_threads()`, set it
to one, and verify one observed intra-op thread. Do not alter inter-op threads.
In `finally`, attempt restoration and record prior, effective, restored, and
restoration-error values. Set/verify failure suppresses timed records and is
`INCONCLUSIVE`; restoration failure prevents `PASS`. CPU/float64, affinity,
clock, processor/OS, thread environment, BLAS configuration, and relevant
power-policy facts are recorded; missing mandatory facts are `INCONCLUSIVE`.

For each problem run three untimed warmup pairs (`pair_index=0,1,2`), followed
by eleven timed pairs with `pair_index=3+repetition_index`. AB means
information then moment exactly when
`(horizon_index+seed_index+kind_index+pair_index)%2 == 0`; otherwise BA.
Warmups verify execution only and do not enter timed or inferential balance.
For the primary coupled `T=31` endpoint, even seed indices have 5 AB/6 BA,
odd indices 6 AB/5 BA: ten of each, 110/110 total.

The timer starts immediately before fresh native-arm construction and ends
after one complete schedule, native finite/SPD checks, and native objective
evaluation. It includes native factor assembly/propagation, solves, and
factorizations; it excludes generation, hashing, exact-oracle and condition
work, conversions to comparison records, moment extraction, GC setup,
serialization, bootstrap statistics, and memory/count diagnostics. Raw
nanosecond times retain independent problem, horizon, seed, kind, repetition,
pair, and order fields. Untimed operation and peak-memory passes use symmetric
real-operation instrumentation and are secondary only.

The scaled envelope is inclusive: `lambda_min(J)>=1e-6`,
`lambda_max(J)<=1e6`, `kappa_2(J)<=1e8`, Cholesky pivot `>=1e-3`,
`||mu||_inf<=16`, and every moment innovation covariance satisfies its local
eigenvalue/condition bounds. No jitter, clipping, pseudoinverse, repair, or
silent omission is allowed. Equivalence covers exact-posterior gap, terminal
`h`, `J`, selected means/covariances, and complete objective, each with its
own operand-shaped allowance. `H4_SOLVER_RELATIVE_BUDGET=1e-9`; each
solver-produced operand contributes it once, oracle operands contribute none;
`invariant_scale=max(1, all compared scalar absolute values or infinity
norms)`. Record rounding, solver, final, and ratio allowances. Ratio must be
strictly below `1e-4`; equality is `INCONCLUSIVE`.

## Inference, status, and artifacts

For each primary seed, use
`median(11 information times)/median(11 moment times)`. The aggregate is
`exp(mean(log(seed_ratio)))`. The 95% paired percentile interval resamples the
20 seed log ratios with replacement for exactly 100,000 replicates using
PCG64/bootstrap seed `20260721`, then exponentiates the 2.5th and 97.5th
percentiles. Seeds, never repetitions, are inferential units.

Status precedence is: any protocol/environment/thread/fixture/condition/table
incompleteness, nonfinite result, or allowance ambiguity is `INCONCLUSIVE`;
otherwise a finite decisive H3-anchor or terminal-law mismatch is `FAIL`;
otherwise `PASS` only if the upper interval endpoint is `<=.80`, `FAIL` only
if the lower endpoint is `>=.80`, and an interval crossing `.80` or exactly
`[.80,.80]` is `INCONCLUSIVE` with a precision obligation. Operations and
memory neither rescue nor overturn the primary status.

JSON records schema/version, canonical hashes and factor provenance, environment
and thread restoration, complete indexed timing rows, arm outcomes, all
operand-shaped equivalence/allowance records, selected moments, condition
records, seed statistics/bootstrap interval, status/obligations, and separate
secondary operation/memory diagnostics. H4 measurements are not used to amend
this preregistration.

## Frozen record, provenance, and early-failure schema

Every scaled problem has `source_kind="scaled_pcg64"`, a positive PCG64 seed,
coordinates `z[t,i]` followed by `m[t,i]`, problem ID
`h4-{kind}-T{horizon}-dz4-dm4-seed{seed}-v1`, and factors
`initial_joint`, then `m_transition[t]`, `z_transition[t]`, and
`observation[t]`. The H3 adapter has `source_kind="h3_anchor"`, `seed=0`,
ID `h4-anchor-{fixture_id}`, H3's exact coordinates/IDs, and no raw draws.
The source is explicit, never inferred from an ID.

Raw draws are immutable `(draw_index,name,shape,values)` records with
row-major flattened values. At each `t`, names and zero-based indices are
`A_m[t]`, `A_z[t]`, `B[t]`, `c_m[t]`, `c_z[t]`, `R_m[t]`, `R_z[t]`, `G[t]`,
`observation_offset[t]`, `observation_noise[t]`, `observed_target[t]`, using
`11*(t-1)+local_index`. Factor matrices are exactly `d x D` residual matrices,
targets have length `d`, and covariance is SPD `d x d`; no normalizer is
stored. Metadata is ordered, disjoint, in range, and has the frozen causal
support: initial normalizes `[z0,m0]`; `m_transition[t]` normalizes `m_t` and
parents `m_(t-1)`; `z_transition[t]` normalizes `z_t` and parents
`z_(t-1),m_t`; observations have no normalized coordinates and parent
`z_t,m_t`. Initial/transition normalized columns are identity and no other
columns are supported. The observation residual target is
`observed_target-offset`; both raw inputs remain provenance.

Only a matched scaled coupled/control pair may differ in kind, problem ID,
core digest, and designated transition-parent columns. Every designated parent
column is zeroed in control, including a column already numerically zero. This
rule never applies to H3 anchors. The generic canonical assembly is
`J=sum(A^T R^-1 A)`, `h=sum(A^T R^-1 b)`, and
`c=-.5 sum(b^T R^-1 b+d*log(2*pi)+logdet(R))`.

Canonical `core` is the ordered public problem content excluding the digest.
Its digest is `SHA256(b"vfe4.h4.neutral-problem.v1\x00" + compact UTF-8
sorted-key finite JSON(core))`. Published bytes are the compact sorted-key
envelope `{"schema_version":"h4-neutral-problem-v1","canonical_sha256":
digest,"problem":core}`. The exact schema literal is validated before the
domain-separated core digest; the embedded digest is never a hash of the full
envelope.

The ordered invariant, measurement, and allowance key universes are frozen in
the public H4 types. A completed-timing PASS or FAIL has all eight measurement
values finite. The sole decisive finite pre-timing H3-anchor FAIL has exactly
`primary_seed_ratio_geometric_mean`, `primary_bootstrap_lower`,
`primary_bootstrap_upper`, `primary_timed_ab_total`, and
`primary_timed_ba_total` unavailable; threshold remains finite `0.80`, and
the stopping residual and allowance-scale fraction come from the anchor. Later
invariants are exactly `(None,None,False,
"not_evaluated_after_decisive_h3_anchor_failure")`. The H3 allowance remains
applicable/numerical; the other allowance records are exactly
`{"applicable":false,"reason":"not_evaluated_after_decisive_h3_anchor_failure"}`.
For inconclusive results, `None` is permitted only when its producing phase did
not complete and the associated invariant/detail plus an obligation name that
phase; no finite unavailable value is fabricated.

The exact measurement keys are `primary_seed_ratio_geometric_mean`,
`primary_bootstrap_lower`, `primary_bootstrap_upper`, `primary_effect_threshold`,
`primary_timed_ab_total`, `primary_timed_ba_total`,
`maximum_solver_stopping_residual`, and `maximum_allowance_scale_fraction`.
The exact allowance keys are `h3_anchor_identity`, `exact_posterior_gap_equivalence`,
`terminal_h_equivalence`, `terminal_J_equivalence`,
`selected_moment_equivalence`, and `complete_objective_equivalence`. An applicable
allowance has exactly `applicable`, `dimension`, `operands`, `absolute_summands`,
`condition_numbers`, `operation_counts`, `solver_contribution`, `invariant_scale`,
`final_allowance`, and `allowance_scale_ratio`; an inapplicable record is exactly
the two-key sentinel. Missing primary measurements map respectively to
`primary_seed_level_inference`, `primary_effect_threshold`,
`primary_timed_order_balance`, `shared_protocol_identity`, or
`all_equivalence_allowances_decisive` and use the frozen inconclusive detail.
The coupled H3 anchor compares its frozen evidence; zero H3 has no frozen
reference evidence and instead requires independent adapter/oracle `c/logZ` agreement.

The exact ordered invariant names are: `h3_anchor_identity`,
`fixed_seed_problem_identity`, `coupled_zero_control_contract`,
`cpu_float64_one_thread`, `shared_protocol_identity`,
`scaled_condition_envelope`, `complete_repetition_table`,
`primary_timed_order_balance`, `exact_posterior_gap_equivalence`,
`terminal_h_equivalence`, `terminal_J_equivalence`,
`selected_moment_equivalence`, `complete_objective_equivalence`,
`all_equivalence_allowances_decisive`, `real_operation_instrumentation`,
`primary_seed_level_inference`, and `primary_effect_threshold`.
The only sentinel reasons are
`not_evaluated_after_decisive_h3_anchor_failure` and
`not_evaluated_after_inconclusive_eligibility`. For an INCONCLUSIVE result,
every inapplicable allowance must use the latter reason exactly. After timing,
FAIL requires invariants 0--7 and the three eligibility invariants
`all_equivalence_allowances_decisive`, `real_operation_instrumentation`, and
`primary_seed_level_inference` to pass, plus either a finite decisive
equivalence miss or a nondegenerate bootstrap interval whose lower endpoint is
at least the threshold. A crossing interval, `[threshold,threshold]`, or an
eligibility failure remains INCONCLUSIVE.
