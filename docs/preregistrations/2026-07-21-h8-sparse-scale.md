# H8 Sparse-Scale Systems Preregistration

Protocol revision: `h8-sparse-scale-v4` (amended 2026-07-26)

This v4 contract supersedes the underspecified staged v3 contract before any
H8 scientific execution.

Status at freeze: protocol only. No H8 correctness grid, production child,
profiler child, negative control, timing measurement, memory measurement, or
promotion decision has been executed. This document therefore contains no
measured endpoint and does not prestate PASS.

H8 is a synthetic empirical systems gate for one block-tridiagonal Gaussian
chain. It is click-run through the single editable `CONFIG` dictionary in
`verify_vfe4.py`; it has no required CLI or second launcher.

The implemented shared-integration path currently binds the exact current-HEAD
`CurrentH8PrerequisiteRefs`; the frozen parent request planner and injected
issued-prefix runner now exist. Lossless typed child evidence and exact problem
endpoints, authoritative runtime-section derivation and independent
revalidation, selected H8 runner/click-run wiring, and removal of the remaining
PASS blockers are still pending. Until all of those slices exist and
revalidate, H8 remains fail-closed `INCONCLUSIVE` in the absence of a witnessed
failure. The separate `verification/h8_preflight.py` path remains
metadata-only and zero-compute: it launches no correctness cell, runtime child,
profiler, control, or test. This preregistration contains no measured endpoint
and no PASS claim.

## Scope and source pins

The sole singular manuscript binding is the SHA-256 of the raw bytes at:

- `Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex`:
  `d733880d3613d32a97b7a12c93ff6c037d0abdfd9ce4810e411769997dbad03c`

This value is carried by `revision.manuscript_sha256`. It binds no MAgent
manuscript and asserts no theorem-level reduction between the VFE4 and MAgent
theories. The existing preregistration/source-pin digest separately binds the
imported VFE4 module bytes through the source-byte values below.

The frozen source-byte SHA-256 values are:

- `Manuscripts/vfe4_whitepaper/05_structured_information_form.tex`:
  `9f982907b1dea9e223d6bec277e68dd0248aeaa026142634f86459578a1768d0`
- `Manuscripts/vfe4_whitepaper/08_hypotheses_limitations.tex`:
  `cc53cd5f419fbff63421b28e38860711b048c18ed362220c5c01f8708574b679`
- `Manuscripts/vfe4_whitepaper/09_appendices.tex`:
  `144fe578b65f5028a9cc0263c070755074b0ee2ac67512e0a49832d7592cdf39`
- `docs/superpowers/specs/2026-07-21-vfe4-codebase-design.md`:
  `18ded9c876e681fdf9e070d113e093626ab41915b6269b65f8a38ac4fbb0eeb0`

The final H7 plan is independently pinned after strict UTF-8 decoding, BOM
rejection, CRLF/lone-CR normalization to LF, and UTF-8 re-encoding:

`3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242`

Any semantic source edit, H7-plan edit, configuration change, or changed
operational interpretation invalidates this preregistration before an H8
calculation can support promotion.

## Operational interpretation

The interpretation is an operational choice, not a theorem derived from the
manuscript:

- `choice_kind="operational_preregistration_not_manuscript_theorem"`
- `K_semantics="each_channel_dimension"`
- `T=128`, `N=T+1=129`
- `K=d_z=d_m=20`
- combined population block size `b=d_z+d_m=40`
- dense-equivalent arithmetic dimension `D=N*b=5160`
- categorical vocabulary `V=3`
- population-major coordinate order `[z_0,m_0,...,z_T,m_T]`
- the initial slice has no parent
- for every `t>=1`, both state and model parent/source supports are the
  singleton `{t-1}`

The canonical ASCII interpretation descriptor is:

```text
choice_kind=operational_preregistration_not_manuscript_theorem|K_semantics=each_channel_dimension|T=128|N=129|K=20|d_z=20|d_m=20|b=40|D=5160|V=3|coordinate_order=[z_0,m_0,...,z_T,m_T]|state_parent_sets=t0:none;t>=1:{t-1}|model_parent_sets=t0:none;t>=1:{t-1}|state_source_support=singleton_previous_slice|model_source_support=singleton_previous_slice|ambiguity_policy=changed_or_clarified_K_invalidates_and_yields_INCONCLUSIVE|source05_sha256=9f982907b1dea9e223d6bec277e68dd0248aeaa026142634f86459578a1768d0|source08_sha256=cc53cd5f419fbff63421b28e38860711b048c18ed362220c5c01f8708574b679|source09_sha256=144fe578b65f5028a9cc0263c070755074b0ee2ac67512e0a49832d7592cdf39|h7_plan_sha256=3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242
```

Its SHA-256 is
`e3fd048126c8133384e026826cf00bbea08280f4e248bc4cd5689e8f9f26e865`.
If the meaning of `K` is clarified or changed, no silent translation is
allowed: H8 is `INCONCLUSIVE` until a replacement interpretation and
preregistration are frozen.

## Sparse layout and exact storage arithmetic

Production owns only:

```text
h             [129,40]       5,160 scalars
J_diag        [129,40,40]  206,400 scalars
J_lower       [128,40,40]  204,800 scalars
L_diag        [129,40,40]  206,400 scalars
L_lower       [128,40,40]  204,800 scalars
Sigma_diag    [129,40,40]  206,400 scalars
Sigma_lower   [128,40,40]  204,800 scalars
```

Precision, factor, and selected-inverse storage are separate categories, each
exactly `411,200` float64 scalars and each independently capped at `411,200`.
The information vector is `5,160` scalars. The forbidden dense-equivalent
matrix would contain `D^2=26,625,600` scalars. Upper adjacent blocks are
transpose uses of lower blocks and never duplicate storage.

The precision pattern offsets are exactly `(-1,0,1)`; factor storage offsets
are exactly `(-1,0)`. Stored block IDs are all diagonal IDs in ascending
population order followed by all lower-adjacent IDs in ascending target order.

The allocation whitelist contains registered scalar/local/channel arrays,
`[N,b]`, `[N,b,1]`, `[N,b,r]` for `1<=r<=b`, `[N,b,b]`,
`[N-1,b,b]`, and exact generator/objective arrays with one population axis
`T`, `N`, or `N-1` and remaining axes no wider than `b` or `V`. Before
allocation, reject any axis `D`, storage above `411,200` float64-equivalent
scalars, two population/pair axes, triangular/all-pairs storage, or an
unregistered shape.

## Block factor and selected inverse

Local Cholesky recursion:

```text
L_diag[0]   = chol(J_diag[0])
L_lower[t]  = solve_triangular(L_diag[t], J_lower[t].T, upper=False).T
S[t+1]      = J_diag[t+1] - L_lower[t] @ L_lower[t].T
L_diag[t+1] = chol(S[t+1])
```

Forward and upper-triangular backward substitution accept only `[N,b]` or
`[N,b,r]`, `1<=r<=b`. In this protocol, "backward" means backward
substitution; autograd `backward()` is never called and H8 makes no gradient
claim.

Selected inverse uses the block Takahashi/LDL-equivalent recurrence. Let
`C_i=L_diag[i]`, `E_i=L_lower[i]`, `D_i=C_i C_i^T`, and
`F_i=E_i C_i^-1`, with `F_i` obtained by a local triangular solve, never an
inverse. Initialize:

```text
Sigma_diag[N-1] = D_[N-1]^-1
```

using a local Cholesky solve. Then, for `i=N-2..0`:

```text
Sigma_lower[i] = -Sigma_diag[i+1] @ F_i
Sigma_diag[i]  = D_i^-1 + F_i.T @ Sigma_diag[i+1] @ F_i
```

The sparse trace is:

```text
sum_i trace(J_left_diag[i] @ Sigma_diag[i])
+ 2*sum_i sum(J_left_lower[i] * Sigma_lower[i])
```

Sparse precision does not imply sparse covariance. The full covariance is
generally dense; H8 computes only diagonal and adjacent covariance blocks
needed by local expectations and the sparse trace. It makes no sparse
covariance claim.

Every factor records local pivot minima, the global minimum, and margins
relative to `H8_MIN_CHOLESKY_PIVOT=1e-8`. No jitter, repair, or dense fallback
is allowed. `HagerHigham1NormEstimate-v1` is the sole global condition
diagnostic: width-one all-positive start, at most eight iterations, zero sign
mapped to `+1`, lexicographically first maximizer, and repeated-index or
dot-product stopping. It is labeled an estimate and cannot alter a budget or
status.

## Deterministic problem and sample generation

Problem schema: `h8-synthetic-chain-v1`.

```python
rng = np.random.Generator(np.random.PCG64(problem_seed))
sn = lambda shape: np.ascontiguousarray(
    rng.standard_normal(size=shape, dtype=np.float64)
)

initial_mean = 0.1 * sn((b,))
Q_initial = sn((b, b))
initial_cov = spd(Q_initial, b)

for t in range(1, N):
    A_m[t] = contract(sn((K, K)) / np.sqrt(K), 0.35)
    c_m[t] = 0.05 * sn((K,))
    Q_m = sn((K, K)); R_m[t] = spd(Q_m, K)
    A_z[t] = contract(sn((K, K)) / np.sqrt(K), 0.35)
    B[t] = contract(sn((K, K)) / np.sqrt(K), 0.20)
    c_z[t] = 0.05 * sn((K,))
    Q_z = sn((K, K)); R_z[t] = spd(Q_z, K)

recognition_initial_mean = 0.1 * sn((b,))
Q_recognition_initial = sn((b, b))
recognition_initial_cov = spd(Q_recognition_initial, b)
for t in range(1, N):
    A_recognition[t] = contract(sn((b, b)) / np.sqrt(b), 0.35)
    c_recognition[t] = 0.05 * sn((b,))
    Q_recognition = sn((b, b))
    R_recognition[t] = spd(Q_recognition, b)

alpha = np.asarray((-0.5, 0.25, 0.75), dtype=np.float64)
for t in range(1, N):
    w[t] = sn((b,)) / np.sqrt(b)
    beta[t] = 0.1 * sn((V,))
    x[t] = (problem_seed + t) % V
```

Here
`contract(M,r)=r*M/max(r,np.linalg.norm(M,ord=2))` and
`spd(Q,n)=0.25*I+0.05*(Q@Q.T)/n`. `Normal(0,1/K)` means variance, so the
standard-deviation multiplier is `1/sqrt(K)`.

The canonical one-line draw descriptor is:

```text
numpy.Generator(numpy.PCG64(problem_seed))|method=standard_normal|dtype=float64|order=C|initial:mu0[b],Q0[b,b]|transition:t=1..T:{A_m[K,K],c_m[K],Q_m[K,K],A_z[K,K],B[K,K],c_z[K],Q_z[K,K]}|recognition_initial:mu_q0[b],Q_q0[b,b]|recognition_transition:t=1..T:{A_q[b,b],c_q[b],Q_q[b,b]}|emission:t=1..T:{w[b],beta[V]}|normal_map_variance=1/dim=>multiply_standard_normal_by_1/sqrt(dim)|serialize=after_all_problem_draws_before_sample_rng|bytes=little-endian-f8-C-contiguous
```

Its SHA-256 is
`7b657e72219f044147a7b414354d34c82bbd5a66d24f669285906d54534723c0`.
All arrays are validated and serialized as C-contiguous little-endian `<f8`
after the final problem draw and before constructing the independent sample
generator. Production and both references parse the same immutable bytes;
they never regenerate or reopen them.

Sample schema: `h8-pcg64-sample-v1`. A new
`Generator(PCG64(sample_noise_seed))` draws exactly `N*b` float64 values in
one call and C-order reshapes to `[N,b]`. The streams never derive from or
advance each other.

Literal correctness table:

| Cell | T | K=d_z=d_m | Problem seed | Sample-noise seed |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 2026072111 | 2026172111 |
| 2 | 1 | 2 | 2026072112 | 2026172112 |
| 3 | 1 | 4 | 2026072114 | 2026172114 |
| 4 | 2 | 1 | 2026072121 | 2026172121 |
| 5 | 2 | 2 | 2026072122 | 2026172122 |
| 6 | 2 | 4 | 2026072124 | 2026172124 |
| 7 | 4 | 1 | 2026072141 | 2026172141 |
| 8 | 4 | 2 | 2026072142 | 2026172142 |
| 9 | 4 | 4 | 2026072144 | 2026172144 |
| 10 | 8 | 1 | 2026072181 | 2026172181 |
| 11 | 8 | 2 | 2026072182 | 2026172182 |
| 12 | 8 | 4 | 2026072184 | 2026172184 |

Production seed/noise pairs are literally `(20260721,20261721)`,
`(20260722,20261722)`, and `(20260723,20261723)`.

## Complete normalized objective and correctness comparisons

The synthetic model is:

```text
m_0,z_0 ~ one normalized joint Gaussian
m_t | m_[t-1] ~ Normal(A_m[t]m_[t-1]+c_m[t],R_m[t])
z_t | z_[t-1],m_t ~ Normal(A_z[t]z_[t-1]+B[t]m_t+c_z[t],R_z[t])
x_t | y_t ~ Categorical(softmax(alpha*(w_t.T y_t)+beta_t)), V=3
```

The objective is the complete `E_q[log p(y,x)] + H(q)`: one initial joint
term, every named model transition, every named state transition, every
normalized categorical emission, and recognition entropy. Model-source KL,
state-source KL, and source entropy are explicit exact zeros because source
supports are singletons. Emissions use stable `log_softmax` and frozen
Gauss-Hermite orders 21 and 17; their absolute difference enters only the
matching emission operand.

Each correctness cell compares the block path, bounded dense PyTorch adapter,
and independently authored NumPy oracle for factor reconstruction, forward
substitution, backward substitution, solve, log determinant, quadratic,
same-noise sample, all selected blocks, sparse trace, entropy, log normalizer,
every objective term, and total. Both dense references reject `T>8` or
`K>4` before allocation. No `allclose`, global tolerance, or post-observation
tuning is admissible.

## Frozen numerical budgets

```text
EPS                    = finfo(float64).eps
ROUNDING_MULTIPLIER    = 4096
SOLVER_RELATIVE_BUDGET = 1e-9
MAX_ALLOWANCE_FRACTION = 1e-4
MIN_CHOLESKY_PIVOT     = 1e-8
```

`gamma(n)=n*EPS/(1-n*EPS)` rejects bools, nonpositive `n`, and
`n*EPS>=1`. For an operand:

```text
4096*gamma(local_operation_count)*max(1,absolute_sum)
+ (1e-9*max(1,infinity_norm) when solver_produced, else 0)
+ the operand's own emission quadrature contribution
```

For a comparison, add both operand allowances and:

```text
4096*gamma(compared_scalar_count+1)
* max(1,left_infinity_norm,right_infinity_norm)
```

The comparison is decisive only when `allowance/scale < 1e-4`; equality is
`INCONCLUSIVE`. A decisive finite residual passes when
`residual<=allowance`; equality passes. A decisive finite residual above its
own allowance fails. The condition estimate never scales an allowance.

The six exact wrong-path correctness controls are a perturbed solve element,
reversed logdet sign, transposed adjacent covariance, duplicated off-diagonal
trace, omitted entropy, and independent replacement sample noise. Each must
be decisive under its own correct-path allowance.

## One-thread child and 30-attempt resource protocol

Normative production is PyTorch float64 CPU under `torch.no_grad()`, one
intra-op thread, and one inter-op thread. Before a child imports NumPy or
PyTorch, the parent sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`. Before tensor work, the child sets and verifies
both PyTorch thread getters equal one.

Production traversal is seed-major:

```text
20260721 repetitions 0,1,2,3,4
20260722 repetitions 0,1,2,3,4
20260723 repetitions 0,1,2,3,4
```

Every repetition is a fresh child process. There are exactly 15 eligible
production runs and no retry. A separate profiler child runs once per seed.
Every negative control runs in isolation. The parent freezes this 30-request
plan, in order:

1. 15 `production` requests in the seed-major/repetition order above;
2. three `profiler` requests in seed-major order, with `repetition=null`;
3. 12 `negative_control` requests in the frozen control order below, each with
   `seed=20260721` and `repetition=null`.

The request sequence is frozen before launch. Every launch actually issued
yields one parent-owned `H8ChildAttemptRecord`, even if the process times out,
exits abnormally, emits malformed output, or supplies no typed child/control
result. A witnessed failure may stop later launches; PASS requires all 30.
There is no retry. Required operation reachability is factorization, forward
substitution, backward substitution, mean solve, logdet, all
diagonal/adjacent selected-inverse blocks, width-one sample, quadratic, sparse
trace, condition estimate, entropy, log normalizer, and complete objective.

The canonical parent/child protocol digest preimage freezes these exact
schema literals:

```text
factor_schema="h8-block-tridiagonal-cholesky-v1"
selected_inverse_schema="h8-block-takahashi-selected-inverse-v1"
condition_estimator_schema="HagerHigham1NormEstimate-v1"
allocation_schema="h8-allocation-observability-v1"
profiler_raw_event_schema="h8-torch-profiler-raw-event-v1"
child_schema="h8-child-v2"
```

The first five values are the frozen evidence-schema literals; the sixth is
the required child-envelope literal. The digest preimage also includes the
complete required-operation inventory above, the exact ordered
negative-control inventory below, and every frozen numerical and boundary
constant, including tolerance, decisiveness, pivot, time, memory, storage,
RHS/sample-width, fill, and forbidden-attempt policies. Parent and child must
recompute the same digest from that complete preimage; omission or substitution
of any literal or inventory is `INCONCLUSIVE`.

The exact resolved H8 validation configuration is
`schema_version="h8-validation-config-v2"`, superseding
`h8-validation-config-v1` before scientific execution. V2 contains these six
new frozen fields:

```text
factor_schema="h8-block-tridiagonal-cholesky-v1"
selected_inverse_schema="h8-block-takahashi-selected-inverse-v1"
condition_estimator_schema="HagerHigham1NormEstimate-v1"
allocation_schema="h8-allocation-observability-v1"
profiler_raw_event_schema="h8-torch-profiler-raw-event-v1"
child_schema="h8-child-v2"
```

All six participate in the canonical resolved-config JSON, so the canonical
H8 config SHA changes from the v1 configuration identity accordingly.

The v4 production/profiler result body under the `h8-child-v2` envelope has
this exact key order:

```text
input_sha256,sample_noise_sha256,problem_evidence,objective,storage,fill,
workspace,counters,allocation,resources,diagnostics,
operation_reachability,residuals,resource_decisions,invariants
```

`problem_evidence` carries the exact generative, recognition, local-SPD,
transition-norm, and observation evidence defined below. Negative-control
results use the same v2 envelope but remain governed by the control-result
body and do not fabricate or insert production `problem_evidence`.

For each accepted production/profiler v2 PASS body, the parent retains a
separate private typed `H8DecodedPassEvidence` attached to the corresponding
attempt. It preserves `sample_noise_sha256`, exact `problem_evidence`,
diagnostics, the complete validated child allocation mapping, and exact
request/envelope/result identities. The lossless factor/allocation runtime
views are derived from this private evidence. Public `H8ChildResult` and the
published `production_runs`/`profiler_runs` schemas remain unchanged.

Each attempt retains its exact request, `status`, ordered `reasons`, optional
typed `result`, `timed_out`, `exit_code`, actual parent `parent_elapsed_ns`,
`request_sha256`, `identities_sha256`, `stdout_sha256`, `stderr_sha256`, and
an optional immutable `nonpass_envelope`, plus optional trusted raw
`operation_reachability`, `residuals`, and `resource_decisions`.
The request hash binds the canonical stdin request bytes; the identities hash
binds the exact canonical `VFE4_H8_CHILD_IDENTITIES_JSON` bytes; stdout/stderr
hashes bind the raw captured streams even when decoding fails. Parent elapsed
time is measured from spawn through parse. It is never written into the
child-authored resource object: `result.resources.parent_elapsed_ns=0` remains
the child protocol sentinel, while the actual value lives at
`attempt.parent_elapsed_ns`.

Resource limits are inclusive:

- parent elapsed time `<=60.0` seconds
- conservative incremental process HWM `<=134,217,728` bytes
- dispatch-observed live PyTorch population storage `<=67,108,864` bytes
- each precision/factor/selected category `<=411,200` scalars
- solve RHS width `<=40`
- sample width exactly `1`
- off-band fill and forbidden attempts exactly `0`

The primary HWM is
`max(0,post_lifetime_peak-pre_current_rss)`. The supplementary peak-to-peak
diagnostic is `max(0,post_lifetime_peak-pre_lifetime_peak)` and cannot rescue
a primary failure. Record all pre/post current RSS, lifetime peak, and private
bytes, both elapsed nanosecond endpoints, and every raw per-run endpoint.

On Windows, `PROCESS_MEMORY_COUNTERS_EX` uses the exact native field order
`cb`, `PageFaultCount`, `PeakWorkingSetSize`, `WorkingSetSize`,
`QuotaPeakPagedPoolUsage`, `QuotaPagedPoolUsage`,
`QuotaPeakNonPagedPoolUsage`, `QuotaNonPagedPoolUsage`, `PagefileUsage`,
`PeakPagefileUsage`, `PrivateUsage`; native size must be 80 bytes on 64-bit
or 44 bytes on 32-bit. A missing API/symbol, null handle, wrong `cb`, layout
mismatch, failed call, negative field, or missing failure error code is an
observability obligation, never zero.

## Allocation observability and profiler contract

The four primary channels are:

1. PyTorch dispatch request/result, stack, alias, and live-storage tracing.
2. A separate pinned PyTorch profiler run with lossless raw-event joins.
3. Backend counters, fill, workspace, RHS, sample, and selected-block records.
4. Clean-subprocess OS high-water memory.

The NumPy guard supplies its assigned controls. The validation field
`tracemalloc_supplementary` is literal JSON `null`, meaning "not collected."
It is excluded from `all_observable`, every memory decision, and overall
status, and cannot close, rescue, or override a claim. Documented lossy
profiler export rows, profiler aggregate/net deltas, and the Hager--Higham
estimate are likewise supplementary and cannot close or override a claim.

Profiler pins:

- `torch==2.9.1`
- `torch/profiler/_memory_profiler.py` SHA-256
  `b80b4d5b58e91d581b18082c462ec7f088ec6b46ea50a1a62e2714d517a6a1b1`
- `torch/profiler/profiler.py` SHA-256
  `2c35f649219fb912728819b7dc0be5a5f1bd54c1efcd9502b62d976aeb278d22`
- profiler API contract SHA-256
  `161a78f04c26fba19bb01ba6417f2cf8c00730ebeb8d007a4af0f4da433ba043`

The profiler call uses `record_shapes=True`, `profile_memory=True`, and
`with_stack=True`. It retains the live profiler object and joins
`profile._memory_profile().timeline` to
`experimental_event_tree()` in memory by the complete
private `TensorKey` identity plus version. Its exact persisted tensor-key
mapping is `{tensor_id,storage_ptr,allocation_id,device}`. Each raw record
retains source-row index, timestamp, action, key, version, signed bytes,
dtype, device, operator, stack, logical shape, classification, matched event
nodes, join-witness hash, and live bytes after the event.

The action union is exactly `PREEXISTING`, `CREATE`, `INCREMENT_VERSION`,
`DESTROY`. `PREEXISTING` establishes the unique baseline;
`CREATE` requires a dead identity at version zero; `INCREMENT_VERSION`
requires the live identity and next version with zero byte delta; `DESTROY`
requires the current live identity/version and removes its bytes. Reconstruct
in `(timestamp_ns,source_row_index)` order. Missing/nonunique joins, duplicate
creates, invalid versions, byte drift, negative live totals, or leaked
nonbaseline storage make the profiler channel unavailable or violated as
specified by status precedence. A lossy
`(timestamp,action,nbytes,category)` export row never substitutes for the
joined record.

Exact ordered negative controls and assigned channels:

1. `torch_matrix_d_d` — dispatch
2. `torch_flat_d2` — dispatch
3. `torch_near_d2` — dispatch storage cap
4. `torch_length_d` — dispatch forbidden axis
5. `torch_block_pair_slab` — dispatch shape semantics
6. `torch_triangular_pair_storage` — dispatch shape semantics
7. `torch_pair_stack` — dispatch stack semantics
8. `torch_eye_full_rhs` — backend counter plus dispatch
9. `torch_dense_eigvalsh` — dispatch operation classifier
10. `numpy_matrix_d_d` — NumPy guard
11. `numpy_outer_d_d` — NumPy guard
12. `numpy_matmul_d_d` — NumPy guard

Each control is intercepted before the dangerous result allocation or kernel.
An unavailable/unwitnessed assigned detector is `INCONCLUSIVE`; an operation
witnessed executing past a required detector is `FAIL`.

## Prerequisites and reference schemas

The exact verifier prefix is
`("H1","H2","H3","H4","H5","H6-Prefix","H7","H8")`.
H6-Prediction is a separately discriminated scientific prerequisite, not a
member of that tuple. H7 transitively binds current H1--H5, the active
H1-prefix-prior sibling, and independently produced H6-Prefix. H6-Prefix has
`predecessor_refs={}` and no H1--H5/H4 premise. H6-Prediction binds exact
H1/H2/H3/H5 scientific premises and never H4 or H4 timing.

Current H7 compatibility registry keys are exactly, in order,
`h1_h5`, `h1_prefix_prior`, `h6_prefix`, retaining every keyed payload hash
and candidate JUnit path/hash.

The sole authorizing H8 registry schema is
`h8-current-candidate-refs-v3`. It contains:

- `H8H1H5Reference(kind,artifact_path,manifest_sha256,result_path,
  result_sha256,content_hashes,payload_hashes,ledger_path,ledger_sha256,
  producer_head,producer_dirty_digest,candidate_junit_sha256,status)`
- `H8H1PrefixPriorReference` with the same common fields and kind
  `h1_prefix_prior`
- `H8H6PrefixReference` with the common fields plus literal
  `config_schema="h6-prefix-config-v3"`,
  `validation_schema="h6-prefix-validation-set-v2"`, and
  `certificate_set_schema="h6-prefix-certificate-set-v2"`;
  `config_sha256`, `workload_plan_sha256`,
  `validation_payload_sha256`, and
  `prefix_certificate_set_sha256`; and the exact ordered nonempty
  `semantic_families` rows
  `(semantic_family_index,semantic_family_sha256,
  validation_payload_sha256,certificate_sha256)`
- `H8H7Reference` with the common fields plus `result_pointer_path`,
  `result_pointer_sha256`, and `fixture_set_sha256`
- `H8H6PredictionReference` with the common fields plus the literal
  `prediction_schema="h6-prediction-amended-v2"`, the exact config,
  readiness, metrics, and result schema discriminators, `experiment_sha256`,
  `config_sha256`, `readiness_artifact_path`, `readiness_manifest_sha256`,
  `readiness_sha256`, exact ordered `correctness_artifact_paths` for
  H1/H2/H3/H5, `h1_prefix_prior_artifact_path`,
  `smc_accuracy_artifact_path`, `smc_accuracy_manifest_sha256`,
  `h6_prefix_artifact_path`, `h6_prefix_manifest_sha256`,
  `blinded_data_artifact_path`, `blinded_data_manifest_sha256`,
  `matching_artifact_path`, `matching_manifest_sha256`,
  `matching_set_sha256`,
  `h1_prefix_prior_generative_factor_schema_sha256`,
  `smc_bias_semantics_sha256`, `objective_gate_spec_sha256`,
  `metrics_sha256`, and its non-null producer JUnit hash

All status tags are literal `pass`. Keyed content and payload maps are
preserved losslessly. A content-hash key is an exact manifest-relative payload
path. H6-Prefix semantic families preserve runner-plan order with contiguous
indices and unique family hashes; each row binds that family's validation and
certificate digest in addition to the validation and certificate-set
aggregates. A legacy keyed `PrefixCaseKey` certificate map is not admissible
in the authorizing v3 shape. No copy of predecessor validation, certificate,
or ledger bytes is admissible.
The direct H1--H5, H1-prefix-prior, and H6-Prefix variants must match their
H7 transitive references field-for-field, including ordered keyed payload
hashes. The amended H6-Prediction retains its own frozen producer head, dirty
digest, and non-null JUnit hash; that producer identity need not match the H8
candidate. Its exact resolved config, readiness, scientific dependency set,
raw endpoints, metrics, result, immutable artifact identities, and ledger must
revalidate against that frozen producer before H8 can authorize. At H8
preflight, every exact artifact, result, ledger, H7 result pointer, shared H7
JUnit preimage, H6-Prediction readiness and complete
scientific-prerequisite artifact set, raw endpoint table, metrics file, and
result file is reopened by its frozen path and rehashed. H7 fixture closure is
rederived from the current-candidate `h1_v1.json`, `h7_v1.json`, and
`h7_density_probes_v1.json` bytes, including the typed density-probe set; the
H7 validation and reference fixture-set hashes must equal those preimages.
The blinded-data
preimage reconstructs the true held-out token count; the matching-set preimage
reconstructs the typed endpoint ownership inventory. The native H6 reader then
rederives raw aggregates, ordered OBJECTIVE/PRIMARY metrics, result identity,
and result-root name. Reopened bytes validate the registry record; they never
reconstruct it and are never copied into H8.
For H6-Prefix specifically, the native bounded reopener requires the exact
five-file manifest inventory, config v3 outer authorization, workload-plan v2
authorization, validation-set v2, and certificate-set v2; it reconstructs the
typed ordered certificate set and H8 compares every aggregate and family row
to the already parsed registry reference.

Registry v1 and v2 remain readable solely for historical diagnosis. Both
retain the legacy keyed H6-Prefix certificate shape and therefore add
`h8_prerequisite_legacy_registry_requires_bounded_h6_prefix_v3`. Registry v1's
`H8LegacyH6PredictionReference` lacks the amended bindings and therefore adds
the named prerequisite obligation
`h8_prerequisite_registry_v1_requires_amended_h6_prediction_v2`.
Registry v2 retains the amended H6-Prediction v2 compatibility island
unchanged; it is nonauthorizing only because its H6-Prefix shape is legacy.
Unavailable or changed immutable bytes add a reference-specific prerequisite
obligation. Either condition makes H8 `INCONCLUSIVE`; neither can authorize
`PASS`.

After publication only, the external current-candidate result has exact
top-level keys `schema_version`, `candidate`, `artifact`, `current_refs`,
`predecessors`; schema is `h8-current-candidate-result-v2`.
`candidate={git_head,dirty_digest,junit_sha256}`;
`artifact={path,manifest_sha256,config_sha256,validation_sha256}`;
`current_refs={path,sha256}`; and `predecessors` contains the five reference
variants verbatim. The external pointer is not part of the artifact manifest.

## Validation payload schema

The in-artifact file is `validation/h8.json`,
`schema_version="h8-sparse-scale-v4"`, `gate="H8"`. Its exact top-level key
order is:

```text
schema_version, gate, status, obligations, bounded_claim, nonclaims,
revision, config, prerequisites, interpretation, protocol, environment,
problems, storage, factor, correctness, allocation, controls,
child_attempts, production_runs, profiler_runs, budgets, invariants, artifacts
```

Required nested inventories:

- `revision`: `git_head`, `dirty_digest`, `dependency_closure_sha256`,
  `manuscript_sha256`, `preregistration_sha256`, `h7_plan_sha256`;
  `manuscript_sha256` is exactly
  `d733880d3613d32a97b7a12c93ff6c037d0abdfd9ce4810e411769997dbad03c`,
  the raw-byte digest of
  `Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex`
- `config`: canonical H8 config hash and exact resolved
  `h8-validation-config-v2`; its canonical JSON binds all six frozen protocol
  schema fields above, so it cannot retain the v1 canonical config SHA
- `prerequisites`: the complete H7 compatibility mapping, five lossless
  tagged H8 reference variants, exact compatibility-check inventory, named
  prerequisite obligations, and `all_current_and_pass`
- `interpretation`: interpretation hash, choice kind, K semantics, all
  dimensions/order/supports, and ambiguity policy
- `protocol`: generator/sample/child schemas, Torch/profiler pins, seed
  tables, operation inventory, and control order, plus the exact literals
  `factor_schema="h8-block-tridiagonal-cholesky-v1"`,
  `selected_inverse_schema="h8-block-takahashi-selected-inverse-v1"`,
  `condition_estimator_schema="HagerHigham1NormEstimate-v1"`,
  `allocation_schema="h8-allocation-observability-v1"`, and
  `profiler_raw_event_schema="h8-torch-profiler-raw-event-v1"`, plus
  `child_schema="h8-child-v2"`. The canonical parent/child protocol digest
  preimage includes the five evidence-schema literals, the child-envelope
  literal, the complete required-operation and negative-control inventories,
  and every frozen numerical and boundary constant.
- `environment`: platform, processor, CPU count, affinity, Python/PyTorch/
  NumPy, CPU/float64/no-grad, threads, thread environment, BLAS, and separate
  hardware/affinity/thread/BLAS hashes
- `problems`: exactly three seed-major records
  `{problem_seed,sample_noise_seed,input_sha256,sample_noise_sha256,
  generative_sha256,recognition_sha256,local_spd_diagnostics,
  transition_norms,observation_sha256}`. `generative_sha256` is the
  domain-separated SHA-256 of canonical JSON for
  `h8-generative-evidence-v1`; it covers layout/seed/vocabulary metadata,
  `alpha`, the initial mean/covariance, every model/state transition, and
  every emission. Every array is represented by exact shape, little-endian
  float64 dtype, and the SHA-256 of its raw C-order bytes; transition
  identities, source supports, and observations remain explicit.
  `recognition_sha256` is the analogous domain-separated
  `h8-recognition-evidence-v1` hash over the recognition initial
  mean/covariance and every ordered recognition transition.
  `local_spd_diagnostics` has
  `schema_version="h8-local-spd-diagnostics-v1"` and retains the minimum
  NumPy-Cholesky diagonal pivot for the generative initial covariance, every
  ordered model covariance, every ordered state covariance, the recognition
  initial covariance, every ordered recognition covariance, and the global
  minimum. These local problem-covariance diagnostics are distinct from the
  production block-factor pivots. `transition_norms` has
  `schema_version="h8-transition-norms-v1"`, `norm="operator_2"`, the ordered
  operator 2-norm arrays for model matrices, state matrices, state-model
  coupling matrices, and recognition matrices, and each inventory maximum.
  They are captured from the already required generator contraction
  calculation and never trigger a second SVD pass in each child.
  `observation_sha256` is the domain-separated SHA-256 of canonical JSON
  `{"domain":"vfe4.h8.observations.v1","records":[[receiver_t,x_t],...]}`
  in receiver order. Each seed record is reconstructed only after its problem
  evidence is byte-identical across all five production repetitions and that
  seed's profiler run; any disagreement is `INCONCLUSIVE`, and first-run
  selection is forbidden

Canonical JSON sorts keys before hashing. The exact generative and recognition
hash-preimage layouts and domain literals are:

```text
generative={
  domain:"vfe4.h8.generative-evidence.v1",
  schema_version:"h8-generative-evidence-v1",
  layout:{horizon,d_z,d_m},
  problem_seed,
  vocabulary_size,
  alpha:{shape,dtype,raw_sha256},
  initial:{mean:{shape,dtype,raw_sha256},
           covariance:{shape,dtype,raw_sha256}},
  model_transitions:[
    {receiver_t,parent_t,source_support,matrix,offset,covariance}, ...],
  state_transitions:[
    {receiver_t,parent_t,source_support,state_matrix,model_matrix,
     offset,covariance}, ...],
  emissions:[{receiver_t,weight,bias,observation}, ...]
}
recognition={
  domain:"vfe4.h8.recognition-evidence.v1",
  schema_version:"h8-recognition-evidence-v1",
  layout:{horizon,d_z,d_m},
  problem_seed,
  initial:{mean:{shape,dtype,raw_sha256},
           covariance:{shape,dtype,raw_sha256}},
  transitions:[
    {receiver_t,parent_t,source_support,matrix,offset,covariance}, ...]
}
```

Every elided array value in a transition or emission record, as well as every
explicit array leaf above, is exactly
`{shape,dtype:"<f8",raw_sha256}`. Both `shape` and `source_support` serialize
as JSON arrays of integers; integer identity, dimension, seed, vocabulary,
receiver, parent, and observation values remain JSON integers rather than
float or string encodings.

- `storage`: information, precision, factor, selected, category-cap, dense-
  forbidden counts, and three category decisions
- `factor` is exactly `{schema_version,algorithm,pattern,runs}`, with
  `schema_version="h8-factor-evidence-v1"`,
  `algorithm="block_tridiagonal_cholesky_local_recursion"`, and
  `pattern="symmetric_block_tridiagonal_diag_lower_only"`. `runs` is exactly
  18 records in authoritative `production_runs` then `profiler_runs` order,
  each
  `{mode,seed,repetition,input_sha256,fill,workspace,
  condition_diagnostics,counters,reconstruction_invariants}` and cross-bound
  to the corresponding result-bearing attempt. `condition_diagnostics`
  retains the complete typed `SparseConditionDiagnostics`, including the
  diagnostic-only condition estimate and every per-block/global pivot
  endpoint. First-run selection, maximum-only reduction, and silent cross-run
  collapse are forbidden
- `correctness`: literal grid order, all cells, count, completeness,
  decisiveness, and pass decision; each cell retains dimensions, both seeds
  and hashes, three source results, every pair comparison, wrong-path
  controls, status, and obligations
- `allocation` is exactly
  `{schema_version,whitelist,runs,tracemalloc_supplementary,all_observable,
  no_forbidden_attempts}`, with
  `schema_version="h8-allocation-evidence-v1"`. `runs` is exactly the same
  18-record authoritative order, each
  `{mode,seed,repetition,input_sha256,allocation,resources}` and cross-bound
  to the corresponding result-bearing attempt. `allocation` is the complete
  validated child mapping, not a reduced `H8AllocationRecord`: it retains
  dispatch events/scopes/storage, the NumPy inventory and guard events,
  profiler API/raw events/lossy supplementary rows, preexisting baseline and
  liveness reconstruction, and raw dispatch/backend cross-checks. Persisted
  profiler tensor keys are exactly
  `{tensor_id,storage_ptr,allocation_id,device}`. `resources` is the exact
  child-authored `H8ResourceRecord`, including `parent_elapsed_ns=0`; actual
  parent elapsed time remains only in the attempt.
  `tracemalloc_supplementary` is literal JSON `null`, meaning "not collected,"
  and is excluded from `all_observable`, all memory decisions, and overall
  status. It cannot close, rescue, or override a claim
- `controls`: exact ordered records retaining requested operation, logical
  shapes, assigned/observed channels, witness/event hash, assignment,
  detection, status, and obligations
- `child_attempts`: the parent-owned frozen-order prefix of at most 30
  attempt records in the 15-production, three-profiler, 12-control inventory;
  a witnessed FAIL may close the prefix, while PASS requires all 30. Every
  record retains
  `request`, `status`, `reasons`, `result_kind`, optional `result_identity`,
  optional immutable `nonpass_envelope`,
  `timed_out`, `exit_code`, actual `parent_elapsed_ns`, `request_sha256`,
  `identities_sha256`, `stdout_sha256`, `stderr_sha256`, and optional raw
  `operation_reachability`, `residuals`, and `resource_decisions`.
  `result_kind="child"` uses exact identity
  `{mode,seed,repetition,input_sha256}`; `result_kind="control"` uses
  `{control_id,event_sha256}`; a launch with no decoded typed result stores
  null for both fields rather than fabricating a result; a parseable envelope
  from a timed-out launch, a parseable non-PASS envelope, or an
  identity-rejected envelope is retained separately without trusting it
- `production_runs`: the ordered decoded child records cross-bound to
  result-bearing production attempts; PASS requires exactly 15 seed-major
  records
- `profiler_runs`: the ordered decoded child records cross-bound to
  result-bearing profiler attempts; PASS requires exactly three seed-major
  records
- `budgets`: epsilon, rounding multiplier, solver fraction, decisiveness
  fraction, minimum pivot, seconds, process bytes, Torch bytes, storage
  scalars, and boundary policy
- `invariants`: every named prerequisite, interpretation, correctness,
  control, attempt completeness/exact order/cross-binding,
  observability/join/liveness, decoded-run completeness, operation, storage,
  fill, pivot, RHS/sample, time/memory, finite, residual, dominance, and
  all-pass decision
- `artifacts`: config, provenance, environment, H7 reference,
  H6-Prediction reference, validation, and manifest paths; no enclosing
  manifest hash or external-pointer hash

The existing `production_runs` and `profiler_runs` arrays remain the
authoritative ordering and published typed-result inventories. The
factor/allocation `runs` arrays are lossless evidence views in that same order,
derived from private `H8DecodedPassEvidence` and cross-bound to those unchanged
public inventories and their result-bearing attempts; they do not replace or
expand either authoritative array. Negative-control results use the same v2
envelope without fabricating private production problem evidence.

The decoded `controls` inventory is the exact ordered prefix of typed
`H8ControlResult` records cross-bound to result-bearing control attempts; PASS
requires all 12. Every production/profiler child record retains its
`H8ChildResult` fields plus parent/child elapsed nanoseconds, exit code,
stdout/stderr hashes, operation reachability, residuals, and resource
decisions. Its nested child-authored `resources` object is not rewritten: the
protocol sentinel `resources.parent_elapsed_ns=0` remains intact, and the
actual parent spawn-through-parse duration appears only in the attempt and the
decoded record's separate top-level `parent_elapsed_ns`. No raw endpoint may
be replaced by a maximum-only summary.

## Status precedence

Before a valid configuration, interpretation, preregistration, and current
PASS prerequisite start exists, missing, stale, ambiguous, differently
configured, or hash-incompatible evidence yields H8 `INCONCLUSIVE` and no
child suite starts. A prior-gate FAIL remains that gate's result; H8 does not
relabel it as an H8 systems failure.

After a valid start, every launch actually issued retains an attempt record.
Any witnessed attempt timeout, OOM/abnormal or nonzero exit,
forbidden allocation/operation, off-band fill, nonfinite contract value,
solver inability, omitted required operation in a completed run,
thread/environment identity mismatch, invalid profiler version/liveness
transition, finite residual/resource/pivot breach, or executed control missed
by its detector is H8 `FAIL`. This witnessed failure dominates missing later
evidence or a missing typed result. In the absence of a witnessed violation,
unavailable/unwitnessed evidence, a missing typed result, a missing or
nonunique profiler join, missing hash, or incomplete control/observability
channel is `INCONCLUSIVE`.

PASS requires the conjunction of all 12 complete decisive correctness cells,
all 30 attempts present in exact order with current request/identity/stream
hashes and PASS status, every result-bearing attempt cross-bound to the
corresponding decoded inventory, all 15 production runs, all three profiler
runs, all 12 assigned controls, all four primary observability channels,
joined and reconciled profiler events, complete operation reachability,
bounded storage/RHS/sample/fill/pivots/time/memory, matching child identities,
finite outputs, and every residual within its own allowance.

The current staged gate has the frozen parent request planner and injected
issued-prefix runner, but it still lacks lossless typed child evidence,
authoritative runtime-section derivation and independent revalidation, and
selected H8 runner/click-run wiring. The remaining PASS locks, including
runtime sections not being bound and the staged parent-orchestrator blocker,
may be removed only after those paths are complete and independently
revalidated. Until then, the schema and status logic above are PASS-capable
contracts, not a claim that the current shared runner can attain PASS;
fail-closed `INCONCLUSIVE` semantics remain in force absent a witnessed
failure.

## Claims and nonclaims

Only a PASS artifact may state:

> The frozen T=128, K=d_z=d_m=20 synthetic chain completed within the
> preregistered sparse storage, allocation, numerical, time, and memory
> contract.

FAIL or INCONCLUSIVE prefixes that sentence with `NOT ESTABLISHED:`.

The exact ordered nonclaims are:

1. `no_language_result`
2. `no_training_result`
3. `no_prediction_result`
4. `no_large_language_model_scale`
5. `no_asymptotic_scaling_law`
6. `no_gpu_claim`
7. `no_exact_global_spectrum`
8. `no_post_h8_training_memory_transfer`

H8 does not certify WikiText-103 training, tokenization, batching, decoding,
optimizer behavior, checkpoints, held-out NLL/perplexity, GPU memory,
throughput, sparse attention, a million-token regime, or an asymptotic
complexity law. Any later WikiText-103 milestone requires its own allocation,
training, evaluation, recording, and figure-generation review.
