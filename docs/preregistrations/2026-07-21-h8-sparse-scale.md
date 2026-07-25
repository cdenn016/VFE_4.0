# H8 Sparse-Scale Systems Preregistration

Protocol revision: `h8-sparse-scale-v1`

Status at freeze: protocol only. No H8 correctness grid, production child,
profiler child, negative control, timing measurement, memory measurement, or
promotion decision has been executed. This document therefore contains no
measured endpoint and does not prestate PASS.

H8 is a synthetic empirical systems gate for one block-tridiagonal Gaussian
chain. It is click-run through the single editable `CONFIG` dictionary in
`verify_vfe4.py`; it has no required CLI or second launcher.

The implemented shared-integration path is source-only. It requires the exact
current-HEAD `CurrentH8PrerequisiteRefs`, enters neither the legacy H1--H5/H7
runner nor any H8 correctness/child/profiler/control execution, and publishes
empty runtime inventories. Its honest result is therefore `INCONCLUSIVE` with
the source-only obligations; this preregistration still contains no measured
endpoint and no PASS claim.

## Scope and source pins

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

## One-thread child and 15-run resource protocol

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
Every negative control runs in isolation. Required operation reachability is
factorization, forward substitution, backward substitution, mean solve,
logdet, all diagonal/adjacent selected-inverse blocks, width-one sample,
quadratic, sparse trace, condition estimate, entropy, log normalizer, and
complete objective.

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

The NumPy guard supplies its assigned controls. `tracemalloc`, documented
lossy profiler export rows, profiler aggregate/net deltas, and the Hager-
Higham estimate are supplementary and cannot close or override a claim.

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
`TensorKey(id,storage.ptr,allocation_id,device)` plus version. Each raw record
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
and candidate JUnit hash.

The H8 registry contains:

- `H8H1H5Reference(kind,artifact_path,manifest_sha256,result_path,
  result_sha256,content_hashes,payload_hashes,ledger_path,ledger_sha256,
  producer_head,producer_dirty_digest,candidate_junit_sha256,status)`
- `H8H1PrefixPriorReference` with the same common fields and kind
  `h1_prefix_prior`
- `H8H6PrefixReference` with the common fields plus
  `certificate_set_sha256` and keyed `certificate_hashes`
- `H8H7Reference` with the common fields plus `result_pointer_path`,
  `result_pointer_sha256`, and `fixture_set_sha256`
- `H8H6PredictionReference` with the common fields plus
  `experiment_sha256` and optional candidate JUnit

All status tags are literal `pass`. Keyed content, payload, and certificate
maps are preserved losslessly. No singular aggregate replacement and no copy
of predecessor validation, certificate, or ledger bytes is admissible.

After publication only, the external current-candidate result has exact
top-level keys `schema_version`, `candidate`, `artifact`, `current_refs`,
`predecessors`; schema is `h8-current-candidate-result-v1`.
`candidate={git_head,dirty_digest,junit_sha256}`;
`artifact={path,manifest_sha256,config_sha256,validation_sha256}`;
`current_refs={path,sha256}`; and `predecessors` contains the five reference
variants verbatim. The external pointer is not part of the artifact manifest.

## Validation payload schema

The in-artifact file is `validation/h8.json`,
`schema_version="h8-sparse-scale-v1"`, `gate="H8"`. Its exact top-level key
order is:

```text
schema_version, gate, status, obligations, bounded_claim, nonclaims,
revision, config, prerequisites, interpretation, protocol, environment,
problems, storage, factor, correctness, allocation, controls,
production_runs, profiler_runs, budgets, invariants, artifacts
```

Required nested inventories:

- `revision`: `git_head`, `dirty_digest`, `dependency_closure_sha256`,
  `manuscript_sha256`, `preregistration_sha256`, `h7_plan_sha256`
- `config`: canonical H8 config hash and exact resolved configuration
- `prerequisites`: the complete H7 compatibility mapping and five lossless
  tagged H8 reference variants
- `interpretation`: interpretation hash, choice kind, K semantics, all
  dimensions/order/supports, and ambiguity policy
- `protocol`: generator/sample/factor/selected-inverse/condition/allocation/
  child schemas, Torch/profiler pins, seed tables, operation inventory, and
  control order
- `environment`: platform, processor, CPU count, affinity, Python/PyTorch/
  NumPy, CPU/float64/no-grad, threads, thread environment, BLAS, and separate
  hardware/affinity/thread/BLAS hashes
- `problems`: three ordered production records with both seeds, input/noise/
  generative/recognition hashes, local SPD diagnostics, transition norms, and
  observation hash
- `storage`: information, precision, factor, selected, category-cap, dense-
  forbidden counts, and three category decisions
- `factor`: algorithm, pattern, fill, workspace, diagnostic condition
  estimate, all pivot minima/margins, counters, and reconstruction invariants
- `correctness`: literal grid order, all cells, count, completeness,
  decisiveness, and pass decision; each cell retains dimensions, both seeds
  and hashes, three source results, every pair comparison, wrong-path
  controls, status, and obligations
- `allocation`: whitelist, dispatch/live storage, profiler API and all raw
  events, preexisting counts/bytes/baseline, reconstructed peak,
  supplementary deltas, backend, OS HWM, supplementary tracemalloc,
  cross-checks, observability, and forbidden-attempt decision
- `controls`: exact ordered records retaining requested operation, logical
  shapes, assigned/observed channels, witness/event hash, assignment,
  detection, status, and obligations
- `production_runs`: exactly 15 seed-major child records
- `profiler_runs`: exactly three seed-major child records
- `budgets`: epsilon, rounding multiplier, solver fraction, decisiveness
  fraction, minimum pivot, seconds, process bytes, Torch bytes, storage
  scalars, and boundary policy
- `invariants`: every named prerequisite, interpretation, correctness,
  control, observability/join/liveness, run completeness, operation, storage,
  fill, pivot, RHS/sample, time/memory, finite, residual, dominance, and
  all-pass decision
- `artifacts`: config, provenance, environment, H7 reference,
  H6-Prediction reference, validation, and manifest paths; no enclosing
  manifest hash or external-pointer hash

Every production/profiler child record retains its `H8ChildResult` fields plus
parent/child elapsed nanoseconds, exit code, stdout/stderr hashes, operation
reachability, residuals, and resource decisions. No raw endpoint may be
replaced by a maximum-only summary.

## Status precedence

Before a valid configuration, interpretation, preregistration, and current
PASS prerequisite start exists, missing, stale, ambiguous, differently
configured, or hash-incompatible evidence yields H8 `INCONCLUSIVE` and no
child suite starts. A prior-gate FAIL remains that gate's result; H8 does not
relabel it as an H8 systems failure.

After a valid start, any witnessed timeout, OOM/abnormal or nonzero exit,
forbidden allocation/operation, off-band fill, nonfinite contract value,
solver inability, omitted required operation in a completed run,
thread/environment identity mismatch, invalid profiler version/liveness
transition, finite residual/resource/pivot breach, or executed control missed
by its detector is H8 `FAIL`. This witnessed failure dominates missing later
evidence. In the absence of a witnessed violation, unavailable/unwitnessed
evidence, a missing or nonunique profiler join, missing hash, or incomplete
control/observability channel is `INCONCLUSIVE`.

PASS requires the conjunction of all 12 complete decisive correctness cells,
all assigned controls, all four primary observability channels, joined and
reconciled profiler events, 15 production runs, three profiler runs, complete
operation reachability, bounded storage/RHS/sample/fill/pivots/time/memory,
matching child identities, finite outputs, and every residual within its own
allowance.

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
