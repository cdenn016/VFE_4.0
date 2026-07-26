# VFE 4.0 H8 Sparse-Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evidence-gate the synthetic H8 reference at `T=128`, `K=d_z=d_m=20` with exact block-tridiagonal Gaussian operations, a complete normalized synthetic objective, and allocation/resource certification that forbids every global `D x D` or equivalent quadratic buffer.

**Architecture:** Add a dependency-free PyTorch float64 CPU block-tridiagonal backend whose public representation is `(h, J_diag, J_lower)` and whose factor is `(L_diag, L_lower)`. A factor-backed information Gaussian, block-local canonical assembler, and synthetic chain objective consume only block-local operations; an independent dense NumPy oracle and a verification-only bounded dense-PyTorch adapter check a small correctness grid. H8 runs in clean child processes under four complementary observability channels, then one fail-closed gate publishes revision-bound results only after the complete final-H7 current-candidate chain and compatible separate H6-Prediction evidence are validated.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, standard-library `subprocess`/`ctypes`/`resource`/`tracemalloc`, pytest with JUnit XML, existing atomic JSON artifacts, installed evidence-gated verification ledger.

**Plan inventory:** 8 implementation/evidence tasks and 74 checkbox steps. Task 8 is source read-only and owns the exact one-JUnit current-candidate evidence lifecycle.

## Global Constraints

- H8 is an **empirical systems gate only**. It makes no language-modeling, held-out prediction, training, large-language-model, million-token, or asymptotic scaling claim.
- Execute H8 only after compatible current H1, H2, H3, H4, H5, H6-Prefix, H6-Prediction, and H7 evidence is PASS. The exact verifier prefix is `("H1","H2","H3","H4","H5","H6-Prefix","H7","H8")`; H6-Prediction remains a separate empirical artifact reference rather than a member of that verifier tuple. H7 transitively binds H1--H5 and the independently produced H6-Prefix sibling reference; H8 separately validates the H7 artifact/ledger and the H6-Prediction artifact. H6-Prefix has no H1--H5/H4 predecessor. H6-Prediction separately binds exact H1/H2/H3/H5 correctness evidence and never H4 or an H4 timing result; H8's own current H1--H5/H7 chain does not retroactively add H4 to Prediction provenance. Missing, stale, differently configured, differently sourced, or hash-incompatible prerequisite evidence makes H8 `INCONCLUSIVE` before the child suite starts.
- Because H8 changes H7-owned config/runner/provenance dependencies, the final H8 candidate reproduces the complete final-H7 current-candidate lifecycle, in this exact order after one JUnit XML and with no tracked edit: H1--H5 artifact/ledger; active H1-prefix-prior artifact/ledger; independently projected H6-Prefix artifact/certificate-set/ledger with `predecessor_refs={}`; H7 predecessor registry; H7 artifact; external H7 result pointer; H7 ledger; H8 predecessor registry; H8 artifact; external H8 result pointer; H8 ledger. It calls H7's pure `project_h1_h5_compatibility_config`, H6's amended exact `project_h1_prefix_prior_v2_config`, `project_h6_prefix_config`, and keyword-only `run_projected_current_candidate` interfaces, plus H8's pure pinned-schema `project_h7_compatibility_config`; it uses reference identities only. The v2 projector produces `h1-prefix-prior-validation-v3`, including the same-candidate JUnit binding, without editing the pinned H7 plan. H1--H5 is produced through the existing ordered verifier, the two H6 projections use the H6 producer, and H7 is produced through its selected verification operation rather than overloading the H6 producer. It never edits `CONFIG`, copies predecessor payloads/certificates/ledgers, or reruns the broad suite. The earlier H6-Prediction artifact remains separate scientific evidence and is never reproduced merely for H8: it stays eligible only when its frozen H1/H2/H3/H5 scientific dependency closure/config/data/estimator/checkpoint identities are unchanged, no H4 timing identity is present, and H7/H8 append-only branches are mechanically unreachable from its selected operation; otherwise H8 is `INCONCLUSIVE`.
- Preregister `K=d_z=d_m=20`, `T=128`, `N=T+1=129`, combined population block size `b=d_z+d_m=40`, and `D=N*b=5160`, in population-major coordinate order `[z_0,m_0,...,z_T,m_T]`, as an **operational H8 choice**. The manuscript text is compatible with, but does not prove or uniquely determine, this interpretation; `K` means each channel dimension here, not the combined block dimension. Bind the exact choice, coordinate order, rationale, and source hashes into `interpretation_sha256`. If the author later changes or clarifies `K`, do not translate silently: invalidate the preregistration and return `INCONCLUSIVE` until a new interpretation is frozen.
- Use a chain only: the initial slice has no parent; every `t>=1` state and model parent/source support is the singleton `{t-1}`. Use a normalized synthetic categorical emission with `V=3`. Do not expand parent width, enumerate source mixtures, ingest a corpus, or enter any training path.
- Normative execution is no-grad PyTorch float64 on CPU with one intra-op thread and one inter-op thread. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` in the parent environment before the child imports NumPy or PyTorch. Every child, before any tensor work, must call `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`, then verify both getters equal one; an unavailable setter/getter or API error is an environment observability obligation, never silently accepted.
- Production precision storage is `J_diag[N,b,b]` plus `J_lower[N-1,b,b]`, exactly `411,200` float64 scalars, with `h[N,b]` stored separately (`5,160` scalars). Factor storage `L_diag/L_lower` and selected inverse storage `Sigma_diag/Sigma_lower` are each exactly `411,200` scalars. Each of the three bounded categories must remain `<=411,200`; upper adjacent blocks are transpose views, never duplicate storage.
- The production allocation whitelist admits only registered scalar/local/channel arrays, `[N,b]`, `[N,b,1]`, `[N,b,r]` with `1<=r<=b`, `[N,b,b]`, `[N-1,b,b]`, and the exact generator/objective arrays whose sole population axis is `T`, `N`, or `N-1` and whose remaining axes are each `<=b` or `V=3`. Reject before allocation any tensor/array with an axis of length `D`, any single storage above `411,200` float64-equivalent scalars, any two population/pair axes, any triangular or equivalent all-pairs slab, and any unregistered production shape. Thus a dense population `J`, `Sigma`, moment matrix `M`, identity, selector, Cholesky factor, length-`D` vector, flat or near-`D^2` buffer, `(D,D)` batch/RHS, `(N,N,b,b)` block tensor, triangular pair storage, or combined slab of all block pairs is forbidden. No convenience diagnostic is exempt.
- The factor interface retains `solve`, `logdet`, `selected_inverse`, `sample`, `quadratic`, and `trace_inverse_product`, and exposes immutable layout, pattern, storage, fill, workspace, condition-estimate, and backend-counter records. Maximum solve RHS width is `b`; H8 sampling accepts exactly one vector/noise path at a time.
- Build the factor-backed Gaussian from `h+factor` without retaining input `J`. Assemble canonical factors by scattering only into one diagonal block or one adjacent block pair; never allocate a length-`D` residual vector or call a length-`D` outer product.
- The production exercise must reach factorization, forward and backward block substitution, mean solve, `logdet`, every diagonal and adjacent selected-inverse block, a width-one sample, quadratic form, sparse trace product, entropy, log normalizer, and the complete synthetic objective.
- Production diagnostics may call `torch.linalg.eigvalsh` only on local matrices of dimension at most `b=40`. Global conditioning uses the named sparse Hager--Higham 1-norm estimator, is labeled an estimate, and is **diagnostic only**: it cannot alter a numerical allowance or PASS/FAIL/INCONCLUSIVE status. Record local Cholesky pivot minima for every block, their global minimum, and both per-block and global margins relative to `H8_MIN_CHOLESKY_PIVOT`; do not publish the estimate as an exact eigenvalue extremum or exact `kappa_2`.
- Allocation certification has four required primary channels: PyTorch dispatch whitelist/stack/live-storage tracing, a separate PyTorch profiler run that preserves and reconciles all four raw timeline actions `PREEXISTING`/`CREATE`/`INCREMENT_VERSION`/`DESTROY`, backend counters, and clean-subprocess OS high-water memory. `tracemalloc`, the documented lossy raw-export rows, and profiler aggregate/net memory deltas are supplementary only and cannot close an allocation claim.
- The exact ordered negative-control IDs and assigned channels are: `torch_matrix_d_d` and `torch_flat_d2` (dispatch); `torch_near_d2` for `(D-1,D-1)` (dispatch storage cap); `torch_length_d` (dispatch forbidden axis); `torch_block_pair_slab` for `(N,N,b,b)`, `torch_triangular_pair_storage` for `(N*(N+1)//2,b,b)`, and `torch_pair_stack` (dispatch shape/stack semantics); `torch_eye_full_rhs` (backend counter plus dispatch); `torch_dense_eigvalsh` (dispatch operation classifier); `numpy_matrix_d_d`, `numpy_outer_d_d`, and `numpy_matmul_d_d` (NumPy guard). Every control must be intercepted before a dangerous allocation or kernel. A missing assigned channel or an unavailable/unwitnessed control is `INCONCLUSIVE`; a control that executes but evades any required detector is an observed `FAIL`.
- Correctness grid is the Cartesian product `T in {1,2,4,8}` and `d_z=d_m in {1,2,4}` in the literal seed-table order below. Compare the production block path against a verification-only dense-PyTorch adapter and an independently authored NumPy dense oracle for solves, forward/backward substitution, log determinant, quadratic, same-noise samples, selected blocks, sparse trace, entropy, log normalizer, and complete objective. Both dense references hard-reject `T>8` or `K>4`; the dense-PyTorch adapter is import-forbidden from `vfe4/**` and used only in correctness preflight. Use operand-shaped budgets and literal residual decisions; `torch.allclose`, `numpy.allclose`, global tolerances, and post-observation threshold tuning are forbidden.
- In the H8 protocol, "backward" means the upper block-triangular backward-substitution half of a linear solve. H8 never calls autograd `backward()` and makes no gradient claim.
- Freeze seeds `20260721`, `20260722`, and `20260723`. Run five cold production subprocess repetitions per seed: exactly 15 eligible runs. Every run must complete in `<=60.0` seconds, incremental process HWM must be `<=128 MiB`, dispatch-observed live PyTorch population storage must be `<=64 MiB`, maximum RHS width must be `1` for sample and `<=40` otherwise, off-band fill and forbidden attempts must be zero, and every required operation must execute.
- Record all raw endpoints, not only maxima: per-run elapsed nanoseconds; pre-run current/lifetime-peak/private bytes; post-run current/lifetime-peak/private bytes; conservative incremental HWM and peak-to-peak diagnostic; dispatch live-storage peak; raw profiler allocation events and reconstructed live peak; backend counters; storage counts; workspace shapes; fill records; per-block/global pivot minima and margins; residual/budget records; platform/processor/CPU; Python/PyTorch/NumPy; affinity; thread settings; and BLAS identity. Canonicalize and hash separate hardware, affinity, thread, and BLAS identity records.
- After config and prerequisites have established a valid child start, any witnessed timeout, OOM/abnormal or nonzero child exit, forbidden allocation/operation, off-band fill, nonfinite contract value, solver inability, completed-run omission of a required operation, reported thread/environment identity mismatch, over-budget finite residual/resource endpoint, or executed negative-control miss dominates as `FAIL`, even if later evidence is unavailable. Only unavailable or unwitnessed evidence, missing observability/hashes, stale prerequisites, or a changed interpretation is `INCONCLUSIVE`; missing later evidence cannot mask a witnessed violation. `PASS` still requires all 15 production runs, all 12 correctness cells, every control/channel, every numerical/resource budget, and every required operation invariant.
- `validation/h8.json` is the normative in-artifact structured result. It binds exact source revision and dirty-content digest, canonical H8 config hash, prerequisite artifact/hash set, manuscript/preregistration hashes, child protocol and environment, raw run/correctness/allocation records, statuses/obligations, and explicit H8 nonclaims. It contains no hash of its own enclosing manifest. Only after atomic publication, write `.verification/h8-current-candidate-<FULL_HEAD>-result.json` with the artifact path plus candidate, manifest, config, validation, JUnit, reference-registry, and predecessor hashes; the H8 ledger must re-read and independently revalidate this external pointer.
- Preserve `.verification/ledger.json`, all four same-candidate predecessor/H7 ledgers, the separate H6-Prediction ledger, and every earlier ledger byte-for-byte. H8 uses only `.verification/h8-<FULL_HEAD>-<H8_CONFIG_SHA>-ledger.json`; an existing `.verification/active.json` blocks activation and is never deleted, overwritten, or repointed manually. Because activation is single-valued, close each lifecycle ledger in its own sequential verifier turn.
- Keep one editable `CONFIG` in `verify_vfe4.py`, one `main`, and one script guard. H8 is selected through that dictionary and package APIs; do not add a required CLI, alternate launcher, environment-only scientific setting, or hidden fallback.
- A post-H8 WikiText-103 training run is a separately approved milestone. It must re-audit the complete training, batching, checkpoint, evaluation, and decoder allocations; H8's synthetic certificate cannot be reused as training-memory evidence.

---

## Normative Sources and Read-Only Context

- H8 gate and limits: `Manuscripts/vfe4_whitepaper/08_hypotheses_limitations.tex:1-9`, `:120-137`, and `:167-175`.
- Sparse information interface and dense-covariance warning: `Manuscripts/vfe4_whitepaper/05_structured_information_form.tex:75-130` and `:372-425`.
- Population-block closure and exact local factor assembly: `Manuscripts/vfe4_whitepaper/09_appendices.tex:161-205` and `:280-347`.
- Repository seam, artifacts, status ladder, and test architecture: `docs/superpowers/specs/2026-07-21-vfe4-codebase-design.md:323-337`, `:666-751`, and `:753-814`.
- Predecessor implementation contracts: the H2 factor-backed vocabulary, H3 structured recognition, H4/H5 sparse/update plans, H6 Prefix/Prediction split and revision-specific ledgers, and the final H7 plan `docs/superpowers/plans/2026-07-21-vfe4-h7-frame-covariance.md` at canonical UTF-8/LF SHA-256 `3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242`, including its exact `H7GateResult`, pure current-candidate projectors, artifact/reference order, external result pointer, and revision-specific ledger. Compute the digest by strict UTF-8 decoding with no BOM, normalizing CRLF and lone CR to LF, re-encoding as UTF-8 with no BOM, and hashing those bytes. The preimage is the frozen H7 file alone, so checkout line-ending policy cannot change the pin and the H8 plan does not hash itself; any later H7 semantic-text edit invalidates every literal H8 pin and requires Task 1 to recompute and update all of them before any H8 calculation.
- Live baseline seams: `vfe4/types/information.py`, `vfe4/numerics/precision.py`, `vfe4/numerics/information.py`, `vfe4/numerics/linear_gaussian.py`, `vfe4/config/schema.py`, `vfe4/config/resolve.py`, `verification/run_gates.py`, `vfe4/artifacts/provenance.py`, and `verify_vfe4.py`. H8 adds a sibling block protocol and does not mutate the bounded dense H2 seam.
- Read-only Research wiki context: `[[VFE Transformer Program]]`, `[[Sparse Attention]]`, `[[Transformer interpretability and scaling]]`, and `[[Neural scaling laws]]`. These pages reinforce that the live V3 `K=20` configuration and long-context/sparse-attention literature do not convert this synthetic Gaussian memory certificate into a language, attention, or scaling-law result. H8 block-tridiagonal precision sparsity is not a sparse-attention claim.

## Frozen Mathematical and Systems Contract

### Layout and block algebra

`BlockChainLayout(T, d_z, d_m)` owns the only coordinate conversion. It exposes `N`, `b`, and `D`; `block_slice(t)` is allowed only in bounded oracle adapters and must never be used to create a global production tensor. Production tensors retain explicit block axes:

```text
h           : [N, b]
J_diag      : [N, b, b]
J_lower     : [N-1, b, b]   # J[t+1,t]
L_diag      : [N, b, b]
L_lower     : [N-1, b, b]   # L[t+1,t]
Sigma_diag  : [N, b, b]
Sigma_lower : [N-1, b, b]   # Sigma[t+1,t]
```

The backend computes

```text
L_diag[0]  = chol(J_diag[0])
L_lower[t] = solve_triangular(L_diag[t], J_lower[t].T, upper=False).T
S[t+1]     = J_diag[t+1] - L_lower[t] @ L_lower[t].T
L_diag[t+1]= chol(S[t+1])
```

using only local `[b,b]` workspaces. Forward and backward substitution operate on `[N,b]` or `[N,b,r]` with `1<=r<=b`. The sparse matvec used for residuals performs the three block contributions directly.

Selected inverse uses the block Takahashi/LDL-equivalent backward recurrence, not one global selector solve. Set `C_i=L_diag[i]`, `E_i=L_lower[i]`, `D_i=C_i@C_i.T`, and `F_i=E_i@C_i^{-1}`, computing `F_i` by a local triangular solve rather than an inverse. Initialize `Sigma_diag[N-1]=D_{N-1}^{-1}` by a local Cholesky solve. For `i=N-2..0`, compute exactly `Sigma_lower[i]=-Sigma_diag[i+1]@F_i` and `Sigma_diag[i]=D_i^{-1}+F_i.T@Sigma_diag[i+1]@F_i`. Store each diagonal block and lower adjacent block once; the upper block is `Sigma_lower[i].T` only at the point of use. The sparse trace is `sum_i trace(J_diag_left[i]@Sigma_diag[i]) + 2*sum_i sum(J_lower_left[i]*Sigma_lower[i])`. Validate symmetry by operand-budget comparison, not duplicate storage.

### Complete synthetic objective

For each seed, construct one deterministic normalized chain for observations `t=1..T`, retaining the manuscript's model-then-state within-slice order:

```text
m_0,z_0 ~ one normalized joint Gaussian
m_t | m_{t-1} ~ Normal(A_m[t] m_{t-1} + c_m[t], R_m[t])
z_t | z_{t-1},m_t ~ Normal(A_z[t] z_{t-1} + B[t] m_t + c_z[t], R_z[t])
x_t | y_t ~ Categorical(softmax(alpha * (w_t.T y_t) + beta_t)), V=3
```

Thus both named source supports are the singleton `t-1`; `m_t` is available to the state factor at the same receiver slice and no second historical parent is introduced. The recognition law is another normalized block-tridiagonal information Gaussian assembled from a strictly SPD block-local initial factor plus adjacent combined-slice linear-Gaussian factors. The problem generator is exactly `np.random.Generator(np.random.PCG64(problem_seed))`; its only random method is `standard_normal(size=shape, dtype=np.float64)`. Every call returns a C-contiguous float64 array, no call is batched across time, no draw is interleaved between the stages below, and no consumer may advance this generator. Here `Normal(0,1/K)` means **variance** `1/K`, so its standard-deviation multiplier is `1/sqrt(K)`; the implementation must not pass `1/K` as a `normal(..., scale=...)` argument. `contract(M,r)=r*M/max(r,np.linalg.norm(M,ord=2))`, and `spd(Q,n)=0.25*np.eye(n,dtype=np.float64)+0.05*(Q@Q.T)/n`. The literal loop nesting and call order is:

```python
rng = np.random.Generator(np.random.PCG64(problem_seed))
sn = lambda shape: np.ascontiguousarray(
    rng.standard_normal(size=shape, dtype=np.float64)
)

# 1. Generative initial law: exactly two calls.
initial_mean = 0.1 * sn((b,))
Q_initial = sn((b, b))
initial_cov = spd(Q_initial, b)

# 2. Generative transitions: exactly seven calls per t, in this order.
for t in range(1, N):
    A_m[t] = contract(sn((K, K)) / np.sqrt(K), 0.35)
    c_m[t] = 0.05 * sn((K,))
    Q_m = sn((K, K)); R_m[t] = spd(Q_m, K)
    A_z[t] = contract(sn((K, K)) / np.sqrt(K), 0.35)
    B[t] = contract(sn((K, K)) / np.sqrt(K), 0.20)
    c_z[t] = 0.05 * sn((K,))
    Q_z = sn((K, K)); R_z[t] = spd(Q_z, K)

# 3. Recognition initial law, then recognition transitions.
recognition_initial_mean = 0.1 * sn((b,))
Q_recognition_initial = sn((b, b))
recognition_initial_cov = spd(Q_recognition_initial, b)
for t in range(1, N):
    A_recognition[t] = contract(sn((b, b)) / np.sqrt(b), 0.35)
    c_recognition[t] = 0.05 * sn((b,))
    Q_recognition = sn((b, b))
    R_recognition[t] = spd(Q_recognition, b)

# 4. Emissions: alpha and x are deterministic; exactly two calls per t.
alpha = np.asarray((-0.5, 0.25, 0.75), dtype=np.float64)
for t in range(1, N):
    w[t] = sn((b,)) / np.sqrt(b)
    beta[t] = 0.1 * sn((V,))
    x[t] = (problem_seed + t) % V
```

The ordered draw-schema table is therefore `(initial_mean,Q_initial)`, then for `t=1..T` `(A_m,c_m,Q_m,A_z,B,c_z,Q_z)`, then `(recognition_initial_mean,Q_recognition_initial)`, then for `t=1..T` `(A_recognition,c_recognition,Q_recognition)`, then for `t=1..T` `(w,beta)`. Its canonical one-line ASCII descriptor is `numpy.Generator(numpy.PCG64(problem_seed))|method=standard_normal|dtype=float64|order=C|initial:mu0[b],Q0[b,b]|transition:t=1..T:{A_m[K,K],c_m[K],Q_m[K,K],A_z[K,K],B[K,K],c_z[K],Q_z[K,K]}|recognition_initial:mu_q0[b],Q_q0[b,b]|recognition_transition:t=1..T:{A_q[b,b],c_q[b],Q_q[b,b]}|emission:t=1..T:{w[b],beta[V]}|normal_map_variance=1/dim=>multiply_standard_normal_by_1/sqrt(dim)|serialize=after_all_problem_draws_before_sample_rng|bytes=little-endian-f8-C-contiguous`, whose SHA-256 is `7b657e72219f044147a7b414354d34c82bbd5a66d24f669285906d54534723c0`. After the final `beta[T]` draw, validate all shapes/finiteness/SPD properties, convert every array to C-contiguous little-endian `<f8`, serialize fields in precisely the table order (with deterministic `alpha` and integer `x` in the canonical metadata), and hash the immutable byte bundle. Serialization occurs once at that point, before the sample-noise generator is constructed; production and both correctness references parse those same bytes without regenerating or reopening them. These literals are generator schema `h8-synthetic-chain-v1`; any method, shape, dtype, loop, call order, vectorization, interleaving, transform, serialization, or schema-hash change invalidates H8 evidence.

The correctness grid uses this literal grid-order seed table; seeds are data, not a formula:

| Cell | `T` | `K=d_z=d_m` | `problem_seed` | `sample_noise_seed` |
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

Production problem seeds map literally to independent sample-noise seeds `{20260721:20261721, 20260722:20261722, 20260723:20261723}`. For both grids, finish and canonically serialize all problem draws before constructing a new `Generator(PCG64(sample_noise_seed))`; draw exactly `N*b` float64 `standard_normal` values in one call and C-order reshape to `[N,b]`. Reuse those serialized noise bytes across production/block, dense-PyTorch, and NumPy paths and across the five resource repetitions for a production seed. This is sample schema `h8-pcg64-sample-v1`; no consumer may advance either stream or derive noise from the problem generator.

For canonical assembly and sparse trace only, derive the exactly equivalent combined conditional in `[z_t,m_t]` order with `A_combined=[[A_z,B@A_m],[0,A_m]]`, `c_combined=[c_z+B@c_m,c_m]`, and `R_combined=[[R_z+B@R_m@B.T,B@R_m],[R_m@B.T,R_m]]`. Construct these as local `b x b` blocks and verify their factorwise density equality on the small grid; never assemble a global matrix. Objective reporting remains separated into the named model and state factors rather than collapsing them into one unlabeled term.

The objective is the complete `E_q[log p(y,x)] + H(q)`: separate exact initial, model-transition, and state-transition expectations from the mean and diagonal/adjacent covariance blocks; exact Gaussian entropy from `logdet J`; and the normalized categorical emission expectation reduced to the one-dimensional Gaussian marginal of `u_t=w_t.T y_t`. Evaluate that expectation with frozen Gauss--Hermite orders 21 and 17, record their difference as a deterministic convergence contribution, and use stable `log_softmax`. The production objective has no source-mixture term because every source support is a singleton; the artifact records state-source KL, model-source KL, and source entropy as exact zeros rather than omitted terms.

### Resource decision endpoints

Use binary mebibytes: `MiB=1,048,576` bytes. Parent wall time is `perf_counter_ns` around process creation through parsed result receipt; `elapsed_seconds=elapsed_ns/1e9` must be `<=60.0`. The primary conservative clean-child HWM is `max(0, post_run_lifetime_peak_bytes-pre_run_current_rss_bytes)` and must be `<=128 MiB`; `max(0,post_run_lifetime_peak_bytes-pre_run_lifetime_peak_bytes)` is recorded only as a peak-to-peak diagnostic and cannot rescue a primary failure. Record pre/post current RSS, lifetime peak, and private bytes separately. Dispatch live-storage peak is the maximum sum of unique live CPU storage byte spans observed through weak references; record alias/storage identities and do not double count views.

On Windows, freeze `PROCESS_MEMORY_COUNTERS_EX` with `_fields_` in this exact order and native `ctypes.wintypes.DWORD`/`ctypes.c_size_t` types: `cb`, `PageFaultCount`, `PeakWorkingSetSize`, `WorkingSetSize`, `QuotaPeakPagedPoolUsage`, `QuotaPagedPoolUsage`, `QuotaPeakNonPagedPoolUsage`, `QuotaNonPagedPoolUsage`, `PagefileUsage`, `PeakPagefileUsage`, `PrivateUsage`. Use native alignment and require `ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)==80` for a 64-bit pointer or `44` for a 32-bit pointer. Load `kernel32` and `psapi` with `use_last_error=True`; set `GetCurrentProcess.argtypes=[]`, `GetCurrentProcess.restype=wintypes.HANDLE`, `GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]`, and `GetProcessMemoryInfo.restype=wintypes.BOOL`. Set `cb=ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)` and clear last error before each call; invoke `GetProcessMemoryInfo(GetCurrentProcess(), byref(record), record.cb)`. A zero return, null handle, wrong `cb`, missing symbol, layout/size mismatch, negative field, or missing failure error code is an observability error. Record `WorkingSetSize` as current RSS, `PeakWorkingSetSize` as lifetime peak, and `PrivateUsage` as private bytes at both snapshots. Hash the adapter/layout/API identity. Linux/macOS adapters retain explicit unit conversion and the same named record where the platform exposes a field; an unavailable required field is not imputed.

The separate profiler is pinned to exactly `torch==2.9.1` and the official `v2.9.1` sources `torch/profiler/_memory_profiler.py` (SHA-256 `b80b4d5b58e91d581b18082c462ec7f088ec6b46ea50a1a62e2714d517a6a1b1`) and `torch/profiler/profiler.py` (SHA-256 `2c35f649219fb912728819b7dc0be5a5f1bd54c1efcd9502b62d976aeb278d22`). Invoke `torch.profiler.profile(record_shapes=True,profile_memory=True,with_stack=True)`, retain that live profiler object, and in the same process read the private pinned `profile._memory_profile().timeline` rows `(timestamp_ns, Action, (TensorKey,version), numbytes)` plus `profile.profiler.kineto_results.experimental_event_tree()`. Do not claim that the documented `export_memory_timeline(...raw.json.gz)` row `(timestamp_ns,action,numbytes,category)` contains identity, operator, stack, or shape; preserve those exported rows only as a lossy secondary cross-check.

Enrichment is one deterministic in-memory join over the already captured timeline and event tree. Index `_EventType.Allocation/_ExtraFields_Allocation` nodes and `_EventType.TorchOp/_ExtraFields_TorchOp` tensor inputs by the full `TensorKey(id,storage.ptr,allocation_id,device)`, then join timeline rows by that exact key and version; `CREATE`/`DESTROY` additionally require the unique allocation-event timestamp and signed byte-size witness, while `PREEXISTING` uses the first temporally ordered tensor-metadata occurrence and `INCREMENT_VERSION` uses the unique data-flow/TorchOp mutation at its timestamp. Derive logical shape/dtype from `_TensorMetadata`; derive operator and stack from the matched TorchOp and its event-tree ancestry. Hash source-row index, matched-node indices, key/version, and ancestry as the join witness. A missing, nonunique, or inconsistent identity/timestamp/bytes/version/operator/stack/shape join makes the profiler channel `INCONCLUSIVE`; no fallback to aggregate rows, nearest-name matching, or file reopen is permitted.

Reconstruct storage liveness in `(timestamp_ns,source_row_index)` order with all four actions. `PREEXISTING` establishes the pre-run live baseline and contributes its unique storage bytes exactly once; record `preexisting_storage_count`, `preexisting_bytes`, and `baseline_live_bytes`, and initialize the reconstructed peak from that baseline. `CREATE` requires a dead identity at version zero and adds its storage bytes. `INCREMENT_VERSION` requires the same live storage and the next version, changes only the active tensor version/category, and has zero live-byte delta. `DESTROY` requires the currently live identity/version and removes its storage bytes. Reject duplicate/bad-version creates, unknown increments/destroys, byte-size changes for one live storage, negative totals, or unterminated non-preexisting/create storage. Classify every live-establishing or version-changing row against the production whitelist using the enriched identity/operator/stack/shape, and report the baseline-inclusive peak. Aggregate positive/net CPU memory deltas remain secondary only. The canonical ASCII API descriptor is `torch==2.9.1|tag=v2.9.1|memory_profile_source_sha256=b80b4d5b58e91d581b18082c462ec7f088ec6b46ea50a1a62e2714d517a6a1b1|profiler_source_sha256=2c35f649219fb912728819b7dc0be5a5f1bd54c1efcd9502b62d976aeb278d22|flags=record_shapes:true,profile_memory:true,with_stack:true|timeline=profile._memory_profile().timeline:(timestamp_ns,action,key_and_version,numbytes)|actions=PREEXISTING,CREATE,INCREMENT_VERSION,DESTROY|event_tree=profile.profiler.kineto_results.experimental_event_tree()|allocation=_EventType.Allocation+_ExtraFields_Allocation|torchop=_EventType.TorchOp+_ExtraFields_TorchOp|join=TensorKey(id,storage.ptr,allocation_id,device)+version|raw_export=(timestamp_ns,action,numbytes,category)|join_unavailable=INCONCLUSIVE`, with frozen SHA-256 `161a78f04c26fba19bb01ba6417f2cf8c00730ebeb8d007a4af0f4da433ba043`; a version, source, symbol, tuple, action, join, or hash mismatch blocks the profiler channel as `INCONCLUSIVE` before it can support PASS.

The named production condition diagnostic is `HagerHigham1NormEstimate-v1`. Compute `||J||_1` exactly from block absolute column sums without a dense matrix. Estimate `||J^{-1}||_1` with block-shaped width-one vectors: initialize `x=ones([N,b])/D`; for at most eight iterations compute `y=solve(J,x)`, update the estimate with `sum(abs(y))`, set zero signs to `+1`, compute `z=solve(J,sign(y))` (the transpose solve is identical because `J` is symmetric), choose the first lexicographic maximizer of `abs(z)`, and stop on a repeated index or `abs(z[j])<=sum(z*x)`; otherwise set `x` to the corresponding block-shaped basis vector. Report `||J||_1*estimate` as `kappa_1_estimate`, together with iteration count, convergence reason, index/sign hashes, and solve residuals. The estimate is diagnostic-only metadata: no multiplier derived from it enters an allowance or status. Solver residuals use the fixed preregistered allowance below.

## File Map and Dependency Boundaries

| Path | Responsibility |
|---|---|
| `vfe4/types/h8.py` | Immutable auxiliary H8 layout/pattern/storage/fill/workspace/condition/counter/correctness/run/reference records and the runtime-checkable block factor protocol; no promotion gate result type. |
| `vfe4/types/results.py` | Add and export `H8GateResult` beside `H7GateResult`; preserve every earlier result type so the runner can append H8 to its closed field union. |
| `vfe4/numerics/block_layout.py` | Strict `BlockChainLayout`, canonical block IDs, shape predicates, and bounded oracle-only flatten/unflatten adapters. |
| `vfe4/numerics/block_tridiagonal.py` | Dependency-free factorization, block substitutions, solve/logdet/selected inverse/sample/quadratic/trace, sparse matvec, condition estimate, and counters. |
| `vfe4/numerics/block_canonical.py` | Block-local canonical accumulators and initial/transition/observation scatter; no global residual vector or outer product. |
| `vfe4/numerics/sparse_information.py` | `FactorBackedInformationGaussian` from `h+factor`, with mean/log-normalizer/entropy/log-prob/selected moments and no retained `J`. |
| `vfe4/generative/reference_h8.py` | Seeded normalized synthetic chain specification and immutable production/reference inputs. |
| `vfe4/recognition/reference_h8.py` | Block-local recognition information assembly and construction of the factor-backed Gaussian. |
| `vfe4/objective/h8_sparse.py` | Complete sparse synthetic ELBO terms, local Gaussian expectations, scalar emission quadrature, sparse trace use, and term records. |
| `vfe4/inference/__init__.py` | Explicitly export only the production H8 allocation interfaces; no verification reference import. |
| `vfe4/inference/h8_allocation.py` | Dispatch whitelist/shape/stack/live-storage tracer, forbidden-op classifier, NumPy guard, raw-event profiler parser, and negative-control functions. |
| `verification/numpy_oracles/h8_dense.py` | Independent NumPy parser, dense assembly, Cholesky solves, selected covariance, trace, entropy, log normalizer, and complete objective for the small grid only. |
| `verification/torch_references/__init__.py` | Private verification reference package surface; not re-exported by production. |
| `verification/torch_references/h8_dense.py` | Verification-only dense-PyTorch adapter/objective, hard bounded to `T<=8,K<=4`, used solely by correctness preflight and never imported by `vfe4/**`. |
| `verification/h8_budget.py` | Frozen operand-shaped rounding/solver/quadrature allowances, decisiveness checks, residual formulas, and resource boundaries. |
| `verification/h8_child.py` | Import-disciplined one-run child, platform HWM adapter, production exercise, profiler-only mode, control-only mode, and one JSON result on stdout. |
| `verification/h8_gate.py` | Prerequisite preflight, 12-cell correctness grid, 15 cold-child orchestration, allocation/control certificates, status precedence, exact `validation/h8.json`, and complete H8 publication payload-map construction. |
| `vfe4/config/schema.py` | Conditional immutable `H8ValidationConfig` and exact resource/protocol literals without changing earlier gate records. |
| `vfe4/config/resolve.py` | Ordered H1--H8 prefix validation, pure H7 candidate projection, current-reference registry binding, H8 interpretation checks, unknown-key rejection, and canonical hashing. |
| `verification/run_gates.py` | Extend H7's selected operation with H8, validate H7/H6-Prediction references, run H8 once, and pass the verification-built payload map to the existing generic atomic publisher. |
| `vfe4/artifacts/atomic.py` | Existing generic whole-directory atomic JSON publication only; it remains gate-agnostic and must not import `verification`. |
| `vfe4/artifacts/provenance.py` | Add processor, affinity, thread/BLAS, child protocol, prerequisite, manuscript, and observability hashes. |
| `verify_vfe4.py` | Extend the single editable `CONFIG` and click-run output through H8; no CLI. |
| `docs/preregistrations/2026-07-21-h8-sparse-scale.md` | Freeze interpretation, generator, algebra, budgets, controls/channels, environment, status mapping, schema, and nonclaims before execution. |
| `tests/unit/test_h8_layout.py` | Layout, shapes, storage arithmetic, block IDs, and forbidden dimensions. |
| `tests/unit/test_h8_block_backend.py` | Local factorization, substitutions, selected inverse, sparse trace, condition estimate, counters, and failure boundaries. |
| `tests/unit/test_h8_information_objective.py` | Factor-backed Gaussian, local scatter, normalized model, quadrature, and complete objective. |
| `tests/oracle/test_h8_numpy_oracle.py` | All 12 small-grid three-way operand-budget comparisons, dense-adapter bounds/independence, and wrong-path decisiveness. |
| `tests/unit/test_h8_allocation.py` | Dispatch/profiler/HWM parsers and every assigned negative control without materializing dangerous production-size buffers. |
| `tests/promotion/test_h8_gate.py` | Prerequisite, 15-run completeness, resource/allocation/status precedence, schema, stale-evidence, and nonclaim coverage. |
| `tests/integration/test_verify_vfe4.py` | H7 selected-operation path extended to reference current H7 plus separate `H6PredictionResult` evidence, return the explicit H8 result variant, and publish H8 once. |
| `tests/unit/test_structural_types.py` | Exact `VerificationRunResult` variant set through H8 and `H8GateResult` validation beside H7. |
| `tests/unit/test_config.py` | Ordered prefix, H8 conditional section, exact frozen values, hash round trip, and unknown/alternative interpretation rejection. |
| `tests/unit/test_atomic_artifacts.py` | Exact H8 artifact/provenance paths, manifest, raw child records, and prerequisite references. |

## Public Interfaces Frozen by This Plan

```python
@dataclass(frozen=True)
class BlockChainLayout:
    horizon: int
    d_z: int
    d_m: int
    @property
    def population_size(self) -> int: ...  # N
    @property
    def block_size(self) -> int: ...       # b
    @property
    def dimension(self) -> int: ...        # D

@dataclass(frozen=True)
class BlockTridiagonalPrecision:
    layout: BlockChainLayout
    diag: Tensor       # [N,b,b]
    lower: Tensor      # [N-1,b,b]

@dataclass(frozen=True)
class SelectedInverseBlocks:
    diag: Tensor       # [N,b,b]
    lower: Tensor      # [N-1,b,b]

@runtime_checkable
class BlockPrecisionFactor(Protocol):
    @property
    def dimension(self) -> int: ...
    @property
    def layout(self) -> BlockChainLayout: ...
    @property
    def pattern(self) -> BlockPatternRecord: ...
    @property
    def storage(self) -> BlockStorageRecord: ...
    @property
    def fill(self) -> BlockFillRecord: ...
    @property
    def workspace(self) -> BlockWorkspaceRecord: ...
    @property
    def diagnostics(self) -> SparseConditionDiagnostics: ...
    @property
    def counters(self) -> BackendCounterSnapshot: ...
    def solve(self, rhs: Tensor) -> Tensor: ...
    def solve_factor(self, rhs: Tensor, *, transpose: bool) -> Tensor: ...
    def logdet(self) -> Tensor: ...
    def selected_inverse(self, blocks: Sequence[BlockId]) -> SelectedInverseBlocks: ...
    def sample(self, noise: Tensor) -> Tensor: ...
    def quadratic(self, value: Tensor) -> Tensor: ...
    def trace_inverse_product(self, left: BlockTridiagonalPrecision) -> Tensor: ...

class BlockTridiagonalCholesky:
    @classmethod
    def factorize(cls, precision: BlockTridiagonalPrecision) -> "BlockTridiagonalCholesky": ...

class BlockCanonicalAssembler:
    @classmethod
    def zeros(cls, layout: BlockChainLayout, *, device: torch.device) -> "BlockCanonicalAssembler": ...
    def add_initial(self, *, mean: Tensor, covariance: Tensor) -> None: ...
    def add_transition(self, *, target: int, matrix: Tensor, offset: Tensor, covariance: Tensor) -> None: ...
    def add_local_information(self, *, target: int, h: Tensor, precision: Tensor) -> None: ...
    def freeze(self) -> tuple[Tensor, BlockTridiagonalPrecision]: ...

@dataclass(frozen=True, init=False)
class FactorBackedInformationGaussian:
    @classmethod
    def from_factor(cls, h: Tensor, factor: BlockPrecisionFactor) -> "FactorBackedInformationGaussian": ...
    def mean(self) -> Tensor: ...
    def log_normalizer(self) -> Tensor: ...
    def entropy(self) -> Tensor: ...
    def log_prob(self, value: Tensor) -> Tensor: ...
    def selected_moment_blocks(self) -> SelectedMomentBlocks: ...

@dataclass(frozen=True)
class H8ChildRequest:
    mode: Literal["production", "profiler", "negative_control"]
    seed: int
    repetition: int | None
    config_sha256: str
    protocol_sha256: str
    control_id: str | None

@dataclass(frozen=True)
class H8ChildResult:
    mode: Literal["production", "profiler", "negative_control"]
    seed: int
    repetition: int | None
    input_sha256: str
    objective: H8ObjectiveTerms | None
    storage: BlockStorageRecord | None
    fill: BlockFillRecord | None
    workspace: BlockWorkspaceRecord | None
    counters: BackendCounterSnapshot | None
    allocation: H8AllocationRecord
    resources: H8ResourceRecord
    invariants: tuple[H8InvariantRecord, ...]

# vfe4/types/results.py, implemented in Task 7 beside H7GateResult
@dataclass(frozen=True)
class H8GateResult:
    gate: Literal["H8"]
    status: GateStatus
    config_sha256: str
    candidate_junit_sha256: str
    current_refs_registry_sha256: str
    h7_manifest_sha256: str
    h6_prediction_manifest_sha256: str
    correctness: tuple[H8CorrectnessCell, ...]
    production_runs: tuple[H8ChildResult, ...]
    profiler_runs: tuple[H8ChildResult, ...]
    controls: tuple[H8ControlResult, ...]
    obligations: tuple[str, ...]

# verification/run_gates.py: append H8GateResult to the existing closed field
# union; do not replace it with a base class or unconstrained protocol.
VerificationRunResult.gate_results: tuple[
    GateResult
    | H3GateResult
    | H4GateResult
    | H5GateResult
    | H6PrefixGateResult
    | H6PredictionResult
    | H7GateResult
    | H8GateResult,
    ...,
]

@dataclass(frozen=True)
class H8H1H5Reference:
    kind: Literal["h1_h5"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

@dataclass(frozen=True)
class H8H1PrefixPriorReference:
    kind: Literal["h1_prefix_prior"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

@dataclass(frozen=True)
class H8H6PrefixSemanticFamilyReference:
    semantic_family_index: int
    semantic_family_sha256: str
    validation_payload_sha256: str
    certificate_sha256: str

@dataclass(frozen=True)
class H8H6PrefixReference:
    kind: Literal["h6_prefix"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    config_schema: Literal["h6-prefix-config-v3"]
    validation_schema: Literal["h6-prefix-validation-set-v2"]
    certificate_set_schema: Literal["h6-prefix-certificate-set-v2"]
    config_sha256: str
    workload_plan_sha256: str
    validation_payload_sha256: str
    prefix_certificate_set_sha256: str
    semantic_families: tuple[H8H6PrefixSemanticFamilyReference, ...]
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

@dataclass(frozen=True)
class H8H7Reference:
    kind: Literal["h7"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    result_pointer_path: str
    result_pointer_sha256: str
    fixture_set_sha256: str
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

@dataclass(frozen=True)
class H8H6PredictionReference:
    kind: Literal["h6_prediction"]
    prediction_schema: Literal["h6-prediction-amended-v2"]
    config_schema: Literal["h6-prediction-config-v2"]
    readiness_schema: Literal["h6-prediction-readiness-v2"]
    metrics_schema: Literal["h6-prediction-metrics-v2"]
    result_schema: Literal["h6-prediction-result-v2"]
    artifact_path: str
    manifest_sha256: str
    result_path: str
    result_sha256: str
    content_hashes: Mapping[str, str]
    payload_hashes: Mapping[str, str]
    experiment_sha256: str
    config_sha256: str
    readiness_artifact_path: str
    readiness_manifest_sha256: str
    readiness_sha256: str
    correctness_artifact_paths: Mapping[str, str]  # exact H1,H2,H3,H5 order
    h1_prefix_prior_artifact_path: str
    smc_accuracy_artifact_path: str
    smc_accuracy_manifest_sha256: str
    h6_prefix_artifact_path: str
    h6_prefix_manifest_sha256: str
    blinded_data_artifact_path: str
    blinded_data_manifest_sha256: str
    matching_artifact_path: str
    matching_manifest_sha256: str
    matching_set_sha256: str
    h1_prefix_prior_generative_factor_schema_sha256: str
    smc_bias_semantics_sha256: str
    objective_gate_spec_sha256: str
    metrics_sha256: str
    ledger_path: str
    ledger_sha256: str
    producer_head: str
    producer_dirty_digest: str
    candidate_junit_sha256: str
    status: Literal["pass"]

@dataclass(frozen=True)
class CurrentH8PrerequisiteRefs:
    candidate_head: str
    candidate_dirty_digest: str
    candidate_junit_sha256: str
    h7_compatibility_refs: Mapping[str, H7PredecessorReference]
    h1_h5: H8H1H5Reference
    h1_prefix_prior: H8H1PrefixPriorReference
    h6_prefix: H8H6PrefixReference | H8LegacyH6PrefixReference
    h7: H8H7Reference
    h6_prediction: H8H6PredictionReference | H8LegacyH6PredictionReference
    registry_sha256: str

    @property
    def prerequisite_obligations(self) -> tuple[str, ...]: ...

def build_h8_problem(seed: int, layout: BlockChainLayout) -> H8Problem: ...
def build_h8_recognition(problem: H8Problem) -> H8RecognitionBuild: ...
def evaluate_h8_objective(problem: H8Problem, recognition: FactorBackedInformationGaussian) -> H8ObjectiveTerms: ...
def h8_pair_allowance(left: OperandRecord, right: OperandRecord) -> AllowanceRecord: ...
def run_h8_child(request: H8ChildRequest) -> H8ChildResult: ...
def evaluate_h8(config: ResolvedConfig) -> H8GateEvaluation: ...
def h8_validation_payload(evaluation: H8GateEvaluation) -> dict[str, object]: ...
def project_h7_compatibility_config(
    raw_h8_config: Mapping[str, object],
    current_h7_refs: Mapping[str, H7PredecessorReference],
) -> Mapping[str, object]: ...
def bind_h8_current_refs(raw_h8_config: Mapping[str, object], refs: CurrentH8PrerequisiteRefs) -> ResolvedConfig: ...
def validate_h8_prerequisite_artifacts(refs: CurrentH8PrerequisiteRefs) -> H8PrerequisiteArtifactValidation: ...
def build_h8_publication_payloads(  # verification/h8_gate.py only
    config: ResolvedConfig,
    evaluation: H8GateEvaluation,
    *,
    h7_reference: H8H7Reference,
    h6_prediction_reference: H8H6PredictionReference,
) -> Mapping[str, object]: ...
def h8_current_candidate_result_payload(
    artifact: CandidateArtifactReference,
    *,
    config_sha256: str,
    validation_sha256: str,
    junit_sha256: str,
    current_refs: CurrentH8PrerequisiteRefs,
) -> Mapping[str, object]: ...
```

The implementation imports and reuses H7's exact frozen `H7PredecessorReference(artifact_path,git_head,dirty_digest,junit_sha256,junit_path,manifest_sha256,payload_hashes,ledger_path,ledger_sha256,reference_sha256)` and H7-owned `project_h1_h5_compatibility_config`, plus H6's exact frozen `CandidateArtifactReference`, amended `project_h1_prefix_prior_v2_config`, `project_h6_prefix_config`, keyword-only `run_projected_current_candidate`, and `reopen_bounded_prefix_certificate_set`. H6-Prefix calls the one-argument projector and producer with `predecessor_refs={}`; any wrapper that restores the old predecessor argument or any nonempty Prefix predecessor mapping is rejected. `project_h7_compatibility_config` consumes `Mapping[str,H7PredecessorReference]` exactly: the ordered H7 registry is captured once as bytes, deserialized once directly to those records, and supplied without an H8 wrapper or field projection. Its output must preserve every predecessor `reference_sha256`, `junit_path`, `junit_sha256`, and complete keyed `payload_hashes` mapping byte-for-byte while removing H8 fields; tests require its canonical bytes to equal the pinned final-H7 schema and contain no H8 or H6-Prediction key. Separately, the tagged H8 reference variants retain complete keyed `content_hashes` and `payload_hashes`, candidate JUnit, and kind-specific certificate/result-pointer/experiment hashes. Every `content_hashes` key is an exact manifest-relative payload path and is independently rehashed after manifest validation. H6-Prefix carries config v3, workload-plan v2, validation-set v2, and certificate-set v2 aggregate identities plus the exact ordered semantic-family rows `(index,family,validation,certificate)`; a legacy `PrefixCaseKey` certificate map is nonauthorizing. H8 must not reopen an artifact to reconstruct a registry reference, replace the ordered family bindings with one aggregate hash, weaken a predecessor schema, or copy predecessor output bytes. It must reopen the exact frozen artifact, result, ledger, H7 pointer, all three H7 transitive JUnit preimages, and H6-Prediction readiness/prerequisite/raw/metrics/result paths to rehash and validate the already parsed records fail-closed. The H6 bounded reopener independently requires the exact five-file inventory, outer config-v3 authorization, internal workload-v2 authorization, validation-set v2, certificate-set v2, and typed ordered family set before H8 compares those identities to the registry. H6-Prediction preflight reconstructs its typed readiness from the exact H1/H2/H3/H5, scorer-v2, finite-SMC, independent Prefix, blinded-data, and matching artifacts; the blinded data identity supplies the actual held-out token count used by raw SMC validation. A naked matching-set or data-identity digest is not closure. `verification/h8_gate.py` owns H8 reference validation and payload construction; `verification/run_gates.py` hands that mapping to `vfe4.artifacts.publish_run_directory`, then validates the published path into a `CandidateArtifactReference` before constructing the external result pointer. No module under `vfe4/**` imports `verification/**`.

H7 fixture closure is independently rederived from the current-candidate
`h1_v1.json`, `h7_v1.json`, and `h7_density_probes_v1.json` bytes. Native H7
parsing validates the typed density-probe set, and both the reopened H7
validation and H8 reference must equal the resulting exact four fixture hashes
and fixture-set identity.

The sole authorizing registry is `h8-current-candidate-refs-v3`. Its direct
H1--H5, H1-prefix-prior, and H6-Prefix variants equal the H7 transitive
references field-for-field. Its H6-Prediction variant retains its own frozen
producer head, dirty digest, and non-null JUnit hash, which need not equal the
H8 candidate; the variant also binds and revalidates the amended
config/readiness/scientific-prerequisite/matching/scorer-v2/SMC-bias/OBJECTIVE/raw/metrics/result
identities above. Missing prerequisite paths or manifest preimages reject
registry v3 rather than creating a self-consistent authorization. Registry v1
and v2 decode their legacy H6-Prefix shapes only for diagnosis and contribute
`h8_prerequisite_legacy_registry_requires_bounded_h6_prefix_v3`. Registry v1
also decodes `H8LegacyH6PredictionReference` and contributes
`h8_prerequisite_registry_v1_requires_amended_h6_prediction_v2`; registry v2
retains amended H6-Prediction v2 unchanged.
Any legacy or unavailable/rehashed-different prerequisite remains
`INCONCLUSIVE` and is never eligible for H8 `PASS`. The H7 plan pin remains
`3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242`
because this amendment changes no H7 semantic text.

`BlockTridiagonalPrecision` and the selected blocks clone and own only their explicit block tensors. `BlockTridiagonalCholesky.factorize` does not retain the input precision. `FactorBackedInformationGaussian` clones `h`, retains the factor, has no `J`/`covariance`/`moment_matrix` property, and rejects serialization requests for them. Existing bounded `DenseCholeskyPrecision` and `InformationGaussian` remain available to H1--H7 and the small dense reference; H8 production factories accept only `BlockPrecisionFactor`.

## Frozen Numerical Budget

Create `verification/h8_budget.py` with literal constants:

```python
H8_EPS = float(np.finfo(np.float64).eps)
H8_ROUNDING_MULTIPLIER = 4096
H8_SOLVER_RELATIVE_BUDGET = 1e-9
H8_MAXIMUM_ALLOWANCE_SCALE_FRACTION = 1e-4
H8_MAX_SECONDS = 60.0
H8_MAX_PROCESS_INCREMENTAL_BYTES = 128 * 1024 * 1024
H8_MAX_TORCH_POPULATION_BYTES = 64 * 1024 * 1024
H8_MAX_STORAGE_SCALARS = 411_200
H8_MIN_CHOLESKY_PIVOT = 1e-8
```

`gamma(n)=n*eps/(1-n*eps)` rejects bools, `n<=0`, and `n*eps>=1`. Every operand record names its shape, scalar count, actual infinity norm, actual absolute-sum bound, local operation count, source (`block`, `dense_torch`, or `numpy`), condition provenance, and whether one solver contribution applies. Its allowance is

```text
4096 * gamma(operation_count) * max(1, absolute_sum) +
(1e-9 * max(1, infinity_norm) if solver_produced else 0).
```

A comparison adds the two operand allowances plus one `4096*gamma(compared_scalar_count+1)*scale` reduction term, where `scale=max(1,left_inf,right_inf)`. The comparison is decisive only when `allowance/scale < 1e-4`; equality is `INCONCLUSIVE`. A residual passes when its literal infinity norm is `<=allowance`; equality passes. A finite decisive residual above allowance is `FAIL`. The solver term is the fixed literal above and is independent of the Hager--Higham diagnostic. Quadrature adds the absolute order-21/order-17 difference exactly once to the emission operand; it is never pooled into unrelated factor, solve, or memory checks. Every factorization records `per_block_min_pivots`, `global_min_pivot`, `per_block_pivot_margins = pivot-H8_MIN_CHOLESKY_PIVOT`, and `global_pivot_margin`; a finite negative margin is `FAIL`. No function accepts a global maximum condition number or a single tolerance for multiple invariants.

For small-grid dense operands only, exact local/dense `kappa_2` from `eigvalsh`/SVD may enter that operand's record. At production scale, use the Hager--Higham `kappa_1` estimate record only as diagnostic metadata; label it `estimate`, record iterations/convergence/sign-vector hash, never use it to set a budget or status, and never write it into `lambda_min`, `lambda_max`, or exact `kappa_2` fields.

---

### Task 1: Freeze H8 Types, Interpretation, Configuration, and Preregistration

**Files:**
- Create: `vfe4/types/h8.py`
- Create: `vfe4/numerics/block_layout.py`
- Create: `docs/preregistrations/2026-07-21-h8-sparse-scale.md`
- Modify: `vfe4/types/__init__.py`
- Modify: `vfe4/config/schema.py`
- Modify: `vfe4/config/resolve.py`
- Modify: `tests/unit/test_config.py`
- Create: `tests/unit/test_h8_layout.py`

**Interfaces:**
- Consumes: existing `GateStatus`, canonical JSON/hash helpers, the final H7 plan at canonical UTF-8/LF SHA-256 `3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242`, H7's selected-operation artifact/lifecycle interfaces and revision-specific ledger identity, and H6's separate `H6PredictionResult` plus `.verification/h6-prediction-<FULL_HEAD>-<EXPERIMENT_SHA>-ledger.json`.
- Produces: every auxiliary immutable H8 record and literal in the public-interface section except `H8GateResult`, `H8ValidationConfig`, strict layout/storage arithmetic, and the dated preregistration used by all later tasks. Task 7 owns `H8GateResult` in `vfe4/types/results.py`.

- [ ] **Step 1: Write RED layout/type tests.** Assert `T=128,d_z=d_m=20 -> N=129,b=40,D=5160`; exact coordinate order; exact `206_400` diagonal, `204_800` lower, `411_200` precision/factor/selected, `5_160` information-vector, and `26_625_600` dense scalar counts; canonical diagonal/adjacent block order; immutable records; no upper-block storage; and rejection of bools, zero/negative dimensions, non-chain patterns, wrong shapes/dtypes/devices, duplicate block IDs, and any scalar count over the frozen category cap.

- [ ] **Step 2: Run only the new layout tests and confirm RED.**

  Run: `python -m pytest tests/unit/test_h8_layout.py -q`

  Expected: FAIL because H8 layout/types do not exist. Do not run the suite.

- [ ] **Step 3: Implement strict layout and immutable auxiliary H8 records.** `BlockChainLayout` computes all dimensions rather than accepting redundant values. `BlockPatternRecord` fixes offsets `(-1,0,1)` for precision and `(-1,0)` for factor storage. Define explicit records for correctness cells, allocation observations, child endpoints, prerequisites, objective terms, allowances, and gate evaluation in `vfe4/types/h8.py`, but do not place `H8GateResult` there. Validate finite numeric fields, exact tuple lengths/orders, unique seed/repetition IDs, and status/obligation consistency in `__post_init__`.

- [ ] **Step 4: Write RED conditional-config and lifecycle-consumer tests.** Preserve every accepted shorter prefix, including H7's exact `("H1","H2","H3","H4","H5","H6-Prefix","H7")`, and add only `("H1","H2","H3","H4","H5","H6-Prefix","H7","H8")` for selected operation `H8`. H8 requires its section and exact values `T=128,K=20,d_z=20,d_m=20,V=3,seeds=[20260721,20260722,20260723],cold_repetitions=5,max_seconds=60.0,max_process_incremental_mib=128,max_torch_population_mib=64,max_rhs_width=40,sample_width=1`, `torch_version="2.9.1"`, the two pinned profiler-source hashes and API-contract hash, the literal ordered problem-draw descriptor/hash, the literal 12-cell problem/noise table, the three production noise seeds, and the exact H7-plan SHA. Freeze H6's one-argument `project_h6_prefix_config(CONFIG)`, keyword-only `run_projected_current_candidate`, `CandidateArtifactReference`, and mandatory `predecessor_refs={}` for H6-Prefix; reject the old projector signature, positional runner calls, nonempty Prefix predecessors, and Prefix records containing H1--H5/H4 identity. Freeze the separate H6-Prediction prerequisite-key set as exactly H1/H2/H3/H5 plus its H1-prefix-prior/finite-SMC/Prefix scientific inputs, with no H4 or H4-timing key. The committed mapping names deterministic `.verification/h8-current-candidate-<FULL_HEAD>-refs.json` plus bootstrap H7/H6-Prediction references. Test that the H7 registry decodes once to the exact `Mapping[str,H7PredecessorReference]`, with complete predecessor `junit_sha256` and keyed `payload_hashes`, and that the separate H8 tagged references preserve every keyed content/payload/certificate/result-pointer/experiment hash and candidate JUnit without aggregation. Reject absent/wrong-HEAD/wrong-digest/changed registry bytes, lossy singular replacement hashes, a registry reread or reference reconstructed from reopened bytes, `K=20,d_z+d_m=20`, `K=40`, a non-chain parent, H8 without H7, H6-Prediction inserted into the verifier tuple, reordered/duplicate prefixes, unknown keys, hidden CLI overrides, and earlier prefixes that carry an H8 section. Require one fail-closed reopen-and-rehash pass over every exact immutable predecessor artifact path after registry parsing.

- [ ] **Step 5: Implement conditional H8 config resolution, pure H7 projection, current-reference binding, and canonical hashing.** Preserve earlier frozen dataclasses; preserve H7's H1--H5/H7 projection interfaces and H6's exact two projectors/keyword-only producer unchanged. Capture the H7 registry bytes once, hash them once, and deserialize them directly to the exact ordered `Mapping[str,H7PredecessorReference]`; pass that same in-memory mapping to `project_h7_compatibility_config(CONFIG,current_h7_refs)`. The nonmutating pinned-schema projection strips H8 fields, binds only current H1--H5/active-prefix-prior/independently produced H6-Prefix references, preserves each H7 reference's JUnit and full `payload_hashes` mapping byte-for-byte, and byte-matches final H7 canonical config; it contains no H6-Prediction key. H6-Prefix carries no H1--H5/H4 predecessor. From the same already validated in-memory source records, construct the separately discriminated H8 references without reopening any registry/artifact or reducing keyed content/payload/certificate hashes to one digest. `bind_h8_current_refs` accepts only the deterministic current-H8 registry whose H7 chain is same-candidate, validates the exact ordered current H1--H5/active-prefix-prior/H6-Prefix/H7 chain plus the separately eligible H6-Prediction result/ledger, and includes every content/payload hash and applicable candidate JUnit in resolved H8 canonical JSON. The H6-Prediction reference must bind exact H1/H2/H3/H5 scientific prerequisite identities and reject H4/H4-timing provenance. H8 resolution verifies all redundant arithmetic, exact Torch/profiler/draw-schema hashes, the operational `interpretation_sha256`, and the pinned final-H7 plan SHA and fails closed on an alternative `K` meaning. Before any H8 calculation, recompute the H7 digest by the strict UTF-8/LF canonicalization above and compare it to the literal; H8 never includes its own bytes in this preimage. Artifact identities are references, not copied PASS booleans or payloads.

- [ ] **Step 6: Author and freeze the preregistration before any H8 calculation.** Copy the global constraints, operational (not theorem-level) `K=d_z=d_m=20` choice and `interpretation_sha256`, exact source/H7-plan hashes, exact `torch==2.9.1` profiler-source/API hashes, literal problem RNG pseudocode plus ordered draw-schema descriptor/hash and serialization point, literal 12-cell problem/noise seed table, exact storage arithmetic, algorithms, budget constants/formulas, 15-run traversal `(seed order, then repetition 0..4)`, four observability channels, the four-action profiler liveness/join contract, every exact assigned negative control, status precedence, exact discriminated reference/external-result schemas, full payload schema, and nonclaims. Include the block selected-inverse derivation and the reason sparse precision does not imply sparse covariance.

- [ ] **Step 7: Run focused GREEN tests.**

  Run: `python -m pytest tests/unit/test_h8_layout.py tests/unit/test_config.py -q`

  Expected: PASS. This is a focused compatibility file, not milestone evidence.

- [ ] **Step 8: Review and commit Task 1.** A fresh reviewer checks interpretation ambiguity, arithmetic, conditional config isolation, exact source paths/case, and absence of placeholders. Commit only Task 1 files with `feat(h8): freeze sparse scale contract`.

### Task 2: Implement the Block-Tridiagonal Factor Backend

**Files:**
- Create: `vfe4/numerics/block_tridiagonal.py`
- Modify: `vfe4/numerics/__init__.py`
- Create: `tests/unit/test_h8_block_backend.py`

**Interfaces:**
- Consumes: `BlockChainLayout`, `BlockTridiagonalPrecision`, selected-block IDs, and H8 diagnostic records from Task 1.
- Produces: `BlockTridiagonalCholesky` implementing `BlockPrecisionFactor`, sparse matvec, selected inverse, Hager--Higham condition estimate, and exact backend counters.

- [ ] **Step 1: Write RED factorization and substitution tests.** On hand-computable block chains, compare each `L_diag/L_lower` block and reconstruction; exercise vector and RHS widths `1`, `b-1`, and `b`; separately reach forward and backward substitutions; reject width `b+1`, flattened full-width `(D,D)`, wrong dtype/device, nonfinite inputs, nonsymmetry, off-band declarations, nonpositive pivots, and aliasing of input tensors. Assert the factor owns exactly `diag+lower` storage and does not retain precision tensors.

- [ ] **Step 2: Run the new backend test file and confirm RED.**

  Run: `python -m pytest tests/unit/test_h8_block_backend.py -q`

  Expected: FAIL because the backend is absent.

- [ ] **Step 3: Implement local factorization and substitutions.** Use `torch.linalg.cholesky_ex` on each Schur block, local triangular solves, checked finite pivots, and no jitter/repair/fallback. Clone only `L_diag/L_lower`. `solve_factor` accepts `[N,b]` or `[N,b,r]`; `solve` calls forward then backward; every call increments operation/shape/RHS counters before execution. Sparse matvec uses only neighboring blocks.

- [ ] **Step 4: Write RED selected-inverse, sample, quadratic, and trace tests.** Require all and only diagonal plus lower-adjacent blocks, canonical order, transpose symmetry, local positive diagonal blocks, width-one sampling with exact supplied noise, quadratic equality, and `trace(left @ self^{-1})` from selected blocks. Reject requests for any off-pattern block, all-pairs materialization, noise width other than one, incompatible layouts, and a dense left operator.

- [ ] **Step 5: Implement the Takahashi recurrence and remaining vocabulary.** Use local `b x b` identities only. `sample` computes `L^{-T} noise`; `quadratic` computes `||L^T value||^2` by block operations; sparse trace contracts `left.diag` with `Sigma_diag` plus both adjacent orientations from one stored `Sigma_lower`. `logdet` sums local diagonal logs. Return immutable clones/views without upper duplication.

- [ ] **Step 6: Write and implement the named condition-estimator tests.** Implement `HagerHigham1NormEstimate-v1` exactly as frozen above: width-one all-positive start, at most eight iterations, zero sign mapped to `+1`, lexicographically first maximizer, repeated-index/dot-product stops, and exact sparse `||J||_1`. Tests record its declared `kappa_1_estimate` beside an independently computed small dense diagnostic, verify deterministic index/sign hashes, assert perturbing/removing the estimate cannot change any allowance or gate status, and assert no field is named `lambda_min`, `lambda_max`, or exact `kappa_2`. Monkeypatch global `eigvalsh/eigvals/svd` to fail while allowing local `<=b` checks.

- [ ] **Step 7: Add local fill/workspace/counter/pivot assertions.** Every backend operation records maximum local shape, scalar count, RHS width, selected-block count, factor/solve/sample/trace call counts, attempted forbidden widths, every local Cholesky minimum pivot, the global minimum, and per-block/global margins relative to `1e-8`. `fill.observed_offband_blocks` is constructionally zero and must be checked, not hard-coded without relation to the factor storage.

- [ ] **Step 8: Run focused GREEN tests.**

  Run: `python -m pytest tests/unit/test_h8_block_backend.py -q`

  Expected: PASS with no global dense diagnostic reached.

- [ ] **Step 9: Review and commit Task 2.** Reviewer checks recurrences against the manuscript/local derivation, input non-retention, counter placement, local-only diagnostics, and RHS bounds. Commit `feat(h8): add block tridiagonal factor`.

### Task 3: Add Block-Local Assembly, Factor-Backed Gaussian, and Complete Objective

**Files:**
- Create: `vfe4/numerics/block_canonical.py`
- Create: `vfe4/numerics/sparse_information.py`
- Create: `vfe4/generative/reference_h8.py`
- Create: `vfe4/recognition/reference_h8.py`
- Create: `vfe4/objective/h8_sparse.py`
- Modify: package `__init__.py` files for explicit exports
- Create: `tests/unit/test_h8_information_objective.py`

**Interfaces:**
- Consumes: Task 2 factor and Task 1 problem/objective records.
- Produces: block-local canonical assembly, `FactorBackedInformationGaussian`, deterministic synthetic problem/recognition builders, and complete sparse H8 objective.

- [ ] **Step 1: Write RED canonical-scatter tests.** Check the manuscript linear-Gaussian contributions for an initial factor, each adjacent transition, and local observation information. Instrument `torch.outer`, `torch.zeros`, `torch.empty`, `torch.eye`, `torch.block_diag`, and concatenation/stacking to reject length `D`, `(D,D)`, `(N,N,b,b)`, or pair-slab requests. Assert only the target/parent diagonal blocks, one lower block, and corresponding `h` blocks change.

- [ ] **Step 2: Implement `BlockCanonicalAssembler`.** Accumulate in float64 `[N,b]`, `[N,b,b]`, `[N-1,b,b]`; convert local covariances to precisions with local Cholesky solves; scatter `A.T@P@A`, `-P@A`, `P`, `-A.T@P@c`, and `P@c` directly. `freeze()` validates symmetry/finiteness, returns owned tensors, and invalidates further mutation.

- [ ] **Step 3: Write RED factor-backed Gaussian tests.** Require construction from `h+BlockPrecisionFactor`; mean shape `[N,b]`; finite log normalizer/entropy/log probability; selected moments built blockwise; and no `J`, `Sigma`, covariance, full moment, flatten, or dense serialization attribute. Prove mutation of source `h`/factor inputs cannot change the object.

- [ ] **Step 4: Implement `FactorBackedInformationGaussian`.** Use block inner products, factor operations, and selected local mean outer products. Never concatenate mean into length `D`; compute `h^T mean` as a sum of block dot products. `log_prob` and quadratic accept block-shaped values only.

- [ ] **Step 5: Write RED deterministic problem/model tests.** Assert chain/singleton sources, `V=3`, normalized local Gaussian factors, normalized `log_softmax` emissions, seed determinism/difference, local SPD generation, frozen norm ceiling, byte-identical serialized inputs across consumers, and rejection of any source support wider than one.

- [ ] **Step 6: Implement the seeded H8 model and recognition builder.** Keep generator use in one module; return immutable float64 records rather than RNG-bearing objects. Build recognition locally and discard precision after factorization. Record input/factor scalar counts and hashes.

- [ ] **Step 7: Write RED complete-objective tests.** Hand-check initial, transition, categorical emission, entropy, zero source term, total, `log_normalizer`, sparse trace, order-21/order-17 convergence, and stable `log_softmax`. Assert every expected factor ID appears exactly once and total is assembled in one function. Patch dense inverse/covariance APIs to fail.

- [ ] **Step 8: Implement the complete objective.** Compute Gaussian expectations from selected diagonal/adjacent moment blocks and local factor constants. Reduce each categorical likelihood to `u_t=w_t.T y_t`, using its local mean/variance and fixed Gauss--Hermite nodes. Return both quadrature orders, term-shaped absolute sums, and the complete order-21 objective; no gradient graph is created.

- [ ] **Step 9: Run focused GREEN tests.**

  Run: `python -m pytest tests/unit/test_h8_information_objective.py -q`

  Expected: PASS.

- [ ] **Step 10: Review and commit Task 3.** Reviewer traces every normalized factor to `Manuscripts/vfe4_whitepaper/05_structured_information_form.tex` and `09_appendices.tex`, checks source singleton accounting, and searches the H8 production imports for dense helpers. Commit `feat(h8): add sparse synthetic objective`.

### Task 4: Build the Independent NumPy Oracle and 12-Cell Correctness Grid

**Files:**
- Create: `verification/numpy_oracles/h8_dense.py`
- Create: `verification/torch_references/__init__.py`
- Create: `verification/torch_references/h8_dense.py`
- Create: `verification/h8_budget.py`
- Create: `tests/oracle/test_h8_numpy_oracle.py`

**Interfaces:**
- Consumes: serialized `H8Problem`, production block APIs, the bounded dense PyTorch `DenseCholeskyPrecision`/`InformationGaussian` primitives only through a new verification adapter, and Task 1 correctness records.
- Produces: independent dense NumPy results, a verification-only bounded dense-PyTorch assembly/objective adapter, frozen operand allowances, 12 complete correctness-cell records, and decisive wrong-path controls.

- [ ] **Step 1: Write RED budget boundary tests.** Test exact `gamma`, operand, pair, reduction, solve-residual, reconstruction-residual, trace, entropy/log-normalizer, and quadrature formulas with unequal shapes/scales/conditions. Cover allowance ratio immediately below, exactly at, and above `1e-4`; residual immediately below, equal to, and above its allowance; solver contribution zero/once/duplicate; and rejection of global kappa, negative/nonfinite/bool values, mismatched scalar counts, or an allowance without named operands.

- [ ] **Step 2: Implement literal operand-shaped budgets.** Keep formula code free of model imports. Every result stores operands, operation counts, absolute sums, scale, rounding/solver/quadrature components, final allowance, residual, ratio, and decisive flag. No convenience `tolerance` argument exists.

- [ ] **Step 3: Write RED bounded-reference and import-boundary tests.** Monkeypatch production assembly/factor/objective helpers to raise while the NumPy oracle runs. Require the NumPy oracle to parse serialized problem records independently, assemble dense `J/h`, use NumPy Cholesky/solves, and compute all named outputs. Require `verification/torch_references/h8_dense.py` to parse the same immutable input, build its dense PyTorch precision/information Gaussian/objective only inside verification, and return the same named raw endpoints. Both references hard-reject `T>8`, `K>4`, unequal `d_z/d_m`, or a production layout before allocating. Walk/import-check every `vfe4/**/*.py` module and fail if it imports `verification`, `verification.torch_references`, or either H8 dense reference.

- [ ] **Step 4: Implement both bounded references.** The NumPy oracle does not import `vfe4.numerics`, production quadrature helpers, or PyTorch; re-derive dense factor assembly and objective formulas from the manuscript and allow dense covariance only inside that bounded file. The dense-PyTorch adapter may reuse bounded dense primitives but owns a separately coded H8 dense assembler and complete normalized objective; it must not call block production assembly/objective helpers. Each module validates `T<=8,K<=4` before any shape calculation or allocation and returns raw arrays/scalars plus operand metadata. Neither module is exported from or importable through `vfe4` package initializers.

- [ ] **Step 5: Write the 12-cell three-way RED matrix.** Traverse the literal table exactly in rows 1--12; do not compute seeds from dimensions. For each `(T,K,problem_seed,sample_noise_seed)`, finish and serialize the `PCG64(problem_seed)` problem stream, then independently draw exactly `N*b` float64 values from `Generator(PCG64(sample_noise_seed)).standard_normal` and C-order reshape once. Compare production block, bounded dense PyTorch, and independent NumPy for factor reconstruction, forward substitution, backward substitution, solve, logdet, quadratic, byte-identical supplied-noise sample, every selected block, sparse trace, entropy, log normalizer, each objective term, and total. Compare all three ordered pairs with operand records; do not call `allclose`.

- [ ] **Step 6: Add decisive wrong-path controls.** Perturb one solve element, reverse logdet sign, transpose one adjacent covariance block, duplicate the off-diagonal trace contribution, drop entropy, and substitute independent sample noise. Each finite perturbation must exceed its own allowance or the cell is `INCONCLUSIVE`; when decisive it must be detected as `FAIL`.

- [ ] **Step 7: Run focused GREEN oracle tests.**

  Run: `python -m pytest tests/oracle/test_h8_numpy_oracle.py -q`

  Expected: PASS for exactly 12 cells and every named comparison.

- [ ] **Step 8: Review and commit Task 4.** Reviewer checks NumPy independence, dense-PyTorch boundedness and objective completeness, the production import boundary, literal seed/noise streams and draw order, pair completeness, no `allclose`, operand-local conditioning, and wrong-path decisiveness. Commit `test(h8): add bounded sparse correctness references`.

### Task 5: Implement Allocation Observability and Assigned Negative Controls

**Files:**
- Create: `vfe4/inference/__init__.py`
- Create: `vfe4/inference/h8_allocation.py`
- Create: `tests/unit/test_h8_allocation.py`

**Interfaces:**
- Consumes: Task 1 layout/records and Task 2 backend counters.
- Produces: `H8DispatchTrace`, raw-event `H8ProfilerTrace`, `H8NumpyAllocationGuard`, the exact production allocation whitelist/classifiers, and isolated negative-control results.

- [ ] **Step 1: Write RED production-whitelist tests at safe dimensions.** Parameterize a logical layout so detectors use small physical/meta operands while applying production `N,b,D` semantics. The allowlist is site-registered scalar/local/channel arrays; `[N,b]`, `[N,b,1]`, `[N,b,r]` for `1<=r<=b`; `[N,b,b]`; `[N-1,b,b]`; and exact generator/objective arrays with one `T`/`N`/`N-1` population axis and all other axes `<=b` or `V`. Require rejection of any axis `D`, any single storage over `411_200` float64-equivalent scalars, any two population/pair axes, unregistered reshape/view/stack output, `(D,D)`, flat `D^2`, near-quadratic `(D-1,D-1)`, length `D`, `(N,N,b,b)`, triangular `(N*(N+1)//2,b,b)`, `(D,D)` RHS, and an equivalent combined pair slab. A view is accepted only when its registered base storage and semantic site are allowed.

- [ ] **Step 2: Implement the dispatch whitelist and live-storage accounting.** Inspect factory/reshape/view/outer/matmul/stack/concatenate requests before allocation where arguments expose logical output shape, then inspect outputs. Record registered semantic site, operator, input/output shapes, stack member shapes/count, dtype/device, float64-equivalent scalar count, storage pointer/span, and forbidden reason. Maintain weak references and sum unique live storage spans after every dispatch event; record peak and alias evidence. Any shape not on the production whitelist fails closed. A forbidden request raises `H8ForbiddenAllocation` only after recording it.

- [ ] **Step 3: Write RED operation-classifier tests.** Detect dense-population `torch.linalg.eigvalsh/eigh/svd/cholesky/inv` operands, `torch.eye(D)`, solve RHS width `D`, global selector patterns, and off-band backend attempts. Allow local eigensolvers only when every matrix axis is `<=b`.

- [ ] **Step 4: Implement backend/dispatch cross-checks.** Reconcile factor/input/selected scalar counts, maximum RHS width, sample width, workspace maxima, registered allocation sites, and operation counts. A counter absent from dispatch when it should allocate, an unregistered dispatch result, or any observed profiler `PREEXISTING`/`CREATE`/`INCREMENT_VERSION`/`DESTROY` row lacking its full TensorKey+version/source-row/dtype/operator/stack/shape/join witness or valid liveness transition is an observability gap (`INCONCLUSIVE`), not proof of absence.

- [ ] **Step 5: Write and implement the NumPy guard.** Wrap only inside H8 child/control contexts and intercept `empty`, `zeros`, `ones`, `full`, `eye`, `identity`, `reshape`, `resize`, `stack`, `concatenate`, `outer`, `matmul`, and array-producing linear-algebra outputs. Apply the same axis-`D`, storage-cap, population-pair, triangular-pair, and registered-shape rules before calling an operation when output shape is inferable; inspect every returned array otherwise. Restore every original callable in `finally`. The three NumPy controls must be reported by this channel.

- [ ] **Step 6: Write raw profiler-event parser tests.** Feed synthetic timestamped source rows for all four actions `PREEXISTING`, `CREATE`, `INCREMENT_VERSION`, and `DESTROY`, each with source-row index, full `TensorKey(id,storage_ptr,allocation_id,device)`, version, bytes, plus the separately joined dtype/operator/stack/logical-shape witness. Require exact deduplication, monotone version transitions, `preexisting_storage_count`, `preexisting_bytes`, `baseline_live_bytes`, zero-byte-delta increments, baseline-inclusive live reconstruction, and peak. Negative cases cover duplicate preexisting/create, create at a bad version, unknown/nonmonotone increment, unknown/double/wrong-version destroy, identity byte drift, negative live total, leaked identity, forbidden shape/operator, missing or nonunique event-tree join, missing dtype/source row/TensorKey/version, and an unclassifiable live-establishing/version-changing row; every join-unavailable case is `INCONCLUSIVE`. Test documented `(timestamp,action,nbytes,category)` export rows only as lossy secondary cross-checks and prove that they cannot satisfy enrichment. The real profiler run is separate from the normative timing/HWM child and cannot alter its endpoints.

- [ ] **Step 7: Execute the exact ordered negative controls safely.** Under pre-execution guards, invoke: `torch.empty((D,D))`; `torch.empty((D*D,))`; `torch.empty((D-1,D-1))`; `torch.empty((D,))`; `torch.empty((N,N,b,b))`; `torch.empty((N*(N+1)//2,b,b))`; pair-slab `torch.stack` with fake/meta members; `torch.eye(D)` plus its logical full-width solve RHS; and dense-population `torch.linalg.eigvalsh` on a meta operand. For NumPy controls, construct only the small control inputs before installing the guard, then invoke guarded `numpy.empty((D,D))`, `numpy.outer` on two preconstructed length-`D` vectors, and `numpy.matmul` on preconstructed `(D,1)`/`(1,D)` operands. Every event carries exact production dimensions and is intercepted before the dangerous result is materialized. Channel assignment is exactly the ordered control list in Global Constraints. If an assigned channel cannot run or produce its event, return `INCONCLUSIVE`; if the operation is witnessed executing past a required detector, return `FAIL`; detection by only an unassigned channel is incomplete.

- [ ] **Step 8: Run focused GREEN tests.**

  Run: `python -m pytest tests/unit/test_h8_allocation.py -q`

  Expected: PASS with all assigned controls detected and benign block operations accepted.

- [ ] **Step 9: Review and commit Task 5.** Reviewer searches for unguarded constructors/conversions, checks alias accounting and `finally` restoration, and confirms tracemalloc is supplementary. Commit `feat(h8): certify sparse allocations`.

### Task 6: Add the Clean Child Runner and Production Resource Exercise

**Files:**
- Create: `verification/h8_child.py`
- Modify: `verification/h8_budget.py`
- Modify: `tests/unit/test_h8_allocation.py`

**Interfaces:**
- Consumes: Tasks 2--5 and exact `H8ChildRequest` JSON.
- Produces: one import-disciplined child protocol with `production`, `profiler`, and `negative_control` modes; raw OS/process/torch/backend/objective endpoints; and one canonical JSON line on stdout.

- [ ] **Step 1: Write RED child-protocol and identity tests.** Parent invocation is exactly `[sys.executable, "-m", "verification.h8_child"]` with repository root as `cwd`, canonical request JSON on stdin, captured stdout/stderr, the frozen thread environment, and `timeout=60.0` for production. Spawn with a small logical fixture and assert environment variables exist before NumPy/PyTorch import; the child calls and then verifies `torch.set_num_threads(1)`/`torch.get_num_threads()==1` and `torch.set_num_interop_threads(1)`/`torch.get_num_interop_threads()==1` before tensor work; request/result schemas are exact; stdout contains one canonical JSON object plus newline only; stderr is captured separately; and malformed/unknown modes fail without partial PASS output. Canonical hardware, affinity, thread, and BLAS records each carry a SHA-256 and every child identity must match the parent-frozen values.

- [ ] **Step 2: Implement platform HWM adapters.** On Windows implement the exact `PROCESS_MEMORY_COUNTERS_EX` field order/types/`cb`/`GetProcessMemoryInfo` error contract frozen above and capture current RSS, lifetime peak, and private bytes immediately before the production operation graph and immediately after it. On Linux/macOS use `resource.getrusage` with explicit platform unit conversion and record the same available semantics. Compute primary `conservative_incremental_hwm_bytes=max(0,post_lifetime_peak-pre_current_rss)`; record `peak_to_peak_diagnostic_bytes=max(0,post_lifetime_peak-pre_lifetime_peak)` separately and never use it to lower the primary endpoint. Unknown platform, API failure, wrong structure layout, or missing required field is an observability error, not zero.

- [ ] **Step 3: Implement `production` mode under `torch.no_grad()` and dispatch tracing.** In required order: set/verify both PyTorch thread counts; verify hardware/affinity/thread/BLAS hashes; build problem; load the independently serialized sample-noise bytes; assemble precision; factor and record every block pivot/margin; discard precision; mean solve; explicit forward and backward substitution checks; logdet; all diagonal/adjacent selected blocks; sample with exactly `[N,b]` noise; quadratic; sparse trace; diagnostic-only Hager--Higham estimate; entropy/log normalizer; complete objective; sparse reconstruction/solve/selected-recursion residuals under fixed allowances; collect counters/storage/workspace/fill; emit hashes and raw endpoints. Assert `torch.is_grad_enabled()` is false at entry and after each callback.

- [ ] **Step 4: Implement separate `profiler` mode.** Repeat the production operation graph in a different one-thread child under the pinned `torch==2.9.1` `torch.profiler.profile(activities=[ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True)` contract. Capture once in memory and return every `PREEXISTING`/`CREATE`/`INCREMENT_VERSION`/`DESTROY` source row with source-row index, full TensorKey+version, bytes, and its exact dtype/operator/stack/shape/join witness; return `preexisting_storage_count`, `preexisting_bytes`, `baseline_live_bytes`, version/dedup/liveness reconciliation, baseline-inclusive peak, and hashes proving the same source/API/problem/protocol/environment. Treat an unavailable API, missing/nonunique join, invalid version/liveness transition, or unclassifiable live-establishing/version-changing action as `INCONCLUSIVE`. Retain documented raw-export rows and aggregate/net deltas only as cross-checks; do not use profiler timings or memory as normative wall/HWM endpoints.

- [ ] **Step 5: Implement isolated `negative_control` mode.** Run exactly one named logical control, require its assigned exception/event, and return a control result. It must never invoke the complete production exercise or allocate the physical production dense buffer.

- [ ] **Step 6: Add child failure-classification tests.** Cover timeout sentinel, MemoryError, witnessed OS OOM/abnormal exit, nonzero exit, truncated/multiple/non-JSON output, nonfinite result, over-resource finite endpoint, missing profiler action/source row/TensorKey/version/dtype/shape/stack/join witness, duplicate identity, invalid increment/destroy version, inconsistent preexisting baseline, leaked/negative liveness, unclassifiable live-establishing/version-changing action, missing BLAS/thread/affinity facts, thread-setter/getter failure, HWM API/layout failure, and hash mismatch. Assert each unavailable/nonunique profiler join is `INCONCLUSIVE`. Preserve raw exit code/stdout/stderr digest for adjudication. After a valid start/config, witnessed timeout/OOM/nonzero exit/forbidden/nonfinite violations remain `FAIL` even when parsing or later observability is incomplete; only an unwitnessed/unavailable fact is `INCONCLUSIVE`.

- [ ] **Step 7: Run focused GREEN child tests.**

  Run: `python -m pytest tests/unit/test_h8_allocation.py -q`

  Expected: PASS. No production-size 15-run suite yet.

- [ ] **Step 8: Review and commit Task 6.** Reviewer checks import ordering, environment capture, HWM units, no profiler contamination of normative endpoints, failure preservation, and complete operation reachability. Commit `feat(h8): add isolated sparse scale runner`.

### Task 7: Build the Fail-Closed H8 Gate and Exact Artifact Schema

**Files:**
- Create: `verification/h8_gate.py`
- Modify: `vfe4/types/results.py`
- Modify: `verification/run_gates.py`
- Modify: `vfe4/artifacts/provenance.py`
- Modify: `verify_vfe4.py`
- Create: `tests/promotion/test_h8_gate.py`
- Modify: `tests/integration/test_verify_vfe4.py`
- Modify: `tests/unit/test_structural_types.py`
- Modify: `tests/unit/test_atomic_artifacts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: H7's atomic artifact and validated revision-specific ledger, the separate H6-Prediction PASS artifact, their transitive H1--H6-Prefix identities, 12 correctness records, child protocol, budget decisions, and canonical H8 config.
- Produces: auxiliary `H8GateEvaluation`, `H8GateResult` in `vfe4/types/results.py`, exact `validation/h8.json`, verification-owned payload-map construction, generic atomic H8 publication, and click-run status/path output.

- [ ] **Step 1: Write RED prerequisite tests.** Require the pinned final-H7 plan SHA and a same-`(HEAD,dirty_digest,candidate_junit_sha256)` H7 PASS `H7GateResult`, `validation/h7.json`, manifest, external `.verification/h7-current-candidate-<FULL_HEAD>-result.json`, source/dependency closure, fixture set, config, and `.verification/h7-<FULL_HEAD>-<FIXTURE_SET_SHA>-ledger.json` identity. Follow H7's current `references/h1_h5.json`, active `references/h1_prefix_prior.json`, and independently produced `references/h6_prefix.json` manifest/payload-or-certificate/ledger/JUnit hashes rather than copying their payloads; reject an H6-Prefix reference with any H1--H5/H4 predecessor. Separately require the immutable H6-Prediction PASS `H6PredictionResult`, manifest, checkpoint/config/data/estimator identities, exact H1/H2/H3/H5 prerequisite hashes, and `.verification/h6-prediction-<PRODUCER_HEAD>-<EXPERIMENT_SHA>-ledger.json`; reject any H4 or H4-timing prerequisite, mechanically prove H7/H8 append-only branches are unreachable from its frozen selected operation, and require every scientific dependency hash to remain exact. Recompute all manifests/hashes and reject a copied status, path substitution, result-pointer or ledger mutation, drift of the H7 chain from the current H8 producer/JUnit, drift of H6-Prediction from its recorded frozen producer/source/JUnit, changed dependency relevant to either result, H7 action/fixture mismatch, either H6 stage absent, or a changed operational `K` interpretation/hash. Missing/stale/ambiguous prerequisites yield `INCONCLUSIVE` before children launch; a current prerequisite `FAIL` blocks promotion and is preserved as `INCONCLUSIVE` with the failed prerequisite named rather than relabeling it as an H8 systems failure.

- [ ] **Step 2: Write RED 15-run orchestration tests.** Traverse seed order `20260721..23`, repetitions `0..4`; create a fresh process for every production run; run one separate profiler child per seed; run each negative control in isolation; enforce parent timeout `60.0`; reject duplicate/missing run IDs, warm reuse, reordered traversal, a result from another seed/config/protocol hash, or aggregation that discards any `PREEXISTING`/`CREATE`/`INCREMENT_VERSION`/`DESTROY` source row, TensorKey+version/dtype/join witness, preexisting baseline field, or raw endpoint.

- [ ] **Step 3: Implement gate preflight and child orchestration.** Preflight parses all immutable problem/reference inputs, runs the literal 12-cell grid through production plus the two bounded verification-only dense references, proves `vfe4/**` cannot import either reference, verifies control detectors, and only then starts production children. Parent constructs and hashes exact hardware/affinity/thread/BLAS identities before child import; every child sets/verifies both PyTorch thread counts and returns matching hashes. Capture `perf_counter_ns` from spawn through parse. Never retry a scientific failure; an infrastructure retry would change the frozen 15-run protocol and therefore yields `INCONCLUSIVE` pending a new preregistration.

- [ ] **Step 4: Write and implement exact status precedence.** First validate config/interpretation/prerequisite/preregistration identities sufficiently to establish whether a child/control start is in scope. Thereafter, a witnessed timeout, OOM/abnormal or nonzero exit, forbidden allocation/operation, off-band fill, nonfinite contract value, solver inability, completed-run omission of a required operation, reported thread/environment identity mismatch, finite residual/resource/pivot breach, invalid profiler dedup/version/liveness transition, or executed-control detector miss immediately dominates as `FAIL`, even if a later event, stack, environment field, or result fragment is unavailable. In the absence of a witnessed violation, map unavailable/unwitnessed evidence, including any missing/nonunique profiler event-tree join or source/TensorKey/version/dtype witness, incomplete observability/control assignment, missing hash, stale prerequisite, or ambiguous interpretation to `INCONCLUSIVE`. Evaluate completeness/decisiveness, child identities, all four profiler actions, storage/RHS/operation reachability, and final all-15 conjunction only after applying that dominance rule. Permit `PASS` only when `every_profiler_action_joined_and_liveness_reconciled` and every other named invariant are known true.

- [ ] **Step 5: Freeze and test `validation/h8.json`.** Set `schema_version="h8-sparse-scale-v2"` and `gate="H8"`. Reject unknown/missing keys at every level. The exact top-level key set is `schema_version`, `gate`, `status`, `obligations`, `bounded_claim`, `nonclaims`, `revision`, `config`, `prerequisites`, `interpretation`, `protocol`, `environment`, `problems`, `storage`, `factor`, `correctness`, `allocation`, `controls`, `production_runs`, `profiler_runs`, `budgets`, `invariants`, and `artifacts`. Exact nested key sets are:

  - `revision={git_head,dirty_digest,dependency_closure_sha256,manuscript_sha256,preregistration_sha256,h7_plan_sha256}`;
  - `config={config_sha256,objective_schema_sha256,protocol_sha256,canonical_json_sha256,selected_operation,ordered_gates,current_refs_registry_sha256,candidate_junit_sha256}`;
  - `prerequisites={h7_compatibility_refs,h1_h5,h1_prefix_prior,h6_prefix,h7,h6_prediction,compatibility_checks,obligations,all_current_and_pass}`. `h7_compatibility_refs` is the canonical serialization of the exact ordered `Mapping[str,H7PredecessorReference]`, including each complete keyed `payload_hashes` mapping and JUnit SHA. The other five values are, verbatim and with no extra or omitted field, canonical serializations of `H8H1H5Reference`, `H8H1PrefixPriorReference`, `H8H6PrefixReference`, `H8H7Reference`, and the amended `H8H6PredictionReference`, respectively. The H6-Prediction variant preserves exact config/readiness/matching/scorer-v2/SMC-bias/OBJECTIVE/metrics/result bindings. Keyed `content_hashes`/`payload_hashes` and the kind-specific certificate-set/certificate, result-pointer/fixture-set, and experiment fields remain explicit. No singular `content_sha256` or `result_or_certificate_sha256` reduction is allowed, and H8 stores references only rather than copying validation, certificate, or ledger bytes;
  - `interpretation={interpretation_sha256,choice_kind,K_semantics,T,N,K,d_z,d_m,b,D,V,coordinate_order,state_parent_sets,model_parent_sets,state_source_support,model_source_support,ambiguity_policy}` and `choice_kind` is exactly `"operational_preregistration_not_manuscript_theorem"`;
  - `protocol={generator_schema,generator_draw_schema_sha256,sample_schema,factor_schema,selected_inverse_schema,condition_estimator_schema,allocation_schema,torch_version,profiler_source_hashes,profiler_api_contract_sha256,profiler_raw_event_schema,child_schema,production_seed_order,production_sample_seed_map,repetition_order,correctness_seed_table,required_operations,negative_control_order}`;
  - `environment={platform,platform_release,processor,cpu_count,affinity,python_version,pytorch_version,numpy_version,device,dtype,grad_enabled,intraop_threads,interop_threads,thread_environment,blas_identity,hardware_identity_sha256,affinity_sha256,thread_identity_sha256,blas_identity_sha256}`;
  - `problems` is an ordered three-record array with each record exactly `{problem_seed,sample_noise_seed,input_sha256,sample_noise_sha256,generative_sha256,recognition_sha256,local_spd_diagnostics,transition_norms,observation_sha256}`;
  - `storage={h_scalars,input_precision_scalars,factor_scalars,selected_inverse_scalars,category_cap_scalars,dense_forbidden_scalars,input_within_cap,factor_within_cap,selected_within_cap}`;
  - `factor={algorithm,pattern,fill,workspace,condition_estimate,per_block_min_pivots,per_block_pivot_margins,global_min_pivot,global_pivot_margin,counters,reconstruction_invariants}`;
  - `correctness={grid_order,cells,cell_count,all_complete,all_decisive,all_pass}`, where every cell is `{T,d_z,d_m,N,b,D,problem_seed,sample_noise_seed,problem_sha256,sample_noise_sha256,source_results,pair_comparisons,wrong_path_controls,status,obligations}` and every comparison carries its exact operands/budget/residual;
  - `allocation={whitelist,dispatch,live_storage,profiler_api,profiler_raw_events,preexisting_storage_count,preexisting_bytes,baseline_live_bytes,profiler_reconstructed_live_peak,profiler_net_deltas_supplementary,backend,os_hwm,tracemalloc_supplementary,cross_checks,all_observable,no_forbidden_attempts}`. `profiler_api={torch_version,memory_profile_source_sha256,profiler_source_sha256,api_contract_sha256}`. Each profiler record is exactly `{source_row_index,timestamp_ns,action,tensor_key,version,nbytes,dtype,device,operator,stack,logical_shape,classification,matched_event_node_indices,join_witness_sha256,live_bytes_after}`, where `action` is the closed union `PREEXISTING|CREATE|INCREMENT_VERSION|DESTROY` and `tensor_key={id,storage_ptr,allocation_id,device}`; a documented lossy export row, when retained, is separately `{timestamp_ns,action,nbytes,category}` and never substitutes for this record. Every OS HWM record is exactly `{adapter,adapter_sha256,pre_current_rss_bytes,pre_lifetime_peak_bytes,pre_private_bytes,post_current_rss_bytes,post_lifetime_peak_bytes,post_private_bytes,conservative_incremental_hwm_bytes,peak_to_peak_diagnostic_bytes}`;
  - `controls` is the exact ordered control array `{control_id,requested_operation,logical_shapes,assigned_channels,observed_channels,execution_witnessed,event_sha256,assignment_complete,detected,status,obligations}`;
  - `production_runs` is the exact 15-record seed-major array and `profiler_runs` the exact three-record seed-major array; each child record uses the `H8ChildResult` field set plus `{parent_elapsed_ns,child_elapsed_ns,exit_code,stdout_sha256,stderr_sha256,operation_reachability,residuals,resource_decisions}`;
  - `budgets={eps,rounding_multiplier,solver_relative_budget,maximum_allowance_scale_fraction,min_cholesky_pivot,max_seconds,max_process_incremental_bytes,max_torch_population_bytes,max_storage_scalars,boundary_policy}`;
  - `invariants={prerequisites_current_and_pass,interpretation_hash_current,correctness_cells_complete,correctness_pass,controls_complete,observability_complete,every_profiler_action_joined_and_liveness_reconciled,production_runs_complete,profiler_runs_complete,required_operations_reached,storage_pass,forbidden_attempts_zero,offband_fill_zero,pivot_margin_pass,rhs_width_pass,sample_width_pass,time_pass,process_memory_pass,torch_memory_pass,finite_pass,residuals_pass,witnessed_failure_dominance_applied,all_pass}`;
  - `artifacts={config_path,provenance_path,environment_path,h7_reference_path,h6_prediction_reference_path,validation_path,manifest_path}`; this in-artifact object deliberately has no `manifest_sha256` or external result-pointer hash;
  - `nonclaims` is the exact ordered tuple `("no_language_result","no_training_result","no_prediction_result","no_large_language_model_scale","no_asymptotic_scaling_law","no_gpu_claim","no_exact_global_spectrum","no_post_h8_training_memory_transfer")` and `bounded_claim` is exactly `"The frozen T=128, K=d_z=d_m=20 synthetic chain completed within the preregistered sparse storage, allocation, numerical, time, and memory contract."` only for PASS; FAIL/INCONCLUSIVE use the same sentence prefixed by `"NOT ESTABLISHED: "`.

- [ ] **Step 6: Extend the one runner and generic artifact family.** Define `H8GateResult` beside `H7GateResult` in `vfe4/types/results.py`; extend `VerificationRunResult.gate_results` with that explicit variant and test the exact closed variant set. Selected operation `H8` validates references and runs only H8. `verification/h8_gate.py` constructs the exact payload map; `verification/run_gates.py` passes it to the existing generic `publish_run_directory`; `vfe4/artifacts` stays gate-agnostic and imports no verification code. One mocked H8 publication contains exactly `config.json`, `provenance.json`, `environment.json`, `references/h7.json`, `references/h6_prediction.json`, `validation/h8.json`, and `manifest.sha256`; its validation prerequisites embed the five exact discriminated H8 reference variants verbatim. Add a full registry -> exact H7 mapping -> H8 variant -> `validation/h8.json` -> external-pointer round-trip test that requires byte-identical keyed `content_hashes`/`payload_hashes`, candidate JUnit values, tags, and kind-specific certificate/result-pointer/experiment fields at every boundary and rejects reopening or singular aggregation. Reference records bind but never copy H1--H6-Prefix, H7, or H6-Prediction payloads/certificates/ledgers. Extend environment provenance with raw hardware/thread/BLAS/affinity identity hashes and child/profiler protocol identities.

  Implement and test `h8_current_candidate_result_payload(...)` in `verification/h8_gate.py`. It accepts only a post-publication, independently revalidated artifact/manifest and returns external `.verification/h8-current-candidate-<FULL_HEAD>-result.json` with exact top-level keys `{schema_version,candidate,artifact,current_refs,predecessors}` and `schema_version="h8-current-candidate-result-v2"`; `candidate={git_head,dirty_digest,junit_sha256}`; `artifact={path,manifest_sha256,config_sha256,validation_sha256}`; `current_refs={path,sha256}`; and `predecessors={h1_h5,h1_prefix_prior,h6_prefix,h7,h6_prediction}`. Those five values are verbatim canonical serializations of `H8H1H5Reference`, `H8H1PrefixPriorReference`, `H8H6PrefixReference`, `H8H7Reference`, and `H8H6PredictionReference`, preserving every keyed content/payload map, applicable JUnit, tag, and kind-specific certificate/result-pointer/experiment field; neither `result_or_certificate_sha256` nor any other singular lossy substitute exists. The normal H8 payload map and generic publisher never include or prewrite this pointer. Task 8 writes it exactly once after artifact validation, outside the manifest, then the H8 ledger re-reads and independently validates every byte/hash and the round-trip equality test compares each predecessor serialization byte-for-byte with the registry-derived variant.

- [ ] **Step 7: Extend the single editable click-run dictionary.** Use the exact H8 section, bootstrap identity locations, pure H7 projection, and deterministic current-candidate registry/result-pointer conventions; retain one `CONFIG`, `main`, and script guard. `main` derives the H8 registry filename from the current full HEAD, validates it with `bind_h8_current_refs`, and never selects a newest/globbed registry. Print each referenced prerequisite status, H8 status, and one artifact path. Do not add argparse or a second launcher.

- [ ] **Step 8: Update bounded documentation before candidate selection.** README and the preregistration describe the exact implemented H8 synthetic systems protocol, H7/H6-Prediction reference boundary, storage/resource limits, observability channels, status rules, and nonclaims. They do not prestate measured residuals, JUnit totals, resource endpoints, or PASS.

- [ ] **Step 9: Run focused GREEN gate/integration tests.**

  Run: `python -m pytest tests/promotion/test_h8_gate.py tests/integration/test_verify_vfe4.py tests/unit/test_structural_types.py tests/unit/test_atomic_artifacts.py -q`

  Expected: PASS using small injected child results; this is not production H8 evidence.

- [ ] **Step 10: Review and commit Task 7.** Reviewer checks witnessed-failure dominance, exact payload and external-pointer schemas, absence of an enclosing-manifest hash, H7/H6-Prediction freshness and ledger preservation, `H8GateResult` placement/closed union, verification-owned payload construction, gate-agnostic generic artifact publication, no broad-rerun path, atomicity, one CONFIG, and nonclaims. Commit `feat(h8): add sparse scale promotion gate`.

### Task 8: Produce One Exact-Revision H8 Milestone Record

**Files:**
- Modify: none. Every tracked protocol, source, test, config, launcher, README, preregistration, and artifact schema is committed before candidate selection.
- Produce outside tracked source: `C:\tmp\vfe4-h8-current-candidate-preflight.json`; `C:\tmp\vfe4-h8-milestone.xml`; one current H1--H5 artifact; one active H1-prefix-prior artifact; one H6-Prefix artifact/certificate set; `.verification/h7-current-candidate-<FULL_HEAD>-refs.json`; one H7 artifact; `.verification/h7-current-candidate-<FULL_HEAD>-result.json`; `.verification/h8-current-candidate-<FULL_HEAD>-refs.json`; one H8 artifact; `.verification/h8-current-candidate-<FULL_HEAD>-result.json`; and five new revision-specific ledgers.
- Preserve `.verification/ledger.json`, the separate H6-Prediction artifact/ledger, every historical ledger/artifact, and every bootstrap input byte-for-byte. Do not commit `.verification`, run artifacts, or JUnit output.

**Interfaces:**
- Consumes: reviewed Task 1--7 source at one clean frozen revision; final H7 plan canonical UTF-8/LF SHA `3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242`; the H7-owned H1--H5 projection, exact H6-owned projected producers, H8-owned H7 projection, and existing selected-operation verifier; bootstrap identities only; unchanged separate H6-Prediction scientific evidence; the installed verification contract; and exact committed H8 config.
- Produces: one machine-readable JUnit milestone shared by every current-candidate artifact; the exact current H1--H5, active H1-prefix-prior, H6-Prefix, and H7 evidence chain; external H7 and H8 result pointers; one atomic H8 artifact; five independently validated revision-specific ledgers; and a terminal evidence report without changing tracked source.

**Evidence policy:** Task 8 is entirely tracked-source read-only. The one JUnit is produced first; every current-candidate artifact and ledger is then produced exactly once, sequentially, at the same frozen `(git_head,dirty_digest,candidate_junit_sha256)`, in final-H7 order and then H8 order. The active verifier marker is single-valued, so every ledger must finish and remove it through the normal Stop hook before the next ledger starts. A source/test/config/fixture/preregistration/artifact-schema defect makes affected evidence `INCONCLUSIVE`: preserve the attempt, return to the owning task, commit a new candidate, and repeat all of Task 8 with one replacement JUnit. Never patch, copy, or selectively rerun an artifact at the frozen revision.

- [ ] **Step 1: Freeze the final H8 source before producing evidence.** Record a full 40-character `HEAD`; require every Task 1--7 file tracked; require empty tracked and index diffs and no nonignored untracked path outside `.verification/`; verify the pinned final-H7 plan SHA; recompute dirty-content, H8 dependency, manuscript, preregistration, interpretation, config, objective, allocation, child, and seed-table hashes; validate the unchanged separate H6-Prediction artifact/ledger and prove its scientific dependency closure is unaffected; require no `.verification/active.json`; hash `.verification/ledger.json` plus every historical ledger/artifact reference into `C:\tmp\vfe4-h8-current-candidate-preflight.json`. Require the JUnit destination and every new artifact/registry/result-pointer/ledger path to be absent rather than overwritten. Historical H1--H7 evidence is bootstrap context only and cannot count as current closure.

- [ ] **Step 2: Run the single exact-revision JUnit milestone once.** Use one command and one XML output; do not precede or follow it with another broad suite:

  ```powershell
  python -m pytest -q --junitxml=C:\tmp\vfe4-h8-milestone.xml
  ```

  Expected: exit 0. Parse totals, failures, errors, skips, and duration from XML only. If it fails, preserve XML/logs, fix on a new committed revision, run focused tests for the fix, and then run one replacement milestone XML for that new revision; never combine totals across revisions.

- [ ] **Step 3: Produce and close the current H1--H5 compatibility artifact exactly once.** In memory, resolve `project_h1_h5_compatibility_config(CONFIG)` and call the existing ordered `run_verification` once; do not route this operation through H6's two-operation producer, edit `CONFIG`, or edit tracked source. Independently validate its manifest, exact candidate revision/digest/JUnit SHA, compatibility config/objective/update/fixture identities, five distinct ordered validation payload hashes, and PASS states. Start, populate one claim per gate/identity check, and validate `.verification/h1-h5-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json`. Only after normal closure removes the active marker may Step 4 begin.

- [ ] **Step 4: Produce and close the active H1-prefix-prior artifact exactly once.** H7's pinned `h7-linear-history-source-v1` scorer consumes this result, so the condition is active. The July 25 amendment supersedes only the old projector name in this lifecycle: call `run_projected_current_candidate(config=project_h1_prefix_prior_v2_config(CONFIG),junit_sha256=junit_sha256,predecessor_refs={})` once and require `h1-prefix-prior-config-v2` plus `h1-prefix-prior-validation-v3`. Validate exact candidate/JUnit, prefix-prior config, scorer/generative schema, fixture, manifest/payload, and PASS state; close `.verification/h1-prefix-prior-<FULL_HEAD>-<MANIFEST_SHA>-ledger.json` in a fresh verifier turn. This artifact is a reference identity, never a payload copied into H7 or H8.

- [ ] **Step 5: Produce and close the current H6-Prefix certificate set exactly once.** Independently call `run_projected_current_candidate(config=project_h6_prefix_config(CONFIG), junit_sha256=junit_sha256, predecessor_refs={})` once using keyword arguments. Validate exact candidate/JUnit, H6 prefix/config/model-family/vocabulary/data-safety identities, every required certificate key, certificate-set SHA, manifest, PASS, and absence of any H1--H5/H4 predecessor identity or H4 timing invocation; close `.verification/h6-prefix-<FULL_HEAD>-<PREFIX_SET_SHA>-ledger.json` in a fresh verifier turn. Step 3 and Step 4 remain sibling H7 inputs rather than Prefix premises. Do not produce finite-SMC, checkpoint, readiness, metric, or H6-Prediction evidence.

- [ ] **Step 6: Bind final-H7 predecessors, produce H7 once, and write its external result pointer.** Atomically write `.verification/h7-current-candidate-<FULL_HEAD>-refs.json` in exact order H1--H5, active H1-prefix-prior, independently produced H6-Prefix, with the exact `H7PredecessorReference` fields, including complete keyed `payload_hashes` and the shared candidate JUnit path/hash. Capture/hash/deserialize the registry once to `Mapping[str,H7PredecessorReference]`; pass that exact in-memory mapping to `project_h7_compatibility_config` without wrapper conversion or reopen, resolve the result, and call the existing H7 selected-operation `run_verification` once rather than passing an H7 config to H6's projected producer. The pure H7 projection must preserve those mappings byte-for-byte, reproduce the pinned final-H7 canonical config, omit every H8 key, and reference rather than copy predecessor payloads/certificates/ledgers. Independently validate the H7 manifest, `H7GateResult=PASS`, `validation/h7.json`, fixture/dependency/action/oracle/control identities, all three references, and absence of H6-Prediction/H8 payloads. Then atomically write `.verification/h7-current-candidate-<FULL_HEAD>-result.json` with the exact artifact path, manifest SHA, fixture-set SHA, refs-registry SHA, revision/digest, and Step 2 JUnit SHA. The pointer is external to the H7 manifest and introduces no cycle.

- [ ] **Step 7: Review and close the revision-specific H7 ledger.** Fresh H7 reviewers consume existing Step 2/6 evidence only and do not rerun a test or gate. Re-read and independently revalidate every byte/hash in the external H7 pointer, then populate/validate `.verification/h7-<FULL_HEAD>-<FIXTURE_SET_SHA>-ledger.json` under the pinned H7 plan's claim inventory in its own verifier turn. Preserve every older H7 ledger. Any required source or schema fix abandons this candidate and restarts all of Task 8.

- [ ] **Step 8: Atomically bind the exact H8 current-reference registry.** Only after the H7 ledger validates, construct from the already validated Step 6 records, without reopening any source or reconstructing a reference from artifact bytes, the exact tagged `H8H1H5Reference`, `H8H1PrefixPriorReference`, `H8H6PrefixReference`, `H8H7Reference`, and amended `H8H6PredictionReference`. Atomically write `.verification/h8-current-candidate-<FULL_HEAD>-refs.json` with `schema_version="h8-current-candidate-refs-v3"`, exact candidate HEAD/digest/JUnit SHA, the exact H7 compatibility mapping, and those ordered variants, preserving every keyed content/payload map, all bounded Prefix aggregate and ordered family identities, and kind-specific result-pointer/experiment fields. Recompute the registry SHA, call `bind_h8_current_refs`, and require that all current-candidate identities match Step 2 while H6-Prediction instead passes its frozen scientific dependency-closure proof. Then reopen and rehash every exact referenced immutable artifact/result/ledger path to validate those parsed records without reconstructing or copying them. Any missing, copied, reordered, aggregated, or changed byte blocks H8; registry v1/v2 adds a named prerequisite obligation and remains `INCONCLUSIVE`.

- [ ] **Step 9: Run H8 exactly once and write its external result pointer.** Run `python verify_vfe4.py` with the committed H8 `CONFIG`; `main` resolves only the deterministic Step 8 registry. Expected: prerequisite statuses are PASS, `H8: pass` prints only if the full conjunction passes, and one artifact path prints. The gate launches exactly 15 cold production children, three separate profiler children, and the isolated ordered controls; preserve every raw result and do not rerun a seed/control. Independently require every observed `PREEXISTING`/`CREATE`/`INCREMENT_VERSION`/`DESTROY` row to retain source index, TensorKey+version, dtype, join witness, and valid baseline/dedup/version/liveness reconciliation; a join unavailable or nonunique is `INCONCLUSIVE`. Validate the generic atomic artifact, exact payload set, `H8GateResult`, `validation/h8.json`, config/validation hashes, the five embedded lossless reference variants, and manifest. Then atomically write `.verification/h8-current-candidate-<FULL_HEAD>-result.json` with those same variants verbatim plus candidate/JUnit, artifact/manifest/config/validation, and current-registry identities. It is not part of `manifest.sha256`.

- [ ] **Step 10: Have fresh H8 reviewers consume existing evidence only.** One reviewer checks block algebra/factor/selected inverse/pivots/objective/budgets; one checks all four profiler actions, TensorKey+version/source-row/dtype joins, preexisting baseline, dedup/version/liveness reconstruction, production whitelist, Windows HWM, controls, and resource endpoints; one checks the lossless registry -> H7 mapping -> H8 variants -> validation artifact -> external pointer round trip, H6-Prediction separation, and status dominance. They cite source, earlier focused outputs, the sole JUnit, current predecessor ledgers, and H8 artifact/pointer. They do not rerun tests or gates. Any Critical/Important fix creates a new candidate and complete replacement lifecycle.

- [ ] **Step 11: Revalidate the external pointer, then close the H8 ledger.** Re-read `.verification/h8-current-candidate-<FULL_HEAD>-result.json`; independently validate candidate HEAD/digest/JUnit, artifact path, manifest/config/validation, current-registry, every complete discriminated predecessor variant, byte-identical keyed maps/kind fields across the full round trip, and no enclosing-manifest hash in `validation/h8.json`. Derive `.verification/h8-<FULL_HEAD>-<H8_CONFIG_SHA>-ledger.json` from the validated artifact, require no active marker/path collision, and start one fresh closure ledger. Record one claim per check: complete prerequisite lifecycle/H6-Prediction separation; operational interpretation/hash; literal 12-cell three-way correctness; import boundary; operation vocabulary; input/factor/selected storage and pivot margins; all four profiler actions with source-row/TensorKey+version/dtype/join witnesses, preexisting counts/bytes/baseline, and dedup/version/liveness reconciliation; every whitelist channel/control; zero fill; RHS/sample widths; all 15 elapsed/conservative-HWM/torch-peak endpoints; hardware/affinity/thread/BLAS hashes; objective completeness; pointer/payload/manifest/JUnit identity; status precedence; and bounded nonclaims. Use current mechanical/reproduced evidence, required independent/skeptic/adjudicator views, and validate the exact ledger while source remains unchanged. Any join unavailable is `INCONCLUSIVE`; missing eligible evidence is never LLM consensus closure.

- [ ] **Step 12: Recheck immutability and report the complete evidence revision.** Recompute HEAD/digest, tracked/index diffs, all frozen hashes, both current registries, both external result pointers, all same-candidate artifacts/manifests/payload-or-certificate hashes, the separate H6-Prediction closure, five new ledger hashes, JUnit hash/XML totals, and every historical ledger hash. Expected: tracked source is unchanged; exactly the final-H7 ordered chain plus H8 evidence is new; every artifact shares the same candidate/JUnit except explicitly separate unchanged H6-Prediction; all five ledgers validate; and every older ledger is unchanged. Report exact revision/XML totals, predecessor paths/hashes, H6-Prefix certificate-set SHA, H7/H8 artifact and pointer paths, all 15 raw endpoints/maxima, pivot margins, correctness residuals only beside their own allowances, raw-event/control inventory, final status, and validated H8 ledger path. State explicitly that artifacts reference rather than copy predecessors. Do not edit tracked documentation or rerun tests/gates after closure.


---

## `validation/h8.json` Status Logic

The gate evaluates the following conjunction; no secondary diagnostic can override it:

```text
prerequisites_current_and_pass
and final_h7_plan_sha_current
and interpretation_hash_current
and seed_and_noise_protocol_hash_current
and correctness_cells_complete == 12
and every_correctness_invariant_passes_its_own_decisive_budget
and assigned_negative_controls_complete
and observability_channels_complete
and every_profiler_action_joined_and_liveness_reconciled
and production_runs_complete == 15
and profiler_runs_complete == 3
and every_required_operation_reached
and every_input_factor_selected_storage_count <= 411200
and every_forbidden_attempt_count == 0
and every_offband_fill_count == 0
and every_cholesky_pivot_margin >= 0
and every_max_rhs_width <= 40
and every_sample_width == 1
and every_elapsed_seconds <= 60.0
and every_conservative_incremental_hwm_bytes <= 134217728
and every_torch_population_peak_bytes <= 67108864
and every_child_hardware_affinity_thread_blas_hash_matches
and every_numerical_output_is_finite
and every_residual_passes_its_own_allowance
```

Status evaluation is two-stage. Before a valid config/interpretation/prerequisite start is established, missing, stale, or failed prior-gate evidence blocks execution and yields H8 `INCONCLUSIVE`, preserving any prior gate's own `FAIL`. After a valid child/control start, any witnessed timeout, OOM/abnormal or nonzero exit, forbidden allocation/operation, off-band fill, nonfinite contract value, solver inability, completed-run omission of a required operation, reported thread/environment identity mismatch, finite residual/resource/pivot breach, or executed-control missed detection is H8 `FAIL` and dominates missing later events or fields. Only unavailable or unwitnessed evidence with no witnessed violation is `INCONCLUSIVE`; unknowns are never coerced to `False`, but they also never mask an observed failure. The peak-to-peak HWM diagnostic, Hager--Higham estimate, `tracemalloc`, and aggregate/net profiler deltas cannot override the conjunction.

## Explicit Out of Scope

- WikiText-2/103 training, tokenizer/data access, held-out NLL/perplexity, optimizer behavior, checkpointing, or predictive comparison.
- GPU/CUDA allocation or throughput, batching, padding, distributed execution, mixed precision, or device kernels.
- Sequence lengths other than the frozen correctness grid and production `T=128`; parent sets wider than one; source mixtures; graph links beyond the chain.
- A claim that sparse covariance exists, that the condition estimate is an exact spectrum, or that `T=128,K=20` implies an asymptotic complexity law.
- Replacing the existing dense bounded reference everywhere, changing H1--H7 behavior, or broad refactoring unrelated to the H8 seam.
- Research-vault ingestion. The implementation result may be offered for later ingest only after the user confirms.

## Final Self-Review Checklist

- [ ] **Spec coverage:** Map every binding H8 requirement to a task, test, payload field, and status rule: operational interpretation/dimensions/order/chain/V/no-grad; literal PCG64 problem/noise seeds and draw order; block storage/factor/nonretention/pivot margins; full operation vocabulary; diagnostic-only condition estimate; four allocation channels and exact controls; 12-cell correctness; 15-run resource protocol; hardware identity hashes; prerequisites; exact schema/result pointers/provenance/nonclaims; and post-H8 training boundary.
- [ ] **Dense-path and import-boundary audit:** Search H8 production imports and call graph for `inv`, `pinv`, `covariance`, `moment_matrix`, `eye(D)`, global `eigvalsh/eigh/svd/cholesky`, flattened `outer`, `block_diag`, full-width selectors/RHS, pair slabs, an axis `D`, storage over `411_200`, or flatten/reshape to `D,D`. Verify both dense references hard-reject `T>8,K>4`, are used only in correctness preflight, and no `vfe4/**` module imports `verification/**`.
- [ ] **Allocation blind-spot audit:** Check dispatch pre-request/post-result whitelist coverage; length-`D`, near-`D^2`, rectangular/triangular/all-pairs semantics; stack/concat/outer/matmul membership; storage aliases/lifetimes; all four profiler actions `PREEXISTING`/`CREATE`/`INCREMENT_VERSION`/`DESTROY`; source-row/TensorKey+version/dtype/operator/stack/shape join witnesses; preexisting count/bytes/baseline; dedup/version/liveness reconciliation; NumPy restoration; backend counters; exact Windows `PROCESS_MEMORY_COUNTERS_EX` layout/`cb`/API errors; conservative HWM formula; and child import/thread ordering. Missing/nonunique joins remain `INCONCLUSIVE`; an executed detector miss or witnessed invalid transition remains `FAIL`.
- [ ] **Type/interface and dependency consistency:** Verify every signature/record matches the public-interface section and pinned final-H7 plan. H6-Prefix uses the one-argument projector, keyword-only producer, empty predecessor mapping, and `CandidateArtifactReference`; H1--H5 and H7 use their selected verifier paths rather than widening that producer. H6-Prediction binds H1/H2/H3/H5 and no H4 timing identity. `H8GateResult` is beside `H7GateResult` in `vfe4/types/results.py`, auxiliary types remain in `h8.py`, the closed `VerificationRunResult` field union includes H8 exactly once, `verification/h8_gate.py` owns payload construction, and generic `vfe4.artifacts` has no verification dependency. Ensure H1--H7 types remain source-compatible.
- [ ] **Budget audit:** Confirm every comparison/residual names exact operands and its own fixed allowance; solver/quadrature contributions occur exactly once; Hager--Higham is diagnostic-only; pivot margin is literal; strict decisiveness uses `<1e-4`; residual pass uses `<=`; and no `allclose`, pooled tolerance, global condition multiplier, or observed retuning exists.
- [ ] **Evidence lifecycle audit:** One exact-revision JUnit XML; then H1--H5, active H1-prefix-prior, independent H6-Prefix with `predecessor_refs={}`, H7 registry/artifact/external pointer/ledger, H8 registry/artifact/external pointer/ledger in exact order; one H8 execution; five new sequentially validated ledgers; separate unchanged H6-Prediction with H1/H2/H3/H5 and no H4 timing premise; all four profiler-action records/baselines preserved; historical ledger hashes unchanged; and tracked docs frozen before candidate selection. Require the full registry -> exact H7 mapping -> discriminated H8 variants -> immutable-path revalidation -> validation artifact -> external pointer round trip to preserve keyed maps and kind fields byte-for-byte. No singular lossy hash, reference reconstructed from reopened bytes, copied predecessor payload/certificate/ledger, enclosing-manifest hash in `validation/h8.json`, repeated broad suite, or post-evidence tracked edit.
- [ ] **Nonclaim audit:** H8 remains synthetic and systems-only; no language, training, predictive, LLM-scale, exact-spectrum, GPU, or asymptotic claim appears in code/docs/artifacts.
- [ ] **Unfilled-marker scan:** No unset values, deferred thresholds, unnamed controls, unspecified schemas, or "similar to" steps remain.

## Execution Handoff

Plan implementation should proceed task-by-task with a fresh reviewer after each bounded commit. Final execution must use the pinned H7 plan and reproduce its current-candidate H1--H5, active H1-prefix-prior, independently produced H6-Prefix, and H7 artifact/pointer/ledger chain from the sole H8-candidate JUnit before H8 runs. H6-Prefix uses `project_h6_prefix_config(CONFIG)` and `predecessor_refs={}`. H6-Prediction remains separate unchanged scientific evidence bound to H1/H2/H3/H5 and never H4 timing. Every downstream artifact binds predecessor identities without copying or rerunning predecessor payloads.
