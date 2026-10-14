# Multi-Agent VFE Simulation Suite Implementation Plan

Date: October 13, 2026 (revision 2, same day). Status as of October 14 afternoon: first-release implementation and required CUDA/scale/control execution complete. See [build handoff](../../implementation/build-handoff.md) and [implementation progress](../../implementation/progress.md) for current evidence; the steps below preserve the original plan.

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (native) or superpowers:subagent-driven-development (only if the user selects it) to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax. Read the design spec and current repository instructions first. Do not add investigator or verifier panels to implementation merely because they are available.

**Goal:** Deliver a click-to-run CUDA simulation suite for 512 or more GL(K,R) agents under the MultiAgentELBO collective free energy, with a single-point agent-lattice mode (the theory's own setting at c\*) and a spatial section mode over flat periodic 2D and 3D bases, plus presets, run artifacts, theory-backed observables, multi-seed aggregation, and a learned-versus-frozen source-prior experiment.

**Architecture:** One packed engine owns valid agent-context incidences, local patch Gaussian blocks, normalized records, independent interaction copies, and chromatic inference. The single-point mode is the engine with one design context and a declared agent-lattice source prior; the spatial mode adds patches, spatial records, and frame-induced base transport. Suite layers (presets, run folders, analytics, batch runs, aggregation, figures) sit on top of the engine and never alter its objective.

**Tech stack:** CUDA Torch through `C:/anaconda/python.exe` (`$CUDA_PYTHON`), Python dataclasses/typing, NumPy for interchange, SciPy for bootstrap intervals, Matplotlib for figures and the viewer, pytest for tiny CPU checks and an explicit CUDA lane. No neural network components and no new compiled kernels initially.

**Spec:** [Spatial simulation design](../specs/2026-10-13-spatial-multi-agent-design.md), including its section 14 (suite scope).

## What changed in revision 2

The first revision planned one spatial inference engine. The user asked for a simulation suite built in a manner similar to `MAgent_Model-main`, which is a guide and not a feature list. This revision takes the old suite's shape (click-to-run launchers with named presets, self-contained run folders, a metric recorder and numerical monitor, multi-seed runs with bootstrap aggregation, publication figures) and admits only scientific features that the live theory defines. It also makes the single-point agent lattice the first runnable mode, because it is the setting the theory states (SPEC section 0, `D={c_*}`) and gives tiny exact oracles against MultiAgentELBO's dense Gaussian network. Run artifacts move ahead of the extensions because every experiment needs them. Two theory-backed features are new: an analytics layer with measurable observables from c09, c10, c13 and g07, and slow source-prior learning with a matched frozen arm (c14, `eq:phil-source-learning`).

### Old-suite features deliberately not carried over

| Old feature | Reason it is excluded |
| --- | --- |
| Hamiltonian/leapfrog belief dynamics, mass matrices, the "high-mass rock" experiment, and the social-science inertia/oscillator suite | The theory defines only overdamped or block-descent kinetics (c08). The only inertia it states is across learning episodes for latent links (`prop:link-map-inertia`, c08b), which is not belief inertia. |
| Ouroboros tower, apex self-referential closure, T6 multi-generation hyperprior, adaptive meta-agent formation dynamics | Not defined in the theory. Autonomous parent dynamics and general formation are open (`open:rg-pointwise-parent-dynamics`, `open:rg-formation-general`, `open:full-graph-hierarchy`). The suite measures declared partitions and coherent blocks instead. |
| Consensus-KL energy `sum beta KL(q_i || Omega q_j)` with `tau * beta log beta` | Replaced by the normalized record ELBO with expected record costs and categorical KL at its proper scale (c06 `prop:obs-attention-elbo`). |
| RG flow fits, universality tests, power-law exponents | No finite-agent RG flow with exponents is stated; RG content in the theory is exact coarse identities and lattice pressure results, not fitted flows. |
| Entropy-production and non-equilibrium detectors on attention | CAVI sweeps are a deterministic relaxation, not a stochastic process; a Schnakenberg EPR would have no declared meaning. Attention reciprocity is kept as a descriptive statistic only. |
| Sphere and hyperbolic bases | Future providers behind the base protocol; not first release. |
| DeGroot / Hegselmann–Krause / bounded-confidence comparisons | Optional later baselines. c14 names them as rivals that also cluster; a separating result needs the open counterfactual-handoff formation, so the suite does not claim one. |

## Global Constraints

- No CLI parser; editable dictionaries in `run_simulation.py`, `run_multi_seed.py`, `run_aggregate.py`, and `experiments/*.py`.
- 512 or more physical agents in the target profiles; population size and grid size are independent configuration values.
- The single-point agent-lattice mode and both periodic spatial dimensions (2D, 3D) must work before the first release is complete.
- Default full K×K Gaussian fibers and full cross-node covariance within each patch channel.
- Default recognition factors between agents and between belief/model channels; label this restriction in outputs.
- Fixed frames, pure-gauge base connection in spatial mode, independent interaction copies (a mutually attending pair has two primitive copies).
- Every Gaussian normalizer and every incident receiver/source contribution is retained.
- CUDA for all large numerical work; explicit tiny CPU lanes and CPU rendering are allowed.
- No hidden CPU fallback, covariance diagonalization, unknown config keys or override keys, or silent SPD repair.
- Do not alter MultiAgentELBO, MAgent_Model-main, or the Research vault as part of implementation. Copy bounded mathematical routines with attribution; no runtime import from sibling checkouts.
- Preserve all WIP and uncommitted config toggles.
- Full function type hints, LaTeX equation docstrings, tensor shape comments, the user's signature ordering (tensors, `float | Tensor`, plain floats, ints, bools, defaulted values, Optionals, kwargs), vertically aligned comments.
- American English; one dated post-edit log in `docs/change-logs/`, updated in place for that day.
- A pure path is always reachable by config: held links, frozen source prior, full patch covariance, chromatic CAVI.

## Open decisions for the user

These do not block Tasks 1 through 8 in single-point mode. Ask before the step that needs them.

1. **Base-transport gate.** MultiAgentELBO's `AGENTS.md` (2026-09-11) and `docs/DIRECTION.md:86` close base-region simulation "before the user lifts the gate". The design spec records that the user authorized planning the spatial specialization in VFE_4.0 regardless. The single-point mode is unaffected by the gate. Confirm before the first spatial run (Task 5, step 7) whether VFE_4.0 spatial work proceeds in parallel with the gate kept in MultiAgentELBO.
2. **Git.** VFE_4.0 is not a repository. This is a multi-session build; initializing a local repository at Task 1 would give per-task commits. No remote or publication is implied. Commit steps below are conditional on the user's answer.
3. **Rival baselines.** Whether to add bounded-confidence/DeGroot reference dynamics as a later optional experiment.

## Review Focus

Each line names an input class the spec implies, with the owning test.

1. Periodic seam and anisotropic or sparse coverage: wrap, displacement, and section evaluation across the seam (Tasks 1–2).
2. Repeated endpoints, self copies, and parallel or mutual independent copies: merged maps before expectation, two primitive IDs per mutual pair (Tasks 3–4).
3. Conflicting simultaneous updates and source-side forces: chromatic sweep descends where unit-step Jacobi ascends; a source that attends to nobody still moves (Task 5).
4. Checkpoint continuation with sampled links and learned source priors: resume equals uninterrupted run (Tasks 6, 9, 10).
5. Aggregating runs whose resolved configs differ beyond seed: aggregation refuses (Task 8).

## Execution guidance

Staged build. Complete the held-link vertical slice in single-point mode, then spatial 2D/3D, then the suite layers, then extensions. Focused tests per substantive batch; a passing batch is not rerun until later edits invalidate it. One broader integration pass before handoff. Prefer native execution with bounded explorer help for existing-source questions; parallelize only modules whose contracts are fixed and whose files have distinct owners.

## Shared data contracts

All names are proposed interfaces. Integers/booleans in metadata are Python values; numerical arrays are Torch tensors on the resolved device unless exported.

```text
vfe4/state.py contracts:
Domain:        contexts[M,d], periods[d], grid_shape: tuple[int,...], cell_volume[M]
               (single-point mode: M=1, d=0, contexts has shape [1,0])
Incidence:     agent_id[S], context_id[S], patch_offset[N+1], local_node_id[S]
SourceSupport: receiver_agent[E0], source_agent[E0], context_id[E0], log_pi[E0], row_offset[R+1]
CandidateRows: row_offset[R+1], receiver_incidence[E], source_incidence[E],
               copy_id[E], log_pi[E], channel[R]
CopyTable:     receiver_agent[C], source_agent[C], context_id[C],
               primitive_id[C] (-1 for self), orientation[C], is_self[C]
LoopTable:     offset[L+1], primitive_id[T], orientation[T], base_incidence[L], weight[L]
GaussianBucket: agent_ids[B], incidence_ids[B,P], mu[B,P*K], chol[B,P*K,P*K]
PatchGaussians: buckets: tuple[GaussianBucket,...], covariance_structure: str
LinkState:     mode: str, omega_mean[C_primitive,K*K],
               precision_chol or precision_diagonal, rng_state
SectionState:  belief: PatchGaussians, model: PatchGaussians,
               beta[E], log_pi[E], links: LinkState, sweep: int
SimulationModel: domain, incidence, frame_field, candidate_rows, copies,
                 loops, baseline, records, fixed_observations
StepReport:    objective_before, objective_after, accepted, retries,
               min_spd_status, estimator_error, normalizer_status
RunResult:     run_directory: Path, final_state: SectionState, completed_sweeps: int
```

One authoritative state per Gaussian bucket; flattened nodal marginals are read-only caches. Formal reversed copies need no duplicate generator or likelihood. A mutually attending direction has a different primitive ID. Self copies have no primitive ID. `log_pi` lives in `SectionState` because Task 10 can learn it; with learning off it is a fixed copy of the model's prior.

Protocols:

```text
BaseManifold: grid(device,dtype), wrap(points), displacement(source,target),
              distance(source,target), volume_weights(), path(source,target)
GroupBackend: identity(batch_shape), compose(A,B), ratio(A,B), exp(omega),
              representation(G,channel), algebra_basis(), validate(G)
FrameField:   evaluate(agent_ids,points) -> Tensor[...,K,K]
ConnectionTransport: reference_transport(path,channel) -> Tensor[...,K,K]
SectionBasis: weights(agent_ids,points) -> sparse local node weights
```

`ratio(A,B)` means `A @ B^(-1)` via a solve. Implementations: `PointBase` and `FlatTorus`, `GLReal`, constant and periodic Fourier `FrameField`, `FlatReferenceConnection`. No unsupported stubs.

## Phase A: engine

### Task 1: Configuration, presets, base geometry

**Files:** create `run_simulation.py`, `vfe4/__init__.py`, `vfe4/config.py`, `vfe4/presets.py`, `vfe4/state.py`, `vfe4/geometry/base.py`, `tests/test_config.py`, `tests/test_base.py`, `pyproject.toml` (dependencies only).

**Interfaces.** Consumes: user dictionaries. Produces: `load_config(raw: dict) -> SimulationConfig`, `resolve_preset(name: str, overrides: dict) -> dict`, `build_domain(config: SimulationConfig) -> Domain`, `PointBase`, `FlatTorus`, explicit device/dtype selection.

- [ ] Read live source changes since the design snapshot (the three spec fingerprints) and workspace instructions; confirm no competing implementation exists.
- [ ] Write the tests first: unknown-key rejection in both the config and the override dict, explicit CUDA-unavailable failure, tuple dimension agreement, endpoint-exclusive periodic coordinates, seam equivalence, and the single-point domain.

```python
def test_periodic_displacement() -> None:
    torus = FlatTorus(periods=(1.0, 1.0), grid_shape=(4, 4))
    x = torch.tensor([[0.95, 0.50]], dtype=torch.float64)
    y = torch.tensor([[0.05, 0.50]], dtype=torch.float64)
    torch.testing.assert_close(torus.displacement(x, y), torch.tensor([[0.10, 0.0]], dtype=x.dtype))


def test_point_base_has_one_context() -> None:
    domain = build_domain(load_config(resolve_preset('tiny_point', {})))
    assert domain.contexts.shape == (1, 0)
    torch.testing.assert_close(domain.cell_volume, torch.ones(1, dtype=domain.cell_volume.dtype))


def test_override_unknown_key_fails() -> None:
    with pytest.raises(KeyError, match='populaton'):
        resolve_preset('tiny_point', {'populaton': {'n_agents': 4}})
```

- [ ] Run `& $env:CUDA_PYTHON -m pytest tests/test_config.py tests/test_base.py -q` (tests construct CPU tensors explicitly); expect failures for missing symbols.
- [ ] Implement `PointBase` and `FlatTorus` in Torch, the schema of the spec section 10, and `PRESETS` with `tiny_point`, `tiny_spatial_2d`, `tiny_spatial_3d`, `point_lattice_512`, `spatial_2d_512`, `spatial_3d_512`, `spatial_2d_1024_sparse`. `resolve_preset` deep-merges overrides and rejects any key absent from the preset schema. Save device, dtype, and every resolved value. Reject unsupported backends.
- [ ] Rerun the two test files; expect pass. Check 2D and 3D with grids at most 4 per axis.
- [ ] Commit if Git was authorized (decision 2).

**Acceptance:** changing a supported value changes the resolved object; no hidden overrides; point, 2D and 3D domain operations work.

### Task 2: Sparse supports and section-valued Gaussian state

**Files:** create `vfe4/geometry/sections.py`, `vfe4/gaussian.py`, `tests/test_sections.py`.

**Interfaces.** Consumes: `Domain`, population/patch config. Produces: `build_incidence(domain, config) -> Incidence`, `initialize_sections(model, config) -> SectionState`, `evaluate_section(state, model, agent_ids, points, channel) -> tuple[Tensor, Tensor]`, `evaluate_gaussian_coefficients(mu_coeff, chol_coeff, weights, frame) -> tuple[Tensor, Tensor]` (flattened P*K mean and Cholesky factor, P weights, K×K frame).

- [ ] Write the tests: nodal reproduction, seam evaluation, positivity, explicit absence outside support, ragged and singleton patches, the single-point case (P=1 everywhere), and cross-node covariance.

```python
def test_cross_node_variance() -> None:
    Sigma = torch.tensor([[2.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5], dtype=Sigma.dtype)
    expected = torch.tensor(0.95, dtype=Sigma.dtype)
    _, evaluated = evaluate_gaussian_coefficients(
        torch.zeros(2, dtype=Sigma.dtype), torch.linalg.cholesky(Sigma),
        weights, torch.eye(1, dtype=Sigma.dtype),
    )
    torch.testing.assert_close(evaluated[0, 0], expected)
```

- [ ] Run; expect failure.
- [ ] Implement deterministic grid-center and seeded placement, periodic stencil incidence, the coverage policy, the compact cardinal bump basis (`r0=0.95h`, scaled coordinates for anisotropy), bucketed full patch Cholesky factors, and the named `node_blocks` family. In single-point mode every agent has one incidence at `c_*`.
- [ ] Rerun; expect pass. Commit if authorized.

**Acceptance:** M, N, and S are distinct; off-support evaluation returns a mask, never a fabricated law; patch entropy is the entropy of the full block.

### Task 3: GL frames, transports, source supports, independent copies

**Files:** create `vfe4/geometry/groups.py`, `vfe4/geometry/transport.py`, `vfe4/geometry/supports.py`, `tests/test_transport.py`, `tests/test_copies.py`.

**Interfaces.** Consumes: `Domain`, `Incidence`, bundle/communication config. Produces: `GLReal`, `ConstantFrameField`, `FourierFrameField`, `FlatReferenceConnection`, `overlap_support(incidence, domain, config) -> SourceSupport`, `agent_lattice_support(n_agents: int, lattice_shape: tuple[int, ...], family: str, radius: int | None, decay: float | None, alpha: float | None) -> SourceSupport`, `build_copy_table(support: SourceSupport, include_self: bool) -> tuple[CandidateRows, CopyTable, LoopTable]`, `base_transport(...)`, `interaction_transport(...)`.

- [ ] Write the tests: pure-gauge endpoint composition and nonflat interaction-loop holonomy with shear/scaling and a negative-determinant frame; passive reframing (`g Sigma g.T`, `g_target Omega g_source^(-1)`); lattice prior rows; mutual-pair copy identity.

```python
def test_agent_lattice_window_rows() -> None:
    support = agent_lattice_support(8, (8,), 'window', 2, None, None)
    pi = support.log_pi.exp()
    degrees = (support.row_offset[1:] - support.row_offset[:-1]).tolist()
    assert degrees == [4] * 8
    for row in torch.split(pi, degrees):
        torch.testing.assert_close(row, torch.full((4,), 0.25, dtype=pi.dtype))


def test_mutual_pair_has_two_primitive_copies() -> None:
    support = agent_lattice_support(4, (4,), 'window', 1, None, None)
    _, copies, loops = build_copy_table(support, include_self=True)
    pair = (copies.receiver_agent == 0) & (copies.source_agent == 1)
    back = (copies.receiver_agent == 1) & (copies.source_agent == 0)
    assert copies.primitive_id[pair].item() != copies.primitive_id[back].item()
    assert (copies.primitive_id[copies.is_self] == -1).all()
    n_mutual_pairs = 4
    assert loops.offset.numel() - 1 == 2 * n_mutual_pairs   # both based two-cycles per pair
```

- [ ] Run; expect failure.
- [ ] Implement the defining representation, batched matrix exponential, solve-based ratio, mean transport, covariance congruence, invertibility checks; both determinant components for frames. Implement `agent_lattice_support` for ring and periodic 2D/3D agent lattices with families `window`, `exponential`, `power`, `complete`, `directed_window` (adapt the row conventions of MultiAgentELBO `realizations/gaussian/locality.py:source_prior`, with attribution; self handled only by `build_copy_table`). Implement `overlap_support` from patch overlap and fixed geometry with an optional maximum-source structural mask.
- [ ] Rerun geometry, copy, and section tests once. Commit if authorized.

**Acceptance:** base loops telescope; interaction loops need not; self transport is identity; locality in single-point mode enters only through the source prior.

### Task 4: Fixed synthetic observations and the normalized collective objective

**Files:** create `vfe4/model.py`, `vfe4/observations.py`, `vfe4/objective.py`, `tests/test_records.py`, `tests/test_objective.py`.

**Interfaces.** Consumes: geometry, state, row/copy tables. Produces: `build_model(config) -> SimulationModel`, the truth fixture, `evaluate_vfe(state, model) -> VFEReport` (termwise: model prior, bridge, private, spatial, communication, categorical KL, entropy, link prior), `optimal_rows(costs, rows) -> Tensor`, `linear_gaussian_moments(mu, Sigma, A, o) -> tuple[Tensor, Tensor]`.

- [ ] Write the tests: same-variable residual, dense exact-evidence identity, expected-cost versus plug-in rows, rows summing to one, passive covariance of the whole objective, parallel copies.

```python
def test_same_variable_identity_residual() -> None:
    Sigma = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    A = torch.eye(2, dtype=Sigma.dtype)
    B = A.clone()
    _, V = linear_gaussian_moments(
        torch.zeros(2, dtype=Sigma.dtype), Sigma, A - B,
        torch.zeros(2, dtype=Sigma.dtype),
    )
    torch.testing.assert_close(V, torch.zeros_like(Sigma))
```

  The dense oracle: for a tiny model with singleton source alphabets, assemble global J, h, c independently and check `F(Q) + log p(o) = KL(Q || posterior)` with the full block-diagonal recognition covariance. In single-point mode, also compare the assembled precision against an attributed copy of MultiAgentELBO `assemble_network` on N≤4, K=2. Do not build dense oracles at large N.
- [ ] Run; expect failure.
- [ ] Implement the periodic Fourier truth fixture (in single-point mode, one random truth vector per agent drawn from a declared population law), sensor maps, fixed noisy private records, secret source labels, one comparison record per row; proper model priors and bridges; spatial records conditioned at zero as the declared continuity assumption (spatial mode only); analytical expectations, patch entropies, categorical KL, sorted ragged log-softmax. Retain all determinant constants and `1/tau`. Merge repeated endpoint maps before expectation.
- [ ] Rerun focused tests (N≤3, patch nodes≤2, K≤2, CPU). Commit if authorized.

**Acceptance:** one normalized fixed joint is scored; truth reaches inference only through records; no recognition marginal becomes a generative prior.

### Task 5: Chromatic inference and the first held-link runs

**Files:** create `vfe4/inference.py`, `vfe4/simulation.py`, `tests/test_inference.py`; complete the launch path in `run_simulation.py`.

**Interfaces.** Consumes: `SimulationModel`, `SectionState`. Produces: `make_schedule(model, config) -> UpdateSchedule`, `step(state, model, schedule, config) -> tuple[SectionState, StepReport]`, `damp_moments(mu, mu_star, Sigma, Sigma_star, w) -> tuple[Tensor, Tensor]`, `run_simulation(config) -> RunResult` (minimal output here; Task 6 owns artifacts).

- [ ] Write the tests: the scheduling counterexample (record `o ~ N(y1+y2+y3, 5/3)`, baseline variances `5/2`; colored sweep decreases F, unit-step Jacobi increases it, since its iteration matrix has spectral radius 1.2); damping; a source with no outgoing attention still updated by the receiver's record.

```python
def test_gaussian_moment_damping() -> None:
    mu = torch.tensor([0.0], dtype=torch.float64)
    star = torch.tensor([2.0], dtype=mu.dtype)
    Sigma = torch.tensor([[1.0]], dtype=mu.dtype)
    Sigma_star = torch.tensor([[3.0]], dtype=mu.dtype)
    mu_new, Sigma_new = damp_moments(mu, star, Sigma, Sigma_star, 0.5)
    torch.testing.assert_close(mu_new, torch.tensor([1.0], dtype=mu.dtype))
    torch.testing.assert_close(Sigma_new, torch.tensor([[3.0]], dtype=mu.dtype))
```

- [ ] Run; expect failure.
- [ ] Implement conflict coloring from all baseline and record scopes, frozen rows within a color, local precision assembly with sparse incident contributions, batched bucket solves, belief colors then model colors at cadence with sufficient-statistic damping, strict SPD failure reporting, and per-stage objective checks. Add the opt-in Fisher natural-coordinate path with collective backtracking, labeled separately from relaxed CAVI.
- [ ] Rerun focused inference tests.
- [ ] Run `tiny_point` end to end. Open the active dictionary, trace each material key through `load_config`, `build_model`, and `make_schedule`, and confirm the selected branches execute.
- [ ] Before the first spatial run, ask decision 1 if still open. Then run `tiny_spatial_2d` and `tiny_spatial_3d` end to end with the same key trace. Commit if authorized.

**Acceptance:** the first vertical slice exists in single-point and spatial modes: both channels, selective rows, fixed independent twists, and nonincreasing fixed-data held-link VFE.

## Phase B: suite layers

### Task 6: Run folders, metric recorder, numerical monitor, resume

**Files:** create `vfe4/io.py`, `vfe4/recorder.py`, `vfe4/monitor.py`, `tests/test_resume.py`, `tests/test_outputs.py`; wire into `vfe4/simulation.py`.

**Interfaces.** Consumes: model, state, `StepReport`, output config. Produces: `RunWriter(run_dir)`, `MetricRecorder.record(sweep: int, report: StepReport, vfe: VFEReport, observables: dict[str, float])`, `NumericalMonitor.check(state, report) -> MonitorStatus`, `save_checkpoint(...)`, `load_checkpoint(path) -> tuple[SimulationModel, SectionState, RecorderState]`.

- [ ] Write the tests: uninterrupted 4 sweeps versus 2 + save + resume + 2 on `tiny_point` (exact equality in float64 on CPU); a failed SPD block produces a run folder whose summary names the sweep, agent, and channel; `metrics.csv` has one row per recorded sweep with every termwise VFE column.
- [ ] Run; expect failure.
- [ ] Implement `runs/<timestamp>-<preset>/` with `config.json` (resolved), `environment.json` (interpreter, Torch, CUDA, device, source provenance), geometry/incidence/copy tables, fixed observations, `metrics.csv`, `summary.json`, rotated checkpoints, `final_state.pt`. Save full patch blocks, never only marginals. Persist RNG and estimator states, `beta`, `log_pi`, links, cadence position, normalizer status. Changing model or data config on resume starts a new run. The monitor batches one status check per stage and stops on non-finite values or failed factorizations; it never repairs.
- [ ] Rerun; expect pass. Commit if authorized.

**Acceptance:** every run is self-describing and resumable; partial runs say where they failed; recording never retains autograd graphs or forces per-edge host transfers.

### Task 7: Analytics observables

**Files:** create `vfe4/analytics/attention.py`, `vfe4/analytics/coherence.py`, `vfe4/analytics/holonomy.py`, `vfe4/analytics/coarse.py`, `tests/test_analytics.py`; register observables with the recorder.

**Interfaces.** Consumes: state, model, copy/loop tables. Produces:
- `row_statistics(beta, rows) -> dict[str, Tensor]`: row entropy, effective degree `exp(H)`, self weight, descriptive reciprocity on mutual pairs.
- `coherent_partition(residual_norm: Tensor, edge_src: Tensor, edge_dst: Tensor, resolution: float, n_agents: int) -> Tensor`: component labels of `(V, Coh_a)`, the mean-field analogue of c13's `Coh_a(x)` evaluated on transported recognition means in a declared reference metric.
- `fixed_sector_rank(transports: Tensor, edge_src: Tensor, edge_dst: Tensor, n_agents: int, rtol: float = 1e-10) -> int`: kernel dimension of the connection Laplacian (c10 `prop:cg-kernel-holonomy`).
- `loop_holonomy_defect(...)`: metric-aware defect per loop in `LoopTable`.
- `discarded_conditional_kl(mu_q, Sigma_q, mu_p, Sigma_p, S) -> Tensor`: Δ of c09 `thm:rg-exact-coarse-vfe` for a declared linear block channel `z = S y` on Gaussian recognition and posterior, computed as `KL(q||Π) - KL(qS||ΠS)` (evidence cancels).
- `field_error(state, model) -> Tensor` in the fixture's reference/sensor frame.

- [ ] Write the tests:

```python
def test_coherent_partition_components() -> None:
    edge_src = torch.tensor([0, 1, 2])
    edge_dst = torch.tensor([1, 2, 3])
    residual_norm = torch.tensor([0.1, 0.9, 0.2], dtype=torch.float64)
    labels = coherent_partition(residual_norm, edge_src, edge_dst, 0.5, 4)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[1] != labels[2]


def test_fixed_sector_rank_detects_holonomy() -> None:
    K = 2
    I = torch.eye(K, dtype=torch.float64)
    R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    edge_src = torch.tensor([1, 2, 0])
    edge_dst = torch.tensor([0, 1, 2])
    assert fixed_sector_rank(torch.stack([I, I, I]), edge_src, edge_dst, 3) == K
    assert fixed_sector_rank(torch.stack([I, I, R]), edge_src, edge_dst, 3) == 0


def expected_conditional_kl_oracle(m_q: torch.Tensor, C_q: torch.Tensor, m_p: torch.Tensor,
                                   C_p: torch.Tensor, n_coarse: int) -> torch.Tensor:
    r"""E_{z ~ q_z} KL(q(r|z) || p(r|z)) for Gaussians in coordinates (z, r)."""
    z, r = slice(0, n_coarse), slice(n_coarse, None)
    G_q = torch.linalg.solve(C_q[z, z], C_q[z, r]).T
    G_p = torch.linalg.solve(C_p[z, z], C_p[z, r]).T
    V_q = C_q[r, r] - G_q @ C_q[z, r]
    V_p = C_p[r, r] - G_p @ C_p[z, r]
    a = (m_q[r] - m_p[r]) - G_p @ (m_q[z] - m_p[z])
    D = G_q - G_p
    P = torch.linalg.inv(V_p)
    k = V_p.shape[0]
    return 0.5 * (torch.trace(P @ V_q) + a @ P @ a + torch.trace(P @ D @ C_q[z, z] @ D.T)
                  - k + torch.logdet(V_p) - torch.logdet(V_q))


def test_discarded_conditional_kl_matches_conditional_oracle() -> None:
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    B = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Sigma_q = A @ A.T + torch.eye(3, dtype=torch.float64)
    Sigma_p = B @ B.T + torch.eye(3, dtype=torch.float64)
    mu_q = torch.randn(3, generator=g, dtype=torch.float64)
    mu_p = torch.randn(3, generator=g, dtype=torch.float64)
    S = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    T = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    expected = expected_conditional_kl_oracle(T @ mu_q, T @ Sigma_q @ T.T,
                                              T @ mu_p, T @ Sigma_p @ T.T, 1)
    torch.testing.assert_close(discarded_conditional_kl(mu_q, Sigma_q, mu_p, Sigma_p, S), expected)
    assert expected >= 0
```

- [ ] Run; expect failure.
- [ ] Implement. `discarded_conditional_kl` needs the dense Gaussian posterior of the blocks it covers; compute it only when the covered dimension is within `analytics.coarse_max_dim` and record `unavailable` otherwise. Never substitute a factorized posterior. Holonomy and reciprocity diagnostics declare their reference metric and frame.
- [ ] Rerun; expect pass. Commit if authorized.

**Acceptance:** each observable cites its theory label in its docstring, states whether it is exact or a declared mean-field analogue, and is recorded at a configured cadence on device.

### Task 8: Multi-seed batches, parameter sweeps, aggregation

**Files:** create `run_multi_seed.py`, `run_aggregate.py`, `vfe4/batch.py`, `vfe4/aggregate.py`, `tests/test_aggregate.py`.

**Interfaces.** Consumes: presets, `run_simulation`, run folders. Produces: `run_batch(base: dict, seeds: list[int], grid: dict[str, list]) -> list[Path]` (one folder per seed × grid point, sequential on one GPU), `load_runs(runs_dir: Path) -> RunTable`, `aggregate_metric(table: RunTable, metric: str, by: str, n_resamples: int, method: str, seed: int) -> dict[str, list[float]]`.

- [ ] Write the tests:

```python
def test_aggregate_constant_metric() -> None:
    table = RunTable.from_columns({'run_id': ['a', 'b', 'c', 'a', 'b', 'c'], 'sweep': [0, 0, 0, 1, 1, 1],
                                   'vfe_total': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]}, configs={})
    out = aggregate_metric(table, 'vfe_total', 'sweep', 999, 'percentile', 0)
    assert out['mean'] == [1.0, 2.0]
    assert out['ci_low'] == [1.0, 2.0] and out['ci_high'] == [1.0, 2.0]


def test_load_runs_rejects_mixed_configs(tmp_path: Path) -> None:
    write_fake_run(tmp_path / 'r0', {'runtime': {'seed': 0}, 'bundle': {'K': 2}})
    write_fake_run(tmp_path / 'r1', {'runtime': {'seed': 1}, 'bundle': {'K': 3}})
    with pytest.raises(ValueError, match='bundle.K'):
        load_runs(tmp_path)
```

  `write_fake_run` is a test helper writing `config.json` and a two-row `metrics.csv`.
- [ ] Run; expect failure.
- [ ] Implement. `load_runs` groups runs by resolved config with `runtime.seed` removed and refuses to pool differing groups unless the caller names the swept keys, which become columns. Intervals use `scipy.stats.bootstrap` (`BCa` default, `percentile` allowed). Write `aggregate.csv` and publication CI-band figures.
- [ ] Rerun; expect pass. Run a 3-seed `tiny_point` batch and aggregate it once. Commit if authorized.

**Acceptance:** batch outputs are ordinary run folders; aggregation never silently pools different models.

## Phase C: model extensions and experiments

### Task 9: Uncertain independent generators and fixed loop priors

**Files:** create `vfe4/latent_links.py`, `tests/test_latent_links.py`; extend model, objective, and inference with an explicit `latent_gaussian` branch.

**Interfaces.** Consumes: held-link reference, copy/loop tables, link config. Produces: generator recognition, chunked expectations, `propose_link_step(...) -> LinkProposal`, accepted estimated-objective steps, normalizer metadata.

- [ ] Write the tests: Gaussian KL and the BLR proposal against direct small-matrix calculations; finite-difference derivatives on K=2 with one and two copies; a quadratic integrand against its analytical expectation; non-SPD projected Hessian rejected with step shrink; identical paired estimates at the same state; doubling samples changes reported uncertainty, not the model; resume of a sampled-link `tiny_point` run with both estimator streams restored equals the uninterrupted run.
- [ ] Run; expect failure.
- [ ] Implement diagonal or full per-copy Gaussian recognition with strictly positive variance and a normalized Gaussian base prior; `held` and `latent_gaussian` as distinct modes; reparameterized antithetic samples with analytical matter expectations conditional on samples; copy chunks only; Gaussian Fisher/BLR updates (no unpreconditioned generator steps); loop action with both based two-cycles and a metric-aware reference holonomy; `logZ=0` exactly when all loop coefficients vanish, else `normalizer_status='unavailable'` and objective labeled relative. Backtrack on common samples; report on an independent stream.
- [ ] Rerun focused CPU tests; small CUDA agreement when the GPU is free. Commit if authorized.

**Acceptance:** the held-link path remains reachable; uncertain-link outputs state their estimator and normalizer status.

### Task 10: Slow source-prior learning with a frozen control arm

**Files:** create `vfe4/source_learning.py`, `experiments/run_formation.py`, `tests/test_source_learning.py`; add `communication.source_rate` and `communication.source_every` to the schema (default `0.0`, the frozen pure path).

**Interfaces.** Consumes: `beta`, `log_pi`, rows. Produces: `source_prior_step(log_pi: Tensor, beta: Tensor, row_offset: Tensor, rate: float) -> Tensor`, and the experiment runner writing paired arms.

The step is the explicit Euler step of `eq:phil-source-learning` (c14), `pi' = (1 - rate) pi + rate beta`, which is the natural-gradient step of the collective free energy in each row under the simplex Fisher metric with rest point the recognition label marginal (`eq:obs-aggregate-prior`, c12). It is an accepted M-coordinate of a generative parameter at fixed Q; it keeps rows on their structural support for `0 < rate <= 1` and never deletes a candidate.

- [ ] Write the tests:

```python
def test_source_prior_natural_step() -> None:
    log_pi = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    beta = torch.tensor([0.9, 0.1], dtype=torch.float64)
    row_offset = torch.tensor([0, 2])
    new_log_pi = source_prior_step(log_pi, beta, row_offset, 0.1)
    torch.testing.assert_close(new_log_pi.exp(), torch.tensor([0.54, 0.46], dtype=torch.float64))
    kl_old = (beta * (beta.log() - log_pi)).sum()
    kl_new = (beta * (beta.log() - new_log_pi)).sum()
    assert kl_new < kl_old
```

  Also test: `source_rate=0` leaves `log_pi` bitwise unchanged over a run; the full objective is nonincreasing across the M-step at fixed Q; resume restores learned rows.
- [ ] Run; expect failure.
- [ ] Implement the step and its schedule slot (after belief colors, at `source_every`). Implement `experiments/run_formation.py`: matched seeds, a learned arm and a frozen arm (`source_rate=0`) on `point_lattice_512` with a declared symmetry-breaking datum per SPEC R13, recording row concentration, coherent-partition block counts at declared resolutions, and fixed-sector rank. Results are labeled descriptive. The engine's mean-field CAVI is not the stochastic kinetics of `def:rg-formation-cycle-experiment`, so these runs do not test `conj:phil-contrast-formation` or `prop:phil-seeded-contrast-formation`.
- [ ] Rerun tests; run one tiny paired experiment. Commit if authorized.

**Acceptance:** learning is opt-in, normalized, and resumable; every formation figure shows both arms.

### Task 11: Publication figures and the live viewer

**Files:** create `vfe4/visualization.py`, `vfe4/viewer.py`; extend `tests/test_outputs.py` only for file existence and non-empty images.

- [ ] Static figures saved as PDF/PNG/SVG: termwise VFE curves; field and error maps (2D) and selectable slices (3D); agent supports; uncertainty with declared frame; selected routing; row-entropy and effective-degree distributions; coherent-partition maps at declared resolutions; fixed-sector rank and holonomy defect over sweeps; single-point lattice views (agents on the declared lattice, colored by block). Labels state frame and objective convention.
- [ ] Viewer: pause, resume, single sweep, stop on detached snapshots; headless runs the same solver; only selected aggregates and slices are transferred.
- [ ] Render a tiny run in each mode, inspect the images, and open the config/metrics for the user. Commit if authorized.

**Acceptance:** figures are publication quality by default and do not mislabel gauge-dependent quantities as invariant.

## Phase D: scale and handoff

### Task 12: CUDA validation and 512/1024-agent measurements

**Files:** create `tests/test_cuda.py`, `experiments/run_scaling.py`, `docs/performance/scaling.md`.

- [ ] Verify `$CUDA_PYTHON` CUDA availability and the current GPU workload; do not compete with an active job. The CUDA lane fails if it did not select CUDA.
- [ ] Tiny CPU/CUDA comparisons in all three modes: GL shears/scales, patch expectations, complete VFE, one accepted sweep. float64 at `atol=1e-8, rtol=1e-7` for well-conditioned cases; one bounded-condition float32 case with its own recorded tolerance.
- [ ] Preflight reports S, E, maximum row degree, block dimensions, predicted storage, and free memory against the configured limit; it never shrinks N, K, the covariance family, or source support to fit.
- [ ] Experiments (not pytest): `point_lattice_512` and `spatial_2d_512`, `spatial_3d_512` held-link for 100 sweeps; `spatial_2d_1024_sparse` for 20 sweeps; one 512-agent uncertain-link profile at a recorded smaller source degree and sample count; one 5-seed `point_lattice_512` batch with aggregation. Log configs, seeds, dtype, timing, peak memory, termwise objectives, SPD status, field error.
- [ ] Profile and optimize measured hotspots only (local factorization, candidate expectations, row reductions, generator derivatives, snapshot transfer). If full patch covariance exceeds resources, show the named `node_blocks` family and report the changed recognition family alongside the failed admission.
- [ ] Save JUnit XML for the CUDA lane and JSON/CSV for experiments. One broader integration pass after all code edits.

**Acceptance:** measured timing and memory for each target profile, with unavailable profiles reported as such.

### Task 13: Pointwise exchange and documentation

**Files:** modify `vfe4/io.py`, `vfe4/config.py`; create `tests/test_exchange.py`, `README.md`, `docs/model-contract.md`; update `docs/change-logs/<date>.md`.

- [ ] Export a chosen context with physical agent IDs, all copy IDs, both law channels, generator recognition, source rows (learned or fixed), comparison maps, priors, and observation normalizers, as either a spatial-disabled model or an explicitly conditioned model with the fixed neighboring values recorded.
- [ ] Test a tiny conditioned context against dense conditioning of the original Gaussian model, and the spatial-disabled reduction.
- [ ] Document setup, presets and click-to-run editing, the three modes, model and recognition assumptions, held versus latent objective status, source learning, every observable with its theory label and exact/analogue status, multi-seed aggregation, figures, resume, numerical failures, and extension protocols. Trace every public config key to its executing path.
- [ ] Update the day's change log. Offer Research-wiki ingestion only after findings exist, and write only after confirmation.

**Acceptance:** the user can run, sweep, aggregate, and inspect the suite independently; exported contexts are typed so a marginal slice is never mistaken for an independent model.

## Proposed preset contents

Design examples; exact names follow the Task 1 schema.

```python
PRESETS['point_lattice_512'] = {
    'runtime': {'device': 'cuda', 'dtype': 'float64', 'seed': 20261013},
    'manifold': {'kind': 'point'},
    'population': {'n_agents': 512, 'lattice_shape': (32, 16)},
    'bundle': {'group': 'GL_R', 'K': 4, 'frame_field': 'constant', 'frame_spread': 0.1},
    'recognition': {'covariance_structure': 'full_patch'},
    'generative': {'bridge': 'identity_gaussian', 'bridge_std': 0.5, 'model_prior_std': 2.0},
    'observations': {'kind': 'synthetic_population', 'private_std': 0.15, 'private_fraction': 0.25},
    'communication': {'comparison': 'full_state', 'channels': ('belief', 'model'),
                      'source_support': 'agent_lattice', 'prior_family': 'window', 'radius': 2,
                      'include_self': True, 'tau': 0.5, 'source_rate': 0.0, 'source_every': 1},
    'links': {'mode': 'held', 'shared_channels': True, 'initial_twist_std': 0.05, 'prior_std': 0.1,
              'generator_precision': 'diagonal', 'mutual_loop_coefficient': 0.0,
              'samples': 8, 'copy_chunk_size': 4096},
    'spatial': {'enabled': False},
    'inference': {'method': 'chromatic_cavi', 'sweeps': 100, 'model_weight': 0.1,
                  'model_every': 1, 'link_every': 5, 'link_dt': 0.02},
    'analytics': {'every': 5, 'coherence_resolutions': (0.1, 0.3, 1.0), 'coarse_max_dim': 2048},
    'output': {'directory': 'runs', 'viewer': False, 'snapshot_every': 5, 'checkpoint_every': 25},
}

PRESETS['spatial_2d_512'] = {
    **PRESETS['point_lattice_512'],
    'manifold': {'kind': 'flat_torus', 'dimension': 2, 'periods': (1.0, 1.0), 'grid_shape': (32, 32)},
    'population': {'n_agents': 512, 'placement': 'grid', 'center_grid': (32, 16),
                   'patch_shape': (5, 5), 'coverage': 'required'},
    'bundle': {'group': 'GL_R', 'K': 4, 'frame_field': 'periodic_fourier', 'frame_amplitude': 0.1},
    'observations': {'kind': 'synthetic_periodic_field', 'private_std': 0.15, 'private_fraction': 0.25},
    'communication': {**PRESETS['point_lattice_512']['communication'],
                      'source_support': 'overlap', 'max_sources': None},
    'spatial': {'enabled': True, 'record_value': 'zero', 'noise_std': 0.5},
}
```

```python
# run_simulation.py
PRESET    = 'point_lattice_512'
OVERRIDES = {'bundle': {'K': 4}, 'inference': {'sweeps': 100}}

if __name__ == '__main__':
    run_simulation(load_config(resolve_preset(PRESET, OVERRIDES)))
```

The 3D profile changes dimension, periods, grid `(16,16,16)`, center grid `(8,8,8)`, and patch `(3,3,3)`. These values are starting assumptions for experiments, not theoretically selected constants.

## Completion boundary

Tasks 1–5 give the exact held-link engine in single-point and spatial modes. Tasks 6–8 make it a suite (artifacts, observables, batches, aggregation). Tasks 9–11 add uncertain links, source learning with a frozen arm, and figures. Tasks 12–13 establish measured CUDA scale and handoff. The release requires all three modes and every stated output and model boundary.

Curved bases, other groups, adaptive meta-agent formation, and spatial renormalization have no checkbox: the protocols leave room for them, and the theory leaves their dynamics open.
