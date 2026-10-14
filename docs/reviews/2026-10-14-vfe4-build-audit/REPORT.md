# VFE_4.0 build audit, October 14, 2026

**Object.** The VFE_4.0 simulation suite at Git HEAD `3ebd31c776a4e619c760d46bcb19d692f598b607`, which was clean, checked against design spec revision 2 and the implementation plan.

**Panel.** Nine investigators ran in parallel under the shared [BRIEF.md](BRIEF.md): variational, gauge theory, numerical analysis, information geometry, config wiring, test validity, claims versus evidence, GPU performance, and supporting modules. All were Claude Opus 5.5, as was the synthesis. Each lens report, with evidence and scratch-script names, is in [lenses/](lenses/).

**Method.** Every investigator was read-only and ran CPU checks only: no GPU work, no large runs, no full suite. Executed checks used tiny systems, independent dense assemblers, finite differences, and source mutations applied through a pytest plugin.

**Status.** Nothing here has been adjudicated. No skeptic, defender, or cross-model verifier was run. Where two or more lenses reached the same finding independently, the entry says so; that is corroboration, not closure.

## Verdict

The mathematical core is correct. An independently written dense assembler reproduces the collective free energy to at most 7.3e-12 on F ≈ 6.6e3. That covers point, projection, spatial full_patch and node_blocks, 3D, and wrap-cycle cases. Each color's CAVI update is the exact dense block optimum, to 3e-16. The coloring never puts two coupled blocks in one color. Damping, the Fisher path, and source learning descend. The latent-link estimators, derivatives, and BLR step match independent recomputation. The objective is passively gauge-invariant by construction. Exchange conditioning matches dense Gaussian conditioning. Resume is bitwise exact. The numeric claims in the docs reproduce from the saved artifacts, and no saved run of the 37 has a VFE increase.

The defects lie around that core:
- **Observables:** two theory-labeled observables are numerically wrong in reachable configurations, and several recorded observables and status fields cannot vary.
- **Run provenance:** aggregation double-counts resumed runs, and a failed step persists its rejected state.
- **Latent links:** latent-link inference starts at the data-generating twists.
- **Tests:** the test suite does not establish several obligations claimed for it; four engine mutations pass every test.
- **Performance:** the full-patch sweep costs O(colors · Σ D³) where the spec requires O(Σ D³).

## Findings by severity

The IDs refer to the lens files. Findings reached independently by more than one lens are merged.

### High

| # | Finding | Lens IDs | Location |
|---|---|---|---|
| H1 | `fixed_sector_rank` (c10) overcounts on deep spanning trees. It uses one global singular-value threshold scaled by path-product norms that grow exponentially with tree depth. On a 512-ring with K=4, radius 1, and twist std 0.5 it reports 2; the true value is 0. It is correct on small graphs and on the active lattice. | G1 | holonomy.py:36-48 |
| H2 | The reciprocity observable counts self rows as reciprocal pairs, which scales every recorded value by the non-self fraction: 4/5 under the active config, including the formation controls. On tiny_point it reports 0.355; the true value is 0.533. | T5 | observables.py:81-88 |
| H3 | Aggregation pseudo-replicates runs. A resumed continuation lands beside its parent in the batch root with the same seed and history, and is pooled as an extra run. Runs with no summary.json are also pooled. Executed: three independent seeds were counted as five rows. | W1, Q1 | aggregate.py:51-72; simulation.py:133-138 |
| H4 | The scheduling counterexample never runs the engine. A mutation that turns chromatic CAVI into Jacobi passes all 33 engine tests. Review Focus 3 is not established by any test. | T1 | test_grounding.py:34-47 |
| H5 | The full-patch sweep rebuilds every agent's covariance twice per color, giving O(colors · Σ D³) where spec §9 requires O(Σ D³). At spatial_2d_512 (24 colors) that is about 135 (512,100,100) materializations per sweep. This affects speed only. | PERF-1 | inference.py:134-162; state.py:96-108 |

### Medium

| # | Finding | Lens IDs |
|---|---|---|
| M1 | Latent-link recognition is initialized at the exact twists that generated the records, which contradicts spec §6. Separately, the generator prior anchor is fixed at 0 with covariance σ²I, so c08b's held-link limit cannot be exercised; shrinking `prior_std` sends the latent mean to 0 rather than to the held twist. The two are one mismatch among prior, initialization, and fixture. | V1, IG-1 |
| M2 | A rejected sweep is not rolled back. `failed_state.pt` holds the rejected iterate, which after an SPD failure means new means paired with old factors. `resume_simulation` accepts that file without complaint. | NUM-1, W2 |
| M3 | Status fields cannot report failure. `min_spd_status` is never assigned, and `accepted` is always True, even after an exhausted link backtrack. `estimator_error` is a placeholder, and a run whose link proposals all fail still reports "completed". scaling.md cites `accepted=1` and `spd_status=0` as runtime evidence. Descent itself holds: aborting on failure guarantees it, and the saved metrics confirm it. | C3, NUM-2, IG-3, Q5, T9 |
| M4 | `runtime.deterministic` is not reapplied on resume, yet the resumed config.json records it as True. | W3, Q2 |
| M5 | The tiny spatial presets have 0 spatial records, 0 non-self copies, and P=1. The CPU/CUDA spatial tests use one agent. Spatial inter-agent assembly therefore has no CPU or CUDA test, although it is correct when checked on nondegenerate configs. | W4, C4 |
| M6 | In spatial mode `communication.decay` is in physical units, while radius and alpha are in grid spacings. The spatial_2d_512 exponential prior is within 16% of uniform. | W5 |
| M7 | The learned/frozen source controls plot VFE, which the M-step lowers by construction, but omit fixture RMSE. RMSE is worse in the learned arm in 5 of 5 seeds. | C1 |
| M8 | Fixed-sector rank and the held-mode holonomy defect are geometry constants: computed once and identical across every saved run. They are nevertheless plotted every sweep and used as learned/frozen comparison observables. | C2 |
| M9 | In spatial_3d_512, analytics context 0 contains one agent, so the coherence, rank, and reciprocity values there are trivial. | C5 |
| M10 | The holonomy defect always reads the belief half of the generators. In a model-only communication run with unshared channels it reports generators that enter no factor. | G2 |
| M11 | Test gaps that let wrong engines pass: dropping sender variance (T2), dropping the sender information force (T3), transposing the twist for K≥2 (T4). Failure reports name no agent or channel, and NumericalMonitor is unreachable (T6). Several plan-required tests are missing; the list is in [lenses/tests.md](lenses/tests.md). | T2, T3, T4, T6 |
| M12 | Performance and memory: three `evaluate_vfe` calls per held sweep where one suffices (PERF-2). Zero-weighted loop exponentials in latent mode (PERF-3). The preflight estimate labeled "conservative" is exceeded by measured peaks by up to 3.7×, yet it gates admission (PERF-4). The point-lattice support builds dense N×N×d tensors, about 10 GiB at N=16,384 (PERF-5). | PERF-2 to PERF-5 |

### Low

- **Frames:** negative-determinant frames cannot be reached from config, and the frame keys affect display and export only; the docs imply more (G3, W6).
- **Tests that cannot fail:** the cocycle and reframing tests are identities that hold for any input (G4, T8). `checked_cholesky` loses the block index on non-finite input and does not read the upper triangle (NUM-3).
- **Aggregation launcher:** run_aggregate cannot aggregate GRID batches, and tuple-valued swept keys crash grouping (Q3, W8). Relative VFE can be compared across loop priors (IG-2).
- **Exchange payload:** it needs vfe4 and `weights_only=False` to load (Q4).
- **Latent transport Gram:** hard-coded for isotropic coordinate records; latent today (V2).
- **Config loader:** accepts malformed and mode-irrelevant keys (W7). run_scaling skips completed profiles by name only (W9).
- **Records:** build-completion.json is stale (C6). The R13 datum does not name H and omits the twist realization (C7). An SE with 1 d.o.f. is presented as an uncertainty (C8).
- **Weak tests:** the diagnostic-stream resume assertion is vacuous (T7), and the exchange test is N=1 only (T10).
- **Minor performance:** host syncs, uncached fixed terms, and dense node_blocks transients (PERF-6 to PERF-8).
- **Conventions:** drift across the supporting modules (Q6).

## Checked and found sound

- **Objective and updates (variational, tests):**
  - The objective equals an independent dense F.
  - Rows are the expected-cost categorical optimum, and the KL is unscaled by τ.
  - CAVI is the exact block minimizer, including sender terms, cross-node terms, and merged self maps.
  - Coloring is valid.
  - Damping and Fisher descent hold provably and are confirmed executed.
  - Record synthesis matches the joint.
  - No live recognition enters the likelihood.
- **Gauge structure:** ΔF = 0 exactly under per-agent and per-point GL changes, including det<0. Θ orientation matches c02. Copy and loop tables are correct. The coherent partition is gauge-inert.
- **Information geometry:**
  - The anchor KL is correct, and the estimators are unbiased.
  - Gradients and Hessian blocks match autograd and finite differences of the live objective.
  - The BLR step equals the Khan and Rue natural-parameter step.
  - log Z labeling is correct.
  - Backtracking uses common random numbers.
  - c09 Δ is exact, and the source step is the simplex-Fisher Euler step.
- **Numerics:**
  - There is no silent SPD repair anywhere.
  - matrix_exp inputs are contiguous.
  - The bump basis is stable down to 1−r = 1e-12.
  - The acceptance tolerance is scale-aware with no false rejections.
  - Float32 conditioning is κ ≤ 10.
- **Wiring:**
  - The active CONFIG equals the point_lattice_512 preset.
  - Unknown keys are rejected, and there is no CPU fallback.
  - The pure path runs in point, 2D, and 3D.
  - Resume is bitwise exact, including latent runs with source learning.
  - Aggregation refuses mixed configs.
- **Supporting modules:** exchange J, h, c matches dense conditioning to 6e-14, and checkpoints are atomic.
- **Claims:** JUnit counts, every scaling.md number, the 5×49 matching control leaves, and the absence of any causal formation claim all check out.

## Recommended order of repair

These are suggestions only; no repair is authorized by this audit.

1. **Wrong observables:** H1 (rank from fundamental-cycle holonomies) and H2 (exclude self rows), then M10.
2. **Run provenance:** H3, M2 and M4 (strict `load_runs`, roll back or save the pre-step state, reapply determinism), then M3 (live status fields, or drop them and correct scaling.md).
3. **Latent links (M1):** draw the fixture twist separately, and add configurable anchors.
4. **Tests:** H4 and M11 (engine counterexample, N≥3 identity, sender-force optimum, gradient against objective FD), plus nondegenerate tiny spatial presets (M5).
5. **Documentation and experiments:** M6 to M9 (decay units, RMSE in controls, constants recorded once, analytics context choice).
6. **Performance:** H5 and M12.
