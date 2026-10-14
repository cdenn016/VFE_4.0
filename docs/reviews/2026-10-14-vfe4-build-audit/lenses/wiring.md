# Lens: config wiring and reachability (audit-implementation-engineer, Claude Opus 5.5)

The scratch scripts are in scratchpad/wiring/.

## W1 (high): aggregation pools a resumed continuation as an extra run, and accepts runs without a summary
**Location:** aggregate.py:51-72; simulation.py:133-138; io.py:27-35.
- **Why:** a continuation is written into the batch root with the same config and seed. `load_runs` never checks seed uniqueness or lineage, and it checks status only when a summary.json exists.
- **Executed:** seeds [1,2] plus a resumed seed 1 pooled as 3 runs, with seeds [1,1,2]. At sweep 4, two of the three values were identical, so one seed was counted twice. Renaming summary.json did not stop the run from being pooled.
- **Failure scenario:** an OOM-killed batch run is resumed per README:34 and aggregated. The seed is double-counted and the BCa interval narrows.
- **Fix:** require a completed summary, reject duplicate seeds, and write continuations somewhere other than the batch root.
- Independently reported as Q1 (supporting lens).

## W2 (medium): failed_state.pt holds the rejected state, and resume continues from it silently
**Location:** inference.py:165-189; simulation.py:107-111.
- **Executed:** a forced failure at sweep 2 gave a checkpoint with sweep=2 but recorder rows [0,1]. `resume_simulation(failed_state.pt, 1)` was accepted and finished at sweep 3.
- **Why it matters:** this contradicts README:34 ("no ... silently turns a failed model into a successful run").
- **Fix:** save the pre-step state, or mark the checkpoint failed and refuse to resume it.
- The same root cause was reported as NUM-1.

## W3 (medium): runtime.deterministic is not reapplied on resume
**Executed:** after a fresh-process resume, the config says True but torch reports False.
- Independently reported as Q2.

## W4 (medium): the tiny spatial presets are degenerate, and no CPU test covers spatial inter-agent communication
**Location:** presets.py:49-50.
- **The presets:** tiny_spatial_2d and tiny_spatial_3d have 0 spatial records, 0 non-self copies and P=1.
- **The tests:** the CPU tests that do add cross-node structure use n_agents=1.
- **The pure path does run:** a nondegenerate config (grid 4^d, centers 2^d, patch 3^d) ran held, source_rate 0, full_patch and CAVI. It had 48 spatial records, 64 non-self copies (784 in 3D), P=9 and a cross-node covariance of 0.0217.
- Related to C4 (single-agent spatial CUDA tests).
- **Fix:** make the tiny spatial presets nondegenerate.

## W5 (medium): communication.decay is in physical units in spatial mode, while radius and alpha are in grid spacings
**Location:** supports.py:64-69 versus 24-28.
- **Executed:** on spatial_2d_512 with decay 1, the maximum pi_max/pi_min ratio in a row is 1.157 (median 1.115). The prior is nearly uniform, whereas the same decay in point mode spans weights from -1 to -24.
- **Fix:** divide by the minimum grid spacing, or document the unit.

## W6 (low): bundle.frame_* keys change no objective or metric
- Three frame settings produced identical metrics.
- In point mode, periodic_fourier is silently built as ConstantFrameField.
- The contract describes a Θ_ij(c) that the engine never builds.
- Consistent with G3. The objective is frame-inert by design because records live in reference coordinates.

## W7 (low): the loader admits malformed and mode-irrelevant keys
- `sweeps=2.5` or `True`, negative resolutions and a negative `coarse_max_dim` all pass the loader. A non-integer sweeps value then fails only after the run folder has been created.
- Point mode accepts spatial population keys without effect.

## W8 (low): run_aggregate.py has no swept_keys entry, so grid batches cannot be aggregated from it
Same as Q3.

## W9 (low): run_scaling skips a profile by name, not by config equality
Stored evidence can go stale after a preset edit. All five stored configs match the presets today.

## Checked and sound
- **Active CONFIG:** equals PRESETS['point_lattice_512'] (66 keys). Unknown keys are rejected, and there is no CUDA-to-CPU fallback.
- **Active run:** reaches full_patch CAVI, with held links and log_pi bitwise unchanged.
- **Pure path:** runs in point, 2D and 3D modes.
- **Resume:** a latent run with source learning and odd cadences, split at sweep 6, equals the uninterrupted run bitwise, including metrics.csv. Resume accepts no config, so the model and data cannot change through it.
- **Aggregation:** refuses differing configs, failed and stopped runs, and duplicate rows.
- **Formation arms:** differ only in source_rate.
