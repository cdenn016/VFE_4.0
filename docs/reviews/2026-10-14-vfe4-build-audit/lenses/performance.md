# Lens: GPU performance and scaling (performance-engineer, Claude Opus 5.5)

All checks ran on CPU only: call, FLOP and sync counts, plus timings at tiny sizes. GPU numbers are quoted from docs/performance. No finding below changes a computed value; all of them affect speed or memory only.

## PERF-1 (high, contradicts spec §9): full patch covariances are rebuilt twice per color
- **Location:** inference.py:134-162 (`laws.marginals(S,K)[0]` at :138, `bucket.covariance()[selected]` at :154); state.py:96-108.
- **Evidence:** cost scales as O(colors·ΣD³) instead of O(ΣD³). The real presets have 24 / 24 / 8 / 15 colors. At spatial_2d_512 there are about 135 covariance materializations of shape (512,100,100) per sweep, about 138 GFLOP fp64, against roughly 3–4 N·D³ of intrinsic work. That is at least about 85 ms of a measured 399 ms sweep (inferred; no CUDA trace). Incoming messages are also recomputed for every incidence in each color.
- **Fix:** add a mean-only scatter, compute `chol[sel]@chol[sel].mT` for the selected rows only, and prefilter messages by color. The results are bit-identical on CPU.

## PERF-2 (medium): `evaluate_vfe` runs three times per held sweep
**Location:** inference.py:172 and 185; simulation.py:91.
- **Evidence:** the time outside the sweep interval is 88 ms per sweep (spatial 2D 512) and 68 ms (3D).
- **Fix:** pass `after_report` to the recorder, and seed the next `before` from it.

## PERF-3 (medium): latent mode evaluates loop exponentials multiplied by a zero coefficient
- **Location:** latent_links.py:95-113.
- **Evidence:** the point_512_uncertain profile runs 16,384 4×4 exponentials per evaluation and multiplies them by `mutual_loop_coefficient = 0`.
- **Fix:** return early when the coefficient is 0. Behavior changes only through NaN·0.

## PERF-4 (medium): the "conservative" preflight estimate is exceeded by measured peaks, yet it gates admission
- **Location:** preflight.py:28-36; simulation.py:55-58.
- **Evidence:** the measured peak divided by the prediction is 1.41× (point 512), 2.0× (batch) and 3.73× (uncertain). It over-predicts the spatial profiles by 2.2–2.3×. This contradicts spec §10, which says configurations must fail before a large allocation.
- **Fix:** calibrate the margins against the measured peaks, or drop the word "conservative".

## PERF-5 (medium): the point-lattice support builds dense N×N×d int64 tensors, even for window priors
- **Location:** supports.py:32.
- **Evidence:** peak memory is 640 MiB at N=4096 and would be about 10 GiB at N=16,384. Preflight does not count it.
- **Fix:** enumerate stencil offsets for the window and directed_window families.

## PERF-6 (low): about 176–224 host syncs per sweep
- **Location:** `checked_cholesky` per color, a per-color `nonzero` on a static schedule, and boolean-mask indexing on model-constant masks.
- **Note:** sync cost on CUDA was not measured.
- **Fix:** precompute the indices and masks, and check once per stage.

## PERF-7 (low): fixed record terms are recomputed every call
**Location:** the normalizer slogdet (8 × 166,400 per sweep on identical matrices), plus the held B and gram.
- **Note:** spec §9 says these are cached.

## PERF-8 (low): node_blocks still allocates dense (N,D,D) covariance and an unused Σ*
- **Location:** inference.py:129; state.py:79-87.
- **Note:** this undercuts the stated O(SK²) storage motivation.

## Checked and sound
- **Autograd:** no graph is retained, because `step` runs under no_grad.
- **Transport cache:** correctly keyed and invalidated.
- **matrix_exp:** the contiguous fix is present.
- **No dense global Λ:** the dense oracles are guarded.
- **Linear costs:** overlap_support is O(S+E).
- **One-time Python loops:** the scheduling and rank loops run once per run.
- **Viewer transfers:** only selected data.
- **Latent Hessian memory term:** adequate.
- **Spatial preflight:** conservative.
