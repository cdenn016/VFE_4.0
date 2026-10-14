# CUDA and population-scale measurements

Snapshot read at 2026-10-14T19:16:57.832334+00:00. The authoritative statuses are saved in [scaling.json](scaling.json); this report uses its referenced resolved configurations, preflight records, run environments, summaries, and CSV metrics. Source JSON SHA-256: `79f9ce6e7a30ab2f2b6f0b04e9ca6bd0cb76429c10ad55f339e37914a4d938bf`.

All 5 named numerical profiles and the five-seed point batch are completed in this snapshot. The earlier uncertain-link failure is retained and described below. Requested sweeps and preflight predictions are kept separate from completed measurements.

## CUDA lane

The saved [CUDA status](../implementation/cuda-status.json) is completed. [JUnit XML](../implementation/cuda.xml) contains 10 tests, 0 failures, 0 errors, and 0 skips. The root task ran the CUDA lane; this report preparation did not run tests or CUDA work. Completed run environments record `C:/anaconda/python.exe`, Torch `2.10.0.dev20251210+cu128`, CUDA runtime 12.8, selected device `cuda`, and NVIDIA GeForce RTX 5090.

## Configured profiles

All profiles configure GL(4,R), float64, seed 20261013, chromatic CAVI, product recognition between agents and channels, and full covariance within each patch channel. The admission budget is 12 GiB (12,884,901,888 bytes). Source-prior learning is disabled. No population, fiber dimension, covariance family, or support reduction was applied to obtain these completed results.

| Profile | Status | N | Agent lattice or base grid | Patch shape | Links | Requested/completed sweeps | Samples |
|---|---|---:|---|---|---|---:|---:|
| point_lattice_512 | completed | 512 | 32 x 16 | point | held | 100/100 | not used |
| spatial_2d_512 | completed | 512 | 32 x 32 | 5 x 5 | held | 100/100 | not used |
| spatial_3d_512 | completed | 512 | 16 x 16 x 16 | 3 x 3 x 3 | held | 100/100 | not used |
| spatial_2d_1024_sparse | completed | 1024 | 32 x 32 | 3 x 3 | held | 20/20 | not used |
| point_512_uncertain | completed | 512 | 32 x 16 | point | latent_gaussian | 20/20 | 4 |

Physical-agent center grids are 32 x 16 for `spatial_2d_512`, 8 x 8 x 8 for `spatial_3d_512`, and 32 x 32 for `spatial_2d_1024_sparse`, separate from their base grids. The uncertain point profile uses radius-one support (maximum degree five including self), four antithetic samples, diagonal per-copy generator precision, shared channel generators, copy chunks of 64, and zero mutual-loop coefficient.

## Preflight predictions

The configured-design work estimates below are predictions, not measured allocator peaks. E counts candidate edges across both channels; patch dimension is the maximum flattened patch-channel dimension.

| Profile | Incidences S | Candidate edges E | Maximum row degree | Patch dimension | Predicted bytes | Predicted MiB |
|---|---:|---:|---:|---:|---:|---:|
| point_lattice_512 | 512 | 13312 | 13 | 4 | 55,967,744 | 53.375 |
| spatial_2d_512 | 12800 | 332800 | 15 | 100 | 1,871,052,800 | 1784.375 |
| spatial_3d_512 | 13824 | 128000 | 8 | 108 | 1,115,291,648 | 1063.625 |
| spatial_2d_1024_sparse | 9216 | 92160 | 5 | 36 | 516,685,824 | 492.750 |
| point_512_uncertain | 512 | 5120 | 5 | 4 | 38,141,952 | 36.375 |

Saved free memory before execution: `point_lattice_512`: 32,192,331,776 bytes (29.981 GiB); `spatial_2d_512`: 31,686,918,144 bytes (29.511 GiB); `spatial_3d_512`: 31,101,812,736 bytes (28.966 GiB); `spatial_2d_1024_sparse`: 31,101,812,736 bytes (28.966 GiB); `point_512_uncertain`: 32,200,720,384 bytes (29.989 GiB). These are admission-time observations, not peak usage.

## Measured completed profiles

Elapsed time is copied from run summaries. Mean sweep time is calculated from nonempty CSV `sweep_seconds` rows and excludes costs outside those intervals. Allocated and reserved peaks are recorded CUDA process-allocator measurements. The held profiles and seed batch share a sequential process and allocator cache; their reserved peaks are not isolated model-memory measurements. The repaired uncertain profile was run separately. These peaks are distinct from preflight estimates and total board occupancy.

| Profile | Sweeps | Elapsed seconds | Mean recorded sweep seconds | Peak allocated bytes (MiB) | Peak reserved bytes (MiB) |
|---|---:|---:|---:|---:|---:|
| point_lattice_512 | 100 | 6.660910 | 0.059787 | 79,155,200 (75.488) | 98,566,144 (94.000) |
| spatial_2d_512 | 100 | 48.654595 | 0.398573 | 819,347,968 (781.391) | 1,052,770,304 (1004.000) |
| spatial_3d_512 | 100 | 25.653701 | 0.188567 | 517,121,536 (493.166) | 1,052,770,304 (1004.000) |
| spatial_2d_1024_sparse | 20 | 2.810612 | 0.117630 | 292,072,448 (278.542) | 1,052,770,304 (1004.000) |
| point_512_uncertain | 20 | 5.559823 | 0.235853 | 142,348,800 (135.754) | 159,383,552 (152.000) |

## Objective, field error, and numerical status

| Profile | Initial VFE estimate | Final VFE estimate | Initial/final fixture-frame RMSE | Accepted sweeps | Recorded SPD statuses | Retries |
|---|---:|---:|---|---:|---|---:|
| point_lattice_512 | 134094.760185 | 9689.417391 | 0.994083 / 0.614064 | 100/100 | 0 | 0 |
| spatial_2d_512 | 5714808.859364 | 294391.414617 | 0.732809 / 0.173090 | 100/100 | 0 | 0 |
| spatial_3d_512 | 6589175.094887 | 331783.095160 | 0.368962 / 0.134321 | 100/100 | 0 | 0 |
| spatial_2d_1024_sparse | 3647735.401886 | 306576.588395 | 0.370765 / 0.192811 | 20/20 | 0 | 0 |
| point_512_uncertain | 133609.150559 | 16025.352342 | 0.994083 / 0.674402 | 20/20 | 0 | 0 |

All completed sweeps record `accepted=1`, `spd_status=0`, and zero retries. No paired before/after objective increase exceeding the recorded acceptance tolerance is present. These are saved runtime indicators, not independent minimum-eigenvalue measurements. The four held profiles record `normalizer_status=exact` and zero estimator error. The uncertain profile records `estimated_vfe_known_normalizer`; its generic `estimator_error` field is `nan`, while the separate diagnostic stream reports finite uncertainty.

At uncertain sweep 20, the paired integration estimate is 16025.352342; the independent diagnostic mean is 16027.379506, with reported standard error 1.797552 from 2 antithetic pair means. The zero loop coefficient makes the prior normalizer known; it does not make the sampled likelihood expectation exact. Acceptance on common samples does not establish monotonicity of independent estimates.

Field RMSE is against each profile's synthetic fixture; point and spatial profiles use different fixture families. Final signed objective contributions are copied from the final CSV rows:

| Term | point_lattice_512 | spatial_2d_512 | spatial_3d_512 | spatial_2d_1024_sparse | point_512_uncertain |
|---|---:|---:|---:|---:|---:|
| model prior | 3555.058086 | 85138.030920 | 90120.040301 | 62265.684136 | 3592.864404 |
| bridge | 1558.290580 | 23737.623782 | 23865.368393 | 55199.801609 | 4102.919581 |
| recognition entropy | -1139.457523 | 13299.860519 | 19690.848351 | -36475.466681 | -2946.056891 |
| categorical KL | 1645.142783 | 1860.088034 | 1379.327212 | 13875.437517 | 1229.567025 |
| private records | -276.200991 | -6977.575285 | -7863.714395 | -5132.153488 | -271.647826 |
| spatial records | 0.000000 | 68626.759203 | 86846.089843 | 137522.027117 | 0.000000 |
| communication records | 4346.584455 | 108706.627444 | 117745.135455 | 79321.258186 | 6774.111863 |
| generator KL | not recorded (held) | not recorded (held) | not recorded (held) | not recorded (held) | 3543.594186 |
| loop prior | not recorded (held) | not recorded (held) | not recorded (held) | not recorded (held) | 0.000000 |

## Uncertain-link failure and repair

The [preserved failed summary](../../runs/scaling/point_512_uncertain/20261014T191125Z-point-d65caf20/summary.json) records a sweep-zero `RuntimeError`: `view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.`. The root task reproduced the strided K=4 `matrix_exp` chunk failure on CPU and CUDA and changed `SimulationModel.transport_moments` to pass `matrices.contiguous()` into `torch.matrix_exp`. This materializes the same matrices in a contiguous layout; it does not change the generative or recognition model. The failed and completed resolved configuration JSONs compare equal (true). The added regression is covered by the refreshed 10-test CUDA lane.

The [repaired summary](../../runs/scaling/point_512_uncertain/20261014T191425Z-point-5ff89772/summary.json) records 20 completed sweeps in 5.559823 seconds. The old failed attempt had no complete runtime or peak-memory result; it remains a failed attempt alongside the completed replacement.

All named Task 12 profiles now have timing and allocated/reserved peak measurements. Kernel-level hotspot timings and a `node_blocks` alternative measurement are not supplied by these artifacts. No kernel profiler or hotspot optimization result is claimed.

## Five-seed point batch

The saved batch is completed for seeds 20261013, 20261014, 20261015, 20261016, 20261017. Each run configures `point_lattice_512`, N=512, K=4, float64, held links, full-patch recognition, and 100 sweeps. The [aggregate CSV](../../runs/scaling/seeds/batch-754bf8ea/aggregate.csv) records the seed-level objective mean and bootstrap interval at each sweep.

| Seed | Status | Sweeps | Elapsed seconds | Allocated peak bytes | Reserved peak bytes | Final VFE | Final RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20261013 | completed | 100 | 6.381249 | 112,054,272 | 1,052,770,304 | 9689.417391 | 0.614064 |
| 20261014 | completed | 100 | 6.341330 | 112,049,152 | 1,052,770,304 | 9680.597853 | 0.660945 |
| 20261015 | completed | 100 | 6.293107 | 112,054,784 | 1,052,770,304 | 9723.649399 | 0.675189 |
| 20261016 | completed | 100 | 6.433383 | 112,050,688 | 1,052,770,304 | 9660.951461 | 0.676850 |
| 20261017 | completed | 100 | 6.315112 | 112,055,296 | 1,052,770,304 | 9728.794603 | 0.622891 |

At sweep 100, the saved mean VFE is 9696.682141, with saved interval [9674.503204, 9719.155253] over 5 runs. This describes this configured batch; it is not a scaling-law fit or a matched speedup between different profiles.

## Source artifacts

point_lattice_512: [summary](../../runs/scaling/point_lattice_512/20261014T190959Z-point-014f4bc0/summary.json), [resolved config](../../runs/scaling/point_lattice_512/20261014T190959Z-point-014f4bc0/config.json), [preflight](../../runs/scaling/point_lattice_512/20261014T190959Z-point-014f4bc0/preflight.json), [environment](../../runs/scaling/point_lattice_512/20261014T190959Z-point-014f4bc0/environment.json), [metrics](../../runs/scaling/point_lattice_512/20261014T190959Z-point-014f4bc0/metrics.csv).

spatial_2d_512: [summary](../../runs/scaling/spatial_2d_512/20261014T191006Z-flat_torus-b325cae9/summary.json), [resolved config](../../runs/scaling/spatial_2d_512/20261014T191006Z-flat_torus-b325cae9/config.json), [preflight](../../runs/scaling/spatial_2d_512/20261014T191006Z-flat_torus-b325cae9/preflight.json), [environment](../../runs/scaling/spatial_2d_512/20261014T191006Z-flat_torus-b325cae9/environment.json), [metrics](../../runs/scaling/spatial_2d_512/20261014T191006Z-flat_torus-b325cae9/metrics.csv).

spatial_3d_512: [summary](../../runs/scaling/spatial_3d_512/20261014T191055Z-flat_torus-1b6ebb03/summary.json), [resolved config](../../runs/scaling/spatial_3d_512/20261014T191055Z-flat_torus-1b6ebb03/config.json), [preflight](../../runs/scaling/spatial_3d_512/20261014T191055Z-flat_torus-1b6ebb03/preflight.json), [environment](../../runs/scaling/spatial_3d_512/20261014T191055Z-flat_torus-1b6ebb03/environment.json), [metrics](../../runs/scaling/spatial_3d_512/20261014T191055Z-flat_torus-1b6ebb03/metrics.csv).

spatial_2d_1024_sparse: [summary](../../runs/scaling/spatial_2d_1024_sparse/20261014T191121Z-flat_torus-929dd5b5/summary.json), [resolved config](../../runs/scaling/spatial_2d_1024_sparse/20261014T191121Z-flat_torus-929dd5b5/config.json), [preflight](../../runs/scaling/spatial_2d_1024_sparse/20261014T191121Z-flat_torus-929dd5b5/preflight.json), [environment](../../runs/scaling/spatial_2d_1024_sparse/20261014T191121Z-flat_torus-929dd5b5/environment.json), [metrics](../../runs/scaling/spatial_2d_1024_sparse/20261014T191121Z-flat_torus-929dd5b5/metrics.csv).

point_512_uncertain: [summary](../../runs/scaling/point_512_uncertain/20261014T191425Z-point-5ff89772/summary.json), [resolved config](../../runs/scaling/point_512_uncertain/20261014T191425Z-point-5ff89772/config.json), [preflight](../../runs/scaling/point_512_uncertain/20261014T191425Z-point-5ff89772/preflight.json), [environment](../../runs/scaling/point_512_uncertain/20261014T191425Z-point-5ff89772/environment.json), [metrics](../../runs/scaling/point_512_uncertain/20261014T191425Z-point-5ff89772/metrics.csv).

This report is scoped to Task 12 and section 14 of the suite design. Coherence partitions remain descriptive mean-field observables; these finite runs do not establish formation, universality, RG exponents, or autonomous coarse dynamics.
