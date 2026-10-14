# VFE_4.0 build handoff, October 14, 2026

The October 13 revision-2 first release is implemented and its required execution work is complete. It supports GL(K,R) agents at a single context or as local sections over flat periodic 2D/3D bases, with fixed frames inducing the base connection, independent directed interaction generators, held or Gaussian latent links, and the declared collective VFE. Curved bases and other groups remain extension work, as scoped in the design.

Open [run_simulation.py](../../run_simulation.py), select C:/anaconda/python.exe, edit PRESET and OVERRIDES, and press Run. The preserved active dictionary selects point_lattice_512, K=4, CUDA and 100 sweeps. Choose spatial_2d_512 or spatial_3d_512 for the spatial modes. [README](../../README.md) explains batches, aggregation, source controls, figures and resume; [model contract](../model-contract.md) records equations, objective status, observable scope, extension boundaries and configuration execution paths.

The final broader CPU pass has 51 passing cases, zero failures/errors/skips, in 2.726 seconds; the explicit CUDA lane has 10 passing cases, zero failures/errors/skips, in 0.730 seconds. They are separate lanes: CUDA was selected through its runner and C:/anaconda/python.exe on the RTX 5090. See [CPU JUnit](build-integration.xml), [CUDA JUnit](cuda.xml) and [completion record](build-completion.json). No passing checks need another run absent relevant changes.

| Profile | Completed sweeps | Recorded elapsed seconds | Peak allocated MiB |
|---|---:|---:|---:|
| Point, 512 agents | 100 | 6.661 | 75.488 |
| Periodic 2D, 512 agents | 100 | 48.655 | 781.391 |
| Periodic 3D, 512 agents | 100 | 25.654 | 493.166 |
| Sparse periodic 2D, 1024 agents | 20 | 2.811 | 278.542 |
| Uncertain point links, 512 agents | 20 | 5.560 | 135.754 |

[Performance report](../performance/scaling.md) links the actual configurations, preflight predictions, GPU environments, metrics, checkpoints, termwise objectives, reference-frame field errors and allocated/reserved peaks. These elapsed values include run orchestration and are not kernel-only timings. Sequential held profiles share allocator cache, so reserved peaks are process measurements. The five-seed point batch and bootstrap aggregate also completed. No population, K, covariance family or source support was silently reduced.

The uncertain profile first failed at sweep zero because a strided K=4 sample chunk reached Torch's matrix exponential. The tiny CPU/CUDA reproduction and regression identified that path. Passing the same matrices in contiguous storage repaired it; the failed run and failed summary remain preserved. Completed profiles were skipped when the runner retried this unfinished profile.

All ten default learned/frozen source-control runs completed 100 sweeps. Five paired seeds have identical saved geometry, observations and synthetic fixtures; resolved configurations differ only in source_rate and output directory. [Matching record](source-controls.json) records 49 matching tensor leaves per pair and all 21 PDF/PNG/SVG comparison figures. [Comparison design](../../runs/formation-comparison/design.json) names the actual arms. The VFE comparison image was inspected. These are descriptive deterministic mean-field controls, without a stochastic formation or RG claim.

The real Qt viewer's pause, resume, one-sweep and stop controls were exercised with synthetic Matplotlib mouse events, including stopped-run checkpoint persistence. The corrected sampled/relative objective labels, first-point marker and control clearance pass their regressions. [Viewer record](viewer-validation.json) and [inspected snapshot](viewer-snapshot.png) preserve that evidence. Manual human interaction remains untested; sampled links retain their stated numerical expectation boundary and nonzero loop coupling retains its unknown log normalizer.

The [offline wheel](../../artifacts/package/vfe4-0.1.0-py3-none-any.whl) contains all 33 current package modules byte-for-byte and passed isolated imports. Its SHA-256 is f34615bd3f04020a353ea9953cc9f441d09bccb9ca0b1176e44eba08d208a427; [packaging record](package-validation.json) contains commands and outputs. Automatic approval review rejected deletion of generated build/ and vfe4.egg-info folders as "blocked by policy"; they remain recorded. This cleanup limitation does not affect the validated wheel.

Implementation coverage follows all 13 plan tasks: configs/domains; packed patches/sections; frames/copies/loops; normalized factors/VFE; colored and Fisher matter updates; run artifacts/resume; scoped observables; seed batches/bootstrap; latent-generator updates; matched source learning; figures/viewer; CUDA/scale measurements; conditioned context exchange/documentation. The final integration cases cover the tiny formula, gradient, scheduling, covariance, family, continuation and exchange obligations; the saved large runs provide population evidence. Earlier CPU figures and conditioned exports remain named in [cpu-artifacts.json](cpu-artifacts.json).

No required first-release work remains. The completion goal is complete and the existing continuation heartbeat is paused. Active configuration, previous run data, sibling repositories and Research-vault sources were preserved. The implementation phase ended without initializing Git or publishing. The user subsequently authorized commit/push/merge/fast-forward to https://github.com/cdenn016/VFE_4.0; publication includes source and compact evidence while large binary run archives remain local. Future group/curvature implementations, manual viewer use, new scientific experiments and Research-wiki ingestion are separate work.
