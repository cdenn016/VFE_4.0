# Lens: claims vs evidence (audit-philosophy-of-science, Claude Opus 5.5)

The scratch script was scratchpad/claims/mono.py; the probes ran on CPU only.

## C1 (medium): matched source controls omit the fixture RMSE, which is worse for the learned arm in 5 of 5 seeds
- **Location:** experiments/run_formation.py:43-44; build-handoff.md:21; source-controls.json.
- **Evidence:** the comparison keys omit `field_rmse_reference`. Final RMSE, learned vs frozen: 0.672/0.614, 0.720/0.661, 0.705/0.675, 0.743/0.677, 0.671/0.623. Learned final VFE is about 7,900 to 7,970 and frozen about 9,660 to 9,730, and almost all of that gap is the categorical KL (about 205 vs about 1,650), which the M-step minimizes by construction. The VFE figure that was "inspected" could not have gone the other way.
- **Fix:** add `field_rmse_reference` to the comparison keys, and state the construction effect and the 5/5 RMSE result in the handoff.
- **Falsifies:** a seed pair with learned RMSE at or below frozen RMSE.

## C2 (medium): the fixed-sector rank is a geometry constant, computed once and cached
- **Location:** vfe4/analytics/observables.py:46-50; holonomy.py:8-49; model-contract.md:41.
- **Evidence:** the rank depends only on held B maps and edge lists; it reads neither recognition nor beta. Every one of the 37 saved runs records a single value: 0 for point runs, 2 for tiny 2D/3D, and 4 for spatial_3d_512. The held-mode holonomy defect is also constant per run. It is nonetheless plotted per sweep and used in the learned/frozen comparison.
- **Fix:** record it once as metadata and drop it from the per-sweep and control figures, or define a beta-weighted Laplacian instead.
- **Falsifies:** a held run with two distinct values.

## C3 (medium): `min_spd_status` is never assigned and `accepted` is always 1, yet scaling.md cites both
- **Location:** vfe4/inference.py:25 and 187-190; scaling.md:53-61.
- **Evidence:** nothing writes `min_spd_status`. Any VFE increase raises instead of recording `accepted=0`. The status columns therefore cannot read otherwise. Descent itself holds: a scan of all 37 runs found zero increases.
- **Fix:** compute the status from the Cholesky info, or delete the column. Attribute the evidence to abort-on-failure plus completion.

## C4 (medium): the CUDA spatial agreement tests use 1 agent and 0 inter-agent records
- **Location:** tests/test_cuda.py:39-63; progress.md:97.
- **Evidence:** the tests override `n_agents=1`. Building the fixtures confirms 0 inter-agent records in both tiny spatial presets. Spatial sender terms, overlap rows, coloring and beta routing were never compared between CPU and CUDA.
- **Fix:** use at least 2^d agents with overlapping patches.

## C5 (medium): analytics context 0 in spatial_3d_512 contains a single agent
- **Location:** observables.py:32; spatial_3d_512 metrics.csv.
- **Evidence:** `analytics_context_agents=1`, rank = K, every coherent-block count = 1, reciprocity = 0. These are trivial values. The 2D runs have 15 agents (512) and 9 agents (1024) at context 0.
- **Fix:** select the context with maximum incidence, or a configured one, and record the choice.

## C6 (low): build-completion.json is stale
It records a 33-module wheel with SHA f34615bd…, while the disk wheel is 4b952424… with 34 modules. It also records 51 CPU cases, superseded by 73.

## C7 (low): the R13 datum is under-specified
The symmetry group H (translations of the 32×16 torus) is not named, and the seeded twist realization, which also breaks H, is omitted. Initial means are exactly zero, so sweep-0 coherence equals 1 by initialization.

## C8 (low): uncertain-link standard error with 1 degree of freedom
The SE of 1.797552 in scaling.md:63 comes from 2 antithetic pair means, without noting the 1 degree of freedom.

## Checked and sound
- JUnit counts and times; every scaling.md number.
- Zero VFE increases across all 37 runs.
- 5×49 bitwise-equal control leaves, with config differences limited to source_rate and directory.
- The source step is the simplex-Fisher natural-gradient Euler step of KL(beta||pi).
- 66 DEFAULTS keys; the active CONFIG.
- No causal formation, RG or learning claim in the docs.

**Verdict:** the numeric claims reproduce, and held descent holds everywhere. Several recorded observables and status fields could not have come out differently, and the one control metric with an external referent is omitted.
