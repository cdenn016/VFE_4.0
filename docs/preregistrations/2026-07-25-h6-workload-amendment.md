# H6 Bounded Prefix Workload and Held-Out Opening Amendment

**Date frozen:** 2026-07-25
**Status:** normative, outcome-blind source amendment; creates no scientific
evidence

This amendment supersedes only conflicting H6-Prefix workload and H6-Prediction
held-out-opening clauses in `2026-07-21-h6-prefix-prediction.md` and
`2026-07-25-h6-audit-amendment.md`. All other preregistered H6 requirements
remain in force. It does not reduce the H6 training inventory.

## Bounded H6-Prefix workload

`H6PrefixWorkloadPlan` is the source-frozen workload contract. Its sole
exhaustive particle count is representative `N=128`; the production ladder is
`(128, 256, 512, 1024)`. The complete representative report covers all
`9,720 + 4,096 = 13,816` cases, with five prediction calls per case:

| Component | Cases | Prediction calls | Particle-call units |
| --- | ---: | ---: | ---: |
| Exhaustive representative report at `N=128` | 13,816 | 69,080 | 8,842,240 |
| Three stratified ladder subsets | 96 | 480 | 286,720 |
| Amended Prefix total | 13,912 | 69,560 | 9,128,960 |

The frozen small global indices are:

```text
0, 2186, 4373, 6560,
6561, 7289, 8018, 8747,
8748, 8990, 9233, 9476,
9477, 9557, 9638, 9719
```

The frozen validation global indices are:

```text
128, 384, 640, 896,
1152, 1408, 1664, 1920,
2176, 2432, 2688, 2944,
3200, 3456, 3712, 3968
```

At each of `N=256`, `512`, and `1024`, the exact 16-small plus 16-validation
stratified subset contributes 32 cases and 160 prediction calls. Its particle
work is therefore `160 * (256 + 512 + 1024) = 286,720` particle-call units.

`signature_and_identity` runs at every particle level. The
`dynamic_target_suffix_leakage` and `cache_identity` checks run exhaustively
at `N=128` and on those exact subsets at `N=256`, `512`, and `1024`.
`source_mask`, `case_inventory`, and `validation_data_safety` are exhaustive
representative-report checks only. Finite-SMC accuracy remains a separate
`N=256` gate. The stratified subsets are not estimator-accuracy evidence.

## H6-Prefix configuration binding

Development-only `h6-prefix-config-v1` accepts only `focused_subset`, with
exactly one `N=4` profile per semantic family, no authorization, and no
workload-plan field. Its `authorized_full` value is rejected before runner
dispatch; it cannot produce post-amendment evidence.

Bounded evidence uses `h6-prefix-config-v2` with `operation="H6-Prefix"`,
`execution_mode="authorized_full"`, the ordered `(128, 256, 512, 1024)`
profile ladder per exact semantic family, and raw
`workload_plan_sha256` equal to `H6PrefixWorkloadPlan().workload_plan_sha256`.
Resolution creates the exact typed plan and includes its full canonical payload
and digest in the resolved canonical configuration. The scientific config
stores only the SHA-256 of the exact authorization phrase
`AUTHORIZE_VFE4_H6_PREFIX_BOUNDED_WORKLOAD_V2`.

## Held-out opening scope, frozen but not implemented here

The exact held-out A0 endpoint is `h6-a0-transformer-v2`. It supplies eight
exact A0 corpus totals, one corpus-summed total for each of the eight frozen
seed checkpoints; it has no Monte Carlo half-width and no SMC bias bound.

The only weighted held-out endpoints are:

- `h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2`
- `h6-a5-structured-parent-specific-prefix-exact-emission-latent-smoothing-v2`

Each weighted endpoint has exactly `8 * 64 * 4 = 2,048` SMC corpus records.
Together they have 4,096 weighted-SMC corpus records and
`2 * 8 * 64 * (128 + 256 + 512 + 1024) = 1,966,080` particle streams. With
the eight exact A0 rows, the one opening produces exactly 4,104 unified
logical scoring rows.

The complete-A5 weighted records are reused unchanged for both `OBJECTIVE`
and `PRIMARY`; they are not separately remapped or rescored. `PRIMARY` is
the exact A0 corpus total minus the complete-A5 `Q2` result. All twelve
trained, validation, and disclosure endpoints remain retained, but the other
nine endpoints are forbidden from mapping held-out bytes.

The conservative simultaneous interval budget remains 352, with critical
value `4.5144904535377144`. These records retain the prior common-stream,
finite-SMC, and Q2 requirements; this amendment changes neither training
selection nor scientific decision rules.

## Boundary

This amendment creates no evidence and authorizes no inventory, training, test
opening, held-out byte mapping, or runner execution. Configuration resolution
and projection are pure, create no evidence, and still do not execute the
runner. The frozen records are source contracts only until a separately
authorized, exact-revision evidence operation opens the required work.
