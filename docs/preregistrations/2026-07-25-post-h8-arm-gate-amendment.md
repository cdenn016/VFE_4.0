# Post-H8 WikiText-103 Arm and Gate Inventory Amendment

Date frozen: 2026-07-25

Status: source protocol only. This amendment authorizes no data acquisition,
source lock, training, test opening, or figure rendering. Those operations
remain blocked until an exact-revision H8 PASS and the post-H8 readiness
contract both exist.

## Precedence

This amendment supersedes every independently entered two-arm count, endpoint
count, tuning-attempt count, scoring-record count, result-row count, and figure
series count in
`docs/superpowers/plans/2026-07-21-vfe4-post-h8-wikitext103-training.md`.
The scientific PRIMARY comparison remains A0 versus the parent-specific
pooled-prefix complete-objective endpoint. The additional rows are controls and
gates; they do not widen the PRIMARY claim.

## Immutable arm inventory

The ordered minimum inventory is:

1. `WT103-A0-AR-v1`
2. `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1`
3. `WT103-A5-FIXED-COMPLETE-v1`
4. `WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1`
5. `WT103-A5-NOLATENT-v1`

The source implementation must represent each row as an immutable
`WT103ArmSpec`. At minimum it binds:

```text
arm_id
factory_id
training_objective
prior_variant
latent_enabled
recognition_enabled
scorer_kind
tuning_grid_id
confirmatory_seed_ids
terminal_checkpoint_role
result_role
nonclaims
arm_spec_sha256
```

`result_role` is one of `PRIMARY_REFERENCE`, `PRIMARY_ENDPOINT`,
`PRIOR_CONTROL`, `OBJECTIVE_GATE`, or `LATENT_PATH_CONTROL`.

The exact roles are:

- A0 autoregressive: `PRIMARY_REFERENCE`.
- Parent-specific pooled-prefix complete: `PRIMARY_ENDPOINT`.
- Fixed-prior complete: `PRIOR_CONTROL`; it changes the joint, so it is not a
  prior-only causal estimate unless a later intervention record proves every
  other field identical.
- Parent-specific pooled-prefix emission-only: `OBJECTIVE_GATE`; it shares the
  selected parent-specific capacity and prior with the complete endpoint and
  differs only in `training_objective`.
- No-latent: `LATENT_PATH_CONTROL`; it is a bundled control unless later
  evidence proves literal held-fixed semantics.

The remaining source literals are frozen as follows. Every row uses
`tuning_grid_id="wt103-six-cell-v1"`, confirmatory seeds
`2026072101..2026072108`, and
`terminal_checkpoint_role="terminal_scoring"`.

| `arm_id` | `factory_id` | `training_objective` | `prior_variant` | latent / recognition | `scorer_kind` | `result_role` |
|---|---|---|---|---|---|---|
| `WT103-A0-AR-v1` | `build_wt103_a0@wt103-arm-v1` | `cross_entropy` | `absent` | false / false | `exact_autoregressive` | `PRIMARY_REFERENCE` |
| `WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1` | `build_wt103_a5_parent_specific@wt103-arm-v1` | `complete_elbo` | `parent_specific_pooled_prefix` | true / true | `weighted_smc` | `PRIMARY_ENDPOINT` |
| `WT103-A5-FIXED-COMPLETE-v1` | `build_wt103_a5_fixed@wt103-arm-v1` | `complete_elbo` | `fixed` | true / true | `weighted_smc` | `PRIOR_CONTROL` |
| `WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1` | `build_wt103_a5_parent_specific@wt103-arm-v1` | `emission_only_ablation_non_elbo` | `parent_specific_pooled_prefix` | true / true | `weighted_smc` | `OBJECTIVE_GATE` |
| `WT103-A5-NOLATENT-v1` | `build_wt103_a5_nolatent@wt103-arm-v1` | `cross_entropy` | `absent` | false / false | `exact_autoregressive` | `LATENT_PATH_CONTROL` |

The complete and emission-only parent-specific rows must share the selected
capacity allocation, model-family identity, normalized prior law, scorer,
particle protocol, optimizer policy, data order, and checkpoint schedule.
Their only scientific intervention is `training_objective`. The no-latent row
is exact only because its frozen factory has no marginalized latent variable;
it may not silently retain an inactive recognition store or use SMC as no-op
work.

Each latent row additionally freezes `source_mixture="exact"`,
`recognition_family="structured_block_tridiagonal_smoothing"`,
`recognition_iterations_per_batch=1`, and update phases
`("recognition_adam_proposal","immutable_detached_snapshot",
"model_adam_proposal")`. A0 and no-latent freeze these as `"absent"`, `0`, and
`("model_ce_adam_proposal",)`. These fields are owned by `WT103ArmSpec`; the
shared experiment profile may not contain a singular A5 prior, objective,
recognition, or update default.

## Immutable gate inventory

`WT103GateSpec` records the ordered logical gates and their dispositions:

```text
SOURCE_LOCK
H8_EXACT_REVISION
POST_H8_READINESS
OBJECTIVE
PRIMARY
PRIOR_CONTROL
LATENT_PATH_CONTROL
```

`OBJECTIVE` is adjudicated before `PRIMARY`. Failure or inconclusiveness of
`OBJECTIVE` prevents a PRIMARY scientific claim without preventing durable
retention of already produced records. Control rows never rescue, reverse, or
promote PRIMARY.

## Derived endpoint inventory

`EndpointInventory.create(arms, gates, tuning_cells, tuning_seeds,
confirmatory_seeds, estimator_protocol)` is the sole source of:

- tuning attempts;
- terminal checkpoints;
- validation and test endpoints;
- SMC/exact scoring records;
- resource-forecast work counts;
- finalized result rows;
- figure panels and ordered series.

Its canonical payload contains the ordered arm and gate hashes, seed and
particle inventories, scorer applicability, checkpoint roles, and derived
counts. `endpoint_inventory_sha256` binds every experiment plan, reservation,
checkpoint, raw score record, final table, and figure specification.

No consumer may accept a separately entered arm count, endpoint count, record
count, result-row count, or figure-series count. A mismatch between a derived
count and observed bytes is a hard validation failure. A missing, duplicate,
nonfinite, failed, or inapplicable-without-reason endpoint makes the associated
gate INCONCLUSIVE and cannot be repaired by reopening the test split.

## Attribution and claim boundaries

The PRIMARY claim is exactly a training-compute-matched whole-architecture
comparison between A0 and the parent-specific pooled-prefix complete endpoint.
Inference-inclusive compute is reported separately and never changes match
eligibility.

The fixed-prior, emission-only, and no-latent rows retain distinct labels in
tables, figures, captions, alt text, and machine-readable artifacts. They may
not be collapsed into one “VFE” series. The V3 complete-objective-versus-CE
observation is a provenance-bounded design risk only; it is not VFE4 evidence.

## Geometry and reach boundary

The first WikiText-103 profile remains depth 1 and freezes:

```text
training sequence length L = 128
block width b = 40
training population dimension D = L*b = 5,120
H8 synthetic N = 129
H8 synthetic population dimension = 5,160
A5 direct source lookback W = 20
A0 direct attention reach = full causal 128
```

The unbanded dense `O(D^3)` calculation is a counterfactual, not measured H8 or
WikiText-103 runtime. The implemented width-20 algorithm must expose every
lag-1 through lag-20 cross-moment needed by exact transition expectations or
declare an approximation and its measured error. Finite direct-address width
does not imply absence of longer-range influence because state can propagate
recursively.

## Click-run and execution boundary

`train_vfe4.py` remains import-safe and uses one editable dictionary with
`operation="idle"` by default and explicit
`source_lock|readiness|train|resume` operations.
`generate_vfe4_figures.py` remains a separate import-safe editable dictionary.
Neither launcher uses a product CLI.

No WikiText-103 loader, download, cache creation, source lock, training engine,
test opening, or figure render may begin until an exact H8 PASS exists for the
same implementation revision and this endpoint inventory is frozen in source.
