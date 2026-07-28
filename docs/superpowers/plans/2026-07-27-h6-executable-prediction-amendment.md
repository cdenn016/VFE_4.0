# H6 Executable Prediction Amendment Implementation Plan

**Date:** 2026-07-27

**Design authority:** `docs/superpowers/specs/2026-07-27-h6-executable-prediction-amendment-design.md`

**Goal:** Replace the fail-closed H6-Prediction v2 buildout surface with an
additive, click-to-run v3 implementation that has a normalized mixed
recognition law, exact finite source reduction, one-sample pathwise continuous
estimation, deterministic CUDA training, exact checkpoint hydration, sealed
WikiText-2 reopening, validation, and the frozen 4,104-row held-out contract.

**Completion boundary:** This plan completes and verifies source behavior with
synthetic fixtures and exactly two tiny CUDA tests. It does not download a
corpus, launch the production training campaign, reserve the one-time held-out
opening, activate H8, or claim a scientific result.

## Execution discipline

- Run every Torch/model test with `C:/anaconda/python.exe`.
- Start each task with its exact focused RED nodes, implement only that task,
  then run the same nodes GREEN once.
- Use a unique ignored `.verification/pytest-h6-v3-taskN` basetemp.
- Do not run the repository-wide suite between tasks.
- Do not run real corpus, H8 profiler, H8 scientific children, or held-out
  operations during development.
- Keep v1/v2 readers intact. A legacy type may be readable but cannot
  authorize v3.
- Use independent implementation review after each coherent milestone rather
  than repeating tests.
- Commit only after a task or coherent two-task milestone is GREEN and
  reviewed.

## Task 1: Freeze additive v3 configuration and core record types

**Files**

- Create `vfe4/types/h6_prediction_v3.py`
- Modify `vfe4/types/__init__.py`
- Modify `vfe4/config/schema.py`
- Modify `vfe4/config/resolve.py`
- Modify `vfe4/config/__init__.py`
- Create `tests/unit/test_h6_prediction_v3_config.py`

**Required records**

- `H6PredictionV3ResolvedConfig`
- `H6PredictionRuntimeIdentity`
- `H6RecognitionEstimatorSpec`
- `H6PredictionV3ReadinessToken`
- `H6AttemptSpecV3`
- `H6AttemptCursorV3`
- `H6ObjectiveManifestV3`
- v3 schema constants and canonical domains from the design

The resolver must bind the recognition family, terminal source mixture,
estimator, counter domain, matching policy, runtime/determinism policy,
checkpoint codec, scoring inventory, and expected row count 4,104. It must
round-trip its canonical JSON and reject unknown/missing fields.

**Focused tests**

- `test_v3_config_requires_recognition_estimator_runtime_and_checkpoint_identity`
- `test_v3_config_binds_expected_test_row_count_4104`
- `test_v2_config_is_readable_but_cannot_authorize_v3_execution`
- `test_v3_resolver_rejects_partial_or_legacy_matching_identity`
- `test_v3_attempt_cursor_binds_next_phase_and_counter_coordinates`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_prediction_v3_config.py `
  -q --basetemp .verification/pytest-h6-v3-task1
```

## Task 2: Implement receiver trajectories and terminal source mixture

**Files**

- Create `vfe4/recognition/h6_prediction_v3.py`
- Modify `vfe4/recognition/__init__.py`
- Modify/extend, but do not weaken, `vfe4/recognition/parameter_store.py`
- Create `tests/unit/test_h6_recognition_v3.py`

**Required behavior**

- Build filtering/smoothing receiver contexts with the frozen sinusoidal
  position descriptor.
- Emit one base Gaussian for receivers `0..T-1`.
- Emit the normalized terminal component family
  `q(y_T | a_T=j,b_T=k,x)`.
- Own categorical residual vectors, lag scalars, and channel-specific
  source-shift vectors exactly once in
  `LanguageRecognitionParameterStore`, which remains the sole recognition
  parameter owner.
- Represent an absent bank as a typed probability-one singleton with no
  trainable parameters and no entropy contribution.
- Normalize ragged source rows over positive-prior support only.
- Treat the mean-evaluated prior as a stopped-model-gradient recognition
  feature, never as the live generative factor.
- Emit immutable identities without detaching the live training graph.
- Do not route v3 through the existing one-law
  `LanguageRecognitionParameterStore.recognition_law()` API; expose a distinct
  receiver-trajectory/component interface so the terminal mixture cannot
  collapse to one Gaussian.

**Focused tests**

- `test_receiver_trajectory_filtering_never_reads_future_tokens`
- `test_smoothing_contexts_are_receiver_distinct`
- `test_terminal_component_rows_are_normalized_and_non_degenerate`
- `test_earlier_receivers_are_source_independent`
- `test_source_rows_reject_zero_prior_mass_and_future_parents`
- `test_recognition_parameter_inventory_has_one_owner_per_live_bank`
- `test_absent_source_bank_is_parameter_free_entropy_free_singleton`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_recognition_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task2
```

## Task 3: Implement counter noise and the executable mixed ELBO

**Files**

- Create `vfe4/training/h6_noise_v3.py`
- Create `vfe4/objective/h6_prediction_v3.py`
- Modify `vfe4/objective/__init__.py`
- Create `tests/unit/test_h6_objective_v3.py`

**Required behavior**

- Reuse the exact `EstimatorStream` uint64/open-uniform/Box-Muller mapping
  under `vfe4.h6.training-rmc-normal.v1`.
- Generate one CPU float64 base draw per receiver, example, and phase, then
  transfer it to the active device.
- Evaluate source-independent receivers with exact single-row reductions.
- Evaluate the terminal receiver with the explicit finite
  `sum_(j,k) beta_T(j) gamma_T(k)` reduction.
- Evaluate the live generative source prior on the sampled earlier history.
- Implement the projected endpoint's moment covariance, Cholesky sampler,
  analytic entropy, and component-KL upper-bound record.
- Make the canonical ordered total and independent total the same live scalar.
- Expose the positive ELBO; optimizers later minimize its negative.

**Focused tests**

- `test_counter_normal_is_stable_and_execution_order_independent`
- `test_terminal_beta_gamma_sum_matches_monolithic_log_ratio_oracle`
- `test_exact_source_law_is_an_evaluated_record_not_a_marker`
- `test_projected_terminal_sampler_includes_between_component_covariance`
- `test_projected_entropy_and_component_kl_bound_match_hand_oracle`
- `test_elbo_gradients_reach_only_the_requested_parameter_block`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_objective_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task3
```

## Task 4: Add the deterministic CUDA training runtime

**Files**

- Create `vfe4/training/h6_runtime_v3.py`
- Create `vfe4/training/h6_transformer_v3.py`
- Retain `vfe4/training/h6_transformer.py` as the CPU reference/scoring path
- Create `tests/unit/test_h6_runtime_v3.py`

**Required behavior**

- Set or validate `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
  initialization.
- Require deterministic algorithms, deterministic cuDNN with benchmark off,
  TF32/reduced-precision reductions off, and math SDPA only.
- Record the exact Python, Torch full version, CUDA runtime, device name,
  compute capability, dtype, and live deterministic settings.
- Create canonical CPU float64 initialization, hash it, then move the training
  copy to `cuda:0` before optimizer construction.
- Refuse missing CUDA, unsupported deterministic operations, runtime drift,
  or fallback.
- Provide an injectable synthetic CPU runtime only for bounded unit tests.

**Focused tests**

- `test_cpu_reference_remains_strict_cpu_float64`
- `test_v3_runtime_identity_requires_cuda0_float64_policy`
- `test_v3_runtime_refuses_unavailable_or_mismatched_cuda_identity`
- `test_v3_cpu_canonical_initialization_is_stable`
- `test_v3_runtime_never_installs_or_selects_another_torch`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_runtime_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task4
```

## Task 5: Recompute matching v3 and readiness

**Files**

- Modify `vfe4/training/parameter_counts.py`
- Create `vfe4/training/h6_matching_v3.py`
- Modify `vfe4/training/h6_readiness.py`
- Create `tests/unit/test_h6_matching_v3.py`
- Create `tests/unit/test_h6_readiness_v3.py`

**Required behavior**

- Freeze `h6-amended-matching-policy-v3`.
- Count `B*(R+1+d)` new recognition parameters with the design's arm-bank
  inventory.
- Add named arithmetic terms for categorical logits, rank-one shifts,
  terminal component realization, exact local sums, projection, backward,
  clipping, and AdamW.
- Preserve the v2 candidate grids, tolerances, passes, batch policy, and
  outcome-blind first-lexicographic selection.
- Keep data I/O, validation, checkpoint serialization, test scoring, device
  transfer, prediction particles, and cache work explicitly excluded.
- Bind the regenerated v3 matching set, estimator, runtime, checkpoint, data,
  and prerequisite identities into readiness.

**Focused tests**

- `test_matching_v3_counts_terminal_source_parameters_by_arm`
- `test_matching_v3_has_complete_named_estimator_flop_terms`
- `test_matching_v3_exclusion_inventory_is_exact`
- `test_readiness_v3_rejects_v2_matching_set`
- `test_readiness_v3_binds_runtime_estimator_and_checkpoint_identities`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_matching_v3.py `
  tests/unit/test_h6_readiness_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task5
```

## Task 6: Reopen the authenticated sealed WikiText-2 store

**Files**

- Create `vfe4/data/h6_sealed_store_v3.py`
- Modify `vfe4/data/wikitext2.py`
- Modify `vfe4/data/access.py`
- Create `tests/unit/test_h6_sealed_reopen_v3.py`

**Required behavior**

- Implement
  `reopen_authenticated_blinded_store_v3(manifest_path, artifact_root)`.
- Revalidate the exact archive/member, tokenizer, tokenized split, fixture,
  window, and enclosing manifest identities before constructing a store.
- Register a fresh opaque store handle with train/validation capabilities.
- Keep test bytes sealed and never reconstruct an opening capability.
- Refuse partial roots, symlinks/reparse substitutions, hash drift, consumed
  reservations, redownload, overwrite, or repair.

**Focused tests**

- `test_reopen_revalidates_manifest_inventory_and_tokenized_splits`
- `test_reopen_returns_registered_store_without_test_rows`
- `test_reopen_refuses_partial_hash_mismatch_or_consumed_opening`
- `test_reopen_never_downloads_replaces_or_repairs_files`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_sealed_reopen_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task6
```

## Task 7: Implement checkpoint v3 encoding and exact hydration

**Files**

- Create `vfe4/training/checkpoint_v3.py`
- Create `tests/unit/test_h6_checkpoint_v3.py`
- Retain `vfe4/training/checkpoint.py` as the v2 reader

**Required behavior**

- Canonically encode sorted named module buffers/parameters and stable named
  AdamW state/group records.
- Store dtype, shape, row-major little-endian bytes, length, and SHA-256.
- Reject aliases, sparse tensors, unsupported dtypes, duplicate/case-colliding
  names, unknown state, and unbound optimizer entries.
- Hydrate in the exact CPU-module → named groups → CPU state → validation →
  device move order.
- Restore pass, batch, next phase, example/sample ordinal, counter block and
  digest, permutation, and validation/checkpoint boundaries.

**Focused tests**

- `test_checkpoint_v3_canonicalizes_named_module_and_optimizer_state`
- `test_checkpoint_v3_rejects_duplicate_or_aliased_tensor_names`
- `test_fresh_cpu_hydration_restores_named_optimizer_groups`
- `test_cursor_restores_next_phase_batch_and_counter_coordinates`
- `test_resume_rejects_runtime_or_determinism_drift`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_checkpoint_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task7
```

## Task 8: Implement planner and phase-owned training engine

**Files**

- Create `vfe4/training/h6_engine_v3.py`
- Create `vfe4/training/h6_experiment_v3.py`
- Keep `vfe4/training/h6_experiment.py` as the v2 fail-closed path
- Create `tests/unit/test_h6_engine_v3.py`
- Create `tests/unit/test_h6_experiment_v3.py`

**Required behavior**

- Reconstruct the 12 exact endpoints, matching reports, six-cell tuning plans,
  two tuning seeds, eight confirmatory seeds, schedules, and attempt specs
  without corpus or outcome input.
- For a latent batch: recognition forward, minimize `-ELBO`, step; fresh
  deterministic post-step recognition forward; immutable detached snapshot;
  model forward with distinct noise, minimize `-ELBO`, step.
- For A0/no-latent: one CE/NLL model phase.
- For `emission_only_ablation_non_elbo`, minimize only the negative live
  emission objective in both active phases, retain `is_elbo=False`, and never
  assemble source, transition, initial, or Gaussian-entropy terms into its
  optimization scalar.
- Validate finite gradients and clip once per active phase.
- Resume exactly at the next recorded phase without replaying an update.
- Publish only terminal/declared boundary checkpoints.

**Focused tests**

- `test_plan_v3_emits_exact_endpoint_attempt_and_schedule_inventory`
- `test_recognition_step_cannot_mutate_model_parameters`
- `test_snapshot_is_fresh_post_step_complete_and_detached`
- `test_model_step_cannot_mutate_recognition_parameters`
- `test_emission_only_endpoint_optimizes_only_live_emission_and_is_not_elbo`
- `test_tiny_cpu_resume_matches_uninterrupted_terminal_bytes`
- `test_train_refuses_readiness_or_matching_identity_drift`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_engine_v3.py `
  tests/unit/test_h6_experiment_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task8
```

## Task 9: Implement CPU validation and checkpoint selection

**Files**

- Create `vfe4/training/h6_validation_v3.py`
- Create `vfe4/artifacts/h6_prediction_v3.py`
- Modify `vfe4/artifacts/__init__.py`
- Create `tests/unit/test_h6_validation_v3.py`

**Required behavior**

- Reconstruct a fresh CPU float64 scoring model from checkpoint v3.
- Use only target-blind prior prediction; recognition cannot enter scoring.
- Publish all tuning validation records, apply the frozen mean/tie-break rule,
  and bind the selected cell and complete checkpoint set.
- Reject any test capability, test bytes, incomplete endpoint inventory, or
  stale runtime/matching/data identity.

**Focused tests**

- `test_validation_scores_only_authorized_validation_capability`
- `test_validation_uses_fresh_cpu_model_from_checkpoint_v3`
- `test_tuning_selection_applies_frozen_mean_and_tie_break`
- `test_checkpoint_selection_binds_complete_inventory`
- `test_validation_cannot_consume_test_opening`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_validation_v3.py `
  -q --basetemp .verification/pytest-h6-v3-task9
```

## Task 10: Enforce the 4,104-row held-out transaction

**Files**

- Create `vfe4/training/h6_test_transaction_v3.py`
- Extend `vfe4/artifacts/h6_prediction_v3.py`
- Modify `vfe4/data/access.py`
- Modify `vfe4/artifacts/atomic.py` only if the existing no-replace primitive
  cannot express the journal transition
- Create `tests/unit/test_h6_test_transaction_v3.py`
- Create `tests/unit/test_h6_prediction_v3_artifacts.py`

**Required behavior**

- Validate complete eligibility before the exclusive reservation.
- Persist the bound `RESERVED` journal, consume the process-local capability,
  score, and finish as `FINALIZED` or terminal `INCONCLUSIVE`.
- Never claim cross-directory atomic rename.
- Accept exactly eight exact A0 totals plus 2,048 complete-A5 and 2,048
  emission-A5 weighted rows.
- Freeze weighted A5 particle levels `(128,256,512,1024)`, 64 replicate
  streams per seed/endpoint, and the common-stream identities across the two
  weighted endpoints.
- Forbid particle count, replicate stream, Monte Carlo half-width, SMC bias
  bound, or weighted-estimator fields on exact A0 records.
- Reuse complete A5 rows for PRIMARY and OBJECTIVE.
- Reject legacy 24,576 inventories, weighted A0, exact A5, other endpoints,
  duplicate keys, or another opening.
- Publish metrics/result/pointers with no-replace semantics.

**Focused tests**

- `test_transaction_validates_eligibility_before_reservation`
- `test_reservation_binds_config_data_checkpoints_and_inventory`
- `test_crash_after_reservation_is_terminal_inconclusive`
- `test_raw_inventory_accepts_exactly_4104_discriminated_rows`
- `test_weighted_a5_rows_require_frozen_particles_replicates_and_common_streams`
- `test_exact_a0_rows_reject_every_weighted_estimator_field`
- `test_complete_a5_rows_are_reused_without_rescoring`
- `test_final_result_and_pointer_are_no_replace_published`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_h6_test_transaction_v3.py `
  tests/unit/test_h6_prediction_v3_artifacts.py `
  -q --basetemp .verification/pytest-h6-v3-task10
```

## Task 11: Wire the click-only launcher and H8 v3 adapter

**Files**

- Modify `train_vfe4.py`
- Modify `verification/h8_gate.py`
- Modify `verification/h8_preflight.py`
- Modify `verification/run_gates.py`
- Modify `vfe4/types/h8.py`
- Modify `docs/preregistrations/2026-07-21-h8-sparse-scale.md`
- Modify `docs/superpowers/plans/2026-07-21-vfe4-h8-sparse-scale.md`
- Create `verification/h8_h6_prediction_v3.py`
- Create `tests/unit/test_train_vfe4_h6_v3_click.py`
- Create `tests/unit/test_h8_h6_prediction_v3_adapter.py`
- Modify `tests/unit/test_h8_preflight.py`
- Modify `tests/unit/test_structural_types.py`

**Required behavior**

- Populate one editable top-level v3 `CONFIG` dictionary.
- Retain `prediction_readiness`, `plan`, `train`, and `score_validation` with
  their existing exact authorization phrases.
- Remove split v3 test operations and expose only
  `score_test_transaction` with
  `AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1`.
- Require no command-line arguments or hidden scientific environment values.
- Route lazily to v3 without importing CUDA before runtime policy setup.
- Add an exact H8 v3 predecessor reference/reopen validator and a new
  `h8-current-candidate-refs-v4` registry/preflight path. Amend the H8
  preregistration and frozen-section hash before any H8 run. Leave v1/v2/v3
  registry and H6-Prediction v2/legacy parsers unchanged, readable only as
  historical records, and unable to stand in for v4.

**Focused tests**

- `test_click_launcher_accepts_only_v3_operation_inventory`
- `test_click_launcher_retains_prediction_readiness_authorization`
- `test_click_launcher_rejects_both_legacy_split_test_operations`
- `test_test_transaction_requires_exact_new_authorization`
- `test_click_launcher_requires_no_cli_arguments`
- `test_h8_reopens_and_validates_h6_v3_result`
- `test_h8_rejects_v2_or_drifted_v3_predecessor`
- `test_h8_preflight_requires_registry_v4_and_prediction_v3`
- `test_h8_registry_v1_v2_v3_remain_legacy_read_only`

**Command**

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/unit/test_train_vfe4_h6_v3_click.py `
  tests/unit/test_h8_h6_prediction_v3_adapter.py `
  tests/unit/test_h8_preflight.py::test_h8_preflight_requires_registry_v4_and_prediction_v3 `
  tests/unit/test_structural_types.py::test_h8_registry_v1_v2_v3_remain_legacy_read_only `
  -q --basetemp .verification/pytest-h6-v3-task11
```

## Task 12: Bounded integration, independent review, and two CUDA tests

**Files**

- Create `tests/integration/test_h6_prediction_v3_fixture.py`
- Create `tests/cuda/test_h6_prediction_v3_cuda.py`
- Create ignored claim/review evidence under `.verification/`

**Fixture milestone**

Use a tiny synthetic sealed store and two tiny endpoints. Exercise:

```text
resolve v3
  -> matching/readiness
  -> plan
  -> tiny train + checkpoint
  -> fresh CPU validation
  -> fake one-time transaction
  -> v3 result/pointer
  -> H8 v3 adapter reopen
```

The fixture may use arithmetic inventory counts rather than materializing
4,104 production rows.

**Focused CPU nodes**

- `test_h6_v3_click_fixture_reaches_closed_result_and_h8_adapter`
- `test_h6_v3_fixture_refuses_identity_drift_at_each_boundary`
- `test_every_v3_dispatcher_rejects_v1_v2_types_before_effects`

Run the fixture nodes, then obtain independent implementation and adversarial
review. Apply only concrete repairs and rerun only affected focused nodes.

**Final CUDA nodes**

- `test_cuda_a0_uninterrupted_and_resume_are_byte_identical`
- `test_cuda_a5_recognition_snapshot_model_and_resume_ownership`

Before those two nodes, verify:

```powershell
& 'C:/anaconda/python.exe' -c `
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Then run exactly:

```powershell
& 'C:/anaconda/python.exe' -m pytest `
  tests/cuda/test_h6_prediction_v3_cuda.py::test_cuda_a0_uninterrupted_and_resume_are_byte_identical `
  tests/cuda/test_h6_prediction_v3_cuda.py::test_cuda_a5_recognition_snapshot_model_and_resume_ownership `
  -q --basetemp .verification/pytest-h6-v3-task12-cuda
```

After both CUDA nodes pass, validate the implementation claim ledger on the
same unchanged revision. Any source repair invalidates the CUDA evidence and
requires rerunning only the affected CUDA node before ledger closure.

Do not activate H8 or start corpus training from this task. Commit the reviewed
H6 source candidate. The separately authorized production H6 campaign and
frozen H1-H7/H6-Prediction prerequisite sequence then produce the exact
candidate evidence needed for H8. Run and close H8 v5 before beginning the
post-H8 WikiText-103 implementation. Later WikiText-103 work must not rewrite
the frozen H8 implementation or relabel the historical H8 artifact as evidence
for a different source revision.

## Post-plan boundary

WikiText-103 is a separate post-H8 schema and implementation plan. It must not
reuse the H6 WikiText-2 data identity, opening, row counts, or readiness
artifacts. Its own click-to-run acquisition, tokenizer/cache, schedules,
training arms, recording, and figures begin only after the H8 v5 artifact and
ledger close.
