# Lens: supporting modules (code-reviewer, Claude Opus 5.5)

The scratch scripts are in scratchpad/quality/.

## Q1 (medium): aggregation pseudo-replicates resumed and summary-less runs
- **Location:** aggregate.py:28-31, 48-72; simulation.py:133-138; io.py:28-34.
- **Evidence:** `load_runs` drops the seed before comparing configs and never rejects duplicate seeds. It checks the summary only if one exists. A resumed continuation lands as a sibling folder with an identical config and carries the parent's metric rows. Executed: 3 independent seeds plus 2 resumed copies plus 1 interrupted run gave n_runs = [5, 5, 2].
- **Fix:** require summary.json, reject duplicate seeds, and record `resumed_from`.

## Q2 (medium): `runtime.deterministic` is lost on resume
- **Location:** config.py:119; io.py:71-76.
- **Evidence:** the flag is applied only inside `load_config`. A fresh-process resume printed config deterministic True with torch deterministic False.
- **Fix:** call `torch.use_deterministic_algorithms(...)` after `load_checkpoint`.

## Q3 (low): GRID batches cannot be aggregated from the launcher
- **Evidence:** run_aggregate.py has no `swept_keys` entry, so loading fails closed. When `swept_keys` is passed directly, a tuple-valued swept key arrives as a JSON list and raises TypeError (unhashable).
- **Fix:** add `swept_keys` to the launcher and convert lists to tuples.

## Q4 (low): the exchange payload needs vfe4 and `weights_only=False` to load
- **Evidence:** a `weights_only=True` load fails, and MultiAgentELBO cannot load the payload without vfe4. `GaussianAction(J, h, c)` itself works.
- **Fix:** serialize plain tensors and dicts, or document the dependency.

## Q5 (low): the `estimator_error` column is a placeholder
- **Evidence:** it is always NaN in latent mode and 0 in held mode. The real standard error is recorded as `diagnostic_vfe_standard_error`.

## Q6 (low, aggregated): convention drift
- **Weak or lying type hints:** batch.py:9, aggregate.py (return types also carry an int `n_runs`), visualization.py:39, sections.py:105.
- **Docstrings:** missing on io, batch, load_config, monitor and recorder.
- **Shape comments:** none in exchange.context_snapshot.
- **Signature layout:** the ordering convention is followed only in section_view.py.

## Checked and sound
- **Exchange J, h, c:** matches dense conditioning of `dense_action` with beta one-hot, to 1e-16 (J, h) and 6e-14 (c), across full_patch/node_blocks, full_state/projection, and point mode with unshared channels.
- **Conditioned recognition:** matches the dense oracle to 3e-17. `dense_action` equals VFE minus entropy to 5e-13.
- **Checkpointing:** atomic saves; RNG streams live in the state; `_transport_cache` is excluded.
- **Monitor:** raises on non-finite values and never repairs.
- **Paths and labels:** Windows-safe paths; correct frame labels in figures and the viewer.
- **Tests:** 29 tests pass.
