# Lens: variational objective and block optimization (audit-variational, Claude Opus 5.5)

These checks use an independent dense assembler, scratchpad/variational/indep.py, which does not reuse `dense_action` or `bank_costs`.

## V1 (medium): latent-link recognition starts at the ground-truth twists that generated the records
- **Location:** model.py:204-209, 245-252, 255 (`initial_links=links`) and 263.
- **Evidence:** a single `omega` draw is used twice. It synthesizes the communication records through `fixed_links`, and it is returned as `initial_links`, which `initialize_sections` copies into the recognition state.
- **Executed check (leak.py):** `torch.equal(fixture twist, model.initial_links.omega_mean)` returns True.
- **Failure scenario:** every latent-mode run, including the point_512_uncertain profile in scaling.md, begins link inference at the truth. This contradicts spec §6: "twists ... never enter inference except through the resulting records". Held mode is correct, because there the twist is a known generative parameter.
- **Fix:** draw the fixture twist separately and initialize latent recognition at the prior mean, or from an independent stream.
- **Relation to IG-1:** that lens found the zero prior anchor. Together these are the same mismatch between the latent link prior, its initialization, and the data-generating twist.

## V2 (low): the latent transport Gram hard-codes an isotropic coordinate-projection record
- **Location:** model.py:69, `G = diag([1/tau]*R + [0]*(K-R))`.
- **Evidence:** this equals key^T P key only for the current banks, where the executed check matches to 1e-14.
- **Failure scenario:** a candidate-specific noise or a rectangular S^K map, both permitted by spec §6, would silently give wrong costs and a wrong sender precision.
- **Fix:** compute G_f = key_f^T P_f key_f per record.

## Checked and sound (executed, float64)
- **evaluate_vfe against the independent dense F:** the absolute difference was at most 7.3e-12 (on F ≈ 6606) across these configurations:
  - point mode with self copies, both channels unshared;
  - projection with R=1;
  - exponential and power priors;
  - spatial 2D with full_patch and with node_blocks;
  - a wrap cycle with max_sources=1;
  - 3D with uncovered contexts.

  Candidate costs matched to 1.4e-14.
- **Categorical KL:** it is not scaled by tau; tau enters only through the record precision and normalizer.
- **optimal_rows:** it uses expected costs, not plug-in values. Every simplex perturbation tested increased F.
- **CAVI:** each color's update equals the dense block optimum, to 3.3e-16 on means and 7e-17 on covariances. This includes sender terms, internal cross-node terms and merged self maps. Under node_blocks the mean equals the joint optimum. No perturbation of mean or covariance decreased F, in held mode or in latent mode.
- **Coloring:** no coupled external record shares a color, across 96, 512 and 736 records tested.
- **Damping and the Fisher path:** per-color ΔF ≤ 0 on every probe. Damping descends by joint convexity in (μ, M2). The Fisher path descends because KL to the block optimum decreases along natural-parameter interpolation.
- **Source learning:** ΔF < 0 at fixed Q for rates 0.1 to 1, and the support stays positive.
- **No live recognition in the likelihood:** records are fixed when the model is built. log_pi changes only through the declared M-step.
- **Record synthesis:** one Gumbel-max label and one record per row and channel, with noise std √tau. The record is broadcast across candidates, which is consistent with c06.
- **Existing tests:** 7 pass.
