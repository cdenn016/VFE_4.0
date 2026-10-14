# Lens: information geometry and uncertain links (audit-info-geometer, Claude Opus 5.5)

The scratch scripts are scratchpad/infogeo/check1-6.py.

## IG-1 (medium): the generator prior anchor is fixed at α_e=0 with C_e=σ²I, so the c08b held-link limit is unreachable
- **Location:** latent_links.py:74-85 and 157-159; model.py:204-209.
- **Evidence:** spec §8 and c08b eq:link-gaussian-kl specify the anchors N(α_e, C_e). The code always uses N(0, σ²I).
- **Held mode:** the random twists `initial_twist_std·randn` are both the held links and the generator of the communication data.
- **Executed check:** shrinking `prior_std` (1e-1, 1e-2, 1e-3) drives the latent mean to 0, not to the held twist; ‖latent−held‖ stays near 0.34 at every value. The zero anchor is disclosed in model-contract.md:25.
- **Failure scenario:** a latent run compared with the held run as the "faithful reference" is compared against a different generative model.
- **Fix:** add per-copy anchors (defaulting to the held omega_mean or to 0) and a configurable C_e.

## IG-2 (low): no guard against comparing relative VFE across loop priors
`aggregate_metric` will group `vfe_total` across `links.mutual_loop_coefficient` values even when rows carry `normalizer_status = relative_vfe_unknown_logZ`. Spec §8 forbids that comparison.

## IG-3 (low): `accepted` stays True after an exhausted link backtrack
**Location:** inference.py:190; latent_links.py:204-214.
- **Evidence:** after 12 failed backtracks the update is rolled back, but the row still records accepted=1, with retries=12 as the only signal.
- **Fix:** add a separate `link_accepted` flag.

## Checked and sound (executed)
- **link_kl:** matches torch kl_divergence for both the diagonal and full families.
- **Sampling:** antithetic pairing is exact, and the sample covariance matches P⁻¹.
- **Transport moments:** the E[exp Ω] and E[exp Ωᵀ G exp Ω] estimators are unbiased (max z = 1.28).
- **Objective:** the estimated objective equals the sample mean of the per-sample integrand.
- **Derivatives:** gradients and Hessian blocks equal autograd and finite differences of the live `evaluate_vfe`, to 1.7e-13 and 6.8e-9 across all channel, family and loop variants. The diagonal and full projections are the correct mean-field BLR projections.
- **BLR step:** equals the natural-parameter BLR step of Khan and Rue (2023), with residual shrinking as Monte Carlo error does.
- **Normalizer labels:** log Z labeling is correct.
- **Backtracking:** uses common random numbers, invalidates the cache, rejects non-SPD proposals and halves the step, and keeps diagnostics on a separate stream.
- **Matter Fisher path:** natural-parameter interpolation toward the exact block optimum. KL(q‖q*) is nonincreasing along the segment. The fisher + node_blocks combination is rejected.
- **c09 Δ:** equals the direct conditional expectation to 4.4e-16. Its availability gating is correct.
- **Source step:** equals the simplex-Fisher natural-gradient Euler step to 5.6e-17.
