# Lens: test validity and silent failures (debugger, Claude Opus 5.5)

Each mutation below was applied through the pytest plugin `scratchpad/tests/mutate.py` (selected by `VFE4_MUTATION`). Without any mutation, 36 of 36 tests pass.

## T1 (high): the scheduling counterexample never runs the engine
- **Location:** tests/test_grounding.py:34-47.
- **Evidence:** the test uses only `schedule.colors` from the engine. It runs a hand-written Gauss-Seidel loop on a hand-built 3×3 J, which is not the plan's system (record N(y1+y2+y3, 5/3), variances 5/2).
- **Mutation `jacobi`:** freezing `mu_nodes` at the start of the sweep (simultaneous updates) still passes all 33 engine tests. The mutation is live: dmu = 0.25 and F changes.
- **Fix:** run `step` on the planned three-agent model and assert descent, then assert that putting everything in one color raises F.

## T2 (medium): the dense ELBO identity is tested only at N=1
- **Evidence:** at N=1 communication is constant: beta=1, KL=0, A−B=0. The test for expected-cost rows perturbs a receiver, which proves nothing about sender variance.
- **Mutation `drop_sender_var`:** all held-mode tests pass. At N=3 the identity gap under this mutation is −20.29, against 5.7e-14 for the real code.
- **Missing:** the plan's comparison against MultiAgentELBO's `assemble_network`, and `test_records.py`.

## T3 (medium): the sender-update test accepts a missing or sign-flipped sender force
- **Location:** tests/test_model_variants.py:42-54. The test asserts only that `norm(mu[1]) > 0`.
- **Mutation `drop_sender_info`:** passes. Agent 1's mean goes from [−0.147, −0.146] to [+0.037, +0.036].
- **Fix:** assert the closed-form block optimum, or dF/dmu_1 = 0.

## T4 (medium): transport orientation for K≥2 is not pinned
- **Evidence:** the finite-difference oracles differentiate `record_integrand` against itself, and the second-moment test uses K=1.
- **Mutation `transpose_twist`:** all 36 tests pass. A gradient-vs-FD-of-`evaluate_vfe` check catches it: max error 3.13, against 4e-8 for the real code.
- **Fix:** add that check as a test.

## T5 (high): the reciprocity observable counts self rows as reciprocal pairs
- **Location:** observables.py:81-88. For a self row the reverse key equals the pair key.
- **Executed (tiny_point):** the reported value is 0.3550; the mean over non-self mutual pairs is 0.5325 (= 0.3550·12/8).
- **Effect:** under the active config (include_self, radius 2) every recorded value, including the formation controls, is scaled by 4/5. No test covers this.
- **Fix:** add `& (local_r != local_s)` to the filter.

## T6 (medium): failure reports name no agent or channel, and NumericalMonitor can never fire
- **Evidence:** `checked_cholesky` reports a bucket-local index. Every non-finite or non-SPD state raises inside `step` before the monitor runs. An injected inf gives the summary error "Non-SPD Gaussian block at [[0]]".
- **Missing:** the plan's test that the summary names the sweep, agent and channel.

## T7 (low): the resume test's diagnostic-stream assertion is vacuous
`analytics.every=5` is longer than the 4-sweep test. Mutation `reseed_diagnostics` passes.

## T8 (low): the frame and reframing tests are identities; no passive-covariance test exists
Mutation `random_frames` passes.

## T9 (low): validations that check nothing
- `run_cpu_validation` makes no assertions.
- A run whose link retries are exhausted still reports "completed".
- The CUDA gate requires at least 9 of the 10 tests.

## T10 (low): the exchange test covers only N=1, K=1 with singleton rows
The code is correct on multi-agent cases with non-MAP labels: dh = 1.8e-15, dc = 4.5e-13.

## Plan-required tests that are missing or not established
- **Task 2:** nodal reproduction, positivity, ragged patches.
- **Task 3:** loop defect, and the prior families other than window.
- **Task 4:** records, passive covariance, parallel copies, the `assemble_network` comparison.
- **Task 5:** the engine counterexample.
- **Task 6:** the failed-SPD summary, termwise columns.
- **Task 7:** row statistics, reciprocity, field error.
- **Task 9:** gradient against the objective.
- **Task 13:** thin coverage.
- **Review Focus 3** is not established. **Review Focus 4** is partly vacuous.

## Checked and sound
- CAVI updates are exact block minimizers of `evaluate_vfe` (|dF| ≤ 2.3e-14).
- The `update_links` gradient matches FD of the objective.
- The dense identity holds at N>1, in spatial mode, and for both covariance structures.
- Exchange matches dense conditioning with arbitrary labels.
- The latent KL, second-moment, standard-error and quadrature oracles are independent.
- The viewer value tests and the held resume equality are real.
- The aggregation refusals are real.
- The CUDA lane checks device agreement only, which is what it claims.
