# L6-formation: meta-agent formation theory (c13, ca_formation_proofs, formation statements of c14)

Reviewer: Claude Opus 5.5 (investigator lens, not adjudicated), 2026-10-12.

## 1. Scope and verdict

Read in full: `c13_formation.tex` (1-1577); `ca_formation_proofs.tex` 1593-2411 (Gaussian core across an early interface, the endogenous four-cycle, the proof of `prop:phil-seeded-contrast-formation`, depth one under exclusive gating); `c14_interpretation.tex` 1-420 (the statements of `prop:phil-seeded-contrast-formation`, `conj:phil-contrast-formation`, `cor:phil-joint-selection` and their surrounding prose); `ca_selection_proofs.tex` (all 824 lines, read for statement matching only, since the proof belongs to another lens); `ca_joint_selection.tex` (all); `ca_temperature_bounds.tex` (all). Skimmed only: `ca_formation_proofs.tex` 1-1592, the finite-depth lemma chain, which the ledger records as "proof reconstructed" at sketch level for 07j and App. A (X07/E5/E6, 2026-09-23). I did not re-derive those lemmas and did not re-open them, and I did not re-litigate the settled rows for the carrier algebra, evaluator optimum, restricted VFE, Fisher metric, parent potential, dissipation, rate composition, or the proof chain. The Gaussian core (`prop:rg-formation-gaussian-core` (b),(c)), the endogenous four-cycle (`prop:rg-formation-endogenous-cycle`), the gated family (`prop:rg-gated-depth-one`), the seeded-contrast proposition, the joint-selection corollary, and the temperature-tracking proposition postdate those ledger rows (none of them is in the 2026-09-23 copy of 07j) and were checked here.

Verdict: in this scope I found no false statement and no failed load-bearing proof step. Every closed-form identity I recomputed holds to round-off. The Gaussian-core asymptotic is reproduced numerically, with the ratio to the predicted value tending to one. The adiabatic escape time has the stated form log(1/eps)/alpha plus a constant. The c14 statements of the selection theorem and its corollary match what `ca_selection_proofs.tex` and `ca_joint_selection.tex` prove, including the corollary's window constant 2, which alpha >= 7/5 justifies. The quantifier-order part of ledger item M5 (epsilon_L fixed before k_L) is repaired in the current text: `thm:finite-depth-adaptive-formation` now fixes epsilon_L after k_L and eta_L (c13:912-914, 1063-1066). The findings below are one framing inflation (medium) and five low items. The four-cycle formation results are path-level (seeded by the initial assignment) and the selection results are law-level statements about the one-time assignment law. The manuscript states both scopes correctly, except as noted in L6-01.

## 2. Findings

### L6-01: "Idle source learning" is stronger than what the seeded-contrast proposition proves
**Location:** `c14_interpretation.tex:183` (heading of `prop:phil-seeded-contrast-formation`), `c14:7-11`, `c14:245-248`, `c01_program_scope.tex:45-47`; proof `ca_formation_proofs.tex:2067-2227`.
**Severity:** medium
**Evidence:** Part (b) proves a lower bound on Pr(contrast formation) that is the same for learned and frozen rows. The proof uses learning only through the row-ratio bound 0.5511/0.4489 < 1.23 on the seed interval [0, t°_*], with t°_* = 2 log(1/eps) + 1 and eps t°_* <= 0.1021. The contrast event, however, may occur on any unit interval inside [0, 2/eps], and part (a) bounds the rows only by |pi_ij(t) - 1/2| <= (1 - e^{-eps t})/2, which reaches 0.43 at t = 2/eps. Nothing in the proposition compares the learned and frozen probabilities of contrast formation. The coupling estimate of the proof gives TV(q^pi_t, q^{1/2}_t) <= (2/c_0) int_0^t |pi - 1/2|_inf ds <= eps t^2/(2 c_0), which is of order 1/eps at t = 2/eps and so vacuous on the stated horizon. The manuscript itself records that equality of the learned and frozen first-block laws is proved only for eps_L = o(T^{1/4}) (`prop:phil-early-block-null`, c13:1172-1177, c14:315-325), and that the proposition's own regime eps_L = T^{1/8} remains open (c13:1198-1200, c14:355-356). Rate check: with t = 2 T^{-1/8}, eps_L t^2 = 4 eps_L T^{-1/4}. This tends to zero exactly when eps_L = o(T^{1/4}), which agrees with that null result and shows why the seeded regime falls outside it. In terms of mechanism, at T = eps^8 the posterior is an argmin of edge costs except where a receiver's two costs agree within T log(prior ratio). That happens for interior receivers of coherent blocks, whose two attended residuals are both at thermal scale, and there the learned prior decides the source. So no argument given makes learning idle beyond the seed interval.
**Fix:** Retitle the proposition, for example to "Seeded contrast formation with a learning-independent lower bound". In c14:7-11, c14:245-248 and c01:45-47, say that the lower bound and the seed mechanism on [0, t°_*] do not depend on learning, and that equality of the learned and frozen formation probabilities at eps_L = T^{1/8} is open, pointing to the existing open statement.
**Falsifies:** A proof that, at eps_L = eps and T = eps^8 on the horizon [0, 2/eps], the learned and frozen probabilities of contrast formation (or the laws of the contrast event) agree or differ by o(1).
**Status:** ESTABLISHED. The mathematical content is correct; the heading and the two prose summaries are inflated.

### L6-02: The seeded lower bound is about 29 orders of magnitude below the actual probability
**Location:** `c14:215-226` (part (b) of `prop:phil-seeded-contrast-formation`); `c14:7-11`; `c01:45-47`.
**Severity:** low
**Evidence:** The bound is (e^{-1/4}/16) times the N(m0, I_12) mass of two balls of radius 1/100 in R^12. Exact values from the noncentral chi-square distribution are 1.06e-30 at m0 = x°, 2.86e-31 at m0 = 0, and 1.55e-36 at a random m0. I ran a Monte Carlo of the frozen-row kinetics (script `seeded_contrast_mc.py`, n = 400 per mean) at eps = 1/100, T = eps^8, horizon 2/eps, using the exact sheet Ornstein-Uhlenbeck flow between proposals and checking the event on a time grid of step 0.1. The empirical Pr(contrast formation) was 0.030 +- 0.009 for m0 = 0, 0.100 +- 0.015 for m0 = x°, and 0.080 +- 0.014 for a random m0. The statement "bounded below uniformly in the regime parameter" is true, and part (b) is honest. But the opening phrase "a collective forms from every Gaussian initial law" reads as a quantitative claim that the bound does not support. The margin over the stationary value 4 e^{-1/(4T)} is real only because that value is about e^{-2.5e15}.
**Fix:** Add one sentence stating that the bound is a positivity statement of size at most about 1e-30 and gives no estimate of the typical formation probability. Alternatively, quote the preparation-class bound of `prop:rg-formation-endogenous-cycle` (p_0 = 0.00456), where the number is meaningful.
**Falsifies:** A grid-free estimate showing that the true probability is itself astronomically small. A grid check can only miss qualifying intervals, so the Monte Carlo values are, up to grid resolution, underestimates.
**Status:** ESTABLISHED. The result is correct; only the scale is unreported.

### L6-03: "Spanning forests of stars with n components" admits trivial stars that do not cover
**Location:** `c13_formation.tex:1421-1422` and `:1469-1470` (`prop:rg-gated-depth-one` (c)).
**Severity:** low
**Evidence:** On the four-cycle, {path on three vertices, isolated vertex} is a spanning forest of two stars (an isolated vertex is the star K_{1,0}) with n = 2 components, but it is not an edge cover. The proof (`ca_formation_proofs.tex:2305-2330`) correctly works with covering forests; only the statement is loose.
**Fix:** Write "the spanning forests of stars, each with at least one edge, with n components", or equivalently "covering forests with n components".
**Falsifies:** An earlier stated convention that a star has at least one edge.
**Status:** ESTABLISHED; wording only.

### L6-04: "Every residual annihilates the kernel of H_M" should read "every selected residual"
**Location:** `ca_formation_proofs.tex:1929-1930` (proof of `prop:rg-formation-endogenous-cycle` (a)).
**Severity:** low
**Evidence:** The exterior residual covectors do not annihilate ker H_M; the exterior residuals grow toward -3/2 along the held path. Only the selected residual covectors lie in range H_M = span{n±_e, n±_f}, and that is all the displayed bound e^{-8t} + sqrt3 rho° e^{-4t} + sqrt3 delta_eps uses. Checks: |Z_M^T ker H_M| = 2.8e-16, and the selected-residual bound at t_eps for eps = 1/100 is 1.25e-5 <= eps^2 = 1e-4.
**Fix:** Replace "every residual" by "every selected residual".
**Falsifies:** None; this is wording.
**Status:** ESTABLISHED; wording only.

### L6-05: The exponent restriction a > 7/2 in the corollary is unnecessary and invites a misreading
**Location:** `c14:295-298` (`cor:phil-joint-selection`); proof `ca_joint_selection.tex` ("The following argument in fact permits arbitrary positive caps h_n").
**Severity:** low
**Evidence:** The proof accepts any positive caps d_n <= h_n. The number 7/2 is the threshold of `prop:phil-temperature-tracking` for the tracking error, not for selection; the corollary itself says the imprint has no temperature-uniform lower bound. Quoting 7/2 in the corollary suggests a proved power-law sufficient condition, and there is none.
**Fix:** State "for any prescribed positive caps h_n, in particular h_n = T_n^a for any a > 0, the thresholds can satisfy d_n <= h_n", and keep the existing sentence that no power-law selection condition is proved.
**Falsifies:** A step of the corollary's proof that uses a > 7/2.
**Status:** ESTABLISHED; the statement is weaker than the proof, not wrong.

### L6-06: "Positive assignment masses near the product prior" is unquantified
**Location:** `c13_formation.tex:646-648` (`def:rg-formation-regime`).
**Severity:** low
**Evidence:** Only the distinguished mass is quantified (|lambda_{i0,+}(0) - sig(-2)| < 1e-2), and it is used at `ca_formation_proofs.tex:1434-1440`. The entropy lemmas state their bounds independently of the source masses (`ca_formation_proofs.tex:557, 632, 777`). The proofs therefore need only positivity, and "near" constrains nothing. An implementer reading the preparation cannot tell what is required.
**Fix:** Replace "near the product prior at vartheta = 1" by "positive", and keep the quantified condition on the distinguished mass.
**Falsifies:** A step of the history-coupling or envelope lemmas that uses quantitative closeness of the non-distinguished masses.
**Status:** Definition; under-specification only.

## 3. Verified without defect

All scripts are in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\scripts\L6-formation\`, with their outputs in the matching `.out` files. Checker: Claude Opus 5.5 (investigator, not adjudicated).

| Label / item | Method | Residual or result |
| --- | --- | --- |
| `prop:rg-formation-endogenous-cycle`: Gram matrix [[3,1],[1,3]]; spectrum of H_M {8,8,4,4,0^8}; R_ev n+ = 0, R_ev n- = 4 n- | explicit 12x12 matrices (`four_cycle_checks.py`) | 0.0 |
| same: held path eq:rg-formation-cycle-center; r_int = -e^{-(8+T)t}; r_ext = e^{-Tt}(-3/2 + e^{-8t}/2) | expm(-K_M t) at t = 0.1, 0.25, 1, 3 | 1.3e-15 |
| same (b): dim W_M = 9; the loop covector annihilates Z_M; W_M contains ker H_M and the fast common normal; K_M block diagonal; K_Z = diag(8+T, 4+5T, 4+5T); physical diag(8+T, 4+T, 4+T); exterior common residuals annihilate Z_M | null spaces and projections | 0.0, 2.8e-16, 5.3e-16; diagonals exact |
| same (c): one-coordinate OU relative entropy 1/2[-a - log(1-a) + lambda a z0^2/T]; physical comparison identity KL(q‖Pi) + E|d_z|^2/(2(1-sigma^2)) = KL(q‖nu_T) + log((4+5T)/(4+T)) | closed form; random Gaussian q | 1.8e-17; 4.4e-16 |
| same (d): D_e = (z_+ + z_1)^2 + z_2^2 and D_f = (z_+ - z_1)^2 + z_3^2; Gaussian identity for E e^{(a+Z)^2/T}; factors sqrt(4/3), sqrt2 | random x; quadrature | 3.6e-15; 0.0 |
| same: constants. p_eps = 0.00753 < 0.008 at eps = 1/100; eps^4 t° increasing; p_0 = 0.004563; 4e^{-1/(4T)} = 0 at T = 1e-16; internal RE at time 0 >= 0.4828; selected residuals <= 0.1527 < 1/5; exterior >= 1.415 > 13/10; Doob/Chebyshev constant 73728; eta_eps | direct evaluation | as stated |
| `prop:phil-seeded-contrast-formation`. (a): Picard contraction (2 eps t_1^2/c_0)^n/n!, row bound, uniqueness. (b): 1/46080 = (2pi)^{-6}pi^6/6!; 737280 = 16 x 46080; row ratio 1.2277 < 1.23; eps t° = 0.10210; four disjoint seed events; signed-permutation and uniform-relabeling symmetry. (c): invariant means (c1 1, c2 1, 0) | proof reconstructed; arithmetic recomputed | exact |
| eq:phil-selection-sheet-determinants and the four class formulas d_M, d_E, d_N, d_C | determinants of all 16 sheets at T = 0.5, 0.05, 0.01 | max relative residual 1.6e-13; class counts 2, 8, 4, 2 |
| One-receiver flips join only the class pairs M-E, N-E, C-E (proof of `prop:phil-temperature-tracking`) | enumeration | confirmed |
| eq:phil-selection-saddle-spectrum (alpha, beta, and eta twice) | finite-difference Jacobian of the exact reduced field at p = 1/2 | agreement to 1e-6 at T = 0.05 and 0.01; alpha = 1.729 and 2.592 |
| Time scale eps^{-1} log(1/eps) of `conj:phil-contrast-formation` | RK4 on the adiabatic field p' = eps(beta-bar(p) - p) from p = 1/2 + eps I | for eps = 1e-2, 1e-3, 1e-4, slow exit time minus log(1/eps)/alpha = 3.886, 3.884, 3.882 at T = 0.05 and 3.544, 3.545, 3.547 at T = 0.01; the offset is constant, as the proof's finite-transit argument requires |
| c14 statement of `conj:phil-contrast-formation` against `ca_selection_proofs.tex` | statement matching | match: T0 from r_E, r_N, r_C <= 1/8; C_T any constant > 1/alpha, independent of the mean; a.e. m0 via the analytic imprint; frozen arm by Harris convergence and rotation symmetry at every T > 0 |
| `cor:phil-joint-selection`: window constant 2 (alpha >= 7/5); initial entropy H_0 = log Z(1/2) - 6 + (tr A-bar + m0^T A-bar m0)/(2T); majorants B_L and B_F; Borel-Cantelli thresholds; frozen matching mass 1/(2 D_0) | proof reconstructed | correct |
| `prop:phil-temperature-tracking`: K_0 = 210 sqrt(2pi/5) = 235.41; vol B_7(1/2) = pi^3/840; common kernel of dimension 5 with Omega eigenvalues {1,1,1,1,5}; max ‖H_S‖ = 8.83 <= 12; active determinants at T = 1/100 (M/T^3 = 5261.9, E/T = 1000.3, N/T = 994.6, C = 3306.1); the L_p bound; assembly of c_*; the Pinsker step | numerical and by hand | all within the stated bounds |
| `prop:rg-gated-depth-one` (c): rank a_F = |F| + N - cc(F) on every edge subset; k_S >= 2(N - n), with equality iff att(S) is a minimum edge cover; every minimum edge cover is attended | C4, C5, triangle plus pendant, K4, K_{1,3} (`gated_depth_one.py`) | rank residual 0; all true |
| same: sheet-weight bounds eq:rg-gated-sheet-weight and merge cost eq:rg-gated-merge-cost | exact determinants, Omega = I + R_ev and Omega = I, T = 1e-2 and 1e-3 | all hold (C4, Omega = I, T = 1e-3: q(cc <= 1) = 6.4e-3 <= K_1 T = 4.4e-2) |
| same (d): block law eq:rg-gated-block-law | exact Gaussian-mixture sampling, a = 0.3, 2e4 samples | holds in every case (C4, Omega = I, T = 1e-3: 0.0495 +- 0.0015 <= 0.2245; C5: 0.052 <= 0.372) |
| `prop:rg-formation-gaussian-core` (b): Delta_{l-1:l}(t) / [(A^2/T) J_{l+1}^4/J_l^3 y_l(xbar(t)/A)] -> 1 | triangle chain, L = 2 and 3, ratios rho = 1e-1 to 1e-6, mean flow with activations, mpmath for the cancellation in v (`gaussian_core.py`) | 0.958, 0.996, 0.9996, 1.0000, 1.0000 for L = 2; 1.0005 at the second interface of L = 3, rho = 1e-6 |
| same, window form with y_l(v0) | k = 12 log(1/rho)/(j log(1/eps)), so that k log(1/eps) rho -> 0 | [0.9989, 0.9995] for L = 2, rho = 1e-6; [0.9994, 0.9999] at the second interface of L = 3. When k log(1/eps) rho is not small (rho = 1e-3, k = 40) the ratio is 0.17 to 0.41, as the hypothesis predicts |
| same: B_{l+1,l} has full row rank for two choices of bridge endpoints; B_ll square; j_l = lambda_min(B_ll^T B_ll) = 1 | ranks and spectra | confirmed |
| same (c): 3Q_l - 4Q_{l+1} >= Q_T + Q_Delta forces Q_1 >= (Q_T + Q_Delta)[(4/3)^{L-1} - 1]; in the doubling regime the core loss has order eps^{2Q_{l+1}+54}, far below eps^14 | algebra | correct |
| `lem:rg-formation-carrier` dimensions (already in the ledger) | rebuilt carrier | dim 19 and 39; rank B 11 and 23; dim W_l = [9, 8] and [19, 17, 16] |
| `thm:finite-depth-adaptive-formation` quantifier order | reading of the text | epsilon_L is now fixed after k_L and eta_L (c13:912-914, 1063-1066); the quantifier part of M5 appears repaired |

## 4. Implementation notes for VFE_4.0

The four-cycle and gated families are fully closed form at the level of single sheets. For each complete assignment S an implementation needs:

- the residual-pair maps a_e and the sheet stiffness H_S = sum_i a_{e(i,S_i)}^T a_{e(i,S_i)};
- the drift matrix A_S = H_S + T Omega, with Omega = I + R_ev for the optimizer and Omega = I for the physical posterior;
- the sheet weight from log det(A_S/T);
- the posterior rows beta_i(j|x), proportional to pi_ij exp(-D_{e(i,j)}(x)/T) and computed by log-sum-exp.

The invariant law mu_p is an exact Gaussian mixture and can be sampled exactly, so the block-law, merge-cost and stationary-comparison statements can be tested against it without simulating any dynamics.

The dynamics are extremely stiff. In the four-cycle regime T = eps^8 is about 1e-16. The attended normal modes relax at rates 4 to 8, the kernel modes at rate T, and the noise scale sqrt(2T) is about 1e-8. Euler-Maruyama is unusable here. Integrate each sheet exactly between proposal times through a per-sheet eigendecomposition (sixteen sheets on the four-cycle), add the exact Ornstein-Uhlenbeck covariance T(1 - e^{-2 lambda tau})/lambda for each mode, and redraw the full assignment at the jumps of a rate-one Poisson clock. `seeded_contrast_mc.py` does exactly this and runs 1200 trajectories of horizon 200 in a few minutes on CPU.

The learned-row experiments are law-level, and this is the main place where the theory leaves the algorithm unspecified. The rows obey pi-dot = eps_L(q_t(S_i = j) - pi), where q_t is the recognition law of the whole process, so the system is of McKean-Vlasov type. The selection proof notes that q_t is not a Gaussian mixture conditional on a state-dependent history, so there is no closed form. An implementation must approximate q_t(S_i = j) with a particle ensemble, or with a moment closure that the theory does not supply. This changes the mechanism under test. The theorem amplifies an imprint of size eps_L I(m0) through a saddle with growth rate alpha, while a finite ensemble of M particles injects row noise of relative size M^{-1/2} that the same saddle amplifies. To test law-level selection rather than noise-induced symmetry breaking:

- M must be large compared with eps_L^{-2} |I(m0)|^{-2};
- the learned and frozen arms should share common random numbers;
- the selected matching should be compared with the sign of the frozen imprint j_T(m0) = v . int_0^inf [q^0_t(sigma = +1) - 1/2] dt, which can itself be estimated from the frozen arm.

The reduced adiabatic field beta-bar(p) - p is closed form from the sixteen determinants and gives a cheap reference trajectory. Its exit time from the saddle is log(1/eps)/alpha plus an O(1) transit (alpha = 2.59 at T = 0.01). That fixes how long a simulation must run: slow time about 5 to 7 for eps between 1e-2 and 1e-4.

The finite-depth family cannot be realized numerically in its stated regime. At depth two the regime has J_1 = eps^{-20}, A = eps^{40}, and T = eps^{86}, and the theorem's epsilon_L, k_L and the lemma constants c_early, c_joint, c_fast, k_geom are existence constants with no numerical values. What can be implemented exactly is:

- the carrier: flags Q_l, blocks B_{sr}, projections P_l, and the Schur complements H_l;
- the Gaussian-core loss Delta = v^T H_zz^{-1} v/(2T) and its asymptotic y_l;
- the marked channel C_{l <- l'}.

The discarded-loss computation suffers catastrophic cancellation when the stiffness ratios are small, because v = (H x-bar)_z has relative size (J_{l+1}/J_l)^2. `gaussian_core.py` evaluates it with 60-digit arithmetic for that reason.

The exact parents beyond the Gaussian core require the tilted hidden integrals psi_{l,S} = -log E exp(-phi_S). They also require the inherited rows beta^l_{SS'} as conditional expectations over the hidden coordinates, for each of the |J| = 2^N assignments: 64 at L = 2, 4096 at L = 3, 2^24 at L = 4. After conditioning, the inherited rows do not factor over receivers. This is the dominant cost and needs Monte Carlo or quadrature with a declared error budget, which the theory does not supply. The activation times t_l = k_L log(1/eps)/J_l, the coherence resolution eta_L A, and the activation rule (push the complete law, copy the parameter) are fully specified. The learning metric m(vartheta) = T M(vartheta) is closed form, with M the Fisher information of `eq:rg-formation-model-fisher`.

The four-cycle parent constructions (`prop:rg-formation-endogenous-cycle` (b)-(d)) are activated from the recognition law conditioned on a path event, the episode prefix. An implementation must therefore condition an ensemble on a history event, by rejection or importance weighting. Whenever the parent leaves a matching stratum, the inherited proposal kernel beta_C must resample the discarded coordinates from the optimizer conditional N(0, T K_Z^{-1}).

The contrast-formation event is defined in continuous time ("an interval of length one"). Any implementation discretizes it, and should report the grid spacing because a coarse grid undercounts. Finally, the explicit constants of the gated proposition are asymptotic and often vacuous at practical temperatures. For Omega = I + R_ev on the four-cycle, K_1 is about 1.2e3, so K_1 T exceeds one at T = 0.01. These constants should not be used as numerical error bars.

## 5. Minimal unresolved obligation in this scope

It is open whether learning changes the probability, or the conditional law, of contrast formation at eps_L = T^{1/8} on [0, 2/eps]; this is the content behind L6-01, and the manuscript already poses it. The next action that would change the answer is a coupled learned/frozen particle simulation at eps = 1/100, T = eps^8, with common random numbers and M large enough that row noise falls below the imprint, reporting both formation probabilities and the first-block laws.
