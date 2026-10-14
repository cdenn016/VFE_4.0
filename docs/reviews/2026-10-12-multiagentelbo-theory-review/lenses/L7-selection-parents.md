# Lens L7-selection-parents: pairing selection, joint selection, persistence, early block null, conditioned and early parents

Reviewer: Claude Opus 5.5 (lens investigator; not adjudicated). Date 2026-10-12.

## 1. Scope and verdict

I read in full: `ca_selection_proofs.tex` (1-824), `ca_joint_selection.tex` (1-250), `ca_temperature_bounds.tex` (1-124), `ca_selected_coherence.tex` (1-266), `ca_first_block_comparison.tex` (1-164), `ca_parent_conditional.tex` (1-257), `ca_parent_dynamics.tex` (1-238), and `ca_parent_early.tex` (1-1142); the statements they prove in `c14_interpretation.tex` (1-380: `prop:phil-seeded-contrast-formation`, `conj:phil-contrast-formation`, `cor:phil-joint-selection` and the following prose); `c13_formation.tex` 217-405 (evaluator lift and laws) and 1100-1577 (the four-cycle definition, `prop:rg-formation-endogenous-cycle`, the gated family); and `ca_formation_proofs.tex` 1829-1958 and 2067-2146 (cycle normals, center path, episode estimates, row well-posedness). Note that the theorem, the corollary, and the seeded proposition are stated in `c14_interpretation.tex`, not in `c13_formation.tex`. I checked the verified ledger (entry of 2026-09-26: `conj:phil-contrast-formation` reconstructed by a checker whose model was not recorded) and the vault notes `multiagentelbo-2026-09-26-*` and `-09-28-early-parent`.

Verdict: I found no critical, high, or medium defect. I followed the proof of the pairing-selection theorem through every step, and each step holds at the level of detail written. The steps are the Harris bound that is uniform in the frozen rows, the imprint expansion with its O(eps^2 t^3) coupling remainder, the nonvanishing of the imprint through the lock analysis, the analytic null zero set, the entropy identity, the modified log-Sobolev constant c_T/(K_T + 8 log(1/p_min)), the tracking ODE with its Pinsker step, the saddle cone bootstrap, and vertex capture. All of its exact constants that I recomputed agree with my recomputation. The joint-selection corollary is a correct Borel-Cantelli construction with honest quantifiers: the thresholds do not depend on the mean, the integer "sufficiently large n" does depend on it, and the selected matching may depend on n. The temperature, persistence, conditioned-parent, parent-dynamics, early-null, and early-parent propositions are also correct as stated. That includes the ESTABLISHED negative, where the discarded KL diverges when log(1/eps) - theta tends to infinity, and the T^{-5/8} divergence at the first completion.

Every probability statement is correctly typed as a statement about the law or about paths. Selection concerns the one-time laws q_t and the deterministic rows they drive. Persistence and both parent-dynamics results are total-variation bounds between path laws, obtained by explicit couplings with common Brownian motion, clocks, and uniforms. Every exceptional set is the zero set of a real-analytic, not identically zero imprint j_T, or, along the schedule, a countable union of such sets together with a Borel-Cantelli limsup, so it is Lebesgue-null. Every limit interchange I checked is justified. Fubini for the imprint integral uses the Harris bound with weight (1+|x|^2). The window supremum is pointwise in the mean. The supremum over mobility is reduced to rational mobilities by continuity. Dominated convergence is applied to F_n <= 1.

The five findings below are low severity: a summary sentence that drops a hypothesis, symbol collisions, a loose but valid constant, a citation whose theorem text I could not extract, and quantitative vacuity that is only partly flagged. A particle simulation reproduces three things: the selected matching agrees with sign(j_T(m0)) and flips under rotation of the mean, the frozen arm stays symmetric at the exact invariant mass 1/(2 D_0), and the off-matching recognition mass decreases as the mobility decreases. It does not resolve the eps^{-1} log(1/eps) rate, because the mobilities I could afford are not small compared with the mixing rate at T = 0.04.

## 2. Findings

### L7-01: The c14 summary of persistence drops the mobility cap needed for the window inclusion
**Location:** `c14_interpretation.tex:306-313` (prose after `cor:phil-joint-selection`); `ca_selected_coherence.tex:81-83, 251-255`
**Severity:** low
**Evidence:** The proposition places the persistence window [t_L, t_L + T^{-1/4}] inside the selection window [t_L, 2 t_L]. This requires T^{-1/4} <= t_L = 2 eps^{-1} log(1/eps), and the proof secures it through the cap eps_L <= T^2 (`ca_selected_coherence.tex:252-254`). The c14 sentence states only that "the schedule tolerances satisfy eta_n <= T_n". The corollary's thresholds satisfy d_n <= e^{-1}, which gives only t_L >= 2e. That does not dominate T_n^{-1/4} for small T_n unless a cap is imposed. The corollary does permit an arbitrary cap h_n, so the result is reachable, but the summary omits the condition.
**Fix:** In c14:309 write "when the schedule tolerances satisfy eta_n <= T_n and the thresholds satisfy d_n <= T_n^2", matching the proposition and the parent propositions (c14:333-335 already state the cap).
**Falsifies:** A proof that the window inclusion holds along the corollary without any cap on d_n.
**Status:** The proposition is ESTABLISHED and correct. Only the summary is incomplete; no inflation.

### L7-02: Symbol collisions across the selection appendices
**Location:** `ca_selection_proofs.tex:96, 121` (Harris constant K_T) against `ca_first_block_comparison.tex:22` (first-block mark K_T); `ca_selection_proofs.tex:617` (saddle eigenvalue eta) against `ca_parent_conditional.tex:9` (KL tolerance eta), `cor:phil-joint-selection` (eta_n), and `c13_formation.tex:1271` (eta_epsilon); rho as a ball radius (`ca_selection_proofs.tex:477`) and as a failure probability (`ca_parent_conditional.tex:18`)
**Severity:** low
**Evidence:** K_T is a numerical constant in one appendix and a random block mark in the next. `prop:phil-early-first-completion` uses objects from both.
**Fix:** Rename the first-block mark (for example to K^blk_T) and the saddle eigenvalue (for example to eta_sad).
**Falsifies:** None. This is notation.
**Status:** Not a mathematical issue.

### L7-03: Loose, valid determinant bound in the temperature proposition
**Location:** `ca_temperature_bounds.tex:69-72`
**Severity:** low
**Evidence:** At T = 1/100 the active determinants are d_M/T^3 = 5261.9, d_E/T = 1000.3, d_N/T = 994.6, and d_C = 3306.1 (`s1_output.txt`). The stated upper bounds 5262, 1017, 2175, and 5120 hold, but 2175 for d_N is more than twice the true value, which looks like a transcription slip. It is harmless because the argument uses only the lower bounds on d_E and d_N and the upper bound on d_C. Likewise the bound ||H_sigma|| <= 12 is valid; the true maximum over the 16 sheets is 8.83.
**Fix:** Optionally replace 2175 by 1000.
**Falsifies:** None.
**Status:** ESTABLISHED; not inflated.

### L7-04: The Harris citation's explicit-constant clause was not text-verified
**Location:** `ca_selection_proofs.tex:89-93` (citing HairerMattingly2011, Assumptions 1-2 and Theorems 1.2-1.3)
**Severity:** low
**Evidence:** The source exists: arXiv:0810.2777, Hairer and Mattingly, "Yet another look at Harris' ergodic theorem for Markov chains". The bib entry (`references.bib:5439`, Progr. Probab. 63, 109-117, 2011) is consistent with that record. The arXiv PDF I fetched was not text-extractable, so I could not quote Theorem 1.3. From memory, Theorem 1.3 gives a contraction in the weighted metric rho_beta with beta = alpha_0/K and an explicit contraction factor that depends only on (alpha, gamma, K, R). That is exactly what the proof needs: the drift constant (gamma = e^{-2T}, K = 12) and the minorization constant alpha_T are uniform in the frozen rows, while the minorizing measure Unif(B_1) x delta_{sigma*(p)} depends on p. Verification against the primary text is incomplete.
**Fix:** None, unless the primary text contradicts this. A locator to the explicit contraction-factor formula would help readers.
**Falsifies:** Theorem 1.3 of the published version having constants that depend on the minorizing measure.
**Status:** The application holds if the citation says what I recall.

### L7-05: The early-null bound is nonvacuous only at astronomically small parameters, and its statement does not say so
**Location:** `ca_first_block_comparison.tex:27-50`; compare `ca_parent_early.tex:758-760`, which does flag the analogous issue for `prop:phil-early-first-completion`
**Severity:** low (informational)
**Evidence:** Even at the best mean, c(m0) <= e^{-1/4} rho_o^{12}/737280 = 1.06e-30 (rho_o = 1/100). The conditional TV bound d_T/c(m0) is therefore below 1 only when 8 eps_L T^{-1/4} < 1e-30. The statement is a limit statement and is correct as written.
**Fix:** Add one sentence to `prop:phil-early-block-null` that the bound is asymptotic and its constant is of order 1e-30, mirroring `ca_parent_early.tex:758`.
**Falsifies:** None.
**Status:** ESTABLISHED; not inflated.

## 3. Verified without defect

| Label / location | Method | Result |
|---|---|---|
| `eq:phil-selection-reduced-field` (V_i = p_i(1-p_i) g_i) | hand derivation | exact |
| Invariance of mu_p (Gibbs redraw plus per-sheet Gaussian factor) | hand | exact; mu_p(sigma given x) is the product of the receiver posterior rows because the Omega term cancels |
| `eq:phil-selection-moment`; R > 24/(1-e^{-2T}); clock event probability e^{-1}/2; Chebyshev radius sqrt(24T); redraw bound e^{-v/T}/16 | hand | correct; matches the HM requirement R > 2K/(1-gamma) |
| Imprint coupling ‖q^eps_t - q^0_t‖_TV <= eps t^2/(2c_0) and `eq:phil-selection-imprint-expansion` | hand (posterior derivative maximum 1/(4p(1-p))) | correct, including the eps^2 t^3/(6c_0) remainder |
| Tied-residual system `eq:phil-selection-tied-residual`, G = 4I - A_{C4} | hand derivation of the gradient flow | exact |
| Six assignments with m_1 = m_3; the four listed patterns; six remaining | enumeration, `s2_output.txt` | exactly 6 + 4 + 6 |
| (1,1,2,0) system; invariant half-spaces with outward derivatives -b/400 - 7d/100 and -a/4 - b/8; b >= e^{-4t}; h(inf) = 6/7 | hand plus `s2` (quadrature h(inf) = 0.857142857) | correct |
| Post-redraw formulas, h_r >= 171/224, odd-even margin | `s2`: formula residual 5.4e-17; minimum margin / ((7/50)e^{-4 t_r}) = 4.97 | correct, conservative |
| Lock failure delta_s; kappa_0 = (7/50)^2/4; E e^{abs(xi)^2/4} = 64; generator bound T e^5; E abs(X_{h_s})^2 = 14 | hand plus `s2` | correct (14 attained exactly) |
| `eq:phil-selection-nonvanishing`: sixteen-assignment average h_s/2 - (5/T) log log s - O(1) | hand | correct |
| Null zero set of a nonzero real-analytic function | standard | correct |
| `eq:phil-selection-entropy-identity`; d/dt log mu_p = sum_i pdot_i (1{sigma_i=+} - betabar_i)/(p_i(1-p_i)) | hand | exact |
| `eq:phil-selection-entropy-contraction`: entropy chain rule; psi <= 2(1 + log R)(z-1)^2; L_p = 4 + 2 log(1/q*_min); chi^2(Unif_B ‖ gamma) <= X_sigma; jump lower bound; (a-b)(log a - log b) >= 4(sqrt a - sqrt b)^2 | hand | correct |
| `eq:phil-selection-learning-entropy`, `-tracking`, `-refined-tracking` (constants eps/(2 lambda_b) and L eps^2/lambda_b^2) | hand | correct |
| `eq:phil-selection-sheet-determinants` and d_M, d_E, d_N, d_C | `s1`: exact rational arithmetic at 13 rational T (det is of degree <= 12), 0 mismatches; float over all 16 sheets, relative residual <= 2e-13 | exact identity |
| Class counts 2/8/4/2; r_E, r_N ~ sqrt(40/7) T; r_C ~ sqrt(5/3) T^{3/2} | `s1` | 2.39036 vs 2.39046; 1.29096 vs 1.29099 at T = 1e-5 |
| Saddle Jacobian entries; eigenvalues alpha, beta, eta; eigenvector v; alpha >= 7/5, beta <= -1/3, eta <= -2/5 when r <= 1/8 | `s1`: finite differences of the reduced field at T = 0.01, 0.03, T_0 | spectra match the formula to 1e-6; ‖J - J^T‖ = 8e-11; T_0 (all r <= 1/8) = 0.0651 |
| Vertex Jacobian (r_E - 1) I; Z_diag and the diagonal flow | `s1` (diagonal equals r_E - 1 to print precision, off-diagonal 0), hand | exact |
| Cone inequalities `eq:phil-selection-cone`; exit time (1 + O(r^2) + o(1)) log(1/eps)/alpha; vertex capture D^+ zeta <= -(nu/2) zeta + TV | hand | correct (pointwise in m0) |
| `cor:phil-joint-selection`: H_0 formula; B_L and B_F majorants; ell_T with 32 log(1/eps); measurability; Borel-Cantelli; mu_T(M) = 1/(2D_0); TV contrast within 2 eta_n of 1 - 1/(2D_0) | hand | correct |
| `prop:phil-temperature-tracking`: 5-dimensional passive kernel with rate 2T; active determinant bounds; K_0 = 210 sqrt(2 pi/5) = 235.4096; vol B_7(1/2) = pi^3/840; C_c and C_j; c_*; T abs(g) <= 1; H_0 <= 13(12 + abs(m0)^2)/(2T); final constants; exponent a > 7/2 | `s1` plus hand | correct; max T (adjacent weight ratio) = 0.436, max T abs(g_i) = 0.173 |
| `prop:phil-selected-pair-persistence`: r' = E b - r; P(S leaves M on the interval) <= (h+1) delta; path coupling (h+2) delta; r_E, r_N <= 2.5T, r_C <= 1.5 T^{3/2}; nonmatching mass <= 16T; integral of sqrt(mu_M mu_M') = mu(M) r_C; departure rate <= 31T; maximal bounds 32Th and 792Th; constants b and e | `s1` (maxima over T <= 0.01: 2.39T, 2.39T, 1.29 T^{3/2}; 14.3T; 28.7T) plus hand | correct |
| `prop:phil-actual-conditioned-parent`: two-order KL identity; chain rule through C; K_Z = diag(8+T, 4+5T, 4+5T); K^phys = diag(8+T, 4+T, 4+T); evaluator tilt normalizer (4+T)/(4+5T); rho <= 3T + 8 sqrt(2/pi) T^{1/4} + 3408 sqrt(T) | `s1` (eigenvalues 4.05, 8.01 for A_M and 4.01, 8.01 physical at T = 0.01) plus hand | correct |
| `prop:phil-actual-parent-dynamics`: conditioning moves TV by at most r; averaged departure 4d/(1-4d); Lipschitz constant eps u^2/(2 c_0 (1-c_0)); moment generator bound 25T - 2T abs(y)^2; total bound | hand | correct; O(T^{1/4}) along the schedule |
| `prop:phil-early-block-null`: abs(b(p) - b(1/2)) <= abs(2p-1); 2 eps H^2; conditional TV d/max(p,q) | hand | correct |
| `lem:phil-early-discarded-information` (i)-(iii); `lem:phil-early-physical-comparison` (joint convexity, 4 abs(z)^2 + 2 abs(z')^2, E abs(z')^2 <= 5T/8) | hand | correct |
| `lem:phil-early-held-seed` (i)-(iv); center-path residuals | `s1` (matrix exponential reproduces internal -e^{-(8+T)t} and exterior e^{-Tt}(-3/2 + e^{-8t}/2) to print precision) plus hand (0.4828 > 0.48; 0.1527 < 0.153; 1.3004; 0.15 eps^2) | correct |
| `prop:phil-early-seeded-parent` (a)-(e): tube failure 73728 eps^4 theta; a_delta^2 = eps^{-4}/18432; 24 exp[-eps^{-4}/(36864 theta)]; departure 4 Lambda e^{-1/T}; density factor c_m; E x_k^2 = rho^2/14; E abs(xi)^4 <= 15T^2/16; Donsker-Varadhan with E e^g = 2; divergence for log(1/eps) - theta -> infinity; e^{8s} = e^{-8} T^{-1/2} at theta_o; likelihood-ratio and L^2 bound in (d) | hand | correct; the ESTABLISHED negative is proved |
| `prop:phil-early-first-completion`: t_c <= (1/4) log(1/eps) + 23/20 (0.97926 eps^2 < 0.98 eps^2); c_12 = 46080/(10395 pi) = 1.4111; (2 c_12/13)(15/16)^{13/2} = 0.14271 >= 0.1427; (1/2) e^{-46/5} (0.1427)(0.01) = 7.209e-8 > 7.2e-8; order T^{-5/8} | hand | correct (two constants hold with margin under 0.1%) |
| `prop:phil-early-parent-dynamics`: radius-1/20 tube 24 exp[-1/(76800 T h)]; exterior common residual > 1.211; partner cost < 0.0764; Gaussian MGF factors sqrt(4/3) and sqrt(2); hazard 4 sqrt(8/3) Lambda e^{-1/T} | hand | correct |
| `rem:phil-early-stationary-mismatch`: (0.4489/0.5511)^4 = 0.4402; 10 R_0^8/384 = 2.6e-18 | hand | correct |

Simulation evidence (`s3_particle_selection.py`; a finite-N, dt = 0.05 surrogate of the law-level McKean-Vlasov system at T = 0.04 <= T_0 = 0.065, with exact held-sheet OU steps; outputs `s3_output_T0.04.txt`, `s3_results_T0.04.json`, `s3_postcheck.txt`). This is a numerical observation, not proof.

- **Imprint.** The frozen-arm estimates (N = 20000) are j = -5.18 for a generic mean, +5.75 for its rotation by one (it should be the exact negative; the gap is Monte Carlo error), and +49.7 for 4u_o.
- **Learned arm.** Runs used N = 4000 particles, eps = 0.02 and 0.01, and two seeds. In all six runs the selected matching equals the sign of j and flips under rotation. The supremum over the window of q(S != M) fell when eps was halved:

  | Mean | eps = 0.02 | eps = 0.01 |
  |---|---|---|
  | generic, seed 2 | 0.0075 | 0.0010 |
  | generic, seed 3 | 0.013 | 0.0020 |
  | rotated | 0.050 | 0.0042 |

  The rows end at the matching vertex to within 1e-4.
- **Frozen arm.** Late q(S0) - q(S1) lies between +0.001 and +0.006. The late q(S0) + q(S1) = 0.665 matches the exact 2 mu_T(M) = 1/D_0 = 0.6648.
- **Rate.** The slow exit time divided by log(1/eps) went from 1.28-1.64 to 0.89-1.28 as eps halved. That is not the asymptotic (1/alpha) log(1/eps) law (1/alpha = 0.53), because eps is not small against the mixing rate, which is of order T. The rate claim is therefore neither confirmed nor contradicted numerically.

## 4. Implementation notes for VFE_4.0

**Closed-form objects.**
- The sixteen sheet drift matrices A_sigma = H_sigma + T(I + R_ev), with H_sigma = sum_e m_e(sigma)(a^b_e a^b_e^T + a^m_e a^m_e^T).
- The sheet weights det(A_sigma/T)^{-1/2}, given by the four verified determinant polynomials.
- The product receiver posterior rows. Compute them in logit form, logit_i = log(p_i/(1-p_i)) - (D_{e_i} - D_{e_{i-1}})/T, because e^{-D/T} underflows already at moderate T.
- The frozen-row invariant law mu_p, a 16-component Gaussian mixture.
- The reduced field V(p), from a multilinear evidence polynomial.
- The channel C, as orthogonal projection onto W_M. Its complement Z_M is spanned by (n_e^+ - n_f^+)/sqrt(2), n_e^-, and n_f^-, and the loop and the common normal coordinate are retained.
- The hidden optimizing conditional N(0, T K_Z^{-1}) with K_Z = diag(8+T, 4+5T, 4+5T), and the physical one with diag(8+T, 4+T, 4+T).
- The exact held-sheet OU transitions. They permit event-driven integration between clock rings, with a ring redrawing at X_{tau-} and self-proposals counted as rings, as in the theory.

**Objects that are not closed form.** These are the recognition law q_t of the switching diffusion, and with it the learning signal q_t(sigma_i = +1); the imprint j_T(m0), which needs Monte Carlo; and every discarded conditional relative entropy of an actually conditioned law (H_opt and H_phys of q_theta^E or q_theta^o).

**Ensemble requirement.** The theory's source learning is a McKean-Vlasov coupling: the rows are driven by the law, not by a sample path. An implementation must do one of two things:
- Carry an ensemble, as `s3` does. Finite-N fluctuations then become an extra symmetry-breaking source, competitive with the imprint when abs(j_T) is small, and the theory says nothing about them. Near the null set of j_T the selected matching of a finite ensemble is not determined by the theorem.
- Carry a parametric surrogate of q_t, such as a Gaussian mixture over sheets. Its closure error is not bounded by the theory.

**Conditioning on the activation event.** Conditioning on E, a completed coherence prefix, is implementable by path filtering of the ensemble. The surviving fraction is about 1 - rho for the late event, but about c(m0), which is tiny, for the early seeded event, so the early parents cannot be sampled by rejection at realistic sizes. The H functionals need conditional-density estimates in the three hidden coordinates. The Donsker-Varadhan lower bound (`lem:phil-early-discarded-information`(iii)) and the Gaussian upper bound (`lem:phil-early-held-seed`(iv)) give computable brackets.

**Choices the theory leaves to the implementer.**
- How the clock is discretized.
- How the law marginal is estimated from finitely many samples.
- The value of T_0. The text proves only existence; I computed 0.0651 for the r <= 1/8 criterion.
- The value of C_T, beyond C_T > 1/alpha. The corollary's choice of 2 is valid for every T <= T_0.

**Impractical regimes.**
- The quantitative tracking bound needs eps <= T^a with a > 7/2. At T = 0.01 that means eps <= 1e-7 and t_L of about 3e8.
- The first-formation regime T = eps^8 with eps <= 1/100 gives T <= 1e-16. That is below double-precision resolution of the restoring gaps relative to order-one residuals, and the formation-arm exponentials e^{-1/T} are exact zeros.

Simulations should use a moderate fixed T <= 0.065 with eps well below the frozen mixing rate (of order T), which means horizons of order 1e4 to 1e5 time units. With batched 12x12 transition matrices the cost is about 2 ms per step for 4000 particles on CPU.

Scripts and outputs: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\scripts\L7-selection-parents\`:
- `s1_sheet_algebra.py` and `s1_output.txt`
- `s2_imprint_lock.py` and `s2_output.txt`
- `s3_particle_selection.py`, `s3_output_T0.04.txt`, `s3_results_T0.04.json`, and `s3_postcheck.txt`
