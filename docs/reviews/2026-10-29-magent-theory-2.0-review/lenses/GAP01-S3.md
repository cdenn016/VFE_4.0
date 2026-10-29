# GAP01-S3: Small-parameter persistence proofs of the per-agent grouping theorems and the nonflat windows

I read in full `chapters/13_per_agent_inheritance.tex` lines 1-398 (the scope 60-398 plus the setup 1-59 that defines the Gibbs rows, the label-preserving null row and the four-state channel), `chapters/14_collective_local_parent.tex` lines 1-240 (the scope 66-90 and the window paragraph 166-178, with the field, cone and local-realization context), and `companion/c13_per_agent_inheritance.tex` lines 1-258 (scope 122-258). For hypotheses I read `chapters/03_dynamics.tex` 140-240 (`thm:dyn-finite-flow`, `prop:dyn-global-flow`, `prop:dyn-row-relaxation`, `prop:dyn-morse`, `def:dyn-chart-flow`, `prop:dyn-chart-flow`) and 315-357 (`def:dyn-per-agent`, `prop:dyn-per-agent`, `def:dyn-per-agent-flow`), and `chapters/05_inheritance.tex` 1375-1400 (`inh:null-flow`, `inh:recursive-formation`, `inh:formation-open`). I read `frontmatter.tex`, `chapters/06_outlook.tex`, `AUTHORING.md` and `README.md` only by grep for how they describe these results. Context records read: the GAP01 entry of `synthesis/COVERAGE.md`; `lenses/D17-PA.md` in full; the passages on chapters 13 and 14 in D18-LP, XC7-IMPL, XC3-SCOPE, XC6-PROG, XC1-GAUGE and XC2-FRAME; D17-PA's `geometry_eps.py` and its outputs (used as an independent brute-force reference); and the 2026-10-22 REPORT entries L03-DYN-03 and T5.

**Verdict.** None of the small-parameter results in scope is false, and each proof is valid in outline, but only as an unquantified "sufficiently small" argument. For the persistence theorems the gap this lens was dispatched for is closed. The limit matrix M is symmetric, so a global Lyapunov argument the manuscript does not use makes them fully explicit. For `pa:persistent-grouping` it gives ε ≤ 0.028 with entry by T = 7.82, and T = 4.72 for ε ≤ 0.01. For `pa:angular-grouping` it gives ε ≤ 1/100 with T = 4.73, uniformly over all angular records and over marked and hidden amplitudes. The chapter 14 import needs, in addition, δ·max|b_i| ≤ 0.055.

The nonflat windows (`pa:nonflat-window`, `lp:nonflat`) are true as qualitative ε → 0 statements on a fixed finite window. Their admissible ε is set by two exact mechanisms the proof never states, and at every positive ε the curvature itself decays, so the finite window is necessary. A rigorous explicit ε_0 for the windows is not established. I give a leading-order law, ε*(T_+) ≈ 0.525 e^(-T_+/3), and a reduced model that matches the record-inclusive brute force to four digits.

The density-level chart flow on {-1,1} x S^1 x S^1, which every invariance and intertwining statement presupposes, is well posed. The manuscript does not prove this; I give a proof with explicit Lipschitz constants (GAP01-S3-01). The four-state open set is rigorous at an explicit but small radius.

**What I reconstructed.** I rederived every field and null field in scope (pa:four-potentials, pa:four-null, pa:bit-flow, pa:angular-ode, pa:angular-null, lp:field), the row identities, the null-box chain, the ε = 0 geometry, and the four-state linearization. I wrote new proofs of three things: the persistence theorems with explicit constants, the chart-flow well-posedness, and the four-state open set at an explicit radius. For the nonflat windows I reconstructed the qualitative continuity argument and derived the structure of the positive-ε cost.

**Scripts.** Scripts and outputs are in `scripts/GAP01-S3/`: `s1_bernoulli_constants`, `s2_angular_constants`, `s3_nonflat_reduced`, `s4_four_state_radius`, `s5_chart_lipschitz` and `s6_ch14_delta`, each a `.py` with its `.out`. Each ran single-threaded under 30 s.

**Status of the numbers.** "Explicit" constants are closed-form inequalities evaluated in double precision with bisection, with no ODE integration. The exception is the four-state Volterra bound, which uses an upper Riemann sum. Nothing is interval-validated. Numerical agreement in s2 to s6 supports the identities and scalings; it proves none of the theorems.

## Findings

### GAP01-S3-01: The density-level per-agent chart flow is used but never shown well posed; a proof with explicit constants is supplied
**Category:** gap
**Severity:** medium
**Location:** chapters/13_per_agent_inheritance.tex:9 ("Gibbs-reduced convention"), :214-237 (pa:angular-flow), :241-295 (pa:angular-grouping), :301-312 (pa:phase-parent); chapters/14_collective_local_parent.tex:43-69 (lp:native-field); chapters/03_dynamics.tex:223-232 (def:dyn-chart-flow, prop:dyn-chart-flow, stated for the reciprocal potential), :354-357 (def:dyn-per-agent-flow, finite tables with evolving rows); AUTHORING.md regularity rule
**Evidence:** pa:angular-flow proves tangency: the centered own-law first variation lies in span{1, z, e(φ), z e(φ), e(ψ)} with the coefficients of pa:angular-ode. Four statements need more than tangency: "preserves pa:angular-family" (13:215), "the exact Fisher velocity on an invariant family" (13:236), "intertwines the original-time full and null flows" (13:302), and the use of the family ODE as *the* flow in pa:angular-grouping and pa:nonflat-window. They need uniqueness of a declared density-level flow. No such flow is defined for the Gibbs-reduced per-agent dynamics on a non-finite space. D17-PA-03 and XC3-SCOPE-02 report the gap; neither proves the statement. My reconstruction follows.

*Chart.* Let λ = ½(δ_{-1} + δ_1) ⊗ H ⊗ H on K = {-1,1} x S^1 x S^1 and λ_M be uniform on M = {-1,1}. The state is (f_i, g_i)_{i≤N} ∈ (L^∞_0(λ))^N x (L^∞_0(λ_M))^N, with q_i = e^{f_i}λ/E_λ e^{f_i}, s_i defined likewise, and norm ‖(f,g)‖ = max_i max(‖f_i‖_∞, ‖g_i‖_∞). The own-block first variation, modulo constants, is φ_i = log q_i − β_ii log G_i − Σ_{j≠i} β_ij log ν_j − log ℓ_i(o_i | ·), with β_i = softmax(log π_i − D_i) and D_ij = E_{q_i}[log q_i − log T_ij]. The chart field is V_i = −(φ_i − E_λ φ_i), as in def:dyn-chart-flow.

*Estimates.*
(a) Λ(f) = log E_λ e^f is 1-Lipschitz in sup norm, and 0 ≤ Λ(f) ≤ ‖f‖_∞ by Jensen (E_λ f = 0). Hence ‖log q_f‖ ≤ 2‖f‖ and ‖Δ log q‖ ≤ 2‖Δf‖.
(b) The chapter 13 channel has dν_j/dλ(z,φ,ψ) = E_λ[e^{f_j} | Z = z]/E_λ e^{f_j}. The chapter 14 channel is the same with conditioning on (Z, Ψ). In both cases ‖log ν_j‖ ≤ 2‖f_j‖ and ‖Δ log ν_j‖ ≤ 2‖Δf_j‖.
(c) For bounded u, E_{q_f}u − E_{q_g}u = ∫_0^1 Cov_{q_s}(u, f − g) ds, so the difference is at most ‖u‖_∞‖f − g‖_∞ in absolute value. On the ball {‖·‖ ≤ R} this gives |ΔD_ij| ≤ L_D‖Δf‖, with L_D = 4 + 4R for peers and L_D = 2 + 2R + Γ for the self, where Γ = ‖log G_i‖_∞ ≤ J + J²/4.
(d) The softmax Jacobian has ℓ¹ norm at most 2 against sup-norm inputs, so ‖Δβ_i‖_1 ≤ 2L_D‖Δf‖.
(e) Combining (a) to (d) gives ‖Δφ_i‖ ≤ [4 + 2L_D max(Γ, 2R)]‖Δf‖, and therefore

    Lip_R(V) ≤ 8 + 4(4 + 4R) max(Γ, 2R) = 8 + 32R + 32R²    (2R ≥ Γ).

The presentation sector has the same form, with the identity peer channel.

*Well-posedness.* Picard-Lindelöf in the Banach space gives a unique maximal C¹ solution with the blow-up alternative, by the argument of prop:dyn-chart-flow (Dieudonné 1960 as cited there).

*Global existence.* Write V_i = −f_i + P_i. Then ‖P_i‖_∞ ≤ osc(β_ii log G_i + Σ_j β_ij log ν_j + log ℓ_i) ≤ 2 max(J, R(t)) + 3ε, since osc log G = 2J, osc log ν_j ≤ 2‖f_j‖ and osc log ℓ = 3ε. Variation of constants gives R(t) ≤ (R(0) + J + 3ε/2) e^{2t}, so there is no blow-up. The null field φ^0_i = β^0_ii(log q_i − log G_i) − log ℓ_i satisfies the same Lipschitz bound and grows at most linearly, since β^0_ii ≤ 1/5.

*Invariance.* The family curve f^{η(t)} = η(t)·(T − E_λ T) is C¹ into L^∞_0 and solves the chart ODE by the coefficient matching of pa:angular-flow. By uniqueness it is the chart solution. The same holds for lp:law.

*Sanity check.* s5 used a discretized 2x24x24 space with 60 random pairs per radius. The largest sampled Lipschitz ratio was 1.011, against bounds 11.5, 32, 72 and 200 at R = 0.1, 0.5, 1 and 2; also ‖P‖/(2max(J,R) + 3ε) ≤ 0.33. The four-state model is finite, so thm:dyn-finite-flow-type arguments cover it, with G depending on the presentation.
**Fix:** Add to sec:dyn-per-agent a definition of the Gibbs-reduced per-agent chart flow and the proposition above (chart, Lipschitz constant on bounded sets, global existence, uniqueness, and the null version). Cite it in pa:angular-flow, pa:phase-parent and lp:native-field. Say that uniqueness is within C¹ curves in L^∞_0; no weak-strong statement is claimed. Measurability of the flow map for the evaluation σ-algebra remains a separate obligation (GAP04).
**Falsifies:** Two chart points in a ball of radius R whose field difference exceeds (8 + 32R + 32R²) times their distance; or a chart solution from a family point that leaves the family.

### GAP01-S3-02: The persistence proofs are nonquantitative although M is symmetric; a global Lyapunov argument gives explicit ε_0 and entrance times, uniform over records and amplitudes
**Category:** improvement
**Severity:** medium
**Location:** chapters/13_per_agent_inheritance.tex:149-163 (pa:bit-limit and the attracting-ball step of pa:persistent-grouping), :278-294 (pa:angular-limit and the entry step of pa:angular-grouping), :392 ("No numerical bound on ε_0 is asserted"); companion/c13_per_agent_inheritance.tex:131-146; chapters/14_collective_local_parent.tex:71-78
**Evidence:** Both proofs combine three steps: the implicit function theorem to place u_ε near u_*, a "sufficiently small" attracting ball around u_ε, and finite-time continuous dependence from the limit trajectory. I checked each step, and each is valid, but none yields a number.

*The Lyapunov argument.* M = −(7/5)I + (2/5)11^T is symmetric with top eigenvalue −1/5, so the Euclidean distance to u_* is a strict Lyapunov function for the exact field. With e = u − u_* and R_ε(u)_i = Σ_{j≠i}(β_ij − π_ij)u_j,

    d|e|/dt ≤ −|e|/5 + |R_ε(u)|.

The rows satisfy |β_ij − π_ij| ≤ π_ij(e^{D_max,i} − 1) with D_max,i ≤ ε²Δ_i²/8. This follows from KL(Bern(l)‖Bern(l')) ≤ (l − l')²/8, the Bregman remainder of log(2cosh(l/2)), whose second derivative in l is at most 1/4. So R_ε has an explicit supremum R̄(ρ, ε) on B(u_*, ρ), and the ball is invariant when R̄ < ρ/5. Since ‖e^{Mt}‖ = e^{−t/5}, the trajectory shadows the limit: |u(t) − ū(t)| ≤ 5R̄ for all t ≥ 0, uniformly over u(0) in the ball.

*Entrance and persistence.* Connectivity of {1,2,3} through the edges (1,2) and (2,3) is constrained by the margins u_2, u_3, 4u_2 − u_1, 4u_2 − u_3 and 4u_3 − u_2. The limit trajectory is ū(t) = u_* + e^{−t/5}c_1 + e^{−7t/5}c_⊥, with c_1 = −½·1 and c_⊥ = (33/14, −16/14, −17/14). Each binding margin has both coefficients ≤ 0 along it and so increases in t. The remaining constraints have rigorous lower bounds above the error. Once every margin exceeds e^{−t/5}r_0 + 5R̄, it does so for all later t. This proves persistence after entry with the records held fixed, without any appeal to continuous dependence.

*Results* (s1, s2):
- **pa:persistent-grouping**, with r_0 = 0.05. In log-odds coordinates the initial set is εB((4,½,−1), 0.05). It starts as singletons, because the no-edge margin of 4u_2 − u_1 is 2/√17 = 0.485.
  - The ball B(u_*, 3.064) is invariant for ε ≤ 0.162, and R̄ ≈ 22.1ε².
  - The full block holds for all t ≥ T, with T = 4.72 at ε = 0.01, 5.84 at 0.02 and 7.82 at 0.028. At ε = 0.028, 5R̄ equals half the asymptotic margin 5/(7√17) = 0.1732.
  - T is finite for ε below about 0.039.
  - Exact flow: entrance at 3.832 (ε = 0.01) to 3.860 (ε = 0.1), against 3.833 for the limit trajectory; the block persists to t = 60.
- **pa:angular-grouping**. Three exact facts carry the bound:
  - A_i(t) = e^{−t}A_i(0) + (1 − e^{−t})εe(ω_i);
  - |C_i(t)| ≤ ε³ whenever |C_i(0)| ≤ ε³;
  - |L_i/ε − u_i| ≤ δ̄ = 2ε²(max(A_0, ε) + ε³).

  After the common entropy cancels, the reduced peer log-weights are −KL(Bern(L_i)‖Bern(L_j)) and the self log-weight is −KL(Bern(L_i)‖Bern(0)) + Jt_i − log I_0(J). The row oscillation is therefore at most ε²Δ²/8 + 2J + J²/4.
  - With r_0 = 0.05 and A_0 = 1/20: T = 4.73 at ε = 0.01, finite for ε < 0.039, uniformly in ω ∈ T³, θ, b_i(0) > 0 and |C_i(0)| ≤ ε³.
  - The null box needs ε ≤ 1/100, so (ε_0, T) = (1/100, 4.73) is an admissible explicit pair for the whole theorem.
  - Exact-field spot checks (6 random records and amplitudes each at ε = 0.01, 0.05 and 0.2) give singletons at t = 0, a persistent full block, and a latest entrance of 4.12.
- **The O(ε²) gap.**
  - u_ε = u_* + ε²w + O(ε⁴), with w = (1275/2401, 1275/2401, 1725/4802) and |w| = 0.8325 (sympy). fsolve gives (u_ε − u_*)/ε² = (0.531049, 0.531049, 0.359250) at ε = 0.025.
  - The gap is uniform on the stated initial set, which is bounded in u. By the shadowing bound it is also uniform in time.
  - XC7-IMPL's values, 2.13e-2 down to 3.32e-4, are the sup norm 0.531ε²; they agree with this coefficient.
**Fix:** In both proofs and at c13:140-146, replace the paragraph built on the implicit function theorem, the small ball and continuous dependence. Use instead the Lyapunov inequality, the row bound and the monotone-margin entrance argument, and state the constants: at least ε_0 = 1/100 and T ≤ 4.73 for r_0 = 0.05. Note that attraction in the (c, A) directions is not needed for selection: only the u-sector is constrained, and (A, C) enter only through δ̄.
**Falsifies:** A point u(0) ∈ B((4,½,−1), 0.05) and an ε ≤ 0.01 whose exact trajectory lacks the full block at some t ≥ 4.73; or a violation of KL(Bern(l)‖Bern(l')) ≤ (l − l')²/8.

### GAP01-S3-03: The size of the positive-ε nonflat window is fixed by two exact mechanisms the proof never states, and at every positive ε the curvature itself decays, so the finite window is necessary
**Category:** improvement
**Severity:** medium
**Location:** chapters/13_per_agent_inheritance.tex:352-392 (pa:nonflat-window), :328-334 (pa:joint-record-geometry), :389 ("uniformly over the angular episode outcomes"); chapters/14_collective_local_parent.tex:166-178 (lp:nonflat); companion/c13_per_agent_inheritance.tex:233-254
**Evidence:** The proof argues convergence to the ε = 0 forms on [0, T_+], and that "History and label scores vanish at this limiting parameter". It never says what limits ε at a given T_+. Three exact facts settle that.

(i) *The marked sector is linear and decoupled.* A_i(t) = α_i + ε_t e(ω_i), with α_i = e^{−t}A_i(0) and ε_t = ε(1 − e^{−t}). Context enters only through A_1(0) = a e(χ), so ∂_c A_1(t) = e^{−t}∂_c A_1(0) for every ε.

(ii) *Orbit cancellation.* The marked action rotates φ and the stored record angle ω together, so the joint orbit score of agent i is ∂_g log q_i(φ − g; A_i(ω − g), C_i). The record-driven part ε_t e(ω_i) drops out of the rotating vector, because e′(ω)·e(φ) + e(ω)·e′(φ) = 0. What remains is the centered −(α_i + zC_i)·e′(φ), taken under the law vM(A_i + zC_i), which still contains the record part. Its conditional mean given (h, z) vanishes identically, and the history's own orbit score is O(εJ) = O(ε⁴).

(iii) *Inertia floor.* C_i(t) = ε³ c̄_i(t) e_1 with c̄_i ≈ (1 − e^{−t})/5. The locked inertia therefore contains κ/2, with κ = Σ_i |C_i|² ≈ 3ε⁶(1 − e^{−t})²/25. Every contextual score decays like e^{−t}.

In the small-amplitude limit these three facts give

    F_ε ≈ −2a e^{−2t}(e^{−2t}D + κ)/(e^{−2t}(a² + D) + κ)²,   with D = Σ_{i≠1}|A_i(0)|².

*Reduced model.* I built a reduced model (s3) on the exact von Mises law of (z, φ), the joint orbit score of (ii), and a Haar history. It drops label scores, the O(ε⁴) nonuniformity of the history law, and the O(ε³) context dependence of l, C and the rows. At c_0 = (0.04, 0.3) it reproduces D17-PA's brute-force record-inclusive curvature:

| t | ε | reduced model | D17-PA brute force |
| --- | --- | --- | --- |
| 1 | 0.1 | −12.4687 | −12.4687 |
| 1 | 0.05 | −12.4910 | −12.4910 |
| 1 | 0.2 | −12.1273 | −12.1292 |
| 5 | 0.05 | −12.3913 | −12.3914 |
| 5 | 0.2 | −0.4309 | −0.4257 |

With the history rotation removed from the orbit score, which is the geometry at a fixed realized episode, the same model gives −2.97 at (t, ε) = (1, 0.05) and −5×10⁻⁴ at (5, 0.05).

*Leading-order threshold.* I computed the largest ε with |F_ε| ≥ |F_0|/2 on the patch a ∈ [0.02, 0.045], d = 0.03, d_3 = 0.025, for t ∈ [0, T_+]. It is ε* = 0.139, 0.099, 0.071 and 0.036 for T_+ = 4, 5, 6 and 8, so ε* ≈ 0.525 e^{−T_+/3}; in general, ε_0 ∝ e^{−T_+/3}(a_−² + D)^{1/6}. At a = 0.02 and t = 5 the reduced model gives |F_ε|/|F_0| = 0.9996, 0.973, 0.744 and 0.219 at ε = 0.02, 0.05, 0.08 and 0.12.

*Decay at fixed ε.* At fixed ε > 0 the curvature tends to zero. For ε = 0.05 the reduced model gives F = −12.39, −3.57, −1.2×10⁻² and −2.9×10⁻⁵ at t = 5, 8, 11 and 14. For ε = 0.02 it gives −12.49, −12.44, −2.44 and −7.2×10⁻³. The ε = 0 value stays at −12.4928. The reason is structural: K ≥ (1 − o(1))κ/2 > 0, while j and its contextual derivatives are O(e^{−t}). So the limits t → ∞ and ε → 0 do not commute, and nonflatness at positive ε is transient, not merely unproved beyond the window.

This sharpens D17-PA-02 and 2026-10-22 T5. At zero coupling T5 found "loss of rank, not loss of curvature"; at positive ε the curvature is lost as well. All numbers in this finding are floating point: they support the scaling and do not prove it.
**Fix:** After pa:nonflat-limit, state (i) to (iii), the leading-order law, and the tradeoff ε_0 ∝ e^{−T_+/3}(a_−² + D)^{1/6}. Say that the joint history experiment is what keeps ε_0 polynomial in e^{−T_+}; a fixed-episode geometry would need ε ≪ e^{−T_+}a. At 13:389, replace "uniformly over the angular episode outcomes", which invites the fixed-episode reading, with "uniformly in the conditional integrands over the compact episode space". Add one sentence saying that for fixed ε > 0 the curvature of the evolved section tends to zero.
**Falsifies:** A record-inclusive computation at (t, ε) = (5, 0.12) and a = 0.02 with |F_ε|/|F_0| > 0.5; or a fixed-ε trajectory whose curvature stays bounded away from zero as t → ∞.

### GAP01-S3-04: The window proofs justify convergence of the contextual jets by bounds alone
**Category:** gap
**Severity:** low
**Location:** chapters/13_per_agent_inheritance.tex:382-385 (proof of pa:nonflat-window); chapters/14_collective_local_parent.tex:178 (lp:nonflat)
**Evidence:** The proof says the first two contextual derivatives "converge uniformly to the displayed limit as ε→0: differentiating the flow gives Ẋ_c = DV(X)X_c and Ẋ_cc = DV(X)X_cc + D²V(X)[X_c,X_c], whose integral inequalities bound the derivatives uniformly". Uniform bounds give precompactness, not convergence to the ε = 0 jets.

The conclusion is nonetheless true. The family field V(x, ε) (pa:angular-ode with pa:angular-weights, I_0(ε³), εo and εe(ω)) is real-analytic jointly in (x, ε) on the tube. The initial section x_0(c; ε), with l = εu^in and A_1 = ae(χ), is analytic in (c, ε). So the flow and its c-derivatives up to order two depend continuously on ε, uniformly on compact (c, t, ω) sets. That is standard C^k dependence on parameters; see Hartman, *Ordinary Differential Equations*, ch. V. I recall this from memory and did not recheck the exact theorem number.

The Fisher integrands are then jointly continuous on a compact sample space with positive densities, and the conditioning event has Pr(E) = 1/8 > 0 at ε = 0. The C² regularity the curvature needs (log p twice differentiable in context, with continuous dependence on (c, t, ε, sample)) therefore holds. The companion gives the right justification at c13:254 ("Analytic finite-time dependence of the flow and its contextual derivatives"). For chapter 14, the same argument in (ε, δ) applies to C² sections, which is what 14:178 asserts.
**Fix:** Replace the clause with "the field is analytic in (x, ε), so the flow and its first two contextual derivatives depend continuously on ε uniformly on the compact tube", and cite a parameter-dependence theorem.
**Falsifies:** A discontinuity in ε of X_cc at ε = 0 on the tube.

### GAP01-S3-05: The chapter 14 grouping import needs an absolute bound on the hidden amplitude and parameter ranges that the chapter does not state
**Category:** consistency
**Severity:** low
**Location:** chapters/14_collective_local_parent.tex:71-80 (grouping import and null box), :168 (lp:nonflat ranges); companion/c13_per_agent_inheritance.tex:233
**Evidence:** 14:71 imports "the interaction-dependent grouping window of pa:angular-grouping". That theorem concerns a model in which ψ is resampled from Haar measure, not transmitted. The adaptation argument at 14:71-78 is valid, but its row correction is O(δ²), not O(ε²). Relative to the common term, the extra peer log-weight is −KL(vM(B_i)‖vM(B_j)) and the self picks up −KL(vM(B_i)‖Haar).

The explicit argument of GAP01-S3-02 therefore needs an extra oscillation term 2(δ max|b_i|)². With it, s6 gives a finite explicit T only for δ max|b_i| ≤ 0.055 at ε = 0.01 (T = 5.41 at 0.02, 13.41 at 0.05, ∞ at 0.08). That is tighter than the null box's |B_i| ≤ 1/10 (14:80). The bound is worst-case. When the b_i are nearly aligned in a pointed cone, the peer weights rise, β_ii falls, and the equilibrium ratio rises above 1/3. When β_ii rises toward 1/3, the ratio falls to 0.

The chapter states neither the δ range nor ε ≤ 1/100, the latter being implicit in "the self KL bound of pa:null-bound". lp:nonflat takes "a, d in positive compact ranges independent of ε, δ" without the smallness ("fixed small positive", 13:355) that the grouping set |A_i(0)| < 1/20 and the null box |A_i| ≤ 1/10 require; c13:233 likewise drops it. D18-LP-14 reports the range part; the δ part is new. Once these ranges are added, no statement at 14:66-90 is stronger than what chapter 13 proves.
**Fix:** In 14:71-80 state 0 < ε ≤ 1/100, δ max|b_i| ≤ 1/20, |A_i(0)| ≤ 1/20 and |C_i(0)| ≤ ε³. In lp:nonflat and at c13:233, restore "small" with a_+, d < 1/20.
**Falsifies:** A grouping failure, explicit or numerical, from the stated initial set at δ max|b_i| ≤ 0.05 and ε ≤ 0.01; or a null violation inside the box.

### GAP01-S3-06: The four-state open set is rigorous at radius 0.1η for η = 10⁻³; it is an O(η) ball around the product-family point, inside the null cone
**Category:** improvement
**Severity:** low
**Location:** chapters/13_per_agent_inheritance.tex:96-126 (pa:four-grouping)
**Evidence:** Work in centered log-table coordinates with the norm ‖x‖² = Σ_blocks ¼Σ_z x_z².

*Linearization.* At the uniform table the linearization is exactly symmetric (s4: ‖A − A^T‖ = 0), with spectrum {1/4, −3/4, −5/4, −9/4} ∪ {−1 ± κ/2}. Hence ‖e^{At}‖ = e^{t/4}.

*Remainder bound.* The remainder N = V − Ax − F_H satisfies the analytic bound ‖N(x)‖ ≤ 2Nb(2‖x‖). It rests on three estimates:
- the centered log(softmax(f)W) differs from fW by at most ‖f‖²_∞/2 in sup norm, because the log-sum-exp remainder is nonnegative;
- |E_{softmax(g)}b − ⟨b,g⟩| ≤ ‖g‖²_∞;
- for ρ_q = cosh J/(1 + cosh J) and ρ_s = ½ the rows are exactly β_12 = σ(KL(q_1‖u) − Jm_1n_1 − KL(q_1‖q_2W)) and γ_12 = σ(KL(s_1‖u) − KL(s_1‖s_2W)). They deviate from ½ only at second order and so enter N only at third order.

The sampled ratio ‖N‖/‖x‖² is at most 0.334, against the bound 8.04 at small ‖x‖.

*Result.* A Volterra bootstrap on e = x − e^{At}x(0) with records |H| = h gives the following. Take η = 10⁻³ and h = 10⁻⁵. For every initial table with ‖x(0) − x_base‖ ≤ 10⁻⁴ = 0.1η, all eight conditional means exceed η/(8√2) for every t ∈ [1.70, 3.00]. The bound was evaluated with a 301-point upper Riemann sum in double precision, so it is not interval-validated. The initial table is unselected, and the null never selects: 2r_0 < atanh η keeps every conditional aligned with its record, and the cones of pa:four-cone are invariant.

*Exact-flow tests* at η = 0.02 and h = 2×10⁻⁴, with 8 random directions per radius: selection on [0, 3] in 8/8 and null selection in 0/8 at radii 0.1η, 0.25η, 0.5η and η. The true radius is therefore at least about η in typical directions.

*Limits on the set.* The cone argument cannot enlarge the open set past the null cone, and the threshold η/(8√2) is tied to η (D17-PA-04 notes that a sign-only selector works). The full open set the theorem claims exists, but it is an O(η) ball of tables within about 4.5η of the uniform table.
**Fix:** Optionally state the explicit triple (η, h, r_0) = (10⁻³, 10⁻⁵, 10⁻⁴) and the window, and say that the open set is a small neighborhood of a near-uniform unstable equilibrium.
**Falsifies:** An initial table within 10⁻⁴ of x_base, at η = 10⁻³ and h = 10⁻⁵, whose exact flow lacks the selection at some t ∈ [1.70, 3.00].

### GAP01-S3-07: The full-flow half of pa:angular-grouping needs no smallness of marked or hidden amplitudes; only the null box does
**Category:** improvement
**Severity:** low
**Location:** chapters/13_per_agent_inheritance.tex:242 (statement), :256-276 (null box), :292 (initial choices)
**Evidence:** In the explicit argument the marked amplitudes enter the u-sector only through δ̄ = 2ε²(max(A_0, ε) + ε³), and the hidden concentrations do not enter at all. A_0 = 0.5 gives T = 4.75 at ε = 0.01 and a finite T for ε < 0.0375 (s2).

The "fixed small positive compact patch" is needed only by pa:null-bound, through |A_i| ≤ 1/10 and b_i ≤ 1/10 for D_ii < log 2. The null sign argument itself needs much less. First, o_i l^0_i ≥ min(o_i l^0_i(0), 5ε), because β^0_ii ≤ 1/5 always. Second, |L − l| ≤ 4ε³(|A^0| + 2ε³). Third, for |A_i(0)| ≤ 1 the box {1/4 ≤ o_i l_i/ε ≤ 60, |A_i| ≤ 1.1, b_i ≤ 1/10, |C_i| ≤ 2ε³} is forward invariant for ε ≤ 1/100. There D_ii ≤ 2.62 and β^0_ii ≥ 0.0178, and the faces are inward: 1 − 60β^0 < 0 and −1.1β^0 + ε < 0. So the third public sign stays negative for these larger amplitudes as well.
**Fix:** State that the full-flow conclusion holds uniformly over any compact marked patch and all b_i(0) > 0, and confine the smallness assumption to the null estimate.
**Falsifies:** A marked amplitude |A_i(0)| ≤ 0.5 and ε ≤ 0.01 for which the full block fails to persist.

### GAP01-S3-08: Extension: an explicit nonflat window at ε = 1/100 is within reach
**Category:** extension
**Severity:** medium
**Location:** chapters/13_per_agent_inheritance.tex:352-392 (pa:nonflat-window); chapters/14_collective_local_parent.tex:166-178 (lp:nonflat)
**Evidence:** At ε = 1/100 and T_+ = 6, every term the reduced model of GAP01-S3-03 drops is very small:
- the context dependence of l and C is O(ε³e^{−t/5});
- the label contextual and orbit scores are O(ε⁴e^{−t}) in amplitude, hence O(ε⁸) in Fisher information;
- the history orbit score is O(ε⁴);
- the inertia floor κ is about 10⁻¹³, against K_0 ≈ 6×10⁻⁹.

Relative corrections are therefore at most about 10⁻⁶. Crude remainder constants, even 10³ times too large, would close a rigorous window [0, 6] on a ∈ [0.02, 0.045]. Four ingredients are needed:
1. Lipschitz bounds of the conditional Fisher integrals in the von Mises parameters, from cumulant bounds of e(φ) (‖∇^k log I_0(|·|)‖ ≤ c_k).
2. Variational-equation bounds for ∂_c l, ∂_c C, ∂_ω C and ∂_ω l from the l-system, whose homogeneous part decays at rate at least β_ii − O(ε²).
3. The row bounds of GAP01-S3-02.
4. Validated quadrature on the torus, where the integrands are analytic and periodic.

Combined with GAP01-S3-02, this would give one explicit triple (ε_0, T, T_+) = (1/100, 4.73, 6) for grouping, null control and nonflat window together. Difficulty: moderate, about a page of estimates.
**Fix:** Carry out the four estimates and state the triple.
**Falsifies:** An explicit remainder bound that exceeds the margins at ε = 1/100.

## Checked without defect

| Label | What was checked | Method | Residual or reasoning |
| --- | --- | --- | --- |
| pa:four-transfer (pa:four-potentials, pa:four-null) | Own-block and null potentials from the reduced per-agent objective; β_ii on both evaluator derivatives; entropy coefficients sum to one; label-preserving null coefficients b^0, g^0 | derivation by hand | exact |
| pa:four-stationarity | ν_j = q_jW = q_j on the synchronized family; full field equals null field there; atanh m = Jn and γ_ii atanh n = β_iiJm | derivation | exact |
| pa:four-linear, pa:four-crossing | Self rows ½ at the uniform table for ρ_q = cosh J/(1 + cosh J), with zero first variation; 12x12 linearization symmetric in the u-norm; spectrum {1/4, −3/4, −5/4, −9/4} ∪ {−1 ± κ/2}; crossing at log 2; values η(3/√2, 1/√2) at log 4 | derivation; s4 central differences | ‖A − A^T‖_max = 0; eigenvalues to 1e-6 at κ = ½ |
| pa:four-cone | Null conditional-mean equations; first-exit cone argument. I also derived the full-flow analog, dc_i(x)/dt = −(1 − c²)[atanh c − ½Σβ_ij λ_j(x) − β_iiJn_i − H_i^q], with λ_j(x) = log ν_j(1,x) − log ν_j(−1,x) | derivation | exact |
| pa:four-grouping | Open-set claim at an explicit radius | s4 | see GAP01-S3-06 |
| pa:bit-flow | Fisher inverse cancels dl/dq(+1); peer KL contributes l_i − l_j; record contributes ε o_i | derivation | exact |
| pa:bit-limit | M, u_* = (15/7, 15/7, 5/7), spectrum {−1/5, −7/5, −7/5}; O(ε²) coefficient w | sympy (s1) | exact; fsolve agrees to 2e-5 at ε = 0.025 |
| pa:selector | Initial same-sign ratio 1/8; equilibrium minimum ratio 1/3; distance 5/(7√17) = 0.1732 from u_* to the binding face | s1 | exact |
| pa:persistent-grouping | Full reconstruction with explicit ε_0 and T, ball invariance, persistence after entry | s1 | see GAP01-S3-02 |
| pa:bit-null | Null is scalar and autonomous per agent; the sign of o_i l_i is preserved; the full block never forms | derivation | exact |
| pa:angular-family, pa:angular-normalizer, pa:public-odds | z-marginal log odds L_i = l_i + h(A_i + C_i) − h(A_i − C_i); density relative to the uniform sign reference | derivation | exact |
| pa:angular-flow (pa:angular-ode) | Coefficient matching; invariance, given uniqueness from GAP01-S3-01 | derivation | exact |
| pa:angular-weights and 13:239 | Entropy cancellation; ∂_{L_j}β_ij = β_ij(1 − β_ij)(m_i − m_j)/2; ∂_αL > 0 from h_1'' = Var > 0 | derivation | exact |
| pa:angular-null | Label-preserving null obtained as the coefficients of the reduced null objective | derivation | exact |
| pa:null-box, pa:null-bound, pa:null-sign-bound | |T|² = 4, so the Hessian is at most 4; D_ii ≤ 2|η − η_G|² ≤ 72ε² + 1/25 + 18ε⁶ < log 2; β^0 ∈ [1/9, 1/5]; faces inward (r: 1 − β^0r; A: ε − 1/90; C: −β^0ε³; b decreasing); |L − l| ≤ 2|C|(|A| + |C|) ≤ (2/5)ε³ + 8ε⁶ < ε/8; o_iL_i ≥ ε/8 | derivation by hand, independent of XC7-IMPL and D17-PA | holds as stated for ε ≤ 1/100 |
| pa:angular-limit | L_i/ε = u_i + O(ε²|c||A|); rows π + O(ε²) for fixed nonzero A; derivative diag(M, −I, −I) + O(ε²) | derivation | orders exact; explicit bounds in GAP01-S3-02 |
| pa:angular-grouping (quantifiers) | Statement: ∃ε_0 ∀ε < ε_0 ∃ relatively open set ∀ω, uniformly on the (A, b) patch. The proof delivers this order | reconstruction; s2 | matches; explicit (ε_0, T) = (1/100, 4.73) |
| pa:inertias, pa:nonflat-limit | Minimization of e^{−2t}V da² + I(dχ + ξ)² + Bξ²; A_0, F_0, h^0; I' = I_1/I_0 + xV; phase Fisher identity | derivation; s3 closed form equals D17-PA's | F_0 = −12.4925 (t = 1), −12.4928 (t = 5) |
| pa:nonflat-window | Qualitative continuity argument | reconstruction (with the repair of GAP01-S3-04); s3 | true; unquantified (GAP01-S3-03, -08) |
| lp:native-field (lp:weights, lp:field) | Peer cost carries −B_j·M(B_i) + h_0(B_j); coefficient matching | derivation | exact |
| lp:selection-limit | M, u_*, O(ε² + δ²) remainder; B-block limit matrix M per component | derivation; s6 | correct; explicit δ constraint in GAP01-S3-05 |
| 14:80 null box | Same bound with |B_i| ≤ 1/10; dB^0/dt = −β^0 B^0; strip invariance | derivation | holds for ε ≤ 1/100 |
| lp:relative-identification | c_ij = ζ_i conj(ζ_j); |ζ_i|² = |c_ij||c_ik|/|c_jk| for nonzero ζ | derivation | exact |
| lp:nonflat (ε = δ = 0) | Same minimization as pa:nonflat-limit, with B renamed D | derivation | exact |
| companion c13:122-258 | cp:selector, cp:affine, cp:attraction, cp:null-parameters (β^0 = e^{−D}/(e^{−D} + 4)), cp:null-box, cp:null-signs, cp:experiment, cp:history-fisher, cp:cost-isometry, cp:inertias and cp:curvature, each compared with the main text | comparison and derivation | consistent; c13:233 drops "small" (GAP01-S3-05); c13:254 gives the correct analytic-dependence justification |

### Answers to the focus questions

| Question | Answer |
| --- | --- |
| 1. pa:persistent-grouping: explicit ε_0, entrance time, persistence, O(ε²) gap | Reconstructed (GAP01-S3-02). Explicit ε_0 = 0.028 with T = 7.82, and T = 4.72 for ε ≤ 0.01, from initial set εB((4,½,−1), 0.05). Persistence after entry follows from monotone margins under a global Lyapunov bound, with the records fixed. The O(ε²) gap holds uniformly on the initial set and, through shadowing, uniformly in time. Exact coefficient w = (1275/2401, 1275/2401, 1725/4802). |
| 2. pa:angular-grouping: uniformity and quantifier order | Uniform over ω ∈ T³, θ, b_i(0) > 0, |C_i(0)| ≤ ε³ and any bounded marked amplitude for the full flow; the null needs a box (GAP01-S3-07). The quantifier order (∃ε_0 ∀ε ∃U ∀ω) matches what the proof delivers. Explicit pair (1/100, 4.73). |
| 3. Positive-ε windows of pa:nonflat-window and lp:nonflat | True as qualitative statements; the continuity argument is reconstructed, with a small repair (GAP01-S3-04). Quantified only at leading order (GAP01-S3-03): ε*(T_+) ≈ 0.525e^{−T_+/3} on the declared patch, and at fixed ε > 0 the curvature tends to zero. For lp:nonflat, δ_0 is not constrained by the geometry at leading order, only by the grouping (δ max|b_i| ≤ 0.055) and the null (≤ 0.1). The C² regularity holds: the field is analytic and the chapter 14 sections are C². A rigorous explicit ε_0 is not established (GAP01-S3-08). |
| 4. Chart flow on {−1,1} x S^1 x S^1 | Existence, uniqueness and global continuation proved, with the chart L^∞_0 and Lip_R ≤ 8 + 32R + 32R² (GAP01-S3-01). |
| 5. pa:four-grouping open set | Explicit radius 0.1η at η = 10⁻³, h = 10⁻⁵, with selection guaranteed on [1.70, 3.00] (GAP01-S3-06). The claim needs no smaller set than stated; the set is an O(η) ball inside the null cone. |
| 6. Chapter 14 statements at 14:66-90 | No statement there is stronger than chapter 13 proves, once ε ≤ 1/100, δ max|b_i| ≤ 0.055 (sufficient) and the amplitude ranges are stated; they are currently unstated (GAP01-S3-05). |

## Prior-review findings in scope

Chapters 13 and 14 did not exist on 2026-10-22. The rows below are earlier findings whose subject recurs in this scope.

| 2026-10-22 ID | Status | Evidence |
| --- | --- | --- |
| L03-DYN-03 (Fisher flow defined and well posed only on finite tables) | partially resolved | prop:dyn-chart-flow (03_dynamics.tex:228-232) covers the reciprocal potential. The Gibbs-reduced per-agent chart flow on {−1,1} x S^1 x S^1, used at 13:214-295 and 14:43-69, is still neither stated nor proved in the text; GAP01-S3-01 supplies the proof. |
| T5 (contextual geometry prepared and decaying) | unresolved, recurs | The per-agent curvature is prepared in A_1(0) and present at ε = 0 for all t (D17-PA-02). At positive ε the curvature also decays to zero (GAP01-S3-03), so the obstruction is no longer only a loss of rank. |

## Extensions and improvements

**An explicit joint window.** The most valuable next step is GAP01-S3-08. It would give one explicit triple (ε_0, T, T_+) = (1/100, 4.73, 6) under which grouping, null control and nonflat geometry hold together, and it would remove "sufficiently small" from the per-agent chapter. The estimates needed are listed in that finding. Two facts keep it short: the marked sector is exactly solvable, and the joint orbit score cancels the record-driven amplitude. Every remaining term is a product of a small power of ε and a factor that decays in t.

**Persistent four-state grouping on an explicit set.** Combine the symmetric linearization of GAP01-S3-06 with a trapping box on the invariant product family q_i = (1 + m_i a)/4, s_i = (1 + n_i b)/4. In φ = atanh m and ψ = atanh n the dynamics on that family is φ̇_i = −φ_i + β_ij φ_j + β_ii J tanh ψ_i + H^q_i and ψ̇_i = −ψ_i + γ_ij ψ_j + β_ii J tanh φ_i + H^s_i. The rows are given exactly by the logistic representations of GAP01-S3-06. On the face φ_i = δ of a positive box the field is at least β_ii(J − 1)δ − h + O(δ³), which is where J > 1 enters. Transversal directions contract at rates 1 ± κ/2, so a tube around the family stays attracted. Together with the explicit window this would upgrade pa:four-grouping to persistent grouping on a full open set of positive laws, the stronger (F1) that D17-PA-04 observed numerically. Difficulty: moderate.

**Sharper explicit thresholds.** The explicit ε_0 of GAP01-S3-02 is conservative by about an order of magnitude: the exact Bernoulli flow still groups at ε = 0.3, and D17-PA finds failure at ε = 2. A Lyapunov function that weights the common mode (rate 1/5) and the difference modes (rate 7/5) separately would sharpen it. So would using the sign of the row deviation: at u_*, peers with similar log odds are favored, so R_ε has a definite direction. Both would move the explicit threshold toward the true failure scale. Difficulty: low.

**An ε-independent initial class.** The theorems scale the initial class with ε (l(0) = εu(0)). For l(0) = c(4, ½, −1) with c of order one, the record-free part of pa:bit-flow does the work first. The common mode relaxes at rate β_ii and the differences faster, so all public signs become positive, with ratios near one, before the records matter. A two-time-scale argument should then give persistent grouping from a fixed open set of laws for all small ε; the null keeps every sign. This removes the ε-scaling, but the records then play no part in forming the group, so it does not strengthen "against conflicting evidence". Difficulty: moderate.

## Not checked

- I did not establish a rigorous explicit ε_0 for pa:nonflat-window or lp:nonflat. What I have is the qualitative continuity argument, the leading-order law and the floating-point reduced model.
- I used no interval or validated arithmetic. The explicit constants are closed-form inequalities evaluated in double precision, and the four-state Volterra bound uses an upper Riemann sum.
- The δ-dependence of the lp:nonflat geometry at positive δ was argued, not computed. The argument is that the rows' O(δ²) terms reach the marked geometry only through β_ii in κ and through the l-sector.
- Measurability of the flow map for the evaluation σ-algebra (GAP04), weak-strong uniqueness outside the L^∞-chart class, and differentiability in quadratic mean of the history experiment were not checked. The last I assumed from compactness and positive densities.
- I did not reconstruct the dynamic-row variant of def:dyn-per-agent-flow; D17-PA ran it numerically.
- pa:phase-parent, pa:geometry-inheritance, pa:joint-record-geometry and pa:hidden-record-obstruction were read. I checked θ-independence and the role of chart-flow uniqueness in the intertwining claim, but did not re-derive the adaptive-history induction or the cost isometry.
- Chapter 14 outside 14:43-90 and 14:166-178 was not checked.
- The Hartman parameter-dependence theorem was cited from memory, and its locator was not rechecked; Dieudonné is taken as the manuscript cites it.
- The selector-threshold dependence (XC2-FRAME-02) and the null-convention question (XC3-SCOPE-01) were not re-examined.
- I did not test the four-state construction for N > 2 or for other parameter values.
