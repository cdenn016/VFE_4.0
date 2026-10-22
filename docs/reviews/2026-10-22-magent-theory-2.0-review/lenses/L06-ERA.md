# L06-ERA: State-controlled erasure and public compression

I read in full `chapters/08_smooth_erasure.tex` (lines 1-222), `chapters/09_public_compression.tex` (1-206), `companion/c08_smooth.tex` (1-76; the erasure part is 66-76, the rest read for context), `companion/c09_public_compression.tex` (1-49), `README.md`, `AUTHORING.md`, `chapters/03_dynamics.tex` lines 1-150 (potential, Gibbs reduction, envelope formula, Fisher flow), and the derivation records `Research/docs/research/2026-10-21-public-compression.md` and `2026-10-21-smooth-kl-formation.md`. I skimmed `chapters/07_smooth_closure.tex` (1-40), `frontmatter.tex`, the outlook, and PIFB2 `24_section.tex` by grep. Verdict: no load-bearing computation in either chapter failed; every displayed identity I recomputed holds (residuals below), the gate-closure and box-invariance arguments close with the stated parameter ordering, and the phase theorem's proof is correct in the Banach chart. The defects are of strength and scope. The theorem titled "Persistent heterogeneous pairs" has provably transient heterogeneity (in-pair separation tends to zero for every admitted trajectory). The erasure inheritance discards a dynamically decoupled factor and a finite label symmetry, and the parent is not an agent of the children's type. The collective phase is a Banach-chart instance of the classical asymptotic-phase fibration with a removable hypothesis `|J|<1/4`. The Gibbs identifiability result is correct, and its scope with a fixed non-identity channel is exactly the injectivity of `q -> qK` on the exposed laws.

Scripts (all single-threaded, each under 2 s) are in `docs/reviews/2026-10-22-magent-theory-2.0-review/scripts/L06-ERA/`: `s01_erasure_symbolic.py`, `s02_erasure_simulation.py`, `s03_product_invariance.py`, `s04_phase_binary.py`, `s05_phase_general_linearization.py`, `s06_gibbs_identifiability_scope.py`, `s07_heterogeneous_coupling.py`, `s08_block_phase_rates.py`. Numerical agreement supports the identities checked; it proves none of the general statements.

## Findings

### L06-ERA-01: The "persistent heterogeneous pairs" lose their heterogeneity; in-pair separation tends to zero for every admitted trajectory
**Category:** inflation
**Severity:** medium
**Location:** chapters/08_smooth_erasure.tex:126-127 and :166 (thm `erase:pairs`, its heading); README.md:18 ("persistent heterogeneous pairs"); companion/c08_smooth.tex:2 (chapter title) and :72
**Evidence:** After the later closure time tau the public system is the unit-Fisher gradient flow of `E_pub` (eq `erase:separated-energy`), so `dE_pub/dt = -sum_i (1-x_i^2)(d_{x_i}E_pub)^2 - sum_i (1-y_i^2)(d_{y_i}E_pub)^2 <= 0`, with equality only at critical points. The product of the two closed boxes, `Q = ([l,u]^2)^2 x ([-u,-l]^2)^2`, is compact and forward invariant (proved in the theorem). By LaSalle, the omega-limit set lies in the critical set of `E_pub` in `Q`. At `eps = 0` the public field decouples; in a box its zeros solve `atanh x = J y`, `atanh y = J x`. Since `g = atanh(.)/J` is increasing, `x = g(g(x))` forces `x = g(x)`, and `atanh(x)/x` increases from 1, so `x = y = 1/2` is the unique zero; it is nondegenerate (local Fisher-Hessian eigenvalues 0.176 and 1.824 at `(1/2,1/2)`). For fixed `T` and `eps -> 0`, the peer field and its derivative are `O(eps^2)` uniformly on `Q` (the Gibbs-row derivative adds `O(eps^4/T)`), so for small `eps` the zero set in `Q` is one nondegenerate point. The transposition of two box-mates maps zeros to zeros, so that point has box-mates equal. Every admitted trajectory therefore converges, exponentially, to a configuration in which the two members of each pair coincide, and `eta_I` converges to a Dirac mass. Numerics (s02; `delta=0.04, rho=0.01, eps=0.015, zeta=0.5`, `T=1` and `T=1e-4`): in-box public separation 1.15e-2 at t=0, 3.3e-4 at t=20, 1.1e-11 at t=120; fitted decay rate 0.171 against the local slow eigenvalue 0.176; final public means `+-0.49779` for all four agents at `T=1`. The theorem text ("does not give a uniform lower bound") is accurate but understates what is provable: only the source preference persists; heterogeneity is a finite-time property.
**Fix:** Retitle the theorem ("Changing and persistent source preferences"), add the proposition above (convergence to the symmetric two-point configuration with box-mates equal, for fixed `T` and small `eps`), and replace "persistent heterogeneous pairs" in README and companion by "persistent preferred pairs whose members are distinct at every finite time and converge to each other". If persistent heterogeneity is the goal, see L06-ERA-09.
**Falsifies:** An admitted trajectory (fixed `T`, small `eps`) whose in-pair separation has positive liminf, or a second critical point of `E_pub` in `Q` for arbitrarily small `eps`.

### L06-ERA-02: Finite-time erasure and the preference change are fixed by the constitutive scale ordering
**Category:** weakness
**Severity:** medium
**Location:** chapters/08_smooth_erasure.tex:32-52 (`erase:flat-step`, `erase:gate`, `erase:compared-law`), lem `erase:gate-entry` (:68-120), :169, :222; companion/c08_smooth.tex:68-72
**Evidence:** (a) The gate means obey `hdot_i = -2(1-h_i^2)(atanh h_i - atanh h_*)` plus a correction bounded by `B rho M_chi/delta`, which the proof forces below `m/2`. My constants: `m = 1.32883`, `M = 1.65073`, `V = 2.31514`, `M_chi = 2`, `B = 0.981818` (sharp: the grid maximum of `|d_alpha d|/rho` equals `B`); admissible `delta < 0.0478`, then `rho <= m delta/(2 B M_chi)` (1.35e-2 at `delta = 0.04`). The gate mean is an internal clock, monotone and decoupled up to a perturbation chosen to be dominated; closure occurs by `3 delta/(2m) <= 0.054` (simulated 0.016-0.018). "There is no external time schedule" (:44) is true, but the schedule is internal and set by the initial gate positions and `h_*`. (b) The private factor of the compared law is `b_{alpha rho z_j}`: the channel ignores the source's private sample and uses its mean times `alpha`, so at `alpha = 0` every private cost equals `psi(z_i)` by definition. (c) The public output kernel `(1-eps) b_{x_i} + eps delta_k` mixes in the receiver's own law, which makes every public peer cost and force `O(eps^2)`; with `eps^2 << rho << delta`, public peer interaction can only rank sources. (d) The two final groups are the two wells of the bistable single-agent energy `psi(x)+psi(y)-Jxy` with `J = log 3 > 1` (minima at `x = y = +-1/2`, saddle at 0), and membership is the supplied initial well. The chapter says the mechanism is engineered (:5, :222); the weakness is that both conclusions (exact decoupling at finite time, the crossed-to-box switch) follow by first-order comparison of prescribed scales, so the example does not test whether the KL dynamics can produce either.
**Fix:** State in the lemma that the gate is an autonomous relaxation perturbed by `O(rho M_chi/delta)`, and in the theorem that the preference change is a switch between the scales `rho zeta^2` and `eps^2`. A less prescribed variant (gate driven by a public statistic) is sketched in L06-ERA-09.
**Falsifies:** A version in which closure or the switch survives without `eps^2 << rho << delta`, or in which group membership is not the initial well.

### L06-ERA-03: At fixed temperature the preferred groups are argmax distinctions in rows that are uniform to O((rho+eps^2)/T)
**Category:** weakness
**Severity:** medium
**Location:** chapters/08_smooth_erasure.tex:30, :148-164 (thm `erase:pairs`), :169, :184-198 (`erase:mixed-term`)
**Evidence:** All row costs are `O(rho) + O(eps^2)` and the gains are sent to zero at fixed `T`, so every Gibbs row is within `O((rho+eps^2)/T)` of uniform. s02 at `T = 1`: initial maximal row weight 0.3342; same-box weight after closure 0.3334. The invariance, closure and ranking arguments only use that weights lie in `[0,1]`, so `T` can be chosen after `eps`. At `T = 1e-4` the same run gives initial crossed weight 1.0000 and same-box weight 0.683 just after closure, increasing as box-mates converge. From `erase:cost-comparison`, after closure `beta_same >= 1/(1 + 2 exp(-Delta/T))` with `Delta = eps^2 [2 l^2 - (u-l)^2/(2(1-u^2))] = 0.4797 eps^2` (0.596 predicted at `T = 1e-4`, 0.683 observed); before closure the crossed partner has weight at least `1/(1 + 2 exp(-(rho zeta^2 - 2u^2 eps^2/(1-u^2))/T))`. The mixed-term witness is a cumulant expansion in `eps^2/T` and `rho/T`, valid only when the gains are small relative to `T`; for `T <~ eps^2` non-autonomy before closure needs a different (direct) comparison.
**Fix:** Put `T` into the parameter ordering and state the explicit row bounds; if near one-hot selection is wanted, take `T <= c eps^2` and replace the witness by a direct comparison of forces under two private assignments.
**Falsifies:** An admitted parameter choice at fixed `T` where some row weight stays bounded away from 1/3 as the gains tend to zero.

### L06-ERA-04: The erasure parent is a decoupled-factor projection plus a finite label quotient, not an agent of the children's type
**Category:** weakness
**Severity:** medium
**Location:** chapters/08_smooth_erasure.tex:173-182 (`erase:separated-energy`), :200-222 (`erase:parent-state`, prop `erase:parent-flow`)
**Evidence:** After tau the state space is (8 public means) x (16 private and gate means) with separated energy and metric; the second factor relaxes autonomously to `z = w = 0`, `h = hhat = h_*`. Discarding a decoupled autonomous factor is projection onto a factor of a product flow, the most elementary exact lumping. The retained `(eta_I)_I` is the whole public configuration modulo `S_2 x S_2`, so no public degree of freedom is lost (the chapter says so). `V[v;M]` depends on the counting measure of both pairs, so a single `eta_I` has no autonomous dynamics; only the pair of parents is closed. The parent carries no law pair `(q_I, s_I)` with an evaluator and KL/Fisher dynamics; the chapter shows the mixture marginals do not close (:220). The parent is therefore not an object on which the potential of ch. 3 is defined, and nothing here can be iterated. By L06-ERA-01, `eta_I` also converges to a Dirac mass.
**Fix:** Describe the result as finite-time exact decoupling of an internal factor; for the recursion program, state the missing requirement explicitly: a retention whose image is again a law pair carrying an inherited KL interaction. L06-ERA-09 and the Extensions section sketch a construction of that type.
**Falsifies:** An inherited KL/Fisher law-pair dynamics for `(q_I, s_I)` on this family.

### L06-ERA-05: The collective-phase theorem is the classical asymptotic phase of a hyperbolic sink with a spectral gap and compresses only transients
**Category:** weakness
**Severity:** medium
**Location:** chapters/09_public_compression.tex:43-113 (thm `thm:public-phase`), :111-113; companion/c09_public_compression.tex:4, :33
**Evidence:** The proof uses: the diagonal is the fixed-point subspace of `S_N` (hence invariant); the common reference is a sink with common rate `1-|J|` and difference rate `1+kappa_N-|J|`; the diagonal flow is globally Lipschitz. `Pi` is projection along strong-stable fibers onto the invariant diagonal (the rate-condition fibration of Fenichel, or the slow Koopman eigenfunctions of the sink). No KL-specific identity enters beyond these rates. Every configuration in `U` converges to the common reference (:111-112), so every fiber and its parent converge to one point, and the compression concerns the decaying part of transients; `Pi` has no finite evaluation (:113). The proof is correct (reconstruction in the table): `V` is analytic on `L^infty` balls because `u -> e^u` is analytic in the Banach algebra `L^infty`; on a 4-point alphabet with `N = 3`, `T_q = 0.7`, `T_s = 1.9` (s05) the linearization matches `A h_i - kappa_N(h_i - hbar)` with `kappa_3 = 3` to 2e-10, the averaged derivative annihilates zero-sum directions at a nonzero diagonal point (5.6e-11 against a derivative scale of 7.2), the residual is quadratic (`|g|/s^2 = 1.6148, 1.6123, 1.6112` at `s = 2e-2, 1e-2, 5e-3`), and the numerically computed `Pi` intertwines to 1.1e-12 while differing from the log-average by 3.1e-3.
**Fix:** State the theorem as the Banach-chart version of the asymptotic-phase fibration of the symmetric invariant subspace, cite the classical sources (L06-ERA-13), and say in the companion that its content is the unique exact slow coordinate of the approach to the common reference. Keep the point that the gap `kappa_N = 2N/(N-1)` comes from the peer interaction itself (it is temperature independent at linear order because row variations vanish at equality), which bears on PIFB2 `24_section.tex:13` (an intrinsic spectral hierarchy rather than an inserted mobility).
**Falsifies:** A step of the proof that uses a KL identity beyond the linearization rates, or a point of `U` whose trajectory does not converge to the reference.

### L06-ERA-06: The hypothesis |J|<1/4 is removable down to the stability threshold |J|<1, and Pi extends to the whole basin
**Category:** improvement
**Severity:** medium
**Location:** chapters/09_public_compression.tex:9, :74 (`eq:public-phase-gap`), :94-99 (`eq:public-phase-backward`), :118, :138
**Evidence:** `Df(u,v)[h,k] = (-h + J phi Cov_{s_v}(psi,k), -k + J psi Cov_{q_u}(phi,h))`. For `phi = +-1` with mean `m` under `q_u`, `|Cov_{q_u}(phi,h)| <= ||h|| E|phi - m| = (1-m^2)||h|| <= ||h||`, so `f` is globally Lipschitz with `L = 1+|J|` in the max norm (the manuscript's `1+2|J|` is valid but loose). The proof needs `mu < 1-|J|` (so `|J| < 1`) and `b` with `1+|J| < b < 1+kappa_N-|J|`, which is possible iff `|J| < kappa_N/2 = N/(N-1)`, automatic for `|J| < 1`. The binary `C^3` claim needs `2b > 4L` with `b < 5-J`, i.e. `10 - 2J > 4 + 4J`, i.e. `J < 1`. At `|J| = 1` the reference loses stability (eigenvalue `-1+|J|` of the operator A). Separately, because `phi_t` is complete, `Pitilde(x) := phi_{-s}(Pi(Phi_s x))` for any `s` with `Phi_s x` in `U` is independent of `s` (by intertwining) and defines a `C^1` intertwining submersion on the basin of `U`; `DPitilde = Dphi_{-s} DPi DPhi_s` stays surjective since `DPhi_s` is invertible.
**Fix:** Replace `|J|<1/4` by `|J|<1` and `L` by `1+|J|`; add the basin extension as a corollary.
**Falsifies:** A value `1/4 <= |J| < 1` at which a step fails (none found).

### L06-ERA-07: The failure of child sampling holds for every Lipschitz intertwining retraction, and its size is second order in the child differences
**Category:** improvement
**Severity:** medium
**Location:** chapters/09_public_compression.tex:115-145 (`eq:public-phase-binary-parent`); companion/c09_public_compression.tex:35-40; README.md:18
**Evidence:** Uniqueness: let `Pi'` be locally Lipschitz on `U` with `Pi'` equal to the identity on the diagonal and `Pi' o Phi_t = phi_t o Pi'`. Both `Pi` and `Pi'` satisfy `|Pi(x) - zbar| <= C||delta||`, so `|phi_t(Pi x) - phi_t(Pi' x)| <= 2C||delta(t)|| <= C' e^{-bt}`, and `||Dphi_{-t}|| <= e^{Lt}` gives `|Pi x - Pi' x| <= C' e^{-(b-L)t} -> 0`. So `Pi` is the only parent of this type, and the binary counterexample shows that no exact parent that is a single-agent law, evolves by the diagonal flow, and restricts to the identity on equal children preserves the uniform-child bit mean. Size: if `R` is a permutation-invariant `C^2` response with `R(z,...,z) = r(z)` (e.g. the arithmetic pool), then `R` and `r o Pi` agree on the diagonal with equal first derivatives (both kill zero-sum directions there), so `|R(x) - r(Pi x)| <= C||delta||^2` on `U` and `<= C e^{-2bt}` along the flow, while the parent signal decays like `e^{-(1-|J|)t}`. Binary check (s04, `J = 0.2`): sympy gives `mdot - f_b(m) = 3(2-J) m a^2` with no order-4 terms, `adot = -(5-J)a + O(3)`, and the order-3 invariance equation gives `c = 3(2-J)/(2(5-J))` exactly; the numerically computed phase gives `(p-m)/(m a^2) = 0.5686, 0.5640, 0.5629, 0.5626` at `m = a = 0.08, 0.04, 0.02, 0.01` against `c = 0.5625`.
**Fix:** Add the uniqueness statement to `thm:public-phase` and the `O(||delta||^2)` defect bound to `sec:public-phase-response`; this turns the counterexample into a statement about the whole parent class.
**Falsifies:** A second Lipschitz intertwining retraction onto the diagonal flow.

### L06-ERA-08: With a fixed non-identity channel the Gibbs response identifies only the pushforwards; deterministic coarse channels allow exact compression of the response and of the incoming force
**Category:** extension
**Severity:** medium
**Location:** chapters/09_public_compression.tex:147-190 (prop `prop:public-gibbs-identifies`, remark :175)
**Evidence:** For a fixed Markov comparison kernel `K`, `Z(r) = sum_j w_j exp(-KL(r || q_j K)/T)`, and the proof of the proposition with `f_j` replaced by `d(q_j K)/d lambda` identifies `sum_j w_j delta_{q_j K}` and nothing more. s06: (a) identity channel, weights recovered from a 7-point Vandermonde fit as (0.5, 0.3, 0.2) exactly; (b) deterministic merge `{0,1,2} -> {0,1}`: two different collections with equal images give `|Delta Z| <= 4.4e-16` over 200 random receivers, and two distinct children with a common image are reproduced by one ordinary law `(0.5,0.25,0.25)` of weight 2 (`|Delta Z| = 0`); BSC(0.2) is injective on laws (det 0.6), so its children are still identified. (c) The incoming Fisher force from receiver `r` on source `j` is `d_t log q_j = beta (K(r/nu_j) - 1)`; for deterministic `K` this gives `nudot_j = beta (r - nu_j)`, closed on the image (residual 0 for two inputs with the same image); for the non-deterministic `K = [[0.9,0.1],[0.2,0.8],[0.55,0.45]]`, two inputs with equal image have image velocities differing by 1.8e-2. So the obstruction in the proposition is exactly injectivity of `q -> qK` on the exposed set, and for deterministic coarse-graining channels both the response and the incoming force descend to images; the remaining local terms then require the conditional-spectrum mechanism of ch. 7 (KL chain rule, `smooth:hidden-profile`) to close.
**Fix:** State the proposition for fixed channels with conclusion `sum_j w_j delta_{q_j K}`; add the deterministic-coarse-channel corollary and the link to ch. 7. Difficulty: the identification part is immediate; a full dynamical closure with local evaluators is a moderate construction.
**Falsifies:** A fixed non-injective channel whose block response separates two collections with identical images (excluded by the formula).

### L06-ERA-09: Persistent heterogeneity is available with heterogeneous couplings; a gate driven by public polarization would be less prescribed
**Category:** extension
**Severity:** medium
**Location:** chapters/08_smooth_erasure.tex:7-30, :126-167
**Evidence:** With identical agents and a unique attracting zero per box, heterogeneity cannot persist (L06-ERA-01). Give agent `i` the coupling `J_i = log 3 + eta_i`. Its local equilibrium solves `atanh x = J_i x`, with `dx*/dJ = x/((1-x^2)^{-1} - J) = 2.13` at `x = 1/2`; the box margins `J_i l - atanh l = 2.26e-3 + eta_i l` stay positive for `|eta_i| <= 4e-3`. Box-mates then converge to distinct equilibria separated by about `2.13|eta_i - eta_k|` (per coordinate) plus `O(eps^2)`. s07 with `eta = (0.002, -0.0015, -0.002, 0.0015)`: in-box separations converge to 1.2104e-2 and 9.078e-3 (identical to four digits at t = 100 and t = 150), boxes respected, both gates closed. The cost is heterogeneous constitutive data, which must be stated. For the gate, replace `H_q` by the public polarization `P = (1/4) sum_i psi(x_i)` and close when `P` exceeds a threshold reached only after all agents settle into wells; the added public force `alpha'(P) psi'(x_i)/4` times the private-cost derivative is `O(rho M_chi/delta)` and must be dominated by the public inward margin (about 1.7e-3), an ordering of the same type as the present one.
**Fix:** Add the heterogeneous-coupling corollary if persistent heterogeneity is to be claimed. Difficulty: low for the `J_i` variant (same proof plus an implicit-function step for the shifted equilibria); moderate for the polarization gate (needs a quantitative entry estimate into the threshold set).
**Falsifies:** Loss of box invariance or of the positive limiting separation for small `|eta|`.

### L06-ERA-10: Lemma erase:gate-entry leaves the public states free but presupposes global existence
**Category:** gap
**Severity:** low
**Location:** chapters/08_smooth_erasure.tex:68-75, :169
**Evidence:** The open class constrains only private and gate means; "reach zero in finite time and remain zero" presupposes that the public means stay in `(-1,1)`. Global continuation is asserted only for the box family (:169). It holds: as `x_i -> +-1` the receiver term `atanh x_i - atanh c` diverges with the inward sign while the remaining public terms stay bounded, and the Fisher factor makes the field `O((1-|x|) log(1/(1-|x|)))`, so the boundary is not reached in finite time.
**Fix:** One sentence in the lemma, or restrict the lemma to compact public sets.
**Falsifies:** A public trajectory reaching `+-1` in finite time.

### L06-ERA-11: Proposition erase:parent-flow proves descent of the flow, not uniqueness for the weak equation
**Category:** gap
**Severity:** low
**Location:** chapters/08_smooth_erasure.tex:208-218
**Evidence:** The proof shows that equal retained measures evolve to equal measures, using uniqueness of the finite ODE and permutation equivariance. The sentence "On the realizable finite-atom states they determine the retained evolution uniquely" reads as uniqueness for the nonlinear continuity equation `erase:empirical-flow` with field `V[.;M(t)]`. Among curves of realizable two-atom measures this follows from the ODE (atoms stay distinct); among general measure-valued curves it needs a Lipschitz stability (Dobrushin-type) estimate for `V` in `(v, M)`.
**Fix:** Restrict the claim to curves of realizable finite-atom states, or add the Lipschitz estimate.
**Falsifies:** Not applicable (precision of statement).

### L06-ERA-12: Forward invariance, uniform constants, and the binary C^3 claim in thm:public-phase are at sketch level
**Category:** gap
**Severity:** low
**Location:** chapters/09_public_compression.tex:77-83 (`eq:public-phase-decay`), :138
**Evidence:** The decay estimates are said to follow from "Taylor expansion and the variation-of-constants formula". The argument closes because in the norm `max(||zbar||, ||delta||)` both linear semigroups are contractions with constant one (`||e^{tA}|| <= e^{-(1-|J|)t}` since the off-diagonal operator has norm at most 1, and the difference semigroup is bounded by `e^{-(1+kappa_N-|J|)t}`), so a small ball is forward invariant; in the plain max norm on `E^N` the full linear semigroup has constant up to 3, which is why this should be said. The `C^3` assertion in the two-agent subfamily needs the third variational equations for `delta` and `||D^k phi_{-t}|| <= C e^{kLt}` for `k <= 4`; it is plausible and my numerics agree with the resulting expansion, but it is not written.
**Fix:** State the adapted norm and the bootstrap in two sentences; either write the `C^3` estimate or obtain the order-3 coefficient directly from `eq:public-phase-integral` (only the quadratic part of `g` and the linearized backward flow contribute at that order).
**Falsifies:** Not applicable (completeness).

### L06-ERA-13: Classical sources for the asymptotic phase and for controlled projectability are missing
**Category:** citation
**Severity:** low
**Location:** chapters/09_public_compression.tex:113, :195-206
**Evidence:** The construction is the rate-condition fibration of N. Fenichel, "Asymptotic stability with rate conditions", Indiana Univ. Math. J. 23 (1974) and part II, 26 (1977); for sinks it gives isostable/slow Koopman coordinates (A. Mauroy, I. Mezic, J. Moehlis, Physica D 261 (2013)); the periodic analog is J. Guckenheimer, "Isochrons and phaseless sets", J. Math. Biol. 1 (1975). Invariance of the diagonal is invariance of a fixed-point subspace (M. Golubitsky, I. Stewart, The Symmetry Perspective, 2002). Eq. `public-controlled-projection` is phi-relatedness of control systems (G. Pappas, G. Lafferriere, S. Sastry, "Hierarchically consistent control systems", IEEE TAC 45 (2000)). I did not re-read these for this review; the bibliographic data are from memory and must be checked. The cited Eldering-Kvalheim-Revzen paper builds on Fenichel; Horstmeyer-Atay covers autonomous lumping only.
**Fix:** Add the primary sources after checking them.
**Falsifies:** Not applicable.

### L06-ERA-14: The identifiability proof needs one generic probe line; the descriptor nonclosure claim lacks its witness
**Category:** improvement
**Severity:** low
**Location:** chapters/09_public_compression.tex:157-190
**Evidence:** The proof uses only `r = 1 + t h` for one bounded mean-zero `h` outside a finite union of hyperplanes fixed by the children. Identification therefore holds for any probe family containing such a line; a restricted family fails only on children whose centered log-density differences are annihilated by every probe direction, consistent with `eq:public-restricted-descriptor`. The Vandermonde nodes `v_j/T` approach each other as children approach (ch. 12 :442 says this), so in the phase setting the resolving power decays like the child differences. The sentence "These reciprocal moments are additional information unless the admitted family supplies a closure identity" (:190) has an explicit witness in the research record: `(1/4,1/4,1/2)` and `(c,c,1-2c)`, `c = (1+sqrt5)/8`, have equal intercept (-0.056633012265) but intercept velocities `1/9` and `(5 sqrt5 - 10)/9 = 0.131149` under `qdot = r - q` (s06(d)).
**Fix:** State the one-line version of the proposition and insert the three-atom witness.
**Falsifies:** Not applicable.

### L06-ERA-15: Symbol overloads across chapters 8 and 9
**Category:** consistency
**Severity:** low
**Location:** 08:9 (`psi(x) = d(x,0)`) versus 09:9 (`psi: M -> {-1,1}`); 08:71 (`p = 3/4`) versus 09:140 (phase mean `p`); 08:95-102 (constants `M`, `V`) versus 08:203-206 (counting measure `M`, field `V[v;M]`); `delta` as gate width (08:37), Dirac mass (08:46) and child differences (09:33); `b` as Bernoulli (08:9), decay rate (09:72) and intercept (09:181)
**Evidence:** Direct reading; within chapter 8 alone `M` and `V` each denote two objects.
**Fix:** Rename the constants (for example `m_K, M_K, V_K`, a private bound `P`, a gate width `Delta`).
**Falsifies:** Not applicable.

### L06-ERA-16: The section heading "An exact collective phase for general laws" overstates the domain
**Category:** inflation
**Severity:** low
**Location:** chapters/09_public_compression.tex:6 (`sec:public-phase`)
**Evidence:** The theorem holds on a small `L^infty` ball of centered log-densities around `(A,B)`, for references carrying balanced `+-1` features and a fixed bilinear evaluator; "general" refers to the sample spaces and the absence of a finite-parameter family. The README's "bounded-density neighborhood" is accurate.
**Fix:** "on general sample spaces" or "for general laws near a stable reference".
**Falsifies:** Not applicable.

## Checked without defect

| Label | Method | Residual or reasoning |
|---|---|---|
| `erase:local` | KL of three-bit products against `rG` summed exactly (s01a) | max 4.3e-16 at 5 random points |
| `erase:family` (invariance claim, :64) | full 8-entry table Fisher flow at a product point with `alpha' = 45.3` (s03) | non-additive part of `udot/u` 1.5e-15; induced mean velocity vs mean flow 8.9e-16 |
| `erase:gate-flow` | envelope derivation by hand; analytic gradient implementing it vs complex-step gradient of `E_red` (s02) | 4.4e-16 and 6.7e-16 |
| `erase:gate-derivative` | sympy | 0; bound `B = 0.981818` attained on the grid |
| `erase:time-choice`, constants `m, M, V, M_chi` | numerics (s01c) | `m = 1.32883`, `M = 1.65073`, `V = 2.31514`, `M_chi = 2`, `delta < 0.0478` |
| `erase:closed-clock`, lem `erase:gate-entry` | proof reconstructed (bootstrap on `K`, inward private field at `rho = 0`); simulation | closure at t = 0.017 / 0.016 (bound 0.045); gates stay closed |
| `erase:well-margins` | numerics | 2.2597e-3 and 2.4375e-3; `J/2 - atanh(1/2) = 0` |
| `erase:public-derivatives` | sympy derivatives and series | residuals 2.2e-16, 4.3e-18; both `O(eps^2)` |
| `erase:cost-comparison` | Taylor in first argument, `d_1^2 d = 1/(1-x^2)` | exact on `[-u,u]` |
| `erase:crossed-pair-gap` | sympy | 0 |
| `erase:separated-energy` | row separation identity | 0.0 numeric |
| `erase:mixed-term` | sympy mixed derivative of `-T log` of the mean of `exp(-X_j/T)` | 0 |
| `erase:mixed-witness` | sympy | 0; numerators 0.9592, 1.0392 at the extreme corners |
| thm `erase:pairs` | proof reconstructed; simulation of box invariance and preference switch | crossed rows `[1 0 3 2]` before, box rows `[2 3 0 1]` after closure, boxes kept |
| `erase:empirical-flow`, prop `erase:parent-flow` | chain rule and descent argument | correct; scope in L06-ERA-11 |
| `eq:public-phase-local` | log-density of `R` integrated by hand; `R` normalized since `E_A e^{J phi s} = cosh J` for `s = +-1` | exact |
| `eq:public-phase-chart` and the chart ODE | `udot = -(h - E_A h)` from `d_t log q = -(h - E_q h)` | exact |
| `eq:public-phase-diagonal` | finite-alphabet field (s05) | 2.5e-16 |
| `eq:public-phase-linearization` | finite differences, `N = 3`, unequal temperatures | 1.95e-10 |
| `eq:public-phase-residual` | averaged derivative on zero-sum directions; scaling of `g` | 5.6e-11; `|g|/s^2` constant 1.61 |
| `eq:public-phase-backward` | Gronwall with `||Df|| <= 1+|J|` | valid (bound loose, L06-ERA-06) |
| `eq:public-phase-integral`, `eq:public-phase-limit`, `eq:public-phase-intertwining`, `eq:public-phase-decay`, `eq:public-phase-gap`, thm `thm:public-phase` | proof reconstructed; numeric phase on a 4-point alphabet | intertwining 1.1e-12; horizon 7 vs 9 change 9.1e-12 |
| `eq:public-phase-binary`, `eq:public-phase-binary-fine` | sympy from the potential; invariance of `q_i = s_i` | 8.9e-16; `dE/dx - dE/dy = 0` on `q = s` |
| `eq:public-phase-binary-expansion` | sympy series | `3(2-J) m a^2`, no order-4 term; `-(5-J)a` |
| `eq:public-phase-binary-parent` | sympy invariance equation; numeric phase | coefficient exact; ratio 0.5626 at `m = a = 0.01` |
| `eq:public-probe-normalizer`, `eq:public-probe-exponential`, prop `prop:public-gibbs-identifies`, `eq:public-probe-identification` | proof reconstructed; Vandermonde recovery (s06a) | weights recovered exactly |
| `eq:public-restricted-descriptor`, `eq:public-restricted-force` | by hand (`alpha_j` is the Gibbs weight; `int h_a = 0` is needed and holds) | exact |
| companion `comp:eq-phase-common`, `comp:eq-phase-rates`, `comp:eq-phase-map`, `comp:eq-phase-response`, `comp:eq-phase-gibbs` | compared with the main text | consistent |

## Extensions and improvements

A uniqueness-and-defect theorem for the phase parent is the cheapest strengthening. With `b > L`, every locally Lipschitz map onto the single-agent laws that restricts to the identity on equal children and intertwines the fine flow with the diagonal flow equals `Pi` (L06-ERA-07). Any permutation-invariant `C^2` response that reduces to `r(z)` on the diagonal then differs from `r o Pi` by `O(||delta||^2)`, decaying like `e^{-2bt}` along the flow. This converts the binary counterexample into a classification statement (the phase is the unique exact single-agent parent, and its response defect is exactly second order), and it removes any suspicion that the failure is an artifact of a choice of `Pi`. Together with the sharpened Lipschitz constant `L = 1+|J|`, the hypothesis `|J| < 1/4` relaxes to the full stability range `|J| < 1`, and the phase extends to the basin by `phi_{-s} o Pi o Phi_s` (L06-ERA-06).

A type-homogeneous parent without a gate is within reach by combining the two chapters. Every subgroup of `S_N` that preserves the priors (for instance `S_{n_1} x S_{n_2}` with uniform nonself priors) has an invariant fixed-point subspace of block-equal configurations. Restricted to it, the original reduced energy becomes `n_I` times a block-agent KL system whose rows include a zero-cost self source with prior weight `(n_I - 1)/(N - 1)`, and the metric becomes `n_I g`; by symmetric criticality the restricted flow is the original flow. This is again a law-pair KL system of the type used in ch. 12 (self-inclusive priors). Near a stable equilibrium where the blocks occupy different wells of a bistable evaluator, the asymptotic phase onto this subspace exists when intra-block (normal) decay rates exceed the tangential ones. s08 checks this for binary agents, `N = 4`, `J = log 3`, identity channels, uniform nonself priors, `T = 0.05`: equilibrium `x = y = (0.49964, -0.49964, 0.49964, -0.49964)`, tangential rates (0.173, 0.176, 1.822, 1.824), normal rates (4.174, 4.174, 5.823, 5.823), ratio 2.29. At `T = 0.1` and `T = 0.2` the two-well configuration does not persist in this instance (the flow from it reaches the symmetric origin inside the sign-symmetric subspace), so deeper wells (larger `J`) or small `T` are needed. The result would be distinct parents of the children's type with exact inherited KL dynamics and a nested symmetry allowing a second level; membership and the symmetric prior remain supplied, fine differences still decay, and the child-sampling defect is still second order. Difficulty: moderate (a rate-condition fibration near a non-symmetric equilibrium plus a parameter-window computation).

For the identifiability side, the fixed-channel version (L06-ERA-08) says exactly which compressions an interface allows: the block response of a fixed kernel `K` determines `sum_j w_j delta_{q_j K}`, so exact compression for Gibbs probes is possible precisely on fibers of `q -> qK`. For deterministic coarse-graining channels the incoming Fisher force also descends (`nudot = beta(r - nu)`), and the within-fiber conditional laws are then the hidden conditionals of ch. 7, whose spectra close exactly. A theorem combining a deterministic coarse comparison channel, receiver conditional reconstruction, and the spectrum completion would give an exact parent that compresses distinct children for the full Gibbs interface of that channel, which is the interface-relative sufficiency the outlook asks for. Difficulty: moderate; the response and incoming-force parts are checked above, and the local-term closure is ch. 7's mechanism.

For the erasure chapter, two inexpensive repairs bring the statements in line with what is provable: retitle `erase:pairs` and state convergence of each pair to a common point (L06-ERA-01), and put the temperature into the parameter ordering with explicit row bounds so that "preferred" has quantitative content (L06-ERA-03). If persistent heterogeneous pairs are wanted, heterogeneous couplings `J_i` give them with the same proof (L06-ERA-09; s07 shows positive limiting separations 1.21e-2 and 9.08e-3). Replacing the dedicated gate coordinate by a public polarization statistic would make the erasure contingent on the public relaxation instead of an internal clock; the needed estimate is that `alpha'` times the private-cost derivative stays below the public inward margin.
