# XC3-SCOPE: Objective discipline, general-law versus Gaussian scope, and regularity across the manuscript

I read in full, statements and proofs, `chapters/03_dynamics.tex` 1-438, `04_prediction.tex` 1-236, `05_inheritance.tex` 1-1399, `06_fusion.tex` 1-707, `06_outlook.tex` 1-16, `07_smooth_closure.tex` 1-321, `08_smooth_erasure.tex` 1-263, `09_public_compression.tex` 1-251, `10_boundary_closure.tex` 1-353, `11_conditional_formation.tex` 1-568, `12_open_formation.tex` 1-580, `13_per_agent_inheritance.tex` 1-398, `14_collective_local_parent.tex` 1-378, `15_contextual_coupling.tex` 1-964, `16_link_base_transport.tex` 1-364, `17_communication_links.tex` 1-500, `18_link_dynamics.tex` 1-641, `19_curvature_hierarchy.tex` 1-268, `appendices/conditional_memory.tex` 1-229, `appendices/interacting_memory.tex` 1-664, `frontmatter.tex` 1-24, `AUTHORING.md`, and items H1, H8 and the R3 regularity cluster of the 2026-10-22 REPORT. I skimmed the companion only through targeted searches (c17, c18) and consulted other lens reports by search only after finishing my own reading, to cross-reference overlaps. The scope contains 186 results (40 theorems, 115 propositions, 21 corollaries, 10 lemmas); every one is classified in the table below. Verdict: the three-objective rule and the general-law rule are now kept in the statements of chapters 03 to 19 and both memory appendices; every chapter names its objective, Gaussian-only results state their family, and the main text flags every Gaussian-family feature it uses. The two defects that matter are both at the boundary of the declared objective or of the declared regularity: the chapter 11 null-flow control is computed under the self-weight-one convention that chapter 13 calls a different control, and under the declared convention `form:kappa-independence` (iii) is false on an open set (checked numerically, independently of D15-WISH-01); and per-agent flows on circle spaces (chapters 13 and 14) are run without a definition, chart or well-posedness statement, so their "invariant family" claims rest on tangency alone. The other six findings are low.

## Findings

### XC3-SCOPE-01: The chapter 11 null-flow control uses the self-weight-one convention that chapter 13 calls a different control; under the declared convention `form:kappa-independence` (iii) is false
**Category:** error
**Severity:** high
**Location:** chapters/11_conditional_formation.tex:316 and :324 (`form:kappa-independence` (iii) and its proof), :6, :393; chapters/05_inheritance.tex:1380 (`inh:null-flow`), :1394; chapters/13_per_agent_inheritance.tex:18-23 (`pa:null-row`); chapters/14_collective_local_parent.tex:80
**Evidence:** The proof of (iii) writes the null belief potential as `sum_i [KL(q_i || W_2(alpha_0, I_2)) + KL(q_i || W_2(alpha_0, Lambda_i^0))]`, so the nonidentity self cost enters with weight one. The statement itself says the null is taken "with every peer cost deleted, each self channel evaluated with the peer laws held at their initial values, and all other data fixed", and the chapter's data are "Every source, including self, is allowed with prior pi_ij = 1/N" with exact Gibbs rows at a positive temperature. `inh:null-flow` deletes the peer costs "with all other data unchanged". Chapter 13 fixes the reading: "The null of inh:null-flow deletes the nonself costs, not their labels or priors ... Replacing these rows by self probability one would specify a different control" (13:18-23), and chapter 14 repeats it: "Its surviving self weight is not one" (14:80). Under that convention the Gibbs-reduced null belief term of agent i is `-T log((1/N) e^{-D_ii/T} + (N-1)/N)`, whose envelope weight is `b_i = (1/N) e^{-D_ii/T} / ((1/N) e^{-D_ii/T} + (N-1)/N) < 1`. Repeating the coefficient matching of `form:full-flow` with this weight gives `alpha_i' = -(1 + b_i)(alpha_i - alpha_0)` and `P_i' = I_2 + b_i C_i^0 - (1 + b_i) P_i`, with `C_i^0` as in the chapter; the clocks are now agent dependent.

`scripts/XC3-SCOPE/ch11_null_convention.py` integrates both conventions (Wishart KL in closed form, `kappa = alpha_0 = 1`, `delta = 1/10`, `P(0) = I_2`, threshold rule `form:threshold-rule`). On the chapter's own example `u = (1, 6/5, 9/5, 21/10)` at `T = 1` the initial self weights are (0.0914, 0.0474, 0.0038, 0.0009) and the null events occur at 0.650, 1.111, 2.058, against 0.347, 0.549, 1.070 for the full flow, so `H^0 != H` at every point. On the admitted configuration `u = (1, 1.2, 1.45, 1.79)` (gaps 0.20, 0.25, 0.34 satisfy `form:gaps`) the full flow joins {1,2}, then {3,4}, then the union, while the declared null joins {1,2}, then {1,2,3}, then the union. `ch11_null_perturb.py`: all 11 random admitted perturbations of size 0.004 show the same reversal, with crossing-time margins 0.017 to 0.053, so `P^0 != P` on a nonempty open subset. The sentence "Hence P^0(x) = P(x), and indeed H^0(x) = H(x) including event times, at every point x of the admitted class. The construction fails clause (F5)" is therefore false under the convention the manuscript declares, and (F5) is formally met through the reweighting of the nonidentity self row, which carries no information about the interaction. At `T = 0.1` the weights are near zero and the sequences agree, so the verdict also depends on the unstated temperature. D15-WISH-01 reports the same defect from a different preparation; this is an independent replication.
**Fix:** Put the row convention inside `inh:null-flow`. If `pa:null-row` stands, replace (iii) by the computed null law, state `H^0 != H` everywhere and `P^0 != P` on an open set, and report instead the coupling-deformation control that (i) and (ii) prove (identical membership history for every `kappa >= 0`). If self weight one stands, say so in `inh:null-flow` and correct 13:18-23 and 14:80.
**Falsifies:** A reading of `inh:null-flow` compatible with "all other data unchanged", priors 1/N and exact Gibbs rows under which the null self weight is one; or an error in the null precision law above that restores a common shape clock.

### XC3-SCOPE-02: Per-agent flows on circle spaces are used without a definition, chart or well-posedness, so "invariant family" rests on tangency alone
**Category:** gap
**Severity:** medium
**Location:** chapters/03_dynamics.tex:354-357 (`def:dyn-per-agent-flow`), :223-232 (`def:dyn-chart-flow`, `prop:dyn-chart-flow`); chapters/13_per_agent_inheritance.tex:9, :176-237 (`pa:angular-flow`), :241-295 (`pa:angular-grouping`), :301-312 (`pa:phase-parent`), :336-390; chapters/14_collective_local_parent.tex:43-69 (`lp:native-field`) and every later result of that chapter; AUTHORING.md:51
**Evidence:** `def:dyn-per-agent-flow` defines the per-agent dynamics only "On finite positive tables satisfying the hypotheses of thm:dyn-finite-flow", with the rows as evolving tables. `prop:dyn-chart-flow` is stated for the reciprocal potential ("Suppose E has a first variation ..."). Chapters 13 and 14 run a different object, the Gibbs-reduced own-block flow ("this is the Gibbs-reduced convention, not finite-mobility row evolution", 13:9), on the space `{-1,1} x S^1 x S^1`, which is neither finite nor covered by a chart statement for per-agent dynamics. The proofs establish tangency: "The full own-law Fisher field preserves pa:angular-family" (13:215); "This is therefore the exact Fisher velocity on an invariant family, rather than a Euclidean update or a projected approximation" (13:236); "This is an invariant family of the own-law Fisher field" (14:68). I re-derived the coefficient matching of `pa:angular-flow` (the l, A, C, b and theta equations) and it is correct. Tangency gives a solution of the full density equation that stays in the family; invariance of the family, and statements such as "intertwines the original-time full and null flows" (`pa:phase-parent`) read as statements about the full model, also need uniqueness of the full flow in a declared chart. The regularity rule requires exactly this: "Any use on densities, circles, or general standard Borel spaces requires a declared chart ..., stated Lipschitz hypotheses on the first variation, well-posedness, and uniqueness of every coarse flow whose autonomy is claimed." The gap is closable at low cost. On these compact spaces every density of the two families has bounded logarithm. The own-block first variation is `sum_j beta_ij [log q_i - log nu_ij] - beta_ii log G_i - log l` with bounded log peer outputs (`mu_j x Haar x Haar` in chapter 13; the `L_j` and `B_j` factors of `lp:peer` in chapter 14), the bounded log evaluator `J z cos(phi)`, bounded log record likelihoods, and Gibbs rows that are smooth functions of bounded costs. The field is therefore Lipschitz on bounded sets of `L^infty_0`, and the Picard-Lindelof argument of `prop:dyn-chart-flow` applies without any gradient structure. D17-PA-03 reports the chapter 13 part; D05-OBJ-06 notes the two row conventions. Chapter 14 and the dependence of every later chapter 14 statement on this step are not covered there.
**Fix:** Add to chapter 3 a definition of the Gibbs-reduced per-agent flow and a per-agent version of `prop:dyn-chart-flow` (bounded log-density chart, Lipschitz hypothesis on the own-block first variation, local well-posedness), verify its hypotheses for the chapter 13 and 14 data, and conclude invariance of the exponential families from tangency plus uniqueness. Otherwise restrict the statements to the parameter ODE on the family and delete the invariance sentences.
**Falsifies:** A chart-level definition and well-posedness statement for per-agent dynamics elsewhere in the main text that chapters 13 and 14 cite; or an explicit statement that their claims concern only the finite-dimensional parameter flow.

### XC3-SCOPE-03: "The restricted flow is the Fisher-orthogonal projection of the full velocity" needs a finite Fisher norm that Gaussian sources with wide receivers do not have
**Category:** gap
**Severity:** low
**Location:** chapters/03_dynamics.tex:213 (sentence after `def:dyn-restricted-flow`), :215-219 (`prop:dyn-gaussian-noninvariance`)
**Evidence:** For a Gaussian source `q_i = N(m_i, s_i^2)` with one identity-channel receiver `q_k = N(m_k, s_k^2)`, the source part of the Fisher velocity is `-beta_ki (q_k/q_i - 1)`, and `E_{q_i}[(q_k/q_i)^2]` is finite exactly when `s_k^2 < 2 s_i^2`. `scripts/XC3-SCOPE/gaussian_noninvariance.py` evaluates the closed form: 2.29, 7.09 and 22.37 at `s_k^2/s_i^2` = 1.9, 1.99, 1.999 and infinite from 2 on (in d dimensions the condition is `2 Sigma_k^{-1} - Sigma_i^{-1}` positive definite). For wider receivers the full velocity is not a Fisher tangent vector at `q_i`, so the projection exists only weakly, through the pairings `E_{q_i}[v t]` with the family tangents `t`, which stay finite (`E_{q_i}[(q_k/q_i) x^n] = E_{q_k}[x^n]`); that weak projection is what `eq:dyn-gaussian-flow` computes. Where the norm is finite, the fraction of the Fisher norm of the full velocity outside span{1, x, x^2} is 0.02 to 0.33 in my four test cases, which supports `prop:dyn-gaussian-noninvariance` (it vanishes when the source term is removed). D04-DYN-05 makes the same point from the well-posedness side.
**Fix:** "The restricted flow is the natural-gradient flow of E on the family; when the full velocity has finite Fisher norm it is the Fisher-orthogonal projection of that velocity", with the integrability condition stated for the Gaussian case.
**Falsifies:** A finite Fisher norm of the source-role velocity for `s_k^2 >= 2 s_i^2`, or a different full-flow metric in which the sentence holds as written.

### XC3-SCOPE-04: Chapter 15 calls its zero-coupling flow the reciprocal Fisher flow, and the abstract omits both the objective and the finite-dimensional family
**Category:** consistency
**Severity:** low
**Location:** chapters/15_contextual_coupling.tex:230 (`ctx:f:thm:ode`), :841, :22, :407; frontmatter.tex:15; AUTHORING.md:87
**Evidence:** `ctx:f:thm:ode`: "At lambda = 0 the equations recover exactly the chosen context-local reciprocal Fisher flows"; 15:841 "recovers the chosen pointwise reciprocal Fisher flow". The flow is the natural-gradient flow on a declared finite-dimensional regular family (15:22, :407), that is the restricted-family flow of `def:dyn-restricted-flow`, which chapter 3 separates from the reciprocal Fisher flow ("It is the full flow only on families that the full flow leaves invariant", 3:213). Chapter 15 never cites `def:dyn-restricted-flow`; its restriction sentences (15:22, :407: "Unrestricted joint Fisher dynamics need not preserve the restriction") concern the product restriction, not invariance of the finite-dimensional family. The abstract paragraph on this chapter, "A separate contextual extension couples neighboring laws through an auxiliary alignment energy. On finite graphs, a regular minimizing branch gives an explicit Fisher force and local uniqueness", names neither the reciprocal potential nor the finite-dimensional families that AUTHORING:87 makes part of the scope.
**Fix:** Write "the restricted-family flows (`def:dyn-restricted-flow`) of the context-local reciprocal potentials" at 15:230 and 15:841, and add "for the reciprocal potential on declared finite-dimensional regular families" to frontmatter:15.
**Falsifies:** A proof that the admitted families are invariant under the full reciprocal flow at zero coupling, in which case the two flows coincide.

### XC3-SCOPE-05: The companion states exact factor means as a fact of the global picture, without the Gaussian-feature caveat of the main text
**Category:** inflation
**Severity:** low
**Location:** companion/c18_link_dynamics.tex:83; contrast chapters/18_link_dynamics.tex:125, chapters/17_communication_links.tex:483; AUTHORING.md:29
**Evidence:** c18:83: "Thus S_i = J_ii^{-1} at its covariance optimum, and the factor means equal the exact posterior mean." The main text attaches the caveat each time: "Exact factor means are a Gaussian property" (18:125) and "Exact means are a feature of Gaussian targets" (17:483). AUTHORING:29 lists "mean-field means that equal exact posterior means" among the Gaussian features that may not be presented as properties of the theory. For a non-Gaussian target the property fails at once: `scripts/XC3-SCOPE/misc_checks.py` (a) takes `p(x,y) proportional to exp(0.3x - 0.2y + 0.9xy)` on `{-1,1}^2`; the exact means are (0.156, 0.012) and the mean-field fixed point is (0.444, 0.197). The companion is outside my result scope; I report it because it restates `it2:global-flow`, which is in scope.
**Fix:** After c18:83 add "This is a feature of the Gaussian target; for non-Gaussian posteriors the factor means generally differ from the exact means."
**Falsifies:** A sentence in c18 near line 83 that already restricts the statement to Gaussian targets.

### XC3-SCOPE-06: The Wishart density flow is posed in an undefined solution class instead of a declared chart
**Category:** gap
**Severity:** low
**Location:** chapters/11_conditional_formation.tex:98-177 (`form:full-flow`, last paragraph of the proof), :197 (`form:mean-field`), :324 (null flow in the proof of `form:kappa-independence`)
**Evidence:** Wishart log densities contain `log det X` and `tr(P X)`, which are unbounded on the cone, so `prop:dyn-chart-flow` does not cover this family. The proof shows tangency of the full density field and then asserts that "any positive dominated regular solution with these initial laws has only the same log-determinant and linear matrix statistics; its parameters satisfy the same ODE". The class "positive dominated regular solution" (norm, time regularity, finiteness of the moment `M_q` along the solution) is never defined. The integrating-factor argument itself is sound once the class is fixed: if `t -> log q_t` satisfies `d/dt log q = -2 log q + 2 c_0 log det X - tr((I_2 + C_q(t)) X) + c(t)` pointwise with a continuous coefficient path, then `log q_t` stays in span{1, log det X, X}, and uniqueness reduces to the parameter ODE, which is global by the eigenvalue bounds in the proof. `form:mean-field` states its regularity as an assumption, which is acceptable.
**Fix:** Define the solution class in `form:full-flow` (for example, `C^1` curves of densities whose log differences from the initial law lie in a weighted space dominating `log det X` and `|X|`, with the density equation holding pointwise), and say that bounded log-density charts do not cover this family.
**Falsifies:** A chart for unbounded-log exponential families elsewhere in the main text that chapter 11 cites.

### XC3-SCOPE-07: `prop:dyn-global-local` defines the mean field only almost everywhere under the recognition factor but uses it under the reference
**Category:** gap
**Severity:** low
**Location:** chapters/03_dynamics.tex:249-266 (`prop:dyn-global-local`)
**Evidence:** The hypothesis `int |log p| dQ < infty` gives, by Fubini, finiteness of `e_bar_i(y_i) = int log p(y_i, y_{-i}) Q_{-i}(dy_{-i})` only for `Q_i`-almost every `y_i`. The definitions `Z_i = int e^{e_bar_i} d lambda_i` and `Q_i^* = e^{e_bar_i} lambda_i / Z_i` need `e_bar_i` on a `lambda_i`-full set; outside the support of `Q_i` the integral can be undefined when both parts of `log p` diverge. The claim that the local free energy "has the unique minimizer Q_i^*" also ranges over competitors for which the integrability hypothesis can fail. The identity itself holds where its terms are defined (`scripts/XC3-SCOPE/ch03_identities.py` (3): residual 0.0 on a 3x4x2 table).
**Fix:** Assume `int |log p(y_i, .)| dQ_{-i} < infty` for `lambda_i`-almost every `y_i` (or define `e_bar_i` with values in `[-infty, infty)` under integrability of the positive part), and state uniqueness over competitors with `int |log p| d(Q_i' x Q_{-i}) < infty`, or prove the lower bound for all competitors by the Gibbs variational inequality.
**Falsifies:** A standing hypothesis that recognition factors are equivalent to their references, which would make the two null sets coincide.

### XC3-SCOPE-08: Chapter 3 calls the per-agent total "a potential of the form eq:dyn-potential"
**Category:** consistency
**Severity:** low
**Location:** chapters/03_dynamics.tex:357
**Evidence:** "The Fisher gradient flow of F_tot keeps those terms and is the reciprocal flow of a potential of the form eq:dyn-potential, with the agent's own model entering as the self source instead of as the additive term E_0." `F_tot = sum_i F_i` contains the persistent record terms of `eq:dyn-per-agent-free-energy` (so it has the form of `eq:dyn-evidence-potential`), fixes both temperatures at one, and has the belief self cost `int KL(q_i || G_i(m, .)) s_i(dm)`, which depends on `s_i` and is not a divergence `KL(q_i || nu^q_ij)` from a compared belief law. This is the one structural identification between objectives in chapter 3 that is not true as written; the sentences after it (no transfer of reciprocal results to per-agent dynamics) are correct. D05-OBJ-08 (ii) makes the same observation.
**Fix:** "is the Fisher gradient flow of a reciprocal potential of the evidence-augmented form eq:dyn-evidence-potential at unit temperatures, with a cross-sector self cost in place of E_0".
**Falsifies:** A definition of "of the form eq:dyn-potential" that admits cross-sector self costs and record terms.

Observed in scope and already reported by other lenses, not renumbered here: the erasure status "(F5) is therefore met only through a shift of an event time" (08:263), which contradicts the definition of (F5) (D13-ERAPUB-02); the stale range "|J| < 1/4 used in ch:public-compression" in `prop:boundary-observable` (10:321; D13-ERAPUB-07); `prop:dyn-no-global-vfe` (b), proved only on two-point alphabets but stated without that restriction (D04-DYN-02, D05-OBJ-10); and "interaction-sensitive selection ... requires receiver-dependent targets" (05:1372, 11:208, 06_outlook:16), a necessity claim drawn from one architecture class (D15-WISH-02).

## Objective, law family and regularity of every result (focus question 1)

The table covers every theorem, proposition, corollary and lemma of chapters 03 to 19 and of the two memory appendices (186 rows, generated by `scripts/XC3-SCOPE/scope_table.py` from the source so that labels and line numbers are exact; the classification is my reading of each statement and proof). Objective codes: GFE global free energy of one normalized joint; PFE per-agent free energy or per-agent dynamics; RP reciprocal law potential (RP* the nonsmooth square-root modification of chapter 6); EXACT exact Bayesian conditioning or the free energy of one fixed declared model; NONE an identity, identifiability, geometric or symmetry statement with no objective; ANY a statement about every update, vector field or gradient flow of the stated kind. Family codes: GEN general laws on standard Borel spaces (with the stated finiteness or density conditions); FIN finite positive tables; BIN binary or Bernoulli laws; TYPE finite type lattice; PB the invariant product-Bernoulli family of chapter 8; LN the restricted lognormal family of chapter 6 (a constraint, not an invariant family); WIS common-precision Wishart laws (invariant because the channels are constant in their input); BC positive bit-circle densities in the C^r chart; EXPC the finite-dimensional exponential families on {-1,1} x S^1 x S^1 of chapters 13 and 14; RFD a declared regular finite-dimensional family; GAUSS Gaussian laws; LGT the linear-Gaussian tangent model; FDM finite-dimensional manifolds; LIN linear algebra on codes and dictionaries. In the regularity column "n/a" means that no flow on a non-finite space is used; a parenthesized number points to the finding above.

| Label | File:line | Objective | Law family | Regularity |
|---|---|---|---|---|
| `prop:dyn-gibbs` | 03_dynamics:61 | RP (rows at fixed laws) | GEN (finite source sets) | n/a |
| `prop:dyn-compensation` | 03_dynamics:136 | RP | GEN | n/a |
| `thm:dyn-finite-flow` | 03_dynamics:155 | RP | FIN | finite-dim ODE, local; C2 law-dependent channels |
| `prop:dyn-global-flow` | 03_dynamics:180 | RP | FIN, fixed channels | global, interior floor, Lojasiewicz convergence |
| `prop:dyn-row-relaxation` | 03_dynamics:188 | RP | FIN rows, frozen costs | n/a |
| `prop:dyn-morse` | 03_dynamics:196 | RP vs reduced RP | FIN | Hessian inertia |
| `prop:dyn-gaussian-noninvariance` | 03_dynamics:215 | RP | GAUSS (negative result) | pointwise velocity; Fisher norm may be infinite (03) |
| `prop:dyn-chart-flow` | 03_dynamics:228 | RP | GEN, bounded log-density chart | declared chart, Lipschitz hyp., local unique, blow-up alternative |
| `prop:dyn-global-local` | 03_dynamics:249 | GFE | GEN (densities) | n/a; domain of e_bar (07) |
| `prop:dyn-jensen` | 03_dynamics:272 | NONE (identity) | GEN | n/a |
| `prop:dyn-label-model` | 03_dynamics:288 | GFE of label model vs RP | FIN | n/a |
| `prop:dyn-per-agent` | 03_dynamics:336 | PFE | GEN with finiteness | n/a |
| `prop:dyn-nonreciprocal` | 03_dynamics:359 | PFE dynamics | BIN, frozen rows | finite-dim linear |
| `prop:dyn-no-global-vfe` | 03_dynamics:375 | RP vs GFE | (a) FIN; (b) fixed channels, proof on two-point alphabets | n/a |
| `thm:dyn-registered` | 03_dynamics:391 | GFE of registered model vs RP | TYPE lattice | n/a |
| `cor:dyn-frozen-source` | 03_dynamics:407 | GFE at frozen sources vs RP rows | GEN | n/a |
| `thm:predictive-chain` | 04_prediction:23 | NONE | GEN | n/a |
| `prop:conditional-lift` | 04_prediction:105 | NONE (Jeffrey) | GEN | n/a |
| `cor:pythagorean` | 04_prediction:124 | NONE | GEN | n/a |
| `prop:common-family-recovery` | 04_prediction:139 | NONE (sufficiency) | GEN | n/a |
| `cor:local-family-recovery` | 04_prediction:181 | NONE | GEN | n/a |
| `thm:predictive-distortion` | 04_prediction:212 | NONE | GEN | n/a |
| `inh:deterministic-descent` | 05_inheritance:66 | ANY update | GEN measurable | n/a |
| `inh:projectable` | 05_inheritance:130 | ANY field | FDM | C^k fields |
| `inh:gradient-inheritance` | 05_inheritance:168 | ANY gradient flow | FDM Riemannian | n/a |
| `inh:equivariant-descent` | 05_inheritance:257 | ANY / gradient | FDM (inf-dim caveat at :290) | n/a |
| `inh:coordinate-projection` | 05_inheritance:294 | ANY gradient flow | FDM product | n/a |
| `inh:no-underdamped` | 05_inheritance:318 | ANY gradient flow | FDM | n/a |
| `inh:role-loss` | 05_inheritance:342 | NONE (declared linear ODE) | BIN product | global linear |
| `inh:finite-lumpability` | 05_inheritance:376 | NONE (Markov chain) | FIN | n/a |
| `inh:intertwining` | 05_inheritance:419 | NONE | FIN | n/a |
| `inh:canonical-parent` | 05_inheritance:471 | ANY update | GEN; analytic case | n/a |
| `inh:canonical-lumping` | 05_inheritance:502 | NONE | FIN | n/a |
| `inh:separator-gluing` | 05_inheritance:568 | NONE (declared joint) | GEN | n/a |
| `inh:composition` | 05_inheritance:651 | NONE | GEN | n/a |
| `inh:smooth-mixture-failure` | 05_inheritance:905 | RP specialization | BIN | smooth on (-1,1)^4 |
| `inh:smooth-boundary-failure` | 05_inheritance:962 | RP specialization | BIN | n/a |
| `inh:smooth-no-finite-entry` | 05_inheritance:983 | RP specialization | BIN | backward uniqueness |
| `inh:causal-memory` | 05_inheritance:1036 | NONE (record process) | GEN | n/a |
| `inh:finest-independent` | 05_inheritance:1135 | NONE (declared joint) | GEN | n/a |
| `inh:conditional-partition` | 05_inheritance:1173 | NONE | GEN | n/a |
| `inh:independence-discontinuity` | 05_inheritance:1199 | NONE | FIN Ising law | n/a |
| `inh:selector-symmetry` | 05_inheritance:1233 | NONE | GEN | n/a |
| `inh:scalar-additive` | 05_inheritance:1287 | ANY scalar flow | scalar parameters | n/a |
| `inh:common-clock` | 05_inheritance:1329 | RP (+ null) | FIN | finite-dim |
| `fus:local-inheritance` | 06_fusion:57 | RP* (nonsmooth) | LN restricted | n/a |
| `fus:uniqueness` | 06_fusion:229 | RP* | LN restricted | maximal monotone (Brezis), global unique |
| `fus:sync` | 06_fusion:269 | RP* | LN restricted | global |
| `fus:isolated-pair` | 06_fusion:323 | RP* | LN restricted | a.e. inequalities |
| `fus:sticking` | 06_fusion:349 | RP* | LN restricted, common-offset class | explicit global |
| `fus:two-stages` | 06_fusion:411 | RP* | LN restricted | explicit |
| `fus:open-two-stage` | 06_fusion:460 | RP* | LN restricted | contraction |
| `fus:dyadic` | 06_fusion:488 | RP* | LN restricted | explicit |
| `smooth:mean-force-criterion` | 07_smooth_closure:18 | ANY Fisher flow | GEN (disclaimer :16) | uniqueness assumed in statement |
| `smooth:submersion` | 07_smooth_closure:44 | ANY | regular positive class | n/a |
| `smooth:peer-cancellation-rule` | 07_smooth_closure:75 | RP | GEN | n/a |
| `smooth:split-dynamics` | 07_smooth_closure:111 | RP | GEN under smooth:hypotheses | formal; rigorous in smooth:wellposed |
| `smooth:spectrum-flow` | 07_smooth_closure:160 | RP | GEN | weak continuity equation, uniqueness (AGS 8.1) |
| `smooth:wellposed` | 07_smooth_closure:197 | RP | GEN | declared L^inf_0 chart, (H1)-(H3), fine and retained unique; T* infinite not claimed |
| `smooth:equal-spectra` | 07_smooth_closure:219 | RP | GEN | uses smooth:wellposed |
| `smooth:cgf-completion` | 07_smooth_closure:241 | RP | GEN | uses smooth:wellposed |
| `smooth:feedback` | 07_smooth_closure:278 | RP + assimilation events | GEN | regular-domain assumption |
| `erase:gate-entry` | 08_smooth_erasure:68 | RP Gibbs-reduced | PB invariant family | finite-dim, global on class |
| `erase:pairs` | 08_smooth_erasure:126 | RP Gibbs-reduced | PB | finite-dim |
| `erase:pair-convergence` | 08_smooth_erasure:181 | RP Gibbs-reduced | PB | LaSalle, exponential |
| `erase:parent-flow` | 08_smooth_erasure:240 | RP Gibbs-reduced | PB | uniqueness only among realizable curves (stated) |
| `thm:public-phase` | 09_public_compression:54 | RP Gibbs-reduced, identity channels | GEN, L^inf_0 chart | declared chart, local Lipschitz; coarse diagonal flow globally Lipschitz |
| `cor:public-phase-unique` | 09_public_compression:129 | RP | GEN | as above |
| `prop:public-gibbs-identifies` | 09_public_compression:182 | NONE (Gibbs response) | GEN bounded densities | n/a |
| `cor:public-gibbs-channel` | 09_public_compression:204 | NONE | GEN | n/a |
| `lem:boundary-fine-wellposed` | 10_boundary_closure:50 | RP identity channels | GEN, L^inf chart | declared chart, local unique |
| `thm:boundary-tree-flow` | 10_boundary_closure:73 | RP | GEN on supplied tree | unique; global via lem:boundary-global |
| `lem:boundary-global` | 10_boundary_closure:130 | RP | GEN | global bounded-log existence |
| `lem:boundary-equivariance` | 10_boundary_closure:168 | RP | GEN | uses fine uniqueness |
| `prop:boundary-fixed-family` | 10_boundary_closure:180 | NONE (symmetry) | GEN | n/a |
| `prop:boundary-joint-spectrum` | 10_boundary_closure:195 | RP | GEN | pointwise ODE + uniqueness |
| `thm:boundary-tree-inheritance` | 10_boundary_closure:222 | RP + assimilation events | GEN | quotient flow by equivariance |
| `prop:boundary-contrast-decay` | 10_boundary_closure:277 | RP | GEN | local linearization |
| `prop:boundary-observable` | 10_boundary_closure:320 | RP | BIN | analytic, inverse function theorem |
| `form:addition` | 11_conditional_formation:27 | NONE | WIS | n/a |
| `form:full-flow` | 11_conditional_formation:98 | RP exact Gibbs | WIS (invariant: constant-in-input channels) | parameter ODE global; density class informal (06) |
| `form:mean-field` | 11_conditional_formation:196 | RP Gibbs-reduced | regular dominated family (assumed) | assumed |
| `form:scalar-closure` | 11_conditional_formation:212 | ANY scalar flow | scalar | n/a |
| `form:parent` | 11_conditional_formation:223 | RP | WIS | finite-dim |
| `form:kappa-independence` | 11_conditional_formation:311 | RP and null flow | WIS | finite-dim; null convention (01) |
| `form:selection` | 11_conditional_formation:329 | RP | WIS | explicit |
| `form:responses` | 11_conditional_formation:410 | RP (no feedback) | WIS | n/a |
| `form:geometry` | 11_conditional_formation:470 | RP (diagnostic geometry) | WIS | n/a |
| `form:geometry-decay` | 11_conditional_formation:547 | RP | WIS | finite-dim |
| `open:force` | 12_open_formation:52 | RP exact Gibbs | BC, C^r chart | local Lipschitz, unique local |
| `open:interior` | 12_open_formation:115 | RP | BIN marginals | global |
| `open:persistence` | 12_open_formation:129 | ANY ODE | FDM | Gronwall |
| `open:routing` | 12_open_formation:148 | RP | BIN (scaled limit) | finite-dim |
| `open:reduction` | 12_open_formation:258 | RP | BC | explicit global, equals unique local solution |
| `open:open-laws` | 12_open_formation:295 | RP | BC open set | perturbation |
| `open:final-merge` | 12_open_formation:305 | RP | BC | n/a |
| `open:parent-flow` | 12_open_formation:362 | RP | BC | quotient by equivariance + uniqueness; no Banach quotient (:402) |
| `open:quotient-gradient` | 12_open_formation:391 | RP | BC; gradient claim on finite-dim von Mises family | pointwise dissipation |
| `open:all-partitions` | 12_open_formation:503 | RP | BC | n/a |
| `open:formation` | 12_open_formation:511 | RP | BC | finite horizon |
| `pa:four-transfer` | 13_per_agent_inheritance:60 | PFE Gibbs-reduced, unit T | FIN | finite-dim |
| `pa:four-grouping` | 13_per_agent_inheritance:96 | PFE | FIN | finite-dim |
| `pa:persistent-grouping` | 13_per_agent_inheritance:138 | PFE | BIN | finite-dim, global |
| `pa:angular-flow` | 13_per_agent_inheritance:214 | PFE | EXPC exponential family | tangency only; no chart for full per-agent density flow (02) |
| `pa:angular-grouping` | 13_per_agent_inheritance:241 | PFE | EXPC | finite-dim on family (02) |
| `pa:phase-parent` | 13_per_agent_inheritance:301 | PFE (+ null) | EXPC | finite-dim on family (02) |
| `pa:geometry-inheritance` | 13_per_agent_inheritance:336 | PFE (diagnostic geometry) | EXPC | n/a |
| `pa:nonflat-window` | 13_per_agent_inheritance:352 | PFE | EXPC | finite window |
| `lp:native-field` | 14_collective_local_parent:43 | PFE Gibbs-reduced | EXPC (R^24 parameters) | tangency only (02) |
| `lp:local-realization` | 14_collective_local_parent:111 | PFE | EXPC | finite-dim jets |
| `lp:generator` | 14_collective_local_parent:184 | PFE | EXPC finite generator | n/a |
| `lp:causal-flow` | 14_collective_local_parent:214 | PFE | EXPC | finite-dim |
| `lp:causal-coherence` | 14_collective_local_parent:272 | PFE | EXPC | n/a |
| `lp:jet-obstruction` | 14_collective_local_parent:319 | PFE (geometry) | EXPC sections | n/a |
| `lp:profile-obstruction` | 14_collective_local_parent:337 | PFE | EXPC | n/a |
| `lp:path-memory-obstruction` | 14_collective_local_parent:345 | PFE | EXPC | n/a |
| `lp:causal-jet-obstruction` | 14_collective_local_parent:372 | PFE | EXPC | n/a |
| `ctx:f:statement:1` | 15_contextual_coupling:107 | NONE (geometry) | RFD | n/a |
| `ctx:f:statement:3` | 15_contextual_coupling:136 | RP + contextual cost | RFD | branch hypothesis |
| `ctx:f:thm:ode` | 15_contextual_coupling:220 | RP + contextual cost | RFD restricted family | finite-dim local, maximal in regular domain |
| `ctx:c:statement:2` | 15_contextual_coupling:527 | RP + contextual cost | RFD | n/a |
| `ctx:c:thm:local` | 15_contextual_coupling:654 | RP + contextual cost | RFD on flat torus | H^s x H^(s-1), s>d/2+2, local unique, continuation |
| `ctx:c:lem:heat` | 15_contextual_coupling:675 | NONE (PDE lemma) | periodic Sobolev | explicit |
| `ctx:c:statement:5` | 15_contextual_coupling:804 | RP + contextual cost | RFD | energy identity on the solution |
| `ibt:frame` | 16_link_base_transport:48 | NONE (geometry) | matrix data | n/a |
| `ibt:displacement` | 16_link_base_transport:125 | NONE | GEN laws with moments | n/a |
| `ibt:reciprocal` | 16_link_base_transport:168 | NONE | matrix data | n/a |
| `ibt:path` | 16_link_base_transport:187 | NONE | matrix data | linear path ODE |
| `ibt:curvature` | 16_link_base_transport:206 | NONE | matrix data | n/a |
| `ibt:group` | 16_link_base_transport:264 | NONE | matrix data | n/a |
| `ibt:alignment` | 16_link_base_transport:288 | NONE | Riemannian target | n/a |
| `ibt:invisible` | 16_link_base_transport:340 | NONE | matrix data | n/a |
| `it1:loss` | 17_communication_links:64 | NONE (reconstruction) | GEN | n/a |
| `it1:noisy` | 17_communication_links:74 | NONE | GEN | n/a |
| `it1:kalman` | 17_communication_links:86 | EXACT | GAUSS | n/a |
| `it1:gaussian-reconstruction` | 17_communication_links:105 | NONE | GAUSS | n/a |
| `it1:identifiability` | 17_communication_links:140 | NONE (identifiability) | (i),(iii) any location noise; (ii) GAUSS | n/a |
| `it1:modalities` | 17_communication_links:170 | NONE | location noise | n/a |
| `it1:routed` | 17_communication_links:185 | NONE | GAUSS noise (location extension :165) | n/a |
| `it1:query-dictionary` | 17_communication_links:199 | NONE | LIN | n/a |
| `it1:stack-compatibility` | 17_communication_links:207 | NONE | LIN | n/a |
| `it1:consensus` | 17_communication_links:232 | NONE | GEN (bimeasurable dictionaries) | n/a |
| `it1:frustration` | 17_communication_links:240 | NONE | LIN | n/a |
| `it1:bandwidth` | 17_communication_links:264 | EXACT (source posterior) | GEN | n/a |
| `it1:learning` | 17_communication_links:320 | EXACT (tangent posterior) | LGT | n/a |
| `it1:complementary` | 17_communication_links:384 | EXACT | LGT | n/a |
| `it1:unreachable` | 17_communication_links:400 | NONE | LIN (any location noise) | n/a |
| `it1:complete-exchange` | 17_communication_links:420 | EXACT | LGT | n/a |
| `it1:pair` | 17_communication_links:432 | NONE | LIN | n/a |
| `it1:exact-pair` | 17_communication_links:453 | NONE | GL(K) exact | n/a |
| `it1:exact-flat` | 17_communication_links:467 | NONE | GL(K) exact | n/a |
| `it2:gauge` | 18_link_dynamics:62 | record law (all objectives) | LGT; general version in remark at :60 | n/a |
| `it2:split` | 18_link_dynamics:80 | GFE and PFE (one record) | GAUSS | n/a |
| `it2:grounding` | 18_link_dynamics:103 | EXACT | GAUSS | n/a |
| `it2:global-flow` | 18_link_dynamics:114 | GFE | GAUSS product recognition | finite-dim mean flow |
| `it2:per-agent-flow` | 18_link_dynamics:127 | PFE | GAUSS | finite-dim mean flow |
| `it2:ring` | 18_link_dynamics:141 | PFE vs GFE | GAUSS, K=1 | finite-dim linear |
| `it2:separation` | 18_link_dynamics:158 | GFE and plug-in PFE | GAUSS | n/a |
| `it2:echo` | 18_link_dynamics:186 | GFE exact/factorized, PFE, cavity | GAUSS | n/a |
| `it2:potential-forces` | 18_link_dynamics:267 | RP vs widened GFE | GAUSS, fixed covariances | finite-dim mean flow |
| `it2:certainty` | 18_link_dynamics:285 | GFE, PFE (i); RP (ii),(iii) | GAUSS | n/a |
| `it2:coupled-flow` | 18_link_dynamics:329 | RP | GAUSS restricted family (:7) | finite-dim |
| `it2:coupled-global` | 18_link_dynamics:390 | RP | GAUSS restricted family | global finite-dim |
| `it2:coupled-critical-structure` | 18_link_dynamics:417 | RP | RFD | Hessian inertia |
| `it2:coupled-threshold` | 18_link_dynamics:445 | RP | GAUSS | linearization |
| `it2:critical-pitchfork` | 18_link_dynamics:533 | RP | GAUSS | center manifold (Ruelle) |
| `it3:induced` | 19_curvature_hierarchy:62 | GFE evidence | LGT; general identity in remark at :49 | n/a |
| `it3:rank-one` | 19_curvature_hierarchy:141 | GFE; PFE and RP replacements | LGT | n/a |
| `it3:dissonance` | 19_curvature_hierarchy:166 | GFE/PFE/RP stationary | LGT | n/a |
| `it3:rigid` | 19_curvature_hierarchy:192 | GFE (supplied rigid restriction) | GEN with normalizers; GAUSS instance | n/a |
| `it3:flat-wall` | 19_curvature_hierarchy:228 | NONE (group holonomy); gain formula GFE | GEN group; LGT gains | n/a |
| `cmem:free-energy` | conditional_memory:40 | EXACT (local FE of one model) | GEN | n/a |
| `cmem:gaussian-posterior` | conditional_memory:87 | EXACT | GAUSS mixture | n/a |
| `cmem:response` | conditional_memory:120 | EXACT | GEN prior, Gaussian noise | n/a |
| `cmem:projection` | conditional_memory:136 | NONE (moment projection) | GAUSS mixture | n/a |
| `cmem:readout-loss` | conditional_memory:164 | NONE | GAUSS mixture | n/a |
| `imem:calibration` | interacting_memory:70 | EXACT | GEN | n/a |
| `imem:history-theorem` | interacting_memory:84 | EXACT | GEN | n/a |
| `imem:sender-receiver` | interacting_memory:118 | EXACT | GEN dominated | n/a |
| `imem:reduction` | interacting_memory:165 | EXACT | GEN | n/a |
| `imem:free-energy` | interacting_memory:238 | EXACT (full/private/local fixed-history FE) | GEN | n/a |
| `imem:reduced-free-energy` | interacting_memory:298 | EXACT | GEN | n/a |
| `imem:gaussian-receiver` | interacting_memory:484 | EXACT | GAUSS banks | n/a |
| `imem:gaussian-closure` | interacting_memory:560 | EXACT | affine GAUSS, one presentation | n/a |
| `imem:mixture-nonclosure` | interacting_memory:584 | EXACT | GAUSS two presentations (counterexample) | n/a |
| `imem:local-retention-failure` | interacting_memory:619 | EXACT | FIN binary | n/a |

## Answers to focus questions 2 to 5

**Question 2 (conclusions transferred between objectives without proof).** Two transfers are not proved as stated: the chapter 11 null control, which silently replaces the declared null objective by a different one (XC3-SCOPE-01), and the identification of the per-agent total with a potential of the reciprocal form (XC3-SCOPE-08). Every other cross-objective statement in scope is either proved or explicitly scoped, and I checked each: `prop:dyn-label-model` and `thm:dyn-registered` relate a normalized global model to the reciprocal potential with the extra terms displayed (verified numerically, residuals at most 8.9e-16); `cor:dyn-frozen-source` is stated at frozen sources, and 3:419 says the resulting episodes are per-agent dynamics, not the reciprocal flow; `prop:dyn-per-agent` gives Gibbs rows at unit temperature only; `it1:bandwidth` "identifies a cost, not a population dynamics" (17:275); `it2:potential-forces` identifies only the mean-dependent quadratic at fixed covariances and rows with a widened global model, which I verified independently (spread 1.4e-14), and 18:283 disclaims the rest; `it3:rank-one` lists per-agent and reciprocal replacements and says "The replacements do not assert equality of dictionary forces across these objectives"; 15:257-262 shows that the per-agent variant of the contextual flow is not a gradient; 13:314 refuses to call the phase parent the gradient of the sum of child free energies; the interacting-memory appendix rules out "identification with the reciprocal law potential" (imem:6) and recomputation of policies inside the free energy (imem:323); chapters 6 to 12 each open by naming the reciprocal potential and disclaiming transfer, and 3:435 declines to claim them for the evidence-augmented potential (the base-only evidence variant of `rem:boundary-evidence` is stated as a separate potential and its transfer is proved there).

**Question 3 (Gaussian-family features presented as theory, and Gaussian-only results without a posed general version).** I found no Gaussian-family feature presented as a property of the theory in the main text. Each feature on the AUTHORING list is flagged where used: data-independent posterior covariance (17:103; `it2:certainty` (i) at 18:287), exact mean-field means (17:483; 18:125), closed-form Schur complements and the linear-Gaussian closure (18:101, 18:205, 18:283; `imem:gaussian-closure` with `imem:mixture-nonclosure` and imem:580, :615), finite-dimensional sufficient statistics (7:254; 6:606-620; imem:560-580), closed-form log-normalizers (19:259-262), and the transitive affine action (15:377, where the zero contextual cost of a transitive action is stated generally). The single unflagged instance is in the companion (XC3-SCOPE-05). Every Gaussian-only result has its general-law version posed (`it1:open-learning`, `it1:open-general`, `it2:open-general`, `it3:open-general`, `imem:open`) or proved first in general form (`it1:loss`, `it1:noisy`, `it1:location`, `it2:gauge-general`, `it3:general-induced`, `cmem:free-energy`, the general history filter of the interacting-memory appendix). Two Gaussian or lognormal results have their general versions discussed only in remarks rather than posed: the finite-time fusion of chapter 6 (`fus:sufficiency`, `fus:smooth-open`) and the negative statement `prop:dyn-gaussian-noninvariance`, whose positive counterpart (which finite-dimensional families are invariant) is not posed; both are taken up under Extensions.

**Question 4 (Fisher flows on non-finite spaces without the regularity declarations).** The reciprocal flow on densities and circles always has its declarations now: chapter 7 (`smooth:hypotheses`, `smooth:wellposed`, retained flow unique, global time not claimed), chapter 9 (bounded log-density chart, local Lipschitz field, globally Lipschitz coarse diagonal flow), chapter 10 (`lem:boundary-fine-wellposed`, `lem:boundary-global`, quotient flow by equivariance and uniqueness), chapter 12 (C^r chart, `open:force`, explicit global solution in `open:reduction`, quotient flow by equivariance with no Banach quotient manifold asserted, 12:402), and chapter 15 (finite-dimensional regular family, H^s with s > d/2 + 2). Coarse-flow uniqueness is proved where autonomy is claimed, with the chapter 8 parent flow explicitly restricted to realizable curves (8:252). The exceptions are the per-agent flows of chapters 13 and 14 on {-1,1} x S^1 x S^1 (XC3-SCOPE-02), the Wishart density flow of chapter 11 (XC3-SCOPE-06), and the weak sense of the Gaussian restricted flow (XC3-SCOPE-03).

**Question 5 (deep checks).** The ten highest-stakes items in this lens, and what I did: (1) the chapter 11 null control, integrated under both conventions (XC3-SCOPE-01); (2) per-agent chart well-posedness for chapters 13 and 14, reconstructed (XC3-SCOPE-02); (3) `prop:dyn-gaussian-noninvariance` and the projection sentence, numerically and in closed form (XC3-SCOPE-03); (4) `prop:dyn-label-model`; (5) `thm:dyn-registered` (i), (iii), (iv), with the type-class formula checked by enumeration; (6) `prop:dyn-global-local`; (7) `it2:potential-forces` (i) by an independent quadrature; (8) `it2:coupled-threshold` (T_c and T_mean in exact arithmetic) and `it2:ring`; (9) the chapter 9 binary expansion behind the asymptotic-phase response defect; (10) the Gaussian-feature caveat by a non-Gaussian counterexample, together with the exact binary exchange of the interacting-memory appendix. Results are in the next table and in the findings.

## Checked without defect

Numerical agreement supports an identity; it does not prove it. "Reconstructed" means I rederived every load-bearing step by hand.

| Label | What was checked | Method | Residual or reasoning |
|---|---|---|---|
| `prop:dyn-label-model` | `T F = E + sum_i [H(q_i) + (T-1) KL(q_i||r_i)] + sum_ij beta_ij G_ij` | `ch03_identities.py`, random 3-agent tables, T in {0.3, 1, 2.5} | at most 8.9e-16 |
| `thm:dyn-registered` (i), (iii), (iv) | type-class probability `|T_q| exp(-n[H(q) + KL])`; `T F = E + C_n + T log|X_n|`; `E_red = -T log P(o|x) - C_n` | enumeration of all sequences at n = 4; two-agent lattice | 1.7e-16, 4.4e-16, 2.8e-16 |
| `prop:dyn-global-local` | local free-energy identity | `ch03_identities.py`, 3x4x2 table | 0.0 (domain caveat XC3-SCOPE-07) |
| `prop:dyn-nonreciprocal` | eigenvalues `-1/(1+w)`, `(-1 - 3w/2 +- i sqrt(3) w/2)/(1+w)` | circulant, numeric | 5.0e-16 |
| `prop:dyn-gaussian-noninvariance` | velocity leaves span{1, x, x^2}; vanishes without the source term | `gaussian_noninvariance.py`, Gauss-Hermite | outside fraction 0.02 to 0.33 |
| `prop:dyn-chart-flow` | local Lipschitz map `l -> E_{q_l} f`, Picard-Lindelof, blow-up alternative, sufficient conditions for KL terms | reconstructed | correct |
| `prop:dyn-global-flow` | Dini floor `m >= min(m(0), -C)`, row floor, compactness, Lojasiewicz | reconstructed | correct |
| `prop:dyn-morse`, `prop:dyn-row-relaxation`, `prop:dyn-gibbs`, `prop:dyn-compensation`, `prop:dyn-jensen`, `prop:dyn-per-agent`, `cor:dyn-frozen-source` | Schur complement and Haynsworth inertia; row log-odds rate T; Gibbs gap; pool identity; Jensen identity; per-agent expansion; chain rule | reconstructed | correct |
| `thm:predictive-chain`, `prop:conditional-lift`, `cor:pythagorean`, `prop:common-family-recovery`, `cor:local-family-recovery`, `thm:predictive-distortion` | chain rule, Jeffrey reconstruction, Halmos-Savage descent, decoder decomposition | reconstructed | correct, general laws |
| `inh:projectable` (remediation-added) | criterion and `C^k` parent from local sections | reconstructed | correct |
| `inh:smooth-mixture-failure`, `inh:smooth-boundary-failure`, `inh:role-loss`, `inh:intertwining` example | 1/8 and 3/32; 20/256 and 17/256; -1.2 and -1.8; `Lambda T = T^c Lambda` | by hand | exact |
| `inh:common-clock` | `h_i = 2 log q_i + V_t` with dynamic or Gibbs rows; null coefficient -1 | reconstructed | correct for its hypotheses |
| `fus:sync` (remediation-added) | `D` Dini inequality, `t_0`, `T_*` bound, persistence by expansion into unit children | reconstructed | correct |
| `fus:two-stages`, `fus:dyadic` | collision times; approach speeds in `[2 lambda, N lambda]` | by hand | exact |
| `smooth:split-dynamics`, `smooth:spectrum-flow`, `smooth:wellposed`, `smooth:cgf-completion` | private equation `d/dt l = H - l`; range bounds and `|dH/dt| <= M^2`; fine and retained uniqueness; `Lambda^t(b) = Lambda^0(ab) - b Lambda^0(a)` | reconstructed | correct |
| `erase:gate-entry`, `erase:pair-convergence` (remediation-added) | constant B; zero `x = y = 1/2` from `atanh(1/2) = (log 3)/2`; Hessian eigenvalues `4/3 +- log 3`; implicit-function uniqueness; LaSalle | reconstructed | correct |
| `thm:public-phase`, `cor:public-phase-unique` | `kappa_N = 2N/(N-1)`; `omega > L = 1 + |J|` admissible; integral formula and uniqueness argument | reconstructed (decay constants not recomputed) | correct in structure |
| `eq:public-phase-binary-expansion`, `eq:public-phase-binary-parent` | `(6 - 3J) m a^2`, `-(5 - J) a`, `c = 3(2 - J)/(2(5 - J))` | `ch09_binary_sympy.py` | exact |
| `prop:public-gibbs-identifies`, restricted witness | hyperplane and Vandermonde argument; intercept 27/32, velocities 1/9 and (5 sqrt 5 - 10)/9 | by hand | exact |
| `lem:boundary-global`, `thm:boundary-tree-flow`, `prop:boundary-contrast-decay` | K bound and row floor; suffix bijection for invariance; rates `lambda_m`, `3 + 2/(N-1)`; Fisher-Rao bounds | reconstructed | correct |
| `form:kappa-independence` (i), (ii) (remediation-added) | shape law; one-dimensional centroid agglomeration without reversals; event times 0.3466, 0.5493, 1.0700 | reconstructed and `ch11_null_convention.py` | exact; (iii) see XC3-SCOPE-01 |
| `pa:persistent-grouping`, `pa:angular-flow` | `M = -(7/5) I + (2/5) 11^T`, eigenvalues, `u_* = (15/7, 15/7, 5/7)`; null equation; coefficient matching | by hand | exact; chart caveat XC3-SCOPE-02 |
| `it2:potential-forces` (i) | tangent reciprocal potential minus `E^S` is mean independent | `ch18_objectives.py`, exact Gauss-Hermite expectation and full Gaussian KL | spread 1.4e-14 |
| `it2:ring` | eigenvalues `-1` and `-(1 - zeta w)` | `ch18_objectives.py`, ring of 5 | 1.0e-15 |
| `it2:coupled-threshold` | stationarity of `x_*`; `g`; `H` positive definite; `g^T H^{-1} g = 6538/5059`; `T_c = 3269/10118`; `5059 H^{-1} g`; `T_mean = 31/122` | `ch18_threshold_sympy.py`, exact rationals | exact |
| `it3:induced` | DAG example 1/3; rank `K(|E| - |V| + 1)` | by hand | exact |
| `cmem:projection`, `cmem:readout-loss`, joint Fisher example | moment identity `KL(p||q) - KL(p||q_G) = KL(q_G||q)`; diagonal loss; reverse bound; derivative `-t^3/(1 - t^2 + c)^2`; coefficient `(1 + t^2)/(1 - t^2)^2`; fourth moment `3 - 2 t^4` | by hand | exact |
| interacting-memory binary exchange | 45/62, 5/22, 7/11, 12/31, lambda_0, lambda_1, 33/64, lambda_2, 437/1918, 21/64, 13/64, 45/64, 171/244, parity 7/16, 9/16, 15/32, 17/32 | `misc_checks.py`, exact fractions | all agree; 308/959 equals 44/137 (unreduced) |
| `imem:information-bound`, `imem:reduction`, `imem:free-energy`, `imem:reduced-free-energy` | two-order chain rule; reconstruction by W; free-energy identities | reconstructed | correct |
| Gaussian-feature caveats (17:483, 18:125) | factor means differ from exact means for a non-Gaussian target | `misc_checks.py` (a) | (0.156, 0.012) versus (0.444, 0.197) |

## Prior-review findings in scope

The brief named H1 and H8 of the 2026-10-22 REPORT. I also checked the well-posedness layer R3, whose items fall under focus question 4. Measurability items in R3 (L01-FOUND-05, -06, -13), the fusion inclusion (L05-FUS-14) and the numerical-scheme items (X5-IMPL-06, L05-FUS-19) belong to other lenses and are not assessed here.

| Prior ID | Status | Evidence in the current text |
|---|---|---|
| 2026-10-22 H1 (no normalized model, evidence or persistent record term) | resolved | `prop:dyn-no-global-vfe` (03:375) proves the non-identification; `def:dyn-record-interrupted` (03:421) defines the piecewise-deterministic process with a citation to Davis 1984; 03:425 lists the four things the potential gives up; `def:dyn-evidence-potential` (03:428-433) declares the evidence-augmented potential and states that chapters 6 to 12 are not claimed for it; abstract 5 and outlook 6 state the separation. The residual wording at 03:357 is XC3-SCOPE-08. |
| 2026-10-22 H8 (Gaussian family not invariant once an agent is a source) | resolved; low residuals | `def:dyn-restricted-flow` (03:204) with the Gaussian natural-gradient formulas; `prop:dyn-gaussian-noninvariance` (03:215), which my script supports; 09:225 explains why chapter 9 works on full charts; 18:7 says chapter 18 concerns the restricted family. Residuals: the Fisher-norm projection is undefined for wide sources (XC3-SCOPE-03); chapter 15 still calls its restricted flows "reciprocal Fisher flows", and the abstract paragraph on that chapter names neither the objective nor the family (XC3-SCOPE-04). |
| 2026-10-22 R3 (one well-posedness layer) | partially resolved | The chart proposition exists (`def:dyn-chart-flow`, `prop:dyn-chart-flow`, 03:225-232), but it is cited nowhere: a grep of the whole manuscript directory finds the label string only in its two headings (03:225, 03:228). Chapters 7, 9 and 10 reprove local existence in their own charts (07:197, 09:33, 10:50). The per-agent flows of chapters 13 and 14 have neither a chart nor a citation (XC3-SCOPE-02). |
| 2026-10-22 L03-DYN-03 (Fisher dynamics used on general densities without well-posedness) | partially resolved | Covered for the reciprocal potential by 03:225-232, 09:33 and 10:50. Not covered for per-agent flows on {-1,1} x S^1 x S^1 (XC3-SCOPE-02). Abstract 5 still says "coupled Fisher dynamics" with no chart or family qualifier. |
| 2026-10-22 L03-DYN-05 (global existence and convergence for fixed channels) | resolved | `prop:dyn-global-flow` (03:180): Dini floor, compactness, Lojasiewicz; reconstructed without defect. |
| 2026-10-22 X5-IMPL-18 (log-barrier global existence) | resolved | Same proposition; the floor argument replaces the log-barrier estimate. |
| 2026-10-22 L07-BND-04 (fine flow on the circle sample space) | resolved | `lem:boundary-fine-wellposed` (10:50). |
| 2026-10-22 L07-BND-15 (global existence by an energy bound) | resolved | `lem:boundary-global` (10:130-138), with the bound `eq:boundary-global-bound` and the row floor; "exists for all t >= 0". |
| 2026-10-22 G2-01 (unstated public-sector hypotheses) | resolved | `smooth:hypotheses` (H1)-(H3) at 07:102-108, with two counterexamples at 07:109; `smooth:wellposed` (07:197). |
| 2026-10-22 G2-02 (autonomy needs uniqueness of the forced coarse flow) | resolved | 07:24 now makes uniqueness a hypothesis of the sentence; 07:195 and `smooth:wellposed` (ii) and (iv) supply it. |
| 2026-10-22 G2-03 (bounded-log hypothesis must be uniform) | resolved | (H1) at 07:105 is uniform in the public coordinate; 07:109 gives the counterexample with bounds growing toward a boundary point. |
| 2026-10-22 G2-04 (range propagation; global public bounds) | partially resolved | Range forward invariance is `smooth:wellposed` (i) (07:200, 07:206). Global public bounds are not given: 07:215 states that T* = infinity "requires further estimates on the public sector, which are not given here". Disclosed. |
| 2026-10-22 G2-05 (uniqueness of the weak spectral equation) | resolved | 07:175 states uniqueness in the narrowly continuous class supported in I, and 07:189 proves it by Ambrosio, Gigli and Savare 2008, Section 8.1. |

## Extensions and improvements

**E1. A per-agent chart flow with global existence (closes XC3-SCOPE-02).** The proof of `prop:dyn-chart-flow` never uses the potential: it uses only that the first variation is Lipschitz on bounded chart sets. A per-agent version can therefore be stated in three lines, and the per-agent structure gives more than the reciprocal one, namely global existence. For agent i the Gibbs-reduced own-block force is `h_i = sum_j beta_ij [log q_i - log nu_ij] + (self and evaluator terms) - E_{s_i} log l_i(o | ., m)` with unit-temperature Gibbs rows (`prop:dyn-per-agent`). In the chart `|log q_i|_inf <= 2 |l_i|_inf`. If every peer output satisfies `|log nu_ij|_inf <= C + c max_j |l_j|_inf` (identity channels; the `mu_j x Haar x Haar` outputs of chapter 13) and the evaluator and record log densities are bounded, then `|h_i|_inf <= C' + c' max_j |l_j|_inf`, because the per-agent force has no source-role term `q_k/q_i`, which is the only term that grows exponentially in the chart norm. Gronwall bounds `max_j |l_j(t)|_inf` by `(max_j |l_j(0)|_inf + C'/c') e^{2c't}`, and the blow-up alternative gives a global solution. With tangency to the angular families, which I rederived for `pa:angular-flow`, uniqueness in the chart identifies the chart solution with the family solution, and that proves the invariance sentences at 13:215, 13:236 and 14:68.

**E2. Which exponential families the reciprocal flow preserves (the positive counterpart of `prop:dyn-gaussian-noninvariance`).** Let `P_T = {exp(eta . T - A(eta)) rho}` be a minimal d-parameter exponential family on a standard Borel space, with identity channels, references and evaluators in `P_T`, and fixed or Gibbs rows. A proposition can be stated in two parts. (a) If every agent law has rho-density constant on the blocks of one fixed finite partition (a Jeffrey lift of a finite table, in the sense of `prop:conditional-lift`), the configuration stays in that class, and on it the reciprocal flow is the finite-table flow of the block alphabet. (b) Conversely, if some agent i has exactly one receiver k with `beta_ki > 0`, the hypothesis of `prop:dyn-gaussian-noninvariance`, and `P_T` is invariant for every configuration in an open subset of `P_T^N`, then the essential range of T consists of d + 1 affinely independent points, so `P_T` is such a Jeffrey lift. Sketch of (a): each of the three force types `log(q_a/r_a)`, `log(q_a/q_j)` and `q_i/q_a` is block constant, and the chain rule gives KL between two laws with a common within-block conditional equal to KL between their block marginals, so the rows are the rows of the quotient; the quotient flow is global when `prop:dyn-global-flow` applies on the block alphabet, and uniqueness in the chart (`prop:dyn-chart-flow`, with reference rho) identifies its lift with the full solution. Sketch of (b): for unbounded T, invariance is read through the logarithmic derivative, as in `prop:dyn-gaussian-noninvariance`. Tangency requires `exp(delta . T)` in span{1, T} on the support for all `delta = eta_k - eta_i` in an open set. If distinct points `y_j` of the essential range satisfy an affine dependency `sum_j c_j y_j = 0`, `sum_j c_j = 0`, then `sum_j c_j exp(delta . y_j) = 0` on an open set, which forces `c = 0` because exponentials with distinct exponents are linearly independent. On three collinear support points `t = 0, 1, 2` this is the explicit identity `e^{2 eta} - (1 + 2(e^eta - 1)) = (e^eta - 1)^2`, so `e^{eta t} = a + b t` forces `eta = 0`.

`scripts/XC3-SCOPE/expfam_invariance.py` integrates the Gibbs-reduced three-agent reciprocal flow with RK4 to t = 4. The block-constant family with three blocks on six points has zero velocity outside its tangent space and stays in the family to 8.2e-15; the binary full family stays to 1.4e-16. The families `T = t` on three points, `T = t` on six points and `T = (t, t^2)` on six points have outside fractions of the initial Fisher velocity 0.29, 0.043 and 0.084 and leave the family (sup log-distances 9.7e-4, 4.2e-3 and 2.3e-4). The script uses two receivers per source, so it supports (a) and illustrates (b) without testing the one-receiver hypothesis. Consequences: every exponential family whose sufficient statistic is not a simple function (Gaussian, von Mises, Poisson, Wishart) needs `def:dyn-restricted-flow` under the reciprocal flow once an agent is a source with an identity channel. Per-agent flows have no source-role term (03:357), so there tangency reduces to conjugacy of peer outputs, evaluators and record likelihoods with the family. Family invariance is therefore objective dependent, which is a concrete answer to part of `op:dyn-three-pictures`.

**E3. Two declared null controls in the formation standard (direction for XC3-SCOPE-01).** `inh:null-flow` deletes the peer costs "with all other data unchanged". When the self channel has a nonzero cost, that null changes two things at once: it removes the interaction, and it reweights the self force by a Gibbs weight `b_i < 1` that depends on `D_ii` and T. In chapter 11 the second effect alone makes (F5) hold formally (XC3-SCOPE-01). I propose declaring two controls inside `inh:null-flow`: the label-preserving null of `pa:null-row`, and the renormalized null in which the surviving self label has probability one, which holds the self force fixed and so isolates the interaction. (F5) would then be tested against the renormalized null whenever a self cost is nonzero, and against both when they differ. Under that reading the chapter 11 verdict "fails (F5)" is correct as written, since under self weight one the null reproduces the full event times (0.3466, 0.5493, 1.0700 in `ch11_null_convention.out`). The chapter 13 grouping, which is stated against the label-preserving null, would need to be rechecked against the renormalized one; I have not done that check. The coupling-deformation statement of `form:kappa-independence` (i) and (ii) (identical membership history for every `kappa >= 0`) is a third, separate control and can be reported under its own name.

**E4. Posing the general-law version of finite-time fusion.** `fus:sufficiency` observes that the square-root cost equals `|Delta mu|` only on location families, and that general families would need semiconvexity. A posed version could separate two cases. In the first, take a regular family whose Fisher geometry is a Hadamard manifold, with transports acting by isometries, and replace each peer cost by the in-family Fisher distance `d_g(theta_i, theta_j)`. For the lognormal location family of the chapter this is exactly `|Delta mu|`, since the Fisher metric of `LN(mu, 1)` in mu is one. Other Hadamard families: the univariate Gaussian family with free variance is a hyperbolic plane of curvature -1/2 (Costa, Santos and Strapasson 2015), and the fixed-mean Gaussian family on SPD matrices carries the affine-invariant metric, which has nonpositive curvature. On a CAT(0) space the distance is jointly convex (Bridson and Haefliger 1999, Prop. II.2.2), so with reference terms that are geodesically convex on the family the potential is convex on the product, and gradient flows of convex functionals on Hadamard spaces (Mayer 1998; Bacak 2014) give unique global solutions. The cone singularity of the distance at coincidence is the same as for `|Delta mu|`, so the sticking mechanism should persist, but this needs proof. In the second case, the full law space, the Fisher distance is the Hellinger angle `2 arccos int sqrt(p q)` on the sphere of radius 2 in L^2. For unit-variance lognormals it equals `2 arccos exp(-(Delta mu)^2/8)`, which agrees with `|Delta mu|` to leading order, is bounded by pi, and is concave on the whole half-line: `scripts/XC3-SCOPE/fusion_hellinger.py` gives the series `x - x^3/48 + O(x^5)` and a second derivative between -0.26 and 0 on (0, 20], after confirming the Bhattacharyya coefficient `exp(-x^2/8)` by quadrature to 2e-12. The restriction to the chapter's family is therefore semiconvex with lambda = -0.26 in the mean coordinate, but not convex. On the full law space the distance is not jointly convex either, so global uniqueness needs lambda-convexity with lambda < 0 on bounded sets in the sense of Ambrosio, Gigli and Savare (2008). The first open step is whether the reference KL terms are lambda-convex along these spherical geodesics. Neither case is proved here.

**E5. Define the restricted flow by the parameter equation (closes XC3-SCOPE-03).** `dtheta/dt = -g(theta)^{-1} d_theta E(p_theta)` needs only differentiability of E along the family. For a Gaussian source this holds even when `E_{q_i}[(q_k/q_i)^2]` is infinite, since `d_theta KL(q_k || p_theta) = -E_{q_k}[d_theta log p_theta]` needs only second moments of `q_k`. The projection statement then becomes a proposition on the subset where the full first variation lies in `L^2(q_i)`. This also gives chapter 15 (XC3-SCOPE-04) and chapter 18 one object to cite.

## Not checked

The following were not checked. The companion was searched (c17, c18) but not read. Figures, the supplement and the working documents Item_1 to Item_3 and Item_7 were not opened. Chapters 01 and 02 and the appendices other than conditional_memory and interacting_memory are outside this lens's scope, except where a scope result cites them. I did not compile LaTeX. I read these detailed estimates without recomputing them: the C^1 regularity and decay constants in `thm:public-phase`; the constants of `open:routing`; the obstruction proofs of chapter 14 (`lp:path-memory-obstruction`, `lp:causal-jet-obstruction`), beyond their objective and family; the Sobolev estimates of `ctx:c:thm:local`; the curvature formulas of chapter 16; `it1:learning` (iv) and (v); the constants c and D of `it2:critical-pitchfork`; and the rate in `it3:rigid`. Prior-review findings outside H1, H8 and the R3 regularity cluster are left to the chapter lenses. Statement E2 (b) is a sketch; the numerical check uses two receivers per source and does not test the one-receiver hypothesis. The global-existence claim in E1 depends on a peer-output bound that I verified for chapter 13 and assumed for the chapter 14 factors `L_j` and `B_j` from their bounded log densities.
