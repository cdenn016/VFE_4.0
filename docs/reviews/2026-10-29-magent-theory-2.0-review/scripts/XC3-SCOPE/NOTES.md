# XC3-SCOPE working notes (not the report)

## ch03 03_dynamics.tex (read 1-438 in full)
- thm:dyn-finite-flow: RP (reciprocal potential), finite positive tables, C2 law-dependent channels; local only. OK.
- prop:dyn-global-flow (remediation-added): RP, finite tables, fixed channels, c_min>0. Reconstructed floor argument (min log q Dini bound, row floor), Lojasiewicz. OK.
- prop:dyn-row-relaxation OK. prop:dyn-morse OK (Haynsworth; Sylvester inertia).
- def:dyn-restricted-flow + eq:dyn-gaussian-flow OK (mu'=-Sigma grad, Sigma'=-2 Sigma G Sigma).
- prop:dyn-gaussian-noninvariance: Gaussian, negative result, proof OK. Text :221 says ch07-12 proved for full density charts, do not transfer to restricted Gaussian flow -> H8 largely resolved in ch03. NEED: check ch15,17,18,19 Gaussian flows are labeled restricted/projection.
- def/prop:dyn-chart-flow: bounded log-density chart L^inf_0(lambda), Lipschitz-on-bounded-sets hypothesis, Picard-Lindeloef, blowup alternative; sufficient conditions (bounded log refs/evaluators, identity channels sharing lambda, or fixed kernels with two-sided density bounds). Local only. OK.
- prop:dyn-global-local: GLOBAL FE, general law. Minor gap: e_bar_i only defined Q_i-a.e. by Fubini but used lambda_i-a.e. in Z_i; competitors outside integrability class. low.
- prop:dyn-jensen general OK. prop:dyn-label-model finite, identity TF = E + sum[H+(T-1)KL] + sum beta G verified by hand.
- prop:dyn-per-agent: PER-AGENT FE, general w/ finiteness. OK. Text :357 "F_tot flow is reciprocal flow of a potential of the form eq:dyn-potential" -- loose: belief self-cost int KL(q||G(m,.)) s(dm) couples sectors, plus record terms (E_o form). low consistency.
- prop:dyn-nonreciprocal: eigenvalues verified by hand. OK.
- prop:dyn-no-global-vfe (a) finite only (stated), (b) proof only on two-point alphabets, statement not explicitly finite ("interior"). low.
- thm:dyn-registered verified (iii),(iv) by hand; finite type lattice; laws are point-recognized variables, not recognition laws (stated). cor:dyn-frozen-source OK.
- def:dyn-evidence-potential; text :435 says ch06-12 not claimed for E_o. H1 addressed in ch03.

## ch04 04_prediction.tex (read 1-236 in full)
- thm:predictive-chain, prop:conditional-lift, cor:pythagorean, prop:common-family-recovery, cor:local-family-recovery, thm:predictive-distortion: objective none, general law, all reconstructed (chain rule/Jeffrey/Halmos-Savage). OK.
- :58-63 Gaussian additive-noise response Cov(Y,X|O)C^-1 is general-prior identity with Gaussian noise; stated. OK. :86 Gaussian closure stated as family-specific. OK.

## ch05 05_inheritance.tex (read 1-1399 in full)
- general dynamics: inh:deterministic-descent (Kechris 14.12), inh:projectable (remediation-added; reconstructed OK, finite-dim C^{k+1} submersion connected fibers), inh:gradient-inheritance, inh:equivariant-descent (finite-dim; :290 says inf-dim charts need closed complement + quotient Banach manifold -> CHECK ch10, ch12, ch13 uses), inh:coordinate-projection, inh:no-underdamped (finite-dim char poly), inh:role-loss (numbers verified), inh:finite-lumpability, inh:intertwining (example verified LambdaT=T^c Lambda by hand), inh:canonical-parent, inh:canonical-lumping, inh:separator-gluing, inh:composition, inh:causal-memory (general), inh:finest-independent, inh:conditional-partition, inh:independence-discontinuity, inh:selector-symmetry: objective none.
- inh:smooth-mixture-failure / boundary-failure / no-finite-entry: RP specialization, binary finite; numbers verified by hand (1/8, 3/32; 20/256, 17/256).
- inh:scalar-additive: general ODE; inh:common-clock: RP, finite tables; proof reconstructed; transfer to null flow correct.
- inh:formation-open asks for RP, per-agent, or global -> fine.

## ch06 06_fusion.tex (read 1-707 in full)
- objective: nonsmooth modification of RP (sqrt KL), fixed rows; family: restricted lognormal fixed variance (declared constraint, not invariant family). :4 states no transfer. finite-dim maximal monotone (Brezis). fus:local-inheritance, fus:uniqueness, fus:sync (remediation-added; reconstructed Dini bound on D, t0, T*), fus:isolated-pair, fus:sticking, fus:two-stages (times verified), fus:open-two-stage, fus:dyadic (speed bounds [2lam, N lam] verified). fus:sufficiency remark flags Gaussian-specific |dmu| convexity. OK; no scope defect found.

## ch07 07_smooth_closure.tex (read 1-321 in full)
- :5 objective RP declared, no transfer. Regularity: hyp smooth:hypotheses (H1-H3) declared chart L^inf_0, Lipschitz first variations, smooth:wellposed proves local well-posedness of retained (coarse) system AND fine system and the lift; T*=infty not claimed (:215). Compliant with regularity rule.
- smooth:split-dynamics private eq d/dt l = H - l reconstructed. smooth:spectrum-flow uniqueness via AGS continuity eq. smooth:cgf-completion Lambda^t(b)=Lambda^0(ab)-b Lambda^0(a) verified. smooth:feedback composite = record-interrupted (declared). smooth:exponential-family flags finite-param closure as family feature. OK.
- smooth:mean-force-criterion / smooth:submersion general weak-form; disclaimers :16. OK.

## ch08 08_smooth_erasure.tex (read 1-263 in full)
- objective RP Gibbs-reduced (:5), finite product-Bernoulli invariant family (:64 argued). erase:gate-entry (B bound verified), erase:pairs, erase:pair-convergence (remediation-added; reconstructed: LaSalle, eps=0 zero x=y=1/2 since atanh(1/2)=J/2, Hessian eig 4/3 +- log3 >0, IFT, symmetry) OK. erase:parent-flow uniqueness only among realizable curves (stated :252). OK.
- :263 "(F5) is therefore met only through a shift of an event time" contradicts inh:recursive-formation (F5) "difference only in event times does not count" -> F5 FAILS. low consistency (formation wording, maybe other lens).

## ch09 09_public_compression.tex (read 1-251 in full)
- objective RP Gibbs-reduced identity channels (:4). Chart: full bounded log-density L^inf_0(A) declared (:23-39), local Lipschitz argued (identity channels -> bounded first variations), diagonal flow f globally Lipschitz L=1+|J| (coarse flow unique). thm:public-phase: kappa_N=2N/(N-1) reconstructed by hand; integral formula, rate condition omega>L, 2|J|<kappa_N verified. cor:public-phase-unique reconstructed. Compliant with regularity rule.
- prop:public-gibbs-identifies reconstructed (hyperplane+Vandermonde). cor:public-gibbs-channel OK. Witness c=(1+sqrt5)/8: intercept 27/32 and velocities 1/9,(5sqrt5-10)/9 verified by hand.
- :225 "mixture-geodesic force leaves every finite-dim exp family... statements made on full density charts" -> H8 (ch09 part) resolved.
- TODO sympy: binary expansion (6-3J) m a^2, rate 5-J, coefficient 3(2-J)/(2(5-J)).

## ch10 10_boundary_closure.tex (read 1-353 in full)
- objective RP identity channels (:4); general laws on supplied tree; lem:boundary-fine-wellposed declares L^inf chart + local Lipschitz; lem:boundary-global global bounded-log existence (Dini bounds reconstructed: K=F(0)+3||log rho||+2; row floor). thm:boundary-tree-flow invariance (suffix bijection) reconstructed. lem:boundary-equivariance; quotient only as orbit space (no Banach quotient manifold claimed) -> consistent with inh:equivariant-descent remark. prop:boundary-contrast-decay rates verified (lambda_m, 3+2/(N-1)). FR distance bounds 2sqrt(2H^2)..pi sqrt(2H^2) verified. :311 Gaussian mixture limitation flagged. Compliant.
- prop:boundary-observable :321 says "|J|<1/4 used in ch:public-compression" but ch09 uses |J|<1 (0<J<1 binary) -> low consistency (not my lens).
- rem:boundary-evidence: evidence-augmented potential, base-only; claims thm transfer with V^q - log L. Plausible (invariant under S). OK (objective explicitly named E_o variant).

## ch11 11_conditional_formation.tex (read 1-568 in full)
- objective RP exact Gibbs rows (:4 no transfer). Family: common-precision Wishart on S++_2 (declared; non-Gaussian exp family). form:full-flow: invariance because channel is constant-in-input (no source term) -> consistent with H8 remark. Regularity: density flow NOT in bounded log chart; uniqueness argued within "positive dominated regular solutions" (:170-175) via integrating factor -> solutions stay in Wishart span. Solution class undefined (low gap).
- form:mean-field: hypotheses "regular dominated family ... differentiation under integrals" declared as assumptions. OK.
- form:kappa-independence (remediation-added): (i),(ii) reconstructed (1-D centroid no reversal). (iii) NULL FLOW: proof writes null belief potential sum_i[KL(q||W(a0,I)) + KL(q||W(a0,Lambda_i^0))] i.e. SELF WEIGHT 1. But inh:null-flow says "deleting every peer cost ... all other data unchanged" (priors 1/N, temperatures). With priors kept and Gibbs rows, beta*_ii = (1/N)e^{-D_ii/T}/((1/N)e^{-D_ii/T}+(N-1)/N) < 1 and null shape law becomes alpha_i' = -(1+beta*_ii(t))(alpha_i-alpha0): agent-dependent clocks. AUTHORING (ch13 rule) "Keep the null's source priors and row entropy after deleting nonself costs." -> inconsistency of null-flow convention; (iii) H^0=H "including event times" depends on convention. CANDIDATE FINDING (medium). Consider numeric check.
- form:parent, form:responses, form:geometry, form:geometry-decay: Wishart-specific, stated. lambda for a0=kappa=1: 2 lambda = 12-4sqrt2 (not checked).

## ch12 12_open_formation.tex (read 1-580 in full)
- objective RP exact Gibbs rows (:4). Chart: C^r log densities in angle, r>=3 (:12) declared; open:force local Lipschitz (no angular derivative) -> unique local flow; open:reduction explicit global solution; open:interior global binary. open:parent-flow quotient via equivariance+uniqueness; moving-frame eq; :402 explicitly no quotient Banach manifold on finite-r chart (consistent with inh:equivariant-descent remark). open:quotient-gradient finite-dim von Mises family: free proper isometric -> OK. open:null-control: null keeps angular self terms (self cost H_i retained, peer costs deleted) -> CHECK null row convention here: rows? they say decoupled field xdot=-(1-x^2)atanh x  (only local term; self summand vanishes in binary eq) fine.
- limit field = Gaussian log-sum-exp (a limit, not Gaussian family). OK.
- Compliant with regularity rule.

## ch13 13_per_agent_inheritance.tex (read 1-398 in full)
- objective PER-AGENT (:5), Gibbs-reduced unit temp. Null convention :18-23 keeps labels+priors, beta0_ii = a e^{-D}/(a e^{-D}+1-a); "Replacing these rows by self probability one would specify a different control" -> CONFIRMS inconsistency with ch11 form:kappa-independence(iii) (self weight 1).
- pa:four-transfer/pa:four-grouping finite tables OK. pa:persistent-grouping: M=-(7/5)I+(2/5)11^T, u*=(15/7,15/7,5/7) verified; null eq verified.
- pa:angular-flow: per-agent flow on {-1,1}xS1xS1 exponential family; tangency verified (C,A,b,theta,l eqs). BUT per-agent dynamics def:dyn-per-agent-flow (03:354) is defined ONLY on finite positive tables; ch13 uses it on circles with no declared chart / Lipschitz / well-posedness of full density flow; "invariant family" and "exact Fisher velocity ... rather than projected approximation" (:236) need uniqueness of full flow. prop:dyn-chart-flow is stated for reciprocal E only. CANDIDATE FINDING regularity (medium). Check ch14 same.
- pa:phase-parent, pa:geometry-inheritance, pa:nonflat-window: on finite-dim family; OK given tangency.

## ch14 14_collective_local_parent.tex (read 1-378 in full)
- objective PER-AGENT (:9) unit temp Gibbs-reduced. lp:native-field: same tangency-only invariance on {-1,1}xS1xS1 exponential family (:68 "invariant family of the own-law Fisher field"); same regularity gap as ch13 (per-agent flow only defined on finite tables def:dyn-per-agent-flow 03:354-355; no chart/well-posedness for per-agent density flow). Null convention :80 keeps labels/priors/row entropy (consistent with ch13; inconsistent with ch11).
- lp:local-realization, lp:generator, lp:causal-flow, lp:causal-coherence, lp:jet-obstruction, lp:profile-obstruction, lp:path-memory-obstruction, lp:causal-jet-obstruction: finite-dim ODE in R^24 natural params; geometry statements. Not deep-checked beyond scope. Obstructions scoped (:335, :343, :378). OK.

## ch15 15_contextual_coupling.tex (read 1-964 in full)
- objective RP + contextual cost (:6), restricted finite-dim regular families declared (:22, :407 "Unrestricted joint Fisher dynamics need not preserve"), Sobolev class explicit (ctx:c:thm:local s>d/2+2), finite ODE (ctx:f:thm:ode), branch hypothesis. :257-262 per-agent variant explicitly not gradient (three-objective discipline). Gaussian witnesses flagged (:318 "full-law Fisher invariance is not asserted", :362). Compliant.
- low: :230 and :841 "recover exactly the chosen context-local reciprocal Fisher flows" -- these are restricted-family flows (def:dyn-restricted-flow), not the full reciprocal density flow; ch15 never cites def:dyn-restricted-flow. low consistency/exposition.
- Witness KL formula verified by hand.

## ch16 16_link_base_transport.tex (read 1-364 in full)
- objective none (geometry selectors); laws general w/ finite first moment, smoothness of moments assumed (:95); non-Gaussian Laplace realization (:240-245). No flows except path ODE (ibt:path Picard). Compliant; nothing in lens.

## ch17 17_communication_links.tex (read 1-500 in full)
- objective: exact inference (Bayesian posterior in tangent model; it1:kalman, it1:learning), none (identifiability, consensus). Gaussian features flagged explicitly (:103 linear decoder/data-independent covariance/rank; :282 split general vs Gaussian conclusions; :483 mean-field exact means = Gaussian feature). it1:location extends (i),(iii) to any noise law, (ii) with J_f; Cramer-Rao J >= Sigma^-1 OK. it1:bandwidth :275 three objectives distinct. it1:consensus general laws. Compliant.
- it1:learning (ii) block inversion reconstructed; (iii) S=sigma^-2 I + lam L_N OK.

## ch18 18_link_dynamics.tex (read 1-641 in full)
- :5-7 three objectives distinguished; restricted family def:dyn-restricted-flow cited (:7) -> H8 resolved for ch18. it2:gauge-general general-law pointwise. it2:split Gaussian shares flagged (:101). it2:global-flow GLOBAL FE Gaussian (invariant family since quadratic energy); exact means flagged as Gaussian (:125). it2:per-agent-flow PER-AGENT Gaussian; it2:ring non-gradient. it2:potential-forces RP mean quadratic = widened global model posterior mean; :283 disclaims identification of covariance objectives (good). it2:certainty (i) flags data-independent covariance as Gaussian feature (good). it2:coupled-flow/global/critical-structure/threshold/pitchfork RP on restricted Gaussian family; it2:open-general poses general-law version. Compliant.
- Per-agent flow on Gaussian laws in R^K: tangency holds (quadratic energies), but per-agent dynamics outside finite tables is undefined in ch03 (same definitional gap as ch13/14) -- here only as restricted-family Fisher flow of means at fixed cov; OK as declared Gaussian flow.
- numeric constants (Tc=3269/10118, Tmean=31/122, c, D) not my lens; maybe quick check Tc.

## ch19 19_curvature_hierarchy.tex (read 1-268 in full)
- objective: global evidence (exact marginal action), with per-agent/RP replacements in it3:rank-one explicitly "do not assert equality of dictionary forces". Gaussian tangent declared; it3:general-induced general-law evidence identity first; it3:open-general poses general version; it3:rigid general-law normalized restriction with explicit normalizer hypotheses + Gaussian instance. it3:dag-example 1/3 verified; rank K(|E|-|V|+1) verified. Compliant.

## appendices/conditional_memory.tex (read 1-229 in full)
- objective: exact inference / local free energy of one declared model (cmem:free-energy general law, finiteness assumed). Gaussian realization labeled (cmem:gaussian-posterior, cmem:projection, cmem:readout-loss). cmem:response general prior + Gaussian noise, hypotheses stated. Verified by hand: projection identity KL(p||q)-KL(p||q_G)=KL(q_G||q); diagonal loss; reverse counterexample bound; readout derivative -t^3/(1-t^2+c)^2; Fisher coefficient (1+t^2)/(1-t^2)^2; 4th moment 3-2t^4; diagonal rebasing => permutation x diagonal. Compliant.

## appendices/interacting_memory.tex (read 1-664 in full)
- objective: exact inference (Bayesian recursion), full/private/local free energies of fixed history (imem:free-energy, imem:reduced-free-energy) general law; :6 "No ... identification with the reciprocal law potential". imem:gaussian-closure Gaussian feature (finite-dim sufficient stat) scoped as "exact special case", imem:mixture-nonclosure shows failure; imem:open poses general. imem:information-bound reconstructed. Compliant.
- numbers (45/62, 5/22, 7/11, 12/31, lambda's, 437/1918, 21/64, 13/64, 45/64, 171/244, parity 7/16 9/16) -> optional script check.

## scripts run (all single-threaded, <30 s)
- ch11_null_convention.py / ch11_null_perturb.py: under ch13 null convention, ch11 null rates 1+b_i (b_i up to 0.23), event times all differ; u=(1,1.2,1.45,1.79), T=1: null partition {12},{123},{1234} vs full {12},{34},{1234}; 11/11 random admitted perturbations flip (margins 0.017-0.053). Independent replication of D15-WISH-01.
- ch03_identities.py: label-model, registered (i)(iii)(iv), global-local, three-cycle: residuals <= 9e-16.
- gaussian_noninvariance.py: outside-tangent Fisher fraction 0.02-0.33; E_{q_i}[(q_k/q_i)^2] = inf iff s2>=2 (cf D04-DYN-05).
- ch18_objectives.py: recip - E^S constant (spread 1.4e-14); ring eigen err 1e-15.
- ch09_binary_sympy.py: (6-3J) m a^2, -(5-J)a, c=3(2-J)/(2(5-J)) exact.
- ch18_threshold_sympy.py: stationarity, g, H>0, Tc=3269/10118, H^-1 g, Tmean=31/122 exact.
- misc_checks.py: Ising pair exact means (0.156,0.012) vs mean-field (0.444,0.197); imem binary exchange all fractions match (308/959 = 44/137 unreduced).
## overlaps with other lenses: D15-WISH-01 (null), D17-PA-03 (per-agent chart), D04-DYN-05 (Gaussian Fisher norm), D05-OBJ-08(ii) (F_tot form), D13-ERAPUB-02 (F5 wording ch08), D04-DYN-02/D05-OBJ-10 (no-global-vfe scope), D13-ERAPUB (boundary-observable J range?)
