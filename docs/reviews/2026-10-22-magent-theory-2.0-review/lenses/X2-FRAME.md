# X2-FRAME: Claims, scope, falsifiability, and internal consistency

I read in full: `frontmatter.tex` (1-12), `README.md` (1-24), `AUTHORING.md` (1-33), `preamble.tex`, the three root files, every main chapter (`01_foundations` 1-118, `02_interactions` 1-156, `02_geometry` 1-334, `03_dynamics` 1-184, `04_prediction` 1-167, `05_inheritance` 1-802, `06_fusion` 1-422, `07_smooth_closure` 1-247, `08_smooth_erasure` 1-222, `09_public_compression` 1-206, `10_boundary_closure` 1-222, `11_conditional_formation` 1-522, `12_open_formation` 1-470, `06_outlook` 1-16), all four appendices, all four supplement chapters, `companion/glossary.tex`, `companion/c11_formation.tex`, and `companion/c12_open_formation.tex` 1-177. I read the openings and closings of the other companion chapters (`c01` 1-20, `c02`, `c03`, `c04`, `c05` 140-172, `c06`, `c07` 1-12 and 95-113, `c08`, `c09`, `c10`). I skimmed, by grep only, `pifb2/24_section.tex`, the integration records `Research/docs/research/2026-10-22-conditional-formation-integration.md` and `2026-10-21-endogenous-formation.md`, and `VFE_4.0/docs/model-contract.md`. Compiled chapter numbers differ from file names (file `11_conditional_formation` is Chapter 12, `12_open_formation` is Chapter 13, `06_outlook` is Chapter 14); I cite files throughout.

Verdict. The restart is sound statement by statement, but it is not yet a foundation for its own target. Every abstract, README, AUTHORING and companion claim I traced matches a result as stated, all cross-references resolve, the main text is readable without the supplement, and every identity I recomputed holds. The definition of inherited recursive formation that the outlook says two constructions satisfy is met by non-interacting agents: the conditional-formation construction holds verbatim at zero coupling. The six-agent open-law theorem's membership history is also reproduced with every peer force deleted. So the existence results do not yet test the opening thesis that "interactions can produce further agents". The remaining problem is spread over five places, two formal open problems are stale, and the construction chapters are independent engineered architectures that cite 4 of the 61 numbered results of the general chapters.

## Findings

### X2-FRAME-01: The formation definition is met by non-interacting agents; the conditional-formation construction holds unchanged at zero coupling
**Category:** weakness
**Severity:** high
**Location:** chapters/05_inheritance.tex:776-785 (inh:recursive-formation); chapters/11_conditional_formation.tex:76-89, 111-190 (form:full-flow, form:ode), 277-290 (form:selector), 292-348 (form:selection); chapters/06_outlook.tex:4; frontmatter.tex:7; chapters/01_foundations.tex:4
**Evidence:** The thesis (01_foundations:4) is "The purpose of the theory is to describe agents whose interactions can produce further agents" (status C, a program goal). The outlook (06_outlook:4) states "Two constructions satisfy the formation obligation in \cref{inh:recursive-formation} under different assumptions within the original smooth KL model" (status D, true as stated). In form:density-force, $h_i=2\log q_i-2c_0\log\det X+\Tr[(I_2+C_q)X]$. Substituting $\log q_i=(\alpha_i-3/2)\log\det X-\Tr(\mathsf PX)+\text{const}$ gives $h_i=2(\alpha_i-\alpha_0)\log\det X+\Tr[(I_2+C_q-2\mathsf P)X]+\text{const}$. The coupling $\kappa$ enters only through $\Lambda_q$, which multiplies the linear statistic $X$, so only $\dot{\mathsf P}$ depends on $\kappa$. The shape law $\dot\alpha_i=-2(\alpha_i-\alpha_0)$ contains neither $\kappa$ nor any other agent. Hence $b_I(t)=b_I(0)e^{-2t}$, and every join time $t=\tfrac12\log(|b_I(0)-b_J(0)|/\delta)$ and every partition in form:selection are the same for all $\kappa\ge0$. At $\kappa=0$ the channel $W_2(\alpha_0,I_2)$ is constant and the reduced energy is a sum of single-agent terms. form:full-flow, form:parent, form:selection and form:responses still hold. So does form:geometry: at $\kappa=0$, $\dot p=-2p+2$ gives $p_+>p_-$ and $\partial_ak_1>0$, so form:curvature is still nonzero.

Independent route (`scripts/X2-FRAME/ch11_kappa_independence.py`): I ran the natural-gradient flow of the one-dimensional analog ($W_1$ = Gamma, exact KL, exact Gamma Fisher matrix, central-difference gradient). Max $|\dot\alpha_i+2(\alpha_i-\alpha_0)|$ is 8.1e-9, 1.2e-8, 4.0e-8 and 1.1e-7 for $\kappa=0,0.5,1,3$. The same run gives $\dot p_i=-2p_i+1+C$ with residual at most 4.9e-8 for unequal precisions, and the example's event times $\tfrac12\log2,\tfrac12\log3,\tfrac12\log(17/2)$ with no $\kappa$ input. The numerical agreement in the analog supports the two-line algebra above; it is not a proof for the $2\times2$ case.

The definition asks for "population dynamics that selects nontrivial blocks". It forbids only reset source data and "a newly tuned attraction". It has no clause requiring that the selection depend on the interaction. It also omits the long-term document's own requirement to "Separate the asymmetry in the initial preparation from the learning that amplifies it" (pifb2/24_section.tex:10). The ch11 rule is also not a usable constitutive law outside form:gaps. "First reaches $\delta$ from above" never joins blocks that are already closer than $\delta$. With initial deviations b = 0, 0.05, 0.25 and $\delta=0.1$, the pair {2,3} joins at $\tfrac12\log2$; the centroid of {2,3} is then already 0.075 from agent 1, so agent 1, the nearest neighbor of agent 2, never joins. In Mayo's (1996) terms, the construction would pass identically with no interaction, so it has no severity against the hypothesis that the interaction does no selection work.
**Fix:** Add an interaction-necessity clause to inh:recursive-formation (sketch in X2-FRAME-09). Replace the abstract sentence "A conditional four-agent construction under the original smooth KL potential selects two parents and then their union" with "In a conditional four-agent construction under the original smooth KL potential, a declared threshold on independently relaxing shape statistics selects two parents and then their union; the membership schedule does not depend on the coupling $\kappa$, including $\kappa=0$." Add the same qualification at 11_conditional_formation:4-10 and 06_outlook:3-4.
**Falsifies:** An error in form:ode showing that the shape equation depends on $\kappa$ or on other agents' shapes, or a stated reading of inh:recursive-formation that excludes selection statistics with decoupled dynamics and under which this construction still qualifies.

### X2-FRAME-02: The open-law routing theorem's membership history is reproduced with every peer force deleted
**Category:** weakness
**Severity:** medium
**Location:** chapters/12_open_formation.tex:4, 125-232 (open:routing), 414-422 (open:formation); frontmatter.tex:7; README.md:3
**Evidence:** Script `ch12_null_model_events.py` uses the manuscript's initial data -A-2, -A+2, A-2, A+2, R-2, R+2, the same native rows $w_{ij}$ (self included) and the selector open:selector ($\varepsilon=1/10$, $\delta=1/20$). It compares the limiting field open:limit with the null field $\dot u=-u$, in which all peer forces are deleted. A=8, R=200: full gives pairs at 0.589 and core parents at 1.974; null gives 0.647 and 1.965. A=64, R=2000: full 0.589 and 4.055; null 0.647 and 4.111. A=4096, R=4e5 (the proof's regime): full gives 0.589 and 8.214 with the third pair exterior through t=10.5; null gives 0.647 and 8.271 with the same final blocks {4,5}, {0,1,2,3}. Under the null flow $u(t)=e^{-t}u(0)$, so the rows are a Gaussian kernel of bandwidth $e^{t}$ applied to the initial data. The membership history is then a kernel-bandwidth hierarchical clustering of the initial positions.

The peer forces do amplify the within-pair contrast. In `ch12_amplification.py`, at the core-parent event $b/a$ is 5.9e-5 (full) against 3.1e-2 (null) for A=64, and 3.0e-2 against 2.5e-1 for A=8. They also shift the first event time by 9%. They change no membership and no order. The proof itself uses a decoupled comparison for the third pair (12_open_formation:209-213). Its regime $A\ge4096$, $R\gg Ae^{L}$ is exactly the scale separation in which this happens.

The text concedes "The initial configuration chooses the eventual blocks" (12:231). The opening "Native source preferences select pairs and then a pair of parents" (12:4) and the abstract's "Native preferential pairs form" nevertheless attribute the selection to the routing dynamics. The theorem itself is status D; the attribution is status S. Caveat: I compared the limit field, not the exact scaled binary field at the theorem's fixed $h$. The proof's tube argument (12:139) transfers to the decoupled exact field in the same way, but I did not run that check.
**Fix:** After open:routing, add: "On the same initial data, the decoupled flow $\dot u=-u$, read by the same rows and \cref{open:selector}, produces the same membership history; the peer forces contract the selected pairs but do not change which blocks form." Then either prove interaction-necessary formation on an open set (X2-FRAME-09) or reword 12:4 as "native rows, read by a fixed threshold, detect the pairs fixed by the initial scale hierarchy".
**Falsifies:** A computation in the exact binary field at the theorem's A, R and h in which the decoupled flow violates a window of open:routing: a premature third-pair join, a failed contrast, or a different order.

### X2-FRAME-03: The remaining formation problem is spread over five places, and two formal open problems are stale
**Category:** consistency
**Severity:** medium
**Location:** chapters/06_fusion.tex:415-422 (fus:smooth-open); chapters/07_smooth_closure.tex:246-247 (smooth:formation-open); chapters/09_public_compression.tex:192-206; chapters/05_inheritance.tex:787-802; chapters/06_outlook.tex:16; companion/c08_smooth.tex (closing paragraph); AUTHORING.md:19
**Evidence:** Neither fus:smooth-open nor smooth:formation-open is cited anywhere (grep over all three documents) or marked as addressed. fus:smooth-open says smooth formation "requires a different formation criterion or an additional mechanism". Chapters 11_conditional and 12_open supply threshold criteria, so this problem is answered in the conditional sense the outlook claims, but the text does not say so.

smooth:formation-open asks for "an initially unpartitioned population whose evolution selects persistent groups, together with a noninjective collective retention that forgets some of their distinct public degrees of freedom while determining all inherited forces, channels, and declared feedback". Both formation constructions meet every clause if "declared feedback" may be empty (both declare observational queries only: 11:381, 12:341) and "persistent" means a finite window. Under other readings they do not meet it. As written, the problem cannot be decided. The outlook's statement (06_outlook:16) is prose without quantifiers. The section `sec:public-recursive-requirement` ends with two displayed requirements and no statement. AUTHORING.md:19 requires "Keep the remaining formation problem stated precisely after accounting for the conditional construction below."
**Fix:** Mark both open problems as answered conditionally, with cross-references to form:selection and open:formation, or delete them. Then state one open problem in the outlook. A draft is in "Extensions and improvements" below.
**Falsifies:** A single precise statement in the text that I missed, or explicit supersession notes on the two open problems.

### X2-FRAME-04: By the smooth-closure chapter's own criterion, the open-law parents barely address aggregation
**Category:** consistency
**Severity:** medium
**Location:** chapters/07_smooth_closure.tex:244; chapters/12_open_formation.tex:265-303 (open:physical-group, open:parent-law, open:phase-recursion), 287-293, 422; chapters/06_outlook.tex:4
**Evidence:** 07_smooth_closure:244 says the conditional-spectrum state "does not combine several distinct public agent laws into one smaller collective state. Retaining their complete labeled tuple and calling it a parent would leave that aggregation problem untouched." The open-law parent $q_I=\bigotimes_{i\in I}\tau_{a_i}q_i$ retains each child's entire law and contextual section, up to one constant rotation per child with $\sum a_i=0$. It keeps primitive counts and ports. The inherited channels are per primitive port, $\mathcal L^{ij}_{IJ}$, and represent "the original primitive sums and rows, not one newly fitted parent KL cost" (12:293). The second-stage selector reads summed primitive rows (open:row-recursion). What is discarded is an $(|I|-1)$-dimensional torus orbit of a symmetry of each agent's own dynamics, and the companion states that "Its state size grows with the number of children" (c12). The outlook counts this among the constructions satisfying the formation obligation without reconciling it with 07:244. The retention theorem is status D; "parent formation" in the aggregation sense of 07:244 is status S.
**Fix:** Either qualify 07:244 to say that a quotient of the labeled tuple by a symmetry group counts as aggregation, or add a compression clause to inh:recursive-formation and state that the open-law parents do not meet it. Example clauses: the retained state space has dimension bounded independently of the descendant count; or the retention is not a quotient by a subgroup of a product of single-agent symmetry groups.
**Falsifies:** Showing that the open-law retention discards more than the $\mathsf H_I$ orbit, or that parent interactions do not reduce to primitive ports.

### X2-FRAME-05: Exact inheritance is built into each architecture and holds for every partition, so selection and inheritance are logically independent; the constructions use almost none of the general theory
**Category:** weakness
**Severity:** medium
**Location:** chapters/11_conditional_formation.tex:206-257 (form:parent, "For any disjoint current partition"); chapters/12_open_formation.tex:263-322 (open:parent-flow); chapters/10_boundary_closure.tex:126-154; chapters/08_smooth_erasure.tex:171-182; chapters/07_smooth_closure.tex:58-79; chapters/05_inheritance.tex (general results)
**Evidence:** Each architecture builds inheritance in by design:

- Conditional formation: inheritance holds for any partition at any time, by Wishart convolution.
- Open-law: the $\mathsf H_I$ quotient works for any $I$, because the rows read only bit marginals and the channel replaces angles (open:cost, open:row).
- Boundary closure: inheritance comes from automorphisms of a supplied tree.
- Erasure: inheritance comes from a flat gate.
- Smooth closure: inheritance comes from shield channels that copy the receiver's own conditional.

In both formation chapters, primitive self-channels are chosen to be nonidentity (11:88-89, 12:12), so that a parent's internal channel has the same form as a primitive's. By contrast, def:int-generator sets $L_{ii}^a=0$ for primitive links (02_interactions:69), and AUTHORING.md:17 asks that primitive identity self-comparisons be kept distinct from inherited loops.

Citation graph (`general_theory_usage.py`): of the 61 numbered results in the general chapters and appendices, 4 are cited by the construction chapters or the outlook: geo:retained-connection, prop:dyn-gibbs, inh:selector-symmetry and inh:recursive-formation. Equation-level reuse of the potential (eq:dyn-potential, eq:dyn-reduced, eq:dyn-envelope) also exists. inh:deterministic-descent, the boundary square, inh:composition, inh:causal-memory, geo:collective and geo:signed-comparison are never invoked; the constructions re-prove their own commuting squares. No construction chapter cites a result of another construction chapter; the only link is a chapter-level pointer from boundary closure to public compression. This is a mild case of the "mathiness" failure mode in Lipton and Steinhardt (arXiv:1807.03341): general apparatus with no inferential role in the main results.
**Fix:** State the two schemes in X2-FRAME-10 and in "Extensions and improvements" as theorems in 05_inheritance, and present the constructions as instances of them. Add one sentence to each formation chapter saying that inheritance holds for every partition, so the selected partition is not distinguished by inheritance.
**Falsifies:** A construction in the manuscript in which inheritance holds for the selected partition and fails for some other partition at the same time.

### X2-FRAME-06: The curvature and rank-two contextual geometry are prepared, not produced
**Category:** weakness
**Severity:** medium
**Location:** chapters/11_conditional_formation.tex:421-514 (form:context, form:geometry, form:curvature); chapters/12_open_formation.tex:359-410 (open:curvature, open:rank, open:equivariant-lift); frontmatter.tex:7; README.md:3
**Evidence:** The contextual dependence is assigned by hand: $\mathsf P(0;a,\chi)=R_\chi\operatorname{diag}(e^a,e^{-a})R_\chi^T$ in one construction, and the tilt $e^{(\kappa-\kappa_*)\cos(\theta-\vartheta_i)}$ followed by $\tau_\chi$ in the other. The dynamics acts pointwise in the context (01_foundations:31). The selection statistics are made context-independent by assumption (11:438 "the selection schedule is also constant over this contextual chart"; 12:371 "Binary coordinates are initially context independent").

The minimizing connection $A^F=-k_1/(k_1+k_2) d\chi$ is the mechanical connection of a two-component rotor with shape-dependent inertia. Its curvature $-(\partial_ak_1)k_2/(k_1+k_2)^2$ (recomputed) is nonzero because the chosen context coordinate varies one component's inertia and not the other's. "Preserved curvature" therefore checks that the retention is a Fisher isometry on prepared jets, which geo:retained-connection already characterizes. It does not show that formation or interaction generates contextual geometry, and in the conditional-formation construction it holds at $\kappa=0$ as well. The appendix open problem open:conn-derived concedes that deriving transport from interactions is open. The calculation is status D. The abstract's "rank-two contextual geometry with nonzero curvature" is accurate only once "prepared" is added.
**Fix:** In the abstract, write: "...and preserve the collective connection and contextual tensor of a prepared two-parameter contextual family, which has rank two and nonzero curvature."
**Falsifies:** A construction in the manuscript whose later contextual sections carry curvature that is neither present in the initial sections nor their pointwise image under the flow.

### X2-FRAME-07: Both formation selectors are known clustering procedures, and neither is cited as such
**Category:** citation
**Severity:** medium
**Location:** chapters/11_conditional_formation.tex:275-348; chapters/12_open_formation.tex:86-94 (open:selector), 128-139 (open:limit)
**Evidence:** Conditional formation: every $b_I(t)=b_I(0)e^{-2t}$, and $b_{I\cup J}$ is the size-weighted centroid. The event schedule is therefore centroid (UPGMC) agglomerative clustering of the initial belief shapes, with merge height $d$ mapped to time $\tfrac12\log(d/\delta)$. In one dimension no inversions occur, because the merged centroid lies strictly between its parts. Centroid sorting is one of the conventional strategies unified by Lance and Williams (1967), "A general theory of classificatory sorting strategies. I. Hierarchical systems", Computer Journal 9(4):373-380.

Open-law: the limiting field $\dot u_i=-u_i+\sum_j(w_{ij}+w_{ji})(u_j-u_i)$ with row-normalized Gaussian weights is a continuous-time Gaussian blurring mean-shift, or self-attention, particle system. Carreira-Perpiñán (ICML 2006, Gaussian blurring mean-shift) studies such systems, with a stopping rule applied once cluster structure appears, analogous to the threshold here. So do Geshkovski, Letrouit, Polyanskiy and Rigollet (NeurIPS 2023, "The emergence of clusters in self-attention dynamics"), where cluster locations are determined by the initial tokens. Influence normalized by relative distance is the model of Motsch and Tadmor (J. Stat. Phys. 2011). None of these is in the 25-entry bibliography. The chapters' own contribution is the inheritance of laws and responses, which these works do not supply.
**Fix:** Cite these works and add one sentence to each chapter identifying the selection schedule with the corresponding clustering procedure.
**Falsifies:** A configuration satisfying form:gaps on which the threshold schedule differs from centroid linkage, or a structural feature showing that open:limit is not of mean-shift/attention form.

### X2-FRAME-08: Result labels, file order and notation obscure what is proved
**Category:** exposition
**Severity:** low
**Location:** preamble.tex:29; chapters/05_inheritance.tex:145-168 (inh:role-loss), 719-748 (inh:independence-discontinuity); chapters/06_fusion.tex:415-422; chapters/11_conditional_formation.tex:277-290 vs chapters/12_open_formation.tex:86-94; MAgent_Theory_2.0.tex:16-30; appendices/notation.tex; companion/c07_fusion.tex:2
**Evidence:**

- Labeling. `\constructionheading` prints "Definition". As a result, inh:role-loss ("No autonomous update of that retained law can reproduce both trajectories") and inh:independence-discontinuity ("$\pi_*(\mathbb P_\epsilon)=\{V\}$ for every $\epsilon>0$") are claims with proofs typeset as Definitions. The open problem fus:smooth-open contains a proved general no-finite-entry lemma that duplicates inh:smooth-no-finite-entry without citing it.
- Constitutive choices. frontmatter.tex:12 promises a separate category of "constitutive choices", but no heading type exists for it. The potential (def:dyn-potential) and the open-law selector are Definitions, while the analogous conditional-formation selector is an unlabeled paragraph.
- Unused macros and prefixes. The remark and hypothesis macros are never used. The prefix `open:` names both the open-law results and the appendix open problem open:conn-derived.
- File order. File names disagree with compiled numbering: two `02_` files; `06_fusion` is Chapter 7 and `06_outlook` is Chapter 14. The nonsmooth fusion chapter sits between the general theory and the smooth constructions, although its interaction is abandoned afterwards.
- Notation collisions within one chapter. In 05_inheritance, $\rho$ is the separator reference law, the record readout and the transmission coefficient (lines 228, 317, 419). In 11_conditional, $q_\pm$ are eigenvalues of $\mathsf Q$ next to the parent laws $q_I$. Across chapters, $\kappa$ means $a-c$, a conditional law, $\kappa_N$, the coupling, a von Mises concentration and $\kappa_{\det}$. The notation appendix gives $B$ three meanings, and $\gamma$ is also a metric in determinant_example.
- Companion wording. The companion title "A Nonsmooth Route to Recurrent Formation", for a two-stage consensus example, uses a term that AUTHORING.md:19 restricts.
**Fix:** Add a `\constitutiveheading` macro. Relabel the two claim-bearing Definitions as Propositions. Cite inh:smooth-no-finite-entry in fus:smooth-open. Rename the files to their compiled numbers, and move the fusion chapter after the open-law chapter or into an appendix. Rename the colliding symbols. Retitle c07 "A Nonsmooth Two-Stage Fusion".
**Falsifies:** The macro definition at preamble.tex:29 settles the labeling point; the other points are settled by inspection.

### X2-FRAME-09: A strengthened, falsifiable formation definition, with evidence that the open-law architecture can meet it
**Category:** extension
**Severity:** high
**Location:** chapters/05_inheritance.tex:776; chapters/06_outlook.tex:16; chapters/12_open_formation.tex
**Evidence:** Sketch. For $\lambda\in[0,1]$, let $\Phi^\lambda$ be the unit-Fisher flow of the original potential with every nonself peer cost multiplied by $\lambda$ (and, for moment channels, with the channel's dependence on the population replaced by its $\lambda=0$ value). Let $H^\lambda(x)$ be the membership history (partitions and event times) that the declared selector produces from initial data $x$. Two candidate clauses:

- Interaction necessity (weak): $H^1\ne H^0$ on a nonempty open subset of the admitted class.
- Interaction necessity (strong): the final partition under $\Phi^1$ is not of the form $\Psi(T(x))$ for any statistic $T$ whose evolution is the same under $\Phi^0$ and $\Phi^1$.

An amplification clause answers pifb2/24_section.tex:10: for $C(t)=\max_{I\in\pi}\operatorname{diam}I/\min_{I\ne J}\operatorname{dist}(I,J)$, require $C_{\Phi^1}(t_k)\le\theta C_{\Phi^0}(t_k)$ for a fixed $\theta<1$ at each event.

Classification of the current constructions:

- Nonsmooth fusion passes: at $\lambda=0$, $u_i=u_i(0)e^{-\kappa t}$ never collide.
- Conditional formation fails every clause (X2-FRAME-01).
- Open-law, in the proof regime, fails both necessity clauses and passes amplification: $b/a$ is 5.9e-5 against 3.1e-2.

The weak clause is attainable in the open-law architecture away from the proof regime. In `ch12_random_search.py` (sorted initial positions uniform on (-25,25)^5, admitted only if every initial off-diagonal weight is below 1/10), 2 of 6 admitted trials differ. For example, u0 = -24.81, -17.28, -14.97, -6.52, 16.5 gives, under the full field, {1,2} at 0.074, then {0,1,2} at 1.524, {0,1,2,3} at 1.798 and all agents at 2.92; the null field joins only {1,2}, at 0.096, before T=5. The mechanism is that under the null flow the rows flatten toward $1/N$ as $e^{-t}u$ shrinks, so contrasts fail, while peer contraction keeps formed blocks tight.

In `ch12_tie_breaking.py`, an equally spaced core -6, -2, 2, 6 gives outer pairs and then their union under the full field, and nothing under the null field. This is a codimension-one tie: a middle-gap perturbation of 0.1 destroys it, and perturbations of 0.2 or more give the middle pair in both. These are exploratory runs of the limit field, not proofs.

Falsifiable predictions for a finite simulation (VFE_4.0 or any faithful integrator of the smooth flow):

- P1, inh:smooth-no-finite-entry. Distinct agents never merge under the smooth exchange-equivariant flow; by backward Grönwall their separation is at least $\|\Delta(0)\|e^{-Lt}$, where $L$ is the Lipschitz constant of the field. A merge that persists as the step size goes to 0 refutes the implementation's fidelity to the theory.
- P2, inh:smooth-mixture-failure. The intertwining residual of a mixture parent equals $-Jca^2$ in the binary model.
- P3, prop:public-gibbs-identifies and the two-source inverse at 12_open_formation:426-442. Unrestricted identity-channel Gibbs probes recover the unordered child laws.
- P4. Conditional-formation event times are $\tfrac12\log(d/\delta)$ independent of $\kappa$.
- P5. In the open-law architecture, peer forces leave membership unchanged in the scale-separated regime and change it at moderate separation.
**Fix:** Adopt the weak necessity clause and the amplification clause in inh:recursive-formation, keep the strong clause as the stated target, and record P1 to P5 as the simulation-facing predictions of the theory.
**Falsifies:** A construction in the manuscript whose peer terms cannot be scaled by $\lambda$ without leaving the admitted domain, which would make $H^\lambda$ ill-defined; or proof-level evidence that the random-search differences are discretization artifacts. I used RK4 with dt=2e-3 and did not halve the step to check convergence.

### X2-FRAME-10: Closure under additive retention for every partition fixes the membership order; a vector-valued escape route
**Category:** extension
**Severity:** high
**Location:** chapters/11_conditional_formation.tex:111-127, 206-226, 277-348
**Evidence:** Proposition sketch. Let scalar parameters $\theta_i$ in an open interval evolve by $\dot\theta_i=G(\theta_i,g(t))$, where $g$ is a label-independent global state (in the conditional-formation construction, the common precision and total moment). Suppose sum retention of every two-element block intertwines with the flow, so that $G(x,g)+G(y,g)$ depends on $x$ and $y$ only through $x+y$. By Jensen's functional equation and continuity, $G(\theta,g)=a(g)\theta+c(g)$. Then $\theta_i-\theta_j$ is multiplied by the common factor $\exp\int_0^ta(g)$, and so is every centroid difference of disjoint blocks. Any selector that joins blocks when the centroid difference first reaches a fixed $\delta$ from above therefore produces the centroid-linkage partition sequence of the initial data, in the same order, for every interaction. Only the event times can depend on the interaction, through $a(g)$. The conditional-formation construction is the instance $a=-2$, $c=2\alpha_0$. The proposition explains X2-FRAME-01 structurally: there, additive closure for all partitions and interaction-dependent membership are incompatible.

Escape route. Take vector parameters $\theta_i\in\mathbb R^m$ with $\dot\theta_i=A(g)\theta_i+c(g)$, where $A(g)\in\mathfrak{gl}(m)$ depends on the population and is not a multiple of the identity. Differences then evolve by a common anisotropic propagator $U(t)=\mathcal T\exp\int A(g)$, and their ordering in a fixed norm can change. For example, with $A=\operatorname{diag}(-1,-3)$, initial differences (1, 0) and (0, 1.5) become (0.37, 0) and (0, 0.075) at t=1, which reverses which pair is closer. Membership can then depend on the interaction through $A(g)$ while additive closure survives. A candidate is a coupled-sector matrix-law family in which the population moment mixes the belief and model shape directions in a sum-compatible way. Difficulty: the proposition is elementary (an hour); constructing a smooth-KL instance with anisotropic, interaction-dependent $A(g)$ is moderate and open.
**Fix:** Add the proposition to 05_inheritance as a no-go for scalar additive architectures, and cite it in the conditional-formation chapter's scope paragraph.
**Falsifies:** A sum-closed architecture with scalar per-agent parameters whose shape law is not affine in the agent's own parameter, which would mean the Jensen step fails, for example because closure holds only for the selected partitions rather than for every two-element block.

## Checked without defect

| Label or claim | Method | Result |
| :-- | :-- | :-- |
| prop:kl-coordinate, prop:int-kl-covariance, prop:record-law | reasoning on pushforward and RN derivative | correct |
| prop:int-endpoint, prop:int-rank, prop:int-moment-equivariance | hand algebra ($\mu'=h^{-1}\mu$, $M'^{-1}=h^TM^{-1}h$) | correct |
| eq:int-nongaussian-loop | hand: $(I+E_{12})(I+E_{21})$ | [[2,1],[1,1]], correct |
| geo:joint-fisher and its binary example | hand: $\sum 1/(4(1+\rho km))$ | $1/(1-\rho^2)$, correct |
| geo:moving-channel, Bernoulli example | total covariance; $p(1-p)$ | correct |
| geo:quotient, nonintegrable example | bracket of $\partial_x$ and $\partial_y+x\partial_z$ | $\partial_z$, not null; correct |
| geo:retained-connection, contraction counterexample | minimize $\tfrac12(1+a)^2+(-1+a)^2$ | $a=1/3$, correct |
| prop:dyn-gibbs, eq:dyn-envelope, thm:dyn-finite-flow dissipation | reasoning | correct |
| prop:dyn-jensen | hand chain of logs | correct |
| thm:predictive-chain, prop:conditional-lift, prop:common-family-recovery, cor:local-family-recovery, thm:predictive-distortion | reasoning | correct |
| inh:deterministic-descent, inh:gradient-inheritance, inh:finite-lumpability, inh:composition, inh:causal-memory | reasoning | correct |
| inh:role-loss numbers | hand | (0.16, 0.68, 0.16); -1.2, -1.8; correct |
| inh:smooth-mixture-failure | sympy, full energy including peer terms | residuals 0; values 1/8 and 3/32 |
| inh:smooth-boundary-failure | sympy | residual 0; 20/256 and 17/256 |
| inh:finest-independent, inh:conditional-partition, inh:independence-discontinuity, inh:selector-symmetry | reasoning (common refinement, odds ratio $e^{4J_e}$, transposition) | correct |
| fus:local-inheritance, fus:predictive-law, fus:evaluator-square, fus:distance | hand Gaussian KL and variance bookkeeping | correct |
| fus:uniqueness, fus:sticking, fus:two-stages, fus:open-gap-condition | hand plus numeric check of $t_1,t_2$ | correct |
| smooth:spectrum-solution, smooth:entropy-rate, smooth:swapped-spectrum, smooth:row-ratio | hand | correct |
| eq:public-phase-binary-expansion, eq:public-phase-binary-parent | sympy series to order 4 | residual 0 |
| eq:boundary-response-derivatives, eq:boundary-observability-det (prop:boundary-observable) | sympy series and block determinant | residual 0 |
| form:full-flow, form:ode (shape and precision laws) | natural gradient on the Gamma analog | residual at most 1.1e-7; $\kappa$-free shape law |
| form:selection, form:first-times, form:second-time; example times | hand plus numeric | correct |
| form:curvature, open:curvature | hand differentiation of $-k_1/(k_1+k_2)$ | correct |
| open:two-metrics | Lagrange minimization | correct |
| prop:conn-curvature example, geo:determinant-metric quadratic form | hand | correct |
| sinh:filter cycle example, sinh:linear-descent, prop:sup-row-equivalence, lem:sgeom-lowrank, prop:sgeom-rankone | hand | correct |
| Cross-references, all three documents | `xref_check.py` | no undefined references; forward references only at chapter level |
| AUTHORING requirement: main readable without the supplement | grep and xref | two textual pointers, 05_inheritance:276 (kernel-inclusion criterion, sinh:linear-descent) and :569 (finite filter, sinh:filter), plus frontmatter:10; neither is used by a later main result; no cross-document cref |
| Companion self-contained | xref | only cf:/comp: labels; no main or supplement references |
| Abstract, README and AUTHORING claims against results | claim-by-claim reading | every claim matches a stated result and its hypotheses; the attribution issues are in X2-FRAME-01, 02 and 06 |

Not checked: thm:public-phase (Banach decay and regularity estimates), geo:fisher-loss (the uniform-integrability step), the erase:gate-entry and erase:pairs constants, the quantitative estimates of open:routing and open:open-laws, the invariance argument of thm:boundary-tree-flow, and smooth:feedback.

## Extensions and improvements

**One precise open problem to replace the scattered statements.** Fix homogeneous constitutive data for the Gibbs-reduced potential of def:dyn-potential, chosen before the initial configuration, and a relabeling-equivariant selector that reads only retained states and accessible histories. Let $\Phi^\lambda$ be the unit-Fisher flow with nonself peer costs scaled by $\lambda\in[0,1]$, and $H^\lambda(x)$ the resulting membership history. Find a nonempty open set $U$ of initial laws such that, for every $x\in U$, three conditions hold:

- (a) Under $\Phi^1$, a nontrivial partition $\pi_1$ is selected at a finite time $t_1$, and its retention satisfies inh:vector-square and the boundary square of 05_inheritance:217 for an interface declared before $t_1$ that includes at least one record-feedback protocol.
- (b) The inherited flow selects, at some $t_2>t_1$, a block containing at least two first-stage parents, with the same constitutive data.
- (c) $H^1(x)\ne H^0(x)$.

An optional clause (a') separates aggregation from symmetry reduction: the retention is not a quotient by a subgroup of a product of single-agent symmetry groups. Then determine whether $U$ can have positive mass under a declared initialization law, and whether (a) to (c) iterate to every finite depth. This statement is decidable for each construction in the manuscript. The nonsmooth fusion chapter meets (a) to (c) apart from smoothness and feedback. The conditional-formation chapter fails (c). The open-law chapter, in its proof regime, fails (c) and, with (a'), fails (a').

**Two scheme theorems that would make the general chapters do work.**

- (i) Symmetry-quotient scheme. Let compact groups $K_i$ act on the agents so that the potential, the metric and the declared interfaces are invariant under $\prod_iK_i$. If the selector reads only invariant statistics, then for every partition $\pi$ and every closed subgroup $\mathsf H_I\subset\prod_{i\in I}K_i$, the quotient retention intertwines the flow (by equivariance and uniqueness), and the interfaces invariant under $\mathsf H_I$ descend. The open-law construction is the instance $K_i=S^1$ with $\mathsf H_I=\{\sum a_i=0\}$; the boundary-closure tree automorphisms are a non-product variant.
- (ii) Convolution-closed scheme. Let the laws lie in a family with additive parameter under convolution, and let the interactions read only total moments. Then sum retention intertwines for every partition exactly when the per-agent parameter law is affine with an agent-independent coefficient. This is the proposition in X2-FRAME-10.

Stating these in 05_inheritance turns the boundary-closure, conditional-formation and open-law constructions into corollaries. It also makes it visible that selection does no inheritance work in either scheme.

**A severity protocol for the implementation.** Each existence claim should ship with its null comparison: the same initial data, the same selector, and $\lambda=0$. The membership histories $H^1$ and $H^0$ and the contrast ratio $C(t)$ should be reported side by side. For VFE_4.0 this is cheap, because the selector is a readout. Together with predictions P1 to P5 of X2-FRAME-09, this gives the theory refutable consequences for a finite simulation, which the manuscript currently does not state.

**Structural reorganization.** Part I (compiled chapters 1 to 6 and appendices A to C) is a coherent framework of exact identities. Part II (compiled chapters 7 to 13) is a set of independent existence constructions. Make that split explicit. Move the nonsmooth fusion chapter to the end of Part II, or into an appendix, since its interaction is not the theory's. Open Part II with the strengthened formation definition and the two scheme theorems, so that each construction is read as a test against a stated null rather than as a further example.

Scripts (all single-threaded, each under 20 s): `scripts/X2-FRAME/xref_check.py`, `general_theory_usage.py`, `ch11_kappa_independence.py`, `ch12_null_model_events.py`, `ch12_amplification.py`, `ch12_random_search.py`, `ch12_moderate_A.py`, `ch12_tie_breaking.py`, `result_spotchecks.py`.
