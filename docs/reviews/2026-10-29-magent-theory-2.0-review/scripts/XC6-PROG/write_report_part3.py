"""Writer for lens XC6-PROG report, part 3 (tables, program maps, extensions, not checked)."""
import pathlib

OUT = pathlib.Path(r"C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-29-magent-theory-2.0-review/lenses/XC6-PROG.md")

TEXT = r"""
## Checked without defect

| Label | What was checked | Method | Residual or reasoning |
| --- | --- | --- | --- |
| lem:measurable-bundle | Borel measurability of $(g,\zeta)\mapsto g_\#\zeta$; atlas-independent standard Borel structure | reconstruction (monotone class; countable disjointified atlas; Borel gluing maps) | correct |
| prop:bundle-morphism | well-definedness, affinity, KL contraction; content morphisms through intertwiners | reconstruction | correct |
| prop:kl-coordinate | invariance under injective Borel maps including the value $+\infty$ | Lusin-Souslin reconstruction | correct |
| prop:record-law, prop:record-covariance | normalization; $\gamma^{-1}B_A=B_{\tau_i(\gamma)^{-1}A}$ | reconstruction | correct |
| def:generative-model, eq:agent-free-energy | normalization; record-free minimizer is $r_iG_i$ | reconstruction | correct |
| prop:content-law | projective consistency; projected free energy by the chain rule | reconstruction | correct |
| prop:loop-blindness | (i) trivial; (ii) graph of a Borel map is Borel, so equal shared-sample laws force $\nu\{H\ne H'\}=0$ | reconstruction | correct (bijectivity not needed for (ii)) |
| correlation examples (`01:230-232`) | hard agreement: product optimum is a point mass at relative entropy $\log2$; soft record: $m=\tanh(Jm)$ | hand computation | correct for $J>0$ (the sign is implicit) |
| companion `c01_objects.tex:21` | model mixture $0.6\cdot0.9+0.4\cdot0.2=0.62$ | arithmetic | correct |
| eq:dyn-local-split, rem:dyn-product | split; $\partial_t\operatorname{LOR}=-(\operatorname{LOR}(\zeta)-\operatorname{LOR}(R))$ | hand computation (additive terms cancel because $\sum_{k,m}km=0$) | correct |
| logarithmic pool (`03:54`) | receiver-only stationarity gives the stated log pool | hand computation | correct |
| prop:dyn-gibbs, eq:dyn-envelope | row gap identity, uniqueness, envelope | reconstruction | correct |
| rem:dyn-self-sources | $\pi_{ii}\le Z_i\le1$; $\dot\pi_i=T_q(\beta_i-\pi_i)$; replicator form; $\dot\pi_{ii}\ge0$ | hand computation | correct |
| eq:dyn-kl-differential, eq:dyn-incoming | variation formulas; incoming term averages to $-\beta$ | hand computation; s2(a) force finite differences | error $2.0\times10^{-9}$ |
| prop:dyn-compensation | identity; Gaussian source-variance force $(v_j-v_i-(m_i-m_j)^2)/(2v_j^2)$ | s2(f); hand computation | residual $2.2\times10^{-16}$ |
| thm:dyn-finite-flow | Shahshahani gradient identification; dissipation | reconstruction; s2(a) | $\mathcal E$ monotone, 11.197 to 1.748 |
| prop:dyn-global-flow (remediation-added) | every inequality of the floor argument; compactness; Lojasiewicz convergence for a gradient flow in a metric uniformly equivalent to the Euclidean one | reconstruction; s2(a) with $c_{\min}=4.7\times10^{-4}$; s2(a') from $m(0)=-20<-C=-12.50$ | floor respected; $m$ nondecreasing while below $-C$, $m(10)=-3.55$; dissipated variance $3.5\times10^{-7}$ at $t=40$ |
| prop:dyn-row-relaxation | $h_j=Tl_j+T(1-\log Z)$ | hand computation | correct |
| prop:dyn-morse | Schur complement of the positive row block; Haynsworth inertia additivity | reconstruction | correct |
| prop:dyn-gaussian-noninvariance | $e^Q$ with nonconstant quadratic $Q$ is not a polynomial | reconstruction | correct |
| prop:dyn-chart-flow | Picard-Lindelof in $L^\infty_0$; continuation alternative; Lipschitz bounds for fixed kernels with two-sided density bounds | reconstruction | correct |
| prop:dyn-global-local | coordinate-ascent identity | reconstruction | correct |
| prop:dyn-jensen, prop:dyn-label-model | identities | s2(e), random finite alphabets | residual $0.0$ |
| eq:dyn-affine-obstruction | strict convexity of $-\log$ | reconstruction | correct |
| prop:dyn-per-agent | per-agent free energy and unit-temperature Gibbs rows | reconstruction (label terms, conditioning on $L$, on $J$ and $m$) | correct |
| prop:dyn-nonreciprocal | circulant spectrum; no Riemannian gradient structure | s2(b) | error $6.7\times10^{-16}$ |
| prop:dyn-no-global-vfe | (a) boundedness argument; (b) mixed derivative | reconstruction; s2(c) | correct as stated; scope in XC6-PROG-02 |
| thm:dyn-registered | (i) to (iii) by exact multinomial enumeration at $n=6$ with $q=(1/2,1/2)$, $s=(1/3,2/3)$; (iv) by reconstruction | s2(d) | residuals at most $2.2\times10^{-16}$ |
| cor:dyn-frozen-source | chain rule and additivity over copies | reconstruction | correct |
| inh:deterministic-descent | fiber test; analytic graph implies Borel (Kechris, Thm. 14.12) | reconstruction | correct |
| inh:projectable (remediation-added) | local criterion, constancy on connected fibers, $C^k$ field from local sections | reconstruction | correct |
| inh:gradient-inheritance and its two examples | horizontal criterion; center of mass; fiber minimization counterexample | reconstruction | correct |
| inh:equivariant-descent | (i) and (ii) | reconstruction | correct |
| inh:coordinate-projection | additivity from vanishing mixed derivative | reconstruction | correct |
| inh:no-underdamped | spectrum of the parent contained in the real fine spectrum | reconstruction | correct |
| inh:role-loss | retained laws $(0.16,0.68,0.16)$; derivatives $-1.2$, $-1.8$ | s4(b) | exact |
| inh:finite-lumpability, inh:intertwining and its example | criterion; Rogers-Pitman; $\Lambda T=T^c\Lambda$; total-variation gap from $\delta_1$ | s4(a) | residual $5.6\times10^{-17}$; TV $0.100000$ for five and six retained states; $1.1\times10^{-16}$ from $\nu\Lambda$ |
| inh:canonical-parent, inh:canonical-lumping | shift parent; coarsest; partition refinement | reconstruction | correct (measurability note XC6-PROG-20) |
| inh:separator-gluing and its two counterexamples | Tonelli; countable $\pi$-systems; XOR triple; four-cycle edge laws | reconstruction | correct |
| inh:composition | pasting of squares | reconstruction | correct |
| finite two-level construction (`05:745-821`) | both reductions lumpable; parent kernels as displayed | s4(f), exhaustive enumeration | errors $5.6\times10^{-17}$ |
| inh:smooth-mixture-failure | $\dot{\bar x}(A)=1/8$, $\dot{\bar x}(B)=3/32$; $\dot{\bar y}$ equal | s4(c), finite differences of the full energy | exact to $10^{-9}$ |
| inh:smooth-boundary-failure | $20/256$ and $17/256$; difference $\tfrac34ma^2$ | s4(d) | exact |
| inh:smooth-no-finite-entry | fixed-point subspace and backward uniqueness | reconstruction | correct |
| inh:causal-memory | two-term decomposition by disintegration along $(M_t,Z_t)$ | reconstruction | correct |
| inh:finest-independent, inh:conditional-partition, inh:independence-discontinuity, inh:selector-symmetry | common refinement; odds ratio $e^{4J_e}$; transposition argument; dihedral block systems | reconstruction | correct |
| covariance formula and anisotropic example (`05:1274-1277`, `:1320-1325`) | $\operatorname{Cov}=\tfrac12(e^{-2t}-e^{-2(1+2\lambda)t})$; differences $(0.3679,0)$ and $(0,0.0747)$ | s4(g) | exact |
| inh:scalar-additive | Jensen's functional equation; centroid ordering | reconstruction | correct |
| inh:common-clock | coefficient $-2$ along the full flow and $-1$ along the null | s4(e), finite tables with an additive-moment channel | spread over $k$ of the residual $1.1\times10^{-15}$ and $6.7\times10^{-16}$ |
| pa:persistent-grouping (limit field only) | $u_*=(15/7,15/7,5/7)$; spectrum $-1/5,-7/5,-7/5$; null keeps $\operatorname{sign}l_i=o_i$ under both null conventions | hand computation | consistent; the full proof was not rechecked |
| companion `c05_inheritance.tex` | role-loss, two-level and center-of-mass examples against the main text | comparison | consistent |

## Prior-review findings in scope

| Prior ID | Status | Evidence (file:line) |
| --- | --- | --- |
| 2026-10-22 C1 | resolved | thm:dyn-registered, `03_dynamics.tex:389-403`, restrictions stated, general-space limit stated open at `:403`; recomputed in s2(d). Its stated consequence that a sample-level identification is impossible is false on atomless spaces (XC6-PROG-02) |
| 2026-10-22 H1 | resolved | three objectives, per-agent and global free energies, def:dyn-evidence-potential (`03_dynamics.tex:236-438`); abstract `frontmatter.tex:5` |
| 2026-10-22 H2 | partially resolved | (F1) to (F6) at `05_inheritance.tex:1382-1392`; topology, quantifier, probe class, counterfactual class and null convention remain open (XC6-PROG-04, -05) |
| 2026-10-22 E1 | resolved | inh:common-clock, inh:scalar-additive, strong and amplifying grades (`05_inheritance.tex:1287-1392`) |
| 2026-10-22 H6 | unresolved for the formation chapters | `02_geometry.tex:66`, `06_outlook.tex:16` (XC6-PROG-12); dynamical roles exist only in chapters 15 to 19 |
| 2026-10-22 X3-PIFB-01 | resolved as proposed; scope overstated | prop:dyn-no-global-vfe, abstract, outlook; see XC6-PROG-02 |
| 2026-10-22 X3-PIFB-02 | resolved | thm:dyn-registered |
| 2026-10-22 X3-PIFB-03 | resolved | `11_conditional_formation.tex:6` (form:kappa-independence), `frontmatter.tex:9` |
| 2026-10-22 X3-PIFB-04 | partially resolved | open:null-control `12_open_formation.tex:248-249`, open:necessity-open `:531`; asserted as fact elsewhere (XC6-PROG-18) and convention-dependent (XC6-PROG-04) |
| 2026-10-22 X3-PIFB-05 | resolved | prop:dyn-compensation `03_dynamics.tex:136-148`; classical attribution missing (XC6-PROG-14) |
| 2026-10-22 X3-PIFB-06 | partially resolved | inh:no-underdamped `05_inheritance.tex:318-340`, prop:dyn-nonreciprocal `03:359-370`; the completely monotone memory kernel was not added |
| 2026-10-22 X3-PIFB-07 | resolved as a remark | rem:dyn-self-sources `03_dynamics.tex:100-106`; the matching theorem is posed as open |
| 2026-10-22 X3-PIFB-08 | partially resolved | tautology lemma `05_inheritance.tex:1117-1126`; no downward construction (XC6-PROG-06) |
| 2026-10-22 X3-PIFB-09 | resolved (not re-derived) | `appendices/determinant_example.tex:43`, `:57` |
| 2026-10-22 X3-PIFB-10 | partially resolved | `03_dynamics.tex:387`, `:403`; stochastic version stated open |
| 2026-10-22 X3-PIFB-11 | not applicable | concerns the PIFB2 text |
| 2026-10-22 X3-PIFB-12 | partially resolved (not checked in depth) | `02_geometry.tex:254` mentions the labeled-mixture term |
| 2026-10-22 X3-PIFB-13 | resolved | no "mass" in `companion/c07_fusion.tex` (grep) |
| 2026-10-22 X3-PIFB-14 | partially resolved | `appendices/record_process.tex:31` (registration budget), `:57` (Dirichlet example); decision functionals not checked |
| 2026-10-22 X3-PIFB-15 | partially resolved | chapters 17 and 18: receiver-owned dictionaries with a normalized anchored loop prior in the Gaussian tangent model; episode-history action absent |
| 2026-10-22 X4-PRIOR-01 | partially resolved | (F6a)/(F6b) at `05_inheritance.tex:1390-1392`; handoff class undefined and no efficacy clause (XC6-PROG-05, -06) |
| 2026-10-22 X4-PRIOR-02 | resolved | `02_interactions.tex:38` ("The primitive self channel ... is constitutive data"); `AUTHORING.md:25` |
| 2026-10-22 X4-PRIOR-03 | resolved (grep only) | `references.bib` has Geshkovski et al., Cheng 1995, HegselmannKrause2002, FristonEtAl2021, PalaciosEtAl2020 |
| 2026-10-22 X4-PRIOR-04 | resolved | `05_inheritance.tex:1268-1279` |
| 2026-10-22 X4-PRIOR-05 | resolved for chapter 11 | `11_conditional_formation.tex:6` (form:geometry-decay); chapter 12 not checked |
| 2026-10-22 X4-PRIOR-06 | resolved in chapter 5 | inh:equivariant-descent (ii), `05_inheritance.tex:257-292`; its use in chapter 12 not checked |
| 2026-10-22 X4-PRIOR-07 | resolved | prop:dyn-label-model `03_dynamics.tex:288-304`; s2(e) |
| 2026-10-22 X4-PRIOR-08 | resolved as a remark | rem:dyn-self-sources |
| 2026-10-22 X4-PRIOR-09 | partially resolved | loop prior in chapters 17 to 19 (tangent model); downward action absent |
| 2026-10-22 X4-PRIOR-10 | resolved | chapter 3 retitled; `AUTHORING.md:39` |
| 2026-10-22 X4-PRIOR-11 | resolved (grep only) | chapter 4 cites HalmosSavage and Skibinsky |
| 2026-10-22 X4-PRIOR-12 | resolved | open:global-anchor `12_open_formation.tex:359-360` |
| 2026-10-22 X4-PRIOR-13 | partially resolved | witnesses at `02_geometry.tex:382`; the Gaussian soldering example and the isotropy criterion were not found |
| 2026-10-22 X4-PRIOR-14 | partially resolved | open:constants `12_open_formation.tex:251-252`; the temperature is not quantified |

## Program maps (focus questions 1 to 5)

**Question 1: PIFB2's central claims and open obligations in 2.0.** Status is supported, partially supported, absent, or contradicted.

| PIFB2 claim or obligation (file) | Status in 2.0 | Location | Note |
| --- | --- | --- | --- |
| Agents as local sections of an associated statistical bundle over a continuous base (`01:6`, `04`) | supported | def:agent-section, lem:measurable-bundle | measurable form added |
| Selective rank-limited channels, endpoint covariance (`02` table, `04`) | supported | def:int-channel, prop:int-kl-covariance; it1:code | code and dictionary distinguished |
| One normalized population model as the foundation; collective VFE (`01:10`, `05:4`, `05:223`) | partially supported; the foundation choice is contradicted | sec:dyn-likelihood, prop:dyn-global-local, def:content-law; formation uses def:dyn-potential | XC6-PROG-01, -03 |
| Attention as source inference (`05:107-144`) | supported with scope | prop:dyn-label-model, thm:dyn-registered, prop:dyn-per-agent | exact only with source-independent Jensen gaps, at $T=1/n$, or at unit temperature per agent |
| Separate consensus potential (`05:146-182`) | supported and promoted | def:dyn-potential, prop:dyn-no-global-vfe | obstruction scope XC6-PROG-02 |
| One joint fiber for state and model; observation ledger (`05:184-191`) | supported | eq:agent-disintegration, geo:joint-fisher, def:generative-model | dynamics uses the product restriction (rem:dyn-product) |
| Latent Lie-algebra links with a normalized loop prior and Fisher recognition flow (`05:31-72`, `05b`) | partially supported | it1:learning, chapter 18 | Gaussian tangent model only |
| Episode-history Proca-type link equation (`05b`, `24:19`) | absent | none | |
| Cross-scale inference: cavity priors and prediction between episodes (`10`) | partially supported | prop:conditional-lift, prop:common-family-recovery, conditional and interacting memory appendices | no parent-to-child construction (`05:1117-1126`) |
| Participatory loop: parents shape child inference (`12:4-7`, `10:58`) | absent from the main text | `05:1117-1126`; Item 7 (not integrated) | XC6-PROG-06 |
| Participation as an operational decision (`12:30-52`) | absent | none | XC6-PROG-08, -11 |
| Exact coarse-graining and memory (`09`, `11`) | supported and extended | inh:deterministic-descent to inh:causal-memory | effective action and cumulant closure (`11`) absent |
| Coarse VFE drop is no formation criterion (`09:21-30`) | supported in substance | inh:canonical-parent (the protocol decides) | |
| Gaussian moment-projection parent (`09:47-83`) | contradicted as an autonomous parent | inh:smooth-mixture-failure, inh:smooth-boundary-failure | |
| Coherence, persistence and autonomy are distinct (`09:99-103`) | supported | (F2), (F3) | |
| Four-cycle selection with learned source priors (`13:6-29`) | absent | rem:dyn-self-sources poses the law-level version as open | XC6-PROG-15 |
| Recursive formation from the actual first parent (`13:31-36`, `24:9-10`) | partially supported | chapters 11 and 12 (two-stage selections), chapter 13 (one stage), inh:formation-open | no construction meets (F1) to (F6) |
| Emergent model variables, slow sector (`24:12-13`) | absent | unit mobilities (`03:194`); learned priors at rate $T$ (rem:dyn-self-sources) | |
| Learned selective interfaces (`24:15-16`) | partially supported | it1:identifiability, it1:learning | formation interfaces are declared |
| Dynamic links and coarse link laws (`24:18-19`) | partially supported | chapter 18; chapter 19 rigid clusters | link-sector inheritance open |
| Emergent geometry: pullback geometry, collective distinguishability (`18`, `20`) | supported as kinematics | geo:pullback, geo:collective, geo:fisher-loss; chapter 16 conditional base transport | diagnostic in formation (XC6-PROG-12); the base is supplied |
| Lorentz structure from Hermitian determinant geometry (`19`, `24:21-22`) | supported as an optional appendix | `appendices/determinant_example.tex` | soldering, scale and propagation absent |
| Emergent time: information clock, Fisher length (`17`) | partially supported | eq:dyn-dissipation only | no clock, no parent clock; time is a relaxation parameter (companion `c03:84`) |
| Persistence, metastability, mass (`08`, `29`, `24:24-25`) | absent | deterministic flows; finite-window persistence | inh:no-underdamped redirects the inertia question |
| Registration budget (`17`) | supported | `appendices/record_process.tex:31` | |
| Empirical identification (`24:30-31`) | absent by design | `AUTHORING.md:79` | |

**Question 2: are the foundations the right ones?** (a) The three objectives: one should be primary, and it should not be the reciprocal potential. The per-agent free energy, rewritten with peers' messages as records in each agent's fixed model, is the right primary dynamics, and the global free energy of a block is the right primary object for formation (XC6-PROG-03, -01, -09). (b) The six-clause standard is achievable in principle (the unintegrated Item 7 claims an instance with declared thresholds, Gaussian location only and numerical margins; I did not check it), but it is too strong in demanding exactness (XC6-PROG-10), too weak in omitting type closure, top-down efficacy and parent agency (XC6-PROG-06), and under-specified in its topology, quantifier, probe class, counterfactual class and null convention (XC6-PROG-04, -05). (c) The general-law framework is the right setting for definitions and identities, and it matters: the finite-alphabet obstruction fails on atomless spaces (XC6-PROG-02). Its dynamical theorems are on finite tables, bounded log-density charts or Gaussian families, and the chart topology silently enters (F1) (XC6-PROG-05). (d) The gauge structure is correctly formulated as kinematics and is idle in formation (XC6-PROG-12). (e) The per-agent free energy is the perceptual half of active inference with no action, policy or model learning; at fixed rows its flow is log-linear social learning (XC6-PROG-11, -14, -08).

**Question 3: continuity with the earlier MultiAgentELBO theory.**

| Earlier item (`SPEC.md`, `CONTENT.md`) | Status in 2.0 | Note |
| --- | --- | --- |
| One normalized joint and one collective ELBO as the program's dynamics (`SPEC:64-66`) | rescoped to one of three objectives, with no formation result | XC6-PROG-03 |
| A selected correlated recognition coupling in the coupling set (`CONTENT:93-98`) | dropped; dynamics uses product recognition | its absence is why part (b) of prop:dyn-no-global-vfe is the operative obstruction (XC6-PROG-02) |
| Law-point energy obstruction (`CONTENT:102-109`) | imported faithfully | eq:dyn-affine-obstruction |
| Attention ELBO and constant-row gap | imported and rescoped | prop:dyn-gibbs, prop:dyn-label-model |
| Latent Lie-algebra links, loop prior, field equation | partly imported | chapters 17 and 18, Gaussian tangent model |
| Fisher clock | dropped | |
| Two timescales and slow models | dropped by policy | unit mobility (`03:194`) |
| Strong lumpability, Borel memory projection | imported faithfully | inh:finite-lumpability, inh:causal-memory |
| Finest independent partition and selector results | imported faithfully, scoped as constraints | `05:1268-1279` |
| Multiscale VFE with random membership and a partition catalog (`SPEC:72-75`) | dropped | the route of XC6-PROG-01 |
| Downward coupling and adaptive tower VFE | dropped | XC6-PROG-06 |
| Finite-depth adaptive formation; gated stationary partition; seeded contrast formation; learned-prior matching selection (`CONTENT:824-943`) | dropped; the law-level matching question is posed as open | XC6-PROG-15 |
| Episode-lifted joint for delayed feedback | dropped | route to top-down feedback |
| Claim rules R13, R14, R15 (`SPEC:213-264`) | R14 weakened to (F6b); R13 and R15 dropped | XC6-PROG-05, -15 |
| Supplement B: base transport, Wilsonian and lattice material | dropped by policy (`AUTHORING.md:5`); base transport rebuilt conditionally in chapter 16 | |

**Question 4: the five highest-value next theorems** are XC6-PROG-01 (collective free-energy formation with a latent structure variable; critical), -07 (per-agent type closure on exponential families; high), -08 (participatory querying by expected free energy; high), -09 (one message-level model with the three objectives as limits; high) and -16 (holonomy-driven walls; medium). Each finding gives a sketch, the obstacle and a difficulty estimate.

**Question 5:** the status of every 2026-10-22 X3-PIFB and X4-PRIOR finding and of C1 is in the table above.

## Extensions and improvements

The decision that unlocks the rest is the choice of a primary objective. The reciprocal potential cannot carry the program: it has no evidence, its stationary set ignores records, and its source force exists only if sources read their receivers' laws. Its separation from a free energy is also not what the abstract says; on atomless spaces it equals the free energy of the fixed model $\bigotimes_iR_i$ at a copula coupling of the agent laws with total correlation $\mathcal E-\mathcal E_0$, so the separating features are the product recognition family and the sum of marginal Fisher metrics. The per-agent free energy is the natural agent objective once peers' communications enter as records: then each $\mathcal F_i$ is the free energy of a fixed normalized model, records persist, and the law-level per-agent free energy is the frozen-episode, large-message limit (cor:dyn-frozen-source). A meta-agent is then a block whose members' models are conditionals of one block model, which is where the global free energy enters.

The first construction to attempt in that setting is XC6-PROG-01. Put a structural latent $\Pi$ on a finite catalog of partitions, conjugate block latents and coded comparison records into one normalized model; its partition posterior is a derived selector, its cavity priors are top-down feedback that use each record once, and in the linear-Gaussian family the block posterior is a closed function of summed natural statistics. The theorem to prove is a static one first: concentration of $Q(\Pi)$ on a nontrivial partition for an open set of record configurations, exact block inheritance, and dependence of the second-level partition on the first-level block statistics. A dynamical version then replaces (F2) by posterior concentration as records accumulate, with an evidence horizon clause in the sense of the earlier R15.

Recursion needs type closure, and XC6-PROG-07 gives the cheapest route. In the per-agent picture a shared-label block is a single receiver at temperature $T/|B|$ holding its pool (Item 7, general laws). If the pooled mean parameter, the dispersion of the members' mean parameters and the record sums form a closed finite state on a declared invariant submanifold of an exponential family, the block is again a per-agent agent and induction gives arbitrary depth. Item 7's characterization limits full closure to the Gaussian location family, so the target is closure on an invariant submanifold or with a finite augmentation.

Participation is the missing physics of the program. XC6-PROG-08 makes the first action in 2.0 a choice of whom to query, scored by an expected free energy; in the shared-target architecture of inh:common-clock, where passive attention gives a common clock, receiver-dependent querying should break the clock and give interaction-necessary selection on an open set of a small Bernoulli instance. This is also the place where the per-agent free energy becomes active inference rather than its perceptual half.

Two low-cost statements tighten chapter 3. XC6-PROG-13 settles the first part of op:dyn-three-pictures negatively by the Monderer-Shapley mixed-partial test and gives the exact criterion for per-agent stationary points to be critical points of $\mathcal F_{\rm tot}$: every source must be a fixed point of the maximum-likelihood deconvolution of its receivers' pool. XC6-PROG-21 identifies the reciprocal peer cost at such points with a static Schrodinger cost. Both rest on prop:dyn-compensation, whose source term is the EM or Richardson-Lucy direction (XC6-PROG-14).

On the formation standard, the immediate repairs are definitional (XC6-PROG-04 and -05): selectors read only the matter configuration and records, the chart topology is fixed, the (F5) quantifier is explicit, the (F4) probe class is declared with $A_I$, and the (F6b) class is the set of post-handoff states reachable from histories without first-stage formation. The program-level repairs are a type-closure clause and an efficacy grade (XC6-PROG-06) and a graded inheritance by exact defect functionals (XC6-PROG-10), so that interaction-created compression can be stated with its defect computed rather than excluded by exactness.

## Not checked

The analytic proofs of the main theorems of chapters 6 to 19 (I read their openings and statements only), the appendices beyond grep, the supplement, and the working documents Items 1 to 3 and 7 (Item 7 read at lines 1-60; its claim to meet (F1) to (F6) was not checked). In PIFB2 I did not read sections 03, 04, 06 to 08, 11, 14 to 23, 25 to 37, or the selection proof `36a`, so the PIFB2 four-cycle theorem is reported from its statement. In the earlier theory I read `SPEC.md` and parts of `CONTENT.md`, not the `.tex` sources or proofs. The vault `index.md` was read by grep, and four concept pages in full. Sources cited "from memory" (Molavi et al. 2018, Richardson 1972, Lucy 1974, Vardi et al. 1985, Csiszar and Tusnady 1984, Besag 1974, Arnold and Press 1989, Heller and Ghahramani 2005, Blei et al. 2010, Leonard 2014) were not reopened; the vault copy of Csiszar and Shields 2004 is password-protected. Part (b) of prop:dyn-no-global-vfe for law-dependent channels on atomless spaces is neither proved nor refuted here. The statement in XC6-PROG-10 that interaction-created exact compression is not available on open classes is a judgment, not a theorem. The Gibbs-row case of the mixed-partial test in XC6-PROG-13 was sampled at four points; the fixed-row case was proved. Whether the graded inheritance of XC6-PROG-10 is compatible with the author's rule against an error-bound program is a question for the author.
"""

with OUT.open("a", encoding="utf-8") as fh:
    fh.write(TEXT)
print(f"appended part 3: {len(TEXT)} characters")
