# R-OPEN remediation log (chapter 12 and companion c12)

Files edited: `M/chapters/12_open_formation.tex`, `M/companion/c12_open_formation.tex` (M = `Research/manuscripts/MAgent_Theory_2.0`). Nothing else touched; nothing compiled. Scripts: `remediation/scripts/R-OPEN/s1_null_control.py` (limit field vs null limit with the selector at A = 8, 64, 4096; proof constants at A = 4096; sharpened early constants; curvature limit) and `s2_expansion.py` (sympy series of h^-2 d(hu,hv), far-agent field, m-independence of collapsed clusters). Outputs beside them. Both single-threaded, under 2 minutes.

Independent checks I made before applying: H4 reproduced (same blocks and order, pairs 0.589 vs 0.647, parent 8.214 vs 8.271 at A = 4096; final merge in both flows at 12.2 vs 12.3); G1 constants t_A = 6.93, S = 1.427, L = 8.358, R ~ 1.75e7, radius-1/2 tube needs T < 7.2e-22; early-phase sharpened constants 1 - 68e^-8 = 0.97719, 1 + 4e^-32 + 4e^-8 = 1.00134, A_min = 33.1 (hand derivation of each inequality redone); C^0 distant bound 8(e^{-1/2}+D)phi(D) rederived by hand; series h^-2 d = (u-v)^2/2 + h^2 (u-v)^2(u^2+2uv+3v^2)/12 + O(h^4) (sympy); far-agent field -u + (2/3)h^2u^3; collapsed clusters B = v/(1+v), da/dt = -a(1+4B) independent of m (sympy, residual 0); curvature limit -2 lambda^2 kappa/(kappa^2+lambda^2)^2 = -0.28402 at (1.5, 1) (numeric); 32/3 coefficient (hand); X1-GAUGE-08 exact reduction (hand derivation, including the forcing coefficient (1 - x^2)/2); X4-PRIOR-12 torsion argument (hand); X1-GAUGE-06 global non-equivariant representative is H_I-invariant (hand).

| Finding | Status | What changed (file, label) / reason |
|---|---|---|
| H4 (X4-PRIOR-01 ch12 part, X2-FRAME-02, X3-PIFB-04) | applied | ch12: new remark `open:null-control` states the null flow (binary null field, scaled limit u' = -u), what the limit-field integrations show at A = 8, 64, 4096, that they are not the exact field at fixed h and not proved, and what is proved (half-gap contraction rate 0.977 to 3.03 vs 1). Opening paragraph and theorem no longer attribute selection to interaction; theorem states it does not establish (F5). c12: new section "What switching off the interaction does". |
| H5 / L09-OPEN-01 | applied | ch12: new `open:all-partitions` (descent of flow, aggregate-response histories, collective costs for every partition and time; rows/events invariant under (S^1)^{2N}) plus the paragraph after it stating that the handoff loss is created by the response interface; theorem restated as the conjunction (a) selection, (b) interface coarsening, (c) exact inheritance, (d) geometry. c12: new section "Selection and inheritance are independent". |
| X1-GAUGE-02 | applied | ch12: "physical"/"nongauge" removed; active rotations defined in section "Orientation compression"; new definition `open:observables` (A_pop and A_P); text states the distinction is resolved only by the external response protocol, whose querier holds a frame no agent holds; Haar channel identified as the S^1 twirl (BartlettRudolphSpekkens2007). Theorem (b) worded relative to the declared algebra. Second route (internal reference agent) not taken; noted in open problem only implicitly. |
| L09-OPEN-02 | applied | ch12: new remark `open:loss-information` (per-record rotation Fisher information, linear growth, protocol choices). Theorem (b) cites it. c12 updated. |
| X2-FRAME-05 | applied | ch12: sentence after `open:all-partitions` that the selected partition is not distinguished by inheritance. Citation-graph part (general results unused) is a ch05 matter, not applied here. |
| X1-GAUGE-03 / L09-OPEN-06 ("public") | applied | One definition: public = coordinate transmitted by peer channels (ch07 convention). ch12 setup states the bit is public and the angle private with Haar private reference, and "public" is used only in this sense; "public-law distinction" removed from the theorem. c12 line 24 fixed. Defining "public" in ch05 is left to that owner. |
| L09-OPEN-03 | applied | ch12: new corollary `open:final-merge` (rows tend to 1/N, every contrast fails eventually, forced one-block merge with empty exterior, one more discarded orientation), proved for every admitted law via `open:reduction`. Opening and theorem say persistence is finite-window. c12 updated. |
| L09-OPEN-04 | applied | ch12: new remark `open:constants` (nonconstructive witness, admissibility T < ~3e-15, written tube step T < ~1e-21); theorem says the temperature is not quantified; `open:open-laws` proof now shows admissible correlations are O(h). |
| L09-OPEN-05, X2-FRAME-06, X4-PRIOR-05 | applied | ch12 geometry section: curvature present at t = 0, independent of partition, preserved because isometry holds for every partition, "supplied by the declared contextual family"; decay e^{-4t} and limits in new display `open:geometry-limits`; theorem (d) says "declared contextual family". Uniform-preservation obligation of the earlier theory not mentioned (cannot cite unpublished work). |
| L09-OPEN-06 | applied | see X1-GAUGE-03. |
| L09-OPEN-07 | out of file scope | Location is 06_outlook only. Recommendation for the lead: name the persistence/recursion dichotomy, the transient relations and final merge (`open:final-merge`), the interface-relative loss (`open:all-partitions`), and point to `open:necessity-open`. |
| L09-OPEN-08 | applied | ch12 `open:geometry-limits` and following sentence: curvature has a uniform positive bound, tensor decays; degeneration is of rank, not curvature. Old line "angular inertia tends to zero" replaced. |
| L09-OPEN-09 | partly applied | R3 stated qualitatively (per-context offsets would remove contextual phase variation) in the anchor paragraph. R8 and the ledger status concern the derivation record and outlook, not my files; deferred to lead. |
| L09-OPEN-10 | applied | ch12 two-tests: "The two scaled logits v_j = a_j/T are the roots". |
| L09-OPEN-11 | applied | ch12: chart stated in setup (positive C^r, logs bounded in C^r); `open:force` proof states local Lipschitz in C^r and sup norm; `open:open-laws` and theorem state the topology (routing open for uniform convergence of log-densities, theorem open in C^r). "Compact regular tube" phrasing removed with the old proof. |
| L09-OPEN-12 | applied | ch12: 32/3 example specifies varkappa_1(y=1|z) = 1/4, varkappa_2(y=1|z) = 3/4 for both z and counting-measure probabilities. c12: "half-distance between pair centers"; "full dynamics is equivariant". |
| L09-OPEN-13 | applied | ch12: BeynThummler2004 at `open:moving-flow`; Prony1795 at the inverse. |
| L09-OPEN-14 / X6-LIT-18 / E4 | deferred (remark added) | ch12 new remark `open:depth` with the m-independence (checked, s2) and the unchecked obstacles; no theorem claimed. Full induction is a new result. |
| L09-OPEN-15 / X5-IMPL-07 | deferred | Validated-numerics witness not done (no numerics in authoring). Listed in `open:necessity-open`; `open:constants` states the gap. Proof already works in scaled coordinates u, with the analytic series stated. |
| L09-OPEN-16 | (a) applied, (b) applied, (c) deferred, (d) open problem | (a) `open:all-partitions` and conjunctive theorem; (b) `open:loss-information`; (c) lambda > 0 coupled evaluator rerun deferred (new result); (d) angle-transmitting channel in `open:necessity-open`. |
| G1-01 | applied | ch12 `open:constants`: common-mode error (2/3)h^2u^3, h-step needs T below ~1e-21 at A = 4096, center-relative comparison named and not carried out; existence unaffected. h-step in proof now states the tube and h(R+3) < 1. |
| G1-02 | partly applied | Early-phase sharpened constants adopted in the proof (`open:early-bounds`, tau < 1.42, a > 0.239A, `open:small-gap` with exponent 0.975, A >= 34). The late-step epsilon_b = 1 interval enclosure is not certified, so the proof keeps "epsilon_b > 0 independent of A" and `open:constants` mentions the enclosure as not part of the proof. |
| G1-03 | applied | ch12 new lemma `open:persistence` (with C^1 observable closeness, bounds `open:persistence-bounds`), used for the core late step, the R-step, the h-step, initial-data perturbation, and `open:open-laws`; closeness facts stated (analyticity in (h^2,u); core-parent row difference <= 2 phi(D_R); gradients by absolute multiples of (1+D^2)phi). |
| G1-04 | applied | ch12: R-step redone in original coordinates against the decoupled reference; moving-coordinate inequality removed; Lipschitz constant stated as independent of A and R. |
| G1-05 | partly applied | C^0 constant 8(e^{-1/2}+D)phi(D) (rederived by hand) put in `open:distant-bound`; Jacobian constant 40 and Lipschitz 64.5 not printed (not independently derived), stated as absolute constants independent of A, R. |
| X1-GAUGE-01 | applied (ch12 part) | ch12 geometry section: no row, channel, record, or vector field reads the connection; A^F is a diagnostic. c12 likewise. |
| X1-GAUGE-04 | applied | ch12 and c12: common group declared as a tied representation; untied group gives flat connection and rank one; curvature belongs to the tied representation. |
| X1-GAUGE-05 | out of file scope | One-bundle vs per-agent structure is a ch01/ch02 decision; ch12 now names its tied group explicitly, which is what that finding asks of ch12. |
| X1-GAUGE-06 vs X4-PRIOR-12 (7.2) | applied, both | Decided on evidence that both are correct and concern different objects: X4's torsion argument excludes any relabeling-symmetric, rotation-equivariant anchor map (continuous or not); X1's global representative exists once relabeling symmetry is dropped and is H_I-invariant (checked by hand). New remark `open:global-anchor` states both plus S^1/Z_n holonomy. |
| X1-GAUGE-07 vs X4-PRIOR-06 (7.1) | applied, decided | Decision: X1 is right that rotation is not differentiable C^r -> C^r, so orbits are not C^1 submanifolds of the finite-r chart and a quotient Banach manifold cannot be built there; X4's orthogonal splitting fixes the linear algebra but not the smoothness of the action. New `open:quotient-gradient`: pointwise horizontality and the dissipation identity `open:quotient-dissipation` on the whole chart (orbit curves are C^1 into the sup-norm chart, where the potential is C^1); full gradient-flow identification via `inh:gradient-inheritance` on the finite-dimensional, flow-invariant von Mises product family, where the action is free, proper, isometric. Text says no quotient Banach manifold is asserted on the finite-r chart and that the moving term vanishes on that family. |
| X1-GAUGE-08 | applied | ch12 new `open:reduction` (autonomous conditional angular relaxation `open:conditional-flow`, exogenous bit forcing `open:bit-forcing`, global existence, convergence, rotation invariance); `open:open-laws` reproved through it with an explicit open condition M*eps < h*e_0. Relative-entropy-of-asymmetry interpretation not added (cite would be quantum-only). |
| X2-FRAME-04 | applied (ch12 part) | ch12 anchor section: parent retains every child law up to a torus orbit and all public bit marginals; by the ch07 criterion it does not aggregate. Qualifying ch07:244 or adding a clause to ch05 is for those owners. |
| X2-FRAME-07 | applied | ch12 after `open:limit`: Gaussian log-sum-exp gradient flow (checked by hand), mean shift and blurring mean shift, self-attention dynamics (Cheng1995, CarreiraPerpinan2006, GeshkovskiEtAl2025). Motsch-Tadmor not added. |
| X2-FRAME-08 | partly applied | ch12: conditional law renamed varkappa (was kappa, colliding with von Mises concentration). Label prefix `open:` collision with `open:conn-derived` not renamed: risk of breaking concurrent cross-references, low value. File order/macros are lead matters. |
| X2-FRAME-09 | deferred | Strengthened clauses already in ch05 (lead). Weak-necessity evidence at moderate separation mentioned without numbers in `open:necessity-open`. |
| X3-PIFB-05 | deferred | Proposition belongs in ch03 (compensation identity); not applied in ch12. |
| X3-PIFB-07 / X4-PRIOR-08 | deferred | Source-prior learning extension is a ch03 matter and a new result; ch12 priors remain fixed. |
| X3-PIFB-11 | applied (ch12 part) | `open:quotient-dissipation` gives the parent dissipation identity in the kinetic quotient metric. PIFB2 edits are outside scope. |
| X4-PRIOR-01 (ch12 part) | applied | ch12 routing proof: parent event between log(A/4)/1.0014 and log(A/4)+S, "when the supplied core separation has contracted to order one"; after-theorem paragraph states the counterfactual part of (F6) is not verified and the second stage depends on the first only through partner exclusion. |
| X4-PRIOR-02 | applied (ch12 part) | ch12 setup: selective self channel on every edge; with the identity self channel peers would carry e^{-H/T}; construction requires the selective self channel, a constitutive choice. ch02/ch03 convention is for those owners. |
| X4-PRIOR-03 | applied (ch12 part) | Clustering/attention citations as in X2-FRAME-07; Hegselmann-Krause and Markov-blanket works not added to ch12 (not about this construction). |
| X4-PRIOR-06 | applied | see X1-GAUGE-07 decision. |
| X4-PRIOR-12 | applied | see X1-GAUGE-06. |
| X4-PRIOR-13 | rejected for ch12 | Asks to import earlier unpublished results (no-balanced-presentation) with local proofs; the ch12 sentence "not one newly fitted parent KL cost" needs no import, and AUTHORING forbids citing the earlier manuscripts. Appendix parts are not my files. |
| X4-PRIOR-14 | partly applied | Explicit (A,R,h) not given (deferred, see L09-OPEN-15); threshold-invariance interval not stated (numerical only). |
| X5-IMPL-06 (ch12 part) | applied | `open:selector` now defines the event time (first closed contact with strict inequalities on a right neighborhood; transverse crossing = unique zero with positive derivative). Numerical-scheme appendix is outside my files. |
| X6-LIT-01 (ch12 part) | applied | ChenEtAl2023 cited for the explicit Fisher-Rao relaxation at `open:conditional-flow`/`open:angular-flow`. |
| X6-LIT-03 (ch12 part) | applied | ONeill1966 at `open:two-metrics` (quotient metric vs induced metric on a section). |
| X6-LIT-04 | applied | BudanurEtAl2015, RowleyEtAl2003, BeynThummler2004, FelsOlver1998 at `open:anchor`; singular set u_i = 0 stated. README/AUTHORING phrasing not mine. |
| X6-LIT-05 | applied | as X2-FRAME-07. |
| X6-LIT-12 (ch12 part) | applied | MardiaJupp2000 at the von Mises facts; Prony1795 at the binary inverse. |
| L03-DYN-03 (ch12 part) | applied | Chart and Lipschitz hypotheses stated in the setup and in `open:force`. General density-level flow is a ch03 matter. |
| L04-INH-11 | partly applied | ch12 states that inheritance comes from equivariance for every partition (`open:all-partitions`) and uses `inh:gradient-inheritance`; a general ch05 descent proposition does not exist yet, so nothing to cite. |
| Brief: reciprocal potential | applied | First sentence of ch12 and of c12. |
| Brief: retitle | applied | Chapter "Preferential Two-Stage Selection from Open Law Classes"; theorem "Open-law two-stage selection with exact inheritance"; section "Physical compression" renamed "Orientation compression"; "formation events" -> "selection events". Companion title kept (contains no "formation"). |

## BIB ENTRIES

```bibtex
@article{BartlettRudolphSpekkens2007,
  author  = {Bartlett, Stephen D. and Rudolph, Terry and Spekkens, Robert W.},
  title   = {Reference frames, superselection rules, and quantum information},
  journal = {Reviews of Modern Physics},
  volume  = {79},
  number  = {2},
  pages   = {555--609},
  year    = {2007},
  doi     = {10.1103/RevModPhys.79.555}
}
@article{BeynThummler2004,
  author  = {Beyn, Wolf-J{\"u}rgen and Th{\"u}mmler, Vera},
  title   = {Freezing solutions of equivariant evolution equations},
  journal = {SIAM Journal on Applied Dynamical Systems},
  volume  = {3},
  number  = {2},
  pages   = {85--116},
  year    = {2004},
  doi     = {10.1137/030600515}
}
@article{RowleyEtAl2003,
  author  = {Rowley, Clarence W. and Kevrekidis, Ioannis G. and Marsden, Jerrold E. and Lust, Kurt},
  title   = {Reduction and reconstruction for self-similar dynamical systems},
  journal = {Nonlinearity},
  volume  = {16},
  number  = {4},
  pages   = {1257--1275},
  year    = {2003},
  doi     = {10.1088/0951-7715/16/4/304}
}
@article{BudanurEtAl2015,
  author  = {Budanur, Nazmi Burak and Cvitanovi{\'c}, Predrag and Davidchack, Ruslan L. and Siminos, Evangelos},
  title   = {Reduction of {SO}(2) symmetry for spatially extended dynamical systems},
  journal = {Physical Review Letters},
  volume  = {114},
  pages   = {084102},
  year    = {2015},
  doi     = {10.1103/PhysRevLett.114.084102}
}
@article{FelsOlver1998,
  author  = {Fels, Mark and Olver, Peter J.},
  title   = {Moving coframes: {I}. {A} practical algorithm},
  journal = {Acta Applicandae Mathematicae},
  volume  = {51},
  number  = {2},
  pages   = {161--213},
  year    = {1998},
  doi     = {10.1023/A:1005878210297}
}
@article{ONeill1966,
  author  = {O'Neill, Barrett},
  title   = {The fundamental equations of a submersion},
  journal = {Michigan Mathematical Journal},
  volume  = {13},
  number  = {4},
  pages   = {459--469},
  year    = {1966},
  doi     = {10.1307/mmj/1028999604}
}
@article{Cheng1995,
  author  = {Cheng, Yizong},
  title   = {Mean shift, mode seeking, and clustering},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {17},
  number  = {8},
  pages   = {790--799},
  year    = {1995},
  doi     = {10.1109/34.400568}
}
@inproceedings{CarreiraPerpinan2006,
  author    = {Carreira-Perpi{\~n}{\'a}n, Miguel {\'A}.},
  title     = {Fast nonparametric clustering with {G}aussian blurring mean-shift},
  booktitle = {Proceedings of the 23rd International Conference on Machine Learning},
  pages     = {153--160},
  year      = {2006},
  doi       = {10.1145/1143844.1143864}
}
@article{GeshkovskiEtAl2025,
  author  = {Geshkovski, Borjan and Letrouit, Cyril and Polyanskiy, Yury and Rigollet, Philippe},
  title   = {A mathematical perspective on {T}ransformers},
  journal = {Bulletin of the American Mathematical Society},
  volume  = {62},
  number  = {3},
  pages   = {427--479},
  year    = {2025},
  note    = {arXiv:2312.10794}
}
@misc{ChenEtAl2023,
  author        = {Chen, Yifan and Huang, Daniel Zhengyu and Huang, Jiaoyang and Reich, Sebastian and Stuart, Andrew M.},
  title         = {Gradient flows for sampling: mean-field models, {G}aussian approximations and affine invariance},
  year          = {2023},
  eprint        = {2310.03597},
  archivePrefix = {arXiv}
}
@book{MardiaJupp2000,
  author    = {Mardia, Kanti V. and Jupp, Peter E.},
  title     = {Directional Statistics},
  publisher = {Wiley},
  address   = {Chichester},
  year      = {2000}
}
@article{Prony1795,
  author  = {de Prony, Gaspard Riche},
  title   = {Essai exp{\'e}rimental et analytique: sur les lois de la dilatabilit{\'e} des fluides {\'e}lastiques et sur celles de la force expansive de la vapeur de l'alkool, {\`a} diff{\'e}rentes temp{\'e}ratures},
  journal = {Journal de l'{\'E}cole Polytechnique},
  volume  = {1},
  number  = {22},
  pages   = {24--76},
  year    = {1795}
}
```

Verification note: bibliographic data are from my own knowledge plus the lens records (X6-LIT, L09-OPEN-13 checked Beyn-Thummler against SIAM/dblp; X6-LIT checked Budanur et al.); I did not fetch the sources in this session. DOIs for Fels-Olver and Carreira-Perpinan and the GeshkovskiEtAl2025 issue number should be confirmed by the lead before merging. Other agents may add the same keys (Cheng1995, GeshkovskiEtAl2025, ONeill1966, BeynThummler2004); deduplicate.

## NEW LABELS

Chapter 12 (all new, grep-checked unique across the three documents): `open:persistence`, `open:persistence-bounds`, `open:null-control`, `open:constants`, `open:reduction`, `open:conditional-flow`, `open:bit-forcing`, `open:final-merge`, `open:global-anchor`, `open:quotient-gradient`, `open:quotient-dissipation`, `open:observables`, `open:loss-information`, `open:geometry-limits`, `open:all-partitions`, `open:depth`, `open:necessity-open`. All previous `open:` labels retained. Companion: no new labels; `co:angular` now labels the conditional form of the same relaxation.

## AUTHOR DECISIONS NEEDED

1. Naming. The standard defines "two-stage selection with exact inheritance" as meeting (F1), (F2), (F3), (F6) but not (F4) or (F5). Chapter 12 meets (F4), does not establish (F5), and meets only the first part of (F6) (the counterfactual part is unverified). I used the term with that qualification stated explicitly after the theorem; the author may prefer a different label or an amendment to the definition.
2. Numerical statements in the manuscript. `open:null-control` and `open:constants` report limit-field integrations and computed constants, labeled as not proved, because the brief asks for the null-flow control to be stated. AUTHORING says no numerical experiments are part of authoring tasks; confirm that reporting them as unproved observations is acceptable.
3. Tied collective group. The geometry uses one S^1 for belief and model angles, now declared as a choice; with the independent group the curvature vanishes. Whether to keep the tied representation, or adopt a coupled evaluator that derives the tie (would require redoing the routing windows), is the author's call.
4. Lead follow-ups outside my files: abstract (frontmatter) still says "native preferential pairs" and "rank-two contextual geometry with nonzero curvature" without "declared family" and the interface-relative loss; outlook items of L09-OPEN-07; ch05 sentence at line 793 ("native preferential selection") is consistent but could cite `open:null-control`.
