# R-INH remediation log (chapter 5, supplement s04, companion c05)

Files edited (M = Research/manuscripts/MAgent_Theory_2.0): M/chapters/05_inheritance.tex, M/supplements/s04_inheritance.tex, M/companion/c05_inheritance.tex. The lead's section "The formation standard" is preserved except for one change made at the lead's request (coordinator message during this task): (F5) now compares partition sequences without event times. `inh:null-flow` gains the sentence defining P(x) and P^0(x) (H with event times removed), and (F5) reads "P != P^0 on a nonempty open subset of the admitted class. A difference only in event times does not count." Reason: with event times in H, a null flow giving the same partitions at shifted times passed (F5); `inh:common-clock` exhibits such an architecture (same partitions, event times halved). The remark after `inh:common-clock` and the companion's (F5) paragraph were updated to match. Everything else in the lead's section is unchanged (verified by diff before this amendment).

Checks: `remediation/scripts/R-INH/check_inh.py` (single-threaded, under 20 s) reproduces (A) the Rogers-Pitman example: Lambda T = T^c Lambda with T^c = [[7/10,3/10],[1/2,1/2]], strong lumpability fails, length-5 retained path laws agree exactly from nu Lambda, and differ by 1/10 in total variation from delta_1 (L04 quoted 53/2000 for an unstated measure; the text uses the TV value computed here); (B) partition refinement returns {0,1},{2,3},{4},{5}; (C) the joint response/next-state counterexample; (D) sympy: DR(-grad E) = -R for the center-of-mass example with arbitrary W; (E) the common-clock identity on 3 agents x 3 symbols with Gibbs rows, a moment-dependent constant-output channel and a shared evaluator: residual spread over symbols 6e-11 (finite differences). `lint.py` checks environment and brace balance, `$` parity, banned words, `---`, and `\; \, \!` (all clean). No compilation (per brief). All `\cref` targets resolve except `form:kappa-independence`, which lives in the lead's text and is expected from the chapter 11 agent.

| Finding | Status | What changed (file, label) |
| :- | :- | :- |
| L04-INH-01 (H2) | applied by lead; supported | The lead's (F4) compression clause and the identity-retention example in the formation section implement it. I added `inh:canonical-parent`/`inh:canonical-lumping`, which make (F4) a precise question (noninjectivity of the canonical parent for the declared interface). 06_outlook not mine. |
| L04-INH-02 (E2) | applied | ch05 new `inh:canonical-parent` (coarsest exact retention containing an interface, shift update, Hermann-Krener analytic case) and `inh:canonical-lumping` (coarsest strongly lumpable refinement by partition refinement; Larsen-Skou, Derisavi-Hermanns-Sanders). `inh:two-retentions` now points to it. Checked by script (B). |
| L04-INH-03 (E3) | applied | ch05 new `inh:intertwining` (Rogers-Pitman, finite discrete time) with self-contained proof, predictive responses Lambda K_a, the three-state example, and the link to "handoff from a declared conditional law" in the memory section. Checked by script (A). |
| L04-INH-04 (H2) | applied | ch05 new `inh:trajectory-remark` (single-trajectory inheritance is vacuous: embedded-arc argument) and the invariant set Omega_I in `inh:inherited-parent`; (F1) of the lead already requires an open admitted class. |
| L04-INH-05 (H2) | applied | ch05 new `inh:inherited-parent`: protocol class A_I and preparation class P_I declared before R_I, invariant set Omega_I, conditions (D) dynamics, (B) joint response/next-state, (C) channels and evaluators including configuration dependence, (G) optional retained connection; states it is the sense of (F3). |
| L04-INH-06 | applied | ch05 `inh:gradient-inheritance` restated as an iff with the horizontal criterion `inh:horizontal-criterion`; old potential hypothesis kept as the "in particular" case; remark that the potential hypothesis freezes discarded dynamics; center-of-mass example (script D); uniform necessity of the Riemannian-submersion condition with proof; fiber-minimization counterexample updated to the new criterion. O'Neill 1966, Baird-Wood 2003 cited. Companion c05 explains the same with the two-particle picture. |
| L04-INH-07 | applied | ch05 new `inh:coordinate-projection` (product metric: projection exact iff E separates) with the consequence for the reciprocal Fisher flow; the finite example is described as exact lumping of a drive-response chain of one agent's marks (intro and end of the finite section; companion). The "two or more reciprocally coupled agents" example is the center-of-mass example in `inh:gradient-inheritance`. |
| L04-INH-08 | applied | ch05 boundary-response section: joint kernel of retained response and next retained state must factor through R; the (u,v) counterexample (script C); sufficient conditional-independence alternative; adaptive protocols per receiver action. Also condition (B) of `inh:inherited-parent`. |
| L04-INH-09 | applied | ch05 channel square now displayed as `inh:channel-square` with the garbling criterion, deterministic reduction, BSC(1/4) counterexample, and configuration-dependence clause; s04 notes that `sinh:linear-descent` covers deterministic linear retentions only. |
| L04-INH-10 | applied | Attributions added where stated: factor map (`inh:deterministic-descent`), Wei-Kuo, Li-Rabitz, Horstmeyer-Atay, Barnett-Seth (`inh:vector-square`), Kemeny-Snell weak lumpability, Dynkin, Burke-Rosenblatt, Buchholz (after `inh:finite-lumpability`), Rogers-Pitman, O'Neill, Baird-Wood, Dawid, Lauritzen (`inh:separator-gluing`), Golubitsky-Stewart-Schaeffer (`inh:smooth-no-finite-entry`), Shalizi-Crutchfield and Barnett-Seth Sec. 2.3 (`inh:causal-memory`), Baum-Petrie and Ephraim-Merhav (s04 `sinh:filter`). New intro paragraph states what is classical and what is new. Halmos-Savage for `prop:common-family-recovery` is in ch04, not mine (not applied). Kemeny-Snell theorem locator not added (unverified). |
| L04-INH-11 (R4, L07-BND-01, X2 scheme (i)) | applied | ch05 new `inh:equivariant-descent` (free proper action: equivariant fields and invariant kernels descend; invariant metric and potential give the quotient-metric gradient flow with a = 1), with stratification and Banach-chart caveats. Cross-references from ch10/ch12 belong to those owners. |
| L04-INH-12 (R4, L01-FOUND-10 part) | applied (definition) / deferred (theorem) | ch05 new `inh:type-closed` and open problem `inh:type-closure-open` (Wishart additive parents as first test). Statistical-bundle morphisms (L01-FOUND-10 main part) belong to ch01. |
| L04-INH-13 | applied | `inh:deterministic-descent` now on standard Borel spaces via the analytic-graph theorem (Kechris Thm 14.12, verified via the book's cross-reference in 12.4), right-inverse version kept as the general measurable case; same argument cited for lumped and boundary kernels. New `inh:projectable` (projectable-field criterion, Li-Rabitz linear form). |
| L04-INH-14 | applied | After `inh:separator-gluing`: two-block gluing statement, XOR counterexample for k >= 3 blocks, four-cycle counterexample, junction-tree requirement (Vorob'ev 1962, Lauritzen 1996). Label kept (only referenced in ch05); the result title already says "elimination". |
| L04-INH-15 | applied | `inh:selector-symmetry` restated for the stabilizer of the law (block systems; primitive groups give only trivial partitions); hypothesis |V| >= 3 removed (handled in proof); dihedral four-cycle consequence added. Statement still covers the full-symmetry case cited by 06_outlook. |
| L04-INH-16 | applied | Memory section: the filter is an information-state Markov process on the simplex, not a configuration retention; compresses only under noise or nonobservability; link to the observability class. s04: cycle example filters are delta_2 and delta_1; information-state remark. |
| L04-INH-17 | applied | `inh:role-loss` proof: "does not point outward on any face", forward invariance, tangent at lambda = 0, equilibrium at the corner; same fix in c05. |
| L04-INH-18 | applied | ch05 convention paragraph in section 1 (circ for maps, juxtaposition for kernels in acting order); `inh:composition` proof rewritten with circ for functions and a sentence relating the kernel R_10 R_21 to R_21 o R_10; channel square linear form and s04 `sinh:linear-square`/`sinh:kernel-condition` use circ. companion/glossary.tex:28 is not mine (left for the lead). |
| L04-INH-19 | deferred | VFE_4.0 analytics change; outside manuscript ownership. |
| Lead request: (F5) event times | applied | `inh:null-flow` defines P, P^0 (partition sequences without times); (F5) is now P != P^0 on an open set, with time-only differences excluded. Remark after `inh:common-clock` and companion updated. |
| H2 (ch5 parts) | applied | Via the lead's formation section plus L04-INH-01/04/05/08 above. |
| E2, E3 | applied | See L04-INH-02, L04-INH-03. |
| R4 | applied in part | Canonical parent, intertwining, equivariant descent, horizontal criterion, coordinate projections, X2 scalar additive scheme (`inh:scalar-additive`), garbling, analytic graph, separator gluing, information state, Markov category paragraph, type closure definition. I3 (ch02), L06-ERA-04 (ch08), X1-GAUGE-07/X4-PRIOR-06 ch12 part, X6-LIT-15 (ch07) belong to other owners. |
| X2-FRAME-01 | applied by lead + supported | Lead's (F5); my `inh:common-clock` and `inh:scalar-additive` give the structural reason. Abstract/ch11/outlook edits not mine. |
| X2-FRAME-03 | not in my files beyond lead's section | Consolidated open problem is the lead's `inh:formation-open`. |
| X2-FRAME-05 | applied (ch05 part) | Scheme theorems stated: symmetry-quotient = `inh:equivariant-descent`; additive scheme = `inh:scalar-additive`. Sentences in formation chapters belong to their owners. |
| X2-FRAME-08 | applied (ch05 part) | `inh:role-loss` and `inh:independence-discontinuity` retyped from \constructionheading (prints Definition) to \propositionheading with proof markers. rho collision removed: separator reference law renamed omega (omega_Z, omega_UV, omega_U, omega_V); transmission coefficient renamed vartheta(x); rho kept for the record readout per AUTHORING. Macros/file names/c07 are not mine. |
| X2-FRAME-09 | applied by lead (clauses); deferred (P1-P5) | Weak/strong necessity and amplification are in the lead's text. Simulation predictions not added (no experiments in authoring scope). |
| X2-FRAME-10 | applied | ch05 new `inh:scalar-additive` (Jensen argument; partition sequence is an initial segment of an initial-data sequence when the common factor is nonincreasing and there are no ties) and the anisotropic escape remark with the diag(-1,-3) example. Proof reconstructed; the "prefix" statement is mine and is proved under the stated monotonicity hypothesis. |
| X4-PRIOR-01 | applied by lead | (F6) counterfactual-handoff clause. |
| X4-PRIOR-04 | applied | Paragraph after `inh:selector-symmetry`: which joint laws the selectors could read in 2.0, the covariance example (verified by hand: off-diagonal of e^{2At}), and that these results are constraints on selectors, not formation mechanisms. |
| X4-PRIOR-06 | applied (ch05 part) | Quotient-gradient clause restored as part (ii) of `inh:equivariant-descent`, finite-dimensional; Banach-chart caveat stated (the X1-GAUGE-07 vs X4-PRIOR-06 chart conflict is left to ch12). |
| X4-PRIOR-09 | deferred | Downward action: memory section now states the exact-reverse-channel tautology and that downward action needs a non-exact reverse channel and the moving-map square; no construction (research). |
| X4-PRIOR-10 | no change needed | ch05:449-451 already disclaims a fixed-joint ELBO identification; retitling is ch03. |
| X1-GAUGE-04 | no change needed in my files | ch05 is only cited as the contrast ("independent sector frames"). |
| X1-GAUGE-05 | applied (ch05 part) | After `inh:smooth-channel-covariance`: the frames are per-agent coordinates on each agent's fiber; passive re-expression constrains nothing. The global bundle decision is ch01. |
| X3-PIFB-06 (E7) | applied (i)-(ii); deferred (iii) | ch05 new `inh:no-underdamped` (real spectrum of inherited linearization) with proof. The completely monotone memory kernel (iii) not added. |
| X3-PIFB-08 | applied (lemma) / deferred (construction) | Tautology lemma stated in the memory section; reference-conditional construction not added. |
| X6-LIT-06 | applied (ch05 part) | Linear pool attribution (Genest-Zidek, Abbas) at the averaged-law section; Markov-blanket distinction paragraph after `inh:inherited-parent` (Kirchhoff et al. 2018, Palacios et al. 2020, Friston et al. 2021). VFE sources and log-linear pool belong to ch03. |
| X6-LIT-09 | applied | Wei-Kuo, Li-Rabitz, Horstmeyer-Atay at `inh:vector-square`; Dynkin and Rogers-Pitman after lumpability; Gray "Thm. 7.3.1" (bib entry is the 2023 corrected first edition that X6 read); Barnett-Seth Sec. 2.3 and Shalizi-Crutchfield at `inh:causal-memory`. |
| X6-LIT-12 (ch05/s04 parts) | applied | Golubitsky-Stewart-Schaeffer at `inh:smooth-no-finite-entry`; Dawid, Lauritzen at `inh:separator-gluing`; Baum-Petrie, Ephraim-Merhav at `sinh:filter` (instead of Rabiner, to cite the primary source). |
| X6-LIT-13 (ch05 part) | applied in part | Gray locator updated. Kemeny-Snell locator not added (unverified). bib record corrections are the lead's (references.bib not mine). |
| X6-LIT-15 | deferred to ch07 owner | Proposition belongs in ch07 per the finding; `inh:gradient-inheritance` now has the iff form it would use. |
| X6-LIT-16 | applied | Convention paragraph in section 1 names the Markov category of standard Borel spaces (Fritz 2020); composition is pasting. |
| L08-CND-01, L08-CND-10 (E1) | applied (ch05 part) | Lead's (F5); my `inh:common-clock` proves the mean-field identity d/dt log(q_i/q_j) = -2 log(q_i/q_j) + c_ij(t) on positive tables under stated hypotheses (shared model-independent evaluator, self excluded from belief sources, constant-output moment channel), plus the null-flow rate -1. This supplies the proof behind the 06_outlook sentence "shared references with additive moment channels make every pairwise law comparison relax on a common clock". The candidate fiber obstruction of L08-CND-10 was not proved (deferred). |
| L01-FOUND-10 | applied in part | Type closure definition (`inh:type-closed`); bundle morphisms are ch01. |
| L03-DYN-09 (ch05:575) | applied | Horizon of `inh:causal-memory` renamed from T to n (T is a transition matrix in the chapter). |
| G2-02 (ch05 part) | applied | Sentence after `inh:vector-square`: weakly differentiable law-valued retentions are not covered and need a direct intertwining proof. |
| X5-IMPL-10 | rejected for this agent | Concerns VFE_4.0 code labels; no manuscript change in my files. |
| Companion c05 | applied | Gradient section: horizontal criterion, two-particle center-of-mass picture, dropping agents is exact only for isolated blocks, no oscillation from exact inheritance; corrected boundary wording; drive-response description; new section "What a parent must remember" (filter as information state, intertwined handoffs, smallest exact parent, probes decide what can be forgotten, stabilizer form of the selector result); new section "The formation standard" explaining (F1) to (F6), the null-flow control, why the identity retention and non-interacting grouping are excluded, the common-clock architectures, the term "two-stage selection with exact inheritance", and the classification matching the lead's text. |

## BIB ENTRIES

```bibtex
@article{RogersPitman1981,
  author = {Rogers, L. C. G. and Pitman, J. W.},
  title = {Markov Functions},
  journal = {The Annals of Probability},
  volume = {9},
  number = {4},
  pages = {573--582},
  year = {1981},
  doi = {10.1214/aop/1176994363}
}
@article{BurkeRosenblatt1958,
  author = {Burke, C. J. and Rosenblatt, M.},
  title = {A {M}arkovian Function of a {M}arkov Chain},
  journal = {The Annals of Mathematical Statistics},
  volume = {29},
  number = {4},
  pages = {1112--1122},
  year = {1958},
  doi = {10.1214/aoms/1177706444}
}
@article{Buchholz1994,
  author = {Buchholz, Peter},
  title = {Exact and Ordinary Lumpability in Finite {M}arkov Chains},
  journal = {Journal of Applied Probability},
  volume = {31},
  number = {1},
  pages = {59--75},
  year = {1994},
  doi = {10.2307/3215235}
}
@book{Dynkin1965,
  author = {Dynkin, E. B.},
  title = {Markov Processes, Volume I},
  series = {Grundlehren der mathematischen Wissenschaften},
  volume = {121},
  publisher = {Springer},
  address = {Berlin},
  year = {1965}
}
@book{Kechris1995,
  author = {Kechris, Alexander S.},
  title = {Classical Descriptive Set Theory},
  series = {Graduate Texts in Mathematics},
  volume = {156},
  publisher = {Springer},
  address = {New York},
  year = {1995},
  doi = {10.1007/978-1-4612-4190-4}
}
@article{WeiKuo1969,
  author = {Wei, James and Kuo, James C. W.},
  title = {A Lumping Analysis in Monomolecular Reaction Systems: Analysis of the Exactly Lumpable System},
  journal = {Industrial \& Engineering Chemistry Fundamentals},
  volume = {8},
  number = {1},
  pages = {114--123},
  year = {1969},
  doi = {10.1021/i160029a019}
}
@article{LiRabitz1989,
  author = {Li, Genyuan and Rabitz, Herschel},
  title = {A General Analysis of Exact Lumping in Chemical Kinetics},
  journal = {Chemical Engineering Science},
  volume = {44},
  number = {6},
  pages = {1413--1430},
  year = {1989},
  doi = {10.1016/0009-2509(89)85014-6}
}
@article{ONeill1966,
  author = {O'Neill, Barrett},
  title = {The Fundamental Equations of a Submersion},
  journal = {Michigan Mathematical Journal},
  volume = {13},
  number = {4},
  pages = {459--469},
  year = {1966},
  doi = {10.1307/mmj/1028999604}
}
@book{BairdWood2003,
  author = {Baird, Paul and Wood, John C.},
  title = {Harmonic Morphisms Between {R}iemannian Manifolds},
  series = {London Mathematical Society Monographs, New Series},
  volume = {29},
  publisher = {Oxford University Press},
  address = {Oxford},
  year = {2003}
}
@book{Lee2013,
  author = {Lee, John M.},
  title = {Introduction to Smooth Manifolds},
  edition = {2},
  series = {Graduate Texts in Mathematics},
  volume = {218},
  publisher = {Springer},
  address = {New York},
  year = {2013},
  doi = {10.1007/978-1-4419-9982-5}
}
@article{LarsenSkou1991,
  author = {Larsen, Kim G. and Skou, Arne},
  title = {Bisimulation through Probabilistic Testing},
  journal = {Information and Computation},
  volume = {94},
  number = {1},
  pages = {1--28},
  year = {1991},
  doi = {10.1016/0890-5401(91)90030-6}
}
@article{DerisaviHermannsSanders2003,
  author = {Derisavi, Salem and Hermanns, Holger and Sanders, William H.},
  title = {Optimal State-Space Lumping in {M}arkov Chains},
  journal = {Information Processing Letters},
  volume = {87},
  number = {6},
  pages = {309--315},
  year = {2003},
  doi = {10.1016/S0020-0190(03)00343-0}
}
@article{Dawid1979,
  author = {Dawid, A. P.},
  title = {Conditional Independence in Statistical Theory},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume = {41},
  number = {1},
  pages = {1--31},
  year = {1979},
  doi = {10.1111/j.2517-6161.1979.tb01052.x}
}
@book{Lauritzen1996,
  author = {Lauritzen, Steffen L.},
  title = {Graphical Models},
  series = {Oxford Statistical Science Series},
  volume = {17},
  publisher = {Clarendon Press},
  address = {Oxford},
  year = {1996}
}
@article{Vorobev1962,
  author = {Vorob'ev, N. N.},
  title = {Consistent Families of Measures and Their Extensions},
  journal = {Theory of Probability and Its Applications},
  volume = {7},
  number = {2},
  pages = {147--163},
  year = {1962},
  doi = {10.1137/1107014}
}
@book{GolubitskyStewartSchaeffer1988,
  author = {Golubitsky, Martin and Stewart, Ian and Schaeffer, David G.},
  title = {Singularities and Groups in Bifurcation Theory, Volume {II}},
  series = {Applied Mathematical Sciences},
  volume = {69},
  publisher = {Springer},
  address = {New York},
  year = {1988},
  doi = {10.1007/978-1-4612-4574-2}
}
@book{DixonMortimer1996,
  author = {Dixon, John D. and Mortimer, Brian},
  title = {Permutation Groups},
  series = {Graduate Texts in Mathematics},
  volume = {163},
  publisher = {Springer},
  address = {New York},
  year = {1996},
  doi = {10.1007/978-1-4612-0731-3}
}
@article{BaumPetrie1966,
  author = {Baum, Leonard E. and Petrie, Ted},
  title = {Statistical Inference for Probabilistic Functions of Finite State {M}arkov Chains},
  journal = {The Annals of Mathematical Statistics},
  volume = {37},
  number = {6},
  pages = {1554--1563},
  year = {1966},
  doi = {10.1214/aoms/1177699147}
}
@article{EphraimMerhav2002,
  author = {Ephraim, Yariv and Merhav, Neri},
  title = {Hidden {M}arkov Processes},
  journal = {IEEE Transactions on Information Theory},
  volume = {48},
  number = {6},
  pages = {1518--1569},
  year = {2002},
  doi = {10.1109/TIT.2002.1003838}
}
@article{GenestZidek1986,
  author = {Genest, Christian and Zidek, James V.},
  title = {Combining Probability Distributions: A Critique and an Annotated Bibliography},
  journal = {Statistical Science},
  volume = {1},
  number = {1},
  pages = {114--135},
  year = {1986},
  doi = {10.1214/ss/1177013825}
}
@article{Abbas2009,
  author = {Abbas, Ali E.},
  title = {A {K}ullback-{L}eibler View of Linear and Log-Linear Pools},
  journal = {Decision Analysis},
  volume = {6},
  number = {1},
  pages = {25--37},
  year = {2009},
  doi = {10.1287/deca.1080.0133}
}
@article{KirchhoffEtAl2018,
  author = {Kirchhoff, Michael and Parr, Thomas and Palacios, Ensor and Friston, Karl and Kiverstein, Julian},
  title = {The {M}arkov Blankets of Life: Autonomy, Active Inference and the Free Energy Principle},
  journal = {Journal of the Royal Society Interface},
  volume = {15},
  number = {138},
  pages = {20170792},
  year = {2018},
  doi = {10.1098/rsif.2017.0792}
}
@article{PalaciosEtAl2020,
  author = {Palacios, Ensor Rafael and Razi, Adeel and Parr, Thomas and Kirchhoff, Michael and Friston, Karl},
  title = {On {M}arkov Blankets and Hierarchical Self-Organisation},
  journal = {Journal of Theoretical Biology},
  volume = {486},
  pages = {110089},
  year = {2020},
  doi = {10.1016/j.jtbi.2019.110089}
}
@article{FristonEtAl2021,
  author = {Friston, Karl J. and Fagerholm, Erik D. and Zarghami, Tahereh S. and Parr, Thomas and Hip{\'o}lito, In{\^e}s and Magrou, Lo{\"i}c and Razi, Adeel},
  title = {Parcels and Particles: {M}arkov Blankets in the Brain},
  journal = {Network Neuroscience},
  volume = {5},
  number = {1},
  pages = {211--251},
  year = {2021},
  doi = {10.1162/netn_a_00175}
}
@article{Fritz2020,
  author = {Fritz, Tobias},
  title = {A Synthetic Approach to {M}arkov Kernels, Conditional Independence and Theorems on Sufficient Statistics},
  journal = {Advances in Mathematics},
  volume = {370},
  pages = {107239},
  year = {2020},
  doi = {10.1016/j.aim.2020.107239}
}
```

Verification notes: Derisavi-Hermanns-Sanders, Vorob'ev, Friston et al. 2021, Buchholz, Burke-Rosenblatt, and Kechris Thm 14.12 were checked against publisher or book pages in this session. The remaining entries are standard records recalled from memory (volume, pages, year); the DOIs of Rogers-Pitman, Buchholz (JSTOR-style), Wei-Kuo, Li-Rabitz, O'Neill, Larsen-Skou, Dawid, Baum-Petrie, Ephraim-Merhav, Genest-Zidek, Abbas, Kirchhoff et al., Palacios et al., Fritz, Lee, Kechris, Golubitsky et al., and Dixon-Mortimer should be confirmed by the lead before merging (drop any DOI that does not resolve). Locators used in text: Kechris Thm. 14.12 and Sec. 18.A, Lee Thm. 21.10, Gray Thm. 7.3.1, Barnett-Seth Sec. 2.3.

## NEW LABELS

Chapter 5 (all checked unique across the three documents): `inh:projectable`, `inh:projectable-criterion`, `inh:trajectory-remark`, `inh:horizontal-criterion`, `inh:equivariant-descent`, `inh:coordinate-projection`, `inh:no-underdamped`, `inh:intertwining`, `inh:intertwining-identity`, `inh:canonical-parent`, `inh:canonical-retention`, `inh:canonical-lumping`, `inh:channel-square`, `inh:inherited-parent`, `inh:type-closed`, `inh:type-closure-open`, `inh:scalar-additive`, `inh:common-clock`, `inh:common-clock-identity`. No label was removed or renamed. Headings changed type: `inh:role-loss` and `inh:independence-discontinuity` (Definition to Proposition), which changes their printed numbers' type only.

## AUTHOR DECISIONS NEEDED

1. None outstanding from this agent. The (F5) event-time loophole I had flagged was resolved by the lead's instruction (see the note at the top). Under the amended (F5) the lead's sentence that chapter 11 "fails (F5)" no longer depends on whether the chapter 11 null flow changes the shape-relaxation rate; it needs only that the partition sequence is coupling-independent and equal to the null flow's, which the chapter 11 owner should confirm against `form:kappa-independence`.
