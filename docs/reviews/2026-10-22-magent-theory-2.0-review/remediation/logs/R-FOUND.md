# R-FOUND remediation log (2026-10-22)

Files edited (M = Research/manuscripts/MAgent_Theory_2.0): chapters/01_foundations.tex, chapters/03_dynamics.tex, chapters/04_prediction.tex, appendices/record_process.tex, appendices/notation.tex, supplements/s01_probability.tex, supplements/s03_variation.tex, companion/c01_objects.tex, companion/c03_dynamics.tex, companion/c04_communication.tex, companion/glossary.tex. Nothing compiled. Lint script (`scripts/R-FOUND/lint.py`): no duplicate labels across the three documents, every `\cref`/`\eqref` in all three documents resolves, environments and braces balanced in my files, no `\;`, `\,`, `\!` (three occurrences in the lead's ch3 text removed). Checks: `scripts/R-FOUND/checks.py`, output `checks.out` (all residuals at most 1e-15; global-floor ODE kept its 1e-9 initial minimum, energy nonincreasing to 3e-16, field 3e-9 at t = 60).

Lead material preserved in full (sec:generative-models, def:generative-model, eq:agent-free-energy, def:shared-model, dressing paragraph; ch3 three objectives, prop:dyn-global-local, def/prop:dyn-per-agent, def:dyn-per-agent-flow, prop:dyn-no-global-vfe, def:dyn-record-interrupted, op:dyn-three-pictures). Corrections to it: (1) def:shared-model follow-up text had belief agreement as KL(s_i || phi_# s_j) = 0; with r_j = phi_# r_i the consistent statement is s_j = phi_# s_i, so it now reads KL(s_j || phi_# s_i) = 0. (2) Style: `\,` in the proof of prop:dyn-global-local and in eq:dyn-per-agent-model, `\int\!\!\int` in eq:dyn-per-agent-free-energy replaced by `\iint`. I reconstructed prop:dyn-global-local, prop:dyn-per-agent and both parts of prop:dyn-no-global-vfe and found no mathematical error.

Coordinator notes: (a) ch1 now says app:connection "records the transformation law of supplied connection data in a moving presentation, and the construction of a connection from agent interactions remains open there", and that no vector field, row, channel or record reads the connection (diagnostic), matching R-GEOM's retitled appendix. (b) glossary RF = F^cR changed to R∘F = F^c∘R (function order, matching ch5 `inh:update-square` and its stated convention "∘ composes maps, juxtaposition composes kernels"); added entries for null flow, partition sequences P, P^0 without event times, clause (F5) ("a shift of event times alone does not count"), and two-stage selection with exact inheritance.

| ID | Status | Change (file, label) / reason |
|---|---|---|
| L01-FOUND-01 | applied | New ch1 section sec:content-laws: def:content-law (coefficient space, P_0, Q, sample-section maps, finite designs, covariant record kernels), prop:content-law (finite-design consistency, projected free energy eq:content-projected), statement that the law-valued configuration is the list of marginals and the reciprocal potential is a separate object (pointer to prop:dyn-no-global-vfe). Hopf witness not imported: deferred (needs its own appendix and the explicit geometry). |
| L01-FOUND-02 | applied | ch1 prop:loop-blindness (law-pair blindness inside the stabilizer; shared-sample graph law determines H a.e.; diagonal transformation); s01 two-route section and c01 extended. |
| L01-FOUND-03 | applied (partly) | ch1: agent-dependent G-spaces Y_i, bundles B_i; frames u_i in P_{c*}, u_i = sigma U_i^{-1}; linear realization U_i^a = rho_a(U_i); E_i^a defined as coordinate copies of V_a; independent sector frames require G = G_q x G_s. Separate symbol for the displayed law not introduced: ch2 (not mine) writes laws on E_i^a as q_i; ch1 now states that laws on E_i^a are the displayed laws. |
| L01-FOUND-04 | applied | ch1: O_i standard Borel; covariance requirement for readouts, record/response kernels and transitions; prop:record-covariance; rotation counterexample; transformation rule A_i g_i^{-1}, B_ij g_j^{-1} for eq:sample-comparison; c01 sentence. |
| L01-FOUND-05 | applied | ch1: jointly Borel actions, second-countable G and C, open U_i, Borel sections, smooth sections only into admitted subbundles; lem:measurable-bundle with proof; Hamel-type warning; disintegration commutes with the action. |
| L01-FOUND-06 | applied (ch3 part) | ch3 after eq:dyn-peer-costs: law-dependent channels are kernels on S x K_j; Borel measurability of costs, rows, E_red (Gray1990). def:int-channel and def:int-generator are in ch2 (not mine). |
| L01-FOUND-07, -08, -09, -14 | not in my files | Locations in ch2, s02, c02 (other agents). ch1 wording "selective or noisy channel" changed to "channel that discards information or adds noise" (L01-FOUND-08 terminology). |
| L01-FOUND-10 | applied | ch1 def:bundle-morphism, prop:bundle-morphism (well defined, affine, KL-contracting) and agent-type paragraph; type closure stated as part of the formation problem (deferred). |
| L01-FOUND-11 | applied | record_process prop:variational-history, eq:variational-history (checked: residual 5.9e-18, KL 0.001207 <= 0.025842). |
| L01-FOUND-12 | applied | ch1 scope sentence; ch3 rem:dyn-product (joint minimizer vs product belief minimizer); c01 paragraph. |
| L01-FOUND-13 | applied as hypothesis | ch1 records section: measurability of Banach-chart flows is a stated hypothesis, with the reason. The Picard-iterate lemma is deferred (not verified per chapter, and L^infinity non-separability makes the general lemma non-trivial). |
| L01-FOUND-15 | applied | record_process: Borel admitted set, a.e. initial state, invariance; measurability lemma; Ionescu-Tulcea extension; "For a fixed latent model m" -> "For a fixed normalized observation density k". |
| L01-FOUND-16 | applied (my files) | notation.tex: added E_i^a, V_a, u_i, sigma, B_i, E_i, R_ij, nu_ij, D_ij, mu_j, and others; closing paragraph lists the B, nu, F overloads. Renaming the content map M_ij^a and moment receiver is in ch2 (not mine). |
| L01-FOUND-17 | applied (my files) | ch1: source rows forward reference; q_i(dk given m) versus G_i(m,dk) sentence; record_process: event kernel R is the prop:record-law kernel. Link-to-channel sentence belongs to ch2. |
| L01-FOUND-18 | applied | ch1 prop:kl-coordinate stated for injective Borel maps (Lusin-Souslin, Kechris1995 Thm 15.1); s01 chain-rule proof rebuilt from f = d mu/d nu, identity in [0, infinity]. |
| L01-FOUND-19 | applied | ch1 and c01: hard-record point mass at log 2; mean-field m = tanh(Jm) threshold J = 1. |
| L01-FOUND-20 | applied (partly) | Kobayashi-Nomizu (Sec. I.5) and Kechris added. Kallenberg locator not added (third-edition numbering not verified). Higham/Culver/Le Bruyn-Procesi are ch2/s02. |
| L03-DYN-01 | applied | Lead added def:dyn-record-interrupted (PDMP). I added the inventory paragraph (four losses, record-independent stationary set) and def:dyn-evidence-potential. |
| L03-DYN-02 | applied | ch3 sec:dyn-variation: reverse influence through the adjoint channel, message-marginal support, identical-interest game; prop:dyn-nonreciprocal (directed Bernoulli 3-cycle, eigenvalues verified symbolically) with Monderer-Shapley and Fruchart et al.; ch4 sentence on the reverse message; c03, c04. |
| L03-DYN-03 | applied | ch3 def:dyn-chart-flow, prop:dyn-chart-flow (bounded log-density charts, Lipschitz first variation, Picard-Lindelof, continuation alternative, dissipation; sufficient conditions for fixed channels; law-dependent channels as hypothesis). |
| L03-DYN-04 | applied | ch3 prop:dyn-row-relaxation (rate T, checked 0.25000000) and mobility paragraph: Gibbs rows as infinite-row-mobility limit, product-law justification, Cencov; c03. |
| L03-DYN-05 | applied | ch3 prop:dyn-global-flow with full proof (min-entry Dini argument, row floors, Lojasiewicz). I reconstructed the argument; numerics in checks.py part 9. |
| L03-DYN-06 | applied | ch3 cor:dyn-frozen-source, eq:dyn-frozen-source (residual 0.0). |
| L03-DYN-07 | applied by lead | Lead paragraph after prop:dyn-no-global-vfe states the Fisher-volume prior; no change. |
| L03-DYN-08 | applied | Citations: Gibbs principle (DonskerVaradhan1975, DupuisEllis1997), Shahshahani1979 + HofbauerSigmund1998, Jeffrey (DiaconisZabell1982), Csiszar1975, HalmosSavage1949, chain rule (KullbackLeibler1951, CoverThomas2006), CourtadeWeissman2014, Cencov1982. Williams 1980 not added (Diaconis-Zabell suffices). |
| L03-DYN-09 | applied (my files) | ch4 message law nu_j -> mu_j (also in prop:conditional-lift and c04); joint laws in thm:predictive-chain -> P_s, P_t; encoder T -> \mathcal T, decoder D -> Delta; c03 uses R_i; s03 states D_j is the comparison-space divergence. ch2 sector marginals nu_i^a not mine (noted in notation). |
| L03-DYN-10 | applied | s03 sentence ("for every V; only receiver-covariance derivatives differ"); c03 coefficients T_q and 1. |
| L03-DYN-11 | applied | Title already changed by lead; ch3 "passive covariance" sentence and the active-equivariance requirement added. |
| L03-DYN-12 | applied | ch4: chain-rule remark after thm:predictive-chain (deterministic-kernel case, [0, infinity]); prop:conditional-lift weakened to q << p; cor:pythagorean (Pythagorean identity, I-projection uniqueness). |
| L03-DYN-13 | applied | ch3 prop:dyn-morse (Schur complement, Haynsworth). |
| L03-DYN-14 | applied as remark | ch3 rem:dyn-self-sources: row bounds (checked on 2000 rows), bounded-confidence reading. The k-cluster stability theorem is deferred. |
| L03-DYN-15 | deferred | VFE_4.0 code change (law-potential route, regression test); outside the manuscript. |
| X5-IMPL-02 (H8) | applied | ch3 def:dyn-restricted-flow with Gaussian formulas eq:dyn-gaussian-flow; restricted flow = Fisher projection; prop:dyn-gaussian-noninvariance with proof; statement that chapters 7-12 results do not transfer to the restricted flow; constant-in-input channels remove the obstruction. |
| X5-IMPL-03 | applied | def:dyn-evidence-potential and inventory paragraph in ch3; ch4 sentence after eq:bayesian-reception; c04 sentence. Which later constructions survive E_o is not claimed (stated). |
| X5-IMPL-11 | applied | ch3 rem:dyn-product: LOR identity (checked, residual 4e-16). |
| X5-IMPL-18 | applied | Covered by prop:dyn-global-flow (stronger: no positivity-of-channel lower bound needed beyond no null columns). |
| X5-IMPL-01, -05, -12 | deferred | VFE_4.0 implementation findings; nothing in the manuscript is wrong (no statement identifies the record NLL with the peer divergence outside prop:sup-row-equivalence). |
| X5-IMPL-08 | applied | ch1: potentials contain no coupling between contexts; content law section supplies laws on fields. A context-coupling term is deferred. |
| X3-PIFB-01 | applied by lead | prop:dyn-no-global-vfe covers (a) and (b). Part (c) (Gaussian slice) is left to op:dyn-three-pictures (lead); not added. |
| X3-PIFB-02 (C1) | applied | ch3 thm:dyn-registered (i)-(iv) with proof, restrictions (equal temperatures, one batch size, uniform lattice prior), Sanov remark (Csiszar1984) with the double limit stated as not established. Checked: identity residual 0.0, posterior = Gibbs to 1.4e-16. |
| X3-PIFB-05 (E6) | applied | ch3 prop:dyn-compensation, eq:dyn-compensation (residual 5.6e-17), Gaussian variance widening. Comparison with the author's unpublished normalized model omitted (AUTHORING forbids citing it). |
| X3-PIFB-06 (E7) | applied (ch3 part) | prop:dyn-nonreciprocal states real spectrum of reciprocal linearizations and the spectrum-containment consequence of exact intertwining. The full proposition (memory kernel) belongs to ch5. |
| X3-PIFB-07, X4-PRIOR-08 (E5) | applied as remark | ch3 eq:dyn-prior-learning (symbolic residual 0), replicator form, self-isolation, requirement to exclude self; pairing theorem deferred (open). Section 7 conflict with L03-DYN-14 resolved: both hold (fixed priors vs learned priors) and both are stated in one remark. |
| X3-PIFB-08 | deferred | Parent-to-child feedback concerns ch5 parents; prop:conditional-lift unchanged except attribution. |
| X3-PIFB-10 | applied as remark | ch3 sentence after thm:dyn-registered: multinomial covariance = T times inverse Fisher metric; stochastic version and metastability stated open. |
| X3-PIFB-14 | applied (record appendix part) | record_process: binary registration budget n log 2; Dirichlet predictive step <= 1/(A+n) (checked). form:responses corollary is ch11 (not mine). |
| X4-PRIOR-04 | not in my files | Fix belongs in ch5. |
| X4-PRIOR-07 | applied | ch3 prop:dyn-label-model, eq:dyn-label-model (residual -8.9e-16); lead's "additional hypotheses" sentence now points to it and thm:dyn-registered. |
| X4-PRIOR-10 | applied by lead | Chapter retitled by lead. |
| X4-PRIOR-11 | applied | ch4: Halmos-Savage, Skibinsky, sufficiency wording; two-point witness after cor:local-family-recovery. Uniqueness part (ii) not restored (low value). |
| X1-GAUGE-01 | applied (ch1 part) | Dressing paragraph by lead kept; ch1 adds that no vector field, row, channel or record reads the connection. |
| X1-GAUGE-02 | applied (ch1 part) | ch1: active transformations and declared observable algebra defined; "gauge" relative to it; external response protocols named. ch12 restatement is another agent's. |
| X1-GAUGE-05 | applied | As L01-FOUND-03. |
| X1-GAUGE-10 | deferred | Auxiliary gauged sigma-model term: research extension (well-posedness open); belongs to geometry chapter. |
| X2-FRAME-01 (ch1 part) | applied | ch1 opening now states the program aim, what the text supplies, and that a construction meeting the formation standard is open. |
| X2-FRAME-08 (notation part) | applied | notation closing paragraph lists the B, nu, F overloads; "physical" removed from H_I entry. |
| L02-GEOM-10 (ch1 part) | applied | ch1 no longer calls app:connection a realization of the connection. |
| L04-INH-10 (ch4 part) | applied | Halmos-Savage at prop:common-family-recovery. |
| L04-INH-18 (glossary) | applied | R∘F = F^c∘R. |
| X6-LIT-01 | applied | ch3 after thm:dyn-finite-flow (replicator, Shahshahani, Fisher's fundamental theorem). |
| X6-LIT-02 | applied | ch4 Jeffrey/Diaconis-Zabell, Csiszar Pythagorean, Halmos-Savage, Petz; s01 Jeffrey sentence; c04. Baez-Fritz not added. |
| X6-LIT-06 | applied (ch3 part) | VFE sources at eq:dyn-fixed-vfe (NealHinton1998, JordanEtAl1999, Friston2010); log-linear pool, KL consensus, DeGroot after def:dyn-potential. Markov-blanket positioning belongs to ch5. |
| X6-LIT-12 (ch3 part) | applied | Gibbs variational principle citation at prop:dyn-gibbs. |
| X6-LIT-13 | deferred (lead) | Stale bib records are in references.bib (lead-owned). Kallenberg locator not verified. |
| X6-LIT-14 | applied | Cencov paragraph in ch3 (distinction joint-law invariance vs per-table invariance). |
| X6-LIT-15 (ch4 part) | deferred | Proposition belongs in ch7; ch4 now identifies the lift as Jeffrey's rule and the I-projection. |
| X6-LIT-16 (ch4 part) | applied | Fritz2020 sentence at prop:common-family-recovery. |
| G2-01 | applied (ch3 part) | prop:dyn-chart-flow is the general proposition ch7 can cite (label prop:dyn-chart-flow). |
| G2-04 | partly | Finite-table global statement in prop:dyn-global-flow; general-space global classes belong to ch7 (stated in ch3 that they are proved where used). |

## BIB ENTRIES

```bibtex
@book{KobayashiNomizu1963,
  author = {Kobayashi, Shoshichi and Nomizu, Katsumi},
  title = {Foundations of Differential Geometry, Volume I},
  publisher = {Interscience Publishers},
  address = {New York},
  year = {1963}
}
@book{Kechris1995,
  author = {Kechris, Alexander S.},
  title = {Classical Descriptive Set Theory},
  series = {Graduate Texts in Mathematics},
  volume = {156},
  publisher = {Springer},
  address = {New York},
  year = {1995}
}
@article{GenestZidek1986,
  author = {Genest, Christian and Zidek, James V.},
  title = {Combining Probability Distributions: A Critique and an Annotated Bibliography},
  journal = {Statistical Science},
  volume = {1}, number = {1}, pages = {114--135}, year = {1986},
  doi = {10.1214/ss/1177013825}
}
@article{Abbas2009,
  author = {Abbas, Ali E.},
  title = {A {K}ullback--{L}eibler View of Linear and Log-Linear Pools},
  journal = {Decision Analysis},
  volume = {6}, number = {1}, pages = {25--37}, year = {2009}
}
@article{BattistelliChisci2014,
  author = {Battistelli, Giorgio and Chisci, Luigi},
  title = {{K}ullback--{L}eibler Average, Consensus on Probability Densities, and Distributed State Estimation with Guaranteed Stability},
  journal = {Automatica},
  volume = {50}, number = {3}, pages = {707--718}, year = {2014}
}
@article{DeGroot1974,
  author = {DeGroot, Morris H.},
  title = {Reaching a Consensus},
  journal = {Journal of the American Statistical Association},
  volume = {69}, number = {345}, pages = {118--121}, year = {1974}
}
@article{DonskerVaradhan1975,
  author = {Donsker, Monroe D. and Varadhan, S. R. Srinivasa},
  title = {Asymptotic Evaluation of Certain {M}arkov Process Expectations for Large Time, {I}},
  journal = {Communications on Pure and Applied Mathematics},
  volume = {28}, number = {1}, pages = {1--47}, year = {1975}
}
@book{DupuisEllis1997,
  author = {Dupuis, Paul and Ellis, Richard S.},
  title = {A Weak Convergence Approach to the Theory of Large Deviations},
  publisher = {Wiley}, address = {New York}, year = {1997}
}
@article{HegselmannKrause2002,
  author = {Hegselmann, Rainer and Krause, Ulrich},
  title = {Opinion Dynamics and Bounded Confidence: Models, Analysis and Simulation},
  journal = {Journal of Artificial Societies and Social Simulation},
  volume = {5}, number = {3}, year = {2002}
}
@book{HofbauerSigmund1998,
  author = {Hofbauer, Josef and Sigmund, Karl},
  title = {Evolutionary Games and Population Dynamics},
  publisher = {Cambridge University Press}, address = {Cambridge}, year = {1998}
}
@article{MondererShapley1996,
  author = {Monderer, Dov and Shapley, Lloyd S.},
  title = {Potential Games},
  journal = {Games and Economic Behavior},
  volume = {14}, number = {1}, pages = {124--143}, year = {1996}
}
@book{Shahshahani1979,
  author = {Shahshahani, Siavash},
  title = {A New Mathematical Framework for the Study of Linkage and Selection},
  series = {Memoirs of the American Mathematical Society},
  volume = {17}, number = {211},
  publisher = {American Mathematical Society}, address = {Providence, RI}, year = {1979}
}
@incollection{Lojasiewicz1963,
  author = {{\L}ojasiewicz, Stanis{\l}aw},
  title = {Une propri{\'e}t{\'e} topologique des sous-ensembles analytiques r{\'e}els},
  booktitle = {Les {\'E}quations aux D{\'e}riv{\'e}es Partielles},
  publisher = {{\'E}ditions du Centre National de la Recherche Scientifique}, address = {Paris},
  pages = {87--89}, year = {1963}
}
@article{AbsilMahonyAndrews2005,
  author = {Absil, P.-A. and Mahony, R. and Andrews, B.},
  title = {Convergence of the Iterates of Descent Methods for Analytic Cost Functions},
  journal = {SIAM Journal on Optimization},
  volume = {16}, number = {2}, pages = {531--547}, year = {2005}
}
@book{Cencov1982,
  author = {{\v{C}}encov, N. N.},
  title = {Statistical Decision Rules and Optimal Inference},
  series = {Translations of Mathematical Monographs},
  volume = {53},
  publisher = {American Mathematical Society}, address = {Providence, RI}, year = {1982}
}
@article{Haynsworth1968,
  author = {Haynsworth, Emilie V.},
  title = {Determination of the Inertia of a Partitioned {H}ermitian Matrix},
  journal = {Linear Algebra and its Applications},
  volume = {1}, number = {1}, pages = {73--81}, year = {1968}
}
@book{Dieudonne1960,
  author = {Dieudonn{\'e}, Jean},
  title = {Foundations of Modern Analysis},
  publisher = {Academic Press}, address = {New York}, year = {1960}
}
@article{FruchartEtAl2021,
  author = {Fruchart, Michel and Hanai, Ryo and Littlewood, Peter B. and Vitelli, Vincenzo},
  title = {Non-Reciprocal Phase Transitions},
  journal = {Nature},
  volume = {592}, pages = {363--369}, year = {2021}
}
@incollection{NealHinton1998,
  author = {Neal, Radford M. and Hinton, Geoffrey E.},
  title = {A View of the {EM} Algorithm that Justifies Incremental, Sparse, and Other Variants},
  booktitle = {Learning in Graphical Models},
  editor = {Jordan, Michael I.},
  publisher = {Kluwer}, address = {Dordrecht}, pages = {355--368}, year = {1998}
}
@article{JordanEtAl1999,
  author = {Jordan, Michael I. and Ghahramani, Zoubin and Jaakkola, Tommi S. and Saul, Lawrence K.},
  title = {An Introduction to Variational Methods for Graphical Models},
  journal = {Machine Learning},
  volume = {37}, number = {2}, pages = {183--233}, year = {1999}
}
@article{Csiszar1984,
  author = {Csisz{\'a}r, Imre},
  title = {Sanov Property, Generalized {I}-Projection and a Conditional Limit Theorem},
  journal = {The Annals of Probability},
  volume = {12}, number = {3}, pages = {768--793}, year = {1984}
}
@article{Petz1988,
  author = {Petz, D{\'e}nes},
  title = {Sufficiency of Channels over von {N}eumann Algebras},
  journal = {The Quarterly Journal of Mathematics},
  volume = {39}, number = {1}, pages = {97--108}, year = {1988}
}
@article{DiaconisZabell1982,
  author = {Diaconis, Persi and Zabell, Sandy L.},
  title = {Updating Subjective Probability},
  journal = {Journal of the American Statistical Association},
  volume = {77}, number = {380}, pages = {822--830}, year = {1982}
}
@article{Csiszar1975,
  author = {Csisz{\'a}r, Imre},
  title = {{I}-Divergence Geometry of Probability Distributions and Minimization Problems},
  journal = {The Annals of Probability},
  volume = {3}, number = {1}, pages = {146--158}, year = {1975}
}
@article{HalmosSavage1949,
  author = {Halmos, Paul R. and Savage, L. J.},
  title = {Application of the {R}adon--{N}ikodym Theorem to the Theory of Sufficient Statistics},
  journal = {The Annals of Mathematical Statistics},
  volume = {20}, number = {2}, pages = {225--241}, year = {1949}
}
@article{Skibinsky1967,
  author = {Skibinsky, Morris},
  title = {Adequate Subfields and Sufficiency},
  journal = {The Annals of Mathematical Statistics},
  volume = {38}, number = {1}, pages = {155--161}, year = {1967}
}
@article{Fritz2020,
  author = {Fritz, Tobias},
  title = {A Synthetic Approach to {M}arkov Kernels, Conditional Independence and Theorems on Sufficient Statistics},
  journal = {Advances in Mathematics},
  volume = {370}, pages = {107239}, year = {2020}
}
@article{CourtadeWeissman2014,
  author = {Courtade, Thomas A. and Weissman, Tsachy},
  title = {Multiterminal Source Coding Under Logarithmic Loss},
  journal = {IEEE Transactions on Information Theory},
  volume = {60}, number = {1}, pages = {740--761}, year = {2014},
  doi = {10.1109/TIT.2013.2288257}
}
```

Verification: GenestZidek1986 pages, CourtadeWeissman2014, Skibinsky1967, and Kechris Theorem 15.1 (Lusin-Souslin) were checked by web search today. The others are standard records given from memory; the lead should spot-check Abbas2009 and Lojasiewicz1963 pages. Existing keys reused: Gray1990, CoverThomas2006, Kallenberg2021, KullbackLeibler1951, AyJostLeSchwachhofer2015, BissiriHolmesWalker2016, Friston2010, BleiEtAl2017.

## NEW LABELS

Main: lem:measurable-bundle, def:bundle-morphism, eq:bundle-morphism, prop:bundle-morphism, prop:record-covariance, sec:content-laws, def:content-law, eq:content-law, prop:content-law, eq:content-projected, prop:loop-blindness (ch1); rem:dyn-product, rem:dyn-self-sources, eq:dyn-prior-learning, prop:dyn-compensation, eq:dyn-compensation, prop:dyn-global-flow, prop:dyn-row-relaxation, prop:dyn-morse, def:dyn-restricted-flow, eq:dyn-gaussian-flow, prop:dyn-gaussian-noninvariance, def:dyn-chart-flow, prop:dyn-chart-flow, prop:dyn-nonreciprocal, eq:dyn-three-cycle, prop:dyn-label-model, eq:dyn-label-model, thm:dyn-registered, cor:dyn-frozen-source, eq:dyn-frozen-source, def:dyn-evidence-potential, eq:dyn-evidence-potential (ch3); cor:pythagorean, eq:pythagorean (ch4); prop:variational-history, eq:variational-history (record appendix). Companion: comp:eq-generative-model.

For other agents: ch7/ch9/ch10/ch12 can cite prop:dyn-chart-flow for local well-posedness, prop:dyn-gaussian-noninvariance/def:dyn-restricted-flow for the Gaussian scope, and def:dyn-evidence-potential for record persistence.

## AUTHOR DECISIONS NEEDED

1. Structure group: ch1 now states one frame per agent inducing both sectors, with G = G_q x G_s required for independent sector changes. Chapters using independent sector frames (ch2, ch5) must declare which; confirm this is the intended default.
2. Whether to import the Hopf-bundle holonomy record witness (from the unpublished design note, restated locally) as an appendix; ch1 now has the content law and prop:loop-blindness it would rest on.
