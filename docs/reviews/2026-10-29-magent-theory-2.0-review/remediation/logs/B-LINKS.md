# B-LINKS remediation log (2026-10-29)

Files edited: `chapters/17_communication_links.tex`, `chapters/18_link_dynamics.tex`, `chapters/19_curvature_hierarchy.tex`, `companion/c17_communication_links.tex`, `companion/c18_link_dynamics.tex`, `companion/c19_curvature_hierarchy.tex`, `Item_1_communication_links.tex`, `Item_2_link_dynamics.tex`, `Item_3_curvature_hierarchy.tex`, `item1_refs.bib`. Not edited: `item1_checks/`, `item2_checks/`, `item3_checks/`, `item2_refs.bib` (its three formerly orphaned entries are now cited). Nothing compiled. Scripts and outputs: `remediation/scripts/B-LINKS/` (`b_links_checks.py` and output, copies of the D25 sensitivity scripts with rerun output, `crossref_output.txt`, the edit scripts, `lint.py`, `citecheck.py`).

Executed checks: exact-rational straight-line quartic and near-cancellation (b_links_checks a, b); the two-agent plug-in instability eigenvalue (2q-1)/(q+1) (c, also derived by hand in the text); monotone law-level covariance iteration from 0 and from the upper bound (d); D25 sensitivity rerun reproduces published c = -0.105655 and gives c = +0.0229 at A_omega = 1, +0.0449 at P0 = 1.5 (float64, not validated). All new BibTeX data checked against Crossref (crossref_output.txt). Lint: no `\,`, `\;`, `\!`, em dashes or banned words in the edited files; environments and braces balanced; every `\cref` in the chapters resolves; missing citation keys are exactly the BIB ENTRIES below.

## Findings

| ID (lens) | Status | Change (file, label) |
| :-- | :-- | :-- |
| H15 / S3-H05 (D26-CURV-01) | applied (decision-independent part) | ch19 `it3:induced`: weak-anchor limit stated "within this tangent model"; small-noise limit R = rho Rbar at fixed P added with proof; new remark `it3:weak-anchor-determinant` (consistency condition, common-mode spread, exact twisted record, Schur-complement crossover `it3:determinant-crossover`, determinant dominates at weak anchors and charges omega c = 0, small-noise window); intro, line 25, `it3:open-fluctuation`; c19 section 2; Item 3. Regime choice left open (decision 13). |
| S3-M14 (D22-COMM-01) | applied (chapter) | ch17 `it1:identifiability` known-code hypothesis; converse (only the effective code identified); `it1:code` shared-language sentence rewritten as a declared hypothesis; c17. Abstract and `def:shared-model` for other owners. |
| S3-M15 (D22-COMM-02) | applied | ch17 remark after `it1:bandwidth`: data processing only for one common kernel, counterexample. Item 1 same. |
| S3-X21 (D22-COMM-03) | applied as precise remark | ch17 new remark `it1:compact-stabilizer` (O(K), SO(K) stabilizers, identifiable dimension r(2K-r-1)/2, short proof). Learning counts for so(K) deferred. |
| S3-L26 (D22-COMM-04, D28-ITEMS-09) | applied | ch17 `it1:frustration` strong connectivity, base point, acyclic counterexample; Item 1. |
| S3-L27 (D22-COMM-05, D28-ITEMS-16) | applied | ch17 `it1:dissonance` compact versus GL labels, unipotent example, dressed defect as joint invariant, extension of `prop:int-loop-defect` to walks; Item 1 (plaquette as declared elementary cycle). |
| S3-L28 (D22-COMM-06) | applied | c17 loop section: code-rank qualifier. |
| S3-L29 (D22-COMM-07) | applied | ch17 after `it1:loss`: loss evaluated at the true dictionary, not constant on the stabilizer coset. |
| S3-L30 (D22-COMM-08 a, b) | applied | ch17 `it1:exponential` closing sentence; line after `it1:loop-comparisons` (effective code identified, dictionary modulo stabilizer). |
| S3-L31 (D22-COMM-08 c, D23-LEARN-11) | applied in part | ch17: noiseless gain G -> O_0; reference rechoice h -> k (lines 30, 307); completion matrices -> C_T, C_S (sans serif); projection Q -> Q_D; sampling rates kappa_e -> upsilon_e; incidence operator -> B_Gamma; temperature T removed from the bandwidth remark; c17 A_ij -> hat A_ij, Q -> Q_D. S, C, nu left (no rename requested by the fixes). Notation appendix entries for other owners. |
| S3-L32 (D22-COMM-10) | applied | (a) location Cramer-Rao with finite-covariance hypothesis and inline proof (no Stam citation, equality case derived); (b) Louis replaced by the total-variance argument with Louis cited for the identity (ch17, c17); (c) frustration attribution restricted to Bandeira et al., Singer-Wu for the connection Laplacian (ch17, c17); (d) Bishop 2006 Section 2.3.1 added beside Kalman. |
| S3-M16 (D23-LEARN-01), S4-L30 (XC5-LIT-10) | applied | ch17: synchronization reading of `it1:consensus` (Singer 2011, Gao et al. 2021); loop prior as cycle-consistency penalty; synchronizability remark after `it1:exact-flat`; coboundary/Hodge attribution (Lim 2020); cellular sheaves (Hansen-Ghrist 2019); ch19 `it3:quotient` (Hansen-Ghrist 2019 Thm 3.1, Eckmann 1944, Lim 2020); c17, c19. |
| S3-L33 (D23-LEARN-03) | applied | ch17 `it1:learning`(iii) clause corrected (compares with sigma^2, lambda > 0, never exceeds prior marginal). |
| S3-L34 (D23-LEARN-04) | applied | ch17 new part (vi) of `it1:learning` (every PD anchor, penalty-method proof); `it1:complementary`(ii) for every PD anchor with new proof; `it1:complete-exchange` for every PD anchor. |
| S3-L35 (D23-LEARN-05) | applied | ch17 `it1:tangent-model`: walks listed once up to cyclic rotation, one form and base; multiplicity remark; ch18 `it2:model` same convention. |
| S3-L36 (D23-LEARN-06) | applied | ch17 after invariance paragraph: Sigma fixed data; log normalizer retained if variable. |
| S3-L37 (D23-LEARN-07) | applied | ch17 `it1:open-learning` reworded (limits, non-commuting order, constraint-first misspecified limit); Item 1. |
| S3-X23 (D23-LEARN-08) | deferred (extension) | One qualitative sentence in `it1:open-learning`; rate theorem not added. |
| S3-X42 (D23-LEARN-09) | applied | ch17 three-agent formula `it1:three-agent-count` replaces the K = 3 example, with converse cases. |
| S3-X43 (D23-LEARN-10) | applied | ch17 new remark `it1:arborescence-loops` (gamma^2 counterexample, arborescence loop family with proof). |
| S3-L38 (D23-LEARN-12) | applied | ch17 tangent model declared as the global picture; `it1:owned` cross-references `it2:echo`. |
| S3-L39 (D23-LEARN-13) | applied | ch17 prose after `it1:pair`; Item 1. |
| S4-M08 (XC1-GAUGE-04) | applied | ch17 invariance paragraph (common reference presentation, O(K) residual group, dressed forms covariant); ch18 `it2:model`; ch19 setting. |
| S4-L06 (XC1-GAUGE-08) | applied (chapter part) | ch17 `it1:why-group` first reason; "pure-gauge" -> coboundary configurations after `it1:unreachable` and in `it1:exact-flat`. Item 7 part for B-IT7. |
| S4-M18 (XC5-LIT-01) | applied | ch17 closing related-work paragraph restored (Friston-Frith, Friston et al., van Enk, Warglien-Gardenfors, Zaslavsky, Amornbunchornvej) with what is new. |
| S4-M22 (XC5-LIT-05), S3-X24 (D28-ITEMS-06) | applied (A); (B), (C) posed | ch17 `it1:learning`(i) general disintegration statement; partial-identification attribution after the proof (Poirier, Gustafson, Dawid); Gaussian-scope paragraph; consistency step left open. |
| XC5-LIT-03 (ch17/19 parts) | applied | plaquette and Gaussian lattice gauge action cite Wilson 1974 (ch17); Forman/Kenyon in `it3:open-fluctuation`. ch15/16 parts for B-CTX. |
| S4-M27 (XC5-LIT-06) | deferred (extension); remark added | `it3:open-fluctuation` cites the spanning-forest expansion; calculation not done. |
| XC5-LIT-09 / S4-L29, S3-L49 (D24-LDYN-05, D28-ITEMS-20) | applied | ch18 Davidson 1973/1974, Quine after `it2:gauge`; Weiss-Freeman and DeMarzo et al. after `it2:echo`; "persuasion" defined against Kamenica-Gentzkow in `it2:split`; ch17 charity (anchor) and Quine (stabilizer), interpretive. |
| XC5-LIT-12, S3-L57 (D25-CRIT-09) | applied | ch18 attribution paragraph at the start of `it2:ch-critical-mode` (Golubitsky-Schaeffer 1985, Landau reading, Bizyaeva et al. 2023); Griffiths 1970 for the tricritical surface. |
| XC5-LIT-13 (my parts) | applied | ch19 `it3:rigid` logarithmic pool (GenestZidek1986); ch17:165 handled under S3-L32. |
| XC5-LIT-16 / S4-L41 | applied as one sentence (extension) | `it1:open-codes` information-design remark (Kamenica-Gentzkow). |
| XC5-LIT-07 (item1_refs) | applied | `Amornbunchornvej2025` updated to Minds and Machines 36(3) art. 36 (2026), DOI; key kept. Vaswani2017 pages added. |
| D02-LINKS-09 (b) | applied (ch17 part) | `it1:dictionary` cites `def:int-group-links`. Rest for B-FOUND. |
| D28-ITEMS-17 | applied | ch17 rank-bound and "formal content" sentences recast as Gaussian instances of `it1:reconstruction`. |
| S3-X50 (GAP05-S3-05) | applied | ch17 `it1:location`: sqrt f in H^1 (Laplace included), DQM (van der Vaart 1998, Ch. 7), positive definiteness automatic; uniform noise remark; c17. |
| D22-COMM ext (kalman rank) | applied | `it1:kalman` without full row rank. |
| S3-M21 (GAP05-S3-01) | applied | ch18 EM fixed-point condition `it2:em-stationarity` (Dempster-Laird-Rubin) replaces moment match; `it2:open-general` wording; c18; Item 2. |
| S3-M22 (D25-CRIT-01) | applied | ch18 remark after the critical-mode theorem: sign of c data dependent, float64 example c ~ +0.023 at A_omega = 1 (rerun here), supercritical case; "for these reference data" in theorem, open problem, c18. Summaries elsewhere for C-SUM. |
| S3-X27 (D25-CRIT-02) | applied | ch18 paragraph after the theorem: general kinetic metric, pullback metric reproduces full relaxation, (D+4)/D as row kinetic cost; c18. |
| S3-X28 (D25-CRIT-03) | applied | ch18 `it2:coupled-global` now concludes convergence to one stationary point (Lojasiewicz, angle condition); post-theorem sentence; c18. |
| S3-X29 (D25-CRIT-04) | deferred (extension) | Not added beyond existing open problem. |
| S3-L55 (D25-CRIT-05) | applied | ch18 and c18 straight-line value 6368971071507328/655026269697361 and reversal (exact rational check). |
| S3-X45 (D25-CRIT-06) | deferred | Not verified here; no text. |
| S3-X46 (D25-CRIT-07) | applied | ch18 jets-uniqueness remark and energy proof of instability at T_c; Ruelle kept (XC5-LIT-12 confirmed hypotheses). |
| S3-L56 (D25-CRIT-08) | applied (local notation sentence) | ch18 notation sentence at the start of the critical section. |
| S3-L48 (D24-LDYN-04), D05-OBJ-06 (ch18 part) | applied | ch18 after `it2:per-agent-model`: record-likelihood model, not `def:dyn-per-agent`, factors averaged separately. Outlook and ch03 for other owners. |
| S3-L50 (D24-LDYN-06) | applied | ch18 `it2:law-potential` joint measurability of the receiver kernel. |
| S3-L51 (D24-LDYN-07, D28-ITEMS-18) | applied | ch18 `it2:split`(ii) S' = S, plug-in contrast; Item 2. |
| S3-L52 (D24-LDYN-08) | applied | "-r" -> "-n" (ch18, Item 2); c18 r_e -> n_e; `it2:gauge`(iii) cites `it1:sheaf-count`, one shift per weakly connected component. |
| S3-L53 (D28-ITEMS-08) | applied | Item 2 Gibbs-row sentence with row entropy. |
| S3-L54 (D28-ITEMS-15) | applied | Item 2 status line and `it2:open-reciprocal`. |
| S3-X44 (D24-LDYN-09) | applied | ch18 `it2:certainty`(iii) unique global minimizer, dictionary variance allowed, proof with explicit derivative; c18; Item 2. |
| S3-X25 (D24-LDYN-01) | applied in part | (d) part: `it2:per-agent-flow`(iv) "fixed points, unique when H invertible"; two-agent analytic instability example in `it2:open-convergence` (hand-derived, checked). Law-level contraction theorem and in-degree-two example deferred. |
| S3-X26 (D24-LDYN-02, D28-ITEMS-05) | applied | ch18 new `it2:covariance-fixed-point` with proof; `it2:per-agent-flow` hypothesis, ring remark, `it2:echo`(iii), `it2:open-convergence` updated; Item 2 open problem. |
| S3-X22 (GAP05-S3-02) | applied | ch18 `it2:gauge-general`: transitive groups; SO(K) absorbability inequality/equality; so(K) tangent criterion. |
| S4-L04 (XC3-SCOPE-05) | applied | c18 Gaussian-target caveat for factor means. |
| XC4-CONS-08 (ch18 part) | applied | ch18 agent-dependent temperatures extend `def:dyn-potential`. |
| S3-M23 (D26-CURV-03) | applied | see S3-M16 row; ch19 line 33 and `it3:quotient`; c19; Item 3. |
| S3-L58 (D26-CURV-06) | applied | ch19 `it3:flat-wall` identity on connecting paths, counterexample, pairing-graph criterion. |
| S3-L59 (D26-CURV-07) | applied | ch19 after `it3:induced`: Lambda_omega belongs to the exact marginal. |
| S3-L60 (D26-CURV-09) | applied | ch19 `it3:rigid` bridge prior as normalized restriction, wall terms. |
| S3-L61 (D26-CURV-10) | applied (i, ii, iv, v) | ch19 intro, line 25, finite-anchor remark; c19 cross-scale factors, section retitled. (iii) abstract for C-SUM. |
| S3-L62 (GAP05-S3-03) | applied | ch19 Lebesgue measure, g^{-|I|} factor, explicit bridge integral, Borel likelihood; c19. |
| S3-L63 (GAP05-S3-04) | applied | ch19 `it3:general-induced` dominating measure, joint measurability, product prior, version dependence. |
| S3-X47 (D26-CURV-05, D28-ITEMS-19) | applied | ch19 rank formula `it3:rank-general` (dim H^1 for full-row-rank codes) with proof; injective corollary; square codes in the ch18 class; condensation criterion (Gleiss et al. 2003). Selective-code example not added. |
| S3-X48 (D26-CURV-08) | applied | ch19 `it3:rigid` normalized soft law converges at O(1/tau), proof extended; c19. |
| S3-X30 (D26-CURV-04) | applied | ch19 `it3:dissonance` exact product H = I + h c^T with recursion; closing paragraph. |
| E06 / D26-CURV-02 | deferred (extension) | Only the fixed-theta exactness used inside `it3:weak-anchor-determinant`. |
| S2-L10 (D10-FORM-11) | applied (ch19 part) | `it3:open-formation` specializes `inh:formation-open`. ch05 sentence for A-STD. |
| S4-M28 (XC6-PROG-16) | applied as refinement sentence | `it3:open-formation`. |
| S3-M17 (D23-LEARN-02, D28-ITEMS-01) | applied | Item 1 `it1:owned` uses the chapter wording plus the law-level contrast. |
| S3-M18 (D28-ITEMS-02) | applied | Item 1 remark after `it1:pair` and Status sentence. |
| S3-M19 (D28-ITEMS-03) | applied | Item 1 remark after `it1:bandwidth`. |
| S3-M20 (D28-ITEMS-04), S4-M04 (XC4-CONS-04) | applied | Item 1 lines 62, 65, 257, 286, 304, 309, 425, 476, 512, 529 brought to chapter wording; Item 2 definition and open problem; status notes at the top of Items 1-3 declare the chapters authoritative for shared labels and list unported refinements. README and integration record for other owners. |
| S3-L40 (D28-ITEMS-07) | applied | Item 1 `it1:why-group`. |
| S3-L41 (D28-ITEMS-10) | applied | Item 1 Status scope sentence. |
| S3-L42 (D28-ITEMS-11) | applied | Item 1 `it1:heads`. |
| S3-L43 (D28-ITEMS-12) | applied | Item 1 Status consensus sentence. |
| S3-L44 (D28-ITEMS-13) | applied | Item 1 magnitudes 1.86 -> 0.261, 3.7 -> 1.74 (recorded script output). |
| S3-L45 (D28-ITEMS-14) | deferred | Brief restricts script changes to changed chapter values; no chapter value tested by the scripts changed. New tests not added. |
| D28-ITEMS-17 (Item 1 parts) | applied in part | line 178 sentence removed, 309 replaced; 122/146 left (chapter authoritative). |
| XC4-CONS-05 (Items 1-3 parts) | applied in part | integration-history prefaces replaced by status notes; numeric check pointers kept pending decision 12. |
| XC4-CONS-10 (item bibs) | applied | item2 entries now cited; Amornbunchornvej updated. references.bib parts for C-SUM. |
| S4-M06 (XC7-IMPL-02), S4-M29, S4-M30 | deferred | Implementation alignment in VFE_4.0; outside ownership. |
| D24-LDYN-03 | for other owners | Item 7 and future_TODO wording. |
| XC2-FRAME-08 | no action here | Outlook table of open problems (C-SUM). |
| D05-OBJ-12 | deferred (extension) | |

## BIB ENTRIES

All checked against Crossref (`scripts/B-LINKS/crossref_output.txt`); van der Vaart chapter locator and Hansen-Ghrist 2019 section/theorem locators follow the lens reports (GAP05-S3-05, D26-CURV-03).

```bibtex
@article{Singer2011,
  author  = {Singer, A.},
  title   = {Angular synchronization by eigenvectors and semidefinite programming},
  journal = {Applied and Computational Harmonic Analysis},
  volume  = {30}, number = {1}, pages = {20--36}, year = {2011},
  doi     = {10.1016/j.acha.2010.02.001}
}
@article{HansenGhrist2019,
  author  = {Hansen, Jakob and Ghrist, Robert},
  title   = {Toward a spectral theory of cellular sheaves},
  journal = {Journal of Applied and Computational Topology},
  volume  = {3}, number = {4}, pages = {315--358}, year = {2019},
  doi     = {10.1007/s41468-019-00038-7}
}
@article{GaoBrodzkiMukherjee2021,
  author  = {Gao, Tingran and Brodzki, Jacek and Mukherjee, Sayan},
  title   = {The geometry of synchronization problems and learning group actions},
  journal = {Discrete \& Computational Geometry},
  volume  = {65}, number = {1}, pages = {150--211}, year = {2021},
  doi     = {10.1007/s00454-019-00100-2}
}
@article{Lim2020,
  author  = {Lim, Lek-Heng},
  title   = {Hodge {L}aplacians on graphs},
  journal = {SIAM Review},
  volume  = {62}, number = {3}, pages = {685--715}, year = {2020},
  doi     = {10.1137/18M1223101}
}
@article{Eckmann1944,
  author  = {Eckmann, Beno},
  title   = {Harmonische {F}unktionen und {R}andwertaufgaben in einem {K}omplex},
  journal = {Commentarii Mathematici Helvetici},
  volume  = {17}, number = {1}, pages = {240--255}, year = {1944},
  doi     = {10.1007/BF02566245}
}
@article{Poirier1998,
  author  = {Poirier, Dale J.},
  title   = {Revising beliefs in nonidentified models},
  journal = {Econometric Theory},
  volume  = {14}, number = {4}, pages = {483--509}, year = {1998},
  doi     = {10.1017/S0266466698144043}
}
@article{Gustafson2010,
  author  = {Gustafson, Paul},
  title   = {Bayesian inference for partially identified models},
  journal = {The International Journal of Biostatistics},
  volume  = {6}, number = {2}, pages = {Article 17}, year = {2010},
  doi     = {10.2202/1557-4679.1206}
}
@book{vanderVaart1998,
  author    = {van der Vaart, A. W.},
  title     = {Asymptotic Statistics},
  publisher = {Cambridge University Press}, address = {Cambridge}, year = {1998},
  doi       = {10.1017/CBO9780511802256}
}
@article{KamenicaGentzkow2011,
  author  = {Kamenica, Emir and Gentzkow, Matthew},
  title   = {Bayesian persuasion},
  journal = {American Economic Review},
  volume  = {101}, number = {6}, pages = {2590--2615}, year = {2011},
  doi     = {10.1257/aer.101.6.2590}
}
@article{Davidson1974,
  author  = {Davidson, Donald},
  title   = {Belief and the basis of meaning},
  journal = {Synthese},
  volume  = {27}, number = {3--4}, pages = {309--323}, year = {1974},
  doi     = {10.1007/BF00484597}
}
@article{DeMarzoVayanosZwiebel2003,
  author  = {DeMarzo, Peter M. and Vayanos, Dimitri and Zwiebel, Jeffrey},
  title   = {Persuasion bias, social influence, and unidimensional opinions},
  journal = {The Quarterly Journal of Economics},
  volume  = {118}, number = {3}, pages = {909--968}, year = {2003},
  doi     = {10.1162/00335530360698469}
}
@article{DempsterLairdRubin1977,
  author  = {Dempster, A. P. and Laird, N. M. and Rubin, D. B.},
  title   = {Maximum likelihood from incomplete data via the {EM} algorithm},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {39}, number = {1}, pages = {1--22}, year = {1977},
  doi     = {10.1111/j.2517-6161.1977.tb01600.x}
}
@book{GolubitskySchaeffer1985,
  author    = {Golubitsky, Martin and Schaeffer, David G.},
  title     = {Singularities and Groups in Bifurcation Theory, Volume {I}},
  series    = {Applied Mathematical Sciences}, volume = {51},
  publisher = {Springer}, address = {New York}, year = {1985},
  doi       = {10.1007/978-1-4612-5034-0}
}
@article{Griffiths1970,
  author  = {Griffiths, Robert B.},
  title   = {Thermodynamics near the two-fluid critical mixing point in {He}$^3$-{He}$^4$},
  journal = {Physical Review Letters},
  volume  = {24}, number = {13}, pages = {715--717}, year = {1970},
  doi     = {10.1103/PhysRevLett.24.715}
}
@article{BizyaevaFranciLeonard2023,
  author  = {Bizyaeva, Anastasia and Franci, Alessio and Leonard, Naomi Ehrich},
  title   = {Nonlinear opinion dynamics with tunable sensitivity},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {68}, number = {3}, pages = {1415--1430}, year = {2023},
  doi     = {10.1109/TAC.2022.3159527}
}
@article{Forman1993,
  author  = {Forman, Robin},
  title   = {Determinants of {L}aplacians on graphs},
  journal = {Topology},
  volume  = {32}, number = {1}, pages = {35--46}, year = {1993},
  doi     = {10.1016/0040-9383(93)90035-T}
}
@article{Kenyon2011,
  author  = {Kenyon, Richard},
  title   = {Spanning forests and the vector bundle {L}aplacian},
  journal = {The Annals of Probability},
  volume  = {39}, number = {5}, pages = {1983--2017}, year = {2011},
  doi     = {10.1214/10-AOP596}
}
@article{GleissLeydoldStadler2003,
  author  = {Gleiss, Petra M. and Leydold, Josef and Stadler, Peter F.},
  title   = {Circuit bases of strongly connected digraphs},
  journal = {Discussiones Mathematicae Graph Theory},
  volume  = {23}, number = {2}, pages = {241--260}, year = {2003},
  doi     = {10.7151/dmgt.1200}
}
```

Existing keys reused: Wilson1974, Dawid1979, GenestZidek1986, Bishop2006, Lojasiewicz1963, AbsilMahonyAndrews2005, Ruelle1973Symmetry, Berk1966, Louis1982, Davidson1973, Quine1960, WeissFreeman1999, and the item1_refs related-work keys. Gleiss et al. end page 260 is from the lens; Crossref gives only the first page.

## NEW OR CHANGED LABELS

New: `it1:compact-stabilizer`, `it1:three-agent-count`, `it1:arborescence-loops` (ch17); `it2:covariance-fixed-point`, `it2:em-stationarity` (ch18); `it3:rank-general`, `it3:weak-anchor-determinant`, `it3:determinant-crossover` (ch19). Each unique across all .tex files. No label renamed or removed. The lead's report of a dangling `it3:weak-anchor-determinant` was transient: the reference (ch19:25) was written one edit before the defining remark (ch19:130); both are now present and the only references are in ch19.

## FOR OTHER OWNERS

- C-SUM: abstract sentence on grounded identification needs the known-code hypothesis (D22-COMM-01); abstract "an affine cost" -> "a quadratic cost of an affine coded residual" (D26-CURV-10 iii); README, AUTHORING:91/93, outlook and future_TODO describe the pitchfork as subcritical without "for these data" (D25-CRIT-01) and README says the working documents "retain their earlier scope" (D28-ITEMS-04; suggest: the Items now carry status notes and the chapters are authoritative); future_TODO:96 "certainty loss" (D24-LDYN-03); notation appendix entries for codes, dictionaries, effective codes, stabilizers, gain O_0 and Q_D (D22-COMM-08); outlook should name both per-agent constructions (D24-LDYN-04); merge BIB ENTRIES; AUTHORING:91 could mention the EM form of source-role stationarity and the H15 regime caveat.
- B-FOUND: `def:shared-model` has no code component and is not "up to frame" (D01-FOUND-05, D22-COMM-01); `prop:int-loop-defect` could be stated for arbitrary based walks (D22-COMM-05); naturality versus active equivariance split in ch02 (XC1-GAUGE-02).
- B-DYN: chapter 03 per-agent picture as a class including the ch18 record-likelihood model (D05-OBJ-06).
- A-STD: ch05:1117 "No parent in the later constructions acts as a source" contradicts ch19 top-down factors (D10-FORM-11).
- B-CTX: Wilson 1974 citations at 15:101 and ch16 (XC5-LIT-03).
- B-IT7: Item 7 "trivial holonomy"/"frame changes" (XC1-GAUGE-08 c) and "Item 2 found that disagreement makes sources less certain" (D24-LDYN-03).
- Lead: VFE_4.0 integration record sentence "the original Item 1-3 working documents receive the same corrections" (D23-LEARN-02, D28-ITEMS-04).

## AUTHOR DECISIONS NEEDED

- Decision 13 (H15): which regime defines the cohomological part of `it3:induced`. Both limits are now stated; the text says the weak-anchor statement is internal to the tangent model and the small-noise window is the consistent one.
- Decision 12: whether the working documents keep their numerical-check pointers and references to the main manuscript (XC4-CONS-05), whether to retire them, and whether `item1_checks` to `item3_checks` should gain tests of the corrected claims (D28-ITEMS-14).
