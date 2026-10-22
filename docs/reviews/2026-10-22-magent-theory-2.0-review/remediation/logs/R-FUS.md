# R-FUS remediation log (chapters 6, 7; companion c07, c08)

Files edited: `chapters/06_fusion.tex`, `chapters/07_smooth_closure.tex`, `companion/c07_fusion.tex`, `companion/c08_smooth.tex`. Not compiled (per brief). Checks: `remediation/scripts/R-FUS/rfus_checks.py`, output `rfus_checks.out.txt`: chi inertness (residual 0), variance derivative n(a-1)/2 per sector, finite-time synchronization bound on three random multiplicity data (sync 0.70-0.97 vs bounds 2.9-4.0), dyadic depth-3 history in the predicted order, spectrum injectivity for A=(0.6,0.4), CGF-difference invariance 2.5e-16, CGF flow residual 1.8e-16, range width ratio = e^{-t}, private-transmitting KL = (1/2)log 3, proximal tie threshold 2h lambda. Numerics support; proofs are in the text.

| ID | Status | Change (file, label) |
|---|---|---|
| L05-FUS-01 | applied | New prop `fus:sync` (finite-time complete synchronization for every datum and multiplicity vector, bound `fus:sync-time`); intro states finite coalescence tree ending in consensus; convergence paragraph now unconditional (06). c07 paragraph. |
| L05-FUS-02 | applied | 07 intro and paragraph after `smooth:split-dynamics`: private sector decoupled, explicit exponential arc, open-loop forcing, solvable baseline; coupled case stated as unavailable in `smooth:formation-open`. c08 paragraph. |
| L05-FUS-03 | applied | 07 after `smooth:spectrum-definition`: equal spectra iff equimeasurable log densities; injective for generic finite references, example A=(0.6,0.4). Used "equimeasurability quotient" instead of "orbit of Aut(V,A)": on atomless spaces equimeasurable functions need not be related by a bimeasurable reference-preserving bijection, and on finite alphabets not always by an atom permutation (e.g. A=(1/2,1/4,1/4)). c08 paragraph. |
| L05-FUS-04 | applied | 06 after `fus:links`: pure gauge V_iV_j^{-1}, chi inert; fixed-link remark corrected to lambda/(1-chi); intro no longer claims transport content. c07 corrected. |
| L05-FUS-05 | applied | 06 after `fus:law-family`: fixed variance is a constraint; derivative n(a-1) (both sectors) at v=1/n; releasing destroys multiplicity encoding. c07 sentence. |
| L05-FUS-06 | applied | New prop `smooth:cgf-completion`, eq `smooth:cgf-flow` (coarser autonomous completion; product example). |
| L05-FUS-07 | applied | New lemma `fus:isolated-pair` and cor `fus:open-two-stage` (two-stage pattern contains an open subset of R^8, proof by contraction plus isolated-pair decay). Class described as invariant affine subspace of codimension r-1. |
| L05-FUS-08 | applied; joint spectrum deferred | New remark `smooth:channel-variants`: fixed source-independent reconstruction still closes (private rate doubles); exact counterexample for a private-transmitting identity channel. Joint-spectrum closure for identity private channels stated as not established. |
| L05-FUS-09 | applied | New remark `smooth:exponential-family`; Hoelder gives the private solution for every kappa ~ A; public part keeps (H1) or a bound on ess sup_z H(z,0). See conflict decision. |
| L05-FUS-10 | applied | 06: B_n named as conditional law given the sufficient statistic (after `fus:message-kernel`); new remark `fus:sufficiency` (general families, Gamma example, convexity caveat). |
| L05-FUS-11 | applied | CarrilloEtAl2011, BonaschiEtAl2015 (intro, at `fus:sticky-flow`), Brezis1973 (`fus:uniqueness`). |
| L05-FUS-12 | applied | 07: log-likelihood-ratio law, Blackwell equivalence (Blackwell1953, Torgersen1991), Renyi identity `smooth:renyi` (vanErvenHarremoes2014), exponential arc (ChenEtAl2023). |
| L05-FUS-13 | applied | `smooth:mean-force-criterion` now requires invariance of the admitted class and uniqueness of the retained equation. |
| L05-FUS-14 | applied | `fus:uniqueness` retitled "Global well-posedness, contraction, and persistence": global existence/uniqueness for every datum (Brezis), minimal-norm right derivative; paragraph on Filippov equivalence (Filippov1988, Clarke1983, PadenSastry1987) and failure of backward uniqueness. |
| L05-FUS-15 | applied | 07: hedge replaced by exact negative statement (Fisher-Rao flows preserve mutual absolute continuity; retained support moves). |
| L05-FUS-16 | applied | c07 retitled "A Nonsmooth Route to Finite Coalescence"; "recurrent fusion" removed. c08 retitled "Hidden Conditional Information Under Smooth Dynamics"; opening reframed as per-agent completion; parents only in erasure section. |
| L05-FUS-17 | applied (partial) | 06: sector index nu -> alpha, signs sigma_ij -> epsilon_ij, mass matrix M -> W, M_left/right -> n_left/right; c07 sector index sigma -> alpha; 07: model conditional lambda_{i,w} -> varkappa_{i,w}, score parts a,b -> f_1,f_2, row ratio a_jk -> theta_jk. Kept: a = e^{-t} in 07 (now unique there); kappa = a-c in 06 (cross-chapter collision remains, see X2-FRAME-08). |
| L05-FUS-18 | (a) applied, (b) deferred | (a) new cor `fus:dyadic` with condition `fus:dyadic-condition` (proof: adjacent blocks approach at speed in [2 lambda, lambda N]). (b) bounded-confidence/multiwell persistence stated in remark `fus:smooth-open` as not established. |
| L05-FUS-19 | applied | New remark `fus:prox` (proximal step `fus:prox-step`, discrete contraction, energy inequality, exact ties at gap <= 2h lambda for the pair term, Crandall-Liggett convergence, Euler chatter). |
| G2-01 | applied | New hypothesis `smooth:hypotheses` (H1)-(H3) and prop `smooth:wellposed` (i)-(iv) with proof (chart, Banach Picard-Lindeloef, fine lift); the prose at old 07:14 and "regularity needed" replaced by references to the hypotheses; both counterexample types stated in prose. |
| G2-02 | applied | `smooth:wellposed`(iv) intertwining on the common maximal interval; `smooth:spectrum-flow` now claims only that the velocity is a function of the state, followed by the uniqueness requirement; `smooth:equal-spectra` "on the common maximal interval"; swapped-spectrum sentence conditioned on uniqueness. |
| G2-03 | applied | (H1) uniform in (z,v); text explains why per-z bounds fail. |
| G2-04 | private part applied; public global bounds deferred | Range invariance and monotonicity of H in proofs of `smooth:spectrum-flow` and `smooth:wellposed`(i). Text says T* = infinity needs further estimates. |
| G2-05 | applied | Uniqueness of the weak spectral equation in `smooth:spectrum-flow` (statement and proof; AmbrosioGigliSavare2008 Sec. 8.1; normalization via f = e^r). |
| G2-06 | applied | "arbitrary dominated law shapes" removed from 07; c08 names the uniform bounded-log class and public hypotheses. |
| L06-ERA-01 (c08) | applied | c08: heterogeneity transient (matches `erase:pair-convergence`); no "persistent heterogeneous pairs" wording. |
| L06-ERA-02 (c08) | applied | c08: gate as internal clock; erasure and switch follow from declared scale ordering. |
| Lead note on ch8 status | applied | c08: null flow reproduces the pairing sequence, only switch time moves; single-stage selection with exact inheritance; parent keeps every public child state and is not a law pair (also L06-ERA-04). |
| X1-GAUGE-01 / H6 / L01-FOUND-08 (ch6 parts) | applied | See L05-FUS-04. |
| X1-GAUGE-03 (ch7 part) | applied | 07 defines public/private by the declared observable algebra (channels read public coordinates; private only via marks). ch12 wording outside my files. |
| L03-DYN-01 / X5-IMPL-03 (ch7 parts) | applied | 07 after `smooth:feedback`: composite is `def:dyn-record-interrupted`; no record term, not monotone across records, no reference-changing rule, record effect decays. c08 sentence. |
| L03-DYN-04 (07:110) | applied | 07 after `smooth:split-dynamics`: rows relax at rate T_q; Gibbs rows are the infinite-mobility limit; same stationary points, different flows. |
| L04-INH-01 (ch7 contrast) | applied | 07 final section refers to (F4). |
| L07-BND-12 (relation to ch7) | remark only | Joint-spectrum suggestion in `smooth:channel-variants`; result belongs to ch10. |
| L08-CND-06 | applied | c07 closing contrast rewritten (per lead note): ch11 schedule is coupling-independent (form:kappa-independence), ch6 schedule disappears at zero coupling; both are two-stage selections with exact inheritance for different reasons. |
| X2-FRAME-03 | applied | `fus:smooth-open`, `smooth:formation-open` converted from open problems to remarks pointing to `inh:formation-open`, `inh:smooth-no-finite-entry`, ch11/ch12 selectors. Labels kept. |
| X2-FRAME-04 | applied | 07: whether a tuple retention (or symmetry quotient) compresses is decided by (F4); no verdict on ch12 made. |
| X2-FRAME-05 | applied | 06: inheritance holds only for synchronized blocks, so the selected partition is distinguished by inheritance. 07: per-agent, no grouping. |
| X2-FRAME-08 (my parts) | applied (partial) | fus:smooth-open cites inh:smooth-no-finite-entry; c07 retitled. File order and global kappa collisions outside my authority. |
| X3-PIFB-13 | applied | 06 after `fus:coordinate-flow` and c07: multiplicity is friction weight and interaction charge, overdamped 1D pair potential, not inertia. |
| X4-PRIOR-01 (ch6 part) | applied | `sec:fus-status`: (F6) fails since the second stage occurs from every counterfactual handoff (by `fus:sync`). |
| X4-PRIOR-03 (ch6 part) | applied | Aggregation-equation and bounded-confidence citations; attention-dynamics items belong to ch12. |
| X4-PRIOR-08 | rejected for my files | 07:233-242 only states the row-ratio identity; prior learning concerns ch3/ch12. |
| X4-PRIOR-09 (ch6 part) | applied/deferred | Pure-gauge statement and pointer to a noncommuting generator; dynamical links deferred. |
| X4-PRIOR-10 (06:176) | applied | 06: "declared gradient flow, not inference in a normalized model". |
| X5-IMPL-06 (ch6 part) | applied | `fus:prox`. |
| X5-IMPL-17 | applied | Open-gap condition stated as iff, with necessity argument. |
| X6-LIT-01 (ch7 parts) | applied | Price1970 at mean force; ChenEtAl2023 at the conditional solution (Lu-Lu-Nolen, Domingo-Enrich-Pooladian not added). |
| X6-LIT-07 | applied | See L05-FUS-11/14; interaction described as nonsmooth modification with classical kernel. |
| X6-LIT-09 (07:244) | no change | Barnett-Seth citation accurate per lens; other items are ch5. |
| X6-LIT-10 | applied | See L05-FUS-12. "Standard measure" wording avoided (different normalization in Le Cam theory); text says law of the log-likelihood ratio under A. |
| X6-LIT-15 | applied | New cor `smooth:submersion`. |
| X6-LIT-19 | applied (partial) | Renyi identity, determination by Lambda on (0,1], flow `smooth:cgf-flow`, exponential-family closure. Invariant-subspace finite-closure test and marked Blackwell-class statement not added. |

## Conflict decisions (REPORT 7.1)

Fusion classification (X2-FRAME-09 vs X4-PRIOR-01 vs L05-FUS-01): compatible. The null flow (lambda = 0) is linear and invertible, so no collisions and (F5) holds on the whole admitted class; strong (F5) fails trivially (final partition constant). By `fus:sync` every parent pair fuses, so (F6b) fails while (F6a) holds; (F4) fails (identical copies; sample-level noninjectivity is reconstructed by a parameter-free kernel). Using the lead's split of (F6), chapter 6 is a two-stage selection with exact inheritance (F1, F2, F3, F5, F6a hold; F4, F6b fail). Text: `sec:fus-status`, c07 last section.

Bounded-log hypothesis (L05-FUS-09 vs G2-03): adopted G2's uniform (H1) for the general well-posedness statement; L05-FUS-09's point kept in `smooth:exponential-family`.

## BIB ENTRIES

```bibtex
@book{Brezis1973,
  author = {Br{\'e}zis, Ha{\"\i}m},
  title = {Op{\'e}rateurs Maximaux Monotones et Semi-Groupes de Contractions dans les Espaces de {H}ilbert},
  series = {North-Holland Mathematics Studies},
  volume = {5},
  publisher = {North-Holland},
  address = {Amsterdam},
  year = {1973}
}
@article{BonaschiEtAl2015,
  author = {Bonaschi, Giovanni A. and Carrillo, Jos{\'e} A. and Di Francesco, Marco and Peletier, Mark A.},
  title = {Equivalence of Gradient Flows and Entropy Solutions for Singular Nonlocal Interaction Equations in {1D}},
  journal = {ESAIM: Control, Optimisation and Calculus of Variations},
  volume = {21},
  number = {2},
  pages = {414--441},
  year = {2015},
  doi = {10.1051/cocv/2014032}
}
@article{CarrilloEtAl2011,
  author = {Carrillo, Jos{\'e} A. and Di Francesco, Marco and Figalli, Alessio and Laurent, Thomas and Slep{\v{c}}ev, Dejan},
  title = {Global-in-Time Weak Measure Solutions and Finite-Time Aggregation for Nonlocal Interaction Equations},
  journal = {Duke Mathematical Journal},
  volume = {156},
  pages = {229--271},
  year = {2011},
  doi = {10.1215/00127094-2010-211}
}
@book{Filippov1988,
  author = {Filippov, Aleksei F.},
  title = {Differential Equations with Discontinuous Righthand Sides},
  series = {Mathematics and Its Applications (Soviet Series)},
  volume = {18},
  publisher = {Kluwer Academic Publishers},
  address = {Dordrecht},
  year = {1988}
}
@book{Clarke1983,
  author = {Clarke, Frank H.},
  title = {Optimization and Nonsmooth Analysis},
  publisher = {Wiley},
  address = {New York},
  year = {1983}
}
@article{PadenSastry1987,
  author = {Paden, Brad E. and Sastry, Shankar S.},
  title = {A Calculus for Computing {F}ilippov's Differential Inclusion with Application to the Variable Structure Control of Robot Manipulators},
  journal = {IEEE Transactions on Circuits and Systems},
  volume = {34},
  number = {1},
  pages = {73--82},
  year = {1987},
  doi = {10.1109/TCS.1987.1086038}
}
@article{CrandallLiggett1971,
  author = {Crandall, Michael G. and Liggett, Thomas M.},
  title = {Generation of Semi-Groups of Nonlinear Transformations on General {B}anach Spaces},
  journal = {American Journal of Mathematics},
  volume = {93},
  number = {2},
  pages = {265--298},
  year = {1971},
  doi = {10.2307/2373376}
}
@article{HegselmannKrause2002,
  author = {Hegselmann, Rainer and Krause, Ulrich},
  title = {Opinion Dynamics and Bounded Confidence: Models, Analysis and Simulation},
  journal = {Journal of Artificial Societies and Social Simulation},
  volume = {5},
  number = {3},
  year = {2002}
}
@article{Price1970,
  author = {Price, George R.},
  title = {Selection and Covariance},
  journal = {Nature},
  volume = {227},
  pages = {520--521},
  year = {1970},
  doi = {10.1038/227520a0}
}
@article{Blackwell1953,
  author = {Blackwell, David},
  title = {Equivalent Comparisons of Experiments},
  journal = {The Annals of Mathematical Statistics},
  volume = {24},
  number = {2},
  pages = {265--272},
  year = {1953},
  doi = {10.1214/aoms/1177729032}
}
@book{Torgersen1991,
  author = {Torgersen, Erik},
  title = {Comparison of Statistical Experiments},
  series = {Encyclopedia of Mathematics and its Applications},
  volume = {36},
  publisher = {Cambridge University Press},
  address = {Cambridge},
  year = {1991}
}
@article{vanErvenHarremoes2014,
  author = {van Erven, Tim and Harremo{\"e}s, Peter},
  title = {R{\'e}nyi Divergence and {K}ullback--{L}eibler Divergence},
  journal = {IEEE Transactions on Information Theory},
  volume = {60},
  number = {7},
  pages = {3797--3820},
  year = {2014},
  doi = {10.1109/TIT.2014.2320500}
}
@misc{ChenEtAl2023,
  author = {Chen, Yifan and Huang, Daniel Zhengyu and Huang, Jiaoyang and Reich, Sebastian and Stuart, Andrew M.},
  title = {Sampling via Gradient Flows in the Space of Probability Measures},
  howpublished = {arXiv:2310.03597},
  year = {2023}
}
@book{AmbrosioGigliSavare2008,
  author = {Ambrosio, Luigi and Gigli, Nicola and Savar{\'e}, Giuseppe},
  title = {Gradient Flows in Metric Spaces and in the Space of Probability Measures},
  edition = {2},
  series = {Lectures in Mathematics ETH Z{\"u}rich},
  publisher = {Birkh{\"a}user},
  address = {Basel},
  year = {2008}
}
@book{Dieudonne1960,
  author = {Dieudonn{\'e}, Jean},
  title = {Foundations of Modern Analysis},
  publisher = {Academic Press},
  address = {New York},
  year = {1960}
}
```

Verified online this session: Bonaschi et al., Carrillo et al. (issue number omitted, sources disagree), Paden-Sastry, van Erven-Harremoes, Chen et al., Crandall-Liggett. From memory (standard references, DOIs for Price and Blackwell not rechecked): Brezis, Filippov, Clarke, Hegselmann-Krause, Price, Blackwell, Torgersen, Ambrosio-Gigli-Savare, Dieudonne. Existing keys reused: Hocking2011, Cortes2006, AyJostLeSchwachhofer2015, AyJostLeSchwachhofer2018, ShaliziCrutchfield2001, BarnettSeth2023. Other agents may propose the same keys (e.g. Brezis1973, Price1970, Blackwell1953); deduplicate.

## NEW LABELS

06: `fus:sync`, `fus:sync-time`, `fus:isolated-pair`, `fus:pair-decay`, `fus:open-two-stage`, `fus:dyadic`, `fus:dyadic-condition`, `fus:prox`, `fus:prox-step`, `fus:sufficiency`, `sec:fus-status`. (`fus:smooth-open` is now a remark; `fus:uniqueness` retitled.)
07: `smooth:submersion`, `smooth:hypotheses`, `smooth:wellposed`, `smooth:renyi`, `smooth:cgf-completion`, `smooth:cgf-flow`, `smooth:exponential-family`, `smooth:channel-variants`, `sec:smooth-records`. (`smooth:formation-open` is now a remark.)
Companion: c07 and c08 chapter titles changed; labels `comp:fusion`, `comp:smooth` kept. Uniqueness of all labels checked by grep across the three documents.

## AUTHOR DECISIONS NEEDED

1. Whether to add the global-in-time public bounds of G2-04 (bounded-kernel and identity channels), ideally as a general Fisher-flow well-posedness proposition in ch3 that chapters 7, 9, 10 and 12 can cite (G2 and L03-DYN-03 recommend this). Chapter 7 currently states local well-posedness only.
