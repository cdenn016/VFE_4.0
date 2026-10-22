# R-GEOM remediation log

Scope: `chapters/02_interactions.tex`, `chapters/02_geometry.tex`, `appendices/base_connection.tex`, `appendices/determinant_example.tex`, `supplements/s02_geometry.tex`, `companion/c02_links.tex`, `companion/c06_geometry.tex`, `companion/figures/link_transport.tex` (paths relative to the manuscript root). Nothing compiled. Verification scripts (single-threaded, each under a minute) are in `remediation/scripts/R-GEOM/`: `s1_structure_identity.py` (sympy, Aff(1) on N(mu, sigma^2): structure identity and Lie-derivative identity, residual 0), `s2_determinant.py` (ranks for proper complex Gaussian targets, canonical-scale identity residual <= 7e-15, scalar curvature -6 exactly), `s3_links_and_collective.py` (reciprocal decomposition residuals 3e-15 and 1e-14, loop-defect invariance 1e-11, agreement example, contraction identity residual 2e-15 and exact example, horizontal-defect witness), `fix_ctrl.py` and `check_labels.py` (lint: no control characters, no `\,` `\;` `\!`, balanced environments, no duplicate labels, no missing `\cref` targets).

## Findings

| ID | Status | What changed (file, label) or reason |
|---|---|---|
| L02-GEOM-01 (I2) | applied | New `geo:shape-pullback` in 02_geometry: A^F = -p^*theta, F = -p^*Omega with Omega horizontal, radical involutive and h^min basic, h^min = (pi o p)^* g_sh under free proper action (`geo:shape-metric`); proof written out. Contact example restricted to supplied connections. Citations Smale 1970, Guichardet 1984, Littlejohn-Reinsch 1997, O'Neill 1966. |
| L02-GEOM-02 (I3) | applied | New `geo:minimal-contraction` (corollary) with identity `geo:minimal-contraction-identity`, proof, worked example 2 = 4/3 + 1/2 + 1/6, telescoping remark; companion c06 gets the identity. Reconstructed by hand and numerically (s3 (d)). Use in ch. 5 left to the chapter 5 owner. |
| L02-GEOM-03 | applied | Paragraph after `geo:shape-pullback`: constant rank fails at isotropy jumps (Lewis-Nigam-Olver 2005); explicit SO(2) example A^F = -eps/(c^2+eps^2), integral -pi, concentrating to -pi delta; h^min bounded by ||dp||^2. The lens's "h^min extends continuously" was not proved and is replaced by the proved bound. Companion c06 mirrors. |
| L02-GEOM-04 | applied | After `geo:gauge-connection`: `geo:transitive-pullback`, per-agent tensor records only the supplied connection for transitive fibers; rank bound rank h^min <= dim H stated after `geo:shape-pullback`; companion c06 mirrors. |
| L02-GEOM-05 | applied | New `geo:radical-curvature` with `geo:structure-identity` and `geo:basic-criterion`, human-readable proof (I derived both identities independently; s1 confirms with residual 0 on a nonabelian action including the dg/dp term). Flat case stated. Contact example checked against the identity. |
| L02-GEOM-06 (R11) | applied in corrected form | New `geo:det-rank` in determinant appendix with `geo:det-horizontal` and proof. The lens's "cannot be an isomorphism with fewer than four agents" is true only on the stratum of maximal orbit dimension: two agents with covariances M, 2M give orbit dim 4, dim H = 4, d M|_H surjective; three agents (I, 2I, generic) give rank 4 (s2). Stated with that restriction; n = 1 always zero; four-agent explicit rational witness (I, 2I, diag(1,2), [[2,1],[1,2]]) with stabilizer the central circle, surjectivity proved by hand. Improper-law phase remark added as a deferred remark. Pointer added in 02_geometry after `geo:aggregate`. |
| L02-GEOM-07 (R11) | applied | Canonical scale kappa = det M, `geo:det-canonical`, entropy-rate reading, dilation timelike, exact soldering flat for constant kappa and R x H^3 with scalar curvature -6 for kappa = det M (s2 exact). PIFB2 flatness remark is outside the manuscript (no change). |
| L02-GEOM-08 | applied | Replaced the gluing sentence after `geo:collective`: global minimizers by partition of unity, uniqueness iff ker J = 0, class-invariant J F on the radical (follows from `geo:structure-identity`). |
| L02-GEOM-09 | applied | Dominating-measure sentence after `geo:regular-family` rewritten: different null sets allowed; finite-dimensional DQM families are dominated (proof sketch inline); DQM attributed to Le Cam. Companion c06 fixed likewise. |
| L02-GEOM-10 | applied | Appendix retitled "Connections in a Moving Presentation", intro says a_i is supplied and no link enters; 02_geometry sentence after `geo:gauge-connection` rewritten; companion c06 says a is supplied and adds the kappa / abs(det g) = 1 condition to the determinant sentence. Chapter 1 line 31 ("explicit directional-transport realization") is outside my files; see decisions. |
| X6-LIT-03 (I2) | applied | Citations as above; Berry/Provost-Vallee quantum analogue not added (interpretive, low value; deferred). Bib record LewisNigamOlver2003 superseded by new key LewisNigamOlver2005 (journal version); old key is now unused in all .tex files. |
| X6-LIT-08 | applied | As L02-GEOM-09. The "Theorems 5.1 and 5.2" locator for AJLS 2018 was not verified; citation at the Pollard paragraph left without locator. |
| X6-LIT-12 (my locations) | applied | Zamir 1998 after `geo:joint-fisher`; Kullback 1959 after `geo:transported-kl`; Reinhart 1959 and Molino 1988 for basic-tensor descent; Kagan-Shepp 2005 for the Bernoulli phenomenon. |
| X6-LIT-13 (my locations) | partially applied | LewisNigamOlver record replaced by new key. KullbackLeibler1951 locator at 02_interactions not added (not verified). |
| X6-LIT-14 (geo:regular-family part) | applied | Cencov 1982 and AJLS 2015 cited after `geo:regular-family` as the reason for the Fisher metric, with the caveat that factor weights are not fixed. The ch. 3 mobility statement is for the dynamics owner. |
| X6-LIT-17 / X1-GAUGE-10 (E10, R8) | applied as remark plus open problem | `geo:context-energy` with the elimination statement (short proof inline) and CP^{N-1} analogy marked structural; new `open:geo-gauged-flow` records the coupled flow and the ensemble route; `open:conn-derived` points to it. Dynamics deferred (new research). |
| X1-GAUGE-01 (H6) | applied | 02_geometry opening paragraph: connection does no dynamical work in ch. 3 to 12 and is a diagnostic of evolved sections; same in appendix intro and companion c06. 02_interactions: new section "Reference consistency" with `eq:int-dressed-cost` (dressing), standing `hyp:int-reference-consistency` (`eq:int-kernel-reference`), and statement that results drawing bundle-level conclusions assume it; Wilson analogy marked as display coordinates. Fusion (ch. 6) pure-gauge statement is outside my files. |
| X1-GAUGE-04 | applied (my part) | 02_interactions states that the tied group G vs G_q x G_s is a constitutive choice on which the collective connection depends. Restating ch. 11/12 curvature claims is for those owners. |
| X1-GAUGE-05 / L01-FOUND-03 | applied (my part) | 02_interactions: E_i^a defined as displayed coordinate copy of V_a; all agents in one fiber; g_i^a = rho_a(g_i) for one group, G_q x G_s for independent changes. Base-connection paragraph on agents' connections rewritten. Figure labels "Source/Receiver fiber" changed to "display". Chapter 1 presentation section is outside my files. |
| X1-GAUGE-09 / X4-PRIOR-09(a) (E9) | applied as proposition; dynamics deferred | `prop:int-loop-defect`, `eq:int-loop-defect` with proof; values 1/2 (shear) and 3/2 (example) recomputed (s3). Stated that adding it to the potential or making generators state variables defines different dynamics not studied. X4-PRIOR-09(b) (downward action) is ch. 5 scope. |
| L01-FOUND-02 (E8) | applied | `prop:int-loop-blindness` with proof; consequence for law-level costs; diagonal transformation of shared-sample pair; companion c02 mirrors. |
| L01-FOUND-06 | applied | `def:int-channel` now a jointly measurable kernel on S x E_j; measurability consequence stated; `def:int-generator` generalized to the tuple of reference laws, with `eq:int-reference-equivariance` and `eq:int-total-variation` updated. |
| L01-FOUND-07 | applied | `eq:int-dressed-cost` and the section "Reference consistency"; quiver invariants (Le Bruyn-Procesi 1990) vs traces of words (Procesi 1976) added at the loops section. |
| L01-FOUND-08 | applied | Exponential links presented as one constitutive example of `def:int-channel` (chapter intro, section title, associated channel delta_{Theta y}); values in exp(gl) subset GL^+ with Culver counterexample; "selective" removed: `prop:int-rank` retitled "Low-rank generator and invertible transport", non-selectivity stated; supplement retitled "Algebra of Exponential Links with Low-Rank Generators"; companion c02 mirrors. Not moved to the supplement (it now carries `prop:int-reciprocal` and `prop:int-loop-defect`). |
| L01-FOUND-09 | applied | `prop:int-reciprocal`, `eq:int-reciprocal`, proof, Laplace example (Omega_12 a rotation, two-cycle pure nonreciprocity), pointer in `open:conn-derived`; companion c02 mirrors. Higham locator given as Chapter 1 (corollary number unverified). |
| L01-FOUND-10 | applied (my part) | Paragraph in 02_geometry "Geometry across a retention": bundle map induced by a principal map and a phi-equivariant kernel, well-definedness proved, Delta_F >= 0 for fixed kernels. Agent type closure deferred to ch. 1/5 owners. |
| L01-FOUND-14 | applied | s02: derivative of the cost needs DQM of Theta -> Theta_# nu_j; uniform counterexample added. |
| L01-FOUND-16 (my files) | partially applied | Covariance renamed M_i^a -> Sigma_i^a in 02_interactions, s02, c02 (separates it from the content map M_ij^a and model space). Moment receiver written as lambda O_i T_j with O_i(t) = mu_i t. Content map M_ij^a kept because ch. 5 and s04 use M_ij for the same map; O, T kept because notation appendix and glossary (not mine) list them. |
| L01-FOUND-17 (my part) | applied | Associated channel delta_{Theta y} and implication of `eq:int-kernel-covariance` in `prop:int-endpoint`. Evaluator and event-kernel sentences belong to ch. 1 and the record appendix. |
| L01-FOUND-20 (my part) | applied | Culver 1966, Higham 2008, Le Bruyn-Procesi 1990, Procesi 1976 added. Kobayashi-Nomizu, Kechris, Kallenberg locators are ch. 1 scope. |
| X3-PIFB-09 (R11) | applied | Same as L02-GEOM-07 (one remark). |
| X3-PIFB-12 | applied | Paragraph after the agreement remark: weighted metric is the labeled-family Fisher metric; context-dependent weights add sum dw^2/w without changing the normal equation; correlated population laws stated as not treated. |
| X3-PIFB-15 | applied as statement | `prop:int-loop-defect` closing sentence: latent generators with their own potential or prior define different dynamics, not studied. Deferred. |
| X4-PRIOR-02 | applied (my part) | `def:int-channel`: primitive self channel is constitutive data, identity one choice; exponential-link self link identity; companion c02 mirrors. AUTHORING.md amendment and ch. 12 statement are outside my files (decision below). |
| X4-PRIOR-13 | partially applied | Horizontal-defect witness (c = 1/10 contraction with nonzero defect, c = 1/2 failure) added after `geo:signed-comparison`, derived locally. Determinant non-vacuity witness supplied by my own four-agent example in `geo:det-rank` instead of the earlier-theory example (source file not read). Isotropy criterion for principal scale maps deferred (needs definitions not present). Predictive-distortion and ch. 12 imports are other owners. |
| L03-DYN-09 | no change needed in my files | nu_i^a in ch. 2 is the canonical use; collisions are in ch. 3, 4, s03, companion c03. |
| L05-FUS-17 | no change needed in my files | Same: ch. 2 already uses a in {q,s}; collisions are in ch. 6, 7, c07. |
| X5-IMPL-09 | rejected for manuscript | Implementation finding about VFE_4.0; no manuscript change required. |
| H6, I2, I3, R8, R11 | applied through the rows above | E10 and dynamical role recorded as `open:geo-gauged-flow`. |

## BIB ENTRIES

```bibtex
@article{Culver1966,
  author  = {Culver, Walter J.},
  title   = {On the Existence and Uniqueness of the Real Logarithm of a Matrix},
  journal = {Proceedings of the American Mathematical Society},
  volume  = {17},
  number  = {5},
  pages   = {1146--1151},
  year    = {1966},
  doi     = {10.1090/S0002-9939-1966-0202740-6}
}
@article{LeBruynProcesi1990,
  author  = {Le Bruyn, Lieven and Procesi, Claudio},
  title   = {Semisimple Representations of Quivers},
  journal = {Transactions of the American Mathematical Society},
  volume  = {317},
  number  = {2},
  pages   = {585--598},
  year    = {1990},
  doi     = {10.1090/S0002-9947-1990-0958897-0}
}
@article{Procesi1976,
  author  = {Procesi, Claudio},
  title   = {The Invariant Theory of $n\times n$ Matrices},
  journal = {Advances in Mathematics},
  volume  = {19},
  number  = {3},
  pages   = {306--381},
  year    = {1976},
  doi     = {10.1016/0001-8708(76)90027-X}
}
@book{Higham2008,
  author    = {Higham, Nicholas J.},
  title     = {Functions of Matrices: Theory and Computation},
  publisher = {Society for Industrial and Applied Mathematics},
  address   = {Philadelphia},
  year      = {2008},
  doi       = {10.1137/1.9780898717778}
}
@book{LeCamYang2000,
  author    = {Le Cam, Lucien and Yang, Grace Lo},
  title     = {Asymptotics in Statistics: Some Basic Concepts},
  edition   = {Second},
  publisher = {Springer},
  address   = {New York},
  year      = {2000},
  doi       = {10.1007/978-1-4612-1166-2}
}
@book{Cencov1982,
  author    = {{\v{C}}encov, N. N.},
  title     = {Statistical Decision Rules and Optimal Inference},
  series    = {Translations of Mathematical Monographs},
  volume    = {53},
  publisher = {American Mathematical Society},
  address   = {Providence, RI},
  year      = {1982}
}
@article{Zamir1998,
  author  = {Zamir, Ram},
  title   = {A Proof of the {F}isher Information Inequality via a Data Processing Argument},
  journal = {IEEE Transactions on Information Theory},
  volume  = {44},
  number  = {3},
  pages   = {1246--1250},
  year    = {1998},
  doi     = {10.1109/18.669301}
}
@book{Kullback1959,
  author    = {Kullback, Solomon},
  title     = {Information Theory and Statistics},
  publisher = {Wiley},
  address   = {New York},
  year      = {1959}
}
@article{Reinhart1959,
  author  = {Reinhart, Bruce L.},
  title   = {Foliated Manifolds with Bundle-Like Metrics},
  journal = {Annals of Mathematics},
  volume  = {69},
  number  = {1},
  pages   = {119--132},
  year    = {1959},
  doi     = {10.2307/1970097}
}
@book{Molino1988,
  author    = {Molino, Pierre},
  title     = {Riemannian Foliations},
  series    = {Progress in Mathematics},
  volume    = {73},
  publisher = {Birkh{\"a}user},
  address   = {Boston},
  year      = {1988},
  doi       = {10.1007/978-1-4684-8670-4}
}
@article{Smale1970,
  author  = {Smale, Stephen},
  title   = {Topology and Mechanics. {I}},
  journal = {Inventiones Mathematicae},
  volume  = {10},
  pages   = {305--331},
  year    = {1970},
  doi     = {10.1007/BF01418778}
}
@article{LittlejohnReinsch1997,
  author  = {Littlejohn, Robert G. and Reinsch, Matthias},
  title   = {Gauge Fields in the Separation of Rotations and Internal Motions in the $n$-Body Problem},
  journal = {Reviews of Modern Physics},
  volume  = {69},
  number  = {1},
  pages   = {213--275},
  year    = {1997},
  doi     = {10.1103/RevModPhys.69.213}
}
@article{ONeill1966,
  author  = {O'Neill, Barrett},
  title   = {The Fundamental Equations of a Submersion},
  journal = {Michigan Mathematical Journal},
  volume  = {13},
  number  = {4},
  pages   = {459--469},
  year    = {1966},
  doi     = {10.1307/mmj/1028999604}
}
@article{LewisNigamOlver2005,
  author  = {Lewis, Debra and Nigam, Nilima and Olver, Peter J.},
  title   = {Connections for General Group Actions},
  journal = {Communications in Contemporary Mathematics},
  volume  = {7},
  number  = {3},
  pages   = {341--374},
  year    = {2005},
  doi     = {10.1142/S0219199705001787}
}
@article{DAddaLuscherDiVecchia1978,
  author  = {D'Adda, A. and L{\"u}scher, M. and Di Vecchia, P.},
  title   = {A $1/n$ Expandable Series of Non-Linear $\sigma$ Models with Instantons},
  journal = {Nuclear Physics B},
  volume  = {146},
  number  = {1},
  pages   = {63--76},
  year    = {1978},
  doi     = {10.1016/0550-3213(78)90432-7}
}
@article{KaganShepp2005,
  author  = {Kagan, Abram and Shepp, Lawrence A.},
  title   = {A Sufficiency Paradox: An Insufficient Statistic Preserving the {F}isher Information},
  journal = {The American Statistician},
  volume  = {59},
  number  = {1},
  pages   = {54--56},
  year    = {2005},
  doi     = {10.1198/000313005X21041}
}
```

Verification: Culver, Kagan-Shepp, Le Bruyn-Procesi, Lewis-Nigam-Olver, Littlejohn-Reinsch, D'Adda-Luscher-Di Vecchia and Smale were checked against publisher or catalog records today. O'Neill, Procesi, Higham, Le Cam-Yang, Cencov, Zamir, Kullback, Reinhart and Molino are from standard records not re-fetched; the lead should confirm the DOIs of O'Neill, Procesi, Reinhart and Zamir before building. The old key `LewisNigamOlver2003` is no longer cited by any .tex file and can be removed.

## NEW LABELS

02_interactions: `hyp:int-reference-consistency`, `eq:int-dressed-cost`, `eq:int-kernel-reference`, `prop:int-reciprocal`, `eq:int-reciprocal`, `prop:int-loop-blindness`, `prop:int-loop-defect`, `eq:int-loop-defect`.
02_geometry: `geo:transitive-pullback`, `geo:radical-curvature`, `geo:structure-identity`, `geo:basic-criterion`, `geo:shape-pullback`, `geo:shape-metric`, `geo:context-energy`, `open:geo-gauged-flow`, `geo:minimal-contraction`, `geo:minimal-contraction-identity`.
determinant_example: `geo:det-canonical`, `geo:det-rank`, `geo:det-horizontal`.
All existing labels retained. `eq:int-reference-equivariance` and `eq:int-total-variation` now take the tuple of reference laws (`\bar{\boldsymbol\nu}^a`) instead of the endpoint pair; cross-references elsewhere are unaffected.

## AUTHOR DECISIONS NEEDED

1. Primitive self channels (X4-PRIOR-02): AUTHORING.md says "Primitive identity self-comparisons", while chapters 11 and 12 use nonidentity self channels. Chapter 2 now states that the self channel is constitutive data with the identity as one choice. AUTHORING.md should be amended to match, or chapters 11 and 12 changed.
2. Chapter 1, line 31, still calls `app:connection` "an explicit directional-transport realization of its connection" (outside my files). With the appendix retitled, that phrase should read "the transformation law of supplied connection data".
3. Structure group (X1-GAUGE-04/05): whether the theory declares one group G with two representations or G_q x G_s is a modeling choice. Chapter 2 now names it as a choice; chapter 1 and chapters 11 and 12 should state which one they use.
