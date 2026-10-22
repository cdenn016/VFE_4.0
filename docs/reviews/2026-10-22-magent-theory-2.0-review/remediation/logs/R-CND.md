# R-CND remediation log: chapter 11 and companion c11

Files edited: `M/chapters/11_conditional_formation.tex`, `M/companion/c11_formation.tex` (M = `Research/manuscripts/MAgent_Theory_2.0/`). Not compiled. Scripts and outputs: `remediation/scripts/R-CND/` (s1 to s5, each single-threaded, under 20 s).

## Verification done before editing

| Check | Script | Result |
|---|---|---|
| Shape law dα_i/dt = −2(α_i − α_0) independent of κ, precisions, other agents; also under the null flow (peer costs deleted, self channel with peers frozen) | s1_shape_law.py (natural gradient, exact Wishart Fisher matrix, unequal precisions, κ ∈ {0,1,5}) | shape residual ≤ 1.6e-6 (finite-difference floor) for full and null flows; precision law P' = −2P + I + C residual ≤ 1.3e-6; null-flow precision velocity differs from full by 1.09 (κ=1), 6.44 (κ=5), so only precisions see the interaction |
| Linkage type of the threshold rule | s2_linkage.py (exact event rule with inversion and cascade assertions vs scipy) | 1000 random configs (N=3..8): 0 mismatches with centroid/UPGMC in sets and times; 251 set-sequence mismatches with single linkage. Explicit 0, 1, 2.05, 3.5: rule and centroid give ((01)(23)), single gives (((01)2)3). Chapter example times reproduced. Tree independent of δ. N=4 exact region: 9382 balanced trees, predicate matches all (0 mismatches); old sufficient class covers 6691 |
| Geometry decay | s3_geometry_decay.py | fitted g_aa rate 6.342891 vs 2|f_p(p*)| = 12 − 4√2 = 6.343146; κ=0 rate 4.0 and F = −1.716 (nonzero without interaction); α0=η0 curvature → ≈ −1.81; α0=1, η0=2 connection → −dχ, F → 0 |
| Mean-field identity | s4_meanfield_nogo.py (finite alphabet, Fisher flow with finite-difference forces) | spread of d/dt log(q_i/q_j) + 2 log(q_i/q_j) over the alphabet 7.4e-9 with shared targets; 7.06 with receiver-dependent targets |
| Anisotropic references (L08-CND-12) | s5_anisotropic_limit.py | natural-gradient residual 2.9e-7 for P' = −2P + Γ + C; block tensor → diag(1.168, 0.199), F → −1.6887 (matches L08) |

Linkage conflict (REPORT section 7): settled as count-weighted centroid (UPGMC). X3-PIFB-03 and X6-LIT-05 "single linkage" is rejected: s2 evidence above, and the chapter's own union time uses the centroid distance d2 + (d1+d3)/2, not the single-linkage gap d2.

Lead note (F5 compares partition sequences P vs P^0): `form:kappa-independence` (iii) now states P^0(x) = P(x), and also H^0 = H with event times, on the whole admitted class. Null flow vs κ = 0, settled on evidence: they are different flows. The null flow deletes peer costs but keeps κ inside each self channel, evaluated with frozen peer laws. Its precision law is P_i' = −2P_i + I + C_i^0, and precisions need not stay common: in s1, the null-flow precision velocity differs from the full flow by 1.09 at κ=1 and by 6.44 at κ=5. At κ = 0 the target equals the reference and P' = −2P + 2I. All three flows (full at any κ, null, κ = 0) have the identical shape law (s1 residuals at the finite-difference floor), so they give the same partition sequence and the same event times. This is now written into (iii) and its proof.

## Findings

| ID | Status | What changed (file, label) |
|---|---|---|
| H3 (REPORT) | applied | New `form:kappa-independence` (ch11): (i) κ-free shape law, b_I(t)=e^{-2t}b_I(0); (ii) general-N theorem: rule = UPGMC dendrogram of initial shapes, join times ½log(D_k/δ), strictly increasing heights, no cascades, persistence, tree independent of κ, precisions, model data, δ; (iii) null flow globally defined on individual-precision Wishart family with same shape trajectories, so H = H^0 on the whole admitted class, failing (F5) and its strong form. Chapter retitled; intro, F-classification paragraph, companion intro and selection section rewritten |
| L08-CND-01 | applied | As H3; "interaction is sensitive to other agents" rewritten to say the coupling moves precisions and never shapes (after `form:coupling`); companion section retitled "An interaction that moves every precision and no shape" |
| L08-CND-02 | applied | Constant model-independent evaluator listed in the declared architecture (ch11 intro, companion intro and closing list); paragraph after `form:energy` explains why it is load-bearing (belief force term, loss of common precision, non-Wishart sum of evaluator outputs); `form:parent` proof notes evaluator descent uses constancy. README/AUTHORING are outside ownership (AUTHORING already lists it) |
| L08-CND-03 | applied | Paragraph after `form:selection` proof: discarded allocation is a dynamical spectator; belief allocation recoverable from event record; companion section retitled "...an uncoupled lost allocation", "real loss" wording removed |
| L08-CND-04 | applied | New `form:geometry-decay` with proof: P(t) → p*I_2, e^{λt}(p+−p−), e^{λt}∂_a p_± converge to nonzero limits, g_aa and inertia decay at rate 2λ, 2λ = 12−4√2 at α0=κ=1; following paragraph gives numbers and the rate-dependent curvature limit; companion geometry section updated |
| L08-CND-05 | applied | `form:responses` changed from theorem to corollary heading (label kept); paragraph states block-sum sufficiency plus determinism and that adaptivity is trivial without record-to-law feedback; feedback version named as outside the interface |
| L08-CND-06 | applied (ch11, c11); c07 part outside ownership | Paragraph "The hierarchy is transient" (asymptotic consensus, distinct children at finite times); companion selection section. `companion/c07_fusion.tex:113` contrast still needs the lead's edit |
| L08-CND-07 | applied | Muirhead1982 (multivariate gamma, Wishart, congruence Jacobian) and Gindikin1975 (shape range) cited; LetacMassam2017 kept for the noncentral existence question (ch11 after `form:wishart`, companion `cf:wishart` paragraph) |
| L08-CND-08 | applied | Sorted shapes a_i → u_i; model log-eigenvalue b → ã; model eigenvalues q_± → p̃_±; inertias k_1,k_2 → 𝓘_1,𝓘_2 (both files), which removes the clash with channel input k |
| L08-CND-09 | applied | `form:gaps` replaced by the exact region δ < min(d1,d3) < d2, max(d1,d3) < d2 + ½min(d1,d3), with necessity up to ties; general-N statement in `form:kappa-independence` (ii) |
| L08-CND-10 | applied in part; obstruction deferred | Identity proved as new `form:mean-field` (shared reference/constant evaluator/receiver-independent additive-moment target ⇒ ∂_t log(q_i/q_j) = −2 log(q_i/q_j) + c_ij; exponential-family natural-parameter differences decay e^{-2t}). The candidate obstruction between exact fiber closure and selective rows is unproved; stated as unsettled in one sentence. The interaction clause already exists in ch5 as (F5) |
| L08-CND-11 | deferred | New open problem `form:approximate-open` (moment-matched parent off the common-precision family, second-order cancellation, invariant spread ratio) |
| L08-CND-12 | applied | New remark `form:anisotropic-reference` (equations, preserved closure and schedule, fixed-point limit, numbers verified by s5, gauge note that references become transforming sections); companion geometry section |
| L01-FOUND-06 | applied (ch11 part) | Sentence after `form:channel`: channel depends on the whole configuration through M_q, as eq:dyn-peer-costs allows, and is jointly measurable. The def:int-channel/def:int-generator changes are ch2 (outside ownership) |
| L02-GEOM-03 | applied (ch11 part) | Paragraph after `form:geometry`: at finite times isotropy is zero and constant rank holds; the jump occurs only in the isotropic limit (also in `form:geometry-decay`) |
| X1-GAUGE-01 | applied (ch11 part) | Geometry section: flow is pointwise in context, no field/row/channel/record reads the connection; connection is a diagnostic of evolved sections (both files) |
| X1-GAUGE-04 | applied | Geometry section states the common SO(2) is a declared tie; paragraph after `form:geometry` proves the independent SO(2)_q × SO(2)_s gives flat connection and rank-one tensor; `form:geometry` title says "for a prepared contextual family" and "for the common SO(2)"; companion |
| X2-FRAME-01 | applied (ch11 parts) | κ=0 statements in `form:kappa-independence`, `form:geometry` ("for every κ ≥ 0"); rule domain (initial separation > δ) made explicit in `form:threshold-rule` with the X2 counterexample behavior described. Abstract/outlook/ch1 are outside ownership |
| X2-FRAME-05 | applied | Sentence after `form:parent`: inheritance holds for every partition, so selection and inheritance are logically independent; intro and companion |
| X2-FRAME-06 | applied | "prepared" in intro, geometry section, theorem title, paragraph after `form:geometry`; abstract wording is the lead's |
| X2-FRAME-07 | applied (ch11 part) | Centroid clustering attributed (SokalMichener1958, LanceWilliams1967) after `form:kappa-independence`; open-law part is ch12 |
| X2-FRAME-08 | applied (ch11 parts) | Selector paragraph promoted to `\definitionheading{Threshold membership rule}{form:threshold-rule}`; q_± clash removed. Macro, file order and other chapters outside ownership |
| X2-FRAME-10 | applied | New `form:scalar-closure` (Jensen midpoint equation ⇒ affine law ⇒ common factor on centroid differences) with proof; vector escape route stated as one sentence (construction deferred). Lead may move it to ch5 |
| X3-PIFB-03 | applied; single-linkage characterization rejected | κ-independence as H3; linkage settled as centroid (s2) |
| X3-PIFB-14 | applied (ch11 part) | Sentence after `form:responses`: decision functionals of the record law are inherited; information gain about the discarded allocation is not. Registration budget and Dirichlet bound belong to the record appendix (outside ownership) |
| X4-PRIOR-01 | applied (ch11 part) | Retitled chapter; UPGMC equivalence, δ as time origin; F-classification paragraph states the second stage forms from every counterfactual handoff with two parents farther than δ apart, so the counterfactual part of (F6) also fails |
| X4-PRIOR-02 | applied (ch11 part) | Paragraph after `form:channel`: nonidentity primitive self channels are a constitutive choice; identity self channels would give D_ii = 0 and non-prior rows; construction requires the choice (companion too) |
| X4-PRIOR-03 | applied (ch11 part) | Centroid-clustering citations and contrast with bounded-confidence dynamics (HegselmannKrause2002) after `form:kappa-independence` |
| X4-PRIOR-05 | applied | `form:geometry-decay`; uniform positive lower bound is false (tensor → 0), stated |
| X4-PRIOR-08 | applied (ch11 part) | Clause after the Gibbs-row sentence: learned source priors would stay fixed because all costs in a row agree |
| X5-IMPL-06 | applied (ch11 part) | `form:threshold-rule` remarks: event times are closed form, no discretization or event location; ties by the component rule |
| X5-IMPL-04, 05, 09, 10, 13 | rejected for these files (not applicable) | Implementation-side findings (VFE_4.0 code, conformance suite, analytics); they cite ch11 labels only as test targets and require no change to the chapter text |
| X6-LIT-01 | applied (ch11 part) | Paragraph after `form:full-flow` proof citing ChenEtAl2023 (explicit Fisher-Rao KL flow; rate 2 explained) |
| X6-LIT-03 | applied (ch11 part) | Paragraph after `form:geometry`: mechanical connection (Smale1970), quotient metric of a Riemannian submersion (ONeill1966) |
| X6-LIT-05 | ch11 part: attribution applied; "single linkage" rejected | See linkage conflict above |
| X6-LIT-12 | applied (ch11 part) | Muirhead1982 and Gindikin1975 for `form:wishart`, `form:addition` |

## BIB ENTRIES

Verified: Gindikin (Springer DOI and Math-Net.ru issue table; English pages 50-52, Russian 56-58), Chen et al. (arXiv abstract page), Sokal and Michener (Biodiversity Heritage Library record). Lance-Williams, Muirhead, Smale, O'Neill, Hegselmann-Krause are standard records as also reported by X6-LIT/X2-FRAME/X4-PRIOR; not re-opened by me.

```bibtex
@book{Muirhead1982,
  author    = {Muirhead, Robb J.},
  title     = {Aspects of Multivariate Statistical Theory},
  publisher = {John Wiley \& Sons},
  address   = {New York},
  year      = {1982}
}

@article{Gindikin1975,
  author  = {Gindikin, S. G.},
  title   = {Invariant generalized functions in homogeneous domains},
  journal = {Functional Analysis and Its Applications},
  volume  = {9},
  number  = {1},
  pages   = {50--52},
  year    = {1975},
  doi     = {10.1007/BF01078179}
}

@article{SokalMichener1958,
  author  = {Sokal, Robert R. and Michener, Charles D.},
  title   = {A statistical method for evaluating systematic relationships},
  journal = {The University of Kansas Science Bulletin},
  volume  = {38},
  number  = {22},
  pages   = {1409--1438},
  year    = {1958}
}

@article{LanceWilliams1967,
  author  = {Lance, G. N. and Williams, W. T.},
  title   = {A general theory of classificatory sorting strategies. {I}. {H}ierarchical systems},
  journal = {The Computer Journal},
  volume  = {9},
  number  = {4},
  pages   = {373--380},
  year    = {1967},
  doi     = {10.1093/comjnl/9.4.373}
}

@article{HegselmannKrause2002,
  author  = {Hegselmann, Rainer and Krause, Ulrich},
  title   = {Opinion dynamics and bounded confidence: models, analysis and simulation},
  journal = {Journal of Artificial Societies and Social Simulation},
  volume  = {5},
  number  = {3},
  year    = {2002}
}

@misc{ChenEtAl2023,
  author        = {Chen, Yifan and Huang, Daniel Zhengyu and Huang, Jiaoyang and Reich, Sebastian and Stuart, Andrew M.},
  title         = {Sampling via Gradient Flows in the Space of Probability Measures},
  year          = {2023},
  eprint        = {2310.03597},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

@article{Smale1970,
  author  = {Smale, Stephen},
  title   = {Topology and mechanics. {I}},
  journal = {Inventiones Mathematicae},
  volume  = {10},
  pages   = {305--331},
  year    = {1970},
  doi     = {10.1007/BF01418778}
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
```

Other agents (ch3, ch2, ch12) may propose the same keys; entries above should be merged by key.

## NEW LABELS

ch11: `form:mean-field`, `form:common-clock` (equation), `form:scalar-closure`, `form:threshold-rule`, `form:kappa-independence`, `form:geometry-decay`, `form:anisotropic-reference`, `form:approximate-open`. Unchanged existing labels kept (including `form:selector`, `form:gaps`, `form:selection`, `form:responses`, now a corollary). Companion: no new labels. Grep confirmed no duplicates and no undefined `form:` references in ch11.

## AUTHOR DECISIONS NEEDED

1. The ch5 definition calls a construction "a two-stage selection with exact inheritance" when it satisfies (F1), (F2), (F3), (F6) but not (F4) or (F5). Chapter 11 also fails the counterfactual part of (F6): its union forms from every counterfactual handoff with two parents farther than δ apart, because all scores contract on one clock. Either weaken the definition's (F6) requirement to the first sentence of (F6) (second-stage selection without reset or new attraction), or give ch11 another name. Chapter 11 currently uses the term and states the (F6) failure explicitly.
2. `form:mean-field` and `form:scalar-closure` are general statements (X2-FRAME-10 and L08-CND-10 suggested ch5). They are placed in ch11 because ch5 is not in my ownership; move them if the general chapter should carry them.
3. Outside my files and still pending: README.md:24 "interacting", the abstract sentence on ch11 and its "rank-two contextual geometry" (add "prepared", "tied belief/model group"), and `companion/c07_fusion.tex:113` (ch11 also reaches consensus, asymptotically).
