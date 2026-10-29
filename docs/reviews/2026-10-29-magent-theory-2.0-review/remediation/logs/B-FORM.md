# B-FORM remediation log (2026-10-29, wave 2)

Files edited (manuscript root `Research/manuscripts/MAgent_Theory_2.0/`, branch `magent-2.0-remediation-2026-10-29`, nothing committed or compiled): `chapters/10_boundary_closure.tex`, `chapters/11_conditional_formation.tex`, `chapters/12_open_formation.tex`, `companion/c10_boundary.tex`, `companion/c11_formation.tex`, `companion/c12_open_formation.tex`. CRLF line endings kept (one accidental LF conversion of ch10 by `sed -i` was reverted to CRLF byte for byte). Reference check (`check_refs.py`): every `\cref` target in the six files resolves; begin/end and braces balance; the only unresolved citation keys are the two new entries below.

Scripts (`V/remediation/scripts/B-FORM/`): `ch11_lp_null.py`, `ch11_lp_delta.py` (single-threaded, under 30 s each) integrate the chapter 11 full flow, the renormalized null and the label-preserving null with the selector on fine shape sums; `list_findings.py`, `show.py`, `locfix.py` extract lens findings; `edit_ch10_sym.py`, `edit_ch10_rest.py`, `edit_ch12.py`, `edit_ch12b.py`, `edit_c12.py` are text-edit helpers; `check_refs.py` checks labels and keys.

What the scripts show (chapter 11, alpha_0 = 1, P(0) = I, T_q = 1): at kappa = 0 and shapes (1.2807, 1.5057, 1.8665, 2.3057) the full flow and the renormalized null give {1,2}, {3,4}, root at 0.406, 0.740, 0.968; the label-preserving null gives {1,2}, {1,2,3}, root at 0.689, 1.340, 1.776 (reproduces GAP03-S2). Minimum other-pair margin at null events 3.5e-3 for delta in {0.09, 0.095, 0.10, 0.105, 0.11}; same null sequence at kappa = 0.05, 0.1; same two sequences in 12 of 12 random perturbations (shapes 4e-3, P(0) 1e-2, kappa in {0, 0.01}, T in [0.9, 1.1], delta in [0.098, 0.102]); at kappa >= 0.5 the null sequence at this point equals the full one. Chapter example at kappa = 1: label-preserving null 0.650, 1.111, 2.058 (same blocks), renormalized null equals the full flow. The first-join reversal of D15-WISH-01 (gaps 0.30, 0.60, 0.29) is an exchange of disjoint merges and no longer counts under the repaired P of `inh:null-flow`; the structural difference above does. The label-preserving null law was derived by hand and implemented independently of the lens scripts.

## Findings table

| ID | Status | What changed (file, label) / reason |
| :- | :- | :- |
| H01, D15-WISH-01, GAP03-S2-01, XC3-SCOPE-01 (and E3), D1 | applied | ch11 `form:kappa-independence` retitled "... and the controls that reproduce it". Old (iii) split: (iii) coupling-deformation control (H independent of kappa, stated as not a control of `inh:null-flow`); (iv) renormalized null (N2): H^0_rn = H, P^0_rn = P everywhere, kept with its proof; (v) frozen-environment control: H^fr = H, with existence proof. New `form:lp-null` (law `form:lp-null-law`, deviation equation `form:lp-null-deviation`): computed (N1) law alpha' = -(1+beta_i)(alpha - alpha_0), P' = -(1+beta_i)P + I + beta_i C_i^0, invariance, global existence for kappa alpha_0 <= N^2, agent-dependent clocks; numerical paragraph after it gives P^0_lp != P on an open set already at kappa = 0, stated as floating-point evidence with margins. Intro (11:6) and verdict rewritten: split, hence fails. |
| XC2-FRAME (checked row on (iii)), XC8-REMED-05/-13 positions | decided | Section 8 disagreement decided on the computation: both positions are correct for their convention; (iii) as written was true for (N2) only. Kept for (N2), replaced for (N1). |
| GAP03-S2-02 | applied | After `form:kappa-independence`: controls remain the fine flow after a handoff, selector reads shape sums from fine shapes; along (N2)/Phi^fr with kappa > 0 the block laws are not Wishart and `form:parent`, `form:responses` are not claimed. |
| D15-WISH-02, S2-M01, XC2-FRAME-04, XC8-REMED-02, D09-INH-06 (ch11, c11 parts) | applied | ch11 paragraph after `form:mean-field`: "needs targets that depend on the receiver" removed; conclusion restricted to selectors reading centered log-ratios; new `form:cost-difference` (D_i - D_j = F(alpha_i) - F(alpha_j) + (alpha_i - alpha_j) tau(t), derived and checked by hand) shows cost-reading selectors can be coupling sensitive. c11:79 rewritten the same way. The 16/150 count of the lens was not reproduced and is not quoted. |
| D15-WISH-05, S2-L42, S4-L08, XC2-FRAME-08 (dup part), XC8-REMED-05 (ch11 part) | applied | ch11 cites `inh:common-clock` case (b) from the `form:mean-field` paragraph and `inh:scalar-additive` after `form:scalar-closure`. Both ch11 propositions kept (ch11's is on regular families, ch05's on tables). |
| D15-WISH-09, S2-E22 | applied | Sentence after `form:scalar-closure`: in the `form:mean-field` class the propagator is -2 times the identity. |
| D15-WISH-06, S2-L43 | applied | 11:25: absolutely continuous part of the Gindikin set {0, 1/2} with (1/2, inf). |
| D15-WISH-07, XC8-REMED-09, D10-FORM-02 (ch11 part) | applied | Verdict paragraph declares U and builds the class of `inh:counterfactual-class`; (F6b) fails (supplied-scale sequence), independently of the null. |
| D15-WISH-08 | applied | c11: "anisotropic local data in both sectors, context dependent in the belief sector". |
| D15-WISH-03 | n/a here | c05 is A-STD's (done there). |
| GAP03-S2-03, S2-M24, S4-L21, XC3-SCOPE-06 | applied | `form:full-flow`: regular solution class defined in the proof, the integrating-factor uniqueness argument with both omitted steps (positivity of Pi, closure), and the statement that bounded log-density charts do not contain the family. Statement updated. `form:mean-field` hypothesis now refers to that class. |
| GAP03-S2-04, D09-INH-07 (ch11 part) | applied | Paragraph after `form:parent`: type closure, natural-coordinate (Bregman) route, failure of horizontal conformality citing `inh:type-closure-open`. Lemmas A/B checked by hand; proofs of `form:full-flow`/`form:parent` not replaced (kept coefficient matching; the new paragraph gives the alternative). |
| GAP03-S2-05, S2-E23 | applied | Paragraph before the context-constant remark: connection class inherited for every smooth section; tensor defect Delta formula; c11:195 corrected ("individual shapes"). Defect formula reconstructed by hand. |
| GAP03-S2-06, S2-L45 | applied | ch11 decay paragraph states a-tilde = 3/10 and the dependence (-1.55 at 0.2, -1.27 at 0.5, lens values); c11 tensor-rate sentence corrected. |
| GAP03-S2-07, S2-E24 | deferred | Extension (convolution-affine families); not verified, no text added. |
| D10-FORM-01, -05, -06, -09 (ch11, ch12 parts), XC6-PROG-04, -05 (ch11/ch12 parts), H04-H08 ch11/12 parts | applied | Recipe reports written for ch11 (verdict paragraph) and ch12 (after `open:formation`): objective, chart, (F1) grade, selector family and constants, A_I/B_I rules, (F4) witness with attributes, (F5) per null, selector-free observable, Phi^fr, (F6a), (F6b) class, grades, classification. |
| D10-FORM-09 (ch12), D16-OPEN-01, S2-M07 | applied | ch12:4 and the verdict: "not shown to be a recursive formation"; classification undetermined ((F2) over a constant set, (F5), (F6b) open), not "two-stage selection". `open:formation` and chapter title reworded to "selection in two stages". c12:7 matched. |
| D16-OPEN-02, S2-M04 | applied | `open:null-control` notes that the near-simultaneous first-stage events are disjoint merges, which the repaired P ignores. |
| D16-OPEN-03, S2-M08, S4-M02, XC4-CONS-02, XC6-PROG-18, XC8-REMED-03 (ch12, c12 parts) | applied | Null reproduction stated as unproved limit-field integrations at 12:4, `open:null-control`, the verdict, c12:7 and c12 null section. Other locations are C-SUM's. |
| D16-OPEN-04, S2-L46 | applied | `open:null-control`: proof that the full-field contraction rate exceeds one on a >= 4, 0 < b <= 2 (two cases, checked by hand); c12:100 corrected. |
| D16-OPEN-05, S2-L47 | applied | `open:loss-information`: equivariance argument, sum of inertias at record times, maximum-likelihood estimate. |
| D16-OPEN-06, GAP02-S2-04, S2-L48 | applied | `open:parent-flow` (up to the first vanishing anchor moment; regular chart not invariant; quotient flow global) and `open:all-partitions` (interval [t_0, t_1)). |
| D16-OPEN-07, S2-L49 | applied | `open:constants` (A = 4096 illustrative, needs eps_b >= 2.3e-3; bounds before the Gronwall factor); c12:96. |
| D16-OPEN-08 | applied | `open:null-control` names both nulls, their coincidence on the binary family and difference on correlated laws (rate 2 vs 1 + beta^0_ii), the readout convention, and the actual R and A of the integrations. |
| D16-OPEN-09, XC8-REMED-10 | for C-SUM | Outlook. |
| D16-OPEN-10, E01, S2-E02, GAP02-S2-02 | deferred | Extensions; `open:necessity-open` now also lists proving the null comparison. |
| GAP02-S2-01, S2-E01 | deferred (caveat applied) | Validated enclosure not adopted as proof (computational certificate outside the text; author choice). `open:constants` now records the horizon caveat (stop at a = sqrt(log 2)). |
| GAP02-S2-03, S2-L51 | applied | R-step: M grows with R, error terms still vanish, persistence needs the decoupled Lipschitz constant. |
| GAP02-S2-05, S2-L52 | applied | 12:489: the flow-map facts used (analytic field, C^2 flow map, Dieudonne1960; equivariance of angular derivatives; only first angular jets), neighborhood open in C^1. Stated with its ingredients, not proved in full. |
| GAP02-S2-06, S2-L53 | applied (alternative fix) | "Its dissipation is that of the quotient metric"; remark that on the von Mises family the clause does not distinguish the metrics. Two-harmonic family not added. |
| GAP02-S2-07, S2-L54 | applied in part | `open:open-laws`: M -> Theta, epsilon -> epsilon_c. delta/delta_0 and L/L_1 kept. |
| XC4-CONS-09 (iv), (ii) | applied | 12:43 "With an identity self channel"; 10:321 "|J|<1". |
| XC4-CONS-12 (ch12) | applied | One `\eqref` -> `\Cref`. |
| XC1-GAUGE-05, XC2-FRAME-09 | for C-SUM | Abstract; ch12 already states the tie. |
| XC6-PROG-12 | applied | Sentences in 10:4 and 12:4: gauge structure supplies forgotten orbits, no dynamical work. |
| S3-L07 / D18-LP-10 | for B-PA | ch14. |
| D14-BND-01, S2-M11 | applied | 10:4 states both nulls coincide (self excluded), preserve family/descriptors/interface, null profile equation, contrast rate one; c10 opening. Checked by hand. |
| D14-BND-02, S2-E13 | applied | New `cor:boundary-contrast-global` (no finite-time loss on any base; global decay and rows -> priors on a finite base), proof by backward uniqueness and LaSalle plus convexity; c10 decay sentence. Lojasiewicz convergence to one equilibrium not claimed. |
| D14-BND-03, S2-E14 | applied | `rem:boundary-heterogeneous`: two-block family = Fix(H'_alpha), invariant. |
| D14-BND-04, S2-E15 | deferred | Non-invariance of conjugacy not verified; the remark keeps it open (whole-trajectory conjugacy stated as the open condition). |
| D14-BND-05, GAP06-S2-02 (a), S2-E16 | deferred | d = 1 Jacobian closed form not reproduced. |
| GAP06-S2-02 (b) | applied | New `rem:boundary-root-blind` (base-uniform shapes invisible to every admitted root history; proof checked by hand); `op:boundary-minimality` updated. |
| D14-BND-06, S2-L31 | applied | Rotation sign corrected (theta + alpha - alpha'); group law stated so the action is a left action. Checked by hand. |
| D14-BND-07, S2-L32 | applied | Kronecker-Weyl condition (1 and c_v rationally independent), counterexample, `\citep[Chapter 1]{Walters1982}` (chapter locator from the standard content, theorem number not checked). |
| D14-BND-08, S2-L33 | applied | `lem:boundary-equivariance` (ii) general form for flow-invariant unions of fixed-point sets; cited in the quotient paragraph. |
| D14-BND-09, XC8-REMED-11, S2-L34, S4-L24 | applied | `prop:boundary-joint-spectrum`: agent-dependent factors L/Z_i determined by mu^q; the two measurable/C^1 identifications. |
| D14-BND-10, S2-L35 | applied | 10:11 restricted; symmetry section needs reversible sign cells. |
| D14-BND-11, XC8-REMED-12, S2-L29/S4-M25, B-DYN item | applied | `prop:dyn-chart-flow` cited at 10:48 and in the proof (continuation alternative added to `lem:boundary-fine-wellposed`); `lem:boundary-global` cites the continuation alternative and `prop:dyn-chart-global`; `inh:equivariant-descent` cited with the reason it does not apply. ch12 `open:force` cites `prop:dyn-chart-flow` (Lipschitz hypothesis; its fixed-kernel sufficient condition does not apply to the selective channel, which has no density bounded below). Both citations checked against the ch03 statement. |
| D14-BND-12, S2-L36, B-DYN item | applied | `rem:boundary-evidence`: case of `eq:dyn-evidence-potential`, norm summed over sectors (ch03 sentence already amended by B-DYN). |
| D14-BND-13, S2-L37 | applied (c10) | c10:38 bounded logarithm; c10:76 local/finite-base decay. AUTHORING is C-SUM's. |
| D14-BND-14, S2-E21 | applied | `prop:boundary-observable` for every real J, with Q(c; J) at a general base point (recomputed by hand: -32, -36). |
| D14-BND-15, S2-L38 | applied | 10:4: (F1) family grade, (F3), (F4) hold; (F2), (F5), (F6) not addressed. |
| D14-BND-16, S2-L39 | applied in part | Permutation pi -> varpi in the symmetry section; constant K -> C_K. phi, lambda, A, H collisions deferred. |
| D14-BND-17, S2-L40 | applied in part | Dirichlet-form identity (checked by hand) with BobkovTetali2006. Book locators for Golubitsky et al., Field not added (not verified). |
| GAP06-S2-05, S2-L41 | applied | `rem:boundary-wreath`: 2-transitive node groups, realized uniformly; cyclic counterexample described qualitatively. |
| D13-ERAPUB-07, XC8-REMED-07, S2-L25 | applied | 10:321 "|J|<1" (checked against ch09:9). |
| GAP04-S4-01, -05, D05-OBJ-03, D01-FOUND-11 (ch10 parts), B-FOUND item, B-DYN item | applied | 10:228 measurability claim restricted (finite base proved; general base assumes the joint Borel flow of `def:dyn-record-interrupted`); record kernel jointly Borel; Borel policies and `def:adaptive-history` cited at 10:249 and in ch12 responses. No chart-measurability proposition exists in ch03 to cite. |
| GAP04-S4-04 | n/a here | Fix is in ch03 (B-DYN). |
| GAP04-S4-06 | deferred | Optional remark. |
| D12-SMOOTH-03 | for B-FUS | ch07 should cite `prop:boundary-joint-spectrum`. |
| D02-LINKS-03 | n/a here | ch11:452 already declares the tied rotation; fix is in ch02. |
| XC4-CONS-08, XC4-CONS-14 | deferred | Cross-chapter naming and unreferenced labels. |
| XC1-GAUGE-01, -03, XC2-FRAME-01 | n/a here | ch10:191 and ch12:416 use "gauge" relative to a declared algebra; no "physical"/"nongauge" in my files (ch12 uses "physical-group" only as a label name). |

Rejected: none. Partial non-adoptions: GAP02-S2-01 (validated certificate not made part of the proof), GAP02-S2-06 (dissipation wording instead of a new family), GAP03-S2-04 (alternative route added, proofs not replaced).

## BIB ENTRIES

```bibtex
@book{AmariNagaoka2000,
  author    = {Amari, Shun-ichi and Nagaoka, Hiroshi},
  title     = {Methods of Information Geometry},
  series    = {Translations of Mathematical Monographs},
  volume    = {191},
  publisher = {American Mathematical Society and Oxford University Press},
  address   = {Providence, RI},
  year      = {2000}
}

@article{BobkovTetali2006,
  author  = {Bobkov, Sergey G. and Tetali, Prasad},
  title   = {Modified logarithmic {S}obolev inequalities in discrete settings},
  journal = {Journal of Theoretical Probability},
  volume  = {19},
  number  = {2},
  pages   = {289--336},
  year    = {2006},
  doi     = {10.1007/s10959-006-0016-3}
}
```
Bibliographic data from the standard records as I know them (AMS TMM 191; J. Theoret. Probab. 19(2), DOI as given by D14-BND-17); not re-verified online in this session. C-SUM should check before merging. Cited in ch11 (natural-coordinate paragraph) and ch10 (Dirichlet form).

## NEW OR CHANGED LABELS

New (unique across the three documents, checked): `form:lp-null` (proposition), `form:lp-null-law`, `form:lp-null-deviation`, `form:cost-difference` (equations), `cor:boundary-contrast-global` (corollary), `rem:boundary-root-blind` (remark). Retitled: `form:kappa-independence` ("Coupling-independent centroid dendrogram and the controls that reproduce it"; items now (i) to (v), old (iii) content is (iv)), `open:formation` ("Open-law selection in two stages with exact inheritance"); chapter 12 title "Preferential Selection in Two Stages from Open Law Classes". No label removed or renamed. External references to `form:kappa-independence` (iii) by item number, if any, now point to the coupling-deformation control: grep found none in other files.

## FOR OTHER OWNERS

- C-SUM: frontmatter:9 and outlook:10 (ch12 null reproduction is unproved limit-field integrations; ch12 is "not shown to be a recursive formation", classification undetermined, not "two-stage selection"; tie of rotations, XC1-GAUGE-05/XC2-FRAME-09); outlook:10 transience and final merge (D16-OPEN-09), "every selection construction" (XC8-REMED-10), outlook:12 chapter 10 null clause (D14-BND-01), outlook:16 common-clock wording (XC4-CONS-03); AUTHORING:67, :71 (D14-BND-13, D16-OPEN-03); ch11 summaries: (F5) split (fails), (F6b) fails, coupling-deformation control separate; chapter 12 title changed. Merge AmariNagaoka2000, BobkovTetali2006.
- A-STD (lead): ch05:2006-2010 says the label-preserving null "changes the membership history on a nonempty open set"; ch11 now supports this by floating-point integration with margins plus the proved agent-dependent clock law; consider "integrations show". ch05:2013-2017 on ch12 is consistent with the new ch12 text.
- B-FUS: ch07:257, :321 cite `prop:boundary-joint-spectrum` (D12-SMOOTH-03).
- B-DYN: if a chart-measurability proposition is added to ch03 (GAP04-S4-01), ch10:228 and :249 can cite it instead of assuming the joint Borel flow on a general base.
- B-PA: ch14 method-of-slices attribution (D18-LP-10).

## AUTHOR DECISIONS NEEDED

1. A-STD's provisional rule that a split (F5) verdict fails: ch11's (F5) verdict depends on it (fails if split counts as failing; would hold formally against (N1) alone otherwise). The ch11 classification does not, because (F6b) fails independently.
2. A-STD's provisional (F2) rule (constants over a set with nonempty interior): ch11 now declares delta in C = [9/100, 11/100] (my choice, tested numerically across C); confirm or replace. ch12 fixes epsilon = 1/10, delta = 1/20 and its theorem is not proved over a set, so (F2) in the repaired form is reported undetermined there.
3. Whether to make the validated interval enclosure of GAP02-S2-01 part of the ch12 proof (it would fix eps_b = 1, admit A = 34, and require the horizon to stop at a = sqrt(log 2)).
