# B-FOUND remediation log (2026-10-29)

Scope: `chapters/01_foundations.tex` (not 01:4), `appendices/record_process.tex`, `chapters/02_interactions.tex`, `chapters/02_geometry.tex`, `appendices/base_connection.tex`, `appendices/determinant_example.tex`, `supplements/s01_probability.tex`, `supplements/s02_geometry.tex`, `companion/c01_objects.tex`, `companion/c02_links.tex`, `companion/c06_geometry.tex`, `companion/figures/*`. Nothing compiled. Checks: `scripts/B-FOUND/lint.py` (environment balance, spacing macros, em dashes, label uniqueness and dangling references; all clean in my files). `scripts/B-FOUND/edit_geometry.py` applied the chapter 02_geometry replacements. Mathematical claims were reconstructed by hand before editing (noted per row); no numerical runs were needed beyond the lenses' own.

| ID | Status | What changed (file, label) / reason |
| :- | :- | :- |
| H03 (01:197, 01:226); D05-OBJ-01, XC6-PROG-02, XC8-REMED-01, XC4-CONS-01, D01-FOUND-02 (ch01 parts) | applied | 01 (sec:generative-models end): states the two proved parts of `prop:dyn-no-global-vfe` with their hypotheses (fixed positive rows, identity-channel edge; (a) finite alphabets, any recognition rule; (b) fixed channels, product recognition) and that neither covers coupled recognition on infinite spaces or law-dependent channels. 01 (after `prop:content-law`): same scope, plus the copula/total-correlation construction (product prior of reference joints, atomless laws) showing the separation rests on product recognition and the kinetic metric; finite-alphabet bound log|K| noted. Reconstructed: KL(Q‖⊗R_i) = Σ KL(ζ_i‖R_i) + TC(Q); Gaussian copula MI −½log(1−ρ²) ranges over [0,∞). Abstract-level wording (decision 8) left to C-SUM. |
| H05 / XC1-GAUGE-01 (ch01 part), D2 | applied | 01: new `def:observable-algebra` declaring A_pop (rows, channel outputs and costs, evaluators, record kernels, readout laws) and A_prot (A_pop plus admitted response laws of a declared protocol); gauge relative to A means no element separates at any time along the declared dynamics; child-level law diagnostics belong to no algebra, so distinctions only they resolve are gauge relative to every declared algebra; "physical"/"public" only relative to a named algebra. This is the basis on which B-PA requalifies chs 13 and 14. |
| D01-FOUND-01 (S1-M01) | applied | 01 (paragraph after `prop:loop-blindness`): law comparisons blind at one configuration; source-role force pulls the likelihood ratio back through H; second-order separation formula for a stabilizing self channel (reconstructed: first variation of KL(ν‖H#ν) vanishes at invariant ν; quadratic KL expansion); first-order possibility on two-cycles; "at the instant of comparison" qualifier; the time-evolved definition at `def:observable-algebra` answers the instantaneous-definition point. Companion c02 sentence updated. |
| D01-FOUND-03 (S1-M02) | applied, modified | 01: "only when" sentence replaced by `prop:predictive-equivalence` with the n-record composite kernel K^(n). Lens part (ii) claimed equality "for every history" from the one-record kernel; that is wrong when records are conditionally independent given (k,m) (the lens example gives different two-record laws: copies of k vs iid Ber(p_m)). Proposition is stated for K^(n) and (iii) uses a new counterexample valid for every n (k-independent likelihood, uniform vs δ_0 evaluator). Recognition-reading remark and "can change" added. Companion c01 detector sentence corrected. |
| D01-FOUND-04 (S1-M03) | applied | 01: reference-consistency paragraph for readouts/record/response kernels (after the covariance requirement); `eq:model-display` (reference-coordinate generative model with frame display), declared density l_i and τ-invariant reference measure λ_O, Jacobian/reference-measure remark; `prop:agent-fe-invariance` with proof (pushforward + `prop:kl-coordinate`); active-transformation counterexample a²/2. Passive change renamed to the tuple **g**=(g_i) (also removes the γ_i clash, D01-FOUND-09). |
| D01-FOUND-05 (S1-M04), XC6-PROG-19 (S4-L18) | applied | 01: arbitrary-bijection notion renamed "isomorphic generative models" (`eq:shared-model` kept); `def:shared-model` now defines sharing up to frame by one g ∈ G (`eq:shared-model-frame`); species = frame version; implication proved inline, converse fails; codes not part of the model. Downstream uses checked: ch03:438, conditional_memory:206, ch13:136, ch17:30/50 are consistent with the frame version. Companion c01 aligned. |
| D01-FOUND-06 (S1-M05) | applied | 01: `eq:composite-record-kernel`; inertness iff composite independent of m; constant evaluator suffices only when ℓ_i does not read m. |
| D01-FOUND-07 (S1-M18) | deferred (extension) | Precise remark only: the second-order loop separation formula after `prop:loop-blindness`. Holonomy-sensitive persistent record and the Z_2 equator import not done. |
| D01-FOUND-08 (S1-M19) | deferred / other owner | Restricted no-go part (c) belongs in ch03 (B-DYN). |
| D01-FOUND-09 (S1-L01 part) | applied in scope; rest other owner | 01: r_i now "presentation prior"; "reference law" reserved; single-agent model stated as self-only case of `eq:dyn-per-agent-model`; magnetization m → σ̄ (01, c01); passive-change γ → **g**; record appendix Dirichlet concentration A → α_+. Notation appendix/glossary coverage of chs 13 to 19 is C-SUM's. |
| D01-FOUND-10 (S1-L02) | applied | 01 `def:content-law`: record densities k_a with declared λ_a, likelihood renamed L_o, Z_o, Π_o and F(Q;o) defined locally; Borel structure of E_i and Borel sections ζ_i^Q stated. |
| D01-FOUND-11 (S1-L03) | applied | record_process: false "state marginal reproduces Γ" corrected (record marginal does; stochastic conditional update reproduces Γ); new `def:adaptive-history` (history-dependent Borel policies, action-dependent kernels, Ionescu-Tulcea, record-interrupted flow kernel `eq:record-flow-kernel`, joint Borel flow, stopping at blow-up and accumulation); state vs objective persistence paragraph. |
| D01-FOUND-12 (S1-L04) | applied | 01: fibered equivariant kernels after `prop:bundle-morphism`; corrected reason for excluding sector pairs (gauge transformation of P̄ gives a point-dependent kernel; only φ_1 ≠ φ_2 gives two groups); agent type now includes the generative model. |
| D01-FOUND-13, D02-LINKS-09(a), XC8-REMED-05 (S1-L05) | applied | `prop:int-loop-blindness` removed from ch02 and replaced by a citation of `prop:loop-blindness` (label was referenced only in ch02; no dangling refs). 02_geometry:343 duplicate argument replaced by citation of `prop:bundle-morphism`. |
| D01-FOUND-14 (S1-L06) | applied | c01: divergence direction stated (variational vs moment matching). |
| D01-FOUND-15 (S1-L07) | partly applied | Kechris Section 17.E added; data processing now cited to Gray1990 Chapter 5; Francois2019 cited at the dressing paragraph. Kallenberg locators not added (third-edition numbering unverified). Fournel journal record not rechecked. |
| D01-FOUND-16 (S1-L08), XC1-GAUGE-09 (a) | applied | 01 after `eq:content-bundles`; 02_interactions:22; base_connection:69; c02. |
| D02-LINKS-01 (S1-M06), XC1-GAUGE-07 (S4-L05) | applied | ch02 after `prop:int-reciprocal`: per-sector principal root does not respect representations (SO(2) charge-two example, recomputed by hand: backtrack 1.6π ≡ −0.4π, root −0.2π vs image 0.8π); group-level split Ω^G = θ_ij exp(−X/2) with strip condition, tied reciprocal parts tied, Ω^G_ji = (Ω^G_ij)^{-1} (reconstructed via Ad). Statement now says GL(V_a)-valued; app:connection reference qualified "extended general linear frame bundle". |
| D02-LINKS-02 (S1-M07) | applied | ch01 and ch02 rotation sentences: stabilizer of N(0,Σ) is the commutant, eigenplane half-turns always present; "centered" added; c01, c02 aligned. |
| D02-LINKS-03 (S1-M08) | applied | ch02 after `prop:int-mismatch`: realization by one link iff Ξ ∈ ker ρ_s ker ρ_q (proved inline); invariant is the class in G/(ker ρ_s ker ρ_q); congruence-action example; "nonzero" → nonidentity. c02 aligned. |
| D02-LINKS-04 (S1-M20) | applied as precise remarks | ch02: law-adjoint pairs (covariant polar form, positive spectrum, Ω Σ_j Ω^T = Σ_i; reconstructed) and moment-family nonreciprocity on independent means (reconstructed; uses L² = tL, inverse I − φ_1(−t)L). |
| D02-LINKS-05 (S1-L09) | applied | `prop:int-loop-defect` now assumes a reference-consistent rule; proof sentence qualified; hyp:int-reference-consistency now says the generator form is a strengthening (channel form e^{L'}=h^{-1}e^Lh, equal within |Im|<π); h_a = ρ_a(h) at ch02:24 and in the hypothesis; tied law-dependent links read both sectors (`def:int-group-links`). |
| D02-LINKS-06 (S1-L10) | applied | `prop:int-reciprocal`: i ≠ j, N_{ij,j} notation, uniqueness by balance (proved), sign(Θ_ii) note, λ_12λ_21 ≤ −4 no-real-root boundary (trace/determinant argument). |
| D02-LINKS-07 (S1-L11) | applied, softened | s02: DQM sentence replaced by sufficient conditions for first differentiability; DQM is the second-order condition; non-sufficiency via two-point ν_i. The lens's non-necessity counterexample is numerical only (no analytic O(ε² log 1/ε) estimate), so the text does not claim non-necessity. |
| D02-LINKS-08, XC1-GAUGE-06 (S1-L12, S4-L01) | applied | ch02:190 quiver sentence: GL(V_a) vertex groups, ρ_a(G) gives larger invariant rings (open-edge Tr(Θ^TΘ) for orthogonal G, checked), common change is one element of G in both sectors. |
| D02-LINKS-09 (b)-(e) | applied in part | (b) `def:int-group-links` now uses reference links θ̄_ij ∈ G with the exponential as optional chart, matching ch17; (c) eq:comp-link tuple form in c02; (d) figure message law μ (also XC4-CONS-09 (vi), D06-PRED-06); (e) mechanical connection and its curvature renamed θ^mech, Ω^mech in `geo:shape-pullback`. Other symbol clashes (H, Q, T, M) left; list to C-SUM for the notation appendix. |
| D02-LINKS-10, GAP08-S1-07 (S1-L13) | applied in part | Culver criterion stated exactly (ch02, s02, with J_1(−1)⊕J_3(−1) and diag(−1,−2)); Higham Sections 1.7, 1.8 for principal root and f(AB)/f(BA) (section titles as reported by D02-LINKS from the contents listing; theorem numbers not added); s02 low-rank lemma attributed as the f(AB)/f(BA) relation; Fréchet integral attributed to Wilcox1967 and Higham Chapter 10; φ_1 attributed to Higham Chapter 10. KullbackLeibler1951 locator not added (not verified). |
| D03-GEOM-01 (S1-M09) | applied | determinant_example remark replaced: no continuous invertible e for n ≤ 3 (stratum argument reconstructed: orbit dims 7/6/4, rank ≤3/2/1, open dense constant-orbit-type set, continuity); four agents minimal and sufficient near the four-agent tuple. 02_geometry:279 adds "at least four agents". |
| D03-GEOM-02 (S1-M10) | applied | 02_geometry: sentence corrected; new `geo:class-curvature` with proofs that JF is representative independent on every pair and JF(Z,·)=0 on the radical (proof reconstructed and written out from the Killing equation and the structure identity). |
| D03-GEOM-03 (S1-M21) | applied | `geo:class-curvature` gives involutivity and basicness at constant rank without freeness; `geo:shape-pullback` restricted to the locally free statements; 02_geometry:177 "always" replaced; c06:137 aligned. |
| D03-GEOM-04 | other owner | ch05 clause (G) (A-STD). |
| D03-GEOM-05 (S1-L15) | applied in scope | 02_geometry after `geo:minimal-contraction`: reduction of the signed identity for collective minimizers (reconstructed). Abstract attribution to C-SUM. |
| D03-GEOM-06 (S1-L16) | applied | Kummer1981 and MarsdenMontgomeryRatiu1990 cited (ch02_geometry, c06); novelty sentence replaced; Hermitian analogue cited to ProvostVallee1980. Stale `LewisNigamOlver2003` record is in references.bib (C-SUM). |
| D03-GEOM-07 (S1-L17) | applied | (i) expected squared score (checked: unweighted sum 4(1+ρ²)/(1−ρ²)²); (ii) transverse vs bundle-like metric; (iii)-(v) companion c06 wording. |
| D03-GEOM-08 (S1-M22) | deferred (extension), text added | `open:geo-gauged-flow`: energy-level consistency stated as a first step ("should converge"), flows named as the hard part. |
| GAP08-S1-05 (S1-L43) | applied | `geo:moving-channel`: centering hypothesis derived (uniqueness of the L² derivative) instead of assumed; proof cites `geo:joint-fisher`; linear-Gaussian sign-change example (recomputed: I(Ȳ)=θ²(4θ²+6)/(θ²+1)²). |
| GAP08-S1-07 (s02 parts) | applied | see D02-LINKS-10 row. s04 information-state citation is A-STD's. |
| GAP09-S1-06 (ch02:39 part) | applied | evaluation sigma-algebra named at ch02:39. Gray Lemma 5.2.2 locator at ch03:26 is B-DYN's. |
| D22-COMM-05 | applied in scope | `prop:int-loop-defect` stated for any closed walk, retitled "Covariance-dressed loop defect"; remark that Q is not a function of the conjugacy class (Q(I+E12; diag(1,4)) = 2, checked). ch17:249 is B-LINKS's. |
| D22-COMM-01 | applied in scope | 01: codes are not components of the generative model; a shared language is a separate declared datum. ch17:30 and the abstract are other owners'. |
| XC1-GAUGE-02 (S4-M11) | applied | 01 dressing paragraph: naturality vs active equivariance, stabilizer reduction, Francois2019; ch02 hypothesis section rewritten (naturality, active equivariance, which data pass); the unsupported promise "every result below ... says so" removed; c02 claim of a standing hypothesis corrected. AUTHORING:53 is C-SUM's. |
| XC1-GAUGE-09 (S4-L07) | applied (a),(b); (c),(d) other owners | (a) ch02:22, base_connection:69, c02; (b) one h ∈ G acting through both representations; base_connection notes the reference change is the inverse of a target change (for (d)). |
| XC4-CONS-13 (ch02:115 part) | applied | `prop:int-mismatch` proof chain displayed. |
| XC4-CONS-14 / XC2-FRAME-08 (label prefixes) | deferred | `open:geo-gauged-flow` is now cited from ch15:596 (B-CTX) and `open:conn-derived` from ch16; renaming would break other owners' references mid-wave. A rename to `op:` should be done once by the lead with a global replace. |
| XC4-CONS-08 (ch01:173 part) | applied | ℓ_i(k,m;do)=l_i(o∣k,m)λ_O(do) reconciles the two record-kernel notations. |
| XC6-PROG-11 (S4-M14) | applied in scope | 01: generative models contain no action, policy or learning; free energies are perceptual. ch03 statement and outlook open problems are B-DYN / C-SUM. |
| XC6-PROG-12 (S4-M15) | applied in scope | 02_geometry:7: within the formation constructions of chs 3 to 12 the gauge structure enters only through discarded orbits. Outlook part is C-SUM's. |
| XC6-PROG-06 (ch01 part) | applied in part | agent type now includes the generative model (prerequisite of type closure); the standard's type-closure and efficacy clauses are A-STD's. |
| S4-M09 / GAP04-S4-01 (01:117) | applied in part | 01:117 now requires joint Borel measurability of (t,x)↦Φ_t(x) inside history laws, notes it holds for continuous flows on finite-dimensional or separable charts (Lusin-Souslin), and points to `def:adaptive-history`. Proposition M (removability for gradient chart flows) belongs to ch03 (B-DYN); 01:117 can then cite it. |
| S4-M26 / GAP04-S4-03 | other owner | Proposition M in ch03 (B-DYN). |
| S4-L25 / GAP04-S4-05 | applied in scope | `def:adaptive-history` supplies the standing declaration (Borel policies, jointly Borel record densities) that chs 7, 10, 12, 13, 14 can cite. |
| D12-SMOOTH-04 | applied in scope | `def:adaptive-history` and `eq:record-flow-kernel` give the record law, joint measurability and stopping rule; `def:dyn-record-interrupted` (ch03) and `smooth:feedback` (ch07) are other owners'. |
| D21-BT-01 (base_connection part) | applied | base_connection curved example: real-spectrum curvature cannot come from a metric-preserving selector (metric connections have M-skew curvature), cites `ibt:eq:metric-compatibility`. Corollary in ch16 is B-CTX's. |
| D21-BT-12 | applied in scope | new figure `companion/figures/cross_context_map.tex` (same-context link vs cross-context map C(c',c)=X(c')Q_X(c)); \input in c16 is B-CTX's. Not compiled. |
| D19-CTXF-01 (02_geometry:7 part) | applied | qualifier "while the minimizing transports stay on their regular branch". |
| D20-CTXC-02 (S3-X14) | applied as precise remark | `open:geo-gauged-flow`: eliminated locally gauge-invariant energy exerts no frame force; row Dirichlet term as the available constitutive choice. |
| D19-CTXF-05 (S3-X11), D20-CTXC-08, D21-BT-04 / GAP10-S3-04 (S3-X19), D21-BT-13 | deferred (extensions / other owners) | ch15/ch16 statements; no text change in my files. |
| D04-DYN-07 | other owner | ch03 (channels read own sector only). def:int-channel unchanged. |
| XC1-GAUGE-08, XC2-FRAME-03 | other owners | ch17/ch19/Item 7; ch01:4 is C-SUM's. |
| XC6-PROG-01 (C01) | deferred (extension) | not implemented. |
| s01_probability | no change | No valid finding requires a change (all identities reconstructed correct by D01-FOUND, XC8-REMED); Kallenberg locator unverified. |

## BIB ENTRIES

Verified against Crossref (DOI records) on 2026-10-29; Kummer's title and end page are from memory (Crossref record lacks title; volume 30, issue 2, first page 281 confirmed).

```bibtex
@article{Kummer1981,
  author  = {Kummer, Martin},
  title   = {On the Construction of the Reduced Phase Space of a {H}amiltonian System with Symmetry},
  journal = {Indiana University Mathematics Journal},
  volume  = {30},
  number  = {2},
  pages   = {281--291},
  year    = {1981},
  doi     = {10.1512/iumj.1981.30.30022}
}
@book{MarsdenMontgomeryRatiu1990,
  author    = {Marsden, Jerrold E. and Montgomery, Richard and Ratiu, Tudor},
  title     = {Reduction, Symmetry, and Phases in Mechanics},
  series    = {Memoirs of the American Mathematical Society},
  volume    = {88},
  number    = {436},
  publisher = {American Mathematical Society},
  address   = {Providence, RI},
  year      = {1990},
  doi       = {10.1090/memo/0436}
}
@article{Francois2019,
  author  = {Fran{\c{c}}ois, Jordan},
  title   = {Artificial versus Substantial Gauge Symmetries: A Criterion and an Application to the Electroweak Model},
  journal = {Philosophy of Science},
  volume  = {86},
  number  = {3},
  pages   = {472--496},
  year    = {2019},
  doi     = {10.1086/703571}
}
@article{Wilcox1967,
  author  = {Wilcox, R. M.},
  title   = {Exponential Operators and Parameter Differentiation in Quantum Physics},
  journal = {Journal of Mathematical Physics},
  volume  = {8},
  number  = {4},
  pages   = {962--982},
  year    = {1967},
  doi     = {10.1063/1.1705306}
}
@article{ProvostVallee1980,
  author  = {Provost, J. P. and Vallee, G.},
  title   = {Riemannian Structure on Manifolds of Quantum States},
  journal = {Communications in Mathematical Physics},
  volume  = {76},
  number  = {3},
  pages   = {289--301},
  year    = {1980},
  doi     = {10.1007/BF02193559}
}
```

## NEW OR CHANGED LABELS

New: `def:observable-algebra`, `prop:agent-fe-invariance`, `prop:predictive-equivalence`, `eq:model-display`, `eq:composite-record-kernel`, `eq:shared-model-frame` (all ch01); `def:adaptive-history`, `eq:record-flow-kernel` (record_process); `geo:class-curvature` (02_geometry). New file `companion/figures/cross_context_map.tex`.

Removed: `prop:int-loop-blindness` (ch02 duplicate; no references anywhere). Changed meaning: `def:shared-model` now defines sharing up to frame (one g ∈ G); the arbitrary-bijection notion is "isomorphic generative models" at `eq:shared-model` (label kept). `prop:int-loop-defect` retitled and generalized to closed walks (label kept). No other labels renamed.

## FOR OTHER OWNERS

- C-SUM: AUTHORING.md:33 (predictions agree under composite kernels, not "only with matched evaluators"; species = frame version) and :53 (naturality vs active equivariance; observable algebras `def:observable-algebra`); abstract/README/outlook/glossary H03 wording at proved scope (decision 8) and frontmatter:24; D03-GEOM-05 abstract attribution; notation appendix and glossary: new symbols (A_pop, A_prot, K^rec, K^(n), l_i, λ_O, L_o, **g**, θ^mech, Ω^mech, N_{ij,j}, α_+, σ̄), "presentation prior" for r_i, chapters 13 to 19 coverage, glossary species entry now matches; merge the five BibTeX entries; delete stale `LewisNigamOlver2003`; ch01:4 Item 7 consistency (XC2-FRAME-03); outlook open problems for action/learning (XC6-PROG-11) and gauge role (XC6-PROG-12).
- B-DYN: if `prop:dyn-no-global-vfe` changes hypotheses or gains part (c) (D01-FOUND-08, XC2-FRAME-06), recheck the two ch01 paraphrases (end of sec:generative-models; after `prop:content-law`); add Proposition M (GAP04-S4-03) and then 01:117 may cite it; `def:dyn-record-interrupted` can cite `def:adaptive-history`/`eq:record-flow-kernel` (D12-SMOOTH-04, GAP04-S4-01); ch03:202 "transported references" (XC1-GAUGE-09 c); ch03:26 Gray Lemma 5.2.2 (GAP09-S1-06); D04-DYN-07; ch03:438 species now means frame version; conditional_memory:206 is consistent.
- B-PA: requalify chs 13 and 14 against `def:observable-algebra` (the erased phase and the common ψ rotation are gauge relative to every declared algebra; the child diagnostic belongs to none); protocol declarations can cite `def:adaptive-history` (GAP04-S4-02, -05).
- A-STD: (F4) probe class relative to `def:observable-algebra`; D03-GEOM-04 clause (G); GAP08-S1-01/02.
- B-CTX: \input{companion/figures/cross_context_map} in c16 (D21-BT-12); D21-BT-01 corollary in ch16 (base_connection now states the consequence); D21-BT-10 cite `prop:int-reciprocal`, whose notation is now N_{ij,j}; D21-BT-09, D21-BT-13, D19-CTXF-01 R_glob, D20-CTXC-02 remark in ch15, XC1-GAUGE-09 (d) at ch15:44.
- B-LINKS: ch17:30 shared-language sentence (codes are not in `def:generative-model`; declare a language hypothesis); ch17:249 dissonance sentence (D22-COMM-05; `prop:int-loop-defect` now covers closed walks); `def:int-group-links` now uses reference links θ̄_ij ∈ G and can be cited from ch17; XC1-GAUGE-08. Dangling reference `it3:weak-anchor-determinant` found by the label check (not in my files).
- B-FUS / B-FORM: chs 7 and 10 history laws can cite `def:adaptive-history` (GAP04-S4-01, -05).
- B-IT7: XC1-GAUGE-08 (c).
- Lead: label-prefix cleanup `open:` → `op:` for `open:geo-gauged-flow` and `open:conn-derived` across files (XC4-CONS-14).

## AUTHOR DECISIONS NEEDED

1. Decision 8 (H03) remains open: chapter 01 now states the proved scope and the copula limit; which statement replaces the abstract sentence is the author's.
2. `def:observable-algebra` uses the time-evolved reading of "gauge" (no element separates at any time along the declared dynamics). This is the reading that keeps D2 and the D01-FOUND-01 witness consistent; if the author prefers an instantaneous definition, the chapter 13/14 requalifications are unaffected, but the loop-class witness would then be called gauge while the dynamics separates it.
