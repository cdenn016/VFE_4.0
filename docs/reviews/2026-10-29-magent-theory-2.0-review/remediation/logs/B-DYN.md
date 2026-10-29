# Remediation log B-DYN, 2026-10-29

Files: `chapters/03_dynamics.tex`, `supplements/s03_variation.tex`, `companion/c03_dynamics.tex`, `chapters/04_prediction.tex`, `appendices/conditional_memory.tex`, `appendices/interacting_memory.tex`, `companion/c04_communication.tex`, `companion/c04_interacting_memory.tex` (paths relative to `M = Research/manuscripts/MAgent_Theory_2.0/`). Nothing compiled. Every claim inserted was rederived by hand before insertion; no numerical script was run. The scripts in `remediation/scripts/B-DYN/` are text-editing helpers (splicing, a symbol rename, CRLF normalization, a structural check for balanced environments, braces and dollars). The check passes on all eight files, and all eight are CRLF again.

## Findings in scope

| ID (lens; cluster) | Status | What changed (file, label) / reason |
| :- | :- | :- |
| H03 (D05-OBJ-01, XC4-CONS-01, XC6-PROG-02, XC8-REMED-01, D04-DYN-02, XC2-FRAME-06, D01-FOUND-02 ch3 parts) | applied in ch03 and c03 | 03:4 opening restated at proved scope. `prop:dyn-no-global-vfe` restated: finite alphabets and positivity (so E is finite on the interior) for both parts; (a) for an identity channel; (b) for every fixed attended channel that is not constant in its input, with the proof extended through the mixed derivative `eq:dyn-no-global-third` and an argument that it is nonzero somewhere in every open set. A new part (c), proved: constant-output channels with equal entropy coefficients give E = cF + b exactly. I reconstructed (b) and (c) myself. New `rem:dyn-no-global-scope` records the lenses' counterexamples to the broad version: the countable auxiliary latent (D05-OBJ), the Gaussian copula on atomless spaces (XC6-PROG, D01-FOUND), constant channels (XC4-CONS, XC8-REMED); that E_red is bounded and so outside (a) (XC2-FRAME); that (b) for E_red, for law-dependent channels and on charts is not established; and the unit-variance mean-slice identity (D04-DYN, reconstructed). 03:52 qualified. c03:12 rewritten. Abstract, ch01, outlook, README, AUTHORING and glossary are listed for other owners. |
| I04 (XC7-IMPL-03; S4-H11) | applied | New `prop:dyn-gaussian-closure` after `prop:dyn-gaussian-noninvariance`. It covers per-agent and global free energies with Gaussian references, affine-mean evaluators of fixed covariance, linear-Gaussian channels and log-quadratic records. The proof shows the first variations are quadratic, so the velocity is tangent to the Gaussian family. Stated as a feature of the family; the restricted flow is the full flow wherever the full equation has unique solutions. `prop:dyn-gaussian-noninvariance` weakened from "exactly one receiver" to "at least one receiver"; the new proof uses positivity of the source terms (reconstructed). The 03:221 scope sentence now reads "full density charts or declared invariant families" (XC4-CONS-09 (i)). |
| S1-M11 (D04-DYN-01, D05-OBJ-02) | applied in my files | c03:10 now says "not a gradient flow", and that the ring still contracts. ch03 adds after `prop:dyn-nonreciprocal` that the ring has a strict Lyapunov function (normal matrix, negative-definite symmetric part). The law-dependent limit-cycle example of D05-OBJ-02 was not added: it rests on numerics I did not reproduce. AUTHORING:38 is listed for C-SUM. |
| S1-M25 (D05-OBJ-12; R14) | applied | New `prop:dyn-per-agent-contraction`: fixed rows, fixed channels without null columns, self weight at least b > 0. Global existence, unique equilibrium, contraction at rate b in max-oscillation of log ratios. Proof by Birkhoff nonexpansiveness and Banach, reconstructed. |
| S1-M12 (D05-OBJ-03), with S2-M23 and S4-M09 ch3 part | applied | `def:dyn-record-interrupted` rewritten as an instance of `def:adaptive-history` with stage kernel `eq:dyn-record-interrupted`. Hypotheses added: an admitted Borel set invariant under assimilation, a jointly Borel flow, record kernels at pre-record states, t_k → ∞, a policy, and stopping at explosion; history law by Ionescu-Tulcea. Davis attribution restricted to the time-augmented state with forced jumps. |
| S1-M13 (D05-OBJ-04) | applied | 03:426 rewritten. In the free-energy pictures the objective changes at each record; −log P(o_{1:k}) is monotone for discrete records; no functional is monotone across the composite. The stationary set is record-independent, but basin selection by a record persists. |
| S1-M14 (D05-OBJ-05) | applied | `def:dyn-per-agent` now requires incoming channels to be independent of (q_i, s_i), and says what fails otherwise. |
| S1-M15 (D05-OBJ-06), S4-M07 (XC3-SCOPE-02), D17-PA-03 and GAP01-S3-01 ch3 parts | applied | The per-agent picture is defined as a class with three named instances: mixture-source, frozen-source episodes, and the ch18 tangent models. `def:dyn-per-agent-flow` now defines the Gibbs-reduced per-agent dynamics and the chart per-agent dynamics. Local well-posedness is stated, since the proof of `prop:dyn-chart-flow` uses no gradient structure, with sufficient Lipschitz conditions, and invariance of a family follows from tangency plus uniqueness. Stability transfer between row conventions is stated as unproved for the per-agent field. The frozen-source sentence (03:419) is corrected. Verifying the hypotheses for ch13/14 data belongs to B-PA. |
| S1-M16 (GAP09-S1-01, D06-PRED-04, GAP08-S1-06) | applied | `prop:dyn-per-agent` restated: F_i defined by `eq:agent-free-energy`; hypothesis KL(Q‖P_i^0) < ∞; Donsker–Varadhan bound on the positive part of the log likelihood; identity in (−∞,∞] by the chain rule with no source densities; rows with e^{−∞}=0. A remark notes that the separated form can read ∞−∞ at the optimum. `cmem:evidence` reduced to 0 < Z(o) < ∞. `cmem:free-energy` now holds for every candidate. `cmem:gibbs` cost is now KL(q_a‖Π_a)−log Z_a, with the separated form when finite. 04:51 qualifier dropped. |
| S1-M23 (D04-DYN-03) | applied | New `prop:dyn-chart-global` for fixed channels (identity or two-sided bounded kernels), with explicit constants `eq:dyn-chart-global-constants`. Proof by essential-infimum and essential-supremum Dini estimates, reconstructed and corrected: the kernel receiver bound uses (u−m)+max(0,−log c). The finite-time-exit example for law-dependent channels was not added (numerical only); the text says no global statement holds for law-dependent channels. |
| S1-M24, S4-M23 (D05-OBJ-07, XC6-PROG-13 part one) | applied | New `prop:dyn-per-agent-not-global`, proof reconstructed: with fixed channels, per-agent forces equal mean-field block forces on an open set only if every attended peer channel is constant in its input. `op:dyn-three-pictures` restated: law-dependent case, and an explicit map for "same stationary points". The directed-edge separation is noted. The deconvolution criterion of XC6-PROG-13 part two is deferred. |
| S4-M25 (XC8-REMED-12) and 8.1 item 21 | decided; citations for others | grep over all .tex finds no `\cref{prop:dyn-chart-flow}` or `prop:dyn-global-flow` outside ch03. 07:195 cites `thm:dyn-finite-flow`, and 09:33 declares its own chart flow. XC8-REMED-12 and S4-M25 are correct, and D04-DYN's prior-review record is wrong. The global-existence half is supplied by `prop:dyn-chart-global`. The citations needed are listed below. |
| S1-L18 (D04-DYN-04) | applied | 03:194 now says the flows share critical points, Morse indices and nullities, hence linear stability at nondegenerate points; local minima correspond (proved via E ≥ E_red). No Lyapunov-stability claim at degenerate points. |
| S1-L19, S4-L20 (D04-DYN-05, XC3-SCOPE-03) | applied | 03:213 rewritten: score-pairing projection; Fisher-orthogonal only with finite Fisher norm (condition 2Σ_k^{-1}−Σ_i^{-1} ≻ 0, rederived); no well-posedness result covers the full flow from Gaussian data. |
| S1-L20 (D04-DYN-06) | applied | Registered model: no kinetic metric (03:194); prime-n obstruction and the m-type lattice (03:389, divisibility rederived); stochastic temperature stated up to the time-unit factor. |
| S1-L21 (D04-DYN-07) | applied | 03:26 channels depend only on own-sector laws in this chapter. |
| S1-L22 (D04-DYN-08, GAP09-S1-06) | partly applied | Topsøe 1979 for `prop:dyn-compensation` (bib verified at kybernetika.cz). Gray Lemma 5.2.2 locator and the Dobrushin attribution (GAP09-S1-06 read the corrected edition). The singular-limit sentence is reworded as a formal Tikhonov-type limit, not proved. Deferred: the Čencov locator and Absil–Kurdyka (the claim they support was removed instead). |
| S1-L23 (D04-DYN-09, D05-OBJ-11) | applied | c03:8 heat-bath qualifier; c03:64 pooled law only with one common fixed channel; c03:94 names three exact settings. |
| S1-L24 (D04-DYN-10) | applied | `prop:dyn-global-flow`: null column implied, N defined, model constant written out. |
| S1-L25 (D05-OBJ-08) | applied | (i) dF_i/dt has an uncontrolled pairing term; (ii) F_tot is of the evidence-augmented form with a cross-sector self cost; (iii) binary mixed partials (rederived), and three-point case kept; (iv) frozen rows declared, plus a proof sketch that the complex pair survives with evolving rows at equal evaluators; (v) tempering: only the row block is Bissiri et al.; (vi) positivity of l_i added to `def:dyn-per-agent-flow`. |
| S1-L26 (D05-OBJ-09) | applied | Labels as blocks; schedule claim restricted; the meaning of "local free energy" fixed in ch03, with a cross-statement in imem. |
| S1-L27 (D05-OBJ-10) | applied | Within the H03 restatement (finiteness and positivity; finite alphabets for (b)). |
| S4-L10 (XC3-SCOPE-08) | applied | Within the 03:357 rewrite. |
| S4-L13 (XC4-CONS-08) | no action | Lens verified the names agree; no defect in my files. |
| S4-L14 (XC4-CONS-09 (i)) | applied | 03:221. The other parts are in other owners' files. |
| S4-L22 (XC3-SCOPE-07) | applied | `prop:dyn-global-local`: λ_i-a.e. integrability; uniqueness over competitors with the same integrability. |
| S4-L34 (XC5-LIT-14) | partly applied | Text now names Łojasiewicz's trajectory theorem and places Absil–Mahony–Andrews as the discrete counterpart. The Łojasiewicz 1984 bib entry is deferred (not verified). |
| S4-L35 (GAP04-S4-04) | applied | 03:194 and 03:234 state the general-space hypotheses (local, weakly continuous) with Ay–Jost–Lê–Schwachhöfer. Theorem number omitted (not checked). |
| D14-BND-12 (ch3 part) | applied | 03:435 exception for `rem:boundary-evidence`. |
| GAP04-S4-03 (S4-M26) | deferred | Proposition M (automatic Borel measurability of gradient chart flows): its 15-line proof relies on locators the lens gives from memory, and I did not reconstruct it at checkable cost. The record-interrupted definition takes joint measurability as a hypothesis instead. |
| GAP04-S4-06 | deferred | Extension (weak-strong uniqueness); no text change requested in my files beyond an optional remark. |
| S1-L29 (D06-PRED-05) | applied | 04:37 "when KL(s‖t) is finite", with the counterexample. |
| S1-L30 (D06-PRED-06) | partly applied | 04 coarse map C renamed τ in `prop:common-family-recovery` and `cor:local-family-recovery`; "minimizes KL(p‖q_G)" fixed in 04:65 and c04:56; companion S_a renamed P_{Y,a}. Deferred: the μ, Z, T, M collisions, and the figure node (B-FOUND). |
| S1-L32 (D06-PRED-08) | applied | 04:14 distinguishes model prediction and belief prediction. |
| S1-L33 (D06-PRED-09) | partly applied | Kullback–Leibler 1951 at 04:37; Skibinsky moved to `cor:local-family-recovery`; Baum–Petrie at `cmem:filter`. Deferred: Jeffrey 1965 (bib not verified), Courtade–Weissman locator. |
| S1-L35 (D06-PRED-11) | partly applied | Rigorous non-stationarity of the moment carrier for a ≥ √(2/(π−2)), added after `cmem:reverse-counterexample`. Heat and Stein identities rederived. Part (i), the exclusive carrier comparison, is deferred. |
| S1-L39 (D06-PRED-15) | deferred | Extension (Gaussian CCA optimizer). |
| S1-M26 (D06-PRED-01) | applied as remark | Tempered-posterior solution of the joint-law Fisher flow on finite tables (two-line derivation) after `cmem:product-label`. Densities not claimed. |
| S1-M27 (D06-PRED-02) | deferred | Extension. The cmem:filter text now says that a result-as-next-cue composition is a different model needing its own filter. |
| S1-M28 (D06-PRED-03) | applied as remark | Cue records carry no information about K_m (Z(o) independent of K). EM rule deferred. |
| S1-L31 (D06-PRED-07) | applied | "bank" defined; token-frame sentence rewritten in agent-frame terms; diagonal sub-parametrization declared. |
| S1-L34 (D06-PRED-10) | partly applied | Fourier equivalence stated: p K_c = q_G K_c iff p is Gaussian. The finite-mixture identifiability part is deferred (no verified citation). |
| S1-L36 (D06-PRED-12) | applied | `cmem:response-bound`, Schur-complement proof. |
| S1-L37 (D06-PRED-13) | applied | Augmented-record shared model, cmem:206. |
| S1-L38 (D06-PRED-14) | applied | Filter sufficient for the whole future; current readouts need (w_l, o_l, C_l). |
| S1-L40 (GAP08-S1-03, GAP09-S1-04) | applied | Normalizer proviso replaced (AM–GM bound; positivity is the condition). |
| S1-L41 (GAP09-S1-03) | partly applied | Exact block decrease and convergence to stationary points of the structured family, not to its minimizer. The Λ < 1 convexity criterion and the two-fixed-point numbers are deferred (not reconstructed). |
| S1-L42 (GAP09-S1-05) | applied | σ-finite λ, jointly measurable l; disintegration of the joint law along (M,V) gives a jointly measurable R; normalized extensions; pointwise Tonelli agreement. Kallenberg theorem number not added. |
| I01 (D07-IMEM1-01, -02, GAP09-S1-02) | applied, decision-independent part | New `imem:product-tilt` (with `imem:evidence-factors`, `imem:product-tilt-formula`): product tilt; consequences (i)–(iii), including fixed cross-ratios and fixed off-diagonal precision; uniqueness from marginals, proved for finite alphabets (zeros allowed) and for integrable log tilts (covers the Gaussian case). BJKS 2004 Lemma 6.7 attribution. imem:213 and :215 rewritten; the :580 remark split into covariance and information coordinates; `imem:retention` carries the time index and schedule; the binary example notes the odds ratio 81/49 and two changing numbers given λ_0. c04 :87, :89 rewritten. Not used: the Ghosal–Nutz–Bernton lemma for the singular infinite case (not read). AUTHORING:9, README:20 and outlook:8 for C-SUM. |
| S1-M32 (D08-IMEM2-03), 8.1 item 3 | deferred (decision 10) | The text states both true facts: two changing numbers suffice with λ_0 fixed, and λ_t has three independent weights. It makes no minimality claim. D08's Brook-lemma minimality proposition is not added. |
| S1-M17, S4-M19 (D07-IMEM1-03, D08-IMEM2-07, XC5-LIT-02) | applied | Related-work paragraph in the imem intro, the 04:86 summary and the c04 opening: Aumann 1976, Geanakoplos–Polemarchakis 1982, Deshpande–Mossel–Sohn 2022 (arXiv page checked; the theorem content is as the lens reports). Nayyar common-information belief identified; Kalman information form at `imem:gaussian-closure`; Sorenson–Alspach at the nonclosure remark. The MH naming game and Griffiths–Kalish are not added (bib not verified). |
| S1-M29 (D07-IMEM1-04, D08-IMEM2-12) | deferred, open problem | Asymptotic revelation and agreement added as a question in `imem:open`. |
| S1-M30 (D08-IMEM2-01) | deferred | Extension (local-retention failure without a public record for d_0 ≠ d_1); exact-rational computation not reproduced. |
| S1-M31 (D08-IMEM2-02) | applied | Corollary paragraph after `imem:mixture-nonclosure`: λ_1 is not a finite Gaussian mixture through its D_0-marginal (bounded-convergence argument reconstructed); the D_1-marginal is two-component. |
| S1-L44 (D07-IMEM1-05) | applied | Line 215 now says the likelihood is always a product; Nayyar belief; Cover–Thomas at the information identity; Donsker–Varadhan at the free energies. |
| S1-L45 (D07-IMEM1-06) | applied | Version independence; pathwise version convention; receiver-denominator condition in `imem:sender-receiver`; policy versions as functions of the reduced state in the retention sentence. The randomization-lemma remark is not added (law-level text). |
| S1-L46 (D07-IMEM1-07, D08-IMEM2-10) | applied | Schedule "fixed known, alternating in the examples" (imem:39, 04:76, c04:21); retention carries t and (j_s). |
| S1-L47 (D07-IMEM1-08, D08-IMEM2-09) | partly applied | Parity record C renamed χ (appendix and companion); S_t renamed Σ^D_t; companion sensor variances C → N; companion evidence Z_t → 𝒵_t; W^{Z_i} → W_i. Deferred: the a, b, P, L, Q, R overloads. |
| S1-L48 (D07-IMEM1-09) | applied | s_{i,n}, q_{i,n} defined; 𝓛_t defined before use (`imem:evidence-factors`); information identity with the single finiteness condition and the bound; "lossy parent compression" rephrased in the vocabulary of `inh:two-retentions`. |
| S1-L49 (D07-IMEM1-10) | applied | Sentence after imem:61: own past samples are conditionally independent of E, so the exclusion loses nothing (factorization argument). |
| S1-L50 (D07-IMEM1-11) | deferred | Extension. |
| S1-L51 (D08-IMEM2-04) | applied | h ≠ 0 quantifier, the h = 0 case, h < 0 symmetry. |
| S1-L52 (D08-IMEM2-05) | applied | Enlarged information fields declared in `imem:local-retention-failure`, or equivalently two initializations; same in the companion and 04:86. |
| S1-L54 (D08-IMEM2-08) | partly applied | History independence of Σ^D_t added to the theorem with proof. Items 2 and 3 are deferred. |
| S1-L55 (D08-IMEM2-11) | applied | 44/137, 93/137 (appendix and companion); "column" slip fixed. |
| S1-L56 (GAP09-S1-07) | applied | imem:52: a jointly measurable disintegration always exists (kernel disintegration of the identity kernel); reference-free form of `imem:bayes`. |
| S1-M33 (GAP08-S1-01) | not mine | Located in s04 (A-STD) and ch05; imem:52 touched only as above. |
| S1-L28 (GAP08-S1-04) | applied | `prop:sup-row-equivalence` restated for a general receiver with finite second moments and entropy (`eq:sup-row-constant-general`), with the Gaussian case as instance, the affine-family derivative, and the zero-record caveat. All rederived. |
| XC6-PROG-14, XC6-PROG-21, XC5-LIT-13 (ch3/cmem parts) | deferred | Citation and extension prose without verified primary sources in this pass. |

Counts: 69 rows. 49 applied (including two as remarks and the item-21 decision), 9 partly applied (the deferred remainder named in each row), 9 deferred (one pending decision 10, one as an open problem), 1 with no action needed, and 1 outside my files. No finding was rejected. Every finding I checked held at the scope I implemented, except two corrections to lens proof sketches. In D04-DYN-03, the kernel receiver term needs (u−m)+max(0,−log c), not max(0,−log c). In D08-IMEM2-03 / I01 the counting disagreement is definitional, not mathematical.

## BIB ENTRIES

All data were checked against Crossref, arXiv or the journal page in this session.

```bibtex
@article{Topsoe1979,
  author  = {Tops{\o}e, Flemming},
  title   = {Information-Theoretical Optimization Techniques},
  journal = {Kybernetika},
  volume  = {15},
  number  = {1},
  pages   = {8--27},
  year    = {1979}
}
@article{BarYossefEtAl2004,
  author  = {Bar-Yossef, Ziv and Jayram, T. S. and Kumar, Ravi and Sivakumar, D.},
  title   = {An Information Statistics Approach to Data Stream and Communication Complexity},
  journal = {Journal of Computer and System Sciences},
  volume  = {68},
  number  = {4},
  pages   = {702--732},
  year    = {2004},
  doi     = {10.1016/j.jcss.2003.11.006}
}
@misc{DeshpandeMosselSohn2022,
  author        = {Deshpande, Yash and Mossel, Elchanan and Sohn, Youngtak},
  title         = {Agreement and Statistical Efficiency in {B}ayesian Perception Models},
  year          = {2022},
  eprint        = {2205.11561},
  archivePrefix = {arXiv},
  primaryClass  = {math.ST},
  note          = {Version 3, 9 August 2023}
}
@article{Aumann1976,
  author  = {Aumann, Robert J.},
  title   = {Agreeing to Disagree},
  journal = {The Annals of Statistics},
  volume  = {4},
  number  = {6},
  pages   = {1236--1239},
  year    = {1976},
  doi     = {10.1214/aos/1176343654}
}
@article{GeanakoplosPolemarchakis1982,
  author  = {Geanakoplos, John D. and Polemarchakis, Heraklis M.},
  title   = {We Can't Disagree Forever},
  journal = {Journal of Economic Theory},
  volume  = {28},
  number  = {1},
  pages   = {192--200},
  year    = {1982},
  doi     = {10.1016/0022-0531(82)90099-0}
}
@article{SorensonAlspach1971,
  author  = {Sorenson, H. W. and Alspach, D. L.},
  title   = {Recursive {B}ayesian Estimation Using {G}aussian Sums},
  journal = {Automatica},
  volume  = {7},
  number  = {4},
  pages   = {465--479},
  year    = {1971},
  doi     = {10.1016/0005-1098(71)90097-5}
}
@article{Kalman1960,
  author  = {Kalman, R. E.},
  title   = {A New Approach to Linear Filtering and Prediction Problems},
  journal = {Journal of Basic Engineering},
  volume  = {82},
  number  = {1},
  pages   = {35--45},
  year    = {1960},
  doi     = {10.1115/1.3662552}
}
```

`Kalman1960` already exists in `item1_refs.bib` with the same data and no DOI; it is not yet in `references.bib`. Existing keys reused: KullbackLeibler1951, Skibinsky1967, BaumPetrie1966, CoverThomas2006, DonskerVaradhan1975, AyJostLeSchwachhofer2015, Gray1990, Kallenberg2021, Davis1984, Lojasiewicz1963, AbsilMahonyAndrews2005, Nayyar2013CommonInformation, BissiriHolmesWalker2016.

## NEW OR CHANGED LABELS

New labels, each unique across the three documents: `prop:dyn-gaussian-closure`, `prop:dyn-chart-global`, `eq:dyn-chart-global-constants`, `prop:dyn-per-agent-contraction`, `prop:dyn-per-agent-not-global`, `rem:dyn-no-global-scope`, `eq:dyn-no-global-third`, `eq:dyn-record-interrupted` (ch03); `eq:sup-row-constant-general` (s03); `cmem:response-bound`; `imem:product-tilt`, `imem:product-tilt-formula`, `imem:evidence-factors`. No label was removed or renamed. Statements changed under existing labels: `prop:dyn-no-global-vfe` (parts (a), (b), new (c)), `prop:dyn-per-agent`, `def:dyn-per-agent`, `def:dyn-per-agent-flow` (now also defines the Gibbs-reduced and chart per-agent dynamics), `def:dyn-record-interrupted`, `prop:dyn-gaussian-noninvariance`, `prop:dyn-global-local`, `op:dyn-three-pictures`, `prop:sup-row-equivalence`, `cmem:evidence`, `cmem:free-energy`, `cmem:gibbs`, `imem:retention` (the tuple now has (j_s)_s and t in place of j_{t+1}), `imem:mixture-nonclosure`, `imem:local-retention-failure` (the parity record is now χ), `imem:gaussian-closure` (covariance now Σ^D_t), `imem:open`.

## FOR OTHER OWNERS

- **C-SUM.**
  - Restate at the proved scope of the new `prop:dyn-no-global-vfe` and `rem:dyn-no-global-scope`: `frontmatter.tex:5`, `:24`; `chapters/06_outlook.tex:6`; `README.md:34`; `AUTHORING.md:39`; `companion/glossary.tex:20`.
  - `AUTHORING.md:38`: "generally has no global Lyapunov functional" should become "is in general not a gradient flow". The positive case is `prop:dyn-per-agent-contraction`.
  - `AUTHORING.md:9`, `README.md:20`, `chapters/06_outlook.tex:8`: the exact changing state is the product tilt of `imem:product-tilt`. With the fixed λ_0 the two marginals determine λ_t, and nonclosure now also covers λ_1.
  - The AUTHORING "Regularity" paragraph could cite `prop:dyn-chart-global` and the Gibbs-reduced and chart per-agent dynamics of `def:dyn-per-agent-flow`.
  - Merge the BIB ENTRIES above.
- **B-FOUND.**
  - `chapters/01_foundations.tex:197`, `:226` (D01-FOUND-02, D05-OBJ-01): restrict to the proved scope. On infinite spaces the copula identification of `rem:dyn-no-global-scope` falsifies "with these marginals as recognition laws".
  - `companion/figures/message_reconstruction.tex`: relabel the node ν = T_#q as μ = T_#q (D06-PRED-06, XC4-CONS-09 (vi)).
  - `01:117` measurability promise (GAP04-S4-03): `def:dyn-record-interrupted` now takes joint Borel measurability of the flow as a hypothesis.
- **B-FUS.**
  - `chapters/07_smooth_closure.tex:195-215` and `chapters/09_public_compression.tex:33` should cite `prop:dyn-chart-flow` (item 21, XC8-REMED-12, D13-ERAPUB-12). The ch07 public flow with identity or bounded kernels can get global existence from the argument of `prop:dyn-chart-global` with the forcing bound 0 ≤ H ≤ M.
  - `07:294` uses `def:dyn-record-interrupted`, which now requires an admitted set invariant under assimilation, a jointly Borel flow, and record kernels at pre-record states.
- **B-FORM.**
  - `chapters/10_boundary_closure.tex:48-71` (`lem:boundary-fine-wellposed`) and `chapters/12_open_formation.tex:12, :67` should cite `prop:dyn-chart-flow`; `lem:boundary-global` can cite `prop:dyn-chart-global` (D14-BND-11, XC8-REMED-12).
  - `10:172`, `:223`, `:252`: check the new hypotheses of `def:dyn-record-interrupted`.
  - `rem:boundary-evidence`: identify its term as a case of `eq:dyn-evidence-potential` (D14-BND-12, second half).
- **B-PA.**
  - Chapters 13 and 14 should cite the Gibbs-reduced and chart per-agent dynamics now defined in `def:dyn-per-agent-flow`, verify its Lipschitz hypotheses for their data, and conclude invariance from tangency plus uniqueness (XC3-SCOPE-02, D17-PA-03, GAP01-S3-01).
  - Rename the per-agent self cost written D^q_ii at 13:9: ch03 now says this cost is not the identity-channel D_ii = 0.
  - Stability transfer between row conventions is unproved for per-agent fields.
- **B-LINKS.** ch03 now names the ch18 tangent models as an instance of the per-agent class (D24-LDYN-04). `18:7` can cite `prop:dyn-gaussian-closure`, which makes the global and per-agent tangent flows exact flows on Gaussian laws. `it2:coupled-global` can use the Łojasiewicz step as in `prop:dyn-global-flow` (D25-CRIT-03).
- **A-STD.** `inh:null-flow` refers to `def:dyn-per-agent-flow`, and 05:1680 already mentions the Gibbs-reduced version, which is now defined in ch03. A null for the global free energy is still needed (D10-FORM-05).

## AUTHOR DECISIONS NEEDED

1. Decision 8 (no-go): which proved statements the abstract-level sentence uses. The chapter now contains (a), (b) for every input-dependent fixed channel, the converse (c), and the scope remark. Whether to prove (b) for E_red, for law-dependent channels or in charts, or to restrict the abstract to fixed channels is open.
2. Decision 10 (changing state of the interacting protocol): the text states that two changing numbers suffice with λ_0 fixed, and makes no minimality claim. Whether to add D08-IMEM2-03's interface-minimality proposition (three weights minimal for a two-step kernel interface) is the author's choice.
