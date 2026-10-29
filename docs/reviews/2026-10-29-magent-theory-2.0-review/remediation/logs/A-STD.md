# A-STD remediation log (2026-10-29)

Files edited (manuscript root `Research/manuscripts/MAgent_Theory_2.0/`, branch `magent-2.0-remediation-2026-10-29`, nothing committed): `chapters/05_inheritance.tex`, `supplements/s04_inheritance.tex`, `companion/c05_inheritance.tex`. Nothing compiled. Line endings kept CRLF.

Scripts (`V/remediation/scripts/A-STD/`): `anchored_consensus.py` (+ `.out`; single-threaded, under 30 s) checks the calibration example; `proof_env.py`, `eqref_to_cref.py`, `fix_linestart_cref.py` are text-edit helpers.

What `anchored_consensus.out` shows, and what was proved by hand: starting at x(0) = c, every difference is (c_i - c_j) m(t) with m(t) = (1 + 2e^{-3t})/3. For delta in (53/120, 0.6) the full flow gives {1,2} -> {3,4} -> all. The null flows give no event. The frozen-environment control gives the same sequence as the full flow. Fifteen perturbations of size 0.01 at delta = 0.5 give the same three verdicts. These closed forms are the proof of `inh:anchored-consensus` (i). The open-set part rests on transversality of the crossings.

## Findings table

| ID | Status | What changed (file, label) / reason |
| :- | :- | :- |
| H01 (ch05 part) | applied | `inh:null-flow` now declares both nulls: (N1) label-preserving, with self weight `inh:null-self-row`; (N2) renormalized, self weight one. Chapter 11 part is B-FORM's. |
| H04 (ch05 part) | applied | `inh:null-flow` now covers: what counts as a peer cost; both nulls; per-agent (dynamic and Gibbs-reduced) and global-free-energy nulls; domain, chart and well-posedness requirement; existence on the admitted class; nonsmooth objectives by declaration; selector readout convention; post-handoff null (stays the fine flow; selector evaluated on fine block statistics); histories on the horizon. |
| H05 (ch05 part), D10-FORM-01, XC2-FRAME-01 (ch05 part), D09-INH-04(b,c), XC6-PROG-05(iii) | applied | New `inh:protocol-classes`. Rules (A) (parent protocols, with D_a) and (P) (child-level probes B_I = what agents, channels, records and declared external queriers can apply to members before the handoff; it contains A_J of the merged blocks). Both rules are functions of the block and the constitutive data only. This is the content of "fixed before the retention". (F4) rewritten: some b in B_I separates two points of a fiber in Omega^reach. Equivalently, the erased distinction is not gauge relative to B_I (chapter 01 sense kept, per D2). (F4) attributes (persistent or transient; symmetry type) are reported. The optional non-decay clause of XC2-FRAME-01 is not made a clause, only an attribute. |
| H06, D10-FORM-02, XC6-PROG-05(iv), XC2-FRAME-05(iv), D15-WISH-07, XC8-REMED-09 | applied | New `inh:counterfactual-class` (R14 type). It contains the actual handoffs plus same-time handoffs from a declared larger class U on which the first stage has not occurred, and excludes states outside the selector domain. (F6b) is now an interior condition (deterministic) or a margin (stochastic). Supplied-scale and first-stage-dependence cases are stated. Lens disagreement (Section 8.5): XC2-FRAME-05's "states in the selector domain" alone was not adopted; R14 was adopted per D3, with the domain exclusion kept. |
| H07, D10-FORM-04, XC6-PROG-05(i,ii), XC2-FRAME-05(i-iii), D09-INH-04(a,d) | applied | (F1): chart topology of the declared dynamics (full grade), or relative openness in a declared invariant family that is not a finite union of orbits (family grade, codimension reported). (F5) is pointwise on an open U. Omega_I^reach is defined in `inh:inherited-parent`, and (F4) is evaluated there. P_I contains the post-handoff laws. `inh:trajectory-remark` wording aligned. |
| H08, XC2-FRAME-02 (ch05 part) | applied | (F2): the selector family S_c, c in C, is fixed before the flows are analyzed, and each real constant ranges over a set with nonempty interior. (F5) holds only for every c in C. The verdict reports the subset of C where it holds, plus a selector-free observable. The chapter 13 numbers are B-PA's. |
| H09, D09-INH-01 | applied | A peer cost is now one whose compared law depends on another agent's law; constant-output costs are self terms and are kept. New `inh:no-interaction-null`: non-interacting agents give Phi^0 = Phi^fr = Phi and fail (F5). Frozen-environment control (N3) stated and required in every (F5) report. |
| H10, D10-FORM-03, S2-E05 (a) | applied | New `inh:anchored-consensus` (calibration), proved in closed form at x(0)=c with an open-set extension; checked by script. Verdict: two-stage selection with exact inheritance that fails (F6b). It passes (F5) in the new strong grade, and P^fr = P. Note: D10-FORM-03's statement that the strong form fails held for the old strong form only; under the redefined grade the example passes it. |
| I02, D10-FORM-15 | applied | Standard rewritten in `inh:recursive-formation` with (N), (F1)-(F6), the grades and the classification rule ("undetermined" when a clause is neither established nor refuted). "Derive the selector" made precise in `inh:formation-open`. |
| S2-M03, D10-FORM-06, XC2-FRAME-05(iii) | applied | Horizon convention: on [0,inf) the sequences are compared directly; on a finite window, P_[0,tau] must not be an initial segment of the null's full sequence. The text explains the clock-doubling case. |
| S2-M04, D16-OPEN-02 | applied | P is defined modulo exchange of consecutive merges of disjoint agent sets. |
| S2-L02, D10-FORM-13, XC2-FRAME-05(i,ii) | applied | Strong grade redefined: (F5) at every point, for every c, against both nulls. Amplifying grade defined: declared metric, evaluated at Phi's events on Phi's partition along both nulls, with a degenerate case. "Final partition" is no longer used. |
| S2-M01, S4-M01, D10-FORM-08, D09-INH-06(a), D15-WISH-02, XC2-FRAME-04, XC8-REMED-02, XC6-PROG-17 (ch05/c05 parts) | applied | The remark after `inh:common-clock` now reads: tilt-invariant selectors fail (F5). Interaction sensitivity requires a failed hypothesis; the hypotheses are listed, including the shared geometric-mean target. Divergence-threshold selectors can pass (F5). The companion was rewritten to match, with the shared-evaluator and self-exclusion hypotheses. |
| S4-M31, XC8-REMED-13 | applied | New corollary `inh:common-tilt` (identity `inh:common-tilt-identity`): the full flow is a common tilt of the null at doubled time. Proved from the proposition's identity. |
| S2-M02, D10-FORM-07, D09-INH-06(b) | applied | The section introduction and the remark after `inh:scalar-additive` now say the result constrains the full flow only, and give the null-flow condition. The companion paragraph was restated with its hypotheses. |
| D15-WISH-03, D15-WISH-05, XC8-REMED-05, XC2-FRAME-08 (dup part) | applied | `inh:common-clock` now has case (a) (self excluded) and case (b) (self included, same channel), with the null conclusions for each convention. It cites `form:mean-field` as the case-(b) instance and `form:scalar-closure` as the same Jensen argument. Chapter 11 must cite back (B-FORM). |
| D15-WISH-09 | applied | Sentence after `inh:scalar-additive`: inside the common-clock class the propagator is -2 times the identity. |
| S2-M09, D10-FORM-09, D09-INH-12, D15-WISH-04, S4-L19, XC8-REMED-08 (ch05/c05 parts) | applied | One definition of "two-stage selection" ("shown to fail"). The companion uses it. "(F6a) for the first sentence" removed. AUTHORING, README and glossary are C-SUM's. |
| S2-M12, D09-INH-02 | applied | `inh:no-underdamped` retitled ("... at images of critical points"). It gains the fiber-minimum clause with a proof, and the Poisson/rotating-frame counterexample D3 replaces the old remark. Companion corrected. |
| S2-M13, D09-INH-03 | applied | (B) of `inh:inherited-parent` now uses output intertwining for restricted P_I = {nu Lambda}, declared with Lambda. An explanatory paragraph was added. |
| S2-M14, D09-INH-04(e,f,g) | applied | Omega_I is Borel and R_I is Borel onto a standard Borel space. New clause (L) for laws. (D) is stated for flows with a well-posed parent flow. "Declared in this order" removed and replaced by `inh:protocol-classes`. |
| S2-M15, D09-INH-05 | applied | `inh:canonical-lumping` now refines by joint response/next-block kernels (bisimulation with outputs), with the proof updated. The canonical-parent claim is restricted to deterministic updates and to finite chains with all-law preparations. The (F4) misstatement was fixed. The flow version is stated in C([0,inf),Y). |
| S4-L23, XC6-PROG-20 | applied | `inh:canonical-parent`: the image is analytic; it is a retention when the image is Borel. |
| S1-L14, D03-GEOM-04 | applied | (G) now names the condition of `geo:retained-connection` and its hypotheses, and cites `geo:minimal-contraction`. |
| S2-E17, D09-INH-14 | applied | `inh:composition` now composes joint kernels, intertwinings and canonical parents, with proofs. |
| S2-E18, D09-INH-15 | applied | New lemma `inh:borel-factorization`. `inh:deterministic-descent` is derived from it. |
| S2-L04, D09-INH-08 | applied | The center-of-mass remark corrected (exception W = -s^2/4; s' = -s - 2W'). Companion corrected. |
| S2-L05, D09-INH-09 | applied | Intertwining is stated as sufficient only, with the strongly lumpable counterexample (verified by hand). (D) now says "for example". |
| S2-L06, D09-INH-10 | applied | "Classify" changed to "two sufficient mechanisms". The text after `inh:coordinate-projection` now says "can come", and projectability is named as the general criterion. |
| S2-L07, D09-INH-11 | applied | Type-closure test: sufficient conformality test with a ≡ 1, plus the exact closedness test. |
| S2-E03, D09-INH-07, GAP03-S2-04 | applied | `inh:type-closure-open`: the Wishart case is answered by `form:parent`, and the conformality failure is recorded with its factor. Lens disagreement (XC6-PROG-06 says not shown) was decided on evidence: GAP03-S2-04's Bregman/natural-gradient proof was reconstructed (Lemma A checked by hand) and the factor 0.480 recomputed. |
| S2-L08, D09-INH-13 | applied | Companion: the random-dynamics smallest parent is a bisimulation. |
| S2-L09, D10-FORM-10 | applied | Deterministic filter sentence corrected, with the doubling-map example (main text and s04). |
| S2-L11, D10-FORM-12 | applied | Barnett–Seth: "implies, converse fails", with the counterexample. The unverified Sec. 2.3 locator was dropped. |
| S2-L12, D10-FORM-14 | applied | Joint-law paragraph: nondegenerate ensemble; global posterior and interacting-memory posterior named. |
| S1-L57, GAP08-S1-02 | applied | Markov hypothesis added (ch05 memory section and companion), with the X_{n+1}=X_{n-1} counterexample. |
| S1-M33, GAP08-S1-01 | applied (remark) | New `sinh:filter-general` in s04 covers the countable alphabet (exact, with measures), the uncountable case (versions; Kallenberg2021) and the doubling map. The ch05 sentence was updated. The full general theorem was not added (extension). |
| GAP08-S1-07 (s04 part) | applied | Information-state attribution `\citep{Astrom1965}` (new key) in s04 and ch05. Higham/Wilcox parts are B-FOUND's (s02). |
| XC4-CONS-09(v) | applied | ch05 restates T and P_z instead of pointing to the supplement. |
| XC4-CONS-12 (ch05, s04 parts) | applied | All `\eqref` were converted to `\cref`/`\Cref`. The hand-made proofs (9 in ch05, 2 in s04) now use the proof environment. The two `\,` in my new text were removed. |
| D10-FORM-11 | applied | "No formed parent in ch 6–14 acts as a source; ch 19 declares top-down factors." `it3:open-formation` is B-LINKS's. |
| XC4-CONS-02, XC8-REMED-03, D16-OPEN-03, XC6-PROG-18 (ch05, c05 parts) | applied | The chapter 12 null reproduction is now stated as "integrations of the limit field indicate". |
| D16-OPEN-01 (ch05 part), D16-OPEN-08 (ch05 part) | applied | Chapter 12 summary: "neither (F5) nor (F6b) established; not shown to be a recursive formation". The null reading is now fixed in `inh:null-flow`. |
| D14-BND-15 (ch05 part) | applied | Chapter 10 added to the summary as inheritance without selection. |
| D11-FUS-01(e) | applied | `inh:null-flow`: nonsmooth objectives declare their controls by the same rules. |
| D05-OBJ-06, D17-PA-03 (ch05 parts) | applied | Nulls defined for the Gibbs-reduced per-agent flow. The flow itself must be defined in ch03 (B-DYN). |
| XC1-GAUGE-01, D17-PA-01 (ch05:1397 part) | applied | `inh:formation-open`: the "physically noninjective parent" wording was removed. The parent now quotients a phase that is gauge relative to every declared algebra, so it fails (F4) as declared. |
| D17-PA-07 (ch05 part) | applied | `inh:formation-open` states one stage, so (F6b) does not follow; the grades are left to ch13. |
| XC2-FRAME-03 (c05 part) | applied | Companion: "No construction in this manuscript is proved to meet every clause." |
| S4-M13, XC6-PROG-10 | applied (weakness stated) | A paragraph after the standard: exactness plus (F4) visibility; no mechanism for interaction-created exact symmetry; graded inheritance not developed. |
| H16, XC6-PROG-06 | deferred (decision 6) | The weakness is stated after the standard and the question added to `inh:formation-open`. No clause added. |
| H17 | deferred (decision 7) | Nulls are defined for all three objectives; no primary objective chosen. |
| S4-M24, XC6-PROG-15 | partly applied | R14's class was imported into (F6b), stated locally. R13 and R15 were not imported (author decision). Persistence type is now required in (F2). |
| S2-E04, D10-FORM-16 | deferred | Extension (Pexider centroid-closure proposition). |
| D10-FORM-17 (b), (d) | deferred | Extension (Gaussian moment-projection counterexample; interacting-memory finest partition). Part (a) is applied (H10). Part (c) is covered in the common-clock remark. |
| D09-INH-02 extension prose (character-twisted descent proposition) | deferred | Extension. |

Rejected: none in full. Partial non-adoptions with reasons:
- XC2-FRAME-05: its (F6b) class was not adopted (D3 mandates R14). Its strong-form replacement was adopted.
- XC2-FRAME-01: its non-decay clause is reported as an attribute only, because a clause would decide beyond D2.
- H08/XC2-FRAME-02: the numerical claim "fails for (1/3, 1)" was not asserted in ch05. The ch13 owner reports it.

## BIB ENTRIES

```bibtex
@article{Astrom1965,
  author  = {{\AA}str{\"o}m, Karl Johan},
  title   = {Optimal control of {M}arkov processes with incomplete state information},
  journal = {Journal of Mathematical Analysis and Applications},
  volume  = {10},
  number  = {1},
  pages   = {174--205},
  year    = {1965},
  doi     = {10.1016/0022-247X(65)90154-X}
}
```
Verified against the Crossref record (DOI above). Cited in `chapters/05_inheritance.tex` (memory section) and `supplements/s04_inheritance.tex` (`sinh:filter`).

## NEW OR CHANGED LABELS

New labels, all unique across the three documents (checked):

| Label | File | What it is |
| :- | :- | :- |
| `inh:borel-factorization` | ch05 | lemma |
| `inh:protocol-classes` | ch05 | definition |
| `inh:null-self-row` | ch05 | equation |
| `inh:no-interaction-null` | ch05 | lemma |
| `inh:counterfactual-class` | ch05 | definition |
| `inh:common-tilt` | ch05 | corollary |
| `inh:common-tilt-identity` | ch05 | equation |
| `inh:anchored-consensus` | ch05 | proposition |
| `inh:anchored-potential` | ch05 | equation |
| `sinh:filter-general` | s04 | remark |

Changed content under unchanged labels:
- Retitled: `inh:no-underdamped`, `inh:canonical-lumping`.
- Rewritten: `inh:null-flow` (now "Null flows and the frozen-environment control"), `inh:recursive-formation`, `inh:inherited-parent`, `inh:common-clock` (cases a/b), `inh:formation-open`.

No label was removed or renamed, so there are no dangling references. All `\cref` targets in the three files resolve.

## FOR OTHER OWNERS

- **B-FORM** (ch10, ch11, ch12, c10–c12):
  - Ch 11: replace `form:kappa-independence` (iii) per D1, citing (N1)/(N2) and `inh:null-self-row`.
  - Ch 11 (F5) verdict = split, hence fails. Report the coupling-deformation control separately.
  - Ch 11: cite `inh:common-clock` case (b) from `form:mean-field`, and `inh:scalar-additive` from `form:scalar-closure`.
  - Ch 11:208: remove "needs targets that depend on the receiver" (S2-M01).
  - Ch 11:393: state the counterfactual class per `inh:counterfactual-class` and the B_I rule.
  - Ch 11: the post-handoff null is the fine null, evaluated on shape sums (GAP03-S2-02). Optionally use GAP03-S2-04 Lemmas A/B.
  - Ch 10:4: "(F1), (F3), (F4) for the supplied hierarchy; (F2), (F5), (F6) not addressed". Also cite `inh:equivariant-descent` (D14-BND-11).
  - Ch 12: name both nulls; the readout convention is now standard. Compare P modulo disjoint-merge order. Write "not shown to be a recursive formation" at 12:4 and 12:526.
- **B-FUS** (ch06–09):
  - Ch 06: declare the null for the nonsmooth potential by the rules of `inh:null-flow`. State (F5) over the selector family and constant set (decision 9 still open). (F4) fails.
  - Ch 08:263: under the current (F5), equal partition sequences mean (F5) fails. A partition-valued selector is needed (D13-ERAPUB-01).
  - Ch 09:251: a probe class "computed from the phase law" is excluded by `inh:protocol-classes`.
- **B-PA** (ch13, ch14, c13, c14):
  - `pa:null-row` / `cp:null-row` are now (N1) in ch05 (`inh:null-self-row`). The verdict must also be tested against (N2) and reported with Phi^fr.
  - Selector family and constant set per (F2)/(F5); report constants (H08) and the selector-free observable (agent 3's odds reverse sign).
  - (F4) fails under B_I (phase gauge relative to declared algebras, D2). Remove "physically noninjective" and "nongauge" (ch13:302, 14:41).
  - Ch13: (F6a) fails; strong grade not met.
- **B-DYN** (ch03): define the Gibbs-reduced per-agent flow on finite tables and in charts (D17-PA-03, D05-OBJ-06). Ch03:221 wording (XC4-CONS-09 i). S1-L28 (s03) is B-DYN's.
- **B-IT7**: `it7:null-row` = (N1). State (F4) against the declared B_I rule, (F6b) against `inh:counterfactual-class`, and the (B) protocol class (D27-ITEM7-02).
- **B-LINKS**: make `it3:open-formation` a specialization of `inh:formation-open` (D10-FORM-11).
- **B-FOUND**: optional — a label on the chapter 01 "gauge relative to an algebra" sentence (01:87), so ch05 can cite it (it currently cites `ch:foundations`). s02 Higham/Wilcox attributions (GAP08-S1-07).
- **C-SUM**:
  - Merge `Astrom1965`.
  - AUTHORING.md:45 formation-standard paragraph → align with the new `inh:recursive-formation` (two nulls, frozen control, protocol/probe rules, R14 class, grades, "shown to fail").
  - README:30/36, glossary:33–36, notation:55 → quote the new `inh:null-flow` (S4-L11, XC4-CONS-06).
  - Outlook:16 "requires receiver-dependent targets" → the ch05 wording (XC4-CONS-03).
  - frontmatter:9/11 → null-reproduction qualifier and the ch13 requalification.

## AUTHOR DECISIONS NEEDED

1. A split (F5) verdict is counted as failing (reading of D1's "tested against both"). Confirm, or say whether a pass against the renormalized null alone suffices.
2. (F5) holds only when it holds for every constant of the declared constant set, and real constants must range over a set with nonempty interior (reading of D3). Confirm.
3. The strong grade was redefined as "(F5) at every point, every constant, both nulls". Under it the anchored consensus passes. Confirm, or drop the grade.
4. The frozen-environment control is a required report, not a clause. Decide whether P ≠ P^fr should become a clause or a grade.
5. Decision 6 (type closure, top-down efficacy): still open; the weakness is stated in the text.
6. R13 and R15 (Curie rule; coherence horizon) were not imported. Decide whether they become grades of (F2) and (F5).

## INTERFACE FOR WAVE 2

All of the following are in `chapters/05_inheritance.tex`.

**Controls** (`inh:null-flow`, "Null flows and the frozen-environment control"):

| Name in text | Symbols | Definition |
| :- | :- | :- |
| (N1) label-preserving null | Phi^0_lp, H^0_lp, P^0_lp | Peer costs set to zero; labels and priors kept; Gibbs self weight `\eqref{inh:null-self-row}` = a e^{-D_ii/T}/(a e^{-D_ii/T}+1-a) |
| (N2) renormalized null | Phi^0_rn, P^0_rn | Deleted peer labels removed; self weight one |
| (N3) frozen-environment control | Phi^fr, P^fr | Nothing deleted; other agents' laws frozen at t=0 in every cost, row, channel and evaluator |

- A "peer cost" is one whose compared law depends on another agent's law. Constant-output costs are kept.
- The two nulls coincide when every self cost vanishes identically or self is excluded.
- Readout convention: the selector reads rows and costs through the original formulas at the control state.
- After a handoff the control stays the fine flow.
- Nulls exist for the reciprocal potential, the per-agent dynamics (dynamic and Gibbs-reduced) and the global free energy. Nonsmooth objectives declare their controls by the same rules.
- Each control must be well posed on the admitted class up to the horizon.
- `inh:no-interaction-null`: non-interacting agents fail (F5).

**Classes:**
- Probe and protocol rules: `inh:protocol-classes` ("Declared protocol and probe classes"), with rule (A) giving A_I (parent protocols with K_a, D_a; no member named) and rule (P) giving B_I (child-level probes available before the handoff, containing A_J of merged blocks). Both are functions of the block and the constitutive data only.
- Counterfactual class: `inh:counterfactual-class`, the class 𝒞 = actual handoffs + same-time handoffs from a declared U ⊇ admitted class without the first-stage event, minus states outside the selector domain.

**Repaired clauses** (`inh:recursive-formation`):

| Clause | Requirement |
| :- | :- |
| (F1) | Full grade (open in the chart of the declared dynamics) or family grade (relatively open in a declared invariant family, not finitely many orbits, codimension reported) |
| (F2) | Selector family S_c, c in C, fixed in advance, constants over sets with nonempty interior; readout convention; persistence type stated |
| (F3) | `inh:inherited-parent` with clauses (L), (D), (B), (C), (G) on Borel Omega_I ⊇ Omega_I^reach |
| (F4) | Some b in B_I separates x ≠ x' in Omega_I^reach with R_I(x)=R_I(x'); attributes reported: persistent or transient, symmetry type or not |
| (F5) | Pointwise on an open U; on [0,∞) P ≠ P^0, on a finite horizon P_[0,τ] not an initial segment of P^0. P is taken modulo exchange of disjoint merges. Holds only against both (N1) and (N2) and for every c in C; a split verdict fails |
| (F6a) | Same selector family, constants and data; no reset |
| (F6b) | Non-occurrence on a set with nonempty interior in 𝒞 (deterministic), or a probability margin (stochastic); otherwise a supplied-scale sequence |

**Grades:**
- Strong: (F5) at every admitted point, every c, both nulls.
- Amplifying: declared metric d; at Φ's event times, on Φ's partition, ϱ(Φ) ≤ λϱ(Φ^0) for both nulls; "degenerate" if within-block diameters vanish.

**Classification:**

| Term | Condition |
| :- | :- |
| inherited recursive formation | all of (F1)–(F6) |
| two-stage selection with exact inheritance | (F1), (F2), (F3), (F6a) hold and (F4), (F5) or (F6b) is *shown* to fail |
| single-stage selection with exact inheritance | (F1)–(F3), one stage |
| undetermined ("not shown to be a recursive formation") | a clause neither established nor refuted; name it |

**Supporting results:**
- `inh:common-clock`, cases (a) and (b): in (a) the nulls coincide at rate −1; in (b) they differ, with no common clock.
- `inh:common-tilt`: tilt-invariant selectors fail (F5) in case (a).
- Calibration example: `inh:anchored-consensus`.
- Open problem: `inh:formation-open`.

**Recipe for a chapter's verdict.** State the following in prose (no bullets), in this order:

1. The objective (reciprocal, per-agent or global), the dynamics and its chart.
2. The (F1) grade.
3. The selector family, the constant set C and the persistence type.
4. The rules A_I and B_I.
5. The (F4) witness, with persistent/transient and symmetry attributes.
6. (F5) against (N1) and against (N2), each over C. Name the subset where it holds, give the selector-free observable, and compare P with P^fr.
7. (F6a).
8. The declared 𝒞 and the (F6b) verdict.
9. The grades.
10. The classification term from the table above, naming each failing or undetermined clause. Never write "physical" or "nongauge" for a distinction that no element of B_I or of the declared algebra separates.
