# B-IT7 remediation log (2026-10-29)

Files edited (manuscript root `Research/manuscripts/MAgent_Theory_2.0/`, branch `magent-2.0-remediation-2026-10-29`, nothing committed, nothing compiled): `Item_7_formation_feedback.tex`, `item7_refs.bib`, `item7_checks/checks.py` (docstring only, no printed value changed). New check files in `item7_checks/`: `certify.py` + `certify_output.txt` (check (27), mpmath interval arithmetic), `controls.py` + `controls_output.txt` (check (28), the nulls N1/N2 and the frozen control N3, plus widened-state probe), `dwell.py` + `dwell_output.txt` (check (29), convex finite-dwell counterexample, rebuilt from the lens script with the joint-metric variant added). CRLF kept in the .tex and .bib. All scripts single-threaded, each under 11 s, run one at a time.

Verification scripts (`V/remediation/scripts/B-IT7/`): `controls.py` (first version, outputs `controls_declared_output.txt`, `controls_derived_*_output.txt`) and `r14_probe.py` (+ output). The lens scripts `d27_attractor_counterexample.py` and `gap07_tangent_verify.py` were rerun read-only and reproduced their printed outputs.

New results computed here (not in the lens reports):
- N2 (renormalized null), declared selector: on all of A it selects only the block of all twelve agents (fixed draw t = 5.55). Proved by hand in `it7:null-rn` (closed-form linear flow, distances within factor 1 ± 0.067 of c(t)|o_ij|, candidate windows at most 0.659 and 0.504 < tau_p), confirmed numerically on 13 states.
- N3 (frozen control), declared selector: four groups, no union (13 states); final left gap 0.870. So P ≠ P^fr.
- Derived selector, fixed draw: N2 selects the two sides of six and then all twelve agents at every window. N3 selects the four groups only. N1 with the A-STD post-handoff convention (control stays the fine flow) gives the same history as the stored shared-null-row run.
- Widened initial states (group spreads × 1.5, 2, 3) have no first stage by t = 2.3. The four groups handed off then have left gaps 0.436, 0.302, 0.203, all below d_u, so they unite. The standard-class (F6b) is therefore open and the natural candidate class points toward failure.

## Findings table

| ID | Status | What changed (label) / reason |
| :- | :- | :- |
| D4 / H12, D27-ITEM7-03, XC2-FRAME-03 (Item 7 part) | applied | Summary paragraph and old it7:279 paragraph replaced by a clause-by-clause status (proved / numerical / open) against the repaired standard and both nulls. Both tables rewritten (`it7:ch-standard`, `it7:derived-standard`). Classification: "undetermined; not shown to be a recursive formation". The certified-vs-numerical boundary is stated: only the first-stage integration is uncertified (GAP07-S3-03). |
| H02, D27-ITEM7-04, GAP07-S3-01 | applied | `it7:attractor` retitled and restricted to all-time persistence. It now carries the no-competing-selection hypothesis and the boundary case. The paragraph and summary claim "no declared Gaussian hierarchy can / needs nonconvex objective" were deleted. New `it7:dwell` (numerical, check (29)) states the counterexample and that it disappears in the joint's own metric. It also states that the joint search found (F6b) instances only with anchors pulling the parents together. New open problem `it7:open-convex`. |
| H13, D27-ITEM7-01 | applied | `it7:exact` retitled "at a fixed partition" and restated for unit-invariant sets between handoffs. New remark `it7:hybrid` states what is not proved (clocks, n_U, obar_U, pi_U, merges, event descent). (F3) is marked open in both tables. |
| H14, D27-ITEM7-02 | applied | New `it7:classes` declares rule (A) (read-only queries of A_U, adaptive schedules allowed, no assimilation) and rule (P) (B_U). (B) is proved at a fixed partition for that class, including finite adaptive schedules (short proof added to `it7:exact`). The counterexample for member-specific assimilation is stated. |
| I03, D27-ITEM7-19 | applied as route only | Remark `it7:mass-parent` (with `it7:blockwise`) states the route and says it is not a result. Open problem `it7:open-hybrid`. |
| S3-M24, D27-ITEM7-05 | applied | (F4) second stage: open, depends on whether the parent type records formation times. The first stage is numerical, with a witness that is not among the checks. Attributes: transient, not a symmetry quotient (`it7:forgets`). |
| S3-M25, D27-ITEM7-06 | applied | Scope changed to "Gaussian families of the sufficient statistic" with the lognormal example. Summary, momentum paragraph, theorem title (`it7:characterization` retitled), table and open problem were edited. |
| S3-M26, D27-ITEM7-07 | applied | `it7:type` restated "at a fixed partition". It says the class-level merge is undeclared and names the n_U-weighted rule. Composition across a union is not proved. |
| S3-M27, D27-ITEM7-08 | applied | After `it7:derived-null` and in the summary: the criterion detects attention, not agreement. Only blocks have a radius. The selector is derived relative to a declared family, prior and window. |
| S3-M28, GAP07-S3-04 | applied | Saddle-node s_c ∈ (1.0722, 1.0723) certified in check (27). Stated in `it7:reduced`, `it7:depth`, the summary and the standard chapter. The upper edge 1.175 (lens run) is not stated. |
| S3-L64, D27-ITEM7-09 | applied | Switch order corrected ({L1,L2,R1∪R2} from 1.75, then both unions from 1.77). The mixed regime is noted. Stagewise necessity is stated as not a clause, per side. |
| S3-L65, D27-ITEM7-10 | applied | "n_U times the free energy at T/n_U" (summary and after `it7:structure-fe`). |
| S3-L66, D27-ITEM7-11, GAP07-S3-09 | applied | `it7:counterfactual` now uses the certified bands (check (27)) and a band of 2e-8 around d_u, with trapping instead of convergence. `it7:derived-counterfactual` was changed in the same way. `it7:reduced` is extended to s ∈ [1.0, 1.3]. The depth claims use trapping and certified roots. |
| D27-ITEM7-12 | applied | Margins corrected to coherence ≥ 0.0045 and wall ≥ 0.43, matching stored check (17), and the grid time range to 2.13–2.35. |
| S3-L68, D27-ITEM7-13, S4-M05/XC4-CONS-05 (partly) | applied (partly) | Every "chapter N" reference replaced by chapter titles or names. Clauses, nulls and classes restated locally (`it7:standard-restated`). The repository path, the tool name and the date of the literature search, and the "seventh item of the future-work list" / "left by Item 3" history were removed. Remaining: references to Items 1–3 and the "check (n)" pointers (see AUTHOR DECISIONS). |
| S3-L69, D27-ITEM7-14 | applied | New `it7:conventions`: well-posedness, no general-law chart claimed, interior mean domain, simultaneous candidates, tie-breaking, evaluation grid. The post-handoff null convention now follows A-STD (the control stays the fine flow), and the derived-chapter null wording was changed to match. |
| D27-ITEM7-15 | applied | Check (7) is described as fixed-partition. The interval minimizer is described and justified per GAP07-S3-06. |
| D27-ITEM7-16 | applied | Null controls added for check (15) (equilibrium of both nulls, KL 0.27; verified from checks.py parameters) and for the record-gap variants (N1 proof constants). |
| S3-L72, D27-ITEM7-17; XC1-GAUGE-08(c) (B-LINKS/B-FOUND handoff) | applied | `it7:dictionaries` now uses the signed-walk holonomy convention. The g_i are said not to be passive frame changes, the identification with the directed flat-wall proposition was dropped, and the converse now reads "not in general", with the stabilizer example. |
| S3-L73, D27-ITEM7-18 | applied | `it7:tempered` hypothesis added (pool parent or members in E). The vector-field comparison is stated, and the summary was changed to match. |
| S3-L74, GAP07-S3-05 | applied | "≥ 1.82 among interval coarsenings". All-coarsening margin 1.715 at the fixed draw (stored check (21)). The certified box bound 1.6681 was not imported (lens computation). |
| S3-L75, GAP07-S3-08 | applied | "Faster than n^{-1/2}, local exponents 0.65–0.77" (from the certified roots of check (27)). Changed in `it7:depth`, the summary and `it7:open-depth`. |
| GAP07-S3-02 (S3-X32) | partly applied | The exact side reduction of check (21) was added as proved text. The null history on all of A under the derived selector was not asserted: it needs the lens's box certificate, which is not reproduced in the checks. The text says the first-stage null selection rests on sampled states and the absence of later events is proved. |
| GAP07-S3-03 (S3-X31) | applied | Only the first stage rests on integration. `it7:open-validated` was restricted to [0, 2.6] and [0, 1.30]. |
| GAP07-S3-06 | applied | "Since K=1" replaced by the stated justification (agreement checks plus the four-unit spread bound). |
| GAP07-S3-07 | applied | Times are stated as grid times, the tau_p window is converted (0.96 ≤ tau_p < 1.75), and "three handoffs" became "two states". Continuous-time ranges from the lens simulator were not imported. |
| GAP07-S3-10 (S3-X33) | deferred (extension) | Posed as open problem `it7:open-convex`. The mirror-symmetric proof was not added. |
| GAP07-S3-11 (S3-X51) | deferred (extension) | Interval lemma not added. The text notes that interval coarsenings need not contain the minimizer. |
| D27-ITEM7-20 | applied as open problem | `it7:open-validated` restated. |
| D27-ITEM7-21 (S3-X49) | deferred (extension) | Stated as an open constraint inside `it7:open-hybrid`. |
| S4-L28, XC5-LIT-08 | applied | O'Clery priority wording corrected. Martini–Egerstedt–Bicchi 2010 and Egerstedt et al. 2012 were added to `item7_refs.bib` and cited in the summary, after `it7:exact` and in `it7:ch-prior`. The tool-run sentence was replaced. The agglomerative information bottleneck citation was deferred (venue not verified). |
| S3-L47, D24-LDYN-03 (B-LINKS handoff) | applied | `it7:open-reciprocal` now uses the frozen-scalar wording. |
| A-STD for B-IT7 | applied | it7:null-row = (N1). (N2) and (N3) were added (`it7:controls`) and computed (`it7:null-rn`, check (28)). (F4) is stated against B_U, (F6b) against the R14 class (open), and the (B) protocol class is declared. |
| S4-L17 (Item 7 aux files in source root) | not mine | Build artifacts (`Item_7_formation_feedback.aux/.log/...`) are outside my ownership list; for the lead. |
| D27-ITEM7 "KulisJordan2012 lacks pages" | deferred | Pages not verified. |

Rejected: none. Partial non-adoptions:
- D27-ITEM7-12 suggested "wall margin at least 0.44". Stored check (17) prints 0.4349, so the text says 0.43.
- XC2-FRAME-03 suggested a README line "Item 7 reports an instance that meets (F1) to (F6) as stated". That line is not adopted, because under the repaired standard the instance is not shown to meet them. C-SUM should use the status in this log.

## BIB ENTRIES

Added directly to `item7_refs.bib` (B-IT7's file, not references.bib). Verified against Crossref: DOI 10.1504/IJSCC.2010.031160 (authors, title, journal, volume 2, issue 1/2/3, first page 100, year 2010) and DOI 10.1109/MCS.2012.2195411 (title, journal, 32(4), 66–73, 2012). Crossref has no author list for the 2012 entry. The authors are taken from the O'Clery et al. reference list, as read by XC5-LIT. The last page 121 of the 2010 entry also comes from that list.

```bibtex
@article{MartiniEgerstedtBicchi2010,
  author  = {Martini, Simone and Egerstedt, Magnus and Bicchi, Antonio},
  title   = {Controllability analysis of multi-agent systems using relaxed equitable partitions},
  journal = {International Journal of Systems, Control and Communications},
  volume  = {2}, number = {1/2/3}, pages = {100--121}, year = {2010},
  doi     = {10.1504/IJSCC.2010.031160}
}
@article{EgerstedtEtAl2012,
  author  = {Egerstedt, Magnus and Martini, Simone and Cao, Ming and Camlibel, Kanat and Bicchi, Antonio},
  title   = {Interacting with networks: How does structure relate to controllability in single-leader, consensus networks?},
  journal = {IEEE Control Systems Magazine},
  volume  = {32}, number = {4}, pages = {66--73}, year = {2012},
  doi     = {10.1109/MCS.2012.2195411}
}
```

## NEW OR CHANGED LABELS

New (all `it7:`, unique; no `it7:` label is referenced outside Item 7): `it7:conventions`, `it7:dwell`, `it7:dwell-energy`, `it7:hybrid`, `it7:mass-parent`, `it7:blockwise`, `it7:null-rn`, `it7:standard-restated`, `it7:classes`, `it7:open-hybrid`, `it7:open-counterfactual`, `it7:open-convex`.

Retitled under the same label: `it7:attractor`, `it7:exact`, `it7:type`, `it7:characterization`, `it7:counterfactual` ("Shifted post-handoff states"), `it7:derived-counterfactual`. None removed. All `\cref` targets resolve, and environments and braces balance (lint script).

## FOR OTHER OWNERS

- **C-SUM**:
  - README (Item 7 is listed only as "unchanged"), the abstract and 01:4: Item 7's status is "not shown to be a recursive formation; (F2) constant set, hybrid (F3), second-stage (F4), (F5) over constants, and standard-class (F6b) open; nulls proved; first stage numerical". This is consistent with "No construction ... is proved to satisfy every clause".
  - `future_TODO.md:94/96`: the convex no-go is retracted and restricted to all-time persistence. Change "Item 2's certainty loss" to "frozen-scalar source widening" (D24-LDYN-03). The open obligations are now `it7:open-hybrid`, `it7:open-counterfactual`, `it7:open-convex` and `it7:open-validated`.
  - Merge the two BibTeX entries only if the main documents cite them; they currently live in `item7_refs.bib`.
- **Lead**: Item 7 build artifacts in the source root (S4-L17). Item 7 also compiles against `references.bib`, so a compile check is needed after C-SUM's bib merge.

## AUTHOR DECISIONS NEEDED

1. Working-document citation rule (XC4-CONS-05): Item 7 still cites Items 1–3 and the main manuscript by title, and keeps the "check (n)" pointers to its accompanying checks. Either record an AUTHORING exception for working documents, or restate the Items 2/3 joint locally and move the check pointers to a build note.
2. Under (F6a), do declared parent anchors that pull parents together count as new constitutive data? This decides whether the joint instances of `it7:dwell` meet (F6). GAP07-S3-01 raised it.
3. Several verdicts depend on A-STD's provisional readings (A-STD decisions 1 and 2): split verdicts fail, and (F5) must hold over a constant set with nonempty interior. Under the second reading Item 7's single declared constants leave (F2)/(F5) open. If single declared constants are accepted, (F5) at the declared constants holds against both nulls (nulls proved; full flow numerical).
4. Declare the counterfactual class U for Item 7 (`it7:open-counterfactual`). The widened initial states examined give uniting handoffs.
