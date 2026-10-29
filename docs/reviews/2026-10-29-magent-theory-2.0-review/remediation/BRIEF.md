# Remediation brief: MAgent Theory 2.0, 2026-10-29

You are one of ten file-scoped remediation agents. The author asked for the fixes of the 2026-10-29 review to begin. Treat the review findings in your scope as requirements, implement them in the manuscript, and log what you did.

## Inputs

Manuscript root M = `C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/`, on git branch `magent-2.0-remediation-2026-10-29`. Do not commit, switch branches, stash, reset, or touch any file outside your ownership list. The Research repository holds unrelated uncommitted work of the author; leave it alone.

Review folder V = `C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-29-magent-theory-2.0-review/`.

Read first:

1. `M/AUTHORING.md`, the standard for every edit (general-law statements first, Gaussian results as instances; declared charts, Lipschitz hypotheses and well-posedness for Fisher flows beyond finite tables; "gauge", "physical" and "public" only relative to a declared observable algebra; the three objectives never conflated).
2. `V/REPORT.md`: Sections 3 (ranked items H01 to H18, I01 to I04), 5 (rows filed under your files), 7 (roadmap R1 to R14), 8 (inter-lens disagreements) and 10 (author decisions). The report was checked against the lens reports; it governs the cluster files `V/synthesis/S1..S4`, and the lens reports `V/lenses/*.md` govern both. Read the lens finding itself (its Evidence, Fix and Falsifies fields) before editing. Grep `V/lenses/` for your file names: every finding located in your files is in scope, whatever lens produced it.
3. The author decisions below.

## Author decisions (2026-10-29)

D1. Null flow (H01, H04). `inh:null-flow` declares both nulls: the label-preserving null (peer edges deleted, labels, priors and all other data unchanged, so the self row keeps its Gibbs weight b_i < 1, as `pa:null-row`, `cp:null-row`, `it7:null-row` and 14:80 already say) and the renormalized null (self row of weight one). Every (F5) verdict states which null it uses, and is tested against both where they differ. In chapter 11, clause (iii) of `form:kappa-independence` is replaced by the computed label-preserving null law (agent-dependent clocks; membership history different on a nonempty open set already at kappa = 0), the renormalized-null statement is kept where it is true, and the coupling-deformation control of (i) and (ii) is stated as a separate control. The remaining undefined parts listed in H04 (domain beyond finite tables, what selectors read along the null, existence, the post-handoff null, and the null for the global free energy) are defined or explicitly scoped out in `inh:null-flow`.

D2. Observable algebra and (F4) (H05). The chapter 01 meaning of "gauge" stands. Chapters 13 and 14 and the abstract are requalified: the erased chapter 13 phase and the chapter 14 psi rotation are gauge relative to every declared algebra, so "physically noninjective" and "nongauge" are removed or replaced by what is proved. The (F4) child-level probe class is declared once in chapter 05, relative to the retention, with the declaration order given real content or dropped. Where a construction then fails (F4), the text says so.

D3. Formation standard (H06 to H10, I02). Adopt the I02 replacement: a declared counterfactual class of R14 type for (F6b), with a margin or interior condition; openness in the chart topology of the dynamics, or relative to a declared invariant family with its grade; (F5) pointwise on a nonempty open set with a declared horizon convention; the selector and its constants fixed before the flows are analyzed, with verdicts reported over a set of constants and a selector-free observable where one exists; the strong and amplifying grades defined or dropped. The anchored linear consensus (H10) becomes a calibration example in chapter 05, and the frozen-environment control for non-interacting agents (H09) is stated. Every construction of chapters 06 to 14 and Item 7 is then reclassified against the repaired clauses by the agent that owns it.

D4. Item 7 (H02, H12 to H14). Item 7 stays a working document with an exact status statement (proved, numerical, open, clause by clause), the convex no-go is retracted or restricted to what is shown, (F3) and (F4) are marked open where they are, and the abstract, chapter 01 and README describe it consistently.

Decisions 5 to 17 of REPORT Section 10 not covered above remain open. Do not decide them: apply the decision-independent part of a fix (for example, state a result at its proved scope) and list the rest under `AUTHOR DECISIONS NEEDED`. Extensions (C01, E01 to E07 and medium or low extension rows) are research and are not implemented; add at most a precise remark or open problem where the lens asks for a text change, and log them as deferred.

## Ownership and waves

Wave 1 (now): A-STD, B-FOUND, B-DYN, B-CTX, B-LINKS. Wave 2 (after A-STD finishes, so that the repaired standard is in place): B-FUS, B-FORM, B-PA, B-IT7. Wave 3: C-SUM.

- A-STD: `chapters/05_inheritance.tex`, `supplements/s04_inheritance.tex`, `companion/c05_inheritance.tex`. Decisions D1 to D3 in chapter 05; all findings in these files; H04 to H10 and I02 in their chapter 05 parts.
- B-FOUND: `chapters/01_foundations.tex` (except its opening summary sentence at 01:4 and any sentence that summarizes formation results, which C-SUM owns), `appendices/record_process.tex`, `chapters/02_interactions.tex`, `chapters/02_geometry.tex`, `appendices/base_connection.tex`, `appendices/determinant_example.tex`, `supplements/s01_probability.tex`, `supplements/s02_geometry.tex`, `companion/c01_objects.tex`, `companion/c02_links.tex`, `companion/c06_geometry.tex`, `companion/figures/*`.
- B-DYN: `chapters/03_dynamics.tex`, `supplements/s03_variation.tex`, `companion/c03_dynamics.tex`, `chapters/04_prediction.tex`, `appendices/conditional_memory.tex`, `appendices/interacting_memory.tex`, `companion/c04_communication.tex`, `companion/c04_interacting_memory.tex`. Includes H03 in chapter 03 (statements at proved scope), I01, I04 (state the Gaussian closure of the global and per-agent flows).
- B-CTX: `chapters/15_contextual_coupling.tex`, `chapters/16_link_base_transport.tex`, `companion/c15_contextual_coupling.tex`, `companion/c16_link_base_transport.tex`.
- B-LINKS: `chapters/17_communication_links.tex`, `chapters/18_link_dynamics.tex`, `chapters/19_curvature_hierarchy.tex`, `companion/c17_communication_links.tex`, `companion/c18_link_dynamics.tex`, `companion/c19_curvature_hierarchy.tex`, `Item_1_communication_links.tex`, `Item_2_link_dynamics.tex`, `Item_3_curvature_hierarchy.tex`, `item1_checks/`, `item2_checks/`, `item3_checks/`, `item1_refs.bib`, `item2_refs.bib`. Includes H15 and bringing Items 1 to 3 to the corrected chapter text with a status note.
- B-FUS: `chapters/06_fusion.tex`, `chapters/07_smooth_closure.tex`, `chapters/08_smooth_erasure.tex`, `chapters/09_public_compression.tex`, `companion/c07_fusion.tex`, `companion/c08_smooth.tex`, `companion/c09_public_compression.tex`. Includes H11.
- B-FORM: `chapters/10_boundary_closure.tex`, `chapters/11_conditional_formation.tex`, `chapters/12_open_formation.tex`, `companion/c10_boundary.tex`, `companion/c11_formation.tex`, `companion/c12_open_formation.tex`. Includes H01 in chapter 11 per D1.
- B-PA: `chapters/13_per_agent_inheritance.tex`, `chapters/14_collective_local_parent.tex`, `companion/c13_per_agent_inheritance.tex`, `companion/c14_collective_local_parent.tex`. Includes the D2 requalification and the H08 threshold dependence.
- B-IT7: `Item_7_formation_feedback.tex`, `item7_checks/`, `item7_refs.bib`. D4; H02, H12 to H14.
- C-SUM: `frontmatter.tex`, the opening summary sentence and formation summaries of `chapters/01_foundations.tex`, `chapters/06_outlook.tex`, `appendices/notation.tex`, `companion/glossary.tex`, `README.md`, `AUTHORING.md`, `future_TODO.md`, `references.bib`. Abstract-level statements (H03, H05, H11, H12, H17, S2-M08, S4-M02), the merge of all logged BibTeX entries, and summaries consistent with what the chapters now prove.

A finding located in a file you do not own: do not edit it; list it under `FOR OTHER OWNERS` in your log with the owner.

## Rules

1. The findings are investigator findings, not adjudicated. Before applying a fix, check the claim against the text and, for mathematical claims, reconstruct it. If a finding is wrong, do not apply it; log it as rejected with the reason and evidence. Where lenses disagree (REPORT Section 8), decide on evidence and log the reasoning.
2. Apply every valid finding in scope at every severity: errors (correct the statement or its proof), gaps (state the missing hypothesis in the result, or prove the missing step where the lens supplies a short checkable proof), inflations (reword headings, statements and prose to what is proved), weaknesses (state them in the text), consistency, citations (attribute named classical mathematics, verified against the primary source), exposition, and low-cost improvements, including the statement upgrades of R14. Do not delete results; rescope and retitle them.
3. Every construction of chapters 06 to 14 states which objective it concerns and reports its (F5) status against the declared nulls of D1 and the repaired clauses of D3.
4. Do not edit `references.bib` unless you are C-SUM. Put each new BibTeX entry, with bibliographic data verified against the publisher or a library record, in your log under `BIB ENTRIES`, and cite it by that key. Reuse existing keys (check `M/references.bib`). AuthorYear keys.
5. Do not compile any document (C-SUM and the lead compile at the end). Write careful LaTeX: balanced environments, existing macros of `M/preamble.tex` (`\theoremheading`, `\propositionheading`, `\definitionheading`, `\remarkheading`, `\openproblemheading` and the rest), `\cref`, labels unique across the three documents (grep before adding). When you rename or remove a label, grep all three documents and list every dangling reference under `FOR OTHER OWNERS`.
6. Style: AUTHORING.md rules and your global writing rules: American English, punctuated displays, no `\;`, `\,`, `\!`, no em dashes, no horizontal rules, no bullet lists in mathematical prose, no banned words. No repository paths, review IDs, tool names or workflow history in the scientific text; no citations of unpublished work by the author.
7. Compute: scripts single-threaded and short, run as `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 30 python <script>`, one at a time, in the foreground; no process pools, no background runs, no GPU or `$CUDA_PYTHON`. Scripts go in `V/remediation/scripts/<agent id>/`. No Lean. Do not spawn agents or workflows.

## Log

Write `V/remediation/logs/<agent id>.md`: a table with one row per finding or item ID in scope (applied, rejected or deferred; what changed, with file and label; for rejected, the reason and evidence), then `BIB ENTRIES` (BibTeX), `NEW OR CHANGED LABELS`, `FOR OTHER OWNERS`, and `AUTHOR DECISIONS NEEDED` (only genuine ones). Your final message: at most 25 lines with counts and the log path.
