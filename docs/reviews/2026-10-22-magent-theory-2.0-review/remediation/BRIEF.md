# Remediation brief: MAgent Theory 2.0, 2026-10-22

You are one of eight chapter-scoped remediation agents. The author has asked for the fixes and remediations from the 2026-10-22 review to begin. Treat the review findings in your scope as requirements, implement them in the manuscript, and log what you did.

## Inputs

Manuscript root M = `C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/` (git branch `magent-2.0-remediation`; do not commit, switch branches, stash, or touch files outside your ownership list).

Read first:

1. `M/AUTHORING.md`. It was rewritten today and records the author's decisions. It is the standard for every edit.
2. The decisions already implemented by the lead (do not undo them; keep your text consistent with them):
   - Agents hold beliefs and generative models: `M/chapters/01_foundations.tex`, section `sec:generative-models` (`def:generative-model`, `eq:agent-free-energy`, `def:shared-model`), and a dressing-field paragraph after `eq:reference-laws`.
   - Three objectives in `M/chapters/03_dynamics.tex`: the reciprocal law potential `def:dyn-potential` (renamed "Reciprocal law potential"), the global picture `prop:dyn-global-local`, the per-agent picture `def:dyn-per-agent`, `prop:dyn-per-agent`, `def:dyn-per-agent-flow`, the no-go `prop:dyn-no-global-vfe`, `def:dyn-record-interrupted`, and `op:dyn-three-pictures`.
   - The formation standard in `M/chapters/05_inheritance.tex`: `inh:null-flow`, `inh:recursive-formation` with clauses (F1) to (F6), the term "two-stage selection with exact inheritance", and `inh:formation-open`.
   - New abstract `M/frontmatter.tex` and outlook `M/chapters/06_outlook.tex`.
3. The review synthesis `C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-22-magent-theory-2.0-review/REPORT.md` (sections 3, 5, 6, 7 for your scope) and the lens reports in `.../lenses/` named in your assignment. Every finding whose Location is in your files is in scope, whatever lens produced it; grep the lens directory for your file names to find them all (X6-LIT citation findings and X1, X2, X4 findings included).

## Rules

1. The findings are investigator findings, not adjudicated. Before applying a fix, check the claim against the text and, for mathematical claims, reconstruct it (small sympy or numpy checks are fine). If a finding is wrong, do not apply it; log it as rejected with the reason. If two lenses conflict (REPORT.md section 7), decide on evidence and log the reasoning.
2. Apply every valid finding in scope at every severity: errors, gaps (state the missing hypothesis inside the result or prove the missing step), inflations (reword headings, statements and prose to what is proved), weaknesses (state them honestly in the text, and add the proposition the lens proved where it is short and checkable, for example a null-control or symmetry-reduction statement), consistency, citations (attribute named classical mathematics), exposition, and low-cost improvements. For an extension that would need a new research result, add at most a precise remark or open problem and log it as deferred. Do not delete results; rescope and retitle them.
3. Every construction in chapters 6 to 12 must say that it concerns the reciprocal law potential (or its nonsmooth modification), report its null-flow status against (F5) where it selects, and use "formation" only as the standard allows.
4. Edit only the files you own. Do not edit `references.bib`: put every new BibTeX entry, with verified bibliographic data, in your log under a heading `BIB ENTRIES`, and use those keys in your text. Check `M/references.bib` first and reuse existing keys. Use AuthorYear keys (for example `RogersPitman1981`).
5. Do not compile the documents (the lead compiles once at the end, because concurrent builds collide). Write careful LaTeX: balanced environments, existing macros from `M/preamble.tex` (`\definitionheading`, `\propositionheading`, `\theoremheading`, `\remarkheading`, `\openproblemheading`, etc.), `\cref` for references, labels that are unique across the three documents (grep before adding).
6. Style: AUTHORING.md rules, American English, punctuated displays, no `\;`, `\,`, `\!`, no horizontal rules, no bullet lists in mathematical prose, no repository paths, review IDs, tool names or workflow history in the scientific text, no citations of unpublished work by the author. Avoid: "key insight", "crucially", "critically", "notably", "importantly", "fundamentally", "essentially", "leverage", "underscore", "delve", "robust" (except technically), "comprehensive", "seamless", "pivotal", "paramount", "nuanced", "landscape", "realm".
7. Compute: scripts single-threaded and short, run as `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python <script>`, one at a time, never in the background, no pools, no GPU. Put them in `C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-22-magent-theory-2.0-review/remediation/scripts/<your agent id>/`. No Lean. Do not spawn agents.

## Log

Write `C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-22-magent-theory-2.0-review/remediation/logs/<your agent id>.md`: a table with one row per finding ID in scope (applied, rejected, or deferred; what changed, with file and label; for rejected, the reason and evidence), then `BIB ENTRIES` (BibTeX), then `NEW LABELS`, then `AUTHOR DECISIONS NEEDED` (only genuine ones). Your final message: at most 25 lines with counts and the log path.
