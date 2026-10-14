# Remediation brief: MultiAgentELBO theory, October 12, 2026

The author has authorized applying the October 12 deep review to the canonical theory in the
Research vault. You are one of seven editor agents working in parallel on disjoint files.

Inputs (read first):

- The review: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\REPORT.md`
  (synthesis) and the lens reports in `...\lenses\` named in your assignment. Each finding there
  has a Location, Evidence, and a proposed Fix. Treat the Fix as the requirement unless you find,
  on reading the text, that it is wrong; then do not apply it and say why in your log.
- The manuscript folder: `C:\Users\chris and christine\Desktop\Research\manuscripts\MultiAgentELBO_Theory\`.
  Read its `SPEC.md` sections 1 to 4 before editing. Follow its prose rules (R1), statement and
  status rules (R2, R4), gap rule (R8), provenance rule (R10: no dates, paths, review stages, or
  "fixed per review" comments in the .tex), citation rule (R12), notation table (section 3), and
  label rules (keep every existing label; new labels use the prefix of their heading type).
  American English. None of: "key insight", "crucially", "critically", "notably", "importantly",
  "it's worth noting", "fundamentally", "leverages", "underscores", "robust" (non-technical),
  "delve", "showcase", em dashes as default punctuation, horizontal rules, LaTeX spacing macros
  (`\;`, `\,`, `\!`). Refer to labels only with `\cref`/`\Cref`.

Author conventions (decided October 12; they govern the model):

1. Mutual attention uses distinct copies. When row i's source set contains a copy j->i and row j's
   source set contains a copy i->j, these are two distinct edge copies, each with its own reversal
   (`Theta_{\bar e} = Theta_e^{-1}` is unchanged), so the links in the two directions are
   independent and `Theta_ij != Theta_ji^{-1}` in general. `Theta_ij` denotes the link of the source
   copy j->i of row i. Every mutually attending pair forms a closed walk i->j->i with holonomy
   `Theta_ij Theta_ji`.
2. At the agent level (depth 0) a self edge, when a model includes it, is a loop copy carrying the
   identity link, `Theta_ii = e_G`, so `D_ii = 0` follows rather than being assumed. At coarse depth
   loop copies carry internal holonomy (`def:full-graph-coarse-links`) and `D != 0` in general.
3. The declared loop family of the generator prior contains every 2-cycle i->j->i formed by a
   mutually attending pair, once based at each endpoint (the walk based at i, product
   `Theta_ij Theta_ji`, and the walk based at j, product `Theta_ji Theta_ij`). A large loop
   coefficient therefore pulls the two directions toward `Theta_ji = Theta_ij^{-1}`, which is the
   hard-constraint limit.

Scope discipline. Edit only the files in your assignment. Do not edit `main.tex`,
`supplement_*.tex`, `SPEC.md`, `CONTENT.md`, `references.bib`, or `ca_open_problems.tex`: the
dispatcher owns them. If your change needs one of them, write a request instead:

- New bibliography entries: append BibTeX to `...\remediation\requests\bib-<your id>.bib`. First
  grep `references.bib` for an existing key and reuse it. A new entry's metadata (authors, title,
  venue, year, volume, pages, DOI or arXiv id) must be checked against Crossref, the publisher, or
  arXiv in this session; never invent a DOI, page range, or locator. Cite a locator (theorem,
  section, or equation) only if you read it in the source; otherwise cite without a locator and
  record that in your log. Key style: match the bib (look at existing keys).
- New or changed open-problem table rows (item, `\cref` locator, settling condition): write them to
  `...\remediation\requests\open-problems-<your id>.md`.
- CONTENT.md changes (new or changed constructions or results): `...\remediation\requests\content-<your id>.md`.

New statements. A new corollary, proposition, or remark that makes a claim carries one status tag
and a complete proof (ESTABLISHED) or a checked citation (STANDARD). Prefer the smallest change:
add a hypothesis, a qualifier, or a scope sentence rather than restructuring. Do not change the
title of any document. Do not weaken a verb to dodge a gap (R8). Do not reference labels that
another agent might be creating; reference only labels that already exist or that you create.

Verification. You cannot build the documents (the dispatcher builds once at the end), but check
that every `\cref` target you add exists (grep all `*.tex`), that braces balance, and that new
labels are unique across all three documents. If you write a new proof, check its load-bearing
step numerically or symbolically with a small script under
`...\remediation\scripts\<your id>\` (bare `python`, CPU, seconds).

Do not touch Lean, tamper tests, git (no commits, no branches, no stash), or any repository other
than the vault files in your assignment.

Log. Write `...\remediation\logs\<your id>.md`: for each finding ID, one line with APPLIED,
APPLIED-MODIFIED (say how), or NOT-APPLIED (say why), and the file:line of the edit. List any
requests you filed. Your final message: at most 20 lines, counts per status and anything the
dispatcher must do.
