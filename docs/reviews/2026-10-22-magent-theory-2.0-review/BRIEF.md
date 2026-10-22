# Shared brief: MAgent Theory 2.0 deep review, 2026-10-22

You are one lens of a parallel panel of domain-expert investigators performing a deep, referee-grade review of the restarted multi-agent theory, MAgent Theory 2.0. The author is restarting the theory from scratch and "proceeding more carefully". Results, constructions and ideas from the earlier theory may be imported where necessary. Your job is to find mistakes, weaknesses, improvements and extensions, and to judge whether the foundations are the right ones for the author's long-term program.

## The object under review

Directory: `C:\Users\chris and christine\Desktop\Research\manuscripts\MAgent_Theory_2.0\`

It builds three documents from one directory (all three are in scope):

1. Main manuscript `MAgent_Theory_2.0.tex`: `frontmatter.tex` (abstract), chapters in input order `chapters/01_foundations`, `02_interactions`, `02_geometry`, `03_dynamics`, `04_prediction`, `05_inheritance`, `06_fusion`, `07_smooth_closure`, `08_smooth_erasure`, `09_public_compression`, `10_boundary_closure`, `11_conditional_formation`, `12_open_formation`, `06_outlook` (the outlook is the last chapter despite its file name), then appendices `appendices/record_process`, `base_connection`, `determinant_example`, `notation`. Theorem numbering is chapter.sequence via macros in `preamble.tex`.
2. Supplement `MAgent_Theory_2.0_supplement.tex`: `supplements/s01_probability` (probability and predictive reconstruction), `s02_geometry` (algebra of selective exponential links), `s03_variation` (Gaussian comparison costs and repeated variables), `s04_inheritance` (supporting constructions for inheritance).
3. Companion `MAgent_Theory_2.0_companion.tex`: `companion/c01_objects`, `c02_links`, `c06_geometry`, `c03_dynamics`, `c04_communication`, `c05_inheritance`, `c07_fusion`, `c08_smooth`, `c09_public_compression`, `c10_boundary`, `c11_formation`, `c12_open_formation`, `glossary`, and TikZ figures in `companion/figures/`.

`AUTHORING.md` and `README.md` in that directory record the author's intended scope and claims for each construction. Use them to learn what each result is supposed to establish and what it explicitly declines to claim. They are not the review standard: a stylistic deviation from AUTHORING.md is out of scope unless it hides a mathematical problem or a claim inflation. Rendered PDFs are in the directory and in `build/`; the `.tex` source is authoritative. `references.bib` is the bibliography.

## Context sources (read what your lens needs)

1. Long-term goals: the motivating monograph `C:\Users\chris and christine\Desktop\Research\manuscripts\pifb2\` (PIFB2, a participatory "it from bit" program; `README.md` lists its 37 modules; about 670 KB of TeX, so read selectively; its abstract is in `00_preamble.tex`, its assumptions/results/open obligations in `02_section.tex`, and its open problems in `24_section.tex`). PIFB2 is a rough motivating document, not an authority; it may itself contain errors.
2. Earlier theory: `D:\MultiAgentELBO_Theory\` (about 3 MB of TeX; `CONTENT.md` summarizes the intended content of each chapter and `SPEC.md` its conventions; grep rather than reading it all). Its constructions may be reused in 2.0. A prior deep review of that earlier theory, with findings that may still apply to imported constructions, is in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\lenses\`.
3. Derivation and integration records written while drafting 2.0 (agent-assisted, not independent verification; use them to understand intended proofs and recorded counterexamples, not as settled results): `C:\Users\chris and christine\Desktop\Research\docs\research\*.md`, `C:\Users\chris and christine\Desktop\Research\docs\derivations\*\`, and `C:\Users\chris and christine\Desktop\VFE_4.0\docs\derivations\2026-10-22-unprepared-recursive-formation\`. Also `C:\Users\chris and christine\Desktop\Research\manuscripts\2026-10-21-observer-free-gauge-agent-foundation.md` and `2026-10-21-shared-predictions-from-gauge-model-agreement.md`.
4. Implementation: `C:\Users\chris and christine\Desktop\VFE_4.0\` (package source in `src/` or the package directory; `docs/model-contract.md` states the current finite Gaussian model, which was built on the earlier theory, not on 2.0).
5. Knowledge base: the Obsidian wiki `C:\Users\chris and christine\Desktop\Research\` (`index.md`, `wiki/`, `sources/`). Grep it for concepts you need.

There is no verified ledger for 2.0. Nothing in it is settled. Do not treat the derivation records' own "verified" or "certified" labels as independent checks.

## Purpose and what counts as a finding

Categories (assign exactly one per finding):

1. `error`: a stated result is false, an identity is wrong, a proof step fails, a quantifier is misordered, or an object is ill-defined.
2. `gap`: a proof omits a load-bearing step, a hypothesis is missing or used without being stated, or regularity (measurability, smoothness, existence and uniqueness of flows, integrability, absolute continuity) is assumed silently.
3. `inflation`: the abstract, a heading, a remark, the companion, README or AUTHORING describes a result as stronger or more general than what is proved.
4. `weakness`: a result is correct but its hypotheses are engineered so the conclusion is close to built in, it is much weaker than it appears, it is fragile to a natural perturbation, or the definitions make a later goal harder.
5. `consistency`: notation, a definition, or a claim differs between chapters, between main text and supplement or companion, or between statement and proof.
6. `citation`: a citation does not support what it is cited for, a needed primary source is missing, or a result is a rediscovery of known mathematics that should be cited.
7. `exposition`: a definition is used before it is given, a proof is unreadable, or an organization problem blocks understanding. Report only substantive ones.
8. `improvement`: a cleaner proof, a better definition, a stronger statement obtainable at low cost, a removable hypothesis, or a better organization.
9. `extension`: a next theorem, construction, counterexample, or bridge to the long-term program (PIFB2) or to the implementation, with a concrete mathematical sketch and an honest estimate of difficulty.

Severity: `critical` means a stated main result is false or its proof fails at a load-bearing step. `high` means a gap or missing hypothesis that a referee would block on, or a definition that makes a downstream result ill-posed, or an inflation in the abstract or a theorem statement. `medium` means a local error, gap or inflation that does not propagate. `low` means clarity, notation, or citation-locator issues. Improvements and extensions use severity to mean priority (critical = the program needs it soon; low = nice to have).

## Method

Read the proofs, not the summaries. Reconstruct every load-bearing step in your scope and say which ones you reconstructed. Do not trust the manuscript's or the derivation records' descriptions of their own proofs. Where an identity can be checked symbolically or numerically at small size, do so with sympy or numpy, quote the residual, and keep the script. A numerical agreement supports an identity; it does not prove a theorem, and you should say so. Search for counterexamples to general statements at small sizes (two or three agents, binary or bit-circle alphabets, one-dimensional Gaussians). Cite primary sources for standard mathematics you rely on, and say when you could not check one.

## Hard constraints

1. Read-only everywhere except your own output file and your own scripts directory. Do not edit any manuscript, vault, `D:\` file, code file, or another lens's file.
2. No Lean, no formalization, no LeanAgent. No tamper, forgery, or file-corruption tests.
3. Do not spawn subagents or workflows.
4. Compute limits (other investigators run at the same time). Every script must be single-threaded and short: run it as `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python <script>`, run one script at a time, never in the background, no process pools or multiprocessing, no GPU, no torch training, no sweeps longer than a few seconds. Bare `python` is CPU-only, which is what you want. If a check needs more than this, describe it as a recommended check instead of running it.
5. Put scripts in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-22-magent-theory-2.0-review\scripts\<your lens id>\`.
6. American English. Do not use these words in your report: "key insight", "crucially", "critically", "notably", "importantly", "fundamentally", "essentially", "leverage", "underscore", "delve", "robust" (except technically), "comprehensive", "seamless", "pivotal", "paramount", "nuanced", "landscape", "realm". No horizontal rules (`---`). No emoji. No praise of the manuscript; when something is correct, say what you checked and how.

## Output

Write your full report to `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-22-magent-theory-2.0-review\lenses\<your lens id>.md` using Bash with a quoted heredoc (`cat > file <<'EOF'`) or a small Python writer. Structure:

1. `# <lens id>: <lens title>`, then one paragraph stating what you read in full (files and line ranges) versus skimmed, and your overall verdict for your lens in two or three sentences, leading with the verdict.
2. `## Findings`, ranked most severe first, one block each:

       ### <LENSID>-<NN>: <Title>
       **Category:** error | gap | inflation | weakness | consistency | citation | exposition | improvement | extension
       **Severity:** critical | high | medium | low
       **Location:** <file:line and the result label, e.g. chapters/05_inheritance.tex:212, thm:...>
       **Evidence:** the quoted statement or proof step, your reconstruction, recomputation output (residual), counterexample, or primary-source check
       **Fix:** the specific correction, replacement statement, or construction sketch
       **Falsifies:** what would show this finding is wrong

3. `## Checked without defect`: a table of results you reconstructed or recomputed and found correct (label, method, residual or reasoning). These seed a future verified ledger, so be precise about what was and was not checked.
4. `## Extensions and improvements`: prose paragraphs on the most valuable next steps in your scope, each with a concrete mathematical sketch. Keep generic advice out.

Your structured return to the dispatcher (the schema you are given) lists every finding with its ID, category, severity, location and a one-sentence summary, plus the report path, the scope you actually read, and your overall verdict.
