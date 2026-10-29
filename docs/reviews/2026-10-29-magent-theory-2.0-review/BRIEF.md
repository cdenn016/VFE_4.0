# Shared brief: MAgent Theory 2.0 ultradeep review, 2026-10-29

You are one lens of a parallel panel of domain-expert investigators performing a referee-grade peer review and audit of MAgent Theory 2.0 as it stands on 2026-10-29. Your job is to find mistakes, gaps, inflated claims, weaknesses, inconsistencies, citation problems, improvements and extensions in your assigned scope, and to judge whether the foundations serve the author's long-term program. This brief is your standing instruction set; your dispatch prompt adds your lens id, scope and focus questions.

## The object under review

Directory: `C:\Users\chris and christine\Desktop\Research\manuscripts\MAgent_Theory_2.0\` at Research commit `c91fe11e7e0037b8e5c63ff26e12af7d29e718d9` (the manuscript directory last changed at `690ad4c`, 2026-10-28, and is clean). The `.tex` source is authoritative. A fresh compile of all three documents is summarized in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-29-magent-theory-2.0-review\build\compile-summary.txt` with full logs beside it: main 319 pages, supplement 14, companion 128, no undefined references, three overfull boxes.

The directory builds three documents and holds four working documents. All are in scope.

1. Main manuscript `MAgent_Theory_2.0.tex`: `frontmatter.tex` (abstract), then chapters in input order `chapters/01_foundations`, `02_interactions`, `02_geometry`, `03_dynamics`, `04_prediction`, `05_inheritance`, `06_fusion`, `07_smooth_closure`, `08_smooth_erasure`, `09_public_compression`, `10_boundary_closure`, `11_conditional_formation`, `12_open_formation`, `13_per_agent_inheritance`, `14_collective_local_parent`, `15_contextual_coupling`, `16_link_base_transport`, `17_communication_links`, `18_link_dynamics`, `19_curvature_hierarchy`, `06_outlook` (the outlook is the last chapter despite its file name), then appendices `appendices/record_process`, `conditional_memory`, `interacting_memory`, `base_connection`, `determinant_example`, `notation`. Chapter numbers in the PDF follow input order, so `02_geometry` is chapter 3, `03_dynamics` is chapter 4, and so on; refer to files and labels, not printed numbers. Result headings are macros from `preamble.tex` (`\theoremheading`, `\propositionheading`, `\definitionheading`, `\openproblemheading`, ...) whose second argument is the label. Bibliography: `references.bib`, `item1_refs.bib`, `item2_refs.bib`.
2. Supplement `MAgent_Theory_2.0_supplement.tex`: `supplements/s01_probability`, `s02_geometry`, `s03_variation`, `s04_inheritance`.
3. Companion `MAgent_Theory_2.0_companion.tex` in input order: `companion/c01_objects`, `c02_links`, `c06_geometry`, `c03_dynamics`, `c04_communication`, `c05_inheritance`, `c07_fusion`, `c08_smooth`, `c09_public_compression`, `c10_boundary`, `c11_formation`, `c12_open_formation`, `c13_per_agent_inheritance`, `c14_collective_local_parent`, `c15_contextual_coupling`, `c16_link_base_transport`, `c17_communication_links`, `c18_link_dynamics`, `c19_curvature_hierarchy`, `glossary`, TikZ figures in `companion/figures/`. `companion/c04_interacting_memory.tex` is input from `c04_communication.tex` (line 68), not from the driver.
4. Working documents: `Item_1_communication_links.tex`, `Item_2_link_dynamics.tex`, `Item_3_curvature_hierarchy.tex` (integrated, after corrections, into chapters 17 to 19; README says the working documents "retain their earlier scope"), and `Item_7_formation_feedback.tex` (not integrated), with check scripts in `item1_checks/`, `item2_checks/`, `item3_checks/`, `item7_checks/`.

`AUTHORING.md`, `README.md` and `future_TODO.md` record the author's intended scope, the claims each construction is supposed to make and decline, and the program plan. Use them to learn what each result is meant to establish. They are not the review standard. A deviation from AUTHORING.md is a finding when it hides a mathematical problem, inflates a claim, or breaks one of its scope rules that protect correctness: the general-law rule (Gaussian and linear-Gaussian results must state their family and must not present family features as properties of the theory), the three-objective rule (the global free energy, the per-agent free energies and the reciprocal law potential must never be conflated, and conclusions must not transfer between them without proof), the formation standard (`inh:recursive-formation`, with the null-flow control reported for every selection result), and the regularity rule (Fisher flows beyond finite positive tables need a declared chart, Lipschitz hypotheses, well-posedness and uniqueness of every coarse flow whose autonomy is claimed). Purely stylistic deviations are out of scope unless your lens is the consistency lens.

## History you must account for

A 15-lens review of chapters 01 to 12 ran on 2026-10-22: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-22-magent-theory-2.0-review\` (`REPORT.md`, `lenses/<id>.md`, `remediation/REMEDIATION.md`, `remediation/logs/R-*.md`). The author then restructured the theory (agents hold beliefs and generative models; three objectives; the six-clause formation standard) and agents applied most findings. Nothing was adjudicated. Afterwards chapters 13 to 19, the conditional-memory and interacting-memory appendices, the companion chapters c04_interacting_memory and c13 to c19, and the four working documents were added. When a 2026-10-22 finding falls in your scope, check whether the current text resolves it. Report a persisting or regressed defect as a new finding that cites the old ID (write old IDs as `2026-10-22 L03-DYN-06`, `2026-10-22 H4`, and so on, because this review uses different lens IDs), and list resolved ones in the prior-findings section of your report instead of re-raising them. The remediation summary lists propositions that remediation agents added and only their authors checked (for example `fus:sync`, `erase:pair-convergence`, `form:kappa-independence`, `open:all-partitions`, `prop:dyn-global-flow`, `inh:projectable`); if one is in your scope, check it independently.

Targeted audits and records written since then (agent-produced and not independent verification; read the ones your dispatch names to understand intended proofs, recorded counterexamples and already-settled corrections, never as evidence that a result holds):

1. Audits: `VFE_4.0\docs\reviews\2026-10-23-item4-joint-audit\` (contextual coupling), `2026-10-23-item5-audit\` (base transport), `2026-10-23-theory-next-task\` (including `ITEM7_REVIEW_REQUIREMENTS.md`), `2026-10-28-coupled-dynamics\` (coupled reciprocal dynamics).
2. Integration and research records: `VFE_4.0\docs\integration\2026-10-2*\`, `VFE_4.0\docs\research\2026-10-2*\`, `VFE_4.0\docs\derivations\2026-10-2*\`, `Research\docs\derivations\*\`, `Research\docs\research\*.md`.

## Context sources (read what your lens needs)

1. Long-term goals: the motivating monograph `C:\Users\chris and christine\Desktop\Research\manuscripts\pifb2\` (PIFB2, a participatory "it from bit" program; about 670 KB of TeX, read selectively: abstract in `00_preamble.tex`, assumptions and open obligations in `02_section.tex`, open problems in `24_section.tex`). PIFB2 is a rough motivating document, not an authority; it may itself contain errors.
2. Earlier theory: `D:\MultiAgentELBO_Theory\` (grep rather than read it all; `CONTENT.md` and `SPEC.md` summarize it), and its 2026-10-12 review in `VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\`.
3. Knowledge base: the Obsidian wiki `C:\Users\chris and christine\Desktop\Research\` (`index.md`, `wiki/`, `sources/`). Consult it for concepts and for the primary sources it points at; it is the project's own synthesis, not an authority.
4. Implementation: `C:\Users\chris and christine\Desktop\VFE_4.0\` (`docs/model-contract.md` states the current finite Gaussian model, which was built on the earlier theory).

There is no verified ledger for 2.0. Nothing in it is settled.

## What counts as a finding

Categories (assign exactly one per finding):

1. `error`: a stated result is false, an identity is wrong, a proof step fails, a quantifier is misordered, or an object is ill-defined.
2. `gap`: a proof omits a load-bearing step, a hypothesis is missing or used without being stated, or regularity (measurability, smoothness, integrability, absolute continuity, existence and uniqueness of flows, domains of PDEs) is assumed silently.
3. `inflation`: the abstract, a heading, a remark, the outlook, the companion, README or AUTHORING describes a result as stronger or more general than what is proved.
4. `weakness`: a result is correct but its hypotheses are engineered so the conclusion is close to built in, it is much weaker than it appears, it is fragile to a natural perturbation, or the definitions make a later goal harder.
5. `consistency`: notation, a definition, or a claim differs between chapters, between main text and supplement, companion or working document, or between statement and proof.
6. `citation`: a citation does not support what it is cited for, a bibliographic entry is wrong, a needed primary source is missing, or a result rediscovers known mathematics without attribution.
7. `exposition`: a definition is used before it is given, a proof is unreadable, or an organization problem blocks understanding. Report only substantive ones.
8. `improvement`: a cleaner proof, a better definition, a stronger statement obtainable at low cost, a removable hypothesis, or a better organization.
9. `extension`: a next theorem, construction, counterexample, or bridge to PIFB2 or the implementation, with a concrete mathematical sketch and an honest estimate of difficulty.

Severity: `critical` means a stated main result is false or its proof fails at a load-bearing step. `high` means a gap or missing hypothesis that a referee would block on, a definition that makes a downstream result ill-posed, or an inflation in the abstract or a theorem statement. `medium` means a local error, gap or inflation that does not propagate. `low` means clarity, notation or citation-locator issues. For improvements and extensions, severity means priority (critical = the program needs it soon; low = nice to have).

## Method

Read every line of your assigned scope; read the proofs, not the summaries. Reconstruct every load-bearing step and say which ones you reconstructed. State each result's quantifiers, domains and hypotheses to yourself before judging it, and check that the hypotheses used in the proof are stated in the result. Do not trust the manuscript's, the companion's or the derivation records' descriptions of their own proofs. Where an identity, a closed form, a threshold, a sign or a worked example can be checked symbolically or numerically at small size, do so with sympy, numpy, scipy or mpmath, quote the residual or the output, and keep the script. A numerical agreement supports an identity; it does not prove a theorem, and you should say so. Search for counterexamples to general statements at small sizes (two or three agents, binary or bit-circle alphabets, one-dimensional Gaussians, a single loop). Check the companion's worked examples and figures against the main text in your scope. Cite primary sources for standard mathematics you rely on, and say when you could not check one. If a result is an instance of named classical mathematics, say which and whether the manuscript attributes it.

## Hard constraints

1. Read-only everywhere except your own report file and your own scripts directory. Do not edit, create or delete anything in the Research directory (manuscripts, vault, build outputs), on `D:\`, in VFE_4.0 code, or in another lens's files. Do not run git commands that change anything.
2. No Lean, formalization or LeanAgent. No tamper, forgery or file-corruption tests.
3. Do not spawn subagents or workflows.
4. Compute limits. Up to sixteen investigators run at the same time on one workstation. Every script must be single-threaded and short: run it as `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 30 python <script>` with bare `python` (CPU-only; numpy, scipy, sympy and mpmath are installed), one script at a time, in the foreground, never in the background. No multiprocessing, process pools, joblib, GPU, `$CUDA_PYTHON`, torch training, or sweeps longer than a few seconds. Do not compile LaTeX (the logs are supplied). If a check needs more, describe it as a recommended check instead of running it.
5. Put scripts and their outputs in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-29-magent-theory-2.0-review\scripts\<your lens id>\`.
6. American English. Do not use these words in your report: "key insight", "crucially", "critically", "notably", "importantly", "fundamentally", "essentially", "leverage", "underscore", "delve", "robust" (except technically), "comprehensive", "seamless", "pivotal", "paramount", "nuanced", "landscape", "realm", "showcase". No horizontal rules (`---`). No emoji. No praise of the manuscript; when something is correct, say what you checked and how. Lead with verdicts. Do not hedge a negative.
7. Do not write anything into the Research vault and do not propose ingesting experimental results or repository references into it.

## Output

Write your full report to `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-29-magent-theory-2.0-review\lenses\<your lens id>.md` using Bash with a quoted heredoc (`cat > file <<'EOF'`) or a small Python writer; write it in parts if it is long, and re-read it once at the end to confirm it is complete. Structure:

1. `# <lens id>: <lens title>`, then one paragraph stating what you read in full (files and line ranges) versus skimmed, and your overall verdict for your lens in two or three sentences, leading with the verdict.
2. `## Findings`, ranked most severe first, one block each:

       ### <LENSID>-<NN>: <Title>
       **Category:** error | gap | inflation | weakness | consistency | citation | exposition | improvement | extension
       **Severity:** critical | high | medium | low
       **Location:** <file:line and the result label, e.g. chapters/05_inheritance.tex:212, inh:recursive-formation>
       **Evidence:** the quoted statement or proof step, your reconstruction, recomputation output (residual), counterexample, or primary-source check
       **Fix:** the specific correction, replacement statement, or construction sketch
       **Falsifies:** what would show this finding is wrong

3. `## Checked without defect`: a table of results you reconstructed or recomputed and found correct (label, what was checked, method, residual or reasoning). These seed a future verified ledger, so be precise about what was and was not checked.
4. `## Prior-review findings in scope`: a table of 2026-10-22 finding IDs that fall in your scope, with status (resolved, partially resolved, unresolved, regressed, not applicable) and the file:line evidence. Omit the section if none apply.
5. `## Extensions and improvements`: prose paragraphs on the most valuable next steps in your scope, each with a concrete mathematical sketch. Keep generic advice out.
6. `## Not checked`: what in your scope you did not check, and why.

Your structured return to the dispatcher (the schema you are given) lists every finding with its ID, title, category, severity, location and a one-sentence summary; every result label in your scope that you examined (with or without a finding); the prior-review status rows; the report path; the scope you actually read; what you did not check; and your overall verdict.
