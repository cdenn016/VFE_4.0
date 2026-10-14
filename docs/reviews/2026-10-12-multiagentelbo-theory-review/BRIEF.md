# Shared brief: MultiAgentELBO theory deep review, 2026-10-12

You are one lens of a parallel expert panel performing a deep, referee-grade review of the
MultiAgentELBO theory. The canonical and only copy of the theory is the Research vault folder

    C:\Users\chris and christine\Desktop\Research\manuscripts\MultiAgentELBO_Theory\

It builds three documents from one directory: the main manuscript (`main.tex`; chapters `c01`
to `c15` and appendices `ca_*`), Supplement G (`supplement_gaussian.tex`; files `g*`), and
Supplement B (`supplement_base.tex`; files `b*`). `CONTENT.md` lists the intended mathematical
content and typed constructions. `SPEC.md` is the authoring spec for the renormalization
write-up; the author says it is specific to renormalization. Use it only to look up notation.
It is NOT the standard for this review, and SPEC-compliance findings (status-tag format, prose
rules, label prefixes, R1 to R16 conformance) are out of scope unless a rule violation hides a
mathematical error.

Purpose. Find real defects in the mathematics and its claims: wrong identities, missing or
false hypotheses, proofs with gaps, status inflation (a statement tagged ESTABLISHED whose proof
does not establish it, or STANDARD whose citation does not say it), circularity, citations that
do not support what they are cited for, and undefined or inconsistent objects. A second purpose:
the folder `C:\Users\chris and christine\Desktop\VFE_4.0` (currently empty) will hold a new
multi-agent implementation that supplements the MultiAgentELBO code
(`C:\Users\chris and christine\Desktop\MultiAgentELBO\src`). For your chapters, also say what an
implementation must compute, which objects are closed form versus requiring approximation or
sampling, and where the theory under-specifies an algorithm (an update rule, a normalizer, a
domain condition) that an implementer would have to invent.

Settled record (read first, do not re-litigate). The verified ledger for this theory is
`C:\Users\chris and christine\Desktop\MultiAgentELBO\docs\reviews\verified-ledger.md` (735
lines; grep it for the labels in your scope rather than reading it all). An entry is settled
unless the manuscript text at its label has changed since, or a later entry narrows it. Do not
re-flag verified identities or re-open recorded rejections. If you find the text at a settled
label has changed and the change introduces a defect, say so explicitly and cite the entry.
Prior review notes in the vault, `Research\sources\manuscripts\multiagentelbo-2026-*.md`, are
also part of the record.

Method. Recompute every load-bearing identity you can, using sympy or small float64 numerical
checks (bare `python` is CPU-only and fine; tiny dimensions, each check well under a few
seconds). Do not trust the manuscript's algebra or its summaries of its own proofs: read the
proof. Verify any citation you rely on against the primary source when feasible (web access is
available); if you could not check it, say so. Put any scripts you write under
`C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\scripts\<your lens id>\`
and quote their output (residuals) in your findings.

Hard constraints. Read-only on the vault and on the MultiAgentELBO repository: do not edit any
manuscript or code file. No Lean or formalization. No tamper or forgery tests. Do not spawn
further agents. American English. Do not use these words in your report: "key insight",
"crucially", "critically", "notably", "importantly", "fundamentally", "leverage", "underscore",
"robust" (except technically), "comprehensive". No horizontal rules.

Output. Write your full report to
`C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-12-multiagentelbo-theory-review\lenses\<your lens id>.md`
(use Bash with a quoted heredoc or a small Python writer). Structure:

1. One paragraph: scope read (files and line ranges actually read in full versus skimmed),
   overall verdict for your lens.
2. Findings, ranked most severe first, one block each:

       ### <ID>: <Title>
       **Location:** <file:line and/or label>
       **Severity:** critical | high | medium | low
       **Evidence:** recomputation output, quoted proof step, or primary-source check
       **Fix:** the specific correction
       **Falsifies:** what would show this finding is wrong
       **Status:** the manuscript's tag (ESTABLISHED/STANDARD/NUMERICAL or heading type) and whether it is inflated

   Severity: critical = a stated result is false or its proof fails at a load-bearing step;
   high = a gap or missing hypothesis that a referee would block on, or a definition that makes
   a downstream result ill-posed; medium = local error or inflation that does not propagate;
   low = clarity, notation, or citation-locator issues.
3. Verified-without-defect table: identities and proofs you recomputed or checked and found
   correct (label, method, residual). These feed the ledger.
4. Implementation notes for VFE_4.0 (prose paragraphs): what must be computed, closed form vs.
   approximate, under-specified algorithmic choices, numerical hazards.

Your final message back to the dispatcher: at most 30 lines, listing finding IDs with severity
and one-line titles, plus the path of the report file. Do not paste the full report.
