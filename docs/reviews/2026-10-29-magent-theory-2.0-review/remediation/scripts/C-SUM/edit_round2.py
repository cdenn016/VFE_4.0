"""Second round of exact-match edits (CRLF preserved) in C-SUM files."""
import sys

root = sys.argv[1]
EOL = "\\\\"

table_rows = [
    ("open:geo-gauged-flow", "Extensions of the contextual Fisher flow"),
    ("op:dyn-three-pictures", "Relating the global, per-agent, and reciprocal objectives"),
    ("inh:type-closure-open", "Type-closed parents"),
    ("inh:formation-open", "A recursive formation with a derived selector"),
    ("op:boundary-minimality", "Minimality of the unordered profile tree"),
    ("form:approximate-open", "Approximate inheritance off the common-precision family"),
    ("open:necessity-open", "Interaction-necessary selection in the bit-circle architecture"),
    ("lp:local-parent-open", "Causal local parents on arbitrary preparations"),
    ("it1:open-learning", "Learning the dictionaries beyond the tangent model"),
    ("it1:open-codes", "Codes as part of the species model"),
    ("it1:open-free-energy", "Dictionaries in the free-energy pictures"),
    ("it1:open-general", "Beyond Gaussian codes"),
    ("it2:open-reciprocal", "Coupled link dynamics beyond the finite example"),
    ("it2:open-convergence", "Convergence of per-agent link flows"),
    ("it2:open-nonlinear", "Link dynamics beyond the tangent model"),
    ("it2:open-general", "Link dynamics for general laws"),
    ("it2:open-messages", "What agents must exchange"),
    ("it3:open-formation", "Formation with top-down feedback"),
    ("it3:open-fluctuation", "Fluctuations beyond the fixed tangent experiment"),
    ("it3:open-general", "Curvature costs for general content laws"),
    ("it3:open-gains", "Learning gains from loop records"),
    ("open:conn-derived", "Selecting and continuing interaction-derived transport"),
    ("imem:open", "Smaller sufficient states, asymptotic agreement, and changing preparations"),
]
table = (
    "\n\nThe open problems stated in the chapters and appendices are collected below by label.\n"
    "\\begingroup\n\\small\n"
    "\\begin{longtable}{@{}p{0.22\\textwidth}p{0.72\\textwidth}@{}}\n"
    f"Open problem & Question {EOL}\n\\endfirsthead\nOpen problem & Question {EOL}\n\\endhead\n"
    + "\n".join(f"\\cref{{{lab}}} & {txt}. {EOL}" for lab, txt in table_rows)
    + "\n\\end{longtable}\n\\endgroup"
)

EDITS = {
    "chapters/06_outlook.tex": [
        ("None of the results establishes an indefinite spacetime metric, a thermodynamic limit, typicality under random initialization, or unbounded recursive formation.",
         "None of the results establishes an indefinite spacetime metric, a thermodynamic limit, typicality under random initialization, or unbounded recursive formation." + table),
        ("Their quotients exist before membership occurs, so the dynamics selects when a description is used without creating a new operational equivalence.",
         "Their quotients exist before membership occurs, so the dynamics selects when a description is used without creating a new operational equivalence. Every exact compression realized here erases data that no agent, channel, or energy term reads, and no mechanism is given by which an interaction creates an exact symmetry or decoupling; a graded notion of inheritance measured by the exact defect functionals of the inheritance chapter is not developed."),
    ],
    "frontmatter.tex": [
        ("For the normalized per-agent free energies, a separate construction gives persistent grouping against conflicting records.",
         "For the normalized per-agent free energies, a separate construction gives persistent grouping against conflicting records; its small-signal limit is anchored opinion pooling of the Friedkin--Johnsen type."),
    ],
    "README.md": [
        ("In `13_per_agent_inheritance` the persistent grouping passes",
         "In `13_per_agent_inheritance` the persistent grouping, whose small-signal limit is the anchored opinion pooling of Friedkin and Johnsen, passes"),
        ("The assumptions supply the cross-agent relation and communication policy; new measurements, model adaptation, minimal retention, and formation remain open.",
         "The assumptions supply the cross-agent relation and communication policy; new measurements, model adaptation, minimal retention, asymptotic agreement of repeated dialogue, and formation remain open."),
    ],
    "AUTHORING.md": [
        ("the mechanical connection and O'Neill quotient metrics, symmetric criticality and fixed-point subspaces, asymptotic phase, centroid clustering), say so",
         "the mechanical connection and O'Neill quotient metrics, symmetric criticality and fixed-point subspaces, asymptotic phase, centroid and linkage clustering, anchored opinion pooling, agreement theorems, the method of slices and freezing), say so"),
        ("Its signed cross-covariance is distinct from the positive posterior Fisher tensor.",
         "Its signed cross-covariance is distinct from the positive posterior Fisher tensor, which bounds it (`cmem:response-bound`)."),
    ],
    "future_TODO.md": [
        ("Relay records are optional; formation with top-down feedback, fluctuation effects, and dynamically determined gains remain open.",
         "Relay records are optional; formation with top-down feedback, fluctuation effects, and dynamically determined gains remain open. A candidate refinement of `it3:open-formation` (recorded 2026-10-29): holonomy-driven walls, in which the learned dictionary sector itself selects blocks. The cohomological part of the induced action is stated in the small-noise limit and, inside the tangent model only, in the weak-anchor limit; which regime defines it is an open author decision."),
    ],
}
for rel, edits in EDITS.items():
    p = f"{root}/{rel}"
    t = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    for old, new in edits:
        assert t.count(old) == 1, (rel, old[:70])
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(t.replace("\n", "\r\n"))
    print("ok", rel, len(edits))
