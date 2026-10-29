"""Exact-match edits of companion/glossary.tex and the closing paragraph of
appendices/notation.tex (CRLF preserved)."""
import sys

EOL = "\\\\"
root = sys.argv[1]


def rows(*lines: str) -> str:
    return "\n".join(f"{line}{EOL}" for line in lines)


G = [
    (rows(r"$r_i,\ell_i,P_i$ & Prior over presentations, record likelihood, and the normalized generative model $r_iG_i\ell_i$. A model law predicts records only through $G_i$ and $\ell_i$."),
     rows(r"$r_i,\ell_i,P_i$ & Presentation prior, record likelihood, and the normalized generative model $r_iG_i\ell_i$. A model law predicts records only through $G_i$ and $\ell_i$.")),
    (rows(r"Three objectives & A global free energy of one normalized joint over all agents; per-agent free energies in which peers' laws are data; the reciprocal law potential, which moves sources toward their receivers and is not a free energy of any fixed model with the agent laws as recognition laws."),
     rows(r"Three objectives & A global free energy of one normalized joint over all agents; per-agent free energies in which peers' laws are data; the reciprocal law potential, which moves sources toward their receivers. With fixed rows on finite alphabets the potential is not the free energy of a finite normalized model when an attended edge is an identity channel, nor, with fixed input-dependent channels, of a model whose recognition is the product of the agent laws; outside these hypotheses exact identifications exist.",
          r"Observable algebra & The declared functions of the configuration that agents, channels, records, or a declared external protocol can read. A distinction is gauge relative to an algebra when no element separates it at any time along the dynamics; words such as physical or public are used only relative to a named algebra.")),
    (rows(r"$\Phi^0$ & The null flow: the same construction with every peer cost deleted. It is the control experiment for selection.",
          r"$P,P^0$ & Partition sequences selected along the flow and along the null flow, without event times.",
          r"(F5) & Interaction necessity: $P\neq P^0$ on an open set of admitted initial data. A shift of event times alone does not count.",
          r"Two-stage selection & Short for two-stage selection with exact inheritance: selection, inheritance, and a second selection among parents, with at least one of compression, interaction necessity, or counterfactual dependence of the second stage failing. It is not a formation, and the failed clauses are stated."),
     rows(r"$\Phi^0_{\mathrm{lp}},\Phi^0_{\mathrm{rn}}$ & The two null flows: every peer cost is deleted, with the peer labels kept, so that a self source keeps a Gibbs weight below one, or with them removed, so that a self source has weight one. They are the control experiments for selection and coincide when no self cost is nonzero.",
          r"$\Phi^{\mathrm{fr}}$ & The frozen-environment control: nothing is deleted, but every other agent's law is held at its initial value. It asks whether the selection already follows from each agent's response to the others' initial laws.",
          r"$P,P^0$ & Partition sequences selected along the flow and along a null flow, without event times and up to the order of merges of disjoint blocks.",
          r"$S_c,\mathsf C$ & A selector family and its set of constants, both fixed before the flows are analyzed. Verdicts are reported over the whole set.",
          r"$A_I,B_I$ & In the formation standard, the protocols a parent must answer as a block and the probes that can address its members before the handoff. Compression asks that some probe of $B_I$ see the erased distinction.",
          r"Counterfactual class & The declared set of actual and counterfactual handoff states against which the second stage must be shown to depend on the first.",
          r"(F5) & Interaction necessity: $P\neq P^0$ at every point of a nonempty open set of admitted data, against both nulls and for every constant of the declared set. A split verdict between the nulls fails, and a shift of event times alone does not count.",
          r"Two-stage selection & Short for two-stage selection with exact inheritance: selection, inheritance, and a second selection among parents, with at least one of compression, interaction necessity, or counterfactual dependence of the second stage shown to fail. It is not a formation, and the failed clauses are stated.",
          r"Single-stage selection & Selection and inheritance with one selection stage, so the recursion clause fails.",
          r"Undetermined & A construction with a clause neither established nor shown to fail; it is not shown to be a recursive formation, and the open clause is named.")),
    (rows(r"$\mathsf H_I,\phi_I$ & Relative constant angular offsets discarded by a block, and its retained local mean phase at the contextual anchor."),
     rows(r"$\mathsf H_I,\phi_I$ & Relative constant angular offsets discarded by a block, and its retained local mean phase at the contextual anchor.",
          r"$\ell_\psi,\mathcal X_{\mathrm{rich}}$ & The score of the common $\psi$ rotation, which is gauge relative to every declared algebra of the collective construction, and the rich preparation class on which its memory obstructions are stated.",
          r"$\mathcal R_{\mathrm{glob}}$ & The part of the regular branch of contextual alignments on which every transport is the unique global minimizer; only there is the reduced contextual cost the minimized one.",
          r"$X,H_X$ & A collective message matrix and $H_X=XX^T$. The displacement connection it defines preserves the fiber metric $H_X^{-1}$, so its holonomy is orthogonal for that metric.")),
]
N = [
    (r"A Markov kernel returns a probability measure, whereas an ordinary map returns one value.",
     r"The letter $\Gamma$ denotes an interaction transition $\Gamma_{ij}$ and, in the causal-memory obstruction of the collective construction, the evolution map of a memory. A Markov kernel returns a probability measure, whereas an ordinary map returns one value."),
]
for rel, edits in (("companion/glossary.tex", G), ("appendices/notation.tex", N)):
    p = f"{root}/{rel}"
    t = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    for old, new in edits:
        assert t.count(old) == 1, (rel, old[:70])
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(t.replace("\n", "\r\n"))
    print("ok", rel, len(edits))
