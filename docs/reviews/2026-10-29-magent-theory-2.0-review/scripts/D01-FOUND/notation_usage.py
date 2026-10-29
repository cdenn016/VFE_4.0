r"""Notation-usage sampler for lens D01-FOUND (read-only text search).

For each symbol meaning, search the LaTeX sources of MAgent Theory 2.0 for a
regular expression that identifies that meaning, and print the number of hits
and the first few file:line locations.  This is a text search, not a proof:
a hit shows that a meaning is used at that line; absence of a hit for a
pattern does not show that the meaning is unused elsewhere.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(
    r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0"
)
DIRS = ["chapters", "appendices", "supplements", "companion"]

# (symbol, meaning, regex)
PATTERNS: list[tuple[str, str, str]] = [
    ("G", "structure group", r"principal \$G\$-bundle|G\\times\\Prob|G=\\GL\(K"),
    ("G_i", "evaluator kernel", r"G_i\(m,|G_i:\\mathsf M_i"),
    ("G_ij", "Jensen-gap term (ch3)", r"G_\{ij\}=\\int"),
    ("\\mathsf G", "collective Fisher metric", r"\\mathsf G_\{\\boldsymbol p\}"),
    ("K", "dimension of V_q", r"K=\\dim V_q|row rank \$K\$|\\GL\(K,\\R\)"),
    ("K", "generic Markov kernel", r"Markov kernel \$K:|common normalized record kernel|Let \$s,t\$ be laws on a model space and let \$K,L\$"),
    ("K_a", "record mechanism on Z", r"K_a\(do_a\\mid z\)"),
    ("K", "subgroup (ch10)", r"subgroup \$K\\le"),
    ("T_q", "row temperature", r"T_q,T_s>0|row temperature"),
    ("T_j^a", "sending map", r"T_j\^a:V_a"),
    ("T", "time / census / horizon", r"entrance time \$T\$|census \$T\$|small \$T\$"),
    ("T", "measurable code", r"measurable code \$T\$|code \$T\$"),
    ("T", "deterministic readout / statistic", r"deterministic readout \$T\$|statistic \$T\$|given \$T\$"),
    ("T", "base transport", r"\$T\(c,c\)=I"),
    ("P", "principal bundle", r"P\\to\\mathcal C"),
    ("P_i", "generative joint", r"P_i\(dm,dk"),
    ("P_0", "content / global prior", r"prior \$P_0"),
    ("P(x)", "partition sequence", r"P\(x\)|P\\neq P\^0"),
    ("g^F_P", "law named P in Fisher metric", r"g\^F_P\("),
    ("\\mathcal F", "free energy", r"\\mathcal F_i\(\\zeta_i;o\)|\\mathcal F\(Q;o\)"),
    ("F_D", "finite-design evaluation map", r"F_D\(z\)"),
    ("F_ij", "deterministic transition", r"F_\{ij\}\(x\)"),
    ("F^c", "inherited update", r"R\\circ F=F\^c\\circ R"),
    ("\\Phi", "content-bundle morphism", r"\\Phi\[u,k\]"),
    ("\\Phi", "flow of a construction", r"\\Phi,\\Phi\^0|\\Phi\^0"),
    ("\\phi", "group homomorphism", r"homomorphism \$\\phi:G"),
    ("\\phi", "presentation bijection (species)", r"bijections \$\\phi\$ of"),
    ("\\phi_I", "anchor phase", r"\\phi_I"),
    ("\\kappa", "Markov kernel (bundle morphism)", r"\\kappa:\\mathsf Y\\rightsquigarrow"),
    ("\\kappa_ij", "content sources (per-agent model)", r"\\kappa_\{ii\}\(m,"),
    ("\\kappa", "coupling strength", r"\\kappa g|kappa-independence|coupling \$\\kappa"),
    ("A", "intertwiner", r"A\\varrho_q\(g\)"),
    ("A_i", "comparison matrix", r"A_i k_i-B_\{ij\}k_j"),
    ("A", "registration update map", r"A:\\mathsf S\\times\\mathsf O"),
    ("A", "Dirichlet total concentration", r"total concentration \$A\$"),
    ("A^F", "minimizing connection", r"A\^F"),
    ("A_I", "belief shape sum", r"A_I,B_I"),
    ("B_ij", "comparison matrix", r"B_\{ij\}g_j\^\{-1\}"),
    ("B_i", "connection representative", r"\$B_i\$|B_i,F_i"),
    ("B^q_IJ", "native rows", r"B\^q_\{IJ\}"),
    ("\\mathcal B_i", "statistical bundle", r"\\mathcal B_i=P"),
    ("\\mathcal E_q", "content bundle", r"\\mathcal E_q=P"),
    ("\\mathcal E_o", "evidence-augmented potential", r"\\mathcal E_o="),
    ("\\mathcal E_0", "sum of local terms", r"\\mathcal E_0=\\sum"),
    ("E_i", "sample bundle", r"E_i=P\\times_G"),
    ("E_i^a", "coordinate copy of V_a", r"E_i\^q,E_i\^s|E_i\^a"),
    ("E", "standard Borel space (loop prop.)", r"bijections of a standard Borel space \$E\$"),
    ("H,H'", "loop transports", r"Let \$H,H'\$ be"),
    ("H_i", "state-to-record kernel", r"H_i:\\mathsf K_i"),
    ("H_p", "orthogonal complement", r"H_\{\\boldsymbol p\}="),
    ("H(x)", "membership history", r"H\(x\)"),
    ("H_a", "message space", r"message space \$H_a\$"),
    ("H(.)", "entropy", r"H\(O_\{1:n\}\)|H\(q_i\)"),
    ("J", "registered transition", r"J\(x,dx',do\)"),
    ("J_p", "infinitesimal action", r"J_\{\\boldsymbol p\}"),
    ("J", "spin coupling", r"e\^\{J\\sigma_1\\sigma_2\}"),
    ("J^q_i", "source label", r"J\^q_i"),
    ("\\mathsf O_i", "record alphabet", r"\\mathsf O_i"),
    ("O_ij,t", "record variable", r"O_\{ij,t\}"),
    ("O_i^a", "receiving map", r"O_i\^a:H_a"),
    ("R_i", "reference joint", r"R_i\(dk,dm\)"),
    ("R", "event kernel (record appendix)", r"kernel \$R:\\mathsf S"),
    ("\\mathcal R_ij", "induced record kernel", r"\\mathcal R_\{ij\}\(x,A\)"),
    ("r_i", "reference law over presentations", r"reference law \$r_i\$|model reference \$r_i\$"),
    ("r_ij", "comparison map", r"r_\{ij\}:E_i\^q"),
    ("r_D'D", "deletion map", r"r_\{D'D\}"),
    ("Z_{t+1}", "random configuration", r"Z_\{t\+1\}"),
    ("\\mathsf Z", "coefficient space", r"\\mathsf Z"),
    ("Z_o", "evidence", r"Z_o"),
    ("D", "finite design", r"\$D\$ is a finite set|D\\subseteq D'"),
    ("D_ij", "peer divergence", r"D\^q_\{ij\}"),
    ("\\ell_i", "record kernel", r"\\ell_i\(k,m;do\)|\\ell_i:\\mathsf K_i"),
    ("\\ell_v", "score", r"\\ell_v"),
    ("\\ell_o", "likelihood of z", r"\\ell_o"),
    ("\\gamma", "passive change on S", r"\\operatorname\{pr\}_i\\circ\\gamma=\\gamma_i"),
    ("\\gamma_i", "model-sector source row", r"\\gamma_i\$ on|\\beta_i,\\gamma_i"),
    ("\\sigma", "reference point in P_c", r"\\sigma\\in P_\{c_\*\}"),
    ("\\sigma_k", "spin variable", r"\\sigma_1\\sigma_2"),
    ("m", "model presentation", r"model presentations \$m"),
    ("m", "magnetization", r"m=\\tanh\(Jm\)"),
    ("\\rho_i", "record readout", r"\\rho_i:\\mathsf S_i"),
    ("\\rho_il", "presentation sources (ch3)", r"\\rho_\{ii\}=r_i"),
    ("\\varrho_q", "representation", r"\\varrho_q:G"),
    ("\\varrho_i", "log odds of references (ch3)", r"log odds \$\\varrho_i\$"),
    ("\\mu_j", "message law", r"\\mu_j\$ & Message law"),
    ("\\mu_i^a", "mean (moment family)", r"\\mu_i\^a"),
    ("\\omega_ij", "link generator", r"\\omega_\{ij\}\\in\\mathfrak g"),
    ("\\omega", "subgroup element (ch10)", r"\\omega\\in K"),
    ("W", "two-cycle loop", r"W=\\Theta_\{ij\}\\Theta_\{ji\}"),
    ("W_2", "Wishart law", r"W_2\("),
    ("\\Theta^s_ij", "bimeasurable model link (ch4)", r"\\Theta\^s_\{ij\}:\\mathsf M_j\\to\\mathsf M_i"),
    ("\\Theta^a_ij", "framed exponential", r"\\Theta_\{ij\}\^a=U_i\^a\\exp"),
    ("\\Theta^q_ij", "image of group link", r"\\Theta_\{ij\}\^q=\\varrho_q"),
    ("Q", "response kernel (ch10)", r"response kernel \$Q\$"),
    ("Q", "recognition law on Z", r"recognition law \$Q\$"),
]


def main() -> int:
    files: list[pathlib.Path] = []
    for d in DIRS:
        files.extend(sorted((ROOT / d).glob("*.tex")))
    texts = {f: f.read_text(encoding="utf-8", errors="replace").splitlines() for f in files}
    for sym, meaning, pat in PATTERNS:
        rx = re.compile(pat)
        hits: list[str] = []
        for f, lines in texts.items():
            for n, line in enumerate(lines, start=1):
                if rx.search(line):
                    hits.append(f"{f.parent.name}/{f.name}:{n}")
        shown = ", ".join(hits[:4])
        print(f"{sym:14s} | {meaning:40s} | {len(hits):3d} | {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
