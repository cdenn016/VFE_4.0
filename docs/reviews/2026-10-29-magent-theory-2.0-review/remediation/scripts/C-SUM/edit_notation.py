"""Exact-match edits of appendices/notation.tex (CRLF preserved).

Row terminators are written as the constant EOL so that no line of this script
ends in a backslash.
"""
import sys

EOL = "\\\\"
p = sys.argv[1]
t = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def rows(*lines: str) -> str:
    return "\n".join(f"{line} {EOL}" for line in lines)


E = [
    (rows(r"$r_i,G_i,\ell_i$ & Prior over presentations, evaluator, and record kernel of agent $i$'s generative model."),
     rows(r"$r_i,G_i,\ell_i$ & Presentation prior, evaluator, and record kernel of agent $i$'s generative model. The term reference law is reserved for the laws $\bar q_i,\bar s_i$ in reference coordinates.",
          r"$l_i,\lambda_{\mathsf O_i}$ & Declared record density of $\ell_i$ and the declared sigma-finite reference measure on $\mathsf O_i$.",
          r"$K_i^{(n)}$ & Composite kernel of $n$ records given a presentation, with $K_i^{(1)}=K_i^{\mathrm{rec}}$.",
          r"$\mathbf g$ & A passive change, the tuple $(g_i)_i$ of elements of $G$ acting on the agents' frames.",
          r"$\mathfrak A_{\mathrm{pop}},\mathfrak A_{\mathrm{prot}}$ & Population and protocol observable algebras. A distinction is gauge relative to an algebra when no element separates it at any time along the declared dynamics.")),
    (rows(r"$\Pi_o,Z_o$ & Posterior and evidence of a fixed normalized model at record value $o$."),
     rows(r"$L_o,\Pi_o,Z_o$ & Likelihood, posterior, and evidence of a fixed normalized model at record value $o$.")),
    (rows(r"$\mathfrak k,A^F$ & Collective orbit metric and a representative of the minimizing connection class."),
     rows(r"$\mathfrak k,A^F$ & Collective orbit metric and a representative of the minimizing connection class.",
          r"$\theta^{\mathrm{mech}},\Omega^{\mathrm{mech}}$ & Mechanical connection of the declared comparison data and its curvature.")),
    (rows(r"$\Theta^q_{ij},\Theta^s_{ij}$ & Invertible source-to-receiver interaction maps."),
     rows(r"$\Theta^q_{ij},\Theta^s_{ij}$ & Invertible source-to-receiver interaction maps.",
          r"$N_{ij,j}$ & Backtrack loop $\Theta_{ji}\Theta_{ij}$ at $j$ of a directed pair.")),
    (rows(r"$\Phi,\Phi^0$ & A flow of a construction and its null flow, with peer costs deleted.",
          r"$H(x),P(x)$ & Membership history with event times and the partition sequence without them; $H^0,P^0$ along the null flow.",
          r"(F1)--(F6) & Clauses of the formation standard; (F5) is interaction necessity, $P\neq P^0$ on an open set."),
     rows(r"$\Phi$ & The flow of a construction, on its declared chart or invariant family.",
          r"$\Phi^0_{\mathrm{lp}},\Phi^0_{\mathrm{rn}}$ & Label-preserving and renormalized null flows: every peer cost is deleted, with the peer labels and priors kept, or with the deleted labels removed so that a self source has weight one.",
          r"$\beta^0_{ii}$ & Self weight of a Gibbs row along the label-preserving null, below one when the self prior is below one.",
          r"$\Phi^{\mathrm{fr}}$ & Frozen-environment control: nothing is deleted, and the laws of the other agents are held at their initial values in every force.",
          r"$H(x),P(x)$ & Membership history with event times and the partition sequence without them, modulo exchange of consecutive merges of disjoint agent sets; $H^0_{\mathrm{lp}},P^0_{\mathrm{lp}}$, $H^0_{\mathrm{rn}},P^0_{\mathrm{rn}}$, and $H^{\mathrm{fr}},P^{\mathrm{fr}}$ along the controls.",
          r"$S_c,\mathsf C$ & A declared selector family and its constant set, fixed before the flows are analyzed.",
          r"$A_I,B_I,R_I$ & In the formation standard, the parent protocol class, the child-level probe class, and the retention of a block $I$.",
          r"$\Omega_I^{\mathrm{reach}}$ & Reachable post-handoff set of a block $I$.",
          r"$\mathcal C,\mathcal U$ & In the counterfactual clause, the declared counterfactual handoff class and the class of initial configurations that generates it.",
          r"(F1)--(F6) & Clauses of the formation standard; (F5) is interaction necessity, $P\neq P^0$ pointwise on a nonempty open admitted set, against both nulls and for every constant in $\mathsf C$.")),
    (rows(r"$n_I,A_I,B_I$ & Primitive count and belief/model shape sums of a retained block in that example."),
     rows(r"$n_I,A_I,B_I$ & Primitive count and belief/model shape sums of a retained block in that example.",
          r"$\beta_i$ & In that example, the self weight of agent $i$ along the label-preserving null.")),
    (rows(r"$\mathcal T,\Delta$ & Encoder kernel and predictive decoder in the coding construction."),
     rows(r"$\mathcal T,\Delta$ & Encoder kernel and predictive decoder in the coding construction.",
          r"$\alpha,\alpha_+$ & Dirichlet parameter of the record-process example and its total concentration.",
          r"$\bar\sigma$ & Magnetization of the Ising example, solving $\bar\sigma=\tanh(J\bar\sigma)$.",
          r"$\lambda_t,F_0^t,F_1^t$ & Joint private-record posterior after $t$ messages and its product-tilt factors, $\lambda_t\propto F_0^tF_1^t\lambda_0$.",
          r"$\Sigma^D_t$ & Covariance of the private records in the Gaussian closure of the interacting protocol.",
          r"$\ell_\psi,\mathcal X_{\mathrm{rich}}$ & Score of the common $\psi$ rotation and the rich preparation class of the collective construction.",
          r"$\mathcal R,\mathcal R_{\mathrm{glob}}$ & Regular critical set of the finite contextual flow and its subset on which every transport is the unique global minimizer.",
          r"$X,H_X,Q_X,P_X$ & Collective message matrix, $H_X=XX^T$, the right inverse $Q_X=X^TH_X^{-1}$, and the projector $P_X=Q_XX$; $H_X^{-1}$ is the induced fiber metric.",
          r"$C(c',c),T(c',c)$ & Cross-context map $X(c')Q_X(c)$ and the direct rotation between message row spaces.",
          r"$\bar\Theta_{ij},R_{ij}$ & A link and its reciprocal part in reference coordinates, $U_i^{-1}\Theta_{ij}U_j$ and $U_i^{-1}\Omega_{ij}U_j$.",
          r"$J_Z,\mathcal H$ & Infinitesimal action on contextual link data and the locked form of the alignment selector.",
          r"$\bar T,\bar\theta,\hat A$ & Code and dictionary in reference coordinates and the effective code $\hat A=\bar T\bar\theta^{-1}$ that messages identify.",
          r"$O_0,Q_D$ & Noiseless gain of the Gaussian decoder and the projection onto the directions that a code reads.")),
    (r"The letter $B$ denotes a connection representative $B_i$, block shape sums $B_I$, and native rows $B^q_{IJ}$, each in its own construction.",
     r"The letter $B$ denotes a connection representative $B_i$, block shape sums $B_I$ in the Wishart construction, the child-level probe class $B_I$ of the formation standard, and native rows $B^q_{IJ}$, each in its own construction; $A_I$ is likewise a shape sum in the Wishart construction and a protocol class in the standard, and $\mathcal C$ denotes a counterfactual class only inside the counterfactual clause."),
]
for old, new in E:
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(t.replace("\n", "\r\n"))
print("ok", len(E))
