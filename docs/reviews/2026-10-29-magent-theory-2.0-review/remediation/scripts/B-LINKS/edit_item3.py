p = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/Item_3_curvature_hierarchy.tex"
s = open(p, encoding="utf-8").read()
reps = [
(r"""\chapter*{Integration note}
\addcontentsline{toc}{chapter}{Integration note}
The corrected results in this working document are integrated into the main manuscript's chapter on curvature costs and supplied hierarchies, with an explanatory chapter in the companion. This standalone account retains the definitions and decisive proofs.""",
 r"""\chapter*{Status}
\addcontentsline{toc}{chapter}{Status}
This working document is superseded by the main manuscript's chapter on curvature costs and supplied hierarchies and its companion chapter, which are authoritative for every statement and for every label that this document shares with them. This document now states that the weak-anchor limit of the induced action is a statement inside the declared tangent model, that the omitted log-determinant dominates in that limit, and that the small-noise limit at fixed anchors gives the same cohomological form consistently; the choice of the regime that defines the cohomological part is open. Later refinements of the chapter are not repeated here; they include the general rank formula as the dimension of the first cohomology of the code sheaf, the condensation criterion for cycle spanning, the exact form of stationary loop products, the measure-theoretic hypotheses of the general evidence remark, the Lebesgue reference measure and explicit bridge normalizer of the rigid restriction, the convergence of the normalized soft Gaussian law, the identity requirement on the connecting paths of the flat-wall proposition and its pairing-graph criterion, and the attributions to discrete Hodge theory and spanning-forest determinants. This standalone account retains the definitions and decisive proofs."""),
(r"""The tangent approximation drops the product $\omega_e(y_i-c_i)$ in $\bar T_e(e^{-\omega_e}y_i-y_j)$.""",
 r"""The tangent approximation drops the terms of second and higher order in $(\omega_e,y_i-c_i)$ in $\bar T_e(e^{-\omega_e}y_i-y_j)$, among them the product $\omega_e(y_i-c_i)$; it is consistent only while the posterior spread of $y_i-c_i$ is small compared with $|c_i|$."""),
(r"""\citep{HansenGhrist2021}.""" if False else r"""is $H^1=H/\operatorname{im}\delta$ \citep{HansenGhrist2021}.""",
 r"""is $H^1=H/\operatorname{im}\delta$ \citep[Section~2.2]{HansenGhrist2019}; cellular sheaves of this kind are used for discourse by \citet{HansenGhrist2021}."""),
(r"""If $P=\epsilon\bar P$ with fixed positive-definite $\bar P$ and $\epsilon\downarrow0$, its record precision tends to""",
 r"""Within this tangent model, if $P=\epsilon\bar P$ with fixed positive-definite $\bar P$ and $\epsilon\downarrow0$, its record precision tends to"""),
(r"""For finite anchors, the shift by $\delta m$ in \cref{it3:affine-action} must also be retained.""",
 r"""For finite anchors, the shift by $\delta m$ in \cref{it3:affine-action} must also be retained.

In the weak-anchor limit the common content shift, which spans $\ker\delta$ for injective codes on a connected graph, is constrained by no record; its spread is of order $\epsilon^{-1/2}$, so the dropped product $\omega_e(y_i-c_i)$ is not small there. At fixed dictionaries the record $\bar T_e(e^{-\omega_e}y_i-y_j)+\varepsilon_e$ that keeps it is still linear in $y$, so the evidence is the formula of \cref{it3:open-fluctuation} with a dictionary-dependent content precision $J(\omega)$. A Schur complement on the common mode shows that the dictionary dependence of $\frac12\log\det J(\omega)$ is of order $\epsilon^{-1}$, larger than the bounded tangent quadratic part, and that it charges directions with $\omega c=0$. If instead $R=\rho\bar R$ with $\rho\downarrow0$ at fixed anchors, $\rho$ times the record precision tends to $\mathcal V^T\Pi_H(\bar R)\mathcal V$ by the same proof with the two terms of $C_P$ exchanged, and the determinant term is smaller than the tangent term by a factor of the order of the content prior variance divided by $|c|^2$. The cohomological part is therefore consistent as the leading term in the window noise variance $\ll$ content prior variance $\ll|c|^2$."""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok", len(reps))
