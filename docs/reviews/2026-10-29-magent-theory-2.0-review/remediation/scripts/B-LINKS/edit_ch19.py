p = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/19_curvature_hierarchy.tex"
s = open(p, encoding="utf-8").read()
reps = [
# 1 intro
(r"""In a Gaussian tangent model its quadratic part measures coded reinterpretations that changes of contents cannot absorb.""",
 r"""In a Gaussian tangent model its quadratic part charges coded reinterpretations; in the weak-anchor or small-noise limit it charges exactly those that changes of contents cannot absorb."""),
# 2 tangent approximation and carried metric
(r"""The matrix anchor is isotropic for the declared reference inner product. Coordinate changes carry this metric with the other constitutive data. There are no grounded dictionary factors and no explicit loop factor: the loop weight is $\lambda=0$. The tangent approximation drops the product $\omega_e(y_i-c_i)$ in $\bar T_e(e^{-\omega_e}y_i-y_j)$. Its operating contents remain fixed.""",
 r"""The matrix anchor is isotropic for the declared reference inner product. Coordinate changes carry this metric with the other constitutive data, so the quantitative statements below hold in the reference presentation in which the anchor is isotropic, and are invariant under the reference changes that preserve it, as in \cref{it1:tangent-model}. There are no grounded dictionary factors and no explicit loop factor: the loop weight is $\lambda=0$. The tangent approximation drops the terms of second and higher order in $(\omega_e,y_i-c_i)$ in $\bar T_e(e^{-\omega_e}y_i-y_j)$, among them the product $\omega_e(y_i-c_i)$. Its operating contents remain fixed. The approximation is consistent only while the posterior spread of $y_i-c_i$ is small compared with $|c_i|$; \cref{it3:weak-anchor-determinant} records what happens when it is not."""),
# 3 H^1 attribution
(r"""Its first cohomology on the graph, which has no two-cells, is $H^1=H/\operatorname{im}\delta$ \citep{HansenGhrist2021}.""",
 r"""Its first cohomology on the graph, which has no two-cells, is $H^1=H/\operatorname{im}\delta$ \citep[Section~2.2]{HansenGhrist2019}. Cellular sheaves of this kind are used for discourse by \citet{HansenGhrist2021}; that paper works with $H^0$."""),
# 4 general induced hypotheses
(r"""Let $\pi(d\omega)$ be a dictionary prior, let $p(dy)$ be a content law on a standard Borel space, and let $\ell(o\mid y,\omega)$ be a chosen density of the record kernel at the registered record $o$. Suppose""",
 r"""Let $\pi(d\omega)$ be a dictionary prior and $p(dy)$ a content law on standard Borel spaces, with contents and dictionaries independent a priori, so that the joint prior is $p\otimes\pi$. Let $\lambda_o$ be a sigma-finite measure on the record space that dominates every record law, and let $\ell(o\mid y,\omega)$ be a jointly measurable version of the record density. Suppose, for the registered record $o$,"""),
(r"""Disintegration of the joint density gives the posterior""",
 r"""Bayes' theorem for this dominated model gives, for $\lambda_o$-almost every record, the posterior"""),
(r"""Where $0<Z(\omega)<\infty$, the contents induce the action $-\log Z(\omega)$. This identity requires no Gaussian family.""",
 r"""Where $0<Z(\omega)<\infty$, the contents induce the action $-\log Z(\omega)$. At one registered record the formula defines a version of the posterior: two jointly measurable versions of the density that differ on a null set of records can give different posteriors at that record. The induced action is therefore defined for a fixed version, canonically the version continuous in $o$ when it exists, as in \cref{it3:induced}. This identity requires no Gaussian family."""),
# 5 induced theorem: weak-anchor limit inside tangent model; small-noise statement
(r"""If $P=\epsilon\bar P$ with fixed positive-definite $\bar P$ and $\epsilon\downarrow0$, its record precision tends to $\mathcal V^T\Pi_H\mathcal V$, where""",
 r"""Within this tangent model, if $P=\epsilon\bar P$ with fixed positive-definite $\bar P$ and $\epsilon\downarrow0$, its record precision tends to $\mathcal V^T\Pi_H\mathcal V$, where"""),
(r"""The superscript $+$ denotes the Moore--Penrose inverse. For every $\omega$,""",
 r"""The superscript $+$ denotes the Moore--Penrose inverse. If instead $R=\rho\bar R$ with fixed positive-definite $\bar R$, fixed $P$ and $\rho\downarrow0$, then $\rho$ times the record precision tends to $\mathcal V^T\Pi_H(\bar R)\mathcal V$, with $\Pi_H(\bar R)$ formed from $\bar R$ in place of $R$. For every $\omega$,"""),
(r"""This is the squared quotient norm of the class of $\mathcal V\omega$ in $H^1$. It vanishes exactly when
$\bar T_e\omega_ec_i=\bar T_e(a_i-a_j)$ for one $a\in Y$ on every edge.""",
 r"""This is the squared quotient norm of the class of $\mathcal V\omega$ in $H^1$. It vanishes exactly when
$\bar T_e\omega_ec_i=\bar T_e(a_i-a_j)$ for one $a\in Y$ on every edge. If every receiver has $c_i\neq0$, the rank of the limiting quadratic form on $X$ is
\begin{equation}
\sum_e\operatorname{rank}\bar T_e-K|\mathsf V|+\dim H^0,
\label{it3:rank-general}
\end{equation}
where $H^0=\ker\delta$; for codes of full row rank this is $\dim H^1$."""),
(r"""Suppose in addition that every code is injective, the underlying undirected multigraph is connected, all operating contents equal one $c\ne0$, and the chains of the declared directed closed walks span its real cycle space. Opposite attending edges count as separate edges of that multigraph. Then the limiting quadratic part vanishes exactly when
$(B\omega)_\gamma c=0$ for every declared walk, and its rank on $X$ is
$K(|\mathsf E|-|\mathsf V|+1)$. For positive $P$, an absorbable reinterpretation obeys""",
 r"""If every code is injective and the underlying undirected multigraph is connected, the rank is $K(|\mathsf E|-|\mathsf V|+1)$; within the full-row-rank class of \cref{it2:model} injective codes are square and invertible. Suppose in addition that all operating contents equal one $c\ne0$ and the chains of the declared directed closed walks span the real cycle space of that multigraph. Opposite attending edges count as separate edges of it. Then the limiting quadratic part vanishes exactly when
$(B\omega)_\gamma c=0$ for every declared walk. For positive $P$, an absorbable reinterpretation obeys"""),
(r"""On the range of $A$ the middle factor tends to zero; on its orthogonal complement it is identity. The limit is \cref{it3:projection}, the residual quadratic form for weighted least squares. Minimizing over $a$ proves \cref{it3:quotient} and its zero criterion.""",
 r"""On the range of $A$ the middle factor tends to zero; on its orthogonal complement it is identity. The limit is \cref{it3:projection}, the residual quadratic form for weighted least squares. For the small-noise limit write $\bar A=\bar R^{-1/2}\delta P^{-1/2}$; then $\rho C_P^{-1}=\bar R^{-1/2}\rho(\rho I+\bar A\bar A^T)^{-1}\bar R^{-1/2}$, and $\rho(\rho I+\bar A\bar A^T)^{-1}$ tends to the orthogonal projection onto the complement of the range of $\bar A$, which is the same as the range of $\bar R^{-1/2}\delta$. Minimizing over $a$ proves \cref{it3:quotient} and its zero criterion. The kernel of $\Pi_H$ is $\operatorname{im}\delta$. When every $c_i\neq0$, $\omega_e\mapsto\omega_ec_i$ is onto $\R^K$, so $\operatorname{im}\mathcal V=\bigoplus_e\operatorname{im}\bar T_e\supseteq\operatorname{im}\delta$, and the rank of $\mathcal V^T\Pi_H\mathcal V$ is $\dim\operatorname{im}\mathcal V-\dim\operatorname{im}\delta$, which is \cref{it3:rank-general}."""),
(r"""For injective codes that criterion becomes $\omega_ec=a_i-a_j$. A vector-valued edge cochain has this form exactly when it pairs to zero with every cycle. To see the converse directly, choose a spanning tree and define $a_i$ by signed sums along its paths from a root. Each remaining edge closes a fundamental cycle, whose zero sum gives its required difference. The cycle-spanning hypothesis makes the declared directed-walk tests sufficient for these signed cycle tests; it does not identify an opposite attending edge with geometric reversal. Since $\omega_e\mapsto\omega_ec$ is onto $\R^K$ for $c\ne0$, the rank is the edge-cochain dimension $K|\mathsf E|$ minus the coboundary dimension $K(|\mathsf V|-1)$.""",
 r"""For injective codes and a common $c$ that criterion becomes $\omega_ec=a_i-a_j$. A vector-valued edge cochain has this form exactly when it pairs to zero with every cycle. To see the converse directly, choose a spanning tree and define $a_i$ by signed sums along its paths from a root. Each remaining edge closes a fundamental cycle, whose zero sum gives its required difference. The cycle-spanning hypothesis makes the declared directed-walk tests sufficient for these signed cycle tests; it does not identify an opposite attending edge with geometric reversal. For injective codes on a connected graph $\dim H^0=K$ and $\operatorname{rank}\bar T_e=K$, so \cref{it3:rank-general} is $K|\mathsf E|-K(|\mathsf V|-1)$."""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok", len(reps))
