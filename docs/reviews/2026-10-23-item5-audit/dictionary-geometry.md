# Item 5: dictionary and connection geometry audit

Reviewed October 23, 2026. Scope: the full standalone `docs/derivations/2026-10-23-link-base-transport/latex/main.tex`, with particular attention to the rectangular dictionary, transformation and curvature formulas, restriction to a declared structure group, patching, and the path limit. This was an analytical review. No numerical script, GPU work, Lean, or adjudication panel was run. This report is the only file changed by this investigator.

No differential-geometry findings.

The results below are supported under the hypotheses actually stated in the note. This verdict does not assert dynamic selection of the dictionary, preservation of rank, continuation through singular strata, equality with the manuscript's Fisher selector, or derivation of the contextual base.

## Rectangular dictionary and induced geometry

The full-row-rank assumption at lines 133–135 supplies exactly the algebra needed at lines 139–177. Write $H=XX^T$, $Q=X^TH^{-1}$, and $P=QX$. Then $XQ=I$, $P^2=P=P^T$, and, under $X'=h^{-1}X$,

\[
H'=h^{-1}Hh^{-T},\qquad Q'=Qh,\qquad P'=P.
\]

Consequently the map $C(c',c)=X(c')Q(c)$ has identity at equal arguments and the claimed endpoint law. Its determinant remains nonzero on a neighborhood of each diagonal point. The derivative in its first argument is $dXQ$, so the convention $C=I-a(\delta c)+O(|\delta c|^2)$ gives $a=-dXQ$. Direct differentiation yields

\[
-d(h^{-1}X)Qh=h^{-1}ah+h^{-1}dh.
\]

This is an affine connection transformation, including the required derivative of the reference change. It is stronger than the automatic passive covariance of dressed pointwise links. The distinction agrees with the live manuscript's `chapters/01_foundations.tex:76` and `chapters/02_interactions.tex:65`.

The curvature identities at lines 228–243 are correct with matrix multiplication inside wedge products. Indeed, differentiating $P=QX$ and $XQ=I$ gives

\[
dP Q=dQ+Q dX Q=(I-P)dQ.
\]

Thus

\[
F=dX\wedge dQ+dXQ\wedge dXQ
=dX\wedge(I-P)dQ
=dX\wedge(I-P)dX^TH^{-1}.
\]

The last step uses $(I-P)X^T=0$. Also $aH=-dX X^T$ and $Ha^T=-X dX^T$, which prove $dH+aH+Ha^T=0$. Equivalently the connection preserves the positive fiber metric $H^{-1}$. Taking the trace after multiplication by $H^{-1}$ gives $\operatorname{tr}a=-\tfrac12d\log\det H$, and hence $\operatorname{tr}F=0$. These are restrictions of this particular selector; the note does not claim that its curvature is arbitrary in $\mathfrak{gl}(K)$.

The induced-connection interpretation can be exhibited without identifying this fiber metric with a Fisher or affine-invariant metric on the SPD cone. The right inverse $Q:E\to\mathcal U\times\mathbb R^r$ embeds $E$ as the subbundle $\operatorname{im}P$; its pulled-back Euclidean metric is $Q^TQ=H^{-1}$. Projecting the ambient derivative gives

\[
X P d(Qv)=X dQ v+dv=dv+a v.
\]

In an orthonormal frame for that subbundle this is the classical Stiefel connection $U^TdU$. The cited primary source gives the complex version $U^\dagger dU$, with $U^\dagger U=I$, in Massamba and Thompson 2003, equations (1.1)–(1.2) and (3.2). This supports the note's attribution to induced subbundle geometry, without a universality assertion for the chosen law-derived columns. [Primary source](https://arxiv.org/pdf/math/0311198).

The explicit example at lines 247–268 also has the stated curvature. For $X=[I_2,v]$, $v=(x,y)^T$, $s=1+x^2+y^2$, one obtains $a_x=-e_1v^T/s$ and $a_y=-e_2v^T/s$. Differentiating and adding their commutator gives

\[
F_{xy}=\frac{1}{s^2}
\begin{pmatrix}-xy&1+x^2\\-(1+y^2)&xy\end{pmatrix}.
\]

At the origin this is the nonzero unit rotation generator. Pullback along $(x,y)\mapsto(x,1+xy)$ multiplies its two-form by Jacobian determinant $x$, proving the second example's stated nonvanishing region. The non-Gaussian realization needs precisely the finite first moments and smooth contextual dependence stated at lines 113 and 133; normalization alone would not supply them.

## Reversal, paths, and global patching

The reciprocalization at lines 188–204 preserves the first jet. The identity $X dQ=-dX Q$ makes the linear terms in $C(c,c')C(c',c)$ cancel. Its spectrum is consequently in the principal-root domain after sufficient local restriction, and $N^{-1/2}=I+O(|\delta c|^2)$. Reversal and endpoint covariance follow from similarity covariance of this branch. The note correctly distinguishes the cross-context reconstruction defect from the original agent backtrack data.

The path convergence proof at lines 208–221 is sufficient for the stated piecewise $C^2$ paths and $C^1$ connection. On each smooth segment, both the discrete factor and exact short propagator have the same first-order term and uniformly bounded second-order remainders. Telescoping yields a bound proportional to $\sum_k\Delta t_k^2\leq\max_k\Delta t_k$. Including path corners among partition points is enough for the displayed proof; partitions not including a corner give the same limit because only finitely many crossing intervals occur and their total length tends to zero. Exact finite transport is defined by the ODE, not by an individual unrefined $C$ or $T$.

The global statement at line 180 is correctly conditional on a supplied global surjection from a fixed trivial message bundle. A rectangular surjection need not trivialize $E$: its right inverse realizes $E$ as a possibly nontrivial subbundle of the trivial message bundle. The overlap law proved above glues its projected connection. A square surjection is an isomorphism and does trivialize $E$. Transition maps must be inserted when a path changes local presentations, as line 220 states. Local trace exactness or zero curvature does not remove possible global holonomy; the note makes no such inference.

## Declared structure group and manuscript compatibility

The constrained selector at lines 279–305 is valid for reference changes valued in the declared group $G$, with the metric transformed as $M'=h^TMh$. Its Hessian is the positive quadratic form $\xi\mapsto\operatorname{tr}(M\xi XX^T\xi^T)$. Full row rank makes its kernel zero on every subalgebra, so the minimizer exists uniquely, depends linearly on the contextual direction, and is $C^1$ with the stated regularity. The affine candidate $h^{-1}\xi h+h^{-1}dh(Y)$ lies in the same Lie algebra and preserves the residual norm, proving the covariance of the minimizer. For the full general linear algebra, the normal equation reduces to $(dX+aX)X^T=0$, recovering the rectangular formula independently of $M$.

For a proper subgroup, metric compatibility and the explicit curvature formula of the unconstrained selector do not automatically transfer. The note correctly avoids asserting that they do. Integrating the resulting Lie-algebra-valued form produces transport in $G$, while the unconstrained finite cross-map need not itself be $G$-valued. The summed two-sector condition at lines 300–305 is the correct injectivity condition for selecting one principal connection in the presence of possibly nonfaithful sector representations. It preserves the single-group convention in `AUTHORING.md:19`.

The message-coordinate extension at lines 381–386 is also consistent, provided the stated message metric transforms with the coordinates, so that the metric right inverse obeys $Q'=sQ$. The combination $dX-X\Gamma$ then transforms by right multiplication by $s^{-1}$. Its connection is intrinsic to the supplied message connection; resetting that connection to zero in a varying coordinate system changes the constitutive data.

The live manuscript's collective Fisher normal equation in `chapters/02_geometry.tex:205` minimizes the actual declared statistical comparison metric. The dictionary selector instead uses its declared message metric and, for a group restriction, its declared fiber metric. Their common normal-equation form establishes no equality. Lines 388–390 preserve this distinction and should survive integration. The inherited scope from `appendices/base_connection.tex` can therefore be strengthened to conditional construction of displacement links and their connection, while dynamic selection, rank preservation, singular continuation, and geometry of the base remain open.

The two-flat-connection curvature formula at lines 81–86 agrees with the source-continuity record `Research/manuscripts/pifb2/2026-10-09-kaluza-klein-pullback/06-collective-curvature.md`, section 1.2. Substituting $a=a_2+\lambda\Delta$ and $D^{a_2}\Delta=-\Delta\wedge\Delta$ independently gives $F=d\lambda\wedge\Delta-\lambda(1-\lambda)\Delta\wedge\Delta$. This carries forward the existing construction and does not import that record's separate soldering, torsion, Lorentzian-metric, or dynamics claims.
