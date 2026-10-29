r"""Exact string replacements in chapter 16 (each old string must occur exactly once)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
R = [
    # opening: objective statement
    (r"They supply a conditional construction for \cref{open:conn-derived}; they add no evolution equation or formation result to the contextual-flow extension in \cref{ch:contextual-coupling}.",
     r"They supply a conditional construction for \cref{open:conn-derived}; they add no evolution equation or formation result to the contextual-flow extension in \cref{ch:contextual-coupling}. The selectors are kinematic: they take supplied fields as input and belong to none of the three objectives of \cref{ch:dynamics}."),
    # reference links renamed
    (r""" \Theta_{ij}=U_i K_{ij}U_j^{-1}:E_j\longrightarrow E_i,
 \qquad K_{ij}'=h^{-1}K_{ij}h.""",
     r""" \Theta_{ij}=U_i \bar\Theta_{ij}U_j^{-1}:E_j\longrightarrow E_i,
 \qquad \bar\Theta_{ij}'=h^{-1}\bar\Theta_{ij}h."""),
    (r"For example, $K_{ij}=\exp(L_{ij})$, where", r"For example, $\bar\Theta_{ij}=\exp(L_{ij})$, where"),
    # reciprocal split
    (r"Suppose each backtrack $N_{ij,j}=\Theta_{ji}\Theta_{ij}$ has spectrum disjoint from $(-\infty,0]$. The ordered-pair subscripts distinguish loops from different neighbors. Define",
     r"Suppose each backtrack $N_{ij,j}=\Theta_{ji}\Theta_{ij}$ has spectrum disjoint from $(-\infty,0]$. The pair $ij$ names the interacting pair and the last subscript the agent at which the loop is based, so $N_{ij,j}$ and $N_{ij,i}$ are the two loops of one pair. The following is the split of \cref{prop:int-reciprocal}, written with these subscripts. Define"),
    (r""" R_{ij}=U_i^{-1}\Omega_{ij}U_j,\qquad
 K_{ij}=U_i^{-1}\Theta_{ij}U_j.""",
     r""" R_{ij}=U_i^{-1}\Omega_{ij}U_j,\qquad
 \bar\Theta_{ij}=U_i^{-1}\Theta_{ij}U_j."""),
    (r"\{K_{ij}\bar \mu_j\}_{(i,j)}\bigr]:H\longrightarrow V.", r"\{\bar\Theta_{ij}\bar \mu_j\}_{(i,j)}\bigr]:H\longrightarrow V."),
    (r"Thus the construction is reference consistent when its original link and interface rules are reference consistent.",
     r"Thus the construction is reference consistent when its original link and interface rules are reference consistent. This is naturality: the rule is built equivariantly from the laws and from declared data that are re-expressed with the reference. It is not an active symmetry of the model with those data held fixed, which fixed receiving axes or labels generally break."),
    # global remark becomes proposition
    (r"The displacement maps are locally invertible, but the connection can be global on a nontrivial bundle if a global surjection $\mathcal C\times\R^r\to E$ with these data is supplied. When $r>K$, such a surjection need not trivialize $E$: its projector describes a moving subspace of the message space. A global square frame is the more restrictive case $r=K$.",
     r"""\propositionheading{Global message surjections}{ibt:global}
Let $E\to\mathcal C$ be a rank-$K$ vector bundle and $X:\mathcal C\times\R^r\to E$ a $C^2$ fiberwise surjective bundle map whose local forms obey \cref{ibt:eq:X-law}. Put $W(c)=\operatorname{im}X(c)^T\subset\R^r$. Then the local forms $a=-dXQ_X$ define a global $C^1$ connection, with covariant derivative
\begin{equation}
 \nabla\sigma=Xd(Q_X\sigma).
 \label{ibt:eq:global-nabla}
\end{equation}
It is the connection induced on the moving subspace $W$ by orthogonal projection of the trivial connection of $\R^r$, transported to $E$ by the isomorphism $X|_W$. The maps $C(c',c)=X(c')Q_X(c)$ are defined for all pairs, equal the identity on the diagonal, and are invertible near it. The map $c\mapsto W(c)$ into the Grassmannian of $K$-planes in $\R^r$ classifies $E$; in particular $r=K$ forces $E$ to be trivial.
\begin{proof}
The affine law follows from the identity in the proof of \cref{ibt:displacement}, so the local forms glue. Since $XQ_X=I$, $XdQ_X=-dXQ_X$, and therefore $Xd(Q_X\sigma)=d\sigma-dXQ_X\sigma=D^a\sigma$. Because $Q_X$ is an isomorphism onto $W$ with inverse $X|_W$, and $X$ annihilates $W^\perp$, this is $X|_W$ applied to the orthogonal projection onto $W$ of the derivative of the $W$-valued section $Q_X\sigma$. The statements about $C$ were proved in \cref{ibt:displacement}. Finally $X|_{W(c)}:W(c)\to E_c$ is an isomorphism depending continuously on $c$, so $E$ is isomorphic to the pullback of the tautological bundle along $c\mapsto W(c)$; for $r=K$ that pullback is trivial.
\end{proof}

When $r>K$, such a surjection need not trivialize $E$. For instance, on the tangent bundle of the unit sphere $S^2\subset\R^3$, the tangential projections of the three coordinate vectors give a surjection with $r=3$; the induced connection is the Levi-Civita connection, and the Euler number $2$ shows that this bundle is nontrivial. A global square frame is the more restrictive case $r=K$."""),
    # Wilson / classical
    (r"These failures do not invalidate its first jet as a connection, but matter before calling a finite map parallel transport. Discrete connections and their infinitesimal limits are classical geometric constructions~\citep{LeokMarsdenWeinstein2005}.",
     r"These failures do not invalidate its first jet as a connection, but matter before calling a finite map parallel transport. Finite maps on links whose ordered products refine to path-ordered transport are the link variables of lattice gauge theory and their continuum limit \citep{Wilson1974}; a discrete-connection formalism is developed in \citet{LeokMarsdenWeinstein2005}."),
    # after ibt:reciprocal proof: Grassmannian reading
    (r"""Its root is $I+O(|\delta c|^2)$, proving the expansion.
\end{proof}
""",
     r"""Its root is $I+O(|\delta c|^2)$, proving the expansion.
\end{proof}

\propositionheading{Principal angles and the reciprocal map}{ibt:grassmann}
Let $W(c)=\operatorname{im}X(c)^T$ and let $\theta_1,\dots,\theta_K\in[0,\pi/2]$ be the principal angles between $W(c)$ and $W(c')$. Then the eigenvalues of $N(c',c)$ are $\cos^2\theta_k$. Wherever $C(c',c)$ is invertible, that is, wherever every $\theta_k<\pi/2$, the operator $N(c',c)$ is similar to a positive operator, $T(c',c)$ is defined without further shrinking, and $T(c',c)$ is an isometry from $(E_c,H_X(c)^{-1})$ to $(E_{c'},H_X(c')^{-1})$. Read through these isometries, $T(c',c)$ is the direct rotation of $W(c)$ onto $W(c')$, which equals parallel transport of the connection \cref{ibt:eq:global-nabla} of the moving subspace along the Grassmannian geodesic from $W(c)$ to $W(c')$.
\begin{proof}
Since $Q_X^TQ_X=H_X^{-1}$, the map $\iota_c=Q_X(c)$ is an isometry from $(E_c,H_X(c)^{-1})$ onto $W(c)$, with inverse $X(c)|_{W(c)}$, and $X(c')$ vanishes on $W(c')^\perp$. Hence $C(c',c)=\iota_{c'}^{-1}\Pi'\iota_c$, where $\Pi'$ is the orthogonal projection onto $W(c')$ restricted to $W(c)$, and similarly $C(c,c')=\iota_c^{-1}\Pi\iota_{c'}$. Thus $N(c',c)$ is conjugate to $\Pi\Pi'$ restricted to $W(c)$, a positive semidefinite self-adjoint operator with eigenvalues $\cos^2\theta_k$ \citep{BjorckGolub1973}; it is invertible exactly when $\Pi'$ is injective on $W(c)$. The map $T=CN^{-1/2}$ is then conjugate to the orthogonal polar factor of $\Pi'$, an isometry of $W(c)$ onto $W(c')$, which is the direct rotation \citep{DavisKahan1970}. Choose principal vectors $w_k\in W(c)$ and $w'_k\in W(c')$ with $\langle w_j,w'_k\rangle=\delta_{jk}\cos\theta_k$, and for $\theta_k>0$ put $n_k=(w'_k-\cos\theta_k w_k)/\sin\theta_k$, an orthonormal family in $W(c)^\perp$. Let $\Xi$ be the skew map with $\Xi w_k=\theta_kn_k$, $\Xi n_k=-\theta_kw_k$, and zero on the remaining complement. The curve $W(t)=\exp(t\Xi)W(c)$ is a Grassmannian geodesic from $W(c)$ to $W(c')$, and the vectors $\exp(t\Xi)w_k=\cos(t\theta_k)w_k+\sin(t\theta_k)n_k$ have derivatives orthogonal to $W(t)$, so they are parallel for the projected connection. At $t=1$ they equal $w'_k$, and the polar factor of $\Pi'$ also sends $w_k$ to $w'_k$, because $\Pi'w_k=\cos\theta_kw'_k$.
\end{proof}

The Grassmannian geodesic need not be the image of any path in the base, so $T(c',c)$ is exact transport along a path that the base may not contain; this is the precise sense of the remark below that a finite endpoint rule contains additional higher-order data. Because $T(c,c')=T(c',c)^{-1}$, ordered products of $T$ form a symmetric one-step rule, and for $C^3$ data their local errors cancel at second order, whereas products of $C$ are first-order accurate.
"""),
    # ibt:path corners
    (r"Include the finitely many corners among partition points. On each smooth segment the exact short propagator and discrete factor equal $I-a(\dot\gamma)\Delta t+O(\Delta t^2)$ uniformly on the compact path image. Their norms are bounded by $1+M\Delta t$. Telescoping the product difference bounds it by $C\exp(2M)\sum_k\Delta t_k^2$, which tends to zero because $\sum_k\Delta t_k^2\leq\max_k\Delta t_k$. This proves convergence for both discrete factors.",
     r"On each step contained in one smooth segment the exact short propagator and discrete factor equal $I-a(\dot\gamma)\Delta t+O(\Delta t^2)$ uniformly on the compact path image. Their norms are bounded by $1+L\Delta t$. Telescoping the product difference bounds the contribution of these steps by $c_0\exp(2L)\sum_k\Delta t_k^2$, which tends to zero because $\sum_k\Delta t_k^2\leq\max_k\Delta t_k$. A step that straddles one of the finitely many corners has both factors equal to $I+O(\Delta t)$, so these steps add at most a constant times the number of corners times $\max_k\Delta t_k$. This proves convergence for both discrete factors and every partition sequence."),
    (r"The solution of $\dot S=Sa(\dot\gamma)$, $S(0)=I$, satisfies $\frac{d}{d t}(S\mathcal T)=0$, proving invertibility.",
     r"The solution of $\dot{\mathcal S}=\mathcal Sa(\dot\gamma)$, $\mathcal S(0)=I$, satisfies $\frac{d}{d t}(\mathcal S\mathcal T)=0$, proving invertibility."),
    # curvature section: corollary
    (r"""The connection preserves the fiber metric $H_X^{-1}$. When $r=K$, $P_X=I$ and the curvature vanishes. Redundant columns permit nonzero curvature.""",
     r"""The connection preserves the fiber metric $H_X^{-1}$. When $r=K$, $P_X=I$ and the curvature vanishes. Redundant columns permit nonzero curvature."""),
    (r"""This projector construction is an instance of induced subbundle geometry, related to the classical universal connection on a Stiefel bundle~\citep{Narasimhan1962,MassambaThompson2003}. No universality claim for this particular real message rule is made: \cref{ibt:eq:metric-compatibility} restricts its curvature. The choice here is how agent laws and directed links supply the columns.""",
     r"""\corollaryheading{Orthogonal holonomy of the rectangular selector}{ibt:orthogonal-holonomy}
Along every path, the transport of \cref{ibt:eq:path-ode} for \cref{ibt:eq:derived-a} is an isometry between the fiber metrics $H_X^{-1}$ at its endpoints, so every holonomy at $c_0$ lies in the orthogonal group of $H_X(c_0)^{-1}$. On a loop along which $E$ is trivialized, its determinant is $1$. Each curvature value $F^a(Y,Z)$ is skew for $H_X^{-1}$ and therefore has purely imaginary spectrum. The same holds when a message connection $\Gamma$ that preserves the message metric is used in \cref{ibt:eq:message-gamma}.
\begin{proof}
With $M=H_X^{-1}$, \cref{ibt:eq:metric-compatibility} gives $dM=Ma+a^TM$. For $\dot{\mathcal T}=-a(\dot\gamma)\mathcal T$, $\frac{d}{dt}(\mathcal T^TM\mathcal T)=\mathcal T^T(-a^TM+dM(\dot\gamma)-Ma)\mathcal T=0$. In a trivialization along the loop, $\det\mathcal T(1)=\exp(-\oint\Tr a)=\exp(\tfrac12\oint d\log\det H_X)=1$. Metric compatibility gives $MF^a+(F^a)^TM=0$, so $M^{1/2}F^aM^{-1/2}$ is skew. A skew $\Gamma$ in \cref{ibt:eq:message-gamma} gives $aH_X+H_Xa^T=-dH_X$ by the same computation.
\end{proof}

Under the structure group $\GL(K)$ the rectangular selector therefore reduces to the $H_X^{-1}$-orthonormal frames and never produces shear or dilation holonomy; noncompact holonomy requires a selector that does not preserve a fiber metric. The curvature formula also shows that $F^a$ depends on $dX$ only through $(I-P_X)dX^T$, which represents the derivative of $c\mapsto W(c)$; $F^a(c)=0$ wherever that map has rank below two.

This projector construction is an instance of induced subbundle geometry: by \cref{ibt:global} it is the pullback along $c\mapsto W(c)$ of the connection that the trivial bundle induces on the tautological subbundle of a Grassmannian, the setting of universal connections \citep{NarasimhanRamanan1961,NarasimhanRamanan1963,MassambaThompson2003}. By \cref{ibt:orthogonal-holonomy} every connection produced by this real rule preserves a fiber metric; whether every metric connection arises for some finite $r$ is the universal-connection question, which is not settled here for this rule. The choice here is how agent laws and directed links supply the columns."""),
    # witness scalar renamed
    (r""" v=\begin{pmatrix}x\\y\end{pmatrix},\qquad s=1+x^2+y^2.""", r""" v=\begin{pmatrix}x\\y\end{pmatrix},\qquad \chi=1+x^2+y^2."""),
    (r""" a&=-\frac{(d v)v^T}{s},\label{ibt:eq:witnessA}\\
 F^a&=\frac{1}{s^2}""", r""" a&=-\frac{(d v)v^T}{\chi},\label{ibt:eq:witnessA}\\
 F^a&=\frac{1}{\chi^2}"""),
    (r"""At the origin $a=0$ but $F^a_{xy}=\begin{pmatrix}0&1\\-1&0\end{pmatrix}\ne0$.""",
     r"""At the origin $a=0$ but $F^a_{xy}=\begin{pmatrix}0&1\\-1&0\end{pmatrix}\ne0$. Here $W(c)$ is the plane orthogonal to $(v,-1)$, so $c\mapsto W(c)$ is the inverse central projection onto the unit sphere, and the connection is the Levi-Civita connection of $S^2$ pulled back by it."""),
    (r""" K_{ij}=\begin{pmatrix}1&0\\y&1\end{pmatrix}
        =\exp(yE_{21}),\qquad K_{ji}=K_{ij}^{-1}.""",
     r""" \bar\Theta_{ij}=\begin{pmatrix}1&0\\y&1\end{pmatrix}
        =\exp(yE_{21}),\qquad \bar\Theta_{ji}=\bar\Theta_{ij}^{-1}."""),
    (r"The labels and receiving axes are declared interface data. No Gaussian closure or evolving distributional ansatz is used.",
     r"The labels, receiving axes, column selection, and message metric are declared data, and the law enters only through its mean: any law with mean $(x,1)^T$ gives the same connection. No Gaussian closure or evolving distributional ansatz is used."),
    # backtrack example replaced
    (r"""Backtracks can affect the connection while the reciprocal agent link stays fixed. In a scalar example take a receiver seed column $1$, source mean $1$, and $\Theta_{ij}=\Theta_{ji}=s(c)>0$. Then $\Omega_{ij}=1$ and $N_{ij,j}=s^2$. The raw linked column gives $X=[1,s]$ and
\begin{equation}
 a=-\frac{sd s}{1+s^2},
\end{equation}
whereas using only the reciprocal column gives $X=[1,1]$ and $a=0$. Both scalar connections are flat, but their transports differ for varying $s$. This proves a backtrack contribution can survive the selector, without asserting that every backtrack distinction is recoverable from one connection.""",
     r"""Backtracks can affect the gauge class of the connection while the reciprocal agent link stays fixed. Take receiving columns $e_1,e_2$, a source mean $(1,1)^T$ in reference coordinates, and $\Theta_{ij}=\Theta_{ji}=\operatorname{diag}(e^x,e^y)$. Then $N_{ij,j}=\operatorname{diag}(e^{2x},e^{2y})$, $S_{ij,j}=\operatorname{diag}(e^x,e^y)$, and $\Omega_{ij}=I$. The raw linked column $(e^x,e^y)^T$ gives the matrix \cref{ibt:eq:witnessX} with $v=(e^x,e^y)^T$, whose curvature is the pullback of \cref{ibt:eq:witnessF} by a map with Jacobian determinant $e^{x+y}\neq0$, hence nonzero everywhere; at the origin it equals $\frac19\begin{pmatrix}-1&2\\-2&1\end{pmatrix}dx\wedge dy$. The reciprocal column $\Omega_{ij}(1,1)^T$ is constant and gives $a=0$. Curvature spectra are gauge invariant, so the backtrack factor alone changes the gauge class of the selected connection. This proves that a backtrack contribution can survive the selector, without asserting that every backtrack distinction is recoverable from one connection. In rank one the rectangular selection is always pure gauge, $a=-\tfrac12d\log H_X$; for example, a scalar link $\Theta_{ij}=\Theta_{ji}=\vartheta(c)>0$ with raw column gives $a=-\vartheta d\vartheta/(1+\vartheta^2)$, which differs from the reciprocal selection $a=0$ only by a gauge transformation."""),
    # group selector
    (r"Let $M(c)$ be a $C^1$ positive-definite fiber metric with $M'=h^TMh$, and let $X$ have full row rank and \cref{ibt:eq:X-law}.",
     r"Let $M(c)$ be a $C^1$ positive-definite fiber metric with $M'=h^TMh$, and let $X$ have full row rank and \cref{ibt:eq:X-law}, with reference changes $h(c)\in G$."),
    (r"""In the unrestricted algebra the normal equation is $(d X+aX)X^T=0$, yielding \cref{ibt:eq:derived-a}.
\end{proof}""",
     r"""In the unrestricted algebra the normal equation is $(d X+aX)X^T=0$, yielding \cref{ibt:eq:derived-a}. The affine law uses $h^{-1}dh\in\mathfrak g$, that is, $h$ valued in $G$.
\end{proof}

For a proper subalgebra the minimizer depends on $M$, which is then an additional comparison choice. For $G=\mathrm{SL}(K)$ and $M=H_X^{-1}$ the Hessian form is $\Tr(H_X^{-1}\xi H_X\eta^T)$, for which the identity is orthogonal to every traceless $\eta$; the minimizer is therefore the traceless part $a+\tfrac1{2K}(d\log\det H_X)I$ of \cref{ibt:eq:derived-a}, with the same curvature, since the removed part is an exact scalar form."""),
    (r"Finite $G$-valued transport follows by \cref{ibt:eq:path-ode} along declared short paths with this derived $a$.",
     r"Finite $G$-valued transport follows by \cref{ibt:eq:path-ode} along every path in the regular region with this derived $a$."),
    (r"To select one principal connection from both, sum their costs, replacing $\xi X$ by $d\varrho_a(\xi)X^a$, and minimize over the same $\xi\in\mathfrak g$.",
     r"To select one principal connection from both, sum their costs with a declared relative sector weight, replacing $\xi X$ by $d\varrho_a(\xi)X^a$, and minimize over the same $\xi\in\mathfrak g$; the weight changes the selected generator."),
    # alignment
    (r"The rectangular construction is sufficient, but a full-rank vector collection is not necessary when the group action on links is itself identifiable.",
     r"The rectangular construction is sufficient, but a full-rank vector collection is not necessary when the group action on links is itself identifiable and an invariant metric on the link target exists. For compact groups the second condition is automatic; for noncompact groups it is restrictive, as explained after \cref{ibt:alignment}."),
    (r"""\propositionheading{Local link alignment}{ibt:alignment}
Near each diagonal point there is a unique critical branch near $e$ that is a local minimizer of
\begin{equation}
 t(c',c)=\mathop{\operatorname{argmin}}_{g\text{ near }e}
       \tfrac12 d_{\mathcal H}(Z(c'),g\cdot Z(c))^2.
 \label{ibt:eq:alignment}
\end{equation}
It satisfies""",
     r"""\propositionheading{Local link alignment}{ibt:alignment}
Near each diagonal point there is a unique smooth branch $t(c',c)$ near $e$ of strict local minimizers of
\begin{equation}
 g\longmapsto\tfrac12 d_{\mathcal H}(Z(c'),g\cdot Z(c))^2.
 \label{ibt:eq:alignment}
\end{equation}
It satisfies"""),
    (r"This is the mechanical connection of the declared target metric, as in the rotation--shape separation of \citet{LittlejohnReinsch1997}.",
     r"This is the mechanical connection of the declared target metric \citep{Smale1970}, as in the rotation--shape separation of \citet{LittlejohnReinsch1997}. With $\mathcal Z$ the declared law family, $\mathcal H$ its product Fisher metric, and $h$ replaced by $h^{-1}$ (\cref{ch:contextual-coupling} writes target changes, this chapter reference changes), $t(c',c)$ is the near-identity branch of \cref{ctx:f:ass:branch} and its first jet is the collective connection of \cref{geo:collective}; the new content here is the general target. For a free action the argument of \cref{geo:shape-pullback} shows that the curvature is the pullback of the curvature of the target's mechanical connection by $Z$, so it depends only on the derivative of the shape map $c\mapsto[Z(c)]$ and vanishes where that map has rank below two."),
    (r"Infinitesimal identifiability does not exclude distant competing minima.",
     r"Infinitesimal identifiability does not exclude distant competing minima; \cref{ctx:f:eq:cusp} is an example of a branch singularity under a free action."),
    (r"""Its first jet gives path transport by the preceding ODE argument.
""",
     r"""Its first jet gives path transport by the preceding ODE argument.

For noncompact $G$ the isometric target is a substantive hypothesis, and injectivity of $J_Z$ does not supply it. If $z$ has stabilizer $G_z$, each element of $G_z$ acts on $T_z\mathcal Z$ by an isometry of $\mathcal H_z$, so the isotropy representation of $G_z$ must have relatively compact image. For $\GL(2)$ acting by conjugation on one link $\operatorname{diag}(\lambda_1,\lambda_2)$ with real $\lambda_1\neq\lambda_2$, the element $\operatorname{diag}(t,1)$ fixes the link and scales the tangent vector $E_{12}$ by $t$, which no positive inner product allows for all $t>0$. Even where the action is free, an invariant metric can fail to exist because the action is not proper. For $\mathrm{SL}(2)$ acting by conjugation on pairs, let $D=\operatorname{diag}(2,1/2)$, $z=(D,I+E_{12})$, $z'=(D,I+E_{21})$, $z_t=\bigl(D,\bigl(\begin{smallmatrix}1+t^2&1\\t^2&1\end{smallmatrix}\bigr)\bigr)$ and $g_t=\operatorname{diag}(t,1/t)$. Both $z$ and $z'$ have trivial infinitesimal stabilizer, $z_t\to z$ and $g_tz_tg_t^{-1}\to z'$ as $t\to0$, while $g_tzg_t^{-1}\to(D,I)$, a point with a one-dimensional stabilizer. An invariant metric on an invariant open set of free points containing $z$ and $z'$ would give $d(g_tz,z')\leq d(z,z_t)+d(g_tz_tg_t^{-1},z')\to0$, which is impossible. Invariant metrics do exist on invariant open sets where the action is proper \citep{Palais1961}, but none is canonical in general. For the defining representation of $\GL(K)$, scalar matrices fix every conjugation link (\cref{ibt:invisible}), so link-only targets are never identifiable and the link term must be dressed by law or interface data, as in the example metric below. The link-only curved example below accordingly uses a compact group.
"""),
    (r"""For example on a full-rank matrix region the terms
\begin{equation}
 \Tr(H_X^{-1}\delta X\delta X^T)
 +\sum_e\Tr(H_X^{-1}\delta K_e H_X\delta K_e^T)
\end{equation}
are positive and invariant under $X\mapsto h^{-1}X$, $K_e\mapsto h^{-1}K_eh$.""",
     r"""For example on a full-rank matrix region the terms
\begin{equation}
 \Tr(H_X^{-1}\delta X\delta X^T)
 +\sum_e\Tr(H_X^{-1}\delta\bar\Theta_e H_X\delta\bar\Theta_e^T)
\end{equation}
are positive and invariant under $X\mapsto h^{-1}X$, $\bar\Theta_e\mapsto h^{-1}\bar\Theta_eh$. Unlike the rectangular selector, the selector of this augmented metric need not preserve $H_X^{-1}$, so it is not subject to \cref{ibt:orthogonal-holonomy}. If the target retains reciprocal parts of tied links from both sectors, the sector-wise principal roots of \cref{ibt:eq:reciprocal} need not be the images of one element of $G$ when the two representations have different charges; a reciprocal part computed in $G$ and then represented in each sector avoids this."""),
    # SO(3) example
    (r"""Let $C_r=I-\operatorname{Ad}_{Q_r}$ and $V_r=d Q_r Q_r^{-1}$. The right-trivialized infinitesimal conjugation is $C_r\xi$, so \cref{ibt:eq:augmented-normal} reads
\begin{equation}
 \mathcal K a=-\sum_{r=1}^2 C_r^*V_r,\qquad
 \mathcal K=\sum_{r=1}^2 C_r^*C_r.
 \label{ibt:eq:SO3-normal}
\end{equation}""",
     r"""Let $\mathsf C_r=I-\operatorname{Ad}_{Q_r}$ and $V_r=d Q_r Q_r^{-1}$. The right-trivialized infinitesimal conjugation is $\mathsf C_r\xi$, so \cref{ibt:eq:augmented-normal} reads
\begin{equation}
 \mathfrak k a=-\sum_{r=1}^2 \mathsf C_r^*V_r,\qquad
 \mathfrak k=\sum_{r=1}^2 \mathsf C_r^*\mathsf C_r.
 \label{ibt:eq:SO3-normal}
\end{equation}"""),
    (r"which are nonparallel on the stated region. Thus $\mathcal K$ is positive definite.", r"which are nonparallel on the stated region. Thus $\mathfrak k$ is positive definite."),
    (r"$V_1=\widehat e_1d x$ and $V_2=C_2\widehat e_3d y$ imply $C_1^*V_1=0$ and $C_2^*V_2=w_\beta\widehat e_3d y$. Since $\mathcal K\widehat e_3=(w_x+w_\beta)\widehat e_3$, one obtains",
     r"$V_1=\widehat e_1d x$ and $V_2=\mathsf C_2\widehat e_3d y$ imply $\mathsf C_1^*V_1=0$ and $\mathsf C_2^*V_2=w_\beta\widehat e_3d y$. Since $\mathfrak k\widehat e_3=(w_x+w_\beta)\widehat e_3$, one obtains"),
    (r"The wedge square vanishes because $a$ has only one base component. This proves nonzero curvature from smoothly varying reciprocal agent links and a declared comparison metric, without adding directional generators or a spatial connection as input. Normalized rotationally invariant agent laws may be attached; their orbit terms are zero and do not alter this example.",
     r"The wedge square vanishes because $a$ has only one base component. This proves nonzero curvature from smoothly varying reciprocal agent links and a declared bi-invariant comparison metric, without adding directional generators or a spatial connection as input. The connection and its curvature take values in the line spanned by $\widehat e_3$, the common normal of the two axes, so the connection reduces to the rotations about $e_3$ and its holonomy is Abelian; the example exhibits no non-Abelian contextual holonomy. Normalized rotationally invariant agent laws may be attached with a product metric on laws and links; their orbit terms are zero and do not alter this example."),
    # invisible
    (r"If a nonzero $Z\in\mathfrak g$ commutes with every admitted link field, then no full connection selector from these inputs alone can obey \cref{ibt:eq:affine} for every smooth reference change $h(c)=\exp(f(c)Z)$.",
     r"If a nonzero $\zeta\in\mathfrak g$ commutes with every admitted link field, then no full connection selector from these inputs alone can obey \cref{ibt:eq:affine} for every smooth reference change $h(c)=\exp(f(c)\zeta)$."),
    (r"""At that point the affine law requires $a'=a+Zd f$, a contradiction.
\end{proof}""",
     r"""At that point the affine law requires $a'=a+\zeta d f$, a contradiction.
\end{proof}

The obstruction is pointwise for selectors that read only values and first derivatives of the links, as the normal equation \cref{ibt:eq:augmented-normal} does. If a nonzero $\zeta$ commutes with every link value $\bar\Theta_e(c_0)$ at one point, take $h=\exp(f\zeta)$ with $f(c_0)=0$ and $df(c_0)\neq0$. Then $\bar\Theta_e'(c_0)=\bar\Theta_e(c_0)$ and $d\bar\Theta_e'(c_0)=d\bar\Theta_e(c_0)+[\bar\Theta_e(c_0),\zeta]df(c_0)=d\bar\Theta_e(c_0)$, while the affine law requires $a'(c_0)=a(c_0)+\zeta df(c_0)$. No such selector is covariant at a point of nonzero common infinitesimal stabilizer."""),
    # comparison data
    (r"Column sets, weights, message metrics, and augmented target metrics affect the selector.",
     r"Column sets, weights, message metrics, augmented target metrics, the auxiliary fiber metric $M$ of a proper subalgebra, and the relative sector weight affect the selector."),
    # message coordinate change
    (r"For a change $w'=s(c)w$ of message coordinates, $X'=Xs^{-1}$, $\Gamma'=s\Gamma s^{-1}-d s s^{-1}$ and $Q_X'=sQ_X$, provided the message metric is transformed as well. Hence $d X'-X'\Gamma'=(d X-X\Gamma)s^{-1}$,",
     r"For a change $w'=\Lambda(c)w$ of message coordinates, $X'=X\Lambda^{-1}$, $\Gamma'=\Lambda\Gamma\Lambda^{-1}-d\Lambda\Lambda^{-1}$ and $Q_X'=\Lambda Q_X$, provided the message metric is transformed as well. Hence $d X'-X'\Gamma'=(d X-X\Gamma)\Lambda^{-1}$,"),
]
for a, b in R:
    n = s.count(a)
    if n != 1:
        sys.exit(f"count {n} for: {a[:90]}")
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
