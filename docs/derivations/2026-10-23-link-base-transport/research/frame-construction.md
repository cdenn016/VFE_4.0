# A connection from a labeled collection of receiving vectors

Date: 2026-10-23. Bounded Item 5 research contribution. This file contains elementary derivations, not a numerical experiment or an audit certificate. No scripts, GPU work, Lean, Research-vault changes, or Item 7 changes were performed.

## Target, sources, and introduced assumptions

The target is a constitutive construction which, for every admitted smooth collection of agent laws and interfaces on an open base domain, produces a displacement family with identity on the diagonal, the endpoint coordinate law, and a first derivative linear in displacement. Its derivative must be a connection, with a possible nonzero curvature. Equivalence means equality of the resulting intrinsic connection and path transports under changes of presentation; it does not mean preservation of all information in the original laws or directed links.

The supplied theory permits general normalized laws, receiving maps, sending maps, and independent directed links. Its current frames are presentations. The construction below additionally assumes a finite collection of persistently labeled receiving vectors which spans the target fiber, smooth variation in the base, and positive weights on their label space. These are constitutive assumptions; they are not consequences of pointwise gauge covariance or of an arbitrary pair of directed links.

The live sources consulted were `Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex`, section “Presentation and probability”; `chapters/02_interactions.tex`, definitions of reference generators and group-valued links, the moment-dependent family, and Proposition `prop:int-reciprocal`; and `appendices/base_connection.tex`, especially `eq:conn-reference-affine` and `open:conn-derived`. The earlier `Research/manuscripts/pifb2/2026-10-09-kaluza-klein-pullback/06-collective-curvature.md`, sections 0 and 1, and the corresponding collective-curvature paragraph of `RESULTS.md`, already formulate averaging flat connections of marked agent frames. The distinction between those marked frames and the live manuscript's presentation matrices is material.

Three routes are available. A single full frame gives a flat connection once that frame is physically marked; its first lemma is reference invariance of the marking. Averaging several such connections can give curvature; its first lemma is that all terms are connections on the same bundle. A redundant collection of receiving vectors gives a connection by weighted regression; its first lemma is the affine transformation law. The last route gives the most direct construction from existing interfaces and is worked through below. The marked-frame route is retained in the final section.

## Full-rank receiving data and its connection

Let \(O\) be an open subset of a smooth manifold and \(E\to O\) a real vector bundle of rank \(K\). Declare the message-label bundle \(H=O\times\mathbb R^r\) with its fixed labeled basis and its flat derivative \(d\). Let \(X:H\to E\) be a \(C^2\) bundle map, surjective at each point, and let \(W:O\to\operatorname{Sym}^+_r\) be \(C^1\). In a local target trivialization, \(X\) is a \(K\times r\) matrix. \(W\) is a positive contravariant form on the label space, unchanged by target coordinate changes. Define

\[
C=XWX^T,\qquad R=WX^TC^{-1},\qquad P=RX.
\]

All inverses exist: for nonzero \(v\in\mathbb R^K\),

\[
v^TCv=(X^Tv)^TW(X^Tv)>0,
\]

because surjectivity of \(X\) makes \(X^T\) injective. Thus \(XR=I_K\), \(P^2=P\), and \(\ker P=\ker X\). On every compact subset of the full-rank domain, the minimum eigenvalue of \(C\) is positive. Define

\[
A=-dX R.
\tag{1}
\]

This is a \(C^1\), endomorphism-valued one-form. In particular, \(A(v)=-dX(v)R\) depends linearly on each tangent vector \(v\).

Under a change of reference presentation \(U_i'=U_i h\), the reference target coordinates transform by \(h^{-1}\). Therefore

\[
X'=h^{-1}X,\quad C'=h^{-1}Ch^{-T},\quad R'=Rh.
\]

The product rule gives

\[
\begin{aligned}
A'&=-d(h^{-1}X)Rh\\
&=h^{-1}Ah+h^{-1}dh.
\end{aligned}
\tag{2}
\]

This is exactly the affine reference law required by the live appendix. Since the same argument applies to transition functions on overlaps, (1) defines a connection on the general linear frame bundle of \(E\) whenever \(X\) and \(W\) are global bundle data of the stated type. A global surjection from a trivial label bundle is an additional global assumption; the local theorem does not assert that a specified finite \(r\) works on every bundle.

In agent display coordinates, put \(X_i=U_iX\). Its weighted right inverse is \(R_i=RU_i^{-1}\), and

\[
B_i=-dX_iR_i=U_iAU_i^{-1}-dU_iU_i^{-1}.
\tag{3}
\]

Thus independent passive changes \(X_i'=g_iX_i\) give

\[
B_i'=g_iB_ig_i^{-1}-dg_i g_i^{-1}.
\]

Reference changes leave the displayed \(B_i\) unchanged, by (2) and (3). These two transformation tests are different; both are satisfied.

Equation (1) is also the unique weighted least-squares solution. Let \(M\) be any positive-definite covariant target metric. For a fixed tangent vector, minimize over \(Z\in\mathfrak{gl}(K,\mathbb R)\)

\[
J(Z)=\operatorname{tr}\bigl[M(dX+ZX)W(dX+ZX)^T\bigr].
\tag{4}
\]

Here all one-forms in (4) are evaluated on that tangent vector. Variation in \(Z\) yields

\[
(dX+ZX)WX^T=0,
\]

so \(Z=-dXWX^TC^{-1}=A\). Strict convexity follows because the quadratic part is \(\operatorname{tr}(MZCZ^T)>0\) for every nonzero \(Z\). The solution is independent of \(M\). If a metric is desired without an external Euclidean choice, \(M=C^{-1}\) is available. A positive covariance inverse obtained from an admitted normalized law is another choice. Neither choice assumes Gaussian laws.

The residual is

\[
dX+AX=dX(I-P).
\tag{5}
\]

Thus the connection matches changes which can be described by a common linear transformation of all columns, and retains the change in their relative configuration as a residual. It does not force every receiving vector to be parallel.

## Curvature, restrictions, and explicit curved examples

Use \(F=dA+A\wedge A\), with matrix multiplication in wedge products. Differentiating \(XR=I\) gives \(dX R+X dR=0\). Consequently,

\[
\begin{aligned}
F&=dX\wedge dR+dX R\wedge dX R\\
 &=dX\wedge(dR+R dX R)\\
 &=dX\wedge(I-P)dR.
\end{aligned}
\tag{6}
\]

The first sign follows from \(d(dX R)=-dX\wedge dR\). For the final equality, multiply the differentiated identity by \(R\) on the left. Expanding the inverse in \(R\) gives the more explicit formula

\[
F=dX\wedge(I-P)(dW X^T+W dX^T)C^{-1}.
\tag{7}
\]

When \(W\) is constant,

\[
F=dX\wedge Q dX^T C^{-1},\qquad
Q=(I-P)W=W-WX^TC^{-1}XW.
\tag{8}
\]

The matrix \(Q\) is symmetric positive semidefinite: writing \(Y=XW^{1/2}\), it equals \(W^{1/2}[I-Y^T(YY^T)^{-1}Y]W^{1/2}\), and the bracket is the Euclidean orthogonal projection onto \(\ker Y\).

If \(r=K\), then \(R=X^{-1}\), \(P=I\), and \(F=0\). More generally, if \(X=V(c)X_0\) for a fixed full-rank \(X_0\) and an invertible \(V(c)\), then \(A=-dV V^{-1}\) and \(F=0\), even for varying \(W\). Redundancy permits curvature; it does not guarantee it.

There is a restriction when \(W\) is fixed. Direct differentiation gives

\[
dC+AC+CA^T=X dW X^T.
\tag{9}
\]

For constant \(W\), the induced metric \(C^{-1}\) is parallel. In particular,

\[
FC+CF^T=0,\qquad \operatorname{tr}F=0.
\tag{10}
\]

The first identity follows either by applying the connection twice to the parallel contravariant metric, or directly from (8), since \((dX\wedge QdX^T)^T=-dX\wedge QdX^T\). Multiplying the first identity in (10) by \(C^{-1}\) and taking the trace gives the second. Also \(\operatorname{tr}A=-\tfrac12d\log\det C\), obtained by taking the trace of (9) after multiplication by \(C^{-1}\). Thus the fixed-weight construction does not realize arbitrary \(GL(K)\) curvature. It produces a general linear connection with a preserved positive metric. In a local orthonormal frame for that metric, its matrix form lies in \(\mathfrak o(K)\). In rank one it is flat.

An explicit rank-two witness uses base coordinates \((x,y)\), \(v=(x,y)^T\), and

\[
X=\begin{pmatrix}1&0&x\\0&1&y\end{pmatrix},\qquad W=I_3,\qquad s=1+x^2+y^2.
\]

Then

\[
C=I_2+vv^T,\qquad
C^{-1}=I_2-\frac{vv^T}{s},\qquad
A=-\frac{dv v^T}{s}.
\tag{11}
\]

Here the inverse is verified by multiplying \((I+vv^T)(I-vv^T/s)\) and using \(v^Tv=s-1\). Since \(\ker X\) is generated by \((-x,-y,1)^T\), the \((3,3)\) entry of \(Q\) is \(1/s\). Only the third column of \(dX\) is nonzero. Equation (8) therefore yields

\[
F=\frac{1}{s^2}
\begin{pmatrix}
-xy&1+x^2\\
-(1+y^2)&xy
\end{pmatrix}dx\wedge dy.
\tag{12}
\]

This is nonzero at every point, and at the origin it is \(\begin{pmatrix}0&1\\-1&0\end{pmatrix}dx\wedge dy\). No prescribed base one-form or noncommuting directional generators were inserted. The varying relative position of the third labeled receiving vector supplies the curvature.

Varying weights remove the trace-curvature restriction. In rank one, take

\[
X=(1,x),\qquad W=\operatorname{diag}(1,e^{2y}).
\]

Directly from (1),

\[
A=-\frac{x e^{2y}}{1+x^2e^{2y}}dx,\qquad
F=dA=\frac{2x e^{2y}}{(1+x^2e^{2y})^2}dx\wedge dy.
\tag{13}
\]

This is nonzero whenever \(x\ne0\). Here the positive scalar label weights are constitutive fields. The construction does not derive their selection from VFE dynamics.

## Displacement links and exact reciprocity

On a common target trivialization, define

\[
D(b,a)=X(b)R(a).
\tag{14}
\]

It has \(D(a,a)=I\), and hence is invertible on a neighborhood of each diagonal point by continuity of its determinant. Its reference transformation law is

\[
D'(b,a)=h(b)^{-1}D(b,a)h(a).
\tag{15}
\]

For \(\delta\) in a coordinate chart,

\[
D(a+\delta,a)=I-A_a(\delta)+O(|\delta|^2).
\tag{16}
\]

The remainder is locally uniform on compact sets because \(X\) is \(C^2\). This meets the local identity, endpoint, and first-derivative conditions in `open:conn-derived`, under the additional labeled-data assumptions above.

The finite family \(D\) generally has neither exact reciprocity nor composition. For (11), \(D(b,0)=I\) but \(D(0,b)=C(b)^{-1}\), so \(D(0,b)D(b,0)\ne I\) for nonzero \(b\). This is compatible with a well-defined connection: the first derivative determines infinitesimal parallel transport, and finite transport is obtained from its path equation, not by identifying all finite \(D\) with exact parallel transports.

The manuscript's reciprocal normalization supplies an exactly reversible local displacement family without changing the connection. Set

\[
N_a(b)=D(a,b)D(b,a),\qquad
\Omega(b,a)=D(b,a)N_a(b)^{-1/2}.
\tag{17}
\]

In a sufficiently small neighborhood of any diagonal point, \(N_a(b)\) is close to \(I\). The inverse square root is well defined there by the convergent matrix power series for \((I+Z)^{-1/2}\), so no global branch assertion is needed. That series commutes with similarity and satisfies the scalar product identities by absolutely convergent matrix-series multiplication. The matrices at the two endpoints are similar, \(N_b(a)=D(b,a)N_a(b)D(b,a)^{-1}\). Substitution therefore gives

\[
\Omega(a,b)\Omega(b,a)=I,
\]

and (15) holds with \(D\) replaced by \(\Omega\). Differentiating \(X R=I\) also gives

\[
D(a,a+\delta)=I+A_a(\delta)+O(|\delta|^2).
\]

Consequently \(N_a(a+\delta)=I+O(|\delta|^2)\) and

\[
\Omega(a+\delta,a)=I-A_a(\delta)+O(|\delta|^2).
\tag{18}
\]

Thus reciprocity can be imposed on these new displacement links without changing (1). It does not delete or identify the original directed attending links or their original backtrack variables. They can be retained in the operational state. The map from the original state to \(A\) is generally many-to-one.

For each piecewise \(C^1\) path, the equation \(\dot T=-A(\dot\gamma)T\), \(T(0)=I\), defines invertible path transport. Existence and uniqueness follow on a compact path interval from the linear integral equation with continuous coefficients, for example by its uniformly convergent iterated-integral series bounded by \(M^n/n!\). Solving \(\dot S=SA(\dot\gamma)\), \(S(0)=I\), shows \(ST=I\) by differentiation. Uniqueness gives concatenation and reversal. This paragraph establishes the exact path transport; it makes no claim here about a numerical discretization or a product-limit rate.

## Realization by general laws and existing interfaces

The columns of \(X\) can be receiving columns \(O_j\), means of persistently labeled laws, or a mixture of such covariant data. The live moment construction already supplies receiving columns \(\bar\mu_j\), each transforming as \(h^{-1}\bar\mu_j\). Stacking a collection which spans the target gives the hypotheses of (1). Only finite first moments are needed to define those columns; the existing moment-dependent links also use finite second moments and nonsingular covariance.

The curved example (11) can use three non-Gaussian normalized laws with means \(e_1,e_2,(x,y)^T\). Start from the centered product density proportional to \(\exp[-(z_1^4+z_2^4)]\), rescale its coordinates to variance one, and translate it to these three means. The normalizing integral and second moments are finite by comparison with Gaussian tails outside a compact set, and each variance is strictly positive because the density is positive on every open set. The resulting translation family has smooth densities, and differentiation under integrals against bounded test functions is justified by integrable polynomial-times-exponential bounds on compact parameter sets. Every law has covariance \(I_2\). The constant label weight \(W=I_3\) is a declared weight on persistent labels, not an untransformed target metric. Reference changes push forward the complete laws and transform their means and covariances as in the live moment example, so the calculation remains valid in every frame. This is an exact law-level realization of the receiving data; it does not identify all law differences with differences of means.

A receiver-specific alternative in display coordinates is to stack vectors such as

\[
X_i=\bigl[\mu_i,\{\Omega_{ij}\mu_j\}_j,\{\Theta_{ij}\mu_j\}_j\bigr].
\tag{19}
\]

The ordering and existence of the labels are declared, and each included reciprocal part must meet its existing branch condition. Every column transforms by the same left factor \(g_i\), since \(\Theta_{ij}'=g_i\Theta_{ij}g_j^{-1}\) and \(\mu_j'=g_j\mu_j\). Thus (19) produces the displayed connection law whenever it is full rank. Including both directed and reciprocal columns permits sensitivity to directed asymmetry while retaining the original \(\Theta\) and backtrack state. There is no claim that this particular stacking is unique or selected dynamically.

Different receivers generally obtain different connections from (19). To claim one connection on the common principal bundle, their reference forms must agree, or a further common aggregation rule must be declared. A scalar affine combination of already defined common-bundle connections is another connection when the scalar weights sum to one and are coordinate invariant. None of these operations proves that one selected connection is the unique physical base geometry.

For every resulting invertible path map \(T\), the kernel \(K_T(z,A)=\mathbf1_A(Tz)\) is a normalized Markov kernel. It acts on every Borel probability law by pushforward. Hence the construction extends to general laws once the receiving data exist; Gaussian closure is not required. Existence of the required moments and smooth sections remains a condition on the chosen realization.

## The label-space assumption and smaller structure groups

The fixed labeled space \(H=O\times\mathbb R^r\) is real cross-context structure. It says which receiving column at one context corresponds to which column at a neighboring context. This need not be a separately supplied base one-form: persistent agent and interface labels can declare a globally constant label basis. It nevertheless goes beyond the data at one isolated context.

The dependence is visible under a context-dependent message-coordinate change \(s(c)\). Write \(X'=Xs^{-1}\), \(W'=sWs^T\), and \(R'=sR\). Then the raw formula gives

\[
-dX'R'=A+Xs^{-1}dsR.
\tag{20}
\]

Thus it is not invariant under arbitrary changing label coordinates while the flat derivative is incorrectly held fixed. If the label space is instead a bundle with a declared connection \(\Gamma\), the coordinate-independent extension is

\[
A=-(dX-X\Gamma)R.
\tag{21}
\]

Under \(\Gamma'=s\Gamma s^{-1}-ds s^{-1}\), one checks

\[
dX'-X'\Gamma'=(dX-X\Gamma)s^{-1},
\]

so (21) is invariant under the message-coordinate change and still satisfies the target affine law (2). Equation (1) is (21) in the declared flat label basis. No claim is made that arbitrary pointwise links select that basis or a nontrivial message connection.

For a proper matrix Lie subgroup \(G\subseteq GL(K,\mathbb R)\), (1) need not lie in \(\mathfrak g\). It is always a connection of the extended general linear bundle. One exact repair is to minimize (4) over \(Z\in\mathfrak g\), assuming a target metric \(M\) with \(M'=h^TMh\) for \(h\in G\). Strict convexity still gives a unique minimizer, denoted \(A_G\), and its coefficients are smooth because its positive Gram matrix is smoothly invertible. Under a reference change, the affine map

\[
Z\longmapsto h^{-1}Zh+h^{-1}dh
\]

maps \(\mathfrak g\)-valued candidates bijectively to themselves. Its residual transforms by \(h^{-1}\), leaving the value of (4) unchanged. Uniqueness therefore proves that \(A_G\) obeys (2) and is a principal \(G\)-connection in that representation. Equivalently, it is the orthogonal projection of (1) onto \(\mathfrak g\) for the pointwise inner product

\[
\langle Z_1,Z_2\rangle_{M,C}=\operatorname{tr}(MZ_1CZ_2^T).
\]

This constrained connection does not in general have the curvature formula (8) or the finite family (14). Its path transports can be constructed directly from its group-valued differential equation. The unconstrained finite matrix normalization alone does not establish membership in a smaller group.

The live theory has one group acting through two sector representations. Independent sector regressions need not come from one principal connection. A compatible construction instead minimizes the sum of the sector objectives over one \(\xi\in\mathfrak g\), with \(Z_a=d\varrho_a(\xi)\). Its minimizer is unique precisely when the joint quadratic form is positive, equivalently when

\[
d\varrho_a(\xi)X_a=0\text{ for every included sector }a
\quad\Longrightarrow\quad\xi=0.
\tag{22}
\]

Full rank of each \(X_a\) reduces (22) to joint infinitesimal faithfulness of the representations. Otherwise there is an unresolved stabilizer direction, and a unique principal connection cannot be claimed from these data alone.

## Why presentation frames alone do not select the connection

In the live manuscript, \(u_i=\sigma U_i^{-1}\). The frame is a choice of description, and a passive change gives \(U_i'=g_iU_i\). Setting \(B_i^0=-dU_iU_i^{-1}\) gives a flat displayed connection and has the correct passive law. But changing the reference section to \(\sigma h\), keeping the same \(u_i\), gives \(U_i'=U_i h\) and

\[
B_i^{0\prime}=B_i^0-U_i(dh h^{-1})U_i^{-1}.
\tag{23}
\]

The original agent bundle data have not changed, but the connection has. Therefore presentation matrices alone do not determine an intrinsic connection. The expression selects a flat reference connection \(A=0\) and then displays it in the agent's coordinates. A varying change of reference must transform that \(A\) affinely; setting it back to zero chooses a different connection.

A physically marked full frame is different. Suppose a section \(f_i\) of the frame bundle is part of the operational data and is to be parallel. In reference coordinates write its columns as an invertible matrix \(V_i\). Under reference changes, \(V_i'=h^{-1}V_i\). Then

\[
a_i=-dV_iV_i^{-1}
\tag{24}
\]

is an intrinsic flat connection on that patch. In the current convention, marking \(u_i\) itself gives \(V_i=U_i^{-1}\), and hence \(a_i=U_i^{-1}dU_i\); its displayed form in that marked frame is zero. A passive re-expression does not move the physical marking. Replacing the marking by an arbitrary new presentation is a change of the constitutive data, rather than another formula for the same connection.

This is the assumption made by the earlier collective-curvature record when it treats the \(U_i\) as marked frames written in one common trivialization with a common left coordinate action. That record is a valid conditional mechanism under its stated assumption. Its frame symbol has a different role from the current presentation matrix. With common-bundle flat connections (24) and invariant scalar weights \(\lambda_i\) summing to one,

\[
a=\sum_i\lambda_i a_i
\]

is a connection, and direct differentiation gives

\[
F_a=\sum_i d\lambda_i\wedge a_i
-\frac12\sum_{i,j}\lambda_i\lambda_j
(a_i-a_j)\wedge(a_i-a_j).
\tag{25}
\]

This formula follows from \(da_i=-a_i\wedge a_i\), expansion of \(a\wedge a\), and \(\sum_i\lambda_i=1\). It can be curved even when every \(a_i\) is flat. What has to be supplied is the physical marking and weight selection. If the frames remain only coordinate choices, that hypothesis has not been met.

## Supported conclusion and remaining obligation

On every smooth full-rank domain, a persistently labeled receiving collection and positive label weights yield the exact connection (1), the locally invertible displacement family (14), and the exactly reversible local family (17) with the same first derivative. Equations (12) and (13) are explicit curved examples, and general non-Gaussian laws realize the receiving data. Fixed weights impose the metric-preservation restriction (9); proper groups and jointly acting sectors require the constrained construction and its identifiability condition.

This closes a conditional construction problem, not a universal uniqueness or dynamics-selection problem. The smallest remaining obligation for the live theory is to specify which operationally present receiving data, persistent labels, and weights define the chosen base transport, and to establish their full-rank domain. If transport is to be determined by pointwise directed links alone with no cross-context identification, that broader target remains open. The next action that materially changes the result is to choose a concrete admitted stacking from the live interfaces, prove its reference consistency and rank on an explicit class of configurations, and state whether its weights are fixed constitutive data or dynamically selected.
