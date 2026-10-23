# Item 5 audit: link alignment and reference gauge

Date: October 23, 2026.

No gauge-theory findings.

The assigned arguments are mathematically sound on their stated regular regions. Conditional integration is supported for the local alignment construction, its SO(3) curvature witness, the invisible-subgroup obstruction, the square-frame calculation, and directed-pair retention. This is an analytical assessment of those statements. No scripts, numerical calculations, GPU work, Lean work, or adversarial adjudication were run.

## Scope and source continuity

I read the complete standalone `docs/derivations/2026-10-23-link-base-transport/latex/main.tex` and `research/link-alignment.md`, together with the derivation README. The live manuscript comparison used `MAgent_Theory_2.0/AUTHORING.md`, the presentation section of `chapters/01_foundations.tex`, the reference-consistency and reciprocal-decomposition results in `chapters/02_interactions.tex`, the collective-connection section of `chapters/02_geometry.tex`, and `appendices/base_connection.tex`. The earlier `pifb2/2026-10-09-kaluza-klein-pullback/06-collective-curvature.md`, Sections 0 and 1, supplies the preceding affine-average construction. The Research gauge theme and GL(K) concept pages were consulted as context, not as mathematical proof.

The distinctions in the live manuscript are preserved. Independent changes of displayed agent coordinates leave the reference data unchanged; a common reference rechoice acts on those reference data and requires an affine connection law. Agent labels identify different descriptions at one context and are not base coordinates. Opposite directed attending links remain independent.

## Local alignment and the first-order normal equation

Location: `latex/main.tex:310-329`; `research/link-alignment.md:29-127`.

For the stated smooth isometric action, set

\[
 f(v,u,g)=\tfrac12d_{\mathcal H}(Z(v),g\cdot Z(u))^2.
\]

At a diagonal point, the squared distance has zero first derivative and its quadratic term in group coordinates is

\[
 \tfrac12\mathcal H_Z(J_Z\xi,J_Z\xi).
\]

Injectivity of \(J_Z\) makes the group Hessian positive definite. The finite-dimensional implicit function theorem therefore gives a unique smooth stationary branch near the identity, with positive Hessian after shrinking the endpoint neighborhood. The identity remains stationary when the endpoints coincide, so uniqueness gives \(t(c,c)=e\). No global minimizer or free proper action is required for this local statement. The note explicitly excludes a conclusion about distant competing minima.

Writing \(v=c+\epsilon Y\) and \(t(v,c)=\exp(-\epsilon a(Y)+O(\epsilon^2))\), the first residual is \(\epsilon(dZ(Y)+J_Za(Y))\). Orthogonality to every orbit tangent yields

\[
 \mathcal H_Z(J_Za(Y),J_Z\eta)
 =-\mathcal H_Z(dZ(Y),J_Z\eta).
\]

There is no missing first-order metric-derivative term. Smooth dependence of \(\mathcal H\) on \(Z\) changes the quadratic norm by a term of order \(\epsilon^3\), because the residual vanishes on the diagonal. This statement concerns differentiation of alignment near that diagonal; it does not authorize freezing the metric or the selected connection when differentiating a later contextual-energy functional.

The normal equation is the mechanical-connection construction for the declared augmented metric. The classical inertia-weighted construction and the horizontal/vertical interpretation appear in Littlejohn and Reinsch 1997, Sections III.E and IV.B; the extension here is the choice of link and law data supplying the target, not a new general connection theorem. [Primary paper](https://chaosbk.physics.gatech.edu/library/LiRe97.pdf).

## Endpoint covariance, reversal, and the field-dependent metric

Location: `latex/main.tex:326-336`; `research/link-alignment.md:91-123,241-264`.

Make the reference convention explicit when integrating: \(Z'(c)=h(c)^{-1}\cdot Z(c)\). For a trial group element \(g\), put \(g'=h(v)^{-1}gh(u)\). Isometry gives the exact identity

\[
 f'(v,u,g')=f(v,u,g).
\]

The transformed branch passes through the identity on the diagonal, and local uniqueness consequently gives \(t'(v,u)=h(v)^{-1}t(v,u)h(u)\). This remains a comparison of branches near the identity after shrinking around any diagonal point. Differentiation gives \(a'=h^{-1}ah+h^{-1}dh\), with the manuscript's sign convention \(D=d+a\). The inhomogeneous gauge-potential transformation and covariant curvature are the classical laws exhibited in Littlejohn and Reinsch 1997, Section III.F, equation (3.74), and Section III.G, equations (3.91)-(3.93), after translating conventions. [Primary paper](https://chaosbk.physics.gatech.edu/library/LiRe97.pdf).

Distance symmetry and isometry give \(f(u,v,g^{-1})=f(v,u,g)\). Inversion carries the original stationary branch into the reversed stationary branch, and uniqueness proves exact reversal. This does not prove finite composition, and the note does not claim it. Its distinction between a local displacement map and exact path transport is consistent with Leok, Marsden, and Weinstein 2005, Sections 5.2 and 6.2, on the infinitesimal limit and approximation order of discrete connections. [Primary paper](https://arxiv.org/pdf/math/0508338).

For the explicit matrix metric, write \(H=XX^T\), \(B=\delta K\), and \(D=\delta X\). Under a fixed group action on the target,

\[
 H'=h^{-1}Hh^{-T},\qquad D'=h^{-1}D,
 \qquad B'=h^{-1}Bh.
\]

Substitution and cyclicity give both

\[
 \operatorname{Tr}(H'^{-1}D'D'^T)
 =\operatorname{Tr}(H^{-1}DD^T),
\]

\[
 \operatorname{Tr}(H'^{-1}B'H'B'^T)
 =\operatorname{Tr}(H^{-1}BHB^T).
\]

The two summands are squared Frobenius norms of \(H^{-1/2}D\) and \(H^{-1/2}BH^{1/2}\), respectively. They define a positive metric on the full-rank matrix/link product target. Thus this example supplies the isometry hypothesis even for nonorthogonal GL changes; no fixed conjugation-invariant Frobenius norm on GL is assumed. The covariance-dependent metric in the research note satisfies the corresponding calculation with \(\Sigma\mapsto h\Sigma h^T\).

The optional probe experiment in the research note is separately normalized. Its displayed coefficient \((m+1)/2\) counts one covariance score from the sampled input and one from each of the \(m\) conditionally independent outputs. Conditional centering kills cross-output and input/output score pairings, and centered Gaussian third moments kill mean/covariance pairings. This supports the stated probe metric. It does not make the simpler link metric the Fisher metric of the original agent laws.

## SO(3) witness

Location: `latex/main.tex:340-361`; `research/link-alignment.md:129-204`.

The right-trivialized tangent to conjugation is

\[
 ([\widehat\xi,Q])Q^{-1}
 =\widehat\xi-\operatorname{Ad}_Q\widehat\xi.
\]

Consequently the stated \(C_r=I-\operatorname{Ad}_{Q_r}\) and locked form \(\mathcal K=\sum_r C_r^*C_r\) have the correct signs and multiplication order. For a rotation about \(n\), the identity

\[
 C^*C=2I-Q-Q^T
 =4\sin^2(\alpha/2)(I-nn^T)
\]

follows directly from Rodrigues' formula. On the admitted open region, both weights are positive and the two axes are nonparallel, so the kernel of \(\mathcal K\) is zero. Since neither angle is zero or \(\pi\), a rotation centralizing both links must fix both oriented axes and is the identity. Thus the example also has a free action, although only infinitesimal injectivity is needed for the local theorem.

Differentiating the two links gives \(V_1=\widehat e_1dx\) and \(V_2=C_2\widehat e_3dy\). The first is killed by \(C_1^*\); the second contributes \(w_\beta\widehat e_3dy\). Since \(\mathcal K\widehat e_3=(w_x+w_\beta)\widehat e_3\), the connection and curvature are exactly

\[
 a=-\frac{w_\beta}{w_x+w_\beta}\widehat e_3dy,
 \qquad
 F^a=\frac{2w_\beta\sin x}{(w_x+w_\beta)^2}
 \widehat e_3dx\wedge dy.
\]

Here \(a\wedge a=0\) and \(dw_x/dx=2\sin x\); hence the curvature is nonzero throughout the stated region. Trivial graph backtracks are compatible with this result because the curvature concerns contextual variation of the link tuple. Constant rotationally invariant laws contribute zero orbit and contextual tangents, so adjoining them does not change this witness. This is an exact algebraic check, not a numerical measurement.

## Invisible subgroup, frame flatness, and directed-pair retention

Locations: `latex/main.tex:65-106,266-274,365-376`.

The invisible-subgroup proof is correct even when every contextual derivative of the links is supplied. For \(h=\exp(fZ)\) commuting pointwise with all admitted links, the transformed link functions equal the original functions, so all their derivatives also agree. At \(f(c_0)=0\), the affine law nevertheless adds \(Zdf(c_0)\). A deterministic selector of the unchanged data cannot produce that term. The claim is restricted to connection components invisible to the complete input tuple. It does not exclude adding laws or interfaces that detect the subgroup, and the note correctly limits the center statement for arbitrary endomorphism links to their actual common commutant.

For a physically marked square frame, \(T_Q(c',c)=Q(c')Q(c)^{-1}\) telescopes exactly and \(a_Q=-dQ Q^{-1}\) has zero curvature by the right Maurer-Cartan identity. Its proper reference law uses \(Q'=h^{-1}Q\). By contrast, resetting the reference form to zero after \(U_i\mapsto U_i h(c)\) changes the displayed connection by \(-U_i dh h^{-1}U_i^{-1}\). The note correctly separates these operations. Its two-connection affine-average formula follows from \(D^{a_2}\Delta=-\Delta\wedge\Delta\), matching the existing collective-curvature record's Section 1.2.

The square-root spectral hypothesis is sufficient. Higham 2008, Theorem 1.29, identifies the unique principal square root on this domain; similarity covariance follows from uniqueness of the root whose spectrum lies in the open right half-plane. [Primary publisher source](https://epubs.siam.org/doi/10.1137/1.9780898717778.ch6). With \(S_j=(\Theta_{ji}\Theta_{ij})^{1/2}\), the retained pair \((\Omega_{ij},S_j)\) reconstructs both original directed links:

\[
 \Theta_{ij}=\Omega_{ij}S_j,
 \qquad \Theta_{ji}=S_j\Omega_{ij}^{-1}.
\]

Thus reciprocalization does not delete directed information when the backtrack factor remains stored. The scalar example distinguishes connections selected from the raw linked column and the reciprocal-only column; it does not assert that a single connection retains all link distinctions. No restriction to inverse original attending links has been introduced.

## Conditions to preserve in integration

The integrated statement should explicitly declare the finite-dimensional Lie group, the associated target section and its reference action, one smooth positive G-invariant target metric, a regular region with injective infinitesimal action, and local squared-distance neighborhoods. These are clarifications of the hypotheses already used in the proofs, not repairs of a counterexample.

The comparison tuple, target metric, contextual base, and smooth dependence remain constitutive data. The construction supplies a connection from those inputs; it does not derive the base, establish preservation of rank or orbit type, or select an operative comparison rule from the existing dynamics. The link-augmented selector generally differs from Item 4's law-only Fisher selector. Substitution into a contextual energy requires the total constitutive derivative and a new dynamics argument. No formation or inheritance conclusion is established by this audit.
