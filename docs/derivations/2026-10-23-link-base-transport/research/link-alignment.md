# Link-augmented alignment and a base connection

This note addresses the bounded Item 5 question: given smooth contextual fields of reciprocal agent links and their backtrack parts, with optional law and interface data, when does a covariant alignment rule determine a connection on the supplied contextual bundle? The result is a conditional construction, an explicit curved example, and a precise obstruction. The base, smooth contextual dependence, admissible group, and alignment metric remain declared data. They are not consequences of a single pointwise interaction graph.

The live source conventions are those of `Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex`, “Presentation and probability”; `chapters/02_interactions.tex`, Proposition `prop:int-reciprocal`; `chapters/02_geometry.tex`, Proposition `geo:collective`; `chapters/15_contextual_coupling.tex`, “Finite transports and their stationarity equation” and “Relation to the infinitesimal connection equation”; and `appendices/base_connection.tex`. In particular, this construction augments the existing law-alignment mechanism by declared interaction data. It is not a new derivation of that existing mechanism from first principles.

## Conventions and data

| Object | Type and convention |
| --- | --- |
| Contextual base | A smooth manifold C, with coordinate neighborhoods U. Its tangent vectors are X in T_c C. |
| Principal bundle | P to C with right principal G-action, where G is a finite-dimensional Lie group. |
| Sector representation | rho_a:G to GL(V_a), with the same G acting on both sectors. The derivation is representation-independent. |
| Displayed agent frames | U_i in G, with displayed sector coordinates obtained by rho_a(U_i). Under independent passive display changes U_i becomes g_i U_i. |
| Reference link | Q_ij = rho_a(U_i)^(-1) Theta_ij rho_a(U_j), an endomorphism of the common V_a. |
| Reciprocal and backtrack data | R_ij = Q_ij (Q_ji Q_ij)^(-1/2), N_j = Q_ji Q_ij, on the open spectral domain avoiding the nonpositive real axis. Thus R_ji=R_ij^(-1). |
| Law data | p in an admitted finite-dimensional smooth family N of normalized laws, with a smooth G-action by pushforward and nondegenerate Fisher metric where it is used. |
| Optional interfaces | O:H to V and T:V to H, with H a declared fixed message space carrying a fixed metric and trivial G-action. Other actions on H must be included explicitly. |
| Augmented target | A smooth manifold M of tuples Z=(p,(R_e),(N_e),(O_i),(T_i)), with any subset of these components admitted. It carries a declared smooth positive-definite Riemannian metric H. |
| Left data action | h acts by p to h#p, Q to rho(h) Q rho(h)^(-1), O to rho(h)O, and T to T rho(h)^(-1). |
| Infinitesimal action | J_Z xi = d/dt at zero of exp(t xi) acting on Z. |
| Alignment map | T(v,u) in G sends reference coordinates at u to reference coordinates at v. This symbol is distinct from an interface T_i. |
| Connection convention | D=d+A and transport has derivative minus A: T(c+epsilon X,c)=e-epsilon A_c(X)+O(epsilon^2). |

In formulas below rho and its differential are suppressed when the represented group is faithful. With several sectors, every matrix expression uses the appropriate rho_a and rho_{a*}; all terms pair with one common xi in the Lie algebra of G.

The reference reciprocal and backtrack matrices transform by simultaneous conjugation. This follows directly from their displayed endpoint laws and the frame dressing. The principal inverse square root is smooth on its stated spectral domain and respects similarity, so this extraction preserves smoothness and covariance. Its output need not belong to a smaller chosen structure group: it is sufficient here that it be a point of the endomorphism component of M. The explicit example below uses group-valued reciprocal links throughout and has N=I.

The essential additional assumptions are that the data field Z:C to M is C^2 in local coordinates and is the local expression of a section of P times_G M; that the G-action is by isometries of H; and that J_Z is injective along the region under consideration. These conditions are not asserted for arbitrary agent configurations. Fixed agent labels and link components are used on the region; changing census or spectral branch requires separate treatment.

## Local construction

Fix c_0 and let z_0=Z(c_0). For endpoints near c_0 define

\[
 f(v,u,g)=\frac12 d_{\mathcal M}(Z(v),g\cdot Z(u))^2.
\]

Here d_M is the intrinsic distance of the declared metric H. Near (z_0,z_0), squared distance is smooth on a sufficiently small normal neighborhood. Restrict g to a coordinate neighborhood of e. At v=u=c_0, the identity is stationary and its group Hessian is

\[
 K_{z_0}(\xi,\eta)=H_{z_0}(J_{z_0}\xi,J_{z_0}\eta).
\]

To check this Hessian, the displacement between z_0 and exp(t xi)z_0 is t J_z0 xi+O(t^2); hence the squared-distance expansion is t^2 H(Jxi,Jxi)/2+O(t^3). Polarization gives the bilinear Hessian. Injectivity of J makes K positive definite.

Apply the following local implicit-function statement to the group stationarity map: if F(a,b) is smooth, F(a_0,b_0)=0, and its partial derivative in a is invertible at that point, then there are neighborhoods on which the equation F(a,b)=0 has a unique smooth solution a=a(b). Here a is a Lie-group coordinate, b=(v,u), and the invertible partial derivative is K. Continuity keeps the Hessian positive after shrinking. Consequently there is a unique smooth local minimizing branch T(v,u) through T(c_0,c_0)=e. For every nearby c, e is stationary at (c,c); local uniqueness gives

\[
 T(c,c)=e.
\]

This is local minimality on the selected branch. If G is compact and the action is free at z_0, the diagonal cost has unique global minimum e. Compactness supplies a positive gap outside a small neighborhood of e; continuity preserves a smaller gap for nearby endpoints. Together with the local Hessian argument this gives a unique global minimizer for nearby endpoint pairs. Noncompact groups need separate attainment and no-escape assumptions for a global assertion. Discrete stabilizers do not prevent the local theorem, but they produce several diagonal global minimizers.

Because the action is isometric,

\[
 f(v,u,g)=f(u,v,g^{-1}).
\]

Both sides use branches through e, so after shrinking their neighborhoods,

\[
 T(u,v)=T(v,u)^{-1}.
\]

This reversal law concerns the constructed contextual comparison. It imposes no condition on unrelated directed attending edges.

## Infinitesimal connection and covariance

Define A_c(X)=-d_v T(v,c)|_{v=c}(X). Differentiability makes this linear in X; smoothness gives a C^1 Lie-algebra-valued one-form. Put v=c+epsilon X in a chart. Then

\[
 Z(v)=Z(c)+\epsilon dZ(X)+O(\epsilon^2),
 \qquad T(v,c)=\exp[-\epsilon A_c(X)+O(\epsilon^2)].
\]

The first-order residual is epsilon(dZ(X)+J_Z A_c(X)). Stationarity to this order is therefore exactly

\[
 \boxed{K_Z(A_c(X),\xi)=-H_Z(dZ(X),J_Z\xi),\qquad
 K_Z(\eta,\xi)=H_Z(J_Z\eta,J_Z\xi).}
\]

This is the existing collective normal equation applied to the augmented tuple. Links enter through their contextual derivatives and their commutator orbit directions. In a block-diagonal metric its link contribution is

\[
 K_{\mathrm{links}}(\eta,\xi)
 =\sum_e\langle[\eta,Q_e],[\xi,Q_e]\rangle_e,
 \qquad
 b_{\mathrm{links}}(X,\xi)
 =\sum_e\langle dQ_e(X),[\xi,Q_e]\rangle_e.
\]

Law terms are g^F(J_p eta,J_p xi) and g^F(dp(X),J_p xi). Interface orbit tangents are xi O and -T xi; their metric pairings enter in the same way. If the metric has mixed blocks, the compact full H formula applies without dropping them.

For a smooth left coordinate change h(c), isometry identifies the re-expressed problems by

\[
 Z'(c)=h(c)\cdot Z(c),\qquad
 T'(v,u)=h(v)T(v,u)h(u)^{-1}.
\]

Local uniqueness proves this equality on the corresponding transported branches. Differentiation at v=u=c gives

\[
 \boxed{A'=\operatorname{Ad}_h A-dh h^{-1}.}
\]

The manuscript uses a common reference rechoice U_i'=U_i k, so its reference tuple transforms by the left action h=k^{-1}. In that notation the same result is

\[
 a'=k^{-1}ak+k^{-1}dk.
\]

Thus the displayed sector connection is

\[
 B_i^a=\rho_a(U_i)\rho_{a*}(a)\rho_a(U_i)^{-1}
       -d\rho_a(U_i)\rho_a(U_i)^{-1}.
\]

These formulas distinguish arbitrary coordinate changes from an active model symmetry. A fixed Frobenius metric on matrices is not invariant under arbitrary GL conjugation. The isometry hypothesis must be checked against the actual metric; transporting the metric tensor along a passive re-expression is allowed, while declaring that tensor fixed is an additional symmetry restriction.

On overlapping charts the same endpoint argument gives the affine law with the bundle transition g_ab. The cocycle identity g_ab g_bc=g_ac then makes the locally unique forms glue to a connection on P. A global frame is not required. Existence of an everywhere regular augmented section is itself a geometric condition: for example, a section valued in a free transitive G-space trivializes P, and a nowhere-zero mean in a complex line trivializes that line. The construction does not establish such a section on arbitrary bundle topology.

The derivative determines path transport by the ODE dP/dt=-A(gamma_dot)P, P(0)=I. Pairwise distance minimizers need not compose as T(w,v)T(v,u)=T(w,u). Their first derivative agrees with this connection, but a finite alignment map is not automatically its exact transport along a prescribed finite curve. Curvature is F=dA+A wedge A; nonzero F implies local path dependence. Flatness alone does not rule out global holonomy.

## A fully explicit curved connection from reciprocal links

Take G=SO(3), identify its Lie algebra with R^3 by a to hat(a), where hat(a)b=a cross b, and use the invariant inner product

\[
 \langle\widehat a,\widehat b\rangle=-\tfrac12\operatorname{tr}(\widehat a\widehat b)=a\cdot b.
\]

Let the supplied base be C=(0,pi) times (0,pi), with coordinates (x,y). Fix beta in (0,pi). For three agents with identity displayed frames, choose two reciprocal reference pairs

\[
 Q_1(x,y)=R_x(x),\qquad
 Q_2(x,y)=R_z(y)R_x(\beta)R_z(-y),
\]

\[
 \Theta_{12}=Q_1,\quad\Theta_{21}=Q_1^{-1},\qquad
 \Theta_{13}=Q_2,\quad\Theta_{31}=Q_2^{-1}.
\]

All backtrack loops are I, their principal roots are I, and their reciprocal parts are exactly the given links. The links can be written as exponentials of the smooth skew generators x hat(e_1) and beta hat(n(y)), with n(y)=(cos y,sin y,0). All agent laws may be normalized isotropic N(0,I_3); they are fixed by SO(3) and contribute no orbit or contextual tangent. The construction therefore obtains its nontrivial connection entirely from the changing links, with no supplied base one-form.

On M=SO(3)^2 choose the product bi-invariant Riemannian metric whose right-trivialized tangent inner product is the one above. This is a declared link-alignment metric, not the Fisher metric of the constant agent laws. For each r set

\[
 C_r=I-\operatorname{Ad}_{Q_r},\qquad V_r=dQ_r Q_r^{-1}.
\]

The right-trivialized conjugation tangent is C_r xi, so the exact normal equation is

\[
 K A=-\sum_{r=1}^2 C_r^*V_r,\qquad K=\sum_{r=1}^2 C_r^*C_r.
\]

For a rotation of angle alpha about unit axis n, Ad_Q is the same rotation on R^3. Orthogonality gives

\[
 C^*C=2I-Q-Q^T=4\sin^2(\alpha/2)(I-nn^T).
\]

Write w_x=4 sin^2(x/2) and w_beta=4 sin^2(beta/2). Then

\[
 K=w_x(I-e_1e_1^T)+w_\beta(I-n(y)n(y)^T).
\]

Its quadratic form is w_x times the squared component perpendicular to e_1 plus w_beta times the squared component perpendicular to n. Both weights are positive and the axes are nonparallel for 0<y<pi. A vector in the kernel would have to be parallel to both axes, hence be zero. Therefore K is positive definite everywhere on C. In fact, a rotation fixing both oriented nonparallel axes is identity, so the SO(3) action on this pair is free. The compact-group global-minimum argument applies to nearby endpoint pairs.

Direct differentiation yields

\[
 V_1=\widehat e_1 dx,\qquad C_1^*V_1=0,
 \qquad V_2=C_2\widehat e_3 dy.
\]

The first equality is differentiation of a rotation about its own axis. For the second link, differentiating its conjugation by R_z(y) gives hat(e_3)-Ad_Q2 hat(e_3). Both axes are perpendicular to e_3, so

\[
 K\widehat e_3=(w_x+w_\beta)\widehat e_3,\qquad
 C_2^*V_2=w_\beta\widehat e_3 dy.
\]

It follows that

\[
 \boxed{A=-\frac{w_\beta}{w_x+w_\beta}\widehat e_3 dy.}
\]

Here A wedge A=0 because the only component is proportional to dy. Since dw_x/dx=2 sin x,

\[
 \boxed{F=\frac{2w_\beta\sin x}{(w_x+w_\beta)^2}
              \widehat e_3 dx\wedge dy\ne0.}
\]

This is an exact algebraic witness. The interaction graph can be a three-agent tree, with every edge reciprocal and every graph backtrack trivial. Its contextual link variation nevertheless induces nonzero base curvature under the stated alignment rule. No numerical experiment or finite-test inference is used.

## The stabilizer and center obstruction

For a general tuple the infinitesimal kernel is exactly

\[
 \ker J_Z=\{\xi:J_p\xi=0,\ [\rho_{a*}(\xi),Q_e]=0\ \text{for every link},
 \ \rho_*(\xi)O_i=0,\ T_i\rho_*(\xi)=0\ \text{for every retained interface}\}.
\]

With positive-definite H, K has this same kernel. At constant rank, the normal equation always has a solution because its right side annihilates the kernel, and solutions form an affine class A+ker J_Z. Its residual dZ+JA is unique. A smooth complement gives smooth local representatives, but an arbitrary pseudoinverse convention need not satisfy the affine connection law. A full connection then requires a covariant additional selection or more data. Changes of rank are outside this regular construction.

There is a stronger impossibility for an ineffective subgroup. Suppose a positive-dimensional subgroup K_0 acts trivially on every admitted input tuple. Any smooth contextual k(c) in K_0 leaves the entire data field, including all its derivatives, unchanged. A deterministic connection selector from only that field would return the same A. Endpoint covariance instead requires

\[
 A'=\operatorname{Ad}_{k}A-dk k^{-1}.
\]

At a chosen c_0 take k(c_0)=e and arbitrary nonzero dk(c_0) in the Lie algebra of K_0. Then the required A'(c_0) differs from A(c_0), a contradiction. The analogous finite contradiction uses k(u)=e and k(v) nonidentity: unchanged data would have to produce both T(v,u) and k(v)T(v,u).

For G-valued links under conjugation, K_0 contains the center of G. For arbitrary endomorphism links it contains the subgroup whose represented matrices commute with every admitted link. In defining GL(d), nonzero scalar matrices always lie in this kernel. Under the scalar complex U(1) action, every complex-linear endomorphism commutes with the phase, so all such link data are blind to the full U(1). These statements must not be extended to a multi-charge U(1) representation or arbitrary real-linear interfaces without checking their actual commutants.

Thus conjugation-only links cannot select the GL scalar connection component or the scalar U(1) connection. Generically they may determine a projective part; constructing it still requires an appropriate quotient group, metric, and regularity. For U(1) itself, conjugation-only group links have J=0 and supply no connection component at all. Choosing the zero representative in every chart does not fix the problem: it violates the inhomogeneous transformation term under nonconstant phase changes.

Law or interface data can remove this kernel, but only when their action sees it. For a real GL scalar generator xi=tI acting on a nondegenerate Gaussian N(mu,Sigma), its law tangent is (t mu,2t Sigma), whose Fisher squared norm is

\[
 t^2\bigl(\mu^T\Sigma^{-1}\mu+2d\bigr)>0\quad(t\ne0).
\]

Gaussianity is only an explicit witness. Any admitted regular law family with nonzero scale orbit tangent suffices. For scalar U(1), a proper centered complex Gaussian remains phase-invariant, whereas a nonzero complex mean is moved by phase and removes that infinitesimal ambiguity. A retained nonzero receiving vector O from a fixed trivial message space also sees xi=i: xi O=iO is nonzero. If the message space is allowed to transform by the same phase, the interface may transform only by conjugation and this resolution disappears. The message representation must therefore be declared.

## A GL-compatible augmented metric

One may assume an invariant positive Riemannian metric on the admitted augmented target, but its existence should not be smuggled in through a fixed Frobenius norm. Here is an explicit real-linear choice. Let p vary in a regular family of laws on R^d, stable under the admitted GL action, and suppose its covariance Sigma(p) is smooth and positive definite. For tangent matrices B,C define

\[
 \langle B,C\rangle_\Sigma
 =\operatorname{tr}(\Sigma B^T\Sigma^{-1}C).
\]

This is the Frobenius pairing of Sigma^(-1/2) B Sigma^(1/2) with the analogous C. It is positive definite and invariant under simultaneous transformations Sigma to h Sigma h^T, B to hBh^(-1), C to hCh^(-1), by substitution and cyclicity of trace. Therefore

\[
 H((\dot p,(\dot Q_e)),(\dot p',(\dot Q'_e)))
 =g_p^F(\dot p,\dot p')+
   \sum_e\operatorname{tr}(\Sigma\dot Q_e^T\Sigma^{-1}\dot Q'_e)
\]

is a smooth positive Riemannian metric on the product family and has an isometric GL action. Its first block is Fisher, but the full metric is a declared link-augmented metric rather than automatically the Fisher metric of p alone.

If a fully normalized statistical realization is desired, one may separately declare the following probe experiment. Observe Y_0 with law p, independently observe X with law N(0,Sigma(p)), and conditionally independently observe Y_e given X with laws N(Q_e X,Sigma(p)). Every factor is normalized. With m links its Fisher metric is

\[
 H_{\mathrm{probe}}
 =g_p^F+
   \frac{m+1}{2}\operatorname{tr}(\Sigma^{-1}d\Sigma\Sigma^{-1}d\Sigma)
   +\sum_e\operatorname{tr}(\Sigma dQ_e^T\Sigma^{-1}dQ_e),
\]

where the quadratic notation denotes the associated polarized bilinear form. To verify it, the conditional mean score for Q_e has conditional variance E[(dQ_e X)^T Sigma^(-1)(dQ_e X)], giving the displayed link term. Scores for distinct outputs are conditionally centered and independent. Mean scores have zero pairing with covariance scores because centered Gaussian third moments vanish. Each of X and the m conditional outputs contributes one covariance Fisher term with coefficient 1/2. Scores from the independent Y_0 observation supply g_p^F and have zero cross pairing with the other centered scores. Transforming every observed vector by h makes the whole experiment the pushforward of the original experiment, so its Fisher metric is GL-invariant. This is an optional added observation model, not an experiment already asserted by the manuscript or forced by the links.

## Supported result and remaining obligation

The local theorem constructs a smooth identity-at-zero displacement family with the required endpoint law and a connection-valued first derivative whenever the augmented metric is isometric and its infinitesimal action is injective. The SO(3) example proves that smooth reciprocal agent links can contribute the entire connection and produce nonzero base curvature under a declared alignment criterion. Conjugation-only data cannot determine components belonging to an ineffective subgroup; scalar GL and scalar U(1) supply minimal counterexamples.

The remaining modeling obligation is selection of the tuple, its contextual continuation, and its metric or normalized probe experiment from the intended agent mechanism. The theorem supplies a viable constitutive construction, not uniqueness among all such constructions. A next decisive step is to choose which available law and interface data are actually retained and to check their combined stabilizer under the manuscript's intended structure group. No scripts, GPU computations, manuscript changes, or changes to Item 7 were made for this note.
