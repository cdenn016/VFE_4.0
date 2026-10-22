<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-f83979013626f813aee847ebad2dcca2c0fa4f2687c0daf601404aed35c64a6a","schema_version":"rigorous-theory-search/v1","target_digest":"f83979013626f813aee847ebad2dcca2c0fa4f2687c0daf601404aed35c64a6a"} -->
# Collective physical compression with complete histories and nonflat geometry

## Result and exact scope

A common physical rotation of all transmitted private-angle laws gives a genuinely collective quotient when the fixed protocol contains relative-angle probes on a connected graph. Rotating a proper block changes a probe across its boundary. Rotating the entire connected population does not. This differs from a private phase that is invisible on every primitive port: the proposed phase loss is unavailable to each proper block while its exterior is retained.

The geometric construction requires the parent state to be the entire contextual section, together with its complete history. At one fixed contextual basepoint c0, use the nonzero sum of the private natural vectors to choose a rotation. Apply that single rotation to every context. For each fixed history, the rotation is independent of the evaluation coordinate. It therefore preserves the actual joint history/current-state Fisher scores, their collective cost, the minimizing marked connection, and its contextual tensor. The parent discards one context-constant physical angle. It does not discard a separately chosen angle at every context, and it does not establish a local or fixed-size parent.

The companion `constructive-per-agent.md` supplies the normalized per-agent model, exact own-block equations, and the interaction-dependent selection estimate for this candidate. This note reconstructs the private-angle field needed for the quotient and proves its geometric and history compatibility. It gives a common nonflat window conditional on that route's uniform finite entrance estimate. The construction uses no probability distribution over law configurations: each initial contextual section is a deterministic preparation, and the histories are random under the declared normalized operational kernels.

The geometry assertion concerns a declared regular family indexed by the evaluation context. Variations of the entire prepared section, including a change in its erased common physical angle, are different parameter directions. The theorem does not preserve Fisher information in those outer directions. Second-stage recursion and every future_TODO extension are outside this assignment.

## Contract, sources, and assumptions

The frozen O1 target asks existentially for a finite population with normalized per-agent mixture-source free energies, positive fixed source priors, unit-temperature rows, a fixed complete protocol, interaction-dependent selection against its prescribed null, positive-dimensional nongauge physical loss, exact original-time inherited dynamics, every admitted finite adaptive response-record history, and actual positive-rank nonflat collective Fisher geometry. A permanently invisible primitive factor and a quotient available for every partition are excluded. A declared invariant family and a declared regular contextual family are allowed; all-contextual-section inheritance is not required.

The live source pass read Chapter 2's collective normal equation, fixed and moving channel score formulas, and geometry-retention conditions; Chapter 3's normalized model and own-block flow; Chapter 5's canonical parent, boundary square, memory distinction and formation standard; and Chapter 14's current phase-parent, joint-record geometry and remaining-obligation sections in `13_per_agent_inheritance.tex`. The preceding run's `record-geometry.md` was read as a scoped source. Its primitive permanently hidden phase is not used as this run's witness.

This route introduces the following additional hypotheses. The configuration is a C2 contextual section on a compact patch C, with a fixed c0 in C. A single realized history updates the whole section coherently by the original per-agent equations evaluated at each context, using the corresponding covariant record templates. Saved record outcomes are fixed physical/lab labels; the marked frame acts on their local direction embeddings, not on those history labels. The selected block is a whole connected component of the predeclared relative-probe graph; in the three-agent witness it is the whole population. Initial private natural vectors lie in a common open pointed cone uniformly over C. Records assimilated into the objective are private-angle independent. Relative-angle probes are predictions unless a separate admissible assimilation law is declared. Selection is reported at fixed predeclared census times 0 and T, rather than by exact continuously reported threshold-crossing times. Each assumption is used below.

## Typed dictionary and conventions

| Object | Type, action, and convention |
| --- | --- |
| Evaluation base | Compact contextual patch C, local coordinates c=(a,chi), fixed c0. This index is distinct from inference time t and from an outer variation of the entire prepared section. |
| Configuration | A section F:c maps to the finite population's positive recognition laws and unit-temperature Gibbs rows. The full state includes the complete realized history H, event labels, action labels, waiting times, and inherited constitutive data. |
| Belief sample | For each child, (z,phi,psi) in {-1,+1} x S1 x S1, with uniform sign and normalized Haar references. Presentation laws use their own proper references. |
| Private law | v_B(psi)=exp[B dot e(psi)]/I0(|B|), B in R2, e(theta)=(cos theta,sin theta). Cartesian B coordinates are regular at B=0. |
| Marked frame group G | S1 acting from the left on current phi samples and marked natural/constitutive vectors, including the local direction embedding of each saved record. It acts trivially on the lab record labels H, psi and B. The presentation sector carries no marked orbit; no belief/model group tie is imposed. |
| Physical group H_phys | One S1 acting by psi_i maps to psi_i+alpha and B_i maps to R_alpha B_i, simultaneously for every child and every context. It is separate from G. |
| Principal marked bundle | A local trivialization C x G; changes of marked frame obey the affine connection law. The actual Fisher minimizing form is denoted A^F_phi. |
| Collective private quotient | The quotient of whole sections by the context-constant H_phys action, with history retained. Its canonical rotation uses the anchor S_F=sum_i B_i(c0). |
| Statistical experiment | P^F0_{c,t}(dh,dy)=Pi^F0_c(dh) Q^{F0,h}_{c,t}(dy), with a fresh current recognition/source-label sample Y. Histories come from a fixed finite episode for the nonflatness calculation. |
| Metric and connection | Fisher metric of this normalized joint law. The common marked connection minimizes E[(s_X+ell_xi)^2]. It is distinct from the private angular velocity needed to keep the anchor canonical. |
| Transport | Marked parallel transport is generated by the actual A^F_phi and can be path dependent. Canonicalization is an endpoint law transformation on the section space, not a replacement for that transport. |

The left plane rotations are R_alpha. Let J0 be the matrix with entries (0,-1;1,0), so dR_alpha/dalpha=J0 R_alpha. A tangent Lie-algebra element of the marked S1 is a real scalar. All sample transformations below preserve the corresponding Haar references. No SPD metric, matrix exponential, retraction, or Gaussian closure is needed for the construction.

## The normalized private channel and its native field

Write the belief factorization as

\[
q_i(z,\phi,\psi)=q_i^{\rm ret}(z,\phi)v_{B_i}(\psi).
\tag{G1}
\]

Here q_i^ret is the positive finite-dimensional marked family used in the companion route. Its public log odds are L_i. The fixed self evaluator is Haar in psi. A nonself content channel copies z and psi and resamples phi from Haar. Its pushforward density is therefore nu_j(z)v_{B_j}(psi), relative to the uniform sign and Haar references; nu_j(z)=2 Pr_j(Z=z), rather than the probability mass. All kernels are normalized. Persistent assimilated likelihoods depend on z,phi and presentation variables, but not on psi.

For fixed peers, the private part of the source-weighted own derivative is, modulo constants,

\[
\log v_{B_i}(\psi)-\sum_{j\ne i}\beta_{ij}\log v_{B_j}(\psi).
\tag{G2}
\]

The Gibbs envelope removes derivatives of the optimizing row from this own derivative. The centered negative derivative is the Fisher replicator log-density velocity. Since log v_B is linear in the two sufficient statistics e(psi), matching its natural vector gives the exact equation

\[
\dot B_i=-B_i+\sum_{j\ne i}\beta_{ij}B_j.
\tag{G3}
\]

The source rows depend on private parameters through their norms and inner products. In the model of the companion route the nonself weight contains

\[
\frac{\exp[B_j\cdot M(B_i)]}{I_0(|B_j|)},
\qquad
M(B)=\mathbb E_{v_B}e(\Psi),
\tag{G4}
\]

and M(R_alpha B)=R_alpha M(B). Thus all rows and all public/marked velocities are invariant under a common physical rotation, while (G3) is equivariant. This is a symmetry of the fixed physical model, unlike rotating marked beliefs while holding their marked evaluator direction fixed. Source-role derivatives from other agents' objectives are not added.

Choose a convex pointed open cone K in R2 with aperture strictly below pi, and suppose B_i(c,0) lies in K, uniformly away from its boundary and zero on the compact patch. The union of this class over common rotations is invariant under H_phys. Equation (G3), with beta_ij positive, preserves K. Indeed

\[
e^t B_i(t)=B_i(0)+\int_0^t\sum_{j\ne i}\beta_{ij}(s)e^s B_j(s)ds.
\tag{G5}
\]

Positive linear combinations preserve the cone. If u is a unit vector strictly positive on its closure, then u dot B_i(t)>=e^{-t}u dot B_i(0)>0. This proves that every B_i and the anchor sum remain nonzero on every finite interval. It also supplies the regularity domain for all relative-probe and canonical formulas below. The same argument applies to private-angle-independent record episodes because their only effect on (G3) is through positive rows.

## A fixed complete relative-probe protocol

For every edge ij of a fixed connected graph, permit a relative-angle query. It draws fresh independent psi_i and psi_j from their current recognition laws and returns r=psi_i-psi_j modulo 2pi. Its normalized Haar density is

\[
k_{ij}(r|B_i,B_j)
=\int v_{B_i}(s+r)v_{B_j}(s)\frac{ds}{2\pi}
=\frac{I_0(|R_{-r}B_i+B_j|)}{I_0(|B_i|)I_0(|B_j|)}.
\tag{G6}
\]

The integral representation proves positivity and normalization; the Bessel expression follows by collecting the two linear circular exponents. It also proves that a common rotation leaves the entire response kernel unchanged. Its first moment is

\[
\mathbb E e^{ir}=m(|B_i|)m(|B_j|)
e^{i(\arg B_i-\arg B_j)},
\qquad m(b)=I_1(b)/I_0(b)>0\quad(b>0).
\tag{G7}
\]

A rotation by alpha of a nonempty proper block multiplies this moment by exp(i alpha), or its conjugate, on an edge crossing that block. Connectedness supplies such an edge. Since both amplitudes are positive, some alpha arbitrarily close to zero changes the kernel. A proper block therefore cannot erase this rotation while preserving the fixed relative probes and its exterior. A connected whole component can erase its common rotation. This proves collective availability of the proposed loss; it does not claim that every other conceivable compression of a proper block is impossible.

The protocol is fixed before selection. It contains all these relative queries, the public and marked queries and normalized evidence transactions of the companion construction, and adaptive choices based on retained history, fixed physical/lab protocol data, and independent external randomization. Every new transaction uses fresh samples. A joint relative query sharing primitive samples uses the normalized joint pushforward of their product recognition law, not a product of edge marginals; a common rotation leaves all those differences unchanged simultaneously. Predictions and assimilation are distinct operations. The relative query (G6) appends its observed value to history but does not introduce a new private-angle-dependent likelihood into the objective. Evidence transactions use the declared private-angle-independent family and remain in the objective. Source-label readouts, if admitted, use their actual normalized rows and do not silently change the inference convention.

Absolute-angle readouts are child diagnostics for assessing physical loss but are excluded from the boundary protocol at every stage. If an external absolute direction is admitted as a probe, or if the connected component has a relative-probe edge to a retained exterior, the quotient proposed here fails. No probe is removed at handoff. The whole-population physical symmetry exists before grouping, but its quotient is not independently available on every block of the initial partition. Thus the construction addresses collective compression; it does not claim that interaction creates a previously absent symmetry.

## The anchored quotient of the whole contextual section

For a section F define

\[
S_F=\sum_i B_i(c_0),
\qquad
C(S)=\frac1{|S|}
\begin{pmatrix}S_x&S_y\\-S_y&S_x\end{pmatrix},
\qquad S\ne0.
\tag{G8}
\]

This matrix is a globally smooth rotation on the punctured plane and sends S to (|S|,0). It does not require choosing a branch of arg. Define R(F,H)=(hat F,H) by

\[
\widehat B_i(c)=C(S_F)B_i(c)
\tag{G9}
\]

for every child and every c, leaving all other law parameters, primitive ports, records, times, labels and constitutive data intact. The parent belief laws are the normalized densities q_i^ret(z,phi)v_Bhat_i(psi), and the parent is their section-valued product with the retained presentation and source-label laws.

The identity C(R_alpha S)R_alpha=C(S) proves that R is constant on each common physical orbit. Conversely, if two canonical sections agree, their original sections differ by the single rotation C(S_F')^{-1}C(S_F), at every context. Thus each regular fiber is exactly one S1 physical orbit. The action is free on this family because its anchor is nonzero. The canonical slice S_y=0,S_x>0 gives a global cross-section on the anchored family.

The child diagnostic E exp(i psi_i)=m(|B_i|)exp(i arg B_i) changes along this orbit. The marked group G acts trivially on it. Consequently the retained section and complete realized history have lost one continuous nongauge physical distinction. This is one common angle of a whole section, not one angle per context. The retained section still contains all relative phases between children and all relative phases between different contexts.

For the exact inherited field, write V_i for the right side of (G3) and V_S=sum_i V_i(c0). Differentiating (G8) gives

\[
\Omega(F)=\frac{\det(S_F,V_S)}{|S_F|^2},
\qquad
\dot C(S_F)=-\Omega(F)J_0C(S_F).
\tag{G10}
\]

Since V is equivariant, the quotient field is

\[
\partial_t\widehat B_i(c)
=V_i(\widehat F)(c)-\Omega(\widehat F)J_0\widehat B_i(c).
\tag{G11}
\]

The other retained velocities are the original invariant velocities. The angular velocity in (G10) is computable from the retained section at c0, so (G11) is autonomous in the retained state and uses the original physical time. The compensating term is exactly DR applied to the original field. It is not an independently designed parent force. It need not be the own-block Fisher gradient of a newly fitted single-agent objective, and no such type-closure claim is made.

The same conclusion follows without differentiating: if F' is a common rotation of F, uniqueness and equivariance give Phi_t(F')=R_alpha Phi_t(F). Canonicalization therefore sends both to the same quotient trajectory. Equivariant record transactions descend in the same way. The parent is generally nonlocal in the evaluation coordinate because its canonical velocity uses c0; this dependence is part of the declared whole-section state.

## Exact finite adaptive histories, including the current-state experiment

Let an action specify a primitive port or relative edge, a normalized query or evidence transaction, and a waiting time. At a retained state, the full joint kernel of its observed response and next retained section is constant on every orbit fiber of (G9). Waiting descends by (G11). Relative responses descend by (G6). Public, marked and source-label responses descend because their laws and rows are invariant. Private-angle-independent evidence transactions commute with a common physical rotation and append the same realized record.

Induction over the causal product of these kernels proves equality of every admitted finite adaptive response-record path, including its dependence with the retained next state. Policies use the complete retained history, fixed physical/lab protocol data, and independent external randomization. An arbitrary additional read of frame-dependent internal coordinates is not admitted. All accessible action labels, waiting times, source marks and event data remain part of the retained state. No equality of response marginals is substituted for equality of the joint response/update kernel.

There are two different historical assertions. The factorization just proved is pointwise future inheritance conditional on the actual retained history at handoff. It does not, in general, imply that two arbitrary initial configurations that later have the same retained state generated that history with equal probabilities. To reconstruct an unconditional past-and-future law, initialize the parent with the actual pushforward law of the handoff state and history. For two initial whole sections related by the common physical rotation in this construction, the stronger equality of their entire history laws does hold, because every earlier kernel and update has the same symmetry. No preparation prior on sections is needed for either statement.

Fix a deterministic prepared initial section F0. For evaluation coordinate c, let Pi_c^F0 be the law of one fixed finite record/probe episode and let Q_{c,t}^{F0,h} be the product of the children's fresh current recognition and Gibbs-label laws after history h. The actual statistical family is

\[
\mathcal P^{F_0}_{c,t}(dh,dy)
=\Pi^{F_0}_c(dh)Q_{c,t}^{F_0,h}(dy).
\tag{G12}
\]

Its score in an evaluation tangent X is

\[
s_X=\partial_X\log\Pi_c(H)
+\partial_X\log Q_{c,t}^{H}(Y),
\qquad
g^{\rm joint}=g^{\rm history}
+\mathbb E_H g^{\rm conditional}.
\tag{G13}
\]

The mixed term vanishes because a normalized conditional density has mean-zero score. Source labels are included in Q. A fresh current sample together with retained history is the experiment in (G12); it is not the experiment consisting of one sample together with a record generated from that same sample.

For each fixed t,h, canonicalize the evolved whole section F_t^h by C(S_{F_t^h}). This matrix may depend on t, h and the prepared section F0, but it does not depend on the evaluation coordinate c. Define the sample map T_h by applying this rotation to every fresh psi coordinate and retaining H and all other sample coordinates. Then

\[
\widehat{\mathcal P}_{c,t}
=(\mathcal P_{c,t})T,
\qquad
T(h,y)=(h,T_hy).
\tag{G14}
\]

For the fixed prepared section and time under consideration, T is a bimeasurable, c-independent map with unit Haar Jacobian. The retained history itself determines which branch T_h is used in this comparison. It follows by differentiating the densities at fixed h that

\[
\widehat s_X(h,T_hy)=s_X(h,y).
\tag{G15}
\]

The constant-in-c assertion is essential. It concerns d_c of the evaluated section. A variation delta F of the entire prepared section generally changes the anchor and has D_F C(S_F)[delta F] nonzero. Equation (G15) does not assert score preservation for those outer variations. In particular it does not preserve the Fisher information of an outer parameter that rotates all physical psi laws.

Across all configurations, R is not one fixed Markov kernel acting only on a current sample Y: its rotation is a functional of the whole law section. For each prepared contextual family, however, (G14) is a fixed isomorphism of its joint sample spaces. This is why (G15) proves actual contextual statistical inheritance without turning the nonlinear configuration retention into an ordinary sample statistic. An inverse sample map depending on the erased preparation angle is not a reconstruction of that physical angle from the parent configuration.

## The actual minimizing marked connection and tensor

The following is an explicit convention for this witness, not an identification with the previous numeric-history-angle action. A saved outcome omega is a fixed lab label in H. Its local direction embedding is a unit vector d_omega(c), initially e(omega) in the chosen trivialization. The marked likelihood is exp[lambda d_omega(c) dot e(phi)]/I0(lambda). Under the passive marked frame g(c), transform d_omega to R_g d_omega, the evaluator direction n to R_g n, A,C to R_g A,R_g C, and the fresh sample phi to phi+g. Keep the lab label omega fixed. In the transformed frame,

\[
\ell_g(\omega|\phi)
=\frac{\exp[\lambda R_g e(\omega)\cdot e(\phi)]}{I_0(\lambda)}
=\ell(\omega|\phi-g).
\tag{G16a}
\]

Integration over the same fixed Haar lab labels proves normalization. The self evaluator transforms by the analogous change of phi variable, and the resampling channel remains normalized Haar. Changing variables in the normalized predictive integral shows that its law of the lab record is unchanged.

Every stored marked query output uses the same convention. Let d_c map the lab circle to the local marked circle by its retained rotational direction embedding. A direct marked query samples local phi and reports the lab label d_c^{-1}(e(phi)). After a frame change, d_c'=R_g d_c and phi'=phi+g, so the reported label is exactly unchanged. This still identifies the direct marked law because the embedding is known and invertible. The fresh CURRENT Y_phi used in (G12) is a local sample coordinate and does transform; its law is distinct from an earlier lab-encoded query result stored in H. Public, model and source labels are already frame invariant, as are private relative-angle outcomes. Thus all stored outcomes, including direct marked queries and evidence labels, are fixed by the marked action.

The normalized kernels and history-based policies therefore give the same law Pi_c(H) under every finite passive frame change of the constitutive data and current laws. Its contextual derivative with respect to c need not vanish. Arbitrary policies that inspect raw frame-dependent internal coordinates would require an additional covariance condition and are not part of the declared protocol.

At fixed H, all KL costs and rows are unchanged; the evolved private vectors B_i and their anchor are therefore unchanged. The marked group on the actual joint sample space is (H,Y_phi,Y_psi) maps to (H,Y_phi+g,Y_psi), with unchanged source labels. It carries the actual joint density to the one generated by the transformed normalized model. It also commutes with T_H in (G14), even when the anchor depends on the saved outcomes. This supplies the orbit-score hypothesis that contextual score isometry alone would not supply.

This is passive frame covariance. Rotating beliefs while holding the evaluator and the local record embeddings fixed is generally not an active symmetry of the fixed-data energy. The physical private rotation is a separate symmetry. Let ell_xi be the actual marked orbit score of the joint family under the just-declared action. Commutation now gives

\[
\widehat\ell_\xi(h,T_hy)=\ell_\xi(h,y).
\tag{G16}
\]

Equations (G15)-(G16) establish, for every evaluation tangent X and every marked comparison velocity xi,

\[
\widehat{\mathcal C}(X,\xi)
=\mathbb E_{\widehat{\mathcal P}}[(\widehat s_X+\widehat\ell_\xi)^2]
=\mathbb E_{\mathcal P}[(s_X+\ell_\xi)^2]
=\mathcal C(X,\xi).
\tag{G17}
\]

Thus the actual locked inertia, momentum and contextual quadratic form agree before minimization. The minimizing connection classes, orbit isotropy and minimized tensor agree. On a locally free marked patch, their unique local forms and their curvatures agree. No Fisher tensor has been attached to retained metadata by prescription: every term is calculated from the normalized parent experiment (G14).

On overlaps, a marked frame change g(c) gives A_phi' = A_phi-dg in the abelian convention. The canonical private quotient commutes with the corresponding transformations of record-embedding and constitutive sections, so both sides obey the same affine law. The nonflat marked transport remains path dependent. The physical anchor velocity Omega in (G10) is a different object and is not substituted for A_phi or its curvature.

If the history experiment is conditioned on a positive-probability public record pattern E, replace Pi_c by Pi_c(.|E). Its score includes -partial_X log Pr_c(E). The parent has the same event probability and the same normalized conditional law, so (G14)-(G17) hold including this term. The angular records and any relative-probe outcomes remain random.

Exact continuous threshold-crossing times require separate care. A deterministic crossing time tau(c) retained as an exact observation has law delta_tau(c), which need not be differentiable in quadratic mean when tau varies. It cannot simply be inserted into a regular density-score proof. Here the selector is sampled at predeclared census times 0,T, using the same state graph. The strict uniform margins of the companion construction make the reported partition labels constant over the admitted contextual patch, conditional on E, while the times are fixed. These event data are retained and have zero contextual score. This changes neither the native flow nor the probes and inserts no erasure gate. No DQM geometry claim is made for continuously reported parameter-dependent crossing times.

## A common finite nonflat window with two small parameters

Use the companion construction with private scale delta>0, record scale epsilon>0, and marked self coupling J=epsilon^3. Write B_i(c,0)=delta b_i(c), where the fixed C2 functions b_i form a compact patch in a common pointed cone. Choose fixed small positive marked amplitudes a_-<a_+ and d, independently of delta and epsilon, and set

\[
A_1(0)=a e(\chi),\quad a\in[a_-,a_+],
\qquad A_2(0)=d e(0),
\qquad C_i(0)=0.
\tag{G18}
\]

Other selected marked factors can have fixed positive amplitudes. Presentation laws may initially be uniform. Public initial log odds are epsilon times the fixed compact initial neighborhood in the companion selection proof. That route gives a common finite entrance time T and a strict persistence window, uniformly over these compact marked/private patches and all angular record outcomes, for sufficiently small delta and epsilon. Let T_+ contain T and a positive subsequent interval. No additional assimilation is performed during this geometry window, although the fixed protocol continues to admit it later.

At delta=epsilon=0, the private laws are Haar, public laws are uniform, Gibbs rows equal their fixed priors, and the fixed initial weak record episode is uniform. Relative queries made using fresh pair samples also have uniform densities at this limit. History and source-label scores vanish at the limiting parameter. The marked laws evolve by A_i(t)=e^{-t}A_i(0). Consequently define

\[
I(x)=xm(x),\qquad V(x)=\operatorname{Var}_{v_x}(\cos\phi),
\qquad D(t)=\sum_{i\ne1}I(e^{-t}|A_i(0)|)>0.
\tag{G19}
\]

The complete limiting marked geometry is

\[
\begin{aligned}
A^F_{\phi,0,t}
&=-\frac{I(e^{-t}a)}{I(e^{-t}a)+D(t)}d\chi,\\
F^F_{\phi,0,t}
&=-\frac{D(t)e^{-t}I'(e^{-t}a)}
{[I(e^{-t}a)+D(t)]^2}da\wedge d\chi,\\
h^{\min}_{aa}&=e^{-2t}V(e^{-t}a)>0,\\
h^{\min}_{\chi\chi}
&=\frac{I(e^{-t}a)D(t)}{I(e^{-t}a)+D(t)}>0,
\qquad h^{\min}_{a\chi}=0.
\end{aligned}
\tag{G20}
\]

These formulas follow by minimizing I(e^{-t}a)(dchi+xi)^2+D(t)xi^2. Differentiation of I0 gives m'(x)=V(x)>0 and I'(x)=m(x)+xV(x)>0. The phase information identity x^2 E sin^2 phi=xm(x) follows by integrating the derivative of sin(phi)exp[x cos(phi)] around the circle. Thus the locked inertia, the two tensor eigenvalues and the absolute curvature coefficient have positive minima on the compact patch and 0<=t<=T_+.

The anchor is singular at delta=0, but the fine statistical family and its vector field are regular in Cartesian B coordinates there. The continuity argument is made for the fine geometry first, without evaluating the parent at delta=0. The normalized finite-episode kernels, laws and ODE coefficients have bounded derivatives on a common regular finite-time tube. Differentiating the flow once and twice in c yields linear variational equations with bounded coefficients and an inhomogeneous term quadratic in first derivatives. Their integral inequalities give uniform C2 convergence as delta,epsilon tend to zero. Compact episode spaces and uniform positivity justify differentiation of the Fisher integrals. Conditioning on E is regular because its limiting probability is positive.

The locked inertia K, momentum j and contextual quadratic form B in C(X,xi)=B(X,X)+2j(X)xi+K xi^2 therefore converge with the derivatives needed for A=-K^{-1}j and F=dA. The lower bounds in (G20) persist. First choose a sufficiently small fixed positive delta, then a sufficiently small positive epsilon, within the common grouping thresholds. The actual fine joint tensor has rank two and its marked connection is nonflat throughout the common post-census window. For positive delta the anchor is regular by (G5), and exact equality (G17) transfers this actual geometry to the parent.

No numerical threshold for delta or epsilon is asserted. Fixed positive initial marked amplitudes and a fixed finite T_+ are essential to this argument. The theorem does not infer nonflatness at an unbounded time from finite-time continuity. Later adaptive private-angle-independent records preserve the quotient closure, but can change the selected group and the geometry; no universal nonflatness assertion is made for every later experiment.

## A sharp fixed-coarsening test and what records cannot repair

The collective section construction above is not an ordinary fixed many-to-one sample statistic. For comparison, the following exact test applies when a proposed parent really is a fixed equivariant Markov coarsening T of the fresh sample, retaining H unchanged. For a randomized kernel, lift to the joint law P(dh,dy)T(y,dz); for a deterministic statistic set Z=T(Y). Work in that joint L2 space and let P be conditional expectation onto (H,Z). Write s_X for the fine contextual score, L xi for its marked orbit score, A_f for the fine minimizing representative, and r_X=s_X+L A_f(X) for its horizontal residual. Assume constant fine and coarse orbit ranks and compare the same Lie algebra convention.

The coarse scores are P s_X and P L xi. Let A_c minimize their cost. Orthogonality of the two minima gives the exact identity

\[
h_f(X,X)-h_c(X,X)
=\|(1-P)r_X\|^2
+\|PL(A_c-A_f)(X)\|^2.
\tag{G21}
\]

To check it, decompose r_X=P r_X+(1-P)r_X, and write P r_X=r_X^c+PL(A_f-A_c)(X). The latter two terms are orthogonal by the coarse normal equation. Taking squared norms proves (G21).

Consequently equality of the minimizing tensor is equivalent to measurability of every fine horizontal residual from (H,Z). For sufficiency, measurability gives E[r_X PL xi]=E[r_X L xi]=0, so A_f already satisfies the coarse normal equation and its residual norm is unchanged. Necessity follows from the two nonnegative terms in (G21). Equality of the entire minimizing affine classes additionally requires ker(PL)=ker(L), so that no new orbit isotropy was introduced. This is a necessary and sufficient condition on the declared contextual section, not on all possible sections. Full orbit Fisher information need not be preserved; the condition concerns the residual and orbit isotropy.

Because H is retained, (1-P) annihilates every history-only score. Splitting the orbit score into history and conditional-current parts gives

\[
(1-P)r_X
=(1-P)\bigl[s_X^{\rm current}+L^{\rm current}A_f(X)\bigr].
\tag{G22}
\]

History information cannot compensate for a conditional-current residual that remains hidden. Records can change A_f, and thereby which residual must be measured, but a positive conditional-variance defect after that optimization is irreducible. Conversely, if the residuals are measurable and isotropy is unchanged, nonzero curvature can be inherited; curvature itself does not obstruct a sufficient statistic.

If one strengthened the requirement to every contextual first jet in a positive Fisher chart, a physically non-gauge vertical direction of a retention would violate tensor equality: its coarse derivative is zero while its fine horizontal norm is positive. That stronger premise is not part of O1. Restricted regular nonflat families, including the whole-section construction above, are permitted. A result on them must not be promoted to information preservation for the erased outer physical parameter.

## Failure cases for records and pointwise canonicalization

First, preserving a connection or preserving some nonflatness does not prove tensor inheritance. Let two retained marked variables have the nonflat von Mises geometry in (G20) at t=0. Add a current real sample V distributed as N(a,1), on which the marked group acts trivially. Before compression record n observations H_k=V+noise_k, with independent N(0,sigma^2) noise and sigma>0, and retain all H_k. The fine experiment includes V and H; the proposed coarse experiment discards V but retains the marked variables and H. Its missing a-score is the residual of V-a given H. Direct Gaussian completion of the square gives

\[
\operatorname{Var}(V|H)
=\frac{\sigma^2}{\sigma^2+n}>0.
\tag{G23}
\]

Thus the minimized a-direction tensor loses exactly this amount, even though the marked connection and its nonzero curvature agree. Finite noisy records cannot supply the missing residual. This is a sample-coarsening counterexample, not a proposed configuration-compression witness. If the fresh current V were independent of the historical V conditional on a, its full unit Fisher contribution would instead be lost; the two experiments must not be conflated.

Second, pointwise canonicalization of a law section is not the anchor construction. For a smooth density p_c(y) and a context-dependent diffeomorphism T_c, put v_X=(DT_c)^{-1}partial_X T_c in fine coordinates. Differentiating the pushforward density at fixed output gives

\[
\widehat s_X(T_cy)
=s_X(y)-\frac{\operatorname{div}[p_c(y)v_X(y)]}{p_c(y)}.
\tag{G24}
\]

The second term is a moving-map score, not an optional frame convention. If T_c rotates psi by -alpha(c), it subtracts the corresponding physical phase score. For a concrete case take private vectors B_i(c)=b_i e(chi), all b_i>0, and retained marked laws with the nonflat (a,chi) geometry above. The native private field is equivariant; one can assess this family at any fixed regular time. Canonicalizing the private phase separately at each c erases chi from every private law. The fine joint chi score has an additional, conditionally centered private component with information

\[
\sum_i b_i m(b_i)>0.
\tag{G25}
\]

Relative-angle history kernels are independent of the common chi, so their scores cannot replace this term. The marked gauge acts trivially on psi. The canonical parent therefore loses (G25) from the minimized chi tensor. This is a physically meaningful loss under the declared comparison gauge even though relative boundary responses are preserved.

For the whole-section anchor at a fixed c0, the rotation is instead fixed at the phase at c0 and leaves the spatial relative phase chi(c)-chi(c0) in the retained section. Its evaluation derivative is retained, and the moving-map term in (G24) is zero because d_c T_h=0. If chi is promoted to an outer parameter rotating the whole prepared section, the term is nonzero again, as it must be. This distinction is the load-bearing geometric mechanism.

There is a separate history-action failure. Context independence of T_H does not imply orbit-score preservation if the chosen group rotates H and T_H is not equivariant. For example, take an angular history value h and a private conditional law v_b(psi-lambda sin h). A group rotating h while leaving psi fixed differentiates the conditional phase through -lambda cos h. Canonicalizing each conditional law by psi maps to psi-lambda sin h removes that orbit-score component, with positive missing inertia lambda^2 b m(b) E cos^2 H. The context scores may still be preserved. The lab-label/local-embedding convention in (G16a), in which H is fixed by the marked group, prevents exactly this failure. Without that convention, or an independently proved anchor equivariance condition, (G17) must not be claimed.

Finally, an absolute-angle query or a relative probe to a fixed external reference defeats the operational quotient before any geometric calculation. A private-angle-dependent persistent likelihood can also destroy common-rotation symmetry unless the likelihood and retained reference are transformed together as physical data. These are explicit changes of the fixed protocol, not harmless additions after handoff.

## Supported conclusion and remaining first-stage interface obligation

The anchored whole-section quotient provides a normalized, physically noninjective collective parent with an exact inherited original-time field and every declared finite adaptive history. On the regular contextual experiment it preserves the full actual Fisher cost, and therefore the minimizing marked connection and tensor. The fine nonflatness calculation and two-parameter continuation give a common nonflat selection window when combined with the companion route's uniform selection estimate and the fixed-census realization.

The remaining integration checks are concrete: use the companion model's exact positive rows and native field, verify its selected block contains the whole connected relative-probe component, retain the entire C2 section and coherent complete history, and impose the same finite-census and regular episode convention in the final theorem. This route does not provide a local or fixed-size parent, does not erase a separately varying physical phase at each context, and does not prove emergent creation of the global symmetry. Its physical loss is collective because no proper block can perform the same erasure across its retained boundary.

No numerical scripts, experiments, GPU operations, Lean work, further delegation, manuscript edits or publication were performed. Only this assigned route artifact was written in the O1 assignment. The next action that can change the O1 conclusion is the parent's reconstruction of this geometry/history theorem together with the companion normalized dynamics and selection theorem under precisely the same section-valued and fixed-protocol assumptions.
