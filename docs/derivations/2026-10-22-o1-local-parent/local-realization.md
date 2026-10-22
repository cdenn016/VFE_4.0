<!-- rigorous-theory-search-metadata {"contract_id": "contract-sha256-70f14344fcdcf4941082bb0826fdae50a6f48a09a54f2945f1984813d5dcb555", "schema_version": "rigorous-theory-search/v1", "target_digest": "70f14344fcdcf4941082bb0826fdae50a6f48a09a54f2945f1984813d5dcb555"} -->
# Finite reference memory and exact evaluation-local closure

## 1. What is local in this result

Fix an evaluation context c and the published reference context c0. There is an exact finite-dimensional system for the current law value and its contextual two-jet at c, coupled to one finite native configuration at c0, with the complete ordered history retained. No other contextual values enter its equations. This is a finite-reference, evaluation-local realization. It is not a retention initialized from the germ at c alone, and a two-jet at one c does not encode the law function at all other contexts. These distinctions are part of the theorem.

The original O1 experiment uses c as the parameter of a prepared statistical family. Its actions choose ports, waits, diagnostics and evidence transactions, rather than relocating an unknown contextual function. A new operation that moves to arbitrary uninitialized contexts would require their state or a section generator. Section 8 explains compatibility when local realizations are provided across C; `locality-obstructions.md` treats a single finite encoding of all of C.

## 2. Native coordinates, normalization and locality

For three agents let x collect (l_i,A_i,C_i,B_i,n_i), eight real coordinates per agent, so dim x=24. The published law, relative to a uniform sign and two normalized Haar circles, is

\[
q_i(z,\phi,\psi)=Z_i^{-1}\exp(l_i z/2+A_i\cdot e_\phi+z C_i\cdot e_\phi+B_i\cdot e_\psi),
\]

\[
Z_i=I_0(|B_i|)\{e^{l_i/2}I_0(|A_i+C_i|)+e^{-l_i/2}I_0(|A_i-C_i|)\}/2.
\]

The presentation probability is s_i(w)=exp(n_i w/2)/(2 cosh(n_i/2)). Retain exactly the published normalized evaluator, copying/resampling channels, Gibbs source labels and record likelihoods. In particular,

\[
L_i=l_i+\log I_0(|A_i+C_i|)-\log I_0(|A_i-C_i|),\qquad
\beta_{ij}=w_{ij}/\sum_r w_{ir},
\]

\[
w_{ii}=\frac{e^{Jt_i}}{5I_0(J)},\qquad
w_{ij}=\frac{2e^{L_jm_i/2+B_j\cdot M(B_i)}}{5\cosh(L_j/2)I_0(|B_j|)},
\]

where m_i=tanh(L_i/2), t_i=E[z cos phi], M(B)=grad log I0(|B|), and J=epsilon cubed. Presentation rows gamma use their exact original KL costs at the same unit temperature. The record coefficients are H_i=sum r o and Lambda_i=sum lambda d_omega(c), with the entire ordered word also retained. The original own-block Fisher field is

\[
\begin{aligned}
\dot l_i&=-l_i+\sum_{j\ne i}\beta_{ij}L_j+H_i,&
\dot A_i&=-A_i+\Lambda_i,\\
\dot C_i&=-C_i+\beta_{ii}J e_1,&
\dot B_i&=-B_i+\sum_{j\ne i}\beta_{ij}B_j,\\
\dot n_i&=-n_i+\sum_{j\ne i}\gamma_{ij}n_j.
\end{aligned} \tag{L1}
\]

In another marked frame e1 and d_omega are their retained local embeddings. Write this field as V(x,c,h). The source rows are functions, not new independent dynamical rows. Derivatives of V include all derivatives of these functions. Nothing in V reads x at neighboring contexts or any contextual derivatives. The current collective minimizing connection is a diagnostic and does not enter (L1). This context-local property, not a generic claim about jet truncation, is why a bounded jet closes.

All the sample laws are strictly positive and normalized on compact sample spaces, and their parameter derivatives are bounded on compact regular tubes. The evaluator and each recorded likelihood retain their original normalizers. No law over configurations is introduced. This is the invariant exponential family of the full own-law Fisher field derived in O1 C10-C15, rather than Euclidean descent of these natural parameters or descent of a sum of child objectives.

## 3. Finite reference state and quotient

Let a=x(c0). Keep the full 24-coordinate reference configuration, not just its anchor sum, because its source weights depend on its public, marked and presentation laws. Its record coefficients use the same saved lab outcomes and their embeddings at c0. Set

\[
S(a)=\sum_i B_i(a),\qquad
U(S)=|S|^{-1}\begin{pmatrix}S_x&S_y\\-S_y&S_x\end{pmatrix}.
\]

Let rho(U) rotate only the B coordinates. Canonical local and reference coordinates are

\[
y=\rho(U(S(a)))x(c),\qquad b=\rho(U(S(a)))a. \tag{L2}
\]

Thus sum_i B_i(b)=(|S|,0). Retain j_c^2 y, b, the current persistent record coefficients, their two-jets if their embeddings vary, fixed constitutive data and the full history h. Include the handoff time and both original census entries. No absolute reference angle or inverse rotation is retained.

The cone proof in O1 G5 applies to a and x. For a vector u positive on the initial cone, u dot B_i(t) is at least exp(-t)u dot B_i(0)>0 on every finite branch. Positive Gibbs weights persist after every finite admitted private-angle-independent evidence append. Therefore S stays nonzero, and U is smooth without choosing an angle branch. Every finite history has finite likelihood coefficients and finite-interval native solutions. No lower bound uniform over an unbounded number of arbitrarily strong records is asserted.

The identity U(R_alpha S)R_alpha=U(S) proves invariance under a common physical rotation of both local and reference laws, including their jets. Conversely, equality of (L2) implies their uncanonicalized local/reference tuples differ by one common rotation. The retained b cannot recover that erased angle. Complete boundary histories and the census rule are also invariant, so memory does not undo the physical loss. The local reference data retain the relative physical phase between c and c0. That retained phase is precisely information that a pointwise quotient would lose.

If the local and reference data come from full sections, this local retention also forgets values outside c and c0. Its sufficiency is for the fixed-context experiment and its two-jet, not for future arbitrary-context readouts from that one local state. We do not misidentify those additional forgotten function values as further universally sufficient compression of the whole section.

## 4. Exact autonomous field and its two-jet

Define V_S(b)=sum_i V_{B_i}(b,c0,h), J2=[[0,-1],[1,0]], and

\[
\Omega(b,h)=\frac{\det(S(b),V_S(b))}{|S(b)|^2}.
\]

Let K act as J2 on each B block and as zero on other coordinates. Differentiating U gives dot U=-Omega J2 U. Common physical equivariance of (L1) gives the closed equations

\[
\dot y=W(y,c,h;b):=V(y,c,h)-\Omega(b,h)K y,\qquad
\dot b=V(b,c0,h)-\Omega(b,h)K b. \tag{L3}
\]

They are exactly the published moving-anchor field evaluated at c and c0. The anchor constraint is invariant: the y-component of dot S(b) is V_S,y-Omega |S|=0. Equation (L3) has the original time coordinate. Its added term is the derivative of the retention, not a force chosen to fit a desired parent trajectory.

At fixed h, b and Omega have no derivative with respect to the evaluation coordinate c. This does not say that b has no dependence on the prepared section, on time, or on h. These are different derivatives. Write y_a=partial_a y, y_ab=partial_a partial_b y for coordinates on C. Between records,

\[
\dot y_a=W_y y_a+W_a,
\]

\[
\dot y_{ab}=W_y y_{ab}+W_{yy}[y_a,y_b]+W_{ya}y_b+W_{yb}y_a+W_{ab}. \tag{L4}
\]

All c partials here hold y,b,h fixed. Known constitutive embeddings and record coefficients supply their own c derivatives. In the original fixed trivialization there is no explicit c dependence in (L1), and these explicit partial terms vanish. In particular the private blocks contain -Omega J2 y_B,a and -Omega J2 y_B,ab, with no partial_c Omega term. The same-order jets of x and of the record templates are enough; no third contextual derivative is generated by this pointwise native field.

For d=2 the law part of j2 y has 24(1+2+3)=144 coordinates, with 24 reference coordinates before its one slice constraint and any consistency constraints. The ordered history has unbounded length. This count is a sufficient representation, not a minimality result or a fixed-memory compression claim. The original likelihood family has nine accumulated scalar/vector coefficients; their jets are functions of retained record data and known embeddings.

At an evidence transaction append its actual outcome and add its normalized likelihood to the objective. The recognition laws are not reset. Update the reference coefficient using that same lab outcome at c0, and differentiate the update through order two at c. At a diagnostic, append the outcome and change neither objective nor recognition laws. These prescriptions close (L3)-(L4) on every finite admitted branch.

## 5. Actual kernels and historical sufficiency

At c the parent recognition law is the product of the normalized q_i with B_i replaced by B_i(y), along with the original presentation and Gibbs-label factors. The reference b is internal deterministic memory, not an extra independent sample in the comparison experiment. Adding a sampled reference law would add unwanted orbit inertia; it is not part of this construction. Primitive ports select the same factors and apply the original channels. Generative predictions use the original normalized per-agent models in these coordinates.

A joint relative query draws each involved psi once and reports all requested differences. It is the pushforward of the product law on primitive vertices, retaining shared samples and cycle constraints. Common rotation preserves the entire pushforward. Public, marked-lab and model responses are unchanged, and all Gibbs rows are invariant. For the actual transaction kernel J, including any shared response/update randomness,

\[
J_{\rho(g)e}^{a}=(\mathrm{id}_{\rm record}\mathbin{\times}\rho(g))_*J_e^a.
\]

Combining this identity with (L2)-(L4) gives one well-defined normalized joint response/next-parent-state kernel. Equal parent states and h supply the same next action and wait law under every admitted history-based policy. Induction proves every finite adaptive joint history, including its dependence on future retained states. It is not an inference from separate response and transition marginals.

For a conditional future theorem, the actual history h and current retained state suffice. For the unconditional history/current Fisher experiment, the past law must also be the actual one. One finite sufficient initialization is to retain a replay seed consisting of the initial local two-jet and initial reference configuration in one common canonical frame, together with the declared protocol and complete h. Generate each hypothetical prefix using the original normalized kernel, propagate its local/reference native states, and apply the appropriate evidence append. This defines Pi_c(dh) by a finite causal product for each finite experiment. The seed is deterministic preparation data, not a prior on configurations. It is bounded in contextual jet order; the prefix length is not bounded.

Equivalently, a compatible parent initialization may be supplied as the actual pushforward of the handoff state and history experiment. Simply appending h to an arbitrary current tuple does not reconstruct its past probability. The explicit seed construction avoids that gap. The two-jet of Pi at c is obtained by differentiating these actual kernels; it is not an independently assigned history score or Fisher tensor. Conditioning on the initial public pattern includes its positive probability normalizer computed by the same preparation experiment.

## 6. Geometry from actual normalized laws

For a fixed prepared section, fixed time t, and each fixed history h, U(S(a_t^h)) depends on the reference evaluation at c0 and is constant in the evaluation coordinate c. On the actual fine joint experiment P_c(dh,dy)=Pi_c(dh)Q_c^h(dy), define T(h,y)=(h,T_h y), rotating only the fresh psi samples by U. It is bimeasurable with unit Haar Jacobian. The finite local program just described produces exactly

\[
\bar P_c=T_*P_c. \tag{L5}
\]

Stored lab outcomes are fixed by the marked passive group; their local direction embeddings transform. The physical reference and T are unchanged by that marked action. Thus T commutes with it even though T_h depends on h. At fixed h, differentiation of (L5) gives equality, under T, of the contextual score and the marked orbit score. Consequently

\[
\bar{\mathcal C}(X,\xi)=E_{\bar P}(\bar s_X+\bar\ell_\xi)^2
=E_P(s_X+\ell_\xi)^2=\mathcal C(X,\xi). \tag{L6}
\]

This includes the history score and the conditioning normalizer. It proves equality of the actual minimizing affine connection class, its isotropy and the minimized tensor. On the locally free marked window the unique connection and its curvature agree. No fine tensor is copied to memory. The phase contribution is carried by the actual B_i(y) dependence of the normalized parent law.

For an explicit check, take B_i(c)=b_i e_{chi(c)}. Pointwise canonicalization deletes the private chi score. Equation (L2) instead retains B_i(y)=b_i e_{chi(c)-chi(c0)}, whose actual chi information is sum_i b_i I1(b_i)/I0(b_i), exactly the missing term. Reference memory is doing statistical work in the density, not just recording a geometric label.

The published common grouping/nonflatness window is inherited directly because (L5) is the same actual law experiment at every c,t. In the published zero-parameter comparison the marked connection is

\[
\mathcal A=-\frac{I(e^{-t}a)}{I(e^{-t}a)+D(t)}d\chi,\quad
\mathcal F=-\frac{D(t)e^{-t}I'(e^{-t}a)}{[I(e^{-t}a)+D(t)]^2}da\wedge d\chi,
\]

with I(x)=x I1(x)/I0(x), D(t)>0, I'(x)>0, and minimized coefficients e^{-2t}Var(cos phi)>0 and I D/(I+D)>0. Compact finite-window margins persist at sufficiently small positive epsilon,delta in the original O1 model. We use the same parameters and preparation, not a new finite-jet ansatz or a numerical threshold. Later arbitrary evidence preserves exact inheritance but need not preserve this particular geometric window or grouping.

## 7. A frame frozen at handoff

There is another useful representation after a fixed census handoff T. Put U_T=U(S(a_T^h)) once and let z(c,t)=rho(U_T)x(c,t), t>=T. For each fixed handoff history this is constant in both c and subsequent t. Equivariance gives dot z=V(z,c,h), so its two-jet follows (L4) with W=V. The normalized z laws preserve all the same Fisher costs. No live reference state is required for its *future local evolution*.

This is a memoryful handoff representation, not an instantaneous pointwise quotient R(x(c,t)). Its initialization already incorporates the relative phase to the chosen reference. Do not re-canonicalize z separately at each c. If one requires the continuously reanchored published representative at later t, one must retain/compute the reference evolution or its accumulated phase. Equality of geometric costs between the two representatives at each fixed t does not itself make their state encodings identical. Past-law initialization still requires Section 5. The fixed initial common frame is a retained coherence condition, not an observable absolute psi reference added to the protocol.

## 8. Compatibility and exact scope

The reference rotation is a single smooth matrix for each t,h and prepared section. If local data at different c are initialized as two-jets of one source section with the same reference state, (L3)-(L4) preserve that compatibility by the chain rule and uniqueness. In marked overlapping frames the usual passive affine law for the minimizing connection holds; psi canonicalization commutes with those transitions. No angle branch, global clock, new geometry-dependent force, second handoff, or new boundary probe is needed.

The local data determine the exact current two-jet, the original-time future two-jet, every future fixed-context kernel, and the local minimizing geometry and curvature. A two-jet does not determine arbitrary law values on a neighborhood. A Taylor polynomial can realize its current jet in a small positive law chart, but its values away from the expansion point are not the unknown original section. A common contextual field still needs compatible initial data over the field, unless a finite generator is supplied. This is the remaining distinction between replacing a nonlocal *runtime dependency* and compressing an arbitrary contextual *function* into one finite state.

To make the local actual-law assertion explicit, at an evaluation point c_star initialize a nearby natural-parameter family by the degree-two polynomial with the retained *initial* two-jet, keeping the reference preparation as a separate constant-in-evaluation-context memory. Evolve this polynomial preparation by the exact native pointwise flow and actual record kernels; do not truncate the evolved law. In a sufficiently small neighborhood it is a normalized positive statistical family, and its two-jet at c_star agrees with the original preparation and, by (L4), its evolved two-jet at every finite regular time. Its actual history/current scores and minimizing connection jet there therefore coincide with the original ones. An extension near a distinct c0 can be glued to the local preparation before evolution if desired; it is the prescribed reference data, not a fitted geometry. This supplies actual law germs realizing the local data. Different polynomial extensions generally disagree away from c_star, which is why this construction is not advertised as the original whole section. Curvature at the point uses the two-jet of these actual laws, not a formal derivative assigned to one isolated density.

The sufficient conditions are therefore explicit: context-local native field; invariant finite law chart; positive nonzero reference anchor; coherent common-reference initialization; exact same-record reference updates; complete history or its actual initialized law with replay seed; the original lab-label marked action; and a fixed regular finite comparison experiment. All are met by the disclosed finite-reference realization of the published O1 witness. Strict germ-local initialization and a single finite encoding of its arbitrary contextual sections are separate claims, addressed by the scoped obstructions.
