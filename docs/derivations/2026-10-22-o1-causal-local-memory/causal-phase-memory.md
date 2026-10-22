<!-- rigorous-theory-search-metadata {"contract_id": "contract-sha256-26ec7fe2a48e4cb1afab6357d8b2bb3c2f89e4e3cf052844f500ffc03dfea722", "schema_version": "rigorous-theory-search/v1", "target_digest": "26ec7fe2a48e4cb1afab6357d8b2bb3c2f89e4e3cf052844f500ffc03dfea722"} -->
# A causal local phase accumulator

## 1. Source, variables and domain

This construction resumes the published O1 model and the finite-reference result, rather than proposing another force or another probe interface. The native field below is the field in `../2026-10-22-o1-local-parent/local-realization.md`, L1, inherited from `../2026-10-22-o1-collective-compression/routes/constructive-per-agent.md`. The live manuscript discussion is `Research/manuscripts/MAgent_Theory_2.0/chapters/14_collective_local_parent.tex`. The existing frozen-handoff-frame observation is its predecessor. The additional result here is an explicit causal implementation with local initialization under a stated coherence condition, together with the exact defect when that condition is absent.

For each of the three agents let x_i=(l_i,A_i,C_i,B_i,n_i), with A_i,C_i,B_i in R2. Relative to uniform sign measure and normalized Haar measures, the recognition density is

\[
q_i(z,\phi,\psi)=Z_i^{-1}\exp(l_i z/2+A_i\cdot e_\phi+z C_i\cdot e_\phi+B_i\cdot e_\psi),
\]

\[
Z_i=I_0(|B_i|)\{e^{l_i/2}I_0(|A_i+C_i|)+e^{-l_i/2}I_0(|A_i-C_i|)\}/2.
\]

The presentation law uses counting measure on w in {−1,+1}, with probability s_i(w)=exp(n_i w/2)/(2 cosh(n_i/2)). All are strictly positive normalized laws. The self evaluator, peer channels, evidence likelihoods, source priors and the Gibbs minimization are exactly those of O1. In particular write

\[
L_i=l_i+\log I_0(|A_i+C_i|)-\log I_0(|A_i-C_i|),\quad
m_i=\tanh(L_i/2),\quad M(B)=\nabla_B\log I_0(|B|).
\]

With J=epsilon cubed, t_i=E[z cos(phi)], source weights are w_ii=exp(J t_i)/(5 I_0(J)) and w_ij=2 exp(L_j m_i/2+B_j dot M(B_i))/(5 cosh(L_j/2) I_0(|B_j|)); beta is their row normalization. Gamma is the original presentation Gibbs row. No row is frozen or replaced. Persistent coefficients H_i and Lambda_i are the exact sums of the retained sign and marked-angle evidence. The native own-block Fisher equations, in the original time t, are

\[
\begin{split}
\dot l_i&=-l_i+\sum_{j\ne i}\beta_{ij}L_j+H_i,\\
\dot A_i&=-A_i+\Lambda_i,\\
\dot C_i&=-C_i+\beta_{ii}J e_1,\\
\dot B_i&=-B_i+\sum_{j\ne i}\beta_{ij}B_j,\\
\dot n_i&=-n_i+\sum_{j\ne i}\gamma_{ij}n_j.
\end{split} \tag{C1}
\]

Call this smooth field V(x,c,h). Context dependence includes the fixed covariant embeddings and retained record coefficients. The equations have no contextual derivatives. Physical rotation rho(R) rotates all B_i and no other coordinate; V is equivariant. This rotation concerns psi. The marked gauge action concerns phi and its embeddings, with numerical lab record labels fixed, and is a different action.

On each finite admitted branch use the published pointed-cone domain for the B_i. If a fixed vector v is positive on their initial cone, v dot B_i(t) is at least exp(−t) v dot B_i(0)>0. Therefore S(x)=sum_i B_i never vanishes on finite intervals. The same statement holds after arbitrary finite permitted evidence appends, since they preserve positivity of the rows and do not jump B. There is no bound uniform over infinitely many arbitrarily strong transactions.

## 2. Local quotient and causal memory

Define J2=[[0,−1],[1,0]], let K act as J2 on every B block and as zero elsewhere, and set

\[
U(S)=|S|^{-1}\begin{pmatrix}S_x&S_y\\-S_y&S_x\end{pmatrix},\qquad
r=\rho(U(S(x)))x.
\]

The r coordinates satisfy sum_i B_i(r)=(|S|,0). This uses only the instantaneous local native configuration at c, with no reference context. Define the local scalar

\[
\omega(r,c,h)=\frac{\det(S(r),\sum_i V_{B_i}(r,c,h))}{|S(r)|^2}.
\]

For a declared actual start time t_*, initialize theta(c,t_*)=0, and evolve

\[
\dot r=W(r,c,h):=V(r,c,h)-\omega(r,c,h)Kr,\qquad
\dot\theta=\omega(r,c,h). \tag{C2}
\]

Theta may be a real temporal lift or an SO(2) memory g satisfying dot g=omega J2 g and g(t_*)=identity. Winding is unnecessary for the law. Unlike the old reference velocity, omega here varies with c and its contextual derivatives must be retained when differentiating C2. It is evaluated from the local native laws and their original Gibbs rows, rather than estimated from external absolute-psi measurements.

The actual parent law coordinates are

\[
z=\rho(R_\theta)r. \tag{C3}
\]

The letter z in C3 denotes the complete parent parameter tuple; the sign argument of q in Section 1 is a separate dummy variable. Equivariance and differentiation give

\[
\dot z=\rho(R_\theta)(\dot r+\dot\theta Kr)
=\rho(R_\theta)V(r,c,h)=V(z,c,h). \tag{C4}
\]

Thus the parent's actual recognition and presentation laws are the normalized native laws evaluated at z, not at the instantaneous slice r alone. Their own-block Fisher evolution is exactly C1. The correction in C2 is the derivative of the coordinate map and cancels in C4; it is not a fitted parent force. Time is unchanged.

Let alpha(c,t) denote a continuous temporal lift of arg S(x(c,t)) on a branch. Then dot alpha=omega and theta=alpha(c,t)−alpha(c,t_*), so

\[
z(c,t)=\rho(U(S(x(c,t_*))))x(c,t). \tag{C5}
\]

C5 is also a direct uniqueness proof of C2–C4. It identifies this construction as a causally implemented frozen *local initial* frame. The old frozen *reference handoff* frame was already known. No new invisibility mechanism is asserted.

## 3. Complete local closure through order two

For d=2 retain j_c^2 r, theta and its first and second contextual derivatives, all persistent coefficients and their required jets, the known constitutive templates, the complete ordered history, the original time, and both census entries. A local canonical initial two-jet is enough to initialize C2; no alpha(c,t_*) or inverse U is retained. A compatible actual preparation/past-law initialization is required, as in Section 4. This is one finite state per context, with an unbounded chronological record, rather than one finite global encoding of every possible section.

Subscripts a,b denote explicit contextual partials, and D denotes differentiation in r. Holding the numerical history fixed, ordinary prolongation gives

\[
\dot r_a=DW r_a+W_a,
\]

\[
\dot r_{ab}=DW r_{ab}+D^2W[r_a,r_b]+DW_a r_b+DW_b r_a+W_{ab}, \tag{C6}
\]

\[
\dot\theta_a=D\omega r_a+\omega_a,
\]

\[
\dot\theta_{ab}=D\omega r_{ab}+D^2\omega[r_a,r_b]
+D\omega_a r_b+D\omega_b r_a+\omega_{ab}. \tag{C7}
\]

The initial theta jets vanish. All derivatives of beta, gamma, record embeddings and omega are included in D, W_a and omega_a; none are held artificially constant. W is algebraic in the local law coordinates and known coefficients, so C6–C7 need no third contextual derivative. In the private blocks,

\[
z_a=R_\theta(r_a+\theta_a J_2r),
\]

\[
z_{ab}=R_\theta(r_{ab}+\theta_aJ_2r_b+\theta_bJ_2r_a
+\theta_{ab}J_2r-\theta_a\theta_b r). \tag{C8}
\]

Other blocks are unchanged. At evidence transactions the laws, r and theta are continuous; only the actual persistent coefficients and history are appended, with their explicit contextual derivatives. Diagnostics append their actual outcomes and do not alter the objective. Local consistency constraints of the slice and of its prolonged jets are preserved by differentiation. Theta contributes six scalar coordinates to a two-jet in two contexts; r has 23 independent instantaneous coordinates. No minimal memory count or net instantaneous dimension reduction is claimed.

For actual contextual germs one uses the normalized C1 laws generated from the actual local initial family, rather than declaring a tensor from arbitrary inconsistent jet values. Every compatible initial two-jet on the coherent slice is realized by a local C2 family on that slice, using a coordinate chart and a sufficiently small patch. Evolving that family by C2 realizes C6–C8. A jet at one context still does not encode values at every other context.

## 4. Every admitted history and the preparation condition

At each fixed c, C5 is one time-independent physical rotation. Every original public, marked-lab and model probe, every source-label law, every relative-angle query and every assimilation transaction is invariant under it. A joint relative query draws each involved primitive psi only once and returns the same entire difference vector; shared-vertex correlations and cycle constraints are preserved. There is no replacement by independent edge draws. The original complete relative-probe graph continues to expose every proper block against its exterior. No probe is deleted or reclassified.

Couple two processes by the original transaction kernel and independent policy randomization. At the first transaction the response and update law agree. The same chronological record gives the same coefficients, and C4 gives the same transformed waiting flow. Induction proves equality of every finite adaptive joint response/update history, including protocols that mix diagnostics, evidence and waits. The policies are exactly the original functions of retained history and lab data; raw frame-dependent internal-state policies are not added. Persistent evidence and census results are copied as actual retained records and not recomputed after a reset. The original interaction-dependent grouping and prescribed null comparison are unchanged because all their evaluated quantities are invariant.

The memory must run from the actual t_* whose preparation is used to define the experiment. If a handoff occurs at T>t_*, the accumulator has already evolved and is retained at T. It can be carried as a passive exact local bookkeeping variable before the handoff. Initializing it to zero at T instead gives C5 with t_*=T, which requires coherence at T for Section 5 and is not generally justified by coherence at the original start. Existing history and memory cannot be erased to fix this.

For an unconditional history/current experiment one must initialize the actual normalized chronological law from the original preparation, or retain a sufficient local preparation seed and replay its fixed kernels. One observed record word is not its own probability law. A local canonical initial value and, for geometry, its two-jet provide a sufficient local replay seed for the native pointwise flow. This seed is phase invariant and adds no sampled reference factor. Full records remain retained even when their sums close C1. No prior over possible prepared sections is introduced. The conditional construction below requires coherence for the actual original preparation, before any random records that change the phase evolution; a purported unobserved prehistory cannot be fabricated from a snapshot.

## 5. When the actual Fisher geometry is inherited

The explicit additional sufficient condition is

\[
\frac{S(x(c,t_*))}{|S(x(c,t_*))|}=n_*\quad\hbox{for every }c\in C, \tag{C9}
\]

where n_* is one unit vector independent of context and of subsequent history. Its numerical value is not retained. Each local canonicalization uses its own S and therefore applies the same rotation U_* without communication. C9 is a restriction on the preparation, not information created by the local dynamics, not an attractor theorem, and not an open condition in the unrestricted section class. The B shapes may vary across context while their collective direction remains fixed.

Let the actual comparison experiment be P_c,t(dh,dY)=Pi_c(dh) Q_c,t^h(dY), including the original recognition, presentation and source-label factors and any public-pattern conditioning normalization. Under C9, C5 gives

\[
\bar P_{c,t}=T_{U_*\,*}P_{c,t}, \tag{C10}
\]

where T applies one fixed Haar-preserving shift to all current psi samples and leaves h, lab labels and all other current samples fixed. It is independent of c,t,h on this prepared experiment and commutes with the marked action. Consequently for every contextual tangent X and marked Lie-algebra element xi,

\[
E_{\bar P}(\bar s_X+\bar\ell_\xi)^2
=E_P(s_X+\ell_\xi)^2. \tag{C11}
\]

This includes history scores and conditioning-normalizer derivatives. The minimizing affine connection classes, isotropy, minimized tensor and its curvature agree. Unsampled theta/seed memory is not appended as an independent Fisher experiment. Its actual contextual dependence already enters q through C3 and C8. Copying a geometry matrix onto an r-only law would fail C11.

There is a nonempty common O1 window satisfying C9. In the published rank-two witness take the small cone vectors B_i(c,t_*)=delta b_i constant in c, with b_i nonzero and, if desired, noncollinear. Keep the published marked family A_1=a e_chi and a second fixed marked amplitude d>0, the interaction-dependent initial public data, persistent conflicting records, and all original fixed probes and census times. The original continuity argument applies to this specialization. At epsilon=delta=0 its marked part has

\[
\mathcal A=-\frac{I(v)}{I(v)+D(t)}d\chi,\quad
\mathcal F=-\frac{D(t)e^{-t}I'(v)}{(I(v)+D(t))^2}da\wedge d\chi,
\]

where v=e^{−t}a, I(v)=v I_1(v)/I_0(v), and D(t)>0 is the other marked orbit inertia. At this limit the actual marked cost is e^{−2t} Var_v(cos(phi)) da squared plus I(v)(d chi+xi) squared plus D(t) xi squared; minimizing in xi gives the displayed connection and angular coefficient. Also I'(v)=I_1(v)/I_0(v)+v Var_v(cos(phi))>0 for v>0. On a compact patch with a>0, the radial Fisher entry e^{−2t} Var_v(cos(phi)) and the angular entry I(v)D/(I(v)+D) are positive and the displayed curvature is nonzero. The weak initial record episode becomes uniform in this limiting experiment, exactly as in O1 G18–G20; it is neither erased nor replaced at positive parameters. Strict margins persist on the published finite window for sufficiently small positive epsilon and delta. The anchor construction is used only at positive delta. C11 transfers the *actual* positive-rank nonflat geometry on that same window. Arbitrary later evidence may change the curvature or grouping; exact history and dynamical inheritance continue on each regular finite branch, but nonflatness after every conceivable later record is not claimed.

All retained local data, theta, seed and history are invariant under one common physical rotation of the original preparation. Conversely, the field of canonical initial data on C9 determines the original preparation up to that one constant rotation: equality of the initial r fields makes the two S directions differ by one context-independent angle. A smooth native fixed-record flow can be inverted locally if this is assessed at a later time. Thus the lost global phase is not recovered by memory, and no independently invisible primitive factor is introduced. Off C9 the same pointwise retention erases more than one constant phase, which causes the following defect.

## 6. The missing initial spatial phase

For a general preparation let alpha_0(c)=arg S(x(c,t_*)). C5 is a c-dependent sample rotation. At corresponding samples its contextual score satisfies

\[
\bar s_X(T_0Y)=s_X(Y)-(X\alpha_0)\ell_{\rm phys}(Y), \tag{C12}
\]

where ell_phys is the score for positive common rotation of B. The sign follows directly from log q(B;psi) and the change psi_parent=psi_fine−alpha_0. This is a genuine change of actual Fisher costs. No general data-processing monotonicity is asserted for this parameter-dependent sample map.

Locally write a=d_c alpha and Omega=partial_t alpha. Smoothness gives

\[
\partial_t a=d_c\Omega,\qquad
a(t)=a(t_*)+d_c\int_{t_*}^{t}\Omega(s)ds. \tag{C13}
\]

The causal accumulator reconstructs the second term exactly. It does not supply the initial spatial one-form a(t_*). Spatial integration would additionally require compatible contextual data and an integration constant; such communication or context transport is not one of the fixed actions. C13 concerns the physical common-phase coordinate, whose spatial derivative is locally exact. It must not be identified with the separate nonflat marked Fisher connection. The conditional construction is complete on C9; C12–C13 explain why it does not settle arbitrary preparations.
