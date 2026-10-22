<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca","schema_version":"rigorous-theory-search/v1","target_digest":"0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca"} -->
# Actual Fisher geometry of a per-agent parent with persistent records

## Supported result

There is an exact geometry and record closure construction for the conditional-reconstruction per-agent model on a declared invariant family. The parent keeps each public law, its marked private density, and its hidden concentration, while discarding a hidden circular phase. It is a normalized law on a fixed sample space, has the original own-block Fisher dynamics, and preserves the scores of the complete record-history/current-state experiment on an explicitly specified contextual family. Its collective minimizing cost, connection, and tensor therefore agree with the fine quantities before and after minimization. Two belief factors alone give an explicit rank-two contextual tensor with nonzero curvature. A positive normalized record likelihood participates in the calculation and persists in the original objective. No belief/model group tie is used.

The raw log-density-spectrum probability law is not a suitable Fisher realization in this example: varying a hidden concentration moves an arcsine-support endpoint and violates quadratic-mean differentiability. A canonical circle law, or an equivalent folded-quantile law on a fixed interval, removes that defect while representing the same retained spectrum.

The lost phase is invisible to the fixed boundary and record interface from the outset. Consequently this is a conditional inheritance/compression component, not endogenous discovery of a compression interface. This route does not establish interaction-dependent grouping or the counterfactual second join. A finite-horizon bridge to a separately proved strict public grouping witness is stated at the end.

## Contract and source continuity

The frozen existential target requires a normalized per-agent mixture-source free energy with fixed positive source priors, exactly Gibbs-reduced or dynamical rows at unit temperature, persistent normalized record likelihoods, a noninjective inherited parent, same-physical-time flow, all admitted joint adaptive response-record histories, and actual nonflat collective Fisher geometry. It also requires interaction necessity against the prescribed no-peer-cost control. The latter and the mandatory four-state transfer are assigned to other routes.

This note chooses exactly Gibbs-reduced unit-temperature rows. It uses a declared invariant family rather than claiming openness in all probability densities. Its hidden phases vary independently as physical initial-state parameters, but are constant over the evaluation-context coordinates used for the geometric assertion. That is a restriction on contextual sections, not an equality constraint between different agents' hidden phases. Arbitrary contextual variation of the discarded phases would lose Fisher information and is explicitly excluded.

The source pass read the current `MAgent_Theory_2.0/AUTHORING.md`, the per-agent mixture-source model and own-block flow in Chapter 3, the collective, score-projection, and retention constructions in Chapter 2, the current formation standard at the end of Chapter 5, and Chapter 12's revised response/geometry and null-control scope. The earlier `endogenous-interface-selection/routes/moving-interface-geometry.md` supplies the moving-map compatibility distinction. The present calculation uses the same conditional-reconstruction and marked-spectrum state supplied by the parallel `interface_histories` route. It does not import reciprocal incoming forces, the old tied-sector curvature example, or a copied tensor on spectral metadata.

## Typed dictionary and conventions

| Object | Type and convention |
| --- | --- |
| Context | A smooth local base C, with coordinates (a,chi) in the explicit example. Evaluation context is separate from inference time t and record index n. |
| Belief sample | K=Z x S1_phi x S1_psi, where Z is a finite public alphabet. The reference is counting measure on Z and normalized Haar measure on both circles. |
| Model sample | A finite public alphabet W in the displayed construction, with a proper positive reference. A separate marked-spectrum model sector may be included with the same argument, but is unnecessary for the curvature. |
| Fine laws | q_i(z,phi,psi)=mu_i(z) f_i,z(phi) v_bi,z(psi-theta_i,z), with v_b(u)=exp(b cos u)/I_0(b), b>0. The model laws s_i(w) are positive. Recognition is the stipulated product, with Gibbs source-label laws beta_i,gamma_i. |
| Parent laws | qhat_i(z,phi,psi)=mu_i(z) f_i,z(phi) v_bi,z(psi), or the folded version on Z x S1 x (0,1). Both are actual normalized laws with fixed references. |
| Interface mark | The retained sample mark is (z,phi). The private phase theta is not an admitted boundary or likelihood argument. |
| Records | Positive normalized kernels on a declared record space, depending on (z,phi) and possibly retained model variables. Their full accumulated negative log likelihood stays in the original per-agent free energy. |
| Gauge group | G=S1 acts from the left on the marked belief coordinate phi of every participating belief law and on its angular record coordinates. It acts trivially on psi and on the displayed model sector. There is no enforced equality between belief and model gauge groups. |
| Physical discarded action | Independent shifts psi -> psi+delta_i,z within public strata. These change hidden angular distributions in fixed sample coordinates. They commute with G and are not elements of this declared gauge representation. |
| Connection | D^A p=dp+J_p A; p'=g_#p, A'=A-dchi_0 for a passive marked-circle rotation chi_0(c). The collective connection is obtained by minimizing the actual joint Fisher cost. |
| Metric | Unit Fisher score inner product of the declared statistical experiment, including source labels and record histories when those are included. The inference kinetic metric after row reduction and the geometry of a larger record experiment are distinguished. |
| Bundle and transports | Work on a trivial local principal S1 bundle over C. Curve transport is obtained from A. A canonical hidden-phase shift is a physical statistical map, not the parallel transport of A. |

The parent of a selected block is the product of its retained primitive factors, with primitive ports, source priors, and directed own-block equations inherited. Product encoding does not turn the parent dynamics into the gradient of the sum of agents' free energies. No such global-gradient assertion is used.

## The original per-agent model and its exact conditional flow

Let g_i(w,z)>0 be a normalized public evaluator and let r_i(w)>0 be a proper model reference. The full evaluator is

\[
G_i(w,dz,d\phi,d\psi)=g_i(w,z)d\phi/(2\pi)d\psi/(2\pi).
\tag{R1}
\]

For a nonself source j, choose a fixed positive public channel B_ij and the normalized receiver reconstruction

\[
K^q_{ij}((z_j,\phi_j,\psi_j),dz,d\phi,d\psi)
=B_{ij}(z_j,z)\kappa_{i,z}(\phi,\psi)d\phi/(2\pi)d\psi/(2\pi),
\tag{R2}
\]

where kappa_i,z=f_i,z v_bi,z(psi-theta_i,z). Thus

\[
q_jK^q_{ij}=(\mu_jB_{ij})(z)\kappa_{i,z},
\qquad
D^q_{ij}=\operatorname{KL}(\mu_i\Vert\mu_jB_{ij}).
\tag{R3}
\]

Equation (R3) is an exact identity of functionals. Differentiating it includes the derivative of the receiver's kappa in (R2); freezing that reconstruction would produce a spurious private peer force. Agents differentiate only their own free energies. No source-role derivative of another agent's objective is added.

Write

\[
H_{i,z}=\int\kappa_{i,z}\log\kappa_{i,z}
=H(f_{i,z})+h(b_{i,z}),
\quad
h(b)=b m(b)-\log I_0(b),
\quad m(b)=E_{v_b}\cos\psi.
\tag{R4}
\]

The self belief cost is

\[
D^q_{ii}=\sum_{z,w}\mu_i(z)s_i(w)
\log\frac{\mu_i(z)}{g_i(w,z)}
+\sum_z\mu_i(z)H_{i,z}.
\tag{R5}
\]

All source priors are fixed positive and normalized. The belief row is beta_ij=pi^q_ij exp(-D^q_ij)/Z_i, and model rows are treated in the same exactly reduced convention. The record potential after a finite history h is W_i,h(z,phi), the sum of negative log likelihoods. It is not removed after arrival. Model-record factors may be included separately.

The conditional variation of the original reduced own free energy contains beta_ii log kappa_i,z+W_i,h, modulo terms constant on the conditional fiber. Conditional centering therefore gives

\[
\partial_t\log\kappa_{i,z}
=\beta_{ii}(H_{i,z}-\log\kappa_{i,z})
+E_{\kappa_{i,z}}W_{i,h}-W_{i,h}.
\tag{R6}
\]

The dependence of beta on the current laws is not differentiated again: the Gibbs envelope gives beta times the complete source-cost derivative. It does not suppress any derivative in (R3).

Since W depends only on the retained mark, (R6) separates exactly into

\[
\begin{aligned}
\partial_t\log f_{i,z}
&=\beta_{ii}\big(H(f_{i,z})-\log f_{i,z}\big)
+E_{f_{i,z}}W_{i,h}-W_{i,h},\\
\dot b_{i,z}&=-\beta_{ii}b_{i,z},
\qquad \dot\theta_{i,z}=0.
\end{aligned}
\tag{R7}
\]

The public force contains its public source contribution plus

\[
\beta_{ii}H_{i,z}+E_{f_{i,z}}W_{i,h}.
\tag{R8}
\]

Equations (R3)-(R8) depend on mu,s,f,b and the accumulated records, not theta. Hence setting theta=0 gives the same original-time own-block equations on actual normalized parent laws. The model equations also close because every appearance of q in the model derivative is through retained quantities. The evaluator, reconstruction kernels, and likelihoods remain normalized after this canonicalization.

This is an invariant family with arbitrary positive marked densities f. For finite histories of positive C^r likelihoods on the compact mark space, r at least 3, use a normalized positive C^r chart for f and interior finite-table charts for mu,s. On a neighborhood bounded away from zero, logarithms, bounded integrals, finite Gibbs sums, and normalization are smooth locally Lipschitz maps. The displayed field therefore has a unique local solution. Positivity and the lower bounds persist on a sufficiently short common interval. Since beta lies in (0,1), b(t)=b(0) exp(-integral beta) remains positive at every regular finite time. Compact contextual patches and compact record spaces give uniform bounds for a fixed finite number of observations. This supplies the local regularity used below.

For the normalized exponential likelihood exp(lambda cos(o-phi))/I_0(lambda), W has a first-harmonic natural vector. If f is initially von Mises with vector parameter a, the smaller finite-dimensional family is invariant and dot a=-beta_ii a+sum lambda e(o). The proof of (R7) covers this case. The explicit record-curvature calculation below instead uses a positive cardioid likelihood; its marked density is permitted to leave the von Mises subfamily after records begin affecting the field.

## Why the raw spectrum is not a regular Fisher realization

The marked spectrum retains the reference-law distribution of

\[
(r,\phi)=\big(\log\kappa(\phi,\psi),\phi\big),
\qquad \eta=(\log\kappa,\phi)_\#\operatorname{Haar}^{\otimes2}.
\tag{R9}
\]

The law e^r eta is normalized. At a fixed phi, for the family above,

\[
r=\log f(\phi)+b\cos(\psi-\theta)-\log I_0(b).
\tag{R10}
\]

Its conditional support has endpoints

\[
L_b(\phi)=\log f(\phi)-b-\log I_0(b),
\qquad
U_b(\phi)=\log f(\phi)+b-\log I_0(b).
\tag{R11}
\]

Under the reference measure it has an arcsine density, and under e^r eta that density is multiplied by a positive bounded weight on every compact positive-b chart. For delta>0,

\[
U_{b+\delta}-U_b=(1-m(b))\delta+O(\delta^2)>0.
\tag{R12}
\]

The new law assigns probability of order sqrt(delta) to r>U_b(phi). Indeed this event is equivalent to cos(psi-theta)>1-c delta+O(delta^2), with c=(1-m(b))/b>0. Its angular interval has length of order sqrt(delta), and the von Mises density is bounded above and below near theta. The old law assigns that set zero probability. Thus the singular mass of the new law relative to the old one is of order sqrt(delta), whereas differentiability in quadratic mean requires singular mass o(delta^2). The condition fails.

Consequently e^r eta is a valid retained probability law but does not possess the claimed regular Fisher metric in the b direction. The issue is not removed by retaining phi alongside r. Nor can a metric be assigned to this raw law by copying the fine amplitude metric.

The canonical parent in the dictionary has no such defect. An alternative uses u in (0,1) with Lebesgue reference and density

\[
\widehat\kappa(\phi,u)
=f(\phi)\frac{\exp[b\cos(\pi u)]}{I_0(b)}.
\tag{R13}
\]

It is normalized because integral_0^1 exp(b cos(pi u))du=I_0(b). The fixed map, for a given physical phase theta,

\[
T_\theta(\phi,\psi)
=\left(\phi,\frac{\arccos(\cos(\psi-\theta))}{\pi}\right)
\tag{R14}
\]

sends the reference Haar measure to Haar times Lebesgue and pushes kappa to (R13). It is the decreasing conditional quantile realization of (R10). The retained spectrum determines b from half the conditional support width and determines log f(phi) from the midpoint plus log I_0(b). It therefore determines (R13) on this family without retaining theta. The canonical circle and folded interval realizations have the same marked and amplitude scores. Both are regular actual parent experiments; neither claims that the raw spectral experiment has acquired a finite Fisher metric.

## Exact Fisher equality and positive-dimensional physical loss

For a tangent variation of one fine factor, write u=psi-theta. Its score is

\[
\ell=\frac{d\mu(z)}{\mu(z)}
+d\log f_z(\phi)
+db_z\big(\cos u-m(b_z)\big)
+d\theta_z b_z\sin u.
\tag{R15}
\]

Conditional normalization centers every displayed private score. Independence of phi and psi conditional on z, and parity in u, give

\[
\begin{aligned}
g^F={}&\sum_z\frac{d\mu(z)^2}{\mu(z)}
+\sum_z\mu(z)g^F_{f_z}
+\sum_z\mu(z)V(b_z)db_z^2
+\sum_z\mu(z)I(b_z)d\theta_z^2,\\
V(b)&=\operatorname{Var}_{v_b}(\cos\psi)>0,
\qquad I(b)=b m(b)>0.
\end{aligned}
\tag{R16}
\]

The identity I(b)=E[b^2 sin^2 psi] follows by integrating the derivative of sin psi exp(b cos psi) around the circle. Positivity of m(b) for b>0 follows by pairing opposite angles in its integral; its derivative is V(b)>0.

On a contextual family with d_c theta_i,z=0, (R15) is measurable from the folded parent variables. The G-orbit score is -partial_phi log f_z, with the corresponding action on records, so it is retained as well. Equation (R16) then proves equality of every collective cost in contextual-plus-orbit directions with the original sum normalization. The kernel, locked inertia, contextual momentum, minimizing affine class, and minimized tensor agree. On an isotropy-free patch the connection is unique and its curvature agrees.

The retention is nevertheless noninjective on the invariant state family: independently changing any theta_i,z leaves mu,f,b, all rows, all records in the admitted interface, and the parent law unchanged. A child diagnostic E[e^{i psi}|z]=m(b_z)e^{i theta_z} distinguishes these states. Its Fisher norm in (R16) is positive. This is a physical distinction in the fixed psi coordinate and is not a G gauge direction, because G acts on phi only.

The diagnostic is not an inherited boundary probe. If an individual hidden-angle query is admitted as a future boundary response, exact compression fails; the protocol cannot remove it at handoff. The construction declares that interface before retention and keeps it throughout. It therefore proves supplied-interface compression, not discovery of the interface.

If theta varies over context, the last term of (R16) is discarded. Since it is orthogonal to the marked gauge direction, collective minimization cannot remove it. Thus the theorem does not preserve all possible contextual sections. The constants theta_i,z are allowed to vary independently between configurations, which supplies the loss fibers, while remaining constant over evaluation context within each asserted section.

## Records are included in the probability experiment

There are two different experiments, and their Fisher metrics must not be conflated.

For one future draw from a law Q_c followed by a fixed normalized sensor ell(o|z,phi), the full joint law is

\[
\mathbb P_c(dy,do)=Q_c(dy)\ell(do|z,\phi).
\tag{R17}
\]

Its contextual score is exactly the Q score. The record does not add information when the complete input sample is also retained. The parent map (R14), leaving the record untouched, preserves that score. If a sensor parameter eta(c) varies with context, the score instead includes partial_eta log ell times d eta. That additional score is a function of retained variables and the record, so it is preserved rather than omitted. Gauge action rotates phi and the angular record together; a difference-only sensor is invariant under that joint action.

For an accumulated history H and a fresh sample from the current recognition state conditioned on H, the joint law is instead

\[
\mathbb P_{c,t}(dh,dy)
=\Pi_c(dh)Q^h_{c,t}(dy).
\tag{R18}
\]

The history need not be independent of the state, and Q^h is the actual solution under the persistent record potential W_h. Under the stated regularity,

\[
\ell^{\rm joint}_X(h,y)
=\ell^{\rm hist}_X(h)+\ell^{Q|h}_X(y),
\qquad
g^{\rm joint}=I(\Pi)+E_\Pi g^F_{Q|H}.
\tag{R19}
\]

The cross term vanishes by conditional normalization, not by an independence assumption. The first term can be nonzero and is kept.

For the protocol here, every next-record kernel depends on the current state only through the retained data. The parent has the same normalized self and peer source predictions on marked variables, the same full record likelihood, and the same waiting-time flow by (R7)-(R8). Its accumulated record likelihood remains the same function W_h. Conditioning successively on the realized history proves equality of every finite adaptive joint history and next-state law. This argument permits a finite adaptive choice among normalized retained-mark sensors and observation times within the common regular window. It introduces no distribution over law configurations. It compares the operational experiments at each specified configuration.

The hidden phases are constant in time and do not depend on record history. Thus, on the asserted contextual family, the product of maps (R14) is fixed across c and h. Every score in (R19) is retained: the history law is identical, and the conditional current-state score is (R15) with dtheta=0. The same statement holds for the G-orbit score of the full joint experiment because the hidden folding commutes with the marked-coordinate action and the entire family is equivariant when records and sensor coordinates are re-expressed together. Consequently the equality of collective costs holds for (R18), not merely for recognition marginals at a chosen observed record.

If source labels are included in the declared statistical family, take Q^h to include the actual Gibbs factors beta_i and gamma_i. Their score terms d log beta_J and d log gamma_L are retained exactly. In the explicit family below their gauge scores are zero, so their contextual Fisher terms add to the tensor but not to locked inertia or momentum. This choice does not change the own-block kinetic metric used after exact row reduction; it specifies the larger experiment whose collective geometry is being compared.

For a single fixed realized history, one may condition on that history and compare Q^h and Qhat^h. Its stored value is then constant in context on that conditional branch. This conditional comparison alone does not establish the geometry of the sampling distribution of histories; (R18)-(R19) supply that missing calculation. No context-varying Dirac record coordinate is treated as a regular Fisher factor.

## Explicit nonflat geometry with informative records in the same model

Use two belief agents, or two belief factors retained by a selected block. They have a common marked-circle frame and distinct physical hidden phases. Initially, in every public stratum, set

\[
f_1(\phi;a,\chi)=v_a(\phi-\chi),
\qquad f_2(\phi)=v_l(\phi),
\qquad a>0,\quad l>0.
\tag{R20}
\]

All hidden concentrations b_i,z are positive and initially context-constant. Public laws and model laws are initially context-constant. The hidden concentrations remain part of the retained state and enter the source costs through (R4)-(R5).

Choose the normalized positive angular likelihood

\[
\ell_\epsilon(o|\phi)=1+\epsilon\cos(o-\phi),
\qquad o\in S^1,\quad 0<\epsilon<1,
\tag{R21}
\]

relative to Haar measure on the record circle. It is strictly positive and integrates to one. Public record factors may be multiplied into this likelihood if each is normalized; they add retained scalar score terms.

For definiteness, records are drawn from each agent's own normalized mixture-source generative model, not from an assumed global joint. Conditional on the current configuration, the operational protocol draws these local records independently. This is a normalized product of declared next-record kernels, not a claim that the local inference objectives come from one global free energy.

Let alpha_i=sum_{j not equal i} pi^q_ij, strictly between zero and one. At (R20), the self source is uniform in phi and every nonself reconstructed source has marked law f_i. Therefore the normalized generative predictive density of one local angular record is

\[
\rho_1(o;a,\chi)=1+r_1(a)\cos(o-\chi),
\quad r_1(a)=\epsilon\alpha_1m(a),
\qquad
\rho_2(o)=1+r_2(l)\cos o,
\quad r_2(l)=\epsilon\alpha_2m(l).
\tag{R22}
\]

Draw one record for each agent at the initial time and retain it. The recognition laws are not reset. They then evolve under the original free energies with -log ell_epsilon of those records present throughout the following interval. Immediately after record arrival, the history is independent of a fresh current recognition sample conditional on the initial configuration; thereafter the history and current state are dependent as in (R18).

For the circular density rho_r(o)=1+r cos o, its phase Fisher inertia is

\[
J(r)=\int\frac{r^2\sin^2o}{1+r\cos o}\frac{do}{2\pi}
=1-\sqrt{1-r^2}.
\tag{R23}
\]

To verify it, write r^2 sin^2 o=(1+r cos o)(1-r cos o)-(1-r^2). The integral of (1+r cos o)^{-1} is (1-r^2)^{-1/2}, obtained directly by the tangent-half-angle substitution. The amplitude-phase cross term vanishes by oddness. The record amplitude information is

\[
I_{a}^{\rm rec}
=[\epsilon\alpha_1m'(a)]^2
\int\frac{\cos^2o}{1+r_1(a)\cos o}\frac{do}{2\pi}>0.
\tag{R24}
\]

Define the complete phase inertias of history plus fresh recognition by

\[
A(a)=I(a)+J(r_1(a)),
\qquad B=I(l)+J(r_2(l)),
\quad I(a)=a m(a).
\tag{R25}
\]

Every term is finite and positive. Since m'(a)=Var_{v_a}(cos phi)>0 and m(a)>0,

\[
A'(a)=m(a)+a m'(a)
+\frac{\epsilon^2\alpha_1^2m(a)m'(a)}
{\sqrt{1-\epsilon^2\alpha_1^2m(a)^2}}>0.
\tag{R26}
\]

The common marked-circle collective cost in the chi direction is therefore

\[
A(a)(d\chi+\xi)^2+B\xi^2.
\tag{R27}
\]

Source-label scores, public/model scores, and hidden-amplitude scores have no gauge orbit component. At this initial section, source costs are independent of chi because they depend on the marked entropy and public laws. Their a-score terms are retained and add a nonnegative scalar term to the amplitude metric. Parity makes the initial amplitude/phase cross terms zero. The actual minimizing form and curvature are

\[
\mathcal A^F=-\frac{A(a)}{A(a)+B}d\chi,
\qquad
\mathcal F^F=-\frac{BA'(a)}{[A(a)+B]^2}da\wedge d\chi\ne0.
\tag{R28}
\]

The minimized tensor satisfies

\[
h_{\chi\chi}^{\min}=\frac{A(a)B}{A(a)+B}>0,
\qquad
h_{aa}^{\min}\ge V(a)+I_a^{\rm rec}>0,
\qquad h_{a\chi}^{\min}=0.
\tag{R29}
\]

This is a rank-two tensor on the stated contextual chart. The record contributions in (R25) cannot be omitted from (R28); doing so would generally give a different connection and curvature. If instead one compares the same input sample jointly with its sensor output, use (R17), in which the input is already observed and no additional J term appears. Both comparisons have exact parent equality, but they are different experiments.

All geometric variables in this example participate in the same per-agent model. The marked laws f enter the self entropy in (R5), hence change beta and the public force (R8). They are reconstructed in actual communication channels and read by (R21). The hidden concentrations b also affect the rows and public/marked dynamics. The curved part is therefore not a dynamically unrelated attached experiment. The hidden phase, however, remains a supplied invisible coordinate of the fixed interface, and that limitation is not removed by its active amplitude.

The formula is exact at the record-arrival slice. For persistence, take a compact patch a in [a_-,a_+] with a_->0 and a compact chi chart. Its positive inertias, tensor eigenvalues, and absolute curvature coefficient have positive lower bounds. The original field, likelihood, and their contextual derivatives are smooth and uniformly bounded on a common regular tube, uniformly over the two compact record values. Thus the conditional solutions, their joint history/current-state densities, and the Fisher data vary continuously for a sufficiently short positive time. The lower bounds remain positive on a common interval [0,T]. Throughout that interval the record potentials remain in the original objective, the marked densities need not remain von Mises, and the parent cost equality remains exact by (R15)-(R19). This is a finite-window existence argument, not an asymptotic approximation or a numerical claim.

## Two record-sensitive failure tests

### Dropping the hidden concentration loses a score immediately

Suppose a smaller parent retains only mu and f, discarding b as well as theta. Initially choose b context-constant and let a vary as in (R20). Then the initial contextual hidden score is zero. With public laws fixed in context, the self cost derivative is

\[
\partial_aD_{ii}^q=\partial_aH(v_a)=aV(a)>0,
\qquad
\partial_a\beta_{ii}
=-\beta_{ii}(1-\beta_{ii})aV(a)\ne0.
\tag{R30}
\]

The conditional equation gives

\[
\partial_a b(t)=-t b\partial_a\beta_{ii}+o(t).
\tag{R31}
\]

The conditional hidden contextual score is therefore

\[
-t b\partial_a\beta_{ii}
\big(\cos(\psi-\theta)-m(b)\big)+o_{L^2}(t),
\tag{R32}
\]

and deleting the hidden factor loses

\[
t^2 b^2(\partial_a\beta_{ii})^2V(b)+o(t^2)
\tag{R33}
\]

per conditional factor, weighted by its public probability. Persistent marked records contribute retained history scores and retained marked forces; they do not cancel this conditionally centered hidden score. The same loss appears in the joint experiment after averaging over histories. The marked gauge acts trivially on the hidden factor, so optimizing the connection cannot absorb it. The spectral/canonical parent succeeds precisely because it retains b and its actual Fisher score.

The raw spectral realization is particularly unsuitable here: (R31) creates contextual b-dependence even if b was initially constant over context, leading directly to the moving-support defect (R12).

### A persistent likelihood that reads the erased phase prevents closure

Admit instead a hidden-angle likelihood

\[
L_\lambda(v|\psi)=\frac{\exp[\lambda\cos(v-\psi)]}{I_0(\lambda)},
\qquad \lambda>0.
\tag{R34}
\]

Even before assimilation, its predictive density depends on theta and distinguishes the quotient fiber. A future probe of this kind cannot descend through the phase-erasing parent.

Its persistent effect is also explicit. For a hidden von Mises natural vector b e(theta), one realized record v contributes lambda e(v) to its own natural-coordinate flow:

\[
\frac{d}{dt}[b e(\theta)]
=-\beta_{ii}b e(\theta)+\lambda e(v),
\quad
\dot b=-\beta_{ii}b+\lambda\cos(v-\theta),
\quad
\dot\theta=\frac{\lambda}{b}\sin(v-\theta).
\tag{R35}
\]

Two states with the same retained b and the same saved record v but different theta generally have different future b velocities. The quotient fails original-time projectability. Retaining the relative orientation v-theta together with v recovers theta locally, eliminating the proposed phase loss. A finite record's failure to identify theta statistically is irrelevant: its retained likelihood already changes the vector field in a theta-dependent way. This is a dynamical obstruction distinct from claiming that finite observations identify a parameter exactly.

## The finite-alphabet limitation and the conditional formation bridge

For the mandatory four-state construction alone, a continuous connected sample-frame group acting by bijections is trivial. Its action maps continuously into the finite permutation group of the alphabet; a connected image in a finite discrete set is one point. Thus its Lie algebra action is zero and its collective connection cannot have nonzero curvature of the type requested here. Finite-state Fisher tensors may have their own Riemannian curvature; that is not curvature of this collective gauge connection. A continuous statistical extension must be declared and must participate in channels, evaluators, or records, as the marked-circle extension above does.

The construction in this note can use the same public evaluator, public channels, model references, source priors, and public-record factors as a public grouping candidate. Small positive marked and hidden concentrations make their extra self entropy H(f)+h(b) small, while leaving positive phase inertia and nonzero local curvature on a chosen positive-concentration contextual patch. Small positive record strength epsilon keeps the additional marked-record perturbation small. On a fixed finite horizon, smooth dependence then makes the extended public field and event functions close to the public candidate, provided the same row convention is used.

Consequently, if a separate public construction proves strict grouping and persistence margins and a strict partition-sequence difference from its prescribed null control, stable under the relevant C1 perturbations, those finite-horizon conclusions persist for sufficiently small positive parameters of this integrated extension. The null model must be perturbed and analyzed in the same way, with its nonself costs deleted and peer dependence frozen exactly as prescribed. This is a conditional bridge; no F5 margins are proved in this route. It cannot establish an infinite-time grouping limit from finite-time continuity alone.

At each chosen block handoff, canonicalization may discard its hidden phase coordinates while retaining all primitive ports, spectra, public states, source rows, event data, and records. This map can be applied before any grouping, so the grouping event does not create the unobservability. Reapplying the same canonicalization at a second join supplies no new loss in phases already erased. An additional noninjective second retention and its counterfactual dependence remain separate obligations.

All constructions are local on positive regular charts. The hidden canonical phase may be recovered from a nonzero first hidden moment on a local chart, but the retained canonical law can be defined directly from its spectrum and amplitude without a global phase lift. Folding maps need not be diffeomorphisms; their score sufficiency, not invertibility, proves the Fisher equalities. The marked S1 connection obeys the usual affine change on overlaps because the complete retention is equivariant and preserves the collective costs. Curvature makes its transport path dependent; no flat endpoint identification is being substituted for that transport.

The new gain is a regular actual-law realization of the marked-spectrum parent, an exact record-history score theorem on the evolved per-agent family, the nonflat record-inclusive two-belief calculation (R22)-(R29), and the persistent-record obstruction (R35). The minimal remaining obligation is a robust public F5 witness in the same constitutive model, followed by a genuinely new noninjective second handoff if recursive formation is claimed. Endogenous creation of the invisible interface is not established.

## Fixed-own-model integration with a common finite grouping window

This section treats the fixed-own-model hybrid in `integrated-grouping-bridge.md`, whose public precursor is `persistent-record-grouping.md`. Its channels differ from the conditional-reconstruction channels in (R1)-(R8): every peer message retains only the public bit and resamples both circles from Haar measure. The conclusions in this section are derived for that hybrid, rather than obtained by identifying its field with the earlier field. The grouping route establishes a finite entrance time independent of epsilon, uniform over the compact initial marked family described below. That fact is the exact compatibility input used here. No nonflatness assertion is made for every later arbitrary record experiment.

### Constitutive laws, own-block flow, and the exact phase quotient

Use the reference probability measure which is uniform on z in {-1,+1} and Haar on phi,psi in S1. Write e(phi)=(cos phi,sin phi). The invariant belief family is

\[
q_i(z,\phi,\psi)
=\frac{\exp[l_i z/2+A_i\cdot e(\phi)+z C_i\cdot e(\phi)]}
{Z(l_i,A_i,C_i)}
\frac{\exp[b_i\cos(\psi-\theta_i)]}{I_0(b_i)},
\tag{R36}
\]

where A_i,C_i are real two-vectors, b_i>0, theta_i is a circular phase, and

\[
Z(l,A,C)
=\frac{e^{l/2}I_0(|A+C|)+e^{-l/2}I_0(|A-C|)}{2}.
\tag{R37}
\]

The fixed self evaluator and the fixed peer channel are

\[
G_i(z,\phi,\psi)
=\frac{\exp[Jz n_i\cdot e(\phi)]}{I_0(J)},
\qquad |n_i|=1,
\qquad
K_{ij}(z_j,\phi_j,\psi_j;dz,d\phi,d\psi)
=\delta_{z_j}(dz)\frac{d\phi}{2\pi}\frac{d\psi}{2\pi}.
\tag{R38}
\]

The normalization of G follows by integrating phi first, for either value of z. The channel is a probability kernel. Keep the independent binary presentation sector of the grouping construction, with its uniform self reference, identity peer channels, proper presentation laws s_i, and Gibbs rows gamma_i. The evaluator (R38) is independent of that model variable, so its presentation dynamics remains decoupled and is retained without modification. This sector carries no marked orbit. In particular there is no identification of a belief group with a model group. Initially all n_i=e(0), so these directions do not encode a chosen population block. The symbol n_i here denotes the fixed unit constitutive direction, not presentation log odds.

The public log odds of (R36) are

\[
L_i=l_i+\log I_0(|A_i+C_i|)-\log I_0(|A_i-C_i|).
\tag{R39}
\]

A record consists of a public bit o and a marked angle omega. One convenient convention, relative to the uniform binary and circular references, is

\[
\ell_\epsilon(o|z)
=\frac{\exp[(\epsilon/2)oz]}{\cosh(\epsilon/2)},
\qquad
\ell_\lambda(\omega|\phi)
=\frac{\exp[\lambda\cos(\omega-\phi)]}{I_0(\lambda)}.
\tag{R40}
\]

Both are positive normalized likelihoods. The factor 1/2 in the binary natural parameter makes the record contribution to dot l equal to epsilon o. Using a binary likelihood with natural parameter epsilon instead changes that contribution to 2 epsilon o and requires the corresponding redefinition in the public equations. Every realized record remains in the objective. Multiple records add their binary and vector natural parameters.

Let nu_j be the density of the public marginal of q_j relative to the uniform sign reference: nu_j(z)=2 Pr_j(Z=z), so [nu_j(+1)+nu_j(-1)]/2=1. This density convention is required in the peer log cost; inserting the probability mass instead adds log 2 only to the nonself costs and incorrectly changes the Gibbs row. With exactly Gibbs-reduced unit-temperature source rows, the self and nonself costs are

\[
D_{ii}=\mathbb E_{q_i}\log q_i
-J\mathbb E_{q_i}[z n_i\cdot e(\phi)]+\log I_0(J),
\qquad
D_{ij}=\mathbb E_{q_i}\log q_i
-\mathbb E_{q_i}\log\nu_j(z).
\tag{R41}
\]

The same entropy term occurs in every cost, so it cancels in the normalized Gibbs row. In the own-block variation, peers are held fixed. The reduced derivative is the Gibbs-weighted own derivative of these costs. Consequently the original Fisher flow stays in (R36) and is exactly

\[
\begin{aligned}
\dot l_i&=-l_i+\sum_{j\ne i}\beta_{ij}L_j+\epsilon o_i,\\
\dot A_i&=-A_i+\lambda e(\omega_i),\\
\dot C_i&=-C_i+\beta_{ii}Jn_i,\\
\dot b_i&=-b_i,\qquad \dot\theta_i=0.
\end{aligned}
\tag{R42}
\]

One way to check the Fisher step is to collect the natural parameters of log q_i. The own derivative of the energy, modulo a constant, is log q_i minus the source-weighted log target minus the persistent log likelihood. Its centered negative is the replicator log-density velocity. Since all these functions lie in the span of z, e(phi), z e(phi), and e(psi), the natural-parameter velocity is the negative current parameter plus the weighted target and record parameters. This proves (R42), without using a Euclidean metric on the parameter vector.

Deleting the nonself costs as prescribed changes the reduced row and gives the corresponding null equations

\[
\dot l_i=-\beta_{ii}^{\varnothing}l_i+\epsilon o_i,
\quad
\dot A_i=-\beta_{ii}^{\varnothing}A_i+\lambda e(\omega_i),
\quad
\dot C_i=\beta_{ii}^{\varnothing}(Jn_i-C_i),
\quad
\dot b_i=-\beta_{ii}^{\varnothing}b_i,
\quad \dot\theta_i=0.
\tag{R43}
\]

The null row is recomputed from its retained self cost and row entropy. It is not the full row in (R42). The fixed peer kernels contain no hidden law dependence to freeze. In both systems the row and retained vector field are independent of theta. In particular, although the hidden concentration cancels from the full Gibbs row, it need not cancel from the null Gibbs row.

Define the parent by replacing every hidden factor v_b(psi-theta) with v_b(psi), retaining l,A,C,b, all primitive ports, source rows, the complete record history, event information, and inherited constitutive data. Its law is the normalized density (R36) with theta=0 on the same fixed sample space. Its vector field is (R42), or (R43) in the null comparison, with the theta coordinate omitted. Thus the map intertwines the original-time flows exactly. At every finite time b remains positive, so each discarded phase is a positive-dimensional physical distinction: the child diagnostic E exp(i psi)=m(b) exp(i theta) separates its values. That diagnostic is excluded from the fixed admitted boundary protocol. The marked-frame group below acts trivially on psi, so this loss is nongauge with respect to that group.

For any finite adaptive continuation of the declared protocol, take normalized response kernels depending only on retained public and marked variables, assimilated likelihoods of the form (R40), and policies using retained history. Strengths, query times, and queried primitive ports may be chosen adaptively. The complete joint kernels descend through this parent. The proof is induction: the current retained state gives the same next response law; conditional on that response, its persistent likelihood gives the same retained next-state update; repeat. The likelihood span in (R40) keeps the displayed finite family invariant. This establishes equality of response/next-state dependence as well as output marginals. It covers future records that change or destroy the original group. It does not assert that grouping or nonflatness survives every such continuation. Admitting arbitrary additional marked likelihoods would require enlarging the retained marked-law family; a psi-reading likelihood has the obstruction in (R34)-(R35).

All densities and kernels above are positive and analytic on the stated finite-dimensional chart. On a finite interval, A and C are bounded directly from (R42), b is bounded and positive, and (R39) bounds L-l by a bounded function of A,C. The l equations then give a finite bound by integration of their linear-growth inequality. The null equations have the same finite-interval bounds because 0<beta^empty<1. Thus a compact initial chart has a common regular finite-time tube. The ordinary finite-dimensional uniqueness and parameter-dependence hypotheses hold there. No Gaussian closure or independent fitted force is used.

### Frame action and the actual history experiment

The comparison group is a common left S1 action on the marked circle of the selected block. With R_g the plane rotation through g, its coordinate law is

\[
\phi'=\phi+g,\quad \omega'=\omega+g,
\quad A_i'=R_gA_i,\quad C_i'=R_gC_i,
\quad n_i'=R_gn_i,
\tag{R44}
\]

while z,l,b,psi,theta and discrete source labels are unchanged. Equation (R36) pushes forward under this action; (R38) and (R40) transform by the same pushforward. In (R39) the two norms are invariant, and all KL costs and Gibbs rows are invariant under simultaneous transformation of the inherited constitutive data. The inference equations are therefore covariant under the corresponding time-independent change of comparison frame.

This is passive frame covariance. With a fixed numerical n_i=e(0) and a fixed numerical record omega_i, rotating only q_i is generally not a symmetry of the fixed-data energy. It would be incorrect to claim otherwise. The physical hidden-phase rotations are a separate group H=(S1)^{number of children}; they preserve the fixed constitutive data and all admitted likelihoods. They are not included in the marked comparison group being minimized over.

For the geometry, fix one finite initial record episode. The record law is obtained by drawing the normalized per-agent generative predictive kernels, independently conditional on the prepared law configuration. This defines an operational product experiment and does not assume a global latent generative model for the interacting population. After this episode no additional assimilation occurs during the finite grouping/geometry window; every episode likelihood stays in the objective.

Let H denote the retained episode and let Q_{c,t}^h be the product of the selected children's fresh recognition laws and their Gibbs source-label laws at time t, conditional on H=h. For a contextual section c, the actual experiment is

\[
\mathcal P_{c,t}(dh,dy)=\Pi_c(dh)Q_{c,t}^h(dy).
\tag{R45}
\]

If the grouping witness uses a particular public record pattern E of positive probability, use Pi_c(dh|E), retain the event label, and include its normalization in the score. The marked angular records remain random in this conditional experiment. This is a conditional joint history/current-state experiment, not the conditional current-state law Q^h alone. The same argument below also applies to the unconditioned experiment for fixed primitive ports. If the selection probability itself is observed, adding its Bernoulli score adds a retained score and leaves the exact parent equality valid.

The complete score is

\[
s_X(h,y)=\partial_X\log\Pi_c(h)
+\partial_X\log Q_{c,t}^h(y),
\qquad
g^{\rm joint}=g^{\rm history}
+\mathbb E_H[g^{\rm conditional}].
\tag{R46}
\]

The cross term vanishes by normalization of every conditional Q^h. Source-label score terms are part of Q^h and are included. With conditional Pi given E, the first derivative also contains -partial_X log P_c(E). Formula (R46) applies unchanged with that normalized history law.

Choose contextual sections on which every theta_i is constant in c. They may be different for different children and different physical initial preparations. Since dot theta_i=0 and the protocol cannot read psi, the product coordinate shift psi_i to psi_i-theta_i is constant in context, time, and history on each such section. It sends (R45) to the actual parent experiment, has unit Haar Jacobian, and commutes with the marked-frame action (R44). Every contextual score and every marked orbit score is therefore carried over exactly. For each tangent X and comparison velocity xi,

\[
\mathcal C_{\rm fine}(X,\xi)
=\mathbb E_{\mathcal P}[(s_X+\xi s_J)^2]
=\mathcal C_{\rm parent}(X,\xi).
\tag{R47}
\]

Equality holds before minimization. It proves equality of the actual minimizing connection and minimized contextual tensor, including history information and source labels. It is not obtained by assigning the parent a copied metric. For sections with varying hidden theta, the parent loses the positive hidden phase Fisher term; (R47) is then false. The section restriction is retained explicitly.

### Uniform nonflatness on the same fixed finite window

Fix small positive numbers a_-<a_+, d, and b_i, independent of epsilon. They may all lie below the small initial-state bound used in the grouping route. Use the compact contextual patch

\[
c=(a,\chi),\quad a\in[a_-,a_+],
\quad A_1(0)=a e(\chi),\quad A_2(0)=d e(0),
\quad C_i(0)=0,
\tag{R48}
\]

with chi in a compact local interval, fixed context-constant hidden phases and concentrations, uniform initial presentation laws, and public initial l_i(0)=epsilon u_i(0) as in the grouping witness. Additional selected children can have fixed positive marked concentrations and merely add their inertias below. Take lambda=epsilon and J=epsilon^3. The grouping route's estimates are uniform on this compact marked patch and give a finite entrance time T, independent of epsilon, plus a fixed positive persistence window. Let T_+ be a finite endpoint containing both.

At epsilon=0, every episode likelihood is uniform, all public laws are uniform, all source costs within a row are equal, and beta equals its fixed prior. The episode law is independent of c, including when conditioned on E. The solution is independent of the realized records and is exactly

\[
A_i(t)=e^{-t}A_i(0),\quad C_i(t)=0,
\quad l_i(t)=0,\quad b_i(t)=e^{-t}b_i(0).
\tag{R49}
\]

Thus the full history/current-state experiment has precisely the same nonzero marked scores as two independent von Mises beliefs with concentrations s(t)=e^{-t}a and r(t)=e^{-t}d. Its history and source-label scores vanish at this limiting parameter, rather than being omitted by definition. Put

\[
I(x)=x\frac{I_1(x)}{I_0(x)}=x m(x),
\qquad B(t)=I(e^{-t}d)>0.
\tag{R50}
\]

For more fixed selected children, B(t) is the sum of their I values. In either case the complete minimizing connection, curvature, and tensor of (R45) at epsilon=0 are

\[
\begin{aligned}
\mathcal A_{0,t}
&=-\frac{I(e^{-t}a)}{I(e^{-t}a)+B(t)}d\chi,\\
\mathcal F_{0,t}
&=-\frac{B(t)e^{-t}I'(e^{-t}a)}
{[I(e^{-t}a)+B(t)]^2}da\wedge d\chi,\\
h_{aa}^{0,t}&=e^{-2t}V(e^{-t}a)>0,\\
h_{\chi\chi}^{0,t}
&=\frac{I(e^{-t}a)B(t)}{I(e^{-t}a)+B(t)}>0,
\qquad h_{a\chi}^{0,t}=0.
\end{aligned}
\tag{R51}
\]

Here V(x)=Var_{v_x}(cos phi)>0 and I'(x)=m(x)+xV(x)>0 for x>0. These identities follow from differentiation of the positive circle integral defining I_0; the phase information identity x^2 E sin^2 phi=xm(x) follows by integrating the derivative of sin(phi) exp[x cos(phi)] around the circle. The coefficient in (R51) is therefore nonzero everywhere on the compact contextual patch and 0<=t<=T_+.

This limiting family has strict positive lower bounds for the locked inertia, both tensor eigenvalues, and the absolute curvature coefficient. The lower bounds can be small when T_+ is large, but they are positive because the initial a and d are fixed independently of epsilon and T_+ is finite. No epsilon-dependent shrinking inertia is being compared to an absolute error estimate.

For completeness, continuity of the actual joint geometry can be checked directly. The finite-episode history density tends with two contextual derivatives to its uniform limit on its compact record sample space. Conditioning on E preserves this convergence since P_c(E) tends to a fixed positive number. On the regular finite-time tube the ODE coefficients and their first three state derivatives are bounded uniformly for sufficiently small epsilon. The first and second contextual derivatives of the flow satisfy

\[
\dot X_c=DV(X)X_c,
\qquad
\dot X_{cc}=DV(X)X_{cc}+D^2V(X)[X_c,X_c],
\tag{R52}
\]

with the corresponding inhomogeneous terms if a constitutive parameter depends on c. Integration of these linear equations and the elementary exponential bound for their integral inequalities gives uniform convergence of the flow and its first two contextual derivatives as epsilon tends to zero. The fixed-length initial record forcing is O(epsilon), J is O(epsilon^3), and the initial public l is O(epsilon). The same bounds are uniform over the compact episode values. Positivity and bounded derivatives of all densities give a common integrable bound for their Fisher-score products and derivatives.

Consequently the complete locked inertia K_epsilon, momentum one-form j_epsilon, and contextual metric B_epsilon in the cost

\[
\mathcal C_\epsilon(X,\xi)
=B_\epsilon(X,X)+2j_\epsilon(X)\xi+K_\epsilon\xi^2
\tag{R53}
\]

converge with the derivatives needed for the connection and curvature to the quantities in (R51), uniformly on the compact patch and finite time interval. Since K_0 has a positive minimum, inversion is continuous there:

\[
\mathcal A_\epsilon=-K_\epsilon^{-1}j_\epsilon,
\qquad
h_\epsilon=B_\epsilon-j_\epsilon\otimes j_\epsilon/K_\epsilon,
\qquad
\mathcal F_\epsilon=d\mathcal A_\epsilon.
\tag{R54}
\]

There is therefore epsilon_0>0 such that for every 0<epsilon<epsilon_0 the actual joint history/current-state tensor has rank two and its connection curvature is nonzero throughout the stated patch and time window. By (R47), the canonical parent has exactly those same quantities. The grouping and null-control strict inequalities may be imposed simultaneously by reducing epsilon_0 to the minimum of the two finite-window thresholds. No numerical lower bound for epsilon_0 is asserted.

The retained marked geometry participates in the fixed model when epsilon>0: C is driven by the self evaluator, L in (R39) depends on A,C, the Gibbs row contains the moment E[z n dot e(phi)], and the marked likelihood reads phi. Thus changing marked data changes public communication and source competition. The discarded psi phase remains a preselected invisible distinction. This construction does not show that the dynamics discovered an initially unavailable sufficient interface.

Under a context-dependent change of marked frame g(c), the full joint score transforms by s_X to s_X+(dg)(X)s_J, giving the local connection law A' = A-dg. Its abelian curvature is unchanged. The displayed contextual patch and principal S1 bundle are trivializable; the nonzero curvature nevertheless gives path-dependent connection transport. No endpoint-only transport is substituted for it. A global real lift of theta is unnecessary for the canonical parent: b and the canonical law can be defined without a phase lift on the regular b>0 stratum. Singular b=0 is outside the chosen patch, and no claim of smooth continuation through it is needed.

The integrated result is exact phase compression and original-time/history/record/geometry inheritance, together with an existence proof of nonflat actual joint geometry on the same fixed finite window as the separately proved public grouping witness. It is conditional on that route's grouping estimates, on the fixed initial record episode, and on the stated contextual sections. A genuinely new noninjective second join and endogenous creation of the invisible interface remain open; neither follows from reapplying the same phase canonicalization.

No scripts, numerical experiments, simulations, GPU operations, or Lean work were run by this route. This assigned derivation is the only workspace file edited in the current assignment; the Research vault was read without changes.
