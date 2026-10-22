<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-f83979013626f813aee847ebad2dcca2c0fa4f2687c0daf601404aed35c64a6a","schema_version":"rigorous-theory-search/v1","target_digest":"f83979013626f813aee847ebad2dcca2c0fa4f2687c0daf601404aed35c64a6a"} -->
# Finite-time operational equivalence and collective relative-angle records

Date: 2026-10-22. First-round analytical derivation for O1. This note proves a scoped obstruction to the creation of exact future equivalence during a regular analytic flow, then proves operational inheritance for the proposed collective common-rotation quotient. These results are compatible. A collective equivalence relation can exist before grouping without being an invisible primitive factor or an admissible quotient of every partition. Its prior existence does not invalidate the collective alternative in the frozen O1 contract.

The full target remains existential over general probability laws and at least C2 constitutive data. It requires current normalized per-agent own-block Fisher dynamics, original-time inheritance, every admitted joint adaptive response-record law, retained persistent records and event data, and actual nonflat collective geometry. The results here neither establish general nonexistence nor supply the independent grouping/null-control and geometric calculations. Second-stage recursion and future_TODO extensions are outside this note.

## 1. Sources, conventions, and assumptions

The current source reads were `Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex`, especially `sec:dyn-per-agent` and `def:dyn-per-agent-flow`; Chapter 5's deterministic descent, canonical parent, boundary, no-finite-entry, memory, and formation sections; the earlier [history-selection.md](../../2026-10-22-endogenous-interface-selection/routes/history-selection.md); and the current [integrated-grouping-bridge.md](../../2026-10-22-per-agent-formation/routes/integrated-grouping-bridge.md) and [operational-compression.md](../../2026-10-22-per-agent-formation/routes/operational-compression.md). The earlier history route uses a reciprocal potential in its displayed dynamics. Its probability-kernel arguments transfer, while its potential or dissipation claims do not transfer to the per-agent field.

The source model has normalized per-agent laws

\[
P_i(dm,dk,J=j,L=l,do)
=\pi^s_{il}\rho_{il}(dm)\pi^q_{ij}\kappa_{ij}(m,dk)\ell_i(do\mid k,m),
\tag{A1}
\]

positive fixed source priors, and persistent record likelihoods. Each recognition block belongs to one agent, and on finite positive tables its actual equation is

\[
\dot u_z^a=-u_z^a\left(h_z^a-\sum_wu_w^ah_w^a\right),
\qquad h_z^a=\partial_{u_z^a}\mathcal F_{i(a)}.
\tag{A2}
\]

Peer laws are current parameters of that agent's functional; all specified own-law dependence of channels or evaluators is differentiated. No derivative of another agent's objective is inserted. The candidate uses exactly Gibbs-reduced source rows at unit temperature, including in the prescribed null. Neither a global free-energy potential nor monotonicity of a sum of agent free energies is assumed.

The state is the law configuration together with all dynamical rows, record potentials, accessible past records, current labels, and any accessible clock. For an exactly reduced-row model, rows are determined functions instead of extra coordinates. Between records, the current history determines an autonomous field on its regular stratum. A new evidence record appends its value and adds its negative log likelihood to the persistent objective; it does not reset the current recognition laws. A diagnostic response is appended to the accessible record but, in the protocol of Section 7, is not assimilated as an extra likelihood term. These are two declared transaction types. No distribution over initial law configurations is presumed.

Time below is the original inference time. There is no discretization and no diffusion. Randomness is only that of the explicitly normalized response kernels. All flow claims concern common regular intervals inside the declared chart; no continuation across a law-support boundary or a blowup is assumed. The contract supplies C2 regularity and unique local flow. Finite dimension, real analyticity, sufficiently many waiting-time experiments, and regular event surfaces are additional hypotheses of specific results below.

## 2. An analytic continuation lemma with its exact domain

Let M be a finite-dimensional real analytic manifold and V an analytic autonomous field with local flow \(\Phi_t\). Fix a retained history h, so no record is appended while following this field. Let \(x_t=\Phi_t x\) and \(y_t=\Phi_t y\) exist on the same connected open interval I. For each scalar response test f used below, assume f is analytic on an open set containing both trajectories over I.

The local analytic-flow fact needed here is the following: an analytic field has a flow jointly analytic in time and initial condition on each local flow neighborhood. One direct justification is to extend its convergent coordinate power series to a small complex polydisc. On a smaller closed polydisc the extension and its first derivative have finite bounds M and L. For sufficiently small complex time radius r, with rM smaller than the distance to the polydisc boundary and rL<1, the Picard maps

\[
z_{n+1}(t,z_0)=z_0+\int_0^tV(z_n(s,z_0))ds
\]

remain in the polydisc and contract uniformly. Each iterate is holomorphic in \((t,z_0)\); uniform convergence on smaller polydiscs gives a holomorphic solution. Restrict to real arguments and patch along the existing trajectory by uniqueness. This argument gives local analyticity and does not assert completeness.

Lemma A1. If

\[
f(x_t)=f(y_t)\quad\text{for every }t\in(T,T+\epsilon)\subset I,
\tag{A3}
\]

then \(f(x_t)=f(y_t)\) for every \(t\in I\).

Proof. The difference \(d_f(t)=f(x_t)-f(y_t)\) is real analytic on I. An analytic scalar function with zeros accumulating at an interior point is locally zero: otherwise its first nonzero Taylor coefficient factors it as \((t-t_0)^k g(t)\), with \(g(t_0)\ne0\), contradicting accumulating zeros. Starting from the open interval in (A3), the set reached by local zero continuation is open in I. At an interior endpoint its derivatives are zero by continuity from the zero side, so its convergent Taylor series is zero and continuation passes that endpoint. It therefore covers connected I. This proves the claim separately for every f. No inverse flow outside I occurs in the proof.

For a standard Borel response space, choose a countable measure-determining family of bounded measurable tests. Require the corresponding response expectations to be analytic in the law-state chart. Equality of all these scalar expectations gives equality of the response measures. Merely declaring a density pointwise analytic in the state does not automatically justify analytic integration on a noncompact sample space. An integrable local complex bound or another valid analytic-integration argument is required. Finite sums and compact-space exponential families on compact parameter neighborhoods satisfy this condition directly.

Corollary A1, passive probes. Suppose the same fixed passive probe kernels \(K_a\) are available throughout I, and their measure-determining expectations are analytic. If every probe has the same law along the two trajectories throughout a nonempty post-T time interval, all the probe laws agree along their entire common interval. Consequently exact persistent passive indistinguishability cannot first appear at a finite regular time on these paired trajectories. Equality at only the instant T is insufficient: the analytic scalar function \(d(t)=t-T\) already shows the missing implication.

This statement concerns existing trajectories. It neither requires a globally invertible flow nor says that every post-handoff state has a predecessor at time zero. For a proposed finite-time fiber, backward comparison is made only between members actually reachable from the admitted initial class.

## 3. Adaptive experiments and analytic response-record functions

For a state e in one fixed-history stratum, an admitted finite experiment E specifies its waiting times, adaptive action rules, normalized responses, and actual record/assimilation updates. Let \(\mathbb P_e^E\) be its joint record law, including all accessible action labels and times. For each member \(\varphi\) of a determining test family on that joint record space, define

\[
f_{E,\varphi}(e)=\int\varphi(z_1,\ldots,z_n)\mathbb P_e^E(dz_1\cdots dz_n).
\tag{A4}
\]

Operational equivalence for this declared experiment family means equality of every function (A4). This is a pointwise definition at a given fine state and requires no preparation law over configurations. Actions allowed by the actual protocol are used; arbitrary additional interventions are not inferred from the word adaptive.

For finite record alphabets with finitely many branches, analytic kernels and analytic fixed-record updates give analytic finite-history probabilities wherever all branch compositions are defined analytically. Explicitly, with branch state recursion \(e_k=U_{a_k,z_k}(\Phi_{\delta_k}e_{k-1})\), a branch weight is

\[
\prod_{k=1}^n p_k(a_k,\delta_k\mid z_{<k})
K_{a_k}(\Phi_{\delta_k}e_{k-1},z_k).
\tag{A5}
\]

It is a product of analytic compositions when the policy depends only on the discrete past, or depends analytically on any continuous retained coordinates that it reads. Sum over branches to obtain (A4). The maps U need not be injective for this analytic calculation. For general record spaces or arbitrary continuous-record policies, analyticity of (A4) is a separate obligation, with uniform integrability and moving-domain issues checked. Analytic kernel formulas alone do not settle it.

Theorem A2, no new equivalence for an analytic, wait-closed experiment family. Suppose every function (A4) is analytic on one common regular domain containing \(e_t=\Phi_t e\) and \(e'_t=\Phi_t e'\) for a connected interval I. Suppose the predeclared experiment family is closed under prefixing every sufficiently small deterministic wait on this domain, with no intervening record or change of constitutive stratum. If \(e_T\) and \(e'_T\) are operationally equivalent, then they are equivalent at every earlier time in I. If the same wait closure holds along the interval, equivalence is preserved forward there as well.

Proof. For fixed E and \(\varphi\), apply the equivalence at T to the admitted experiment consisting of a wait s followed by E. For all sufficiently small \(s\geq0\), its expectation is \(f_{E,\varphi}(\Phi_s e_T)\), equal to the corresponding expectation from \(e'_T\). Lemma A1 continues this equality along I. Do this for every E and determining test. For forward propagation, a distinguishing experiment after any sufficiently small wait would be a distinguishing prefixed experiment beforehand. Iterate on the existing interval. All response/update dependence is already inside the joint functions (A4), so no assumption of independent response and transition noise is used.

The common domain is load-bearing. If a proposed experiment ceases to exist, hits an unhandled boundary, or crosses an unsupported event stratum when started earlier, analytic continuation of its probability has not been justified. A fixed terminal horizon can likewise prevent closure under waiting. One may state the theorem for a smaller common family and common window, but that does not prove equivalence for experiments omitted by that restriction.

For passive probes, equality of analytic Lie derivatives \(L_V^k f\) for all \(k\geq0\) at a state pair is equivalent to equality of their analytic time traces on the common connected interval, since these are exactly the Taylor coefficients. No finite derivative truncation is claimed. This is the local observation-function viewpoint developed in Hermann and Krener, [Nonlinear Controllability and Observability](https://www.math.ucdavis.edu/~krener/1-25/10.IEEETAC77.pdf), IEEE TAC 22 (1977), 728–740; the proof used here is the direct continuation argument above, with no import of that paper's global completeness assumptions or controlled-system equivalence claims.

The theorem does not say that a preexisting collective equivalence is a hidden primitive factor. It imposes no product decomposition on its fibers, does not make a proper block sufficient when exterior agents remain accessible, and does not show that the quotient is available for every partition. Those are different assertions. Thus it blocks a claim of newly created equivalence during an analytic segment, while allowing the collective construction of Section 7.

## 4. What finite observations and C2 regularity do not imply

Even analytic dynamics and positive analytic observations do not make a discrete observation schedule equivalent to arbitrary waiting-time experiments. Here is a mathematical failure test, not an O1 witness. On \(M=\mathbb R\times(-1,1)\), set \(\dot s=1\), \(\dot a=0\), and

\[
K_{s,a}(1)=\frac12+\frac a4\operatorname{sinc}(\pi s),
\qquad \operatorname{sinc}(\pi s)=\begin{cases}\sin(\pi s)/(\pi s),&s\ne0,\\1,&s=0.\end{cases}
\tag{A6}
\]

The response is strictly between 1/4 and 3/4 and analytic. If only immediate and integer-wait samples are admitted, with fresh independent response bits, the states \((0,a)\) and \((0,b)\), \(a\ne b\), differ in their immediate law. Their successors \((1,a)\), \((1,b)\) have identical entire integer-grid response laws because every sampled sinc value is zero. An arbitrary noninteger wait distinguishes them. The family fails the wait-closure hypothesis of Theorem A2. This example is an observation-schedule warning, not an admissible proof of O1: it has no normalized per-agent mechanism, grouping, null comparison, or nonflat inherited geometry. One also may not turn it into a solution by retiring informative probes at handoff.

For merely C2 data, the continuation proof stops at the exact line requiring analytic continuation. If a C2 scalar difference vanishes for all \(t\geq T\) nearby, its value and first two derivatives at T vanish, but this does not determine its past. The scalar function \((T-t)_+^3\) demonstrates that implication's failure. For smooth data, a one-sided flat function gives the corresponding failure even with every derivative zero at T. The frozen target expressly excludes inserting such a flat erasure gate as a construction. These examples identify the logical gap in an attempted universal theorem; they do not supply admissible witnesses.

A separate C1 obstruction can survive without analyticity. If the claimed equivalence relation is a closed embedded submanifold of \(M\times M\), without boundary on the regular neighborhood, and \((V,V)\) is tangent to it, uniqueness makes it locally invariant in both time directions. An existing paired trajectory cannot first enter it at finite regular time: times lying in that submanifold are closed by continuity and open by local backward uniqueness. On a connected existence interval, a nonempty such set is the whole interval. An arbitrary C2 operational equivalence relation need not be such an embedded submanifold or have this two-sided local structure. Its singularities or boundary cannot be silently removed from the proof.

## 5. Retained past, records, and stochastic assimilation

Retaining the past means retaining its realized values and their dependence with subsequent records. It does not generally mean retaining the function that assigns each alternative initial fine state a probability for that past. Let \(H_\tau\) include the handoff time, full accessible history, and labels. Under each fixed initial fine state x, suppose a proposed parent reproduces the fine conditional future kernel \(F\) after the actual retained handoff state \(R(X_\tau,H_\tau)\). Then

\[
\mathbb P_x(H_\tau\in dh,\text{future}\in dw)
=\mathbb P_x(H_\tau\in dh)
F(R(X_\tau(h),h),dw)
\tag{A7}
\]

in a deterministic conditional-state realization, and the same formula uses the declared conditional state distribution when the fine state remains stochastic. Substituting the equal parent future kernel preserves the original joint law for that same initial x. It does not require \(\mathbb P_x(H_\tau\in dh)=\mathbb P_y(H_\tau\in dh)\) for two different initial states. In particular, coincidence of a realized finite record does not prove predictive equivalence, while differing probabilities of that past do not by themselves rule out a valid conditional handoff after it.

For a deterministic assimilation update \(U_z\), the actual one-event object is the joint kernel

\[
J_e(dz,de')=K_e(dz)\delta_{U_z(e)}(de').
\tag{A8}
\]

For a stochastic assimilation, replace the Dirac measure by the actual conditional transition, preserving its dependence on z. A proposed retention R must make \((\mathrm{id},R)_*J_e\) constant on R fibers. Equality of the z marginal and equality of the next-R marginal separately are insufficient. This is the boundary criterion already stated in Chapter 5.

For positive finite record alphabets and deterministic U, equality of the entire future record law before an event has the recursive form: equal next-record probabilities and, for each record of positive common probability, equivalent successors. Equality of the successors on only one realized branch does not imply this condition. Even equality on every branch does not recover unequal next-record probabilities. Therefore no backward equivalence theorem across arbitrary stochastic record transactions follows from Theorem A2. One must prove an equivalence-reflection property of the actual joint event kernel, or prove that the full operation is already represented by the analytic wait-closed functions on the compared pre-event stratum.

Noninjective assimilation can erase distinctions despite retaining z if distinct input states have the same complete post-event state \((U_z(e),h\mathbin{\frown}z)\). Keeping z restores injectivity only when the joint update actually encodes the lost information. Conversely, an analytic reset to a constant is noninjective despite analytic formulas; analyticity alone is no safeguard at a jump. Such inserted reset maps are forbidden by O1, so this observation is a scope limitation, not a proposed escape.

The source convention is stronger than an arbitrary assimilation: q and s are unchanged at the instant of a record, h is appended, and the persistent objective is updated. On each fixed-record branch this full state update is injective; remove the final record and keep the unchanged law coordinates. A finite sequence of these appends and regular deterministic flow segments is injective when all complete fine coordinates and records are kept. This does not prove that the future operational equivalence relation is reflected backward across the append: the next stratum has a different record-conditioned vector field. That additional assertion needs its own argument.

For comparison, ordinary Bayesian multiplication by a fixed strictly positive likelihood also has an injective fixed-record update on its admissible domain. If

\[
q^z(dk)=\frac{L_z(k)q(dk)}{\int L_z dq},\qquad L_z>0\quad q\text{-almost surely},
\tag{A9}
\]

then \(q\) is recovered by normalizing \(L_z^{-1}q^z\). Equal posteriors and the same known positive likelihood imply equal priors wherever these finite normalizations exist. Zero likelihoods or a likelihood that depends on unknown discarded state invalidate that inverse. The current construction uses record append, not an instantaneous Bayesian posterior replacement.

For general stochastic evolution, replace trajectory comparisons by the corresponding joint-law or conditional-kernel identities. Analyticity in time of semigroup expectations, injectivity of a transition operator, or conditional sufficiency may give additional obstructions, but none follows just from C2 constitutive data. One-time marginal agreement or moment matching cannot substitute for (A8).

## 6. Regular event times do not supply free compression

Suppose a selection event is the transversal crossing of \(g(x)=0\), with g and V C1 and \(Dg(V)\ne0\) at the crossing. Locally its time \(\tau(x)\) is differentiable, and its event-state map is \(H(x)=\Phi_{\tau(x)}x\). Differentiating \(g(H(x))=0\) gives

\[
D\tau_xv=-\frac{Dg_{H(x)}D\Phi_{\tau(x),x}v}{Dg_{H(x)}V(H(x))},
\qquad
DH_xv=D\Phi_{\tau(x),x}v+V(H(x))D\tau_xv.
\tag{A10}
\]

The local implicit-function assertion used here applies because the time derivative of \(g(\Phi_t x)\) is the nonzero denominator. The fixed-time flow derivative is invertible on the existing local flow tube: composing with its local negative-time solution gives the identity. Thus

\[
DH_xv=0,\ D\tau_xv=0\quad\Longrightarrow\quad v=0.
\tag{A11}
\]

The joint map \(x\mapsto(H(x),\tau(x))\) is locally injective, with inverse \(x=\Phi_{-\tau}H\) on that tube. The event-state map alone loses the position along an orbit, since \(DH(V)=0\), but storing the event time restores it. No global inverse beyond the common tube is invoked.

For a further quotient R at the event, a genuinely lost tangent must satisfy both \(DR_{H(x)}DH_xv=0\) and \(D\tau_xv=0\), along with the constraints from all retained history coordinates. Two states with unequal stored event times are not a fiber of the complete retention. If equal times T are stored and the event merely changes a label, the analytic comparison of Section 3 applies on the unchanged flow segment. Merely stopping or relabeling an analytic trajectory does not create a new response equivalence. If an event changes constitutive laws or the supported protocol, continuation across it requires a new proof. An imposed deletion of probes, retirement rule, or reset is excluded by the contract.

## 7. The proposed collective relative-angle protocol

This section addresses the concrete proposal supplied during the current run. For N agents, retain the original public and marked-phi sectors and their evidence laws. In the psi sector let

\[
q_i^\psi(d\psi)=f_{B_i}(\psi)H(d\psi),\qquad
f_{B_i}(\psi)=\frac{\exp(B_i\cdot e_\psi)}{I_0(|B_i|)},
\quad B_i=b_i e_{\theta_i},\ b_i>0,
\tag{A12}
\]

where \(H(d\psi)=d\psi/(2\pi)\). Work on a regular window with at least one nonzero B and, for the proper-block distinction below, every \(b_i>0\). The common circle action rotates every B by the same angle alpha, leaves all other law coordinates fixed, and leaves every stored record value fixed. This is declared a physical common phase, distinct from the passive frame/gauge action used in the geometric sector.

The diagnostic action names a subset E of the edges of the complete graph. The protocol also admits direct samples from each agent's complete public/marked-phi marginal and its model law s. These additional samples are diagnostic records and are unchanged by the psi rotation. At one relative-angle query, sample independent \(\Psi_i\sim q_i^\psi\), once for each involved agent, and output the complete vector

\[
D_E(\Psi)=(\Psi_i-\Psi_j)_{(i,j)\in E}\pmod{2\pi}.
\tag{A13}
\]

Edges sharing a vertex share its sampled angle. The outcome is appended to the diagnostic history and leaves all laws and evidence potentials unchanged. In particular, no relative-angle assimilation has been asserted. Evidence transactions use the original strictly positive normalized bit-plus-marked-phi likelihood, independent of psi, and append that likelihood to the persistent objective. Adaptive action and waiting-time rules may depend on the complete recorded past and retained invariant coordinates. A policy that reads an unretained absolute psi phase is not covered.

The selection protocol records its graph/partition at the predeclared census times 0 and T. The deterministic T is chosen uniformly on the admitted initial/contextual patch; strict grouping margins make the recorded labels constant on the patch used for the geometric comparison. Both the times and labels are retained. A continuously observed first-crossing time is not part of this fixed protocol. No action is retired at T. This convention avoids treating a varying deterministic hitting time as a smooth positive-density observation, and it does not use event-time omission to hide an accessible record.

Proposition A3, joint response invariance. The normalized law

\[
K_E(B,\cdot)=(D_E)_*\bigotimes_i q_i^\psi
\tag{A14}
\]

is invariant under simultaneous rotation of all B. It preserves the full dependence among all reported edges.

Proof. Couple the rotated experiment by replacing every \(\Psi_i\) by \(\Psi_i+\alpha\). This has the rotated product input law because \(f_{R_\alpha B_i}(\psi+\alpha)=f_{B_i}(\psi)\), while every difference in (A13) is unchanged. Pushforward of a probability law is normalized. Since the coupling uses the same complete vector of sampled angles, the conclusion concerns the entire joint response, not its edge marginals.

For connected E, choose a spanning tree and root r. Coordinates \(d_i=\psi_i-\psi_r\), with \(d_r=0\), give the joint density on the N-1 independent differences

\[
k_B(d)=\int_{\mathbb T}\prod_i f_{B_i}(\alpha+d_i)H(d\alpha).
\tag{A15}
\]

This integrates to one by the torus change of variables \((\alpha,d)\leftrightarrow\psi\). For disconnected E use one common angle per component and integrate their product Haar measure. Additional edges are deterministic functions of the tree differences. In particular a triangle's three reported differences satisfy a cycle constraint; assigning them an independent product density would change the experiment. The appropriate reference for redundant outputs is \((D_E)_*H^{\otimes N}\), supported on that constraint set. Positivity and analytic dependence on finite B follow from compact angular integration and a locally nonzero normalization denominator; every bounded measurable response test therefore has an analytic expectation in B.

The common-rotation quotient discards a genuinely collective phase in this protocol. It does not discard an unobservable phase of any isolated primitive. Indeed, for each admitted edge,

\[
\mathbb E e^{i(\Psi_i-\Psi_j)}
=A(b_i)A(b_j)e^{i(\theta_i-\theta_j)},
\qquad A(b)=\frac{I_1(b)}{I_0(b)}>0\quad(b>0).
\tag{A16}
\]

The formula follows by independence and rotation of each circular integral. Positivity of I1 for b>0 follows by pairing the regions with cosine c and -c: their contribution is proportional to \(c\sinh(bc)>0\) for c>0. If a proper nonempty subset I is rotated by \(\alpha\notin2\pi\mathbb Z\) while an exterior agent remains fixed, a crossing edge changes (A16) by \(e^{\pm i\alpha}\). Its response law changes. With all crossing edges admitted, no such proper-block rotation can be forgotten while preserving the declared boundary protocol. An absolute-angle diagnostic distinguishes the global rotation at the child level, but it cannot also be an inherited future boundary action if that same physical phase is discarded.

When N is at least three and all b are positive, the pair response laws also identify the b values from the moduli of (A16): for distinct i,j,k,

\[
A(b_i)=\sqrt{\frac{|m_{ij}||m_{ik}|}{|m_{jk}|}},
\qquad m_{ij}=\mathbb E e^{i(\Psi_i-\Psi_j)}.
\tag{A17}
\]

The map A is strictly increasing because \(A'(b)=\operatorname{Var}_{q_b}(\cos\Psi)>0\). Relative arguments of m determine the phase differences. Thus on this finite-dimensional family the complete pair protocol identifies the psi configuration up to the common rotation, rather than silently losing further amplitude data. This is identifiability from exact response laws, not an estimator from finitely many samples. If amplitudes are retained directly, (A17) is unnecessary for sufficiency.

There is a stronger proper-block statement than the rotation test. Fix any outside agent j with \(b_j>0\), and write \(m_i=A(b_i)e^{i\theta_i}\). Every crossing response identifies \(m_i=m_{ij}/\overline{m_j}\) for each inside agent i. Strict monotonicity of A then identifies its whole vector B_i. Thus, with the exterior state fixed, no noninjective retention of a proper block's B variables preserves all those crossing responses. In the candidate's product family \(q_i=q_i^{z,\phi}\otimes q_i^\psi\), the direct public/marked-phi and model-law queries identify the remaining primitive laws. The collective loss therefore cannot be assigned to a permanently invisible primitive coordinate in this admitted family.

## 8. Joint response/next-state descent and complete adaptive histories

Let G be the common circle action on the full augmented fine state, with all accessible records fixed under it, and let R denote the orbit retention. Suppose the actual own-block field is G-equivariant on every reachable evidence-history stratum,

\[
V_h(gx)=Dg_xV_h(x),
\tag{A18}
\]

and every evidence transaction has an invariant response law and an equivariant fixed-record append update. These are explicit hypotheses to verify in the candidate's full constitutive formulas. A sufficient condition for (A18) is invariance of each \(\mathcal F_i\) under simultaneous rotation, together with invariance of each agent's own Fisher metric and the product action on its own coordinates. Differentiating \(\mathcal F_i(gx)=\mathcal F_i(x)\) in an own-block direction gives covariance of that own differential; raising its index with the transformed Fisher metric gives (A18). This argument is valid for the per-agent field and uses no common potential. The bit/phi likelihood is psi-independent, so its persistent addition does not break this symmetry.

Theorem A4, operational inheritance by the collective orbit. Under (A18) and the stated transaction hypotheses, the orbit retention has exact original-time free evolution and preserves every finite joint adaptive history of the protocol in Section 7, including all action labels, waiting times, diagnostic outcomes, evidence outcomes, and retained event labels/times whose selectors are G-invariant. The response and next retained state are preserved jointly. The statement applies to every admitted branch that stays in the common regular domain; a full-window claim must show that every admitted branch does so, or provide compatible quotient charts covering those branches. Observations cannot be discarded when they leave a preferred chart.

Proof. Uniqueness applied to (A18) gives \(\Phi_t(gx)=g\Phi_t(x)\) on the common regular interval. Therefore \(R\Phi_t\) is constant on each orbit and defines the parent free evolution in the same t. On a free regular action chart, R is a submersion and differentiation gives \(DR V=\bar V R\); no rescaling is made.

Write \(J^a_e(dz,de')\) for the actual joint response and next-state kernel of a transaction after its declared wait. It is enough that

\[
J^a_{ge}=(\mathrm{id}_{z}\times g)_*J^a_e.
\tag{A19}
\]

For relative diagnostics, Proposition A3 gives the response invariance, the law state stays fixed, and the same invariant record z is appended. Thus (A19) holds. For evidence, it follows from the assumed invariant normalized evidence law and equivariant append of its persistent likelihood. If response and update use shared random variables, the requirement remains (A19) for their actual joint kernel; separate marginal checks are not substituted.

Since \(Rg=R\), the probability kernel

\[
\bar J^a_{R(e)}=(\mathrm{id}_z\times R)_*J^a_e
\tag{A20}
\]

is well defined and normalized. Locally choose a section by setting \(\theta_1=0\) when \(B_1\ne0\); different representative choices give the same kernel by (A19). For this circle action the global orbit space is also standard Borel: identify the orbit by \(b_1\) and the vectors \(R_{-\theta_1}B_i\), together with the unchanged coordinates, on the chart \(B_1\ne0\). Thus no stochastic preparation or averaging over configurations is needed to define (A20).

At equal retained states and equal histories, the same adaptive policy supplies the next action and wait law. The parent wait is exact by flow equivariance, and (A20) gives the same joint next record and next retained state. Iterating these equal conditional kernels proves equality of every finite joint adaptive history. The proof includes correlations between a response and subsequent evolution because it descends their joint kernel at each step. Carrying the original handoff history and applying (A7) gives equality of its joint dependence with every future record.

For completeness, local retained coordinates \(b_1\) and \(\widetilde B_i=R_{-\theta_1}B_i\) have the exact derivatives

\[
\dot b_1=\frac{B_1\cdot V_1}{b_1},\qquad
\Omega=\dot\theta_1=\frac{(JB_1)\cdot V_1}{b_1^2},\qquad
\dot{\widetilde B_i}=R_{-\theta_1}(V_i-\Omega JB_i),
\tag{A21}
\]

where J is rotation by \(\pi/2\). Equivariance makes each right-hand side independent of the representative's common phase. These are quotient derivatives of the actual fine field, not fitted parent forces.

The theorem is operational and dynamical. Its R is the orbit quotient of the state at a fixed contextual point. The assembled geometric parent may use a finer retention of the whole contextual section, canonicalizing by one common rotation fixed from an anchor of that section. A context-independent rotation preserves contextual derivatives by the same constant linear map. Choosing a different rotation at every contextual point introduces derivatives of that rotation and does not follow from the operational proof. Thus pointwise response equivalence must not be substituted for the whole-section canonical parent needed by the geometric route. A finer section retention still inherits the operational kernels because they already factor through the pointwise orbit.

The theorem does not prove that a particular normalized parent belief/model construction computes the required collective Fisher tensor or minimizing connection, that the selected block forms, or that its partition sequence differs from the prescribed null after event times are removed. Those remain obligations of the other routes. It also does not authorize assimilating relative-angle diagnostic records into a new joint likelihood: such assimilation could leave the proposed finite-dimensional family and would need a fresh derivative, closure, and symmetry calculation.

## 9. Failure test and remaining obligation

For a claimed new finite-time equivalence during one analytic segment, exhibit two actual fine states with equal complete retention at T, including history and event times. Fix any admitted finite adaptive experiment that distinguished their backward states. Check its analytic expectation on the entire common flow tube and verify that prefix waits are still admitted. Theorem A2 then contradicts the claimed post-T equivalence. A single persistent, positive-support separating action suffices; tiny probability is not zero probability.

For the collective candidate, a proper-block phase rotation fails through the cross-edge moment (A16). A global rotation passes the response test (A14), but must also pass the own-block field test (A18), the actual joint event test (A19), and the independent normalized-law and geometric tests. Any absolute psi probe retained in the inherited protocol, any phase-sensitive evidence likelihood with its absolute outcome kept, or any state-reading policy that depends on the discarded absolute phase breaks the stated quotient. Replacing jointly sampled edge differences by independent edge outputs also breaks the declared experiment.

The strongest established result is a conditional exact operational parent for the collective circle symmetry, together with a proved obstruction to newly appearing equivalence on one analytic, wait-closed, fixed-history flow segment. The latter leaves the general C2/general-law existential target open and does not invalidate a collective relation that exists before selection. The minimal remaining obligation for the candidate is to verify (A18) and evidence equivariance from its complete normalized per-agent formulas, then combine this operational quotient with the actual grouping/null result and the actual nonflat collective geometry. An admissible alternative route through nonanalytic data or an event must supply its concrete native mechanism and full joint kernels; a forbidden erasure gate, deleted action, or inserted reset cannot fill that gap.

Only source reads, one bounded primary-source lookup, and analytical derivations were performed. No numerical or symbolic script, simulation, process pool, GPU query/use, Lean work, or Research-vault edit ran. This route file is the only written artifact of this assignment. The primary-source PDF was located at the author's institutional archive; the proofs in this note are supplied explicitly rather than relying on an unexamined external theorem.
