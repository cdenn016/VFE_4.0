<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca","schema_version":"rigorous-theory-search/v1","target_digest":"0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca"} -->
# Exact operational compression under per-agent mixture-source inference

This route constructs an exact noninjective parent for a declared operational interface. Its receiver-reconstruction version uses a pointwise normalized, recognition-dependent per-agent model: private conditional distributions affect the self-versus-peer source rows, and those rows determine their relaxation rates. Section 6 also gives a fixed-channel version whose model stays fixed during an own-block update with peers held fixed. A regular circle-family realization retains actual normalized laws, nonconstant persistent record likelihoods, the original-time dynamics, and every finite adaptive history of its predeclared marked observations. It discards positive-dimensional physical phase data. The construction does not derive why this interface should be chosen. Its hidden phase is absent from the operational probes from the outset, and the quotient exists before any grouping event. It therefore supplies a conditional inheritance mechanism, not a closure of the frozen formation target.

## 1. Scope, source continuity, and state space

The frozen target asks for an existential construction with interaction-dependent grouping, compared with the prescribed null flow, and compressive inherited parents with exact dynamics, records, and geometry. This assignment addresses the operational compression and history clauses. The mandatory four-state transfer, grouping/null-control result, and geometry calculation belong to the other routes. No inference about those clauses follows from the quotient below.

The current `AUTHORING.md`, Chapter 3 `sec:dyn-per-agent`, Chapter 5's deterministic, canonical-parent, boundary, memory, null-flow, and formation results, Chapter 7 `07_smooth_closure.tex`, and Chapter 9 `09_public_compression.tex` were read first. The earlier `history-selection.md` and `selection-observability-bridge.md` were also read. Chapter 7 already constructs a conditional-spectrum completion for the reciprocal objective. The calculation here changes its dynamics: in the per-agent model the private KL is weighted by the inferred self-source probability, which also reads that private KL. The earlier reciprocal formulas cannot simply be copied. Chapter 9's public phase parent does not preserve ordinary child sampling; the present response class is specified explicitly before retention.

Fix a finite population of identical species, with public alphabets \(Z,W\), private probability spaces \((V,A^q),(U,A^s)\), and

\[
K=Z\times V,\qquad M=W\times U.
\tag{O1}
\]

Use counting measure on the finite public alphabets and the stated probability measures on private spaces. Let

\[
q_i(z,dv)=\mu_i(z)\kappa_{i,z}(dv),\qquad
s_i(w,du)=\sigma_i(w)\varkappa_{i,w}(du).
\tag{O2}
\]

All public probabilities are positive. Private log densities relative to \(A^q,A^s\) are bounded on the regular domain under consideration. No law over configurations is supplied or required. Recognition is exactly the original product \(q_i\otimes s_i\).

Every agent can attend to every other agent. For definiteness choose fixed homogeneous priors \(\pi_{ii}=p\in(0,1)\), \(\pi_{ij}=(1-p)/(N-1)\) for \(j\ne i\), separately in the two sectors if desired. There are no supplied blocks or graph labels. Use exactly optimized source rows at unit temperature throughout, including in the null flow. A choice of dynamic rows would be another flow.

Choose a strictly positive normalized public evaluator \(G^c(w,z)\), a positive public model reference \(r^c(w)\), and fixed normalized public channels \(B_{ij}^q,B_{ij}^s\). Identity public channels are permitted because their compared public laws remain positive. Set

\[
G((w,u),d(z,v))=G^c(w,z)A^q(dv),\qquad
r(dw,du)=r^c(w)A^s(du).
\tag{O3}
\]

The nonself channels reconstruct the receiver's actual conditional law:

\[
\mathcal K_{ij}^q((z_j,v_j),d(z,v))
=B_{ij}^q(z_j,z)\kappa_{i,z}(dv),\qquad j\ne i,
\tag{O4}
\]

and analogously in the model sector. They are normalized, law dependent, and homogeneous when the public channels are homogeneous. The self source is (O3), not (O4). The selected grouping is not encoded in these kernels. The public/private distinction is, however, declared interface data; the construction does not derive it.

Fix marks \(T^q:V\to D^q\), \(T^s:U\to D^s\). Allowed record actions have positive normalized likelihoods of the form

\[
L_a^q(o^q\mid z,T^q(v))L_a^s(o^s\mid w,T^s(u)).
\tag{O5}
\]

For each finite accumulated record list, let \(W_i^q(z,d)\) and \(W_i^s(w,e)\) be the corresponding sums of negative log likelihoods, including the action labels. These terms remain in the objective between later records. A new record changes \(W\) and the stored history; it does not instantaneously reset q or s. The factorization in (O5) is a route-specific assumption that preserves the product recognition convention.

The continuous state consists of all current laws and their record potentials; the discrete state includes the recorded actions, outputs, event times, and current membership labels. Between events the dynamics is deterministic. At a record event the randomness is exactly the declared normalized observation kernel. No diffusion, numerical time discretization, or new parent attraction is introduced.

## 2. Normalization and the complete own-block derivative

Write \(\nu_{ij}^{q,c}=\mu_jB_{ij}^q\), and similarly for the model sector. For every regular configuration, including correlated public/private laws,

\[
q_j\mathcal K_{ij}^q=\nu_{ij}^{q,c}\kappa_i,
\qquad
\mathrm{KL}(q_i\Vert q_j\mathcal K_{ij}^q)
=\mathrm{KL}(\mu_i\Vert\nu_{ij}^{q,c}).
\tag{O6}
\]

The proof is cancellation of \(\kappa_{i,z}\) in the density ratio, followed by conditional integration. The identity holds as the receiver law changes. Differentiating it therefore includes the derivative of \(\kappa_i\) inside \(\mathcal K_{ij}^q\). Freezing that channel while differentiating q would produce an extra private peer force and would not be the specified model. This is the load-bearing full-channel derivative.

There is a modeling distinction here. Write the normalized model as \(P_i[x]\). Because (O4) reads the receiver's current conditional, changing \(x_i\) changes \(P_i[x_i;x_{-i}]\) even when every peer law is fixed. The identity
\(\mathcal F_i(x)=\mathrm{KL}(Q_i(x)\Vert\Pi_i[x](\cdot\mid o))-\log Z_i[x]\)
holds pointwise, but its derivative includes motion of both the posterior target and its evidence. It is not descent toward the posterior of one fixed generative model. The constitutive channel map is fixed; its value is recognition dependent. This route follows the literal full derivative of the law-dependent per-agent functional. If the intended per-agent picture requires the entire generative model to remain fixed while its recognition blocks vary, (O4) is outside that stricter subclass. The fixed-channel version below satisfies that stronger condition. Pointwise normalization alone does not settle this distinction.

Define

\[
\ell_{i,z}^q=\log\frac{d\kappa_{i,z}}{dA^q},\qquad
H_i^q(z)=\int\ell_{i,z}^q d\kappa_{i,z},
\tag{O7}
\]

and the analogous quantities \(\ell_{i,w}^s,H_i^s(w)\). The displayed integral in (O7) denotes ordinary integration; no additional measure over law states occurs. The self and peer costs are

\[
\begin{aligned}
C_{ii}^q&=\sum_w\sigma_i(w)\mathrm{KL}(\mu_i\Vert G^c(w,\cdot))
                  +\sum_z\mu_i(z)H_i^q(z),\\
C_{ij}^q&=\mathrm{KL}(\mu_i\Vert\nu_{ij}^{q,c}),\quad j\ne i,\\
C_{ii}^s&=\mathrm{KL}(\sigma_i\Vert r^c)+\sum_w\sigma_i(w)H_i^s(w),\\
C_{il}^s&=\mathrm{KL}(\sigma_i\Vert\nu_{il}^{s,c}),\quad l\ne i.
\end{aligned}
\tag{O8}
\]

Every source is a probability law. In particular the original per-agent generative model

\[
P_i(dm,dk,J=j,L=l,do)
=\pi_{il}^s\rho_{il}(dm)\pi_{ij}^q\kappa_{ij}(m,dk)
\ell_i(do\mid k,m)
\tag{O9}
\]

is normalized: integrate its record likelihood, its content source, its presentation source, and its two positive source priors in that order. With finite bounded record potentials, all free-energy terms are finite. For a whole finite adaptive record experiment, use the causal product of the normalized chosen likelihoods; action probabilities are part of its declared experimental policy.

The exact minimizing rows are

\[
\beta_{ij}=\frac{\pi_{ij}^q e^{-C_{ij}^q}}{\sum_k\pi_{ik}^q e^{-C_{ik}^q}},
\qquad
\gamma_{il}=\frac{\pi_{il}^s e^{-C_{il}^s}}{\sum_k\pi_{ik}^s e^{-C_{ik}^s}}.
\tag{O10}
\]

All row entries are positive. The record term does not appear separately in these costs because the likelihood in (O9) is the same for every latent source label. It affects the rows through the evolving recognition laws.

Let \(\mathcal F_i^c\) be the public per-agent free energy with the same fixed rows, public sources, and public self evaluator, before adding record potentials. The full free energy, at any fixed row values, is exactly

\[
\begin{aligned}
\mathcal F_i={}&\mathcal F_i^c
+\beta_{ii}\sum_z\mu_i(z)H_i^q(z)
+\gamma_{ii}\sum_w\sigma_i(w)H_i^s(w)\\
&+\sum_z\mu_i(z)\overline W_i^q(z)
+\sum_w\sigma_i(w)\overline W_i^s(w),
\end{aligned}
\tag{O11}
\]

where \(\overline W_i^q(z)=\int W_i^q(z,T^q(v))\kappa_{i,z}(dv)\), and similarly for s. After row reduction, the envelope derivative is obtained by substituting (O10) in the fixed-row derivative. No derivative of an optimized row is added.

Let \(h_i^{q,c}(z)\) and \(h_i^{s,c}(w)\) denote the own-block public derivatives of \(\mathcal F_i^c\) at fixed rows. They do not include derivatives of other agents' objectives. For fixed public channels, up to constants that disappear under centering, they are explicitly

\[
\begin{aligned}
h_i^{q,c}(z)
&=\log\mu_i(z)-\beta_{ii}\sum_w\sigma_i(w)\log G^c(w,z)
 -\sum_{j\ne i}\beta_{ij}\log\nu_{ij}^{q,c}(z),\\
h_i^{s,c}(w)
&=\log\sigma_i(w)-\gamma_{ii}\log r^c(w)
 -\sum_{l\ne i}\gamma_{il}\log\nu_{il}^{s,c}(w)
 -\beta_{ii}\sum_z\mu_i(z)\log G^c(w,z).
\end{aligned}
\tag{O12}
\]

The full q force is

\[
h_i^q(z,v)=h_i^{q,c}(z)+\beta_{ii}\ell_{i,z}^q(v)
+W_i^q(z,T^q(v)),
\tag{O13}
\]

and the s force exchanges sectors and uses \(\gamma_{ii}\). In (O13), the derivative of the local conditional KL is \(\ell\) up to a constant. Every nonself private derivative has already canceled in (O6). A derivative of another receiver's objective is absent because each agent updates only its own recognition blocks.

Conditional averaging of the unit-mobility Fisher equation, followed by subtracting the marginal log derivative, gives

\[
\begin{aligned}
\dot\mu_i(z)&=-\mu_i(z)\left[
h_i^{q,c}(z)+\beta_{ii}H_i^q(z)+\overline W_i^q(z)
-\mathbb E_{\mu_i}(h_i^{q,c}+\beta_{ii}H_i^q+\overline W_i^q)
\right],\\
\partial_t\ell_{i,z}^q(v)&=
\beta_{ii}\bigl(H_i^q(z)-\ell_{i,z}^q(v)\bigr)
+\overline W_i^q(z)-W_i^q(z,T^q(v)).
\end{aligned}
\tag{O14}
\]

The model equations use \(\sigma_i,\gamma_{ii},H_i^s,W_i^s\). These are the original per-agent physical-time equations. Unlike the reciprocal Chapter 7 model, the self weight responds to the private entropy through (O8), and it controls private relaxation in (O14). The private profile is not a precomputed open-loop forcing. No global Lyapunov function is asserted for this directed coupled population.

## 3. Exact marked-spectrum closure with persistent records

Retain the joint reference-law distribution

\[
\eta_{i,z}^q=(\ell_{i,z}^q,T^q)_\#A^q,
\qquad
\eta_{i,w}^s=(\ell_{i,w}^s,T^s)_\#A^s.
\tag{O15}
\]

It obeys

\[
\int e^r\eta_{i,z}^q(dr,dd)=1,\quad
H_i^q(z)=\int re^r\eta_{i,z}^q(dr,dd),\quad
\overline W_i^q(z)=\int e^rW_i^q(z,d)\eta_{i,z}^q(dr,dd).
\tag{O16}
\]

All costs in (O8), rows in (O10), and public forces in (O14) are functions of the retained public laws, spectra, and records. For a differentiable bounded test f on the bounded log-density range, differentiate the pushforward against the fixed reference to obtain

\[
\frac{d}{dt}\int f(r,d)\eta_{i,z}^q(dr,dd)
=\int \partial_rf(r,d)
\left[\beta_{ii}(H_i^q(z)-r)+\overline W_i^q(z)-W_i^q(z,d)\right]
\eta_{i,z}^q(dr,dd).
\tag{O17}
\]

For an interval with no new records, put

\[
B_i(t)=\int_0^t\beta_{ii}(v)dv,\quad
a_i(t)=e^{-B_i(t)},\quad
c_i(t)=\int_0^t e^{-(B_i(t)-B_i(v))}dv.
\tag{O18}
\]

The conditional density and the retained spectrum are exactly

\[
\begin{aligned}
\frac{d\kappa_{i,z}(t)}{dA^q}(v)
&=\frac{\exp[a_i(t)\ell_{i,z}^q(0,v)-c_i(t)W_i^q(z,T^q(v))]}{Z_{i,z}(t)},\\
Z_{i,z}(t)
&=\int\exp[a_i(t)r-c_i(t)W_i^q(z,d)]\eta_{i,z}^q(0,dr,dd),\\
\eta_{i,z}^q(t)
&=\bigl((r,d)\mapsto(a_i(t)r-c_i(t)W_i^q(z,d)-\log Z_{i,z}(t),d)\bigr)_\#\eta_{i,z}^q(0).
\end{aligned}
\tag{O19}
\]

To verify the formula, \(\dot a=-\beta_{ii}a\), \(\dot c=1-\beta_{ii}c\). The derivative of the unnormalized log density is \(-\beta_{ii}(ar-cW)-W\). Differentiating the log normalizer subtracts its conditional mean, yielding precisely (O14). The coefficient \(\beta_{ii}\) is still solved jointly with the retained public equations; (O19) is not an assertion that it can be specified independently.

This is a closed probability-level retention. On finite public alphabets, bounded private log-density charts, bounded record potentials, and locally Lipschitz public data, its finite-time solution is obtained by coupling the finite public equations with (O18)--(O19); the exponentials and their positive normalizers are locally Lipschitz on bounded sets. The actual fine lift in (O19), substituted into the public equation, proves same-time intertwining. The finite-dimensional realization below supplies an explicit regular witness and avoids relying on a weak spectrum equation for a smooth Fisher-geometry claim.

The normalized spectral sample law

\[
Q_i^c(dz,dr,dd)=\mu_i(z)e^r\eta_{i,z}^q(dr,dd)
\tag{O20}
\]

reproduces all marked readouts. Its support in r generally moves, so (O20) is not automatically a regular dominated statistical family with finite Fisher metric. No geometry claim is made from this raw realization. The field (O17) is a transport equation, not a Fisher replicator equation on a fixed r support.

## 4. A regular normalized parent with continuous physical fibers

For an explicit invariant family, take

\[
V=U=S^1\times S^1,\qquad
A^q=A^s=\mathrm{Haar}\otimes\mathrm{Haar},\qquad
T(\phi,\psi)=\phi.
\tag{O21}
\]

The gauge group used by a later geometry construction may rotate the marked coordinate \(\phi\). The hidden \(\psi\) coordinate below is a fixed physical coordinate, not a gauge frame. Let \(e(\phi)=(\cos\phi,\sin\phi)\) and

\[
\kappa_{i,z}(\phi,\psi)=
\frac{\exp[A_{i,z}\cdot e(\phi)+b_{i,z}\cos(\psi-\theta_{i,z})]}
{I_0(|A_{i,z}|)I_0(b_{i,z})},
\quad A_{i,z}\in\mathbb R^2,\ b_{i,z}>0,\ \theta_{i,z}\in S^1.
\tag{O22}
\]

Here \(I_0(v)=\int_{S^1}e^{v\cos\phi}d\phi/(2\pi)\). Every density is positive and normalized. The amplitudes, public masses, and physical phases may differ independently between agents and between public values. This is an explicitly declared invariant family, open in its parameter chart; it is not an open class of all smooth densities. Use the analogous parameters for s when the model sector is nontrivial.

Two normalized persistent record likelihoods suffice. For a public bit \(z\in\{-1,1\}\), use

\[
L_\epsilon(o\mid z)=\frac{e^{\epsilon oz/2}}{2\cosh(\epsilon/2)},\qquad o\in\{-1,1\},\quad\epsilon>0.
\tag{O23}
\]

For the marked angle, with output Haar measure, use

\[
L_\lambda(y\mid z,\phi)=\frac{e^{\lambda_z\cos(y-\phi)}}{I_0(\lambda_z)},
\qquad \lambda_z\geq0.
\tag{O24}
\]

Normalization follows by summing the two values in (O23) and shifting the circle integral in (O24). Both are strictly positive; (O23) is nonconstant and persists even if \(\lambda_z=0\). The accumulated record potential has the exact form

\[
W_i^q(z,\phi)=v_i(z)-c_i(z)\cdot e(\phi),
\qquad
c_i(z)=\sum_{\text{marked records }y}\lambda_z e(y),
\tag{O25}
\]

where \(v_i(z)\) includes the public record term and the likelihood normalizers. All records, actions, and their dependence with later queries remain stored. The finite sufficient values \(v,c\) determine the future objective; they do not license erasing the historical outputs from the retained accessible record.

Set

\[
m(A)=\nabla_A\log I_0(|A|),\qquad R(b)=\frac{I_0'(b)}{I_0(b)}.
\]

These functions extend smoothly at \(A=0\). Direct integration in (O22) gives

\[
\begin{aligned}
H_i^q(z)&=A_{i,z}\cdot m(A_{i,z})+b_{i,z}R(b_{i,z})
-\log I_0(|A_{i,z}|)-\log I_0(b_{i,z}),\\
\overline W_i^q(z)&=v_i(z)-c_i(z)\cdot m(A_{i,z}).
\end{aligned}
\tag{O26}
\]

Both are independent of \(\theta_{i,z}\). Substituting (O22) and (O25) into (O14), and comparing the independent trigonometric coefficients, proves the invariant-family equations

\[
\dot A_{i,z}=-\beta_{ii}A_{i,z}+c_i(z),\qquad
\dot b_{i,z}=-\beta_{ii}b_{i,z},\qquad
\dot\theta_{i,z}=0.
\tag{O27}
\]

The model-sector equations use \(\gamma_{ii}\). The remaining equations are exactly (O8), (O10), (O12), and the first line of (O14), with (O26). This proves closure; neither equal private profiles nor equality between different agents is assumed.

Define canonical normalized laws by setting only the hidden phase to zero:

\[
\widehat\kappa_{i,z}(\phi,\psi)=
\frac{\exp[A_{i,z}\cdot e(\phi)+b_{i,z}\cos\psi]}
{I_0(|A_{i,z}|)I_0(b_{i,z})},\qquad
\widehat q_i(z,d\phi,d\psi)=\mu_i(z)\widehat\kappa_{i,z}(\phi,\psi)dA^q.
\tag{O28}
\]

Use the same construction for \(\widehat s_i\). The original references, evaluator, and reconstruction channels with canonical conditionals are normalized and give the same costs and rows. Thus the canonical factors have the exact projected physical-time equations. Because \(\theta\) is constant in time, the canonical representative has no omitted moving-frame term. A model where records read \(\psi\), or the hidden phase is driven by another term, would need a new calculation.

For any finite block I, the actual parent laws are

\[
q_I=\bigotimes_{i\in I}\widehat q_i,\qquad
s_I=\bigotimes_{i\in I}\widehat s_i.
\tag{O29}
\]

The parent retains the primitive port labels, current public and shape parameters, full operational history, event times, and reference data. It carries the inherited directed own-block field (O27) and (O14) on those ports. It is not asserted to descend the scalar sum \(\sum_{i\in I}\mathcal F_i\): differentiating that sum with respect to all internal child laws would add internal source-role forces. Nor is it asserted to be a fixed-size single-source-row agent. No new force is fitted.

This retention is genuinely noninjective as a map of physical laws. Changing any \(\theta_{i,z}\) while keeping the retained data fixed leaves (O28)--(O29) unchanged, but the child diagnostic

\[
\mathbb E_{q_i}\bigl[1_{Z=z}e^{i\Psi}\bigr]
=\mu_i(z)R(b_{i,z})e^{i\theta_{i,z}}
\tag{O30}
\]

changes. For \(b>0\), \(R(b)>0\): the expectation of \(\cos\psi\) under \(e^{b\cos\psi}/I_0(b)\) vanishes at b=0 and its derivative is the strictly positive variance of \(\cos\psi\). Each hidden phase therefore gives a continuous nongauge fiber on a local phase chart.

The retained family is regular on the fixed circle reference. Over any finite record-free interval, \(0<\beta_{ii}<1\) gives
\(b_{i,z}(0)e^{-t}\leq b_{i,z}(t)\leq b_{i,z}(0)\), and
\(|A_{i,z}(t)|\leq |A_{i,z}(0)|+t|c_i(z)|\). Consequently the diagnostic distinction does not disappear at finite time. The public log-ratio equations are locally Lipschitz. For identity public channels, their absolute derivatives are bounded by a constant times one plus the largest public log-ratio, on every finite horizon with finitely many bounded records: the terms in (O12) are weighted logarithms of public probabilities, and (O26) is bounded by \(|A|+b+|v|+|c|\). The resulting integral inequality
\(M(t)\leq M(0)+C\int_0^t(1+M(s))ds\)
implies \(M(t)\leq(M(0)+1)e^{Ct}-1\). Thus no public probability reaches zero at a finite time. Together with (O27), this establishes unique finite-time continuation for every finite bounded record sequence in this invariant family. Other public channels can be handled on their declared regular neighborhoods; no general-space global existence theorem is assumed.

## 5. Operational kernels and response/next-state dependence

Declare the operational protocol before selecting blocks. A predictive transaction may query any primitive port, sample its public coordinate and marked angle, draw native source labels and transmitted public messages, or return a jointly normalized function of finitely many such draws. Its own simulated prediction is not assimilated as new evidence. An evidence transaction uses a declared normalized record kernel such as (O23)--(O24), and its outcome is appended to the persistent potential. The transaction type is an action label retained in the history. Choices may depend on the whole retained history. The same primitive ports remain queryable after a handoff. The protocol never includes a hidden absolute-angle \(\psi\) readout, before or after handoff. Samples are fresh between transactions unless a reused sample's retained public and marked coordinates are explicitly stored.

For recognition-based records the next density is, for example,

\[
p_i(o,y)=\sum_z\mu_i(z)L_\epsilon(o\mid z)
\int_{S^1}L_\lambda(y\mid z,\phi)
\frac{e^{A_{i,z}\cdot e(\phi)}}{I_0(|A_{i,z}|)}\frac{d\phi}{2\pi}.
\tag{O31}
\]

It is normalized and independent of every hidden phase. If the chosen operational record law is instead the normalized generative marginal of (O9), its self branches use the private reference mark law and its peer branches use the receiver conditional mark law. Both are retained: in (O15) they are the mark marginal of \(\eta\) and of \(e^r\eta\), respectively; in (O28) they are the Haar mark law and the displayed von Mises mark law. The public source laws and the fixed source priors are retained as well. Hence that generative marginal also factors through the parent. The route must choose which observation law it uses; their equality is not presumed. An external trace may retain a latent source mark without supplying it to the inferring agent. If the agent is instead told its latent J or L, its source posterior is conditioned on that label and the full-support optimized-row convention (O10) must be replaced. No equivalence to that changed update is claimed.

A normalized inherited port channel is explicit: marginalize a source parent to primitive port j, read its public coordinate, apply \(B_{ij}\), and draw the target marked conditional from \(\widehat\kappa_{i,z}\). For a whole-parent output, append normalized canonical factors on all untouched coordinates. This carries the existing primitive comparison, including its recipient dependence; it does not replace all child costs by one parent KL. The same construction applies to model ports and self evaluators.

Given a recorded output, the public and shape state is unchanged at the instant of recording, while the stored history is appended and \(v,c\) are updated by their exact likelihood contribution. The next vector field is then computed from this new retained state. The joint law of the response and the next retained state therefore factors, not merely their separate marginals. For an adaptive sequence, equal retained states give equal deterministic waiting-time evolution, equal next response kernels, and the same retained update for every output. Induction through the causal product proves equality of every admitted finite joint response-record history, including all dependence with future states.

Any membership selector that depends only on the retained public/shape state and operational history has the same event under hidden-phase changes. Its timestamp and selected mode are consequently equal on the phase fibers. Keeping these data and the complete earlier records does not restore the phases. This is a conditional statement about a selector; it does not construct the interaction-dependent selector itself.

The distinction between the diagnostic in (O30) and the inherited protocol is mandatory. If sampling \(\psi\) remains an admitted future boundary action, (O30) immediately disproves descent. If that action is sampled with positive probability and its label is recorded, equality of the joint action/output law forces equality of its conditional output law by division by the positive action probability. Thus rarity cannot rescue exact compression. The present protocol has no post-handoff deletion of a probe. It supplies a physical distinction detectable by a separately declared child diagnostic, but does not claim to preserve that diagnostic as a future parent response.

## 6. Same-time composition and the prescribed null control

For disjoint blocks I and J, joining their parent laws takes products and concatenates their primitive ports and records. Direct canonicalization of \(I\cup J\) agrees with canonicalizing I and J separately, because each primitive hidden phase is removed independently. The dynamics and the marked kernels are unchanged, so this construction is associative at the same physical time. It does not prove a second compressive join: if each child parent already discarded all its primitive hidden phases, the second join removes no additional dimension. A second noninjective handoff needs another construction.

Under the prescribed null flow, nonself costs are deleted while the row entropy and source priors remain. The corresponding null row is therefore

\[
\beta_{ii}^0=
\frac{\pi_{ii}^qe^{-C_{ii}^q}}
{\pi_{ii}^qe^{-C_{ii}^q}+\sum_{j\ne i}\pi_{ij}^q},
\tag{O32}
\]

with analogous \(\gamma_{ii}^0\). Nonself costs are zero in the reduced null objective; the source labels are not removed and the self probability is not set to one. The own public force now has only its self-source contribution and the persistent record term. The conditional equation remains (O14) with \(\beta_{ii}^0\), since no nonself private force was present after the exact cancellation. Both the marked-spectrum and canonical-phase retentions close this null flow too.

Consequently the existence of this quotient alone proves no interaction necessity. The full and null flows have different row-dependent private clocks, but a clock difference is not a difference of partition sequences. The root route must establish (F5) using its actual selector and then check that its robust finite window survives this extension. For the public-bit specialization with small private amplitudes, (O26) gives small entropy perturbations, but no unproved stability transfer is declared here.

### A fixed-channel version with a fixed model during each own-block update

Replace only the nonself kernels (O4) by

\[
\mathcal K_{ij}^{q,\mathrm{fix}}((z_j,v_j),d(z,v))
=B_{ij}^q(z_j,z)A^q(dv),
\tag{O33}
\]

and analogously for s. With peers fixed, every content and presentation source, evaluator, reference, and record likelihood is now independent of the updating agent's recognition law. Thus this is literally the variational free energy of one fixed normalized per-agent model during that own-block variation.

The relative-entropy chain rule gives, for every source including self,

\[
C_{ij}^q=C_{ij}^{q,c}+\overline H_i^q,\qquad
\overline H_i^q=\sum_z\mu_i(z)H_i^q(z),
\tag{O34}
\]

and the corresponding model identity. The receiver-private term is common to every candidate source. Therefore

\[
\beta_{ij}=\frac{\pi_{ij}^q e^{-C_{ij}^{q,c}}}{\sum_k\pi_{ik}^q e^{-C_{ik}^{q,c}}},\qquad
\mathcal F_i^{\mathrm{red}}=
\mathcal F_i^{c,\mathrm{red}}+\overline H_i^q+\overline H_i^s
+\mathbb E_{q_i}W_i^q+\mathbb E_{s_i}W_i^s.
\tag{O35}
\]

The exact own q force is \(h_i^{q,c}+\ell_i^q+W_i^q\). Hence the q equations are (O14) with the coefficient of every private entropy/log-density term changed from \(\beta_{ii}\) to one; the s coefficient likewise changes from \(\gamma_{ii}\) to one. The marked-spectrum formula follows from (O17)--(O19) with \(B_i(t)=t\), \(a_i(t)=e^{-t}\), \(c_i(t)=1-e^{-t}\). On the circle family the equations are

\[
\dot A_{i,z}=-A_{i,z}+c_i(z),\qquad
\dot b_{i,z}=-b_{i,z},\qquad
\dot\theta_{i,z}=0.
\tag{O36}
\]

No moving-channel derivative is needed in this fixed version. The normalized canonical laws, conditional-spectrum retention, physical distinction (O30), same-time closure, and complete marked-history proof remain valid. Generative records now use the fixed private reference for every source branch; recognition-predictive records still use the current marked recognition law. Their distinction remains explicit.

For public-coordinate-independent profiles \(A_{i,z}=A_i\), \(b_{i,z}=b_i\), and public-coordinate-independent marked likelihoods, the entropy and marked-record forces are constant in z and cancel under marginal centering. This invariant family embeds the original public per-agent equations exactly. For a binary public law of log odds l, uniform public self evaluator, identity public channels, and records (O23),

\[
\dot l_i=-l_i+\sum_{j\ne i}\beta_{ij}l_j+\epsilon\sum_{\text{public records}}o,
\tag{O37}
\]

with the rows computed from the original public costs alone. It can therefore carry a separately proved public (F5) selection without a private-amplitude perturbation of the full public flow. This is a direct product-type hybrid, not a proof that selection created the compression interface.

The prescribed null control must still delete the whole nonself costs (O34), rather than cancel their common term after deletion. Its self row is (O32), including the private self entropy. The null private coefficient is consequently \(\beta_{ii}^0\), and its public log-odds field in this product-profile binary specialization is

\[
\dot l_i^0=-\beta_{ii}^0l_i^0+\epsilon\sum o.
\tag{O38}
\]

In particular a fixed positive record sum gives a strictly positive vector field at \(l_i^0=0\), while a negative sum gives a strictly negative field there. The corresponding record-matched sign half-line is forward invariant by uniqueness and the first-crossing test. This observation remains true for all private amplitudes because \(\beta_{ii}^0>0\); it is not, by itself, a grouping theorem.

## 7. Geometry boundary and remaining obligations

The actual laws in (O28)--(O29), rather than raw spectral laws with moving support, are available for the geometry route. If the hidden physical phase offsets are constant over the contextual section, canonicalization is a context-independent rotation of each conditional hidden circle. Contextual density scores are carried by that fixed rotation. A common group acting only on the marked circles commutes with it. These facts provide the input to an equality-of-minimizing-cost calculation; the nonflat tensor and connection proof belongs to the geometry artifact. If \(\theta_{i,z}\) varies with context, its discarded score can carry Fisher information, and no equality is claimed without an additional retained section mechanism.

The supported result is an exact operational parent on a declared invariant family, with normalized laws, original-time closure, physical phase loss, persistent likelihoods, and complete admitted adaptive histories. Its loss is relative to an operational interface fixed at the start. Under the frozen contract's distinction between a child diagnostic and inherited boundary probes, it witnesses the conditional physical noninjectivity required of a proposed retention. If the intended (F4) probe is also required to remain in the future boundary class, the same construction fails (F3) by (O30). No ambiguity between these readings is used to claim formation.

The minimal unresolved obligation for this route is to derive a compressive operational interface or a genuinely collective loss from the interactions, rather than use the fixed hidden-phase invariance. Coupling this parent to an independently proved (F5) public selector would still be a hybrid conditional result: it would not show that the interaction created the interface, and the present associative joining does not furnish a second information-losing handoff. The next action that would change that conclusion is a concrete predeclared protocol and channel model whose noninjective sufficient fibers depend nontrivially on the interaction-selected state, followed by the same full derivative and joint-history tests.

The novelty relative to the source construction is the exact self-row-weighted conditional force, its two-way coupling through inferred source preference, and its persistent-record extension under the normalized per-agent model. The spectrum idea itself is reused, not claimed as a new general principle. All evidence here is analytic. No simulation, numerical or symbolic script, process experiment, GPU operation, Lean work, or manuscript edit was performed. This route file is the only output written for this assignment.
