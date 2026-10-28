# Exact interaction of two conditional-prediction agents

The construction below gives a normalized interacting record process and exact correlated inference for every finite communication history. Its constitutive choices are a common preparation law, private initial records, a known alternating protocol, and reliable delivery of each message to both agents' record registers. The speaker transmits a fresh sample from its current result prediction. The receiver conditions on that sampling mechanism, including the speaker's inaccessible private evidence. The derivation first retains the complete joint posterior, then exactly reduces its changing part to the joint posterior of the two private records, with a fixed conditional reconstruction of represented variables. No moment projection enters any equation.

The target is existence of one explicit interacting realization, not equivalence with every composition of cue/result blocks. The common prior, its cross-agent coupling, and the communication policy are supplied assumptions. They are not derived from sharing a bank. All variables are defined at one context; no base connection or continuous-time Fisher flow is introduced.

## Source continuity and the remaining interface

The live [conditional-memory appendix](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/appendices/conditional_memory.tex:1>) supplies the normalized local evaluator, exact correlated posterior, local free-energy identity, and a carried-presentation filter with fresh emissions. The [internal-record appendix](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/appendices/record_process.tex:1>) distinguishes a normalized registered process from Bayesian assimilation under its actual observation law. The [foundations](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:33>) distinguish a joint recognition law from the product restriction, and the [communication chapter](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/17_communication_links.tex:1>) distinguishes receiving a realized message from reconstruction of a supplied sender law. The prior [sequential-record analysis](../2026-10-28-cue-result-magent-relation/sequential-records-and-retention.md) identifies the missing likelihood for a message generated from a peer's current belief.

The construction addresses that likelihood. It is a common-prior communication experiment with separate information available to each agent. It does not identify the experiment with the manuscript's product-family global coordinate flow, normalized per-agent mixture-source flow, or reciprocal law potential.

## A normalized preparation with correlated local banks

For \(i\in\{0,1\}\), let

\[
\mathsf Z_i=\mathsf A_i\times\mathsf X_i\times\mathsf Y_i,
\qquad Z_i=(A_i,X_i,Y_i),
\]

be standard Borel spaces of model presentation, cue, and result. Let \(r_i\) be a proper reference on \(\mathsf A_i\), \(\mu_{i,a}\) a normalized cue kernel, and \(K_{i,a}(x,dy)\) a normalized result kernel. The local prior is

\[
\pi_i(da,dx,dy)=r_i(da)\mu_{i,a}(dx)K_{i,a}(x,dy).
\tag{1}
\]

Choose a normalized coupling \(P_0(dz_0,dz_1)\) with marginals \(\pi_0,\pi_1\). This choice supplies the relation between what the agents represent. An explicit way to preserve the two result kernels is to choose any coupling \(\Xi\) of the presentation/cue laws and set

\[
P_0(dz_0,dz_1)
=\Xi(da_0,dx_0,da_1,dx_1)
K_{0,a_0}(x_0,dy_0)K_{1,a_1}(x_1,dy_1).
\tag{2}
\]

Integrating \(y_0,y_1\) gives one conditional on the presentation/cue pair, so (2) is normalized and has the desired bank marginals. The product coupling is allowed, but a nontrivial coupling or common represented target is needed if information about one agent's cue is to inform the other agent's own cue.

Let \(D_i\in\mathsf D_i\) be a private initial record, with normalized kernel \(L_i(x_i,dd_i)\). The preparation law of

\[
E=(Z_0,Z_1,D_0,D_1)
\]

is

\[
\Lambda_0(de)
=P_0(dz_0,dz_1)L_0(x_0,dd_0)L_1(x_1,dd_1).
\tag{3}
\]

Thus the two private records are conditionally independent given the represented variables. Initially agent \(i\) knows its actual \(d_i\), the model (1)–(3), and the protocol. It does not know the other private record. Its local joint recognition law is

\[
\zeta_{i,0}
=P(Z_i\in\cdot\mid D_i=d_i)
=s_{i,0}(da_i)q_{i,0}(dx_i,dy_i\mid a_i).
\tag{4}
\]

No factorization of presentation and content is imposed. Conditioning (3) on \(D_i\), then marginalizing the other agent, recovers the original local cue/result posterior under (1).

The variables \(Z_i\) are represented possibilities in a declared common model. An augmented realization may sample concealed internal preparation values \(Z_i\) and then generate \(D_i\) by (3). Nothing requires these values to be external objects. For the record process alone one can instead integrate out \(Z\), sample the joint private-record marginal \(\Lambda_0^{D_0,D_1}\), and use the same conditional models and protocol below. That collapsed realization produces exactly the same communication records. Statements about posteriors of \(Z\) refer to the augmented model (3), rather than inferring an ontology from normalization.

## The actual communication kernel

Write \(\mathcal H_t=\sigma(H_1,\ldots,H_t)\) for the common transcript and

\[
\mathcal I_{i,t}=\sigma(D_i,H_1,\ldots,H_t)
\]

for the information available to agent \(i\). At time \(t\), the common posterior and the full private posterior are

\[
\Gamma_t=P(E\in\cdot\mid\mathcal H_t),
\qquad
\Pi_{i,t}=P(E\in\cdot\mid\mathcal I_{i,t}).
\tag{5}
\]

The local recognition law is the \(Z_i\) marginal of \(\Pi_{i,t}\). The common posterior is computed from the common model and transcript; it is not a disclosure of either private record.

For a probability law \(\gamma\) on \(E\), write \(B_i(\gamma,d_i;de)\) for a regular conditional law of \(E\) given \(D_i=d_i\). Fix the conditional-law versions as part of the protocol. Conditional-law statements are almost sure statements. For an autonomous law-valued implementation, use a jointly measurable disintegration rule on its admitted law family; finite tables and the dominated construction below give that rule explicitly. Equivalently, on general standard Borel spaces one may construct each finite history first and use its measurable conditional kernels in \((h_{1:t},d_i)\). Null histories and null conditioning records receive fixed arbitrary normalized extensions.

Let \(j_t\in\{0,1\}\) be a fixed alternating speaker schedule. Both agents know it. At the next step, speaker \(j=j_{t+1}\) draws a fresh variable

\[
\widetilde Y_{t+1}\sim
S_j(\Gamma_t,D_j;dy),
\qquad
S_j(\gamma,d_j;\cdot)
=(Y_j)_\#B_j(\gamma,d_j).
\tag{6}
\]

Conditional on \((D_j,\mathcal H_t)\), this draw uses fresh randomization independent of \(E\). A fixed normalized encoder \(C_j:\mathsf Y_j\rightsquigarrow\mathsf H\) then produces

\[
H_{t+1}\sim C_j(\widetilde Y_{t+1},dh).
\]

The effective message kernel is

\[
\kappa_j(\gamma,d_j;dh)
=\int C_j(y,dh)S_j(\gamma,d_j;dy).
\tag{7}
\]

The receiver sees the realized \(H_{t+1}\), rather than the complete law \(S_j\). The speaker also sees the delivered message, and both append it to their registers. Future transmission policies use only the current conditional laws and the schedule; they do not additionally read discarded transmission randomization.

The fresh sample \(\widetilde Y_{t+1}\) and the represented variable \(Y_j\) are different variables. Equation (6) makes them identically distributed conditional on the speaker's information, but conditionally independent. Consequently,

\[
H_{t+1}\perp (Z,D_{1-j})
\mid(D_j,\mathcal H_t).
\tag{8}
\]

Replacing (7) by a direct sensor \(C_j(Y_j,dh)\) defines another generative model. This distinction remains necessary even when the two models have the same one-step public predictive marginal.

Indeed the public prediction satisfies the exact calibration identity

\[
\begin{aligned}
\bar\kappa_j(\gamma;dh)
&=\int\kappa_j(\gamma,d_j;dh)\gamma(de)\\
&=\int C_j(y,dh)\gamma^{Y_j}(dy).
\end{aligned}
\tag{9}
\]

To prove it, substitute (7), average the conditional law of \(Y_j\) over the \(D_j\) marginal, and recover \(\gamma^{Y_j}\). Equation (9) identifies the distribution of the message alone. The posterior depends on the joint law of that message and \(E\), which is \(\gamma(de)\kappa_j(\gamma,d_j(e);dh)\), not generally \(\gamma(de)C_j(Y_j(e),dh)\).

## Exact finite histories and Bayes recursion

For the density formulas, suppose \(C_j(y,dh)=c_j(h\mid y)\tau(dh)\) for a fixed sigma-finite measure \(\tau\). Then

\[
k_j(h\mid d_j,\gamma)
=\int c_j(h\mid y)S_j(\gamma,d_j;dy)
\tag{10}
\]

is normalized in \(h\). Suppose the following predictive denominator is finite and positive at the observed message:

\[
\bar k_j(h\mid\gamma)
=\int k_j(h\mid d_j(e),\gamma)\gamma(de).
\]

Define

\[
\mathcal U_j(\gamma,h)(de)
=\frac{k_j(h\mid d_j(e),\gamma)}
{\bar k_j(h\mid\gamma)}\gamma(de).
\tag{11}
\]

It is a probability law. The denominator condition holds for predictive-almost every message; the null cases receive the fixed extension already specified.

**Proposition.** Starting from \(\Gamma_0=\Lambda_0\), the recursion

\[
\Gamma_{t+1}=\mathcal U_{j_{t+1}}(\Gamma_t,H_{t+1})
\tag{12}
\]

is exactly the posterior in (5) for the normalized finite-history law

\[
\mathbb P_n(de,dh_{1:n})
=\Lambda_0(de)
\prod_{t=1}^n
\kappa_{j_t}\bigl(\Gamma_{t-1}(h_{1:t-1}),d_{j_t}(e);dh_t\bigr).
\tag{13}
\]

Here the product is successive kernel integration. Every finite marginal is consistent with the previous one.

**Proof.** The policy at time \(t\) is a measurable function of the already generated history and private record. It is normalized by (7). Integrating the last message therefore gives one at each previous \((e,h_{1:n-1})\), proving normalization and consistency by induction. Conditional on \(\mathcal H_t\), the current law of \(E\) is \(\Gamma_t\); multiplication by the next actual conditional likelihood gives the joint conditional law
\(\Gamma_t(de)\kappa_j(\Gamma_t,d_j(e);dh)\). Its marginal in \(h\) is (9), and division by that marginal gives (11). This proves the posterior identity by induction from \(\Gamma_0=\Lambda_0\). The nondominated kernel version is the disintegration of the same joint conditional law; it does not require a fictitious density for deterministic messages.

The private posterior follows without another model:

\[
\Pi_{i,t}=B_i(\Gamma_t,d_i),
\qquad
\zeta_{i,t}=(Z_i)_\#\Pi_{i,t}.
\tag{14}
\]

This is conditional probability under (13), first with the common transcript and then with the agent's private record. Both presentation and content dependence within an agent, and dependence on the other agent's private evidence, are retained.

A fully explicit continuous-record realization uses densities
\(L_i(x,dd_i)=l_i(d_i\mid x)\lambda_i(dd_i)\).
Let \(\eta=P_0\otimes\lambda_0\otimes\lambda_1\). Initially
\(g_0=d\Gamma_0/d\eta=l_0l_1\), and every later posterior has density
\[
g_{t+1}(e)=
\frac{g_t(e)k_j(h_{t+1}\mid d_j(e),\Gamma_t)}
{\int g_t(e')k_j(h_{t+1}\mid d_j(e'),\Gamma_t)\eta(de')}.
\tag{15}
\]
For a private record \(d_i\), divide
\(g_t(z,d_i,d_{1-i})P_0(dz)\lambda_{1-i}(dd_{1-i})\)
by its integral. These operations give the conditional kernels in (6) and (14) wherever their integrals are finite and positive. They allow arbitrary underlying bank laws \(P_0\); Gaussianity is unnecessary.

## What the sender and receiver infer

The actual likelihood has an immediate consequence that prevents self-generated pseudo-evidence.

**Proposition.** If \(j\) sends the next message, then

\[
\Pi_{j,t+1}=\Pi_{j,t}
\tag{16}
\]

almost surely. The common posterior may change even though the speaker's private posterior does not.

**Proof.** Once \(D_j=d_j\) and the old common transcript are given, the factor \(k_j(h\mid d_j,\Gamma_t)\) in (11) is constant in every remaining represented or private variable. It cancels from the conditional numerator and denominator. This is also exactly the conditional independence (8). The proof applies to the entire \(E\) posterior, and hence to its local presentation/content marginal. Fresh randomization from a current posterior cannot provide that posterior's owner with additional evidence about the variable it sampled.

For the other agent \(i=1-j\), let
\[
R_{ij,t}(dd_j\mid z_i)
=P(D_j\in dd_j\mid Z_i=z_i,D_i=d_i,\mathcal H_t).
\]
Its correct effective likelihood is
\[
\ell_{i\leftarrow j,t}(h\mid z_i)
=\int k_j(h\mid d_j,\Gamma_t)R_{ij,t}(dd_j\mid z_i).
\tag{17}
\]
Consequently
\[
\zeta_{i,t+1}(dz_i)
=\frac{\ell_{i\leftarrow j,t}(h\mid z_i)}
{\int\ell_{i\leftarrow j,t}(h\mid z_i')\zeta_{i,t}(dz_i')}
\zeta_{i,t}(dz_i).
\tag{18}
\]
To derive (17), condition the actual kernel (7) on \((Z_i,D_i,\mathcal H_t)\) and integrate the remaining unknown \(D_j\). Marginalizing the joint private update then gives (18). The effective likelihood depends on the conditional relationship between the receiver's represented state and the sender's private evidence. Its result is not determined by the local marginal \(\zeta_{i,t}\) alone.

If the sender's policy reads only common information, so that \(k_j(h\mid d_j,\Gamma_t)\) is independent of \(d_j\), then it cancels in (11) and neither agent learns anything about \(E\). A deterministic copy of a previously registered public message is a special case, handled by the kernel version. A fresh posterior sample may reveal information about \(D_j\) to its peer, but it is not an independent new measurement of the represented cue.

There is also a precise limit on the evidence being redistributed. Conditional on the pair \(D=(D_0,D_1)\), the entire communication history is independent of \(Z\), because all factors in (13) depend on \(e\) only through \(D\). Hence, when the mutual informations below are finite,

\[
I(Z;H_{1:n}\mid D_i)
=I(Z;D_{1-i}\mid D_i)
-I(Z;D_{1-i}\mid D_i,H_{1:n}).
\tag{19}
\]

The identity follows by expanding \(I(Z;(D_{1-i},H_{1:n})\mid D_i)\) in its two possible orders and using conditional independence for the term given both private records. Communication can disclose existing private evidence; this fixed-preparation protocol does not create new measurements of \(Z\).

## The free energy of the declared history

At a fixed observed history \(h_{1:n}\), set

\[
\begin{aligned}
\mathcal L_n(e;h)
&=\prod_{t=1}^n
k_{j_t}\bigl(h_t\mid d_{j_t}(e),\Gamma_{t-1}(h_{1:t-1})\bigr),\\
Z_n(h)&=\int\mathcal L_n(e;h)\Lambda_0(de).
\end{aligned}
\tag{20}
\]

The quantities \(\Gamma_{t-1}(h_{1:t-1})\) in the likelihood are the protocol states determined from the actual prior and observed prefix. They are fixed when varying a candidate recognition law. The history posterior is
\(\Gamma_n=\mathcal L_n\Lambda_0/Z_n\).
For \(Q\ll\Lambda_0\), with the finite relative entropies and integrable logarithms needed to avoid an undefined difference of infinities, define

\[
\mathcal F_n(Q;h)
=\operatorname{KL}(Q\Vert\Lambda_0)
-\mathbb E_Q\log\mathcal L_n(E;h).
\tag{21}
\]

Substitution of the posterior density gives

\[
\mathcal F_n(Q;h)
=\operatorname{KL}(Q\Vert\Gamma_n)-\log Z_n(h).
\tag{22}
\]

The unique unrestricted minimum is the exact history posterior, up to null sets. This is a unit-temperature variational identity for one fixed observed-data model.

Agent \(i\) can instead use the private base law
\(\Lambda_{0,i}^{d_i}=P(E\in\cdot\mid D_i=d_i)\).
Replacing \(\Lambda_0\) in (20)–(22) by this base gives

\[
\mathcal F_{i,n}(Q;d_i,h)
=\operatorname{KL}(Q\Vert\Pi_{i,n})
-\log p(h_{1:n}\mid d_i).
\tag{23}
\]

Each agent conditions the same common model on different available information. The objectives (23) therefore have different posterior minimizers; the construction provides no monotone sum of their values as new random records arrive.

There is an exact local version as well. Disintegrate the finite unnormalized measure
\(\mathcal L_n\Lambda_{0,i}^{d_i}\)
with respect to \(Z_i\); let its marginal be \(\nu_{i,n}\), of total mass
\(Z_{i,n}=p(h_{1:n}\mid d_i)\). Then
\(\zeta_{i,n}=\nu_{i,n}/Z_{i,n}\).
For a candidate local correlated recognition
\(\mathcal Q_i(da_i,dk_i)=s_i(da_i)q_i(dk_i\mid a_i)\),
where \(k_i=(x_i,y_i)\), define its local free energy by the log density ratio against \(\nu_{i,n}\). If the same absolute-continuity and finiteness conditions hold,

\[
\begin{aligned}
\mathcal F_{i,n}^{\mathrm{loc}}(\mathcal Q_i)
={}&\operatorname{KL}(s_i\Vert s_{i,n})\\
&+\int\operatorname{KL}
\bigl(q_i(\cdot\mid a_i)\Vert q_{i,n}(\cdot\mid a_i)\bigr)s_i(da_i)
-\log Z_{i,n}.
\end{aligned}
\tag{24}
\]

This is the original correlated free-energy split with the actual composed record likelihood. Its proof is (22) on the marginal space followed by the chain rule for relative entropy. The marginalized likelihood already contains integration over the unknown peer evidence. Replacing that integration by a plug-in message mean or an independent-replicate likelihood changes the model before any moment projection is considered.

The full objective also exposes information omitted by local recognition. For any admissible \(Q\) on \(E\), disintegration over \(Z_i\) gives

\[
\operatorname{KL}(Q\Vert\Pi_{i,n})
=\operatorname{KL}(Q^{Z_i}\Vert\zeta_{i,n})
+\int\operatorname{KL}
\bigl(Q(\cdot\mid Z_i=z_i)\Vert
\Pi_{i,n}(\cdot\mid Z_i=z_i)\bigr)Q^{Z_i}(dz_i).
\tag{25}
\]

Minimizing only the first term does not identify the conditional law needed in (17).

Recomputing the speaker's historical policy using a variational candidate \(Q\) inside (20) would make the likelihood itself vary with the candidate. That is a different optimization problem. Equations (21)–(24) assert exact inference under the declared protocol; they do not derive a reciprocal law potential, natural-gradient dynamics, or bank-learning rule.

## An exact reduction to the joint private-record posterior

The entire \(E\) posterior need not be evolved as an unconstrained law. Disintegrate the fixed preparation law as

\[
\Lambda_0(dz,dd)=W(d,dz)\lambda_0(dd),
\qquad
\lambda_0=P(D\in\cdot),\qquad W(d,\cdot)=P(Z\in\cdot\mid D=d).
\tag{26}
\]

The conditional reconstruction \(W\) is fixed model data, with its null-set version fixed once. Every message likelihood in (20) depends on \(e\) only through \(d\). Therefore

\[
\Gamma_t(dz,dd)=W(d,dz)\lambda_t(dd),
\qquad
\lambda_t=P(D\in\cdot\mid H_{1:t}).
\tag{27}
\]

**Proof.** For any bounded measurable \(f(z)\) and \(a(d)\), substitute
\(\Lambda_0=W\lambda_0\) into the posterior integral
\(\int f(z)a(d)\mathcal L_t(d;h)\Lambda_0(dz,dd)/Z_t\).
Integrating \(z\) first yields
\(\int a(d)[\int f(z)W(d,dz)]\lambda_t(dd)\).
The equality for product functions determines the joint law by a monotone-class argument. Thus the conditional law of \(Z\) given the complete private record pair does not change after communication.

For each agent let \(W_i(d,dz_i)\) be the local marginal of \(W\), and define the fixed kernel

\[
V_j(d,dh)=\int C_j(y,dh)W^{Y_j}(d,dy).
\tag{28}
\]

Write \(\lambda(dd\mid D_i=d_i)\) for the conditional law of the pair \(D=(D_0,D_1)\) under the joint law \(\lambda\). The exact local recognition and message kernel become

\[
\begin{aligned}
\zeta_i^{\lambda,d_i}(dz_i)
&=\int W_i(d,dz_i)\lambda(dd\mid D_i=d_i),\\
\kappa_j^\lambda(d_j,dh)
&=\int V_j(d,dh)\lambda(dd\mid D_j=d_j).
\end{aligned}
\tag{29}
\]

The conditional pair law is supported on the observed coordinate \(D_i=d_i\) almost surely. Equations (29) are conditional mixtures, preserving all dependence between the local presentation, cue, and result through \(W_i\). Their evaluation requires the joint private-record law, not merely its two marginals.

When \(V_j(d,dh)=v_j(h\mid d)\tau(dh)\), write the second equation of (29) as the density \(k_j^\lambda(h\mid d_j)\). The closed recursion is

\[
\lambda_{t+1}(dd)
=\frac{k_j^{\lambda_t}(h_{t+1}\mid d_j)}
{\int k_j^{\lambda_t}(h_{t+1}\mid d_j')\lambda_t(dd')}
\lambda_t(dd).
\tag{30}
\]

It reconstructs exactly the full update (11), because multiplication by the message density leaves the conditional factor \(W(d,dz)\) unchanged. Conversely, marginalizing (11) over \(Z\) yields (30). This proves equivalence of the full and reduced descriptions for every admitted finite history.

The exact reduction also applies to free energy. Let a candidate \(Q\) on \(E\) disintegrate as \(Q(dz,dd)=Q^Z(d,dz)\nu(dd)\). Under the same finiteness hypotheses as (22),

\[
\mathcal F_n(Q;h)
=\operatorname{KL}(\nu\Vert\lambda_n)
+\int\operatorname{KL}\bigl(Q^Z(d,\cdot)\Vert W(d,\cdot)\bigr)\nu(dd)
-\log Z_n(h).
\tag{31}
\]

This follows from (22) and the relative-entropy chain rule applied to (27). For fixed \(\nu\), the unique minimizing conditional is \(W\) almost everywhere. The remaining exact variational problem is

\[
\mathcal F_n^{D}(\nu;h)
=\operatorname{KL}(\nu\Vert\lambda_0)
-\mathbb E_\nu\log\mathcal L_n(D;h)
=\operatorname{KL}(\nu\Vert\lambda_n)-\log Z_n(h).
\tag{32}
\]

Thus eliminating represented variables here is exact conditional reconstruction, not a Gaussian or moment approximation. On the admitted family \(\{W\lambda\}\), the map to \(\lambda\) is invertible by the fixed \(W\). It reduces the changing representation but does not discard information about \(Z\) that was present in that family. No compressive parent claim follows.

## Retained state and its scope

For current inference and future messages of this fixed protocol, the sufficient state available to agent \(i\) is the fixed preparation and reconstruction data, its private record \(d_i\), the common joint private-record posterior \(\lambda_t\), and the next speaker. To keep every actual record query available as well, take the complete retained state to be

\[
\mathcal R_{i,t}
=\bigl(P_0,L_0,L_1,W,C_0,C_1,d_i,\lambda_t,j_{t+1},H_{1:t}\bigr).
\tag{33}
\]

The transcript is an explicit persistent register. The result sample used solely for a transmission need not be reinterpreted as a measurement; its possible retention for another future policy would require specifying that policy.

To prove future sufficiency, condition on any realized information state in (33). Reconstruct \(\Gamma_t=W\lambda_t\) and the private law \(B_i(\Gamma_t,d_i)\). The next message is drawn from (29); the common private-record state changes by (30); and the transcript register appends the observed message. Induction gives the complete joint law of \(E\) and every finite future sequence of protocol messages. No inaccessible past variable beyond the represented private records is required. Equality of (33) therefore identifies all such conditional future laws. This is a sufficient retention, not a minimality theorem or a compressive parent construction.

In a finite example, \(\lambda_t\) is a table on the private-record pair. Binary private records require four masses, regardless of how many model/cue/result states the fixed reconstruction \(W\) contains. Its number of entries does not grow with exchanges. The record register still grows, and \(W\) itself can be large. On general spaces an exact probability law can require unbounded descriptive complexity. Neither case makes a local Gaussian carrier a sufficient replacement.

The supplied coupling in (3) is operationally relevant. If it factorizes across the two agents, then at every fixed public transcript the likelihood in (20) factorizes into one function of \(D_0\) and one of \(D_1\). Consequently \(\Gamma_t\) remains a product across \((Z_0,D_0)\) and \((Z_1,D_1)\). Conditional on its own \(D_i\), an agent's own \(Z_i\) posterior does not change when the peer speaks. It can learn about the peer, but own-target inference needs the supplied cross-agent relation. Identical bank parameters alone do not supply this relation.

The construction closes normalized communication and Bayesian assimilation for the declared two-agent protocol. It permits correlated presentations and contents throughout, and specifies the information that later inference consumes. Further compression of the joint private-record state, a practical exact implementation for general continuous banks, learned private policies, new measurements over time, and adaptation of the common model remain separate questions. Those questions can now be posed against an explicit joint law instead of an assumed composition of local likelihoods.

This note uses analytical derivations and source reads only. No numerical or symbolic scripts, experiments, training, GPU queries, Lean work, or edits outside this assigned research note were performed.
