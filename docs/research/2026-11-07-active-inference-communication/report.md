# Active inference through framed, selective communication

Research record, November 7, 2026. This is an independent model investigation stored in VFE_4.0 for continuity. It is not an extension adopted into MAgent Theory 2.0, an implementation, or a claim that collective formation has been established. Three constructive investigators contributed communication geometry, expected-free-energy policies, and causal population dynamics; a follow-up examined adjacent communication architectures. No simulations or training were run.

## Question and agreed scope

There are N agents with private beliefs, generative models, and cognitive frames. An agent chooses whom to address or attend to, a modality, and what to transmit through a lower-dimensional channel. A receiver interprets that channel in its own higher-dimensional internal space. Communication is an action whose anticipated consequences enter the acting agent's expected free energy (EFE).

Each agent has its own preferences. Two comparisons are requested: explicit models of a receiver's interpretation versus predictive models learned from observable responses; and possible blanket roles included in the agents' priors versus formation without any prescribed group or blanket role. Whether interfaces should be fixed or learned is undecided. The proposed sequence first fixes interfaces, then introduces parameter uncertainty and adaptation as distinct extensions. General probability laws are the primary setting; Gaussian channels provide an explicit example.

The formation hypothesis is that interaction can produce a persistent population partition into interior agents, boundary agents, and an exterior. This is a hypothesis about the operational population law, not a consequence of the word “agent,” low-dimensional messages, or a shared softmax formula.

## Existing constructions and the actual extension

The current local attention source is `GL(K)_attention_v2.tex`; no file with the exact unsuffixed name was found in the live Attention directory. Its sections `sec:framed_states`, `sec:role_measurements`, and `sec:attention_operator` define Gaussian beliefs, frames, query/key comparison spaces, and value/output channels. Its current compatibility score and entropy-regularized routing are not an EFE policy derivation. The content map can be singular while its frame-dressed endpoint law remains covariant.

`minimal_agent_model.tex`, section `ma:sec:model`, derives an exact Gaussian receiver from a comparison-record model. Its source posterior, gain, and precision-weighted residual depend on declared snapshot and independence assumptions. Its multi-head factorization requires orthogonality conditions on both receiver queries and sender utterances. Those assumptions should not be silently transferred to interacting agents with persistent shared histories.

`adaptive_vfe_interfaces/main.tex`, especially “What is to be derived,” “A causal population protocol and the meaning of reciprocity,” and “Failure cases and interpretation,” already supplies grounded inference of uncertain interfaces and a normalized chronological protocol. It explicitly separates association inference from attention as action and leaves future-directed policy optimization open. Its latent interface matrices are fixed over the inference window; their estimates adapt. This differs from physically changing the interface.

MAgent Theory 2.0, Chapter 17, supplies conditional reconstruction, learned dictionaries, and their identifiability limits. Chapter 4 and `appendices/interacting_memory.tex` supply a common-preparation, fixed-schedule communication construction and exact history-dependent receiver likelihoods. Chapters 10 and 11 state limitations of their supplied hierarchy and grouping mechanisms. These are useful ingredients and warnings, without importing that manuscript's population objective or formation standard as a specification.

The extension investigated here is therefore controlled, strategic communication between agents with distinct subjective models and preferences, followed by a separate test of endogenous boundary formation. The [literature map](literature.md) records close precedents, including papers that already choose both an expression and a source of information.

## What belongs to an agent

Let E_i be agent i's represented content space, initially a copy of R^K, and let q_i be a probability law on it. The generative model P_i is a collection of normalized transition and observation kernels. An operational internal state R_i includes the variables that implement q_i, P_i, the cognitive frame, and relevant memory. R_i is not a fresh sample from q_i: a belief about an unknown quantity and the physical or computational state carrying that belief are different objects.

A frame U_i:V_i -> E_i gives reference coordinates, with

\[
\widetilde q_i=(U_i^{-1})_\#q_i.
\]

Writing U_i=exp(phi_i), with phi_i in gl(K), is a convenient restricted parameterization. A single real exponential does not cover all of GL^+(K). Independent copies V_i of R^K also do not establish that their coordinates have identical semantics. Treating U_j U_i^{-1} as the complete translation between agents would supply that common identification.

For modality m, let H_m be a public message space, such as R^{r_m} with r_m<K. A normalized encoding kernel K_i^m:E_i -> Prob(H_m) and a selected content law rho_i^a determine the message law

\[
\nu_i^a=\rho_i^a K_i^m.
\]

An action a can choose recipient, modality, content-selection instruction, listening allocation, and transmission budget. Silence and refusal are admissible actions. Taking rho_i^a=q_i produces a coded sample of the current belief. Sending its projected mean, sending a fresh sample, and sending a description of its entire marginal law are different protocols. An arbitrary law generally has no finite exact description. A lower-dimensional linear encoding imposes a rank restriction. A real-valued packet is not by itself a finite bit-capacity constraint; noise, precision, quantization, power, or another transmission budget must be specified.

## Reception and cognitive geometry

Receiver j supplies its own likelihood L_{ji}^m(dy|x_j,h_j,a). If it has density ell and finite positive evidence, reception is

\[
q_j^+(dx_j)=
\frac{\ell_{ji}^m(y|x_j,h_j,a)q_j^-(dx_j)}
{\int\ell_{ji}^m(y|x_j',h_j,a)q_j^-(dx_j')}.
\]

This is exact Bayesian conditioning under j's model. Its correspondence with the actual sender's encoding and strategic behavior is a calibration question. The message can change beliefs about coordinates it does not directly encode because j's model relates those coordinates to the observed ones.

For a linear channel, use distinct maps B_i^m:E_i -> H_m for encoding and A_{ji}^m:E_j -> H_m for the receiver's predicted observation. An optional full dictionary D_{ji}:E_i -> E_j gives

\[
B_i^m=T_i^mU_i^{-1},
D_{ji}=U_j\overline D_{ji}U_i^{-1},
A_{ji}^m=B_i^mD_{ji}^{-1}.
\]

The dictionary is additional structure. The two directed dictionaries need not be inverses. A single dictionary across modalities is an additional compatibility condition, not a consequence of assigning a frame to every agent. For stacked maps B_* and A_*, an invertible D satisfying B_*D^{-1}=A_* exists exactly when their linear row relations agree, equivalently ker(B_*^T)=ker(A_*^T). Equality of rank alone is insufficient.

Under independent passive coordinate changes g_i,

\[
q_i'=g_{i\#}q_i, U_i'=g_iU_i,
B_i^{m\prime}=B_i^mg_i^{-1},
A_{ji}^{m\prime}=A_{ji}^mg_j^{-1},
D_{ji}'=g_jD_{ji}g_i^{-1}.
\]

The public message law is unchanged, and q_j^{+\prime}=g_{j\#}q_j^+. A model's kernels must transform too: a transition P_i(dx'|x,a) pushes forward its output and changes its input argument. A generative model is not simply a vector rotated alongside a mean.

Changing every coordinate-dependent object consistently is a re-description. Changing a frame or encoder while holding grounded observation relationships fixed can change the public predictive law and is model adaptation. Pure coordinate freedom cannot itself supply a communication benefit or an identifiable learning target.

For the explicit Gaussian case, suppose

\[
q_j^-=\mathcal N(\mu_j,\Sigma_j),
Y=A_{ji}X_j+\varepsilon,
\varepsilon\sim\mathcal N(0,R),
\Sigma_j\succ0,\ R\succ0.
\]

Then

\[
O_{ji}=\Sigma_jA_{ji}^T(A_{ji}\Sigma_jA_{ji}^T+R)^{-1},
\]
\[
\mu_j^+=\mu_j+O_{ji}(y-A_{ji}\mu_j),
\Sigma_j^+=\Sigma_j-O_{ji}A_{ji}\Sigma_j.
\]

The posterior precision is Sigma_j^{-1}+A_{ji}^T R^{-1}A_{ji}. Its increment has rank at most r_m, even though all mean coordinates can change. For A=(1,0), mu=0, covariance [[1,rho],[rho,1]], and scalar noise variance v>0, observing y=1 gives mu^+=(1,rho)^T/(1+v). Receivers with opposite correlations infer opposite changes in the untransmitted coordinate. Their models supply that difference.

There is also an exact statement about receiving an entire noiseless message law. For aligned sender and receiver laws q,p on a standard Borel space, finite KL(q||p), and measurable T, define

\[
\widehat q(dx)=\int p(dx|T=y)(T_\#q)(dy).
\]

Disintegration and the KL chain rule give

\[
D_{\mathrm{KL}}(q\Vert\widehat q)
=D_{\mathrm{KL}}(q\Vert p)
-D_{\mathrm{KL}}(T_\#q\Vert T_\#p).
\]

The reconstruction preserves the transmitted marginal and supplies the receiver's conditional law on each fiber. Exact recovery requires agreement of the sender's and receiver's conditionals almost everywhere under the transmitted marginal. This restates the conditional-reconstruction mechanism already in the live manuscripts; it is not a new result of this investigation.

## A causal population process

Let W_t be an operational environment and H_{i,t} the history available to agent i. One round has the order

\[
A_{i,t}\sim\Pi_{i,t}(\cdot|H_{i,t}),
M_{i,t}\sim E_{i,t}(\cdot|H_{i,t},A_{i,t}),
\]
\[
(Y_t,W_{t+1})\sim K_t(\cdot|W_t,A_t,M_t).
\]

Each agent appends its own action, emission, and accessible delivery record to its history. Private persistent state can be included in the history or in the state of its kernels. Replies are subsequent actions, so the definition contains no simultaneous circular dependence between a message and its reply. A delivery kernel can include acceptance, congestion, and unobserved packets. Conditional independence of agents' private random seeds, if used, is an assumption.

Proper initial laws and measurable normalized kernels define a normalized finite chronological joint: integrate the final event first and continue backward. This operational law does not require the agents to share a prior or to have subjective models that are conditionals of one common generative model.

For planning, agent i separately constructs a normalized predictive law

\[
Q_i^{\pi_i}(dZ,dY|H_i,\operatorname{do}\pi_i),
\]

where Y is its future accessible record and Z is a specified epistemic target. Z may include environmental variables and uncertain parameters governing other agents' responses. A model of a peer's beliefs must integrate over unknown private evidence rather than grant the planner direct access to that peer's actual posterior.

If an outgoing message is computed from known history and independent randomization, it supplies no new information to its sender about the actual hidden world:

\[
I(W;M_i|H_i,A_i)=0.
\]

The receiver can learn because it lacks the sender's private evidence. The sender can learn from a reply, a delivery failure, or an environmental consequence. Counting its own posterior sample as a newly measured world state would change the model incorrectly.

Repeated messages also need their chronological likelihood. If one private bit has reliability 0.8 and is reported twice without new evidence, the second report leaves the receiver's posterior at 0.8. Treating it as a second independent measurement gives 16/17 instead. Persistent history and evidence dependence therefore matter even before model learning is added.

## Expected free energy and policy choice

For an exact bounded construction, take finite Z,Y and a finite candidate policy set. Fix agent i's accessible history, require a normalized preference distribution C_i(y)>0 on predictive support, and assume finite terms. Write Q_i^pi(z,y)=Q_i^pi(z)L_i^pi(y|z). Define the normalized preference-biased joint

\[
\widetilde P_i^\pi(z,y)=C_i(y)Q_i^\pi(z|y).
\]

Any normalized conditional can be assigned on outcomes of zero predictive probability. The declared EFE convention is

\[
G_i(\pi)=\mathbb E_{Q_i^\pi}
[\log Q_i^\pi(Z)-\log\widetilde P_i^\pi(Z,Y)].
\]

Substitution gives both identities

\[
G_i(\pi)=-\mathbb E\log C_i(Y)-I_{Q_i^\pi}(Z;Y)
=D_{\mathrm{KL}}(Q_i^\pi(Y)\Vert C_i)+H_{Q_i^\pi}(Y|Z).
\]

Thus the agent balances its own preferred consequences with information about its specified hidden target. This is an exact identity for the stated construction, not a derivation of EFE from current-record variational free energy alone. [Millidge, Tschantz, and Buckley](https://doi.org/10.1162/neco_a_01354) analyze why that distinction matters.

Replacing the future posterior by an approximation R_i^pi(z|y) adds the following term to G_i:

\[
\mathbb E_{Q_i^\pi(Y)}
D_{\mathrm{KL}}(Q_i^\pi(Z|Y)\Vert R_i^\pi(Z|Y)).
\]

The correction cannot be dropped in an exact equivalence claim. For a persistent target, trajectory information should be counted as I(Z;Y_{1:T})=sum_t I(Z;Y_t|Y_{<t}); summing independent one-step information gains can count the same information repeatedly.

A separate variational choice of policy distribution minimizes

\[
\mathcal J_i(r_i)=\sum_\pi r_i(\pi)[G_i(\pi)+k_i(\pi)]
+\tau_iD_{\mathrm{KL}}(r_i\Vert\rho_i),
\]

where rho_i is a positive normalized habit distribution, tau_i>0, and k_i is a declared transmission or action cost not already counted in C_i. The unique minimizer is

\[
r_i^*(\pi)=
\frac{\rho_i(\pi)e^{-[G_i(\pi)+k_i(\pi)]/\tau_i}}
{\sum_{\pi'}\rho_i(\pi')e^{-[G_i(\pi')+k_i(\pi')]/\tau_i}}.
\]

This follows from J_i(r_i)=tau_i KL(r_i||r_i^*)-tau_i log Z_i. Its resemblance to the GL attention softmax is algebraic. The latter scores present compatibility; this policy distribution scores predicted consequences.

For example, let Z be a fair unknown bit. Equally costly listening actions A and B respectively return an independent fair bit and the value of Z. With uniform preferences over the received bit, G(A)=log 2 and G(B)=0. A current compatibility score can still prefer A. Informative disagreement can therefore be preferable to familiar agreement.

Continuous variants need declared reference measures and finite integrals. Comparing modalities of different dimensions by unrelated “uniform preferences” is not neutral: preference normalizers and density units can change the score. A common task-outcome representation with a declared treatment of reply variables, or a normalized disjoint-union model including modality, avoids silently dropping those terms.

## Modeling the receiver and strategic interpretation

An explicit receiver model represents an uncertain receiver state and parameter theta_j, a belief update, and a response policy. A behavioral model predicts the distribution of observable replies directly from history and the intervention. Both are models and both can support EFE planning. A behavioral predictor is not necessarily model-free.

They should be compared with the same action set, accessible records, physical budgets, preference law, and epistemic target. If they induce the same Q_i^pi(Z,Y), they give the same EFE. Agreement of outcome marginals alone is insufficient because the information terms can differ. Extra hidden variables in the explicit model should not silently supply an extra curiosity reward absent from the comparison model.

As a finite illustration, suppose a particular receiver translates message m in {-1,+1} into action -m. The sender prefers action +1 with C_i(A)=exp(eta A)/(2 cosh eta), eta>0. If decoding is known and no hidden uncertainty remains, G_i(+1)-G_i(-1)=2 eta. The appropriate message is -1. A learned behavioral model that predicts the same translation makes the same choice. This is a coordinate-translation example, not an assertion that deception is favored.

If the receiver knows that the sender selects messages strategically, that selection belongs in its likelihood. Schematically,

\[
\ell_j(a,m|z_j,H_j)=\int
\Pi_i(a|h_i)E_i(m|h_i,a)R_j(dh_i|z_j,H_j).
\]

Recipient choice, modality, timing, and silence can all be informative. For its own planning the sender intervenes on its action; for the receiver that same action is an observation of another agent's choice. These are different conditionings.

In a one-message policy model with attributed Gibbs probabilities, the receiver's negative log likelihood includes G_{i|j}(m,z)/tau-log rho_{i|j}(m|z)+log Z_{i|j}(z). The last term generally depends on the inferred state. [Gwon, Murphy, and Chung's October 2026 preprint](https://arxiv.org/abs/2610.04347), sections 3.6–3.7, is a close precedent for this construction. Helping a listener understand is valuable to the sender only insofar as it affects the sender's preferred consequences, information, or explicitly social preferences.

## What changes when interfaces are learned

First distinguish inference about an unknown fixed operating map from an action that changes the map. The first is parameter learning; the second changes future likelihoods and needs a transition or adaptation rule. A third operation, passive coordinate transformation, changes neither observable predictions nor EFE.

A known rank-r code can identify a full dictionary only modulo transformations invisible to that code. With unknown code and dictionary, even their factorization can be unidentifiable: (Tg)(Dg)^{-1}=TD^{-1}. Complementary modalities and sufficiently informative common referents can reduce these ambiguities, but no universal identification follows from exchanging messages.

Two elementary failures constrain any learning proposal. For independent standard-normal X,Z and C=aX-bZ+epsilon with noise variance sigma^2>0, the likelihood of C=0 is proportional to (a^2+b^2+sigma^2)^{-1/2}; agreement-only fitting favors a=b=0. Conversely, with fixed positive channel noise, scaling a nonzero Gaussian encoder without a power cost makes its information gain grow without bound. Grounding, finite resources, and identifiable feedback must therefore be supplied before optimizing interfaces.

## Collective boundaries: a precise hypothesis

For a partition of agent indices into nonempty I,B,E, let R_I,R_B,R_E denote declared operational states, including necessary private memory and model parameters. An instantaneous statistical blanket requires

\[
I(R_{I,t};R_{E,t}|R_{B,t})=0.
\]

The exterior may also include environmental variables, not only other agents. A path blanket instead requires

\[
I(R_{I,0:T};R_{E,0:T}|R_{B,0:T})=0.
\]

Neither statement is established by a dense interior, a small graph cut, similar beliefs, or a spectral community label. Minimum size is a further condition if a minimal Markov boundary is sought. A blanket can be nonminimal.

Whole-agent boundaries need special care. Independent fair bits X,E and a boundary variable B=X xor E have I(X;E|B)=log 2. A boundary that combines signals from both sides can create dependence when conditioned on. Thus a relay graph is not itself the required probabilistic factorization.

An exact conditional construction is available. Split boundary variables into sensory and active ports B=(S,A). Assume

\[
p_0(x,e,b)=p_0(b)p_0(x|b)p_0(e|b),
\]

and conditionally independent transition factors

\[
K_X(dx'|x,b)K_E(de'|e,b)
K_S(ds'|e,b)K_A(da'|x,b).
\]

For finite spaces, or dominated kernels with the usual almost-everywhere conditioning, multiplying the factors gives

\[
p(x_{0:T},e_{0:T},b_{0:T})
=p_0(b_0)F(x_{0:T},b_{0:T})G(e_{0:T},b_{0:T}).
\]

At fixed boundary path, the normalizer factors by summing or integrating over x and e separately. Consequently X_{0:T} and E_{0:T} are conditionally independent given B_{0:T}. Own-history policies are allowed when their dependencies stay on the relevant side and the boundary history. Shared unobserved innovations or boundary factors jointly depending on the two sides require further analysis.

This proves a path blanket under supplied factorization; it does not prove that policies create that factorization. Nor does it prove an instantaneous blanket. For example, X_1=E_1=B_0 with fair B_0 and fresh independent B_1 is screened off by the boundary history but not by B_1 alone. Boundary memory or another sufficient-state construction matters. [Beck and Ramstead](https://arxiv.org/html/2502.21217v1) also distinguish path and steady-state blanket claims in their detection framework.

The blanket criterion is gauge invariant. For separate bimeasurable bijections f,g,h,

\[
I(f(R_I);h(R_E)|g(R_B))=I(R_I;R_E|R_B).
\]

Conditioning on g(R_B) is equivalent to conditioning on R_B, and the two conditional laws in the defining KL are pushed forward by the same bijection. Thus pure frame changes cannot create a blanket. Physical changes in policy, channel, or retained variables can change the joint law. Noninvertible coarse-graining can change conditional independences and must be analyzed separately.

## Why condensation is conditional, and what to compare

Individual preference structures can support cooperation, competition, indifference, or refusal. They do not imply a common Lyapunov function. Even smooth local descent dx_i/dt=-grad_{x_i}F_i gives a derivative of sum_j F_j containing cross terms of unconstrained sign. This is consistent with the model-specific aggregate-EFE behavior reported by [Ruiz-Serra, Sweeney, and Harré](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1793.pdf).

Positive-temperature Gibbs policy probabilities with finite costs and positive priors never create structural zeros among admitted actions. Exact separation would need support changes, hard gates, deterministic selection with strict dominance, or an independent proof of statistical cancellation. Small cross-boundary action probabilities motivate an approximate, finite-horizon blanket question; they are not an exact blanket proof. Even tiny per-step leakage can matter over long histories.

The requested comparison should retain two distinct cases. In the role-informed case, agents have latent hypotheses about possible internal, sensory, or active roles, but no fixed identity assignment. [Palacios et al.](https://doi.org/10.1016/j.jtbi.2019.110089) provide a closely related simulation precedent. In the role-unspecified case, preferences and models contain no group or blanket label; a boundary would have to result from the costs and predictive benefits of the exchanges themselves. Both cases should use the same physical channel opportunities and explicit individual preferences so the effect of supplied role structure can be assessed.

A candidate mechanism in the second case is complementary information combined with costly communication. Some agents become useful interpreters or relays; others benefit from using them; the relays have their own reasons to continue. This can make a concentrated interface advantageous. It is a mechanism to investigate, not an established formation result. It may instead yield all-to-all exchange, silence, transient coalitions, exploitative relationships, or disconnected echo chambers.

A small next model can use N=6 or 8 agents, K=2 or 3 represented dimensions, scalar noisy modalities, heterogeneous but specified preferences, a nonzero communication cost, and delayed observable replies. Compare explicit receiver models with learned response kernels under matched information and resources; compare role-informed and role-unspecified priors; begin with fixed grounded interfaces and then add interface learning. These are proposed model-design comparisons, not experiments run in this session.

Boundary assessment should name the operational law, retained state, partition rule, tolerance, and horizon. For a nondegenerate joint Gaussian law, instantaneous conditional independence can be checked by the I,E block of its precision matrix being zero; that shortcut does not extend to non-Gaussian laws. In a finite model, compute the actual conditional mutual information. A fitted constrained blanket model is a candidate to compare against alternatives, not independent proof that the process has its imposed independence. Partitions selected using all observed states can introduce selection dependence, so selection itself belongs in the analysis.

Persistence, responsiveness to external perturbations, and continued participation under each agent's preferences are separate from screening-off. If a further claim is made that the group is itself an active-inference agent, its retained predictive state, preference representation, and policy law require their own construction. A statistical blanket alone does not complete that step.

## Status and continuation

Established here are a typed controlled-channel formulation, its passive covariance, an exact Gaussian receiver, a normalized finite chronological protocol, exact identities for a declared EFE convention, and a conditional path-blanket construction. These are analytical constructions under stated assumptions; no numerical evidence was generated. Existing manuscript results and standard probability identities are attributed rather than claimed as new.

The open research question is whether a specified nonempty regime of heterogeneous preferences and adaptive communication can generate and maintain the required boundary structure without a supplied blanket-role prior. The role-informed case is a distinct comparator. The next decisive step is to choose the operational world, individual preferred consequences, and finite communication budget, then derive the resulting policy and joint law before attempting a formation simulation. The Research vault and live manuscripts were consulted read-only.
