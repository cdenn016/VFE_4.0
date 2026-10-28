# Gaussian communication and the retained dependence

This note gives an exact first-message calculation for two correlated cue/result banks, an exactly closed Gaussian special case, and a fixed-model counterexample to retaining only the agents' local joint beliefs. The user requests an interacting model with correlated presentation/content beliefs and communication-generated records before moment projection. The Gaussian family, common known preparation, private cue sensors, posterior-sampling transmission, and reliable common transcript are introduced modeling assumptions. They are not assertions about the implemented transformer or about unrestricted agent dynamics.

There are two separate constructions. A normalized coupling of the local banks gives the first communication likelihood by integration over the sender's private record. A binary preparation with an accessible parity record gives two reachable histories with identical local joint beliefs and different subsequent assimilation. The first load-bearing step is the conditional law of the transmitted message; the second is equality of the retained states before, but not after, the same received message. Neither requires a numerical experiment.

## A normalized coupling of two Gaussian banks

For agents indexed by i=0,1, let A_i take values in finite sets, and let the cue and result be vectors X_i and Y_i. Write Z_i=(A_i,X_i,Y_i). Choose a strictly positive label table p_ab with total mass one and marginals pi_0a and pi_1b. For every label pair choose

\[
\begin{pmatrix}X_0\\X_1\end{pmatrix}\mid(A_0=a,A_1=b)
\sim\mathcal N\left(
\begin{pmatrix}\nu_{0a}\\\nu_{1b}\end{pmatrix},
\begin{pmatrix}L_{0a}&S_{ab}\\S_{ab}^T&L_{1b}\end{pmatrix}
\right),
\tag{1}
\]

where every block covariance is positive definite. For example,
S_ab=L_0a^(1/2) J_ab L_1b^(1/2) with operator norm of J_ab less than one is sufficient. Choose positive-definite Q_ia and arbitrary F_ia, and generate the results independently conditional on the cues and labels by

\[
Y_i\mid X_i=x,A_i=a
\sim\mathcal N\bigl(\omega_{ia}+F_{ia}(x-\nu_{ia}),Q_{ia}\bigr).
\tag{2}
\]

The resulting joint prior P_0(dz_0,dz_1) is normalized: sum its label probabilities, integrate the joint cue density, and integrate the two result kernels. Because the i-th marginal of (1) does not depend on the other label, its local marginal is exactly

\[
P_0(A_i=a,dx_i,dy_i)
=\pi_{ia}\mathcal N(dx_i;\nu_{ia},L_{ia})K_{ia}(x_i,dy_i).
\tag{3}
\]

Thus it retains the published local cue/result bank while allowing both correlated labels and correlated cues across agents. The coupling table and cross-covariances are additional common model data. They cannot be inferred from the two local banks alone.

Private records are

\[
D_i=X_i+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,C_i),\qquad C_i\succ0,
\tag{4}
\]

with independent sensor noises, also independent of the prepared variables. Agent i sees D_i but not D_(1-i). Before the first public message the sender, agent 1, therefore uses precisely its local posterior from (3)-(4). Define

\[
\begin{aligned}
W_b&=L_{1b}+C_1,&
w_b(d)&=\frac{\pi_{1b}\mathcal N(d;\nu_{1b},W_b)}
 {\sum_c\pi_{1c}\mathcal N(d;\nu_{1c},W_c)},\\
G_b&=L_{1b}W_b^{-1},&
P_b&=L_{1b}-L_{1b}W_b^{-1}L_{1b},\\
m_b(d)&=\omega_{1b}+F_{1b}G_b(d-\nu_{1b}),&
V_b&=Q_{1b}+F_{1b}P_bF_{1b}^T.
\end{aligned}
\tag{5}
\]

Completing the Gaussian quadratic gives P_b=(L_1b^(-1)+C_1^(-1))^(-1), so P_b and V_b are positive definite. The sender's exact result prediction is the mixture with weights w_b(d), means m_b(d), and covariances V_b. Its label and contents generally remain dependent; no product recognition restriction is imposed.

The communication policy samples a fresh variable \(\widetilde Y_1\) from this prediction and sends

\[
H=B\widetilde Y_1+\eta,\qquad \eta\sim\mathcal N(0,R),\qquad R\succ0.
\tag{6}
\]

The sampling randomness and encoder noise are independent of the actual represented variables and private records conditional on the sender's information. In particular, \(\widetilde Y_1\) is not the actual prepared Y_1. The normalized message density conditional on the private record is

\[
\kappa(h\mid d)
=\sum_b w_b(d)\mathcal N(h;Bm_b(d),R+BV_bB^T).
\tag{7}
\]

All matrices and the policy are common knowledge; only the realized D_1 is private. A direct physical sensor with conditional density \(\mathcal N(h;BY_1,R)\) would define a different experiment.

## The receiver's actual likelihood and posterior

Conditional on the complete represented state z, the message density is

\[
\lambda(h\mid z)
=\int\mathcal N(d;x_1,C_1)\kappa(h\mid d)dd.
\tag{8}
\]

The integral first generates the sender's private record under its actual sensor, then applies its posterior-sampling policy. It does not replace either operation by a direct observation of Y_1. Given D_0=d_0 and H=h, the exact augmented posterior is

\[
\Pi(dz,dd_1\mid d_0,h)
=\frac{P_0(dz)\mathcal N(d_0;x_0,C_0)
 \mathcal N(d_1;x_1,C_1)\kappa(h\mid d_1)}{Z(d_0,h)}dd_1.
\tag{9}
\]

Here Z is the integral of the numerator. It is positive and finite at every finite d_0,h: all component Gaussian densities are positive, the sensor density is bounded, and (7) is bounded by the largest Gaussian density maximum of its finitely many components. The same facts justify the integrations by Tonelli's theorem for nonnegative integrands. Integrating h and d_0 before the prepared state recovers one, proving normalization of the full experiment.

For the receiver's own local posterior, define

\[
\begin{aligned}
u_{ab}(x)&=\nu_{1b}+S_{ab}^TL_{0a}^{-1}(x-\nu_{0a}),\\
U_{ab}&=C_1+L_{1b}-S_{ab}^TL_{0a}^{-1}S_{ab},\\
\Lambda_0(h\mid a,x)&=\sum_b\frac{p_{ab}}{\pi_{0a}}
 \int\mathcal N(d;u_{ab}(x),U_{ab})\kappa(h\mid d)dd.
\end{aligned}
\tag{10}
\]

The conditional cue formula in (10) follows by completing the joint quadratic in (1). Since the marginal law of X_0 given A_0=a is independent of b, conditioning further on X_0=x does not change the weights p_ab/pi_0a of A_1. The positive Schur complement in (1), plus C_1, makes U_ab positive definite. The receiver's correlated local recognition is

\[
\zeta_0(a,dx,dy\mid d_0,h)
\propto\pi_{0a}\mathcal N(dx;\nu_{0a},L_{0a})K_{0a}(x,dy)
 \mathcal N(d_0;x,C_0)\Lambda_0(h\mid a,x).
\tag{11}
\]

Normalizing (11) gives the same marginal as (9). For this first-message experiment, the result conditional K_0a remains unchanged after conditioning on A_0,X_0, since all new likelihood terms are independent of Y_0 at that conditioning. The presentation weights and cue conditionals change together. The integral in (10) is the exact interaction term, before moment matching or any Gaussian closure.

## An affine Gaussian subcase that closes exactly

Take one component per agent, scalar cues/results, and
Cov(X_0,X_1)=S with S squared less than L_0 L_1. Write the sender regression as f_1 and encoder as b. Put

\[
g=\frac{L_1}{L_1+C_1},\qquad P=\frac{L_1C_1}{L_1+C_1},\qquad a=bf_1g.
\tag{12}
\]

One can implement the posterior sample using an independent Gaussian \(\xi\) of variance Q_1+f_1 squared P. Substituting D_1=X_1+epsilon_1 into (5)-(6) gives

\[
H=b\omega_1+a(X_1-\nu_1)+a\epsilon_1+b\xi+\eta.
\tag{13}
\]

Consequently

\[
\begin{aligned}
H\mid X_1=x_1&\sim
 \mathcal N\bigl(b\omega_1+a(x_1-\nu_1),N_{\mathrm{eff}}\bigr),\\
N_{\mathrm{eff}}&=a^2C_1+b^2(Q_1+f_1^2P)+R.
\end{aligned}
\tag{14}
\]

The direct sensor H_phys=bY_1+eta instead has gain bf_1 and conditional variance b squared Q_1+R. Thus posterior sampling attenuates the signal by g and adds the conditional variance

\[
b^2 f_1^2(g^2 C_1+P)
=b^2 f_1^2\frac{L_1C_1(2L_1+C_1)}{(L_1+C_1)^2}.
\tag{15}
\]

Despite this difference, the unconditional message distributions agree:

\[
\begin{aligned}
\operatorname{Var}(H)
&=a^2L_1+N_{\mathrm{eff}}
=b^2(f_1^2L_1+Q_1)+R
=\operatorname{Var}(H_{\mathrm{phys}}),\\
\operatorname{Cov}(X_0,H)&=aS,\qquad
\operatorname{Cov}(X_0,H_{\mathrm{phys}})=bf_1S.
\end{aligned}
\tag{16}
\]

Both have mean b omega_1. The variance identity uses g squared (L_1+C_1)+P=L_1. Equal unconditional message laws therefore do not identify their likelihoods or their implications for another agent.

For an observed d_0, first condition the jointly Gaussian cues on D_0. The conditional quantities are

\[
\begin{aligned}
m_0&=\nu_0+\frac{L_0}{L_0+C_0}(d_0-\nu_0),&
v_0&=\frac{L_0C_0}{L_0+C_0},\\
m_1^0&=\nu_1+\frac{S}{L_0+C_0}(d_0-\nu_0),&
v_1^0&=L_1-\frac{S^2}{L_0+C_0},\\
c_{01}^0&=\frac{SC_0}{L_0+C_0}.&&
\end{aligned}
\tag{17}
\]

Let m_H=b omega_1+a(m_1^0-nu_1) and V_H=a squared v_1^0+N_eff. Completing the next conditional Gaussian gives

\[
\begin{aligned}
X_0\mid d_0,h&\sim\mathcal N(m_0^+,v_0^+),\\
m_0^+&=m_0+\frac{a c_{01}^0}{V_H}(h-m_H),\\
v_0^+&=v_0-\frac{a^2(c_{01}^0)^2}{V_H},\\
Y_0\mid d_0,h&\sim
\mathcal N\bigl(\omega_0+f_0(m_0^+-\nu_0),Q_0+f_0^2v_0^+\bigr).
\end{aligned}
\tag{18}
\]

The Schur complements are positive because the prior and both sensor and encoder noises are nondegenerate. Equation (18) is exact conditioning, not projection.

There is also an exact repeated-message closure for this singleton-label affine Gaussian family. The full common Gaussian law of (Z_0,Z_1,D_0,D_1) is sufficient, but it is larger than necessary. Write D=(D_0,D_1) and retain the fixed preparation conditional W(d,dz)=P_0(dz|D=d). Every subsequent message depends on the represented state only through D and the preceding transcript. Hence the exact common posterior factors as

\[
\Gamma_t(dz,dd)=W(d,dz)\lambda_t(dd),\qquad
\lambda_t=\mathcal L(D\mid H_{1:t}).
\tag{18a}
\]

Indeed, at a fixed transcript the product of all message likelihoods is a function of d alone. It reweights the marginal law of D and leaves the conditional law of Z given D unchanged. Thus the dynamic state can be the joint Gaussian lambda_t on the two private records, with its cross-covariance, together with the fixed W, known model and policy, schedule, and each agent's realized private record. W reconstructs every required represented-variable conditional. This is a statistical sufficiency statement; any separately observable record register must also be preserved.

To prove Gaussian closure, the fixed W has an affine conditional mean and constant covariance. If lambda_t is Gaussian, W lambda_t is jointly Gaussian, so Y_j given D_j=d has an affine conditional mean and covariance independent of d. A fresh posterior sample followed by a linear Gaussian encoder gives a message with mean affine in D_j and constant conditional covariance. Multiplying lambda_t by this likelihood leaves it Gaussian. The initial lambda_0 is Gaussian, and induction proves exact finite-dimensional closure at every finite step. No step replaces the joint private-record covariance by its diagonal or separate-agent covariances. This is a special exact Gaussian filter; it is not a finite-mixture closure theorem for multiple presentations.

## Multiple components need not give a finite Gaussian-mixture posterior

The obstruction already appears with a singleton receiver label and two sender labels b in {-1,+1}. Let their prior probabilities be 1/2, receiver cue mean be zero, sender cue means be b alpha with alpha>0, and use the same positive cue variances L_0,L_1 and cross-covariance S>0 in both label components. Require S squared less than L_0 L_1. Choose sender result means b m with m>0, common positive result variance Q, and F_1b=0. Use the identity encoder with positive variance R. Receiver sensor variance C_0 and sender sensor variance C_1 are positive.

The sender's positive-label posterior weight is

\[
w_+(d)=\frac1{1+\exp[-2\alpha d/(L_1+C_1)]}.
\tag{19}
\]

For a fixed received h>0, put c_+(h)=N(h;m,Q+R) and c_-(h)=N(h;-m,Q+R). Both are positive and c_+>c_-. Equation (7) becomes

\[
\kappa(h\mid d)=c_-+(c_+-c_-)w_+(d).
\tag{20}
\]

Given X_0=x and A_1=b, D_1 has mean b alpha+(S/L_0)x and variance L_1-S squared/L_0+C_1. The label probabilities stay 1/2 because the receiver cue marginal does not depend on b. Hence (10) and bounded convergence give

\[
\lim_{x\to+\infty}\Lambda_0(h\mid x)=c_+,
\qquad
\lim_{x\to-\infty}\Lambda_0(h\mid x)=c_-.
\tag{21}
\]

To justify the limit, represent each conditional D_1 as its moving mean plus one fixed centered Gaussian. Pointwise, w_+(D_1) tends to one or zero, and it is bounded between zero and one. This supplies an integrable dominator independently of x.

Before the message, X_0 given D_0=d_0 has a nondegenerate Gaussian density g_0(x). After the message its exact density is

\[
p^+(x)=\frac{g_0(x)\Lambda_0(h\mid x)}{Z_h},\qquad
Z_h=\int g_0(x)\Lambda_0(h\mid x)dx.
\tag{22}
\]

The ratio p^+/g_0 therefore has different positive finite limits c_+/Z_h and c_-/Z_h at the two tails. No finite positive mixture of nondegenerate Gaussians can have this property. Indeed, a component with variance larger than that of g_0 makes the ratio diverge at both tails. A component with equal variance but a different mean makes it diverge at one tail. Positivity of the weights prevents cancellation. Every component with smaller variance vanishes relative to g_0 at both tails. Thus the only remaining components contributing a nonzero finite limit have the same mean and variance as g_0, and they contribute the same limit at both tails. This contradicts (21).

The posterior in (22) is therefore not any finite Gaussian mixture. This is an exact counterexample to universal finite-mixture closure under the specified posterior-sampling communication policy, even though every prior component, private sensor, and encoder is Gaussian. It is not a failure of normalization or exact inference: (9)-(11) give the exact normalized answer. Nor is it an obstruction to an integral representation or a richer retained state.

## A reachable collision for local-only retention

The next example is finite and uses one fixed preparation and one fixed communication policy. All variables are bits. Draw X_0 fairly and draw B independently with P(B=0)=rho, where 0<rho<1. Set X_1=X_0 xor B. Given the cues, independently draw A_i with P(A_i=X_i)=3/4 and draw Y_i with P(Y_i=1-X_i)=3/4. Each X_i is fair, so each local bank has

\[
P(A_i=a)=\tfrac12,\qquad
P(X_i=a\mid A_i=a)=\tfrac34,\qquad
P(Y_i=1-x\mid X_i=x,A_i=a)=\tfrac34.
\tag{23}
\]

Thus the local presentation and cue are correlated, and the result kernel is a noisy negative response. The generation of A_i conditional on X_i is just a convenient preparation order; Bayes' rule gives exactly the cue/result factorization in (23).

An internal preparation readout publicly registers C=X_0 xor X_1=B. This is a normalized deterministic readout of the two prepared cues; it introduces no extra external observed variable. Agent 0 has the uninformative private record D_0=star. Agent 1 has a private record with P(D_1=X_1)=3/4, using independent noise. The actual register C belongs to the common history. Both C=0 and C=1 occur with positive probability.

Conditional on either C, each individual cue is still fair. Therefore the complete local joint laws of (A_i,X_i,Y_i) given C are identical for C=0 and C=1. For either fixed d, the sender's local law given D_1=d,C is also independent of C, since its local joint (A_1,X_1,Y_1,D_1) is independent of the parity. The receiver's local law is unchanged as well. In particular, the two histories with C=0 or C=1 and the same realized d_1 produce identical pairs of local recognition laws and identical private-record values.

Agent 1 now sends a fresh posterior sample H of Y_1. From (23) and the sensor,

\[
P(H=1\mid D_1=0,C)=\tfrac58,\qquad
P(H=1\mid D_1=1,C)=\tfrac38.
\tag{24}
\]

Integrating the sender-private record gives

\[
P(H=1\mid X_1=0,C)=\tfrac9{16},\qquad
P(H=1\mid X_1=1,C)=\tfrac7{16}.
\tag{25}
\]

Both equations follow by multiplying two two-outcome kernels. For example, the first probability in (25) is (3/4)(5/8)+(1/4)(3/8)=9/16. Since X_1 is fair given C, P(H=1|C)=1/2 in both histories. Nevertheless, the same received message H=1 gives the receiver

\[
P(X_0=1\mid C=0,H=1)=\tfrac7{16},\qquad
P(X_0=1\mid C=1,H=1)=\tfrac9{16}.
\tag{26}
\]

For C=0 the two cues are equal, while for C=1 they are opposite. The receiver's model and result marginals also differ:

\[
\begin{array}{c|cc}
 &C=0&C=1\\
P(A_0=1\mid C,H=1)&15/32&17/32\\
P(Y_0=1\mid C,H=1)&17/32&15/32.
\end{array}
\tag{27}
\]

These follow from P(A_0=1|X_0)=1/4+X_0/2 and P(Y_0=1|X_0)=3/4-X_0/2. They show a difference in the next inference and the next result prediction, not merely in an inaccessible cross-correlation.

Let R retain only the two local joint recognition laws, their fixed banks, private-record values, and the communication schedule, while deleting C and the corresponding cross-agent dependence. The two prepared histories above have the same R. Applying the exact assimilation map for the same positive-probability message h=1 gives different R by (26)-(27). Hence no exact update on this retained state can reproduce the declared protocol. This is a direct failure of the existing retained-fiber condition R(x)=R(x') implies R(F_h(x))=R(F_h(x')). It remains a failure if one stores every local label/content dependence exactly; the missing information here is cross-agent and history dependence.

Keeping C resolves this particular collision. Keeping the common posterior Gamma over the represented variables and private records also resolves it, because Gamma encodes the parity relation. For the reduced form (18a), this preparation readout must be retained in the fixed reconstruction W_C=P_0(Z|D,C), or included in the preparation record coordinates. The local-only retention deletes that distinguishing data. The example does not prove minimality of Gamma, exclude smaller sufficient states for restricted interfaces, or say that every fixed initialization necessarily produces a local-belief collision. It supplies two reachable histories of the one specified preparation and policy. No Gaussian projection was used anywhere in this counterexample.

## Projection comes after the exact experiment

The exact interacting update acts on the common joint law and its conditionals, or on the equivalent lambda state with its fixed reconstruction W. Only after deriving it can one choose to replace a local result marginal by a Gaussian with matching moments, or to replace the common joint by another specified family. The first operation does not, by itself, define the second: local moments do not supply the conditional law of an unobserved peer record given the receiver's represented state.

For a local posterior Pi_(A,Y) with finite entropy and nondegenerate finite covariance, let w be its label marginal and q_G its Gaussian result moment match. Where the displayed relative entropies are finite,

\[
\operatorname{KL}(\Pi_{A,Y}\Vert w\otimes q_G)
=I_\Pi(A;Y)+\operatorname{KL}(\Pi_Y\Vert q_G).
\tag{28}
\]

Insert w tensor Pi_Y into the logarithmic density ratio to prove the equality. It separates local label/result dependence from non-Gaussian marginal information. It does not include the further interagent or private-evidence dependence needed by (10) and (26). An approximation that replaces Gamma therefore needs an explicit reconstruction of those omitted conditionals and must be assessed against the resulting message and assimilation laws.

Storing moments alongside a complete common transcript, fixed initial model, and enough information to reconstruct Gamma is a different retention. If that reconstruction is exact, the Gaussian carrier is an auxiliary output and does not itself prove any irreversible information loss. The counterexample concerns a retention that actually deletes the distinguishing information.

## Source continuity, evidence, and scope

The consulted live sources are the MAgent Theory 2.0 `AUTHORING.md`, `appendices/conditional_memory.tex` at `cmem:model`, `cmem:gaussian-posterior`, `cmem:projection`, `cmem:readout-loss`, and `cmem:sequential`; `appendices/record_process.tex` at `prop:registered-history` and `prop:bayesian-history`; and `chapters/05_inheritance.tex` at `inh:two-retentions` and `inh:deterministic-descent`. The earlier `docs/research/2026-10-28-cue-result-magent-relation/sequential-records-and-retention.md` already identifies the need for a history-conditioned likelihood and warns that local moments do not generally preserve future Gaussian records. This note resolves a specified interacting case of that obligation rather than replacing those earlier results.

All calculations here are analytical. No scripts, numerical experiments, sampling, GPU queries, Lean work, implementation edits, or Research-vault edits were performed. Only this assigned research artifact was written. The exact constructions establish a communication likelihood and posterior for the stated cooperative common-model protocol, exact Gaussian closure for its affine singleton-label specialization using the joint private-record law and fixed reconstruction, and a concrete local-only retention failure. They do not establish a minimal general retained state, an efficient multi-component approximation, autonomous learning of the bank, equivalence to reciprocal Fisher dynamics, or formation. The next reduction question is whether the joint private-record law admits a smaller protocol-specific sufficient statistic while preserving all future transmission policies and assimilations.
