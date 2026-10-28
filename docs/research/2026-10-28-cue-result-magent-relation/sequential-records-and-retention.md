# Repeated cue/result inference: records and retained state

The one-block correspondence does not require a new theory of records. MAgent already distinguishes a normalized history of internal events from Bayesian assimilation under a particular observation model. The relevant constructions are in [the internal-record appendix](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/appendices/record_process.tex:1>), especially `prop:registered-history`, `prop:bayesian-history`, and `prop:variational-history`. The issue for cue/result memory is specifying which of these constructions its repeated calls realize.

## An exact sequential realization

Keep the finite bank fixed and let the component presentation evolve by a stochastic matrix T. At each depth l, generate a fresh cue X_l from the selected component, generate its result Y_l, and observe O_l through the declared cue likelihood. Conditional on the component sequence, take the draws at different depths to be independent. For fixed positive-definite record covariances C_l the finite joint law is

\[
P(a_{0:n},dx_{0:n},dy_{0:n},do_{0:n})
=\hat\pi_{a_0}\prod_{l=1}^n T_{a_{l-1}a_l}
\prod_{l=0}^n\left[\mu_{a_l}(dx_l)K_{a_l}(x_l,dy_l)
\mathcal L_{C_l}(x_l,do_l)\right].
\]

Every factor is a normalized conditional law. Summing the last label and integrating its variables gives one conditional on the preceding label; iterating proves normalization and consistency. Standard Borel kernels give the corresponding infinite process. The construction can also use record covariances chosen measurably from past records, provided this predictable design is part of the model.

Write w_l(a)=P(A_l=a|O_{0:l}). By the Markov property and conditional independence of the next emission,

\[
v_l(b)=\sum_a w_{l-1}(a)T_{ab},\qquad
Z_b(o_l;C_l)=\int\mu_b(dx)\mathcal L_{C_l}(o_l|x),
\]

and Bayes' rule gives

\[
w_l(b)=\frac{v_l(b)Z_b(o_l;C_l)}{\sum_c v_l(c)Z_c(o_l;C_l)}.
\]

Conditional on A_l=b and the present record, the posterior of (X_l,Y_l) is exactly the one-block cue/result posterior. The previous history affects it only through v_l. These statements follow by integrating the normalized joint above, so they are exact for this declared model.

This construction identifies a limited but useful sufficiency result. The label vector w_l is sufficient for the next label prediction in this model, because the next cue is freshly drawn given A_{l+1} and does not depend on Y_l. Discarding the present continuous mixture does not spoil that particular label filter. It can still spoil a readout of the present result distribution. If the next cue, a message, or a transition depends on Y_l or on the pair (A_l,Y_l), the displayed sufficiency argument no longer applies. Its filtering state must then retain the appropriate joint law or satisfy a separate sufficiency condition.

The construction is a finite-state hidden Markov emission model with a conditional result variable. It does not identify the implemented transformer stack with this process: the next cue there is produced by deterministic computation, including communication from other tokens, rather than by the emission mechanism just declared. The current note makes the same restriction in `crm:rem:gpb`.

## The actual carried prior has a distinct meaning

The manuscript and optional code use

\[
\hat\pi^{l+1}_{ib}\propto\hat\pi_b(v_{ib})^{\vartheta},
\qquad v_i=w_i^lT,
\]

followed by multiplication by the new Gaussian evidence. For \(\vartheta=1\) and uniform \(\hat\pi\), this is exactly the label step above. With a nonuniform static prior, that prior is multiplied again at every call; with other exponents, the prediction is tempered. These are legitimate algorithmic choices but are not the unmodified filter of the displayed homogeneous chain.

The note already provides an exact local alternative interpretation. With a baseline prior \(\hat\pi\), observe a simplex-valued record o under

\[
o\mid B=b\sim\operatorname{Dir}(\alpha\mathbf 1+\vartheta e_b),
\qquad \alpha>0.
\]

The normalization constant is independent of b, so observing o=v gives precisely \(P(B=b|o=v)\propto\hat\pi_bv_b^\vartheta\). Combining this with the cue likelihood is a normalized two-record model if their conditional relationship is declared appropriately. Conditional independence given B and its cue is one possible assumption; it is an additional assumption about the joint records. The numerical formula for the posterior does not establish that the network's computed v has this sampling law or conditional independence.

Correlation with an earlier record does not make Bayesian inference impossible. It requires using the actual conditional likelihood given the retained history. For example, if a later record O_2=f(O_1) is deterministic and O_1 is already available, then

\[
P(Z\in B\mid O_1,O_2)=P(Z\in B\mid O_1)
\]

for every measurable B. Treating O_2 as an independent replicate would instead count the same evidence twice. Tempering that replicate by a fixed exponent does not generally recover the true conditional likelihood. This example concerns information already in the receiver's history. Communication can deliver information that is new to that receiver even when the complete multi-agent computation is deterministic.

MAgent's registered-event construction can represent the actual stack as a normalized process once the complete state, interaction update, record readout, and initial law are specified. A deterministic update is a Dirac transition kernel; an accessible record can be stored in an explicit register. That construction establishes a law of computation. Establishing that the algorithm's beliefs are the Bayesian posteriors of its actual record process is a further claim. The internal-record appendix already states this distinction.

## What the Gaussian carrier preserves

For a result mixture p with finite second moments, moment matching preserves the expectations of affine and quadratic functions of Y. Thus it preserves one affine-Gaussian prediction's mean and covariance when that prediction reads only Y. It does not preserve the complete law of that prediction, a nonlinear likelihood, or future posterior conditioning in general.

An explicit admissible result mixture illustrates the boundary. Let

\[
p_t=\tfrac12\mathcal N(t,1-t^2)+\tfrac12\mathcal N(-t,1-t^2),
\qquad 0<t<1.
\]

This is a cue/result bank output with identical cue laws, zero regressions, equal weights, result centers ±t, and result variances 1-t². Its moment projection is q_G=N(0,1). Let a later, normalized sensor be O|Y=y ~ N(y,c), with c>0. At the record value zero,

\[
(p_tK_c)(0)=\frac{\exp[-t^2/(2(1-t^2+c))]}{\sqrt{2\pi(1-t^2+c)}},
\qquad
(q_GK_c)(0)=\frac1{\sqrt{2\pi(1+c)}}.
\]

They differ for every t>0. Indeed, the logarithm of the first expression has derivative \(-t^3/(1-t^2+c)^2<0\), and the expressions agree at t=0. These are densities at zero; continuity gives differing probabilities on a sufficiently small interval around zero. Even an ordinary Gaussian observation can therefore distinguish the exact mixture from its Gaussian carrier.

This is a direct counterexample to universal preservation by moment matching, not a claim about the size of a language-model error. It does not assert that the input-to-moments map of every fixed bank is noninjective. If a complete retained state includes enough bank data and input records to reconstruct the original posterior, that larger retention must be evaluated on its own terms.

If both w and q_G are retained as a product law, the exact one-step loss is

\[
\operatorname{KL}(\Pi_{A,Y}\Vert w\otimes q_G)
=I_{\Pi}(A;Y)+\operatorname{KL}(\Pi_Y\Vert q_G).
\]

The first term is the dependence removed by replacing a joint law with its marginals; the second is the marginal information removed by Gaussian projection. The proof and the additional diagonal-projection cost are in [the probabilistic derivation](</C:/Users/chris and christine/.codex/worktrees/items123-publication/VFE_4.0/docs/research/2026-10-28-cue-result-magent-relation/probabilistic-bridge.md>). This identity is not a claim that the code stores and samples a product law: it quantifies that particular mathematical interpretation of the retained state.

## Relation to MAgent inheritance

The relevant MAgent distinction is between a law-level sample channel and a map retaining a complete agent configuration (`inh:two-retentions`). Gaussian moment projection is a nonlinear operation on a law; it is not automatically a fixed Markov kernel on each represented sample. Nor does a projected law alone specify future interactions, learning, or the bank parameters required to compute them.

For a declared complete update F and configuration retention R, exact deterministic inheritance requires

\[
R(x)=R(x')\implies R(F(x))=R(F(x')).
\]

This is the existing `inh:deterministic-descent` criterion in [Chapter 5](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/05_inheritance.tex:66>). Applying it to cue/result memory requires specifying the complete state, allowed future records and messages, bank parameters, and any label state. The example above shows that retaining only result moments does not preserve all Gaussian readout laws. It does not establish an obstruction to richer retained states, task-restricted interfaces, or non-Gaussian parents.

The practical theoretical continuation is therefore concrete: define a correlated finite-presentation agent, give its record and communication kernels, and choose whether its next operation reads the exact joint, the exact result marginal, or a projected carrier. The hidden Markov construction above is one exactly closed special case for label prediction. A complete interacting stack needs a declared rule for how communication changes its cue record and uncertainty, together with an explicit account of projection. No new formation or recursive inheritance theorem follows from the local memory correspondence alone.

The step of exact conditioning followed by an inclusive-KL projection has a classical precedent in assumed-density filtering. Minka's [2001 paper, Section 2](https://tminka.github.io/papers/ep/minka-ep-uai.pdf) defines that procedure and derives moment matching for exponential-family approximations. This is a correspondence of the local approximation operation; the present stack has not thereby been identified as filtering a specified common sequential joint.

All statements here were derived analytically. Static artifact checks were performed separately; no numerical experiments, sampling, training, GPU queries, Lean work, or live source edits were performed for this note.
