# Cue/result memory and MAgent Theory 2.0

The cue/result memory is an exact local realization of MAgent's generative-model definition, followed by a declared projection of its posterior. It supplies a concrete statistical mechanism for private computation: infer which learned mechanism applies to a cue, then predict a different, correlated result. Its closest theoretical home is MAgent's general correlated law s(dm)q(dk|m). The developed product-family Fisher dynamics is a narrower construction and does not reproduce this conditioning step.

The author's observation that the block works well as an MLP replacement is consistent with the expressive function class already derived in the cue/result note. This investigation establishes the mathematical relationship, rather than reproducing that performance observation. It does not identify the successful run or infer which configuration produced it.

This comparison continues the October 24 same-state memory investigation. The original identification of template labels with model presentations was already present there. The present work develops the additional cue/result conditional kernel, correlated recognition, sequential-record interpretation, and information retained by the output. The live sources were read in the Research vault; current implementation statements come from VFE_3.0. The three specialist derivations and the sequential analysis are linked below.

## The exact local model

Let A be a finite component label, X a latent cue, Y its result, and O an observed cue record. With positive component weights and positive-definite cue, result-noise, and record covariances, the joint law is

\[
\begin{aligned}
P(a,dx,dy,do)
&=\hat\pi_a\mathcal N(dx;\nu_a,L_a)
\mathcal N(dy;\omega_a+F_a(x-\nu_a),Q_a)
\mathcal N(do;x,C).
\end{aligned}
\]

In [MAgent's definition of a generative model](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:170>), take the presentation m=A, represented contents k=(X,Y), reference r(a)=\hat\pi_a, evaluator G_a equal to the joint cue/result Gaussian, and likelihood equal to the record kernel. This gives the identical normalized joint. The result-only space k=Y is also possible: integrate out X and use the induced likelihood P(O|Y,A). The [probabilistic derivation](</C:/Users/chris and christine/.codex/worktrees/items123-publication/VFE_4.0/docs/research/2026-10-28-cue-result-magent-relation/probabilistic-bridge.md>) gives that likelihood explicitly, so the comparison does not depend on doubling the persistent state dimension.

Conditioning on O=\xi gives

\[
w_a=\frac{\hat\pi_a\mathcal N(\xi;\nu_a,L_a+C)}
{\sum_b\hat\pi_b\mathcal N(\xi;\nu_b,L_b+C)},
\]

and

\[
\begin{aligned}
m_{X,a}&=\nu_a+L_a(L_a+C)^{-1}(\xi-\nu_a),\\
P_{X,a}&=(L_a^{-1}+C^{-1})^{-1},\\
m_{Y,a}&=\omega_a+F_a(m_{X,a}-\nu_a),\\
P_{Y,a}&=Q_a+F_aP_{X,a}F_a^T.
\end{aligned}
\]

The posterior has the form w_a q_a(dx,dy), generally with nonzero dependence between the component and contents. The result is the mixture \(\sum_a w_a\mathcal N(m_{Y,a},P_{Y,a})\). In MAgent's notation, w is a belief s over model presentations. It is distinct from the fixed reference prior, the learned bank parameters, and the peer-selection rows \(\beta,\gamma\). Calling the components independent agents would require specifying their own state, records, and dynamics; the model-presentation interpretation already provides an exact account without those extra assumptions.

The Gaussian family is computationally convenient, not required for the identification. On standard Borel spaces, any proper reference r, normalized cue laws \(\mu_m\), conditional result kernels \(K_m\), and record kernel \(\mathcal L\) define

\[
P(dm,dx,dy,do)=r(dm)\mu_m(dx)K_m(x,dy)\mathcal L(x,do).
\]

At records with finite positive evidence this has the same MAgent disintegration. General conditional mechanisms can therefore replace the Gaussian ones without changing the underlying interpretation.

## What changes when the result differs from the cue

Same-state retrieval estimates X from a noisy observation of X. At fixed C and bank, its response is constrained by its posterior covariance. Cue/result retrieval estimates Y instead. The exact response identity is

\[
D_\xi\mathbb E[Y\mid\xi]
=\operatorname{Cov}(Y,X\mid\xi)C^{-1}.
\]

The cross-covariance may be negative or nonsymmetric, so Bayesian prediction can reverse or rotate a response. Already for one component the Jacobian is \(FL(L+C)^{-1}\). For any prescribed dense matrix H, the choice \(F=H(L+C)L^{-1}\) realizes H exactly, with a suitable result center providing any affine intercept. This is also contained in the working note's stronger [softmax-gated affine-expert construction](</C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/cue_result_memory.tex:237>): with common cue covariance and fixed record covariance, unrestricted dense regressions realize arbitrary affine experts with an arbitrary affine-logit softmax gate.

This gives a mathematical reason the construction can fill an MLP slot. It combines nonlinear selection with component-dependent transformations and propagates uncertainty. It does not prove an advantage over an MLP, identify the cause of a measured gain, or establish the same expressivity for the current rank-two departure from identity and varying input covariance.

The statistical operation is related to Gaussian mixture regression: infer an output distribution by conditioning a joint input/output mixture. Calinon's [2016 tutorial, Section 5.1, equations (13)–(19)](https://calinon.ch/papers/Calinon-JIST2015.pdf) gives the conditional mixture and its Gaussian moment approximation. Here the explicit noisy cue record adds C to the evidence covariance and affects the conditional gain. The task-trained bank, frame convention, uncertainty-carrying interface, and optional regime state specify how that classical operation is used in this architecture.

## Exact inference and the product-family dynamics

For a candidate correlated recognition law Q(a,dk)=s_aq_a(dk), the local free energy obeys

\[
\mathcal F(Q;\xi)
=\operatorname{KL}(s\Vert w)
+\sum_a s_a\operatorname{KL}(q_a\Vert\Pi_a^\xi)
-\log Z(\xi),
\]

where \(\Pi_a^\xi\) is the exact component posterior and Z is total evidence. Thus exact conditioning minimizes the unrestricted local free energy. Its component weights are Gibbs weights for the optimized component costs \(-\log Z_a\).

MAgent's developed dynamics mostly imposes the product restriction q(k)s(m). The cue/result note's structured alternative imposes a product only over (A,X), while retaining the conditional result kernel K_a. That choice gives the label update

\[
s_a\propto\hat\pi_a\exp[-\operatorname{KL}(q_X\Vert\mu_a)],
\]

with a single shared cue posterior. This is a different approximation from exact component-wise conditioning. The contrast is the same one already formulated in MAgent between expected-log costs and log-integrated evidence. There is no universal entropy ordering that would make one set of normalized component weights always sharper.

The distinction suggests a specific theoretical extension: develop interactions and dynamics for correlated agents s(a)q(k|a), using the existing general joint-law definition. The implemented block is evidence that such a state has computational uses; it does not derive a common reciprocal potential, a population ELBO, or a natural-gradient learning rule for the full stack.

## Projection and geometry

The block returns a Gaussian with the exact result mean and covariance, or its frame-diagonal counterpart. The full Gaussian is

\[
q_G=\arg\min_{q\text{ Gaussian}}\operatorname{KL}(\Pi_Y^\xi\Vert q).
\]

The direction of this divergence matters. It is not generally the Gaussian minimizer of \(\operatorname{KL}(q\Vert\Pi_Y^\xi)\) used by variational free energy. Exact conditioning followed by this projection shares the local approximation operation of assumed-density filtering, whose moment-matching step is derived in [Minka (2001), Section 2](https://tminka.github.io/papers/ep/minka-ep-uai.pdf). That does not establish a common sequential probabilistic model for all the blocks.

If the retained state is interpreted as the product of exact label weights w and the Gaussian result carrier, its loss is exactly

\[
\operatorname{KL}(\Pi_{A,Y}^\xi\Vert w\otimes q_G)
=I_{\Pi^\xi}(A;Y)+\operatorname{KL}(\Pi_Y^\xi\Vert q_G).
\]

Carrying w therefore preserves model belief but does not retain the component-conditional result laws. A further diagonal projection discards covariance information. The [sequential-record analysis](</C:/Users/chris and christine/.codex/worktrees/items123-publication/VFE_4.0/docs/research/2026-10-28-cue-result-magent-relation/sequential-records-and-retention.md>) gives an admissible mixture whose moment carrier is N(0,1), while a subsequent Gaussian observation distinguishes the mixture from that carrier. The consequence is a precise limit on universal predictive sufficiency, not a performance diagnosis of a trained model.

The corresponding statistical geometries also differ. At fixed bank and C, the Fisher tensor of the exact posterior family of (A,X,Y), parameterized by \(\xi\), is

\[
I_\xi=C^{-1}\operatorname{Cov}(X\mid\xi)C^{-1}.
\]

It is positive even when the result-mean response reverses a direction; a Fisher tensor and a cross-prediction Jacobian measure different objects. For a general label/result Gaussian family, the joint Fisher tensor contains categorical information plus the weight-averaged component Gaussian information. It is not the Fisher tensor of the moment carrier. The [geometry derivation](</C:/Users/chris and christine/.codex/worktrees/items123-publication/VFE_4.0/docs/research/2026-10-28-cue-result-magent-relation/geometry-and-model-state.md>) gives an explicit family with constant Gaussian carrier and strictly positive joint Fisher length.

The passive frame transformation agrees with MAgent: one GL(K) acts on cue and result, and trivially on the finite labels. The regression transforms by conjugation but need not be invertible, so it is an internal response kernel rather than a group-valued inter-agent dictionary. Local passive covariance survives the chosen reference chart. Arbitrary rebasing of that chart generally takes diagonal banks outside their parameter family. A global base connection, curvature, or formation mechanism is not specified by these frame laws.

## Shared models, records, and the live implementation

A shared bank gives the same evaluator and baseline categorical reference. MAgent's literal shared-model definition also requires compatible record kernels. At equal C this condition holds in the common reference chart. With token-dependent C_i, the direct statement is a shared conditional mechanism indexed by supplied uncertainty; identifying one fixed likelihood across tokens needs an explicit side-information convention. Sharing parameters also does not make the tokens' posterior beliefs equal.

The optional carried prior has the form \(\hat\pi_b(w^lT)_b^\vartheta\), after normalization. For unit exponent and uniform static prior, its label update is exact filtering in the normalized hidden Markov emission model constructed in the sequential note. A nonuniform static prior multiplies the predicted law again. The source's Dirichlet-record construction provides another exact local interpretation of this multiplication and tempering. It does not establish that a belief computed earlier in the same network has the declared sampling law or is independent of the next cue. Correct treatment of reuse requires the actual conditional record model.

The [implementation correspondence](</C:/Users/chris and christine/.codex/worktrees/items123-publication/VFE_4.0/docs/research/2026-10-28-cue-result-magent-relation/implementation-correspondence.md>) records exact source locations. In the inspected launcher, the bank is tied across layers, inference is exact, the regression is a rank-two departure from identity, and carried regime state is off. Layer normalization is active before the memory call, with the configured covariance treatment. Optional regime state, when enabled, persists across layers within one stack invocation; it is initialized afresh on each invocation. These facts describe the inspected files, not the author's successful training run.

The bank parameters persist and are trained by the outer task loss. The current call conditions on its input and returns moments; it does not assimilate the input into an episodic store or run a local EM learner. Thus three different objects should remain separate: learned statistical memory in the bank, transient posterior component belief, and a register of accumulated observations. MAgent provides definitions for all three roles, but the current implementation does not instantiate all three.

## The next theoretical step

The useful continuation is a concrete MAgent model with finite presentations and correlated content, whose evaluator is a cue/result kernel. Specify its communication input, the record and uncertainty supplied after communication, the retained posterior, and the operation that consumes that posterior. The existing exact one-block free-energy identity then applies without alteration. Projection can be included explicitly as the selected computational approximation.

For a depth interpretation, choose whether records are generated by a declared predictive process or whether depth is an iterative inference algorithm over one already-declared joint. The first option requires the actual history-conditioned record kernel. The second requires an objective and an update rule for that joint. The present stack is not proved to satisfy either description merely because each block has a normalized local model. MAgent's internal-record appendix already supplies the distinction between normalized histories and Bayesian assimilation, so this work can start from that formulation.

For inheritance or coarse-graining, specify which future readouts must survive and apply the existing configuration-retention criterion. Matching first two moments is sufficient for expectations of affine and quadratic readouts, but fails for all future Gaussian observation laws. This limits that retention, not general-law parent models. A richer joint state or a restricted operational interface remains possible.

The completed work consists of analytical identities, explicit constructions and counterexamples, primary-source checks, and static tracing of the live implementation. No training, trajectory experiment, GPU query, Lean work, code/configuration edit, or manuscript integration was performed. The report and four supporting notes are local research artifacts; the source-preservation result is recorded separately in `validation.json`.
