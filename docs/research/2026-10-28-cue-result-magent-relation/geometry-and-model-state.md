# Cue/result memory as an MAgent statistical agent: geometry and model state

The cue/result bank gives an exact instance of MAgent's general joint agent law before the output mixture is collapsed. Its finite component label is a model presentation, its posterior weights are a belief over presentations, and its component cue/result laws are evaluators. The existing reference-coordinate implementation supplies the same passive frame convention as MAgent. It does not supply an inter-agent connection or identify the Gaussian output carrier's Fisher geometry with that of the joint posterior.

## Target and sources

The target is a typed identification for every finite bank with positive component weights and positive cue, result-noise, and record covariances, at one fixed context and one fixed record. The identification is an equality of normalized probability models and their coordinate transformation laws. A statement about shared models across tokens must additionally respect the record-covariance and carried-prior conventions. No performance claim is investigated here; the user's observation that the block works well as an MLP replacement is separate evidence.

The source definitions are the cue/result note's `crm:def:model`, `crm:prop:posterior`, `crm:prop:mproj`, `crm:prop:passive`, `crm:prop:rebasing`, `crm:lem:monomial`, and `crm:def:carried`; MAgent's `def:agent-section`, `eq:agent-disintegration`, `def:generative-model`, `def:shared-model`, `geo:joint-fisher`, `geo:pullback`, and `hyp:int-reference-consistency`. The relevant live sources are [cue_result_memory.tex](</C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/cue_result_memory.tex:56>), [01_foundations.tex](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:8>), [02_geometry.tex](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/02_geometry.tex:38>), and [02_interactions.tex](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/02_interactions.tex:55>). `AUTHORING.md` was read. The earlier [same-state comparison](</C:/Users/chris and christine/Desktop/VFE_3.0/explainer_video/research/mlp_as_generative_models.md:1>) already identified template indices with presentations; the present relation extends that existing construction to cue/result laws and carried regime belief.

## Exact typed embedding

For a bank of width $K$ and $M$ components, use the following objects at a fixed context $c_*$.

| MAgent object | Cue/result realization |
| --- | --- |
| Structure group | One group $G=\mathrm{GL}(K)$ |
| Represented content space $\mathsf K$ | $V\oplus V$, with $V=\mathbb R^K$ and content $k=(x,y)$ |
| Presentation space $\mathsf M$ | The finite set $\{1,\ldots,M\}$, with presentation $a$ |
| Content action | $h\cdot(x,y)=(hx,hy)$ |
| Presentation action | $h\cdot a=a$ for the same $h\in G$ |
| Presentation reference $r$ | $r(a)=\hat\pi_a$ |
| Evaluator $\mathcal G(a,dx,dy)$ | $\mathcal N(dx;\nu_a,L_a)\mathcal N(dy;\omega_a+F_a(x-\nu_a),Q_a)$ |
| Record kernel at supplied $C$ | $\ell_C((x,y),a;do)=\mathcal N(do;x,C)$ |
| Record value | $o=\xi$ |
| Joint recognition | $\zeta_{\xi,C}(a,dx,dy)=w_a q_a(dx,dy\mid\xi,C)$ |
| Model belief $s$ | $s(a)=w_a$ |
| Result marginal | $\sum_a w_a\mathcal N(dy;m_{Y,a},P_{Y,a})$ |

The finite presentation space is standard Borel and its trivial action is jointly Borel. This satisfies the general definition in MAgent [01_foundations.tex, lines 8–43](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:8>), without introducing a second structure group. The doubled content representation is $h\mapsto\operatorname{diag}(h,h)$. The result noise $Q_a\succ0$ and cue covariance $L_a\succ0$ make the joint component covariance positive definite, and the Gaussian record kernel is normalized. Thus $r\mathcal G\ell_C$ is exactly the joint law declared in `crm:eq:model`.

The trivial action on labels means that changing a content frame does not rename which component is under consideration. It does not make the model sector predictively inert: different components have different evaluators. The posterior weights are therefore actual beliefs over models in the sense of `def:generative-model`. Their uncertainty concerns the selected component, not an unknown bank parameter tensor. The tensors of the bank are fixed constitutive data during this conditioning step.

The result projection $(x,y,a)\mapsto(y,a)$ is an equivariant measurable map for the same group action. Consequently $w_a\mathcal N(dy;m_{Y,a},P_{Y,a})$ is also a legitimate joint statistical agent law on $V\times\{1,\ldots,M\}$. That statement concerns the exact label/result law. Returning a single Gaussian for $Y$, even while separately carrying $w$, does not by itself retain the component-conditional laws or their dependence on the label. MAgent explicitly permits that dependence in its general definition and states that its developed dynamical constructions impose the narrower product restriction [01_foundations.tex:43](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:43>).

## Passive frames, rebasing, and the fixed bank

Let $U_i\in\mathrm{GL}(K)$ carry reference vectors to token $i$'s local coordinates, as in `crm:eq:reference` and MAgent `eq:reference-laws`. The local doubled-content frame is $\operatorname{diag}(U_i,U_i)$. The token computes from

\[
\xi_i=U_i^{-1}\mu_i,\qquad C_i=U_i^{-1}\Sigma_iU_i^{-T}.
\]

Under the passive change $U_i'=g_iU_i$, $\mu_i'=g_i\mu_i$, $\Sigma_i'=g_i\Sigma_i g_i^T$, both inputs remain unchanged. Therefore the reference posterior, its label weights, and its projected output remain unchanged, while the local result transforms by $g_i$. This is exactly the dressing construction described in MAgent [01_foundations.tex, lines 76–85](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:76>) and [02_interactions.tex, lines 55–72](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/02_interactions.tex:55>). It establishes consistent local re-expression for arbitrary bank parameters.

A common change of reference is a different operation. Use the cue/result convention $\xi'=h\xi$, $C'=hCh^T$, corresponding to $U_i'=U_i h^{-1}$ if local physical coordinates are held fixed. MAgent's convention $U_i'=U_i h_{\mathrm{MA}}$ consequently has $h_{\mathrm{MA}}=h^{-1}$. Re-express the bank as

\[
\nu_a'=h\nu_a,\quad L_a'=hL_ah^T,\quad
\omega_a'=h\omega_a,\quad F_a'=hF_ah^{-1},\quad
Q_a'=hQ_ah^T,\quad \hat\pi_a'=\hat\pi_a.
\]

Then each component evidence acquires the same density factor $|\det h|^{-1}$, which cancels in its normalized weight. The cue gain becomes $hL_a(L_a+C)^{-1}h^{-1}$, and the exact conditional moments and output moments transform by $h$. These are the transformation laws of the evaluator and record mechanism, including the bank as supplied data. They instantiate MAgent's reference consistency rather than deriving a symmetry of a bank held numerically fixed.

For a fixed bank, an active symmetry must also leave its constitutive data unchanged, possibly after a separately allowed relabeling. Such a symmetry need not exist. For a one-dimensional one-component bank with $\nu=1$, $L=Q=1$, preserving $L$ permits $h=\pm1$, while preserving $\nu$ forces $h=1$. Even though the model can be re-expressed under every nonzero scaling, this particular bank has no nonidentity linear active symmetry.

The monomial result in `crm:lem:monomial` is the group preserving the entire family of diagonal covariance matrices under congruence. It is not the symmetry group of every fixed bank. To see the family statement, for $i\neq j$,

\[
(hDh^T)_{ij}=\sum_k h_{ik}h_{jk}d_k.
\]

Vanishing for every positive diagonal $D$ forces $h_{ik}h_{jk}=0$ for every $k$. Invertibility then gives one nonzero entry per row and column. A general $h\in\mathrm{GL}(K)$ takes diagonal $L_a,Q_a$ outside that reference parametrization. Full covariance banks are closed under arbitrary rebasing. Frame-diagonal output projection has the same distinction: it commutes with monomial rebasing but generally does not commute with mixing the reference axes. None of this changes the already proved local passive covariance with the reference chart held fixed.

The construction above is at one context, or on a patch with the declared common reference. To make it a field over an arbitrary nontrivial principal bundle, the bank must itself be supplied as associated data with these overlap laws, or the bundle must admit a declared reduction/trivialization compatible with the bank. A globally fixed reference frame cannot simply be assumed on every principal bundle. No such global extension is needed for the existing one-context block, and the block alone specifies no base connection, transport along a curve, or curvature.

## What is shared across tokens

At matched record covariance $C$, tokens reading one bank have the same reference prior, evaluator, and likelihood. In local doubled-content coordinates the map between tokens is

\[
\psi_{ji}=\operatorname{diag}(U_jU_i^{-1},U_jU_i^{-1}),
\qquad \phi_{ji}(a)=a.
\]

Choose the record alphabet in the common reference coordinates and write the local likelihood as

\[
\ell_i((x_i,y_i),a;do)=\mathcal N(do;U_i^{-1}x_i,C).
\]

Then $r_j=\phi_{ji\#}r_i$, $\mathcal G_j(\phi_{ji}(a))=\psi_{ji\#}\mathcal G_i(a)$, and $\ell_j(\psi_{ji}k_i,\phi_{ji}(a);\cdot)=\ell_i(k_i,a;\cdot)$. These are exactly the three clauses of MAgent's shared-model definition [01_foundations.tex, lines 185–195](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:185>). Different observed values $\xi_i$ are compatible with one shared model and different posterior beliefs.

The implemented interpretation supplies $C_i$ from each token's uncertainty. If $C_i\neq C_j$, the likelihoods $\mathcal N(do;x,C_i)$ and $\mathcal N(do;x,C_j)$ are different. Equality under the displayed frame map fails, even though the bank is shared. A minimal counterexample uses one-dimensional identity frames and $C_i=1$, $C_j=2$: no transformation of the likelihood mean can make two Gaussian laws with those different variances identical on the same record alphabet. Thus the literal species conclusion needs the covariance condition. What holds without it is a shared conditional mechanism $(x,C)\mapsto\mathcal N(do;x,C)$ indexed by supplied side information. Treating that conditional family as the definition of sharing is a reasonable explicit extension; it is not the fixed-kernel equality already written in `def:shared-model`.

The same care applies to the carried prior. Reading

\[
r_i^{\ell+1}(b)\propto\hat\pi_b(w_i^\ell T)_b^{\vartheta}
\]

as a different reference law for each token generally violates the shared-reference clause under the identity label map. The note's Dirichlet-record construction supplies another interpretation: keep the same baseline $\hat\pi$, and let tokens observe different predicted-regime records under the same declared likelihood. On that interpretation, the posterior differences come from information rather than different baseline models. This is an exact one-step normalized construction; the statistical status of reusing a computed belief as a further record remains the separate issue stated in `crm:prop:dirichlet` and its following paragraph.

Carrying $w$ is therefore a concrete extension of the earlier same-state note's proposed persistent model belief. A transition table $T$ is a Markov channel on the finite model presentations. It is not a $G$-valued model transport: with the trivial label action, the same content gauge transformation acts trivially on $w$ and $T$, and a stochastic table need not be invertible. Learning or carrying a categorical model belief does not require a nontrivial continuous model-sector representation.

## Regression is a response kernel, not a gauge dictionary

For a component $a$, the map from a cue sample to a result is the normalized kernel

\[
K_a(x,dy)=\mathcal N(dy;\omega_a+F_a(x-\nu_a),Q_a).
\]

This is admitted by MAgent's general normalized-channel definition [02_interactions.tex, lines 26–37](</C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/02_interactions.tex:26>). Its local regression is $F_{i,a}=U_iF_aU_i^{-1}$, its local affine offset is $U_i(\omega_a-F_a\nu_a)$, and its local noise covariance is $U_iQ_aU_i^T$. Passive covariance follows by conjugation and congruence.

The matrix $F_a$ is not thereby a group element or an inter-agent dictionary. It may be singular, and the kernel adds positive noise. In one dimension $F_a=0$, $Q_a>0$ is an admissible component that ignores its cue and samples a result; no invertible transport can equal that kernel. Also $F_a=-1$ is permitted by the cue/result model and lies outside the real scalar exponential image. A low-rank departure $I+UV^T$ can destroy a direction, whereas $\exp(OT)$ is always invertible, as MAgent's `prop:int-rank` states. The power to reverse, rotate, or discard a cue direction is therefore compatible with the general channel framework without imposing link or connection semantics on the regression.

## Fisher geometry belongs to the retained law

Consider a smooth parameterized family of exact posterior label/result laws

\[
\zeta_\theta(a,dy)=w_a(\theta)\mathcal N(dy;m_a(\theta),P_a(\theta)),
\]

with $w_a>0$, $P_a\succ0$. On any local regular chart, the score splits into the label score and the centered conditional Gaussian score. Their cross term has zero conditional expectation. For tangent directions $v,u$, MAgent's `geo:joint-fisher` therefore gives the exact expression

\[
g^F_\zeta(v,u)=\sum_a\frac{(d_vw_a)(d_uw_a)}{w_a}
+\sum_a w_a\left[(d_vm_a)^TP_a^{-1}(d_um_a)
+\frac12\operatorname{tr}(P_a^{-1}d_vP_a P_a^{-1}d_uP_a)\right].
\]

The same formula holds for the doubled cue/result content with its $2K$-dimensional component conditional covariance. Null parameter directions, if any, must be removed before calling the tensor a metric. These conditions are met locally by positive finite Gaussian mixtures with the label retained; the formula is not the sum of the Fisher metric of a Gaussian output carrier and the categorical metric of $w$.

For an especially direct connection to `crm:thm:response`, hold the bank and $C$ fixed and use the observed cue $\xi$ as a parameter of the exact posterior of $(A,X,Y)$. Its directional score is

\[
\ell_v=v^TC^{-1}(X-\mathbb E[X\mid\xi]),
\]

so its Fisher tensor in record coordinates is

\[
I_\xi=C^{-1}\operatorname{Cov}(X\mid\xi)C^{-1}.
\]

By contrast, the result-mean response is $D_\xi\mathbb E[Y\mid\xi]=\operatorname{Cov}(Y,X\mid\xi)C^{-1}$. The former measures distinguishability of the complete posterior family; the latter measures sensitivity of one posterior expectation. A negative or rotating mean response creates no conflict with positive Fisher geometry because these are different tensors.

Marginalizing $X$ or the component label is a fixed measurable operation. Its marginal score is the conditional expectation of the full score, so the difference between the full and marginal Fisher tensors is the expectation of a conditional score covariance and is positive semidefinite. Moment projection to a Gaussian is a different, nonlinear operation on the law space. The fixed-channel Fisher contraction statement cannot simply be invoked for it.

A scalar example exposes the possible loss. Use identical cue components, equal weights, $F_1=F_2=0$, result means $\omega_1=t$, $\omega_2=-t$, and result noises $Q_1=Q_2=1-t^2$ for $0<t<1$. For every record, the result mixture is

\[
p_t=\tfrac12\mathcal N(t,1-t^2)+\tfrac12\mathcal N(-t,1-t^2).
\]

Its Gaussian moment carrier is the constant $\mathcal N(0,1)$, hence has zero Fisher length along $t$. The label/result joint has Fisher coefficient

\[
g^F_\zeta(\partial_t,\partial_t)
=\frac1{1-t^2}+\frac{2t^2}{(1-t^2)^2}
=\frac{1+t^2}{(1-t^2)^2}>0.
\]

Even the result marginal varies: its fourth moment is $3-2t^4$. This example is a family of admissible banks, not a claim about a measured training trajectory. It proves that equality of moment carriers does not establish equality of joint laws or their statistical geometry.

MAgent's contextual pullback further requires a base, a smooth section, and a principal connection, with $h^\omega(v,u)=g^F(D^\omega\zeta v,D^\omega\zeta u)$. The memory's algebra supplies none of those data automatically. It supplies a concrete posterior family on which that construction could be evaluated if the data are declared. Using only the Gaussian carrier instead would define another geometry. The exact conditioning step and backpropagation of a task loss also do not amount to the Fisher gradient flow used in MAgent's dynamical theorems.

## Result and remaining obligation

The exact geometric match is a finite, correlated model-presentation/content agent with a normalized evaluator and internal record kernel, in the same frame convention and one-group architecture as MAgent. The cue/result regression enriches what an evaluator can predict; carrying the label posterior supplies a persistent finite model-belief state. Their useful architectural role is compatible with that reading.

The nearest unresolved connection is an explicit retained-state and conditioning contract for repeated blocks: specify whether component conditionals or only projected moments and labels constitute the agent, how the record covariance enters the shared model, and which later operations must preserve the discarded distinctions. That contract would determine the appropriate joint Fisher family and any claimed species or inheritance statement. No global connection, natural-gradient learning rule, or formation theorem follows merely from the existing block.

This investigation used analytical derivations and source reads only. No executable calculation, training, GPU work, Lean work, or manuscript edit was performed. Only this assigned report fragment was written.
