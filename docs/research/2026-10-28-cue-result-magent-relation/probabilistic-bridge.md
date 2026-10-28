# The cue/result bank as a concrete MAgent generative model

The cue/result block is an exact instance of the single-agent generative-model definition in MAgent Theory 2.0, followed by a specified approximation to its posterior. Its component label can be the model presentation, its joint cue/result law the evaluator, and the incoming cue the record. The exact posterior uses the general correlated agent law \(s(dm)q(dk\mid m)\). It is generally outside the product family \(s(dm)q(dk)\) used by the theory's developed dynamics. The cue/result construction adds a conditional result variable, which permits signed and rotating responses within Bayesian inference itself.

## Scope and continuity

The target is equality of normalized one-step probability models and posterior laws, for every admitted bank and every record with finite positive evidence. This is stronger than a resemblance between formulas, and weaker than an equivalence of complete transformer and multi-agent dynamics. The supplied assumptions are the finite positive component prior, positive Gaussian cue/result/record covariances, arbitrary component regressions, and fixed input record covariance of the working note. The general-law construction below makes explicit which parts do not require Gaussianity. No training result is derived or reproduced here; the user's observation that the replacement performs well is separate from these mathematical identities.

The earlier [MLP-slot investigation](<C:/Users/chris and christine/Desktop/VFE_3.0/explainer_video/research/mlp_as_generative_models.md>) already identified the same-state bank with the theory's reference, evaluator, and record likelihood. The extension below preserves that identification and adds the cue/result conditional law, the resulting correlated recognition law, and the exact distinction between posterior conditioning and its Gaussian carrier. Old implementation and configuration statements in that earlier note are not used as current facts.

The defining sources are [MAgent foundations, lines 35–45](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:35>) for joint recognition and the product restriction; [lines 170–197](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:170>) for the model, free energy, and shared-model definitions; [MAgent dynamics, lines 9–24](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex:9>) for the conditional-KL split; and [cue/result model and posterior, lines 76–149](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/cue_result_memory.tex:76>). The live authoring conventions were read. Manuscripts remain unedited.

## A general-law construction

Let \(\mathsf M,\mathsf X,\mathsf Y,\mathsf O\) be standard Borel spaces. Let \(r(dm)\) be a proper law of model presentations, \(\mu_m(dx)\) a normalized cue kernel, \(K_m(x,dy)\) a normalized conditional result kernel, and \(\mathcal L(x,do)\) a normalized record kernel. Assume joint measurability in every displayed conditioning variable. Define

\[
P(dm,dx,dy,do)
=r(dm)\mu_m(dx)K_m(x,dy)\mathcal L(x,do).
\tag{1}
\]

Successively integrating the kernels gives total mass one. The declaration includes \(Y\perp O\mid(X,m)\). Neither a Gaussian family nor a finite-dimensional represented space is needed.

In the notation of Definition \(\mathrm{def:generative\text{-}model}\), choose

\[
\mathsf K=\mathsf X\times\mathsf Y,\qquad
G(m,d(x,y))=\mu_m(dx)K_m(x,dy),\qquad
\ell((x,y),m;do)=\mathcal L(x,do).
\tag{2}
\]

Then (1) is precisely \(r(dm)G(m,dk)\ell(k,m;do)\). The evaluator is a model of correlated cue and result contents; the label \(m\) selects that evaluator.

Suppose \(\mathcal L(x,do)=l(o\mid x)\lambda(do)\) for a declared reference measure, and fix a record \(o\) with

\[
0<Z_m(o):=\int l(o\mid x)\mu_m(dx)<\infty
\quad\text{for }r\text{-almost every }m,
\qquad
0<Z(o):=\int Z_m(o)r(dm)<\infty.
\tag{3}
\]

The conditional law and presentation posterior are

\[
\Pi_m^o(dx,dy)
=\frac{l(o\mid x)\mu_m(dx)K_m(x,dy)}{Z_m(o)},
\qquad
w^o(dm)=\frac{Z_m(o)}{Z(o)}r(dm).
\tag{4}
\]

Thus the exact agent posterior is

\[
\zeta^o(dm,dx,dy)=w^o(dm)\Pi_m^o(dx,dy).
\tag{5}
\]

This is the theory's general disintegration \(s(dm)q(dk\mid m)\). The result marginal is the model average \(\int \Pi_m^o(\mathsf X,dy)w^o(dm)\). Because each result kernel integrates to one, the presentation evidence \(Z_m(o)\) depends on the cue law and record kernel, not on the conditional result law. In the Gaussian bank, the regression and result noise affect the result posterior but do not enter the component evidence.

One can also retain the original result space as the represented-content space. Disintegrate \(\mu_m(dx)K_m(x,dy)\) into \(G_Y(m,dy)R_m(y,dx)\). Standard Borel disintegration supplies such versions, with arbitrary definitions on the appropriate null sets. Define

\[
\ell_Y(y,m;do)=\int\mathcal L(x,do)R_m(y,dx).
\tag{6}
\]

This is normalized and gives the same marginal law \(P(dm,dy,do)=r(dm)G_Y(m,dy)\ell_Y(y,m;do)\). Therefore the cue can either be an explicit latent content or be integrated into a presentation-dependent likelihood. The latter dependence is permitted by the MAgent definition.

## Exact Gaussian identification

For the cue/result bank take \(\mathsf M=\{1,\ldots,M\}\), \(\mathsf X=\mathsf Y=\mathsf O=\mathbb R^K\), counting measure on the labels, and Lebesgue measure on the continuous variables. Set

\[
\begin{aligned}
r(a)&=\hat\pi_a,\\
\mu_a(dx)&=\mathcal N(dx;\nu_a,L_a),\\
K_a(x,dy)&=\mathcal N(dy;\omega_a+F_a(x-\nu_a),Q_a),\\
\mathcal L_C(x,do)&=\mathcal N(do;x,C).
\end{aligned}
\tag{7}
\]

The covariance \(C\succ0\) is conditioned-on side information. All component weights are positive, \(L_a,Q_a\succ0\), and every \(F_a\) is allowed. These assumptions give positive finite evidence for every \(\xi\in\mathbb R^K\), and all moments and entropy terms used below are finite for the displayed Gaussian and finite-mixture laws. The evaluator is the \(2K\)-dimensional Gaussian

\[
G_a=\mathcal N\left(
\begin{pmatrix}\nu_a\\\omega_a\end{pmatrix},
\begin{pmatrix}
L_a&L_aF_a^T\\
F_aL_a&F_aL_aF_a^T+Q_a
\end{pmatrix}
\right).
\tag{8}
\]

Its covariance is positive definite because it is the covariance of
\((X,Y)=(\nu_a+\epsilon_X,\omega_a+F_a\epsilon_X+\epsilon_Y)\) with independent positive-definite noise covariances \(L_a,Q_a\). Equivalently, the block Schur complement is \(Q_a\succ0\).

Equations (3)–(5) give

\[
Z_a(\xi)=\mathcal N(\xi;\nu_a,L_a+C),\qquad
s(a)=w_a=\frac{\hat\pi_aZ_a(\xi)}{\sum_b\hat\pi_bZ_b(\xi)}.
\tag{9}
\]

The exact content conditional is Gaussian:

\[
\begin{aligned}
m_{X,a}&=\nu_a+L_a(L_a+C)^{-1}(\xi-\nu_a),\\
P_{X,a}&=(C^{-1}+L_a^{-1})^{-1},\\
m_{Y,a}&=\omega_a+F_a(m_{X,a}-\nu_a),\\
P_{Y,a}&=Q_a+F_aP_{X,a}F_a^T,\\
\operatorname{Cov}(X,Y\mid \xi,a)&=P_{X,a}F_a^T.
\end{aligned}
\tag{10}
\]

These are exactly the cue/result note's equations \(\mathrm{crm:eq:weights}\) through \(\mathrm{crm:eq:mixture}\), with the joint-law typing supplied by MAgent.

For the result-only embedding (6), put

\[
V_a=F_aL_aF_a^T+Q_a,\qquad
B_a=L_aF_a^TV_a^{-1},\qquad
S_a=L_a-L_aF_a^TV_a^{-1}F_aL_a.
\]

The explicit evaluator and likelihood are

\[
G_Y(a,dy)=\mathcal N(dy;\omega_a,V_a),
\qquad
\ell_{Y,C}(y,a;do)
=\mathcal N(do;\nu_a+B_a(y-\omega_a),C+S_a).
\tag{11}
\]

Here \(S_a=\operatorname{Cov}(X\mid Y,A=a)\succ0\), and \(C+S_a\succ0\). To verify (11), condition the Gaussian evaluator (8) on \(Y\) and add the independent record noise. Integrating \(Y\) recovers \(Z_a(\xi)\), and conditioning on the record recovers the result posterior in (10). This gives an exact MAgent realization with \(K\)-dimensional represented result contents, without requiring that the working state store the cue as another explicit content coordinate.

The dictionary of objects is consequently:

| Cue/result object | MAgent role in this construction |
| --- | --- |
| Component label \(A\) | Model presentation \(m\) |
| Shared prior \(\hat\pi\) | Reference law \(r\) over presentations |
| Component cue/result distribution | Evaluator \(G(m,dk)\) |
| Incoming \((\xi,C)\) | Record value and declared likelihood side information |
| Component posterior \(w\) | Belief \(s\) over model presentations |
| Component-conditional posterior | Belief \(q(dk\mid m)\) |
| Exact result mixture | Marginal belief about the inferred result |
| Returned Gaussian moments | Specified compressed carrier of that marginal |

The model belief is over component presentations. It is not a posterior over the bank parameters \((\nu,L,\omega,F,Q,\hat\pi)\), which are fixed during this inference step. Nor is \(w\) automatically one of the theory's peer-selection rows \(\beta\) or \(\gamma\): those choose a source agent, whereas \(w\) selects a model presentation in this embedding. A source-label reinterpretation is another possible model and needs its own explicit typing.

## Exact free-energy decomposition

The relation to the theory's variational free energy is also exact before moment projection. For a candidate recognition law

\[
Q(dm,dk)=s(dm)q_m(dk),
\]

assume the absolute continuity and finiteness conditions that make the following relative entropies and likelihood expectations finite. The free energy at the fixed record is

\[
\begin{aligned}
\mathcal F(Q;o)
&=\mathbb E_Q\left[\log\frac{dQ}{d(rG)}(m,k)-\log l(o\mid k,m)\right]\\
&=\operatorname{KL}(s\Vert r)
 +\int\left[
\operatorname{KL}(q_m\Vert G_m)-\mathbb E_{q_m}\log l(o\mid k,m)
\right]s(dm).
\end{aligned}
\tag{12}
\]

Writing

\[
f_m(q_m;o)
=\operatorname{KL}(q_m\Vert G_m)-\mathbb E_{q_m}\log l(o\mid k,m)
=\operatorname{KL}(q_m\Vert\Pi_m^o)-\log Z_m(o)
\tag{13}
\]

and using \(dw^o/dr=Z_m/Z\) gives

\[
\boxed{
\mathcal F(Q;o)
=\operatorname{KL}(s\Vert w^o)
 +\int\operatorname{KL}(q_m\Vert\Pi_m^o)s(dm)-\log Z(o).
}
\tag{14}
\]

This proves the global minimizer \(Q=\zeta^o\) for the unrestricted single-agent recognition law, up to null sets. In the finite-component case, fixing the content conditionals and varying the normalized label vector gives

\[
s_a=\frac{\hat\pi_a e^{-f_a(q_a;\xi)}}
{\sum_b\hat\pi_b e^{-f_b(q_b;\xi)}}.
\tag{15}
\]

At the exact component posteriors \(f_a=-\log Z_a\), equation (15) becomes (9). Thus the exact gate does have a Gibbs variational form, with the optimized component free energies, equivalently negative log evidences, as its costs. Those costs are not the divergences between the incoming Gaussian law and the cue templates used by a product approximation.

For the Gaussian model the natural structured variational family is

\[
Q(a,dx,dy)=s_a q_a(dx)K_a(x,dy).
\tag{16}
\]

The result conditional is already its exact conditional law and cancels from the density ratio in (12). This gives

\[
\mathcal F
=\operatorname{KL}(s\Vert\hat\pi)
 +\sum_as_a\left[
\operatorname{KL}(q_a\Vert\mu_a)-\mathbb E_{q_a}\log\mathcal N(\xi;X,C)
\right].
\tag{17}
\]

The Gaussian choices \(q_a=\mathcal N(m_{X,a},P_{X,a})\) attain the exact posterior, so no approximation is introduced by this structured family for one Gaussian bank step.

## What the product restriction changes

The theoretical dynamics explicitly imposes \(\zeta=q\otimes s\), and its local term is \(\operatorname{KL}(s\Vert r)+\mathbb E_s\operatorname{KL}(q\Vert G_m)\). Its product restriction and the generated dependence are discussed in \(\mathrm{rem:dyn\text{-}product}\), [dynamics lines 9–24](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex:9>). The exact cue/result inference uses component-dependent conditionals instead.

For the structured approximation used in the cue/result note, replace (16) by

\[
Q_{\mathrm{MF}}(a,dx,dy)=s_a q(dx)K_a(x,dy).
\tag{18}
\]

This is a product over \((A,X)\), while the result conditional is retained. It is not a product over \((A,(X,Y))\). That distinction matters when relating it to the general MAgent presentation/content split.

Because the record likelihood in (7) is independent of \(a\) at fixed \(X\), its expectation is common to the label alternatives. Minimization of (17) with a single shared \(q\) therefore gives

\[
s_a\propto\hat\pi_a\exp[-\operatorname{KL}(q\Vert\mu_a)]
\propto\hat\pi_a\exp[\mathbb E_q\log\mu_a(X)].
\tag{19}
\]

Varying \(q\) at fixed \(s\), including its entropy and enforcing normalization, gives

\[
q(x)\propto
\mathcal N(\xi;x,C)\exp\left(\sum_as_a\log\mu_a(x)\right).
\tag{20}
\]

For Gaussian \(\mu_a\), this has

\[
\Sigma_X^{-1}=C^{-1}+\sum_as_aL_a^{-1},\qquad
\Sigma_X^{-1}m_X=C^{-1}\xi+\sum_as_aL_a^{-1}\nu_a.
\tag{21}
\]

These are exactly the note's proposed, unimplemented structured mean-field updates at [lines 399–408](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/cue_result_memory.tex:399>). They have the same expected-log mechanism as MAgent's product-family local update. They do not reproduce the exact bank by a cheaper evaluation. If the augmented content space \((X,Y)\) were fully factorized from \(A\), or if the result-only embedding (11) were product-factorized, the variational family and resulting updates would again be different.

An exact gap identity makes the difference in routing explicit. Define the normalized record law on cue space

\[
\rho_\xi(dx)=\mathcal N(dx;\xi,C),
\]

so Gaussian symmetry gives \(Z_a(\xi)=\mathbb E_{\rho_\xi}\mu_a(X)\). Let \(q_a^*=\rho_\xi\mu_a/Z_a\), the exact component cue posterior. Then

\[
\log Z_a(\xi)
=\mathbb E_{\rho_\xi}\log\mu_a(X)
 +\operatorname{KL}(\rho_\xi\Vert q_a^*).
\tag{22}
\]

Indeed, \(\log(d\rho_\xi/dq_a^*)=\log Z_a-\log\mu_a\), and integration proves the formula. Consequently, even the first mean-field label update initialized at \(q=\rho_\xi\) differs from the exact evidence gate by a component-dependent nonnegative correction. The normalized weights coincide only if these corrections are the same across components. No universal ordering of the two label entropies follows from their being nonnegative; the correction differences may favor different components. This is the same logical distinction between log-integrated likelihood and integrated log likelihood addressed by \(\mathrm{prop:dyn\text{-}jensen}\), [dynamics lines 270–310](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex:270>).

The closest exact correspondence is therefore the general single-agent model of Chapter 1. With only self-source candidates, the per-agent mixture-source model of Chapter 3 reduces to the same normalized generative joint. Its displayed product-family free energy and Fisher flow remain restrictions on recognition and dynamics, so equality of the generative model does not equate their updates to exact bank conditioning. Reciprocal source-role forces, a common population potential, and local bank learning do not arise from the one-step identity (14).

## What cue/result inference adds

Same-state retrieval asks for \(X\) after receiving a noisy record of \(X\). Cue/result retrieval asks for another correlated variable \(Y\). For the Gaussian record kernel and fixed \(C\), differentiating the normalized posterior gives

\[
D_\xi\mathbb E[Y\mid\xi]
=\operatorname{Cov}(Y,X\mid\xi)C^{-1}.
\tag{23}
\]

The finite Gaussian mixture justifies differentiation because its component moments are smooth and finite; the working note gives the exponential-family derivation in \(\mathrm{crm:thm:response}\), [lines 193–215](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/cue_result_memory.tex:193>). Equation (23) also holds beyond Gaussian component priors whenever the additive Gaussian observation model and differentiation/integrability hypotheses hold.

For \(Y=X\), the covariance in (23) is symmetric positive definite and the whitened response is positive definite. For a distinct result variable it is a cross-covariance, which may be signed or nonsymmetric. Already with one Gaussian component,

\[
D_\xi\mathbb E[Y\mid\xi]=F L(L+C)^{-1}.
\tag{24}
\]

At fixed \(L,C\succ0\), any prescribed dense linear response \(H\) is obtained exactly by \(F=H(L+C)L^{-1}\); choose the result offset to obtain the desired affine intercept. This statement uses unrestricted dense regressions. It does not apply to every fixed-rank departure \(F=I+UV^T\).

This is an explicit statistical mechanism for private computation: evidence about a cue changes a prediction about another content. A rotation of the output mean is not a forbidden Euclidean gradient on a Lie algebra or a Fisher descent step on the cue variable. It is a conditional prediction under an admissible normalized evaluator. The arbitrary dense regression is compatible with a positive-definite joint law because \(Q\succ0\) supplies its Schur complement. The distinction is useful for MAgent: a model evaluator can transform inferred contents through a conditional kernel while every probability law remains normalized, without requiring the forward map itself to be a gradient map.

## The returned carrier and its exact information loss

The block computes the result marginal

\[
p_Y^\xi=\sum_aw_a\mathcal N(m_{Y,a},P_{Y,a})
\]

and returns the Gaussian \(q_G=\mathcal N(\bar m,S)\) with its first two moments, or the diagonal version \(q_D=\mathcal N(\bar m,\operatorname{diag}S)\). The source proves

\[
q_G=\arg\min_{\text{Gaussian }q}\operatorname{KL}(p_Y^\xi\Vert q),
\qquad
q_D=\arg\min_{\text{diagonal Gaussian }q}\operatorname{KL}(p_Y^\xi\Vert q).
\tag{25}
\]

This is the inclusive-divergence projection at [cue/result lines 168–177](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/cue_result_memory.tex:168>). MAgent's variational free energy instead minimizes a divergence with recognition in the first slot, \(\operatorname{KL}(q\Vert p)\). Equation (25) therefore does not make the returned Gaussian the restricted-family minimizer of the same local free energy.

The distinction can be checked without a numerical experiment. Let \(p_a=\tfrac12\mathcal N(-a,1)+\tfrac12\mathcal N(a,1)\). Its moment projection is \(q_a=\mathcal N(0,a^2+1)\). A single component obeys \(\operatorname{KL}(\mathcal N(a,1)\Vert p_a)\leq\log2\), because \(p_a\geq\tfrac12\mathcal N(a,1)\). On the other hand,

\[
p_a(y)\leq(2\pi)^{-1/2}
\exp[-(|y|-a)^2/2],
\]

which implies

\[
\operatorname{KL}(q_a\Vert p_a)
\geq a^2-a\sqrt{\frac{2(a^2+1)}{\pi}}
-\frac12\log(a^2+1)\longrightarrow\infty
\quad(a\to\infty).
\tag{26}
\]

For all sufficiently separated components, the moment projection cannot minimize the reverse divergence. Such mixtures occur within the cue/result family by taking equal cue laws, \(F_a=0\), result offsets \(\pm a\), and unit result noise.

If one retains only the exact label posterior \(w\) and a Gaussian result carrier and interprets them as the product \(w\otimes q_G\), the precise inclusive loss relative to the exact label-result joint \(\Pi_{A,Y}^\xi\) is

\[
\operatorname{KL}(\Pi_{A,Y}^\xi\Vert w\otimes q_G)
=I_{\Pi^\xi}(A;Y)
 +\operatorname{KL}(p_Y^\xi\Vert q_G).
\tag{27}
\]

The proof inserts the product \(w\otimes p_Y^\xi\) into the logarithmic density ratio and integrates. Thus model-content dependence and non-Gaussian marginal information are separate discarded quantities. Storing both marginals does not retain their joint dependence. When the output is diagonal, comparison of the two Gaussian log densities adds the exact further cost

\[
\operatorname{KL}(p_Y^\xi\Vert q_D)
-\operatorname{KL}(p_Y^\xi\Vert q_G)
=\frac12\log\frac{\prod_kS_{kk}}{\det S}.
\tag{28}
\]

All terms are finite here because the mixture is finite with strictly positive component covariances. These identities quantify one-step statistical loss. They do not establish whether the discarded information matters to a particular downstream task or whether an inherited operational interface is closed.

## Supported conclusion and next obligation

The cue/result bank supplies a concrete normalized evaluator and record model for MAgent, with exact Bayesian selection among model presentations and exact conditional prediction before Gaussian projection. Its practical expressiveness comes from allowing cue and result to be distinct correlated variables; it does not require abandoning a probabilistic account of private computation. The general-law formulation (1) shows that this interpretation is not tied to Gaussian closure.

What remains before importing this as one of the theory's dynamical mechanisms is the choice of retained correlated state and a declared composition of communication, record generation, conditioning, and projection through depth. Equality with the existing product-family Fisher flow is not an unresolved algebraic identity; those are different updates. A useful next theoretical construction would retain \(s(a)q(k\mid a)\), specify its communication interface, and state explicitly when and why its projection is performed. Local learning of the bank would additionally need a predictive observation model delivering task evidence to its parameters.

Only analytical derivations and source reads were performed for this fragment. No scripts, symbolic or numerical experiments, GPU or Lean work, training, code changes, or manuscript edits were run. The formulas above provide the checkable evidence for the probabilistic correspondence; performance claims remain the user's reported observation.
