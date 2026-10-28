# Coupled reciprocal flow with dictionary uncertainty

This note derives the complete restricted Gaussian Fisher flow of the reciprocal potential in the live manuscript, including changing content covariances, dictionary covariances, and source rows. The general-law variation comes first. The two row conventions have the same interior stationary equations but different trajectories. In the Gaussian restriction, proper Gaussian self references and a fixed positive-definite dictionary prior prevent escape to infinity or covariance singularity. With positive fixed source priors and positive temperatures, both row conventions have global forward solutions from every interior initial state.

The existing manuscript already establishes Gibbs reduction, the envelope identity, Gaussian Fisher blocks, and the equality of full/reduced negative and zero Hessian indices. The additional work here is their explicit application to the coupled Item 2 variables, the complete covariance equations, and a compactness argument for this Gaussian dictionary family. The negative covariance term in the reduced Hessian is ordinary finite log-sum-exp differentiation. It is not a new general theorem about attention.

## Sources and exact scope

The definitions used here are `def:dyn-potential`, `prop:dyn-gibbs`, `eq:dyn-envelope`, `prop:dyn-morse`, and `def:dyn-restricted-flow` in `Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex`, and `it2:model`, `it2:potential`, `it2:potential-forces`, and `it2:certainty` in `chapters/18_link_dynamics.tex`. The open target is `it2:open-reciprocal`. The live `AUTHORING.md` and the wiki's `Replicator dynamics.md` supply the objective and scope conventions. All were consulted read-only. The prior source already identifies the restricted Gaussian flow as distinct from the unrestricted law-space flow, which generally leaves the Gaussian family.

All data held fixed in the variation are declared below. In particular, the operating contents, codes, noise, dictionary prior, source priors, and temperatures do not learn. The graph is finite and directed; opposite attending edges remain independent. The note analyzes one content sector. A second sector with fixed normalized data can be added at the general variational level, but it is not silently included in the Gaussian formulas. There are no persistent comparison-record likelihoods in this reciprocal potential. No formation, contextual, nonlinear-group, or full law-space closure claim is made.

## General-law potential and its variation

For each agent $i$, let $q_i$ be a probability law on a standard Borel content space and let $p_i$ be a fixed proper reference. For each directed edge $e:j\to i$, let $r_e$ be a probability law on a standard Borel dictionary space. The dictionary recognition law is restricted to the product $r=\bigotimes_e r_e$; its fixed proper reference $\pi_\omega$ may correlate edges. Write $\mathcal I_i$ for the finite nonempty source-edge set in row $i$, with fixed positive normalized prior $\pi_i$, temperature $T_i>0$, and row $\beta_i\in\Delta_{\mathcal I_i}^{\circ}$. Rows for agents without candidate edges may simply be omitted.

Fixed normalized message kernels define conditional receiver and source laws on a common message space,

\[
A_e(\omega)=q_iK_e^{\mathrm{rec}}(\omega),\qquad
B_e=q_jK_e^{\mathrm{src}},\qquad
\Delta_e(\omega)=D_{\mathrm{KL}}(A_e(\omega)\Vert B_e),
\]

and the edge cost is

\[
C_e(q,r)=\int\Delta_e(\omega)r_e(d\omega).
\tag{1}
\]

The expectation in (1) is outside the divergence. Replacing it by the divergence of an averaged receiver law defines a different objective. Require $q_i\ll p_i$, $r\ll\pi_\omega$, and $A_e(\omega)\ll B_e$ for $r_e$-almost every $\omega$, and assume the displayed relative entropies and expected costs are finite. Whenever derivatives are used, assume regular dominated charts, positive message densities on their common support, and integrable envelopes sufficient to differentiate the integrals twice. These are additional regularity hypotheses; standard Borel measurability alone does not supply them.

The potential and its matter part are

\[
\begin{aligned}
F_0(q,r)&=\sum_iD_{\mathrm{KL}}(q_i\Vert p_i)+D_{\mathrm{KL}}(r\Vert\pi_\omega),\\
F(q,r,\beta)&=F_0(q,r)+\sum_i\left[\sum_{e\in\mathcal I_i}\beta_e C_e+T_iD_{\mathrm{KL}}(\beta_i\Vert\pi_i)\right].
\end{aligned}
\tag{2}
\]

The product law $\bigotimes_iq_i\otimes\bigotimes_er_e\otimes\bigotimes_i\operatorname{Cat}(\beta_i)$ supplies the product Fisher metric. This is not a claim that (2) is a variational free energy of a fixed generative model over the agents' represented contents. The peer divergences are a constitutive reciprocal interaction between laws.

For mass-preserving variations of message laws,

\[
d\Delta_e=\int\log(A_e/B_e)dA_e-\int(A_e/B_e)dB_e,
\]

and therefore

\[
dC_e=\int\Delta_e(\omega)dr_e(\omega)
+\int r_e(d\omega)\left[\int\log(A_e/B_e)dA_e-\int(A_e/B_e)dB_e\right].
\tag{3}
\]

At fixed kernels the receiver variation is $dA_e=(dq_i)K_e^{\mathrm{rec}}(\omega)$ and the source variation is $dB_e=(dq_j)K_e^{\mathrm{src}}$. Thus the receiver contributes the pullback of $\log(A_e/B_e)$, while the source contributes the negative pullback of $A_e/B_e$. An agent receives every contribution from every role in which it appears. If any kernel were law-dependent, its derivative would be an additional term in (3); it is excluded here by the fixed-data hypothesis.

For the dominated calculation choose fixed content measures $\nu_i$ and dictionary measures $\lambda_e$. All content densities are relative to $\nu_i$, and the joint dictionary reference density is relative to $\bigotimes_e\lambda_e$. In addition to finite relative entropy, the expanded log-density expectations in the following expression must be integrable; finite KL alone need not imply separate integrability. The dictionary block first variation of (2), up to a function constant in its sample $\omega_e$, is

\[
\frac{\delta F}{\delta r_e}(\omega_e)
=\log r_e(\omega_e)-\mathbb E_{r_{-e}}\log\pi_\omega(\omega_e,\omega_{-e})
+\beta_e\Delta_e(\omega_e).
\tag{4}
\]

The second term retains the correlated loop prior. Equation (4) is the exact product-recognition variation, not a replacement of the prior by independent marginals. In a regular parametric family, every block velocity is minus its inverse Fisher tensor applied to the corresponding derivative of (2).

For fixed $q,r$, the minimizing row and reduced potential are

\[
Z_i=\sum_{e\in\mathcal I_i}\pi_e e^{-C_e/T_i},\qquad
\beta_e^*=\frac{\pi_e e^{-C_e/T_i}}{Z_i},\qquad
F_{\mathrm{red}}=F_0-\sum_iT_i\log Z_i.
\tag{5}
\]

The exact row gap and differential are

\[
F=F_{\mathrm{red}}+\sum_iT_iD_{\mathrm{KL}}(\beta_i\Vert\beta_i^*),\qquad
dF_{\mathrm{red}}=dF_0+\sum_e\beta_e^*dC_e.
\tag{6}
\]

These are the manuscript's existing Gibbs and envelope identities with the expected dictionary cost as their argument. A change in dictionary recognition enters through the first term of (3); it must not be omitted from $dC_e$.

## Explicit Gaussian objective

Use fixed reference coordinates and identify the dictionary Lie algebra with a finite Euclidean vector space of dimension $d_\omega$. For the manuscript's full matrix dictionary this is the vectorization of $\mathfrak{gl}(K,\mathbb R)$. Let

\[
q_i=\mathcal N(\mu_i,\Sigma_i),\quad
p_i=\mathcal N(m_i,P_i^{-1}),\quad
r_e=\mathcal N(a_e,V_e),
\]

All Gaussian densities in this section use Lebesgue measure, and the finite categorical rows use counting measure. Here $P_i,\Sigma_i,V_e$ are positive definite. A fixed linear map $\kappa_i:\mathbb R^{d_\omega}\to\mathbb R^K$ is the map $\omega\mapsto\omega c_i$ at the fixed operating content in the manuscript. It may be nonzero and need not be injective. Fix each code $\bar T_e:\mathbb R^K\to\mathbb R^{n_e}$ and $R_e\succ0$, where $n_e$ is the message dimension, distinguished here from the dictionary law $r_e$. Full row rank is the manuscript convention, although the formulas and compactness proof below do not require it.

Write

\[
A=\sigma^{-2}I+\lambda B^TWB+\operatorname{diag}(G_e)\succ0,\qquad
b=(G_e\hat\omega_e)_e,\qquad a_0=A^{-1}b,
\tag{7}
\]

with $\sigma>0$, $\lambda\geq0$, $G_e\succeq0$, and fixed positive loop forms $W$. The prior is the normalized joint Gaussian $\pi_\omega=\mathcal N(a_0,A^{-1})$. The result also holds for any fixed $A\succ0,b$; the displayed form records its connection to the anchored loop construction.

For edge $e:j\to i$, define

\[
\begin{aligned}
H_e&=\bar T_e\kappa_i,&
S_{i,e}&=\bar T_e\Sigma_i\bar T_e^T+R_e,&
S_{j,e}&=\bar T_e\Sigma_j\bar T_e^T+R_e,\\
J_e&=S_{j,e}^{-1},&
d_e&=\bar T_e(\mu_i-\mu_j)-H_ea_e,&
Q_e&=S_{i,e}+d_ed_e^T+H_eV_eH_e^T.
\end{aligned}
\tag{8}
\]

The conditional message laws are

\[
A_e(\omega_e)=\mathcal N(\bar T_e\mu_i-H_e\omega_e,S_{i,e}),\qquad
B_e=\mathcal N(\bar T_e\mu_j,S_{j,e}).
\]

Taking the expectation of the Gaussian KL gives exactly

\[
C_e=\frac12\left[\operatorname{tr}(J_eQ_e)-n_e+\log\det S_{j,e}-\log\det S_{i,e}\right].
\tag{9}
\]

The term $H_eV_eH_e^T$ is dictionary uncertainty in the mean of the conditional receiver message. It is not included inside $\log\det S_{i,e}$, because the expectation is taken after computing each conditional message KL.

The Gaussian self term is

\[
D_{\mathrm{KL}}(q_i\Vert p_i)
=\frac12\left[(\mu_i-m_i)^TP_i(\mu_i-m_i)+\operatorname{tr}(P_i\Sigma_i)-\log\det(P_i\Sigma_i)-K\right].
\tag{10}
\]

With $a=(a_e)_e$ and $V=\operatorname{diag}(V_e)$, the dictionary term is

\[
D_{\mathrm{KL}}(r\Vert\pi_\omega)
=\frac12\left[(a-a_0)^TA(a-a_0)+\operatorname{tr}(AV)-\log\det(AV)-d_{\mathrm{tot}}\right].
\tag{11}
\]

Here $\log\det(AV)=\log\det A+\sum_e\log\det V_e$, although $AV$ need not be symmetric. Its determinant is positive because it is similar to a positive-definite matrix. Since $V$ is block diagonal, $\operatorname{tr}(AV)=\sum_e\operatorname{tr}(A_{ee}V_e)$. Off-diagonal prior blocks remain in $a^TAa$. Removing them would change the mean flow; replacing $A_{ee}^{-1}$ by $(A^{-1})_{ee}$ in a factorized covariance update would change the variational family.

## Complete derivatives and Fisher equations

Let $\operatorname{in}(i)$ contain the edges with receiver $i$, and $\operatorname{out}(i)$ the edges with sender $i$. In every formula below, use $\beta$ as an independent variable for the full flow, or substitute $\beta^*(x)$ from (5) for the reduced flow. The envelope identity means that no first-order $d\beta^*$ term is added after that substitution.

Denote the ordinary mean derivatives by $g_{\mu_i},g_{a_e}$, and the symmetric covariance derivatives under the trace pairing by $G_{\Sigma_i},G_{V_e}$. Direct differentiation gives

\[
g_{\mu_i}=P_i(\mu_i-m_i)
+\sum_{e\in\operatorname{in}(i)}\beta_e\bar T_e^TJ_ed_e
-\sum_{e\in\operatorname{out}(i)}\beta_e\bar T_e^TJ_ed_e,
\tag{12}
\]

\[
g_{a_e}=(Aa-b)_e-\beta_eH_e^TJ_ed_e,
\tag{13}
\]

\[
\begin{aligned}
2G_{\Sigma_i}={}&P_i-\Sigma_i^{-1}\\
&+\sum_{e\in\operatorname{in}(i)}\beta_e\bar T_e^T(J_e-S_{i,e}^{-1})\bar T_e\\
&+\sum_{e\in\operatorname{out}(i)}\beta_e\bar T_e^T(J_e-J_eQ_eJ_e)\bar T_e,
\end{aligned}
\tag{14}
\]

and

\[
2G_{V_e}=A_{ee}-V_e^{-1}+\beta_eH_e^TJ_eH_e.
\tag{15}
\]

For the receiver part of (14), only $S_{i,e}$ changes, so the derivative of (9) is $\frac12\operatorname{tr}[(J_e-S_{i,e}^{-1})dS_{i,e}]$. For the source part, $dJ_e=-J_e(dS_{j,e})J_e$, giving $\frac12\operatorname{tr}[(J_e-J_eQ_eJ_e)dS_{j,e}]$. This retains the receiver covariance, mean disagreement, and dictionary variance in the source force. If a graph includes an identified self variable, both relevant contributions must be added; treating its receiver and source as independent copies is incorrect.

The product Gaussian Fisher metric is

\[
\begin{aligned}
g_x(\delta x,\delta x)={}&\sum_i\left[\delta\mu_i^T\Sigma_i^{-1}\delta\mu_i
+\frac12\operatorname{tr}(\Sigma_i^{-1}\delta\Sigma_i\Sigma_i^{-1}\delta\Sigma_i)\right]\\
&+\sum_e\left[\delta a_e^TV_e^{-1}\delta a_e
+\frac12\operatorname{tr}(V_e^{-1}\delta V_eV_e^{-1}\delta V_e)\right].
\end{aligned}
\tag{16}
\]

Consequently the complete matter equations are

\[
\dot\mu_i=-\Sigma_i g_{\mu_i},\qquad
\dot\Sigma_i=-2\Sigma_iG_{\Sigma_i}\Sigma_i,\qquad
\dot a_e=-V_eg_{a_e},\qquad
\dot V_e=-2V_eG_{V_e}V_e.
\tag{17}
\]

In particular,

\[
\dot V_e=V_e-V_e(A_{ee}+\beta_eH_e^TJ_eH_e)V_e.
\tag{18}
\]

For the full product Fisher flow, the row metric is $\sum_e(d\beta_e)^2/\beta_e$ on $\sum_e d\beta_e=0$, and the additional equations are

\[
\begin{aligned}
h_e&=C_e+T_i\bigl(\log(\beta_e/\pi_e)+1\bigr),\\
\dot\beta_e&=-\beta_e\left(h_e-\sum_{f\in\mathcal I_i}\beta_fh_f\right),\qquad e\in\mathcal I_i.
\end{aligned}
\tag{19}
\]

For the reduced convention, (19) is absent and $\beta=\beta^*(x)$ is reevaluated at every $x$. Its kinetic metric is (16), exactly the matter-only convention already stated in Chapter 3. It is not the pullback of the full product metric to the graph $x\mapsto(x,\beta^*(x))$, which would add a row-score contribution and give a different mobility. Neither convention assumes that the finite-rate row trajectory stays Gibbs.

The full dissipation identity is

\[
\begin{aligned}
\dot F={}&-\sum_i\left[g_{\mu_i}^T\Sigma_ig_{\mu_i}
+2\operatorname{tr}\bigl((\Sigma_i^{1/2}G_{\Sigma_i}\Sigma_i^{1/2})^2\bigr)\right]\\
&-\sum_e\left[g_{a_e}^TV_eg_{a_e}
+2\operatorname{tr}\bigl((V_e^{1/2}G_{V_e}V_e^{1/2})^2\bigr)\right]
-\sum_i\operatorname{Var}_{\beta_i}(h).
\end{aligned}
\tag{20}
\]

For the reduced flow, $\dot F_{\mathrm{red}}$ is the same expression without the row variance. This follows by contracting each derivative with (17) and (19). It does not imply a sign for any individual covariance velocity or for any individual edge cost.

## Stationarity and the effect of adapting rows

Because all Fisher blocks are positive definite in the interior, stationarity is equivalent to (12)–(15) vanishing and, for the full flow, $\beta=\beta^*(x)$. In particular,

\[
V_e^{-1}=A_{ee}+\beta_eH_e^TJ_eH_e,
\tag{21}
\]

\[
\left[A+\operatorname{diag}(\beta_eH_e^TJ_eH_e)\right]a
=b+\left(\beta_eH_e^TJ_e\bar T_e(\mu_i-\mu_j)\right)_e,
\tag{22}
\]

while the content means satisfy (12) and the content covariance inverse equals the right side of (14) after removing $-\Sigma_i^{-1}$. This is a coupled nonlinear system: the precisions $J_e$, dictionary variances, and Gibbs rows all depend on the unknown state. The positive-definite matrix in (22) gives a unique dictionary mean for fixed contents, content covariances, and rows; it does not give uniqueness of the full stationary point.

Let $x$ be local coordinates for all matter parameters, let $c_e=\nabla_x C_e$, and define $\bar c_i=\sum_{e\in\mathcal I_i}\beta_e^*c_e$. Ordinary differentiation of (5) gives

\[
d\beta_e^*[\xi]=-{\beta_e^*\over T_i}\left(dC_e[\xi]-\sum_f\beta_f^*dC_f[\xi]\right),
\]

and the reduced Hessian is

\[
H_{\mathrm{red}}
=H_0+\sum_e\beta_e^*H_e^C
-\sum_i{1\over T_i}\left[\sum_{e\in\mathcal I_i}\beta_e^*c_ec_e^T-\bar c_i\bar c_i^T\right].
\tag{23}
\]

Here $H_e^C=\nabla_x^2C_e$ and $H_0=\nabla_x^2F_0$; the code map $H_e$ in (8) is a different object. The first two terms are the Hessian with the rows frozen at their current Gibbs values. For a direction $\xi$, the final correction is

\[
-\sum_iT_i^{-1}\operatorname{Var}_{\beta_i^*}\bigl(dC_e[\xi]\bigr).
\tag{24}
\]

Thus row adaptation can only lower this quadratic form relative to that same frozen-row Hessian. This is a comparison at the same state and the same weights. It is not a monotonicity theorem in temperature, because changing temperature normally also changes the weights and stationary state.

For completeness, the manuscript's full/reduced inertia relation follows directly here. At a joint stationary point, let $v_i$ be a row tangent with $\sum_ev_e=0$, set $z_e=dC_e[\xi]$ and $\bar z_i=\sum_e\beta_e^*z_e$, and complete the square:

\[
\begin{aligned}
\delta^2F[(\xi,v),(\xi,v)]
={}&\xi^TH_{\mathrm{red}}\xi\\
&+\sum_iT_i\sum_{e\in\mathcal I_i}{\left[v_e+\beta_e^*(z_e-\bar z_i)/T_i\right]^2\over\beta_e^*}.
\end{aligned}
\tag{25}
\]

The shift of $v_i$ is tangent because its entries sum to zero, and it is an invertible linear change of tangent variables. The second term is positive definite on the row tangent space. Hence the full and reduced Hessians have the same negative and zero indices; the full Hessian has the additional positive row directions. Equation (25) is the concrete Schur-complement argument underlying `prop:dyn-morse`.

At a stationary point the linearization of a Fisher gradient flow is $-g^{-1}H$; derivatives of $g^{-1}$ multiply the vanished first derivative. It is similar to the symmetric matrix $-g^{-1/2}Hg^{-1/2}$. All eigenvalues are real. A positive-definite Hessian gives local exponential asymptotic stability; a negative Hessian direction gives linear and nonlinear instability. If the Hessian has zero eigenvalues, this argument does not settle nonlinear stability. Positive row mobilities could change relaxation rates but not these inertia conclusions. A finite-rate row model therefore cannot create a Hopf bifurcation at an interior critical point of this exact reciprocal gradient system. This statement does not apply to a per-agent field or to a non-gradient modification.

In particular, an explicit example would be unstable if it has a stationary point and a direction $\xi$ for which the variance in (24) exceeds the frozen-row quadratic form. Establishing that condition requires the complete matter Hessian, including covariance and dictionary directions; the existing fixed-covariance mean theorem alone does not provide it.

## Global forward existence in the Gaussian restriction

**Theorem.** Fix a finite graph and all data above with $P_i\succ0$, $A\succ0$, $R_e\succ0$, $T_i>0$, and $\pi_e>0$. Fix the codes and $\kappa_i$, with no assumption that the latter vanish. For every finite initial mean, positive-definite initial covariance, and positive initial row, the full system (17),(19) has a unique solution for every $t\geq0$. The matter variables remain in a compact subset of their regular Gaussian parameter space, and every row entry has a uniform positive lower bound depending on the initial state and fixed data. The reduced system (17) with (5) has the same global existence and compactness conclusions. Each potential has at least one interior stationary point. Along either flow its dissipated squared Fisher-gradient norm is integrable, tends to zero, and every accumulation point is stationary.

**Proof.** All displayed formulas are smooth on the product of Euclidean mean spaces, positive-definite covariance cones, and simplex interiors. Local existence and uniqueness follow from the finite-dimensional locally Lipschitz vector field. On this local solution, (20) gives the decreasing energy bound.

Every term in (2) is nonnegative. In the reduced potential, $C_e\geq0$ implies $0<Z_i\leq1$, so $F_{\mathrm{red}}\geq F_0\geq0$. Therefore either energy sublevel bounds every content self KL and the joint dictionary KL by its initial energy $E_\mathrm{init}$.

For a content covariance, diagonalize $P_i^{1/2}\Sigma_iP_i^{1/2}$ with eigenvalues $\lambda_k>0$. Formula (10) is

\[
\frac12\|\mu_i-m_i\|_{P_i}^2+\frac12\sum_k(\lambda_k-\log\lambda_k-1).
\]

The scalar function $f(\lambda)=\lambda-\log\lambda-1$ is nonnegative and tends to infinity as $\lambda\downarrow0$ or $\lambda\uparrow\infty$. Thus the means are bounded and every covariance is bounded above and bounded away from zero. For the dictionary covariance, use the same argument on $A^{1/2}VA^{1/2}$ in (11). It bounds $a$ and the eigenvalues of the full block-diagonal $V$, and hence every $V_e$. The fixed positive-definite matrices convert these bounds to ordinary Euclidean and spectral bounds. The matter part of the energy sublevel is consequently contained in a compact set $\mathcal K$ lying strictly inside the Gaussian domain.

Equation (9) is continuous on $\mathcal K$, so all costs lie in a finite interval $0\leq C_e\leq M$. For the reduced convention, (5) immediately gives

\[
\beta_e^*\geq\pi_e e^{-M/T_i}.
\tag{26}
\]

For the full flow, define the reference-adjusted log-ratio within one row,

\[
\eta_{ef}=\log(\beta_e/\pi_e)-\log(\beta_f/\pi_f).
\]

Subtracting the two equations for $\dot\beta/\beta$ in (19) gives the exact forced linear equation

\[
\dot\eta_{ef}=-T_i\eta_{ef}-(C_e-C_f).
\tag{27}
\]

The costs need not be fixed. Variation of constants and $|C_e-C_f|\leq M$ give

\[
|\eta_{ef}(t)|\leq e^{-T_it}|\eta_{ef}(0)|+{M\over T_i}(1-e^{-T_it})\leq L_i,
\]

where $L_i=\max\{M/T_i,\max_{e,f}|\eta_{ef}(0)|\}$. Set $u_e=\beta_e/\pi_e$. The ratio bound implies $u_f\leq e^{L_i}u_e$ for every $f$, while $\sum_f\pi_fu_f=1$, hence

\[
\beta_e(t)\geq\pi_e e^{-L_i}>0.
\tag{28}
\]

A singleton row is identically one. The full trajectory therefore stays in a compact subset of the regular product domain. Its vector field is bounded and locally Lipschitz on a neighborhood of that set, so the local solution cannot have a finite maximal time. The same continuation argument applies to the reduced flow on $\mathcal K$, proving global existence.

The reduced energy has a global minimum: choose any finite reference point and minimize on its nonempty compact sublevel. All possible points with lower energy lie in that sublevel, and its compactness was just proved for the matter variables. The minimizer is interior, so it is a critical point. Appending its Gibbs rows gives an interior stationary point of the full potential through (6). This argument proves existence, not uniqueness.

Let $\mathcal D(t)=-\dot F(t)$ or $-\dot F_{\mathrm{red}}(t)$. Integrating the energy identity gives $\int_0^\infty\mathcal D(t)dt\leq E_\mathrm{init}$. On the compact invariant set, $\mathcal D$ and its derivative along the smooth vector field are bounded, so $\mathcal D(t)$ is uniformly continuous. A nonnegative uniformly continuous integrable function tends to zero: otherwise infinitely many disjoint intervals of a fixed positive width carry a fixed positive area. Every accumulation point consequently has vanishing Fisher gradient and is stationary. This completes the proof.

The theorem does not assert a unique stationary point or convergence to a unique global minimizer. Proving convergence of each trajectory to one critical point would additionally invoke an analytic-gradient convergence theorem with its hypotheses stated; it is not needed for the existence, compactness, or local stability results above. The complete trajectory stays away from the boundary, but the lower bounds can be very small when a temperature is small. The theorem does not pass uniformly to $T=0$, removal of the content references, a singular dictionary prior, moving operating contents, or nonlinear group-valued dictionaries.

## What this resolves and what remains

Equations (12)–(19) close the finite-dimensional coupled system for the declared Gaussian restriction, and (21)–(25) give its exact stationary equations and stability test. The global theorem prevents finite-time blowup, covariance collapse, and loss of positive row weights under the stated properness hypotheses. It does not transfer the scalar frozen-variable uncertainty monotonicity to the coupled flow. It also does not claim that general-law trajectories remain Gaussian, that adaptive routing produces groups, or that those groups have inherited parent dynamics.

The companion [stability-example.md](stability-example.md) supplies a concrete finite graph with a fully specified stationary point and a complete Hessian calculation, including every covariance and dictionary variable. Its nonlinear critical-temperature behavior remains open. The general-law counterpart outside regular finite-dimensional families also remains open: it needs a specified law-space metric and an invariant domain on which the source-ratio force in (3) is locally regular, together with well-posedness and compactness hypotheses. No numerical script, GPU job, Lean work, manuscript edit, or Item 7 edit was performed by the author of this derivation. The coordinating agent's bounded symbolic and finite-difference checks are recorded in [REPORT.md](REPORT.md).
