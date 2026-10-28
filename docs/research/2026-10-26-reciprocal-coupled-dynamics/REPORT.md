# Coupled reciprocal dynamics: a bounded resolution of Item 2

The coupled Gaussian tangent system admits a complete Fisher vector field, global forward solutions under proper fixed references, and an exact local stability calculation with every content and dictionary mean and variance free. In the explicit three-agent example, the stationary state is exponentially stable for routing temperature $T>3269/10118$ and has exactly one unstable direction for $0<T<3269/10118$. Freezing its five variances instead predicts stability whenever $T>31/122$. Thus at $T=3/10$, the frozen-variance prediction is wrong for the complete system.

This resolves the requested finite-family dynamics and local-stability obligation. It concerns the reciprocal law potential. It does not establish unrestricted law-space well-posedness, a fixed normalized generative joint for the peer interactions, or formation of an inherited parent. The live manuscripts and Item 7 were consulted without edits. The investigation is saved in the VFE_4.0 worktree and is uncommitted.

## Continuity with the existing manuscript

The source is the live `Research/manuscripts/MAgent_Theory_2.0/` at the revision and byte hashes recorded in [source-baseline.json](source-baseline.json). Chapter 3 already establishes the Gibbs row gap, the envelope identity, finite-rate row relaxation, the restricted Gaussian Fisher metric, and the full/reduced Hessian inertia relation. Its Gaussian noninvariance proposition prevents identifying the restricted flow with unrestricted law-space dynamics. Chapter 18, especially `it2:potential`, `it2:potential-forces`, `it2:certainty`, and `it2:open-reciprocal`, supplies the tangent message family and the remaining coupled-dynamics question. These statements were read with `AUTHORING.md` and the wiki's replicator-dynamics and Fisher-information entries.

The work here applies those results to changing dictionary recognition and content covariance, proves a compactness statement for the declared Gaussian family, and gives a complete finite stability example. The general row Schur-complement identity is retained as an existing result. The detailed derivations are [full-flow.md](full-flow.md) and [stability-example.md](stability-example.md).

## General-law variation and complete Gaussian equations

Let $q_i$ be normalized content laws, let $r=\bigotimes_e r_e$ be product dictionary recognition, and let $\pi_\omega$ be a fixed normalized prior that may correlate edges. Fixed normalized kernels produce receiver and source message laws $A_e(\omega)$ and $B_e$. The expected cost is

\[
C_e(q,r)=\mathbb E_{r_e}D_{\mathrm{KL}}(A_e(\omega)\Vert B_e).
\]

With positive fixed normalized source priors $\pi_i$ and temperatures $T_i$, the reciprocal potential is

\[
\begin{aligned}
\mathcal F&=F_0+\sum_i\left[\sum_{e\in\mathcal I_i}\beta_e C_e
+T_iD_{\mathrm{KL}}(\beta_i\Vert\pi_i)\right],\\
F_0&=\sum_iD_{\mathrm{KL}}(q_i\Vert p_i)+D_{\mathrm{KL}}(r\Vert\pi_\omega).
\end{aligned}
\]

The expectation is outside the message divergence; averaging message laws before computing the divergence changes the potential. Receiver, source, and dictionary variations are all retained. Differentiation requires the regular dominated charts and integrable envelopes stated in the full derivation; standard Borel measurability alone does not supply them.

In the Gaussian tangent realization, $q_i=\mathcal N(\mu_i,\Sigma_i)$ and $r_e=\mathcal N(a_e,V_e)$. Operating contents, codes, noise, and prior data are fixed, while all these Gaussian parameters vary. For edge $e:j\to i$, define

\[
\begin{aligned}
H_e&=\bar T_e\kappa_i,& S_{i,e}&=\bar T_e\Sigma_i\bar T_e^T+R_e,\\
J_e&=(\bar T_e\Sigma_j\bar T_e^T+R_e)^{-1},&
d_e&=\bar T_e(\mu_i-\mu_j)-H_ea_e,\\
Q_e&=S_{i,e}+d_ed_e^T+H_eV_eH_e^T.
\end{aligned}
\]

The exact cost is $C_e=[\operatorname{tr}(J_eQ_e)-n_e-\log\det J_e-\log\det S_{i,e}]/2$. Equations (12)–(15) of the flow note give every derivative. In particular, the source covariance force includes $\bar T_e^T(J_e-J_eQ_eJ_e)\bar T_e/2$, retaining the dictionary variance in $Q_e$.

The product Fisher equations are

\[
\begin{aligned}
\dot\mu_i&=-\Sigma_i\nabla_{\mu_i}\mathcal F,&
\dot\Sigma_i&=-2\Sigma_i\nabla_{\Sigma_i}\mathcal F\Sigma_i,\\
\dot a_e&=-V_e\nabla_{a_e}\mathcal F,&
\dot V_e&=-2V_e\nabla_{V_e}\mathcal F V_e.
\end{aligned}
\]

Covariance derivatives use the symmetric trace pairing. The full convention also evolves the rows by their categorical Fisher gradient. The reduced convention substitutes $\beta_e^*\propto\pi_e e^{-C_e/T_i}$ and keeps the matter-family Fisher metric. The two have the same interior stationary equations and generally different trajectories; the reduced metric is not the metric pulled back from the Gibbs graph.

Adaptive rows subtract $\sum_iT_i^{-1}\operatorname{Cov}_{\beta_i^*}(\nabla C_e)$ from the matter Hessian with rows frozen at the same weights. The complete-square identity in the flow note gives the same negative and zero Hessian indices for full row relaxation and the reduced potential. This compares a given state and row, rather than asserting general monotonicity as temperature changes.

The natural-gradient convention is the inverse Fisher tensor applied to the ordinary derivative, as in theorem 1 and section 3 of [Amari, *Natural Gradient Works Efficiently in Learning*, Neural Computation 10 (1998), 251–276](https://bsi-ni.brain.riken.jp/database/file/176/181.pdf). The Gaussian blocks and all results used here are derived in the linked notes.

## Global forward existence within the Gaussian family

For a finite graph, assume fixed proper Gaussian content references, a fixed positive-definite joint Gaussian dictionary prior, positive message-noise covariances, positive source priors, and positive temperatures. Both full and reduced flows have unique solutions for every forward time from every interior initial condition. Means remain bounded, every covariance stays bounded above and away from zero, and row weights have uniform positive lower bounds depending on initial state and fixed data. An interior stationary point exists, dissipation tends to zero, and every accumulation point is stationary.

The exact decreasing potential bounds the self and dictionary Gaussian KL terms. These bound means and covariance spectra because $u-\log u-1$ diverges at both ends of $(0,\infty)$. The resulting compact matter set bounds every cost. For full row relaxation, each reference-adjusted log-ratio obeys $\dot\eta_{ef}=-T_i\eta_{ef}-(C_e-C_f)$, which supplies a positive row floor. Smooth finite-dimensional continuation gives global existence. Compactness and dissipation supply stationary accumulation points. The theorem does not assert uniqueness or convergence to one particular point, and does not extend automatically to zero temperature, improper references, moving operating contents, or unrestricted probability-law dynamics.

## Exact finite stability result

Use three scalar content laws and two edges $1\to0$ and $2\to0$, with identity codes, unit message noise, fixed $\kappa_0=1$, and no loop term. Content reference precisions are $(1,5/4,5/4)$ and reference means are $(0,-23/15,23/15)$. Each dictionary reference is $\mathcal N(0,4/3)$. In coordinate order

\[
x=(\mu_0,\mu_1,\mu_2,a_1,a_2,v_0,v_1,v_2,V_1,V_2),
\]

all ten coordinates are free. The full system adds $b\in(0,1)$ with prior $(1/2,1/2)$. For every $T>0$, a stationary point is

\[
x_*=(0,-4/3,4/3,1/3,-1/3,1,1,1,1,1),\qquad b_*=1/2.
\]

Both costs equal $1/2$. Let $H$ be the ten-dimensional matter Hessian with the row fixed at $1/2$, and $g=\nabla(C_1-C_2)(x_*)$. The detailed note gives a positive sum-of-squares certificate for $H\succ0$ on the entire tangent space and the exact inverse action

\[
H^{-1}g=\frac1{5059}(2400,-960,-960,-1600,-1600,0,-716,716,2440,-2440)^T.
\]

Consequently

\[
H_T=H-\frac{gg^T}{4T},\qquad
T_c=\frac14g^TH^{-1}g=\frac{3269}{10118}\simeq0.32308757.
\]

Congruence by $H^{-1/2}$ leaves one potentially changing sign. This proves positive definiteness above $T_c$, one negative direction below $T_c$, and exactly one zero direction at $T_c$. The stationary Fisher Jacobian is similar to a symmetric matrix with opposite Hessian inertia. Hence there is exponential stability above the threshold and a hyperbolic saddle with one unstable direction below it. Completing the square in the full eleven-dimensional Hessian proves the same classification for finite-rate row relaxation.

With all five variances frozen at one, the threshold is instead $31/122\simeq0.25409836$. The exact inequality

\[
\frac{31}{122}<\frac3{10}<\frac{3269}{10118}
\]

gives the claimed discrepancy. The zero mode at the full threshold includes source and dictionary variances. The nonlinear classification at $T_c$ remains open; a pitchfork or a pair of stable selected states has not been proved.

## Executed checks and limits

[check_stability.py](check_stability.py) reconstructs the scalar objective and checks the stationary gradients, full positive-definite Hessian certificate, inverse action, both exact thresholds, direct reduced Hessian, and full row Hessian using SymPy rational arithmetic. All checks passed; [stability-checks.json](stability-checks.json) records the matrices and results. Numerical Jacobian eigenvalues at $T=1/4,3/10,1/2$ accompany the analytic proof. At $T=3/10$, the largest eigenvalues are approximately $0.134325$ for the reduced system, $0.0194835$ for full row relaxation, and $-0.286659$ for the frozen-variance mean system.

[check_matrix_flow.py](check_matrix_flow.py) checks all general derivative formulas at one two-dimensional, three-edge state with noncommuting covariances, four-dimensional dictionary factors, nonzero operating contents, and a correlated prior. Central differences cover all 57 matter coordinates and the full flow's extra row coordinate. The largest derivative error was $2.32\times10^{-9}$; the largest dissipation residual was $5.69\times10^{-14}$. [matrix-flow-checks.json](matrix-flow-checks.json) records these finite checks. They support matrix order and the implemented derivatives; the written derivation establishes the algebraic scope.

Both scripts used a bounded CPU lane with thread limits of four. No trajectory integration, training, GPU work, benchmark, full test suite, or Lean formalization was performed. Long-time destinations, the nonlinear center direction at $T_c$, cycle-dependent examples, unrestricted law-space closure, and endogenous formation remain separate obligations. File preservation and scope checks are recorded in [completion-validation.json](completion-validation.json).
