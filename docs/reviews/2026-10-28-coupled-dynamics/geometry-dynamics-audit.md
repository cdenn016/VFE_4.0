# Fisher geometry and global dynamics audit, 2026-10-28

No information-geometry findings.

The reviewed claims in equations (16)–(28) of `docs/research/2026-10-26-reciprocal-coupled-dynamics/full-flow.md` hold under their stated finite-dimensional Gaussian, fixed-data, and positivity hypotheses. No correction is required by this lens. This is an analytical reconstruction of the specified formulas and proof, not an unrestricted law-space well-posedness result or an assessment of the nonlinear critical-temperature mode.

## Inspected scope and primary source

Read the complete flow note, including its general-law regularity statement and normalized correlated dictionary prior. Read the relevant live definitions and proofs in `Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex`: the finite-table flow, row relaxation, full/reduced critical structure, regular finite-dimensional Fisher flow, Gaussian restriction, Gaussian noninvariance, and bounded log-density charts. Read Chapter 18's tangent model, reciprocal potential, and open coupled-dynamics question, together with `AUTHORING.md`. Consulted the wiki's information-geometry theme, Fisher metric discussion, and replicator-dynamics page for continuity; those pages were not used as proof of the audited identities.

The primary natural-gradient source was read directly: [Amari, *Natural Gradient Works Efficiently in Learning*, Neural Computation 10 (1998), 251–276](https://bsi-ni.brain.riken.jp/database/file/176/181.pdf), theorem 1, equation (2.3), and section 3.1, equation (3.5). It identifies the natural gradient as the inverse Riemannian metric applied to the ordinary derivative and defines the statistical metric by the expected score product. The Gaussian and categorical calculations below reconstruct that definition for the declared recognition family.

## Analytical reconstruction

For a Gaussian with centered sample $z=y-\mu$, the score of a mean/covariance variation $(u,U)$ is

\[
s_{u,U}=u^T\Sigma^{-1}z+
\frac12\left(z^T\Sigma^{-1}U\Sigma^{-1}z-
\operatorname{tr}(\Sigma^{-1}U)\right).
\]

Odd centered moments make the mean/covariance cross term zero. Contracting the second and fourth moments gives

\[
\mathbb E[s_{u,U}s_{v,V}]
=u^T\Sigma^{-1}v+
\frac12\operatorname{tr}(\Sigma^{-1}U\Sigma^{-1}V).
\]

Thus the covariance block is one half the conventional affine-invariant SPD metric. Pairing this metric with the trace derivative verifies the factor $2\Sigma G_\Sigma\Sigma$ in the inverse metric. The same calculation applies to dictionary recognition because each dictionary factor is an explicitly normalized Gaussian law; it is not a metric asserted merely from a matrix parameter. Independent factors have centered scores, so their Fisher cross blocks vanish. Correlation in the fixed prior affects the potential, not this product recognition metric. These observations validate full-flow lines 198–225 and the covariance terms in the dissipation identity at lines 239–252.

For a categorical row, a tangent $v$ with $\sum_e v_e=0$ has score $v_e/\beta_e$, and its Fisher metric is $\sum_e v_e^2/\beta_e$. The vector $\beta_e(h_e-\sum_f\beta_fh_f)$ pairs with every tangent as $\sum_eh_ev_e$. This verifies the centered row gradient and negative variance in equations (19) and (20), including their temperature convention.

The Gibbs graph would contribute an additional metric term

\[
\sum_i\sum_{e\in\mathcal I_i}
\frac{(d\beta_e^*[\xi])^2}{\beta_e^*}
=\sum_i T_i^{-2}\operatorname{Var}_{\beta_i^*}(dC_e[\xi]).
\]

The note explicitly excludes that pullback metric from its reduced convention at line 237. Its chosen matter metric therefore agrees with the earlier manuscript convention and is internally consistent. Identical stationary points do not identify the two transient vector fields.

At a joint critical point, the full second variation on a matter tangent $\xi$ and a zero-sum row tangent $v$ is the frozen-row matter quadratic plus $2\sum_ev_edC_e[\xi]+T_i\sum_ev_e^2/\beta_e^*$. Completing the square by the zero-sum shift $v_e\mapsto v_e+\beta_e^*(dC_e[\xi]-\mathbb E_{\beta_i^*}dC[\xi])/T_i$ gives equation (25). The shift is invertible on each tangent space, so the remaining row form adds positive directions and preserves the negative and zero indices. At a critical point the derivatives of the inverse metric multiply a vanished first derivative; the Jacobian is $-g^{-1}H$, similar to $-g^{-1/2}Hg^{-1/2}$. The real-spectrum and nondegenerate local-stability conclusions at lines 295–308 follow. Zero modes still need nonlinear analysis.

The compactness argument at lines 314–365 also holds. Every self KL and the joint dictionary KL is nonnegative; the reduced interaction is nonnegative because each cost is nonnegative and the source prior is normalized. An energy bound therefore bounds each eigenvalue of $P_i^{1/2}\Sigma_iP_i^{1/2}$ and $A^{1/2}VA^{1/2}$ using $f(u)=u-\log u-1$. If the latter eigenvalues lie in $[u_-,u_+]$ with $u_->0$, then

\[
u_-A^{-1}\preceq V\preceq u_+A^{-1}.
\]

Consequently every block $V_e$ stays positive definite and bounded even when $A$ has off-diagonal blocks. The mean quadratic terms are coercive because $P_i$ and $A$ are positive definite. The product covariance restriction is closed within these spectral bounds, so the matter sublevel is compact inside its regular domain.

Bounded costs force the exact adjusted log-odds equation (27). Variation of constants bounds every adjusted row contrast, and normalization then gives $\beta_e\geq\pi_e e^{-L_i}$. The vector field remains smooth on a neighborhood of the resulting compact interior set. Finite-time exit is therefore excluded by ordinary finite-dimensional continuation. The reduced sublevels are themselves compact, so minimizing on one proves existence of an interior minimizer; appending its Gibbs row proves full stationarity. On the compact trajectory closure, the nonnegative dissipation and its derivative are bounded, giving uniform continuity. Integrability then implies dissipation tends to zero, and continuity implies every accumulation point is stationary. No single-point convergence claim is needed for these conclusions.

## Conditions to retain in integration

State the Gaussian product family, positive-definite content precisions and joint dictionary precision, positive-definite message noise, positive fixed source priors and temperatures, fixed codes and operating contents, and interior initial conditions inside the integrated theorem. The stationary/minimizer and compactness statements are finite-family results. If a general parametric Fisher sentence is retained, make explicit that the regular family has finite, positive-definite Fisher tensor and justified objective differentiation; finite KL or standard Borel measurability alone is insufficient.

Keep the full row relaxation and matter-metric Gibbs reduction as separately defined conventions. Preserve the factor two in covariance velocities, the categorical entropy term and normalized source prior, the distinction between product recognition and correlated prior, and the difference between nondegenerate Hessian stability and nonlinear behavior at a zero mode. Do not promote stationary accumulation to a unique limit without an additional stated argument, or extend the row floor uniformly to zero temperature. None of these conditions is missing from the audited Gaussian theorem; they are the conditions that must survive manuscript compression.

No numerical scripts, GPU work, Lean work, manuscript edits, or Item 7 changes were performed by this investigator. The only authored file is this report.
