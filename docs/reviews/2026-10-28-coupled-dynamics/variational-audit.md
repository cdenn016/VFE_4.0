# Variational audit of the coupled reciprocal flow

No variational-inference findings.

The inspected scope is `docs/research/2026-10-26-reciprocal-coupled-dynamics/full-flow.md`, equations (1)–(25), against the live `Research/manuscripts/MAgent_Theory_2.0/AUTHORING.md`, Chapter 3's reciprocal potential, Gibbs reduction, envelope identity, restricted-family convention, and Hessian-inertia proposition, and Chapter 18's `it2:model`, `it2:potential`, `it2:potential-forces`, `it2:certainty`, and `it2:open-reciprocal`. This review concerns the stated reciprocal potential and its Gaussian restriction. It does not assess simulator code or establish a fixed-generative-model ELBO interpretation.

The independent algebraic reconstruction found no discrepancy in the product-recognition variation, expected conditional message KL, reciprocal matter forces, or row reduction. For a product recognition law with a correlated reference, expanding the dictionary KL and varying one factor gives

\[
\frac{\delta F}{\delta r_e}
=\log r_e-\mathbb E_{r_{-e}}\log\pi_\omega+\beta_e\Delta_e
\]

up to a sample-independent constant. This agrees with (4) and preserves the correlated reference; replacing its conditional expected log density with the log of a marginal density would change the objective. In the Gaussian restriction, the joint reference precision contributes its full matrix to the mean force and its diagonal block to each factor covariance force, as in (13) and (15).

The conditional message KL is averaged after forming the divergence. Its mean-square term therefore contributes dictionary uncertainty through

\[
Q_e=S_{i,e}+d_ed_e^T+H_eV_eH_e^T.
\]

Holding the other arguments fixed, differentiating the source message covariance gives

\[
d_{S_{j,e}}C_e
=\tfrac12\operatorname{tr}[(J_e-J_eQ_eJ_e)dS_{j,e}],
\]

whereas the receiver covariance derivative is one half of the trace pairing with \(J_e-S_{i,e}^{-1}\). Thus (14) retains all three source-force contributions, including dictionary uncertainty. The dictionary covariance derivative gives \(2G_{V_e}=A_{ee}-V_e^{-1}+\beta_eH_e^TJ_eH_e\), consistent with the stationarity equation (21).

Substitution of \(\log(\beta_e^*/\pi_e)=-C_e/T_i-\log Z_i\) into the row block gives exactly \(-T_i\log Z_i\). Expanding the difference at an arbitrary row gives \(T_iD_{\mathrm{KL}}(\beta_i\Vert\beta_i^*)\), which establishes (5)–(6) directly. The first derivative is therefore the weighted cost derivative in (6). Differentiating that expression gives the negative row covariance in (23), and completing the row tangent square gives (25). These checks support the stated local stationarity and Hessian-inertia conclusions; they do not imply uniqueness of the coupled stationary point or nonlinear stability in zero-Hessian directions.

Two requirements apply when integrating this note into Chapter 18. They are integration constraints, not defects in the reviewed note.

1. Chapter 18's existing `it2:potential` at `chapters/18_link_dynamics.tex:212` is a fixed-row potential. The new variable-row objective must explicitly add \(\sum_i T_iD_{\mathrm{KL}}(\beta_i\Vert\pi_i)\), with fixed positive normalized priors and positive temperatures. Substituting \(\beta^*(x)\) into the old entropy-free scalar would leave the additional derivative \(\sum_e C_e\,d\beta_e^*\), so it would not produce the claimed envelope force. The old fixed-row mean theorem remains valid in its declared scope.
2. Rename the note's linear map \(H_e=\bar T_e\kappa_i\) on integration, because `chapters/18_link_dynamics.tex:15` already uses \(H_e\) for the message space. Use a distinct map symbol consistently in the uncertainty, derivative, and stationary equations; preserve the chapter's message-space notation.

This investigator ran no scripts, numerical experiments, GPU work, or Lean checks, and changed no manuscript source. The evidence above is a direct algebraic reconstruction, not a claim of independently reproduced numerical outputs. Global forward existence and the separate stability example lie outside this review's assigned variational scope.
