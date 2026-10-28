# Coupled reciprocal dynamics: audit and integration, 2026-10-28

The three independent domain audits found no mathematical defect in the reviewed coupled-flow derivation or its exact scalar stability example. The supported results are now integrated into `Research/manuscripts/MAgent_Theory_2.0/chapters/18_link_dynamics.tex` and `companion/c18_link_dynamics.tex`. The global existence and stability conclusions retain their Gaussian tangent-family scope. The nonlinear critical mode, unrestricted law-space dynamics, and formation remain open. Nothing was committed or published.

## Audit scope and findings

The audit examined [full-flow.md](../../research/2026-10-26-reciprocal-coupled-dynamics/full-flow.md), [stability-example.md](../../research/2026-10-26-reciprocal-coupled-dynamics/stability-example.md), both checking scripts, and their saved results against the live manuscript's objective definitions, restricted-family convention, Gaussian noninvariance result, and authoring rules. Three investigators independently reconstructed the variational derivatives, Fisher geometry and global-existence proof, and scalar stability calculation. Their reports are [variational-audit.md](variational-audit.md), [geometry-dynamics-audit.md](geometry-dynamics-audit.md), and [stability-audit.md](stability-audit.md). The investigators performed analytical checks and read saved outputs; the coordinator subsequently executed both checking scripts in the bounded CPU lane. No adversarial adjudication panel was requested or run.

The variational audit verified that the expected conditional KL is formed before averaging over dictionary recognition, that source-role covariance forces include dictionary uncertainty, and that a correlated dictionary reference contributes the full precision matrix to the mean force and its diagonal blocks to the product-factor covariance forces. It required the integration to state the categorical relative-entropy term before Gibbs reduction. The prior chapter's entropy-free scalar applies only at fixed rows. Substituting state-dependent Gibbs weights into that scalar would not give the envelope derivative.

The geometry audit reconstructed the product Gaussian and categorical Fisher metrics, including the factor two in covariance velocities, and verified compact interior sublevels and the adjusted-log-odds bound. It retained the distinction between the matter-family metric used by Gibbs-reduced dynamics and the pullback of the full metric to the Gibbs graph. It verified stationary accumulation, without inferring uniqueness or convergence to one stationary point.

The stability audit reconstructed the positive quadratic certificate on all ten matter coordinates, the exact inverse Hessian action, and the eleventh row coordinate. It required the receiver's candidate set to contain exactly the two peers in the example; adding an identity self candidate changes the Gibbs problem. It also identified the explanatory distinction between the reduced critical null vector and the full null vector. These were integration conditions and clarifications, not defects in the original derivation.

## Integrated result

The main source now defines the complete variable-row reciprocal objective, its Gibbs minimum and gap identity, and the general-law first variation under dominated regularity assumptions. It gives every content and dictionary mean and covariance derivative for the Gaussian product family, the two row conventions, the dissipation identities, and the coupled stationary equations. The linear dictionary-to-message map is denoted by `K_e` so it does not collide with the existing message-space symbol `H_e`.

The global-forward-existence theorem fixes a finite graph, codes and operating maps, proper Gaussian content references, a normalized joint dictionary prior with positive-definite precision, positive-definite message noises, positive fixed normalized source priors and temperatures, and interior initial data. It proves bounded means, covariance eigenvalues bounded above and away from zero, positive row floors, global solutions, an interior stationary point, vanishing dissipation, and stationary accumulation. None of these conclusions asserts that unrestricted law-space dynamics preserves Gaussianity.

The critical-structure proposition gives the negative row-covariance correction to the matter Hessian and the full row completion of squares. The scalar example then proves the exact threshold

\[
T_c=\frac{3269}{10118},
\]

with local exponential stability above it and one unstable direction below it, for both full row relaxation and matter-metric Gibbs reduction. Its fixed-variance threshold is $31/122$, so at $T=3/10$ freezing the variances predicts stability while both complete systems are unstable. The full and reduced transient vector fields remain distinct. Nonlinear behavior at the single zero mode is not classified, and the instability is not a formation theorem.

The companion explains the derivative roles, the covariance mobility, the prior-based compactness argument, and the complete scalar calculation. The Item 2 backlog, authoring scope, and manuscript overview were updated. The standalone Item 2 document retains its earlier scope; Item 7 remains separate and unchanged.

The source file requested as Chapter 18 is currently numbered Chapter 19 in the main PDF because an earlier chapter is split across two source files. Its printed section numbers are 19.6--19.9, on PDF pages 251--260. The corresponding companion is numbered Chapter 18, with the new material on PDF pages 103--107. This integration did not change chapter numbering.

## Executed validation

Both existing checking scripts passed without modification. [check-runs.json](../../integration/2026-10-28-coupled-dynamics/check-runs.json) records their source hashes and executions. [stability-checks.json](../../integration/2026-10-28-coupled-dynamics/stability-checks.json) records exact stationary gradients, the positive Hessian certificate, inverse action, reduced and full Hessian identities, and both rational thresholds. Floating-point eigenvalues are supplementary spot checks, not the proof of the stability criterion.

The separate [matrix derivative check](../../integration/2026-10-28-coupled-dynamics/matrix-flow-checks.json) tested one noncommuting two-dimensional example with a correlated prior: 57 matter coordinates and one additional full-row coordinate. Maximum matter derivative errors were below $2.4\times10^{-9}$; the row error was below $6.1\times10^{-10}$; dissipation residuals were below $5.7\times10^{-14}$. These finite checks support derivative order and signs, not a general theorem by numerical sampling. No trajectory experiment, simulator test suite, training, GPU work, or Lean work was performed.

Sequential TeX Live builds succeeded for the main manuscript and companion, producing 290 and 117 pages. References and citations resolved. The companion has no overfull or underfull boxes. The main retains three preexisting overfull boxes of 9.51393 pt, 20.93712 pt, and 9.39738 pt outside the edited chapter, and has one non-clipping underfull paragraph in the edited chapter. [build-results.json](../../integration/2026-10-28-coupled-dynamics/build-results.json) records the build summary. The main's 25 rendered pages and the companion's 14 rendered pages, including contents and adjacent chapter transitions, were visually inspected. No clipping, equation overlap, or missing glyphs was observed there. Text-span bounds were also checked across all 407 pages with no physical-page violations. The visual review does not claim a fresh inspection of every unchanged page.

The checked PDFs were copied into the canonical manuscript folder. [completion-validation.json](../../integration/2026-10-28-coupled-dynamics/completion-validation.json) records their hashes, unchanged-source and WIP checks, and the final change manifest. Of the 168 protected WIP files, 167 retain their complete baseline bytes; the existing Research daily log retains its byte-identical original prefix with this session's entry appended. All 13 Item 7 files, its backlog section, and 77 manuscript files outside the authorized targets are unchanged. Research notes and checking scripts from October 26 remain unchanged. The first final-manifest check misdecoded a Unicode path using the Windows default encoding; explicit UTF-8 decoding corrected the check without changing that file.

## Remaining scope

The nonlinear center direction at $T_c$, stationary branches and destinations on graphs with cycles, nonlinear dictionaries, unrestricted law-space well-posedness, and endogenous formation remain open. The new result resolves the coupled Gaussian flow and one complete local stability example. Publication remains a separate step after the requested report.
