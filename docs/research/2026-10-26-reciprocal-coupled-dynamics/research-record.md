# Coupled reciprocal dynamics, 2026-10-26

The authorized task is to derive coupled Fisher dynamics for means, covariances, and Gibbs source rows in the reciprocal law potential, then characterize stationary points and local stability in a small explicit system. Dictionary recognition variables remain dynamic wherever the declared potential couples them. This is a bounded investigation of Item 2's `it2:open-reciprocal`, not integration into the live manuscript or a formation claim. Item 7 and the Research vault remain read-only.

## Existing results and exact gap

The live `chapters/03_dynamics.tex` already defines the general-law reciprocal potential, proves the exact Gibbs row gap and envelope identity, specifies full row relaxation and Gibbs-reduced dynamics, proves the row-block Schur-complement inertia identity, and defines the restricted Gaussian Fisher flow. Its Gaussian noninvariance proposition forbids treating that restricted flow as a closed full-law evolution. The live `chapters/18_link_dynamics.tex` adds factorized dictionary recognition to a tangent communication potential, gives the fixed-covariance and fixed-row mean flow, and computes a source-role covariance derivative and a frozen-variable scalar stationary-variance result. These statements are the starting point.

The remaining obligation is the complete coupled vector field and an application of the critical-point criteria when content and dictionary uncertainties move with source selection. The row-Hessian identity is already known in this manuscript. Its explicit negative cost-gradient covariance form is ordinary differentiation of a log-sum-exp; its role in a concrete fully coupled stability calculation is the target here.

## Declared scope

The general part uses finite source sets with strictly positive fixed reference probabilities and temperatures, regular normalized content and dictionary families on standard Borel spaces, fixed reference laws and kernels, finite relative entropies, and justified differentiation. Both direct Fisher relaxation of rows and instantaneous Gibbs minimization are specified. The reduced flow uses the existing matter-family Fisher metric, without silently replacing it by the metric induced by the Gibbs-row graph.

The explicit realization uses the manuscript's Gaussian tangent communication family with fixed operating contents, fixed linear codes, positive message-noise covariance, and a positive-definite normalized dictionary prior. Content means and covariances and factorized dictionary means and covariances all evolve. Any simplification of graph, dimension, or loop prior is stated in the construction. Full-space stability requires all free variables, not only a symmetry-restricted calculation. Source priors and reference models remain constitutive data.

## Work and stopping rule

Two constructive derivations run in parallel: the complete flow and stationary conditions, and a finite explicit stability example. They write separate notes. Root handles source continuity, exact symbolic or tiny numerical checks, and the final synthesis. Agents run no scripts; root runs at most one bounded CPU calculation at a time, with at most four threads, no process pools, no GPU, and no Lean.

Use at most three substantive rounds and stop earlier if two consecutive rounds yield no checkable mathematical progress. The deliverable is a derivation with explicit assumptions and a proved finite result or precisely located remaining obligation. No automatic audit/adjudication panel, manuscript edit, commit, publication, or background continuation is part of this task.

## Progress

Round 1 read the live sources and preserved the existing row critical-structure theorem. The two constructive derivations produced the complete Gaussian Fisher equations and a three-agent exact stationary example. The general derivation includes compactness and global forward existence under fixed proper references, positive temperatures, and regular Gaussian initial conditions. The local example includes all ten content/dictionary mean and variance coordinates and, for full row relaxation, the eleventh source-row coordinate.

Round 2 checked and consolidated the derivations. Exact symbolic arithmetic independently reconstructed the scalar potential, all stationary gradients, the positive sum-of-squares certificate, inverse Hessian action, Gibbs-reduced Hessian, and full row block. The first calculation was extended to compare frozen and changing variances, then rerun for that new check. It proved $31/122<3/10<3269/10118$. A separate bounded matrix finite-difference check covered all 57 matter coordinates and the extra row coordinate at a noncommuting two-dimensional example with a correlated prior; derivative and dissipation residuals passed the recorded tolerances. No substantive failed ansatz or unresolved algebraic discrepancy occurred. The initial final-document patch encountered a changed closing paragraph in the completed worker note; rereading that paragraph resolved the patch context without changing the mathematics.

The requested finite-family local result is complete away from the critical temperature; a third substantive round is unnecessary for this bounded task. The nonlinear critical direction, long-time destinations, cycles, unrestricted law-space well-posedness, and formation are open extensions. [REPORT.md](REPORT.md) supplies the result, evidence, and limits. The Research vault and Item 7 remain read-only. No integration or publication is included in this investigation.
