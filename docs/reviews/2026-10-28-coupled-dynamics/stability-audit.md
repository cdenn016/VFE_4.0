# Scalar coupled-stability audit, 2026-10-28

No numerical-analysis findings. I found no mathematical defect in the stated scalar stability example or a discrepancy between its formulas and the executable symbolic check. This conclusion concerns the declared Gaussian tangent family with fixed channel and prior data, all ten content/dictionary mean and variance coordinates free, and an optional eleventh finite-rate row coordinate. It does not extend to unrestricted law-space dynamics or nonlinear group-valued dictionaries.

The inspected derivation is `docs/research/2026-10-26-reciprocal-coupled-dynamics/stability-example.md`, particularly lines 36–106, 110–120, 124–209, and 213–276. I read `check_stability.py` in full and its saved `stability-checks.json`, and compared the associated claims in `full-flow.md` and `REPORT.md`. Source continuity used the live `Research/manuscripts/MAgent_Theory_2.0/AUTHORING.md`, Chapter 18's `it2:model`, `it2:potential`, `it2:potential-forces`, `it2:certainty`, and `it2:open-reciprocal`, and Chapter 3's Gibbs/envelope identities, `prop:dyn-morse`, `def:dyn-restricted-flow`, and Gaussian noninvariance result. The wiki index and natural-gradient entry were consulted for the metric convention. No source or manuscript was edited by this investigator. No script, GPU job, Lean work, or numerical experiment was run; saved numerical outputs were inspected as prior evidence, not reported as a fresh execution.

## Independent reconstruction

For one edge of weight $1/2$, set $d_j=s_j+\epsilon z_j$, $S_0=2+\epsilon h_0$, $S_j=2+\epsilon h_j$, and $V_j=1+\epsilon k_j$, with $s_j\in\{1,-1\}$. Expansion of the actual rational and logarithmic cost gives the Hessian quadratic contribution

\[
\frac14z_j^2-\frac{s_j}{4}z_jh_j+
\frac1{16}h_0^2-\frac18h_0h_j+
\frac3{16}h_j^2-\frac18h_jk_j.
\]

These coefficients agree with equation (12) of the derivation. Adding the self mean terms and the five self variance terms, then completing squares, gives its equation (11). In particular, the combined coefficient of $h_j^2$ on the completed-square side is $1/16+1/16+71/128+1/128=11/16$, and that of $h_0^2$ is $1/2+2/16=5/8$. The displayed strictly positive mean and variance terms force every component of a zero vector of the quadratic form to vanish. Positivity is therefore established on all ten coordinates before the later symmetry calculation.

The first derivatives also cancel as written. The two source self mean forces are $(1/4,-1/4)$ and their weighted source-role forces are $(-1/4,1/4)$. The dictionary means have the same cancellation. Each source variance has self derivative $1/8$ and weighted edge derivative $-1/8$; each dictionary variance has self derivative $-1/8$ and weighted edge derivative $1/8$. Receiver covariance forces vanish, and its two mean forces cancel. Equal costs make $b_*=1/2$ stationary for every positive temperature.

The inverse-action computation can be checked without a matrix inverse. The stated reduced three-variable system gives

\[
122t-31u=124,\qquad -16t+87u=-28,\qquad w=\frac12+\frac u8.
\]

Eliminating $u$ yields $10118t=9920$, so

\[
(t,u,w)=\frac1{5059}(4960,-716,2440).
\]

The minimizing mean coordinates are $(x_0,y,z)=(15,-6,-10)t/31$, yielding the full vector in equation (13). Substitution into the linear functional gives

\[
g^TH^{-1}g=t-\frac u2+\frac w2
=\frac{4960+358+1220}{5059}
=\frac{6538}{5059}.
\]

The use of symmetry only evaluates this inverse action; it does not restrict the positivity or stability claim. With the variances fixed, the scalar quadratic coefficient is $61/62$, so the inverse quadratic form is $62/61$ and the separate mean-only threshold is $31/122$.

The Gibbs Hessian correction at the equal-cost point is $-gg^T/(4T)$. Congruence by $H^{-1/2}$ gives a rank-one perturbation of the identity, with nine positive eigenvalues and the remaining sign determined by $1-6538/(20236T)$. This yields $T_c=3269/10118$. The full row entropy has second derivative $4T$ at $b=1/2$; its mixed block is $g$, giving the displayed eleven-dimensional completion of squares. Thus both row conventions have the same inertia and different positive mobility matrices.

At a critical point the mobility derivative multiplies a zero gradient. Consequently the linearization is similar to $-M_*^{1/2}H_TM_*^{1/2}$, so its eigenvalues are real and have the opposite inertia. Above the threshold, continuity of the positive Hessian and mobility gives a neighborhood in which the energy difference is comparable to squared displacement and its derivative is bounded above by a negative multiple of itself, proving local exponential stability. Below the threshold, the nondegenerate Hessian has one negative direction, hence the Jacobian has one positive eigenvalue and no zero eigenvalues. The claimed hyperbolic saddle classification follows. This proof uses the exact Hessian; it does not infer signs from rounded numerical eigenvalues.

## Executable check and numerical scope

The symbolic code declares all ten matter variables before differentiation, adds the row variable for the full Hessian, retains the expected dictionary variance term in each edge cost, and uses exact `Rational` constants for the model and stationary point. Its principal-minor and sum-of-squares checks address the entire fixed-row Hessian. Its reduced Hessian is differentiated from the logarithmic objective, not only assigned from the proposed rank-one identity. The floating-point eigenvalue spot checks use the symmetric similarity representatives with the stationary Fisher mobilities; the use of elementwise square root is valid here because those mobility matrices are explicitly diagonal. The fixed-variance mean block has identity mobility at the selected point.

The saved output reports the same rational thresholds, zero gradients, and Hessian identities. Its floating-point eigenvalues at three temperatures are supplementary checks and are labeled that way in both the derivation and the report. I did not reproduce those numerical outputs in this no-script audit, and make no independent finite-precision accuracy or scaling claim. No ill-conditioned large-system computation or discretized dynamics is part of the claimed result.

## Integration conditions

Integrate the exact objective, state order, reference parameters, fixed operating map, positive noise, product recognition, and the complete positive Hessian certificate with the result. State that the receiver has precisely the two peer candidates in this example; adding an identity self candidate changes its Gibbs problem. Retain the distinction between the reduced matter Fisher metric and the pullback metric of a Gibbs graph. The full critical null vector is $(H^{-1}g,-1)$ at $T_c$, while its reduced matter component is $H^{-1}g$; making this distinction explicit would improve the explanatory companion. This is a presentation clarification, not a defect in the displayed inertia argument.

Keep the critical-temperature nonlinear classification, selected branches, trajectory destinations, cycles, unrestricted law-space behavior, and formation open. The exact inequality $31/122<3/10<3269/10118$ supports the failure of transferring frozen-variance stability to this complete restricted-family system. It does not establish monotone uncertainty growth or a general temperature-ordering theorem for arbitrary stationary branches.
