# Item 4 continuum proof investigation

The continuum local theorem reconstructs under its stated hypotheses. I found no derivative-loss, self-mapping, continuation, or energy-identity defect in that theorem. One displayed formula contains a definite TeX error. Integration also needs to preserve the restricted scope: the two notes establish coupling of retained collective shape, while independent smoothing of rows, communication links, and frames remains open. This is an investigator report, not an adjudication status.

The audited continuum source is `docs/derivations/2026-10-23-continuum-context-coupling/main.tex`, denoted C below. The finite source is `docs/derivations/2026-10-23-finite-context-coupling/main.tex`, denoted F. Live manuscript paths below are relative to `Research/manuscripts/MAgent_Theory_2.0/`. I read C in full, F for its assumptions, construction, scope, and zero-coupling statements, `AUTHORING.md`, `chapters/02_geometry.tex` at the collective connection, shape pullback, contextual energy, and gauged-flow problem, `chapters/03_dynamics.tex` at the reciprocal potential and flow regularity definitions, and Item 4 in `future_TODO.md`. This investigation used analytical reconstruction and source reads. It ran no mathematical scripts, numerical experiments, Lean, or external theorem-verification process, and changed no source manuscript.

## Exact target and hypothesis match

For every dimension d at least one, every integer s greater than d/2+2, m=s-1, and every fixed positive lambda, C:294-311 asserts local existence, uniqueness, local Lipschitz dependence, and a continuation alternative for the displayed coupled Fisher-gradient system. The base is the fixed flat torus with normalized volume. The statistical configuration is in a finite-dimensional regular family, with positive smooth Fisher metric and a smooth free proper isometric diagonal group action on an invariant region. The initial map lies compactly inside one adapted quotient/fiber chart. The references, evaluators, source priors, and channel rules are fixed in time, and the resulting reciprocal potential V is finite and smooth on a neighborhood of the image. The family is explicitly restricted to product belief/model recognition. V need not descend to the quotient.

Initial data satisfy u0 in H^s and r0,b0 in H^m. The asserted solution space is

\[
u\in C_tH^s\cap L^2_tH^{s+1}\cap H^1_tH^m,
\qquad r,b\in C_tH^m\cap H^1_tH^m,
\]

with b additionally in C^1_tH^m. Equivalence means equality of the chart solution of the negative Fisher-gradient equations and the reconstructed law-field equations, as well as covariance under time-independent passive frame changes carrying the constitutive data. It does not mean equivalence to unrestricted density evolution, per-agent own-block inference, or a graph-refinement limit.

These assumptions are supplied by the note rather than consequences inferred from probability normalization. I introduce no stronger geometric or regularity hypothesis to reconstruct the proof. Extending smooth coefficients off a compact interior neighborhood is a proof device; it does not change the vector field on that neighborhood.

## Definite finding

### C1. The auxiliary normal equation prints a spurious product

Location: C:139-143, especially C:142.

Severity: low; repair before integration.

The source has `...J_p\xi)quad\hbox{for all }\xi`, with no backslash before `quad`. In math mode the four letters are mathematical symbols. The intended equation is

\[
\mathfrak k_p(A^*_\mu,\xi)=-\mathcal G_p(\partial_\mu p,J_p\xi)
\quad\text{for every }\xi\in\mathfrak g.
\]

This is a transcription defect in a load-bearing displayed equation, not a defect in its derivation: differentiating the quadratic contextual energy gives precisely this corrected normal equation, and C:147-153 uses the correct identity. Restore the missing backslash or use ordinary punctuation before the quantifier. The finding would be falsified by a changed source line that correctly separates the formula from its quantifier.

## Reconstructed proof

### Auxiliary elimination and the nonbasic potential

Freeness makes J injective, so the inertia form kfrak(xi,eta)=G(J xi,J eta) is positive definite. For each base direction the contextual energy is a strictly convex quadratic in A_mu. Its first variation is G(partial_mu p+J A_mu,J xi), which proves the corrected normal equation. Thus A*=-p*theta and the residual is the horizontal projection of dp (C:135-155).

In the adapted chart, the metric has block form

\[
G=du^T h(u)du+(dr+C(u,r)du)^T k(u,r)(dr+C(u,r)du).
\]

Consequently the reduced contextual energy is lambda/2 times the Dirichlet energy of u. Varying the full energy and pairing with this full metric gives

\[
h u_t+C^T k(r_t+C u_t)=-V_u+\lambda h\tau_h,
\qquad k(r_t+C u_t)=-V_r,
\qquad B b_t=-V_b.
\]

Elimination gives exactly C:214-218, including the term +h^{-1}C^T V_r in the shape equation. This term cannot be dropped for a fixed reference that breaks active symmetry. The polar-coordinate example C:271-288 checks it: with phi=r+eta u, V_r=a u sin(phi), V_u=u-a cos(phi)+a eta u sin(phi), the two eta terms cancel in the radial force and phi_t=-(a/u)sin(phi). At lambda=0 this is the Cartesian flow p_t=-(p-a e1).

The intrinsic fixed-A variation also has the stated sign. For X_mu(p)=J_p A_mu, the variation of the covariant derivative is nabla_mu U+nabla_U X_mu. Since X_mu is Killing, the adjoint of nabla X_mu is minus itself. Periodic integration by parts therefore gives negative gradient

\[
\tau_A=\sum_\mu(\nabla_\mu v_\mu+\nabla_{v_\mu}X_\mu).
\]

The envelope term from differentiating A* vanishes by stationarity. Its reduced gradient is horizontal and equals H_a tau_h^a, as in C:248-253. No uniform parabolicity in every law coordinate has been assumed.

### Heat estimate and smooth compositions

For nonzero Fourier modes, the energy estimate in C:338-345 controls the H^{s+1} norm because

\[
|k|^2(1+|k|^2)^s\geq\tfrac12(1+|k|^2)^{s+1}.
\]

The forcing pairing is controlled by H^{s-1} using (1+|k|^2)/|k|^2 at most two. The zero mode is an ordinary integral of its forcing and is bounded uniformly for T at most one. The equation then bounds v_t in L^2 H^{s-1}. Fourier truncation and the same estimate give the claimed strong solution and continuity into H^s. The semigroup displacement estimate follows directly from (1-exp(-a))^2 at most a and has the required H^s to H^{s-1} loss. Thus C:316-348 does not rely on false operator-norm continuity of the heat semigroup on H^s.

For the nonlinear estimates, m>d/2 makes H^m an algebra, while partial_mu u is in H^m because u is in H^{m+1}. Smooth compositions are locally Lipschitz on bounded H^m sets with compact pointwise image. One way to supply the derivative-product step abbreviated at C:371 is to differentiate a smooth coefficient N up to order m. A resulting product has factors partial^{alpha_j}z with the sum of derivative orders at most m. Sobolev interpolation places the factor of derivative order j in L^{2m/j}; Holder's inequality then puts the product in L^2. The H^m and L-infinity bounds control the constants. Apply this estimate to DN along the segment between two fields in a smooth coefficient extension, and multiply by their difference using the H^m algebra bound. Fixed smooth periodic x dependence adds only bounded coefficient derivatives. Therefore F,C,Q,L satisfy precisely C:378-387. No spatial derivative of r or b enters F.

### Self-mapping and contraction

On a closed path set in X_T=C_tH^s times C_tH^m times C_tH^m, impose a fixed norm bound and small C_tH^m displacement from the initial data. Solve the heat equation for u+ with forcing F(z), then integrate the fiber equation using the new derivative u+_t. This choice is essential: an input path in X_T does not have a time derivative.

The uniform bound A_T in C:406-407 controls u+_t in L^2H^m. The fiber displacement is bounded by a constant times sqrt(T) M_C A_T plus T M_Q, while the shape displacement in H^m is bounded by sqrt(lambda T)||u0||_{H^s}+T M_F. The row displacement is T M_L. Every displacement tends to zero for fixed bounded initial norm. Sobolev embedding then keeps the pointwise image inside the regular chart and maintains positive metric bounds. The shape H^s bound is the initial norm plus O(sqrt(T)), so choosing a strictly larger radius gives the asserted self-mapping.

For equal initial data, the heat difference estimate gives

\[
\|\delta u^+\|_{C H^s}+\|\delta u_t^+\|_{L^2H^m}
\leq K_\lambda L_F\sqrt T\|\delta z\|_{X_T}.
\]

The fiber difference is the sum of C(z)delta u+_t, delta C times the other u+_t, and delta Q. Integration and the algebra estimate bound it by

\[
\bigl(c M_CK_\lambda L_F T+c L_C A_T\sqrt T+L_QT\bigr)
\|\delta z\|_{X_T}.
\]

The row difference is at most L_L T times the input difference. The complete contraction factor tends to zero. In particular, the factor multiplying the potentially dangerous delta C times u_t term is sqrt(T); u_t is not being assumed uniformly bounded in C_tH^m. This establishes the claimed fixed point without derivative loss.

The heat estimate gives the extra L^2H^{s+1} and H^1H^m regularity of u; substitution gives H^1H^m regularity of r,b. The row right side is continuous into H^m, so b is C^1_tH^m. Retaining the initial-data differences in these estimates gives Lipschitz dependence in X_T and in the heat norms. Substitution gives it in the H^1H^m fiber and row norms as claimed. Solutions in the stated regularity class obey these same integral equations; short-interval uniqueness and concatenation give uniqueness on their common interval.

### Uniform continuation time

A uniform bound on the H^s/H^m norms and a fixed compact interior image give uniform bounds for F,C,Q,L and their Lipschitz constants on a neighborhood. The shape displacement estimate uses the bounded H^s initial norm with an explicit sqrt(T) rate into H^m. The fiber displacement uses sqrt(T) times a uniformly bounded A_T. Thus the same positive restart interval works at every late starting time. Starting within that interval of a proposed finite endpoint extends the solution beyond it by uniqueness. This verifies C:441; no compactness of bounded subsets of H^s and no uniform H^s semigroup displacement estimate is needed.

### Energy identity and reconstruction

The energy is a C^1 functional on the H^m chart neighborhood because m>d/2+1: both the field and its first spatial derivatives have the needed boundedness and product estimates. The solution is H^1 in time into H^m, so the Banach-space chain rule gives absolute continuity of the energy. Differentiating the Dirichlet energy produces h_u u_t (du,du) and 2h(du,du_t), which are time integrable. Periodic spatial integration by parts gives C:463-465. Substituting the full block equations yields

\[
\frac{d\mathcal F_*}{dt}
=-\int_C\left[u_t^T h u_t+V_r^T k^{-1}V_r+V_b^T B^{-1}V_b\right]d\mu_C.
\]

This is the squared norm in the original coupled metric, not a mistaken sum of uncoupled u and r kinetic terms. It proves exactly the dissipation stated in C:446-467.

Smooth composition gives p=Psi(u,r) in C_tH^m and H^1_tH^m. For A*=-theta(p)dp, one spatial derivative gives C_tH^{m-1}. Its time derivative consists of Dtheta(p)[p_t]dp and theta(p)dp_t. Since m-1>d/2, the product estimates give L^2_tH^{m-1}; hence the reconstructed connection has the stated H^1 regularity. The residual cancels its vertical part algebraically and is H_a partial_mu u^a in C_tH^m. These are exactly the assertions in C:469-477. They do not add an independent connection evolution equation.

### Exact witness and zero coupling

For the witness, the local Gaussian KL equals one half of ||mu_i-nu_i|| squared plus one half of ||nu_i|| squared plus one; the constant follows from the two-dimensional model covariance in the evaluator expectation. The two directed peer costs sum to the two squared inter-agent mean differences. Their Hessian is the symmetric matrix H in C:510-512.

With tau=(sqrt(5)-1)/2 and v=(1,tau,-1,-tau), the identity tau squared plus tau equals one gives Hv=(5-tau)v and ||e1 v^T|| squared=5-sqrt(5). For P=R_phi a V0, the minimizing connection is -phi_x J. Consequently D_xP=R_phi a_xV0 and D_xD_xP=R_phi a_xxV0. The full matrix flow reduces to a_t=lambda a_xx-kappa a with phi_t=0, so C:564 is an exact solution. Its positive lower bound keeps the collective configuration away from the singular zero orbit on every fixed finite time interval. Averaging cos squared and sin squared gives C:585-595 and exact dissipation. The n=1 versus n=2 comparison has identical P and P_x at x=0 and velocity difference -3 lambda a1 R_phi(0)V0, as claimed. Singleton rows do not test the nontrivial row equation; C:597 states that limitation.

An arbitrary smooth periodic real phase need not have its entire image in one group-coordinate chart. This does not refute the displayed explicit solution: direct substitution proves it independently of the one-chart theorem. A small phase amplitude, for example absolute epsilon less than pi/2 in phi=epsilon sin(x), supplies a witness fitting one angle chart on every fixed finite interval. No additional theorem claim is needed.

At lambda=0, the finite equations simply remove the contextual force. In the continuum the vector field is a locally Lipschitz ordinary differential equation on the all-H^m chart, and no auxiliary A is selected by the zero-coupling energy. These claims agree with F:248-267 and C:481-493. The regularity counterexample is valid: a unit-covariance Gaussian location family has Euclidean Fisher metric, while a normalized correlated Gaussian reference with precision matrix [[1,-a],[-a,1]], 0<a<1, gives the displayed quadratic potential. Its solution u(t)=exp(-t)[cosh(at)u0+sinh(at)r0] transfers an H^m but non-H^s component into u for every t>0. As C:493 explicitly says, this example is for the broader regular statistical-gradient class; it is not represented as a necessary special case of the fixed multi-agent potential. Thus there is no unsupported uniform positive-lambda to zero-lambda regularity claim.

## Integration conditions and remaining scope

The auxiliary normal equation, product-Fisher metric, and quotient energy agree with the live manuscript's `chapters/02_geometry.tex`:192-244 and :256-265. The note's stronger free/proper hypothesis is stated explicitly; it does not silently replace the manuscript's more general constant-rank connection class. The reciprocal law potential, source and channel derivatives, product restriction, and fixed-data distinction agree with `chapters/03_dynamics.tex`:9-52 and :200-229 and with `AUTHORING.md`:31-51. The proof establishes a regular finite-dimensional family extension, not the unrestricted bounded-log-density dynamics elsewhere in that chapter.

There are two editorial scope obligations before integration. First, `chapters/02_geometry.tex`:7 currently says that the connection is diagnostic for the later context-local constructions. A new contextual-flow section must distinguish its new dynamical coupling from those unchanged later examples. Its scoped theorem can replace part of `open:geo-gauged-flow`, while global continuation, other bases, and unrestricted family questions remain open. Second, `future_TODO.md`:76 asks for smoothness of beliefs, models, rows, links, and frames to be produced by the dynamics. C:314, :479, :601-603 and F:396-402 expressly leave independent row/link/frame smoothing absent. Integration therefore supports partial completion of that item and must not claim all of its original smoothing target has been achieved. F:402's sentence describing the entire continuum problem as remaining should also be reconciled with the new conditional continuum theorem when the two notes are presented together.

The strongest supported conclusion from this investigation is the reconstructed local well-posedness and exact dissipation theorem on the declared regular finite-dimensional chart over the flat torus, together with its exact two-sector witness. There is no unresolved analytical step in this scoped reconstruction. The next action that changes integration readiness is to correct C:142 and integrate with the stated scope qualifications; a claim of complete Item 4 closure would require an additional mechanism and theorem for the independently requested row/link/frame smoothing.
