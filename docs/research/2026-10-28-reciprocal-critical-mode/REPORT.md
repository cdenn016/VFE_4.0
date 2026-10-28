# Critical-mode result for the coupled reciprocal example

The three-agent Gaussian tangent example has a subcritical pitchfork at

\[
T_c=\frac{3269}{10118}.
\]

Two nearby asymmetric equilibrium branches exist for temperatures just above this value, and both are saddles with exactly one unstable direction. The symmetric equilibrium is nonlinearly unstable at the critical temperature. No locally stable asymmetric pair is born as temperature decreases through the threshold. These statements allow all five means and all five variances to vary and hold for both declared Fisher flows.

## Exact local normal forms

Keep the references, positive unit message noise, fixed operating map, and exactly two peer candidates of the published example. The receiver has no added identity self candidate. The matter coordinate order is

\[
x=(\mu_0,\mu_1,\mu_2,a_1,a_2,v_0,v_1,v_2,V_1,V_2),
\]

and the symmetric state is

\[
x_*=(0,-4/3,4/3,1/3,-1/3,1,1,1,1,1).
\]

Write $\delta=T-T_c$. For full row relaxation use the physical row imbalance $s=b-1/2$. For Gibbs reduction use the observable imbalance $s=b^*(x,T)-1/2$ on its center manifold. In the original time variable, the respective local invariant reductions are

\[
\begin{aligned}
\dot s_{\mathrm{full}}&=-\frac{4\delta s+c s^3}{D+4}
 +O(\delta^2|s|+|\delta||s|^3+|s|^5),\\
\dot s_{\mathrm{red}}&=-\frac{4\delta s+c s^3}{D}
 +O(\delta^2|s|+|\delta||s|^3+|s|^5),
\end{aligned}
\]

where

\[
c=-\frac{657534393590529472}{6223404588394626861}
\approx-0.1056550934864,
\qquad
D=\frac{19189456}{25593481}>0.
\]

The remainder bounds are uniform on sufficiently small state-parameter neighborhoods. Finite-smoothness equivariant center manifolds suffice; analyticity of a center manifold is not assumed. The hypotheses of the parameter-dependent equivariant existence theorem are checked against [Ruelle, Theorem 1.2, pp. 138–139](https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B39%5D.pdf) in the dynamical derivation. The nonzero coefficients above are computed from this example, not taken from that general theorem.

The numerical values of the exact coefficients are

| Flow | Coefficient of $\delta s$ | Coefficient of $s^3$ |
| --- | ---: | ---: |
| Gibbs-reduced | $-5.334904960307369$ | $0.1409149705805845$ |
| Full row relaxation | $-0.8421444352731884$ | $0.022244212259460213$ |

For the common physical amplitude, full relaxation has $D/(D+4)=4797364/30390845$ times the leading center velocity of Gibbs reduction. Equivalently, its leading time scale is about $6.34$ times longer. This identifies neither higher-order trajectories nor stable transients between the systems. The reduced flow retains the matter-family Fisher metric specified in Chapter 18; adding the pullback of the row metric would change that flow.

## Nearby branches and stability

There are state and temperature neighborhoods in which the symmetric equilibrium is the only equilibrium for $\delta\leq0$. For every sufficiently small $\delta>0$ there are exactly two additional equilibria. Their row imbalances are

\[
s_\pm(\delta)=\pm\sqrt{-\frac4c}\sqrt{\delta}
 +O(\delta^{3/2})
 =\pm6.152969698825556\sqrt{\delta}+O(\delta^{3/2}).
\]

The matter coordinates satisfy $x_\pm=X(s_\pm)$, where $X$ is the unique local solution of the ten matter stationary equations at fixed row. In particular,

\[
X(s)=x_*+ps+\tfrac12rs^2+O(s^3),
\qquad
p=\frac1{5059}(-2400,960,960,1600,1600,0,716,-716,-2440,2440)^T.
\]

The complete exact $r=X''(0)$ is given in the stationary derivation and checking output. Source variances and both dictionary variances participate already at first order. Fixing them would solve a different problem.

| Equilibrium | Temperature | Reduced flow | Full flow |
| --- | --- | --- | --- |
| Symmetric | $T>T_c$ | Locally exponentially stable | Locally exponentially stable |
| Symmetric | $T=T_c$ | Nonlinearly unstable; nine stable linear modes | Nonlinearly unstable; ten stable linear modes |
| Symmetric | $T<T_c$ | One unstable and nine stable modes | One unstable and ten stable modes |
| Either nearby asymmetric branch | $T>T_c$ | One unstable and nine stable modes | One unstable and ten stable modes |

At either asymmetric branch the small unstable eigenvalue is $8\delta/d+O(\delta^2)$, where $d=D$ for Gibbs reduction and $d=D+4$ for full relaxation. At the critical temperature the scalar drift is $(-c/d)s^3+O(s^5)$, which points away from zero on both sides. With the destabilizing parameter $T_c-T$, this is the subcritical pitchfork convention.

## Load-bearing calculation

Let $A=F+(C_1+C_2)/2$, $\Delta=C_1-C_2$, $H=A_{xx}(x_*)\succ0$, and $g=\Delta_x(x_*)$. The matter stationary curve solves $A_x(X)+s\Delta_x(X)=0$. Its tangent is $p=-H^{-1}g$. Set $k=A_{xxx}[p,p]+2\Delta_{xx}p$, so $r=-H^{-1}k$. For the stationary scalar potential $\Phi_T(s)=A(X(s))+s\Delta(X(s))+Th(s)$, exact expansion gives

\[
\Phi_{T_c}''''(0)
=A_{xxxx}[p,p,p,p]+4\Delta_{xxx}[p,p,p]
 -3k^TH^{-1}k+32T_c
=6c<0.
\]

The negative term $-3k^TH^{-1}k$ comes from the stable matter response. Keeping only the straight critical line would omit it. Evenness under reflection and exchange makes $\Phi_T'(s)=s q(s^2,\delta)$, with $q(0,\delta)=4\delta$ and $q_z(0,0)=c<0$. An analytic implicit-function argument then proves the complete local branch count. Completing squares in the full Hessian proves the displayed stability indices, and positive Fisher mobilities preserve their sign interpretation.

The stationary curve is not an invariant center manifold. Writing its dynamical correction as $w=Y-X$ and $B=H^{-1}M_*^{-1}p$, matter invariance forces the leading response $w=-B\dot s$, with $g^TB=-D$. The Gibbs row constraint yields the denominator $D$; the full row equation yields $D+4$. In particular, at criticality the two invariant graphs differ from $X$ by $(c/d)Bs^3+O(s^4)$. This is why stationary elimination alone cannot supply the kinetic coefficients.

## Evidence and scope

[stationary-reduction.md](stationary-reduction.md) gives the complete objective, exact symmetry, implicit-function reduction, rational fourth-order calculation, local branch theorem, and Hessian and Lyapunov arguments. [dynamics-reduction.md](dynamics-reduction.md) gives the equivariant center-manifold hypotheses, invariance equations, physical amplitude conventions, kinetic coefficients, and stability consequences. These are constructive derivations, not an adversarial audit or a verifier-panel decision.

[check_critical_mode.py](check_critical_mode.py) completed successfully with `C:/Python314/python.exe`, using exact SymPy arithmetic in 0.454 seconds. [critical-mode-checks.json](critical-mode-checks.json) records the rational jets and coefficients. The checks reconstruct the quartic coefficient a second way from $X'''(0)$, verify the exact reflection/exchange symmetry, verify the leading center-manifold equations at orders $s^3$ and $\delta s$, and recover the linear parameter coefficients by normalized left-null projection of the actual Jacobians. This was one small symbolic calculation with at most four library threads, with no trajectory integration, parameter scan, GPU work, or Lean work. Exact local algebra is not evidence for global behavior.

The initial source and protected-file hashes are recorded in [source-baseline.json](source-baseline.json); the final comparison is recorded in [completion-validation.json](completion-validation.json). The work adds this local research packet and updates the VFE daily change log. It does not integrate or publish the result. Research manuscript sources, Item 7, simulation code/configuration, and unrelated protected WIP remain outside the edit scope.

The requested local critical-mode obligation is resolved for this fixed Gaussian tangent example. Distant stable equilibria, folds, global destinations after leaving the neighborhood, graph-cycle effects, unrestricted probability-law dynamics, and formation remain separate questions. The local theorem does not establish hysteresis or selection of a stable group.
