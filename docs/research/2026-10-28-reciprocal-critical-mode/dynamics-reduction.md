# The actual critical dynamics under full and Gibbs-reduced Fisher metrics

This note resolves the dynamical part of the local critical-mode calculation for the specified three-agent Gaussian tangent example. It uses all ten matter coordinates. The result is a subcritical pitchfork for both continuous Fisher flows: two nearby asymmetric saddles exist on the high-temperature side, and the symmetric critical point is nonlinearly unstable at the threshold. The two flows have different leading time scales.

## Target, assumptions, and state space

The supplied system is the example in `docs/research/2026-10-26-reciprocal-coupled-dynamics/stability-example.md` and the live manuscript's Chapter 18, `it2:coupled-threshold`. Its fixed Gaussian references, two-peer row, positive unit message noise, fixed operating maps, and unit Fisher mobilities are retained. No mean or variance is frozen. The state is

\[
x=(\mu_0,\mu_1,\mu_2,a_1,a_2,v_0,v_1,v_2,V_1,V_2)
\in\mathbb R^5\times(0,\infty)^5,
\qquad s=b-\tfrac12\in(-\tfrac12,\tfrac12).
\]

Write

\[
A(x)=F(x)+\tfrac12(C_1(x)+C_2(x)),\qquad
\Delta(x)=C_1(x)-C_2(x),\qquad
G(x,s)=A(x)+s\Delta(x),
\]

where the costs and self terms are precisely those of the supplied example. Define

\[
h(s)=(\tfrac12+s)\log(1+2s)+(\tfrac12-s)\log(1-2s),
\qquad
L(s)=h'(s)=\log\frac{1+2s}{1-2s}.
\]

The full potential is \(\mathcal E_T(x,s)=G(x,s)+Th(s)\). Its continuous deterministic dynamics, with original time parameter \(t\), is

\[
\dot x=-M(x)G_x(x,s),\qquad
\dot s=-(\tfrac14-s^2)[\Delta(x)+TL(s)].
\tag{1}
\]

The Gibbs-reduced dynamics is

\[
\dot x=-M(x)G_x(x,s^*(x,T)),\qquad
s^*(x,T)=\frac{1}{1+e^{\Delta(x)/T}}-\frac12,
\tag{2}
\]

with the supplied matter inverse Fisher metric

\[
M(x)=\operatorname{diag}(v_0,v_1,v_2,V_1,V_2,
2v_0^2,2v_1^2,2v_2^2,2V_1^2,2V_2^2).
\]

There is no stochastic term or time discretization. Both fields are analytic on their open domains for \(T>0\). All analysis below takes place in a sufficiently small interior neighborhood of the specified equilibrium, so no boundary condition or extension of the dynamics to zero variance or zero row weight is used. The already proved interior global-existence statement for this Gaussian example is not needed to infer the local bifurcation.

The target is to find neighborhoods and constants such that, for every sufficiently small \(\delta=T-T_c\), the two systems admit local one-dimensional invariant reductions at fixed \(\delta\), compute their first nonzero nonlinear term in the original time, and classify every equilibrium sufficiently near the symmetric branch. The intended comparison is exact agreement of nearby equilibrium configurations and their stability indices, together with a comparison of leading center velocities. Equality of transient paths is not asserted.

The only auxiliary choices introduced here are a finite-regularity local center manifold and its coordinate chart. These choices do not alter the coefficients computed below. The Gaussian restriction and the kinetic conventions are supplied assumptions, not conclusions about unrestricted law dynamics.

## Stationary data used by the dynamical reduction

At

\[
x_*=(0,-4/3,4/3,1/3,-1/3,1,1,1,1,1),
\qquad T_c=\frac{3269}{10118},
\]

let \(H=A_{xx}(x_*)\), \(g=\Delta_x(x_*)\), and \(p=-H^{-1}g\). The established positivity certificate gives \(H\succ0\), and

\[
p=\frac1{5059}(-2400,960,960,1600,1600,0,716,-716,-2440,2440)^T,
\qquad g^Tp=-4T_c.
\tag{3}
\]

The implicit-function theorem gives a unique local analytic stationary matter curve \(X(s)\) satisfying

\[
G_x(X(s),s)=0,\qquad X(0)=x_*,\qquad X'(0)=p.
\tag{4}
\]

This is a stationary elimination at fixed row, with no invariance assertion. Set

\[
\Phi(s,T)=G(X(s),s)+Th(s).
\]

The stationary calculation supplied by the coordinator gives

\[
\Phi_s(s,T_c+\delta)=4\delta s+c s^3
+O(|\delta||s|^3+|s|^5),
\qquad
c=-\frac{657534393590529472}{6223404588394626861}<0.
\tag{5}
\]

Its value is approximately \(-0.1056550934864\). For a checkable formula for this input, if

\[
k=A_{xxx}[p,p]+2\Delta_{xx}p,
\]

then differentiating (4) twice gives \(X''(0)=-H^{-1}k\), and four differentiations of the stationary potential give

\[
6c=A_{xxxx}[p,p,p,p]+4\Delta_{xxx}[p,p,p]
-3k^TH^{-1}k+32T_c
=-\frac{1315068787181058944}{2074468196131542287}.
\tag{6}
\]

The exact contractions and the complete \(X''(0)\) are recorded in `critical-mode-checks.json`. Equation (6) includes the response of every stable matter coordinate; it is not the derivative along a straight critical line.

The metric quantity needed for the actual dynamics is

\[
D=p^TM_*^{-1}p
=\frac{19189456}{25593481}>0,
\qquad M_*=M(x_*).
\tag{7}
\]

For example its numerator is checked directly from (3):

\[
D=\frac{2400^2+2(960)^2+2(1600)^2+716^2+2440^2}{5059^2}.
\]

## Symmetry and existence of a valid invariant reduction

Define the linear reflection/exchange \(R\) by

\[
R x=(-\mu_0,-\mu_2,-\mu_1,-a_2,-a_1,
v_0,v_2,v_1,V_2,V_1).
\]

It fixes \(x_*\), exchanges the costs, preserves \(A\), and satisfies

\[
\Delta(Rx)=-\Delta(x),\qquad
M(Rx)=RM(x)R^T,\qquad Rp=-p.
\tag{8}
\]

Thus (1) is equivariant under \((x,s)\mapsto(Rx,-s)\), and (2) is equivariant under \(x\mapsto Rx\), with \(s^*(Rx,T)=-s^*(x,T)\). Uniqueness in (4) gives \(X(-s)=RX(s)\), so \(\Phi\) is even.

The center-manifold result used here is the finite-dimensional case of Ruelle's Theorem 1.2: a finite-smoothness parameter-dependent equivariant vector field fixing the origin, with isolated imaginary-axis eigenvalues and the remaining spectrum separated from that axis, has an equivariant local center manifold in state-parameter space. It is tangent to the center eigenspace times the parameter line, contains the equilibrium parameter line and every sufficiently nearby equilibrium, and admits an equivariant parameter-preserving chart. One may choose any fixed finite regularity below that of the data. See [D. Ruelle, *Bifurcations in the Presence of a Symmetry Group*, Theorem 1.2, pp. 138–139](https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B39%5D.pdf).

Here the shifted fields fix the origin for all \(\delta\), are analytic locally, and have an orthogonal \(\mathbb Z_2\) action. At \(\delta=0\), their Jacobians have a simple zero eigenvalue and respectively nine or ten negative eigenvalues. Thus the theorem applies, after a smooth symmetry-preserving local cutoff if a globally defined field is required. Choose a \(C^7\) local reduction; no analytic center manifold is assumed.

The reduced Hessian at \(T_c\) is \(K=H-gg^T/(4T_c)\), with kernel spanned by \(p\). Its nine positive directions give nine strictly negative eigenvalues of \(-M_*K\). The full Hessian has kernel spanned by \((p,1)\) and ten positive directions, giving ten strictly negative eigenvalues with the full inverse metric. These spectral statements follow from similarity to a real symmetric matrix, so there are no additional neutral or oscillatory modes to eliminate.

For the full reduction choose the physical coordinate \(s=b-1/2\). For the reduced flow choose the observable coordinate

\[
s=s^*(x,T)=b^*(x,T)-\tfrac12.
\tag{9}
\]

This is a valid coordinate on its center manifold because

\[
D_xs^*(x_*,T_c)p=-\frac{g^Tp}{4T_c}=1.
\tag{10}
\]

Both choices respect reflection, so the scalar center field \(f(s,\delta)=\dot s\) is odd in \(s\), vanishes at \(s=0\), and has zero linear coefficient at \(\delta=0\). In particular \(f(s,0)=O(s^3)\).

## Why the stationary curve is insufficient, and the correct coefficients

Let \(Y(s,\delta)\) denote the matter coordinates on either selected invariant center manifold in the coordinate just specified. The common matter invariance equation is

\[
Y_s(s,\delta)f(s,\delta)=-M(Y(s,\delta))G_x(Y(s,\delta),s).
\tag{11}
\]

At \(\delta=0\), \(Y(0,0)=x_*\), \(Y_s(0,0)=p\), and the left side is \(O(s^3)\). Expanding the right side and using invertibility of \(H\) yields

\[
Y_{ss}(0,0)=X''(0).
\tag{12}
\]

Consequently the invariant center manifold and stationary matter curve agree through quadratic order at the critical temperature. They need not agree at cubic order.

To compute that difference and the scalar velocity, put

\[
B=H^{-1}M_*^{-1}p,\qquad g^TB=-D,
\qquad w(s,\delta)=Y(s,\delta)-X(s).
\tag{13}
\]

Oddness and smoothness permit

\[
f(s,\delta)=a\delta s+q s^3
+O(\delta^2|s|+|\delta||s|^3+|s|^5).
\tag{14}
\]

In (11), matching the coefficients of \(\delta s\) and \(s^3\) gives, respectively,

\[
p a=-M_*H w_{\delta s},\qquad
p q=-M_*H w_{s^3}.
\]

Terms from the variation of \(M\) multiply a force already of these orders and do not enter either displayed coefficient. Therefore

\[
w_{\delta s}=-aB,\qquad w_{s^3}=-qB,
\qquad
g^Tw_{\delta s}=Da,\qquad g^Tw_{s^3}=Dq.
\tag{15}
\]

In the Gibbs-reduced convention (9), the row stationarity constraint is exact on the whole center manifold:

\[
0=\Delta(Y(s,\delta))+(T_c+\delta)L(s).
\tag{16}
\]

Substituting (5) and (15) into (16), the \(\delta s\) and \(s^3\) coefficients are \(4+Da\) and \(c+Dq\). Hence

\[
\boxed{\dot s=-\frac{4\delta s+c s^3}{D}
+O(\delta^2|s|+|\delta||s|^3+|s|^5).}
\tag{17}
\]

In the full convention the row equation (1) instead gives

\[
a=-\tfrac14(4+Da),\qquad q=-\tfrac14(c+Dq),
\]

and thus

\[
\boxed{\dot s=-\frac{4\delta s+c s^3}{D+4}
+O(\delta^2|s|+|\delta||s|^3+|s|^5).}
\tag{18}
\]

All remainders in (17)–(18) are uniform for sufficiently small \((s,\delta)\). For example odd \(C^7\) dependence gives \(f=s[a_1(\delta)+a_3(\delta)s^2+O(s^4)]\), with \(a_1(\delta)=-4\delta/d+O(\delta^2)\) and \(a_3(\delta)=-c/d+O(\delta)\), where \(d=D\) or \(D+4\). This proves the stated bound without treating an analytic center manifold as given.

The difference from stationary elimination is explicitly nonzero:

\[
Y(s,0)-X(s)=\frac{c}{d}B s^3+O(s^4),
\qquad d=D\ \text{or}\ D+4.
\tag{19}
\]

Here \(B\ne0\) and \(c\ne0\). Merely inserting \(X(s)\) into the full row equation would give the incorrect denominator \(4\), because it would set the matter velocity to zero while the row moves along a curve with nonzero tangent \(p\). Equation (19) supplies the missing matter force.

The left zero modes provide a coordinate-independent check. Normalized to pair to one with the right modes \(p\) and \((p,1)\), they are

\[
\ell_{\mathrm{red}}^T=\frac{p^TM_*^{-1}}D,
\qquad
\ell_{\mathrm{full}}^T=\frac{(p^TM_*^{-1},4)}{D+4}.
\tag{20}
\]

The denominators are the squared Fisher lengths of \(p\) and \((p,1)\). The row contributes \(4\) in the full metric. The reduced flow in the source uses only the matter metric, so adding a Gibbs-graph pullback contribution would change that specified system.

## Nearby equilibria and their full stability

Full stationarity is equivalent to \(x=X(s)\) and \(\Phi_s(s,T)=0\). Reduced stationarity gives the same configurations with \(s=b^*(x,T)-1/2\). Since \(\Phi\) is analytic and even, write its derivative as

\[
\Phi_s(s,T_c+\delta)=s Q(s^2,\delta),\qquad
Q(z,\delta)=4\delta+cz+O(|\delta||z|+z^2).
\]

The implicit-function theorem at \((z,\delta)=(0,0)\), using \(Q_z(0,0)=c<0\), gives the unique small nonzero branch condition

\[
z(\delta)=-\frac{4\delta}{c}+O(\delta^2).
\tag{21}
\]

Thus, for some sufficiently small \(\varepsilon>0\), every \(0<\delta<\varepsilon\) has exactly two additional sufficiently nearby equilibria,

\[
s_\pm(\delta)=\pm\sqrt{-\frac{4\delta}{c}}+O(\delta^{3/2}),
\qquad x_\pm(\delta)=X(s_\pm(\delta)).
\tag{22}
\]

For \(-\varepsilon<\delta<0\), and also for \(\delta=0\), there are no nonzero equilibria in the corresponding sufficiently small neighborhood. This is a local statement; it does not exclude distant stationary branches.

At either nonzero branch,

\[
\Phi_{ss}(s_\pm,T_c+\delta)=-8\delta+O(\delta^2)<0.
\tag{23}
\]

The matter block \(G_{xx}(X(s),s)\) remains positive definite locally. Its Schur complement in the full Hessian is exactly \(\Phi_{ss}\), so the full Hessian has one negative direction and ten positive directions. Eliminating the positive row block gives one negative and nine positive directions for the Gibbs-reduced Hessian. Positive-definite Fisher mobilities convert these into exactly one unstable eigenvalue and respectively ten or nine stable eigenvalues. The center unstable rate is

\[
\lambda_\pm(\delta)=\frac{8\delta}{d}+O(\delta^2)>0.
\tag{24}
\]

The symmetric branch is locally exponentially stable for \(\delta>0\), with center eigenvalue \(-4\delta/d+O(\delta^2)\), and has exactly one unstable direction for \(\delta<0\). At \(\delta=0\),

\[
\dot s=\frac{-c}{d}s^3+O(s^5),\qquad \frac{-c}{d}>0.
\tag{25}
\]

For sufficiently small nonzero \(s\), the right side has the sign of \(s\) and absolute value at least \((-c)|s|^3/(2d)\). A positive initial center coordinate therefore increases until leaving a fixed sufficiently small positive interval, with the analogous behavior on the negative side. Such initial conditions can be arbitrarily close to the equilibrium, which proves nonlinear instability at \(T_c\). The remaining nine reduced or ten full linear modes are strictly stable. Zero linear growth in the critical direction must not be described as neutral nonlinear stability.

With the conventional destabilizing parameter \(\mu=T_c-T=-\delta\), equations (17)–(18) have positive coefficients of both \(\mu s\) and \(s^3\). This is the subcritical pitchfork: the nearby asymmetric branches lie on the side where the symmetric equilibrium is still stable, and both are saddles. No pair of locally stable asymmetric equilibria is born as temperature decreases through \(T_c\).

## Scope of the comparison and execution record

The critical temperatures, nearby equilibrium configurations, and stability counts agree exactly under the full/Gibbs equilibrium correspondence. Their leading center coefficients differ by the positive factor

\[
\frac{d_{\mathrm{full}}}{d_{\mathrm{red}}}
=\frac{D+4}{D}=\frac{121563380}{19189456}.
\]

Thus full unit row relaxation is slower in the leading critical dynamics. A constant time rescaling matches the displayed leading scalar polynomials. Higher-order terms, stable transients, and complete paths are not identified by this calculation. There is no stochastic replacement or path-law assertion.

This worker used source reads and the displayed analytical coefficient matching, and ran no numerical or symbolic scripts, trajectory experiments, GPU work, or Lean work. The coordinator's exact arithmetic is in `check_critical_mode.py` with output in `critical-mode-checks.json`; the values used here are explicit in (3), (6), and (7). No Research vault or Item 7 file was edited.

The local critical-mode obligation is resolved under the specified Gaussian family and Fisher conventions. Determining where trajectories go after leaving this neighborhood, or whether more distant stable asymmetric equilibria and folds exist, requires continuation or a separate global argument; neither follows from the local pitchfork.
