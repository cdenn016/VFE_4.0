# Stationary reduction of the three-agent critical mode

The exact local stationary bifurcation is subcritical. With all ten matter coordinates free, two nonsymmetric stationary branches approach the symmetric state from temperatures above the critical temperature. Both branches have one unstable direction for either declared Fisher flow. At the critical temperature the symmetric state is Lyapunov unstable, despite its nonnegative Hessian. The stationary calculation below does not identify its implicit-function curve with an invariant manifold of either vector field.

## Target, sources, and scope

The target is the complete set of stationary points in a neighborhood of the specified symmetric state, for every temperature in a sufficiently small two-sided neighborhood of $T_c$, together with their Hessian inertia and local stability for the full and Gibbs-reduced Fisher flows. Equivalence means an exact bijection of stationary points under elimination of the interior row variable, with equal negative and zero Hessian indices. It does not mean equality of trajectories or relaxation rates.

The supplied model is the Gaussian tangent example in `Research/manuscripts/MAgent_Theory_2.0/chapters/18_link_dynamics.tex`, the proposition and theorem labeled `it2:coupled-critical-structure` and `it2:coupled-threshold`, including its exact two-peer candidate row. The complete prior derivation is `docs/research/2026-10-26-reciprocal-coupled-dynamics/stability-example.md`. Both sources and the live `AUTHORING.md` were read. All means and variances remain variables. The reference laws, positive message noise, fixed operating map, product-family Fisher metric, and unit row mobility are supplied constitutive data. Analyticity on the positive-variance interior follows from the displayed formulas; it is not a new physical assumption. This result concerns the finite Gaussian tangent model, without a claim of Gaussian invariance of unrestricted law dynamics, global destinations, or formation.

There are two relevant reduction mechanisms. Stationary elimination first requires an invertible matter Hessian and produces an exact scalar stationary equation. An invariant center-manifold calculation first requires solving the stable invariance equation of the actual Fisher vector field and also determines kinetic coefficients. The first mechanism closes the stationary and inertia obligations here; the dynamical normal-form calculation is separate. A restriction to the straight null line omits the stable-coordinate correction computed below and does not solve the stationary problem.

## Exact model and symmetry

The matter domain is $\mathcal D=\mathbb R^5\times(0,\infty)^5$, with coordinates

\[
x=(\mu_0,\mu_1,\mu_2,a_1,a_2,v_0,v_1,v_2,V_1,V_2).
\]

Put $S_i=v_i+1$, $d_j=\mu_0-\mu_j-a_j$, and

\[
C_j(x)=\frac12\left[\frac{S_0+d_j^2+V_j}{S_j}-1+
\log S_j-\log S_0\right],\qquad j=1,2.
\]

The proper reference-law contribution is

\[
\begin{aligned}
F(x)={}&\frac12\sum_{i=0}^2
\left[P_i v_i+P_i(\mu_i-m_i)^2-1-\log(P_i v_i)\right]\\
&+\frac12\sum_{j=1}^2
\left[\frac34 V_j+\frac34 a_j^2-1-\log\left(\frac34V_j\right)\right],
\end{aligned}
\]

where $(P_0,P_1,P_2)=(1,5/4,5/4)$ and $(m_0,m_1,m_2)=(0,-23/15,23/15)$. Define

\[
A=F+\frac{C_1+C_2}{2},\qquad \Delta=C_1-C_2,
\qquad G(x,s)=A(x)+s\Delta(x),\qquad s=b-\frac12.
\]

For $T>0$ and $|s|<1/2$, the full potential is

\[
\mathcal E_T(x,s)=G(x,s)+T h(s),\qquad
h(s)=\left(\frac12+s\right)\log(1+2s)
 +\left(\frac12-s\right)\log(1-2s).
\tag{1}
\]

Every denominator and logarithm is well-defined on the stated open domain. In particular,

\[
h'(s)=\log\frac{1+2s}{1-2s},\qquad
h''(s)=\frac4{1-4s^2},\qquad h''(0)=4,\qquad h''''(0)=32.
\tag{2}
\]

Consider the linear involution

\[
Jx=(-\mu_0,-\mu_2,-\mu_1,-a_2,-a_1,
v_0,v_2,v_1,V_2,V_1).
\tag{3}
\]

It preserves $\mathcal D$, is orthogonal in these coordinates, and fixes

\[
x_*=(0,-4/3,4/3,1/3,-1/3,1,1,1,1,1).
\]

Indeed, the source reference means are exchanged and negated, their precisions agree, and the dictionary references are centered with equal precision, so $F(Jx)=F(x)$. Also $d_1(Jx)=-d_2(x)$, $S_1(Jx)=S_2(x)$, and $V_1(Jx)=V_2(x)$, whence $C_1(Jx)=C_2(x)$ and conversely. Therefore

\[
A(Jx)=A(x),\qquad \Delta(Jx)=-\Delta(x),\qquad
\mathcal E_T(Jx,-s)=\mathcal E_T(x,s).
\tag{4}
\]

This is a symmetry of this fixed-data example. No broader frame-invariance assertion is needed.

## Elimination of every stable matter coordinate

At $(x_*,0)$, $A_x=0$, $\Delta=0$, and

\[
g=\Delta_x(x_*)=
\left(1,-\frac12,-\frac12,-\frac12,-\frac12,
0,-\frac14,\frac14,\frac14,-\frac14\right)^T.
\]

Let $H=A_{xx}(x_*)$. Its positivity is available from the prior derivation, and can be checked directly as follows. For
$\xi=(u_0,u_1,u_2,w_1,w_2,z_0,z_1,z_2,k_1,k_2)$, set
$r_j=u_0-u_j-w_j$ and $\epsilon_1=1$, $\epsilon_2=-1$. Direct differentiation gives

\[
\begin{aligned}
\xi^T H\xi={}&u_0^2+\frac54(u_1^2+u_2^2)
 +\frac34(w_1^2+w_2^2)+\frac12z_0^2\\
&+\sum_{j=1}^2\left[
\frac14\left(r_j-\frac{\epsilon_jz_j}{2}\right)^2
 +\frac1{16}(z_0-z_j)^2+\frac{71}{128}z_j^2
 +\frac12\left(k_j-\frac{z_j}{8}\right)^2\right].
\end{aligned}
\tag{5}
\]

Vanishing forces all ten components to vanish, so $H$ is positive definite.

The finite-dimensional analytic implicit-function theorem used here states that an analytic map $f(x,s)$ with $f(x_*,0)=0$ and invertible $D_xf(x_*,0)$ admits a unique local analytic solution $x=X(s)$ of $f(x,s)=0$, and every solution in a sufficiently small product neighborhood lies on that graph. Apply it to $f=G_x$; equation (5) verifies the invertibility hypothesis. Thus, after shrinking neighborhoods, there is a unique analytic map

\[
X:(-\varepsilon,\varepsilon)\longrightarrow\mathcal D,
\qquad X(0)=x_*,\qquad G_x(X(s),s)=0.
\tag{6}
\]

It is independent of temperature. Continuity allows the neighborhoods to be chosen so that

\[
K(s)=G_{xx}(X(s),s)\succ0
\tag{7}
\]

throughout. The same positivity can be imposed on a convex matter neighborhood, if a strict local minimum at fixed $s$ is desired. Uniqueness, (4), and a $J$-invariant choice of neighborhood imply

\[
X(-s)=JX(s).
\tag{8}
\]

The scalar stationary potential

\[
\Phi_T(s)=G(X(s),s)+T h(s)
\tag{9}
\]

is analytic and even. By the ordinary chain rule and (6),

\[
\Phi_T'(s)=\Delta(X(s))+T h'(s).
\tag{10}
\]

Consequently $(x,s)$ is a full stationary point in the chosen neighborhood if and only if $x=X(s)$ and $\Phi_T'(s)=0$. This is an exact equivalence, not a truncation.

For the row-reduced problem, strict convexity of (1) in $s$ gives the unique interior optimizer

\[
b_T^*(x)=\frac1{1+e^{\Delta(x)/T}},\qquad
s_T^*(x)=b_T^*(x)-\frac12,
\]

and

\[
E_T(x)=\mathcal E_T(x,s_T^*(x))
=F(x)-T\log\frac{e^{-C_1(x)/T}+e^{-C_2(x)/T}}2.
\tag{11}
\]

The derivative of the optimizing row multiplies $\partial_s\mathcal E_T=0$, so $\nabla E_T(x)=\nabla_x\mathcal E_T(x,s_T^*(x))$. This proves the stationary-point bijection with the reduced flow. Continuity of $s_T^*$ supplies a common local neighborhood for this correspondence.

## The fourth-order coefficient including matter response

All tensors in this section are evaluated at $x_*$. Differentiating (6) once and twice gives

\[
p:=X'(0)=-H^{-1}g,
\qquad
k:=A_{xxx}[p,p]+2\Delta_{xx}p,
\qquad r:=X''(0)=-H^{-1}k.
\tag{12}
\]

The tensor $A_{xxx}[p,p]$ here is a covector, with components $\sum_{j,\ell}(\partial_i\partial_j\partial_\ell A)p_jp_\ell$, identified with its coordinate column. In this formula the product is an ordinary tensor contraction; no Fisher inverse is inserted into the stationary equations.

Put $N=5059$. Exact solution of the first system gives

\[
p=\frac1N(-2400,960,960,1600,1600,0,716,-716,-2440,2440)^T.
\tag{13}
\]

In particular $p\ne0$, and

\[
g^T H^{-1}g=\frac{6538}{5059}=4T_c,
\qquad T_c=\frac{3269}{10118}.
\tag{14}
\]

Equation (10) gives $\Phi_T''(0)=-g^TH^{-1}g+4T=4(T-T_c)$. Odd derivatives vanish by evenness. To obtain the next coefficient, write
$X(s)=x_*+ps+rs^2/2+ts^3/6+O(s^4)$.
The fourth-order coefficient of $G(X(s),s)$ is

\[
\frac1{24}A_{xxxx}[p,p,p,p]
 +\frac16\Delta_{xxx}[p,p,p]
 +\frac18r^THr+\frac14r^Tk
 +\frac16(Hp+g)^Tt.
\]

The last term vanishes by the first equation in (12), while the preceding two terms combine to $-k^TH^{-1}k/8$. Therefore the exact identity is

\[
\boxed{\displaystyle
\Phi_T''''(0)=A_{xxxx}[p,p,p,p]+4\Delta_{xxx}[p,p,p]
 -3k^TH^{-1}k+32T.}
\tag{15}
\]

The term $-3k^TH^{-1}k$ is the response of the eliminated matter coordinates. A straight-line evaluation at $x_*+ps$ would omit it.

For explicit checking, the remaining derivative vectors are

\[
k=\frac1{N^2}
(0,25951840,-25951840,25951840,-25951840,
-1747040,13494712,13494712,-6827120,-6827120)^T,
\tag{16}
\]

and

\[
r=\frac1{9501N^2}
(0,-121009274880,121009274880,-201682124800,201682124800,
3143421952,-117071906560,-117071906560,115094945920,115094945920)^T.
\tag{17}
\]

Equations (5), (16), and (17) permit direct verification that $Hr=-k$. The other contractions in (15) can be reduced to a single rational function. Along $x_*+pt$, write $u=t/N$; then

\[
C_1(x_*+pt)=\frac12\left[
\frac{2-6180u+12300800u^2}{1+358u}-1+\log(1+358u)\right],
\qquad C_2(x_*+pt)=C_1(x_*-pt).
\tag{18}
\]

The coefficient of $u^2$ in the rational term of (18) is $14769568$. Its coefficients of $u^3$ and $u^4$ are respectively $-358\cdot14769568$ and $358^2\cdot14769568$. The fourth derivative of the self-law contribution along this line is $6(716^4+2440^4)/N^4$, since the mean terms are at most quadratic and only the logarithms of the four varying variances contribute at order four. It follows that

\[
\begin{aligned}
A_{xxxx}[p,p,p,p]
&=\frac{6(716^4+2440^4)+12\cdot358^2\cdot14769568-3\cdot358^4}{N^4}
=\frac{236914859731152}{655026269697361},\\
\Delta_{xxx}[p,p,p]
&=\frac{-6\cdot358\cdot14769568+2\cdot358^3}{N^3}
=-\frac{31633266640}{129477420379},\\
3k^TH^{-1}k&=-3k^Tr
=\frac{21485600170644766720}{2074468196131542287}.
\end{aligned}
\tag{19}
\]

Thus the fourth derivative of the matter envelope is

\[
\left.\frac{d^4}{ds^4}G(X(s),s)\right|_{s=0}
=-\frac{22762585001939744016}{2074468196131542287},
\]

and addition of the row entropy gives

\[
Q:=\Phi_{T_c}''''(0)
=-\frac{1315068787181058944}{2074468196131542287}<0,
\qquad
c:=\frac Q6
=-\frac{657534393590529472}{6223404588394626861}<0.
\tag{20}
\]

These are exact rational identities. The decimal $c\approx-0.1056550934864$ is only a convenient evaluation of the exact value.

## Local branch theorem

There exist neighborhoods of $x_*$ and $s=0$, and a number $\delta>0$ with $\delta<T_c$, such that for every $\tau=T-T_c$ with $|\tau|<\delta$, all stationary points in these neighborhoods have the following form. The symmetric point $(x_*,0)$ always exists. If $\tau\leq0$, it is the only stationary point there. If $0<\tau<\delta$, there are exactly two further points

\[
(X(s_+(\tau)),s_+(\tau)),\qquad (X(s_-(\tau)),s_-(\tau)),
\]

where $s_-(\tau)=-s_+(\tau)$ and

\[
s_\pm(\tau)=\pm\sqrt{-\frac4c}\sqrt{\tau}+O(\tau^{3/2}),
\qquad
X(s_\pm(\tau))=x_*+p s_\pm(\tau)+\frac12r s_\pm(\tau)^2
 +O(\tau^{3/2}).
\tag{21}
\]

The neighborhoods may be chosen so that the two points are distinct, since $p\ne0$ makes a suitable linear coordinate of $X(s)$ strictly monotone near zero.

To prove completeness and count the branches, analyticity and evenness give the exact factorization

\[
\Phi_T'(s)=s q(s^2,\tau),
\quad q(z,\tau)=4\tau+cz+\frac{16}{3}\tau z+z^2R(z,\tau),
\tag{22}
\]

where $R$ is analytic near $(0,0)$. This follows either by the convergent even power series or directly from (2), (15), and (20). In particular $q_z(0,0)=c<0$. The analytic implicit-function theorem therefore gives a unique analytic $z=\chi(\tau)$ with $q(\chi(\tau),\tau)=0$ near zero, satisfying

\[
\chi(0)=0,\qquad \chi'(0)=-\frac4c>0.
\tag{23}
\]

Shrink the neighborhoods so that $q_z<0$, all these roots remain inside the selected $z$ interval, and $\chi(\tau)$ has the sign of $\tau$. Monotonicity in $z$ excludes additional roots in that interval. Real $s$ has $z=s^2\geq0$, so the only nonzero roots occur for $\tau>0$ and are precisely $s=\pm\sqrt{\chi(\tau)}$. This proves the stated count and expansions. No numerical continuation or inference from a finite temperature scan is used.

## Full and reduced Hessian inertia and stability

At a point on (6), set $\ell(s)=\Delta_x(X(s))$ and $d_T(s)=T h''(s)>0$. Differentiating (6) gives $X'(s)=-K(s)^{-1}\ell(s)$, hence

\[
\Phi_T''(s)=d_T(s)-\ell(s)^TK(s)^{-1}\ell(s).
\tag{24}
\]

The full Hessian is

\[
\mathbb H_T=
\begin{pmatrix}K&\ell\\\ell^T&d_T\end{pmatrix}.
\]

For arbitrary perturbations $(\xi,\eta)$, the exact square completion is

\[
(\xi,\eta)^T\mathbb H_T(\xi,\eta)
=(\xi+K^{-1}\ell\eta)^TK(\xi+K^{-1}\ell\eta)
 +\Phi_T''(s)\eta^2.
\tag{25}
\]

At any stationary point this proves that the ten positive matter directions are supplemented by the sign of $\Phi_T''(s)$. Alternatively, eliminating the positive row block gives

\[
H_{\mathrm{red}}=K-\frac{\ell\ell^T}{d_T},\qquad
(\xi,\eta)^T\mathbb H_T(\xi,\eta)
=\xi^TH_{\mathrm{red}}\xi+d_T\left(\eta+\frac{\ell^T\xi}{d_T}\right)^2.
\tag{26}
\]

Both changes of variables are invertible, so the full and reduced Hessians have identical negative and zero indices and differ by exactly one positive direction.

At the symmetric point, $\Phi_T''(0)=4\tau$. At either nonsymmetric branch, (22) gives

\[
\Phi_T''(s_\pm)=2\chi(\tau)q_z(\chi(\tau),\tau)
=-8\tau+O(\tau^2)<0.
\tag{27}
\]

The positive, negative, and zero inertia triples are therefore

| Stationary point and temperature | Full Hessian | Reduced Hessian |
| --- | --- | --- |
| Symmetric, $T>T_c$ | $(11,0,0)$ | $(10,0,0)$ |
| Symmetric, $T<T_c$ | $(10,1,0)$ | $(9,1,0)$ |
| Symmetric, $T=T_c$ | $(10,0,1)$ | $(9,0,1)$ |
| Either nearby nonsymmetric branch, $T>T_c$ | $(10,1,0)$ | $(9,1,0)$ |

The actual positive inverse Fisher metrics are

\[
M(x)=\operatorname{diag}(v_0,v_1,v_2,V_1,V_2,
2v_0^2,2v_1^2,2v_2^2,2V_1^2,2V_2^2)
\]

for the reduced flow and $\operatorname{diag}(M(x),1/4-s^2)$ for the full flow. At a critical point of a smooth potential $V$, a gradient vector field $-\mathcal M\nabla V$ has Jacobian $-\mathcal M D^2V$, since the derivative of $\mathcal M$ multiplies the vanished gradient. This matrix is similar to the symmetric matrix $-\mathcal M^{1/2}D^2V\mathcal M^{1/2}$. Thus the sign counts above give the unstable dimensions. The finite-dimensional linear stability theorem used here states that a $C^1$ vector field whose equilibrium Jacobian has spectrum strictly in the left half-plane has a locally exponentially stable equilibrium, while an equilibrium with a Jacobian eigenvalue of positive real part is unstable. Its hypotheses apply because these vector fields are analytic on their open domains and both metrics are positive definite there. Consequently the symmetric branch is exponentially stable above $T_c$, the symmetric branch below $T_c$ is a hyperbolic saddle with one unstable direction, and each nearby nonsymmetric branch is also a hyperbolic saddle with one unstable direction. With instability parameter $T_c-T$, this is the subcritical pitchfork convention.

At $T=T_c$, the stationary reduction also establishes Lyapunov instability without identifying an invariant center manifold. By (20),

\[
\Phi_{T_c}(s)-\Phi_{T_c}(0)=\frac Q{24}s^4+O(s^6)<0
\quad\text{for sufficiently small }s\ne0.
\tag{28}
\]

For the full potential, $(X(s),s)$ are therefore points arbitrarily close to the equilibrium with lower energy. For the reduced potential,
$E_{T_c}(X(s))\leq\mathcal E_{T_c}(X(s),s)=\Phi_{T_c}(s)$, while $E_{T_c}(x_*)=\Phi_{T_c}(0)$; it too has arbitrarily close lower-energy points. The branch theorem makes the equilibrium isolated in both cases.

For completeness, an isolated equilibrium of a smooth positive-metric gradient flow with arbitrarily close lower-energy points cannot be Lyapunov stable. If it were stable, choose a closed ball inside the state domain containing no other critical point, and a smaller initial neighborhood whose trajectories remain in the interior of that ball. Choose a lower-energy point in this initial neighborhood. Its trajectory is trapped in a compact set, so the smooth local flow extends for all positive time there, and its energy decreases to a finite limit. Every omega-limit point must be critical: otherwise a neighborhood of an omega-limit point has the dissipation $\nabla V^T\mathcal M\nabla V$ bounded below by a positive number, and bounded speed on the compact set supplies a fixed residence time on each return, contradicting convergence of the energy. The only possible omega-limit point is the isolated equilibrium. Continuity then forces the limiting energy to equal the equilibrium energy, contradicting the strictly smaller initial energy and monotonicity. This proves the claimed critical-temperature instability for both flows.

## Evidence and remaining boundary

This worker performed analytical derivation and read-only source inspection; no script, trajectory computation, GPU task, or Lean process was run by this worker. The coordinator's exact rational calculation is recorded separately in `check_critical_mode.py` and `critical-mode-checks.json`. Equations (5), (12), and (16)–(19) expose the arithmetic needed to check the fourth-order coefficient without relying on a reported decimal or on agreement between agents.

The stationary branch count, Hessian inertia, and local stability classifications are established for this fixed finite Gaussian tangent example near $(x_*,1/2,T_c)$. The companion artifact `dynamics-reduction.md` supplies the separate reduction of the actual full and reduced center dynamics and their kinetic coefficients; the implicit-function graph (6) alone does not supply that reduction. Global continuation of the two saddle branches, possible distant stable states, trajectories after they leave this neighborhood, and any formation interpretation remain outside this result.
