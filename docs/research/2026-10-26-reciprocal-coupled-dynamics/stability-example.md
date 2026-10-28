# An exact temperature threshold with coupled content and dictionary uncertainty

This construction gives one fully specified finite realization of the tangent reciprocal potential in which a stationary point is locally exponentially stable at high routing temperature and has exactly one unstable direction at low temperature. Every content mean and variance and every dictionary mean and variance is a dynamical coordinate. Freezing the variances changes the stability threshold. The statement is local and concerns the declared Gaussian restricted-family flow; it does not establish formation, a nonlinear bifurcation type, or closure of the unrestricted probability-law dynamics.

## Source and scope

The starting definitions are the live `Research/manuscripts/MAgent_Theory_2.0/chapters/18_link_dynamics.tex`, definition `it2:potential` and equations `it2:message-laws` and `it2:potential-eq`. The Gibbs identity, row dynamics, and full/reduced critical-point relation already appear in `chapters/03_dynamics.tex` as `prop:dyn-gibbs`, `prop:dyn-row-relaxation`, and `prop:dyn-morse`. This note applies that existing mechanism to a completely explicit stationary point and computes its full Hessian. It does not claim the Schur-complement relation as a new general result. The conventions in `AUTHORING.md` and `wiki/concepts/Replicator dynamics.md` were consulted; the former's distinction between the reciprocal potential and either variational free energy is retained.

Use one belief sector, with singleton model spaces. There are three content laws on the real line and two directed edges, $1\to0$ and $2\to0$. The two sources may each have a singleton identity self row, whose cost is identically zero. The only nontrivial source row belongs to receiver $0$. The graph has no directed cycle, so the dictionary prior has no loop term, $\lambda=0$. All codes are the scalar identity, the independent Gaussian message noise has variance one, and the fixed operating content at receiver $0$ is $c_0=1$, hence $\kappa_0\omega=\omega$. The dictionaries are represented by scalar Lie-algebra coordinates; their message action is the tangent action defined in the source chapter, not an assertion of an exact nonlinear group-action formula.

Write

\[
q_i=\mathcal N(\mu_i,v_i),\qquad
r_j=\mathcal N(a_j,V_j),\qquad i=0,1,2,\quad j=1,2.
\]

All five variances are positive. The content reference laws have precisions and means

\[
P_0=1,\quad P_1=P_2=\frac54,\qquad
m_0=0,\quad m_1=-\frac{23}{15},\quad m_2=\frac{23}{15}.
\]

The dictionary prior is the normalized product

\[
\pi_\omega=\mathcal N\left(0,\frac43\right)\otimes
\mathcal N\left(0,\frac43\right),\qquad A=\frac34.
\]

Thus no interacting agent is held fixed: only its reference law and the declared channel data are fixed. In particular, both source laws vary and receive the source-role derivative of the reciprocal objective.

## Exact objective and dynamics

Set $S_i=v_i+1$, $d_j=\mu_0-\mu_j-a_j$, and

\[
C_j(x)=\frac12\left[
\frac{S_0+d_j^2+V_j}{S_j}-1+
\log S_j-\log S_0\right].
\tag{1}
\]

This is exactly the expectation under $r_j$ of the divergence from the receiver's message law $\mathcal N(\mu_0-\omega,S_0)$ to the source message law $\mathcal N(\mu_j,S_j)$. It includes dictionary uncertainty before source selection. Define

\[
F(x)=\frac12\sum_{i=0}^2\left[
P_iv_i+P_i(\mu_i-m_i)^2-1-\log(P_iv_i)\right]
+\frac12\sum_{j=1}^2\left[
AV_j+Aa_j^2-1-\log(AV_j)\right].
\tag{2}
\]

The state order used below is

\[
x=(\mu_0,\mu_1,\mu_2,a_1,a_2,v_0,v_1,v_2,V_1,V_2).
\tag{3}
\]

For $T>0$, use the positive normalized prior row $(1/2,1/2)$. The full objective on the open state space $\mathbb R^5\times(0,\infty)^5\times(0,1)$ is

\[
\mathcal E_T(x,b)=F(x)+bC_1(x)+(1-b)C_2(x)
+T\left[b\log(2b)+(1-b)\log(2(1-b))\right].
\tag{4}
\]

The Gibbs value and reduced objective are

\[
b^*(x)=\frac{e^{-C_1(x)/T}}{e^{-C_1(x)/T}+e^{-C_2(x)/T}},
\qquad
E_T(x)=F(x)-T\log\frac{e^{-C_1(x)/T}+e^{-C_2(x)/T}}2.
\tag{5}
\]

Use the product Fisher metric with unit mobility, whose inverse on the matter and dictionary coordinates is

\[
M(x)=\operatorname{diag}\left(
v_0,v_1,v_2,V_1,V_2,2v_0^2,2v_1^2,2v_2^2,2V_1^2,2V_2^2
\right).
\tag{6}
\]

The Gibbs-reduced dynamics is $\dot x=-M(x)\nabla E_T(x)$. Its row is the instantaneous value (5), not an independently evolved variable. The finite-relaxation dynamics instead is

\[
\dot x=-M(x)\nabla_x\mathcal E_T(x,b),\qquad
\dot b=-b(1-b)\left[C_1-C_2+T\log\frac b{1-b}\right].
\tag{7}
\]

Both vector fields are smooth on their stated open domains. These are continuous ODEs; no discretization is used. Their respective objective derivatives are

\[
\dot E_T=-\nabla E_T^{\mathsf T}M\nabla E_T,
\qquad
\dot{\mathcal E}_T=-\nabla_x\mathcal E_T^{\mathsf T}M\nabla_x\mathcal E_T
-b(1-b)(\partial_b\mathcal E_T)^2.
\tag{8}
\]

In particular, routing changes the local Hessian without creating a non-gradient flow or complex linear eigenvalues at a stationary point.

## A stationary point independent of temperature

For every $T>0$, the point

\[
x_*=\left(0,-\frac43,\frac43,\frac13,-\frac13,1,1,1,1,1\right),
\qquad b_* =\frac12,
\tag{9}
\]

is stationary for both flows. At this point $S_i=2$, $d_1=1$, $d_2=-1$, and $C_1=C_2=1/2$, which proves the row statement. The content-reference mean derivatives at the two sources are $(1/4,-1/4)$; the weighted source-role derivatives are $(-1/4,1/4)$. The dictionary-reference mean derivatives are also $(1/4,-1/4)$, canceled by the dictionary edge derivatives. At the receiver the two mean derivatives cancel each other.

For the variance derivatives, the receiver self term and both receiver-role terms vanish separately. At either source, the self derivative is $1/8$ and the weighted edge derivative is $-1/8$. At either dictionary, the self derivative is $-1/8$ and the weighted edge derivative is $1/8$. These cancellations verify all ten matter/dictionary stationary equations. They are independent of $T$, because the Gibbs row remains $(1/2,1/2)$ on this symmetric stationary branch.

## The full ten-dimensional Hessian

Let

\[
H=\nabla^2\left(F+\frac12C_1+\frac12C_2\right)(x_*),
\qquad g=\nabla(C_1-C_2)(x_*).
\]

In order (3),

\[
g=\left(1,-\frac12,-\frac12,-\frac12,-\frac12,0,
-\frac14,\frac14,\frac14,-\frac14\right)^{\mathsf T}.
\tag{10}
\]

For an arbitrary perturbation

\[
\xi=(x_0,x_1,x_2,b_1,b_2,h_0,h_1,h_2,k_1,k_2),
\quad z_j=x_0-x_j-b_j,\quad s_1=1,\quad s_2=-1,
\]

direct differentiation of (1)–(2) gives the exact sum of squares

\[
\begin{aligned}
\xi^{\mathsf T}H\xi={}&x_0^2+\frac54(x_1^2+x_2^2)
+\frac34(b_1^2+b_2^2)+\frac12h_0^2\\
&+\sum_{j=1}^2\left[
\frac14\left(z_j-\frac{s_jh_j}{2}\right)^2
+\frac1{16}(h_0-h_j)^2
+\frac{71}{128}h_j^2
+\frac12\left(k_j-\frac{h_j}{8}\right)^2\right].
\end{aligned}
\tag{11}
\]

This is strictly positive for every nonzero perturbation. For example, vanishing forces all five mean perturbations to zero through their displayed positive terms, then $(h_0,h_1,h_2)$ to zero, then $(k_1,k_2)$ to zero. Therefore $H\succ0$ on the complete ten-dimensional tangent space, with no symmetry restriction or frozen covariance.

For checking the coefficients, the contribution of one weighted edge to the Hessian quadratic form, before completing squares, is

\[
\frac14z_j^2-\frac{s_j}{4}z_jh_j
+\frac1{16}h_0^2-\frac18h_0h_j
+\frac3{16}h_j^2-\frac18h_jk_j.
\tag{12}
\]

Adding the five self-variance terms $\frac12(h_0^2+h_1^2+h_2^2+k_1^2+k_2^2)$ gives (11). The cross terms in (12) are why the full stability calculation cannot be replaced by the fixed-covariance mean theorem.

The inverse action needed below is the explicit vector

\[
H^{-1}g=\frac1{5059}
(2400,-960,-960,-1600,-1600,0,-716,716,2440,-2440)^{\mathsf T},
\tag{13}
\]

so

\[
g^{\mathsf T}H^{-1}g=\frac{6538}{5059}.
\tag{14}
\]

One short derivation of (13) uses the reflection/exchange symmetry of the fixed-row objective. The vector $H^{-1}g$ lies in the subspace $x_1=x_2=y$, $b_1=b_2=z$, $h_0=0$, $h_1=-h_2=u$, $k_1=-k_2=w$. Write $t=x_0-y-z$. Minimizing $x_0^2+(5/2)y^2+(3/2)z^2$ at fixed $t$ gives $(x_0,y,z)=(15,-6,-10)t/31$ and value $15t^2/31$. The quadratic form and linear functional reduce to

\[
Q(t,u,w)=\frac{61}{62}t^2-\frac12tu+\frac{11}{8}u^2-\frac14uw+w^2,
\qquad g^{\mathsf T}\xi=t-\frac u2+\frac w2.
\tag{15}
\]

The associated three-by-three linear system is

\[
\begin{pmatrix}
61/62&-1/4&0\\
-1/4&11/8&-1/8\\
0&-1/8&1
\end{pmatrix}
\begin{pmatrix}t\\u\\w\end{pmatrix}
=\begin{pmatrix}1\\-1/2\\1/2\end{pmatrix},
\]

whose exact solution is $(t,u,w)=(4960,-716,2440)/5059$. Substitution gives (13). The symmetry is used only to evaluate this inverse action; the positivity certificate (11) already covers the complete space.

## Exact stability threshold

Twice differentiating the finite Gibbs sum at its equal-cost point gives

\[
H_T=\nabla^2E_T(x_*)=H-\frac1{4T}gg^{\mathsf T}.
\tag{16}
\]

This is the two-source instance of the negative cost-gradient covariance term in the Hessian of the soft minimum. Congruence by $H^{-1/2}$ transforms (16) into

\[
I-\frac1{4T}(H^{-1/2}g)(H^{-1/2}g)^{\mathsf T}.
\]

It has nine eigenvalues equal to one and one equal to $1-g^{\mathsf T}H^{-1}g/(4T)$. Therefore, with

\[
T_c=\frac14g^{\mathsf T}H^{-1}g=\frac{3269}{10118},
\tag{17}
\]

the Hessian $H_T$ is positive definite for $T>T_c$, has exactly one negative eigenvalue and nine positive eigenvalues for $0<T<T_c$, and has a one-dimensional kernel at $T=T_c$.

At a stationary point the derivative of the mobility contributes nothing to the linearization, so the Gibbs-reduced Jacobian is $-M(x_*)H_T$. It is similar to the symmetric matrix $-M(x_*)^{1/2}H_TM(x_*)^{1/2}$. Consequently $x_*$ is locally exponentially stable for $T>T_c$. For $0<T<T_c$ it is a hyperbolic saddle with exactly one unstable direction. These are statements about all ten coordinates. At the threshold the linearization has a zero mode, and the nonlinear classification is not established here.

For the finite-relaxation row flow, the full eleven-dimensional Hessian at $(x_*,1/2)$ is

\[
\mathbb H_T=\begin{pmatrix}H&g\\g^{\mathsf T}&4T\end{pmatrix}.
\tag{18}
\]

The exact completion of squares

\[
\begin{pmatrix}\xi\\\delta b\end{pmatrix}^{\mathsf T}
\mathbb H_T
\begin{pmatrix}\xi\\\delta b\end{pmatrix}
=\xi^{\mathsf T}H_T\xi
+4T\left(\delta b+\frac{g^{\mathsf T}\xi}{4T}\right)^2
\tag{19}
\]

proves the same stability threshold, with one additional positive Hessian direction. Its Fisher inverse metric is $\operatorname{diag}(M(x_*),1/4)$, so the same similarity argument establishes local exponential stability for $T>T_c$ and exactly one unstable direction for $0<T<T_c$. This explicitly instantiates the existing full/reduced inertia proposition. It does not identify the two transient flows or their relaxation rates.

## What freezing uncertainty misses

If all five variances are instead fixed at one, the remaining five-coordinate mean Hessian has the same negative Gibbs correction but lacks covariance response. Its inverse quadratic form is

\[
g_{\mathrm{mean}}^{\mathsf T}H_{\mathrm{mean}}^{-1}g_{\mathrm{mean}}=\frac{62}{61},
\qquad T_{\mathrm{mean}}=\frac{31}{122}.
\tag{20}
\]

This follows from (15) with $u=w=0$. The exact inequalities

\[
\frac{31}{122}<\frac3{10}<\frac{3269}{10118}
\tag{21}
\]

show that at $T=3/10$ the frozen-covariance mean system is locally stable, whereas the fully coupled system has one unstable direction. The unstable direction is not purely a row or a content-mean direction: the null direction at the threshold is proportional to (13) and includes source variances and both dictionary variances.

This establishes a specific failure of transferring fixed-covariance stability to the complete restricted-family dynamics. It does not establish that uncertainty increases monotonically, that trajectories converge to newly selected groups, or that a pair of stable branches emerges at the threshold. Those require further analysis.

## Verification and remaining obligation

This worker performed the displayed exact algebra without running scripts or numerical experiments. The coordinating agent independently checked the stationary gradient, the Hessian sum-of-squares identity, the inverse action, the threshold, the direct Gibbs Hessian, and the full row Hessian with exact symbolic arithmetic; the executable checks and their output are owned by that agent and should be read with the final investigation record. No Research vault file or Item 7 source was edited.

The load-bearing local-stability result is complete for this finite Gaussian tangent realization away from $T_c$. Its minimal next local obligation is the nonlinear center-direction analysis at $T_c$, if a bifurcation classification is desired. Long-time destinations below the threshold, connection or loop effects on a graph with cycles, non-Gaussian full-law dynamics, and any formation interpretation remain separate problems.
