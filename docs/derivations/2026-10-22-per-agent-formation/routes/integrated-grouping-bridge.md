<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca","schema_version":"rigorous-theory-search/v1","target_digest":"0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca"} -->
# Persistent grouping for the fixed-evaluator angular family

Date: 2026-10-22. This bridge proves the grouping and null-comparison obligation for the normalized angular family used in the integrated construction. For every sufficiently small fixed positive record strength, a nonempty relatively open set in an exactly invariant finite-dimensional family starts with singleton blocks, reaches a persistent three-agent block, and never reaches that block under the prescribed null. The proof allows initial angular amplitudes in a fixed small compact patch independent of the record strength. Persistence is conditional on the one retained initial record episode, with no new record assimilated during the claimed interval. It does not prove compression, equality of inherited histories, or collective curvature.

The source-row convention, persistent-record interpretation, and prescribed null come from the current Chapter 3 and Chapter 5 construction, as recorded in [per-agent-transfer.md](per-agent-transfer.md). The affine three-agent grouping mechanism is the one derived in [persistent-record-grouping.md](persistent-record-grouping.md). The new obligation here is to show that normalized angular variables can enter the actual public forces and rows without destroying that mechanism, and to bound the null for all time rather than invoking finite-time perturbation theory indefinitely.

## 1. Precisely stated result and model

The admitted belief space is \(\mathcal K=\{-1,1\}\times\mathbb T\times\mathbb T\), with reference measure \(\mu_0=(\tfrac12\delta_{-1}+\tfrac12\delta_1)\otimes H\otimes H\), where \(H(d\varphi)=d\varphi/(2\pi)\). The model space is \(\{-1,1\}\) with counting measure. Write \(e_\varphi=(\cos\varphi,\sin\varphi)\), and

\[
I_0(r)=\int_{\mathbb T}e^{r\cos\varphi}H(d\varphi),\qquad
h(V)=\log\int_{\mathbb T}e^{V\cdot e_\varphi}H(d\varphi)=\log I_0(|V|).
\]

For three agents, take the fixed source priors \(\pi_{ii}=a=1/5\), \(\pi_{ij}=p=2/5\) for \(j\ne i\), in both sectors. Use exact Gibbs reduction at unit label temperature and own-law Fisher mobility one. For each fixed \(\epsilon>0\), set \(J=\epsilon^3\), \(\lambda=\epsilon\). These are fixed constitutive parameters throughout either full or null evolution.

The belief family consists of all finite natural parameters

\[
q_i(z,\varphi,\psi)=\frac1{Z_i}
\exp\left[\frac{l_i z}{2}+A_i\cdot e_\varphi+zC_i\cdot e_\varphi+B_i\cdot e_\psi\right],
\tag{B1}
\]

where this is a density relative to \(\mu_0\), \(A_i,C_i,B_i\in\mathbb R^2\), and

\[
Z_i=I_0(|B_i|)\frac{e^{l_i/2}I_0(|A_i+C_i|)+e^{-l_i/2}I_0(|A_i-C_i|)}2.
\tag{B2}
\]

On \(B_i\ne0\), use \(B_i=b_i e_{\theta_i}\), with \(b_i>0\). The ambient Cartesian coordinates \(B_i\) remain regular at zero. The model recognition law is \(s_i(w)=e^{n_iw/2}/[2\cosh(n_i/2)]\).

The self reference is uniform in \(w\). The self evaluator, a density relative to \(\mu_0\), is the fixed normalized law

\[
G(w;z,\varphi,\psi)=\frac{e^{Jz\cos\varphi}}{I_0(J)}.
\tag{B3}
\]

It is independent of \(w\) and of every recognition parameter. Integrating first over \(\varphi\), and then over the uniform sign and \(\psi\), proves normalization. No parameter-dependent evaluator is held artificially fixed during an own update.

Each peer channel copies \(z\) and resamples \(\varphi,\psi\) independently from Haar measure. Thus the communicated peer belief is \(\nu_j=\mu_j\otimes H\otimes H\), where \(\mu_j\) is the sign marginal of \(q_j\). This is a fixed normalized channel, independent of all laws. The communicated model law is \(s_j\), using the identity channel. The public sign log odds are exactly

\[
L_i=l_i+h(A_i+C_i)-h(A_i-C_i),\qquad
\mu_i(z)=\frac{e^{L_i z/2}}{2\cosh(L_i/2)}.
\tag{B4}
\]

The record space is \(\{-1,1\}\times\mathbb T\), with counting measure for \(o\) and Haar measure for \(\omega\). The common strictly positive normalized likelihood is

\[
\ell_{\epsilon,\lambda}(o,\omega\mid z,\varphi,\psi,w)
=\frac{e^{\epsilon oz/2}}{2\cosh(\epsilon/2)}
\frac{e^{\lambda\cos(\omega-\varphi)}}{I_0(\lambda)}.
\tag{B5}
\]

Its sum-integral over \((o,\omega)\) is one for every latent state. Condition on the realized records \((o_1,o_2,o_3)=(1,1,-1)\) and \(\omega_i=0\), and retain them in every later objective. Continuous record values are conditioned through the strictly positive observation density. The species, priors, likelihood, evaluator, and peer channels are identical across agents; the differing binary records are observed data.

For clarity, the local joint law conditional on the current peer laws is

\[
P_i(w,k,j,r,o,\omega)=
\pi_{ir}R_{ir}(w)\pi_{ij}T_{ij}(k)\ell_{\epsilon,\lambda}(o,\omega\mid k,w),
\tag{B6}
\]

where \(R_{ii}=1/2\), \(R_{ir}=s_r\) for \(r\ne i\), \(T_{ii}=G\), and \(T_{ij}=\nu_j\) for \(j\ne i\). The references in (B6) are the declared counting/Haar/product measures. Summing or integrating over the record, content state, its source label, model state, and model source label, in that order, gives one.

The theorem below concerns a relative-open subset of the exactly invariant family (B1), not an open subset of all smooth densities. The extra smallness and compact-patch assumptions are construction hypotheses introduced here. No law over initial configurations or full stochastic record protocol is presumed.

## 2. Exact variational field and the angular dependence of the rows

Use recognition \(Q_i=s_iq_i\beta_i\gamma_i\), with normalized nonnegative rows. Define the negative ELBO by

\[
F_i=\mathbb E_{Q_i}\left[
\log\frac{s_i(W)q_i(K)\beta_{iJ}\gamma_{iR}}
{\pi_{iR}R_{iR}(W)\pi_{iJ}T_{iJ}(K)
\ell_{\epsilon,\lambda}(o_i,0\mid K,W)}\right].
\tag{B7}
\]

All laws have full support. Compact angle spaces and finite parameters make the likelihood logs, density logs, and all divergences finite; they are analytic on finite parameter charts. The product recognition restriction is part of the model being solved.

Let

\[
D_{ii}=\mathbb E_{q_i}[\log q_i-\log G],\qquad
D_{ij}=\mathbb E_{q_i}[\log q_i-\log\nu_j]\quad(j\ne i),
\]

and let \(E_{ii}=D(s_i\Vert\operatorname{Ber}(1/2))\), \(E_{ij}=D(s_i\Vert s_j)\). The record term is

\[
\mathbb E_{q_i}[-\log\ell(o_i,0\mid K)]
=\log(2\cosh(\epsilon/2))+\log I_0(\lambda)
-\frac{\epsilon o_i}2\mathbb E_{q_i}Z-\lambda\mathbb E_{q_i}\cos\varphi.
\]

Consequently (B7) equals the two source-weighted divergence sums, their row KL divergences, and this record term. Normalized row minimization gives

\[
\beta_{ij}=\frac{\pi_{ij}e^{-D_{ij}}}{\sum_r\pi_{ir}e^{-D_{ir}}},\qquad
\gamma_{ij}=\frac{\pi_{ij}e^{-E_{ij}}}{\sum_r\pi_{ir}e^{-E_{ir}}}.
\tag{B8}
\]

This is the unique minimizer: the Lagrange equations are \(D_{ij}+\log(\beta_{ij}/\pi_{ij})+1+\xi_i=0\), the Hessian is positive on normalized tangents, and a zero component cannot minimize with positive prior and finite cost. Differentiating the reduced \(-\log\sum_j\pi_{ij}e^{-D_{ij}}\) gives \(\sum_j\beta_{ij}dD_{ij}\) exactly.

Set \(m_i=\mathbb E_{q_i}Z=\tanh(L_i/2)\), \(t_i=\mathbb E_{q_i}[Z\cos\varphi]\), and \(S_i=\mathbb E_{q_i}\log q_i\). Then

\[
D_{ii}=S_i-Jt_i+\log I_0(J),\qquad
D_{ij}=S_i-\frac{L_jm_i}2+\log\cosh(L_j/2).
\tag{B9}
\]

The common entropy term cancels from every full Gibbs row. In particular the exact unnormalized row weights are

\[
w_{ii}=\frac{a e^{Jt_i}}{I_0(J)},\qquad
w_{ij}=\frac{p e^{L_jm_i/2}}{\cosh(L_j/2)},\qquad
\beta_{ij}=\frac{w_{ij}}{\sum_rw_{ir}}.
\tag{B10}
\]

They are independent of the \(\psi\)-amplitude and phase. They do depend on the \(\varphi\)-sector through \(L_i,L_j,t_i\). The model rows \(\gamma\) depend only on \(n\).

The own-law Fisher metric is \(\int (v v'/q_i)d\mu_0+\sum_w\zeta(w)\zeta'(w)/s_i(w)\) on normalized tangents. Equivalently, in natural coordinates its belief block is the covariance matrix of the sufficient statistics \((z,\cos\varphi,\sin\varphi,z\cos\varphi,z\sin\varphi,\cos\psi,\sin\psi)\), with the first natural coordinate \(l/2\). This covariance is positive definite: a linear combination that is almost surely constant under a positive density is constant under the reference measure, and separation of its Fourier modes and sign parity forces every coefficient to vanish. Therefore the parameterization has no hidden zero tangent direction.

The exact own Fisher flow stays within (B1), since its log-density velocity is a linear combination of the same statistics plus the normalization constant. Matching coefficients gives

\[
\begin{aligned}
\dot l_i&=-l_i+\sum_{j\ne i}\beta_{ij}L_j+\epsilon o_i,\\
\dot A_i&=-A_i+\epsilon e_1,\\
\dot C_i&=-C_i+\beta_{ii}\epsilon^3e_1,\\
\dot B_i&=-B_i,\\
\dot n_i&=-n_i+\sum_{j\ne i}\gamma_{ij}n_j.
\end{aligned}
\tag{B11}
\]

Thus \(\dot b_i=-b_i\), \(\dot\theta_i=0\) when \(b_i>0\). This is the restriction of the full own-law Fisher field, not a projection approximation or a changed clock. All peer laws are conditioning data in the receiver's own derivative; their evolution and the Gibbs dependence enter the coupled vector field itself.

The coupling to \(\varphi\) is weak but nonzero for positive \(\epsilon\). For example, at \(A=\alpha e_1,C=ce_1\) with \(c>0\),

\[
\frac{\partial L}{\partial\alpha}=h_1'(\alpha+c)-h_1'(\alpha-c)>0,
\qquad h_1(r)=\log I_0(r),
\tag{B12}
\]

because \(h_1''(r)=\operatorname{Var}_r(\cos\varphi)>0\) for every finite \(r\). The derivative in the \(e_1\) direction is also positive for any \(A\) when \(C=ce_1\), since it is the integral of \(\partial_{11}h=\operatorname{Var}(\cos\varphi)>0\) over a segment of length \(2c\). If initially \(C=0\), (B11) generates a strictly positive \(e_1\)-component at every positive time. For a peer \(j\ne i\), holding the receiving law and other public odds fixed,

\[
\frac{\partial\beta_{ij}}{\partial L_j}
=\frac12\beta_{ij}(1-\beta_{ij})(m_i-m_j).
\tag{B13}
\]

This is nonzero whenever the two public means differ. If \(U_i=\sum_{k\ne i}\beta_{ik}L_k\), then

\[
\frac{\partial U_i}{\partial L_j}
=\beta_{ij}\left[1+\frac{(L_j-U_i)(m_i-m_j)}2\right]>0
\tag{B14}
\]

throughout a sufficiently small public-odds neighborhood. Equations (B12)–(B14) explicitly exhibit an angular change in the peer row and receiving force. No claim that the hidden \(\psi\)-phase enters those forces is made. The geometric consequences of these variables require a separate calculation.

## 3. Exact null and a uniform invariant region

The null deletes the nonself KL costs but retains their labels, priors and row entropy. The channels and self evaluator are already fixed, so there is no surviving hidden peer dependence to freeze. Its content self weight is

\[
\beta_{ii}^0=\frac{a e^{-D_{ii}}}{a e^{-D_{ii}}+2p},
\tag{B15}
\]

and its exact Fisher equations are

\[
\begin{aligned}
\dot l_i^0&=-\beta_{ii}^0l_i^0+\epsilon o_i,\\
\dot A_i^0&=-\beta_{ii}^0A_i^0+\epsilon e_1,\\
\dot C_i^0&=\beta_{ii}^0(\epsilon^3e_1-C_i^0),\\
\dot B_i^0&=-\beta_{ii}^0 B_i^0,\\
\dot n_i^0&=-\gamma_{ii}^0n_i^0.
\end{aligned}
\tag{B16}
\]

Here \(\gamma_{ii}^0=a e^{-E_{ii}}/(a e^{-E_{ii}}+2p)\). Common angular entropy does not cancel from (B15), because the other costs were deleted. It must therefore be bounded in the null proof.

Let \(0<\epsilon\leq1/100\) and define the following closed region for each agent:

\[
\frac14\leq\frac{o_i l_i}{\epsilon}\leq12,\qquad
|A_i|\leq\frac1{10},\qquad
|C_i|\leq2\epsilon^3,\qquad
|B_i|\leq\frac1{10}.
\tag{B17}
\]

It is forward invariant under the null. The proof supplies constants rather than assuming small perturbations remain small indefinitely.

Use natural parameters \(\eta=(l/2,A,C,B)\) and \(\eta_G=(0,0,Je_1,0)\). The sufficient-statistic vector has squared norm exactly 4. Its log-partition Hessian is its covariance, so its operator norm is at most 4 for every finite parameter. The integral Taylor remainder for the exponential-family KL gives

\[
D(q_\eta\Vert q_{\eta_G})
\leq2|\eta-\eta_G|^2
\leq72\epsilon^2+\frac1{25}+18\epsilon^6<\log2
\tag{B18}
\]

on (B17). The final inequality follows already from \(\epsilon\leq1/100\) and \(\log2>1/2\). Since every KL is nonnegative, (B15) consequently obeys the exact bounds

\[
\frac19\leq\beta_{ii}^0\leq\frac15.
\tag{B19}
\]

Put \(r_i=o_i l_i/\epsilon\). Then \(\dot r_i=1-\beta_{ii}^0r_i\), which points inward at both \(r_i=1/4\) and \(r_i=12\). At \(|A_i|=1/10\), its outward radial derivative is at most \(\epsilon-1/90<0\). At \(|C_i|=2\epsilon^3\), the radial derivative is at most \(-\beta_{ii}^0\epsilon^3<0\). The \(B\) norm decreases. Therefore no face of the region permits a first exit. Bounds (B18)–(B19) hold for the entire trajectory by this argument, not by extrapolation from a finite interval.

The public odds keep the same signs as \(l_i\). In fact \(\nabla h(V)=\mathbb E_V e_\varphi\), \(\nabla h(0)=0\), and \(\nabla^2h(V)=\operatorname{Cov}_V(e_\varphi)\) has operator norm at most one. Hence

\[
|h(A+C)-h(A-C)|
\leq2|C|(|A|+|C|)
\leq\frac25\epsilon^3+8\epsilon^6.
\tag{B20}
\]

For \(\epsilon\leq1/100\), this is less than \(\epsilon/8\). Combining (B17) and (B20) gives

\[
o_iL_i\geq\frac\epsilon8>0
\quad\text{for every finite null time}.
\tag{B21}
\]

The parameter bounds also give global continuation of the null flow; its \(n\) equation cannot blow up at finite time. If \(b_i(0)>0\), then \(b_i(t)=b_i(0)\exp[-\int_0^t\beta_{ii}^0(s)ds]>0\) at every finite time. Thus the positive-amplitude chart is retained throughout this comparison, although no uniform positive amplitude lower bound as \(t\to\infty\) is claimed.

## 4. Full-flow persistence with fixed small angular marks

The following compact initial patches are allowed. Let \(\delta=1/20\). Take \(l/\epsilon\) near \(u^{\mathrm{in}}=(4,1/2,-1)\), \(|A_i(0)|<\delta\), \(0<|B_i(0)|<\delta\), and \(|C_i(0)|<\epsilon^3\), with \(n_i(0)\) in any fixed bounded open intervals. In particular, choose any compact subpatch

\[
A_1(0)=r e_\chi,\quad r\in[r_-,r_+]\subset(0,\delta),\quad
A_2(0)=d e_1,\quad 0<d<\delta,
\tag{B22}
\]

with \(\chi\) in a small compact interval about zero, other \(A_i(0)\) fixed small, \(C_i(0)=0\), and \(b_i(0)\) fixed in \((0,\delta)\). The proof is uniform over this compact patch and over a sufficiently small relative-open neighborhood of it. Positivity of the first angular component, if desired, is retained by both equations for \(A\).

For the full flow define \(u_i=l_i/\epsilon\), \(c_i=C_i/\epsilon^3\), while leaving \(A_i\) unscaled. On bounded sets of \((u,c,A)\), the function

\[
\frac{L_i}{\epsilon}
=u_i+\frac{h(A_i+\epsilon^3c_i)-h(A_i-\epsilon^3c_i)}{\epsilon}
=u_i+O(\epsilon^2)
\tag{B23}
\]

extends analytically to \(\epsilon=0\), with the error and its first derivatives uniformly bounded by a constant times \(\epsilon^2\). This follows by integrating \(\nabla h\) on the segment between \(A_i-\epsilon^3c_i\) and \(A_i+\epsilon^3c_i\); all derivatives of \(h\) are bounded on the chosen compact set. Consequently \(m_i=O(\epsilon)\). Also \(t_i=O(\epsilon)\): at \(l=C=0\), the sign is uniform and independent of \(\varphi\), and differentiating the positive finite-dimensional density gives a uniform first-order bound in \((l,C)\). Equation (B10) now gives

\[
\beta_{ij}=\pi_{ij}+O(\epsilon^2)
\tag{B24}
\]

uniformly with first derivatives in the scaled variables, despite the \(A_i\) and \(b_i\) initially being independent of \(\epsilon\). This is where the exact entropy cancellation is needed; applying the small-density KL argument without it would leave an unjustified fixed angular error in the full rows.

The autonomous subsystem \(x=(u,c,A)\) therefore satisfies

\[
\begin{aligned}
\dot u&=M u+o+O_{C^1}(\epsilon^2),
&M&=-\frac75 I+\frac25\mathbf1\mathbf1^\top,\\
\dot c_i&=-c_i+a e_1+O_{C^1}(\epsilon^2),\\
\dot A_i&=-A_i+\epsilon e_1.
\end{aligned}
\tag{B25}
\]

It is independent of \(B,n\). The \(\epsilon=0\) field is affine with equilibrium

\[
x_*:\qquad
u_*=(15/7,15/7,5/7),\qquad c_{i,*}=a e_1,\qquad A_{i,*}=0.
\tag{B26}
\]

Its derivative is symmetric negative definite: the eigenvalues of \(M\) are \(-1/5,-7/5,-7/5\), and the other blocks are \(-I\). The precise implicit-function fact used here is that a continuously differentiable map \(f(x,\epsilon)\) with \(f(x_*,0)=0\) and invertible \(D_xf(x_*,0)\) has a unique local continuously differentiable zero branch. The analytic extension in (B23)–(B25) supplies its hypotheses. Thus a nearby equilibrium \(x_\epsilon\) exists for all sufficiently small positive \(\epsilon\), with

\[
u_\epsilon=u_*+O(\epsilon^2),\qquad
c_{i,\epsilon}=a e_1+O(\epsilon^2),\qquad A_{i,\epsilon}=\epsilon e_1.
\tag{B27}
\]

The derivative of the full field differs from the limiting derivative by \(O(\epsilon^2)\); the \(+\epsilon e_1\) angular forcing has zero state derivative. On a fixed small ball around \(x_*\), its symmetric part is therefore at most \(-I/10\) for sufficiently small \(\epsilon\). After recentering at \(x_\epsilon\), the integral mean-value identity gives

\[
\frac{d}{dt}|x-x_\epsilon|^2
=2(x-x_\epsilon)\cdot[f_\epsilon(x)-f_\epsilon(x_\epsilon)]
\leq-\frac15|x-x_\epsilon|^2
\tag{B28}
\]

in a smaller ball. That ball is forward invariant and attracting. It can be chosen so that every public \(L_i\) is positive and every pair ratio \(\min(L_i,L_j)/\max(L_i,L_j)\) exceeds \(1/4\), since the limiting smallest ratio is \(1/3\). These strict margins survive (B23) and (B27).

The limiting affine solution tends to \(x_*\) uniformly over the compact initial patch (B22) and fixed small compact neighborhoods of \(u^{\mathrm{in}},c(0)\). Choose one finite \(T\), independent of \(\epsilon\), at which every such limiting trajectory lies strictly inside the attraction ball. Finite-time continuous dependence then places every exact trajectory there for all sufficiently small \(\epsilon>0\), uniformly over a smaller relative-open neighborhood of that compact initial patch. This finite-time statement can be checked directly by subtracting the two integral equations on a common compact tube and applying the scalar bound \(d(t)\leq d(0)+\int_0^t[Kd(s)+C\epsilon]ds\); the tube remains valid for small \(\epsilon\) by its strict margin. Equation (B28), rather than a further finite-time estimate, supplies the subsequent all-time persistence.

Full solutions exist for every finite time. Indeed \(A,B,C\) are bounded by their initial values and fixed forcing bounds in (B11), while \(|L_i-l_i|\leq2|C_i|\) follows from \(|\nabla h|\leq1\). Thus the maximum absolute \(l_i\) has at most a linear growth bound, and the same is true for \(n\). These bounds prevent finite-time escape of the natural parameters.

For the geometry patch, the exact full equations give

\[
A_i(T)=e^{-T}A_i(0)+\epsilon(1-e^{-T})e_1,
\qquad b_i(T)=e^{-T}b_i(0).
\tag{B29}
\]

At the \(\epsilon=0\) limit, both marks in (B22) and the positive \(b_i\) remain nonzero at this same finite \(T\). Thus the grouping proof permits a compact nonzero angular patch at a common finite post-selection time. It does not itself calculate any connection or curvature. Since \(b_i(t)\to0\), no uniform positive phase-metric bound at infinite time follows from this argument.

## 5. The public selector and exact counterfactual

Before applying either flow, connect agents \(i,j\) precisely when their public odds \(L_i,L_j\) are nonzero, have the same sign, and obey

\[
\frac{\min(|L_i|,|L_j|)}{\max(|L_i|,|L_j|)}>\frac14.
\tag{B30}
\]

Blocks are the connected components. The rule is permutation covariant and uses the actually transmitted sign marginal, not the internal parameter \(l_i\). At \(C(0)=0\), \(L(0)=l(0)\), so the only same-sign initial pair has ratio \(1/8\). Equation (B23) preserves these initial strict singleton conditions on a relative-open neighborhood including \(C(0)\ne0\). All patches above can be chosen inside the strict interior of (B17), apart from harmless fixed equalities used to specify the compact subpatch; the neighborhood itself has no such equality restriction.

For every sufficiently small fixed \(\epsilon>0\), the full flow enters the region in which all three pair conditions are strict by the fixed finite time \(T\), and stays there by (B28). Its partition therefore contains the persistent block \(\{1,2,3\}\). The null uses the same initial point, fixed evaluator, source priors, record values, row convention and selector. Equation (B21) keeps its first two public odds positive and its third negative for all finite time. Its partition never contains \(\{1,2,3\}\). Removing all event times cannot make the two partition sequences equal.

This proves the following quantified statement: there exists \(\epsilon_0\in(0,1/100]\) such that for every fixed \(0<\epsilon<\epsilon_0\), there is a nonempty relatively open initial set in the invariant family (B1), containing neighborhoods of the compact marked patch (B22), for which (B30) starts with singleton blocks, selects the three-agent block under the full flow by one common finite time, and retains it thereafter, whereas its null never selects that block. The initial records remain fixed throughout this assertion; later assimilation is a separate obligation. A common \(T\) and \(\epsilon_0\) can be chosen uniformly over the specified compact angular patch. Simultaneous permutations of the initial states and observed records produce a relabeling-covariant admitted class.

The theorem extends uniformly to every retained angular record vector \(\omega=(\omega_1,\omega_2,\omega_3)\in\mathbb T^3\). Keep the binary records \(o=(1,1,-1)\), and replace the record forcing in the full and null \(A_i\) equations by \(\epsilon e_{\omega_i}\). Every null radial estimate in (B17)–(B21) is unchanged because \(|e_{\omega_i}|=1\); those estimates never required a positive first angular component. The scaled full field changes only through the exact forcing \(+\epsilon e_{\omega_i}\) in \(\dot A_i\), and its equilibrium has \(A_{i,\epsilon}=\epsilon e_{\omega_i}\). All error and derivative bounds used in (B23)–(B28) are uniform in the compact angular-record torus, while the limiting field is independent of \(\omega\). The same finite-entry argument therefore gives a common \(\epsilon_0,T\), attraction ball, and initial neighborhood for all angular records. In particular,

\[
A_i(T)=e^{-T}A_i(0)+\epsilon(1-e^{-T})e_{\omega_i},
\tag{B31}
\]

with the same nonzero limiting marks as (B29). Angular records can consequently remain random under a separately specified normalized preparation/record law on the full torus: the grouping conclusion holds conditionally for every angular record in that law. This uniform statement does not specify that joint law, erase the angular records, or prove any post-handoff history identity. No new record is assimilated during the persistence interval.

If instead \(A(0)=O(\epsilon)\) and \(b(0)=O(\epsilon)\), the same theorem applies as a specialization. Under the null, (B19) bounds \(|A_i(t)|\) by \(\max(|A_i(0)|,9\epsilon)\), while \(b_i\) decreases and \(|C_i|\leq2\epsilon^3\). Thus (B20) sharpens to \(|L-l|=O(\epsilon^4)\) uniformly in time. Under the full flow, \(A_i/\epsilon\) converges toward \(e_{\omega_i}\), equal to \(e_1\) for the baseline records. The fixed-small-mark version above is useful because it retains nonzero angular marks in the \(\epsilon\to0\) geometry limit at a fixed finite time.

## 6. Scope left for the integrated construction

The bridge supplies an exact normalized fixed-evaluator realization, an invariant positive exponential family, persistent nonconstant records, a genuine dependence of public forces and Gibbs rows on the \(\varphi\)-sector, and an interaction-dependent persistent partition on a relative-open initial class. Its null conclusion is supported by a uniform invariant-region proof. The \(\psi\)-phase remains absent from public forces and full rows, exactly as the equations show.

The next obligation is to combine this same family, without changing its fixed protocol or evaluator, with an explicit noninjective retention and a calculation of the actual collective minimizing connection and contextual tensor. In particular a phase that was operationally invisible before grouping is not by itself evidence that a sufficient interface was selected endogenously. The present grouping theorem neither resolves that distinction nor proves the full run contract. No scripts, numerical experiments, GPU work, or Lean work were run. Only this assigned route file was written.
