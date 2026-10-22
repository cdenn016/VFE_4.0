<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca","schema_version":"rigorous-theory-search/v1","target_digest":"0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca"} -->
# Exact per-agent transfer and an interaction-dependent selection example

Date: 2026-10-22. This route concerns the mandatory four-state transfer subsidiary of the run contract. It proves exact own-block and null vector fields, classifies the linearization at the uniform law, and constructs a state-selected two-agent block on a full open set of positive tables with persistent nonconstant records. The block persists on a nonzero time window and never occurs under its null control. It does not construct a compressed inherited parent or prove the full existential target.

The supplied target requires normalized per-agent mixture-source models, their own Fisher updates, the prescribed deletion of nonself costs, and a comparison of partition sequences with event times removed. Its full equivalence requirement includes original-time dynamics, all admitted joint adaptive response-record histories, and collective geometry after positive-dimensional physical compression. The additional choices made here are finite four-state spaces, a product recognition family, exactly Gibbs-reduced source labels at temperature one, homogeneous positive source priors, a fixed evaluator, and the specific normalized kernels below. No global normalized law over agents, global potential, stochastic preparation law, or parent dynamics is assumed.

Source continuity: the current [Chapter 3](../../../../../Research/manuscripts/MAgent_Theory_2.0/chapters/03_dynamics.tex), particularly its mixture-source and record constructions, and [Chapter 5](../../../../../Research/manuscripts/MAgent_Theory_2.0/chapters/05_inheritance.tex), from line 1375 onward, determine the per-agent and null conventions. The earlier [variational-selection route](../../2026-10-22-endogenous-interface-selection/routes/variational-selection.md) supplies the conditional-interface algebra. Its reciprocal-potential stability proof is not used below. The manuscript sources were read in the live Research checkout; their mathematical claims are assessed separately here.

## 1. Normalized model and conditional interface

Let the population have size \(N\geq2\), with belief states \(k=(a,x)\in\{-1,1\}^2\) and model states \(m=(b,u)\in\{-1,1\}^2\). All reference measures are counting measures. Each \(q_i(a,x)\) and \(s_i(b,u)\) is an arbitrary strictly positive normalized table. Write

\[
m_i=\mathbb E_{q_i}A,\qquad n_i=\mathbb E_{s_i}B.
\]

Fix \(J>1\), \(0<\alpha,\kappa<1\), and self-source probabilities \(0<\rho_q,\rho_s<1\). The homogeneous source priors are

\[
\pi^q_{ii}=\rho_q,\quad \pi^q_{ij}=\frac{1-\rho_q}{N-1}\ (j\ne i),\qquad
\pi^s_{ii}=\rho_s,\quad \pi^s_{ij}=\frac{1-\rho_s}{N-1}\ (j\ne i).
\]

The self reference and evaluator are

\[
r(b,u)=\tfrac14,\qquad G((b,u),(a,x))=\frac{\exp(Jab)}{4\cosh J}.
\]

For every model state, \(\sum_{a,x}G((b,u),(a,x))=1\). For a positive four-state law \(p\), define

\[
\mu_p(x)=\sum_a p(a,x),\quad L_p(b\mid x)=\frac{p(b,x)}{\mu_p(x)},\quad
W_b(y\mid x)=\frac{1+\kappa bxy}{2},
\]

\[
K[p]((a,x),(b,y))=
\bigl[\alpha\mathbf1_{a=b}+(1-\alpha)L_p(b\mid x)\bigr]W_b(y\mid x).
\]

Every row of \(K[p]\) sums to one and is strictly positive. It is invertible: first apply the two-by-two interface matrix \(\alpha I+(1-\alpha)\mathbf1L_p(\cdot\mid x)\) at each \(x\), whose determinant is \(\alpha\), and then apply \(W_b\) at each \(b\), whose determinant is \(\kappa b\). Thus no rank loss has been inserted into this channel. Direct summation gives

\[
(pK[p])(b,y)=\sum_xp(b,x)W_b(y\mid x)=:(pW)(b,y).
\tag{1}
\]

The right-hand side uses a fixed stochastic matrix \(W\) which preserves the interface sign and mixes content. Its entries can vanish between different interface signs, but \(pW\) is strictly positive for every positive \(p\).

The state derivative is not discarded. For any normalized tangent \(v\),

\[
DL_p[v](b\mid x)=
\frac{v(b,x)\mu_p(x)-p(b,x)\sum_av(a,x)}{\mu_p(x)^2},
\]

and differentiation of (1), including \(DK[p]\), gives \(D(pK[p])[v]=vW\). For agent \(i\), put

\[
\nu_j^q=q_jW,\quad \nu_j^s=s_jW,\quad
R_{i\ell}(m)=\begin{cases}r(m),&\ell=i,\\\nu_\ell^s(m),&\ell\ne i,\end{cases}
\]

\[
T_{ij}(m,k)=\begin{cases}G(m,k),&j=i,\\\nu_j^q(k),&j\ne i.\end{cases}
\]

Given the current peer laws, agent \(i\)'s normalized joint law is

\[
P_i(m,k,j,\ell,o\mid q_{-i},s_{-i})=
\pi^s_{i\ell}R_{i\ell}(m)\pi^q_{ij}T_{ij}(m,k)\mathcal L_i(o\mid k,m).
\tag{2}
\]

Summing in order over \(o,k,j,m,\ell\) proves normalization. The peer laws are conditioning data for this local model. Their later changes couple the agents' vector fields, but do not turn (2) into a common global joint law.

## 2. Persistent records and the exact free energy

For one observed pair \(o=(o_q,o_s)\in\{-1,1\}^2\), take

\[
\mathcal L_h(o\mid a,x,b,u)=
\frac{\exp(h_qo_qa)}{2\cosh h_q}
\frac{\exp(h_so_sb)}{2\cosh h_s},\qquad h_q,h_s>0.
\tag{3}
\]

This likelihood is strictly positive, normalized over \(o\), and nonconstant in the latent signs. A finite retained record word \(o_{i,1:M_i}\), conditionally independent given the same latent pair, has the normalized product likelihood. Its negative expected log likelihood is

\[
\begin{aligned}
\mathcal C_i
&=\mathbb E_{q_i\otimes s_i}
 \left[-\sum_{t=1}^{M_i}\log\mathcal L_{h_t}(o_{i,t}\mid K,M)\right]\\
&=C_i-H_i^q m_i-H_i^s n_i,
\end{aligned}
\tag{4}
\]

where \(H_i^q=\sum_th_{q,t}o_{q,i,t}\), \(H_i^s=\sum_th_{s,t}o_{s,i,t}\), and \(C_i=\sum_t[\log(2\cosh h_{q,t})+\log(2\cosh h_{s,t})]\). These fields remain in the objective after the records arrive. The signed weighted sums and \(C_i\) suffice to evaluate this static likelihood term; they do not by themselves preserve a protocol which can query the ordered record word. The construction below conditions on one realized record per agent and makes no claim about a preparation distribution or all adaptive record processes.

Choose the recognition family

\[
Q_i(m,k,j,\ell)=s_i(m)q_i(k)\beta_{ij}\gamma_{i\ell},
\qquad \beta_i,\gamma_i\in\Delta_{N-1}.
\]

For the observed record, the negative ELBO is defined before simplification by

\[
F_i=\mathbb E_{Q_i}\left[
\log\frac{s_i(M)q_i(K)\beta_{iJ}\gamma_{iL}}
{\pi^s_{iL}R_{iL}(M)\pi^q_{iJ}T_{iJ}(M,K)\mathcal L_i(o_i\mid K,M)}\right].
\tag{5}
\]

All sums are finite; positivity gives absolute continuity and finite logarithms. On each compact subset of the positive tables the constitutive data and their derivatives are analytic and bounded. Define the self and peer costs by their expectations:

\[
\begin{aligned}
A_i&=\mathbb E_{q_i\otimes s_i}[\log q_i(K)-\log G(M,K)]
=\sum_kq_i(k)\log q_i(k)-Jm_in_i+\log(4\cosh J),\\
B_i&=\mathbb E_{s_i}[\log s_i(M)-\log r(M)]
=\sum_ms_i(m)\log s_i(m)+\log4,\\
D^q_{ij}&=\mathbb E_{q_i}[\log q_i(K)-\log\nu_j^q(K)],\\
D^s_{ij}&=\mathbb E_{s_i}[\log s_i(M)-\log\nu_j^s(M)]\quad(j\ne i).
\end{aligned}
\]

Equation (5) is exactly

\[
F_i=\beta_{ii}A_i+\sum_{j\ne i}\beta_{ij}D^q_{ij}
+\operatorname{KL}(\beta_i\Vert\pi_i^q)
+\gamma_{ii}B_i+\sum_{j\ne i}\gamma_{ij}D^s_{ij}
+\operatorname{KL}(\gamma_i\Vert\pi_i^s)+\mathcal C_i.
\tag{6}
\]

The product recognition restriction matters. If a correlated recognition law \(R_i(k,m)\) has the same marginals and remains independent of the labels, then at fixed rows

\[
F_i[R_i]-F_i[q_i\otimes s_i]
=I_{R_i}(K;M)-\beta_{ii}J\operatorname{Cov}_{R_i}(A,B).
\tag{7}
\]

The record likelihood in (3) contributes no additional correction because it separates in the two signs. After row reduction the exact correction is instead

\[
I_{R_i}(K;M)-\log\frac{
\rho_qe^{-A_i+J\operatorname{Cov}_{R_i}(A,B)}+
\sum_{j\ne i}\pi^q_{ij}e^{-D^q_{ij}}}
{\rho_qe^{-A_i}+\sum_{j\ne i}\pi^q_{ij}e^{-D^q_{ij}}}.
\]

Thus no correlated-family equivalence is claimed.

## 3. Gibbs rows and own-block Fisher dynamics

For finite costs \(c_j\), minimization of \(\sum_jb_jc_j+\sum_jb_j\log(b_j/\pi_j)\) subject to \(\sum_jb_j=1\) gives

\[
c_j+\log(b_j/\pi_j)+1+\lambda=0,
\qquad b_j=\frac{\pi_je^{-c_j}}{\sum_\ell\pi_\ell e^{-c_\ell}}.
\]

The diagonal Hessian \(1/b_j\) is positive on normalized tangents, and the entropy directional derivative excludes a zero minimizer component when \(\pi_j>0\); this is the unique constrained optimum. Accordingly,

\[
Z_i^q=\rho_qe^{-A_i}+\sum_{j\ne i}\pi^q_{ij}e^{-D^q_{ij}},\qquad
Z_i^s=\rho_se^{-B_i}+\sum_{j\ne i}\pi^s_{ij}e^{-D^s_{ij}},
\]

\[
\beta_{ii}=\rho_qe^{-A_i}/Z_i^q,\quad
\beta_{ij}=\pi^q_{ij}e^{-D^q_{ij}}/Z_i^q,
\quad \gamma_{ii}=\rho_se^{-B_i}/Z_i^s,\quad
\gamma_{ij}=\pi^s_{ij}e^{-D^s_{ij}}/Z_i^s.
\tag{8}
\]

The reduced objective is \(\bar F_i=-\log Z_i^q-\log Z_i^s+\mathcal C_i\). Differentiation of \(-\log Z\) gives the Gibbs-weighted cost differentials exactly. No derivatives of optimized rows remain in this first derivative; row derivatives do enter its Jacobian.

Use the own-state Fisher metric

\[
g_i((v,w),(v',w'))=\sum_k\frac{v(k)v'(k)}{q_i(k)}
+\sum_m\frac{w(m)w'(m)}{s_i(m)},\quad
\sum_kv(k)=\sum_mw(m)=0.
\tag{9}
\]

The negative natural-gradient flow, at unit mobility and in the original time, is

\[
\dot q_i(k)=-q_i(k)(h_i^q(k)-\mathbb E_{q_i}h_i^q),\qquad
\dot s_i(m)=-s_i(m)(h_i^s(m)-\mathbb E_{s_i}h_i^s),
\tag{10}
\]

with potentials modulo state-independent constants

\[
\begin{aligned}
h_i^q(a,x)&=\log q_i(a,x)-\sum_{j\ne i}\beta_{ij}\log\nu_j^q(a,x)
-\beta_{ii}Jn_i a-H_i^q a,\\
h_i^s(b,u)&=\log s_i(b,u)-\sum_{j\ne i}\gamma_{ij}\log\nu_j^s(b,u)
-\beta_{ii}Jm_i b-H_i^s b.
\end{aligned}
\tag{11}
\]

The model-sector evaluator force is weighted by \(\beta_{ii}\), not \(\gamma_{ii}\). In each own derivative, every nonself \(\nu_j\) depends only on a peer law. The absence of a sender force in (11) is therefore exact for the specified local dynamics. Peer transport and row derivatives must still be included when differentiating the coupled vector field. A negative gradient of a sum of all agents' reduced objectives would be a different dynamics.

For the coordinate change \(p(a,x)=\mu(x)L(a\mid x)\), the metric becomes

\[
\sum_{a,x}\frac{dp(a,x)^2}{p(a,x)}
=\sum_x\frac{d\mu(x)^2}{\mu(x)}
+\sum_x\mu(x)\sum_a\frac{dL(a\mid x)^2}{L(a\mid x)}.
\tag{12}
\]

The cross terms vanish because \(\sum_adL(a\mid x)=0\). Equation (10) consequently implies

\[
\dot L(a\mid x)=-L(a\mid x)\bigl(h(a,x)-\mathbb E_{L(\cdot\mid x)}h(\cdot,x)\bigr).
\tag{13}
\]

This is a coordinate transformation of the same flow, with no change of mobility or clock. Analyticity on the positive chart gives a unique local solution there. No common dissipation identity for the coupled population is asserted.

## 4. The prescribed null control

Delete each nonself KL cost while retaining its source label, prior probability and row entropy. Freeze any hidden peer dependence; here the self evaluator is fixed and the deleted costs contain all peer dependence, so freezing introduces no surviving term. Use the same exact Gibbs convention. Then

\[
Z_i^{q,0}=\rho_qe^{-A_i}+1-\rho_q,\qquad
Z_i^{s,0}=\rho_se^{-B_i}+1-\rho_s,
\]

\[
b_i^0=\frac{\rho_qe^{-A_i}}{Z_i^{q,0}},\qquad
g_i^0=\frac{\rho_se^{-B_i}}{Z_i^{s,0}},
\qquad \bar F_i^0=-\log Z_i^{q,0}-\log Z_i^{s,0}+\mathcal C_i.
\tag{14}
\]

Its Fisher potentials are

\[
h_i^{q,0}(a,x)=b_i^0[\log q_i(a,x)-Jn_i a]-H_i^q a,
\]

\[
h_i^{s,0}(b,u)=g_i^0\log s_i(b,u)-b_i^0Jm_i b-H_i^s b.
\tag{15}
\]

Replacing the rows by self-only rows would incorrectly remove the factors \(b_i^0,g_i^0\). The null agents evolve independently, each as a true Fisher gradient flow of its own reduced objective.

There is an exact full-table invariant relevant to selection. Set

\[
c_i^q(x)=\mathbb E[A\mid X=x],\qquad c_i^s(u)=\mathbb E[B\mid U=u].
\]

Equations (13) and (15) give

\[
\begin{aligned}
\dot c_i^q(x)&=(1-c_i^q(x)^2)
\{b_i^0[Jn_i-\operatorname{atanh}c_i^q(x)]+H_i^q\},\\
\dot c_i^s(u)&=(1-c_i^s(u)^2)
\{b_i^0Jm_i-g_i^0\operatorname{atanh}c_i^s(u)+H_i^s\}.
\end{aligned}
\tag{16}
\]

If all four conditionals of an agent have sign \(\sigma\in\{-1,1\}\), and \(\sigma H_i^q,\sigma H_i^s>0\), that strict sign cone is forward invariant. At a first zero of any conditional, the other conditionals have nonnegative signed values, hence so do \(\sigma m_i,\sigma n_i\); the signed derivative in (16) is strictly positive, contradicting a first exit. This applies to arbitrary positive tables, including content-sign correlations.

Null solutions remain positive at every finite time. For any two belief states, their log ratio satisfies

\[
\frac{d}{dt}\log\frac{q_i(a,x)}{q_i(a',x')}
=-b_i^0\log\frac{q_i(a,x)}{q_i(a',x')}
+(b_i^0Jn_i+H_i^q)(a-a').
\]

Since \(0<b_i^0<1\), \(|n_i|\leq1\), its magnitude stays finite on each bounded interval by the scalar integral inequality. The model log ratios obey the same bound with \(g_i^0\) and \(b_i^0Jm_i+H_i^s\). Finite log ratios and normalization bound all probabilities away from zero on each finite interval, so the local solution continues for all finite times. The null sign invariant therefore is not merely a short-time statement.

## 5. Homogeneous reference, stationary points, and stability

The synchronized product-content family

\[
q_i(a,x)=\frac{1+ma}{4},\qquad s_i(b,u)=\frac{1+nb}{4}\quad\text{for every }i
\tag{17}
\]

is invariant when the retained record fields are common across agents. On it, \(q_iW=q_i\), \(s_iW=s_i\), all peer costs vanish, and all peer own-gradients are constant on the relevant tangent. The full and null vector fields coincide exactly on this entire family, including their Gibbs rows. With

\[
e(m)=\sum_{a=\pm1}\frac{1+ma}{2}\log\frac{1+ma}{2},
\]

\[
A=e(m)-Jmn+\log(2\cosh J),\qquad B=e(n)+\log2,
\]

and \(b,g\) given by (14), the common flow is

\[
\dot m=(1-m^2)\{b[Jn-\operatorname{atanh}m]+H^q\},
\]

\[
\dot n=(1-n^2)\{bJm-g\operatorname{atanh}n+H^s\}.
\tag{18}
\]

Therefore the record-free stationarity equations are

\[
\operatorname{atanh}m=Jn,\qquad
g\operatorname{atanh}n=bJm.
\tag{19}
\]

The reciprocal model's proposed \(m=n=r_*\), \(\operatorname{atanh}r_*=Jr_*\), is not generically stationary here: it additionally requires \(b=g\) at that point. On (17) no selector driven only by the common state history can distinguish full from null. This is an exact failure of interaction necessity on that reference family, not a nonexistence result for nearby independently variable laws.

For a full-table stability calculation at the record-free uniform state, write

\[
q_i(a,x)=\tfrac14(1+m_i a+v_i x+c_i ax),\qquad
s_i(b,u)=\tfrac14(1+n_i b+w_i u+d_i bu).
\]

Transport maps \((m,v,c)\) to \((m,\kappa c,\kappa v)\). At the origin put

\[
b=\frac{\rho_q}{\rho_q+(1-\rho_q)\cosh J},\qquad g=\rho_s,
\qquad (Af)_i=\frac1{N-1}\sum_{j\ne i}f_j.
\]

All cost first variations vanish at this point, so the Gibbs-row first variations vanish as well. The exact linearization is

\[
\dot m=-m+(1-b)Am+bJn,\qquad
\dot n=-n+(1-g)An+bJm,
\tag{20}
\]

\[
\dot v=-v+\kappa(1-b)Ac,\quad \dot c=-c+\kappa(1-b)Av,
\quad
\dot w=-w+\kappa(1-g)Ad,\quad \dot d=-d+\kappa(1-g)Aw.
\tag{21}
\]

The spatial eigenvalues of \(A\) are \(a=1\) and \(a=-1/(N-1)\). The interface matrix for each spatial mode is

\[
M(a)=\begin{pmatrix}-1+(1-b)a&bJ\\bJ&-1+(1-g)a\end{pmatrix}.
\tag{22}
\]

All content eigenvalues \(-1\mathbin{\pm}\kappa(1-b)a\) and \(-1\mathbin{\pm}\kappa(1-g)a\) are strictly negative. The mean interface mode is stable precisely when \(bJ^2<g\), and is a saddle when \(bJ^2>g\). Every transverse interface mode is stable precisely when

\[
b^2J^2<\frac{N-b}{N-1}\frac{N-g}{N-1}.
\tag{23}
\]

These conclusions follow directly from the negative traces and determinant signs of the symmetric two-by-two matrices. At an equality there is a zero eigenvalue and this calculation makes no nonlinear classification. The null linearization has \(M(1)\) separately at each agent, and content eigenvalues \(-b,-g\). Thus peer coupling can remove transverse instabilities while leaving the common unstable direction unchanged.

Stable nonzero branches also exist for admissible positive peer priors, but they occur for both full and null. Here is a direct local proof. Let \(r_*\in(0,1)\) be the unique positive solution of \(\operatorname{atanh}r_*=Jr_*\). Uniqueness follows because \(\operatorname{atanh}r/r\) increases strictly from 1 to infinity: its numerator is the average of the increasing function \(1/(1-t^2)\) over \([0,r]\). Also \(d=1/(1-r_*^2)>J\). Let \(A_*,B_*\) be the costs in (18) at \(m=n=r_*\). For a parameter \(t\in(0,1)\), choose fixed priors by

\[
\frac{\rho_q}{1-\rho_q}=\frac{t}{1-t}e^{A_*},\qquad
\frac{\rho_s}{1-\rho_s}=\frac{t}{1-t}e^{B_*}.
\tag{24}
\]

Then \(b=g=t\) at both synchronized points \(m=n=\pm r_*\), so they are exact equilibria of both vector fields. As \(t\uparrow1\), each local objective tends to \(A_i+B_i\). At either equilibrium its full-table second variation is

\[
\sum_k\frac{\delta q_i(k)^2}{q_i(k)}
+\sum_m\frac{\delta s_i(m)^2}{s_i(m)}-2J\delta m_i\delta n_i.
\tag{25}
\]

For a prescribed \(\delta m_i\), the first sum is at least \(d\delta m_i^2\) by Cauchy-Schwarz applied to \(A-m_i\); equality leaves only the Bernoulli sign tangent. The same holds for \(n_i\), and all other tangent directions add a strictly positive contribution. Thus (25) is positive definite because \(d>J\). The limiting Fisher-gradient Jacobian is similar to a symmetric negative-definite matrix. The full and null Jacobians depend continuously on \(t\) near 1. Their spectra consequently remain in the negative half-plane for every sufficiently large fixed \(t<1\), giving locally exponentially attracting equilibria on the full independent-table space. For completeness, local attraction follows by taking \(P=\int_0^\infty e^{A_0^\top u}e^{A_0u}du\) for the limiting stable matrix: nearby linearizations obey \(A_t^\top P+PA_t<0\), and the quadratic nonlinear remainder is dominated in a sufficiently small neighborhood. This is a fresh own-objective calculation; no reciprocal Hessian is imported. These shared attracting branches provide no interaction-necessity witness.

## 6. A full-open interaction-dependent grouping with records

Take \(N=2\), \(J=3/2\), \(\rho_s=1/2\), and

\[
\rho_q=\frac{\cosh J}{1+\cosh J}.
\tag{26}
\]

At the record-free uniform state, \(b=g=1/2\). The mean interface eigenvalues are \(1/4,-5/4\); the transverse interface eigenvalues are \(-3/4,-9/4\). All content modes are stable for any fixed \(0<\kappa<1\). The null has the positive eigenvalue \(1/4\) at each agent separately.

First take zero record fields as a reference calculation and initialize the invariant product-content family by

\[
m_1(0)=n_1(0)=3\varepsilon,\qquad
m_2(0)=n_2(0)=-\varepsilon,\qquad 0<\varepsilon<1/3.
\tag{27}
\]

The product-content family with independently variable \(m_i,n_i\) is invariant, because all forces are independent of content. Equality \(m_i=n_i\) is needed only in the linear initial direction and is not assumed invariant for the nonlinear flow. The linear solution in that direction is

\[
m_i(t)=n_i(t)=\varepsilon\left[e^{t/4}+(a_i-1)e^{-3t/4}\right],
\qquad (a_1,a_2)=(3,-1).
\tag{28}
\]

At \(t_*=\log4\), the coefficients are \((3/\sqrt2,1/\sqrt2)\), so both agents have positive conditional interface means. Agent 2's linear means cross zero at \(\log2\) with strictly positive derivative. On any fixed finite interval containing these times, the actual nonlinear solution equals (28) plus \(O(\varepsilon^2)\), in both state and time derivative. To see the scope of the estimate, write the analytic vector field near zero as \(f(z)=Az+R(z)\), \(\lVert R(z)\rVert\leq C\lVert z\rVert^2\), and use

\[
z(t)=e^{At}z(0)+\int_0^te^{A(t-u)}R(z(u))du.
\]

A bound \(\lVert z(t)\rVert\leq C_T\varepsilon\) on the fixed interval closes by continuity for small \(\varepsilon\), giving the stated remainder; substitution into \(\dot z=f(z)\) gives the derivative estimate. This is a finite-time asymptotic estimate used to prove an exact strict inequality for all sufficiently small fixed \(\varepsilon>0\), not a claim of equality with the linear trajectory.

Now fix such an \(\varepsilon\). Introduce the same strictly positive likelihood (3) for both agents with \(h_q=h_s=h>0\), and retain the realized outcomes \((+1,+1)\) for agent 1 and \((-1,-1)\) for agent 2. Thus

\[
H_1^q=H_1^s=h,\qquad H_2^q=H_2^s=-h.
\tag{29}
\]

Choose a sufficiently small fixed \(h>0\). Continuous dependence of the analytic vector field on these fields preserves the full-flow strict positivity at \(t_*\), the transverse sign change near \(\log2\), and a nonzero subsequent interval of positivity. The same conclusions hold on a nonempty full open neighborhood \(U\) of the initial four positive tables, including arbitrary small content marginals and sign-content correlations. The required strict initial conditional signs are preserved by making \(U\) smaller.

Declare the selector before following the trajectory: beginning with two singleton blocks, merge the pair when all eight conditional means \(c_i^q(\pm1),c_i^s(\pm1)\) are at least a fixed threshold \(\theta>0\), or when they are all at most \(-\theta\). For example, choose \(\theta=\varepsilon/(8\sqrt2)\) and then shrink \(\varepsilon,h,U\) as above. The linear agent-2 means are strictly increasing on the crossing interval. Hence for small perturbations each relevant conditional crosses \(\theta\) with positive derivative; the last such crossing triggers the merge, and all stay above threshold on a nonzero common interval afterward. The event time is the maximum of finitely many simple crossing times. It is piecewise smooth and is smooth on subdomains where the last crossing is unique; no globally smooth event surface is assumed.

The null control uses exactly the same initial laws, record outcomes, likelihood, priors, row reduction and selector. On \(U\), agent 1 starts in its positive conditional cone and has positive fields; agent 2 starts in its negative conditional cone and has negative fields. Equation (16) preserves those opposite cones for every finite null time. Consequently the null never qualifies for either merge condition. Removing event times, the full partition history contains

\[
\bigl(\{\{1\},\{2\}\},\{\{1,2\}\}\bigr),
\]

whereas the null history contains only \(\{\{1\},\{2\}\}\). This difference cannot be removed by moving a null event to a later time. The selected full block persists for a stated nonzero interval; no indefinite persistence under the conflicting record fields is asserted.

All constitutive data and the rule are invariant under exchanging the agents. The record values and initial laws move with the agents under relabeling; they are data, not encoded membership labels. Taking the union of \(U\) with its permuted copy gives a relabeling-covariant initial class. The selector uses the same conditionals that determine \(K[p]\), but same-sign selection does not imply identical kernels or an exact sufficient quotient.

## 7. Scope and the remaining obligation

The exact transferred dynamics supports interaction-dependent state grouping on a full open set of positive four-state laws, with a normalized nonconstant record likelihood and an exact prescribed null comparison. The old synchronized reference has full/null identity, and its old nonzero equilibrium condition requires an extra row-weight equality; neither its reciprocal stability nor its interaction necessity transfers automatically.

No child information has been discarded in this route. The grouping rule currently annotates the full child state. It therefore proves neither a positive-dimensional physical compression nor an inherited autonomous flow. It also supplies no equality of adaptive response-record laws, no minimizing collective connection calculation, and no nonzero curvature result. The next material obligation is to fix the admitted boundary protocol for this selected block and construct a noninjective retention, including its event data and necessary record memory, whose fibers are preserved by (10) and whose output kernels preserve every admitted joint response-record law. If every such retention is locally injective for this candidate, that would rule out this candidate's compression, while leaving the contract's general existential target open.

This artifact contains analytical calculations only. No numerical experiment, sampler, model run, GPU operation, or Lean work was performed. Only this assigned route file was written; the Research sources and prior route were read without edits.
