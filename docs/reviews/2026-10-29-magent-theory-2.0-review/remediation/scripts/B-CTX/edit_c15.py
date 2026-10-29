r"""Exact string replacements in companion c15 (each old string must occur exactly once)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
R = [
    # local potential symbol in the finite part
    (r""" V(p,\beta,\gamma)={}&\sum_i\KL(q_i\otimes s_i\Vert R_i)\\""",
     r""" \mathcal E(p,\beta,\gamma)={}&\sum_i\KL(q_i\otimes s_i\Vert R_i)\\"""),
    (r"The product experiment in \cref{comp:ctx:metric} specifies its mobility; it does not make this potential an ELBO of a fixed population model. Own-block per-agent updates omit some of \cref{comp:ctx:roles} and have no general total-energy dissipation identity.",
     r"The product experiment in \cref{comp:ctx:metric} specifies its mobility; it does not make this potential the variational free energy of a fixed population model. Own-block per-agent updates belong to the per-agent objective, omit some of \cref{comp:ctx:roles}, and have no general total-energy dissipation identity. They also need their own regularity hypothesis: smoothness of the sum of all costs does not make each agent's own-block gradient locally Lipschitz, because law-dependent channels can cancel nonsmooth parts between receivers."),
    (r"Let the contextual vertices have masses $m_v>0$ and the unoriented edges have weights $w_e>0$.",
     r"Let the contextual vertices have masses $m_v>0$ and the unoriented edges have weights $w_e>0$. Agent $i$ at one context is compared with agent $i$ at the next; this identification of agents across contexts is part of the model."),
    (r"Independent attending links between agents at one context remain separate data; their opposite directions need not be inverses. Finite edge transports are group elements rather than connection one-forms \citep{LeokMarsdenWeinstein2005}.",
     r"Independent attending links between agents at one context remain separate data; their opposite directions need not be inverses. Finite edge transports are group elements rather than connection one-forms; they transform like the link variables of lattice gauge theory \citep{Wilson1974}, and discrete connections have their own geometric theory \citep{LeokMarsdenWeinstein2005}."),
    (r"For every edge, choose a stationary transport where the compared endpoints have a unique minimizing geodesic, squared distance is smooth, and the transport Hessian is positive definite in group coordinates. Continue only the resulting local minimizing branch $U_e^*(p_u,p_v)$ while these conditions hold.",
     r"For every edge, choose a stationary transport where the compared endpoints have a unique minimizing geodesic, squared distance is smooth, and the transport Hessian is positive definite in group coordinates. Continue only the resulting local minimizing branch $U_e^*(p_u,p_v)$ while these conditions hold. The continuation lives on the set of regular critical pairs $(p,U)$, not on law space: carried around a closed loop of laws, a branch can return to a different transport, and it can stop being the global minimizer without violating any condition."),
    (r"Its transport derivative vanishes, so differentiation of $\mathcal C_*(p)=\mathcal C(p,U^*(p))$ keeps only endpoint derivatives. A local minimizing branch need not realize the global infimum over $G$.",
     r"""Its transport derivative vanishes, so differentiation of $\mathcal C_*(p)=\mathcal C(p,U^*(p))$ keeps only endpoint derivatives. A local minimizing branch need not realize the global infimum over $G$.

Invariance of the edge cost under a common motion of both endpoints, combined with stationarity in $U_e$, makes the endpoint force at $p_v$ Fisher orthogonal to the group orbit: the connecting geodesic is horizontal. The contextual force therefore never rotates frames. Where the selected branch is the global minimizer and the action is free and proper, half the minimized squared distance is half the squared distance between orbits, so $\mathcal C_*$ is a graph Dirichlet energy of the shape map into the quotient $\mathcal N/G$. For planar location laws this is the orthogonal Procrustes problem \citep{Schonemann1966} and the Procrustes distance of shape theory \citep{Kendall1984}. For full Gaussian laws no closed-form Fisher--Rao distance is known \citep{PineleStrapassonCosta2020}; a declared invariant divergence such as relative entropy then gives a computable edge cost with the same local proofs and the same infinitesimal normal equation, at the price of a changed model."""),
    (r"""\mathbb G&=\sum_v m_v\left(\mathsf G_{p_v}
                      +\sum_i g^F_{\beta_{iv}}+\sum_i g^F_{\gamma_{iv}}\right),\\
 \mathcal F_*&=\sum_v m_v V_v+\lambda\mathcal C_*,\qquad\lambda\geq0.""",
     r"""\mathbb G&=\sum_v m_v\left(\mathsf G_{p_v}
                      +\sum_i g^{\mathrm{row}}_{\beta_{iv}}+\sum_i g^{\mathrm{row}}_{\gamma_{iv}}\right),\\
 \mathcal F_*&=\sum_v m_v\mathcal E_v+\lambda\mathcal C_*,\qquad\lambda\geq0."""),
    (r" \dot p_v=-\operatorname{grad}_{\mathsf G}V_v", r" \dot p_v=-\operatorname{grad}_{\mathsf G}\mathcal E_v"),
    (r"The masses cancel from the local force, so zero coupling recovers the chosen unit-mobility reciprocal flow.",
     r"The masses cancel from the local force, so zero coupling recovers the chosen unit-mobility restricted-family flow of the reciprocal potential."),
    (r"With $C^3$ local potentials, the reduced field is $C^1$ in ordinary law and interior-row coordinates, so the integral equation is a contraction on a sufficiently short interval. This gives a unique local solution and its maximal continuation in the regular domain. Energy decrease does not prevent escape from that domain.",
     r"With $C^2$ local potentials, the reduced field is $C^1$ in ordinary law and interior-row coordinates, so the integral equation is a contraction on a sufficiently short interval. This gives a unique local solution; carried together with its transports, it has a maximal continuation on the regular critical set. Energy decrease does not prevent escape from that set, and the reduced energy is the minimized cost only while the branch stays globally minimizing."),
    (r"The minimizer switches from $I$ to $-I$, with every rotation minimizing at zero. The reduced energy is not differentiable there. Conversely, for compact $G$ acting freely at $p_0$, agreement of the contextual endpoints at $(p_0,p_0)$ has a unique minimizer and positive Hessian $\mathsf G(J\cdot,J\cdot)$; regular alignments exist near that agreement.",
     r"The minimizer switches from $I$ to $-I$, with every rotation minimizing at zero. The reduced energy is not differentiable there, and its descent directions depend on which minimizing rotation is used, so a gradient flow of the minimized energy is not unique from that point. Conversely, for a free action at $p_0$ that is compact, or proper on a complete family, agreement of the contextual endpoints at $(p_0,p_0)$ has a unique minimizer and positive Hessian $\mathsf G(J\cdot,J\cdot)$; globally minimizing regular alignments exist near that agreement. This covers $\GL(K)$ acting on Gaussian laws away from consensus. At consensus it fails for $K\geq3$: identical Gaussian laws with nonzero mean share a stabilizer of dimension $(K-1)(K-2)/2$, so the transport Hessian is singular there."),
    # continuum part
    (r"Supply the base $\mathcal C=\mathbb T^d=\R^d/(2\pi\mathbb Z)^d$",
     r"Write $V(x,p,b)$ for the local potential $\mathcal E$ at context $x$, with rows in log-odds coordinates $b$. Supply the base $\mathcal C=\mathbb T^d=\R^d/(2\pi\mathbb Z)^d$"),
    (r"Thus contextual energy sees shape variation. Pure common frame variation costs zero, and a transitive group action leaves no shape directions to regularize.",
     r"Thus contextual energy sees shape variation. Pure common frame variation costs zero, and a transitive group action leaves no shape directions to regularize. This is the elimination of a nondynamical gauge field in a gauged sigma model, the mechanism of the $CP^{N-1}$ model \citep{DAddaLuscherDiVecchia1978} and of hidden local symmetry \citep{BandoKugoYamawaki1988}; the Fisher target metric, the nonbasic reference force, and the frame and row equations are what the present construction adds."),
    (r"Its Fisher matrix is $B_{jk}=\beta_j\delta_{jk}-\beta_j\beta_k$; collect these positive blocks for all rows. The field metric is the integral of \cref{comp:ctx:adapted-metric} and $db^TBdb$.",
     r"Its Fisher matrix is $\mathsf B_{jk}=\beta_j\delta_{jk}-\beta_j\beta_k$; collect these positive blocks for all rows. The field metric is the integral of \cref{comp:ctx:adapted-metric} and $db^T\mathsf Bdb$."),
    (r"k(r_t+Cu_t)&=-V_r,\qquad B b_t=-V_b.", r"k(r_t+Cu_t)&=-V_r,\qquad \mathsf B b_t=-V_b."),
    (r" b_t=-B(b)^{-1}V_b.", r" b_t=-\mathsf B(b)^{-1}V_b."),
    (r"A smooth frame change fixed in evolution time obeys $A'=gAg^{-1}-(dg)g^{-1}$ and carries the equations covariantly.",
     r"A smooth frame change fixed in evolution time obeys $A'=\operatorname{Ad}_gA-(dg)g^{-1}$ and carries the equations covariantly."),
    (r"The complete law equation is degenerate in frame directions: its frame equation contains $u_t$ rather than an independent frame Laplacian.",
     r"The complete law equation is degenerate in frame directions: its frame equation contains $u_t$ rather than an independent frame Laplacian. This is forced by gauge invariance, not by the particular energy: any contextual energy invariant under context-dependent frame changes has a horizontal Fisher gradient, so after elimination the frame velocity is the pointwise one of the local potential."),
    (r"       -T\left(b_j-\log\frac{\pi_j}{\pi_l}\right).", r"       -T_a\left(b_j-\log\frac{\pi_j}{\pi_l}\right)."),
    (r"For each source row, \cref{comp:ctx:coupled-pde} reads", r"For each source row of sector $a\in\{q,s\}$, \cref{comp:ctx:coupled-pde} reads"),
    (r"Instantaneous Gibbs rows $\beta_j^*\propto\pi_j e^{-D_j/T}$ instead replace their energy by $-T\log\sum_j\pi_j e^{-D_j/T}$, producing a different law-only flow.",
     r"Instantaneous Gibbs rows $\beta_j^*\propto\pi_j e^{-D_j/T_a}$ instead replace their energy by $-T_a\log\sum_j\pi_j e^{-D_j/T_a}$, producing a different law-only flow."),
    (r"Assume the periodic geometry above, the declared regular finite-dimensional product family, a free proper action, and one adapted chart with smooth metric coefficients and finite smooth $V(x,u,r,b)$ on a neighborhood of the initial image. Fix $\lambda>0$, an integer $s>d/2+2$, and $m=s-1$.",
     r"Assume the periodic geometry above, the declared regular finite-dimensional product family, a free proper action, and one adapted chart with smooth metric coefficients and finite smooth $V(x,u,r,b)$ on that open chart. Fix $\lambda>0$, an integer $s>d/2+1$, and $m=s-1$."),
    (r"It depends locally Lipschitz-continuously on the initial data in these norms on a common short interval, and continues while",
     r"Its existence time depends only on $\lambda$, bounds for the data norms, and a compact set containing the image. It depends locally Lipschitz-continuously on the initial data in these norms on a common short interval, and continues while"),
    (r"Fourier expansion of the forced heat equation gives, for $T\leq1$,", r"Fourier expansion of the forced heat equation gives, for a time horizon $T\leq1$,"),
    (r"Setting $\lambda=0$ instead gives a locally Lipschitz ODE in $H^m$ for all coordinates.",
     r"Every estimate uses only $m>d/2$, through the algebra property, the embedding into continuous functions, and smooth composition \citep{Taylor2011}. Setting $\lambda=0$ instead gives a locally Lipschitz ODE in $H^m$ for all coordinates."),
    (r""" V(P)=2+\frac12\langle P,PH\rangle_F,\qquad
 H=\begin{pmatrix}""", r""" V(P)=2+\frac12\langle P,P\mathsf H\rangle_F,\qquad
 \mathsf H=\begin{pmatrix}"""),
    (r" P_t=-PH+\lambda(\partial_x+\alpha_*J)^2P.", r" P_t=-P\mathsf H+\lambda(\partial_x+\alpha_*J)^2P."),
    (r"Put $\tau=(\sqrt5-1)/2$, $\kappa=5-\tau$, $v=(1,\tau,-1,-\tau)^T$, and $P_0=e_1v^T$. Let $R_\varphi=\exp(\varphi J)$. Since $Hv=\kappa v$, the field $P(t,x)=R_{\varphi(x)}a(t,x)P_0$, with smooth periodic phase $|\varphi|<\pi/4$ and positive amplitude $a$, obeys",
     r"Put $\varsigma=(\sqrt5-1)/2$, $\kappa=5-\varsigma$, $v=(1,\varsigma,-1,-\varsigma)^T$, and $E_0=e_1v^T$. Let $R_\varphi=\exp(\varphi J)$. Since $\mathsf Hv=\kappa v$, the field $P(t,x)=R_{\varphi(x)}a(t,x)E_0$, with any smooth periodic phase $\varphi$ and positive amplitude $a$, obeys"),
    (r"The remaining amplitude is gauge invariant: $a=\|P\|_F/\|P_0\|_F$.", r"The remaining amplitude is gauge invariant: $a=\|P\|_F/\|E_0\|_F$."),
    (r"their local velocities differ by $-3\lambda a_1R_{\varphi(0)}P_0$.", r"their local velocities differ by $-3\lambda a_1R_{\varphi(0)}E_0$."),
    (r"The statistical shape approaches the singular zero-mean orbit only at infinite time in this explicit solution.",
     r"The statistical shape approaches the singular zero-mean orbit only at infinite time in this explicit solution. With a fixed reference that breaks the rotation symmetry, the same family can instead reach the zero-mean orbit in finite time with bounded norms and decreasing energy; the law flow passes through it smoothly while the quotient description breaks down."),
    (r"This example uses a restricted Gaussian location family and singleton rows; it supplies no invariant Gaussian family for the full density flow and no test of nontrivial row dynamics.",
     r"This example uses a restricted Gaussian location family and singleton rows; it supplies no invariant Gaussian family for the full density flow and no test of nontrivial row dynamics. It is the linear radial sector of the system: the quotient metric is flat along its ray, the references are rotation invariant so the nonbasic force vanishes, the phase does not move, and the connection is pure gauge with trivial holonomy around the circle. It therefore tests none of the nonlinear, frame, or nonbasic parts of the theorem."),
]
for a, b in R:
    n = s.count(a)
    if n != 1:
        sys.exit(f"count {n} for: {a[:80]}")
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
