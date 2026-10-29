p = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/companion/c18_link_dynamics.tex"
s = open(p, encoding="utf-8").read()
reps = [
(r"$T_e:\R^K\to\R^{r_e}$", r"$T_e:\R^K\to\R^{n_e}$"),
(r"Thus $S_i=J_{ii}^{-1}$ at its covariance optimum, and the factor means equal the exact posterior mean.",
 r"Thus $S_i=J_{ii}^{-1}$ at its covariance optimum, and the factor means equal the exact posterior mean. This is a feature of the Gaussian target; for non-Gaussian posteriors the factor means generally differ from the exact means."),
(r"More generally, for an exponential-family source $q_\eta$ with statistic $t$, one has $\partial_\eta\KL(p\Vert q_\eta)=\E_{q_\eta}t-\E_pt$ under regularity and finite moments. Source stationarity matches those expected statistics.",
 r"More generally, let an exponential-family source $q_\eta$ with statistic $t$ be compared through a fixed kernel $\mathcal K$. Under regularity and finite moments, $\partial_\eta\KL(p\Vert q_\eta\mathcal K)=\E_{q_\eta}t-\E_{\tilde p_\eta}t$, where $\tilde p_\eta=\int p(dh)q_\eta(\cdot\mid h)$ is the source's own posterior reconstruction of the compared message law. Source stationarity is therefore the fixed-point condition of an EM step; only for an identity kernel is it the plain moment match $\E_{q_\eta}t=\E_pt$."),
(r"Freeze the means, the receiver variance $v_k$, the noise $R>0$, and a deterministic dictionary. Let the sender have only its self term of precision $p_j$ and one outgoing edge of unit weight. Its stationary variance satisfies",
 r"Freeze the means, the receiver variance $v_k$, the noise $R>0$, and the dictionary variance $V_e\geq0$. Let the sender have only its self term of precision $p_j$ and one outgoing edge of unit weight. With $a=v_k+R+d^2+c_k^2V_e$, its stationary variance satisfies"),
(r"""f(v,d^2)=p_j-v^{-1}+(v+R)^{-1}
                  -(v_k+R+d^2)(v+R)^{-2}=0.""",
 r"""f(v,a)=p_j-v^{-1}+(v+R)^{-1}
                  -a(v+R)^{-2}=0."""),
(r"At a nondegenerate minimum, $\partial_v f>0$, whereas $\partial_{d^2}f=-(v+R)^{-2}<0$. Implicit differentiation gives $dv/d(d^2)>0$.",
 r"For every $v>0$, $\partial_vf=[R^3+3R^2v+4Rv^2+2(a-R)v^2]/[v^2(v+R)^3]>0$, and $f$ runs from $-\infty$ to $p_j>0$, so there is exactly one stationary variance and it is the global minimizer. Since $\partial_af=-(v+R)^{-2}<0$, implicit differentiation gives $dv/da>0$."),
(r"Finite-rate row relaxation and this reduced convention generally follow different paths.",
 r"Finite-rate row relaxation and this reduced convention generally follow different paths. Near the critical point below, pullback-metric reduction reproduces the leading center coefficients of full relaxation."),
(r"These conclusions require neither a unique equilibrium nor a claim of convergence to one particular point.",
 r"Because both potentials are real analytic on the interior and the Fisher metric is uniformly equivalent to the Euclidean one on the trajectory closure, the {\L}ojasiewicz inequality gives finite trajectory length, so every trajectory converges to one stationary point. The limit can depend on the initial state, since uniqueness of the equilibrium is not proved."),
(r"The negative response term accounts for the stable matter coordinates moving as the row changes. A calculation on the straight zero-mode line would omit it.",
 r"The negative response term accounts for the stable matter coordinates moving as the row changes. A calculation on the straight zero-mode line would omit it and give the positive fourth derivative $6368971071507328/655026269697361\approx9.72$, predicting a supercritical pitchfork; the response term exceeds $32T_c$ by less than $0.2$ percent and reverses the classification."),
(r"These branches lie on the high-temperature side, where the symmetric state is still stable.",
 r"These branches lie on the high-temperature side, where the symmetric state is still stable. The sign of $c$ is a property of these reference data: a float64 continuation, not interval validated, gives $c\approx+0.023$ when the dictionary reference precision is $1$ instead of $3/4$, and then the pitchfork is supercritical."),
(r"Adding the Gibbs row's pullback metric to the reduced flow would change the specified dynamics.",
 r"Adding the Gibbs row's pullback metric to the reduced flow changes the specified dynamics in a definite way: the row direction adds $1/(b_*(1-b_*))=4$ to the squared length of $p$, so the pullback-metric reduction has exactly the leading center coefficients of full relaxation. The factor $(D+4)/D$ is the kinetic cost of the row omitted by matter-metric reduction."),
(r"This is a subcritical pitchfork with destabilizing parameter $T_c-T$.",
 r"For these data this is a subcritical pitchfork with destabilizing parameter $T_c-T$."),
]
for a, b in reps:
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok", len(reps))
