# X3-PIFB: Alignment with the PIFB2 long-term program and an extension roadmap

Read in full: PIFB2 `00_preamble`, `01`, `02`, `05`, `09`, `10`, `11`, `12`, `13`, `16`, `17`, `18`, `20`, `24`; PIFB2 selectively: `05b_lie_links.tex:120-200`, `07_section.tex:68-230`, `08_section.tex` (opening, lines 96, 124, 157, 177-213 and headings), `19_section.tex:1-60,119-240`, `29_section.tex:1-60,150-260`, `36_section.tex:1-80`, `04` headings; not read: `21`, `23` beyond headings, `28`, `30`-`35`, `36a`-`36c`. In 2.0 read in full: `frontmatter.tex`, `chapters/06_outlook.tex`, `README.md`, `AUTHORING.md`, `chapters/01_foundations.tex`, `02_interactions.tex`, `02_geometry.tex`, `03_dynamics.tex`, `04_prediction.tex`, `05_inheritance.tex`, `appendices/determinant_example.tex`, `base_connection.tex`, `record_process.tex`; statements plus the proofs I needed: `06_fusion.tex:1-300,405-420`, `07_smooth_closure.tex:1-260`, `08_smooth_erasure.tex:1-15,120-140`, `09_public_compression.tex:1-60`, `10_boundary_closure.tex:1-60`, `11_conditional_formation.tex:1-460` (including the proofs of `form:full-flow` and `form:selection`), `12_open_formation.tex:1-520` (including the proof of `open:routing`); companion passages `c07_fusion.tex:9-35`, `c11_formation.tex:80-100`, `c12_open_formation.tex:133` by grep. Supplements not read. Verdict: 2.0 is a careful rebuild of PIFB2's consensus-potential track, its pullback geometry and its retained-law inheritance, but it has dropped PIFB2's foundational claim (one collective VFE of a normalized population model) without saying so in the abstract or outlook, and its formation results do not meet PIFB2's own standard of interaction-driven selection against a control: in the conditional construction the membership schedule is exactly independent of the interaction, and in the proved open-law configuration a no-peer-force control reproduces every event. Both gaps have concrete repairs inside 2.0: a registered-agreement (method of types) model makes the 2.0 potential exactly T times a collective VFE with T equal to the reciprocal message count, and the open-law limit field exhibits, numerically, interaction-necessary two-stage formation from equally spaced initial data.

Scripts (all single-threaded, each under 20 s, CPU) are in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-22-magent-theory-2.0-review\scripts\X3-PIFB\`: `s1_registered_agreement_types.py`, `s2_nogo_and_source_force.py`, `s3_spectra_priors_memory.py`, `s4_lorentz_entropy_jeffreys.py`, `s5_formation_controls.py`, `s6_gap_amplification.py`, `s6b_gap_window.py`. Numerical agreement below supports identities; it does not prove theorems, and the formation simulations are observations on the limiting field only.

## Findings

### X3-PIFB-01: The 2.0 potential severs PIFB2's generative identification, and the severance is unstated at program level
**Category:** weakness
**Severity:** high
**Location:** `frontmatter.tex:5`; `chapters/03_dynamics.tex:35-49` (`def:dyn-potential`), `:151-184` (`eq:dyn-fixed-vfe`, `eq:dyn-affine-obstruction`); `chapters/06_outlook.tex:1-16`; PIFB2 `01_section.tex` para. 4, `05_section.tex` (`eq:population_joint`, `eq:collective_vfe`, `eq:pointwise_free_energy`)
**Evidence:** PIFB2 states "We take the normalized joint as the foundation" and keeps the consensus potential as a separate worked object ("It is not asserted to be the evidence bound"). 2.0 inverts the priority: `def:dyn-potential` is "a constitutive choice at the level of agent laws", the VFE appears only in Section 3.4 as "a second construction", the abstract's first dynamical sentence is "A declared interaction potential generates coupled Fisher dynamics", and the outlook never mentions the generative question. 2.0's only obstruction (`eq:dyn-affine-obstruction`) excludes expected fixed sample energies $\E_{q\otimes p}e$. Three sharper statements hold.

(a) Finite alphabets, every recognition rule. Let $\mathcal E$ have fixed positive rows and one attended edge $j\to i$, $i\ne j$, with identity channel. On a finite space every finite value of $\KL(Q\Vert P)$ obeys $\KL(Q\Vert P)=\sum Q\log Q-\sum Q\log P\le\max_{\operatorname{supp}P}(-\log P)$. The potential is finite on the open interior and unbounded there, since $\mathcal E\ge\beta_{ij}\KL(q_i\Vert q_j)\to\infty$ along $q_i\to\delta_a$, $q_j\to\delta_b$, $a\ne b$ (s2(b): $\mathcal E=5.30,10.48,15.19,19.81$ at $(x_1,x_2)=(t,-t)$, $t=0.9,\dots,0.9999$). Hence no finite normalized population model (any finite latent and record variables, records absorbed into the posterior) and no rule $x\mapsto Q(x)$ with agent marginals $q_k,s_k$ give $\mathcal E=a\KL(Q(x)\Vert P)+b$ with $a>0$ on the interior.

(b) Product recognition, general spaces. A mean-field ELBO $\KL(\bigotimes_k\zeta_k\Vert P)$ is a sum of single-factor entropies plus a multi-affine function, so its mixed derivative $\partial_{q_i}\partial^2_{q_j}$ vanishes identically. For $\mathcal E$ at fixed rows and channels the same derivative in directions $w$ (on $q_i$) and $v$ (on $q_j$) is $\beta_{ij}\int w\,(vK_{ij})^2/(q_jK_{ij})^2$. s2(a): the symbolic ELBO derivative is $0$ for an arbitrary positive table $P$ on $\{\pm1\}^2$; for the mutual binary peer potential it is $-2x_2/((1-x_2)^2(1+x_2)^2)$, value $-0.7246$ at $(0.1,0.3)$. No affine identity holds on any open set where it is nonzero.

(c) Where the identification survives. On the slice of unit-variance Gaussians, $\mathcal E$ equals, up to a constant, the mean-field VFE of the PIFB2 linear-Gaussian record model (`eq:selective_record`, $A_i=B_{ij}=I$) with one agreement record of precision $\beta_{ij}$ per directed edge: for two agents with references $a,b$ the 2.0 minimizer has mean gap $(a-b)/5$, the exact posterior mean gap with record precision 2 (s3(a); the receiver-only fixed point gives $(a-b)/3$). Off that slice, s2(d) gives exactly (sympy residual $0$)
$\KL(\mathcal N(m_i,v_i)\Vert\mathcal N(m_j,v_j))=[\text{PIFB2 record cost with }\tau\Gamma:=v_j]-H(q_i)-\tfrac12$:
the loss is located in a law-dependent record noise (the source covariance) and in the receiver entropy, which 2.0 counts $1+\sum_j\beta_{ij}$ times.
**Fix:** Add (a) and (b) as a proposition in Section 3.4 in place of the informal last paragraph; add one abstract sentence and one outlook paragraph stating that the potential is not the VFE of a normalized sample-level population model; adopt X3-PIFB-02 as the route back.
**Falsifies:** A finite normalized model and recognition rule reproducing $\mathcal E$ on the interior (excluded by the boundedness argument), or an affine ELBO identity on an open set where the mixed derivative is nonzero.

### X3-PIFB-02: Registered-agreement theorem: the 2.0 potential is T times an exact collective VFE, with T = 1/n
**Category:** extension
**Severity:** critical
**Location:** `chapters/03_dynamics.tex:35-89` (`def:dyn-potential`, `prop:dyn-gibbs`, `eq:dyn-reduced`), `:162` (row costs "need additional hypotheses"); PIFB2 `36_section.tex` (`thm:belief_configuration_gibbs_lift`), `07_section.tex:68-92` (a circular assignment of priors "needs its own normalized joint")
**Evidence:** Construction and proof sketch (finite alphabets). Fix $n\ge1$ and $T_q=T_s=T:=1/n$. Let $\mathcal X_n$ be a finite set of configurations $x=(q_i,s_i)_i$ whose targets $q_i,s_i,q_i\otimes s_i$ are $n$-types (for example all marginals $m$-types with $n=m^2$) and whose message laws $\nu^q_{ij}(x)=q_j\mathcal K^q_{ij}[x]$, $\nu^s_{ij}(x)$ and references $R_i$ are positive; law-dependent channels are allowed. Model $\mathbb P_n$: $X$ uniform on $\mathcal X_n$; independent labels $J^q_i\sim\pi^q_i$, $J^s_i\sim\pi^s_i$; conditionally independent message batches $Y^0_i\sim R_i^{\otimes n}$, $Y^q_i\sim\nu^q_{iJ^q_i}(X)^{\otimes n}$, $Y^s_i\sim\nu^s_{iJ^s_i}(X)^{\otimes n}$; registered binary readouts $o^0_i=\mathbf 1\{\operatorname{type}(Y^0_i)=q_i\otimes s_i\}$, $o^q_i=\mathbf 1\{\operatorname{type}(Y^q_i)=q_i\}$, $o^s_i=\mathbf 1\{\operatorname{type}(Y^s_i)=s_i\}$. Observe $o=\mathbf 1$ (every agreement registered).

(i) $\mathbb P_n(o^q_i=1\mid x,J^q_i=j)=|T_{q_i}|\exp\{-n[H(q_i)+D^q_{ij}(x)]\}$ (Cover and Thomas 2006, Thm 11.1.2).
(ii) $\mathbb P_n(J^q_i=j\mid x,o)=\beta^{q*}_{ij}(x)$ of `eq:dyn-gibbs`: the factor $|T_{q_i}|e^{-nH(q_i)}$ is common to the row (the same cancellation as `open:cost`).
(iii) For recognition $\delta_x\otimes\prod_i\beta_i\otimes\prod_i\gamma_i$, with $\mathcal F_V=\KL(Q\Vert\mathbb P_n(\cdot\mid o))-\log\mathbb P_n(o)$,
$T\mathcal F_V=\mathcal E(x,\beta,\gamma)+C_n(x)+T\log|\mathcal X_n|$, where $C_n$ sums $c_n(p)=H(p)-\tfrac1n\log|T_p|\in[0,|\operatorname{supp}p|\log(n+1)/n]$ over the registered targets (Cover and Thomas, Thm 11.1.3).
(iv) Minimizing rows by `prop:dyn-gibbs`: $\mathcal E_{\rm red}(x)=-T\log\mathbb P_n(o\mid x)-C_n(x)$, and $\mathcal E_{\rm red}(x)+C_n(x)+T\log|\mathcal X_n|\ge-T\log\mathbb P_n(o)$, the collective evidence bound.
(v) General standard Borel spaces: with the registration "empirical law in a $\tau$-neighborhood $B_\epsilon$ of the target", $\lim_{\epsilon\to0}\lim_{n\to\infty}-\tfrac1n\log\mathbb P(o^q_i=1\mid x,j)=D^q_{ij}(x)$ by Sanov's theorem in the $\tau$-topology (Csiszar 1984, Ann. Probab. 12:768-793; Dembo and Zeitouni 1998, Thm 6.2.10; both cited from memory, not re-opened). At finite $n$ the exact Bernoulli registration $\mathbb P(o^q_i=1\mid x,j)=e^{-D^q_{ij}(x)/T}\in(0,1]$ is a normalized record kernel for every $x$ because $D\ge0$ (the optimality-variable device of Levine 2018, arXiv:1805.00909).

s1 residuals at $n=4,\dots,7$ on ternary alphabets: (i) relative $2.6\times10^{-15}$; (ii) $3.3\times10^{-16}$; (iii)/(iv) $4.4\times10^{-16}$, with $c_n$ inside its bounds in every trial; unreduced row objective versus $T\times$ label VFE $4.2\times10^{-16}$.

Consequences: the row temperature is the reciprocal number of exchanged messages; Gibbs rows are exact posteriors of which source was heard (this answers the "additional hypotheses" sentence at `03_dynamics.tex:162`); the incoming force is the likelihood gradient in a source's emission law; the 2.0 Fisher flow is a continuum interpolation of MAP descent; PIFB2's Gibbs lift becomes a posterior of registered records with configuration temperature equal to row temperature; and the instantaneous circular assignment of priors that PIFB2 Section 7 says needs its own normalized joint receives one. The identification is at the configuration level: laws are emission parameters and test targets, not recognition marginals of a posterior over contents (X3-PIFB-01 shows the latter is impossible).
**Fix:** State (i)-(iv) as a theorem in Chapter 3 for finite alphabets and (v) as a remark. Restrictions to state: equal row temperatures and a common $n$ for the reference events (otherwise a weighted composite likelihood, i.e. a power posterior in the sense of Bissiri, Holmes and Walker 2016, JRSS-B 78:1103); a uniform configuration prior on the lattice (otherwise $-T\log\rho(x)$ is added); exactness only on the type lattice. Prerequisites: the method of types; from 2.0 only `prop:dyn-gibbs`. Difficulty: low for the finite theorem; medium for the general-space double limit.
**Falsifies:** Failure of the type-class identity (standard and recomputed to machine precision), or a 2.0 potential term that is not a KL between a target law and a law that emits messages (none exists in `def:dyn-potential`).

### X3-PIFB-03: The conditional formation schedule does not depend on the interaction
**Category:** weakness
**Severity:** high
**Location:** `chapters/11_conditional_formation.tex:128-170` (proof of `form:full-flow`, `form:density-force` at :142), `:279-330` (`form:selector` at :283, `form:selection` at :292); `frontmatter.tex:7`; `README.md` ("an interacting four-agent Wishart construction")
**Evidence:** From `form:density-force`, $h_i(X)=2\log q_i(X)-2c_0\log\det X+\Tr[(I_2+C_q)X]$ with $c_0=\alpha_0-3/2$. Writing $\log q_i=(\alpha_i-\tfrac32)\log\det X-\Tr(\mathsf PX)+\text{const}$ and matching the $\log\det X$ coefficient in $\partial_t\log q_i=-(h_i-\E h_i)$ gives $\dot\alpha_i=-2(\alpha_i-\alpha_0)$; $\kappa$ and $C_q$ enter only the coefficient of $X$, hence only $\dot{\mathsf P}$. The selector reads only $b_I=A_I/n_I-\alpha_0$. Every conclusion of `form:selection` (event order, $t_L,t_R,t_2$, coexistence, absence of cross joins, subthreshold persistence, loss of model-shape distinctions) therefore holds verbatim at $\kappa=0$, where $\mathcal K^q_{ij}=W_2(\alpha_0,I_2)$ equals the reference and $\mathcal E_q=2\sum_i\KL(q_i\Vert W_2(\alpha_0,I_2))$ is a sum of single-agent terms. The proof itself says "Every initial belief-shape difference is multiplied by $e^{-2t}$": membership is single-linkage threshold clustering of the initial shapes under a contraction common to all agents (s5(a) recomputes the event times from the gaps alone). PIFB2 `13_section.tex` judges selection against a control ("Selection against a symmetric frozen control therefore tests a stronger mechanism than observing a coherent cluster"); this construction does not pass the corresponding no-interaction control. I did not check whether `form:responses` and `form:geometry` also survive at $\kappa=0$ (precision then obeys $\dot{\mathsf P}=-2\mathsf P+2I_2$).
**Fix:** State in `form:selection` and in the abstract that the membership schedule is independent of $\kappa$ and of every peer datum, and describe Chapter 11 as inheritance through joins with a kinematic selector. For interaction-driven selection, couple the selector-read coordinate to the population, for example a channel shape depending on $M_q$ so that $\dot\alpha_i$ acquires a $\kappa$-dependent term, and prove the events absent at $\kappa=0$.
**Falsifies:** A $\kappa$-dependent term in the shape equation (it would contradict `form:density-force`) or a selector that reads $\kappa$-dependent data.

### X3-PIFB-04: The proved open-law configuration is reproduced by a no-peer-force control; interaction-necessary formation exists numerically
**Category:** weakness
**Severity:** medium
**Location:** `chapters/12_open_formation.tex:125-251` (`open:routing`, `open:limit` at :137), `:414-430` (`open:formation`); `frontmatter.tex:7` ("Native preferential pairs form")
**Evidence:** s5(b) integrates the limiting field `open:limit` (RK4, $dt=10^{-3}$) from the proof's configuration ($A=4096$, $b(0)=2$, third pair at $R=10^7$) and applies `open:selector` ($\varepsilon=1/10$, $\delta=1/20$). Full field: pairs $\{0,1\},\{2,3\},\{4,5\}$ at $t=0.589$, core parents join at $t=8.214$. Control with peer forces deleted and rows only read out, $u(t)=u(0)e^{-t}$: the same pairs at $t=0.647$ and the same join at $t=8.271$. The proof's own bounds explain this: within-pair weights $e^{-2b^2}$ grow because $b$ shrinks, and the three-scale hierarchy ($b(0)=2$, $A\ge4096$, $R\gg A$) is in the initial data. The control is a counterfactual diagnostic, not a 2.0 dynamics. Numerical observation (s6, s6b): for equally spaced reflection-symmetric data $u=(-9,-3(1+\epsilon),3(1+\epsilon),9)$ the full field produces both stages (pairs, then their union) for $\epsilon\in[-0.02,0.05]$ while the control produces none; at $\epsilon=-0.02$ the middle pair is initially closer, so the interaction reverses the nearest-neighbor ordering (the gap ratio $g_{\rm out}/g_{\rm in}$ moves from about $1.03$ below $1$; it is constant under the control). These runs use the limit field, no exterior, and reflection-symmetric data only; they are not a theorem.
**Fix:** Add a control statement to `open:routing`. Then prove an interaction-necessary version: reduce `open:limit` on reflection-symmetric four-agent data to a two-dimensional ODE in $(x_1,x_3)$, prove the strict threshold windows there, add a distant exterior pair as in the existing proof, perturb to an open set, and transfer by `open:open-laws`. Difficulty: medium.
**Falsifies:** Disappearance of the events under step refinement, or a proof that the gap-ratio dynamics coincide with the control.

### X3-PIFB-05: The incoming force is tied empirical-Bayes prior fitting; source precision moves opposite to PIFB2's normalized model
**Category:** extension
**Severity:** high
**Location:** `chapters/03_dynamics.tex:102-117` (`eq:dyn-channel-differential`, `eq:dyn-incoming`); `chapters/12_open_formation.tex:50-67` (`open:force`); PIFB2 `12_section.tex` (`eq:aggregate_prior_identity`), `08_section.tex` (stiffness)
**Evidence:** For the receivers $i$ of source $j$ with a common channel $K_j$, $W_j=\sum_i\beta_{ij}$ and $\bar q_{\to j}=W_j^{-1}\sum_i\beta_{ij}q_i$, the compensation identity
$\sum_i\beta_{ij}\KL(q_i\Vert q_jK_j)=\sum_i\beta_{ij}\KL(q_i\Vert\bar q_{\to j})+W_j\KL(\bar q_{\to j}\Vert q_jK_j)$
holds (s2(c): residual $1.1\times10^{-16}$; ambient force residual $4.4\times10^{-16}$). Its first term is free of $q_j$. So at fixed rows the complete $q_j$ force is the gradient of $(1+B_j)\KL(q_j\Vert\Gamma_j)+W_j\KL(\bar q_{\to j}\Vert q_jK_j)$, with $B_j=\sum_k\beta_{jk}$, $\Gamma_j\propto(\tilde g_j\prod_k(\nu^q_{jk})^{\beta_{jk}})^{1/(1+B_j)}$ and $\log\tilde g_j=\int\log G_j(m,\cdot)s_j(dm)$. The receiver part is a log-linear pool (Genest and Zidek 1986, Stat. Sci. 1:114); the source part is the mass-covering M-step of fitting one shared prior to $j$'s receivers, whose unrestricted minimizer for $K_j=\mathrm{id}$ is $\bar q_{\to j}$, exactly PIFB2's optimal prior `eq:aggregate_prior_identity`. This reproduces `open:force`. In the Gaussian covariance sector (s2(d)) the 2.0 source force on $v_j$ is $(v_j-v_i-(m_i-m_j)^2)/(2v_j^2)$, zero at $v_j=v_i+(m_i-m_j)^2$, so a confident source is widened toward its receivers' spread; in PIFB2's normalized record model the same derivative is $1/(2s)>0$, so outgoing records always narrow the source.
**Fix:** State the decomposition as a proposition in Chapter 3 (receiver log-linear pool, source arithmetic pool); it names the participatory loop of PIFB2 Section 12 inside 2.0. Next theorem: recompute PIFB2 Section 8's covariance stiffness for the 2.0 potential including the source block and prove at a symmetric consensus point that the source-covariance block has the sign structure above. Difficulty: low.
**Falsifies:** The identities are exact; the interpretation fails for receiver-specific channels, where the identity applies only within each channel class.

### X3-PIFB-06: Friction without inertia survives inheritance; the frozen-template dynamics can oscillate
**Category:** extension
**Severity:** high
**Location:** `chapters/03_dynamics.tex:124-149` (`thm:dyn-finite-flow`); `chapters/05_inheritance.tex:73-86` (`inh:vector-square`); PIFB2 `08_section.tex:7`, `29_section.tex:152-184`, `07_section.tex:127-148` (`prop:exact_posterior_fisher_rates`), `24_section.tex` (Physical inertia)
**Evidence:** Proposition sketch. (i) At an interior critical point $x^*$ of the 2.0 Fisher gradient flow (rows dynamic or reduced), the linearization $L=-g^{-1}\operatorname{Hess}\mathcal E$ is $g$-self-adjoint, hence has real spectrum, nonpositive at a local minimum. (ii) If a $C^1$ submersion $R$ intertwines the flow near $x^*$ (`inh:vector-square`), differentiating $DR_xB(x)=B^c(R(x))$ at $x^*$ with $B(x^*)=0$ gives $DR\,L=L^c\,DR$; $\ker DR$ is $L$-invariant and $L^c$ is conjugate to the induced quotient map, so $\operatorname{spec}L^c\subseteq\operatorname{spec}L\subset\mathbb R$. No exactly inherited parent has an underdamped linear mode at an inherited equilibrium. (iii) For a $g$-orthogonal split $y=(a,b)$ of the linearized flow $\dot y=-Sy$, exact elimination gives $\dot a=-S_{aa}a-S_{ab}e^{-tS_{bb}}b_0+\int_0^tK(t-u)a(u)du$ with $K(t)=\sum_ke^{-\lambda_kt}(S_{ab}u_k)(S_{ab}u_k)^\top\succeq0$, a completely monotone memory (Bernstein's theorem; Widder 1941). (iv) The receiver-only field, the continuous limit of PIFB2's frozen-template episodes, is not a gradient field. s3(b), three 1-D Gaussian agents (mean and variance) on a directed 3-cycle: 2.0 potential spectrum $\{-4.669,-4.562,-3.741,-3.612,-1.073,-1.003\}$, maximum imaginary part $0$; receiver-only spectrum $\{-2.5\pm0.866i\ (\text{twice}),-1.0,-1.0\}$. s3(d): minimum eigenvalues of $K(t)$ are $4.08,1.05,0.16,0.005$ at $t=0,0.3,1,3$; memory-equation residual $3.3\times10^{-6}$ (trapezoid rule, $h=10^{-3}$). The 2.0 spectrum is spread, so PIFB2's `prop:exact_posterior_fisher_rates` (linearization $-I$) does not constrain law-level potentials.
**Fix:** Add (i)-(iii) to Chapter 5 as "Exact inheritance of a Fisher gradient flow creates no underdamped linear mode". For PIFB2's inertia target, name the admissible mechanisms: nonreciprocal (receiver-only or frozen-template) fields, explicit driving, or a declared second-order kinetics. Difficulty: low.
**Falsifies:** An inherited parent equilibrium with complex linearization eigenvalues under an exact intertwining of a Fisher gradient flow.

### X3-PIFB-07: Learned source priors are the 2.0 potential's own Fisher flow at rate T, and identity self channels drive isolation
**Category:** extension
**Severity:** high
**Location:** `chapters/03_dynamics.tex:47,89,147`; `chapters/12_open_formation.tex:12,38-46`; PIFB2 `13_section.tex` (`eq:formation_source_learning`), `24_section.tex` (Emergent model variables)
**Evidence:** Treat $\pi_i$ as a coordinate of `def:dyn-potential` with unit Fisher mobility. Then $\partial\mathcal E/\partial\pi_{ij}=-T_q\beta_{ij}/\pi_{ij}$ and $\dot\pi_i=T_q(\beta_i-\pi_i)$ (s4(c), symbolic residual $0$): PIFB2's learning law with $\epsilon_L=T_q$ and the recognition row replaced by the current row. With Gibbs rows, $\dot\pi_{ij}=(T_q/Z_i)\pi_{ij}(e^{-D_{ij}/T_q}-Z_i)$, replicator dynamics with fitness $e^{-D/T}$ (Hofbauer and Sigmund 1998). This is an intrinsic slow sector at small $T_q$ with no inserted multiplier, which Chapter 3 (line 147) requires of any slow sector. But if $D_{ii}\equiv0$ (identity self channel in the source set), $\dot\pi_{ii}=(T_q/Z_i)\pi_{ii}(1-Z_i)\ge0$, with equality iff $D_{ij}=0$ on $\operatorname{supp}\pi_i$, and $\inf_{\pi,\beta}\mathcal E=\mathcal E_0+\sum_i\min_{j\in\mathcal N_i}D^q_{ij}+(\text{model sector})=\mathcal E_0$: isolation. s3(c), two binary agents with references $\pm0.6$ and $T=0.5$: the minimum over time of $\dot\pi_{\rm self}$ is $5.5\times10^{-15}\ge0$, $\pi_{\rm self}\to1$, laws $\to$ references. The Chapter 12 rows include self, and the angular self term cancels from the Gibbs normalization, so the relative self cost is zero there too.
**Fix:** If PIFB2's mechanism is imported, exclude self from the learnable rows (PIFB2's four-cycle does) or hold the self prior fixed. Next theorem: for `open:limit` without self and with priors learned at rate $T=h^2$, prove that from almost every initial condition near the symmetric four-cycle the priors converge to a perfect matching (the law-level analog of PIFB2's selection theorem, now in the joint limit PIFB2 avoids). Difficulty: high.
**Falsifies:** For the proposed theorem, a simulation of the joint flow on the symmetric four-cycle that converges to a non-matching pattern or to uniform priors from an open set.

### X3-PIFB-08: No parent-to-child feedback; a construction inside 2.0
**Category:** extension
**Severity:** medium
**Location:** `chapters/05_inheritance.tex:641-646`; `chapters/04_prediction.tex:56-73` (`prop:conditional-lift`); PIFB2 `10_section.tex` (`eq:cavity_predictive_prior`, `eq:cross_scale_shadow`), `12_section.tex`
**Evidence:** No 2.0 parent acts as a source for its children (grep for top-down, downward and parent priors finds none). Sketch: a parent law $\zeta_I=\Psi_I(x)$ pushed through a downward channel is a law-dependent source in the sense of `eq:dyn-peer-costs`, so $\lambda\sum_{i\in I}\KL(q_i\Vert\zeta_I\mathcal K_{iI})$ is admissible without a new primitive. Tautology lemma: if $\zeta_I=C_\#(\bigotimes_{i\in I}\zeta_i)$ and $\mathcal K_{iI}$ is the regular conditional law of child $i$ given $C$ under $\bigotimes\zeta_i$, then $\zeta_I\mathcal K_{iI}=\zeta_i$ (marginal of a disintegration), so exact reverse conditionals exert no force. Nontrivial feedback uses a reference conditional: by `prop:conditional-lift` the block term $\lambda\KL(\bigotimes\zeta_i\Vert\int p(\cdot\mid h)\nu(dh))=\lambda\int\KL(\bigotimes\zeta_i(\cdot\mid h)\Vert p(\cdot\mid h))\nu(dh)$ charges exactly the information the parent cannot reconstruct, and its flow pushes children toward sufficiency of the retention. In Chapter 11 a Cholesky matrix-Beta split (Olkin and Rubin 1964, Ann. Math. Stat. 35:261; cited from memory, check the factorization convention) sends $W_2(A_I,\mathsf P)$ to $W_2(A_I/n_I,\mathsf P)$, giving the in-family cohesion cost $\KL(W_2(\alpha_i,\mathsf P)\Vert W_2(\bar\alpha_I,\mathsf P))$; its source-side force through $\bar\alpha_I$ has not been computed.
**Fix:** Add the lemma and the reference-conditional construction, then test inheritance of the augmented flow with `inh:vector-square`. Difficulty: medium.
**Falsifies:** The source-side term breaking the Wishart family, or the sum retention failing the vector square, which would require retaining more data.

### X3-PIFB-09: The determinant metric has a canonical scale and equals one half of Fisher minus entropy rate squared
**Category:** improvement
**Severity:** medium
**Location:** `appendices/determinant_example.tex:7-31` (`geo:det-moment`, `geo:determinant-metric`); PIFB2 `17_section.tex`, `19_section.tex:148-170`
**Evidence:** For $M\succ0$ and Hermitian $H,K$, Cayley-Hamilton for $M^{-1}H$ and polarization give $\eta_{\det}(H,K)/\det M=\tfrac12[\Tr(M^{-1}HM^{-1}K)-\Tr(M^{-1}H)\Tr(M^{-1}K)]$ (s4(a): residual $3.2\times10^{-14}$, signature $(1,3)$ in 200 random trials). Since $\det(gMg^\dagger)=|\det g|^2\det M$, the choice $\kappa_{\det}:=\det M$ satisfies the required transformation law, removing a supplied datum, and then
$\gamma=\tfrac12(h^{\mathcal{CN}}-\theta\otimes\theta)$, with $h^{\mathcal{CN}}(X,Y)=\Tr(M^{-1}e(X)M^{-1}e(Y))$ and $\theta(X)=\Tr(M^{-1}e(X))=D_X\log\det M$,
where $h^{\mathcal{CN}}$ is the Fisher metric of $\mathcal{CN}(0,M)$ pulled back by $e$ and $\theta$ is the covariant derivative of its entropy $\log\det(\pi eM)$. A direction is timelike iff the squared entropy rate exceeds the squared Fisher speed; a pure dilation $e=cM$ gives $\gamma=-c^2$. This joins PIFB2's information clock (Section 17) to its Lorentz cone (Section 19) by an exact identity; the subtracted term is gauge invariant, unlike the frame term PIFB2 Section 18 rejects.
**Fix:** Add the identity and the canonical scale as a remark after `geo:determinant-metric`; note that $\gamma$ is then invariant under constant rescaling of $M$, so physical scale remains a calibration. Difficulty: low.
**Falsifies:** The identity is algebraic. The interpretation is limited for non-Gaussian $p_i$, where $h^{\mathcal{CN}}$ is the Fisher metric of the moment projection, not of the actual laws.

### X3-PIFB-10: A stochastic 2.0 with message-count temperature brings PIFB2's metastability and fixed-temperature selection into reach
**Category:** extension
**Severity:** medium
**Location:** `chapters/03_dynamics.tex:124-149`; PIFB2 `13_section.tex`, `29_section.tex:226-250`, `36_section.tex` (`rem:gibbs_lift_interpretation`)
**Evidence:** On finite alphabets the Fisher-Rao volume of the open $(k-1)$-simplex is $\pi^{k/2}/\Gamma(k/2)$, the Dirichlet$(1/2)$ (Jeffreys) normalizer (s4(b), formula consistency), so the normalized Fisher-Rao volume is a proper, gauge-invariant configuration reference, and $e^{-\mathcal E_{\rm red}/T}\rho$ is normalizable because $\mathcal E_{\rm red}\ge0$ ($\mathcal E_0\ge0$, $Z_i\le1$). Resampling $n$ messages per step has covariance $(\operatorname{diag}p-pp^\top)/n=T\,g^{-1}$, the inverse Shahshahani (Fisher) metric (Shahshahani 1979, Mem. AMS 211; cited from memory), so the configuration temperature equals the row temperature of X3-PIFB-02. With that noise, Eyring-Kramers-type residence times (PIFB2 `eq:persistence_arrhenius`) become available for formed blocks.
**Fix:** Define the reflected Fisher-Langevin dynamics at temperature $T$ with its metric-divergence drift, prove invariance of $e^{-\mathcal E_{\rm red}/T}\rho$, then study barrier crossing between routing patterns. Difficulty: high (boundary behavior, Ito correction, metastability estimates).
**Falsifies:** A mismatch between the Wright-Fisher drift and the Riemannian Langevin drift that cannot be absorbed by the reference measure.

### X3-PIFB-11: PIFB2's parent-clock statements do not transfer to moving-representative parents
**Category:** consistency
**Severity:** low
**Location:** PIFB2 `17_section.tex` (`eq:clock_dissipation`, `eq:clock_score_contraction`); `chapters/06_outlook.tex:6`; `AUTHORING.md` (moving-canonicalization terms); companion `c12_open_formation.tex:133`
**Evidence:** For 2.0's unit-mobility flows, `thm:dyn-finite-flow` gives $g(\dot u,\dot u)=\sum_a\operatorname{Var}_{u^a}(h^a)=-\dot{\mathcal E}$, so PIFB2's clock identity holds at the fine level. The Chapter 12 parent flow is not the Fisher gradient of a parent potential for its statistical Fisher metric ("its statistical Fisher metric differs from the kinetic quotient metric"), and its retention is state-dependent, so neither PIFB2's identity $\dot\vartheta^2=-\eta\dot{\mathcal V}$ nor its fixed-channel contraction applies to the parent clock. In Chapter 11 the sum is sufficient for the common precision, so the parent precision-sector clock equals the joint fine clock: no slowdown, consistent with PIFB2's "no universal $1/N$ slowdown".
**Fix:** In PIFB2 Section 17 restrict the parent-clock statements to fixed retentions, and define an inherited parent's clock with its kinetic quotient metric when the dissipation identity is wanted.
**Falsifies:** A proof that the Chapter 12 retention is a fixed Markov morphism of the statistical families.

### X3-PIFB-12: Collective geometry is a constant-weight product metric; PIFB2's correlation geometry is unreachable
**Category:** weakness
**Severity:** medium
**Location:** `chapters/02_geometry.tex:160-198` (`geo:collective-data`, `geo:collective`); PIFB2 `20_section.tex` (`eq:collective_correlation_geometry`, `eq:labeled_collective_fisher`)
**Evidence:** 2.0 takes $\mathsf G=\sum_iw_ig^F_{p_i}$ with constant $w_i$. PIFB2 shows that the joint collective tensor sees correlation directions invisible to every marginal ($G_{Q_\rho}=(1+\rho^2)/(1-\rho^2)^2$, recomputed), and that a labeled mixture gives the weighted sum a probabilistic meaning with the extra term $\sum dw_i\otimes dw_i/w_i$. Without a population joint law, 2.0 cannot express the former.
**Fix:** Cheap step: use routing-derived weights $w_i(c)$ with the labeled metric. The group acts trivially on scalar weights, so the cost becomes $\Vert d\boldsymbol p(X)+J_{\boldsymbol p}a\Vert^2_{\mathsf G_w}+\sum_i(dw_i(X))^2/w_i$, the normal equation `geo:normal-equation` and the minimizing class are unchanged at $\mathsf G_{w(c)}$, and $h^{\min}$ gains a connection-independent positive term. Full step: use the configuration-level joint of X3-PIFB-02. Difficulty: low for the first, medium for the second.
**Falsifies:** A nontrivial group action on the weights, which would couple them to the normal equation.

### X3-PIFB-13: "Inherited mass" in the fusion companion is a friction weight; Chapter 6 is overdamped one-dimensional self-gravitating sticky dust
**Category:** exposition
**Severity:** low
**Location:** companion `c07_fusion.tex:9,33`; `chapters/06_fusion.tex` (`fus:potential`, `fus:inclusion`)
**Evidence:** `fus:potential` is $\sum_in_i[C+\mathcal U]+\lambda\sum_{i<j}n_in_j(|\Delta\mu^q|+|\Delta\mu^s|)$ with metric $\operatorname{diag}(n_i)$, so $\dot\mu_i=-\nabla\mathcal U-\lambda\sum_jn_j\sigma_{ij}$: the multiplicity cancels from each agent's own velocity, which depends only on the others' multiplicities. This is the one-dimensional Newtonian pair potential $n_in_j|x_i-x_j|$ under overdamped dynamics in a harmonic trap, an Aristotelian analog of equal passive and active mass. PIFB2 Section 29 rejects identifying precision or multiplicity with inertial mass; "mass" here invites that reading.
**Fix:** Call $n$ a multiplicity or friction weight, and add one sentence on the exact overdamped one-dimensional gravity form and its first-order character.
**Falsifies:** None; terminology.

### X3-PIFB-14: Registration budget, decisions and slow predictive memory are importable at low cost
**Category:** extension
**Severity:** low
**Location:** `appendices/record_process.tex` (`prop:bayesian-history`); `chapters/11_conditional_formation.tex:383-400` (`form:responses`); PIFB2 `17_section.tex` (`prop:binary_registration`), `12_section.tex` (`eq:participatory_policy`), `07_section.tex` (`eq:accumulation_slow_bound`)
**Evidence:** The record appendix has no information budget; PIFB2's $I(X;O_{1:n})\le n\ln2$ applies verbatim to its binary registrations. With a Dirichlet model law, the prediction-assimilation process of `prop:bayesian-history` gives PIFB2's slow predictive bound $\Vert\bar m_n-\bar m_{n-1}\Vert_{\rm TV}\le1/(A+n)$, a slow model sector derived from records. Since `form:responses` equates joint record laws under every adaptive policy, every decision functional of the record law (expected cost, record entropy) is inherited by the parent; information gain about discarded distinctions is not.
**Fix:** Add these three corollaries where the record appendix and `form:responses` are stated. Difficulty: low.
**Falsifies:** None beyond the cited identities.

### X3-PIFB-15: Latent links and episode-history inertia are absent
**Category:** extension
**Severity:** low
**Location:** `chapters/02_interactions.tex:62-71` (`def:int-generator`); PIFB2 `05_section.tex` (`eq:pifb_generator_prior`, `eq:pifb_latent_collective_vfe`), `05b_lie_links.tex:120-200`
**Evidence:** 2.0 generators are deterministic functionals of reference laws; PIFB2's latent generators with a normalized loop prior, their Fisher recognition flow and the episode-history Proca-type equation have no counterpart.
**Fix:** Within 2.0, add generators as configuration coordinates with a declared potential $\tfrac12\Vert\omega-\mathring\alpha\Vert^2_{\mathring\Sigma^{-1}}+\mathcal A_{\rm loop}(\omega)$ and Euclidean mobility; for histories, an episode action $\mathbf m\sum_n\KL(\zeta^{(n+1)}\Vert\zeta^{(n)})$ gives, to second order, PIFB2's Euclidean-sign difference equation. Under X3-PIFB-02 both become log-priors of the configuration model. Difficulty: medium; low priority until the formation and generative questions are settled.
**Falsifies:** None; absence.

## Checked without defect

| Label | What was checked | Method | Result |
|---|---|---|---|
| `prop:dyn-gibbs` (`eq:dyn-row-gap`) | row identity, uniqueness | algebra; numerically inside s1(iv) | residual $\le4.2\times10^{-16}$ |
| `eq:dyn-envelope` | reduced derivative | $-T\,d\log Z=\sum\beta^*dD$ | exact |
| `thm:dyn-finite-flow` (`eq:dyn-dissipation`) | Fisher gradient identification and dissipation | algebra; also $g(\dot u,\dot u)=-\dot{\mathcal E}$ | exact; local existence not re-proved |
| `prop:dyn-jensen` | identity | algebra under stated integrability | exact |
| `eq:dyn-affine-obstruction` | strict convexity argument | algebra | correct; weaker than X3-PIFB-01 |
| `thm:predictive-chain` | chain rule | reconstruction | correct |
| `prop:conditional-lift` | marginal preservation and loss identity | reconstruction | correct |
| `thm:predictive-distortion` | decoder decomposition | reconstruction | correct |
| `geo:joint-fisher` | proof and $\zeta_\rho$ example $1/(1-\rho^2)$ | hand computation | exact |
| `geo:retained-connection` | proof; counterexample minimizer $a=1/3$ | hand computation | exact |
| `geo:determinant-metric` | $\eta=-\det$, scaling, signature | s4(a) | residual $3.2\times10^{-14}$ |
| `inh:deterministic-descent` | fiber test, uniqueness, vector-square converse | reconstruction | correct |
| `inh:gradient-inheritance` | horizontal conformality argument | reconstruction | correct |
| `inh:causal-memory` | two-term decomposition | reconstruction | correct |
| `open:force` | incoming term $-\sum_k\beta_{ki}\mu_k/\mu_i$ | compensation identity, s2(c) | consistent |
| `form:full-flow` | shape equation from `form:density-force` | coefficient matching | $\dot\alpha_i=-2(\alpha_i-\alpha_0)$, $\kappa$-free |
| `form:selection` | event-time formulas | algebra, s5(a) | correct; see X3-PIFB-03 |
| `open:routing` | event pattern on the limit field `open:limit` | s5(b), numerical | reproduced; exact binary field and open set not checked |
| `pifb2:eq:aggregate_prior_identity` | compensation identity | s2(c) | residual $1.1\times10^{-16}$ |
| `pifb2:eq:mixture_free_energy` | row-mixture KL | algebra | exact |
| `pifb2:eq:collective_correlation_geometry` | $(1+\rho^2)/(1-\rho^2)^2$ | hand computation | exact |
| `pifb2:eq:isotropic_base_fisher` | $2K\,d\sigma^2/\sigma^2$ | hand computation | exact |
| `pifb2:prop:meta_clock_slowdown` | both ratios | algebra | correct |
| `pifb2:eq:clock_dissipation` | contraction of the gradient equation | algebra | correct |
| `pifb2:thm:belief_configuration_gibbs_lift` | identity and minimizer | reconstruction | correct |
| `pifb2:prop:exact_posterior_fisher_rates` | linearization $-I$ | reconstruction | correct; does not constrain 2.0 (X3-PIFB-06) |

## Extensions and improvements

The generative question has a clean answer. A sample-level identification, with agent laws as recognition marginals of one population posterior, is impossible for the 2.0 potential (X3-PIFB-01). The configuration-level identification of X3-PIFB-02 is exact and costs little: agents emit $n$ messages from their current laws through their channels, receivers register whether the received type equals their own law, and the 2.0 potential at a configuration is $T=1/n$ times the collective VFE of that normalized model at a point recognition, up to receiver-only terms of order $\log n/n$. The Gibbs rows become exact source posteriors and the incoming force becomes a likelihood gradient. This keeps the law-level formation machinery of 2.0 and restores PIFB2's one collective VFE at the level of configurations, which is where PIFB2 Section 36 placed its Gibbs lift. The finite theorem needs only the method of types; the general-space version needs the Sanov double limit.

The formation results need a control. PIFB2 judged its four-cycle theorem against a frozen-prior control; the 2.0 analog is a no-interaction control. Chapter 11 fails it exactly (X3-PIFB-03), and the configuration used to prove Chapter 12 fails it numerically (X3-PIFB-04). The same Chapter 12 limit field nevertheless produces both stages from equally spaced data where the control produces none, and it reverses an initial nearest-neighbor ordering. The reflection-invariant four-agent reduction of `open:limit` is a two-dimensional ODE in which the windows can be proved with strict margins; the remaining steps (distant exterior, open set, transfer to positive laws) reuse the existing proof.

PIFB2's learned source priors are not foreign to 2.0: they are the potential's own Fisher flow in the prior coordinates, with rate equal to the row temperature (X3-PIFB-07). This is the intrinsic slow sector that Chapter 3 asks for. The identity-self-channel convention defeats it, because self priors then grow monotonically toward isolation, so the import must exclude self from the learnable rows. With that change, the law-level analog of PIFB2's selection theorem becomes a precise open problem in the joint limit $T=h^2\to0$.

The persistence and inertia part of PIFB2 should be redirected. 2.0's flows are Fisher gradient flows, every exact inheritance has spectrum inside the fine spectrum, and inherited memory kernels are completely monotone (X3-PIFB-06). Inertia therefore cannot come from 2.0 inheritance near equilibria. Two mechanisms remain: the nonreciprocal receiver-only dynamics, which is the continuous limit of PIFB2's frozen-template episodes and does oscillate on a directed cycle, and noise with barriers (X3-PIFB-10), whose temperature is naturally the message-count temperature of X3-PIFB-02.

On geometry, the determinant appendix gains a canonical scale and an exact relation between PIFB2's information clock and its Lorentz cone (X3-PIFB-09): with $\kappa_{\det}=\det M$ the Lorentzian form is one half of the moment-Gaussian Fisher pullback minus the square of the covariant entropy rate. This remains kinematics; selecting a nondegenerate $e$ by the dynamics is still PIFB2's open soldering problem. Collective geometry can take routing-derived weights at no cost to the connection theory (X3-PIFB-12), and parent-to-child feedback is expressible as a law-dependent source once the tautology of exact reverse conditionals is avoided (X3-PIFB-08).

## PIFB2-to-2.0 map

| PIFB2 item | Status in 2.0 | 2.0 location | Note |
|---|---|---|---|
| One collective VFE from a normalized population (05) | absent as foundation | `eq:dyn-fixed-vfe` only | X3-PIFB-01, -02 |
| Attention as source inference (05) | partially supported | `prop:dyn-gibbs` | rows are exact posteriors under X3-PIFB-02 |
| Consensus potential with channels (05) | supported; promoted to foundation | `def:dyn-potential` | no observation term; $\lambda_h=1$ via $R_i$ |
| Selective records and link covariance (04, 05) | supported | `prop:int-endpoint`, `prop:int-rank`, `def:int-channel` | |
| Latent Lie-algebra links, loop prior (05, 05b) | absent | `def:int-generator` is deterministic | X3-PIFB-15 |
| One statistical bundle, joint state-model law (04, 07, 18) | supported | `def:agent-section`, `geo:joint-fisher` | dynamics uses the product restriction |
| Priors from prediction (07, 10) | partially supported | `prop:bayesian-history`, `smooth:feedback` | no peer predictive priors |
| Slow sector, accumulation memory (07) | absent | Ch. 3 unit mobility | X3-PIFB-07, -14 |
| Elimination gives memory (07, 11, 29) | partially supported | `inh:causal-memory`, Ch. 5 filter remark | X3-PIFB-06(iii) |
| Precision as stiffness, Fisher as friction (08) | implicitly supported | `thm:dyn-finite-flow` | source covariance differs, X3-PIFB-05 |
| Persistence, metastability, no mass (29) | absent | deterministic dynamics | X3-PIFB-06, -10 |
| Episode-history Proca (05b) | absent | | X3-PIFB-15 |
| Parent as pushforward, coarse VFE drop (09) | partially supported | `inh:two-retentions` | no population VFE to drop |
| Link descent square (09) | supported | Ch. 5 kernel square, supplement | |
| Gaussian moment-projection parent (09) | contradicted as an autonomous parent | `inh:smooth-mixture-failure`, `inh:smooth-boundary-failure` | Ch. 11 realizes the extensive sum |
| Coherence, persistence, autonomy distinct (09) | supported | `inh:recursive-formation`, boundary square | |
| Reverse conditionals (10) | supported | `prop:conditional-lift`, `prop:common-family-recovery` | |
| Cavity messages within an episode (10) | absent | Ch. 5 remark only | |
| Top-down parent-to-child feedback (10, 12) | absent | | X3-PIFB-08 |
| Effective action, cumulants, closure residual (11) | absent | | available at configuration level via X3-PIFB-02 |
| Induced self terms (11) | supported | inherited nonidentity self channel, `form:parent` | |
| Generator descent for persistent laws (11) | supported, extended | `inh:deterministic-descent`, `inh:finite-lumpability` | |
| Aggregate prior as origin of priors (12) | implicitly supported | incoming force | X3-PIFB-05 |
| Participation as decision (12) | absent | adaptive queries without valuation | X3-PIFB-14 |
| Four-cycle selection with learned priors (13) | absent | priors fixed, Ch. 3 line 47 | X3-PIFB-07 |
| Second formation episode from the actual parent (13, 24) | partially supported | `form:selection`, `open:formation` | control fails, X3-PIFB-03, -04 |
| Fisher-length clock, dissipation (17) | partially supported | `thm:dyn-finite-flow` | X3-PIFB-11 |
| Binary registration budget (17) | absent | `def:registered-event` | X3-PIFB-14 |
| Fixed-channel Fisher contraction (17, 20) | supported | `geo:fisher-loss`, `geo:moving-channel` | |
| Pullback geometry, connection dependence, rank (18) | supported, sharpened | `geo:pullback`, `geo:gauge-connection`, `geo:quotient` | involutivity counterexample is new |
| Hermitian determinant Lorentz geometry (19) | supported as an optional appendix | `geo:determinant-metric` | soldering candidate from the collective connection; X3-PIFB-09 |
| Collective geometry, labeled mixture (20) | partially supported | `geo:collective-data`, `geo:collective` | X3-PIFB-12 |
| Public agreement, private variation (20) | supported | Ch. 4, 7, 9 | |
| Physical targets (21) | absent by design | `AUTHORING.md` | |

## PIFB2 claims needing revision

1. Claim-status table (Section 2) and Section 13: "general selective-interface and recursive formation" is no longer wholly open; 2.0 gives conditional two-stage constructions, one with a fixed selective channel. Record also that neither passes a no-interaction control (X3-PIFB-03, -04).
2. Section 9 Gaussian summaries: add that the mixture (moment-projection) retention fails autonomous inheritance under smooth peer-KL dynamics (`inh:smooth-mixture-failure`), while the extensive-sum convention inherits exactly in Chapter 11.
3. Section 7 and the open problem on emergent model variables: `prop:exact_posterior_fisher_rates` is correct but concerns exact-posterior VFE descent; law-level potentials have spread Fisher spectra and learned priors at rate $T$ (X3-PIFB-06, -07).
4. Section 8: the covariance-sector stiffness must include the source block, whose force has the opposite sign from the normalized record model (X3-PIFB-05).
5. Section 17: parent-clock contraction and the clock-dissipation identity hold for fixed retentions and gradient parents only (X3-PIFB-11).
6. Sections 5 and 7: divergence-based rows and instantaneous circular priors do have a normalized joint, at the configuration level (X3-PIFB-02); the statement that the consensus potential lacks an evidence-bound reading should be narrowed to the sample level, where X3-PIFB-01 proves it.
7. Section 5 framing ("the principal construction of the monograph is the normalized collective model"): 2.0 inverts it. The author must choose; X3-PIFB-02 lets both readings coexist.

## Prioritized roadmap

1. Registered-agreement theorem (X3-PIFB-02) with the no-go proposition (X3-PIFB-01), plus one abstract sentence and one outlook paragraph. Low difficulty; restores the single collective VFE at configuration level and gives $T=1/n$.
2. No-interaction controls for Chapters 11 and 12 (X3-PIFB-03, -04), and an interaction-necessary open-law routing theorem from equally spaced data. Medium.
3. Receiver-pool and source-pool decomposition of the force, with its covariance-sector consequence (X3-PIFB-05). Low.
4. "No underdamped mode by inheritance" proposition and completely monotone memory (X3-PIFB-06). Low.
5. Learned source priors at unit mobility without self, and the law-level matching-selection problem (X3-PIFB-07). High.
6. Parent-to-child feedback as a law-dependent source, with the tautology lemma and the Chapter 11 cohesion term (X3-PIFB-08). Medium.
7. Stochastic 2.0 with message-count temperature, Jeffreys reference and metastable persistence (X3-PIFB-10). High.
8. Canonical determinant scale and the Fisher-minus-entropy-rate identity (X3-PIFB-09). Low.
9. Routing-derived collective weights, then configuration-level joint geometry (X3-PIFB-12). Low, then medium.
10. Registration budget, inherited decisions and accumulation memory, then latent links and a history action (X3-PIFB-14, -15). Low, then medium.
