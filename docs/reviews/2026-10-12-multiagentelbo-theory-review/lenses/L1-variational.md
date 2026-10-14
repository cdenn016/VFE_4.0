# L1-variational: variational inference and free-energy lens

Reviewer: Claude Opus 5.5 (investigator check, not adjudicated), 2026-10-12. Scripts: `scripts/L1-variational/`
(`a_gaussian_elbo.py`, `b_two_timescale.py`, `c_cycle_and_witnesses.py`, `c2_witnesses.py`, `d_followups.py`),
all CPU float64 or exact rational/SymPy, each under 10 s.

## 1. Scope and verdict

I read in full `c03_probability.tex` (1-562), `c04_generative.tex` (1-363), `c05_evidence_bound.tex` (1-1228),
`ca_measure.tex` (1-210) and `c08_two_timescales.tex` (1-691), together with the definitions they lean on:
the standing conventions `c01_program_scope.tex:153-178`, the running-example links `c02_geometry.tex:522-558`,
`def:rg-formation-cycle-experiment` and `prop:rg-formation-endogenous-cycle` (`c13_formation.tex:1205-1325`),
`hyp:hist-exact-vfe-lift` (`c07_fiber_geometry.tex:421-460`), `prop:hist-scalar-mobility-orbit` and
`def:hist-oriented-semiconjugacy` (`c07:785-813`, `c07:1190-1224`). I grepped the verified ledger for every label in
scope and did not re-open its settled rows. The verdict for this lens is that the variational core is sound: the
joint is normalized as claimed, the extended bound is well posed for every law, every identity I recomputed on a
correlated linear-Gaussian three-agent model with cyclic records holds to 1e-13 or better (finite-difference gradients to 4e-10), the E/M bookkeeping
(exact E-coordinate, accepted M-coordinate, evidence-free acceptance, evidence monotonicity) is correct and does not
borrow the Neal-Hinton license beyond one shared functional, and every quantitative claim of the two-timescale
chapter (envelope identity, rate-separated flow, correlated-metric witness, four-cycle Hessian blocks,
characteristic polynomial, rate bounds, partial-refresh witness) recomputes, symbolically where the claim is exact. I found one local error
(a descent statement in the parent sweep that fails under its literal reading) and four low-severity precision
issues. No finding is critical or high.

## 2. Findings

### L1-1: The parent sweep need not decrease the stated free energy when the record conditionals are mismatched
**Location:** `c08_two_timescales.tex:572-585`, `thm:rg-memory-inference-descent`(iv), last sentence; free energy `eq:rg-memory-fixed-vfe` (`:560-563`).
**Severity:** medium
**Evidence:** Part (iv) proves the loss identity \eqref{eq:rg-memory-reverse-conditional-loss}, which is a statement about
$\KL(q\Vert p)$ only, and then concludes "Hence $\mathsf T$ decreases \eqref{eq:rg-memory-fixed-vfe}". That free energy
is $\KL(q\Vert p)+\E_q\KL(\psi^\prime_\xi\Vert\psi_\xi)-\log\mathcal Z_o$ (part (i)), and (iv) applies $\mathsf T$ to $q$
("For every initial $q$, $\mathsf T$ preserves component masses"). With $\psi^\prime$ held, the second term becomes
$\E_{q\mathsf T}\KL(\psi^\prime_\xi\Vert\psi_\xi)$ after the sweep and can rise by more than $\KL(q\Vert p)$ falls.
Witness (`d_followups.py` [1a]): $\Xi=\{0,1\}$ with singleton parent fibers (so $\mathsf D=I$), uniform $p$,
$\mathsf a_{01}=1/2$, $\psi_1=(0.999,0.001)$, $\psi^\prime_1=(0.001,0.999)$, $\psi^\prime_0=\psi_0$, start $q=\delta_0$: the
free energy goes from 0.6931 to 3.4465 (constant $-\log\mathcal Z_o$ omitted). Under the other reading, the marked
sweep of part (ii), the value does decrease, but it decreases strictly even at a $q$ that already equals the
conditional target whenever $\psi^\prime\neq\psi$ on its support ([1b]: 0.3426 to 0), so the stated strictness
condition omits $\psi^\prime_\xi=\psi_\xi$. In the matched case $\psi^\prime=\psi$ both readings agree and the claim is
correct, which is what the ledger row for `thm:rg-memory-inference-descent` checked (loss identity for
$\mathsf K\in\{\mathsf M,\mathsf D,\mathsf T\}$).
**Fix:** State the conclusion of (iv) for the marked sweep $\widetilde{\mathsf T}$ acting on $q\otimes\psi^\prime$, with
equality exactly when $q(\xi)=q(\Xi^\prime)p(\xi\given\Xi^\prime)$ on each component and $\psi^\prime_\xi=\psi_\xi$ for
$q$-almost every $\xi$; or restrict (iv) to $\psi^\prime=\psi$ and write $\KL(q\Vert p)$ in place of
\eqref{eq:rg-memory-fixed-vfe}.
**Falsifies:** A reading of (iv) under which $\mathsf T$ acts on $q\otimes\psi^\prime$ by the marked kernel and
"strictly unless C" means only "if not C then strict" (not an equality characterization); under that reading the
sentence is true and the finding reduces to a request to name the marked kernel.
**Status:** ESTABLISHED; inflated in the last sentence of (iv) only. Parts (i) to (iii), (v) and the loss identity
are correct. No other chapter cites (iv) (grep), so the error does not propagate.

### L1-2: A continuum of parameter values is evaluated at one fixed observation without the version hypothesis used elsewhere
**Location:** `c08_two_timescales.tex:96-116`, `def:obs-fast-slow-blocks` and its use in `lem:obs-envelope-slow-flow` and `prop:obs-rate-separated-flow`; `c05_evidence_bound.tex:978-1002`, `prop:elbo-random-link-vfe`.
**Severity:** low
**Evidence:** (H1) (`c05:122-131`) says "A comparison of finitely many parameter values uses an observation in the
intersection of their regular sets", and `prop:elbo-exact-m-coordinate` (`c05:1043-1050`) requires "a common
regular-observation set or jointly measurable pointwise versions for the candidates compared". The fast-slow
objective \eqref{eq:obs-fast-slow-objective} puts the generative parameter $\theta$ in the slow block and moves it
along a $C^1$ flow, so it evaluates $-\log p_{\theta,V}(o\given X)+\KL(Q\Vert\boldsymbol\Pi_{\theta,V,o,X})$ at one
$o$ for uncountably many $\theta$ and differentiates it in $\theta$. Each $\mathsf O^{\rm reg}_\theta$ is only
$\mathbb P^{\mathsf O}_\theta$-full, an uncountable intersection need not be measurable or nonempty, and the pointwise
value at a fixed $o$ is version dependent (`rem:prob-version-witness`). The definition sidesteps this by declaring
$F$ "real-valued" and the lemma by assuming $C^1$, so nothing proved is false, but the hypothesis that makes the
object exist is not named. `prop:elbo-random-link-vfe` has the same gap for $\Theta\mapsto p(o\given\Theta)$, and it
also needs $\Theta\mapsto\boldsymbol\Pi_\Theta$ to be a kernel for $\E_R\KL(Q_\Theta\Vert\boldsymbol\Pi_\Theta)$ and
for "the posterior has conditional law $\boldsymbol\Pi_\Theta$" to make sense.
**Fix:** Add to `def:obs-fast-slow-blocks` (and to `prop:elbo-random-link-vfe`) a declared jointly measurable version
$(\theta,o,y)\mapsto p_{\theta,V}(o,y\given X)$, positive finite evidence at the observed $o$ for every $\theta$ in
$\mathcal U_{\rm s}$, $C^1$ dependence of that version on $\theta$, and, for the links, a measurable posterior kernel
$\Theta\mapsto\boldsymbol\Pi_\Theta$, all as model data in the sense of `hyp:prob-regular-conditional-convention`.
**Falsifies:** A standing convention elsewhere in the main manuscript that already declares jointly measurable,
parameter-continuous density versions for every parameterized family (I found none in c01, c03 or c05).
**Status:** Definition and ESTABLISHED proposition; not inflated, an unnamed hypothesis.

### L1-3: The four-cycle "exactly when both slow" needs the link rate to be of the order of the model rate
**Location:** `c08_two_timescales.tex:355-361`, closing sentence of `prop:obs-slow-model-link-dichotomy`.
**Severity:** low
**Evidence:** The slow eigenvalue of the selected-edge block is
$\varsigma_-=2(1+\varepsilon_{\rm m}+\varepsilon_{\check\omega})-2\sqrt{(1-\varepsilon_{\rm m})^2+\varepsilon_{\check\omega}^2}
=4\varepsilon_{\rm m}+2\varepsilon_{\check\omega}+O(\varepsilon^2)$. With both rates small but
$\varepsilon_{\check\omega}\gg\varepsilon_{\rm m}$ the model disagreement is gone long before time $1/\varepsilon_{\rm m}$:
at $\varepsilon_{\rm m}=10^{-4}$, $\varepsilon_{\check\omega}=10^{-2}$ one gets $\varsigma_-=2.03\times10^{-2}$ and
$|r^m_i(1/(12\varepsilon_{\rm m}))|=4.5\times10^{-8}$, while at $\varepsilon_{\check\omega}=\varepsilon_{\rm m}=10^{-4}$
the same quantity is 0.6065 (`d_followups.py` [2]). The proved arms are $\varepsilon_{\check\omega}=1$,
$\varepsilon_{\rm m}=1$, and $\varepsilon_{\check\omega}=\varepsilon_{\rm m}\le1/2$; "exactly when the model means and
the links are both slow" reads naturally as "both rates small", which is not sufficient.
**Fix:** Replace "both slow" by "$\varepsilon_{\check\omega}=O(\varepsilon_{\rm m})$", or confine "exactly" to the
three arms and add the asymptotic slow rate $\varsigma_-\simeq4\varepsilon_{\rm m}+2\varepsilon_{\check\omega}$. The
physicist summary (`:690-691`, "only if the links are slow too") is correct as an "only if".
**Falsifies:** Reading "In these arms" as confining the "exactly when" to the three arms; under that reading the
sentence is correct and the finding is a clarity request.
**Status:** ESTABLISHED; every displayed rate, coefficient and bound recomputes (Section 3); the closing sentence is mildly inflated.

### L1-4: The conjugacy and the length limit of the rate-separated proposition concern the lift, not the finite-rate flow
**Location:** `c08_two_timescales.tex:181-204`, `prop:obs-rate-separated-flow`, from "The reduced field" to the end of the statement.
**Severity:** low
**Evidence:** $\mathscr V^\star:=(Df^\star v,v)$ is by definition the image of the reduced field under the graph map,
so \eqref{eq:obs-reduced-semiconjugacy} and the conjugacy hold by construction, and the $\mathsf G_\varepsilon$-length
statement is about the lifted reduced orbit, which no member of the $\varepsilon$-family traces (the remark at
`:229-231` notes that the finite-rate velocity $(0,v)$ is tangent to $\mathcal S^\star$ only where $Df^\star v=0$). On a
nonlinear Gaussian instance (`b_two_timescale.py` [2]) the finite-rate orbit stays off the graph by
$\max_{t\ge1/2}|f_\varepsilon-f^\star(s_\varepsilon)|=0.52\varepsilon$ and tracks the reduced slow orbit to
$0.227\varepsilon$, for $\varepsilon$ from $10^{-1}$ to $10^{-3}$; this is Tikhonov behavior, which the chapter
correctly leaves open (`open:obs-slow-manifold`). The chapter summary (`:687-688`) states the situation correctly.
**Fix:** One sentence in the proposition: the conjugacy and the length limit are properties of the lift
$\mathscr V^\star$ and say nothing about orbits of \eqref{eq:obs-rate-separated-flow} off $\mathcal S^\star$.
**Falsifies:** Nothing mathematical; the finding is retracted if the author regards the existing summary sentence as sufficient.
**Status:** ESTABLISHED and correct; the risk is a reading as a dynamical reduction result.

### L1-5: Hypothesis (H1) asks for a latent density that the extended branch never uses
**Location:** `c05_evidence_bound.tex:122-131` (H1) and `:225-277` `def:elbo-extended`; uses at the law-point and configuration levels, `c03_probability.tex:492-514` (`prop:prob-law-point-point-mass`) and `:516-542` (`prop:prob-configuration-lift-identity`).
**Severity:** low
**Evidence:** `def:elbo-extended` assumes (H1), whose first clause requires the joint to have a density
$p_{\theta,V}(o,y\given X)$ against $\nu^{\mathsf Y}\otimes\nu^{\mathsf O}$. The extended objects themselves,
\eqref{eq:obs-global-evidence}, \eqref{eq:obs-global-posterior} and the second form of \eqref{eq:elbo-slice-measure},
are built from the record likelihood $L_o$ and the latent kernel $\mathbb P^{\mathsf Y}_{\theta,V}$ alone. The
law-point and configuration results invoke \eqref{eq:elbo-extended-definition} in a setting where
`open:prob-law-point-prior` says the law fibers carry no canonical sigma-finite reference, and a lifted law with a
deterministic configuration kernel has no density against a product reference. The gap is harmless at a fixed
parameter because $\nu^{\mathsf Y}:=\mathbb P^{\mathsf Y}_{\theta,V}$ always serves (the joint density is then $L_o$),
but `def:prob-reference-measures` asks for parameter-independent references, so the reader has to supply this step.
**Fix:** Split (H1): the extended branch needs only record kernels dominated by $\nu^{\mathsf O}$ with a jointly
measurable likelihood, plus (H2); the latent density enters with (H3) and (H4). State once that
$\nu^{\mathsf Y}=\mathbb P^{\mathsf Y}_{\theta,V}$ is admissible at fixed parameters.
**Falsifies:** A passage in c03 to c05 that already states the extended branch for kernel-only latent laws (I found
none; `hyp:local-interaction-kernels` admits law-point blocks but does not relax (H1)).
**Status:** Definition; no ESTABLISHED claim is false.

## 3. Verified without defect

| Label | Method | Residual or value |
|---|---|---|
| `thm:elbo-extended-gap`, `thm:elbo-exact-identity`, `eq:elbo-factorization` | 6-dim correlated Gaussian $Q$ on a 3-agent linear-Gaussian DAG with records of scopes {1,2}, {2,3}, {3}; KL form against the factorwise classical split | 3.6e-15 |
| `eq:obs-collective-energy-form` | same instance, $\KL(Q\Vert\mathbb P^{\mathsf Y})+\E_QH_o$ | 0.0 |
| `eq:elbo-total-correlation-chain`, `eq:elbo-total-correlation-signs` | arbitrary product reference over agent blocks; pseudo-ELBO against ELBO + TC; at $Q=\boldsymbol\Pi$, pseudo-ELBO minus $\log p$ equals TC of the posterior, 1.36817 | 1.8e-15, 3.6e-15 |
| `lem:obs-block-posterior`, `eq:obs-local-energy-form` | block = agent 2; $-\log Z_B+\KL(r\Vert\boldsymbol\Pi_B)$ against $\KL(r\Vert\mathbb P_B)+\E_rH_B$ at 1296 Gauss-Hermite nodes | max 7.8e-14 |
| `eq:obs-posterior-block-kl-chain`, `thm:obs-local-global-potential`, `eq:obs-local-global-decomposition` | correlated $Q$ and $Q^\prime$ with a common outside marginal, $\Delta F=3.2056$ | 8.0e-15, 8.9e-15, 7.1e-15 |
| `prop:elbo-exact-e-coordinate`(iii) | $\KL(\kappa R\Vert\boldsymbol\Pi)-\KL(R\Vert\boldsymbol\Pi_{-B})$; the best of 2000 perturbed Gaussian kernels exceeds the optimum by 0.0505 | 1.8e-15 |
| `prop:obs-local-block-descent`(ii), `cor:obs-potential-game` | product Gaussian family; finite-difference $\nabla_{\mu_2}F-\nabla_{\mu_2}\overline F_2$; unilateral change of $r_2$ | 3.6e-10; 1.8e-14 |
| `cor:elbo-acceptance-evidence-free`, `prop:elbo-evidence-monotonicity` | $\theta$ = prior mean of $m_1$; 24 exact EM steps | 0.0; minimum evidence increment +7.1e-3 |
| `lem:obs-envelope-slow-flow` | nonlinear instance $k\given m\sim\mathcal N(m,e^{-m})$, product Gaussian $Q$, Newton solve for $f^\star$; the closed-form $F$ was checked against a 2e6-sample Monte Carlo (3.1788 against 3.1770 with standard error 0.0016) | max 5.2e-11 over 20 points |
| `prop:obs-rate-separated-flow` (dissipation); `open:obs-slow-manifold` (Tikhonov behavior under its hypotheses) | Radau integration, $\varepsilon$ from 1e-3 to 1e-1 | $F$ nonincreasing; $|s_\varepsilon-s|\le0.227\varepsilon$ |
| `rem:obs-rate-separation-witnesses` | SymPy: critical line $f=\frac{1+2c}{1+c}s$, rate $1/(1-c)$, $dF/dt=(1+\varepsilon)^2/(12\varepsilon)-4/3$, roots $7\pm4\sqrt3$, congruence fast component | all residuals 0 |
| `prop:obs-slow-model-link-dichotomy`(i) | SymPy: the Hessian of $U_{S^{(0)}}$ in the stated orthonormal frame equals \eqref{eq:obs-cycle-held-blocks}; minors $4+3T$, $(4+T)(4+5T)$, $T(8+T)(4+5T)$ | exact |
| `prop:obs-slow-model-link-dichotomy`(ii) | characteristic polynomial, discriminant and kernel $(1,1,\sqrt2)$ exact; bounds by matrix exponential on time grids | fast link margin 4.4e-16; fast models 1.1e-14; slow-pair bounds hold with margins -1.0e-3, -3.0e-3, -3.0e-3; at $\varepsilon=1/25$, $\max|r^b|/\varepsilon=0.316$ and $\min|r^m|=0.596$ |
| `prop:obs-slow-model-link-dichotomy`(iii) | hand recomputation of the drift bound, the noise variance $2T(2\varepsilon_{\rm m}+\varepsilon_{\check\omega})$ and the Doob exponent | correct |
| `rem:rg-memory-partial-refresh` | exact rationals: $1/6+\omega/12$ and $1/2-\omega/4$; difference 7/30 at $\omega=3/10$; both 1/4 at $\omega=1$ | exact |
| `prop:prob-access-restriction`(ii) | Chernoff floor and minimax over $t$ | 0.08849703 at $\xi=0.54826$; minimax 0.08849704 at $t=0.78336$ |
| `prop:elbo-integrability-witnesses`(c), (d) | width sum 0.0722885 < $e^{-2}$; (d) by hand | correct |
| `prop:elbo-random-link-vfe` | finite instance, 4 link values and 5 latent states | 0.0 |
| `rem:gen-appended-energy` | finite instance | 0.0 |
| `prop:elbo-exact-m-coordinate` counterexample, `prop:gen-moving-target-witness`, `prop:restrict-unrelated-joint-elbo`, `prop:elbo-finite-step-nonmonotonicity` | settled ledger rows; re-derived by hand | consistent |
| `lem:prob-kl-chain-rule`(a) to (e), `prop:elbo-relative-log-representation`, `thm:prob-kernel-rn-measurable-version`, `cor:prob-kernel-kl-measurability`, `thm:prob-mixed-family-common-domination` | proofs read line by line under the c01 standard-Borel default | correct |
| c04 running example, residual read by $a_2$ | $\Theta_{\bar e_1}=\Theta_{e_1}^{-1}$, $\mathrm{Hol}=R_z(\pi/2)R_x(\pi/2)$ of angle $2\pi/3$, so $\lvert(I-R)v\rvert=2\sin(\pi/3)\lvert v_\perp\rvert=\sqrt3\lvert v_\perp\rvert$ | correct |
| Citations | Danskin 1967 "Ch. III" locator not verified (the Springer page sits behind a login); Kallenberg 2021 randomization lemma cited without a locator; Monderer and Shapley 1996, Sec. 2, not opened | not checked |

Where the measure-theoretic hypotheses are used. The c01 default that every measurable space is standard Borel
is used in four ways: existence of regular conditional laws (the posterior version, the baseline block
disintegration \eqref{eq:obs-baseline-block-disintegration}, the conditional kernels in `lem:prob-kl-chain-rule`
and the coordinate swap in its proof of (d)); countably generated sigma-algebras (the partition martingale in
`thm:prob-kernel-rn-measurable-version`, the partition supremum in `cor:prob-kernel-kl-measurability`, the
quantization proof of `lem:prob-kl-chain-rule`(b)); the Kuratowski isomorphism in
`thm:obs-agent-interaction-equivalence`; and the separating generator in
`prop:prob-marginals-do-not-determine-joint`. It suffices everywhere I checked, provided the law fibers used as
law-point coordinates are Borel subsets of $\mathcal P(\mathsf K)$ with its evaluation sigma-algebra (standard Borel
when $\mathsf K$ is, Kechris, Classical Descriptive Set Theory, Thm 17.23, cited from memory), which the default
asserts but does not argue. Absolute continuity enters only as (H3), for the classical split; the extended bound,
the chain-rule decompositions, the potential identity and the E-coordinate all hold in $[0,+\infty]$ without it.
Finite positive evidence (H2) is what makes $\log p(o)$ finite and the extended bound well defined; it is a
pointwise condition on a version, which is harmless for one parameter value and becomes the L1-2 issue for a
continuum.

## 4. Implementation notes for VFE_4.0

What must be computed. The one objective is $\mathcal F^{\rm ext}_o(Q)=\KL(Q\Vert\mathbb P^{\mathsf Y}_\theta)+\E_QH_o$
(`eq:obs-collective-energy-form`), which needs neither the evidence nor the posterior: the latent prior is a
directed product of model and state factors, and the records contribute $\E_QE_{a,o}$ once each. An implementation
should compute this form. It should not sum per-agent local free energies, which count each record
$\lvert\partial a\rvert$ times (`eq:obs-singleton-incident-counting`), and it should not substitute block-marginal
entropies, which overstates the bound by exactly the total correlation (1.368 nats at the exact posterior of my
small model). For Gaussian recognition laws with linear-Gaussian factors and records every term is closed form,
and `a_gaussian_elbo.py` is a working template (prior moments from the structural-equation form, evidence and
posterior by Gaussian conditioning, local quantities by affine conditionals). For state-dependent noise of the
running-example type, $k\given m\sim\mathcal N(m,e^{-m})$, the expected log factor is still closed form through
lognormal moments (`b_two_timescale.py`). Binary success records $e^{-E}$ with quadratic $E$ are closed form;
failure outcomes contribute $\E_Q\log(1-e^{-E})$, which is not, and need quadrature or sampling. The evidence
$\log p(o)$ is needed only to report the absolute value of the bound; accepted M-steps can use the evidence-free
difference \eqref{eq:elbo-acceptance-evidence-free}.

Local updates and their order. The local free energy of a block needs the baseline conditional
$\mathbb P^{\mathsf Y}_B(\cdot\given y_{-B})$, which for a directed prior involves the block factors, the factors of
its children and a normalizer over $y_B$; it is closed form only in conjugate cases. The monotonicity guarantees of
`prop:elbo-exact-e-coordinate`(iv) and `cor:obs-potential-game` cover unilateral, sequential updates only.
Simultaneous exact mean-field updates are not covered, and they do fail: on a three-block Gaussian with pairwise
correlation 0.6 the Jacobi sweep raises $F$ from 0.46 to 9.0 in ten sweeps while Gauss-Seidel lowers it to 6e-5
(`a_gaussian_elbo.py` [8]). The damping of (iv) is a mixture of kernels, which leaves a Gaussian family; damping in
parameter space is not covered by (iv). Within one exponential family,
$\KL(q_\mu\Vert p)=A^*(\mu)-\langle\eta_p,\mu\rangle+A(\eta_p)$ is convex in the mean parameters, so damping toward an
in-family target in mean (moment) coordinates inherits the inequality of (iv), and damping in natural coordinates
does not in general. Point-mass or rank-deficient recognition of a continuous latent lies on the singular branch
with bound $-\infty$ (`prop:elbo-subspace-support-singular`, `prop:prob-law-point-point-mass`): MAP-type updates are
not recognition optima, and code must return an infinite divergence rather than NaN for them.

Two timescales. The theory declares, and does not derive, the fast/slow split, the rate ratio and the block
metrics; an implementer has to choose them, and the choice changes the selected orbit
(`prop:hist-scalar-mobility-orbit`). A correlated belief-model metric with row rates is not a descent field
(`rem:obs-rate-separation-witnesses`); use the congruence form $\Delta_\varepsilon\mathsf G\Delta_\varepsilon$. The
reduced slow gradient is $\partial_sF(f^\star(s),s)$ by the envelope identity, so no differentiation through the
inner solve is needed, but the error of that gradient is first order in the unconverged fast residual. The explicit
rate-separated flow is stiff (fast eigenvalues scale as $1/\varepsilon$); an implicit integrator or an inner Newton
solve for $f^\star$ is the stable choice, and finite steps need an acceptance test on the same functional
(`prop:elbo-finite-step-nonmonotonicity`). When $\theta$ is in the slow block and the fast family is restricted,
the reduced flow converges to a minimizer of $F_{\rm red}$, not to the maximum-likelihood parameter: in my instance
the product-family reduced minimizer is $\theta=0.912$ while $\theta_{\rm ML}=1.166$, with a residual gap of 0.308
nats. The theory claims nothing else, but reported learned models are variational-EM estimates.

Under-specified choices an implementer must invent. The coupling selector $\mathsf S$ (the product coupling is
admissible but pays the total-correlation gap, and the exact E-coordinate leaves the product family); the update
order and damping parameterization above; a halting rule (convergence of the alternating scheme is open,
`open:elbo-alternating-convergence`); the acceptance rule for M-steps (a line search or trust region on
\eqref{eq:elbo-acceptance}); density versions at the observed $o$ (choose continuous versions and declare them, see
L1-2); and, in the parent sweep, whether the record conditionals are refreshed together with the fine state (L1-1:
the marked sweep is the one that descends). Numerical hazards: $e^{-\mathcal D/T}$ at $T=\epsilon^8$ in the
formation regime underflows and needs log-domain row normalization; lognormal moments $e^{s+v/2}$ overflow for
large model means; infinite record energies ($L_o=0$) must be carried as $+\infty$ in $H_o$, so that the extended
bound reads $-\infty$ rather than NaN.
