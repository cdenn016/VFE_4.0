# L05-FUS: Finite fusion and smooth sufficient states

Read in full: `chapters/06_fusion.tex` (1–422), `chapters/07_smooth_closure.tex` (1–247), `companion/c07_fusion.tex` (1–113), `companion/c08_smooth.tex` (1–76; lines 66–76 are the erasure example, outside this lens), and for context `chapters/03_dynamics.tex` (1–184) and `chapters/02_interactions.tex` (1–156), `chapters/05_inheritance.tex` (1–110, 536–560, 770–803), `AUTHORING.md`, `README.md`, `frontmatter.tex`, `chapters/06_outlook.tex`, and the drafting records `Research/docs/research/2026-10-21-endogenous-formation.md` and `2026-10-21-smooth-kl-formation.md`. Skimmed: PIFB2 `02_section.tex`, `13_section.tex`, `24_section.tex` (formation passages only). Verdict: I found no mathematical error in either chapter. Every displayed identity, the four-agent collision times, the open-gap formula, the conditional-spectrum solution, the entropy rate, the inverse map, the record example and the marked-feedback update were reconstructed by hand and checked symbolically or numerically (residuals at most 2e-10). The defects are structural. The fusion model drives every initial datum, not only the admitted class, to a single block in finite time; its law-dependent links are pure gauge with an inert parameter χ; and its fixed-variance family is a constraint rather than an invariant family. The smooth-closure completion is exact but nearly built in: the reconstruction architecture turns each private conditional into an uncoupled e-geodesic relaxation that reaches the public flow only as open-loop forcing. Its compression equals the symmetry group of the declared private reference (none for generic finite references), it is not minimal, and it fails as soon as a channel carries sender private data.

Scripts and their saved outputs are in `docs/reviews/2026-10-22-magent-theory-2.0-review/scripts/L05-FUS/` (`fus_identities.py`, `fus_dynamics.py`, `fus_variance.py`, `smooth_closure_flow.py` parts 1 and 2, `smooth_closure_records.py`, `smooth_closure_variants.py`, each with a `.out.txt`). All ran single-threaded with a 20 s timeout. Numerical agreement supports the identities; it does not prove the theorems.

## Findings

### L05-FUS-01: Every solution reaches complete synchronization in finite time, so the two-parent stage is necessarily transient
**Category:** weakness
**Severity:** medium
**Location:** chapters/06_fusion.tex:360–370 (convergence paragraph), thm fus:two-stages (286–306), fus:contraction; companion/c07_fusion.tex:113
**Evidence:** The chapter proves convergence to the all-zero configuration "for the admitted initial data" (line 369), and the companion mentions "eventual global consensus". A stronger statement holds for every initial datum and every mass vector. Since 0 is stationary, fus:contraction gives ‖μ(t)‖_M ≤ e^{−κt}R₀ with R₀ = ‖μ(0)‖_M, hence max_i|μ_i^ν(t)| ≤ e^{−κt}R₀ because n_i ≥ 1. Let D^ν = max_i μ_i^ν − min_i μ_i^ν, with top set T, bottom set B and masses n_T, n_B. At almost every t, all maximizers share the derivative of the maximum (max − μ_k ≥ 0 has a minimum at t), and the n-weighted average over T cancels the internal subgradients by antisymmetry. This gives d/dt max = −a·max + c·avg_T μ^{ν'} − λ(N − n_T), and similarly for the minimum. Hence, while D^ν > 0, D^ν' ≤ −aD^ν + 2cR₀e^{−κt} − λ(2N − n_T − n_B) ≤ −aD^ν + 2cR₀e^{−κt} − λN. After t₀ = κ^{−1}log⁺(4cR₀/(λN)), D^ν decreases at rate at least λN/2 while positive and cannot leave zero afterward. Both sectors are therefore synchronized by T* ≤ t₀ + 2max_ν D^ν(t₀)/(λN). Numerical observation (`fus_dynamics.py` Part E, three random four-agent data): complete synchronization at t ≈ 1.95, 1.30 and 0.93, while |μ| is still 0.44, 0.31 and 0.68, so consensus precedes convergence to zero. In this model every parent exists only on a finite interval, and every merge tree ends in one block after at most N−1 events. Persistent heterogeneous organization cannot occur. The main chapter does not state this.
**Fix:** State the proposition above in the handoff section. It also makes the convergence sentence unconditional. Say in the chapter introduction that the construction produces a finite coalescence tree that ends in global consensus. L05-FUS-18 sketches a variant with persistent parents.
**Falsifies:** A solution of fus:inclusion with two blocks that stay distinct for all t ≥ 0, which requires the almost-everywhere derivative identity for the maximum to fail.

### L05-FUS-02: The smooth-closure architecture makes the private sector passive: an uncoupled e-geodesic that acts on the public flow as open-loop forcing
**Category:** weakness
**Severity:** medium
**Location:** chapters/07_smooth_closure.tex:50–64 (smooth:reference, smooth:shield), 93–115 (smooth:split-dynamics), 133–164 (smooth:spectrum-flow); companion/c08_smooth.tex:18–27
**Evidence:** By smooth:peer-cancellation-rule every peer cost is a functional of public marginals only, and the private reference A_i is fixed and independent of z and w. The private equation ∂_tℓ_{i,z} = H_i(z) − ℓ_{i,z} therefore involves no public law, no row and no other agent. Its solution κ_{i,z}(t) ∝ κ_{i,z}(0)^a A_i^{1−a}, a = e^{−t}, is the exponential arc from κ(0) to A_i. The public flow sees only H_i(z,t) = aΛ'_{i,z}(a) − Λ_{i,z}(a), where Λ = log∫e^{ar}ξ⁰(dr), and this function is fixed at time zero. The "autonomous" retained state is thus the public Fisher flow driven by a precomputable time-dependent potential ∫H_i(·,t)dμ_i. The full-table finite-difference Fisher flow (`smooth_closure_flow.py`; three agents, a law-dependent public channel that depends on source and receiver, Gibbs rows) reproduces d log κ/dt = H − ℓ with residual 1.0e-10 and κ_t = κ₀^aA^{1−a}/Z with residual 1.1e-11 at t = 1. Private information never influences any interaction. The program's motivating document identifies the opposite case as the interesting one, private variables that "later affect parent interactions" (PIFB2 13_section.tex:36).
**Fix:** State explicitly after smooth:split-dynamics that the private equations decouple and are solved in closed form independently of the public dynamics, so the completion is the public flow plus open-loop forcing. Present it as a solvable baseline, and list the coupled cases (L05-FUS-08) as the open obligation.
**Falsifies:** A dependence of κ_{i,z}(t) on any public law, row or other agent in the stated architecture. The finite-difference test found none.

### L05-FUS-03: The spectrum compression equals the symmetry group of the declared private reference, and it is injective for generic finite references
**Category:** weakness
**Severity:** medium
**Location:** chapters/07_smooth_closure.tex:119–131 (smooth:spectrum-definition), 164, 172–178 (smooth:swapped-spectrum); README.md:18; AUTHORING.md:25
**Evidence:** ξ_z = (ℓ_z)_#A identifies two conditionals exactly when their densities are A-preserving rearrangements of each other. On a finite private alphabet ξ_z({r}) = A({ℓ_z = r}). If no two distinct subsets of atoms of A have equal mass, equal spectra force equal level sets and hence equal conditionals. Example (`smooth_closure_records.py` R5): A = (0.6, 0.4) and the conditionals (0.75, 0.25) and (0.25, 0.75) give spectra {(0.6, 0.223), (0.4, −0.470)} and {(0.6, −0.875), (0.4, 0.629)}, which differ. The swapped example works only because A is uniform. Noninjectivity is therefore a property of the declared reference (uniform finite or atomless), not of the dynamics.
**Fix:** After smooth:spectrum-definition, add that the spectrum is the orbit of ℓ_z under Aut(V,A), the group of A-preserving bimeasurable bijections. The retention is injective when that group acts trivially on densities, for example on finite alphabets with generic reference masses.
**Falsifies:** Two distinct conditionals with equal spectra under a reference whose atom subsets all have distinct masses.

### L05-FUS-04: The law-dependent links of the fusion example are pure gauge, and χ has no dynamical effect
**Category:** weakness
**Severity:** medium
**Location:** chapters/06_fusion.tex:8–9, 127–159 (fus:links, fus:distance); companion/c07_fusion.tex:43–56
**Evidence:** Θ_ij^ν = U_i e^{χμ_i^ν}e^{−χμ_j^ν}U_j^{−1} = V_iV_j^{−1} with V_i = U_i e^{χμ_i^ν}. Every loop is therefore trivial and Θ_ji = Θ_ij^{−1}: the link is a law-dependent gauge transformation of the trivial connection. The normalized cost equals |μ_i − μ_j| identically in χ (sympy residual 0, `fus_identities.py` check 5), so the potential and the flow are the same for every χ ∈ (0,1), including the limit of identity transport. Frames are fixed (line 30). The remark "Holding the link fixed while differentiating would give a different dynamics" amounts to replacing λ by λ/(1−χ). The introduction's claim that the construction "also specifies ... gauge transport" overstates the role of the transport.
**Fix:** State that the transport is a pure-gauge, law-dependent frame change that drops out of the normalized cost, so the example carries no gauge content beyond passive covariance. If gauge structure is meant to matter, use a transport whose parameter survives normalization, for example a nonabelian generator on a vector log space.
**Falsifies:** A dependence of fus:potential or fus:coordinate-flow on χ. There is none.

### L05-FUS-05: The fixed-variance lognormal family is a constraint, not an invariant family of a variance-inclusive Fisher flow
**Category:** weakness
**Severity:** medium
**Location:** chapters/06_fusion.tex:20–30 (fus:law-family), 179–188 (fus:inclusion); AUTHORING.md:21
**Evidence:** With free log variances v, the derivative of the local KL at v = 1/n is n(a−1)/2 (`fus_variance.py`), which is nonzero unless a = 1. At restored variance 1, the normalized peer cost between restored messages has source-variance derivative −|μ_i − μ_j|/2 (−0.5 at Δμ = 1), which is nonzero for every a whenever the means differ. The variance-1/n family, which carries the multiplicity, is therefore not invariant under the Fisher gradient flow of the same potential on Gaussians with free variances. The chapter's flow is a constrained gradient flow on a non-invariant submanifold. The general theory permits a declared regular family, so this is a legitimate modeling choice, but the factor 1/n_i is "inherited" only because the variances are frozen by assumption.
**Fix:** State that the variance is a fixed constraint and that releasing it destroys the multiplicity encoding. Alternatively, choose a = 1 and a peer comparison that is variance-neutral at the source, then prove invariance.
**Falsifies:** A variance-inclusive potential of the stated form whose variance gradient vanishes on the family. The computation shows that it does not vanish.

### L05-FUS-06: The conditional spectrum is sufficient but not minimal for the public flow
**Category:** improvement
**Severity:** medium
**Location:** chapters/07_smooth_closure.tex:133–170 (smooth:spectrum-flow, smooth:equal-spectra)
**Evidence:** The public flow depends on the private data only through H_i(z,t) modulo functions of t alone, because Fisher centering removes those. Since H = a²(d/da)(Λ/a) and Λ(1) = 0, this is equivalent to knowing the differences Λ_{i,z} − Λ_{i,z₀} on (0,1]. The differences evolve autonomously: Λ_z^t(b) − Λ_{z₀}^t(b) = [Λ_z⁰(ab) − Λ_{z₀}⁰(ab)] − b[Λ_z⁰(a) − Λ_{z₀}⁰(a)]. Example on V = V₁×V₂ with uniform A: compare κ_z = κ_z^{(1)}⊗ψ with κ_z^{(1)}⊗ψ', where ψ = (0.5, 0.5) and ψ' = (0.9, 0.1). The spectra differ at every z, yet the full Fisher flows give public trajectories that agree to 9.5e-11 on t ∈ [0, 0.5]. A control with ψ at one public value and ψ' at the other differs by 2.2e-2 (`smooth_closure_flow.py` part 2). smooth:equal-spectra is the special case in which the differences vanish.
**Fix:** Add that the public flow and the rows are determined by the public laws together with z ↦ Λ_{i,z}|_{(0,1]} modulo z-independent functions. The full spectrum is needed only for private-sensitive records. This answers "what the completion adds" more precisely than the present text.
**Falsifies:** Two configurations with equal CGF differences but different public trajectories.

### L05-FUS-07: Finite-time fusion with exact inheritance does not need the measure-zero common-offset class
**Category:** improvement
**Severity:** medium
**Location:** chapters/06_fusion.tex:228–245 (fus:sticking), 277–282, 308–330 (fus:open-gap-condition)
**Evidence:** The admitted class is an affine subspace of codimension r−1 in R^{2r}. Neither the persistence lemma nor the handoff calculation (334–358) uses it; both hold on any synchronized stratum in which both sectors coincide. Combined with L05-FUS-01, every initial datum yields a finite sequence of full-agent synchronization events, and each event produces a persistent block with exact inherited dynamics. Off the class, contacts in the two sectors are not simultaneous. A single-sector contact sticks only if |cΔμ^{ν'}| ≤ λ(n_i+n_j). Agents not in the pair lie on one side of the coincident pair, so they contribute no relative force. Otherwise the order in that sector reverses. Part D of `fus_dynamics.py` gives an example: two unit agents with Δμ^q(0) = 0.1, Δμ^s(0) = −5, a = 1, c = 0.5 and λ = 0.3 cross transversally in q at t ≈ 0.032 (Δμ^q stays in [−0.50, −0.024] on [0.2, 2]) and stick only at t ≈ 2.05 in q and t ≈ 2.38 in s. Part C: three random off-class perturbations of size 0.05 keep the two-stage pattern. The two pairs synchronize in both sectors near t ≈ 1.18–1.25, then all four near t ≈ 2.24–2.26.
**Fix:** Restate the scope of fus:sticking: finite-time fusion with exact inheritance holds for all data, and the common-offset class supplies explicit times. Prove that the two-stage pattern is open in R⁸. Before the first contact the vector field is affine with a fixed sign pattern, so contact times vary continuously. At each outer contact the cross-sector gap is small, so the sticking inequality holds with margin. The condition D > 0 keeps the middle gap positive.
**Falsifies:** An open neighborhood of the four-agent example in which a middle pair synchronizes first, or a stuck single-sector pair that later separates.

### L05-FUS-08: Channel variants: closure does not need receiver reconstruction, but it fails for channels that transmit sender private data
**Category:** extension
**Severity:** medium
**Location:** chapters/07_smooth_closure.tex:5, 58–64 (smooth:shield), 115
**Evidence:** (V1) Use a fixed reconstruction kernel K((z_j,v_j),(z_i,v_i)) = B_ij[μ](z_j,z_i)A_i(v_i), independent of the state and of the source. Every D_ij then gains the same term ∫H_i dμ_i, which passes through the Gibbs rows additively. The private energy doubles, and d log κ/dt = 2(H − ℓ) (finite-difference residual 5.7e-11). Equal-spectrum swaps keep the public trajectories equal to 3.9e-12 (`smooth_closure_variants.py`). With a fixed reconstruction ϰ_i ≠ A_i the same argument applies with the mark log(dϰ_i/dA_i). (V2) Use a private-transmitting channel K = B(z_j,z_i)δ_{v_j}(v_i). Swapping agent 0's private points, which preserves its spectra, changes the public belief velocity by 0.173 at t = 0. The per-(agent, public value) spectrum is therefore not an invariant once sender private coordinates are transmitted. Are the reconstruction channels fixed independently of the state? No: K_ij^q[q] contains the receiver's current κ_{i,z}, and B_ij depends on the public laws. Variant V1 shows, however, that this state dependence is not what produces closure. What matters is that the channel discards the sender's private coordinate and that the reconstruction is source-independent, or else is declared as a mark.
**Fix (sketch):** State the mechanism as a symmetry reduction. Let G be the group of A-preserving rearrangements under which every channel, reference and record kernel is equivariant. The Fisher–Rao flow commutes with G, and the G-orbit of the private log-density data is an exact completion. Receiver reconstruction gives G = ∏_{i,z}Aut(V_i,A_i), with independent rearrangements and hence per-z spectra. An identity private channel on a common (V,A) gives only the diagonal action, and the completion is the joint spectrum Ξ = (v ↦ (ℓ_{k,z}(v))_{k,z})_#A. Ξ is closed because every KL integrand and its variation is a function of that vector. Difficulty: the equivariance statement is short, and on atomless spaces the regularity is routine under bounded log densities.
**Falsifies:** A channel that discards sender private data and uses a fixed, source-independent reconstruction, yet for which equal spectra give different public flows.

### L05-FUS-09: The bounded-log-density hypothesis is unnecessary for the conditional part, and exponential families give exact finite-parameter subclosures
**Category:** improvement
**Severity:** low
**Location:** chapters/07_smooth_closure.tex:91, 140–158 (smooth:spectrum-solution, smooth:conditional-solution), 164
**Evidence:** For a ∈ [0,1], Hölder's inequality gives Z(a) = ∫(dκ/dA)^a dA ≤ 1, and Z(a) > 0 when κ ~ A, so κ_t ∝ κ₀^aA^{1−a} is defined for every κ₀ ~ A. For t > 0, H(t) is finite even when H(0) = ∞, because ℓe^{aℓ} ≤ Ce^{ℓ} for large ℓ. Pointwise, ℓ(t,v) = e^{−t}ℓ₀(v) + ∫₀^t e^{−(t−s)}H(s)ds. Normalization fixes the integral as −Λ(a), so the solution is unique in this class. If κ_z(0) ∝ exp(θ_z·T)A, then κ_z(t) ∝ exp(e^{−t}θ_z·T)A: the natural parameters contract linearly, and H = aθ·∇ψ(aθ) − ψ(aθ). Gaussian private conditionals relative to a Gaussian A, which the bounded-log hypothesis excludes, therefore close on finitely many parameters per public value. The theorem implies nothing about finite-parameter closure of the public laws, which remain general.
**Fix:** For the private part, replace the bounded-log assumption by κ ~ A together with whatever integrability the public force requires (for example, H_i(·,t) bounded on the support of μ_i for t > 0). Add the exponential-family case as the finite-parameter remark.
**Falsifies:** A κ₀ ~ A for which Z(a) is 0 or infinite for some a ∈ (0,1].

### L05-FUS-10: B_n and the joint reconstruction are sufficiency (Rao–Blackwell) kernels, so the inheritance mechanism extends beyond lognormal laws
**Category:** improvement
**Severity:** low
**Location:** chapters/06_fusion.tex:83–125 (fus:sample-retention, fus:message-kernel, fus:evaluator-square), 373–391 (fus:reconstruction)
**Evidence:** For n i.i.d. children from a family with a sufficient statistic S_n (here the mean log sample), the conditional law of the children given S_n does not depend on the parameter. For the Gaussian location family it is N(Z̄**1**, I − **11**ᵀ/n) (residual 0), and its one-child marginal is B_n. This explains why fresh messages descend exactly and why the evaluator square, the predictive square and the generative residual kernel (residual 5.6e-17) do not depend on the parameter. The same holds for any exponential family; for Gamma children, for example, X₁ | X̄ ~ nX̄·Beta(k, (n−1)k). The only lognormal-specific ingredient is that sqrt(2KL) between restored messages is the convex function |Δμ|.
**Fix:** Name B_n as the conditional law of one child sample given the parent sufficient statistic, citing sufficiency (Ay, Jost, Lê and Schwachhöfer 2015 is already in the bibliography). Add the generalization as a remark, with the convexity caveat.
**Falsifies:** An exponential family in which the conditional law of the children given S_n depends on the parameter. Sufficiency rules this out.

### L05-FUS-11: Missing primary sources for the nonsmooth dynamics: subgradient-flow well-posedness and finite-time aggregation for the |x| kernel
**Category:** citation
**Severity:** low
**Location:** chapters/06_fusion.tex:4–11, 179–225, 254–275
**Evidence:** In the τ-frame, the sticking flow is the particle form of the one-dimensional aggregation equation ∂_τρ = ∂_z(ρ∂_z(W*ρ)) with W = λ|z|. Sticking and finite-time total collapse are known for this equation (Carrillo, Di Francesco, Figalli, Laurent and Slepčev, Duke Math. J. 156 (2011) 229–271; Bonaschi, Carrillo, Di Francesco and Peletier, ESAIM:COCV 21 (2015) 414–441). Existence, uniqueness, the energy identity and the minimal-norm selection for subgradient flows of convex functions are due to Brezis, *Opérateurs maximaux monotones* (1973). The cited works of Cortés (2006) and Hocking et al. (2011) cover finite-time consensus and convex clustering, not these points. I give the bibliographic details from memory and did not reopen these sources in this session.
**Fix:** Cite these sources at the inclusion and at fus:sticky-flow, and say that the τ-frame dynamics is the known sticky-particle solution.
**Falsifies:** The cited works not treating the W = |x| kernel or convex subgradient flows as stated.

### L05-FUS-12: The conditional spectrum is the standard measure of the dichotomy (κ_z, A), and its log-Laplace transform is a Rényi divergence
**Category:** citation
**Severity:** low
**Location:** chapters/07_smooth_closure.tex:119–131, 140–151
**Evidence:** ξ_z is the law of the log-likelihood ratio under the reference. It determines the binary experiment (κ_z, A) up to Blackwell equivalence (Blackwell, Ann. Math. Statist. 24 (1953) 265–272; Torgersen, *Comparison of Statistical Experiments*, 1991; Le Cam 1986). It is the information spectrum of Han (2003), and it gives every f-divergence through D_f(κ_z‖A) = ∫f(e^r)ξ_z(dr). Its log-Laplace transform is Λ_z(a) = (a−1)D_a(κ_z‖A), where D_a is the Rényi divergence (van Erven and Harremoës, IEEE Trans. Inf. Theory 60 (2014) 3797–3820). The flow is the e-geodesic toward A (Amari and Nagaoka 2000). I give these details from memory.
**Fix:** Cite these sources, and note that the retained state is the maximal invariant of the dichotomy. This also answers which private statistics any future f-divergence can depend on.
**Falsifies:** ξ_z failing to determine the dichotomy up to equivalence.

### L05-FUS-13: The mean-force criterion needs an invariant admitted class
**Category:** gap
**Severity:** low
**Location:** chapters/07_smooth_closure.tex:16–25 (smooth:mean-force-criterion)
**Evidence:** The statement says that an autonomous velocity exists "precisely when the centered conditional mean on the right is determined by μ on the admitted class". The proof ends with "equality of these velocities gives equality of retained flows". That conclusion requires the criterion to hold along both fine trajectories, so the admitted class must be invariant under the fine flow. Otherwise the trajectories leave the class, and the retained velocity map is undefined at the points they reach.
**Fix:** Add "invariant under the fine flow" to the statement. Then μ(t) solves μ̇ = V(μ), and uniqueness applies.
**Falsifies:** A reading of "admitted class" that already includes invariance. The text does not say so.

### L05-FUS-14: Global well-posedness and the solution concept of the fusion inclusion are not stated
**Category:** gap
**Severity:** low
**Location:** chapters/06_fusion.tex:179–225 (fus:inclusion, fus:uniqueness), 367–370
**Evidence:** Lemma fus:uniqueness is conditional ("On any interval where a solution exists"), and global existence is shown only for the constructed class. F is convex and finite, and it is (a−c)-strongly convex in the M metric. Brezis's theorem therefore gives a unique Lipschitz solution for every initial datum, with right derivative −M^{−1}(∂F)°, the minimal-norm element. For a coincident block, the minimal-norm choice has zero internal row sums, which is the construction's choice of "zero internal subgradients". By Clarke's characterization of the generalized gradient, the inclusion coincides with the Filippov regularization of the almost-everywhere gradient, so no Filippov ambiguity arises. Backward uniqueness fails: the flow is a contraction semigroup, not a group, and this is what permits finite-time sticking.
**Fix:** State these three facts after fus:inclusion, with references (Brezis 1973; Clarke 1983, Thm 2.5.1; Paden and Sastry 1987).
**Falsifies:** A convex, finite F on R^{2r} without a global subgradient flow. No such F exists.

### L05-FUS-15: The hedge about a Fisher gradient on the spectrum coordinate can be replaced by an exact negative statement
**Category:** improvement
**Severity:** low
**Location:** chapters/07_smooth_closure.tex:164
**Evidence:** The text says the autonomous equations "have not been identified with a new unrestricted Fisher gradient on probability laws over the spectrum coordinate". A Fisher–Rao flow Q̇ = −Q(h − E_Q h) keeps Q_t mutually absolutely continuous with Q₀. The retained conditional e^rξ_z^t(dr) has support a·supp ξ_z⁰ − Λ_z(a), which moves unless κ_z = A. Hence no Fisher–Rao gradient on laws over (z, r) generates the inherited flow. It is instead a continuity equation in r with velocity field H(z) − r.
**Fix:** Replace the sentence with this statement.
**Falsifies:** A conditional with κ_z ≠ A whose spectrum support is constant in t.

### L05-FUS-16: Companion headings overstate the scope
**Category:** inflation
**Severity:** low
**Location:** companion/c07_fusion.tex:2 ("A Nonsmooth Route to Recurrent Formation") and 113 ("establishes recurrent fusion"); companion/c08_smooth.tex:2–5
**Evidence:** AUTHORING.md:19 forbids claiming recurrent formation from a supplied example. The fusion dynamics always ends in one block after at most N−1 events (L05-FUS-01), and the theorem proves two stages. The opening of c08 frames the per-agent spectrum completion in parent and child language ("A parent must retain that contribution"), but chapter 7 constructs no parent.
**Fix:** Retitle c07 as "A Nonsmooth Route to Finite Coalescence" or "Two-Stage Fusion". In c08, say that the spectrum is a per-agent state completion and that parents appear only in the erasure section.
**Falsifies:** A proof of unboundedly repeated formation in the fusion model.

### L05-FUS-17: Notation collisions across the two chapters and the companion
**Category:** consistency
**Severity:** low
**Location:** 06_fusion.tex:127 (ν as sector index, whereas 02_interactions.tex:11 uses a, with ν_i^a for marginals, and 03_dynamics.tex:26 uses ν_ij for compared laws); c07_fusion.tex:43 (σ as sector index); 06_fusion.tex:196 (σ_ij as signs) versus 07_smooth_closure.tex:47 (σ_i as model marginals); λ as the coupling in 06 versus λ_{i,w} in 07; in 07, a denotes a score part (34), e^{−t} (143) and the row log ratio a_jk (235); in 06, M denotes the mass matrix, M_left/M_right denote masses, and 𝖬 is the model space
**Evidence:** These symbols mean different things in adjacent chapters of one document.
**Fix:** Use a ∈ {q,s} for sectors throughout, rename the row ratio and the score components in 07, and rename the mass matrix.
**Falsifies:** Not applicable (notation).

### L05-FUS-18: Extensions: coalescence of arbitrary depth on the class, and persistent parents through bounded-range contacts
**Category:** extension
**Severity:** low
**Location:** chapters/06_fusion.tex:284–330; fus:smooth-open
**Evidence:** (a) Depth k. In the τ-frame, place 2^k unit agents in a dyadic configuration whose level-ℓ gaps G_ℓ satisfy G_{ℓ+1} > C·2^kG_ℓ. Block speeds are bounded by λ2^k, so all level-ℓ merges finish before any level-(ℓ+1) gap closes. Induction then gives a k-level hierarchy with exact inheritance at every level on an open subset of the common-offset class, with event times t_ℓ = κ^{−1}log(1+κτ_ℓ). Difficulty: low. (b) Persistence. Replace all-to-all contacts by contacts with a cutoff, λ_ij = λ**1**{|μ_i − μ_j| < ρ} (bounded confidence), or use a multi-well local reference. Blocks separated by more than ρ stop interacting, so several parents persist. The potential is then no longer convex and the contact graph depends on the state, so uniqueness must be re-proved through one-sided Lipschitz estimates on each contact configuration. Difficulty: medium.
**Fix:** Add (a) as a corollary. Treat (b) as the next construction if persistent parents are wanted.
**Falsifies:** (a) A dyadic configuration with the stated gap ratios whose merge order differs.

### L05-FUS-19: Specify the proximal step as the discrete update
**Category:** extension
**Severity:** low
**Location:** chapters/06_fusion.tex:370–371
**Evidence:** The proximal step is μ^{k+1} = argmin_x F(x) + (2h)^{−1}‖x − μ^k‖_M². Strong convexity gives the discrete contraction ‖μ^{k+1} − μ̃^{k+1}‖_M ≤ (1+h(a−c))^{−1}‖μ^k − μ̃^k‖_M and the energy inequality F(μ^{k+1}) + (2h)^{−1}‖μ^{k+1} − μ^k‖_M² ≤ F(μ^k). The step creates exact ties: in the scalar two-agent case, a tie occurs exactly when the gap before the step is at most 2hλ. It preserves synchronized blocks by symmetry and converges at rate O(h) on bounded intervals (Crandall–Liggett). Explicit Euler with the sign function chatters and never sticks exactly. This scheme is the natural bridge to a VFE_4.0 implementation of fusion.
**Fix:** Add the scheme and its two inequalities to the final remark of the handoff section.
**Falsifies:** Not applicable (proposed construction).

## Checked without defect

| Label | Method | Residual or reasoning |
|---|---|---|
| fus:evaluator, fus:precision | sympy: joint precision from r_n and G_n | 0 |
| fus:local-inheritance, fus:local-formulas | sympy Gaussian KL in log coordinates; Fisher information of the location family | KL − (C + nU) = 0; metric n by hand |
| fus:predictive-law | sympy | 0 |
| fus:message-kernel | variance sum | 0 |
| fus:evaluator-square, fus:predictive-square | both routes are linear Gaussian kernels; means and variances compared | 0 |
| fus:sample-retention (with joint-reference and conditional-noise inheritance) | numpy, n_j = (1, 2, 4) | ≤ 5.6e-17 |
| fus:distance | sympy, general χ | 0 (χ is inert, see L05-FUS-04) |
| fus:links | endpoint covariance; reference equivariance (scalar generators commute) | holds; pure gauge |
| fus:potential, fus:inclusion, fus:coordinate-flow | subdifferential sum rule, division by n_i, antisymmetric selections | matches |
| fus:uniqueness, fus:contraction | strong monotonicity with constant a − c = λ_min(A) in the M metric; swap symmetry for equal masses | holds; existence see L05-FUS-14 |
| fus:sticking, fus:admitted-class, fus:invariant-form, fus:time-change, fus:sticky-flow | reduction to (u, v); differentiation of e^{−κt}z(τ); block velocity, relative speed λ(n_I+n_J), mass-weighted average at contact; smoothed full eight-dimensional simulation | max \|v_i − v₀e^{−(a+c)t}\| = 2.5e-11; zero internal subgradients equal the minimal-norm choice |
| fus:two-stages, fus:collision-times | hand calculation; event-driven exact solution in the original time t (no τ) at (a,c,λ,d) = (1, 0.5, 0.3, 1) and (2, 1.5, 1, 0.5); smoothed full system | residuals 0 and 2.2e-16; smoothed first contacts 1.21219 and 2.25199 versus 1.21227 and 2.25202 (detection-threshold bias) |
| fus:open-gap-condition and τ_final | hand calculation; event-driven solution at gaps (0.7, 2.9, 1.1) and (1.3, 3.0, 0.4); the case D < 0, gaps (1, 1.2, 1.3), gives a three-block first | exact agreement; D > 0 is also necessary |
| fus:flow-square and the handoff identities | sums of energies, pulled-back metric, contact sums, uniqueness | holds on synchronized strata, where R is injective (as the text says) |
| residual KL (n−1)C | 2n-dimensional Gaussian KL, n = 2, 3, 5 | ≤ 1e-15 |
| fus:reconstruction | conditional Gaussian covariance, cross covariance with the mean, one-child marginal, reused-sample covariance 1/n | 0 |
| generative residual kernel (line 386) | (I − **11**ᵀ/n)⊗Σ_{s₁G₁} | 5.6e-17 |
| fus:smooth-open | backward-uniqueness argument (as in inh:smooth-no-finite-entry) | holds for locally Lipschitz fields tangent to a closed invariant submanifold |
| smooth:fine-score, smooth:mean-force-criterion, smooth:mean-force | weak form with bounded test functions | holds; invariance needed (L05-FUS-13) |
| smooth:hidden-profile, smooth:fisher-split | chain rule; conditional centering | hold |
| smooth:peer-cancellation-rule, smooth:peer-cancellation | identity; finite-difference derivative of the full-table energy, including dependence on the receiver conditional | T1 residual ≤ 1.9e-10 |
| smooth:split-dynamics, smooth:energy-split, smooth:split-flow | full-table finite-difference Fisher flow, three agents, law-dependent channel, Gibbs rows | ≤ 1.1e-10 |
| smooth:spectrum-definition, smooth:spectrum, smooth:spectrum-normalization | direct | hold |
| smooth:spectrum-flow, smooth:spectrum-weak, smooth:spectrum-solution, smooth:conditional-solution | derivation; explicit solution against the RK4 flow; equal-spectrum swap | 1.1e-11; public difference 4.0e-12; distinct spectra differ by 2.3e-2 |
| smooth:entropy-rate | finite difference along the explicit solution | 1.4e-11 |
| smooth:equal-spectra | full flow against an independent coarse E_c Fisher flow | 6.6e-11 |
| smooth:swapped-spectrum, smooth:spectrum-inverse | direct; numeric inverse | exact; 1.7e-16 |
| record example (line 190) | direct | 5/8, 3/8; (9/10, 1/10), (1/2, 1/2) |
| smooth:record-normalizers, smooth:feedback, smooth:feedback-update | marked formulas against direct Bayes through record, flow, record; within-mark versus cross-mark rearrangements | 2.2e-16; within-mark identical, cross-mark different (P(o=1) = 0.6 versus 0.5) |
| smooth:marked-sample-law | normalization and pushforward | holds |
| smooth:row-ratio | Fisher row equation and integration | holds |
| comp:smooth-energy, comp:smooth-flow, comp:spectrum, comp:spectrum-evolution, and the c07 formulas and table | compared with the main text | consistent |

## Extensions and improvements

The fusion chapter has more in it than it states. With the finite-time synchronization proposition of L05-FUS-01 and the well-posedness facts of L05-FUS-14, the chapter can prove for every initial datum that the trajectory produces a finite tree of full-agent synchronization events. Each event yields a persistent block whose inherited dynamics is exact, and the last event yields one block. The common-offset class then serves only to compute explicit times. Openness of the two-stage pattern in the full eight-dimensional space follows from continuity of contact times under the affine pre-contact dynamics, together with the single-sector sticking inequality |cΔμ^{ν'}| ≤ λ(n_i+n_j), which holds with margin near the class. A dyadic configuration with scale-separated gaps gives exact inheritance across any finite number of levels. None of this produces persistence, because consensus is global. Persistent parents need a state-dependent contact range or a multi-well local term, which costs the convexity on which uniqueness now rests.

The lognormal example is a special case of a general mechanism. In an exponential family, the parent law is the law of the children's sufficient statistic, the Fisher metric is multiplied by the multiplicity, and the parameter-free conditional law of the children given that statistic supplies both the message kernel B_n and the joint reconstruction. Only the identity sqrt(2KL) = |Δμ|, which makes the potential convex, is specific to Gaussian location families. On a general regular family, sqrt(2KL(p_θ‖p_θ')) = ‖θ − θ'‖_{g} + O(|θ − θ'|²), so near the diagonal the interaction is a conic, group-lasso-like penalty. Locally, a pair sticks whenever the relative smooth force has dual norm below the cone strength. Global uniqueness then requires semiconvexity estimates instead of convexity. This would give finite-time fusion with exact statistical inheritance on, for example, Gamma or Poisson families at moderate cost.

For the smooth closure, the cleanest formulation is symmetry reduction. Let G be the group of A-preserving rearrangements under which all channels, references and record kernels are equivariant. The Fisher–Rao flow commutes with G, so the G-orbit of the private log-density data is an exact completion. The chapter's receiver-reconstruction architecture maximizes G, with independent rearrangements per agent and per public value, which is why per-z spectra suffice. It also removes every private coupling. Channels that transmit sender private data reduce G to the diagonal action and require the joint spectrum Ξ (L05-FUS-08), and this is the case the program cares about. For the public flow alone, the minimal statistic is coarser still: the z-dependence of the Rényi log-Laplace transforms on (0,1] modulo z-independent functions (L05-FUS-06). On exponential families based at A the private sector closes on finitely many natural parameters, which contract as e^{−t} (L05-FUS-09). This is the finite-parameter statement the chapter currently disclaims. The open obligation, which neither chapter addresses, is a completion in which private conditionals feed back into public interactions, for example through private-transmitting channels or rows that depend on private data, and which still compresses something.

For the implementation, the proximal (implicit Euler) step of L05-FUS-19 is the discretization that keeps exact sticking, the discrete contraction and the energy inequality. If fusion is ported into the finite Gaussian model of VFE_4.0, that step should be specified in the manuscript before any code is written.
