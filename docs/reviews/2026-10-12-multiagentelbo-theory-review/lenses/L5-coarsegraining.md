# L5-coarsegraining: exact finite-step coarse-graining of the pointwise agent network

Reviewer: Claude Opus 5.5 (lens L5), 2026-10-12. Scripts and saved outputs:
`scripts/L5-coarsegraining/{hoeffding_growth.py, multiscale_vfe.py, cuts_and_tower.py, pullback_tilt.py}` with `*_output.txt`
(bare `python`, numpy 2.4.4, scipy 1.17.1, float64, each script under 5 s).

## 1. Scope and verdict

Read in full: `c09_coarse_channels.tex` (1-929), `ca_proofs_c09.tex` (1-311), `c11_network_coarsegraining.tex` (1-1644), `ca_proofs_c11.tex` (1-494), `c12_multiscale_vfe.tex` (1-1383). Read in full except the proof details of the Gaussian holonomy closed form and the latent-link corollary, which I skimmed: `c10_network_state.tex` (1-1218). Settled record consulted: the verified-ledger rows for 06/07b/07c (now c09-c12), the open-item index M1-M12, and the 2026-09-30 adjudication `A5-contracted-tower.md` (EVIDENCE_VERIFIED, Fable verifier), which settles the width forms now stated as `cor:rg-contracted-perturbation-tower`. I re-opened no settled item. The text at those labels matches the settled statements. M1 (the hierarchical factorization) is repaired: every depth factor now reads the complete next-depth state `Y_{l+1}=(Z_{l+1},N_{l+1})`, and `prop:full-graph-dropped-datum` prices the reduction.

Verdict for this lens: no stated result failed a recomputation. Every exact identity I could recompute holds to rounding: the chain rule, the Hoeffding identities, the telescoped multiplier cocycle, the native bound, the width recursion, the cut algebra, and the tower decomposition. "Exact" is used correctly for measure-level statements. The cost of exactness is stated: the state grows (hyperedges, a copy set that never shrinks, path memory), and every retained projection is a declared approximation with a displayed residual. I found two medium gaps and five low items. First, a consequence the text does not state changes how the multiscale VFE should be read and implemented: for an adaptive tower, the tower VFE equals the depth-0 VFE when recognition is pushed through the channels, and exceeds it otherwise. Second, the interaction coordinates of c11 are typed on a fixed product of agent coordinates, while the c12 state carries a random vertex set and edge variables. No finding is critical or high.

## 2. Findings

### L5-1: The adaptive-tower VFE is the depth-0 VFE; the decomposition redistributes it (unstated)
**Location:** `c12_multiscale_vfe.tex:166-257` (`lem:full-graph-adaptive-reversal`, `thm:full-graph-vfe-decomposition`); framing at `:4-12`, `:19`, `:359-361`.
**Severity:** medium
**Evidence:** For an adaptive tower the generative law is `P_0(do,dy_0) prod_l C^struct_l`. The channels read neither `o` nor recognition, so the tower posterior is `Pi_0 (x) prod C^struct`. For any tower recognition `Q`, data processing under the depth-0 marginal gives `F_tower(Q) = -log p(o) + KL(Q||Pi_tower) >= -log p(o) + KL(Q_0||Pi_0) = F_0(Q_0)`, with equality iff the conditional law of the higher depths given `Y_0` under `Q` is the structural channel. This is the bridge-lift identity of `thm:rg-exact-coarse-vfe` (ii), iterated. `multiscale_vfe.py` checks it on a 2-step tower (6, 8, 8 states, random stochastic structural channels, 100 trials). For pushed recognition, |F_tower - F_0| <= 1.3e-15 and the decomposition residual is 1.8e-15. For a bottom-up Markov recognition with different channels, min(F_tower - F_0) = 1.56 > 0. The chapter states none of the following consequences. (i) In an adaptive tower, the partition, graph, and parent-child terms of `eq:full-graph-vfe-factorized`, together with the top and observation terms, sum to `F_0`. They redistribute `KL(Q_0||Pi_0)` and add no objective. (ii) When the structural channel has deterministic components (pushed event laws, dressed links, parent coordinates that are functions of the children), a recognition family parameterized independently at depth `l+1` is singular with respect to `Pi_tower`, and `F_tower = +inf`. (iii) The sentence that a hierarchy meant to emerge variationally "puts R_l inside both the generative and the recognition law" (`:359-361`) only has content in a declared generative hierarchy or the partition catalog. In an adaptive tower the channel fixes the laws of `R_l`. The c13 formation witnesses are adaptive towers (`:19`), so their tower VFE is the fine VFE.
**Fix:** After `thm:full-graph-vfe-decomposition`, add a corollary: "For an adaptive tower, `F_tower(Q) >= F_0(Q_0)` with equality iff `Q` is `Q_0` pushed through the structural channels; the decomposition then rewrites `F_0`." Restrict the reading of the partition and parent-child terms as independent objectives to declared hierarchies and the catalog model.
**Falsifies:** An adaptive tower (structural channels reading neither `o` nor recognition) with a pushed recognition and `F_tower != F_0`.
**Status:** The theorem is ESTABLISHED and correct. The gap is an omitted consequence, and it leaves the framing inflated.

### L5-2: The c11 interaction coordinates do not type the c12 tower state (random vertex set, edge variables)
**Location:** `c11_network_coarsegraining.tex:421-435` (`def:rg-network-scale`), `:558-571` (`def:rg-interaction-step`), table `:1594-1615`; `c12_multiscale_vfe.tex:81-131`, `:380-411`, `:1318-1325` (`open:full-graph-hierarchy` asks for a closure residual "in the sense of `prop:rg-interaction-residual`").
**Severity:** medium
**Evidence:** The exact step `T_l = hd_{l+1} Abar_l asm_l` needs one fixed product `Y_{l+1} = prod_{i in V_{l+1}} Y_{l+1,i}` and a product reference equivalent to `mbar_{l+1}`. In c12, `R_l` is random and each label fixes its own branch carrier and vertex set (`:380-382`). The coarse law is therefore a mixture over `gamma` on a disjoint union, not a product-equivalent law on one product. Conditioning on a realized `gamma` replaces the base by the tilted law `mbar_l(dy|gamma) ∝ mbar_l(dy) C^R_l(y,gamma)` whenever the mark reads `y`. The tower state also contains edge variables (`eta_e`, `Theta_e`, `h`) that are not coordinates of any agent, and the Hoeffding decomposition over agents gives them no component. Neither chapter defines the step, its residual, or the multipliers for this state, yet the open problem uses the residual at this class.
**Fix:** State the step branchwise: for each `gamma` with positive mass, use the base `mbar_l(.|gamma)`, the channel conditioned on `gamma`, and the branch carrier's product reference, and weight the branch residuals by the law of `R_l`. Either include the edge variables as further product factors with their own references, or declare that the interaction layer covers agent coordinates at fixed graph datum only.
**Falsifies:** A definition already in the text that assigns Hoeffding coordinates to a random-membership step with latent links.
**Status:** Definitions plus an open problem. No tag is inflated, but `open:full-graph-hierarchy` relies on an object its own class leaves undefined.

### L5-3: The two-agent remark quotes the wrong native bound
**Location:** `c11_network_coarsegraining.tex:625-627` (`rem:rg-two-agent-example`).
**Severity:** low
**Evidence:** `prop:rg-interaction-native-growth-bound` bounds `||D_{n<-l}||` by `3^{|V_n|}-1`, which uses the target size. Here `V_1={I}`, so the bound is 2. The remark says "the bound 3^2-1=8", which uses the source size. The computed exact norm at `xi_2=0.7` is 1.0. The remark's other statements check out: the `xi_1'` formula holds to 8.9e-16, and the pair-direction image is `tanh(xi_2)` exactly (`hoeffding_growth_output.txt`).
**Fix:** Replace "3^2-1=8" with "3^1-1=2".
**Status:** The remark is tagged ESTABLISHED. This is a local misquotation.

### L5-4: The dropped-datum cost depends on which factor is reduced
**Location:** `c12_multiscale_vfe.tex:201-229` (`prop:full-graph-dropped-datum`).
**Severity:** low
**Evidence:** "If the depth-l factors ... read only (Z_{l+1},R_l)" leaves open whether `P^l_R` still reads `Y_{l+1}`. The proof replaces only the conditional law of `Y_l`. Under that reading the stated cost `sum_l I(Y_l;N_{l+1}|Z_{l+1},R_l)` holds to 1.8e-15 over 100 random towers. If the `R` factor also reads only `Z_{l+1}`, the cost becomes `sum_l [I(R_l;N_{l+1}|Z_{l+1}) + I(Y_l;N_{l+1}|Z_{l+1},R_l)]`, which also holds to 1.6e-15. The extra term was 0.0379 in one instance. The log 2 witness is exact; its partition is fixed, so the extra term vanishes there.
**Fix:** Say "the `N_l` and `Z_l` factors read `(Z_{l+1},R_l)` while `P^l_R` reads `Y_{l+1}`", or add the `I(R_l;N_{l+1}|Z_{l+1})` term.
**Status:** Tagged ESTABLISHED, and correct under the proof's reading.

### L5-5: The native growth bound lacks the tier qualifier, and a sharper bound is already available
**Location:** `c11_network_coarsegraining.tex:9-12`, `:710-738`, `:1636-1640`.
**Severity:** low
**Evidence:** I computed exact operator norms of the multiplier cocycle on binary networks with all states enumerated, for 6->3->1 and 6->3->2. For binary agents every `H_B` is one-dimensional, so the operator norm is a weighted column maximum. The runs used deterministic parity and majority channels, stochastic block-product channels, and generic channels, with non-product full-support base laws and 200 orbits per case at two scales. The telescoped form matched the product of one-step derivatives to 3.4e-15, and central finite differences of the composed nonlinear map matched to 7.5e-10. The ratio `||D_{n<-l}||/(3^{|V_n|}-1)` never exceeded 0.80, and an adversarial one-step search reached 12.45 against 26 for 6->3. The proposition is proved and correct, with two caveats. (i) The intro and the physicists' summary state the bound without the bounded-tier qualifier that the table carries. Gaussian agents are outside that tier, and no analog holds in `L^2(mbar)`: for `f=y_1+y_2` under a correlated Gaussian base, the extraction ratio is `2/sqrt(2+2 rho)`, which is 141 at `rho=-0.9999`. (ii) Applying `thm:rg-action-lp-contraction` to each tilted reverse kernel already gives `||D_{n<-l}|| <= (3^{|V_n|}-1) prod_k delta(Cdag^{psi_k}_k)`. This bound was never violated in the runs above (largest value of the norm minus the bound: -2.3e-5), and it decays with depth for stochastic channels.
**Fix:** Add "in the bounded tier" at `:11` and `:1639`. State the product-of-coefficients form as a corollary, with `delta(Cdag_psi) <= 1-e^{-w}(1-delta)` (A5, part (c)) to make it computable.
**Falsifies:** A bounded-tier orbit with `||D_{n<-l}|| > (3^{|V_n|}-1) prod_k delta(Cdag^{psi_k}_k)`.
**Status:** Tagged ESTABLISHED and correct. The summary-level wording drops a hypothesis.

### L5-6: "Only at g=0 may the tilted operator be replaced by U_l" is too strong
**Location:** `c11_network_coarsegraining.tex:593` (`lem:rg-interaction-step`).
**Severity:** low
**Evidence:** Take a deterministic channel and the nonzero `psi=f o chi` (`||g||_G=3.35`). The tilt is constant on each reverse row, so `Cdag_psi=Cdag` to 1.1e-16 (`pullback_tilt_output.txt`). This is the mechanism behind `rem:rg-trivial-invariant-pairs` (c).
**Fix:** "In general only at g=0; also wherever `psi_g` is constant on reverse rows, as on the pullback sector of a deterministic channel."
**Status:** Local wording.

### L5-7: The Hoeffding (1963) equation locator is unverified
**Location:** `ca_proofs_c11.tex:82` (Sec. 4, Eq. (4.16)).
**Severity:** low
**Evidence:** CONTENT.md marks this locator as pending. A web search confirmed the content of the lemma (`(b-a)^2/8`) but not the equation number, and I did not read the primary paper. The bound itself is correct.
**Fix:** Verify against JASA 58(301):13-30, or cite the lemma without an equation number.
**Status:** The locator is unverified. Nothing found so far shows it is wrong.

## 3. Verified without defect

| Label | Method | Residual |
|---|---|---|
| `thm:rg-exact-coarse-vfe`, `thm:cg-evidence-preserving-channel` | 200 random 7->3 channels: F = F^c + Delta, evidence, Pi C = Pi^c | 8.9e-16 |
| `prop:cg-stochastic-infimum` (ii) | direct | gap 0.5623351446188083 = 2log2-(3/4)log3 |
| `prop:cg-hard-identification` (ii) | direct | 0.003120 -> 0.058892 |
| `prop:cg-message-passing-residual` witness, `prop:rg-scale-functoriality` (iii)-(iv) | by hand | exact |
| `thm:rg-hoeffding-action-isomorphism` | alphabets 3x2x4, random product reference: commutation, idempotence, resolution, product = Mobius, locality, zero means | <= 4.4e-16; hd ratio <= 3.81 < 26 |
| sharpness `(4p-1)^n-(2p-1)^n` | n = 1..6, p = 0.999 | <= 3.4e-13 |
| `rem:rg-two-agent-example` (formula, derivative) | 200 random xi | 8.9e-16 |
| `prop:rg-interaction-intra-block-invariance` | 100 trials, 6->3 | 8.9e-16 (control with non-product base: 0.245) |
| `lem:rg-interaction-step`, `prop:rg-interaction-native-growth-bound`, `eq:rg-interaction-cocycle-telescoping` | exhaustive binary N = 6, 4 channel classes, 2 scales | telescope 3.4e-15, FD 7.5e-10, ratio <= 0.80 |
| `cor:rg-contracted-perturbation-tower` (a), (c), (e) | 300 random depth-6 stochastic towers, per-pair Birkhoff diameter | recursion never violated (max excess -3.6e-8); offsets 7.1e-15; consistent with A5 |
| `lem:cg-cut-completed-square` | 300 random GL(3) cuts: energy, Delta in both forms, interval identity, interior admissible points with PSD self terms | <= 6.8e-13 |
| `prop:cg-balanced-cut-presentation` | balance (both forms), interval membership, end symmetry, gauge law, orthogonal polar case | 2.3e-10, 0, 2.5e-10, 1.1e-6 (conditioning-limited, cond(g) ~ 5e3), 9.6e-14 |
| `prop:cg-no-balanced-law-presentation` example | Delta for twists I, 2I | 9.7e-17 |
| `open:cg-cut-compression` witness | balanced link at t = +-0.3, +-1e-4 | +-I exactly |
| `prop:cg-kernel-holonomy` (iv) | null space | dim 2, loop product 2I |
| `rem:cg-holonomy-score-loop` | numeric minimization | value and argmin to 2e-8 |
| `lem:full-graph-adaptive-reversal` | conditional of `(R_0,Y_0)` given all deeper variables equals the one given `Y_1` | 3.3e-16 |
| `thm:full-graph-vfe-decomposition` | adaptive tower and declared hierarchy | 1.8e-15 both |
| `prop:full-graph-dropped-datum` | stated cost and witness | 1.8e-15; log 2 exact |
| `prop:full-graph-downward-coupling` mixed derivative; `prop:rg-geometry-weights` expansions; `prop:link-coarse-generator-law` instances (kappa4 = 12, kappa3 = 1, diag(-1,-2,3,...)) | by hand | exact |
| `thm:rg-memory-duration`, `thm:rg-memory-projection` cycle witness, `prop:rg-bridge-kernels` witness, `prop:hist-law-fisher-agentization-nogo`, `cor:rg-perturbation-divergence-bounds` | proof reconstruction / hand computation | exact |

## 4. Implementation notes for VFE_4.0

One exact step requires the pushed triple (generative joint, posterior, recognition) through one recognition-independent channel, plus the discarded term Delta as an expected conditional KL. On finite spaces all of these are closed form. They are also closed form for linear Gaussian channels acting on Gaussian laws, where the reverse kernel is Gaussian regression. In every other case, the reverse kernel `Cdag` and the conditional partition `exp(-H^c) = int exp(-H) Cdag` need quadrature or sampling. The effective action is defined only relative to the pushed reference, so code must carry `(rho, m)` as a pair and never renormalize `m` silently. Offsets (log evidence ratios) pass through each arrow with coefficient one, while widths contract (`cor:rg-contracted-perturbation-tower` (e)). Under a channel the ELBO rises at unchanged evidence (`eq:cg-elbo-monotone`), and the constant channel attains `log p(o)`. Partitions and channels must therefore never be selected by maximizing a coarse ELBO. They are declared inputs, and the text supplies no selector (`open:cg-natural-selector`).

For the multiscale objective, L5-1 governs. If VFE_4.0 builds an adaptive tower, its tower VFE with pushed recognition is the depth-0 VFE, and the per-depth terms are diagnostics. Freely parameterized coarse recognition makes the tower VFE infinite as soon as any part of the structural channel is deterministic. A multiscale objective with independent coarse content needs one of two models: a declared hierarchy with chosen `K_down`, `P_R`, and `P_N` kernels, or the partition-catalog model with the normalizers of `prop:rg-geometry-weights`. In the adaptive case the reversal kernels `P_R(.|Y_{l+1})`, `P_N`, and `K_down` are Bayes reversals. Outside finite and Gaussian-linear models they are generally intractable, and replacing them with parametric kernels changes the model rather than approximating it, so such a replacement must be labeled.

The implementer must make and document several choices the theory leaves open:
- The parent belief map. `C_A` is declared, and `prop:rg-linear-cross-scale` offers a weighted transported mean that depends on the spanning tree whenever holonomy acts nontrivially.
- The spanning trees. Under nonflat links they are part of the channel (`prop:full-graph-coarse-links`). The rooted forest of `prop:rg-forest-tree-composition` (ii) makes nested lifts compose.
- Coarse beliefs under overlapping membership. Only the event laws are specified.
- A discrepancy measure for the cross-scale morphism defects `eq:rg-cross-morphism-defects`, which the text defines only as pairs of maps.
- The retained projection `P^ret` and its residual (`eq:rg-interaction-coordinate-residual`). This is the only place truncation enters, and its residual should be reported.

Costs and hazards:
- The Hoeffding layer has `prod|Y_i|-1` dimensions (`2^{|V|}-1` for binary agents), and the native bound grows like `3^{|V_n|}`. Exact interaction coordinates are therefore practical only for small finite blocks.
- Continuous Gaussian agents need a separate tier that the chapter does not construct, and the native bound gives no control there (L5-5).
- The coarse copy set never shrinks (`E_{l+1} := E_l`), so per-depth graph cost does not fall.
- Parallel cut copies compress losslessly into rectangular endpoint maps with stalk dimension at most `dim V_I + dim V_J`. A single link with no new self terms exists only when all dressed twists coincide (`lem:cg-cut-completed-square` (iv)).
- The balanced link is discontinuous at singular `Lambda_IJ` (`open:cg-cut-compression`), so forming `W_bal^{-1} Lambda_IJ` needs a guard on `det Lambda_IJ`. Geometric-mean square roots were accurate to about 1e-10 at condition numbers near 5e3.
- Holonomy-fixed Gaussian parents need only exact linear algebra: project the mean onto `Fix(H)` and the covariance onto `{S : kSk^T = S}`, for example via the null space of `k (x) k - I` over the generators.
- With `GL^+(K)` links that have a density, the fixed parent family is empty almost surely (`cor:link-holonomy-parent-almost-sure`). The score is then `+inf` unless the links are restricted to a group conjugate into `O(K)`.
- Width contraction along a tower requires stochastic channels. Deterministic hard-partition channels have `delta = 1` and `Delta = inf`, so residual widths add with depth.
