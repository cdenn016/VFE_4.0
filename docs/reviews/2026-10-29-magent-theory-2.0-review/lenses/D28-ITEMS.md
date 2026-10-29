# D28-ITEMS: Working documents Items 1 to 3 versus the integrated chapters 17 to 19

I read the following files in full:
- the three working documents: `Item_1_communication_links.tex` (1-543), `Item_2_link_dynamics.tex` (1-305) and `Item_3_curvature_hierarchy.tex` (1-288);
- `chapters/17_communication_links.tex` (1-500) and `chapters/19_curvature_hierarchy.tex` (1-268);
- `chapters/18_link_dynamics.tex` (1-642). Lines 304-624, the coupled-dynamics sections added on 2026-10-28, were read but not verified;
- the three check scripts and their recorded outputs;
- the 2026-10-23 `gauge-audit.md`, `variational-audit.md` and `REPORT.md`, and the 2026-10-23 source-snapshot copy of Item 3;
- the integration `implementation-record.md` and `integration-validation.json`;
- the parts of the main manuscript that the Items invoke: `chapters/03_dynamics.tex` lines 1-75 and 315-440 (reciprocal potential, per-agent model, registered agreement) and `chapters/02_interactions.tex` lines 205-245 (loop blindness, law-dressed defect).

I compared the snapshot Items 1 and 2 with the current versions through sentence-level diffs rather than line by line, and I skimmed the 2026-10-22 report for E8, E9 and R9.

Verdict:
- Chapters 17 to 19 are faithful, correctly repaired versions of the working documents. I found no false theorem, proposition or corollary statement in either version, and every load-bearing proof step I reconstructed holds.
- The working documents are not in the same state. Item 3 is now identical to chapter 19, and Item 2 matches the first part of chapter 18 except for one normalization sentence. Item 1, however, still carries four statements that chapter 17 corrected and that are false or unsupported as written (ownership, complete repair, the bandwidth-to-reciprocal identification, and attention as a special case), and the integration record says otherwise.
- The check scripts are frozen at the pre-correction state. They reproduce their recorded outputs byte for byte, but they test none of the integration corrections, and they do not reproduce two magnitudes that Item 1 quotes.

## Findings

### D28-ITEMS-01: Item 1 identifies the per-agent picture with receiver-factorized reverse-KL recognition, and its conclusion that ownership leaves the estimates unchanged is false for the per-agent model with peers' laws as data
**Category:** error
**Severity:** medium
**Location:** `Item_1_communication_links.tex:512`, `it1:owned` (corrected at `chapters/17_communication_links.tex:483`)

**Evidence:** Item 1 says: "In the per-agent picture of the main manuscript each receiver infers only its own dictionaries, so its recognition law is one factor q_i(ω_i·) of a law factorized over receivers ... In the Gaussian tangent model, ownership therefore leaves the estimates of the dictionaries unchanged and makes each receiver overconfident about them."

The computation behind this sentence is the reverse-KL block mean-field approximation of the single global posterior (Bishop 2006, Section 10.1.2). That approximation belongs to the global picture (`03_dynamics.tex:5`). In the main manuscript's per-agent picture (`def:dyn-per-agent`, `03_dynamics.tex:319-334`), each agent has its own normalized model in which the peers' laws are data. Item 2's tangent version (`it2:per-agent-model`, `Item_2:51-66`) integrates each loop factor against the peers' dictionary laws, which widens its precision to Ω_iγ = ((λW_γ)^{-1} + Σ_f V_f)^{-1}.

Reconstruction: in the plug-in model the dictionary rows equal those of the global precision (`it2:per-agent-flow`(iii)), so the block fixed point is the posterior mean. The widened loop precision changes those rows, so the law-level fixed point is not the posterior mean.

I computed all three on Item 1's tangent learning model: the complete exchange of three agents, K = 3, rank-one codes, both directed triangles and the three two-cycles, λ = 2 (`scripts/D28-ITEMS/d28_item1_verify.py`, block F).

| Recognition | Largest deviation of means from exact | Total variance |
| :-- | :-- | :-- |
| exact posterior | 0 | 10.83 |
| receiver-factorized reverse KL | 4.7e-15 | 6.77 |
| plug-in per-agent fixed point | 1.6e-14 | not computed |
| law-level per-agent fixed point | 0.0444 (largest exact entry 0.538) | 9.25 |

Item 2's own `it2:echo` parts (ii) and (iii) already show that these recognitions differ in variance. The remark is therefore correct for the global factorized recognition and for the plug-in per-agent model, and false for the per-agent model that uses peers' laws. Chapter 17 rewrote it as "Under a variational restriction with factors owned by receivers"; Item 1 keeps the uncorrected version, which violates the three-objective rule.

**Fix:** Replace "In the per-agent picture of the main manuscript" with "Under a receiver-factorized variational restriction of the global posterior", and add: "The plug-in per-agent model of Item 2 has the same means and covariances; its law-level per-agent model changes both." Alternatively, adopt chapter 17's paragraph.

**Falsifies:** A proof that the law-level per-agent fixed-point means equal the posterior means for every loop configuration. The 0.044 of block F would then have to be a coding error.

### D28-ITEMS-02: "Everything invisible is repaired" fails for a dissonant truth; the limit is a confident flat representative
**Category:** inflation
**Severity:** medium
**Location:** `Item_1_communication_links.tex:476` (remark after `it1:pair`) and `Item_1:22` (Status: "repairs its dictionaries through reciprocity"); corrected at `chapters/17_communication_links.tex:447`

**Evidence:** Item 1 says: "When the two row spaces are independent and together have dimension K, everything invisible is repaired and no misunderstanding is detectable: complete repair costs complete trust."

By `it1:learning`(v) and `it1:complementary`(ii), Bω_∞ → QBω* as λ → ∞. When the row spaces are complementary, Q = 0 and N ∩ ker B = 0. The limit mean is therefore flat and the limiting covariance on N tends to zero. If Bω* ≠ 0, that limit cannot be ω*.

Blocks E and E2 of `d28_item1_verify.py` test this with K = 3, code e_1^T on edge 1→0, and rows e_2^T, e_3^T on edge 0→1:
- For a flat truth, |ω_∞ − ω*| = 4.6e-7 at λ = 1e7, as the remark says.
- For a dissonant truth, |ω_∞ − ω*| = 2.255, which equals |Bω*|, while |Bω_∞| = 2.4e-7.
- In the dissonant case the largest limiting invisible variance is 1.0e-7, and the squared Mahalanobis distance of the truth under the limit law is 1.9e8.

The recognition law therefore becomes certain of invisible coordinates that are not the true ones. The closing clause "complete repair costs complete trust" admits a charitable reading, but "everything invisible is repaired" asserts recovery, and the Status sentence uses "repairs" without qualification.

**Fix:** Use chapter 17's wording: "the invisible flat-truth coordinates are recovered as the loop weight grows, while every tangent loop discrepancy can also be absorbed by an invisible change". In the Status, replace "repairs its dictionaries" with "completes its dictionaries toward consonance, recovering them when the truth is flat".

**Falsifies:** A dissonant ω* in the complementary case with ω_∞ → ω*. This is impossible, because Bω_∞ → 0 ≠ Bω*.

### D28-ITEMS-03: Item 1 transfers the bandwidth Gibbs costs to the reciprocal potential without the registration hypothesis
**Category:** inflation
**Severity:** medium
**Location:** `Item_1_communication_links.tex:304` (after `it1:bandwidth`); corrected at `chapters/17_communication_links.tex:275`

**Evidence:** Item 1 says: "The law-level peer costs of the main manuscript's reciprocal potential therefore arise in the limit of many messages through an injective, noiseless code".

The two costs have different first arguments:
- The limit in `it1:bandwidth` is E_ν[log ν_j − log ν_k] = KL(ν‖ν_k) − KL(ν‖ν_j). Its first argument ν is the law of the messages actually heard.
- The reciprocal peer cost (`eq:dyn-peer-costs`, `03_dynamics.tex:30-36`) is D_ij = KL(q_i‖ν_ij). Its first argument is the receiver's own law.

An injective noiseless code preserves relative entropy, so the bandwidth cost becomes KL(law heard ‖ transported q_j), not KL(q_i‖·). The two coincide only when the law heard equals q_i. That is the registration event of `thm:dyn-registered` (`03_dynamics.tex:391-406`), not a consequence of the strong law of large numbers. The bandwidth posterior is also a normalized Bayesian source model, so the sentence carries a conclusion from one objective to another without proof, against the three-objective rule. The finite-n clause, which cites the registered-agreement theorem, is correct.

**Fix:** Use chapter 17's sentence, or write: "Under type registration of the receiver's own law (`thm:dyn-registered`) these costs are the reciprocal peer costs; without registration their first argument is the law of what was heard."

**Falsifies:** A derivation that KL(q_i‖ν_ij) is the n → ∞ log-odds rate when messages come from a source whose law differs from q_i.

### D28-ITEMS-04: Item 1 keeps ten passages that chapter 17 corrected, the integration record says it received the same corrections, and the two documents use the same labels for different statements
**Category:** consistency
**Severity:** medium
**Location:** `Item_1_communication_links.tex:62, 65, 257, 286, 304, 309, 425, 476, 512, 529`; `VFE_4.0/docs/integration/2026-10-26-items1-3/implementation-record.md` (second paragraph: "The original Item 1–3 working documents receive the same corrections"); `README.md:22` ("The standalone working documents retain their earlier scope")

**Evidence:** The sentence diff `scripts/D28-ITEMS/diff_item1_vs_ch17.txt` shows these chapter-17 corrections absent from Item 1:

| Correction in chapter 17 | ch17 line | Uncorrected Item 1 line |
| :-- | :-- | :-- |
| geometric reversal versus independent opposite edges | 47 | 62 |
| "need not exist" | 50 | 65 |
| attention remark | 220 | 257 |
| plaquette as a declared elementary cycle | 249 | 286 |
| bandwidth objective scope | 275 | 304 |
| identity prior as a declared alignment hypothesis | 280 | 309 |
| "select a flat representative" instead of "reports consonance" | 396 | 425 |
| pair recovery | 447 | 476 |
| variational restriction for ownership | 483 | 512 |
| free-energy open problem | 495 | 529 |

Item 1's own Status claims only "rank, compatibility, integrability, and observational-scope corrections", which is accurate. The integration record's sentence is not.

The other two documents are in a different state:
- `diff_item3_vs_ch19.txt` shows that Item 3's body is identical to chapter 19 apart from headings.
- Item 2's body is identical to ch18:1-302 apart from one sentence (D28-ITEMS-08) and notation.
- README's "retain their earlier scope" is therefore wrong for Item 3, whose earlier hierarchy profiles, tower and fluctuation claims were removed.

The documents also share labels (`it1:*`, `it2:*`, `it3:*`; `label_map_output.txt`). All 33 Item 1 labels recur in chapter 17, but `it1:owned`, `it1:heads` and `it1:dissonance` carry different content in the two places. `it2:open-reciprocal` names different problems in Item 2 and in chapter 18. A verified ledger keyed by label would conflate them.

**Fix:**
- In Item 1, either add a "Superseded statements" list that points to the chapter-17 replacements, or apply them.
- Correct the integration-record sentence.
- Reword README, for example: "Item 1 retains its numerical-check narrative and some superseded remarks; Items 2 and 3 now match chapters 18 and 19."
- Before a ledger is built, prefix the working-document labels (for example `w1:`), or record that the chapter is authoritative for any shared label.

**Falsifies:** An Item 1 version later than Research commit `690ad4c` that carries the chapter-17 wording.

### D28-ITEMS-05: The law-level per-agent covariance update has exactly one fixed point; this supplies the existence that `it2:per-agent-flow` presupposes and settles part of `it2:open-convergence`
**Category:** extension
**Severity:** medium
**Location:**
- `Item_2_link_dynamics.tex:144` and `chapters/18_link_dynamics.tex:128` (`it2:per-agent-flow`: "Hold the peers' covariances at a fixed point of the per-agent covariance updates")
- `Item_2:288` and `ch18:632` (`it2:open-convergence`: "Determine whether the law-level covariance updates have a unique fixed point")
- `Item_2:207` and `ch18:191` (`it2:echo`(iii): "at its symmetric covariance fixed point")

**Evidence:** Neither document proves that this fixed point exists. A proof of existence and uniqueness follows.

Setup. Write Φ_i(S) = A_i(S)^{-1} with A_i = F_i + G_i(S), where:
- F_i = diag(P_i, σ^{-2}I + G_e) is fixed and positive definite;
- G_i(S) = Σ_e L_e^T(R_e + T_eΣ_jT_e^T)^{-1}L_e + Σ_γ B^T((λW_γ)^{-1} + Σ_f V_f)^{-1}B;
- Σ_j and V_f are principal blocks of the peers' covariances.

Steps.
- (a) Φ is order preserving: if S ⪯ S', then G_i(S) ⪰ G_i(S'), so Φ(S) ⪯ Φ(S').
- (b) Φ is strictly subhomogeneous. For c > 1, R + cTΣT^T ⪯ c(R + TΣT^T), so G_i(cS) ⪰ c^{-1}G_i(S). Hence A_i(cS) ⪰ c^{-1}A_i(S) + (1 − c^{-1})F_i ≻ c^{-1}A_i(S), that is, Φ(cS) ≺ cΦ(S).
- (c) Uniqueness. Let S and S' be fixed points and t* = inf{t : S' ⪯ tS}. If t* > 1, then S' = Φ(S') ⪯ Φ(t*S) ≺ t*S, which contradicts the minimality of t*. So S' ⪯ S, and by symmetry S = S'.
- (d) Existence. Since 0 ⪯ Φ(0) and Φ(S) ⪯ F^{-1} for every S, the iteration from 0 increases monotonically to a fixed point. The iteration from F^{-1} decreases to the same point.

Numerical support: `scripts/D28-ITEMS/d28_item2_covfixed.py` ran 12 random complete exchanges (K = 2, codes of rank 1 or 2, λ between 0.5 and 5, grounding on and off).
- The increasing and decreasing iterations agreed to at most 1.1e-14 within 37 iterations.
- The worst monotonicity value was −5e-16.
- The smallest strict-subhomogeneity margin, the least eigenvalue of 2Φ(S) − Φ(2S), was 0.032.

In the echo model every fixed point is therefore symmetric. I also checked this from 200 random asymmetric starts for ℓ = 2, 3 and 5; the spread of the limits was at most 4e-14 (`d28_item2_verify.py` block C).

This is the standard argument for order-preserving, strictly subhomogeneous cone maps (associated with Krause and with Nussbaum). I did not consult those sources here, and the proof above is self-contained. It applies to the Gaussian tangent family only.

**Fix:** Add this as a proposition before `it2:per-agent-flow` in chapter 18, and in Item 2 if Item 2 is kept current. Replace "a fixed point" with "the unique fixed point, reached monotonically from zero". Restrict `it2:open-convergence` to the mean flow and to non-Gaussian laws.

**Falsifies:** A tangent configuration with two distinct covariance fixed points, which would contradict step (c).

### D28-ITEMS-06: The general-prior limit of tangent learning is a disintegration statement, so part of `it1:open-learning` reduces to standard arguments
**Category:** improvement
**Severity:** medium
**Location:**
- `Item_1_communication_links.tex:311` and `chapters/17_communication_links.tex:282` (the Gaussian-scope split of the tangent model)
- `it1:learning`(i)-(ii) (`Item_1:352-359`, `ch17:323-330`)
- `it1:open-learning`, parts First and Fourth (`Item_1:518, 523`; `ch17:489`)

**Evidence:** `it1:learning`(i) holds for every noise law: the likelihood depends on ω only through α = Y^Tω. From this:
- (A) For any prior Π with density π, Bayes' rule makes the posterior conditional law of z = Z^Tω given α equal the prior conditional, for every sample.
- (B) Suppose the posterior of α is consistent at α* = Y^Tω*, and α ↦ π(·|α) is weakly continuous at α*. Then the posterior converges weakly to δ_{α*} ⊗ π(·|α*), which is the prior conditioned on the coset ω* + N. Consistency should hold for the identifiable linear location-regression model under the regularity of `it1:location`, by standard posterior-consistency arguments, but I did not prove it. The Gaussian formula `it1:limit-mean` is the instance.
- (C) For loop priors π_λ ∝ exp(−U(ω) − λΦ(Bω)), with U continuous, Φ ≥ 0 continuous, and Φ(v) = 0 exactly when v = 0: assuming no minimizing sequence escapes to infinity, the Laplace principle concentrates the conditional at α* on argmin_z Φ(B(ω* + Zz)) as λ → ∞. For a flat truth that set is z* + Z^T(N ∩ ker B), so the filled directions are recovered exactly. This answers part Fourth in the tangent model.
- For the nonlinear grounded model, identifiability (i) makes the likelihood depend on θ only through Tθ^{-1}. The map θ ↦ Tθ^{-1} is a submersion of GL(K) onto the full-row-rank matrices, so the coarea formula gives the conditional law on the fiber θ* Stab(T). Part First then reduces to consistency plus the shape of π on that fiber.

AUTHORING.md:29 asks for this ordering: "Where a general-law statement is available, such as a chain-rule, data-processing, or disintegration identity, state it first". Item 1 instead lists "The limiting law as a conditioned Gaussian" among the Gaussian features and leaves the general version open.

**Fix:** State (A) and (B) before the Gaussian formulas in chapter 17 and in Item 1. Narrow `it1:open-learning` to the consistency conditions on the nonlinear fiber and to the selection that π makes on it. Difficulty: low for (A), moderate for (B) and (C).

**Falsifies:** A noise law for which the conditional law of Z^Tω given Y^Tω changes with the data. Step (A) excludes this.

### D28-ITEMS-07: Item 1 says no invertible dictionary exists between agents whose representations differ
**Category:** error
**Severity:** low
**Location:** `Item_1_communication_links.tex:65`, `it1:why-group`; corrected at `chapters/17_communication_links.tex:50`

**Evidence:** Item 1 says: "For agents of different kinds, whose internal spaces differ in dimension or representation, an invertible dictionary does not exist." When the dimensions are equal and the representations are inequivalent, invertible linear maps between the spaces do exist. What can fail is a group-valued dictionary or an invertible intertwiner. Chapter 17 says "need not exist".

**Fix:** "For unequal dimensions no invertible linear map exists; for inequivalent representations of equal dimension, a group-valued dictionary or invertible intertwiner need not exist."

**Falsifies:** A cross-kind definition in Item 1 that requires equivariance. I found none.

### D28-ITEMS-08: Item 2's Gibbs-row sentence needs a normalized, entropy-regularized row objective that Item 2 never defines
**Category:** gap
**Severity:** low
**Location:** `Item_2_link_dynamics.tex:242` (remark after `it2:potential`); corrected at `chapters/18_link_dynamics.tex:212-249` (`it2:law-potential`, `it2:row-gap`) and `ch18:265`

**Evidence:** Item 2's potential (`it2:potential-eq`, `Item_2:236-240`) has fixed positive rows and no row entropy. Its sentence "With Gibbs rows in place of fixed rows, the Gibbs envelope identity gives the same derivatives with the rows evaluated at their Gibbs values" holds for the reduced potential −T log Σ π e^{−C/T} of `prop:dyn-gibbs` (`03_dynamics.tex:61`). It does not hold for Σ_e β*_e(x)C_e(x).

Block B of `d28_item2_verify.py` compares the three derivatives:
- the fixed-row derivative evaluated at the Gibbs rows: (−0.1199, −0.4278);
- the derivative of −T log Z: equal to the first to 1.4e-10;
- the derivative of the entropy-free sum with Gibbs rows substituted: (−0.5666, −0.2082), which differs by 0.447.

Chapter 18 defines E_T and states "Substitution of β*(x) into the entropy-free weighted sum does not give this derivative."

**Fix:** In Item 2, define E_T = E + Σ_i T_i KL(β_i‖π_i) with normalized rows, or cite `it2:row-gap`.

**Falsifies:** An explicit row-entropy term in Item 2. I found none.

### D28-ITEMS-09: `it1:frustration` omits the strong connectivity its equivalence needs
**Category:** gap
**Severity:** low
**Location:** `Item_1_communication_links.tex:277-281`; `chapters/17_communication_links.tex:240-244`

**Evidence:** The corollary states no graph hypothesis, and "the holonomies" has no base point. Block H of `d28_item1_verify.py` gives a counterexample on a weakly connected graph:
- component A = {0, 1}, whose two-cycle holonomy is a rotation by 0.7;
- component B = {2, 3}, whose two-cycle holonomy is the identity;
- a single edge 2 → 0 joining them.

The consensus energy has least eigenvalue 0.056 > 0, yet the holonomies based at vertex 2 are {I}, with common fixed space R². Under strong connectivity the holonomy groups at different base points are conjugate, and the statement holds.

**Fix:** Add "Let Γ be strongly connected" to the corollary, in both documents.

**Falsifies:** A reading that fixes the base point i_0 of `it1:consensus` and its hypotheses. That reading repairs the corollary; the statement as written does not carry those hypotheses.

### D28-ITEMS-10: Item 1's Gaussian-scope summary drops the regularity under which the Fisher-information parts hold for location families
**Category:** inflation
**Severity:** low
**Location:** `Item_1_communication_links.tex:24` (Status); related wording at `Item_1:186` and `chapters/17_communication_links.tex:165`

**Evidence:** The Status says: "identifiability, the modality and routing corollaries, the structural parts of the learning chapter and the exact identification results hold for every location family of noise". `it1:location` proves (i) and (iii) for every noise law, but (ii) only when ν has a C^1 density with finite positive-definite Fisher information.

Uniform noise on [−1,1]^r is a location family for which the model is not differentiable in quadratic mean and no Fisher form exists. For that family, identifiability (ii), the Fisher parts of `it1:modalities` and the missing-information identity used in `it1:routed`(ii) are not available, while (i) and (iii) still hold.

Two smaller points. The Status also lists the query-map proposition, a law-free linear-algebra statement, among the general-law results. The Cramér-Rao location bound J_f ⪰ Cov(ε)^{-1} at `Item_1:186` is used without a source.

**Fix:** "... hold in their stabilizer parts for every additive noise law, and in their Fisher-information parts for location families with a C^1 density and finite positive-definite Fisher information".

**Falsifies:** A Fisher form for the uniform location family under which (ii) holds.

### D28-ITEMS-11: Item 1 calls standard attention a special case of the construction and credits it with evidence pooling
**Category:** inflation
**Severity:** low
**Location:** `Item_1_communication_links.tex:240-257`, `it1:heads`; corrected at `chapters/17_communication_links.tex:219-220`

**Evidence:** Item 1 says "Standard attention is a special case of the construction here in three respects". One of the three listed respects is "The output map is learned rather than derived from the receiver's covariance". That is a departure from the construction's derived decoder (`it1:kalman`), not a specialization of it.

Item 1 then says "so the heads already pool their evidence as in" `it1:modalities` and `it1:routed`. Those corollaries are Fisher-information statements about a likelihood model in which one dictionary generates the messages. Standard attention has no such likelihood, so the conclusion does not transfer.

The linear-algebra part is correct: stacked invertible query and key maps determine a unique θ with W_K θ^{-1} = W_Q.

**Fix:** Use chapter 17's wording.

**Falsifies:** A likelihood for standard attention under which the query and key maps are a receiver reading and a sender code of one dictionary.

### D28-ITEMS-12: Item 1's summary says nontrivial holonomy obstructs agreement, but the consensus theorem gives agreement on its fixed space and for invariant laws
**Category:** inflation
**Severity:** low
**Location:** `Item_1_communication_links.tex:22` (Status), against `it1:consensus` (`Item_1:270`) and `Item_1:275`

**Evidence:** The Status says "so nontrivial holonomy on a loop is an obstruction to agreement, a dissonance". By `it1:consensus`, consensus means form ∩ Fix(H_γ), which is nonzero for a nontrivial rotation of R^3 about an axis. Consensus laws are the holonomy-invariant laws, which include the centered isotropic Gaussian under rotation holonomies; `Item_1:275` says so itself. Only agreement on all contents is obstructed.

**Fix:** "nontrivial holonomy restricts agreement to the contents and laws it fixes".

**Falsifies:** Not applicable. The finding follows directly from the theorem.

### D28-ITEMS-13: Two magnitudes quoted in Item 1 are not reproduced by the committed check script
**Category:** consistency
**Severity:** low
**Location:** `Item_1_communication_links.tex:494` (after `it1:exact-pair`) and `Item_1:507` (after `it1:exact-flat`); `item1_checks/checks.py:388-417` (blocks 9h and 9i)

**Evidence:** The quoted magnitudes and the values from my rerun (`rerun_item1_checks_output.txt`, identical to the recorded output) are:

| Quantity | Item 1 says | Committed script gives |
| :-- | :-- | :-- |
| exact pair: message-law residual | 7×10^{-16} | 1.1e-16 and 4.4e-16 |
| exact pair: reciprocity residual | 4×10^{-16} | 1.1e-16 |
| exact pair: largest difference from the truth | 1.86 | 0.261 |
| exact flat: message-law residual | 7×10^{-15} | 6.7e-16 |
| exact flat: holonomy residual | 4×10^{-15} | 6.9e-16 |
| exact flat: largest difference from the truth | 3.7 | 1.74 |

Read-only `git show` of the Research history shows the recorded output has carried 0.261 and 1.74 since commit `c9f6861` (2026-10-22), so the two quoted differences have never matched a committed run. The qualitative claim, a residual set that differs from the truth, is supported; the stated magnitudes are not.

**Fix:** Quote the recorded values, or commit the script variant that produced 1.86 and 3.7.

**Falsifies:** A committed script version whose output is 1.86 and 3.7.

### D28-ITEMS-14: The check scripts predate the corrections; they reproduce their outputs but do not test the corrected claims, and Items 2 and 3 no longer cite them
**Category:** consistency
**Severity:** low
**Location:** `item1_checks/`, `item2_checks/`, `item3_checks/`. All six files are byte-identical to the copies in the 2026-10-23 `source-snapshot/`.

**Evidence:** All three scripts ran under the cap, one at a time (OMP, MKL, OPENBLAS and NUMEXPR threads set to 1, `timeout 30`, Python 3.14.4, numpy 2.4.4, scipy 1.17.1). Each exited 0 and reproduced its recorded output byte for byte.

How the scripts relate to the current documents:
- Item 1 cites them for 14 numerical statements. Twelve match; the remaining two are D28-ITEMS-13.
- Items 2 and 3 cite no checks. Item 2 blocks (9) and (11) and Item 3 block (4) test claims that were removed at integration: filling-report variances, stability over 40 random exchanges, and wall-concentration and meta-level-dictionary profiles.
- Item 1 block (9j) tests the reverse-KL factorization, not the per-agent picture that `it1:owned` names.

No script tests any of the following:
- `it1:stack-compatibility` (G5) and `it1:bandwidth` (G6);
- `it1:complementary`(ii) with a nonzero detectable space. Item 1 block (9e) uses the complete exchange, where Q = 0;
- the ring at c = 0 (V4) and the Gibbs-row normalization;
- the DAG residual (G1), the flat-wall counterexample (G2) and the normalizer counterexample (G3);
- `it3:observation`, `it3:cap`, and the reciprocal replacement in `it3:rank-one`.

The D28 scripts cover each of these items (`d28_item1_verify.py` blocks C, D, G, H; `d28_item2_verify.py` blocks B, D; `d28_item3_verify.py` blocks A, B, D, E, F, G).

**Fix:** Add these tests to the item check scripts, and state in each script's docstring which document version it supports.

**Falsifies:** Test blocks that I missed. I read all three scripts in full.

### D28-ITEMS-15: Item 2's status line, one open problem and its notation are stale relative to chapter 18
**Category:** consistency
**Severity:** low
**Location:** `Item_2_link_dynamics.tex:16` (status), `Item_2:284-285` (`it2:open-reciprocal`), `Item_2:3` (subject line "global and per-agent free energies" and the title "Dictionaries in the Two Free Energies"), and the symbol r throughout

**Evidence:**
- The status line says "General-law extensions and the joint mean, covariance, and row dynamics remain open". But ch18:304-624 gives the coupled Gaussian Fisher equations, global existence, a three-agent threshold and the local pitchfork.
- `it2:open-reciprocal` has a different title and content in ch18:628.
- Item 2 uses r for four things: the code width r_e (:31), the residual (:99), the dictionary law (:229), and the dimension in the Gaussian KL formula (:253, "−r"). Chapter 18 renames the width to n_e but keeps r for the other three, including the generic KL formula at ch18:276.

**Fix:** Mark the joint Gaussian dynamics as treated in chapter 18, and rename the symbol for the KL dimension.

**Falsifies:** Not applicable; the stale wording is visible directly.

### D28-ITEMS-16: A plaquette defined as "a shortest cycle", and class-function defect labels, do not control dissonance on directed graphs or for unipotent loops
**Category:** weakness
**Severity:** low
**Location:** `Item_1_communication_links.tex:286`, `it1:dissonance`; chapter 17 corrected the plaquette sentence and shares the second sentence (`chapters/17_communication_links.tex:249`)

**Evidence:**
- Plaquettes. In any graph with two-cycles, every shortest directed cycle is a two-cycle. In the complete exchange, "all plaquettes consonant" therefore leaves the directed triangles unconstrained, while exact flat identification needs the triangles (`Item_1:502`). Chapter 17 uses "a declared elementary cycle" instead.
- Defect labels. Both documents say "A quantitative label is any conjugation-invariant defect of H_γ". A continuous conjugation-invariant function of H alone takes its identity value on the nonidentity unipotent I + E_12 (`02_interactions.tex:233`), so it cannot vanish only at consonance. The dressed defect is invariant under joint conjugation of (H, Σ); it is not a function of H alone. Both documents go on to cite the dressed defect, so the remedy is present, but the general clause is wrong for GL dictionaries.

**Fix:** Write "any defect invariant under the joint action on the holonomy and the base covariance", and add that continuous class functions of H alone cannot separate unipotent holonomy from consonance.

**Falsifies:** A restriction of the label to orthogonal dictionaries. The sentence states that restriction only for its example.

### D28-ITEMS-17: Item 1 presents Gaussian features and modeling choices as the formal content of the general picture
**Category:** exposition
**Severity:** low
**Location:**
- `Item_1_communication_links.tex:122` ("The rank bound makes precise the loss in the motivating picture"; also `ch17:101`)
- `Item_1:146` ("This is the formal content of the statement that the receiver internalizes the message"; also `ch17:125`)
- `Item_1:178` ("This is the precise form of the statement that words cannot carry what a belief means to its holder"; removed in ch17)
- `Item_1:309` ("The identity is the natural prior mean"; ch17:280 calls it a declared hypothesis)

**Evidence:** `Item_1:124` itself classifies the rank bound as a Gaussian feature. For non-Gaussian priors the posterior covariance can change inside ker A. For example, take K = 2, A = (1, 0), ȳ_1 ~ N(0,1) and ȳ_2 | ȳ_1 ~ N(0, 1 + ȳ_1²). Then Var(ȳ_2 | h) = 1 + E[ȳ_1² | h], which depends on the message h. The general-law content is the reconstruction `it1:reconstruction`: the receiver keeps its own conditional law given the coded statistic. The Gaussian projector is the instance. AUTHORING.md:29 states that Gaussian features may not be presented as properties of the theory.

**Fix:** State the general conditional-law fact first (see Extensions), and call the rank bound and the lift direction Σ_pA^T its Gaussian instance.

**Falsifies:** A reading of "the motivating picture" as itself Gaussian. The Status at `Item_1:15` presents it as general.

### D28-ITEMS-18: In `it2:split`(ii) the per-agent covariance S' equals S, so the receiver's single-record update coincides with the global one
**Category:** exposition
**Severity:** low
**Location:** `Item_2_link_dynamics.tex:110`; `chapters/18_link_dynamics.tex:94`

**Evidence:** S' = T(Σ_i⁰ + C_ρ)T^T + R + TΣ_j⁰T^T = T(Σ_i⁰ + Σ_j⁰ + C_ρ)T^T + R = S. sympy gives S' − S = 0 (`d28_item2_verify.py` block A). Hence δy_i and κ_iδω_e coincide with part (i); the only difference is that the sender does not move. A separate symbol S' suggests that the receiver's update differs.

**Fix:** Write S in (ii) and add "the receiver's shifts coincide with (i)".

**Falsifies:** Not applicable; the identity is exact algebra.

### D28-ITEMS-19: The rank in `it3:induced` needs neither the cycle-spanning hypothesis nor a common operating content
**Category:** improvement
**Severity:** low
**Location:** `Item_3_curvature_hierarchy.tex:103-105` and `chapters/19_curvature_hierarchy.tex:91-93` (second paragraph of `it3:induced`); proof at `Item_3:117` and `ch19:105`

**Evidence:** Π_H has kernel im δ, and im δ ⊆ im V = ⊕_e im T_e whenever every receiver has c_i ≠ 0. Therefore rank V^TΠ_HV = Σ_e rank T_e − K|V| + dim H^0. For injective codes on a connected graph this is K(|E| − |V| + 1); for full-row-rank codes it is dim H^1 of the code sheaf. Only the loop-sum characterization needs a common c and cycle spanning.

`d28_item3_verify.py` checks this in three settings:
- Block A, the DAG: rank 1, which equals K(|E| − |V| + 1).
- Block B, a graph whose single directed cycle does not span the cycle space: rank 4, which equals the formula, while zero directed-loop sums still allow a normalized penalty of 24.3. This is a second G1 counterexample, beyond the acyclic one.
- Block H, non-injective codes with receiver-specific c_i: in four trials the rank equals Σ rank T_e − K|V| + dim H^0.

**Fix:** Split the paragraph. State the rank under c_i ≠ 0, using the general formula or its injective special case. State the loop-sum characterization under a common c and cycle spanning.

**Falsifies:** A configuration with c_i ≠ 0 where the rank departs from the formula. The kernel argument above excludes it.

### D28-ITEMS-20: The dropped Weiss-Freeman sentence gives exact cavity means for overlapping loop factors
**Category:** improvement
**Severity:** low
**Location:** `Item_2_link_dynamics.tex:219` and `chapters/18_link_dynamics.tex:203` ("it is not a general exactness theorem for overlapping loop factors"). The 2026-10-23 snapshot of Item 2 cited `WeissFreeman1999` here; the entry is still in `item2_refs.bib` but is no longer cited.

**Evidence:** For Gaussian belief propagation on loopy graphs, Weiss and Freeman showed that at a fixed point the means are exact while the variances generally are not. On overlapping loops, the cavity construction of `it2:echo`(iv) is Gaussian belief propagation on the factor graph of anchors, grounded terms and loop factors. Its means at convergence are therefore the posterior means; only the exactness of the variances is specific to the single loop. I did not reread Weiss and Freeman for this review.

**Fix:** Restore one sentence with the citation.

**Falsifies:** A version of the cavity construction that differs from factor-graph Gaussian belief propagation.

## Checked without defect

Results reconstructed or recomputed and found correct. Where a statement appears in both a working document and a chapter with identical text, one row covers both. Script references are to `scripts/D28-ITEMS/`, and "recorded (x)" refers to the item check outputs, which my rerun reproduced byte for byte.

| Label | What was checked | Method | Residual or reasoning |
| :-- | :-- | :-- | :-- |
| `it1:loss` | the identity and its chain-rule proof; the chapter-17 corollary form | reconstructed; finite alphabet with a non-injective code (`d28_item1_verify` A); recorded (1e) | 1.7e-16; −4.4e-16 |
| `it1:noisy` | the bound; integral joint convexity through data processing on (h, ȳ); the Gaussian mean and covariance formula | reconstructed; 2000 random finite-alphabet instances (B) | minimum slack 5.1e-4 ≥ 0 |
| `it1:kalman` | posterior, Woodbury precision, mutual information | reconstructed; recorded (2b), (2c) | 3.3e-15; 4.4e-16 |
| `it1:gaussian-reconstruction`, `it1:mean-update` | AG = I, P² = P, PΣ_p symmetric, covariance identity, mean update | reconstructed; recorded (1a)-(1d) | ≤ 7.1e-15 |
| `it1:exponential` | e^{OT} = I + Oφ_1(TO)T and e^P = I + (e − 1)P | reconstructed; recorded (3) | 8.9e-16 |
| `it1:identifiability` (i)-(iii) | characteristic-function argument for any additive noise; Fisher form and kernel; stabilizer dimension K(K − r) | reconstructed; recorded (4a)-(4c) | rank 10 = Kr at K = 5, r = 2 |
| `it1:location` | Fisher form with J_f and the same kernel; Gaussian noise minimizes J at fixed covariance | reconstructed; recorded (10) | 0.26 percent (Monte Carlo) |
| `it1:modalities` | intersection of stabilizers; additivity; equality cases split at Σr_c = K | reconstructed; recorded (5) | 15 = 5 × 3 |
| `it1:routed` | (i) weights; (ii) missing-information inequality; (iii) kernel | reconstructed; recorded (7a)-(7c) | 0.64 percent; least eigenvalue −6.8e-16, which is the common kernel |
| `it1:query-dictionary` | coset; GL^+ sign flip | reconstructed; recorded (8a), (8b), which test the coset only | ≤ 8.9e-16 |
| `it1:stack-compatibility` | left-kernel criterion, construction, coset, counterexample | reconstructed; rank-deficient compatible stack (C) | 1.8e-15; the counterexample's left kernels differ |
| `it1:consensus` | laws and means; path independence through return paths and the bimeasurable inverse | reconstructed | none |
| `it1:frustration` | kernel equals consensus; positivity under strong connectivity | reconstructed; recorded (6) | 0.054 (hypothesis omitted, D28-ITEMS-09) |
| `it1:bandwidth` | strong-law limit; KL difference under finiteness (G6 repair) | reconstructed; Monte Carlo (G) | 0.0429 against 0.0410 at n = 2e5, within Monte Carlo error |
| `it1:tangent-model` | Λ_0 ≻ 0; second-order linearization errors; ‖h^{-1}Xh‖²_{h^{-1}Σh^{-T}} = ‖X‖²_Σ; no conjugation-invariant positive-definite form on gl(K) | trace identity by hand; recorded (9k) | ratio 2.30 |
| `it1:learning` (i)-(v) | block inversion; limit mean (I − Π_N)ω*; covariance; (iii), including equality with the prior covariance on ker L_N; (iv); (v) | reconstructed; recorded (9b)-(9e); (iii) in block D | 5.6e-17; rank L_N = 6 = K dim(K_0 + K_1) |
| `it1:complementary` (i)-(iii) | W-orthogonal projection; Bω_∞ → QBω* with Q ≠ 0 | reconstructed; block D | 3.9e-6 at λ = 1e6, decaying as 1/λ; loop energy → D(ω*) = 3.10 |
| `it1:unreachable` | cochains vanishing on cycles are coboundaries; K(dim H^0 − K) | reconstructed; recorded (9f) | counts 0, 9, 6, 9 match the text |
| `it1:complete-exchange` | spanning tree of the edges 0 → i; fundamental cycles | reconstructed; recorded (9f) | no unreachable directions |
| `it1:pair` | B(N), its complement, Q = P_∩, dimensions | reconstructed; recorded (9g) | (6,0,3) and (9,3,0); D matches to 6e-15 |
| `it1:exact-pair`, `it1:exact-flat` | coset algebra; the H^0' criterion; the complete-exchange hypothesis | reconstructed; recorded (9h), (9i) | residuals ≤ 6.9e-16 (quoted magnitudes differ, D28-ITEMS-13) |
| `it1:owned`, reverse-KL part | exact means; block-inverse covariances below the marginals | reconstructed; recorded (9j); block F | 5.8e-15 |
| `it2:model`, `it2:global`, `it2:per-agent-model` | linearization; J ≻ 0; Gaussian convolution of record and loop factor | reconstructed | none |
| `it2:gauge-general`, `it2:gauge` | GL^+ transitivity; parts (i)-(iii) | reconstructed; recorded (1a), (1b) | 2.2e-16; dimensions 24, 6, 2 |
| `it2:split`, `it2:grounding` | conditioning shares; κκ^T = |c|²I; Schur complement of order 1/g | reconstructed; sympy (`d28_item2_verify` A); recorded (2), (3) | S' − S = 0; g|Tωc| → 0.223 |
| `it2:global-flow`, `it2:per-agent-flow` | free energy and optimum; similarity; real spectrum; rows and asymmetry | reconstructed; recorded (4)-(6) | max |Im| = 0; 0.0; 2.6e-16 |
| `it2:ring` | block circulant; triangular A^{-1}C'; w; limits, including c = 0 | reconstructed; block D | ≤ 3e-10; at c = 0, w = 0.6176 for every s |
| `it2:separation` | block elimination; H_zz block diagonal; order 1/g | reconstructed; recorded (8) | errors of order 1/g |
| `it2:echo` (i)-(iv) | Sherman-Morrison; the quadratic; limits; cavity equals exact for every ℓ | sympy (C); recorded (10) | exact identities |
| `it2:potential-forces` (i)-(iv) | Gaussian KL expansion; E^S; similarity; rows | reconstructed; recorded (12), (13) | 6.9e-9 (finite differences); 2.2e-16 |
| `it2:certainty` (i)-(iii) | data independence; source-covariance derivative; implicit function theorem | reconstructed; sympy (E); recorded (14) | 4.2e-11; dv/d(d²) > 0 |
| `it3:general-induced` | Bayes formula for a jointly measurable version of the density | reconstructed | none |
| `it3:induced` | affine action; ε → 0 limit Π_H; quotient norm; loop characterization under cycle spanning; rank; anchor bound | reconstructed; recorded (1a), (1b); `d28_item3_verify` A, B, C, H | 1/3; ranks match; penalty 9.3e-15 on zero loop sums when spanning holds |
| `it3:observation` | the E_22 example; quarter turn with Q = 2 and KL = 0 | direct computation | none |
| `it3:rank-one`, `it3:dissonance` | push-through; plug-in and law-level forms; reciprocal replacement; telescoping | recorded (2), (3); block F | ≤ 2.7e-15; 4.4e-16 |
| `it3:cap` | Gaussian convolution | sympy (E) | exact |
| `it3:rigid` | factorization; normalizer hypotheses; counterexample; order 1/τ by block elimination | reconstructed; block D; recorded (5) | ∫p² diverges; 2.7e-4 and 2.7e-7 |
| `it3:flat-wall`, `it3:tower` | path hypothesis; group equivalence; residual formula; counterexample | reconstructed; block G | none |
| chapter 19 against Item 3 | identity of the bodies | sentence diff | only the headings differ |
| ch18:1-302 against Item 2 | identity apart from `it2:law-potential`, the normalized-row sentence and r_e → n_e | sentence diff | as stated |
| Gaussian-scope statements in Items 1-3 (focus question 5) | consistency with AUTHORING.md:29 | read against the rule | Items 2 and 3 are consistent; Item 1 deviates in D28-ITEMS-06, -10 and -17 |

Integration corrections (focus question 1). The 2026-10-23 audit identifiers G1-G6, V3 and V4 are from `gauge-audit.md` and `variational-audit.md`. "Present in Item" says whether the working document received the correction.

| Correction | Category | Item location | Chapter location | Present in Item | Assessment |
| :-- | :-- | :-- | :-- | :-- | :-- |
| G1: cycle-spanning hypothesis, strong connectivity sufficient, DAG example | directed graph | Item_3:103-132 | ch19:91-120 | yes | correct (A, B, C) |
| G2: internal directed paths and bridges in both directions | directed graph | Item_3:241-256 | ch19:229-244 | yes | correct (G) |
| independent opposite edges; geometric reverse uses the inverse | directed graph | Item_3:47; Item_2:31 | ch19:35; ch18:15; ch17:47 | Item 1: no | correct |
| plaquette as a declared elementary cycle | directed graph | Item_1:286 keeps the old wording | ch17:249 | no | D28-ITEMS-16 |
| unrestricted G versus the single-exponential image | identifiability | Item_1:54 | ch17:39 | yes | correct |
| part (i) for any additive noise | identifiability | Item_1:162, 175 | ch17:141, 154 | yes | correct |
| G5: modality equality cases; stack compatibility | identifiability | Item_1:197, 228-238 | ch17:176, 207-217 | yes | correct (C) |
| G4: law comparison, dressed defect, grounded and single-topic experiments | identifiability | Item_1:288, 517-523; Item_3:138-148 | ch17:251-259; ch19:126-136 | yes | correct |
| G6: bandwidth finiteness | integrability | Item_1:293-304 | ch17:264-275 | yes | correct |
| G3: normalizer, reference measure, counterexample | normalization | Item_3:204-238 | ch19:192-226 | yes | correct (D) |
| normalized rows with row entropy (E_T, `it2:row-gap`) | normalization | absent (Item_2:242) | ch18:212-249, 265 | no | D28-ITEMS-08 |
| law-level loop factor at λ = 0; product of peer marginals | normalization | Item_2:66 | ch18:50 | yes | correct |
| V3: plug-in qualification; derivatives versus velocities | objective scope | Item_2:14, 148, 155 | ch18:132, 139 | yes | correct |
| V4: ring weak-anchor limit only for c ≠ 0 | regime | Item_2:16, 164, 169 | ch18:148, 153 | yes | correct (D) |
| the reciprocal mean quadratic does not identify objectives; certainty result limited to frozen quantities | objective scope | Item_2:260, 271, 277 | ch18:283, 294, 300 | yes | correct |
| rank-one replacements do not equate forces | objective scope | Item_3:168 | ch19:156 | yes | correct (F) |
| bandwidth without transfer to the reciprocal potential; ownership as a variational restriction; free-energy open problem | objective scope | absent (Item_1:304, 512, 529) | ch17:275, 483, 495 | no | D28-ITEMS-01, -03, -04 |
| affine record term kept; Hessian not read as a preference for flatness | scope | Item_3:134-136 | ch19:122-124 | yes | correct |
| pair recovery versus detection; attention remark | scope | absent (Item_1:476, 257) | ch17:447, 220 | no | D28-ITEMS-02, -11 |
| formation-standard disclaimers | scope | Item_3:11, 263 | ch19:5, 251 | yes | consistent with `inh:recursive-formation` |

Results in the working documents but dropped from the chapters (focus question 3):
- **Item 1, kept only in the working document:**
  - the numerical remarks. They are correct, apart from D28-ITEMS-13, and reasonably omitted from a theory chapter.
  - the explicit complete-exchange counts: 9 of 36 unreachable with two-cycles only, 0 with both triangles, 6 with a common unread direction, and 9 of 18 for the directed triangle. They are correct (recorded (9f)) and would make a useful worked example in chapter 17 or its companion.
  - the related-work paragraph, `Item_1:535`. FristonFrith2015, FristonEtAl2020Language, WynerZiv1976 and Tishby1999 are appropriate prior-art pointers for receiver decoding with side information and for communication as inference; one sentence of them is worth restoring in chapter 17. I did not check what each cited source says.
  - the attention table and the "words cannot carry" sentence. They are overstated and were correctly dropped.
- **Item 2:** nothing in Item 2 is missing from chapter 18. The snapshot's Weiss-Freeman sentence is D28-ITEMS-20.
- **Item 3:** nothing in Item 3 is missing from chapter 19. The snapshot's hierarchy profiles were numerical observations, its tower and fluctuation claims were overstated, and its "Observation is optional" claim was false (G4); all were correctly dropped.

Check scripts (focus question 4). All three were run one at a time, single-threaded, with a 30-second timeout; outputs are in `scripts/D28-ITEMS/rerun_item*_checks_output.txt`.

| Script | Rerun | Same as recorded | What it supports now | Untested claims |
| :-- | :-- | :-- | :-- | :-- |
| `item1_checks/checks.py` | exit 0, under 1 s | byte identical, 48 lines | 14 statements cited in Item 1; 12 match, 2 magnitudes differ | `it1:stack-compatibility`, `it1:bandwidth`, `it1:complementary`(ii) with Q ≠ 0, the GL^+ claim, the per-agent version of `it1:owned` |
| `item2_checks/checks.py` | exit 0 | byte identical, 35 lines | not cited by the current Item 2; supports `it2:gauge`, `it2:split`, `it2:grounding`, both flows, `it2:ring` at c ≠ 0, `it2:separation`, `it2:echo`(i)-(iii), `it2:potential-forces`, `it2:certainty` | `it2:ring` at c = 0, row normalization, `it2:echo`(iv), dictionary rows of `it2:potential-forces`(iv); blocks 9 and 11 test removed claims |
| `item3_checks/checks.py` | exit 0 | byte identical, 28 lines | not cited by the current Item 3; supports `it3:induced` on a strongly connected graph, `it3:rank-one`, `it3:dissonance`, the `it3:rigid` limit | the DAG example, the `it3:flat-wall` counterexample, the normalizer counterexample, `it3:observation`, `it3:cap`, the reciprocal replacement; block 4 tests the removed wall profiles |

## Prior-review findings in scope

The 2026-10-22 review covered chapters 01 to 12, so none of its findings is located in this lens's files. The loop-sector program it recommended is what Items 1 to 3 and chapters 17 to 19 implement, so I record the status of those findings here.

| Finding | Status | Evidence |
| :-- | :-- | :-- |
| 2026-10-22 E8 | resolved within this scope | ch17:251-259, Item_1:288, Item_3:138-148 and ch19:126-136 separate law comparisons, shared-sample experiments and grounded experiments; `prop:int-loop-blindness` is at `02_interactions.tex:212` |
| 2026-10-22 E9 (with X1-GAUGE-09 and X4-PRIOR-09) | partially resolved | the defect is defined in `prop:int-loop-defect` (`02_interactions.tex:219`) and cited at ch17:249-257; it enters no objective ("It is not a term of the original potential", `02_interactions.tex:234`); nonlinear dressed-defect learning is still open (ch17:489) |
| 2026-10-22 R9 | partially resolved | ch17-18 make dictionaries latent variables with an anchored tangent loop prior, and dictionary laws enter the reciprocal potential (ch18:251-283); ch19 derives an induced evidence action; links still do no work in the formation chapters (`02_interactions.tex:234`), and persistent shared-sample records remain open |

## Extensions and improvements

**Per-agent convergence.** D28-ITEMS-05 settles the covariance half of `it2:open-convergence` in the Gaussian tangent family. The mean half then becomes a linear question. With the covariances at their unique fixed point, the per-agent mean flow is μ̇ = −A^{-1}(Hμ − b) with H = A − C. If the spectral radius of A^{-1}C is below one, every eigenvalue of A^{-1}H = I − A^{-1}C has positive real part. The fixed point is then unique and the flow converges exponentially. A computable sufficient condition is ‖A^{-1/2}CA^{-1/2}‖_2 < 1, which holds when the record and loop precisions are small relative to the self and anchor precisions. The directed ring meets it with spectral radius w < 1, which reproduces `it2:ring`'s stability. A matching negative statement would also be useful: a configuration whose spectral radius exceeds one and whose per-agent flow diverges, although the global flow converges by `it2:global-flow`. Difficulty: low for the sufficient condition, moderate for a sharp characterization.

**Nonlinear learning.** D28-ITEMS-06 turns part of `it1:open-learning` into a concrete nonlinear theorem. Assume the following:
- the hypothesis of `it1:exact-flat`, that flat configurations are pure gauge;
- a pure-gauge truth, and constant H^0 formed with the codes T'_e;
- location noise with finite Fisher information;
- a prior density π_λ(θ) ∝ exp(−U(θ) − λΣ_γΦ(H_γ)), with U continuous and coercive and Φ ≥ 0 continuous, vanishing only at I. The dressed defect `it1:dressed-defect` is one choice of Φ;
- a no-escape condition: inf Φ > 0 on the fiber outside every neighborhood of θ*.

Under these assumptions, the n → ∞ posterior should be the prior disintegrated on the fiber θ* Stab(T). As λ → ∞, the Laplace principle should then concentrate it at θ*, because by `it1:exact-flat` the fiber's flat set is {θ*}. This would be the nonlinear counterpart of the last sentence of `it1:complete-exchange`, and it would make `it1:exact-flat` the λ = ∞ support of the learning problem. Difficulty: moderate. The consistency step and the no-escape condition are the work.

**Cohomological bridge.** D28-ITEMS-19 links Item 1 and Item 3 through one sheaf. For full-row-rank codes and nonzero operating contents, rank(V^TΠ_HV) = Σ_e dim H_e − K|V| + dim H^0 = dim H^1 of the code sheaf, the same sheaf whose H^0 counts Item 1's unreachable directions through K(dim H^0 − K). The induced curvature penalty of `it3:induced` therefore has rank exactly dim H^1, and the Euler characteristic dim H^0 − dim H^1 = K|V| − Σ_e dim H_e ties the two counts together. Stating this in chapter 19 would make the code sheaf the common object of chapters 17 and 19. Difficulty: low.

**Conditional reception for general laws.** For any prior p on R^K and any message kernel whose density depends on ȳ only through Aȳ, the posterior is p(ȳ)ℓ(h | Aȳ)/Z. Its conditional law given Aȳ is therefore the prior's, and only the law of the r-dimensional statistic is updated. This is the general-law statement behind `it1:kalman`'s rank bound, and it is the statement AUTHORING.md asks to see first. The covariance can still change in more than r directions, as the example in D28-ITEMS-17 shows. Difficulty: trivial.

**Tests.** The tests listed in D28-ITEMS-14 already exist in the D28 scripts and can be copied into the item check scripts, with a docstring line naming the document version each script supports.

## Not checked

- The coupled sections ch18:304-624 (`it2:coupled-flow`, `it2:coupled-global`, `it2:coupled-critical-structure`, `it2:coupled-threshold`, `it2:critical-pitchfork`) were read but not verified. In particular I did not check the rational constants T_c = 3269/10118, T_mean = 31/122, c or D. They have no counterpart in Item 2 and fall outside the comparison this lens was assigned.
- The companion chapters c17-c19 and the Item PDFs. They are outside the assigned scope, and I did not compare the PDFs with the TeX sources.
- Whether the related-work citations of Item 1 support what they are cited for. I looked only at the bibliographic entries; the existence of arXiv:2512.09831 and the volume and year of LahavNeemeh2022 were not verified.
- The content of the chapter-17 cross-references `prop:conditional-lift`, `cor:pythagorean` and `sec:generative-models`.
- The posterior-consistency step (B) and the Laplace step (C) of D28-ITEMS-06. They are sketched, not proved.
- The Monte Carlo checks were run only at their committed seeds.
- The snapshot Items 1 and 2 were compared through diffs rather than reread line by line. The MultiAgentELBO material cited by the snapshot Item 3 was not checked.
- Item 7, which is outside the assigned scope.
