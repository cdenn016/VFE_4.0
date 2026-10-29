# D08-IMEM2: Interacting conditional-prediction agents: binary exchange, Gaussian closure and counterexamples

I read in full `appendices/interacting_memory.tex` lines 1-664: lines 325-664 line by line as the assigned scope, and lines 1-324 for the definitions and the results the scope invokes (imem:protocol, imem:history-theorem, imem:sender-receiver, imem:reduction, imem:free-energy, imem:reduced-free-energy). I also read in full `companion/c04_interacting_memory.tex` lines 1-233 and its lead-in `companion/c04_communication.tex:55-68`. The cross-referenced results I read were `chapters/05_inheritance.tex:22-103` (inh:two-retentions, inh:deterministic-descent), `appendices/conditional_memory.tex:80-160` (cmem:gaussian-posterior, cmem:projection) and `appendices/record_process.tex:1-60` (def:registered-event). For focus question 5 I read the scope sentences in `frontmatter.tex:7`, `README.md:20`, `AUTHORING.md:9`, `chapters/04_prediction.tex:73-86` and `chapters/06_outlook.tex:8`. Of the research records I read `finite-example.md` and `gaussian-and-retention.md` in full, and `check_interaction.py` (head) and `interaction-checks.json` only after my own computations. I skimmed the 2026-10-22 review only to confirm that none of its findings falls in this scope.

**Verdict:** no stated result in lines 325-664 or in the companion section is false. All 63 exact checks pass: every displayed number of the two-way binary exchange and of the parity example, in rational and symbolic arithmetic. The scalar Gaussian formulas hold identically in sympy, the repeated-closure recursion equals generic Gaussian conditioning in exact rational arithmetic, and the nonclosure proof is complete for every received h ≠ 0. The defects are:

- a missing quantifier (h = 0);
- a counterexample that rests on a public record outside the declared information structure;
- a closure theorem and a nonclosure proposition stated for different objects;
- missing classical attribution;
- notation.

The most consequential result of this lens is an extension that is not in the manuscript. Retention of complete local beliefs already fails almost surely inside the declared protocol and inside the affine Gaussian closure family when the two cue dimensions differ, with no public record. For scalar cues it is exact.

Scripts and outputs are in `C:\Users\chris and christine\Desktop\VFE_4.0\docs\reviews\2026-10-29-magent-theory-2.0-review\scripts\D08-IMEM2\`. Each was run single-threaded with `timeout 30` under bare `python`, one at a time:

| Script | What it does | Result |
| :-- | :-- | :-- |
| `binary_exchange.py` | exact Fractions; implements the protocol from imem:message-kernel and imem:history-law, not from imem:reduction | 47/47 checks pass |
| `parity_counterexample.py` | sympy, exact in the parity probability ρ | 16/16 pass |
| `gaussian_scalar_symbolic.py` | sympy; scalar formulas against generic Gaussian conditioning | every residual 0 |
| `gaussian_recursion_exact.py` | sympy rationals; closure recursion and the Gaussian local-retention collision | see findings |
| `mixture_nonclosure.py` | scipy quadrature | numerical |
| `gaussian_receiver_and_projection.py` | scipy quadrature | numerical |
| `gaussian_long_run.py` | float64 probe | numerical |

Each script's output is in the `.out` file beside it.

## Findings

### D08-IMEM2-01: Local-only retention fails inside the declared protocol and the affine Gaussian closure family when d_0 ≠ d_1; for scalar cues it is exact
**Category:** extension
**Severity:** medium
**Location:** appendices/interacting_memory.tex:617-653 (imem:local-retention-failure) and :560-580 (imem:gaussian-closure); companion/c04_interacting_memory.tex:213-225; summarized at chapters/04_prediction.tex:86 and chapters/06_outlook.tex:8
**Evidence:** The proposition's collision needs a public parity record that the declared information fields do not contain (D08-IMEM2-05). A collision without any public record exists in the singleton-presentation affine Gaussian family of imem:gaussian-closure.

The mechanism starts with the covariance. In the reduced recursion, A_t and R_t depend on λ_t only through S_t, so S_{t+1}^{-1} = S_t^{-1} + A_t^T R_t^{-1} A_t makes S_t a deterministic function of the step. This is verified exactly: two different message histories give identical S_1, S_2, S_3.

Each agent's complete local law at its actual private value is Gaussian with a fixed covariance. Its mean is affine in m_t through E[X_i | D] = α_i + β_i D_i + Γ_i D_{1-i}, and the Y_i-rows are F_i times the X_i-rows. Acting on (m^0, m^1), the pair of local laws is therefore T_t = M_t m_t + c_t with

M_t = [[−Γ_0 S_10 S_00^{-1}, Γ_0], [Γ_1, −Γ_1 S_01 S_11^{-1}]],

so rank M_t ≤ 2 min(d_0, d_1) < d_0 + d_1 whenever d_0 ≠ d_1.

Part B of `gaussian_recursion_exact.py` computes in exact rationals with:

- d = (1, 2), e = (1, 2), schedule (1, 0, 1);
- all noises positive definite, joint cue covariance positive definite.

It gives:

- rank M_2 = 2 < 3, with kernel dm = (0, −22/43, 1);
- m_2 has a nondegenerate law (rank Cov(m_2) = rank(Cov D − S_2) = 3);
- the histories h = ((1/2, −1), 2) and h' = h + (−73669/29541, −7040/9847, 0) give identical complete local laws (means and covariances) for both agents at t = 2. This equality is computed directly from the generic joint Gaussian of (Z, D, H_{1:3}), independently of the recursion code;
- after the same H_3 = (0, 1), the receiver's local mean differs by (−0.0050427, −0.0100854) (exact rationals, displayed as floats), while the sender's local law is unchanged and equal in both histories.

The failure is not confined to a null set of histories. Under the protocol law, Var(receiver's local mean after H_3 | T_2, H_3) is exactly nonzero (rank 1, trace 7.93e-5). So no update defined almost everywhere on the pair of local laws and the next message reproduces the protocol. This is a failure of the fiber test (inh:fiber-test) with no deleted register.

Conversely, for scalar cues the X-part of M_t has determinant Γ_0Γ_1(S_01²/(S_00 S_11) − 1). This is nonzero whenever Γ_0Γ_1 ≠ 0, because S_t ≻ 0. Moreover Γ_0 = Cov(X_0, D_1 | D_0)/Var(D_1 | D_0) under the preparation has numerator S N_0/(L_0 + N_0), so it is nonzero iff S ≠ 0. If S = 0 the preparation factorizes and no local law ever changes (appendix line 215). Hence in the scalar family the pair of local laws, the private values, the fixed model and the step index form an exact retained state: the fiber test of inh:deterministic-descent holds. The scalar run gives rank M_t = 2 at t = 0, ..., 4.

For d_0 = d_1 the factorization M_t = diag(Γ_0, Γ_1)·[[−S_10 S_00^{-1}, I], [I, −S_01 S_11^{-1}]] gives injectivity iff Γ_0 and Γ_1 are invertible. The second factor has determinant det(S_01 S_11^{-1} S_10 S_00^{-1} − I), which is nonzero because the squared canonical correlations of S_t ≻ 0 are below one. This step is by derivation, not computation.
**Fix:** Add a proposition to the section. "In the affine Gaussian protocol with one presentation per agent, S_t is history independent and each complete local law is an affine image of m_t. If rank M_t = d_0 + d_1 at every step (in particular d_0 = d_1 with Γ_0, Γ_1 invertible, which covers scalar cues with S ≠ 0), the pair of local laws with the private values and the step index is an exact retained state. For d_0 ≠ d_1 it can fail almost surely." Cite the d = (1, 2) example. Then say in the remark after imem:local-retention-failure (line 653), in chapters/04_prediction.tex:86 and in the outlook that local-only retention can fail without any public record.
**Falsifies:** An error in the generic-joint computation (rerun Part B and its independent confirmation block), or a reading of the retained state that also keeps m_t or λ_t. For the positive scalar claim: a scalar example with S ≠ 0 and rank M_t < 2.

### D08-IMEM2-02: The nonclosure proposition concerns the receiver's cue posterior, not the common state λ_t of the closure theorem; the same proof covers λ_1
**Category:** improvement
**Severity:** medium
**Location:** appendices/interacting_memory.tex:584-615 (imem:mixture-nonclosure) against :560-578 (imem:gaussian-closure); README.md:20; companion/c04_interacting_memory.tex:194-211
**Evidence:** imem:gaussian-closure asserts that the common private-record posterior λ_t stays Gaussian. The section titled "Failure of finite-mixture closure" proves something about a different object: the receiver's local cue posterior P(X_0 | D_0 = d_0, H = h). README:20 says that multiple presentations "need not remain a finite Gaussian mixture". "Remain" fits λ_t, which starts as the two-component mixture λ_0 = Σ_b ½ N((0, bα), Σ_D). It does not fit the receiver posterior, which starts as one Gaussian.

The gap closes at no cost. The D_0-marginal of λ_1 is λ_0^{D_0}(d_0) P(H = h | D_0 = d_0)/P(H = h), and λ_0^{D_0} = N(0, L_0 + N_0) is a single Gaussian because the receiver cue mean is zero under both labels. Since P(H = h | D_0 = d_0) = E[ℓ_0(h | X_0) | D_0 = d_0], with X_0 | D_0 = d_0 shifting to ±∞ and 0 < ℓ_0 < c_+, bounded convergence gives limits c_+ and c_-. The proof's own tail lemma then excludes a finite Gaussian mixture for this marginal, hence for λ_1, because marginals of finite Gaussian mixtures are finite Gaussian mixtures.

Numerically (`mixture_nonclosure.py`, h = 0.8), P(H = h | D_0 = ±80) equals c_± to ten digits. By contrast the D_1-marginal of λ_1 is exactly a two-component mixture, (c_-/2)N(d; −α, T) + (c_+/2)N(d; α, T), with residual 1.4e-17, because λ_0^{D_1}(d) w_+(d) = ½ N(d; α, T). The nonclosure therefore enters through the cross-agent conditional of D_0 given D_1, not through the sender's own record.
**Fix:** Add a corollary after imem:mixture-nonclosure: "λ_1 is not a finite Gaussian mixture; its D_0-marginal has density ratio to N(0, L_0 + N_0) with distinct tail limits c_±/P(H = h), although its D_1-marginal is an exact two-component mixture." Then the closure and nonclosure statements concern the same retained object, and README:20 is supported as written.
**Falsifies:** A finite Gaussian-mixture representation of λ_1^{D_0}, or failure of the bounded-convergence limit of P(H = h | D_0 = d_0).

### D08-IMEM2-03: For finite positive preparations with injective encoded tables, λ_t is a minimal changing state; this settles imem:open for the binary example
**Category:** extension
**Severity:** medium
**Location:** appendices/interacting_memory.tex:663-664 (imem:open), :205-213 (imem:retention); companion/c04_interacting_memory.tex:89
**Evidence:** The open problem asks for a smaller protocol-specific state that preserves every admitted future message law and receiver update.

Proposed statement:

- **Setting.** D_0 and D_1 finite, λ_t > 0, and for each j and d_j the encoded laws {V_j(d_j, d_{1-j}; ·)}_{d_{1-j}} affinely independent.
- **Claim.** Any changing state from which the protocol computes the next two speakers' message kernels at every value of their private records determines λ_t. Every common update needs these kernels: it multiplies λ_t by k_j(h | d_j) for all d_j, and every receiver likelihood integrates over the unknown d_j.

Proof sketch:

1. The next speaker j's kernels κ_j^{λ_t}(d_j, ·) = Σ_{d_{1-j}} V_j(d_j, d_{1-j}; ·) λ_t(d_{1-j} | d_j) determine λ_t(· | d_j) for all d_j.
2. After the message, λ_{t+1}(d_j | d_i) ∝ λ_t(d_j | d_i) k_j(h | d_j) with a known positive factor. So the following speaker's kernels determine λ_t(· | d_i) for all d_i.
3. Brook's lemma (D. Brook, Biometrika 51 (1964) 481-483) recovers a positive joint law from its two families of full conditionals, λ(d_0, d_1)/λ(d_0*, d_1*) = [λ(d_0 | d_1*)/λ(d_0* | d_1*)]·[λ(d_1 | d_0)/λ(d_1* | d_0)].

In the binary exchange each row of V_0 and V_1 has distinct entries (for example, at d_0 = 0 the entries are 2/3 and 4/7). `binary_exchange.py` recovers λ_1 = (45, 21, 35, 27)/128 exactly from agent 0's kernels (7/11, 12/31) at t = 1 and agent 1's kernels at t = 2 after H_2 = 1. In that example no changing state smaller than the three independent weights of λ_t serves this interface. A strictly smaller state can exist only when some V_j (or W_i) fails to be injective on conditionals, or on continuous record spaces.
**Fix:** State the proposition with the interface made explicit (common update and both receivers' likelihoods over two steps). Restrict imem:open to non-injective tables and continuous record spaces. The companion's "four masses, or three independent probabilities" (c04:89) then becomes a minimality statement for the example.
**Falsifies:** Two distinct reachable positive λ_t, in a model with affinely independent tables, that give identical next-two-step kernels at all private values. The proof obligations are positivity (Brook's lemma fails without it) and the exact definition of the admitted interface for per-agent rather than common states.

### D08-IMEM2-04: imem:mixture-nonclosure omits the quantifier on the received message; at h = 0 the posterior is the Gaussian prior
**Category:** gap
**Severity:** low
**Location:** appendices/interacting_memory.tex:585 (statement), :592 (proof fixes h > 0); chapters/04_prediction.tex:86
**Evidence:** The statement says that the receiver cue posterior "after one message is not any finite positive mixture of nondegenerate Gaussian laws". The proof fixes "any received h > 0". At h = 0, c_+ = c_- = N(0; m, Q + R), so k(0 | d) and ℓ_0(0 | x) are constant. Numerically ℓ_0(0 | x) = 0.18134268371 at x = −20, 0, 20 (`mixture_nonclosure.py`), and p^+ = g_0 is Gaussian. For h < 0 the two tail limits exchange and the argument goes through unchanged. The claim therefore holds for every h ≠ 0, which is almost surely, but not for every message.
**Fix:** Change the statement to "for every receiver private value d_0 and every received h ≠ 0", and add one sentence on h = 0.
**Falsifies:** Nothing; the h = 0 case is exact by the symmetry c_+(0) = c_-(0).

### D08-IMEM2-05: The parity counterexample uses a public record outside the declared information structure, and its conclusion is close to built in
**Category:** weakness
**Severity:** low
**Location:** appendices/interacting_memory.tex:619-653 (imem:local-retention-failure, imem:parity-record), against :39-44 (imem:information) and :23-27 (imem:preparation); companion/c04_interacting_memory.tex:216-225
**Evidence:** The declared information fields are I_{i,t} = σ(D_i, H_{1:t}), and private records are kernels L_i(u_i, dd_i) of the agent's own cue. The record C = X_0 ⊕ X_1 cannot be a component of either private record: `parity_counterexample.py` gives P(C = 1 | X_i = x) = 1 − ρ for both x and both i. Nor is it a posterior-sampled message. The proof nonetheless uses the extended fields σ(C, D_i, H_{1:t}) and the sampling law P(Y_1 | D_1, C), neither of which is declared. The record-process construction (def:registered-event, app:records) would supply the formalism but is not invoked.

The retained state deletes C by definition, and both local laws are independent of C (exact in ρ). The conclusion therefore reduces to this: deleting a registered variable that flips Cov(X_0, X_1 | C) from +1/4 to −1/4 loses that information. Every displayed number (5/8, 3/8, 9/16, 7/16, 1/2, 7/16, 9/16, 15/32, 17/32) is exact as an identity in ρ, and the sender's local law is unchanged by its own message in both histories. The counterexample is correct; it is weaker than its heading suggests, and D08-IMEM2-01 removes the need for the record.
**Fix:** Either declare the extension (I_{i,t} = σ(C, D_i, H_{1:t}) via def:registered-event, with policies conditioned on C), or present the two conditional preparations Λ_0(· | C = c) as two initializations. Each is of the declared form with the same local banks. Add the within-protocol Gaussian collision of D08-IMEM2-01 as the primary counterexample.
**Falsifies:** A reading of imem:information that already admits public preparation records.

### D08-IMEM2-06: The abstract's closure sentence omits the one-presentation condition that the next proposition shows to be load-bearing
**Category:** exposition
**Severity:** low
**Location:** frontmatter.tex:7
**Evidence:** The abstract says: "An affine Gaussian specialization closes exactly, while finite Gaussian-mixture closure and retention of local beliefs alone can fail." The two-presentation preparation of imem:mixture-nonclosure is also affine Gaussian (F_{1b} = 0, Gaussian noises), and it does not close. The qualifier "one presentation per agent" appears in README:20, chapters/04_prediction.tex:86 and chapters/06_outlook.tex:8, but not in the abstract. AUTHORING's general-law rule asks Gaussian results to state their family. The sentence is existential and true, but the family it names is not the one that closes.
**Fix:** "With one presentation per agent, an affine Gaussian specialization closes exactly; with several presentations the exact posterior can leave every finite Gaussian-mixture family, and retention of local beliefs alone can fail."
**Falsifies:** Nothing (wording).

### D08-IMEM2-07: Classical filtering and dialogue results are re-derived without attribution
**Category:** citation
**Severity:** low
**Location:** appendices/interacting_memory.tex:560-580 (imem:gaussian-reduced-recursion), :582-615, :4-6; companion/c04_interacting_memory.tex:5, :171-174
**Evidence:**

- **Kalman filter.** (imem:gaussian-reduced-recursion) is the information-form measurement update of the Kalman filter for a static state D observed through H_{t+1} = b_t + A_t D + η_t, and the history independence of S_t is the data independence of the Riccati recursion. The manuscript attributes exactly this precision update to Kalman (1960) at chapters/17_communication_links.tex:98 and companion/c17_communication_links.tex:53 (entry `Kalman1960` in item1_refs.bib), but not here.
- **Gaussian-sum filtering.** The nonclosure belongs with H. W. Sorenson and D. L. Alspach, Automatica 7 (1971) 465-479. Finite Gaussian sums are closed under likelihoods that are themselves finite Gaussian sums in the state, and k(h | d) = c_- + (c_+ − c_-) w_+(d) is not one. That is the structural reason for imem:mixture-nonclosure.
- **Iterated learning.** Posterior-sampling transmission is the "sampler" learner of Bayesian iterated learning (T. L. Griffiths and M. L. Kalish, Cognitive Science 31 (2007) 441-480).
- **Agreement dialogues.** Repeated two-agent exchange of posterior-derived messages under a common prior is the setting of J. D. Geanakoplos and H. M. Polemarchakis, J. Econ. Theory 28 (1982) 192-200.

Only Nayyar et al. (2013) is cited, for the common-information organization. I give these references from my own knowledge and did not re-open them in this session. A grep finds none of Sorenson, Alspach, Griffiths, Geanakoplos or Brook in the three .bib files.
**Fix:** Cite `Kalman1960` (already in the bibliography) and Anderson and Moore, Optimal Filtering (1979), at imem:gaussian-closure. Cite Sorenson and Alspach at the nonclosure remark (line 615) with the sentence above. Add one sentence relating posterior-sample exchange to posterior-announcement dialogues and to sampler iterated learning.
**Falsifies:** An existing attribution in the appendix or the companion section (none found).

### D08-IMEM2-08: Low-cost strengthenings of imem:gaussian-closure
**Category:** improvement
**Severity:** low
**Location:** appendices/interacting_memory.tex:560-578
**Evidence:**

1. **S_t is history independent.** A_t, the slope part of b_t, and R_t depend on λ_t only through S_t, so S_t is a deterministic function of t. The exact check gives identical S_1, S_2, S_3 for two different histories. The only history-dependent coordinate is m_t ∈ R^{d_0 + d_1}, which is affine in h_{1:t}.
2. **The positive-definite cue covariance is unused.** N_i ≻ 0 already gives λ_0 = N(ν, Cov X + N) ≻ 0, and a degenerate W is harmless because every likelihood depends on D only.
3. **The singleton condition is needed for speakers only.** If an agent with several presentations never speaks, λ_t stays a finite Gaussian mixture with the initial number of components. The reasons: λ_0 is a finite mixture over that agent's labels; the silent agent's labels leave the speaker's cue marginal Gaussian (X_1 | A_0 = a ~ N(ν_1, L_1) for every a); and sender invariance keeps the speaker's predictive law fixed. Under the declared alternating schedule both agents speak, so both singleton conditions are used.

Items 2 and 3 are by derivation, not computation.
**Fix:** Add item 1 to the theorem ("S_t is determined by t; m_t carries the history"). Label the positive-definite cue-covariance hypothesis as sufficient or drop it. Add item 3 as a remark.
**Falsifies:** An example in which S_t depends on the realized messages. The displayed recursion excludes this, since its inputs are S_t and fixed data.

### D08-IMEM2-09: Notation collisions in the appendix and the companion section
**Category:** consistency
**Severity:** low
**Location:**

- appendices/interacting_memory.tex: S is the sampling kernel S_{j,t} (:57), the cross-covariance S_{ab} (:433-437), the scalar cross-covariance S (:508) and the covariance S_t of λ_t (:566-573);
- appendices/interacting_memory.tex: a, b are labels (:428-505), the scalar gain and encoder slope (:512-556) and, as b_t, the offset (:566);
- appendices/interacting_memory.tex: P is the coupling P_0 (:15), the posterior covariance P_b (:454) and the scalar variance P (:511);
- appendices/interacting_memory.tex: C is the encoder C_j (:55) and the parity record C (:624);
- companion/c04_interacting_memory.tex: C is the encoder C_j (:29-37), the sensor variance C, C_i (:176-203) and the parity bit C (:216-225).

**Evidence:** The appendix renames the sensor covariance N_i at line 447 to distinguish it "from the encoder kernel C_i". The companion section reverts to C for the sensor variance while keeping C_j for the encoder and using C for the parity bit, three meanings within one section. In the proof of imem:gaussian-closure, S_t (the covariance of λ_t) coexists with S_{j,t} (the speaker's sampling kernel that generates the very message being assimilated). The 2026-10-22 review raised the same pattern for chapters 2-5 (2026-10-22 L03-DYN-09); it recurs in the new text.
**Fix:** In the companion use N_i for the sensor variances and B for the parity bit, as the appendix does. In the appendix rename the scalar gain and slope (for instance γ and β) and the covariance of λ_t (for instance Σ_t).
**Falsifies:** Not applicable (notation).

### D08-IMEM2-10: The declared alternating schedule is left by the repeated-sample comparison; no proof uses alternation
**Category:** consistency
**Severity:** low
**Location:** appendices/interacting_memory.tex:39 against :418-424; companion/c04_interacting_memory.tex:21 against :169
**Evidence:** imem:information fixes "a fixed known alternating sequence of speakers". The comparison "If agent 1 instead sends two fresh samples before a reply" uses schedule (1, 1, 0). Its numbers are correct (21/64, 13/64, 45/64, 171/244, exact). None of imem:history-theorem, imem:sender-receiver, imem:reduction or imem:gaussian-closure uses alternation; each holds for any fixed known schedule. The exact scripts run the reduction and sender invariance for schedules (1, 0, 1, 0) and (1, 1).
**Fix:** Replace "alternating" by "fixed known" in imem:information and in companion line 21.
**Falsifies:** A step in the general proofs that needs alternation (none found).

### D08-IMEM2-11: Unreduced fractions and a row/column slip in the binary example
**Category:** exposition
**Severity:** low
**Location:** appendices/interacting_memory.tex:397, :402, :416; companion/c04_interacting_memory.tex:162, :164
**Evidence:** 308/959 = 44/137 and 651/959 = 93/137 (gcd 7); the final 437/1918 is in lowest terms. Line 416 says "The sender invariance in each row". In the table the invariance is visible down each column at that agent's own transmission: agent 1 from "None" to H_1 = 1, and agent 0 from H_1 = 1 to H_1 = H_2 = 1.
**Fix:** Write 44/137 and 93/137, and "in each column, at that agent's own transmission".
**Falsifies:** Not applicable.

### D08-IMEM2-12: Asymptotic pooling of the private records in the Gaussian protocol
**Category:** extension
**Severity:** low
**Location:** appendices/interacting_memory.tex:560-580, :142-149 (imem:information-bound), :663-664 (imem:open)
**Evidence:** `gaussian_long_run.py` iterates the covariance part of the reduced recursion for a scalar alternating example. It gives t·diag(S_t) = (19.1, 1.206), (21.5, 1.217), (22.8, 1.223), (23.6, 1.227) at t = 50, 100, 200, 400, and the speaker slopes converge to nonzero values (−0.4474 for agent 0). This is a numerical observation for one parameter set.

Conjecture: if the speakers' message slopes (the encoded coefficient of D_j in E[Y_j | D_j, H_{1:t}]) stay bounded away from zero, then S_t^{-1} grows linearly and S_t = Θ(1/t). λ_t then concentrates at the true D, and every private posterior converges to W(D, ·). By imem:information-bound, I(Z; H_{1:n} | D_i) → I(Z; D_{1-i} | D_i), so the exchange eventually transfers all of the peer's private evidence. This would be a sample-exchange analogue of the Geanakoplos-Polemarchakis agreement theorem, reached asymptotically rather than in finitely many rounds.
**Fix:** As a construction sketch, prove a lower bound on the slopes from their limit F_j(Γ_jj + Γ_{j,1-j} lim S_{(1-j)j}/S_jj) and identify the exceptional parameter set where it vanishes. A Riccati comparison then gives Θ(1/t). Difficulty: moderate.
**Falsifies:** Parameters with nonzero cross-covariance for which S_t does not tend to zero.

## Checked without defect

| Label | What was checked | Method | Residual or reasoning |
| :-- | :-- | :-- | :-- |
| imem:binary-preparation | normalization, positive atoms, local bank form (A_i fair, P(X_i = A_i given A_i) = 3/4) | exact enumeration of 256 atoms | exact |
| imem:binary-correlations | all four identities at every information state of all 31 histories with t ≤ 4 (schedule 1,0,1,0); cue probabilities strictly inside (0,1) | exact; for all t, message probabilities lie in [1/4, 3/4], so every history has positive probability and p_i stays interior by induction | exact |
| imem:binary-first-likelihood | 5/8, 3/8, 9/16, 7/16, 17/32, 15/32; the receiver record does not enter P(H_1 given X_0) | exact | exact |
| imem:binary-first-posterior | 45/62 and the D_0 = 0 value 5/22 | exact | exact |
| imem:binary-reply | 7/11, 12/31, both different from the initial 5/8, 3/8; reproduced by the reduced readout of imem:reduced-readouts | exact | exact |
| imem:binary-private-weights | λ_0, λ_1, λ_2; normalizations 1/2 and 33/64 | exact | exact |
| imem:binary-reconstruction | P(X_0 = 1 given D), V_0, V_1, and P(X_1 = 1 given D) by exchange | exact | exact |
| imem:binary-second-posterior | 308/959 (= 44/137) and 437/1918 | brute force on the history law | exact |
| imem:binary-exchange | the full table | brute force | exact |
| imem:reduction (binary instance) | Γ_t = Wλ_t on all 31 histories with t ≤ 4 | exact | exact; the four weights reconstruct the expanded posterior |
| imem:sender-receiver (binary instance) | invariance of the entire private posterior at every own transmission, all histories | exact | exact |
| imem:history-likelihood (binary instance) | L_2(d; h) and Z_2 = 33/128, 31/128, 31/128, 33/128 for the four histories (agrees with interaction-checks.json); λ_2 = λ_0 L_2/Z_2 | exact | exact |
| imem:binary-repeated-samples and the two interpretation errors | 21/64, 13/64, 45/64, 171/244, direct-sensor 9/14, deterministic echo | exact | exact |
| imem:information-bound (also companion c04:227-232) | n = 2, i = 0 and 1 | floats on the exact law | residual 6.2e-15 |
| imem:full-free-energy-identity, imem:private-record-free-energy (binary instance) | h = (1, 1), random Q on 256 atoms and random ν on 4 atoms | floats on the exact law | residual ≤ 4.4e-16 |
| imem:gaussian-coupling | S = L_0^{1/2} J L_1^{1/2} with ‖J‖ < 1 gives the Schur complement L_1^{1/2}(I − J^T J)L_1^{1/2} ≻ 0; each local bank is independent of the other label | derivation | holds |
| imem:gaussian-private-posterior | P_b = (L^{-1} + N^{-1})^{-1}, the gain, the mixture form; agrees with cmem:gaussian-posterior-formula | derivation; sympy in the scalar case | holds |
| imem:gaussian-message-density, imem:gaussian-augmented-posterior | normalizer positive and finite; component density maxima independent of d; Tonelli | derivation | holds |
| imem:gaussian-receiver | u_ab, U_ab, label weights p_ab/π_0a and ℓ_0, with two labels per agent, scalar | formula against 2-D quadrature that uses only the bivariate cue density | residual ≤ 1.6e-12 |
| imem:scalar-gain, imem:scalar-message, imem:scalar-calibration | predictive gain f_1 g and variance Q_1 + f_1² P; N_eff; Var(H) = Var(H_dir); Cov(X_0, H) = aS; Cov(X_0, H_dir) = b f_1 S; g²(L_1 + N_1) + P = L_1 | sympy against generic conditioning with a fresh sample drawn from the generically computed predictive | every residual 0 |
| imem:scalar-receiver | m_0^+, v_0^+, the law of Y_0 given (d_0, h), V_H, Cov(X_0, H given D_0) | sympy | every residual 0; sampled/direct receiver gain ratio 3/5 at a test point |
| imem:gaussian-closure, imem:gaussian-reduced-observation, imem:gaussian-reduced-recursion | recursion equals generic conditioning; Γ_t = Wλ_t (joint mean and covariance); S_t ≻ 0 (Sylvester); vector example d = (1, 2) over 3 steps and scalar example over 4 steps; every proof step reconstructed | exact rationals | exact equality |
| imem:nonclosure-logistic, imem:nonclosure-message | w_+(d) and k(h given d) | numerical | ≤ 1.1e-16 |
| imem:nonclosure-tails | ℓ_0(h given −60) = c_- and ℓ_0(h given +60) = c_+ to ten digits; the bounded-convergence argument reconstructed | numerical and derivation | holds |
| tail lemma in the proof of imem:mixture-nonclosure | wider, equal-variance shifted, narrower and identical components, with positive weights | derivation | complete for every h ≠ 0; covers every finite size M |
| imem:parity-record, imem:parity-message, imem:parity-collision, imem:parity-local-output | every value as an identity in ρ; local laws independent of C; sender invariance; Cov(X_0, X_1 given C) = ±1/4 | sympy | exact |
| inh:deterministic-descent as used in imem:local-retention-failure | only the necessity direction (update square implies fiber test) is used, and it needs no measurability hypothesis | reading | correct use |
| imem:projection-loss | holds; it also holds for a q that is not the moment match, which only interprets the second term | quadrature, two-label mixture | residual 4e-15 |
| companion c04:106-233 | every fraction of c04:109-169, the scalar formulas c04:176-192, the nonclosure c04:194-211 and the parity values c04:216-223 agree with the appendix and with the checks above | comparison | agree; only the notation differs (D08-IMEM2-09) |

## Prior-review findings in scope

The appendix and the companion section postdate the 2026-10-22 review, and none of its findings falls in this scope.

| Prior ID | Status | Evidence |
| :-- | :-- | :-- |
| 2026-10-22 L03-DYN-09 | not applicable | Raised for chapters 2-5, the supplement and c03; the same notation-collision pattern recurs in the new text and is raised as D08-IMEM2-09 (appendices/interacting_memory.tex:57, :433, :508, :566; companion/c04_interacting_memory.tex:29, :176, :216). |

## Extensions and improvements

**Retention in the affine Gaussian family.** The singleton-presentation filter supports a short theorem that the manuscript is missing (D08-IMEM2-01, -08). The covariance S_t is a deterministic function of the step, so the changing state is m_t ∈ R^{d_0 + d_1}, and every complete local law is an affine image M_t m_t + c_t. Local-only retention, with the fixed model, the private values and the step index, is exact exactly when the fiber test holds. A sufficient condition is rank M_t = d_0 + d_1 at every step, which reduces to invertibility of Γ_0 and Γ_1 when d_0 = d_1; this covers every scalar preparation with S ≠ 0. When d_0 ≠ d_1 the kernel is nontrivial, and the d = (1, 2) example fails almost surely. The open converse is whether a nontrivial kernel forces failure for generic parameters. The obligation is that M_{t+1}Φ_t does not vanish on ker M_t, where Φ_t is the linear part of the mean update: a determinant condition in the model parameters. It is moderate work and settles the Gaussian part of imem:open.

**Minimality of the joint private-record state.** For finite positive preparations, Brook's lemma turns the open problem into a statement about the injectivity of the encoded tables V_j and reconstruction tables W_i on conditional laws (D08-IMEM2-03). When the tables are injective, λ_t is minimal for the interface "common update and receiver likelihoods over the next two steps"; in the binary example this is verified exactly. A smaller state can exist only through non-injective tables, and then the quotient of λ_t by the joint kernel of all future readout maps is computable by finite linear algebra. For continuous record spaces, the analogous step needs compatibility and uniqueness results for full conditional densities (Arnold and Press, J. Amer. Statist. Assoc. 84 (1989) 152-156, from my own knowledge, not re-checked). This is harder, because positivity and integrability conditions enter.

**Closure and nonclosure for the same object.** Stating the nonclosure for λ_1 through its D_0-marginal (D08-IMEM2-02) aligns the section with the closure theorem. It also explains the mechanism: the D_1-marginal stays an exact two-component mixture, and the failure enters through the cross-agent conditional of D_0 given D_1. That suggests an approximate filter that keeps λ_t in a conditionally Gaussian family given a finite latent label, replaces the logistic weight w_+(d) by a finite Gaussian-sum or probit-type approximation, and is assessed by its error in the next two speakers' message kernels. That is the interface identified in D08-IMEM2-03, not the local relative entropy of imem:projection-loss. The error bound would be an L1 bound on k(h given d) propagated through the reweighting, which is elementary for one step.

**A finite collision without a public record.** Whether the declared binary protocol, with no public record, admits two reachable histories with equal local laws and different next updates remains open (D08-IMEM2-05). At each t there are finitely many reachable λ_t, so a collision needs an algebraic coincidence. The construction: choose two preparation parameters (for example the cue-coupling and sensor accuracies) as unknowns, impose equality of the two agents' cue probabilities for two fixed histories of equal length and parity, and solve the resulting two polynomial equations. Then check exactly that the next receiver update differs. Difficulty is low to moderate. With D08-IMEM2-01 this would give finite and Gaussian versions of the counterexample inside the declared protocol.

**Long-run behavior.** The Gaussian probe (D08-IMEM2-12) suggests that the exchange pools the private records at rate 1/t. A proof would connect the appendix to the agreement literature and give the first dynamical statement about this protocol beyond finite histories.

## Not checked

- **General proofs in lines 1-324.** The measure-theoretic statements (non-dominated disintegration, measurable choice of conditional versions, the "admissible domain" extensions of the free-energy identities, and the denominator claim of imem:history-theorem) were checked only through the binary and Gaussian instances, not as general proofs. They are outside the assigned line range.
- **imem:free-energy and imem:reduced-free-energy.** These were checked numerically in the binary instance only. The local split (imem:local-free-energy-split) was not recomputed.
- **Cited sources.** I did not check the content of Nayyar et al. (2013) or Kallenberg (2021) against the citing sentences. I did not re-open in this session the primary sources I cite: Kalman 1960 (present in item1_refs.bib), Anderson and Moore 1979, Sorenson and Alspach 1971, Griffiths and Kalish 2007, Geanakoplos and Polemarchakis 1982, Brook 1964, Arnold and Press 1989.
- **Generality of the extensions.** I showed failure for d_0 ≠ d_1 for one example only, did not prove the generic converse of the rank criterion, and observed asymptotic pooling in one numerical run.
- **Rendered PDF.** I did not read it; the build logs show no warning in either scope file. The companion section has no figure.
