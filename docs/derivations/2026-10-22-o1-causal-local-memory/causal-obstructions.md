<!-- rigorous-theory-search-metadata {"contract_id": "contract-sha256-26ec7fe2a48e4cb1afab6357d8b2bb3c2f89e4e3cf052844f500ffc03dfea722", "schema_version": "rigorous-theory-search/v1", "target_digest": "26ec7fe2a48e4cb1afab6357d8b2bb3c2f89e4e3cf052844f500ffc03dfea722"} -->
# Two scoped causal-memory obstructions

## 1. What the first obstruction allows

Theorem P concerns a strictly local, spatial-order-zero causal retention at each fixed context c. It may read the *entire* native local value path x(c,s), s from the actual initialization to t, together with the complete actual chronological history, persistent evidence, original census records and fixed templates. Its temporal memory need not be finite-dimensional, finite-horizon or Markovian. It may compute arbitrary causal functionals of that path. It cannot read contextual derivatives, neighboring contexts, a reference context or an externally initialized contextual phase field. Assume that the complete retained parent state, including all memory and any preparation seed, is invariant under a constant physical rotation of the fine preparation, as required for erasing that common angle. The actual parent laws must be functions of the retained state and the fixed data. A phase carrier that still distinguishes the erased angle is outside this hypothesis and fails that physical compression requirement.

The conclusion is that this class cannot inherit the actual minimized Fisher tensor for the rich open O1 preparation class. This is stronger than failure of one finite temporal-memory ansatz. It is weaker than a theorem about every conceivable causal memory.

For deterministic retention the invariance is pathwise. The same argument holds for a stochastic retention kernel if its complete retained-state law is rotation invariant, its randomization is independent of the fine phase, and its resulting actual experiment has the stipulated regularity. The kernel must itself be the declared parent retention; a hidden phase-bearing random seed or a calibrated contextual phase field cannot be silently omitted when assessing invariance.

## 2. Pointwise phase-field symmetry of every fixed history

Choose any smooth time-independent function kappa:C→R, and replace the initial preparation by

\[
x^\kappa(c,t_*)=\rho(R_{\kappa(c)})x(c,t_*).
\]

The native field is pointwise in c. Its physical equivariance and fixed-record uniqueness imply

\[
x^{\kappa,h}(c,t)=\rho(R_{\kappa(c)})x^h(c,t). \tag{P1}
\]

All fixed responses, including the full joint relative-angle response, and every permitted update likelihood are invariant at each c. The same actual policy, wait and transaction randomness therefore couples the histories identically. In particular

\[
\Pi_c^\kappa=\Pi_c, \tag{P2}
\]

as an entire contextual family, with identical persistent coefficients and retained census records. Initial and later record labels remain fixed lab labels; they are not rotated with kappa. No conclusion about independent primitive invisibility is used: a proper block rotated against an unrotated exterior is still exposed by the original probes. P1 rotates the full interacting block at each context.

At fixed c the two complete local value paths differ by the single constant rotation R_kappa(c). A strictly local rotation-invariant path retention must give the same retained state, or the same retained-state kernel, at that c. Therefore its *actual parent law family* is identical for x and x^kappa, for arbitrary kappa allowed by the preparation class. This follows even if memory has arbitrary temporal size. Causal processing cannot distinguish two paths that belong to the same declared input equivalence class.

## 3. The fine geometry distinguishes those families

At a chosen point c_* take kappa(c_*)=0 and X kappa(c_*)=lambda, holding the other initial derivatives fixed except for the indicated common phase. Small supported smooth perturbations realize an interval of lambda. At c_* the actual current laws, their marked orbit scores and the full history law agree. The fine contextual score changes by

\[
s_X^\kappa=s_X+\lambda\ell_{\rm phys}. \tag{P3}
\]

Conditional on the history, psi factors independently from the marked/public recognition variables. The common physical phase has no score in the Gibbs-label or history law, since both are invariant. The marked action has no psi component. Thus ell_phys is orthogonal to every marked orbit score, also in the joint history/current L2 space and after the original public-pattern conditioning. Its squared norm is

\[
\|\ell_{\rm phys}\|^2
=E_H\sum_{i=1}^3 |B_i^h|\frac{I_1(|B_i^h|)}{I_0(|B_i^h|)}>0. \tag{P4}
\]

Here the conditional sum runs over the current primitive recognition factors of the declared collective comparison experiment. Their normalization gives the identity Var(B dot J2 e_psi)=|B| I_1(|B|)/I_0(|B|). The cone condition makes every B_i nonzero on each finite branch. The strict inequality is local in the regular finite experiment; no uniform positive lower bound over all future histories is required.

Because P3 adds a score orthogonal to the marked orbit, the fine minimizing affine connection class stays the same at c_*. If s_X^hor=s_X+ell_Amin(X), then

\[
h_{\rm fine}^{\kappa}(X,X)=h_{\rm fine}(X,X)
+2\lambda E[s_X^{\rm hor}\ell_{\rm phys}]
+\lambda^2\|\ell_{\rm phys}\|^2. \tag{P5}
\]

The right side is not constant on any open interval of lambda. The parent actual family and hence its Fisher tensor are identical for all kappa by Section 2. They cannot equal P5 throughout that interval. This proves Theorem P. It is unnecessary to infer an almost-sure conditional geometric identity from equality of averaged tensors; P3–P5 are identities directly in the full joint experiment.

The perturbations can be made inside the strict finite-window O1 class: public grouping and fixed censuses are exactly invariant, and small smooth kappa preserves the common positive-rank nonflat window by continuity. A C2-small bump with a nonzero first derivative is enough. The obstruction does not rely on replacing the problem by a flat or degenerate family. Nor does it follow from the response quotient alone; the positive horizontal score of the actual comparison law supplies the additional geometric demand.

The theorem can be falsified by an in-scope retained-state functional that distinguishes x and x^kappa despite local rotation invariance, or by a vanishing physical horizontal score within the stated experiment. A memory that reads spatial derivatives or an initial contextual phase datum instead changes the hypotheses and does not falsify it.

## 4. Smooth bounded-jet dynamical memory

Theorem J covers a different and larger spatial-input class. Fix a finite contextual order k. At initialization let a finite-dimensional memory m be a smooth local function of c, j_c^k x(t_*) and the fixed original initial records. Between records it obeys a smooth ODE

\[
\dot m=G(m,j_c^k x,c,h), \tag{J1}
\]

and each actual record gives a smooth branchwise update m^+=U_{a,o}(m,j_c^k x,c,h). The actual normalized parent-law map is smooth in m,j_c^k x,c,h, with positive Fisher metric on identifiable law tangents. Memory is deterministic given the actual history. The order k is a common bound for initialization, evolution, readout and all finite branches. All templates are fixed known data. The full retained state must erase the physical common angle. No nonlocal/reference seed, independent phase field, unbounded-jet initialization or spatial PDE is hidden in m.

On a rich smooth open section class admitting independent finite-jet variations, such a memory cannot universally preserve both the original minimizing marked affine connection classes and the minimized Fisher tensor. This holds even if the chronological history is arbitrarily long, provided each history being compared is finite. A growing number of fixed-dimensional temporal recurrence steps is not an escape.

The proof has two parts. First, for each fixed history h and time, the augmented pointwise system for (j^k x,m) closes at order k. Initializing and solving J1 gives

\[
m_t=\Psi_{t,h}(c,j_c^k x(t_*)).
\]

Each native fixed-record flow is locally invertible in law coordinates; evidence appends do not reset those coordinates. Its finite jet prolongation is locally invertible as well, with triangular highest-order block given by the invertible value-flow derivative. Thus there is a smooth actual-law map

\[
\bar p^h(c,t)=F_{t,h}(c,j_c^k x(c,t)). \tag{J2}
\]

This statement eliminates the generated finite memory only as a function of the *current jet on a fixed-record branch*. It does not claim that a current snapshot determines the unconditional past law. The actual past law remains the initialized chronological experiment, with the complete history retained. Its lower contextual scores are included in the second part of the proof.

Fix the k-jet for k≥1 and vary a (k+1)-jet in one contextual tangent direction. The fine first score, fine history density and first history score are unchanged. The parent law and its marked orbit map at the point are fixed, but its contextual tangent becomes u+tau w. Equality of the full minimizing affine classes makes any fine minimizer A_f a parent minimizer for every tau. Equality of minimized tensors then requires

\[
\|u+\bar J A_f+\tau w\|^2=h_{\rm fine}(X,X) \tag{J3}
\]

on an interval. The coefficient of tau squared is the nonnegative Fisher norm of w. It must vanish. Contraction of symmetric (k+1)-tensors by a nonzero X surjects onto symmetric k-tensors, so the actual-law derivative with respect to every highest-order jet vanishes. Descending gives independence from all positive-order jets.

For random histories perform the variation on the highest *initial* jet, keeping its lower jets fixed. The history density and fine first score stay fixed, and the highest-jet variation on branch h is its invertible prolonged value-flow derivative applied to the chosen initial variation. J3 is then an identity in the joint L2 space; its quadratic coefficient is E_H[||w_H||^2]. It vanishes. A finite basis of highest-jet variations and branchwise invertibility give w_H=0 almost surely for every direction. With the declared smooth positive kernels this holds on the reachable support continuity domain. There is no inference of conditional equality from averaged equality. Public-pattern normalization uses the unchanged lower jets in this variation.

After descending to order zero, common-rotation invariance of the actual parent law makes it kill the physical phase tangent. That tangent has the strictly positive fine horizontal norm P4, or equivalently produces P5 while leaving the parent tangent unchanged. This contradiction proves J. This is the earlier N1 differential-order argument with a new reduction J1→J2 showing why a broad class of *causally generated* memory is still in its scope. No fixed-statistic data-processing theorem is imposed on the nonlinear jet-to-law map.

The proof uses equality of minimizing affine classes as well as tensor equality. If only a minimized tensor were required and minimizers could vary, the parent high-jet tangent could be marked-vertical and J3 at one fixed A_f would not follow. The frozen problem requires both. The proof applies on smooth sections inside the C2 class; it does not differentiate an arbitrary merely C2 field k+1 times. Small polynomial bumps supported on successively small neighborhoods permit independent high jets with small C2 norm.

## 5. Boundaries of the result

The conditional accumulator in `causal-phase-memory.md` is not a counterexample to P or J. Its initial-coherence subfamily is not the rich open preparation class used by either obstruction. On that subfamily the missing initial phase one-form is identically zero and the otherwise fatal variation is unavailable. The accumulator solves J1 at k=0, and its actual law is the native flow from the local canonical initial value, exactly as J2 predicts.

P excludes arbitrary temporal memory whose spatial input is only the local value path modulo a constant physical rotation. J excludes smooth finite-dimensional deterministic recurrences reading a uniformly bounded contextual jet, even with complete raw records. J does not by itself cover stochastic memories with additional sampled internal state: their complete joint experiment and score regularity would have to be specified and the positive-norm argument repeated there. P still covers stochastic kernels satisfying its explicit invariant-input hypotheses.

Neither theorem excludes a causally acquired contextual phase-coherence datum through additional permitted spatial information, a regular memory with genuinely unbounded contextual order, a differently specified infinite-dimensional differential memory, constrained preparation generators, or another statistical realization outside the stated classes. The original protocol supplies no context-relocation channel, so one cannot simply assume that histories visit and calibrate neighboring contexts. Whether an admissible mechanism outside P and J can retain precisely the needed contextual phase relation, erase one common angle, and generate the original actual geometry remains unresolved. An unrestricted nonexistence conclusion is not established.
