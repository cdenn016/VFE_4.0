<!-- rigorous-theory-search-metadata {"contract_id": "contract-sha256-70f14344fcdcf4941082bb0826fdae50a6f48a09a54f2945f1984813d5dcb555", "schema_version": "rigorous-theory-search/v1", "target_digest": "70f14344fcdcf4941082bb0826fdae50a6f48a09a54f2945f1984813d5dcb555"} -->
# What a bounded local jet cannot replace

## 1. The moving anchor is not germ-local

For c distinct from c0, two smooth sections can agree in an entire neighborhood of c and differ only near c0. A smooth physical rotation supported near c0 changes its phase there without changing any local germ at c. If the anchor rotation is nonzero, the published canonical local B at c changes. Thus no function of any finite jet, or even the complete germ at c, reproduces the published representative without reference memory. This statement is about that representative, not the existence of other statistically equivalent realizations.

Even if its phase at the current time is supplied, a single anchor angle or sum does not generally close its velocity. With reference B_1=b e1, B_2=b e2, B_3=b e1, the private contribution to dot S has coefficients involving the receiver-dependent Gibbs rows. Reference public and marked laws change those weights while keeping S fixed. Hence Omega=det(S,dot S)/|S|^2 is not a function of S alone. Symmetry cases can have Omega=0, but they do not prove closure on the open O1 domain. Retaining the full native reference tuple, as in L2, supplies a sufficient finite closure. No minimal reference dimension is claimed.

For an exact instance put A=C=0, l_1=l_3=0, l_2=v, and retain the same positive b in all three vectors. The self angular moment t_i is zero. Define s=1/(5 I0(J)), k=2/(5 I0(b)), E=exp(b I1(b)/I0(b)), u=sech(v/2), and D=s+kE+ku. The original Gibbs formula gives beta_12=beta_32=ku/D, beta_13=beta_31=kE/D, and beta_21=beta_23=k/(s+2k). Consequently

\[
\Omega=\frac{4ku/D-2kE/D-2k/(s+2k)}5,\qquad
\frac{d\Omega}{du}=\frac{4ks+6k^2E}{5D^2}>0.
\]

Changing v changes Omega while S=(2b,b) stays fixed. These are regular positive native laws. The computation refutes universal sum-only closure on that chart; analyticity also prevents a sum-only identity on an open independent-coordinate subset of the same connected chart. It does not refute a special symmetry-constrained invariant subfamily.

## 2. The finite differential-order obstruction

The following result excludes a family of local realizations, not all memory mechanisms. It is stronger than failure of a particular Taylor truncation.

Theorem N1. On a rich open class of smooth sections of the positive O1 law chart, suppose a parent actual-law section is defined by a smooth finite-order differential rule

\[
\bar p(c)=F(c,j_c^k x,h), \tag{N1}
\]

with values in a regular positive-Fisher statistical manifold. Its dependence on finite local native-history summaries may be included if those are themselves finite-jet functions on the fixed-history stratum. All additional contextual templates are fixed known data. There is no independently retained phase-bearing field or external/reference-context memory. Require equality of the fine and parent minimizing marked connection classes and minimized tensors for every admitted section in that open class. Then F, as an actual-law map, is independent of all positive-order jets. Consequently such a rule cannot also erase the common physical psi angle and preserve the geometry on that class.

The regularity in this theorem is smooth; it applies to the smooth subset of the published C2 class. It does not assume arbitrary C2 fields have k+1 derivatives. The class must permit independent finite-jet perturbations and possess a common regular marked orbit rank. This is satisfied in a smooth C2-open neighborhood of a strict-margin O1 family: high derivatives can be varied locally with small amplitudes, and strict finite-window nonflatness and selection margins persist. If a restricted section generator imposes differential relations on its jets, the theorem does not apply.

Proof of derivative independence. Let k>=1 and fix c, the entire k-jet, h and known data. For a nonzero contextual direction X vary the (k+1)-jet, while leaving all lower jets fixed. Contracting symmetric (k+1)-tensors with X surjects onto symmetric k-tensors: in coordinates X=partial_1, a homogeneous polynomial primitive in coordinate 1 realizes any chosen derivative. Small supported polynomial perturbations realize these variations within the smooth open class. Only an interval of amplitudes is needed.

At the point the parent law and its orbit map bar J are fixed. Its contextual law tangent changes as bar u_X(t)=bar u_X(0)+t w, with w the derivative of F in the varied highest-jet direction. The fine contextual tangent and its scores are fixed, because the fine actual law and history scores depend on their first jets. They do not use a (k+1)-jet. The fine minimizing class and h_min(X,X) are therefore fixed. Choose one representative A_f(X) in that class. Equality of the classes makes that same representative a parent minimizer for every t. Tensor equality requires

\[
\|\bar u_X(0)+\bar J A_f(X)+t w\|_{\bar g}^2=h_{\rm fine}^{\min}(X,X) \tag{N2}
\]

throughout an interval. The t-squared coefficient is bar g(w,w), so positivity forces w=0. Surjectivity of the symmetric contraction makes every highest-jet derivative of the actual-law map vanish. Descend one order and repeat to reach order zero. In density charts this is a statement about tangent scores in L2; nonidentifiable coordinate directions do not evade it because they do not change the actual law. Fixed retained history scores and a conditioning normalizer do not change the t-squared argument.

For the full random-history experiment this argument is made in the joint L2 space, not by deducing a conditional equality from an unconditional one. Vary the highest *initial* derivative and keep its lower initial jet fixed. Pointwise native evolution and fixed-record append maps leave the history density and all fine first scores unchanged at c. On a branch h the variation of the current highest derivative is D Phi_h applied to the chosen initial variation, with an invertible D Phi_h on its regular fixed-record flow stratum. The t-squared coefficient in (N2) is then the nonnegative integral E_H[bar g(w_H,w_H)]. It vanishes, so w_H=0 almost surely. Applying this to a finite basis of highest-jet variations and using branchwise invertibility yields derivative independence on the reachable branch law. Smooth positive branch kernels extend the conclusion to their full-support continuity domain; no pointwise claim on arbitrary null versions is needed. The public-pattern conditioning normalizer also depends only on the fixed lower jets in this variation. This proves the joint-experiment version used here.

Proof of phase obstruction. The order-zero actual-law map must be invariant under simultaneous physical psi rotation if it loses that phase for all sections, including sections that are constant in a neighborhood. Consider a contextual tangent w_phys rotating every B_i, with marked and other coordinates fixed. The fine psi score has squared norm

\[
g(w_{\rm phys},w_{\rm phys})=\sum_i |B_i|\frac{I_1(|B_i|)}{I_0(|B_i|)}>0. \tag{N3}
\]

The marked orbit acts on phi and not psi. The psi factor is independent of the marked/public factors conditional on history, its score has mean zero, and the Gibbs and relative-history laws are invariant under common psi rotation. Thus this physical score is orthogonal to every marked orbit score. It is horizontal, with strictly positive fine minimized norm. The parent derivative in that direction is zero, which gives zero parent minimized norm. This contradicts tensor equality. The argument also works inside a rank-two nonflat section class: hold its marked derivatives fixed and vary the common psi-phase derivative. The fine class is fixed and its minimized quadratic form acquires a positive quadratic term, whereas the order-zero invariant parent does not. No flat-family escape is being used to assert a nonflat-window result.

For finite histories, the pointwise transformation x(c,t) -> rho(R_{alpha(c)})x(c,t), with alpha fixed in time, preserves every native kernel and update at each c. The actual history law and persistent evidence are unchanged as functions of c. Therefore their retention cannot supply the missing derivative of alpha. An added phase-bearing memory initialized from c0 evades the hypotheses of N1 and is exactly the datum used in L2. This is an escape from a scoped obstruction, not a counterexample to it.

## 3. Why equality of the tensor alone is not enough for N1

The proof used equality of the entire minimizing classes in the same marked Lie-algebra convention. If only the minimized tensor were fixed, a high-derivative variation of the parent tangent could be pure marked orbit motion and its minimizer could change. The norm polynomial (N2) at one fixed A_f would then not be constant. Likewise, retaining only some nonzero curvature is weaker than inheriting the original tensor and connection. Neither weakening is authorized by the question. We do not claim N1 under those weakened hypotheses.

No fixed Markov data-processing theorem is applied to the nonlinear jet-to-law map F. The proof works directly with its actual parent tangent and positive Fisher metric. This matters because parameter-dependent sample maps can add information and are not covered by fixed-statistic score projection.

## 4. A single regular finite encoding of arbitrary sections

Theorem N2. Suppose one finite-dimensional smooth parent state r in a manifold of dimension m must reproduce the original direct public response law at every c in a contextual patch, on a C2-open set of sections. Assume its retention R and each response-evaluation map are C1 on finite-dimensional smooth variations of the section. On a fixed retained-history stratum, such a finite encoding does not exist, even before imposing the psi quotient or geometric inheritance.

Proof. Choose m+1 disjoint small contextual neighborhoods away from c0 and one point c_j in each. Perturb the public natural parameter l_1 by sum_j a_j f_j(c), where f_j(c_j)=1 and f_j vanishes on the other neighborhoods. At the selected time these are independent chart directions; native fixed-history flow can be inverted locally to obtain admitted initial perturbations. Strict census and finite-window regularity margins persist for small amplitudes. The direct sign response has probability Pr(Z=+1)=sigmoid(L_1), and partial L_1/partial l_1=1. Hence the Jacobian of its m+1 probabilities at the chosen points with respect to a=(a_1,...,a_{m+1}) is a diagonal matrix with positive entries. Its rank is m+1.

If all those response laws factor through r=R(x), that Jacobian factors through an m-dimensional tangent space and has rank at most m, a contradiction. Raw record words are kept fixed in this variation, so their discrete/continuous values do not add rank. Any finite-dimensional smooth memory is counted in m. A finite number of jets and reference evaluations are also counted there. No special preparation prior, numerical dimensional estimate or infinite-time limit occurs.

This theorem concerns one global finite state that promises responses at all c. It does not refute a field with one finite state per c, an evaluation-local realization with finite reference memory, a finite-dimensional generator family, an infinite-dimensional memory, or arbitrary discontinuous/Borel encodings of a function into a real number. Such encodings are outside the regular finite-law-coordinate notion. The published protocol does not itself add context-relocation actions; N2 is relevant if a single finite parent is meant to replace its entire indexed experiment. It is not a reason to delete or change a probe.

## 5. Consequences and removal obligations

The geometric difficulty is not a hidden spatial derivative in the native evolution: L4 closes. It is supplying a phase-coherent actual law family while removing only one context-constant phase. A finite derivative-only quotient cannot do that universally under N1. A full arbitrary contextual function cannot be stored in one regular finite state under N2. Reference memory removes the first local closure problem but does not encode all arbitrary section values. A finite preparation generator removes the second obstruction on its restricted image. These are distinct obligations.

Nothing here proves general impossibility of exact local parents on other declared invariant families, with additional phase-bearing memory, or with other statistically sufficient law realizations outside the stated differential class. Interaction-created invisibility, second-stage recursion, and all future_TODO extensions remain deferred.
