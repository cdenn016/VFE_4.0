"""Writer for lens report D18-LP, part 3 (findings 10-16). Appends to the report file."""
from pathlib import Path

OUT = Path(r"C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-29-magent-theory-2.0-review/lenses/D18-LP.md")

TEXT = r"""### D18-LP-10: The slice canonicalization, the moving-frame field and the accumulator are the method of slices and the freezing method, attributed in chapter 12 but not here
**Category:** citation
**Severity:** low
**Location:** lp:retention (14:103-109), lp:omega and lp:closed-reference (14:114-126, which cite only HorstmeyerAtay2016), lp:causal-system (14:215-221), lp:phase-information (14:159-164)
**Evidence:** `12_open_formation.tex:329` attributes the anchored normalization to the method of slices ("a first-Fourier-mode slice \citep{BudanurEtAl2015} in the method of slices ... \citep{RowleyEtAl2003,BeynThummler2004}, a cross-section in the sense of moving frames \citep{FelsOlver1998}") and calls its moving equation "the in-slice equation of the freezing method" (12:379). Chapter 14's $U(S)$ is a slice with phase condition $S_y=0$, `lp:closed-reference` is the in-slice equation with an anchored (reference) phase condition, and `lp:causal-system` with $\dot g=\omega_{\rm loc}J_2g$ is the in-slice equation plus the reconstruction equation for the group variable. Chapter 12 also already states the content of `lp:phase-information` ("Choosing the offsets separately at each context would instead remove contextual phase variation itself and change the contextual geometry; the anchored choice avoids this", 12:337). Chapter 14 cross-references neither. AUTHORING.md asks that instances of named classical mathematics be attributed with what is new stated. I did not recheck these four records against the publishers (the web search tool was unavailable); they are present in `references.bib` with volume, pages and DOI.
**Fix:** Cite the four records at `lp:retention`, `lp:closed-reference` and `lp:causal-system`, cross-reference `open:anchor` and `open:moving-flow`, and state what is new here: a common collective rotation exposed against every proper block, the per-agent field, and evaluation-local jets.

### D18-LP-11: The dynamics used is a Gibbs-reduced per-agent flow on circle alphabets, which chapter 3 does not define
**Category:** gap
**Severity:** low
**Location:** lp:native-field (14:43-69, "This is an invariant family of the own-law Fisher field"); 14:23 ("Rows are reduced exactly at unit temperature"); 14:80 (the prescribed null); AUTHORING.md:81 ("the original unit-temperature per-agent field")
**Evidence:** `def:dyn-per-agent-flow` (03:354-355) is the flow `eq:dyn-fisher-flow` "on finite positive tables", and `thm:dyn-finite-flow` lets $u^a$ range over all tables $q_i,s_i,\beta_i,\gamma_i$, so the defined per-agent dynamics evolves rows at unit mobility; `inh:null-flow` (05:1380) refers to that flow. Chapter 14 instead minimizes the rows at every state (as chapter 13 declares at 13:9) and runs on $\{\pm1\}\times S^1\times S^1$. The proof's "centered logarithmic Fisher velocity" and "invariant family of the own-law Fisher field" presuppose a density-level field on positive densities of a continuous alphabet, and its uniqueness, which the manuscript does not define (the regularity rule of the brief). The family-level statement holds: script s01 computes the Gibbs-reduced per-agent objective by brute-force quadrature (64 by 64 grid, channel output by numerical marginalization), takes the natural gradient in the chart $\eta=(l,A,C,B)$ with the quadrature covariance of $(z/2,e_\phi,ze_\phi,e_\psi)$, and matches `lp:field` to 3.8e-10 (content) and 2.0e-11 (presentation), with rows matching `lp:weights` to 1.1e-16, at a generic state with $J=0.37$. This is the chapter 14 counterpart of D17-PA-03.
**Fix:** Define the dynamics of this chapter as the natural-gradient flow $\dot\eta=-g^F(\eta)^{-1}\nabla_\eta\mathcal F_i^{\rm red}$ of the Gibbs-reduced per-agent objective on the declared exponential family (analytic field, linear growth, global on finite intervals by `lp:cone`), name it as distinct from `def:dyn-per-agent-flow`, and define the null as the corresponding reduced null; drop "invariant family of the own-law Fisher field" unless the density-level field and its uniqueness are supplied.

### D18-LP-12: The joint-$L^2$ descent in lp:jet-obstruction omits the parent's own history score
**Category:** gap
**Severity:** low
**Location:** lp:jet-obstruction, second proof paragraph (14:330)
**Evidence:** "Vary the highest initial derivative while holding lower initial jets fixed. The history law and fine first scores stay fixed..." refers to the fine history law. The theorem does not assume that the parent's history law equals the fine one. If the parent generates its own history from its laws at past transaction times, which are themselves $F(c,j^k_cx(t_m),h)$, its history score also varies affinely in the top-jet parameter, by some $w^{\rm hist}(H)$. The conclusion survives: $w^{\rm hist}(H)$ is a history function with mean zero and $w_H$ is a conditional score with conditional mean zero, so they are orthogonal in the joint $L^2$ space and $\E\|w^{\rm hist}+w_H\|^2=\E|w^{\rm hist}|^2+\E\|w_H\|^2=0$ forces both to vanish. The step is not written.
**Fix:** Add the orthogonality sentence, or add operational history inheritance to the hypotheses.

### D18-LP-13: The clause "Transactions use fresh samples unless reuse is explicitly recorded" is undefined and would break the kernel induction
**Category:** gap
**Severity:** low
**Location:** chapters/14_collective_local_parent.tex:84; the inductions at 14:139 (lp:local-realization) and 14:267 (lp:causal-flow); companion/c14_collective_local_parent.tex:63
**Evidence:** The inductions use that each transaction kernel is a function of the current retained state and the history. If a primitive absolute sample is reused by a later query, the later output depends on the conditional law of that sample given the earlier outputs; the joint law stays invariant under common rotation, but the kernel is no longer a function of the stated retained data, so the parent would have to retain the reused latent sample (in canonical coordinates) or its conditional law. The chapter never says what "explicitly recorded" reuse adds to the retained state.
**Fix:** Forbid reuse in the admitted protocol, or state that reused samples, expressed in the canonical frame, are part of the retained state, and extend the induction.

### D18-LP-14: Compatibility of the null-box and nonflat-window ranges is not stated
**Category:** gap
**Severity:** low
**Location:** chapters/14_collective_local_parent.tex:80 (null box $|A_i|\le1/10$, $|C_i|\le2\epsilon^3$, $|B_i|\le1/10$), 168 ("$a,d$ in positive compact ranges independent of $\epsilon,\delta$"), 195
**Evidence:** The null's invariance argument (inherited from `pa:null-bound`) needs $\epsilon\le1/100$ and initial $|A_i|\le1/10$, $|B_i|\le1/10$; chapter 13 takes $|A_i(0)|,b_i(0)<1/20$ (13:292). Chapter 14 gives no upper bound for $a$, $d$ or $\delta|b_i|$, so the stated window family and the stated null box are not shown to intersect. The intersection is clearly nonempty (script s07 uses $|A_i|\le0.05$, $\delta=0.05$, $\epsilon=0.01$ and the null keeps $\beta^0_{ii}\in[0.1998,0.2]$ and the third public sign negative through $t=60$).
**Fix:** State $0<a_-<a_+<1/20$, $0<d<1/20$, $\delta\max_i|b_i|<1/20$ and $\epsilon<\min(\epsilon_0,1/100)$.

### D18-LP-15: Notation collisions and an undefined term
**Category:** consistency
**Severity:** low
**Location:** chapters/14_collective_local_parent.tex:97, 168, 244, 283, 303-309, 320; companion/c14_collective_local_parent.tex:131
**Evidence:** $D$ denotes the marked inertia $D(t)$ (14:168), the row denominator (14:307) and the state derivative (14:244); $v$ denotes the cone vector (14:97), $e^{-t}a$ (14:168) and the public log odds $l_2$ (14:303). The companion (c14:131) sends the reader to `cp:curvature`, where $B(t)$ is the marked inertia, while $B$ is the physical natural parameter throughout c14. The transmitted physical angle is called "private" (14:143, 159, 166, 255), although "private" elsewhere (chapters 7 to 9) denotes a coordinate that no channel transmits, and this $\psi$ is copied by every peer channel; the 2026-10-22 review raised the same convention conflict for chapter 12 (L09-OPEN-06, X1-GAUGE-03). "Positive chart" (14:320) is undefined, and 14:283 calls an earlier paragraph of the same chapter "the published finite-window continuity argument".
**Fix:** Rename (for example $\mathcal D(t)$ for the marked inertia, $N$ for the row denominator), use "physical" or "transmitted" instead of "private" for $\psi$, define the chart, and refer to the earlier paragraph by label.

### D18-LP-16: lp:profile-obstruction is a dimension count against an encoding no construction proposes
**Category:** weakness
**Severity:** low
**Location:** lp:profile-obstruction (14:337-343)
**Evidence:** The proof shows that $m+1$ disjoint bump perturbations give a diagonal response Jacobian of rank $m+1$, which cannot factor through $\R^m$. This excludes only one regular finite state encoding infinitely many independent profile values, which none of the chapter's constructions attempts (each keeps a state per context or a generator restricted by hypothesis). The proof is correct; its contribution is to rule out a strawman.
**Fix:** Demote to a remark, or replace by a statement with content, for example the minimal memory per context needed for geometric inheritance: after one reference read at preparation the frozen-frame realization needs no live reference memory at all (14:182), which answers the question "not a proved minimal one" for that representative.

"""

with OUT.open("a", encoding="utf-8") as f:
    f.write(TEXT)
print("appended", len(TEXT), "chars")
