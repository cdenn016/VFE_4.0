# R-BND remediation log (chapter 10 and companion c10)

Files edited: `M/chapters/10_boundary_closure.tex`, `M/companion/c10_boundary.tex`. Not compiled (per brief). Check script: `remediation/scripts/R-BND/decay_jensen_check.py` (+ `.out`): contrast eigenvalues of 3I - 2 sum_k pi_k P_k equal 3 + 2/(N-1) to 1e-12 for d = 2, 3, 4; random rows give min rate >= 1; Jensen gap exactly 0 for k = d-1 and for a profile depending only on bit 1 at k = 0. Other identities were reconstructed by hand (see notes).

| Finding | Status | What changed (file, label) |
| --- | --- | --- |
| H7 / L07-BND-01 | applied | New section `sec:boundary-symmetry`: `lem:boundary-equivariance` (equivariance of field, (i) flow invariance of Fix(H) and restricted gradient, (ii) conjugation and descent to Fix(H)/K, (iii) descent of invariant response kernels and equivariant assimilations), cited to Palais 1979, Golubitsky-Stewart-Schaeffer 1988, Field 2007. New `prop:boundary-fixed-family` proving the family is exactly Fix(H_alpha) (sign-measurability by fiberwise ergodic arc rotations, Walters 1982; node mirror elements; induction; transitivity). Base-dependent automorphisms realized as fibered maps Phi_tau; phase rotations conjugate Fix(H_alpha) to Fix(H_alpha'). Paragraph stating the discarded data are gauge relative to a declared observable algebra (energy, metric, fresh responses, admitted records) and listing what symmetry does not supply (D_k and pair-law form, incoming term, eq:boundary-tree-response, decay, observability). Proof of `thm:boundary-tree-inheritance` now invokes the lemma. Intro rewritten. Companion: new symmetry paragraph. |
| E11 / L07-BND-12 | applied (short) | `prop:boundary-joint-spectrum` with proof (pointwise finite ODE with spectrum-determined scalar inputs; push-forward argument plus uniqueness of `lem:boundary-fine-wellposed`); paragraph identifying the root case as an instance, and naming what the hierarchy adds (node-level conditional spectra, stated without proof). |
| L07-BND-02 | applied | `prop:boundary-contrast-decay` (invariance of u-independent set and row convergence; linearization eq:boundary-contrast-linear with Walsh rates lambda_m >= 1 and common rate 3 + 2/(N-1) at class priors; fiberwise energy minimizer), and the remark that it equals 1 + kappa_N of eq:public-phase-linearization at J = 0. Global convergence and finite-time vanishing stated as not established. Witness paragraph kept with "persist on a nonzero time interval" plus decay sentence. Companion: decay paragraph. Linearization rederived (u-mean bookkeeping of f0 along the trajectory). |
| L07-BND-03 | applied | Paragraph after the inheritance proof: admitted records = compression-invariant and broadcast (base-only), histories are functionals of f_0(t); a likelihood reading sigma_0 multiplies the branches by L(o|r,u_1), L(o|r,-u_1) and exposes the phase via eq:boundary-hetero-root; one-subtree delivery does the same. Theorem statement now says "applied to every agent". Companion updated. |
| L07-BND-04 / L03-DYN-03 (ch.10 part) | applied | New `lem:boundary-fine-wellposed` (bounded log-density Banach chart, eq:boundary-fine-force, eq:boundary-fine-flow, Picard-Lindelof, normalization invariance, Fisher-gradient identification, dissipation). `thm:boundary-tree-flow` now states its chart and invokes uniqueness. |
| L07-BND-05 | applied | Signs introduced abstractly (conditionally independent fair); coin realization sigma_v = theta_v s_v(r) and circle realization eq:boundary-tree-signs; text says the atomless circle is used only for continuous phase families and the TV/Hellinger statements. Companion: one sentence. Abstract wording "An atomless profile construction" is in frontmatter.tex (not my file): see AUTHOR DECISIONS / lead note below. |
| L07-BND-06 | applied | `prop:boundary-observable` now states uniform model reference, reciprocal potential, binary spaces, and J not in {2, 6} (noting it includes |J| < 1/4); proof notes the leading coefficient vanishes only at J = 2, 6 and that a nonuniform reference adds a term. Factorization 4(6-3J)(6-J) = 12(J-2)(J-6) checked by hand. |
| L07-BND-07 | rejected (out of scope) | Location is a research record outside the manuscript; no construction from it appears in ch. 10 or c10. |
| L07-BND-08 | applied | Stray comma in eq:boundary-tree-jensen removed; exact equality condition (no dependence on bits k+2..d), identical vanishing for k = d-1, and convexity of D_k by joint convexity stated after the equation. Numerically confirmed (script). |
| L07-BND-09 | applied | TV range |delta| <= pi and the (2pi - |delta|) case; new eq:boundary-tree-hellinger; Fisher-Rao distance 2 arccos(1 - H^2) bounded between 2 sqrt(2H^2) and pi sqrt(2H^2) (proved via phi >= 2 sin(phi/2) and sin x >= 2x/pi); Holder-1/2 and non-rectifiability. Companion updated. |
| L07-BND-10 | applied | Companion title "A Parent That Keeps Its Fresh-Sample Responses"; opening now says "evolution and the fresh uniform-descendant response interface". |
| L07-BND-11 | applied | Comma fixed; metric renamed G^c (mathcal); lowest-pair values renamed p_+, p_-; TV densities p_+, p_-; reuse sample points xi, xi' and pair base p; observability: binary states kappa, kappa', Bernoulli law p_x, model contrast e, small parameter varepsilon. Companion nodes renamed L, R, 0 (root); line 52 rephrased (each leaf's fine force equals the profile force at its own path bits; no averaging over receivers). |
| L07-BND-13 | deferred (remark) | `rem:boundary-wreath`: states which properties the proofs use and the b-ary/class-prior/ancestor-phase generalization with the modified P_k; explicitly "we do not write out these cases". |
| L07-BND-14 | applied (remark) | `rem:boundary-heterogeneous` with eq:boundary-hetero-root (direct computation) and the conjugacy condition; dynamic closure of two-block families stated as not established (lens evidence was numerical only). |
| L07-BND-15 / G2-04 (ch.10 part) | applied | `lem:boundary-global` with proof: c_q bound from energy decrease, per-base-point max/min Dini comparison on the finite bit space, row log-odds lower bound; global existence. Theorem statement now asserts existence for all t >= 0. Constants rederived (K = F(0) + 3 ||log rho|| + 2). |
| L07-BND-16 | deferred | `op:boundary-minimality`. |
| L07-BND-17 | applied | One-paragraph scope statement with the Gaussian mixture computation (no implementation references). |
| Measurable selection (REPORT 8.3, line 132) | applied | Explicit Borel canonical representative by bottom-up lexicographic ordering of child value vectors. |
| L03-DYN-04 (ch.10 line 103) | applied | Gibbs-reduced and evolving-row flows stated to be different flows; row log-odds relax at rate T_a; retentions hold for each. |
| X5-IMPL-03 (ch.10 part) | applied | Records stated as assimilation events of def:dyn-record-interrupted; potential unchanged, force depends on the record only through the updated profile. |
| X5-IMPL-14 | applied (remark) | `rem:boundary-evidence`: persistent base-only evidence term, declared as an evidence-augmented potential distinct from def:dyn-potential; invariance under S; theorems and global bound carry over with V^q - log L. |
| X5-IMPL-15 | applied | Coin realization with finite base reduces to finite tables (thm:dyn-finite-flow applies); Jensen equality at k = d-1. |
| X2-FRAME-05 (ch.10 part) | applied | Covered by the symmetry section: inheritance is stated to follow from exact symmetries; intro says no selection claim. |
| Brief rule 3 (reciprocal potential, null-flow status, "formation") | applied | Intro and companion opening say the chapter concerns the reciprocal law potential, supplies its hierarchy, makes no selection claim; (F2), (F5) not addressed and null-flow comparison vacuous. "Formation" not used except "no endogenous formation time has been derived". |

Notes. All new mathematical statements were reconstructed before insertion: the fixed-point characterization (mirror elements verified edge by edge, including the phase adjustment of mirrored circles), the realization of base-dependent automorphisms Phi_tau by triangular half-period shifts, the joint-spectrum descent argument, the global bound constants, and the Hellinger/Fisher-Rao inequalities. The joint-spectrum proof treats the pointwise-applied path as a C^1 chart curve without a separate argument; this is the one step stated at sketch level.

## BIB ENTRIES

```bibtex
@article{Palais1979,
  author  = {Palais, Richard S.},
  title   = {The principle of symmetric criticality},
  journal = {Communications in Mathematical Physics},
  volume  = {69},
  number  = {1},
  pages   = {19--30},
  year    = {1979},
  doi     = {10.1007/BF01941322}
}

@book{GolubitskyStewartSchaeffer1988,
  author    = {Golubitsky, Martin and Stewart, Ian and Schaeffer, David G.},
  title     = {Singularities and Groups in Bifurcation Theory, Volume {II}},
  series    = {Applied Mathematical Sciences},
  volume    = {69},
  publisher = {Springer-Verlag},
  address   = {New York},
  year      = {1988}
}

@book{Field2007,
  author    = {Field, Michael J.},
  title     = {Dynamics and Symmetry},
  series    = {ICP Advanced Texts in Mathematics},
  volume    = {3},
  publisher = {Imperial College Press},
  address   = {London},
  year      = {2007}
}

@book{Walters1982,
  author    = {Walters, Peter},
  title     = {An Introduction to Ergodic Theory},
  series    = {Graduate Texts in Mathematics},
  volume    = {79},
  publisher = {Springer-Verlag},
  address   = {New York},
  year      = {1982}
}
```

Bibliographic data are standard and were not re-fetched online in this session; the lead may wish to confirm the Palais DOI before merging.

## NEW LABELS

Chapter 10: `lem:boundary-fine-wellposed`, `eq:boundary-fine-force`, `eq:boundary-fine-flow`, `lem:boundary-global`, `eq:boundary-global-bound`, `sec:boundary-symmetry`, `lem:boundary-equivariance`, `prop:boundary-fixed-family`, `prop:boundary-joint-spectrum`, `rem:boundary-heterogeneous`, `eq:boundary-hetero-root`, `rem:boundary-wreath`, `rem:boundary-evidence`, `op:boundary-minimality`, `prop:boundary-contrast-decay`, `eq:boundary-contrast-linear`, `eq:boundary-tree-hellinger`. All grepped unique across the manuscript tree before adding. Existing labels unchanged (section title of `sec:boundary-tree-loss` changed to "Public distinctions discarded at every depth").

## AUTHOR DECISIONS NEEDED

1. frontmatter.tex (lead-owned) says "An atomless profile construction"; L07-BND-05 shows atomlessness is unused by the exact statements. Suggested wording: "A profile construction on public sign partitions". AUTHORING.md's boundary paragraph says "public sign partitions" already and is consistent.
2. L04-INH-11 proposes stating equivariant descent once in chapter 5. `lem:boundary-equivariance` is currently local to chapter 10; if chapter 5 gains a general lemma, this one should be reduced to a citation of it.
