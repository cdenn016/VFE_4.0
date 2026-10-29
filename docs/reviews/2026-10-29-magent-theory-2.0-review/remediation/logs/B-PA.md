# B-PA remediation log (2026-10-29, wave 2)

Files edited (manuscript root `Research/manuscripts/MAgent_Theory_2.0/`, branch `magent-2.0-remediation-2026-10-29`, nothing committed, nothing compiled): `chapters/13_per_agent_inheritance.tex`, `chapters/14_collective_local_parent.tex`, `companion/c13_per_agent_inheritance.tex`, `companion/c14_collective_local_parent.tex`. CRLF line endings kept (checked: every newline is CRLF in all four files).

Scripts (`V/remediation/scripts/B-PA/`): `bit_constants.py` with outputs `bit_constants_bern.out`, `bit_constants_pert.out`, `bit_constants_four.out` (single-threaded RK4, each under 110 s; the first attempt ran too long and was stopped, then rerun with a coarser step in three parts). `apply_edits.py`, `replace_block.py`, `rename_symbols.py`, `fix_bs.py` are text-edit helpers; `lint.py` checks environment balance, banned macros and dashes, dangling and duplicate labels, and cite keys (clean on all four files: no dangling `\cref`, no duplicate label, braces and `$` balanced).

What the numerics show (support, not proof). Bernoulli construction, exact Gibbs rows, from `l(0)=ε(4,1/2,-1)`, `ε ∈ {0.01, 0.05}`: for `c ∈ {0.13,…,0.33}` the full flow gives `1|2|3 > 12|3 > 123`, both nulls `1|2|3 > 12|3`, the frozen-environment control `1|2|3 > 12|3 > 123`; for `c ∈ {0.34,…,0.9}` full and both nulls give `1|2|3 > 12|3`. Final `l/ε`: full (2.143, 2.143, 0.714), N1 (5.00, 5.00, −5.00), N2 (1, 1, −1), frozen (0.80, 2.20, 0.80). Sup of agent 3's ratio 0.33334 (ε=0.01), 0.33355 (ε=0.05). Twenty perturbations at ε=0.02, c=1/4: P equals P^fr in 20/20, equals neither null in any. Four-state on the product-content family at the base point: full selects at t=0.78 and keeps the pair to t=120; N1 and N2 never select; frozen selects on [0.70, 3.56] and then loses it.

What was proved by hand and written into ch13. With `u=l/ε` the Bernoulli field is exactly `u̇ = Mu + o + R_ε(u)`, `R_ε,i = Σ_{j≠i}(β_ij−π_ij)u_j`; Bernoulli KL ≤ (l−l')²/8 gives `|R_ε| ≤ (24√3/5)(e^{18ε²}−1)` on the ball of radius 13/4 about u_*, which is invariant for ε ≤ 1/20; symmetry of M gives the uniform-in-time shadowing `|u−ū| ≤ 5R̄_ε`. The limit-trajectory selector margins (`pa:bit-margins`) have negative time coefficients, hence increase monotonically. This proves the theorem for every compact constant set in (1/8,1/3) and the reversal for (1/3,1) (both nulls give the same `1|2|3, 12|3`; the N1 ratio argument uses that `f(r)/r` is strictly decreasing on the invariant interval [1/4,12] for ε ≤ 1/100). The frozen control is affine in `e^{-t}` in the limit, with crossings `x12=2(11−4c)/(17+32c)`, `x13=4(1−c)/(9+16c)`, `x23=2(4−11c)/(18−17c)`, giving the full sequence for c ∈ (1/8,1/3).

## Findings table

| ID | Status | What changed (file, label) / reason |
| :- | :- | :- |
| H05, D2 (ch13/14 parts), XC1-GAUGE-01, XC2-FRAME-01 (ch13/14 parts), D17-PA-01, D18-LP-02, S3-M02 | applied (D17-PA-01's classification modified) | ch13: intro; section retitled "A hidden-phase quotient with inherited records"; protocol paragraph now declares A_I and B_I per `inh:protocol-classes`; `pa:phase-parent` statement and proof: the erased phase is gauge relative to B_I and every algebra (`def:observable-algebra`), the diagnostic belongs to no algebra, (F4) fails; attributes stated (orbits of a free proper hidden-torus symmetry; separation fades, phase Fisher information ~e^{-2t}). 13:316 and 13:326 requalified. ch14: intro, section titles ("A transmitted angle", "A collective rather than primitive quotient", `sec:lp-collective`), 14:41 replaced ("separate action, outside the marked structure group … gauge relative to every observable algebra"), new (F4) paragraph after `lp:relative-identification`; every "physical" naming of ψ replaced ("common ψ rotation", "transmitted angle"); `ℓ_phys → ℓ_ψ`, `I_phys → I_ψ`. Companions c13:5, :74, :184-190 and c14:5, :7, :11, :23, :71, :77, :116, :137, :181, :192, :196, :202 aligned. D17-PA-01 proposed "(F4) holds; fibers inert and transient": not adopted, because under the repaired (F4) (probe rule B_I) the clause fails; its attributes are reported. No "physical", "nongauge" or "private" (for ψ) remains in the four files. |
| H08, XC2-FRAME-02 | applied | ch13: selector is now a family S_c (`pa:selector`); `pa:persistent-grouping` restated for every compact C ⊂ (1/8,1/3) against both nulls, with a new shadowing proof (`pa:bit-remainder`, `pa:bit-shadow`, `pa:bit-margins`); new `pa:selector-threshold` (proved): for compact C' ⊂ (1/3,1) full flow and both nulls give the same sequence, and for C ⊂ (1/8,1/3) the frozen control reproduces P. (F5) report paragraph: holds for constant sets inside (1/8,1/3), fails for sets meeting (1/3,1); endpoints are flow quantities; selector-free observable = agent 3's sign reversal (5ε/7 vs ≈−5ε, −ε); frozen control also reverses it (4ε/5). `pa:angular-grouping` restated over C with both nulls; remark transfers `pa:selector-threshold`. ch14 import paragraph gives the same verdict. c13/c14 aligned. |
| A-STD interface (N1/N2/N3, (F2) family, (F4), (F6a), recipe) | applied | ch13 null paragraph (N1 as instance of `inh:null-self-row`, N2, N3, readout convention); four-state, Bernoulli and angular verdicts against both nulls; ch13 final section rewritten as "Report against the formation standard" in recipe order; ch14 new final section of the same form. Classification: angular and ch14 constructions are single-stage selections with exact inheritance, (F4) shown to fail, (F6a) fails; Bernoulli and four-state have no compressive parent. |
| D17-PA-02, GAP01-S3-03, S3-M03 | applied | ch13 remark after `pa:nonflat-window`: curvature present at t=0, at ε=0 and along the nulls, prepared in A_1(0); window is a choice of interval; decay at fixed ε>0 stated as indicated by a reduced model, not proved. ch14 sentence after `lp:nonflat`; c13:254, c14:131 aligned. GAP01-S3-03's leading-order law ε* ≈ 0.525 e^{-T+/3} not added (numerical). |
| D17-PA-03, D18-LP-11, GAP01-S3-01, XC3-SCOPE-02, S3-M01, D05-OBJ-06 (ch13/14 parts), B-DYN items | applied | ch13:9 cites the Gibbs-reduced and chart dynamics of `def:dyn-per-agent-flow`, states that stability transfer between row conventions is unproved; `pa:angular-flow` proof verifies the sufficient Lipschitz conditions (peer output log density 2-Lipschitz; reconstructed) and concludes invariance from tangency plus uniqueness; `lp:native-field` likewise; c13:120, c13:34 aligned. |
| B-DYN "rename D^q_ii" | applied as disambiguation | ch13:9 now says these self costs are the evaluator and reference divergences (the D_ii of `inh:null-self-row`), not the vanishing identity-channel self divergences. Not renamed, because A-STD's `inh:null-self-row` uses D_ii for the same cost and a rename would desynchronize ch05. |
| D17-PA-04, S3-X04 | deferred (improvement needing a proof) | ch13 after `pa:four-grouping`: numerical persistence noted as unproved. |
| D17-PA-05, S3-X34 | applied | `pa:geometry-inheritance` restated in two parts with proof (θ-score orthogonal to retained and orbit scores; `pa:cost-isometry` now carries the extra term); cites `geo:retained-connection`, `geo:minimal-contraction`; c13:231 corrected. |
| D17-PA-06, S3-L01 | applied | `pa:nonflat-window` proof and c13:222: history and conditioning-normalizer scores vanish identically in the witness (reconstructed: generative predictive kernel reads only the peers' initial public odds, context independent). |
| D17-PA-07, S3-L02 | applied | ch13 final section: (F6a) fails, grades stated, four-state (F2) finite-window and no compression. |
| D17-PA-08, S3-L03 | applied | ch13: inert law-dependent part of K[p] stated; "transfers a conditional channel" and "reciprocal condition" removed; c13:42 rewritten without K, W; c13 presentation label r → g. |
| D17-PA-09, S3-L04 | applied in part | `pa:phase-parent`: equivariant quotient of `inh:equivariant-descent`, history induction after `lem:boundary-equivariance`, BarnettCrutchfield2015, LarsenSkou1991. `inh:canonical-parent` not cited: whether this parent is the coarsest one was not checked (b_i is retained but does not enter the full rows). |
| D17-PA-10, S3-X05 | deferred (extension) | — |
| GAP01-S3-02, S3-X06 | applied in part | Bernoulli proof rewritten with the Lyapunov/shadowing argument and explicit remainder; ball invariant for ε ≤ 1/20. The lens's bisection constants (ε0 = 0.028, T = 7.82 etc.) not adopted (floating-point, not validated). Angular proof kept, with a reduction remark. c13:140-146 rewritten. |
| GAP01-S3-04, S3-L05 | applied | ch13 window proof and ch14 `lp:nonflat`: analytic dependence on parameters, `\citep[Chapter V]{Hartman2002}`. |
| GAP01-S3-05, S3-L06, D18-LP-14 | applied | ch14 import: 0<ε≤1/100, \|A_i(0)\|≤1/20, \|C_i(0)\|≤ε³, δ max\|b_i\|≤1/20; `lp:nonflat`: a_+, d < 1/20; c13:233 "small a,d<1/20". |
| GAP01-S3-06 | applied in part | qualitative note (O(η) neighborhood of a near-uniform unstable equilibrium); numerical triple not adopted. |
| GAP01-S3-07, S3-X35 | applied | remark after `pa:angular-grouping`. |
| GAP01-S3-08, S3-X07 | deferred (extension) | — |
| S4-M17, XC2-FRAME-07, XC6-PROG-14 (ch13 part) | applied | DeGroot1974, FriedkinJohnsen1990, JadbabaieEtAl2012 after `pa:selector-threshold`, with what is new; c13 cites FriedkinJohnsen1990. Items (2)-(5) of XC6-PROG-14 concern ch03 (B-DYN/C-SUM). |
| S4-L26, XC7-IMPL-09 | applied in part | closed-form entrance time of the limit trajectory for c=1/4 (t ≈ 3.83, root of 5/28 − 3y/8 − 13y^7/14); explicit ε0 = 1/20 for ball invariance in the Bernoulli proof. No validated ε for the windows. |
| S4-L33, XC5-LIT-13 (ch13 part) | applied | Zamir1998 and `geo:joint-fisher` at `pa:joint-record-geometry`; c13 names the chain rule. |
| GAP04-S4-02, S4-M10 | applied (hypothesis; regularity proof not supplied) | ch13 and ch14 protocols require bounded expected transactions on compact patches and state why (non-DQM otherwise); `pa:joint-record-geometry` notes the DQM presupposition (`geo:dqm`). The proposed regularity theorem was not written (no checked proof). |
| GAP04-S4-05 | applied | Borel policies and jointly Borel record densities, citing `def:adaptive-history`, in both protocols; c13 aligned. |
| D18-LP-01 (E05) | deferred (extension) | Only the analytic-preparation remark is stated, inside `lp:local-parent-open`. |
| D18-LP-03 | applied | new display `lp:decay` after `lp:cone`; c14 aligned. |
| D18-LP-04, S3-M04 | applied | freezing-method attribution in `lp:causal-flow`; `lp:causal-coherence` now `\corollaryheading` with "the hypothesis is exactly the condition that makes the sample map independent of context"; c14 aligned. |
| D18-LP-05, S3-M06 | applied | `lp:path-memory-obstruction` retitled "Order-zero obstruction for rotation-invariant value-path retentions" with the reduction paragraph; c14 "unlimited temporal storage" removed or qualified. README/AUTHORING are C-SUM's. |
| D18-LP-06 | applied (ch14 part) | `lp:causal-coherence` says sufficient, not shown necessary; the identity of `lp:phase-field-defect` shows the defect can vanish at one time. Abstract/README/outlook are C-SUM's. |
| D18-LP-07, S3-M07 | applied | new `lp:rich-class`; all three obstructions cite 𝒳_rich; "positive chart" defined there. |
| D18-LP-08, S3-M08 | applied | new `lp:local-parent-open` with known answers. |
| D18-LP-09, GAP10-S3-07, S3-X08 (B-CTX item) | applied | new `lp:phase-field-defect` (history decomposition `lp:history-decomposition`, defect identity `lp:phase-defect-identity` with history-dependent Θ), proof reconstructed (von Mises identities, conditional orthogonality); consequence: causal parent inherits the connection class for every preparation. |
| D18-LP-10, XC5-LIT-11, S4-L31 | applied | RowleyEtAl2003, BudanurEtAl2015 at `lp:closed-reference`; BeynThummler2004, RowleyMarsden2000 at `lp:causal-flow`; cross-references `open:anchor`, `open:moving-flow`; what is new stated. |
| D18-LP-12, S3-L08 | applied | orthogonality sentence in the proof of `lp:jet-obstruction`. |
| D18-LP-13, S3-L09 | applied | ch14 protocol: every transaction draws fresh samples, no reuse; c13: reused samples are part of the retained state. |
| D18-LP-15, S3-L10 | applied | D(t) → 𝒟(t), v → a_t, sum-example D → N and v → ϖ, "private" → "transmitted"/"ψ", "published … argument" → reference; c14 note on cp:curvature's B(t). |
| D18-LP-16, S3-L11 | applied (weakness stated) | sentence after `lp:profile-obstruction`; kept as proposition (rule: do not delete results). |
| XC1-GAUGE-03, S4-M12 | applied | new remark `lp:group-dependence` (enlarged group S^1_marked × S^1_ψ makes ℓ_ψ an orbit score; obstructions refer to the declared marked group); closing paragraph cites it; c14 aligned. Reconstructed: the ψ component of the comparison velocity absorbs the phase gradient. |
| XC4-CONS-09 (iii) | applied | ch13:5 "\cref{ch:fusion} to \cref{ch:open-formation}". |
| XC4-CONS symbol collision G (memory field, ch14) | applied | `\dot m = Γ(…)` in `lp:causal-jet-obstruction`. |
| XC4-CONS-08 (K[p], ℓ notations in ch13/c13) | deferred | cross-chapter name unification belongs to the lead or C-SUM. |
| XC6-PROG-04 (ch13 part) | applied | both nulls stated; readout convention noted as met trivially (selector reads public laws only). |
| XC6-PROG-07 | deferred (extension) | — |
| D10-FORM-05/-06 (ch13 parts) | applied via A-STD's standard | verdicts are on [0,∞) for the persistent constructions; four-state verdict holds on any horizon (nulls never select). |

Rejected: none outright. Partial non-adoptions: D17-PA-01's "(F4) holds" (superseded by the repaired (F4)); GAP01-S3-02/-06 numerical constants (not validated); D17-PA-09's `inh:canonical-parent` citation (coarsest-parent claim unchecked); B-DYN's rename (kept symbol, disambiguated).

## BIB ENTRIES

All verified against Crossref records on 2026-10-29 (DOI lookups); Hartman's Chapter V locator is from memory.

```bibtex
@article{FriedkinJohnsen1990,
  author  = {Friedkin, Noah E. and Johnsen, Eugene C.},
  title   = {Social influence and opinions},
  journal = {The Journal of Mathematical Sociology},
  volume  = {15},
  number  = {3--4},
  pages   = {193--206},
  year    = {1990},
  doi     = {10.1080/0022250X.1990.9990069}
}
@article{JadbabaieEtAl2012,
  author  = {Jadbabaie, Ali and Molavi, Pooya and Sandroni, Alvaro and Tahbaz-Salehi, Alireza},
  title   = {Non-{B}ayesian social learning},
  journal = {Games and Economic Behavior},
  volume  = {76},
  number  = {1},
  pages   = {210--225},
  year    = {2012},
  doi     = {10.1016/j.geb.2012.06.001}
}
@article{BarnettCrutchfield2015,
  author  = {Barnett, Nix and Crutchfield, James P.},
  title   = {Computational Mechanics of Input--Output Processes: Structured Transformations and the $\epsilon$-Transducer},
  journal = {Journal of Statistical Physics},
  volume  = {161},
  number  = {2},
  pages   = {404--451},
  year    = {2015},
  doi     = {10.1007/s10955-015-1327-5}
}
@article{RowleyMarsden2000,
  author  = {Rowley, Clarence W. and Marsden, Jerrold E.},
  title   = {Reconstruction equations and the {K}arhunen--{L}o{\`e}ve expansion for systems with symmetry},
  journal = {Physica D: Nonlinear Phenomena},
  volume  = {142},
  number  = {1--2},
  pages   = {1--19},
  year    = {2000},
  doi     = {10.1016/S0167-2789(00)00042-7}
}
@book{Hartman2002,
  author    = {Hartman, Philip},
  title     = {Ordinary Differential Equations},
  edition   = {Second},
  series    = {Classics in Applied Mathematics},
  volume    = {38},
  publisher = {Society for Industrial and Applied Mathematics},
  address   = {Philadelphia, PA},
  year      = {2002},
  doi       = {10.1137/1.9780898719222}
}
```
(FriedkinJohnsen1990 is already in `item7_refs.bib`; same record. The Classics series volume number 38 is from memory, not in the Crossref record.)

## NEW OR CHANGED LABELS

New (all unique across the three documents, checked): `pa:bit-remainder`, `pa:bit-shadow`, `pa:bit-margins` (equations), `pa:selector-threshold` (proposition) in ch13; `sec:lp-collective`, `lp:decay`, `lp:history-decomposition`, `lp:phase-defect-identity` (equations/section), `lp:phase-field-defect` (lemma), `lp:group-dependence` (remark), `lp:rich-class` (definition), `lp:local-parent-open` (open problem) in ch14.

Changed content under unchanged labels: `pa:selector` (now the family S_c), `pa:persistent-grouping` (retitled; constant set and both nulls), `pa:angular-grouping` (constant set, both nulls), `pa:geometry-inheritance` and `pa:cost-isometry` (two-part statement, extra term), `pa:phase-parent` ((F4) fails), `lp:causal-coherence` (theorem → corollary), `lp:path-memory-obstruction` (retitled), `cp:attraction` (now the shadowing bound), `cp:selector` (threshold c). Symbols: `ℓ_phys → ℓ_ψ`, `I_phys → I_ψ` (label `lp:physical-phase-inertia` kept to avoid breaking references). No label removed or renamed; no `pa:`/`lp:` label is referenced outside the four files.

## FOR OTHER OWNERS

- **C-SUM**: merge the five BibTeX entries. frontmatter:11 ("physically noninjective parent", "proves persistent interaction-dependent grouping"): the grouping holds against both nulls for thresholds in (1/8,1/3) and fails above 1/3, the frozen control reproduces it, and the parent quotients a phase gauge relative to every declared algebra, so it fails (F4). frontmatter:13: "requires initial collective-phase coherence" → "is proved under"; "rich open preparation class" → `lp:rich-class`; obstructions refer to the declared marked group (`lp:group-dependence`). Outlook:12, :14, :16 and README:28, :30 likewise; README "unlimited temporal storage" qualify (D18-LP-05). AUTHORING.md:75 ("erased physical phase"), :77, :81 ("One common nongauge physical phase is lost collectively" → "one common orientation, gauge relative to every declared algebra, is erased by the whole-block retention"), :85 ("allows unlimited temporal memory" → order-zero on fixed-record branches). Notation/glossary: S_c and constant sets, ℓ_ψ, I_ψ, 𝒳_rich, 𝒟(t), a_t, Γ.
- **B-DYN**: XC6-PROG-14 items (2)-(5) (multi-agent active inference, EM/Richardson-Lucy, partitioned VI, Besag compatibility) are ch03 attributions.
- **A-STD**: none required; `inh:formation-open` and 05:2017-2020 already match the new ch13/14 verdicts.

## AUTHOR DECISIONS NEEDED

1. The constant set C of the ch13/ch14 ratio selector. (F5) holds against both nulls for every compact C ⊂ (1/8,1/3) and fails for any C meeting (1/3,1); the natural range of a ratio threshold would make it fail. The text reports both; which C is declared decides the clause. This depends on A-STD's provisional decision 2.
2. The frozen-environment control reproduces P (and the sign reversal) for every constant where (F5) holds. If P ≠ P^fr becomes a clause or grade (A-STD decision 4), the ch13 and ch14 (F5) verdicts fail at every constant.
3. A probe rule that admits hidden-angle queries for singletons and withdraws them after the handoff would make the ch13 retention pass (F4) while deleting a probe at the handoff. The text declares no such rule; whether `inh:protocol-classes` should forbid it is the author's call.
4. The structure group for ch14's geometry: with the common ψ rotation in the group, the obstructions and the coherence requirement lose their force (`lp:group-dependence`); the text gives no internal reason for the smaller group.
5. Strong grade (A-STD decision 3): under the redefinition it holds for ch13 in a degenerate sense (admitted class = the theorem's initial set); the text says so.
