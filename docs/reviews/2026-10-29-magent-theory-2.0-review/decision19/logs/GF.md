# GF: formation under the global free energy (decision 19 test case)

Manuscript files touched: `chapters/05_global_formation.tex` (new) and three permitted regions of `chapters/05_inheritance.tex`. Nothing compiled, nothing committed. `MAgent_Theory_2.0.tex` not edited; the lead adds `\input{chapters/05_global_formation}` (placement: after `\input{chapters/05_inheritance}`, before `06_fusion`, since ch05 now cites `ch:global-formation`, `gf:first-event`, `gf:structure-factor`, `gf:beyond`).

## Construction

Global model (`gf:model`): agents' contents y_i, nested structure Pi = (Pi1, Pi2) on a declared finite catalog with positive prior (`gf:catalog`), second-level latents w_C ~ r, first-level latents z_B ~ k(w_C, .), children y_i ~ p_i(.|z_B), single-agent records l_i(o_i|y_i); block kernels independent of the block. No comparison records: coupling already lives in the shared latents, and without two-agent factors the agent updates are order-free and the nulls delete exactly the sharing of nu = (Pi, w, z). Records are static configuration coordinates.

Recognition (`gf:recognition`): structured family Q_nu x prod Q_i (non-agent block unfactorized), and the committed family delta_Pi x Q(w,z|Pi) used for the dynamics (soft family's block retention is not exact, `gf:soft-remark`).

Dynamics (`gf:sweep`): coordinate descent of the global free energy in sweeps (nu block, then all agents); time = sweep index; selector = mode of the structure factor at the agents' current laws (derived, no constants). Gaussian instance: closed-form sweep m' = a o + b K_Pi m (`gf:gaussian-sweep`), a unit natural-gradient step of the collapsed free energy; fixed point C(C+R)^{-1} o.

Instance (`gf:declared`): N=4, s^2=1, rho^2=1/4, tau^2=1/10, sigma^2=4, all 60 nested structures, uniform prior; center o=(2.9,3.1,4.4,4.6), m0=(-9.2,-3.0,4.2,10.4).

## Results (labels)

- `gf:structure-factor` (general law): Q*(Pi) propto pi(Pi) exp(-F_Pi), F_Pi = -log pseudo-evidence of the expected child kernels; committed minimizer = mode; for exponential-type children F_Pi depends on agents only through block sums t_B, so the second-level posterior given Pi1 = A depends only on (t_B) (static inheritance, no conjugacy needed).
- `gf:gaussian-structure`: F_Pi = -log N(m; 0, C_Pi) + Pi-free; selector = Bayesian model selection on the means. Stationary relation: Pi_f = Pihat(E[y|o,Pi_f]) (self-consistency), not the MAP of the records; equal in the limit R -> 0.
- `gf:gaussian-sweep`, `gf:persistence`: closed form, contraction (factor <= max b_i in the S^{-1} norm), exact posterior means at fixed structure (Gaussian feature), invariant-ball persistence.
- `gf:inheritance` (Gaussian): retention R_B = (lambda_B, precision-weighted block mean, weighted record mean); (L), (D), (B), (C) on any sweep-invariant set where the first level stays A; parent sweep `gf:parent-sweep`.
- `gf:beyond`: exact (D) iff the summed agent update is a function of a summed record statistic; for two identical agents with continuous update this forces an affine update (Jensen equation).
- `gf:top-down`: parent acts through the geometric mean of the child kernel; exact cavity identity; Gaussian means agree with exact, mean-field variance eta_i <= exact.
- `gf:type-closure`: parent is a model of the same class (child noise 1/lambda_B, record noise lambda_B^{-1} b_B/a_B, singleton first-level catalog); type-closed with coordinate descent as the dynamics.
- `gf:first-event` (general, block-homogeneous kernels, no multi-agent records): while the structure is finest, full flow = label-preserving null; the first selection event is always a null event.
- `gf:instance`: full P = finest (0), Pi_sep (1), Pi_join (2 on); lp null identical; rn finest, Pi_join; parent: Pi_join only self-consistent structure, union from every parent state selecting Pi_sep; union retention exact.
- `gf:regrouping` (observation at one point), `gf:open`.

## Verdict table

| Clause | Verdict | Basis |
|---|---|---|
| Objective / dynamics / chart | global free energy; sweep coordinate descent on committed family; Gaussian parameter manifold x record coordinates | `gf:sweep`, `gf:declared` |
| (F1) | holds, full grade | U open, Pihat = finest at sweep 0 |
| (F2) | holds; derived selector, C has no constant; persistence = invariant block relation (ball certificate) | `gf:structure-factor`, `gf:persistence` |
| (F3) | holds, both stages | `gf:inheritance`, `gf:instance`(iv) |
| (F4) | holds; persistent (record contrasts, a*(o_i-o_j)); part from m0 transient; persistent part of symmetry type only on the first-level-A region; with records as constants only the transient part remains | report paragraph |
| (F5) | FAILS: split; lp null reproduces P exactly on U; rn skips Pi_sep (difference by coincident events) | `gf:first-event`, `gf:instance`(ii) |
| selector-free observable | within-block contrasts contracted by a/kappa = 214/255 vs both nulls; limits differ | `gf:contraction` |
| P vs P^fr | equal (120 sweeps at the center, no tail certificate) | gf_instance.out |
| grades | strong n/a; amplifying fails (states coincide with lp at the first event) | |
| (F6a) | holds | `gf:instance`(iii) |
| (F6b) | FAILS: union from every counterfactual handoff (records held); falsifier tested: stationary second-level outcome independent of handoff on all of the class; parent closes | `gf:instance`(iii) |
| Classification | two-stage selection with exact inheritance | |

Caveat recorded in the chapter: the structure posterior is nearly flat (selected structure mass 0.052 under the structure factor at the limit, 0.050 under the exact posterior of the records); the committed family converts small score gaps (min 0.0142 nats) into selection.

## Conventions added to chapter 05 (`inh:null-flow`, global part)

General rules for models with non-agent coordinates nu: both nulls marginalize multi-agent records and keep single-agent ones; label-preserving null = product over agents of the marginal on (y_i, nu) with a private copy of nu per agent (priors kept); renormalized null = private copies integrated out; frozen-environment control = private copy per agent updated by the full-flow rule with other agents' laws held at initial values; selectors reading a non-agent block read the original block minimizer at the control's agent laws; controls use the full flow's recognition family and schedule. The sentence "No construction ... no further convention is developed for it here" was replaced by a pointer to `ch:global-formation`. Summary paragraph: two sentences with this verdict. `inh:formation-open`: rewritten to state what the global construction achieves and the remaining routes.

## Open items

Nested two-stage paths with interaction necessity against lp and parent multistability (only in the regime of several self-consistent structures; the first event never contributes); block-dependent priors or multi-agent records; dynamics of the structured (soft) family; dynamical inheritance for non-affine updates; openness of the regrouping observation; tail certificate for the frozen control.

## Scripts (V/decision19/scripts/GF/), run single-threaded, foreground

- `gf_model.py`, `gf_controls.py`: helpers.
- `gf_symbolic.py` -> `gf_symbolic.out`: exact rational/symbolic checks of the collapsed step, conditional-mean identity, fixed point, Pi-free variance term, block decomposition, type closure (heterogeneous noises), finest = lp null, variance inequality. All True.
- `gf_instance.py` -> `gf_instance.out`: histories (full, lp, rn, frozen), margins, ball certificates, 200-sample perturbation check at eps up to 1e-2 (0 changes), observables, posterior masses, parent reproduction (error 9e-16), parent self-consistency, counterfactual grid (11486 states, union from all within 3 sweeps), regrouping example.
- `gf_search.py`, `gf_diag.py`, `gf_scan_gap.py`, `gf_search_bistable.py`, `gf_search2.py`, `gf_pick.py`: exploratory searches used to find the instance (not evidence for the theorems).
- `gf_labels.py`: label uniqueness and reference check (no missing refs, no duplicates, no gf labels elsewhere).

## BIB ENTRIES (verified against Crossref records; not added to references.bib)

```bibtex
@article{KassRaftery1995,
  author  = {Kass, Robert E. and Raftery, Adrian E.},
  title   = {Bayes Factors},
  journal = {Journal of the American Statistical Association},
  volume  = {90},
  number  = {430},
  pages   = {773--795},
  year    = {1995},
  doi     = {10.1080/01621459.1995.10476572}
}
@article{LindleySmith1972,
  author  = {Lindley, D. V. and Smith, A. F. M.},
  title   = {Bayes Estimates for the Linear Model},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume  = {34},
  number  = {1},
  pages   = {1--18},
  year    = {1972},
  doi     = {10.1111/j.2517-6161.1972.tb00885.x}
}
@article{CeleuxGovaert1992,
  author  = {Celeux, Gilles and Govaert, G{\'e}rard},
  title   = {A classification {EM} algorithm for clustering and two stochastic versions},
  journal = {Computational Statistics \& Data Analysis},
  volume  = {14},
  number  = {3},
  pages   = {315--332},
  year    = {1992},
  doi     = {10.1016/0167-9473(92)90042-E}
}
@inproceedings{HellerGhahramani2005,
  author    = {Heller, Katherine A. and Ghahramani, Zoubin},
  title     = {Bayesian Hierarchical Clustering},
  booktitle = {Proceedings of the 22nd International Conference on Machine Learning},
  pages     = {297--304},
  year      = {2005},
  doi       = {10.1145/1102351.1102389}
}
```
Reused keys: NealHinton1998, DempsterLairdRubin1977, Bishop2006.

## NEW LABELS

ch:global-formation; gf:sec-model, gf:catalog, gf:model, gf:prior, gf:free-energy, gf:covariance, gf:recognition, gf:structure-factor, gf:pseudo-evidence, gf:structure-posterior, gf:block-statistic, gf:gaussian-structure, gf:gaussian-score; gf:sec-dynamics, gf:sweep, gf:agent-update, gf:gaussian-sweep, gf:mean-map, gf:fixed-point, gf:persistence; gf:sec-inheritance, gf:inheritance, gf:retention, gf:parent-sweep, gf:beyond, gf:soft-remark, gf:top-down, gf:cavity, gf:type-closure; gf:sec-controls, gf:first-event; gf:sec-report, gf:declared, gf:instance, gf:contraction, gf:regrouping, gf:open.
