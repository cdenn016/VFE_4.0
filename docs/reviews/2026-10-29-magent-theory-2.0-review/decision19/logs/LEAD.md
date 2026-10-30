# Decision 19: lead log (2026-10-30)

The author decided report decision 7 (chat decision 19): the global free energy is primary for formation, the per-agent free energies describe agent dynamics, and the reciprocal potential is a configuration-level device read globally only through an exact identification. A formation result is re-proved under the global free energy before the reciprocal track is extended. This log covers the lead's edits. The partition construction is logged in `GF.md`.

| Item | Status | Change |
| :- | :- | :- |
| E04 (XC6-PROG-09), linear-Gaussian bridge | applied | `thm:dyn-three-limits` in `chapters/03_dynamics.tex`, after `prop:dyn-per-agent-not-global`, with `eq:dyn-three-limits`. Parts: (i) reciprocal stationary means `(I+L_W)^{-1}a`, `W=beta+beta^T`; (ii) sample-message model `M_n` at `T=1/n` with `T F(N(mu,tau I)) = E(mu) + c_n(tau)` for every `n` and `tau`, exact posterior `N(mu_rec, T(I+L_W)^{-1})`, mean-field means at the same point; (iii) both per-agent instances give `(I+L_out)mu=a`, equal to the reciprocal map iff per-agent rows are `beta+beta^T`; (iv) per-agent map is a global posterior-mean map iff the rows are reversible (Kolmogorov cycle condition), with `lambda_ij = varpi_i beta_ij`, and then the frozen-source flow is the restricted Fisher flow of that global free energy on the slice `N(mu_i, 1/varpi_i)`. Remark after the proof: the source-role force is the back-action of the receiver's comparison record on the compared latent; two-agent gaps `(a-b)/5` against `(a-b)/3`; scope limited to the slice, with the free-variance counterexample (gap about 0.2065, variances widened to about 1.041). |
| op:dyn-three-pictures | applied | Rescoped to what remains beyond the mean slice. |
| ch03 opening | applied | Paragraph fixing the standing of the three objectives, citing `ch:global-formation`. |
| AUTHORING.md | applied | Standing of the objectives recorded in the three-objectives section; "No objective is declared primary" replaced by the scope of `thm:dyn-three-limits`; `06_fusion` to `12_open_formation` labelled surrogate results; new formation work posed under the global free energy first. |
| Outlook | applied | "no objective is declared primary" replaced by the primacy statement and the three-limit summary. |
| Abstract | applied | Two sentences: the mean-slice identification and the primacy statement. |

Checks: `scripts/lead/three_limits.py` (sympy, exact): stationarity systems (i) and (iii), the free-energy identity (ii) for three agents with general rows, the reversible-row identity `Lambda_0 (I+L_out) = Lambda_0 + L_lambda`, asymmetry for the forward three-cycle, the two-agent gaps, and the free-variance stationary point by `nsolve`.

BIB ENTRIES added to `references.bib`: Gershgorin1931, Kolmogorov1936 (Math. Ann. 112, 155 to 160, doi 10.1007/BF01565412), Kelly1979 (Wiley).

NEW LABELS: `thm:dyn-three-limits`, `eq:dyn-three-limits`.
