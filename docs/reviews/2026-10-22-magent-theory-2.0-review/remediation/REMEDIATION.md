# MAgent Theory 2.0 remediation, 2026-10-22

The author accepted two decisions after the review (`../REPORT.md`): agents hold beliefs and generative models, with both a global free energy and per-agent free energies studied; and the strengthened formation standard. The remediation implemented them and the review findings across the three documents in `Research/manuscripts/MAgent_Theory_2.0/` on the local Research branch `magent-2.0-remediation` (uncommitted).

## Lead changes

`AUTHORING.md` was rewritten (beliefs and generative models, the three objectives, the formation standard, regularity and gauge conventions, rescoped constructions). Chapter 1 gained `sec:generative-models` (`def:generative-model`, `eq:agent-free-energy`, `def:shared-model`) and a dressing-field paragraph. Chapter 3 was retitled "Free Energies and Coupled Dynamics of Agent Laws" and gained `prop:dyn-global-local`, the per-agent mixture-source model and free energy (`def:dyn-per-agent`, `prop:dyn-per-agent`, `def:dyn-per-agent-flow`), the no-go `prop:dyn-no-global-vfe`, `def:dyn-record-interrupted` and `op:dyn-three-pictures`. The per-agent identity, the mean-field coordinate step and the no-go mixed derivative were checked in `scripts/fe_identities.py` (residual 0.0; F decreased 0.4405 to 0.3115; derivative -0.72455 at (0.1, 0.3)). Chapter 5's last section became the formation standard (`inh:null-flow`, `inh:recursive-formation` with (F1) to (F6), the (F6a)/(F6b) split, single- and two-stage selections with exact inheritance, `inh:formation-open`). The abstract, outlook and README were rewritten. Bibliography: 7 lead entries plus 100 agent entries merged (`scripts/merge_bib.py`).

## Chapter agents

Eight agents remediated disjoint file sets under `BRIEF.md`; their per-finding logs are in `logs/` and their checks in `scripts/<agent>/`. Approximate counts from the logs: R-FOUND about 55 applied and 12 deferred; R-GEOM 29 applied, 4 partial, 1 rejected; R-INH about 40 applied, 7 deferred; R-FUS 39 applied, 10 partial, 1 rejected; R-ERA 26 applied, 1 deferred; R-BND 19 applied, 2 deferred, 1 rejected; R-CND 31 applied, 2 deferred, 2 rejected (the single-linkage claims; centroid linkage confirmed against scipy, 0 mismatches in 1000 trials); R-OPEN 44 applied, 9 partial, 6 deferred, 1 rejected.

Classification against the standard after remediation: chapter 6 fusion meets (F5) and (F6a), fails (F4) and (F6b); chapter 8 erasure is a single-stage selection; chapter 10 supplies its hierarchy (no selection claim); chapter 11 Wishart fails (F5) and (F6b); chapter 12 open-law meets (F4) and (F6a), does not establish (F5), and (F6b) is unverified. All are selections with exact inheritance; none is a recursive formation.

## Build

All three documents compile with no undefined references or citations (main 184 pages, previously 68; supplement 14; companion 63). Two overfull boxes of 9 and 21 pt remain in the main text. A backslash-collapse defect from shell writes was found in the companion (`\\` written as `\`) and repaired; `scripts/scan_backslash.py` and `scripts/scan_control_space.py` found no other instance.

## Not verified

The remediation results are agent-applied and unadjudicated. Many of the 107 new bibliography entries were written from memory; the logs flag DOIs and page ranges to confirm (O'Neill, Procesi, Reinhart, Zamir, Fels and Olver, Carreira-Perpinan, Geshkovski et al., Abbas, Lojasiewicz, and the ERA set). New propositions added by agents (for example `fus:sync`, `erase:pair-convergence`, `form:kappa-independence`, `open:all-partitions`, `prop:dyn-global-flow`, `inh:projectable`) were checked by their authors only.

## Author decisions outstanding

1. Structure group: one G acting on both sectors through two representations (current chapter 1 default) or G_q x G_s for independent sector frames.
2. Global-in-time well-posedness: add a general Fisher-flow proposition in chapter 3 for chapters 7, 9, 10 and 12 (chapter 7 currently local only).
3. Whether to import the Hopf-bundle holonomy record witness as an appendix.
4. Whether chapter 12 keeps the tied belief/model rotation group, on which its curvature depends.
5. Whether unproved numerical observations (chapter 12 null control and constants, chapter 8 switch times) may stay in the text as labeled remarks.
6. Whether to move `form:mean-field` and `form:scalar-closure` (chapter 11) and `lem:boundary-equivariance` (chapter 10) into chapter 5 as general results.
7. Whether to prove the heterogeneous-coupling erasure variant (currently an unproved remark).
8. The main manuscript has nearly tripled in length; whether to move secondary propositions to the supplement.
