# Evidence Pack — h7-raw-factor-vs-grouped-kl-trace

## Active config

This is a source-design adjudication before H7 calibration or trial/control
execution. No runtime gate or authorization setting is active. The frozen H7
protocol remains source-only and fail-closed.

## Code references

- `vfe4/types/h6.py:2436` — `H6LanguageElboTerms` owns the canonical 13-slot
  inventory and requires the slot values to add to the reported ELBO.
- `vfe4/objective/language_elbo.py:1107` —
  `CompleteLanguageELBOFactorTrace` wraps that exact additive inventory as an
  authoritative complete-objective trace.
- `vfe4/objective/h7_covariance.py:1399` — the current H7 evaluator maps the
  initial raw slot to \(K_0\), source/transition raw slots to grouped KL
  contributions, keeps explicit entropy slots, and then sums all 13 slots.
- `vfe4/objective/h5_complete.py:888` — the earlier complete-objective path
  treats raw slots as \(E_q[\log p]\) primitive contributions plus
  recognition entropy; their sum is the ELBO.
- `verification/mp_oracles/h7_covariance.py:3564` — the independent oracle
  evaluates conditional Gaussian KLs from the full laws.
- `verification/mp_oracles/h7_covariance.py:3735` — the oracle assembles the
  grouped local ELBO as emissions minus positive initial/source/transition
  KLs; joint entropy is a diagnostic and is not added a second time.
- `tests/unit/test_h7_complete_objective.py:137` — current H7 tests fabricate
  slot values and preserve totals by moving values between factors, so they
  do not establish scientifically authoritative slot meanings.

## Manuscript references

- `Manuscripts/vfe4_whitepaper/06_elbo_coordinate_updates.tex:163` — defines
  positive \(K_0\) and positive source/transition KL terms.
- `Manuscripts/vfe4_whitepaper/06_elbo_coordinate_updates.tex:250` — writes
  the complete ELBO as expected emission terms minus those positive KLs,
  without a separately added recognition-entropy term.

These manuscript equations are the construction under evaluation, not
external authority.

## Canon excerpts

- Variational-inference canon defines
  \(\mathcal L(q)=E_q[\log p(x,z)]-E_q[\log q(z)]
  =E_q[\log p(x,z)]+H(q)\).
- The equivalent posterior-gap form is
  \(\log p(x)=\mathcal L(q)+\mathrm{KL}(q(z)\|p(z\mid x))\).
- When \(p\) and \(q\) factorize compatibly, grouped positive KL penalties
  may replace the corresponding raw \(E_q[\log p]\) and entropy pieces, but
  they cannot be added alongside the same entropy contributions without
  double counting.
- External anchors for the adjudicators: Bishop, *Pattern Recognition and
  Machine Learning*, chapter 10; Blei, Kucukelbir, and McAuliffe (2017),
  “Variational Inference: A Review for Statisticians,” ELBO derivation;
  Cover and Thomas, *Elements of Information Theory*, Gaussian entropy and
  KL identities.

## What this evidence does NOT settle

- Whether the exact two-slot entropy partition proposed for receiver times
  1 and 2 is the only valid partition.
- Whether fixture-produced traces should keep the phrase “post-H6” or gain an
  explicit provenance discriminator.
- Whether the current H7 slot relabeling can be repaired locally or requires
  a schema revision.
