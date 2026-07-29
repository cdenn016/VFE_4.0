# Blue Opening — h7-raw-factor-vs-grouped-kl-trace

## Steelman (opposing position)

The whitepaper's local ELBO is most naturally reported as emissions minus positive initial, source, and transition KL terms, so H7 should expose those scientifically meaningful grouped quantities directly instead of preserving lower-level expected-log-factor bookkeeping.

## Position

The claim is defensible. `CompleteLanguageELBOFactorTrace` is a provenance-preserving additive trace of the raw ELBO representation
\[
\mathcal L(q)=\mathbb E_q[\log p(x,z)]+H(q).
\]
H7 may derive a second, grouped representation from the complete \(p\) and \(q\) laws, but it must not rename an existing \(\mathbb E_q[\log p]\) slot as a KL while continuing to add the entropy that the KL already contains.

## Evidence

- The standard identity is \(\mathcal L(q)=\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q(z)]\). If \(p(x,z)=\prod_j f_j(x,z)\), the raw trace is \(\sum_j\mathbb E_q[\log f_j]+H(q)\). A grouped term satisfies
  \[
  -\mathrm{KL}(q_C\Vert p_C)
  =\mathbb E_{q_C}[\log p_C]+H(q_C).
  \]
  Grouping therefore consumes the matching entropy contribution; it is not a semantic relabeling of \(\mathbb E_q[\log p_C]\). This follows from [Blei, Kucukelbir, and McAuliffe (2017), §2.2](https://arxiv.org/abs/1601.00670), [Bishop (2006), Chapter 10, equations (10.2)–(10.4)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), and the equivalent reconstruction-minus-KL form in [Kingma and Welling (2014), §2.2 and Appendix B](https://arxiv.org/abs/1312.6114).

- The repository already implements that distinction. `vfe4/objective/h5_complete.py:894-929` evaluates the initial expected generative log density; `:930-957` evaluates source terms as \(\sum q\log p\); `:958-1006` evaluates expected transition log densities; and `:1007-1009` plus `:1088-1108` computes recognition entropy separately. These are the raw primitives of the first canonical form, not positive KL values.

- The trace contract preserves those primitives. `vfe4/types/h6.py:2481-2486` requires every stored factor value to add exactly to the reported ELBO. `vfe4/objective/language_elbo.py:1131-1143` copies the ordered factor identities and values into `CompleteLanguageELBOFactorTrace`, while `:1163-1173` requires the exact 13-slot, \(T=2\) post-H6 trace. H7 therefore inherits the producer's meanings as well as its total.

- The current H7 mapping does not derive KLs. `vfe4/objective/h7_covariance.py:1410-1424` associates raw source and transition slots with names ending in `_kl`, and `:1430-1433` obtains each value only by summing those preserved slots. By contrast, the independent oracle constructs \(K_0\) from both complete laws (`verification/mp_oracles/h7_covariance.py:3742-3747`), computes source log-ratios and conditional Gaussian KLs (`:3770-3787`), and excludes the separately reported joint entropy when it forms the grouped local ELBO (`:3801-3807`). That is the required derivation boundary.

The strongest attack is that grouped KLs are the cleaner, coordinate-invariant scientific observables, whereas isolated differential-entropy and expected-log-density terms can shift under reparameterization. That attack favors publishing a grouped diagnostic, but it does not justify changing the meaning of a provenance-bound additive trace. Retaining both views resolves the issue: the raw trace audits factor accounting, and the independently derived grouped trace audits the complete-law identity.

## Falsification conditions

This position is wrong if the live producer is shown to define the 13 stored slots as signed grouped KLs already, with every matching entropy slot marked nonadditive; if a new trace schema gives the regrouped values new identities and removes every absorbed entropy contribution from its sum; or if an explicit derivation proves that the proposed one-to-one relabeling preserves the complete ELBO for every admitted law without counting any \(\mathbb E_q[\log q]\) contribution twice. None of those conditions holds in the cited implementation.
