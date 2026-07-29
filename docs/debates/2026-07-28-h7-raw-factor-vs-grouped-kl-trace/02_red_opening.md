# Red Opening — h7-raw-factor-vs-grouped-kl-trace

## Steelman

If the 13 additive slots are already defined as disjoint
\(E_q[\log p]\) contributions plus a disjoint partition of \(H(q)\),
then relabeling any generative-factor slot as a KL term while retaining
the nonzero entropy slots misstates the operands and double-counts the
\(-E_q[\log q]\) contribution.

## Position

The anti-double-counting part of the claim is sound, but the claim does
not establish that raw-factor semantics are the only canonical
representation or that the current implementation enforces them.
External variational-inference canon treats
\(E_q[\log p(x,z)]+H(q)\) and expected log likelihood minus grouped KL
as equivalent representations of one ELBO. The current H6/H7 types seal
slot order, identity hashes, scalar values, and the additive total, but
they do not seal whether a non-emission slot denotes
\(E_q[\log p_f]\), \(-\mathrm{KL}(q_f\Vert p_f)\), or another signed
contribution. The claim therefore turns a defensible schema choice into
an unsupported necessity.

This does not defend the current relabeling in
`vfe4/objective/h7_covariance.py`. That mapping is semantically
unverified. It means the implementation must choose and encode one of
two valid representations: preserve raw slots and build grouped-KL
diagnostics separately, or make signed grouped KLs authoritative and
move recognition entropy to a nonadditive diagnostic. Only the first
choice preserves a separate, nonzero entropy slot inside the present
additive total, but that extra premise is not enforced by the present
type contract.

## Evidence

1. Canon supplies two equal ELBO forms, not a preferred storage layout.
   Blei, Kucukelbir, and McAuliffe define
   \(\mathcal L(q)=E_q[\log p(z,x)]-E_q[\log q(z)]\), then rewrite the
   same quantity as expected log likelihood minus
   \(\mathrm{KL}(q(z)\Vert p(z))\)
   [Blei, Kucukelbir, and McAuliffe 2017 §2.2, Eq. 13](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf).
   Kingma and Welling use the grouped reconstruction-minus-KL form as a
   direct variational-bound estimator
   [Kingma and Welling 2014 §2.2–2.3, Eq. 3](https://arxiv.org/abs/1312.6114).
   These are algebraically identical objectives. Neither source requires
   an implementation trace to store the raw-factor expansion.

2. Grouped conditional KLs are exact when the two laws have compatible
   factorizations. The relative-entropy chain rule is
   \(D(p(x,y)\Vert q(x,y))=D(p(x)\Vert q(x))+
   D(p(y\mid x)\Vert q(y\mid x))\)
   [Cover and Thomas 2006 §2.5, Theorem 2.5.3](https://onlinelibrary.wiley.com/doi/10.1002/047174882X.ch2).
   Repeated application yields the ordered conditional-KL decomposition
   used by the whitepaper. Wainwright and Jordan also exhibit several
   equivalent primal, mixed, and dual forms of KL rather than assigning
   one form privileged status
   [Wainwright and Jordan 2008 §5.2.2, Eqs. 5.9–5.12](https://www.cs.columbia.edu/~blei/fogm/2025F/readings/WainwrightJordan2008.pdf).
   Canon rules out adding a grouped KL and its already-absorbed entropy
   again; it does not rule out making the grouped form the primary
   representation.

3. The H6 interface leaves contribution semantics opaque.
   `LanguageElboExpectation.contribution` returns only a tensor and has
   no representation tag or operand decomposition
   (`vfe4/objective/language_elbo.py:425-462`). Its assembler reads those
   tensors, sums them, and checks equality with an independently
   accumulated scalar
   (`vfe4/objective/language_elbo.py:826-875`). A provider can therefore
   supply signed grouped KLs, zero-valued entropy slots, and the matching
   grouped total without violating this interface.

4. The owned H6 record checks structure and arithmetic, not the claimed
   raw meaning. `H6LanguageElboTerms.__post_init__` requires all seven
   partition names and equality of the summed decomposition to the
   reported total (`vfe4/types/h6.py:2451-2491`). Its hashed payload
   records partition labels, identities, values, and totals, but no
   `raw_expected_log_factor` versus `grouped_kl` discriminator
   (`vfe4/types/h6.py:2493-2507`). The H7 wrapper then copies the 13
   values and checks only source identity, order, scalar shape,
   finiteness, and hash integrity
   (`vfe4/objective/language_elbo.py:1107-1200`). Thus, the current
   machine contract cannot prove the claim's semantic premise.

5. The live H7 mapping exposes the ambiguity rather than resolving it.
   It maps each source and transition partition directly to an identifier
   ending in `_kl` (`vfe4/objective/h7_covariance.py:1399-1434`), while
   its child-operand inventory describes those identifiers as
   \(-q,+p\) pairs (`vfe4/objective/h7_covariance.py:2618-2644`). No check
   between those child operands and the copied scalar appears in either
   block. The independent oracle shows the alternative coherent grouped
   design: it computes source log-ratios and negative conditional
   Gaussian KLs (`verification/mp_oracles/h7_covariance.py:3770-3787`),
   records joint entropy as a diagnostic, and explicitly omits that
   diagnostic from the grouped complete ELBO
   (`verification/mp_oracles/h7_covariance.py:3801-3807`). This is
   consistent with the chain rule above and is a concrete counterexample
   to the assertion that raw slots are mathematically mandatory.

6. A one-factor counterexample isolates the disputed necessity. Let
   \(\ell=E_q[\log p(x\mid z)]\),
   \(r=E_q[\log p(z)]\), and \(h=H(q)\). The raw trace is
   \(\ell+r+h\). The grouped trace is
   \(\ell-\mathrm{KL}(q\Vert p)\), since
   \(-\mathrm{KL}(q\Vert p)=r+h\). Both are exact. A trace with
   emission \(\ell\), a signed grouped-KL slot
   \(-\mathrm{KL}(q\Vert p)\), and a zero additive entropy slot satisfies
   the current structural and total-equality checks. A trace that stores
   both \(-\mathrm{KL}(q\Vert p)\) and the nonzero \(h\) as additive
   terms is wrong. The invariant is one-time accounting, not raw storage.

## Falsification conditions

- The claim fails as a statement about the current implementation if an
  exact `LanguageElboExpectation` containing signed grouped KL values,
  zero entropy contributions, and a matching independent total is
  accepted by `_evaluate_language_elbo` and
  `require_h7_complete_factor_trace`. The static call chain above
  predicts acceptance.
- This red objection fails if the repository adds or identifies an
  already-binding contract that mechanically proves every non-entropy
  slot equals the corresponding \(E_q[\log p_f]\) expectation and every
  entropy slot equals a disjoint, nonzero component of \(H(q)\). Under
  those added premises, preserving raw slots and deriving grouped KLs
  separately follows.
- I cannot falsify the narrower algebraic statement that a nonzero
  entropy contribution cannot be added on top of grouped KL terms that
  already contain that same entropy without double counting. The
  falsified portion is the asserted uniqueness and present enforcement
  of the raw 13-slot representation.
