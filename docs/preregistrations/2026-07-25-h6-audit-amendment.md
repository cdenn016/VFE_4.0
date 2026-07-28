# H6 Audit Amendment: Architecture, Source Prior, and Prediction Gates

**Date frozen:** 2026-07-25
**Status:** normative source-build amendment; no scientific evidence is
created by this document
**Amends:** `2026-07-21-h6-prefix-prediction.md` and the H6 implementation
plan
**Inputs:** the July 24 peer review and plan amendments, followed by static
code and theory reconciliation at source revision `a238717`

This amendment supersedes only the conflicting H6 architecture, source-prior,
matching, SMC-bias, objective-gate, and interpretation clauses identified
below. All unchanged safety, data-access, one-opening, failure, and provenance
requirements remain in force.

## 1. Primary A0 is a causal transformer

The primary A0 identity is `h6-a0-transformer-v2`. It is a normalized
autoregressive categorical model with:

- vocabulary size `V=258`;
- maximum receiver horizon and learned-position capacity `L_max=32`;
- one causal transformer block;
- hidden width `h=52`;
- two equal attention heads of width `26`;
- pre-normalization before attention and before the MLP, followed by a final
  LayerNorm;
- scaled dot-product attention with `is_causal=True`, zero dropout, and the
  deterministic CPU float64 math policy used by the H6 source checks;
- a `h -> 4h -> h` MLP with tanh-approximate GELU;
- residual connections around attention and the MLP; and
- separate token embedding and untied decoder weights.

A zero, non-parameter BOS row occupies position zero. Prefix tokens occupy
positions `1..t-1`, so every one of the 32 learned position rows is live at
some declared receiver. There is no learned BOS parameter and no dormant
position capacity.

The exact active-parameter formula is

```text
P_A0(h) = 2*V*h + 32*h + 12*h^2 + 15*h + V.
```

Thus `P_A0(52)=61,982`. Width `h=53` is inadmissible because it cannot split
into two equal heads. The WikiText-103 formula containing `128*h` does not
apply at H6 scale.

The former mean-pooled prefix model may remain only as an explicitly named
non-primary floor or no-latent descriptive control. `build_a0` may not return
it, and no artifact may deserialize its old identity as
`h6-a0-transformer-v2`.

## 2. Primary A5 uses a parent-specific pooled-prefix prior

The primary A5 identity is
`h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2`.
It uses a normalized, target-blind, parent-specific pooled-prefix generative
source prior. For source bank `b`, receiver `t`, and supported parent `j`,

```text
q_t = 0                                      when t = 1
q_t = mean(E[x_1], ..., E[x_(t-1)])         when t > 1
s_b,t,j = q_t^T (k_b,t,j + W_b y_b,j) + beta_b,t,j.
```

The last declared parent is the categorical gauge anchor. Its slot key and
bias are exact zeros, but its complete content score `q_t^T W_b y_b,anchor`
is retained. Every supported free logit is its complete raw score minus the
complete anchor score, followed by the existing masked log-softmax over the
declared support.

The scorer schema is
`parent-specific-pooled-prefix-bilinear-v1`, with:

```text
token_summary = mean-prior-token-embeddings-v1
parent_content = bank-projection-of-candidate-row-v1
anchor = last-declared-parent-complete-score-subtraction-v1
normalization = masked-log-softmax-from-declared-parents-v1
```

This is a normalized stochastic source selector. Its token query is still
mean pooled and therefore is not generally order-sensitive. It is not
standard transformer attention and carries no H7 frame-covariance claim.

The old implementation, which averages projected latent history before
scoring every parent slot, is separately identified as
`pooled_history_conditioned`. It cannot satisfy the new primary identity.

The parameter count is unchanged by the parent-addressability repair:

```text
P_prior = V*c + c*(d_z+d_m) + (c+1)*(A_z+A_m),
```

where `A_b` is the number of non-anchor supported parent slots in bank `b`.
For two dense H6 banks with equal latent width `d`, this is
`V*c + 2*c*d + (c+1)*T*(T-1)`.

## 3. H1-Prefix scorer-v2 prerequisite

The existing H1-Prefix fixture has zero latent projections and zero latent
history, so it cannot certify parent-content addressability. A sibling
scorer-v2 prerequisite must use `T=2`, `d_z=d_m=1`, `V=3`, nonzero bank
projections, distinct nonzero parent latents, and active/swapped histories
under one fixed target-free prefix.

Its independent NumPy route must verify:

- production-versus-oracle probabilities for both banks;
- exact support masking and normalization;
- active and swapped complete-objective decompositions;
- evidence-minus-posterior-KL equality;
- invariance to the current target and suffix; and
- a parent swap that changes and swaps the supported probability assignment.

Only a PASS artifact bound to the scorer-v2 schema can satisfy the new H6
primary prerequisite. Ordinary scalar `h1-v1` remains unchanged.

The scorer-v2 producer keeps `h1-prefix-prior-config-v2` but publishes the
versioned `h1-prefix-prior-validation-v3` payload. Version 3 adds the exact
candidate JUnit SHA-256 (or an explicit `null` outside a candidate-evidence
lifecycle) to the immutable validation bytes. H7 may accept this predecessor
only when the config, fixture, scorer, generative-factor schema, source
identity, and non-null JUnit identity all match the same candidate. Historical
validation v1/v2 artifacts remain readable where applicable but cannot
authorize amended H6/H7 promotion.

## 4. Matching remains jointly fail-closed

The primary scientific match is whole-schedule training-arithmetic matching,
not inference or end-to-end compute matching. Eligibility retains both hard
conditions:

```text
parameter_relative_difference <= 0.01
whole_schedule_training_FLOP_relative_difference <= 0.05.
```

Before selecting the primary A5 nuisance allocation, the implementation must
remove only exact shared-frame computational redundancy: compute each live
`U_t=exp(phi_t)` once per channel and forward graph, reuse each valid
source-frame solve or pullback, preserve values and gradients, and invalidate
the evaluation cache after every parameter update or detached evaluation
boundary. This common-subexpression elimination changes neither the
normalized model nor its active parameters.

The A5 candidate inventory is the finite Cartesian product

```text
latent_width D         = (2,4,8)
prior_context_width C  = (4,6,8)
emission_width E       = (72,84,85,86,87,88,89)
recognition_width R    = (113,114,115,116,117,118)
```

This is a v3-only PRIMARY amendment: it adds the single ascending emission
width `72` without changing the legacy v2 constants or any frozen component
grid. The resulting 378 candidates are enumerated in ascending lexicographic
`(d,c,e,r)`
order. The search may not inspect corpus bytes, losses, gradients, validation
metrics, test metrics, or prediction FLOPs. After filtering on the two hard
gates, the first eligible row in that order is selected. If the eligible set
is empty, no winner exists.

The pre-outcome v3 formula audit selects `(d,c,e,r)=(2,8,72,117)` first.
Against `P_A0=61,982`, it has `P_A5=61,454`, an exact relative gap of
`0.851860%`. On the 258-token synthetic matching fixture its exact
whole-training arithmetic totals are `F_A0=178,715,214` and
`F_A5=187,045,140`, a `4.661006%` gap; the independently checked
production/asymptotic workload gives approximately `4.73217%`. The next
eligible lexicographic row is `(2,8,72,118)`, so first-lexicographic selection
still chooses `r=117`. These are analytical pre-outcome matching facts, not
training evidence. The implementation may not add filler parameters, dormant
state, redundant recomputation, fake phases, no-op optimizer work, or wider
tolerances.

Prediction, particle, cache, checkpoint-load, and scoring costs are reported
in a separate inference-inclusive ledger and never change training-match
eligibility.

The two hard gates authorize only the amended `PRIMARY` comparison:
`h6-a0-transformer-v2` versus the jointly selected parent-specific complete
A5 endpoint. This clause supersedes the earlier blanket requirement that every
descriptive/component endpoint must close both gates before PRIMARY can be
eligible. The frozen component inventories and selectors are not widened.
Each component retains its exact predeclared active endpoint, ownership,
whole-schedule training ledger, formula-selection status, and obligations as a
disclosure. When its formula selection is INCONCLUSIVE, the associated matrix
record is explicitly unauthorized for a matched component conclusion and
cannot promote or demote PRIMARY. No unmatched component result may be cited
as a compute-matched attribution. Overall matching authorization and H6 v3
readiness are derived only from the canonical PRIMARY selection and PRIMARY
matrix record. Every component selector, status, and obligation nevertheless
remains fully bound into the matching-set digest and must fail closed if
mutated.

The `OBJECTIVE` endpoints are not a second training-compute match. Both are
derived only from the exact eligible PRIMARY selection, share its selected A5
nuisance allocation and ownership, and differ only in `objective_kind`. Their
intended complete-versus-emission computation difference may leave the generic
five-percent matrix report INCONCLUSIVE; that arithmetic status neither
authorizes a compute-matched OBJECTIVE claim nor blocks the separately
preregistered logical OBJECTIVE gate.

Scorer authorization is derived from each exact selected endpoint config:
every `latent_enabled=True` endpoint carries the full weighted-SMC
`(128,256,512,1024)` table, while A0 and the explicit no-latent A5 control
carry one exact particle-free row. Config-ID prefixes or caller-supplied
scorer labels cannot authorize either path.

## 5. Raw SMC bias and Richardson Q2 are distinct

For a nonnegative unbiased particle estimate `Z_hat_N` of `Z`, Jensen's
inequality gives

```text
E[log Z_hat_N] <= log Z.
```

Therefore raw particle log likelihood is downward-or-equal biased in
expectation and raw NLL is upward-or-equal biased. This statement applies to
the raw estimator `Y_N=log Z_hat_N`.

The reported H6 endpoint remains

```text
Q2 = 2*Y_1024 - Y_512.
```

If `E[Y_N]=Y+c/N+d/N^2+...`, then the leading remaining Q2 bias is
`-d/(2*N^2)+...`; Jensen's sign on `c` does not determine the sign of `d`.
Accordingly:

- Q2 keeps the two-sided conditional geometric remainder
  `B=U2/(1-0.75)`;
- the existing estimator-error boxes remain two-sided;
- contraction failure remains INCONCLUSIVE; and
- no artifact may label Q2 as one-sided unless a separate signed expansion
  is proved and preregistered.

A deterministic continuous linear-Gaussian calibration is added alongside
the existing finite categorical gate. It is a sensitivity and coverage
control, not proof that its bias bound transfers to trained language-model
checkpoints or WikiText-103.

## 6. OBJECTIVE is a logical blocking gate

The OBJECTIVE comparison uses the parent-specific pooled-prefix prior on both
sides and changes only:

```text
objective_kind: complete_elbo -> emission_only_ablation_non_elbo.
```

Its oriented paired estimand and practical margin are:

```text
d_obj = NLL(parent-specific complete)
        - NLL(parent-specific emission-only)
delta_obj = -log(0.99) = 0.01005033585350145.
```

Using the estimator-aware paired interval:

- PASS requires `upper <= delta_obj`;
- FAIL requires `lower > delta_obj`; and
- all other complete or estimator-ineligible cases are INCONCLUSIVE.

H6 preserves one all-or-none test opening. OBJECTIVE is adjudicated logically
before PRIMARY from that frozen opening. If OBJECTIVE is FAIL or
INCONCLUSIVE, PRIMARY is recorded as
`NOT_EVALUATED_AFTER_OBJECTIVE_GATE`. This ordering does not claim to save
confirmatory training or test-scoring compute.

The emission-only arm remains explicitly `is_elbo=false`.
Its ordered emission factors must be recomputed by the exact live `BuiltArm`
model from typed observation/latent expectation contexts. The record binds the
complete live model-state SHA-256 captured across that evaluation. Arbitrary
caller-provided scalar contributions, factor identities, config-only model
family hashes, or a state hash that changes during evaluation cannot authorize
the OBJECTIVE endpoint.

## 7. Controls, risk, and downstream boundaries

The fixed-prior complete A5 endpoint becomes the PRIOR changed-joint control.
The latent-path and other component rows remain separately labeled and may
not be folded into the primary whole-architecture claim.

The V3 free-energy-versus-cross-entropy tension is recorded as
`v3-free-energy-versus-cross-entropy-tension-v1`, with status
`historical_risk_signal_not_vfe4_evidence` and mitigation gate
`h6-objective-gate-v1`. Exact V3 numbers require a verified primary artifact,
revision, config, and digest; a reviewer transcription is not evidence.

The July 22 H4 result remains INCONCLUSIVE. The operand-local H4 source and
preregistration repair already exist, so H4 requires a fresh separately
authorized artifact rather than another source rewrite. H4 remains
nonblocking for H6-Prediction.

The asymmetric `d_z=2,d_m=3` numerical coverage is a sibling H2/H5
objective-and-update oracle. It does not mutate scalar `h1-v1` or widen the
bounded equal-dimensional H8 fixture.

H7 and H8 remain source-only and INCONCLUSIVE. Invalidation follows the
consumer interface, not the date of this amendment:

| Change | Direct identity to repin | Downstream consumer |
|---|---|---|
| B1 parent-specific scorer | H1-Prefix and H6-Prefix | future H7 evidence, then any H8 evidence that binds that H7 result |
| A1 transformer baseline | H6-Prediction | future H8 evidence only |
| A2 Q2 semantics/calibration | H6-Prediction | future H8 evidence only |
| B2 OBJECTIVE gate | H6-Prediction | future H8 evidence only |
| C3 fresh H4 evidence | H4 result only | consumers that explicitly bind the new H4 result |
| C4 rectangular sibling oracle | its sibling result only | no existing H1 scalar identity |

H6-Prediction is not an H7 premise and must not be added as one. These
changes do not alter the committed H7/H8 mathematics. No WikiText-103
implementation or source lock is authorized until an exact-revision H8 PASS
exists.

## 8. Execution boundary

All ordinary buildout checks are node-scoped. There is no broad pytest run,
training run, data download, profiler, parameter grid, numerical campaign, or
test opening in this amendment slice. Any command that imports Torch or
constructs a model uses `C:/anaconda/python.exe`; bare `python` is not an
eligible model/CUDA interpreter on this machine.
