# H3 structured-posterior adequacy preregistration

- Date frozen: 2026-07-21
- Coupled fixture: `vfe4/validation/fixtures/h3_coupled_v1.json` (`h3-coupled-v1`)
- Coupled fixture SHA-256: `6779f5b0a2e27aa5e203764bcc4d84c1b1daedb9423fcefdf28dce3cf7e40e03`
- Zero-control fixture: `vfe4/validation/fixtures/h3_zero_control_v1.json` (`h3-zero-control-v1`)
- Zero-control fixture SHA-256: `ba600e09e0ae7e2b7576fbf4446a8e5b38a605c7621eb0cd5586689dccb89acf`

This preregistration and both independently serialized fixture files were
written before any H3 optimizer or gate calculation. Neither fixture, either
observation, any threshold, nor any optimizer setting was selected from an H3
optimizer residual. Raw-file hashes bind the bytes as stored; no normalized or
re-serialized JSON is substituted for those bytes.

## Bounded claim and fixed laws

H3 is a bounded synthetic adequacy gate. It tests whether a differentiable
full-SPD Gaussian recognition family can close the posterior gap on one frozen
coupled four-dimensional conjugate Gaussian law, while a coordinatewise
factorized diagonal Gaussian cannot, and whether that advantage disappears on
a separately authored zero-coupling control whose exact posterior factorizes.
It is not a language-model, predictive, training, scaling, or general theorem
about correlation.

Both laws are source-free, use `T=1`, `d_z=d_m=1`, float64 CPU arithmetic, and
the exact coordinate order `[z0,m0,z1,m1]`. Every generative factor is a
normalized scalar Gaussian encoded by a coordinate row, scalar target, and
strictly positive variance.

The coupled law has independent `z0,m0 ~ N(0,1)`,
`m1|m0 ~ N(0.8*m0,0.36)`,
`z1|z0,m1 ~ N(0.7*z0+0.6*m1,0.25)`, and
`x1|(z1,m1) ~ N((z1,m1),0.64 I2)` at `x=(1.1,0.2)`. Its six frozen rows are:

```text
([1,0,0,0], 0.0, 1.0)
([0,1,0,0], 0.0, 1.0)
([0,-0.8,0,1], 0.0, 0.36)
([-0.7,0,1,-0.6], 0.0, 0.25)
([0,0,1,0], 1.1, 0.64)
([0,0,0,1], 0.2, 0.64)
```

Its frozen posterior reference is

```text
J = [[ 2.96,              0.0,   -2.8,               1.68             ],
     [ 0.0,               2.77777777777778, 0.0,    -2.22222222222222],
     [-2.8,               0.0,    5.5625,           -2.4              ],
     [ 1.68,             -2.22222222222222, -2.4,    5.78027777777778]]
h = [0.0, 0.0, 1.71875, 0.3125]
log p(x) = -2.6536596233553
analytic fine-factorized reverse-KL gap = 0.6815463199745935 nats
```

The zero control is a separate JSON document, not a runtime transformation of
the coupled object. It preserves the initial and observation maps/covariances,
declares independent `m1 ~ N(0,0.36)` and `z1 ~ N(0,0.25)`, and observes
`x=(0.4,-0.7)`. Its six independently authored rows are:

```text
([1,0,0,0], 0.0, 1.0)
([0,1,0,0], 0.0, 1.0)
([0,0,0,1], 0.0, 0.36)
([0,0,1,0], 0.0, 0.25)
([0,0,1,0], 0.4, 0.64)
([0,0,0,1], -0.7, 0.64)
```

The zero-control frozen posterior precision is exactly
`diag(1,1,5.5625,4.34027777777778)`. Its exact posterior factorizes; the
independent NumPy oracle must derive a zero analytic factorized reverse-KL gap.

## Three independent computational paths

The paths are a differentiable PyTorch structured full-SPD recognition arm, a
differentiable PyTorch fine-factorized diagonal recognition arm, and an
independent NumPy exact-posterior/evidence/analytic reverse-KL mean-field
oracle. Only `verification/h3_gate.py` may compare all three paths.

Production evaluates `E_q[log p(y,x)] + H(q)` directly from the six normalized
initial, transition, and observation factor expectations. It may not evaluate
`log_evidence-KL`, consume oracle outputs, or import `verification`. Both
PyTorch arms parameterize precision as `J_q=L_q L_q.T`, with each Cholesky
diagonal equal to `exp(raw_diagonal)`. The structured family learns all six
strict-lower entries of a `4x4` factor; the factorized family has no
off-diagonal parameters. Each arm receives fresh parameters and a fresh
optimizer from the common initialization `mu=zeros(4), J_q=I4`. No randomness,
warm start, jitter, clipping, pseudoinverse, repair, or regularization is
allowed.

## Frozen optimizer and convergence contract

Every arm maximizes ELBO by minimizing its negative with:

```text
torch.optim.LBFGS(
    lr=1.0,
    max_iter=1,
    max_eval=25,
    tolerance_grad=1e-12,
    tolerance_change=1e-18,
    history_size=20,
    line_search_fn="strong_wolfe",
)
```

### Pre-promotion optimizer-coherence amendment (2026-07-22)

During focused Task 4 implementation, before any H3 promotion run, gate
decision, or milestone evidence was produced, the installed PyTorch L-BFGS
implementation was confirmed to interpret `tolerance_change` as a stopping
bound on the directional derivative as well as on parameter and loss changes.
The original `1e-15` value could therefore stop a valid strong-Wolfe trajectory
while its gradient infinity norm remained above the independently frozen
`1e-8` terminal target. To make the internal optimizer stop coherent with that
unchanged target, `tolerance_change` is amended to
`(1e-8)^2 / 100 = 1e-18`. The terminal gradient target, accepted-objective
change target, optimizer family, line search, iteration and closure budgets,
and every other setting and decision rule remain unchanged. This is a
pre-promotion protocol correction, not a post-gate threshold adjustment.

The same focused pre-promotion review also found that, near the factorizing
zero-control optimum, changes in the full normalized ELBO can be smaller than
one float64 unit in the last place of that full scalar. Strong-Wolfe would then
see a quantized zero loss change even though the direct ELBO gradient remained
above its target. Each outer iteration therefore recenters only its closure
loss at a detached, immutable reference `q0` captured from that iteration's
first closure, with no extra objective evaluation. Every closure minimizes the
negative of the following direct normalized-factor difference:

```text
a  = row @ q.mean  - target
a0 = row @ q0.mean - target
s  = solve_triangular(L_q, row)
s0 = solve_triangular(L_q0, row)
variance_delta = dot(s-s0, s+s0)
expected_square_delta = (a-a0)*(a+a0)
                        + variance_delta
factor_delta = -0.5*expected_square_delta/variance
entropy_delta = -sum(log(diag(L_q)) - log(diag(L_q0)))
ELBO_delta = sum(factor_delta) + entropy_delta
```

The reference is reset at every accepted outer step and is independent of the
optimized tensors. Thus `grad_q ELBO_delta = grad_q ELBO`, and an additive
constant leaves the strong-Wolfe conditions unchanged while the factored
mean-square difference avoids subtracting rounded full-loss scalars. The
solve-vector identity `dot(s-s0,s+s0)` likewise evaluates
`||s||^2-||s0||^2` without subtracting two rounded order-one variances; this
completes the same algebraic stabilization for covariance and strict-lower
precision-Cholesky directions. Accepted
diagnostics and artifact values remain evaluations of the full direct ELBO.
No evidence, canonical form, oracle value, normalized constant, inverse,
repair, tolerance, convergence criterion, or decision threshold enters this
recentered closure. This seam was adopted before H3 promotion or gate evidence
and does not loosen any acceptance rule.

The hard caps are 200 accepted outer iterations and 5,000 closure evaluations.
The closure budget is enforced before every objective evaluation by a dedicated
budget-exhaustion exception. Every value and gradient must remain finite.
Convergence requires terminal gradient infinity norm at most `1e-8` and
absolute accepted-objective change at most `1e-12` for three consecutive
accepted iterations.

## Frozen admissibility envelope

For each exact posterior and terminal recognition law, `D=4`,
`lambda_min(J)>=1e-4`, `lambda_max(J)<=1e4`, `kappa_2(J)<=1e6`, and
`||mu||_inf<=4`. An envelope violation is rejected, never repaired.

## Frozen operand-local absolute allowances

Let `eps=np.finfo(np.float64).eps`, `gamma(n)=n*eps/(1-n*eps)`, `C=4096`,
`N(D)=16*D+64`, and the per-optimized-operand solver contribution be `1e-7`
nats. There is no relative tolerance and no singular or run-global allowance.

For each scalar, using only its value, absolute-summand accumulation, its own
SPD operand condition numbers, and its optimized-arm flag:

```text
rounding = C*gamma(N(D))*max(1,*kappas)*max(1,abs(value),absolute_sum)
scalar_allowance = (1e-7 if optimized else 0) + rounding
```

For a pair `(left,right)`, the allowance is the two scalar allowances plus
`C*gamma(D+2)*max(1,abs(left),abs(right),abs(left)+abs(right))`. A three- or
four-operand identity allowance is the sum of exactly those operand scalar
allowances plus one signed-reduction term using `C*gamma(D+3)` or
`C*gamma(D+4)`, respectively. No invariant borrows an unrelated condition
number, scale, or run-wide maximum. There is no blanket `allclose`, residual-
tuned threshold, or post-result adjustment.

Every decision allowance must be strictly less than one percent of its named
decisiveness scale. Nat-valued adequacy and control decisions use the coupled
analytic gap `G`; canonical path/reference comparisons use
`max(1,abs(each compared operand))` in their own units.

## Frozen signed thresholds and status rules

For the coupled gap, `margin_gap=G-0.50` and `A_gap` is the operand-local pair
allowance for `(G,0.50)`. For structured closure,
`margin_resolve=0.01*G-KL_cs` and `A_resolve` is the pair allowance for
`(0.01*G,KL_cs)`. Each threshold uses this exact three-way decision:

- `margin > allowance`: PASS eligibility;
- `margin < -allowance`: finite FAIL; and
- `-allowance <= margin <= allowance`: INCONCLUSIVE, with respectively
  `resolve coupled gap threshold outside allowance` or
  `resolve structured closure threshold outside allowance`.

An allowance is never added to the favorable side. A finite, converged
equality invariant outside its allowance is FAIL. Missing or mismatched hashes,
parse/control failure, PyTorch/NumPy/frozen-reference disagreement,
nonconvergence, nonfinite trajectory, envelope failure, nonfactorizing control,
or a nondecisive allowance is INCONCLUSIVE with an explicit open obligation.

H3 passes only if both thresholds are PASS-eligible; all four arms converge;
the coupled factorized terminal KL matches `G`; each arm closes
`log p(x)-ELBO=KL(q||p)`; the coupled ELBO-delta/KL-delta four-operand identity
closes; both zero-control KLs and their ELBO delta close; both zero-control
ELBO/KL identities close; and every relevant allowance is decisive. PASS has
no obligations and every invariant passes. FAIL has no obligations and at
least one available finite decision invariant fails. INCONCLUSIVE has one or
more explicit obligations and cannot be promoted by agent consensus.

## Frozen artifact schema

### Implemented verification surface (pre-promotion)

The fail-closed H3 evaluator lives in `verification/h3_gate.py`, with its
focused promotion contract in `tests/promotion/test_h3_gate.py`. The evaluator
implements the frozen 19-invariant order, records all 14 allowance-bearing
comparisons by operand, applies the signed three-way threshold rule, and stops
before downstream parsing, oracle evaluation, model construction, or
optimization when either raw fixture digest is wrong. This implementation
note records the code surface only; it is not promotion evidence and makes no
test, gate-status, or milestone claim.

The single `verify_vfe4.py` launcher and `verification/run_gates.py` runner
extend that surface to the exact H1, H1/H2, and H1/H2/H3 prefixes. The runner
revalidates the full resolved configuration, derives an H1/H2-only projection
for the unchanged legacy evaluators, conditionally captures each requested raw
fixture once, and publishes only the requested validation payloads in one
atomic run directory. H1 and H1/H2 prefixes contain no H3 profile, fixture
identity, consumer, or payload. This is likewise a pre-promotion implementation
statement rather than measured evidence.

An H3-prefix click run publishes one atomic `validation/h3.json` beside the
unchanged H1/H2 payloads. The H3 payload contains:

- schema version, gate, status, and obligations;
- both fixture IDs, relative paths, byte counts, and raw expected/observed
  SHA-256 digests;
- the canonical config SHA-256 and exact H3 profile: ordered recognition
  families, common zero/identity initialization, operation, expected autograd
  scope, optimizer settings, and decision settings;
- frozen references and independent oracle outputs/diagnostics;
- four arm initializations, terminal laws, convergence facts, and canonical
  accepted-trace digests;
- every KL, ELBO, evidence difference, adequacy delta, and resolved fraction;
- allowance constants and an `allowances_by_invariant` object containing each
  allowance-bearing comparison's kind, exact operands, per-operand absolute
  sums/condition numbers/optimized flags/scalar allowances, final allowance,
  residual or signed margin, decisiveness scale, and ratio;
- signed `threshold_decisions` for `coupled_oracle_gap_minimum` and
  `coupled_structured_fraction_resolved`, including ordered operands, favorable
  formula/direction, margin, pair allowance, boundaries, eligibility, and the
  exact indecision obligation;
- the exact ordered invariant records; and
- this bounded claim and the explicit later-gate nonclaims.

The run captures each required fixture's raw bytes once and uses the same bytes
for hashing and all consumers. Raw fixture hashes and canonical trace JSON
digests are labeled as separate hash domains. JSON contains finite numbers only
and row-major terminal matrices. The manifest hashes every validation payload.

## Explicit deferrals and nonclaims

This H3 gate establishes, at most, numerical adequacy and optimizer reachability
on the two bounded conjugate fixtures. It does not establish generalization,
asymptotic stability, performance superiority, gradient correctness outside
the tested objective, language-model validity, or causal/predictive behavior.
H4 cost evidence, H5 update guarantees, H6 prefix/prediction evidence, H7 frame
covariance, and H8 sparse scaling are explicitly deferred and cannot be
inferred from H3. L-BFGS traces are H3 convergence records, not H5 monotonicity
evidence.
