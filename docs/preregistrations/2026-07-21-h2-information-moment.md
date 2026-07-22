# H2 information--moment preregistration

Date frozen: 2026-07-21  
Fixture: `vfe4/validation/fixtures/h1_v1.json` (`h1-v1`)  
Fixture SHA-256: `388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b`

This document was frozen before the H2 promotion calculation. The fixture data were not selected or altered using H2 residuals.

## Scope and fixed law

- H2 is representation verification only. It makes no optimizer, performance, gradient, H3/H4/H5/H7/H8, prediction, or scaling claim.
- The unchanged `h1-v1` law has `T=2`, `D=6`, continuous order `[z0,m0,z1,m1,z2,m2]`, and source-path order `[(0,0),(1,0),(0,1),(1,1)]`, written below as `(model source, state source)`.
- The four positive recognition source weights are exactly `0.30`, `0.10`, `0.12`, and `0.48`, in that order.
- Every fixed-source recognition and generative Gaussian component is evaluated separately and aggregation uses those exact recognition weights. The source-marginal mixture is never moment-projected or described as one Gaussian. No source-mixture global precision is defined or computed.
- Natural coordinates are `(h,-J/2)`, expectation coordinates are `(mu,M)` with `M=Sigma+mu mu^T`, and `(mu,Sigma)` is the moment representation, not the Fisher-dual pair.
- Float64 is mandatory. Jitter, pseudo-inversion, clipping, repair, rescaling, and regularization are forbidden.

## Frozen admissibility and observed conditioning

The broader rejection envelope, fixed before promotion, is `D <= 6`, `lambda_min(J) >= 1e-4`, `lambda_max(J) <= 1e4`, `kappa_2(J) <= 1e6`, and `||mu||_inf <= 4`. Every component record includes minimum Cholesky pivot, both extreme eigenvalues, condition number, and mean infinity norm.

The eight already observed Task 3 records are:

| path `(b1,a1)` | law | minimum pivot | `lambda_min` | `lambda_max` | `kappa_2` | `||mu||_inf` |
|---|---:|---:|---:|---:|---:|---:|
| `(0,0)` | q | 1.0090206053192705 | 0.37007370675301715 | 5.337573190781433 | 14.423000319619213 | 0.25000000000000033 |
| `(0,0)` | p | 1.0063000691083757 | 0.25362513626306854 | 10.09008489358491 | 39.783457752859064 | 0.29950000000000004 |
| `(0,1)` | q | 0.9215846837929884 | 0.3152789291069631 | 5.456159733933167 | 17.305817897148728 | 0.24999999999999997 |
| `(0,1)` | p | 0.8867826840726607 | 0.2287766949200716 | 9.688312280567013 | 42.34833571641485 | 0.19999999999999998 |
| `(1,0)` | q | 0.7921430105782453 | 0.27674255734050857 | 5.689056029145741 | 20.557214198703214 | 0.2499999999999999 |
| `(1,0)` | p | 0.9373952812201246 | 0.2469638126865089 | 10.189172671632388 | 41.2577557853236 | 0.2771000000000003 |
| `(1,1)` | q | 0.7921430105782449 | 0.26758127265870907 | 6.053485971289834 | 22.622980715884598 | 0.24999999999999997 |
| `(1,1)` | p | 0.8981907036954554 | 0.22580664749462606 | 9.478533815515279 | 41.976327626673864 | 0.20000000000000007 |

The preregistered calibration statement remains the plan-frozen conservative summary: `lambda_min >= 0.22580664749462973`, `lambda_max <= 10.189172671632396`, `kappa_2 <= 42.348335716414404`, and `||mu||_inf <= 0.29950000000000004`. The component table reports the platform evaluation that motivated those already frozen bounds; the gate applies the broader envelope, not rounded calibration extrema.

## Frozen absolute budgets

Let `eps=np.finfo(np.float64).eps`, `gamma(n)=n*eps/(1-n*eps)`, `C=256`, and `N(D)=8*D+32`.

- A path allowance is `C*gamma(N(D))*max(1, every SPD operand kappa_2)*max(1, output infinity norm, absolute-summand accumulation infinity norm)`.
- Backward allowances for `J@mu-h` and `J@Sigma[:,B]-E_B` omit condition number and are `C*gamma(N(D))*max(1, ||J||_inf*||solution||_inf+||rhs||_inf)`.
- A complete-ELBO allowance is the sum of its exactly 12 signed local-term allowances plus `C*gamma(13)*max(1,sum(abs(term) for term in signed_terms))`.
- A pair allowance is the two local allowances plus `C*gamma(D+2)*max(1,||left||_inf,||right||_inf)`.

There is no relative tolerance, blanket `allclose`, empirical post-tuning, or threshold tuning on `h1-v1`. At observed `kappa_2 <= 42.35`, the descriptive pair budget is approximately `3.86e-10 * scale`; each decision nevertheless uses its own literal invariant-specific calculation.

## Preregistered invariant families

The exact inventory covers:

1. fixture identity and SHA-256 availability;
2. all eight q/p component envelope and Cholesky-pivot records;
3. direct-information versus unchanged H1 moment means and every selected covariance block for all eight q/p components;
4. `J@mu-h` and `J@Sigma[:,B]-E_B` backward residuals;
5. information versus independent NumPy log normalizers and entropies;
6. information versus unchanged H1 and independent NumPy oriented `KL(q||p)` and Gaussian log ratio;
7. information versus unchanged H1 and independent NumPy for each component emission, source ratio, Gaussian contribution, and complete component value;
8. information versus unchanged H1 and independent NumPy for every aggregate local contribution, joint recognition entropy, and the exactly 12-term complete ELBO; and
9. the four negative controls below.

An available finite residual above its allowance is `FAIL`. Missing evidence, fixture/hash mismatch, factor failure, an indecisive mathematical wrong path, or inability to instrument the inverse path is `INCONCLUSIVE`.

## Exactly four promotion negative controls

No other promotion control is admitted:

1. misread canonical `(h,J)` as moment `(mu,J)` (`h=mu`);
2. reverse the Gaussian log-determinant ratio;
3. substitute the diagonal precision inverse `J_ii^-1` (a conditional covariance shortcut) for the required Schur/marginal emission covariance; and
4. instrument every production factor, forbid inverse/pseudoinverse/Cholesky-inverse and any width-`D` or all-column selected solve, require zero real-evaluation attempts, then deliberately inject `solve(I_D)` and require detection.

For each mathematical control, decisiveness is frozen as

`wrong_residual >= 1e-3 * max(1, abs(correct_value), abs(wrong_value))`.

Anything smaller is `INCONCLUSIVE`, never a pass.

## Nonclaims

Passing H2 would establish only componentwise and exact source-weighted numerical agreement, on this immutable bounded fixture, among the direct PyTorch information representation, the unchanged PyTorch H1 moment representation, and an independently parsed NumPy dense-moment calculation. It does not establish a global Gaussian representation of the mixture, generalization beyond the frozen envelope, asymptotic stability, performance superiority, optimizer correctness, gradient correctness, predictive validity, scaling behavior, or any later-gate claim.
