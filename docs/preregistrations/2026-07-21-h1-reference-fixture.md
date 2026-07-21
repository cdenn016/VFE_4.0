# H1 reference fixture preregistration

Date: 2026-07-21
Fixture: `h1-v1`
Status: frozen before the H1 calculations

## Scope and source hierarchy

This fixture is the bounded `T=2`, scalar-state, scalar-model, vocabulary-three specialization of the normalized VFE 4.0 probability model. The normative probability semantics come from `Manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex` and `Manuscripts/vfe4_whitepaper/`. The VFE 4.0 codebase design fixes the software boundaries; the JSON file in `vfe4/validation/fixtures/h1_v1.json` fixes the numerical data. MAgent may inform independent-oracle style and V3 may inform launcher ergonomics, but neither changes this fixture's probability law.

The continuous coordinate order is exactly

```text
[z0, m0, z1, m1, z2, m2].
```

At each modeled slice the directed order is `b_t -> m_t -> a_t -> z_t -> x_t`. Both sources at `t=1` are fixed at parent 0. At `t=2`, sources 0 and 1 are admitted. The generative model is fixed before the recognition law.

## Generative and recognition laws

The deterministic scalar frames are `U=(1.0,1.25,0.8)`, and the only H1 population transport is the exact coboundary

```text
omega(t,j) = U_t / U_j.
```

For selected model source `b_t=j` and state source `a_t=k`, the normalized generative kernels are

```text
m_t ~ Normal(omega(t,j) m_j + c_m[t], R_m[t])
z_t ~ Normal(omega(t,k) z_k + B[t] m_t + c_z[t], R_z[t]).
```

Each `R_m[t]` and `R_z[t]` is a strictly positive scalar variance. Each observation factor is a categorical distribution obtained by applying `log_softmax` to `w_z[t] z_t + w_m[t] m_t + bias[t]`.

The recognition law is independently normalized. It uses a correlated bivariate Gaussian initial law, normalized `Q(b_t)` rows, normalized `Q(a_t | b_t)` rows, and its own declared affine Gaussian model and state kernels. Recognition model kernels have location `slope*m_b + offset`. Recognition state kernels have location `z_slope*z_a + m_slope*m_t + offset`. These recognition coefficients are not population-frame transports and are not copied from the generative model.

## Labels, source ordering, and positive support

Observations retain the manuscript labels `{1,2,3}`. They enter tensor indexing only through the checked conversion

```text
label_to_index(label) = label - 1.
```

Thus label 1 selects decoder row 0 and label 2 selects decoder row 1. Labels 0 and 4 fail rather than wrapping or clipping.

The four `(a2,b2)` paths are frozen in this order:

```text
(0,0), (1,0), (0,1), (1,1).
```

The recognition path weights are consequently `(0.30,0.10,0.12,0.48)`. Any positive recognition source mass outside positive generative-prior support invalidates the fixture.

## Three H1 calculations and allowances

H1 will compare three separately assembled calculations:

1. the production monolithic complete-component ELBO;
2. the production local conditional-factor ELBO decomposition;
3. the independent NumPy evidence/posterior-KL identity.

The shared JSON data do not authorize shared mathematical assembly. Generative and recognition `6x6` Gaussian components are built independently from their declared directed affine-noise chains. The later NumPy oracle will independently parse the JSON and independently assemble its components.

Orders 21 and 17 are the frozen Gauss-Hermite evaluation and convergence-check orders. An order-to-order absolute difference is a fixture-specific convergence estimate, not a proved universal numerical-error bound. The maximum permitted convergence estimate is exactly `1e-9`. Each later pairwise comparison receives the sum of the two participating calibrated allowances plus its separately calculated compensated-summation rounding allowance; no single global tolerance replaces term-shaped allowances. The stochastic estimation budget is zero.

## Preregistered failure injections

H1 must detect, at minimum:

- omission of both categorical source entropy/KL contributions;
- substitution of raw selected logits for selected `log_softmax` values;
- substitution of recognition mixture components or weights for generative evidence;
- malformed JSON or altered schema/structural literals;
- non-SPD initial covariance or nonpositive conditional variance;
- nonnormalized source rows or recognition mass outside positive prior support;
- invalid one-based observation labels;
- a component assembly that changes the frozen continuous or source-path order.

## Nonclaims

Passing H1 will establish agreement for this one frozen finite fixture under its recorded float64 calculations and allowances. It will not prove predictive quality, posterior adequacy, optimizer convergence, training stability, gauge learning, nontrivial base curvature or holonomy, scalability beyond the bounded dense `6x6` check, H2 or later hypotheses, or a universal quadrature error theorem. Frames remain fixed deterministic structure; no `phi`, frame gradient, optimizer, ELBO objective implementation, independent oracle, or promotion gate is introduced by the present model/recognition task.
