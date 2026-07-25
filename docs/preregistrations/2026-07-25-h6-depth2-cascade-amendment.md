# H6 Depth-2 Cascade Composition-Risk Amendment

**Date:** 2026-07-25

**Probe identity:** `H6-DEPTH2-CASCADE-v1`

**Disposition:** nonblocking composition-risk probe
**Amends:** the depth language in the H6 audit/buildout amendment only

## Scope and claim boundary

This amendment defines depth as a normalized generative composition before any
trainable depth-2 arm or configuration literal may exist. It does not add a
`model_depth` integer, repeat an inference update, construct an H6 training
arm, alter an H6 endpoint, or change the depth-1 WikiText-103 profile. A source
or oracle result from this probe cannot promote H6, establish scalable depth,
license a WikiText-103 run, or change the primary prediction comparison.

The probe asks one narrower question: do two explicitly normalized causal
layers compose into one normalized joint whose complete ELBO has an auditable
local inventory?

## Frozen cascade law

For each layer `ell in {1,2}`, state and model channels are scalar. The initial
variables are `(z1_0,m1_0,z2_0,m2_0)`. At receiver `t`, layer `ell` owns
categorical state/model sources `(aell_t,bell_t)` and continuous variables
`(zell_t,mell_t)`. The explicit causal parent support is

```text
P_t = {0,...,t-1}.
```

Every one of the four source banks has one strictly positive, normalized row
on `P_t` for every receiver. Its conditioning label is
`causal_latent_prefix_only`: it may depend on the causal latent prefix but not
on `x_t`, a future token, or recognition state. The frozen normalized
factorization is

```text
p(m1_0) p(z1_0 | m1_0)
p(m2_0 | z1_0,m1_0) p(z2_0 | z1_0,m1_0,m2_0)

product_t [
  p(a1_t | prefix1) p(b1_t | prefix1)
  p(m1_t | m1_{b1_t})
  p(z1_t | z1_{a1_t},m1_t)

  p(a2_t | prefix2) p(b2_t | prefix2)
  p(m2_t | m2_{b2_t},m1_t)
  p(z2_t | z2_{a2_t},m2_t,z1_t)

  p(x_t | z2_t,m2_t)
].
```

Every continuous factor is a proper scalar Gaussian regression with positive
variance. The sole emission is a normalized `V`-way categorical row selected
by the sign of an affine half-space in `(z2_t,m2_t)`. This gives a
top-layer-dependent normalized emission while retaining an exact scalar
Gaussian reduction for its recognition expectation.

Layer-1 and layer-2 parameters have distinct immutable ownership labels. This
is a source-level ownership contract only: no optimizer, parameter tensor,
training schedule, or model factory is introduced by this amendment.

## Complete objective

The recognition probe is mean-field over all twelve continuous variables at
`T=2` and has a separately normalized categorical posterior for all eight
source variables. The complete ELBO contains exactly:

1. four initial Gaussian expectations, two per layer;
2. eight source-prior expectations, one per layer/channel/receiver;
3. eight transition expectations, one per layer/channel/receiver;
4. two top-layer emission expectations, one per receiver; and
5. the full recognition entropy: twelve Gaussian entropies and eight source
   entropies.

Thus the frozen local inventory has 42 uniquely named terms. The monolithic
calculation may not consume or resummate those local records; both calculations
must independently equal the scalar oracle.

## Tiny independent source gate

The only frozen development probe is:

```text
T = 2
d_z = d_m = 1 in each layer
V = 3
P_1 = (0)
P_2 = (0,1)
```

The independent oracle imports neither `vfe4`, Torch, nor NumPy. It enumerates
the 16 complete source paths. For each of the four possible emission-region
assignments, it enumerates all `3^2 = 9` token sequences. Scalar Gaussian
normalization, Gaussian expected log densities, threshold probabilities, and
recognition entropies are reduced directly with `math` formulas.

The source gate requires:

```text
source mass = 1
every conditional token mass = 1
all 12 Gaussian factors have positive variance and unit analytic integral
production local sum = production monolithic objective = independent objective
```

The bounded node is:

```powershell
C:/anaconda/python.exe -m pytest tests/oracle/test_h6_depth_oracle.py::test_depth2_cascade_is_normalized_and_complete_objective_matches -q
```

This buildout turn intentionally performs static source checks only. Until the
exact node is run at a frozen source revision, the mechanical gate result is
`INCONCLUSIVE`; source-level implementation does not itself constitute a
scientific PASS. A failure or inconclusive result leaves every trainable
depth-2 arm/config absent.

## Nonclaims

- No H6 promotion or predictive-improvement claim.
- No scalable-depth, attention-equivalence, or long-context claim.
- No optimizer independence claim beyond disjoint declared ownership.
- No H7 covariance, gauge, curvature, or holonomy claim.
- No H8 sparse-scaling claim.
- No WikiText-103 loader, training, recording, or figure change.
- No fallback interpretation in which repeated inference steps count as model
  depth.
