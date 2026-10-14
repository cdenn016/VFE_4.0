# Spatial multi-agent VFE simulation design

Date: October 13, 2026. Status: proposed design for the user's review; implementation has not started.

The user requested the design and detailed implementation plan together. This document defines the first spatial realization; the companion plan gives its implementation order. No changes to the Research manuscripts, MultiAgentELBO, or the earlier simulator are part of this task.

## 1. Intended result and scope

Build a click-to-run scientific simulator in VFE_4.0 for at least 512 physical agents, with population size controlled by dictionaries in `run_simulation.py`. Each agent is a local section-bearing object over an open patch of a two- or three-dimensional base, carrying a belief law, a model law, and a principal frame. Agents infer from noisy private observations of a known synthetic spatial field and normalized inter-agent records. This observation choice was confirmed by the user during planning.

The initial base is flat and periodic: mathematically a flat torus `T^d = R^d / (L_1 Z × ... × L_d Z)`, for `d=2,3`. The initial principal bundle is trivialized. The structure group is GL(K,R), with full Gaussian law fibers in its defining representation. The base connection is induced from smooth agent frames relative to a zero reference connection. Independent interaction twists remain separate data or inferred latent variables.

The first release includes both dimensions, spatially extended agents, evolving full Gaussian belief/model channels, selective source recognition, held independent links as an exact numerical reference, an opt-in uncertain-link inference path, CUDA numerical execution, saved runs, diagnostics, static figures, and an optional live snapshot viewer. “Exact” here describes analytical Gaussian expectations and block optima in the declared held-link recognition family, subject to floating-point error. It does not describe the full population posterior when recognition is factorized.

Curved bases, nontrivial bundle topology, other groups, moving supports, and spatial renormalization are extension targets. Their interfaces are designed now; implementing all those realizations is outside the first release. The other agent's pointwise renormalization work remains independent.

## 2. Source context and current workspace

VFE_4.0 currently contains theory-review records under `docs/reviews/`; it has no application package or active simulation configuration. `git rev-parse --show-toplevel` reports that it is not a Git repository. Planning files can be saved here, but this task does not initialize Git, create a remote, or publish anything.

Live sources were read in `C:/Users/chris and christine/Desktop/Research/manuscripts/MultiAgentELBO_Theory/`. The Research checkout HEAD was `f41a0d12cd46df6d5cd5aaba1307c0789f2561f1`; live manuscript bytes, including any WIP, were the planning inputs. MultiAgentELBO was inspected at `9e7be09c55eb446b76edeb9bcc0bea454f40dbd9`. Neither SHA certifies every statement below.

| Live source | Contract used |
| --- | --- |
| `SPEC.md`, section 0 | Main theory is at c*; Supplement B owns base transport. Common frames do not identify connections or independent links. |
| `c02_geometry.tex:222`, `def:geo-agent` | Agent is `(C_i; q_i^b, q_i^m, u_i)`, with law sections over a support. |
| `c02_geometry.tex:308`, `def:geo-graph-links` | Attending directions have independent copies; each copy owns a formal inverse reversal. Agent self links are identity. |
| `b01_base_geometry.tex:263`, `def:geo-connections` | Connections are chosen data, distinct from frames and graph links. |
| `b01_base_geometry.tex:651`, `def:prob-finite-design` | Design contexts and valid agent-context incidences define the finite latent space. |
| `b01_base_geometry.tex:674`, `hyp:prob-context-section-response` | Section response/evaluation needs a declared family; marginal samples alone do not determine it. |
| `b01_base_geometry.tex:793`, `hyp:gen-design-product` | Product prior across design points is an assumption. Records may supply posterior spatial coupling. |
| `c04_generative.tex:94,122` | Kernels cannot read the recognition law being scored; normalized records can attach to cyclic and overlapping scopes. |
| `c06_attention.tex:128`, `prop:obs-attention-elbo` | Constant recognition rows use expected costs, with categorical KL at its proper scale. |
| `c08_two_timescales.tex:94`, `rem:obs-simultaneous-block-updates` | Chromatic updates can descend; unrestricted simultaneous replacements can increase collective VFE. |
| `g02_gaussian_realization.tex:504`, `def:gen-lg-forest` | Proper model priors and linear Gaussian model-to-belief evaluators. |
| `g05_fisher_geometry.tex:49`, `prop:ig-fisher-moment-chart` | Gaussian Fisher metric and affine congruence covariance. |
| `c08b_lie_links.tex:141,446,834,1520` | Exponential twists, Gaussian-based normalized generator prior, link VFE, and Fisher recognition flow. |

The frame-induced connection in this design is a **declared specialization**, not a connection derived by the current Supplement B. B1 explicitly says its connections are chosen and none is inferred from frames (`b01_base_geometry.tex:21`). The latest user request authorizes planning this spatial specialization in VFE_4.0 despite the earlier single-point scope gate in MultiAgentELBO. It does not change that project's direction files.

Planning fingerprints for sections most likely to affect implementation:

```text
b01_base_geometry.tex    SHA256 7FA101EAB02A210989CFC2EF9004B03465B8C63FE09BC77EF264050B1DD2228B
c06_attention.tex        SHA256 234AC5AE359D4D6542CCA398C2B70EAB41DED7AC52672EB052B294F179358E1A
c08_two_timescales.tex   SHA256 BE030C4610C270EC750D9127A9DB801AEF20E13C442F297A5B22D60DA9224877
```

Before implementation, read any changed relevant source section and adjust this contract if needed. Do not treat dated line numbers or manuscript status tags as proof of executable behavior.

## 3. Architecture choice and reuse

| Approach | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Port MAgent_Model-main wholesale | Existing field initialization, manifold helpers, and figures | Dense population-by-population-by-grid work; old consensus objective; broad launcher | Reject as the runtime foundation. |
| Wrap the MultiAgentELBO RG runner per base point | Reuses current pointwise research machinery | Imports pointwise graph/history assumptions and cannot supply sections or spatial coupling by repetition alone | Use only for explicit later exchange or small references. |
| New packed spatial runtime with selected mathematical primitives | Owns spatial semantics, GPU layout, normalized joint, and update schedule | Requires a small new package and focused tests | Recommended. |

Adapt the torus geometry, solve-based frame ratios, Gaussian congruence, and figure styling from MAgent_Model-main. Its live entry is `run_experiment.py:4902`; `gauge_agent/manifolds.py:1042` supplies `Torus`, `gauge_agent/gauge_structure.py:345` supplies `connection_one_form`, and `gauge_agent/statistical_manifold.py:357` supplies full Gaussian KL. Reimplement selected equations in the new batched contracts, with attribution and tests; do not import the old `FullVFE` assembler.

MultiAgentELBO references include `realizations/gaussian/network.py:75` (`assemble_network`), `two_channel_gaussian.py:31,134` (proper two-channel joint and record VFE), `latent_natural.py:13` (`gaussian_blr_step`), and `independent_copies.py:84` (based mutual two-cycles). The dense assembled Gaussian action is a tiny exact oracle. Existing rectangular source banks reject self sources and are not the new ragged-row contract.

No runtime import from a mutable sibling checkout is required. Copy only bounded mathematical implementations where useful, record their source, and keep the new package independently runnable. The pointwise RG integration consumes explicit exported arrays later rather than depending on its changing runner.

## 4. Base, patches, and actual sections

Let `D={c_d}` be an endpoint-exclusive periodic grid. Let `D_i` be the selected collocation nodes of agent i, and `I_D={(i,d):c_d in C_i}`. Store valid incidences only. An observation mask suppresses a record, not the latent variable or agent support. The number of grid points M and number of physical agents N are independent configuration values.

For the initial isotropic grids, choose smooth compact nodal bumps of radius `r0=0.95h`. For d=2,3, `sqrt(d)h/2 < r0 < h`, and require `r0 < min(L)/2`. Use torus distance inside each bump:

$$
\psi_{ir}(c)=\begin{cases}
\exp[-1/(1-\operatorname{dist}_{T}(c,c_{ir})^2/r_0^2)], & \operatorname{dist}_{T}(c,c_{ir})<r_0,\\
0, & \text{otherwise}.
\end{cases}
$$

Declare the agent's open support as `C_i={c: sum_r psi_ir(c)>0}` and the cardinal partition of unity `phi_ir=psi_ir/sum_s psi_is`. This support is an open union of balls, rather than a separately prescribed rectangular domain. It is smooth inside its support, cardinal at grid nodes, and covers gaps between local grid nodes. In anisotropic grids use scaled local coordinates and an ellipsoidal bump with the same normalized radius. No zero-extension smoothness at the support boundary is claimed.

Each channel owns Gaussian coefficients on the whole patch. In the common reference trivialization, define

$$
x_i^x(c)=\sum_{r\in D_i}\phi_{ir}(c)x_{ir}^x,\qquad
q_i^x(c)=\rho_x(U_i(c))_\#\operatorname{Law}(x_i^x(c)).
$$

Default recognition is

$$
Q=\left[\prod_i Q_i^b(k_{D_i})Q_i^m(m_{D_i})\right]
\left[\prod_{i,d,x}\operatorname{Cat}(\beta_{i,d}^x)\right],
$$

with an additional generator factor R in latent-link mode. Each patch channel has full covariance across its nodes. Independence between agents and between the two channels is an explicit mean-field restriction. The generative bridge still couples belief and model coordinates. A full joint channel block can be added later if that restriction needs to be removed.

For reference-frame coefficient covariance blocks `Sigma_i,rs`, section evaluation gives

$$
\mu_i^x(c)=\rho_x(U_i(c))\sum_r\phi_{ir}(c)\mu_{ir}^x,
$$

$$
\Sigma_i^x(c)=\rho_x(U_i(c))
\left[\sum_{r,s}\phi_{ir}(c)\phi_{is}(c)\Sigma_{i,rs}^x\right]
\rho_x(U_i(c))^\top.
$$

This constructs a law on a chosen finite family of smooth random sections and reproduces the nodal marginals. It does not solve the general continuum/refinement-limit problem in `open:prob-continuum-theory`. Interpolating only marginal covariances would omit cross-node uncertainty and is not this model.

An opt-in `covariance_structure='node_blocks'` factors each patch channel across its nodes while retaining full K×K covariance per node. It is a named faster recognition family, not an approximation silently applied to the default. Diagonal K-coordinate covariances are not closed under arbitrary GL(K,R) re-expression and are excluded from the initial faithful path.

## 5. Frames, group representation, and two transports

Use the manuscript convention `u_i(c)=sigma(c) U_i(c)^(-1)`. Passive re-expression is `U_i'(c)=g_i(c)U_i(c)`, with local means transformed by `g_i` and covariances by congruence. Reference-frame coefficients are the authoritative numerical state; local coordinates are evaluated from them. This makes passive frame changes a coordinate operation, with no physical state update.

Initial frame fields are smooth, periodic, and fixed during inference:

$$
U_i(c)=F_i\exp(B_i(c)),
$$

where `F_i` is an invertible seed frame and `B_i(c)` is a configured periodic Fourier field of real matrices. Constant frames are a validation profile, not the only supported initialization. The group backend accepts either determinant component of GL(K,R); continuous frame updates preserve the component on a connected patch. This parameterized field is a selected chart family. No claim that one matrix exponential covers GL(K,R), or even all GL+(K,R), is made.

Choose reference principal connection `A0=0`. Its local form and base transport are

$$
A_i=-dU_i U_i^{-1},\qquad
\Omega_i(c'\leftarrow c)=U_i(c')U_i(c)^{-1}.
$$

Compute endpoint ratios with solves. This connection is pure gauge: its curvature is zero and smooth periodic frames have trivial base-loop holonomy. Nonidentity endpoint matrices do not indicate curvature. Finite-difference curvature estimates can have discretization residuals; exact endpoint composition is the principal check. Differentiate the group frame U, not its algebra coordinates as though matrix exponentiation commuted with differentiation. Changing base Riemannian curvature alone would not change this conclusion about the principal connection.

A later connection provider may supply reference transport `P0(gamma)` along an explicit base path:

$$
\Omega_i(\gamma)=U_i(c')P_0(\gamma)U_i(c)^{-1}.
$$

This separates gauge-frame re-expression, a principal connection, and the base's tangent connection. Curved-base geodesics and tangent transport belong to the manifold provider. Their dimension need not equal K.

At a shared context, interaction copy `e:j->i` has

$$
\Theta_{d,e}^x=U_i(c_d)\exp(\omega_{d,e}^x)U_j(c_d)^{-1}.
$$

Two attending directions use independent primitive copies, each with its own formal reversal. Reversal of a copy uses `-omega`, while the opposite receiver row owns another generator. Agent-level self links are identity and have no free generator. Parallel copies remain individually identified. Graph loop products follow the ordered copy list; mutual two-cycles are based at both endpoints. Their holonomy need not vanish even though the base connection is pure gauge.

Default belief/model channels share one frame and one twist family, as a declared common-link specialization; separate channel twist families are supported by the same record contract. Initial held twists and generator anchors may vary smoothly over overlap patches by the same reference-frame basis construction. Off-grid twist evaluation is an extension of sampled generator fields, not additional observed data.

Frames are coordinate structure in this first realization. Simultaneously re-expressing frames, laws, and factors is not an extra learning mechanism. Treating frames as independently dynamical physical variables would require a separate generative/kinetic declaration and is not silently added.

## 6. The normalized generative model and observations

At every admitted incidence choose proper Gaussian root model law and bridge:

$$
m_{i,d}\sim\mathcal N(a_{i,d},H_{i,d}^{-1}),\qquad
k_{i,d}\mid m_{i,d}\sim\mathcal N(B_{i,d}m_{i,d}+b_{i,d},Q_{i,d}).
$$

All covariances are SPD. The baseline is a product over incidences; spatial records provide posterior dependence. The first bridge is `B=I,b=0` in reference coordinates, with fixed known noise. Model recognition can evolve slowly under this fixed joint. Changing generative parameters is a separate accepted M-coordinate, not accomplished by feeding live recognition marginals into the likelihood.

Attach conditionally independent normalized record kernels. Private records are `o_a ~ N(H_a k_i,d, R_a)` in a gauge-inert sensor space. The observation fixture generates a fixed periodic Fourier field, private sensor maps/masks, noise, and selected communication records. Ground-truth state, source labels, and twists are kept in a separate fixture artifact and never enter inference except through the resulting records.

Within-agent spatial records have reference residual

$$
r_{i,d'd}^x=x_{i,d'}^x-P_0(\gamma_{d'd})x_{i,d}^x.
$$

The default conditions these records at zero as a **declared continuity assumption**, with finite noise; it does not secretly observe the true spatial gradient. Covariance/strength can encode grid spacing and geometric weights, but the full Gaussian record normalizer changes accordingly. Optional measured gradient records are an explicit different sensing task.

For communication, each receiver/context/channel row draws a latent source copy `J_i,d^x ~ Cat(pi_i,d^x)`, then emits one record in a common comparison space:

$$
o_{i,d}^x\mid J=e,Y,\omega\sim
\mathcal N(S_e^Q x_{i,d}^x-S_e^K\exp(\omega_{d,e}^x)x_{j,d}^x,\tau_{i,d}^x W_{i,d}^{x,-1}).
$$

The displayed coordinates are in the reference trivialization. Local-frame maps transform accordingly. Rectangular `S_Q,S_K` are distinct comparison maps; full-state identity comparison is the initial profile. This is one selected-source record per row, not an observation independently attached to every candidate. Noise and record space are common across candidates for the simple expected-energy softmax. Candidate-specific noise is legal only if its log determinant stays in the candidate cost.

The source prior is positive on its declared, possibly directed copy support. Default support consists of agents overlapping at a sampled context, with a smooth geometry-based prior from patch centers and overlap weights. This base-derived locality is an explicit spatial choice. Complete support on each overlap is the reference profile; an optional maximum source count creates a hard structural mask at construction. It never deletes a positive candidate during an E-step based on its current posterior weight.

Include identity self copies by configuration to support singleton rows. Their cost is evaluated using one repeated latent variable: combine endpoint maps before taking covariance. At tied identity comparison, the state residual is identically zero. Do not incorrectly use `Sigma_i+Sigma_i` there. The same repeated-variable rule handles within-agent spatial records with cross-node covariance.

The full joint is

$$
p(Y,J,\omega,o)=p(\omega)
\prod_{(i,d)\in I_D}p(m_{i,d})p(k_{i,d}\mid m_{i,d})
\prod_{i,d,x}\pi_{i,d}^x(J_{i,d}^x)
\prod_a p(o_a\mid Y_{\partial a},J,\omega).
$$

In held-link mode omit the random generator sector. Normalization follows the finite proper latent baseline and normalized record kernels; graph cycles require no Gibbs partition function for the matter/record extension. Keep each Gaussian determinant and observation constant. A record is counted once globally, but its derivatives enter every incident variable's update.

## 7. Collective VFE, exact Gaussian calculations, and update schedule

Compute one objective:

$$
F(Q;\theta)=E_Q[\log q-\log p_\theta(Y,J,\omega,o)].
$$

Report baseline/model prior, bridge, private records, spatial records, communication records, categorical KL, recognition entropy, and link-prior terms separately. The Gaussian entropy is that of each complete patch block. It is not the sum of entropies of its nodal marginals. Avoid relabeling the old consensus-KL energy as this evidence bound.

For a Gaussian record with residual mean `rbar`, covariance `V_r`, record precision W and temperature tau, expected negative log likelihood is

$$
d_e=\tfrac12\left[R\log(2\pi\tau)-\log\det W+
\tau^{-1}(\bar r^\top W\bar r+\operatorname{Tr}(W V_r))\right].
$$

For distinct independent agent blocks `V_r=A Sigma_i A^T+B Sigma_j B^T`. For two nodes of the same block, retain both cross covariance terms. For self/repeated coordinates, merge their linear maps first. The source row optimum is

$$
\beta_e^*=\frac{\pi_e e^{-d_e}}{\sum_{f\in\mathcal N_{i,d}}\pi_f e^{-d_f}}.
$$

The communication contribution is `sum_e beta_e d_e + KL(beta||pi)`. When all candidates have the same Gaussian normalizer this agrees with `softmax(log pi-E[D]/tau)`. Do not multiply categorical KL by tau inside the collective VFE while leaving the rest of the objective unchanged.

Held links and frozen beta give Gaussian best responses. For patch/channel block a,

$$
\Sigma_a^*=\Lambda_{aa}^{-1},\qquad
\mu_a^*=\Lambda_{aa}^{-1}\left(h_a-\sum_{b\ne a}\Lambda_{ab}\mu_b\right).
$$

Assemble only local precision blocks and sparse incident information contributions; never materialize global dense Lambda for large runs. Build conflict colors once from all admitted baseline and record scopes. For source labels held independent under Q, the quadratic Gaussian block objective also has a pair-factor decomposition; any less conservative coloring must be checked against that explicit decomposition. Initial implementation uses the conservative scope coloring.

One sweep: optimize the categorical rows, hold them fixed while traversing belief color classes, refresh rows if configured at this stage boundary, traverse model color classes at their configured cadence, then perform an optional link step. All agent blocks in a color can update on CUDA together. Belief and model blocks of the same agent are not treated as independent simultaneous updates. Receiver and sender contributions are both retained.

The default held-link profile uses exact block optima for beliefs and slower relaxed block optima for models. Damping model Gaussian sufficient statistics gives

$$
\mu'=(1-w)\mu+w\mu^*,\qquad
\Sigma'=(1-w)\Sigma+w\Sigma^*+w(1-w)(\mu-\mu^*)(\mu-\mu^*)^\top.
$$

This is mean-coordinate relaxed CAVI, not explicit Euler Fisher flow. An opt-in Fisher natural-coordinate integrator uses a stated metric and collective-objective backtracking. Neither path uses raw Euclidean Lie-algebra updates. Every accepted deterministic step must satisfy SPD and nonincrease on the same fixed-data objective within a scale-aware floating-point tolerance. Color CAVI needs no speculative line search to establish its mathematical block optimum; the objective check detects implementation/numerical failures.

Inference time is a numerical relaxation parameter, not base position or physical time. Frozen synthetic data is the initial task. Later time-varying sensing needs a declared process/filter and separate frame/output indices; evidence objectives before and after changing data cannot be compared as one descent trajectory.

## 8. Uncertain links and the numerical approximation boundary

The exact held-link stage is required before dynamic links. Then support proper Gaussian recognition of independent primitive generators, with diagonal precision in an orthonormal reference algebra basis as the scalable initial family; full per-copy algebra covariance is opt-in. Each uncertainty is nonzero and SPD. Held parameters are not misrepresented as Dirac recognition with finite KL to a continuous prior.

Use the manuscript's generator prior

$$
p(\omega)=Z_S^{-1}e^{-S(\omega)}\prod_e\mathcal N(\alpha_e,C_e),\qquad
S=\sum_{p\in\mathbb L}c_p Q_p(\omega).
$$

Include both based mutual two-cycles in the loop table. Zero loop coefficients give a normalized Gaussian prior with `Z_S=1`, the first uncertain-link profile. Nonzero loops use a fixed prior, including its chosen reference fiber metric. GL loop defect norms require that metric to transform with reference changes; raw Frobenius norms are not invariant under arbitrary GL conjugation with the metric held fixed.

Matter expectations are analytical conditional on sampled generators; exponentials and loop products require numerical expectations for general nonabelian Gaussian generators. Use chunked, reproducible reparameterized antithetic samples. Derivatives and projected expected Hessians must correspond to the declared integrand. The default diagonal generator family computes only the projected diagonal Hessian; the full-algebra family computes its entire per-copy block. A Gauss-Newton approximation, if introduced later, must have a separately named mode.

The Fisher/BLR proposal is

$$
\Lambda_e'=(1-dt)\Lambda_e+dt\operatorname{pr}_{\Xi}E_R[\nabla^2 i],\qquad
\mu_e'=\mu_e-dt(\Lambda_e')^{-1}E_R[\nabla i].
$$

Reject non-SPD proposals and shrink dt rather than clipping eigenvalues invisibly. Evaluate proposed and old states with the same standard-normal samples during backtracking, and use an independent diagnostic sample stream to report estimation variability. Such acceptance establishes decrease of an estimated objective; it does not prove decrease of the exact expectation. Save sample count, streams, estimator standard error, and rejected-step counts.

For a nonzero fixed loop action, `F=F_without_logZ+log Z_S`. The recognition gradient and differences can omit this one constant while the prior is held fixed. The output must then label its objective as relative VFE and mark the prior normalizer unavailable. It must not report an absolute ELBO or compare different loop priors using that value. Initial release does not learn loop coefficients/anchors/covariances with an unknown parameter-dependent normalizer. Small-dimensional integration can validate the implementation, but it is not a scalable general normalizer algorithm.

This approximation is confined to the uncertain-link mode. The deterministic held-link Gaussian realization remains available as the faithful numerical reference at 512 or more agents.

## 9. GPU layout, numerical policy, and scaling

The inspected interpreter is `C:/anaconda/python.exe`: Torch `2.10.0.dev20251210+cu128`, CUDA runtime 12.8, `cuda.is_available()=True`, NVIDIA GeForce RTX 5090, reported memory 32607 MiB. At inspection 21629 MiB was used and utilization was 67%; no simulation or benchmark was launched. Recheck availability, current workload, and free memory at implementation/run time. Do not compete with the existing GPU job.

All field generation, frame evaluation, Gaussian expectations, categorical reductions, local solves, link sampling, gradients, proposals, and geometry computations run on the selected device. CPU is used for one-time graph coloring if needed, small test oracles, file I/O, and Matplotlib display. `device='cuda'` fails explicitly when CUDA is unavailable; it never quietly falls back to CPU. The user can explicitly select CPU for tiny validation cases.

Define S=number of incidences, E=number of candidate copies, P_i=nodes in patch i, D_i=P_i K. Use packed incidence/candidate arrays and bucket equal patch sizes for batched dense **local** covariance/precision solves. Primitive copy IDs, row offsets, endpoint incidences, formal reversals, channel IDs, and ordered loop tables are separate from base mesh edges.

Core costs are state `O(sum_i D_i^2)` in full patch mode, candidate maps `O(E K^2)` when cached, local factorization `O(sum_i D_i^3)`, and chunked record work. The faster node-block family reduces local covariance storage to `O(S K^2)`. Geometry-only overlap construction uses stencil hashing/spatial cells rather than a full N×N×M overlap tensor. With arbitrary larger patches, full patch covariance becomes expensive even when the population graph is sparse; this is an explicit configurable tradeoff.

Proposed reproducible starting profiles:

| Profile | Physical agents | Base grid | Patch nodes/agent | K | Mean incidences/context |
| --- | ---: | --- | ---: | ---: | ---: |
| `spatial_2d_512` | 512, centers on a 32×16 subgrid | 32×32 | 5×5=25 | 4 | 12.5 |
| `spatial_3d_512` | 512, centers on an 8×8×8 subgrid | 16×16×16 | 3×3×3=27 | 4 | 3.375 |

These are starting configurations, not ceilings. For N not matching the center subgrid, use deterministic seeded center placement and validate coverage. User-defined empty regions are allowed if explicitly configured, with no latent records at uncovered contexts; the default coverage policy requires every design point to have an agent.

In the 2D profile S=12800. The two channel Cholesky states alone use `2*512*100^2*8 = 81,920,000 bytes`, about 78.1 MiB in float64. Complete nonself overlap rows contain 153600 candidate copies for this particular center/patch layout; identity self copies add S. These are analytical array counts, not measured peak VRAM. Expected terms, local precisions, samples, autograd, and plot buffers increase the total. Preflight reports actual S, E, maximum row degree, block dimensions, predicted storage, and available memory; profiling measures the real peak.

By comparison the old dense `(512,512,64^2,8,8)` float32 matrix payload alone is 256 GiB. Merely chunking receiver rows does not remove its dense scalar rows or autograd retention.

Start with float64 geometry/laws and eager Torch for the correctness path. Provide float32 after matched CPU/CUDA comparisons and conditioning tests. Matrix exponentials, Cholesky, log determinants, covariance/precision updates, and objective acceptance stay outside bf16 autocast. Low precision is optional only after a measured accuracy budget. Seeded CUDA runs are reproducible within a declared numerical policy; seed equality alone does not guarantee bitwise equality across devices.

Use Cholesky factors and solves, sorted row reductions, cached fixed-factor terms, and one batched numerical status check per stage. Avoid `.item()` per agent/edge. CUDA acceptance, progress display, and timing require deliberate synchronization points; eliminating all synchronization is not a correctness goal. An explicit deterministic mode must either use supported deterministic reductions or fail with the offending operation named.

PyTorch documents that [cholesky_ex](https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html) can avoid synchronization with `check_errors=False`, whereas [solve](https://docs.pytorch.org/docs/main/generated/torch.linalg.solve.html) synchronizes CUDA and points to `solve_ex`. The installed build exposes both `_ex` operations. [scatter_reduce_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.scatter_reduce_.html) documents potential CUDA nondeterminism. Test selected kernels in the installed build; newer documentation is not a guarantee about this nightly installation. `torch.compile` is a later opt-in optimization after profiling pure stable-shape kernels; it is not required for the first functioning simulator.

Never silently repair a failed Cholesky with Euclidean `epsilon I` under an arbitrary GL coordinate change. Reject the update and record the failed block. Any opt-in covariance floor uses a declared reference SPD metric, transformed by congruence, and records that it changes the selected model or numerical path.

## 10. Configuration, package boundaries, and outputs

Create a small package named `vfe4`. `run_simulation.py` owns the editable dictionary and calls `load_config(CONFIG)` then `run_simulation(config)`. There is no CLI parser. Unknown keys, incompatible dimensions, invalid SPD factors, nonpositive tau, empty source alphabets, unsupported groups/manifolds, and estimates exceeding the configured memory allowance fail before a large allocation.

Configuration families: runtime/device/dtype/seed; manifold/dimension/periods/grid; population/count/placement/patch width; bundle/group/K/frame field; recognition/covariance structure; generative model/bridge; observations/field/private sensing; communication/maps/source prior; links/held or latent/prior/loops; spatial records; inference/schedule/rates/cadence; output/viewer/checkpoints. Frozen resolved config, not just the editable dictionary, is saved with every run. Schema validation does not add hidden overrides.

Core modules and responsibilities:

```text
run_simulation.py             editable CONFIG and click-to-run main guard
vfe4/config.py               typed config, explicit validation, device resolution
vfe4/state.py                domain/incidence/row/copy/patch-Gaussian records
vfe4/geometry/base.py        BaseManifold protocol and FlatTorus
vfe4/geometry/groups.py      GroupBackend and GLReal operations/representations
vfe4/geometry/sections.py    compact basis, support incidence, section evaluation
vfe4/geometry/transport.py   frame fields, base ConnectionTransport, interactions
vfe4/model.py                normalized Gaussian baseline and record banks
vfe4/observations.py         synthetic truth, sensors, records, masks
vfe4/gaussian.py             batched expectations, local precision algebra, damping
vfe4/objective.py            termwise collective VFE and ragged source optimum
vfe4/inference.py            coloring, chromatic CAVI, accepted geometric steps
vfe4/latent_links.py         generator recognition, samples, projected Fisher/BLR
vfe4/simulation.py           initialization and sweep orchestration
vfe4/io.py                   resolved-config/metrics/checkpoint/run exchange
vfe4/visualization.py        snapshot viewer and publication figures
tests/                      tiny CPU numerical/geometry/contract tests
experiments/run_scaling.py   config-driven large-N CUDA measurements
```

The base/group/connection protocols have one working implementation each at first. Do not fill the repository with unsupported stubs. Future geometry plugs into coordinate/volume/path operations and an atlas; future groups plug into representation/algebra operations. SU(N) or GL(K,C) also needs a declared complex Gaussian or realification convention; changing a group name alone is insufficient.

Save under `runs/<timestamp>-<profile>/`: `config.json`, environment/source provenance, geometry/incidence/copy tables, fixed observations, initial/final states, `metrics.csv`, requested checkpoints, and PNG/PDF/SVG figures. Checkpoints include row distributions, link uncertainty, frame parameters, RNG states, estimator streams, iteration/cadence position, and normalizer status. Resume with the stored model; changed model/data configuration starts a new run. Exact resume is tested within the same supported environment.

The viewer consumes detached snapshots at a configured cadence and shows patch centers/supports, a selected agent's belief/model field, truth and error, uncertainty, selected attention edges, and VFE terms. For 3D use selectable 2D slices plus a limited 3D agent overview. Initial controls are pause, resume, single sweep, and stop. Plotting downloads only selected aggregate/slice data; it never sends all candidate matrices to CPU. Headless runs retain the same inference and output contracts.

Gauge-sensitive coordinates are labeled. Compare physical field errors in the fixture's reference/sensor frame, and use metric-aware uncertainty/holonomy diagnostics. Plotting `trace(Sigma)` or ordinary eigenvalues without declaring a frame is not a GL-invariant diagnostic.

## 11. Boundary with pointwise renormalization

An export can select a context, preserve its physical agent/copy IDs, reference-frame laws, priors, comparison maps, observations, source rows, and link recognition. Spatial factors couple contexts. A marginal context slice of recognition is not automatically a normalized independent generative model for that context.

Exports therefore explicitly carry one of: a context model conditioned on other contexts with conditioning values recorded; a model with spatial factors disabled by declaration; or a marginal whose induced factors are actually retained. Only the first two simple cases are required initially. The pointwise worker decides its supported retained-state import; VFE_4.0 does not claim spatial RG or infer a hierarchy by clustering fields. No shared-file edits or messages to that other chat are needed to implement this boundary.

## 12. Validation and delivery criteria

Tiny CPU tests validate geometry, section evaluation, covariance/entropy, repeated-variable records, exact row updates, block CAVI, source-side effects, SPD rejection, config reachability, and checkpoint continuation. Independent dense Gaussian assembly and low-dimensional integration are validation tools; they do not determine the large-network scientific direction.

GPU tests use `C:/anaconda/python.exe` and the project's explicit CUDA lane. Large 512/1024-agent cases are config-driven experiments, never unit tests. Record JUnit results for test execution and machine-readable measurements for scaling. Validate CPU/CUDA agreement at small size and measure 2D/3D peak memory and sweep times once a free GPU window exists. No performance or descent result is claimed by this plan.

The first release is ready when the user can edit N, K, dimension, patch size, data/noise, and learning schedules in one dictionary and run; all required configurations reach their selected paths; both periodic dimensions produce evolving sections and source rows; accepted held-link sweeps reduce the declared fixed-data VFE within numerical tolerance; uncertain-link outputs identify their approximation and normalizer status; all large numerical tensors stay on CUDA; and a saved run can be inspected and resumed. The 512-agent requirement is established by actual runs in both dimensions, with one 1024-agent sparse case establishing that N is not hard-coded. Timing and VRAM targets remain measurements, not promises.

## 13. Proposed decisions for review

Confirmed: synthetic spatial field with noisy private observations and inter-agent records. Proposed defaults: fixed overlapping supports; 2D first, then 3D before release; K=4 for initial profiles; two Gaussian channels; full cross-node patch covariance; fixed smooth periodic frames inducing the pure-gauge base connection; independent directed interaction twists; static generative noise and source priors initially; color-sweep inference; held links as the default exact reference and uncertain links opt-in. No moving-agent mechanics, dynamical base curvature, or independent physical frame learning is assumed.

The detailed implementation order, interfaces, concrete tests, and acceptance conditions are in `../plans/2026-10-13-spatial-multi-agent-implementation.md`. Implementation begins only on the user's subsequent instruction, as requested.

## 14. Suite scope (revision of October 13, 2026)

The user asked for a simulation suite built in a manner similar to `MAgent_Model-main`, and stated that the old suite is a guide rather than a feature list. The suite therefore adopts the old project's shape (click-to-run launchers with named presets, self-contained run folders, a metric recorder and numerical monitor, multi-seed runs with bootstrap aggregation, publication figures and a viewer) and admits scientific features only where the live theory defines them.

The engine runs in three modes. The single-point mode has one design context `c_*` (SPEC section 0), one incidence per agent, constant frames, no spatial records, and locality only through a declared agent-lattice source prior (ring or periodic 2D/3D lattice; window, exponential, power, complete and directed-window families). It is the theory's own setting and is unaffected by the MultiAgentELBO base-transport gate. The spatial 2D and 3D modes are the construction of sections 4 through 7.

Observables recorded by the suite, each labeled exact or a declared mean-field analogue: termwise collective VFE; row entropy, effective degree, self weight, and descriptive reciprocity; coherent-block partitions at declared resolutions (the analogue of `Coh_a` in c13, evaluated on transported recognition means); the fixed-sector rank of the connection Laplacian (c10 `prop:cg-kernel-holonomy`); metric-aware loop holonomy defects; the discarded conditional KL of a declared linear block channel (c09 `thm:rg-exact-coarse-vfe`), computed only when the dense posterior of the covered blocks fits a configured size; and field error in the fixture frame.

Slow source-prior learning is an opt-in M-coordinate, the explicit Euler step of `eq:phil-source-learning` (c14), which is natural-gradient descent of the free energy in each prior row under the simplex Fisher metric. Every learning experiment runs a matched frozen arm (SPEC R13). The engine's mean-field CAVI differs from the stochastic kinetics of `def:rg-formation-cycle-experiment`, so its formation runs are descriptive and do not test `conj:phil-contrast-formation`.

Excluded, because the theory does not define them or leaves them open: Hamiltonian belief dynamics and mass matrices, the ouroboros tower and apex closure, adaptive meta-agent formation dynamics, fitted RG flows and exponents, and entropy-production detectors on a deterministic relaxation. Bounded-confidence and DeGroot comparisons are a possible later baseline, without a separation claim.
