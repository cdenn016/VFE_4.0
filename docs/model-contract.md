# Finite simulation model and execution contract

The source is the live `Research/manuscripts/MultiAgentELBO_Theory` collection: b01 finite incidence design, c04 normalized joint and records, c06 row recognition, c08 controlled block inference, c08b generator priors/BLR, c09 exact coarse VFE, c10 connection-Laplacian kernel, c13 coherent partitions, and c14 source-prior flow. This implementation chooses one finite Gaussian realization; it does not assume every conjecture in those manuscripts is established. The earlier MAgent_Model supplies geometric/visual precedents, rather than the objective.

## Sections, frames, and records

The spatial design is a periodic endpoint-exclusive grid. Agent patches are periodic stencils about configured centers. Their local sections use normalized smooth cardinal bump functions on the open union of bump supports. A coefficient Gaussian has a full covariance over all nodes of one agent patch. Evaluation pushes the complete covariance through the basis matrix and the fixed frame U(c). `evaluate_section` returns mean, covariance, and an explicit support mask. Its values outside support are placeholders, not a defined section. The current uniform stencil builder also packs coefficient laws by patch cardinality so the state can support ragged patches supplied by a later geometry backend.

Coefficient coordinates and every comparison metric are in a fixed reference frame. The reported field RMSE is therefore explicitly in that frame. The spatial difference record compares reference coefficients at neighboring contexts of the same agent; in agent charts this is precisely transport by the induced flat connection. With A0=0, A=-dU U^-1 and Omega(y,x)=U(y)U(x)^-1. Periodic Fourier frames are smooth and periodic. Global periodic frames and this reference connection have trivial base holonomy. Flat base geometry does not imply that the independently twisted inter-agent network has trivial holonomy.

Held attending transport is Theta_ij(c)=U_i(c) exp(omega_e) U_j(c)^-1. Each nonself receiving direction owns its own primitive, even when the opposite direction exists. Self copies are identity and own no generator. GL operations admit either determinant component, while default exponential frame initialization and generator twists are in the positive component. The implemented loop library contains both based mutual two-cycles. It does not add longer cycles or identify independent copies as inverses. The reference fiber metric for the loop defect is identity; multiplying all frames is a chart change, not a new metric.

At each incidence the proper root is m~N(0,s_model^2 I), and the bridge is k|m~N(m,s_bridge^2 I). Private observations are normalized Gaussian records. Spatial continuity is a normalized zero-difference record. Each communicating receiving row has a categorical source a~pi and one normalized Gaussian observation o|a~N(A k_receiver-B_a k_sender,tau I), in the chosen channel. Full comparison uses identity; projection uses the first R reference coordinates. Synthetic generators create one hidden source and one observation per row and then repeat that same observation across its source candidates. Receiver and source variables are never sampled from live peer recognition laws in the generative model.

## Recognition and update scope

The default recognition law is the product over agents of separate belief-patch and model-patch Gaussians, categorical row laws, and (when enabled) per-copy generator Gaussians. It allows cross-node covariance within a patch, but no cross-agent or belief/model covariance. `node_blocks` changes that family by deleting cross-node recognition covariance; generative spatial records remain present. Neither family is the complete posterior.

The collective VFE retains every Gaussian determinant constant, Gaussian entropy, categorical KL, receiver term, sender term, and repeated-variable covariance. A self record merges its maps before assembly. The optimal categorical row is softmax(log pi minus expected normalized NLL), without a separate entropy temperature. Tau scales both quadratic and Gaussian normalizer. The dense Gaussian oracle is restricted to held generators and a configured dimension cap. A beta-weighted action is an expected variational action; an actual generative Gaussian posterior requires singleton source alphabets or explicitly conditioned labels.

Held inference updates categorical rows, then hypergraph-colored belief agents and scheduled model agents. CAVI damping interpolates Gaussian sufficient statistics (including the between-mean covariance). The full-patch Fisher method interpolates precision/information coordinates. Every accepted collective sweep checks its VFE. A tiny three-variable SPD counterexample grounds the scheduling requirement; it makes no large-N formation claim.

Latent generators use proper Gaussian recognition in flattened Frobenius coordinates, diagonal or full per-copy precision, and a normalized zero-mean Gaussian anchor. Matter expectations use both E[exp(Omega)] and E[exp(Omega)^T G exp(Omega)]; exp(E[Omega]) is not substituted. Derivatives are gradients and true Hessians of each nonlinear record/loop factor, followed by projection to the declared per-copy family. Cross-copy loop Hessian blocks are discarded by that factorization, while loop gradients still depend on all participating sampled generators. The update is P'=(1-dt)P+dt H and mu'=mu-dt P'^-1 g. Non-SPD proposals shrink the step. Acceptance backtracks on common samples of the complete estimated objective and can retain the original link state after exhausting retries. This is an estimated-descent statement, not exact expectation descent. Raw Monte Carlo history can rise when integration points refresh; paired before/after fields describe acceptance.

The Gaussian anchor KL is analytical. A loop coefficient of zero has logZ=0; nonzero loop coupling has unknown logZ and gives `relative_vfe_unknown_logZ`. `estimated_vfe_known_normalizer` identifies sampled likelihood expectations with the normalized Gaussian anchor. Independent antithetic diagnostics use a distinct checkpointed RNG stream; standard error is computed across independent pair means. The paired integration estimator's error field is NaN because no independent uncertainty estimate is attached to those exact sample points. The separate diagnostic fields state their own mean, error and pair count.

Source learning uses pi'=(1-rate)pi+rate beta, after scheduled matter updates, at fixed Q. Rate zero leaves log priors bitwise unchanged. Positive rates preserve normalized structural candidates in exact arithmetic. If rate one encounters a numerically zero beta, the implementation fails rather than silently deleting structural support. Learned/frozen comparisons share the seeded asymmetric synthetic population and noisy private data. This breaks observational symmetry explicitly. These deterministic CAVI experiments do not test the c14 stochastic kinetics conjecture.

## Observables and exchange

| Observable | Theory connection | Recorded meaning |
|---|---|---|
| VFE terms, paired acceptance | c04/c08/c08b | Exact held expectation, or declared sampled/relative objective |
| Row entropy, exp(entropy), self weight | c06 | Descriptive categorical statistics |
| Field RMSE | synthetic fixture | Gauge-coordinate-dependent reference-frame error |
| Mean-field coherent block counts | c13 coherence relation | Connected closure of transported mean differences below a declared resolution, at context 0 |
| Fixed-sector rank | c10 connection-Laplacian kernel | Dimension from rooted transport constraints for held full-state maps and strictly positive record weights, at context 0 |
| Based-loop defect | c08b loop action | Mean-generator realization and identity reference fiber metric; no uncertainty claim |
| Reciprocity | c08b directed copies | Mean absolute beta_ij-beta_ji over available reverse copies at context 0 |
| Discarded conditional KL | c09 exact coarse VFE | Exact Gaussian KL loss under context-average reduction only when source alphabets are singleton, links held, and dense dimension admitted |

Unavailable discarded KL is marked explicitly; the general source-label posterior is a mixture and is not replaced by a factorized Gaussian. Fixed-sector rank is unavailable for projected/uncertain comparisons. Coherence and rank concern the declared context's active agents, not an unexplained whole spatial graph.

Context exchange conditions every source row on one recorded candidate and latent generators on the recorded mean realization. It restricts the complete joint action to recorded omitted coefficient values, retaining boundary-only normalization constants. It conditions the original patch recognition law on those same values. J,h,c use the ordering of all retained belief coordinates followed by all retained model coordinates. The payload includes the original records, source/copy tables and generator recognition, so the conditioned action can be traced to its original model. It is consumable through MultiAgentELBO `GaussianAction`, whose action is x^T J x/2-h^T x+c. No generic importer was found in that repository; this suite adds no dependency or modification there.

## Configuration paths

All keys originate in `presets.DEFAULTS`, merge through `strict_merge`, and pass `config.load_config`. Unknown keys fail. The following paths execute them; a key may be conditional on its named mode.

| Keys | Executing path |
|---|---|
| runtime.device, dtype, seed, deterministic | config loader; geometry/model RNG construction; Torch deterministic setting |
| runtime.max_vram_gb | simulation CUDA admission and preflight |
| manifold.kind, dimension, periods, grid_shape | geometry.base.build_domain/FlatTorus |
| population.n_agents, lattice_shape | point support, frame creation, scheduling |
| population.placement, center_grid, patch_shape, coverage | geometry.sections.build_incidence |
| bundle.group, K | loader GL_R admission; Gaussian/GL shapes throughout |
| bundle.frame_field, frame_spread, frame_amplitude | geometry.transport.build_frame_field; amplitude applies to Fourier frames |
| recognition.covariance_structure | section initialization, marginalization and inference covariance branch |
| generative.bridge, bridge_std, model_prior_std | loader identity bridge admission; objective and local precision assembly |
| observations.kind, private_std, private_fraction | observations.synthetic_truth and model.build_model |
| communication.comparison, comparison_dim, channels | model comparison records/channel rows; comparison_dim applies to projection |
| communication.source_support, prior_family, radius, decay, alpha | geometry.supports; window radius uses toroidal Manhattan distance at a point, and minimum spacing units over overlapping patches; exponential/power parameters apply to their families |
| communication.include_self, max_sources | copy-table identity insertion; overlap source truncation before normalization; max_sources is rejected for point mode |
| communication.tau | Gaussian record precision/normalizer and transport second moments |
| communication.source_rate, source_every | inference source-prior step |
| links.mode, shared_channels, initial_twist_std | model primitive allocation; record statistics; inference link schedule; fixture generator realization |
| links.prior_std, generator_precision | initial recognition precision and latent Gaussian anchor KL/BLR |
| links.mutual_loop_coefficient, samples, copy_chunk_size | latent expectations, true derivatives, fixed loop prior and independent diagnostics |
| spatial.enabled, record_value, noise_std | model spatial-pair record builder; zero record admission |
| inference.method, sweeps, belief_weight, model_weight, model_every | simulation loop, chromatic matter updates and damping |
| inference.link_every, link_dt | scheduled latent generator proposals/backtracking |
| analytics.every, coherence_resolutions, coarse_max_dim | simulation observable cadence, partition thresholds, dense oracle/exchange admission |
| output.directory, viewer, snapshot_every, checkpoint_every, figures | run writer, detached viewer, checkpoint schedule and static rendering |

Group, manifold, frame-field, and reference-connection operations have separate modules. Adding curvature requires chart/path-aware reference transport, quadrature/design incidence and boundary transition maps, then assembly of compatible records. Adding a group requires its representation, algebra parameterization, exponential/retraction and precision-coordinate metric, then the generator derivatives in that basis. Only GL_R and the point/flat-torus bases are admitted now; names of unimplemented backends do not fall back to another model. Refining the spatial grid changes the finite likelihood design; cell volumes are geometry metadata, not an unannounced continuum rescaling of record noise.
