# CONTENT.md change request from E4

New or changed constructions and results in E4's files. Add them where CONTENT.md records the multiscale chapter (c12), the network coarse-graining chapter (c11), and the cut-link section (c10).

1. New `cor:full-graph-adaptive-tower-vfe` (c12, ESTABLISHED). For the adaptive tower with generative input P_0 and composite structural channel C: the tower has the depth-0 evidence and posterior version Pi_0 (x) C; for every tower recognition law Q, F_tower(Q) = F_0(Q_0) + E_{Q_0} KL(Q(.|y_0) || C(y_0,.)) in (-inf, +inf] (`eq:full-graph-adaptive-tower-vfe`); equality for finite F_0 exactly for the pushed law Q_0 (x) C; equal infima; for the pushed law the five terms of `eq:full-graph-vfe-factorized` sum to F_0. Part (ii): if a structural step has a deterministic component phi(Y_{l+1}) = f(Y_l, R_l), any recognition law charging {phi != f} has F_tower = +inf, in particular a coarse factor that gives parent coordinates an atomless law independent of the children. Checked numerically: `remediation/scripts/E4/e4_checks.py`, part A (residual 4.4e-16; singular case +inf).

2. New `rem:full-graph-anchored-star` (c12, ESTABLISHED). The one-level Gaussian star with link transports z_i | z_A ~ N(Theta_{e_i} z_A, Sigma_i), proper apex prior and a Gaussian record of the children, is a declared hierarchy (L = 1) whose nondegenerate Gaussian recognition laws, including product laws with a free parent factor, have finite tower free energy; as for every joint, the tower free energy is at least the free energy of the children's marginal model. Points to Supplement G (`sec:obs-star`) by name only, since the main manuscript has no cref into a supplement.

3. New `open:full-graph-declared-gaussian-hierarchy` (c12). Multilevel declared Gaussian hierarchy (stars of stars) whose depth-l conditionals use inverse tree transports, cut terms along non-loop copies with the links of `def:full-graph-coarse-links`, and PSD self terms; settled by such a law that is also the adaptive tower of its depth-0 marginal through weighted-transported-mean channels with Gaussian noise, for every datum with nontrivial holonomy in a first-level block, or by a counterexample datum. Table row in `requests/open-problems-E4.md`.

4. Changed `open:full-graph-hierarchy` (c12): the closure residual is now defined branchwise (per value of R_l of positive mass and almost every value of the graph data, base = conditional law of Z_l, arrow = conditioned structural channel, product references on V_l and on the branch carrier's vertex set, averaged over the law of the conditioning data). Settling condition unchanged.

5. Changed `prop:full-graph-dropped-datum` (c12): the hypothesis now says that P_N and K_down read only (Z_{l+1}, R_l) while P_R still reads Y_{l+1}; added the second reading (P_R reads only Z_{l+1}), whose cost is sum_l I((R_l, Y_l); N_{l+1} | Z_{l+1}). Checked: `e4_checks.py` part B (1.8e-16).

6. Changed `prop:rg-interaction-native-growth-bound` (c11): added `eq:rg-interaction-dobrushin-growth-bound`, ||D_{n<-l}|| <= (3^{|V_n|}-1) prod_k delta_k with delta_k the Dobrushin coefficient of the tilted reverse kernel, and delta_k <= 1 - exp(-2||[psi_k]||_osc)(1 - delta_k^circ). Checked: `e4_checks.py` part C (bound never violated on 30 random 4->2->1 towers; max excess -0.048; tilt bound max excess -0.032).

7. Changed `lem:rg-interaction-step` (c11): the tilted operator equals U_l also where psi_g is constant on the reverse rows, as for psi_g = f o chi under a deterministic channel.

8. Locator check: `Hoeffding1963` Sec. 4, Eq. (4.16) (cited at `ca_proofs_c11.tex:82`) was read in the primary paper (JASA 58(301), p. 22) in this session: E e^{h(X_i - mu_i)} <= e^{h^2 (b_i - a_i)^2 / 8}. If CONTENT.md marks this locator as pending, it can be marked verified.

9. Convention (c10, `sec:cg-sheaf-closure`): the cut energy is written without the factor one half; W_e and the Lambda blocks are quadratic-form coefficients, and the precision blocks of a law proportional to exp(-E_cut) are 2 Lambda_II, -2 Lambda_IJ, 2 Lambda_JJ. Supplement G's weights and precision blocks are twice c10's.
