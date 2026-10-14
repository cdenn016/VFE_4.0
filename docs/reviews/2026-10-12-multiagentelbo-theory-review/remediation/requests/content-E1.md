# CONTENT.md changes requested by E1

Changed constructions:

- `def:geo-graph-links` (c02): the copy set is a structural datum declared before the rows; a mutually attending pair uses two distinct copies with independent links (Theta_ij != Theta_ji^{-1} in general), forming the closed walk i->j->i with holonomy Theta_ij Theta_ji; endpoint notation Theta_ij is the link of the source copy j->i of row i; loop copies admitted; agent-level self edge is a loop copy with identity link; coarse loop copies carry internal holonomy (`def:full-graph-coarse-links`).
- `def:obs-attention-label-model` (c06): same conventions restated at the rows; energies typed at the description level of the declared model (sample, law point, or configuration); D_ii = 0 is a consequence of the identity self link for transported-residual or transported-divergence energies, not for energies with distinct query and key maps.

New or extended results (all ESTABLISHED, proofs in `ca_proofs_c06.tex`):

- `prop:obs-selective-receiver-update`(b): dot-product form of the comparison-record weights, `eq:obs-selective-record-dot-product`, metric (Sigma^pred)^{-1}.
- `prop:obs-selective-receiver-update`(d): restricted to a common comparison space (distinct spaces need a declared common record space).
- `prop:obs-selective-receiver-update`(e): self copy; weight and component `eq:obs-selective-self-component`; for tied query and key maps the self component is an attention sink with factor `eq:obs-selective-self-sink`, det(I + W~ S^Q Sigma S^Q^T)^{1/2}, and residual weight w_ii I + (1 - w_ii) Sigma^+ Sigma^{-1}.
- `prop:obs-selective-receiver-update`(f): exact H-head receiver update `eq:obs-selective-multihead-update` with covariances `eq:obs-selective-multihead-covariances`; per-head rows are marginals of a coupled label table; operator form of `eq:obs-selective-attention-residual`.
- `prop:obs-attention-constant-row-gap`: self-copy clause of the random-source logit shift (self logit raised by Tr(W Cov Y_i)/(2 tau) at the identity self link).

Correspondence table `tab:obs-transformer-correspondence`: each row names its model (label, law-point, comparison-record); new comparison-record query/key entry; new positional-encoding row (ALiBi as log pi_ij, derived; RoPE as a coboundary twist of the comparison space, declared); multi-head row now "derived at one scale, recursion open". The text before the table names the three rows the theory defines (generative posterior row, constant recognition row, predictive-covariance row) without choosing one.
