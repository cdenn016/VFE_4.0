# B-CTX remediation log (2026-10-29)

Files edited (manuscript root `Research/manuscripts/MAgent_Theory_2.0/`, branch `magent-2.0-remediation-2026-10-29`, nothing committed, nothing compiled): `chapters/15_contextual_coupling.tex`, `chapters/16_link_base_transport.tex`, `companion/c15_contextual_coupling.tex`, `companion/c16_link_base_transport.tex`. Edit and check scripts: `remediation/scripts/B-CTX/` (string-replacement scripts and `check_tex.py`, which reports balanced environments and braces, even `$` per paragraph, and no `\,`, `\;`, `\!`, `---`, `\eqref` or "ELBO" in the four files). Bibliographic data were checked against Crossref records (DOIs below); Lee 2018 and Taylor 2011 against their Springer DOI records.

Section 3 items located here: none of H01 to H18 or I01 to I04 is located in these files. H18's per-chapter objective statement is applied (ch15 states the reciprocal potential, with the own-block remark as the per-agent exception; ch16 states that its selectors are kinematic and belong to no objective). No construction in these chapters selects membership, so D1 to D3 do not apply; ch15 now says so explicitly.

| ID | Status | What changed / reason |
| :- | :- | :- |
| D19-CTXF-01 (S3-M09) | applied | ch15: regular critical set `ctx:f:eq:regular-set` and its global part R_glob defined after `ctx:f:ass:branch`; `ctx:f:thm:ode` now states the maximal solution lifted with its transports on R, continuation in compact subsets of R (in transports as well as laws), and that C_* is the minimized cost only on R_glob; leaving R_glob is not an endpoint and is undetected. New subsection "A regular branch can stop being the minimizer" with the location-covariance SO(2) monodromy witness (costs 0 and 0.18, Hessians 14.1525 and 13.9725; reconstructed by hand: covariance Fisher term 14.0625 plus or minus |mu|^2 = 0.09; branch returns at theta = pi by symmetry and an IFT bound). Opening sentence and "What the contextual energy controls" qualified to R_glob. Companion c15: branch hypothesis paragraph and well-posedness sentence updated. |
| D19-CTXF-02 (S3-M10) | applied | ch15 after `ctx:f:global-branch`: GL(K) stabilizer {A: A mu = mu, A Sigma A^T = Sigma}, dimension (K-1)(K-2)/2 for mu != 0 (O(K) for mu = 0); branch hypothesis fails at every consensus and one-agent context for K >= 3; stratum flow is different dynamics without active invariance. Theorem stated at its proved scope (free non-consensus configurations). Companion updated. Scope paragraph of ch15 updated. |
| D19-CTXF-03 (S3-X09) | applied | Compact-group paragraph replaced by `ctx:f:global-branch` (compact G, or proper action on complete N), with proof via the 1-Lipschitz bound, Hopf-Rinow compactness and a gap; near-agreement statement extended to proper free actions; Gaussian completeness and properness of GL(K) stated with reasons. The "unique minimizing transported law modulo a discrete stabilizer" generalization of the lens is not adopted; the proposition requires a unique minimizer. |
| D19-CTXF-04 (S3-X10) | applied | New `ctx:f:horizontal` (stationary transports give horizontal endpoint forces; Noether momentum vanishes along the geodesic), with proof; no-torque consequence after `ctx:f:eq:flow`; orbit-distance identity `ctx:f:eq:orbit-distance` (graph Dirichlet energy into Q on R_glob, O'Neill cited). |
| D19-CTXF-05 (S3-X11) | deferred | Plaquette holonomy proposition is an extension (nonabelian sign not checked by the lens). No text change. |
| D19-CTXF-06 (S3-X12) | applied as remark | `ctx:f:divergence-cost`: which proofs use only smoothness, invariance and positive Hessian; invariant KL or symmetrized costs; orientation; same infinitesimal normal equation; global version needs its own compactness argument (the Lipschitz bound is distance-specific). Theorems not restated for divergence costs (extension). |
| D19-CTXF-07 (S3-X36) | applied | ch15 after `ctx:f:eq:cusp`: Danskin statement, no single descent direction, uniqueness not available; branch hypothesis restores it. Companion cusp paragraph updated. The lens's five-trajectory numerical claim is not put in the text. |
| D19-CTXF-08 (S3-L12) | applied | Schonemann1966 and Kendall1984 cited at `ctx:f:eq:procrustes` and in the orbit-distance paragraph; companion likewise. Gower and Kabsch not added. |
| D19-CTXF-09 (S3-L13) | applied | (a) cross-context agent identification stated; (b) stability under all of G; (c) location-family sentence clarified (not totally geodesic); (d) companion finite part now uses \mathcal E_v and g^row, and introduces V(x,p,b) for the continuum. |
| GAP10-S3-01 (S3-M11) | applied | Own-block remark rewritten: per-agent objective named; requires each F_iv (with record terms) C^2 jointly on the chart; states that regularity of the reciprocal sum does not transfer, with the lens's counterexample A in one sentence (checked: D_12 + D_21 = (m1 - m2)^2 + |m1|^(7/2), own-block derivative contains |m1|^(3/4)). Hypothesis at `ctx:f:eq:local` and in `ctx:f:thm:ode` lowered from C^3 to C^2 (sufficient for a C^1 field; the branch energy is smooth since metric and action are). Companion own-block sentence updated. |
| D20-CTXC-01 | applied in part | Misleading sentence "The law map itself will generally have only H^m regularity" replaced: H^m law map for H^m frame data; persistence of H^s frames neither asserted nor excluded. Persistence proposition deferred: the lens supplies a sketch that needs a mollifier argument, not a short checkable proof. |
| D20-CTXC-02 | applied as remark | ch15 after `ctx:c:reduced-energy`: any locally gauge-invariant contextual energy has horizontal field-metric gradient (differentiate along exp(a xi(x))), so frame diffusion is impossible after elimination; admissible gauge-invariant row term noted as a constitutive choice not made. Companion sentence. Scope paragraph updated. |
| D20-CTXC-03 (S3-X15) | deferred | Chart-free and shape-controlled continuation needs H^m x-dependence of data and tame estimates at the higher threshold; not proved here. Existing text already says a chart exit may be removable. |
| D20-CTXC-04 (S3-X16) | applied (b1) | Finite-time singular-orbit witness in the polar fixed-reference example (reconstructed: phi = pi invariant, u = -a + (u0 + a)e^{-t}, t* = log(1 + u0/a), Cartesian law smooth through 0). Scope paragraph and companion mention it. (b2) not added. |
| D20-CTXC-05 (S3-X17) | deferred | Uniform-in-lambda theory and O(sqrt lambda) limit are an extension; text already disclaims a uniform singular-limit theorem. |
| D20-CTXC-06 (S3-X37) | applied | Threshold lowered to s > d/2 + 1 (m > d/2, m >= 1) after reconstructing every use: algebra, embedding, Moser composition, Lipschitz bounds, the energy identity (F_* is C^1 on H^s x H^m x H^m for m > d/2, m >= 1), reconstruction H^m . H^{m-1}. d = 1 example now s = 2, m = 1. Existence time dependence stated in the theorem. Reason at the energy-identity proof corrected. C^infinity data-to-solution map not claimed. Companion theorem updated. |
| D20-CTXC-07 (S3-L14) | applied | Witness described as the linear radial sector (flat ray, V_r = 0, no frame dynamics, pure gauge, singleton rows) in ch15 and c15. |
| D20-CTXC-08 (S3-X38) | applied in part | "a one-dimensional base gives no curvature witness" replaced by the base-circle holonomy statement (trivial for every periodic phase). d = 2 curvature formula and Hopf-cone geometric-phase witness deferred (extension). |
| D20-CTXC-09 (S3-L15) | applied | Companion phase restriction removed; V smooth on the open regular chart in both texts; companion uses Ad_g and E_0. |
| D20-CTXC-10 (S3-L16) | applied in part | Base written \mathbb T^d (no longer C); frame forcing Q(z) -> \Phi(z); golden constant tau -> \varsigma; tangent V -> W in `ctx:c:eq:collective`; witness matrix V_0 -> E_0; witness Hessian H -> \mathsf H; row Fisher matrix B -> \mathsf B. Horizontal coefficient C and the continuity class CH^j kept (renaming C^alpha_a would touch every continuum equation; the base collision, the confusing one, is removed). |
| D20-CTXC-11 (S3-L17) | applied in part | Taylor2011 (Chapter 13) cited for Moser composition; Lee2013 Theorem 21.10 cited for the quotient manifold theorem. Cassel journal version: references.bib is C-SUM's (see FOR OTHER OWNERS). Moser 1966 not cited (no publisher record verified). |
| XC7-IMPL-07 (S4-M16) | applied | Within `ctx:f:divergence-cost`, with PineleStrapassonCosta2020; companion. |
| XC7-IMPL-08 (S4-L40) | applied | ch15 after `ctx:c:eq:matrixflow`: undamped vertical cancellation; adapted chart or exact vertical part; horizontality check. |
| XC5-LIT-03 (S4-M20) | applied | Wilson1974 at `ctx:f:eq:endpoint`, at the start of the ch16 reversal section (replacing "classical"), and in both companions; LMW described as a discrete-connection formalism. |
| XC5-LIT-04 (S4-M21) | applied | Gauged sigma model attribution (DAddaLuscherDiVecchia1978, BandoKugoYamawaki1988, `open:geo-gauged-flow`) after `ctx:c:reduced-energy` and in c15; orbit distance and Procrustes as above. |
| XC5-LIT-15 (S4-L38) | applied in part | Orbit-distance identity stated; Gamma-convergence refinement deferred (extension). |
| XC3-SCOPE-04 (S4-L09) | applied | "restricted-family flows (`def:dyn-restricted-flow`)" in `ctx:f:thm:ode`, the zero-coupling paragraph, and c15. Abstract part: FOR OTHER OWNERS. |
| XC4-CONS-07 (S4-L12) | applied | Row temperature in `ctx:c:eq:logrow` and c15 written T_a, distinct from the horizon T. |
| XC4-CONS-12 (S4-L16) | applied | All 38 \eqref in ch15 converted to \cref/\Cref/\crefrange; "Lemma \ref" -> \Cref; three "ELBO" in ch15 and one in c15 replaced by "variational free energy"; the lowercase sentence-initial \cref at 16:222 removed with the rewritten paragraph. |
| XC4-CONS-14 | applied (ch15 part) | Generated labels renamed (see labels). |
| XC1-GAUGE-02 (ch16:113) | applied | Reference consistency of the message rule identified as naturality, not active symmetry. |
| XC1-GAUGE-07 (ch16:306) | applied | Tied reciprocal parts across sectors: compute the reciprocal part in G, then represent. |
| XC1-GAUGE-09(d) | applied | ch15 after `ctx:f:eq:comparisons`: h_v is a target change, inverse of the ch16 reference change. Parts (a) to (c) are in files of other owners. |
| XC1-GAUGE-10, D21-BT-11 (S3-L24) | applied | Narasimhan1962 replaced by NarasimhanRamanan1961 and NarasimhanRamanan1963 in ch16 and c16; Smale1970 cited at the alignment mechanical connection; Davis-Kahan and Bjorck-Golub with D21-BT-03. Wilczek-Zee not added. |
| D21-BT-01 (S3-M12) | applied | New corollary `ibt:orthogonal-holonomy` with proof (transport preserves H_X^{-1}, det = 1 on trivialized loops, skew curvature, metric message connection case); GL(K) consequence stated; augmented-metric selector noted as not subject to it. c16 sentence strengthened. AUTHORING:89 is C-SUM's. |
| D21-BT-02, GAP10-S3-02 (S3-M13) | applied | ch16 after `ibt:alignment`: isotropy obstruction (diag(t,1) example), non-properness obstruction with the SL(2) pair example (reconstructed: det z_t = 1, conjugation limits), existence on proper strata (Palais1961), GL(K) link-only never identifiable; :286 qualified; c16. Section 8 item 14 decided on evidence: both lenses are consistent; no canonical metric in general (D21-BT-02), a canonical Kempf-Ness metric exists on a smaller locus (GAP10-S3-03, deferred as extension). |
| D21-BT-03 (S3-X18) | applied | New `ibt:grassmann` with proof: principal angles, positivity of N wherever C is invertible, T isometric, direct rotation, equals transport along the Grassmannian geodesic (explicit principal-vector construction); remark on base-path images and second-order symmetric products. c16. |
| D21-BT-04 (S3-X19) | applied (a) | Pointwise one-jet obstruction after `ibt:invisible`, with proof; c16. Parts (b) to (d) deferred (open:conn-derived is B-FOUND's). |
| D21-BT-05 (S3-L18) | applied (in my files) | ch16 and c16: labels, axes, column selection and metric declared; law enters only through its mean; "declared bi-invariant metric" in the SO(3) example. Abstract, outlook, README: FOR OTHER OWNERS. |
| D21-BT-06 (S3-L19) | applied | (i) straddling-step sentence, convergence for every partition sequence; (ii) h valued in G in `ibt:group`; (iii) M and sector weight listed as comparison data, plus the selector-dependence sentence; (iv) "short" deleted; (v) product metric at the SO(3) example. |
| D21-BT-07 (S3-L20) | applied (example scope) | SO(3) example: connection and curvature in span(e_3), holonomy Abelian; the general two-link claim not stated (proof only sketched by the lens). c16 likewise. |
| D21-BT-08 (S3-L21) | applied | Scalar backtrack example replaced by the diag(e^x, e^y) example (reconstructed: Omega = I, raw column gives the witness pulled back with Jacobian e^{x+y}, F(0) = (1/9)[[-1,2],[-2,1]]; reciprocal column constant, a = 0); rank-one pure-gauge statement kept. c16. |
| D21-BT-09 (S3-L22) | applied | ch16 after `ibt:alignment`: identification with `ctx:f:ass:branch` and `geo:collective`, convention h -> h^{-1}, pointer to the ch15 cusp; c16. |
| D21-BT-10 (S3-L23) | applied in part | Reference links K_ij, K_e -> \bar\Theta_{ij}, \bar\Theta_e; locked form \mathcal K -> \mathfrak k; C_r -> \mathsf C_r; ibt:invisible element Z -> \zeta; witness scalar s -> \chi; message coordinate change s(c) -> \Lambda(c); proof bound M -> L and solution S -> \mathcal S; subscripts explained and `prop:int-reciprocal` cited. The sending map T in L_ij = O_iT_j, the endpoint map T, and H_i are unchanged (global notation). Notation appendix: FOR OTHER OWNERS. |
| D21-BT-12 (S3-L25) | applied | `ibt:alignment` statement rephrased. Companion figure: B-FOUND supplied `companion/figures/cross_context_map.tex` (a bare tikzpicture with no label or caption). It is now `\input` in c16 after `comp:ibt:cross`, inside a figure environment with a caption and label `fig:comp-cross-context`, and the text refers to it. The figure was not rendered (rule 5). Its node text uses \operatorname and the sector superscript $\Theta^a_{ij}$, which match c02's notation. It shows the lift/decode map and the moving row spaces, but not principal angles. |
| D21-BT-13 (S3-X39) | applied | Shape-map factorization of curvature for both selectors (ch16 after the corollary and after `ibt:alignment`), c16. |
| GAP10-S3-03 (S3-X20) | deferred | Kempf-Ness metric theorem is an extension; listed in Section 8 decision above. |
| GAP10-S3-04 | deferred | SU(2) monodromy at isolated isotropy jumps belongs to `open:conn-derived` (B-FOUND's file). |
| GAP10-S3-05 (S3-X40) | applied | 16:160 remark turned into `ibt:global` with proof (global connection, projected-trivial form, C on all pairs, classifying map, r = K trivial) and the TS^2 example; the converse (range equals metric connections) stated as not settled. The r <= K + n count is not stated. c16. |
| GAP10-S3-06 (S3-X41) | applied | SL(K), M = H_X^{-1}: traceless part, same curvature (reconstructed: identity orthogonal to sl(K) for Tr(H^{-1} xi H eta^T)). c16. |
| GAP10-S3-07 | not in scope | chapter 14 (B-PA). |
| D03-GEOM-08 | deferred | Energy-level refinement step is an extension located mainly in 02_geometry. |
| D02-LINKS-06(d) | not applied | ch16 already uses N_{ij,j}; the fix is in ch02 (B-FOUND). |

## BIB ENTRIES

```bibtex
@book{Lee2018,
  author = {Lee, John M.},
  title = {Introduction to Riemannian Manifolds},
  edition = {2},
  series = {Graduate Texts in Mathematics},
  volume = {176},
  publisher = {Springer},
  address = {Cham},
  year = {2018},
  doi = {10.1007/978-3-319-91755-9}
}
@article{Kendall1984,
  author = {Kendall, David G.},
  title = {Shape Manifolds, {Procrustean} Metrics, and Complex Projective Spaces},
  journal = {Bulletin of the London Mathematical Society},
  volume = {16},
  number = {2},
  pages = {81--121},
  year = {1984},
  doi = {10.1112/blms/16.2.81}
}
@article{Schonemann1966,
  author = {Sch{\"o}nemann, Peter H.},
  title = {A Generalized Solution of the Orthogonal {Procrustes} Problem},
  journal = {Psychometrika},
  volume = {31},
  number = {1},
  pages = {1--10},
  year = {1966},
  doi = {10.1007/BF02289451}
}
@article{PineleStrapassonCosta2020,
  author = {Pinele, Julianna and Strapasson, Jo{\~a}o E. and Costa, Sueli I. R.},
  title = {The {Fisher--Rao} Distance between Multivariate Normal Distributions: Special Cases, Bounds and Applications},
  journal = {Entropy},
  volume = {22},
  number = {4},
  pages = {404},
  year = {2020},
  doi = {10.3390/e22040404}
}
@article{BandoKugoYamawaki1988,
  author = {Bando, Masako and Kugo, Taichiro and Yamawaki, Koichi},
  title = {Nonlinear Realization and Hidden Local Symmetries},
  journal = {Physics Reports},
  volume = {164},
  number = {4--5},
  pages = {217--314},
  year = {1988},
  doi = {10.1016/0370-1573(88)90019-1}
}
@book{Taylor2011,
  author = {Taylor, Michael E.},
  title = {Partial Differential Equations {III}: Nonlinear Equations},
  edition = {2},
  series = {Applied Mathematical Sciences},
  volume = {117},
  publisher = {Springer},
  address = {New York},
  year = {2011},
  doi = {10.1007/978-1-4419-7049-7}
}
@article{NarasimhanRamanan1961,
  author = {Narasimhan, M. S. and Ramanan, S.},
  title = {Existence of Universal Connections},
  journal = {American Journal of Mathematics},
  volume = {83},
  number = {3},
  pages = {563--572},
  year = {1961},
  doi = {10.2307/2372896}
}
@article{NarasimhanRamanan1963,
  author = {Narasimhan, M. S. and Ramanan, S.},
  title = {Existence of Universal Connections {II}},
  journal = {American Journal of Mathematics},
  volume = {85},
  number = {2},
  pages = {223--231},
  year = {1963},
  doi = {10.2307/2373211}
}
@article{BjorckGolub1973,
  author = {Bj{\"o}rck, {\AA}ke and Golub, Gene H.},
  title = {Numerical Methods for Computing Angles between Linear Subspaces},
  journal = {Mathematics of Computation},
  volume = {27},
  number = {123},
  pages = {579--594},
  year = {1973},
  doi = {10.1090/S0025-5718-1973-0348991-3}
}
@article{DavisKahan1970,
  author = {Davis, Chandler and Kahan, W. M.},
  title = {The Rotation of Eigenvectors by a Perturbation. {III}},
  journal = {SIAM Journal on Numerical Analysis},
  volume = {7},
  number = {1},
  pages = {1--46},
  year = {1970},
  doi = {10.1137/0707001}
}
@article{Palais1961,
  author = {Palais, Richard S.},
  title = {On the Existence of Slices for Actions of Non-Compact {Lie} Groups},
  journal = {Annals of Mathematics},
  volume = {73},
  number = {2},
  pages = {295--323},
  year = {1961},
  doi = {10.2307/1970335}
}
```

Crossref confirmed title, authors, journal, volume, issue, first page and year for every DOI above (Crossref gives first pages 563, 223 and 295 only for the two Narasimhan-Ramanan papers and Palais; the last pages are from the review lenses' Crossref checks, XC1-GAUGE-10 and XC5-LIT-07, and are not re-verified here; the Lee 2018 volume number 176 is from memory, Crossref returned the series but not the volume). Existing keys reused: Wilson1974, DAddaLuscherDiVecchia1978, Lee2013, Smale1970, ONeill1966, LittlejohnReinsch1997, LeokMarsdenWeinstein2005, MassambaThompson2003, Higham2008, Amari1998, EellsSampson1964, CasselEtAl2024.

## NEW OR CHANGED LABELS

Renamed (used only inside chapter 15; grep of all three documents found no other reference): `ctx:f:statement:1` -> `ctx:f:transport-equation`; `ctx:f:statement:3` -> `ctx:f:elimination`; `ctx:c:statement:2` -> `ctx:c:reduced-energy`; `ctx:c:statement:5` -> `ctx:c:energy-identity`.

New in ch15: `ctx:f:horizontal`, `ctx:f:eq:horizontal`, `ctx:f:eq:regular-set`, `ctx:f:global-branch`, `ctx:f:eq:orbit-distance`, `ctx:f:divergence-cost`. New in ch16: `ibt:global`, `ibt:eq:global-nabla`, `ibt:grassmann`, `ibt:orthogonal-holonomy`. New in c16: `fig:comp-cross-context`. All unique across the three documents (grep).

## FOR OTHER OWNERS

- C-SUM, `frontmatter.tex:15`: the chapter 15 paragraph should name the reciprocal potential on declared finite-dimensional regular families (XC3-SCOPE-04) and say that the reduced cost is the minimized cost only while the branch is globally minimizing (D19-CTXF-01), and that for GL(K) frames with K >= 3 consensus is excluded (D19-CTXF-02). `frontmatter.tex:17`: "derives base transport" -> constructs it once labels, columns, weights and a metric are declared; "built only from reciprocal rotation links" -> "from reciprocal rotation links and a declared bi-invariant metric", with Abelian holonomy (D21-BT-05, D21-BT-07); the rectangular selector has orthogonal holonomy (D21-BT-01).
- C-SUM, `chapters/06_outlook.tex:16`, `README.md:24`, `README.md:26`: same wording as above.
- C-SUM, `AUTHORING.md:87`: threshold now s > d/2 + 1 (shape in H^s, frame and rows in H^(s-1)); add that no locally gauge-invariant eliminated energy diffuses frames. `AUTHORING.md:89`: replace "it is a GL connection unless a shared Lie-algebra constraint is imposed" by "a connection on the GL(K) frame bundle whose holonomy is orthogonal for the induced metric H_X^{-1}"; add that link-only alignment needs an invariant target metric, which conjugation targets of GL(K) never admit.
- C-SUM, `references.bib`: add the eleven BIB ENTRIES; update `CasselEtAl2024` to the journal version (J. Math. Imaging Vision 67(6), 2025, doi 10.1007/s10851-025-01270-w, Crossref-checked; article number not returned by Crossref), keeping or renaming the key; `Narasimhan1962` is now cited nowhere in the four files (grep of all .tex files finds no remaining citation).
- C-SUM, `appendices/notation.tex`: chapter 16 symbols (X, H_X, Q_X, P_X, W, C, N, T, \bar\Theta_{ij}, R_{ij}, \mathcal H, J_Z, \mathfrak k) and the chapter 15 sets R, R_glob (D21-BT-10, D01-FOUND coverage).
- C-SUM, `future_TODO.md:78`: record the frame-diffusion no-go and the admissible row Dirichlet term (D20-CTXC-02).
- B-FOUND, `chapters/02_geometry.tex:7`: "adds the minimized contextual energy" holds only on the globally minimizing part of the branch set (D19-CTXF-01). `open:geo-gauged-flow` (02_geometry:264): add the frame-diffusion no-go (D20-CTXC-02) and the uniform-in-lambda question (D20-CTXC-05).
- B-FOUND, `appendices/base_connection.tex:71-74` (`open:conn-derived`): continuation content of D21-BT-04 (b)-(d) and the Z_2 winding obstruction of GAP10-S3-04.
- B-FOUND, `chapters/02_interactions.tex:192-210`: D02-LINKS-06 (write N_{ij,j}); chapter 16 now cites `prop:int-reciprocal`.

## AUTHOR DECISIONS NEEDED

None specific to these files. The default comparison metric for noncompact link targets (Kempf-Ness dressing, GAP10-S3-03) is a constitutive choice left open in the text.
