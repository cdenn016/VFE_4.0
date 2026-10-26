# Items 1–3 manuscript integration, 2026-10-26

The supported communication, dictionary-learning, and supplied-hierarchy results are integrated into the main MAgent Theory 2.0 manuscript and its companion. This implements the existing October 23 audit corrections. It does not add a new investigation or a new formation claim. The user subsequently authorized publication and canonical synchronization after the integration checks below.

The main manuscript now contains Chapter 18, “Communication Codes and Learned Alignment,” Chapter 19, “Learning Directed Links under Three Objectives,” and Chapter 20, “Curvature Costs and Supplied Hierarchies.” Their sources are `chapters/17_communication_links.tex`, `chapters/18_link_dynamics.tex`, and `chapters/19_curvature_hierarchy.tex`. The companion adds corresponding Chapters 17–19 in `companion/c17_communication_links.tex`, `companion/c18_link_dynamics.tex`, and `companion/c19_curvature_hierarchy.tex`. Existing chapters retain their order. The original Item 1–3 working documents receive the same corrections and refreshed PDFs.

## Existing findings applied

The requirements are recorded in `docs/reviews/2026-10-23-theory-next-task/gauge-audit.md` and `variational-audit.md`. The following table maps their identifiers to the implemented repairs; it records implementation rather than a new adjudication.

| Finding | Repair and integrated location |
| :--- | :--- |
| G1 | The induced-action theorem requires the declared directed loops to span the underlying cycle space; strong connectivity is a sufficient condition. Independent opposite edges remain distinct from geometric inverse traversals. The three-edge acyclic counterexample retains its residual of one third. Main §20.2; companion §19.2. |
| G2 | The flat-wall statement requires internal directed paths joining every relevant opposite bridge pair and nonempty bridge sets in both directions. Main §20.4; companion §19.6. |
| G3 | The general-law rigid restriction declares a common reference measure, finite positive self-product normalizers, finite positive constant internal record factors, and a finite positive remaining bridge normalizer. It is distinguished from conditioning on a probability-zero diagonal and from a soft-to-rigid limit. Main §20.4; companion §19.5. |
| G4 | Law-comparison KL, the covariance-dressed link defect, grounded-code identifiability, and single-topic identifiability are kept separate. The quarter-turn example has zero transported-law KL but dressed defect two. Grounded edge records can identify dictionaries without relaying one sample around a whole loop. Main §§2.5, 18.5, 20.2; companion §§2.3, 17.5, 19.3. |
| G5 | Arbitrary stacked readings are compatible precisely when their row-relation spaces agree. The dimension-bound equality cases distinguish total message width at most the content dimension from saturation above it. Main §18.4; companion §17.4. |
| G6 | The bandwidth limit is an integrable expected log-likelihood ratio. Its difference-of-KL representation requires finite relative entropies. Main §18.6; companion §17.6. |
| V3 | Equality of dictionary derivatives is restricted to the plug-in comparison. Law-level record and loop widening remain explicit, and equality of derivative rows is separated from equality of Fisher-preconditioned velocities. Main §19.3; companion §18.3. |
| V4 | The weak-anchor ring limit requires nonzero operating content. At zero content the coupling remains constant. Main §19.3; companion §18.4. |

The integration also distinguishes unrestricted group-valued dictionaries from the image of a single real exponential. It preserves the complete affine evidence action at nonzero records, so a curvature Hessian is not promoted to a record-independent preference for flatness. Equal-gain cancellation concerns the linearized loop sum. The top-down precision cap requires nonzero parent operating content, and the stationary rank-one calculation retains its fixed-content Gaussian tangent hypotheses.

## Scope and editorial integration

General-law communication kernels, conditional reconstruction, identifiability, consensus, and normalized rigid restrictions are stated before their Gaussian computations. The tangent learning model retains its declared anchor, loop prior, and grounded observation design. Global normalized inference, per-agent inference, and the reciprocal potential remain separate objectives. The reciprocal covariance calculation concerns a source-role force with specified quantities held fixed; it does not settle the coupled dynamics of contents, covariances, and source rows.

The front matter, reading order, outlook, README, authoring scope, and Item 1–3 backlog entries are updated. The two master documents include the new chapters and the existing Item 1 and Item 2 bibliography databases. No bibliography entries are changed. The main and companion additions omit numerical experiment reports and development workflow text. The original working documents remain available as corrected supporting material.

The supplied partition and rigid hierarchy calculations establish conditional static reductions and the stated Gaussian minimizer limit. They do not establish selection, persistence, complete operational inheritance, or recursive formation. Nonlinear dictionary learning, general-law dynamical extensions, and the coupled covariance/source-row problem remain open. Item 7, its backlog section, and all unrelated WIP are protected against the pre-edit baseline.

## Validation

All five affected TeX documents were compiled sequentially with TeX Live 2025 and `latexmk`. The main manuscript has 284 pages and the companion 113 pages; their new chapters start at PDF pages 225, 243, 255 and 90, 98, 104, respectively. The corrected working PDFs have 29, 18, and 15 pages. References and citations resolve in all five builds. The six new chapter sources passed the focused citation, forbidden-spacing, control-character, and prose checks performed during integration.

The main build retains three preexisting overfull boxes, at 9.51393, 20.93712, and 9.39738 points. It adds no overfull box. One underfull line in the regular exponential-family paragraph appears in both the main manuscript and the Item 2 working document; both renderings were visually inspected and retained without clipping or overlap. The companion and three working-document final logs contain no LaTeX warnings or overfull boxes. These are compilation and layout checks, not new mathematical experiments.

The PDF inspection checked every text span in all five PDFs against its physical page bounds and found none outside. All pages of the new main and companion chapters, their following material, contents pages, and selected changed earlier passages were rendered: 74 main pages and 35 companion pages. All ten contact sheets were visually inspected, with full-page checks of main pages 237 and 253 and companion pages 92 and 107. The three working PDFs received additional first-page and middle-page visual samples, plus the Item 2 underfull paragraph. No visible layout repair was needed. `pdf-validation.json` records the main and companion PDFs and page coverage; `integration-validation.json` records build summaries, changed paths, and preservation results against `baseline.json`.

The final preservation comparison found all 147 protected WIP files unchanged, along with all 13 Item 7 files and the exact Item 7 backlog section. Exactly the planned 17 existing manuscript files changed, and exactly six chapter sources were added. Both repository whitespace checks passed and both Git indexes remained empty. The PDF contact-sheet helper initially failed on a pixmap size attribute after the preservation checks had passed; correcting that helper produced the final successful receipt and samples. No manuscript edit was needed for that rendering issue.

No training, GPU job, numerical research experiment, or Lean work was run. No commit, push, merge, or canonical synchronization was performed in this integration step.
