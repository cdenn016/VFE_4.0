# Item 5: manuscript consistency and mathematical scope

Reviewed October 23, 2026. Scope: the complete standalone `docs/derivations/2026-10-23-link-base-transport/latex/main.tex` and its README against the live `Research/manuscripts/MAgent_Theory_2.0` authoring rules, foundation and interaction conventions, collective geometry, moving-presentation appendix, contextual-coupling chapter, corresponding companion chapters, and Item 5 backlog. Item 7 was excluded. This review used source inspection and analytical comparison; it ran no numerical scripts, GPU work, or Lean. No manuscript file was edited.

No variational-inference findings. I found no substantive inconsistency in the requested mathematical scope. The standalone result can be integrated as a conditional constitutive construction. It does not establish that the existing agent dynamics selects this comparison rule, preserves its identifiable region, or inherits its link sector under a parent map.

## Connections that must remain distinct

The law Fisher selector in `chapters/02_geometry.tex:192–227` minimizes the norm of `d p + J_p A` in the declared collective statistical metric. The metric follows actual normalized laws; a marginal sum represents the joint metric only under the stated product restriction (`02_geometry.tex:38–62`). Constant comparison weights and comparison maps are constitutive inputs (`02_geometry.tex:203`). Nonzero infinitesimal isotropy leaves an affine class of connections rather than a unique selector (`02_geometry.tex:230`).

The message selector in the standalone note uses a persistent labeled message space and its declared metric (`main.tex:111–124`). Its formula is

\[
a_X=-dX X^T(XX^T)^{-1}.
\]

It is a connection derived from a full-rank vector dictionary. A dictionary of means can discard all changes of shape or dependence that preserve those means. Its metric `H_X^{-1}` is a fiber metric, not the Fisher metric of the original agent law. The rectangular construction also satisfies metric compatibility and zero trace curvature (`main.tex:228–245`); integrating it as an arbitrary general-linear connection would lose a proved restriction.

The enlarged-target selector uses a declared Riemannian metric on laws, interfaces, and links (`main.tex:310–336`). Its locked equation has the same mathematical form as the collective normal equation, but its target and kinetic metric may differ. The note explicitly states, “An interface or link norm is not automatically the Fisher metric of a normalized statistical experiment” (`main.tex:336`). A Fisher interpretation would require a separately specified normalized experiment and verification of the induced score metric. No such interpretation is needed for the stated geometric construction.

Required integration wording:

> The message and link selectors use their declared comparison metrics. They coincide with the collective Fisher selector only when the targets, infinitesimal actions, metrics, and compared data are identified. A shared affine transformation law alone does not establish that equality. Their use as connections in a statistical pullback does not identify their defining comparison metric with the Fisher metric of that law.

## Frames, reference consistency, and the principal group

The foundation defines a frame as a point of one principal bundle, with one group element inducing the belief and model representations (`chapters/01_foundations.tex:8–16,76–85`). It distinguishes passive re-expression from active change at fixed frames. The interaction chapter makes common-reference consistency substantive, whereas per-agent dressing covariance holds for every reference rule (`chapters/02_interactions.tex:55–72,120–136`).

The standalone transformations agree with these conventions (`main.tex:30–59`). In particular, `U_i -> U_i h` requires the affine change `a -> h^{-1} a h + h^{-1} d h`. The square-frame construction correctly identifies the error in resetting the reference connection to zero after every context-dependent reference change (`main.tex:65–79`). The declared marked matrix `Q` transforms as `h^{-1}Q`; it is additional covariant model data, not the passive presentation matrix `U_i` itself.

The phrase “physical frame vectors” at `main.tex:66` should be integrated as “declared marked frame vectors” unless an observable algebra that reads the marks is specified. This is an authoring clarification, not a defect in the flat-connection identity. The live foundation reserves physical distinguishability for an explicit agent or external response algebra (`01_foundations.tex:87`).

The unrestricted rectangular formula initially produces a connection on the general-linear frame bundle. The note explicitly states that it need not lie in an arbitrarily prescribed smaller Lie algebra (`main.tex:279`), then supplies constrained least squares for that algebra (`main.tex:281–297`). Its joint-sector extension minimizes over one common Lie-algebra variable, with combined infinitesimal identifiability (`main.tex:300–305`). This matches `AUTHORING.md:19`, `01_foundations.tex:10–16,70,83`, and `02_interactions.tex:22`. The integrated statement must retain that shared variable; choosing two unrelated matrix connections independently would not establish one principal connection.

The existing theory admits separate belief and model link fields, with the tied case an explicit additional condition (`02_interactions.tex:95–118`). The new base selector does not force those link fields to be tied. Both may enter the comparison target under the same principal action. For a one-sector matrix derivation, say that the sector is temporarily fixed and then state how the principal selector acts through both representations.

Required integration wording:

> One principal group acts on both sectors. The construction may compare their separate link fields, but its principal connection is selected by one common Lie-algebra variable. Its sector connections are the images of that variable under the two differentiated representations. Equality of the two interaction link fields remains a separate hypothesis.

## Directed links and contextual transport

The main interaction chapter separates independently attending directed edges from geometric reversal (`chapters/02_interactions.tex:188–210`). Its principal-root factorization is retained exactly in the standalone note (`main.tex:88–106`), including the spectral-domain condition. The note retains raw links or both reciprocal and backtrack factors. It never assumes that the original directed links are inverses.

The cross-context matrix `N(c',c)=C(c,c')C(c',c)` is a different object from the same-context agent backtrack. The note makes that distinction at `main.tex:206`. Reciprocalization changes the finite displacement map without removing the backtrack fields used to form the message dictionary. The scalar example at `main.tex:270–274` shows that backtrack variation can change the resulting connection even when the reciprocal agent link remains fixed. Conversely, the selected connection need not encode every original link distinction; neither exact reconstruction of the directed pair from `(Omega,S)` nor first-jet construction of a connection proves that stronger retention claim.

The note also distinguishes finite displacement maps from exact path transport (`main.tex:184–223`). The endpoint rule has identity at coincidence and the required first jet, while path-ordered products converge to the derived ODE transport. An unrefined endpoint map is not asserted to satisfy the path composition law. This resolves the specific construction request in `future_TODO.md:80–82` on the stated regular regions.

The curved `SO(3)` star at `main.tex:338–361` establishes contextual curvature from an identifiable tuple of reciprocal link fields. Its interaction graph has no reciprocal cycle, so this is not a claim that contextual curvature equals graph-cycle holonomy. That distinction agrees with `companion/c06_geometry.tex:5` and `companion/c02_links.tex:63`.

## No automatic transfer of the contextual Fisher dynamics

The contextual-coupling chapter fixes its statistical comparisons (`chapters/15_contextual_coupling.tex:24–48,409–416`) and minimizes its own law-distance or law-Fisher contextual energy. Its envelope calculation uses stationarity of that same cost (`15_contextual_coupling.tex:136–153,499–515,598–614`). The row equations lack a direct contextual derivative because that particular contextual term has no row dependence (`15_contextual_coupling.tex:205–218,582–594`).

By contrast, the new dictionary permits weights depending on source rows and laws (`main.tex:124`), and an enlarged target can include interfaces or link fields. Substituting this selector into a different contextual energy does not make its derivative an envelope derivative of that energy. The note already records this restriction (`main.tex:388`). Retain it in the integrated chapter and companion. When the selector is made dynamical, its full constitutive dependence must be differentiated; row, link, or interface terms can then appear. Item 4's existing force, dissipation, Sobolev, and continuation theorems apply to their stated model only.

Required integration wording:

> The construction here selects transport from contextual agent data. It does not change the law Fisher energy or the evolution equations of the contextual-coupling chapter. If this new selector is substituted into an energy, every dependence on laws, rows, interfaces, and links must enter the variation. The envelope simplification applies only when the selector is stationary for that same energy. Well-posedness, regularity preservation, and inheritance then require analysis for the resulting system.

The note therefore leaves the three existing objectives intact: normalized global inference, per-agent inference, and the reciprocal law potential (`AUTHORING.md:31–37`). It proves neither an ELBO representation for a comparison metric nor a formation or inherited-dynamics theorem. Its base, topology, dimension, labels, metric, and regular identifiable region are supplied (`main.tex:111,133–134,310,376,381–390`). These are declared hypotheses rather than missing consequences of the construction.

## Exact manuscript passages to update after integration

The following are integration edits, not defects in the presently scoped manuscript.

| Live passage | Present wording or scope | Required change |
| --- | --- | --- |
| `chapters/01_foundations.tex:47` | “the construction of a connection from agent interactions remains open there” | Point to the new conditional construction, then retain the unresolved selection and regularity obligations. |
| `chapters/02_geometry.tex:105` | The moving-presentation appendix “does not derive a connection from the agents' laws or links” | If the appendix is expanded, distinguish its supplied-form calculation from its new constitutive selectors; if a new chapter is added, cite that chapter separately. |
| `appendices/base_connection.tex:5` | “nothing here derives” the form, and the construction is an open problem | Keep this statement only for the supplied-form subsection and add a pointer to the conditional derivation. |
| `appendices/base_connection.tex:71–72` | Open problem requests identity, first derivative, and endpoint laws, then says the family remains unconstructed | Replace the construction gap with the remaining choice by dynamics, regularity preservation, singular-region continuation, and link-sector inheritance problem. |
| `chapters/15_contextual_coupling.tex:383,961` | A separate derivation from inter-agent links is still required; this chapter proves no such derivation | State that this Fisher-flow construction does not derive it, and point to the separate message/link selector. Preserve every other scope restriction. |
| `companion/c06_geometry.tex:9` | “Nothing below derives” the supplied form | Scope this sentence to the supplied-form calculation, then explain the new normalized cross-context comparison in a self-contained section or chapter. |
| `companion/c02_links.tex:63` | A finite nontrivial communication loop establishes no base connection by itself | Keep this claim. Add that a smooth contextual field plus the declared comparison rule can supply one through the new construction. |
| `companion/c15_contextual_coupling.tex:241` | Deriving displacement transport from communication links is “a further construction” | Point to the new construction while retaining that it defines a different selector unless the targets and metrics are identified. |
| `future_TODO.md:80–82` | Item 5 requests the displacement family and retention of backtrack data | Mark conditional existence as supplied; retain the operative-rule, rank/identifiability preservation, and dynamics/inheritance obligations. Do not mark generic dynamical selection complete. |

Suggested replacement for the current open problem:

> Smooth full-rank message dictionaries and identifiable tuples of contextual link data provide constitutive displacement families with identity at zero displacement, compatible endpoint laws, and first derivatives defining connections. The unresolved modeling problem is to identify which retained agent data and comparison metric determine the operative rule, and to show that the intended dynamics supplies and preserves its regular region. Singular continuation and inheritance of the link sector require separate results. Pointwise link covariance alone selects neither the rule nor the contextual base.

The companion should include the normalization `C(c',c)=X(c')Q_X(c)`, the identity `XQ_X=I`, and the expansion yielding `a_X=-dXQ_X`, together with the meaning of persistent message labels. It should explain the backtrack distinction and the shared-group constraint in its own text. The main text or appendix must contain the decisive proofs rather than linking to the standalone note, as required by `AUTHORING.md:9–13`.
