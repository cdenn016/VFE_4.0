"""Exact-match edits of future_TODO.md (CRLF preserved)."""
import sys

p = sys.argv[1]
t = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")

old94_start = "Working document, first part (2026-10-23): `Item_7_formation_feedback.tex`"
old96_end = "Checks in `item7_checks/`."
i = t.index(old94_start)
j = t.index(old96_end, i) + len(old96_end)
new_item7 = (
    "Working document `Item_7_formation_feedback.tex` (2026-10-23, revised 2026-10-29), formation with top-down feedback, taking up Item 3's open problem `it3:open-formation`. "
    "Feedback through an added attention source changes no row exponent; in gradient objectives an invariant top-down cost exerts no force on block momentum. "
    "The earlier convex no-go is retracted and restricted: a globally attracting post-handoff equilibrium excludes (F6b) only when persistence means that the candidate relation holds at all later times, the equilibrium lies off the candidate boundary, and no competing block is selected first (`it7:attractor`); "
    "for the document's finite-dwell selector a strictly convex Gaussian hierarchy can meet (F6b) (`it7:dwell`), and in the Items 2/3 global joint, run in its own metric, such instances were found numerically only with declared parent anchors that pull the parents together. "
    "The working mechanism is a shared source label at the handoff: the block's row is the Gibbs row of its pooled law at temperature T/n (general laws), and in Gaussian families of the sufficient statistic the pooled parent is exactly inherited between handoffs at a fixed partition, with transient, record-slaved fibers; universal pooled closure characterizes those families (Bernoulli fails).\n\n"
    "Status against the repaired standard (2026-10-29): the twelve-agent instance is not shown to be a recursive formation, and its classification is undetermined. "
    "Proved: fixed-partition inheritance, both null histories on the whole admitted class (the label-preserving null never selects, the renormalized null selects only the block of all twelve agents), and the certified roots and saddle-node of the gap equation. "
    "Numerical: the first-stage history of the shared-label flow (fixed-step integration with margins). "
    "Open: (F2) and (F5) over a declared constant set with nonempty interior, (F3) for the hybrid process with candidate clocks and merges, second-stage (F4), and (F6b) against a declared counterfactual class; widened initial states examined so far give uniting handoffs. "
    "The frozen-environment control selects the four groups and no union. "
    "The derived selector (`it7:ch-derived`) replaces the thresholds by the structure free energy of coarsenings; it detects attention, not agreement, and with tau_p = 1.25 the label-preserving null and the frozen control select the four groups only, the renormalized null the two sides and then all agents, and the shared-label flow the groups and then their unions, so no window gives interaction necessity at both stages against both nulls. "
    "Open obligations are recorded in the document as `it7:open-hybrid`, `it7:open-counterfactual`, `it7:open-convex` and `it7:open-validated`; the reciprocal-potential version (frozen-scalar source widening, not a certainty loss) and depth remain open as well. Checks in `item7_checks/`."
)
t = t[:i] + new_item7 + t[j:]

E = [
    ("Codes are agent-specific; a shared language is part of a species' shared generative model (`def:shared-model`).",
     "Codes are agent-specific and are not components of the generative model; a shared language is a separately declared hypothesis (`def:shared-model` defines sharing of a generative model up to frame and contains no code)."),
    ("The local critical-mode result (2026-10-28) is integrated: a subcritical pitchfork with nearby asymmetric saddles for T above the threshold, nonlinear instability at the threshold, and full/reduced Fisher center denominators D+4 and D.",
     "The local critical-mode result (2026-10-28) is integrated: for the declared reference data, a subcritical pitchfork with nearby asymmetric saddles for T above the threshold, nonlinear instability at the threshold, and full/reduced Fisher center denominators D+4 and D; the sign of the quartic coefficient depends on the data (a float64 variant is supercritical). The global-existence theorem also gives convergence of each trajectory to one stationary point."),
    ("This Item 4 result does not settle Items 5, 6, or 7; the separate conditional Item 5 construction is recorded below.",
     "This Item 4 result does not settle Items 5, 6, or 7; the separate conditional Item 5 construction is recorded below. Recorded 2026-10-29: after algebraic elimination no locally gauge-invariant contextual energy diffuses frames, because its field-metric gradient is horizontal, so frame smoothing needs a term that is not eliminated; a row Dirichlet term is an admissible gauge-invariant constitutive choice that the chapter does not make. The reduced cost is the minimized cost only on the globally minimizing part of the regular branch, and for general linear frames of dimension at least three the branch hypothesis fails at every consensus context."),
    ("Report every result against the formation standard, including the null-flow control.",
     "Report every result against the formation standard, including both nulls and the frozen-environment control of `inh:null-flow` and the counterfactual class of `inh:counterfactual-class`."),
    ("The other outstanding author decisions are listed in the remediation summary in the VFE_4.0 review folder.",
     "The other outstanding author decisions are listed in the remediation records of the VFE_4.0 review folders (2026-10-22 and 2026-10-29)."),
]
for old, new in E:
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(t.replace("\n", "\r\n"))
print("ok")
